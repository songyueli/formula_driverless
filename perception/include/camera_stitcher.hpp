#pragma once

#include <opencv2/core.hpp>
#include <vector>

#ifdef PERCEPTION_USE_VPI
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/WarpMap.h>
#include <vpi/algo/Remap.h>
#endif

// Stitches N cameras that share a single center of projection (identical
// mount position, differing only in orientation) into one cylindrical
// panorama -- azimuth maps LINEARLY to output column, giving UNIFORM
// angular resolution everywhere. A planar (rectilinear) projection was
// tried first for this (keeps the ground plane's horizon perfectly
// straight), but its resolution grows with tan(theta) toward the edges,
// which stretches round 3D objects (cones) into an asymmetric, visibly
// warped silhouette the further they sit from center -- confirmed directly
// against this panorama's own cones near the seams. Ground-plane lines bow
// slightly under cylindrical instead of staying straight, but that's
// cosmetic only -- deriving a cone's ground position via ray-casting is
// exactly as valid here as under a planar projection, given the correct
// (cylindrical) pixel -> ray formula; "visually straight" was never a real
// requirement for that math to work.
//
// The reprojection itself is exact -- not a feature-matching approximation
// -- because a shared optical center means every camera pair is related by
// a homography at every scene depth, and here the relative rotation between
// cameras is known exactly (it's baked into the SDF, not estimated).
//
// All the per-pixel geometry is computed once in the constructor. In overlap
// regions between two cameras, one camera is picked as the sole source for
// each pixel (whichever is closer to its own optical center there) rather
// than blending -- averaging two INDEPENDENTLY rendered/anti-aliased views
// of the same thin object (even when the underlying geometry lines up
// exactly) shows up as a visible soft double-image, worst on small objects
// since they're almost entirely edge/AA pixels. A hard seam only ever shows
// one camera's own rendering of any given pixel, so that can't happen.
//
// Stitch() is N remaps + a masked copy -- NOT actually cheap at this
// panorama's resolution, contrary to what this comment used to claim:
// confirmed directly via /timing/perception/stitch that plain CPU
// cv::remap (PERCEPTION_USE_VPI undefined -- see CMakeLists.txt) costs
// ~36-38ms/frame, the dominant cost in perception's entire ~65ms cycle
// (more than 2x YOLO/TensorRT inference itself), while tegrastats showed
// one CPU core pinned at 100% and the GPU (GR3D_FREQ) mostly idle during
// that same window -- exactly the profile of CPU-bound work sitting next
// to genuinely idle compute. When PERCEPTION_USE_VPI is defined (Jetson
// only -- NVIDIA VPI is a JetPack-provided library, not something the x86
// dev machine has), each camera's remap instead runs through VPI's Remap
// algorithm on the VPI_BACKEND_CUDA backend: the exact same dense
// per-pixel mapping this class already computes gets baked into a
// VPIWarpMap once in the constructor (see m_vpiSources), same interpolation
// (linear) and border behavior (zero-fill for pixels outside a camera's
// own frame) as the CPU path, so output should be numerically equivalent
// modulo ordinary interpolation-implementation differences -- just moved
// off the CPU entirely onto the otherwise-idle GPU, which also leaves the
// GPU free to run YOLO/TensorRT concurrently rather than contending with
// it, unlike a hypothetical CPU-thread-parallelization fix.
class CameraStitcher
{
public:
    struct CameraConfig
    {
        // Yaw of this camera relative to the reference camera (index 0),
        // radians, positive = rotated toward the reference camera's left.
        // Pitch/roll are assumed identical across all cameras (true for this
        // rig -- see model.sdf) so a single yaw angle fully describes the
        // relative rotation.
        double yawRad = 0.0;
    };

    // camWidth/camHeight/camHFovRad: shared intrinsics for every source
    // camera (must match model.sdf's <image> and <horizontal_fov>).
    // cameras: relative orientation of each source camera, same order every
    // frame will be passed to Stitch().
    // outWidth/outHeight/outHFovRad: panorama size and horizontal field of
    // view. outHFovRad should be a little less than the cameras' true
    // combined coverage so every panorama pixel has at least one valid
    // source (no black slivers at the extreme edges).
    CameraStitcher(int camWidth, int camHeight, double camHFovRad,
                    const std::vector<CameraConfig> &cameras,
                    int outWidth, int outHeight, double outHFovRad);

#ifdef PERCEPTION_USE_VPI
    // Owns live VPI handles (payloads/images/stream) on this build --
    // destructor tears them down, so a shallow copy would double-free/
    // double-destroy on scope exit. Only declared under
    // PERCEPTION_USE_VPI: the CPU-only path's members (cv::Mat, already
    // reference-counted) have no such hazard, so that build keeps the
    // compiler-generated destructor/copy ops rather than gaining an
    // artificial copy restriction it doesn't need. perception.cpp only
    // ever constructs one CameraStitcher and never copies it, so this
    // platform-dependent copyability difference has no actual caller
    // impact today.
    ~CameraStitcher();
    CameraStitcher(const CameraStitcher &) = delete;
    CameraStitcher &operator=(const CameraStitcher &) = delete;
#endif

    // images must be CV_8UC3, camWidth x camHeight, same order as the
    // CameraConfig list passed to the constructor.
    cv::Mat Stitch(const std::vector<cv::Mat> &images) const;

private:
    int m_outWidth;
    int m_outHeight;
    cv::Mat m_owner; // CV_8UC1 -- winning camera index per output pixel (255 = none)

#ifdef PERCEPTION_USE_VPI
    struct VpiSourceCam
    {
        // Baked once in the constructor from this camera's mapX/mapY (the
        // same per-pixel geometry the CPU path uses) -- see VPIWarpMap's
        // own docs for why the dense keypoint grid, not a coarser one, was
        // used: this preserves the CPU path's exact per-pixel mapping
        // rather than trading a small amount of accuracy for a coarser,
        // interpolated-between-control-points grid.
        VPIPayload remapPayload = nullptr;
        // Re-pointed at whatever cv::Mat Stitch() receives each call via
        // vpiImageSetWrappedOpenCVMat (cheap -- just updates the wrapper's
        // buffer pointer, no allocation/copy) rather than recreated every
        // frame.
        VPIImage inputWrapper = nullptr;
        // Wraps outputMat below, created once -- VPI writes the remapped
        // result directly into outputMat's own backing memory every call,
        // so reading it back after vpiStreamSync is zero-copy.
        VPIImage outputWrapper = nullptr;
        cv::Mat outputMat;
    };
    VPIStream m_stream = nullptr;
    std::vector<VpiSourceCam> m_vpiSources;
#else
    struct SourceCam
    {
        cv::Mat mapX; // CV_32FC1 -- cv::remap x-coordinates
        cv::Mat mapY; // CV_32FC1 -- cv::remap y-coordinates
    };
    std::vector<SourceCam> m_sources;
#endif
};
