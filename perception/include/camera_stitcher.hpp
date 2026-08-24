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
//
// Enabling VPI alone did NOT fix the ~36-38ms/frame cost, though --
// confirmed directly via /timing/perception/stitch still reading ~40ms with
// VPI active. The remaining cost was the masked copyTo step, done at FULL
// panorama size for EACH of the 3 cameras regardless of remap backend --
// 3x more pixel-copy work than actually needed, since each camera only
// ever owns roughly a third of the panorama (see m_owner's comment:
// ownership is column-only, so each camera's region is one contiguous
// band, not scattered pixels). Both the VPI output buffers and the CPU
// cv::remap destination are now sized to each camera's own bounding rect
// (see VpiSourceCam::roi / SourceCam::roi) instead of the full panorama,
// so the masked copy -- and, on the VPI side, the remap itself -- only
// ever touches the pixels that camera can actually contribute, cutting
// total work from ~3x panorama size down to ~1x (the 3 rects tile the
// panorama with only a thin overlap at each seam).
class CameraStitcher
{
public:
    struct CameraConfig
    {
        // Yaw of this camera relative to the reference camera (index 0),
        // radians, positive = rotated toward the reference camera's left,
        // measured about the CHASSIS's own vertical axis (model.sdf's own
        // yaw convention) -- NOT about the reference camera's own (already
        // pitched) local "up" axis. Those two are only the same rotation
        // when sharedPitchRad is 0; see the constructor's own comment for
        // why conflating them was a real, confirmed bug for this rig, which
        // does pitch every camera.
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
    // sharedPitchRad: the pitch every camera shares (true for this rig --
    // model.sdf gives camera_front/left/right the identical 0.3 rad pitch,
    // only yaw differs). Pitch/roll being identical across cameras does NOT
    // mean a single yaw angle fully describes their relative rotation, as
    // this class used to assume (see git history) -- when sharedPitchRad is
    // nonzero, the pitch and yaw rotations don't commute, so the true
    // relative rotation between two cameras that share a pitch but differ
    // in yaw is Ry(-pitch)*Rz(-yaw)*Ry(pitch), not a bare Rz(-yaw) about the
    // reference camera's own local "up". Confirmed as a real, non-
    // hypothetical bug (2026-08-23): a live ground-truth comparison found
    // cone-detection error growing sharply with azimuth (0.09m median near
    // dead-ahead vs 1.0m+ past ~30deg, almost entirely in the TANGENTIAL/
    // bearing component, not radial/range) -- exactly the signature of an
    // angular error that's zero on-axis and grows off-axis, not sensor
    // noise. The bare-Rz(-yaw) approximation this replaced was off by
    // ~4deg at 0.68 rad yaw / 0.3 rad pitch (verified numerically), which
    // alone accounts for the observed magnitude at typical cone range.
    CameraStitcher(int camWidth, int camHeight, double camHFovRad,
                    const std::vector<CameraConfig> &cameras,
                    int outWidth, int outHeight, double outHFovRad,
                    double sharedPitchRad);

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
    // CV_8UC1 -- winning camera index per output pixel (255 = none). The
    // "winning" decision (see the .cpp constructor) depends only on output
    // COLUMN, not row (theta -- and hence weight -- is a pure function of
    // u), so each camera's owned region is a single contiguous band running
    // the full output height, not scattered pixels -- that's what makes a
    // per-camera bounding rect (see VpiSourceCam::roi / SourceCam::roi) a
    // tight fit rather than a loose one that still covers most of the
    // panorama.
    cv::Mat m_owner;

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
        // This camera's own bounding rect within the FULL panorama (see
        // m_owner's own comment below) -- remap/output only ever covers
        // this camera's OWN pixels, not the whole panorama, so outputMat is
        // sized to roi, not outWidth x outHeight.
        cv::Rect roi;
    };
    VPIStream m_stream = nullptr;
    std::vector<VpiSourceCam> m_vpiSources;
#else
    struct SourceCam
    {
        cv::Mat mapX; // CV_32FC1 -- cv::remap x-coordinates, cropped to roi
        cv::Mat mapY; // CV_32FC1 -- cv::remap y-coordinates, cropped to roi
        cv::Rect roi; // see VpiSourceCam::roi's comment above -- same idea, CPU path
    };
    std::vector<SourceCam> m_sources;
#endif
};
