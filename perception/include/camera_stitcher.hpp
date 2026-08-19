#pragma once

#include <opencv2/core.hpp>
#include <vector>

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
// Stitch() is then just N remaps + a masked copy, cheap enough to run every
// frame.
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

    // images must be CV_8UC3, camWidth x camHeight, same order as the
    // CameraConfig list passed to the constructor.
    cv::Mat Stitch(const std::vector<cv::Mat> &images) const;

private:
    struct SourceCam
    {
        cv::Mat mapX; // CV_32FC1 -- cv::remap x-coordinates
        cv::Mat mapY; // CV_32FC1 -- cv::remap y-coordinates
    };

    int m_outWidth;
    int m_outHeight;
    std::vector<SourceCam> m_sources;
    cv::Mat m_owner; // CV_8UC1 -- winning camera index per output pixel (255 = none)
};
