#include "camera_stitcher.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace
{
// Pinhole intrinsics with square pixels, derived from Gazebo's
// horizontal_fov (the only FOV parameter it exposes -- vertical FOV and
// focal length follow from the image's aspect ratio, since pixels are
// square). Exact here because it's simulation; on a real rig this would
// come from an actual calibration instead.
struct Intrinsics
{
    double fx, fy, cx, cy;
};

Intrinsics MakeIntrinsics(int width, int height, double hfovRad)
{
    const double fx = (width / 2.0) / std::tan(hfovRad / 2.0);
    return Intrinsics{fx, fx, width / 2.0, height / 2.0};
}
} // namespace

CameraStitcher::CameraStitcher(int camWidth, int camHeight, double camHFovRad,
                                const std::vector<CameraConfig> &cameras,
                                int outWidth, int outHeight, double outHFovRad)
    : m_outWidth(outWidth), m_outHeight(outHeight)
{
    const Intrinsics K = MakeIntrinsics(camWidth, camHeight, camHFovRad);

    // Cylindrical panorama: azimuth maps LINEARLY to output column (fPano is
    // pixels/radian, not pixels/tan(radian)), so angular resolution is
    // UNIFORM everywhere -- unlike a planar/rectilinear projection, where
    // resolution grows with tan(theta) toward the edges. That explosion is
    // exactly what stretches round 3D objects (cones) into an asymmetric,
    // "warped" silhouette the further they sit from center -- verified
    // directly against this panorama's own cones near the seams before
    // switching back to this. Ground-plane lines bow slightly instead of
    // staying perfectly straight, but that's cosmetic only: deriving a
    // cone's ground position via ray-casting is exactly as valid here as
    // under a planar projection, as long as the correct (cylindrical) pixel
    // -> ray formula is used -- "visually straight" was never a real
    // requirement, just an earlier, overweighted assumption.
    const double fPano = K.fx;
    const double camHalfFov = camHFovRad / 2.0;

    m_sources.resize(cameras.size());
    std::vector<cv::Mat> weights(cameras.size()); // temporary -- only used to pick m_owner below

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        const double yaw = cameras[i].yawRad;
        const double cosYaw = std::cos(yaw);
        const double sinYaw = std::sin(yaw);

        cv::Mat mapX(outHeight, outWidth, CV_32FC1, cv::Scalar(-1));
        cv::Mat mapY(outHeight, outWidth, CV_32FC1, cv::Scalar(-1));
        cv::Mat weight(outHeight, outWidth, CV_32FC1, cv::Scalar(0));

        for (int v = 0; v < outHeight; ++v)
        {
            for (int u = 0; u < outWidth; ++u)
            {
                // theta/h are defined in SDF's own convention (+theta =
                // toward world-left, +h = toward world-up); increasing u
                // scans the panorama left-to-right (world-left to
                // world-right), and increasing v scans top-to-bottom (up to
                // down) -- both the OPPOSITE sense of "positive =
                // left/up", hence the minus signs.
                const double theta = -(u - outWidth / 2.0) / fPano;
                const double h     = -(v - outHeight / 2.0) / fPano;

                // Ray on the unit cylinder, in the reference (front)
                // camera's frame, using SDF-style axes: fwd, left, up.
                const double fwd  = std::cos(theta);
                const double left = std::sin(theta);
                const double up   = h;

                // Un-rotate into this camera's own frame: this camera IS the
                // reference frame rotated by +yaw about the vertical axis, so
                // a reference-frame ray is expressed in its frame by rotating
                // by -yaw.
                const double fwdI  =  fwd * cosYaw + left * sinYaw;
                const double leftI = -fwd * sinYaw + left * cosYaw;
                const double upI   =  up;

                if (fwdI <= 1e-6)
                {
                    continue; // behind this camera
                }

                // image-x increases toward this camera's own right (i.e.
                // decreasing "left"); image-y increases downward (decreasing
                // "up") -- the standard pinhole/image-row convention.
                const double srcU = K.cx - K.fx * (leftI / fwdI);
                const double srcV = K.cy - K.fy * (upI / fwdI);

                if (srcU < 0 || srcU >= camWidth - 1 || srcV < 0 || srcV >= camHeight - 1)
                {
                    continue; // outside this camera's frame
                }

                mapX.at<float>(v, u) = static_cast<float>(srcU);
                mapY.at<float>(v, u) = static_cast<float>(srcV);

                // Distance from this camera's own optical axis (theta ==
                // yaw), used below to pick a single "owning" camera per
                // output pixel -- not to blend (see camera_stitcher.hpp for
                // why blending independently-rendered views causes ghosting
                // on small objects).
                const double distFromCenter = std::abs(theta - yaw);
                weight.at<float>(v, u) = static_cast<float>(
                    std::clamp(1.0 - distFromCenter / camHalfFov, 0.0, 1.0));
            }
        }

        m_sources[i] = SourceCam{mapX, mapY};
        weights[i] = weight;
    }

    // Hard seam: each output pixel is owned by whichever camera's weight is
    // highest there (i.e. whichever camera is angularly closest to that
    // pixel), placing the seam at the midpoint of each overlap. 255 = no
    // camera covers this pixel (shouldn't happen while outHFovRad stays
    // inside the cameras' true combined coverage).
    m_owner = cv::Mat(outHeight, outWidth, CV_8UC1, cv::Scalar(255));
    cv::Mat bestWeight(outHeight, outWidth, CV_32FC1, cv::Scalar(0));
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        for (int v = 0; v < outHeight; ++v)
        {
            for (int u = 0; u < outWidth; ++u)
            {
                const float w = weights[i].at<float>(v, u);
                if (w > bestWeight.at<float>(v, u))
                {
                    bestWeight.at<float>(v, u) = w;
                    m_owner.at<uint8_t>(v, u) = static_cast<uint8_t>(i);
                }
            }
        }
    }
}

cv::Mat CameraStitcher::Stitch(const std::vector<cv::Mat> &images) const
{
    cv::Mat result(m_outHeight, m_outWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    for (size_t i = 0; i < images.size(); ++i)
    {
        cv::Mat warped;
        cv::remap(images[i], warped, m_sources[i].mapX, m_sources[i].mapY,
                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        const cv::Mat ownerMask = (m_owner == static_cast<uint8_t>(i));
        warped.copyTo(result, ownerMask);
    }

    return result;
}
