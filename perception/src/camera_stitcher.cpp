#include "camera_stitcher.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

#ifdef PERCEPTION_USE_VPI
#include <vpi/OpenCVInterop.hpp>
#include <vpi/Status.h>

#include <sstream>
#include <stdexcept>

namespace
{
// VPI calls return a VPIStatus rather than throwing -- this converts a
// failure into an exception with VPI's own detailed message attached
// (vpiGetLastStatusMessage), rather than a bare status code that would
// leave "why" to manual lookup. Thrown, not just logged: a construction-
// time failure here (e.g. VIC/CUDA backend genuinely unavailable despite
// PERCEPTION_USE_VPI being compiled in) means CameraStitcher can't function
// at all, so perception.cpp's own startup should fail loudly rather than
// silently run with a half-initialized stitcher.
void CheckVpiStatus(VPIStatus status, const char *what)
{
    if (status != VPI_SUCCESS)
    {
        char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];
        vpiGetLastStatusMessage(buffer, sizeof(buffer));
        std::ostringstream ss;
        ss << what << " failed: " << vpiStatusGetName(status) << ": " << buffer;
        throw std::runtime_error(ss.str());
    }
}
} // namespace
#endif

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

#ifdef PERCEPTION_USE_VPI
    CheckVpiStatus(vpiStreamCreate(VPI_BACKEND_CUDA, &m_stream), "vpiStreamCreate");
    m_vpiSources.resize(cameras.size());
#else
    m_sources.resize(cameras.size());
#endif
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

#ifdef PERCEPTION_USE_VPI
        // Bake mapX/mapY into a dense VPIWarpMap -- one control point per
        // OUTPUT pixel (horizInterval/vertInterval = 1), matching the CPU
        // path's per-pixel accuracy exactly rather than trading it for a
        // coarser, VPI-interpolated grid. (0,0) sentinel-free: mapX/mapY's
        // own -1 default (see above, "outside this camera's frame") is
        // passed straight through as the keypoint coordinate -- VPI_BORDER_
        // ZERO (see Stitch() below) treats any keypoint outside the source
        // image as zero, same as cv::remap's BORDER_CONSTANT does today,
        // so this preserves the exact same "black where this camera has no
        // coverage" behavior with no extra sentinel handling needed here.
        VPIWarpMap warpMap = {};
        warpMap.grid.numHorizRegions = 1;
        warpMap.grid.numVertRegions = 1;
        warpMap.grid.regionWidth[0] = static_cast<int16_t>(outWidth);
        warpMap.grid.regionHeight[0] = static_cast<int16_t>(outHeight);
        warpMap.grid.horizInterval[0] = 1;
        warpMap.grid.vertInterval[0] = 1;
        CheckVpiStatus(vpiWarpMapAllocData(&warpMap), "vpiWarpMapAllocData");

        // BUG FIX: this used to loop v/u up to warpMap.numVertPoints/
        // numHorizPoints -- confirmed via a live crash (SIGSEGV inside this
        // loop, caught with gdb) that those are NOT equal to outHeight/
        // outWidth: vpiWarpMapAllocData pads the control-point grid up to a
        // multiple of 16 internally (confirmed directly in the crashing
        // process: numHorizPoints=2432 vs the real outWidth=2427,
        // numVertPoints=1136 vs outHeight=1130), so iterating to those
        // padded bounds read mapX/mapY (sized exactly outHeight x outWidth)
        // out of bounds. pitchBytes IS still respected exactly for the
        // WRITE side (row stride into warpMap.keypoints, which VPI sized
        // generously enough to cover the padded width) -- only the loop
        // bounds themselves needed to be outHeight/outWidth, matching what
        // mapX/mapY actually contain. Any padding rows/columns VPI
        // allocated beyond outHeight/outWidth are simply left untouched --
        // Remap's own output is defined by regionWidth[0]/regionHeight[0]
        // (outWidth/outHeight, see the grid setup above), so that padding
        // is purely an internal alignment detail, never mapped to a real
        // output pixel.
        for (int v = 0; v < outHeight; ++v)
        {
            VPIKeypointF32 *row = reinterpret_cast<VPIKeypointF32 *>(
                reinterpret_cast<uint8_t *>(warpMap.keypoints) + v * warpMap.pitchBytes);
            for (int u = 0; u < outWidth; ++u)
            {
                row[u] = VPIKeypointF32{mapX.at<float>(v, u), mapY.at<float>(v, u)};
            }
        }

        CameraStitcher::VpiSourceCam &src = m_vpiSources[i];
        CheckVpiStatus(vpiCreateRemap(VPI_BACKEND_CUDA, &warpMap, &src.remapPayload), "vpiCreateRemap");
        vpiWarpMapFreeData(&warpMap); // payload has what it needs now -- see the fisheye sample's same pattern

        // Input wrapper seeded with a throwaway same-size/type Mat just to
        // establish format/dimensions -- Stitch() repoints it at the real
        // per-frame image via vpiImageSetWrappedOpenCVMat every call.
        cv::Mat placeholder(camHeight, camWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        CheckVpiStatus(
            vpiImageCreateWrapperOpenCVMat(placeholder, VPI_IMAGE_FORMAT_RGB8, 0, &src.inputWrapper),
            "vpiImageCreateWrapperOpenCVMat (input)");

        // Output buffer is OWNED here (unlike the input) and wrapped once
        // -- VPI writes directly into this cv::Mat's own backing memory on
        // every vpiSubmitRemap, so Stitch() reads it back with no extra
        // copy after vpiStreamSync.
        src.outputMat = cv::Mat(outHeight, outWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        CheckVpiStatus(
            vpiImageCreateWrapperOpenCVMat(src.outputMat, VPI_IMAGE_FORMAT_RGB8, 0, &src.outputWrapper),
            "vpiImageCreateWrapperOpenCVMat (output)");
#else
        m_sources[i] = SourceCam{mapX, mapY};
#endif
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

#ifdef PERCEPTION_USE_VPI
CameraStitcher::~CameraStitcher()
{
    // Sync first: destroying an image/payload that a still-in-flight
    // submitted op references is undefined, so make sure nothing is
    // outstanding before tearing anything down. Destroy order otherwise
    // doesn't matter to VPI (handles are independently reference-counted),
    // this just follows the natural create-order-reversed convention.
    if (m_stream != nullptr)
    {
        vpiStreamSync(m_stream);
    }
    for (auto &src : m_vpiSources)
    {
        vpiPayloadDestroy(src.remapPayload);
        vpiImageDestroy(src.inputWrapper);
        vpiImageDestroy(src.outputWrapper);
    }
    vpiStreamDestroy(m_stream);
}
#endif

cv::Mat CameraStitcher::Stitch(const std::vector<cv::Mat> &images) const
{
    cv::Mat result(m_outHeight, m_outWidth, CV_8UC3, cv::Scalar(0, 0, 0));

#ifdef PERCEPTION_USE_VPI
    for (size_t i = 0; i < images.size(); ++i)
    {
        const auto &src = m_vpiSources[i];
        CheckVpiStatus(vpiImageSetWrappedOpenCVMat(src.inputWrapper, images[i]),
                       "vpiImageSetWrappedOpenCVMat");
        CheckVpiStatus(
            vpiSubmitRemap(m_stream, VPI_BACKEND_CUDA, src.remapPayload, src.inputWrapper,
                           src.outputWrapper, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0),
            "vpiSubmitRemap");
    }
    // One sync after submitting all 3 cameras' remaps, not one per camera
    // -- the 3 ops queue on the SAME stream and run in submission order
    // regardless, so an earlier per-camera sync would only have serialized
    // waiting without changing execution order, just added latency for no
    // benefit. (Running the 3 cameras across separate streams for genuine
    // GPU-side concurrency is a further, not-yet-done optimization -- see
    // this class's own header comment.)
    CheckVpiStatus(vpiStreamSync(m_stream), "vpiStreamSync");

    for (size_t i = 0; i < images.size(); ++i)
    {
        const cv::Mat ownerMask = (m_owner == static_cast<uint8_t>(i));
        m_vpiSources[i].outputMat.copyTo(result, ownerMask);
    }
#else
    for (size_t i = 0; i < images.size(); ++i)
    {
        cv::Mat warped;
        cv::remap(images[i], warped, m_sources[i].mapX, m_sources[i].mapY,
                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        const cv::Mat ownerMask = (m_owner == static_cast<uint8_t>(i));
        warped.copyTo(result, ownerMask);
    }
#endif

    return result;
}
