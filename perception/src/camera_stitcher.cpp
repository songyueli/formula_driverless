#include "camera_stitcher.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

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
                                int outWidth, int outHeight, double outHFovRad,
                                double sharedPitchRad)
    : m_outWidth(outWidth), m_outHeight(outHeight)
{
    const Intrinsics K = MakeIntrinsics(camWidth, camHeight, camHFovRad);
    const double cosPitch = std::cos(sharedPitchRad);
    const double sinPitch = std::sin(sharedPitchRad);

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
    // Full-panorama-sized per-camera maps, kept around only long enough to
    // (a) pick m_owner and (b) crop down to each camera's own bounding rect
    // below -- see this class's header comment for why cropping matters
    // (avoids 3x redundant remap/copy work). Not stored as class members:
    // once cropped into m_vpiSources[i]/m_sources[i], the full-size version
    // has no further use.
    std::vector<cv::Mat> fullMapXs(cameras.size()), fullMapYs(cameras.size());

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

                // Un-rotate into this camera's own frame. NOT a bare -yaw
                // rotation about the reference (front) camera's own local
                // "up" axis -- that was this class's original approach, and
                // a confirmed real bug (see this class's header comment):
                // model.sdf specifies yaw about the CHASSIS's vertical axis,
                // not each already-pitched camera's own tilted one, and
                // those two axes only coincide when pitch is 0. Applied here
                // as 3 explicit rotations (matching model.sdf's own
                // roll-pitch-yaw convention step for step, R = Rz(yaw) *
                // Ry(pitch) * Rx(roll) with roll=0) rather than one
                // hand-multiplied matrix, so each step can be checked
                // against a named rotation directly: undo the shared pitch
                // to reach the chassis frame, apply -yaw THERE (the chassis'
                // own vertical axis, where model.sdf's yaw is actually
                // specified), then re-apply pitch to land in camera i's own
                // (also pitched) frame.
                const double aFwd  = fwd * cosPitch + up * sinPitch;
                const double aLeft = left;
                const double aUp   = -fwd * sinPitch + up * cosPitch;

                const double bFwd  = aFwd * cosYaw + aLeft * sinYaw;
                const double bLeft = -aFwd * sinYaw + aLeft * cosYaw;
                const double bUp   = aUp;

                const double fwdI  = bFwd * cosPitch - bUp * sinPitch;
                const double leftI = bLeft;
                const double upI   = bFwd * sinPitch + bUp * cosPitch;

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

        fullMapXs[i] = mapX;
        fullMapYs[i] = mapY;
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

    // Pass 2: now that m_owner is final, compute each camera's own
    // bounding rect (see m_owner's header comment -- ownership is
    // column-only, so this is a tight fit, not a loose one) and build the
    // per-camera remap/output resources scoped to JUST that rect instead
    // of the full outWidth x outHeight panorama -- see this class's header
    // comment for why (cuts ~3x redundant remap/copy work down to ~1x).
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        int minU = outWidth, maxU = -1, minV = outHeight, maxV = -1;
        for (int v = 0; v < outHeight; ++v)
        {
            for (int u = 0; u < outWidth; ++u)
            {
                if (m_owner.at<uint8_t>(v, u) == static_cast<uint8_t>(i))
                {
                    minU = std::min(minU, u);
                    maxU = std::max(maxU, u);
                    minV = std::min(minV, v);
                    maxV = std::max(maxV, v);
                }
            }
        }
        // maxU < minU would mean this camera owns NO pixels at all --
        // can't happen with 3 cameras covering a shared 80 deg FOV each
        // contributing its own share, so not handled as a real case (an
        // empty rect below would just make this camera silently invisible
        // in Stitch(), which is a real bug, not a graceful degradation).
        //
        // Width/height padded up to a multiple of 16 (confirmed directly,
        // not assumed, as the fix for a real crash: at some camera
        // resolutions -- reproduced deterministically, including on a
        // freshly rebooted Jetson with zero prior GPU/driver state, so
        // this isn't leftover corruption from other testing -- an
        // unpadded rect made vpiCreateRemap() throw
        // cudaErrorIllegalAddress inside libnvvpi.so itself, every time,
        // for a rect whose width/height were NOT multiples of 16; the
        // vpiWarpMapAllocData call directly below already documents that
        // VPI pads its own internal control-point storage to a multiple
        // of 16 -- padding the REQUESTED region to match here means VPI
        // is never asked to build a remap payload for a non-16-aligned
        // logical size in the first place, sidestepping whatever its own
        // internal padding does wrong in that case, rather than trying to
        // fix a closed-source library from outside it). Grown
        // symmetrically where possible, clamped to stay within the full
        // panorama -- correctness is unaffected either way: Stitch()'s
        // ownerMask (see below) already masks the composited output down
        // to just this camera's OWN pixels regardless of how much extra
        // margin the rect itself carries, so a few extra border pixels
        // being remapped-but-unused costs a little wasted work, not
        // correctness.
        const auto roundUp16 = [](int v) { return ((v + 15) / 16) * 16; };
        const int rawWidth = maxU - minU + 1;
        const int rawHeight = maxV - minV + 1;
        const int padWidth = std::min(outWidth, roundUp16(rawWidth));
        const int padHeight = std::min(outHeight, roundUp16(rawHeight));
        int rectX = std::clamp(minU - (padWidth - rawWidth) / 2, 0, outWidth - padWidth);
        int rectY = std::clamp(minV - (padHeight - rawHeight) / 2, 0, outHeight - padHeight);
        const cv::Rect rect(rectX, rectY, padWidth, padHeight);
        const cv::Mat croppedMapX = fullMapXs[i](rect);
        const cv::Mat croppedMapY = fullMapYs[i](rect);

#ifdef PERCEPTION_USE_VPI
        // Bake the CROPPED mapX/mapY into a dense VPIWarpMap -- one control
        // point per pixel of THIS camera's own rect (horizInterval/
        // vertInterval = 1), matching the CPU path's per-pixel accuracy
        // exactly rather than trading it for a coarser, VPI-interpolated
        // grid. (0,0) sentinel-free: mapX/mapY's own -1 default (see the
        // pass-1 loop above, "outside this camera's frame") is passed
        // straight through as the keypoint coordinate -- VPI_BORDER_ZERO
        // (see Stitch() below) treats any keypoint outside the source
        // image as zero, same as cv::remap's BORDER_CONSTANT does today,
        // so this preserves the exact same "black where this camera has no
        // coverage" behavior with no extra sentinel handling needed here.
        VPIWarpMap warpMap = {};
        warpMap.grid.numHorizRegions = 1;
        warpMap.grid.numVertRegions = 1;
        warpMap.grid.regionWidth[0] = static_cast<int16_t>(rect.width);
        warpMap.grid.regionHeight[0] = static_cast<int16_t>(rect.height);
        warpMap.grid.horizInterval[0] = 1;
        warpMap.grid.vertInterval[0] = 1;
        CheckVpiStatus(vpiWarpMapAllocData(&warpMap), "vpiWarpMapAllocData");

        // Same padding gotcha as before cropping was added (see git
        // history / this class's own past fix): vpiWarpMapAllocData pads
        // the control-point grid up to a multiple of 16 internally, so the
        // fill loop below MUST stay bounded by rect.width/rect.height (what
        // croppedMapX/Y actually contain), not warpMap's own possibly-
        // larger numHorizPoints/numVertPoints.
        for (int v = 0; v < rect.height; ++v)
        {
            VPIKeypointF32 *row = reinterpret_cast<VPIKeypointF32 *>(
                reinterpret_cast<uint8_t *>(warpMap.keypoints) + v * warpMap.pitchBytes);
            for (int u = 0; u < rect.width; ++u)
            {
                row[u] = VPIKeypointF32{croppedMapX.at<float>(v, u), croppedMapY.at<float>(v, u)};
            }
        }

        CameraStitcher::VpiSourceCam &src = m_vpiSources[i];
        CheckVpiStatus(vpiCreateRemap(VPI_BACKEND_CUDA, &warpMap, &src.remapPayload), "vpiCreateRemap");
        vpiWarpMapFreeData(&warpMap); // payload has what it needs now -- see the fisheye sample's same pattern
        src.roi = rect;

        // Input wrapper seeded with a throwaway same-size/type Mat just to
        // establish format/dimensions -- Stitch() repoints it at the real
        // per-frame image via vpiImageSetWrappedOpenCVMat every call. Full
        // camera size, NOT cropped -- the INPUT is always the whole source
        // camera frame; only the OUTPUT is scoped to this camera's rect.
        cv::Mat placeholder(camHeight, camWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        CheckVpiStatus(
            vpiImageCreateWrapperOpenCVMat(placeholder, VPI_IMAGE_FORMAT_RGB8, 0, &src.inputWrapper),
            "vpiImageCreateWrapperOpenCVMat (input)");

        // Output buffer is OWNED here (unlike the input) and wrapped once
        // -- VPI writes directly into this cv::Mat's own backing memory on
        // every vpiSubmitRemap, so Stitch() reads it back with no extra
        // copy after vpiStreamSync. Sized to rect, not the full panorama.
        src.outputMat = cv::Mat(rect.height, rect.width, CV_8UC3, cv::Scalar(0, 0, 0));
        CheckVpiStatus(
            vpiImageCreateWrapperOpenCVMat(src.outputMat, VPI_IMAGE_FORMAT_RGB8, 0, &src.outputWrapper),
            "vpiImageCreateWrapperOpenCVMat (output)");
#else
        m_sources[i] = SourceCam{croppedMapX.clone(), croppedMapY.clone(), rect};
#endif
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

    // Masked copy scoped to each camera's own rect (see VpiSourceCam::roi's
    // comment) -- ownerMask/result(roi) are both roi-sized now, not full
    // panorama, cutting this from 3x panorama-sized work down to ~1x.
    for (size_t i = 0; i < images.size(); ++i)
    {
        const cv::Rect &roi = m_vpiSources[i].roi;
        const cv::Mat ownerMask = (m_owner(roi) == static_cast<uint8_t>(i));
        m_vpiSources[i].outputMat.copyTo(result(roi), ownerMask);
    }
#else
    for (size_t i = 0; i < images.size(); ++i)
    {
        const cv::Rect &roi = m_sources[i].roi;
        cv::Mat warped;
        cv::remap(images[i], warped, m_sources[i].mapX, m_sources[i].mapY,
                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        const cv::Mat ownerMask = (m_owner(roi) == static_cast<uint8_t>(i));
        warped.copyTo(result(roi), ownerMask);
    }
#endif

    return result;
}
