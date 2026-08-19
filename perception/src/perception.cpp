#include <gz/transport/Node.hh>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/pointcloud_packed.pb.h>

#include <opencv2/core.hpp>

#include <common/types.hpp>
#include "camera_stitcher.hpp"
#include "cone_detector.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

// Perception process
// ------------------
// Inputs (subscribe):
//   /camera/front/image   gz::msgs::Image
//   /camera/left/image    gz::msgs::Image
//   /camera/right/image   gz::msgs::Image
//   /lidar/points         gz::msgs::PointCloudPacked
//
// Output (publish):
//   /camera/stitched/image  gz::msgs::Image  -- the 3 cameras merged into
//                            one cylindrical panorama (see camera_stitcher.hpp
//                            for why cylindrical, not planar -- uniform
//                            angular resolution avoids stretching round
//                            objects like cones, which planar's tan(theta)
//                            resolution growth does not). Used for BOTH
//                            visualization and detection (TODO 2) -- a planar
//                            projection was tried first and specifically
//                            ruled out for detection input because of that
//                            stretching, confirmed directly against this
//                            panorama's own cones near the seams; cylindrical
//                            doesn't have that problem, so the single-pass
//                            (not per-camera) detection plan holds.
//   /cone_detections      gz::msgs::Pose_V  (or a custom proto once protoc is added)
//
// The camera geometry constants below (position being shared across all 3,
// horizontal_fov, left/right yaw) mirror simulation/models/fsd_car/model.sdf
// and must be kept in sync with it by hand -- unlike foxglove_bridge.cpp's
// /tf tree, which fetches sensor extrinsics live from the
// /world/<world>/scene/info service instead of duplicating them. Wiring this
// through that same service would remove the duplication; skipped for now
// for a faster first pass.
//
// Steps to implement (do in order):
//   TODO 1: Subscribe to one camera topic and print the image dimensions to
//            confirm messages are arriving (roadmap step 5). DONE.
//   TODO 1b: Stitch all 3 camera feeds into one panorama, for visualization
//            AND as the detection input (single inference pass). DONE.
//   TODO 2: Load the YOLO ONNX model via ONNX Runtime and run inference on
//            the stitched frame -- detect bounding boxes and classify cone
//            color. DONE (see cone_detector.hpp) -- CPU execution provider
//            only for now; the Jetson has no prebuilt CUDA ONNX Runtime
//            release (checked, see CMakeLists.txt), so real GPU inference
//            there needs a from-source build, tracked as separate follow-up
//            work, not blocking this step.
//   TODO 3: Subscribe to /lidar/points and project each detected bounding
//            box into the point cloud to get a 3-D cone position in the car
//            frame -- using the cylindrical pixel -> ray formula (not a
//            plain pinhole one), since detections come from the stitched
//            frame now.
//   TODO 4: Pack (x, y, color) for each detection and publish on /cone_detections.

namespace
{
constexpr int kCamWidth = 1440;
constexpr int kCamHeight = 1080;
constexpr double kCamHFovRad = 0.7854;   // 45 deg -- model.sdf <horizontal_fov>
constexpr double kCamYawLeftRad = 0.6;   // model.sdf camera_left <pose> yaw
constexpr double kCamYawRightRad = -0.6; // model.sdf camera_right <pose> yaw

// Panorama size / FOV: 80 deg (a bit under double a single camera's own
// 45 deg), prioritizing vertical lookahead over peripheral horizontal
// coverage.
//
// Cylindrical projection (see camera_stitcher.hpp for why, over planar):
// width uses the LINEAR cylindrical width-from-FOV formula (f * HFovRad,
// not 2*f*tan(HFovRad/2)) so panorama-center resolution matches the source
// cameras' own focal length (fx = 720 / tan(22.5 deg) =~ 1738 px/rad):
// 1738 * 1.3963 =~ 2427 -- notably narrower than the ~2917 a planar
// projection needed for the same 80 deg, since cylindrical has no
// tan(theta) blowup toward the edges.
//
// Height: srcV/srcU are computed relative to the OUTPUT's own reference
// axis (front, theta=0). For FRONT (yaw=0), fwd/fwdI ratio = cos(theta),
// same as any single rectilinear camera's own falloff toward its edges --
// 1.0 at center, worst at FRONT's own edges (the seams, offset = 34.377/2
// = 17.19 deg from center): cos(17.19 deg) =~ 0.9553. For LEFT/RIGHT the
// same ratio is 1.0 at THEIR OWN optical center (yaw = +-34.377 deg, which
// falls inside their owned range) and, examined via calculus, monotonic in
// theta within their owned range -- so it's ALSO worst exactly at the
// seams (0.9553, matching FRONT's value there by symmetry), improving in
// both directions from there (back toward center, and back out toward the
// wide edges). So the true worst case across the WHOLE panorama is at the
// two seams, not the outer edges: needed height =
// 2 * fx * (source's own half-height / fy) / 0.9553 =~ 1130 -- covers
// every pixel with no gaps, no per-camera-region cropping tradeoff needed
// (unlike the planar version this replaced).
constexpr int kPanoWidth = 2427;
constexpr int kPanoHeight = 1130;
constexpr double kPanoHFovRad = 1.3963; // 80 deg

// Run from the repo root (matches every other script/binary in this
// project -- see dev_sim.sh), so this resolves relative to /workspace.
// runs/detect/runs/detect/... (doubly-nested, and NOT under ml/, despite
// train.py's project="runs/detect" default reading like it should resolve
// under wherever it's invoked from) is where ultralytics actually wrote
// this on the Jetson -- confirmed by listing the real directory over SSH,
// not assumed from the default parameter alone. Git-ignored (see
// .gitignore), so this only exists on whichever machine actually ran
// training (the Jetson, as of this writing) -- doesn't need to, and isn't
// meant to, exist in a fresh clone.
constexpr const char *kModelPath = "runs/detect/runs/detect/train/weights/best.onnx";
constexpr float kConfThreshold = 0.25f;

// Matches ml/prepare_data.py's CLASSES list exactly -- same order, since
// that's the order the model was trained to output class indices in.
constexpr const char *kClassNames[] = {"blue", "yellow", "orange", "large_orange"};

gz::msgs::Image ToImageMsg(const cv::Mat &_img)
{
    gz::msgs::Image msg;
    msg.set_width(static_cast<unsigned int>(_img.cols));
    msg.set_height(static_cast<unsigned int>(_img.rows));
    msg.set_step(static_cast<unsigned int>(_img.step));
    msg.set_pixel_format_type(gz::msgs::PixelFormatType::RGB_INT8);
    msg.set_data(_img.data, _img.total() * _img.elemSize());
    return msg;
}

// Wraps the message's own buffer as a cv::Mat with no copy -- matches
// RGB_INT8, which is what a plain <camera> sensor block emits by default
// (no <format> element set in model.sdf). Caller must .clone() before the
// backing gz::msgs::Image goes away if the data needs to outlive it.
cv::Mat FromImageMsg(const gz::msgs::Image &_msg)
{
    return cv::Mat(static_cast<int>(_msg.height()), static_cast<int>(_msg.width()),
                    CV_8UC3, const_cast<char *>(_msg.data().data()),
                    static_cast<size_t>(_msg.step()));
}

// Simulation timestamp, used to check whether 3 independently-arriving
// camera frames actually belong to (close enough to) the same tick -- see
// tryStitchAndPublish in main() for why "latest" isn't good enough on its
// own, and why this needs a tolerance rather than exact equality: measured
// empirically, camera_front's stamp is consistently ~1 tick (1/30 s) ahead
// of camera_left/camera_right's, every frame -- a fixed pipeline skew (very
// likely sensor update ordering inside gz-sim, front being declared first
// in the SDF), not random jitter. Exact-match would reject every frame.
struct StampKey
{
    int64_t sec = -1;
    int32_t nsec = -1;

    double ToSeconds() const { return static_cast<double>(sec) + static_cast<double>(nsec) * 1e-9; }
};

// Tolerance: comfortably above the observed ~33 ms front-vs-left/right skew
// (one 30 Hz tick), comfortably below 2 ticks (~67 ms), so a genuine
// 2-tick-stale mismatch still gets rejected.
constexpr double kStampToleranceSec = 0.045;

bool StampsClose(const StampKey &_a, const StampKey &_b)
{
    return std::abs(_a.ToSeconds() - _b.ToSeconds()) <= kStampToleranceSec;
}

StampKey ToStampKey(const gz::msgs::Image &_msg)
{
    return StampKey{_msg.header().stamp().sec(), _msg.header().stamp().nsec()};
}
} // namespace

int main()
{
    gz::transport::Node node;

    CameraStitcher stitcher(
        kCamWidth, kCamHeight, kCamHFovRad,
        {{0.0}, {kCamYawLeftRad}, {kCamYawRightRad}}, // front, left, right
        kPanoWidth, kPanoHeight, kPanoHFovRad);

    // kModelPath is git-ignored (see .gitignore) and only exists on a
    // machine that's actually run training -- fail with a clear, actionable
    // message rather than letting ONNX Runtime's own (much less obvious)
    // exception surface if it's missing on this checkout.
    std::unique_ptr<ConeDetector> detector;
    try
    {
        detector = std::make_unique<ConeDetector>(kModelPath, kConfThreshold);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load cone detection model at '" << kModelPath << "': " << e.what()
                   << "\nDid you run training (ml/train.py) on this machine, or copy the .onnx"
                      " file over from one that did?\n";
        return 1;
    }

    auto stitchedPub = node.Advertise<gz::msgs::Image>("/camera/stitched/image");

    std::mutex mtx;
    cv::Mat latestFront, latestLeft, latestRight;
    StampKey stampFront{}, stampLeft{}, stampRight{};
    bool haveFront = false, haveLeft = false, haveRight = false;

    // Stitches only when all 3 buffered frames carry CLOSE-ENOUGH simulation
    // timestamps (see StampsClose/kStampToleranceSec) -- not just "whatever's
    // latest", and not exact equality either. Triggering on "latest" sounds
    // harmless (all 3 cameras publish at the same 30 Hz), but gz-transport
    // doesn't guarantee their callbacks fire in lockstep, so "latest" can
    // silently mean "this tick's front + last tick's left/right" -- verified
    // empirically: camera_front's stamp runs a consistent ~1 tick ahead of
    // camera_left/right's, every frame (a fixed pipeline skew, not jitter).
    // Any such mismatch shows up as a faint double image at the blended
    // seams -- worst on small/thin objects, since a sub-pixel shift is a
    // large fraction of their size. Exact equality would reject every frame
    // given that fixed skew, hence the tolerance instead.
    auto tryStitchAndPublish = [&]()
    {
        if (!(haveFront && haveLeft && haveRight))
        {
            return;
        }
        if (!(StampsClose(stampFront, stampLeft) && StampsClose(stampLeft, stampRight)))
        {
            return;
        }
        const cv::Mat stitched = stitcher.Stitch({latestFront, latestLeft, latestRight});
        stitchedPub.Publish(ToImageMsg(stitched));

        // TODO 2 confirmation step: print what's detected, matching the same
        // "print to confirm it's actually working" pattern as TODO 1's
        // camera-dimension print. TODO 3/4 (lidar projection + publishing
        // to /cone_detections) replace this with real downstream use.
        const std::vector<ConeDetector::Detection> detections = detector->Detect(stitched);
        for (const auto &d : detections)
        {
            const char *className = (d.classId >= 0 && static_cast<size_t>(d.classId) <
                                      sizeof(kClassNames) / sizeof(kClassNames[0]))
                                          ? kClassNames[d.classId]
                                          : "unknown";
            std::cout << "  cone: " << className << " conf=" << d.confidence << " bbox=["
                       << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "]\n";
        }
    };

    std::function<void(const gz::msgs::Image &)> onCameraFront =
        [&](const gz::msgs::Image &_msg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "camera_front: " << _msg.width() << "x" << _msg.height()
                   << " format=" << gz::msgs::PixelFormatType_Name(_msg.pixel_format_type()) << '\n';
        latestFront = FromImageMsg(_msg).clone();
        stampFront = ToStampKey(_msg);
        haveFront = true;
        tryStitchAndPublish();
    };

    std::function<void(const gz::msgs::Image &)> onCameraLeft =
        [&](const gz::msgs::Image &_msg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        latestLeft = FromImageMsg(_msg).clone();
        stampLeft = ToStampKey(_msg);
        haveLeft = true;
        tryStitchAndPublish();
    };

    std::function<void(const gz::msgs::Image &)> onCameraRight =
        [&](const gz::msgs::Image &_msg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        latestRight = FromImageMsg(_msg).clone();
        stampRight = ToStampKey(_msg);
        haveRight = true;
        tryStitchAndPublish();
    };

    if (!node.Subscribe("/camera/front/image", onCameraFront))
    {
        std::cerr << "Failed to subscribe to /camera/front/image\n";
        return 1;
    }
    if (!node.Subscribe("/camera/left/image", onCameraLeft))
    {
        std::cerr << "Failed to subscribe to /camera/left/image\n";
        return 1;
    }
    if (!node.Subscribe("/camera/right/image", onCameraRight))
    {
        std::cerr << "Failed to subscribe to /camera/right/image\n";
        return 1;
    }

    gz::transport::waitForShutdown();
    return 0;
}
