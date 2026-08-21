#include <gz/transport/Node.hh>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/pointcloud_packed.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/uint64.pb.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <common/scoped_timer.hpp>
#include <common/types.hpp>
#include "camera_stitcher.hpp"
#include "cone_detector.hpp"
#ifdef PERCEPTION_USE_TENSORRT
#include "cone_detector_trt.hpp"
#endif
#include "lidar_projector.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

// Perception process
// ------------------
// Inputs (subscribe):
//   /camera/front/image     gz::msgs::Image
//   /camera/left/image      gz::msgs::Image
//   /camera/right/image     gz::msgs::Image
//   /lidar/points/points    gz::msgs::PointCloudPacked -- NOT /lidar/points,
//                            which is gpu_lidar's raw LaserScan output;
//                            /points is a second, auto-published topic with
//                            the point cloud (matches foxglove_bridge.cpp's
//                            own subscription; field layout -- x/y/z as
//                            FLOAT32 at offsets 0/4/8, point_step=18 --
//                            confirmed directly via `gz topic -e`, not
//                            assumed, see lidar_projector.cpp)
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
//   /cone_detections        gz::msgs::Pose_V -- one Pose per LOCALIZED
//                            detection (has a lidar match; see TODO 3),
//                            name=class string ("blue"/"yellow"/"orange"/
//                            "large_orange"), position=(x,y,z) in the car
//                            (chassis) frame. No custom .proto: Pose_V
//                            already carries everything needed (a name
//                            string plus a position) without adding a
//                            protoc build step for one message type.
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
//   TODO 3: Subscribe to /lidar/points/points and project each detected
//            bounding box into the point cloud to get a 3-D cone position
//            in the car frame. DONE (see lidar_projector.hpp) -- projects
//            lidar points into the panorama's own cylindrical pixel space
//            (not a plain pinhole projection, since detections come from
//            the cylindrical panorama) using known, fixed lidar/camera
//            extrinsics from model.sdf, then matches each detection
//            bbox against the closest (by range) point that lands inside
//            it. Lidar runs at 10 Hz vs the cameras' 30 Hz and isn't
//            timestamp-synchronized against detections -- a detection can
//            legitimately get no match, or match a point cloud up to
//            ~1 lidar tick stale; acceptable for now, revisit if it causes
//            real position error once the car is actually moving.
//   TODO 4: Pack (x, y, color) for each detection and publish on
//            /cone_detections. DONE -- only detections with a lidar match
//            (TODO 3's pos != none) are published, since an unlocalized
//            detection has no (x,y) to pack in the first place.

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
// train.py anchors its `project` default to its own file location (see
// ml/train.py's DEFAULT_PROJECT), so training always writes here regardless
// of which directory the training command itself was invoked from. Not
// tracked in git (see .gitignore's ml/runs/ ignore -- nothing under there
// is tracked at all now), so this only exists on whichever machine
// actually ran training/exported the model -- doesn't need to, and isn't
// meant to, exist in a fresh clone.
//
// Two backends, selected at COMPILE time (see CMakeLists.txt's
// PERCEPTION_USE_TENSORRT block): TensorRT (GPU) when available -- Jetson
// builds only, since that's the only place TensorRT actually gets found --
// falling back to ConeDetector (ONNX Runtime, CPU) everywhere else (the
// Mac/x86 dev machine has no NVIDIA GPU at all). Same pattern as
// control.cpp's ActiveController/planning.cpp's kActive* function
// pointers: one line picks the implementation, nothing downstream needs
// to know or care which one is active. best.engine is NOT interchangeable
// with best.onnx -- it's tied to the exact GPU + TensorRT version it was
// built on (see cone_detector_trt.hpp), and has to be rebuilt via trtexec
// any time best.onnx changes.
#ifdef PERCEPTION_USE_TENSORRT
using ActiveDetector = ConeDetectorTrt;
constexpr const char *kModelPath = "ml/runs/detect/train/weights/best.engine";
#else
using ActiveDetector = ConeDetector;
constexpr const char *kModelPath = "ml/runs/detect/train/weights/best.onnx";
#endif
constexpr float kConfThreshold = 0.25f;

// A cone straddling the panorama's own left/right edge is only partially
// visible there (its far side is simply outside the panorama's FOV
// entirely, not occluded by anything -- see kPanoHFovRad above), and can
// trigger two overlapping candidate boxes that the model's own baked-in
// NMS doesn't merge -- confirmed directly: two near-identical boxes both
// touching x=kPanoWidth, IoU high enough that a person would call them
// the same detection, both surviving to Detect()'s output and both
// separately matching the same nearest lidar point (same 3D position,
// different pixel boxes) -- exactly the kind of duplicate a downstream
// SLAM landmark gate can't distinguish from two real nearby cones.
// Excluding any box that touches within this margin of either edge drops
// the ambiguous case rather than trying to de-duplicate it after the
// fact. Debug visualization (/camera/detections/image) still draws these
// -- this only excludes them from lidar localization/publishing, since
// that image's whole purpose is showing YOLO's raw, unfiltered output.
constexpr float kPanoEdgeMarginPx = 5.0f;

// Matches ml/prepare_data.py's CLASSES list exactly -- same order, since
// that's the order the model was trained to output class indices in.
constexpr const char *kClassNames[] = {"blue", "yellow", "orange", "large_orange"};

// RGB order (not BGR) -- matches this whole pipeline's own image
// convention (RGB_INT8 throughout, see FromImageMsg's comment), NOT
// OpenCV's usual BGR default. Same order/indexing as kClassNames.
const cv::Scalar kClassColorsRgb[] = {
    cv::Scalar(60, 60, 255),   // blue   (brightened -- pure (0,0,255) reads
                               // near-black against a dark cone in a small
                               // preview; same reasoning for the others)
    cv::Scalar(255, 220, 40),  // yellow
    cv::Scalar(255, 140, 30),  // orange
    cv::Scalar(255, 90, 0),    // large_orange (darker orange, distinguishable
                               // from orange at a glance)
};

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

    // Must exactly match CameraStitcher's own internal focal length (its
    // K.fx, computed from the same kCamWidth/kCamHFovRad) -- not exposed as
    // a public accessor there, so recomputed here from the same two
    // already-shared constants rather than an independent magic number.
    const double fPano = (kCamWidth / 2.0) / std::tan(kCamHFovRad / 2.0);
    LidarProjector lidarProjector(kPanoWidth, kPanoHeight, fPano);

    // kModelPath is git-ignored (see .gitignore) and only exists on a
    // machine that's actually run training (or, for the TensorRT .engine
    // path, also run trtexec against that machine's own best.onnx) -- fail
    // with a clear, actionable message rather than letting the underlying
    // ONNX Runtime/TensorRT exception surface if it's missing.
    std::unique_ptr<ActiveDetector> detector;
    try
    {
        detector = std::make_unique<ActiveDetector>(kModelPath, kConfThreshold);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load cone detection model at '" << kModelPath << "': " << e.what()
#ifdef PERCEPTION_USE_TENSORRT
                   << "\nDid you build an engine for THIS model with trtexec? (see"
                      " cone_detector_trt.hpp -- a .engine from a different best.onnx or a"
                      " different TensorRT version won't load.)\n";
#else
                   << "\nDid you run training (ml/train.py) on this machine, or copy the .onnx"
                      " file over from one that did?\n";
#endif
        return 1;
    }

    auto stitchedPub = node.Advertise<gz::msgs::Image>("/camera/stitched/image");
    auto detectionsPub = node.Advertise<gz::msgs::Pose_V>("/cone_detections");

    // Visualization only -- lets a human directly compare YOLO's raw boxes
    // against the panorama in Foxglove. Deliberately NOT a new gz-transport
    // message type (e.g. packing bbox+class+confidence into some Float_V
    // and having foxglove_bridge.cpp convert it to Foxglove's native
    // ImageAnnotations overlay, which would be the more "proper" way to do
    // this): reusing the exact same gz::msgs::Image + ToImageMsg() path
    // already proven for /camera/stitched/image is much lower-risk when it
    // can't be tested live before being needed (no annotation-schema or
    // additional-message-type details to get right sight-unseen). Also
    // draws EVERY detection, including ones with no lidar match -- shows
    // YOLO's actual accuracy independent of lidar fusion, unlike
    // /cone_detections (TODO 4), which only carries localized ones.
    auto detectionsImgPub = node.Advertise<gz::msgs::Image>("/camera/detections/image");

    // Per-cycle compute time (microseconds), covering the actual stitch +
    // detect + localize + publish work in tryStitchAndPublish below -- see
    // common/scoped_timer.hpp. NOT the whole callback: the "waiting for all
    // 3 camera frames to arrive" early-return path isn't real work.
    auto timingPub = node.Advertise<gz::msgs::UInt64>("/timing/perception");

    // Sub-stage breakdown of the same total /timing/perception covers --
    // added because the aggregate number alone (confirmed live at
    // 64-68ms/cycle, far above planning/control's 30-50us and even
    // localization's) couldn't say WHICH part was actually costing that:
    // tegrastats showed one CPU core pinned at 100% while GR3D_FREQ (GPU)
    // stayed mostly idle (0-39%), meaning the bottleneck is plausibly
    // CPU-bound work -- stitching, YOLO post-processing, or synchronous
    // stdout logging (kLogTimeUs's own comment elsewhere in this file
    // already flagged per-detection printing as a real latency
    // contributor) -- rather than GPU inference itself. These let that be
    // confirmed instead of guessed.
    auto stitchTimingPub = node.Advertise<gz::msgs::UInt64>("/timing/perception/stitch");
    auto detectTimingPub = node.Advertise<gz::msgs::UInt64>("/timing/perception/detect");
    // Localize() calls + annotation drawing + coneMsg construction --
    // everything in the per-detection loop EXCEPT the std::cout calls
    // (see /timing/perception/log below, measured separately specifically
    // to isolate that suspected cost rather than leave it folded in here).
    auto postprocessTimingPub = node.Advertise<gz::msgs::UInt64>("/timing/perception/postprocess");
    auto logTimingPub = node.Advertise<gz::msgs::UInt64>("/timing/perception/log");

    std::mutex mtx;
    cv::Mat latestFront, latestLeft, latestRight;
    StampKey stampFront{}, stampLeft{}, stampRight{};
    bool haveFront = false, haveLeft = false, haveRight = false;

    // Stitches only when all 3 buffered frames carry CLOSE-ENOUGH simulation
    // timestamps (see StampsClose/kStampToleranceSec) -- not just "whatever's
    // latest", and not exact equality either: measured empirically,
    // camera_front's stamp runs a consistent ~1 tick (1/30 s) ahead of
    // camera_left/right's, every frame (a fixed pipeline skew, not jitter),
    // so exact equality would reject every frame given that fixed skew.
    //
    // Only called from onCameraFront below, not from all 3 camera
    // callbacks -- calling it from all 3 was tried first and, on tracing
    // through the actual arrival pattern, doesn't do what it sounds like it
    // should: front arrives and checks against still-stale left/right,
    // passes (triple 1); moments later left arrives and checks against
    // fresh front + still-stale right, passes again (triple 2, genuinely
    // different from triple 1 since left changed); then right arrives and
    // passes a third time (triple 3, different again). All 3 are distinct
    // triples, not repeats, so a "have I already processed this exact
    // triple" dedup doesn't collapse any of them -- confirmed directly:
    // distinct but near-identical detection results appeared 2-3x between
    // console prints of a single new camera_front frame. Gating on front's
    // callback alone -- the camera confirmed (not assumed) to always arrive
    // first each cycle -- naturally rate-limits to once per cycle instead,
    // using whatever's currently buffered for left/right (consistently
    // ~1 tick "behind" per that same empirical data, comfortably inside
    // the tolerance).
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

        fsd::ScopedTimer timer([&timingPub](int64_t _us)
        {
            gz::msgs::UInt64 msg;
            msg.set_data(static_cast<uint64_t>(_us));
            timingPub.Publish(msg);
        });
        auto publishTimingUs = [](gz::transport::Node::Publisher &_pub, int64_t _us)
        {
            gz::msgs::UInt64 msg;
            msg.set_data(static_cast<uint64_t>(_us));
            _pub.Publish(msg);
        };

        const auto stitchStart = std::chrono::steady_clock::now();
        const cv::Mat stitched = stitcher.Stitch({latestFront, latestLeft, latestRight});
        publishTimingUs(stitchTimingPub, std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - stitchStart).count());
        stitchedPub.Publish(ToImageMsg(stitched));

        // Still prints every detection either way (matching the same
        // "print to confirm it's actually working" pattern as TODO 1's
        // camera-dimension print), but only LOCALIZED ones (a lidar match
        // exists, and the box doesn't touch the panorama edge -- see
        // kPanoEdgeMarginPx) get packed into coneMsg and published -- an
        // unlocalized detection has no (x,y) to give downstream consumers.
        // ActiveDetector::Detection, whichever backend is active -- ConeDetector
        // and ConeDetectorTrt have structurally-identical but DISTINCT Detection
        // types (no shared base), so this must deduce via auto rather than
        // naming either one explicitly.
        const auto detectStart = std::chrono::steady_clock::now();
        const auto detections = detector->Detect(stitched);
        publishTimingUs(detectTimingPub, std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - detectStart).count());

        gz::msgs::Pose_V coneMsg;
        cv::Mat annotated = stitched.clone();
        int64_t logUs = 0;
        const auto postprocessStart = std::chrono::steady_clock::now();
        for (const auto &d : detections)
        {
            const bool knownClass = d.classId >= 0 && static_cast<size_t>(d.classId) <
                                                            sizeof(kClassNames) / sizeof(kClassNames[0]);
            const char *className = knownClass ? kClassNames[d.classId] : "unknown";
            const cv::Scalar color = knownClass ? kClassColorsRgb[d.classId] : cv::Scalar(255, 255, 255);

            const bool touchesPanoEdge =
                d.x1 <= kPanoEdgeMarginPx || d.x2 >= static_cast<float>(kPanoWidth) - kPanoEdgeMarginPx;
            const auto pos = touchesPanoEdge
                                  ? std::nullopt
                                  : lidarProjector.Localize(d.x1, d.y1, d.x2, d.y2);

            // Logging isolated into its own contiguous, separately-timed
            // block (see /timing/perception/log's comment above) rather
            // than interleaved with the touchesPanoEdge/pos computation
            // above -- both to get a clean timed span and because
            // std::cout's own cost shouldn't be attributed to the
            // Localize()/annotation work measured below.
            const auto logStart = std::chrono::steady_clock::now();
            std::cout << "  cone: " << className << " conf=" << d.confidence << " bbox=["
                       << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "]";
            if (touchesPanoEdge)
            {
                std::cout << " pos=none (touches panorama edge, excluded)";
            }
            else if (pos)
            {
                std::cout << " pos=(" << pos->x << "," << pos->y << "," << pos->z
                           << ") range=" << pos->range << "m";
            }
            else
            {
                std::cout << " pos=none";
            }
            std::cout << '\n';
            logUs += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - logStart).count();

            if (!touchesPanoEdge && pos)
            {
                gz::msgs::Pose *p = coneMsg.add_pose();
                p->set_name(className);
                p->mutable_position()->set_x(pos->x);
                p->mutable_position()->set_y(pos->y);
                p->mutable_position()->set_z(pos->z);
            }

            cv::rectangle(annotated, cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                          cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)), color, 2);
            char label[64];
            std::snprintf(label, sizeof(label), "%s %.2f", className, d.confidence);
            cv::putText(annotated, label, cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1) - 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
        const int64_t postprocessTotalUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - postprocessStart).count();
        publishTimingUs(postprocessTimingPub, postprocessTotalUs - logUs);
        publishTimingUs(logTimingPub, logUs);

        detectionsPub.Publish(coneMsg);
        detectionsImgPub.Publish(ToImageMsg(annotated));
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

    // Neither of these calls tryStitchAndPublish() -- see the comment on
    // that lambda above for why only onCameraFront does. At startup, before
    // left/right have ever received a frame, this just means the very
    // first stitch waits for front's NEXT tick after both have arrived at
    // least once (at most ~33 ms) rather than firing immediately -- not a
    // correctness concern.
    std::function<void(const gz::msgs::Image &)> onCameraLeft =
        [&](const gz::msgs::Image &_msg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        latestLeft = FromImageMsg(_msg).clone();
        stampLeft = ToStampKey(_msg);
        haveLeft = true;
    };

    std::function<void(const gz::msgs::Image &)> onCameraRight =
        [&](const gz::msgs::Image &_msg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        latestRight = FromImageMsg(_msg).clone();
        stampRight = ToStampKey(_msg);
        haveRight = true;
    };

    // Locks the same mutex as the camera callbacks -- Localize() (called
    // from tryStitchAndPublish, itself called under that lock) reads
    // lidarProjector's cached points while this writes them; without
    // sharing the lock the two could race on the same std::vector.
    std::function<void(const gz::msgs::PointCloudPacked &)> onLidar =
        [&](const gz::msgs::PointCloudPacked &_msg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        lidarProjector.SetPointCloud(_msg);
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
    if (!node.Subscribe("/lidar/points/points", onLidar))
    {
        std::cerr << "Failed to subscribe to /lidar/points/points\n";
        return 1;
    }

    gz::transport::waitForShutdown();
    return 0;
}
