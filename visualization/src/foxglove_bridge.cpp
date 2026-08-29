#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include <cstdio>

#include <gz/transport/Node.hh>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/pointcloud_packed.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/scene.pb.h>
#include <gz/msgs/uint64.pb.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <foxglove/channel.hpp>
#include <foxglove/error.hpp>
#include <foxglove/messages.hpp>
#include <foxglove/schema.hpp>
#include <foxglove/websocket.hpp>

#ifdef FOXGLOVE_BRIDGE_HAS_TEGRASTATS
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <cstdio>
#include <regex>
#include <thread>
#endif

// Foxglove bridge process
// -----------------------
// Subscribes to gz-transport topics and republishes them through a
// foxglove::WebSocketServer so the Foxglove desktop app can visualize them.
//
// Usage:
//   foxglove_bridge <world_name>
// <world_name> must match the <world name="..."> in whichever .sdf is
// currently running (e.g. "track", "trackdrive") -- gz-transport namespaces
// every world topic under /world/<world_name>/..., so this is the one place
// that has to agree with the launcher script and the .sdf file. Passed as a
// runtime argument rather than hardcoded, so changing which world is running
// doesn't require touching this file or rebuilding.
//
// Connect Foxglove to:  ws://<docker-host>:8765
//
// Transform tree (/tf, Foxglove FrameTransforms):
//   world -> fsd_car                         (dynamic, from Pose_V every tick
//                                              -- GROUND TRUTH, straight from
//                                              Gazebo's own physics state)
//   world -> fsd_car_estimated               (dynamic, from localization's
//                                              /estimated_pose -- the EKF's
//                                              fused ESTIMATE, not truth;
//                                              see /estimated_vehicle below)
//   fsd_car -> {camera_front, camera_left, camera_right, os1_128}
//     (static, fetched once at startup via the /world/<world_name>/scene/info
//      *service* -- not the topic of the same name, which is a one-shot
//      broadcast a bridge started even slightly late would simply miss.
//      Each sensor's pose-relative-to-model is composed from its parent
//      link's pose and its own pose within that link, both taken straight
//      from Gazebo's resolved model description -- not duplicated by hand.)
// Everything else (cameras, lidar) is published in its own sensor frame
// with no baked-in world pose; Foxglove resolves placement by walking /tf.
//
// Currently forwards:
//   /world/<world_name>/pose/info (gz.msgs.Pose_V), driving:
//     -> /vehicle_pose (Foxglove PoseInFrame)   — the "fsd_car" entry
//     -> /scene        (Foxglove SceneUpdate)   — every cone_{blue,yellow,
//                       orange}_* entry as a cone (a CylinderPrimitive with
//                       top_scale=0) plus its stripe band(s) as additional
//                       tapered CylinderPrimitives (AppendStripeCylinders,
//                       spec table shared with ConeSpecForName -- see
//                       simulation/models/cone_*/model.sdf for the
//                       center/span numbers this mirrors), so the track is
//                       viewable in Foxglove's 3D panel without needing
//                       Gazebo's own GUI. Deliberately just the (static)
//                       track -- the car has its own /vehicle topic below,
//                       so either can be shown/hidden independently in
//                       Foxglove.
//     -> /tf (Foxglove FrameTransforms)         — world->fsd_car
//   fsd_car's wheel link poses (fetched once from the same scene service
//   used for sensor frames below, not hand-duplicated from model.sdf)
//     -> /vehicle (Foxglove SceneUpdate), frame_id "fsd_car" -- a chassis
//        CubePrimitive plus 4 wheel CylinderPrimitives, republished every
//        Pose_V tick alongside /scene/vehicle_pose. Placement in world
//        space comes entirely from /tf's dynamic world->fsd_car transform,
//        not from any pose math here. Wheels don't animate steering or
//        spin: spin is invisible on a symmetric cylinder anyway, and
//        steering would need parsing live joint_state, which nothing here
//        needs yet.
//   /camera/{front,left,right}/image (gz.msgs.Image)
//     -> /camera/{front,left,right} (Foxglove CompressedImage, JPEG q80 --
//        see makeImageSubscriber's own comment for why compressed, not
//        RawImage), frame_id matching the sensor's own name
//        (camera_front/camera_left/camera_right)
//   /camera/stitched/image (gz.msgs.Image, published by perception --
//     the 3 raw cameras merged into one cylindrical panorama)
//     -> /camera/stitched (Foxglove CompressedImage, JPEG q80), frame_id
//        "camera_front"
//   /camera/detections/image (gz.msgs.Image, published by perception --
//     the same panorama with every YOLO detection's bbox + class + confidence
//     drawn directly into the pixels, for visually comparing detection
//     accuracy in Foxglove; includes detections with no lidar match too,
//     unlike /cone_detections, which only carries localized ones)
//     -> /camera/detections (Foxglove CompressedImage, JPEG q80), frame_id
//        "camera_front"
//   /lidar/points/points (gz.msgs.PointCloudPacked -- gpu_lidar's derived
//   point-cloud topic; /lidar/points itself is a raw LaserScan, not this)
//     -> /lidar (Foxglove PointCloud), frame_id "os1_128"
//   /cone_detections (gz.msgs.Pose_V, published by perception -- one Pose
//     per LOCALIZED detection this frame, position in the car/chassis
//     frame, name() set to the class name)
//     -> /cone_detections (Foxglove SceneUpdate), a CylinderPrimitive per
//        cone colored/sized by class, frame_id "fsd_car" so Foxglove places
//        them via /tf without any transform math here. Entity id
//        "cone_detections" is replaced wholesale every publish (including
//        with zero cylinders when nothing is localized), so it always shows
//        only the current frame's detections, not an accumulating trail.
//   /estimated_pose (gz.msgs.Pose, published by localization -- the EKF's
//     fused [x, y, yaw] estimate; see localization.cpp)
//     -> /tf (world -> fsd_car_estimated)     -- z is fixed at the chassis's
//        known ride height, NOT something the 2D-only EKF actually
//        estimates; rendering at the filter's real z=0 would look like the
//        car sank into the ground, an artifact of what's unmodeled, not of
//        estimate quality.
//     -> /estimated_vehicle (Foxglove SceneUpdate), same chassis+wheel
//        shape as /vehicle (BuildVehicleEntity(), shared with it) but
//        translucent green so it reads as a ghost overlay for comparing
//        against /vehicle (ground truth) rather than obscuring it.
//   /estimated_landmarks (gz.msgs.Pose_V, published by localization -- the
//     SLAM map's current landmark estimates, WORLD frame, name() = class
//     name; see ekf.hpp)
//     -> /estimated_landmarks (Foxglove SceneUpdate), a translucent cone
//        per landmark (reuses DetectionConeSpec for size/color), frame_id
//        "world" -- overlay this against /scene's opaque ground-truth
//        cones to watch the SLAM map converge as the car drives.
//   /planned_path (gz.msgs.Pose_V, published by planning -- ordered
//     centerline waypoints, body frame, nearest-ahead first)
//     -> /planned_path (Foxglove SceneUpdate), a single LINE_STRIP through
//        the waypoints, frame_id "fsd_car" (same reasoning as
//        /cone_detections: it's already body-frame, so /tf places it with
//        no transform math here either). Replaced wholesale every publish,
//        same as /cone_detections, so it never shows a stale path.
//
// Jetson system stats (Jetson-only, guarded by FOXGLOVE_BRIDGE_HAS_TEGRASTATS
// -- see CMakeLists.txt's find_program(tegrastats) check; absent entirely on
// the Mac/x86 dev build):
//   `tegrastats` doesn't originate from Gazebo/gz-transport at all -- it's a
//   host-machine monitoring tool with no relationship to the sim, so unlike
//   every other topic in this file it isn't a gz-transport subscription.
//   Spawned ONCE as a long-lived subprocess (not re-spawned per sample --
//   its own --interval flag controls sampling rate) on a background thread,
//   parsed line-by-line, and logged to:
//     -> /jetson_stats (custom JSON schema via foxglove::RawChannel -- none
//        of the SDK's built-in typed channels fit a flat stats snapshot).
//        Always includes a "raw" field with the unparsed tegrastats line
//        alongside the best-effort-parsed fields, since tegrastats' text
//        format isn't a stable/versioned API -- see ParseTegrastatsLineToJson.
//
// Per-stage timing:
//   /timing/{perception,localization,planning,control} (gz.msgs.UInt64,
//     published by each of those 4 processes -- see common/scoped_timer.hpp
//     -- microseconds spent on that process's own last unit of real work)
//     -> /timing (Foxglove RawChannel, "json" encoding + a small jsonschema
//        describing the 4 fields) -- none of the SDK's built-in typed
//        channels used elsewhere in this file (PoseInFrame, SceneUpdate,
//        etc.) fit 4 independent named scalars, so this uses RawChannel
//        directly instead, the SDK's supported escape hatch for a custom
//        data shape. Combined into ONE topic rather than 4 separate ones:
//        the 4 stages update at different, unrelated rates, so this
//        latest-value-per-field cache is republished in full every time ANY
//        one of the 4 ticks (same pattern as /tf above, which similarly
//        merges multiple independently-arriving sources into one output).

namespace
{
constexpr const char *kCarModelName = "fsd_car";
constexpr const char *kFrameId = "world";

foxglove::messages::Timestamp Now()
{
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(now);
    const auto nsecs = std::chrono::duration_cast<std::chrono::nanoseconds>(now - secs);
    return foxglove::messages::Timestamp{
        static_cast<uint32_t>(secs.count()),
        static_cast<uint32_t>(nsecs.count())};
}

foxglove::messages::Pose ToFoxglovePose(const gz::msgs::Pose &_pose)
{
    return foxglove::messages::Pose{
        foxglove::messages::Vector3{
            _pose.position().x(), _pose.position().y(), _pose.position().z()},
        foxglove::messages::Quaternion{
            _pose.orientation().x(), _pose.orientation().y(),
            _pose.orientation().z(), _pose.orientation().w()}};
}

// Only RGB_INT8 is handled -- the only format any publisher in this
// pipeline actually emits (see perception.cpp's ToImageMsg and every
// <camera> sensor's default in model.sdf, both RGB_INT8), same "unsupported
// -- caller should skip the frame" policy the RawImage path this replaced
// used for every other gz::msgs::PixelFormatType. Unlike that RawImage
// path, JPEG-encoding actually depends on getting the channel order right
// (cv::imencode always expects BGR, matching its own imshow/imwrite
// convention), not just a label Foxglove reads back -- so this can't stay
// a format-name lookup once compression is involved; a wrong-order guess
// here would silently swap red and blue in every frame instead of just
// mislabeling metadata.
bool IsSupportedImageFormat(gz::msgs::PixelFormatType _fmt)
{
    return _fmt == gz::msgs::PixelFormatType::RGB_INT8;
}

foxglove::messages::PackedElementField::NumericType ToFoxgloveFieldType(
    gz::msgs::PointCloudPacked::Field::DataType _t)
{
    using GzT = gz::msgs::PointCloudPacked::Field;
    using FgT = foxglove::messages::PackedElementField::NumericType;
    switch (_t)
    {
        case GzT::INT8:    return FgT::INT8;
        case GzT::UINT8:   return FgT::UINT8;
        case GzT::INT16:   return FgT::INT16;
        case GzT::UINT16:  return FgT::UINT16;
        case GzT::INT32:   return FgT::INT32;
        case GzT::UINT32:  return FgT::UINT32;
        case GzT::FLOAT32: return FgT::FLOAT32;
        case GzT::FLOAT64: return FgT::FLOAT64;
        default:           return FgT::UNKNOWN;
    }
}

// v' = v + w*(2*(u x v)) + u x (2*(u x v)) -- standard quaternion-rotate-vector
// formula, avoids building a full rotation matrix.
foxglove::messages::Vector3 RotateVector(
    const foxglove::messages::Quaternion &_q, const foxglove::messages::Vector3 &_v)
{
    const double ux = _q.x, uy = _q.y, uz = _q.z, w = _q.w;
    const double tx = 2 * (uy * _v.z - uz * _v.y);
    const double ty = 2 * (uz * _v.x - ux * _v.z);
    const double tz = 2 * (ux * _v.y - uy * _v.x);
    return foxglove::messages::Vector3{
        _v.x + w * tx + (uy * tz - uz * ty),
        _v.y + w * ty + (uz * tx - ux * tz),
        _v.z + w * tz + (ux * ty - uy * tx)};
}

// Hamilton product: _a applied first, then _b (i.e. result = _a * _b).
foxglove::messages::Quaternion MultiplyQuaternion(
    const foxglove::messages::Quaternion &_a, const foxglove::messages::Quaternion &_b)
{
    return foxglove::messages::Quaternion{
        _a.w * _b.x + _a.x * _b.w + _a.y * _b.z - _a.z * _b.y,
        _a.w * _b.y - _a.x * _b.z + _a.y * _b.w + _a.z * _b.x,
        _a.w * _b.z + _a.x * _b.y - _a.y * _b.x + _a.z * _b.w,
        _a.w * _b.w - _a.x * _b.x - _a.y * _b.y - _a.z * _b.z};
}

// Composes _childInParent (a pose expressed relative to _parent) into
// _parent's own frame -- i.e. "where is the child, expressed in whatever
// frame _parent itself is expressed in".
foxglove::messages::Pose ComposePose(
    const foxglove::messages::Pose &_parent, const foxglove::messages::Pose &_childInParent)
{
    const auto &pp = *_parent.position;
    const auto &po = *_parent.orientation;
    const auto &cp = *_childInParent.position;
    const auto &co = *_childInParent.orientation;

    const auto rotated = RotateVector(po, cp);
    foxglove::messages::Pose result;
    result.position = foxglove::messages::Vector3{
        pp.x + rotated.x, pp.y + rotated.y, pp.z + rotated.z};
    result.orientation = MultiplyQuaternion(po, co);
    return result;
}

// One stripe band, as a fraction of the cone's total height from its base --
// same center_frac/span_frac numbers as the comments in each
// simulation/models/cone_*/model.sdf (e.g. blue: center 0.5, span 0.22).
struct StripeSpec
{
    double center_frac;
    double span_frac;
    foxglove::messages::Color color;
};

struct ConeSpec
{
    double radius;
    double length;
    foxglove::messages::Color color;
    std::array<StripeSpec, 2> stripes{};
    int stripe_count = 0;
};

// Dimensions/colors/stripes must match simulation/models/cone_{blue,yellow,orange}/model.sdf.
bool ConeSpecForName(const std::string &_name, ConeSpec *_spec)
{
    if (_name.rfind("cone_blue", 0) == 0)
    {
        *_spec = ConeSpec{0.115, 0.325, foxglove::messages::Color{0, 0, 0.8, 1}};
        _spec->stripes[0] = StripeSpec{0.5, 0.22, foxglove::messages::Color{1, 1, 1, 1}};
        _spec->stripe_count = 1;
        return true;
    }
    if (_name.rfind("cone_yellow", 0) == 0)
    {
        *_spec = ConeSpec{0.115, 0.325, foxglove::messages::Color{0.9, 0.9, 0, 1}};
        _spec->stripes[0] = StripeSpec{0.5, 0.30, foxglove::messages::Color{0, 0, 0, 1}};
        _spec->stripe_count = 1;
        return true;
    }
    if (_name.rfind("cone_orange", 0) == 0)
    {
        // #CC6600, matching simulation/models/cone_orange/model.sdf.
        *_spec = ConeSpec{0.1425, 0.505, foxglove::messages::Color{0.8, 0.4, 0, 1}};
        _spec->stripes[0] = StripeSpec{0.34, 0.16, foxglove::messages::Color{1, 1, 1, 1}};
        _spec->stripes[1] = StripeSpec{0.64, 0.16, foxglove::messages::Color{1, 1, 1, 1}};
        _spec->stripe_count = 2;
        return true;
    }
    return false;
}

// Appends _spec's stripe band(s) as additional CylinderPrimitives around
// _conePose, so rendered cones match the banding in
// simulation/models/cone_{blue,yellow,orange}/model.sdf instead of coming
// out as a flat, unbanded solid color. Each band reuses the parent cone's
// own linear taper -- radius(h) = base_radius * (1 - h/length), the same
// formula model.sdf's own stripe meshes are built from -- via
// bottom_scale/top_scale, rather than a constant-radius cylinder, so it
// stays flush with the parent cone's silhouette instead of flaring past it
// (see cone_blue/model.sdf's stripe comment for why that was rejected).
void AppendStripeCylinders(
    const ConeSpec &_spec, const foxglove::messages::Pose &_conePose,
    std::vector<foxglove::messages::CylinderPrimitive> *_cylinders)
{
    for (int i = 0; i < _spec.stripe_count; ++i)
    {
        const auto &stripe = _spec.stripes[i];
        const double hBottom = (stripe.center_frac - stripe.span_frac / 2.0) * _spec.length;
        const double hTop = (stripe.center_frac + stripe.span_frac / 2.0) * _spec.length;
        const double radiusBottom = _spec.radius * (1.0 - hBottom / _spec.length);
        const double radiusTop = _spec.radius * (1.0 - hTop / _spec.length);

        // Offset from the cone's own center (its Pose's origin, per the
        // standard SDF primitive convention -- base at -length/2, apex at
        // +length/2), composed through the cone's actual world pose rather
        // than added directly, so this still lands correctly if the cone
        // isn't perfectly upright.
        foxglove::messages::Pose localOffset;
        localOffset.position = foxglove::messages::Vector3{
            0, 0, (hBottom + hTop) / 2.0 - _spec.length / 2.0};
        localOffset.orientation = foxglove::messages::Quaternion{0, 0, 0, 1};

        foxglove::messages::CylinderPrimitive band;
        band.pose = ComposePose(_conePose, localOffset);
        band.size = foxglove::messages::Vector3{radiusBottom * 2, radiusBottom * 2, hTop - hBottom};
        band.bottom_scale = 1.0;
        band.top_scale = radiusTop / radiusBottom;
        band.color = stripe.color;
        _cylinders->push_back(band);
    }
}

// Same visual spec table as ConeSpecForName above, but keyed on perception's
// raw YOLO class names ("blue"/"yellow"/"orange"/"large_orange") rather than
// ground-truth model-instance names ("cone_blue_12") -- the two naming
// schemes don't overlap, so this can't just reuse that lookup. The sim only
// ships one orange cone model (already sized as the big 505mm cone), so
// "orange" and "large_orange" both map to it; there's no ground-truth model
// for a small orange cone to distinguish them against.
bool DetectionConeSpec(const std::string &_className, ConeSpec *_spec)
{
    if (_className == "blue")
    {
        *_spec = ConeSpec{0.115, 0.325, foxglove::messages::Color{0, 0, 0.8, 1}};
        return true;
    }
    if (_className == "yellow")
    {
        *_spec = ConeSpec{0.115, 0.325, foxglove::messages::Color{0.9, 0.9, 0, 1}};
        return true;
    }
    if (_className == "orange" || _className == "large_orange")
    {
        // #CC6600, matching simulation/models/cone_orange/model.sdf.
        *_spec = ConeSpec{0.1425, 0.505, foxglove::messages::Color{0.8, 0.4, 0, 1}};
        return true;
    }
    return false;
}

// Fetches the static sensor->model_root offsets straight from Gazebo's
// resolved model description (link.pose() composed with sensor.pose()),
// rather than duplicating the SDF's numbers by hand a second time here.
std::unordered_map<std::string, foxglove::messages::Pose> FetchStaticSensorPoses(
    gz::transport::Node &_node, const std::string &_sceneService)
{
    std::unordered_map<std::string, foxglove::messages::Pose> poses;

    gz::msgs::Scene sceneMsg;
    bool result = false;
    const bool ok = _node.Request(_sceneService, 5000u, sceneMsg, result);
    if (!ok || !result)
    {
        std::cerr << "Warning: failed to fetch " << _sceneService
                  << " -- sensor frames will be missing from /tf\n";
        return poses;
    }

    for (const auto &model : sceneMsg.model())
    {
        if (model.name() != kCarModelName)
        {
            continue;
        }
        for (const auto &link : model.link())
        {
            const auto linkPose = ToFoxglovePose(link.pose());
            for (const auto &sensor : link.sensor())
            {
                poses[sensor.name()] = ComposePose(linkPose, ToFoxglovePose(sensor.pose()));
            }
        }
    }
    return poses;
}

constexpr const char *kWheelLinkNames[] = {
    "front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"};
constexpr double kWheelRadius = 0.26;  // must match fsd_car/model.sdf
constexpr double kWheelLength = 0.18;
// A bare cylinder's symmetry axis defaults to Z (vertical); this is the
// same 90deg-about-X roll applied to the wheel visual/collision geometry in
// model.sdf, needed here too since the fetched link poses below are the
// LINK frames (identity orientation, deliberately -- see model.sdf), not
// the rotated geometry within them.
const foxglove::messages::Quaternion kWheelRollQuat{0.70710678118654752, 0, 0, 0.70710678118654752};

// Fetches each wheel link's pose relative to the model root, straight from
// Gazebo's resolved model description (same service and pattern as
// FetchStaticSensorPoses above), rather than duplicating model.sdf's wheel
// offsets by hand a second time here.
std::unordered_map<std::string, foxglove::messages::Pose> FetchStaticWheelPoses(
    gz::transport::Node &_node, const std::string &_sceneService)
{
    std::unordered_map<std::string, foxglove::messages::Pose> poses;

    gz::msgs::Scene sceneMsg;
    bool result = false;
    const bool ok = _node.Request(_sceneService, 5000u, sceneMsg, result);
    if (!ok || !result)
    {
        std::cerr << "Warning: failed to fetch " << _sceneService
                  << " -- /vehicle will be missing its wheels\n";
        return poses;
    }

    for (const auto &model : sceneMsg.model())
    {
        if (model.name() != kCarModelName)
        {
            continue;
        }
        for (const auto &link : model.link())
        {
            for (const char *wheelName : kWheelLinkNames)
            {
                if (link.name() == wheelName)
                {
                    poses[link.name()] = ToFoxglovePose(link.pose());
                }
            }
        }
    }
    return poses;
}

// Shared between /vehicle (ground truth, called from onPoseV) and
// /estimated_vehicle (the EKF's estimate, called from onEstimatedPose) --
// same shape either way, just a different frame_id/id/colors so Foxglove
// can place and distinguish the two independently.
foxglove::messages::SceneEntity BuildVehicleEntity(
    const std::string &_frameId, const std::string &_id,
    const foxglove::messages::Color &_chassisColor,
    const foxglove::messages::Color &_wheelColor,
    const std::unordered_map<std::string, foxglove::messages::Pose> &_wheelStaticPoses)
{
    foxglove::messages::SceneEntity entity;
    entity.timestamp = Now();
    entity.frame_id = _frameId;
    entity.id = _id;

    foxglove::messages::CubePrimitive chassis;
    chassis.pose = foxglove::messages::Pose{
        foxglove::messages::Vector3{0, 0, 0},
        foxglove::messages::Quaternion{0, 0, 0, 1}};
    chassis.size = foxglove::messages::Vector3{1.8, 0.84, 0.3};  // matches model.sdf chassis box
    chassis.color = _chassisColor;
    entity.cubes.push_back(chassis);

    for (const auto &entry : _wheelStaticPoses)
    {
        foxglove::messages::CylinderPrimitive wheel;
        wheel.pose = foxglove::messages::Pose{entry.second.position, kWheelRollQuat};
        wheel.size = foxglove::messages::Vector3{
            kWheelRadius * 2, kWheelRadius * 2, kWheelLength};
        wheel.bottom_scale = 1.0;  // full cylinder, not a cone
        wheel.top_scale = 1.0;
        wheel.color = _wheelColor;
        entity.cylinders.push_back(wheel);
    }
    return entity;
}

#ifdef FOXGLOVE_BRIDGE_HAS_TEGRASTATS
// JSON Schema for /jetson_stats -- see ParseTegrastatsLineToJson for what
// actually populates each field.
constexpr const char *kJetsonStatsSchemaJson = R"({
  "type": "object",
  "properties": {
    "ram_used_mb": {"type": "number"},
    "ram_total_mb": {"type": "number"},
    "cpu_core_pct": {"type": "array", "items": {"type": "number"}},
    "gpu_pct": {"type": "number"},
    "temps_c": {"type": "object"},
    "power_mw": {"type": "object"}
  }
})";

// Escapes the characters tegrastats' own output can actually contain and
// that would otherwise break a hand-built JSON string (quotes,
// backslashes). Not a general-purpose JSON escaper -- tegrastats never
// emits control characters or non-ASCII, so this covers everything that
// can actually occur here.
std::string JsonEscape(const std::string &_s)
{
    std::string out;
    out.reserve(_s.size());
    for (char c : _s)
    {
        if (c == '"' || c == '\\')
        {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

// Best-effort parse of a single tegrastats output line into a flat JSON
// object. tegrastats' text format is NOT a stable, versioned API -- these
// regexes were written against the documented/typical L4T format (RAM
// X/YMB, CPU [n%@freq,...], GR3D_FREQ n%, name@temperatureC, VDD_/VIN_RAIL
// nmW/mW) and verified against a real sample from this device's actual
// tegrastats output -- if a future L4T release changes the format, a field
// silently going missing here is the symptom to watch for.
std::string ParseTegrastatsLineToJson(const std::string &_line)
{
    std::ostringstream json;
    json << "{";

    std::smatch m;
    if (std::regex_search(_line, m, std::regex(R"(RAM (\d+)/(\d+)MB)")))
    {
        json << "\"ram_used_mb\":" << m[1].str() << ",\"ram_total_mb\":" << m[2].str() << ",";
    }

    if (std::regex_search(_line, m, std::regex(R"(CPU \[([^\]]*)\])")))
    {
        const std::string coreList = m[1].str();
        json << "\"cpu_core_pct\":[";
        const std::regex corePct(R"((\d+)%@)");
        bool first = true;
        for (auto it = std::sregex_iterator(coreList.begin(), coreList.end(), corePct);
             it != std::sregex_iterator(); ++it)
        {
            if (!first)
            {
                json << ",";
            }
            json << (*it)[1].str();
            first = false;
        }
        json << "],";
    }

    if (std::regex_search(_line, m, std::regex(R"(GR3D_FREQ (\d+)%)")))
    {
        json << "\"gpu_pct\":" << m[1].str() << ",";
    }

    json << "\"temps_c\":{";
    {
        const std::regex tempPattern(R"((\w+)@(-?[\d.]+)C)");
        bool first = true;
        for (auto it = std::sregex_iterator(_line.begin(), _line.end(), tempPattern);
             it != std::sregex_iterator(); ++it)
        {
            if (!first)
            {
                json << ",";
            }
            json << "\"" << JsonEscape((*it)[1].str()) << "\":" << (*it)[2].str();
            first = false;
        }
    }
    json << "},";

    json << "\"power_mw\":{";
    {
        // Matches VDD_* rails (e.g. VDD_GPU_SOC) and VIN_* rails (e.g.
        // VIN_SYS_5V0) -- confirmed against a real tegrastats line on the
        // actual Jetson, which includes both prefixes, not just VDD_.
        // Captures the full rail name (prefix included) as the JSON key,
        // since VDD_ vs VIN_ is meaningful (voltage-domain vs. input rail).
        const std::regex powerPattern(R"(((?:VDD|VIN)_\w+) (\d+)mW/\d+mW)");
        bool first = true;
        for (auto it = std::sregex_iterator(_line.begin(), _line.end(), powerPattern);
             it != std::sregex_iterator(); ++it)
        {
            if (!first)
            {
                json << ",";
            }
            json << "\"" << JsonEscape((*it)[1].str()) << "\":" << (*it)[2].str();
            first = false;
        }
    }
    // Two closing braces: power_mw's own object, then the outer object
    // opened by this function's very first `json << "{"`. BUG FIX: this
    // was a single "}" (closing only power_mw), silently truncating every
    // published message -- confirmed directly via Foxglove's own parser
    // error ("Expected ',' or '}' after property value") once a client
    // actually validated the JSON instead of just eyeballing the printed
    // text, which happens to look complete at a glance since the missing
    // brace is invisible unless you count.
    json << "}}";
    return json.str();
}

// Runs tegrastats as a single long-lived child process (its own --interval
// flag controls sampling rate -- this does NOT re-spawn it per sample) and
// feeds each line it prints to _onLine as it arrives, blocking until either
// the pipe closes or _stop becomes true. PR_SET_PDEATHSIG makes the child
// die if this process dies first (e.g. Ctrl+C via dev_sim.sh's cleanup()),
// rather than surviving as an orphan -- plain popen()/pclose() does NOT
// guarantee that, since pclose() just waits for the child instead of
// killing it.
//
// Shutdown note: _stop is only checked between lines, so if tegrastats is
// mid-interval when _stop is set, this can block for up to one more
// interval before returning -- acceptable for a monitoring sidecar with a
// multi-second interval, not something this makes any effort to preempt
// immediately.
void RunTegrastats(int _intervalMs, const std::atomic<bool> &_stop,
                    const std::function<void(const std::string &)> &_onLine)
{
    int pipeFds[2];
    if (pipe(pipeFds) != 0)
    {
        std::cerr << "tegrastats: pipe() failed -- /jetson_stats will not be published\n";
        return;
    }

    const pid_t pid = fork();
    if (pid < 0)
    {
        std::cerr << "tegrastats: fork() failed -- /jetson_stats will not be published\n";
        close(pipeFds[0]);
        close(pipeFds[1]);
        return;
    }

    if (pid == 0)
    {
        // Child: exec tegrastats with stdout wired to the pipe.
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        close(pipeFds[0]);
        dup2(pipeFds[1], STDOUT_FILENO);
        close(pipeFds[1]);
        const std::string intervalArg = std::to_string(_intervalMs);
        execlp("tegrastats", "tegrastats", "--interval", intervalArg.c_str(),
               static_cast<char *>(nullptr));
        _exit(127);  // execlp only returns on failure
    }

    // Parent
    close(pipeFds[1]);
    FILE *stream = fdopen(pipeFds[0], "r");
    if (!stream)
    {
        std::cerr << "tegrastats: fdopen() failed -- /jetson_stats will not be published\n";
        close(pipeFds[0]);
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
        return;
    }

    char buf[4096];
    while (!_stop.load() && fgets(buf, sizeof(buf), stream))
    {
        std::string line(buf);
        if (!line.empty() && line.back() == '\n')
        {
            line.pop_back();
        }
        if (!line.empty())
        {
            _onLine(line);
        }
    }

    fclose(stream);
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
}
#endif  // FOXGLOVE_BRIDGE_HAS_TEGRASTATS
}  // namespace

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <world_name>\n"
                   << "  <world_name> must match the <world name=\"...\"> of the\n"
                   << "  currently running .sdf, e.g.: " << argv[0] << " trackdrive\n";
        return 1;
    }
    const std::string worldName = argv[1];
    const std::string poseTopic = "/world/" + worldName + "/pose/info";
    const std::string sceneService = "/world/" + worldName + "/scene/info";

    foxglove::WebSocketServerOptions ws_options;
    ws_options.host = "0.0.0.0";  // 127.0.0.1 wouldn't be reachable through -p 8765:8765
    ws_options.port = 8765;

    auto server_result = foxglove::WebSocketServer::create(std::move(ws_options));
    if (!server_result.has_value())
    {
        std::cerr << "Failed to create Foxglove server: "
                  << foxglove::strerror(server_result.error()) << '\n';
        return 1;
    }
    auto server = std::move(server_result.value());

    auto pose_channel_result = foxglove::messages::PoseInFrameChannel::create("/vehicle_pose");
    if (!pose_channel_result.has_value())
    {
        std::cerr << "Failed to create /vehicle_pose channel: "
                  << foxglove::strerror(pose_channel_result.error()) << '\n';
        return 1;
    }
    auto pose_channel = std::move(pose_channel_result.value());

    auto tf_channel_result = foxglove::messages::FrameTransformsChannel::create("/tf");
    if (!tf_channel_result.has_value())
    {
        std::cerr << "Failed to create /tf channel: "
                  << foxglove::strerror(tf_channel_result.error()) << '\n';
        return 1;
    }
    auto tf_channel = std::move(tf_channel_result.value());

    gz::transport::Node node;

    const auto sensorStaticPoses = FetchStaticSensorPoses(node, sceneService);
    std::cout << "Fetched " << sensorStaticPoses.size()
              << " static sensor transform(s) from " << sceneService << '\n';

    const auto wheelStaticPoses = FetchStaticWheelPoses(node, sceneService);
    std::cout << "Fetched " << wheelStaticPoses.size()
              << " static wheel pose(s) from " << sceneService << '\n';

    auto camFrontChannel      = foxglove::messages::CompressedImageChannel::create("/camera/front").value();
    auto camLeftChannel       = foxglove::messages::CompressedImageChannel::create("/camera/left").value();
    auto camRightChannel      = foxglove::messages::CompressedImageChannel::create("/camera/right").value();
    auto camStitchedChannel   = foxglove::messages::CompressedImageChannel::create("/camera/stitched").value();
    auto camDetectionsChannel = foxglove::messages::CompressedImageChannel::create("/camera/detections").value();
    auto lidarChannel         = foxglove::messages::PointCloudChannel::create("/lidar").value();

    // JPEG-encoded (was RawImage, i.e. raw RGB bytes with no compression)
    // -- confirmed directly as the cause of a real user-reported slowdown
    // (2026-08-23): watching Foxglove felt slow even though the underlying
    // gz-transport topics measured at the expected ~9Hz on the Jetson
    // itself, no network involved. 5 raw 720x540x3 RGB channels at ~9Hz
    // each is ~52 MB/s of uncompressed image data pushed over the
    // websocket to whatever's actually viewing Foxglove (typically a
    // laptop over WiFi to the car) -- comfortably beyond what that link
    // can sustain, so frames queue up and playback lags even though
    // nothing in the pipeline itself is behind. JPEG quality 80 cuts each
    // frame by roughly 15-20x for a natural/camera image, bringing total
    // throughput down to single-digit MB/s. This is a viewer-bandwidth
    // fix only -- doesn't touch /camera/*/image or /camera/stitched/image
    // themselves (still full-resolution, uncompressed, at whatever rate
    // perception publishes -- see perception.cpp's kDebugImageEveryN),
    // since those still need pixel-exact data for actual detection input,
    // not just human viewing.
    auto makeImageSubscriber = [](foxglove::messages::CompressedImageChannel &_channel,
                                   std::string _frameId)
    {
        return std::function<void(const gz::msgs::Image &)>(
            [&_channel, _frameId](const gz::msgs::Image &_msg)
            {
                if (!IsSupportedImageFormat(_msg.pixel_format_type()))
                {
                    return;  // unsupported format, skip rather than send garbage
                }

                const cv::Mat rgb(static_cast<int>(_msg.height()), static_cast<int>(_msg.width()),
                                   CV_8UC3, const_cast<char *>(_msg.data().data()),
                                   static_cast<size_t>(_msg.step()));
                cv::Mat bgr;
                cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
                std::vector<uchar> jpegBuf;
                cv::imencode(".jpg", bgr, jpegBuf, {cv::IMWRITE_JPEG_QUALITY, 80});

                foxglove::messages::CompressedImage img;
                img.timestamp = Now();
                img.frame_id = _frameId;
                img.format = "jpeg";
                const auto *bytes = reinterpret_cast<const std::byte *>(jpegBuf.data());
                img.data.assign(bytes, bytes + jpegBuf.size());

                _channel.log(img);
            });
    };

    auto onCamFront      = makeImageSubscriber(camFrontChannel, "camera_front");
    auto onCamLeft       = makeImageSubscriber(camLeftChannel, "camera_left");
    auto onCamRight      = makeImageSubscriber(camRightChannel, "camera_right");
    // frame_id "camera_front" is deliberate, not a copy-paste slip: both the
    // stitched panorama and the annotated-detections image are built in the
    // front camera's frame (see camera_stitcher.cpp), so that's the /tf
    // entry Foxglove should place them relative to.
    auto onCamStitched   = makeImageSubscriber(camStitchedChannel, "camera_front");
    auto onCamDetections = makeImageSubscriber(camDetectionsChannel, "camera_front");

    if (!node.Subscribe("/camera/front/image", onCamFront))
    {
        std::cerr << "Failed to subscribe to /camera/front/image\n";
        return 1;
    }
    if (!node.Subscribe("/camera/left/image", onCamLeft))
    {
        std::cerr << "Failed to subscribe to /camera/left/image\n";
        return 1;
    }
    if (!node.Subscribe("/camera/right/image", onCamRight))
    {
        std::cerr << "Failed to subscribe to /camera/right/image\n";
        return 1;
    }
    if (!node.Subscribe("/camera/stitched/image", onCamStitched))
    {
        std::cerr << "Failed to subscribe to /camera/stitched/image\n";
        return 1;
    }
    if (!node.Subscribe("/camera/detections/image", onCamDetections))
    {
        std::cerr << "Failed to subscribe to /camera/detections/image\n";
        return 1;
    }

    // gpu_lidar auto-publishes two topics: <topic> itself carries a raw
    // LaserScan, and <topic>/points carries the PointCloudPacked version we
    // actually want. No pose is baked in here -- the "os1_128" frame_id is
    // placed via /tf (fsd_car -> os1_128, published alongside world -> fsd_car
    // below) instead.
    std::function<void(const gz::msgs::PointCloudPacked &)> onLidar =
        [&lidarChannel](const gz::msgs::PointCloudPacked &_msg)
    {
        foxglove::messages::PointCloud cloud;
        cloud.timestamp = Now();
        cloud.frame_id = "os1_128";
        cloud.point_stride = _msg.point_step();

        for (const auto &f : _msg.field())
        {
            foxglove::messages::PackedElementField field;
            field.name = f.name();
            field.offset = f.offset();
            field.type = ToFoxgloveFieldType(f.datatype());
            cloud.fields.push_back(field);
        }

        const auto *bytes = reinterpret_cast<const std::byte *>(_msg.data().data());
        cloud.data.assign(bytes, bytes + _msg.data().size());

        lidarChannel.log(cloud);
    };

    if (!node.Subscribe("/lidar/points/points", onLidar))
    {
        std::cerr << "Failed to subscribe to /lidar/points/points\n";
        return 1;
    }

    auto detections_channel_result = foxglove::messages::SceneUpdateChannel::create("/cone_detections");
    if (!detections_channel_result.has_value())
    {
        std::cerr << "Failed to create /cone_detections channel: "
                  << foxglove::strerror(detections_channel_result.error()) << '\n';
        return 1;
    }
    auto detections_channel = std::move(detections_channel_result.value());

    std::function<void(const gz::msgs::Pose_V &)> onConeDetections =
        [&detections_channel](const gz::msgs::Pose_V &_msg)
    {
        foxglove::messages::SceneEntity entity;
        entity.timestamp = Now();
        entity.frame_id = kCarModelName;
        entity.id = "cone_detections";

        for (const auto &pose : _msg.pose())
        {
            ConeSpec spec;
            if (!DetectionConeSpec(pose.name(), &spec))
            {
                continue;
            }

            // Position-only from perception (see lidar_projector.cpp) --
            // orientation is never set on these Pose messages, so an unset
            // sub-message reads back as all-zero, which is a degenerate
            // (zero-length) quaternion, not identity. Set identity
            // explicitly rather than passing that through ToFoxglovePose().
            foxglove::messages::Pose fgPose;
            fgPose.position = foxglove::messages::Vector3{
                pose.position().x(), pose.position().y(), pose.position().z()};
            fgPose.orientation = foxglove::messages::Quaternion{0, 0, 0, 1};

            foxglove::messages::CylinderPrimitive cone;
            cone.pose = fgPose;
            cone.size = foxglove::messages::Vector3{
                spec.radius * 2, spec.radius * 2, spec.length};
            cone.bottom_scale = 1.0;
            cone.top_scale = 0.0;
            cone.color = spec.color;
            entity.cylinders.push_back(cone);
        }

        foxglove::messages::SceneUpdate update;
        update.entities.push_back(std::move(entity));
        detections_channel.log(update);
    };

    if (!node.Subscribe("/cone_detections", onConeDetections))
    {
        std::cerr << "Failed to subscribe to /cone_detections\n";
        return 1;
    }

    auto planned_path_channel_result = foxglove::messages::SceneUpdateChannel::create("/planned_path");
    if (!planned_path_channel_result.has_value())
    {
        std::cerr << "Failed to create /planned_path channel: "
                  << foxglove::strerror(planned_path_channel_result.error()) << '\n';
        return 1;
    }
    auto planned_path_channel = std::move(planned_path_channel_result.value());

    std::function<void(const gz::msgs::Pose_V &)> onPlannedPath =
        [&planned_path_channel](const gz::msgs::Pose_V &_msg)
    {
        foxglove::messages::SceneEntity entity;
        entity.timestamp = Now();
        entity.frame_id = kCarModelName;
        entity.id = "planned_path";

        foxglove::messages::LinePrimitive line;
        line.type = foxglove::messages::LinePrimitive::LineType::LINE_STRIP;
        line.thickness = 0.05;
        line.scale_invariant = false;
        line.color = foxglove::messages::Color{0.1, 0.9, 0.9, 1};  // cyan
        for (const auto &pose : _msg.pose())
        {
            line.points.push_back(foxglove::messages::Point3{
                pose.position().x(), pose.position().y(), pose.position().z()});
        }
        // Only attach the line if it has at least 2 points -- Foxglove
        // would just ignore a 1-point or empty LINE_STRIP anyway, but this
        // makes the "no path this cycle" case explicit rather than
        // incidental.
        if (line.points.size() >= 2)
        {
            entity.lines.push_back(line);
        }

        foxglove::messages::SceneUpdate update;
        update.entities.push_back(std::move(entity));
        planned_path_channel.log(update);
    };
    if (!node.Subscribe("/planned_path", onPlannedPath))
    {
        std::cerr << "Failed to subscribe to /planned_path\n";
        return 1;
    }

    auto scene_channel_result = foxglove::messages::SceneUpdateChannel::create("/scene");
    if (!scene_channel_result.has_value())
    {
        std::cerr << "Failed to create /scene channel: "
                  << foxglove::strerror(scene_channel_result.error()) << '\n';
        return 1;
    }
    auto scene_channel = std::move(scene_channel_result.value());

    auto vehicle_channel_result = foxglove::messages::SceneUpdateChannel::create("/vehicle");
    if (!vehicle_channel_result.has_value())
    {
        std::cerr << "Failed to create /vehicle channel: "
                  << foxglove::strerror(vehicle_channel_result.error()) << '\n';
        return 1;
    }
    auto vehicle_channel = std::move(vehicle_channel_result.value());

    auto estimated_vehicle_channel_result =
        foxglove::messages::SceneUpdateChannel::create("/estimated_vehicle");
    if (!estimated_vehicle_channel_result.has_value())
    {
        std::cerr << "Failed to create /estimated_vehicle channel: "
                  << foxglove::strerror(estimated_vehicle_channel_result.error()) << '\n';
        return 1;
    }
    auto estimated_vehicle_channel = std::move(estimated_vehicle_channel_result.value());

    // The EKF only estimates 2D (x, y, yaw) -- there's no z/roll/pitch in
    // its state (see localization/include/ekf.hpp) -- so the ghost car's
    // height here is a fixed stand-in for the chassis's known ride height,
    // not something the filter itself produced. Must match model.sdf's
    // trackdrive.sdf spawn z (0.31).
    constexpr double kEstimatedRideHeightM = 0.31;
    constexpr const char *kEstimatedFrameId = "fsd_car_estimated";

    std::function<void(const gz::msgs::Pose &)> onEstimatedPose =
        [&tf_channel, &estimated_vehicle_channel, &wheelStaticPoses](const gz::msgs::Pose &_msg)
    {
        foxglove::messages::FrameTransform worldToEstimated;
        worldToEstimated.timestamp = Now();
        worldToEstimated.parent_frame_id = kFrameId;
        worldToEstimated.child_frame_id = kEstimatedFrameId;
        worldToEstimated.translation = foxglove::messages::Vector3{
            _msg.position().x(), _msg.position().y(), kEstimatedRideHeightM};
        worldToEstimated.rotation = foxglove::messages::Quaternion{
            _msg.orientation().x(), _msg.orientation().y(),
            _msg.orientation().z(), _msg.orientation().w()};

        foxglove::messages::FrameTransforms transforms;
        transforms.transforms.push_back(worldToEstimated);
        tf_channel.log(transforms);

        // Translucent green so it reads as a ghost overlay against /vehicle
        // (ground truth, opaque dark gray/near-black) rather than obscuring
        // it when the two nearly coincide.
        auto entity = BuildVehicleEntity(
            kEstimatedFrameId, "fsd_car_estimated",
            foxglove::messages::Color{0.1, 0.9, 0.3, 0.5},
            foxglove::messages::Color{0.1, 0.6, 0.2, 0.5},
            wheelStaticPoses);

        foxglove::messages::SceneUpdate update;
        update.entities.push_back(std::move(entity));
        estimated_vehicle_channel.log(update);
    };
    if (!node.Subscribe("/estimated_pose", onEstimatedPose))
    {
        std::cerr << "Failed to subscribe to /estimated_pose\n";
        return 1;
    }

    auto estimated_landmarks_channel_result =
        foxglove::messages::SceneUpdateChannel::create("/estimated_landmarks");
    if (!estimated_landmarks_channel_result.has_value())
    {
        std::cerr << "Failed to create /estimated_landmarks channel: "
                  << foxglove::strerror(estimated_landmarks_channel_result.error()) << '\n';
        return 1;
    }
    auto estimated_landmarks_channel = std::move(estimated_landmarks_channel_result.value());

    // Localization's SLAM map, in the world frame (unlike /cone_detections,
    // which is per-frame and body-frame) -- translucent so it reads as an
    // overlay against /scene's opaque ground-truth cones, letting you watch
    // the estimated map converge onto the real one as the car drives.
    std::function<void(const gz::msgs::Pose_V &)> onEstimatedLandmarks =
        [&estimated_landmarks_channel](const gz::msgs::Pose_V &_msg)
    {
        foxglove::messages::SceneEntity entity;
        entity.timestamp = Now();
        entity.frame_id = kFrameId;
        entity.id = "estimated_landmarks";

        for (const auto &pose : _msg.pose())
        {
            ConeSpec spec;
            if (!DetectionConeSpec(pose.name(), &spec))
            {
                continue;
            }

            foxglove::messages::Pose fgPose;
            // Landmark z isn't tracked (2D SLAM) or published -- place the
            // marker resting on the ground plane (half its own height)
            // rather than centered through it at an implicit z=0.
            fgPose.position = foxglove::messages::Vector3{
                pose.position().x(), pose.position().y(), spec.length / 2.0};
            fgPose.orientation = foxglove::messages::Quaternion{0, 0, 0, 1};

            foxglove::messages::CylinderPrimitive marker;
            marker.pose = fgPose;
            marker.size = foxglove::messages::Vector3{spec.radius * 2, spec.radius * 2, spec.length};
            marker.bottom_scale = 1.0;
            marker.top_scale = 0.0;  // cone shape, matching ground truth
            marker.color = foxglove::messages::Color{spec.color.r, spec.color.g, spec.color.b, 0.5};
            entity.cylinders.push_back(marker);
        }

        foxglove::messages::SceneUpdate update;
        update.entities.push_back(std::move(entity));
        estimated_landmarks_channel.log(update);
    };
    if (!node.Subscribe("/estimated_landmarks", onEstimatedLandmarks))
    {
        std::cerr << "Failed to subscribe to /estimated_landmarks\n";
        return 1;
    }

    // Per-stage compute-time aggregation -- see the /timing doc comment at
    // the top of this file for the full rationale.
    constexpr const char *kTimingSchemaJson =
        R"({"type":"object","properties":{)"
        R"("perception_us":{"type":"integer"},)"
        R"("perception_stitch_us":{"type":"integer"},)"
        R"("perception_detect_us":{"type":"integer"},)"
        R"("perception_postprocess_us":{"type":"integer"},)"
        R"("perception_log_us":{"type":"integer"},)"
        R"("localization_us":{"type":"integer"},)"
        R"("planning_us":{"type":"integer"},)"
        R"("control_us":{"type":"integer"}}})";

    foxglove::Schema timingSchema;
    timingSchema.name = "StageTimingUs";
    timingSchema.encoding = "jsonschema";
    timingSchema.data = reinterpret_cast<const std::byte *>(kTimingSchemaJson);
    timingSchema.data_len = std::char_traits<char>::length(kTimingSchemaJson);

    auto timing_channel_result = foxglove::RawChannel::create("/timing", "json", timingSchema);
    if (!timing_channel_result.has_value())
    {
        std::cerr << "Failed to create /timing channel: "
                  << foxglove::strerror(timing_channel_result.error()) << '\n';
        return 1;
    }
    auto timing_channel = std::move(timing_channel_result.value());

    // Latest known value per stage -- each field updates independently as
    // its own /timing/<stage> topic ticks (different stages, different
    // rates), and the full struct is republished together every time ANY
    // one field updates, so /timing always reflects the most recent known
    // duration for all stages at once. The 4 perception_* sub-fields mirror
    // perception.cpp's own /timing/perception/{stitch,detect,postprocess,
    // log} breakdown (see that file) -- added once the aggregate
    // /timing/perception number alone wasn't enough to see WHICH part of
    // perception's ~65ms/cycle cost was dominant (turned out to be
    // stitching, not detection or logging).
    struct StageTimingUs
    {
        uint64_t perception = 0;
        uint64_t perceptionStitch = 0;
        uint64_t perceptionDetect = 0;
        uint64_t perceptionPostprocess = 0;
        uint64_t perceptionLog = 0;
        uint64_t localization = 0;
        uint64_t planning = 0;
        uint64_t control = 0;
    };
    StageTimingUs stageTiming;

    auto publishTiming = [&timing_channel, &stageTiming]()
    {
        char buf[320];
        const int len = std::snprintf(buf, sizeof(buf),
            R"({"perception_us":%llu,"perception_stitch_us":%llu,"perception_detect_us":%llu,)"
            R"("perception_postprocess_us":%llu,"perception_log_us":%llu,)"
            R"("localization_us":%llu,"planning_us":%llu,"control_us":%llu})",
            static_cast<unsigned long long>(stageTiming.perception),
            static_cast<unsigned long long>(stageTiming.perceptionStitch),
            static_cast<unsigned long long>(stageTiming.perceptionDetect),
            static_cast<unsigned long long>(stageTiming.perceptionPostprocess),
            static_cast<unsigned long long>(stageTiming.perceptionLog),
            static_cast<unsigned long long>(stageTiming.localization),
            static_cast<unsigned long long>(stageTiming.planning),
            static_cast<unsigned long long>(stageTiming.control));
        timing_channel.log(reinterpret_cast<const std::byte *>(buf), static_cast<size_t>(len));
    };

    std::function<void(const gz::msgs::UInt64 &)> onPerceptionTiming =
        [&stageTiming, &publishTiming](const gz::msgs::UInt64 &_msg)
    {
        stageTiming.perception = _msg.data();
        publishTiming();
    };
    if (!node.Subscribe("/timing/perception", onPerceptionTiming))
    {
        std::cerr << "Failed to subscribe to /timing/perception\n";
        return 1;
    }

    std::function<void(const gz::msgs::UInt64 &)> onPerceptionStitchTiming =
        [&stageTiming, &publishTiming](const gz::msgs::UInt64 &_msg)
    {
        stageTiming.perceptionStitch = _msg.data();
        publishTiming();
    };
    if (!node.Subscribe("/timing/perception/stitch", onPerceptionStitchTiming))
    {
        std::cerr << "Failed to subscribe to /timing/perception/stitch\n";
        return 1;
    }

    std::function<void(const gz::msgs::UInt64 &)> onPerceptionDetectTiming =
        [&stageTiming, &publishTiming](const gz::msgs::UInt64 &_msg)
    {
        stageTiming.perceptionDetect = _msg.data();
        publishTiming();
    };
    if (!node.Subscribe("/timing/perception/detect", onPerceptionDetectTiming))
    {
        std::cerr << "Failed to subscribe to /timing/perception/detect\n";
        return 1;
    }

    std::function<void(const gz::msgs::UInt64 &)> onPerceptionPostprocessTiming =
        [&stageTiming, &publishTiming](const gz::msgs::UInt64 &_msg)
    {
        stageTiming.perceptionPostprocess = _msg.data();
        publishTiming();
    };
    if (!node.Subscribe("/timing/perception/postprocess", onPerceptionPostprocessTiming))
    {
        std::cerr << "Failed to subscribe to /timing/perception/postprocess\n";
        return 1;
    }

    std::function<void(const gz::msgs::UInt64 &)> onPerceptionLogTiming =
        [&stageTiming, &publishTiming](const gz::msgs::UInt64 &_msg)
    {
        stageTiming.perceptionLog = _msg.data();
        publishTiming();
    };
    if (!node.Subscribe("/timing/perception/log", onPerceptionLogTiming))
    {
        std::cerr << "Failed to subscribe to /timing/perception/log\n";
        return 1;
    }

    std::function<void(const gz::msgs::UInt64 &)> onLocalizationTiming =
        [&stageTiming, &publishTiming](const gz::msgs::UInt64 &_msg)
    {
        stageTiming.localization = _msg.data();
        publishTiming();
    };
    if (!node.Subscribe("/timing/localization", onLocalizationTiming))
    {
        std::cerr << "Failed to subscribe to /timing/localization\n";
        return 1;
    }

    std::function<void(const gz::msgs::UInt64 &)> onPlanningTiming =
        [&stageTiming, &publishTiming](const gz::msgs::UInt64 &_msg)
    {
        stageTiming.planning = _msg.data();
        publishTiming();
    };
    if (!node.Subscribe("/timing/planning", onPlanningTiming))
    {
        std::cerr << "Failed to subscribe to /timing/planning\n";
        return 1;
    }

    std::function<void(const gz::msgs::UInt64 &)> onControlTiming =
        [&stageTiming, &publishTiming](const gz::msgs::UInt64 &_msg)
    {
        stageTiming.control = _msg.data();
        publishTiming();
    };
    if (!node.Subscribe("/timing/control", onControlTiming))
    {
        std::cerr << "Failed to subscribe to /timing/control\n";
        return 1;
    }

    // Cones are static — gz-transport may only include their pose in the
    // first Pose_V message rather than every tick, so cache what we've seen
    // instead of relying on it being present in the message currently in hand.
    std::unordered_map<std::string, foxglove::messages::Pose> conePoseCache;

    std::function<void(const gz::msgs::Pose_V &)> onPoseV =
        [&pose_channel, &scene_channel, &vehicle_channel, &tf_channel, &conePoseCache,
         &sensorStaticPoses, &wheelStaticPoses](const gz::msgs::Pose_V &_msg)
    {
        bool haveCarPose = false;
        foxglove::messages::Pose carPose;

        for (const auto &pose : _msg.pose())
        {
            if (pose.name() == kCarModelName)
            {
                carPose = ToFoxglovePose(pose);
                haveCarPose = true;

                foxglove::messages::PoseInFrame frame;
                frame.timestamp = Now();
                frame.frame_id = kFrameId;
                frame.pose = carPose;
                pose_channel.log(frame);
                continue;
            }

            ConeSpec spec;
            if (ConeSpecForName(pose.name(), &spec))
            {
                conePoseCache[pose.name()] = ToFoxglovePose(pose);
            }
        }

        if (haveCarPose)
        {
            foxglove::messages::FrameTransforms transforms;

            foxglove::messages::FrameTransform worldToCar;
            worldToCar.timestamp = Now();
            worldToCar.parent_frame_id = kFrameId;
            worldToCar.child_frame_id = kCarModelName;
            worldToCar.translation = carPose.position;
            worldToCar.rotation = carPose.orientation;
            transforms.transforms.push_back(worldToCar);

            for (const auto &entry : sensorStaticPoses)
            {
                foxglove::messages::FrameTransform carToSensor;
                carToSensor.timestamp = Now();
                carToSensor.parent_frame_id = kCarModelName;
                carToSensor.child_frame_id = entry.first;
                carToSensor.translation = entry.second.position;
                carToSensor.rotation = entry.second.orientation;
                transforms.transforms.push_back(carToSensor);
            }

            tf_channel.log(transforms);

            // Published in the fsd_car frame -- the world->fsd_car transform
            // logged just above places it, so no pose math is needed here at
            // all, unlike /scene's track cones (which are in the world frame).
            auto vehicleEntity = BuildVehicleEntity(
                kCarModelName, kCarModelName,
                foxglove::messages::Color{0.1, 0.1, 0.1, 1},
                foxglove::messages::Color{0.05, 0.05, 0.05, 1},
                wheelStaticPoses);

            foxglove::messages::SceneUpdate vehicleUpdate;
            vehicleUpdate.entities.push_back(std::move(vehicleEntity));
            vehicle_channel.log(vehicleUpdate);
        }

        if (!haveCarPose && conePoseCache.empty())
        {
            return;
        }

        foxglove::messages::SceneUpdate update;

        foxglove::messages::SceneEntity trackEntity;
        trackEntity.timestamp = Now();
        trackEntity.frame_id = kFrameId;
        trackEntity.id = "track_cones";
        for (const auto &entry : conePoseCache)
        {
            ConeSpec spec;
            ConeSpecForName(entry.first, &spec);

            foxglove::messages::CylinderPrimitive cone;
            cone.pose = entry.second;
            cone.size = foxglove::messages::Vector3{
                spec.radius * 2, spec.radius * 2, spec.length};
            cone.bottom_scale = 1.0;  // full diameter at the base
            cone.top_scale = 0.0;     // point at the top -- makes it a cone, not a cylinder
            cone.color = spec.color;
            trackEntity.cylinders.push_back(cone);

            AppendStripeCylinders(spec, entry.second, &trackEntity.cylinders);
        }
        update.entities.push_back(std::move(trackEntity));

        scene_channel.log(update);
    };

    if (!node.Subscribe(poseTopic, onPoseV))
    {
        std::cerr << "Failed to subscribe to " << poseTopic << '\n';
        return 1;
    }

#ifdef FOXGLOVE_BRIDGE_HAS_TEGRASTATS
    // 2s interval: cheap in both senses -- tegrastats itself isn't a
    // meaningful CPU cost at any reasonable rate, but system stats also
    // don't need anything faster than this to be useful for tracking
    // trends (thermal throttling, memory pressure, power draw) alongside a
    // sim run; this is monitoring, not a control-loop input.
    constexpr int kTegrastatsIntervalMs = 2000;

    std::atomic<bool> stopTegrastats{false};
    std::thread tegrastatsThread;

    auto tegrastatsSchema = foxglove::Schema{
        "JetsonStats", "jsonschema",
        reinterpret_cast<const std::byte *>(kJetsonStatsSchemaJson),
        std::strlen(kJetsonStatsSchemaJson)};
    auto tegrastatsChannelResult =
        foxglove::RawChannel::create("/jetson_stats", "json", tegrastatsSchema);
    if (!tegrastatsChannelResult.has_value())
    {
        std::cerr << "Failed to create /jetson_stats channel: "
                  << foxglove::strerror(tegrastatsChannelResult.error())
                  << " -- system stats disabled\n";
    }
    else
    {
        auto tegrastatsChannel =
            std::make_shared<foxglove::RawChannel>(std::move(tegrastatsChannelResult.value()));
        tegrastatsThread = std::thread(
            [tegrastatsChannel, &stopTegrastats]()
            {
                RunTegrastats(kTegrastatsIntervalMs, stopTegrastats,
                              [tegrastatsChannel](const std::string &_line)
                              {
                                  const std::string json = ParseTegrastatsLineToJson(_line);
                                  tegrastatsChannel->log(
                                      reinterpret_cast<const std::byte *>(json.data()), json.size());
                              });
            });
        std::cout << "tegrastats -> /jetson_stats (every " << kTegrastatsIntervalMs << "ms)\n";
    }
#endif

    std::cout << "Foxglove bridge listening on ws://0.0.0.0:8765 -- forwarding "
              << poseTopic << " -> /vehicle_pose, /scene, /vehicle, /tf; "
              << "/estimated_pose -> /estimated_vehicle, /tf; "
              << "/estimated_landmarks -> /estimated_landmarks; "
              << "3 cameras -> /camera/{front,left,right}, /lidar/points/points -> /lidar\n";

    gz::transport::waitForShutdown();

#ifdef FOXGLOVE_BRIDGE_HAS_TEGRASTATS
    stopTegrastats = true;
    if (tegrastatsThread.joinable())
    {
        tegrastatsThread.join();
    }
#endif

    return 0;
}
