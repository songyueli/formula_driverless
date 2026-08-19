#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

#include <gz/transport/Node.hh>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/pointcloud_packed.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/scene.pb.h>

#include <foxglove/error.hpp>
#include <foxglove/messages.hpp>
#include <foxglove/websocket.hpp>

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
//   world -> fsd_car                         (dynamic, from Pose_V every tick)
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
//     -> /scene        (Foxglove SceneUpdate)   — car chassis as a cube,
//                       every cone_{blue,yellow,orange}_* entry as a cone
//                       (a CylinderPrimitive with top_scale=0), so the whole
//                       track is viewable in Foxglove's 3D panel without
//                       needing Gazebo's own GUI.
//     -> /tf (Foxglove FrameTransforms)         — world->fsd_car
//   /camera/{front,left,right}/image (gz.msgs.Image)
//     -> /camera/{front,left,right} (Foxglove RawImage), frame_id matching
//        the sensor's own name (camera_front/camera_left/camera_right)
//   /camera/stitched/image (gz.msgs.Image, published by perception --
//     the 3 raw cameras merged into one cylindrical panorama)
//     -> /camera/stitched (Foxglove RawImage), frame_id "camera_front"
//   /camera/detections/image (gz.msgs.Image, published by perception --
//     the same panorama with every YOLO detection's bbox + class + confidence
//     drawn directly into the pixels, for visually comparing detection
//     accuracy in Foxglove; includes detections with no lidar match too,
//     unlike /cone_detections, which only carries localized ones)
//     -> /camera/detections (Foxglove RawImage), frame_id "camera_front"
//   /lidar/points/points (gz.msgs.PointCloudPacked -- gpu_lidar's derived
//   point-cloud topic; /lidar/points itself is a raw LaserScan, not this)
//     -> /lidar (Foxglove PointCloud), frame_id "os1_128"
//
// TODO: add /planned_path, /cone_detections once planning/perception are
// publishing real data.

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

std::string ToFoxgloveEncoding(gz::msgs::PixelFormatType _fmt)
{
    switch (_fmt)
    {
        case gz::msgs::PixelFormatType::RGB_INT8:  return "rgb8";
        case gz::msgs::PixelFormatType::RGBA_INT8: return "rgba8";
        case gz::msgs::PixelFormatType::BGR_INT8:  return "bgr8";
        case gz::msgs::PixelFormatType::BGRA_INT8: return "bgra8";
        case gz::msgs::PixelFormatType::L_INT8:    return "mono8";
        case gz::msgs::PixelFormatType::L_INT16:   return "mono16";
        default: return "";  // unsupported -- caller should skip the frame
    }
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

struct ConeSpec
{
    double radius;
    double length;
    foxglove::messages::Color color;
};

// Dimensions/colors must match simulation/models/cone_{blue,yellow,orange}/model.sdf.
bool ConeSpecForName(const std::string &_name, ConeSpec *_spec)
{
    if (_name.rfind("cone_blue", 0) == 0)
    {
        *_spec = ConeSpec{0.115, 0.325, foxglove::messages::Color{0, 0, 0.8, 1}};
        return true;
    }
    if (_name.rfind("cone_yellow", 0) == 0)
    {
        *_spec = ConeSpec{0.115, 0.325, foxglove::messages::Color{0.9, 0.9, 0, 1}};
        return true;
    }
    if (_name.rfind("cone_orange", 0) == 0)
    {
        *_spec = ConeSpec{0.1425, 0.505, foxglove::messages::Color{0.9, 0.35, 0, 1}};
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

    auto camFrontChannel      = foxglove::messages::RawImageChannel::create("/camera/front").value();
    auto camLeftChannel       = foxglove::messages::RawImageChannel::create("/camera/left").value();
    auto camRightChannel      = foxglove::messages::RawImageChannel::create("/camera/right").value();
    auto camStitchedChannel   = foxglove::messages::RawImageChannel::create("/camera/stitched").value();
    auto camDetectionsChannel = foxglove::messages::RawImageChannel::create("/camera/detections").value();
    auto lidarChannel         = foxglove::messages::PointCloudChannel::create("/lidar").value();

    auto makeImageSubscriber = [](foxglove::messages::RawImageChannel &_channel,
                                   std::string _frameId)
    {
        return std::function<void(const gz::msgs::Image &)>(
            [&_channel, _frameId](const gz::msgs::Image &_msg)
            {
                const std::string encoding = ToFoxgloveEncoding(_msg.pixel_format_type());
                if (encoding.empty())
                {
                    return;  // unsupported format, skip rather than send garbage
                }

                foxglove::messages::RawImage img;
                img.timestamp = Now();
                img.frame_id = _frameId;
                img.width = _msg.width();
                img.height = _msg.height();
                img.encoding = encoding;
                img.step = _msg.step();

                const auto *bytes = reinterpret_cast<const std::byte *>(_msg.data().data());
                img.data.assign(bytes, bytes + _msg.data().size());

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

    auto scene_channel_result = foxglove::messages::SceneUpdateChannel::create("/scene");
    if (!scene_channel_result.has_value())
    {
        std::cerr << "Failed to create /scene channel: "
                  << foxglove::strerror(scene_channel_result.error()) << '\n';
        return 1;
    }
    auto scene_channel = std::move(scene_channel_result.value());

    // Cones are static — gz-transport may only include their pose in the
    // first Pose_V message rather than every tick, so cache what we've seen
    // instead of relying on it being present in the message currently in hand.
    std::unordered_map<std::string, foxglove::messages::Pose> conePoseCache;

    std::function<void(const gz::msgs::Pose_V &)> onPoseV =
        [&pose_channel, &scene_channel, &tf_channel, &conePoseCache, &sensorStaticPoses](
            const gz::msgs::Pose_V &_msg)
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
        }
        update.entities.push_back(std::move(trackEntity));

        if (haveCarPose)
        {
            foxglove::messages::SceneEntity carEntity;
            carEntity.timestamp = Now();
            carEntity.frame_id = kFrameId;
            carEntity.id = "fsd_car";

            foxglove::messages::CubePrimitive chassis;
            chassis.pose = carPose;
            chassis.size = foxglove::messages::Vector3{1.8, 0.5, 0.3};  // matches model.sdf chassis box
            chassis.color = foxglove::messages::Color{0.1, 0.1, 0.1, 1};
            carEntity.cubes.push_back(chassis);

            update.entities.push_back(std::move(carEntity));
        }

        scene_channel.log(update);
    };

    if (!node.Subscribe(poseTopic, onPoseV))
    {
        std::cerr << "Failed to subscribe to " << poseTopic << '\n';
        return 1;
    }

    std::cout << "Foxglove bridge listening on ws://0.0.0.0:8765 -- forwarding "
              << poseTopic << " -> /vehicle_pose, /scene, /tf; "
              << "3 cameras -> /camera/{front,left,right}, /lidar/points/points -> /lidar\n";

    gz::transport::waitForShutdown();
    return 0;
}
