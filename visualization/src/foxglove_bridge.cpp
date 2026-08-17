#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <utility>

#include <gz/transport/Node.hh>
#include <gz/msgs/pose_v.pb.h>

#include <foxglove/error.hpp>
#include <foxglove/messages.hpp>
#include <foxglove/websocket.hpp>

// Foxglove bridge process
// -----------------------
// Subscribes to gz-transport topics and republishes them through a
// foxglove::WebSocketServer so the Foxglove desktop app can visualize them.
//
// Connect Foxglove to:  ws://<docker-host>:8765
//
// Currently forwards:
//   /world/empty/pose/info (gz.msgs.Pose_V, filtered to the "fsd_car" model)
//     -> /vehicle_pose (Foxglove PoseInFrame)
//
// TODO: add /camera/*/image, /lidar/points, /planned_path, /cone_detections
// once perception/planning are publishing real data.

namespace
{
constexpr const char *kPoseTopic = "/world/empty/pose/info";
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
}  // namespace

int main()
{
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

    gz::transport::Node node;

    std::function<void(const gz::msgs::Pose_V &)> onPoseV =
        [&pose_channel](const gz::msgs::Pose_V &_msg)
    {
        for (const auto &pose : _msg.pose())
        {
            if (pose.name() != kCarModelName)
            {
                continue;
            }

            foxglove::messages::PoseInFrame frame;
            frame.timestamp = Now();
            frame.frame_id = kFrameId;
            frame.pose = foxglove::messages::Pose{
                foxglove::messages::Vector3{
                    pose.position().x(), pose.position().y(), pose.position().z()},
                foxglove::messages::Quaternion{
                    pose.orientation().x(), pose.orientation().y(),
                    pose.orientation().z(), pose.orientation().w()}};

            pose_channel.log(frame);
            break;
        }
    };

    if (!node.Subscribe(kPoseTopic, onPoseV))
    {
        std::cerr << "Failed to subscribe to " << kPoseTopic << '\n';
        return 1;
    }

    std::cout << "Foxglove bridge listening on ws://0.0.0.0:8765 -- forwarding "
              << kPoseTopic << " (\"" << kCarModelName << "\") -> /vehicle_pose\n";

    gz::transport::waitForShutdown();
    return 0;
}
