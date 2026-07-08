#include <gz/transport/Node.hh>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/pointcloud_packed.pb.h>
#include <gz/msgs/pose.pb.h>

// Foxglove bridge process
// -----------------------
// Subscribes to gz-transport topics and republishes them through a
// foxglove::WebSocketServer so the Foxglove desktop app can visualize them.
//
// Connect Foxglove to:  ws://localhost:8765
//
// Topics to forward (add more as the pipeline grows):
//   /camera/front/image    → Foxglove RawImage channel
//   /lidar/points          → Foxglove PointCloud channel
//   /vehicle_pose          → Foxglove Pose channel
//   /planned_path          → Foxglove Path / Marker channel
//   /cone_detections       → Foxglove Marker channel (colored spheres)
//
// Steps to implement (do in order):
//   TODO 1: Add the Foxglove C++ SDK to the project (download or vcpkg/conan).
//            Link it in CMakeLists.txt under the foxglove_bridge target.
//   TODO 2: Create a foxglove::WebSocketServer on port 8765.
//   TODO 3: Subscribe to each gz-transport topic listed above.
//   TODO 4: In each callback, convert the gz::msgs type to the corresponding
//            Foxglove schema and call server.publish().
//   TODO 5: Confirm live visualization in the Foxglove app (roadmap step 6).

int main()
{
    gz::transport::Node node;

    // TODO: initialize foxglove::WebSocketServer

    gz::transport::waitForShutdown();
    return 0;
}
