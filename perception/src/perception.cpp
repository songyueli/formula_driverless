#include <gz/transport/Node.hh>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/pointcloud_packed.pb.h>

#include <common/types.hpp>
#include <iostream>
#include <functional>

// Perception process
// ------------------
// Inputs (subscribe):
//   /camera/front/image   gz::msgs::Image
//   /camera/left/image    gz::msgs::Image
//   /camera/right/image   gz::msgs::Image
//   /lidar/points         gz::msgs::PointCloudPacked
//
// Output (publish):
//   /cone_detections      gz::msgs::Pose_V  (or a custom proto once protoc is added)
//
// Steps to implement (do in order):
//   TODO 1: Subscribe to one camera topic and print the image dimensions to
//            confirm messages are arriving (roadmap step 5).
//   TODO 2: Load the YOLO ONNX model via TensorRT and run inference on each
//            camera frame — detect bounding boxes and classify cone color.
//   TODO 3: Subscribe to /lidar/points and project each detected bounding box
//            into the point cloud to get a 3-D cone position in the car frame.
//   TODO 4: Pack (x, y, color) for each detection and publish on /cone_detections.

int main()
{
    gz::transport::Node node;

    std::function<void(const gz::msgs::Image &)> onCameraFront = 
        [](const gz::msgs::Image &_msg)
    {
        std::cout << "camera_front: " << _msg.width() << "x" << _msg.height() 
            << " format=" << gz::msgs::PixelFormatType_Name(_msg.pixel_format_type()) << '\n';
    };

    if (!node.Subscribe("/camera/front/image", onCameraFront))
    {
        std::cerr << "Failed to subscribe to /camera/front/image\n";
        return 1;
    }

    gz::transport::waitForShutdown();
    return 0;
}
