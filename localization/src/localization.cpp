#include <gz/transport/Node.hh>
#include <gz/msgs/odometry.pb.h>
#include <gz/msgs/imu.pb.h>

#include <common/types.hpp>

// Localization process
// --------------------
// Inputs (subscribe):
//   /model/fsd_car/odometry   gz::msgs::Odometry   (wheel odometry from Gazebo)
//   /imu                      gz::msgs::IMU        (if an IMU sensor is added to the car)
//   /lidar/points             gz::msgs::PointCloudPacked (optional: scan-matching)
//
// Output (publish):
//   /vehicle_pose             gz::msgs::Pose
//
// Steps to implement (do in order):
//   TODO 1: Subscribe to odometry and integrate wheel speeds to get a dead-reckoning
//            pose estimate. Publish it on /vehicle_pose.
//   TODO 2: Add IMU data to correct heading drift (complementary filter or EKF).
//   TODO 3: (Advanced) Add SLAM or scan-matching with the lidar for drift-free
//            long-run pose estimation.

int main()
{
    gz::transport::Node node;

    gz::transport::waitForShutdown();
    return 0;
}
