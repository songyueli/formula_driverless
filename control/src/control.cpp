#include <gz/transport/Node.hh>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/twist.pb.h>

#include <common/types.hpp>

// Control process
// ---------------
// Inputs (subscribe):
//   /planned_path    gz::msgs::Pose_V   (ordered waypoints from planning)
//   /vehicle_pose    gz::msgs::Pose     (current estimated pose from localization)
//
// Output (publish):
//   /cmd_ackermann   gz::msgs::AckermannSteering
//     .speed          target forward speed (m/s)
//     .steering_angle target front-wheel steering angle (radians)
//
// Steps to implement (do in order):
//   TODO 1: Implement pure pursuit:
//            - Pick a look-ahead waypoint on /planned_path at a fixed distance
//              ahead of the vehicle.
//            - Compute the required steering angle to arc toward that waypoint.
//            - Publish a constant speed and the computed steering angle.
//   TODO 2: Add speed control — slow down for tight corners (curvature-based).
//   TODO 3: (Advanced) Replace pure pursuit with MPC for better performance
//            at higher speeds and tighter tracks.

int main()
{
    gz::transport::Node node;

    gz::transport::waitForShutdown();
    return 0;
}
