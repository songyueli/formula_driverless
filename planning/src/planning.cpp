#include <gz/transport/Node.hh>
#include <gz/msgs/pose.pb.h>

#include <common/types.hpp>

// Path planning process
// ---------------------
// Inputs (subscribe):
//   /cone_detections   (list of fsd::ConePose — blue left boundary, yellow right)
//   /vehicle_pose      gz::msgs::Pose
//
// Output (publish):
//   /planned_path      gz::msgs::Pose_V  (sequence of (x,y) waypoints)
//
// Steps to implement (do in order):
//   TODO 1: Receive cone detections and separate them into left (blue) and
//            right (yellow) boundary sets.
//   TODO 2: Compute the centerline between the two boundary sets.
//            Start with the simplest approach: for each blue cone, find the
//            nearest yellow cone and place a waypoint at their midpoint.
//   TODO 3: Sort waypoints by distance ahead of the vehicle to get an ordered path.
//   TODO 4: (Optional) Smooth the path with a spline to reduce sharp direction
//            changes before handing it to the controller.
//   TODO 5: Publish the ordered waypoints on /planned_path.

int main()
{
    gz::transport::Node node;

    gz::transport::waitForShutdown();
    return 0;
}
