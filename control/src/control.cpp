#include <functional>
#include <iostream>
#include <mutex>

#include <gz/transport/Node.hh>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/twist.pb.h>
#include <gz/msgs/uint64.pb.h>

#include <common/scoped_timer.hpp>
#include <common/types.hpp>

#include "control_types.hpp"
#include "pure_pursuit_controller.hpp"

// Control process
// ---------------
// Pure pursuit, operating entirely in the car's own body frame for the
// actual STEERING decision (see planning.cpp for why: /planned_path
// waypoints are already relative to the car, so no absolute pose is needed
// to steer toward one). /estimated_pose IS subscribed to now, but only for
// PurePursuitController's stuck-detection/reverse-recovery path (see
// control_types.hpp's ControlInputs::worldX/worldY comment) -- normal
// driving never reads it.
//
// The actual control algorithm lives in PurePursuitController (see
// pure_pursuit_controller.hpp), not here -- this file's job is just
// gz-transport plumbing: converting /planned_path (plus the latest
// /estimated_pose) into a ControlInputs, calling ActiveController::
// Compute(), and converting the resulting DriveCommand into a
// /cmd_ackermann message. See pure_pursuit_controller.hpp for the
// swappable-controller pattern this project uses and exactly where/how an
// MPC-based controller would plug in later.
//
// Inputs (subscribe):
//   /planned_path    gz.msgs.Pose_V   (ordered body-frame waypoints from
//                    planning, nearest-ahead first)
//   /estimated_pose  gz.msgs.Pose     (localization's world-frame pose
//                    estimate -- cached, read only by the stuck-detection
//                    path, see above)
//
// Output (publish):
//   /cmd_ackermann   gz.msgs.Twist    (NOT a dedicated AckermannSteering
//                    message, despite what this stub originally said --
//                    confirmed empirically via `gz topic -i` on a running
//                    sim: the AckermannSteering system plugin subscribes
//                    with gz.msgs.Twist, the same "commanded body velocity"
//                    convention gz-sim's other drive plugins use (DiffDrive,
//                    etc.). linear.x = forward speed (m/s), angular.z =
//                    desired YAW RATE (rad/s), NOT a steering angle
//                    directly -- the plugin converts that internally via
//                    its own bicycle-model kinematics.)

namespace
{
using ActiveController = PurePursuitController;
}  // namespace

int main()
{
    gz::transport::Node node;
    ActiveController controller;

    auto cmdPub = node.Advertise<gz::msgs::Twist>("/cmd_ackermann");

    // Per-cycle compute time (microseconds) -- currently trivial pure
    // pursuit, but wrapping the whole callback body (rather than hand-
    // picking which lines to time) means this keeps working unchanged if
    // this later grows into something heavier (e.g. MPC).
    auto timingPub = node.Advertise<gz::msgs::UInt64>("/timing/control");

    // Latest /estimated_pose, cached for the stuck-detection path only --
    // see control_types.hpp's ControlInputs::worldX/worldY comment. A
    // plain mutex-guarded cache (not synchronized against onPlannedPath
    // beyond that) is fine here: this only ever needs to be "recent
    // enough", not exactly time-aligned with any specific /planned_path
    // message, since it's used to detect being stuck over many cycles, not
    // for a precise per-cycle geometric decision.
    std::mutex poseMutex;
    double lastWorldX = 0.0, lastWorldY = 0.0;
    bool havePose = false;
    std::function<void(const gz::msgs::Pose &)> onEstimatedPose =
        [&](const gz::msgs::Pose &_msg)
    {
        std::lock_guard<std::mutex> lock(poseMutex);
        lastWorldX = _msg.position().x();
        lastWorldY = _msg.position().y();
        havePose = true;
    };
    if (!node.Subscribe("/estimated_pose", onEstimatedPose))
    {
        std::cerr << "Failed to subscribe to /estimated_pose\n";
        return 1;
    }

    std::function<void(const gz::msgs::Pose_V &)> onPlannedPath =
        [&](const gz::msgs::Pose_V &_msg)
    {
        fsd::ScopedTimer timer([&timingPub](int64_t _us)
        {
            gz::msgs::UInt64 msg;
            msg.set_data(static_cast<uint64_t>(_us));
            timingPub.Publish(msg);
        });

        ControlInputs inputs;
        inputs.path.reserve(static_cast<size_t>(_msg.pose_size()));
        for (const auto &pose : _msg.pose())
        {
            inputs.path.push_back(ControlInputs::Waypoint{pose.position().x(), pose.position().y()});
        }
        {
            std::lock_guard<std::mutex> lock(poseMutex);
            inputs.worldX = lastWorldX;
            inputs.worldY = lastWorldY;
            inputs.poseValid = havePose;
        }

        const DriveCommand cmd = controller.Compute(inputs);

        gz::msgs::Twist twist;
        twist.mutable_linear()->set_x(cmd.speed);
        twist.mutable_angular()->set_z(cmd.yawRate);
        cmdPub.Publish(twist);
    };

    if (!node.Subscribe("/planned_path", onPlannedPath))
    {
        std::cerr << "Failed to subscribe to /planned_path\n";
        return 1;
    }

    std::cout << "control: /planned_path (+/estimated_pose) -> /cmd_ackermann (pure pursuit)\n";

    gz::transport::waitForShutdown();
    return 0;
}
