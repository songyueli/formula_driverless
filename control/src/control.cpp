#include <cmath>
#include <functional>
#include <iostream>

#include <gz/transport/Node.hh>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/twist.pb.h>

#include <common/types.hpp>

// Control process
// ---------------
// Pure pursuit, operating entirely in the car's own body frame (see
// planning.cpp for why: /planned_path waypoints are already relative to
// the car, so no absolute pose is needed to steer toward one -- this
// process doesn't subscribe to /estimated_pose at all).
//
// Inputs (subscribe):
//   /planned_path    gz.msgs.Pose_V   (ordered body-frame waypoints from
//                    planning, nearest-ahead first)
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
//
// Algorithm (classic pure pursuit):
//   1. Pick the lookahead waypoint: the first path point at or beyond
//      kLookaheadDistance from the car's own origin (falls back to the
//      farthest available point if none are that far -- a sparse/short
//      detected path shouldn't mean no command at all).
//   2. curvature = 2*y / (x^2 + y^2) for a target at body-frame (x, y) --
//      the standard pure pursuit result for the circular arc through the
//      origin, heading along +X, that passes through the target.
//   3. yaw_rate = speed * curvature (curvature = yaw_rate / speed by
//      definition), published directly as angular.z alongside a constant
//      commanded speed.
//
// TODO 2: curvature-based speed control (slow down for tight corners) --
// constant speed for now, per the original stub's own suggested order.
// TODO 3: (advanced) MPC instead of pure pursuit for higher-speed tracking.

namespace
{
constexpr double kSpeed = 3.0;              // m/s, constant for now
constexpr double kLookaheadDistance = 3.0;  // meters
}  // namespace

int main()
{
    gz::transport::Node node;

    auto cmdPub = node.Advertise<gz::msgs::Twist>("/cmd_ackermann");

    std::function<void(const gz::msgs::Pose_V &)> onPlannedPath =
        [&cmdPub](const gz::msgs::Pose_V &_msg)
    {
        if (_msg.pose_size() == 0)
        {
            // No path this cycle -- stop rather than keep driving on a
            // stale command.
            gz::msgs::Twist stop;
            stop.mutable_linear()->set_x(0.0);
            stop.mutable_angular()->set_z(0.0);
            cmdPub.Publish(stop);
            return;
        }

        // planning.cpp publishes these sorted nearest-ahead-first already.
        double targetX = 0.0, targetY = 0.0;
        bool found = false;
        for (const auto &pose : _msg.pose())
        {
            const double x = pose.position().x();
            const double y = pose.position().y();
            if (x * x + y * y >= kLookaheadDistance * kLookaheadDistance)
            {
                targetX = x;
                targetY = y;
                found = true;
                break;
            }
        }
        if (!found)
        {
            const auto &last = _msg.pose(_msg.pose_size() - 1);
            targetX = last.position().x();
            targetY = last.position().y();
        }

        const double lookaheadSq = targetX * targetX + targetY * targetY;
        const double curvature = lookaheadSq > 1e-6 ? (2.0 * targetY / lookaheadSq) : 0.0;
        const double yawRate = kSpeed * curvature;

        gz::msgs::Twist cmd;
        cmd.mutable_linear()->set_x(kSpeed);
        cmd.mutable_angular()->set_z(yawRate);
        cmdPub.Publish(cmd);
    };

    if (!node.Subscribe("/planned_path", onPlannedPath))
    {
        std::cerr << "Failed to subscribe to /planned_path\n";
        return 1;
    }

    std::cout << "control: /planned_path -> /cmd_ackermann (pure pursuit)\n";

    gz::transport::waitForShutdown();
    return 0;
}
