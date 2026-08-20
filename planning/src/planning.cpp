#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include <gz/transport/Node.hh>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/uint64.pb.h>

#include <common/scoped_timer.hpp>
#include <common/types.hpp>

// Path planning process
// ----------------------
// Reactive, body-frame local planning: no persistent map, no localization
// dependency -- every time perception publishes a fresh set of cone
// detections (already in the car's own body frame, see
// lidar_projector.cpp), this recomputes a short centerline path from
// scratch and republishes it, the same way most FSAE driverless stacks'
// reactive layer works. Deliberately does NOT subscribe to /estimated_pose:
// a body-frame target point needs no absolute pose to steer toward it (see
// control.cpp), so keeping planning/control decoupled from localization
// quality means a driving-loop bug can't be confused with a localization
// bug, and vice versa -- localization runs independently (useful for
// telemetry, lap-counting, eventual global planning against a real map),
// but isn't in the critical path of just driving around the track.
//
// Inputs (subscribe):
//   /cone_detections   gz.msgs.Pose_V   (perception's localized cone
//                       detections, body frame, name() = class name)
//
// Output (publish):
//   /planned_path      gz.msgs.Pose_V   (ordered centerline waypoints,
//                       body frame, nearest-ahead first; position only, no
//                       orientation -- pure pursuit only needs a target
//                       point, see control.cpp)
//
// Algorithm:
//   1. Split detections into blue (left boundary) / yellow (right boundary)
//      -- orange/large_orange are start/finish markers, not boundary cones,
//      and aren't used here.
//   2. For each blue cone, pair with its nearest yellow cone and take the
//      midpoint as a candidate waypoint, skipping pairs farther apart than
//      kMaxPairDistance -- real track width is >=3m (Formula Student
//      rules), so a much larger nearest-neighbor gap means there's no real
//      boundary cone visible on the other side, not a wide track.
//   3. Sort by body-frame x (forward distance) so control gets an ordered,
//      nearest-first path.
//
// TODO: spline-smooth the path (currently raw midpoints, which can zigzag
// with cone-spacing irregularities) before handing it to control.

namespace
{
struct Cone
{
    double x;
    double y;
};

constexpr double kMaxPairDistance = 8.0;  // meters -- see algorithm note above
}  // namespace

int main()
{
    gz::transport::Node node;

    auto pathPub = node.Advertise<gz::msgs::Pose_V>("/planned_path");

    // Per-cycle compute time (microseconds) -- currently trivial nearest-
    // neighbor pairing, but wrapping the whole callback body (rather than
    // hand-picking which lines to time) means this keeps working unchanged
    // if this algorithm later grows into something heavier.
    auto timingPub = node.Advertise<gz::msgs::UInt64>("/timing/planning");

    std::function<void(const gz::msgs::Pose_V &)> onConeDetections =
        [&pathPub, &timingPub](const gz::msgs::Pose_V &_msg)
    {
        fsd::ScopedTimer timer([&timingPub](int64_t _us)
        {
            gz::msgs::UInt64 msg;
            msg.set_data(static_cast<uint64_t>(_us));
            timingPub.Publish(msg);
        });

        std::vector<Cone> blue, yellow;
        for (const auto &pose : _msg.pose())
        {
            const Cone cone{pose.position().x(), pose.position().y()};
            if (pose.name() == "blue")
            {
                blue.push_back(cone);
            }
            else if (pose.name() == "yellow")
            {
                yellow.push_back(cone);
            }
        }

        std::vector<Cone> waypoints;
        waypoints.reserve(blue.size());
        for (const auto &b : blue)
        {
            const Cone *nearest = nullptr;
            double bestDistSq = kMaxPairDistance * kMaxPairDistance;
            for (const auto &y : yellow)
            {
                const double dx = y.x - b.x;
                const double dy = y.y - b.y;
                const double distSq = dx * dx + dy * dy;
                if (distSq < bestDistSq)
                {
                    bestDistSq = distSq;
                    nearest = &y;
                }
            }
            if (nearest)
            {
                waypoints.push_back(Cone{(b.x + nearest->x) / 2.0, (b.y + nearest->y) / 2.0});
            }
        }

        std::sort(waypoints.begin(), waypoints.end(),
                  [](const Cone &a, const Cone &c) { return a.x < c.x; });

        gz::msgs::Pose_V pathMsg;
        for (const auto &wp : waypoints)
        {
            gz::msgs::Pose *p = pathMsg.add_pose();
            p->mutable_position()->set_x(wp.x);
            p->mutable_position()->set_y(wp.y);
        }
        pathPub.Publish(pathMsg);
    };

    if (!node.Subscribe("/cone_detections", onConeDetections))
    {
        std::cerr << "Failed to subscribe to /cone_detections\n";
        return 1;
    }

    std::cout << "planning: /cone_detections -> /planned_path\n";

    gz::transport::waitForShutdown();
    return 0;
}
