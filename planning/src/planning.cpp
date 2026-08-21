#include <functional>
#include <iostream>
#include <vector>

#include <gz/transport/Node.hh>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/uint64.pb.h>

#include <common/scoped_timer.hpp>
#include <common/types.hpp>

#include "path_generator.hpp"
#include "track_boundaries.hpp"

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
// Pipeline: split into two independently swappable stages (see
// track_boundaries.hpp / path_generator.hpp) --
//   1. Boundary extraction: raw classified cones -> left/right track
//      boundaries. Default: ColorSplitBoundaries (trusts perception's
//      color classification directly).
//   2. Path generation: boundaries -> ordered path. Default:
//      NearestPairMidpointPath (nearest cross-boundary pairing +
//      midpoint).
// kActiveBoundaryExtractor/kActivePathGenerator below are the one line
// each to change to try a different algorithm for either stage
// independently -- e.g. a Delaunay-triangulation-based extractor or path
// generator (see the FUTURE EXTENSION POINT comments in those headers).

namespace
{
const fsd::BoundaryExtractorFn kActiveBoundaryExtractor = fsd::ColorSplitBoundaries;
const fsd::PathGeneratorFn kActivePathGenerator = fsd::NearestPairMidpointPath;
}  // namespace

int main()
{
    gz::transport::Node node;

    auto pathPub = node.Advertise<gz::msgs::Pose_V>("/planned_path");

    // Per-cycle compute time (microseconds) -- currently trivial nearest-
    // neighbor pairing, but wrapping the whole callback body (rather than
    // hand-picking which lines to time) means this keeps working unchanged
    // if either pluggable stage later grows into something heavier.
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

        std::vector<fsd::ClassifiedCone> cones;
        cones.reserve(static_cast<size_t>(_msg.pose_size()));
        for (const auto &pose : _msg.pose())
        {
            cones.push_back(fsd::ClassifiedCone{pose.position().x(), pose.position().y(), pose.name()});
        }

        const fsd::TrackBoundaries boundaries = kActiveBoundaryExtractor(cones);
        const std::vector<fsd::PathPoint> waypoints = kActivePathGenerator(boundaries);

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
