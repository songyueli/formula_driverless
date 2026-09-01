#include <algorithm>
#include <atomic>
#include <functional>
#include <iostream>
#include <vector>

#include <gz/transport/Node.hh>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/uint64.pb.h>

#include <common/frame_transform.hpp>
#include <common/scoped_timer.hpp>
#include <common/types.hpp>

#include "centerline_extractor.hpp"
#include "corridor.hpp"
#include "landmark_map.hpp"
#include "path_generator.hpp"
#include "path_utils.hpp"
#include "racing_line_optimizer.hpp"
#include "spline.hpp"
#include "track_boundaries.hpp"

// Path planning process
// ----------------------
// TWO parallel pipelines, gated by a one-way readiness latch:
//
//   1. REACTIVE (original, unmodified): every /cone_detections message
//      (body frame, one camera cycle) recomputes a short centerline path
//      from scratch using only cones visible in that single instant -- no
//      memory, no localization dependency. See path_generator.hpp/
//      track_boundaries.hpp. Serves as the cold-start fallback: it's what
//      publishes /planned_path until the landmark pipeline below has
//      enough data to take over, and keeps running (cheaply) afterward so
//      it's never left stale, matching this project's established
//      "swappable-controller"-style convention of keeping a simpler prior
//      implementation available (see pure_pursuit_controller.hpp) rather
//      than deleting it.
//
//   2. LANDMARK-BASED RACING LINE (new): draws from the EKF's ACCUMULATED,
//      world-frame landmark map (/estimated_landmarks, /estimated_pose --
//      both already published by localization.cpp) instead of one
//      instant's detections. Confirmed live as the fix for the reactive
//      pipeline's own real failure mode: a path built from only what's
//      visible RIGHT NOW has no memory of cones seen a moment ago and now
//      out of view, which was causing the car to visibly run off the
//      actual track. This is a DELIBERATE break from the reactive
//      pipeline's "decoupled from localization quality" principle above --
//      an accepted tradeoff: from here on, localization drift can surface
//      as apparent planning bugs.
//
//      Stages (each independently testable, own file):
//        a. landmark_map.hpp    -- windowed query of the landmark cache
//        b. centerline_extractor.hpp -- ordered midpoints (swappable, see
//           its own FUTURE EXTENSION POINT comment for Delaunay)
//        c. spline.hpp          -- centripetal Catmull-Rom, densely sampled
//        d. corridor.hpp        -- per-sample track-width bound
//        e. racing_line_optimizer.hpp -- least-curvature line within that
//           corridor (generalizes path_generator.cpp's MinimizeCurvature)
//      Recomputes on /estimated_landmarks arrival (perception-cycle rate,
//      ~9-10Hz), NOT /estimated_pose (up to ~50Hz) -- re-running this
//      whole pipeline every pose tick would waste CPU recomputing against
//      landmark data that hasn't actually changed. The single world->body
//      conversion uses one pose snapshot taken at the top of that
//      callback, so it stays internally consistent even though pose
//      updates faster than landmarks do.
//
//      Readiness latch (one-way: once true, stays true -- a momentary dip
//      in nearby landmark count shouldn't flap back to the reactive
//      pipeline mid-drive): the window around the vehicle needs at least 2
//      blue + 2 yellow landmarks and a pose to have arrived at least once.
//
// Inputs (subscribe):
//   /cone_detections    gz.msgs.Pose_V  (body frame -- reactive pipeline)
//   /estimated_pose     gz.msgs.Pose    (world frame -- landmark pipeline)
//   /estimated_landmarks gz.msgs.Pose_V (world frame, color in name() --
//                        landmark pipeline; recompute trigger)
//
// Output (publish):
//   /planned_path       gz.msgs.Pose_V  (ordered waypoints, BODY frame,
//                        nearest-ahead first, position only -- unchanged
//                        shape regardless of which pipeline produced it,
//                        so control.cpp needs no changes)

namespace
{
const fsd::BoundaryExtractorFn kActiveBoundaryExtractor = fsd::ColorSplitBoundaries;
const fsd::PathGeneratorFn kActivePathGenerator = fsd::NearestPairMidpointPath;
const fsd::MidpointExtractorFn kActiveMidpointExtractor = fsd::TwoPointMidpointExtractor;

// Last-line-of-defense filter, applied to BOTH pipelines' final body-frame
// output right before publish: drops any waypoint that ends up BEHIND the
// vehicle (negative body-frame x) -- confirmed directly (2026-08-31) as a
// real, user-reported symptom ("the planned line gets drawn behind the
// car"). Rather than chase down every possible upstream cause across two
// pipelines and several stages each (a spline's phantom-endpoint
// extrapolation, a smoothing pass's pull, a stale pose used for the
// world->body conversion mid-turn, etc. -- any of which COULD occasionally
// produce one), this guarantees the invariant unconditionally at the one
// place both pipelines' output has to pass through anyway. A small
// negative tolerance (not a hard x>=0) absorbs ordinary floating-point
// noise around the origin without discarding a legitimately-just-ahead
// point.
constexpr double kBehindCarTolerance = -0.05;  // meters

std::vector<fsd::PathPoint> RemoveBehindCarPoints(std::vector<fsd::PathPoint> _waypoints)
{
    _waypoints.erase(
        std::remove_if(_waypoints.begin(), _waypoints.end(),
                        [](const fsd::PathPoint &_p) { return _p.x < kBehindCarTolerance; }),
        _waypoints.end());
    return _waypoints;
}

// Landmark-pipeline tuning constants. Flagged, same as this project's
// other first-attempt constants: not yet tuned against a second
// independent live measurement -- reasonable starting points, expect to
// retune from real driving, not to have guessed harder up front.
constexpr double kWindowRadius = 20.0;          // meters
constexpr double kSplineSampleSpacing = 0.5;    // meters of arc length
constexpr double kCorridorSafetyMargin = 0.3;   // meters
constexpr double kCorridorMinHalfWidth = 0.5;   // meters
constexpr double kCorridorMaxHalfWidth = 2.0;   // meters
// Ported from path_generator.cpp's kCurvatureSmoothing* -- explicitly
// expected to need different values here: dense spline samples (every
// ~0.5m) sit much closer together than the sparse per-cycle midpoints
// those were originally tuned against, so the same absolute meter-based
// max-step behaves differently at finer sample spacing.
constexpr int kRacingLineIterations = 15;
constexpr double kRacingLineRate = 0.35;
constexpr double kRacingLineMaxStep = 0.5;      // meters

fsd::ConeColor ConeColorFromName(const std::string &_name)
{
    if (_name == "blue") return fsd::ConeColor::Blue;
    if (_name == "yellow") return fsd::ConeColor::Yellow;
    if (_name == "orange") return fsd::ConeColor::Orange;
    return fsd::ConeColor::Unknown;
}
}  // namespace

int main()
{
    gz::transport::Node node;

    auto pathPub = node.Advertise<gz::msgs::Pose_V>("/planned_path");
    auto timingPub = node.Advertise<gz::msgs::UInt64>("/timing/planning");

    // Debug topics for the landmark-based pipeline's own intermediate
    // stages -- all WORLD frame (unlike /planned_path, which is body
    // frame), one per stage, so each can be inspected directly in
    // Foxglove instead of only trusting the final published path. Added
    // specifically to debug this pipeline's own live behavior (a real
    // forward/backward-window bug already found and fixed this way, plus
    // a reported oscillation still being chased down) -- not meant to be
    // permanent instrumentation the way /timing/planning is; fine to
    // remove once this pipeline is trusted, same spirit as the FOXGLOVE
    // BRIDGE's own TEMPORARY debug landmark topic.
    auto debugMidpointsPub = node.Advertise<gz::msgs::Pose_V>("/planning/debug_midpoints");
    auto debugSplinePub = node.Advertise<gz::msgs::Pose_V>("/planning/debug_spline");
    auto debugCorridorLeftPub = node.Advertise<gz::msgs::Pose_V>("/planning/debug_corridor_left");
    auto debugCorridorRightPub = node.Advertise<gz::msgs::Pose_V>("/planning/debug_corridor_right");
    auto debugRacingLinePub = node.Advertise<gz::msgs::Pose_V>("/planning/debug_racing_line");

    fsd::LandmarkMap landmarkMap;
    std::atomic<bool> landmarkPipelineActive{false};

    std::function<void(const gz::msgs::Pose_V &)> onConeDetections =
        [&pathPub, &timingPub, &landmarkPipelineActive](const gz::msgs::Pose_V &_msg)
    {
        // Still computed every cycle even once the landmark pipeline takes
        // over publishing -- see this file's header comment for why: an
        // always-warm fallback, never left stale.
        fsd::ScopedTimer timer([&timingPub, &landmarkPipelineActive](int64_t _us)
        {
            if (landmarkPipelineActive.load(std::memory_order_relaxed))
            {
                return;  // don't double-publish /timing/planning against the landmark pipeline's own
            }
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
        const std::vector<fsd::PathPoint> waypoints = RemoveBehindCarPoints(kActivePathGenerator(boundaries));

        if (landmarkPipelineActive.load(std::memory_order_relaxed))
        {
            return;  // landmark pipeline is now the sole publisher
        }

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

    std::function<void(const gz::msgs::Pose &)> onEstimatedPose =
        [&landmarkMap](const gz::msgs::Pose &_msg)
    {
        const double yaw = fsd::YawFromPlanarQuaternion(_msg.orientation().z(), _msg.orientation().w());
        landmarkMap.UpdatePose(fsd::Pose2D{_msg.position().x(), _msg.position().y(), yaw});
    };
    if (!node.Subscribe("/estimated_pose", onEstimatedPose))
    {
        std::cerr << "Failed to subscribe to /estimated_pose\n";
        return 1;
    }

    std::function<void(const gz::msgs::Pose_V &)> onEstimatedLandmarks =
        [&landmarkMap, &landmarkPipelineActive, &pathPub, &timingPub, &debugMidpointsPub, &debugSplinePub,
         &debugCorridorLeftPub, &debugCorridorRightPub, &debugRacingLinePub](const gz::msgs::Pose_V &_msg)
    {
        std::vector<fsd::WorldCone> landmarks;
        landmarks.reserve(static_cast<size_t>(_msg.pose_size()));
        for (const auto &pose : _msg.pose())
        {
            landmarks.push_back(fsd::WorldCone{
                pose.position().x(), pose.position().y(), ConeColorFromName(pose.name())});
        }
        landmarkMap.UpdateLandmarks(std::move(landmarks));

        const fsd::LandmarkMap::WindowResult window = landmarkMap.QueryWindow(kWindowRadius);
        const bool ready = window.poseValid && window.blue.size() >= 2 && window.yellow.size() >= 2;
        if (!ready && !landmarkPipelineActive.load(std::memory_order_relaxed))
        {
            return;  // not ready yet -- reactive pipeline stays the sole publisher
        }
        if (ready)
        {
            landmarkPipelineActive.store(true, std::memory_order_relaxed);  // one-way latch
        }

        fsd::ScopedTimer timer([&timingPub](int64_t _us)
        {
            gz::msgs::UInt64 msg;
            msg.set_data(static_cast<uint64_t>(_us));
            timingPub.Publish(msg);
        });

        const std::vector<fsd::PathPoint> midpoints =
            kActiveMidpointExtractor(window.blue, window.yellow, window.vehiclePose);
        const std::vector<fsd::PathPoint> splineSamples =
            fsd::FitAndSampleSpline(midpoints, kSplineSampleSpacing);
        const std::vector<fsd::CorridorSample> corridor =
            fsd::ComputeCorridor(splineSamples, window.blue, window.yellow, kCorridorSafetyMargin,
                                  kCorridorMinHalfWidth, kCorridorMaxHalfWidth);
        const std::vector<fsd::PathPoint> racingLineWorld =
            fsd::OptimizeRacingLine(corridor, kRacingLineIterations, kRacingLineRate, kRacingLineMaxStep);

        // Debug topics for every intermediate stage -- see their
        // declarations above for why. All world frame.
        auto publishWorldPath = [](gz::transport::Node::Publisher &_pub, const std::vector<fsd::PathPoint> &_pts)
        {
            gz::msgs::Pose_V msg;
            for (const auto &p : _pts)
            {
                gz::msgs::Pose *pose = msg.add_pose();
                pose->mutable_position()->set_x(p.x);
                pose->mutable_position()->set_y(p.y);
            }
            _pub.Publish(msg);
        };
        publishWorldPath(debugMidpointsPub, midpoints);
        publishWorldPath(debugSplinePub, splineSamples);
        publishWorldPath(debugRacingLinePub, racingLineWorld);
        {
            std::vector<fsd::PathPoint> corridorLeft, corridorRight;
            corridorLeft.reserve(corridor.size());
            corridorRight.reserve(corridor.size());
            for (const auto &c : corridor)
            {
                const double leftNormalX = -c.tangentY;
                const double leftNormalY = c.tangentX;
                corridorLeft.push_back(fsd::PathPoint{c.point.x + c.halfWidth * leftNormalX,
                                                        c.point.y + c.halfWidth * leftNormalY});
                corridorRight.push_back(fsd::PathPoint{c.point.x - c.halfWidth * leftNormalX,
                                                         c.point.y - c.halfWidth * leftNormalY});
            }
            publishWorldPath(debugCorridorLeftPub, corridorLeft);
            publishWorldPath(debugCorridorRightPub, corridorRight);
        }

        // Single world->body conversion, right before publish -- the whole
        // pipeline above stays in world frame throughout (see this file's
        // header comment).
        std::vector<fsd::PathPoint> racingLineBody;
        racingLineBody.reserve(racingLineWorld.size());
        for (const auto &wp : racingLineWorld)
        {
            const fsd::Point2D body = fsd::WorldToBody(window.vehiclePose, wp.x, wp.y);
            racingLineBody.push_back(fsd::PathPoint{body.x, body.y});
        }

        // Safety-net clamp, same as the reactive pipeline's own final step
        // (path_utils.hpp) -- reused unmodified, since EnforceMinTurnRadius/
        // EnforceMinClearance's geometry is already correctly defined
        // relative to the vehicle at the body-frame origin. Needs ALL
        // nearby cones (including orange, which the centerline extraction
        // itself deliberately excludes) converted to body frame too.
        std::vector<fsd::ClassifiedCone> allConesBody;
        allConesBody.reserve(window.blue.size() + window.yellow.size() + window.orange.size());
        for (const auto *coneList : {&window.blue, &window.yellow, &window.orange})
        {
            for (const auto &c : *coneList)
            {
                const fsd::Point2D body = fsd::WorldToBody(window.vehiclePose, c.x, c.y);
                allConesBody.push_back(fsd::ClassifiedCone{body.x, body.y, ""});
            }
        }
        // No re-ordering needed here: WorldToBody is a rigid (distance- and
        // order-preserving) transform, so the world-frame ordering
        // OptimizeRacingLine/the corridor/spline already carried forward
        // (ultimately from centerline_extractor's OrderWaypointsByTraversal
        // starting at the vehicle's world position) is still correct in
        // body frame.
        //
        // EnforceMinTurnRadius runs LAST -- confirmed directly (2026-08-31)
        // as a real, live bug the other way around: EnforceMinClearance's
        // cone-avoidance push has no awareness of the turn-radius
        // constraint, so it can shove a waypoint back OUTSIDE the vehicle's
        // achievable curvature after EnforceMinTurnRadius had just pulled
        // it in -- confirmed live via the actual picked pure-pursuit target
        // implying curvature nearly 2x the physical max, right at a
        // hairpin apex where the racing line deliberately hugs the inside
        // boundary (see path_generator.cpp's own NearestPairMidpointPath
        // for the fuller explanation -- same bug, same fix, both call
        // sites).
        racingLineBody =
            fsd::EnforceMinTurnRadius(fsd::EnforceMinClearance(std::move(racingLineBody), allConesBody));
        racingLineBody = RemoveBehindCarPoints(std::move(racingLineBody));

        gz::msgs::Pose_V pathMsg;
        for (const auto &wp : racingLineBody)
        {
            gz::msgs::Pose *p = pathMsg.add_pose();
            p->mutable_position()->set_x(wp.x);
            p->mutable_position()->set_y(wp.y);
        }
        pathPub.Publish(pathMsg);
    };
    if (!node.Subscribe("/estimated_landmarks", onEstimatedLandmarks))
    {
        std::cerr << "Failed to subscribe to /estimated_landmarks\n";
        return 1;
    }

    std::cout << "planning: /cone_detections -> /planned_path (reactive, cold-start)\n";
    std::cout << "planning: /estimated_landmarks + /estimated_pose -> /planned_path "
                 "(landmark-based racing line, once ready)\n";

    gz::transport::waitForShutdown();
    return 0;
}
