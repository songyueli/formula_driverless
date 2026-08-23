#include <cmath>
#include <functional>
#include <iostream>
#include <mutex>
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
// Reactive, body-frame local planning: every time perception publishes a
// fresh set of cone detections (already in the car's own body frame, see
// lidar_projector.cpp), this recomputes a short centerline path from
// scratch and republishes it. PATH GENERATION itself stays purely
// current-frame-driven for exactly the reason this file used to be fully
// localization-independent: a driving-loop bug shouldn't be confused with
// a localization bug. /estimated_pose and /estimated_landmarks ARE now
// subscribed to, but strictly as an ADDITIONAL clearance-check input, not
// a path-generation input -- see AugmentWithNearbyLandmarks's own comment
// for exactly why and what confirmed this was necessary.
//
// Inputs (subscribe):
//   /cone_detections     gz.msgs.Pose_V  (perception's localized cone
//                        detections, body frame, name() = class name)
//   /estimated_pose      gz.msgs.Pose    (localization's world-frame pose
//                        estimate -- used only to transform landmarks
//                        below into body frame, never for path generation
//                        itself)
//   /estimated_landmarks gz.msgs.Pose_V  (localization's persistent SLAM
//                        map, world frame -- see AugmentWithNearbyLandmarks)
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
//      NearestPairMidpointPath (nearest cross-boundary pairing + midpoint,
//      now also clearance-checked -- see path_generator.cpp's
//      EnforceClearance).
// kActiveBoundaryExtractor/kActivePathGenerator below are the one line
// each to change to try a different algorithm for either stage
// independently -- e.g. a Delaunay-triangulation-based extractor or path
// generator (see the FUTURE EXTENSION POINT comments in those headers).

namespace
{
const fsd::BoundaryExtractorFn kActiveBoundaryExtractor = fsd::ColorSplitBoundaries;
const fsd::PathGeneratorFn kActivePathGenerator = fsd::NearestPairMidpointPath;

// Landmarks farther than this from the car are irrelevant to this cycle's
// path/clearance decisions anyway -- matches path_generator.cpp's own
// kMaxPairDistance, the farthest a raw detection is ever paired against.
constexpr double kLandmarkInclusionRadius = 8.0;  // meters

// A landmark already within this distance of some RAW detection this
// cycle is almost certainly the SAME physical cone (just also currently
// visible) -- skip it to avoid feeding path generation/clearance-checking
// two entries for one cone, which could otherwise get paired against each
// other or double-count for clearance purposes. Well under real cone
// spacing (>=~2m, confirmed via this track's own layout elsewhere in this
// codebase), so this can't accidentally suppress a genuinely distinct
// nearby cone.
constexpr double kDuplicateSuppressionRadius = 1.0;  // meters
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

    // Latest /estimated_pose and /estimated_landmarks, cached the same
    // "recent enough, not precisely time-aligned" way control.cpp already
    // caches these same two topics for its own, different purpose -- see
    // AugmentWithNearbyLandmarks below for how they're used.
    std::mutex mapMutex;
    double lastPoseX = 0.0, lastPoseY = 0.0, lastPoseYaw = 0.0;
    bool havePose = false;
    std::vector<fsd::ClassifiedCone> lastLandmarksWorld;  // x,y = world frame here, not body

    std::function<void(const gz::msgs::Pose &)> onEstimatedPose =
        [&](const gz::msgs::Pose &_msg)
    {
        std::lock_guard<std::mutex> lock(mapMutex);
        lastPoseX = _msg.position().x();
        lastPoseY = _msg.position().y();
        const double qz = _msg.orientation().z();
        const double qw = _msg.orientation().w();
        lastPoseYaw = 2.0 * std::atan2(qz, qw);
        havePose = true;
    };
    if (!node.Subscribe("/estimated_pose", onEstimatedPose))
    {
        std::cerr << "Failed to subscribe to /estimated_pose\n";
        return 1;
    }

    std::function<void(const gz::msgs::Pose_V &)> onEstimatedLandmarks =
        [&](const gz::msgs::Pose_V &_msg)
    {
        std::vector<fsd::ClassifiedCone> landmarks;
        landmarks.reserve(static_cast<size_t>(_msg.pose_size()));
        for (const auto &pose : _msg.pose())
        {
            landmarks.push_back(
                fsd::ClassifiedCone{pose.position().x(), pose.position().y(), pose.name()});
        }
        std::lock_guard<std::mutex> lock(mapMutex);
        lastLandmarksWorld = std::move(landmarks);
    };
    if (!node.Subscribe("/estimated_landmarks", onEstimatedLandmarks))
    {
        std::cerr << "Failed to subscribe to /estimated_landmarks\n";
        return 1;
    }

    // Adds nearby SLAM landmarks (world frame, transformed to body frame
    // using the cached pose) to a cycle's raw detections BEFORE boundary
    // extraction/path generation see them -- confirmed directly as
    // necessary, not just a nice-to-have: a live full-lap test found the
    // car repeatedly stalling in a marginal-detection stretch where only
    // ONE boundary color was visible per frame (SingleSideOffsetPath's own
    // fixed-offset fallback engaging every time), even though the OTHER
    // boundary's cones were already reliably known -- localization had
    // mapped them from earlier, better-angled frames -- and sat only a few
    // meters away the whole time. Pure per-frame reactive planning had no
    // way to know that; this does, without making PATH GENERATION itself
    // depend on localization quality (see this file's own header comment
    // for why that separation still matters) -- augmented landmarks are
    // just more points fed into the exact same current-frame pipeline,
    // including the exact same EnforceClearance safety check every raw
    // detection already goes through.
    auto augmentWithNearbyLandmarks = [&](std::vector<fsd::ClassifiedCone> &_cones)
    {
        std::vector<fsd::ClassifiedCone> landmarksWorld;
        double poseX, poseY, poseYaw;
        bool poseValid;
        {
            std::lock_guard<std::mutex> lock(mapMutex);
            landmarksWorld = lastLandmarksWorld;
            poseX = lastPoseX;
            poseY = lastPoseY;
            poseYaw = lastPoseYaw;
            poseValid = havePose;
        }
        if (!poseValid)
        {
            return;
        }
        const double cosYaw = std::cos(poseYaw);
        const double sinYaw = std::sin(poseYaw);
        for (const auto &lm : landmarksWorld)
        {
            // World -> body frame (inverse of the world_x = x + bodyX*cos -
            // bodyY*sin convention used everywhere else in this codebase,
            // e.g. ekf.cpp's own AddLandmark).
            const double dx = lm.x - poseX;
            const double dy = lm.y - poseY;
            const double bodyX = dx * cosYaw + dy * sinYaw;
            const double bodyY = -dx * sinYaw + dy * cosYaw;
            // Behind the car -- confirmed live as a real bug, not just a
            // theoretical edge case: an early full-lap test stalled almost
            // immediately near spawn because a landmark at body-frame
            // x=-5.38m ended up in /planned_path, and pure pursuit's own
            // lookahead-distance selection (see pure_pursuit_controller.cpp)
            // picks whichever waypoint first crosses kLookaheadDistance in
            // *magnitude*, with no forward-facing check of its own --
            // producing nonsensical steering toward a point the car would
            // have to spin around to reach.
            if (bodyX <= 0.0)
            {
                continue;
            }
            const double rangeSq = bodyX * bodyX + bodyY * bodyY;
            if (rangeSq > kLandmarkInclusionRadius * kLandmarkInclusionRadius)
            {
                continue;
            }
            bool isDuplicate = false;
            for (const auto &c : _cones)
            {
                const double ddx = c.x - bodyX;
                const double ddy = c.y - bodyY;
                if (ddx * ddx + ddy * ddy < kDuplicateSuppressionRadius * kDuplicateSuppressionRadius)
                {
                    isDuplicate = true;
                    break;
                }
            }
            if (!isDuplicate)
            {
                _cones.push_back(fsd::ClassifiedCone{bodyX, bodyY, lm.colorClass});
            }
        }
    };

    std::function<void(const gz::msgs::Pose_V &)> onConeDetections =
        [&](const gz::msgs::Pose_V &_msg)
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
        augmentWithNearbyLandmarks(cones);

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

    std::cout << "planning: /cone_detections (+/estimated_pose, /estimated_landmarks) "
              << "-> /planned_path\n";

    gz::transport::waitForShutdown();
    return 0;
}
