#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
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
#include "lap_detector.hpp"
#include "path_generator.hpp"
#include "path_utils.hpp"
#include "racing_line_cache.hpp"
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
    // "orange" and "large_orange" are two DISTINCT classes perception's
    // own model outputs (see perception.cpp's kClassNames) -- the FSAE
    // start/finish gate uses large orange cones specifically, a real,
    // regularly-detected class, not a hypothetical one. BUG FIX
    // (2026-09-01): "large_orange" had no case here, so every such cone
    // fell through to ConeColor::Unknown, which LandmarkMap's own query
    // functions silently drop entirely (see their switch statements'
    // `default: break;`) -- confirmed directly: this track's only two
    // orange cones (the start/finish gate, per
    // simulation/worlds/trackdrive.sdf) sit right at the recurring
    // trouble spot several closed-loop pipeline bugs were traced back to
    // this session, and were completely invisible to planning the whole
    // time -- zero clearance protection, no influence on anything.
    // Treated identically to "orange" here: neither is a centerline
    // color, both matter only for clearance, and nothing downstream needs
    // to distinguish the two sizes.
    if (_name == "orange" || _name == "large_orange") return fsd::ConeColor::Orange;
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
    // Detects lap completion so the pipeline below can switch from a
    // windowed (local, forward-facing) query to the full-track CLOSED-LOOP
    // one -- see lap_detector.hpp's own header comment. Lives here (not
    // inside the onEstimatedLandmarks callback) so it persists its
    // leave/return state across every call, same as landmarkPipelineActive.
    fsd::LapDetector lapDetector;

    // Persists the racing line and the corridor's own left/right bounds
    // across cycles (2026-08-31 user request: "stored, and tuned as we
    // navigate around the track... less jitter") -- see
    // racing_line_cache.hpp's own comment for why recomputing everything
    // from scratch every cycle was producing jitter even when the
    // underlying track geometry hadn't actually changed. Separate caches
    // (not one shared instance) so the racing line's own points never
    // collide in the same spatial grid as the corridor boundary points --
    // they occupy genuinely different world positions in normal driving,
    // but keeping them in separate maps removes any chance of an
    // unintended interaction rather than relying on that always being true.
    fsd::RacingLineCache racingLineCache;
    fsd::RacingLineCache corridorLeftCache;
    fsd::RacingLineCache corridorRightCache;
    // How much of each cycle's freshly-computed value to blend in -- see
    // BlendAndStore's own comment. 0.3 is a first, untested-live value:
    // low enough to meaningfully damp cycle-to-cycle jitter, high enough
    // that the cache still tracks genuine track changes (a landmark
    // correction, not just noise) within a handful of cycles rather than
    // lagging noticeably behind reality. Retune from a fresh live
    // measurement the same way every other first-attempt constant in this
    // pipeline has been.
    constexpr double kBlendAlpha = 0.3;

    // Closed-loop pipeline state, computed ONCE (not every cycle) -- see
    // this file's 2026-08-31 postmortem comment at the closedLoop branch
    // below for why. The whole track's topology doesn't change quickly
    // once mapped; re-deciding it from scratch every ~10Hz cycle off
    // live, momentarily-noisy vehicle pose was the actual source of the
    // recurring stuck-car bugs here, not any single fixable defect.
    // closedLoopComputed is a one-way latch, same convention as
    // landmarkPipelineActive/LapDetector's own m_lapComplete.
    bool closedLoopComputed = false;
    std::vector<fsd::PathPoint> storedClosedMidpoints;
    std::vector<fsd::PathPoint> storedClosedSpline;
    std::vector<fsd::PathPoint> storedClosedCorridorLeft;
    std::vector<fsd::PathPoint> storedClosedCorridorRight;
    std::vector<fsd::PathPoint> storedClosedRacingLine;
    // Persistent "where on the loop is the car right now" cursor -- see
    // the per-cycle nearest-point lookup below for why this can't be a
    // fresh global nearest-distance scan every cycle. Initialized to 0
    // when the loop is first computed (the vehicle IS at/near index 0 at
    // that moment, by construction of ClosedLoopMidpointExtractor's own
    // ordering cursor).
    size_t closedLoopCursor = 0;

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
        [&landmarkMap, &landmarkPipelineActive, &lapDetector, &pathPub, &timingPub, &debugMidpointsPub,
         &debugSplinePub, &debugCorridorLeftPub, &debugCorridorRightPub, &debugRacingLinePub, &racingLineCache,
         &corridorLeftCache, &corridorRightCache, &closedLoopComputed, &storedClosedMidpoints, &storedClosedSpline,
         &storedClosedCorridorLeft, &storedClosedCorridorRight, &storedClosedRacingLine,
         &closedLoopCursor](const gz::msgs::Pose_V &_msg)
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

        // window.poseValid is guaranteed true here: either `ready` required
        // it directly, or landmarkPipelineActive was already latched true
        // by an earlier cycle where it was (poseValid itself never resets
        // to false once set -- see LandmarkMap::UpdatePose).
        lapDetector.Update(window.vehiclePose.x, window.vehiclePose.y);
        // Once a lap completes, switch from the windowed (local, forward-
        // facing) landmark set to every landmark ever seen (QueryAll(),
        // which legitimately does return the FULL discovered map --
        // confirmed directly in ekf.cpp's Landmarks(): active landmarks
        // plus m_retiredLandmarks are already merged before ever reaching
        // /estimated_landmarks). `window` itself (fetched above) stays the
        // windowed query regardless -- it's still what lapDetector and the
        // clearance safety-net below use, since clearance is an inherently
        // LOCAL concept.
        //
        // ARCHITECTURE: the closed-loop midpoints/spline/corridor/racing-
        // line are computed EXACTLY ONCE, the first cycle closedLoop goes
        // true, not every cycle -- re-deciding the whole track's topology
        // from scratch off live, momentarily-noisy vehicle pose every
        // ~10Hz cycle was the root source of most stuck-car bugs found
        // here (2026-08-31/09-01). Each cycle after that looks up the
        // nearest point on the stored line to the vehicle's CURRENT
        // position (see the lookup below) and walks forward from there --
        // "map once, localize within it", the same split every SLAM-
        // adjacent system uses.
        //
        // STATUS (2026-09-01, end of a long multi-session debugging
        // effort -- see chat history for the full blow-by-blow): SEVEN
        // real, distinct bugs were found and fixed in this pipeline,
        // each confirmed via live testing and/or a direct dump of the
        // actual frozen array: (1) premature lap detection (fixed via
        // distance-gating in lap_detector.cpp), (2) global mutual-NN
        // midpoint pairing not scaling to full-track candidate counts
        // (fixed via ClosedLoopMidpointExtractor's per-color-chain
        // approach), (3) a chain hop-distance cap silently truncating
        // the track (fixed by raising centerline_extractor.cpp's
        // kClosedChainMaxHop, confirmed run-to-run variable), (4) the
        // midpoint chain's walk direction being undecidable without a
        // heading reference (fixed by ClosedLoopMidpointExtractor's own
        // reverse+rotate check), (5) a global nearest-distance lookup
        // jumping to an unrelated branch of the loop wherever the track
        // passes close to itself (fixed via a persistent cursor + local
        // window), (6) that same window unconditionally wrapping across
        // the array's own seam right at the ambiguous switchover point
        // (fixed by clamping wraparound until the cursor is legitimately
        // past the loop's midpoint), (7) the window's nearest-by-distance
        // choice not actually leading to real forward progress in body
        // frame near a recurring hard-turn area close to the switchover
        // point (fixed by directly checking a point well into each
        // candidate's own forward continuation, with progressive window
        // widening when the immediate neighborhood has no usable
        // candidate at all).
        //
        // With all seven fixes in place, the pipeline reliably produced
        // long, clean autonomous laps (400+m of continuous driving in
        // the best validation run, including successfully continuing
        // through the seam into a second lap) with only rare, self-
        // recovering thin-path cycles (down from constant failure). The
        // LAST thing observed, twice, was the car ending up physically
        // WEDGED (frozen, normal chassis height, no active collision
        // signature) after a mid-severity bump (a brief chassis-height
        // spike) somewhere later in a long run -- diagnostics showed the
        // published path was healthy at the time (not empty/thin), so
        // this was a SEPARATE issue from everything above, in
        // pure_pursuit_controller.cpp's own stuck-detection watchdog, not
        // a planning-data defect: the watchdog correctly detected zero
        // progress but reset its own cycle counter to throttle repeated
        // log spam, which ALSO let the very next cycle fall through to
        // full normal driving again -- the car oscillated between one
        // stop cycle and ~90 cycles of normal commands forever, instead
        // of actually holding position. Fixed 2026-09-01 with a one-way
        // m_permanentlyStuck latch (pure_pursuit_controller.hpp/.cpp) --
        // confirmed live (233 consecutive zero /cmd_ackermann commands
        // during a reproduced wedge, vs. continuous pulsing before).
        //
        // Re-enabled (2026-09-01) now that both the planning-side bugs
        // and the controller-side wedge-handling bug are fixed and
        // validated. A genuine physical wedge event can still happen
        // occasionally (a separate question of how often the car
        // contacts something, not addressed by either fix) -- but the
        // car now safely holds position when it does, rather than
        // unpredictably resuming full-speed commands into whatever it's
        // stuck against.
        const bool closedLoop = lapDetector.LapComplete();
        const fsd::LandmarkMap::WindowResult activeSet = closedLoop ? landmarkMap.QueryAll() : window;

        if (closedLoop && !closedLoopComputed)
        {
            storedClosedMidpoints =
                fsd::ClosedLoopMidpointExtractor(activeSet.blue, activeSet.yellow, activeSet.vehiclePose);
            storedClosedSpline = fsd::FitAndSampleClosedSpline(storedClosedMidpoints, kSplineSampleSpacing);
            const std::vector<fsd::CorridorSample> corridor = fsd::ComputeCorridor(
                storedClosedSpline, activeSet.blue, activeSet.yellow, activeSet.orange, kCorridorSafetyMargin,
                kCorridorMinHalfWidth, kCorridorMaxHalfWidth, /*closed=*/true);
            storedClosedRacingLine = fsd::OptimizeRacingLine(corridor, kRacingLineIterations, kRacingLineRate,
                                                              kRacingLineMaxStep, /*closed=*/true);
            storedClosedCorridorLeft.clear();
            storedClosedCorridorRight.clear();
            storedClosedCorridorLeft.reserve(corridor.size());
            storedClosedCorridorRight.reserve(corridor.size());
            for (const auto &c : corridor)
            {
                const double leftNormalX = -c.tangentY;
                const double leftNormalY = c.tangentX;
                storedClosedCorridorLeft.push_back(
                    fsd::PathPoint{c.point.x + c.halfWidth * leftNormalX, c.point.y + c.halfWidth * leftNormalY});
                storedClosedCorridorRight.push_back(
                    fsd::PathPoint{c.point.x - c.halfWidth * leftNormalX, c.point.y - c.halfWidth * leftNormalY});
            }
            closedLoopComputed = true;
            closedLoopCursor = 0;
            // Max consecutive gap in the stored racing line -- diagnostic
            // only, printed once here rather than requiring another
            // offline-harness round-trip if this recurs. A large value
            // means some stretch of the loop is sparse/discontinuous
            // (missed cone detections, not a code bug) -- see
            // centerline_extractor.cpp's kClosedChainMaxHop comment for
            // the confirmed real variability here run to run.
            double maxGap = 0.0;
            for (size_t i = 1; i < storedClosedRacingLine.size(); ++i)
            {
                const double dx = storedClosedRacingLine[i].x - storedClosedRacingLine[i - 1].x;
                const double dy = storedClosedRacingLine[i].y - storedClosedRacingLine[i - 1].y;
                maxGap = std::max(maxGap, std::sqrt(dx * dx + dy * dy));
            }
            std::cerr << "planning: lap complete after " << lapDetector.DistanceTraveled()
                      << "m -- computed closed-loop pipeline once (blue=" << activeSet.blue.size()
                      << " yellow=" << activeSet.yellow.size() << " orange=" << activeSet.orange.size()
                      << ", racing line points=" << storedClosedRacingLine.size()
                      << ", max consecutive gap=" << maxGap << "m)\n";
            // Dump the ACTUAL frozen array to disk -- diagnostic only.
            // Confirmed necessary (2026-09-01): re-running the pipeline
            // offline against a LATER capture of /estimated_landmarks
            // does NOT reproduce the same array a live failure actually
            // froze, since landmark estimates keep changing after the
            // one-time computation -- an offline replay's own numbers
            // (line heading, corridor width) don't match what the live
            // process was actually using when it got stuck. Writing the
            // real, exact array out here removes that gap entirely.
            std::ofstream dump("/tmp/planning_closed_loop_dump.csv");
            if (!dump.is_open())
            {
                std::cerr << "planning: FAILED to open dump file, errno=" << errno << " (" << strerror(errno)
                          << ")\n";
            }
            dump << "idx,x,y,dirDeg,turnDeg,halfWidth\n";
            double prevDir = 0.0;
            bool havePrevDir = false;
            const size_t dn = storedClosedRacingLine.size();
            for (size_t i = 0; i < dn; ++i)
            {
                const fsd::PathPoint &p = storedClosedRacingLine[i];
                const fsd::PathPoint &next = storedClosedRacingLine[(i + 1) % dn];
                const double dir = std::atan2(next.y - p.y, next.x - p.x) * 180.0 / M_PI;
                double turn = 0.0;
                if (havePrevDir)
                {
                    turn = dir - prevDir;
                    while (turn > 180.0) turn -= 360.0;
                    while (turn < -180.0) turn += 360.0;
                }
                havePrevDir = true;
                prevDir = dir;
                const double hw = i < corridor.size() ? corridor[i].halfWidth : -1.0;
                dump << i << "," << p.x << "," << p.y << "," << dir << "," << turn << "," << hw << "\n";
            }
        }

        // Open (pre-lap-completion) pipeline: unchanged, recomputed every
        // cycle from the windowed query, same as always.
        std::vector<fsd::PathPoint> openMidpoints;
        std::vector<fsd::PathPoint> openSpline;
        std::vector<fsd::CorridorSample> openCorridor;
        std::vector<fsd::PathPoint> openRacingLine;
        if (!closedLoop)
        {
            openMidpoints = kActiveMidpointExtractor(activeSet.blue, activeSet.yellow, activeSet.vehiclePose);
            openSpline = fsd::FitAndSampleSpline(openMidpoints, kSplineSampleSpacing);
            openCorridor = fsd::ComputeCorridor(openSpline, activeSet.blue, activeSet.yellow, activeSet.orange,
                                                 kCorridorSafetyMargin, kCorridorMinHalfWidth, kCorridorMaxHalfWidth,
                                                 /*closed=*/false);
            openRacingLine = fsd::OptimizeRacingLine(openCorridor, kRacingLineIterations, kRacingLineRate,
                                                      kRacingLineMaxStep, /*closed=*/false);
            // Blend each freshly-optimized point toward whatever's cached at
            // its grid cell -- this is what actually damps cycle-to-cycle
            // jitter (see racing_line_cache.hpp's header comment); keyed by
            // the fresh (pre-blend) position since track geometry is static
            // cycle to cycle, so the same physical point should land in the
            // same cell. Only meaningful here (the open pipeline still
            // recomputes every cycle) -- the closed pipeline is computed
            // once, so there's nothing to blend across cycles for it.
            //
            // SAFETY: the blended point is a mix of THIS cycle's corridor-
            // clamped point and a PAST cycle's corridor-clamped point -- if
            // the corridor narrowed between those two cycles, the blend is
            // not itself guaranteed to stay inside the CURRENT corridor.
            // Live-tested 2026-08-31: without this re-clamp, the car drove
            // close enough to the track edge to clip cones and go airborne.
            for (size_t i = 0; i < openRacingLine.size() && i < openCorridor.size(); ++i)
            {
                fsd::PathPoint &wp = openRacingLine[i];
                const double freshX = wp.x, freshY = wp.y;
                wp = racingLineCache.BlendAndStore(freshX, freshY, wp, kBlendAlpha);
                wp = fsd::ClampToCorridor(wp, openCorridor[i]);
                racingLineCache.Overwrite(freshX, freshY, wp);
            }
        }

        // Debug topics for every intermediate stage -- see their
        // declarations above for why. All world frame. Closed: publish the
        // STORED (frozen) values -- identical every cycle, so zero jitter
        // by construction, not just damped. Open: publish this cycle's
        // fresh values, same as always.
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
        publishWorldPath(debugMidpointsPub, closedLoop ? storedClosedMidpoints : openMidpoints);
        publishWorldPath(debugSplinePub, closedLoop ? storedClosedSpline : openSpline);
        publishWorldPath(debugRacingLinePub, closedLoop ? storedClosedRacingLine : openRacingLine);
        if (closedLoop)
        {
            publishWorldPath(debugCorridorLeftPub, storedClosedCorridorLeft);
            publishWorldPath(debugCorridorRightPub, storedClosedCorridorRight);
        }
        else
        {
            std::vector<fsd::PathPoint> corridorLeft, corridorRight;
            corridorLeft.reserve(openCorridor.size());
            corridorRight.reserve(openCorridor.size());
            for (const auto &c : openCorridor)
            {
                const double leftNormalX = -c.tangentY;
                const double leftNormalY = c.tangentX;
                corridorLeft.push_back(fsd::PathPoint{c.point.x + c.halfWidth * leftNormalX,
                                                        c.point.y + c.halfWidth * leftNormalY});
                corridorRight.push_back(fsd::PathPoint{c.point.x - c.halfWidth * leftNormalX,
                                                         c.point.y - c.halfWidth * leftNormalY});
            }
            for (auto &wp : corridorLeft)
            {
                wp = corridorLeftCache.BlendAndStore(wp.x, wp.y, wp, kBlendAlpha);
            }
            for (auto &wp : corridorRight)
            {
                wp = corridorRightCache.BlendAndStore(wp.x, wp.y, wp, kBlendAlpha);
            }
            publishWorldPath(debugCorridorLeftPub, corridorLeft);
            publishWorldPath(debugCorridorRightPub, corridorRight);
        }

        // What actually gets published to /planned_path.
        // Closed: nearest-point lookup into the STORED (already
        // correctly-oriented, never re-decided) racing line, then a
        // bounded forward slice walking in that array's own fixed index
        // order -- no direction re-decision happens here at all, which is
        // the whole point of computing the loop once (see the closedLoop
        // comment above).
        //
        // The lookup itself uses a PERSISTENT CURSOR + local search
        // window, NOT a fresh global nearest-distance scan every cycle --
        // confirmed live (2026-08-31) as a real, severe bug the global-
        // scan version had: wherever a closed loop passes close to
        // itself (a start/finish area next to another section, a hairpin
        // fold, any real track has spots like this), the globally-
        // nearest-by-raw-distance index can belong to a completely
        // different, unrelated branch of the loop than the one the car
        // is actually driving on -- confirmed directly via added
        // diagnostics: nearestIdx landed at index 529 of a 532-point loop
        // (right near the array's OTHER end) while the car was still
        // only partway along its actual route from index ~0, because
        // that far-away-in-ARC-LENGTH point happened to be physically
        // close by EUCLIDEAN distance. The resulting forward slice then
        // pointed somewhere geometrically unrelated to the car's real
        // heading -- every one of the 60 points ended up behind it,
        // empty /planned_path. This is the exact same class of failure
        // path_utils.hpp's own OrderWaypointsByTraversal (hop cap) and
        // corridor.cpp's NearestChainLateralDistance (local search
        // window) already had to solve elsewhere in this file's own
        // pipeline -- global nearest-distance search is fundamentally
        // unsafe on a shape that folds back near itself; only a window
        // anchored to where the car was LAST known to be is safe. Window
        // half-width chosen generously past normal per-cycle movement
        // (kMaxSpeed=5-8m/s / ~9-10Hz cycle rate is under 1m, i.e. under
        // 2 index steps at 0.5m sample spacing) while staying far short
        // of the whole-loop jump that caused this bug.
        // Open: this cycle's freshly-computed racing line, used whole (it
        // was already scoped to the windowed/forward-facing query, so
        // never needs this kind of bounding).
        constexpr size_t kClosedLoopPublishCount = 60;
        constexpr int kClosedLoopCursorWindow = 30;  // index units (~15m at 0.5m spacing)
        std::vector<fsd::PathPoint> racingLineForPublish;
        size_t diagNearestIdx = 0;
        double diagNearestDist = -1.0;
        if (closedLoop)
        {
            if (storedClosedRacingLine.empty())
            {
                racingLineForPublish.clear();
            }
            else
            {
                const int n = static_cast<int>(storedClosedRacingLine.size());
                // Wrap the window past the array's own seam (index n-1
                // back to index 0) ONLY once the cursor has legitimately
                // advanced well past the halfway point -- confirmed live
                // (2026-08-31) as necessary, not defensive: right at
                // computation time the cursor starts at 0, and index 0's
                // own physical neighborhood is genuinely, ambiguously
                // close to index n-1's neighborhood too (they're the two
                // ends of the SAME point on a closed loop -- the vehicle's
                // switchover position). An unconditionally-wrapping window
                // search at that moment can lock onto n-1 just as easily
                // as 0, and the stored line's own local direction walking
                // FROM n-1 reads as ~170-180 degrees opposite the
                // vehicle's actual heading there (confirmed directly via
                // the storedLineHeadingVsVehicleDeg diagnostic) -- a
                // forward slice from the wrong side of the seam, nearly
                // every point behind the car. Clamping (no wraparound)
                // near the start removes that ambiguity entirely: index
                // n-1 simply isn't a candidate until the cursor has
                // covered enough real distance that "wrap back toward 0"
                // can only mean "genuinely completing another lap", not
                // "confused about which side of the start line this is".
                const bool nearEndOfLoop = closedLoopCursor > static_cast<size_t>(n) / 2;
                // Prefer the closest candidate that actually leads to real
                // forward progress -- checked DIRECTLY (does the point
                // kForwardCheckSteps further along this candidate's own
                // index order land ahead of the car in BODY frame?), not
                // via a local-tangent proxy. A local-tangent-alignment
                // version of this check was tried first and confirmed
                // live (2026-09-01) as insufficient on its own: a
                // candidate can pass a generous (100 degree) tangent
                // check at its OWN point while the array still curves
                // away over the next several samples, producing a
                // forward slice that's still entirely behind the car in
                // body frame. Checking the actual body-frame position of
                // a point well INTO the forward slice is a direct test of
                // the thing that actually matters (RemoveBehindCarPoints'
                // own criterion), not an indirect proxy for it.
                const double vehicleCosYaw = std::cos(activeSet.vehiclePose.yaw);
                const double vehicleSinYaw = std::sin(activeSet.vehiclePose.yaw);
                constexpr int kForwardCheckSteps = 20;  // ~10m ahead at 0.5m spacing
                constexpr double kForwardCheckMinBodyX = 1.0;  // meters
                // Search a given [lo,hi] index range for the closest
                // candidate passing the forward-progress check; returns
                // whether one was found, its index, and its distSq.
                auto searchRange = [&](int lo, int hi, size_t &outIdx, double &outDistSq) -> bool
                {
                    bool found = false;
                    for (int off = lo; off <= hi; ++off)
                    {
                        const size_t i = static_cast<size_t>(((off % n) + n) % n);
                        const size_t ahead = (i + static_cast<size_t>(kForwardCheckSteps)) % static_cast<size_t>(n);
                        const double adx = storedClosedRacingLine[ahead].x - activeSet.vehiclePose.x;
                        const double ady = storedClosedRacingLine[ahead].y - activeSet.vehiclePose.y;
                        // Body-frame x of the "ahead" point, using the same
                        // rotation WorldToBody applies (see
                        // common/frame_transform.hpp) -- inlined here
                        // rather than calling it, since only x is needed.
                        const double aheadBodyX = adx * vehicleCosYaw + ady * vehicleSinYaw;
                        if (aheadBodyX < kForwardCheckMinBodyX)
                        {
                            continue;  // this candidate doesn't lead anywhere ahead of the car
                        }
                        const double dx = storedClosedRacingLine[i].x - activeSet.vehiclePose.x;
                        const double dy = storedClosedRacingLine[i].y - activeSet.vehiclePose.y;
                        const double distSq = dx * dx + dy * dy;
                        if (!found || distSq < outDistSq)
                        {
                            outDistSq = distSq;
                            outIdx = i;
                            found = true;
                        }
                    }
                    return found;
                };
                // Progressive widening -- confirmed live (2026-09-01) as
                // necessary: near the recurring hard-turn area right by
                // the switchover point, EVERY candidate within the normal
                // +-30 window can fail the forward-progress check (the
                // whole local neighborhood curves away from the car's
                // current heading there), and falling back to plain
                // nearest at that point just recreates the original bug
                // (confirmed directly: nearestIdx pinned at 0-1,
                // storedLineHeadingVsVehicleDeg ~157-162 degrees, empty
                // /planned_path, car stuck). Widening the search --
                // still clamped/wrap-guarded by nearEndOfLoop exactly as
                // the normal window is -- lets the lookup skip PAST a
                // locally-bad stretch to a genuinely usable point further
                // along, rather than accepting a known-bad nearest one.
                size_t nearestIdx = closedLoopCursor;
                double bestDistSq = 0.0;
                bool foundUsable = false;
                for (int window : {kClosedLoopCursorWindow, 100, 250})
                {
                    int lo, hi;
                    if (nearEndOfLoop)
                    {
                        lo = static_cast<int>(closedLoopCursor) - window;
                        hi = static_cast<int>(closedLoopCursor) + window;
                    }
                    else
                    {
                        lo = std::max(0, static_cast<int>(closedLoopCursor) - window);
                        hi = std::min(n - 1, static_cast<int>(closedLoopCursor) + window);
                    }
                    if (searchRange(lo, hi, nearestIdx, bestDistSq))
                    {
                        foundUsable = true;
                        break;
                    }
                }
                if (!foundUsable)
                {
                    // Even the widest search found nothing usable in the
                    // STORED array -- confirmed live (2026-09-01) as a
                    // real, recurring case, not just a theoretical one:
                    // some track sections end up genuinely sparse in the
                    // one-time snapshot (perception simply didn't detect/
                    // localize enough cones there by lap-completion time --
                    // a real data gap, not a lookup bug; see
                    // centerline_extractor.cpp's kClosedChainMaxHop
                    // comment for the same root cause elsewhere). No
                    // widening of the SAME frozen data fixes a genuine
                    // hole in it. Falling back to "plain nearest anyway"
                    // here (an earlier version of this fix) just
                    // republishes a known-bad point -- confirmed live: the
                    // car sat sweeping in place for minutes because the
                    // underlying gap can't be searched around by turning.
                    //
                    // Instead, fall back to a FRESH reactive-pipeline
                    // computation from the CURRENT windowed live cone
                    // view (`window`, always fetched above regardless of
                    // closedLoop) -- the same math the pre-lap-completion
                    // pipeline already uses, just invoked for this one
                    // cycle. This sidesteps the frozen array's gap
                    // entirely: live detections exist right now even
                    // where the ONE-TIME snapshot didn't capture enough.
                    // Doesn't touch storedClosedRacingLine or
                    // closedLoopCursor -- once the car drives past this
                    // sparse stretch, the normal stored-line lookup
                    // resumes on its own next cycle.
                    const std::vector<fsd::PathPoint> fallbackMidpoints =
                        kActiveMidpointExtractor(window.blue, window.yellow, window.vehiclePose);
                    const std::vector<fsd::PathPoint> fallbackSpline =
                        fsd::FitAndSampleSpline(fallbackMidpoints, kSplineSampleSpacing);
                    const std::vector<fsd::CorridorSample> fallbackCorridor = fsd::ComputeCorridor(
                        fallbackSpline, window.blue, window.yellow, window.orange, kCorridorSafetyMargin,
                        kCorridorMinHalfWidth, kCorridorMaxHalfWidth, /*closed=*/false);
                    racingLineForPublish = fsd::OptimizeRacingLine(fallbackCorridor, kRacingLineIterations,
                                                                    kRacingLineRate, kRacingLineMaxStep,
                                                                    /*closed=*/false);
                    diagNearestIdx = closedLoopCursor;  // unchanged -- didn't move the cursor this cycle
                    diagNearestDist = -1.0;             // sentinel: fallback path was used, not a lookup
                }
                else
                {
                    closedLoopCursor = nearestIdx;
                    diagNearestIdx = nearestIdx;
                    diagNearestDist = std::sqrt(bestDistSq);
                    const size_t un = storedClosedRacingLine.size();
                    const size_t count = std::min(un, kClosedLoopPublishCount);
                    racingLineForPublish.reserve(count);
                    for (size_t k = 0; k < count; ++k)
                    {
                        racingLineForPublish.push_back(storedClosedRacingLine[(nearestIdx + k) % un]);
                    }
                }
            }
        }
        else
        {
            racingLineForPublish = openRacingLine;
        }

        // Single world->body conversion, right before publish -- the whole
        // pipeline above stays in world frame throughout (see this file's
        // header comment).
        std::vector<fsd::PathPoint> racingLineBody;
        racingLineBody.reserve(racingLineForPublish.size());
        for (const auto &wp : racingLineForPublish)
        {
            const fsd::Point2D body = fsd::WorldToBody(window.vehiclePose, wp.x, wp.y);
            racingLineBody.push_back(fsd::PathPoint{body.x, body.y});
        }
        double diagPreClampMinX = 0.0, diagPreClampMaxX = 0.0;
        if (!racingLineBody.empty())
        {
            diagPreClampMinX = diagPreClampMaxX = racingLineBody.front().x;
            for (const auto &p : racingLineBody)
            {
                diagPreClampMinX = std::min(diagPreClampMinX, p.x);
                diagPreClampMaxX = std::max(diagPreClampMaxX, p.x);
            }
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
        const size_t diagPreClampCount = racingLineBody.size();
        racingLineBody =
            fsd::EnforceMinTurnRadius(fsd::EnforceMinClearance(std::move(racingLineBody), allConesBody));
        const size_t diagPostClampCount = racingLineBody.size();
        racingLineBody = RemoveBehindCarPoints(std::move(racingLineBody));

        // Diagnostic only, rate-limited to once every ~2s (not every empty
        // cycle) -- added specifically to pin down a confirmed live empty-
        // /planned_path failure (2026-08-31) that an offline harness
        // couldn't reproduce (the harness always recomputes fresh at the
        // CURRENT vehicle position, which trivially places index 0 right
        // next to it -- the actual bug only shows up querying a FROZEN
        // array from a position that's drifted away from where it was
        // computed, which only the live process's own real per-cycle
        // state can exhibit).
        // Loosened from "empty" to "thin" (< 10 points) -- confirmed live
        // (2026-08-31) that this pipeline can get stuck with a NON-empty
        // but too-thin path (2 points), which the original empty-only
        // check never caught. Also now reports the stored line's own
        // local heading near the cursor vs. the vehicle's actual current
        // heading -- testing the hypothesis that at a sharp turn the
        // vehicle's nose can be un-aligned enough with the stored line's
        // local direction there that most of the forward window reads as
        // body-frame "behind" even though it's the genuinely correct
        // upcoming path (a case the windowed/open pipeline never faces,
        // since its own forward-facing landmark filter keeps everything
        // roughly aligned with current heading before it ever reaches
        // this stage).
        if (closedLoop && racingLineBody.size() < 10)
        {
            static int thinLogCounter = 0;
            if (thinLogCounter++ % 10 == 0)
            {
                double storedLineHeadingDeg = 0.0;
                if (!storedClosedRacingLine.empty())
                {
                    const size_t n2 = storedClosedRacingLine.size();
                    const fsd::PathPoint &a = storedClosedRacingLine[diagNearestIdx % n2];
                    const fsd::PathPoint &b = storedClosedRacingLine[(diagNearestIdx + 10) % n2];
                    const double lineHeading = std::atan2(b.y - a.y, b.x - a.x);
                    double diff = lineHeading - activeSet.vehiclePose.yaw;
                    while (diff > M_PI) diff -= 2 * M_PI;
                    while (diff < -M_PI) diff += 2 * M_PI;
                    storedLineHeadingDeg = diff * 180.0 / M_PI;
                }
                std::cerr << "planning: closed-loop /planned_path THIN (" << racingLineBody.size()
                          << " pts) -- nearestIdx=" << diagNearestIdx << " nearestDist=" << diagNearestDist
                          << "m preClampCount=" << diagPreClampCount << " preClampBodyX=[" << diagPreClampMinX
                          << "," << diagPreClampMaxX << "]" << " postClampCount=" << diagPostClampCount
                          << " allConesBody=" << allConesBody.size() << " vehiclePose=("
                          << activeSet.vehiclePose.x << "," << activeSet.vehiclePose.y << ","
                          << activeSet.vehiclePose.yaw << ")"
                          << " storedLineHeadingVsVehicleDeg=" << storedLineHeadingDeg << "\n";
            }
        }

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
