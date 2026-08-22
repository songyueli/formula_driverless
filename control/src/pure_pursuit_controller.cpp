#include "pure_pursuit_controller.hpp"

#include <algorithm>
#include <cmath>

namespace
{
// Straight-line target speed -- confirmed to hold up at full active-landmark
// capacity (kMaxActiveLandmarks=80) once the EKF-side scaling fixes (sparse
// corrections, spatial-grid retired-landmark search) were in place.
constexpr double kMaxSpeed = 3.0;  // m/s
// Floor speed for the tightest corners a live path ever produces -- a
// nonempty path never commands slower than this (an EMPTY path is handled
// separately, at kCreepSpeed below -- see its own comment for why that's
// deliberately even slower than this floor, not the same value).
constexpr double kMinSpeed = 1.0;  // m/s
constexpr double kLookaheadDistance = 3.0;  // meters
// Linear falloff from kMaxSpeed as |curvature| grows: speed = kMaxSpeed -
// kCurvatureSpeedGain * |curvature|, clamped to [kMinSpeed, kMaxSpeed].
// curvature = 2*y / (x^2+y^2) (see step 2 below) tops out around
// 2*kLookaheadDistance / kLookaheadDistance^2 = 2/kLookaheadDistance when
// the lookahead target sits directly to the side -- the tightest turn this
// geometry can express at the current lookahead distance. Sized so that
// worst case maps to exactly kMinSpeed rather than clamping well short of
// it (which would make speed indifferent to curvature over most of the
// real range) or overshooting past kMinSpeed before curvature maxes out.
constexpr double kMaxExpectedCurvature = 2.0 / kLookaheadDistance;
constexpr double kCurvatureSpeedGain = (kMaxSpeed - kMinSpeed) / kMaxExpectedCurvature;

// Creep speed for an EMPTY path (see the early-return below) -- well under
// kMinSpeed, cautious but nonzero. Confirmed directly as a real, fatal
// failure mode with the old "stop dead on empty path" behavior: a live
// full-lap test got permanently stuck (identical /estimated_pose and
// /cone_detections=pos=none-for-every-detection, frame after frame, for
// 100+ seconds straight) after every currently-visible cone happened to
// sit just outside lidar_projector.cpp's kMaxValidRange (20m) at the exact
// moment the path went empty. At speed=0 the car's own viewpoint never
// changes, so the SAME cones stay just out of range forever -- a permanent
// deadlock from one bad cycle, exactly the same failure class
// path_generator.cpp's SingleSideOffsetPath already fixed for the
// one-boundary-empty case ("With zero velocity, the car's own viewing
// angle never changes on the next frame either, so this was a permanent
// deadlock from a single bad frame") -- this is that same fix's
// counterpart for the BOTH-boundaries-empty case, which
// SingleSideOffsetPath's own guard (`left.empty() && !right.empty()`)
// deliberately doesn't cover.
//
// yawRate=0 (straight line, not curving toward a guess) is the same
// "car's own forward axis already approximates the local track direction
// closely enough over a short interval" reasoning SingleSideOffsetPath's
// own comment already relies on -- creeping forward a small distance is
// what's needed to bring previously-out-of-range cones back into view,
// not a real steering decision. BUT that reasoning only holds when the
// car's forward axis is ALREADY roughly aligned with the track -- confirmed
// directly as insufficient on its own by a second live stall (same full-lap
// test, further along): the car went empty-path with its heading nearly
// perpendicular to the track's actual direction (a sharp, likely
// over-corrected turn), and *170+ seconds of straight creep never
// recovered it* -- position AND heading both sat frozen, because driving
// straight from a badly-wrong heading just keeps facing the same wrong way
// forever, the exact same "one bad cycle becomes permanent" failure this
// whole mechanism exists to avoid, just for heading instead of position.
// See kSweepYawRate below for the escalation this added. 0.5 m/s is
// comfortably below kMinSpeed (the slowest a WORKING path ever commands)
// so a false-empty single cycle costs negligible ground, while a genuine
// multi-cycle gap (like the confirmed 100+ second stall) still recovers
// instead of stalling forever.
constexpr double kCreepSpeed = 0.5;  // m/s

// Escalation for kCreepSpeed's second failure mode (badly-wrong heading,
// not just "needs a few more meters of straight travel"): after this many
// CONSECUTIVE empty-path cycles, straight creep alone has had a fair,
// bounded chance (a few seconds at this pipeline's ~30-60Hz planning rate)
// and evidently isn't working, so a persistent turn gets added on top of
// the forward creep -- speed*yawRate traces a circular arc (turn radius =
// speed/kSweepYawRate), sweeping the car's own forward-facing sensors
// through a full 360 deg of heading (2*pi/kSweepYawRate =~ 21s at
// kCreepSpeed) rather than staring down the same wrong direction forever.
// Always the SAME turn direction (never re-decided each cycle) so
// consecutive empty cycles compound into one continuous sweep instead of
// jittering back and forth and covering no new heading at all.
constexpr int kStraightCreepCycles = 60;
constexpr double kSweepYawRate = 0.3;  // rad/s

// A CONSTANT speed here (kCreepSpeed, unchanged from the sweep's own
// starting point) would trace a FIXED-radius circle (~1.7m at kCreepSpeed)
// forever, not a widening search -- confirmed directly as a real,
// insufficient-on-its-own escalation: a live full-lap test hit a THIRD
// stall (after the straight-creep fix resolved the first two) where
// /estimated_pose sat within a few CENTIMETERS of the same point for
// 300+ seconds while the logged detections stayed empty (no `pos=(` lines
// at all) the entire time -- the sweep was actively turning (heading kept
// changing) but endlessly retracing the same tiny circle, which happened
// to contain no cone anywhere on its circumference. Growing speed over the
// duration of the sweep -- yawRate held fixed -- widens that same circle
// into a genuine outward (Archimedean-like) spiral, so a sweep that keeps
// finding nothing eventually reaches ANY reachable cone regardless of how
// far outside the initial tight circle it sits, rather than searching the
// same small disk forever. Capped at kMaxSweepSpeed (well under kMaxSpeed
// -- see its own comment -- since this is still a blind search, not a
// confident, path-following command) so the spiral eventually levels off
// at a large-but-bounded radius instead of accelerating without limit.
// Growth rate sized for roughly one full tight loop (~21s) before the
// radius really starts opening up, giving the common, cheaper case (the
// answer was nearby, just missed by the initial heading) a real chance
// before committing to a wide search.
constexpr double kSweepSpeedGrowthPerCycle = 0.002;  // m/s added per swept cycle
constexpr double kMaxSweepSpeed = 2.0;               // m/s
}  // namespace

// Algorithm (classic pure pursuit):
//   1. Pick the lookahead waypoint: the first path point at or beyond
//      kLookaheadDistance from the car's own origin (falls back to the
//      farthest available point if none are that far -- a sparse/short
//      detected path shouldn't mean no command at all).
//   2. curvature = 2*y / (x^2 + y^2) for a target at body-frame (x, y) --
//      the standard pure pursuit result for the circular arc through the
//      origin, heading along +X, that passes through the target. This is
//      purely geometric (depends only on the target's position, not speed),
//      so it's computed before speed.
//   3. Curvature-based speed scaling: linearly reduce speed from kMaxSpeed
//      (straight path, curvature ~0) down to kMinSpeed (tightest turn this
//      geometry produces) rather than driving every corner at the same
//      speed a straight ever is -- tighter turns need a smaller yaw_rate
//      margin to stay controllable at a given lookahead distance, and
//      constant full speed through a corner was the last part of this
//      controller still a flat stub (see the original TODO this replaces).
//   4. yaw_rate = speed * curvature (curvature = yaw_rate / speed by
//      definition) -- using the SCALED speed, not kMaxSpeed, so the
//      reported yaw_rate stays consistent with the speed actually commanded.
DriveCommand PurePursuitController::Compute(const ControlInputs &inputs)
{
    if (inputs.path.empty())
    {
        // No path this cycle -- creep (and, if this persists, sweep in a
        // widening spiral) rather than stop dead. See kCreepSpeed's comment
        // for why a hard stop here is a confirmed permanent-deadlock bug,
        // not a safe default; kSweepYawRate's for why straight creep alone
        // isn't always enough; and kSweepSpeedGrowthPerCycle's for why a
        // FIXED-speed sweep isn't either.
        ++m_consecutiveEmptyCycles;
        if (m_consecutiveEmptyCycles <= kStraightCreepCycles)
        {
            return DriveCommand{kCreepSpeed, 0.0};
        }
        const int sweepCycles = m_consecutiveEmptyCycles - kStraightCreepCycles;
        const double sweepSpeed = std::min(kMaxSweepSpeed,
                                            kCreepSpeed + kSweepSpeedGrowthPerCycle * sweepCycles);
        return DriveCommand{sweepSpeed, kSweepYawRate};
    }
    m_consecutiveEmptyCycles = 0;

    // inputs.path is already sorted nearest-ahead-first (see planning.cpp).
    double targetX = 0.0, targetY = 0.0;
    bool found = false;
    for (const auto &wp : inputs.path)
    {
        if (wp.x * wp.x + wp.y * wp.y >= kLookaheadDistance * kLookaheadDistance)
        {
            targetX = wp.x;
            targetY = wp.y;
            found = true;
            break;
        }
    }
    if (!found)
    {
        const auto &last = inputs.path.back();
        targetX = last.x;
        targetY = last.y;
    }

    const double lookaheadSq = targetX * targetX + targetY * targetY;
    const double curvature = lookaheadSq > 1e-6 ? (2.0 * targetY / lookaheadSq) : 0.0;

    const double speed = std::clamp(kMaxSpeed - kCurvatureSpeedGain * std::abs(curvature),
                                     kMinSpeed, kMaxSpeed);
    const double yawRate = speed * curvature;

    return DriveCommand{speed, yawRate};
}
