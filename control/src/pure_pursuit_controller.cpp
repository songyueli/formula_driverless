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
// Fixed direction (m_sweepDirection) for the DURATION of one sweep
// attempt -- never re-decided mid-sweep -- so consecutive empty cycles
// compound into one continuous search instead of jittering back and forth
// and covering no new heading at all. It CAN flip between attempts, though
// -- see m_sweepDirection's own header comment for why a direction that
// never changes at all is its own confirmed failure mode.
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

// Even the widening spiral (kSweepSpeedGrowthPerCycle above) assumes the
// car can actually MOVE along whatever it commands -- confirmed directly
// as a real, un-covered failure mode: a live full-lap test found
// /cmd_ackermann continuously commanding forward speed (up to
// kMaxSweepSpeed, angular.z pinned at kSweepYawRate) while true
// /estimated_pose sat frozen at the same position for 100+ seconds, and
// the nearest ground-truth cone was 0.96m from the car -- physically
// wedged against it. No forward-only command (creep, sweep, or an ever-
// wider spiral) can ever produce real displacement from CONTACT.
//
// This is NOT scoped to the empty-path sweep alone, unlike an earlier
// version of this fix -- confirmed directly as insufficient by a SECOND
// live collision, at a different point in the same corner: /cmd_ackermann
// showed a completely ordinary, continuously-varying NORMAL pure-pursuit
// command (curvature-scaled yaw rate, speed up near kMaxSpeed) -- not the
// sweep's fixed kSweepYawRate -- while true position again sat frozen.
// planning's own boundary/path generation has no obstacle-awareness at
// all (see track_boundaries.hpp/path_generator.hpp), so a geometrically
// "reasonable" path can still walk the car directly into contact with a
// real cone in a tight enough corner. Every prior escalation in this file
// assumed "commanded speed didn't work" only ever meant "wrong direction"
// or "no path" -- neither covers "confidently driving into an obstacle" --
// so this stuck-check now runs FIRST, unconditionally, ahead of BOTH the
// empty-path and normal-path branches, and overrides either one the same
// way once triggered.
//
// kStuckCyclesBeforeReverse is deliberately short relative to the sweep's
// own ~21s-per-loop timescale, and shorter still relative to how far even
// kMinSpeed (1.0 m/s) should carry a normally-driving car: a WORKING
// sweep or normal drive should show meaningful net displacement well
// within one loop / one second, so little-to-no displacement this early
// is already strong evidence of a physical block, not just "hasn't found
// the way out yet". At this pipeline's ~30Hz planning cycle (camera-rate-
// limited, see perception.cpp), 90 cycles =~ 3s.
constexpr int kStuckCyclesBeforeReverse = 90;
// Below this net displacement (from the anchor -- see m_anchorX/Y's
// comment in the header for why it rolls forward on progress rather than
// staying fixed) after kStuckCyclesBeforeReverse, treated as "not actually
// moving" -- comfortably above ordinary EKF pose jitter, comfortably below
// the meaningful distance even kMinSpeed's normal-driving floor (let alone
// a working sweep) should cover in 3 seconds.
constexpr double kStuckDistanceThreshold = 0.3;  // meters
// Reverse maneuver: mostly straight back, same cautious magnitude as
// kCreepSpeed but negative, PLUS a small turn (kReverseTurnRate) so
// backing up also reorients the car rather than just retracing its own
// approach in reverse. 45 base cycles at kReverseSpeed =~ 0.75m of
// backward travel at ~30Hz.
constexpr double kReverseSpeed = -0.5;  // m/s
constexpr int kReverseCycles = 45;
// Same direction as kSweepYawRate (consistent, not re-decided per
// attempt) but noticeably gentler -- this is a controlled backup, not a
// search sweep. Combined with kReverseCycles scaling on retry (see
// kMaxReverseAttempt below), this means a LONGER reverse also turns the
// car MORE, so escalating attempts naturally end up facing further and
// further from the original (evidently blocked) approach angle without
// needing separately-tuned per-attempt logic.
constexpr double kReverseTurnRate = 0.15;  // rad/s

// A single fixed-magnitude reverse is a real, confirmed-insufficient
// mitigation on its own: a live full-lap test found the car repeatedly
// backing up ~0.75m, immediately resuming the sweep, and getting stuck
// again in essentially the same spot -- 0.75m plus the sweep's own
// re-approach was consistently not enough clearance/reorientation to
// actually escape a sufficiently tight corner, even though each
// individual reverse executed correctly (confirmed directly: caught it
// live, -0.5 m/s for the full 45 cycles, then the sweep resuming and
// re-ramping exactly as designed -- the MECHANISM wasn't broken, its
// single fixed magnitude just wasn't always enough). Escalating -- longer
// reverse, and therefore (see kReverseTurnRate) more reorientation -- on
// each attempt that doesn't lead to real progress mirrors the sweep's own
// widening-spiral philosophy (kSweepSpeedGrowthPerCycle): try the cheap,
// minimal recovery first, commit to a bigger one only once that's
// confirmed insufficient. Capped at kMaxReverseAttempt so this still
// terminates in a bounded worst-case backup rather than growing forever.
constexpr int kMaxReverseAttempt = 4;
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
    // Universal stuck-detection watchdog -- runs FIRST, unconditionally,
    // ahead of both branches below, and overrides either one once
    // triggered. See kStuckCyclesBeforeReverse's comment for why this has
    // to cover BOTH normal path-following and the empty-path creep/sweep,
    // not just one of them (a confirmed real collision happened during
    // each).
    if (m_reverseCyclesRemaining > 0)
    {
        --m_reverseCyclesRemaining;
        if (m_reverseCyclesRemaining == 0)
        {
            // Reversing is done -- let the NEXT cycle re-anchor from
            // wherever this backward travel actually left the car, rather
            // than reusing the stale (stuck) anchor point.
            m_haveAnchor = false;
        }
        return DriveCommand{kReverseSpeed, m_sweepDirection * kReverseTurnRate};
    }

    if (inputs.poseValid)
    {
        if (!m_haveAnchor)
        {
            m_anchorX = inputs.worldX;
            m_anchorY = inputs.worldY;
            m_haveAnchor = true;
            m_cyclesSinceAnchor = 0;
        }
        else
        {
            ++m_cyclesSinceAnchor;
            const double dx = inputs.worldX - m_anchorX;
            const double dy = inputs.worldY - m_anchorY;
            const double displacement = std::sqrt(dx * dx + dy * dy);
            if (displacement >= kStuckDistanceThreshold)
            {
                // Real progress -- roll the measurement window forward
                // rather than accumulating against an increasingly stale
                // start point, and remember it happened (see
                // m_madeProgressSinceLastReverse's header comment) so the
                // NEXT reverse, if any, knows this was a fresh episode
                // rather than an immediate retry.
                m_anchorX = inputs.worldX;
                m_anchorY = inputs.worldY;
                m_cyclesSinceAnchor = 0;
                m_madeProgressSinceLastReverse = true;
            }
            else if (m_cyclesSinceAnchor > kStuckCyclesBeforeReverse)
            {
                // Escalate only if the PREVIOUS reverse (if any) was
                // immediately followed by getting stuck again with no real
                // progress in between -- see kMaxReverseAttempt's comment.
                m_reverseAttempt = m_madeProgressSinceLastReverse
                                       ? 0
                                       : std::min(m_reverseAttempt + 1, kMaxReverseAttempt);
                m_madeProgressSinceLastReverse = false;
                m_reverseCyclesRemaining = kReverseCycles * (1 + m_reverseAttempt);
                // Flip search direction every time -- see m_sweepDirection's
                // header comment for why a fixed direction can turn this
                // into an endless "back up, re-approach the same obstacle"
                // loop that no amount of extra reverse DISTANCE alone fixes.
                m_sweepDirection = -m_sweepDirection;
                return DriveCommand{kReverseSpeed, m_sweepDirection * kReverseTurnRate};
            }
        }
    }
    // else: no pose yet (startup) -- stuck-detection just doesn't run
    // until a pose arrives, which happens well before this could matter in
    // practice.

    if (inputs.path.empty())
    {
        // No path this cycle -- creep, and if this persists, sweep in a
        // widening spiral, rather than stop dead. See kCreepSpeed's
        // comment for why a hard stop here is a confirmed permanent-
        // deadlock bug, not a safe default; kSweepYawRate's for why
        // straight creep alone isn't always enough; and
        // kSweepSpeedGrowthPerCycle's for why a fixed-speed sweep isn't
        // either. (Recovering from a PHYSICAL block is the stuck-check
        // above, not this escalation -- that's a different problem: "no
        // detections at all", not "detections exist but something's in
        // the way".)
        ++m_consecutiveEmptyCycles;
        if (m_consecutiveEmptyCycles <= kStraightCreepCycles)
        {
            return DriveCommand{kCreepSpeed, 0.0};
        }
        const int sweepCycles = m_consecutiveEmptyCycles - kStraightCreepCycles;
        const double sweepSpeed = std::min(kMaxSweepSpeed,
                                            kCreepSpeed + kSweepSpeedGrowthPerCycle * sweepCycles);
        return DriveCommand{sweepSpeed, m_sweepDirection * kSweepYawRate};
    }
    m_consecutiveEmptyCycles = 0;
    m_sweepDirection = 1.0;  // fresh baseline for whatever the NEXT stuck episode is

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
