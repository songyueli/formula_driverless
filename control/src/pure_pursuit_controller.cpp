#include "pure_pursuit_controller.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace
{
// Straight-line target speed -- confirmed to hold up at full active-landmark
// capacity (kMaxActiveLandmarks=80) once the EKF-side scaling fixes (sparse
// corrections, spatial-grid retired-landmark search) were in place.
//
// Speed scaling history (2026-08-31): the original curvature-based scheme
// (instantaneous + EMA + a forward preview) was removed at user request --
// felt like it slowed "tremendously" for no clear benefit. Removing it
// entirely was then confirmed live to let the car drift off the intended
// path on corners (the EMA/preview machinery existed for a reason, just
// not one this simpler replacement needs the same complexity for). Speed
// is now a direct linear function of the STEERING ANGLE the current
// curvature implies (see kWheelBase/kSteeringLimit below), not curvature
// itself -- the angle naturally saturates at the vehicle's own physical
// steering_limit, so there's no need for a separate artificial curvature
// cap the way the old kMaxExpectedCurvature was.
// Tried 6.0/7.0 before (2026-08-31) and both got stuck -- traced at the
// time to two real bugs since fixed: (1) the EnforceMinClearance/
// EnforceMinTurnRadius ordering bug (a clearance push could shove a
// waypoint back outside the vehicle's achievable curvature after the
// turn-radius clamp had just pulled it in -- see path_generator.cpp's own
// NearestPairMidpointPath), and (2) the AckermannSteering plugin's
// steer_p_gain being an unset, soft default (see model.sdf's own comment
// -- steering took 1.5-2+ seconds to converge on a held command). Retried
// 7.0 again after both fixes (plus the behind-car path filter and
// speed-proportional adaptive lookahead, added the same session) -- still
// crashed hard, but traced THAT to a third, genuinely different bug: at
// speed=7.0, the adaptive lookahead formula wanted 7.0m but kMaxLookahead
// was still 6.0 (a leftover from before this constant existed), silently
// giving LESS than a full second of lookahead exactly where more was
// needed most -- see kMaxLookahead's own comment. Tried 8.0 next (same
// session) with kMaxLookahead raised to 9.0 for headroom -- still crashed,
// this time a rollover, WHILE the racing-line cache (racing_line_cache.cpp)
// was also active for the first time in the same test. Reverted to 5.0
// (last confirmed-clean full lap) to isolate: is it the speed, the cache,
// or their interaction? Retry 8.0 again only after the cache itself is
// separately confirmed clean at 5.0.
// Reverted from 20.0 (2026-09-01): confirmed live as a real crash, not a
// theoretical risk -- chassis z spiked to 1.26m (vs. the normal ~0.31m
// baseline and every prior confirmed collision's own worst case, ~0.56m)
// within the first 30 seconds of driving. 20 m/s matches real FSAE top
// speed, but this software (lookahead/corridor/turn-radius tuning) has
// never been validated anywhere near it -- 5.0 is the last speed with a
// full confirmed-clean run.
constexpr double kMaxSpeed = 5.0;  // m/s
// Floor speed at/beyond the vehicle's max steering angle. Raised from the
// original 1.0 (2026-08-31, user report: "too slow") -- 1.0 was tuned
// against the OLD raw-centerline path, which genuinely needed near-max
// steering angle sustained through the whole hairpin; the new landmark-
// based racing-line pipeline (see planning/include/racing_line_optimizer.hpp)
// widens exactly that corner specifically so the car rarely needs to
// approach kSteeringLimit at all anymore, so the old floor's own
// justification is largely gone. 2.0 is a first retry, not yet validated
// live the way 1.0 was -- retest at the hairpin specifically before
// trusting it the same way.
constexpr double kMinSpeed = 2.0;  // m/s -- reverted alongside kMaxSpeed, see its own comment

// Lookahead distance is now proportional to speed (m_lastSpeed, see its
// own comment in the header for why last-cycle's speed, not this cycle's),
// bounded to [kMinLookahead, kMaxLookahead] -- replaces a single fixed
// 3.0m used regardless of speed (2026-08-31, user request). A FIXED
// lookahead at a higher speed corresponds to a shorter REACTION TIME (the
// same lookahead distance is covered in less time), which is a real
// mechanism for exactly the "steering feels lagged/oscillating" symptom
// this session chased: too-short a lookahead at speed makes the geometric
// target -- and therefore the commanded curvature -- change more sharply
// from cycle to cycle than the vehicle (even with steer_p_gain now fixed)
// can track smoothly. kLookaheadTimeConstant is a "look this many seconds
// ahead" gain, the standard way adaptive pure-pursuit lookahead is tuned;
// 1.0s is a first, untested-live value -- retest and retune from a fresh
// measurement the same way every other first-attempt constant in this file
// has been.
constexpr double kLookaheadTimeConstant = 1.0;  // seconds
constexpr double kMinLookahead = 2.0;  // meters
// Confirmed directly (2026-08-31) as a real mismatch, not just a
// theoretical one: at kMaxSpeed=7.0 with kLookaheadTimeConstant=1.0, the
// "look 1 second ahead" formula wants 7.0m, but this was clamped to 6.0 --
// backwards from the whole point of adaptive lookahead (MORE lookahead at
// higher speed for stability), giving LESS than a full second of
// look-ahead exactly in the speed regime that needed it most. Set with
// headroom above kMaxSpeed's own value so this clamp only ever engages as
// a genuine safety bound, not a silent ceiling on the adaptive formula
// itself. Left at 9.0 (headroom above kMaxSpeed) even though kMaxSpeed
// itself was reverted to 5.0 below -- this constant was never implicated
// in the rollover, no reason to also revert it.
constexpr double kMaxLookahead = 9.0;  // meters
// Bicycle-model geometry for converting curvature to a real steering angle
// -- same wheel_base and steering_limit as
// simulation/models/fsd_car/model.sdf's AckermannSteering plugin (MUST
// stay in sync with it) and path_generator.cpp's own kMinTurnRadius
// (steering_limit = atan(wheel_base / kMinTurnRadius)).
constexpr double kWheelBase = 1.55;  // meters
constexpr double kSteeringLimit = 0.332;  // radians

// Creep speed for an EMPTY path (see the early-return below) -- well under
// kMaxSpeed, cautious but nonzero. Confirmed directly as a real, fatal
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
// comfortably below kMinSpeed (1.0 m/s, the slowest a WORKING path ever
// commands) so a false-empty single cycle costs negligible ground, while a genuine
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
// kMinSpeed (1.0 m/s, the slowest a WORKING path ever commands) should
// carry a normally-driving car: a WORKING sweep or normal drive should show
// meaningful net displacement well within one loop / one second, so
// little-to-no displacement this early is already strong evidence of a
// physical block, not just "hasn't found the way out yet". At this
// pipeline's ~30Hz planning cycle (camera-rate-limited, see
// perception.cpp), 90 cycles =~ 3s.
constexpr int kStuckCyclesBeforeReverse = 90;
// Below this net displacement (from the anchor -- see m_anchorX/Y's
// comment in the header for why it rolls forward on progress rather than
// staying fixed) after kStuckCyclesBeforeReverse, treated as "not actually
// moving" -- comfortably above ordinary EKF pose jitter, comfortably below
// the meaningful distance even kMinSpeed's normal-driving floor (let alone
// a working sweep) should cover in 3 seconds.
constexpr double kStuckDistanceThreshold = 0.3;  // meters

// Independent, LONGER-timescale companion to the short-term check above --
// see m_haveLongTermAnchor's own header comment for the confirmed live gap
// this closes: the short-term anchor's own "roll forward on any 0.3m
// crossing" design, correct for not penalizing real driving, is
// vulnerable to small noise/wobble (wheels spinning against a genuine
// physical block) randomly walking past 0.3m before kStuckCyclesBeforeReverse
// cycles ever elapse, resetting the counter forever. This check instead
// only samples position every kLongTermStuckWindow cycles (not
// continuously), so it can't be perpetually reset by noise -- only by
// ACTUAL displacement since the last snapshot. kLongTermStuckWindow (~10s
// at 30Hz) and kLongTermStuckThreshold (comfortably more than one
// snapshot period's worth of noise, comfortably less than kMinSpeed's own
// normal-driving distance over that period) are both deliberately looser
// than the short-term check's own -- this is a slower-to-fire backstop
// for exactly the case the fast check structurally can't catch, not a
// replacement for it.
constexpr int kLongTermStuckWindow = 300;      // cycles (~10s at 30Hz)
constexpr double kLongTermStuckThreshold = 1.0;  // meters
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
//   1. Pick the lookahead waypoint: the first path point at or beyond the
//      current lookahead distance (proportional to last cycle's speed,
//      see kLookaheadTimeConstant's own comment) from the car's own
//      origin (falls back to the farthest available point if none are
//      that far -- a sparse/short detected path shouldn't mean no command
//      at all).
//   2. curvature = 2*y / (x^2 + y^2) for a target at body-frame (x, y) --
//      the standard pure pursuit result for the circular arc through the
//      origin, heading along +X, that passes through the target. This is
//      purely geometric (depends only on the target's position, not speed),
//      so it's computed before speed.
//   3. Convert curvature to the steering angle it implies via the bicycle
//      model (atan(wheel_base * curvature)), and scale speed linearly
//      between kMaxSpeed (zero angle) and kMinSpeed (at/beyond
//      kSteeringLimit) -- see kMinSpeed's own comment for this scheme's
//      history.
//   4. yaw_rate = speed * curvature (curvature = yaw_rate / speed by
//      definition) -- using the SCALED speed from step 3, so the reported
//      yaw_rate stays consistent with the speed actually commanded.
DriveCommand PurePursuitController::Compute(const ControlInputs &inputs)
{
    // Universal stuck-detection watchdog -- runs FIRST, unconditionally,
    // ahead of both branches below, and overrides either one once
    // triggered. See kStuckCyclesBeforeReverse's comment for why this has
    // to cover BOTH normal path-following and the empty-path creep/sweep,
    // not just one of them (a confirmed real collision happened during
    // each).
    //
    // SAFETY OVERRIDE (2026-08-23): reverse driving is disallowed under FS
    // driverless rules, no exceptions -- including recovering from a
    // physical stall, which is what this watchdog was originally built for
    // (see kReverseSpeed/kReverseCycles below, now unused by this branch).
    // A stuck car must come to a full stop and be treated as a failure
    // requiring external intervention (operator/sim reset), not attempt to
    // self-recover by reversing. This branch used to command
    // DriveCommand{kReverseSpeed, ...} here and in the stuck-trigger branch
    // below; both now command a hard stop instead. m_reverseCyclesRemaining
    // is consequently never set to a nonzero value anymore (see below) --
    // this dead branch is left in place (rather than deleted) so a future,
    // rules-compliant recovery strategy (e.g. requesting an operator
    // takeover, or a rules-legal forward-only re-route) has an obvious slot
    // to land in without re-threading the watchdog's own state machine.
    if (m_reverseCyclesRemaining > 0)
    {
        --m_reverseCyclesRemaining;
        if (m_reverseCyclesRemaining == 0)
        {
            m_haveAnchor = false;
        }
        return DriveCommand{0.0, 0.0};
    }

    // Checked before everything else, including the anchor/displacement
    // logic below -- see m_permanentlyStuck's own header comment for the
    // confirmed real bug this closes (without this early return, the
    // log-throttle reset a few lines down let the car resume full normal
    // driving for ~kStuckCyclesBeforeReverse cycles between each [STUCK]
    // pulse, forever, instead of actually holding position).
    if (m_permanentlyStuck)
    {
        return DriveCommand{0.0, 0.0};
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
                // SAFETY OVERRIDE (2026-08-23): see this function's top
                // comment -- reverse is disallowed, so this no longer
                // starts a reverse maneuver (m_reverseCyclesRemaining stays
                // 0, m_reverseAttempt/m_madeProgressSinceLastReverse are
                // consequently dead state too, left alone for the same
                // reason noted above). Loudly logged to stderr (unbuffered,
                // unlike stdout here -- see perception.cpp's own per-
                // detection prints for why an unflushed stdout buffer can
                // sit unseen for minutes) specifically so an external
                // watcher can detect this and reset the sim -- a stuck car
                // that cannot reverse has no other way out.
                std::fprintf(stderr,
                    "[STUCK] no forward progress for %d cycles at approx (%.2f,%.2f) -- "
                    "reverse recovery is DISABLED (FS rules); holding position permanently, needs external reset\n",
                    m_cyclesSinceAnchor, inputs.worldX, inputs.worldY);
                // Latch permanently stopped -- see m_permanentlyStuck's own
                // header comment. The old `m_cyclesSinceAnchor = 0` reset
                // that used to live here was meant only to throttle
                // repeated LOG spam, but it also silently let the state
                // machine resume normal driving for the next ~
                // kStuckCyclesBeforeReverse cycles before re-triggering --
                // now handled by the early-return latch check at the top
                // of this function instead, so this trigger only needs to
                // fire once.
                m_permanentlyStuck = true;
                return DriveCommand{0.0, 0.0};
            }
        }

        // Independent long-term check -- see kLongTermStuckWindow/
        // kLongTermStuckThreshold's own comment for why the short-term
        // check above can miss a real stuck condition (noise/wobble
        // perpetually resetting its rolling anchor). Snapshots position
        // only once every kLongTermStuckWindow cycles, so only genuine
        // displacement since the last snapshot -- not per-cycle noise --
        // can ever reset it.
        if (!m_haveLongTermAnchor)
        {
            m_longTermAnchorX = inputs.worldX;
            m_longTermAnchorY = inputs.worldY;
            m_haveLongTermAnchor = true;
            m_cyclesSinceLongTermAnchor = 0;
        }
        else
        {
            ++m_cyclesSinceLongTermAnchor;
            if (m_cyclesSinceLongTermAnchor >= kLongTermStuckWindow)
            {
                const double ldx = inputs.worldX - m_longTermAnchorX;
                const double ldy = inputs.worldY - m_longTermAnchorY;
                const double longTermDisplacement = std::sqrt(ldx * ldx + ldy * ldy);
                if (longTermDisplacement < kLongTermStuckThreshold)
                {
                    std::fprintf(stderr,
                        "[STUCK-LONGTERM] under %.2fm net displacement over %d cycles at approx "
                        "(%.2f,%.2f) -- reverse recovery is DISABLED (FS rules); holding position "
                        "permanently, needs external reset\n",
                        kLongTermStuckThreshold, m_cyclesSinceLongTermAnchor, inputs.worldX, inputs.worldY);
                    m_permanentlyStuck = true;
                    return DriveCommand{0.0, 0.0};
                }
                // Real net progress over the window -- snapshot forward.
                m_longTermAnchorX = inputs.worldX;
                m_longTermAnchorY = inputs.worldY;
                m_cyclesSinceLongTermAnchor = 0;
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

    // Lookahead distance for THIS cycle, from LAST cycle's speed -- see
    // kLookaheadTimeConstant's own comment for why proportional-to-speed,
    // and m_lastSpeed's own comment in the header for why one cycle
    // delayed.
    const double lookaheadDistance =
        std::clamp(kLookaheadTimeConstant * m_lastSpeed, kMinLookahead, kMaxLookahead);

    // inputs.path is already sorted nearest-ahead-first (see planning.cpp).
    double targetX = 0.0, targetY = 0.0;
    bool found = false;
    for (const auto &wp : inputs.path)
    {
        if (wp.x * wp.x + wp.y * wp.y >= lookaheadDistance * lookaheadDistance)
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

    // Speed linearly proportional to the steering angle this curvature
    // implies -- kMaxSpeed at zero angle, kMinSpeed at/beyond
    // kSteeringLimit -- see kMinSpeed/kWheelBase/kSteeringLimit's own
    // comments.
    const double steeringAngle = std::atan(kWheelBase * curvature);
    const double steeringFraction = std::clamp(std::abs(steeringAngle) / kSteeringLimit, 0.0, 1.0);
    const double speed = kMaxSpeed - (kMaxSpeed - kMinSpeed) * steeringFraction;
    const double yawRate = speed * curvature;

    m_lastSpeed = speed;  // for NEXT cycle's lookahead distance
    return DriveCommand{speed, yawRate, targetX, targetY};
}
