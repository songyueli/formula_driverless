#pragma once

#include "control_types.hpp"

// The controller "interface" this project uses is a plain convention, not
// an abstract base class: any controller is a concrete class with a
// Compute(const ControlInputs&) -> DriveCommand method. main() (see
// control.cpp) calls that method through a single ActiveController type
// alias, so swapping algorithms is a one-line change there -- same
// compile-time-selection pattern perception.cpp already uses for
// ConeDetector vs. ConeDetectorTrt (see its ActiveDetector alias).
//
// A virtual base class was deliberately NOT used: nothing in this
// codebase uses runtime polymorphism (see camera_stitcher.hpp,
// cone_detector.hpp, lidar_projector.hpp, ekf.hpp -- all concrete
// classes), there's no present need to choose a controller at runtime
// rather than at build time, and the compiler already enforces the
// Compute() contract naturally: main() calls ActiveController::Compute(),
// so any class substituted in that doesn't provide a matching method
// simply fails to compile. If runtime switching is ever actually needed
// (e.g. a command-line flag to pick a controller without rebuilding),
// converting this concrete-class-plus-alias pattern into an abstract
// base + factory is a small, well-contained change to make at that
// point -- not worth the indirection cost before there's a real need
// for it.
//
// PurePursuitController is today's only implementation: classic pure
// pursuit, operating entirely in the car's own body frame (see
// control.cpp's header comment for why no absolute pose is needed).
// Refactored out of what used to be control.cpp's callback body
// verbatim -- same math, same constants, same behavior, just organized
// as a class instead of a lambda.
//
// Where an MPC controller would plug in: a new class, e.g.
// MpcController, in its own header/source pair, with the same
// Compute(const ControlInputs&) -> DriveCommand signature. What it will
// need that this class doesn't:
//   - Vehicle state feedback (current speed at minimum, possibly yaw
//     rate) -- ControlInputs doesn't carry this yet (see its own header
//     comment for why not); adding it means both extending that struct
//     AND wiring a new subscription in control.cpp's main() (localization
//     already tracks velocity internally -- see ekf.cpp's m_x(3)/m_x(4)
//     -- but doesn't currently publish it on any topic; that's a
//     prerequisite, not something this controller-swapping work unlocks
//     by itself).
//   - A vehicle dynamics/kinematics model and a prediction horizon --
//     purely internal to MpcController, no interface changes needed for
//     those.
//   - Actuation constraints (max yaw rate, max speed change per cycle) --
//     could live as MpcController's own constexpr constants, same
//     pattern kSpeed/kLookaheadDistance already use here.
class PurePursuitController
{
public:
    // No longer const: tracks consecutive empty-path cycles across calls
    // (see pure_pursuit_controller.cpp's kSweepYawRate comment) so a
    // persistent gap can escalate from a straight creep to a sweeping turn
    // instead of staring down the same wrong heading forever. Every other
    // call still only reads inputs and this object's own constants -- no
    // gz-transport/global state involved, so this stays trivially testable
    // and control.cpp's single long-lived `controller` instance is exactly
    // the right lifetime for this counter to persist across cycles in.
    DriveCommand Compute(const ControlInputs &inputs);

private:
    int m_consecutiveEmptyCycles = 0;

    // Universal stuck-detection state -- see kStuckCyclesBeforeLatch's
    // comment in the .cpp for why this exists (a confirmed real deadlock:
    // physically wedged against a cone, so no forward/turning command --
    // NORMAL path-following included, not just the empty-path creep/sweep
    // -- could ever produce real displacement) and why it has to run
    // regardless of which branch of Compute() is otherwise active.
    // m_anchor{X,Y} is the world position the CURRENT displacement window
    // is measured from; it rolls forward to the latest position every time
    // real progress is confirmed (see Compute()), so "stuck" means "hasn't
    // meaningfully moved in the last kStuckCyclesBeforeLatch cycles",
    // not "hasn't moved since some arbitrary fixed point in the past".
    bool m_haveAnchor = false;
    double m_anchorX = 0.0;
    double m_anchorY = 0.0;
    int m_cyclesSinceAnchor = 0;
    // Total Compute() calls since process start -- see kStartupGraceCycles's
    // own comment in the .cpp for why the stuck-watchdog needs this: without
    // it, the anchor above starts timing from the first valid pose, which
    // arrives before the rest of the pipeline (perception/planning) has
    // actually warmed up, and a confirmed real false-trigger during that
    // window permanently disables the car for the rest of the process's
    // life (m_permanentlyStuck never resets).
    int m_totalCycles = 0;
    // One-way latch: once the stuck watchdog fires, STAYS stopped forever
    // (this process's lifetime) rather than resuming normal driving after
    // the log-throttle reset below. Confirmed live (2026-09-01) as a real,
    // not theoretical, gap: without this, m_cyclesSinceAnchor resetting to
    // 0 purely to throttle repeated [STUCK] log spam ALSO silently let the
    // very next cycle fall through to normal path-following again (neither
    // the "real progress" nor the "still stuck" branch matched right after
    // a reset), so a genuinely, permanently wedged car oscillated between
    // one cycle of a real stop and ~kStuckCyclesBeforeLatch cycles of
    // full normal driving commands, forever -- never actually holding
    // position the way the log message ("holding position, needs external
    // reset") claimed. See Compute()'s own comment at the trigger site.
    bool m_permanentlyStuck = false;
    // Independent, LONGER-timescale stuck check, alongside the short-term
    // rolling-anchor one above -- confirmed live (2026-09-01) as
    // necessary, not redundant: the short-term anchor ROLLS FORWARD any
    // time displacement crosses kStuckDistanceThreshold (0.3m), which is
    // exactly right for not penalizing genuine driving, but is vulnerable
    // to small NOISE/WOBBLE (e.g. wheels spinning against a real physical
    // block, rocking the chassis) randomly walking past 0.3m from a
    // constantly-chasing anchor before kStuckCyclesBeforeLatch cycles
    // ever elapse -- confirmed directly: ground truth AND /estimated_pose
    // both agreed the car sat in the same ~0.3m patch for 120+ seconds,
    // continuously commanding ~4.8 m/s forward, while the short-term
    // watchdog never fired even once. This check instead snapshots
    // position only every kLongTermStuckWindow cycles (not continuously
    // rolling), so genuine noise can't perpetually reset it -- it only
    // resets when the car has ACTUALLY covered real ground since the last
    // snapshot.
    bool m_haveLongTermAnchor = false;
    double m_longTermAnchorX = 0.0;
    double m_longTermAnchorY = 0.0;
    int m_cyclesSinceLongTermAnchor = 0;
    // +1 or -1, multiplies kSweepYawRate. Always +1 currently (2026-09-02):
    // the escalating-reverse-retry logic that used to flip this sign
    // between attempts (so a sweep re-driving the car back into the same
    // obstacle every loop could try the OTHER direction) was removed along
    // with all reverse-attempt code, since reverse driving is disallowed
    // under FS rules and was already dead-code hard-stopped before this
    // (see the SAFETY OVERRIDE comment at Compute()'s stuck-watchdog site).
    // Kept as a member (not simplified to a constant) as an obvious hook if
    // some other, rules-compliant reason to flip the sweep direction shows
    // up later.
    double m_sweepDirection = 1.0;

    // The speed COMMANDED last cycle, used to compute THIS cycle's
    // lookahead distance (see kLookahead* constants in the .cpp) -- a
    // deliberate one-cycle delay, not a bug: lookahead distance picks the
    // target waypoint, which determines curvature, which determines speed,
    // so "lookahead depends on speed" and "speed depends on lookahead's
    // pick" can't both be resolved within the same cycle without an
    // artificial iteration. Using the PREVIOUS cycle's already-computed
    // speed instead is the standard way real pure-pursuit implementations
    // break this exact cycle; at this pipeline's cycle rate, speed changes
    // gradually enough between consecutive cycles that a one-cycle-stale
    // value is a negligible approximation. Initialized to 0.0 (not
    // kMinSpeed, which lives in the .cpp's anonymous namespace and isn't
    // reachable from this header) -- the lookahead formula clamps to
    // kMinLookahead regardless, so 0.0 already produces the same
    // smallest-most-cautious-lookahead result on the very first cycle.
    // Smoothed cycle-to-cycle (EMA, see kSpeedSmoothingAlpha in the .cpp)
    // -- NOT the raw min(reactiveSpeed, previewSpeed) result, which is
    // used directly to compute this same cycle's own commanded speed. See
    // kSpeedSmoothingAlpha's own comment for why raw was insufficient once
    // kBrakePreviewDistance's forward-preview scan was added: confirmed
    // live as visible target-distance jitter (2.3-5.1m swings cycle to
    // cycle) traced to the preview scan's own max-over-window statistic
    // reacting to single-cycle path-shape noise from the reactive
    // pipeline's own live recompute, which then fed back into THIS cycle's
    // lookahead distance and made it jitter too.
    double m_lastSpeed = 0.0;

    // Same EMA treatment as m_lastSpeed, applied to yawRate (2026-09-04,
    // user report: "a lot of jitter in the planned path that's causing
    // oscillations"). yawRate was previously computed fresh every cycle
    // from raw curvature (speed*curvature, no smoothing at all) -- unlike
    // speed, which already gets this same treatment. Root cause of the
    // jitter itself: /planned_path is body-frame, recomputed from a stable
    // WORLD-frame source every cycle via WorldToBody using the CURRENT
    // pose estimate -- so ordinary pose-estimation noise (a few cm of x/y,
    // a fraction of a degree of yaw) gets baked directly into the
    // published path's own body-frame coordinates even when the
    // underlying world-frame line hasn't moved, and can flip which
    // waypoint the lookahead search picks as the reactive target cycle to
    // cycle. Re-framing /planned_path itself to world frame was considered
    // and rejected -- pure pursuit's own curvature math fundamentally needs
    // the vehicle at the origin, so that would require control.cpp to
    // subscribe to pose itself and redo this geometry, a much bigger
    // change than smoothing the output. Initialized to 0.0 for the same
    // reason m_lastSpeed is -- a straight-line command on the very first
    // cycle is the safe default regardless.
    double m_lastYawRate = 0.0;
};
