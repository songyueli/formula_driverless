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

    // Universal stuck-detection state -- see kStuckCyclesBeforeReverse's
    // comment in the .cpp for why this exists (a confirmed real deadlock:
    // physically wedged against a cone, so no forward/turning command --
    // NORMAL path-following included, not just the empty-path creep/sweep
    // -- could ever produce real displacement) and why it has to run
    // regardless of which branch of Compute() is otherwise active.
    // m_anchor{X,Y} is the world position the CURRENT displacement window
    // is measured from; it rolls forward to the latest position every time
    // real progress is confirmed (see Compute()), so "stuck" means "hasn't
    // meaningfully moved in the last kStuckCyclesBeforeReverse cycles",
    // not "hasn't moved since some arbitrary fixed point in the past".
    // m_reverseCyclesRemaining counts down an in-progress backward
    // maneuver, checked first in Compute() so it always runs to completion
    // once triggered.
    bool m_haveAnchor = false;
    double m_anchorX = 0.0;
    double m_anchorY = 0.0;
    int m_cyclesSinceAnchor = 0;
    int m_reverseCyclesRemaining = 0;
    // Reverse ATTEMPT number in the current stuck episode -- 0 the first
    // time reverse triggers; increments each time the car gets stuck AGAIN
    // without any real (kStuckDistanceThreshold) progress happening since
    // the previous reverse finished, reset to 0 once real progress DOES
    // happen. Scales reverse duration on each retry -- see kMaxReverse
    // Attempt's comment in the .cpp for why a single fixed-magnitude
    // backup isn't always enough.
    int m_reverseAttempt = 0;
    // Set whenever the "real progress" branch fires in Compute() (see
    // kStuckDistanceThreshold), cleared when a new reverse maneuver
    // starts -- read at the START of the NEXT reverse to decide whether
    // that one is a fresh episode (progress happened since the last
    // reverse) or an escalating retry (it didn't).
    bool m_madeProgressSinceLastReverse = false;
    // +1 or -1, multiplies kSweepYawRate (and kReverseTurnRate) -- flips
    // sign on every escalating reverse retry (see Compute()'s own comment
    // at the reverse-trigger site for why): if the sweep's fixed turn
    // direction happens to be exactly what re-drives the car back into
    // the same obstacle every loop, no amount of extra reverse DISTANCE
    // alone fixes that -- only trying the OTHER direction does.
    double m_sweepDirection = 1.0;
};
