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
};
