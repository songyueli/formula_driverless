#include "pure_pursuit_controller.hpp"

namespace
{
constexpr double kSpeed = 3.0;              // m/s, constant for now
constexpr double kLookaheadDistance = 3.0;  // meters
}  // namespace

// Algorithm (classic pure pursuit) -- unchanged from the original
// control.cpp callback:
//   1. Pick the lookahead waypoint: the first path point at or beyond
//      kLookaheadDistance from the car's own origin (falls back to the
//      farthest available point if none are that far -- a sparse/short
//      detected path shouldn't mean no command at all).
//   2. curvature = 2*y / (x^2 + y^2) for a target at body-frame (x, y) --
//      the standard pure pursuit result for the circular arc through the
//      origin, heading along +X, that passes through the target.
//   3. yaw_rate = speed * curvature (curvature = yaw_rate / speed by
//      definition).
//
// TODO 2: curvature-based speed control (slow down for tight corners) --
// constant speed for now, per the original stub's own suggested order.
DriveCommand PurePursuitController::Compute(const ControlInputs &inputs) const
{
    if (inputs.path.empty())
    {
        // No path this cycle -- stop rather than keep driving on a stale
        // command.
        return DriveCommand{0.0, 0.0};
    }

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
    const double yawRate = kSpeed * curvature;

    return DriveCommand{kSpeed, yawRate};
}
