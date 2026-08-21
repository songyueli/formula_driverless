#include "pure_pursuit_controller.hpp"

#include <algorithm>
#include <cmath>

namespace
{
// Straight-line target speed -- confirmed to hold up at full active-landmark
// capacity (kMaxActiveLandmarks=80) once the EKF-side scaling fixes (sparse
// corrections, spatial-grid retired-landmark search) were in place.
constexpr double kMaxSpeed = 3.0;  // m/s
// Floor speed for the tightest corners a live path ever produces -- never
// fully stops for a curve, only for an empty path (see the early-return
// below).
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

    const double speed = std::clamp(kMaxSpeed - kCurvatureSpeedGain * std::abs(curvature),
                                     kMinSpeed, kMaxSpeed);
    const double yawRate = speed * curvature;

    return DriveCommand{speed, yawRate};
}
