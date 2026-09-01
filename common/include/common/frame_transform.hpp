#pragma once

#include <cmath>

// World<->body frame conversion for the landmark-based racing-line planner
// (planning/include/racing_line_optimizer.hpp and friends) -- the first
// real consumer of a shared transform utility in this codebase; nothing
// else here had one (the closest prior art was ad-hoc inline rotation math
// in tools/eval/eval_localization.cpp, not reusable).
//
// Convention matches the rest of this codebase: body +X = forward, +Y =
// left (see control.cpp's ControlInputs::Waypoint doc and pure pursuit's
// curvature sign convention); world frame is whatever ENU-like frame
// localization's EKF operates in (see geodetic.hpp).
namespace fsd
{
struct Pose2D
{
    double x;
    double y;
    double yaw;  // radians
};

// /estimated_pose (gz.msgs.Pose, published by localization.cpp's
// publishEstimate) only ever sets orientation().z()/.w() -- a pure yaw
// rotation, no roll/pitch -- so this only needs the planar z/w form, not a
// general quaternion-to-Euler conversion. yaw = 2*atan2(z, w) is the exact
// inverse of publishEstimate's own z=sin(yaw/2), w=cos(yaw/2).
inline double YawFromPlanarQuaternion(double _qz, double _qw)
{
    return 2.0 * std::atan2(_qz, _qw);
}

struct Point2D
{
    double x;
    double y;
};

// Converts a WORLD-frame point into the BODY frame of a vehicle at
// _vehicleWorldPose. Standard rigid-body inverse transform: translate by
// -vehicle position, then rotate by -yaw.
inline Point2D WorldToBody(const Pose2D &_vehicleWorldPose, double _worldX, double _worldY)
{
    const double dx = _worldX - _vehicleWorldPose.x;
    const double dy = _worldY - _vehicleWorldPose.y;
    const double cosYaw = std::cos(_vehicleWorldPose.yaw);
    const double sinYaw = std::sin(_vehicleWorldPose.yaw);
    return Point2D{dx * cosYaw + dy * sinYaw, -dx * sinYaw + dy * cosYaw};
}

// Inverse of WorldToBody -- included for symmetry/future debug
// visualization use, not required by the racing-line pipeline itself
// (which only ever needs the one world->body conversion right before
// publish).
inline Point2D BodyToWorld(const Pose2D &_vehicleWorldPose, double _bodyX, double _bodyY)
{
    const double cosYaw = std::cos(_vehicleWorldPose.yaw);
    const double sinYaw = std::sin(_vehicleWorldPose.yaw);
    return Point2D{
        _vehicleWorldPose.x + _bodyX * cosYaw - _bodyY * sinYaw,
        _vehicleWorldPose.y + _bodyX * sinYaw + _bodyY * cosYaw};
}
}  // namespace fsd
