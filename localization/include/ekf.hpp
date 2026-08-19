#pragma once

#include <Eigen/Dense>

// 6-state Extended Kalman Filter for 2D ground-vehicle localization.
//
// State: [x, y, yaw, vx_body, vy_body, yaw_rate]
//   - x, y, yaw are in the WORLD frame (meters, meters, radians;
//     yaw = 0 at +X axis, CCW positive -- matches fsd::VehiclePose).
//   - vx_body, vy_body are in the VEHICLE's own body frame (forward, left)
//     -- this is deliberate, not an oversight: it matches the ground-speed
//     sensor's own measurement frame directly (no rotation needed for that
//     correction), at the cost of needing a yaw-dependent rotation in the
//     motion model to turn body velocity into world-frame position change.
//
// Corrections are applied as they arrive (asynchronous EKF): each sensor
// callback in localization.cpp calls Predict(dt) to catch the state up to
// the new measurement's timestamp, then the matching Correct*() method.
namespace fsd
{

class Ekf
{
public:
    static constexpr int kStateDim = 6;
    using StateVec = Eigen::Matrix<double, kStateDim, 1>;
    using StateCov = Eigen::Matrix<double, kStateDim, kStateDim>;

    Ekf();

    // Advances the state by _dt seconds using the constant body-velocity /
    // constant yaw-rate motion model. Safe to call with _dt <= 0 (no-op) --
    // callers don't need to special-case the first message.
    void Predict(double _dt);

    // Ground-speed sensor: measures vx_body/vy_body directly.
    void CorrectBodyVelocity(double _vx, double _vy, double _stddevVx, double _stddevVy);

    // IMU gyro Z axis: measures yaw_rate directly. Note this is NOT the
    // same as an absolute heading measurement -- see CorrectHeading.
    void CorrectYawRate(double _yawRate, double _stddev);

    // A single GNSS antenna's position, already converted to local ENU
    // meters (see geodetic.hpp) -- east/north map directly to this filter's
    // world x/y (trackdrive.sdf's <spherical_coordinates> uses ENU with
    // heading_deg=0, verified empirically, see model.sdf's GNSS sensor
    // comment block). _antennaOffset{X,Y} is that antenna's known mount
    // position in the vehicle's body frame (see fsd_car/model.sdf), which
    // this accounts for via a nonlinear (yaw-dependent) measurement model
    // rather than a naive direct position substitution.
    void CorrectGnssPosition(double _measuredEast, double _measuredNorth,
                              double _antennaOffsetX, double _antennaOffsetY,
                              double _stddev);

    // Dual-antenna GNSS-compass heading: an ABSOLUTE yaw measurement (unlike
    // CorrectYawRate, which only measures the rate of turning and says
    // nothing about which way the car is actually pointed). _measuredYaw
    // and the internal innovation are wrapped to [-pi, pi] since yaw is
    // periodic.
    void CorrectHeading(double _measuredYaw, double _stddev);

    // A cone landmark: _measuredBodyX/Y is where perception's
    // /cone_detections says the cone is, in the vehicle's own body frame;
    // _landmarkWorldX/Y is that same cone's KNOWN position from the track
    // map (see cone_map.hpp). The measurement model predicts what a
    // body-frame detection of a landmark at that known world position
    // SHOULD look like given the current state estimate -- the mirror
    // image of CorrectGnssPosition's h(x) (which goes body-offset ->
    // world), so this one goes world-landmark -> body-frame prediction.
    void CorrectLandmark(double _measuredBodyX, double _measuredBodyY,
                          double _landmarkWorldX, double _landmarkWorldY,
                          double _stddev);

    const StateVec &State() const { return m_x; }
    const StateCov &Covariance() const { return m_P; }

private:
    StateVec m_x;
    StateCov m_P;
};

} // namespace fsd
