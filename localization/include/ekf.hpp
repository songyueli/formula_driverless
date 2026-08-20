#pragma once

#include <vector>

#include <Eigen/Dense>

#include <common/types.hpp>

// EKF-SLAM: jointly estimates the vehicle's own pose/velocity AND the
// world-frame positions of every cone landmark it has observed, as one
// correlated state vector -- not vehicle-only localization against a
// separately-known map (that was the pre-SLAM version of this class; see
// git history).
//
// State = [vehicle (6): x, y, yaw, vx_body, vy_body, yaw_rate,
//          landmark_0 (2): x, y,
//          landmark_1 (2): x, y,
//          ...]
//
// Vehicle sub-state semantics are unchanged from before: position/yaw in
// the world frame, velocity in the vehicle's own body frame. Landmarks are
// always world-frame and, being cones, assumed static -- Predict() only
// evolves the vehicle sub-state; landmark rows/columns of the covariance
// are otherwise untouched by prediction (their process noise is exactly
// zero, since they don't move).
//
// The state grows dynamically as new landmarks are discovered
// (CorrectOrAddLandmark), which is why this uses Eigen::VectorXd/MatrixXd
// throughout rather than fixed-size types.
//
// What makes this SLAM rather than "localization with a map": landmark
// positions are STATE VARIABLES with their own uncertainty, correlated
// with the vehicle's pose and with each other through the full covariance
// matrix (not block-diagonal). Re-observing a landmark corrects both its
// own estimate AND, through that correlation, the vehicle's pose -- most
// visibly on loop closure (driving back past cones seen early in a lap),
// where accumulated drift can partially collapse in one update instead of
// only ever being corrected by GNSS.
namespace fsd
{

class Ekf
{
public:
    static constexpr int kVehicleStateDim = 6;

    struct LandmarkEstimate
    {
        double x;
        double y;
        ConeColor color;
    };

    Ekf();

    // Advances the state by _dt seconds using the constant body-velocity /
    // constant yaw-rate motion model for the vehicle sub-state; landmarks
    // are untouched (they don't move). Safe to call with _dt <= 0 (no-op).
    void Predict(double _dt);

    // Ground-speed sensor: measures vx_body/vy_body directly.
    void CorrectBodyVelocity(double _vx, double _vy, double _stddevVx, double _stddevVy);

    // IMU gyro Z axis: measures yaw_rate directly.
    void CorrectYawRate(double _yawRate, double _stddev);

    // A single GNSS antenna's position, already converted to local ENU
    // meters (see geodetic.hpp). _antennaOffset{X,Y} is that antenna's
    // known mount position in the vehicle's body frame.
    void CorrectGnssPosition(double _measuredEast, double _measuredNorth,
                              double _antennaOffsetX, double _antennaOffsetY,
                              double _stddev);

    // Dual-antenna GNSS-compass heading: an absolute yaw measurement.
    void CorrectHeading(double _measuredYaw, double _stddev);

    // A cone detection in the vehicle's own body frame, from
    // /cone_detections. Internally: try to match it (Euclidean-gated, see
    // kLandmarkGateDist in ekf.cpp) against an existing SAME-COLOR tracked
    // landmark; if found, apply a joint vehicle+landmark correction (both
    // get updated, see the class comment above); if not, add it as a new
    // landmark, growing the state by 2 via a proper covariance-
    // augmentation Jacobian (not just a bare append with a guessed
    // uncertainty -- see AddLandmark in ekf.cpp for why that matters).
    void CorrectOrAddLandmark(double _measuredBodyX, double _measuredBodyY,
                               ConeColor _color, double _stddev);

    double X() const { return m_x(0); }
    double Y() const { return m_x(1); }
    double Yaw() const { return m_x(2); }

    std::vector<LandmarkEstimate> Landmarks() const;

private:
    void CorrectMatchedLandmark(int _landmarkIndex, double _measuredBodyX,
                                 double _measuredBodyY, double _stddev);
    void AddLandmark(double _measuredBodyX, double _measuredBodyY,
                      ConeColor _color, double _stddev);

    Eigen::VectorXd m_x;
    Eigen::MatrixXd m_P;
    // m_landmarkColors[i] corresponds to state indices
    // kVehicleStateDim + 2*i (x) and kVehicleStateDim + 2*i + 1 (y).
    std::vector<ConeColor> m_landmarkColors;
};

} // namespace fsd
