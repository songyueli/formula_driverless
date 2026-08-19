#include "ekf.hpp"

#include <cmath>

namespace
{
double NormalizeAngle(double _angle)
{
    while (_angle > M_PI) _angle -= 2.0 * M_PI;
    while (_angle < -M_PI) _angle += 2.0 * M_PI;
    return _angle;
}
} // namespace

namespace fsd
{

Ekf::Ekf()
{
    m_x.setZero();
    // Large initial uncertainty -- deliberately not zero. The first GNSS
    // fix (or any correction) should be trusted almost entirely over this
    // made-up starting guess, and a large P is what makes the Kalman gain
    // do that.
    m_P = StateCov::Identity() * 1.0e6;
}

void Ekf::Predict(double _dt)
{
    if (_dt <= 0.0)
    {
        return;
    }

    const double yaw = m_x(2);
    const double vx = m_x(3);
    const double vy = m_x(4);
    const double yawRate = m_x(5);

    const double cosYaw = std::cos(yaw);
    const double sinYaw = std::sin(yaw);

    StateVec xNext = m_x;
    xNext(0) += (vx * cosYaw - vy * sinYaw) * _dt;
    xNext(1) += (vx * sinYaw + vy * cosYaw) * _dt;
    xNext(2) = NormalizeAngle(yaw + yawRate * _dt);
    // vx, vy, yawRate: constant-velocity assumption between updates -- a
    // real car accelerates and turns, which is exactly what the process
    // noise Q below allows for, rather than the motion model itself.

    StateCov F = StateCov::Identity();
    F(0, 2) = (-vx * sinYaw - vy * cosYaw) * _dt;
    F(0, 3) = cosYaw * _dt;
    F(0, 4) = -sinYaw * _dt;
    F(1, 2) = (vx * cosYaw - vy * sinYaw) * _dt;
    F(1, 3) = sinYaw * _dt;
    F(1, 4) = cosYaw * _dt;
    F(2, 5) = _dt;

    // Process noise: position/yaw have no noise term of their own here --
    // their uncertainty growth comes entirely from propagating F * P * F^T
    // with the velocity/yaw_rate terms below, which is the physically
    // correct way to do it (uncertain velocity held for a longer dt should
    // mean more position uncertainty, and F already encodes that coupling).
    StateCov Q = StateCov::Zero();
    constexpr double kVelNoiseDensity = 0.5;     // (m/s)^2 per second
    constexpr double kYawRateNoiseDensity = 0.1; // (rad/s)^2 per second
    Q(3, 3) = kVelNoiseDensity * _dt;
    Q(4, 4) = kVelNoiseDensity * _dt;
    Q(5, 5) = kYawRateNoiseDensity * _dt;

    m_x = xNext;
    m_P = F * m_P * F.transpose() + Q;
}

void Ekf::CorrectBodyVelocity(double _vx, double _vy, double _stddevVx, double _stddevVy)
{
    Eigen::Matrix<double, 2, kStateDim> H = Eigen::Matrix<double, 2, kStateDim>::Zero();
    H(0, 3) = 1.0;
    H(1, 4) = 1.0;

    const Eigen::Vector2d z(_vx, _vy);
    const Eigen::Vector2d h(m_x(3), m_x(4));
    const Eigen::Vector2d y = z - h;

    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R(0, 0) = _stddevVx * _stddevVx;
    R(1, 1) = _stddevVy * _stddevVy;

    const Eigen::Matrix2d S = H * m_P * H.transpose() + R;
    const Eigen::Matrix<double, kStateDim, 2> K = m_P * H.transpose() * S.inverse();

    m_x += K * y;
    m_x(2) = NormalizeAngle(m_x(2));
    m_P = (StateCov::Identity() - K * H) * m_P;
}

void Ekf::CorrectYawRate(double _yawRate, double _stddev)
{
    Eigen::Matrix<double, 1, kStateDim> H = Eigen::Matrix<double, 1, kStateDim>::Zero();
    H(0, 5) = 1.0;

    const double y = _yawRate - m_x(5);
    const double S = (H * m_P * H.transpose())(0, 0) + _stddev * _stddev;
    const Eigen::Matrix<double, kStateDim, 1> K = m_P * H.transpose() / S;

    m_x += K * y;
    m_x(2) = NormalizeAngle(m_x(2));
    m_P = (StateCov::Identity() - K * H) * m_P;
}

void Ekf::CorrectGnssPosition(double _measuredEast, double _measuredNorth,
                               double _antennaOffsetX, double _antennaOffsetY,
                               double _stddev)
{
    const double yaw = m_x(2);
    const double cosYaw = std::cos(yaw);
    const double sinYaw = std::sin(yaw);

    // h(x): predicted antenna position = chassis position + its known
    // mount offset, rotated into the world frame by the current yaw
    // estimate. Nonlinear in yaw, unlike CorrectBodyVelocity/CorrectYawRate
    // above, so this needs its own Jacobian rather than a fixed H.
    const Eigen::Vector2d h(
        m_x(0) + _antennaOffsetX * cosYaw - _antennaOffsetY * sinYaw,
        m_x(1) + _antennaOffsetX * sinYaw + _antennaOffsetY * cosYaw);
    const Eigen::Vector2d z(_measuredEast, _measuredNorth);
    const Eigen::Vector2d y = z - h;

    Eigen::Matrix<double, 2, kStateDim> H = Eigen::Matrix<double, 2, kStateDim>::Zero();
    H(0, 0) = 1.0;
    H(0, 2) = -_antennaOffsetX * sinYaw - _antennaOffsetY * cosYaw;
    H(1, 1) = 1.0;
    H(1, 2) = _antennaOffsetX * cosYaw - _antennaOffsetY * sinYaw;

    const Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * (_stddev * _stddev);

    const Eigen::Matrix2d S = H * m_P * H.transpose() + R;
    const Eigen::Matrix<double, kStateDim, 2> K = m_P * H.transpose() * S.inverse();

    m_x += K * y;
    m_x(2) = NormalizeAngle(m_x(2));
    m_P = (StateCov::Identity() - K * H) * m_P;
}

void Ekf::CorrectHeading(double _measuredYaw, double _stddev)
{
    Eigen::Matrix<double, 1, kStateDim> H = Eigen::Matrix<double, 1, kStateDim>::Zero();
    H(0, 2) = 1.0;

    const double y = NormalizeAngle(_measuredYaw - m_x(2));
    const double S = (H * m_P * H.transpose())(0, 0) + _stddev * _stddev;
    const Eigen::Matrix<double, kStateDim, 1> K = m_P * H.transpose() / S;

    m_x += K * y;
    m_x(2) = NormalizeAngle(m_x(2));
    m_P = (StateCov::Identity() - K * H) * m_P;
}

} // namespace fsd
