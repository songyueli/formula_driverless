#include <cmath>
#include <functional>
#include <iostream>
#include <optional>

#include <gz/transport/Node.hh>
#include <gz/msgs/imu.pb.h>
#include <gz/msgs/navsat.pb.h>
#include <gz/msgs/odometry.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/time.pb.h>

#include "ekf.hpp"
#include "geodetic.hpp"

// Localization process
// ---------------------
// Fuses the car's sensors into a single pose/velocity estimate via a 6-state
// EKF (see ekf.hpp for the state definition and motion model). This is an
// ESTIMATE, not ground truth -- unlike /vehicle (foxglove_bridge), which is
// driven directly from Gazebo's own true pose, this is only as good as the
// sensors and the filter, same as it would be on the real car.
//
// Inputs (subscribe):
//   /ground_speed   gz.msgs.Odometry  (Sensoric-emulated ground speed sensor
//                   -- body-frame vx/vy, see fsd_car/model.sdf)
//   /imu            gz.msgs.IMU       (VN-300-emulated IMU -- angular_velocity.z
//                   used as a yaw_rate measurement)
//   /gnss/front,    gz.msgs.NavSat    (VN-300-emulated dual-antenna GNSS --
//   /gnss/rear                        each antenna's lat/lon is a position
//                                      fix; the pair together gives an
//                                      absolute GNSS-compass heading fix)
//
// Output (publish):
//   /estimated_pose gz.msgs.Pose      (x, y, yaw -- see PublishEstimate())
//
// Deliberately NOT yet included (next steps, not done here):
//   - Cone-landmark corrections against a known track map (needs the map
//     extracted from trackdrive.sdf first, plus a data-association step --
//     a separate, substantial piece of work)
//   - AckermannSteering's own /model/fsd_car/odometry (redundant with
//     /ground_speed + /imu for now; also its position/orientation are
//     anchored to an arbitrary local start frame, not the world frame, so
//     it's not usable as-is for an absolute correction anyway)

namespace
{
// Must match simulation/worlds/trackdrive.sdf's <spherical_coordinates>.
constexpr double kRefLatDeg = 32.9;
constexpr double kRefLonDeg = -117.1;
constexpr double kRefAltM = 100.0;

// GNSS antenna mount offsets in the body frame (forward, left), meters --
// must match fsd_car/model.sdf's gnss_front/gnss_rear sensor <pose> x/y.
constexpr double kFrontAntennaX = 0.5, kFrontAntennaY = 0.0;
constexpr double kRearAntennaX = -0.5, kRearAntennaY = 0.0;

// Measurement noise stddevs used as this filter's R -- must match the
// noise actually configured on each sensor in fsd_car/model.sdf (in the
// sensor's own native/measured units, not necessarily the units the noise
// is applied in internally -- see model.sdf's GNSS comment block for why
// that distinction matters for the navsat sensors specifically).
constexpr double kGroundSpeedStddev = 0.02;    // m/s
constexpr double kGyroZStddev = 0.0012;        // rad/s
constexpr double kGnssPositionStddev = 0.01;   // meters (RTK Fixed)
constexpr double kGnssHeadingStddevDeg = 0.15; // VN-300 static GNSS-compass spec
constexpr double kGnssHeadingStddev = kGnssHeadingStddevDeg * M_PI / 180.0;

double StampToSeconds(const gz::msgs::Time &_stamp)
{
    return static_cast<double>(_stamp.sec()) + static_cast<double>(_stamp.nsec()) * 1e-9;
}
} // namespace

int main()
{
    gz::transport::Node node;
    fsd::Ekf ekf;
    const fsd::GeodeticConverter geo(kRefLatDeg, kRefLonDeg, kRefAltM);

    auto posePub = node.Advertise<gz::msgs::Pose>("/estimated_pose");

    // Predict() needs elapsed SIM time, not wall time -- this sim runs well
    // under real-time under the current sensor load (measured ~0.1x
    // real-time factor with 3 cameras + lidar + YOLO all running), so wall
    // time would badly overstate how far the constant-velocity motion model
    // should be trusted to extrapolate.
    std::optional<double> lastTime;
    auto predictTo = [&](double _now)
    {
        if (lastTime)
        {
            ekf.Predict(_now - *lastTime);
        }
        lastTime = _now;
    };

    auto publishEstimate = [&]()
    {
        const auto &x = ekf.State();
        gz::msgs::Pose msg;
        msg.mutable_position()->set_x(x(0));
        msg.mutable_position()->set_y(x(1));
        const double halfYaw = x(2) / 2.0;
        msg.mutable_orientation()->set_z(std::sin(halfYaw));
        msg.mutable_orientation()->set_w(std::cos(halfYaw));
        posePub.Publish(msg);
    };

    // Antenna ENU fixes are cached so a heading correction can be computed
    // whenever EITHER antenna updates, using the other's most recent fix
    // rather than requiring both to arrive at exactly the same instant.
    std::optional<fsd::EnuPosition> lastFrontEnu, lastRearEnu;
    auto correctHeadingIfPossible = [&]()
    {
        if (!lastFrontEnu || !lastRearEnu)
        {
            return;
        }
        // front - rear points along the vehicle's forward (+X body) axis;
        // atan2(north, east) matches this filter's yaw convention (0 = +X
        // axis, CCW positive) since world X/Y map directly to East/North.
        const double measuredYaw = std::atan2(
            lastFrontEnu->north - lastRearEnu->north,
            lastFrontEnu->east - lastRearEnu->east);
        ekf.CorrectHeading(measuredYaw, kGnssHeadingStddev);
    };

    std::function<void(const gz::msgs::Odometry &)> onGroundSpeed =
        [&](const gz::msgs::Odometry &_msg)
    {
        predictTo(StampToSeconds(_msg.header().stamp()));
        ekf.CorrectBodyVelocity(_msg.twist().linear().x(), _msg.twist().linear().y(),
                                 kGroundSpeedStddev, kGroundSpeedStddev);
        publishEstimate();
    };
    if (!node.Subscribe("/ground_speed", onGroundSpeed))
    {
        std::cerr << "Failed to subscribe to /ground_speed\n";
        return 1;
    }

    std::function<void(const gz::msgs::IMU &)> onImu =
        [&](const gz::msgs::IMU &_msg)
    {
        predictTo(StampToSeconds(_msg.header().stamp()));
        ekf.CorrectYawRate(_msg.angular_velocity().z(), kGyroZStddev);
        publishEstimate();
    };
    if (!node.Subscribe("/imu", onImu))
    {
        std::cerr << "Failed to subscribe to /imu\n";
        return 1;
    }

    std::function<void(const gz::msgs::NavSat &)> onGnssFront =
        [&](const gz::msgs::NavSat &_msg)
    {
        predictTo(StampToSeconds(_msg.header().stamp()));
        const auto enu = geo.ToEnu(_msg.latitude_deg(), _msg.longitude_deg(), _msg.altitude());
        ekf.CorrectGnssPosition(enu.east, enu.north, kFrontAntennaX, kFrontAntennaY,
                                 kGnssPositionStddev);
        lastFrontEnu = enu;
        correctHeadingIfPossible();
        publishEstimate();
    };
    if (!node.Subscribe("/gnss/front", onGnssFront))
    {
        std::cerr << "Failed to subscribe to /gnss/front\n";
        return 1;
    }

    std::function<void(const gz::msgs::NavSat &)> onGnssRear =
        [&](const gz::msgs::NavSat &_msg)
    {
        predictTo(StampToSeconds(_msg.header().stamp()));
        const auto enu = geo.ToEnu(_msg.latitude_deg(), _msg.longitude_deg(), _msg.altitude());
        ekf.CorrectGnssPosition(enu.east, enu.north, kRearAntennaX, kRearAntennaY,
                                 kGnssPositionStddev);
        lastRearEnu = enu;
        correctHeadingIfPossible();
        publishEstimate();
    };
    if (!node.Subscribe("/gnss/rear", onGnssRear))
    {
        std::cerr << "Failed to subscribe to /gnss/rear\n";
        return 1;
    }

    std::cout << "localization: fusing /ground_speed, /imu, /gnss/{front,rear} "
              << "-> /estimated_pose\n";

    gz::transport::waitForShutdown();
    return 0;
}
