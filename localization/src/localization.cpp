#include <cmath>
#include <functional>
#include <iostream>
#include <optional>
#include <string>

#include <gz/transport/Node.hh>
#include <gz/msgs/imu.pb.h>
#include <gz/msgs/navsat.pb.h>
#include <gz/msgs/odometry.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/time.pb.h>

#include "cone_map.hpp"
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
//   /cone_detections gz.msgs.Pose_V   (perception's localized cone
//                     detections, body frame -- matched against the known
//                     track map fetched at startup via the scene service,
//                     see cone_map.hpp, for an additional position
//                     correction independent of GNSS)
//
// Deliberately NOT yet included (next step, not done here):
//   - AckermannSteering's own /model/fsd_car/odometry (redundant with
//     /ground_speed + /imu for now; also its position/orientation are
//     anchored to an arbitrary local start frame, not the world frame, so
//     it's not usable as-is for an absolute correction anyway)

namespace
{
// Hardcoded rather than a CLI arg (unlike foxglove_bridge's world_name)
// since this whole project only ever runs one world -- see dev_sim.sh.
constexpr const char *kWorldName = "trackdrive";

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

// Cone-landmark correction tuning. Unlike the sensor noise figures above,
// there's no datasheet for "how accurate is our own lidar+YOLO pipeline's
// cone localization" -- this is an engineering estimate, not a spec.
constexpr double kLandmarkStddev = 0.1;    // meters
constexpr double kLandmarkGateDist = 1.0;  // meters -- see FindNearestSameColor

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

    const std::string sceneService = std::string("/world/") + kWorldName + "/scene/info";
    const auto coneMap = fsd::FetchConeMap(node, sceneService);
    std::cout << "localization: fetched " << coneMap.size()
              << " known cone position(s) from " << sceneService << '\n';

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

    std::function<void(const gz::msgs::Pose_V &)> onConeDetections =
        [&](const gz::msgs::Pose_V &_msg)
    {
        // No predictTo() here: perception never sets Pose_V's header stamp
        // (see perception.cpp), and the other three sensor streams already
        // keep the state's prediction reasonably current between detection
        // frames arriving at the camera's ~30Hz.
        for (const auto &pose : _msg.pose())
        {
            const fsd::ConeColor color = fsd::ConeColorFromClassName(pose.name());
            if (color == fsd::ConeColor::Unknown)
            {
                continue;
            }

            // Re-fetched every iteration (not hoisted above the loop) so a
            // detection later in the SAME message benefits from an earlier
            // one's correction already having tightened the state estimate,
            // rather than every detection in this frame transforming off
            // one stale pre-loop snapshot.
            const auto &x = ekf.State();
            const double yaw = x(2);
            const double bodyX = pose.position().x();
            const double bodyY = pose.position().y();
            const double worldX = x(0) + bodyX * std::cos(yaw) - bodyY * std::sin(yaw);
            const double worldY = x(1) + bodyX * std::sin(yaw) + bodyY * std::cos(yaw);

            const auto matched = fsd::FindNearestSameColor(coneMap, worldX, worldY, color,
                                                             kLandmarkGateDist);
            if (!matched)
            {
                continue; // no map cone nearby -- skip rather than risk a bad association
            }
            ekf.CorrectLandmark(bodyX, bodyY, matched->x, matched->y, kLandmarkStddev);
        }
        publishEstimate();
    };
    if (!node.Subscribe("/cone_detections", onConeDetections))
    {
        std::cerr << "Failed to subscribe to /cone_detections\n";
        return 1;
    }

    std::cout << "localization: fusing /ground_speed, /imu, /gnss/{front,rear}, "
              << "/cone_detections -> /estimated_pose\n";

    gz::transport::waitForShutdown();
    return 0;
}
