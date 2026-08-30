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
#include <gz/msgs/uint64.pb.h>

#include <common/scoped_timer.hpp>
#include "cone_color.hpp"
#include "ekf.hpp"
#include "geodetic.hpp"

// Localization process
// ---------------------
// Fuses the car's sensors into a joint vehicle-pose + cone-landmark
// estimate via EKF-SLAM (see ekf.hpp for the full state definition, motion
// model, and why this is SLAM rather than localization-against-a-known-
// map). This is an ESTIMATE, not ground truth -- unlike /vehicle
// (foxglove_bridge), which is driven directly from Gazebo's own true pose,
// this is only as good as the sensors and the filter, same as it would be
// on the real car. The landmark map is built ENTIRELY from what the car
// itself has observed -- nothing here reads Gazebo's ground-truth cone
// positions (that was the pre-SLAM version; see git history for
// cone_map.hpp/cpp, which this replaces).
//
// Inputs (subscribe):
//   /ground_speed    gz.msgs.Odometry  (Sensoric-emulated ground speed
//                    sensor -- body-frame vx/vy, see fsd_car/model.sdf)
//   /imu             gz.msgs.IMU       (VN-300-emulated IMU --
//                    angular_velocity.z used as a yaw_rate measurement)
//   /gnss/front,     gz.msgs.NavSat    (VN-300-emulated dual-antenna GNSS
//   /gnss/rear                         -- each antenna's lat/lon is a
//                                        position fix; the pair together
//                                        gives an absolute GNSS-compass
//                                        heading fix)
//   /cone_detections gz.msgs.Pose_V    (perception's localized cone
//                    detections, body frame -- matched against this
//                    process's OWN growing landmark map, or added as a new
//                    landmark if unmatched; see Ekf::CorrectOrAddLandmark)
//
// Output (publish):
//   /estimated_pose      gz.msgs.Pose    (x, y, yaw)
//   /estimated_landmarks gz.msgs.Pose_V  (every tracked landmark's current
//                        estimated WORLD position -- unlike
//                        /cone_detections, which is per-frame and body-
//                        frame, this is the filter's persistent, growing
//                        map, republished after each /cone_detections
//                        batch)
//
// Deliberately NOT yet included (next step, not done here):
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

// Max allowed sim-time gap between the front and rear antenna fixes used
// for a single heading correction. Confirmed as a real, live failure (not
// theoretical): direct instrumentation of ApplyCorrection on an 8-minute
// run caught 3 separate ~180-degree heading corrections (innovation
// magnitude ~2.5-3.05 rad, essentially pi) landing at near-unity gain
// (kGnssHeadingStddev is tiny -- the filter treats this measurement as
// nearly ground truth), each immediately preceded by a 17-20m GNSS
// position correction -- the signature of dead-reckoning having drifted
// far off between corrections. Both antennas nominally publish at 400Hz
// (fsd_car/model.sdf), so under normal operation front/rear are never more
// than a few ms apart -- but correctHeadingIfPossible (below) pairs
// whichever fixes are currently cached with NO bound on how old either is.
// If the single-threaded callback executor ever backlogs one topic's queue
// relative to the other (plausible given this project's own confirmed RTF
// stalls -- see kMaxPredictDt's own history in ekf.cpp), a fresh front fix
// can get paired against a rear fix from well before it: with a 1m
// baseline, even a modest staleness gap during a turn is enough to rotate
// the front-minus-rear vector by close to pi. Set to 100ms: two orders of
// magnitude looser than the nominal 2.5ms inter-sample gap (so ordinary
// jitter never blocks a correction), but far tighter than the multi-second
// staleness needed to explain the observed 17-20m drift.
constexpr double kMaxAntennaPairingAgeS = 0.1;

// Every Nth /gnss/front or /gnss/rear message actually applies an EKF
// correction (position + heading); the rest only update predictTo() and
// the cached antenna ENU/timestamp used for staleness checking (both
// cheap -- no O(n^2) work). Confirmed as the live root cause of the
// periodic dt=1.0 predict-clamp events chased after the antenna-staleness
// fix above: `top -H` during an active run showed localization's own
// thread pinned at 99.9% CPU continuously (not gz-sim, not perception),
// and both GNSS antennas plus the IMU are each configured at 400Hz
// (fsd_car/model.sdf) -- unrealistic for real GNSS-compass hardware (real
// RTK/compass units typically output 5-50Hz; 400Hz reads like a copied
// default, not a deliberate spec, unlike kGnssHeadingStddevDeg's own
// datasheet-sourced value). Every correction pays Ekf::ApplyCorrection's
// O(n^2) covariance update regardless of measurement dimension (P has
// nonzero vehicle-landmark cross terms even for a 1D heading measurement,
// so the update is dense over the full n x n matrix -- see that function's
// own comment); with n up to ~166 once landmarks approach
// kMaxActiveLandmarks, two antennas firing 400 corrections/sec each is
// enough on its own to saturate one ARM core, starving gz-transport's
// receive loop -- which is confirmed lossy under backpressure (drops
// rather than queues, see kMaxPredictDt's own comment) -- of the CPU time
// it needs to keep up, producing exactly the message-timestamp gaps that
// then force a large, uncorrected dead-reckoning jump. 20 -> ~20Hz per
// antenna (40Hz combined), comfortably inside real hardware's range, an
// order of magnitude below what was measured saturating the core.
constexpr int kGnssCorrectionThrottle = 20;

// Cone-landmark correction tuning. Unlike the sensor noise figures above,
// there's no datasheet for "how accurate is our own lidar+YOLO pipeline's
// cone localization" -- this is an engineering estimate, not a spec. The
// data-association GATE distance lives in ekf.cpp now (it's an internal
// detail of matching against the filter's own landmark state), not here.
//
// Briefly raised to 0.3 while chasing what looked like landmark
// "overconfidence" (1.3-5m discrepancies failing the Mahalanobis gate
// against a landmark's own nearest active neighbor) -- turned out to be a
// DIFFERENT bug: this specific simulated track has real cones spaced only
// ~1.97m apart (confirmed directly via Gazebo's own scene service), not
// the >=5m this codebase assumed everywhere, and PruneStaleActiveDuplicates
// (ekf.cpp) was using a 2.0m radius that actively merged genuinely
// DISTINCT, closely-spaced real cones -- the Mahalanobis gate was
// correctly rejecting them as different objects the whole time. Reverted
// back to 0.1 now that the real bug (kDuplicatePruneRadius in ekf.cpp) is
// fixed, to test that fix in isolation rather than have two changes
// confound each other.
//
// That flat 0.1 stayed correct only for NEAR detections -- confirmed
// directly as a real bug during a full-lap test: a long (~15-20m) gap in
// track cone coverage forced the car through a stretch where every
// currently-visible cone sat near lidar_projector.cpp's own kMaxValidRange
// (20m) ceiling, and comparing live /estimated_landmarks against ground
// truth afterward found a cluster of near-duplicate landmark entries
// strung along the car's path through exactly that stretch (58 unmatched
// estimates vs. 28 in an earlier, shorter run -- growing specifically in
// this region), plus matched-pair errors up to 1.9m there vs. ~0.1-0.3m
// typical elsewhere. Root cause: real cone-localization noise (bearing
// error converted to cross-range position error, plus sparser lidar
// returns and smaller/noisier YOLO boxes at distance) grows with range,
// but every detection -- 2m or 20m -- was going into
// Ekf::CorrectOrAddLandmark's Mahalanobis gate with the SAME tight 0.1m
// stddev. At long range that gate was far tighter than the real noise on
// each detection, so repeat sightings of the SAME distant cone kept
// failing to match each other and got added as brand-new landmarks
// instead -- textbook duplicate-spawning from an overconfident R, not a
// data-association logic bug (ekf.cpp's gate itself is correct; it was
// just being fed a wrong noise estimate for the range it applies to).
//
// LandmarkStddev(range) below replaces the flat constant: linear growth
// with range is the standard model for a bearing-based sensor (constant
// ANGULAR precision converts to position error proportional to range), and
// is a small, targeted change -- CorrectOrAddLandmark/AddLandmark already
// take stddev as a per-call parameter, so this only touches what value
// gets passed in, not the EKF's own gating/fusion math. kBase=0.1 keeps
// today's near-field behavior (already validated: ~0.1-0.3m matched-pair
// error in the well-covered parts of the track) unchanged; kRangeCoeff
// chosen so range=20m (the far edge of what lidar_projector.cpp will even
// return, see kMaxValidRange there) gives stddev=0.7m -- loose enough that
// the ~1.6-1.9m real discrepancies observed at that range now fall inside
// a sane multi-sigma gate instead of being rejected outright.
constexpr double kLandmarkBaseStddev = 0.1;        // meters, at range -> 0
constexpr double kLandmarkRangeNoiseCoeff = 0.03;  // additional meters of stddev per meter of range

double LandmarkStddev(double _measuredBodyX, double _measuredBodyY)
{
    const double range = std::hypot(_measuredBodyX, _measuredBodyY);
    return kLandmarkBaseStddev + kLandmarkRangeNoiseCoeff * range;
}

const char *ConeColorName(fsd::ConeColor _color)
{
    switch (_color)
    {
        case fsd::ConeColor::Blue:   return "blue";
        case fsd::ConeColor::Yellow: return "yellow";
        case fsd::ConeColor::Orange: return "orange";
        default:                     return "unknown";
    }
}

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
    auto landmarksPub = node.Advertise<gz::msgs::Pose_V>("/estimated_landmarks");
    // TEMPORARY (rigorous jitter measurement): a SEPARATE topic, not
    // /estimated_landmarks itself -- foxglove_bridge.cpp's DetectionConeSpec
    // does an EXACT string match on name() ("blue"/"yellow"/"orange") to
    // pick each cone's visualization color/size, so encoding a landmark's
    // stable uid (see Ekf::LandmarkEstimate::uid in ekf.hpp) into THAT
    // field would silently make every landmark vanish from the live
    // Foxglove view (DetectionConeSpec returns false on no match, and the
    // caller just skips the cone) instead of just adding debug info.
    // Publishing the SAME data plus a "<color>#<uid>" name on this second
    // topic instead means external tooling can group by uid to track one
    // SPECIFIC physical landmark's position over time unambiguously --
    // /estimated_landmarks itself has no per-entry identity, so measuring
    // "did this landmark's position actually change" from outside used to
    // require fragile nearest-same-color-position matching between polls,
    // which is exactly what breaks down in the dense-landmark regime
    // jitter needs to be measured in. Remove once jitter is fully
    // characterized.
    auto landmarksDebugPub = node.Advertise<gz::msgs::Pose_V>("/estimated_landmarks_debug");

    // Per-cycle compute time (microseconds). Unlike the other 3 processes,
    // localization has 5 distinct callbacks (one per sensor stream) rather
    // than one -- each is its own "cycle", at its own rate and cost (the
    // cone-detection callback does real EKF landmark matching; the others
    // are a predict+correct+publish each), so all 5 publish to the same
    // topic below rather than needing 5 separate ones -- callers just see
    // "how long did localization's last piece of work take", whichever
    // stream triggered it.
    auto timingPub = node.Advertise<gz::msgs::UInt64>("/timing/localization");
    auto publishTiming = [&timingPub](int64_t _us)
    {
        gz::msgs::UInt64 msg;
        msg.set_data(static_cast<uint64_t>(_us));
        timingPub.Publish(msg);
    };

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
        gz::msgs::Pose msg;
        msg.mutable_position()->set_x(ekf.X());
        msg.mutable_position()->set_y(ekf.Y());
        const double halfYaw = ekf.Yaw() / 2.0;
        msg.mutable_orientation()->set_z(std::sin(halfYaw));
        msg.mutable_orientation()->set_w(std::cos(halfYaw));
        posePub.Publish(msg);
    };

    auto publishLandmarks = [&]()
    {
        const auto landmarks = ekf.Landmarks();

        gz::msgs::Pose_V msg;
        for (const auto &lm : landmarks)
        {
            gz::msgs::Pose *p = msg.add_pose();
            p->set_name(ConeColorName(lm.color));
            p->mutable_position()->set_x(lm.x);
            p->mutable_position()->set_y(lm.y);
        }
        landmarksPub.Publish(msg);

        // TEMPORARY (rigorous jitter measurement) -- see landmarksDebugPub's
        // declaration above for why this is a separate topic/message rather
        // than folded into the loop above.
        gz::msgs::Pose_V debugMsg;
        for (const auto &lm : landmarks)
        {
            gz::msgs::Pose *p = debugMsg.add_pose();
            p->set_name(std::string(ConeColorName(lm.color)) + "#" + std::to_string(lm.uid));
            p->mutable_position()->set_x(lm.x);
            p->mutable_position()->set_y(lm.y);
        }
        landmarksDebugPub.Publish(debugMsg);
    };

    // Antenna ENU fixes are cached so a heading correction can be computed
    // whenever EITHER antenna updates, using the other's most recent fix
    // rather than requiring both to arrive at exactly the same instant --
    // but only if that "most recent fix" is actually recent: see
    // kMaxAntennaPairingAgeS's own comment for the confirmed failure a
    // missing staleness bound here causes.
    std::optional<fsd::EnuPosition> lastFrontEnu, lastRearEnu;
    std::optional<double> lastFrontTime, lastRearTime;
    auto correctHeadingIfPossible = [&]()
    {
        if (!lastFrontEnu || !lastRearEnu)
        {
            return;
        }
        if (std::abs(*lastFrontTime - *lastRearTime) > kMaxAntennaPairingAgeS)
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
        fsd::ScopedTimer timer(publishTiming);
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
        fsd::ScopedTimer timer(publishTiming);
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
        fsd::ScopedTimer timer(publishTiming);
        predictTo(StampToSeconds(_msg.header().stamp()));
        const auto enu = geo.ToEnu(_msg.latitude_deg(), _msg.longitude_deg(), _msg.altitude());
        lastFrontEnu = enu;
        lastFrontTime = StampToSeconds(_msg.header().stamp());
        static int throttleCalls = 0;
        if (++throttleCalls % kGnssCorrectionThrottle == 0)
        {
            ekf.CorrectGnssPosition(enu.east, enu.north, kFrontAntennaX, kFrontAntennaY,
                                     kGnssPositionStddev);
            correctHeadingIfPossible();
        }
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
        fsd::ScopedTimer timer(publishTiming);
        predictTo(StampToSeconds(_msg.header().stamp()));
        const auto enu = geo.ToEnu(_msg.latitude_deg(), _msg.longitude_deg(), _msg.altitude());
        lastRearEnu = enu;
        lastRearTime = StampToSeconds(_msg.header().stamp());
        static int throttleCalls = 0;
        if (++throttleCalls % kGnssCorrectionThrottle == 0)
        {
            ekf.CorrectGnssPosition(enu.east, enu.north, kRearAntennaX, kRearAntennaY,
                                     kGnssPositionStddev);
            correctHeadingIfPossible();
        }
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
        fsd::ScopedTimer timer(publishTiming);
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
            // Data association (match vs. new-landmark) happens INSIDE the
            // EKF now, against its own tracked landmark state -- not
            // against a ground-truth map, see Ekf::CorrectOrAddLandmark.
            // Stddev scales with range -- see LandmarkStddev's own comment
            // for why a flat constant here was a confirmed, real bug.
            ekf.CorrectOrAddLandmark(pose.position().x(), pose.position().y(), color,
                                      LandmarkStddev(pose.position().x(), pose.position().y()));
        }
        publishLandmarks();
        publishEstimate();
    };
    if (!node.Subscribe("/cone_detections", onConeDetections))
    {
        std::cerr << "Failed to subscribe to /cone_detections\n";
        return 1;
    }

    std::cout << "localization: fusing /ground_speed, /imu, /gnss/{front,rear}, "
              << "/cone_detections -> /estimated_pose, /estimated_landmarks\n";

    gz::transport::waitForShutdown();
    return 0;
}
