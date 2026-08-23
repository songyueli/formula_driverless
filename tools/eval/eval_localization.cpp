// Ground-truth accuracy evaluator -- NOT part of the autonomy pipeline
// itself, a standalone diagnostic tool. Subscribes to Gazebo's own
// ground-truth entity poses (/world/<world>/pose/info, the SAME topic
// visualization/src/foxglove_bridge.cpp reads for its ghost overlay) plus
// this project's own perception/localization outputs
// (/cone_detections, /estimated_pose, /estimated_landmarks_debug), and
// continuously reports how far each one is from ground truth. Exists
// because "is the SLAM stack accurate enough for path planning" can't be
// answered by reading the filter's own reported covariance (that's the
// filter grading its own homework) -- it has to be checked against the
// simulator's own true entity poses, which only Gazebo (not this process)
// actually knows.
//
// Two error metrics, matching the two accuracy requirements this tool was
// built to check:
//   1. Per-detection error: /cone_detections is perception's raw, PER-FRAME,
//      body-frame cone localization (camera+lidar fusion only -- no EKF
//      involved). To grade this WITHOUT the EKF's own pose error leaking
//      in, each detection is transformed to world coordinates using
//      Gazebo's own ground-truth vehicle pose (not /estimated_pose) at the
//      time of the message, then matched to the nearest same-color
//      ground-truth cone. This isolates perception/lidar-projection
//      accuracy from localization accuracy.
//   2. Landmark error: /estimated_landmarks_debug (localization's persistent
//      per-uid map, see ekf.hpp's LandmarkEstimate::uid) is already in
//      world frame and already reflects however many corrections the EKF
//      has fused for that landmark -- compared directly against the
//      nearest same-color ground-truth cone, no transform needed. Each
//      uid's FIRST-seen error (a landmark's position right when it's
//      created, before any re-observation has a chance to refine it) is
//      tracked separately from its LATEST error, so "initial lap" accuracy
//      (first pass, most landmarks freshly created) can be distinguished
//      from "after revisiting" accuracy.
//
// Ground-truth vehicle pose is also compared directly against
// /estimated_pose, as a diagnostic for WHY landmark/detection error might
// be high (a landmark's world position is only as good as the vehicle pose
// used to place it).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <gz/transport/Node.hh>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/pose_v.pb.h>

namespace
{

struct Pose2D
{
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
};

double YawFromQuaternion(double _qw, double _qx, double _qy, double _qz)
{
    return std::atan2(2.0 * (_qw * _qz + _qx * _qy), 1.0 - 2.0 * (_qy * _qy + _qz * _qz));
}

// Ground-truth model-instance name ("cone_blue_12") -> canonical color
// string, matching perception's own YOLO class names ("blue"/"yellow"/
// "orange") so detections and ground truth can be compared on a shared key.
std::optional<std::string> GroundTruthColor(const std::string &_name)
{
    if (_name.rfind("cone_blue", 0) == 0) return "blue";
    if (_name.rfind("cone_yellow", 0) == 0) return "yellow";
    if (_name.rfind("cone_orange", 0) == 0) return "orange";
    return std::nullopt;
}

// perception's YOLO classes -- "large_orange" is still an orange cone
// ground-truth-wise (the sim only ships one orange cone model, see
// foxglove_bridge.cpp's DetectionConeSpec for the same mapping).
std::string DetectionColor(const std::string &_className)
{
    if (_className == "large_orange") return "orange";
    return _className;
}

struct GroundTruthCone
{
    double x, y;
    std::string color;
};

// Running quantile/summary stats over a growing sample -- kept as a plain
// vector and sorted on demand (Report() runs at ~1Hz, sample counts here
// are at most a few thousand for a full lap -- sorting that on report is
// negligible next to the 1s reporting period).
struct ErrorStats
{
    std::vector<double> samples;

    void Add(double _v) { samples.push_back(_v); }

    void Report(const char *_label) const
    {
        if (samples.empty())
        {
            std::printf("  %-24s n=0\n", _label);
            return;
        }
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
        const double median = sorted[sorted.size() / 2];
        const double p95 = sorted[static_cast<size_t>(sorted.size() * 0.95)];
        const double max = sorted.back();
        const size_t overThreshold = static_cast<size_t>(
            std::count_if(sorted.begin(), sorted.end(), [](double v) { return v > 0.1; }));
        std::printf("  %-24s n=%6zu mean=%.3fm median=%.3fm p95=%.3fm max=%.3fm  (>0.10m: %zu, %.1f%%)\n",
                    _label, sorted.size(), mean, median, p95, max, overThreshold,
                    100.0 * overThreshold / sorted.size());
    }
};

} // namespace

int main(int argc, char **argv)
{
    const std::string worldName = argc > 1 ? argv[1] : "trackdrive";
    const std::string posetopic = "/world/" + worldName + "/pose/info";
    const std::string csvPath = argc > 2 ? argv[2] : "/tmp/eval_localization.csv";

    gz::transport::Node node;

    std::mutex mtx;
    std::unordered_map<std::string, GroundTruthCone> groundTruthCones; // accumulated across ticks
    Pose2D groundTruthVehicle;
    bool haveGroundTruthVehicle = false;

    ErrorStats detectionErrors;      // metric 1: per-frame perception error
    ErrorStats landmarkFirstErrors;  // metric 2a: landmark error at first sighting (uid)
    ErrorStats landmarkLatestErrors; // metric 2b: landmark error, most recent estimate
    ErrorStats poseErrorPos;         // diagnostic: /estimated_pose vs ground truth
    ErrorStats poseErrorYawDeg;

    std::unordered_map<uint64_t, double> landmarkFirstErrorByUid; // uid -> error at first sighting
    std::unordered_map<uint64_t, std::string> landmarkColorByUid;

    std::ofstream csv(csvPath, std::ios::trunc);
    csv << "t_wall,kind,error_m,range_m,color,uid,radial_err_m,tangential_err_m\n";
    const auto t0 = std::chrono::steady_clock::now();
    auto nowSec = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    // Nearest same-color ground-truth cone to (x,y); returns nullopt if
    // none of that color has been observed on the ground-truth topic yet.
    auto nearestGroundTruth = [&](double _x, double _y, const std::string &_color)
        -> std::optional<double>
    {
        double best = -1.0;
        for (const auto &entry : groundTruthCones)
        {
            if (entry.second.color != _color) continue;
            const double dx = entry.second.x - _x;
            const double dy = entry.second.y - _y;
            const double d = std::hypot(dx, dy);
            if (best < 0.0 || d < best) best = d;
        }
        return best < 0.0 ? std::nullopt : std::optional<double>(best);
    };

    // Same match, but also returns the matched cone's own (x,y) -- used by
    // onDetections to decompose error into radial (along the detection's
    // own ray from the car) vs. tangential (cross-ray) components, so a
    // range-axis calibration issue can be told apart from a bearing one.
    auto nearestGroundTruthCone = [&](double _x, double _y, const std::string &_color)
        -> std::optional<GroundTruthCone>
    {
        double best = -1.0;
        std::optional<GroundTruthCone> bestCone;
        for (const auto &entry : groundTruthCones)
        {
            if (entry.second.color != _color) continue;
            const double dx = entry.second.x - _x;
            const double dy = entry.second.y - _y;
            const double d = std::hypot(dx, dy);
            if (best < 0.0 || d < best)
            {
                best = d;
                bestCone = entry.second;
            }
        }
        return bestCone;
    };

    std::function<void(const gz::msgs::Pose_V &)> onGtPose =
        [&](const gz::msgs::Pose_V &_msg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (const auto &pose : _msg.pose())
        {
            if (pose.name() == "fsd_car")
            {
                groundTruthVehicle.x = pose.position().x();
                groundTruthVehicle.y = pose.position().y();
                groundTruthVehicle.yaw = YawFromQuaternion(
                    pose.orientation().w(), pose.orientation().x(),
                    pose.orientation().y(), pose.orientation().z());
                haveGroundTruthVehicle = true;
                continue;
            }
            const auto color = GroundTruthColor(pose.name());
            if (!color) continue;
            groundTruthCones[pose.name()] = GroundTruthCone{
                pose.position().x(), pose.position().y(), *color};
        }
    };
    if (!node.Subscribe(posetopic, onGtPose))
    {
        std::fprintf(stderr, "Failed to subscribe to %s\n", posetopic.c_str());
        return 1;
    }

    std::function<void(const gz::msgs::Pose_V &)> onDetections =
        [&](const gz::msgs::Pose_V &_msg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!haveGroundTruthVehicle) return;
        const double cosYaw = std::cos(groundTruthVehicle.yaw);
        const double sinYaw = std::sin(groundTruthVehicle.yaw);
        for (const auto &pose : _msg.pose())
        {
            const std::string color = DetectionColor(pose.name());
            const double bx = pose.position().x();
            const double by = pose.position().y();
            const double range = std::hypot(bx, by);
            const double worldX = groundTruthVehicle.x + bx * cosYaw - by * sinYaw;
            const double worldY = groundTruthVehicle.y + bx * sinYaw + by * cosYaw;
            const auto matched = nearestGroundTruthCone(worldX, worldY, color);
            if (!matched) continue;
            const double err = std::hypot(matched->x - worldX, matched->y - worldY);
            detectionErrors.Add(err);
            // Decompose into radial (along the detection's own ray from the
            // car) vs. tangential (perpendicular to it) -- lets a range-axis
            // calibration/bias issue (radial) be told apart from a
            // bearing/angular one (tangential) instead of only seeing the
            // combined scalar magnitude.
            const double rayAngle = std::atan2(worldY - groundTruthVehicle.y, worldX - groundTruthVehicle.x);
            const double cosRay = std::cos(rayAngle), sinRay = std::sin(rayAngle);
            const double errX = matched->x - worldX, errY = matched->y - worldY;
            const double radialErr = errX * cosRay + errY * sinRay;
            const double tangentialErr = -errX * sinRay + errY * cosRay;
            csv << nowSec() << ",detection," << err << "," << range << "," << color << ","
                << "," << radialErr << "," << tangentialErr << "\n";
        }
    };
    if (!node.Subscribe("/cone_detections", onDetections))
    {
        std::fprintf(stderr, "Failed to subscribe to /cone_detections\n");
        return 1;
    }

    std::function<void(const gz::msgs::Pose_V &)> onLandmarksDebug =
        [&](const gz::msgs::Pose_V &_msg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (const auto &pose : _msg.pose())
        {
            const std::string &name = pose.name(); // "<color>#<uid>"
            const auto hashPos = name.find('#');
            if (hashPos == std::string::npos) continue;
            const std::string color = name.substr(0, hashPos);
            const uint64_t uid = std::strtoull(name.c_str() + hashPos + 1, nullptr, 10);

            const double x = pose.position().x();
            const double y = pose.position().y();
            const auto err = nearestGroundTruth(x, y, color);
            if (!err) continue;

            landmarkLatestErrors.Add(*err);
            csv << nowSec() << ",landmark_latest," << *err << ",,"  << color << "," << uid << "\n";

            if (landmarkFirstErrorByUid.find(uid) == landmarkFirstErrorByUid.end())
            {
                landmarkFirstErrorByUid[uid] = *err;
                landmarkColorByUid[uid] = color;
                landmarkFirstErrors.Add(*err);
                csv << nowSec() << ",landmark_first," << *err << ",," << color << "," << uid << "\n";
            }
        }
    };
    if (!node.Subscribe("/estimated_landmarks_debug", onLandmarksDebug))
    {
        std::fprintf(stderr, "Failed to subscribe to /estimated_landmarks_debug\n");
        return 1;
    }

    // /estimated_pose is a plain gz.msgs.Pose, not Pose_V -- separate
    // subscription/type from the ones above.
    std::function<void(const gz::msgs::Pose &)> onPose =
        [&](const gz::msgs::Pose &_msg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!haveGroundTruthVehicle) return;
        const double ex = _msg.position().x();
        const double ey = _msg.position().y();
        const double eyaw = YawFromQuaternion(_msg.orientation().w(), 0.0, 0.0, _msg.orientation().z());
        const double posErr = std::hypot(ex - groundTruthVehicle.x, ey - groundTruthVehicle.y);
        double yawErr = eyaw - groundTruthVehicle.yaw;
        while (yawErr > M_PI) yawErr -= 2 * M_PI;
        while (yawErr < -M_PI) yawErr += 2 * M_PI;
        poseErrorPos.Add(posErr);
        poseErrorYawDeg.Add(std::abs(yawErr) * 180.0 / M_PI);
        csv << nowSec() << ",pose," << posErr << ",,,\n";
    };
    if (!node.Subscribe("/estimated_pose", onPose))
    {
        std::fprintf(stderr, "Failed to subscribe to /estimated_pose\n");
        return 1;
    }

    std::printf("eval_localization: watching %s, /cone_detections, /estimated_pose, "
                "/estimated_landmarks_debug -- logging to %s\n", posetopic.c_str(), csvPath.c_str());
    std::printf("Requirement: detection error <= 0.10m, initial landmark error <= 0.20m\n\n");

    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::lock_guard<std::mutex> lock(mtx);
        std::printf("=== t=%.1fs  ground-truth cones known=%zu  vehicle_gt=(%.2f,%.2f,%.1fdeg) ===\n",
                    nowSec(), groundTruthCones.size(), groundTruthVehicle.x, groundTruthVehicle.y,
                    groundTruthVehicle.yaw * 180.0 / M_PI);
        poseErrorPos.Report("pose position error");
        poseErrorYawDeg.Report("pose yaw error (deg)");
        detectionErrors.Report("detection error [req<=0.10m]");
        landmarkFirstErrors.Report("landmark 1st-seen err [req<=0.20m]");
        landmarkLatestErrors.Report("landmark latest error");
        std::printf("\n");
        std::fflush(stdout);
        csv.flush();
    }

    return 0;
}
