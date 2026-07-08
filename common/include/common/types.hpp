#pragma once

#include <vector>
#include <utility>

namespace fsd {

enum class ConeColor {
    Blue,
    Yellow,
    Orange,
    Unknown
};

// A single detected cone in the vehicle's local frame (meters).
struct ConePose {
    float x;
    float y;
    ConeColor color;
};

// Estimated vehicle pose in the world frame.
struct VehiclePose {
    float x;        // meters
    float y;        // meters
    float heading;  // radians, 0 = +X axis, CCW positive
};

// Sequence of (x, y) waypoints in the world frame.
struct PlannedPath {
    std::vector<std::pair<float, float>> waypoints;
};

} // namespace fsd
