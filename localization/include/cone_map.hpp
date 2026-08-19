#pragma once

#include <optional>
#include <string>
#include <vector>

#include <gz/transport/Node.hh>

#include <common/types.hpp>

// The track's known cone positions (ground truth), used as landmarks for
// EKF corrections in localization.cpp. Fetched from Gazebo's own resolved
// scene description once at startup (verified empirically: a top-level
// model's scene-service pose matches its known /world/.../pose/info world
// position exactly, e.g. cone_orange_00 at (116.533, 35.314) both ways) --
// this is simulation-only infrastructure, a real deployment would load a
// pre-surveyed track map from a file instead, but the downstream matching
// and correction logic (FindNearestSameColor, Ekf::CorrectLandmark) is
// written the same way either way.
namespace fsd
{

struct MapCone
{
    double x;
    double y;
    ConeColor color;
};

// Queries <world>/scene/info for every cone_{blue,yellow,orange}_* model
// and returns their world-frame positions and colors. Returns an empty
// vector (not an error) if the request fails -- callers should treat that
// the same as "no landmarks available yet", not crash.
std::vector<MapCone> FetchConeMap(gz::transport::Node &_node, const std::string &_sceneService);

// Nearest map cone of the given color to (worldX, worldY), or nullopt if
// none are within _maxDist meters -- the gate that keeps a bad data
// association (wrong cone, or a genuinely novel/unmapped object) from
// corrupting the filter.
std::optional<MapCone> FindNearestSameColor(const std::vector<MapCone> &_map,
                                             double _worldX, double _worldY,
                                             ConeColor _color, double _maxDist);

// Maps perception's raw YOLO class name ("blue"/"yellow"/"orange"/
// "large_orange") to the shared ConeColor enum used both here and for the
// ground-truth map above -- the two naming schemes don't otherwise overlap
// (see foxglove_bridge.cpp's DetectionConeSpec for the same distinction).
ConeColor ConeColorFromClassName(const std::string &_className);

} // namespace fsd
