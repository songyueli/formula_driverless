#include "cone_map.hpp"

#include <gz/msgs/scene.pb.h>

namespace fsd
{
namespace
{
ConeColor ConeColorFromModelName(const std::string &_name)
{
    if (_name.rfind("cone_blue", 0) == 0)
    {
        return ConeColor::Blue;
    }
    if (_name.rfind("cone_yellow", 0) == 0)
    {
        return ConeColor::Yellow;
    }
    if (_name.rfind("cone_orange", 0) == 0)
    {
        return ConeColor::Orange;
    }
    return ConeColor::Unknown;
}
} // namespace

std::vector<MapCone> FetchConeMap(gz::transport::Node &_node, const std::string &_sceneService)
{
    std::vector<MapCone> map;

    gz::msgs::Scene sceneMsg;
    bool result = false;
    const bool ok = _node.Request(_sceneService, 5000u, sceneMsg, result);
    if (!ok || !result)
    {
        return map;
    }

    for (const auto &model : sceneMsg.model())
    {
        const ConeColor color = ConeColorFromModelName(model.name());
        if (color == ConeColor::Unknown)
        {
            continue; // not a cone -- the scene also lists fsd_car, ground_plane, etc.
        }
        map.push_back(MapCone{model.pose().position().x(), model.pose().position().y(), color});
    }
    return map;
}

std::optional<MapCone> FindNearestSameColor(const std::vector<MapCone> &_map,
                                             double _worldX, double _worldY,
                                             ConeColor _color, double _maxDist)
{
    std::optional<MapCone> best;
    double bestDistSq = _maxDist * _maxDist;
    for (const auto &cone : _map)
    {
        if (cone.color != _color)
        {
            continue;
        }
        const double dx = cone.x - _worldX;
        const double dy = cone.y - _worldY;
        const double distSq = dx * dx + dy * dy;
        if (distSq < bestDistSq)
        {
            bestDistSq = distSq;
            best = cone;
        }
    }
    return best;
}

ConeColor ConeColorFromClassName(const std::string &_className)
{
    if (_className == "blue")
    {
        return ConeColor::Blue;
    }
    if (_className == "yellow")
    {
        return ConeColor::Yellow;
    }
    if (_className == "orange" || _className == "large_orange")
    {
        // The sim only ships one orange cone model (see foxglove_bridge.cpp's
        // DetectionConeSpec for the same note) -- no ground-truth model to
        // distinguish "orange" from "large_orange" against, so both map here.
        return ConeColor::Orange;
    }
    return ConeColor::Unknown;
}

} // namespace fsd
