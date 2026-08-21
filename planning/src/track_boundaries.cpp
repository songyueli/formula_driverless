#include "track_boundaries.hpp"

namespace fsd
{
TrackBoundaries ColorSplitBoundaries(const std::vector<ClassifiedCone> &cones)
{
    TrackBoundaries boundaries;
    for (const auto &cone : cones)
    {
        if (cone.colorClass == "blue")
        {
            boundaries.left.push_back(cone);
        }
        else if (cone.colorClass == "yellow")
        {
            boundaries.right.push_back(cone);
        }
    }
    return boundaries;
}
}  // namespace fsd
