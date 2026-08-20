#include "cone_color.hpp"

namespace fsd
{

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
        // The sim only ships one orange cone model -- no ground-truth
        // model to distinguish "orange" from "large_orange" against, so
        // both map here (see foxglove_bridge.cpp's DetectionConeSpec for
        // the same note).
        return ConeColor::Orange;
    }
    return ConeColor::Unknown;
}

} // namespace fsd
