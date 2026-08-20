#pragma once

#include <string>

#include <common/types.hpp>

// Maps perception's raw YOLO class name ("blue"/"yellow"/"orange"/
// "large_orange") to the shared ConeColor enum -- used by localization to
// match /cone_detections into its own SLAM landmark state (see ekf.hpp).
namespace fsd
{
ConeColor ConeColorFromClassName(const std::string &_className);
} // namespace fsd
