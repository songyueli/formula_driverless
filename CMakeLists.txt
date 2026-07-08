cmake_minimum_required(VERSION 3.22)
project(fsd_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Gazebo dependencies
# Package names are versioned (e.g. gz-transport13). Adjust the suffix if your
# Gazebo Jetty install uses different version numbers — check with:
#   dpkg -l | grep gz-transport
# ---------------------------------------------------------------------------
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10      REQUIRED)

# ---------------------------------------------------------------------------
# Shared type definitions (header-only, no compiled library needed)
# ---------------------------------------------------------------------------
include_directories(common/include)

# ---------------------------------------------------------------------------
# Autonomy pipeline executables — one per process
# ---------------------------------------------------------------------------

add_executable(perception    perception/src/perception.cpp)
target_link_libraries(perception
    gz-transport13::gz-transport13
    gz-msgs10::gz-msgs10)

add_executable(localization  localization/src/localization.cpp)
target_link_libraries(localization
    gz-transport13::gz-transport13
    gz-msgs10::gz-msgs10)

add_executable(planning      planning/src/planning.cpp)
target_link_libraries(planning
    gz-transport13::gz-transport13
    gz-msgs10::gz-msgs10)

add_executable(control       control/src/control.cpp)
target_link_libraries(control
    gz-transport13::gz-transport13
    gz-msgs10::gz-msgs10)

add_executable(foxglove_bridge  visualization/src/foxglove_bridge.cpp)
target_link_libraries(foxglove_bridge
    gz-transport13::gz-transport13
    gz-msgs10::gz-msgs10)
# TODO: link against the Foxglove C++ SDK once it is added to the project
