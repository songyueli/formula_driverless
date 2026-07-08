cmake_minimum_required(VERSION 3.22)
project(fsd_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Gazebo Jetty dependencies (verified inside the fsd_sim Docker container)
#   gz-sim10       10.4.0
#   gz-transport15 15.0.2
#   gz-msgs12      12.0.1
# ---------------------------------------------------------------------------
find_package(gz-transport15 REQUIRED)
find_package(gz-msgs12      REQUIRED)

# ---------------------------------------------------------------------------
# Shared type definitions (header-only, no compiled library needed)
# ---------------------------------------------------------------------------
include_directories(common/include)

# ---------------------------------------------------------------------------
# Autonomy pipeline executables — one per process
# ---------------------------------------------------------------------------

add_executable(perception    perception/src/perception.cpp)
target_link_libraries(perception
    gz-transport15::gz-transport15
    gz-msgs12::gz-msgs12)

add_executable(localization  localization/src/localization.cpp)
target_link_libraries(localization
    gz-transport15::gz-transport15
    gz-msgs12::gz-msgs12)

add_executable(planning      planning/src/planning.cpp)
target_link_libraries(planning
    gz-transport15::gz-transport15
    gz-msgs12::gz-msgs12)

add_executable(control       control/src/control.cpp)
target_link_libraries(control
    gz-transport15::gz-transport15
    gz-msgs12::gz-msgs12)

add_executable(foxglove_bridge  visualization/src/foxglove_bridge.cpp)
target_link_libraries(foxglove_bridge
    gz-transport15::gz-transport15
    gz-msgs12::gz-msgs12)
# TODO: link against the Foxglove C++ SDK once it is added to the project
