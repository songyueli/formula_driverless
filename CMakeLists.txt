cmake_minimum_required(VERSION 3.22)
project(fsd_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Gazebo Jetty dependencies (verified inside the fsd_sim Docker container)
#   gz-sim10       10.4.0
#   gz-transport 15.0.2
#   gz-msgs      12.0.1
# ---------------------------------------------------------------------------
find_package(gz-transport REQUIRED)
find_package(gz-msgs      REQUIRED)

# ---------------------------------------------------------------------------
# Foxglove C++ SDK — prebuilt release binary (matches the container's
# x86_64-unknown-linux-gnu target, so nothing needs to be compiled from source)
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
  foxglove_sdk
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL https://github.com/foxglove/foxglove-sdk/releases/download/sdk%2Fv0.24.0/foxglove-v0.24.0-cpp-x86_64-unknown-linux-gnu.zip
)
FetchContent_MakeAvailable(foxglove_sdk)
find_package(foxglove-sdk CONFIG REQUIRED HINTS "${foxglove_sdk_SOURCE_DIR}")
foxglove_sdk_add_cpp_library(foxglove_cpp TYPE STATIC)

# ---------------------------------------------------------------------------
# Shared type definitions (header-only, no compiled library needed)
# ---------------------------------------------------------------------------
include_directories(common/include)

# ---------------------------------------------------------------------------
# Autonomy pipeline executables — one per process
# ---------------------------------------------------------------------------

add_executable(perception    perception/src/perception.cpp)
target_link_libraries(perception
    gz-transport::gz-transport
    gz-msgs::gz-msgs)

add_executable(localization  localization/src/localization.cpp)
target_link_libraries(localization
    gz-transport::gz-transport
    gz-msgs::gz-msgs)

add_executable(planning      planning/src/planning.cpp)
target_link_libraries(planning
    gz-transport::gz-transport
    gz-msgs::gz-msgs)

add_executable(control       control/src/control.cpp)
target_link_libraries(control
    gz-transport::gz-transport
    gz-msgs::gz-msgs)

add_executable(foxglove_bridge  visualization/src/foxglove_bridge.cpp)
target_link_libraries(foxglove_bridge
    gz-transport::gz-transport
    gz-msgs::gz-msgs
    foxglove_cpp)
