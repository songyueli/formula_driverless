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
# Foxglove C++ SDK — prebuilt release binary, picked by host architecture
# (amd64 dev machines vs. arm64 boards like Jetson) so nothing needs to be
# compiled from source on either. A hardcoded x86_64-only URL here silently
# produces an unlinkable arm64 build ("Relocations in generic ELF (EM: 62)")
# rather than a clear error, so this has to be architecture-aware.
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(FOXGLOVE_SDK_TARGET "aarch64-unknown-linux-gnu")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
  set(FOXGLOVE_SDK_TARGET "x86_64-unknown-linux-gnu")
else()
  message(FATAL_ERROR "No prebuilt Foxglove SDK release known for CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
endif()

include(FetchContent)
FetchContent_Declare(
  foxglove_sdk
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL https://github.com/foxglove/foxglove-sdk/releases/download/sdk%2Fv0.24.0/foxglove-v0.24.0-cpp-${FOXGLOVE_SDK_TARGET}.zip
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
