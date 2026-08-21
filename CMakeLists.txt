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
# OpenCV -- used by perception for camera-frame stitching (see
# perception/include/camera_stitcher.hpp). apt's libopencv-dev is CPU-only,
# so Stitch() runs on the CPU everywhere for now. GPU acceleration
# (cv::cuda::remap) needs OpenCV built from source with CUDA enabled -- not
# done here; the cv::cuda API surface doesn't even exist in the headers of a
# CPU-only build, so this isn't a runtime toggle, it's a separate rebuild.
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

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
# ONNX Runtime -- prebuilt release binary, picked by host architecture, same
# pattern as the Foxglove SDK above (and same reason: a hardcoded x86_64
# URL would silently produce an unlinkable arm64 build). CPU-only build on
# both architectures -- Microsoft ships a separate prebuilt CUDA build for
# x64 (onnxruntime-linux-x64-gpu_cuda12-*), but NOT for aarch64/Jetson
# (checked the actual release assets, not assumed); real GPU inference on
# the Jetson needs a from-source build, tracked as separate future work.
#
# This release doesn't ship a working CMake config package (its
# onnxruntimeConfig.cmake references files the release tarball doesn't
# actually include -- confirmed by trying find_package() against it, not
# assumed), so its include dir and .so are wired up manually below instead
# of via find_package like the Foxglove SDK. BUILD_RPATH points `perception`
# at the extracted .so location so it's found at runtime without needing
# LD_LIBRARY_PATH set by hand.
# ---------------------------------------------------------------------------
set(ONNXRUNTIME_VERSION "1.29.0")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(ONNXRUNTIME_TARGET "linux-aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
  set(ONNXRUNTIME_TARGET "linux-x64")
else()
  message(FATAL_ERROR "No prebuilt ONNX Runtime release known for CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
endif()

FetchContent_Declare(
  onnxruntime
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${ONNXRUNTIME_TARGET}-${ONNXRUNTIME_VERSION}.tgz
)
FetchContent_MakeAvailable(onnxruntime)

# ---------------------------------------------------------------------------
# Eigen -- header-only, used by localization for the EKF's matrix math
# (state covariance propagation, Kalman gain). Fetched via Populate (not
# MakeAvailable) deliberately: Eigen's own CMakeLists.txt configures a full
# build (tests, docs, optional BLAS/LAPACK detection) that we don't want or
# need for a header-only dependency -- Populate just downloads/extracts the
# source without ever running it.
# ---------------------------------------------------------------------------
FetchContent_Declare(
  eigen
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()

# ---------------------------------------------------------------------------
# Shared type definitions (header-only, no compiled library needed)
# ---------------------------------------------------------------------------
include_directories(common/include)

# ---------------------------------------------------------------------------
# Autonomy pipeline executables — one per process
# ---------------------------------------------------------------------------

add_executable(perception
    perception/src/perception.cpp
    perception/src/camera_stitcher.cpp
    perception/src/cone_detector.cpp
    perception/src/lidar_projector.cpp)
target_include_directories(perception PRIVATE
    perception/include
    ${OpenCV_INCLUDE_DIRS}
    ${onnxruntime_SOURCE_DIR}/include)
target_link_libraries(perception
    gz-transport::gz-transport
    gz-msgs::gz-msgs
    ${OpenCV_LIBS}
    ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so)
set_target_properties(perception PROPERTIES
    BUILD_RPATH "${onnxruntime_SOURCE_DIR}/lib")

add_executable(localization
    localization/src/localization.cpp
    localization/src/ekf.cpp
    localization/src/geodetic.cpp
    localization/src/cone_color.cpp)
target_include_directories(localization PRIVATE
    localization/include
    ${eigen_SOURCE_DIR})
target_link_libraries(localization
    gz-transport::gz-transport
    gz-msgs::gz-msgs)

add_executable(planning
    planning/src/planning.cpp
    planning/src/track_boundaries.cpp
    planning/src/path_generator.cpp)
target_include_directories(planning PRIVATE
    planning/include)
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

# ---------------------------------------------------------------------------
# tegrastats -- Jetson-only system-monitoring tool (part of L4T/JetPack, not
# a library to link against -- foxglove_bridge spawns it as a subprocess and
# parses its stdout). Detected rather than assumed present so the Mac/x86
# dev build (no tegrastats there at all) still compiles; when absent,
# foxglove_bridge simply doesn't publish /jetson_stats.
# ---------------------------------------------------------------------------
find_program(TEGRASTATS_EXECUTABLE tegrastats)
if(TEGRASTATS_EXECUTABLE)
  target_compile_definitions(foxglove_bridge PRIVATE FOXGLOVE_BRIDGE_HAS_TEGRASTATS)
endif()
