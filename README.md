# Formula Student Driverless — Simulation & Autonomy Stack

A from-scratch autonomous driving stack for a Formula Student Driverless car,
built on Gazebo (no ROS), gz-transport, and C++17.

## Project structure

```
simulation/     Gazebo world files (.sdf) and car model
perception/     Cone detection & color classification (cameras + lidar)
localization/   Vehicle pose estimation (lidar/IMU/odometry fusion)
planning/       Path planning from cone positions and vehicle pose
control/        Steering and speed commands to the simulated car
visualization/  Foxglove bridge — forwards topics to the Foxglove app
common/         Shared C++ type definitions (header-only)
ml/             Python environment for YOLO training (not part of the C++ build)
```

## Prerequisites

- Ubuntu 24.04 (Noble)
- Gazebo Jetty:
  ```bash
  sudo curl https://packages.osrfoundation.org/gazebo.gpg \
    --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
    https://packages.osrfoundation.org/gazebo/ubuntu-stable \
    $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
  sudo apt-get update && sudo apt-get install gz-jetty
  ```
- CMake >= 3.22, GCC with C++17 support

## Build

```bash
cmake -B build
cmake --build build
```

## Run the simulation

```bash
gz sim simulation/worlds/empty.sdf
```

## Visualization

Install the [Foxglove app](https://foxglove.dev/download), then run the bridge:
```bash
./build/foxglove_bridge
```
Connect Foxglove to `ws://localhost:8765`.

## Build roadmap

- [ ] Step 1 — Empty Gazebo world (ground plane) — confirm it runs
- [ ] Step 2 — Add car model with AckermannSteering plugin, drive manually
- [ ] Step 3 — Add lidar and 3 cameras to the car
- [ ] Step 4 — Build cone track (blue/yellow/orange cones)
- [ ] Step 5 — First gz-transport C++ program: subscribe to a sensor topic
- [ ] Step 6 — Foxglove bridge: live sensor visualization
- [ ] Step 7 — Perception: cone detection and color classification
- [ ] Step 8 — Localization: vehicle pose from sensor fusion
- [ ] Step 9 — Planning + Control: close the autonomy loop
