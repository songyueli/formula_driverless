# Formula Student Driverless — Simulation & Autonomy Stack

A from-scratch autonomous driving stack for a Formula Student Driverless car,
built on Gazebo (no ROS), gz-transport, and C++17. The car perceives cones
with cameras + lidar, fuses them into a live SLAM map, plans a racing line
from that map, and drives it with pure pursuit — closing the full
perception → localization → planning → control loop with no human input
once launched.

## Project structure

```
simulation/     Gazebo world files (.sdf) and the car model (Ackermann steering)
perception/     Cone detection & color classification (3-camera stitch + YOLO + lidar)
localization/   EKF-SLAM: fused vehicle pose + a live cone-landmark map
planning/       Two path-planning pipelines (reactive + landmark-based racing line)
control/        Pure pursuit steering/speed commands, stuck-watchdog recovery
visualization/  Foxglove bridge — forwards live + debug topics to the Foxglove app
common/         Shared C++ type definitions (header-only)
tools/eval/     Offline localization-accuracy evaluation harness
ml/             Python environment for YOLO cone-detector training (see ml/README.md)
```

## How the autonomy loop fits together

**Perception** stitches the 3 cameras into one cylindrical panorama (uniform
angular resolution, so round cones near the seams don't stretch the way a
planar projection would), runs a YOLO model over it to detect and
color-classify cones (blue/yellow/orange/large_orange), and localizes each
detection in 3D by projecting the matching lidar return onto its bounding
box. Runs on TensorRT (GPU) when available — the Jetson build — falling
back to plain ONNX Runtime (CPU) elsewhere.

**Localization** fuses GNSS (dual-antenna, so it gets an absolute compass
heading, not just position), IMU, a ground-speed sensor, and perception's
localized cone detections into a single EKF-SLAM state: vehicle pose *and*
a live landmark map, estimated jointly. Nothing here reads Gazebo's
ground-truth cone positions — the map is built entirely from what the car
has actually observed, the same as it would have to be on the real car.

**Planning** runs two pipelines side by side, gated by a one-way readiness
latch:
1. *Reactive* — recomputes a short centerline from only the current
   camera/lidar cycle's cone detections, no memory or localization
   dependency. This is the cold-start fallback that drives the car before
   enough of the track has been mapped.
2. *Landmark-based racing line* — once the SLAM map has enough nearby
   landmarks, switches to drawing from the accumulated, world-frame map
   instead: ordered cone midpoints → a densely-sampled spline → a
   per-sample corridor (safety-margined track-width bound) → a
   minimum-curvature racing line solved as a box-constrained QP within that
   corridor. Runs as an open, forward-only window before lap 1 completes,
   then as the full closed loop once it has.

**Control** is pure pursuit in the car's own body frame: picks a lookahead
target off the published path, computes the curvature to reach it, and
scales speed by how much lateral grip that curvature demands (a
friction-circle model shared with a forward-preview braking check that
scans further ahead for an upcoming tight corner). A stuck-watchdog
(short- and long-term) detects genuine lack of progress and recovers with
a creep/widening-spiral search — reverse is never used, matching FS rules.

## Prerequisites

- Ubuntu 24.04 (Noble) — native on a Jetson (JetPack) or an x86 dev machine
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
- OpenCV (`libopencv-dev`) — perception's camera stitching
- `./scripts/install_native_deps.sh` installs the above automatically (its
  own fast-path check makes it a no-op once everything's already present,
  so it's safe to run before every build)

The Foxglove C++ SDK and ONNX Runtime are fetched automatically by CMake,
architecture-aware (arm64 for Jetson, x86_64 for a dev laptop) — nothing
else to install for those two.

## Build

```bash
./scripts/build.sh
```

## Run the full autonomy stack

```bash
./scripts/dev_sim.sh [world_name]   # world_name defaults to "trackdrive"
```

Launches Gazebo (headless) plus perception, localization, planning,
control, and the Foxglove bridge together, and cleans up all of them on
exit (Ctrl+C). Once running, the car drives itself — no manual
`/cmd_ackermann` commands needed. `world_name` must match a file in
`simulation/worlds/` (`empty`, `track`, or `trackdrive`).

Runs the same way natively on a Jetson or an x86 dev machine — nothing here
is container-specific.

## Visualization

Install the [Foxglove app](https://foxglove.dev/download), then run the
bridge (already started for you by `dev_sim.sh`; run it standalone only if
you're driving `/cmd_ackermann` manually against a bare `gz sim` world):
```bash
./build/foxglove_bridge <world_name>
```
Connect Foxglove to `ws://localhost:8765`. Besides the live scene (cones,
vehicle pose, camera panorama, lidar), the bridge exposes the
landmark-based planning pipeline's own intermediate stages as debug
topics — `/planning/debug_midpoints`, `/planning/debug_spline`,
`/planning/debug_corridor_left`/`_right`, and `/planning/debug_racing_line`
— useful for inspecting exactly what the planner is doing at each stage,
not just its final output.

## Status

Perception, localization, planning, and control are all implemented and
closing the loop end-to-end (this replaces an earlier build-in-progress
roadmap that predated all of it). Active areas of ongoing tuning:
racing-line/corridor safety margins near tight corners, cornering-speed
and braking laws, and steering-actuator response — the last of these now
has direct instrumentation (a `[STEER]` commanded-vs-actual trace in
`visualization/src/foxglove_bridge.cpp`) rather than needing to be inferred
from driving behavior alone.
