#!/usr/bin/env bash
# Launches gz sim (headless), the foxglove bridge, perception,
# localization, planning, and control together, and cleans up all six on
# exit. Once running, the car drives itself autonomously (planning/control
# form a closed loop off perception's cone detections, no manual
# /cmd_ackermann commands needed).
#
# Run from inside the fsd_dev container, from the repo root:
#   ./scripts/dev_sim.sh [world_name]
# world_name defaults to "trackdrive" and must match both the filename
# (simulation/worlds/<world_name>.sdf) and its internal <world name="...">.
# This is the ONLY place that name needs to be set — it's threaded through
# to the wait-for-startup check and passed as an argument to foxglove_bridge,
# rather than being duplicated (and silently drifting out of sync, as
# happened before this was consolidated).
#
# Every pipeline stage stays a separate process (not threads in one binary)
# deliberately -- keeps each one free to eventually move to a different
# host/container (e.g. perception running natively on the Jetson for direct
# TensorRT/CUDA access, while the bridge stays in Docker) without having to
# be pulled apart out of a merged process later.
#
# Ctrl+C stops all six.

set -e

WORLD_NAME="${1:-trackdrive}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GZ_SIM_RESOURCE_PATH="${REPO_ROOT}/simulation/models"

WORLD="${REPO_ROOT}/simulation/worlds/${WORLD_NAME}.sdf"
BRIDGE="${REPO_ROOT}/build/foxglove_bridge"
PERCEPTION="${REPO_ROOT}/build/perception"
LOCALIZATION="${REPO_ROOT}/build/localization"
PLANNING="${REPO_ROOT}/build/planning"
CONTROL="${REPO_ROOT}/build/control"

if [ ! -f "$WORLD" ]; then
  echo "error: $WORLD not found" >&2
  exit 1
fi

if [ ! -x "$BRIDGE" ] || [ ! -x "$PERCEPTION" ] || [ ! -x "$LOCALIZATION" ] \
   || [ ! -x "$PLANNING" ] || [ ! -x "$CONTROL" ]; then
  echo "error: build outputs not found — build first with: ./scripts/build.sh" >&2
  exit 1
fi

echo "World: ${WORLD_NAME} (${WORLD})"

# Track child PIDs so all six get cleaned up together (Ctrl+C, error, normal exit).
SIM_PID=""
BRIDGE_PID=""
PERCEPTION_PID=""
LOCALIZATION_PID=""
PLANNING_PID=""
CONTROL_PID=""

cleanup() {
  echo
  echo "Shutting down..."
  [ -n "$CONTROL_PID" ] && kill "$CONTROL_PID" 2>/dev/null
  [ -n "$PLANNING_PID" ] && kill "$PLANNING_PID" 2>/dev/null
  [ -n "$LOCALIZATION_PID" ] && kill "$LOCALIZATION_PID" 2>/dev/null
  [ -n "$PERCEPTION_PID" ] && kill "$PERCEPTION_PID" 2>/dev/null
  [ -n "$BRIDGE_PID" ] && kill "$BRIDGE_PID" 2>/dev/null
  [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "Starting gz sim (headless, GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH)..."
gz sim -s -r "$WORLD" &
SIM_PID=$!

# Poll for the pose topic instead of a blind sleep, so the bridge only
# starts once the world has actually finished loading.
echo "Waiting for gz sim to come up..."
UP=""
for i in $(seq 1 30); do
  if gz topic -l 2>/dev/null | grep -q "/world/${WORLD_NAME}/pose/info"; then
    UP=1
    break
  fi
  sleep 1
done

if [ -z "$UP" ]; then
  echo "error: gz sim didn't come up within 30s" >&2
  exit 1
fi

echo "gz sim is up. Starting foxglove_bridge..."
"$BRIDGE" "$WORLD_NAME" &
BRIDGE_PID=$!

echo "Starting perception..."
"$PERCEPTION" &
PERCEPTION_PID=$!

echo "Starting localization..."
"$LOCALIZATION" &
LOCALIZATION_PID=$!

echo "Starting planning..."
"$PLANNING" &
PLANNING_PID=$!

echo "Starting control..."
"$CONTROL" &
CONTROL_PID=$!

wait "$SIM_PID" "$BRIDGE_PID" "$PERCEPTION_PID" "$LOCALIZATION_PID" "$PLANNING_PID" "$CONTROL_PID"
