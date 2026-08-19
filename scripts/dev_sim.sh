#!/usr/bin/env bash
# Launches gz sim (headless), the foxglove bridge, and the perception
# process together, and cleans up all three on exit.
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
# perception and foxglove_bridge stay separate processes (not threads in one
# binary) deliberately, matching every other pipeline stage -- planning and
# control will follow the same pattern once they're implemented. This also
# keeps perception free to eventually move to a different host/container
# than the bridge (e.g. running natively on the Jetson for direct
# TensorRT/CUDA access, while the bridge stays in Docker) without having to
# be pulled apart out of a merged process later.
#
# Ctrl+C stops all three.

set -e

WORLD_NAME="${1:-trackdrive}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GZ_SIM_RESOURCE_PATH="${REPO_ROOT}/simulation/models"

WORLD="${REPO_ROOT}/simulation/worlds/${WORLD_NAME}.sdf"
BRIDGE="${REPO_ROOT}/build/foxglove_bridge"
PERCEPTION="${REPO_ROOT}/build/perception"

if [ ! -f "$WORLD" ]; then
  echo "error: $WORLD not found" >&2
  exit 1
fi

if [ ! -x "$BRIDGE" ] || [ ! -x "$PERCEPTION" ]; then
  echo "error: build outputs not found — build first with: ./scripts/build.sh" >&2
  exit 1
fi

echo "World: ${WORLD_NAME} (${WORLD})"

# Track child PIDs so all three get cleaned up together (Ctrl+C, error, normal exit).
SIM_PID=""
BRIDGE_PID=""
PERCEPTION_PID=""

cleanup() {
  echo
  echo "Shutting down..."
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

wait "$SIM_PID" "$BRIDGE_PID" "$PERCEPTION_PID"
