#!/usr/bin/env bash
# Launches gz sim (headless) and the foxglove bridge together, and cleans
# up both processes on exit.
#
# Run from inside the fsd_dev container, from the repo root:
#   ./scripts/dev_sim.sh
#
# Ctrl+C stops both processes.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GZ_SIM_RESOURCE_PATH="${REPO_ROOT}/simulation/models"

WORLD="${REPO_ROOT}/simulation/worlds/empty.sdf"
BRIDGE="${REPO_ROOT}/build/foxglove_bridge"

if [ ! -x "$BRIDGE" ]; then
  echo "error: $BRIDGE not found — build first with: cmake --build build" >&2
  exit 1
fi

# Track child PIDs so both get cleaned up together (Ctrl+C, error, normal exit).
SIM_PID=""
BRIDGE_PID=""

cleanup() {
  echo
  echo "Shutting down..."
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
  if gz topic -l 2>/dev/null | grep -q "/world/empty/pose/info"; then
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
"$BRIDGE" &
BRIDGE_PID=$!

wait "$SIM_PID" "$BRIDGE_PID"
