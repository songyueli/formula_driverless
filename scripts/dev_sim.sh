#!/usr/bin/env bash
# Launches gz sim (headless), the foxglove bridge, perception,
# localization, planning, and control together, and cleans up all six on
# exit. Once running, the car drives itself autonomously (planning/control
# form a closed loop off perception's cone detections, no manual
# /cmd_ackermann commands needed).
#
# Environment-agnostic on purpose -- no docker exec, no container-only
# paths, nothing here cares whether it's running natively (Jetson host or
# an x86 laptop) or inside the Docker container. It only needs whatever
# environment it's invoked in to already have Gazebo Jetty + the built
# binaries in ./build (see build.sh, same story there). The Jetson runs
# this natively now (its host OS caught up to Ubuntu 24.04, which is what
# Gazebo Jetty needs -- Docker was only ever a workaround for the old
# Ubuntu 20.04 host, see Dockerfile); a laptop without a matching native
# Gazebo Jetty install can still use the Docker container, same script
# either way.
#
# Run from the repo root:
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
# host/container from the others later without having to be pulled apart
# out of a merged process first.
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

# jetson_clocks locks CPU/GPU/EMC to their max frequencies -- only present
# on Jetson hardware (a no-op check on a laptop/CI, where the command
# doesn't exist). Not a nice-to-have: confirmed directly (2026-08-23) that
# a Jetson boots with its GPU at minimum clock (306 of 1300 MHz) until this
# runs, which alone was a 2.5x real-time-factor difference (0.12 -> 0.31)
# for this exact pipeline -- bigger than any camera-resolution change.
# `-n` (non-interactive) so a sudoers misconfiguration fails fast here
# instead of hanging this script on a password prompt; failure is logged
# but not fatal, since the sim is still usable (just slower) without it.
if command -v jetson_clocks >/dev/null 2>&1; then
  echo "Jetson detected -- locking clocks to max via jetson_clocks..."
  sudo -n jetson_clocks || echo "warning: jetson_clocks failed (needs passwordless sudo) -- continuing without it" >&2
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
