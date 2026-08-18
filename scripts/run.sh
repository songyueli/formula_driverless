#!/usr/bin/env bash
# Launch the FSD development container — hardware-agnostic.
#
# Run from the repo root:
#   ./scripts/run.sh
#
# Then, from a second terminal, attach with:
#   docker exec -it fsd_dev bash
#
# What gets auto-detected:
#   GPU runtime  — if Docker has an "nvidia" runtime registered (Jetson or
#                  any Linux box with nvidia-container-toolkit installed),
#                  --runtime nvidia is added so gpu_lidar and rendering use
#                  the real GPU. Otherwise falls back to software rendering
#                  (Mesa/llvmpipe) — works, just slower.
#   Networking   — Linux hosts get --network host (simplest: no port-mapping
#                  needed for the Foxglove bridge). macOS (Docker Desktop)
#                  doesn't support --network host properly, so it falls back
#                  to -p 8765:8765 instead.
#
# What's fixed regardless of hardware:
#   -it --rm             interactive shell, container removed on exit (code
#                         lives on the host via --volume, never in the container)
#   --volume              mounts the repo at /workspace; host edits are
#                         instantly visible inside the container and vice versa
#   --name fsd_dev        stable name so a second terminal can attach via
#                         docker exec -it fsd_dev bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="fsd_sim:latest"
CONTAINER_NAME="fsd_dev"

if ! command -v docker &> /dev/null; then
  echo "error: docker not found — install Docker Desktop (Mac) or Docker Engine (Linux) first." >&2
  exit 1
fi

if ! docker image inspect "${IMAGE_NAME}" &> /dev/null; then
  echo "error: image ${IMAGE_NAME} not found — build it first:" >&2
  echo "  docker build -t ${IMAGE_NAME} ." >&2
  exit 1
fi

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "error: container ${CONTAINER_NAME} is already running — attach to it instead:" >&2
  echo "  docker exec -it ${CONTAINER_NAME} bash" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "error: a stopped container named ${CONTAINER_NAME} already exists — remove it first:" >&2
  echo "  docker rm ${CONTAINER_NAME}" >&2
  exit 1
fi

# --- GPU runtime detection -------------------------------------------------
GPU_ARGS=()
if docker info --format '{{.Runtimes}}' 2>/dev/null | grep -qi nvidia; then
  GPU_ARGS=(--runtime nvidia)
  GPU_DESC="nvidia (GPU passthrough)"
else
  GPU_DESC="none (software rendering — slower, but works)"
fi

# --- Networking detection ---------------------------------------------------
HOST_OS="$(uname -s)"
NET_ARGS=()
if [ "${HOST_OS}" = "Darwin" ]; then
  NET_ARGS=(-p 8765:8765)
  NET_DESC="port-mapped (-p 8765:8765) — Docker Desktop on macOS doesn't support --network host"
else
  NET_ARGS=(--network host)
  NET_DESC="host networking (--network host)"
fi

echo "Host OS:      ${HOST_OS}"
echo "GPU runtime:  ${GPU_DESC}"
echo "Networking:   ${NET_DESC}"
echo

docker run -it --rm \
  "${GPU_ARGS[@]}" \
  "${NET_ARGS[@]}" \
  --volume "${REPO_ROOT}:/workspace" \
  --name "${CONTAINER_NAME}" \
  "${IMAGE_NAME}"
