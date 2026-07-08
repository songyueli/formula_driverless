#!/usr/bin/env bash
# Launch the FSD development container.
#
# Run from the repo root:
#   ./scripts/run.sh
#
# What each flag does:
#   -it                 interactive terminal (you get a shell)
#   --rm                delete the container when you exit (keeps things tidy;
#                       your code is safe — it lives on the host, not in the container)
#   --runtime nvidia    give the container access to the Jetson GPU via
#                       nvidia-container-runtime (needed for gpu_lidar later)
#   --network host      container shares the host network stack, so the Foxglove
#                       bridge WebSocket on port 8765 is reachable from your Mac
#                       without any port-mapping setup
#   --volume            mount the repo into /workspace so edits on the host are
#                       instantly visible inside the container and vice versa
#   --name fsd_dev      gives the container a stable name so you can attach to it
#                       from a second terminal with: docker exec -it fsd_dev bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="fsd_sim:latest"

docker run -it --rm \
  --runtime nvidia \
  --network host \
  --volume "${REPO_ROOT}:/workspace" \
  --name fsd_dev \
  "${IMAGE_NAME}"
