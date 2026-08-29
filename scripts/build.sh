#!/usr/bin/env bash
# Configures (if needed) and builds the C++ workspace in one step.
#
# Environment-agnostic -- just cmake, works the same whether run natively
# (Jetson host or an x86 laptop, as long as Gazebo Jetty + OpenCV are
# installed) or inside the Docker container. See dev_sim.sh for the same
# note in more detail.
#
# Run from the repo root:
#   ./scripts/build.sh
#
# Installs OS-level dependencies (Gazebo Jetty, build tools, OpenCV) first
# via install_native_deps.sh, so a fresh machine doesn't need to chase them
# down by hand before this can succeed -- that script's own fast-path check
# makes this a no-op (no sudo prompt) once they're already present, so it
# doesn't slow down normal rebuild iteration.
#
# `cmake -B build` is safe to call every time -- it only reconfigures when
# CMakeLists.txt (or the CMakeCache) actually changed, so this doesn't add
# meaningful overhead to rebuilds during normal iteration.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${REPO_ROOT}/scripts/install_native_deps.sh"

cmake -B "${REPO_ROOT}/build" -S "${REPO_ROOT}"
cmake --build "${REPO_ROOT}/build" -j"$(nproc)"
