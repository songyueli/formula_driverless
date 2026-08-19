#!/usr/bin/env bash
# Configures (if needed) and builds the C++ workspace in one step.
#
# Run from inside the fsd_dev container, from the repo root:
#   ./scripts/build.sh
#
# `cmake -B build` is safe to call every time -- it only reconfigures when
# CMakeLists.txt (or the CMakeCache) actually changed, so this doesn't add
# meaningful overhead to rebuilds during normal iteration.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cmake -B "${REPO_ROOT}/build" -S "${REPO_ROOT}"
cmake --build "${REPO_ROOT}/build" -j"$(nproc)"
