#!/usr/bin/env bash
# One-time setup: builds gz-transport15 + gz-msgs12 (matching what the
# Gazebo Jetty simulator uses) natively on a Jetson-class Ubuntu 20.04
# (focal) host -- i.e. NOT inside the Docker container.
#
# Why this exists: perception needs to run natively (not in Docker) so it
# can use the Jetson's already-correct TensorRT/CUDA install, but it also
# needs to talk to the simulator (which runs in the Ubuntu 24.04 Docker
# container) over gz-transport. gz-transport15/gz-msgs12 aren't packaged
# for Ubuntu 20.04 via apt, and a schema-compatible substitute (Gazebo
# Garden's gz-transport12/gz-msgs9, which IS packaged for focal) was tried
# first -- confirmed via a live packet capture to NOT interoperate with
# transport15 at the discovery-protocol level, despite matching message
# schemas. So this builds the exact matching versions from source instead.
#
# Run from the repo root, on the Jetson itself (not in the container):
#   ./scripts/setup_native_gz.sh
#
# Everything installs to ~/gz_install (not /usr/local), so this never
# touches system package manager state and is safe to re-run or delete
# (rm -rf ~/gz_install ~/gz_src to start over).
#
# Known gotchas baked into this script:
#   - Ubuntu 20.04's stock CMake (3.16) is too old for gz-cmake5 (needs
#     3.22+), so a newer one is installed via pip first.
#   - Ubuntu 20.04's stock protobuf (3.6.1, from 2018) is missing APIs
#     gz-msgs12 actually calls (e.g. SimpleDescriptorDatabase::
#     FindAllMessageNames) -- confirmed via a real compile error, not
#     assumed. protobuf v21.12 is built from source instead (the last
#     release before protobuf made abseil-cpp a hard dependency, which
#     would otherwise add yet another from-source build to this chain).
#   - gz-utils4's optional "log" component needs a newer spdlog than
#     apt's (1.5.0) ships headers for. It's not needed (gz-transport only
#     requires gz-utils' "cli" component), so the build is run with `-k`
#     to skip past that one failure rather than chase a spdlog upgrade.
#   - CLI11 and cppzmq are header-only and unpackaged for focal, so both
#     are built from source too (fast, no real compilation).

set -e

PREFIX="$HOME/gz_install"
SRC="$HOME/gz_src"
CMAKE="$HOME/.local/bin/cmake"

echo "Install prefix: $PREFIX"
echo "Source checkouts: $SRC"
echo

# --- Step 0: newer CMake (gz-cmake5 requires >=3.22, focal ships 3.16) -----
if ! "$CMAKE" --version &>/dev/null; then
  echo "==> Installing CMake via pip (build tool only, no runtime ABI impact)"
  python3 -m pip install --user --upgrade "cmake>=3.22"
fi
echo "==> Using $("$CMAKE" --version | head -1)"

mkdir -p "$SRC"

# --- Step 1: system apt packages that ARE new enough on focal --------------
echo "==> Installing apt dependencies (ZeroMQ, uuid, sqlite3, tinyxml2, eigen3)"
sudo apt-get install -y --no-install-recommends \
  libzmq3-dev uuid-dev libsqlite3-dev libtinyxml2-dev libeigen3-dev

# --- Helper: clone (if missing) + configure + build + install --------------
build_cmake_project() {
  local name="$1" repo="$2" tag="$3"; shift 3
  local extra_args=("$@")
  if [ ! -d "$SRC/$name" ]; then
    echo "==> Cloning $name ($tag)"
    git clone --depth 1 --branch "$tag" "$repo" "$SRC/$name"
  fi
  echo "==> Configuring $name"
  "$CMAKE" -B "$SRC/$name/build" -S "$SRC/$name" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DBUILD_TESTING=OFF \
    "${extra_args[@]}"
  echo "==> Building $name"
  "$CMAKE" --build "$SRC/$name/build" -j"$(nproc)"
  echo "==> Installing $name"
  "$CMAKE" --install "$SRC/$name/build"
}

# --- Step 2: header-only libraries not packaged for focal -------------------
build_cmake_project CLI11 https://github.com/CLIUtils/CLI11.git v2.4.2 \
  -DCLI11_BUILD_TESTS=OFF -DCLI11_BUILD_EXAMPLES=OFF

build_cmake_project cppzmq https://github.com/zeromq/cppzmq.git v4.10.0 \
  -DCPPZMQ_BUILD_TESTS=OFF

# --- Step 3: newer protobuf (see gotcha note above) -------------------------
if [ ! -d "$SRC/protobuf" ]; then
  echo "==> Cloning protobuf v21.12 with submodules"
  git clone --depth 1 --branch v21.12 --recurse-submodules --shallow-submodules \
    https://github.com/protocolbuffers/protobuf.git "$SRC/protobuf"
fi
echo "==> Configuring protobuf"
"$CMAKE" -B "$SRC/protobuf/build" -S "$SRC/protobuf" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -Dprotobuf_BUILD_TESTS=OFF \
  -Dprotobuf_BUILD_SHARED_LIBS=ON \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
echo "==> Building protobuf (slow -- real C++ compile, several minutes on Jetson)"
"$CMAKE" --build "$SRC/protobuf/build" -j"$(nproc)"
echo "==> Installing protobuf"
"$CMAKE" --install "$SRC/protobuf/build"

# --- Step 4: gz-cmake5 -------------------------------------------------------
build_cmake_project gz-cmake https://github.com/gazebosim/gz-cmake.git gz-cmake5

# --- Step 5: gz-utils4 (build with -k -- see spdlog gotcha above) -----------
if [ ! -d "$SRC/gz-utils" ]; then
  echo "==> Cloning gz-utils (gz-utils4)"
  git clone --depth 1 --branch gz-utils4 https://github.com/gazebosim/gz-utils.git "$SRC/gz-utils"
fi
echo "==> Configuring gz-utils4"
"$CMAKE" -B "$SRC/gz-utils/build" -S "$SRC/gz-utils" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX"
echo "==> Building gz-utils4 (continuing past the optional log/spdlog failure)"
"$CMAKE" --build "$SRC/gz-utils/build" -j"$(nproc)" -- -k || true
echo "==> Installing gz-utils4 (also expected to stop partway at the log component)"
"$CMAKE" --install "$SRC/gz-utils/build" || true
echo "    (verify what actually matters is present:)"
find "$PREFIX" -iname "libgz-utils.so*" -o -iname "gz-utils-cli-config.cmake"

# --- Step 6: gz-math9 --------------------------------------------------------
build_cmake_project gz-math https://github.com/gazebosim/gz-math.git gz-math9

# --- Step 7: gz-msgs12 -------------------------------------------------------
build_cmake_project gz-msgs https://github.com/gazebosim/gz-msgs.git gz-msgs12

# --- Step 8: gz-transport15 --------------------------------------------------
build_cmake_project gz-transport https://github.com/gazebosim/gz-transport.git gz-transport15

echo
echo "Done. gz-transport15/gz-msgs12 installed to $PREFIX."
echo "When building perception natively, add to its CMake configure step:"
echo "  -DCMAKE_PREFIX_PATH=$PREFIX"
