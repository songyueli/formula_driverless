#!/usr/bin/env bash
# Installs the native OS-level dependencies build.sh's cmake invocation
# needs (Gazebo Jetty + C++ toolchain + OpenCV) -- mirrors the Dockerfile's
# own apt steps exactly, so this and the Docker image never drift apart.
# Everything else the build needs (Foxglove SDK, ONNX Runtime, Eigen) is
# already fetched automatically by CMakeLists.txt via FetchContent, so
# nothing further to install for those.
#
# Idempotent and safe to call on every build: the fast-path check below
# skips straight past apt entirely (and any sudo prompt) once dependencies
# are already present, so normal rebuild iteration isn't slowed down or
# interrupted asking for a password every time.
#
# Only meaningful on a Debian/Ubuntu host with apt-get -- a Jetson running
# natively, an x86 Linux dev machine, or already inside the Docker
# container itself (where this just hits the fast-path immediately, since
# the Dockerfile already installed everything at image-build time). On
# macOS there's no apt-get and no native Gazebo Jetty package at all --
# see scripts/run.sh, the Docker container is the only option there.
#
# Run from the repo root (also called automatically by build.sh, so this
# normally never needs to be run by hand):
#   ./scripts/install_native_deps.sh

set -e

if command -v gz &>/dev/null && command -v cmake &>/dev/null \
   && dpkg -s libopencv-dev &>/dev/null 2>&1; then
  exit 0
fi

if ! command -v apt-get &>/dev/null; then
  echo "error: no apt-get found -- native builds need a Debian/Ubuntu host" >&2
  echo "  (Jetson, or an x86 Linux machine). On macOS, use ./scripts/run.sh" >&2
  echo "  to build and run inside the Docker container instead." >&2
  exit 1
fi

echo "Installing native dependencies (Gazebo Jetty, build tools, OpenCV)..."

# Step 1: prerequisites for adding the OSRF apt repository -- mirrors
# Dockerfile's own Step 1.
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  curl \
  ca-certificates \
  lsb-release \
  gnupg

# Step 2: add the OSRF Gazebo repository -- mirrors Dockerfile's Step 2.
# dpkg --print-architecture returns "arm64" on the Jetson, so the correct
# package variant is selected automatically. Skipped if already added, so
# re-running this script doesn't recreate/duplicate the sources file.
if [ ! -f /etc/apt/sources.list.d/gazebo-stable.list ]; then
  curl https://packages.osrfoundation.org/gazebo.gpg \
    --output /tmp/pkgs-osrf-archive-keyring.gpg
  sudo install -m 0644 /tmp/pkgs-osrf-archive-keyring.gpg \
    /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  rm /tmp/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
fi

# Step 3: install Gazebo Jetty + C++ build tools -- mirrors Dockerfile's
# Step 3. gz-jetty is a meta-package that pulls in gz-sim, gz-transport,
# gz-msgs, and all their dependencies in one shot.
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  gz-jetty \
  build-essential \
  cmake \
  git \
  libopencv-dev

echo "Native dependencies installed."
