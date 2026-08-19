# Development environment for the FSD simulation stack.
# Base: Ubuntu 24.04 (Noble) — required by Gazebo Jetty.
# This image is NOT the deployment target; the Jetson runs this container
# only because its native OS (Ubuntu 20.04 / JetPack) predates Gazebo Jetty.

FROM ubuntu:24.04

# Prevent apt from prompting for timezone or keyboard layout during install.
ENV DEBIAN_FRONTEND=noninteractive

# ── Step 1: prerequisites for adding the OSRF apt repository ─────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    lsb-release \
    gnupg \
 && rm -rf /var/lib/apt/lists/*

# ── Step 2: add the OSRF Gazebo repository ───────────────────────────────────
# This is the official source for all Gazebo releases.
# dpkg --print-architecture returns "arm64" on the Jetson, so the correct
# package variant is selected automatically.
RUN curl https://packages.osrfoundation.org/gazebo.gpg \
      --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) \
      signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
      https://packages.osrfoundation.org/gazebo/ubuntu-stable \
      $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/gazebo-stable.list

# ── Step 3: install Gazebo Jetty + C++ build tools ───────────────────────────
# gz-jetty is a meta-package that pulls in gz-sim, gz-transport, gz-msgs,
# and all their dependencies in one shot.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gz-jetty \
    build-essential \
    cmake \
    git \
    libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
# The repo is mounted here at runtime via --volume; nothing is copied in.
WORKDIR /workspace

CMD ["/bin/bash"]
