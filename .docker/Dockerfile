# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# CI toolchain image for infomeasure-rs: Rust toolchain + system libs + Python env.
# Compiled Rust artifacts are cached on the self-hosted runner (CARGO_TARGET_DIR
# on the /srv/target volume), so this image carries no pre-built cargo cache.
# Build with: --build-arg RUST=stable or --build-arg RUST=beta

ARG RUST=stable
FROM ghcr.io/rust-lang/rust:latest

# Set the Rust version (stable or beta)
RUN rustup default ${RUST}

# Default cargo locations (pipelines override these to the persistent volume)
ENV CARGO_HOME=/usr/local/cargo
ENV CARGO_TARGET_DIR=/usr/local/cargo/target

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    libclang-dev \
    llvm-dev \
    libudev-dev \
    pkg-config \
    libssl-dev \
    curl \
    # GPU (wgpu -> Vulkan via NVK): Vulkan loader + Mesa drivers (NVK) + tools.
    # Requires /dev/dri mounted into the container (see laptop agent volumes).
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN ln -sf /root/.local/bin/uv /usr/local/bin/uv

# Pre-install Rust tools
RUN rustup component add rustfmt clippy

# Install Python dependencies for validation tests
# Create venv and symlink for validation crate
WORKDIR /build/tests/validation_crate
COPY tests/validation_crate/requirements.txt ./
RUN mkdir -p /root/.venv && uv venv /root/.venv --python python3 && \
    uv pip install --python /root/.venv/bin/python -r requirements.txt && \
    ln -sfn /root/.venv /build/tests/validation_crate/.venv

WORKDIR /build
