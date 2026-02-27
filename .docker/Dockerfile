# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Pre-built image with Rust dependencies for infomeasure-rs
# Build with: --build-arg RUST=stable or --build-arg RUST=beta

ARG RUST=stable
FROM ghcr.io/rust-lang/rust:latest

# Set the Rust version (stable or beta)
RUN rustup default ${RUST}

# Explicitly set cargo directories to ensure artifacts are in a known location
ENV CARGO_HOME=/usr/local/cargo
ENV CARGO_TARGET_DIR=/usr/local/cargo/target

# Use default cargo locations (no override needed)
# Cargo will use /usr/local/cargo for cache and /usr/local/cargo/target for builds

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
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN ln -sf /root/.local/bin/uv /usr/local/bin/uv

# Pre-install Rust tools
RUN rustup component add rustfmt clippy

# Set up workspace for dependency pre-building
WORKDIR /build
# copy full source
COPY . .

# Pre-build multiple profiles for caching in CI
# dev profile - used by cargo build, cargo test
RUN cargo build --workspace

# test profile - used by cargo test (includes test binaries)
RUN cargo build --workspace --tests

# check profile - used by cargo clippy (built via cargo check)
RUN cargo check --workspace

# doc profile - used by cargo doc (built via cargo doc)
RUN cargo doc --no-deps --workspace

# Remove benches to keep image smaller, but keep .cargo for cargo config
RUN rm -rf benches

# Install Python dependencies for validation tests
# Create venv and symlink for validation crate
WORKDIR /build/tests/validation_crate
RUN mkdir -p /root/.venv && uv venv /root/.venv --python python3 && \
    uv pip install --python /root/.venv/bin/python -r requirements.txt && \
    ln -sfn /root/.venv /build/tests/validation_crate/.venv

WORKDIR /build
