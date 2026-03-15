# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Pre-built image with Rust dependencies for infomeasure-rs
# Build with: --build-arg RUST=stable or --build-arg RUST=beta

ARG RUST=stable
FROM ghcr.io/rust-lang/rust:latest

# Set the Rust version (stable or beta)
RUN rustup default ${RUST}

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

# Pre-build dependencies (including tests/benchmarks to caching everything)
RUN cargo build --workspace

# Remove benches and local cargo config to keep image smaller and avoid conflicts
# Cargo cache at default location (/usr/local/cargo/target) will be preserved
RUN rm -rf benches .cargo

# Install Python dependencies for validation tests
# Create venv and symlink for validation crate
WORKDIR /build/tests/validation_crate
RUN mkdir -p /root/.venv && uv venv /root/.venv --python python3 && \
    uv pip install --python /root/.venv/bin/python -r requirements.txt && \
    ln -sfn /root/.venv /build/tests/validation_crate/.venv

WORKDIR /build
