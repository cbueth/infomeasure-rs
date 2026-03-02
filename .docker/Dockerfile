# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Pre-built image with Rust dependencies for infomeasure-rs
# Build with: --build-arg RUST=stable or --build-arg RUST=beta

ARG RUST=stable
FROM ghcr.io/rust-lang/rust:latest

# Set the Rust version (stable or beta)
RUN rustup default ${RUST}

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
ENV PATH="/root/.local/bin:$PATH"

# Pre-install Rust tools
RUN rustup component add rustfmt clippy

# Set up workspace for dependency pre-building
WORKDIR /build
# copy full source
COPY . .

# Pre-build dependencies (including tests/benchmarks to caching everything)
RUN cargo build --workspace

# Remove source code to keep image minimal. Only cargo cache remaing
RUN rm -rf src tests benches

# Install Python dependencies for validation tests
WORKDIR /build/tests/validation_crate
RUN uv venv && uv pip install -r requirements.txt

WORKDIR /build
