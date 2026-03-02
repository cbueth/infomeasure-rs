# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Pre-built image with Rust dependencies for infomeasure-rs
# Build with: --build-arg RUST=stable or --build-arg RUST=beta

ARG RUST=stable
FROM rust:${RUST}

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

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY tests/validation_crate/Cargo.toml ./tests/validation_crate/ 2>/dev/null || true
COPY tests/validation_crate/requirements.txt ./tests/validation_crate/ 2>/dev/null || true

RUN mkdir -p src && \
    cargo build --workspace && \
    cargo build --workspace --tests && \
    rustup component add rustfmt clippy

WORKDIR /build/tests/validation_crate
RUN uv venv && uv pip install -r requirements.txt

WORKDIR /build
