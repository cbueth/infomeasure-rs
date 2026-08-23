#!/bin/sh
# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Locate the NVIDIA Vulkan ICD inside the container and make the Vulkan loader
# use it. The laptop injects the driver libs via the legacy nvidia runtime
# (default-runtime=nvidia, NVIDIA_VISIBLE_DEVICES=all), mounting them at their
# nix store path (NOT a fixed FHS path). The driver's own nvidia_icd.json points
# at libGLX_nvidia.so, which fails in a headless container (no X11) — the NVIDIA
# README says to use libEGL_nvidia.so instead.
#
# This script:
#   1. finds libEGL_nvidia.so under /usr/local/nvidia or the nix store
#   2. writes a minimal ICD json pointing at it
#   3. exports VK_ICD_FILENAMES + LD_LIBRARY_PATH so the loader + deps resolve
#
# Call it in each GPU pipeline step before running vulkan:
#   . /usr/local/bin/init-nvidia-vulkan.sh
#
# Prints the ICD path and fails loudly if the NVIDIA lib is missing (e.g. the
# nvidia runtime did not inject the GPU).

set -eu

# Search order: nvidia-docker 1.0 compatibility dirs, then the nix store.
lib=""
for d in /usr/local/nvidia/lib /usr/local/nvidia/lib64 /nix/store; do
  found="$(find "$d" -maxdepth 4 -name 'libEGL_nvidia.so' 2>/dev/null | head -n1 || true)"
  if [ -n "$found" ]; then
    lib="$found"
    break
  fi
done

if [ -z "$lib" ]; then
  echo "ERROR: libEGL_nvidia.so not found in container (GPU not injected?)" >&2
  echo "  ls /usr/local/nvidia/lib64:" >&2
  ls -la /usr/local/nvidia/lib64 2>&1 >&2 || true
  echo "  nvidia-smi:" >&2
  nvidia-smi 2>&1 >&2 || true
  exit 1
fi

icd="/tmp/nvidia_icd.json"
cat > "$icd" <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "$lib",
    "api_version": "1.3"
  }
}
EOF

export VK_ICD_FILENAMES="$icd"
export LD_LIBRARY_PATH="$(dirname "$lib")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "NVIDIA Vulkan ICD: $lib" >&2
