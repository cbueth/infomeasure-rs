#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""
Generate unified benchmark JSON for the interactive viewer.

Reads a run directory (rust_results.json + python_*.json + metadata.json)
and optional hardware_specs.txt, outputs a flat JSON blob under docs/.

Usage:
    python scripts/generate_benchmark_json.py \
        --run-dir internal/benchmark_results/runs/run_<timestamp>_ \
        --hardware internal/hardware_specs.txt \
        --output docs/benchmark_data.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional


def parse_hardware(path: str) -> Dict[str, str]:
    """Parse system_profiler output into a dict."""
    info = {}
    try:
        with open(path) as f:
            text = f.read()
        patterns = {
            "cpu": r"Chip:\s*(.+)",
            "cores": r"Total Number of Cores:\s*(\d+)",
            "memory": r"Memory:\s*(.+)",
            "model": r"Model Name:\s*(.+)",
            "os": r"System Firmware Version:\s*(.+)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text)
            if m:
                info[key] = m.group(1).strip()
        if "os" in info:
            info.pop("os")  # firmware version, not macOS version
            info["os"] = "macOS"
    except (FileNotFoundError, OSError):
        info = {"cpu": "unknown", "cores": "?", "memory": "?", "os": "?"}
    return info


def add_param(params: Dict[str, Any], key: str, value) -> None:
    """Add a param to the dict, using None for missing."""
    params[key] = value if value is not None else None


def extract_rust_benchmarks(data: Dict, features: str) -> list[Dict]:
    """Convert Rust criterion results to unified format."""
    results = []
    for group_name, benches in data.get("benches", {}).items():
        group_lower = group_name.lower()

        # Skip internal implementation benchmarks
        if "count_frequencies" in group_lower or "slice" in group_lower:
            continue

        # Skip Number of States experiment (different x-axis) and entropy_discrete_small (duplicate)
        if "number of states" in group_lower:
            continue

        # Skip scaling benchmark groups (different x-axis, breaks KL/KSG tab)
        if group_lower.startswith("scaling_"):
            continue

        # Skip kl_nd groups (multi-dimensional KL, different x-axis)
        if "kl_nd" in group_lower or "_nd" in group_lower:
            continue

        # Infer approach from group name
        if "kernel" in group_lower:
            approach = "kernel"
        elif "ordinal" in group_lower:
            approach = "ordinal"
        elif "renyi" in group_lower:
            approach = "renyi"
        elif "tsallis" in group_lower:
            approach = "tsallis"
        elif "ksg" in group_lower or "knn" in group_lower:
            approach = "kl"
        elif "expfam" in group_lower or "kl" in group_lower:
            approach = "kl"
        elif "discrete" in group_lower:
            approach = "discrete"
        else:
            approach = "discrete"

        for bench_key, stats in benches.items():
            params: Dict[str, Any] = {}
            key_lower = bench_key.lower()

            # Parse params from criterion key
            # Size: look for Data Size group, then trailing /N, then N prefix
            sz = _extract_int(key_lower, r"data size/(\d+)")
            if sz is None:
                sz = _extract_int(key_lower, r"/(\d+)(?:/|$)")
            if sz is None:
                sz = _extract_int(key_lower, r"[nN](\d+)_")
            add_param(params, "size", sz)
            add_param(params, "k", _extract_int(key_lower, r"[^k]k(\d+)"))
            add_param(
                params, "bandwidth", _extract_float(key_lower, r"bw_?(\d+(?:_\d+)?)")
            )
            if params["bandwidth"] is not None:
                params["bandwidth"] = float(str(params["bandwidth"]).replace("_", "."))
            add_param(params, "order", _extract_int(key_lower, r"order_?(\d+)"))
            add_param(params, "delay", _extract_int(key_lower, r"delay_?(\d+)"))
            add_param(
                params, "alpha", _extract_float(key_lower, r"alpha(\d+(?:[._]\d+)?)")
            )
            add_param(
                params, "q", _extract_float(key_lower, r"(?:^|[^a])q(\d+(?:[._]\d+)?)")
            )

            add_param(params, "dims", _extract_int(key_lower, r"(\d+)d(?:/|$)"))
            if params.get("dims") is None:
                params["dims"] = 1
            params.pop("history_len", None)

            # Extract correction method for discrete approach only
            method = None
            if approach == "discrete":
                method = "mle"
                # For Rust: group name is authoritative (e.g. entropy_discrete_zhang)
                # For Python: bench name contains the method
                known_methods = [
                    "ansb",
                    "bayes",
                    "bonachela",
                    "chao_shen",
                    "chao_wang_jost",
                    "grassberger",
                    "miller_madow",
                    "nsb",
                    "shrink",
                    "zhang",
                ]
                # Match longer method names first (chao_wang_jost before chao_shen)
                known_methods_sorted = sorted(known_methods, key=len, reverse=True)
                # Check group name first (authoritative for Rust)
                for m in known_methods_sorted:
                    if m in group_lower:
                        method = m
                        break
                # Fall back to bench key search
                if method == "mle":
                    for m in known_methods_sorted:
                        if m in key_lower:
                            method = m
                            break
            params["method"] = method

            # Extract kernel type from bench key if present (e.g. entropy_kernel/box/bw0_1/100)
            # For mi_kernel, te_kernel, cte_kernel, Rust doesn't include kernel type → default "box"
            if approach == "kernel":
                relative = bench_key[len(group_name) :].lstrip("/")
                ktype = relative.split("/")[0]
                if ktype.startswith("bw"):
                    params["kernel_type"] = "box"
                else:
                    # Normalize: strip any bandwidth suffix (e.g. "box_bw0_1" → "box")
                    base = ktype.split("_bw")[0].split("/")[0]
                    if base in ("box", "gaussian"):
                        params["kernel_type"] = base
                    else:
                        params["kernel_type"] = ktype
            else:
                params["kernel_type"] = None

            # Infer measure from group name
            measure = _infer_measure(group_lower, bench_key)

            entry = {
                "id": bench_key,
                "approach": approach,
                "group": group_name,
                "measure": measure,
                "language": "rust",
                "features": features,
                "params": params,
                "statistics": {
                    "mean": stats.get("mean", 0),
                    "stddev": stats.get("stddev", 0),
                    "min": stats.get("min", 0),
                    "max": stats.get("max", 0),
                    "median": stats.get("median", 0),
                    "samples": stats.get("n_samples", 0),
                    "ci_lower": stats.get("ci_lower"),
                    "ci_upper": stats.get("ci_upper"),
                },
            }
            results.append(entry)
    return results


def extract_python_benchmarks(data: Dict) -> list[Dict]:
    """Convert Python benchmark results to unified format."""
    results = []
    for bench in data.get("benchmarks", []):
        params: Dict[str, Any] = {}
        p = bench.get("params", {})

        add_param(
            params, "size", _extract_int(str(bench.get("name", "")), r"/(\d+)(?:/|$)")
        )
        add_param(params, "k", p.get("k"))
        add_param(params, "bandwidth", p.get("bandwidth"))
        kt = p.get("kernel")
        if kt is not None:
            kt = str(kt).split("_bw")[0].split("/")[0]
        add_param(params, "kernel_type", kt)
        add_param(params, "order", p.get("order"))
        add_param(params, "delay", p.get("delay"))
        add_param(params, "alpha", p.get("alpha"))
        add_param(params, "q", p.get("q"))
        add_param(params, "dims", p.get("dims", 1))
        params.pop("history_len", None)

        # Method for discrete approach (default mle, detect from name if present)
        approach_name = str(bench.get("group", "")).lower()
        if approach_name == "discrete":
            method = "mle"
            name_lower = str(bench.get("name", "")).lower()
            for m in [
                "ansb",
                "bayes",
                "bonachela",
                "chao_shen",
                "chao_wang_jost",
                "grassberger",
                "miller_madow",
                "nsb",
                "shrink",
                "zhang",
            ]:
                if m in name_lower:
                    method = m
                    break
            params["method"] = method
        else:
            params["method"] = None

        stats = bench.get("statistics", {})
        entry = {
            "id": bench.get("name", ""),
            "approach": bench.get("group", ""),
            "group": bench.get("group", ""),
            "measure": bench.get("measure", ""),
            "language": "python",
            "features": "",
            "params": params,
            "statistics": {
                "mean": stats.get("mean", 0),
                "stddev": stats.get("stddev", 0),
                "min": stats.get("min", 0),
                "max": stats.get("max", 0),
                "median": stats.get("median", 0),
                "samples": stats.get("samples", len(bench.get("times", []))),
                "ci_lower": stats.get("ci_lower"),
                "ci_upper": stats.get("ci_upper"),
            },
        }
        results.append(entry)
    return results


def _extract_int(text: str, pattern: str) -> Optional[int]:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _extract_float(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text)
    return float(m.group(1).replace("_", ".")) if m else None


def _infer_measure(group_lower: str, bench_key: str) -> str:
    """Infer measure (entropy/mi/cmi/te/cte) from group+key."""
    # Check for measure suffix in group name (handles scaling_kernel_mi, etc.)
    for suffix, measure in [
        ("_cte", "cte"),
        ("_cmi", "cmi"),
        ("_te", "te"),
        ("_mi", "mi"),
    ]:
        if group_lower.endswith(suffix):
            return measure

    # Use group name prefix as primary signal
    prefix = group_lower.split("_")[0] if "_" in group_lower else group_lower
    if prefix in ("cte",):
        return "cte"
    if prefix in ("cmi",):
        return "cmi"
    if prefix in ("te",):
        return "te"
    if prefix in ("mi",):
        return "mi"

    # Fall back to full key search
    kl = group_lower + " " + bench_key.lower()
    if "renyi" in kl or "tsallis" in kl:
        return "entropy"
    return "entropy"


def load_json(path: Path) -> Optional[Dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Warning: {path}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate unified benchmark JSON for viewer"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory (contains rust_results.json + python_*.json)",
    )
    parser.add_argument(
        "--hardware",
        default="internal/hardware_specs.txt",
        help="Path to hardware_specs.txt",
    )
    parser.add_argument(
        "--output",
        default="docs/benchmark_data.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: run directory not found: {run_dir}")
        sys.exit(1)

    hardware = parse_hardware(args.hardware)

    # Read metadata
    meta_path = run_dir / "metadata.json"
    meta = load_json(meta_path) or {}
    features = meta.get("features", [])
    features_str = ",".join(sorted(features)) if features else ""
    gpu = any("gpu" in str(f).lower() for f in features)
    versions = meta.get("versions", {})

    benchmarks = []

    # Rust results
    rust_path = run_dir / "rust_results.json"
    rust_data = load_json(rust_path)
    if rust_data:
        benchmarks.extend(extract_rust_benchmarks(rust_data, features_str))
        print(
            f"  Rust: {sum(1 for b in benchmarks if b['language'] == 'rust')} entries"
        )

    # Python results
    for py_file in sorted(run_dir.glob("python_*.json")):
        py_data = load_json(py_file)
        if py_data:
            benchmarks.extend(extract_python_benchmarks(py_data))
            print(
                f"  Python: {py_file.name} → {sum(1 for b in benchmarks if b['language'] == 'python')} entries (cumulative)"
            )

    if not benchmarks:
        print("Error: no benchmark data found")
        sys.exit(1)

    # Build hardware info
    hardware_info = {
        "cpu": hardware.get("cpu", "unknown"),
        "cores": hardware.get("cores", "?"),
        "memory": hardware.get("memory", "?"),
        "os": hardware.get("os", "?"),
        "gpu": "Apple GPU (M-series)" if gpu else "N/A (CPU only)",
    }

    output = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "run_id": run_dir.name,
            "features": features,
            "hardware": hardware_info,
            "versions": {
                "python": versions.get("python", "unknown"),
                "rustc": versions.get("rustc", "unknown"),
                "infomeasure_python": versions.get("infomeasure_python", "unknown"),
                "infomeasure_rust": versions.get("infomeasure_rust", "unknown"),
            },
        },
        "benchmarks": benchmarks,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Wrote {len(benchmarks)} benchmarks to {output_path}")
    langs = {}
    for b in benchmarks:
        langs[b["language"]] = langs.get(b["language"], 0) + 1
    for lang, count in sorted(langs.items()):
        print(f"    {lang}: {count}")
    print(f"    approaches: {sorted(set(b['approach'] for b in benchmarks))}")
    print(f"    measures: {sorted(set(b['measure'] for b in benchmarks))}")


if __name__ == "__main__":
    main()
