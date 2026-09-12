#!/usr/bin/env julia
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Cross-package collector: DiscreteEntropy.jl (Julia).
#
# Coverage is deliberately MLE only: discrete entropy and mutual information.
# The cost of discrete estimation is dominated by the histogram/plug-in
# structure, and bias corrections (except NSB/ANSB) barely change runtime; the
# full corrected set for infomeasure is on the infomeasure comparison page.
# Writes a schema-v2 fragment (results/discreteentropyjl.json).

using DiscreteEntropy
using JSON
using Printf
using Statistics

get_arg(name, default) = (i = findfirst(==(name), ARGS); i === nothing ? default : ARGS[i + 1])

data_dir = get_arg("--data-dir", get(ENV, "BENCH_DATA_DIR", "target/bench-data"))
out = get_arg("--out", joinpath(data_dir, "results", "discreteentropyjl.json"))
sizes = parse.(Int, split(get_arg("--sizes", "100,400"), ","))
seeds = parse.(Int, split(get_arg("--seeds", ""), ","))
warmup = parse(Int, get_arg("--warmup", "3"))
iterations = parse(Int, get_arg("--iterations", "10"))

read_i32(path) = collect(reinterpret(Int32, read(path)))
col(flat, n, cols, c) = [flat[i * cols + c + 1] for i in 0:(n - 1)]

function stats_dict(times)
    n = length(times)
    m = mean(times)
    sd = n > 1 ? std(times) : 0.0
    half = 1.96 * sd / sqrt(n)
    Dict(
        "mean" => m, "stddev" => sd, "min" => minimum(times), "max" => maximum(times),
        "median" => median(times), "samples" => n,
        "ci_lower" => m - half, "ci_upper" => m + half,
    )
end

params(n) = Dict(
    "n" => n, "k" => nothing, "bandwidth" => nothing, "order" => nothing,
    "delay" => 1, "alpha" => nothing, "q" => nothing, "dims" => 1,
    "method" => "mle", "kernel_type" => nothing,
)

function benchmark(name, measure, sizes, seeds, data_dir, warmup, iterations)
    results = Dict[]
    for n in sizes
        times = Float64[]
        value = 0.0
        for seed in seeds
            if measure == "entropy"
                x = col(read_i32(joinpath(data_dir, "entropy_discrete_s$(seed)_n$(n).bin")), n, 1, 0)
                f = () -> to_bits(estimate_h(from_data(x, Samples), MaximumLikelihood))
            else
                flat = read_i32(joinpath(data_dir, "mi_discrete_s$(seed)_n$(n).bin"))
                x = col(flat, n, 2, 0)
                y = col(flat, n, 2, 1)
                B = maximum(y) + 1
                xy = x .* B .+ y
                f = () -> to_bits(
                    mutual_information(
                        from_data(x, Samples), from_data(y, Samples),
                        from_data(xy, Samples), MaximumLikelihood,
                    ),
                )
            end
            for _ in 1:warmup
                f()
            end
            for _ in 1:iterations
                t0 = time_ns()
                value = f()
                push!(times, (time_ns() - t0) / 1e9)
            end
        end
        st = stats_dict(times)
        @printf("  %7s %-14s n=%-6d %9.3f ms\n", measure, "discrete", n, st["mean"] * 1e3)
        push!(results, Dict(
            "id" => "$(measure)/discrete/n$(n)/discreteentropyjl",
            "package" => "discreteentropyjl",
            "language" => "julia",
            "measure" => measure,
            "approach" => "discrete",
            "function" => measure == "entropy" ? "estimate_h(..., MaximumLikelihood)" :
                          "mutual_information(..., MaximumLikelihood)",
            "params" => params(n),
            "statistics" => st,
            "value" => value,
            "notes" => nothing,
        ))
    end
    results
end

benchmarks = vcat(
    benchmark("entropy", "entropy", sizes, seeds, data_dir, warmup, iterations),
    benchmark("mi", "mi", sizes, seeds, data_dir, warmup, iterations),
)

meta = Dict(
    "schema" => 2,
    "run_id" => "fragment_discreteentropyjl",
    "hardware" => nothing,
    "runtime" => Dict(
        "threads" => 1, "warmup" => warmup, "iterations" => iterations, "short" => warmup <= 1,
    ),
    "seeds" => seeds,
    "packages" => [Dict(
        "id" => "discreteentropyjl", "language" => "julia",
        "version" => string(pkgversion(DiscreteEntropy)),
        "limitations" => "MLE only, discrete entropy/MI. No CMI/TE, no continuous/KSG/kernel. \
        Bias-corrected discrete estimators are intentionally not benchmarked here.",
    )],
    "coverage" => [["entropy", "discrete"], ["mi", "discrete"]],
)

mkpath(dirname(out))
open(out, "w") do io
    JSON.print(io, Dict("meta" => meta, "benchmarks" => benchmarks), 2)
end
println("wrote $(length(benchmarks)) entries to $(out)")
