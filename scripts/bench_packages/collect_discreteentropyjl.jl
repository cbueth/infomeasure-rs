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
family = get_arg("--family", "main")
states = parse.(Int, split(get_arg("--states", "5,10,25,50,200"), ","))
budget = parse(Float64, get_arg("--budget", "2.0"))
caps = Dict{String,Int}()
for pair in split(get_arg("--caps", "entropy:200,mi:200"), ",")
    kv = split(pair, ":")
    length(kv) == 2 && (caps[String(kv[1])] = parse(Int, kv[2]))
end
cap_for(m) = get(caps, m, 0)
short = get(ENV, "BENCH_SHORT", "0") in ("1", "true", "True")
if short
    warmup_max = 1; warmup_budget = 0.0; min_iters = 1; max_iters = 3; iter_budget = 0.0
else
    warmup_max = parse(Int, get(ENV, "BENCH_WARMUP_MAX", "3"))
    warmup_budget = parse(Float64, get(ENV, "BENCH_WARMUP_BUDGET_S", "0.4"))
    min_iters = parse(Int, get(ENV, "BENCH_MIN_ITERS", "3"))
    max_iters = parse(Int, get(ENV, "BENCH_MAX_ITERS", "10"))
    iter_budget = parse(Float64, get(ENV, "BENCH_ITER_BUDGET_S", "1.5"))
end

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

function benchmark(name, measure, sizes, seeds, data_dir)
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
            w0 = time_ns()
            w = 0
            while true
                f()
                w += 1
                if w >= warmup_max
                    break
                end
                if warmup_budget > 0 && (time_ns() - w0) / 1e9 >= warmup_budget
                    break
                end
            end
            t0 = time_ns()
            k = 0
            while true
                s = time_ns()
                value = f()
                push!(times, (time_ns() - s) / 1e9)
                k += 1
                if k >= max_iters
                    break
                end
                if k >= min_iters && iter_budget > 0 && (time_ns() - t0) / 1e9 >= iter_budget
                    break
                end
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

function time_fn(f)
    w0 = time_ns(); w = 0
    while true
        f(); w += 1
        w >= warmup_max && break
        warmup_budget > 0 && (time_ns() - w0) / 1e9 >= warmup_budget && break
    end
    times = Float64[]; value = 0.0
    t0 = time_ns(); k = 0
    while true
        s = time_ns(); value = f(); push!(times, (time_ns() - s) / 1e9); k += 1
        k >= max_iters && break
        k >= min_iters && iter_budget > 0 && (time_ns() - t0) / 1e9 >= iter_budget && break
    end
    times, value
end

function params_alphabet(states, n)
    p = params(n)
    p["states"] = states
    p["method"] = "mle"
    p
end

function benchmark_alphabet(measure, states_list, sizes, seeds, data_dir, budget)
    results = Dict[]
    for states in states_list
        stopped = false
        for n in sizes
            stopped && break
            times = Float64[]
            value = 0.0
            for seed in seeds
                if measure == "entropy"
                    x = col(read_i32(joinpath(data_dir, "entropy_discrete_b$(states)_s$(seed)_n$(n).bin")), n, 1, 0)
                    f = () -> to_bits(estimate_h(from_data(x, Samples), MaximumLikelihood))
                else
                    flat = read_i32(joinpath(data_dir, "mi_discrete_b$(states)_s$(seed)_n$(n).bin"))
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
                t, v = time_fn(f)
                append!(times, t)
                value = v
            end
            st = stats_dict(times)
            @printf("  %7s b%-4d n=%-6d %9.3f ms\n", measure, states, n, st["mean"] * 1e3)
            push!(results, Dict(
                "id" => "$(measure)/discrete/mle/b$(states)/n$(n)/discreteentropyjl",
                "package" => "discreteentropyjl", "language" => "julia",
                "measure" => measure, "approach" => "discrete",
                "function" => measure == "entropy" ? "estimate_h(..., MaximumLikelihood)" :
                              "mutual_information(..., MaximumLikelihood)",
                "representative" => true,
                "params" => params_alphabet(states, n),
                "statistics" => st, "value" => value, "notes" => nothing,
            ))
            if st["mean"] > budget
                @printf("  -> b%d n=%d exceeded %ss; skipping larger N\n", states, n, budget)
                stopped = true
            end
        end
    end
    results
end

if family == "alphabet"
    alphabet_out = get_arg("--out", joinpath(data_dir, "results", "discreteentropyjl_alphabet.json"))
    benches = vcat(
        benchmark_alphabet("entropy", filter(s -> s <= cap_for("entropy"), states), sizes, seeds, data_dir, budget),
        benchmark_alphabet("mi", filter(s -> s <= cap_for("mi"), states), sizes, seeds, data_dir, budget),
    )
    ameta = Dict(
        "schema" => 2, "run_id" => "fragment_discreteentropyjl_alphabet", "hardware" => nothing,
        "runtime" => Dict("threads" => 1, "adaptive" => !short, "family" => "alphabet", "budget_s" => budget),
        "seeds" => seeds,
        "packages" => [Dict(
            "id" => "discreteentropyjl", "language" => "julia",
            "version" => string(pkgversion(DiscreteEntropy)),
            "limitations" => "MLE only, discrete entropy/MI. No CMI/TE.",
        )],
        "coverage" => [["entropy", "discrete"], ["mi", "discrete"]],
    )
    mkpath(dirname(alphabet_out))
    open(alphabet_out, "w") do io
        JSON.print(io, Dict("meta" => ameta, "benchmarks" => benches), 2)
    end
    println("wrote $(length(benches)) entries to $(alphabet_out)")
    exit(0)
end

benchmarks = vcat(
    benchmark("entropy", "entropy", sizes, seeds, data_dir),
    benchmark("mi", "mi", sizes, seeds, data_dir),
)

meta = Dict(
    "schema" => 2,
    "run_id" => "fragment_discreteentropyjl",
    "hardware" => nothing,
    "runtime" => Dict(
        "threads" => 1, "adaptive" => !short, "warmup_max" => warmup_max,
        "warmup_budget_s" => warmup_budget, "min_iters" => min_iters,
        "max_iters" => max_iters, "iter_budget_s" => iter_budget,
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
