#!/usr/bin/env Rscript
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Cross-package collector: RTransferEntropy (R).
#
# Covers Shannon transfer entropy (calc_te). RTransferEntropy uses a
# quantile/binned discrete estimator, so it is a "nearest equivalent" to the
# infomeasure/JIDT discrete TE, not the identical plug-in estimator. Writes a
# schema-v2 fragment (results/rtransferentropy.json).

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  i <- match(name, args)
  if (is.na(i) || i == length(args)) default else args[i + 1]
}

data_dir <- get_arg("--data-dir", Sys.getenv("BENCH_DATA_DIR", "target/bench-data"))
out <- get_arg("--out", file.path(data_dir, "results", "rtransferentropy.json"))
sizes <- as.integer(strsplit(get_arg("--sizes", "100,400"), ",")[[1]])
seeds <- as.numeric(strsplit(get_arg("--seeds", ""), ",")[[1]])
warmup <- as.integer(get_arg("--warmup", "3"))
iterations <- as.integer(get_arg("--iterations", "10"))

suppressMessages(library(RTransferEntropy))
set_quiet(TRUE)

read_i32 <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  readBin(con, integer(), n = file.info(path)$size / 4L, size = 4L, endian = "little")
}

stats_json <- function(times) {
  n <- length(times)
  m <- mean(times)
  sdv <- if (n > 1) sd(times) else 0
  half <- 1.96 * sdv / sqrt(n)
  sprintf(
    paste0('{"mean":%.12g,"stddev":%.12g,"min":%.12g,"max":%.12g,',
           '"median":%.12g,"samples":%d,"ci_lower":%.12g,"ci_upper":%.12g}'),
    m, sdv, min(times), max(times), median(times), n, m - half, m + half
  )
}

entries <- character(0)
for (n in sizes) {
  times <- numeric(0)
  value <- NA_real_
  for (seed in seeds) {
    path <- file.path(data_dir, sprintf("te_discrete_s%d_n%d.bin", as.integer(seed), n))
    flat <- read_i32(path)
    x <- flat[seq(1, length(flat), by = 2)]
    y <- flat[seq(2, length(flat), by = 2)]
    for (w in seq_len(warmup)) invisible(calc_te(x, y))
    for (it in seq_len(iterations)) {
      t0 <- proc.time()[["elapsed"]]
      value <- calc_te(x, y)
      times <- c(times, proc.time()[["elapsed"]] - t0)
    }
  }
  cat(sprintf("  %7s %-14s n=%-6d %9.3f ms\n", "te", "discrete", n, mean(times) * 1e3))
  entries <- c(entries, sprintf(paste0(
    '{"id":"te/discrete/n%d/rtransferentropy","package":"rtransferentropy",',
    '"language":"r","measure":"te","approach":"discrete","function":"calc_te",',
    '"params":{"n":%d,"k":null,"bandwidth":null,"order":null,"delay":1,"alpha":null,',
    '"q":null,"dims":1,"method":"mle","kernel_type":null},',
    '"statistics":%s,"value":%.12g,',
    '"notes":"Quantile-binned discrete estimator; nearest equivalent to the plug-in TE."}'
  ), n, n, stats_json(times), value))
}

seeds_json <- paste(sprintf("%d", as.integer(seeds)), collapse = ",")
fragment <- sprintf(
  '{"meta":{"schema":2,"run_id":"fragment_rtransferentropy","hardware":null,"runtime":{"threads":1,"warmup":%d,"iterations":%d,"short":%s},"seeds":[%s],"packages":[{"id":"rtransferentropy","language":"r","version":"%s","limitations":"Shannon TE only (quantile-binned discrete estimator); no entropy/MI/CMI, no KSG/kernel/ordinal, no Rényi/Tsallis."}],"coverage":[["te","discrete"]]},"benchmarks":[%s]}',
  warmup, iterations, if (warmup <= 1) "true" else "false", seeds_json,
  as.character(packageVersion("RTransferEntropy")),
  paste(entries, collapse = ",")
)

dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
writeLines(fragment, out)
cat(sprintf("wrote %d entries to %s\n", length(entries), out))
