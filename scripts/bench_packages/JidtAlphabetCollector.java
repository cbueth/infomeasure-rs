// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Alphabet-scaling collector: JIDT (native Java).
//
// Times the discrete estimators across state counts on the shared
// `\`<measure>_discrete_b<states>_s<seed>_n<n>.bin\`` datasets and writes a
// schema-v2 fragment (`results/jidt_alphabet.json`). Larger N is skipped once a
// cell's mean exceeds `--budget`.
//
// Build/run (inside the bench image):
//   javac -cp "$JIDT_JAR" -d /tmp/jidtcls scripts/bench_packages/JidtAlphabetCollector.java
//   java -cp "$JIDT_JAR:/tmp/jidtcls" JidtAlphabetCollector \
//        --data-dir target/bench-data --states 5,10,200 --sizes 100,400 \
//        --caps entropy:200,cte:50 --budget 2.0 --seeds 1,2

import infodynamics.measures.discrete.ConditionalMutualInformationCalculatorDiscrete;
import infodynamics.measures.discrete.ConditionalTransferEntropyCalculatorDiscrete;
import infodynamics.measures.discrete.EntropyCalculatorDiscrete;
import infodynamics.measures.discrete.MutualInformationCalculatorDiscrete;
import infodynamics.measures.discrete.TransferEntropyCalculatorDiscrete;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class JidtAlphabetCollector {

    static final String[] MEASURES = {"entropy", "mi", "cmi", "te", "cte"};

    static int ncols(String measure) {
        switch (measure) {
            case "entropy": return 1;
            case "mi": case "te": return 2;
            default: return 3;
        }
    }

    static int[] readI32(Path p) throws IOException {
        byte[] b = Files.readAllBytes(p);
        int[] out = new int[b.length / 4];
        for (int i = 0; i < out.length; i++) {
            out[i] = (b[i * 4] & 0xFF) | ((b[i * 4 + 1] & 0xFF) << 8)
                    | ((b[i * 4 + 2] & 0xFF) << 16) | ((b[i * 4 + 3] & 0xFF) << 24);
        }
        return out;
    }

    static int[][] load(String measure, int states, int seed, int n, Path dir)
            throws IOException {
        Path p = dir.resolve(measure + "_discrete_b" + states + "_s" + seed + "_n" + n + ".bin");
        return colsI(readI32(p), n, ncols(measure));
    }

    static int[][] colsI(int[] flat, int n, int cols) {
        int[][] out = new int[cols][n];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < cols; c++) out[c][i] = flat[i * cols + c];
        return out;
    }

    /// Timed region. `base` is the true alphabet size for every column.
    static double run(String measure, int base, int[][] c) throws Exception {
        switch (measure) {
            case "entropy": {
                EntropyCalculatorDiscrete e = new EntropyCalculatorDiscrete(base);
                e.initialise();
                e.addObservations(c[0]);
                return e.computeAverageLocalOfObservations();
            }
            case "mi": {
                MutualInformationCalculatorDiscrete m =
                        new MutualInformationCalculatorDiscrete(base, base, 0);
                m.initialise();
                m.addObservations(c[0], c[1]);
                return m.computeAverageLocalOfObservations();
            }
            case "cmi": {
                ConditionalMutualInformationCalculatorDiscrete m =
                        new ConditionalMutualInformationCalculatorDiscrete(base, base, base);
                m.initialise();
                m.addObservations(c[0], c[1], c[2]);
                return m.computeAverageLocalOfObservations();
            }
            case "te": {
                TransferEntropyCalculatorDiscrete t =
                        new TransferEntropyCalculatorDiscrete(base, 1);
                t.initialise();
                t.addObservations(c[0], c[1]);
                return t.computeAverageLocalOfObservations();
            }
            default: {
                ConditionalTransferEntropyCalculatorDiscrete t =
                        ConditionalTransferEntropyCalculatorDiscrete.newInstance(base, 1, 1);
                t.initialise();
                t.addObservations(c[0], c[1], c[2]);
                return t.computeAverageLocalOfObservations();
            }
        }
    }

    static String stats(List<Double> times) {
        int n = times.size();
        double sum = 0;
        for (double t : times) sum += t;
        double mean = sum / n;
        double var = 0;
        if (n > 1) {
            for (double t : times) var += (t - mean) * (t - mean);
            var /= (n - 1);
        }
        double sd = Math.sqrt(var);
        List<Double> sorted = new ArrayList<>(times);
        sorted.sort(Double::compare);
        double median = (n % 2 == 1)
                ? sorted.get(n / 2)
                : 0.5 * (sorted.get(n / 2 - 1) + sorted.get(n / 2));
        double half = 1.96 * sd / Math.sqrt(n);
        return String.format(Locale.ROOT,
                "{\"mean\":%.12g,\"stddev\":%.12g,\"min\":%.12g,\"max\":%.12g,"
                        + "\"median\":%.12g,\"samples\":%d,\"ci_lower\":%.12g,\"ci_upper\":%.12g}",
                mean, sd, sorted.get(0), sorted.get(n - 1), median, n, mean - half, mean + half);
    }

    static String functionName(String measure) {
        switch (measure) {
            case "entropy": return "EntropyCalculatorDiscrete";
            case "mi": return "MutualInformationCalculatorDiscrete";
            case "cmi": return "ConditionalMutualInformationCalculatorDiscrete";
            case "te": return "TransferEntropyCalculatorDiscrete";
            default: return "ConditionalTransferEntropyCalculatorDiscrete";
        }
    }

    static String params(String measure, int states, int n) {
        String method = "\"mle\"";
        String delay = (measure.equals("te") || measure.equals("cte")) ? "1" : "null";
        return String.format(Locale.ROOT,
                "{\"n\":%d,\"states\":%d,\"k\":null,\"bandwidth\":null,\"order\":null,"
                        + "\"delay\":%s,\"alpha\":null,\"q\":null,\"dims\":1,\"method\":%s,"
                        + "\"kernel_type\":null}",
                n, states, delay, method);
    }

    static List<Integer> ints(String csv) {
        List<Integer> out = new ArrayList<>();
        for (String s : csv.split(",")) {
            if (!s.trim().isEmpty()) out.add(Integer.parseInt(s.trim()));
        }
        return out;
    }

    static double mean(List<Double> times) {
        double s = 0;
        for (double t : times) s += t;
        return s / times.size();
    }

    public static void main(String[] args) throws Exception {
        Path dir = Paths.get(System.getenv().getOrDefault("BENCH_DATA_DIR", "target/bench-data"));
        String out = null;
        List<Integer> sizes = new ArrayList<>();
        List<Integer> seeds = new ArrayList<>();
        List<Integer> states = new ArrayList<>();
        Map<String, Integer> caps = new HashMap<>();
        double budget = 2.0;
        boolean shortMode = false;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--data-dir": dir = Paths.get(args[++i]); break;
                case "--out": out = args[++i]; break;
                case "--sizes": sizes = ints(args[++i]); break;
                case "--seeds": seeds = ints(args[++i]); break;
                case "--states": states = ints(args[++i]); break;
                case "--budget": budget = Double.parseDouble(args[++i]); break;
                case "--caps":
                    for (String pair : args[++i].split(",")) {
                        String[] kv = pair.split(":");
                        if (kv.length == 2) caps.put(kv[0].trim(), Integer.parseInt(kv[1].trim()));
                    }
                    break;
                case "--short": shortMode = true; break;
                default: throw new IllegalArgumentException("unknown arg " + args[i]);
            }
        }
        if (out == null) out = dir.resolve("results").resolve("jidt_alphabet.json").toString();

        int warmupMax, minIters, maxIters;
        double warmupBudget, iterBudget;
        if (shortMode) {
            warmupMax = 1; warmupBudget = 0; minIters = 1; maxIters = 3; iterBudget = 0;
        } else {
            warmupMax = Integer.parseInt(System.getenv().getOrDefault("BENCH_WARMUP_MAX", "3"));
            warmupBudget = Double.parseDouble(System.getenv().getOrDefault("BENCH_WARMUP_BUDGET_S", "0.4"));
            minIters = Integer.parseInt(System.getenv().getOrDefault("BENCH_MIN_ITERS", "3"));
            maxIters = Integer.parseInt(System.getenv().getOrDefault("BENCH_MAX_ITERS", "10"));
            iterBudget = Double.parseDouble(System.getenv().getOrDefault("BENCH_ITER_BUDGET_S", "1.5"));
        }

        StringBuilder b = new StringBuilder();
        b.append("{\"meta\":{\"schema\":2,\"run_id\":\"fragment_jidt_alphabet\",\"hardware\":null,");
        b.append("\"runtime\":{\"threads\":1,\"adaptive\":").append(!shortMode)
                .append(",\"family\":\"alphabet\",\"budget_s\":").append(budget).append("},");
        b.append("\"seeds\":[");
        for (int i = 0; i < seeds.size(); i++) {
            if (i > 0) b.append(',');
            b.append(seeds.get(i));
        }
        b.append("],\"packages\":[{\"id\":\"jidt\",\"language\":\"java\",\"version\":\"1.6.1\",");
        b.append("\"released\":\"2023-08-22\",\"limitations\":\"Discrete only; values in bits.\"}],");
        b.append("\"coverage\":[");
        for (int i = 0; i < MEASURES.length; i++) {
            if (i > 0) b.append(',');
            b.append("[\"").append(MEASURES[i]).append("\",\"discrete\"]");
        }
        b.append("]},\"benchmarks\":[");

        boolean first = true;
        int count = 0;
        for (String measure : MEASURES) {
            int cap = caps.getOrDefault(measure, 0);
            for (int base : states) {
                if (base > cap) continue;
                boolean stopped = false;
                for (int n : sizes) {
                    if (stopped) break;
                    List<Double> times = new ArrayList<>();
                    double value = 0;
                    for (int seed : seeds) {
                        int[][] data = load(measure, base, seed, n, dir);
                        long w0 = System.nanoTime();
                        for (int w = 0; w < warmupMax; w++) {
                            run(measure, base, data);
                            if (warmupBudget > 0 && (System.nanoTime() - w0) / 1e9 >= warmupBudget) break;
                        }
                        long t0 = System.nanoTime();
                        for (int it = 0; it < maxIters; it++) {
                            long s = System.nanoTime();
                            value = run(measure, base, data);
                            times.add((System.nanoTime() - s) / 1e9);
                            if (it + 1 >= minIters && iterBudget > 0
                                    && (System.nanoTime() - t0) / 1e9 >= iterBudget) break;
                        }
                    }
                    if (!first) b.append(',');
                    first = false;
                    b.append("{\"id\":\"").append(measure).append("/discrete/mle/b").append(base)
                            .append("/n").append(n).append("/jidt\",");
                    b.append("\"package\":\"jidt\",\"language\":\"java\",");
                    b.append("\"measure\":\"").append(measure).append("\",");
                    b.append("\"approach\":\"discrete\",");
                    b.append("\"function\":\"").append(functionName(measure)).append("\",");
                    b.append("\"representative\":true,\"params\":").append(params(measure, base, n)).append(',');
                    b.append("\"statistics\":").append(stats(times)).append(',');
                    b.append("\"value\":").append(String.format(Locale.ROOT, "%.12g", value)).append('}');
                    count++;
                    double m = mean(times);
                    System.out.printf(Locale.ROOT, "  %-8s b%-4d n=%-6d %9.3f ms%n", measure, base, n, m * 1e3);
                    if (m > budget) {
                        System.out.printf(Locale.ROOT, "  -> b%d n=%d exceeded %ss; skipping larger N%n", base, n, budget);
                        stopped = true;
                    }
                }
            }
        }
        b.append("]}");

        Path outPath = Paths.get(out);
        Files.createDirectories(outPath.getParent());
        Files.writeString(outPath, b.toString());
        System.out.println("wrote " + count + " entries to " + outPath);
    }
}
