// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Cross-package collector: JIDT (native Java).
//
// Times only the estimator call on the shared canonical datasets and writes a
// schema-v2 fragment (`results/jidt.json`). Coverage: discrete all five
// measures; continuous KSG entropy/MI/CMI/TE; box-kernel entropy/MI/TE.
// Continuous CTE and kernel CMI/CTE are not yet covered.
//
// Build/run (inside the bench image):
//   javac -cp "$JIDT_JAR" -d /tmp/jidtcls scripts/bench_packages/JidtCollector.java
//   java -cp "$JIDT_JAR:/tmp/jidtcls" JidtCollector \
//        --data-dir target/bench-data --sizes 100,400 --seeds 1,2

import infodynamics.measures.continuous.kernel.EntropyCalculatorMultiVariateKernel;
import infodynamics.measures.continuous.kernel.MutualInfoCalculatorMultiVariateKernel;
import infodynamics.measures.continuous.kernel.TransferEntropyCalculatorKernel;
import infodynamics.measures.continuous.kozachenko.EntropyCalculatorMultiVariateKozachenko;
import infodynamics.measures.continuous.kraskov.ConditionalMutualInfoCalculatorMultiVariateKraskov1;
import infodynamics.measures.continuous.kraskov.ConditionalTransferEntropyCalculatorKraskov;
import infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov1;
import infodynamics.measures.continuous.kraskov.TransferEntropyCalculatorKraskov;
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
import java.util.List;
import java.util.Locale;

public class JidtCollector {

    static final String[] MEASURES = {"entropy", "mi", "cmi", "te", "cte"};
    static final String[] APPROACHES = {"discrete", "ksg", "kernel_box", "kernel_gaussian"};

    static int ncols(String measure) {
        switch (measure) {
            case "entropy": return 1;
            case "mi": case "te": return 2;
            default: return 3;
        }
    }

    static boolean supported(String measure, String approach) {
        if (approach.equals("discrete")) return true;
        if (approach.equals("ksg")) return true;
        if (approach.equals("kernel_box")) return measure.equals("entropy") || measure.equals("mi") || measure.equals("te");
        return false; // JIDT has no Gaussian-kernel estimator
    }

    static double[] readF64(Path p) throws IOException {
        byte[] b = Files.readAllBytes(p);
        double[] out = new double[b.length / 8];
        for (int i = 0; i < out.length; i++) {
            long v = 0;
            for (int k = 7; k >= 0; k--) v = (v << 8) | (b[i * 8 + k] & 0xFFL);
            out[i] = Double.longBitsToDouble(v);
        }
        return out;
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

    static int[][] colsI(int[] flat, int n, int cols) {
        int[][] out = new int[cols][n];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < cols; c++) out[c][i] = flat[i * cols + c];
        return out;
    }

    static double[][][] colsD(double[] flat, int n, int cols) {
        double[][][] out = new double[cols][n][1];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < cols; c++) out[c][i][0] = flat[i * cols + c];
        return out;
    }

    static double[] flat(double[][] col) {
        double[] out = new double[col.length];
        for (int i = 0; i < col.length; i++) out[i] = col[i][0];
        return out;
    }

    static void setKsg(Object calc) throws Exception {
        if (calc instanceof MutualInfoCalculatorMultiVariateKraskov1) {
            MutualInfoCalculatorMultiVariateKraskov1 m = (MutualInfoCalculatorMultiVariateKraskov1) calc;
            m.setProperty("k", "4");
            m.setProperty("NORMALISE", "false");
        } else if (calc instanceof ConditionalMutualInfoCalculatorMultiVariateKraskov1) {
            ConditionalMutualInfoCalculatorMultiVariateKraskov1 m = (ConditionalMutualInfoCalculatorMultiVariateKraskov1) calc;
            m.setProperty("k", "4");
            m.setProperty("NORMALISE", "false");
        }
        safeSetProperty(calc, "NUM_THREADS", "1");
    }

    /// Load one dataset's columns. Called outside the timed region.
    static Object load(String measure, String approach, int n, int seed, Path dir)
            throws IOException {
        String kind = approach.equals("discrete") ? "discrete" : "continuous";
        Path p = dir.resolve(measure + "_" + kind + "_s" + seed + "_n" + n + ".bin");
        int cols = ncols(measure);
        if (approach.equals("discrete")) {
            return colsI(readI32(p), n, cols);
        }
        return colsD(readF64(p), n, cols);
    }

    /// Best-effort property set (some calculators reject unknown properties).
    static void safeSetProperty(Object calc, String name, String value) {
        try {
            calc.getClass().getMethod("setProperty", String.class, String.class)
                    .invoke(calc, name, value);
        } catch (Exception ignored) {
            // Calculator does not support this property.
        }
    }

    static double run(String measure, String approach, Object data) throws Exception {
        if (approach.equals("discrete")) {
            int[][] c = (int[][]) data;
            int base = (measure.equals("te") || measure.equals("cte")) ? 5 : 10;
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

        double[][][] c = (double[][][]) data;
        if (approach.equals("ksg")) {
            switch (measure) {
                case "entropy": {
                    EntropyCalculatorMultiVariateKozachenko e =
                            new EntropyCalculatorMultiVariateKozachenko();
                    e.setProperty("k", "4");
                    safeSetProperty(e, "NUM_THREADS", "1");
                    e.initialise(1);
                    e.setObservations(c[0]);
                    return e.computeAverageLocalOfObservations();
                }
                case "mi": {
                    MutualInfoCalculatorMultiVariateKraskov1 m =
                            new MutualInfoCalculatorMultiVariateKraskov1();
                    setKsg(m);
                    m.initialise(1, 1);
                    m.setObservations(c[0], c[1]);
                    return m.computeAverageLocalOfObservations();
                }
                case "cmi": {
                    ConditionalMutualInfoCalculatorMultiVariateKraskov1 m =
                            new ConditionalMutualInfoCalculatorMultiVariateKraskov1();
                    setKsg(m);
                    m.initialise(1, 1, 1);
                    m.setObservations(c[0], c[1], c[2]);
                    return m.computeAverageLocalOfObservations();
                }
                case "te": {
                    TransferEntropyCalculatorKraskov t = new TransferEntropyCalculatorKraskov();
                    t.setProperty("k", "4");
                    t.setProperty("NORMALISE", "false");
                    safeSetProperty(t, "NUM_THREADS", "1");
                    t.initialise(1);
                    t.setObservations(flat(c[0]), flat(c[1]));
                    return t.computeAverageLocalOfObservations();
                }
                default: {
                    ConditionalTransferEntropyCalculatorKraskov t =
                            new ConditionalTransferEntropyCalculatorKraskov();
                    t.setProperty("k", "4");
                    t.setProperty("NORMALISE", "false");
                    safeSetProperty(t, "NUM_THREADS", "1");
                    t.initialise(1, 1, 1);
                    t.setObservations(flat(c[0]), flat(c[1]), flat(c[2]));
                    return t.computeAverageLocalOfObservations();
                }
            }
        }

        // kernel_box
        switch (measure) {
            case "entropy": {
                EntropyCalculatorMultiVariateKernel e =
                        new EntropyCalculatorMultiVariateKernel();
                e.setProperty("NORMALISE", "false");
                e.initialise(1, 0.5);
                e.setObservations(c[0]);
                return e.computeAverageLocalOfObservations();
            }
            case "mi": {
                MutualInfoCalculatorMultiVariateKernel m =
                        new MutualInfoCalculatorMultiVariateKernel();
                m.setProperty("NORMALISE", "false");
                m.initialise(1, 1, 0.5);
                m.setObservations(c[0], c[1]);
                return m.computeAverageLocalOfObservations();
            }
            default: {
                TransferEntropyCalculatorKernel t = new TransferEntropyCalculatorKernel();
                t.setProperty("NORMALISE", "false");
                t.initialise(1, 0.5);
                t.setObservations(flat(c[0]), flat(c[1]));
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

    static String functionName(String measure, String approach) {
        if (approach.equals("discrete")) {
            switch (measure) {
                case "entropy": return "EntropyCalculatorDiscrete";
                case "mi": return "MutualInformationCalculatorDiscrete";
                case "cmi": return "ConditionalMutualInformationCalculatorDiscrete";
                case "te": return "TransferEntropyCalculatorDiscrete";
                default: return "ConditionalTransferEntropyCalculatorDiscrete";
            }
        }
        if (approach.equals("ksg")) {
            switch (measure) {
                case "entropy": return "EntropyCalculatorMultiVariateKozachenko";
                case "mi": return "MutualInfoCalculatorMultiVariateKraskov1";
                case "cmi": return "ConditionalMutualInfoCalculatorMultiVariateKraskov1";
                case "te": return "TransferEntropyCalculatorKraskov";
                default: return "ConditionalTransferEntropyCalculatorKraskov";
            }
        }
        switch (measure) {
            case "entropy": return "EntropyCalculatorMultiVariateKernel";
            case "mi": return "MutualInfoCalculatorMultiVariateKernel";
            default: return "TransferEntropyCalculatorKernel";
        }
    }

    static String params(String measure, String approach, int n) {
        String bw = approach.startsWith("kernel") ? "0.5" : "null";
        String kt = approach.startsWith("kernel") ? "\"box\"" : "null";
        String method = approach.equals("discrete") ? "\"mle\"" : "null";
        String k = approach.equals("ksg") ? "4" : "null";
        return String.format(Locale.ROOT,
                "{\"n\":%d,\"k\":%s,\"bandwidth\":%s,\"order\":null,\"delay\":1,"
                        + "\"alpha\":null,\"q\":null,\"dims\":1,\"method\":%s,\"kernel_type\":%s}",
                n, k, bw, method, kt);
    }

    public static void main(String[] args) throws Exception {
        Path dir = Paths.get(System.getenv().getOrDefault("BENCH_DATA_DIR", "target/bench-data"));
        String out = null;
        List<Integer> sizes = new ArrayList<>();
        List<Integer> seeds = new ArrayList<>();
        boolean shortMode = false;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--data-dir": dir = Paths.get(args[++i]); break;
                case "--out": out = args[++i]; break;
                case "--sizes":
                    for (String s : args[++i].split(",")) sizes.add(Integer.parseInt(s.trim()));
                    break;
                case "--seeds":
                    for (String s : args[++i].split(",")) seeds.add(Integer.parseInt(s.trim()));
                    break;
                case "--short": shortMode = true; break;
                default: throw new IllegalArgumentException("unknown arg " + args[i]);
            }
        }
        if (out == null) out = dir.resolve("results").resolve("jidt.json").toString();
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
        b.append("{\"meta\":{\"schema\":2,\"run_id\":\"fragment_jidt\",\"hardware\":null,");
        b.append("\"runtime\":{\"threads\":1,\"adaptive\":").append(!shortMode)
                .append(",\"warmup_max\":").append(warmupMax)
                .append(",\"warmup_budget_s\":").append(warmupBudget)
                .append(",\"min_iters\":").append(minIters)
                .append(",\"max_iters\":").append(maxIters)
                .append(",\"iter_budget_s\":").append(iterBudget).append("},");
        b.append("\"seeds\":[");
        for (int i = 0; i < seeds.size(); i++) {
            if (i > 0) b.append(',');
            b.append(seeds.get(i));
        }
        b.append("],\"packages\":[{\"id\":\"jidt\",\"language\":\"java\",\"version\":\"1.6.1\",");
        b.append("\"released\":\"2023-08-22\",\"artifact_sha256\":\"2d367c244b729877fdaf0608884cf97ae964a8035c3020215b799812143a5b11\",");
        b.append("\"limitations\":\"No Gaussian-kernel estimator; no kernel conditional MI/CTE. Also ships a linear-Gaussian estimator family (not compared; no infomeasure counterpart).\"}]},");
        b.append("\"benchmarks\":[");

        boolean first = true;
        int count = 0;
        for (String measure : MEASURES) {
            for (String approach : APPROACHES) {
                if (!supported(measure, approach)) continue;
                for (int n : sizes) {
                    List<Double> times = new ArrayList<>();
                    double value = 0;
                    for (int seed : seeds) {
                        // File I/O is deliberately outside the timed region.
                        Object data = load(measure, approach, n, seed, dir);
                        long w0 = System.nanoTime();
                        for (int w = 0; w < warmupMax; w++) {
                            run(measure, approach, data);
                            if (warmupBudget > 0 && (System.nanoTime() - w0) / 1e9 >= warmupBudget) {
                                break;
                            }
                        }
                        long t0 = System.nanoTime();
                        for (int it = 0; it < maxIters; it++) {
                            long s = System.nanoTime();
                            value = run(measure, approach, data);
                            times.add((System.nanoTime() - s) / 1e9);
                            if (it + 1 >= minIters && iterBudget > 0
                                    && (System.nanoTime() - t0) / 1e9 >= iterBudget) {
                                break;
                            }
                        }
                    }
                    if (!first) b.append(',');
                    first = false;
                    b.append("{\"id\":\"").append(measure).append('/').append(approach)
                            .append("/n").append(n).append("/jidt\",");
                    b.append("\"package\":\"jidt\",\"language\":\"java\",");
                    b.append("\"measure\":\"").append(measure).append("\",");
                    b.append("\"approach\":\"").append(approach).append("\",");
                    b.append("\"function\":\"").append(functionName(measure, approach)).append("\",");
                    b.append("\"params\":").append(params(measure, approach, n)).append(',');
                    b.append("\"statistics\":").append(stats(times)).append(',');
                    b.append("\"value\":").append(String.format(Locale.ROOT, "%.12g", value)).append('}');
                    count++;
                    System.out.printf(Locale.ROOT, "  %7s %-16s n=%-6d %9.3f ms%n",
                            measure, approach, n, times.stream().mapToDouble(Double::doubleValue).average().orElse(0) * 1e3);
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
