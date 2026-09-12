// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Minimal JIDT smoke: KSG MI on correlated Gaussian data with the exact
// semantics we will use for the comparison (k=4, no noise, no normalisation).
// Compiled at image build time against infodynamics.jar.

import infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov1;
import java.util.Random;

public class JidtSmoke {
    public static void main(String[] args) throws Exception {
        int n = 200;
        double[][] x = new double[n][1];
        double[][] y = new double[n][1];
        Random r = new Random(42);
        for (int i = 0; i < n; i++) {
            x[i][0] = r.nextGaussian();
            y[i][0] = 0.5 * x[i][0] + Math.sqrt(0.75) * r.nextGaussian();
        }
        MutualInfoCalculatorMultiVariateKraskov1 mi =
            new MutualInfoCalculatorMultiVariateKraskov1();
        mi.setProperty("NOISE_LEVEL_TO_ADD", "0");
        mi.setProperty("NORMALISE", "false");
        mi.setProperty("k", "4");
        mi.initialise(1, 1);
        mi.setObservations(x, y);
        double v = mi.computeAverageLocalOfObservations();
        System.out.printf("      JIDT KSG MI = %.6f%n", v);
    }
}
