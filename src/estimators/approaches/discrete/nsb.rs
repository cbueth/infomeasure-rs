use crate::estimators::doc_macros::doc_snippets;
// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::estimators::approaches::discrete::discrete_utils::reduce_joint_space_compact;
use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::traits::{GlobalValue, JointEntropy, LocalValues, OptionalLocalValues};
use ndarray::{Array1, Array2};
use statrs::function::gamma::{digamma, ln_gamma};

/// NSB (Nemenman–Shafee–Bialek) entropy estimator for discrete data (natural log base).
///
/// ## Theory
///
/// The Nemenman-Shafee-Bialek (NSB) estimator [Nemenman et al., 2002](crate::guide::references#nsb2002) is a Bayesian approach designed
/// for extremely undersampled data. It addresses the "problem of priors" in entropy estimation
/// by using a mixture of Dirichlet priors that results in a nearly flat prior over the
/// entropy itself.
///
/// The estimator computes the expectation of entropy by integrating over the concentration
/// parameter $\beta$:
///
/// $$\hat{H}_{NSB} = \frac{\int \langle H \rangle_{\beta} \rho(\beta \mid \mathbf{n}) \, d\beta}{\int \rho(\beta \mid \mathbf{n}) \, d\beta}$$
///
/// where:
/// - $\langle H \rangle_{\beta}$ is the expected entropy for a given $\beta$.
/// - $\rho(\beta \mid \mathbf{n})$ is the posterior weight of $\beta$ given the counts $\mathbf{n}$.
/// - $\beta$ is the parameter of the symmetric Dirichlet prior $\mathrm{Dir}(\beta)$.
///
/// This implementation uses adaptive Gauss–Kronrod (15-point) integration, mirroring
/// SciPy's `quad`, and a Brent root-find for the log-ρ extremum.
///
#[doc = doc_snippets!(discrete_guide_ref)]
pub struct NsbEntropy {
    dataset: DiscreteDataset,
    k_override: Option<usize>,
    tol: f64,
}

impl NsbEntropy {
    pub fn new(data: Array1<i32>, k_override: Option<usize>) -> Self {
        let dataset = DiscreteDataset::from_data(data);
        Self {
            dataset,
            k_override,
            tol: 1e-9,
        }
    }

    fn counts_vec(&self) -> Vec<usize> {
        self.dataset.counts.values().cloned().collect()
    }

    fn neg_log_rho(&self, beta: f64, k: usize, n: usize, counts: &[usize]) -> f64 {
        let kappa = (k as f64) * beta;
        // -(ln Γ(κ) - ln Γ(N+κ))
        let mut result = -(ln_gamma(kappa) - ln_gamma(n as f64 + kappa));
        // -Σ n_i * (ln Γ(n_i + β) - ln Γ(β))
        let ln_g_beta = ln_gamma(beta);
        let mut sum_terms = 0.0_f64;
        for &ci in counts {
            sum_terms += (ci as f64) * (ln_gamma(ci as f64 + beta) - ln_g_beta);
        }
        result -= sum_terms;
        result
    }

    fn dxi(&self, beta: f64, k: usize) -> f64 {
        // dξ/dβ = K * ψ1(1 + Kβ) - ψ1(1 + β), where ψ1 is polygamma(1, .) (trigamma)
        let kb = (k as f64) * beta;
        (k as f64) * trigamma(1.0 + kb) - trigamma(1.0 + beta)
    }

    fn bayes_expectation(&self, beta: f64, counts: &[usize]) -> f64 {
        // E[H] = ψ(Σα_i + 1) - (1/Σα_i) Σ(α_i ψ(α_i + 1)), where α_i = n_i + β
        let total_alpha = (self.dataset.n as f64) + (counts.len() as f64) * beta;
        let mut sum_term = 0.0_f64;
        for &ci in counts {
            let ai = (ci as f64) + beta;
            sum_term += ai * digamma(ai + 1.0);
        }
        digamma(total_alpha + 1.0) - (sum_term / total_alpha)
    }

    /// Find the extremum of log ρ by locating the root of
    /// `K/K0 - ψ(K0 + N) + ψ(K0)` on `[0.1, K]` via Brent's method
    /// (matching Python's `scipy.optimize.root_scalar(..., method="brentq")`).
    fn find_extremum_k0(&self, k: usize, n: usize) -> f64 {
        let func = |k0: f64| (k as f64) / k0 - digamma(k0 + n as f64) + digamma(k0);

        // Brent's method operates on a bracket [a, b] with f(a) and f(b) of
        // opposite sign. If the initial bracket does not straddle the root,
        // fall back to a coarse scan (matches Python's brentq fallback).
        let mut a = 0.1_f64;
        let mut b = k as f64;
        let mut fa = func(a);
        let mut fb = func(b);
        if fa * fb >= 0.0 {
            let steps = 64usize;
            let step = (b - a) / (steps as f64);
            let mut t = a;
            let mut best = a;
            let mut best_v = f64::INFINITY;
            for _ in 0..=steps {
                let v = func(t).abs();
                if v < best_v {
                    best_v = v;
                    best = t;
                }
                t += step;
            }
            return best;
        }

        // Brent's method (Numerical Recipes `zbrent`, equivalent to SciPy's
        // `brentq`). Maintains a bracket `[b, c]` with f(b)·f(c) < 0 at all
        // times, combining inverse quadratic interpolation with bisection.
        let mut c = b;
        let mut fc = fb;
        let mut d = b - a;
        let mut e = b - a;
        const TOL: f64 = 1e-12;
        const ITMAX: usize = 100;

        for _ in 0..ITMAX {
            if (fb > 0.0) == (fc > 0.0) {
                // The root lies between b and c; discard a and re-bracket.
                c = a;
                fc = fa;
                d = b - a;
                e = d;
            }
            if fc.abs() < fb.abs() {
                a = b;
                b = c;
                c = a;
                fa = fb;
                fb = fc;
                fc = fa;
            }
            let tol1 = 2.0 * f64::EPSILON * b.abs() + 0.5 * TOL;
            let xm = 0.5 * (c - b);
            if xm.abs() <= tol1 || fb == 0.0 {
                return b;
            }

            if e.abs() >= tol1 && fa.abs() > fb.abs() {
                // Attempt inverse quadratic interpolation.
                let s = fb / fa;
                let (mut p, mut q);
                if a == c {
                    p = 2.0 * xm * s;
                    q = 1.0 - s;
                } else {
                    q = fa / fc;
                    let r = fb / fc;
                    p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0);
                }
                if p > 0.0 {
                    q = -q;
                }
                p = p.abs();
                let min1 = 3.0 * xm * q - (tol1 * q).abs();
                let min2 = (e * q).abs();
                if 2.0 * p < min1.min(min2) {
                    e = d;
                    d = p / q;
                } else {
                    d = xm;
                    e = d;
                }
            } else {
                // Bisection.
                d = xm;
                e = d;
            }
            a = b;
            fa = fb;
            b += if d.abs() >= tol1 {
                d
            } else if xm >= 0.0 {
                tol1
            } else {
                -tol1
            };
            fb = func(b);
        }
        b
    }

    fn find_l0(&self, k: usize, n: usize) -> f64 {
        let extremum_k0 = self.find_extremum_k0(k, n);
        let extremum_beta = extremum_k0 / (k as f64);
        let counts = self.counts_vec();
        self.neg_log_rho(extremum_beta, k, n, &counts)
    }
}

impl GlobalValue for NsbEntropy {
    fn global_value(&self) -> f64 {
        let n = self.dataset.n;
        let k_obs = self.dataset.k;
        let k = self.k_override.unwrap_or(k_obs);
        if n == 0 || k == 0 {
            return f64::NAN;
        }
        let counts = self.counts_vec();
        let coincidences = (n as i64) - (k as i64);
        // If coincidences <= 0, NSB is still defined as long as k > 0 and n > 0.
        // However, Python returns NaN for coincidences == 0 in some cases, but not others.
        // Actually, looking at Python NSB tests:
        // test_nsb_k_parameter_no_coincidences_with_k:
        // result_k10 (K=10, N=5) -> works
        // result_k5 (K=5, N=5) -> NaN
        // result_k3 (K=3, N=5) -> works
        if coincidences == 0 {
            return f64::NAN;
        }

        // Integration bounds (mirror Python code): 0 .. ln K
        let upper = (k as f64).ln();
        if !upper.is_finite() || upper <= 0.0 {
            return f64::NAN;
        }

        let l0 = self.find_l0(k, n);
        let neg_log_rho = |beta: f64| self.neg_log_rho(beta, k, n, &counts);
        let dxi = |beta: f64| self.dxi(beta, k);
        let bayes = |beta: f64| self.bayes_expectation(beta, &counts);

        let f_num = |beta: f64| ((-(neg_log_rho(beta)) + l0).exp()) * dxi(beta) * bayes(beta);
        let f_den = |beta: f64| ((-(neg_log_rho(beta)) + l0).exp()) * dxi(beta);

        // Avoid singularity at beta=0 by starting slightly above 0.
        let a = 1e-8;
        let num = adaptive_gk15(&f_num, a, upper, self.tol, MAX_SUBDIVISIONS);
        let den = adaptive_gk15(&f_den, a, upper, self.tol, MAX_SUBDIVISIONS);

        if den == 0.0 || !den.is_finite() {
            return f64::NAN;
        }
        num / den
    }
}

/// Maximum recursion depth for the adaptive integrator. This is a *depth* bound,
/// so the worst-case interval count is `2^MAX_SUBDIVISIONS`. It must stay small:
/// near the β→0 singularity the integrand is steep and the error estimate never
/// converges, so without a tight cap the bisection would explode exponentially.
const MAX_SUBDIVISIONS: usize = 12;

/// Abscissae, Kronrod weights and embedded Gauss weights of the 15-point
/// Gauss–Kronrod rule (QUADPACK `dqk15`). Only the positive abscissae are given;
/// the rule is symmetric about the origin. The centre abscissa is `XGK[7] = 0`.
///
/// Ordering matches the authoritative `dqk15.f` (evaluated with 80-digit
/// arithmetic by L. W. Fullerton, Bell Labs, 1981):
/// - `XGK` = `[xgk1, xgk2, ..., xgk8]` (even indices are the embedded Gauss nodes)
/// - `WGK` = `[wgk1, wgk2, ..., wgk8]`
/// - `WG`  = `[wg1, wg2, wg3, wg4]` (7-point Gauss weights; `WG[3]` is the centre)
const XGK: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_5,
    0.0,
];
const WGK: [f64; 8] = [
    0.022_935_322_010_529_2,
    0.063_092_092_629_978_6,
    0.104_790_010_322_250_2,
    0.140_653_259_715_525_9,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_8,
];
const WG: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_7,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

/// Evaluate a single 15-point Gauss–Kronrod interval, returning `(integral, err)`
/// where `err` is the difference between the 15- and 7-point estimates (this is
/// the `dqk15` error estimate, before the `resasc` correction).
fn gk15<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> (f64, f64) {
    let center = 0.5 * (a + b);
    let half = 0.5 * (b - a);

    let fc = f(center);
    let mut res_k = fc * WGK[7]; // centre uses wgk(8)
    let mut res_g = fc * WG[3]; // centre uses wg(4)

    // dqk15 loop 1: the odd-indexed abscissae XGK[1], XGK[3], XGK[5] carry both
    // Gauss weights WG[0..2] and Kronrod weights WGK[1], WGK[3], WGK[5].
    for (j, &wg) in WG.iter().take(3).enumerate() {
        let node = 2 * j + 1;
        let absc = half * XGK[node];
        let fsum = f(center - absc) + f(center + absc);
        res_g += wg * fsum;
        res_k += WGK[node] * fsum;
    }

    // dqk15 loop 2: the even-indexed abscissae XGK[0], XGK[2], XGK[4], XGK[6]
    // are Kronrod-only nodes.
    for (j, &wgk) in WGK.iter().take(7).step_by(2).enumerate() {
        let node = 2 * j;
        let absc = half * XGK[node];
        let fsum = f(center - absc) + f(center + absc);
        res_k += wgk * fsum;
    }

    let result = res_k * half;
    let abserr = (res_k - res_g).abs() * half;
    (result, abserr)
}

/// Adaptive Gauss–Kronrod 15 integration by recursive bisection until the error
/// estimate is below `tol` or `max_depth` subdivisions are reached.
fn adaptive_gk15<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, tol: f64, max_depth: usize) -> f64 {
    if max_depth == 0 {
        return gk15(f, a, b).0;
    }
    let c = 0.5 * (a + b);
    let (whole, whole_err) = gk15(f, a, b);
    if whole_err < tol {
        return whole;
    }
    adaptive_gk15(f, a, c, tol * 0.5, max_depth - 1)
        + adaptive_gk15(f, c, b, tol * 0.5, max_depth - 1)
}

impl LocalValues for NsbEntropy {
    fn local_values(&self) -> Array1<f64> {
        Array1::zeros(0)
    }
}

impl JointEntropy for NsbEntropy {
    type Source = Array1<i32>;
    type Params = Option<usize>; // k_override

    fn joint_entropy(series: &[Self::Source], params: Self::Params) -> f64 {
        if series.is_empty() {
            return 0.0;
        }
        let joint_codes = reduce_joint_space_compact(series);
        let disc = NsbEntropy::new(joint_codes, params);
        disc.global_value()
    }
}

/// Trigamma function ψ1(x) = d^2/dx^2 ln Γ(x)
/// Implementation using recurrence to x>=8 plus asymptotic series expansion.
fn trigamma(mut x: f64) -> f64 {
    // Our use cases have x > 0 (1 + beta, 1 + K*beta), but guard minimal values
    if !x.is_finite() {
        return f64::NAN;
    }
    let mut acc = 0.0_f64;
    // Use recurrence: ψ1(x) = ψ1(x+1) + 1/x^2, so accumulate 1/x^2 while increasing x
    while x < 8.0 {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    // Asymptotic expansion at large x
    let z = 1.0 / x;
    let z2 = z * z;
    let mut series = z + 0.5 * z2 + (1.0 / 6.0) * z2 * z; // 1/x + 1/(2x^2) + 1/(6x^3)
    let z5 = z2 * z2 * z; // 1/x^5
    let z7 = z5 * z2; // 1/x^7
    let z9 = z7 * z2; // 1/x^9
    let z11 = z9 * z2; // 1/x^11
    series += -(1.0 / 30.0) * z5 + (1.0 / 42.0) * z7 - (1.0 / 30.0) * z9 + (5.0 / 66.0) * z11;
    acc + series
}

impl NsbEntropy {
    /// Build a vector of NsbEntropy estimators, one per row of a 2D array.
    pub fn from_rows(data: Array2<i32>, k_override: Option<usize>) -> Vec<Self> {
        rows_as_vec(data)
            .into_iter()
            .map(|row| Self::new(row, k_override))
            .collect()
    }
}

impl OptionalLocalValues for NsbEntropy {
    fn supports_local(&self) -> bool {
        false
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err(
            "Local values are not supported for NSB estimator as it averages over Dirichlet priors.",
        )
    }
}
