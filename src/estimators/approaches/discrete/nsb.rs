use ndarray::{Array1, Array2};
use std::f64::{INFINITY, NAN};
use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::traits::{GlobalValue, OptionalLocalValues};
use statrs::function::gamma::{digamma, ln_gamma};
// Try to use trigamma from statrs if available; otherwise we could add another crate.

/// NSB (Nemenman–Shafee–Bialek) entropy estimator for discrete data (natural log base).
///
/// A Bayesian estimator that averages over Dirichlet priors via a 1/K mixture, leading to
/// numerically integrating expectations over β ∈ (0, ln K). This implementation uses adaptive
/// Simpson integration and safeguards near β=0. Optionally override K when known. Global-only.
pub struct NsbEntropy {
    dataset: DiscreteDataset,
    k_override: Option<usize>,
    tol: f64,
    max_recursion: usize,
}

impl NsbEntropy {
    pub fn new(data: Array1<i32>, k_override: Option<usize>) -> Self {
        let dataset = DiscreteDataset::from_data(data);
        Self { dataset, k_override, tol: 1e-6, max_recursion: 12 }
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

    fn find_l0(&self, k: usize, n: usize) -> f64 {
        // Find extremum of log rho using a simple search over K0 in [0.1, K]
        let func = |k0: f64| -> f64 { (k as f64) / k0 - digamma(k0 + n as f64) + digamma(k0) };
        let mut best_k0 = 0.1_f64;
        let mut best_val = INFINITY;
        let steps = 200usize;
        let upper = k as f64;
        let mut t = 0.1_f64;
        let step = (upper - 0.1_f64) / (steps as f64);
        for _ in 0..steps {
            let v = func(t).abs();
            if v < best_val { best_val = v; best_k0 = t; }
            t += step;
        }
        let extremum_beta = best_k0 / (k as f64);
        // We'll compute l0 at this beta
        let counts = self.counts_vec();
        self.neg_log_rho(extremum_beta, k, n, &counts)
    }

    // Adaptive Simpson integration
    fn simpson<F: Fn(f64) -> f64>(&self, f: &F, a: f64, b: f64, tol: f64, depth: usize) -> f64 {
        fn simp<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> f64 {
            let c = 0.5 * (a + b);
            let h = b - a;
            (h / 6.0) * (f(a) + 4.0 * f(c) + f(b))
        }
        let c = 0.5 * (a + b);
        let s_ab = simp(f, a, b);
        let s_ac = simp(f, a, c);
        let s_cb = simp(f, c, b);
        if depth == 0 || (s_ac + s_cb - s_ab).abs() < 15.0 * tol {
            return s_ac + s_cb + (s_ac + s_cb - s_ab) / 15.0;
        }
        self.simpson(f, a, c, tol / 2.0, depth - 1) + self.simpson(f, c, b, tol / 2.0, depth - 1)
    }
}

impl GlobalValue for NsbEntropy {
    fn global_value(&self) -> f64 {
        let n = self.dataset.n as usize;
        let k_obs = self.dataset.k;
        let k = self.k_override.unwrap_or(k_obs);
        if n == 0 || k == 0 { return NAN; }
        let counts = self.counts_vec();
        let coincidences = n.saturating_sub(k);
        if coincidences == 0 { return NAN; }

        // Integration bounds (mirror Python code): 0 .. ln K
        let upper = (k as f64).ln();
        if !upper.is_finite() || upper <= 0.0 { return NAN; }

        let l0 = self.find_l0(k, n);
        let neg_log_rho = |beta: f64| self.neg_log_rho(beta, k, n, &counts);
        let dxi = |beta: f64| self.dxi(beta, k);
        let bayes = |beta: f64| self.bayes_expectation(beta, &counts);

        let f_num = |beta: f64| ((-(neg_log_rho(beta)) + l0).exp()) * dxi(beta) * bayes(beta);
        let f_den = |beta: f64| ((-(neg_log_rho(beta)) + l0).exp()) * dxi(beta);

        // Avoid singularity at beta=0 by starting slightly above 0
        let a = 1e-8;
        let num = self.simpson(&f_num, a, upper, self.tol, self.max_recursion);
        let den = self.simpson(&f_den, a, upper, self.tol, self.max_recursion);

        if den == 0.0 || !den.is_finite() { return NAN; }
        num / den
    }
}

impl OptionalLocalValues for NsbEntropy {
    fn supports_local(&self) -> bool { false }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err("Local values are not supported for NSB estimator")
    }
}


/// Trigamma function ψ1(x) = d^2/dx^2 ln Γ(x)
/// Implementation using recurrence to x>=8 plus asymptotic series expansion.
fn trigamma(mut x: f64) -> f64 {
    // Our use cases have x > 0 (1 + beta, 1 + K*beta), but guard minimal values
    if !x.is_finite() { return f64::NAN; }
    let mut acc = 0.0_f64;
    // Use recurrence: ψ1(x) = ψ1(x+1) + 1/x^2, so accumulate 1/x^2 while increasing x
    while x < 8.0 {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    // Asymptotic expansion at large x
    let z = 1.0 / x;
    let z2 = z * z;
    let mut series = z + 0.5 * z2 + (1.0/6.0) * z2 * z; // 1/x + 1/(2x^2) + 1/(6x^3)
    let z5 = z2 * z2 * z;   // 1/x^5
    let z7 = z5 * z2;       // 1/x^7
    let z9 = z7 * z2;       // 1/x^9
    let z11 = z9 * z2;      // 1/x^11
    series += -(1.0/30.0) * z5 + (1.0/42.0) * z7 - (1.0/30.0) * z9 + (5.0/66.0) * z11;
    acc + series
}


impl NsbEntropy {
    /// Build a vector of NsbEntropy estimators, one per row of a 2D array.
    pub fn from_rows(data: Array2<i32>, k_override: Option<usize>) -> Vec<Self> {
        rows_as_vec(data).into_iter().map(|row| Self::new(row, k_override)).collect()
    }
}
