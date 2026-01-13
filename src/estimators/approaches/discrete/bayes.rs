use std::collections::HashMap;

use ndarray::{Array1, Array2};

use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::approaches::discrete::discrete_utils::reduce_joint_space_compact;
use crate::estimators::traits::{CrossEntropy, GlobalValue, JointEntropy, LocalValues, OptionalLocalValues};

/// Choices for the Dirichlet concentration parameter alpha
#[derive(Clone, Debug)]
pub enum AlphaParam {
    /// Explicit numeric alpha value
    Value(f64),
    /// Jeffreys prior: alpha = 0.5
    Jeffrey,
    /// Laplace/add-one smoothing: alpha = 1.0
    Laplace,
    /// Schürmann–Grassberger prior: alpha = 1/K
    SchGrass,
    /// Minimax prior: alpha = sqrt(N)/K
    MinMax,
}

/// Bayesian entropy estimator for discrete data with Dirichlet prior (natural log base).
///
/// Computes entropy from the posterior mean probabilities (n_i + α)/(N + Kα).
/// Choose α via AlphaParam (Jeffrey, Laplace, Schürmann–Grassberger, Minimax, or explicit).
/// Optionally, override support size K for unobserved categories. Global-only.
///
/// Cross-entropy is supported between two distributions.
///
/// Joint entropy is supported by reducing the joint space of multiple variables to a single
/// discrete representation before estimation.
///
/// Local values are not implemented for Bayes estimator as it is a Bayesian
/// estimator that averages over Dirichlet priors.
pub struct BayesEntropy {
    dataset: DiscreteDataset,
    alpha: AlphaParam,
    /// Optional override for support size K
    k_override: Option<usize>,
}

impl CrossEntropy for BayesEntropy {
    /// Cross-entropy H(P, Q) = -Σ_x p_bayes(x) log q_bayes(x) over common support
    fn cross_entropy(&self, other: &BayesEntropy) -> f64 {
        // Build Bayesian distributions (value -> prob)
        let (p_probs, p_uniq) = self.bayes_probs();
        let (q_probs, q_uniq) = other.bayes_probs();
        let p_map: HashMap<i32, f64> = p_uniq.into_iter().zip(p_probs.into_iter()).collect();
        let q_map: HashMap<i32, f64> = q_uniq.into_iter().zip(q_probs.into_iter()).collect();

        // Intersection of supports
        let mut h = 0.0_f64;
        for (&val, &pval) in &p_map {
            if let Some(&qval) = q_map.get(&val) {
                if pval > 0.0 && qval > 0.0 {
                    h -= pval * qval.ln();
                }
            }
        }
        h
    }
}

impl BayesEntropy {
    pub fn new(data: Array1<i32>, alpha: AlphaParam, k_override: Option<usize>) -> Self {
        let dataset = DiscreteDataset::from_data(data);
        Self { dataset, alpha, k_override }
    }

    /// Build a vector of BayesEntropy estimators, one per row of a 2D array.
    pub fn from_rows(data: Array2<i32>, alpha: AlphaParam, k_override: Option<usize>) -> Vec<Self> {
        rows_as_vec(data).into_iter().map(|row| Self::new(row, alpha.clone(), k_override)).collect()
    }

    /// Compute Bayesian probabilities (n + alpha) / (N + K * alpha)
    pub fn bayes_probs(&self) -> (Vec<f64>, Vec<i32>) {
        let n = self.dataset.n as f64;
        let k_obs = self.dataset.k;
        let k = self.k_override.unwrap_or(k_obs) as f64;
        let alpha = self.resolve_alpha(k as usize, self.dataset.n);
        let denom = n + k * alpha;
        let mut probs: Vec<f64> = Vec::with_capacity(k_obs);
        let mut uniq: Vec<i32> = Vec::with_capacity(k_obs);
        for (val, cnt) in self.dataset.counts.iter() {
            uniq.push(*val);
            probs.push((*cnt as f64 + alpha) / denom);
        }
        (probs, uniq)
    }

    fn resolve_alpha(&self, k: usize, n: usize) -> f64 {
        match self.alpha {
            AlphaParam::Value(a) => a,
            AlphaParam::Jeffrey => 0.5,
            AlphaParam::Laplace => 1.0,
            AlphaParam::SchGrass => {
                if k == 0 { 0.0 } else { 1.0 / k as f64 }
            }
            AlphaParam::MinMax => {
                if k == 0 { 0.0 } else { (n as f64).sqrt() / k as f64 }
            }
        }
    }
}

impl GlobalValue for BayesEntropy {
    /// Global entropy: -Σ p_bayes ln p_bayes
    fn global_value(&self) -> f64 {
        let (probs, _uniq) = self.bayes_probs();
        let mut h = 0.0_f64;
        for p in probs.into_iter() {
            if p > 0.0 { h -= p * p.ln(); }
        }
        h
    }
}

impl JointEntropy for BayesEntropy {
    type Source = Array1<i32>;
    type Params = (AlphaParam, Option<usize>);

    fn joint_entropy(series: &[Self::Source], params: Self::Params) -> f64 {
        if series.is_empty() { return 0.0; }
        let joint_codes = reduce_joint_space_compact(series);
        let disc = BayesEntropy::new(joint_codes, params.0, params.1);
        disc.global_value()
    }
}

impl LocalValues for BayesEntropy {
    fn local_values(&self) -> Array1<f64> {
        Array1::zeros(0)
    }
}

impl OptionalLocalValues for BayesEntropy {
    fn supports_local(&self) -> bool { false }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err("Local values are not supported for Bayes estimator as it is a Bayesian estimator that averages over Dirichlet priors.")
    }
}
