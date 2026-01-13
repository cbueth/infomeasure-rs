use crate::estimators::approaches::discrete::discrete_utils::reduce_joint_space_compact;
use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::traits::{
    CrossEntropy, GlobalValue, JointEntropy, LocalValues, OptionalLocalValues,
};
use ndarray::{Array1, Array2};

/// Miller–Madow entropy estimator for discrete data (natural log base).
///
/// Adds the small-sample bias correction (K-1)/(2N) to the MLE (Shannon) estimate.
/// Local values are the MLE local values uniformly offset by the correction.
/// Useful when K is moderate relative to N and a simple analytical correction suffices.
///
/// Cross-entropy is supported between two distributions.
///
/// Joint entropy is supported by reducing the joint space of multiple variables to a single
/// discrete representation before estimation.
pub struct MillerMadowEntropy {
    dataset: DiscreteDataset,
}

impl MillerMadowEntropy {
    pub fn new(data: Array1<i32>) -> Self {
        let dataset = DiscreteDataset::from_data(data);
        Self { dataset }
    }

    /// Build a vector of MillerMadowEntropy estimators, one per row of a 2D array.
    pub fn from_rows(data: Array2<i32>) -> Vec<Self> {
        rows_as_vec(data).into_iter().map(Self::new).collect()
    }

    #[inline]
    fn correction(&self) -> f64 {
        // (K - 1) / (2N)
        (self.dataset.k.saturating_sub(1) as f64) / (2.0 * self.dataset.n as f64)
    }
}

impl GlobalValue for MillerMadowEntropy {
    fn global_value(&self) -> f64 {
        // H_MM = H_MLE + (K-1)/(2N)
        let n_f = self.dataset.n as f64;
        let mut h = 0.0_f64;
        for &cnt in self.dataset.counts.values() {
            let p = (cnt as f64) / n_f;
            h -= if p > 0.0 { p * p.ln() } else { 0.0 };
        }
        h + self.correction()
    }
}

impl LocalValues for MillerMadowEntropy {
    fn local_values(&self) -> Array1<f64> {
        let corr = self.correction();
        // Local MLE values = -ln p(x); add global MM correction uniformly
        let p_local = self.dataset.map_probs();
        -p_local.mapv(f64::ln) + corr
    }
}

impl CrossEntropy for MillerMadowEntropy {
    /// Cross-entropy H_MM(P, Q) = -Σ_x p(x) ln q(x) + correction
    /// where correction = (((Kp + Kq)/2) - 1) / (Np + Nq)
    fn cross_entropy(&self, other: &MillerMadowEntropy) -> f64 {
        use std::collections::HashSet;
        // Build sets of supports
        let supp_p: HashSet<i32> = self.dataset.counts.keys().cloned().collect();
        let supp_q: HashSet<i32> = other.dataset.counts.keys().cloned().collect();
        let inter: HashSet<i32> = supp_p.intersection(&supp_q).cloned().collect();
        if inter.is_empty() {
            return 0.0;
        }
        // Probability maps (ML)
        let p_map = &self.dataset.dist; // value -> p
        let q_map = &other.dataset.dist; // value -> q
        let mut h = 0.0_f64;
        for v in inter {
            if let (Some(&p), Some(&q)) = (p_map.get(&v), q_map.get(&v)) {
                if p > 0.0 && q > 0.0 {
                    h -= p * q.ln();
                }
            }
        }
        let n_total = (self.dataset.n + other.dataset.n) as f64;
        let k_avg_minus1 = ((self.dataset.k + other.dataset.k) as f64) / 2.0 - 1.0;
        h + (k_avg_minus1 / n_total)
    }
}

impl JointEntropy for MillerMadowEntropy {
    type Source = Array1<i32>;
    type Params = ();

    fn joint_entropy(series: &[Self::Source], _params: Self::Params) -> f64 {
        if series.is_empty() {
            return 0.0;
        }
        let joint_codes = reduce_joint_space_compact(series);
        let disc = MillerMadowEntropy::new(joint_codes);
        GlobalValue::global_value(&disc)
    }
}

impl OptionalLocalValues for MillerMadowEntropy {
    fn supports_local(&self) -> bool {
        true
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Ok(self.local_values())
    }
}
