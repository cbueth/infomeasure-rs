use ndarray::{Array1, Array2};
use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::approaches::discrete::discrete_utils::reduce_joint_space_compact;
use crate::estimators::traits::{LocalValues, OptionalLocalValues, GlobalValue, JointEntropy};

/// Zhang entropy estimator for discrete data (Lozano 2017 fast formulation).
///
/// Implements an efficient series expansion that corrects the MLE bias by summing
/// a per-count term t2(n_i) over the support; supports local values which map each
/// sample to its symbol's contribution.
///
/// Cross-entropy is not implemented for Zhang estimator due to
/// theoretical inconsistencies in applying bias corrections from
/// different distributions.
///
/// Joint entropy is supported by reducing the joint space of multiple variables to a single
/// discrete representation before estimation.
pub struct ZhangEntropy {
    dataset: DiscreteDataset,
}

impl ZhangEntropy {
    pub fn new(data: Array1<i32>) -> Self {
        let dataset = DiscreteDataset::from_data(data);
        Self { dataset }
    }

    /// Build a vector of ZhangEntropy estimators, one per row of a 2D array.
    pub fn from_rows(data: Array2<i32>) -> Vec<Self> {
        rows_as_vec(data).into_iter().map(Self::new).collect()
    }

    #[inline]
    fn t2_for_count(n: usize, N: usize) -> f64 {
        if n == 0 || n >= N {
            return 0.0;
        }
        // Following Python implementation exactly:
        // factors = 1.0 - (valid_counts[:, None] - 1.0) / (N - k_values[None, :])
        // t1_matrix = factors.cumprod(axis=1)
        // t2_values = np_sum(t1_masked * reciprocal_k, axis=1)
        let nf = n as f64;
        let n_total = N as f64;
        let mut h_hat = 0.0_f64;
        let mut t1 = 1.0_f64;
        for k in 1..=(N - n) {
            let kf = k as f64;
            let factor = 1.0 - (nf - 1.0) / (n_total - kf);
            t1 *= factor;
            h_hat += t1 / kf;
        }
        h_hat
    }
        }

impl GlobalValue for ZhangEntropy {
    fn global_value(&self) -> f64 {
        let n = self.dataset.n;
        let mut h = 0.0_f64;
        let nf = n as f64;
        for &cnt in self.dataset.counts.values() {
            h += (cnt as f64 / nf) * Self::t2_for_count(cnt, n);
        }
        h
    }
}

impl LocalValues for ZhangEntropy {
    fn local_values(&self) -> Array1<f64> {
        // Precompute contribution per unique symbol
        use std::collections::HashMap;
        let mut contrib: HashMap<i32, f64> = HashMap::with_capacity(self.dataset.k);
        let n = self.dataset.n;
        for (&val, &cnt) in self.dataset.counts.iter() {
            let t2 = Self::t2_for_count(cnt, n);
            // In Zhang estimator, local values are defined as the inner summation t2(n_i)
            // such that global_H = sum( p_i * t2(n_i) ) = mean( t2(n_sampled) )
            contrib.insert(val, t2);
        }
        self.dataset.data.mapv(|v| contrib[&v])
    }
}

impl JointEntropy for ZhangEntropy {
    type Source = Array1<i32>;
    type Params = ();

    fn joint_entropy(series: &[Self::Source], _params: Self::Params) -> f64 {
        if series.is_empty() { return 0.0; }
        let joint_codes = reduce_joint_space_compact(series);
        let disc = ZhangEntropy::new(joint_codes);
        GlobalValue::global_value(&disc)
    }
}

impl OptionalLocalValues for ZhangEntropy {
    fn supports_local(&self) -> bool { true }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> { Ok(self.local_values()) }
}
