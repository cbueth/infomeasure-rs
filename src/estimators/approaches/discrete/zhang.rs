use ndarray::{Array1, Array2};
use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::approaches::discrete::discrete_utils::reduce_joint_space_compact;
use crate::estimators::traits::{LocalValues, OptionalLocalValues, JointEntropy};

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
        if n == 0 || n >= N { return 0.0; }
        // t2 = sum_{k=1}^{N-n} (1/k) * prod_{j=1}^{k} (1 - (n-1)/(N-j))
        let nf = n as f64;
        let n_minus_1 = nf - 1.0;
        let n_total = N as f64;
        let mut t2 = 0.0_f64;
        let mut prod = 1.0_f64;
        for k in 1..=(N - n) {
            let denom = n_total - (k as f64);
            let factor = 1.0 - (n_minus_1 / denom);
            prod *= factor;
            t2 += prod / (k as f64);
        }
        t2
    }
}

impl LocalValues for ZhangEntropy {
    fn local_values(&self) -> Array1<f64> {
        // Precompute contribution per unique symbol
        // contribution per symbol is t2(n_i)
        // Then map to each data point
        use std::collections::HashMap;
        let mut contrib: HashMap<i32, f64> = HashMap::with_capacity(self.dataset.k);
        let N = self.dataset.n;
        for (&val, &cnt) in self.dataset.counts.iter() {
            let t2 = Self::t2_for_count(cnt, N);
            contrib.insert(val, t2);
        }
        self.dataset.data.mapv(|v| contrib[&v])
    }

    // global_value uses default mean(local_values)
}

impl JointEntropy for ZhangEntropy {
    type Source = Array1<i32>;
    type Params = ();

    fn joint_entropy(series: &[Self::Source], _params: Self::Params) -> f64 {
        if series.is_empty() { return 0.0; }
        let joint_codes = reduce_joint_space_compact(series);
        let disc = ZhangEntropy::new(joint_codes);
        disc.global_value()
    }
}

impl OptionalLocalValues for ZhangEntropy {
    fn supports_local(&self) -> bool { true }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> { Ok(self.local_values()) }
}
