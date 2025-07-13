use ndarray::Array1;
use crate::estimators::traits::LocalValues;
use std::collections::HashMap;

pub struct DiscreteEntropy {
    data: Array1<i32>,
    counts: HashMap<i32, usize>,
}

impl DiscreteEntropy {
    pub fn new(data: Array1<i32>) -> Self {
        let counts = count_frequencies(&data);
        Self { data, counts }
    }
}

impl LocalValues for DiscreteEntropy {
    /// Calculate local entropy values for each element in the dataset.
    fn local_values(&self) -> Array1<f64> {
        let n = self.data.len() as f64;

        // Map each value in data to its corresponding count, then take the natural logarithm
        let local_values = self.data.mapv(|value| {
            let count = self.counts[&value] as f64;
            count.ln()
        });

        // Apply the transformation: n.ln() - local_values
        n.ln() - local_values
    }

    /// Calculate global entropy for the data set.
    /// Separate implementation, not inferred from local_values.
    fn global_value(&self) -> f64 {
        let n = self.data.len() as f64;
        let probabilities: Array1<f64> = self.counts.values().map(|&count| count as f64 / n).collect();
        // -sum(p*ln(p)) with probability - vectorized
        let log_probabilities = probabilities.mapv(f64::ln);
        -probabilities.dot(&log_probabilities)
    }
}


/// Helper function to count the occurrences of each value in an array.
fn count_frequencies(data: &Array1<i32>) -> HashMap<i32, usize> {
    // Count the occurrences of each value in an array.
    let mut frequency_map = HashMap::new();

    // Iterate over the array and update the frequency map.
    for &value in data.iter() {
        *frequency_map.entry(value).or_insert(0) += 1;
    }

    frequency_map
}
