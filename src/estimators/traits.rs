use ndarray::Array1;

pub trait GlobalValue {
    /// Compute and return the global value of the measure.
    fn global_value(&self) -> f64;
}

pub trait LocalValues {
    /// Compute and return the local values of the measure.
    /// To be overridden by specific measures.
    fn local_values(&self) -> Array1<f64>;

    /// Override global_value to derive it as the mean of local values.
    fn global_value(&self) -> f64 {
        let local_vals = self.local_values();
        local_vals.mean().expect("Local values should not be empty.")
    }
}