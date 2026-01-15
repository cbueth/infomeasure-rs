use infomeasure::estimators::entropy::{Entropy, GlobalValue, LocalValues};
use ndarray::{Array1, array};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

fn main() {
    // Example 2D data (3 points in 2D space)
    let data = array![
        [1.0, 1.5], // Point 1
        [2.0, 3.0], // Point 2
        [4.0, 5.0]  // Point 3
    ];

    // Instantiate the kernel approaches estimator for 2D data
    let bandwidth = 1.0;
    // let kernel_entropy = KernelEntropy::new_2d(data, bandwidth);
    let _kernel_entropy = Entropy::nd_kernel::<2>(data, bandwidth);

    // Calculate local and global approaches
    // let local_values = kernel_entropy.local_values();
    // let global_value = kernel_entropy.global_value();

    // println!("Local Entropy Values: {:?}", local_values);
    // println!("Global Entropy Value: {}", global_value);

    // Example 1D data (100 points, gaussian data)
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let data_1d = Array1::from_iter((0..100).map(|_| normal.sample(&mut rng)));

    // Create kernel entropy estimator for 1D data
    let bandwidth_1d = 0.5;
    let kernel_entropy_1d = Entropy::new_kernel(data_1d, bandwidth_1d);

    // Calculate local and global entropy values
    let local_values_1d = kernel_entropy_1d.local_values();
    let global_value_1d = kernel_entropy_1d.global_value();

    println!("1D Local Entropy Values: {local_values_1d:?}");
    println!("1D Global Entropy Value: {global_value_1d}");
}
