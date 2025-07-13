use infomeasure::estimators::entropy::{Entropy, LocalValues};
use ndarray::array;

fn main() {
    // Example 2D data (3 points in 2D space)
    let data = array![
        [1.0, 1.5],  // Point 1
        [2.0, 3.0],  // Point 2
        [4.0, 5.0]   // Point 3
    ];

    // Instantiate the kernel approaches estimator for 2D data
    let bandwidth = 1.0;
    // let kernel_entropy = KernelEntropy::new_2d(data, bandwidth);
    let kernel_entropy = Entropy::new_kernel(data, bandwidth);

    // Calculate local and global approaches
    // let local_values = kernel_entropy.local_values();
    // let global_value = kernel_entropy.global_value();

    // println!("Local Entropy Values: {:?}", local_values);
    // println!("Global Entropy Value: {}", global_value);
}
