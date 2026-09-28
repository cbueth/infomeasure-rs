// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Kullback–Leibler and Jensen–Shannon divergence examples.
//!
//! Run with: `cargo run --example kld_jsd_example`

use infomeasure::estimators::composite_measures::{Kld, jsd, jsd_kernel_1d};
use infomeasure::estimators::entropy::Entropy;
use ndarray::array;

fn main() {
    let p = Entropy::new_discrete(array![0, 0, 0, 0, 0, 1, 2, 3]);
    let q = Entropy::new_discrete(array![0, 1, 1, 1, 1, 2, 2, 2]);

    let kld_pq = p.kld(&q);
    let kld_qp = q.kld(&p);
    println!("KLD(P || Q) = {kld_pq}");
    println!("KLD(Q || P) = {kld_qp}  (asymmetric)");

    let jsd_two = jsd(
        &[
            Entropy::new_discrete(array![0, 0, 0, 0, 0, 1, 2, 3]),
            Entropy::new_discrete(array![0, 1, 1, 1, 1, 2, 2, 2]),
        ],
        None,
    );
    let jsd_three = jsd(
        &[
            Entropy::new_discrete(array![0, 0, 0, 0, 0, 1, 2, 3]),
            Entropy::new_discrete(array![0, 1, 1, 1, 1, 2, 2, 2]),
            Entropy::new_discrete(array![0, 1, 1, 1, 2, 2, 3, 3]),
        ],
        Some(&[0.5, 0.25, 0.25]),
    );
    println!("JSD(P, Q) = {jsd_two}  (symmetric, bounded by ln 2)");
    println!("JSD_π(P, Q, R) = {jsd_three}");

    let x = array![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
    let y = array![1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
    let jsd_kernel = jsd_kernel_1d(&[x, y], None, 1.0, "gaussian");
    println!("JSD (kernel, gaussian) = {jsd_kernel}");
}
