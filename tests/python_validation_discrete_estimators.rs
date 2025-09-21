use ndarray::array;
use approx::assert_abs_diff_eq;

use infomeasure::estimators::approaches::discrete::{
    mle::DiscreteEntropy,
    miller_madow::MillerMadowEntropy,
    shrink::ShrinkEntropy,
    grassberger::GrassbergerEntropy,
    zhang::ZhangEntropy,
    bayes::{BayesEntropy, AlphaParam},
    bonachela::BonachelaEntropy,
    chao_shen::ChaoShenEntropy,
    chao_wang_jost::ChaoWangJostEntropy,
    ansb::AnsbEntropy,
    nsb::NsbEntropy,
};
use infomeasure::estimators::traits::{GlobalValue, LocalValues};
use validation::python;

mod test_helpers; // reuse assertion helper
use test_helpers::assert_entropy_values_close;

#[test]
fn validate_discrete_against_python() {
    let data = array![1,1,2,3,3,4,5];

    // Rust
    let rust = DiscreteEntropy::new(data.clone());
    let h_rust = rust.global_value();
    let locals_rust = rust.local_values();

    // Python
    let h_py = python::calculate_entropy(&data.to_vec(), "discrete", &[]).expect("python failed");
    let locals_py = python::calculate_local_entropy(&data.to_vec(), "discrete", &[]).expect("python failed");

    assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "discrete global");
    for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
        assert_entropy_values_close(*lr, *lp, 1e-10, 1e-6, "discrete local");
    }
}

#[test]
fn validate_miller_madow_against_python() {
    let data = array![1,1,2,3,3,4,5];

    // Rust
    let rust = MillerMadowEntropy::new(data.clone());
    let h_rust = rust.global_value();
    let locals_rust = rust.local_values();

    // Python
    let h_py = python::calculate_entropy(&data.to_vec(), "miller_madow", &[]).expect("python failed");
    let locals_py = python::calculate_local_entropy(&data.to_vec(), "miller_madow", &[]).expect("python failed");

    assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "miller_madow global");
    for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
        assert_entropy_values_close(*lr, *lp, 1e-10, 1e-6, "miller_madow local");
    }
}

#[test]
fn validate_shrink_against_python() {
    let data = array![1,1,1,2,2,3,4];

    // Rust
    let rust = ShrinkEntropy::new(data.clone());
    let h_rust = rust.global_value();
    let locals_rust = rust.local_values();

    // Python
    let h_py = python::calculate_entropy(&data.to_vec(), "shrink", &[]).expect("python failed");
    let locals_py = python::calculate_local_entropy(&data.to_vec(), "shrink", &[]).expect("python failed");

    assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "shrink global");
    for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
        assert_entropy_values_close(*lr, *lp, 1e-10, 1e-6, "shrink local");
    }
}

#[test]
fn validate_grassberger_against_python() {
    let data = array![1,1,2,3,3,4,5];

    // Rust local values
    let rust = GrassbergerEntropy::new(data.clone());
    let locals_rust = rust.local_values();
    let h_rust = rust.global_value();

    // Python returns locals
    let locals_py = python::calculate_local_entropy(&data.to_vec(), "grassberger", &[]).expect("python failed");
    // The Python global result() reduces local values to a scalar; request global too
    let h_py = python::calculate_entropy(&data.to_vec(), "grassberger", &[]).expect("python failed");

    for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
        assert_entropy_values_close(*lr, *lp, 1e-10, 1e-6, "grassberger local");
    }
    assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "grassberger global");
}

#[test]
fn validate_zhang_against_python() {
    let data = array![1,1,2,3,3,4,5];

    let rust = ZhangEntropy::new(data.clone());
    let locals_rust = rust.local_values();
    let h_rust = rust.global_value();

    let locals_py = python::calculate_local_entropy(&data.to_vec(), "zhang", &[]).expect("python failed");
    let h_py = python::calculate_entropy(&data.to_vec(), "zhang", &[]).expect("python failed");

    for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
        assert_entropy_values_close(*lr, *lp, 1e-10, 1e-6, "zhang local");
    }
    assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "zhang global");
}

#[test]
fn validate_bonachela_against_python() {
    let data = array![1,1,2,3,3,4,5];
    let rust = BonachelaEntropy::new(data.clone());
    let h_rust = rust.global_value();
    let h_py = python::calculate_entropy(&data.to_vec(), "bonachela", &[]).expect("python failed");
    assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "bonachela global");
}

#[test]
fn validate_chao_shen_against_python() {
    let data = array![1,1,2,3,3,4,5];
    let rust = ChaoShenEntropy::new(data.clone());
    let h_rust = rust.global_value();
    let h_py = python::calculate_entropy(&data.to_vec(), "chao_shen", &[]).expect("python failed");
    assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "chao_shen global");
}

#[test]
fn validate_chao_wang_jost_against_python() {
    // Use dataset with some singletons and doubletons
    let data = array![1,1,1,2,2,3,5,8];
    let rust = ChaoWangJostEntropy::new(data.clone());
    let h_rust = rust.global_value();
    let h_py = python::calculate_entropy(&data.to_vec(), "chao_wang_jost", &[]).or_else(|_| {
        // Some Python versions accept alias "cwj"
        python::calculate_entropy(&data.to_vec(), "cwj", &[])
    }).expect("python failed");
    assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "chao_wang_jost global");
}

#[test]
fn validate_bayes_against_python_laplace() {
    let data = array![1,1,2,3,3,4,5];
    let rust = BayesEntropy::new(data.clone(), AlphaParam::Laplace, None);
    let h_rust = rust.global_value();
    // alpha as string literal in Python kwargs: alpha="laplace"
    let h_py = python::calculate_entropy(&data.to_vec(), "bayes", &[("alpha".to_string(), "\"laplace\"".to_string())]).expect("python failed");
    assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "bayes(laplace) global");
}

#[test]
fn validate_bayes_against_python_numeric_alpha() {
    let data = array![1,1,2,3,3,4,5];
    let alpha = 0.5; // Jeffreys
    let rust = BayesEntropy::new(data.clone(), AlphaParam::Value(alpha), None);
    let h_rust = rust.global_value();
    let h_py = python::calculate_entropy(&data.to_vec(), "bayes", &[("alpha".to_string(), format!("{}", alpha))]).expect("python failed");
    assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "bayes(alpha=0.5) global");
}

#[test]
fn validate_ansb_against_python() {
    // Data with coincidences N=7, K=5
    let data = array![1,2,3,4,5,1,2];
    let rust = AnsbEntropy::new(data.clone(), None, 0.1);
    let h_rust = rust.global_value();
    let h_py = python::calculate_entropy(&data.to_vec(), "ansb", &[]).expect("python failed");
    // ANSB can be a bit sensitive; but should match tightly
    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
}

#[test]
fn validate_nsb_against_python() {
    // Use the Python example dataset with coincidences
    let data = array![1,2,3,4,5,1,2];
    let rust = NsbEntropy::new(data.clone(), None);
    let h_rust = rust.global_value();
    let h_py = python::calculate_entropy(&data.to_vec(), "nsb", &[]).expect("python failed");
    // NSB uses numerical integration; use a looser tolerance
    assert_entropy_values_close(h_rust, h_py, 5e-4, 5e-3, "nsb global");
}


#[test]
fn validate_discrete_python_datasets() {
    // Datasets drawn from Python tests (base comparisons ignored; base=e used here)
    let data_cases: Vec<Vec<i32>> = vec![
        vec![1, 0, 1, 0],
        vec![1, 2, 3, 4, 5],
        vec![1, 1, 1, 1, 1],
    ];
    for data in data_cases.into_iter() {
        let arr = ndarray::Array1::from(data.clone());
        let rust = DiscreteEntropy::new(arr);
        let h_rust = rust.global_value();
        let locals_rust = rust.local_values();
        let h_py = python::calculate_entropy(&data, "discrete", &[]).expect("python failed");
        let locals_py = python::calculate_local_entropy(&data, "discrete", &[]).expect("python failed");
        assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "discrete (python datasets) global");
        for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
            assert_entropy_values_close(*lr, *lp, 1e-10, 1e-6, "discrete (python datasets) local");
        }
    }
}

#[test]
fn validate_miller_madow_python_datasets() {
    let data_cases: Vec<Vec<i32>> = vec![
        vec![1, 0, 1, 0],
        vec![1, 2, 3, 4, 5],
        vec![1, 1, 1, 1, 1],
    ];
    for data in data_cases.into_iter() {
        let arr = ndarray::Array1::from(data.clone());
        let rust = MillerMadowEntropy::new(arr);
        let h_rust = rust.global_value();
        let locals_rust = rust.local_values();
        let h_py = python::calculate_entropy(&data, "miller_madow", &[]).expect("python failed");
        let locals_py = python::calculate_local_entropy(&data, "miller_madow", &[]).expect("python failed");
        assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "miller_madow (python datasets) global");
        for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
            assert_entropy_values_close(*lr, *lp, 1e-10, 1e-6, "miller_madow (python datasets) local");
        }
    }
}

#[test]
fn validate_bayes_with_k_parameter_against_python() {
    // From Python tests: ([1,0,1,0], alpha=1.0, K=3) and K=5
    let data = vec![1,0,1,0];
    let rust_k3 = BayesEntropy::new(ndarray::Array1::from(data.clone()), AlphaParam::Laplace, Some(3));
    let h_rust_k3 = rust_k3.global_value();
    let h_py_k3 = python::calculate_entropy(&data, "bayes", &[("alpha".into(), "1.0".into()), ("K".into(), "3".into())]).expect("python failed");
    assert_entropy_values_close(h_rust_k3, h_py_k3, 1e-10, 1e-6, "bayes K=3 global");

    let rust_k5 = BayesEntropy::new(ndarray::Array1::from(data.clone()), AlphaParam::Laplace, Some(5));
    let h_rust_k5 = rust_k5.global_value();
    let h_py_k5 = python::calculate_entropy(&data, "bayes", &[("alpha".into(), "1.0".into()), ("K".into(), "5".into())]).expect("python failed");
    assert_entropy_values_close(h_rust_k5, h_py_k5, 1e-10, 1e-6, "bayes K=5 global");
}

#[test]
fn validate_bonachela_small_dataset_against_python() {
    // From Python tests: [1,1,2]
    let data = vec![1,1,2];
    let rust = BonachelaEntropy::new(ndarray::Array1::from(data.clone()));
    let h_rust = rust.global_value();
    let h_py = python::calculate_entropy(&data, "bonachela", &[]).expect("python failed");
    assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "bonachela [1,1,2] global");
}

#[test]
fn validate_zhang_small_dataset_against_python() {
    // From Python tests: [1,1,2]
    let data = vec![1,1,2];
    let rust = ZhangEntropy::new(ndarray::Array1::from(data.clone()));
    let locals_rust = rust.local_values();
    let h_rust = rust.global_value();
    let locals_py = python::calculate_local_entropy(&data, "zhang", &[]).expect("python failed");
    let h_py = python::calculate_entropy(&data, "zhang", &[]).expect("python failed");
    for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
        assert_entropy_values_close(*lr, *lp, 1e-10, 1e-6, "zhang [1,1,2] local");
    }
    assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "zhang [1,1,2] global");
}

#[test]
fn validate_shrink_python_datasets() {
    // From Python tests: all-same, binary, all-different
    let datasets: Vec<Vec<i32>> = vec![
        vec![1,1,1,1,1],
        vec![1,0,1,0],
        vec![1,2,3,4,5],
    ];
    for data in datasets.into_iter() {
        let rust = ShrinkEntropy::new(ndarray::Array1::from(data.clone()));
        let h_rust = rust.global_value();
        let locals_rust = rust.local_values();
        let h_py = python::calculate_entropy(&data, "shrink", &[]).expect("python failed");
        let locals_py = python::calculate_local_entropy(&data, "shrink", &[]).expect("python failed");
        assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "shrink (python datasets) global");
        for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
            assert_entropy_values_close(*lr, *lp, 1e-10, 1e-6, "shrink (python datasets) local");
        }
    }
}

#[test]
fn validate_chao_shen_python_datasets() {
    // From Python tests: all-different and mixed counts
    let datasets: Vec<Vec<i32>> = vec![
        vec![1,2,3,4,5],
        vec![1,1,2,3,3,4,5],
    ];
    for data in datasets.into_iter() {
        let rust = ChaoShenEntropy::new(ndarray::Array1::from(data.clone()));
        let h_rust = rust.global_value();
        let h_py = python::calculate_entropy(&data, "chao_shen", &[]).expect("python failed");
        assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "chao_shen (python datasets) global");
    }
}

#[test]
fn validate_grassberger_small_dataset_against_python() {
    // From Python style small dataset
    let data = vec![1,1,2];
    let rust = GrassbergerEntropy::new(ndarray::Array1::from(data.clone()));
    let locals_rust = rust.local_values();
    let h_rust = rust.global_value();
    let locals_py = python::calculate_local_entropy(&data, "grassberger", &[]).expect("python failed");
    let h_py = python::calculate_entropy(&data, "grassberger", &[]).expect("python failed");
    for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
        assert_entropy_values_close(*lr, *lp, 1e-10, 1e-6, "grassberger [1,1,2] local");
    }
    assert_entropy_values_close(h_rust, h_py, 1e-10, 1e-6, "grassberger [1,1,2] global");
}
