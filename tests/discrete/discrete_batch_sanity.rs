use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::discrete::ansb::AnsbEntropy;
use infomeasure::estimators::approaches::discrete::bayes::{AlphaParam, BayesEntropy};
use infomeasure::estimators::approaches::discrete::bonachela::BonachelaEntropy;
use infomeasure::estimators::approaches::discrete::chao_shen::ChaoShenEntropy;
use infomeasure::estimators::approaches::discrete::chao_wang_jost::ChaoWangJostEntropy;
use infomeasure::estimators::approaches::discrete::grassberger::GrassbergerEntropy;
use infomeasure::estimators::approaches::discrete::miller_madow::MillerMadowEntropy;
use infomeasure::estimators::approaches::discrete::mle::DiscreteEntropy;
use infomeasure::estimators::approaches::discrete::nsb::NsbEntropy;
use infomeasure::estimators::approaches::discrete::shrink::ShrinkEntropy;
use infomeasure::estimators::approaches::discrete::zhang::ZhangEntropy;
use infomeasure::estimators::{GlobalValue, LocalValues, OptionalLocalValues};
use ndarray::{Array2, array};

#[test]
fn batch_rows_matches_per_row_all_discrete() {
    // 3 rows with varied distributions; keep equal lengths per row
    let data: Array2<i32> = array![
        [1, 1, 2, 3, 3, 4, 5, 5], // mixed
        [0, 1, 0, 1, 0, 1, 0, 1], // binary alternating
        [5, 5, 5, 5, 5, 5, 5, 5], // constant
    ];

    // 1) Discrete (MLE)
    {
        let batch = DiscreteEntropy::from_rows(data.clone());
        for (i, row) in data.rows().into_iter().enumerate() {
            let h_row = DiscreteEntropy::new(row.to_owned()).global_value();
            assert_abs_diff_eq!(batch[i].global_value(), h_row, epsilon = 1e-12);
            // local/global relation
            let lm = batch[i].local_values().mean().unwrap();
            assert_abs_diff_eq!(batch[i].global_value(), lm, epsilon = 1e-12);
        }
    }

    // 2) Miller–Madow
    {
        let batch = MillerMadowEntropy::from_rows(data.clone());
        for (i, row) in data.rows().into_iter().enumerate() {
            let est = MillerMadowEntropy::new(row.to_owned());
            assert_abs_diff_eq!(batch[i].global_value(), est.global_value(), epsilon = 1e-12);
            let lm = batch[i].local_values().mean().unwrap();
            assert_abs_diff_eq!(batch[i].global_value(), lm, epsilon = 1e-12);
        }
    }

    // 3) Shrink
    {
        let batch = ShrinkEntropy::from_rows(data.clone());
        for (i, row) in data.rows().into_iter().enumerate() {
            let est = ShrinkEntropy::new(row.to_owned());
            assert_abs_diff_eq!(batch[i].global_value(), est.global_value(), epsilon = 1e-12);
            let lm = batch[i].local_values().mean().unwrap();
            assert_abs_diff_eq!(batch[i].global_value(), lm, epsilon = 1e-12);
        }
    }

    // 4) Grassberger
    {
        let batch = GrassbergerEntropy::from_rows(data.clone());
        for (i, row) in data.rows().into_iter().enumerate() {
            let est = GrassbergerEntropy::new(row.to_owned());
            assert_abs_diff_eq!(batch[i].global_value(), est.global_value(), epsilon = 1e-9);
            let lm = batch[i].local_values().mean().unwrap();
            assert_abs_diff_eq!(batch[i].global_value(), lm, epsilon = 1e-9);
        }
    }

    // 5) Zhang
    {
        let batch = ZhangEntropy::from_rows(data.clone());
        for (i, row) in data.rows().into_iter().enumerate() {
            let est = ZhangEntropy::new(row.to_owned());
            assert_abs_diff_eq!(batch[i].global_value(), est.global_value(), epsilon = 1e-9);
            let lm = batch[i].local_values().mean().unwrap();
            assert_abs_diff_eq!(batch[i].global_value(), lm, epsilon = 1e-9);
        }
    }

    // 6) Bonachela (global only)
    {
        let batch = BonachelaEntropy::from_rows(data.clone());
        for (i, row) in data.rows().into_iter().enumerate() {
            let est = BonachelaEntropy::new(row.to_owned());
            assert_abs_diff_eq!(batch[i].global_value(), est.global_value(), epsilon = 1e-12);
            assert!(!batch[i].supports_local());
        }
    }

    // 7) Chao-Shen (global only)
    {
        let batch = ChaoShenEntropy::from_rows(data.clone());
        for (i, row) in data.rows().into_iter().enumerate() {
            let est = ChaoShenEntropy::new(row.to_owned());
            assert_abs_diff_eq!(batch[i].global_value(), est.global_value(), epsilon = 1e-12);
            assert!(!batch[i].supports_local());
        }
    }

    // 8) Chao-Wang-Jost (global only)
    {
        let batch = ChaoWangJostEntropy::from_rows(data.clone());
        for (i, row) in data.rows().into_iter().enumerate() {
            let est = ChaoWangJostEntropy::new(row.to_owned());
            assert_abs_diff_eq!(batch[i].global_value(), est.global_value(), epsilon = 1e-12);
            assert!(!batch[i].supports_local());
        }
    }

    // 9) ANSB (global only)
    {
        let batch = AnsbEntropy::from_rows(data.clone(), None, 0.3);
        for (i, row) in data.rows().into_iter().enumerate() {
            let est = AnsbEntropy::new(row.to_owned(), None, 0.3);
            let b = batch[i].global_value();
            let r = est.global_value();
            if b.is_nan() || r.is_nan() {
                assert!(b.is_nan() && r.is_nan());
            } else {
                assert_abs_diff_eq!(b, r, epsilon = 1e-12);
            }
            assert!(!batch[i].supports_local());
        }
    }

    // 10) Bayes (global only) with Laplace alpha
    {
        let batch = BayesEntropy::from_rows(data.clone(), AlphaParam::Laplace, None);
        for (i, row) in data.rows().into_iter().enumerate() {
            let est = BayesEntropy::new(row.to_owned(), AlphaParam::Laplace, None);
            assert_abs_diff_eq!(batch[i].global_value(), est.global_value(), epsilon = 1e-12);
            assert!(!batch[i].supports_local());
        }
    }

    // 11) NSB (global only)
    {
        let batch = NsbEntropy::from_rows(data.clone(), None);
        for (i, row) in data.rows().into_iter().enumerate() {
            let est = NsbEntropy::new(row.to_owned(), None);
            let b = batch[i].global_value();
            let r = est.global_value();
            if b.is_nan() || r.is_nan() {
                assert!(b.is_nan() && r.is_nan());
            } else {
                assert_abs_diff_eq!(b, r, epsilon = 5e-3);
            }
            assert!(!batch[i].supports_local());
        }
    }
}
