use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};

pub mod data;
pub mod hardware;

pub use data::*;
pub use hardware::*;
