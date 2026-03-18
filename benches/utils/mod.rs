#![allow(unused_imports, dead_code)]

use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};

pub mod data;
pub mod hardware;

pub use data::*;
pub use hardware::*;
