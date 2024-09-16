use clap::ValueEnum;
use serde::{Deserialize, Serialize};

pub mod categorical;
pub mod expectation;
pub mod quantile;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Serialize, Deserialize)]
pub enum LossFunction {
    Huber,
    Squared,
}
