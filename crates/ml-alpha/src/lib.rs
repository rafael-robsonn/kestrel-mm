#![forbid(unsafe_code)]

mod ekf;
mod gbdt;
pub mod toxicity_model;
#[cfg(feature = "trainer")]
pub mod train;

pub use ekf::{EkfConfig, EkfEstimate, FairValueEkf};
pub use gbdt::{LANES, ObliviousForest, ToxicityEstimate, predict_toxicity, sigmoid};
pub use toxicity_model::{TOXICITY_FOREST, ToxicityForest};

/// Adverse-selection horizon the shipped toxicity model was trained for.
pub const TOXICITY_HORIZON_NS: u64 = 100_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AlphaError {
    #[error("invalid model configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("non-finite observation")]
    NonFiniteInput,
    #[error("innovation covariance not positive definite")]
    SingularInnovation,
    #[error("filter state diverged; re-initialised")]
    Diverged,
    #[error("training failed: {0}")]
    Training(&'static str),
}
