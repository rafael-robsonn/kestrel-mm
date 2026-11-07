#![forbid(unsafe_code)]

mod engine;
mod ladder;
mod vpin;

pub use engine::{BookState, FeatureConfig, FeatureEngine, VPIN_BUCKETS};
pub use ladder::{LadderLanes, MicroPrice, hsum, ofi_levels, weighted_micro_price};
pub use vpin::Vpin;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FeatureError {
    #[error("invalid feature configuration: {0}")]
    InvalidConfig(&'static str),
}
