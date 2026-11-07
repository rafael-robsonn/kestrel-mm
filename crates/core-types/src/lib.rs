#![forbid(unsafe_code)]

mod cache;
mod event;
mod features;
mod fixed;
mod ids;
mod market;

pub use cache::CachePadded;
pub use event::{EventBody, Fill, MarketEvent};
pub use features::{FEATURE_NAMES, FeatureVector, N_FEATURES, SIGNED_FEATURES, feature};
pub use fixed::{Price, Qty, SCALE, notional};
pub use ids::{OrderId, Side, Timestamp};
pub use market::{ABSENT_ASK_PX, ABSENT_BID_PX, DEPTH_LEVELS, DepthSnapshot, Quote};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CoreError {
    #[error("non-finite floating-point input")]
    NonFinite,
    #[error("fixed-point overflow")]
    Overflow,
    #[error("tick size must be strictly positive")]
    InvalidTick,
}
