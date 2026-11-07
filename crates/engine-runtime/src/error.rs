use microstructure_simd::FeatureError;
use ml_alpha::AlphaError;
use orderbook_l3::BookError;
use strategy_mm::StrategyError;

use crate::itch::ItchError;
use crate::ring::RingError;

/// Hot-path error. `Copy`, no heap payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Book(#[from] BookError),
    #[error(transparent)]
    Feature(#[from] FeatureError),
    #[error(transparent)]
    Alpha(#[from] AlphaError),
    #[error(transparent)]
    Strategy(#[from] StrategyError),
    #[error(transparent)]
    Itch(#[from] ItchError),
    #[error(transparent)]
    Ring(#[from] RingError),
    #[error("invalid configuration: {0}")]
    Config(&'static str),
}

/// Setup, I/O and thread-management errors. Never constructed on the hot path.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("thread `{0}` panicked")]
    ThreadPanicked(&'static str),
    #[error("invalid argument: {0}")]
    Argument(String),
}
