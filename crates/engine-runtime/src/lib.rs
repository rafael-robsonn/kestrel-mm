#![deny(unsafe_code)]

#[allow(unsafe_code)]
pub mod clock;
pub mod engine;
pub mod error;
pub mod itch;
#[allow(unsafe_code)]
pub mod ring;
pub mod runtime;
pub mod sim;
pub mod telemetry;

pub use engine::{EngineConfig, EngineStats, PaperPosition, StrategyEngine};
pub use error::{EngineError, RuntimeError};
pub use runtime::{CorePlan, Pacing, RunReport, RuntimeConfig, run_pipeline};
