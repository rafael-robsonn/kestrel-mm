#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use engine_runtime::sim::{MarketSimulator, SimConfig, encode_itch};
use engine_runtime::{CorePlan, EngineConfig, Pacing, RuntimeConfig, run_pipeline};

fn feed(events: usize) -> (SimConfig, Arc<[u8]>, usize) {
    let cfg = SimConfig {
        seed: 99,
        ..SimConfig::default()
    };
    let mut sim = MarketSimulator::new(cfg).unwrap();
    let ev = sim.generate(events);
    let n = ev.len();
    (cfg, encode_itch(&ev, "SIMX").unwrap().into(), n)
}

fn runtime(cfg: &SimConfig, pacing: Pacing) -> RuntimeConfig {
    RuntimeConfig {
        engine: EngineConfig::for_instrument(cfg.start_price, cfg.tick, cfg.lot).unwrap(),
        cores: CorePlan::default(),
        md_ring_capacity: 1 << 12,
        telemetry_ring_capacity: 1 << 16,
        pacing,
        quote_log: None,
    }
}

#[test]
fn three_thread_pipeline_processes_every_message() {
    let (cfg, feed, n) = feed(200_000);
    let report = run_pipeline(runtime(&cfg, Pacing::Unthrottled), feed).unwrap();
    assert_eq!(report.messages as usize, n);
    assert_eq!(report.decode_errors, 0);
    assert_eq!(report.engine.events as usize, n);
    assert_eq!(report.engine.book_rejects, 0);
    assert_eq!(report.telemetry_records + report.telemetry_drops, report.messages);
    assert!(report.engine.quotes as f64 > 0.95 * n as f64);
    assert!(report.compute.count > 0 && report.compute.p50 > 0);
}

#[test]
fn paced_mode_writes_quote_log() {
    let (cfg, feed, n) = feed(20_000);
    let path = std::env::temp_dir().join(format!("mm-engine-quotes-{}.csv", std::process::id()));
    let rt = RuntimeConfig {
        quote_log: Some(path.clone()),
        ..runtime(
            &cfg,
            Pacing::FixedRate {
                msgs_per_sec: 2_000_000.0,
            },
        )
    };
    let report = run_pipeline(rt, feed).unwrap();
    assert_eq!(report.messages as usize, n);
    let csv = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert!(lines[0].starts_with("seq,exch_ts_ns,bid_px"));
    assert_eq!(lines.len() as u64 - 1, report.quotes_logged);
    assert!(report.quotes_logged > 0);
}
