//! Trains the adverse-selection model and regenerates `ml-alpha/src/toxicity_model.rs`.
//!
//! Sample: every aggressive print (first fill of each aggressor burst), features taken from
//! the engine state *before* the print, oriented to the aggressor. Label: 1 if the mid
//! `TOXICITY_HORIZON_NS` later is beyond the touch price the passive side traded at, i.e.
//! the maker's 100 ms markout is negative.

use std::path::PathBuf;
use std::time::Instant;

use core_types::{EventBody, FEATURE_NAMES, N_FEATURES, Side};
use engine_runtime::sim::{MarketSimulator, SimConfig};
use engine_runtime::{EngineConfig, RuntimeError, StrategyEngine};
use ml_alpha::train::{TrainParams, log_loss, roc_auc, train};
use ml_alpha::{TOXICITY_HORIZON_NS, sigmoid};

struct Dataset {
    x: Vec<[f64; N_FEATURES]>,
    y: Vec<f64>,
    events: usize,
    seconds: f64,
    mean_spread: f64,
}

fn collect(seed: u64, n_events: usize) -> Result<Dataset, RuntimeError> {
    let sim_cfg = SimConfig {
        seed,
        ..SimConfig::default()
    };
    let mut sim = MarketSimulator::new(sim_cfg)?;
    let events = sim.generate(n_events);
    let mut engine = StrategyEngine::new(EngineConfig::for_instrument(
        sim_cfg.start_price,
        sim_cfg.tick,
        sim_cfg.lot,
    )?)?;

    let mut mids: Vec<(u64, f64)> = Vec::with_capacity(events.len());
    let mut raw: Vec<([f64; N_FEATURES], u64, f64, f64)> = Vec::new();
    let mut last_burst: Option<(u64, Side)> = None;
    let mut prev_ok = false;
    let mut spread_sum = 0.0;

    for ev in &events {
        let pre_features = *engine.features();
        let pre_state = *engine.book_state();
        let aggressor = match ev.body {
            EventBody::Execute { order_id, .. } => engine.book().order(order_id).map(|o| o.side.opposite()),
            EventBody::Trade { side, .. } => Some(side),
            _ => None,
        };
        let ok = matches!(engine.on_event(ev), Ok(Some(_)));
        if let Some(side) = aggressor
            && prev_ok
            && last_burst != Some((ev.exch_ts.nanos(), side))
        {
            last_burst = Some((ev.exch_ts.nanos(), side));
            let sign = side.sign() as f64;
            let touch = if side == Side::Bid {
                pre_state.best_ask
            } else {
                pre_state.best_bid
            };
            raw.push((pre_features.oriented(sign).values, ev.exch_ts.nanos(), sign, touch));
        }
        if ok {
            let st = engine.book_state();
            mids.push((ev.exch_ts.nanos(), st.mid));
            spread_sum += st.best_ask - st.best_bid;
        }
        prev_ok = ok;
    }

    let end = mids.last().map_or(0, |m| m.0);
    let mut x = Vec::with_capacity(raw.len());
    let mut y = Vec::with_capacity(raw.len());
    for (features, ts, sign, touch) in raw {
        let target = ts + TOXICITY_HORIZON_NS;
        if target > end {
            continue;
        }
        let i = mids.partition_point(|m| m.0 <= target);
        let Some(&(_, mid_h)) = mids.get(i.saturating_sub(1)) else {
            continue;
        };
        x.push(features);
        y.push(if sign * (mid_h - touch) > 0.0 { 1.0 } else { 0.0 });
    }
    let seconds = (events.last().map_or(0, |e| e.exch_ts.nanos()) - sim_cfg.start_ts.nanos()) as f64 * 1e-9;
    Ok(Dataset {
        x,
        y,
        events: events.len(),
        seconds,
        mean_spread: spread_sum / mids.len().max(1) as f64,
    })
}

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Result<T, RuntimeError> {
    match args.iter().position(|a| a == name) {
        Some(i) => args
            .get(i + 1)
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| RuntimeError::Argument(format!("{name} expects a value"))),
        None => Ok(default),
    }
}

fn main() -> Result<(), RuntimeError> {
    let args: Vec<String> = std::env::args().collect();
    let train_events: usize = arg(&args, "--events", 4_000_000)?;
    let holdout_events: usize = arg(&args, "--holdout", 1_000_000)?;
    let params = TrainParams {
        trees: arg(&args, "--trees", 48)?,
        depth: arg(&args, "--depth", 5)?,
        learning_rate: arg(&args, "--lr", 0.1)?,
        ..TrainParams::default()
    };
    let default_out = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../ml-alpha/src/toxicity_model.rs"
    ));
    let out: PathBuf = arg(&args, "--out", default_out)?;
    let dry_run = args.iter().any(|a| a == "--dry-run");

    let t0 = Instant::now();
    let train_set = collect(0xA11CE, train_events)?;
    let test_set = collect(0xB0B, holdout_events)?;
    println!(
        "simulated {} + {} events ({:.1} s + {:.1} s exchange time), mean spread {:.2} ticks, {:.1}s wall",
        train_set.events,
        test_set.events,
        train_set.seconds,
        test_set.seconds,
        train_set.mean_spread,
        t0.elapsed().as_secs_f64()
    );
    let base_rate = train_set.y.iter().sum::<f64>() / train_set.y.len().max(1) as f64;
    println!(
        "samples train={} holdout={} base_rate={:.4}",
        train_set.x.len(),
        test_set.x.len(),
        base_rate
    );

    let t1 = Instant::now();
    let forest = train(&train_set.x, &train_set.y, &params).map_err(engine_runtime::EngineError::from)?;
    let p_test: Vec<f64> = test_set.x.iter().map(|r| sigmoid(forest.margin(r))).collect();
    let p_train: Vec<f64> = train_set.x.iter().map(|r| sigmoid(forest.margin(r))).collect();
    let test_base = test_set.y.iter().sum::<f64>() / test_set.y.len().max(1) as f64;
    let auc = roc_auc(&p_test, &test_set.y);
    let ll = log_loss(&p_test, &test_set.y);
    let ll_base = log_loss(&vec![base_rate; test_set.y.len()], &test_set.y);
    println!(
        "trained {} trees depth {} in {:.2}s",
        params.trees,
        params.depth,
        t1.elapsed().as_secs_f64()
    );
    println!(
        "train   auc={:.4} logloss={:.4}",
        roc_auc(&p_train, &train_set.y),
        log_loss(&p_train, &train_set.y)
    );
    println!("holdout auc={auc:.4} logloss={ll:.4} (base-rate logloss {ll_base:.4}, holdout base rate {test_base:.4})");
    println!("split counts:\n{}", forest.describe());
    for (name, (lo, hi)) in FEATURE_NAMES.iter().zip(feature_ranges(&train_set.x)) {
        println!("  {name:<16} p01={lo:>9.4} p99={hi:>9.4}");
    }

    if dry_run {
        return Ok(());
    }
    let header = "// @generated by `cargo run --release -p engine-runtime --features trainer --bin train-toxicity`. Do not edit.";
    let src = forest
        .to_rust_source(
            header,
            &[
                ("TRAIN_SAMPLES", train_set.x.len() as f64),
                ("BASE_RATE", base_rate),
                ("HOLDOUT_AUC", auc),
                ("HOLDOUT_LOGLOSS", ll),
                ("HOLDOUT_BASE_LOGLOSS", ll_base),
            ],
        )
        .map_err(engine_runtime::EngineError::from)?;
    std::fs::write(&out, src)?;
    println!("wrote {}", out.display());
    Ok(())
}

fn feature_ranges(x: &[[f64; N_FEATURES]]) -> [(f64, f64); N_FEATURES] {
    let mut out = [(0.0, 0.0); N_FEATURES];
    for (f, slot) in out.iter_mut().enumerate() {
        let mut col: Vec<f64> = x.iter().map(|r| r[f]).collect();
        if col.is_empty() {
            continue;
        }
        col.sort_unstable_by(f64::total_cmp);
        *slot = (col[col.len() / 100], col[col.len() * 99 / 100]);
    }
    out
}
