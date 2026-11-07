use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use engine_runtime::sim::{MarketSimulator, SimConfig, encode_itch};
use engine_runtime::{CorePlan, EngineConfig, EngineError, Pacing, RuntimeConfig, RuntimeError, run_pipeline};

const USAGE: &str = "\
mm-engine [--events N] [--seed S] [--mode throughput|latency|both] [--rate MSG_PER_S]
          [--cores INGEST,STRATEGY,TELEMETRY] [--quotes PATH]";

struct Args {
    events: usize,
    seed: u64,
    mode: String,
    rate: f64,
    cores: CorePlan,
    quotes: Option<PathBuf>,
}

fn parse_args() -> Result<Args, RuntimeError> {
    let mut args = Args {
        events: 2_000_000,
        seed: 0xC0FFEE,
        mode: "both".into(),
        rate: 500_000.0,
        cores: CorePlan {
            ingest: Some(2),
            strategy: Some(4),
            telemetry: Some(6),
        },
        quotes: None,
    };
    let mut it = std::env::args().skip(1);
    let bad = |what: &str| RuntimeError::Argument(format!("{what}\n{USAGE}"));
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| bad(&format!("{flag} expects a value")));
        match flag.as_str() {
            "--events" => args.events = value()?.parse().map_err(|_| bad("--events"))?,
            "--seed" => args.seed = value()?.parse().map_err(|_| bad("--seed"))?,
            "--mode" => args.mode = value()?,
            "--rate" => args.rate = value()?.parse().map_err(|_| bad("--rate"))?,
            "--quotes" => args.quotes = Some(PathBuf::from(value()?)),
            "--cores" => {
                let v = value()?;
                let parts: Vec<Option<usize>> = v.split(',').map(|p| p.trim().parse().ok()).collect();
                if parts.len() != 3 {
                    return Err(bad("--cores expects three comma-separated indices"));
                }
                args.cores = CorePlan {
                    ingest: parts[0],
                    strategy: parts[1],
                    telemetry: parts[2],
                };
            }
            "--no-pin" => args.cores = CorePlan::default(),
            "-h" | "--help" => return Err(bad("help")),
            other => return Err(bad(&format!("unknown flag {other}"))),
        }
    }
    if !matches!(args.mode.as_str(), "throughput" | "latency" | "both") {
        return Err(bad("--mode"));
    }
    Ok(args)
}

fn main() -> Result<(), RuntimeError> {
    let args = parse_args()?;
    let sim_cfg = SimConfig {
        seed: args.seed,
        ..SimConfig::default()
    };

    let t0 = Instant::now();
    let mut sim = MarketSimulator::new(sim_cfg)?;
    let events = sim.generate(args.events);
    let feed: Arc<[u8]> = encode_itch(&events, "SIMX").map_err(EngineError::from)?.into();
    println!(
        "feed: {} ITCH messages, {:.1} MiB, generated in {:.2}s",
        events.len(),
        feed.len() as f64 / (1 << 20) as f64,
        t0.elapsed().as_secs_f64()
    );

    let engine = EngineConfig::for_instrument(sim_cfg.start_price, sim_cfg.tick, sim_cfg.lot)?;
    let base = RuntimeConfig {
        engine,
        cores: args.cores,
        md_ring_capacity: 1 << 16,
        telemetry_ring_capacity: 1 << 16,
        pacing: Pacing::Unthrottled,
        quote_log: None,
    };

    if matches!(args.mode.as_str(), "throughput" | "both") {
        let quote_log = if args.mode == "throughput" {
            args.quotes.clone()
        } else {
            None
        };
        let report = run_pipeline(
            RuntimeConfig {
                quote_log,
                ..base.clone()
            },
            Arc::clone(&feed),
        )?;
        println!("\n{report}");
    }
    if matches!(args.mode.as_str(), "latency" | "both") {
        let paced = RuntimeConfig {
            pacing: Pacing::FixedRate {
                msgs_per_sec: args.rate,
            },
            quote_log: args.quotes,
            ..base
        };
        let report = run_pipeline(paced, feed)?;
        println!("\n{report}");
    }
    Ok(())
}
