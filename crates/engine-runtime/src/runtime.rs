use core::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use core_affinity::CoreId;
use core_types::{MarketEvent, Quote};

use crate::clock::{TickClock, ticks};
use crate::engine::{EngineConfig, EngineStats, PaperPosition, StrategyEngine};
use crate::error::{EngineError, RuntimeError};
use crate::itch::FrameCursor;
use crate::ring::{Consumer, Producer, spsc};
use crate::telemetry::{
    LatencyHistogram, LatencySummary, RecordKind, TelemetryRecord, write_quote_header, write_quote_row,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pacing {
    /// Replay as fast as the pipeline accepts; measures throughput, latency is queueing.
    Unthrottled,
    /// Release one message every `1/msgs_per_sec` seconds (busy-wait on the TSC).
    FixedRate { msgs_per_sec: f64 },
}

/// Logical core indices into `core_affinity::get_core_ids()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CorePlan {
    pub ingest: Option<usize>,
    pub strategy: Option<usize>,
    pub telemetry: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub engine: EngineConfig,
    pub cores: CorePlan,
    pub md_ring_capacity: usize,
    pub telemetry_ring_capacity: usize,
    pub pacing: Pacing,
    pub quote_log: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Default)]
struct IngestReport {
    messages: u64,
    decode_errors: u64,
    pinned: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct StrategyReport {
    stats: EngineStats,
    paper: PaperPosition,
    final_mid: f64,
    telemetry_drops: u64,
    pinned: bool,
}

#[derive(Clone, Debug, Default)]
struct TelemetryReport {
    tick_to_quote: LatencySummary,
    compute: LatencySummary,
    queue: LatencySummary,
    records: u64,
    quotes_logged: u64,
    skew_clamped: u64,
    pinned: bool,
}

#[derive(Clone, Debug, Default)]
pub struct RunReport {
    pub pacing: Option<Pacing>,
    pub messages: u64,
    pub decode_errors: u64,
    pub engine: EngineStats,
    pub paper: PaperPosition,
    pub final_mid_ticks: f64,
    pub telemetry_drops: u64,
    pub telemetry_records: u64,
    pub quotes_logged: u64,
    /// Cross-core samples where the consumer TSC read earlier than the producer's
    /// (inter-core TSC offset larger than the hop); recorded as 0 ns.
    pub tsc_skew_clamped: u64,
    pub wall_seconds: f64,
    /// Ingest (pre-decode) to quote computed, cross-thread, ns.
    pub tick_to_quote: LatencySummary,
    /// `StrategyEngine::on_event` only, ns.
    pub compute: LatencySummary,
    /// SPSC hand-off: published by ingest to popped by strategy, ns.
    pub queue: LatencySummary,
    pub pinned: [bool; 3],
    pub tsc_ghz: f64,
    pub tick_size: f64,
}

impl fmt::Display for RunReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mode = match self.pacing {
            Some(Pacing::FixedRate { msgs_per_sec }) => format!("paced @ {msgs_per_sec:.0} msg/s"),
            _ => "unthrottled".to_string(),
        };
        writeln!(f, "mode                {mode}")?;
        writeln!(
            f,
            "messages            {} ({} decode errors) in {:.3} s -> {:.2} M msg/s",
            self.messages,
            self.decode_errors,
            self.wall_seconds,
            self.messages as f64 / self.wall_seconds.max(1e-9) / 1e6
        )?;
        writeln!(
            f,
            "engine              events={} quotes={} book_rejects={} strategy_rejects={} filter_resets={}",
            self.engine.events,
            self.engine.quotes,
            self.engine.book_rejects,
            self.engine.strategy_rejects,
            self.engine.filter_resets
        )?;
        writeln!(
            f,
            "telemetry           records={} drops={} quotes_logged={} tsc_skew_clamped={}",
            self.telemetry_records, self.telemetry_drops, self.quotes_logged, self.tsc_skew_clamped
        )?;
        writeln!(
            f,
            "paper               fills={} position={} pnl_mtm={:.2}",
            self.paper.fills,
            core_types::Qty::from_raw(self.paper.position),
            self.paper.mark_to_market(self.final_mid_ticks * self.tick_size)
        )?;
        writeln!(f, "pinned [T1,T2,T3]   {:?} (TSC {:.3} GHz)", self.pinned, self.tsc_ghz)?;
        writeln!(f, "latency ns  tick->quote  {}", self.tick_to_quote)?;
        writeln!(f, "latency ns  on_event     {}", self.compute)?;
        write!(f, "latency ns  spsc hop     {}", self.queue)
    }
}

fn pin(core: Option<CoreId>) -> bool {
    core.is_some_and(core_affinity::set_for_current)
}

fn join<T>(handle: thread::JoinHandle<T>, name: &'static str) -> Result<T, RuntimeError> {
    handle.join().map_err(|_| RuntimeError::ThreadPanicked(name))
}

fn ingest_loop(
    feed: Arc<[u8]>,
    mut tx: Producer<MarketEvent>,
    pacing: Pacing,
    clock: TickClock,
    core: Option<CoreId>,
) -> IngestReport {
    let pinned = pin(core);
    let interval = match pacing {
        Pacing::FixedRate { msgs_per_sec } if msgs_per_sec > 0.0 => clock.from_ns(1e9 / msgs_per_sec),
        _ => 0,
    };
    let mut cursor = FrameCursor::new(&feed);
    let mut report = IngestReport {
        pinned,
        ..IngestReport::default()
    };
    let mut release = ticks();
    loop {
        if interval > 0 {
            while ticks() < release {
                core::hint::spin_loop();
            }
            release = release.wrapping_add(interval);
        }
        let t0 = ticks();
        let Some(item) = cursor.next_event() else { break };
        match item {
            Ok((exch_ts, body)) => {
                tx.push_spin(MarketEvent {
                    seq: report.messages,
                    exch_ts,
                    recv_tsc: t0,
                    body,
                });
                report.messages += 1;
            }
            Err(_) => report.decode_errors += 1,
        }
    }
    report
}

fn strategy_loop(
    cfg: EngineConfig,
    mut rx: Consumer<MarketEvent>,
    mut tx: Producer<TelemetryRecord>,
    core: Option<CoreId>,
) -> Result<StrategyReport, EngineError> {
    let pinned = pin(core);
    let mut engine = StrategyEngine::new(cfg)?;
    let mut drops = 0u64;
    loop {
        let Some(ev) = rx.pop() else {
            if rx.is_finished() {
                break;
            }
            core::hint::spin_loop();
            continue;
        };
        let dequeue_tsc = ticks();
        let result = engine.on_event(&ev);
        let done_tsc = ticks();
        let (kind, quote) = match result {
            Ok(Some(q)) => (RecordKind::Quote, q),
            Ok(None) => (RecordKind::NoQuote, Quote::default()),
            Err(_) => (RecordKind::Rejected, Quote::default()),
        };
        let tox = engine.toxicity();
        let record = TelemetryRecord {
            seq: ev.seq,
            exch_ts: ev.exch_ts,
            recv_tsc: ev.recv_tsc,
            dequeue_tsc,
            done_tsc,
            inventory_lots: engine.inventory_lots(),
            tox_bid: tox.bid as f32,
            tox_ask: tox.ask as f32,
            kind,
            quote,
        };
        if tx.push(record).is_err() {
            drops += 1;
        }
    }
    Ok(StrategyReport {
        stats: *engine.stats(),
        paper: *engine.paper(),
        final_mid: engine.book_state().mid,
        telemetry_drops: drops,
        pinned,
    })
}

fn telemetry_loop(
    mut rx: Consumer<TelemetryRecord>,
    clock: TickClock,
    quote_log: Option<PathBuf>,
    core: Option<CoreId>,
) -> Result<TelemetryReport, RuntimeError> {
    let pinned = pin(core);
    let mut writer = match quote_log {
        Some(path) => {
            let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
            write_quote_header(&mut w)?;
            Some(w)
        }
        None => None,
    };
    let mut tick_to_quote = LatencyHistogram::new();
    let mut compute = LatencyHistogram::new();
    let mut queue = LatencyHistogram::new();
    let mut records = 0u64;
    let mut logged = 0u64;
    let mut skew = 0u64;
    let mut idle = 0u32;
    let mut elapsed = |later: u64, earlier: u64| {
        if later < earlier {
            skew += 1;
        }
        clock.to_ns(later.saturating_sub(earlier)) as u64
    };
    loop {
        let Some(r) = rx.pop() else {
            if rx.is_finished() {
                break;
            }
            idle += 1;
            if idle > 1_024 {
                thread::yield_now();
                idle = 0;
            } else {
                core::hint::spin_loop();
            }
            continue;
        };
        idle = 0;
        records += 1;
        compute.record(elapsed(r.done_tsc, r.dequeue_tsc));
        queue.record(elapsed(r.dequeue_tsc, r.recv_tsc));
        if r.kind == RecordKind::Quote {
            tick_to_quote.record(elapsed(r.done_tsc, r.recv_tsc));
            if let Some(w) = writer.as_mut() {
                write_quote_row(w, &r)?;
                logged += 1;
            }
        }
    }
    if let Some(mut w) = writer {
        w.flush()?;
    }
    Ok(TelemetryReport {
        tick_to_quote: tick_to_quote.summary(),
        compute: compute.summary(),
        queue: queue.summary(),
        records,
        quotes_logged: logged,
        skew_clamped: skew,
        pinned,
    })
}

/// Runs the three-stage pipeline over an in-memory ITCH buffer and blocks until the feed is
/// drained:
///
/// T1 `md-ingest`: frame + decode -> `MarketEvent` -> SPSC A
/// T2 `strategy`: SPSC A -> `StrategyEngine::on_event` -> `TelemetryRecord` -> SPSC B
/// T3 `telemetry`: SPSC B -> latency histograms, optional CSV quote log
///
/// T1 back-pressures on a full ring (spin); T2 never blocks and drops telemetry on a full
/// ring B (counted).
pub fn run_pipeline(cfg: RuntimeConfig, feed: Arc<[u8]>) -> Result<RunReport, RuntimeError> {
    let clock = TickClock::global();
    let (md_tx, md_rx) = spsc::<MarketEvent>(cfg.md_ring_capacity).map_err(EngineError::from)?;
    let (tl_tx, tl_rx) = spsc::<TelemetryRecord>(cfg.telemetry_ring_capacity).map_err(EngineError::from)?;
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let core = |i: Option<usize>| i.and_then(|i| core_ids.get(i).copied());
    let (c_ingest, c_strategy, c_telemetry) = (
        core(cfg.cores.ingest),
        core(cfg.cores.strategy),
        core(cfg.cores.telemetry),
    );

    let start = Instant::now();
    let quote_log = cfg.quote_log.clone();
    let telemetry = thread::Builder::new()
        .name("telemetry".into())
        .spawn(move || telemetry_loop(tl_rx, clock, quote_log, c_telemetry))?;
    let engine_cfg = cfg.engine;
    let strategy = thread::Builder::new()
        .name("strategy".into())
        .spawn(move || strategy_loop(engine_cfg, md_rx, tl_tx, c_strategy))?;
    let pacing = cfg.pacing;
    let ingest = thread::Builder::new()
        .name("md-ingest".into())
        .spawn(move || ingest_loop(feed, md_tx, pacing, clock, c_ingest))?;

    let ingest = join(ingest, "md-ingest")?;
    let strategy = join(strategy, "strategy")??;
    let telemetry = join(telemetry, "telemetry")??;
    let wall_seconds = start.elapsed().as_secs_f64();

    Ok(RunReport {
        pacing: Some(cfg.pacing),
        messages: ingest.messages,
        decode_errors: ingest.decode_errors,
        engine: strategy.stats,
        paper: strategy.paper,
        final_mid_ticks: strategy.final_mid,
        telemetry_drops: strategy.telemetry_drops,
        telemetry_records: telemetry.records,
        quotes_logged: telemetry.quotes_logged,
        tsc_skew_clamped: telemetry.skew_clamped,
        wall_seconds,
        tick_to_quote: telemetry.tick_to_quote,
        compute: telemetry.compute,
        queue: telemetry.queue,
        pinned: [ingest.pinned, strategy.pinned, telemetry.pinned],
        tsc_ghz: clock.ghz(),
        tick_size: cfg.engine.book.tick_size.to_f64(),
    })
}
