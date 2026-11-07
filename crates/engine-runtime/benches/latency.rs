#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use core_types::{DepthSnapshot, EventBody, Fill, MarketEvent, OrderId, Price, Qty, Side, Timestamp};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use engine_runtime::itch::{FrameCursor, ItchEncoder, decode_frame};
use engine_runtime::ring::spsc;
use engine_runtime::sim::{MarketSimulator, SimConfig, encode_itch};
use engine_runtime::{EngineConfig, StrategyEngine};
use microstructure_simd::FeatureEngine;
use ml_alpha::{FairValueEkf, TOXICITY_FOREST, ToxicityEstimate, predict_toxicity};
use orderbook_l3::{DiscardFills, LimitOrder, OrderBook, TimeInForce};
use strategy_mm::{AvellanedaStoikov, MarketInputs};

const STREAM_EVENTS: usize = 1_000_000;

struct Fixture {
    sim: SimConfig,
    engine_cfg: EngineConfig,
    events: Vec<MarketEvent>,
    feed: Vec<u8>,
    frame_offsets: Vec<usize>,
}

fn fixture() -> Fixture {
    let sim = SimConfig {
        seed: 0xBE7C4,
        ..SimConfig::default()
    };
    let events = MarketSimulator::new(sim).unwrap().generate(STREAM_EVENTS);
    let feed = encode_itch(&events, "SIMX").unwrap();
    let mut frame_offsets = Vec::with_capacity(events.len());
    let mut pos = 0;
    while pos < feed.len() {
        frame_offsets.push(pos);
        pos += decode_frame(&feed[pos..]).unwrap().0;
    }
    let engine_cfg = EngineConfig::for_instrument(sim.start_price, sim.tick, sim.lot).unwrap();
    Fixture {
        sim,
        engine_cfg,
        events,
        feed,
        frame_offsets,
    }
}

/// Engine warmed through the first half of the stream; returns it with a mid-session state.
fn warm_engine(fx: &Fixture) -> StrategyEngine {
    let mut engine = StrategyEngine::new(fx.engine_cfg).unwrap();
    for ev in &fx.events[..STREAM_EVENTS / 2] {
        let _ = engine.on_event(ev);
    }
    engine
}

fn bench_itch(c: &mut Criterion, fx: &Fixture) {
    let mut g = c.benchmark_group("itch");
    g.throughput(Throughput::Elements(1));
    let mut add = Vec::new();
    ItchEncoder::new(1, "SIMX")
        .encode(
            &mut add,
            Timestamp(34_200_000_000_000),
            &EventBody::Add {
                order_id: OrderId(1),
                side: Side::Bid,
                price: Price::from_raw(10_000_000_000),
                qty: Qty::from_raw(10_000_000_000),
            },
        )
        .unwrap();
    g.bench_function("decode_add", |b| b.iter(|| decode_frame(black_box(&add)).unwrap()));
    let mut i = 0usize;
    g.bench_function("decode_stream", |b| {
        b.iter(|| {
            let off = fx.frame_offsets[i];
            i = if i + 1 == fx.frame_offsets.len() { 0 } else { i + 1 };
            decode_frame(black_box(&fx.feed[off..])).unwrap()
        })
    });
    g.finish();
}

fn bench_book(c: &mut Criterion, fx: &Fixture) {
    let mut g = c.benchmark_group("orderbook");
    g.throughput(Throughput::Elements(1));
    let engine = warm_engine(fx);
    let mut book = OrderBook::new(fx.engine_cfg.book).unwrap();
    for ev in &fx.events[..STREAM_EVENTS / 2] {
        let _ = book.apply(&ev.body, &mut DiscardFills);
    }
    let best_bid = engine.book().best_bid().unwrap().price;
    let mut id = 1u64 << 40;
    g.bench_function("add_cancel_at_touch", |b| {
        b.iter(|| {
            id += 1;
            let order = LimitOrder {
                id: OrderId(id),
                side: Side::Bid,
                price: best_bid,
                qty: Qty::from_raw(100 * core_types::SCALE),
                tif: TimeInForce::Gtc,
            };
            book.submit(black_box(order), &mut DiscardFills).unwrap();
            book.cancel(OrderId(id)).unwrap()
        })
    });

    let mut sweep_book = OrderBook::new(fx.engine_cfg.book).unwrap();
    let tick = fx.sim.tick.raw();
    let base = fx.sim.start_price.raw();
    g.bench_function("sweep_5_levels", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for k in 0..iters {
                sweep_book.reset();
                for lvl in 0..5 {
                    for j in 0..4 {
                        let o = LimitOrder {
                            id: OrderId(lvl * 10 + j),
                            side: Side::Ask,
                            price: Price::from_raw(base + (lvl as i64 + 1) * tick),
                            qty: Qty::from_raw(100 * core_types::SCALE),
                            tif: TimeInForce::Gtc,
                        };
                        sweep_book.submit(o, &mut DiscardFills).unwrap();
                    }
                }
                let taker = LimitOrder {
                    id: OrderId(1_000 + k),
                    side: Side::Bid,
                    price: Price::from_raw(base + 6 * tick),
                    qty: Qty::from_raw(2_000 * core_types::SCALE),
                    tif: TimeInForce::Ioc,
                };
                let t = Instant::now();
                black_box(sweep_book.submit(black_box(taker), &mut DiscardFills).unwrap());
                total += t.elapsed();
            }
            total
        })
    });

    let mut snap = DepthSnapshot::EMPTY;
    g.bench_function("snapshot_8_levels", |b| {
        b.iter(|| engine.book().snapshot(black_box(Timestamp(1)), &mut snap))
    });
    g.finish();
}

fn bench_models(c: &mut Criterion, fx: &Fixture) {
    let engine = warm_engine(fx);
    let mut g = c.benchmark_group("models");
    g.throughput(Throughput::Elements(1));

    let mut snaps = [DepthSnapshot::EMPTY; 2];
    let mut replay = OrderBook::new(fx.engine_cfg.book).unwrap();
    for ev in &fx.events[..STREAM_EVENTS / 2] {
        let _ = replay.apply(&ev.body, &mut DiscardFills);
    }
    replay.snapshot(Timestamp(1_000_000), &mut snaps[0]);
    let _ = replay.apply(&fx.events[STREAM_EVENTS / 2].body, &mut DiscardFills);
    replay.snapshot(Timestamp(2_000_000), &mut snaps[1]);
    let ring: Vec<DepthSnapshot> = (0..64u64)
        .map(|k| DepthSnapshot {
            ts: Timestamp(1_000_000 + k * 50_000),
            ..snaps[(k & 1) as usize]
        })
        .collect();
    let mut features = FeatureEngine::new(fx.engine_cfg.features).unwrap();
    let fill = Fill {
        maker_id: OrderId(1),
        taker_id: OrderId::NONE,
        maker_side: Side::Ask,
        price: Price::ZERO,
        qty: Qty::from_raw(100 * core_types::SCALE),
    };
    let mut k = 0usize;
    g.bench_function("features_update_simd", |b| {
        b.iter(|| {
            k = (k + 1) & 63;
            features.on_trade(&fill);
            black_box(features.update(black_box(&ring[k])).copied())
        })
    });

    let fv = *engine.features();
    g.bench_function("gbdt_toxicity_both_sides", |b| {
        b.iter(|| predict_toxicity(&TOXICITY_FOREST, black_box(&fv)))
    });

    let mut ekf = FairValueEkf::new(fx.engine_cfg.ekf).unwrap();
    let step = fx.engine_cfg.ekf.sample_interval_ns;
    let mut t = 0u64;
    g.bench_function("ekf_step", |b| {
        b.iter(|| {
            t += step;
            let z = 10_000.0 + ((t / step) % 7) as f64 * 0.5;
            let stepped = ekf.observe(black_box(z), Timestamp(t)).unwrap();
            debug_assert!(stepped);
            stepped
        })
    });

    let quoter = AvellanedaStoikov::new(fx.engine_cfg.strategy).unwrap();
    let inputs = MarketInputs {
        ts: Timestamp(0),
        reference: 10_000.3,
        best_bid: 10_000.0,
        best_ask: 10_001.0,
        sigma: 4.0,
        drift: 2.0,
    };
    let tox = ToxicityEstimate { bid: 0.3, ask: 0.45 };
    g.bench_function("avellaneda_stoikov_quote", |b| {
        b.iter(|| quoter.quote(black_box(&inputs), black_box(3), black_box(tox)).unwrap())
    });
    g.finish();
}

fn bench_spsc(c: &mut Criterion) {
    let mut g = c.benchmark_group("spsc");
    g.throughput(Throughput::Elements(1));
    let (mut tx, mut rx) = spsc::<MarketEvent>(1 << 10).unwrap();
    let ev = MarketEvent {
        seq: 0,
        exch_ts: Timestamp(0),
        recv_tsc: 0,
        body: EventBody::Delete { order_id: OrderId(1) },
    };
    g.bench_function("push_pop_same_thread", |b| {
        b.iter(|| {
            tx.push(black_box(ev)).unwrap();
            rx.pop().unwrap()
        })
    });
    g.finish();
}

fn bench_e2e(c: &mut Criterion, fx: &Fixture) {
    let mut g = c.benchmark_group("e2e");
    g.throughput(Throughput::Elements(1));
    g.measurement_time(Duration::from_secs(10));
    let mut engine = StrategyEngine::new(fx.engine_cfg).unwrap();
    let mut cursor = FrameCursor::new(&fx.feed);
    let mut seq = 0u64;
    g.bench_function("itch_bytes_to_quote", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            let mut left = iters;
            while left > 0 {
                if cursor.is_exhausted() {
                    engine.reset();
                    cursor = FrameCursor::new(&fx.feed);
                }
                let start = Instant::now();
                while left > 0 {
                    let Some(Ok((exch_ts, body))) = cursor.next_event() else {
                        break;
                    };
                    let ev = MarketEvent {
                        seq,
                        exch_ts,
                        recv_tsc: 0,
                        body,
                    };
                    black_box(engine.on_event(black_box(&ev)).ok());
                    seq += 1;
                    left -= 1;
                }
                total += start.elapsed();
            }
            total
        })
    });
    g.finish();
}

/// Same logical core as the strategy thread in `mm-engine` (CorePlan default).
const BENCH_CORE: usize = 4;

fn benches(c: &mut Criterion) {
    if let Some(core) = core_affinity::get_core_ids().and_then(|ids| ids.get(BENCH_CORE).copied()) {
        core_affinity::set_for_current(core);
    }
    let fx = fixture();
    bench_itch(c, &fx);
    bench_book(c, &fx);
    bench_models(c, &fx);
    bench_spsc(c);
    bench_e2e(c, &fx);
}

criterion_group! {
    name = latency;
    config = Criterion::default().warm_up_time(Duration::from_secs(2)).measurement_time(Duration::from_secs(4)).noise_threshold(0.03);
    targets = benches
}
criterion_main!(latency);
