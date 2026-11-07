#![allow(clippy::unwrap_used, clippy::expect_used)]

use core_types::{MarketEvent, Price, Qty, Side};
use engine_runtime::itch::FrameCursor;
use engine_runtime::sim::{MarketSimulator, SimConfig, encode_itch};
use engine_runtime::{EngineConfig, EngineError, StrategyEngine};
use orderbook_l3::LevelView;

fn levels(book: &orderbook_l3::OrderBook, side: Side) -> Vec<LevelView> {
    let mut buf = vec![
        LevelView {
            price: Price::ZERO,
            qty: Qty::ZERO,
            orders: 0
        };
        4096
    ];
    let n = book.depth(side, &mut buf);
    buf.truncate(n);
    buf
}

#[test]
fn itch_replay_reconstructs_exchange_book() {
    let cfg = SimConfig {
        seed: 42,
        ..SimConfig::default()
    };
    let mut sim = MarketSimulator::new(cfg).unwrap();
    let events = sim.generate(300_000);
    let feed = encode_itch(&events, "SIMX").unwrap();

    let mut engine =
        StrategyEngine::new(EngineConfig::for_instrument(cfg.start_price, cfg.tick, cfg.lot).unwrap()).unwrap();
    let mut cursor = FrameCursor::new(&feed);
    let mut n = 0u64;
    let mut quotes = 0u64;
    while let Some(item) = cursor.next_event() {
        let (exch_ts, body) = item.unwrap();
        assert_eq!(body, events[n as usize].body);
        let ev = MarketEvent {
            seq: n,
            exch_ts,
            recv_tsc: 0,
            body,
        };
        match engine.on_event(&ev) {
            Ok(Some(q)) => {
                quotes += 1;
                if q.bid_active && q.ask_active {
                    assert!(q.bid_px < q.ask_px);
                }
                if let (Some(bb), Some(ba)) = (engine.book().best_bid(), engine.book().best_ask()) {
                    assert!(q.bid_px < ba.price, "bid quote crosses the market");
                    assert!(q.ask_px > bb.price, "ask quote crosses the market");
                }
            }
            Ok(None) => {}
            Err(EngineError::Book(e)) => panic!("replay rejected event {n}: {e}"),
            Err(_) => {}
        }
        n += 1;
    }
    assert_eq!(n as usize, events.len());
    assert!(quotes as f64 > 0.95 * n as f64, "quoted on {quotes}/{n} events");

    let replay = engine.book();
    let venue = sim.book();
    replay.audit().unwrap();
    assert_eq!(replay.order_count(), venue.order_count());
    for side in [Side::Bid, Side::Ask] {
        assert_eq!(replay.volume(side), venue.volume(side));
        assert_eq!(levels(replay, side), levels(venue, side));
    }
}
