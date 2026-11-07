#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, VecDeque};

use core_types::{Fill, OrderId, Price, Qty, Side};
use orderbook_l3::{BookConfig, BookError, LevelView, LimitOrder, OrderBook, TimeInForce};
use proptest::prelude::*;

const TICK_RAW: i64 = 1_000_000;
const BASE_TICKS: i64 = 10_000;

#[derive(Clone, Debug)]
enum Op {
    Submit { bid: bool, tick: i64, qty: i64, ioc: bool },
    Cancel { pick: usize },
    Reduce { pick: usize, qty: i64 },
    Execute { pick: usize, qty: i64 },
    Replace { pick: usize, tick: i64, qty: i64 },
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        6 => (any::<bool>(), -20i64..20, 1i64..40, prop::bool::weighted(0.2))
            .prop_map(|(bid, tick, qty, ioc)| Op::Submit { bid, tick, qty, ioc }),
        2 => any::<usize>().prop_map(|pick| Op::Cancel { pick }),
        1 => (any::<usize>(), 1i64..30).prop_map(|(pick, qty)| Op::Reduce { pick, qty }),
        1 => (any::<usize>(), 1i64..30).prop_map(|(pick, qty)| Op::Execute { pick, qty }),
        1 => (any::<usize>(), -20i64..20, 1i64..40).prop_map(|(pick, tick, qty)| Op::Replace { pick, tick, qty }),
    ]
}

fn price(tick: i64) -> Price {
    Price::from_raw((BASE_TICKS + tick) * TICK_RAW)
}

fn q(units: i64) -> Qty {
    Qty::from_raw(units)
}

/// Naive reference: BTreeMap of FIFO queues. Obviously correct, O(n) everywhere.
#[derive(Default)]
struct ReferenceBook {
    bids: BTreeMap<i64, VecDeque<(u64, i64)>>,
    asks: BTreeMap<i64, VecDeque<(u64, i64)>>,
}

impl ReferenceBook {
    fn side(&mut self, side: Side) -> &mut BTreeMap<i64, VecDeque<(u64, i64)>> {
        match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        }
    }

    fn submit(&mut self, id: u64, side: Side, px: i64, mut qty: i64, ioc: bool) -> Vec<(u64, i64, i64)> {
        let mut fills = Vec::new();
        loop {
            if qty == 0 {
                break;
            }
            let opp = self.side(side.opposite());
            let best = match side {
                Side::Bid => opp.keys().next().copied().filter(|&a| a <= px),
                Side::Ask => opp.keys().next_back().copied().filter(|&b| b >= px),
            };
            let Some(level_px) = best else { break };
            let queue = opp.get_mut(&level_px).unwrap();
            while qty > 0 {
                let Some(front) = queue.front_mut() else { break };
                let t = qty.min(front.1);
                front.1 -= t;
                qty -= t;
                fills.push((front.0, level_px, t));
                if front.1 == 0 {
                    queue.pop_front();
                }
            }
            if queue.is_empty() {
                opp.remove(&level_px);
            }
        }
        if qty > 0 && !ioc {
            self.side(side).entry(px).or_default().push_back((id, qty));
        }
        fills
    }

    fn find(&self, id: u64) -> Option<(Side, i64, i64)> {
        for (side, map) in [(Side::Bid, &self.bids), (Side::Ask, &self.asks)] {
            for (&px, queue) in map {
                if let Some(&(_, qty)) = queue.iter().find(|o| o.0 == id) {
                    return Some((side, px, qty));
                }
            }
        }
        None
    }

    fn remove_qty(&mut self, id: u64, amount: i64) {
        let Some((side, px, _)) = self.find(id) else { return };
        let map = self.side(side);
        let queue = map.get_mut(&px).unwrap();
        let pos = queue.iter().position(|o| o.0 == id).unwrap();
        queue[pos].1 -= amount;
        if queue[pos].1 <= 0 {
            queue.remove(pos);
        }
        if queue.is_empty() {
            map.remove(&px);
        }
    }

    fn live_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .bids
            .values()
            .chain(self.asks.values())
            .flatten()
            .map(|o| o.0)
            .collect();
        ids.sort_unstable();
        ids
    }

    fn levels(&self, side: Side) -> Vec<(i64, i64, u32)> {
        let summarize = |(&px, qu): (&i64, &VecDeque<(u64, i64)>)| (px, qu.iter().map(|o| o.1).sum(), qu.len() as u32);
        match side {
            Side::Bid => self.bids.iter().rev().map(summarize).collect(),
            Side::Ask => self.asks.iter().map(summarize).collect(),
        }
    }
}

#[derive(Default)]
struct Ledger {
    submitted: i64,
    matched: i64,
    executed: i64,
    cancelled: i64,
    discarded: i64,
}

fn check_invariants(book: &OrderBook, model: &ReferenceBook, ledger: &Ledger) -> Result<(), TestCaseError> {
    prop_assert!(book.audit().is_ok(), "audit: {:?}", book.audit());

    if let (Some(b), Some(a)) = (book.best_bid(), book.best_ask()) {
        prop_assert!(b.price < a.price, "crossed: bid {} ask {}", b.price, a.price);
    }

    let resting = book.volume(Side::Bid).raw() + book.volume(Side::Ask).raw();
    prop_assert_eq!(
        ledger.submitted,
        2 * ledger.matched + ledger.executed + ledger.cancelled + ledger.discarded + resting,
        "volume not conserved"
    );

    let mut buf = [LevelView {
        price: Price::ZERO,
        qty: Qty::ZERO,
        orders: 0,
    }; 64];
    for side in [Side::Bid, Side::Ask] {
        let n = book.depth(side, &mut buf);
        let got: Vec<(i64, i64, u32)> = buf[..n]
            .iter()
            .map(|l| (l.price.raw(), l.qty.raw(), l.orders))
            .collect();
        prop_assert_eq!(got, model.levels(side));
    }
    prop_assert_eq!(book.order_count(), model.live_ids().len());
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    #[test]
    fn book_matches_reference_and_preserves_invariants(ops in prop::collection::vec(op(), 1..250)) {
        let cfg = BookConfig {
            tick_size: Price::from_raw(TICK_RAW),
            min_price: price(-2_000),
            levels: 4_096,
            max_orders: 4_096,
        };
        let mut book = OrderBook::new(cfg).unwrap();
        let mut model = ReferenceBook::default();
        let mut ledger = Ledger::default();
        let mut next_id = 1u64;

        for op in ops {
            match op {
                Op::Submit { bid, tick, qty, ioc } => {
                    let side = if bid { Side::Bid } else { Side::Ask };
                    let id = next_id;
                    next_id += 1;
                    let order = LimitOrder {
                        id: OrderId(id),
                        side,
                        price: price(tick),
                        qty: q(qty),
                        tif: if ioc { TimeInForce::Ioc } else { TimeInForce::Gtc },
                    };
                    let mut fills: Vec<Fill> = Vec::new();
                    let mut sink = |f: &Fill| fills.push(*f);
                    let report = book.submit(order, &mut sink).unwrap();
                    let expected = model.submit(id, side, price(tick).raw(), qty, ioc);

                    let got: Vec<(u64, i64, i64)> = fills.iter().map(|f| (f.maker_id.0, f.price.raw(), f.qty.raw())).collect();
                    prop_assert_eq!(&got, &expected, "fill sequence diverged from price-time reference");
                    for f in &fills {
                        prop_assert_eq!(f.aggressor(), side);
                        prop_assert!(match side {
                            Side::Bid => f.price <= order.price,
                            Side::Ask => f.price >= order.price,
                        }, "fill through taker limit");
                    }
                    prop_assert_eq!(report.filled.raw(), fills.iter().map(|f| f.qty.raw()).sum::<i64>());
                    prop_assert_eq!(report.fills as usize, fills.len());

                    ledger.submitted += qty;
                    ledger.matched += report.filled.raw();
                    ledger.discarded += qty - report.filled.raw() - report.resting.raw();
                }
                Op::Cancel { pick } => {
                    let ids = model.live_ids();
                    if ids.is_empty() { continue; }
                    let id = ids[pick % ids.len()];
                    let view = book.cancel(OrderId(id)).unwrap();
                    let (_, _, open) = model.find(id).unwrap();
                    prop_assert_eq!(view.qty.raw(), open);
                    model.remove_qty(id, open);
                    ledger.cancelled += open;
                    prop_assert_eq!(book.cancel(OrderId(id)), Err(BookError::UnknownOrder(OrderId(id))));
                }
                Op::Reduce { pick, qty } => {
                    let ids = model.live_ids();
                    if ids.is_empty() { continue; }
                    let id = ids[pick % ids.len()];
                    let (_, _, open) = model.find(id).unwrap();
                    let left = book.reduce(OrderId(id), q(qty)).unwrap();
                    let removed = qty.min(open);
                    prop_assert_eq!(left.raw(), open - removed);
                    model.remove_qty(id, removed);
                    ledger.cancelled += removed;
                }
                Op::Execute { pick, qty } => {
                    let ids = model.live_ids();
                    if ids.is_empty() { continue; }
                    let id = ids[pick % ids.len()];
                    let (side, px, open) = model.find(id).unwrap();
                    let res = book.execute(OrderId(id), q(qty), &mut |_: &Fill| {});
                    if qty > open {
                        let is_exceeds = matches!(res, Err(BookError::ExecutionExceedsOpen { .. }));
                        prop_assert!(is_exceeds);
                    } else {
                        let fill = res.unwrap();
                        prop_assert_eq!((fill.maker_side, fill.price.raw(), fill.qty.raw()), (side, px, qty));
                        model.remove_qty(id, qty);
                        ledger.executed += qty;
                    }
                }
                Op::Replace { pick, tick, qty } => {
                    let ids = model.live_ids();
                    if ids.is_empty() { continue; }
                    let id = ids[pick % ids.len()];
                    let (side, _, open) = model.find(id).unwrap();
                    let new_id = next_id;
                    next_id += 1;
                    let mut fills: Vec<Fill> = Vec::new();
                    let mut sink = |f: &Fill| fills.push(*f);
                    let report = book.replace(OrderId(id), OrderId(new_id), price(tick), q(qty), &mut sink).unwrap();
                    model.remove_qty(id, open);
                    let expected = model.submit(new_id, side, price(tick).raw(), qty, false);
                    let got: Vec<(u64, i64, i64)> = fills.iter().map(|f| (f.maker_id.0, f.price.raw(), f.qty.raw())).collect();
                    prop_assert_eq!(got, expected);
                    ledger.cancelled += open;
                    ledger.submitted += qty;
                    ledger.matched += report.filled.raw();
                }
            }
            check_invariants(&book, &model, &ledger)?;
        }
    }

    #[test]
    fn aggressive_sweep_never_leaves_book_crossed(
        levels in prop::collection::vec((1i64..10, 1i64..20), 1..30),
        sweep_tick in -15i64..15,
        sweep_qty in 1i64..400,
        bid in any::<bool>(),
    ) {
        let cfg = BookConfig::centered(price(0), Price::from_raw(TICK_RAW), 1_024, 1_024).unwrap();
        let mut book = OrderBook::new(cfg).unwrap();
        let mut id = 1u64;
        for (offset, qty) in levels {
            let (side, tick) = if bid { (Side::Ask, offset) } else { (Side::Bid, -offset) };
            book.submit(
                LimitOrder { id: OrderId(id), side, price: price(tick), qty: q(qty), tif: TimeInForce::Gtc },
                &mut |_: &Fill| {},
            ).unwrap();
            id += 1;
        }
        let side = if bid { Side::Bid } else { Side::Ask };
        let before = book.volume(side.opposite()).raw();
        let report = book.submit(
            LimitOrder { id: OrderId(id), side, price: price(sweep_tick), qty: q(sweep_qty), tif: TimeInForce::Gtc },
            &mut |_: &Fill| {},
        ).unwrap();
        prop_assert_eq!(before - report.filled.raw(), book.volume(side.opposite()).raw());
        prop_assert_eq!(report.filled.raw() + report.resting.raw(), sweep_qty);
        if let (Some(b), Some(a)) = (book.best_bid(), book.best_ask()) {
            prop_assert!(b.price < a.price);
        }
        prop_assert!(book.audit().is_ok());
    }
}
