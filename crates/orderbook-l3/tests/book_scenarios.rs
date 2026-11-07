#![allow(clippy::unwrap_used, clippy::expect_used)]

use core_types::{DEPTH_LEVELS, DepthSnapshot, EventBody, Fill, OrderId, Price, Qty, Side, Timestamp};
use orderbook_l3::{BookConfig, BookError, DiscardFills, LimitOrder, OrderBook, TimeInForce};

fn px(v: f64) -> Price {
    Price::from_f64(v).unwrap()
}

fn qty(v: i64) -> Qty {
    Qty::from_units(v).unwrap()
}

fn book() -> OrderBook {
    OrderBook::new(BookConfig::centered(px(100.0), px(0.01), 4096, 1024).unwrap()).unwrap()
}

fn gtc(id: u64, side: Side, price: f64, q: i64) -> LimitOrder {
    LimitOrder {
        id: OrderId(id),
        side,
        price: px(price),
        qty: qty(q),
        tif: TimeInForce::Gtc,
    }
}

#[test]
fn price_time_priority_within_level() {
    let mut b = book();
    b.submit(gtc(1, Side::Ask, 100.01, 5), &mut DiscardFills).unwrap();
    b.submit(gtc(2, Side::Ask, 100.01, 5), &mut DiscardFills).unwrap();
    b.submit(gtc(3, Side::Ask, 100.00, 5), &mut DiscardFills).unwrap();

    let mut fills: Vec<Fill> = Vec::new();
    let mut sink = |f: &Fill| fills.push(*f);
    let r = b.submit(gtc(10, Side::Bid, 100.01, 12), &mut sink).unwrap();

    assert_eq!(r.filled, qty(12));
    assert_eq!(r.resting, Qty::ZERO);
    let makers: Vec<u64> = fills.iter().map(|f| f.maker_id.0).collect();
    assert_eq!(makers, vec![3, 1, 2]);
    assert_eq!(fills[0].price, px(100.00));
    assert_eq!(fills[1].price, px(100.01));
    assert_eq!(fills[2].qty, qty(2));
    assert!(
        fills
            .iter()
            .all(|f| f.aggressor() == Side::Bid && f.taker_id == OrderId(10))
    );
    assert_eq!(b.order(OrderId(2)).unwrap().qty, qty(3));
    b.audit().unwrap();
}

#[test]
fn residual_rests_and_ioc_discards() {
    let mut b = book();
    b.submit(gtc(1, Side::Ask, 100.02, 4), &mut DiscardFills).unwrap();

    let r = b.submit(gtc(2, Side::Bid, 100.05, 10), &mut DiscardFills).unwrap();
    assert_eq!((r.filled, r.resting), (qty(4), qty(6)));
    assert_eq!(b.best_bid().unwrap().price, px(100.05));
    assert!(b.best_ask().is_none());

    let ioc = LimitOrder {
        tif: TimeInForce::Ioc,
        ..gtc(3, Side::Ask, 100.00, 9)
    };
    let r = b.submit(ioc, &mut DiscardFills).unwrap();
    assert_eq!((r.filled, r.resting), (qty(6), Qty::ZERO));
    assert_eq!(b.order_count(), 0);
    assert!(b.best_bid().is_none());
    b.audit().unwrap();
}

#[test]
fn reduce_keeps_priority_execute_reports_maker() {
    let mut b = book();
    b.submit(gtc(1, Side::Bid, 99.99, 10), &mut DiscardFills).unwrap();
    b.submit(gtc(2, Side::Bid, 99.99, 10), &mut DiscardFills).unwrap();
    assert_eq!(b.reduce(OrderId(1), qty(7)).unwrap(), qty(3));

    let mut order = Vec::new();
    b.for_each_order_at(Side::Bid, px(99.99), |o| order.push(o.id.0))
        .unwrap();
    assert_eq!(order, vec![1, 2]);

    let fill = b.execute(OrderId(1), qty(3), &mut DiscardFills).unwrap();
    assert_eq!(fill.aggressor(), Side::Ask);
    assert_eq!(fill.taker_id, OrderId::NONE);
    assert!(b.order(OrderId(1)).is_none());
    assert_eq!(b.best_bid().unwrap().qty, qty(10));

    assert!(matches!(
        b.execute(OrderId(2), qty(11), &mut DiscardFills),
        Err(BookError::ExecutionExceedsOpen { .. })
    ));
    assert_eq!(b.reduce(OrderId(2), qty(50)).unwrap(), Qty::ZERO);
    assert_eq!(b.order_count(), 0);
    b.audit().unwrap();
}

#[test]
fn replace_loses_priority() {
    let mut b = book();
    b.submit(gtc(1, Side::Ask, 100.10, 5), &mut DiscardFills).unwrap();
    b.submit(gtc(2, Side::Ask, 100.10, 5), &mut DiscardFills).unwrap();
    b.replace(OrderId(1), OrderId(3), px(100.10), qty(6), &mut DiscardFills)
        .unwrap();

    let mut order = Vec::new();
    b.for_each_order_at(Side::Ask, px(100.10), |o| order.push((o.id.0, o.qty)))
        .unwrap();
    assert_eq!(order, vec![(2, qty(5)), (3, qty(6))]);
    assert_eq!(b.volume(Side::Ask), qty(11));
    b.audit().unwrap();
}

#[test]
fn rejects_invalid_input_without_mutation() {
    let mut b = book();
    b.submit(gtc(1, Side::Bid, 100.00, 1), &mut DiscardFills).unwrap();

    assert_eq!(
        b.submit(gtc(1, Side::Bid, 100.00, 1), &mut DiscardFills),
        Err(BookError::DuplicateOrderId(OrderId(1)))
    );
    assert_eq!(
        b.submit(gtc(2, Side::Bid, 100.005, 1), &mut DiscardFills),
        Err(BookError::TickMisaligned(px(100.005)))
    );
    assert!(matches!(
        b.submit(gtc(2, Side::Bid, 500.0, 1), &mut DiscardFills),
        Err(BookError::PriceOutOfRange(_))
    ));
    assert!(matches!(
        b.submit(gtc(2, Side::Bid, 100.0, 0), &mut DiscardFills),
        Err(BookError::InvalidQuantity(_))
    ));
    assert_eq!(
        b.submit(
            LimitOrder {
                id: OrderId::NONE,
                ..gtc(0, Side::Bid, 100.0, 1)
            },
            &mut DiscardFills
        ),
        Err(BookError::ReservedOrderId)
    );
    assert_eq!(b.cancel(OrderId(99)), Err(BookError::UnknownOrder(OrderId(99))));
    assert_eq!(
        b.replace(OrderId(1), OrderId(1), px(900.0), qty(1), &mut DiscardFills)
            .map(|_| ()),
        Err(BookError::PriceOutOfRange(px(900.0)))
    );
    assert_eq!(b.order_count(), 1);
    b.audit().unwrap();
}

#[test]
fn capacity_is_enforced_before_matching() {
    let cfg = BookConfig::centered(px(100.0), px(0.01), 64, 2).unwrap();
    let mut b = OrderBook::new(cfg).unwrap();
    b.submit(gtc(1, Side::Bid, 99.99, 1), &mut DiscardFills).unwrap();
    b.submit(gtc(2, Side::Bid, 99.98, 1), &mut DiscardFills).unwrap();
    assert_eq!(
        b.submit(gtc(3, Side::Ask, 99.99, 5), &mut DiscardFills),
        Err(BookError::CapacityExhausted)
    );
    assert_eq!(b.order_count(), 2);
    let ioc = LimitOrder {
        tif: TimeInForce::Ioc,
        ..gtc(3, Side::Ask, 99.99, 5)
    };
    assert_eq!(b.submit(ioc, &mut DiscardFills).unwrap().filled, qty(1));
    b.audit().unwrap();
}

#[test]
fn snapshot_orders_levels_and_fills_sentinels() {
    let mut b = book();
    for i in 0..3 {
        b.submit(
            gtc(i, Side::Bid, 99.99 - i as f64 * 0.01, 1 + i as i64),
            &mut DiscardFills,
        )
        .unwrap();
    }
    for i in 0..10 {
        b.submit(gtc(100 + i, Side::Ask, 100.01 + i as f64 * 0.02, 1), &mut DiscardFills)
            .unwrap();
    }
    let mut snap = DepthSnapshot::EMPTY;
    b.snapshot(Timestamp(42), &mut snap);
    assert_eq!(snap.ts, Timestamp(42));
    assert_eq!((snap.bid_levels, snap.ask_levels), (3, DEPTH_LEVELS as u8));
    assert_eq!(snap.bid_px[..3], [px(99.99).raw(), px(99.98).raw(), px(99.97).raw()]);
    assert_eq!(snap.bid_qty[..4], [qty(1).raw(), qty(2).raw(), qty(3).raw(), 0]);
    assert_eq!(snap.bid_px[3], core_types::ABSENT_BID_PX);
    assert!(snap.ask_px.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(snap.ask_px[7], px(100.15).raw());
}

#[test]
fn trade_event_notifies_without_mutation() {
    let mut b = book();
    let mut seen = Vec::new();
    let mut sink = |f: &Fill| seen.push(*f);
    b.apply(
        &EventBody::Trade {
            side: Side::Ask,
            price: px(100.0),
            qty: qty(3),
            match_id: 7,
        },
        &mut sink,
    )
    .unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].aggressor(), Side::Ask);
    assert_eq!(b.order_count(), 0);
}

#[test]
fn reset_restores_empty_state() {
    let mut b = book();
    for i in 0..100 {
        b.submit(gtc(i, Side::Bid, 99.0 + (i % 50) as f64 * 0.01, 1), &mut DiscardFills)
            .unwrap();
    }
    b.reset();
    assert_eq!(b.order_count(), 0);
    assert!(b.best_bid().is_none());
    b.audit().unwrap();
    b.submit(gtc(1, Side::Ask, 100.0, 1), &mut DiscardFills).unwrap();
    b.audit().unwrap();
}
