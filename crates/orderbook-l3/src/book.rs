use core_types::{
    ABSENT_ASK_PX, ABSENT_BID_PX, DEPTH_LEVELS, DepthSnapshot, EventBody, Fill, OrderId, Price, Qty, Side, Timestamp,
};

use crate::arena::{NIL, OrderArena, OrderNode};
use crate::bitmap::{LevelBitmap, MAX_BITMAP_LEVELS};
use crate::error::{AuditError, BookError};
use crate::index::OrderIndex;

/// Maximum number of addressable price levels per side.
pub const MAX_LEVELS: u32 = MAX_BITMAP_LEVELS as u32;

/// Static geometry of the book. All storage is sized from this at construction.
///
/// The addressable price range is `[min_price, min_price + (levels - 1) * tick_size]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BookConfig {
    pub tick_size: Price,
    pub min_price: Price,
    pub levels: u32,
    pub max_orders: u32,
}

impl BookConfig {
    /// Ladder of `levels` ticks whose midpoint is `center` rounded down to the tick grid.
    pub fn centered(center: Price, tick_size: Price, levels: u32, max_orders: u32) -> Result<Self, BookError> {
        if !tick_size.is_positive() {
            return Err(BookError::InvalidConfig("tick_size must be positive"));
        }
        let lo_ticks = center.to_ticks(tick_size) - i64::from(levels / 2);
        let min_price = Price::from_ticks(lo_ticks.max(0), tick_size)
            .map_err(|_| BookError::InvalidConfig("min_price overflow"))?;
        let cfg = Self {
            tick_size,
            min_price,
            levels,
            max_orders,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn max_price(&self) -> Price {
        Price::from_raw(self.min_price.raw() + (i64::from(self.levels) - 1) * self.tick_size.raw())
    }

    fn validate(&self) -> Result<(), BookError> {
        if !self.tick_size.is_positive() {
            return Err(BookError::InvalidConfig("tick_size must be positive"));
        }
        if !self.min_price.is_aligned(self.tick_size) {
            return Err(BookError::InvalidConfig("min_price not tick-aligned"));
        }
        if self.levels == 0 || self.levels > MAX_LEVELS {
            return Err(BookError::InvalidConfig("levels outside [1, MAX_LEVELS]"));
        }
        if self.max_orders == 0 || self.max_orders == u32::MAX {
            return Err(BookError::InvalidConfig("max_orders outside [1, u32::MAX)"));
        }
        let span = (i64::from(self.levels) - 1).checked_mul(self.tick_size.raw());
        match span.and_then(|s| self.min_price.raw().checked_add(s)) {
            Some(_) => Ok(()),
            None => Err(BookError::InvalidConfig("price range overflows i64")),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimeInForce {
    /// Residual rests on the book.
    Gtc,
    /// Residual is discarded after matching.
    Ioc,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LimitOrder {
    pub id: OrderId,
    pub side: Side,
    pub price: Price,
    pub qty: Qty,
    pub tif: TimeInForce,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SubmitReport {
    pub filled: Qty,
    pub resting: Qty,
    pub fills: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LevelView {
    pub price: Price,
    pub qty: Qty,
    pub orders: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OrderView {
    pub id: OrderId,
    pub side: Side,
    pub price: Price,
    pub qty: Qty,
}

/// Receives every trade generated while mutating the book. Implementations run inside
/// the matching loop and must not block or allocate.
pub trait FillSink {
    fn on_fill(&mut self, fill: &Fill);
}

impl<F: FnMut(&Fill)> FillSink for F {
    #[inline(always)]
    fn on_fill(&mut self, fill: &Fill) {
        self(fill)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DiscardFills;

impl FillSink for DiscardFills {
    #[inline(always)]
    fn on_fill(&mut self, _: &Fill) {}
}

#[derive(Clone, Copy, Debug)]
struct PriceLevel {
    head: u32,
    tail: u32,
    total: i64,
    count: u32,
}

impl PriceLevel {
    const EMPTY: Self = Self {
        head: NIL,
        tail: NIL,
        total: 0,
        count: 0,
    };
}

#[derive(Debug)]
struct Ladder {
    side: Side,
    levels: Box<[PriceLevel]>,
    occupancy: LevelBitmap,
    total: i64,
}

impl Ladder {
    fn new(side: Side, levels: usize) -> Self {
        Self {
            side,
            levels: vec![PriceLevel::EMPTY; levels].into_boxed_slice(),
            occupancy: LevelBitmap::new(levels),
            total: 0,
        }
    }

    fn reset(&mut self) {
        self.levels.fill(PriceLevel::EMPTY);
        self.occupancy.reset();
        self.total = 0;
    }

    #[inline(always)]
    fn best(&self) -> Option<usize> {
        match self.side {
            Side::Bid => self.occupancy.max(),
            Side::Ask => self.occupancy.min(),
        }
    }

    #[inline(always)]
    fn next_worse(&self, i: usize) -> Option<usize> {
        match self.side {
            Side::Bid => self.occupancy.next_below(i),
            Side::Ask => self.occupancy.next_above(i),
        }
    }

    /// Detaches `slot` from its level list and updates aggregates. Does not free the slot.
    #[inline(always)]
    fn unlink(&mut self, arena: &mut OrderArena, slot: u32) {
        let node = *arena.get(slot);
        let li = node.level as usize;
        let level = &mut self.levels[li];
        if node.prev == NIL {
            level.head = node.next;
        } else {
            arena.get_mut(node.prev).next = node.next;
        }
        if node.next == NIL {
            level.tail = node.prev;
        } else {
            arena.get_mut(node.next).prev = node.prev;
        }
        level.count -= 1;
        level.total -= node.qty;
        self.total -= node.qty;
        if level.head == NIL {
            self.occupancy.clear(li);
        }
    }
}

/// Price-time priority L3 limit order book over a dense, pre-allocated price ladder.
///
/// Storage: one [`OrderArena`] slab for orders, one fixed-size open-addressing index for
/// `OrderId -> slot`, and per-side arrays of price levels holding intrusive doubly-linked
/// FIFO queues. No heap allocation occurs after [`OrderBook::new`].
///
/// Complexity: add/cancel/reduce/execute O(1) expected; matching O(1) per fill plus
/// O(1) per emptied level (bitmap scan); best bid/ask O(1).
///
/// Invariant (checked by [`OrderBook::audit`]): after every public mutation the book is
/// uncrossed, i.e. `best_bid < best_ask` whenever both sides are non-empty.
#[derive(Debug)]
pub struct OrderBook {
    cfg: BookConfig,
    arena: OrderArena,
    index: OrderIndex,
    bids: Ladder,
    asks: Ladder,
}

impl OrderBook {
    pub fn new(cfg: BookConfig) -> Result<Self, BookError> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            arena: OrderArena::new(cfg.max_orders),
            index: OrderIndex::with_max_entries(cfg.max_orders as usize),
            bids: Ladder::new(Side::Bid, cfg.levels as usize),
            asks: Ladder::new(Side::Ask, cfg.levels as usize),
        })
    }

    pub fn config(&self) -> &BookConfig {
        &self.cfg
    }

    /// Empties the book in place in O(capacity) without reallocating.
    pub fn reset(&mut self) {
        self.arena.reset();
        self.index.reset();
        self.bids.reset();
        self.asks.reset();
    }

    #[inline(always)]
    fn level_of(&self, price: Price) -> Result<usize, BookError> {
        let tick = self.cfg.tick_size.raw();
        let off = match price.raw().checked_sub(self.cfg.min_price.raw()) {
            Some(off) if off >= 0 => off,
            _ => return Err(BookError::PriceOutOfRange(price)),
        };
        let idx = off / tick;
        if idx * tick != off {
            return Err(BookError::TickMisaligned(price));
        }
        if idx >= i64::from(self.cfg.levels) {
            return Err(BookError::PriceOutOfRange(price));
        }
        Ok(idx as usize)
    }

    #[inline(always)]
    fn price_of(&self, idx: usize) -> Price {
        Price::from_raw(self.cfg.min_price.raw() + idx as i64 * self.cfg.tick_size.raw())
    }

    #[inline(always)]
    fn ladder(&self, side: Side) -> &Ladder {
        match side {
            Side::Bid => &self.bids,
            Side::Ask => &self.asks,
        }
    }

    #[inline(always)]
    fn ladder_mut(&mut self, side: Side) -> &mut Ladder {
        match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        }
    }

    /// Matches `order` against the opposite side in price-time priority, then rests the
    /// residual if `tif == Gtc`. Each trade is reported to `sink` at the maker's price.
    ///
    /// Validation happens before any state change: on `Err` the book is untouched.
    pub fn submit<S: FillSink + ?Sized>(&mut self, order: LimitOrder, sink: &mut S) -> Result<SubmitReport, BookError> {
        if order.id.is_none() {
            return Err(BookError::ReservedOrderId);
        }
        if !order.qty.is_positive() {
            return Err(BookError::InvalidQuantity(order.qty));
        }
        let idx = self.level_of(order.price)?;
        if order.tif == TimeInForce::Gtc {
            if self.index.get(order.id).is_some() {
                return Err(BookError::DuplicateOrderId(order.id));
            }
            if !self.arena.has_vacancy() {
                return Err(BookError::CapacityExhausted);
            }
            let own = self.ladder(order.side);
            if own.total.checked_add(order.qty.raw()).is_none()
                || own.levels[idx].total.checked_add(order.qty.raw()).is_none()
            {
                return Err(BookError::QuantityOverflow);
            }
        }

        let min_raw = self.cfg.min_price.raw();
        let tick_raw = self.cfg.tick_size.raw();
        let maker_side = order.side.opposite();
        let (own, opp) = match order.side {
            Side::Bid => (&mut self.bids, &mut self.asks),
            Side::Ask => (&mut self.asks, &mut self.bids),
        };

        let mut remaining = order.qty.raw();
        let mut fills = 0u32;
        while remaining > 0 {
            let Some(best) = opp.best() else { break };
            let crosses = match order.side {
                Side::Bid => best <= idx,
                Side::Ask => best >= idx,
            };
            if !crosses {
                break;
            }
            let price = Price::from_raw(min_raw + best as i64 * tick_raw);
            let level = &mut opp.levels[best];
            while remaining > 0 && level.head != NIL {
                let slot = level.head;
                let maker = self.arena.get_mut(slot);
                let traded = remaining.min(maker.qty);
                maker.qty -= traded;
                remaining -= traded;
                level.total -= traded;
                opp.total -= traded;
                fills += 1;
                sink.on_fill(&Fill {
                    maker_id: maker.id,
                    taker_id: order.id,
                    maker_side,
                    price,
                    qty: Qty::from_raw(traded),
                });
                if maker.qty == 0 {
                    let next = maker.next;
                    let maker_id = maker.id;
                    level.head = next;
                    if next == NIL {
                        level.tail = NIL;
                    } else {
                        self.arena.get_mut(next).prev = NIL;
                    }
                    level.count -= 1;
                    self.index.remove(maker_id);
                    self.arena.free(slot);
                }
            }
            if level.head == NIL {
                opp.occupancy.clear(best);
            }
        }

        let filled = order.qty.raw() - remaining;
        if remaining == 0 || order.tif == TimeInForce::Ioc {
            return Ok(SubmitReport {
                filled: Qty::from_raw(filled),
                resting: Qty::ZERO,
                fills,
            });
        }

        let level = &mut own.levels[idx];
        let Some(slot) = self.arena.alloc(OrderNode {
            id: order.id,
            qty: remaining,
            level: idx as u32,
            prev: level.tail,
            next: NIL,
            side: order.side,
        }) else {
            return Err(BookError::CapacityExhausted);
        };
        if level.tail == NIL {
            level.head = slot;
        } else {
            self.arena.get_mut(level.tail).next = slot;
        }
        level.tail = slot;
        level.count += 1;
        level.total += remaining;
        own.total += remaining;
        own.occupancy.set(idx);
        self.index.insert(order.id, slot);

        Ok(SubmitReport {
            filled: Qty::from_raw(filled),
            resting: Qty::from_raw(remaining),
            fills,
        })
    }

    /// Removes a resting order entirely. Returns its state immediately before removal.
    pub fn cancel(&mut self, id: OrderId) -> Result<OrderView, BookError> {
        let slot = self.index.get(id).ok_or(BookError::UnknownOrder(id))?;
        let node = *self.arena.get(slot);
        let price = self.price_of(node.level as usize);
        let ladder = match node.side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        ladder.unlink(&mut self.arena, slot);
        self.index.remove(id);
        self.arena.free(slot);
        Ok(OrderView {
            id,
            side: node.side,
            price,
            qty: Qty::from_raw(node.qty),
        })
    }

    /// Partial cancel (ITCH `X`). Keeps time priority. Reducing by at least the open
    /// quantity removes the order. Returns the remaining open quantity.
    pub fn reduce(&mut self, id: OrderId, qty: Qty) -> Result<Qty, BookError> {
        if !qty.is_positive() {
            return Err(BookError::InvalidQuantity(qty));
        }
        let slot = self.index.get(id).ok_or(BookError::UnknownOrder(id))?;
        let node = *self.arena.get(slot);
        if qty.raw() >= node.qty {
            self.cancel(id)?;
            return Ok(Qty::ZERO);
        }
        let delta = qty.raw();
        self.arena.get_mut(slot).qty -= delta;
        let ladder = self.ladder_mut(node.side);
        ladder.levels[node.level as usize].total -= delta;
        ladder.total -= delta;
        Ok(Qty::from_raw(node.qty - delta))
    }

    /// Execution of a resting order reported by the venue (ITCH `E`). The aggressor is
    /// unknown, so the fill carries `taker_id = OrderId::NONE`.
    pub fn execute<S: FillSink + ?Sized>(&mut self, id: OrderId, qty: Qty, sink: &mut S) -> Result<Fill, BookError> {
        if !qty.is_positive() {
            return Err(BookError::InvalidQuantity(qty));
        }
        let slot = self.index.get(id).ok_or(BookError::UnknownOrder(id))?;
        let node = *self.arena.get(slot);
        if qty.raw() > node.qty {
            return Err(BookError::ExecutionExceedsOpen {
                order_id: id,
                open: Qty::from_raw(node.qty),
                requested: qty,
            });
        }
        let fill = Fill {
            maker_id: id,
            taker_id: OrderId::NONE,
            maker_side: node.side,
            price: self.price_of(node.level as usize),
            qty,
        };
        if qty.raw() == node.qty {
            let ladder = match node.side {
                Side::Bid => &mut self.bids,
                Side::Ask => &mut self.asks,
            };
            ladder.unlink(&mut self.arena, slot);
            self.index.remove(id);
            self.arena.free(slot);
        } else {
            self.arena.get_mut(slot).qty -= qty.raw();
            let ladder = self.ladder_mut(node.side);
            ladder.levels[node.level as usize].total -= qty.raw();
            ladder.total -= qty.raw();
        }
        sink.on_fill(&fill);
        Ok(fill)
    }

    /// Cancel/replace (ITCH `U`): removes `id` and submits `new_id` on the same side as a
    /// GTC order, losing time priority. All validation precedes the cancel.
    pub fn replace<S: FillSink + ?Sized>(
        &mut self,
        id: OrderId,
        new_id: OrderId,
        price: Price,
        qty: Qty,
        sink: &mut S,
    ) -> Result<SubmitReport, BookError> {
        if new_id.is_none() {
            return Err(BookError::ReservedOrderId);
        }
        if !qty.is_positive() {
            return Err(BookError::InvalidQuantity(qty));
        }
        let slot = self.index.get(id).ok_or(BookError::UnknownOrder(id))?;
        let side = self.arena.get(slot).side;
        let idx = self.level_of(price)?;
        if new_id != id && self.index.get(new_id).is_some() {
            return Err(BookError::DuplicateOrderId(new_id));
        }
        let own = self.ladder(side);
        if own.total.checked_add(qty.raw()).is_none() || own.levels[idx].total.checked_add(qty.raw()).is_none() {
            return Err(BookError::QuantityOverflow);
        }
        self.cancel(id)?;
        self.submit(
            LimitOrder {
                id: new_id,
                side,
                price,
                qty,
                tif: TimeInForce::Gtc,
            },
            sink,
        )
    }

    /// Applies a normalised feed event. `Add` goes through the matching path so a crossed
    /// feed can never leave the book crossed. `Trade` only notifies `sink`.
    #[inline]
    pub fn apply<S: FillSink + ?Sized>(&mut self, body: &EventBody, sink: &mut S) -> Result<(), BookError> {
        match *body {
            EventBody::Add {
                order_id,
                side,
                price,
                qty,
            } => {
                self.submit(
                    LimitOrder {
                        id: order_id,
                        side,
                        price,
                        qty,
                        tif: TimeInForce::Gtc,
                    },
                    sink,
                )?;
            }
            EventBody::Execute { order_id, qty, .. } => {
                self.execute(order_id, qty, sink)?;
            }
            EventBody::Cancel { order_id, qty } => {
                self.reduce(order_id, qty)?;
            }
            EventBody::Delete { order_id } => {
                self.cancel(order_id)?;
            }
            EventBody::Replace {
                order_id,
                new_order_id,
                price,
                qty,
            } => {
                self.replace(order_id, new_order_id, price, qty, sink)?;
            }
            EventBody::Trade { side, price, qty, .. } => {
                sink.on_fill(&Fill {
                    maker_id: OrderId::NONE,
                    taker_id: OrderId::NONE,
                    maker_side: side.opposite(),
                    price,
                    qty,
                });
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn level_view(&self, ladder: &Ladder, idx: usize) -> LevelView {
        let l = &ladder.levels[idx];
        LevelView {
            price: self.price_of(idx),
            qty: Qty::from_raw(l.total),
            orders: l.count,
        }
    }

    #[inline(always)]
    pub fn best_bid(&self) -> Option<LevelView> {
        self.bids.best().map(|i| self.level_view(&self.bids, i))
    }

    #[inline(always)]
    pub fn best_ask(&self) -> Option<LevelView> {
        self.asks.best().map(|i| self.level_view(&self.asks, i))
    }

    /// Writes up to `out.len()` levels of `side`, best first. Returns the count written.
    pub fn depth(&self, side: Side, out: &mut [LevelView]) -> usize {
        let ladder = self.ladder(side);
        let mut cur = ladder.best();
        let mut n = 0;
        while let Some(i) = cur
            && n < out.len()
        {
            out[n] = self.level_view(ladder, i);
            n += 1;
            cur = ladder.next_worse(i);
        }
        n
    }

    /// Fills `out` with the top [`DEPTH_LEVELS`] levels per side. O(`DEPTH_LEVELS`).
    #[inline]
    pub fn snapshot(&self, ts: Timestamp, out: &mut DepthSnapshot) {
        out.ts = ts;
        out.bid_levels = Self::fill_side(self, &self.bids, &mut out.bid_px, &mut out.bid_qty, ABSENT_BID_PX);
        out.ask_levels = Self::fill_side(self, &self.asks, &mut out.ask_px, &mut out.ask_qty, ABSENT_ASK_PX);
    }

    #[inline(always)]
    fn fill_side(
        &self,
        ladder: &Ladder,
        px: &mut [i64; DEPTH_LEVELS],
        qty: &mut [i64; DEPTH_LEVELS],
        absent: i64,
    ) -> u8 {
        let mut cur = ladder.best();
        let mut n = 0;
        while let Some(i) = cur
            && n < DEPTH_LEVELS
        {
            px[n] = self.price_of(i).raw();
            qty[n] = ladder.levels[i].total;
            n += 1;
            cur = ladder.next_worse(i);
        }
        for k in n..DEPTH_LEVELS {
            px[k] = absent;
            qty[k] = 0;
        }
        n as u8
    }

    pub fn order(&self, id: OrderId) -> Option<OrderView> {
        let node = self.arena.get(self.index.get(id)?);
        Some(OrderView {
            id,
            side: node.side,
            price: self.price_of(node.level as usize),
            qty: Qty::from_raw(node.qty),
        })
    }

    /// Visits the orders resting at `price` on `side` in time priority.
    pub fn for_each_order_at(&self, side: Side, price: Price, mut f: impl FnMut(OrderView)) -> Result<(), BookError> {
        let idx = self.level_of(price)?;
        let mut cur = self.ladder(side).levels[idx].head;
        while cur != NIL {
            let node = self.arena.get(cur);
            f(OrderView {
                id: node.id,
                side,
                price,
                qty: Qty::from_raw(node.qty),
            });
            cur = node.next;
        }
        Ok(())
    }

    /// Displayed quantity at `price` on `side`; zero for an empty level, `None` if `price`
    /// is outside the ladder or off-tick. O(1).
    #[inline(always)]
    pub fn level_volume(&self, side: Side, price: Price) -> Option<Qty> {
        let idx = self.level_of(price).ok()?;
        Some(Qty::from_raw(self.ladder(side).levels[idx].total))
    }

    #[inline(always)]
    pub fn volume(&self, side: Side) -> Qty {
        Qty::from_raw(self.ladder(side).total)
    }

    #[inline(always)]
    pub fn order_count(&self) -> usize {
        self.arena.live() as usize
    }

    pub fn capacity(&self) -> usize {
        self.arena.capacity()
    }

    /// Exhaustive structural check. O(levels + orders); intended for tests and
    /// out-of-band health checks, never the hot path.
    pub fn audit(&self) -> Result<(), AuditError> {
        let mut orders = 0usize;
        for ladder in [&self.bids, &self.asks] {
            let mut side_total = 0i64;
            for (idx, level) in ladder.levels.iter().enumerate() {
                if (level.head != NIL) != ladder.occupancy.contains(idx) {
                    return Err(AuditError("occupancy bitmap out of sync with level"));
                }
                let mut cur = level.head;
                let mut prev = NIL;
                let mut count = 0u32;
                let mut total = 0i64;
                while cur != NIL {
                    let node = self.arena.get(cur);
                    if node.prev != prev {
                        return Err(AuditError("broken prev link"));
                    }
                    if node.level as usize != idx || node.side != ladder.side {
                        return Err(AuditError("order filed under wrong level or side"));
                    }
                    if node.qty <= 0 {
                        return Err(AuditError("non-positive resting quantity"));
                    }
                    if self.index.get(node.id) != Some(cur) {
                        return Err(AuditError("index does not map order to its slot"));
                    }
                    count += 1;
                    total += node.qty;
                    prev = cur;
                    cur = node.next;
                }
                if level.tail != prev {
                    return Err(AuditError("level tail does not match last order"));
                }
                if level.count != count || level.total != total {
                    return Err(AuditError("level aggregates disagree with orders"));
                }
                side_total += total;
                orders += count as usize;
            }
            if side_total != ladder.total {
                return Err(AuditError("side volume disagrees with levels"));
            }
        }
        if orders != self.arena.live() as usize {
            return Err(AuditError("arena live count disagrees with resting orders"));
        }
        if let (Some(b), Some(a)) = (self.bids.best(), self.asks.best())
            && b >= a
        {
            return Err(AuditError("book crossed or locked"));
        }
        Ok(())
    }
}
