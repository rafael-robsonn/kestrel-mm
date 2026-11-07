//! Agent-based order-level market simulator. Runs a real matching engine
//! ([`OrderBook`]) and emits the ITCH-equivalent event stream an exchange would publish.
//!
//! Agents: passive liquidity providers anchored between the displayed mid and the latent
//! fair value, size-dependent cancellations, noise takers, informed takers whose intensity
//! grows with `|fair - mid|`, cancel/replace, and non-displayed prints. Offline tool: it
//! allocates freely and is never on the hot path.

use core_types::{EventBody, Fill, MarketEvent, OrderId, Price, Qty, Side, Timestamp};
use orderbook_l3::{BookConfig, LimitOrder, OrderBook, TimeInForce};

use crate::error::EngineError;
use crate::itch::{ItchEncoder, ItchError};

/// xoshiro256** seeded through SplitMix64.
#[derive(Clone, Debug)]
pub struct SimRng {
    s: [u64; 4],
    spare_normal: Option<f64>,
}

impl SimRng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
            spare_normal: None,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform on the open interval `(0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * core::f64::consts::PI * u2;
        self.spare_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    pub fn exponential(&mut self, mean: f64) -> f64 {
        -mean * self.uniform().ln()
    }

    /// Failures before the first success, success probability `p`.
    pub fn geometric(&mut self, p: f64) -> u64 {
        (self.uniform().ln() / (1.0 - p).ln()).floor() as u64
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.uniform() < p
    }

    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() >> 32) as usize * n) >> 32
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimConfig {
    pub seed: u64,
    pub tick: Price,
    pub start_price: Price,
    pub lot: Qty,
    pub start_ts: Timestamp,
    /// Mean agent actions per second of exchange time.
    pub action_rate_hz: f64,
    /// Fair-value diffusion, ticks/√s.
    pub fair_sigma: f64,
    pub jump_rate_hz: f64,
    pub jump_ticks: f64,
    /// Resting-order count around which cancellation pressure balances additions.
    pub target_orders: usize,
    /// Fraction of the fair-mid gap the passive providers lean towards.
    pub provider_lean: f64,
    pub informed_weight: f64,
    pub noise_taker_weight: f64,
    pub hidden_weight: f64,
    pub replace_weight: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            seed: 0x5EED,
            tick: Price::from_raw(1_000_000),
            start_price: Price::from_raw(100 * core_types::SCALE),
            lot: Qty::from_raw(100 * core_types::SCALE),
            start_ts: Timestamp(34_200_000_000_000),
            action_rate_hz: 20_000.0,
            fair_sigma: 6.0,
            jump_rate_hz: 0.2,
            jump_ticks: 4.0,
            target_orders: 600,
            provider_lean: 0.35,
            informed_weight: 0.10,
            noise_taker_weight: 0.06,
            hidden_weight: 0.01,
            replace_weight: 0.04,
        }
    }
}

/// Exchange-side simulator. `book()` is the venue's book; an engine replaying the emitted
/// events must reproduce it exactly.
#[derive(Debug)]
pub struct MarketSimulator {
    cfg: SimConfig,
    rng: SimRng,
    book: OrderBook,
    live: Vec<OrderId>,
    fills: Vec<Fill>,
    next_id: u64,
    next_match: u64,
    seq: u64,
    ts: u64,
    fair: f64,
    tick_raw: i64,
    lot_raw: i64,
}

impl MarketSimulator {
    pub fn new(cfg: SimConfig) -> Result<Self, EngineError> {
        let book_cfg = BookConfig::centered(cfg.start_price, cfg.tick, 1 << 16, 1 << 17)?;
        Ok(Self {
            rng: SimRng::new(cfg.seed),
            book: OrderBook::new(book_cfg)?,
            live: Vec::with_capacity(4 * cfg.target_orders),
            fills: Vec::with_capacity(64),
            next_id: 1,
            next_match: 1,
            seq: 0,
            ts: cfg.start_ts.nanos(),
            fair: cfg.start_price.to_ticks(cfg.tick) as f64,
            tick_raw: cfg.tick.raw(),
            lot_raw: cfg.lot.raw(),
            cfg,
        })
    }

    pub fn book(&self) -> &OrderBook {
        &self.book
    }

    pub fn fair_value_ticks(&self) -> f64 {
        self.fair
    }

    fn px(&self, ticks: i64) -> Price {
        Price::from_raw(ticks * self.tick_raw)
    }

    fn emit(&mut self, out: &mut Vec<MarketEvent>, body: EventBody) {
        out.push(MarketEvent {
            seq: self.seq,
            exch_ts: Timestamp(self.ts),
            recv_tsc: 0,
            body,
        });
        self.seq += 1;
    }

    fn lots(&mut self, base: u64, p: f64) -> Qty {
        Qty::from_raw((base + self.rng.geometric(p)).min(50) as i64 * self.lot_raw)
    }

    fn submit(&mut self, out: &mut Vec<MarketEvent>, side: Side, ticks: i64, qty: Qty, tif: TimeInForce) {
        let id = OrderId(self.next_id);
        self.next_id += 1;
        let order = LimitOrder {
            id,
            side,
            price: self.px(ticks),
            qty,
            tif,
        };
        let mut fills = core::mem::take(&mut self.fills);
        fills.clear();
        let result = self.book.submit(order, &mut |f: &Fill| fills.push(*f));
        if let Ok(report) = result {
            for f in &fills {
                let match_id = self.next_match;
                self.next_match += 1;
                self.emit(
                    out,
                    EventBody::Execute {
                        order_id: f.maker_id,
                        qty: f.qty,
                        match_id,
                    },
                );
            }
            if report.resting.is_positive() {
                self.emit(
                    out,
                    EventBody::Add {
                        order_id: id,
                        side,
                        price: order.price,
                        qty: report.resting,
                    },
                );
                self.live.push(id);
            }
        }
        self.fills = fills;
    }

    fn pick_live(&mut self) -> Option<(usize, OrderId)> {
        for _ in 0..8 {
            if self.live.is_empty() {
                return None;
            }
            let i = self.rng.below(self.live.len());
            let id = self.live[i];
            if self.book.order(id).is_some() {
                return Some((i, id));
            }
            self.live.swap_remove(i);
        }
        None
    }

    /// Advances exchange time by one exponential inter-arrival and runs one agent action.
    pub fn step(&mut self, out: &mut Vec<MarketEvent>) {
        let dt = self.rng.exponential(1.0 / self.cfg.action_rate_hz);
        self.ts += ((dt * 1e9) as u64).max(1);
        self.fair += self.cfg.fair_sigma * dt.sqrt() * self.rng.normal();
        if self.rng.chance(self.cfg.jump_rate_hz * dt) {
            let sign = if self.rng.chance(0.5) { 1.0 } else { -1.0 };
            self.fair += sign * self.cfg.jump_ticks;
        }

        let tick = self.cfg.tick;
        let best_bid = self.book.best_bid().map(|l| l.price.to_ticks(tick));
        let best_ask = self.book.best_ask().map(|l| l.price.to_ticks(tick));
        let (bb, ba) = match (best_bid, best_ask) {
            (Some(b), Some(a)) => (b, a),
            (None, Some(a)) => {
                let qty = self.lots(1, 0.35);
                let px = (self.fair.floor() as i64).min(a - 1);
                return self.submit(out, Side::Bid, px, qty, TimeInForce::Gtc);
            }
            (Some(b), None) => {
                let qty = self.lots(1, 0.35);
                let px = (self.fair.ceil() as i64).max(b + 1);
                return self.submit(out, Side::Ask, px, qty, TimeInForce::Gtc);
            }
            (None, None) => {
                let f = self.fair.round() as i64;
                let qty = self.lots(1, 0.35);
                return self.submit(out, Side::Bid, f - 1, qty, TimeInForce::Gtc);
            }
        };
        let mid = 0.5 * (bb + ba) as f64;
        let gap = self.fair - mid;

        let informed = self.cfg.informed_weight * ((gap.abs() - 0.75) / 2.0).clamp(0.0, 1.0);
        let cancel = 0.40 * (self.live.len() as f64 / self.cfg.target_orders as f64).min(3.0);
        let weights = [
            0.50,
            cancel,
            informed,
            self.cfg.noise_taker_weight,
            self.cfg.replace_weight,
            self.cfg.hidden_weight,
        ];
        let total: f64 = weights.iter().sum();
        let mut r = self.rng.uniform() * total;
        let mut action = 0;
        for (i, w) in weights.iter().enumerate() {
            if r < *w {
                action = i;
                break;
            }
            r -= w;
        }

        match action {
            0 => {
                let anchor = mid + self.cfg.provider_lean * gap;
                let depth = self.rng.geometric(0.3) as i64;
                let qty = self.lots(1, 0.35);
                if self.rng.chance(0.5) {
                    let px = ((anchor - 0.5).floor() as i64 - depth).min(ba - 1);
                    self.submit(out, Side::Bid, px, qty, TimeInForce::Gtc);
                } else {
                    let px = ((anchor + 0.5).ceil() as i64 + depth).max(bb + 1);
                    self.submit(out, Side::Ask, px, qty, TimeInForce::Gtc);
                }
            }
            1 => {
                let Some((i, id)) = self.pick_live() else { return };
                let open = self.book.order(id).map(|o| o.qty.raw()).unwrap_or(0);
                let shares = open / core_types::SCALE;
                if shares >= 2 && self.rng.chance(0.15) {
                    let cut = Qty::from_raw(shares / 2 * core_types::SCALE);
                    if self.book.reduce(id, cut).is_ok() {
                        self.emit(out, EventBody::Cancel { order_id: id, qty: cut });
                    }
                } else if self.book.cancel(id).is_ok() {
                    self.live.swap_remove(i);
                    self.emit(out, EventBody::Delete { order_id: id });
                }
            }
            2 => {
                let qty = self.lots(2, 0.25);
                let tif = if self.rng.chance(0.3) {
                    TimeInForce::Gtc
                } else {
                    TimeInForce::Ioc
                };
                if gap > 0.0 {
                    let limit = (self.fair.floor() as i64).clamp(ba, ba + 2);
                    self.submit(out, Side::Bid, limit, qty, tif);
                } else {
                    let limit = (self.fair.ceil() as i64).clamp(bb - 2, bb);
                    self.submit(out, Side::Ask, limit, qty, tif);
                }
            }
            3 => {
                let qty = self.lots(1, 0.5);
                if self.rng.chance(0.5) {
                    self.submit(out, Side::Bid, ba + 1, qty, TimeInForce::Ioc);
                } else {
                    self.submit(out, Side::Ask, bb - 1, qty, TimeInForce::Ioc);
                }
            }
            4 => {
                let Some((i, id)) = self.pick_live() else { return };
                let Some(view) = self.book.order(id) else { return };
                let old = view.price.to_ticks(tick);
                let step = if self.rng.chance(0.5) { 1 } else { -1 };
                let ticks = match view.side {
                    Side::Bid => (old + step).min(ba - 1),
                    Side::Ask => (old + step).max(bb + 1),
                };
                let new_id = OrderId(self.next_id);
                self.next_id += 1;
                let price = self.px(ticks);
                if self
                    .book
                    .replace(id, new_id, price, view.qty, &mut |_: &Fill| {})
                    .is_ok()
                {
                    self.live[i] = new_id;
                    self.emit(
                        out,
                        EventBody::Replace {
                            order_id: id,
                            new_order_id: new_id,
                            price,
                            qty: view.qty,
                        },
                    );
                }
            }
            _ => {
                let side = if self.rng.chance(0.6) == (gap > 0.0) {
                    Side::Bid
                } else {
                    Side::Ask
                };
                let price = self.px(if side == Side::Bid { ba } else { bb });
                let match_id = self.next_match;
                self.next_match += 1;
                let qty = self.lots(1, 0.5);
                self.emit(
                    out,
                    EventBody::Trade {
                        side,
                        price,
                        qty,
                        match_id,
                    },
                );
            }
        }
    }

    /// Runs agent actions until at least `events` market events have been emitted.
    pub fn generate(&mut self, events: usize) -> Vec<MarketEvent> {
        let mut out = Vec::with_capacity(events + 64);
        while out.len() < events {
            self.step(&mut out);
        }
        out
    }
}

/// Serialises events as framed ITCH 5.0 messages.
pub fn encode_itch(events: &[MarketEvent], symbol: &str) -> Result<Vec<u8>, ItchError> {
    let mut enc = ItchEncoder::new(1, symbol);
    let mut buf = Vec::with_capacity(events.len() * 40);
    for ev in events {
        enc.encode(&mut buf, ev.exch_ts, &ev.body)?;
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_moments() {
        let mut r = SimRng::new(1);
        let n = 200_000;
        let (mut s, mut s2) = (0.0, 0.0);
        for _ in 0..n {
            let z = r.normal();
            s += z;
            s2 += z * z;
        }
        assert!((s / n as f64).abs() < 0.01);
        assert!((s2 / n as f64 - 1.0).abs() < 0.02);
        let mean_exp = (0..n).map(|_| r.exponential(2.0)).sum::<f64>() / n as f64;
        assert!((mean_exp - 2.0).abs() < 0.03);
        assert!((0..1000).all(|_| r.below(7) < 7));
    }

    #[test]
    fn deterministic_for_seed_and_book_stays_valid() {
        let run = |seed| {
            let mut sim = MarketSimulator::new(SimConfig {
                seed,
                ..SimConfig::default()
            })
            .unwrap();
            let ev = sim.generate(20_000);
            sim.book().audit().unwrap();
            ev
        };
        assert_eq!(run(3), run(3));
        assert_ne!(run(3), run(4));
    }
}
