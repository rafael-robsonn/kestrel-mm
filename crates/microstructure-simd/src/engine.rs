use core_types::{DepthSnapshot, FeatureVector, Fill, Price, Qty, SCALE, Timestamp, feature};
use wide::f64x8;

use crate::FeatureError;
use crate::ladder::{LadderLanes, MicroPrice, hsum, ofi_levels, weighted_micro_price};
use crate::vpin::Vpin;

/// Number of completed volume buckets in the VPIN window.
pub const VPIN_BUCKETS: usize = 50;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureConfig {
    pub tick_size: Price,
    /// Level weights are `exp(-level_decay * m)` for `m = 0..8`.
    pub level_decay: f64,
    /// e-folding time of OFI, depth and trade-flow accumulators, seconds.
    pub flow_tau_s: f64,
    /// e-folding time of the realised-variance estimator, seconds.
    pub vol_tau_s: f64,
    pub vpin_bucket: Qty,
}

impl FeatureConfig {
    fn validate(&self) -> Result<(), FeatureError> {
        if !self.tick_size.is_positive() {
            return Err(FeatureError::InvalidConfig("tick_size must be positive"));
        }
        for v in [self.level_decay, self.flow_tau_s, self.vol_tau_s] {
            if !v.is_finite() || v < 0.0 {
                return Err(FeatureError::InvalidConfig(
                    "decay parameters must be finite and non-negative",
                ));
            }
        }
        if self.flow_tau_s == 0.0 || self.vol_tau_s == 0.0 {
            return Err(FeatureError::InvalidConfig("time constants must be positive"));
        }
        Ok(())
    }
}

/// Microstructure state exposed alongside the feature vector, prices in ticks.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct BookState {
    pub ts: Timestamp,
    pub best_bid: f64,
    pub best_ask: f64,
    pub mid: f64,
    pub micro: f64,
}

const EPS: f64 = 1e-9;

/// Streaming feature extractor. Call [`FeatureEngine::on_trade`] for every fill emitted
/// while applying an event, then [`FeatureEngine::update`] once with the post-event
/// snapshot. All state is inline; no heap allocation after construction.
///
/// Decay is time-based: with `Δt` the exchange-time gap since the previous update,
/// every accumulator `A` evolves as `A ← A·exp(-Δt/τ) + x`.
#[derive(Clone, Debug)]
pub struct FeatureEngine {
    inv_tick_raw: f64,
    weights: f64x8,
    weight_sum: f64,
    inv_flow_tau: f64,
    inv_vol_tau: f64,
    prev: Option<LadderLanes>,
    last_ts: Timestamp,
    last_mid: f64,
    ofi_acc: f64x8,
    depth_avg: f64x8,
    flow_signed: f64,
    flow_total: f64,
    pending_signed: f64,
    pending_total: f64,
    var_sum: f64,
    var_time: f64,
    vpin: Vpin<VPIN_BUCKETS>,
    out: FeatureVector,
    state: BookState,
}

impl FeatureEngine {
    pub fn new(cfg: FeatureConfig) -> Result<Self, FeatureError> {
        cfg.validate()?;
        let mut w = [0.0; 8];
        for (m, wm) in w.iter_mut().enumerate() {
            *wm = (-cfg.level_decay * m as f64).exp();
        }
        Ok(Self {
            inv_tick_raw: 1.0 / cfg.tick_size.raw() as f64,
            weights: f64x8::new(w),
            weight_sum: w.iter().sum(),
            inv_flow_tau: 1.0 / cfg.flow_tau_s,
            inv_vol_tau: 1.0 / cfg.vol_tau_s,
            prev: None,
            last_ts: Timestamp(0),
            last_mid: 0.0,
            ofi_acc: f64x8::ZERO,
            depth_avg: f64x8::ZERO,
            flow_signed: 0.0,
            flow_total: 0.0,
            pending_signed: 0.0,
            pending_total: 0.0,
            var_sum: 0.0,
            var_time: 0.0,
            vpin: Vpin::new(cfg.vpin_bucket)?,
            out: FeatureVector::default(),
            state: BookState::default(),
        })
    }

    pub fn reset(&mut self) {
        self.prev = None;
        self.last_ts = Timestamp(0);
        self.last_mid = 0.0;
        self.ofi_acc = f64x8::ZERO;
        self.depth_avg = f64x8::ZERO;
        self.flow_signed = 0.0;
        self.flow_total = 0.0;
        self.pending_signed = 0.0;
        self.pending_total = 0.0;
        self.var_sum = 0.0;
        self.var_time = 0.0;
        self.vpin.reset();
        self.out = FeatureVector::default();
        self.state = BookState::default();
    }

    #[inline]
    pub fn on_trade(&mut self, fill: &Fill) {
        let q = fill.qty.raw() as f64 / SCALE as f64;
        self.pending_signed += fill.aggressor().sign() as f64 * q;
        self.pending_total += q;
        self.vpin.on_trade(fill.aggressor(), fill.qty);
    }

    /// Advances all estimators to `snap.ts`. Returns `None` while either side is empty.
    ///
    /// Cost: two `exp`, one `sqrt`, ~20 eight-lane vector ops, no branches on depth.
    #[inline]
    pub fn update(&mut self, snap: &DepthSnapshot) -> Option<&FeatureVector> {
        let cur = LadderLanes::from_snapshot(snap, self.inv_tick_raw);
        let dt = snap.ts.seconds_since(self.last_ts);
        let first = self.prev.is_none();
        let (flow_decay, vol_decay) = if first {
            (0.0, 0.0)
        } else {
            ((-dt * self.inv_flow_tau).exp(), (-dt * self.inv_vol_tau).exp())
        };

        let ofi = match &self.prev {
            Some(prev) => ofi_levels(&cur, prev),
            None => f64x8::ZERO,
        };
        let fd = f64x8::splat(flow_decay);
        let level_depth = (cur.bid_qty + cur.ask_qty) * 0.5;
        self.ofi_acc = self.ofi_acc * fd + ofi;
        self.depth_avg = self.depth_avg * fd + level_depth * (1.0 - flow_decay);
        self.flow_signed = self.flow_signed * flow_decay + self.pending_signed;
        self.flow_total = self.flow_total * flow_decay + self.pending_total;
        self.pending_signed = 0.0;
        self.pending_total = 0.0;
        self.prev = Some(cur);
        self.last_ts = snap.ts;

        if !snap.is_two_sided() {
            return None;
        }

        let MicroPrice {
            mid,
            micro,
            imbalance,
            spread,
        } = weighted_micro_price(&cur, self.weights);
        if self.last_mid != 0.0 && !first {
            let dm = mid - self.last_mid;
            self.var_sum = self.var_sum * vol_decay + dm * dm;
            self.var_time = self.var_time * vol_decay + dt;
        }
        self.last_mid = mid;

        let acc = self.ofi_acc.to_array();
        let depth = self.depth_avg.to_array();
        let w_acc = hsum(self.weights * self.ofi_acc);
        let w_depth = hsum(self.weights * self.depth_avg) / self.weight_sum;

        let v = &mut self.out.values;
        v[feature::OFI_L1] = acc[0] / (depth[0] + EPS);
        v[feature::OFI_MULTI] = w_acc / (self.weight_sum * w_depth + EPS);
        v[feature::DEPTH_IMBALANCE] = imbalance;
        v[feature::MICRO_DEVIATION] = micro - mid;
        v[feature::SPREAD] = spread;
        v[feature::VPIN] = self.vpin.value();
        v[feature::VOLATILITY] = if self.var_time > EPS {
            (self.var_sum / self.var_time).sqrt()
        } else {
            0.0
        };
        v[feature::TRADE_FLOW] = self.flow_signed / (self.flow_total + EPS);

        self.state = BookState {
            ts: snap.ts,
            best_bid: cur.bid_px.as_array()[0],
            best_ask: cur.ask_px.as_array()[0],
            mid,
            micro,
        };
        Some(&self.out)
    }

    pub fn features(&self) -> &FeatureVector {
        &self.out
    }

    pub fn book_state(&self) -> &BookState {
        &self.state
    }

    pub fn vpin(&self) -> f64 {
        self.vpin.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{OrderId, Side};

    const TICK_RAW: i64 = 1_000_000;

    fn cfg() -> FeatureConfig {
        FeatureConfig {
            tick_size: Price::from_raw(TICK_RAW),
            level_decay: 0.5,
            flow_tau_s: 0.5,
            vol_tau_s: 5.0,
            vpin_bucket: Qty::from_units(100).unwrap(),
        }
    }

    fn snap(ts_ms: u64, bid: (i64, i64), ask: (i64, i64)) -> DepthSnapshot {
        let mut s = DepthSnapshot::EMPTY;
        s.ts = Timestamp(ts_ms * 1_000_000);
        s.bid_px[0] = bid.0 * TICK_RAW;
        s.bid_qty[0] = bid.1 * SCALE;
        s.ask_px[0] = ask.0 * TICK_RAW;
        s.ask_qty[0] = ask.1 * SCALE;
        s.bid_levels = 1;
        s.ask_levels = 1;
        s
    }

    fn buy(q: i64) -> Fill {
        Fill {
            maker_id: OrderId(1),
            taker_id: OrderId::NONE,
            maker_side: Side::Ask,
            price: Price::ZERO,
            qty: Qty::from_units(q).unwrap(),
        }
    }

    #[test]
    fn one_sided_book_yields_none() {
        let mut fe = FeatureEngine::new(cfg()).unwrap();
        let mut s = snap(0, (100, 10), (101, 10));
        s.ask_levels = 0;
        s.ask_px[0] = core_types::ABSENT_ASK_PX;
        s.ask_qty[0] = 0;
        assert!(fe.update(&s).is_none());
    }

    #[test]
    fn buying_pressure_produces_positive_signed_features() {
        let mut fe = FeatureEngine::new(cfg()).unwrap();
        fe.update(&snap(0, (100, 10), (101, 10)));
        fe.on_trade(&buy(150));
        fe.update(&snap(1, (100, 20), (101, 3)));
        fe.on_trade(&buy(150));
        let f = *fe.update(&snap(2, (101, 12), (102, 2))).unwrap();

        assert!(f.values[feature::OFI_L1] > 0.0);
        assert!(f.values[feature::OFI_MULTI] > 0.0);
        assert!(f.values[feature::DEPTH_IMBALANCE] > 0.5);
        assert!(f.values[feature::MICRO_DEVIATION] > 0.0);
        assert_eq!(f.values[feature::SPREAD], 1.0);
        assert!((f.values[feature::TRADE_FLOW] - 1.0).abs() < 1e-6);
        assert!((f.values[feature::VPIN] - 1.0).abs() < 1e-12);
        assert!(f.values[feature::VOLATILITY] > 0.0);
        assert!(f.values.iter().all(|v| v.is_finite()));
        assert_eq!(fe.book_state().mid, 101.5);
    }

    #[test]
    fn flow_decays_with_exchange_time() {
        let mut fe = FeatureEngine::new(cfg()).unwrap();
        fe.update(&snap(0, (100, 10), (101, 10)));
        fe.on_trade(&buy(10));
        fe.update(&snap(1, (100, 10), (101, 10)));
        let before = fe.flow_signed;
        fe.update(&snap(501, (100, 10), (101, 10)));
        assert!((fe.flow_signed / before - (-1.0f64).exp()).abs() < 1e-9);
    }
}
