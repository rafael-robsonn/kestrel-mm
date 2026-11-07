#![forbid(unsafe_code)]

use core_types::{Price, Qty, Quote, Timestamp};
use ml_alpha::ToxicityEstimate;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StrategyError {
    #[error("invalid strategy parameter: {0}")]
    InvalidParameter(&'static str),
    #[error("non-finite market input")]
    NonFiniteInput,
    #[error("quote price outside representable range")]
    PriceOverflow,
    #[error("computed quote is crossed")]
    CrossedQuote,
}

/// Time-to-horizon `T - t` used by the model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Horizon {
    /// Constant look-ahead `τ`, the usual choice for continuous intraday quoting.
    Rolling { tau_s: f64 },
    /// Finite session ending at `end`; `τ` is floored at `min_tau_s`.
    Session { end: Timestamp, min_tau_s: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AsConfig {
    pub tick_size: Price,
    pub lot_size: Qty,
    /// Risk aversion `γ`, per tick.
    pub gamma: f64,
    /// Fill-intensity decay `k` in `λ(δ) = A e^{-kδ}`, per tick.
    pub k: f64,
    pub horizon: Horizon,
    /// Quoting on the side that would increase `|q|` stops at this many lots.
    pub max_inventory_lots: i64,
    /// `η`: half-spread add-on per unit toxicity, in units of `σ√h`.
    pub toxicity_spread_mult: f64,
    /// `ζ`: reservation shift per unit toxicity asymmetry, in units of `σ√h`.
    pub toxicity_skew_mult: f64,
    /// A side is withdrawn when its toxicity probability reaches this level.
    pub toxicity_pull: f64,
    /// `p̄`: unconditional toxicity already priced into `k`; only `p - p̄` widens the spread.
    pub toxicity_baseline: f64,
    /// `h`: adverse-selection horizon, seconds.
    pub adverse_horizon_s: f64,
    /// Horizon over which the filtered drift is projected into the reservation price, seconds.
    pub drift_horizon_s: f64,
    pub min_half_spread_ticks: f64,
    /// Lower bound on `σ`, ticks/√s; protects the inventory term in dead markets.
    pub sigma_floor: f64,
}

impl Default for AsConfig {
    fn default() -> Self {
        Self {
            tick_size: Price::from_raw(1_000_000),
            lot_size: Qty::from_raw(100 * core_types::SCALE),
            gamma: 0.05,
            k: 2.0,
            horizon: Horizon::Rolling { tau_s: 1.0 },
            max_inventory_lots: 20,
            toxicity_spread_mult: 1.0,
            toxicity_skew_mult: 0.5,
            toxicity_pull: 0.9,
            toxicity_baseline: 0.0,
            adverse_horizon_s: 0.1,
            drift_horizon_s: 0.1,
            min_half_spread_ticks: 0.5,
            sigma_floor: 0.5,
        }
    }
}

/// Market state in ticks. `reference` is the price the reservation is anchored to
/// (micro-price or filtered fair value).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct MarketInputs {
    pub ts: Timestamp,
    pub reference: f64,
    pub best_bid: f64,
    pub best_ask: f64,
    /// Diffusion, ticks/√s.
    pub sigma: f64,
    /// Drift, ticks/s.
    pub drift: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct QuoteDecision {
    pub quote: Quote,
    pub reservation: f64,
    pub half_spread_bid: f64,
    pub half_spread_ask: f64,
}

/// Avellaneda-Stoikov (2008) market maker with an adverse-selection overlay.
///
/// Base model, all prices in ticks, `q` inventory in lots, `τ = T - t`:
///
/// `r = s − q γ σ² τ`
/// `δ^a + δ^b = γ σ² τ + (2/γ) ln(1 + γ/k)`
///
/// Overlay with side toxicities `p^b, p^a` from the GBDT, baseline `p̄` and `ν = σ√h`:
///
/// `r' = r + μ T_μ + ζ (p^a − p^b) ν`
/// `δ^b = max(½(δ^a + δ^b) + η (p^b − p̄)⁺ ν, δ_min)`, likewise `δ^a` with `p^a`
///
/// Quotes are `⌊r' − δ^b⌋` and `⌈r' + δ^a⌉`, clamped post-only against the touch. A side
/// is withdrawn when its toxicity reaches `toxicity_pull` or inventory hits the limit.
/// O(1), one `sqrt`; the `ln` term is precomputed.
#[derive(Clone, Debug)]
pub struct AvellanedaStoikov {
    cfg: AsConfig,
    spread_const: f64,
    sqrt_h: f64,
    tick_raw: f64,
}

const TICK_LIMIT: f64 = 9.0e15;

impl AvellanedaStoikov {
    pub fn new(cfg: AsConfig) -> Result<Self, StrategyError> {
        let positive = [cfg.gamma, cfg.k, cfg.adverse_horizon_s, cfg.sigma_floor];
        if positive.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(StrategyError::InvalidParameter(
                "gamma, k, adverse_horizon_s, sigma_floor must be > 0",
            ));
        }
        let non_negative = [
            cfg.toxicity_spread_mult,
            cfg.toxicity_skew_mult,
            cfg.drift_horizon_s,
            cfg.min_half_spread_ticks,
        ];
        if non_negative.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(StrategyError::InvalidParameter(
                "overlay parameters must be finite and >= 0",
            ));
        }
        if cfg.toxicity_pull.is_nan() || cfg.toxicity_pull <= 0.0 || cfg.toxicity_pull > 1.0 {
            return Err(StrategyError::InvalidParameter("toxicity_pull must be in (0, 1]"));
        }
        if !(0.0..1.0).contains(&cfg.toxicity_baseline) {
            return Err(StrategyError::InvalidParameter("toxicity_baseline must be in [0, 1)"));
        }
        if !cfg.tick_size.is_positive() || !cfg.lot_size.is_positive() || cfg.max_inventory_lots <= 0 {
            return Err(StrategyError::InvalidParameter(
                "tick, lot and inventory limit must be positive",
            ));
        }
        match cfg.horizon {
            Horizon::Rolling { tau_s } if tau_s.is_finite() && tau_s > 0.0 => {}
            Horizon::Session { min_tau_s, .. } if min_tau_s.is_finite() && min_tau_s > 0.0 => {}
            _ => return Err(StrategyError::InvalidParameter("horizon must be positive")),
        }
        Ok(Self {
            cfg,
            spread_const: (2.0 / cfg.gamma) * (1.0 + cfg.gamma / cfg.k).ln(),
            sqrt_h: cfg.adverse_horizon_s.sqrt(),
            tick_raw: cfg.tick_size.raw() as f64,
        })
    }

    pub fn config(&self) -> &AsConfig {
        &self.cfg
    }

    #[inline(always)]
    fn tau(&self, now: Timestamp) -> f64 {
        match self.cfg.horizon {
            Horizon::Rolling { tau_s } => tau_s,
            Horizon::Session { end, min_tau_s } => end.seconds_since(now).max(min_tau_s),
        }
    }

    #[inline(always)]
    fn to_price(&self, ticks: f64) -> Result<Price, StrategyError> {
        if !ticks.is_finite() || ticks.abs() >= TICK_LIMIT {
            return Err(StrategyError::PriceOverflow);
        }
        Price::from_ticks(ticks as i64, self.cfg.tick_size).map_err(|_| StrategyError::PriceOverflow)
    }

    #[inline]
    pub fn quote(
        &self,
        m: &MarketInputs,
        inventory_lots: i64,
        tox: ToxicityEstimate,
    ) -> Result<QuoteDecision, StrategyError> {
        let finite = [m.reference, m.best_bid, m.best_ask, m.sigma, m.drift, tox.bid, tox.ask];
        if finite.iter().any(|v| !v.is_finite()) {
            return Err(StrategyError::NonFiniteInput);
        }
        let cfg = &self.cfg;
        let p_bid = tox.bid.clamp(0.0, 1.0);
        let p_ask = tox.ask.clamp(0.0, 1.0);
        let sigma = m.sigma.max(cfg.sigma_floor);
        let var_tau = sigma * sigma * self.tau(m.ts);
        let q = inventory_lots as f64;
        let nu = sigma * self.sqrt_h;

        let reservation = m.reference + m.drift * cfg.drift_horizon_s - q * cfg.gamma * var_tau
            + cfg.toxicity_skew_mult * (p_ask - p_bid) * nu;
        let half = 0.5 * (cfg.gamma * var_tau + self.spread_const);
        let excess_bid = (p_bid - cfg.toxicity_baseline).max(0.0);
        let excess_ask = (p_ask - cfg.toxicity_baseline).max(0.0);
        let half_bid = (half + cfg.toxicity_spread_mult * excess_bid * nu).max(cfg.min_half_spread_ticks);
        let half_ask = (half + cfg.toxicity_spread_mult * excess_ask * nu).max(cfg.min_half_spread_ticks);

        let bid_ticks = (reservation - half_bid).floor().min(m.best_ask - 1.0);
        let ask_ticks = (reservation + half_ask).ceil().max(m.best_bid + 1.0);
        if bid_ticks >= ask_ticks {
            return Err(StrategyError::CrossedQuote);
        }

        let bid_active = inventory_lots < cfg.max_inventory_lots && p_bid < cfg.toxicity_pull;
        let ask_active = inventory_lots > -cfg.max_inventory_lots && p_ask < cfg.toxicity_pull;
        let reservation_raw = reservation * self.tick_raw;
        if !reservation_raw.is_finite() || reservation_raw.abs() >= 9.0e18 {
            return Err(StrategyError::PriceOverflow);
        }

        Ok(QuoteDecision {
            quote: Quote {
                ts: m.ts,
                bid_px: self.to_price(bid_ticks)?,
                ask_px: self.to_price(ask_ticks)?,
                bid_qty: if bid_active { cfg.lot_size } else { Qty::ZERO },
                ask_qty: if ask_active { cfg.lot_size } else { Qty::ZERO },
                reservation_px: Price::from_raw(reservation_raw.round() as i64),
                bid_active,
                ask_active,
            },
            reservation,
            half_spread_bid: half_bid,
            half_spread_ask: half_ask,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn mm() -> AvellanedaStoikov {
        AvellanedaStoikov::new(AsConfig::default()).unwrap()
    }

    fn market(reference: f64) -> MarketInputs {
        MarketInputs {
            ts: Timestamp(0),
            reference,
            best_bid: 9_999.0,
            best_ask: 10_001.0,
            sigma: 3.0,
            drift: 0.0,
        }
    }

    const CLEAN: ToxicityEstimate = ToxicityEstimate { bid: 0.0, ask: 0.0 };

    fn ticks(p: Price) -> i64 {
        p.to_ticks(AsConfig::default().tick_size)
    }

    #[test]
    fn flat_inventory_is_symmetric() {
        let d = mm().quote(&market(10_000.0), 0, CLEAN).unwrap();
        assert_eq!(d.reservation, 10_000.0);
        assert_eq!(d.half_spread_bid, d.half_spread_ask);
        assert_eq!(10_000 - ticks(d.quote.bid_px), ticks(d.quote.ask_px) - 10_000);
        assert!(d.quote.bid_active && d.quote.ask_active);
    }

    #[test]
    fn closed_form_reservation_and_spread() {
        let cfg = AsConfig::default();
        let d = mm().quote(&market(10_000.0), 3, CLEAN).unwrap();
        let var_tau = 9.0 * 1.0;
        assert!((d.reservation - (10_000.0 - 3.0 * cfg.gamma * var_tau)).abs() < 1e-9);
        let total = cfg.gamma * var_tau + (2.0 / cfg.gamma) * (1.0 + cfg.gamma / cfg.k).ln();
        assert!((d.half_spread_bid + d.half_spread_ask - total).abs() < 1e-9);
    }

    #[test]
    fn inventory_limit_withdraws_accumulating_side() {
        let long = mm().quote(&market(10_000.0), 20, CLEAN).unwrap().quote;
        assert!(!long.bid_active && long.bid_qty.is_zero() && long.ask_active);
        let short = mm().quote(&market(10_000.0), -20, CLEAN).unwrap().quote;
        assert!(short.bid_active && !short.ask_active);
    }

    #[test]
    fn toxic_side_widens_then_pulls() {
        let base = mm().quote(&market(10_000.0), 0, CLEAN).unwrap();
        let tox = mm()
            .quote(&market(10_000.0), 0, ToxicityEstimate { bid: 0.0, ask: 0.6 })
            .unwrap();
        assert!(tox.half_spread_ask > base.half_spread_ask);
        assert_eq!(tox.half_spread_bid, base.half_spread_bid);
        assert!(tox.reservation > base.reservation);
        let pulled = mm()
            .quote(&market(10_000.0), 0, ToxicityEstimate { bid: 0.95, ask: 0.0 })
            .unwrap();
        assert!(!pulled.quote.bid_active && pulled.quote.ask_active);
    }

    #[test]
    fn baseline_toxicity_is_not_charged_twice() {
        let cfg = AsConfig {
            toxicity_baseline: 0.35,
            ..AsConfig::default()
        };
        let s = AvellanedaStoikov::new(cfg).unwrap();
        let at_base = s
            .quote(&market(10_000.0), 0, ToxicityEstimate { bid: 0.35, ask: 0.35 })
            .unwrap();
        let clean = s.quote(&market(10_000.0), 0, CLEAN).unwrap();
        assert_eq!(at_base.half_spread_bid, clean.half_spread_bid);
        let above = s
            .quote(&market(10_000.0), 0, ToxicityEstimate { bid: 0.6, ask: 0.35 })
            .unwrap();
        assert!(above.half_spread_bid > at_base.half_spread_bid);
    }

    #[test]
    fn session_horizon_shrinks_inventory_penalty() {
        let cfg = AsConfig {
            horizon: Horizon::Session {
                end: Timestamp(10_000_000_000),
                min_tau_s: 0.01,
            },
            ..AsConfig::default()
        };
        let s = AvellanedaStoikov::new(cfg).unwrap();
        let early = s.quote(&market(10_000.0), 5, CLEAN).unwrap();
        let late = s
            .quote(
                &MarketInputs {
                    ts: Timestamp(9_990_000_000),
                    ..market(10_000.0)
                },
                5,
                CLEAN,
            )
            .unwrap();
        assert!(10_000.0 - early.reservation > 10_000.0 - late.reservation);
    }

    #[test]
    fn rejects_invalid_parameters_and_inputs() {
        assert!(
            AvellanedaStoikov::new(AsConfig {
                gamma: 0.0,
                ..AsConfig::default()
            })
            .is_err()
        );
        assert!(
            AvellanedaStoikov::new(AsConfig {
                toxicity_pull: 1.5,
                ..AsConfig::default()
            })
            .is_err()
        );
        assert!(
            AvellanedaStoikov::new(AsConfig {
                toxicity_baseline: 1.0,
                ..AsConfig::default()
            })
            .is_err()
        );
        assert!(
            AvellanedaStoikov::new(AsConfig {
                horizon: Horizon::Rolling { tau_s: -1.0 },
                ..AsConfig::default()
            })
            .is_err()
        );
        assert_eq!(
            mm().quote(&market(f64::NAN), 0, CLEAN),
            Err(StrategyError::NonFiniteInput)
        );
        assert_eq!(mm().quote(&market(1e300), 0, CLEAN), Err(StrategyError::PriceOverflow));
    }

    proptest! {
        #[test]
        fn quotes_are_uncrossed_post_only_and_monotone(
            reference in 9_950.0f64..10_050.0,
            spread in 1u32..6,
            sigma in 0.0f64..40.0,
            drift in -100.0f64..100.0,
            q in -25i64..25,
            p_bid in 0.0f64..1.0,
            p_ask in 0.0f64..1.0,
        ) {
            let best_bid = 10_000.0;
            let m = MarketInputs { ts: Timestamp(0), reference, best_bid, best_ask: best_bid + f64::from(spread), sigma, drift };
            let s = mm();
            let tox = ToxicityEstimate { bid: p_bid, ask: p_ask };
            let d = s.quote(&m, q, tox).unwrap();
            let (b, a) = (ticks(d.quote.bid_px) as f64, ticks(d.quote.ask_px) as f64);
            prop_assert!(b < a);
            prop_assert!(b <= m.best_ask - 1.0);
            prop_assert!(a >= m.best_bid + 1.0);
            prop_assert!(d.half_spread_bid >= s.config().min_half_spread_ticks);

            let more_long = s.quote(&m, q + 1, tox).unwrap();
            prop_assert!(more_long.reservation < d.reservation);

            let worse = ToxicityEstimate { bid: (p_bid + 0.05).min(1.0), ask: p_ask };
            let w = s.quote(&m, q, worse).unwrap();
            prop_assert!(w.half_spread_bid >= d.half_spread_bid);
            prop_assert!(ticks(w.quote.bid_px) <= ticks(d.quote.bid_px));
        }
    }
}
