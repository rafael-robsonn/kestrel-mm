use core_types::{DepthSnapshot, FeatureVector, Fill, MarketEvent, Price, Qty, Quote, Side, feature};
use microstructure_simd::{BookState, FeatureConfig, FeatureEngine};
use ml_alpha::{EkfConfig, FairValueEkf, TOXICITY_FOREST, ToxicityEstimate, predict_toxicity};
use orderbook_l3::{BookConfig, FillSink, OrderBook};
use strategy_mm::{AsConfig, AvellanedaStoikov, Horizon, MarketInputs, QuoteDecision};

use crate::error::EngineError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineConfig {
    pub book: BookConfig,
    pub features: FeatureConfig,
    pub ekf: EkfConfig,
    pub strategy: AsConfig,
    /// EKF steps required before its `σ` replaces the realised-volatility feature.
    pub ekf_warmup_steps: u32,
}

impl EngineConfig {
    /// Defaults for a single equity: 65 536-tick ladder centred on `center`, 128k orders.
    pub fn for_instrument(center: Price, tick: Price, lot: Qty) -> Result<Self, EngineError> {
        Ok(Self {
            book: BookConfig::centered(center, tick, 1 << 16, 1 << 17)?,
            features: FeatureConfig {
                tick_size: tick,
                level_decay: 0.4,
                flow_tau_s: 0.25,
                vol_tau_s: 5.0,
                vpin_bucket: Qty::from_raw(lot.raw().saturating_mul(25)),
            },
            ekf: EkfConfig {
                sample_interval_ns: 100_000_000,
                obs_noise: 0.02,
                logvar_mean: 3.0,
                ..EkfConfig::default()
            },
            strategy: AsConfig {
                tick_size: tick,
                lot_size: lot,
                gamma: 0.018,
                k: 3.0,
                horizon: Horizon::Rolling { tau_s: 1.0 },
                max_inventory_lots: 20,
                toxicity_spread_mult: 1.0,
                toxicity_skew_mult: 0.5,
                toxicity_pull: 0.9,
                toxicity_baseline: ml_alpha::toxicity_model::BASE_RATE,
                adverse_horizon_s: 0.1,
                drift_horizon_s: 0.1,
                min_half_spread_ticks: 0.5,
                sigma_floor: 0.5,
            },
            ekf_warmup_steps: 100,
        })
    }
}

/// Simulated own position from conservative paper fills.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PaperPosition {
    /// Signed raw quantity.
    pub position: i64,
    /// Cash in price units (negative after buying).
    pub cash: f64,
    pub fills: u64,
    pub traded: i64,
}

impl PaperPosition {
    pub fn mark_to_market(&self, mid: f64) -> f64 {
        self.cash + Qty::from_raw(self.position).to_f64() * mid
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngineStats {
    pub events: u64,
    pub quotes: u64,
    pub book_rejects: u64,
    pub strategy_rejects: u64,
    pub filter_resets: u64,
}

/// Estimated FIFO position of the virtual quote at its price level: displayed quantity
/// that must trade before the quote would.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct QueuePosition {
    price: Price,
    ahead: i64,
}

/// Fills generated while the book applies one event. Feeds trade-driven features and
/// crosses the live quote against a FIFO queue model: a displayed print at our price first
/// consumes the queue ahead, the remainder fills us; a print strictly through our price
/// fills us directly; hidden prints (`maker_id == NONE`) fill only when through.
struct HotSink<'a> {
    features: &'a mut FeatureEngine,
    paper: &'a mut PaperPosition,
    live: &'a mut Quote,
    queue: &'a mut [QueuePosition; 2],
}

impl FillSink for HotSink<'_> {
    #[inline(always)]
    fn on_fill(&mut self, fill: &Fill) {
        self.features.on_trade(fill);
        let live = &mut *self.live;
        let (active, px, qty, through, sign) = match fill.aggressor() {
            Side::Bid => (
                live.ask_active,
                live.ask_px,
                &mut live.ask_qty,
                fill.price > live.ask_px,
                -1,
            ),
            Side::Ask => (
                live.bid_active,
                live.bid_px,
                &mut live.bid_qty,
                fill.price < live.bid_px,
                1,
            ),
        };
        if !active || !qty.is_positive() {
            return;
        }
        let queue = &mut self.queue[fill.maker_side.index()];
        let eligible = if through {
            fill.qty.raw()
        } else if fill.price == px && queue.price == px && !fill.maker_id.is_none() {
            let consumed = fill.qty.raw().min(queue.ahead);
            queue.ahead -= consumed;
            fill.qty.raw() - consumed
        } else {
            0
        };
        if eligible <= 0 {
            return;
        }
        let done = Qty::from_raw(eligible).min(*qty);
        *qty = qty.saturating_sub(done);
        self.paper.position += sign * done.raw();
        self.paper.cash -= sign as f64 * px.to_f64() * done.to_f64();
        self.paper.fills += 1;
        self.paper.traded += done.raw();
    }
}

/// Everything the strategy thread owns. All buffers are allocated in [`StrategyEngine::new`];
/// [`StrategyEngine::on_event`] performs no heap allocation (enforced by `tests/zero_alloc.rs`).
#[derive(Debug)]
pub struct StrategyEngine {
    cfg: EngineConfig,
    book: OrderBook,
    features: FeatureEngine,
    ekf: FairValueEkf,
    ekf_steps: u32,
    quoter: AvellanedaStoikov,
    snapshot: DepthSnapshot,
    live: Quote,
    queue: [QueuePosition; 2],
    decision: QuoteDecision,
    toxicity: ToxicityEstimate,
    paper: PaperPosition,
    stats: EngineStats,
    lot_raw: i64,
}

impl StrategyEngine {
    pub fn new(cfg: EngineConfig) -> Result<Self, EngineError> {
        if !cfg.strategy.lot_size.is_positive() {
            return Err(EngineError::Config("lot size must be positive"));
        }
        if cfg.features.tick_size != cfg.book.tick_size || cfg.strategy.tick_size != cfg.book.tick_size {
            return Err(EngineError::Config(
                "tick size must agree across book, features and strategy",
            ));
        }
        Ok(Self {
            cfg,
            book: OrderBook::new(cfg.book)?,
            features: FeatureEngine::new(cfg.features)?,
            ekf: FairValueEkf::new(cfg.ekf)?,
            ekf_steps: 0,
            quoter: AvellanedaStoikov::new(cfg.strategy)?,
            snapshot: DepthSnapshot::EMPTY,
            live: Quote::default(),
            queue: [QueuePosition::default(); 2],
            decision: QuoteDecision::default(),
            toxicity: ToxicityEstimate::default(),
            paper: PaperPosition::default(),
            stats: EngineStats::default(),
            lot_raw: cfg.strategy.lot_size.raw(),
        })
    }

    /// Restores the post-construction state without reallocating.
    pub fn reset(&mut self) {
        self.book.reset();
        self.features.reset();
        self.ekf.reset();
        self.ekf_steps = 0;
        self.snapshot = DepthSnapshot::EMPTY;
        self.live = Quote::default();
        self.queue = [QueuePosition::default(); 2];
        self.decision = QuoteDecision::default();
        self.toxicity = ToxicityEstimate::default();
        self.paper = PaperPosition::default();
        self.stats = EngineStats::default();
    }

    /// Full hot path for one event: book mutation, depth snapshot, SIMD features, EKF,
    /// GBDT toxicity, Avellaneda-Stoikov. Returns `Ok(None)` while the book is one-sided.
    ///
    /// A rejected book mutation leaves the book unchanged and returns `Err`; the caller
    /// decides whether the feed is still trustworthy.
    #[inline]
    pub fn on_event(&mut self, ev: &MarketEvent) -> Result<Option<Quote>, EngineError> {
        self.stats.events += 1;
        let mut sink = HotSink {
            features: &mut self.features,
            paper: &mut self.paper,
            live: &mut self.live,
            queue: &mut self.queue,
        };
        if let Err(e) = self.book.apply(&ev.body, &mut sink) {
            self.stats.book_rejects += 1;
            return Err(e.into());
        }
        self.book.snapshot(ev.exch_ts, &mut self.snapshot);
        let Some(fv) = self.features.update(&self.snapshot) else {
            self.live = Quote::default();
            return Ok(None);
        };
        let fv = *fv;
        let state = *self.features.book_state();

        match self.ekf.observe(state.micro, ev.exch_ts) {
            Ok(true) => self.ekf_steps = self.ekf_steps.saturating_add(1),
            Ok(false) => {}
            Err(_) => {
                self.stats.filter_resets += 1;
                self.ekf_steps = 0;
            }
        }
        let est = *self.ekf.estimate();
        let sigma = if self.ekf_steps >= self.cfg.ekf_warmup_steps {
            est.sigma
        } else {
            fv.values[feature::VOLATILITY]
        };

        self.toxicity = predict_toxicity(&TOXICITY_FOREST, &fv);
        let inputs = MarketInputs {
            ts: ev.exch_ts,
            reference: state.micro,
            best_bid: state.best_bid,
            best_ask: state.best_ask,
            sigma,
            drift: est.drift,
        };
        match self
            .quoter
            .quote(&inputs, self.paper.position / self.lot_raw, self.toxicity)
        {
            Ok(decision) => {
                self.decision = decision;
                self.live = decision.quote;
                self.track_queue(Side::Bid, decision.quote.bid_px);
                self.track_queue(Side::Ask, decision.quote.ask_px);
                self.stats.quotes += 1;
                Ok(Some(decision.quote))
            }
            Err(e) => {
                self.live = Quote::default();
                self.stats.strategy_rejects += 1;
                Err(e.into())
            }
        }
    }

    /// Joining a level puts the quote behind its displayed size; staying at the same price
    /// keeps the estimate, which can only shrink as orders ahead cancel.
    #[inline(always)]
    fn track_queue(&mut self, side: Side, price: Price) {
        let displayed = self.book.level_volume(side, price).map_or(0, |q| q.raw());
        let q = &mut self.queue[side.index()];
        if q.price == price {
            q.ahead = q.ahead.min(displayed);
        } else {
            *q = QueuePosition {
                price,
                ahead: displayed,
            };
        }
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    pub fn book(&self) -> &OrderBook {
        &self.book
    }

    pub fn features(&self) -> &FeatureVector {
        self.features.features()
    }

    pub fn book_state(&self) -> &BookState {
        self.features.book_state()
    }

    pub fn toxicity(&self) -> ToxicityEstimate {
        self.toxicity
    }

    pub fn decision(&self) -> &QuoteDecision {
        &self.decision
    }

    pub fn paper(&self) -> &PaperPosition {
        &self.paper
    }

    pub fn inventory_lots(&self) -> i64 {
        self.paper.position / self.lot_raw
    }

    pub fn stats(&self) -> &EngineStats {
        &self.stats
    }

    pub fn ekf(&self) -> &FairValueEkf {
        &self.ekf
    }
}
