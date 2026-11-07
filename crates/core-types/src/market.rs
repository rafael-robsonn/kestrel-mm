use crate::{Price, Qty, Timestamp};

/// Number of price levels per side carried in a [`DepthSnapshot`].
pub const DEPTH_LEVELS: usize = 8;

/// Raw price written into empty bid slots. Lower than any representable bid.
pub const ABSENT_BID_PX: i64 = i64::MIN;
/// Raw price written into empty ask slots. Higher than any representable ask.
pub const ABSENT_ASK_PX: i64 = i64::MAX;

/// Top-of-book ladder in raw fixed-point units, structure-of-arrays layout for SIMD loads.
///
/// Invariant: slots `>= bid_levels` hold `(ABSENT_BID_PX, 0)`, slots `>= ask_levels`
/// hold `(ABSENT_ASK_PX, 0)`. Bid prices strictly decrease, ask prices strictly increase.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C, align(64))]
pub struct DepthSnapshot {
    pub bid_px: [i64; DEPTH_LEVELS],
    pub bid_qty: [i64; DEPTH_LEVELS],
    pub ask_px: [i64; DEPTH_LEVELS],
    pub ask_qty: [i64; DEPTH_LEVELS],
    pub ts: Timestamp,
    pub bid_levels: u8,
    pub ask_levels: u8,
}

impl DepthSnapshot {
    pub const EMPTY: Self = Self {
        bid_px: [ABSENT_BID_PX; DEPTH_LEVELS],
        bid_qty: [0; DEPTH_LEVELS],
        ask_px: [ABSENT_ASK_PX; DEPTH_LEVELS],
        ask_qty: [0; DEPTH_LEVELS],
        ts: Timestamp(0),
        bid_levels: 0,
        ask_levels: 0,
    };

    #[inline(always)]
    pub const fn is_two_sided(&self) -> bool {
        self.bid_levels > 0 && self.ask_levels > 0
    }

    #[inline(always)]
    pub const fn best_bid(&self) -> Option<(Price, Qty)> {
        if self.bid_levels == 0 {
            None
        } else {
            Some((Price::from_raw(self.bid_px[0]), Qty::from_raw(self.bid_qty[0])))
        }
    }

    #[inline(always)]
    pub const fn best_ask(&self) -> Option<(Price, Qty)> {
        if self.ask_levels == 0 {
            None
        } else {
            Some((Price::from_raw(self.ask_px[0]), Qty::from_raw(self.ask_qty[0])))
        }
    }
}

impl Default for DepthSnapshot {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Two-sided quote produced by the strategy. Inactive sides carry zero quantity.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C, align(64))]
pub struct Quote {
    pub ts: Timestamp,
    pub bid_px: Price,
    pub ask_px: Price,
    pub bid_qty: Qty,
    pub ask_qty: Qty,
    pub reservation_px: Price,
    pub bid_active: bool,
    pub ask_active: bool,
}

const _: () = assert!(core::mem::size_of::<Quote>() == 64);
