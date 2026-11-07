use core_types::{Qty, Side};

use crate::FeatureError;

/// Volume-Synchronised Probability of Informed Trading (Easley, López de Prado, O'Hara 2012).
///
/// Trades are packed into equal-volume buckets of size `V`; a trade straddling a bucket
/// boundary is split. Over the last `N` completed buckets:
///
/// `VPIN = Σ |V^B_τ - V^S_τ| / (N · V)`
///
/// Aggressor side is taken from the L3 feed, so no bulk-volume classification is needed.
/// Bucket imbalances are kept as exact `i64` raw quantities and the window sum is
/// maintained incrementally: `on_trade` is O(1) amortised and O(N) worst case for a
/// single print larger than the whole window; `value` is O(1).
#[derive(Clone, Debug)]
pub struct Vpin<const N: usize> {
    bucket_volume: i64,
    buckets: [i64; N],
    head: usize,
    filled: usize,
    window_abs: i64,
    cur_buy: i64,
    cur_sell: i64,
}

impl<const N: usize> Vpin<N> {
    const NON_EMPTY_WINDOW: () = assert!(N > 0);

    pub fn new(bucket_volume: Qty) -> Result<Self, FeatureError> {
        let () = Self::NON_EMPTY_WINDOW;
        if !bucket_volume.is_positive() {
            return Err(FeatureError::InvalidConfig("vpin bucket volume must be positive"));
        }
        if bucket_volume.raw().checked_mul(N as i64).is_none() {
            return Err(FeatureError::InvalidConfig("vpin window volume overflows i64"));
        }
        Ok(Self {
            bucket_volume: bucket_volume.raw(),
            buckets: [0; N],
            head: 0,
            filled: 0,
            window_abs: 0,
            cur_buy: 0,
            cur_sell: 0,
        })
    }

    pub fn reset(&mut self) {
        self.buckets = [0; N];
        self.head = 0;
        self.filled = 0;
        self.window_abs = 0;
        self.cur_buy = 0;
        self.cur_sell = 0;
    }

    #[inline(always)]
    fn close_bucket(&mut self) {
        let imbalance = (self.cur_buy - self.cur_sell).abs();
        self.window_abs += imbalance - self.buckets[self.head];
        self.buckets[self.head] = imbalance;
        self.head = if self.head + 1 == N { 0 } else { self.head + 1 };
        if self.filled < N {
            self.filled += 1;
        }
        self.cur_buy = 0;
        self.cur_sell = 0;
    }

    #[inline]
    pub fn on_trade(&mut self, aggressor: Side, qty: Qty) {
        let mut q = qty.raw();
        if q <= 0 {
            return;
        }
        let v = self.bucket_volume;
        let room = v - (self.cur_buy + self.cur_sell);
        if q > room {
            let whole = (q - room) / v;
            let skip = (whole - N as i64).max(0);
            q -= skip * v;
        }
        while q > 0 {
            let room = v - (self.cur_buy + self.cur_sell);
            let take = q.min(room);
            match aggressor {
                Side::Bid => self.cur_buy += take,
                Side::Ask => self.cur_sell += take,
            }
            q -= take;
            if take == room {
                self.close_bucket();
            }
        }
    }

    /// `0.0` until the first bucket completes.
    #[inline(always)]
    pub fn value(&self) -> f64 {
        if self.filled == 0 {
            0.0
        } else {
            self.window_abs as f64 / (self.filled as f64 * self.bucket_volume as f64)
        }
    }

    pub fn completed_buckets(&self) -> usize {
        self.filled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn q(v: i64) -> Qty {
        Qty::from_raw(v)
    }

    #[test]
    fn one_sided_flow_saturates() {
        let mut v = Vpin::<4>::new(q(100)).unwrap();
        v.on_trade(Side::Bid, q(1000));
        assert_eq!(v.completed_buckets(), 4);
        assert!((v.value() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn balanced_flow_is_zero() {
        let mut v = Vpin::<4>::new(q(100)).unwrap();
        for _ in 0..20 {
            v.on_trade(Side::Bid, q(25));
            v.on_trade(Side::Ask, q(25));
        }
        assert_eq!(v.value(), 0.0);
    }

    #[test]
    fn splits_trades_across_buckets() {
        let mut v = Vpin::<2>::new(q(10)).unwrap();
        v.on_trade(Side::Bid, q(15));
        v.on_trade(Side::Ask, q(5));
        assert_eq!(v.completed_buckets(), 2);
        assert!((v.value() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn print_shorter_than_window_does_not_skip_buckets() {
        let mut v = Vpin::<5>::new(q(17)).unwrap();
        v.on_trade(Side::Ask, q(1));
        v.on_trade(Side::Bid, q(17));
        assert_eq!(v.completed_buckets(), 1);
        assert!((v.value() - 15.0 / 17.0).abs() < 1e-12);
    }

    fn reference(trades: &[(bool, i64)], v: i64, n: usize) -> f64 {
        let mut buckets = Vec::new();
        let (mut b, mut s) = (0i64, 0i64);
        for &(buy, qty) in trades {
            for _ in 0..qty {
                if buy {
                    b += 1
                } else {
                    s += 1
                }
                if b + s == v {
                    buckets.push((b - s).abs());
                    b = 0;
                    s = 0;
                }
            }
        }
        if buckets.is_empty() {
            return 0.0;
        }
        let tail = &buckets[buckets.len().saturating_sub(n)..];
        tail.iter().sum::<i64>() as f64 / (tail.len() as f64 * v as f64)
    }

    proptest! {
        #[test]
        fn matches_unit_by_unit_reference(trades in prop::collection::vec((any::<bool>(), 1i64..120), 1..80)) {
            let mut v = Vpin::<5>::new(q(17)).unwrap();
            for &(buy, qty) in &trades {
                v.on_trade(if buy { Side::Bid } else { Side::Ask }, q(qty));
            }
            let expected = reference(&trades, 17, 5);
            prop_assert!((v.value() - expected).abs() < 1e-12, "{} vs {}", v.value(), expected);
            prop_assert!((0.0..=1.0).contains(&v.value()));
        }
    }
}
