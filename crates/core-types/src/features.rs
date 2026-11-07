/// Width of the model input vector. Exactly one cache line of `f64`.
pub const N_FEATURES: usize = 8;

/// Feature slot indices into [`FeatureVector::values`].
pub mod feature {
    /// Level-1 OFI, exponentially decayed, normalised by mean L1 depth. Signed.
    pub const OFI_L1: usize = 0;
    /// Depth-weighted multi-level OFI, normalised by weighted mean depth. Signed.
    pub const OFI_MULTI: usize = 1;
    /// `(D_b - D_a) / (D_b + D_a)` over weighted depth. Signed, in `[-1, 1]`.
    pub const DEPTH_IMBALANCE: usize = 2;
    /// `(micro - mid) / tick`. Signed.
    pub const MICRO_DEVIATION: usize = 3;
    /// `(ask_1 - bid_1) / tick`. Unsigned.
    pub const SPREAD: usize = 4;
    /// Volume-synchronised probability of informed trading. Unsigned, in `[0, 1]`.
    pub const VPIN: usize = 5;
    /// EWMA realised volatility of the mid, ticks per sqrt-second. Unsigned.
    pub const VOLATILITY: usize = 6;
    /// Decayed signed aggressor volume over decayed total volume. Signed, in `[-1, 1]`.
    pub const TRADE_FLOW: usize = 7;
}

pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "ofi_l1",
    "ofi_multi",
    "depth_imbalance",
    "micro_deviation",
    "spread_ticks",
    "vpin",
    "volatility",
    "trade_flow",
];

/// `true` for features whose sign flips with the side of the book.
pub const SIGNED_FEATURES: [bool; N_FEATURES] = [true, true, true, true, false, false, false, true];

#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[repr(C, align(64))]
pub struct FeatureVector {
    pub values: [f64; N_FEATURES],
}

const _: () = assert!(core::mem::size_of::<FeatureVector>() == 64);

impl FeatureVector {
    /// Re-expresses the vector from the perspective of an aggressor on side `sign`
    /// (`+1.0` buy, `-1.0` sell): signed features are multiplied by `sign`, unsigned
    /// ones are copied. Non-finite entries are mapped to `0.0`.
    #[inline(always)]
    pub fn oriented(&self, sign: f64) -> Self {
        let mut out = [0.0; N_FEATURES];
        for i in 0..N_FEATURES {
            let v = if SIGNED_FEATURES[i] {
                self.values[i] * sign
            } else {
                self.values[i]
            };
            out[i] = if v.is_finite() { v } else { 0.0 };
        }
        Self { values: out }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_flips_only_signed_features() {
        let fv = FeatureVector {
            values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, f64::NAN],
        };
        let o = fv.oriented(-1.0);
        assert_eq!(o.values, [-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 0.0]);
    }
}
