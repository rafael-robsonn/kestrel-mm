use core::fmt;

use crate::CoreError;

/// Fixed-point scale shared by [`Price`] and [`Qty`]: one unit equals `10^8` raw.
pub const SCALE: i64 = 100_000_000;
const SCALE_F64: f64 = SCALE as f64;

macro_rules! fixed_point {
    ($name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        #[repr(transparent)]
        pub struct $name(i64);

        impl $name {
            pub const ZERO: Self = Self(0);
            pub const MIN: Self = Self(i64::MIN);
            pub const MAX: Self = Self(i64::MAX);
            /// Smallest representable increment (`10^-8` units).
            pub const EPSILON: Self = Self(1);

            #[inline(always)]
            pub const fn from_raw(raw: i64) -> Self {
                Self(raw)
            }

            #[inline(always)]
            pub const fn raw(self) -> i64 {
                self.0
            }

            /// Exact conversion from whole units. Fails if `units * 10^8` overflows `i64`.
            #[inline]
            pub const fn from_units(units: i64) -> Result<Self, CoreError> {
                match units.checked_mul(SCALE) {
                    Some(raw) => Ok(Self(raw)),
                    None => Err(CoreError::Overflow),
                }
            }

            /// Round-half-away-from-zero conversion. Rejects NaN/inf and values outside
            /// `[i64::MIN, i64::MAX] / 10^8`.
            #[inline]
            pub fn from_f64(value: f64) -> Result<Self, CoreError> {
                if !value.is_finite() {
                    return Err(CoreError::NonFinite);
                }
                let scaled = (value * SCALE_F64).round();
                if scaled >= i64::MAX as f64 || scaled < i64::MIN as f64 {
                    return Err(CoreError::Overflow);
                }
                Ok(Self(scaled as i64))
            }

            /// Lossy above `2^53` raw (≈ 9.0e7 units).
            #[inline(always)]
            pub fn to_f64(self) -> f64 {
                self.0 as f64 / SCALE_F64
            }

            #[inline(always)]
            pub const fn checked_add(self, rhs: Self) -> Option<Self> {
                match self.0.checked_add(rhs.0) {
                    Some(v) => Some(Self(v)),
                    None => None,
                }
            }

            #[inline(always)]
            pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
                match self.0.checked_sub(rhs.0) {
                    Some(v) => Some(Self(v)),
                    None => None,
                }
            }

            #[inline(always)]
            pub const fn saturating_add(self, rhs: Self) -> Self {
                Self(self.0.saturating_add(rhs.0))
            }

            #[inline(always)]
            pub const fn saturating_sub(self, rhs: Self) -> Self {
                Self(self.0.saturating_sub(rhs.0))
            }

            #[inline(always)]
            pub const fn is_positive(self) -> bool {
                self.0 > 0
            }

            #[inline(always)]
            pub const fn is_zero(self) -> bool {
                self.0 == 0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let sign = if self.0 < 0 { "-" } else { "" };
                let abs = self.0.unsigned_abs();
                let scale = SCALE.unsigned_abs();
                write!(f, "{sign}{}.{:08}", abs / scale, abs % scale)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self)
            }
        }
    };
}

fixed_point!(Price);
fixed_point!(Qty);

impl Price {
    /// `true` iff `self` is an integer multiple of `tick`. `tick` must be positive.
    #[inline(always)]
    pub const fn is_aligned(self, tick: Price) -> bool {
        tick.0 > 0 && self.0 % tick.0 == 0
    }

    /// Floor division by `tick`. `tick` must be positive.
    #[inline(always)]
    pub const fn to_ticks(self, tick: Price) -> i64 {
        self.0.div_euclid(tick.0)
    }

    #[inline]
    pub const fn from_ticks(ticks: i64, tick: Price) -> Result<Self, CoreError> {
        if tick.0 <= 0 {
            return Err(CoreError::InvalidTick);
        }
        match ticks.checked_mul(tick.0) {
            Some(raw) => Ok(Self(raw)),
            None => Err(CoreError::Overflow),
        }
    }
}

impl Qty {
    #[inline(always)]
    pub const fn min(self, rhs: Self) -> Self {
        if self.0 <= rhs.0 { self } else { rhs }
    }
}

/// `price * qty` in raw `10^-8` currency units, computed in `i128` (cannot overflow).
#[inline(always)]
pub const fn notional(price: Price, qty: Qty) -> i128 {
    (price.0 as i128 * qty.0 as i128) / SCALE as i128
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_f64() {
        let p = Price::from_f64(101.23456789).unwrap();
        assert_eq!(p.raw(), 10_123_456_789);
        assert!((p.to_f64() - 101.23456789).abs() < 1e-12);
        assert_eq!(Price::from_f64(f64::NAN), Err(CoreError::NonFinite));
        assert_eq!(Price::from_f64(1e12), Err(CoreError::Overflow));
    }

    #[test]
    fn display_is_exact() {
        assert_eq!(Price::from_raw(-150_000_001).to_string(), "-1.50000001");
        assert_eq!(Qty::from_units(7).unwrap().to_string(), "7.00000000");
    }

    #[test]
    fn tick_arithmetic() {
        let tick = Price::from_f64(0.01).unwrap();
        let p = Price::from_f64(100.05).unwrap();
        assert!(p.is_aligned(tick));
        assert_eq!(p.to_ticks(tick), 10_005);
        assert_eq!(Price::from_ticks(10_005, tick), Ok(p));
        assert!(!Price::from_f64(100.055).unwrap().is_aligned(tick));
        assert_eq!(Price::from_ticks(1, Price::ZERO), Err(CoreError::InvalidTick));
    }

    #[test]
    fn notional_uses_wide_intermediate() {
        let p = Price::from_units(50_000).unwrap();
        let q = Qty::from_units(1_000_000).unwrap();
        assert_eq!(notional(p, q), 50_000_000_000i128 * SCALE as i128);
    }
}
