use core::fmt;

/// Exchange-assigned order reference. `u64::MAX` is reserved as [`OrderId::NONE`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct OrderId(pub u64);

impl OrderId {
    pub const NONE: Self = Self(u64::MAX);

    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == u64::MAX
    }
}

impl fmt::Debug for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            f.write_str("OrderId(NONE)")
        } else {
            write!(f, "OrderId({})", self.0)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Side {
    Bid = 0,
    Ask = 1,
}

impl Side {
    #[inline(always)]
    pub const fn opposite(self) -> Self {
        match self {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        }
    }

    /// `+1` for [`Side::Bid`], `-1` for [`Side::Ask`].
    #[inline(always)]
    pub const fn sign(self) -> i64 {
        match self {
            Side::Bid => 1,
            Side::Ask => -1,
        }
    }

    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Nanoseconds. Exchange timestamps are nanoseconds since midnight (ITCH convention).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
#[repr(transparent)]
pub struct Timestamp(pub u64);

impl Timestamp {
    #[inline(always)]
    pub const fn nanos(self) -> u64 {
        self.0
    }

    /// Saturating difference in seconds.
    #[inline(always)]
    pub fn seconds_since(self, earlier: Timestamp) -> f64 {
        self.0.saturating_sub(earlier.0) as f64 * 1e-9
    }
}
