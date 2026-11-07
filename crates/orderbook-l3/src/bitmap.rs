/// Three-level occupancy bitmap over `[0, 64^3)` price-level indices.
///
/// `l1` bit `w` is set iff `l0[w] != 0`; `l2` bit `k` is set iff `l1[k] != 0`.
/// `set`, `clear`, `min`, `max`, `next_above`, `next_below` are O(1): at most three
/// word scans using `lzcnt`/`tzcnt`.
#[derive(Debug)]
pub(crate) struct LevelBitmap {
    l0: Box<[u64]>,
    l1: Box<[u64]>,
    l2: u64,
}

pub(crate) const MAX_BITMAP_LEVELS: usize = 64 * 64 * 64;

#[inline(always)]
const fn mask_below(bit: usize) -> u64 {
    (1u64 << bit) - 1
}

#[inline(always)]
const fn mask_above(bit: usize) -> u64 {
    if bit == 63 { 0 } else { u64::MAX << (bit + 1) }
}

#[inline(always)]
const fn high_bit(word: u64) -> usize {
    63 - word.leading_zeros() as usize
}

#[inline(always)]
const fn low_bit(word: u64) -> usize {
    word.trailing_zeros() as usize
}

impl LevelBitmap {
    pub(crate) fn new(capacity: usize) -> Self {
        let l0_words = capacity.div_ceil(64).max(1);
        let l1_words = l0_words.div_ceil(64).max(1);
        Self {
            l0: vec![0u64; l0_words].into_boxed_slice(),
            l1: vec![0u64; l1_words].into_boxed_slice(),
            l2: 0,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.l0.fill(0);
        self.l1.fill(0);
        self.l2 = 0;
    }

    #[inline(always)]
    pub(crate) fn contains(&self, i: usize) -> bool {
        self.l0[i >> 6] & (1u64 << (i & 63)) != 0
    }

    #[inline(always)]
    pub(crate) fn set(&mut self, i: usize) {
        let w = i >> 6;
        let k = w >> 6;
        self.l0[w] |= 1u64 << (i & 63);
        self.l1[k] |= 1u64 << (w & 63);
        self.l2 |= 1u64 << k;
    }

    #[inline(always)]
    pub(crate) fn clear(&mut self, i: usize) {
        let w = i >> 6;
        let k = w >> 6;
        self.l0[w] &= !(1u64 << (i & 63));
        if self.l0[w] == 0 {
            self.l1[k] &= !(1u64 << (w & 63));
            if self.l1[k] == 0 {
                self.l2 &= !(1u64 << k);
            }
        }
    }

    #[inline(always)]
    fn max_in_l1(&self, k: usize) -> usize {
        let w = (k << 6) | high_bit(self.l1[k]);
        (w << 6) | high_bit(self.l0[w])
    }

    #[inline(always)]
    fn min_in_l1(&self, k: usize) -> usize {
        let w = (k << 6) | low_bit(self.l1[k]);
        (w << 6) | low_bit(self.l0[w])
    }

    #[inline(always)]
    pub(crate) fn max(&self) -> Option<usize> {
        if self.l2 == 0 {
            None
        } else {
            Some(self.max_in_l1(high_bit(self.l2)))
        }
    }

    #[inline(always)]
    pub(crate) fn min(&self) -> Option<usize> {
        if self.l2 == 0 {
            None
        } else {
            Some(self.min_in_l1(low_bit(self.l2)))
        }
    }

    /// Highest set index strictly below `i`.
    #[inline]
    pub(crate) fn next_below(&self, i: usize) -> Option<usize> {
        let w = i >> 6;
        let m0 = self.l0[w] & mask_below(i & 63);
        if m0 != 0 {
            return Some((w << 6) | high_bit(m0));
        }
        let k = w >> 6;
        let m1 = self.l1[k] & mask_below(w & 63);
        if m1 != 0 {
            let w2 = (k << 6) | high_bit(m1);
            return Some((w2 << 6) | high_bit(self.l0[w2]));
        }
        let m2 = self.l2 & mask_below(k);
        if m2 != 0 {
            Some(self.max_in_l1(high_bit(m2)))
        } else {
            None
        }
    }

    /// Lowest set index strictly above `i`.
    #[inline]
    pub(crate) fn next_above(&self, i: usize) -> Option<usize> {
        let w = i >> 6;
        let m0 = self.l0[w] & mask_above(i & 63);
        if m0 != 0 {
            return Some((w << 6) | low_bit(m0));
        }
        let k = w >> 6;
        let m1 = self.l1[k] & mask_above(w & 63);
        if m1 != 0 {
            let w2 = (k << 6) | low_bit(m1);
            return Some((w2 << 6) | low_bit(self.l0[w2]));
        }
        let m2 = self.l2 & mask_above(k);
        if m2 != 0 {
            Some(self.min_in_l1(low_bit(m2)))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::collection::btree_set;
    use proptest::prelude::*;

    #[test]
    fn empty_bitmap() {
        let b = LevelBitmap::new(MAX_BITMAP_LEVELS);
        assert_eq!(b.max(), None);
        assert_eq!(b.min(), None);
        assert_eq!(b.next_above(0), None);
        assert_eq!(b.next_below(MAX_BITMAP_LEVELS - 1), None);
    }

    #[test]
    fn crosses_word_boundaries() {
        let mut b = LevelBitmap::new(MAX_BITMAP_LEVELS);
        for i in [0, 63, 64, 4095, 4096, MAX_BITMAP_LEVELS - 1] {
            b.set(i);
        }
        assert_eq!(b.min(), Some(0));
        assert_eq!(b.max(), Some(MAX_BITMAP_LEVELS - 1));
        assert_eq!(b.next_above(0), Some(63));
        assert_eq!(b.next_above(64), Some(4095));
        assert_eq!(b.next_above(4096), Some(MAX_BITMAP_LEVELS - 1));
        assert_eq!(b.next_below(4096), Some(4095));
        assert_eq!(b.next_below(64), Some(63));
        b.clear(MAX_BITMAP_LEVELS - 1);
        assert_eq!(b.max(), Some(4096));
        b.clear(0);
        assert_eq!(b.min(), Some(63));
    }

    proptest! {
        #[test]
        fn matches_btreeset(
            items in btree_set(0usize..20_000, 0..200),
            removals in btree_set(0usize..20_000, 0..100),
            probe in 0usize..20_000,
        ) {
            let mut b = LevelBitmap::new(20_000);
            let mut model = items.clone();
            for &i in &items { b.set(i); }
            for &i in &removals {
                if model.remove(&i) { b.clear(i); }
            }
            prop_assert_eq!(b.min(), model.first().copied());
            prop_assert_eq!(b.max(), model.last().copied());
            prop_assert_eq!(b.next_above(probe), model.range(probe + 1..).next().copied());
            prop_assert_eq!(b.next_below(probe), model.range(..probe).next_back().copied());
            prop_assert_eq!(b.contains(probe), model.contains(&probe));
        }
    }
}
