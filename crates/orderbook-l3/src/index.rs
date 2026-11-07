use core_types::OrderId;

const EMPTY: u64 = u64::MAX;
const FIBONACCI: u64 = 0x9E37_79B9_7F4A_7C15;

#[derive(Clone, Copy, Debug)]
struct Slot {
    key: u64,
    val: u32,
}

const VACANT: Slot = Slot { key: EMPTY, val: 0 };

/// Fixed-capacity `OrderId -> arena slot` map: open addressing, linear probing,
/// Fibonacci hashing, backward-shift deletion (no tombstones).
///
/// Capacity is a power of two sized to at least `2 * max_orders`, so load factor
/// stays `<= 0.5` and expected probe length is O(1). `u64::MAX` is reserved.
#[derive(Debug)]
pub(crate) struct OrderIndex {
    slots: Box<[Slot]>,
    mask: usize,
    shift: u32,
}

impl OrderIndex {
    pub(crate) fn with_max_entries(max_entries: usize) -> Self {
        let cap = (max_entries.max(1) * 2).next_power_of_two().max(2);
        Self {
            slots: vec![VACANT; cap].into_boxed_slice(),
            mask: cap - 1,
            shift: 64 - cap.trailing_zeros(),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.slots.fill(VACANT);
    }

    #[inline(always)]
    fn home(&self, key: u64) -> usize {
        (key.wrapping_mul(FIBONACCI) >> self.shift) as usize
    }

    #[inline]
    pub(crate) fn get(&self, id: OrderId) -> Option<u32> {
        let mut i = self.home(id.0);
        loop {
            let s = self.slots[i];
            if s.key == id.0 {
                return Some(s.val);
            }
            if s.key == EMPTY {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }

    /// Caller guarantees `id` is absent, not `OrderId::NONE`, and the table is below
    /// its sizing bound.
    #[inline]
    pub(crate) fn insert(&mut self, id: OrderId, val: u32) {
        let mut i = self.home(id.0);
        while self.slots[i].key != EMPTY {
            i = (i + 1) & self.mask;
        }
        self.slots[i] = Slot { key: id.0, val };
    }

    #[inline]
    pub(crate) fn remove(&mut self, id: OrderId) -> Option<u32> {
        let mut i = self.home(id.0);
        loop {
            let s = self.slots[i];
            if s.key == id.0 {
                break;
            }
            if s.key == EMPTY {
                return None;
            }
            i = (i + 1) & self.mask;
        }
        let removed = self.slots[i].val;
        let mut hole = i;
        let mut j = i;
        loop {
            j = (j + 1) & self.mask;
            let s = self.slots[j];
            if s.key == EMPTY {
                break;
            }
            let h = self.home(s.key);
            let displaced_past_hole = if hole <= j {
                h <= hole || h > j
            } else {
                h <= hole && h > j
            };
            if displaced_past_hole {
                self.slots[hole] = s;
                hole = j;
            }
        }
        self.slots[hole] = VACANT;
        Some(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashMap;

    #[test]
    fn insert_get_remove() {
        let mut ix = OrderIndex::with_max_entries(4);
        ix.insert(OrderId(10), 1);
        ix.insert(OrderId(20), 2);
        assert_eq!(ix.get(OrderId(10)), Some(1));
        assert_eq!(ix.remove(OrderId(10)), Some(1));
        assert_eq!(ix.get(OrderId(10)), None);
        assert_eq!(ix.get(OrderId(20)), Some(2));
        assert_eq!(ix.remove(OrderId(10)), None);
    }

    proptest! {
        #[test]
        fn agrees_with_hashmap(ops in proptest::collection::vec((any::<bool>(), 0u64..64), 1..400)) {
            let mut ix = OrderIndex::with_max_entries(64);
            let mut model: HashMap<u64, u32> = HashMap::new();
            for (n, (insert, key)) in ops.into_iter().enumerate() {
                if insert {
                    if let std::collections::hash_map::Entry::Vacant(e) = model.entry(key) {
                        e.insert(n as u32);
                        ix.insert(OrderId(key), n as u32);
                    }
                } else {
                    prop_assert_eq!(ix.remove(OrderId(key)), model.remove(&key));
                }
                for k in 0..64u64 {
                    prop_assert_eq!(ix.get(OrderId(k)), model.get(&k).copied());
                }
            }
        }
    }
}
