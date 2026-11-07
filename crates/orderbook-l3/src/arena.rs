use core_types::{OrderId, Side};

pub(crate) const NIL: u32 = u32::MAX;

/// Resting order stored in the arena. `prev`/`next` form the intrusive FIFO list of
/// its price level; while the slot is free, `next` links the free list.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OrderNode {
    pub id: OrderId,
    pub qty: i64,
    pub level: u32,
    pub prev: u32,
    pub next: u32,
    pub side: Side,
}

impl OrderNode {
    const VACANT: Self = Self {
        id: OrderId::NONE,
        qty: 0,
        level: NIL,
        prev: NIL,
        next: NIL,
        side: Side::Bid,
    };
}

const _: () = assert!(core::mem::size_of::<OrderNode>() == 32);

/// Fixed-capacity slab of [`OrderNode`] with an intrusive LIFO free list.
/// `alloc`/`free` are O(1) and never touch the global allocator after construction.
#[derive(Debug)]
pub(crate) struct OrderArena {
    nodes: Box<[OrderNode]>,
    free_head: u32,
    live: u32,
}

impl OrderArena {
    pub(crate) fn new(capacity: u32) -> Self {
        let mut arena = Self {
            nodes: vec![OrderNode::VACANT; capacity as usize].into_boxed_slice(),
            free_head: NIL,
            live: 0,
        };
        arena.reset();
        arena
    }

    pub(crate) fn reset(&mut self) {
        let n = self.nodes.len();
        for (i, node) in self.nodes.iter_mut().enumerate() {
            *node = OrderNode::VACANT;
            node.next = if i + 1 < n { (i + 1) as u32 } else { NIL };
        }
        self.free_head = if n == 0 { NIL } else { 0 };
        self.live = 0;
    }

    #[inline(always)]
    pub(crate) fn has_vacancy(&self) -> bool {
        self.free_head != NIL
    }

    #[inline(always)]
    pub(crate) fn live(&self) -> u32 {
        self.live
    }

    #[inline(always)]
    pub(crate) fn capacity(&self) -> usize {
        self.nodes.len()
    }

    #[inline(always)]
    pub(crate) fn alloc(&mut self, node: OrderNode) -> Option<u32> {
        let slot = self.free_head;
        if slot == NIL {
            return None;
        }
        let entry = &mut self.nodes[slot as usize];
        self.free_head = entry.next;
        *entry = node;
        self.live += 1;
        Some(slot)
    }

    #[inline(always)]
    pub(crate) fn free(&mut self, slot: u32) {
        let entry = &mut self.nodes[slot as usize];
        *entry = OrderNode::VACANT;
        entry.next = self.free_head;
        self.free_head = slot;
        self.live -= 1;
    }

    #[inline(always)]
    pub(crate) fn get(&self, slot: u32) -> &OrderNode {
        &self.nodes[slot as usize]
    }

    #[inline(always)]
    pub(crate) fn get_mut(&mut self, slot: u32) -> &mut OrderNode {
        &mut self.nodes[slot as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhausts_and_recycles() {
        let mut a = OrderArena::new(3);
        let node = OrderNode {
            id: OrderId(1),
            qty: 5,
            ..OrderNode::VACANT
        };
        let s0 = a.alloc(node).unwrap();
        let s1 = a.alloc(node).unwrap();
        let s2 = a.alloc(node).unwrap();
        assert!(a.alloc(node).is_none());
        assert_eq!(a.live(), 3);
        a.free(s1);
        assert_eq!(a.alloc(node), Some(s1));
        a.free(s0);
        a.free(s2);
        assert_eq!(a.live(), 1);
    }
}
