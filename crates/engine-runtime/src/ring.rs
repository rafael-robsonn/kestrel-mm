use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use core_types::CachePadded;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RingError {
    #[error("ring capacity must be a power of two >= 2, got {0}")]
    InvalidCapacity(usize),
}

struct Shared<T> {
    /// Next slot the consumer will read. Written only by the consumer.
    head: CachePadded<AtomicUsize>,
    /// Next slot the producer will write. Written only by the producer.
    tail: CachePadded<AtomicUsize>,
    closed: CachePadded<AtomicBool>,
    slots: Box<[UnsafeCell<MaybeUninit<T>>]>,
    mask: usize,
}

// SAFETY: slot access is partitioned by the head/tail protocol: the producer writes only
// slots in `[tail, head + cap)`, the consumer reads only slots in `[head, tail)`, and every
// hand-over is ordered by a Release store / Acquire load pair on `tail` or `head`.
unsafe impl<T: Send> Sync for Shared<T> {}
// SAFETY: `Shared<T>` owns `T` values; moving it between threads is sound when `T: Send`.
unsafe impl<T: Send> Send for Shared<T> {}

/// Creates a bounded wait-free single-producer single-consumer queue.
///
/// Capacity must be a power of two. Head and tail live on separate cache lines; each side
/// caches the other's cursor and only reloads it (one Acquire load) when the cached value
/// says the ring is full/empty, so the steady state touches one shared line per operation.
/// `T: Copy` means slots never need dropping.
pub fn spsc<T: Copy + Send>(capacity: usize) -> Result<(Producer<T>, Consumer<T>), RingError> {
    if capacity < 2 || !capacity.is_power_of_two() {
        return Err(RingError::InvalidCapacity(capacity));
    }
    let slots: Box<[UnsafeCell<MaybeUninit<T>>]> =
        (0..capacity).map(|_| UnsafeCell::new(MaybeUninit::uninit())).collect();
    let shared = Arc::new(Shared {
        head: CachePadded::new(AtomicUsize::new(0)),
        tail: CachePadded::new(AtomicUsize::new(0)),
        closed: CachePadded::new(AtomicBool::new(false)),
        slots,
        mask: capacity - 1,
    });
    Ok((
        Producer {
            shared: Arc::clone(&shared),
            tail: 0,
            cached_head: 0,
            capacity,
        },
        Consumer {
            shared,
            head: 0,
            cached_tail: 0,
        },
    ))
}

pub struct Producer<T> {
    shared: Arc<Shared<T>>,
    tail: usize,
    cached_head: usize,
    capacity: usize,
}

pub struct Consumer<T> {
    shared: Arc<Shared<T>>,
    head: usize,
    cached_tail: usize,
}

impl<T: Copy + Send> Producer<T> {
    /// Wait-free. Returns the value back if the ring is full.
    #[inline]
    pub fn push(&mut self, value: T) -> Result<(), T> {
        let tail = self.tail;
        if tail.wrapping_sub(self.cached_head) == self.capacity {
            self.cached_head = self.shared.head.load(Ordering::Acquire);
            if tail.wrapping_sub(self.cached_head) == self.capacity {
                return Err(value);
            }
        }
        let slot = &self.shared.slots[tail & self.shared.mask];
        // SAFETY: `tail - head < capacity`, so the consumer has released this slot (its
        // Release store of `head` happened-before our Acquire load) and will not read it
        // until we publish `tail + 1` below. We are the only producer (`&mut self`).
        unsafe { (*slot.get()).write(value) };
        self.tail = tail.wrapping_add(1);
        self.shared.tail.store(self.tail, Ordering::Release);
        Ok(())
    }

    /// Spins with `spin_loop` hints until the value is accepted.
    #[inline]
    pub fn push_spin(&mut self, mut value: T) {
        loop {
            match self.push(value) {
                Ok(()) => return,
                Err(v) => {
                    value = v;
                    core::hint::spin_loop();
                }
            }
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> Drop for Producer<T> {
    fn drop(&mut self) {
        self.shared.closed.store(true, Ordering::Release);
    }
}

impl<T: Copy + Send> Consumer<T> {
    /// Wait-free. `None` if the ring is currently empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        let head = self.head;
        if head == self.cached_tail {
            self.cached_tail = self.shared.tail.load(Ordering::Acquire);
            if head == self.cached_tail {
                return None;
            }
        }
        let slot = &self.shared.slots[head & self.shared.mask];
        // SAFETY: `head < tail`; the producer initialised this slot before its Release store
        // of `tail`, which our Acquire load observed. The producer will not overwrite it
        // until we publish `head + 1`. `T: Copy`, so reading by value leaves no obligation.
        let value = unsafe { (*slot.get()).assume_init_read() };
        self.head = head.wrapping_add(1);
        self.shared.head.store(self.head, Ordering::Release);
        Some(value)
    }

    /// `true` once the producer is dropped and every published item has been consumed.
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.shared.closed.load(Ordering::Acquire) && self.head == self.shared.tail.load(Ordering::Acquire)
    }
}

impl<T> core::fmt::Debug for Producer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Producer")
            .field("tail", &self.tail)
            .field("capacity", &self.capacity)
            .finish()
    }
}

impl<T> core::fmt::Debug for Consumer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Consumer").field("head", &self.head).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_capacity() {
        assert!(spsc::<u64>(0).is_err());
        assert!(spsc::<u64>(3).is_err());
        assert!(spsc::<u64>(4).is_ok());
    }

    #[test]
    fn fifo_with_wraparound_and_backpressure() {
        let (mut tx, mut rx) = spsc::<u64>(4).unwrap();
        for round in 0..10u64 {
            for i in 0..4 {
                tx.push(round * 10 + i).unwrap();
            }
            assert_eq!(tx.push(99), Err(99));
            for i in 0..4 {
                assert_eq!(rx.pop(), Some(round * 10 + i));
            }
            assert_eq!(rx.pop(), None);
        }
        assert!(!rx.is_finished());
        drop(tx);
        assert!(rx.is_finished());
    }

    #[test]
    fn cross_thread_preserves_order() {
        const N: u64 = 2_000_000;
        let (mut tx, mut rx) = spsc::<[u64; 8]>(1024).unwrap();
        let producer = std::thread::spawn(move || {
            for i in 0..N {
                tx.push_spin([i, i ^ 0xFF, i, i, i, i, i, i.wrapping_mul(31)]);
            }
        });
        let mut expected = 0u64;
        while expected < N {
            if let Some(v) = rx.pop() {
                assert_eq!(v[0], expected);
                assert_eq!(v[1], expected ^ 0xFF);
                assert_eq!(v[7], expected.wrapping_mul(31));
                expected += 1;
            } else {
                core::hint::spin_loop();
            }
        }
        producer.join().unwrap();
        assert!(rx.is_finished());
    }
}
