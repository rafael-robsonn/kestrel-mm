use core::fmt;
use core::ops::{Deref, DerefMut};

/// Pads and aligns `T` to a 64-byte cache line so that independently written
/// values (e.g. SPSC head/tail cursors) never share a line.
#[derive(Default)]
#[repr(C, align(64))]
pub struct CachePadded<T> {
    value: T,
}

impl<T> CachePadded<T> {
    pub const fn new(value: T) -> Self {
        Self { value }
    }

    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T> Deref for CachePadded<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> DerefMut for CachePadded<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T: fmt::Debug> fmt::Debug for CachePadded<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CachePadded").field(&self.value).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::AtomicUsize;

    #[test]
    fn occupies_full_line() {
        assert_eq!(core::mem::align_of::<CachePadded<AtomicUsize>>(), 64);
        assert_eq!(core::mem::size_of::<[CachePadded<AtomicUsize>; 2]>(), 128);
    }
}
