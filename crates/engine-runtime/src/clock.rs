use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Raw timestamp counter. On x86_64 this is `RDTSC` (invariant TSC assumed, ~7 ns,
/// not serialising); elsewhere nanoseconds since first use.
#[inline(always)]
pub fn ticks() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: RDTSC is available on every x86_64 CPU and has no memory side effects.
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        static ORIGIN: OnceLock<Instant> = OnceLock::new();
        ORIGIN.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
}

/// Linear mapping from [`ticks`] deltas to nanoseconds, calibrated against the OS clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TickClock {
    ns_per_tick: f64,
}

impl TickClock {
    /// Busy-waits for `window` to measure the tick rate. Call once at startup.
    pub fn calibrate(window: Duration) -> Self {
        let t0 = Instant::now();
        let c0 = ticks();
        while t0.elapsed() < window {
            core::hint::spin_loop();
        }
        let c1 = ticks();
        let ns = t0.elapsed().as_nanos() as f64;
        let delta = c1.wrapping_sub(c0).max(1) as f64;
        Self {
            ns_per_tick: ns / delta,
        }
    }

    /// Process-wide calibration (20 ms, performed once).
    pub fn global() -> Self {
        static CLOCK: OnceLock<TickClock> = OnceLock::new();
        *CLOCK.get_or_init(|| Self::calibrate(Duration::from_millis(20)))
    }

    #[inline(always)]
    pub fn to_ns(&self, ticks: u64) -> f64 {
        ticks as f64 * self.ns_per_tick
    }

    #[inline(always)]
    pub fn from_ns(&self, ns: f64) -> u64 {
        (ns / self.ns_per_tick) as u64
    }

    pub fn ghz(&self) -> f64 {
        1.0 / self.ns_per_tick
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_is_consistent_with_os_clock() {
        let clock = TickClock::calibrate(Duration::from_millis(50));
        let c0 = ticks();
        let t0 = Instant::now();
        std::thread::sleep(Duration::from_millis(200));
        let measured = clock.to_ns(ticks() - c0);
        let reference = t0.elapsed().as_nanos() as f64;
        assert!((measured / reference - 1.0).abs() < 0.05, "{measured} vs {reference}");
    }
}
