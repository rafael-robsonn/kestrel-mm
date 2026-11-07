use core::fmt;
use std::io::Write;

use core_types::{Quote, Timestamp};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RecordKind {
    Quote,
    NoQuote,
    Rejected,
}

/// Per-event record handed from the strategy thread to the telemetry thread.
/// Two cache lines; `Copy`; written into a pre-allocated SPSC slot.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct TelemetryRecord {
    pub seq: u64,
    pub exch_ts: Timestamp,
    /// Ingest thread, before decoding the frame.
    pub recv_tsc: u64,
    /// Strategy thread, after popping the event.
    pub dequeue_tsc: u64,
    /// Strategy thread, after the quote is computed.
    pub done_tsc: u64,
    pub inventory_lots: i64,
    pub tox_bid: f32,
    pub tox_ask: f32,
    pub kind: RecordKind,
    pub quote: Quote,
}

const _: () = assert!(core::mem::size_of::<TelemetryRecord>() == 128);

const LINEAR: usize = 128;
const SUB: usize = 64;
const BUCKETS: usize = LINEAR + (64 - 7) * SUB;

/// Log-linear histogram over `u64` nanoseconds: exact below 128, then 64 sub-buckets per
/// octave (relative error < 1.6%). Fixed 3 776-entry array; `record` is O(1).
#[derive(Clone)]
pub struct LatencyHistogram {
    counts: Box<[u64; BUCKETS]>,
    total: u64,
    sum: f64,
    min: u64,
    max: u64,
}

impl fmt::Debug for LatencyHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LatencyHistogram")
            .field("total", &self.total)
            .field("summary", &self.summary())
            .finish()
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn bucket(v: u64) -> usize {
    if v < LINEAR as u64 {
        v as usize
    } else {
        let m = 63 - v.leading_zeros() as usize;
        LINEAR + (m - 7) * SUB + ((v >> (m - 6)) as usize - SUB)
    }
}

fn bucket_mid(i: usize) -> u64 {
    if i < LINEAR {
        i as u64
    } else {
        let m = (i - LINEAR) / SUB + 7;
        let sub = ((i - LINEAR) % SUB) as u64;
        let width = 1u64 << (m - 6);
        (SUB as u64 + sub) * width + width / 2
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LatencySummary {
    pub count: u64,
    pub mean: f64,
    pub min: u64,
    pub p50: u64,
    pub p90: u64,
    pub p99: u64,
    pub p999: u64,
    pub max: u64,
}

impl fmt::Display for LatencySummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "n={:<9} mean={:>8.1} min={:>6} p50={:>6} p90={:>6} p99={:>7} p99.9={:>7} max={:>9}",
            self.count, self.mean, self.min, self.p50, self.p90, self.p99, self.p999, self.max
        )
    }
}

impl LatencyHistogram {
    pub fn new() -> Self {
        Self {
            counts: Box::new([0; BUCKETS]),
            total: 0,
            sum: 0.0,
            min: u64::MAX,
            max: 0,
        }
    }

    #[inline]
    pub fn record(&mut self, ns: u64) {
        self.counts[bucket(ns)] += 1;
        self.total += 1;
        self.sum += ns as f64;
        self.min = self.min.min(ns);
        self.max = self.max.max(ns);
    }

    pub fn percentile(&self, p: f64) -> u64 {
        if self.total == 0 {
            return 0;
        }
        let target = ((p * self.total as f64).ceil() as u64).clamp(1, self.total);
        let mut seen = 0;
        for (i, &c) in self.counts.iter().enumerate() {
            seen += c;
            if seen >= target {
                return bucket_mid(i).clamp(self.min, self.max);
            }
        }
        self.max
    }

    pub fn summary(&self) -> LatencySummary {
        if self.total == 0 {
            return LatencySummary::default();
        }
        LatencySummary {
            count: self.total,
            mean: self.sum / self.total as f64,
            min: self.min,
            p50: self.percentile(0.50),
            p90: self.percentile(0.90),
            p99: self.percentile(0.99),
            p999: self.percentile(0.999),
            max: self.max,
        }
    }
}

pub(crate) fn write_quote_header(w: &mut impl Write) -> std::io::Result<()> {
    writeln!(
        w,
        "seq,exch_ts_ns,bid_px,bid_qty,ask_px,ask_qty,reservation_px,tox_bid,tox_ask,inventory_lots"
    )
}

pub(crate) fn write_quote_row(w: &mut impl Write, r: &TelemetryRecord) -> std::io::Result<()> {
    let q = &r.quote;
    writeln!(
        w,
        "{},{},{},{},{},{},{},{:.4},{:.4},{}",
        r.seq,
        r.exch_ts.nanos(),
        q.bid_px,
        q.bid_qty,
        q.ask_px,
        q.ask_qty,
        q.reservation_px,
        r.tox_bid,
        r.tox_ask,
        r.inventory_lots
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_are_monotone_and_tight() {
        let mut prev = 0;
        for v in (0..1_000_000u64).step_by(37).chain([u64::MAX / 2, u64::MAX]) {
            let b = bucket(v);
            assert!(b >= prev && b < BUCKETS);
            prev = b;
            if v > 0 {
                let mid = bucket_mid(b) as f64;
                assert!((mid - v as f64).abs() / v as f64 <= 0.016, "v={v} mid={mid}");
            }
        }
    }

    #[test]
    fn percentiles_on_uniform_data() {
        let mut h = LatencyHistogram::new();
        for v in 1..=10_000u64 {
            h.record(v);
        }
        let s = h.summary();
        assert_eq!((s.count, s.min, s.max), (10_000, 1, 10_000));
        assert!((s.p50 as f64 - 5_000.0).abs() < 100.0);
        assert!((s.p99 as f64 - 9_900.0).abs() < 160.0);
        assert!((s.mean - 5_000.5).abs() < 1e-9);
    }
}
