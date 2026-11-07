use core_types::{DEPTH_LEVELS, DepthSnapshot, SCALE};
use wide::bytemuck::cast;
use wide::{f64x2, f64x4, f64x8};

const _: () = assert!(DEPTH_LEVELS == 8, "ladder kernels assume one f64x8 per side");

/// Depth snapshot converted to SIMD lanes: prices in ticks, quantities in units.
/// Absent levels keep the snapshot sentinels (finite, extreme prices with zero quantity),
/// so every kernel below is branch-free over all eight lanes.
#[derive(Clone, Copy, Debug)]
pub struct LadderLanes {
    pub bid_px: f64x8,
    pub bid_qty: f64x8,
    pub ask_px: f64x8,
    pub ask_qty: f64x8,
}

#[inline(always)]
fn lanes(src: &[i64; DEPTH_LEVELS], scale: f64) -> f64x8 {
    let mut out = [0.0f64; DEPTH_LEVELS];
    for (o, &s) in out.iter_mut().zip(src.iter()) {
        *o = s as f64 * scale;
    }
    f64x8::new(out)
}

impl LadderLanes {
    #[inline(always)]
    pub fn from_snapshot(snap: &DepthSnapshot, inv_tick_raw: f64) -> Self {
        let inv_scale = 1.0 / SCALE as f64;
        Self {
            bid_px: lanes(&snap.bid_px, inv_tick_raw),
            bid_qty: lanes(&snap.bid_qty, inv_scale),
            ask_px: lanes(&snap.ask_px, inv_tick_raw),
            ask_qty: lanes(&snap.ask_qty, inv_scale),
        }
    }
}

/// Per-level order flow imbalance between consecutive snapshots
/// (Cont, Kukanov, Stoikov 2014; multi-level form of Xu, Gould, Howison 2019):
///
/// `e^b_m = 1{P^b_m ≥ P^b'_m} q^b_m − 1{P^b_m ≤ P^b'_m} q^b'_m`
/// `e^a_m = 1{P^a_m ≤ P^a'_m} q^a_m − 1{P^a_m ≥ P^a'_m} q^a'_m`
/// `OFI_m = e^b_m − e^a_m`
///
/// Primes denote the previous snapshot. Branch-free: four compares, four masked
/// selects, three subtractions on 8 lanes.
#[inline(always)]
pub fn ofi_levels(cur: &LadderLanes, prev: &LadderLanes) -> f64x8 {
    let bid_up = cur.bid_px.simd_ge(prev.bid_px);
    let bid_dn = cur.bid_px.simd_le(prev.bid_px);
    let ask_dn = cur.ask_px.simd_le(prev.ask_px);
    let ask_up = cur.ask_px.simd_ge(prev.ask_px);
    let e_bid = (bid_up & cur.bid_qty) - (bid_dn & prev.bid_qty);
    let e_ask = (ask_dn & cur.ask_qty) - (ask_up & prev.ask_qty);
    e_bid - e_ask
}

/// Horizontal sum as a fixed reduction tree, `((a₀+a₄)+(a₂+a₆)) + ((a₁+a₅)+(a₃+a₇))`, built
/// from lane-wise vector adds. Hardware horizontal reductions associate differently per
/// target feature set; a fixed tree keeps every feature bit-identical across `target-cpu`
/// builds of the same platform.
#[inline(always)]
pub fn hsum(v: f64x8) -> f64 {
    let [lo, hi]: [f64x4; 2] = cast(v);
    let [l, h]: [f64x2; 2] = cast(lo + hi);
    let p = (l + h).to_array();
    p[0] + p[1]
}

/// Output of [`weighted_micro_price`]; all prices in ticks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MicroPrice {
    pub mid: f64,
    pub micro: f64,
    pub imbalance: f64,
    pub spread: f64,
}

/// Depth-weighted micro-price over the 8-level ladder with level weights `w`:
///
/// `D_b = Σ w_m q^b_m`, `V_b = Σ w_m q^b_m P^b_m / D_b` (likewise ask)
/// `micro = (V_a D_b + V_b D_a) / (D_a + D_b)`, clamped to `[P^b_1, P^a_1]`
/// `imbalance = (D_b − D_a) / (D_b + D_a)`
///
/// Requires a two-sided ladder. Four FMA chains plus two horizontal reductions.
#[inline(always)]
pub fn weighted_micro_price(l: &LadderLanes, w: f64x8) -> MicroPrice {
    let wb = w * l.bid_qty;
    let wa = w * l.ask_qty;
    let d_b = hsum(wb);
    let d_a = hsum(wa);
    let n_b = hsum(wb * l.bid_px);
    let n_a = hsum(wa * l.ask_px);
    let best_bid = l.bid_px.as_array()[0];
    let best_ask = l.ask_px.as_array()[0];
    let mid = 0.5 * (best_bid + best_ask);
    let depth = d_a + d_b;
    let (micro, imbalance) = if depth > 0.0 && d_a > 0.0 && d_b > 0.0 {
        let v_b = n_b / d_b;
        let v_a = n_a / d_a;
        (
            ((v_a * d_b + v_b * d_a) / depth).clamp(best_bid, best_ask),
            (d_b - d_a) / depth,
        )
    } else {
        (mid, 0.0)
    };
    MicroPrice {
        mid,
        micro,
        imbalance,
        spread: best_ask - best_bid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ABSENT_ASK_PX, ABSENT_BID_PX};

    const TICK_RAW: i64 = 1_000_000;

    fn snap(bids: &[(i64, i64)], asks: &[(i64, i64)]) -> DepthSnapshot {
        let mut s = DepthSnapshot::EMPTY;
        for (i, &(p, q)) in bids.iter().enumerate() {
            s.bid_px[i] = p * TICK_RAW;
            s.bid_qty[i] = q * SCALE;
        }
        for (i, &(p, q)) in asks.iter().enumerate() {
            s.ask_px[i] = p * TICK_RAW;
            s.ask_qty[i] = q * SCALE;
        }
        s.bid_levels = bids.len() as u8;
        s.ask_levels = asks.len() as u8;
        s
    }

    fn lanes_of(s: &DepthSnapshot) -> LadderLanes {
        LadderLanes::from_snapshot(s, 1.0 / TICK_RAW as f64)
    }

    fn scalar_ofi(cur: &DepthSnapshot, prev: &DepthSnapshot) -> [f64; 8] {
        let mut out = [0.0; 8];
        let f = |x: i64| x as f64 / SCALE as f64;
        for m in 0..8 {
            let (pb, qb, pb0, qb0) = (cur.bid_px[m], f(cur.bid_qty[m]), prev.bid_px[m], f(prev.bid_qty[m]));
            let (pa, qa, pa0, qa0) = (cur.ask_px[m], f(cur.ask_qty[m]), prev.ask_px[m], f(prev.ask_qty[m]));
            let eb = if pb >= pb0 { qb } else { 0.0 } - if pb <= pb0 { qb0 } else { 0.0 };
            let ea = if pa <= pa0 { qa } else { 0.0 } - if pa >= pa0 { qa0 } else { 0.0 };
            out[m] = eb - ea;
        }
        out
    }

    #[test]
    fn ofi_matches_scalar_definition() {
        let prev = snap(&[(100, 5), (99, 7), (98, 2)], &[(101, 4), (102, 3)]);
        let cases = [
            snap(&[(100, 9), (99, 7), (98, 2)], &[(101, 4), (102, 3)]),
            snap(&[(101, 1), (100, 5), (99, 7)], &[(102, 3)]),
            snap(&[(99, 7), (98, 2)], &[(100, 6), (101, 4), (102, 3)]),
            snap(&[], &[]),
        ];
        for cur in &cases {
            let got = ofi_levels(&lanes_of(cur), &lanes_of(&prev)).to_array();
            let want = scalar_ofi(cur, &prev);
            for m in 0..8 {
                assert!((got[m] - want[m]).abs() < 1e-9, "level {m}: {} vs {}", got[m], want[m]);
            }
        }
    }

    #[test]
    fn ofi_signs() {
        let prev = snap(&[(100, 5)], &[(101, 5)]);
        let bid_add = ofi_levels(&lanes_of(&snap(&[(100, 8)], &[(101, 5)])), &lanes_of(&prev)).to_array();
        assert_eq!(bid_add[0], 3.0);
        let ask_lift = ofi_levels(&lanes_of(&snap(&[(100, 5)], &[(102, 2)])), &lanes_of(&prev)).to_array();
        assert_eq!(ask_lift[0], 5.0);
        let bid_hit = ofi_levels(&lanes_of(&snap(&[(99, 4)], &[(101, 5)])), &lanes_of(&prev)).to_array();
        assert_eq!(bid_hit[0], -5.0);
        assert_eq!(bid_hit[1], 0.0);
    }

    #[test]
    fn sentinels_are_finite_and_inert() {
        let l = lanes_of(&snap(&[(100, 1)], &[(101, 1)]));
        assert!(l.bid_px.to_array().iter().all(|v| v.is_finite()));
        assert_eq!(l.bid_px.to_array()[1], ABSENT_BID_PX as f64 / TICK_RAW as f64);
        assert_eq!(l.ask_px.to_array()[7], ABSENT_ASK_PX as f64 / TICK_RAW as f64);
        let m = weighted_micro_price(&l, f64x8::splat(1.0));
        assert_eq!(m.mid, 100.5);
        assert_eq!(m.micro, 100.5);
    }

    #[test]
    fn micro_leans_toward_thin_side() {
        let l = lanes_of(&snap(&[(100, 9)], &[(101, 1)]));
        let m = weighted_micro_price(&l, f64x8::splat(1.0));
        assert!((m.micro - 100.9).abs() < 1e-9);
        assert!((m.imbalance - 0.8).abs() < 1e-12);
        assert_eq!(m.spread, 1.0);

        let deep = lanes_of(&snap(&[(100, 1), (99, 50)], &[(101, 1), (102, 1)]));
        let m = weighted_micro_price(&deep, f64x8::splat(1.0));
        assert!(m.micro <= 101.0 && m.micro >= 100.0);
        assert!(m.imbalance > 0.9);
    }
}
