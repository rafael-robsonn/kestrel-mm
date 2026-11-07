use core_types::{FeatureVector, N_FEATURES, SIGNED_FEATURES};
use wide::{f64x8, u64x8};

/// Trees evaluated per SIMD block.
pub const LANES: usize = 8;

const _: () = assert!(N_FEATURES == LANES, "feature vector must fill exactly one f64x8");

const SELL_ORIENTATION: [f64; N_FEATURES] = {
    let mut o = [1.0; N_FEATURES];
    let mut i = 0;
    while i < N_FEATURES {
        if SIGNED_FEATURES[i] {
            o[i] = -1.0;
        }
        i += 1;
    }
    o
};

/// Ensemble of oblivious (symmetric) regression trees in block structure-of-arrays layout.
///
/// Every tree of depth `D` applies one `(feature, threshold)` test per level, so its leaf
/// index is the bit-vector of `D` comparisons:
///
/// `leaf_t(x) = Σ_d [x[f_{t,d}] > θ_{t,d}] · 2^d`, `margin(x) = b + Σ_t w_{t, leaf_t(x)}`
///
/// Trees are packed `LANES` per block. For block `k` and level `d`, `features[k][d]` holds
/// the eight feature indices and `thresholds[k][d]` the eight thresholds, so one level of
/// eight trees is a lane gather of `x`, one compare and one mask-or. The gather is a single
/// `vpermq` when compiled with AVX-512 and eight L1 loads otherwise. A forest of `T` trees
/// costs `⌈T/8⌉·D` such steps plus `T` leaf loads; no data-dependent branches. Padding
/// trees carry zero leaves. NaN inputs compare false. `L` must equal `2^D` (checked at
/// compile time).
#[derive(Debug)]
pub struct ObliviousForest<const B: usize, const D: usize, const L: usize> {
    pub base_score: f64,
    pub features: [[u64x8; D]; B],
    pub thresholds: [[f64x8; D]; B],
    pub leaves: [[[f64; L]; LANES]; B],
}

#[cfg(target_feature = "avx512f")]
#[inline(always)]
fn gather(x: f64x8, indices: u64x8) -> f64x8 {
    x.shuffle(indices)
}

#[cfg(not(target_feature = "avx512f"))]
#[inline(always)]
fn gather(x: f64x8, indices: u64x8) -> f64x8 {
    let lanes = x.as_array();
    let idx = indices.as_array();
    f64x8::new(core::array::from_fn(|k| lanes[idx[k] as usize & (LANES - 1)]))
}

#[inline(always)]
pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

impl<const B: usize, const D: usize, const L: usize> ObliviousForest<B, D, L> {
    const SHAPE: () = assert!(L == 1 << D && D > 0 && D <= 16 && B > 0);

    #[inline(always)]
    fn leaf_indices(&self, block: usize, x: f64x8) -> [u64; LANES] {
        let mut idx = u64x8::ZERO;
        for d in 0..D {
            let xs = gather(x, self.features[block][d]);
            let hit = xs.simd_gt(self.thresholds[block][d]).to_bits();
            idx |= hit & u64x8::splat(1 << d);
        }
        idx.to_array()
    }

    #[inline(always)]
    fn block_sum(&self, block: usize, idx: &[u64; LANES]) -> f64 {
        let leaves = &self.leaves[block];
        let mut a = 0.0;
        let mut b = 0.0;
        for k in (0..LANES).step_by(2) {
            a += leaves[k][idx[k] as usize & (L - 1)];
            b += leaves[k + 1][idx[k + 1] as usize & (L - 1)];
        }
        a + b
    }

    /// Raw additive score (log-odds for a logistic objective).
    #[inline]
    pub fn margin(&self, x: &[f64; N_FEATURES]) -> f64 {
        let () = Self::SHAPE;
        let xv = f64x8::new(*x);
        let mut sum = self.base_score;
        for k in 0..B {
            sum += self.block_sum(k, &self.leaf_indices(k, xv));
        }
        sum
    }

    /// Scores two inputs in one pass, sharing every parameter load.
    #[inline]
    pub fn margin_pair(&self, a: &[f64; N_FEATURES], b: &[f64; N_FEATURES]) -> (f64, f64) {
        self.margin_pair_lanes(f64x8::new(*a), f64x8::new(*b))
    }

    #[inline(always)]
    fn margin_pair_lanes(&self, av: f64x8, bv: f64x8) -> (f64, f64) {
        let () = Self::SHAPE;
        let (mut sa, mut sb) = (self.base_score, self.base_score);
        for k in 0..B {
            sa += self.block_sum(k, &self.leaf_indices(k, av));
            sb += self.block_sum(k, &self.leaf_indices(k, bv));
        }
        (sa, sb)
    }

    /// Scalar tree-by-tree evaluation. Reference implementation for testing the SIMD path.
    pub fn margin_scalar(&self, x: &[f64; N_FEATURES]) -> f64 {
        let mut sum = self.base_score;
        for k in 0..B {
            let feats: Vec<[u64; LANES]> = self.features[k].iter().map(|v| v.to_array()).collect();
            let thrs: Vec<[f64; LANES]> = self.thresholds[k].iter().map(|v| v.to_array()).collect();
            for lane in 0..LANES {
                let mut idx = 0usize;
                for d in 0..D {
                    if x[feats[d][lane] as usize] > thrs[d][lane] {
                        idx |= 1 << d;
                    }
                }
                sum += self.leaves[k][lane][idx];
            }
        }
        sum
    }

    #[inline]
    pub fn probability(&self, x: &[f64; N_FEATURES]) -> f64 {
        sigmoid(self.margin(x))
    }
}

/// Probability that a passive fill on each side is adversely selected within the
/// model horizon.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ToxicityEstimate {
    /// Risk on our bid: the aggressor is a seller.
    pub bid: f64,
    /// Risk on our ask: the aggressor is a buyer.
    pub ask: f64,
}

/// Evaluates a forest trained on aggressor-oriented features for both aggressor
/// directions: buy-oriented scores the ask, sell-oriented scores the bid.
///
/// Equivalent to `sigmoid(margin(features.oriented(±1)))`, computed with one lane mask for
/// non-finite inputs and one lane multiply for the orientation.
#[inline]
pub fn predict_toxicity<const B: usize, const D: usize, const L: usize>(
    forest: &ObliviousForest<B, D, L>,
    features: &FeatureVector,
) -> ToxicityEstimate {
    let raw = f64x8::new(features.values);
    let buy = raw & raw.is_finite();
    let sell = buy * f64x8::new(SELL_ORIENTATION);
    let (m_buy, m_sell) = forest.margin_pair_lanes(buy, sell);
    ToxicityEstimate {
        bid: sigmoid(m_sell),
        ask: sigmoid(m_buy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TOXICITY_FOREST;
    use proptest::prelude::*;

    fn small_forest() -> ObliviousForest<1, 2, 4> {
        let mut leaves = [[0.0; 4]; LANES];
        leaves[0] = [1.0, 2.0, 3.0, 4.0];
        leaves[1] = [-1.0, -2.0, -3.0, -4.0];
        ObliviousForest {
            base_score: 0.5,
            features: [[
                u64x8::new([0, 7, 0, 0, 0, 0, 0, 0]),
                u64x8::new([1, 0, 0, 0, 0, 0, 0, 0]),
            ]],
            thresholds: [[
                f64x8::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                f64x8::new([10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ]],
            leaves: [leaves],
        }
    }

    fn tree_walk(x: &[f64; 8]) -> f64 {
        let t0 = match (x[0] > 0.0, x[1] > 10.0) {
            (false, false) => 1.0,
            (true, false) => 2.0,
            (false, true) => 3.0,
            (true, true) => 4.0,
        };
        let t1 = match (x[7] > 0.0, x[0] > 5.0) {
            (false, false) => -1.0,
            (true, false) => -2.0,
            (false, true) => -3.0,
            (true, true) => -4.0,
        };
        0.5 + t0 + t1
    }

    #[test]
    fn simd_matches_tree_walk() {
        let f = small_forest();
        let inputs = [
            [0.0; 8],
            [1.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [f64::NAN, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
        ];
        for x in &inputs {
            assert_eq!(f.margin(x), tree_walk(x));
            assert_eq!(f.margin_scalar(x), tree_walk(x));
        }
        let (a, b) = f.margin_pair(&inputs[1], &inputs[2]);
        assert_eq!((a, b), (tree_walk(&inputs[1]), tree_walk(&inputs[2])));
    }

    #[test]
    fn orientation_selects_side() {
        let f = small_forest();
        let fv = FeatureVector {
            values: [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        };
        let tox = predict_toxicity(&f, &fv);
        assert!((tox.ask - sigmoid(tree_walk(&fv.oriented(1.0).values))).abs() < 1e-14);
        assert!((tox.bid - sigmoid(tree_walk(&fv.oriented(-1.0).values))).abs() < 1e-14);
        assert!(tox.ask != tox.bid);
    }

    proptest! {
        #[test]
        fn shipped_model_simd_equals_scalar(x in prop::array::uniform8(-20.0f64..20.0)) {
            let simd = TOXICITY_FOREST.margin(&x);
            let scalar = TOXICITY_FOREST.margin_scalar(&x);
            prop_assert!((simd - scalar).abs() < 1e-12, "{} vs {}", simd, scalar);
            let p = TOXICITY_FOREST.probability(&x);
            prop_assert!((0.0..=1.0).contains(&p));
        }

        #[test]
        fn vectorised_prediction_matches_oriented_scalar(
            x in prop::array::uniform8(-20.0f64..20.0),
            poison in 0usize..10,
        ) {
            let mut values = x;
            if poison < 8 {
                values[poison] = if poison % 2 == 0 { f64::NAN } else { f64::INFINITY };
            }
            let fv = FeatureVector { values };
            let tox = predict_toxicity(&TOXICITY_FOREST, &fv);
            let ask = sigmoid(TOXICITY_FOREST.margin_scalar(&fv.oriented(1.0).values));
            let bid = sigmoid(TOXICITY_FOREST.margin_scalar(&fv.oriented(-1.0).values));
            prop_assert!((tox.ask - ask).abs() < 1e-12 && (tox.bid - bid).abs() < 1e-12);
        }
    }
}
