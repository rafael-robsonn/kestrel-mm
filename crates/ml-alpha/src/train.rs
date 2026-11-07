//! Offline histogram trainer for [`ObliviousForest`](crate::ObliviousForest) under logistic
//! loss, and code generation of the trained model as Rust `static` data. Allocates freely;
//! never linked into the hot path.

use core::fmt::Write as _;

use core_types::{FEATURE_NAMES, N_FEATURES};

use crate::AlphaError;
use crate::gbdt::{LANES, sigmoid};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainParams {
    pub trees: usize,
    pub depth: usize,
    pub learning_rate: f64,
    pub l2: f64,
    pub bins: usize,
    /// Minimum hessian mass on each side of a split for it to count towards the gain.
    pub min_child_hessian: f64,
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            trees: 48,
            depth: 5,
            learning_rate: 0.1,
            l2: 1.0,
            bins: 32,
            min_child_hessian: 5.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainedForest {
    pub base_score: f64,
    pub depth: usize,
    pub features: Vec<Vec<u8>>,
    pub thresholds: Vec<Vec<f64>>,
    pub leaves: Vec<Vec<f64>>,
}

impl TrainedForest {
    pub fn margin(&self, x: &[f64; N_FEATURES]) -> f64 {
        let mut sum = self.base_score;
        for t in 0..self.leaves.len() {
            let mut idx = 0;
            for d in 0..self.depth {
                idx |= usize::from(x[self.features[t][d] as usize] > self.thresholds[t][d]) << d;
            }
            sum += self.leaves[t][idx];
        }
        sum
    }

    /// Per-feature count of splits, a cheap importance proxy.
    pub fn split_counts(&self) -> [usize; N_FEATURES] {
        let mut counts = [0; N_FEATURES];
        for tree in &self.features {
            for &f in tree {
                counts[f as usize] += 1;
            }
        }
        counts
    }

    /// Emits a Rust module defining `TOXICITY_FOREST` as a `static`
    /// [`ObliviousForest`](crate::ObliviousForest) in block-SoA layout. Trees are padded to a
    /// multiple of [`LANES`] with zero-leaf trees.
    pub fn to_rust_source(&self, header: &str, metadata: &[(&str, f64)]) -> Result<String, AlphaError> {
        let all_finite = self.base_score.is_finite()
            && self.thresholds.iter().flatten().all(|v| v.is_finite())
            && self.leaves.iter().flatten().all(|v| v.is_finite());
        if !all_finite {
            return Err(AlphaError::Training("non-finite model parameter"));
        }
        let trees = self.leaves.len();
        let blocks = trees.div_ceil(LANES).max(1);
        let n_leaves = 1usize << self.depth;
        let feature = |t: usize, d: usize| self.features.get(t).map_or(0, |f| f[d]);
        let threshold = |t: usize, d: usize| self.thresholds.get(t).map_or(0.0, |th| th[d]);

        let mut s = String::new();
        let fmt_err = |_| AlphaError::Training("formatting failed");
        writeln!(s, "{header}").map_err(fmt_err)?;
        writeln!(s, "#![allow(clippy::approx_constant, clippy::excessive_precision)]\n").map_err(fmt_err)?;
        writeln!(s, "use wide::{{f64x8, u64x8}};\n").map_err(fmt_err)?;
        writeln!(s, "use crate::gbdt::ObliviousForest;\n").map_err(fmt_err)?;
        writeln!(s, "pub const TREES: usize = {trees};").map_err(fmt_err)?;
        writeln!(s, "pub const TREE_BLOCKS: usize = {blocks};").map_err(fmt_err)?;
        writeln!(s, "pub const DEPTH: usize = {};", self.depth).map_err(fmt_err)?;
        writeln!(s, "pub const LEAVES: usize = 1 << DEPTH;").map_err(fmt_err)?;
        for (name, value) in metadata {
            writeln!(s, "pub const {name}: f64 = {value:?};").map_err(fmt_err)?;
        }
        writeln!(
            s,
            "\npub type ToxicityForest = ObliviousForest<TREE_BLOCKS, DEPTH, LEAVES>;\n"
        )
        .map_err(fmt_err)?;
        writeln!(s, "#[rustfmt::skip]").map_err(fmt_err)?;
        writeln!(s, "pub static TOXICITY_FOREST: ToxicityForest = ObliviousForest {{").map_err(fmt_err)?;
        writeln!(s, "    base_score: {:?},", self.base_score).map_err(fmt_err)?;
        writeln!(s, "    features: [").map_err(fmt_err)?;
        for b in 0..blocks {
            writeln!(s, "        [").map_err(fmt_err)?;
            for d in 0..self.depth {
                let lanes: Vec<u64> = (0..LANES).map(|k| u64::from(feature(b * LANES + k, d))).collect();
                writeln!(s, "            u64x8::new({lanes:?}),").map_err(fmt_err)?;
            }
            writeln!(s, "        ],").map_err(fmt_err)?;
        }
        writeln!(s, "    ],\n    thresholds: [").map_err(fmt_err)?;
        for b in 0..blocks {
            writeln!(s, "        [").map_err(fmt_err)?;
            for d in 0..self.depth {
                let lanes: Vec<f64> = (0..LANES).map(|k| threshold(b * LANES + k, d)).collect();
                writeln!(s, "            f64x8::new({lanes:?}),").map_err(fmt_err)?;
            }
            writeln!(s, "        ],").map_err(fmt_err)?;
        }
        writeln!(s, "    ],\n    leaves: [").map_err(fmt_err)?;
        let zero = vec![0.0; n_leaves];
        for b in 0..blocks {
            writeln!(s, "        [").map_err(fmt_err)?;
            for k in 0..LANES {
                let leaves = self.leaves.get(b * LANES + k).unwrap_or(&zero);
                writeln!(s, "            {leaves:?},").map_err(fmt_err)?;
            }
            writeln!(s, "        ],").map_err(fmt_err)?;
        }
        writeln!(s, "    ],\n}};").map_err(fmt_err)?;
        Ok(s)
    }

    pub fn describe(&self) -> String {
        let counts = self.split_counts();
        let mut s = String::new();
        for (name, c) in FEATURE_NAMES.iter().zip(counts) {
            let _ = writeln!(s, "  {name:<16} {c:>4} splits");
        }
        s
    }
}

fn quantile_thresholds(values: &mut [f64], bins: usize) -> Vec<f64> {
    values.sort_unstable_by(f64::total_cmp);
    let n = values.len();
    let mut out: Vec<f64> = (1..bins).map(|k| values[(k * n / bins).min(n - 1)]).collect();
    out.dedup();
    if out.last().is_some_and(|&t| t >= values[n - 1]) {
        out.pop();
    }
    out
}

/// Gradient-boosted oblivious trees, logistic loss, second-order (Newton) leaf values:
///
/// `g_i = p_i - y_i`, `h_i = p_i(1 - p_i)`, `w_leaf = -η · ΣG / (ΣH + λ)`
///
/// Each level picks the single `(feature, threshold)` maximising
/// `Σ_leaves [G_L²/(H_L+λ) + G_R²/(H_R+λ)]` from per-leaf gradient histograms over
/// quantile bins. O(n · F · D) per tree.
pub fn train(x: &[[f64; N_FEATURES]], y: &[f64], params: &TrainParams) -> Result<TrainedForest, AlphaError> {
    let n = x.len();
    if n == 0 || n != y.len() {
        return Err(AlphaError::Training("empty or mismatched training set"));
    }
    if params.depth == 0 || params.depth > 12 || params.trees == 0 || params.bins < 2 || params.bins > 256 {
        return Err(AlphaError::Training("invalid training parameters"));
    }
    if x.iter().flatten().any(|v| !v.is_finite()) || y.iter().any(|v| !(0.0..=1.0).contains(v)) {
        return Err(AlphaError::Training("non-finite feature or label outside [0, 1]"));
    }

    let mut cuts: Vec<Vec<f64>> = Vec::with_capacity(N_FEATURES);
    for f in 0..N_FEATURES {
        let mut col: Vec<f64> = x.iter().map(|r| r[f]).collect();
        cuts.push(quantile_thresholds(&mut col, params.bins));
    }
    let binned: Vec<[u8; N_FEATURES]> = x
        .iter()
        .map(|r| {
            let mut b = [0u8; N_FEATURES];
            for f in 0..N_FEATURES {
                b[f] = cuts[f].partition_point(|&t| r[f] > t) as u8;
            }
            b
        })
        .collect();

    let mean = (y.iter().sum::<f64>() / n as f64).clamp(1e-6, 1.0 - 1e-6);
    let base_score = (mean / (1.0 - mean)).ln();
    let mut margin = vec![base_score; n];
    let mut grad = vec![0.0; n];
    let mut hess = vec![0.0; n];
    let mut leaf_of = vec![0usize; n];

    let n_leaves = 1usize << params.depth;
    let max_bins = params.bins;
    let mut forest = TrainedForest {
        base_score,
        depth: params.depth,
        features: Vec::with_capacity(params.trees),
        thresholds: Vec::with_capacity(params.trees),
        leaves: Vec::with_capacity(params.trees),
    };
    let score = |g: f64, h: f64| g * g / (h + params.l2);

    for _ in 0..params.trees {
        for i in 0..n {
            let p = sigmoid(margin[i]);
            grad[i] = p - y[i];
            hess[i] = (p * (1.0 - p)).max(1e-6);
        }
        leaf_of.fill(0);
        let mut tree_features = Vec::with_capacity(params.depth);
        let mut tree_thresholds = Vec::with_capacity(params.depth);

        for d in 0..params.depth {
            let active = 1usize << d;
            let mut hist = vec![[0.0f64; 2]; active * N_FEATURES * max_bins];
            for i in 0..n {
                let base = leaf_of[i] * N_FEATURES * max_bins;
                for f in 0..N_FEATURES {
                    let cell = &mut hist[base + f * max_bins + binned[i][f] as usize];
                    cell[0] += grad[i];
                    cell[1] += hess[i];
                }
            }
            let mut best: Option<(f64, usize, usize)> = None;
            for f in 0..N_FEATURES {
                let n_cuts = cuts[f].len();
                if n_cuts == 0 {
                    continue;
                }
                let mut gain = vec![0.0f64; n_cuts];
                for l in 0..active {
                    let cells = &hist[(l * N_FEATURES + f) * max_bins..][..max_bins];
                    let (gt, ht) = cells.iter().fold((0.0, 0.0), |a, c| (a.0 + c[0], a.1 + c[1]));
                    let (mut gl, mut hl) = (0.0, 0.0);
                    for j in 0..n_cuts {
                        gl += cells[j][0];
                        hl += cells[j][1];
                        let (gr, hr) = (gt - gl, ht - hl);
                        gain[j] += if hl < params.min_child_hessian || hr < params.min_child_hessian {
                            score(gt, ht)
                        } else {
                            score(gl, hl) + score(gr, hr)
                        };
                    }
                }
                for (j, &g) in gain.iter().enumerate() {
                    if best.is_none_or(|(b, _, _)| g > b) {
                        best = Some((g, f, j));
                    }
                }
            }
            let (f, j) = match best {
                Some((_, f, j)) => (f, j),
                None => (0, 0),
            };
            let threshold = cuts[f].get(j).copied().unwrap_or(f64::MAX);
            tree_features.push(f as u8);
            tree_thresholds.push(threshold);
            for i in 0..n {
                leaf_of[i] |= usize::from(x[i][f] > threshold) << d;
            }
        }

        let mut g_leaf = vec![0.0; n_leaves];
        let mut h_leaf = vec![0.0; n_leaves];
        for i in 0..n {
            g_leaf[leaf_of[i]] += grad[i];
            h_leaf[leaf_of[i]] += hess[i];
        }
        let leaves: Vec<f64> = (0..n_leaves)
            .map(|l| -params.learning_rate * g_leaf[l] / (h_leaf[l] + params.l2))
            .collect();
        for i in 0..n {
            margin[i] += leaves[leaf_of[i]];
        }
        forest.features.push(tree_features);
        forest.thresholds.push(tree_thresholds);
        forest.leaves.push(leaves);
    }
    Ok(forest)
}

/// Mean binary cross-entropy, probabilities clipped to `[1e-12, 1 - 1e-12]`.
pub fn log_loss(p: &[f64], y: &[f64]) -> f64 {
    let n = p.len().max(1) as f64;
    p.iter()
        .zip(y)
        .map(|(&p, &y)| {
            let p = p.clamp(1e-12, 1.0 - 1e-12);
            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        })
        .sum::<f64>()
        / n
}

/// Area under the ROC curve via the Mann-Whitney U statistic with mid-ranks for ties.
pub fn roc_auc(scores: &[f64], labels: &[f64]) -> f64 {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_unstable_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut rank_sum_pos = 0.0;
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let mid_rank = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            if labels[k] > 0.5 {
                rank_sum_pos += mid_rank;
            }
        }
        i = j + 1;
    }
    let n_pos = labels.iter().filter(|&&l| l > 0.5).count() as f64;
    let n_neg = labels.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset(n: usize) -> (Vec<[f64; N_FEATURES]>, Vec<f64>) {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let mut r = [0.0; N_FEATURES];
            for v in r.iter_mut() {
                *v = next() * 2.0 - 1.0;
            }
            let logit = 3.0 * r[0] - 2.0 * r[5] + if r[2] > 0.5 { 1.5 } else { 0.0 };
            y.push(if next() < sigmoid(logit) { 1.0 } else { 0.0 });
            x.push(r);
        }
        (x, y)
    }

    #[test]
    fn learns_signal_and_ignores_noise() {
        let (x, y) = dataset(20_000);
        let forest = train(
            &x[..15_000],
            &y[..15_000],
            &TrainParams {
                trees: 30,
                depth: 4,
                ..TrainParams::default()
            },
        )
        .unwrap();
        let p: Vec<f64> = x[15_000..].iter().map(|r| sigmoid(forest.margin(r))).collect();
        let auc = roc_auc(&p, &y[15_000..]);
        assert!(auc > 0.78, "auc {auc}");
        let base = y[15_000..].iter().sum::<f64>() / 5_000.0;
        let base_ll = log_loss(&vec![base; 5_000], &y[15_000..]);
        assert!(log_loss(&p, &y[15_000..]) < base_ll);
        let counts = forest.split_counts();
        assert!(counts[0] + counts[5] + counts[2] > counts[1] + counts[3] + counts[4] + counts[6] + counts[7]);
    }

    #[test]
    fn auc_edge_cases() {
        assert_eq!(roc_auc(&[0.1, 0.9], &[0.0, 1.0]), 1.0);
        assert_eq!(roc_auc(&[0.9, 0.1], &[0.0, 1.0]), 0.0);
        assert_eq!(roc_auc(&[0.5, 0.5], &[0.0, 1.0]), 0.5);
    }

    #[test]
    fn emitted_source_is_well_formed() {
        let (x, y) = dataset(2_000);
        let forest = train(
            &x,
            &y,
            &TrainParams {
                trees: 2,
                depth: 2,
                ..TrainParams::default()
            },
        )
        .unwrap();
        let src = forest.to_rust_source("// header", &[("HOLDOUT_AUC", 0.7)]).unwrap();
        assert!(src.contains("pub static TOXICITY_FOREST: ToxicityForest"));
        assert!(src.contains("pub const HOLDOUT_AUC: f64 = 0.7;"));
        assert!(!src.contains("NaN") && !src.contains("inf"));
    }
}
