//! Descriptor matching (`modules/features2d/src/matchers.cpp`).
//!
//! # Picking a norm
//!
//! The norm has to suit the descriptor. Binary descriptors — ORB, BRISK,
//! AKAZE — are bit strings and want [`NormType::Hamming`]; float descriptors
//! like SIFT want [`NormType::L2`]. Using L2 on a binary descriptor still
//! *runs* and returns plausible-looking nonsense, which is why the matcher
//! takes the norm explicitly rather than guessing from the element type.

use cv_core::{cv_assert, Result};

use crate::DMatch;

/// The `M x N` matrix of squared L2 distances between every query and train
/// descriptor.
///
/// # Why this is a matrix multiply
///
/// Expanding the square gives `||q - t||^2 = ||q||^2 + ||t||^2 - 2 q.t`, and
/// the only term that couples the two sets is `q.t` — a plain `(M x D) (D x N)`
/// GEMM. That turns an O(M N D) scalar loop into one call a tuned backend can
/// run on whatever hardware is present.
///
/// With the `rlx` feature this dispatches to `cv_accel::matmul_f32`, which
/// runs on the CPU or on Metal/CUDA/Vulkan depending on which backend features
/// are enabled. Without it, the same arithmetic runs as a scalar triple loop.
/// Both paths are checked against each other in the tests, because a faster
/// answer that disagrees with the reference is not a faster answer.
pub fn l2sqr_matrix(query: &[Vec<f32>], train: &[Vec<f32>]) -> Result<Vec<f32>> {
    let m = query.len();
    let n = train.len();
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let d = query[0].len();
    cv_assert!(
        query.iter().all(|q| q.len() == d) && train.iter().all(|t| t.len() == d),
        "every descriptor must have the same length"
    );

    let qn: Vec<f32> = query.iter().map(|q| q.iter().map(|v| v * v).sum()).collect();
    let tn: Vec<f32> = train.iter().map(|t| t.iter().map(|v| v * v).sum()).collect();

    // Flatten to row-major `q` (M x D) and `t^T` (D x N).
    let qf: Vec<f32> = query.iter().flatten().copied().collect();
    let mut tt = vec![0.0f32; d * n];
    for (j, t) in train.iter().enumerate() {
        for (k, v) in t.iter().enumerate() {
            tt[k * n + j] = *v;
        }
    }

    let dot = dot_products(&qf, &tt, m, d, n)?;
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            // Clamp: the expansion can go slightly negative on identical
            // descriptors through cancellation, and a negative squared
            // distance would come back NaN from the square root.
            out[i * n + j] = (qn[i] + tn[j] - 2.0 * dot[i * n + j]).max(0.0);
        }
    }
    Ok(out)
}

#[cfg(feature = "rlx")]
fn dot_products(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    cv_accel::matmul_f32(a, b, m, k, n).map_err(|e| {
        cv_core::Error::new(cv_core::Code::StsError, format!("rlx matmul failed: {e}"))
    })
}

#[cfg(not(feature = "rlx"))]
fn dot_products(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    Ok(out)
}

/// `cv::NormTypes`, for the subset that makes sense for descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// `cv::NORM_L1` — sum of absolute differences.
    L1,
    /// `cv::NORM_L2` — Euclidean.
    L2,
    /// `cv::NORM_L2SQR` — Euclidean squared; same ordering as L2, no square root.
    L2Sqr,
    /// `cv::NORM_HAMMING` — differing bits. For binary descriptors.
    Hamming,
    /// `cv::NORM_HAMMING2` — differing bits counted in 2-bit blocks, for
    /// descriptors whose bits are meaningful in pairs.
    Hamming2,
}

impl NormType {
    /// Distance between two byte descriptors under this norm.
    #[must_use]
    pub fn distance(self, a: &[u8], b: &[u8]) -> f32 {
        match self {
            NormType::L1 => a
                .iter()
                .zip(b)
                .map(|(x, y)| (i32::from(*x) - i32::from(*y)).unsigned_abs())
                .sum::<u32>() as f32,
            NormType::L2 => NormType::L2Sqr.distance(a, b).sqrt(),
            NormType::L2Sqr => a
                .iter()
                .zip(b)
                .map(|(x, y)| {
                    let d = i32::from(*x) - i32::from(*y);
                    (d * d) as u32
                })
                .sum::<u32>() as f32,
            NormType::Hamming => {
                a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum::<u32>() as f32
            }
            NormType::Hamming2 => {
                // Each 2-bit block counts as one difference if either bit differs.
                a.iter()
                    .zip(b)
                    .map(|(x, y)| {
                        let d = x ^ y;
                        // Fold each pair of bits down to one.
                        ((d & 0b0101_0101) | ((d >> 1) & 0b0101_0101)).count_ones()
                    })
                    .sum::<u32>() as f32
            }
        }
    }

    /// Distance between two `f32` descriptors — SIFT and friends.
    #[must_use]
    pub fn distance_f32(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            NormType::L1 => a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum(),
            NormType::L2 => self.l2sqr_f32(a, b).sqrt(),
            NormType::L2Sqr => self.l2sqr_f32(a, b),
            // A Hamming distance over floats is meaningless; upstream would
            // assert on the descriptor type, and returning a number here would
            // hide the mistake.
            NormType::Hamming | NormType::Hamming2 => f32::NAN,
        }
    }

    fn l2sqr_f32(self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }
}

/// `cv::BFMatcher` — brute force over every pair.
#[derive(Debug, Clone, Copy)]
pub struct BfMatcher {
    norm: NormType,
    cross_check: bool,
}

impl BfMatcher {
    /// `cv::BFMatcher::create(normType, crossCheck)`.
    ///
    /// With `cross_check`, a pair survives only if each descriptor is the
    /// other's best match. That discards most false pairs without needing a
    /// distance threshold, at the cost of returning at most one match per
    /// query.
    #[must_use]
    pub fn new(norm: NormType, cross_check: bool) -> Self {
        Self { norm, cross_check }
    }

    /// The norm this matcher uses.
    #[must_use]
    pub fn norm(&self) -> NormType {
        self.norm
    }

    /// `cv::BFMatcher::match(queryDescriptors, trainDescriptors, matches)`.
    pub fn match_descriptors(&self, query: &[Vec<u8>], train: &[Vec<u8>]) -> Result<Vec<DMatch>> {
        check_shapes(query, train)?;
        let mut out = Vec::new();
        for (qi, q) in query.iter().enumerate() {
            let Some((ti, d)) = best(self.norm, q, train) else { continue };
            if self.cross_check {
                // Is the query also the train descriptor's best?
                let Some((back, _)) = best(self.norm, &train[ti], query) else { continue };
                if back != qi {
                    continue;
                }
            }
            out.push(DMatch {
                query_idx: qi as i32,
                train_idx: ti as i32,
                img_idx: 0,
                distance: d,
            });
        }
        Ok(out)
    }

    /// `cv::BFMatcher::knnMatch(queryDescriptors, trainDescriptors, matches, k)`.
    ///
    /// `cross_check` and `knn` are mutually exclusive upstream — cross-checking
    /// admits at most one match per query, so asking for k of them is
    /// contradictory. This returns an error rather than silently ignoring one
    /// of the two settings.
    pub fn knn_match(
        &self,
        query: &[Vec<u8>],
        train: &[Vec<u8>],
        k: usize,
    ) -> Result<Vec<Vec<DMatch>>> {
        cv_assert!(!self.cross_check, "knnMatch and crossCheck cannot both be used");
        cv_assert!(k > 0, "k must be positive");
        check_shapes(query, train)?;
        let mut out = Vec::with_capacity(query.len());
        for (qi, q) in query.iter().enumerate() {
            let mut all: Vec<DMatch> = train
                .iter()
                .enumerate()
                .map(|(ti, t)| DMatch {
                    query_idx: qi as i32,
                    train_idx: ti as i32,
                    img_idx: 0,
                    distance: self.norm.distance(q, t),
                })
                .collect();
            all.sort_by(|a, b| a.distance.total_cmp(&b.distance));
            all.truncate(k);
            out.push(all);
        }
        Ok(out)
    }

    /// `cv::BFMatcher::radiusMatch(queryDescriptors, trainDescriptors, matches, maxDistance)`.
    ///
    /// Every match within `max_distance`, sorted nearest first. Unlike
    /// `knn_match` the count varies per query — a descriptor with no neighbour
    /// inside the radius gets an empty list rather than a bad match.
    pub fn radius_match(
        &self,
        query: &[Vec<u8>],
        train: &[Vec<u8>],
        max_distance: f32,
    ) -> Result<Vec<Vec<DMatch>>> {
        check_shapes(query, train)?;
        let mut out = Vec::with_capacity(query.len());
        for (qi, q) in query.iter().enumerate() {
            let mut hits: Vec<DMatch> = train
                .iter()
                .enumerate()
                .filter_map(|(ti, t)| {
                    let d = self.norm.distance(q, t);
                    (d <= max_distance).then_some(DMatch {
                        query_idx: qi as i32,
                        train_idx: ti as i32,
                        img_idx: 0,
                        distance: d,
                    })
                })
                .collect();
            hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
            out.push(hits);
        }
        Ok(out)
    }

    /// [`Self::knn_match`] for `f32` descriptors, such as SIFT's.
    pub fn knn_match_f32(
        &self,
        query: &[Vec<f32>],
        train: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<DMatch>>> {
        cv_assert!(!self.cross_check, "knnMatch and crossCheck cannot both be used");
        cv_assert!(k > 0, "k must be positive");
        cv_assert!(
            !matches!(self.norm, NormType::Hamming | NormType::Hamming2),
            "a Hamming norm is meaningless for float descriptors"
        );
        // L2 and its square go through the GEMM path; L1 has no such form.
        let matrix = matches!(self.norm, NormType::L2 | NormType::L2Sqr)
            .then(|| l2sqr_matrix(query, train))
            .transpose()?;

        let n = train.len();
        let mut out = Vec::with_capacity(query.len());
        for (qi, q) in query.iter().enumerate() {
            let mut all: Vec<DMatch> = train
                .iter()
                .enumerate()
                .map(|(ti, t)| {
                    let distance = match &matrix {
                        Some(mm) => {
                            let d2 = mm[qi * n + ti];
                            if self.norm == NormType::L2 { d2.sqrt() } else { d2 }
                        }
                        None => self.norm.distance_f32(q, t),
                    };
                    DMatch {
                        query_idx: qi as i32,
                        train_idx: ti as i32,
                        img_idx: 0,
                        distance,
                    }
                })
                .collect();
            all.sort_by(|a, b| a.distance.total_cmp(&b.distance));
            all.truncate(k);
            out.push(all);
        }
        Ok(out)
    }
}

fn best(norm: NormType, q: &[u8], train: &[Vec<u8>]) -> Option<(usize, f32)> {
    train
        .iter()
        .enumerate()
        .map(|(i, t)| (i, norm.distance(q, t)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

fn check_shapes(query: &[Vec<u8>], train: &[Vec<u8>]) -> Result<()> {
    if let (Some(q), Some(t)) = (query.first(), train.first()) {
        cv_assert!(
            q.len() == t.len(),
            "descriptor lengths differ: query {} vs train {}",
            q.len(),
            t.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(bytes: &[u8]) -> Vec<u8> {
        bytes.to_vec()
    }

    #[test]
    fn the_norms_compute_what_they_claim() {
        let a = [0b0000_0000u8, 0xFF];
        let b = [0b0000_1111u8, 0x0F];
        assert_eq!(NormType::Hamming.distance(&a, &b), 8.0, "4 bits + 4 bits");
        // L1 over the same bytes: |0-15| + |255-15|.
        assert_eq!(NormType::L1.distance(&a, &b), 15.0 + 240.0);
        assert_eq!(NormType::L2Sqr.distance(&a, &b), 15.0 * 15.0 + 240.0 * 240.0);
        assert_eq!(
            NormType::L2.distance(&a, &b),
            (15.0f32 * 15.0 + 240.0 * 240.0).sqrt()
        );
        // Identical descriptors are at distance zero under every norm.
        for n in [NormType::L1, NormType::L2, NormType::L2Sqr, NormType::Hamming, NormType::Hamming2] {
            assert_eq!(n.distance(&a, &a), 0.0, "{n:?} on identical input");
        }
    }

    #[test]
    fn hamming2_counts_bit_pairs_not_bits() {
        // Both bits of one 2-bit block differ: Hamming says 2, Hamming2 says 1.
        let a = [0b0000_0000u8];
        let b = [0b0000_0011u8];
        assert_eq!(NormType::Hamming.distance(&a, &b), 2.0);
        assert_eq!(NormType::Hamming2.distance(&a, &b), 1.0);
        // One bit in each of two blocks: both say 2.
        let c = [0b0000_0101u8];
        assert_eq!(NormType::Hamming.distance(&a, &c), 2.0);
        assert_eq!(NormType::Hamming2.distance(&a, &c), 2.0);
    }

    #[test]
    fn a_hamming_norm_on_floats_is_refused_rather_than_answered() {
        assert!(NormType::Hamming.distance_f32(&[1.0], &[2.0]).is_nan());
        let m = BfMatcher::new(NormType::Hamming, false);
        assert!(m.knn_match_f32(&[vec![1.0]], &[vec![2.0]], 1).is_err());
    }

    #[test]
    fn matching_finds_the_nearest_descriptor() {
        let train = vec![d(&[0x00, 0x00]), d(&[0xFF, 0xFF]), d(&[0x0F, 0x0F])];
        let query = vec![d(&[0x00, 0x01]), d(&[0xFF, 0xFE])];
        let m = BfMatcher::new(NormType::Hamming, false);
        let out = m.match_descriptors(&query, &train).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].train_idx, 0, "nearest to all-zeros");
        assert_eq!(out[1].train_idx, 1, "nearest to all-ones");
        assert_eq!(out[0].distance, 1.0, "one bit apart");
        assert_eq!(out[0].query_idx, 0);
    }

    #[test]
    fn cross_check_discards_one_sided_matches() {
        // train[0] is nearest to both queries, but only one query is train[0]'s
        // own nearest, so cross-checking must drop the other.
        let train = vec![d(&[0x00]), d(&[0xF0])];
        let query = vec![d(&[0x01]), d(&[0x07])];
        let plain = BfMatcher::new(NormType::Hamming, false)
            .match_descriptors(&query, &train)
            .unwrap();
        assert_eq!(plain.len(), 2, "without cross-check both queries match");

        let checked = BfMatcher::new(NormType::Hamming, true)
            .match_descriptors(&query, &train)
            .unwrap();
        assert!(checked.len() < plain.len(), "cross-check must reject something");
        assert_eq!(checked[0].query_idx, 0, "the mutually-nearest pair survives");
        assert_eq!(checked[0].train_idx, 0);
    }

    #[test]
    fn knn_returns_k_sorted_matches() {
        let train = vec![d(&[0x00]), d(&[0x01]), d(&[0x03]), d(&[0xFF])];
        let query = vec![d(&[0x00])];
        let m = BfMatcher::new(NormType::Hamming, false);
        let knn = m.knn_match(&query, &train, 3).unwrap();
        assert_eq!(knn.len(), 1);
        assert_eq!(knn[0].len(), 3);
        // Sorted nearest first.
        assert!(knn[0].windows(2).all(|p| p[0].distance <= p[1].distance));
        assert_eq!(knn[0][0].train_idx, 0);
        assert_eq!(knn[0][0].distance, 0.0);
        // Asking for more than exist gives what exists.
        assert_eq!(m.knn_match(&query, &train, 99).unwrap()[0].len(), 4);
    }

    #[test]
    fn knn_and_cross_check_together_are_refused() {
        let m = BfMatcher::new(NormType::Hamming, true);
        let e = m.knn_match(&[d(&[0])], &[d(&[0])], 2).unwrap_err();
        assert!(format!("{e}").contains("crossCheck"), "the error should say why: {e}");
        assert!(m.knn_match(&[d(&[0])], &[d(&[0])], 0).is_err(), "k must be positive");
    }

    #[test]
    fn radius_match_returns_only_what_is_close_enough() {
        let train = vec![d(&[0x00]), d(&[0x01]), d(&[0x03]), d(&[0xFF])];
        let query = vec![d(&[0x00]), d(&[0xFF])];
        let m = BfMatcher::new(NormType::Hamming, false);
        let r = m.radius_match(&query, &train, 2.0).unwrap();
        assert_eq!(r.len(), 2);
        // Within 2 bits of all-zeros: itself, 0x01 and 0x03.
        assert_eq!(r[0].len(), 3);
        assert!(r[0].iter().all(|x| x.distance <= 2.0));
        assert!(r[0].windows(2).all(|p| p[0].distance <= p[1].distance));
        // All-ones has only itself within 2 bits.
        assert_eq!(r[1].len(), 1);
        assert_eq!(r[1][0].train_idx, 3);
        // A radius of zero keeps only exact hits.
        let z = m.radius_match(&query, &train, 0.0).unwrap();
        assert_eq!(z[0].len(), 1);
    }

    #[test]
    fn float_matching_works_for_sift_style_descriptors() {
        let train = vec![vec![1.0f32, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let query = vec![vec![0.9f32, 0.1, 0.0]];
        let m = BfMatcher::new(NormType::L2, false);
        let knn = m.knn_match_f32(&query, &train, 2).unwrap();
        assert_eq!(knn[0][0].train_idx, 0, "nearest is the first axis");
        assert!(knn[0][0].distance < knn[0][1].distance);
    }

    #[test]
    fn mismatched_descriptor_lengths_are_caught() {
        let m = BfMatcher::new(NormType::Hamming, false);
        assert!(m.match_descriptors(&[d(&[0, 0])], &[d(&[0])]).is_err());
        // Empty inputs are not an error; there is simply nothing to match.
        assert!(m.match_descriptors(&[], &[d(&[0])]).unwrap().is_empty());
        assert!(m.match_descriptors(&[d(&[0])], &[]).unwrap().is_empty());
    }

    /// Norms and crossCheck against cv2 4.13.0.
    #[test]
    fn norms_and_cross_check_match_cv2() {
        // cv2.norm([[0,255]], [[15,15]], <code>)
        let a = [0u8, 255];
        let b = [15u8, 15];
        assert_eq!(NormType::L1.distance(&a, &b), 255.0);
        assert_eq!(NormType::L2Sqr.distance(&a, &b), 57825.0);
        assert!((NormType::L2.distance(&a, &b) - 240.468_29).abs() < 1e-4);
        assert_eq!(NormType::Hamming.distance(&a, &b), 8.0);
        assert_eq!(NormType::Hamming2.distance(&a, &b), 4.0, "cv2 counts bit pairs");

        // cv2.BFMatcher(NORM_HAMMING, crossCheck=...).match(query, train)
        let train = vec![vec![0x00u8], vec![0xF0]];
        let query = vec![vec![0x01u8], vec![0x07]];
        let plain = BfMatcher::new(NormType::Hamming, false)
            .match_descriptors(&query, &train)
            .unwrap();
        // cv2: [(0, 0, 1.0), (1, 0, 3.0)]
        assert_eq!(
            plain
                .iter()
                .map(|m| (m.query_idx, m.train_idx, m.distance))
                .collect::<Vec<_>>(),
            vec![(0, 0, 1.0), (1, 0, 3.0)]
        );
        let checked = BfMatcher::new(NormType::Hamming, true)
            .match_descriptors(&query, &train)
            .unwrap();
        // cv2: [(0, 0, 1.0)]
        assert_eq!(
            checked
                .iter()
                .map(|m| (m.query_idx, m.train_idx, m.distance))
                .collect::<Vec<_>>(),
            vec![(0, 0, 1.0)]
        );
    }

    /// The GEMM distance matrix must agree with the naive triple loop.
    ///
    /// This is the guard on the `rlx` path: it runs under both feature
    /// configurations, so whichever one is compiled is checked against a
    /// reference computed here from first principles. A faster answer that
    /// disagrees with the reference is not a faster answer.
    #[test]
    fn the_distance_matrix_agrees_with_the_naive_reference() {
        // Deterministic pseudo-random descriptors, a realistic 128 long.
        let mk = |seed: u32, n: usize| -> Vec<Vec<f32>> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    (0..128)
                        .map(|_| {
                            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            ((s >> 16) & 0xFF) as f32
                        })
                        .collect()
                })
                .collect()
        };
        let query = mk(7, 9);
        let train = mk(99, 13);

        let got = l2sqr_matrix(&query, &train).unwrap();
        assert_eq!(got.len(), 9 * 13);
        for (i, q) in query.iter().enumerate() {
            for (j, t) in train.iter().enumerate() {
                let want: f32 = q.iter().zip(t).map(|(a, b)| (a - b) * (a - b)).sum();
                let g = got[i * 13 + j];
                // Relative tolerance: the expansion reassociates the sum, so
                // the last bits legitimately differ from a direct subtraction.
                assert!(
                    (g - want).abs() <= 1e-3 * want.max(1.0),
                    "({i},{j}): matrix {g} vs direct {want}"
                );
            }
        }
    }

    #[test]
    fn identical_descriptors_give_a_non_negative_zero_distance() {
        // Cancellation in the expansion can go slightly negative; a negative
        // squared distance would come back NaN from the square root.
        let d = vec![vec![250.0f32; 128], vec![3.0; 128]];
        let m = l2sqr_matrix(&d, &d).unwrap();
        assert!(m.iter().all(|v| *v >= 0.0), "negative squared distance: {m:?}");
        assert!(m[0] < 1e-2, "a descriptor is at distance ~0 from itself, got {}", m[0]);
        assert!(m[3] < 1e-2);
        // And the matched knn distance is finite, not NaN.
        let bf = BfMatcher::new(NormType::L2, false);
        let knn = bf.knn_match_f32(&d, &d, 2).unwrap();
        assert!(knn[0].iter().all(|x| x.distance.is_finite()), "{knn:?}");
        assert_eq!(knn[0][0].train_idx, 0, "nearest to itself");
    }

    #[test]
    fn the_matrix_path_and_the_direct_path_rank_matches_identically() {
        let mk = |seed: u32, n: usize| -> Vec<Vec<f32>> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    (0..64)
                        .map(|_| {
                            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                            ((s >> 20) & 0x7F) as f32
                        })
                        .collect()
                })
                .collect()
        };
        let query = mk(3, 6);
        let train = mk(11, 10);
        // L2 goes through the GEMM; L1 does not, and is the control.
        let via_gemm = BfMatcher::new(NormType::L2, false)
            .knn_match_f32(&query, &train, 3)
            .unwrap();
        for (qi, row) in via_gemm.iter().enumerate() {
            let mut direct: Vec<(usize, f32)> = train
                .iter()
                .enumerate()
                .map(|(ti, t)| (ti, NormType::L2.distance_f32(&query[qi], t)))
                .collect();
            direct.sort_by(|a, b| a.1.total_cmp(&b.1));
            assert_eq!(row[0].train_idx as usize, direct[0].0, "query {qi} best match");
            assert!(
                (row[0].distance - direct[0].1).abs() <= 1e-2 * direct[0].1.max(1.0),
                "query {qi}: {} vs {}",
                row[0].distance,
                direct[0].1
            );
        }
    }

    #[test]
    fn the_matrix_handles_empty_and_ragged_input() {
        assert!(l2sqr_matrix(&[], &[vec![1.0]]).unwrap().is_empty());
        assert!(l2sqr_matrix(&[vec![1.0]], &[]).unwrap().is_empty());
        // Ragged descriptors are an error, not a silent truncation.
        assert!(l2sqr_matrix(&[vec![1.0, 2.0]], &[vec![1.0]]).is_err());
        assert!(l2sqr_matrix(&[vec![1.0], vec![1.0, 2.0]], &[vec![1.0]]).is_err());
    }
}
