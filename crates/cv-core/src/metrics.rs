//! Cosine similarity and distance.
//!
//! # This is an extension, not a port
//!
//! OpenCV has no cosine metric. Its norm-type enum covers only infinity, L1,
//! L2, squared-L2 and the two Hamming variants; the only "cosine" in its API is
//! the Discrete Cosine Transform. Nothing here corresponds to an upstream
//! symbol, so nothing here counts toward the parity figure.
//!
//! The metric is included because it is the standard way to compare
//! descriptors and embeddings, where magnitude carries no information and only
//! direction matters.
//!
//! # Conventions
//!
//! Inputs are treated as flat vectors: every channel of every pixel, in
//! row-major order. A `1xK` matrix, a `Kx1` matrix and a `1x(K/3)` three-channel
//! matrix are all the same K-element vector.
//!
//! Similarity lies in `[-1, 1]` and distance is `1 - similarity`, so distance
//! lies in `[0, 2]`. Accumulation is in `f64` regardless of input depth.

use crate::bulk::to_f64;
use crate::error::{Code, Result};
use crate::mat::{Mat, MatLike, MatLikeMut};
use crate::types::CV_64FC1;
use crate::{cv_assert, cv_error, Error};

/// Dot product and the two L2 norms, in one pass.
fn accumulate(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    (dot, na.sqrt(), nb.sqrt())
}

/// Similarity from precomputed parts, handling the degenerate case.
///
/// A zero-length vector has no direction, so no angle is defined. This returns
/// `0.0` — the same choice scikit-learn makes — rather than a NaN that would
/// silently poison a downstream sort. Callers that need to distinguish "no
/// similarity" from "undefined" should test the norm themselves.
#[inline]
fn similarity_from(dot: f64, na: f64, nb: f64) -> f64 {
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    // Rounding can push an identical pair a hair outside [-1, 1]; clamping
    // keeps `acos` and `1 - s` well behaved for callers.
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Flatten and check that two operands describe vectors of equal length.
fn pair<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B) -> Result<(Vec<f64>, Vec<f64>)> {
    let (x, y) = (to_f64(a)?, to_f64(b)?);
    if x.len() != y.len() {
        return cv_error!(
            Code::StsUnmatchedSizes,
            "vectors have {} and {} elements",
            x.len(),
            y.len()
        );
    }
    if x.is_empty() {
        return cv_error!(Code::StsBadSize, "cosine metrics need a non-empty vector");
    }
    Ok((x, y))
}

/// Cosine similarity of two equal-length vectors, in `[-1, 1]`.
///
/// Returns `0.0` if either operand has zero length.
pub fn cosine_similarity<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B) -> Result<f64> {
    let (x, y) = pair(a, b)?;
    let (dot, na, nb) = accumulate(&x, &y);
    Ok(similarity_from(dot, na, nb))
}

/// Cosine distance, `1 - cosine_similarity`, in `[0, 2]`.
pub fn cosine_distance<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B) -> Result<f64> {
    Ok(1.0 - cosine_similarity(a, b)?)
}

/// Similarity of `query` against every **row** of `descriptors`.
///
/// `descriptors` is `N x K` (with any channel count, so long as
/// `cols * channels == K`), `query` is any shape holding `K` values. The result
/// is a `1 x N` `CV_64FC1` row.
pub fn cosine_similarities<Q: MatLike + ?Sized, D: MatLike + ?Sized>(
    query: &Q,
    descriptors: &D,
) -> Result<Mat> {
    let q = to_f64(query)?;
    let k = descriptors.cols() as usize * descriptors.channels() as usize;
    cv_assert!(!q.is_empty(), "cosine metrics need a non-empty query");
    if q.len() != k {
        return cv_error!(
            Code::StsUnmatchedSizes,
            "query has {} elements but each descriptor row has {k}",
            q.len()
        );
    }
    let n = descriptors.rows();
    cv_assert!(n > 0, "descriptors must have at least one row");

    // The query's norm is fixed across the whole batch, so hoist it.
    let q_norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rows = to_f64(descriptors)?;

    let mut out = Mat::zeros(1, n, CV_64FC1)?;
    let dst = out.row_as_mut::<f64>(0)?;
    for (i, slot) in dst.iter_mut().enumerate() {
        let row = &rows[i * k..(i + 1) * k];
        let mut dot = 0.0;
        let mut nr = 0.0;
        for (a, b) in q.iter().zip(row.iter()) {
            dot += a * b;
            nr += b * b;
        }
        *slot = similarity_from(dot, q_norm, nr.sqrt());
    }
    Ok(out)
}

/// Distance of `query` against every row of `descriptors`; see
/// [`cosine_similarities`].
pub fn cosine_distances<Q: MatLike + ?Sized, D: MatLike + ?Sized>(
    query: &Q,
    descriptors: &D,
) -> Result<Mat> {
    let mut m = cosine_similarities(query, descriptors)?;
    let row = m.row_as_mut::<f64>(0)?;
    for v in row.iter_mut() {
        *v = 1.0 - *v;
    }
    Ok(m)
}

/// The row of `descriptors` closest to `query`, as `(index, distance)`.
///
/// Ties resolve to the lowest index, which keeps the result deterministic when
/// several descriptors are equidistant.
pub fn nearest_by_cosine<Q: MatLike + ?Sized, D: MatLike + ?Sized>(
    query: &Q,
    descriptors: &D,
) -> Result<(i32, f64)> {
    let d = cosine_distances(query, descriptors)?;
    let row = d.row_as::<f64>(0)?;
    let mut best = 0usize;
    for (i, v) in row.iter().enumerate() {
        if *v < row[best] {
            best = i;
        }
    }
    Ok((best as i32, row[best]))
}

/// Scale a vector to unit L2 length, returning `CV_64FC1`.
///
/// Once vectors are normalised, cosine similarity is just a dot product, so
/// this is worth doing once up front when the same descriptors are queried
/// repeatedly. A zero vector is returned unchanged.
pub fn l2_normalize<M: MatLike + ?Sized>(m: &M) -> Result<Mat> {
    let v = to_f64(m)?;
    if v.is_empty() {
        return Err(Error::new(Code::StsBadSize, "cannot normalize an empty matrix".to_string()));
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let scale = if norm == 0.0 { 1.0 } else { 1.0 / norm };
    let scaled: Vec<f64> = v.iter().map(|x| x * scale).collect();
    crate::bulk::mat_from_f64(m.rows(), m.cols(), m.mat_type().with_depth(crate::Depth::F64), &scaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CV_32FC1, CV_8UC1};
    use crate::Rect;

    fn v(data: &[f64]) -> Mat {
        Mat::from_vec(1, data.len() as i32, 1, data).unwrap()
    }

    #[test]
    fn identical_vectors_have_similarity_one() {
        let a = v(&[1.0, 2.0, 3.0]);
        assert!((cosine_similarity(&a, &a).unwrap() - 1.0).abs() < 1e-12);
        assert!(cosine_distance(&a, &a).unwrap().abs() < 1e-12);
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() {
        let a = v(&[1.0, 0.0]);
        let b = v(&[0.0, 1.0]);
        assert_eq!(cosine_similarity(&a, &b).unwrap(), 0.0);
        assert_eq!(cosine_distance(&a, &b).unwrap(), 1.0);
    }

    #[test]
    fn opposite_vectors_have_similarity_minus_one() {
        let a = v(&[1.0, 2.0, 3.0]);
        let b = v(&[-1.0, -2.0, -3.0]);
        assert!((cosine_similarity(&a, &b).unwrap() + 1.0).abs() < 1e-12);
        // Distance spans [0, 2], not [0, 1].
        assert!((cosine_distance(&a, &b).unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn similarity_ignores_magnitude() {
        // This is the whole point of the metric: only direction matters.
        let a = v(&[1.0, 2.0, 3.0]);
        let b = v(&[100.0, 200.0, 300.0]);
        assert!((cosine_similarity(&a, &b).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn matches_a_hand_computed_value() {
        // a.b = 1*4 + 2*5 + 3*6 = 32; |a| = sqrt(14); |b| = sqrt(77)
        let a = v(&[1.0, 2.0, 3.0]);
        let b = v(&[4.0, 5.0, 6.0]);
        let want = 32.0 / (14.0f64.sqrt() * 77.0f64.sqrt());
        assert!((cosine_similarity(&a, &b).unwrap() - want).abs() < 1e-12);
    }

    #[test]
    fn zero_vectors_yield_zero_similarity_not_nan() {
        // A NaN here would silently corrupt any downstream sort.
        let z = v(&[0.0, 0.0, 0.0]);
        let a = v(&[1.0, 2.0, 3.0]);
        assert_eq!(cosine_similarity(&z, &a).unwrap(), 0.0);
        assert_eq!(cosine_similarity(&a, &z).unwrap(), 0.0);
        assert_eq!(cosine_similarity(&z, &z).unwrap(), 0.0);
        assert!(cosine_distance(&z, &a).unwrap().is_finite());
    }

    #[test]
    fn result_stays_within_bounds_on_random_input() {
        let mut rng = crate::Rng::new(31);
        for _ in 0..200 {
            let a: Vec<f64> = (0..16).map(|_| rng.uniform_f64(-5.0, 5.0)).collect();
            let b: Vec<f64> = (0..16).map(|_| rng.uniform_f64(-5.0, 5.0)).collect();
            let s = cosine_similarity(&v(&a), &v(&b)).unwrap();
            assert!((-1.0..=1.0).contains(&s), "similarity {s} out of range");
            let d = cosine_distance(&v(&a), &v(&b)).unwrap();
            assert!((0.0..=2.0).contains(&d), "distance {d} out of range");
        }
    }

    #[test]
    fn shape_does_not_matter_only_element_order() {
        // 1xK, Kx1 and a multi-channel row are the same vector.
        let row = Mat::from_vec(1, 4, 1, &[1.0f64, 2.0, 3.0, 4.0]).unwrap();
        let col = Mat::from_vec(4, 1, 1, &[1.0f64, 2.0, 3.0, 4.0]).unwrap();
        let two_ch = Mat::from_vec(1, 2, 2, &[1.0f64, 2.0, 3.0, 4.0]).unwrap();
        let other = Mat::from_vec(1, 4, 1, &[4.0f64, 3.0, 2.0, 1.0]).unwrap();
        let a = cosine_similarity(&row, &other).unwrap();
        assert!((cosine_similarity(&col, &other).unwrap() - a).abs() < 1e-15);
        assert!((cosine_similarity(&two_ch, &other).unwrap() - a).abs() < 1e-15);
    }

    #[test]
    fn works_across_depths() {
        let f = Mat::from_vec(1, 3, 1, &[1.0f32, 2.0, 3.0]).unwrap();
        let u = Mat::from_vec(1, 3, 1, &[2u8, 4, 6]).unwrap();
        assert_eq!(f.mat_type(), CV_32FC1);
        assert_eq!(u.mat_type(), CV_8UC1);
        // Same direction, different depth and magnitude.
        assert!((cosine_similarity(&f, &u).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mismatched_lengths_and_empty_input_are_rejected() {
        let a = v(&[1.0, 2.0]);
        let b = v(&[1.0, 2.0, 3.0]);
        assert_eq!(cosine_similarity(&a, &b).unwrap_err().code, Code::StsUnmatchedSizes);
        let e = Mat::zeros(0, 0, CV_64FC1).unwrap();
        assert!(cosine_similarity(&e, &e).is_err());
    }

    #[test]
    fn batch_agrees_with_the_pairwise_form() {
        let desc =
            Mat::from_vec(4, 3, 1, &[1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0])
                .unwrap();
        let q = v(&[1.0, 0.0, 0.0]);
        let sims = cosine_similarities(&q, &desc).unwrap();
        assert_eq!(sims.size(), crate::Size::new(4, 1));

        for i in 0..4 {
            let row = desc.roi(Rect::new(0, i, 3, 1)).unwrap();
            let want = cosine_similarity(&q, &row).unwrap();
            assert!((sims.at::<f64>(0, i).unwrap() - want).abs() < 1e-15, "row {i}");
        }
        let d = cosine_distances(&q, &desc).unwrap();
        assert!((d.at::<f64>(0, 0).unwrap() - 0.0).abs() < 1e-12);
        assert!((d.at::<f64>(0, 3).unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn batch_rejects_a_wrong_length_query() {
        let desc = Mat::zeros(3, 4, CV_64FC1).unwrap();
        let q = v(&[1.0, 2.0]);
        assert_eq!(
            cosine_similarities(&q, &desc).unwrap_err().code,
            Code::StsUnmatchedSizes
        );
    }

    #[test]
    fn nearest_picks_the_closest_row_and_breaks_ties_low() {
        let desc =
            Mat::from_vec(3, 2, 1, &[0.0f64, 1.0, 1.0, 0.0, 2.0, 0.0]).unwrap();
        let q = v(&[1.0, 0.0]);
        let (idx, dist) = nearest_by_cosine(&q, &desc).unwrap();
        // Rows 1 and 2 are both exactly aligned with the query; the lower index
        // must win so the result is deterministic.
        assert_eq!(idx, 1);
        assert!(dist.abs() < 1e-12);
    }

    #[test]
    fn l2_normalize_produces_unit_length() {
        let a = v(&[3.0, 4.0]);
        let n = l2_normalize(&a).unwrap();
        let vals = n.to_vec::<f64>().unwrap();
        assert!((vals[0] - 0.6).abs() < 1e-15);
        assert!((vals[1] - 0.8).abs() < 1e-15);
        let len: f64 = vals.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((len - 1.0).abs() < 1e-15);
    }

    #[test]
    fn normalising_does_not_change_similarity() {
        let a = v(&[1.0, 2.0, 3.0]);
        let b = v(&[4.0, -5.0, 6.0]);
        let raw = cosine_similarity(&a, &b).unwrap();
        let norm = cosine_similarity(&l2_normalize(&a).unwrap(), &l2_normalize(&b).unwrap()).unwrap();
        assert!((raw - norm).abs() < 1e-14);
    }

    #[test]
    fn l2_normalize_leaves_a_zero_vector_alone() {
        let z = v(&[0.0, 0.0]);
        assert_eq!(l2_normalize(&z).unwrap().to_vec::<f64>().unwrap(), vec![0.0, 0.0]);
    }

    #[test]
    fn accepts_strided_views() {
        let data: Vec<f64> = (0..25).map(|i| i as f64).collect();
        let m = Mat::from_vec(5, 5, 1, &data).unwrap();
        let a = m.roi(Rect::new(1, 1, 3, 1)).unwrap();
        let b = m.roi(Rect::new(1, 2, 3, 1)).unwrap();
        assert!(!m.roi(Rect::new(1, 1, 3, 2)).unwrap().is_continuous());
        let s = cosine_similarity(&a, &b).unwrap();
        // Rows (6,7,8) and (11,12,13) point in nearly the same direction.
        assert!(s > 0.99 && s <= 1.0, "{s}");
    }
}
