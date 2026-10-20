//! Eigenvalues, SVD, PCA and clustering
//! (`modules/core/src/lapack.cpp`, `pca.cpp`, `kmeans.cpp`, `matmul.dispatch.cpp`).
//!
//! Everything here computes in `f64` and returns `CV_64F`, regardless of input
//! depth. Upstream dispatches `float` and `double` separately; a port with no
//! BLAS underneath gains nothing from the narrower path and loses accuracy, so
//! there is one implementation.

use crate::error::{Code, Result};
use crate::mat::{Mat, MatLike, MatLikeMut};
use crate::types::CV_64FC1;
use crate::{cv_assert, cv_error};

/// Read a single-channel matrix into a dense row-major `f64` buffer.
fn dense<S: MatLike + ?Sized>(m: &S) -> Result<(Vec<f64>, usize, usize)> {
    cv_assert!(m.channels() == 1, "expected a single-channel matrix");
    let (h, w) = (m.rows() as usize, m.cols() as usize);
    let mut v = vec![0.0f64; h * w];
    for y in 0..h {
        for x in 0..w {
            v[y * w + x] = m.at_f64(y as i32, x as i32, 0)?;
        }
    }
    Ok((v, h, w))
}

/// Wrap a dense buffer as a `CV_64FC1` matrix.
fn undense(v: &[f64], h: usize, w: usize) -> Result<Mat> {
    Mat::from_vec(h as i32, w as i32, 1, v)
}

/// [`jacobi`] on a raw buffer, for callers outside this module.
pub(crate) fn jacobi_dense(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    jacobi(a, n)
}


/// Cyclic Jacobi eigen-decomposition of a symmetric matrix, in place.
///
/// Returns eigenvalues in `w` and eigenvectors as the **rows** of `v`, sorted
/// by descending eigenvalue — upstream's convention for `cv::eigen`, which
/// differs from LAPACK's ascending order.
fn jacobi(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; n * n];
    for (i, e) in v.iter_mut().step_by(n + 1).enumerate() {
        let _ = i;
        *e = 1.0;
    }
    // 100 sweeps is far beyond what cyclic Jacobi needs (it converges
    // quadratically); the loop exits on the off-diagonal norm.
    for _ in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        if off <= 1e-300 || off.sqrt() < 1e-15 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let (app, aqq) = (a[p * n + p], a[q * n + q]);
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let mut vals: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&x, &y| vals[y].partial_cmp(&vals[x]).unwrap_or(core::cmp::Ordering::Equal));
    let sorted_vals: Vec<f64> = idx.iter().map(|&i| vals[i]).collect();
    // Eigenvectors as rows, sign-normalised so the largest-magnitude component
    // is positive. Without this the sign is arbitrary and results are unstable
    // between runs and platforms.
    let mut rows = vec![0.0f64; n * n];
    for (r, &c) in idx.iter().enumerate() {
        let mut big = 0.0f64;
        let mut sign = 1.0f64;
        for k in 0..n {
            let e = v[k * n + c];
            if e.abs() > big {
                big = e.abs();
                sign = if e < 0.0 { -1.0 } else { 1.0 };
            }
        }
        for k in 0..n {
            rows[r * n + k] = v[k * n + c] * sign;
        }
    }
    vals = sorted_vals;
    (vals, rows)
}

/// `cv::eigen(src, eigenvalues, eigenvectors)` — symmetric matrices only.
///
/// Returns `(values, vectors)` where `values` is `n x 1` in descending order
/// and each **row** of `vectors` is the unit eigenvector for the matching
/// value, as upstream documents.
pub fn eigen<S: MatLike + ?Sized>(src: &S) -> Result<(Mat, Mat)> {
    let (mut a, h, w) = dense(src)?;
    cv_assert!(h == w, "eigen requires a square matrix, got {h}x{w}");
    cv_assert!(h > 0, "eigen requires a non-empty matrix");
    // Upstream asserts symmetry too; a silently wrong answer is worse.
    for i in 0..h {
        for j in i + 1..h {
            let (p, q) = (a[i * w + j], a[j * w + i]);
            let tol = 1e-9 * (1.0 + p.abs() + q.abs());
            if (p - q).abs() > tol {
                return cv_error!(
                    Code::StsBadArg,
                    "eigen requires a symmetric matrix; [{i}][{j}]={p} but [{j}][{i}]={q}"
                );
            }
        }
    }
    let (vals, vecs) = jacobi(&mut a, h);
    Ok((undense(&vals, h, 1)?, undense(&vecs, h, h)?))
}

/// Result of [`svd`]: `src = u * diag(w) * vt`.
#[derive(Debug)]
pub struct Svd {
    /// Left singular vectors, `m x k`.
    pub u: Mat,
    /// Singular values, `k x 1`, descending.
    pub w: Mat,
    /// Right singular vectors, transposed: `k x n`.
    pub vt: Mat,
}

/// `cv::SVD::compute(src, w, u, vt)` — the thin decomposition.
///
/// One-sided Jacobi on `A` directly, which is what upstream does:
/// `SVD::compute` dispatches to `JacobiSVD` in `lapack.cpp`.
///
/// This used to eigen-decompose `A^T A`, on the grounds that the callers here —
/// `solve`, `PCA`, homography fitting — are well conditioned enough not to
/// care. Two of them are; the method is not. Squaring the condition number
/// loses every singular value below `sqrt(eps) * sigma_max`, and on a matrix
/// with singular values a decade or two apart down to `1e-9` the shortcut
/// returned the smallest **108% wrong** where upstream holds it to nine
/// digits. See `svd_holds_the_small_singular_values_the_shortcut_would_have_lost`.
pub fn svd<S: MatLike + ?Sized>(src: &S) -> Result<Svd> {
    let (_, m, n) = dense(src)?;
    cv_assert!(m > 0 && n > 0, "SVD requires a non-empty matrix");
    let r = crate::linalg2::svd_decomp(src)?;
    Ok(Svd { u: r.u, w: r.w, vt: r.vt })
}

/// `cv::SVD::backSubst(w, u, vt, rhs, dst)` — solve using a decomposition in hand.
///
/// Singular values at or below `eps * sum(w)` are dropped, with
/// `eps = 2 * DBL_EPSILON`, which is upstream's `SVBkSb` threshold. That is
/// what makes the result the least-squares solution of a rank-deficient system
/// rather than an error.
pub fn svd_back_subst<W, U, V, B>(w: &W, u: &U, vt: &V, rhs: &B) -> Result<Mat>
where
    W: MatLike + ?Sized,
    U: MatLike + ?Sized,
    V: MatLike + ?Sized,
    B: MatLike + ?Sized,
{
    let (wv, wh, ww) = dense(w)?;
    let (uv, m, k) = dense(u)?;
    let (vv, kk, n) = dense(vt)?;
    let (bv, bh, nb) = dense(rhs)?;
    cv_assert!(wh * ww == k && k == kk, "SVD::backSubst: w, u and vt disagree on rank");
    cv_assert!(bh == m, "SVD::backSubst: rhs has {bh} rows but u has {m}");

    let threshold = wv.iter().sum::<f64>() * (f64::EPSILON * 2.0);
    let mut x = vec![0.0f64; n * nb];
    for i in 0..k {
        if wv[i].abs() <= threshold {
            continue;
        }
        let wi = 1.0 / wv[i];
        for j in 0..nb {
            // (uᵀ b)_i scaled by 1/w_i, spread along v_i.
            let mut s = 0.0;
            for t in 0..m {
                s += uv[t * k + i] * bv[t * nb + j];
            }
            s *= wi;
            for r in 0..n {
                x[r * nb + j] += s * vv[i * n + r];
            }
        }
    }
    undense(&x, n, nb)
}

/// How [`calc_covar_matrix`] and [`pca_compute`] read their input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataLayout {
    /// `COVAR_ROWS` — one sample per row; `PCA::DATA_AS_ROW`.
    Rows,
    /// `COVAR_COLS` — one sample per column; `PCA::DATA_AS_COL`.
    Cols,
}

/// `cv::CovarFlags`, with upstream's values.
///
/// Upstream or-s these into one argument. Here [`calc_covar_matrix`] takes the
/// layout and the scaling as separate parameters, so the constants exist for
/// translating a flags word rather than for passing.
pub mod covar_flags {
    /// `COVAR_SCRAMBLED` — compute the `count x count` scatter of the samples
    /// rather than the `dim x dim` covariance. Not implemented here.
    pub const COVAR_SCRAMBLED: i32 = 0;
    /// `COVAR_NORMAL` — the ordinary `dim x dim` covariance; always in effect.
    pub const COVAR_NORMAL: i32 = 1;
    /// `COVAR_USE_AVG` / `USE_AVG` — take the mean from the caller instead of
    /// computing it. Upstream carries both spellings for the same value.
    pub const COVAR_USE_AVG: i32 = 2;
    /// `COVAR_SCALE` — divide by the sample count; the `normalize` argument.
    pub const COVAR_SCALE: i32 = 4;
    /// `COVAR_ROWS` — [`super::DataLayout::Rows`].
    pub const COVAR_ROWS: i32 = 8;
    /// `COVAR_COLS` — [`super::DataLayout::Cols`].
    pub const COVAR_COLS: i32 = 16;
}

/// `cv::SVD::Flags` — options upstream's `SVD` constructor takes.
///
/// [`svd`] always computes the thin `u`/`vt` from an untouched copy of the
/// input, so none of these change its behaviour; they are declared for callers
/// translating a flags word.
pub mod svd_flags {
    /// `SVD::MODIFY_A` — permit overwriting the input. Never needed here: this
    /// port takes `&self` and allocates its own working buffer.
    pub const MODIFY_A: i32 = 1;
    /// `SVD::NO_UV` — compute only the singular values.
    pub const NO_UV: i32 = 2;
    /// `SVD::FULL_UV` — return square `u` and `vt` rather than the thin form.
    pub const FULL_UV: i32 = 4;
}

/// `cv::calcCovarMatrix(samples, covar, mean, flags)`.
///
/// Returns `(covar, mean)`. `normalize` divides by the sample count, matching
/// upstream's `CV_COVAR_NORMAL | CV_COVAR_SCALE`; without it the result is the
/// scatter matrix, which is what `CV_COVAR_NORMAL` alone gives.
pub fn calc_covar_matrix<S: MatLike + ?Sized>(
    samples: &S,
    layout: DataLayout,
    normalize: bool,
) -> Result<(Mat, Mat)> {
    let (v, h, w) = dense(samples)?;
    let (count, dim) = match layout {
        DataLayout::Rows => (h, w),
        DataLayout::Cols => (w, h),
    };
    cv_assert!(count > 0 && dim > 0, "calcCovarMatrix requires a non-empty sample set");
    let at = |s: usize, d: usize| match layout {
        DataLayout::Rows => v[s * w + d],
        DataLayout::Cols => v[d * w + s],
    };

    let mut mean = vec![0.0f64; dim];
    for (d, m) in mean.iter_mut().enumerate() {
        *m = (0..count).map(|s| at(s, d)).sum::<f64>() / count as f64;
    }
    let mut cov = vec![0.0f64; dim * dim];
    for s in 0..count {
        for i in 0..dim {
            let di = at(s, i) - mean[i];
            for j in 0..dim {
                cov[i * dim + j] += di * (at(s, j) - mean[j]);
            }
        }
    }
    if normalize {
        for c in &mut cov {
            *c /= count as f64;
        }
    }
    Ok((undense(&cov, dim, dim)?, undense(&mean, 1, dim)?))
}

/// Result of [`pca_compute`].
#[derive(Debug)]
pub struct Pca {
    /// Principal axes as rows, most significant first.
    pub eigenvectors: Mat,
    /// Variance along each axis.
    pub eigenvalues: Mat,
    /// Sample mean, one row.
    pub mean: Mat,
}

impl Pca {
    /// `cv::PCA::project(vec)` — centre and rotate onto the retained axes.
    pub fn project<S: MatLike + ?Sized>(&self, data: &S) -> Result<Mat> {
        let (v, n, d) = dense(data)?;
        let k = self.eigenvectors.rows() as usize;
        cv_assert!(
            d == self.eigenvectors.cols() as usize,
            "PCA::project expects {} columns, got {d}",
            self.eigenvectors.cols()
        );
        let mut out = vec![0.0f64; n * k];
        for s in 0..n {
            for r in 0..k {
                let mut acc = 0.0;
                for c in 0..d {
                    acc += (v[s * d + c] - self.mean.at_f64(0, c as i32, 0)?)
                        * self.eigenvectors.at_f64(r as i32, c as i32, 0)?;
                }
                out[s * k + r] = acc;
            }
        }
        undense(&out, n, k)
    }

    /// `cv::PCA::backProject(vec)` — the inverse of [`Pca::project`].
    pub fn back_project<S: MatLike + ?Sized>(&self, data: &S) -> Result<Mat> {
        let (v, n, k) = dense(data)?;
        let d = self.eigenvectors.cols() as usize;
        cv_assert!(
            k == self.eigenvectors.rows() as usize,
            "PCA::backProject expects {} columns, got {k}",
            self.eigenvectors.rows()
        );
        let mut out = vec![0.0f64; n * d];
        for s in 0..n {
            for c in 0..d {
                let mut acc = self.mean.at_f64(0, c as i32, 0)?;
                for r in 0..k {
                    acc += v[s * k + r] * self.eigenvectors.at_f64(r as i32, c as i32, 0)?;
                }
                out[s * d + c] = acc;
            }
        }
        undense(&out, n, d)
    }
}

/// `cv::PCACompute(data, mean, eigenvectors, maxComponents)`, and the
/// `cv::PCA(data, mean, flags, maxComponents)` constructor it wraps.
///
/// Also covers `cv::PCACompute2`, which differs only in additionally returning
/// the eigenvalues — [`Pca`] carries them either way, so there is one entry
/// point here rather than upstream's two.
///
/// `max_components` of 0 keeps every axis, as upstream's default does.
///
/// Returns `min(samples, dimensions)` axes, as upstream does — `pca.cpp` sets
/// `count = std::min(len, in_count)`. Three points span at most three
/// directions however many dimensions they live in.
///
/// # Accuracy
///
/// Computed from the SVD of the centred data, not from the covariance matrix.
/// Upstream builds the covariance and eigen-decomposes it, and the numbers
/// below say why that is worth diverging over: a principal value is a
/// *variance*, `sigma^2 / n`, so the data's dynamic range enters squared and
/// the covariance route loses two digits for every one of range.
///
/// | `sigma_i / sigma_max` | variance | here | via covariance |
/// |---|---|---|---|
/// | `1e0` | `1.6e-2` | exact | `3.4e-16` |
/// | `1e-4` | `1.6e-10` | `5.0e-16` | `3.6e-9` |
/// | `1e-6` | `1.6e-14` | `2.5e-12` | `1.2e-5` |
/// | `1e-9` | `1.5e-20` | `1.2e-10` | `0.59` |
///
/// This is not a divergence anyone can observe where cv2 is itself accurate:
/// `examples/parity_check.rs` compares both the mean and the axes against a
/// real `cv2 4.13.0` and they still agree to `1e-9`, with the full run
/// unchanged at 298 exact and 0 mismatches. It shows up only in the range
/// where cv2's own answer has stopped being right — on a three-sample input,
/// `cv2.PCACompute2` reports a third eigenvalue of `-1.99e-16`, a negative
/// variance, where this returns `+4.3e-33`.
///
/// Pinned by `pca_resolves_variances_the_covariance_route_would_have_lost`.
pub fn pca_compute<S: MatLike + ?Sized>(
    data: &S,
    layout: DataLayout,
    max_components: i32,
) -> Result<Pca> {
    let (v, h, w) = dense(data)?;
    let (count, dim) = match layout {
        DataLayout::Rows => (h, w),
        DataLayout::Cols => (w, h),
    };
    cv_assert!(count > 0 && dim > 0, "PCACompute requires a non-empty sample set");
    let at = |s: usize, d: usize| match layout {
        DataLayout::Rows => v[s * w + d],
        DataLayout::Cols => v[d * w + s],
    };

    let mut mean = vec![0.0f64; dim];
    for (d, m) in mean.iter_mut().enumerate() {
        *m = (0..count).map(|s| at(s, d)).sum::<f64>() / count as f64;
    }
    // The centred samples, one per row, whichever way they arrived.
    let mut x = vec![0.0f64; count * dim];
    for s in 0..count {
        for d in 0..dim {
            x[s * dim + d] = at(s, d) - mean[d];
        }
    }

    let (sv, _u, vt) = crate::linalg2::jacobi_svd_dense(&x, count, dim);
    // `count` samples span at most `count` directions, which is why upstream
    // caps the count this way rather than at `dim`.
    let k = count.min(dim);
    let keep = if max_components <= 0 { k } else { (max_components as usize).min(k) };

    let mut ev = Mat::zeros(keep as i32, dim as i32, CV_64FC1)?;
    let mut va = Mat::zeros(keep as i32, 1, CV_64FC1)?;
    for r in 0..keep {
        // A principal value is a variance, and `calcCovarMatrix` scales by the
        // sample count rather than by `count - 1`; matching that keeps the
        // numbers identical to `cv2.PCACompute2`.
        va.pixel_mut::<f64>(r as i32, 0)?[0] = sv[r] * sv[r] / count as f64;
        for c in 0..dim {
            ev.pixel_mut::<f64>(r as i32, c as i32)?[0] = vt[r * dim + c];
        }
    }
    Ok(Pca { eigenvectors: ev, eigenvalues: va, mean: undense(&mean, 1, dim)? })
}

/// `cv::Mahalanobis(v1, v2, icovar)`.
pub fn mahalanobis<A: MatLike + ?Sized, B: MatLike + ?Sized, C: MatLike + ?Sized>(
    v1: &A,
    v2: &B,
    icovar: &C,
) -> Result<f64> {
    let (a, _, _) = dense(v1)?;
    let (b, _, _) = dense(v2)?;
    let (ic, ih, iw) = dense(icovar)?;
    cv_assert!(a.len() == b.len(), "Mahalanobis operands must have equal length");
    cv_assert!(
        ih == a.len() && iw == a.len(),
        "icovar must be {}x{}, got {ih}x{iw}",
        a.len(),
        a.len()
    );
    let d: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let mut acc = 0.0;
    for i in 0..d.len() {
        for j in 0..d.len() {
            acc += d[i] * ic[i * iw + j] * d[j];
        }
    }
    Ok(acc.max(0.0).sqrt())
}

/// Result of [`kmeans`].
#[derive(Debug)]
pub struct KMeans {
    /// Cluster index per sample.
    pub labels: Vec<i32>,
    /// Cluster centres, one per row.
    pub centers: Mat,
    /// Final compactness — the sum of squared distances to the assigned centre.
    pub compactness: f64,
}

/// `cv::KmeansFlags` — how [`kmeans`] picks its starting centres.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum KMeansFlags {
    /// `KMEANS_RANDOM_CENTERS` — draw each initial centre uniformly at random.
    RandomCenters = 0,
    /// `KMEANS_USE_INITIAL_LABELS` — start from a labelling the caller supplies.
    ///
    /// Upstream applies this to the first attempt only and seeds the rest
    /// randomly; the same is done here.
    UseInitialLabels = 1,
    /// `KMEANS_PP_CENTERS` — Arthur and Vassilvitskii's k-means++ seeding.
    #[default]
    PpCenters = 2,
}

/// `cv::kmeans(data, K, bestLabels, criteria, attempts, flags, centers)`.
///
/// Lloyd's algorithm run `attempts` times with the best compactness kept.
/// Seeding is driven by `rng` so runs are reproducible; upstream seeds from its
/// global RNG, so cluster *numbering* will not match cv2 even when the
/// partition does.
///
/// `initial_labels` is upstream's in/out `bestLabels`, read only when `flags`
/// is [`KMeansFlags::UseInitialLabels`].
///
/// The signature is wide because upstream's is: splitting `criteria` into a
/// builder would hide which of its two halves a given caller actually set.
#[allow(clippy::too_many_arguments)]
pub fn kmeans<S: MatLike + ?Sized>(
    data: &S,
    k: i32,
    attempts: i32,
    max_iter: i32,
    epsilon: f64,
    flags: KMeansFlags,
    initial_labels: Option<&[i32]>,
    rng: &mut crate::rng::Rng,
) -> Result<KMeans> {
    let (v, n, d) = dense(data)?;
    cv_assert!(k > 0, "kmeans needs at least one cluster, got {k}");
    cv_assert!(
        n >= k as usize,
        "kmeans needs at least as many samples as clusters: {n} < {k}"
    );
    if flags == KMeansFlags::UseInitialLabels {
        let Some(l) = initial_labels else {
            return cv_error!(
                Code::StsBadArg,
                "KMEANS_USE_INITIAL_LABELS needs an initial labelling"
            );
        };
        cv_assert!(l.len() == n, "initial labels: expected {n} entries, got {}", l.len());
        cv_assert!(
            l.iter().all(|&c| c >= 0 && c < k),
            "initial labels must all be in 0..{k}"
        );
    }
    let k = k as usize;
    let attempts = attempts.max(1);
    let dist2 = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };

    let mut best: Option<KMeans> = None;
    for attempt in 0..attempts {
        let mut centers: Vec<f64> = Vec::with_capacity(k * d);
        // Only the first attempt honours a supplied labelling; the rest fall
        // back to k-means++ so repeated attempts explore, as upstream does.
        let seeding = match (flags, attempt) {
            (KMeansFlags::UseInitialLabels, 0) => KMeansFlags::UseInitialLabels,
            (KMeansFlags::UseInitialLabels, _) | (KMeansFlags::PpCenters, _) => {
                KMeansFlags::PpCenters
            }
            (KMeansFlags::RandomCenters, _) => KMeansFlags::RandomCenters,
        };

        match seeding {
            KMeansFlags::UseInitialLabels => {
                // Centre of each supplied cluster; an empty one takes a random sample.
                let labels = initial_labels.expect("validated above");
                let mut counts = vec![0usize; k];
                let mut sums = vec![0.0f64; k * d];
                for (s, &c) in labels.iter().enumerate() {
                    counts[c as usize] += 1;
                    for j in 0..d {
                        sums[c as usize * d + j] += v[s * d + j];
                    }
                }
                for c in 0..k {
                    if counts[c] == 0 {
                        let pick = (rng.next_u32() as usize) % n;
                        centers.extend_from_slice(&v[pick * d..(pick + 1) * d]);
                    } else {
                        centers.extend((0..d).map(|j| sums[c * d + j] / counts[c] as f64));
                    }
                }
            }
            KMeansFlags::RandomCenters => {
                for _ in 0..k {
                    let pick = (rng.next_u32() as usize) % n;
                    centers.extend_from_slice(&v[pick * d..(pick + 1) * d]);
                }
            }
            KMeansFlags::PpCenters => {
                let first = (rng.next_u32() as usize) % n;
                centers.extend_from_slice(&v[first * d..(first + 1) * d]);
                let mut nearest: Vec<f64> =
                    (0..n).map(|s| dist2(&v[s * d..(s + 1) * d], &centers[..d])).collect();
                for c in 1..k {
                    let total: f64 = nearest.iter().sum();
                    let pick = if total <= 0.0 {
                        (rng.next_u32() as usize) % n
                    } else {
                        let target = rng.uniform_f64(0.0, total);
                        let mut acc = 0.0;
                        let mut chosen = n - 1;
                        for (s, &w) in nearest.iter().enumerate() {
                            acc += w;
                            if acc >= target {
                                chosen = s;
                                break;
                            }
                        }
                        chosen
                    };
                    centers.extend_from_slice(&v[pick * d..(pick + 1) * d]);
                    for (s, near) in nearest.iter_mut().enumerate() {
                        *near =
                            near.min(dist2(&v[s * d..(s + 1) * d], &centers[c * d..(c + 1) * d]));
                    }
                }
            }
        }

        let mut labels = vec![0i32; n];
        let mut compactness = 0.0;
        for _ in 0..max_iter.max(1) {
            compactness = 0.0;
            for s in 0..n {
                let sample = &v[s * d..(s + 1) * d];
                let (mut bi, mut bd) = (0usize, f64::INFINITY);
                for c in 0..k {
                    let e = dist2(sample, &centers[c * d..(c + 1) * d]);
                    if e < bd {
                        bd = e;
                        bi = c;
                    }
                }
                labels[s] = bi as i32;
                compactness += bd;
            }
            let mut sums = vec![0.0f64; k * d];
            let mut counts = vec![0usize; k];
            for s in 0..n {
                let c = labels[s] as usize;
                counts[c] += 1;
                for j in 0..d {
                    sums[c * d + j] += v[s * d + j];
                }
            }
            let mut shift = 0.0f64;
            for c in 0..k {
                if counts[c] == 0 {
                    continue; // keep an empty cluster where it is
                }
                for j in 0..d {
                    let nv = sums[c * d + j] / counts[c] as f64;
                    shift += (nv - centers[c * d + j]).abs();
                    centers[c * d + j] = nv;
                }
            }
            if shift <= epsilon {
                break;
            }
        }
        if best.as_ref().map_or(true, |b| compactness < b.compactness) {
            best = Some(KMeans {
                labels: labels.clone(),
                centers: undense(&centers, k, d)?,
                compactness,
            });
        }
    }
    match best {
        Some(b) => Ok(b),
        None => cv_error!(Code::StsError, "kmeans produced no result"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(v: &[f64], r: i32, c: i32) -> Mat {
        Mat::from_vec(r, c, 1, v).unwrap()
    }

    #[test]
    fn eigen_of_a_diagonal_matrix_is_its_diagonal() {
        let a = m(&[3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0], 3, 3);
        let (w, _) = eigen(&a).unwrap();
        let got: Vec<f64> = (0..3).map(|i| w.at_f64(i, 0, 0).unwrap()).collect();
        assert_eq!(got, vec![3.0, 2.0, 1.0], "descending, as cv::eigen documents");
    }

    #[test]
    fn eigen_reconstructs_the_original() {
        let a = m(&[4.0, 1.0, 1.0, 3.0], 2, 2);
        let (w, v) = eigen(&a).unwrap();
        // A == V^T diag(w) V for row-eigenvector V.
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0;
                for r in 0..2 {
                    acc += w.at_f64(r, 0, 0).unwrap()
                        * v.at_f64(r, i, 0).unwrap()
                        * v.at_f64(r, j, 0).unwrap();
                }
                let want = a.at_f64(i, j, 0).unwrap();
                assert!((acc - want).abs() < 1e-9, "[{i}][{j}]: {acc} vs {want}");
            }
        }
    }

    #[test]
    fn eigen_rejects_an_asymmetric_matrix() {
        let a = m(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!(eigen(&a).is_err(), "cv::eigen is documented for symmetric input only");
    }

    /// `A = U diag(sigma) Vt` with known singular values and *mixed* columns.
    ///
    /// The mixing is the point. Without `Vt` the columns of `A` are orthogonal,
    /// `A^T A` is diagonal, nothing cancels, and eigen-decomposing it is
    /// perfectly accurate — a test built that way reports no difference between
    /// the two routes and concludes there is nothing to choose between them.
    /// The loss needs columns that are nearly parallel with comparable norms,
    /// so a small singular value is the residue of cancellation between large
    /// numbers.
    fn graded(rows: usize, cols: usize, sigma: &[f64]) -> Vec<f64> {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        // Modified Gram-Schmidt, twice: one pass leaves the columns orthogonal
        // only to about sqrt(eps), which would put more error into `A` than the
        // effect being measured.
        let mut ortho = |m: usize, n: usize| {
            let mut q: Vec<f64> = (0..m * n).map(|_| rnd()).collect();
            for _ in 0..2 {
                for j in 0..n {
                    for k in 0..j {
                        let d: f64 = (0..m).map(|r| q[r * n + j] * q[r * n + k]).sum();
                        for r in 0..m {
                            q[r * n + j] -= d * q[r * n + k];
                        }
                    }
                    let nrm: f64 = (0..m).map(|r| q[r * n + j].powi(2)).sum::<f64>().sqrt();
                    for r in 0..m {
                        q[r * n + j] /= nrm;
                    }
                }
            }
            q
        };
        let u = ortho(rows, cols);
        let v = ortho(cols, cols);
        let mut a = vec![0.0; rows * cols];
        for r in 0..rows {
            for j in 0..cols {
                a[r * cols + j] =
                    (0..cols).map(|k| u[r * cols + k] * sigma[k] * v[j * cols + k]).sum();
            }
        }
        a
    }

    /// What the `A^T A` shortcut would have cost, and what [`svd`] does instead.
    ///
    /// The shortcut is recomputed here rather than kept in the library, so the
    /// comparison stays honest without shipping a second implementation that
    /// nothing should call.
    #[test]
    fn svd_holds_the_small_singular_values_the_shortcut_would_have_lost() {
        let (rows, cols) = (48usize, 5usize);
        let want = [1.0, 1e-3, 1e-5, 1e-7, 1e-9];
        let raw = graded(rows, cols, &want);
        let a = m(&raw, rows as i32, cols as i32);
        let got = svd(&a).unwrap();

        // Eigenvalues of `A^T A`, square-rooted. `eigen` is itself accurate, so
        // what this measures is the cost of forming the product.
        let mut gram = vec![0.0f64; cols * cols];
        for p in 0..cols {
            for q in 0..cols {
                gram[p * cols + q] =
                    (0..rows).map(|r| raw[r * cols + p] * raw[r * cols + q]).sum();
            }
        }
        let (vals, _) = eigen(&m(&gram, cols as i32, cols as i32)).unwrap();

        let rel = |x: f64, w: f64| (x - w).abs() / w;
        eprintln!("{:>10} {:>12} {:>12}", "true", "svd", "A^T A");
        for (i, w) in want.iter().enumerate() {
            let shortcut = vals.at_f64(i as i32, 0, 0).unwrap().max(0.0).sqrt();
            let ours = got.w.at_f64(i as i32, 0, 0).unwrap();
            eprintln!("{:>10.0e} {:>12.2e} {:>12.2e}", w, rel(ours, *w), rel(shortcut, *w));
        }

        for (i, w) in want.iter().enumerate() {
            let e = rel(got.w.at_f64(i as i32, 0, 0).unwrap(), *w);
            assert!(e < 1e-6, "svd lost sigma {i} ({w:.0e}): relative error {e:.2e}");
        }
        // And the shortcut does not, which is why it is not what `svd` uses.
        let last = vals.at_f64(cols as i32 - 1, 0, 0).unwrap().max(0.0).sqrt();
        assert!(
            rel(last, want[cols - 1]) > 1e-2,
            "the A^T A route was accurate after all ({:.2e}); if that holds up, \
             `svd` could go back to the cheaper method",
            rel(last, want[cols - 1])
        );
    }

    /// Upstream returns `min(samples, dimensions)` components, not `dimensions`.
    ///
    /// `pca.cpp`: `count = std::min(len, in_count)`, and when there are fewer
    /// samples than dimensions it takes the "scrambled" covariance `AA'`
    /// instead of `A'A` and back-transforms. With three samples in five
    /// dimensions, `cv2.PCACompute` returns a `3x5` matrix of axes; anything
    /// beyond the third is not determined by three points.
    ///
    /// Checked against a real `cv2 4.13.0`, which reports eigenvalues
    /// `[3.890, 0.5027, -1.99e-16]` for the same input — three of them.
    #[test]
    fn pca_returns_one_component_per_sample_when_samples_are_scarce() {
        // Three samples, five dimensions.
        let data = graded(3, 5, &[1.0, 0.5, 0.25, 0.125, 0.0625]);
        let mat = m(&data, 3, 5);
        let pca = pca_compute(&mat, DataLayout::Rows, 0).unwrap();
        assert_eq!(
            pca.eigenvectors.rows(),
            3,
            "three points span at most three directions; upstream returns \
             min(samples, dims) = 3, this returned {}",
            pca.eigenvectors.rows()
        );
        assert_eq!(pca.eigenvectors.cols(), 5, "each axis still lives in 5 dimensions");
        assert_eq!(pca.eigenvalues.rows(), 3);
    }

    /// PCA resolves small variances, which the covariance route could not.
    ///
    /// A principal value is a variance, `sigma^2 / n`, so the data's dynamic
    /// range enters *squared*. Building the covariance matrix and
    /// eigen-decomposing it — what upstream does, and what this used to do —
    /// therefore loses two digits for every one of range: measured, the
    /// relative error grew as `eps * (sigma_max / sigma_i)^2`, so a component
    /// six decades down had five good digits and one nine decades down had
    /// none at all.
    ///
    /// Going through the SVD of the centred data removes the squaring. The
    /// numbers still agree with `cv2.PCACompute2` wherever cv2 is itself
    /// accurate — checked in `examples/parity_check.rs` — and differ only where
    /// it is not. On the three-sample case in
    /// `pca_returns_one_component_per_sample_when_samples_are_scarce`, cv2
    /// reports a third eigenvalue of `-1.99e-16`: a *negative* variance, which
    /// is the covariance route's rounding showing through. This returns
    /// `+4.3e-33`.
    #[test]
    fn pca_resolves_variances_the_covariance_route_would_have_lost() {
        let (rows, cols) = (64usize, 5usize);
        let sigma = [1.0, 1e-2, 1e-4, 1e-6, 1e-9];
        // Already centred, so PCA's own centring is a no-op and the spectrum it
        // sees is the one built here.
        let mut data = graded(rows, cols, &sigma);
        for j in 0..cols {
            let mean: f64 = (0..rows).map(|r| data[r * cols + j]).sum::<f64>() / rows as f64;
            for r in 0..rows {
                data[r * cols + j] -= mean;
            }
        }
        let mat = m(&data, rows as i32, cols as i32);

        // The truth, from the same accurate decomposition verified against
        // known singular values in the test above.
        let direct = crate::linalg2::svd_decomp(&mat).unwrap();
        let truth: Vec<f64> = (0..cols)
            .map(|i| direct.w.at_f64(i as i32, 0, 0).unwrap().powi(2) / rows as f64)
            .collect();

        // The route this used to take, recomputed here so the comparison stays
        // honest without shipping an implementation nothing should call.
        let (cov, _) = calc_covar_matrix(&mat, DataLayout::Rows, true).unwrap();
        let (old_vals, _) = eigen(&cov).unwrap();

        let pca = pca_compute(&mat, DataLayout::Rows, 0).unwrap();
        eprintln!("{:>12} {:>14} {:>12} {:>12}", "sigma/sigma0", "variance", "now", "covariance");
        for (i, t) in truth.iter().enumerate() {
            let now = (pca.eigenvalues.at_f64(i as i32, 0, 0).unwrap() - t).abs() / t;
            let was = (old_vals.at_f64(i as i32, 0, 0).unwrap() - t).abs() / t;
            eprintln!("{:>12.1e} {:>14.4e} {:>12.2e} {:>12.2e}", (t / truth[0]).sqrt(), t, now, was);
        }

        for (i, t) in truth.iter().enumerate() {
            let e = (pca.eigenvalues.at_f64(i as i32, 0, 0).unwrap() - t).abs() / t;
            assert!(e < 1e-9, "component {i} came out with relative error {e:.2e}");
        }
        // And the covariance route does not, which is why it is not used.
        let last = (old_vals.at_f64(cols as i32 - 1, 0, 0).unwrap() - truth[cols - 1]).abs()
            / truth[cols - 1];
        assert!(
            last > 1e-3,
            "the covariance route resolved the smallest variance after all \
             ({last:.2e}); if that holds up, PCA could go back to it"
        );
    }

    #[test]
    fn svd_reconstructs_the_original() {
        let a = m(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let s = svd(&a).unwrap();
        for i in 0..3 {
            for j in 0..2 {
                let mut acc = 0.0;
                for r in 0..2 {
                    acc += s.u.at_f64(i, r, 0).unwrap()
                        * s.w.at_f64(r, 0, 0).unwrap()
                        * s.vt.at_f64(r, j, 0).unwrap();
                }
                let want = a.at_f64(i, j, 0).unwrap();
                assert!((acc - want).abs() < 1e-9, "[{i}][{j}]: {acc} vs {want}");
            }
        }
        // Singular values are non-negative and descending.
        assert!(s.w.at_f64(0, 0, 0).unwrap() >= s.w.at_f64(1, 0, 0).unwrap());
        assert!(s.w.at_f64(1, 0, 0).unwrap() >= 0.0);
    }

    #[test]
    fn covariance_of_a_known_set() {
        // Two samples, two dims: (0,0) and (2,4). Mean (1,2); the normalised
        // covariance is [[1,2],[2,4]].
        let a = m(&[0.0, 0.0, 2.0, 4.0], 2, 2);
        let (c, mu) = calc_covar_matrix(&a, DataLayout::Rows, true).unwrap();
        assert_eq!(mu.at_f64(0, 0, 0).unwrap(), 1.0);
        assert_eq!(mu.at_f64(0, 1, 0).unwrap(), 2.0);
        assert_eq!(c.at_f64(0, 0, 0).unwrap(), 1.0);
        assert_eq!(c.at_f64(0, 1, 0).unwrap(), 2.0);
        assert_eq!(c.at_f64(1, 1, 0).unwrap(), 4.0);
    }

    #[test]
    fn pca_finds_the_axis_of_a_line_and_roundtrips() {
        // Points along y = 2x carry all their variance on one axis.
        let pts: Vec<f64> = (0..10).flat_map(|i| [i as f64, 2.0 * i as f64]).collect();
        let data = m(&pts, 10, 2);
        let p = pca_compute(&data, DataLayout::Rows, 1).unwrap();
        assert_eq!(p.eigenvectors.rows(), 1);
        let (ex, ey) = (
            p.eigenvectors.at_f64(0, 0, 0).unwrap(),
            p.eigenvectors.at_f64(0, 1, 0).unwrap(),
        );
        // The axis is parallel to (1, 2)/sqrt(5).
        assert!((ey / ex - 2.0).abs() < 1e-9, "slope {}", ey / ex);
        let proj = p.project(&data).unwrap();
        let back = p.back_project(&proj).unwrap();
        for i in 0..10 {
            for j in 0..2 {
                let want = pts[(i * 2 + j) as usize];
                assert!((back.at_f64(i, j, 0).unwrap() - want).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn mahalanobis_with_identity_is_euclidean() {
        let a = m(&[0.0, 0.0], 1, 2);
        let b = m(&[3.0, 4.0], 1, 2);
        let i = m(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        assert!((mahalanobis(&a, &b, &i).unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn kmeans_separates_two_far_apart_blobs() {
        let mut pts = Vec::new();
        for i in 0..20 {
            pts.push(i as f64 * 0.01);
            pts.push(i as f64 * 0.01);
        }
        for i in 0..20 {
            pts.push(100.0 + i as f64 * 0.01);
            pts.push(100.0 + i as f64 * 0.01);
        }
        let data = m(&pts, 40, 2);
        let mut rng = crate::rng::Rng::new(12345);
        let r = kmeans(&data, 2, 3, 50, 1e-6, KMeansFlags::PpCenters, None, &mut rng).unwrap();
        // Every point in the first blob shares a label, likewise the second,
        // and the two labels differ.
        let a = r.labels[0];
        let b = r.labels[20];
        assert_ne!(a, b, "the two blobs must land in different clusters");
        assert!(r.labels[..20].iter().all(|&l| l == a));
        assert!(r.labels[20..].iter().all(|&l| l == b));
    }

    #[test]
    fn every_kmeans_seeding_finds_the_same_two_blobs() {
        let mut pts = Vec::new();
        for i in 0..20 {
            pts.push(i as f64 * 0.01);
            pts.push(i as f64 * 0.01);
        }
        for i in 0..20 {
            pts.push(100.0 + i as f64 * 0.01);
            pts.push(100.0 + i as f64 * 0.01);
        }
        let data = m(&pts, 40, 2);
        // A deliberately wrong starting labelling: the split is at 10, not 20.
        let initial: Vec<i32> = (0..40).map(|i| i32::from(i >= 10)).collect();
        for (flags, labels) in [
            (KMeansFlags::PpCenters, None),
            (KMeansFlags::RandomCenters, None),
            (KMeansFlags::UseInitialLabels, Some(&initial[..])),
        ] {
            let mut rng = crate::rng::Rng::new(12345);
            let r = kmeans(&data, 2, 3, 50, 1e-6, flags, labels, &mut rng).unwrap();
            let (a, b) = (r.labels[0], r.labels[20]);
            assert_ne!(a, b, "{flags:?} merged the blobs");
            assert!(r.labels[..20].iter().all(|&l| l == a), "{flags:?}");
            assert!(r.labels[20..].iter().all(|&l| l == b), "{flags:?}");
        }
    }

    #[test]
    fn initial_labels_are_required_and_validated() {
        let data = m(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], 3, 2);
        let mut rng = crate::rng::Rng::new(1);
        let f = KMeansFlags::UseInitialLabels;
        assert!(kmeans(&data, 2, 1, 10, 1e-6, f, None, &mut rng).is_err());
        // Wrong length, and a label outside 0..k.
        assert!(kmeans(&data, 2, 1, 10, 1e-6, f, Some(&[0, 1]), &mut rng).is_err());
        assert!(kmeans(&data, 2, 1, 10, 1e-6, f, Some(&[0, 1, 2]), &mut rng).is_err());
    }
}
