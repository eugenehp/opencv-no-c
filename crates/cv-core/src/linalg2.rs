//! One-sided Jacobi SVD, back-substitution, PCA projection, PSNR and the
//! simplex solver (`modules/core/src/lapack.cpp`, `pca.cpp`, `lpsolver.cpp`).
//!
//! # Why a second SVD
//!
//! [`crate::linalg::svd`] eigen-decomposes `A^T A`, which **squares the
//! condition number**: a matrix with singular values spanning 1e8 loses the
//! small ones entirely, because `A^T A` spans 1e16 and that is past `f64`'s
//! resolution. Upstream works on `A` directly and does not have that problem.
//!
//! [`svd_decomp`] here is the one-sided Jacobi method, which also works on `A`
//! directly: it orthogonalises pairs of columns by plane rotations until they
//! are mutually orthogonal, at which point the column norms *are* the singular
//! values. Accuracy is relative to each singular value rather than to the
//! largest one, so the small ones survive.

use crate::types::CV_64FC1;
use crate::{cv_assert, Mat, MatLike, MatLikeMut, Result};

/// A thin SVD: `A = U * diag(w) * Vt`.
#[derive(Debug)]
pub struct SvdResult {
    /// Left singular vectors as columns, `m x k`.
    pub u: Mat,
    /// Singular values, descending, `k x 1`.
    pub w: Mat,
    /// Right singular vectors as rows, `k x n`.
    pub vt: Mat,
}

/// `cv::SVDecomp(src, w, u, vt, flags)` — one-sided Jacobi.
///
/// Accurate on badly conditioned input, unlike [`crate::linalg::svd`]; prefer
/// this one when the matrix might be near-singular.
pub fn svd_decomp<S: MatLike + ?Sized>(src: &S) -> Result<SvdResult> {
    let (m, n) = (src.rows() as usize, src.cols() as usize);
    cv_assert!(m > 0 && n > 0, "SVDecomp requires a non-empty matrix");
    cv_assert!(src.channels() == 1, "SVDecomp requires a single-channel matrix");

    // Work on the tall orientation; the thin case is recovered by transposing.
    let transposed = m < n;
    let (rows, cols) = if transposed { (n, m) } else { (m, n) };
    let mut a = vec![0.0f64; rows * cols];
    for i in 0..m {
        for j in 0..n {
            let v = src.at_f64(i as i32, j as i32, 0)?;
            if transposed {
                a[j * cols + i] = v;
            } else {
                a[i * cols + j] = v;
            }
        }
    }
    // V accumulates the rotations.
    let mut v = vec![0.0f64; cols * cols];
    for i in 0..cols {
        v[i * cols + i] = 1.0;
    }

    // Sweep until every pair of columns is orthogonal.
    let eps = f64::EPSILON;
    for _ in 0..60 {
        let mut changed = false;
        for p in 0..cols {
            for q in (p + 1)..cols {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for r in 0..rows {
                    let (x, y) = (a[r * cols + p], a[r * cols + q]);
                    alpha += x * x;
                    beta += y * y;
                    gamma += x * y;
                }
                if gamma.abs() <= eps * (alpha * beta).sqrt() || gamma == 0.0 {
                    continue;
                }
                changed = true;
                // The rotation that zeroes this pair's inner product.
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for r in 0..rows {
                    let (x, y) = (a[r * cols + p], a[r * cols + q]);
                    a[r * cols + p] = c * x - s * y;
                    a[r * cols + q] = s * x + c * y;
                }
                for r in 0..cols {
                    let (x, y) = (v[r * cols + p], v[r * cols + q]);
                    v[r * cols + p] = c * x - s * y;
                    v[r * cols + q] = s * x + c * y;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Column norms are the singular values; sort them descending.
    let mut order: Vec<(f64, usize)> = (0..cols)
        .map(|j| {
            let nrm = (0..rows).map(|r| a[r * cols + j].powi(2)).sum::<f64>().sqrt();
            (nrm, j)
        })
        .collect();
    order.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));

    let k = cols;
    let mut u = vec![0.0f64; rows * k];
    let mut w = vec![0.0f64; k];
    let mut vt = vec![0.0f64; k * cols];
    for (slot, &(nrm, j)) in order.iter().enumerate() {
        w[slot] = nrm;
        for r in 0..rows {
            // A zero singular value leaves an arbitrary direction; leaving it
            // zero is honest, and callers filter on `w` anyway.
            u[r * k + slot] = if nrm > 0.0 { a[r * cols + j] / nrm } else { 0.0 };
        }
        for r in 0..cols {
            vt[slot * cols + r] = v[r * cols + j];
        }
    }

    // Undo the transpose: A' = V W U' means U and Vt swap roles.
    let (uu, vv) = if transposed {
        (transpose_of(&vt, k, cols), transpose_of(&u, rows, k))
    } else {
        (mat_from(&u, rows, k)?, mat_from(&vt, k, cols)?)
    };
    Ok(SvdResult { u: uu, w: mat_from(&w, k, 1)?, vt: vv })
}

fn mat_from(d: &[f64], r: usize, c: usize) -> Result<Mat> {
    Mat::from_vec(r as i32, c as i32, 1, d)
}

fn transpose_of(d: &[f64], r: usize, c: usize) -> Mat {
    let mut o = vec![0.0f64; r * c];
    for i in 0..r {
        for j in 0..c {
            o[j * r + i] = d[i * c + j];
        }
    }
    Mat::from_vec(c as i32, r as i32, 1, &o).expect("transpose has the same element count")
}

/// `cv::SVBackSubst(w, u, vt, rhs, dst)`.
///
/// Solves `A x = rhs` from a decomposition of `A`, discarding directions whose
/// singular value is negligible — which is what makes it work on a singular or
/// near-singular system where a direct solve would divide by ~0 and return
/// noise.
pub fn sv_back_subst<R: MatLike + ?Sized>(svd: &SvdResult, rhs: &R) -> Result<Mat> {
    let k = svd.w.rows() as usize;
    let m = svd.u.rows() as usize;
    let n = svd.vt.cols() as usize;
    cv_assert!(
        rhs.rows() as usize == m,
        "the right-hand side has {} rows, expected {m}",
        rhs.rows()
    );
    let nrhs = rhs.cols() as usize;

    // Threshold below which a direction is treated as null, as upstream does.
    let wmax = (0..k).map(|i| svd.w.at_f64(i as i32, 0, 0).unwrap_or(0.0)).fold(0.0, f64::max);
    let tol = f64::EPSILON * (m.max(n) as f64) * wmax;

    let mut out = Mat::zeros(n as i32, nrhs as i32, CV_64FC1)?;
    for c in 0..nrhs {
        // t = U^T rhs, scaled by 1/w.
        let mut t = vec![0.0f64; k];
        for (i, ti) in t.iter_mut().enumerate() {
            let wi = svd.w.at_f64(i as i32, 0, 0)?;
            if wi <= tol {
                continue;
            }
            let mut s = 0.0;
            for r in 0..m {
                s += svd.u.at_f64(r as i32, i as i32, 0)? * rhs.at_f64(r as i32, c as i32, 0)?;
            }
            *ti = s / wi;
        }
        // x = V t
        for j in 0..n {
            let mut s = 0.0;
            for (i, ti) in t.iter().enumerate() {
                s += svd.vt.at_f64(i as i32, j as i32, 0)? * ti;
            }
            out.pixel_mut::<f64>(j as i32, c as i32)?[0] = s;
        }
    }
    Ok(out)
}

/// `cv::PCAProject(data, mean, eigenvectors, result)`.
///
/// Subtracts the mean and projects onto the principal axes.
pub fn pca_project<D: MatLike + ?Sized, M: MatLike + ?Sized, E: MatLike + ?Sized>(
    data: &D,
    mean: &M,
    eigenvectors: &E,
) -> Result<Mat> {
    let dim = eigenvectors.cols();
    let comps = eigenvectors.rows();
    cv_assert!(
        data.cols() == dim,
        "data has {} columns but the axes have {dim}",
        data.cols()
    );
    cv_assert!(mean.cols() == dim, "the mean has {} columns, expected {dim}", mean.cols());
    let mut out = Mat::zeros(data.rows(), comps, CV_64FC1)?;
    for r in 0..data.rows() {
        for c in 0..comps {
            let mut s = 0.0;
            for j in 0..dim {
                s += (data.at_f64(r, j, 0)? - mean.at_f64(0, j, 0)?)
                    * eigenvectors.at_f64(c, j, 0)?;
            }
            out.pixel_mut::<f64>(r, c)?[0] = s;
        }
    }
    Ok(out)
}

/// `cv::PCABackProject(data, mean, eigenvectors, result)`.
///
/// The inverse of [`pca_project`]. Exact only when every axis was kept; with
/// fewer components it returns the best reconstruction available from them,
/// which is the point of the reduction.
pub fn pca_back_project<D: MatLike + ?Sized, M: MatLike + ?Sized, E: MatLike + ?Sized>(
    data: &D,
    mean: &M,
    eigenvectors: &E,
) -> Result<Mat> {
    let dim = eigenvectors.cols();
    let comps = eigenvectors.rows();
    cv_assert!(
        data.cols() == comps,
        "data has {} columns but there are {comps} components",
        data.cols()
    );
    let mut out = Mat::zeros(data.rows(), dim, CV_64FC1)?;
    for r in 0..data.rows() {
        for j in 0..dim {
            let mut s = mean.at_f64(0, j, 0)?;
            for c in 0..comps {
                s += data.at_f64(r, c, 0)? * eigenvectors.at_f64(c, j, 0)?;
            }
            out.pixel_mut::<f64>(r, j)?[0] = s;
        }
    }
    Ok(out)
}

/// `cv::PSNR(src1, src2, R)` — peak signal-to-noise ratio in decibels.
///
/// `r` is the maximum representable value, 255 for 8-bit. Identical inputs have
/// no error and infinite ratio; upstream returns `INFINITY` there and so does
/// this, rather than a large finite number that would look like a measurement.
pub fn psnr<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B, r: f64) -> Result<f64> {
    cv_assert!(
        a.rows() == b.rows() && a.cols() == b.cols() && a.channels() == b.channels(),
        "PSNR needs matching shapes, got {}x{}x{} and {}x{}x{}",
        a.rows(),
        a.cols(),
        a.channels(),
        b.rows(),
        b.cols(),
        b.channels()
    );
    cv_assert!(r.partial_cmp(&0.0) == Some(std::cmp::Ordering::Greater), "the peak value must be positive and not NaN, got {r}");
    let mut sse = 0.0f64;
    let mut count = 0usize;
    for y in 0..a.rows() {
        for x in 0..a.cols() {
            for c in 0..a.channels() {
                let d = a.at_f64(y, x, c)? - b.at_f64(y, x, c)?;
                sse += d * d;
                count += 1;
            }
        }
    }
    if sse <= 0.0 {
        return Ok(f64::INFINITY);
    }
    let mse = sse / count as f64;
    Ok(10.0 * (r * r / mse).log10())
}

/// What [`solve_lp`] concluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpResult {
    /// `cv::SOLVELP_SINGLE` — one optimal solution.
    Single,
    /// `cv::SOLVELP_MULTI` — optimal, but not uniquely.
    Multi,
    /// `cv::SOLVELP_UNBOUNDED` — the objective can grow without limit.
    Unbounded,
    /// `cv::SOLVELP_UNFEASIBLE` — the constraints cannot all be met.
    Unfeasible,
}

/// `cv::solveLP(Func, Constr, z)` — maximise `func . x` subject to
/// `constr * x <= b`, `x >= 0`.
///
/// `constr` is the constraint matrix with the bound `b` as its **last column**,
/// matching upstream's packing. Simplex with Bland's rule, which trades a
/// little speed for a guarantee against cycling on degenerate vertices.
pub fn solve_lp<F: MatLike + ?Sized, C: MatLike + ?Sized>(
    func: &F,
    constr: &C,
) -> Result<(LpResult, Vec<f64>)> {
    let n = (func.rows() * func.cols()) as usize;
    cv_assert!(n > 0, "the objective must have at least one variable");
    let m = constr.rows() as usize;
    cv_assert!(m > 0, "there must be at least one constraint");
    cv_assert!(
        constr.cols() as usize == n + 1,
        "the constraint matrix needs {} columns (variables plus the bound), got {}",
        n + 1,
        constr.cols()
    );

    let objective = |i: usize| -> Result<f64> {
        if func.rows() == 1 {
            func.at_f64(0, i as i32, 0)
        } else {
            func.at_f64(i as i32, 0, 0)
        }
    };

    // Tableau: m constraint rows plus one objective row; n variables plus m
    // slacks plus the right-hand side.
    let width = n + m + 1;
    let mut t = vec![0.0f64; (m + 1) * width];
    for i in 0..m {
        for j in 0..n {
            t[i * width + j] = constr.at_f64(i as i32, j as i32, 0)?;
        }
        t[i * width + n + i] = 1.0;
        let b = constr.at_f64(i as i32, n as i32, 0)?;
        if b < 0.0 {
            // A negative bound means the origin is not feasible; this simplex
            // starts there, so it cannot say anything useful.
            return Ok((LpResult::Unfeasible, vec![0.0; n]));
        }
        t[i * width + width - 1] = b;
    }
    for j in 0..n {
        t[m * width + j] = -objective(j)?;
    }

    let mut basis: Vec<usize> = (0..m).map(|i| n + i).collect();
    for _ in 0..10_000 {
        // Bland's rule: the lowest-indexed improving column.
        let Some(col) = (0..(n + m)).find(|&j| t[m * width + j] < -1e-12) else {
            break;
        };
        // Tightest ratio picks the leaving row; ties go to the lowest basis index.
        let mut best: Option<(f64, usize)> = None;
        for i in 0..m {
            let a = t[i * width + col];
            if a > 1e-12 {
                let ratio = t[i * width + width - 1] / a;
                let better = match best {
                    None => true,
                    Some((r, bi)) => ratio < r - 1e-12 || (ratio <= r + 1e-12 && basis[i] < basis[bi]),
                };
                if better {
                    best = Some((ratio, i));
                }
            }
        }
        let Some((_, row)) = best else {
            return Ok((LpResult::Unbounded, vec![0.0; n]));
        };
        // Pivot.
        let piv = t[row * width + col];
        for j in 0..width {
            t[row * width + j] /= piv;
        }
        for i in 0..=m {
            if i == row {
                continue;
            }
            let f = t[i * width + col];
            if f == 0.0 {
                continue;
            }
            for j in 0..width {
                t[i * width + j] -= f * t[row * width + j];
            }
        }
        basis[row] = col;
    }

    let mut x = vec![0.0f64; n];
    for (i, &b) in basis.iter().enumerate() {
        if b < n {
            x[b] = t[i * width + width - 1];
        }
    }
    // A zero reduced cost on a non-basic column means another optimum exists.
    let multi = (0..(n + m))
        .any(|j| !basis.contains(&j) && t[m * width + j].abs() < 1e-12);
    Ok((if multi { LpResult::Multi } else { LpResult::Single }, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(r: i32, c: i32, d: &[f64]) -> Mat {
        Mat::from_vec(r, c, 1, d).unwrap()
    }

    #[test]
    fn svd_reconstructs_the_original() {
        let a = m(4, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 1.0, 4.0]);
        let s = svd_decomp(&a).unwrap();
        for i in 0..4 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += s.u.at_f64(i, k, 0).unwrap()
                        * s.w.at_f64(k, 0, 0).unwrap()
                        * s.vt.at_f64(k, j, 0).unwrap();
                }
                let want = a.at_f64(i, j, 0).unwrap();
                assert!((acc - want).abs() < 1e-9, "({i},{j}): {acc} vs {want}");
            }
        }
        // Singular values come back descending and non-negative.
        let w: Vec<f64> = (0..3).map(|i| s.w.at_f64(i, 0, 0).unwrap()).collect();
        assert!(w.windows(2).all(|p| p[0] >= p[1]), "not descending: {w:?}");
        assert!(w.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn svd_handles_the_wide_case() {
        let a = m(2, 5, &[1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0, 1.0, 5.0]);
        let s = svd_decomp(&a).unwrap();
        assert_eq!((s.u.rows(), s.u.cols()), (2, 2));
        assert_eq!((s.vt.rows(), s.vt.cols()), (2, 5));
        for i in 0..2 {
            for j in 0..5 {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += s.u.at_f64(i, k, 0).unwrap()
                        * s.w.at_f64(k, 0, 0).unwrap()
                        * s.vt.at_f64(k, j, 0).unwrap();
                }
                assert!((acc - a.at_f64(i, j, 0).unwrap()).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn svd_keeps_small_singular_values_the_normal_equations_would_lose() {
        // Condition number 1e8: A^T A spans 1e16, past f64's resolution, so the
        // squared-condition approach cannot see the smallest value.
        let a = m(3, 3, &[1.0, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-8]);
        let s = svd_decomp(&a).unwrap();
        let w: Vec<f64> = (0..3).map(|i| s.w.at_f64(i, 0, 0).unwrap()).collect();
        assert!((w[0] - 1.0).abs() < 1e-12, "{w:?}");
        assert!((w[1] - 1e-4).abs() < 1e-16, "{w:?}");
        // The point of the exercise: this one survives with full relative accuracy.
        assert!(
            (w[2] - 1e-8).abs() < 1e-20,
            "the smallest singular value should be 1e-8, got {}",
            w[2]
        );
    }

    #[test]
    fn svd_of_a_singular_matrix_reports_a_zero() {
        // Rank 1: two singular values must vanish.
        let a = m(3, 3, &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0]);
        let s = svd_decomp(&a).unwrap();
        let w: Vec<f64> = (0..3).map(|i| s.w.at_f64(i, 0, 0).unwrap()).collect();
        assert!(w[0] > 10.0, "{w:?}");
        assert!(w[1] < 1e-10 && w[2] < 1e-10, "rank 1 leaves two zeros: {w:?}");
    }

    #[test]
    fn back_substitution_solves_a_well_posed_system() {
        let a = m(3, 3, &[2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0]);
        let b = m(3, 1, &[8.0, -11.0, -3.0]);
        let s = svd_decomp(&a).unwrap();
        let x = sv_back_subst(&s, &b).unwrap();
        // The textbook answer is (2, 3, -1).
        for (i, want) in [2.0, 3.0, -1.0].into_iter().enumerate() {
            let got = x.at_f64(i as i32, 0, 0).unwrap();
            assert!((got - want).abs() < 1e-9, "x[{i}] = {got}, expected {want}");
        }
    }

    #[test]
    fn back_substitution_gives_the_least_squares_answer_when_singular() {
        // Rank-deficient: a direct solve would divide by ~0.
        let a = m(3, 2, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let b = m(3, 1, &[2.0, 2.0, 2.0]);
        let s = svd_decomp(&a).unwrap();
        let x = sv_back_subst(&s, &b).unwrap();
        // Any x with x0 + x1 = 2 fits; the minimum-norm one is (1, 1).
        let (x0, x1) = (x.at_f64(0, 0, 0).unwrap(), x.at_f64(1, 0, 0).unwrap());
        assert!((x0 + x1 - 2.0).abs() < 1e-9, "got ({x0}, {x1})");
        assert!(x0.is_finite() && x1.is_finite(), "must not divide by a zero singular value");
    }

    #[test]
    fn pca_projection_round_trips_when_every_axis_is_kept() {
        let data = m(5, 3, &[
            2.0, 1.0, 0.5, 3.0, 2.5, 1.0, 4.0, 3.5, 2.0, 5.5, 5.0, 3.0, 7.0, 6.5, 4.5,
        ]);
        let pca = crate::linalg::pca_compute(&data, crate::linalg::DataLayout::Rows, 0).unwrap();
        let proj = pca_project(&data, &pca.mean, &pca.eigenvectors).unwrap();
        let back = pca_back_project(&proj, &pca.mean, &pca.eigenvectors).unwrap();
        for r in 0..5 {
            for c in 0..3 {
                let (a, b) = (data.at_f64(r, c, 0).unwrap(), back.at_f64(r, c, 0).unwrap());
                assert!((a - b).abs() < 1e-8, "({r},{c}): {a} vs {b}");
            }
        }
    }

    #[test]
    fn pca_validates_its_shapes() {
        let data = m(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mean = m(1, 3, &[2.5, 3.5, 4.5]);
        let axes = m(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert!(pca_project(&data, &mean, &axes).is_ok());
        let wrong = m(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(pca_project(&wrong, &mean, &axes).is_err());
        assert!(pca_back_project(&wrong, &mean, &axes).is_ok(), "2 columns == 2 components");
        let bad = m(2, 3, &[0.0; 6]);
        assert!(pca_back_project(&bad, &mean, &axes).is_err());
    }

    #[test]
    fn psnr_is_infinite_for_identical_images_and_finite_otherwise() {
        let a = Mat::from_vec(2, 2, 1, &[10u8, 20, 30, 40]).unwrap();
        assert_eq!(psnr(&a, &a, 255.0).unwrap(), f64::INFINITY);

        let b = Mat::from_vec(2, 2, 1, &[11u8, 21, 31, 41]).unwrap();
        // Every pixel off by one: MSE = 1, so PSNR = 10*log10(255^2) = 48.13 dB.
        let p = psnr(&a, &b, 255.0).unwrap();
        assert!((p - 48.130_803_608_679_04).abs() < 1e-9, "got {p}");

        // Mismatched shapes and a non-positive peak are errors.
        let c = Mat::from_vec(2, 3, 1, &[0u8; 6]).unwrap();
        assert!(psnr(&a, &c, 255.0).is_err());
        assert!(psnr(&a, &b, 0.0).is_err());
    }

    #[test]
    fn solve_lp_finds_a_known_optimum() {
        // Maximise 3x + 2y subject to x + y <= 4, x + 3y <= 6, x <= 3.
        // The optimum is at (3, 1) with value 11.
        let f = m(1, 2, &[3.0, 2.0]);
        let c = m(3, 3, &[1.0, 1.0, 4.0, 1.0, 3.0, 6.0, 1.0, 0.0, 3.0]);
        let (res, x) = solve_lp(&f, &c).unwrap();
        assert!(matches!(res, LpResult::Single | LpResult::Multi), "got {res:?}");
        assert!((x[0] - 3.0).abs() < 1e-9 && (x[1] - 1.0).abs() < 1e-9, "got {x:?}");
        let value = 3.0 * x[0] + 2.0 * x[1];
        assert!((value - 11.0).abs() < 1e-9, "objective {value}, expected 11");
    }

    #[test]
    fn solve_lp_detects_an_unbounded_objective() {
        // Maximise x with no upper bound on it.
        let f = m(1, 2, &[1.0, 0.0]);
        let c = m(1, 3, &[0.0, 1.0, 5.0]);
        let (res, _) = solve_lp(&f, &c).unwrap();
        assert_eq!(res, LpResult::Unbounded);
    }

    #[test]
    fn solve_lp_reports_an_infeasible_system() {
        // A negative bound puts the origin outside the feasible region.
        let f = m(1, 1, &[1.0]);
        let c = m(1, 2, &[1.0, -5.0]);
        assert_eq!(solve_lp(&f, &c).unwrap().0, LpResult::Unfeasible);
    }

    #[test]
    fn solve_lp_validates_its_input() {
        let f = m(1, 2, &[1.0, 1.0]);
        // The constraint matrix must carry the bound as an extra column.
        let bad = m(1, 2, &[1.0, 1.0]);
        assert!(solve_lp(&f, &bad).is_err());
        let empty = Mat::zeros(0, 3, CV_64FC1).unwrap();
        assert!(solve_lp(&f, &empty).is_err());
    }

    /// Values recorded from cv2 4.13.0.
    #[test]
    fn core_algorithms_match_cv2() {
        // SVDecomp
        let a = m(4, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 1.0, 4.0]);
        let w: Vec<f64> = (0..3)
            .map(|i| svd_decomp(&a).unwrap().w.at_f64(i, 0, 0).unwrap())
            .collect();
        for (got, want) in w
            .iter()
            .zip([17.927_632_318_525_827, 1.712_101_964_122_515, 0.817_744_652_196_638_7])
        {
            assert!((got - want).abs() < 1e-12, "singular values {w:?}");
        }

        // The ill-conditioned case cv2 also gets exactly right.
        let b = m(3, 3, &[1.0, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-8]);
        let w2: Vec<f64> =
            (0..3).map(|i| svd_decomp(&b).unwrap().w.at_f64(i, 0, 0).unwrap()).collect();
        assert_eq!(w2, vec![1.0, 1e-4, 1e-8], "cv2 reports exactly these");

        // PSNR — bit-exact against cv2.
        let p = psnr(
            &Mat::from_vec(2, 2, 1, &[10u8, 20, 30, 40]).unwrap(),
            &Mat::from_vec(2, 2, 1, &[11u8, 21, 31, 41]).unwrap(),
            255.0,
        )
        .unwrap();
        assert_eq!(p, 48.130_803_608_679_1, "cv2 gives exactly this");

        // solveLP — cv2 returns SOLVELP_SINGLE with z = (3, 1).
        let (r, z) = solve_lp(&m(1, 2, &[3.0, 2.0]), &m(3, 3, &[1.0, 1.0, 4.0, 1.0, 3.0, 6.0, 1.0, 0.0, 3.0]))
            .unwrap();
        assert_eq!(r, LpResult::Single);
        assert_eq!(z, vec![3.0, 1.0]);
    }
}
