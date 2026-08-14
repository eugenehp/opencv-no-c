//! Dense linear algebra (`modules/core/src/matmul.dispatch.cpp`, `lapack.cpp`).
//!
//! Everything here is single-channel `CV_32F` or `CV_64F`, matching upstream's
//! restrictions. Decompositions are computed in `f64` regardless of the input
//! depth — upstream does the same, because a `CV_32F` LU factorisation loses
//! too much to be useful for `invert`/`solve`.

use crate::error::{Code, Result};
use crate::mat::{Mat, MatLike, MatLikeMut};
use crate::types::{Depth, MatType};
use crate::{cv_assert, cv_error};

/// `cv::DecompTypes`, with upstream's values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Decomp {
    /// `DECOMP_LU` — Gaussian elimination with partial pivoting.
    Lu = 0,
    /// `DECOMP_SVD` — singular value decomposition.
    ///
    /// The only method that accepts a singular or non-square matrix: it
    /// produces the pseudo-inverse, so an over-determined system is solved in
    /// the least-squares sense.
    Svd = 1,
    /// `DECOMP_EIG` — eigenvalue decomposition, symmetric matrices only.
    Eig = 2,
    /// `DECOMP_CHOLESKY` — for symmetric positive-definite systems only.
    Cholesky = 3,
    /// `DECOMP_QR` — Householder QR, also usable on over-determined systems.
    Qr = 4,
}

/// `DECOMP_NORMAL` — solve the normal equations `AᵀA x = Aᵀb` instead of `A x = b`.
///
/// Upstream or-s this into the method argument; here it selects
/// [`solve_normal`] instead, because a Rust enum cannot carry the extra bit.
pub const DECOMP_NORMAL: i32 = 16;

/// Reject anything that is not a single-channel float matrix.
fn require_float_1ch<M: MatLike + ?Sized>(m: &M, what: &str) -> Result<()> {
    if m.channels() != 1 {
        return cv_error!(Code::StsUnsupportedFormat, "{what} requires a single-channel matrix");
    }
    if !matches!(m.depth(), Depth::F32 | Depth::F64) {
        return cv_error!(
            Code::StsUnsupportedFormat,
            "{what} requires CV_32F or CV_64F, got {}",
            m.depth().name()
        );
    }
    Ok(())
}

/// Read a single-channel matrix into a dense row-major `f64` buffer.
fn to_f64<M: MatLike + ?Sized>(m: &M) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(m.total());
    for y in 0..m.rows() {
        for x in 0..m.cols() {
            out.push(m.at_f64(y, x, 0)?);
        }
    }
    Ok(out)
}

/// Store a dense `f64` buffer back into a fresh matrix of the given depth.
fn from_f64(data: &[f64], rows: i32, cols: i32, depth: Depth) -> Result<Mat> {
    let mut dst = Mat::zeros(rows, cols, MatType::new(depth, 1))?;
    match depth {
        Depth::F32 => {
            for y in 0..rows {
                let d = dst.row_as_mut::<f32>(y)?;
                for x in 0..cols as usize {
                    d[x] = data[y as usize * cols as usize + x] as f32;
                }
            }
        }
        _ => {
            for y in 0..rows {
                let d = dst.row_as_mut::<f64>(y)?;
                for x in 0..cols as usize {
                    d[x] = data[y as usize * cols as usize + x];
                }
            }
        }
    }
    Ok(dst)
}

/// `cv::GemmFlags`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GemmFlags {
    /// `GEMM_1_T` — transpose the first operand.
    pub transpose_a: bool,
    /// `GEMM_2_T` — transpose the second operand.
    pub transpose_b: bool,
    /// `GEMM_3_T` — transpose the third operand.
    pub transpose_c: bool,
}

/// `cv::gemm(src1, src2, alpha, src3, beta, dst, flags)`.
///
/// Computes `alpha * op(A) * op(B) + beta * op(C)`. Pass `None` for `c` to skip
/// the accumulation term.
pub fn gemm<A: MatLike + ?Sized, B: MatLike + ?Sized, C: MatLike + ?Sized>(
    a: &A,
    b: &B,
    alpha: f64,
    c: Option<&C>,
    beta: f64,
    flags: GemmFlags,
) -> Result<Mat> {
    require_float_1ch(a, "gemm")?;
    require_float_1ch(b, "gemm")?;
    if a.depth() != b.depth() {
        return cv_error!(Code::StsUnmatchedFormats, "gemm operands must share a depth");
    }

    let (ar, ac) = dims(a, flags.transpose_a);
    let (br, bc) = dims(b, flags.transpose_b);
    if ac != br {
        return cv_error!(
            Code::StsUnmatchedSizes,
            "gemm inner dimensions disagree: {ar}x{ac} times {br}x{bc}"
        );
    }

    let av = to_f64(a)?;
    let bv = to_f64(b)?;
    let (a_cols, b_cols) = (a.cols() as usize, b.cols() as usize);
    let at = |i: usize, j: usize| {
        if flags.transpose_a {
            av[j * a_cols + i]
        } else {
            av[i * a_cols + j]
        }
    };
    let bt = |i: usize, j: usize| {
        if flags.transpose_b {
            bv[j * b_cols + i]
        } else {
            bv[i * b_cols + j]
        }
    };

    let mut out = vec![0.0f64; ar as usize * bc as usize];
    for i in 0..ar as usize {
        for k in 0..ac as usize {
            let aik = at(i, k);
            if aik == 0.0 {
                continue;
            }
            for j in 0..bc as usize {
                out[i * bc as usize + j] += aik * bt(k, j);
            }
        }
    }
    for v in out.iter_mut() {
        *v *= alpha;
    }

    if let Some(cm) = c {
        require_float_1ch(cm, "gemm")?;
        let (cr, cc) = dims(cm, flags.transpose_c);
        if cr != ar || cc != bc {
            return cv_error!(
                Code::StsUnmatchedSizes,
                "gemm third operand is {cr}x{cc}, expected {ar}x{bc}"
            );
        }
        let cv = to_f64(cm)?;
        let c_cols = cm.cols() as usize;
        for i in 0..ar as usize {
            for j in 0..bc as usize {
                let v = if flags.transpose_c { cv[j * c_cols + i] } else { cv[i * c_cols + j] };
                out[i * bc as usize + j] += beta * v;
            }
        }
    }

    from_f64(&out, ar, bc, a.depth())
}

/// Effective dimensions of an operand, honouring its transpose flag.
fn dims<M: MatLike + ?Sized>(m: &M, transposed: bool) -> (i32, i32) {
    if transposed {
        (m.cols(), m.rows())
    } else {
        (m.rows(), m.cols())
    }
}

/// Plain matrix product — `A * B`, the common case of [`gemm`].
pub fn matmul<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B) -> Result<Mat> {
    gemm::<A, B, Mat>(a, b, 1.0, None, 0.0, GemmFlags::default())
}

/// `cv::setIdentity(mtx, s)`.
pub fn set_identity<M: MatLikeMut + ?Sized>(m: &mut M, value: f64) -> Result<()> {
    let (rows, cols, cn) = (m.rows(), m.cols(), m.channels());
    for y in 0..rows {
        for x in 0..cols {
            for c in 0..cn {
                let v = if x == y { value } else { 0.0 };
                set_one(m, y, x, c, v)?;
            }
        }
    }
    Ok(())
}

/// Depth-generic single-channel store for a `MatLikeMut`.
fn set_one<M: MatLikeMut + ?Sized>(m: &mut M, y: i32, x: i32, c: i32, v: f64) -> Result<()> {
    use crate::saturate::Elem;
    let i = c as usize;
    match m.depth() {
        Depth::U8 => m.pixel_mut::<u8>(y, x)?[i] = u8::from_f64(v),
        Depth::S8 => m.pixel_mut::<i8>(y, x)?[i] = i8::from_f64(v),
        Depth::U16 => m.pixel_mut::<u16>(y, x)?[i] = u16::from_f64(v),
        Depth::S16 => m.pixel_mut::<i16>(y, x)?[i] = i16::from_f64(v),
        Depth::S32 => m.pixel_mut::<i32>(y, x)?[i] = i32::from_f64(v),
        Depth::F32 => m.pixel_mut::<f32>(y, x)?[i] = v as f32,
        Depth::F64 => m.pixel_mut::<f64>(y, x)?[i] = v,
        Depth::F16 => return cv_error!(Code::StsNotImplemented, "CV_16F is not ported yet"),
    }
    Ok(())
}

/// `cv::trace(mtx)` — sum of the diagonal.
pub fn trace<M: MatLike + ?Sized>(m: &M) -> Result<f64> {
    let n = m.rows().min(m.cols());
    let mut s = 0.0;
    for i in 0..n {
        s += m.at_f64(i, i, 0)?;
    }
    Ok(s)
}

/// LU decomposition with partial pivoting, in place.
///
/// Returns the determinant of the factorisation (including the sign from row
/// swaps), or `None` if the matrix is singular to working precision.
fn lu_decompose(a: &mut [f64], n: usize, perm: &mut [usize]) -> Option<f64> {
    for (i, p) in perm.iter_mut().enumerate() {
        *p = i;
    }
    let mut det = 1.0f64;

    for col in 0..n {
        // Partial pivot: the largest magnitude entry in this column.
        let mut best = col;
        let mut best_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > best_val {
                best_val = v;
                best = row;
            }
        }
        if best_val < 1e-300 {
            return None;
        }
        if best != col {
            for k in 0..n {
                a.swap(col * n + k, best * n + k);
            }
            perm.swap(col, best);
            det = -det;
        }
        let pivot = a[col * n + col];
        det *= pivot;
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            a[row * n + col] = factor;
            for k in (col + 1)..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
        }
    }
    Some(det)
}

/// `cv::determinant(mtx)` — square, single-channel float matrices only.
pub fn determinant<M: MatLike + ?Sized>(m: &M) -> Result<f64> {
    require_float_1ch(m, "determinant")?;
    cv_assert!(m.rows() == m.cols(), "determinant requires a square matrix");
    let n = m.rows() as usize;
    let mut a = to_f64(m)?;
    let mut perm = vec![0usize; n];
    Ok(lu_decompose(&mut a, n, &mut perm).unwrap_or(0.0))
}

/// `cv::invert(src, dst, method)`.
///
/// Returns `(inverse, ok)`. When the matrix is singular the inverse is all
/// zeros and `ok` is false, mirroring upstream's zero return value.
///
/// `DECOMP_SVD` produces the Moore–Penrose pseudo-inverse and so is the one
/// method that accepts a non-square `src`; `DECOMP_QR` is rejected, exactly as
/// upstream rejects it here even though `solve` accepts it.
///
/// Upstream returns the *reciprocal condition number* from the SVD and EIG
/// paths rather than a flag. That number is available from [`crate::svd`]
/// directly, so this signature keeps the boolean it shares with the LU and
/// Cholesky paths: `ok` is false exactly where upstream would return zero.
pub fn invert<M: MatLike + ?Sized>(m: &M, method: Decomp) -> Result<(Mat, bool)> {
    require_float_1ch(m, "invert")?;
    if method == Decomp::Qr {
        return cv_error!(Code::StsBadArg, "invert does not support DECOMP_QR; use solve");
    }
    // Only the SVD path is defined for a non-square matrix.
    if method != Decomp::Svd {
        cv_assert!(m.rows() == m.cols(), "invert requires a square matrix");
    }
    let (rows, cols) = (m.rows() as usize, m.cols() as usize);
    let n = cols;

    let inv = match method {
        Decomp::Lu => invert_lu(&to_f64(m)?, n),
        Decomp::Cholesky => invert_cholesky(&to_f64(m)?, n),
        Decomp::Svd => invert_svd(&to_f64(m)?, rows, cols),
        Decomp::Eig => invert_eig(&to_f64(m)?, n),
        Decomp::Qr => unreachable!("rejected above"),
    };
    // The pseudo-inverse of an `m x n` matrix is `n x m`.
    let (orows, ocols) = (cols as i32, rows as i32);
    match inv {
        Some(data) => Ok((from_f64(&data, orows, ocols, m.depth())?, true)),
        None => Ok((Mat::zeros(orows, ocols, m.mat_type())?, false)),
    }
}

/// Moore–Penrose pseudo-inverse through the SVD, as `invert(DECOMP_SVD)` does.
///
/// Returns an `n x m` buffer. `None` when every singular value is below the
/// threshold, which is the case upstream reports as a zero condition number.
fn invert_svd(src: &[f64], m: usize, n: usize) -> Option<Vec<f64>> {
    let (w, u, vt) = crate::linalg::svd_dense(src, m, n);
    let nm = m.min(n);
    // Identity right-hand side: column j of the pseudo-inverse is the
    // back-substitution of e_j.
    let mut out = vec![0.0f64; n * m];
    let threshold = w.iter().take(nm).sum::<f64>() * (f64::EPSILON * 2.0);
    let mut any = false;
    for i in 0..nm {
        if w[i].abs() <= threshold {
            continue;
        }
        any = true;
        let wi = 1.0 / w[i];
        for r in 0..n {
            let vr = vt[i * n + r] * wi;
            for c in 0..m {
                out[r * m + c] += vr * u[c * nm + i];
            }
        }
    }
    any.then_some(out)
}

/// Inverse of a symmetric matrix through its eigen-decomposition.
fn invert_eig(src: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = src.to_vec();
    // `jacobi` returns eigenvalues descending with the matching eigenvector in
    // each *row*, which is the `vt` of `A = V diag(w) Vᵀ` for symmetric `A`.
    let (w, vt) = crate::linalg::jacobi_dense(&mut a, n);
    let threshold = w.iter().map(|v| v.abs()).sum::<f64>() * (f64::EPSILON * 2.0);
    let mut out = vec![0.0f64; n * n];
    let mut any = false;
    for i in 0..n {
        if w[i].abs() <= threshold {
            continue;
        }
        any = true;
        let wi = 1.0 / w[i];
        for r in 0..n {
            let vr = vt[i * n + r] * wi;
            for c in 0..n {
                out[r * n + c] += vr * vt[i * n + c];
            }
        }
    }
    any.then_some(out)
}

/// Invert via LU by solving against each identity column.
fn invert_lu(src: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = src.to_vec();
    let mut perm = vec![0usize; n];
    lu_decompose(&mut a, n, &mut perm)?;

    let mut out = vec![0.0f64; n * n];
    for col in 0..n {
        // Right-hand side is the permuted identity column.
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = if perm[i] == col { 1.0 } else { 0.0 };
            for k in 0..i {
                sum -= a[i * n + k] * y[k];
            }
            y[i] = sum;
        }
        for i in (0..n).rev() {
            let mut sum = y[i];
            for k in (i + 1)..n {
                sum -= a[i * n + k] * out[k * n + col];
            }
            out[i * n + col] = sum / a[i * n + i];
        }
    }
    Some(out)
}

/// Cholesky factorisation `A = L L^T`, returning `L` row-major.
fn cholesky(src: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = src[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                // A non-positive pivot means the matrix is not positive definite.
                if sum <= 0.0 {
                    return None;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Invert a symmetric positive-definite matrix through its Cholesky factor.
fn invert_cholesky(src: &[f64], n: usize) -> Option<Vec<f64>> {
    let l = cholesky(src, n)?;
    let mut out = vec![0.0f64; n * n];
    for col in 0..n {
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = if i == col { 1.0 } else { 0.0 };
            for k in 0..i {
                sum -= l[i * n + k] * y[k];
            }
            y[i] = sum / l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut sum = y[i];
            for k in (i + 1)..n {
                // L^T[i][k] == L[k][i]
                sum -= l[k * n + i] * out[k * n + col];
            }
            out[i * n + col] = sum / l[i * n + i];
        }
    }
    Some(out)
}

/// `cv::solve(src1, src2, dst, flags)` — solve `A x = b`.
///
/// Returns `(x, ok)`; `ok` is false when `A` is singular, matching upstream.
///
/// `DECOMP_LU`, `DECOMP_CHOLESKY` and `DECOMP_EIG` need a square `A`;
/// `DECOMP_SVD` and `DECOMP_QR` also accept an over-determined one and return
/// the least-squares solution. An under-determined system is rejected, as
/// upstream rejects it.
pub fn solve<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    a: &A,
    b: &B,
    method: Decomp,
) -> Result<(Mat, bool)> {
    solve_impl(a, b, method, false)
}

/// `cv::solve(src1, src2, dst, method | DECOMP_NORMAL)`.
///
/// Solves `AᵀA x = Aᵀb`. For an over-determined system that is the same
/// least-squares answer `DECOMP_QR` gives, reached by forming a square system
/// first; it costs accuracy (the condition number is squared) and saves work
/// when `A` has far more rows than columns.
pub fn solve_normal<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    a: &A,
    b: &B,
    method: Decomp,
) -> Result<(Mat, bool)> {
    solve_impl(a, b, method, true)
}

fn solve_impl<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    a: &A,
    b: &B,
    method: Decomp,
    is_normal: bool,
) -> Result<(Mat, bool)> {
    require_float_1ch(a, "solve")?;
    require_float_1ch(b, "solve")?;
    let (m, n) = (a.rows() as usize, a.cols() as usize);
    if a.rows() != b.rows() {
        return cv_error!(
            Code::StsUnmatchedSizes,
            "solve: A has {} rows but b has {}",
            a.rows(),
            b.rows()
        );
    }
    if m < n {
        return cv_error!(Code::StsBadArg, "solve cannot handle under-determined systems");
    }
    // Only SVD and QR are defined on a rectangular system; the others need the
    // normal equations to square it up first.
    if m != n && !is_normal && !matches!(method, Decomp::Svd | Decomp::Qr) {
        return cv_error!(
            Code::StsBadArg,
            "solve: {method:?} requires a square coefficient matrix, got {m}x{n}"
        );
    }

    let av = to_f64(a)?;
    let bv = to_f64(b)?;
    let nb = b.cols() as usize;
    // Upstream ignores DECOMP_NORMAL when the system is already square.
    let is_normal = is_normal && m != n;

    let (sys, rhs, rows) = if is_normal {
        let mut ata = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                ata[i * n + j] = (0..m).map(|t| av[t * n + i] * av[t * n + j]).sum();
            }
        }
        let mut atb = vec![0.0f64; n * nb];
        for i in 0..n {
            for j in 0..nb {
                atb[i * nb + j] = (0..m).map(|t| av[t * n + i] * bv[t * nb + j]).sum();
            }
        }
        (ata, atb, n)
    } else {
        (av, bv, m)
    };

    let x = match method {
        Decomp::Lu => invert_lu(&sys, n).map(|inv| mul_dense(&inv, n, n, &rhs, nb)),
        Decomp::Cholesky => invert_cholesky(&sys, n).map(|inv| mul_dense(&inv, n, n, &rhs, nb)),
        Decomp::Eig => invert_eig(&sys, n).map(|inv| mul_dense(&inv, n, n, &rhs, nb)),
        // Upstream downgrades SVD to EIG once the normal equations have made
        // the system square and symmetric; the answer is the same and cheaper.
        Decomp::Svd if is_normal => {
            invert_eig(&sys, n).map(|inv| mul_dense(&inv, n, n, &rhs, nb))
        }
        Decomp::Svd => solve_svd(&sys, rows, n, &rhs, nb),
        Decomp::Qr => solve_qr(&sys, rows, n, &rhs, nb),
    };

    match x {
        Some(data) => Ok((from_f64(&data, n as i32, nb as i32, a.depth())?, true)),
        None => Ok((Mat::zeros(n as i32, nb as i32, a.mat_type())?, false)),
    }
}

/// Dense row-major `p x q` times `q x r`.
fn mul_dense(lhs: &[f64], p: usize, q: usize, rhs: &[f64], r: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; p * r];
    for i in 0..p {
        for k in 0..q {
            let l = lhs[i * q + k];
            if l == 0.0 {
                continue;
            }
            for j in 0..r {
                out[i * r + j] += l * rhs[k * r + j];
            }
        }
    }
    out
}

/// Least-squares solve through the SVD, dropping singular values below
/// `2 * DBL_EPSILON * sum(w)` — upstream's `SVBkSb` threshold.
fn solve_svd(a: &[f64], m: usize, n: usize, b: &[f64], nb: usize) -> Option<Vec<f64>> {
    let (w, u, vt) = crate::linalg::svd_dense(a, m, n);
    let k = m.min(n);
    let threshold = w.iter().sum::<f64>() * (f64::EPSILON * 2.0);
    let mut x = vec![0.0f64; n * nb];
    let mut any = false;
    for i in 0..k {
        if w[i].abs() <= threshold {
            continue;
        }
        any = true;
        let wi = 1.0 / w[i];
        for j in 0..nb {
            let mut s = 0.0;
            for t in 0..m {
                s += u[t * k + i] * b[t * nb + j];
            }
            s *= wi;
            for r in 0..n {
                x[r * nb + j] += s * vt[i * n + r];
            }
        }
    }
    any.then_some(x)
}

/// Least-squares solve by Householder QR, as `DECOMP_QR` does.
///
/// The reflections are applied to `b` as they are formed rather than
/// accumulating `Q`, so only `R` and the transformed right-hand side are ever
/// materialised.
fn solve_qr(a: &[f64], m: usize, n: usize, b: &[f64], nb: usize) -> Option<Vec<f64>> {
    let mut r = a.to_vec();
    let mut qtb = b.to_vec();

    for col in 0..n {
        // Householder vector for the sub-column below the diagonal.
        let mut norm = 0.0f64;
        for i in col..m {
            norm += r[i * n + col] * r[i * n + col];
        }
        norm = norm.sqrt();
        if norm < 1e-300 {
            return None;
        }
        // Choose the sign that avoids cancellation in the leading entry.
        let alpha = if r[col * n + col] > 0.0 { -norm } else { norm };
        let mut v = vec![0.0f64; m - col];
        for i in col..m {
            v[i - col] = r[i * n + col];
        }
        v[0] -= alpha;
        let vnorm2: f64 = v.iter().map(|x| x * x).sum();
        if vnorm2 < 1e-300 {
            // Already in upper-triangular form for this column.
            r[col * n + col] = alpha;
            for i in col + 1..m {
                r[i * n + col] = 0.0;
            }
            continue;
        }

        for j in col..n {
            let dot: f64 = (col..m).map(|i| v[i - col] * r[i * n + j]).sum();
            let f = 2.0 * dot / vnorm2;
            for i in col..m {
                r[i * n + j] -= f * v[i - col];
            }
        }
        for j in 0..nb {
            let dot: f64 = (col..m).map(|i| v[i - col] * qtb[i * nb + j]).sum();
            let f = 2.0 * dot / vnorm2;
            for i in col..m {
                qtb[i * nb + j] -= f * v[i - col];
            }
        }
    }

    // Back-substitute against the upper-triangular R.
    let mut x = vec![0.0f64; n * nb];
    for j in 0..nb {
        for i in (0..n).rev() {
            let mut s = qtb[i * nb + j];
            for k in i + 1..n {
                s -= r[i * n + k] * x[k * nb + j];
            }
            if r[i * n + i].abs() < 1e-300 {
                return None;
            }
            x[i * nb + j] = s / r[i * n + i];
        }
    }
    Some(x)
}

/// `cv::scaleAdd(src1, alpha, src2, dst)` — `alpha * src1 + src2`.
pub fn scale_add<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    a: &A,
    alpha: f64,
    b: &B,
) -> Result<Mat> {
    crate::arithm::add_weighted(a, alpha, b, 1.0, 0.0)
}

/// `cv::mulTransposed(src, dst, aTa)`.
///
/// With `a_ta` true computes `A^T A`, otherwise `A A^T`.
pub fn mul_transposed<M: MatLike + ?Sized>(m: &M, a_ta: bool) -> Result<Mat> {
    let flags = if a_ta {
        GemmFlags { transpose_a: true, ..Default::default() }
    } else {
        GemmFlags { transpose_b: true, ..Default::default() }
    };
    gemm::<M, M, Mat>(m, m, 1.0, None, 0.0, flags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CV_32FC1, CV_64FC1};

    fn f64m(rows: i32, cols: i32, v: &[f64]) -> Mat {
        Mat::from_vec(rows, cols, 1, v).unwrap()
    }

    /// Reference values from `cv2.solve` on the same matrices; the tolerance is
    /// wide enough only for the different-but-equivalent factorisation order,
    /// not for a different answer.
    #[test]
    fn every_decomposition_solves_the_same_square_system() {
        let a = f64m(3, 3, &[4.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0]);
        let b = f64m(3, 1, &[1.0, 2.0, 3.0]);
        let expect = [0.0, 0.142_857_142_857_142_85, 0.428_571_428_571_428_55];
        for m in [Decomp::Lu, Decomp::Cholesky, Decomp::Svd, Decomp::Eig, Decomp::Qr] {
            let (x, ok) = solve(&a, &b, m).unwrap();
            assert!(ok, "{m:?} reported failure");
            let got = x.to_vec::<f64>().unwrap();
            for (g, e) in got.iter().zip(expect) {
                assert!((g - e).abs() < 1e-12, "{m:?}: got {got:?}");
            }
        }
    }

    #[test]
    fn svd_and_qr_least_squares_fit_an_overdetermined_system() {
        // y = 3.5 + 1.4 x is the least-squares line through these four points.
        let a = f64m(4, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
        let b = f64m(4, 1, &[6.0, 5.0, 7.0, 10.0]);
        for m in [Decomp::Svd, Decomp::Qr] {
            let (x, ok) = solve(&a, &b, m).unwrap();
            assert!(ok, "{m:?} reported failure");
            let got = x.to_vec::<f64>().unwrap();
            assert!((got[0] - 3.5).abs() < 1e-12 && (got[1] - 1.4).abs() < 1e-12, "{m:?}: {got:?}");
        }
        // DECOMP_NORMAL reaches the same answer through A^T A.
        for m in [Decomp::Lu, Decomp::Svd, Decomp::Cholesky] {
            let (x, ok) = solve_normal(&a, &b, m).unwrap();
            assert!(ok, "normal|{m:?} reported failure");
            let got = x.to_vec::<f64>().unwrap();
            assert!((got[0] - 3.5).abs() < 1e-9 && (got[1] - 1.4).abs() < 1e-9, "{m:?}: {got:?}");
        }
    }

    #[test]
    fn lu_rejects_a_rectangular_system_but_svd_takes_it() {
        let a = f64m(4, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
        let b = f64m(4, 1, &[6.0, 5.0, 7.0, 10.0]);
        assert!(solve(&a, &b, Decomp::Lu).is_err());
        assert!(solve(&a, &b, Decomp::Svd).is_ok());
        // Under-determined is refused whatever the method, as upstream refuses it.
        let wide = f64m(2, 4, &[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0]);
        let rhs = f64m(2, 1, &[1.0, 2.0]);
        assert!(solve(&wide, &rhs, Decomp::Svd).is_err());
    }

    #[test]
    fn svd_and_eig_invert_a_symmetric_matrix() {
        let a = f64m(3, 3, &[4.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0]);
        // cv2.invert(a, flags=cv2.DECOMP_SVD)
        let expect = [0.3, 0.0, -0.1, 0.0, 0.285_714_285_714_285_75, -0.142_857_142_857_142_88,
                      -0.1, -0.142_857_142_857_142_82, 0.271_428_571_428_571_4];
        for m in [Decomp::Svd, Decomp::Eig, Decomp::Lu] {
            let (inv, ok) = invert(&a, m).unwrap();
            assert!(ok, "{m:?} reported failure");
            let got = inv.to_vec::<f64>().unwrap();
            for (g, e) in got.iter().zip(expect) {
                assert!((g - e).abs() < 1e-12, "{m:?}: got {got:?}");
            }
        }
    }

    #[test]
    fn svd_inverts_a_rectangular_matrix_to_its_pseudo_inverse() {
        let a = f64m(4, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
        let (inv, ok) = invert(&a, Decomp::Svd).unwrap();
        assert!(ok);
        assert_eq!(inv.size(), crate::Size::new(4, 2));
        // cv2.invert(a, flags=cv2.DECOMP_SVD)
        let expect = [1.0, 0.5, 0.0, -0.5, -0.3, -0.1, 0.1, 0.3];
        let got = inv.to_vec::<f64>().unwrap();
        for (g, e) in got.iter().zip(expect) {
            assert!((g - e).abs() < 1e-12, "got {got:?}");
        }
        // QR is rejected here exactly as upstream rejects it.
        assert!(invert(&a, Decomp::Qr).is_err());
    }

    #[test]
    fn a_singular_system_reports_failure() {
        let a = f64m(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let b = f64m(2, 1, &[1.0, 2.0]);
        assert!(!solve(&a, &b, Decomp::Lu).unwrap().1);
        assert!(!invert(&a, Decomp::Lu).unwrap().1);
    }

    #[test]
    fn matmul_computes_the_product() {
        let a = f64m(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = f64m(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.size(), crate::Size::new(2, 2));
        assert_eq!(c.to_vec::<f64>().unwrap(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn gemm_honours_transpose_flags() {
        let a = f64m(3, 2, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]); // A^T of the 2x3 above
        let b = f64m(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = gemm::<_, _, Mat>(
            &a,
            &b,
            1.0,
            None,
            0.0,
            GemmFlags { transpose_a: true, ..Default::default() },
        )
        .unwrap();
        assert_eq!(c.to_vec::<f64>().unwrap(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn gemm_scales_and_accumulates() {
        let a = f64m(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b = f64m(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let c = f64m(2, 2, &[10.0, 10.0, 10.0, 10.0]);
        let out = gemm(&a, &b, 2.0, Some(&c), 0.5, GemmFlags::default()).unwrap();
        assert_eq!(out.to_vec::<f64>().unwrap(), vec![7.0, 9.0, 11.0, 13.0]);
    }

    #[test]
    fn gemm_rejects_mismatched_inner_dimensions() {
        let a = f64m(2, 3, &[0.0; 6]);
        let b = f64m(2, 2, &[0.0; 4]);
        assert_eq!(matmul(&a, &b).unwrap_err().code, Code::StsUnmatchedSizes);
    }

    #[test]
    fn determinant_matches_known_values() {
        assert!((determinant(&f64m(2, 2, &[1.0, 2.0, 3.0, 4.0])).unwrap() + 2.0).abs() < 1e-12);
        let m = f64m(3, 3, &[6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0]);
        assert!((determinant(&m).unwrap() - (-306.0)).abs() < 1e-9);
        // Identity has determinant 1.
        let mut i = Mat::zeros(4, 4, CV_64FC1).unwrap();
        set_identity(&mut i, 1.0).unwrap();
        assert!((determinant(&i).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn determinant_of_a_singular_matrix_is_zero() {
        let m = f64m(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        assert_eq!(determinant(&m).unwrap(), 0.0);
    }

    #[test]
    fn invert_lu_reproduces_the_identity() {
        let m = f64m(3, 3, &[4.0, 7.0, 2.0, 3.0, 6.0, 1.0, 2.0, 5.0, 3.0]);
        let (inv, ok) = invert(&m, Decomp::Lu).unwrap();
        assert!(ok);
        let prod = matmul(&m, &inv).unwrap().to_vec::<f64>().unwrap();
        for (i, v) in prod.iter().enumerate() {
            let expect = if i % 4 == 0 { 1.0 } else { 0.0 };
            assert!((v - expect).abs() < 1e-10, "entry {i} was {v}");
        }
    }

    #[test]
    fn invert_reports_failure_on_a_singular_matrix() {
        let m = f64m(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let (inv, ok) = invert(&m, Decomp::Lu).unwrap();
        assert!(!ok);
        assert_eq!(inv.to_vec::<f64>().unwrap(), vec![0.0; 4]);
    }

    #[test]
    fn cholesky_inverts_a_positive_definite_matrix() {
        // [[4,2],[2,3]] is symmetric positive definite.
        let m = f64m(2, 2, &[4.0, 2.0, 2.0, 3.0]);
        let (inv, ok) = invert(&m, Decomp::Cholesky).unwrap();
        assert!(ok);
        let prod = matmul(&m, &inv).unwrap().to_vec::<f64>().unwrap();
        for (i, v) in prod.iter().enumerate() {
            let expect = if i % 3 == 0 { 1.0 } else { 0.0 };
            assert!((v - expect).abs() < 1e-12, "entry {i} was {v}");
        }
    }

    #[test]
    fn cholesky_rejects_a_non_positive_definite_matrix() {
        let m = f64m(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let (_, ok) = invert(&m, Decomp::Cholesky).unwrap();
        assert!(!ok);
    }

    #[test]
    fn solve_recovers_a_known_solution() {
        // 2x + y = 5 ; x + 3y = 10  ->  x = 1, y = 3
        let a = f64m(2, 2, &[2.0, 1.0, 1.0, 3.0]);
        let b = f64m(2, 1, &[5.0, 10.0]);
        let (x, ok) = solve(&a, &b, Decomp::Lu).unwrap();
        assert!(ok);
        let v = x.to_vec::<f64>().unwrap();
        assert!((v[0] - 1.0).abs() < 1e-12, "{v:?}");
        assert!((v[1] - 3.0).abs() < 1e-12, "{v:?}");
    }

    #[test]
    fn solve_needs_pivoting_when_the_first_pivot_is_zero() {
        // Without partial pivoting this divides by zero.
        let a = f64m(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let b = f64m(2, 1, &[3.0, 7.0]);
        let (x, ok) = solve(&a, &b, Decomp::Lu).unwrap();
        assert!(ok);
        let v = x.to_vec::<f64>().unwrap();
        assert!((v[0] - 7.0).abs() < 1e-12, "{v:?}");
        assert!((v[1] - 3.0).abs() < 1e-12, "{v:?}");
    }

    #[test]
    fn trace_sums_the_diagonal() {
        let m = f64m(3, 3, &[1.0, 9.0, 9.0, 9.0, 2.0, 9.0, 9.0, 9.0, 3.0]);
        assert_eq!(trace(&m).unwrap(), 6.0);
    }

    #[test]
    fn set_identity_scales_the_diagonal() {
        let mut m = Mat::zeros(3, 3, CV_32FC1).unwrap();
        set_identity(&mut m, 2.0).unwrap();
        assert_eq!(m.to_vec::<f32>().unwrap(), vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn mul_transposed_computes_gram_matrices() {
        let m = f64m(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let ata = mul_transposed(&m, true).unwrap();
        assert_eq!(ata.size(), crate::Size::new(3, 3));
        // (A^T A)[0][0] = 1*1 + 4*4 = 17
        assert_eq!(ata.at::<f64>(0, 0).unwrap(), 17.0);

        let aat = mul_transposed(&m, false).unwrap();
        assert_eq!(aat.size(), crate::Size::new(2, 2));
        // (A A^T)[0][0] = 1+4+9 = 14
        assert_eq!(aat.at::<f64>(0, 0).unwrap(), 14.0);
    }

    #[test]
    fn f32_matrices_are_supported() {
        let a = Mat::from_vec(2, 2, 1, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (inv, ok) = invert(&a, Decomp::Lu).unwrap();
        assert!(ok);
        assert_eq!(inv.depth(), Depth::F32);
        let prod = matmul(&a, &inv).unwrap().to_vec::<f32>().unwrap();
        assert!((prod[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn integer_and_multichannel_inputs_are_rejected() {
        let m = Mat::zeros(2, 2, crate::types::CV_8UC1).unwrap();
        assert_eq!(determinant(&m).unwrap_err().code, Code::StsUnsupportedFormat);
        let c3 = Mat::zeros(2, 2, crate::types::CV_32FC3).unwrap();
        assert_eq!(determinant(&c3).unwrap_err().code, Code::StsUnsupportedFormat);
    }
}
