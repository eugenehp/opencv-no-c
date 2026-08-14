//! Polynomial roots and non-symmetric eigenvalues
//! (`modules/core/src/mathfuncs.cpp`, `lapack.cpp`).

use crate::{cv_assert, Mat, MatLike, Result};

/// `cv::solveCubic(coeffs, roots)`.
///
/// Accepts either four coefficients `[a0, a1, a2, a3]` for
/// `a0 x^3 + a1 x^2 + a2 x + a3`, or three for the monic form. Returns the
/// real roots; the length of the result is upstream's return value.
///
/// This is Cardano's method with the trigonometric branch, transcribed from
/// upstream rather than solved with a general root finder — the branch
/// structure is what decides how many real roots are reported, and a generic
/// solver would disagree about the double-root case.
#[must_use]
pub fn solve_cubic(coeffs: &[f64]) -> Vec<f64> {
    const THIRD: f64 = 1.0 / 3.0;
    let (a1, a2, a3) = match coeffs.len() {
        4 => {
            let a0 = coeffs[0];
            if a0 == 0.0 {
                // Degenerates to a quadratic, which upstream handles by
                // falling through to the same code with one fewer root.
                return solve_quadratic(coeffs[1], coeffs[2], coeffs[3]);
            }
            (coeffs[1] / a0, coeffs[2] / a0, coeffs[3] / a0)
        }
        3 => (coeffs[0], coeffs[1], coeffs[2]),
        _ => return Vec::new(),
    };

    let q = (a1 * a1 - 3.0 * a2) / 9.0;
    let r = (a1 * (2.0 * a1 * a1 - 9.0 * a2) + 27.0 * a3) / 54.0;
    let q3 = q * q * q;
    let d = q3 - r * r;
    let t2 = a1 * THIRD;

    if d > 0.0 {
        // Three distinct real roots: the trigonometric form avoids the
        // complex arithmetic Cardano's formula would otherwise need here.
        let theta = (r / q3.sqrt()).clamp(-1.0, 1.0).acos();
        let t0 = -2.0 * q.sqrt();
        let t1 = theta * THIRD;
        vec![
            t0 * t1.cos() - t2,
            t0 * (t1 + 2.0 * std::f64::consts::PI * THIRD).cos() - t2,
            t0 * (t1 + 4.0 * std::f64::consts::PI * THIRD).cos() - t2,
        ]
    } else if d == 0.0 {
        // A repeated root; upstream reports the two distinct values.
        let s = if r < 0.0 { -1.0 } else { 1.0 };
        let e = s * (r.abs()).cbrt();
        vec![-2.0 * e - t2, e - t2]
    } else {
        // One real root and a complex pair.
        let dd = (-d).sqrt();
        let mut e = (dd + r.abs()).cbrt();
        if r > 0.0 {
            e = -e;
        }
        let x0 = if e == 0.0 { -t2 } else { (e + q / e) - t2 };
        vec![x0]
    }
}

/// The quadratic fallback for a cubic whose leading coefficient vanished.
fn solve_quadratic(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a == 0.0 {
        return if b == 0.0 { Vec::new() } else { vec![-c / b] };
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        Vec::new()
    } else if disc == 0.0 {
        vec![-b / (2.0 * a)]
    } else {
        let s = disc.sqrt();
        vec![(-b - s) / (2.0 * a), (-b + s) / (2.0 * a)]
    }
}

/// A complex number, as `solve_poly` reports its roots.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// Magnitude.
    #[must_use]
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    fn sub(self, o: Self) -> Self {
        Self { re: self.re - o.re, im: self.im - o.im }
    }
    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        if d == 0.0 {
            return Self { re: 0.0, im: 0.0 };
        }
        Self {
            re: (self.re * o.re + self.im * o.im) / d,
            im: (self.im * o.re - self.re * o.im) / d,
        }
    }
}

/// `cv::solvePoly(coeffs, roots, maxIters)`.
///
/// Coefficients run from the constant term upward, matching upstream's order —
/// `coeffs[0] + coeffs[1] x + ... `. Uses the Durand-Kerner iteration, which
/// converges on all roots at once, including complex ones.
pub fn solve_poly(coeffs: &[f64], max_iters: usize) -> Result<Vec<Complex>> {
    // Drop trailing zero coefficients: they raise the reported degree without
    // adding roots, and would make the leading-coefficient division blow up.
    let mut c: Vec<f64> = coeffs.to_vec();
    while c.len() > 1 && c.last().is_some_and(|v| *v == 0.0) {
        c.pop();
    }
    cv_assert!(c.len() >= 2, "solvePoly needs at least a linear polynomial");
    let n = c.len() - 1;
    let lead = c[n];

    // Start on a circle, offset so no two seeds coincide — identical seeds are
    // a fixed point of the iteration and it would never separate them.
    let mut roots: Vec<Complex> = (0..n)
        .map(|i| {
            let a = 0.4 + 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            Complex { re: a.cos(), im: a.sin() }
        })
        .collect();

    let eval = |z: Complex| {
        let mut acc = Complex { re: 0.0, im: 0.0 };
        for &ci in c.iter().rev() {
            acc = acc.mul(z);
            acc.re += ci;
        }
        acc
    };

    for _ in 0..max_iters.max(1) {
        let mut moved: f64 = 0.0;
        for i in 0..n {
            let zi = roots[i];
            let mut den = Complex { re: lead, im: 0.0 };
            for (j, &zj) in roots.iter().enumerate() {
                if i != j {
                    den = den.mul(zi.sub(zj));
                }
            }
            let step = eval(zi).div(den);
            roots[i] = zi.sub(step);
            moved = moved.max(step.abs());
        }
        if moved < 1e-14 {
            break;
        }
    }
    Ok(roots)
}

/// `cv::eigenNonSymmetric(src, eigenvalues, eigenvectors)`.
///
/// Eigenvalues of a general square matrix by Hessenberg reduction followed by
/// the shifted QR iteration.
///
/// # Real eigenvalues only
///
/// Upstream documents this as assuming real eigenvalues, and so does this: a
/// matrix with a genuine complex pair — a rotation, say — has no real answer
/// to report, and this returns an error rather than the real parts, which
/// would look like eigenvalues and satisfy nothing.
pub fn eigen_non_symmetric<S: MatLike + ?Sized>(src: &S) -> Result<Vec<f64>> {
    let n = src.rows() as usize;
    cv_assert!(
        src.rows() == src.cols() && n > 0,
        "eigenNonSymmetric needs a square matrix, got {}x{}",
        src.rows(),
        src.cols()
    );
    cv_assert!(src.channels() == 1, "eigenNonSymmetric needs a single channel");
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = src.at_f64(i as i32, j as i32, 0)?;
        }
    }

    // Reduce to upper Hessenberg by Householder, so each QR sweep is O(n^2)
    // instead of O(n^3).
    for k in 1..n.saturating_sub(1) {
        let mut scale = 0.0;
        for i in k..n {
            scale += a[i * n + (k - 1)].abs();
        }
        if scale == 0.0 {
            continue;
        }
        let mut h = 0.0;
        let mut v = vec![0.0f64; n];
        for i in k..n {
            v[i] = a[i * n + (k - 1)] / scale;
            h += v[i] * v[i];
        }
        let mut g = v[k].hypot((h - v[k] * v[k]).max(0.0).sqrt());
        if v[k] > 0.0 {
            g = -g;
        }
        h -= v[k] * g;
        v[k] -= g;
        if h.abs() < 1e-300 {
            continue;
        }
        for j in 0..n {
            let mut f = 0.0;
            for i in k..n {
                f += v[i] * a[i * n + j];
            }
            f /= h;
            for i in k..n {
                a[i * n + j] -= f * v[i];
            }
        }
        for row in a.chunks_mut(n) {
            let mut f = 0.0;
            for i in k..n {
                f += v[i] * row[i];
            }
            f /= h;
            for i in k..n {
                row[i] -= f * v[i];
            }
        }
        a[k * n + (k - 1)] = scale * g;
        for i in (k + 1)..n {
            a[i * n + (k - 1)] = 0.0;
        }
    }

    // Shifted QR on the Hessenberg form, deflating from the bottom.
    let mut eigs = vec![0.0f64; n];
    let mut hi = n as i64 - 1;
    let mut iters = 0usize;
    let max_iters = 100 * n + 1000;
    while hi >= 0 {
        // Find a negligible subdiagonal entry to split at.
        let mut lo = hi;
        while lo > 0 {
            let s = a[((lo - 1) as usize) * n + (lo - 1) as usize].abs()
                + a[(lo as usize) * n + lo as usize].abs();
            let s = if s == 0.0 { 1.0 } else { s };
            if a[(lo as usize) * n + (lo - 1) as usize].abs() <= f64::EPSILON * s {
                a[(lo as usize) * n + (lo - 1) as usize] = 0.0;
                break;
            }
            lo -= 1;
        }
        if lo == hi {
            // A 1x1 block: the eigenvalue is on the diagonal.
            eigs[hi as usize] = a[(hi as usize) * n + hi as usize];
            hi -= 1;
            iters = 0;
            continue;
        }
        if lo == hi - 1 {
            // A 2x2 block: solve it directly.
            let (i, j) = ((hi - 1) as usize, hi as usize);
            let (p, q, r, s) = (a[i * n + i], a[i * n + j], a[j * n + i], a[j * n + j]);
            let tr = p + s;
            let det = p * s - q * r;
            let disc = tr * tr / 4.0 - det;
            if disc < -1e-12 * (tr * tr + det.abs()).max(1.0) {
                return Err(crate::Error::new(
                    crate::Code::StsNotImplemented,
                    "eigenNonSymmetric found a complex eigenvalue pair; upstream assumes real eigenvalues"
                        .to_string(),
                ));
            }
            let sq = disc.max(0.0).sqrt();
            eigs[i] = tr / 2.0 + sq;
            eigs[j] = tr / 2.0 - sq;
            hi -= 2;
            iters = 0;
            continue;
        }

        iters += 1;
        if iters > max_iters {
            return Err(crate::Error::new(
                crate::Code::StsNoConv,
                format!("eigenNonSymmetric did not converge in {max_iters} iterations"),
            ));
        }
        // Wilkinson shift from the trailing 2x2.
        let (i, j) = ((hi - 1) as usize, hi as usize);
        let (p, q, r, s) = (a[i * n + i], a[i * n + j], a[j * n + i], a[j * n + j]);
        let tr = p + s;
        let det = p * s - q * r;
        let disc = tr * tr / 4.0 - det;
        let mu = if disc >= 0.0 {
            let sq = disc.sqrt();
            let (e1, e2) = (tr / 2.0 + sq, tr / 2.0 - sq);
            if (e1 - s).abs() < (e2 - s).abs() { e1 } else { e2 }
        } else {
            // Exceptional shift, to break the cycle a complex pair would cause.
            tr / 2.0
        };

        let (l, h) = (lo as usize, hi as usize);
        for d in a.iter_mut().step_by(n + 1).take(h + 1).skip(l) {
            *d -= mu;
        }
        // Givens QR of the shifted block, applied as RQ to complete the sweep.
        let mut rot = Vec::with_capacity(h - l);
        for k in l..h {
            let (x, y) = (a[k * n + k], a[(k + 1) * n + k]);
            let d = x.hypot(y);
            let (c, sn) = if d == 0.0 { (1.0, 0.0) } else { (x / d, y / d) };
            rot.push((c, sn));
            for col in k..n {
                let (u, v) = (a[k * n + col], a[(k + 1) * n + col]);
                a[k * n + col] = c * u + sn * v;
                a[(k + 1) * n + col] = -sn * u + c * v;
            }
        }
        for (k, &(c, sn)) in (l..h).zip(&rot) {
            for row in 0..=(k + 1).min(n - 1) {
                let (u, v) = (a[row * n + k], a[row * n + k + 1]);
                a[row * n + k] = c * u + sn * v;
                a[row * n + k + 1] = -sn * u + c * v;
            }
        }
        for d in a.iter_mut().step_by(n + 1).take(h + 1).skip(l) {
            *d += mu;
        }
    }

    // Upstream returns eigenvalues in descending order.
    eigs.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    Ok(eigs)
}

/// `cv::flipND(src, dst, axis)` — reverse one axis of an N-dimensional array.
///
/// `shape` is in row-major order, `data` matches it. A negative `axis` counts
/// from the end, as it does upstream.
pub fn flip_nd<T: Copy>(data: &[T], shape: &[usize], axis: i32) -> Result<Vec<T>> {
    let nd = shape.len();
    cv_assert!(nd > 0, "flipND needs at least one dimension");
    let total: usize = shape.iter().product();
    cv_assert!(data.len() == total, "data length {} does not match shape {shape:?}", data.len());
    let ax = if axis < 0 { axis + nd as i32 } else { axis };
    cv_assert!(
        ax >= 0 && (ax as usize) < nd,
        "axis {axis} is out of range for a {nd}-dimensional array"
    );
    let ax = ax as usize;
    // Row-major strides.
    let mut stride = vec![1usize; nd];
    for i in (0..nd - 1).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; nd];
    for _ in 0..total {
        let mut off = 0;
        for d in 0..nd {
            let k = if d == ax { shape[d] - 1 - idx[d] } else { idx[d] };
            off += k * stride[d];
        }
        out.push(data[off]);
        for d in (0..nd).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(out)
}

/// `cv::transposeND(src, order, dst)` — permute the axes of an N-dimensional array.
pub fn transpose_nd<T: Copy>(
    data: &[T],
    shape: &[usize],
    order: &[usize],
) -> Result<(Vec<T>, Vec<usize>)> {
    let nd = shape.len();
    cv_assert!(order.len() == nd, "the permutation must name all {nd} axes");
    let mut seen = vec![false; nd];
    for &o in order {
        cv_assert!(o < nd, "axis {o} is out of range for a {nd}-dimensional array");
        cv_assert!(!seen[o], "axis {o} appears twice in the permutation");
        seen[o] = true;
    }
    let total: usize = shape.iter().product();
    cv_assert!(data.len() == total, "data length {} does not match shape {shape:?}", data.len());

    let mut stride = vec![1usize; nd];
    for i in (0..nd.max(1) - 1).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    let new_shape: Vec<usize> = order.iter().map(|&o| shape[o]).collect();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; nd];
    for _ in 0..total {
        let off: usize = (0..nd).map(|d| idx[d] * stride[order[d]]).sum();
        out.push(data[off]);
        for d in (0..nd).rev() {
            idx[d] += 1;
            if idx[d] < new_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok((out, new_shape))
}

/// `cv::sumElems(src)` — the per-channel sum, up to four channels.
pub fn sum_elems<S: MatLike + ?Sized>(src: &S) -> Result<[f64; 4]> {
    let mut acc = [0.0f64; 4];
    let cn = src.channels().min(4);
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            for c in 0..cn {
                acc[c as usize] += src.at_f64(y, x, c)?;
            }
        }
    }
    Ok(acc)
}

/// `cv::rectangleIntersectionArea(a, b)` — the overlap of two rectangles,
/// given as `(x, y, width, height)`.
#[must_use]
pub fn rectangle_intersection_area(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> f64 {
    let w = (a.0 + a.2).min(b.0 + b.2) - a.0.max(b.0);
    let h = (a.1 + a.3).min(b.1 + b.3) - a.1.max(b.1);
    if w <= 0.0 || h <= 0.0 {
        0.0
    } else {
        w * h
    }
}

/// Build a [`Mat`] from a row-major `f64` buffer — the shape these solvers take.
pub fn mat_from_rows(rows: i32, cols: i32, data: &[f64]) -> Result<Mat> {
    Mat::from_vec(rows, cols, 1, data)
}

/// `cv::SolveLPResult` — what [`solve_lp`] concluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SolveLpResult {
    /// `SOLVELP_LOST` — feasible, but rounding lost the solution.
    Lost = -3,
    /// `SOLVELP_UNBOUNDED` — the objective can grow without limit.
    Unbounded = -2,
    /// `SOLVELP_UNFEASIBLE` — no point satisfies every constraint.
    Unfeasible = -1,
    /// `SOLVELP_SINGLE` — one maximum, returned in `z`.
    Single = 0,
    /// `SOLVELP_MULTI` — several maxima; an arbitrary one is returned.
    Multi = 1,
}

/// `cv::solveLP(Func, Constr, z, constr_eps)` — maximise `c·x` s.t. `Ax ≤ b`, `x ≥ 0`.
///
/// `func` is `c` as a row (a column is accepted too and read as `cᵀ`, which is
/// upstream's convenience). `constr` is the `m x (n+1)` matrix whose last
/// column is `b` and whose remaining columns are `A`. On success the maximiser
/// is returned as an `n x 1` `CV_64F` column.
///
/// This is the two-phase simplex method from *Introduction to Algorithms*,
/// which is the source upstream transcribes. **Bland's rule** picks both the
/// entering and the leaving variable — always the lowest index among the
/// candidates. That is slower than Dantzig's steepest-ascent choice and it is
/// the point: Bland's rule cannot cycle, and a degenerate LP will cycle
/// forever under the faster rule.
pub fn solve_lp<F: MatLike + ?Sized, C: MatLike + ?Sized>(
    func: &F,
    constr: &C,
    constr_eps: f64,
) -> Result<(SolveLpResult, Mat)> {
    // `c` as a flat row, however it was handed in.
    cv_assert!(
        func.channels() == 1 && constr.channels() == 1,
        "solveLP takes single-channel matrices"
    );
    cv_assert!(func.rows() == 1 || func.cols() == 1, "solveLP: Func must be a vector");
    let n = (func.rows() * func.cols()) as usize;
    let mut c = Vec::with_capacity(n);
    for i in 0..n as i32 {
        let (y, x) = if func.rows() == 1 { (0, i) } else { (i, 0) };
        c.push(func.at_f64(y, x, 0)?);
    }

    let m = constr.rows() as usize;
    cv_assert!(m > 0, "solveLP needs at least one constraint");
    cv_assert!(
        constr.cols() as usize == n + 1,
        "solveLP: Constr must be m x (n+1); got {} columns for {n} variables",
        constr.cols()
    );
    let mut a = vec![vec![0.0f64; n]; m];
    let mut b = vec![0.0f64; m];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = constr.at_f64(i as i32, j as i32, 0)?;
        }
        b[i] = constr.at_f64(i as i32, n as i32, 0)?;
    }

    let empty = || Mat::zeros(n as i32, 1, crate::types::CV_64FC1);
    let mut d = match Dictionary::initialise(&c, &a, &b, constr_eps) {
        Some(d) => d,
        None => return Ok((SolveLpResult::Unfeasible, empty()?)),
    };
    match d.optimise() {
        Some(()) => {}
        None => return Ok((SolveLpResult::Unbounded, empty()?)),
    }

    let x = d.solution(n);
    // Verify the answer really satisfies the constraints: a simplex that ran to
    // completion can still land outside them once rounding is accounted for,
    // and that is exactly what SOLVELP_LOST reports.
    for i in 0..m {
        let lhs: f64 = (0..n).map(|j| a[i][j] * x[j]).sum();
        if lhs > b[i] + constr_eps {
            return Ok((SolveLpResult::Lost, empty()?));
        }
    }
    if x.iter().any(|v| *v < -constr_eps) {
        return Ok((SolveLpResult::Lost, empty()?));
    }

    // A zero reduced cost on a non-basic variable means that variable can enter
    // without changing the objective, so the maximum is attained along an edge.
    let multi = d.has_degenerate_optimum();
    let result = if multi { SolveLpResult::Multi } else { SolveLpResult::Single };
    Ok((result, mat_from_rows(n as i32, 1, &x)?))
}

/// The simplex tableau in the "slack form" of Cormen et al.
///
/// `basis[i]` names the variable basic in row `i`; `nonbasic[j]` names the
/// variable in column `j`. Variables `0..n` are the originals, `n..n+m` the
/// slacks, and `n+m` the auxiliary variable of phase one.
struct Dictionary {
    /// `b[i]` — value of the basic variable in row `i`.
    b: Vec<f64>,
    /// `a[i][j]` — coefficient of non-basic `j` in row `i`.
    a: Vec<Vec<f64>>,
    /// Objective coefficients over the non-basic variables.
    c: Vec<f64>,
    /// Objective value at the current vertex.
    v: f64,
    basis: Vec<usize>,
    nonbasic: Vec<usize>,
    eps: f64,
}

impl Dictionary {
    /// Build a feasible starting dictionary, running phase one if needed.
    ///
    /// Returns `None` when the constraints have no feasible point at all.
    fn initialise(c: &[f64], a: &[Vec<f64>], b: &[f64], eps: f64) -> Option<Dictionary> {
        let (n, m) = (c.len(), b.len());
        let mut d = Dictionary {
            b: b.to_vec(),
            a: a.to_vec(),
            c: c.to_vec(),
            v: 0.0,
            basis: (n..n + m).collect(),
            nonbasic: (0..n).collect(),
            eps,
        };
        // With every b_i non-negative the origin is already feasible.
        let worst = (0..m).min_by(|&i, &j| d.b[i].partial_cmp(&d.b[j]).unwrap());
        match worst {
            Some(k) if d.b[k] < -eps => {
                d.phase_one(k)?;
                Some(d)
            }
            _ => Some(d),
        }
    }

    /// Phase one: introduce `x_{n+m}`, drive it to zero, then drop it.
    fn phase_one(&mut self, worst: usize) -> Option<()> {
        let aux = self.nonbasic.len() + self.basis.len();
        // Auxiliary problem: maximise -x_aux subject to Ax - x_aux <= b.
        let saved_c = std::mem::replace(&mut self.c, vec![0.0; self.nonbasic.len()]);
        let saved_v = std::mem::replace(&mut self.v, 0.0);
        for row in &mut self.a {
            row.push(-1.0);
        }
        self.nonbasic.push(aux);
        self.c.push(-1.0);

        // One forced pivot makes the dictionary feasible: bring x_aux in on the
        // row that violates its bound the most.
        let enter = self.nonbasic.len() - 1;
        self.pivot(worst, enter);
        self.optimise()?;

        // Feasible only if the auxiliary variable is back at zero.
        if self.v < -self.eps {
            return None;
        }
        // If x_aux is still basic it sits at zero; pivot it out on any row.
        if let Some(row) = self.basis.iter().position(|&x| x == aux) {
            let col = (0..self.nonbasic.len()).find(|&j| self.a[row][j].abs() > self.eps)?;
            self.pivot(row, col);
        }
        let col = self.nonbasic.iter().position(|&x| x == aux)?;
        self.nonbasic.remove(col);
        for row in &mut self.a {
            row.remove(col);
        }

        // Re-express the real objective over the current non-basic set.
        self.c = vec![0.0; self.nonbasic.len()];
        self.v = saved_v;
        for (var, coeff) in saved_c.iter().enumerate() {
            if coeff.abs() <= self.eps {
                continue;
            }
            if let Some(j) = self.nonbasic.iter().position(|&x| x == var) {
                self.c[j] += coeff;
            } else if let Some(i) = self.basis.iter().position(|&x| x == var) {
                // Substitute the basic row for this variable.
                self.v += coeff * self.b[i];
                for j in 0..self.nonbasic.len() {
                    self.c[j] -= coeff * self.a[i][j];
                }
            }
        }
        Some(())
    }

    /// Pivot until no entering variable improves the objective.
    ///
    /// `None` means the problem is unbounded.
    fn optimise(&mut self) -> Option<()> {
        // The iteration cap is a backstop: Bland's rule guarantees termination,
        // so reaching it would mean a bug rather than a hard problem.
        let limit = 1000 * (self.nonbasic.len() + self.basis.len()).max(1);
        for _ in 0..limit {
            // Bland: lowest-indexed variable with a positive reduced cost.
            let enter = (0..self.nonbasic.len())
                .filter(|&j| self.c[j] > self.eps)
                .min_by_key(|&j| self.nonbasic[j]);
            let Some(enter) = enter else { return Some(()) };

            // Tightest ratio; ties broken again by lowest variable index.
            let mut leave: Option<(f64, usize)> = None;
            for i in 0..self.basis.len() {
                if self.a[i][enter] <= self.eps {
                    continue;
                }
                let ratio = self.b[i] / self.a[i][enter];
                let better = match leave {
                    None => true,
                    Some((best, bi)) => {
                        ratio < best - self.eps
                            || (ratio <= best + self.eps && self.basis[i] < self.basis[bi])
                    }
                };
                if better {
                    leave = Some((ratio, i));
                }
            }
            // No positive coefficient in the entering column: unbounded.
            let (_, leave) = leave?;
            self.pivot(leave, enter);
        }
        Some(())
    }

    /// Exchange the basic variable of row `r` with the non-basic in column `c`.
    fn pivot(&mut self, r: usize, col: usize) {
        let p = self.a[r][col];
        // Rewrite row r in terms of the entering variable.
        self.b[r] /= p;
        for j in 0..self.a[r].len() {
            self.a[r][j] /= p;
        }
        self.a[r][col] = 1.0 / p;

        for i in 0..self.basis.len() {
            if i == r {
                continue;
            }
            let f = self.a[i][col];
            if f == 0.0 {
                continue;
            }
            self.b[i] -= f * self.b[r];
            for j in 0..self.a[i].len() {
                if j == col {
                    self.a[i][j] = -f * self.a[r][j];
                } else {
                    self.a[i][j] -= f * self.a[r][j];
                }
            }
        }

        let f = self.c[col];
        if f != 0.0 {
            self.v += f * self.b[r];
            for j in 0..self.c.len() {
                if j == col {
                    self.c[j] = -f * self.a[r][j];
                } else {
                    self.c[j] -= f * self.a[r][j];
                }
            }
        }
        std::mem::swap(&mut self.basis[r], &mut self.nonbasic[col]);
    }

    /// Values of the first `n` variables at the current vertex.
    fn solution(&self, n: usize) -> Vec<f64> {
        let mut x = vec![0.0f64; n];
        for (i, &var) in self.basis.iter().enumerate() {
            if var < n {
                x[var] = self.b[i];
            }
        }
        x
    }

    /// Does some non-basic variable have a zero reduced cost?
    fn has_degenerate_optimum(&self) -> bool {
        self.c.iter().any(|v| v.abs() <= self.eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut v: Vec<f64>) -> Vec<f64> {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    #[test]
    fn solve_cubic_finds_three_separated_roots() {
        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        let r = sorted(solve_cubic(&[1.0, -6.0, 11.0, -6.0]));
        assert_eq!(r.len(), 3);
        for (got, want) in r.iter().zip([1.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
        // The monic three-coefficient form is the same polynomial.
        let m = sorted(solve_cubic(&[-6.0, 11.0, -6.0]));
        assert_eq!(m.len(), 3);
        for (a, b) in m.iter().zip(&r) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn solve_cubic_handles_one_real_root() {
        // x^3 + x + 1 has a single real root near -0.6823.
        let r = solve_cubic(&[1.0, 0.0, 1.0, 1.0]);
        assert_eq!(r.len(), 1, "one real root and a complex pair");
        let x = r[0];
        assert!((x * x * x + x + 1.0).abs() < 1e-9, "root {x} does not satisfy the polynomial");
    }

    #[test]
    fn solve_cubic_handles_a_repeated_root() {
        // (x-2)^2 (x+4) = x^3 - 12x + 16
        let r = sorted(solve_cubic(&[1.0, 0.0, -12.0, 16.0]));
        assert!(!r.is_empty());
        for x in &r {
            let v = x * x * x - 12.0 * x + 16.0;
            assert!(v.abs() < 1e-7, "root {x} gives {v}");
        }
    }

    #[test]
    fn solve_cubic_degenerates_gracefully() {
        // A zero leading coefficient makes it a quadratic.
        let r = sorted(solve_cubic(&[0.0, 1.0, -3.0, 2.0]));
        assert_eq!(r, vec![1.0, 2.0]);
        // And a zero quadratic term too.
        assert_eq!(solve_cubic(&[0.0, 0.0, 2.0, -4.0]), vec![2.0]);
        // Too few coefficients is empty, not a panic.
        assert!(solve_cubic(&[1.0, 2.0]).is_empty());
    }

    #[test]
    fn solve_poly_finds_all_roots_including_complex() {
        // (x-1)(x+2)(x^2+1) = x^4 + x^3 - x^2 + x - 2
        // Upstream's coefficient order is constant-first.
        let roots = solve_poly(&[-2.0, 1.0, -1.0, 1.0, 1.0], 300).unwrap();
        assert_eq!(roots.len(), 4);
        let want = [(1.0, 0.0), (-2.0, 0.0), (0.0, 1.0), (0.0, -1.0)];
        for (wr, wi) in want {
            let hit = roots
                .iter()
                .any(|z| (z.re - wr).abs() < 1e-6 && (z.im - wi).abs() < 1e-6);
            assert!(hit, "missing root {wr}{wi:+}i in {roots:?}");
        }
    }

    #[test]
    fn solve_poly_matches_solve_cubic_on_real_roots() {
        // Same polynomial, both routes: x^3 - 6x^2 + 11x - 6.
        let cubic = sorted(solve_cubic(&[1.0, -6.0, 11.0, -6.0]));
        let poly = solve_poly(&[-6.0, 11.0, -6.0, 1.0], 300).unwrap();
        let mut re: Vec<f64> = poly.iter().map(|z| z.re).collect();
        re.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (a, b) in re.iter().zip(&cubic) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        for z in &poly {
            assert!(z.im.abs() < 1e-6, "these roots are all real");
        }
    }

    #[test]
    fn solve_poly_validates_and_trims() {
        assert!(solve_poly(&[1.0], 10).is_err());
        assert!(solve_poly(&[], 10).is_err());
        // Trailing zeros do not invent roots.
        let r = solve_poly(&[-1.0, 1.0, 0.0, 0.0], 100).unwrap();
        assert_eq!(r.len(), 1);
        assert!((r[0].re - 1.0).abs() < 1e-9);
    }

    #[test]
    fn eigen_non_symmetric_matches_a_known_spectrum() {
        // Upper triangular: the eigenvalues are on the diagonal.
        let m = mat_from_rows(3, 3, &[4.0, 1.0, 2.0, 0.0, 3.0, -1.0, 0.0, 0.0, 7.0]).unwrap();
        let e = eigen_non_symmetric(&m).unwrap();
        assert_eq!(e.len(), 3);
        for (got, want) in e.iter().zip([7.0, 4.0, 3.0]) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want} (descending order)");
        }
    }

    #[test]
    fn eigen_non_symmetric_handles_a_genuinely_non_symmetric_matrix() {
        // Eigenvalues 1, 2, 3 with a non-symmetric similarity applied.
        let m = mat_from_rows(3, 3, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 6.0, -11.0, 6.0]).unwrap();
        let e = eigen_non_symmetric(&m).unwrap();
        let s = sorted(e);
        for (got, want) in s.iter().zip([1.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn eigen_non_symmetric_refuses_a_complex_pair() {
        // A 90-degree rotation has eigenvalues +/- i and no real answer.
        let m = mat_from_rows(2, 2, &[0.0, -1.0, 1.0, 0.0]).unwrap();
        let e = eigen_non_symmetric(&m);
        assert!(e.is_err(), "a rotation has no real eigenvalues to report");
        let msg = format!("{}", e.unwrap_err());
        assert!(msg.contains("complex"), "the error should say why: {msg}");
    }

    #[test]
    fn eigen_non_symmetric_validates_shape() {
        let m = mat_from_rows(2, 3, &[1.0; 6]).unwrap();
        assert!(eigen_non_symmetric(&m).is_err());
    }

    #[test]
    fn flip_nd_reverses_the_named_axis_only() {
        // 2x3, row-major.
        let d = [1, 2, 3, 4, 5, 6];
        assert_eq!(flip_nd(&d, &[2, 3], 0).unwrap(), vec![4, 5, 6, 1, 2, 3]);
        assert_eq!(flip_nd(&d, &[2, 3], 1).unwrap(), vec![3, 2, 1, 6, 5, 4]);
        // Negative axes count from the end, as upstream allows.
        assert_eq!(flip_nd(&d, &[2, 3], -1).unwrap(), flip_nd(&d, &[2, 3], 1).unwrap());
        assert_eq!(flip_nd(&d, &[2, 3], -2).unwrap(), flip_nd(&d, &[2, 3], 0).unwrap());
        // Flipping twice is the identity.
        let once = flip_nd(&d, &[2, 3], 1).unwrap();
        assert_eq!(flip_nd(&once, &[2, 3], 1).unwrap(), d.to_vec());
        // Out-of-range axes are errors, not silent no-ops.
        assert!(flip_nd(&d, &[2, 3], 2).is_err());
        assert!(flip_nd(&d, &[2, 3], -3).is_err());
        assert!(flip_nd(&d, &[2, 4], 0).is_err());
    }

    #[test]
    fn flip_nd_works_in_three_dimensions() {
        let d: Vec<i32> = (0..24).collect();
        let f = flip_nd(&d, &[2, 3, 4], 2).unwrap();
        // The innermost axis reverses within each run of 4.
        assert_eq!(&f[..4], &[3, 2, 1, 0]);
        assert_eq!(&f[4..8], &[7, 6, 5, 4]);
    }

    #[test]
    fn transpose_nd_permutes_axes() {
        let d = [1, 2, 3, 4, 5, 6];
        let (t, shape) = transpose_nd(&d, &[2, 3], &[1, 0]).unwrap();
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(t, vec![1, 4, 2, 5, 3, 6]);
        // The identity permutation changes nothing.
        let (same, s2) = transpose_nd(&d, &[2, 3], &[0, 1]).unwrap();
        assert_eq!((same, s2), (d.to_vec(), vec![2, 3]));
    }

    #[test]
    fn transpose_nd_round_trips_in_three_dimensions() {
        let d: Vec<i32> = (0..24).collect();
        let (t, s) = transpose_nd(&d, &[2, 3, 4], &[2, 0, 1]).unwrap();
        assert_eq!(s, vec![4, 2, 3]);
        // The inverse permutation restores the original.
        let (back, s2) = transpose_nd(&t, &s, &[1, 2, 0]).unwrap();
        assert_eq!(s2, vec![2, 3, 4]);
        assert_eq!(back, d);
    }

    #[test]
    fn transpose_nd_rejects_a_bad_permutation() {
        let d = [1, 2, 3, 4, 5, 6];
        assert!(transpose_nd(&d, &[2, 3], &[0, 0]).is_err(), "a repeated axis is not a permutation");
        assert!(transpose_nd(&d, &[2, 3], &[0]).is_err());
        assert!(transpose_nd(&d, &[2, 3], &[0, 2]).is_err());
    }

    #[test]
    fn rectangle_intersection_area_matches_geometry() {
        assert_eq!(rectangle_intersection_area((0.0, 0.0, 4.0, 4.0), (2.0, 2.0, 4.0, 4.0)), 4.0);
        // Disjoint, and merely touching, are both zero.
        assert_eq!(rectangle_intersection_area((0.0, 0.0, 1.0, 1.0), (5.0, 5.0, 1.0, 1.0)), 0.0);
        assert_eq!(rectangle_intersection_area((0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0)), 0.0);
        // Containment gives the smaller area.
        assert_eq!(rectangle_intersection_area((0.0, 0.0, 10.0, 10.0), (2.0, 3.0, 1.0, 2.0)), 2.0);
    }

    /// Values recorded from cv2 4.13.0, in cv2's own emission order.
    #[test]
    fn solve_cubic_matches_cv2_root_for_root_and_in_order() {
        let cases: [(&[f64], &[f64]); 5] = [
            (&[1.0, -6.0, 11.0, -6.0], &[1.0, 2.999_999_999_999_999_6, 2.0]),
            (&[1.0, 0.0, 1.0, 1.0], &[-0.682_327_803_828_019_3]),
            (&[1.0, 0.0, -12.0, 16.0], &[-4.0, 2.0]),
            (&[2.0, -3.0, -3.0, 2.0], &[-1.0, 1.999_999_999_999_999_6, 0.500_000_000_000_000_3]),
            (&[1.0, -1.0, -1.0, 1.0], &[-1.0, 1.0]),
        ];
        for (coeffs, want) in cases {
            let got = solve_cubic(coeffs);
            assert_eq!(got.len(), want.len(), "root count for {coeffs:?}: cv2 reports {}", want.len());
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() < 1e-12, "{coeffs:?}: got {got:?}, cv2 gives {want:?}");
            }
        }
    }

    /// cv2's eigenNonSymmetric on the same matrices.
    #[test]
    fn eigen_non_symmetric_matches_cv2() {
        let tri = mat_from_rows(3, 3, &[4.0, 1.0, 2.0, 0.0, 3.0, -1.0, 0.0, 0.0, 7.0]).unwrap();
        assert_eq!(eigen_non_symmetric(&tri).unwrap(), vec![7.0, 4.0, 3.0]);
        // cv2: [2.999999999999983, 2.0000000000000266, 0.9999999999999911]
        let comp = mat_from_rows(3, 3, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 6.0, -11.0, 6.0]).unwrap();
        let e = eigen_non_symmetric(&comp).unwrap();
        for (g, w) in e.iter().zip([3.0, 2.0, 1.0]) {
            assert!((g - w).abs() < 1e-12, "got {e:?}, cv2 agrees to ~1e-14");
        }
    }

    /// The values are `cv2.solveLP` on the same `Func`/`Constr` pairs.
    #[test]
    fn solve_lp_matches_cv2() {
        // Cormen's worked example: maximise 3x+y+2z.
        let c = mat_from_rows(1, 3, &[3.0, 1.0, 2.0]).unwrap();
        let a = mat_from_rows(
            3,
            4,
            &[1.0, 1.0, 3.0, 30.0, 2.0, 2.0, 5.0, 24.0, 4.0, 1.0, 2.0, 36.0],
        )
        .unwrap();
        let (r, z) = solve_lp(&c, &a, 1e-12).unwrap();
        assert_eq!(r, SolveLpResult::Single);
        let got = z.to_vec::<f64>().unwrap();
        for (g, w) in got.iter().zip([8.0, 4.0, 0.0]) {
            assert!((g - w).abs() < 1e-9, "got {got:?}");
        }

        // Box constraints: each variable pinned by its own row.
        let c = mat_from_rows(1, 2, &[1.0, 1.0]).unwrap();
        let a = mat_from_rows(2, 3, &[1.0, 0.0, 2.0, 0.0, 1.0, 3.0]).unwrap();
        let (r, z) = solve_lp(&c, &a, 1e-12).unwrap();
        assert_eq!(r, SolveLpResult::Single);
        assert_eq!(z.to_vec::<f64>().unwrap(), vec![2.0, 3.0]);

        // A column Func is read as its transpose, as upstream allows.
        let ct = mat_from_rows(2, 1, &[1.0, 1.0]).unwrap();
        let (r2, z2) = solve_lp(&ct, &a, 1e-12).unwrap();
        assert_eq!((r2, z2.to_vec::<f64>().unwrap()), (r, vec![2.0, 3.0]));
    }

    #[test]
    fn solve_lp_reports_unbounded_and_unfeasible() {
        let c = mat_from_rows(1, 2, &[1.0, 1.0]).unwrap();
        // -x - y <= -5 leaves the objective free to grow.
        let unbounded = mat_from_rows(1, 3, &[-1.0, -1.0, -5.0]).unwrap();
        assert_eq!(solve_lp(&c, &unbounded, 1e-12).unwrap().0, SolveLpResult::Unbounded);
        // x + y <= -5 with x, y >= 0 has no feasible point.
        let unfeasible = mat_from_rows(2, 3, &[1.0, 1.0, -5.0, 1.0, 0.0, 3.0]).unwrap();
        assert_eq!(solve_lp(&c, &unfeasible, 1e-12).unwrap().0, SolveLpResult::Unfeasible);
    }

    #[test]
    fn solve_lp_runs_phase_one_and_detects_multiple_maxima() {
        let c = mat_from_rows(1, 2, &[1.0, 1.0]).unwrap();
        // The first row makes the origin infeasible, forcing phase one.
        let a = mat_from_rows(2, 3, &[-1.0, -1.0, -2.0, 1.0, 1.0, 10.0]).unwrap();
        let (r, z) = solve_lp(&c, &a, 1e-12).unwrap();
        assert_eq!(r, SolveLpResult::Multi);
        let got = z.to_vec::<f64>().unwrap();
        assert!((got[0] + got[1] - 10.0).abs() < 1e-9, "the optimum lies on x+y=10; got {got:?}");

        // x + y <= 4 alone: every point of the edge is optimal.
        let edge = mat_from_rows(1, 3, &[1.0, 1.0, 4.0]).unwrap();
        let (r, z) = solve_lp(&c, &edge, 1e-12).unwrap();
        assert_eq!(r, SolveLpResult::Multi);
        assert_eq!(z.to_vec::<f64>().unwrap(), vec![4.0, 0.0]);
    }

    #[test]
    fn sum_elems_adds_each_channel() {
        let m = Mat::from_vec(1, 3, 3, &[1u8, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let s = sum_elems(&m).unwrap();
        assert_eq!([s[0], s[1], s[2]], [12.0, 15.0, 18.0]);
        assert_eq!(s[3], 0.0, "the unused channel stays zero");
    }
}
