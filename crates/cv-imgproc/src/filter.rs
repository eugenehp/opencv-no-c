//! Linear and rank filters (`modules/imgproc/src/filter.dispatch.cpp`,
//! `smooth.dispatch.cpp`, `median_blur.simd.hpp`).
//!
//! As upstream documents, `filter2D` computes a **correlation**, not a
//! convolution: the kernel is not flipped. `sepFilter2D`, `blur`,
//! `GaussianBlur` and `Sobel` are all expressed in terms of it.

use cv_core::{cv_assert, cv_error, BorderType, Code, Depth, Mat, MatLike, Point, Result, Size};


/// `cv::getGaussianKernel(ksize, sigma)`.
///
/// Reproduces `getGaussianKernelBitExact`, including the hard-coded kernels
/// upstream returns for `ksize <= 9` when `sigma <= 0`, and the
/// `sigma = 0.15 * n + 0.35` default.
///
/// Upstream evaluates this in `softdouble` (a software float implementation) to
/// be bit-reproducible across platforms; this port uses hardware `f64`, so the
/// tabulated small kernels are exact while larger ones may differ in the last
/// ULP.
pub fn get_gaussian_kernel(ksize: i32, sigma: f64) -> Result<Vec<f64>> {
    cv_assert!(ksize > 0, "kernel size must be positive, got {ksize}");

    if sigma <= 0.0 {
        // Exact small-kernel tables from getGaussianKernelBitExact.
        let tab: Option<&[f64]> = match ksize {
            1 => Some(&[1.0]),
            3 => Some(&[0.25, 0.5, 0.25]),
            5 => Some(&[0.0625, 0.25, 0.375, 0.25, 0.0625]),
            7 => Some(&[0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]),
            9 => Some(&[
                4.0 / 256.0,
                13.0 / 256.0,
                30.0 / 256.0,
                51.0 / 256.0,
                60.0 / 256.0,
                51.0 / 256.0,
                30.0 / 256.0,
                13.0 / 256.0,
                4.0 / 256.0,
            ]),
            _ => None,
        };
        if let Some(t) = tab {
            return Ok(t.to_vec());
        }
    }

    // Transcribed from `getGaussianKernelBitExact`, operation for operation.
    //
    // Two details are load-bearing and neither is algebraically visible:
    // the default sigma is a *fused* multiply-add, and the sum is built from
    // the half kernel doubled plus the exact centre term rather than by adding
    // all n values. Computing either the obvious way costs a few ULP, and a
    // few ULP in the kernel moves keypoints in every detector downstream.
    let sigma_x = if sigma > 0.0 { sigma } else { (ksize as f64).mul_add(0.15, 0.35) };
    let scale2x = -0.125 / (sigma_x * sigma_x);

    let n = ksize as usize;
    let n2 = (n - 1) / 2;
    let mut values = vec![0.0f64; n2 + 1];
    let mut sum = 0.0f64;
    let mut x = 1i64 - ksize as i64;
    for v in values.iter_mut().take(n2) {
        // x runs in steps of 2, so it is twice the true offset — which is why
        // `scale2x` carries -0.125 rather than -0.5.
        let t = cv_core::softfloat::exp_bit_exact((x * x) as f64 * scale2x);
        *v = t;
        sum += t;
        x += 2;
    }
    sum *= 2.0;
    // The centre term is exactly exp(0) = 1 and is never computed.
    sum += 1.0;
    if n % 2 == 0 {
        sum += 1.0;
    }

    let mul1 = 1.0 / sum;
    let mut out = vec![0.0f64; n];
    for i in 0..n2 {
        let t = values[i] * mul1;
        out[i] = t;
        out[n - 1 - i] = t;
    }
    out[n2] = mul1;
    if n % 2 == 0 {
        out[n2 + 1] = out[n2];
    }
    Ok(out)
}

/// `cv::getStructuringElement`-free helper: a normalised box kernel.
fn box_kernel(n: i32) -> Vec<f64> {
    vec![1.0 / n as f64; n as usize]
}

/// Resolve the default anchor `(-1, -1)` to the kernel centre.
fn resolve_anchor(anchor: Point, kw: i32, kh: i32) -> Point {
    Point::new(
        if anchor.x < 0 { kw / 2 } else { anchor.x },
        if anchor.y < 0 { kh / 2 } else { anchor.y },
    )
}

/// `cv::filter2D(src, dst, ddepth, kernel, anchor, delta, borderType)`.
///
/// `kernel` is given row-major with shape `ksize`. `ddepth = None` keeps the
/// source depth, matching upstream's `ddepth = -1`.
///
/// # Acceleration
///
/// Built with the `rlx` feature, this dispatches to a batched `f32`
/// correlation on the RLX CPU backend when the shape allows, and falls back to
/// the portable kernel otherwise. That path accumulates in `f32` rather than
/// `f64`, so results can differ in the last unit in the last place; the
/// `rlx_matches_portable_*` tests bound the disagreement. Upstream OpenCV
/// behaves the same way when built against IPP or OpenCL.
pub fn filter_2d<S: MatLike + ?Sized>(
    src: &S,
    kernel: &[f64],
    ksize: Size,
    ddepth: Option<Depth>,
    anchor: Point,
    delta: f64,
    border: BorderType,
) -> Result<Mat> {
    cv_assert!(!src.is_empty(), "filter2D requires a non-empty source");
    cv_assert!(ksize.width > 0 && ksize.height > 0, "kernel size must be positive");
    if kernel.len() != (ksize.width * ksize.height) as usize {
        return cv_error!(
            Code::StsUnmatchedSizes,
            "kernel has {} values but ksize is {}x{}",
            kernel.len(),
            ksize.width,
            ksize.height
        );
    }

    #[cfg(feature = "rlx")]
    {
        if let Some(out) = accel::filter_2d(src, kernel, ksize, ddepth, anchor, delta, border)? {
            return Ok(out);
        }
    }
    filter_2d_portable(src, kernel, ksize, ddepth, anchor, delta, border)
}

/// The portable `filter2D` kernel, always available regardless of features.
///
/// Exposed so callers — and the differential tests — can pin the pure-Rust path
/// even in an accelerated build.
pub fn filter_2d_portable<S: MatLike + ?Sized>(
    src: &S,
    kernel: &[f64],
    ksize: Size,
    ddepth: Option<Depth>,
    anchor: Point,
    delta: f64,
    border: BorderType,
) -> Result<Mat> {
    let a = resolve_anchor(anchor, ksize.width, ksize.height);
    let depth = ddepth.unwrap_or_else(|| src.depth());
    let cn = src.channels() as usize;
    let (rows, cols) = (src.rows(), src.cols());

    // Extrapolate once into a padded copy, so the inner loop never consults
    // `border_interpolate` again. Doing it per tap costs two calls per kernel
    // element per pixel — 98 of them for a 7x7 kernel.
    //
    // A constant border contributes zero to the sum, which is exactly what the
    // old per-tap path did when `sample` returned its out-of-range sentinel.
    let padded = cv_core::copy_make_border(
        src,
        a.y,
        ksize.height - 1 - a.y,
        a.x,
        ksize.width - 1 - a.x,
        border,
        cv_core::Scalar::default(),
    )?;
    let pw = padded.cols() as usize;
    let buf = cv_core::to_f64(&padded)?;

    // Precompute the non-zero taps as flat offsets into `buf`. Skipping zero
    // taps matches the previous behaviour and makes sparse kernels much
    // cheaper (a Sobel 3x3 is a third zeros).
    let mut taps: Vec<(usize, f64)> = Vec::with_capacity(kernel.len());
    for ky in 0..ksize.height as usize {
        for kx in 0..ksize.width as usize {
            let k = kernel[ky * ksize.width as usize + kx];
            if k != 0.0 {
                taps.push(((ky * pw + kx) * cn, k));
            }
        }
    }

    let width = cols as usize * cn;
    let mut out = vec![0.0f64; rows as usize * width];
    for y in 0..rows as usize {
        let row_base = y * pw * cn;
        let dst_row = &mut out[y * width..(y + 1) * width];
        for (i, o) in dst_row.iter_mut().enumerate() {
            // `i` walks channel-interleaved, so the channel offset is already
            // folded into it and each tap is a single flat index.
            let base = row_base + i;
            let mut acc = delta;
            for &(off, k) in &taps {
                acc += k * buf[base + off];
            }
            *o = acc;
        }
    }

    cv_core::mat_from_f64(rows, cols, cv_core::MatType::new(depth, cn as i32), &out)
}

/// RLX-backed `filter2D`, compiled only with the `rlx` feature.
#[cfg(feature = "rlx")]
mod accel {
    use super::*;
    use cv_core::{copy_make_border, MatType, Scalar};

    /// Attempt the accelerated path.
    ///
    /// Returns `Ok(None)` when the request is outside what the backend covers,
    /// so the caller falls back rather than failing.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn filter_2d<S: MatLike + ?Sized>(
        src: &S,
        kernel: &[f64],
        ksize: Size,
        ddepth: Option<Depth>,
        anchor: Point,
        delta: f64,
        border: BorderType,
    ) -> Result<Option<Mat>> {
        // BORDER_TRANSPARENT has no meaning for a padded correlation.
        if border == BorderType::Transparent {
            return Ok(None);
        }

        // Only take the accelerated route where it has been measured to win.
        //
        // This gate matters more than it looks, and it has been re-derived
        // twice. When the portable kernel was naive, RLX beat it everywhere.
        // After the portable rewrite, RLX-on-CPU became *slower* — until the
        // RLX-side dispatch fix (`im2col_gemm_is_degenerate`) made its direct
        // convolution kick in for single-channel images.
        //
        // What remains is fixed marshalling cost: pad, widen to f64,
        // planarise to f32, run, de-planarise, store. That is proportional to
        // pixels, while the convolution it replaces is proportional to
        // pixels x kernel area. So the crossover is a *work* threshold, not
        // separate size and kernel thresholds.
        //
        // Measured, 3-channel `CV_8U`, RLX CPU backend:
        //
        // ```text
        //   work = pixels * kernel area
        //   640x480  k3   2.8M   0.74x   loss
        //   256x256  k7   3.2M   ~1x     wash
        //   640x480  k5   7.7M   2.47x   win
        //   1280x720 k3   8.3M   1.27x   win
        //   1280x720 k5  23.0M   2.42x   win
        //   1280x720 k7  45.2M   4.02x   win
        // ```
        const MIN_WORK: u64 = 5_000_000;
        let work = src.total() as u64 * (ksize.width * ksize.height) as u64;
        if work < MIN_WORK {
            return Ok(None);
        }

        // The planar batching below assumes an interleaved image, not an
        // arbitrary-channel tensor.
        let cn = src.channels();
        if !(1..=4).contains(&cn) {
            return Ok(None);
        }

        let a = resolve_anchor(anchor, ksize.width, ksize.height);
        let (kw, kh) = (ksize.width, ksize.height);
        // Pad so the valid-region correlation lands on the original extent.
        let (left, top) = (a.x, a.y);
        let (right, bottom) = (kw - 1 - a.x, kh - 1 - a.y);
        if left < 0 || top < 0 || right < 0 || bottom < 0 {
            return Ok(None);
        }

        // A constant border contributes zero to the sum, which is what the
        // portable kernel does when `sample` returns the sentinel.
        let padded = copy_make_border(src, top, bottom, left, right, border, Scalar::default())?;
        let (ph, pw) = (padded.rows() as usize, padded.cols() as usize);

        // Split into per-channel planes so one kernel serves the whole batch.
        // Marshalling used to dominate this path: converting through
        // `at_f64`/`set_channel` per pixel cost more than the convolution
        // saved, making the "accelerated" route slower than the portable one.
        let flat = cv_core::to_f64(&padded)?;
        let cnu = cn as usize;
        let plane = ph * pw;
        let mut planes = vec![0f32; cnu * plane];
        for c in 0..cnu {
            let dst = &mut planes[c * plane..(c + 1) * plane];
            for (i, o) in dst.iter_mut().enumerate() {
                *o = flat[i * cnu + c] as f32;
            }
        }
        let kf: Vec<f32> = kernel.iter().map(|&v| v as f32).collect();

        let out = cv_accel::correlate2d_f32(
            &planes,
            cnu,
            ph,
            pw,
            &kf,
            kh as usize,
            kw as usize,
        )?;

        let (oh, ow) = (src.rows() as usize, src.cols() as usize);
        if out.len() != cnu * oh * ow {
            return Ok(None);
        }
        // De-planarise back to interleaved and store in one bulk pass.
        let opl = oh * ow;
        let mut inter = vec![0.0f64; cnu * opl];
        for c in 0..cnu {
            let srcp = &out[c * opl..(c + 1) * opl];
            for (i, v) in srcp.iter().enumerate() {
                inter[i * cnu + c] = *v as f64 + delta;
            }
        }
        let depth = ddepth.unwrap_or_else(|| src.depth());
        let dst = cv_core::mat_from_f64(
            src.rows(),
            src.cols(),
            MatType::new(depth, cn),
            &inter,
        )?;
        Ok(Some(dst))
    }
}

/// `cv::sepFilter2D(src, dst, ddepth, kernelX, kernelY, anchor, delta, borderType)`.
///
/// Applies `kernel_x` along rows and `kernel_y` along columns. The intermediate
/// pass is kept in `f64` so no precision is lost between the two halves.
pub fn sep_filter_2d<S: MatLike + ?Sized>(
    src: &S,
    kernel_x: &[f64],
    kernel_y: &[f64],
    ddepth: Option<Depth>,
    anchor: Point,
    delta: f64,
    border: BorderType,
) -> Result<Mat> {
    cv_assert!(!src.is_empty(), "sepFilter2D requires a non-empty source");
    cv_assert!(!kernel_x.is_empty() && !kernel_y.is_empty(), "separable kernels must be non-empty");
    let (kw, kh) = (kernel_x.len() as i32, kernel_y.len() as i32);
    let a = resolve_anchor(anchor, kw, kh);
    let cn = src.channels() as usize;
    let (rows, cols) = (src.rows(), src.cols());

    // Pad in both axes once. The old two-pass version extrapolated
    // horizontally inside pass one and vertically inside pass two, calling
    // `border_interpolate` per tap; padding up front is equivalent, because
    // the horizontal filter of an extrapolated row equals the extrapolated row
    // of the horizontal filter.
    let padded = cv_core::copy_make_border(
        src,
        a.y,
        kh - 1 - a.y,
        a.x,
        kw - 1 - a.x,
        border,
        cv_core::Scalar::default(),
    )?;
    let (ph, pw) = (padded.rows() as usize, padded.cols() as usize);
    let buf = cv_core::to_f64(&padded)?;

    let xt: Vec<(usize, f64)> = kernel_x
        .iter()
        .enumerate()
        .filter(|(_, k)| **k != 0.0)
        .map(|(i, k)| (i * cn, *k))
        .collect();
    let yt: Vec<(usize, f64)> =
        kernel_y.iter().enumerate().filter(|(_, k)| **k != 0.0).map(|(i, k)| (i, *k)).collect();

    // Horizontal pass over every padded row: full height, output width.
    let width = cols as usize * cn;
    let mut tmp = vec![0.0f64; ph * width];
    for y in 0..ph {
        let src_row = &buf[y * pw * cn..(y + 1) * pw * cn];
        let dst_row = &mut tmp[y * width..(y + 1) * width];
        for (i, o) in dst_row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for &(off, k) in &xt {
                acc += k * src_row[i + off];
            }
            *o = acc;
        }
    }

    // Vertical pass over the intermediate.
    let mut out = vec![0.0f64; rows as usize * width];
    for y in 0..rows as usize {
        let dst_row = &mut out[y * width..(y + 1) * width];
        for (i, o) in dst_row.iter_mut().enumerate() {
            let mut acc = delta;
            for &(ky, k) in &yt {
                acc += k * tmp[(y + ky) * width + i];
            }
            *o = acc;
        }
    }

    let depth = ddepth.unwrap_or_else(|| src.depth());
    cv_core::mat_from_f64(rows, cols, cv_core::MatType::new(depth, cn as i32), &out)
}

/// `cv::blur(src, dst, ksize, anchor, borderType)` — normalised box filter.
pub fn blur<S: MatLike + ?Sized>(src: &S, ksize: Size, border: BorderType) -> Result<Mat> {
    cv_assert!(ksize.width > 0 && ksize.height > 0, "blur kernel size must be positive");
    sep_filter_2d(
        src,
        &box_kernel(ksize.width),
        &box_kernel(ksize.height),
        None,
        Point::new(-1, -1),
        0.0,
        border,
    )
}

/// `cv::boxFilter(src, dst, ddepth, ksize, anchor, normalize, borderType)`.
pub fn box_filter<S: MatLike + ?Sized>(
    src: &S,
    ksize: Size,
    normalize: bool,
    border: BorderType,
) -> Result<Mat> {
    cv_assert!(ksize.width > 0 && ksize.height > 0, "box kernel size must be positive");
    let (kx, ky) = if normalize {
        (box_kernel(ksize.width), box_kernel(ksize.height))
    } else {
        (vec![1.0; ksize.width as usize], vec![1.0; ksize.height as usize])
    };
    sep_filter_2d(src, &kx, &ky, None, Point::new(-1, -1), 0.0, border)
}

/// `cv::getGaussianKernel` quantised to fixed point by error diffusion.
///
/// This is `getGaussianKernelFixedPoint_ED`. The quantisation is not
/// independent per tap: the residual from each coefficient is carried into the
/// next, and the centre tap is then set to whatever makes the weights sum to
/// exactly `1 << shift`. That exactness is the point — it guarantees a flat
/// image survives the filter unchanged with no renormalisation.
///
/// `shift = 8` reproduces upstream's `ufixedpoint16` path for `CV_8U`.
pub fn get_gaussian_kernel_fixed(ksize: i32, sigma: f64, shift: u32) -> Result<Vec<i64>> {
    cv_assert!(ksize > 0 && ksize % 2 == 1, "fixed-point Gaussian needs an odd ksize");
    cv_assert!(shift > 0 && shift <= 32, "shift must be in 1..=32");
    let k = get_gaussian_kernel(ksize, sigma)?;
    let n = ksize as usize;
    let mult = 1i64 << shift;

    let mut res = vec![0i64; n];
    let mut err = 0.0f64;
    let mut sum = 0i64;
    for i in 0..n / 2 {
        // Carry the previous tap's rounding residual into this one.
        let adj = k[i] * mult as f64 + err;
        let v0 = adj.round_ties_even() as i64;
        err = adj - v0 as f64;
        res[i] = v0;
        res[n - 1 - i] = v0;
        sum += v0;
    }
    // The centre absorbs the remainder so the weights total exactly `mult`.
    res[n / 2] = mult - 2 * sum;
    Ok(res)
}

/// `cv::GaussianBlur(src, dst, ksize, sigmaX, sigmaY, borderType)`.
///
/// A zero `ksize` dimension is derived from sigma the way upstream does:
/// `ksize = cvRound(sigma * (depth == CV_8U ? 3 : 4) * 2 + 1) | 1`.
///
/// # `CV_8U` takes a separate path
///
/// Upstream's 8-bit `GaussianBlur` is **not** `sepFilter2D` with a float
/// kernel. It quantises the kernel to 8-bit fixed point (summing to exactly
/// 256) and filters in integers. The difference is observable: a real OpenCV
/// build's own `sepFilter2D`, given its own `getGaussianKernel` output,
/// disagrees with its own `GaussianBlur` on 5.6% of pixels. This port
/// reproduces the fixed-point path, verified bit-exact against `cv2`.
pub fn gaussian_blur<S: MatLike + ?Sized>(
    src: &S,
    ksize: Size,
    sigma_x: f64,
    sigma_y: f64,
    border: BorderType,
) -> Result<Mat> {
    let sy = if sigma_y <= 0.0 { sigma_x } else { sigma_y };
    let depth_factor = if src.depth() == Depth::U8 { 3.0 } else { 4.0 };

    let kw = if ksize.width > 0 {
        ksize.width
    } else {
        cv_core::cv_round(sigma_x * depth_factor * 2.0 + 1.0) | 1
    };
    let kh = if ksize.height > 0 {
        ksize.height
    } else {
        cv_core::cv_round(sy * depth_factor * 2.0 + 1.0) | 1
    };
    cv_assert!(kw > 0 && kw % 2 == 1, "GaussianBlur width must be positive and odd, got {kw}");
    cv_assert!(kh > 0 && kh % 2 == 1, "GaussianBlur height must be positive and odd, got {kh}");

    if src.depth() == Depth::U8 {
        return gaussian_blur_fixed(src, kw, kh, sigma_x, sy, border);
    }
    let gx = get_gaussian_kernel(kw, sigma_x)?;
    let gy = get_gaussian_kernel(kh, sy)?;
    if src.depth() == Depth::F32 {
        return gaussian_blur_f32(src, &gx, &gy, border);
    }
    sep_filter_2d(src, &gx, &gy, None, Point::new(-1, -1), 0.0, border)
}

/// The `CV_32F` path for [`gaussian_blur`].
///
/// # Why this is not [`sep_filter_2d`]
///
/// That routine accumulates in `f64`, which is *more* accurate but is not what
/// upstream does. For a `CV_32F` image OpenCV builds an `f32` kernel and
/// accumulates in `f32`, and its two passes are not symmetric with each other:
///
/// * the **row** filter, for kernels above size 5, sums tap by tap;
/// * the **column** filter (`SymmColumnFilter`) sums each mirrored pair *first*
///   and scales it by the pair's single coefficient, fusing the multiply.
///
/// In `f32`, `f*(a+b)` and `f*a + f*b` differ. Using a plain sum in both passes
/// leaves the rows bit-exact and the columns about one ULP out on 60% of
/// pixels — invisible once an `f32` result is rounded back to `u8`, and quite
/// visible to anything that then differentiates the result.
fn gaussian_blur_f32<S: MatLike + ?Sized>(
    src: &S,
    gx: &[f64],
    gy: &[f64],
    border: BorderType,
) -> Result<Mat> {
    let (rows, cols) = (src.rows(), src.cols());
    let cn = src.channels();
    let mut buf = vec![0.0f32; (rows * cols * cn) as usize];
    for y in 0..rows {
        for x in 0..cols {
            for c in 0..cn {
                buf[((y * cols + x) * cn + c) as usize] = src.at_f64(y, x, c)? as f32;
            }
        }
    }
    let blurred = gaussian_blur_planar(&buf, cols, rows, cn, gx, gy, border)?;

    let mut out = Mat::zeros(rows, cols, src.mat_type())?;
    for y in 0..rows {
        for x in 0..cols {
            for c in 0..cn {
                let v = blurred[((y * cols + x) * cn + c) as usize];
                cv_core::arithm::set_channel_generic(&mut out, y, x, c, f64::from(v))?;
            }
        }
    }
    Ok(out)
}

/// [`gaussian_blur`]'s `f32` core, over an interleaved slice.
///
/// Exposed because KAZE's nonlinear scale space blurs `f32` planes thousands of
/// times and must do it exactly as upstream does — the asymmetry documented on
/// `gaussian_blur_f32` is the whole point, and duplicating it would be a
/// standing invitation for the two copies to drift apart.
///
/// `gx` and `gy` are the exact `f64` taps; each is rounded once to `f32`, as
/// upstream does when it builds the kernel at `CV_32F` for an `f32` image.
pub fn gaussian_blur_planar(
    src: &[f32],
    cols: i32,
    rows: i32,
    cn: i32,
    gx: &[f64],
    gy: &[f64],
    border: BorderType,
) -> Result<Vec<f32>> {
    cv_assert!(
        src.len() == (rows * cols * cn) as usize,
        "expected {} samples for {rows}x{cols}x{cn}, got {}",
        rows * cols * cn,
        src.len()
    );
    let kx: Vec<f32> = gx.iter().map(|v| *v as f32).collect();
    let ky: Vec<f32> = gy.iter().map(|v| *v as f32).collect();
    let (rx, ry) = (kx.len() as i32 / 2, ky.len() as i32 / 2);

    // Border indices are the same for every row (and every column), so resolve
    // each axis once instead of once per pixel — this runs on every diffusion
    // step and the interpolation is not cheap.
    let cols_at: Vec<Vec<usize>> = (0..cols)
        .map(|x| {
            (0..kx.len())
                .map(|i| {
                    cv_core::border::border_interpolate(x + i as i32 - rx, cols, border)
                        .map(|v| v as usize)
                })
                .collect::<Result<_>>()
        })
        .collect::<Result<_>>()?;

    // Row pass: plain tap-by-tap sum, into an f32 buffer.
    let w = (cols * cn) as usize;
    let mut tmp = vec![0.0f32; rows as usize * w];
    for y in 0..rows as usize {
        for x in 0..cols as usize {
            for c in 0..cn as usize {
                let mut acc = 0.0f32;
                for (i, k) in kx.iter().enumerate() {
                    acc = k.mul_add(src[(y * w) + cols_at[x][i] * cn as usize + c], acc);
                }
                tmp[y * w + x * cn as usize + c] = acc;
            }
        }
    }

    // Column pass: symmetric pairs, as upstream's SymmColumnFilter does.
    let mut out = vec![0.0f32; rows as usize * w];
    for y in 0..rows {
        // One resolved index per offset, reused across the whole row.
        let rows_at: Vec<(usize, usize)> = (1..=ry)
            .map(|j| {
                Ok((
                    cv_core::border::border_interpolate(y + j, rows, border)? as usize,
                    cv_core::border::border_interpolate(y - j, rows, border)? as usize,
                ))
            })
            .collect::<Result<_>>()?;
        for idx in 0..w {
            let mut acc = ky[ry as usize] * tmp[y as usize * w + idx];
            for (j, &(sp, sm)) in rows_at.iter().enumerate() {
                let pair = tmp[sp * w + idx] + tmp[sm * w + idx];
                acc = ky[ry as usize + j + 1].mul_add(pair, acc);
            }
            out[y as usize * w + idx] = acc;
        }
    }
    Ok(out)
}

/// The `CV_8U` integer path for [`gaussian_blur`].
fn gaussian_blur_fixed<S: MatLike + ?Sized>(
    src: &S,
    kw: i32,
    kh: i32,
    sigma_x: f64,
    sigma_y: f64,
    border: BorderType,
) -> Result<Mat> {
    /// Fractional bits, matching upstream's `ufixedpoint16`.
    const SHIFT: u32 = 8;
    let kx = get_gaussian_kernel_fixed(kw, sigma_x, SHIFT)?;
    let ky = get_gaussian_kernel_fixed(kh, sigma_y, SHIFT)?;

    let cn = src.channels() as usize;
    let (rows, cols) = (src.rows(), src.cols());
    let (rx, ry) = (kw / 2, kh / 2);
    let padded = cv_core::copy_make_border(src, ry, ry, rx, rx, border, cv_core::Scalar::default())?;
    let (ph, pw) = (padded.rows() as usize, padded.cols() as usize);
    let buf = cv_core::to_f64(&padded)?;

    // Horizontal pass over every padded row, accumulating in units of 2^SHIFT.
    let width = cols as usize * cn;
    let mut tmp = vec![0i64; ph * width];
    for y in 0..ph {
        let srow = &buf[y * pw * cn..(y + 1) * pw * cn];
        let drow = &mut tmp[y * width..(y + 1) * width];
        for (i, o) in drow.iter_mut().enumerate() {
            let mut acc = 0i64;
            for (k, &kv) in kx.iter().enumerate() {
                acc += srow[i + k * cn] as i64 * kv;
            }
            *o = acc;
        }
    }

    // Vertical pass, then one rounding shift by 2*SHIFT.
    let round = 1i64 << (2 * SHIFT - 1);
    let mut out = vec![0.0f64; rows as usize * width];
    for y in 0..rows as usize {
        for i in 0..width {
            let mut acc = 0i64;
            for (k, &kv) in ky.iter().enumerate() {
                acc += tmp[(y + k) * width + i] * kv;
            }
            out[y * width + i] = ((acc + round) >> (2 * SHIFT)) as f64;
        }
    }
    cv_core::mat_from_f64(rows, cols, src.mat_type(), &out)
}

/// `cv::medianBlur(src, dst, ksize)`.
///
/// The median is taken per channel over a `ksize x ksize` window, with
/// `BORDER_REPLICATE` extrapolation — the border mode upstream hard-codes.
pub fn median_blur<S: MatLike + ?Sized>(src: &S, ksize: i32) -> Result<Mat> {
    cv_assert!(ksize > 1 && ksize % 2 == 1, "medianBlur ksize must be odd and > 1, got {ksize}");
    cv_assert!(!src.is_empty(), "medianBlur requires a non-empty source");
    let cn = src.channels() as usize;
    let r = ksize / 2;
    let (rows, cols) = (src.rows(), src.cols());

    // Replicate padding, applied once, matching the border mode upstream
    // hard-codes for this filter.
    let padded =
        cv_core::copy_make_border(src, r, r, r, r, BorderType::Replicate, cv_core::Scalar::default())?;
    let pw = padded.cols() as usize;
    let buf = cv_core::to_f64(&padded)?;

    let k = ksize as usize;
    let offsets: Vec<usize> =
        (0..k).flat_map(|ky| (0..k).map(move |kx| (ky * pw + kx) * cn)).collect();
    let mid = offsets.len() / 2;

    let width = cols as usize * cn;
    let mut out = vec![0.0f64; rows as usize * width];
    let mut window: Vec<f64> = vec![0.0; offsets.len()];

    for y in 0..rows as usize {
        let row_base = y * pw * cn;
        for i in 0..width {
            let base = row_base + i;
            for (w, &off) in window.iter_mut().zip(offsets.iter()) {
                *w = buf[base + off];
            }
            // `select_nth_unstable` is O(n) where a full sort is O(n log n),
            // and only the middle element is needed.
            window.select_nth_unstable_by(mid, |a, b| {
                a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
            });
            out[y * width + i] = window[mid];
        }
    }

    cv_core::mat_from_f64(rows, cols, src.mat_type(), &out)
}

/// `cv::getDerivKernels(dx, dy, ksize)` — the Sobel/Scharr separable pair.
///
/// Returns `(kernel_x, kernel_y)`. `ksize` must be 1, 3, 5 or 7;
/// `ksize == -1` selects the 3x3 Scharr kernel, as upstream's `CV_SCHARR` does.
pub fn get_deriv_kernels(dx: i32, dy: i32, ksize: i32) -> Result<(Vec<f64>, Vec<f64>)> {
    cv_assert!(dx >= 0 && dy >= 0 && dx + dy > 0, "invalid derivative orders {dx},{dy}");
    // Upstream dispatches to Scharr for any non-positive ksize, not just -1.
    if ksize <= 0 {
        cv_assert!(dx + dy == 1, "Scharr requires dx + dy == 1");
        let smooth = vec![3.0, 10.0, 3.0];
        let deriv = vec![-1.0, 0.0, 1.0];
        return Ok(if dx == 1 {
            (deriv, smooth)
        } else {
            (smooth, deriv)
        });
    }
    cv_assert!(
        ksize > 0 && ksize % 2 == 1 && ksize <= 7,
        "Sobel ksize must be 1, 3, 5 or 7, got {ksize}"
    );
    let kx = sobel_1d(dx, ksize)?;
    let ky = sobel_1d(dy, ksize)?;
    Ok((kx, ky))
}

/// One axis of a Sobel kernel — a direct transcription of `getSobelKernels`.
///
/// The in-place `oldval`/`newval` rotation below looks odd out of context, but
/// it is exactly upstream's formulation: `ksize - order - 1` smoothing passes
/// build a binomial row, then `order` difference passes differentiate it.
fn sobel_1d(order: i32, ksize: i32) -> Result<Vec<f64>> {
    cv_assert!(order < ksize, "derivative order {order} must be below ksize {ksize}");
    let n = ksize as usize;

    if ksize == 1 {
        return Ok(vec![1.0]);
    }
    if ksize == 3 {
        return Ok(match order {
            0 => vec![1.0, 2.0, 1.0],
            1 => vec![-1.0, 0.0, 1.0],
            _ => vec![1.0, -2.0, 1.0],
        });
    }

    // One slot longer than the kernel; the loops read `ker[j]` up to `j == n`.
    let mut ker = vec![0i64; n + 1];
    ker[0] = 1;

    for _ in 0..(ksize - order - 1) {
        let mut oldval = ker[0];
        for j in 1..=n {
            let newval = ker[j] + ker[j - 1];
            ker[j - 1] = oldval;
            oldval = newval;
        }
    }
    for _ in 0..order {
        let mut oldval = -ker[0];
        for j in 1..=n {
            let newval = ker[j - 1] - ker[j];
            ker[j - 1] = oldval;
            oldval = newval;
        }
    }
    Ok(ker[..n].iter().map(|&v| v as f64).collect())
}

/// `cv::Sobel(src, dst, ddepth, dx, dy, ksize, scale, delta, borderType)`.
// The argument list mirrors upstream's one-for-one; renaming or bundling them
// into a struct would make the correspondence harder to audit.
#[allow(clippy::too_many_arguments)]
pub fn sobel<S: MatLike + ?Sized>(
    src: &S,
    ddepth: Option<Depth>,
    dx: i32,
    dy: i32,
    ksize: i32,
    scale: f64,
    delta: f64,
    border: BorderType,
) -> Result<Mat> {
    let (mut kx, ky) = get_deriv_kernels(dx, dy, ksize)?;
    if scale != 1.0 {
        for v in kx.iter_mut() {
            *v *= scale;
        }
    }
    sep_filter_2d(src, &kx, &ky, ddepth, Point::new(-1, -1), delta, border)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{MatLikeMut, MatType, Scalar};

    fn u8m(rows: i32, cols: i32, v: &[u8]) -> Mat {
        Mat::from_vec(rows, cols, 1, v).unwrap()
    }

    #[test]
    fn gaussian_kernels_match_the_exact_small_tables() {
        assert_eq!(get_gaussian_kernel(1, 0.0).unwrap(), vec![1.0]);
        assert_eq!(get_gaussian_kernel(3, 0.0).unwrap(), vec![0.25, 0.5, 0.25]);
        assert_eq!(get_gaussian_kernel(5, 0.0).unwrap(), vec![0.0625, 0.25, 0.375, 0.25, 0.0625]);
        assert_eq!(get_gaussian_kernel(7, -1.0).unwrap()[3], 0.28125);
    }

    #[test]
    fn gaussian_kernels_are_normalised_and_symmetric() {
        for n in [3, 5, 7, 9, 11, 15, 21] {
            for sigma in [0.8, 1.0, 2.5, 5.0] {
                let k = get_gaussian_kernel(n, sigma).unwrap();
                assert_eq!(k.len(), n as usize);
                let s: f64 = k.iter().sum();
                assert!((s - 1.0).abs() < 1e-12, "n={n} sigma={sigma} sum={s}");
                for i in 0..k.len() / 2 {
                    assert!((k[i] - k[k.len() - 1 - i]).abs() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn default_sigma_follows_the_upstream_formula() {
        // sigma = 0.15*n + 0.35 when sigma <= 0 and n is not tabulated.
        let k = get_gaussian_kernel(11, 0.0).unwrap();
        let expected_sigma = 11.0 * 0.15 + 0.35;
        let manual = get_gaussian_kernel(11, expected_sigma).unwrap();
        for (a, b) in k.iter().zip(manual.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn identity_kernel_is_a_copy() {
        let m = u8m(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let r = filter_2d(&m, &[1.0], Size::new(1, 1), None, Point::new(-1, -1), 0.0, BorderType::Reflect101).unwrap();
        assert_eq!(r.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn filter2d_correlates_rather_than_convolves() {
        // An asymmetric kernel distinguishes the two. With [0,0,1] as a 1x3
        // correlation kernel and anchor at the centre, the output picks up the
        // pixel to the RIGHT.
        let m = u8m(1, 3, &[10, 20, 30]);
        let r = filter_2d(&m, &[0.0, 0.0, 1.0], Size::new(3, 1), None, Point::new(-1, -1), 0.0, BorderType::Replicate).unwrap();
        assert_eq!(r.to_vec::<u8>().unwrap(), vec![20, 30, 30]);
    }

    #[test]
    fn filter2d_delta_is_added() {
        let m = u8m(1, 2, &[10, 20]);
        let r = filter_2d(&m, &[1.0], Size::new(1, 1), None, Point::new(-1, -1), 5.0, BorderType::Replicate).unwrap();
        assert_eq!(r.to_vec::<u8>().unwrap(), vec![15, 25]);
    }

    #[test]
    fn constant_border_contributes_zero() {
        let m = u8m(1, 1, &[100]);
        // 1x3 averaging kernel; with a constant (zero) border the result is 100/3.
        let r = filter_2d(&m, &[1.0 / 3.0; 3], Size::new(3, 1), None, Point::new(-1, -1), 0.0, BorderType::Constant).unwrap();
        assert_eq!(r.to_vec::<u8>().unwrap(), vec![33]);
    }

    #[test]
    fn blur_of_a_uniform_image_is_the_same_image() {
        let m = Mat::new_with(5, 5, MatType::new(Depth::U8, 3), Scalar::new(10.0, 20.0, 30.0, 0.0)).unwrap();
        let r = blur(&m, Size::new(3, 3), BorderType::Reflect101).unwrap();
        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(r.pixel::<u8>(y, x).unwrap(), &[10, 20, 30], "at {y},{x}");
            }
        }
    }

    #[test]
    fn blur_averages_a_known_window() {
        let m = u8m(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let r = blur(&m, Size::new(3, 3), BorderType::Replicate).unwrap();
        // Centre pixel is the mean of 1..9 = 5.
        assert_eq!(r.at::<u8>(1, 1).unwrap(), 5);
    }

    #[test]
    fn box_filter_unnormalised_sums() {
        let m = u8m(3, 3, &[1, 1, 1, 1, 1, 1, 1, 1, 1]);
        let r = box_filter(&m, Size::new(3, 3), false, BorderType::Replicate).unwrap();
        assert_eq!(r.at::<u8>(1, 1).unwrap(), 9);
    }

    #[test]
    fn gaussian_blur_preserves_a_uniform_image() {
        let m = Mat::new_with(7, 7, MatType::new(Depth::U8, 1), Scalar::single(123.0)).unwrap();
        let r = gaussian_blur(&m, Size::new(5, 5), 1.5, 0.0, BorderType::Reflect101).unwrap();
        assert!(r.to_vec::<u8>().unwrap().iter().all(|&v| v == 123));
    }

    #[test]
    fn gaussian_blur_derives_ksize_from_sigma() {
        let m = Mat::new_with(9, 9, MatType::new(Depth::U8, 1), Scalar::single(50.0)).unwrap();
        // ksize 0 must not error; it is derived from sigma.
        let r = gaussian_blur(&m, Size::new(0, 0), 1.0, 0.0, BorderType::Reflect101).unwrap();
        assert_eq!(r.size(), Size::new(9, 9));
    }

    #[test]
    fn gaussian_blur_rejects_even_kernels() {
        let m = Mat::zeros(5, 5, MatType::new(Depth::U8, 1)).unwrap();
        assert!(gaussian_blur(&m, Size::new(4, 3), 1.0, 0.0, BorderType::Reflect101).is_err());
    }

    #[test]
    fn median_blur_removes_salt_and_pepper() {
        let m = u8m(3, 3, &[5, 5, 5, 5, 255, 5, 5, 5, 5]);
        let r = median_blur(&m, 3).unwrap();
        assert_eq!(r.at::<u8>(1, 1).unwrap(), 5);
    }

    #[test]
    fn median_blur_preserves_a_step_edge() {
        // A median filter must not blur a clean edge the way a box filter does.
        let m = u8m(1, 6, &[0, 0, 0, 255, 255, 255]);
        let r = median_blur(&m, 3).unwrap();
        assert_eq!(r.to_vec::<u8>().unwrap(), vec![0, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn median_blur_rejects_even_ksize() {
        let m = Mat::zeros(3, 3, MatType::new(Depth::U8, 1)).unwrap();
        assert!(median_blur(&m, 2).is_err());
        assert!(median_blur(&m, 1).is_err());
    }

    #[test]
    fn sobel_kernels_match_the_classic_3x3() {
        let (kx, ky) = get_deriv_kernels(1, 0, 3).unwrap();
        assert_eq!(kx, vec![-1.0, 0.0, 1.0]);
        assert_eq!(ky, vec![1.0, 2.0, 1.0]);

        let (kx, ky) = get_deriv_kernels(0, 1, 3).unwrap();
        assert_eq!(kx, vec![1.0, 2.0, 1.0]);
        assert_eq!(ky, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn scharr_kernels_match_upstream() {
        let (kx, ky) = get_deriv_kernels(1, 0, -1).unwrap();
        assert_eq!(kx, vec![-1.0, 0.0, 1.0]);
        assert_eq!(ky, vec![3.0, 10.0, 3.0]);
    }

    #[test]
    fn sobel_5x5_x_kernel_matches_upstream() {
        let (kx, ky) = get_deriv_kernels(1, 0, 5).unwrap();
        assert_eq!(kx, vec![-1.0, -2.0, 0.0, 2.0, 1.0]);
        assert_eq!(ky, vec![1.0, 4.0, 6.0, 4.0, 1.0]);
    }

    #[test]
    fn sobel_detects_a_vertical_edge() {
        let m = u8m(3, 4, &[0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255]);
        let g = sobel(&m, Some(Depth::F32), 1, 0, 3, 1.0, 0.0, BorderType::Replicate).unwrap();
        // Strong positive response at the transition column.
        assert!(g.at::<f32>(1, 1).unwrap() > 500.0, "{}", g.at::<f32>(1, 1).unwrap());
        // Flat regions respond with zero.
        assert_eq!(g.at::<f32>(1, 3).unwrap(), 0.0);
    }

    #[test]
    fn sobel_of_a_flat_image_is_zero() {
        let m = Mat::new_with(5, 5, MatType::new(Depth::U8, 1), Scalar::single(100.0)).unwrap();
        let g = sobel(&m, Some(Depth::F32), 1, 0, 3, 1.0, 0.0, BorderType::Reflect101).unwrap();
        assert!(g.to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn separable_matches_the_equivalent_2d_kernel() {
        let m = u8m(5, 5, &(0u8..25).collect::<Vec<_>>());
        let kx = vec![0.25, 0.5, 0.25];
        let ky = vec![0.25, 0.5, 0.25];
        let sep = sep_filter_2d(&m, &kx, &ky, Some(Depth::F32), Point::new(-1, -1), 0.0, BorderType::Reflect101).unwrap();

        let mut full = vec![0.0; 9];
        for (i, a) in ky.iter().enumerate() {
            for (j, b) in kx.iter().enumerate() {
                full[i * 3 + j] = a * b;
            }
        }
        let two_d = filter_2d(&m, &full, Size::new(3, 3), Some(Depth::F32), Point::new(-1, -1), 0.0, BorderType::Reflect101).unwrap();

        for (a, b) in sep.to_vec::<f32>().unwrap().iter().zip(two_d.to_vec::<f32>().unwrap().iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    /// Build a deterministic test image large enough to clear the accelerated
    /// path's size threshold (see `accel::filter_2d`, which needs >= 200k
    /// pixels *and* a >= 7x7 kernel before it will dispatch).
    fn accel_image(cn: i32) -> Mat {
        let (w, h) = (520, 400);
        let mut m = Mat::zeros(h, w, MatType::new(Depth::U8, cn)).unwrap();
        for y in 0..h {
            for x in 0..w {
                let p = m.pixel_mut::<u8>(y, x).unwrap();
                for (c, slot) in p.iter_mut().enumerate() {
                    *slot = (((x * 7 + y * 13 + c as i32 * 29) % 256) as u8).wrapping_add(c as u8);
                }
            }
        }
        m
    }

    /// The dispatching `filter_2d` must agree with the portable kernel.
    ///
    /// In a default build these are the same code path, so this is a tautology
    /// that still guards the plumbing. Built with `--features rlx` it is the
    /// real check: the RLX path accumulates in f32 and must stay within one
    /// level of the f64 reference on 8-bit output.
    #[test]
    fn rlx_matches_portable_u8() {
        // A 7x7 kernel is what actually trips the accelerated path; the 3x3
        // entries confirm the fallback still agrees.
        let kernels: [(Vec<f64>, Size); 3] = [
            (vec![1.0 / 49.0; 49], Size::new(7, 7)),
            (vec![1.0 / 9.0; 9], Size::new(3, 3)),
            (vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0], Size::new(3, 3)),
        ];
        for cn in [1, 3] {
            let img = accel_image(cn);
            for (k, ks) in &kernels {
                for border in [BorderType::Replicate, BorderType::Reflect101, BorderType::Constant]
                {
                    let fast =
                        filter_2d(&img, k, *ks, None, Point::new(-1, -1), 0.0, border).unwrap();
                    let slow = filter_2d_portable(
                        &img,
                        k,
                        *ks,
                        None,
                        Point::new(-1, -1),
                        0.0,
                        border,
                    )
                    .unwrap();
                    let (a, b) = (fast.to_vec::<u8>().unwrap(), slow.to_vec::<u8>().unwrap());
                    assert_eq!(a.len(), b.len());
                    let worst = a
                        .iter()
                        .zip(b.iter())
                        .map(|(p, q)| (*p as i32 - *q as i32).abs())
                        .max()
                        .unwrap_or(0);
                    assert!(worst <= 1, "cn={cn} {ks:?} {border:?}: worst diff {worst}");
                }
            }
        }
    }

    #[test]
    fn rlx_matches_portable_f32() {
        let src = {
            let mut m = Mat::zeros(400, 520, MatType::new(Depth::F32, 1)).unwrap();
            for y in 0..400 {
                for x in 0..520 {
                    m.pixel_mut::<f32>(y, x).unwrap()[0] = ((x * 3 + y * 5) % 101) as f32 / 101.0;
                }
            }
            m
        };
        let k = vec![1.0 / 49.0; 49];
        let ks = Size::new(7, 7);
        let fast = filter_2d(&src, &k, ks, None, Point::new(-1, -1), 0.0, BorderType::Reflect101)
            .unwrap();
        let slow =
            filter_2d_portable(&src, &k, ks, None, Point::new(-1, -1), 0.0, BorderType::Reflect101)
                .unwrap();
        for (p, q) in fast.to_vec::<f32>().unwrap().iter().zip(slow.to_vec::<f32>().unwrap().iter())
        {
            assert!((p - q).abs() < 1e-5, "{p} vs {q}");
        }
    }

    #[test]
    fn rlx_respects_the_anchor_and_delta() {
        // An off-centre anchor shifts the result; the accelerated path pads
        // asymmetrically to match, so this catches a padding sign error.
        let img = accel_image(1);
        // A 7x7 kernel tapping one place right of centre: large enough to
        // dispatch, asymmetric enough to catch a padding sign error.
        let mut k = vec![0.0; 49];
        k[3 * 7 + 4] = 1.0;
        let ks = Size::new(7, 7);
        for anchor in [Point::new(-1, -1), Point::new(0, 0), Point::new(5, 2)] {
            let fast =
                filter_2d(&img, &k, ks, None, anchor, 7.0, BorderType::Replicate).unwrap();
            let slow = filter_2d_portable(&img, &k, ks, None, anchor, 7.0, BorderType::Replicate)
                .unwrap();
            assert_eq!(
                fast.to_vec::<u8>().unwrap(),
                slow.to_vec::<u8>().unwrap(),
                "anchor {anchor:?}"
            );
        }
    }

    #[test]
    fn small_images_still_produce_correct_results() {
        // Below the acceleration threshold the dispatcher must fall through
        // cleanly rather than skip the filter.
        let img = Mat::new_with(4, 4, MatType::new(Depth::U8, 1), Scalar::single(80.0)).unwrap();
        let out = filter_2d(
            &img,
            &[1.0 / 9.0; 9],
            Size::new(3, 3),
            None,
            Point::new(-1, -1),
            0.0,
            BorderType::Replicate,
        )
        .unwrap();
        assert!(out.to_vec::<u8>().unwrap().iter().all(|&v| v == 80));
    }

    #[test]
    fn mismatched_kernel_length_is_rejected() {
        let m = u8m(3, 3, &[0; 9]);
        let e = filter_2d(&m, &[1.0, 2.0], Size::new(3, 3), None, Point::new(-1, -1), 0.0, BorderType::Replicate)
            .unwrap_err();
        assert_eq!(e.code, Code::StsUnmatchedSizes);
    }

    /// `getGaussianKernel` is bit-exact against cv2 4.13.0.
    ///
    /// Pinned as raw `f64` bits, because the whole point is the last bit: this
    /// kernel feeds every Gaussian blur, and a 1-ULP difference here moves
    /// keypoints in SIFT and KAZE downstream. Reaching it needed OpenCV's own
    /// `exp` — see [`cv_core::softfloat`] for why hardware `exp` is not enough.
    #[test]
    fn gaussian_kernel_is_bit_exact_against_cv2() {
        let cases: &[(i32, f64, &[u64])] = &[
            (5, 1.0_f64, &[0x3fabe5f0dc491a0f, 0x3fcf41fd54c58786, 0x3fd9c486742831f7, 0x3fcf41fd54c58786, 0x3fabe5f0dc491a0f]),
            (7, 1.6_f64, &[0x3fa69a6046bffc72, 0x3fbe0251cbe73c6e, 0x3fcaf550684d54ac, 0x3fd062eea00f0e00, 0x3fcaf550684d54ac, 0x3fbe0251cbe73c6e, 0x3fa69a6046bffc72]),
            (9, 2.0_f64, &[0x3f9c4b2edd754a9a, 0x3fb0f7df8bcf7483, 0x3fbfb36c70cb525a, 0x3fc70fef927ff566, 0x3fca22092707fbb5, 0x3fc70fef927ff566, 0x3fbfb36c70cb525a, 0x3fb0f7df8bcf7483, 0x3f9c4b2edd754a9a]),
            (21, 0.0_f64, &[0x3f5f9b153fc15b37, 0x3f7128d9d79a663f, 0x3f812c0f7eac007f, 0x3f8facd4b62e23bb, 0x3f9aec655ccf5b59, 0x3fa51733bb1fca28, 0x3fae73e6b99f23ac, 0x3fb4430cea9d97ea, 0x3fb8d963c175abea, 0x3fbc1608953dc51d, 0x3fbd4193680f71f3, 0x3fbc1608953dc51d, 0x3fb8d963c175abea, 0x3fb4430cea9d97ea, 0x3fae73e6b99f23ac, 0x3fa51733bb1fca28, 0x3f9aec655ccf5b59, 0x3f8facd4b62e23bb, 0x3f812c0f7eac007f, 0x3f7128d9d79a663f, 0x3f5f9b153fc15b37]),
        ];
        for (n, sigma, want) in cases {
            let got = get_gaussian_kernel(*n, *sigma).unwrap();
            assert_eq!(got.len(), want.len(), "n={n}");
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    *w,
                    "n={n} sigma={sigma} tap {i}: {g:?} vs cv2 {:?}",
                    f64::from_bits(*w)
                );
            }
            // And it still sums to one, which the bit comparison alone would
            // not catch if both sides were wrong the same way.
            let s: f64 = got.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "n={n} sums to {s}");
        }
    }

    /// `GaussianBlur` is bit-exact against cv2 4.13.0.
    ///
    /// A sweep over three image sizes and five kernel/sigma combinations was
    /// exact on every pixel; this pins a representative slice. Exactness here
    /// rests on [`get_gaussian_kernel`] being bit-exact, which in turn rests on
    /// [`cv_core::softfloat::exp_bit_exact`] — the chain is only as good as its
    /// first link, so both ends are pinned.
    #[test]
    fn gaussian_blur_is_bit_exact_against_cv2() {
        const SRC: [u8; 80] = [114, 104, 41, 77, 130, 82, 247, 199, 163, 92, 212, 98, 122, 202, 24, 155, 237, 103, 79, 119, 227, 135, 181, 181, 222, 196, 19, 88, 66, 30, 207, 22, 56, 63, 78, 75, 253, 127, 118, 161, 45, 7, 238, 158, 1, 105, 22, 251, 197, 165, 13, 62, 8, 72, 100, 108, 79, 29, 71, 242, 206, 114, 198, 28, 3, 159, 81, 179, 35, 243, 80, 245, 194, 204, 215, 234, 32, 43, 13, 173];
        let m = Mat::from_vec(8, 10, 1, &SRC).unwrap();
        let cases: &[(i32, f64, &[u8])] = &[
            (5, 1.0_f64, &[132, 122, 115, 118, 121, 143, 165, 149, 121, 109, 141, 131, 126, 130, 130, 143, 152, 132, 108, 99, 140, 131, 130, 136, 136, 137, 133, 117, 102, 97, 106, 102, 110, 113, 110, 116, 127, 131, 129, 127, 71, 78, 97, 98, 87, 94, 112, 133, 147, 153, 80, 86, 96, 90, 86, 94, 98, 109, 129, 146, 127, 130, 125, 108, 104, 111, 100, 94, 107, 125, 152, 155, 148, 128, 123, 125, 102, 87, 95, 113]),
            (7, 1.6_f64, &[131, 128, 126, 129, 133, 139, 139, 130, 118, 113, 129, 127, 126, 129, 132, 137, 137, 129, 119, 114, 120, 119, 120, 122, 125, 129, 130, 126, 120, 118, 107, 107, 109, 111, 113, 118, 122, 124, 126, 126, 99, 99, 102, 104, 104, 108, 114, 121, 127, 129, 104, 104, 105, 103, 102, 103, 108, 115, 123, 125, 119, 117, 115, 110, 107, 104, 105, 107, 114, 116, 128, 126, 122, 116, 111, 105, 103, 104, 109, 110]),
            (0, 1.2_f64, &[131, 125, 121, 123, 128, 142, 152, 140, 119, 111, 135, 130, 126, 129, 132, 141, 144, 131, 113, 105, 130, 126, 125, 129, 130, 133, 132, 122, 111, 106, 105, 105, 110, 113, 112, 117, 124, 128, 127, 127, 83, 88, 97, 99, 95, 101, 113, 126, 138, 143, 91, 95, 99, 96, 93, 97, 103, 112, 127, 136, 123, 124, 121, 110, 106, 107, 102, 100, 110, 119, 142, 142, 136, 122, 117, 115, 103, 95, 103, 111]),
        ];
        for (k, sigma, want) in cases {
            let g = gaussian_blur(&m, Size::new(*k, *k), *sigma, *sigma, BorderType::Reflect101)
                .unwrap();
            for y in 0..8 {
                for x in 0..10 {
                    assert_eq!(
                        g.at_f64(y, x, 0).unwrap() as u8,
                        want[(y * 10 + x) as usize],
                        "k={k} sigma={sigma} at ({y},{x})"
                    );
                }
            }
        }
    }


    /// The `CV_32F` blur is bit-exact against cv2 4.13.0.
    ///
    /// Pinned as raw `f32` bits, and kept separate from the `u8` test because
    /// the two take different paths — and because `u8` cannot see the bug this
    /// covers. Rounding an `f32` result back to an integer hides a 3e-5
    /// difference, so the column filter's missing pair-symmetry stayed
    /// invisible until an `f32` image was compared directly. See
    /// [`gaussian_blur_f32`].
    #[test]
    fn gaussian_blur_f32_is_bit_exact_against_cv2() {
        const SRC: [u32; 48] = [0x42d6fd39, 0x435de9dd, 0x437510fd, 0x429246da, 0x41e94ae3, 0x4319cd82, 0x432a37b7, 0x4346456d, 0x4323dea7, 0x43369959, 0x436964cc, 0x43696c04, 0x436c4795, 0x435b667f, 0x4337dbce, 0x436a268d, 0x40552c3e, 0x40d8f4b2, 0x434b662f, 0x42deff19, 0x433ec966, 0x42f75253, 0x4365f1ab, 0x4184ea20, 0x432bc6aa, 0x3fb7a226, 0x4217091f, 0x4353cef7, 0x4295304a, 0x437abdf8, 0x42b1c101, 0x43481354, 0x4374d665, 0x42a0f52f, 0x437972d1, 0x4333dc2e, 0x43770199, 0x42989514, 0x4302cdcf, 0x433ce40f, 0x4376489d, 0x428eabcb, 0x42f5534e, 0x43478f88, 0x4355e63a, 0x437bdfb6, 0x42595ee7, 0x437b7da7];
        const WANT: [u32; 48] = [0x4316a776, 0x431b26e8, 0x4323e9ea, 0x4329e5d0, 0x432c7023, 0x432e426a, 0x432fef56, 0x43309bd8, 0x430ee881, 0x4313fa04, 0x431e7969, 0x4326ba4b, 0x432aae92, 0x432c6a6a, 0x432d3e2f, 0x432d7741, 0x4300cf28, 0x4306bc16, 0x43142b25, 0x4320895c, 0x4326e852, 0x43280018, 0x4326d1c0, 0x43260e61, 0x42f72e78, 0x43017b59, 0x430fbde2, 0x431deb70, 0x4324d2fd, 0x43241d7e, 0x43209394, 0x431ed766, 0x4300f1ab, 0x430618e9, 0x43132a00, 0x43208a71, 0x43260986, 0x4322e80f, 0x431d49a2, 0x431ad501, 0x430529aa, 0x4309e631, 0x431625ad, 0x4322c20c, 0x432750e4, 0x4323031d, 0x431c82c6, 0x4319c57d];
        let src: Vec<f32> = SRC.iter().map(|b| f32::from_bits(*b)).collect();
        let m = Mat::from_vec(6, 8, 1, &src).unwrap();
        let g = gaussian_blur(&m, Size::new(0, 0), 1.6, 1.6, BorderType::Reflect101).unwrap();
        for y in 0..6 {
            for x in 0..8 {
                let got = g.at_f64(y, x, 0).unwrap() as f32;
                let want = f32::from_bits(WANT[(y * 8 + x) as usize]);
                assert_eq!(got.to_bits(), want.to_bits(), "({y},{x}): {got} vs cv2 {want}");
            }
        }
    }
}
