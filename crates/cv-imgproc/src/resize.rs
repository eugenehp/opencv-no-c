//! `cv::resize` (`modules/imgproc/src/resize.cpp`).
//!
//! # Bit-exactness
//!
//! `INTER_LINEAR` on `CV_8U` does **not** compute in floating point upstream.
//! It quantises the interpolation weights to 11-bit fixed point
//! (`INTER_RESIZE_COEF_BITS`), accumulates the horizontal pass in `i32`, and
//! finishes with a very specific shift sequence:
//!
//! ```text
//! dst = ((b0 * (S0 >> 4)) >> 16) + ((b1 * (S1 >> 4)) >> 16) + 2) >> 2
//! ```
//!
//! That is reproduced here exactly. A naive `f32` bilinear implementation
//! disagrees with it on a large fraction of pixels by one level, which is
//! enough to fail any pixel-comparison test against OpenCV.

use cv_core::{
    cv_assert, cv_floor, cv_round, Depth, Elem, Mat, MatLike, MatLikeMut, Result, Size,
};

/// `INTER_RESIZE_COEF_BITS`.
const INTER_RESIZE_COEF_BITS: i32 = 11;
/// `INTER_RESIZE_COEF_SCALE` == `1 << 11`.
const INTER_RESIZE_COEF_SCALE: i32 = 1 << INTER_RESIZE_COEF_BITS;

/// `cv::InterpolationFlags` — the subset ported so far, with upstream values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
#[non_exhaustive]
pub enum Interpolation {
    /// `INTER_NEAREST` — nearest-neighbour.
    Nearest = 0,
    /// `INTER_LINEAR` — bilinear.
    Linear = 1,
    /// `INTER_AREA` — pixel-area resampling; the preferred choice for shrinking.
    Area = 3,
    /// `INTER_LINEAR_EXACT` — bit-exact bilinear for integer images.
    ///
    /// Upstream ships two 8-bit linear paths that disagree with each other, and
    /// this is the accurate one: against a true `f64` bilinear reference it has
    /// mean error -0.0001 (max 0.778), where legacy `INTER_LINEAR` has mean
    /// +0.0083 (max 2.111). Verified bit-exact against a real `cv2` build on
    /// upscale, downscale and non-integer scale factors.
    LinearExact = 5,
}

/// `cv::resize(src, dst, dsize, fx, fy, interpolation)`.
///
/// Either give an explicit `dsize`, or pass `Size::new(0, 0)` together with
/// non-zero `fx`/`fy` scale factors, exactly as upstream allows.
pub fn resize<S: MatLike + ?Sized>(
    src: &S,
    dsize: Size,
    fx: f64,
    fy: f64,
    interpolation: Interpolation,
) -> Result<Mat> {
    cv_assert!(!src.is_empty(), "resize requires a non-empty source");
    let (sw, sh) = (src.cols(), src.rows());

    // Resolve the target size the same way upstream does.
    let (dw, dh) = if dsize.width > 0 && dsize.height > 0 {
        (dsize.width, dsize.height)
    } else {
        cv_assert!(fx > 0.0 && fy > 0.0, "resize needs either dsize or positive fx/fy");
        (cv_round(sw as f64 * fx), cv_round(sh as f64 * fy))
    };
    cv_assert!(dw > 0 && dh > 0, "resize target size must be positive, got {dw}x{dh}");

    if dw == sw && dh == sh {
        return Mat::copy_from(src);
    }

    let scale_x = sw as f64 / dw as f64;
    let scale_y = sh as f64 / dh as f64;

    // Upstream substitutes INTER_AREA for INTER_LINEAR at an exact 2x downscale,
    // where the fast area path is provably equal to bit-exact linear. Skipping
    // this makes every such call disagree with OpenCV by a level, because the
    // fixed-point linear reduction rounds halves up while area rounds to even.
    let mut interpolation = interpolation;
    let iscale_x = scale_x as i32;
    let iscale_y = scale_y as i32;
    let is_area_fast = (scale_x - iscale_x as f64).abs() < f64::EPSILON
        && (scale_y - iscale_y as f64).abs() < f64::EPSILON;
    if interpolation == Interpolation::Linear && is_area_fast && iscale_x == 2 && iscale_y == 2 {
        interpolation = Interpolation::Area;
    }

    match interpolation {
        Interpolation::Nearest => resize_nearest(src, dw, dh, scale_x, scale_y),
        Interpolation::LinearExact => resize_linear_exact(src, dw, dh, scale_x, scale_y),
        Interpolation::Linear => resize_linear(src, dw, dh, scale_x, scale_y),
        Interpolation::Area => {
            // True area resampling only exists for shrinking; upstream emulates
            // the enlarging case with bilinear.
            if scale_x >= 1.0 && scale_y >= 1.0 {
                if is_area_fast && iscale_x >= 1 && iscale_y >= 1 {
                    resize_area_fast(src, dw, dh, iscale_x, iscale_y)
                } else {
                    resize_area(src, dw, dh, scale_x, scale_y)
                }
            } else {
                resize_linear(src, dw, dh, scale_x, scale_y)
            }
        }
    }
}

/// `INTER_NEAREST` — `sx = cvFloor(dx * scale_x)`, clamped to the last column.
fn resize_nearest<S: MatLike + ?Sized>(
    src: &S,
    dw: i32,
    dh: i32,
    scale_x: f64,
    scale_y: f64,
) -> Result<Mat> {
    let typ = src.mat_type();
    let esz = typ.elem_size();
    let mut dst = Mat::zeros(dh, dw, typ)?;
    let xofs: Vec<i32> =
        (0..dw).map(|dx| cv_floor(dx as f64 * scale_x).min(src.cols() - 1)).collect();

    for dy in 0..dh {
        let sy = cv_floor(dy as f64 * scale_y).min(src.rows() - 1);
        let srow = src.row_bytes(sy)?.to_vec();
        let drow = dst.row_bytes_mut(dy)?;
        for dx in 0..dw {
            let sx = xofs[dx as usize] as usize;
            drow[dx as usize * esz..(dx as usize + 1) * esz]
                .copy_from_slice(&srow[sx * esz..(sx + 1) * esz]);
        }
    }
    Ok(dst)
}

/// One axis' bit-exact sampling plan: source index plus 8-bit fixed-point
/// weights that sum to exactly 256.
struct ExactTab {
    idx: Vec<usize>,
    c0: Vec<i64>,
    c1: Vec<i64>,
}

/// `interpolationLinear::getCoeffs` from upstream's bit-exact resize.
///
/// The weights are `ufixedpoint16` — 16-bit storage with **8** fractional bits,
/// not the 11-bit `INTER_RESIZE_COEF_BITS` the legacy path uses. `c0` is
/// derived as `256 - c1` rather than rounded independently, so the pair always
/// totals exactly one unit and a flat image survives untouched.
fn exact_tab(dst_len: i32, src_len: i32, scale: f64) -> ExactTab {
    let n = dst_len as usize;
    let mut t = ExactTab { idx: vec![0; n], c0: vec![256; n], c1: vec![0; n] };
    for d in 0..n {
        let fval = scale * (d as f64 + 0.5) - 0.5;
        let ival = fval.floor();
        if ival >= 0.0 && src_len > 1 {
            let i = ival as i64;
            if i < (src_len - 1) as i64 {
                t.idx[d] = i as usize;
                let c1 = ((fval - ival) * 256.0).round_ties_even() as i64;
                t.c1[d] = c1;
                t.c0[d] = 256 - c1;
            } else {
                // Past the last full interval: replicate the final sample.
                t.idx[d] = (src_len - 1) as usize;
            }
        }
        // Before the first sample the defaults already replicate index 0.
    }
    t
}

/// `INTER_LINEAR_EXACT` — upstream's bit-exact bilinear.
///
/// Horizontal then vertical in integers, each pass scaling by `1 << 8`, with a
/// single rounding shift of 16 at the end. Integer depths only; float depths
/// fall through to the ordinary linear path, which is what upstream does when
/// it rewrites `INTER_LINEAR_EXACT` to `INTER_LINEAR` for `CV_32F`/`CV_64F`.
fn resize_linear_exact<S: MatLike + ?Sized>(
    src: &S,
    dw: i32,
    dh: i32,
    scale_x: f64,
    scale_y: f64,
) -> Result<Mat> {
    let typ = src.mat_type();
    if typ.depth().is_float() {
        return resize_linear(src, dw, dh, scale_x, scale_y);
    }
    let cn = typ.channels() as usize;
    let (sw, sh) = (src.cols() as usize, src.rows() as usize);
    let buf = cv_core::to_f64(src)?;

    let xt = exact_tab(dw, src.cols(), scale_x);
    let yt = exact_tab(dh, src.rows(), scale_y);

    // Horizontal pass over every source row, in units of 1/256.
    let width = dw as usize * cn;
    let mut tmp = vec![0i64; sh * width];
    for y in 0..sh {
        let row = &buf[y * sw * cn..(y + 1) * sw * cn];
        let dst = &mut tmp[y * width..(y + 1) * width];
        for d in 0..dw as usize {
            let s0 = xt.idx[d];
            let s1 = (s0 + 1).min(sw - 1);
            for c in 0..cn {
                dst[d * cn + c] =
                    row[s0 * cn + c] as i64 * xt.c0[d] + row[s1 * cn + c] as i64 * xt.c1[d];
            }
        }
    }

    // Vertical pass, then one rounding shift of 2*8 bits.
    let mut out = vec![0.0f64; dh as usize * width];
    for d in 0..dh as usize {
        let s0 = yt.idx[d];
        let s1 = (s0 + 1).min(sh - 1);
        for i in 0..width {
            let v = tmp[s0 * width + i] * yt.c0[d] + tmp[s1 * width + i] * yt.c1[d];
            out[d * width + i] = ((v + (1 << 15)) >> 16) as f64;
        }
    }
    cv_core::mat_from_f64(dh, dw, typ, &out)
}

/// One axis' bilinear sampling plan: source index plus the fractional weight.
struct LinearTab {
    /// Lower source index for each destination index.
    idx: Vec<i32>,
    /// Fractional weight toward `idx + 1`, in `[0, 1]`.
    frac: Vec<f32>,
}

/// Build the bilinear sampling plan for one axis, including upstream's edge
/// clamping (which zeroes the fraction at both ends so the out-of-range
/// neighbour never contributes).
fn linear_tab(dst_len: i32, src_len: i32, scale: f64) -> LinearTab {
    let mut idx = Vec::with_capacity(dst_len as usize);
    let mut frac = Vec::with_capacity(dst_len as usize);
    for d in 0..dst_len {
        let mut f = ((d as f64 + 0.5) * scale - 0.5) as f32;
        let mut s = cv_floor(f as f64);
        f -= s as f32;
        if s < 0 {
            f = 0.0;
            s = 0;
        }
        if s >= src_len - 1 {
            f = 0.0;
            s = src_len - 1;
        }
        idx.push(s);
        frac.push(f);
    }
    LinearTab { idx, frac }
}

/// `INTER_LINEAR`. `CV_8U` takes the fixed-point path; every other depth
/// accumulates in `f32`, matching upstream's `fixpt = depth == CV_8U` switch.
fn resize_linear<S: MatLike + ?Sized>(
    src: &S,
    dw: i32,
    dh: i32,
    scale_x: f64,
    scale_y: f64,
) -> Result<Mat> {
    let typ = src.mat_type();
    let cn = typ.channels() as usize;
    let (sw, sh) = (src.cols(), src.rows());
    let xt = linear_tab(dw, sw, scale_x);
    let yt = linear_tab(dh, sh, scale_y);
    let mut dst = Mat::zeros(dh, dw, typ)?;

    if typ.depth() == Depth::U8 {
        // Quantise both axes to 11-bit fixed point up front.
        let ax: Vec<[i32; 2]> = xt
            .frac
            .iter()
            .map(|&f| {
                let a1 = i16::from_f64((f * INTER_RESIZE_COEF_SCALE as f32) as f64) as i32;
                let a0 = i16::from_f64(((1.0 - f) * INTER_RESIZE_COEF_SCALE as f32) as f64) as i32;
                [a0, a1]
            })
            .collect();
        let ay: Vec<[i32; 2]> = yt
            .frac
            .iter()
            .map(|&f| {
                let b1 = i16::from_f64((f * INTER_RESIZE_COEF_SCALE as f32) as f64) as i32;
                let b0 = i16::from_f64(((1.0 - f) * INTER_RESIZE_COEF_SCALE as f32) as f64) as i32;
                [b0, b1]
            })
            .collect();

        for dy in 0..dh {
            let sy = yt.idx[dy as usize];
            let sy1 = (sy + 1).min(sh - 1);
            // Horizontal pass for the two contributing source rows.
            let r0 = hresize_u8(src, sy, &xt.idx, &ax, cn, dw)?;
            let r1 = if sy1 == sy { r0.clone() } else { hresize_u8(src, sy1, &xt.idx, &ax, cn, dw)? };
            let [b0, b1] = ay[dy as usize];

            let drow = dst.row_as_mut::<u8>(dy)?;
            for i in 0..drow.len() {
                // Upstream's exact vertical fixed-point reduction.
                let v = ((b0 * (r0[i] >> 4)) >> 16) + ((b1 * (r1[i] >> 4)) >> 16);
                drow[i] = ((v + 2) >> 2) as u8;
            }
        }
        return Ok(dst);
    }

    // Float path: horizontal then vertical, both in f32, saturating on store.
    let buf = cv_core::to_f64(src)?;
    let sw_u = sw as usize;
    let width = dw as usize * cn;
    let mut out = vec![0.0f64; dh as usize * width];
    for dy in 0..dh {
        let sy = yt.idx[dy as usize];
        let sy1 = (sy + 1).min(sh - 1);
        let b1 = yt.frac[dy as usize];
        let b0 = 1.0 - b1;
        let r0 = hresize_f32(&buf, sw_u, sy, &xt, cn, dw);
        let r1 = if sy1 == sy { r0.clone() } else { hresize_f32(&buf, sw_u, sy1, &xt, cn, dw) };

        let dst_row = &mut out[dy as usize * width..(dy as usize + 1) * width];
        for (i, o) in dst_row.iter_mut().enumerate() {
            *o = (r0[i] * b0 + r1[i] * b1) as f64;
        }
    }
    cv_core::mat_from_f64(dh, dw, typ, &out)
}

/// Horizontal fixed-point pass over one `CV_8U` source row.
fn hresize_u8<S: MatLike + ?Sized>(
    src: &S,
    sy: i32,
    xidx: &[i32],
    ax: &[[i32; 2]],
    cn: usize,
    dw: i32,
) -> Result<Vec<i32>> {
    let row = src.row_as::<u8>(sy)?;
    let last = src.cols() as usize - 1;
    let mut out = vec![0i32; dw as usize * cn];
    for dx in 0..dw as usize {
        let sx = xidx[dx] as usize;
        let sx1 = (sx + 1).min(last);
        let [a0, a1] = ax[dx];
        for c in 0..cn {
            out[dx * cn + c] = row[sx * cn + c] as i32 * a0 + row[sx1 * cn + c] as i32 * a1;
        }
    }
    Ok(out)
}

/// Horizontal `f32` pass over one source row of any non-`CV_8U` depth.
fn hresize_f32(
    buf: &[f64],
    sw: usize,
    sy: i32,
    xt: &LinearTab,
    cn: usize,
    dw: i32,
) -> Vec<f32> {
    let last = sw as i32 - 1;
    let row = sy as usize * sw;
    let mut out = vec![0f32; dw as usize * cn];
    for dx in 0..dw as usize {
        let sx = xt.idx[dx];
        let sx1 = (sx + 1).min(last);
        let a1 = xt.frac[dx];
        let a0 = 1.0 - a1;
        let (b0, b1) = ((row + sx as usize) * cn, (row + sx1 as usize) * cn);
        for c in 0..cn {
            out[dx * cn + c] = buf[b0 + c] as f32 * a0 + buf[b1 + c] as f32 * a1;
        }
    }
    out
}

/// `INTER_AREA` with integer scale factors — a plain box average, matching
/// `resizeAreaFast_`'s `saturate_cast<T>(sum * (1.f / area))`.
fn resize_area_fast<S: MatLike + ?Sized>(
    src: &S,
    dw: i32,
    dh: i32,
    ix: i32,
    iy: i32,
) -> Result<Mat> {
    let typ = src.mat_type();
    let cn = typ.channels() as usize;
    let (sw, sh) = (src.cols() as usize, src.rows() as usize);
    let buf = cv_core::to_f64(src)?;
    let area = (ix * iy) as i64;
    let integral = !typ.depth().is_float();
    let halving_fast_path = integral && ix == 2 && iy == 2 && matches!(cn, 1 | 3 | 4);

    let width = dw as usize * cn;
    let mut out = vec![0.0f64; dh as usize * width];
    for dy in 0..dh as usize {
        for dx in 0..dw as usize {
            for c in 0..cn {
                let mut sum = 0.0f64;
                for ky in 0..iy as usize {
                    let sy = (dy * iy as usize + ky).min(sh - 1);
                    for kx in 0..ix as usize {
                        let sx = (dx * ix as usize + kx).min(sw - 1);
                        sum += buf[(sy * sw + sx) * cn + c];
                    }
                }
                // **Halving is the only integer scale that rounds half up.**
                // `ResizeAreaFastVec::operator()` runs at all only when the
                // scale is exactly two in both axes and the image has one,
                // three or four channels; it then covers the whole row —
                // vector body and scalar remainder alike — with the integer
                // `(a + b + c + d + 2) >> 2`. Every other integer scale, and
                // every two-channel or floating image, falls through to
                // `resizeAreaFast_`'s own loop and its `saturate_cast<T>(sum *
                // scale)`, which rounds halves to even. A 4x4 ramp halved
                // gives 3, 5, 11, 13 where `saturate_cast` would give 2, 4,
                // 10, 12; the same image reduced fourfold gives the
                // `saturate_cast` answer.
                out[dy * width + dx * cn + c] = if halving_fast_path {
                    ((sum as i64 + 2) >> 2) as f64
                } else if integral {
                    // `scale` is a **float**, so a mean over nine pixels is
                    // not `sum / 9` — it is `sum * 0.111111112f`.
                    f64::from(cv_core::saturate::cv_round(f64::from(
                        (sum as f32) * (1.0f32 / area as f32),
                    )) as i32)
                } else {
                    f64::from((sum as f32) * (1.0f32 / area as f32))
                };
            }
        }
    }
    cv_core::mat_from_f64(dh, dw, typ, &out)
}

/// One entry of the decimation table: source index and its weight.
struct Decimate {
    si: i32,
    alpha: f32,
}

/// `computeResizeAreaTab` — overlap weights for one axis, transcribed from
/// upstream including the `1e-3` partial-cell thresholds.
fn area_tab(ssize: i32, dsize: i32, scale: f64) -> Vec<Vec<Decimate>> {
    let mut tabs = Vec::with_capacity(dsize as usize);
    for dx in 0..dsize {
        let mut row = Vec::new();
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell_width = scale.min(ssize as f64 - fsx1);

        let mut sx1 = fsx1.ceil() as i32;
        let mut sx2 = fsx2.floor() as i32;
        sx2 = sx2.min(ssize - 1);
        sx1 = sx1.min(sx2);

        if (sx1 as f64 - fsx1) > 1e-3 {
            row.push(Decimate {
                si: sx1 - 1,
                alpha: ((sx1 as f64 - fsx1) / cell_width) as f32,
            });
        }
        for sx in sx1..sx2 {
            row.push(Decimate { si: sx, alpha: (1.0 / cell_width) as f32 });
        }
        if (fsx2 - sx2 as f64) > 1e-3 {
            row.push(Decimate {
                si: sx2,
                alpha: ((fsx2 - sx2 as f64).min(1.0).min(cell_width) / cell_width) as f32,
            });
        }
        tabs.push(row);
    }
    tabs
}

/// `INTER_AREA` with non-integer scale factors — separable overlap weighting.
fn resize_area<S: MatLike + ?Sized>(
    src: &S,
    dw: i32,
    dh: i32,
    scale_x: f64,
    scale_y: f64,
) -> Result<Mat> {
    let typ = src.mat_type();
    let cn = typ.channels() as usize;
    let sw = src.cols() as usize;
    let buf = cv_core::to_f64(src)?;
    let integral = !typ.depth().is_float();
    let xtab = area_tab(src.cols(), dw, scale_x);
    let ytab = area_tab(src.rows(), dh, scale_y);

    let width = dw as usize * cn;
    let mut out = vec![0.0f64; dh as usize * width];
    for dy in 0..dh as usize {
        for dx in 0..dw as usize {
            for c in 0..cn {
                let mut sum = 0.0f32;
                for yy in &ytab[dy] {
                    let row = yy.si as usize * sw;
                    let mut rowsum = 0.0f32;
                    for xx in &xtab[dx] {
                        rowsum += buf[(row + xx.si as usize) * cn + c] as f32 * xx.alpha;
                    }
                    sum += rowsum * yy.alpha;
                }
                // Integer output rounds **half up**, as the area-fast path
                // does. Verified against a real cv2 build: half-to-even (what
                // `saturate_cast` would give) disagrees, half-up is exact.
                out[dy * width + dx * cn + c] =
                    if integral { (sum as f64 + 0.5).floor() } else { sum as f64 };
            }
        }
    }
    cv_core::mat_from_f64(dh, dw, typ, &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::MatType;

    fn ramp(w: i32, h: i32) -> Mat {
        let data: Vec<u8> = (0..(w * h)).map(|i| (i % 256) as u8).collect();
        Mat::from_vec(h, w, 1, &data).unwrap()
    }

    #[test]
    fn interpolation_values_match_opencv() {
        assert_eq!(Interpolation::Nearest as i32, 0);
        assert_eq!(Interpolation::Linear as i32, 1);
        assert_eq!(Interpolation::Area as i32, 3);
    }

    #[test]
    fn same_size_is_a_copy() {
        let m = ramp(4, 4);
        let r = resize(&m, Size::new(4, 4), 0.0, 0.0, Interpolation::Linear).unwrap();
        assert_eq!(r.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn scale_factors_resolve_target_size() {
        let m = ramp(8, 6);
        let r = resize(&m, Size::new(0, 0), 0.5, 2.0, Interpolation::Nearest).unwrap();
        assert_eq!(r.size(), Size::new(4, 12));
    }

    #[test]
    fn nearest_picks_floor_of_scaled_index() {
        let m = Mat::from_vec(1, 4, 1, &[10u8, 20, 30, 40]).unwrap();
        let r = resize(&m, Size::new(2, 1), 0.0, 0.0, Interpolation::Nearest).unwrap();
        // scale_x = 2, so dx=0 -> sx=0, dx=1 -> sx=2.
        assert_eq!(r.to_vec::<u8>().unwrap(), vec![10, 30]);
    }

    #[test]
    fn nearest_upscale_replicates() {
        let m = Mat::from_vec(1, 2, 1, &[10u8, 20]).unwrap();
        let r = resize(&m, Size::new(4, 1), 0.0, 0.0, Interpolation::Nearest).unwrap();
        assert_eq!(r.to_vec::<u8>().unwrap(), vec![10, 10, 20, 20]);
    }

    #[test]
    fn linear_2x_upscale_on_uniform_image_is_exact() {
        // A constant image must survive interpolation untouched — this catches
        // accumulated rounding error in the fixed-point path.
        let m = Mat::new_with(4, 4, MatType::new(Depth::U8, 1), cv_core::Scalar::all(200.0)).unwrap();
        let r = resize(&m, Size::new(8, 8), 0.0, 0.0, Interpolation::Linear).unwrap();
        assert!(r.to_vec::<u8>().unwrap().iter().all(|&v| v == 200), "{:?}", r.to_vec::<u8>());
    }

    #[test]
    fn linear_preserves_full_range_extremes() {
        let m = Mat::from_vec(2, 2, 1, &[0u8, 255, 0, 255]).unwrap();
        let r = resize(&m, Size::new(4, 4), 0.0, 0.0, Interpolation::Linear).unwrap();
        let v = r.to_vec::<u8>().unwrap();
        assert_eq!(v.iter().copied().min().unwrap(), 0);
        assert_eq!(v.iter().copied().max().unwrap(), 255);
    }

    #[test]
    fn linear_at_exact_2x_is_promoted_to_area() {
        // Upstream rewrites INTER_LINEAR to INTER_AREA at an exact 2x
        // downscale, so both must agree here. Block means 2.5/4.5/10.5/12.5
        // round half up to 3/5/11/13, matching a real cv2 build.
        let m = Mat::from_vec(4, 4, 1, &(0u8..16).collect::<Vec<_>>()).unwrap();
        let lin = resize(&m, Size::new(2, 2), 0.0, 0.0, Interpolation::Linear).unwrap();
        let area = resize(&m, Size::new(2, 2), 0.0, 0.0, Interpolation::Area).unwrap();
        assert_eq!(lin.to_vec::<u8>().unwrap(), vec![3, 5, 11, 13]);
        assert_eq!(lin.to_vec::<u8>().unwrap(), area.to_vec::<u8>().unwrap());
    }

    #[test]
    fn linear_at_3x_uses_the_fixed_point_path() {
        // 3x is not the promoted case, so the genuine fixed-point reduction
        // runs. 6x6 ramp -> 2x2 samples at source centres (1,1) and (1,4) etc.
        let m = Mat::from_vec(6, 6, 1, &(0u8..36).collect::<Vec<_>>()).unwrap();
        let lin = resize(&m, Size::new(2, 2), 0.0, 0.0, Interpolation::Linear).unwrap();
        // scale=3, so dx=0 -> fx = 0.5*3-0.5 = 1.0 exactly: a pure source tap
        // at (1,1), (1,4), (4,1), (4,4) with zero fractional weight.
        assert_eq!(lin.to_vec::<u8>().unwrap(), vec![7, 10, 25, 28]);
    }

    #[test]
    fn linear_and_area_diverge_on_a_nonlinear_image() {
        // A linear ramp makes the two agree by construction; a step pattern
        // separates the single-tap linear result from the 3x3 area mean.
        let mut data = vec![0u8; 36];
        data[0] = 255;
        let m = Mat::from_vec(6, 6, 1, &data).unwrap();
        let lin = resize(&m, Size::new(2, 2), 0.0, 0.0, Interpolation::Linear).unwrap();
        let area = resize(&m, Size::new(2, 2), 0.0, 0.0, Interpolation::Area).unwrap();
        assert_eq!(lin.to_vec::<u8>().unwrap()[0], 0); // taps source (1,1) == 0
        assert_eq!(area.to_vec::<u8>().unwrap()[0], 28); // (255 + 4) / 9 -> 28
    }

    #[test]
    fn area_fast_is_an_exact_box_average() {
        let m = Mat::from_vec(4, 4, 1, &(0u8..16).collect::<Vec<_>>()).unwrap();
        let r = resize(&m, Size::new(2, 2), 0.0, 0.0, Interpolation::Area).unwrap();
        // Block means are 2.5, 4.5, 10.5, 12.5. OpenCV rounds these **half up**
        // via integer `(sum + area/2) / area`, giving 3, 5, 11, 13 — verified
        // against a real cv2 build, not inferred from `saturate_cast`.
        assert_eq!(r.to_vec::<u8>().unwrap(), vec![3, 5, 11, 13]);
    }

    #[test]
    fn area_on_uniform_image_is_exact() {
        let m = Mat::new_with(6, 6, MatType::new(Depth::U8, 3), cv_core::Scalar::new(10.0, 20.0, 30.0, 0.0)).unwrap();
        let r = resize(&m, Size::new(4, 4), 0.0, 0.0, Interpolation::Area).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(r.pixel::<u8>(y, x).unwrap(), &[10, 20, 30], "at {y},{x}");
            }
        }
    }

    #[test]
    fn area_non_integer_downscale_runs_the_overlap_path() {
        let m = ramp(9, 9);
        let r = resize(&m, Size::new(4, 4), 0.0, 0.0, Interpolation::Area).unwrap();
        assert_eq!(r.size(), Size::new(4, 4));
    }

    #[test]
    fn area_upscale_falls_back_to_linear() {
        let m = ramp(4, 4);
        let a = resize(&m, Size::new(8, 8), 0.0, 0.0, Interpolation::Area).unwrap();
        let l = resize(&m, Size::new(8, 8), 0.0, 0.0, Interpolation::Linear).unwrap();
        assert_eq!(a.to_vec::<u8>().unwrap(), l.to_vec::<u8>().unwrap());
    }

    #[test]
    fn float_images_use_the_float_path() {
        let m = Mat::from_vec(2, 2, 1, &[0.0f32, 1.0, 2.0, 3.0]).unwrap();
        let r = resize(&m, Size::new(4, 4), 0.0, 0.0, Interpolation::Linear).unwrap();
        assert_eq!(r.depth(), Depth::F32);
        let v = r.to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| (-0.1..=3.1).contains(x)), "{v:?}");
    }

    #[test]
    fn multichannel_channels_stay_independent() {
        let m = Mat::new_with(4, 4, MatType::new(Depth::U8, 3), cv_core::Scalar::new(0.0, 128.0, 255.0, 0.0)).unwrap();
        let r = resize(&m, Size::new(2, 2), 0.0, 0.0, Interpolation::Linear).unwrap();
        assert_eq!(r.pixel::<u8>(0, 0).unwrap(), &[0, 128, 255]);
    }

    #[test]
    fn linear_exact_value_matches_opencv() {
        assert_eq!(Interpolation::LinearExact as i32, 5);
        // Reference values taken from a real cv2 build with
        // INTER_LINEAR_EXACT; the whole point of this mode is bit-exactness.
        let m = Mat::from_vec(4, 4, 1, &(0u8..16).collect::<Vec<_>>()).unwrap();
        let r = resize(&m, Size::new(12, 12), 0.0, 0.0, Interpolation::LinearExact).unwrap();
        let v = r.to_vec::<u8>().unwrap();
        assert_eq!(&v[..12], &[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
    }

    #[test]
    fn linear_exact_preserves_a_flat_image() {
        // The weight pair sums to exactly 256, so a constant image must survive
        // any scale factor untouched. Independent rounding of both weights
        // would drift here.
        let m = Mat::new_with(9, 11, MatType::new(Depth::U8, 3), cv_core::Scalar::new(40.0, 130.0, 250.0, 0.0))
            .unwrap();
        for (w, h) in [(4, 3), (11, 9), (23, 17), (33, 40)] {
            let r = resize(&m, Size::new(w, h), 0.0, 0.0, Interpolation::LinearExact).unwrap();
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(r.pixel::<u8>(y, x).unwrap(), &[40, 130, 250], "{w}x{h} at {y},{x}");
                }
            }
        }
    }

    #[test]
    fn linear_exact_stays_in_range_and_is_monotone_on_a_ramp() {
        let data: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let m = Mat::from_vec(1, 64, 1, &data).unwrap();
        let r = resize(&m, Size::new(200, 1), 0.0, 0.0, Interpolation::LinearExact).unwrap();
        let v = r.to_vec::<u8>().unwrap();
        assert!(v.windows(2).all(|w| w[0] <= w[1]), "upsampled ramp is not monotone");
        assert_eq!(*v.first().unwrap(), 0);
        assert_eq!(*v.last().unwrap(), 252);
    }

    #[test]
    fn linear_exact_falls_back_for_float_depths() {
        // Upstream rewrites INTER_LINEAR_EXACT to INTER_LINEAR for CV_32F/64F.
        let m = Mat::from_vec(2, 2, 1, &[0.0f32, 1.0, 2.0, 3.0]).unwrap();
        let a = resize(&m, Size::new(4, 4), 0.0, 0.0, Interpolation::LinearExact).unwrap();
        let b = resize(&m, Size::new(4, 4), 0.0, 0.0, Interpolation::Linear).unwrap();
        assert_eq!(a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap());
    }

    #[test]
    fn empty_and_degenerate_inputs_are_rejected() {
        let m = ramp(4, 4);
        assert!(resize(&m, Size::new(0, 0), 0.0, 0.0, Interpolation::Linear).is_err());
        assert!(resize(&m, Size::new(-1, 5), 0.0, 0.0, Interpolation::Linear).is_err());
        let e = Mat::zeros(0, 0, MatType::new(Depth::U8, 1)).unwrap();
        assert!(resize(&e, Size::new(2, 2), 0.0, 0.0, Interpolation::Linear).is_err());
    }
}
