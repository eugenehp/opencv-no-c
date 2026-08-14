//! Assorted array operations
//! (`modules/core/src/convert_scale.dispatch.cpp`, `count_non_zero.dispatch.cpp`,
//! `matrix_transform.cpp`, `matmul.dispatch.cpp`, `check.cpp`).

use crate::error::Result;
use crate::geometry::Point;
use crate::mat::{Mat, MatLike, MatLikeMut};
use crate::saturate::{cv_round, Elem};
use crate::types::{Depth, MatType, CV_64FC1, CV_8UC1};
use crate::cv_assert;

/// `cv::convertScaleAbs(src, dst, alpha, beta)` — `|src*alpha + beta|` as `CV_8U`.
///
/// The absolute value is taken *before* saturation, so a large negative input
/// clamps to 255 rather than 0. That ordering is upstream's and it is the whole
/// reason the function exists.
pub fn convert_scale_abs<S: MatLike + ?Sized>(src: &S, alpha: f64, beta: f64) -> Result<Mat> {
    let cn = src.channels();
    let mut dst = Mat::zeros(src.rows(), src.cols(), MatType::new(Depth::U8, cn))?;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            for c in 0..cn {
                // `DEF_CVT_SCALE_ABS_FUNC(8u, cvtabs_32f, uchar, uchar,
                // float)`: the working type is `float`, so alpha, beta and the
                // product all round to `f32` before the absolute value. Doing
                // it in `f64` disagrees with cv2 on about 1.7% of pixels.
                let v = ((src.at_f64(y, x, c)? as f32).mul_add(alpha as f32, beta as f32)).abs();
                dst.pixel_mut::<u8>(y, x)?[c as usize] = u8::from_f64(f64::from(v));
            }
        }
    }
    Ok(dst)
}

/// `cv::findNonZero(src, idx)` — the coordinates of every non-zero element.
///
/// Returned in raster order, as upstream produces them.
pub fn find_non_zero<S: MatLike + ?Sized>(src: &S) -> Result<Vec<Point>> {
    cv_assert!(src.channels() == 1, "findNonZero requires a single-channel matrix");
    let mut out = Vec::new();
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            if src.at_f64(y, x, 0)? != 0.0 {
                out.push(Point::new(x, y));
            }
        }
    }
    Ok(out)
}

/// `cv::hasNonZero(src)` — short-circuits on the first non-zero element.
pub fn has_non_zero<S: MatLike + ?Sized>(src: &S) -> Result<bool> {
    cv_assert!(src.channels() == 1, "hasNonZero requires a single-channel matrix");
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            if src.at_f64(y, x, 0)? != 0.0 {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// `cv::completeSymm(m, lowerToUpper)` — mirror one triangle onto the other.
pub fn complete_symm<M: MatLike + MatLikeMut + ?Sized>(m: &mut M, lower_to_upper: bool) -> Result<()> {
    let n = m.rows();
    cv_assert!(n == m.cols(), "completeSymm requires a square matrix");
    cv_assert!(m.channels() == 1, "completeSymm requires a single-channel matrix");
    for i in 0..n {
        for j in i + 1..n {
            let (from, to) = if lower_to_upper { ((j, i), (i, j)) } else { ((i, j), (j, i)) };
            let v = m.at_f64(from.0, from.1, 0)?;
            crate::arithm::set_channel_generic(m, to.0, to.1, 0, v)?;
        }
    }
    Ok(())
}

/// `cv::patchNaNs(a, val)` — replace every NaN with `val`.
pub fn patch_nans<M: MatLike + MatLikeMut + ?Sized>(m: &mut M, val: f64) -> Result<()> {
    cv_assert!(
        matches!(m.depth(), Depth::F32 | Depth::F64),
        "patchNaNs requires a float matrix, got {}",
        m.depth().name()
    );
    let cn = m.channels();
    for y in 0..m.rows() {
        for x in 0..m.cols() {
            for c in 0..cn {
                if m.at_f64(y, x, c)?.is_nan() {
                    crate::arithm::set_channel_generic(m, y, x, c, val)?;
                }
            }
        }
    }
    Ok(())
}

/// Quarter-turn selector for [`rotate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotateCode {
    /// `cv::ROTATE_90_CLOCKWISE`
    Clockwise90 = 0,
    /// `cv::ROTATE_180`
    Rotate180 = 1,
    /// `cv::ROTATE_90_COUNTERCLOCKWISE`
    CounterClockwise90 = 2,
}

/// `cv::rotate(src, dst, rotateCode)` — an exact quarter turn, no resampling.
pub fn rotate<S: MatLike + ?Sized>(src: &S, code: RotateCode) -> Result<Mat> {
    let (h, w) = (src.rows(), src.cols());
    let (oh, ow) = match code {
        RotateCode::Rotate180 => (h, w),
        _ => (w, h),
    };
    let cn = src.channels();
    let mut dst = Mat::zeros(oh, ow, src.mat_type())?;
    for y in 0..h {
        for x in 0..w {
            let (ny, nx) = match code {
                RotateCode::Clockwise90 => (x, h - 1 - y),
                RotateCode::Rotate180 => (h - 1 - y, w - 1 - x),
                RotateCode::CounterClockwise90 => (w - 1 - x, y),
            };
            for c in 0..cn {
                let v = src.at_f64(y, x, c)?;
                crate::arithm::set_channel_generic(&mut dst, ny, nx, c, v)?;
            }
        }
    }
    Ok(dst)
}

/// Where an extreme value sits, from [`min_max_idx`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Extremes {
    /// Smallest value found.
    pub min_val: f64,
    /// Largest value found.
    pub max_val: f64,
    /// Location of `min_val`, or `None` if the matrix was fully masked out.
    pub min_loc: Option<Point>,
    /// Location of `max_val`.
    pub max_loc: Option<Point>,
}

/// `cv::minMaxIdx(src, minVal, maxVal, minIdx, maxIdx, mask)`.
///
/// Ties keep the first location in raster order, matching upstream.
pub fn min_max_idx<S: MatLike + ?Sized, M: MatLike + ?Sized>(
    src: &S,
    mask: Option<&M>,
) -> Result<Extremes> {
    cv_assert!(src.channels() == 1, "minMaxIdx requires a single-channel matrix");
    if let Some(m) = mask {
        cv_assert!(m.mat_type() == CV_8UC1, "mask must be CV_8UC1");
        cv_assert!(m.size() == src.size(), "mask size must match the source");
    }
    let mut e = Extremes {
        min_val: f64::INFINITY,
        max_val: f64::NEG_INFINITY,
        min_loc: None,
        max_loc: None,
    };
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            if let Some(m) = mask {
                if m.at_f64(y, x, 0)? == 0.0 {
                    continue;
                }
            }
            let v = src.at_f64(y, x, 0)?;
            if v < e.min_val {
                e.min_val = v;
                e.min_loc = Some(Point::new(x, y));
            }
            if v > e.max_val {
                e.max_val = v;
                e.max_loc = Some(Point::new(x, y));
            }
        }
    }
    if e.min_loc.is_none() {
        e.min_val = 0.0;
        e.max_val = 0.0;
    }
    Ok(e)
}

/// `cv::reduceArgMin(src, dst, axis)` / `cv::reduceArgMax`.
///
/// `axis == 0` reduces down each column and yields one row; `axis == 1` reduces
/// along each row and yields one column. Ties keep the lowest index.
pub fn reduce_arg<S: MatLike + ?Sized>(src: &S, axis: i32, take_max: bool) -> Result<Mat> {
    cv_assert!(src.channels() == 1, "reduceArg requires a single-channel matrix");
    cv_assert!(axis == 0 || axis == 1, "axis must be 0 or 1, got {axis}");
    let (h, w) = (src.rows(), src.cols());
    let (oh, ow) = if axis == 0 { (1, w) } else { (h, 1) };
    let mut dst = Mat::zeros(oh, ow, MatType::new(Depth::S32, 1))?;
    let outer = if axis == 0 { w } else { h };
    let inner = if axis == 0 { h } else { w };
    for o in 0..outer {
        let (mut best, mut bi) = (f64::NAN, 0i32);
        for i in 0..inner {
            let v = if axis == 0 { src.at_f64(i, o, 0)? } else { src.at_f64(o, i, 0)? };
            let better = if i == 0 {
                true
            } else if take_max {
                v > best
            } else {
                v < best
            };
            if better {
                best = v;
                bi = i;
            }
        }
        let (y, x) = if axis == 0 { (0, o) } else { (o, 0) };
        dst.pixel_mut::<i32>(y, x)?[0] = bi;
    }
    Ok(dst)
}

/// `cv::checkRange(a, quiet, pos, minVal, maxVal)`.
///
/// Returns `Ok(None)` when every element is finite and inside `[min, max)`, or
/// `Ok(Some(pos))` with the first offending location. The half-open interval is
/// upstream's.
pub fn check_range<S: MatLike + ?Sized>(src: &S, min_val: f64, max_val: f64) -> Result<Option<Point>> {
    let cn = src.channels();
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            for c in 0..cn {
                let v = src.at_f64(y, x, c)?;
                if !v.is_finite() || v < min_val || v >= max_val {
                    return Ok(Some(Point::new(x, y)));
                }
            }
        }
    }
    Ok(None)
}

/// `cv::broadcast(src, shape, dst)` — stretch size-1 axes to `(rows, cols)`.
pub fn broadcast<S: MatLike + ?Sized>(src: &S, rows: i32, cols: i32) -> Result<Mat> {
    let (h, w) = (src.rows(), src.cols());
    cv_assert!(
        (h == rows || h == 1) && (w == cols || w == 1),
        "cannot broadcast {w}x{h} to {cols}x{rows}"
    );
    let cn = src.channels();
    let mut dst = Mat::zeros(rows, cols, src.mat_type())?;
    for y in 0..rows {
        for x in 0..cols {
            let (sy, sx) = (if h == 1 { 0 } else { y }, if w == 1 { 0 } else { x });
            for c in 0..cn {
                let v = src.at_f64(sy, sx, c)?;
                crate::arithm::set_channel_generic(&mut dst, y, x, c, v)?;
            }
        }
    }
    Ok(dst)
}

/// `cv::cubeRoot(val)` — the real cube root, sign-preserving.
#[must_use]
pub fn cube_root(v: f32) -> f32 {
    v.cbrt()
}

/// `cv::transform(src, dst, m)` — apply a per-pixel matrix to the channels.
///
/// `m` is `dcn x scn` or `dcn x (scn + 1)`, the extra column being a translation.
pub fn transform<S: MatLike + ?Sized, M: MatLike + ?Sized>(src: &S, m: &M) -> Result<Mat> {
    let scn = src.channels();
    let dcn = m.rows();
    cv_assert!(
        m.cols() == scn || m.cols() == scn + 1,
        "transform matrix must have {scn} or {} columns, got {}",
        scn + 1,
        m.cols()
    );
    let has_shift = m.cols() == scn + 1;
    let mut dst = Mat::zeros(src.rows(), src.cols(), MatType::new(src.depth(), dcn))?;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            for d in 0..dcn {
                let mut acc = if has_shift { m.at_f64(d, scn, 0)? } else { 0.0 };
                for s in 0..scn {
                    acc += m.at_f64(d, s, 0)? * src.at_f64(y, x, s)?;
                }
                crate::arithm::set_channel_generic(&mut dst, y, x, d, acc)?;
            }
        }
    }
    Ok(dst)
}

/// `cv::perspectiveTransform(src, dst, m)`.
///
/// `src` holds 2- or 3-channel points; `m` is `(n+1) x (n+1)`. Each point is
/// divided through by the final homogeneous coordinate, and upstream's guard
/// against dividing by zero — substituting a large finite value — is kept.
pub fn perspective_transform<S: MatLike + ?Sized, M: MatLike + ?Sized>(
    src: &S,
    m: &M,
) -> Result<Mat> {
    let cn = src.channels();
    cv_assert!(cn == 2 || cn == 3, "perspectiveTransform needs 2 or 3 channels, got {cn}");
    cv_assert!(
        m.rows() == cn + 1 && m.cols() == cn + 1,
        "perspectiveTransform matrix must be {}x{}",
        cn + 1,
        cn + 1
    );
    let mut dst = Mat::zeros(src.rows(), src.cols(), src.mat_type())?;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            let mut hom = vec![0.0f64; (cn + 1) as usize];
            for (d, h) in hom.iter_mut().enumerate() {
                let d = d as i32;
                let mut acc = m.at_f64(d, cn, 0)?;
                for s in 0..cn {
                    acc += m.at_f64(d, s, 0)? * src.at_f64(y, x, s)?;
                }
                *h = acc;
            }
            let w = hom[cn as usize];
            let scale = if w != 0.0 { 1.0 / w } else { f64::MAX };
            for c in 0..cn {
                crate::arithm::set_channel_generic(&mut dst, y, x, c, hom[c as usize] * scale)?;
            }
        }
    }
    Ok(dst)
}

/// `cv::randShuffle(dst, iterFactor, rng)` — Fisher-Yates over the elements.
pub fn rand_shuffle<M: MatLike + MatLikeMut + ?Sized>(
    m: &mut M,
    iter_factor: f64,
    rng: &mut crate::rng::Rng,
) -> Result<()> {
    let (h, w) = (m.rows(), m.cols());
    let cn = m.channels();
    let n = (h * w) as usize;
    if n < 2 {
        return Ok(());
    }
    let iters = ((n as f64) * iter_factor).max(1.0) as usize;
    for _ in 0..iters {
        let a = (rng.next_u32() as usize) % n;
        let b = (rng.next_u32() as usize) % n;
        if a == b {
            continue;
        }
        let (ay, ax) = ((a as i32) / w, (a as i32) % w);
        let (by, bx) = ((b as i32) / w, (b as i32) % w);
        for c in 0..cn {
            let (va, vb) = (m.at_f64(ay, ax, c)?, m.at_f64(by, bx, c)?);
            crate::arithm::set_channel_generic(m, ay, ax, c, vb)?;
            crate::arithm::set_channel_generic(m, by, bx, c, va)?;
        }
    }
    Ok(())
}

/// Distance metric for [`batch_distance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceType {
    /// `cv::NORM_L1` — sum of absolute differences.
    L1,
    /// `cv::NORM_L2` — Euclidean.
    L2,
    /// `cv::NORM_L2SQR` — squared Euclidean, no square root.
    L2Sqr,
    /// `cv::NORM_HAMMING` — differing bytes, for `CV_8U` descriptors.
    Hamming,
}

/// `cv::batchDistance(src1, src2, dist, dtype, nidx, normType, K, ...)`.
///
/// One row per query, one column per training sample. `K` of 0 returns the full
/// matrix; a positive `K` returns the `K` nearest per query along with their
/// indices, sorted ascending by distance.
pub fn batch_distance<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    queries: &A,
    train: &B,
    metric: DistanceType,
    k: i32,
) -> Result<(Mat, Mat)> {
    cv_assert!(
        queries.cols() == train.cols(),
        "batchDistance operands must have equal dimensionality: {} vs {}",
        queries.cols(),
        train.cols()
    );
    cv_assert!(queries.channels() == 1 && train.channels() == 1, "batchDistance is single-channel");
    if metric == DistanceType::Hamming {
        cv_assert!(
            queries.depth() == Depth::U8 && train.depth() == Depth::U8,
            "NORM_HAMMING requires CV_8U descriptors"
        );
    }
    let (nq, nt, d) = (queries.rows(), train.rows(), queries.cols());
    let keep = if k <= 0 { nt } else { k.min(nt) };

    let mut dist = Mat::zeros(nq, keep, CV_64FC1)?;
    let mut idx = Mat::zeros(nq, keep, MatType::new(Depth::S32, 1))?;
    for q in 0..nq {
        let mut row: Vec<(f64, i32)> = Vec::with_capacity(nt as usize);
        for t in 0..nt {
            let mut acc = 0.0f64;
            for c in 0..d {
                let (a, b) = (queries.at_f64(q, c, 0)?, train.at_f64(t, c, 0)?);
                match metric {
                    DistanceType::L1 => acc += (a - b).abs(),
                    DistanceType::L2 | DistanceType::L2Sqr => acc += (a - b) * (a - b),
                    DistanceType::Hamming => acc += ((a as u8) ^ (b as u8)).count_ones() as f64,
                }
            }
            if metric == DistanceType::L2 {
                acc = acc.sqrt();
            }
            row.push((acc, t));
        }
        if k > 0 {
            // Stable by index so equal distances keep training order.
            row.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        }
        for (c, &(v, i)) in row.iter().take(keep as usize).enumerate() {
            dist.pixel_mut::<f64>(q, c as i32)?[0] = v;
            idx.pixel_mut::<i32>(q, c as i32)?[0] = i;
        }
    }
    Ok((dist, idx))
}

/// `cv::cvRound(value)` exposed for `f64`, and `cvTrunc`/`cvFloor`/`cvCeil`'s
/// 64-bit sibling `cvRound64`.
#[must_use]
pub fn cv_round64(v: f64) -> i64 {
    // Same round-half-to-even rule as `cvRound`, widened.
    v.round_ties_even() as i64
}

/// `cv::cvTrunc(value)` — truncation toward zero.
#[must_use]
pub fn cv_trunc(v: f64) -> i32 {
    v.trunc() as i32
}

/// `cv::normalize(src, dst, alpha, beta, NORM_MINMAX)` for a chosen output depth.
///
/// The existing [`crate::normalize_minmax`] keeps the source depth; this variant
/// takes upstream's `dtype` argument.
pub fn normalize_minmax_to<S: MatLike + ?Sized>(
    src: &S,
    alpha: f64,
    beta: f64,
    depth: Depth,
) -> Result<Mat> {
    let e = min_max_idx::<S, Mat>(src, None)?;
    let (lo, hi) = (e.min_val.min(alpha.min(beta)), e.max_val);
    let _ = lo;
    let (dmin, dmax) = (alpha.min(beta), alpha.max(beta));
    let span = e.max_val - e.min_val;
    let scale = if span.abs() > f64::EPSILON { (dmax - dmin) / span } else { 0.0 };
    let shift = dmin - e.min_val * scale;
    let _ = hi;
    let cn = src.channels();
    let mut dst = Mat::zeros(src.rows(), src.cols(), MatType::new(depth, cn))?;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            for c in 0..cn {
                let v = src.at_f64(y, x, c)? * scale + shift;
                crate::arithm::set_channel_generic(&mut dst, y, x, c, v)?;
            }
        }
    }
    Ok(dst)
}

/// `cv::convertFp16(src, dst)` — round-trip through IEEE binary16.
///
/// This port has no `CV_16F` depth, so the conversion is exposed as a value
/// quantiser: the result is `CV_32F` holding exactly the values binary16 can
/// represent. That is what callers actually want from `convertFp16` when they
/// are checking whether a network's weights survive half precision.
pub fn quantize_fp16<S: MatLike + ?Sized>(src: &S) -> Result<Mat> {
    let cn = src.channels();
    let mut dst = Mat::zeros(src.rows(), src.cols(), MatType::new(Depth::F32, cn))?;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            for c in 0..cn {
                let v = f16_roundtrip(src.at_f64(y, x, c)? as f32);
                dst.pixel_mut::<f32>(y, x)?[c as usize] = v;
            }
        }
    }
    Ok(dst)
}

/// Round an `f32` to the nearest IEEE binary16 value and back.
fn f16_roundtrip(v: f32) -> f32 {
    if !v.is_finite() {
        return v;
    }
    let bits = v.to_bits();
    let sign = bits >> 31;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mant = bits & 0x007f_ffff;
    if exp > 15 {
        return if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY };
    }
    if exp < -24 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    // Number of mantissa bits binary16 keeps at this exponent (10 when normal,
    // fewer once subnormal).
    let keep = if exp >= -14 { 10 } else { 10 + (exp + 14) };
    let drop = 23 - keep;
    let step = 1u32 << drop;
    let half = step >> 1;
    let mut m = mant + half;
    // Round half to even, matching hardware conversion.
    if mant & step == 0 && mant & (half.wrapping_sub(1)) == 0 && mant & half != 0 {
        m = mant;
    }
    let mut e = exp;
    let mut q = (m / step) * step;
    if q > 0x007f_ffff {
        q = 0;
        e += 1;
        if e > 15 {
            return if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY };
        }
    }
    let out = (sign << 31) | (((e + 127) as u32) << 23) | q;
    f32::from_bits(out)
}

/// `cv::borderInterpolate` already lives in [`crate::border`]; this is the
/// scalar `cv::round`/`cv::floor` family's missing member, `cvIsNaN`.
#[must_use]
pub fn cv_is_nan(v: f64) -> bool {
    v.is_nan()
}

/// `cv::cvIsInf(value)`.
#[must_use]
pub fn cv_is_inf(v: f64) -> bool {
    v.is_infinite()
}

/// Guard used by callers that need a rounded index rather than a value.
#[must_use]
pub fn round_to_index(v: f64) -> i32 {
    cv_round(v)
}

/// `cv::setRNGSeed(seed)` — construct a generator with upstream's entry point name.
#[must_use]
pub fn set_rng_seed(seed: u64) -> crate::rng::Rng {
    crate::rng::Rng::new(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CV_32FC1, CV_8UC1};

    fn u8m(v: &[u8], r: i32, c: i32) -> Mat {
        Mat::from_vec(r, c, 1, v).unwrap()
    }
    fn f64m(v: &[f64], r: i32, c: i32) -> Mat {
        Mat::from_vec(r, c, 1, v).unwrap()
    }

    #[test]
    fn convert_scale_abs_takes_the_absolute_value_before_saturating() {
        let m = f64m(&[-300.0, -5.0, 5.0, 300.0], 1, 4);
        let r = convert_scale_abs(&m, 1.0, 0.0).unwrap();
        let got: Vec<u8> = r.to_vec::<u8>().unwrap();
        // -300 -> |−300| = 300 -> saturates to 255, not to 0.
        assert_eq!(got, vec![255, 5, 5, 255]);
    }

    #[test]
    fn find_non_zero_returns_raster_order() {
        let m = u8m(&[0, 3, 0, 0, 0, 7], 2, 3);
        assert_eq!(find_non_zero(&m).unwrap(), vec![Point::new(1, 0), Point::new(2, 1)]);
        assert!(has_non_zero(&m).unwrap());
        assert!(!has_non_zero(&u8m(&[0, 0], 1, 2)).unwrap());
    }

    #[test]
    fn complete_symm_mirrors_the_requested_triangle() {
        let mut m = f64m(&[1.0, 2.0, 9.0, 4.0], 2, 2);
        complete_symm(&mut m, false).unwrap(); // upper -> lower
        assert_eq!(m.at_f64(1, 0, 0).unwrap(), 2.0);
        let mut m2 = f64m(&[1.0, 2.0, 9.0, 4.0], 2, 2);
        complete_symm(&mut m2, true).unwrap(); // lower -> upper
        assert_eq!(m2.at_f64(0, 1, 0).unwrap(), 9.0);
    }

    #[test]
    fn rotate_turns_by_the_right_quarter() {
        // 1 2
        // 3 4
        let m = u8m(&[1, 2, 3, 4], 2, 2);
        let cw = rotate(&m, RotateCode::Clockwise90).unwrap();
        assert_eq!(cw.to_vec::<u8>().unwrap(), vec![3, 1, 4, 2]);
        let ccw = rotate(&m, RotateCode::CounterClockwise90).unwrap();
        assert_eq!(ccw.to_vec::<u8>().unwrap(), vec![2, 4, 1, 3]);
        let r180 = rotate(&m, RotateCode::Rotate180).unwrap();
        assert_eq!(r180.to_vec::<u8>().unwrap(), vec![4, 3, 2, 1]);
    }

    #[test]
    fn rotate_is_not_square_only() {
        let m = u8m(&[1, 2, 3, 4, 5, 6], 2, 3);
        let cw = rotate(&m, RotateCode::Clockwise90).unwrap();
        assert_eq!((cw.rows(), cw.cols()), (3, 2));
        assert_eq!(cw.to_vec::<u8>().unwrap(), vec![4, 1, 5, 2, 6, 3]);
    }

    #[test]
    fn min_max_idx_keeps_the_first_of_a_tie() {
        let m = f64m(&[5.0, 1.0, 1.0, 5.0], 2, 2);
        let e = min_max_idx::<Mat, Mat>(&m, None).unwrap();
        assert_eq!(e.min_val, 1.0);
        assert_eq!(e.max_val, 5.0);
        assert_eq!(e.min_loc, Some(Point::new(1, 0)));
        assert_eq!(e.max_loc, Some(Point::new(0, 0)));
    }

    #[test]
    fn min_max_idx_honours_a_mask() {
        let m = f64m(&[5.0, 1.0, 2.0, 3.0], 2, 2);
        let mask = Mat::from_vec(2, 2, 1, &[0u8, 0, 1, 1]).unwrap();
        let e = min_max_idx(&m, Some(&mask)).unwrap();
        assert_eq!(e.min_val, 2.0);
        assert_eq!(e.max_val, 3.0);
    }

    #[test]
    fn reduce_arg_finds_the_extreme_index() {
        // 1 9 3
        // 7 2 8
        let m = f64m(&[1.0, 9.0, 3.0, 7.0, 2.0, 8.0], 2, 3);
        let amax_rows = reduce_arg(&m, 1, true).unwrap(); // per row
        assert_eq!(amax_rows.to_vec::<i32>().unwrap(), vec![1, 2]);
        let amin_cols = reduce_arg(&m, 0, false).unwrap(); // per column
        assert_eq!(amin_cols.to_vec::<i32>().unwrap(), vec![0, 1, 0]);
    }

    #[test]
    fn check_range_uses_a_half_open_interval() {
        let m = f64m(&[0.0, 5.0, 9.999], 1, 3);
        assert_eq!(check_range(&m, 0.0, 10.0).unwrap(), None);
        let at_top = f64m(&[0.0, 10.0], 1, 2);
        assert_eq!(check_range(&at_top, 0.0, 10.0).unwrap(), Some(Point::new(1, 0)));
        let nan = f64m(&[f64::NAN], 1, 1);
        assert_eq!(check_range(&nan, -1e9, 1e9).unwrap(), Some(Point::new(0, 0)));
    }

    #[test]
    fn broadcast_stretches_a_row_and_a_column() {
        let row = f64m(&[1.0, 2.0, 3.0], 1, 3);
        let b = broadcast(&row, 2, 3).unwrap();
        assert_eq!(b.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let col = f64m(&[1.0, 2.0], 2, 1);
        let b2 = broadcast(&col, 2, 3).unwrap();
        assert_eq!(b2.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        assert!(broadcast(&row, 2, 4).is_err(), "3 cannot stretch to 4");
    }

    #[test]
    fn patch_nans_replaces_only_nans() {
        let mut m = Mat::from_vec(1, 3, 1, &[1.0f32, f32::NAN, 3.0]).unwrap();
        patch_nans(&mut m, -1.0).unwrap();
        assert_eq!(m.to_vec::<f32>().unwrap(), vec![1.0, -1.0, 3.0]);
        assert_eq!(m.mat_type(), CV_32FC1);
    }

    #[test]
    fn transform_applies_the_channel_matrix() {
        // Two-channel input, swap the channels and add a shift.
        let src = Mat::from_vec(1, 2, 2, &[1.0f64, 2.0, 3.0, 4.0]).unwrap();
        let m = f64m(&[0.0, 1.0, 10.0, 1.0, 0.0, 20.0], 2, 3);
        let d = transform(&src, &m).unwrap();
        assert_eq!(d.to_vec::<f64>().unwrap(), vec![12.0, 21.0, 14.0, 23.0]);
    }

    #[test]
    fn perspective_transform_divides_by_the_homogeneous_coordinate() {
        // Scale by 2 with w = 2, so the result is the identity.
        let src = Mat::from_vec(1, 1, 2, &[3.0f64, 5.0]).unwrap();
        let m = f64m(&[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0], 3, 3);
        let d = perspective_transform(&src, &m).unwrap();
        assert_eq!(d.to_vec::<f64>().unwrap(), vec![3.0, 5.0]);
    }

    #[test]
    fn batch_distance_l2_and_knn() {
        let q = f64m(&[0.0, 0.0], 1, 2);
        let t = f64m(&[3.0, 4.0, 1.0, 0.0, 0.0, 2.0], 3, 2);
        let (d, i) = batch_distance(&q, &t, DistanceType::L2, 0).unwrap();
        assert_eq!(d.to_vec::<f64>().unwrap(), vec![5.0, 1.0, 2.0]);
        let (d2, i2) = batch_distance(&q, &t, DistanceType::L2, 2).unwrap();
        assert_eq!(d2.to_vec::<f64>().unwrap(), vec![1.0, 2.0]);
        assert_eq!(i2.to_vec::<i32>().unwrap(), vec![1, 2]);
        assert_eq!(i.to_vec::<i32>().unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn batch_distance_hamming_counts_differing_bits() {
        let q = Mat::from_vec(1, 2, 1, &[0b1111_0000u8, 0]).unwrap();
        let t = Mat::from_vec(1, 2, 1, &[0b1010_0000u8, 0b1u8]).unwrap();
        let (d, _) = batch_distance(&q, &t, DistanceType::Hamming, 0).unwrap();
        assert_eq!(d.at_f64(0, 0, 0).unwrap(), 3.0, "two bits in byte 0, one in byte 1");
    }

    #[test]
    fn rand_shuffle_permutes_without_losing_elements() {
        let mut m = f64m(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let mut rng = crate::rng::Rng::new(7);
        rand_shuffle(&mut m, 2.0, &mut rng).unwrap();
        let mut got = m.to_vec::<f64>().unwrap();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn fp16_quantisation_snaps_to_representable_values() {
        // 1.0 and 0.5 are exact in binary16; 1e-8 underflows to zero and 1e5
        // overflows to infinity (binary16's max is 65504).
        let m = Mat::from_vec(1, 4, 1, &[1.0f32, 0.5, 1e-8, 1e5]).unwrap();
        let q = quantize_fp16(&m).unwrap();
        let v = q.to_vec::<f32>().unwrap();
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 0.5);
        assert_eq!(v[2], 0.0);
        assert!(v[3].is_infinite());
        // 0.1 is not representable: it must move.
        let m2 = Mat::from_vec(1, 1, 1, &[0.1f32]).unwrap();
        let q2 = quantize_fp16(&m2).unwrap().to_vec::<f32>().unwrap()[0];
        assert_ne!(q2, 0.1f32);
        assert!((q2 - 0.1).abs() < 1e-3, "but it must stay close: {q2}");
    }

    #[test]
    fn normalize_minmax_to_changes_depth() {
        let m = f64m(&[0.0, 5.0, 10.0], 1, 3);
        let n = normalize_minmax_to(&m, 0.0, 255.0, Depth::U8).unwrap();
        assert_eq!(n.mat_type(), CV_8UC1);
        assert_eq!(n.to_vec::<u8>().unwrap(), vec![0, 128, 255]);
    }

    #[test]
    fn cube_root_keeps_the_sign() {
        assert!((cube_root(27.0) - 3.0).abs() < 1e-6);
        assert!((cube_root(-27.0) + 3.0).abs() < 1e-6);
    }
}
