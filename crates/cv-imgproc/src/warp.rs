//! Geometric transforms (`modules/imgproc/src/imgwarp.cpp`).
//!
//! # Sub-pixel quantisation
//!
//! Upstream does not evaluate the inverse map in floating point per pixel. It
//! accumulates the map in `AB_BITS = 10` fixed point and truncates the
//! fractional part to `INTER_BITS = 5` bits, so every sample coordinate lands
//! on a multiple of 1/32 of a pixel. That quantisation is reproduced here,
//! because it — not the weight precision — is what moves sampled pixels.
//!
//! The interpolation weights are then computed in `f32` from the quantised
//! fraction, whereas upstream looks them up in a 15-bit table
//! (`INTER_REMAP_COEF_BITS`). The table is itself built from the same
//! fractions, so results agree except where the 15-bit rounding tips a value
//! across a level.
//!
//! # Where the reference library does something else
//!
//! `remap` and `warpPerspective` both offer the work to a hardware HAL before
//! running any of the above, and the reference build answers with KleidiCV.
//! Its kernels take 8- and 16-bit **unsigned** images of **one or two**
//! channels, interpolate from the **exact** fraction rather than the 1/32
//! grid, and round halves **away from zero** rather than to even. Three- and
//! four-channel images of the same depth, and every signed or floating depth,
//! fall through to the portable path above. `warpAffine` shows no such split.
//!
//! Both behaviours are reproduced, because parity is measured against the
//! library as installed. Neither is in `imgwarp.cpp`, so neither can be found
//! by reading it — see the `hal_path` comments for how each was established.

use cv_core::saturate::cv_round;
use cv_core::types::Depth;
use cv_core::{
    border_interpolate, cv_assert, cv_error, BorderType, Code, Mat, MatLike, Result, Scalar, Size,
};

use crate::color::set_channel;
use crate::resize::Interpolation;

/// `INTER_BITS` — fractional bits kept for sub-pixel coordinates.
const INTER_BITS: i32 = 5;
/// `INTER_TAB_SIZE` — number of sub-pixel steps per whole pixel.
const INTER_TAB_SIZE: i32 = 1 << INTER_BITS;
/// `AB_BITS` — fixed-point bits used while accumulating the affine map.
const AB_BITS: i32 = 10;
/// `AB_SCALE`.
const AB_SCALE: f64 = (1 << AB_BITS) as f64;

/// A 2x3 affine transform, stored row-major — the equivalent of `cv::Matx23d`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine2x3(pub [f64; 6]);

impl Affine2x3 {
    /// Build from six coefficients in row-major order.
    #[must_use]
    pub const fn new(m: [f64; 6]) -> Self {
        Affine2x3(m)
    }

    /// Apply to a point.
    #[must_use]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let m = &self.0;
        (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
    }

    /// `cv::invertAffineTransform(M, iM)`.
    ///
    /// Returns `None` when the linear part is singular.
    #[must_use]
    pub fn invert(&self) -> Option<Affine2x3> {
        let m = &self.0;
        let d = m[0] * m[4] - m[1] * m[3];
        if d == 0.0 {
            return None;
        }
        let d = 1.0 / d;
        let a11 = m[4] * d;
        let a12 = -m[1] * d;
        let a21 = -m[3] * d;
        let a22 = m[0] * d;
        let b1 = -a11 * m[2] - a12 * m[5];
        let b2 = -a21 * m[2] - a22 * m[5];
        Some(Affine2x3([a11, a12, b1, a21, a22, b2]))
    }
}

/// A 3x3 projective transform, stored row-major.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Perspective3x3(pub [f64; 9]);

impl Perspective3x3 {
    /// Build from nine coefficients in row-major order.
    #[must_use]
    pub const fn new(m: [f64; 9]) -> Self {
        Perspective3x3(m)
    }

    /// Apply to a point, dividing through by the homogeneous coordinate.
    #[must_use]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let m = &self.0;
        let w = m[6] * x + m[7] * y + m[8];
        let w = if w != 0.0 { 1.0 / w } else { 0.0 };
        ((m[0] * x + m[1] * y + m[2]) * w, (m[3] * x + m[4] * y + m[5]) * w)
    }

    /// Matrix inverse, or `None` when singular.
    #[must_use]
    pub fn invert(&self) -> Option<Perspective3x3> {
        let m = &self.0;
        let c = |a: usize, b: usize, c2: usize, d: usize| m[a] * m[b] - m[c2] * m[d];
        let i0 = c(4, 8, 5, 7);
        let i1 = c(2, 7, 1, 8);
        let i2 = c(1, 5, 2, 4);
        let det = m[0] * i0 + m[3] * i1 + m[6] * i2;
        if det == 0.0 {
            return None;
        }
        let d = 1.0 / det;
        Some(Perspective3x3([
            i0 * d,
            i1 * d,
            i2 * d,
            c(5, 6, 3, 8) * d,
            c(0, 8, 2, 6) * d,
            c(2, 3, 0, 5) * d,
            c(3, 7, 4, 6) * d,
            c(1, 6, 0, 7) * d,
            c(0, 4, 1, 3) * d,
        ]))
    }
}

/// `cv::getRotationMatrix2D(center, angle, scale)`.
///
/// The angle is in degrees and rotation is counter-clockwise, with the origin
/// at the top-left of the image.
#[must_use]
pub fn get_rotation_matrix_2d(center: (f64, f64), angle: f64, scale: f64) -> Affine2x3 {
    let angle = angle * core::f64::consts::PI / 180.0;
    let alpha = angle.cos() * scale;
    let beta = angle.sin() * scale;
    Affine2x3([
        alpha,
        beta,
        (1.0 - alpha) * center.0 - beta * center.1,
        -beta,
        alpha,
        beta * center.0 + (1.0 - alpha) * center.1,
    ])
}

/// `cv::getAffineTransform(src, dst)` — the unique affine map taking three
/// source points to three destination points.
pub fn get_affine_transform(src: [(f64, f64); 3], dst: [(f64, f64); 3]) -> Result<Affine2x3> {
    // Solve two independent 3x3 systems, one per output coordinate.
    let a = [
        [src[0].0, src[0].1, 1.0],
        [src[1].0, src[1].1, 1.0],
        [src[2].0, src[2].1, 1.0],
    ];
    let bx = [dst[0].0, dst[1].0, dst[2].0];
    let by = [dst[0].1, dst[1].1, dst[2].1];
    let rx = solve3(&a, &bx).ok_or_else(|| {
        cv_core::Error::new(Code::StsBadArg, "source points are collinear".to_string())
    })?;
    let ry = solve3(&a, &by).ok_or_else(|| {
        cv_core::Error::new(Code::StsBadArg, "source points are collinear".to_string())
    })?;
    Ok(Affine2x3([rx[0], rx[1], rx[2], ry[0], ry[1], ry[2]]))
}

/// `cv::getPerspectiveTransform(src, dst)` — the projective map taking four
/// source points to four destination points, normalised so `m[8] == 1`.
pub fn get_perspective_transform(
    src: [(f64, f64); 4],
    dst: [(f64, f64); 4],
) -> Result<Perspective3x3> {
    // Eight unknowns from four point correspondences, two rows each.
    let mut a = [[0.0f64; 8]; 8];
    let mut b = [0.0f64; 8];
    for i in 0..4 {
        let (sx, sy) = src[i];
        let (dx, dy) = dst[i];
        a[i] = [sx, sy, 1.0, 0.0, 0.0, 0.0, -sx * dx, -sy * dx];
        a[i + 4] = [0.0, 0.0, 0.0, sx, sy, 1.0, -sx * dy, -sy * dy];
        b[i] = dx;
        b[i + 4] = dy;
    }
    let x = solve_n(&mut a, &mut b).ok_or_else(|| {
        cv_core::Error::new(Code::StsBadArg, "degenerate point configuration".to_string())
    })?;
    Ok(Perspective3x3([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1.0]))
}

/// Gaussian elimination for a 3x3 system.
fn solve3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    let mut m = [[0.0f64; 4]; 3];
    for i in 0..3 {
        m[i][..3].copy_from_slice(&a[i]);
        m[i][3] = b[i];
    }
    for col in 0..3 {
        let piv = (col..3).max_by(|&i, &j| {
            m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap_or(core::cmp::Ordering::Equal)
        })?;
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        for row in 0..3 {
            if row == col {
                continue;
            }
            let f = m[row][col] / m[col][col];
            let (pivot_row, target) = (m[col], &mut m[row]);
            for k in col..4 {
                target[k] -= f * pivot_row[k];
            }
        }
    }
    Some([m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]])
}

/// Gaussian elimination for an 8x8 system.
fn solve_n(a: &mut [[f64; 8]; 8], b: &mut [f64; 8]) -> Option<[f64; 8]> {
    const N: usize = 8;
    for col in 0..N {
        let piv = (col..N).max_by(|&i, &j| {
            a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap_or(core::cmp::Ordering::Equal)
        })?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in 0..N {
            if row == col {
                continue;
            }
            let f = a[row][col] / a[col][col];
            let (pivot_row, target) = (a[col], &mut a[row]);
            for k in col..N {
                target[k] -= f * pivot_row[k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0f64; N];
    for i in 0..N {
        x[i] = b[i] / a[i][i];
    }
    Some(x)
}

/// Quantise a floating coordinate to `INTER_BITS` fractional bits the way
/// upstream's fixed-point map accumulation does.
///
/// Returns `(whole, fraction_index)` where `fraction_index` is in
/// `0..INTER_TAB_SIZE`.
/// The old single-expression quantiser, kept only for the tests that pin its
/// behaviour.
///
/// Production code no longer uses it: `warpAffine` rounds the x- and
/// y-contributions to integers *separately* before summing, which this cannot
/// express, and `remap` carries an exact fraction rather than quantising at all.
#[cfg(test)]
fn quantise(v: f64) -> (i32, i32) {
    let scaled = (v * AB_SCALE) + AB_SCALE / (INTER_TAB_SIZE as f64) / 2.0;
    let fixed = (scaled.floor() as i64) >> (AB_BITS - INTER_BITS);
    ((fixed >> INTER_BITS) as i32, (fixed & (INTER_TAB_SIZE as i64 - 1)) as i32)
}


/// The HAL's rounding on the way back to an integer pixel.
///
/// `saturate_cast<uchar>` rounds halves to **even**, and that is what the
/// portable path ends up doing everywhere the accumulator is floating point.
/// The KleidiCV kernels round halves **away from zero** instead. It only shows
/// on an exact tie, which arbitrary coordinate maps almost never produce — but
/// `warpPolar`'s first row samples whole and half pixels, and there 106 of the
/// 19200 samples land exactly on a half.
fn round_half_up(v: f64, hal_path: bool, depth: Depth) -> f64 {
    if hal_path && !depth.is_float() {
        // `round` in Rust is already ties-away-from-zero.
        v.round()
    } else {
        v
    }
}

/// Sample `src` at a quantised sub-pixel location.
#[allow(clippy::too_many_arguments)]
fn sample_at<S: MatLike + ?Sized>(
    src: &S,
    sx: i32,
    sy: i32,
    // Sub-pixel offsets in `0..1`, already quantised by the caller if that
    // caller's upstream quantises.
    fx: f64,
    fy: f64,
    // Whether to blend with `remapBilinear`'s 15-bit integer weights. Those
    // exist only for the 8-bit destination — `linear_tab` gives every other
    // depth a `float` weight table — and they are only correct when the
    // fraction is already on the 1/32 grid.
    fixed_point: bool,
    c: i32,
    interpolation: Interpolation,
    border: BorderType,
    border_value: f64,
) -> Result<f64> {
    let fetch = |y: i32, x: i32| -> Result<f64> {
        let yy = border_interpolate(y, src.rows(), border)?;
        let xx = border_interpolate(x, src.cols(), border)?;
        if yy < 0 || xx < 0 {
            Ok(border_value)
        } else {
            src.at_f64(yy, xx, c)
        }
    };

    match interpolation {
        Interpolation::Nearest => {
            // Round to the nearer whole pixel using the retained fraction.
            let nx = if fx >= 0.5 { sx + 1 } else { sx };
            let ny = if fy >= 0.5 { sy + 1 } else { sy };
            fetch(ny, nx)
        }
        _ => {
            let v00 = fetch(sy, sx)?;
            let v01 = fetch(sy, sx + 1)?;
            let v10 = fetch(sy + 1, sx)?;
            let v11 = fetch(sy + 1, sx + 1)?;

            if !fixed_point || src.depth() != Depth::U8 {
                return Ok(v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy);
            }
            // `remapBilinear` for integer destinations: `short` weights at
            // `INTER_REMAP_COEF_SCALE = 1 << 15`. With 1/32 fractions the four
            // products are exact multiples of 32 and already sum to 32768, so
            // `initInterTab2D`'s fixup never fires and this reduces to a 1/1024
            // blend with a `+512` rounding term.
            let t = f64::from(INTER_TAB_SIZE);
            let (qx, qy) = ((fx * t).round(), (fy * t).round());
            let (gx, gy) = (t - qx, t - qy);
            let acc = v00 * gx * gy + v01 * qx * gy + v10 * gx * qy + v11 * qx * qy;
            Ok(((acc + 512.0).floor() as i64 >> 10) as f64)
        }
    }
}

/// `cv::warpAffine(src, dst, M, dsize, flags, borderMode, borderValue)`.
///
/// `m` maps **source to destination**; the inverse is taken internally, which
/// is what `WARP_INVERSE_MAP` disables. Pass `inverse_map = true` when `m`
/// already maps destination to source.
#[allow(clippy::too_many_arguments)]
pub fn warp_affine<S: MatLike + ?Sized>(
    src: &S,
    m: Affine2x3,
    dsize: Size,
    interpolation: Interpolation,
    inverse_map: bool,
    border: BorderType,
    border_value: Scalar,
) -> Result<Mat> {
    cv_assert!(!src.is_empty(), "warpAffine requires a non-empty source");
    cv_assert!(dsize.width > 0 && dsize.height > 0, "destination size must be positive");
    let inv = if inverse_map {
        m
    } else {
        m.invert().ok_or_else(|| {
            cv_core::Error::new(Code::StsBadArg, "affine matrix is not invertible".to_string())
        })?
    };

    // `round_delta` is AB_SCALE/2 for NEAREST and AB_SCALE/INTER_TAB_SIZE/2
    // otherwise — a half-step in whichever grid the result lands on.
    let round_delta = if interpolation == Interpolation::Nearest {
        (AB_SCALE / 2.0) as i64
    } else {
        (AB_SCALE / f64::from(INTER_TAB_SIZE) / 2.0) as i64
    };

    let cn = src.channels();
    let mut dst = Mat::zeros(dsize.height, dsize.width, src.mat_type())?;
    for dy in 0..dsize.height {
        for dx in 0..dsize.width {
            // Upstream builds the map from **two separately rounded** integer
            // parts, not one `f64` expression:
            //   X0       = cvRound((M1*y + M2) * AB_SCALE) + round_delta
            //   adelta[x] = cvRound(M0*x * AB_SCALE)
            //   X        = (X0 + adelta[x]) >> (AB_BITS - INTER_BITS)
            // Summing in `f64` and flooring once is *not* the same: each
            // `cvRound` discards up to half an AB unit, and the two errors do
            // not cancel. It moved 2% of pixels across a sub-pixel boundary.
            let a = inv.0;
            let x0 = cv_round((a[1] * f64::from(dy) + a[2]) * AB_SCALE) as i64 + round_delta;
            let y0 = cv_round((a[4] * f64::from(dy) + a[5]) * AB_SCALE) as i64 + round_delta;
            let xq = (x0 + cv_round(a[0] * f64::from(dx) * AB_SCALE) as i64) >> (AB_BITS - INTER_BITS);
            let yq = (y0 + cv_round(a[3] * f64::from(dx) * AB_SCALE) as i64) >> (AB_BITS - INTER_BITS);
            let (sx, sy) = ((xq >> INTER_BITS) as i32, (yq >> INTER_BITS) as i32);
            // For NEAREST the larger `round_delta` has already rounded the
            // coordinate, and upstream stores no sub-pixel index at all —
            // carrying one here would round a second time.
            let (fx, fy) = if interpolation == Interpolation::Nearest {
                (0.0, 0.0)
            } else {
                let t = f64::from(INTER_TAB_SIZE);
                (
                    ((xq & i64::from(INTER_TAB_SIZE - 1)) as f64) / t,
                    ((yq & i64::from(INTER_TAB_SIZE - 1)) as f64) / t,
                )
            };
            for c in 0..cn {
                let v = sample_at(
                    src,
                    sx,
                    sy,
                    fx,
                    fy,
                    true,
                    c,
                    interpolation,
                    border,
                    border_value.0[(c as usize).min(3)],
                )?;
                set_channel(&mut dst, dy, dx, c, v)?;
            }
        }
    }
    Ok(dst)
}

/// `cv::warpPerspective(src, dst, M, dsize, flags, borderMode, borderValue)`.
#[allow(clippy::too_many_arguments)]
pub fn warp_perspective<S: MatLike + ?Sized>(
    src: &S,
    m: Perspective3x3,
    dsize: Size,
    interpolation: Interpolation,
    inverse_map: bool,
    border: BorderType,
    border_value: Scalar,
) -> Result<Mat> {
    cv_assert!(!src.is_empty(), "warpPerspective requires a non-empty source");
    cv_assert!(dsize.width > 0 && dsize.height > 0, "destination size must be positive");
    let inv = if inverse_map {
        m
    } else {
        m.invert().ok_or_else(|| {
            cv_core::Error::new(Code::StsBadArg, "perspective matrix is not invertible".to_string())
        })?
    };

    let cn = src.channels();
    // Same split as `remap`: KleidiCV answers `cv_hal_warpPerspective` for 8-
    // and 16-bit unsigned images of one or two channels and interpolates from
    // the exact fraction; everything else falls through to the portable path,
    // which snaps to 1/32. `warpAffine` has no such divergence — it quantises
    // at every depth and channel count.
    let hal_path = matches!(src.depth(), Depth::U8 | Depth::U16) && cn <= 2;
    let mut dst = Mat::zeros(dsize.height, dsize.width, src.mat_type())?;
    for dy in 0..dsize.height {
        for dx in 0..dsize.width {
            let a = inv.0;
            let (x, y) = (f64::from(dx), f64::from(dy));
            let w0 = a[6] * x + a[7] * y + a[8];
            let recip = if w0 != 0.0 { 1.0 / w0 } else { 0.0 };
            let sxf = (a[0] * x + a[1] * y + a[2]) * recip;
            let syf = (a[3] * x + a[4] * y + a[5]) * recip;
            let (sx, fx, sy, fy) = if hal_path {
                let (bx, by) = (sxf.floor(), syf.floor());
                (bx as i32, sxf - bx, by as i32, syf - by)
            } else {
                let t = f64::from(INTER_TAB_SIZE);
                let (qx, qy) = (cv_round(sxf * t), cv_round(syf * t));
                (qx >> INTER_BITS, f64::from(qx & 31) / t, qy >> INTER_BITS, f64::from(qy & 31) / t)
            };
            for c in 0..cn {
                let v = sample_at(
                    src,
                    sx,
                    sy,
                    fx,
                    fy,
                    !hal_path,
                    c,
                    interpolation,
                    border,
                    border_value.0[(c as usize).min(3)],
                )?;
                set_channel(&mut dst, dy, dx, c, round_half_up(v, hal_path, src.depth()))?;
            }
        }
    }
    Ok(dst)
}

/// `cv::remap(src, dst, map1, map2, interpolation, borderMode, borderValue)`.
///
/// `map_x` and `map_y` must be `CV_32FC1` and the same size; the result takes
/// their size.
pub fn remap<S: MatLike + ?Sized, MX: MatLike + ?Sized, MY: MatLike + ?Sized>(
    src: &S,
    map_x: &MX,
    map_y: &MY,
    interpolation: Interpolation,
    border: BorderType,
    border_value: Scalar,
) -> Result<Mat> {
    if map_x.size() != map_y.size() {
        return cv_error!(Code::StsUnmatchedSizes, "remap maps must have the same size");
    }
    cv_assert!(map_x.channels() == 1 && map_y.channels() == 1, "remap maps must be single-channel");

    let cn = src.channels();

    // **`remap` is one of the places the reference library disagrees with its
    // own source.** `RemapInvoker` takes `cvRound(v * INTER_TAB_SIZE)`, splits
    // it into a base `v >> INTER_BITS` and a five-bit table index, and looks
    // the weights up in `initInterTab2D` — so the portable path snaps every
    // sample to 1/32 of a pixel. But `remap()` first offers the work to
    // `cv_hal_remap32f`, and the reference build here answers with KleidiCV,
    // which interpolates from the **exact** fraction in float. KleidiCV takes
    // only 8- and 16-bit unsigned images of one or two channels; three- and
    // four-channel images of the same depth, and every signed or floating
    // depth, fall through to the quantised path. Measured, not read: on a
    // 255-per-pixel edge the two answers are two levels apart, and the split
    // is exactly at three channels.
    let hal_path = matches!(src.depth(), Depth::U8 | Depth::U16) && cn <= 2;

    let mut dst = Mat::zeros(map_x.rows(), map_x.cols(), src.mat_type())?;
    for dy in 0..map_x.rows() {
        for dx in 0..map_x.cols() {
            let (mvx, mvy) = (map_x.at_f64(dy, dx, 0)?, map_y.at_f64(dy, dx, 0)?);
            let (sx, fx, sy, fy) = if interpolation == Interpolation::Nearest {
                // `saturate_cast<short>` of the float coordinate, i.e. a whole
                // round — no table, no fraction.
                (cv_round(mvx), 0.0, cv_round(mvy), 0.0)
            } else if hal_path {
                (mvx.floor() as i32, mvx - mvx.floor(), mvy.floor() as i32, mvy - mvy.floor())
            } else {
                let t = f64::from(INTER_TAB_SIZE);
                // An arithmetic shift, not a division: the base pixel of a
                // negative coordinate is its floor, and `-1 >> 5` is `-1`.
                let (qx, qy) = (cv_round(mvx * t), cv_round(mvy * t));
                (qx >> INTER_BITS, f64::from(qx & 31) / t, qy >> INTER_BITS, f64::from(qy & 31) / t)
            };
            for c in 0..cn {
                let v = sample_at(
                    src,
                    sx,
                    sy,
                    fx,
                    fy,
                    !hal_path,
                    c,
                    interpolation,
                    border,
                    border_value.0[(c as usize).min(3)],
                )?;
                set_channel(&mut dst, dy, dx, c, round_half_up(v, hal_path, src.depth()))?;
            }
        }
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::{CV_32FC1, CV_8UC1, CV_8UC3};
    use cv_core::MatLikeMut;

    fn ramp(w: i32, h: i32) -> Mat {
        let data: Vec<u8> = (0..(w * h)).map(|i| (i % 256) as u8).collect();
        Mat::from_vec(h, w, 1, &data).unwrap()
    }

    fn identity() -> Affine2x3 {
        Affine2x3::new([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    }

    #[test]
    fn rotation_by_zero_is_a_scaled_identity() {
        let m = get_rotation_matrix_2d((10.0, 20.0), 0.0, 1.0);
        assert!((m.0[0] - 1.0).abs() < 1e-12);
        assert!(m.0[1].abs() < 1e-12);
        assert!(m.0[2].abs() < 1e-12);
        assert!((m.0[4] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rotation_fixes_its_centre() {
        let c = (7.5, 3.25);
        for angle in [0.0, 30.0, 90.0, 180.0, -45.0] {
            let m = get_rotation_matrix_2d(c, angle, 1.0);
            let (x, y) = m.apply(c.0, c.1);
            assert!((x - c.0).abs() < 1e-9 && (y - c.1).abs() < 1e-9, "angle {angle}: {x},{y}");
        }
    }

    #[test]
    fn rotation_by_90_matches_the_expected_matrix() {
        // OpenCV's convention rotates counter-clockwise about the centre.
        let m = get_rotation_matrix_2d((0.0, 0.0), 90.0, 1.0);
        let (x, y) = m.apply(1.0, 0.0);
        assert!(x.abs() < 1e-12, "{x}");
        assert!((y + 1.0).abs() < 1e-12, "{y}");
    }

    #[test]
    fn affine_invert_roundtrips() {
        let m = get_rotation_matrix_2d((4.0, 5.0), 37.0, 1.7);
        let inv = m.invert().unwrap();
        let (x, y) = m.apply(11.0, -3.0);
        let (bx, by) = inv.apply(x, y);
        assert!((bx - 11.0).abs() < 1e-9, "{bx}");
        assert!((by + 3.0).abs() < 1e-9, "{by}");
    }

    #[test]
    fn singular_affine_has_no_inverse() {
        assert!(Affine2x3::new([1.0, 2.0, 0.0, 2.0, 4.0, 0.0]).invert().is_none());
    }

    #[test]
    fn get_affine_transform_reproduces_the_correspondences() {
        let s = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let d = [(3.0, 4.0), (5.0, 4.0), (3.0, 9.0)];
        let m = get_affine_transform(s, d).unwrap();
        for i in 0..3 {
            let (x, y) = m.apply(s[i].0, s[i].1);
            assert!((x - d[i].0).abs() < 1e-9 && (y - d[i].1).abs() < 1e-9, "point {i}");
        }
    }

    #[test]
    fn collinear_points_are_rejected() {
        let s = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let d = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        assert!(get_affine_transform(s, d).is_err());
    }

    #[test]
    fn get_perspective_transform_reproduces_the_correspondences() {
        let s = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let d = [(2.0, 1.0), (12.0, 3.0), (11.0, 14.0), (1.0, 9.0)];
        let m = get_perspective_transform(s, d).unwrap();
        for i in 0..4 {
            let (x, y) = m.apply(s[i].0, s[i].1);
            assert!((x - d[i].0).abs() < 1e-8 && (y - d[i].1).abs() < 1e-8, "point {i}: {x},{y}");
        }
    }

    #[test]
    fn perspective_invert_roundtrips() {
        let s = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let d = [(2.0, 1.0), (12.0, 3.0), (11.0, 14.0), (1.0, 9.0)];
        let m = get_perspective_transform(s, d).unwrap();
        let inv = m.invert().unwrap();
        let (x, y) = m.apply(4.0, 6.0);
        let (bx, by) = inv.apply(x, y);
        assert!((bx - 4.0).abs() < 1e-7, "{bx}");
        assert!((by - 6.0).abs() < 1e-7, "{by}");
    }

    #[test]
    fn identity_warp_is_a_copy() {
        let m = ramp(6, 5);
        let w = warp_affine(
            &m,
            identity(),
            Size::new(6, 5),
            Interpolation::Linear,
            false,
            BorderType::Constant,
            Scalar::default(),
        )
        .unwrap();
        assert_eq!(w.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn integer_translation_shifts_exactly() {
        let m = ramp(5, 4);
        let t = Affine2x3::new([1.0, 0.0, 2.0, 0.0, 1.0, 1.0]);
        let w = warp_affine(
            &m,
            t,
            Size::new(5, 4),
            Interpolation::Nearest,
            false,
            BorderType::Constant,
            Scalar::default(),
        )
        .unwrap();
        // Source (0,0) lands at destination (2,1).
        assert_eq!(w.at::<u8>(1, 2).unwrap(), m.at::<u8>(0, 0).unwrap());
        assert_eq!(w.at::<u8>(3, 4).unwrap(), m.at::<u8>(2, 2).unwrap());
        // The uncovered strip keeps the border value.
        assert_eq!(w.at::<u8>(0, 0).unwrap(), 0);
    }

    #[test]
    fn rotation_by_180_reverses_the_image() {
        let m = Mat::from_vec(2, 2, 1, &[1u8, 2, 3, 4]).unwrap();
        // Rotating about the centre of the 2x2 pixel grid.
        let r = get_rotation_matrix_2d((0.5, 0.5), 180.0, 1.0);
        let w = warp_affine(
            &m,
            r,
            Size::new(2, 2),
            Interpolation::Nearest,
            false,
            BorderType::Constant,
            Scalar::default(),
        )
        .unwrap();
        assert_eq!(w.to_vec::<u8>().unwrap(), vec![4, 3, 2, 1]);
    }

    #[test]
    fn constant_border_fills_uncovered_pixels() {
        let m = Mat::new_with(4, 4, CV_8UC1, Scalar::single(200.0)).unwrap();
        let t = Affine2x3::new([1.0, 0.0, 10.0, 0.0, 1.0, 10.0]);
        let w = warp_affine(
            &m,
            t,
            Size::new(4, 4),
            Interpolation::Linear,
            false,
            BorderType::Constant,
            Scalar::single(7.0),
        )
        .unwrap();
        assert!(w.to_vec::<u8>().unwrap().iter().all(|&v| v == 7));
    }

    #[test]
    fn inverse_map_flag_skips_the_internal_inversion() {
        let m = ramp(6, 6);
        let fwd = Affine2x3::new([1.0, 0.0, 2.0, 0.0, 1.0, 0.0]);
        let a = warp_affine(&m, fwd, Size::new(6, 6), Interpolation::Nearest, false, BorderType::Constant, Scalar::default()).unwrap();
        let b = warp_affine(&m, fwd.invert().unwrap(), Size::new(6, 6), Interpolation::Nearest, true, BorderType::Constant, Scalar::default()).unwrap();
        assert_eq!(a.to_vec::<u8>().unwrap(), b.to_vec::<u8>().unwrap());
    }

    #[test]
    fn warp_preserves_a_uniform_multichannel_image() {
        let m = Mat::new_with(8, 8, CV_8UC3, Scalar::new(10.0, 20.0, 30.0, 0.0)).unwrap();
        let r = get_rotation_matrix_2d((4.0, 4.0), 23.0, 1.0);
        let w = warp_affine(&m, r, Size::new(8, 8), Interpolation::Linear, false, BorderType::Replicate, Scalar::default()).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(w.pixel::<u8>(y, x).unwrap(), &[10, 20, 30], "at {y},{x}");
            }
        }
    }

    #[test]
    fn perspective_identity_is_a_copy() {
        let m = ramp(5, 5);
        let id = Perspective3x3::new([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let w = warp_perspective(&m, id, Size::new(5, 5), Interpolation::Linear, false, BorderType::Constant, Scalar::default()).unwrap();
        assert_eq!(w.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn remap_with_an_identity_map_is_a_copy() {
        let m = ramp(5, 4);
        let mut mx = Mat::zeros(4, 5, CV_32FC1).unwrap();
        let mut my = Mat::zeros(4, 5, CV_32FC1).unwrap();
        for y in 0..4 {
            for x in 0..5 {
                mx.pixel_mut::<f32>(y, x).unwrap()[0] = x as f32;
                my.pixel_mut::<f32>(y, x).unwrap()[0] = y as f32;
            }
        }
        let r = remap(&m, &mx, &my, Interpolation::Linear, BorderType::Constant, Scalar::default())
            .unwrap();
        assert_eq!(r.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn remap_can_flip_horizontally() {
        let m = Mat::from_vec(1, 4, 1, &[10u8, 20, 30, 40]).unwrap();
        let mut mx = Mat::zeros(1, 4, CV_32FC1).unwrap();
        let my = Mat::zeros(1, 4, CV_32FC1).unwrap();
        for x in 0..4 {
            mx.pixel_mut::<f32>(0, x).unwrap()[0] = (3 - x) as f32;
        }
        let r = remap(&m, &mx, &my, Interpolation::Nearest, BorderType::Replicate, Scalar::default())
            .unwrap();
        assert_eq!(r.to_vec::<u8>().unwrap(), vec![40, 30, 20, 10]);
    }

    #[test]
    fn remap_rejects_mismatched_maps() {
        let m = ramp(4, 4);
        let mx = Mat::zeros(4, 4, CV_32FC1).unwrap();
        let my = Mat::zeros(3, 4, CV_32FC1).unwrap();
        assert_eq!(
            remap(&m, &mx, &my, Interpolation::Linear, BorderType::Constant, Scalar::default())
                .unwrap_err()
                .code,
            Code::StsUnmatchedSizes
        );
    }

    #[test]
    fn subpixel_coordinates_are_quantised_to_a_thirty_second() {
        // Two coordinates within the same 1/32 cell must sample identically.
        let (a, fa) = quantise(3.0 + 1.0 / 64.0);
        let (b, fb) = quantise(3.0 + 1.4 / 64.0);
        assert_eq!((a, fa), (b, fb));
        // And a full step apart must differ.
        let (_, fc) = quantise(3.0 + 3.0 / 32.0);
        assert_ne!(fa, fc);
    }
}
