//! Element-wise transcendental and polar/Cartesian conversions
//! (`modules/core/src/mathfuncs.cpp`).
//!
//! Upstream restricts these to `CV_32F` and `CV_64F`, and so do we: the integer
//! overloads do not exist in the C++ API either.

use crate::error::{Code, Result};
use crate::mat::{Mat, MatLike, MatLikeMut};
use crate::types::Depth;
use crate::{cv_assert, cv_error};

/// Apply a scalar function element-wise, preserving depth.
fn map1<S: MatLike + ?Sized>(src: &S, f: impl Fn(f64) -> f64) -> Result<Mat> {
    require_float(src.depth(), "this function")?;
    let mut dst = Mat::zeros(src.rows(), src.cols(), src.mat_type())?;
    for y in 0..src.rows() {
        match src.depth() {
            Depth::F32 => {
                let row: Vec<f32> = src.row_as::<f32>(y)?.to_vec();
                let d = dst.row_as_mut::<f32>(y)?;
                for (o, &v) in d.iter_mut().zip(row.iter()) {
                    *o = f(v as f64) as f32;
                }
            }
            _ => {
                let row: Vec<f64> = src.row_as::<f64>(y)?.to_vec();
                let d = dst.row_as_mut::<f64>(y)?;
                for (o, &v) in d.iter_mut().zip(row.iter()) {
                    *o = f(v);
                }
            }
        }
    }
    Ok(dst)
}

/// Combine two same-typed matrices element-wise.
fn map2<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    a: &A,
    b: &B,
    f: impl Fn(f64, f64) -> f64,
) -> Result<Mat> {
    require_float(a.depth(), "this function")?;
    if a.size() != b.size() || a.mat_type() != b.mat_type() {
        return cv_error!(Code::StsUnmatchedSizes, "operands must match in size and type");
    }
    let mut dst = Mat::zeros(a.rows(), a.cols(), a.mat_type())?;
    for y in 0..a.rows() {
        match a.depth() {
            Depth::F32 => {
                let (x, z) = (a.row_as::<f32>(y)?.to_vec(), b.row_as::<f32>(y)?.to_vec());
                let d = dst.row_as_mut::<f32>(y)?;
                for i in 0..d.len() {
                    d[i] = f(x[i] as f64, z[i] as f64) as f32;
                }
            }
            _ => {
                let (x, z) = (a.row_as::<f64>(y)?.to_vec(), b.row_as::<f64>(y)?.to_vec());
                let d = dst.row_as_mut::<f64>(y)?;
                for i in 0..d.len() {
                    d[i] = f(x[i], z[i]);
                }
            }
        }
    }
    Ok(dst)
}

/// Reject integer depths for float-only entry points.
fn require_float(depth: Depth, what: &str) -> Result<()> {
    if !matches!(depth, Depth::F32 | Depth::F64) {
        return cv_error!(
            Code::StsUnsupportedFormat,
            "{what} requires CV_32F or CV_64F, got {}",
            depth.name()
        );
    }
    Ok(())
}

/// `cv::sqrt(src, dst)`.
pub fn sqrt<S: MatLike + ?Sized>(src: &S) -> Result<Mat> {
    map1(src, f64::sqrt)
}

/// `cv::exp(src, dst)`.
pub fn exp<S: MatLike + ?Sized>(src: &S) -> Result<Mat> {
    map1(src, f64::exp)
}

/// `cv::log(src, dst)`.
///
/// Upstream returns a large negative constant rather than `-inf` for zero
/// input; this port returns the IEEE result, which is `-inf`.
pub fn log<S: MatLike + ?Sized>(src: &S) -> Result<Mat> {
    map1(src, f64::ln)
}

/// `cv::pow(src, power, dst)`.
///
/// For a negative base with a non-integral exponent upstream computes
/// `|src|^power`; that behaviour is reproduced here.
pub fn pow<S: MatLike + ?Sized>(src: &S, power: f64) -> Result<Mat> {
    let integral = power == power.trunc();
    map1(src, move |v| if integral { v.powf(power) } else { v.abs().powf(power) })
}

/// `cv::magnitude(x, y, dst)` — `sqrt(x^2 + y^2)`.
pub fn magnitude<A: MatLike + ?Sized, B: MatLike + ?Sized>(x: &A, y: &B) -> Result<Mat> {
    map2(x, y, |a, b| (a * a + b * b).sqrt())
}

/// Degrees-per-radian coefficients of upstream's odd-order atan polynomial.
const ATAN2_P1: f32 = 0.999_787_84_f32 * (180.0 / core::f32::consts::PI);
const ATAN2_P3: f32 = -0.325_808_4_f32 * (180.0 / core::f32::consts::PI);
const ATAN2_P5: f32 = 0.155_578_65_f32 * (180.0 / core::f32::consts::PI);
const ATAN2_P7: f32 = -0.044_326_555_f32 * (180.0 / core::f32::consts::PI);

/// `cv::fastAtan2(y, x)` — the angle of `(x, y)` in degrees, in `[0, 360)`.
///
/// A transcription of upstream's `atan_f32`, not a call to [`f64::atan2`]. The
/// difference is deliberate and it is the whole point: upstream evaluates a
/// degree-7 odd polynomial in `f32` on the ratio of the smaller magnitude to
/// the larger, so its answer is accurate to about 0.01 degrees and no better.
/// A correctly-rounded `atan2` is *more* accurate but disagrees with OpenCV on
/// essentially every pixel, so anything comparing against OpenCV's output — a
/// gradient histogram, a HOG descriptor, a stored feature vector — needs this
/// approximation rather than the true value.
///
/// The `DBL_EPSILON` added to the denominator is upstream's guard against
/// `0/0`; it is `f64::EPSILON` narrowed to `f32`, not `f32::EPSILON`.
#[must_use]
pub fn fast_atan2(y: f32, x: f32) -> f32 {
    const EPS: f32 = f64::EPSILON as f32;
    let (ax, ay) = (x.abs(), y.abs());
    let mut a = if ax >= ay {
        let c = ay / (ax + EPS);
        let c2 = c * c;
        (((ATAN2_P7 * c2 + ATAN2_P5) * c2 + ATAN2_P3) * c2 + ATAN2_P1) * c
    } else {
        let c = ax / (ay + EPS);
        let c2 = c * c;
        90.0 - (((ATAN2_P7 * c2 + ATAN2_P5) * c2 + ATAN2_P3) * c2 + ATAN2_P1) * c
    };
    if x < 0.0 {
        a = 180.0 - a;
    }
    if y < 0.0 {
        a = 360.0 - a;
    }
    a
}

/// `cv::phase(x, y, dst, angleInDegrees)` — `atan2(y, x)` mapped to `[0, 2pi)`.
///
/// Uses [`fast_atan2`], matching upstream bit for bit; see its documentation
/// for why the more accurate `atan2` is the wrong answer here. Upstream
/// narrows `CV_64F` operands to `f32` before the call and scales radians by
/// `(float)(CV_PI/180)`, both of which are reproduced.
pub fn phase<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    x: &A,
    y: &B,
    angle_in_degrees: bool,
) -> Result<Mat> {
    let scale: f32 = if angle_in_degrees { 1.0 } else { (core::f64::consts::PI / 180.0) as f32 };
    map2(x, y, move |a, b| (fast_atan2(b as f32, a as f32) * scale) as f64)
}

/// `cv::cartToPolar(x, y, magnitude, angle, angleInDegrees)`.
pub fn cart_to_polar<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    x: &A,
    y: &B,
    angle_in_degrees: bool,
) -> Result<(Mat, Mat)> {
    Ok((magnitude(x, y)?, phase(x, y, angle_in_degrees)?))
}

/// Upstream's quarter-degree sine table: `sin(2*pi*i/64)` for `i` in `0..64`.
///
/// Held to `f64` precision because `SinCos_32f` interpolates in `f64` and only
/// narrows on store. Entry 32 is `1.2246e-16` rather than zero — that is
/// `sin(pi)` as actually computed, and keeping the error reproduces upstream.
#[rustfmt::skip]
// The 1/sqrt(2) entries are upstream`s literals; they must not be swapped for
// f64::consts::FRAC_1_SQRT_2, which differs in the last place.
#[allow(clippy::approx_constant)]
const SIN_TABLE: [f64; 64] = [
    0.0, 0.0980171403295606,
    0.19509032201612825, 0.29028467725446233,
    0.3826834323650898, 0.47139673682599764,
    0.5555702330196022, 0.6343932841636455,
    0.7071067811865475, 0.773010453362737,
    0.8314696123025452, 0.8819212643483549,
    0.9238795325112867, 0.9569403357322089,
    0.9807852804032304, 0.9951847266721968,
    1.0, 0.9951847266721969,
    0.9807852804032304, 0.9569403357322089,
    0.9238795325112867, 0.881921264348355,
    0.8314696123025455, 0.7730104533627371,
    0.7071067811865476, 0.6343932841636455,
    0.5555702330196022, 0.47139673682599786,
    0.3826834323650899, 0.2902846772544624,
    0.1950903220161286, 0.09801714032956083,
    1.2246e-16, -0.09801714032956059,
    -0.19509032201612836, -0.2902846772544621,
    -0.38268343236508967, -0.47139673682599764,
    -0.555570233019602, -0.6343932841636453,
    -0.7071067811865475, -0.7730104533627367,
    -0.8314696123025452, -0.8819212643483549,
    -0.9238795325112865, -0.9569403357322088,
    -0.9807852804032303, -0.9951847266721969,
    -1.0, -0.9951847266721969,
    -0.9807852804032304, -0.9569403357322089,
    -0.9238795325112866, -0.881921264348355,
    -0.8314696123025455, -0.7730104533627369,
    -0.7071067811865477, -0.6343932841636459,
    -0.5555702330196022, -0.4713967368259979,
    -0.3826834323650904, -0.2902846772544625,
    -0.19509032201612872, -0.0980171403295605,
];

/// One entry of upstream's `SinCos_32f`: `(cos(a) * m, sin(a) * m)`.
///
/// Table lookup on the nearest 1/64 turn plus a short Taylor correction, all in
/// `f64`, narrowed to `f32` on store. Calling `sin`/`cos` directly is more
/// accurate but disagrees with OpenCV; see [`fast_atan2`] for the same argument.
fn sin_cos(mag: f64, angle: f64, angle_in_degrees: bool) -> (f64, f64) {
    const N: i32 = 64;
    const K2: f64 = (2.0 * core::f64::consts::PI) / N as f64;
    const SIN_A0: f64 = -0.166_630_293_345_647 * K2 * K2 * K2;
    const COS_A0: f64 = -0.499_818_138_450_326 * K2 * K2;

    let k1 = if angle_in_degrees {
        N as f64 / 360.0
    } else {
        N as f64 / (2.0 * core::f64::consts::PI)
    };
    let mut t = angle * k1;
    let it = crate::saturate::cv_round(t);
    t -= it as f64;
    let sin_idx = (it & (N - 1)) as usize;
    let cos_idx = ((N / 4 - sin_idx as i32) & (N - 1)) as usize;

    let sin_b = (SIN_A0 * t * t + K2) * t;
    let cos_b = COS_A0 * t * t + 1.0;
    let (sin_a, cos_a) = (SIN_TABLE[sin_idx], SIN_TABLE[cos_idx]);

    let sin_val = (sin_a * cos_b + cos_a * sin_b) * mag;
    let cos_val = (cos_a * cos_b - sin_a * sin_b) * mag;
    (cos_val, sin_val)
}

/// `cv::polarToCart(magnitude, angle, x, y, angleInDegrees)`.
///
/// Both the `CV_32F` and `CV_64F` entry points funnel through upstream's
/// `SinCos_32f`, which stores through `float`; the 64-bit path buffers into
/// `float` first, so it is no more accurate than the 32-bit one.
pub fn polar_to_cart<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    mag: &A,
    angle: &B,
    angle_in_degrees: bool,
) -> Result<(Mat, Mat)> {
    let x = map2(mag, angle, move |m, a| sin_cos(m, a, angle_in_degrees).0 as f32 as f64)?;
    let y = map2(mag, angle, move |m, a| sin_cos(m, a, angle_in_degrees).1 as f32 as f64)?;
    Ok((x, y))
}

/// `cv::LUT(src, lut, dst)`.
///
/// `src` must be `CV_8U` or `CV_8S`; `lut` must hold 256 entries with either
/// one channel or the same channel count as `src`.
pub fn lut<S: MatLike + ?Sized, L: MatLike + ?Sized>(src: &S, table: &L) -> Result<Mat> {
    cv_assert!(
        matches!(src.depth(), Depth::U8 | Depth::S8),
        "LUT requires an 8-bit source, got {}",
        src.depth().name()
    );
    let entries = table.total();
    cv_assert!(entries == 256, "LUT table must have 256 entries, got {entries}");
    let scn = src.channels();
    let lcn = table.channels();
    cv_assert!(
        lcn == 1 || lcn == scn,
        "LUT table must have 1 or {scn} channels, got {lcn}"
    );

    let dst_type = table.mat_type().with_channels(scn);
    let mut dst = Mat::zeros(src.rows(), src.cols(), dst_type)?;

    // Flatten the table to (entry, channel) lookups once.
    let mut flat = vec![0.0f64; 256 * lcn as usize];
    let mut k = 0usize;
    for ty in 0..table.rows() {
        for tx in 0..table.cols() {
            for c in 0..lcn {
                flat[k] = table.at_f64(ty, tx, c)?;
                k += 1;
            }
        }
    }

    for y in 0..src.rows() {
        for x in 0..src.cols() {
            for c in 0..scn {
                // CV_8S indexes the table from -128 as entry 0, as upstream does.
                let raw = src.at_f64(y, x, c)?;
                let idx = if src.depth() == Depth::S8 { raw as i32 + 128 } else { raw as i32 };
                let tc = if lcn == 1 { 0 } else { c };
                let v = flat[idx as usize * lcn as usize + tc as usize];
                crate::arithm::set_channel(&mut dst, y, x, c, v)?;
            }
        }
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CV_32FC1, CV_8UC1};

    fn f32m(v: &[f32]) -> Mat {
        Mat::from_vec(1, v.len() as i32, 1, v).unwrap()
    }

    #[test]
    fn sqrt_exp_log_are_elementwise() {
        let m = f32m(&[1.0, 4.0, 9.0, 16.0]);
        assert_eq!(sqrt(&m).unwrap().to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        let e = exp(&f32m(&[0.0, 1.0])).unwrap().to_vec::<f32>().unwrap();
        assert!((e[0] - 1.0).abs() < 1e-6);
        assert!((e[1] - std::f32::consts::E).abs() < 1e-5);

        let l = log(&f32m(&[1.0, std::f32::consts::E])).unwrap().to_vec::<f32>().unwrap();
        assert!(l[0].abs() < 1e-6);
        assert!((l[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pow_uses_absolute_base_for_fractional_exponents() {
        // Upstream computes |src|^p when p is not an integer.
        let m = f32m(&[-4.0, 4.0]);
        let p = pow(&m, 0.5).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(p, vec![2.0, 2.0]);

        // Integral exponents keep the sign.
        let c = pow(&m, 3.0).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(c, vec![-64.0, 64.0]);
    }

    #[test]
    fn magnitude_is_the_hypotenuse() {
        let x = f32m(&[3.0, 5.0]);
        let y = f32m(&[4.0, 12.0]);
        assert_eq!(magnitude(&x, &y).unwrap().to_vec::<f32>().unwrap(), vec![5.0, 13.0]);
    }

    #[test]
    fn phase_is_never_negative() {
        // atan2 would give -90 for (0, -1); upstream reports 270.
        let x = f32m(&[1.0, 0.0, -1.0, 0.0]);
        let y = f32m(&[0.0, 1.0, 0.0, -1.0]);
        let deg = phase(&x, &y, true).unwrap().to_vec::<f32>().unwrap();
        assert!((deg[0] - 0.0).abs() < 1e-3, "{}", deg[0]);
        assert!((deg[1] - 90.0).abs() < 1e-3, "{}", deg[1]);
        assert!((deg[2] - 180.0).abs() < 1e-3, "{}", deg[2]);
        assert!((deg[3] - 270.0).abs() < 1e-3, "{}", deg[3]);
        assert!(deg.iter().all(|&a| (0.0..360.0).contains(&a)));
    }

    #[test]
    fn polar_roundtrip_recovers_cartesian() {
        let x = f32m(&[3.0, -1.0, 0.0]);
        let y = f32m(&[4.0, 2.0, -5.0]);
        let (m, a) = cart_to_polar(&x, &y, false).unwrap();
        let (x2, y2) = polar_to_cart(&m, &a, false).unwrap();
        // 5e-4, not 1e-7: both directions are upstream's approximations, so the
        // roundtrip cannot be tighter than they are. cv2 4.13.0 on this exact
        // input returns 2.9995692 for the first x and errs by 4.31e-4; this
        // port returns 2.9995687. Tightening the bound here would only mean
        // diverging from OpenCV.
        for (p, q) in x.to_vec::<f32>().unwrap().iter().zip(x2.to_vec::<f32>().unwrap().iter()) {
            assert!((p - q).abs() < 5e-4, "{p} vs {q}");
        }
        for (p, q) in y.to_vec::<f32>().unwrap().iter().zip(y2.to_vec::<f32>().unwrap().iter()) {
            assert!((p - q).abs() < 5e-4, "{p} vs {q}");
        }
    }

    #[test]
    fn integer_depths_are_rejected() {
        let m = Mat::zeros(1, 4, CV_8UC1).unwrap();
        assert_eq!(sqrt(&m).unwrap_err().code, Code::StsUnsupportedFormat);
    }

    #[test]
    fn lut_maps_through_a_256_entry_table() {
        let table: Vec<u8> = (0..256).map(|i| (255 - i) as u8).collect();
        let t = Mat::from_vec(1, 256, 1, &table).unwrap();
        let src = Mat::from_vec(1, 4, 1, &[0u8, 1, 128, 255]).unwrap();
        let out = lut(&src, &t).unwrap();
        assert_eq!(out.to_vec::<u8>().unwrap(), vec![255, 254, 127, 0]);
    }

    #[test]
    fn lut_can_change_depth() {
        let table: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let t = Mat::from_vec(1, 256, 1, &table).unwrap();
        let src = Mat::from_vec(1, 2, 1, &[0u8, 255]).unwrap();
        let out = lut(&src, &t).unwrap();
        assert_eq!(out.mat_type(), CV_32FC1);
        assert_eq!(out.to_vec::<f32>().unwrap(), vec![0.0, 1.0]);
    }

    #[test]
    fn lut_broadcasts_a_single_channel_table() {
        let table: Vec<u8> = (0..256).map(|i| (i / 2) as u8).collect();
        let t = Mat::from_vec(1, 256, 1, &table).unwrap();
        let src = Mat::from_vec(1, 1, 3, &[10u8, 100, 200]).unwrap();
        let out = lut(&src, &t).unwrap();
        assert_eq!(out.pixel::<u8>(0, 0).unwrap(), &[5, 50, 100]);
    }

    #[test]
    fn lut_rejects_a_wrong_sized_table() {
        let t = Mat::zeros(1, 128, CV_8UC1).unwrap();
        let src = Mat::zeros(1, 1, CV_8UC1).unwrap();
        assert!(lut(&src, &t).is_err());
    }
}
