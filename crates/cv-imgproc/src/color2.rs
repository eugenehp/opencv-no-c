//! The remaining colour spaces: XYZ, Lab, Luv, YUV, HLS and the packed 16-bit
//! formats (`modules/imgproc/src/color_lab.cpp`, `color_yuv.cpp`,
//! `color_hsv.cpp`, `color_rgb.cpp`).
//!
//! # Why these are separate from [`crate::color`]
//!
//! The conversions there are exact integer transforms with fixed-point
//! constants transcribed from upstream. These are the *device-independent*
//! spaces, which go through a linear-RGB stage and a white point, so they are
//! floating-point by nature and share a different set of helpers.

use cv_core::types::{Depth, MatType};
use cv_core::{cv_assert, Mat, MatLike, Result};

/// D65 white point, upstream's reference for both Lab and Luv.
const XN: f64 = 0.950_456;
const YN: f64 = 1.0;
const ZN: f64 = 1.088_754;

/// sRGB gamma expansion — the step that makes XYZ *linear* in light.
///
/// Skipping it is the usual mistake: the result still looks like a colour
/// space and every value is wrong, most visibly in the shadows.
fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The inverse of [`srgb_to_linear`].
fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Linear RGB in `[0, 1]` to CIE XYZ, with upstream's matrix.
fn rgb_to_xyz(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    (
        0.412_453 * r + 0.357_580 * g + 0.180_423 * b,
        0.212_671 * r + 0.715_160 * g + 0.072_169 * b,
        0.019_334 * r + 0.119_193 * g + 0.950_227 * b,
    )
}

/// CIE XYZ back to linear RGB.
fn xyz_to_rgb(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    (
        3.240_479 * x - 1.537_150 * y - 0.498_535 * z,
        -0.969_256 * x + 1.875_991 * y + 0.041_556 * z,
        0.055_648 * x - 0.204_043 * y + 1.057_311 * z,
    )
}

/// The cube-root-with-linear-toe used by L*a*b*.
fn lab_f(t: f64) -> f64 {
    if t > 0.008_856 {
        t.cbrt()
    } else {
        7.787 * t + 16.0 / 116.0
    }
}

fn lab_f_inv(t: f64) -> f64 {
    if t * t * t > 0.008_856 {
        t * t * t
    } else {
        (t - 16.0 / 116.0) / 7.787
    }
}

/// Which conversion [`cvt_color2`] performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorConversion2 {
    /// `cv::COLOR_BGR2XYZ`
    Bgr2Xyz,
    /// `cv::COLOR_XYZ2BGR`
    Xyz2Bgr,
    /// `cv::COLOR_RGB2XYZ`
    Rgb2Xyz,
    /// `cv::COLOR_XYZ2RGB`
    Xyz2Rgb,
    /// `cv::COLOR_BGR2Lab`
    Bgr2Lab,
    /// `cv::COLOR_Lab2BGR`
    Lab2Bgr,
    /// `cv::COLOR_RGB2Lab`
    Rgb2Lab,
    /// `cv::COLOR_Lab2RGB`
    Lab2Rgb,
    /// `cv::COLOR_BGR2Luv`
    Bgr2Luv,
    /// `cv::COLOR_Luv2BGR`
    Luv2Bgr,
    /// `cv::COLOR_RGB2Luv`
    Rgb2Luv,
    /// `cv::COLOR_Luv2RGB`
    Luv2Rgb,
    /// `cv::COLOR_LBGR2Lab` — from **linear** BGR, skipping the sRGB gamma.
    LinearBgr2Lab,
    /// `cv::COLOR_LRGB2Lab`
    LinearRgb2Lab,
    /// `cv::COLOR_Lab2LBGR`
    Lab2LinearBgr,
    /// `cv::COLOR_Lab2LRGB`
    Lab2LinearRgb,
    /// `cv::COLOR_LBGR2Luv`
    LinearBgr2Luv,
    /// `cv::COLOR_LRGB2Luv`
    LinearRgb2Luv,
    /// `cv::COLOR_Luv2LBGR`
    Luv2LinearBgr,
    /// `cv::COLOR_Luv2LRGB`
    Luv2LinearRgb,
    /// `cv::COLOR_BGR2YUV`
    Bgr2Yuv,
    /// `cv::COLOR_YUV2BGR`
    Yuv2Bgr,
    /// `cv::COLOR_RGB2YUV`
    Rgb2Yuv,
    /// `cv::COLOR_YUV2RGB`
    Yuv2Rgb,
    /// `cv::COLOR_BGR2HLS`
    Bgr2Hls,
    /// `cv::COLOR_HLS2BGR`
    Hls2Bgr,
    /// `cv::COLOR_RGB2HLS`
    Rgb2Hls,
    /// `cv::COLOR_HLS2RGB`
    Hls2Rgb,
    /// `cv::COLOR_BGR2BGR565`
    Bgr2Bgr565,
    /// `cv::COLOR_BGR5652BGR`
    Bgr5652Bgr,
    /// `cv::COLOR_BGR2BGR555`
    Bgr2Bgr555,
    /// `cv::COLOR_BGR5552BGR`
    Bgr5552Bgr,
    /// `cv::COLOR_RGBA2mRGBA` — premultiply by alpha.
    Rgba2mRgba,
    /// `cv::COLOR_mRGBA2RGBA` — undo the premultiplication.
    MRgba2Rgba,
}

impl ColorConversion2 {
    /// Is the RGB side of this conversion **already linear**?
    ///
    /// The `L`-prefixed codes say the caller's RGB has had the sRGB transfer
    /// curve removed already, so applying it again would darken the shadows by
    /// a factor of twelve. That is the entire difference between
    /// `COLOR_BGR2Lab` and `COLOR_LBGR2Lab`.
    fn linear_rgb(self) -> bool {
        matches!(
            self,
            ColorConversion2::LinearBgr2Lab
                | ColorConversion2::LinearRgb2Lab
                | ColorConversion2::Lab2LinearBgr
                | ColorConversion2::Lab2LinearRgb
                | ColorConversion2::LinearBgr2Luv
                | ColorConversion2::LinearRgb2Luv
                | ColorConversion2::Luv2LinearBgr
                | ColorConversion2::Luv2LinearRgb
        )
    }

    /// Does the RGB side of this conversion put blue first?
    fn blue_first(self) -> bool {
        matches!(
            self,
            ColorConversion2::Bgr2Lab
                | ColorConversion2::Lab2Bgr
                | ColorConversion2::Bgr2Luv
                | ColorConversion2::Luv2Bgr
                | ColorConversion2::LinearBgr2Lab
                | ColorConversion2::Lab2LinearBgr
                | ColorConversion2::LinearBgr2Luv
                | ColorConversion2::Luv2LinearBgr
        )
    }

    /// Channels the source must have.
    fn src_channels(self) -> i32 {
        match self {
            ColorConversion2::Bgr5652Bgr | ColorConversion2::Bgr5552Bgr => 2,
            ColorConversion2::Rgba2mRgba | ColorConversion2::MRgba2Rgba => 4,
            _ => 3,
        }
    }

    /// Channels the destination will have.
    fn dst_channels(self) -> i32 {
        match self {
            ColorConversion2::Bgr2Bgr565 | ColorConversion2::Bgr2Bgr555 => 2,
            ColorConversion2::Rgba2mRgba | ColorConversion2::MRgba2Rgba => 4,
            _ => 3,
        }
    }
}

/// `cv::cvtColor(src, dst, code)` for the spaces in [`ColorConversion2`].
///
/// # Ranges
///
/// For `CV_8U` the outputs use upstream's packed ranges, not the natural ones:
/// `L` is scaled by 255/100, and `a`/`b`/`u`/`v` are offset by 128 so they fit
/// unsigned. A caller reading Lab out of an 8-bit image and expecting
/// `L` in 0..100 gets a silently wrong answer, which is why it is stated here.
/// Float images carry the natural ranges.
pub fn cvt_color2<S: MatLike + ?Sized>(src: &S, code: ColorConversion2) -> Result<Mat> {
    let want = code.src_channels();
    cv_assert!(
        src.channels() == want,
        "{code:?} needs a {want}-channel source, got {}",
        src.channels()
    );
    let (h, w) = (src.rows(), src.cols());
    let dcn = code.dst_channels();
    let eight_bit = src.depth() == Depth::U8;
    // The packed formats are always 8-bit pairs; everything else keeps depth.
    let dst_depth = match code {
        ColorConversion2::Bgr2Bgr565 | ColorConversion2::Bgr2Bgr555 => Depth::U8,
        _ => src.depth(),
    };
    let mut out = Mat::zeros(h, w, MatType::new(dst_depth, dcn))?;

    // Scale that maps the source range onto 0..1 for the linear stages.
    let scale = if eight_bit { 1.0 / 255.0 } else { 1.0 };

    for y in 0..h {
        for x in 0..w {
            let c0 = src.at_f64(y, x, 0)?;
            let c1 = if want > 1 { src.at_f64(y, x, 1)? } else { 0.0 };
            let c2 = if want > 2 { src.at_f64(y, x, 2)? } else { 0.0 };
            let c3 = if want > 3 { src.at_f64(y, x, 3)? } else { 0.0 };

            let vals: Vec<f64> = match code {
                ColorConversion2::Bgr2Xyz | ColorConversion2::Rgb2Xyz => {
                    let (r, g, b) = if code == ColorConversion2::Bgr2Xyz {
                        (c2 * scale, c1 * scale, c0 * scale)
                    } else {
                        (c0 * scale, c1 * scale, c2 * scale)
                    };
                    let (xx, yy, zz) = rgb_to_xyz(r, g, b);
                    let k = if eight_bit { 255.0 } else { 1.0 };
                    vec![xx * k, yy * k, zz * k]
                }
                ColorConversion2::Xyz2Bgr | ColorConversion2::Xyz2Rgb => {
                    let (xx, yy, zz) = (c0 * scale, c1 * scale, c2 * scale);
                    let (r, g, b) = xyz_to_rgb(xx, yy, zz);
                    let k = if eight_bit { 255.0 } else { 1.0 };
                    if code == ColorConversion2::Xyz2Bgr {
                        vec![b * k, g * k, r * k]
                    } else {
                        vec![r * k, g * k, b * k]
                    }
                }
                ColorConversion2::Bgr2Lab
                | ColorConversion2::Rgb2Lab
                | ColorConversion2::LinearBgr2Lab
                | ColorConversion2::LinearRgb2Lab => {
                    let (r, g, b) = if code.blue_first() {
                        (c2 * scale, c1 * scale, c0 * scale)
                    } else {
                        (c0 * scale, c1 * scale, c2 * scale)
                    };
                    let lin = |v: f64| if code.linear_rgb() { v } else { srgb_to_linear(v) };
                    let (xx, yy, zz) = rgb_to_xyz(lin(r), lin(g), lin(b));
                    let (fx, fy, fz) = (lab_f(xx / XN), lab_f(yy / YN), lab_f(zz / ZN));
                    let l = 116.0 * fy - 16.0;
                    let a = 500.0 * (fx - fy);
                    let bb = 200.0 * (fy - fz);
                    if eight_bit {
                        vec![l * 255.0 / 100.0, a + 128.0, bb + 128.0]
                    } else {
                        vec![l, a, bb]
                    }
                }
                ColorConversion2::Lab2Bgr
                | ColorConversion2::Lab2Rgb
                | ColorConversion2::Lab2LinearBgr
                | ColorConversion2::Lab2LinearRgb => {
                    let (l, a, bb) = if eight_bit {
                        (c0 * 100.0 / 255.0, c1 - 128.0, c2 - 128.0)
                    } else {
                        (c0, c1, c2)
                    };
                    let fy = (l + 16.0) / 116.0;
                    let fx = fy + a / 500.0;
                    let fz = fy - bb / 200.0;
                    let (xx, yy, zz) =
                        (lab_f_inv(fx) * XN, lab_f_inv(fy) * YN, lab_f_inv(fz) * ZN);
                    let (r, g, b) = xyz_to_rgb(xx, yy, zz);
                    let enc = |v: f64| if code.linear_rgb() { v } else { linear_to_srgb(v) };
                    let (r, g, b) = (enc(r), enc(g), enc(b));
                    let k = if eight_bit { 255.0 } else { 1.0 };
                    if code.blue_first() {
                        vec![b * k, g * k, r * k]
                    } else {
                        vec![r * k, g * k, b * k]
                    }
                }
                ColorConversion2::Bgr2Luv
                | ColorConversion2::Rgb2Luv
                | ColorConversion2::LinearBgr2Luv
                | ColorConversion2::LinearRgb2Luv => {
                    let (r, g, b) = if code.blue_first() {
                        (c2 * scale, c1 * scale, c0 * scale)
                    } else {
                        (c0 * scale, c1 * scale, c2 * scale)
                    };
                    let lin = |v: f64| if code.linear_rgb() { v } else { srgb_to_linear(v) };
                    let (xx, yy, zz) = rgb_to_xyz(lin(r), lin(g), lin(b));
                    let d = xx + 15.0 * yy + 3.0 * zz;
                    let (up, vp) = if d.abs() < 1e-12 {
                        (0.0, 0.0)
                    } else {
                        (4.0 * xx / d, 9.0 * yy / d)
                    };
                    let dn = XN + 15.0 * YN + 3.0 * ZN;
                    let (unp, vnp) = (4.0 * XN / dn, 9.0 * YN / dn);
                    let yr = yy / YN;
                    let l = if yr > 0.008_856 {
                        116.0 * yr.cbrt() - 16.0
                    } else {
                        903.3 * yr
                    };
                    let u = 13.0 * l * (up - unp);
                    let v = 13.0 * l * (vp - vnp);
                    if eight_bit {
                        // Upstream's 8-bit packing for Luv.
                        vec![l * 255.0 / 100.0, (u + 134.0) * 255.0 / 354.0, (v + 140.0) * 255.0 / 262.0]
                    } else {
                        vec![l, u, v]
                    }
                }
                ColorConversion2::Luv2Bgr
                | ColorConversion2::Luv2Rgb
                | ColorConversion2::Luv2LinearBgr
                | ColorConversion2::Luv2LinearRgb => {
                    let (l, u, v) = if eight_bit {
                        (
                            c0 * 100.0 / 255.0,
                            c1 * 354.0 / 255.0 - 134.0,
                            c2 * 262.0 / 255.0 - 140.0,
                        )
                    } else {
                        (c0, c1, c2)
                    };
                    let dn = XN + 15.0 * YN + 3.0 * ZN;
                    let (unp, vnp) = (4.0 * XN / dn, 9.0 * YN / dn);
                    let yy = if l > 8.0 {
                        YN * ((l + 16.0) / 116.0).powi(3)
                    } else {
                        YN * l / 903.3
                    };
                    let (up, vp) = if l.abs() < 1e-9 {
                        (unp, vnp)
                    } else {
                        (u / (13.0 * l) + unp, v / (13.0 * l) + vnp)
                    };
                    let (xx, zz) = if vp.abs() < 1e-12 {
                        (0.0, 0.0)
                    } else {
                        let x = 9.0 * yy * up / (4.0 * vp);
                        let z = (9.0 * yy / vp - x - 15.0 * yy) / 3.0;
                        (x, z)
                    };
                    let (r, g, b) = xyz_to_rgb(xx, yy, zz);
                    let enc = |v: f64| if code.linear_rgb() { v } else { linear_to_srgb(v) };
                    let (r, g, b) = (enc(r), enc(g), enc(b));
                    let k = if eight_bit { 255.0 } else { 1.0 };
                    if code.blue_first() {
                        vec![b * k, g * k, r * k]
                    } else {
                        vec![r * k, g * k, b * k]
                    }
                }
                ColorConversion2::Bgr2Yuv | ColorConversion2::Rgb2Yuv => {
                    let (r, g, b) = if code == ColorConversion2::Bgr2Yuv {
                        (c2, c1, c0)
                    } else {
                        (c0, c1, c2)
                    };
                    let yv = 0.299 * r + 0.587 * g + 0.114 * b;
                    let delta = if eight_bit { 128.0 } else { 0.5 };
                    vec![yv, 0.492 * (b - yv) + delta, 0.877 * (r - yv) + delta]
                }
                ColorConversion2::Yuv2Bgr | ColorConversion2::Yuv2Rgb => {
                    let delta = if eight_bit { 128.0 } else { 0.5 };
                    let (yv, u, v) = (c0, c1 - delta, c2 - delta);
                    let r = yv + 1.140 * v;
                    let g = yv - 0.394 * u - 0.581 * v;
                    let b = yv + 2.032 * u;
                    if code == ColorConversion2::Yuv2Bgr {
                        vec![b, g, r]
                    } else {
                        vec![r, g, b]
                    }
                }
                ColorConversion2::Bgr2Hls | ColorConversion2::Rgb2Hls => {
                    let (r, g, b) = if code == ColorConversion2::Bgr2Hls {
                        (c2 * scale, c1 * scale, c0 * scale)
                    } else {
                        (c0 * scale, c1 * scale, c2 * scale)
                    };
                    let mx = r.max(g).max(b);
                    let mn = r.min(g).min(b);
                    let l = (mx + mn) * 0.5;
                    let d = mx - mn;
                    let (hh, ss) = if d.abs() < 1e-12 {
                        (0.0, 0.0)
                    } else {
                        let s = if l < 0.5 { d / (mx + mn) } else { d / (2.0 - mx - mn) };
                        let h = if mx == r {
                            60.0 * (g - b) / d
                        } else if mx == g {
                            120.0 + 60.0 * (b - r) / d
                        } else {
                            240.0 + 60.0 * (r - g) / d
                        };
                        (if h < 0.0 { h + 360.0 } else { h }, s)
                    };
                    if eight_bit {
                        // 8-bit hue is halved so it fits, as upstream does.
                        vec![hh / 2.0, l * 255.0, ss * 255.0]
                    } else {
                        vec![hh, l, ss]
                    }
                }
                ColorConversion2::Hls2Bgr | ColorConversion2::Hls2Rgb => {
                    let (hh, l, s) = if eight_bit {
                        (c0 * 2.0, c1 / 255.0, c2 / 255.0)
                    } else {
                        (c0, c1, c2)
                    };
                    let (r, g, b) = if s.abs() < 1e-12 {
                        (l, l, l)
                    } else {
                        let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
                        let p = 2.0 * l - q;
                        let hk = hh / 360.0;
                        let f = |mut t: f64| -> f64 {
                            if t < 0.0 {
                                t += 1.0;
                            }
                            if t > 1.0 {
                                t -= 1.0;
                            }
                            if t < 1.0 / 6.0 {
                                p + (q - p) * 6.0 * t
                            } else if t < 0.5 {
                                q
                            } else if t < 2.0 / 3.0 {
                                p + (q - p) * (2.0 / 3.0 - t) * 6.0
                            } else {
                                p
                            }
                        };
                        (f(hk + 1.0 / 3.0), f(hk), f(hk - 1.0 / 3.0))
                    };
                    let k = if eight_bit { 255.0 } else { 1.0 };
                    if code == ColorConversion2::Hls2Bgr {
                        vec![b * k, g * k, r * k]
                    } else {
                        vec![r * k, g * k, b * k]
                    }
                }
                ColorConversion2::Bgr2Bgr565 => {
                    let (b, g, r) = (c0 as u32, c1 as u32, c2 as u32);
                    let p = ((b >> 3) & 0x1f) | (((g >> 2) & 0x3f) << 5) | (((r >> 3) & 0x1f) << 11);
                    vec![f64::from(p & 0xff), f64::from(p >> 8)]
                }
                ColorConversion2::Bgr5652Bgr => {
                    let p = (c0 as u32) | ((c1 as u32) << 8);
                    // Replicate the high bits into the low ones so full-scale
                    // input comes back as full scale rather than 248.
                    let b = ((p & 0x1f) * 255 + 15) / 31;
                    let g = (((p >> 5) & 0x3f) * 255 + 31) / 63;
                    let r = (((p >> 11) & 0x1f) * 255 + 15) / 31;
                    vec![f64::from(b), f64::from(g), f64::from(r)]
                }
                ColorConversion2::Bgr2Bgr555 => {
                    let (b, g, r) = (c0 as u32, c1 as u32, c2 as u32);
                    let p = ((b >> 3) & 0x1f) | (((g >> 3) & 0x1f) << 5) | (((r >> 3) & 0x1f) << 10);
                    vec![f64::from(p & 0xff), f64::from(p >> 8)]
                }
                ColorConversion2::Bgr5552Bgr => {
                    let p = (c0 as u32) | ((c1 as u32) << 8);
                    let b = ((p & 0x1f) * 255 + 15) / 31;
                    let g = (((p >> 5) & 0x1f) * 255 + 15) / 31;
                    let r = (((p >> 10) & 0x1f) * 255 + 15) / 31;
                    vec![f64::from(b), f64::from(g), f64::from(r)]
                }
                ColorConversion2::Rgba2mRgba => {
                    let a = c3;
                    let f = if eight_bit { a / 255.0 } else { a };
                    vec![c0 * f, c1 * f, c2 * f, a]
                }
                ColorConversion2::MRgba2Rgba => {
                    let a = c3;
                    let f = if eight_bit { a / 255.0 } else { a };
                    if f.abs() < 1e-9 {
                        vec![0.0, 0.0, 0.0, a]
                    } else {
                        vec![c0 / f, c1 / f, c2 / f, a]
                    }
                }
            };
            for (c, v) in vals.iter().enumerate() {
                cv_core::arithm::set_channel_generic(&mut out, y, x, c as i32, *v)?;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bgr8(b: u8, g: u8, r: u8) -> Mat {
        Mat::from_vec(1, 1, 3, &[b, g, r]).unwrap()
    }
    fn bgr32(b: f32, g: f32, r: f32) -> Mat {
        Mat::from_vec(1, 1, 3, &[b, g, r]).unwrap()
    }

    #[test]
    fn xyz_round_trips() {
        for (b, g, r) in [(0.1f32, 0.5, 0.9), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)] {
            let src = bgr32(b, g, r);
            let xyz = cvt_color2(&src, ColorConversion2::Bgr2Xyz).unwrap();
            let back = cvt_color2(&xyz, ColorConversion2::Xyz2Bgr).unwrap();
            for c in 0..3 {
                let a = src.at_f64(0, 0, c).unwrap();
                let z = back.at_f64(0, 0, c).unwrap();
                assert!((a - z).abs() < 1e-6, "channel {c}: {a} vs {z}");
            }
        }
    }

    #[test]
    fn white_maps_to_the_reference_white_point() {
        // Linear RGB (1,1,1) is the D65 white, so Y should be 1.
        let w = cvt_color2(&bgr32(1.0, 1.0, 1.0), ColorConversion2::Bgr2Xyz).unwrap();
        assert!((w.at_f64(0, 0, 1).unwrap() - 1.0).abs() < 1e-3, "Y = {}", w.at_f64(0, 0, 1).unwrap());
    }

    #[test]
    fn lab_of_white_is_lightness_100_and_neutral_chroma() {
        let l = cvt_color2(&bgr32(1.0, 1.0, 1.0), ColorConversion2::Bgr2Lab).unwrap();
        assert!((l.at_f64(0, 0, 0).unwrap() - 100.0).abs() < 0.5, "L = {}", l.at_f64(0, 0, 0).unwrap());
        assert!(l.at_f64(0, 0, 1).unwrap().abs() < 1.0, "a should be near 0");
        assert!(l.at_f64(0, 0, 2).unwrap().abs() < 1.0, "b should be near 0");
        // Black is L = 0.
        let k = cvt_color2(&bgr32(0.0, 0.0, 0.0), ColorConversion2::Bgr2Lab).unwrap();
        assert!(k.at_f64(0, 0, 0).unwrap().abs() < 1e-6);
    }

    #[test]
    fn lab_round_trips_in_float_and_eight_bit() {
        for (b, g, r) in [(0.2f32, 0.6, 0.4), (0.9, 0.1, 0.5)] {
            let src = bgr32(b, g, r);
            let lab = cvt_color2(&src, ColorConversion2::Bgr2Lab).unwrap();
            let back = cvt_color2(&lab, ColorConversion2::Lab2Bgr).unwrap();
            for c in 0..3 {
                let (a, z) = (src.at_f64(0, 0, c).unwrap(), back.at_f64(0, 0, c).unwrap());
                assert!((a - z).abs() < 1e-5, "float channel {c}: {a} vs {z}");
            }
        }
        for (b, g, r) in [(40u8, 160, 90), (255, 0, 128)] {
            let src = bgr8(b, g, r);
            let lab = cvt_color2(&src, ColorConversion2::Bgr2Lab).unwrap();
            let back = cvt_color2(&lab, ColorConversion2::Lab2Bgr).unwrap();
            for c in 0..3 {
                let (a, z) = (src.at_f64(0, 0, c).unwrap(), back.at_f64(0, 0, c).unwrap());
                assert!((a - z).abs() <= 2.0, "8-bit channel {c}: {a} vs {z}");
            }
        }
    }

    #[test]
    fn eight_bit_lab_uses_upstreams_packed_ranges() {
        // L scaled by 255/100 and a/b offset by 128 — not the natural ranges.
        let l = cvt_color2(&bgr8(255, 255, 255), ColorConversion2::Bgr2Lab).unwrap();
        assert!((l.at_f64(0, 0, 0).unwrap() - 255.0).abs() < 2.0, "white L should pack to 255");
        let grey = cvt_color2(&bgr8(128, 128, 128), ColorConversion2::Bgr2Lab).unwrap();
        assert!((grey.at_f64(0, 0, 1).unwrap() - 128.0).abs() < 2.0, "neutral a packs to 128");
    }

    #[test]
    fn luv_round_trips_and_white_is_neutral() {
        let w = cvt_color2(&bgr32(1.0, 1.0, 1.0), ColorConversion2::Bgr2Luv).unwrap();
        assert!((w.at_f64(0, 0, 0).unwrap() - 100.0).abs() < 0.5, "L = {}", w.at_f64(0, 0, 0).unwrap());
        assert!(w.at_f64(0, 0, 1).unwrap().abs() < 1.0, "u should be near 0 for white");
        for (b, g, r) in [(0.3f32, 0.7, 0.5), (0.05, 0.2, 0.8)] {
            let src = bgr32(b, g, r);
            let luv = cvt_color2(&src, ColorConversion2::Bgr2Luv).unwrap();
            let back = cvt_color2(&luv, ColorConversion2::Luv2Bgr).unwrap();
            for c in 0..3 {
                let (a, z) = (src.at_f64(0, 0, c).unwrap(), back.at_f64(0, 0, c).unwrap());
                assert!((a - z).abs() < 1e-4, "channel {c}: {a} vs {z}");
            }
        }
    }

    #[test]
    fn yuv_round_trips_and_grey_has_no_chroma() {
        let g = cvt_color2(&bgr8(100, 100, 100), ColorConversion2::Bgr2Yuv).unwrap();
        assert!((g.at_f64(0, 0, 0).unwrap() - 100.0).abs() < 1.0, "Y of grey is its level");
        assert!((g.at_f64(0, 0, 1).unwrap() - 128.0).abs() < 1.0, "U is the neutral 128");
        assert!((g.at_f64(0, 0, 2).unwrap() - 128.0).abs() < 1.0, "V is the neutral 128");
        // Forward values pinned against cv2 4.13.0.
        for ((b, gg, r), want) in [
            ((30u8, 200u8, 90u8), [148.0, 70.0, 77.0]),
            ((0, 0, 255), [76.0, 91.0, 255.0]),
            ((100, 100, 100), [100.0, 128.0, 128.0]),
        ] {
            let yuv = cvt_color2(&bgr8(b, gg, r), ColorConversion2::Bgr2Yuv).unwrap();
            for c in 0..3 {
                let got = yuv.at_f64(0, 0, c).unwrap();
                assert!(
                    (got - want[c as usize]).abs() <= 1.0,
                    "BGR({b},{gg},{r}) channel {c}: got {got}, cv2 gives {}",
                    want[c as usize]
                );
            }
        }

        // The round trip only holds for colours inside the YUV gamut. Pure red
        // is not: its V saturates at 255 and the information is gone. cv2 has
        // the same loss — it returns (1, 17, 221) for (0, 0, 255) — so
        // asserting an exact round trip there would be asserting something
        // OpenCV does not do either.
        for (b, gg, r) in [(30u8, 200, 90), (100, 100, 100)] {
            let src = bgr8(b, gg, r);
            let yuv = cvt_color2(&src, ColorConversion2::Bgr2Yuv).unwrap();
            let back = cvt_color2(&yuv, ColorConversion2::Yuv2Bgr).unwrap();
            for c in 0..3 {
                let (a, z) = (src.at_f64(0, 0, c).unwrap(), back.at_f64(0, 0, c).unwrap());
                assert!((a - z).abs() <= 2.0, "channel {c}: {a} vs {z}");
            }
        }
        let saturated = cvt_color2(
            &cvt_color2(&bgr8(0, 0, 255), ColorConversion2::Bgr2Yuv).unwrap(),
            ColorConversion2::Yuv2Bgr,
        )
        .unwrap();
        assert!(
            (saturated.at_f64(0, 0, 1).unwrap() - 17.0).abs() <= 2.0,
            "out-of-gamut red should come back near cv2's 17, got {}",
            saturated.at_f64(0, 0, 1).unwrap()
        );
    }

    #[test]
    fn hls_round_trips_and_matches_known_colours() {
        // Pure red: hue 0, lightness 0.5, saturation 1.
        let h = cvt_color2(&bgr32(0.0, 0.0, 1.0), ColorConversion2::Bgr2Hls).unwrap();
        assert!(h.at_f64(0, 0, 0).unwrap().abs() < 1e-6, "H = {}", h.at_f64(0, 0, 0).unwrap());
        assert!((h.at_f64(0, 0, 1).unwrap() - 0.5).abs() < 1e-6, "L should be 0.5");
        assert!((h.at_f64(0, 0, 2).unwrap() - 1.0).abs() < 1e-6, "S should be 1");
        // Pure green sits at 120 degrees.
        let g = cvt_color2(&bgr32(0.0, 1.0, 0.0), ColorConversion2::Bgr2Hls).unwrap();
        assert!((g.at_f64(0, 0, 0).unwrap() - 120.0).abs() < 1e-6);
        // Grey has no hue or saturation.
        let n = cvt_color2(&bgr32(0.4, 0.4, 0.4), ColorConversion2::Bgr2Hls).unwrap();
        assert!(n.at_f64(0, 0, 2).unwrap().abs() < 1e-9, "grey is unsaturated");

        for (b, gg, r) in [(0.2f32, 0.7, 0.4), (0.9, 0.3, 0.1)] {
            let src = bgr32(b, gg, r);
            let hls = cvt_color2(&src, ColorConversion2::Bgr2Hls).unwrap();
            let back = cvt_color2(&hls, ColorConversion2::Hls2Bgr).unwrap();
            for c in 0..3 {
                let (a, z) = (src.at_f64(0, 0, c).unwrap(), back.at_f64(0, 0, c).unwrap());
                assert!((a - z).abs() < 1e-6, "channel {c}: {a} vs {z}");
            }
        }
    }

    #[test]
    fn eight_bit_hue_is_halved_so_it_fits() {
        // Green is 120 degrees, which packs to 60 in an 8-bit channel.
        let g = cvt_color2(&bgr8(0, 255, 0), ColorConversion2::Bgr2Hls).unwrap();
        assert!((g.at_f64(0, 0, 0).unwrap() - 60.0).abs() < 1.0, "H = {}", g.at_f64(0, 0, 0).unwrap());
    }

    #[test]
    fn packed_565_and_555_round_trip_at_full_scale() {
        // The replication in the unpack is what makes 255 come back as 255
        // rather than 248.
        for code in [
            (ColorConversion2::Bgr2Bgr565, ColorConversion2::Bgr5652Bgr),
            (ColorConversion2::Bgr2Bgr555, ColorConversion2::Bgr5552Bgr),
        ] {
            let src = bgr8(255, 255, 255);
            let packed = cvt_color2(&src, code.0).unwrap();
            assert_eq!(packed.channels(), 2);
            let back = cvt_color2(&packed, code.1).unwrap();
            for c in 0..3 {
                assert_eq!(back.at_f64(0, 0, c).unwrap(), 255.0, "{:?} channel {c}", code.0);
            }
            // Black too.
            let k = cvt_color2(&cvt_color2(&bgr8(0, 0, 0), code.0).unwrap(), code.1).unwrap();
            for c in 0..3 {
                assert_eq!(k.at_f64(0, 0, c).unwrap(), 0.0);
            }
        }
    }

    #[test]
    fn packed_565_keeps_more_green_than_555() {
        // 565 gives green six bits; 555 gives it five. A mid-green should land
        // closer in 565.
        let src = bgr8(0, 100, 0);
        let e = |c0, c1| {
            let p = cvt_color2(&src, c0).unwrap();
            let b = cvt_color2(&p, c1).unwrap();
            (b.at_f64(0, 0, 1).unwrap() - 100.0).abs()
        };
        let e565 = e(ColorConversion2::Bgr2Bgr565, ColorConversion2::Bgr5652Bgr);
        let e555 = e(ColorConversion2::Bgr2Bgr555, ColorConversion2::Bgr5552Bgr);
        assert!(e565 <= e555, "565 error {e565} should not exceed 555 error {e555}");
    }

    #[test]
    fn premultiplied_alpha_round_trips() {
        let src = Mat::from_vec(1, 1, 4, &[200u8, 100, 50, 128]).unwrap();
        let m = cvt_color2(&src, ColorConversion2::Rgba2mRgba).unwrap();
        // Half alpha halves the colour channels and leaves alpha alone.
        assert!((m.at_f64(0, 0, 0).unwrap() - 100.0).abs() <= 1.0);
        assert_eq!(m.at_f64(0, 0, 3).unwrap(), 128.0);
        let back = cvt_color2(&m, ColorConversion2::MRgba2Rgba).unwrap();
        for c in 0..3 {
            let (a, z) = (src.at_f64(0, 0, c).unwrap(), back.at_f64(0, 0, c).unwrap());
            assert!((a - z).abs() <= 3.0, "channel {c}: {a} vs {z}");
        }
    }

    #[test]
    fn zero_alpha_unpremultiplies_to_black_rather_than_dividing_by_zero() {
        let src = Mat::from_vec(1, 1, 4, &[0u8, 0, 0, 0]).unwrap();
        let back = cvt_color2(&src, ColorConversion2::MRgba2Rgba).unwrap();
        for c in 0..4 {
            assert_eq!(back.at_f64(0, 0, c).unwrap(), 0.0);
        }
    }

    #[test]
    fn the_channel_count_is_checked() {
        let grey = Mat::zeros(2, 2, cv_core::types::CV_8UC1).unwrap();
        let e = cvt_color2(&grey, ColorConversion2::Bgr2Lab).unwrap_err();
        assert!(format!("{e}").contains("3-channel"), "got {e}");
        let three = bgr8(1, 2, 3);
        assert!(cvt_color2(&three, ColorConversion2::Rgba2mRgba).is_err());
        assert!(cvt_color2(&three, ColorConversion2::Bgr5652Bgr).is_err());
    }

    #[test]
    fn rgb_and_bgr_variants_differ_only_in_channel_order() {
        let bgr = bgr32(0.2, 0.5, 0.8);
        let rgb = bgr32(0.8, 0.5, 0.2); // the same colour, other order
        let a = cvt_color2(&bgr, ColorConversion2::Bgr2Lab).unwrap();
        let b = cvt_color2(&rgb, ColorConversion2::Rgb2Lab).unwrap();
        for c in 0..3 {
            assert!(
                (a.at_f64(0, 0, c).unwrap() - b.at_f64(0, 0, c).unwrap()).abs() < 1e-9,
                "channel {c} differs between the BGR and RGB entry points"
            );
        }
    }

    /// A single BGR pixel as `CV_32FC3`, mirroring the module's own fixtures.
    fn pix32(c0: f64, c1: f64, c2: f64) -> Mat {
        Mat::from_vec(1, 1, 3, &[c0 as f32, c1 as f32, c2 as f32]).unwrap()
    }

    #[test]
    fn the_rgb_luv_codes_are_the_bgr_ones_with_the_channels_swapped() {
        let bgr = pix32(0.2, 0.6, 0.9);
        let rgb = pix32(0.9, 0.6, 0.2);
        let a = cvt_color2(&bgr, ColorConversion2::Bgr2Luv).unwrap();
        let b = cvt_color2(&rgb, ColorConversion2::Rgb2Luv).unwrap();
        for c in 0..3 {
            let (x, y) = (a.at_f64(0, 0, c).unwrap(), b.at_f64(0, 0, c).unwrap());
            assert!((x - y).abs() < 1e-9, "channel {c}: {x} vs {y}");
        }
        // ...and the inverse puts the channels back in the order asked for.
        let back_bgr = cvt_color2(&a, ColorConversion2::Luv2Bgr).unwrap();
        let back_rgb = cvt_color2(&a, ColorConversion2::Luv2Rgb).unwrap();
        for c in 0..3 {
            let x = back_bgr.at_f64(0, 0, c).unwrap();
            let y = back_rgb.at_f64(0, 0, 2 - c).unwrap();
            assert!((x - y).abs() < 1e-9, "channel {c}: {x} vs {y}");
        }
        assert!((back_bgr.at_f64(0, 0, 0).unwrap() - 0.2).abs() < 1e-4);
        assert!((back_bgr.at_f64(0, 0, 2).unwrap() - 0.9).abs() < 1e-4);
    }

    #[test]
    fn the_linear_rgb_codes_skip_the_srgb_gamma() {
        let p = pix32(0.2, 0.5, 0.8);
        for (gamma, linear) in [
            (ColorConversion2::Bgr2Lab, ColorConversion2::LinearBgr2Lab),
            (ColorConversion2::Rgb2Lab, ColorConversion2::LinearRgb2Lab),
            (ColorConversion2::Bgr2Luv, ColorConversion2::LinearBgr2Luv),
            (ColorConversion2::Rgb2Luv, ColorConversion2::LinearRgb2Luv),
        ] {
            let a = cvt_color2(&p, gamma).unwrap();
            let b = cvt_color2(&p, linear).unwrap();
            // Skipping the expansion leaves the values brighter, so L is higher.
            let (la, lb) = (a.at_f64(0, 0, 0).unwrap(), b.at_f64(0, 0, 0).unwrap());
            assert!(lb > la, "{linear:?} L {lb} should exceed {gamma:?} L {la}");
        }
    }

    #[test]
    fn the_linear_rgb_codes_round_trip() {
        let p = pix32(0.25, 0.5, 0.75);
        for (fwd, back) in [
            (ColorConversion2::LinearBgr2Lab, ColorConversion2::Lab2LinearBgr),
            (ColorConversion2::LinearRgb2Lab, ColorConversion2::Lab2LinearRgb),
            (ColorConversion2::LinearBgr2Luv, ColorConversion2::Luv2LinearBgr),
            (ColorConversion2::LinearRgb2Luv, ColorConversion2::Luv2LinearRgb),
        ] {
            let mid = cvt_color2(&p, fwd).unwrap();
            let out = cvt_color2(&mid, back).unwrap();
            for c in 0..3 {
                let (want, got) = (p.at_f64(0, 0, c).unwrap(), out.at_f64(0, 0, c).unwrap());
                assert!((want - got).abs() < 1e-4, "{fwd:?}/{back:?} channel {c}: {want} vs {got}");
            }
        }
    }
}
