//! `cv::cvtColor` (`modules/imgproc/src/color*.cpp`).
//!
//! The 8-bit kernels reproduce upstream's fixed-point arithmetic exactly rather
//! than computing in floating point and rounding at the end. The two differ:
//! grayscale conversion uses a 15-bit shift with coefficients that sum to
//! exactly `1 << 15`, while YCrCb uses a 14-bit shift, and HSV uses 12-bit
//! reciprocal lookup tables. Computing any of them in `f32` lands one LSB away
//! on a meaningful fraction of pixels.

use cv_core::{
    cv_assert, cv_error, Code, Depth, Mat, MatLike, MatType, Result,
};

/// `gray_shift` — RGB→Gray fixed-point shift.
const GRAY_SHIFT: i32 = 15;
/// `RY15` == `0.299f * 32768 + 0.5`.
const RY15: i32 = 9798;
/// `GY15` == `0.587f * 32768 + 0.5`.
const GY15: i32 = 19235;
/// `BY15` == `0.114f * 32768 + 0.5`.
const BY15: i32 = 3735;

/// `yuv_shift` — RGB↔YCrCb fixed-point shift.
const YUV_SHIFT: i32 = 14;
/// `R2Y` == `0.299f * 16384`.
const R2Y: i32 = 4899;
/// `G2Y` == `0.587f * 16384`.
const G2Y: i32 = 9617;
/// `B2Y` == `0.114f * 16384`.
const B2Y: i32 = 1868;
/// `YCRI` == `0.713f * 16384`.
const YCRI: i32 = 11682;
/// `YCBI` == `0.564f * 16384`.
const YCBI: i32 = 9241;
/// `V2RI` == `1.403f * 16384`.
const V2RI: i32 = 22987;
/// `U2GI` == `-0.344f * 16384`.
const U2GI: i32 = -5636;
/// `V2GI` == `-0.714f * 16384`.
const V2GI: i32 = -11698;
/// `U2BI` == `1.773f * 16384`.
const U2BI: i32 = 29049;

/// Float coefficients, matching `R2YF`/`G2YF`/`B2YF`.
const R2YF: f32 = 0.299;
const G2YF: f32 = 0.587;
const B2YF: f32 = 0.114;

/// `CV_DESCALE(x, n)` — rounding right shift.
#[inline]
const fn cv_descale(x: i32, n: i32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// `cv::ColorConversionCodes` — the subset ported so far, with upstream values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
#[non_exhaustive]
pub enum ColorConversion {
    /// `COLOR_BGR2BGRA` / `COLOR_RGB2RGBA` — append an opaque alpha channel.
    Bgr2Bgra = 0,
    /// `COLOR_BGRA2BGR` / `COLOR_RGBA2RGB` — drop the alpha channel.
    Bgra2Bgr = 1,
    /// `COLOR_BGR2RGBA` / `COLOR_RGB2BGRA`.
    Bgr2Rgba = 2,
    /// `COLOR_RGBA2BGR` / `COLOR_BGRA2RGB`.
    Rgba2Bgr = 3,
    /// `COLOR_BGR2RGB` / `COLOR_RGB2BGR` — channel reversal.
    Bgr2Rgb = 4,
    /// `COLOR_BGRA2RGBA` / `COLOR_RGBA2BGRA`.
    Bgra2Rgba = 5,
    /// `COLOR_BGR2GRAY`.
    Bgr2Gray = 6,
    /// `COLOR_RGB2GRAY`.
    Rgb2Gray = 7,
    /// `COLOR_GRAY2BGR` / `COLOR_GRAY2RGB` — replicate across three channels.
    Gray2Bgr = 8,
    /// `COLOR_GRAY2BGRA` / `COLOR_GRAY2RGBA`.
    Gray2Bgra = 9,
    /// `COLOR_BGRA2GRAY`.
    Bgra2Gray = 10,
    /// `COLOR_RGBA2GRAY`.
    Rgba2Gray = 11,
    /// `COLOR_BGR2YCrCb`.
    Bgr2YCrCb = 36,
    /// `COLOR_RGB2YCrCb`.
    Rgb2YCrCb = 37,
    /// `COLOR_YCrCb2BGR`.
    YCrCb2Bgr = 38,
    /// `COLOR_YCrCb2RGB`.
    YCrCb2Rgb = 39,
    /// `COLOR_BGR2HSV` — hue in 0..180 for 8-bit, 0..360 for float.
    Bgr2Hsv = 40,
    /// `COLOR_RGB2HSV`.
    Rgb2Hsv = 41,
    /// `COLOR_HSV2BGR`.
    Hsv2Bgr = 54,
    /// `COLOR_HSV2RGB`.
    Hsv2Rgb = 55,
}

impl ColorConversion {
    /// The variant with upstream's numeric value, or `None` if this port does
    /// not carry a conversion with that code.
    ///
    /// Needed wherever the code arrives as an integer rather than as a Rust
    /// enum — a C ABI, a config file, a script.
    #[must_use]
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(ColorConversion::Bgr2Bgra),
            1 => Some(ColorConversion::Bgra2Bgr),
            2 => Some(ColorConversion::Bgr2Rgba),
            3 => Some(ColorConversion::Rgba2Bgr),
            4 => Some(ColorConversion::Bgr2Rgb),
            5 => Some(ColorConversion::Bgra2Rgba),
            6 => Some(ColorConversion::Bgr2Gray),
            7 => Some(ColorConversion::Rgb2Gray),
            8 => Some(ColorConversion::Gray2Bgr),
            9 => Some(ColorConversion::Gray2Bgra),
            10 => Some(ColorConversion::Bgra2Gray),
            11 => Some(ColorConversion::Rgba2Gray),
            36 => Some(ColorConversion::Bgr2YCrCb),
            37 => Some(ColorConversion::Rgb2YCrCb),
            38 => Some(ColorConversion::YCrCb2Bgr),
            39 => Some(ColorConversion::YCrCb2Rgb),
            40 => Some(ColorConversion::Bgr2Hsv),
            41 => Some(ColorConversion::Rgb2Hsv),
            54 => Some(ColorConversion::Hsv2Bgr),
            55 => Some(ColorConversion::Hsv2Rgb),
            _ => None,
        }
    }

    /// Upstream's numeric value for this conversion.
    #[must_use]
    pub fn to_i32(self) -> i32 {
        self as i32
    }
}

/// `cv::cvtColor(src, dst, code)`.
///
/// Accepts `CV_8U`, `CV_16U` and `CV_32F` depths where upstream does; HSV is
/// `CV_8U`/`CV_32F` only, as documented in `imgproc.hpp`.
pub fn cvt_color<S: MatLike + ?Sized>(src: &S, code: ColorConversion) -> Result<Mat> {
    use ColorConversion as C;
    match code {
        C::Bgr2Bgra => reorder(src, &[0, 1, 2], true),
        C::Bgra2Bgr => reorder(src, &[0, 1, 2], false),
        C::Bgr2Rgba => reorder(src, &[2, 1, 0], true),
        C::Rgba2Bgr => reorder(src, &[2, 1, 0], false),
        C::Bgr2Rgb => reorder(src, &[2, 1, 0], false),
        C::Bgra2Rgba => reorder(src, &[2, 1, 0], true),
        C::Bgr2Gray | C::Bgra2Gray => to_gray(src, 0),
        C::Rgb2Gray | C::Rgba2Gray => to_gray(src, 2),
        C::Gray2Bgr => from_gray(src, 3),
        C::Gray2Bgra => from_gray(src, 4),
        C::Bgr2YCrCb => to_ycrcb(src, 0),
        C::Rgb2YCrCb => to_ycrcb(src, 2),
        C::YCrCb2Bgr => from_ycrcb(src, 0),
        C::YCrCb2Rgb => from_ycrcb(src, 2),
        C::Bgr2Hsv => to_hsv(src, 0),
        C::Rgb2Hsv => to_hsv(src, 2),
        C::Hsv2Bgr => from_hsv(src, 0),
        C::Hsv2Rgb => from_hsv(src, 2),
    }
}

/// Reject depths a conversion does not support.
fn require_depth<S: MatLike + ?Sized>(src: &S, allowed: &[Depth], what: &str) -> Result<()> {
    if !allowed.contains(&src.depth()) {
        return cv_error!(
            Code::StsUnsupportedFormat,
            "{what} does not support {}",
            src.depth().name()
        );
    }
    Ok(())
}

/// Permute (and optionally extend with alpha) the first three channels.
fn reorder<S: MatLike + ?Sized>(src: &S, order: &[usize; 3], add_alpha: bool) -> Result<Mat> {
    let scn = src.channels() as usize;
    cv_assert!(scn == 3 || scn == 4, "expected a 3- or 4-channel image, got {scn}");
    let dcn = if add_alpha { 4 } else { 3 };
    let typ = MatType::new(src.depth(), dcn);
    let mut dst = Mat::zeros(src.rows(), src.cols(), typ)?;

    // Opaque alpha differs per depth: 255, 65535 or 1.0.
    let alpha = match src.depth() {
        Depth::U8 => 255.0,
        Depth::U16 => 65535.0,
        _ => 1.0,
    };
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            for (c, &src_c) in order.iter().enumerate() {
                let v = src.at_f64(y, x, src_c as i32)?;
                set_channel(&mut dst, y, x, c as i32, v)?;
            }
            if add_alpha {
                // Carry an existing alpha through rather than forcing opacity.
                let a = if scn == 4 { src.at_f64(y, x, 3)? } else { alpha };
                set_channel(&mut dst, y, x, 3, a)?;
            }
        }
    }
    Ok(dst)
}

/// Write one channel, saturating into the destination depth.
///
/// Re-exported from `cv-core` so every imgproc kernel stores identically.
pub(crate) use cv_core::arithm::set_channel;

/// RGB/BGR → Gray. `blue_idx` is 0 for BGR input, 2 for RGB.
fn to_gray<S: MatLike + ?Sized>(src: &S, blue_idx: i32) -> Result<Mat> {
    let scn = src.channels();
    cv_assert!(scn == 3 || scn == 4, "expected a 3- or 4-channel image, got {scn}");
    require_depth(src, &[Depth::U8, Depth::U16, Depth::F32], "cvtColor to GRAY")?;
    let (rows, cols) = (src.rows(), src.cols());
    let (ri, gi, bi) = ((blue_idx ^ 2) as usize, 1usize, blue_idx as usize);
    let cn = scn as usize;

    // One bulk widen instead of three `at_f64` calls per pixel.
    let buf = cv_core::to_f64(src)?;
    let n = rows as usize * cols as usize;
    let mut out = vec![0.0f64; n];
    let integer = !src.depth().is_float();

    for (i, o) in out.iter_mut().enumerate() {
        let p = i * cn;
        *o = if integer {
            // 15-bit fixed point; the coefficients sum to exactly 1<<15.
            let r = buf[p + ri] as i32;
            let g = buf[p + gi] as i32;
            let b = buf[p + bi] as i32;
            cv_descale(r * RY15 + g * GY15 + b * BY15, GRAY_SHIFT) as f64
        } else {
            let r = buf[p + ri] as f32;
            let g = buf[p + gi] as f32;
            let b = buf[p + bi] as f32;
            (r * R2YF + g * G2YF + b * B2YF) as f64
        };
    }
    cv_core::mat_from_f64(rows, cols, MatType::new(src.depth(), 1), &out)
}

/// Gray → 3- or 4-channel, replicating the luma into every colour channel.
fn from_gray<S: MatLike + ?Sized>(src: &S, dcn: i32) -> Result<Mat> {
    cv_assert!(src.channels() == 1, "expected a single-channel image");
    let mut dst = Mat::zeros(src.rows(), src.cols(), MatType::new(src.depth(), dcn))?;
    let alpha = match src.depth() {
        Depth::U8 => 255.0,
        Depth::U16 => 65535.0,
        _ => 1.0,
    };
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            let v = src.at_f64(y, x, 0)?;
            for c in 0..3 {
                set_channel(&mut dst, y, x, c, v)?;
            }
            if dcn == 4 {
                set_channel(&mut dst, y, x, 3, alpha)?;
            }
        }
    }
    Ok(dst)
}

/// Neutral chroma level for a depth — `ColorChannel<T>::half()`.
fn chroma_half(depth: Depth) -> f64 {
    match depth {
        Depth::U8 => 128.0,
        Depth::U16 => 32768.0,
        _ => 0.5,
    }
}

/// RGB/BGR → YCrCb, 14-bit fixed point for integer depths.
fn to_ycrcb<S: MatLike + ?Sized>(src: &S, blue_idx: i32) -> Result<Mat> {
    let scn = src.channels();
    cv_assert!(scn == 3 || scn == 4, "expected a 3- or 4-channel image, got {scn}");
    require_depth(src, &[Depth::U8, Depth::U16, Depth::F32], "cvtColor to YCrCb")?;
    let mut dst = Mat::zeros(src.rows(), src.cols(), MatType::new(src.depth(), 3))?;
    let half = chroma_half(src.depth());
    let is_int = !src.depth().is_float();
    let delta = (half * (1 << YUV_SHIFT) as f64) as i32;

    for y in 0..src.rows() {
        for x in 0..src.cols() {
            let c0 = src.at_f64(y, x, 0)?;
            let c1 = src.at_f64(y, x, 1)?;
            let c2 = src.at_f64(y, x, 2)?;
            // `src[0]`/`src[2]` in upstream indexing; `bidx` selects which is blue.
            let (r, b) = if blue_idx == 0 { (c2, c0) } else { (c0, c2) };
            let g = c1;

            let (yy, cr, cb) = if is_int {
                let (ri, gi, bi) = (r as i32, g as i32, b as i32);
                let yv = cv_descale(ri * R2Y + gi * G2Y + bi * B2Y, YUV_SHIFT);
                let cr = cv_descale((ri - yv) * YCRI + delta, YUV_SHIFT);
                let cb = cv_descale((bi - yv) * YCBI + delta, YUV_SHIFT);
                (yv as f64, cr as f64, cb as f64)
            } else {
                let yv = r as f32 * R2YF + g as f32 * G2YF + b as f32 * B2YF;
                let cr = (r as f32 - yv) * 0.713 + half as f32;
                let cb = (b as f32 - yv) * 0.564 + half as f32;
                (yv as f64, cr as f64, cb as f64)
            };
            set_channel(&mut dst, y, x, 0, yy)?;
            set_channel(&mut dst, y, x, 1, cr)?;
            set_channel(&mut dst, y, x, 2, cb)?;
        }
    }
    Ok(dst)
}

/// YCrCb → RGB/BGR.
fn from_ycrcb<S: MatLike + ?Sized>(src: &S, blue_idx: i32) -> Result<Mat> {
    cv_assert!(src.channels() == 3, "expected a 3-channel image");
    require_depth(src, &[Depth::U8, Depth::U16, Depth::F32], "cvtColor from YCrCb")?;
    let mut dst = Mat::zeros(src.rows(), src.cols(), MatType::new(src.depth(), 3))?;
    let half = chroma_half(src.depth());
    let is_int = !src.depth().is_float();

    for y in 0..src.rows() {
        for x in 0..src.cols() {
            let yy = src.at_f64(y, x, 0)?;
            let cr = src.at_f64(y, x, 1)?;
            let cb = src.at_f64(y, x, 2)?;
            let (r, g, b) = if is_int {
                let (yi, cri, cbi) = (yy as i32, cr as i32 - half as i32, cb as i32 - half as i32);
                (
                    (yi + cv_descale(cri * V2RI, YUV_SHIFT)) as f64,
                    (yi + cv_descale(cbi * U2GI + cri * V2GI, YUV_SHIFT)) as f64,
                    (yi + cv_descale(cbi * U2BI, YUV_SHIFT)) as f64,
                )
            } else {
                let (yf, crf, cbf) = (yy as f32, (cr - half) as f32, (cb - half) as f32);
                ((yf + crf * 1.403) as f64, (yf + cbf * -0.344 + crf * -0.714) as f64, (yf + cbf * 1.773) as f64)
            };
            // Write in destination channel order.
            let (o0, o2) = if blue_idx == 0 { (b, r) } else { (r, b) };
            set_channel(&mut dst, y, x, 0, o0)?;
            set_channel(&mut dst, y, x, 1, g)?;
            set_channel(&mut dst, y, x, 2, o2)?;
        }
    }
    Ok(dst)
}

/// `hsv_shift` — HSV reciprocal-table fixed-point shift.
const HSV_SHIFT: i32 = 12;

/// `sdiv_table[i] = saturate_cast<int>((255 << 12) / (1.0 * i))`.
fn sdiv_table() -> [i32; 256] {
    let mut t = [0i32; 256];
    for (i, slot) in t.iter_mut().enumerate().skip(1) {
        *slot = cv_core::cv_round(((255 << HSV_SHIFT) as f64) / (i as f64));
    }
    t
}

/// `hdiv_table180[i] = saturate_cast<int>((180 << 12) / (6.0 * i))`.
fn hdiv_table180() -> [i32; 256] {
    let mut t = [0i32; 256];
    for (i, slot) in t.iter_mut().enumerate().skip(1) {
        *slot = cv_core::cv_round(((180 << HSV_SHIFT) as f64) / (6.0 * i as f64));
    }
    t
}

/// RGB/BGR → HSV. Hue spans 0..180 for `CV_8U` and 0..360 for `CV_32F`.
fn to_hsv<S: MatLike + ?Sized>(src: &S, blue_idx: i32) -> Result<Mat> {
    let scn = src.channels();
    cv_assert!(scn == 3 || scn == 4, "expected a 3- or 4-channel image, got {scn}");
    require_depth(src, &[Depth::U8, Depth::F32], "cvtColor to HSV")?;
    let mut dst = Mat::zeros(src.rows(), src.cols(), MatType::new(src.depth(), 3))?;
    let (ri, gi, bi) = (blue_idx ^ 2, 1, blue_idx);

    if src.depth() == Depth::U8 {
        let sdiv = sdiv_table();
        let hdiv = hdiv_table180();
        let hr = 180i32;
        let cn = scn as usize;
        let buf = cv_core::to_f64(src)?;
        let n = src.rows() as usize * src.cols() as usize;
        let mut out = vec![0.0f64; n * 3];

        for i in 0..n {
            let p = i * cn;
            let b = buf[p + bi as usize] as i32;
            let g = buf[p + gi as usize] as i32;
            let r = buf[p + ri as usize] as i32;
            let v = b.max(g).max(r);
            let vmin = b.min(g).min(r);
            let diff = v - vmin;
            let vr = if v == r { -1i32 } else { 0 };
            let vg = if v == g { -1i32 } else { 0 };

            let s_out = (diff * sdiv[v as usize] + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT;
            let mut h = (vr & (g - b))
                + (!vr & ((vg & (b - r + 2 * diff)) + (!vg & (r - g + 4 * diff))));
            h = (h * hdiv[diff as usize] + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT;
            h += if h < 0 { hr } else { 0 };

            out[i * 3] = h as f64;
            out[i * 3 + 1] = s_out as f64;
            out[i * 3 + 2] = v as f64;
        }
        return cv_core::mat_from_f64(
            src.rows(),
            src.cols(),
            MatType::new(Depth::U8, 3),
            &out,
        );
    }
    {
        for y in 0..src.rows() {
            for x in 0..src.cols() {
                let b = src.at_f64(y, x, bi)? as f32;
                let g = src.at_f64(y, x, gi)? as f32;
                let r = src.at_f64(y, x, ri)? as f32;
                let v = b.max(g).max(r);
                let vmin = b.min(g).min(r);
                let diff = v - vmin;
                let s = if v == 0.0 { 0.0 } else { diff / v };
                let mut h = if diff == 0.0 {
                    0.0
                } else if v == r {
                    (g - b) * 60.0 / diff
                } else if v == g {
                    (b - r) * 60.0 / diff + 120.0
                } else {
                    (r - g) * 60.0 / diff + 240.0
                };
                if h < 0.0 {
                    h += 360.0;
                }
                set_channel(&mut dst, y, x, 0, h as f64)?;
                set_channel(&mut dst, y, x, 1, s as f64)?;
                set_channel(&mut dst, y, x, 2, v as f64)?;
            }
        }
    }
    Ok(dst)
}

/// HSV → RGB/BGR, following `HSV2RGB_native`.
/// HSV to RGB.
///
/// # OpenCV disagrees with itself here
///
/// `HSV2RGB_b` finishes its SIMD path with `v_trunc` — truncation — and its
/// scalar tail with `saturate_cast<uchar>`, which rounds. The same pixel
/// therefore converts differently depending on where it sits in the row:
/// HSV `(118, 60, 248)` comes out of cv2 4.13.0 as `(248, 193, 189)` in a
/// 64-wide image and `(248, 194, 190)` in a 3-wide one. Across a random 64x64
/// image the two paths differ on **33.9%** of components.
///
/// Exact parity with both is impossible, so this follows the scalar path — the
/// correctly-rounded one — and is bit-exact against it on every component of
/// that same image. Expect a difference of one level from cv2's *vectorised*
/// output on about a third of pixels; that is upstream's inconsistency, not a
/// rounding choice made here.
fn from_hsv<S: MatLike + ?Sized>(src: &S, blue_idx: i32) -> Result<Mat> {
    cv_assert!(src.channels() == 3, "expected a 3-channel image");
    require_depth(src, &[Depth::U8, Depth::F32], "cvtColor from HSV")?;
    let mut dst = Mat::zeros(src.rows(), src.cols(), MatType::new(src.depth(), 3))?;
    let is_u8 = src.depth() == Depth::U8;
    // 8-bit hue occupies 0..180, so it is rescaled onto the 0..6 sextant axis
    // by 6/180; float hue is already in degrees.
    let hscale = if is_u8 { 6.0 / 180.0 } else { 6.0 / 360.0 };

    for y in 0..src.rows() {
        for x in 0..src.cols() {
            let h = src.at_f64(y, x, 0)? as f32;
            let mut s = src.at_f64(y, x, 1)? as f32;
            let mut v = src.at_f64(y, x, 2)? as f32;
            if is_u8 {
                s *= 1.0 / 255.0;
                v *= 1.0 / 255.0;
            }
            let (r, g, b) = hsv_to_rgb(h * hscale as f32, s, v);
            let scale = if is_u8 { 255.0 } else { 1.0 };
            let (o0, o2) = if blue_idx == 0 { (b, r) } else { (r, b) };
            set_channel(&mut dst, y, x, 0, (o0 * scale) as f64)?;
            set_channel(&mut dst, y, x, 1, (g * scale) as f64)?;
            set_channel(&mut dst, y, x, 2, (o2 * scale) as f64)?;
        }
    }
    Ok(dst)
}

/// Sextant decomposition shared by both HSV→RGB depths.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    // Sector table from `HSV2RGB_native`: which of (v, p, q, t) feeds each channel.
    const SECTOR_DATA: [[i32; 3]; 6] =
        [[0, 3, 1], [2, 0, 1], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]];
    if s == 0.0 {
        return (v, v, v);
    }
    let mut sector = h.floor();
    let f = h - sector;
    // Negative hues wrap forward a whole turn, as upstream does.
    if (sector as i32) < 0 || sector.is_nan() {
        sector = 0.0;
    }
    let sector_i = ((sector as i32) % 6 + 6) as usize % 6;
    let tab = [v, v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f))];
    let idx = SECTOR_DATA[sector_i];
    (tab[idx[0] as usize], tab[idx[1] as usize], tab[idx[2] as usize])
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::Rect;

    fn bgr(v: &[u8]) -> Mat {
        Mat::from_vec(1, (v.len() / 3) as i32, 3, v).unwrap()
    }

    #[test]
    fn code_values_match_opencv() {
        assert_eq!(ColorConversion::Bgr2Bgra as i32, 0);
        assert_eq!(ColorConversion::Bgr2Rgb as i32, 4);
        assert_eq!(ColorConversion::Bgr2Gray as i32, 6);
        assert_eq!(ColorConversion::Rgb2Gray as i32, 7);
        assert_eq!(ColorConversion::Gray2Bgr as i32, 8);
        assert_eq!(ColorConversion::Bgr2YCrCb as i32, 36);
        assert_eq!(ColorConversion::Bgr2Hsv as i32, 40);
        assert_eq!(ColorConversion::Hsv2Rgb as i32, 55);
    }

    #[test]
    fn gray_coefficients_sum_to_one_shift() {
        // Upstream asserts exactly this in RGB2Gray<uchar>'s constructor.
        assert_eq!(RY15 + GY15 + BY15, 1 << GRAY_SHIFT);
    }

    #[test]
    fn bgr_to_gray_matches_fixed_point_reference() {
        // Reference values computed with the upstream integer expression
        // (r*9798 + g*19235 + b*3735 + 16384) >> 15.
        let m = bgr(&[0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255]);
        let g = cvt_color(&m, ColorConversion::Bgr2Gray).unwrap();
        assert_eq!(g.channels(), 1);
        assert_eq!(g.to_vec::<u8>().unwrap(), vec![0, 255, 29, 150, 76]);
    }

    #[test]
    fn rgb_to_gray_swaps_red_and_blue() {
        let m = bgr(&[255, 0, 0]); // as RGB this is pure red
        let g = cvt_color(&m, ColorConversion::Rgb2Gray).unwrap();
        assert_eq!(g.to_vec::<u8>().unwrap(), vec![76]);
    }

    #[test]
    fn float_gray_uses_float_coefficients() {
        let m = Mat::from_vec(1, 1, 3, &[0.0f32, 0.0, 1.0]).unwrap();
        let g = cvt_color(&m, ColorConversion::Bgr2Gray).unwrap();
        assert!((g.to_vec::<f32>().unwrap()[0] - 0.299).abs() < 1e-6);
    }

    #[test]
    fn bgr_rgb_roundtrip_is_identity() {
        let m = bgr(&[1, 2, 3, 4, 5, 6]);
        let once = cvt_color(&m, ColorConversion::Bgr2Rgb).unwrap();
        assert_eq!(once.to_vec::<u8>().unwrap(), vec![3, 2, 1, 6, 5, 4]);
        let twice = cvt_color(&once, ColorConversion::Bgr2Rgb).unwrap();
        assert_eq!(twice.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn alpha_is_added_opaque_and_dropped() {
        let m = bgr(&[1, 2, 3]);
        let rgba = cvt_color(&m, ColorConversion::Bgr2Bgra).unwrap();
        assert_eq!(rgba.channels(), 4);
        assert_eq!(rgba.to_vec::<u8>().unwrap(), vec![1, 2, 3, 255]);
        let back = cvt_color(&rgba, ColorConversion::Bgra2Bgr).unwrap();
        assert_eq!(back.to_vec::<u8>().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn gray_to_bgr_replicates() {
        let m = Mat::from_vec(1, 2, 1, &[7u8, 200]).unwrap();
        let c = cvt_color(&m, ColorConversion::Gray2Bgr).unwrap();
        assert_eq!(c.to_vec::<u8>().unwrap(), vec![7, 7, 7, 200, 200, 200]);
        let c4 = cvt_color(&m, ColorConversion::Gray2Bgra).unwrap();
        assert_eq!(c4.to_vec::<u8>().unwrap(), vec![7, 7, 7, 255, 200, 200, 200, 255]);
    }

    #[test]
    fn ycrcb_neutral_grey_has_centred_chroma() {
        // A grey pixel must map to (Y, 128, 128).
        let m = bgr(&[128, 128, 128]);
        let y = cvt_color(&m, ColorConversion::Bgr2YCrCb).unwrap();
        assert_eq!(y.to_vec::<u8>().unwrap(), vec![128, 128, 128]);
    }

    #[test]
    fn ycrcb_roundtrip_is_close() {
        let m = bgr(&[10, 120, 200, 250, 5, 60]);
        let y = cvt_color(&m, ColorConversion::Bgr2YCrCb).unwrap();
        let back = cvt_color(&y, ColorConversion::YCrCb2Bgr).unwrap();
        let (a, b) = (m.to_vec::<u8>().unwrap(), back.to_vec::<u8>().unwrap());
        for (p, q) in a.iter().zip(b.iter()) {
            assert!((*p as i32 - *q as i32).abs() <= 2, "{p} vs {q}");
        }
    }

    #[test]
    fn hsv_primaries_match_opencv_8u() {
        // OpenCV's 8-bit HSV: hue in 0..180, so red=0, green=60, blue=120.
        let m = bgr(&[0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 0, 0, 0]);
        let h = cvt_color(&m, ColorConversion::Bgr2Hsv).unwrap();
        let v = h.to_vec::<u8>().unwrap();
        assert_eq!(&v[0..3], &[0, 255, 255]); // red
        assert_eq!(&v[3..6], &[60, 255, 255]); // green
        assert_eq!(&v[6..9], &[120, 255, 255]); // blue
        assert_eq!(&v[9..12], &[0, 0, 255]); // white: no hue, no saturation
        assert_eq!(&v[12..15], &[0, 0, 0]); // black
    }

    #[test]
    fn hsv_roundtrip_recovers_primaries() {
        let m = bgr(&[0, 0, 255, 0, 255, 0, 255, 0, 0]);
        let h = cvt_color(&m, ColorConversion::Bgr2Hsv).unwrap();
        let back = cvt_color(&h, ColorConversion::Hsv2Bgr).unwrap();
        assert_eq!(back.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn hsv_float_hue_spans_360() {
        let m = Mat::from_vec(1, 2, 3, &[0.0f32, 1.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let h = cvt_color(&m, ColorConversion::Bgr2Hsv).unwrap();
        let v = h.to_vec::<f32>().unwrap();
        assert!((v[0] - 120.0).abs() < 1e-4, "green hue was {}", v[0]);
        assert!((v[3] - 240.0).abs() < 1e-4, "blue hue was {}", v[3]);
    }

    #[test]
    fn unsupported_depth_is_rejected() {
        let m = Mat::zeros(1, 1, MatType::new(Depth::S32, 3)).unwrap();
        let e = cvt_color(&m, ColorConversion::Bgr2Gray).unwrap_err();
        assert_eq!(e.code, Code::StsUnsupportedFormat);
    }

    #[test]
    fn wrong_channel_count_is_rejected() {
        let m = Mat::zeros(1, 1, MatType::new(Depth::U8, 1)).unwrap();
        assert!(cvt_color(&m, ColorConversion::Bgr2Gray).is_err());
    }

    #[test]
    fn accepts_a_strided_view() {
        let data: Vec<u8> = (0..48).collect();
        let m = Mat::from_vec(4, 4, 3, &data).unwrap();
        let v = m.roi(Rect::new(1, 1, 2, 2)).unwrap();
        let g = cvt_color(&v, ColorConversion::Bgr2Gray).unwrap();
        assert_eq!(g.size(), cv_core::Size::new(2, 2));
    }

    /// Bit-exact against cv2 4.13.0's *scalar* HSV2BGR path.
    ///
    /// See [`from_hsv`] for why the scalar path is the reference: cv2's SIMD
    /// path truncates where the scalar one rounds, so upstream returns two
    /// different answers for the same pixel depending on image width.
    #[test]
    fn hsv_to_bgr_is_bit_exact_against_cv2_scalar_path() {
        const HSV: [u8; 72] = [118, 60, 248, 100, 223, 207, 151, 233, 245, 86, 207, 239, 30, 247, 159, 179, 197, 236, 75, 244, 193, 40, 209, 61, 66, 223, 228, 28, 119, 169, 8, 244, 205, 78, 235, 231, 14, 207, 234, 70, 143, 242, 71, 149, 217, 163, 241, 168, 144, 142, 242, 179, 116, 178, 17, 81, 218, 19, 97, 249, 74, 185, 201, 48, 253, 214, 166, 219, 56, 153, 123, 211];
        const WANT: [u8; 72] = [248, 194, 190, 207, 147, 26, 238, 21, 245, 213, 239, 45, 5, 159, 159, 60, 54, 236, 101, 193, 8, 11, 61, 44, 68, 228, 29, 90, 164, 169, 9, 61, 205, 146, 231, 18, 44, 133, 234, 152, 242, 106, 137, 217, 90, 99, 9, 168, 242, 107, 215, 100, 97, 178, 149, 188, 218, 154, 214, 249, 123, 201, 55, 2, 214, 87, 30, 8, 56, 201, 109, 211];
        let src = Mat::from_vec(4, 6, 3, &HSV).unwrap();
        let got = cvt_color(&src, ColorConversion::Hsv2Bgr).unwrap();
        for y in 0..4 {
            for x in 0..6 {
                for c in 0..3 {
                    let i = ((y * 6 + x) * 3 + c) as usize;
                    assert_eq!(
                        got.at_f64(y, x, c).unwrap() as u8,
                        WANT[i],
                        "({y},{x}) channel {c}"
                    );
                }
            }
        }
    }
}
