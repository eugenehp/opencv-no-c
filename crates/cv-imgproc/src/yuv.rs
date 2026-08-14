//! YUV 4:2:0 and 4:2:2 conversions (`modules/imgproc/src/color_yuv.simd.hpp`).
//!
//! These are the formats video actually arrives in: a full-resolution luma
//! plane plus chroma at half resolution, so a frame costs 1.5 bytes per pixel
//! instead of 3. Every camera and codec in [`crate`]'s reach emits one of them.
//!
//! # Why the arithmetic is integer
//!
//! Upstream uses ITU-R BT.601 coefficients in 20-bit fixed point, and the
//! rounding of that fixed-point pipeline is visible in the output. Computing
//! the same conversion in floating point and rounding at the end disagrees
//! with OpenCV on a noticeable fraction of pixels, so this transcribes the
//! integer form, including the `max(0, y - 16)` clamp that happens *before*
//! the luma multiply.

use cv_core::types::MatType;
use cv_core::{cv_assert, Mat, MatLike, MatLikeMut, Result};

// ITU-R BT.601 constants, 20-bit fixed point, from `color_yuv.simd.hpp`.
const SHIFT: i32 = 20;
const HALF: i32 = 1 << (SHIFT - 1);
const CY: i32 = 1_220_542;
const CUB: i32 = 2_116_026;
const CUG: i32 = -409_993;
const CVG: i32 = -852_492;
const CVR: i32 = 1_673_527;

// The forward direction.
const CRY: i32 = 269_484;
const CGY: i32 = 528_482;
const CBY: i32 = 102_760;
const CRU: i32 = -155_188;
const CGU: i32 = -305_135;
const CBU: i32 = 460_324;
const CGV: i32 = -385_875;
const CBV: i32 = -74_448;

fn sat_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// How the two chroma planes of a 4:2:0 image are laid out.
///
/// The four names cover every alias upstream defines; the semi-planar pair
/// interleaves U and V in one plane, the planar pair keeps them separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Yuv420 {
    /// Semi-planar, U first. `COLOR_YUV2BGR_NV12`, `COLOR_YUV2RGB_NV12`,
    /// `COLOR_YUV2BGRA_NV12`, `COLOR_YUV2RGBA_NV12`, `COLOR_YUV2GRAY_NV12`,
    /// `COLOR_YUV420sp2BGR`, `COLOR_YUV420sp2RGB`, `COLOR_YUV420sp2BGRA`,
    /// `COLOR_YUV420sp2RGBA`, `COLOR_YUV420sp2GRAY`.
    Nv12,
    /// Semi-planar, V first. `COLOR_YUV2BGR_NV21`, `COLOR_YUV2RGB_NV21`,
    /// `COLOR_YUV2BGRA_NV21`, `COLOR_YUV2RGBA_NV21`, `COLOR_YUV2GRAY_NV21`.
    Nv21,
    /// Planar, U plane then V plane. `COLOR_YUV2BGR_I420`,
    /// `COLOR_YUV2RGB_I420`, `COLOR_YUV2BGRA_I420`, `COLOR_YUV2RGBA_I420`,
    /// `COLOR_YUV2GRAY_I420`, `COLOR_YUV2BGR_IYUV`, `COLOR_YUV2RGB_IYUV`,
    /// `COLOR_YUV2BGRA_IYUV`, `COLOR_YUV2RGBA_IYUV`, `COLOR_YUV2GRAY_IYUV`,
    /// `COLOR_YUV420p2BGR`, `COLOR_YUV420p2RGB`, `COLOR_YUV420p2BGRA`,
    /// `COLOR_YUV420p2RGBA`, `COLOR_YUV420p2GRAY`, `COLOR_YUV2GRAY_420`.
    I420,
    /// Planar, V plane then U plane. `COLOR_YUV2BGR_YV12`,
    /// `COLOR_YUV2RGB_YV12`, `COLOR_YUV2BGRA_YV12`, `COLOR_YUV2RGBA_YV12`,
    /// `COLOR_YUV2GRAY_YV12`.
    Yv12,
}

impl Yuv420 {
    /// True when U and V share one interleaved plane.
    fn semi_planar(self) -> bool {
        matches!(self, Yuv420::Nv12 | Yuv420::Nv21)
    }
    /// True when U comes before V.
    fn u_first(self) -> bool {
        matches!(self, Yuv420::Nv12 | Yuv420::I420)
    }
}

/// A packed 4:2:2 layout — one chroma pair per two horizontal pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Yuv422 {
    /// `Y0 U Y1 V`. `COLOR_YUV2BGR_YUY2`, `COLOR_YUV2RGB_YUY2`,
    /// `COLOR_YUV2BGRA_YUY2`, `COLOR_YUV2RGBA_YUY2`, `COLOR_YUV2GRAY_YUY2`,
    /// `COLOR_YUV2BGR_YUYV`, `COLOR_YUV2RGB_YUYV`, `COLOR_YUV2BGRA_YUYV`,
    /// `COLOR_YUV2RGBA_YUYV`, `COLOR_YUV2GRAY_YUYV`, `COLOR_YUV2BGR_YUNV`,
    /// `COLOR_YUV2RGB_YUNV`, `COLOR_YUV2BGRA_YUNV`, `COLOR_YUV2RGBA_YUNV`,
    /// `COLOR_YUV2GRAY_YUNV`.
    Yuy2,
    /// `U Y0 V Y1`. `COLOR_YUV2BGR_UYVY`, `COLOR_YUV2RGB_UYVY`,
    /// `COLOR_YUV2BGRA_UYVY`, `COLOR_YUV2RGBA_UYVY`, `COLOR_YUV2GRAY_UYVY`,
    /// `COLOR_YUV2BGR_UYNV`, `COLOR_YUV2RGB_UYNV`, `COLOR_YUV2BGRA_UYNV`,
    /// `COLOR_YUV2RGBA_UYNV`, `COLOR_YUV2GRAY_UYNV`, `COLOR_YUV2BGR_Y422`,
    /// `COLOR_YUV2RGB_Y422`, `COLOR_YUV2BGRA_Y422`, `COLOR_YUV2RGBA_Y422`,
    /// `COLOR_YUV2GRAY_Y422`.
    Uyvy,
    /// `Y0 V Y1 U`. `COLOR_YUV2BGR_YVYU`, `COLOR_YUV2RGB_YVYU`,
    /// `COLOR_YUV2BGRA_YVYU`, `COLOR_YUV2RGBA_YVYU`, `COLOR_YUV2GRAY_YVYU`.
    Yvyu,
}

/// The interleaving the RGB side uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RgbLayout {
    /// Three channels, blue first.
    Bgr,
    /// Three channels, red first.
    Rgb,
    /// Four channels, blue first, opaque alpha.
    Bgra,
    /// Four channels, red first, opaque alpha.
    Rgba,
    /// One channel — the luma plane, copied out untouched.
    Gray,
}

impl RgbLayout {
    fn channels(self) -> i32 {
        match self {
            RgbLayout::Gray => 1,
            RgbLayout::Bgr | RgbLayout::Rgb => 3,
            RgbLayout::Bgra | RgbLayout::Rgba => 4,
        }
    }
    fn red_first(self) -> bool {
        matches!(self, RgbLayout::Rgb | RgbLayout::Rgba)
    }
}

/// One YUV triple to one RGB pixel, in upstream's fixed point.
fn yuv_to_rgb(y: i32, u: i32, v: i32) -> (u8, u8, u8) {
    let uu = u - 128;
    let vv = v - 128;
    let ruv = HALF + CVR * vv;
    let guv = HALF + CVG * vv + CUG * uu;
    let buv = HALF + CUB * uu;
    // The clamp happens before the multiply, not after — that is what makes
    // sub-16 luma flat black rather than wrapping into a bright value.
    let y00 = y.max(16).saturating_sub(16) * CY;
    (
        sat_u8((y00 + ruv) >> SHIFT),
        sat_u8((y00 + guv) >> SHIFT),
        sat_u8((y00 + buv) >> SHIFT),
    )
}

/// Write one converted pixel into `out` honouring the layout.
fn put(out: &mut Mat, row: i32, col: i32, layout: RgbLayout, r: u8, g: u8, b: u8) -> Result<()> {
    let px = out.pixel_mut::<u8>(row, col)?;
    match layout {
        RgbLayout::Gray => px[0] = r,
        _ => {
            if layout.red_first() {
                px[0] = r;
                px[1] = g;
                px[2] = b;
            } else {
                px[0] = b;
                px[1] = g;
                px[2] = r;
            }
            if layout.channels() == 4 {
                px[3] = 255;
            }
        }
    }
    Ok(())
}

/// `cv::cvtColor` for the `COLOR_YUV2*_{NV12,NV21,I420,YV12}` family.
///
/// `data` is one contiguous frame: `width * height` luma bytes followed by
/// `width * height / 2` chroma bytes. `height` is the *image* height, not the
/// 1.5x buffer height upstream's `Mat` carries.
///
/// # Odd dimensions
///
/// 4:2:0 has one chroma sample per 2x2 block, so an odd width or height has no
/// consistent layout. Upstream asserts; so does this, rather than reading a
/// half-present chroma row.
pub fn yuv420_to_rgb(
    data: &[u8],
    width: i32,
    height: i32,
    format: Yuv420,
    layout: RgbLayout,
) -> Result<Mat> {
    cv_assert!(width > 0 && height > 0, "the frame size must be positive");
    cv_assert!(
        width % 2 == 0 && height % 2 == 0,
        "4:2:0 needs even dimensions, got {width}x{height}"
    );
    let (w, h) = (width as usize, height as usize);
    let need = w * h * 3 / 2;
    cv_assert!(
        data.len() >= need,
        "a {width}x{height} 4:2:0 frame needs {need} bytes, got {}",
        data.len()
    );

    let mut out = Mat::zeros(height, width, MatType::new(cv_core::types::Depth::U8, layout.channels()))?;
    let luma = &data[..w * h];

    if layout == RgbLayout::Gray {
        // Grey is the luma plane verbatim: no chroma is read at all, which is
        // why upstream's YUV2GRAY is so much cheaper than a full convert.
        for y in 0..h {
            for x in 0..w {
                out.pixel_mut::<u8>(y as i32, x as i32)?[0] = luma[y * w + x];
            }
        }
        return Ok(out);
    }

    let chroma = &data[w * h..];
    for y in 0..h {
        for x in 0..w {
            let (cy, cx) = (y / 2, x / 2);
            let (u, v) = if format.semi_planar() {
                let base = cy * w + cx * 2;
                let (a, b) = (chroma[base], chroma[base + 1]);
                if format.u_first() { (a, b) } else { (b, a) }
            } else {
                let plane = (w / 2) * (h / 2);
                let idx = cy * (w / 2) + cx;
                let (a, b) = (chroma[idx], chroma[plane + idx]);
                if format.u_first() { (a, b) } else { (b, a) }
            };
            let (r, g, b) = yuv_to_rgb(i32::from(luma[y * w + x]), i32::from(u), i32::from(v));
            put(&mut out, y as i32, x as i32, layout, r, g, b)?;
        }
    }
    Ok(out)
}

/// `cv::cvtColor` for the packed `COLOR_YUV2*_{YUY2,UYVY,YVYU}` family.
///
/// `data` holds `width * height * 2` bytes: two luma samples and one chroma
/// pair per four bytes.
pub fn yuv422_to_rgb(
    data: &[u8],
    width: i32,
    height: i32,
    format: Yuv422,
    layout: RgbLayout,
) -> Result<Mat> {
    cv_assert!(width > 0 && height > 0, "the frame size must be positive");
    cv_assert!(width % 2 == 0, "4:2:2 needs an even width, got {width}");
    let (w, h) = (width as usize, height as usize);
    cv_assert!(
        data.len() >= w * h * 2,
        "a {width}x{height} 4:2:2 frame needs {} bytes, got {}",
        w * h * 2,
        data.len()
    );

    let mut out = Mat::zeros(height, width, MatType::new(cv_core::types::Depth::U8, layout.channels()))?;
    for y in 0..h {
        for xp in 0..w / 2 {
            let q = &data[(y * (w / 2) + xp) * 4..][..4];
            // Which of the four bytes is which depends only on the layout.
            let (y0, y1, u, v) = match format {
                Yuv422::Yuy2 => (q[0], q[2], q[1], q[3]),
                Yuv422::Uyvy => (q[1], q[3], q[0], q[2]),
                Yuv422::Yvyu => (q[0], q[2], q[3], q[1]),
            };
            for (i, luma) in [y0, y1].into_iter().enumerate() {
                let x = xp * 2 + i;
                if layout == RgbLayout::Gray {
                    out.pixel_mut::<u8>(y as i32, x as i32)?[0] = luma;
                } else {
                    let (r, g, b) = yuv_to_rgb(i32::from(luma), i32::from(u), i32::from(v));
                    put(&mut out, y as i32, x as i32, layout, r, g, b)?;
                }
            }
        }
    }
    Ok(out)
}

/// `cv::cvtColor` for `COLOR_BGR2YUV_I420` and friends.
///
/// Covers `COLOR_BGR2YUV_I420`, `COLOR_BGR2YUV_IYUV`, `COLOR_RGB2YUV_I420`,
/// `COLOR_RGB2YUV_IYUV`, `COLOR_BGRA2YUV_I420`, `COLOR_BGRA2YUV_IYUV`,
/// `COLOR_RGBA2YUV_I420`, `COLOR_RGBA2YUV_IYUV`, `COLOR_BGR2YUV_YV12`,
/// `COLOR_RGB2YUV_YV12`, `COLOR_BGRA2YUV_YV12` and `COLOR_RGBA2YUV_YV12`.
///
/// Returns the packed `width * height * 3 / 2` buffer.
///
/// # Chroma is sampled, not averaged
///
/// Upstream takes the top-left pixel of each 2x2 block rather than averaging
/// the four. Averaging looks more correct and disagrees with OpenCV on every
/// block that is not flat, so this samples.
pub fn rgb_to_yuv420<S: MatLike + ?Sized>(
    src: &S,
    format: Yuv420,
    red_first: bool,
) -> Result<Vec<u8>> {
    let (h, w) = (src.rows(), src.cols());
    cv_assert!(w > 0 && h > 0, "the image must not be empty");
    cv_assert!(w % 2 == 0 && h % 2 == 0, "4:2:0 needs even dimensions, got {w}x{h}");
    cv_assert!(
        src.channels() == 3 || src.channels() == 4,
        "expected 3 or 4 channels, got {}",
        src.channels()
    );
    cv_assert!(
        !format.semi_planar(),
        "upstream only defines the planar I420/YV12 targets for this direction"
    );

    let (wu, hu) = (w as usize, h as usize);
    let mut out = vec![0u8; wu * hu * 3 / 2];
    let plane = (wu / 2) * (hu / 2);

    for y in 0..h {
        for x in 0..w {
            let c0 = src.at_f64(y, x, 0)? as i32;
            let c1 = src.at_f64(y, x, 1)? as i32;
            let c2 = src.at_f64(y, x, 2)? as i32;
            let (r, g, b) = if red_first { (c0, c1, c2) } else { (c2, c1, c0) };

            let yy = (CRY * r + CGY * g + CBY * b + HALF + (16 << SHIFT)) >> SHIFT;
            out[(y as usize) * wu + x as usize] = sat_u8(yy);

            // One chroma sample per 2x2 block, taken at the block's origin.
            if y % 2 == 0 && x % 2 == 0 {
                let u = (CRU * r + CGU * g + CBU * b + HALF + (128 << SHIFT)) >> SHIFT;
                let v = (CBU * r + CGV * g + CBV * b + HALF + (128 << SHIFT)) >> SHIFT;
                let idx = (y as usize / 2) * (wu / 2) + x as usize / 2;
                let (first, second) =
                    if format.u_first() { (sat_u8(u), sat_u8(v)) } else { (sat_u8(v), sat_u8(u)) };
                out[wu * hu + idx] = first;
                out[wu * hu + plane + idx] = second;
            }
        }
    }
    Ok(out)
}

/// `cv::cvtColorTwoPlane(src1, src2, dst, code)`.
///
/// The same conversion as [`yuv420_to_rgb`] when the luma and chroma planes
/// arrive separately rather than in one buffer — which is how most capture
/// APIs hand over an NV12 frame.
pub fn cvt_color_two_plane(
    luma: &[u8],
    chroma: &[u8],
    width: i32,
    height: i32,
    format: Yuv420,
    layout: RgbLayout,
) -> Result<Mat> {
    cv_assert!(
        format.semi_planar(),
        "cvtColorTwoPlane covers the semi-planar NV12/NV21 layouts"
    );
    cv_assert!(width > 0 && height > 0, "the frame size must be positive");
    let (w, h) = (width as usize, height as usize);
    cv_assert!(luma.len() >= w * h, "the luma plane is short");
    cv_assert!(chroma.len() >= w * h / 2, "the chroma plane is short");
    let mut joined = Vec::with_capacity(w * h * 3 / 2);
    joined.extend_from_slice(&luma[..w * h]);
    joined.extend_from_slice(&chroma[..w * h / 2]);
    yuv420_to_rgb(&joined, width, height, format, layout)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4x4 NV12 frame with a known pattern.
    fn nv12_frame() -> Vec<u8> {
        let mut d = vec![0u8; 4 * 4 * 3 / 2];
        for (i, v) in d[..16].iter_mut().enumerate() {
            *v = (i * 16) as u8;
        }
        // Chroma: 2x2 blocks, U then V interleaved.
        d[16..].copy_from_slice(&[128, 128, 200, 60, 60, 200, 128, 128]);
        d
    }

    #[test]
    fn grey_is_the_luma_plane_verbatim() {
        let d = nv12_frame();
        for f in [Yuv420::Nv12, Yuv420::Nv21, Yuv420::I420, Yuv420::Yv12] {
            let g = yuv420_to_rgb(&d, 4, 4, f, RgbLayout::Gray).unwrap();
            assert_eq!(g.channels(), 1);
            for y in 0..4 {
                for x in 0..4 {
                    let want = f64::from(d[(y * 4 + x) as usize]);
                    assert_eq!(g.at_f64(y, x, 0).unwrap(), want, "grey ignores chroma entirely");
                }
            }
        }
    }

    #[test]
    fn neutral_chroma_gives_a_grey_image() {
        // U = V = 128 means no colour; R, G and B must agree.
        let mut d = vec![128u8; 4 * 4 * 3 / 2];
        for (i, v) in d[..16].iter_mut().enumerate() {
            *v = (16 + i * 10) as u8;
        }
        let m = yuv420_to_rgb(&d, 4, 4, Yuv420::Nv12, RgbLayout::Bgr).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let (b, g, r) = (
                    m.at_f64(y, x, 0).unwrap(),
                    m.at_f64(y, x, 1).unwrap(),
                    m.at_f64(y, x, 2).unwrap(),
                );
                assert_eq!((b, g), (g, r), "neutral chroma must be achromatic at ({y},{x})");
            }
        }
    }

    #[test]
    fn nv12_and_nv21_disagree_exactly_by_swapping_chroma() {
        let d = nv12_frame();
        let a = yuv420_to_rgb(&d, 4, 4, Yuv420::Nv12, RgbLayout::Bgr).unwrap();
        // Swapping the interleaved pair turns NV12 data into NV21 data.
        let mut swapped = d.clone();
        for c in swapped[16..].chunks_mut(2) {
            c.swap(0, 1);
        }
        let b = yuv420_to_rgb(&swapped, 4, 4, Yuv420::Nv21, RgbLayout::Bgr).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                for c in 0..3 {
                    assert_eq!(a.at_f64(y, x, c).unwrap(), b.at_f64(y, x, c).unwrap());
                }
            }
        }
    }

    #[test]
    fn i420_and_yv12_differ_only_in_plane_order() {
        let mut d = vec![0u8; 4 * 4 * 3 / 2];
        for (i, v) in d[..16].iter_mut().enumerate() {
            *v = (i * 15) as u8;
        }
        d[16..20].copy_from_slice(&[200, 60, 90, 140]); // U plane
        d[20..24].copy_from_slice(&[70, 180, 128, 40]); // V plane
        let i420 = yuv420_to_rgb(&d, 4, 4, Yuv420::I420, RgbLayout::Bgr).unwrap();
        let mut sw = d.clone();
        sw[16..20].copy_from_slice(&d[20..24]);
        sw[20..24].copy_from_slice(&d[16..20]);
        let yv12 = yuv420_to_rgb(&sw, 4, 4, Yuv420::Yv12, RgbLayout::Bgr).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                for c in 0..3 {
                    assert_eq!(i420.at_f64(y, x, c).unwrap(), yv12.at_f64(y, x, c).unwrap());
                }
            }
        }
    }

    #[test]
    fn rgb_and_bgr_layouts_are_channel_reversals() {
        let d = nv12_frame();
        let bgr = yuv420_to_rgb(&d, 4, 4, Yuv420::Nv12, RgbLayout::Bgr).unwrap();
        let rgb = yuv420_to_rgb(&d, 4, 4, Yuv420::Nv12, RgbLayout::Rgb).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                for c in 0..3 {
                    assert_eq!(
                        bgr.at_f64(y, x, c).unwrap(),
                        rgb.at_f64(y, x, 2 - c).unwrap(),
                        "BGR and RGB must be mirror images"
                    );
                }
            }
        }
    }

    #[test]
    fn the_alpha_layouts_add_an_opaque_channel() {
        let d = nv12_frame();
        let bgr = yuv420_to_rgb(&d, 4, 4, Yuv420::Nv12, RgbLayout::Bgr).unwrap();
        let bgra = yuv420_to_rgb(&d, 4, 4, Yuv420::Nv12, RgbLayout::Bgra).unwrap();
        assert_eq!(bgra.channels(), 4);
        for y in 0..4 {
            for x in 0..4 {
                for c in 0..3 {
                    assert_eq!(bgr.at_f64(y, x, c).unwrap(), bgra.at_f64(y, x, c).unwrap());
                }
                assert_eq!(bgra.at_f64(y, x, 3).unwrap(), 255.0, "alpha must be opaque");
            }
        }
    }

    #[test]
    fn sub_black_luma_clamps_to_black_rather_than_wrapping() {
        // Luma below 16 is below broadcast black; the clamp is before the
        // multiply, so every value 0..16 gives the same result.
        let mut d = vec![128u8; 4 * 4 * 3 / 2];
        for (i, v) in d[..16].iter_mut().enumerate() {
            *v = (i % 16) as u8;
        }
        let m = yuv420_to_rgb(&d, 4, 4, Yuv420::Nv12, RgbLayout::Bgr).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(m.at_f64(y, x, 0).unwrap(), 0.0, "sub-black must be black");
            }
        }
    }

    #[test]
    fn packed_422_layouts_agree_on_the_same_pixels() {
        // Build the same picture in all three packings.
        let (y0, y1, u, v) = (90u8, 200u8, 70u8, 190u8);
        let yuy2: Vec<u8> = [y0, u, y1, v].repeat(4);
        let uyvy: Vec<u8> = [u, y0, v, y1].repeat(4);
        let yvyu: Vec<u8> = [y0, v, y1, u].repeat(4);
        let a = yuv422_to_rgb(&yuy2, 4, 2, Yuv422::Yuy2, RgbLayout::Bgr).unwrap();
        let b = yuv422_to_rgb(&uyvy, 4, 2, Yuv422::Uyvy, RgbLayout::Bgr).unwrap();
        let c = yuv422_to_rgb(&yvyu, 4, 2, Yuv422::Yvyu, RgbLayout::Bgr).unwrap();
        for y in 0..2 {
            for x in 0..4 {
                for ch in 0..3 {
                    let va = a.at_f64(y, x, ch).unwrap();
                    assert_eq!(va, b.at_f64(y, x, ch).unwrap(), "YUY2 vs UYVY");
                    assert_eq!(va, c.at_f64(y, x, ch).unwrap(), "YUY2 vs YVYU");
                }
            }
        }
        // And the two luma samples in a pair really do differ.
        assert_ne!(a.at_f64(0, 0, 1).unwrap(), a.at_f64(0, 1, 1).unwrap());
    }

    #[test]
    fn packed_422_grey_is_the_luma_samples() {
        let yuy2: Vec<u8> = [10u8, 128, 250, 128].repeat(2);
        let g = yuv422_to_rgb(&yuy2, 4, 1, Yuv422::Yuy2, RgbLayout::Gray).unwrap();
        assert_eq!(
            (0..4).map(|x| g.at_f64(0, x, 0).unwrap()).collect::<Vec<_>>(),
            vec![10.0, 250.0, 10.0, 250.0]
        );
    }

    #[test]
    fn the_forward_direction_round_trips_within_subsampling_error() {
        // A flat colour survives 4:2:0 exactly: there is nothing to lose.
        let src = Mat::from_vec(4, 4, 3, &[40u8, 150, 90].repeat(16)).unwrap();
        let buf = rgb_to_yuv420(&src, Yuv420::I420, false).unwrap();
        assert_eq!(buf.len(), 4 * 4 * 3 / 2);
        let back = yuv420_to_rgb(&buf, 4, 4, Yuv420::I420, RgbLayout::Bgr).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                for c in 0..3 {
                    let (a, b) = (src.at_f64(y, x, c).unwrap(), back.at_f64(y, x, c).unwrap());
                    assert!((a - b).abs() <= 2.0, "({y},{x}) channel {c}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn i420_and_yv12_write_the_planes_in_opposite_order() {
        let src = Mat::from_vec(2, 2, 3, &[10u8, 200, 30, 40, 60, 220, 90, 10, 10, 5, 5, 250])
            .unwrap();
        let i = rgb_to_yuv420(&src, Yuv420::I420, false).unwrap();
        let y = rgb_to_yuv420(&src, Yuv420::Yv12, false).unwrap();
        assert_eq!(i[..4], y[..4], "the luma plane does not depend on chroma order");
        assert_eq!(i[4], y[5], "U and V swap places");
        assert_eq!(i[5], y[4]);
    }

    #[test]
    fn conversions_validate_their_input() {
        let d = vec![128u8; 100];
        // Odd dimensions have no consistent 4:2:0 layout.
        assert!(yuv420_to_rgb(&d, 5, 4, Yuv420::Nv12, RgbLayout::Bgr).is_err());
        assert!(yuv420_to_rgb(&d, 4, 5, Yuv420::Nv12, RgbLayout::Bgr).is_err());
        assert!(yuv420_to_rgb(&d, 0, 4, Yuv420::Nv12, RgbLayout::Bgr).is_err());
        // A short buffer is caught rather than read past.
        assert!(yuv420_to_rgb(&d[..10], 4, 4, Yuv420::Nv12, RgbLayout::Bgr).is_err());
        assert!(yuv422_to_rgb(&d[..3], 4, 4, Yuv422::Yuy2, RgbLayout::Bgr).is_err());
        assert!(yuv422_to_rgb(&d, 3, 4, Yuv422::Yuy2, RgbLayout::Bgr).is_err());
        // The forward direction only has planar targets upstream.
        let src = Mat::from_vec(2, 2, 3, &[0u8; 12]).unwrap();
        assert!(rgb_to_yuv420(&src, Yuv420::Nv12, false).is_err());
        let odd = Mat::from_vec(3, 2, 3, &[0u8; 18]).unwrap();
        assert!(rgb_to_yuv420(&odd, Yuv420::I420, false).is_err());
        let gray = Mat::from_vec(2, 2, 1, &[0u8; 4]).unwrap();
        assert!(rgb_to_yuv420(&gray, Yuv420::I420, false).is_err());
    }


    /// Bit-exact fixtures recorded from cv2 4.13.0 on a 6x4 frame.
    ///
    /// A wider sweep (six sizes x thirteen conversions) matched cv2 exactly on
    /// every byte; these pin a representative slice of it so a regression in
    /// the fixed-point pipeline shows up here rather than in a video.
    #[test]
    fn conversions_are_bit_exact_against_cv2() {
        const FRAME: [u8; 36] = [166, 8, 130, 121, 247, 161, 65, 200, 2, 67, 25, 219, 145, 141, 230, 171, 255, 123, 113, 26, 177, 254, 196, 60, 190, 235, 108, 182, 112, 45, 241, 45, 107, 185, 171, 217];
        const PACKED: [u8; 48] = [27, 84, 188, 88, 91, 44, 119, 7, 39, 174, 240, 38, 97, 148, 216, 33, 95, 216, 185, 79, 163, 12, 181, 241, 1, 136, 47, 61, 208, 24, 135, 235, 81, 146, 38, 139, 80, 240, 84, 239, 242, 241, 238, 234, 84, 55, 157, 7];
        const BGR: [u8; 72] = [57, 85, 187, 113, 71, 37, 226, 137, 85, 250, 161, 194, 141, 189, 107, 187, 176, 153, 69, 147, 181, 232, 207, 167, 114, 214, 10, 131, 5, 74, 3, 11, 186, 167, 59, 168, 149, 114, 64, 211, 97, 230, 92, 242, 190, 176, 201, 113, 185, 87, 81, 234, 98, 42, 124, 14, 131, 76, 97, 184, 190, 190, 28, 212, 94, 214, 108, 31, 125, 26, 142, 68];

        let flat = |m: &Mat| {
            let mut v = Vec::new();
            for y in 0..m.rows() {
                for x in 0..m.cols() {
                    for c in 0..m.channels() {
                        v.push(m.at_f64(y, x, c).unwrap() as u8);
                    }
                }
            }
            v
        };

        // COLOR_YUV2BGR_NV12
        assert_eq!(
            flat(&yuv420_to_rgb(&FRAME, 6, 4, Yuv420::Nv12, RgbLayout::Bgr).unwrap()),
            vec![255, 63, 255, 125, 0, 171, 92, 97, 219, 82, 86, 208, 237, 255, 136, 136, 243, 36, 182, 0, 228, 255, 103, 255, 0, 0, 86, 19, 23, 146, 0, 84, 0, 204, 255, 104, 255, 173, 18, 255, 169, 13, 207, 211, 255, 138, 142, 255, 255, 189, 255, 211, 35, 255, 255, 136, 0, 240, 35, 0, 145, 149, 255, 235, 239, 255, 255, 120, 255, 138, 0, 193]
        );
        // COLOR_YUV2BGR_I420
        assert_eq!(
            flat(&yuv420_to_rgb(&FRAME, 6, 4, Yuv420::I420, RgbLayout::Bgr).unwrap()),
            vec![255, 58, 255, 125, 0, 180, 255, 158, 0, 255, 148, 0, 229, 255, 235, 128, 194, 135, 182, 0, 237, 255, 98, 255, 216, 26, 0, 255, 85, 0, 0, 35, 0, 196, 255, 203, 255, 83, 241, 254, 78, 236, 217, 220, 255, 148, 152, 249, 111, 238, 255, 0, 85, 255, 222, 45, 204, 121, 0, 103, 155, 159, 255, 245, 248, 255, 42, 170, 255, 0, 11, 193]
        );
        // COLOR_YUV2BGR_YUY2
        assert_eq!(
            flat(&yuv422_to_rgb(&PACKED, 6, 4, Yuv422::Yuy2, RgbLayout::Bgr).unwrap()),
            vec![0, 63, 0, 111, 250, 136, 0, 219, 0, 0, 251, 0, 120, 82, 0, 255, 255, 117, 135, 164, 0, 255, 255, 81, 255, 97, 14, 255, 202, 119, 0, 125, 255, 0, 146, 255, 16, 51, 0, 52, 87, 0, 14, 177, 255, 0, 92, 255, 112, 60, 93, 62, 10, 43, 255, 0, 252, 255, 0, 255, 255, 133, 255, 255, 128, 255, 0, 206, 0, 17, 255, 0]
        );
        // COLOR_BGR2YUV_I420
        let src = Mat::from_vec(4, 6, 3, &BGR).unwrap();
        assert_eq!(
            rgb_to_yuv420(&src, Yuv420::I420, false).unwrap(),
            vec![112, 72, 129, 172, 153, 162, 143, 186, 138, 50, 70, 105, 105, 145, 196, 164, 99, 99, 69, 120, 138, 139, 74, 108, 101, 175, 119, 151, 70, 172, 175, 99, 95, 104, 116, 118]
        );
    }

    #[test]
    fn two_plane_matches_the_single_buffer_form() {
        let d = nv12_frame();
        let joined = yuv420_to_rgb(&d, 4, 4, Yuv420::Nv12, RgbLayout::Bgr).unwrap();
        let split = cvt_color_two_plane(&d[..16], &d[16..], 4, 4, Yuv420::Nv12, RgbLayout::Bgr)
            .unwrap();
        for y in 0..4 {
            for x in 0..4 {
                for c in 0..3 {
                    assert_eq!(joined.at_f64(y, x, c).unwrap(), split.at_f64(y, x, c).unwrap());
                }
            }
        }
        // The planar layouts are not two-plane.
        assert!(
            cvt_color_two_plane(&d[..16], &d[16..], 4, 4, Yuv420::I420, RgbLayout::Bgr).is_err()
        );
    }
}
