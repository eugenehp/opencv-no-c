//! Histogram comparison, back projection and CLAHE
//! (`modules/imgproc/src/histogram.cpp`, `clahe.cpp`).

use cv_core::types::{CV_32FC1, CV_8UC1};
use cv_core::{cv_assert, Mat, MatLike, MatLikeMut, Result};

/// `cv::HistCompMethods`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistCompare {
    /// `cv::HISTCMP_CORREL` — higher is more similar, 1 is identical.
    Correlation,
    /// `cv::HISTCMP_CHISQR` — lower is more similar, 0 is identical.
    ChiSquare,
    /// `cv::HISTCMP_INTERSECT` — higher is more similar.
    Intersection,
    /// `cv::HISTCMP_BHATTACHARYYA` — lower is more similar, 0 is identical.
    Bhattacharyya,
    /// `cv::HISTCMP_CHISQR_ALT` — the symmetric chi-square used for texture.
    ChiSquareAlt,
    /// `cv::HISTCMP_KL_DIV` — Kullback-Leibler divergence, asymmetric.
    KlDiv,
}

/// `cv::compareHist(H1, H2, method)`.
///
/// # Divergence guards
///
/// Upstream skips a bin when the *first* histogram's entry is zero for
/// `CHISQR` and `KL_DIV`, which makes those two asymmetric in a way that is
/// easy to mistake for a bug. That behaviour is kept, because a caller
/// comparing against a reference histogram relies on it.
pub fn compare_hist(h1: &[f64], h2: &[f64], method: HistCompare) -> Result<f64> {
    cv_assert!(
        h1.len() == h2.len(),
        "compareHist got {} and {} bins",
        h1.len(),
        h2.len()
    );
    cv_assert!(!h1.is_empty(), "compareHist needs at least one bin");
    let n = h1.len() as f64;

    Ok(match method {
        HistCompare::Correlation => {
            let (m1, m2) = (h1.iter().sum::<f64>() / n, h2.iter().sum::<f64>() / n);
            let (mut num, mut d1, mut d2) = (0.0, 0.0, 0.0);
            for (a, b) in h1.iter().zip(h2) {
                let (x, y) = (a - m1, b - m2);
                num += x * y;
                d1 += x * x;
                d2 += y * y;
            }
            let den = (d1 * d2).sqrt();
            if den > 0.0 {
                num / den
            } else {
                1.0
            }
        }
        HistCompare::ChiSquare => {
            let mut acc = 0.0;
            for (a, b) in h1.iter().zip(h2) {
                if *a == 0.0 {
                    continue; // upstream's guard; the metric is asymmetric
                }
                let d = a - b;
                acc += d * d / a;
            }
            acc
        }
        HistCompare::ChiSquareAlt => {
            let mut acc = 0.0;
            for (a, b) in h1.iter().zip(h2) {
                let s = a + b;
                if s == 0.0 {
                    continue;
                }
                let d = a - b;
                acc += d * d / s;
            }
            2.0 * acc
        }
        HistCompare::Intersection => h1.iter().zip(h2).map(|(a, b)| a.min(*b)).sum(),
        HistCompare::Bhattacharyya => {
            let (s1, s2) = (h1.iter().sum::<f64>(), h2.iter().sum::<f64>());
            let mut acc = 0.0;
            for (a, b) in h1.iter().zip(h2) {
                acc += (a * b).max(0.0).sqrt();
            }
            let den = (s1 * s2).sqrt();
            // 1 - correlation, floored at 0 to absorb rounding just below zero.
            let inner = if den > 0.0 { 1.0 - acc / den } else { 1.0 };
            inner.max(0.0).sqrt()
        }
        HistCompare::KlDiv => {
            let mut acc = 0.0;
            for (a, b) in h1.iter().zip(h2) {
                if *a == 0.0 {
                    continue;
                }
                // A zero in the second histogram makes the divergence infinite;
                // upstream substitutes a tiny value rather than returning inf.
                let bb = if *b == 0.0 { 1e-10 } else { *b };
                acc += a * (a / bb).ln();
            }
            acc
        }
    })
}

/// `cv::calcBackProject(images, channels, hist, backProject, ranges, scale)`.
///
/// Single-channel, 256-bin form: every pixel is replaced by the histogram
/// weight of its own value. The result is `CV_8U`, saturating.
pub fn calc_back_project<S: MatLike + ?Sized>(src: &S, hist: &[f64], scale: f64) -> Result<Mat> {
    cv_assert!(src.channels() == 1, "calcBackProject needs a single-channel image");
    cv_assert!(!hist.is_empty(), "the histogram must have at least one bin");
    let bins = hist.len() as f64;
    let mut out = Mat::zeros(src.rows(), src.cols(), CV_8UC1)?;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            let v = src.at_f64(y, x, 0)?;
            // Values run 0..256; map into the bin count.
            let b = ((v * bins / 256.0) as usize).min(hist.len() - 1);
            out.pixel_mut::<u8>(y, x)?[0] = cv_core::saturate::Elem::from_f64(hist[b] * scale);
        }
    }
    Ok(out)
}

/// `cv::createCLAHE(clipLimit, tileGridSize)` then `apply`.
///
/// Contrast-limited adaptive histogram equalisation: equalise each tile
/// separately, clip the histogram peaks and redistribute the excess, then
/// bilinearly interpolate between tile mappings so the tiles do not show.
///
/// `CV_8UC1` only, as upstream's is for the 8-bit path.
pub fn clahe<S: MatLike + ?Sized>(
    src: &S,
    clip_limit: f64,
    tiles_x: i32,
    tiles_y: i32,
) -> Result<Mat> {
    cv_assert!(src.mat_type() == CV_8UC1, "CLAHE requires CV_8UC1");
    cv_assert!(tiles_x > 0 && tiles_y > 0, "tile grid must be positive");
    let (h, w) = (src.rows(), src.cols());
    cv_assert!(h >= tiles_y && w >= tiles_x, "image is smaller than the tile grid");

    let (tw, th) = (w / tiles_x, h / tiles_y);
    let tile_total = tw * th;
    // `255 / tileArea`, **not** `255 / sum(hist)`. After the clip-and-
    // redistribute the histogram still sums to the tile area, so the two agree
    // in exact arithmetic — but the redistribution below is integer and its
    // residual pass makes the sum differ, and dividing by the actual sum then
    // scales the whole curve slightly.
    let lut_scale = 255.0 / f64::from(tile_total);
    // An **integer** clip limit, floored, with a minimum of one.
    let clip = if clip_limit > 0.0 {
        ((clip_limit * f64::from(tile_total) / 256.0) as i32).max(1)
    } else {
        0
    };

    // One 256-entry lookup table per tile.
    let mut luts = vec![[0u8; 256]; (tiles_x * tiles_y) as usize];
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let (x0, y0) = (tx * tw, ty * th);
            let (x1, y1) = (x0 + tw, y0 + th);
            let mut hist = [0i32; 256];
            for y in y0..y1 {
                for x in x0..x1 {
                    hist[src.at_f64(y, x, 0)? as usize] += 1;
                }
            }
            if clip > 0 {
                let mut clipped = 0i32;
                for b in hist.iter_mut() {
                    if *b > clip {
                        clipped += *b - clip;
                        *b = clip;
                    }
                }
                // Integer redistribution: every bin gets the same whole share,
                // and the remainder is handed out one at a time at a stride
                // across the range. Spreading the remainder as a fraction
                // instead is smoother and is not what the LUT is built from.
                let batch = clipped / 256;
                let mut residual = clipped - batch * 256;
                for b in hist.iter_mut() {
                    *b += batch;
                }
                if residual != 0 {
                    let step = (256 / residual).max(1);
                    let mut i2 = 0usize;
                    while i2 < 256 && residual > 0 {
                        hist[i2] += 1;
                        residual -= 1;
                        i2 += step as usize;
                    }
                }
            }
            let mut acc = 0i32;
            let lut = &mut luts[(ty * tiles_x + tx) as usize];
            for (i2, &c) in hist.iter().enumerate() {
                acc += c;
                lut[i2] = cv_core::saturate::cv_round(f64::from(acc) * lut_scale).clamp(0, 255) as u8;
            }
        }
    }

    // Bilinear blend between the four surrounding tile centres.
    let mut out = Mat::zeros(h, w, CV_8UC1)?;
    let (inv_th, inv_tw) = (1.0f32 / th as f32, 1.0f32 / tw as f32);
    for y in 0..h {
        // The blend weight comes from the **unclamped** tile index. Clamping
        // first and then taking the fraction gives weight zero in the outer
        // half-tile, so the border blends with nothing; upstream clamps the
        // index it reads and keeps the weight it computed.
        let tyf = y as f32 * inv_th - 0.5;
        let ty1 = tyf.floor() as i32;
        let ya = tyf - ty1 as f32;
        let (ty1, ty2) = (ty1.max(0), (ty1 + 1).min(tiles_y - 1));
        for x in 0..w {
            let txf = x as f32 * inv_tw - 0.5;
            let tx1 = txf.floor() as i32;
            let xa = txf - tx1 as f32;
            let (tx1, tx2) = (tx1.max(0), (tx1 + 1).min(tiles_x - 1));

            let v = src.at_f64(y, x, 0)? as usize;
            // The blend is evaluated in f32 upstream, and the rounding at the
            // end sits close enough to a half that the width matters.
            let g = |ty: i32, tx: i32| f32::from(luts[(ty * tiles_x + tx) as usize][v]);
            let top = g(ty1, tx1) * (1.0 - xa) + g(ty1, tx2) * xa;
            let bot = g(ty2, tx1) * (1.0 - xa) + g(ty2, tx2) * xa;
            let res = top * (1.0 - ya) + bot * ya;
            out.pixel_mut::<u8>(y, x)?[0] =
                cv_core::saturate::cv_roundf(res).clamp(0, 255) as u8;
        }
    }
    Ok(out)
}

/// `cv::COLORMAP_*` — the palettes [`apply_color_map`] understands.
///
/// Every entry of every palette is transcribed from OpenCV; see
/// [`crate::colormap_data`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMap {
    /// `cv::COLORMAP_AUTUMN`
    Autumn,
    /// `cv::COLORMAP_BONE`
    Bone,
    /// `cv::COLORMAP_JET`
    Jet,
    /// `cv::COLORMAP_WINTER`
    Winter,
    /// `cv::COLORMAP_RAINBOW`
    Rainbow,
    /// `cv::COLORMAP_OCEAN`
    Ocean,
    /// `cv::COLORMAP_SUMMER`
    Summer,
    /// `cv::COLORMAP_SPRING`
    Spring,
    /// `cv::COLORMAP_COOL`
    Cool,
    /// `cv::COLORMAP_HSV`
    Hsv,
    /// `cv::COLORMAP_PINK`
    Pink,
    /// `cv::COLORMAP_HOT`
    Hot,
    /// `cv::COLORMAP_PARULA`
    Parula,
    /// `cv::COLORMAP_MAGMA`
    Magma,
    /// `cv::COLORMAP_INFERNO`
    Inferno,
    /// `cv::COLORMAP_PLASMA`
    Plasma,
    /// `cv::COLORMAP_VIRIDIS`
    Viridis,
    /// `cv::COLORMAP_CIVIDIS`
    Cividis,
    /// `cv::COLORMAP_TWILIGHT`
    Twilight,
    /// `cv::COLORMAP_TWILIGHT_SHIFTED`
    TwilightShifted,
    /// `cv::COLORMAP_TURBO`
    Turbo,
    /// `cv::COLORMAP_DEEPGREEN`
    Deepgreen,
    /// A plain grey ramp.
    ///
    /// Not an upstream palette — OpenCV has no `COLORMAP_GRAY` — but useful
    /// as an identity when a call site is parameterised over a colour map.
    Gray,
}

impl ColorMap {
    /// The 256-entry BGR table this palette applies.
    #[must_use]
    pub fn palette(self) -> &'static crate::colormap_data::Palette {
        use crate::colormap_data as d;
        match self {
            ColorMap::Autumn => &d::AUTUMN,
            ColorMap::Bone => &d::BONE,
            ColorMap::Jet => &d::JET,
            ColorMap::Winter => &d::WINTER,
            ColorMap::Rainbow => &d::RAINBOW,
            ColorMap::Ocean => &d::OCEAN,
            ColorMap::Summer => &d::SUMMER,
            ColorMap::Spring => &d::SPRING,
            ColorMap::Cool => &d::COOL,
            ColorMap::Hsv => &d::HSV,
            ColorMap::Pink => &d::PINK,
            ColorMap::Hot => &d::HOT,
            ColorMap::Parula => &d::PARULA,
            ColorMap::Magma => &d::MAGMA,
            ColorMap::Inferno => &d::INFERNO,
            ColorMap::Plasma => &d::PLASMA,
            ColorMap::Viridis => &d::VIRIDIS,
            ColorMap::Cividis => &d::CIVIDIS,
            ColorMap::Twilight => &d::TWILIGHT,
            ColorMap::TwilightShifted => &d::TWILIGHT_SHIFTED,
            ColorMap::Turbo => &d::TURBO,
            ColorMap::Deepgreen => &d::DEEPGREEN,
            ColorMap::Gray => &GRAY_RAMP,
        }
    }

    /// Every upstream palette, for callers that want to enumerate them.
    #[must_use]
    pub fn all() -> [ColorMap; 22] {
        [
            ColorMap::Autumn,
            ColorMap::Bone,
            ColorMap::Jet,
            ColorMap::Winter,
            ColorMap::Rainbow,
            ColorMap::Ocean,
            ColorMap::Summer,
            ColorMap::Spring,
            ColorMap::Cool,
            ColorMap::Hsv,
            ColorMap::Pink,
            ColorMap::Hot,
            ColorMap::Parula,
            ColorMap::Magma,
            ColorMap::Inferno,
            ColorMap::Plasma,
            ColorMap::Viridis,
            ColorMap::Cividis,
            ColorMap::Twilight,
            ColorMap::TwilightShifted,
            ColorMap::Turbo,
            ColorMap::Deepgreen,
        ]
    }
}

/// The identity ramp backing [`ColorMap::Gray`].
static GRAY_RAMP: crate::colormap_data::Palette = {
    let mut t = [[0u8; 3]; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = [i as u8, i as u8, i as u8];
        i += 1;
    }
    t
};

/// `cv::applyColorMap(src, dst, colormap)`.
///
/// The input is taken as a level in `0..=255`; anything above saturates, as it
/// does upstream. Output is `CV_8UC3` BGR.
pub fn apply_color_map<S: MatLike + ?Sized>(src: &S, cm: ColorMap) -> Result<Mat> {
    cv_assert!(src.channels() == 1, "applyColorMap needs a single-channel image");
    use cv_core::types::{Depth, MatType};
    let lut = cm.palette();
    let mut out = Mat::zeros(src.rows(), src.cols(), MatType::new(Depth::U8, 3))?;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            let v = (src.at_f64(y, x, 0)?.round().clamp(0.0, 255.0)) as usize;
            let e = lut[v];
            let p = out.pixel_mut::<u8>(y, x)?;
            p[0] = e[0];
            p[1] = e[1];
            p[2] = e[2];
        }
    }
    Ok(out)
}

/// `cv::accumulate(src, dst)` — `dst += src`.
pub fn accumulate<S: MatLike + ?Sized, D: MatLike + MatLikeMut + ?Sized>(
    src: &S,
    dst: &mut D,
) -> Result<()> {
    accumulate_with(src, dst, |a, s| a + s)
}

/// `cv::accumulateSquare(src, dst)` — `dst += src^2`.
pub fn accumulate_square<S: MatLike + ?Sized, D: MatLike + MatLikeMut + ?Sized>(
    src: &S,
    dst: &mut D,
) -> Result<()> {
    accumulate_with(src, dst, |a, s| a + s * s)
}

/// `cv::accumulateWeighted(src, dst, alpha)` — a running average.
pub fn accumulate_weighted<S: MatLike + ?Sized, D: MatLike + MatLikeMut + ?Sized>(
    src: &S,
    dst: &mut D,
    alpha: f64,
) -> Result<()> {
    accumulate_with(src, dst, move |a, s| a * (1.0 - alpha) + s * alpha)
}

/// `cv::accumulateProduct(src1, src2, dst)` — `dst += src1*src2`.
pub fn accumulate_product<A: MatLike + ?Sized, B: MatLike + ?Sized, D: MatLike + MatLikeMut + ?Sized>(
    a: &A,
    b: &B,
    dst: &mut D,
) -> Result<()> {
    cv_assert!(a.size() == b.size(), "accumulateProduct operands must match in size");
    cv_assert!(a.size() == dst.size(), "the accumulator must match the source size");
    let cn = a.channels();
    for y in 0..a.rows() {
        for x in 0..a.cols() {
            for c in 0..cn {
                let v = dst.at_f64(y, x, c)? + a.at_f64(y, x, c)? * b.at_f64(y, x, c)?;
                cv_core::arithm::set_channel_generic(dst, y, x, c, v)?;
            }
        }
    }
    Ok(())
}

/// Shared body of the accumulate family.
fn accumulate_with<S: MatLike + ?Sized, D: MatLike + MatLikeMut + ?Sized>(
    src: &S,
    dst: &mut D,
    f: impl Fn(f64, f64) -> f64,
) -> Result<()> {
    cv_assert!(
        src.size() == dst.size(),
        "the accumulator is {}x{} but the source is {}x{}",
        dst.cols(),
        dst.rows(),
        src.cols(),
        src.rows()
    );
    cv_assert!(
        src.channels() == dst.channels(),
        "channel counts differ: {} vs {}",
        src.channels(),
        dst.channels()
    );
    cv_assert!(
        matches!(dst.depth(), cv_core::types::Depth::F32 | cv_core::types::Depth::F64),
        "the accumulator must be CV_32F or CV_64F, got {}",
        dst.depth().name()
    );
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            for c in 0..src.channels() {
                let v = f(dst.at_f64(y, x, c)?, src.at_f64(y, x, c)?);
                cv_core::arithm::set_channel_generic(dst, y, x, c, v)?;
            }
        }
    }
    Ok(())
}

/// `cv::getRectSubPix(image, patchSize, center, patch)`.
///
/// Bilinear sampling around a sub-pixel centre; out-of-range taps use the
/// nearest edge pixel, matching upstream's replicate behaviour.
pub fn get_rect_sub_pix<S: MatLike + ?Sized>(
    src: &S,
    width: i32,
    height: i32,
    center: (f64, f64),
) -> Result<Mat> {
    cv_assert!(width > 0 && height > 0, "patch size must be positive");
    let (h, w) = (src.rows(), src.cols());
    let cn = src.channels();
    let mut out = Mat::zeros(height, width, src.mat_type())?;
    // Upstream's origin: the patch centre lands on `center`.
    let x0 = center.0 - (width as f64 - 1.0) * 0.5;
    let y0 = center.1 - (height as f64 - 1.0) * 0.5;
    for j in 0..height {
        for i in 0..width {
            let (fx, fy) = (x0 + i as f64, y0 + j as f64);
            let (ix, iy) = (fx.floor() as i32, fy.floor() as i32);
            let (ax, ay) = (fx - ix as f64, fy - iy as f64);
            for c in 0..cn {
                let at = |yy: i32, xx: i32| -> Result<f64> {
                    src.at_f64(yy.clamp(0, h - 1), xx.clamp(0, w - 1), c)
                };
                let v = at(iy, ix)? * (1.0 - ax) * (1.0 - ay)
                    + at(iy, ix + 1)? * ax * (1.0 - ay)
                    + at(iy + 1, ix)? * (1.0 - ax) * ay
                    + at(iy + 1, ix + 1)? * ax * ay;
                cv_core::arithm::set_channel_generic(&mut out, j, i, c, v)?;
            }
        }
    }
    Ok(out)
}

/// `cv::WarpPolarMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolarMode {
    /// `cv::WARP_POLAR_LINEAR` — radius maps linearly.
    Linear,
    /// `cv::WARP_POLAR_LOG` — radius maps logarithmically.
    Log,
}

/// `cv::warpPolar(src, dst, dsize, center, maxRadius, flags)`.
///
/// The output's x axis is radius and its y axis is angle over a full turn.
pub fn warp_polar<S: MatLike + ?Sized>(
    src: &S,
    width: i32,
    height: i32,
    center: (f64, f64),
    max_radius: f64,
    mode: PolarMode,
) -> Result<Mat> {
    cv_assert!(width > 0 && height > 0, "output size must be positive");
    cv_assert!(max_radius > 0.0, "maxRadius must be positive, got {max_radius}");

    // Upstream builds two **f32** coordinate maps and hands them to `remap`.
    // Doing the interpolation here instead looks equivalent and is not:
    // `remap` carries its own rounding rule and its own border handling, and
    // both show up in the output. Note the deliberate `f32` round-trip on the
    // radius — upstream stores `bufRhos` as float and promotes it back.
    let mut map_x = Mat::zeros(height, width, cv_core::types::CV_32FC1)?;
    let mut map_y = Mat::zeros(height, width, cv_core::types::CV_32FC1)?;
    let k_angle = core::f64::consts::TAU / f64::from(height);
    let rhos: Vec<f32> = match mode {
        PolarMode::Linear => {
            let k = max_radius / f64::from(width);
            (0..width).map(|i| (f64::from(i) * k) as f32).collect()
        }
        PolarMode::Log => {
            let k = max_radius.ln() / f64::from(width);
            // `exp(x) - 1`, not `exp_m1(x)`: they differ in the last bits.
            (0..width).map(|i| (f64::from(i) * k).exp() - 1.0).map(|v| v as f32).collect()
        }
    };
    for phi in 0..height {
        let a = k_angle * f64::from(phi);
        let (sp, cp) = (a.sin(), a.cos());
        for rho in 0..width {
            let r = f64::from(rhos[rho as usize]);
            map_x.pixel_mut::<f32>(phi, rho)?[0] = (r * cp + center.0) as f32;
            map_y.pixel_mut::<f32>(phi, rho)?[0] = (r * sp + center.1) as f32;
        }
    }
    crate::warp::remap(
        src,
        &map_x,
        &map_y,
        crate::Interpolation::Linear,
        cv_core::BorderType::Constant,
        cv_core::Scalar::all(0.0),
    )
}

/// `cv::linearPolar(src, dst, center, maxRadius, flags)`.
pub fn linear_polar<S: MatLike + ?Sized>(
    src: &S,
    center: (f64, f64),
    max_radius: f64,
) -> Result<Mat> {
    warp_polar(src, src.cols(), src.rows(), center, max_radius, PolarMode::Linear)
}

/// `cv::logPolar(src, dst, center, M, flags)`.
pub fn log_polar<S: MatLike + ?Sized>(src: &S, center: (f64, f64), max_radius: f64) -> Result<Mat> {
    warp_polar(src, src.cols(), src.rows(), center, max_radius, PolarMode::Log)
}

/// `cv::blendLinear(src1, src2, weights1, weights2, dst)`.
pub fn blend_linear<A: MatLike + ?Sized, B: MatLike + ?Sized, W: MatLike + ?Sized>(
    a: &A,
    b: &B,
    w1: &W,
    w2: &W,
) -> Result<Mat> {
    cv_assert!(a.size() == b.size(), "blendLinear sources must match in size");
    cv_assert!(a.size() == w1.size() && a.size() == w2.size(), "weights must match the sources");
    let cn = a.channels();
    let mut out = Mat::zeros(a.rows(), a.cols(), a.mat_type())?;
    for y in 0..a.rows() {
        for x in 0..a.cols() {
            let (p, q) = (w1.at_f64(y, x, 0)?, w2.at_f64(y, x, 0)?);
            let den = p + q;
            for c in 0..cn {
                let v = if den > 0.0 {
                    (a.at_f64(y, x, c)? * p + b.at_f64(y, x, c)? * q) / den
                } else {
                    0.0
                };
                cv_core::arithm::set_channel_generic(&mut out, y, x, c, v)?;
            }
        }
    }
    Ok(out)
}

/// `cv::sqrBoxFilter(src, dst, ddepth, ksize, anchor, normalize, borderType)`.
pub fn sqr_box_filter<S: MatLike + ?Sized>(src: &S, ksize: i32, normalize: bool) -> Result<Mat> {
    cv_assert!(ksize > 0 && ksize % 2 == 1, "ksize must be odd and positive, got {ksize}");
    cv_assert!(src.channels() == 1, "sqrBoxFilter here is single-channel");
    let (h, w) = (src.rows(), src.cols());
    let r = ksize / 2;
    let area = (ksize * ksize) as f64;
    let mut out = Mat::zeros(h, w, CV_32FC1)?;
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for dy in -r..=r {
                for dx in -r..=r {
                    // BORDER_REFLECT_101, upstream's default.
                    let sy = cv_core::reflect101(y + dy, h);
                    let sx = cv_core::reflect101(x + dx, w);
                    let v = src.at_f64(sy, sx, 0)?;
                    acc += v * v;
                }
            }
            out.pixel_mut::<f32>(y, x)?[0] = if normalize { (acc / area) as f32 } else { acc as f32 };
        }
    }
    Ok(out)
}


/// `cv::bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType)`.
///
/// Edge-preserving: the spatial Gaussian is multiplied by a range Gaussian on
/// the intensity difference, so a strong edge stops the smoothing across it.
pub fn bilateral_filter<S: MatLike + ?Sized>(
    src: &S,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
) -> Result<Mat> {
    cv_assert!(d > 0, "diameter must be positive, got {d}");
    let sigma_color = if sigma_color <= 0.0 { 1.0 } else { sigma_color };
    let sigma_space = if sigma_space <= 0.0 { 1.0 } else { sigma_space };
    let (h, w) = (src.rows(), src.cols());
    let cn = src.channels();
    let r = d / 2;

    // Precompute the spatial weights once; the range weight depends on the
    // pixel. The support is a **disc**, not the full square — upstream skips
    // every tap with `sqrt(i*i + j*j) > radius`. At `d = 5` that is 13 taps
    // rather than 25, so filling the square blurs far too wide.
    let gs = -0.5 / (sigma_space * sigma_space);
    let mut taps: Vec<(i32, i32, f64)> = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = f64::from(dx * dx + dy * dy);
            if d2.sqrt() > f64::from(r) {
                continue;
            }
            taps.push((dy, dx, (d2 * gs).exp()));
        }
    }
    let gc = -0.5 / (sigma_color * sigma_color);

    let mut out = Mat::zeros(h, w, src.mat_type())?;
    for y in 0..h {
        for x in 0..w {
            let centre: Vec<f64> = (0..cn).map(|c| src.at_f64(y, x, c)).collect::<Result<_>>()?;
            let mut acc = vec![0.0f64; cn as usize];
            let mut wsum = 0.0f64;
            for &(dy, dx, sw) in &taps {
                {
                    let sy = cv_core::reflect101(y + dy, h);
                    let sx = cv_core::reflect101(x + dx, w);
                    // Upstream's colour distance is **L1 across the channels**,
                    // then squared by the weight table lookup — not the squared
                    // L2 the name suggests. For one channel the two agree, so a
                    // grayscale fixture cannot tell them apart; on BGR the
                    // difference is 28 levels on 96% of pixels.
                    let mut l1 = 0.0;
                    for c in 0..cn {
                        l1 += (src.at_f64(sy, sx, c)? - centre[c as usize]).abs();
                    }
                    let d2 = l1 * l1;
                    let wgt = sw * (d2 * gc).exp();
                    for c in 0..cn {
                        acc[c as usize] += wgt * src.at_f64(sy, sx, c)?;
                    }
                    wsum += wgt;
                }
            }
            for c in 0..cn {
                let v = if wsum > 0.0 { acc[c as usize] / wsum } else { centre[c as usize] };
                cv_core::arithm::set_channel_generic(&mut out, y, x, c, v)?;
            }
        }
    }
    Ok(out)
}

/// `cv::stackBlur(src, dst, ksize)`.
///
/// A box blur applied with a triangular weighting, which approximates a
/// Gaussian at a fraction of the cost. Separable: rows then columns.
pub fn stack_blur<S: MatLike + ?Sized>(src: &S, ksize: i32) -> Result<Mat> {
    cv_assert!(ksize > 0 && ksize % 2 == 1, "ksize must be odd and positive, got {ksize}");
    let (h, w) = (src.rows(), src.cols());
    let cn = src.channels();
    let r = ksize / 2;
    // Triangular kernel: 1, 2, ..., r+1, ..., 2, 1.
    let kern: Vec<f64> = (-r..=r).map(|i| (r + 1 - i.abs()) as f64).collect();
    let norm: f64 = kern.iter().sum();

    let mut tmp = vec![0.0f64; (h * w * cn) as usize];
    for y in 0..h {
        for x in 0..w {
            for c in 0..cn {
                let mut acc = 0.0;
                for (k, &kv) in kern.iter().enumerate() {
                    let sx = cv_core::reflect101(x + k as i32 - r, w);
                    acc += kv * src.at_f64(y, sx, c)?;
                }
                tmp[((y * w + x) * cn + c) as usize] = acc / norm;
            }
        }
    }
    let mut out = Mat::zeros(h, w, src.mat_type())?;
    for y in 0..h {
        for x in 0..w {
            for c in 0..cn {
                let mut acc = 0.0;
                for (k, &kv) in kern.iter().enumerate() {
                    let sy = cv_core::reflect101(y + k as i32 - r, h);
                    acc += kv * tmp[((sy * w + x) * cn + c) as usize];
                }
                cv_core::arithm::set_channel_generic(&mut out, y, x, c, acc / norm)?;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::{CV_32FC1, CV_8UC1};

    fn ramp(w: i32, h: i32) -> Mat {
        let mut m = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                m.pixel_mut::<u8>(y, x).unwrap()[0] = ((x * 7 + y * 13) % 256) as u8;
            }
        }
        m
    }

    #[test]
    fn compare_hist_scores_identical_histograms_perfectly() {
        let h = vec![1.0, 2.0, 3.0, 4.0];
        assert!((compare_hist(&h, &h, HistCompare::Correlation).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(compare_hist(&h, &h, HistCompare::ChiSquare).unwrap(), 0.0);
        assert_eq!(compare_hist(&h, &h, HistCompare::Intersection).unwrap(), 10.0);
        assert!(compare_hist(&h, &h, HistCompare::Bhattacharyya).unwrap() < 1e-9);
        assert_eq!(compare_hist(&h, &h, HistCompare::ChiSquareAlt).unwrap(), 0.0);
        assert!(compare_hist(&h, &h, HistCompare::KlDiv).unwrap().abs() < 1e-12);
    }

    #[test]
    fn correlation_is_minus_one_for_an_inverted_histogram() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        let c = compare_hist(&a, &b, HistCompare::Correlation).unwrap();
        assert!((c + 1.0).abs() < 1e-12, "got {c}");
    }

    #[test]
    fn chi_square_is_asymmetric_because_upstream_skips_zero_bins() {
        let a = vec![0.0, 1.0];
        let b = vec![5.0, 1.0];
        // a->b skips bin 0 entirely; b->a does not.
        assert_eq!(compare_hist(&a, &b, HistCompare::ChiSquare).unwrap(), 0.0);
        assert_eq!(compare_hist(&b, &a, HistCompare::ChiSquare).unwrap(), 5.0);
    }

    #[test]
    fn mismatched_bin_counts_are_rejected() {
        assert!(compare_hist(&[1.0], &[1.0, 2.0], HistCompare::Correlation).is_err());
    }

    #[test]
    fn back_projection_maps_values_through_the_histogram() {
        // Two bins: values below 128 get weight 10, above get 200.
        let m = Mat::from_vec(1, 3, 1, &[0u8, 100, 200]).unwrap();
        let bp = calc_back_project(&m, &[10.0, 200.0], 1.0).unwrap();
        assert_eq!(bp.to_vec::<u8>().unwrap(), vec![10, 10, 200]);
    }

    #[test]
    fn back_projection_saturates() {
        let m = Mat::from_vec(1, 1, 1, &[0u8]).unwrap();
        let bp = calc_back_project(&m, &[1000.0], 1.0).unwrap();
        assert_eq!(bp.to_vec::<u8>().unwrap(), vec![255]);
    }

    #[test]
    fn clahe_preserves_size_and_spreads_a_flat_region() {
        let src = ramp(32, 32);
        let out = clahe(&src, 2.0, 4, 4).unwrap();
        assert_eq!((out.rows(), out.cols()), (32, 32));
        // Output uses more of the range than a low-contrast input would.
        let v = out.to_vec::<u8>().unwrap();
        let (lo, hi) = (*v.iter().min().unwrap(), *v.iter().max().unwrap());
        assert!(hi as i32 - lo as i32 > 100, "range {lo}..{hi}");
    }

    #[test]
    fn clahe_rejects_a_grid_larger_than_the_image() {
        let src = ramp(4, 4);
        assert!(clahe(&src, 2.0, 8, 8).is_err());
    }

    #[test]
    fn clahe_clip_limit_changes_the_result() {
        // A ramp gives every pixel in a tile a distinct value, so no bin ever
        // exceeds the limit and clipping is a no-op. Repeating values is what
        // builds the histogram peaks the clip is there to cut.
        let mut src = Mat::zeros(32, 32, CV_8UC1).unwrap();
        for y in 0..32 {
            for x in 0..32 {
                src.pixel_mut::<u8>(y, x).unwrap()[0] = if x < 16 { 40 } else { 200 };
            }
        }
        let a = clahe(&src, 1.0, 4, 4).unwrap().to_vec::<u8>().unwrap();
        let b = clahe(&src, 40.0, 4, 4).unwrap().to_vec::<u8>().unwrap();
        assert_ne!(a, b, "clipping must have an effect");
    }

    #[test]
    fn jet_matches_cv2_at_the_anchor_levels() {
        // cv2 4.13.0, COLORMAP_JET, BGR: 0 -> (128,0,0), 64 -> (255,128,0),
        // 128 -> (126,255,130), 255 -> (0,0,128).
        let src = Mat::from_vec(1, 4, 1, &[0u8, 64, 128, 255]).unwrap();
        let out = apply_color_map(&src, ColorMap::Jet).unwrap();
        let want = [[128, 0, 0], [255, 128, 0], [126, 255, 130], [0, 0, 128]];
        for (i, w) in want.iter().enumerate() {
            for (c, &wc) in w.iter().enumerate() {
                let got = out.at_f64(0, i as i32, c as i32).unwrap() as i32;
                assert!((got - wc).abs() <= 1, "level {i} channel {c}: got {got}, cv2 gives {wc}");
            }
        }
    }

    #[test]
    fn hsv_wraps_like_cv2_does() {
        // Hue 0 and hue 360 are the same colour, so COLORMAP_HSV maps 0 and 255
        // to (0,0,255) alike. cv2 4.13.0 does the same; this is not a defect.
        let src = Mat::from_vec(1, 2, 1, &[0u8, 255]).unwrap();
        let out = apply_color_map(&src, ColorMap::Hsv).unwrap();
        let at = |x| (0..3).map(|c| out.at_f64(0, x, c).unwrap()).collect::<Vec<_>>();
        assert_eq!(at(0), vec![0.0, 0.0, 255.0]);
        assert_eq!(at(0), at(1));
    }

    #[test]
    fn color_maps_produce_three_channels_and_vary() {
        let src = Mat::from_vec(1, 3, 1, &[0u8, 128, 255]).unwrap();
        for cm in [ColorMap::Jet, ColorMap::Hot, ColorMap::Bone, ColorMap::Cool] {
            let out = apply_color_map(&src, cm).unwrap();
            assert_eq!(out.channels(), 3);
            let a = (0..3).map(|c| out.at_f64(0, 0, c).unwrap()).collect::<Vec<_>>();
            let b = (0..3).map(|c| out.at_f64(0, 2, c).unwrap()).collect::<Vec<_>>();
            assert_ne!(a, b, "{cm:?} maps 0 and 255 to the same colour");
        }
        // GRAY is the identity on every channel.
        let g = apply_color_map(&src, ColorMap::Gray).unwrap();
        assert_eq!(g.at_f64(0, 1, 0).unwrap(), 128.0);
        assert_eq!(g.at_f64(0, 1, 2).unwrap(), 128.0);
    }

    #[test]
    fn accumulate_family_adds_into_a_float_buffer() {
        let src = Mat::from_vec(1, 2, 1, &[2.0f32, 3.0]).unwrap();
        let mut acc = Mat::zeros(1, 2, CV_32FC1).unwrap();
        accumulate(&src, &mut acc).unwrap();
        accumulate(&src, &mut acc).unwrap();
        assert_eq!(acc.to_vec::<f32>().unwrap(), vec![4.0, 6.0]);

        let mut sq = Mat::zeros(1, 2, CV_32FC1).unwrap();
        accumulate_square(&src, &mut sq).unwrap();
        assert_eq!(sq.to_vec::<f32>().unwrap(), vec![4.0, 9.0]);

        let mut pr = Mat::zeros(1, 2, CV_32FC1).unwrap();
        accumulate_product(&src, &src, &mut pr).unwrap();
        assert_eq!(pr.to_vec::<f32>().unwrap(), vec![4.0, 9.0]);
    }

    #[test]
    fn accumulate_weighted_converges_to_the_source() {
        let src = Mat::from_vec(1, 1, 1, &[100.0f32]).unwrap();
        let mut acc = Mat::zeros(1, 1, CV_32FC1).unwrap();
        for _ in 0..200 {
            accumulate_weighted(&src, &mut acc, 0.1).unwrap();
        }
        assert!((acc.at_f64(0, 0, 0).unwrap() - 100.0).abs() < 1e-3);
    }

    #[test]
    fn accumulate_rejects_an_integer_accumulator() {
        let src = Mat::from_vec(1, 1, 1, &[1u8]).unwrap();
        let mut acc = Mat::zeros(1, 1, CV_8UC1).unwrap();
        assert!(accumulate(&src, &mut acc).is_err(), "CV_8U cannot accumulate");
    }

    #[test]
    fn get_rect_sub_pix_at_an_integer_centre_is_a_plain_crop() {
        let m = ramp(8, 8);
        let p = get_rect_sub_pix(&m, 3, 3, (3.0, 3.0)).unwrap();
        assert_eq!((p.rows(), p.cols()), (3, 3));
        for j in 0..3 {
            for i in 0..3 {
                assert_eq!(p.at_f64(j, i, 0).unwrap(), m.at_f64(2 + j, 2 + i, 0).unwrap());
            }
        }
    }

    #[test]
    fn get_rect_sub_pix_interpolates_at_a_half_pixel() {
        let m = Mat::from_vec(2, 2, 1, &[0u8, 10, 20, 30]).unwrap();
        let p = get_rect_sub_pix(&m, 1, 1, (0.5, 0.5)).unwrap();
        assert!((p.at_f64(0, 0, 0).unwrap() - 15.0).abs() < 1e-9, "mean of the four");
    }

    #[test]
    fn warp_polar_maps_a_disc_and_keeps_the_size() {
        let src = ramp(32, 32);
        let out = warp_polar(&src, 40, 60, (16.0, 16.0), 16.0, PolarMode::Linear).unwrap();
        assert_eq!((out.rows(), out.cols()), (60, 40));
        // Column 0 is the centre pixel repeated for every angle.
        let c = src.at_f64(16, 16, 0).unwrap();
        for j in 0..60 {
            assert!((out.at_f64(j, 0, 0).unwrap() - c).abs() < 1e-9);
        }
    }

    #[test]
    fn log_polar_differs_from_linear_polar() {
        let src = ramp(32, 32);
        let a = linear_polar(&src, (16.0, 16.0), 16.0).unwrap().to_vec::<u8>().unwrap();
        let b = log_polar(&src, (16.0, 16.0), 16.0).unwrap().to_vec::<u8>().unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn blend_linear_with_equal_weights_is_the_mean() {
        let a = Mat::from_vec(1, 2, 1, &[0u8, 100]).unwrap();
        let b = Mat::from_vec(1, 2, 1, &[100u8, 200]).unwrap();
        let w = Mat::from_vec(1, 2, 1, &[1.0f32, 1.0]).unwrap();
        let out = blend_linear(&a, &b, &w, &w).unwrap();
        assert_eq!(out.to_vec::<u8>().unwrap(), vec![50, 150]);
    }

    #[test]
    fn sqr_box_filter_of_a_constant_is_the_square() {
        let m = Mat::from_vec(5, 5, 1, &[3u8; 25]).unwrap();
        let out = sqr_box_filter(&m, 3, true).unwrap();
        assert!((out.at_f64(2, 2, 0).unwrap() - 9.0).abs() < 1e-4);
        let un = sqr_box_filter(&m, 3, false).unwrap();
        assert!((un.at_f64(2, 2, 0).unwrap() - 81.0).abs() < 1e-3, "9 taps of 9");
    }

    #[test]
    fn bilateral_filter_preserves_a_hard_edge_that_a_box_blur_would_smear() {
        // Left half 0, right half 255.
        let mut m = Mat::zeros(9, 10, CV_8UC1).unwrap();
        for y in 0..9 {
            for x in 5..10 {
                m.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        let bf = bilateral_filter(&m, 5, 10.0, 5.0).unwrap();
        // A tight range sigma keeps the two sides separated.
        assert!(bf.at_f64(4, 4, 0).unwrap() < 20.0, "left stays dark: {}", bf.at_f64(4, 4, 0).unwrap());
        assert!(bf.at_f64(4, 5, 0).unwrap() > 235.0, "right stays bright: {}", bf.at_f64(4, 5, 0).unwrap());
        // A box blur of the same width would land near the midpoint.
        let bl = crate::filter::blur(&m, cv_core::Size::new(5, 5), cv_core::BorderType::Reflect101).unwrap();
        assert!(bl.at_f64(4, 4, 0).unwrap() > 60.0, "the box blur does smear it");
    }

    #[test]
    fn bilateral_filter_with_a_huge_range_sigma_behaves_like_a_blur() {
        let m = ramp(12, 12);
        let bf = bilateral_filter(&m, 5, 1e6, 2.0).unwrap();
        assert_eq!((bf.rows(), bf.cols()), (12, 12));
        // It must actually change something.
        assert_ne!(bf.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn stack_blur_smooths_and_preserves_a_constant() {
        let flat = Mat::from_vec(7, 7, 1, &[42u8; 49]).unwrap();
        let out = stack_blur(&flat, 3).unwrap();
        assert!(out.to_vec::<u8>().unwrap().iter().all(|&v| v == 42), "a constant must survive");

        let m = ramp(16, 16);
        let b = stack_blur(&m, 5).unwrap();
        assert_eq!((b.rows(), b.cols()), (16, 16));
        assert_ne!(b.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn stack_blur_rejects_an_even_kernel() {
        assert!(stack_blur(&ramp(4, 4), 4).is_err());
    }

    /// Every palette, checked end to end against cv2 4.13.0.
    ///
    /// Per-channel column sums catch a wrong or shifted table; the probes
    /// catch an off-by-one at the ends, which a sum would hide.
    #[test]
    fn every_palette_matches_cv2_exactly() {
        type Probe = (usize, [u8; 3]);
        let cases: &[(ColorMap, &str, [i64; 3], &[Probe])] = &[
            (ColorMap::Autumn, "AUTUMN", [0, 32640, 65280], &[(0, [0, 0, 255]), (1, [0, 1, 255]), (85, [0, 85, 255]), (128, [0, 128, 255]), (200, [0, 200, 255]), (254, [0, 254, 255]), (255, [0, 255, 255])]),
            (ColorMap::Bone, "BONE", [35180, 32133, 29593], &[(0, [0, 0, 0]), (1, [1, 1, 1]), (85, [103, 74, 74]), (128, [144, 123, 112]), (200, [207, 207, 179]), (254, [254, 254, 254]), (255, [255, 255, 255])]),
            (ColorMap::Jet, "JET", [30591, 32448, 30592], &[(0, [128, 0, 0]), (1, [132, 0, 0]), (85, [255, 212, 0]), (128, [126, 255, 130]), (200, [0, 92, 255]), (254, [0, 0, 132]), (255, [0, 0, 128])]),
            (ColorMap::Winter, "WINTER", [48967, 32640, 0], &[(0, [255, 0, 0]), (1, [254, 1, 0]), (85, [213, 85, 0]), (128, [191, 128, 0]), (200, [155, 200, 0]), (254, [128, 254, 0]), (255, [128, 255, 0])]),
            (ColorMap::Rainbow, "RAINBOW", [19630, 32512, 37066], &[(0, [0, 0, 255]), (1, [0, 2, 255]), (85, [0, 212, 255]), (128, [0, 255, 125]), (200, [235, 20, 0]), (254, [255, 0, 167]), (255, [255, 0, 170])]),
            (ColorMap::Ocean, "OCEAN", [32640, 21805, 10965], &[(0, [0, 0, 0]), (1, [1, 0, 0]), (85, [85, 0, 0]), (128, [128, 64, 0]), (200, [200, 172, 90]), (254, [254, 254, 252]), (255, [255, 255, 255])]),
            (ColorMap::Summer, "SUMMER", [26112, 48960, 32640], &[(0, [102, 128, 0]), (1, [102, 128, 1]), (85, [102, 170, 85]), (128, [102, 192, 128]), (200, [102, 228, 200]), (254, [102, 254, 254]), (255, [102, 255, 255])]),
            (ColorMap::Spring, "SPRING", [32640, 32640, 65280], &[(0, [255, 0, 255]), (1, [254, 1, 255]), (85, [170, 85, 255]), (128, [127, 128, 255]), (200, [55, 200, 255]), (254, [1, 254, 255]), (255, [0, 255, 255])]),
            (ColorMap::Cool, "COOL", [65280, 32640, 32640], &[(0, [255, 255, 0]), (1, [255, 254, 1]), (85, [255, 170, 85]), (128, [255, 127, 128]), (200, [255, 55, 200]), (254, [255, 1, 254]), (255, [255, 0, 255])]),
            (ColorMap::Hsv, "HSV", [32483, 32483, 32738], &[(0, [0, 0, 255]), (1, [0, 6, 255]), (85, [0, 255, 0]), (128, [250, 247, 0]), (200, [255, 0, 180]), (254, [6, 0, 255]), (255, [0, 0, 255])]),
            (ColorMap::Pink, "PINK", [37098, 41500, 48569], &[(0, [0, 0, 0]), (1, [6, 6, 10]), (85, [120, 120, 184]), (128, [148, 171, 208]), (200, [192, 236, 236]), (254, [254, 255, 255]), (255, [255, 255, 255])]),
            (ColorMap::Hot, "HOT", [6641, 26140, 52142], &[(0, [0, 0, 0]), (1, [0, 0, 2]), (85, [0, 0, 212]), (128, [0, 65, 255]), (200, [0, 245, 255]), (254, [250, 255, 255]), (255, [255, 255, 255])]),
            (ColorMap::Parula, "PARULA", [35600, 42013, 27276], &[(0, [135, 42, 53]), (1, [138, 44, 51]), (85, [203, 156, 11]), (128, [157, 185, 57]), (200, [75, 191, 227]), (254, [15, 250, 249]), (255, [14, 251, 249])]),
            (ColorMap::Magma, "MAGMA", [27942, 21146, 41031], &[(0, [4, 0, 0]), (1, [5, 0, 1]), (85, [129, 31, 114]), (128, [121, 55, 183]), (200, [105, 152, 253]), (254, [189, 251, 252]), (255, [191, 253, 252])]),
            (ColorMap::Inferno, "INFERNO", [18007, 21570, 41293], &[(0, [4, 0, 0]), (1, [5, 0, 1]), (85, [109, 28, 120]), (128, [84, 55, 188]), (200, [7, 157, 251]), (254, [161, 253, 250]), (255, [164, 255, 252])]),
            (ColorMap::Plasma, "PLASMA", [28303, 21987, 46057], &[(0, [135, 8, 13]), (1, [136, 7, 16]), (85, [158, 23, 156]), (128, [120, 71, 204]), (200, [57, 161, 251]), (254, [36, 247, 240]), (255, [33, 249, 240])]),
            (ColorMap::Viridis, "VIRIDIS", [28105, 35125, 21356], &[(0, [84, 1, 68]), (1, [86, 2, 68]), (85, [142, 104, 49]), (128, [140, 145, 33]), (200, [87, 207, 112]), (254, [35, 231, 251]), (255, [37, 231, 253])]),
            (ColorMap::Cividis, "CIVIDIS", [26792, 32363, 31942], &[(0, [78, 34, 0]), (1, [79, 35, 0]), (85, [109, 93, 87]), (128, [120, 124, 125]), (200, [104, 180, 196]), (254, [54, 230, 254]), (255, [56, 232, 254])]),
            (ColorMap::Twilight, "TWILIGHT", [35434, 27366, 35542], &[(0, [226, 217, 226]), (1, [226, 217, 225]), (85, [165, 68, 94]), (128, [54, 20, 48]), (200, [89, 106, 188]), (254, [225, 217, 226]), (255, [226, 217, 226])]),
            (ColorMap::TwilightShifted, "TWILIGHT_SHIFTED", [35334, 27269, 35459], &[(0, [55, 20, 48]), (1, [57, 19, 49]), (85, [194, 161, 124]), (128, [226, 217, 226]), (200, [80, 68, 165]), (254, [55, 19, 49]), (255, [54, 20, 47])]),
            (ColorMap::Turbo, "TURBO", [26240, 38112, 35759], &[(0, [59, 18, 48]), (1, [67, 21, 50]), (85, [182, 228, 26]), (128, [60, 252, 164]), (200, [23, 102, 244]), (254, [2, 5, 126]), (255, [3, 4, 122])]),
            (ColorMap::Deepgreen, "DEEPGREEN", [21805, 32640, 10965], &[(0, [0, 0, 0]), (1, [0, 1, 0]), (85, [0, 85, 0]), (128, [64, 128, 0]), (200, [172, 200, 90]), (254, [254, 254, 252]), (255, [255, 255, 255])]),
        ];
        assert_eq!(cases.len(), 22, "cv2 4.13.0 defines 22 COLORMAP_* palettes");

        // A 1x256 ramp through applyColorMap must reproduce the palette.
        let ramp: Vec<u8> = (0..=255).collect();
        let src = Mat::from_vec(1, 256, 1, &ramp).unwrap();
        for (cm, name, sums, probes) in cases {
            let out = apply_color_map(&src, *cm).unwrap();
            let mut got = [0i64; 3];
            for x in 0..256 {
                for c in 0..3 {
                    got[c as usize] += out.at_f64(0, x, c).unwrap() as i64;
                }
            }
            assert_eq!(&got, sums, "COLORMAP_{name} channel sums");
            for (i, want) in *probes {
                for c in 0..3 {
                    assert_eq!(
                        out.at_f64(0, *i as i32, c).unwrap() as u8,
                        want[c as usize],
                        "COLORMAP_{name} level {i} channel {c}"
                    );
                }
            }
        }
    }

    /// The grey ramp is this port's own, not an upstream palette.
    #[test]
    fn the_grey_ramp_is_the_identity() {
        let ramp: Vec<u8> = (0..=255).collect();
        let src = Mat::from_vec(1, 256, 1, &ramp).unwrap();
        let out = apply_color_map(&src, ColorMap::Gray).unwrap();
        for x in 0..256 {
            for c in 0..3 {
                assert_eq!(out.at_f64(0, x, c).unwrap(), f64::from(x));
            }
        }
        assert!(!ColorMap::all().contains(&ColorMap::Gray), "Gray is not an upstream palette");
    }
}
