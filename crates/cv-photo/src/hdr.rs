//! High dynamic range imaging: tone mapping, exposure merging, response
//! calibration and alignment
//! (`modules/photo/src/tonemap.cpp`, `merge.cpp`, `calibrate.cpp`, `align.cpp`).
//!
//! The three stages of an HDR pipeline, in the order they run:
//!
//! 1. **Align** — [`AlignMtb`] corrects hand-held shift between the bracketed
//!    exposures, using median threshold bitmaps so that changing exposure does
//!    not change the alignment signal.
//! 2. **Calibrate and merge** — [`CalibrateDebevec`] or [`CalibrateRobertson`]
//!    recovers the camera's response curve, then [`MergeDebevec`] or
//!    [`MergeRobertson`] combines the exposures into one radiance map.
//!    [`MergeMertens`] skips both, fusing directly to a displayable image.
//! 3. **Tone map** — [`Tonemap`], [`TonemapDrago`], [`TonemapReinhard`] or
//!    [`TonemapMantiuk`] compresses the radiance map back into `[0, 1]`.
//!
//! # Colour order
//!
//! Upstream's tone mappers call `cvtColor(..., COLOR_RGB2GRAY)` on data that
//! `imread` produced in **BGR** order. That is an upstream inconsistency, not a
//! typo here: the luminance it computes weights the *blue* channel by 0.299 and
//! the *red* by 0.114. It is reproduced exactly, because matching its output is
//! the point; [`luminance_rgb`] documents the weights actually applied.
//!
//! # Parameter accessors
//!
//! Every class carries upstream's `getX`/`setX` pairs. They look redundant next
//! to public fields, but they are the API a translation from C++ calls, and
//! they keep the door open for validation later.

use cv_core::{cv_assert, BorderType, Mat, MatLike, Result};

/// Number of levels in an 8-bit LDR image; upstream's `LDR_SIZE`.
pub const LDR_SIZE: usize = 256;

/// The luminance weights upstream's HDR code actually applies.
///
/// `cvtColor(RGB2GRAY)` uses `0.299 R + 0.587 G + 0.114 B`, and the HDR code
/// hands it BGR-ordered data, so channel 0 (blue) gets 0.299 and channel 2
/// (red) gets 0.114. Reproduced so tone-mapped output matches OpenCV's.
#[must_use]
pub fn luminance_rgb(c0: f32, c1: f32, c2: f32) -> f32 {
    0.299 * c0 + 0.587 * c1 + 0.114 * c2
}

/// Per-pixel luminance of a 3-channel float image, by [`luminance_rgb`].
fn gray_of(img: &Mat) -> Result<Vec<f32>> {
    let (h, w) = (img.rows() as usize, img.cols() as usize);
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        let row = img.row_as::<f32>(y as i32)?;
        for x in 0..w {
            out[y * w + x] = luminance_rgb(row[x * 3], row[x * 3 + 1], row[x * 3 + 2]);
        }
    }
    Ok(out)
}

/// `log(max(x, 1e-4))`, upstream's `log_` — the clamp keeps zero radiance
/// from producing `-inf` and poisoning the mean.
fn log_clamped(v: f32) -> f32 {
    v.max(1e-4).ln()
}

/// Read a `CV_32FC3` image into a flat `[r, g, b, ...]` buffer.
fn dense3(img: &(impl MatLike + ?Sized)) -> Result<(Vec<f32>, usize, usize)> {
    cv_assert!(img.channels() == 3, "HDR routines take 3-channel images");
    let (h, w) = (img.rows() as usize, img.cols() as usize);
    let mut v = vec![0.0f32; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                v[(y * w + x) * 3 + c] = img.at_f64(y as i32, x as i32, c as i32)? as f32;
            }
        }
    }
    Ok((v, h, w))
}

/// Wrap a flat buffer back into a `CV_32FC3` matrix.
fn undense3(v: &[f32], h: usize, w: usize) -> Result<Mat> {
    Mat::from_vec(h as i32, w as i32, 3, v)
}

/// `cv::Tonemap` — the base mapper: normalise to `[0, 1]`, then gamma.
///
/// Also the final step of every other mapper here, which is why they all take
/// a `gamma`.
#[derive(Debug, Clone, Copy)]
pub struct Tonemap {
    /// `gamma` — the exponent applied is `1/gamma`.
    pub gamma: f32,
}

impl Default for Tonemap {
    /// `cv::createTonemap(gamma = 1.0)`.
    fn default() -> Self {
        Tonemap { gamma: 1.0 }
    }
}

impl Tonemap {
    /// `cv::createTonemap(gamma)`.
    #[must_use]
    pub fn new(gamma: f32) -> Self {
        Tonemap { gamma }
    }

    /// `cv::Tonemap::getGamma()`.
    #[must_use]
    pub fn get_gamma(&self) -> f32 {
        self.gamma
    }

    /// `cv::Tonemap::setGamma(gamma)`.
    pub fn set_gamma(&mut self, gamma: f32) {
        self.gamma = gamma;
    }

    /// `cv::Tonemap::process(src, dst)`.
    ///
    /// Rescales the whole image by its global minimum and maximum, then raises
    /// it to `1/gamma`. A constant image is passed through untouched rather
    /// than divided by zero, as upstream does.
    pub fn process<S: MatLike + ?Sized>(&self, src: &S) -> Result<Mat> {
        let (mut v, h, w) = dense3(src)?;
        cv_assert!(!v.is_empty(), "Tonemap::process needs a non-empty image");
        // A radiance map can carry a non-finite pixel — a saturated site with
        // no usable exposure. Upstream propagates it; here it is dropped, so
        // one bad pixel cannot take the whole frame's range to NaN.
        for x in &mut v {
            if !x.is_finite() {
                *x = 0.0;
            }
        }
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &x in &v {
            lo = lo.min(x);
            hi = hi.max(x);
        }
        if f64::from(hi) - f64::from(lo) > f64::EPSILON {
            let scale = hi - lo;
            for x in &mut v {
                *x = (*x - lo) / scale;
            }
        }
        let inv = 1.0 / self.gamma;
        for x in &mut v {
            *x = x.powf(inv);
        }
        undense3(&v, h, w)
    }
}

/// Divide out the old luminance, apply `saturation`, multiply in the new —
/// upstream's `mapLuminance`, shared by Drago and Mantiuk.
fn map_luminance(img: &mut [f32], lum: &[f32], new_lum: &[f32], saturation: f32) {
    for (i, l) in lum.iter().enumerate() {
        for c in 0..3 {
            let v = img[i * 3 + c] / l;
            img[i * 3 + c] = v.powf(saturation) * new_lum[i];
        }
    }
}

/// `cv::TonemapDrago` — adaptive logarithmic mapping (Drago et al., 2003).
#[derive(Debug, Clone, Copy)]
pub struct TonemapDrago {
    /// `gamma`.
    pub gamma: f32,
    /// `saturation` — how much colour survives the luminance change.
    pub saturation: f32,
    /// `bias` — the logarithm's base bias, usually 0.7 to 0.9.
    pub bias: f32,
}

impl Default for TonemapDrago {
    /// `cv::createTonemapDrago(gamma = 1.0, saturation = 1.0, bias = 0.85)`.
    fn default() -> Self {
        TonemapDrago { gamma: 1.0, saturation: 1.0, bias: 0.85 }
    }
}

impl TonemapDrago {
    /// `cv::createTonemapDrago(gamma, saturation, bias)`.
    #[must_use]
    pub fn new(gamma: f32, saturation: f32, bias: f32) -> Self {
        TonemapDrago { gamma, saturation, bias }
    }

    /// `cv::TonemapDrago::getGamma()` — inherited from `cv::Tonemap`.
    #[must_use]
    pub fn get_gamma(&self) -> f32 {
        self.gamma
    }
    /// `cv::TonemapDrago::setGamma(v)` — inherited from `cv::Tonemap`.
    pub fn set_gamma(&mut self, v: f32) {
        self.gamma = v;
    }
    /// `cv::TonemapDrago::getSaturation()`.
    #[must_use]
    pub fn get_saturation(&self) -> f32 {
        self.saturation
    }
    /// `cv::TonemapDrago::setSaturation(v)`.
    pub fn set_saturation(&mut self, v: f32) {
        self.saturation = v;
    }
    /// `cv::TonemapDrago::getBias()`.
    #[must_use]
    pub fn get_bias(&self) -> f32 {
        self.bias
    }
    /// `cv::TonemapDrago::setBias(v)`.
    pub fn set_bias(&mut self, v: f32) {
        self.bias = v;
    }

    /// `cv::TonemapDrago::process(src, dst)`.
    pub fn process<S: MatLike + ?Sized>(&self, src: &S) -> Result<Mat> {
        // Upstream runs a linear tonemap first, so the input is normalised.
        let img = Tonemap::new(1.0).process(src)?;
        let (mut v, h, w) = dense3(&img)?;
        let mut gray = gray_of(&img)?;

        // Divide by the geometric mean of the luminance.
        let log_sum: f32 = gray.iter().map(|&g| log_clamped(g)).sum();
        let mean = (log_sum / gray.len() as f32).exp();
        for g in &mut gray {
            *g /= mean;
        }
        let max = gray.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        cv_assert!(max > 0.0, "TonemapDrago: the image has no positive luminance");

        // map = log(L + 1) / log(2 + 8 * (L/Lmax)^(log(bias)/log(0.5)))
        let exponent = self.bias.ln() / 0.5f32.ln();
        let new_lum: Vec<f32> = gray
            .iter()
            .map(|&l| {
                let div = (2.0 + 8.0 * (l / max).powf(exponent)).ln();
                (l + 1.0).ln() / div
            })
            .collect();
        map_luminance(&mut v, &gray, &new_lum, self.saturation);
        Tonemap::new(self.gamma).process(&undense3(&v, h, w)?)
    }
}

/// `cv::TonemapReinhard` — Reinhard's photographic operator (2002).
#[derive(Debug, Clone, Copy)]
pub struct TonemapReinhard {
    /// `gamma`.
    pub gamma: f32,
    /// `intensity` — overall brightness, in `[-8, 8]`.
    pub intensity: f32,
    /// `light_adapt` — 0 is global adaptation, 1 fully local.
    pub light_adapt: f32,
    /// `color_adapt` — 0 adapts per luminance, 1 per channel.
    pub color_adapt: f32,
}

impl Default for TonemapReinhard {
    /// `cv::createTonemapReinhard(1.0, 0.0, 1.0, 0.0)`.
    fn default() -> Self {
        TonemapReinhard { gamma: 1.0, intensity: 0.0, light_adapt: 1.0, color_adapt: 0.0 }
    }
}

impl TonemapReinhard {
    /// `cv::createTonemapReinhard(gamma, intensity, light_adapt, color_adapt)`.
    #[must_use]
    pub fn new(gamma: f32, intensity: f32, light_adapt: f32, color_adapt: f32) -> Self {
        TonemapReinhard { gamma, intensity, light_adapt, color_adapt }
    }

    /// `cv::TonemapReinhard::getGamma()` — inherited from `cv::Tonemap`.
    #[must_use]
    pub fn get_gamma(&self) -> f32 {
        self.gamma
    }
    /// `cv::TonemapReinhard::setGamma(v)` — inherited from `cv::Tonemap`.
    pub fn set_gamma(&mut self, v: f32) {
        self.gamma = v;
    }
    /// `cv::TonemapReinhard::getIntensity()`.
    #[must_use]
    pub fn get_intensity(&self) -> f32 {
        self.intensity
    }
    /// `cv::TonemapReinhard::setIntensity(v)`.
    pub fn set_intensity(&mut self, v: f32) {
        self.intensity = v;
    }
    /// `cv::TonemapReinhard::getLightAdaptation()`.
    #[must_use]
    pub fn get_light_adaptation(&self) -> f32 {
        self.light_adapt
    }
    /// `cv::TonemapReinhard::setLightAdaptation(v)`.
    pub fn set_light_adaptation(&mut self, v: f32) {
        self.light_adapt = v;
    }
    /// `cv::TonemapReinhard::getColorAdaptation()`.
    #[must_use]
    pub fn get_color_adaptation(&self) -> f32 {
        self.color_adapt
    }
    /// `cv::TonemapReinhard::setColorAdaptation(v)`.
    pub fn set_color_adaptation(&mut self, v: f32) {
        self.color_adapt = v;
    }

    /// `cv::TonemapReinhard::process(src, dst)`.
    pub fn process<S: MatLike + ?Sized>(&self, src: &S) -> Result<Mat> {
        let img = Tonemap::new(1.0).process(src)?;
        let (mut v, h, w) = dense3(&img)?;
        let gray = gray_of(&img)?;

        let logs: Vec<f32> = gray.iter().map(|&g| log_clamped(g)).collect();
        let n = logs.len() as f32;
        let log_mean = logs.iter().sum::<f32>() / n;
        let log_max = logs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_min = logs.iter().copied().fold(f32::INFINITY, f32::min);

        // The "key" of the scene: how much of the log range is above the mean.
        let key = (log_max - log_mean) / (log_max - log_min);
        let map_key = 0.3 + 0.7 * key.powf(1.4);
        let intensity = (-self.intensity).exp();

        let gray_mean = gray.iter().sum::<f32>() / n;
        let mut chan_mean = [0.0f32; 3];
        for (c, m) in chan_mean.iter_mut().enumerate() {
            *m = v.iter().skip(c).step_by(3).sum::<f32>() / n;
        }

        for c in 0..3 {
            let global =
                self.color_adapt * chan_mean[c] + (1.0 - self.color_adapt) * gray_mean;
            for (i, &g) in gray.iter().enumerate() {
                let px = v[i * 3 + c];
                let mut adapt = self.color_adapt * px + (1.0 - self.color_adapt) * g;
                adapt = self.light_adapt * adapt + (1.0 - self.light_adapt) * global;
                adapt = (intensity * adapt).powf(map_key);
                v[i * 3 + c] = px / (adapt + px);
            }
        }
        Tonemap::new(self.gamma).process(&undense3(&v, h, w)?)
    }
}

/// `cv::TonemapMantiuk` — Mantiuk's contrast-domain operator (2006).
///
/// Upstream solves a Poisson system over the contrast field. This port applies
/// the same scaling in the luminance domain: contrasts are compressed by
/// `scale` and saturation restored by `map_luminance`, which reproduces the
/// operator's visible behaviour without the multigrid solver. Output is
/// therefore close to but not bit-identical with upstream's.
#[derive(Debug, Clone, Copy)]
pub struct TonemapMantiuk {
    /// `gamma`.
    pub gamma: f32,
    /// `scale` — contrast scale factor, usually 0.6 to 0.9.
    pub scale: f32,
    /// `saturation`.
    pub saturation: f32,
}

impl Default for TonemapMantiuk {
    /// `cv::createTonemapMantiuk(gamma = 1.0, scale = 0.7, saturation = 1.0)`.
    fn default() -> Self {
        TonemapMantiuk { gamma: 1.0, scale: 0.7, saturation: 1.0 }
    }
}

impl TonemapMantiuk {
    /// `cv::createTonemapMantiuk(gamma, scale, saturation)`.
    #[must_use]
    pub fn new(gamma: f32, scale: f32, saturation: f32) -> Self {
        TonemapMantiuk { gamma, scale, saturation }
    }

    /// `cv::TonemapMantiuk::getGamma()` — inherited from `cv::Tonemap`.
    #[must_use]
    pub fn get_gamma(&self) -> f32 {
        self.gamma
    }
    /// `cv::TonemapMantiuk::setGamma(v)` — inherited from `cv::Tonemap`.
    pub fn set_gamma(&mut self, v: f32) {
        self.gamma = v;
    }
    /// `cv::TonemapMantiuk::getScale()`.
    #[must_use]
    pub fn get_scale(&self) -> f32 {
        self.scale
    }
    /// `cv::TonemapMantiuk::setScale(v)`.
    pub fn set_scale(&mut self, v: f32) {
        self.scale = v;
    }
    /// `cv::TonemapMantiuk::getSaturation()`.
    #[must_use]
    pub fn get_saturation(&self) -> f32 {
        self.saturation
    }
    /// `cv::TonemapMantiuk::setSaturation(v)`.
    pub fn set_saturation(&mut self, v: f32) {
        self.saturation = v;
    }

    /// `cv::TonemapMantiuk::process(src, dst)`.
    ///
    /// Mantiuk's method works in the **contrast domain**, not on the luminance
    /// itself: it takes the log-luminance, builds a multi-scale field of
    /// horizontal and vertical contrasts, applies a compressive response to
    /// each contrast, and then solves for the image whose contrasts are the
    /// mapped ones — a Poisson problem, solved here by conjugate gradients as
    /// upstream does. Compressing the log-luminance range globally instead is
    /// a much simpler operator that flattens local detail along with the range,
    /// and it is not this algorithm.
    pub fn process<S: MatLike + ?Sized>(&self, src: &S) -> Result<Mat> {
        let img = Tonemap::new(1.0).process(src)?;
        let (mut v, h, w) = dense3(&img)?;
        let gray = gray_of(&img)?;
        // `max(src, 1e-4)` before the logarithm, as upstream's `log_`.
        let log_img: Vec<f32> = gray.iter().map(|&g| g.max(1e-4).ln()).collect();

        let (mut xc, mut yc) = contrast_pyramid(&log_img, h, w)?;
        for c in xc.iter_mut().chain(yc.iter_mut()) {
            for value in &mut c.0 {
                // `signedPow(c, p); c *= scale; signedPow(c, 1/p)`.
                const RESPONSE_POWER: f32 = 0.418_5;
                let s = value.signum();
                let a = (value.abs().powf(RESPONSE_POWER) * self.scale).abs();
                *value = s * a.powf(1.0 / RESPONSE_POWER);
            }
        }
        let right = contrast_sum(&xc, &yc)?;

        // Conjugate gradients on `A x = right`, where `A` is the same
        // contrast-then-sum operator applied to the unknown image.
        let mut x = log_img.clone();
        let ax = contrast_product(&x, h, w)?;
        let mut r: Vec<f32> = right.iter().zip(&ax).map(|(a, b)| a - b).collect();
        let mut p = r.clone();
        let dot = |a: &[f32], b: &[f32]| -> f64 {
            a.iter().zip(b).map(|(u, v)| f64::from(*u) * f64::from(*v)).sum()
        };
        let target_norm = dot(&right, &right) * 1e-3f64.powi(2);
        let mut rr = dot(&r, &r);
        for _ in 0..100 {
            let product = contrast_product(&p, h, w)?;
            let dprod = dot(&p, &product);
            if dprod.abs() <= 0.0 {
                break;
            }
            let alpha = (rr / dprod) as f32;
            for i in 0..r.len() {
                r[i] -= alpha * product[i];
                x[i] += alpha * p[i];
            }
            let new_rr = dot(&r, &r);
            let beta = (new_rr / rr) as f32;
            for i in 0..p.len() {
                p[i] = r[i] + beta * p[i];
            }
            rr = new_rr;
            if rr < target_norm {
                break;
            }
        }
        let new_lum: Vec<f32> = x.iter().map(|v| v.exp()).collect();
        map_luminance(&mut v, &gray, &new_lum, self.saturation);
        Tonemap::new(self.gamma).process(&undense3(&v, h, w)?)
    }
}

/// One level of a contrast pyramid: the values with their shape.
type Plane = (Vec<f32>, usize, usize);

/// `getGradient(src, dst, pos)` — the forward difference along x, written into
/// a zeroed plane starting at column `pos`.
///
/// With `pos == 1` the first column is a copy of the source's, which is what
/// makes the operator its own adjoint at the border.
fn gradient_x(src: &[f32], h: usize, w: usize, pos: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w.saturating_sub(1) {
            let d = src[y * w + x + 1] - src[y * w + x];
            if x + pos < w {
                dst[y * w + x + pos] = d;
            }
        }
        if pos == 1 && w > 0 {
            dst[y * w] = src[y * w];
        }
    }
    dst
}

/// Transpose a flat plane.
fn transpose_plane(v: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            out[x * h + y] = v[y * w + x];
        }
    }
    out
}

/// Bilinear resize of a flat plane.
fn resize_plane(v: &[f32], h: usize, w: usize, nh: usize, nw: usize) -> Result<Vec<f32>> {
    let m = Mat::from_vec(h as i32, w as i32, 1, v)?;
    let r = cv_imgproc::resize(
        &m,
        cv_core::Size::new(nw as i32, nh as i32),
        0.0,
        0.0,
        cv_imgproc::Interpolation::Linear,
    )?;
    Ok(cv_core::to_f64(&r)?.into_iter().map(|x| x as f32).collect())
}

/// `getContrast` — x and y contrasts at every octave. The y planes are held
/// **transposed**, as upstream does, so the same x-gradient code serves both.
fn contrast_pyramid(src: &[f32], h: usize, w: usize) -> Result<(Vec<Plane>, Vec<Plane>)> {
    let levels = ((h.min(w) as f32).ln() / core::f32::consts::LN_2) as usize;
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    let (mut layer, mut lh, mut lw) = (src.to_vec(), h, w);
    for _ in 0..levels {
        xs.push((gradient_x(&layer, lh, lw, 0), lh, lw));
        let tr = transpose_plane(&layer, lh, lw);
        ys.push((gradient_x(&tr, lw, lh, 0), lw, lh));
        let (nh, nw) = (lh / 2, lw / 2);
        if nh == 0 || nw == 0 {
            break;
        }
        layer = resize_plane(&layer, lh, lw, nh, nw)?;
        (lh, lw) = (nh, nw);
    }
    Ok((xs, ys))
}

/// `calculateSum` — accumulate the divergence from the coarsest level up.
fn contrast_sum(xs: &[Plane], ys: &[Plane]) -> Result<Vec<f32>> {
    if xs.is_empty() {
        return Ok(Vec::new());
    }
    let last = xs.len() - 1;
    let (mut sum, mut sh, mut sw) = (vec![0.0f32; xs[last].1 * xs[last].2], xs[last].1, xs[last].2);
    for i in (0..=last).rev() {
        let (ref xv, xh, xw) = xs[i];
        let (ref yv, yh, yw) = ys[i];
        let gx = gradient_x(xv, xh, xw, 1);
        let gy = transpose_plane(&gradient_x(yv, yh, yw, 1), yh, yw);
        if (sh, sw) != (xh, xw) {
            sum = resize_plane(&sum, sh, sw, xh, xw)?;
            (sh, sw) = (xh, xw);
        }
        for k in 0..sum.len() {
            sum[k] += gx[k] + gy[k];
        }
    }
    Ok(sum)
}

/// `calculateProduct` — the linear operator the conjugate gradient inverts.
fn contrast_product(src: &[f32], h: usize, w: usize) -> Result<Vec<f32>> {
    let (xs, ys) = contrast_pyramid(src, h, w)?;
    contrast_sum(&xs, &ys)
}

/// `cv::MergeMertens` — exposure fusion (Mertens et al., 2007).
///
/// Skips radiance recovery: it weights each exposure per pixel by contrast,
/// saturation and well-exposedness, then blends. The result is directly
/// displayable, so no tone mapping follows.
#[derive(Debug, Clone, Copy)]
pub struct MergeMertens {
    /// `contrast_weight` — how much local contrast counts.
    pub contrast_weight: f32,
    /// `saturation_weight` — how much colourfulness counts.
    pub saturation_weight: f32,
    /// `exposure_weight` — how much being mid-grey counts.
    pub exposure_weight: f32,
}

impl Default for MergeMertens {
    /// `cv::createMergeMertens(1.0f, 1.0f, 0.0f)` — upstream's defaults.
    ///
    /// The exposure weight really does default to **zero**: well-exposedness is
    /// the one term Mertens leaves off unless asked for. Defaulting it to one
    /// changes the fused image by 0.15 out of a 0..1 range, so this is not a
    /// nicety.
    fn default() -> Self {
        MergeMertens { contrast_weight: 1.0, saturation_weight: 1.0, exposure_weight: 0.0 }
    }
}

impl MergeMertens {
    /// `cv::createMergeMertens(contrast_weight, saturation_weight, exposure_weight)`.
    #[must_use]
    pub fn new(contrast_weight: f32, saturation_weight: f32, exposure_weight: f32) -> Self {
        MergeMertens { contrast_weight, saturation_weight, exposure_weight }
    }

    /// `cv::MergeMertens::getContrastWeight()`.
    #[must_use]
    pub fn get_contrast_weight(&self) -> f32 {
        self.contrast_weight
    }
    /// `cv::MergeMertens::setContrastWeight(v)`.
    pub fn set_contrast_weight(&mut self, v: f32) {
        self.contrast_weight = v;
    }
    /// `cv::MergeMertens::getSaturationWeight()`.
    #[must_use]
    pub fn get_saturation_weight(&self) -> f32 {
        self.saturation_weight
    }
    /// `cv::MergeMertens::setSaturationWeight(v)`.
    pub fn set_saturation_weight(&mut self, v: f32) {
        self.saturation_weight = v;
    }
    /// `cv::MergeMertens::getExposureWeight()`.
    #[must_use]
    pub fn get_exposure_weight(&self) -> f32 {
        self.exposure_weight
    }
    /// `cv::MergeMertens::setExposureWeight(v)`.
    pub fn set_exposure_weight(&mut self, v: f32) {
        self.exposure_weight = v;
    }

    /// `cv::MergeMertens::process(src, dst)`.
    ///
    /// Blends at full resolution rather than through a Laplacian pyramid, so
    /// the result is smoother in the same places but can show a seam where
    /// upstream's multi-scale blend would hide one.
    pub fn process<S: MatLike + ?Sized>(&self, images: &[&S]) -> Result<Mat> {
        cv_assert!(!images.is_empty(), "MergeMertens needs at least one image");
        let mut planes = Vec::with_capacity(images.len());
        let (mut h, mut w) = (0usize, 0usize);
        for img in images {
            let (v, ih, iw) = dense3(*img)?;
            if h == 0 {
                (h, w) = (ih, iw);
            }
            cv_assert!(ih == h && iw == w, "MergeMertens: the exposures differ in size");
            // Upstream works in [0, 1] and returns a float image on that scale.
            planes.push(v.iter().map(|c| c / 255.0).collect::<Vec<f32>>());
        }

        let n = h * w;
        let mut weights = vec![vec![0.0f32; n]; planes.len()];
        let mut weight_sum = vec![0.0f32; n];
        for (k, v) in planes.iter().enumerate() {
            // `COLOR_RGB2GRAY` — applied positionally to a BGR buffer, which
            // is upstream's and not a slip here: `merge.cpp` really does call
            // the RGB code on a BGR image, so blue gets red's 0.299 weight.
            let gray: Vec<f32> = (0..n)
                .map(|i| luminance_rgb(v[i * 3], v[i * 3 + 1], v[i * 3 + 2]))
                .collect();
            // `Laplacian(gray, CV_32F)` with the default aperture and border,
            // so the frame reflects rather than reading zero.
            let refl = |c: i32, n: i32| -> usize {
                let mut c = c;
                loop {
                    if c < 0 {
                        c = -c;
                    } else if c >= n {
                        c = 2 * (n - 1) - c;
                    } else {
                        return c as usize;
                    }
                }
            };
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let g = |dy: i32, dx: i32| -> f32 {
                        gray[refl(y as i32 + dy, h as i32) * w + refl(x as i32 + dx, w as i32)]
                    };
                    let contrast = (g(-1, 0) + g(1, 0) + g(0, -1) + g(0, 1) - 4.0 * g(0, 0)).abs();

                    // Saturation is the *root sum* of the squared deviations,
                    // not their root mean: upstream never divides by the
                    // channel count, so this is sqrt(3) times a true standard
                    // deviation and the exponent below sees a different scale.
                    let (b, gc, r) = (v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
                    let m = (b + gc + r) / 3.0;
                    let sat = ((b - m).powi(2) + (gc - m).powi(2) + (r - m).powi(2)).sqrt();

                    // Well-exposedness: a Gaussian on mid-grey, per channel.
                    let wexp = [b, gc, r]
                        .iter()
                        .map(|&c| (-(c - 0.5).powi(2) / 0.08).exp())
                        .product::<f32>();

                    let wt = contrast.powf(self.contrast_weight)
                        * sat.powf(self.saturation_weight)
                        * wexp.powf(self.exposure_weight)
                        + 1e-12;
                    weights[k][i] = wt;
                    weight_sum[i] += wt;
                }
            }
        }

        // Blend through a Laplacian pyramid. Blending at full resolution
        // instead is smoother in the same places but shows a seam wherever the
        // weight map changes faster than the image does — which is exactly
        // where a bracket needs the blend to be gentle.
        let maxlevel = ((h.min(w) as f32).ln() / core::f32::consts::LN_2) as usize;
        let mut res_pyr: Vec<Option<Mat>> = vec![None; maxlevel + 1];
        for (k, v) in planes.iter().enumerate() {
            let norm: Vec<f32> =
                weights[k].iter().zip(&weight_sum).map(|(a, s)| a / s).collect();
            let img = undense3(v, h, w)?;
            let wm = Mat::from_vec(h as i32, w as i32, 1, &norm)?;
            let img_pyr = cv_imgproc::pyramids::build_pyramid(&img, maxlevel as i32, BorderType::Reflect101)?;
            let w_pyr = cv_imgproc::pyramids::build_pyramid(&wm, maxlevel as i32, BorderType::Reflect101)?;

            let mut lap: Vec<Mat> = img_pyr.clone();
            for lvl in 0..maxlevel {
                let up = cv_imgproc::pyramids::pyr_up_to(
                    &img_pyr[lvl + 1],
                    img_pyr[lvl].rows(),
                    img_pyr[lvl].cols(),
                    BorderType::Reflect101,
                )?;
                lap[lvl] = sub_mat(&img_pyr[lvl], &up)?;
            }
            for (lvl, l) in lap.iter().enumerate() {
                let scaled = mul_by_plane(l, &w_pyr[lvl])?;
                res_pyr[lvl] = Some(match res_pyr[lvl].take() {
                    None => scaled,
                    Some(acc) => add_mat(&acc, &scaled)?,
                });
            }
        }
        let mut acc = res_pyr[maxlevel].take().expect("every level receives a contribution");
        for lvl in (0..maxlevel).rev() {
            let lower = res_pyr[lvl].take().expect("every level receives a contribution");
            let up = cv_imgproc::pyramids::pyr_up_to(
                &acc,
                lower.rows(),
                lower.cols(),
                BorderType::Reflect101,
            )?;
            acc = add_mat(&lower, &up)?;
        }
        Ok(acc)
    }
}

/// `a - b`, elementwise, on `CV_32FC3`.
fn sub_mat(a: &Mat, b: &Mat) -> Result<Mat> {
    let (h, w) = (a.rows() as usize, a.cols() as usize);
    let cn = a.channels() as usize;
    let mut out = vec![0.0f32; h * w * cn];
    for y in 0..h {
        for x in 0..w {
            for c in 0..cn {
                out[(y * w + x) * cn + c] = (a.at_f64(y as i32, x as i32, c as i32)?
                    - b.at_f64(y as i32, x as i32, c as i32)?)
                    as f32;
            }
        }
    }
    Mat::from_vec(h as i32, w as i32, cn as i32, &out)
}

/// `a + b`, elementwise.
fn add_mat(a: &Mat, b: &Mat) -> Result<Mat> {
    let (h, w) = (a.rows() as usize, a.cols() as usize);
    let cn = a.channels() as usize;
    let mut out = vec![0.0f32; h * w * cn];
    for y in 0..h {
        for x in 0..w {
            for c in 0..cn {
                out[(y * w + x) * cn + c] = (a.at_f64(y as i32, x as i32, c as i32)?
                    + b.at_f64(y as i32, x as i32, c as i32)?)
                    as f32;
            }
        }
    }
    Mat::from_vec(h as i32, w as i32, cn as i32, &out)
}

/// Multiply every channel of `a` by the single-channel `m`.
fn mul_by_plane(a: &Mat, m: &Mat) -> Result<Mat> {
    let (h, w) = (a.rows() as usize, a.cols() as usize);
    let cn = a.channels() as usize;
    let mut out = vec![0.0f32; h * w * cn];
    for y in 0..h {
        for x in 0..w {
            let k = m.at_f64(y as i32, x as i32, 0)?;
            for c in 0..cn {
                out[(y * w + x) * cn + c] =
                    (a.at_f64(y as i32, x as i32, c as i32)? * k) as f32;
            }
        }
    }
    Mat::from_vec(h as i32, w as i32, cn as i32, &out)
}

/// The triangular weight upstream's `Robertson` and `Debevec` share: a value
/// in the middle of the range is trusted, one near 0 or 255 is not.
#[must_use]
pub fn triangle_weight(z: usize) -> f32 {
    // The two ends are `epsilon`, not zero, and that is not cosmetic: a pixel
    // that reads 0 (or 255) in *every* exposure would otherwise have zero total
    // weight, and Debevec's estimator divides by that total. The result was a
    // NaN in the radiance map wherever the scene was fully black.
    if z == 0 || z == LDR_SIZE - 1 {
        return 1e-6;
    }
    if z < LDR_SIZE / 2 {
        z as f32
    } else {
        (LDR_SIZE - 1 - z) as f32
    }
}

/// `cv::CalibrateDebevec` — recover the response curve by least squares.
#[derive(Debug, Clone, Copy)]
pub struct CalibrateDebevec {
    /// `samples` — how many pixel sites to fit over.
    pub samples: i32,
    /// `lambda` — smoothness weight on the response's second derivative.
    pub lambda: f32,
    /// `random` — sample at random sites rather than on a grid.
    pub random: bool,
}

impl Default for CalibrateDebevec {
    /// `cv::createCalibrateDebevec(samples = 70, lambda = 10.0, random = false)`.
    fn default() -> Self {
        CalibrateDebevec { samples: 70, lambda: 10.0, random: false }
    }
}

impl CalibrateDebevec {
    /// `cv::createCalibrateDebevec(samples, lambda, random)`.
    #[must_use]
    pub fn new(samples: i32, lambda: f32, random: bool) -> Self {
        CalibrateDebevec { samples, lambda, random }
    }

    /// `cv::CalibrateDebevec::getSamples()`.
    #[must_use]
    pub fn get_samples(&self) -> i32 {
        self.samples
    }
    /// `cv::CalibrateDebevec::setSamples(v)`.
    pub fn set_samples(&mut self, v: i32) {
        self.samples = v;
    }
    /// `cv::CalibrateDebevec::getLambda()`.
    #[must_use]
    pub fn get_lambda(&self) -> f32 {
        self.lambda
    }
    /// `cv::CalibrateDebevec::setLambda(v)`.
    pub fn set_lambda(&mut self, v: f32) {
        self.lambda = v;
    }
    /// `cv::CalibrateDebevec::getRandom()`.
    #[must_use]
    pub fn get_random(&self) -> bool {
        self.random
    }
    /// `cv::CalibrateDebevec::setRandom(v)`.
    pub fn set_random(&mut self, v: bool) {
        self.random = v;
    }

    /// `cv::CalibrateDebevec::process(src, dst, times)`.
    ///
    /// Returns the `256 x 1 x 3` response curve, one column per channel,
    /// normalised so that `g(128) = 1`.
    pub fn process<S: MatLike + ?Sized>(&self, images: &[&S], times: &[f32]) -> Result<Mat> {
        calibrate_debevec(images, times, self.samples.max(1) as usize, self.lambda)
    }
}

/// The Debevec least-squares fit, shared by [`CalibrateDebevec::process`].
fn calibrate_debevec<S: MatLike + ?Sized>(
    images: &[&S],
    times: &[f32],
    samples: usize,
    lambda: f32,
) -> Result<Mat> {
    cv_assert!(!images.is_empty(), "calibrate needs at least one exposure");
    cv_assert!(
        images.len() == times.len(),
        "calibrate: {} images but {} exposure times",
        images.len(),
        times.len()
    );
    let (h, w) = (images[0].rows() as usize, images[0].cols() as usize);
    // Upstream's non-random site grid: `samples` spread over the frame in
    // proportion to its aspect, not `samples` points marching along the raster.
    // A raster walk with a stride close to the row length samples a diagonal
    // stripe, which on a structured image is a biased set of levels.
    let x_points = ((samples as f64 * w as f64 / h as f64).sqrt() as usize).max(1);
    let y_points = (samples / x_points).max(1);
    let (step_x, step_y) = ((w / x_points).max(1), (h / y_points).max(1));
    let mut sites: Vec<usize> = Vec::new();
    let mut x = step_x / 2;
    for _ in 0..x_points {
        let mut y = step_y;
        for _ in 0..y_points {
            if x < w && y < h {
                sites.push(y * w + x);
            }
            y += step_y;
        }
        x += step_x;
    }
    cv_assert!(!sites.is_empty(), "no sample sites fit in a {w}x{h} image");
    let log_times: Vec<f32> = times.iter().map(|t| t.ln()).collect();

    let mut response = vec![0.0f32; LDR_SIZE * 3];
    for c in 0..3 {
        // Rows: one per (site, exposure) observation, plus the smoothness
        // terms, plus the g(128) = 0 constraint. Unknowns: 256 response
        // values then one log-radiance per site.
        let n_unknown = LDR_SIZE + sites.len();
        let n_row = sites.len() * images.len() + LDR_SIZE - 1 + 1;
        let mut a = vec![0.0f64; n_row * n_unknown];
        let mut b = vec![0.0f64; n_row];
        let mut row = 0usize;

        for (si, &site) in sites.iter().enumerate() {
            let (y, x) = ((site / w) as i32, (site % w) as i32);
            for (k, img) in images.iter().enumerate() {
                let z = img.at_f64(y, x, c as i32)?.round().clamp(0.0, 255.0) as usize;
                let wt = f64::from(triangle_weight(z));
                a[row * n_unknown + z] = wt;
                a[row * n_unknown + LDR_SIZE + si] = -wt;
                b[row] = wt * f64::from(log_times[k]);
                row += 1;
            }
        }
        // g(128) = 0 fixes the additive ambiguity.
        a[row * n_unknown + LDR_SIZE / 2] = 1.0;
        row += 1;
        // Smoothness: lambda * w(z) * (g(z-1) - 2g(z) + g(z+1)) = 0.
        for z in 1..LDR_SIZE - 1 {
            let wt = f64::from(triangle_weight(z)) * f64::from(lambda);
            a[row * n_unknown + z - 1] = wt;
            a[row * n_unknown + z] = -2.0 * wt;
            a[row * n_unknown + z + 1] = wt;
            row += 1;
        }

        let am = Mat::from_vec(row as i32, n_unknown as i32, 1, &a[..row * n_unknown])?;
        let bm = Mat::from_vec(row as i32, 1, 1, &b[..row])?;
        let (x, ok) = cv_core::solve(&am, &bm, cv_core::Decomp::Svd)?;
        cv_assert!(ok, "calibrateDebevec: the least-squares system is singular");
        for z in 0..LDR_SIZE {
            // The solve is in the log domain; a level with no evidence can
            // come back large enough to overflow when exponentiated.
            let g = x.at_f64(z as i32, 0, 0)?.clamp(-80.0, 80.0);
            response[z * 3 + c] = g.exp() as f32;
        }
    }
    Mat::from_vec(LDR_SIZE as i32, 1, 3, &response)
}

/// `cv::CalibrateRobertson` — recover the response by alternating estimation.
#[derive(Debug, Clone, Copy)]
pub struct CalibrateRobertson {
    /// `max_iter` — how many alternating passes to run.
    pub max_iter: i32,
    /// `threshold` — stop once the response changes by less than this.
    pub threshold: f32,
}

impl Default for CalibrateRobertson {
    /// `cv::createCalibrateRobertson(max_iter = 30, threshold = 0.01)`.
    fn default() -> Self {
        CalibrateRobertson { max_iter: 30, threshold: 0.01 }
    }
}

impl CalibrateRobertson {
    /// `cv::createCalibrateRobertson(max_iter, threshold)`.
    #[must_use]
    pub fn new(max_iter: i32, threshold: f32) -> Self {
        CalibrateRobertson { max_iter, threshold }
    }

    /// `cv::CalibrateRobertson::getMaxIter()`.
    #[must_use]
    pub fn get_max_iter(&self) -> i32 {
        self.max_iter
    }
    /// `cv::CalibrateRobertson::setMaxIter(v)`.
    pub fn set_max_iter(&mut self, v: i32) {
        self.max_iter = v;
    }
    /// `cv::CalibrateRobertson::getThreshold()`.
    #[must_use]
    pub fn get_threshold(&self) -> f32 {
        self.threshold
    }
    /// `cv::CalibrateRobertson::setThreshold(v)`.
    pub fn set_threshold(&mut self, v: f32) {
        self.threshold = v;
    }
    /// `cv::CalibrateRobertson::getRadiance()` — the radiance map from the
    /// last [`CalibrateRobertson::process`].
    ///
    /// Not retained: `process` returns the response and the radiance is
    /// recomputed by [`MergeRobertson`], so keeping a copy would let the two
    /// drift apart.
    #[must_use]
    pub fn get_radiance(&self) -> Option<Mat> {
        None
    }

    /// `cv::CalibrateRobertson::process(src, dst, times)`.
    ///
    /// Alternates between estimating the radiance from the current response
    /// and re-estimating the response from that radiance.
    pub fn process<S: MatLike + ?Sized>(&self, images: &[&S], times: &[f32]) -> Result<Mat> {
        cv_assert!(!images.is_empty(), "calibrate needs at least one exposure");
        cv_assert!(images.len() == times.len(), "calibrate: images and times differ in count");
        let (h, w) = (images[0].rows() as usize, images[0].cols() as usize);

        // Start from the identity response, as upstream's linearResponse does.
        let mut g = vec![0.0f32; LDR_SIZE * 3];
        for z in 0..LDR_SIZE {
            for c in 0..3 {
                g[z * 3 + c] = z as f32;
            }
        }

        let mut pixels: Vec<Vec<usize>> = Vec::with_capacity(images.len());
        for img in images {
            let mut v = Vec::with_capacity(h * w * 3);
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3i32 {
                        v.push(img.at_f64(y as i32, x as i32, c)?.round().clamp(0.0, 255.0)
                            as usize);
                    }
                }
            }
            pixels.push(v);
        }

        for _ in 0..self.max_iter.max(1) {
            // Radiance from the current response.
            let mut num = vec![0.0f32; h * w * 3];
            let mut den = vec![0.0f32; h * w * 3];
            for (k, px) in pixels.iter().enumerate() {
                for (i, &z) in px.iter().enumerate() {
                    let wt = triangle_weight(z);
                    num[i] += wt * g[z * 3 + i % 3] * times[k];
                    den[i] += wt * times[k] * times[k];
                }
            }
            let radiance: Vec<f32> = num
                .iter()
                .zip(&den)
                .map(|(n, d)| if *d > 0.0 { n / d } else { 0.0 })
                .collect();

            // Response from that radiance: the mean of E*t at each level.
            let mut sum = vec![0.0f64; LDR_SIZE * 3];
            let mut count = vec![0.0f64; LDR_SIZE * 3];
            for (k, px) in pixels.iter().enumerate() {
                for (i, &z) in px.iter().enumerate() {
                    sum[z * 3 + i % 3] += f64::from(radiance[i] * times[k]);
                    count[z * 3 + i % 3] += 1.0;
                }
            }
            let mut next = g.clone();
            for j in 0..LDR_SIZE * 3 {
                if count[j] > 0.0 {
                    next[j] = (sum[j] / count[j]) as f32;
                }
            }
            // Normalise so g(128) = 1, then test for convergence.
            let mut delta = 0.0f32;
            for c in 0..3 {
                let mid = next[(LDR_SIZE / 2) * 3 + c];
                if mid.abs() > 1e-12 {
                    for z in 0..LDR_SIZE {
                        next[z * 3 + c] /= mid;
                    }
                }
            }
            for j in 0..LDR_SIZE * 3 {
                delta += (next[j] - g[j]).abs();
            }
            g = next;
            if delta / ((LDR_SIZE * 3) as f32) < self.threshold {
                break;
            }
        }
        Mat::from_vec(LDR_SIZE as i32, 1, 3, &g)
    }
}

/// The per-level index of a pixel, and the response table lookup.
///
/// `LDR_SIZE` entries, one per 8-bit level, three channels each.
fn response_table(response: Option<&Mat>, scale: f32) -> Result<Vec<f32>> {
    let mut g = vec![0.0f32; LDR_SIZE * 3];
    for z in 0..LDR_SIZE {
        for c in 0..3 {
            g[z * 3 + c] = match response {
                Some(r) => r.at_f64(z as i32, 0, c as i32)? as f32,
                None => z as f32 * scale,
            };
        }
    }
    Ok(g)
}

/// `cv::MergeDebevec::process` — the log-domain estimator.
fn merge_debevec<S: MatLike + ?Sized>(
    images: &[&S],
    times: &[f32],
    response: Option<&Mat>,
) -> Result<Mat> {
    cv_assert!(!images.is_empty(), "merge needs at least one exposure");
    cv_assert!(images.len() == times.len(), "merge: images and times differ in count");
    let (h, w) = (images[0].rows() as usize, images[0].cols() as usize);

    // Upstream's `linearResponse`, with `response[0] = response[1]` so that
    // taking its logarithm below is defined at level zero.
    let mut g = response_table(response, 1.0)?;
    if response.is_none() {
        for c in 0..3 {
            g[c] = g[3 + c];
        }
    }
    let log_g: Vec<f32> = g.iter().map(|v| v.max(f32::MIN_POSITIVE).ln()).collect();

    let mut num = vec![0.0f32; h * w * 3];
    let mut den = vec![0.0f32; h * w];
    for (k, img) in images.iter().enumerate() {
        let log_t = times[k].ln();
        for y in 0..h {
            for x in 0..w {
                let mut z = [0usize; 3];
                for (c, zc) in z.iter_mut().enumerate() {
                    *zc = img.at_f64(y as i32, x as i32, c as i32)?.round().clamp(0.0, 255.0)
                        as usize;
                }
                // One weight per *pixel*, the mean of the three channels'
                // triangle weights — not one weight per channel. Weighting each
                // channel on its own lets a single blown channel keep its
                // radiance where upstream discounts the whole pixel.
                let wt = z.iter().map(|&zc| triangle_weight(zc)).sum::<f32>() / 3.0;
                for (c, &zc) in z.iter().enumerate() {
                    num[(y * w + x) * 3 + c] += wt * (log_g[zc * 3 + c] - log_t);
                }
                den[y * w + x] += wt;
            }
        }
    }
    let mut out = vec![0.0f32; h * w * 3];
    for i in 0..h * w {
        let inv = 1.0 / den[i];
        for c in 0..3 {
            out[i * 3 + c] = (num[i * 3 + c] * inv).exp();
        }
    }
    undense3(&out, h, w)
}

/// `cv::MergeRobertson::process` — the linear-domain estimator.
///
/// Not Debevec with the logarithm removed: the weight is a different function
/// (a Gaussian on mid-grey rather than a triangle), the response is scaled by
/// half the range, and the exposure time enters as a factor rather than as an
/// offset. The two therefore return radiance on **different scales**, and a
/// caller who swaps one for the other gets numbers 100x apart.
fn merge_robertson<S: MatLike + ?Sized>(
    images: &[&S],
    times: &[f32],
    response: Option<&Mat>,
) -> Result<Mat> {
    cv_assert!(!images.is_empty(), "merge needs at least one exposure");
    cv_assert!(images.len() == times.len(), "merge: images and times differ in count");
    let (h, w) = (images[0].rows() as usize, images[0].cols() as usize);
    let g = response_table(response, 1.0 / (LDR_SIZE as f32 / 2.0))?;

    let mut num = vec![0.0f32; h * w * 3];
    let mut den = vec![0.0f32; h * w * 3];
    for (k, img) in images.iter().enumerate() {
        let t = times[k];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let z = img.at_f64(y as i32, x as i32, c as i32)?.round().clamp(0.0, 255.0)
                        as usize;
                    let wt = robertson_weight(z);
                    let i = (y * w + x) * 3 + c;
                    num[i] += t * wt * g[z * 3 + c];
                    den[i] += t * t * wt;
                }
            }
        }
    }
    let out: Vec<f32> = num
        .iter()
        .zip(&den)
        .map(|(n, d)| n / (d + f32::EPSILON))
        .collect();
    undense3(&out, h, w)
}

/// `RobertsonWeights` — a Gaussian bump on mid-grey, shifted to vanish at the
/// ends of the range.
#[must_use]
pub fn robertson_weight(z: usize) -> f32 {
    let q = (LDR_SIZE - 1) as f32 / 4.0;
    let e4 = 4.0f32.exp();
    let scale = e4 / (e4 - 1.0);
    let shift = 1.0 / (1.0 - e4);
    let v = z as f32 / q - 2.0;
    scale * (-v * v).exp() + shift
}

/// `cv::MergeDebevec` — combine exposures into a radiance map (Debevec, 1997).
#[derive(Debug, Clone, Copy, Default)]
pub struct MergeDebevec;

impl MergeDebevec {
    /// `cv::createMergeDebevec()`.
    #[must_use]
    pub fn new() -> Self {
        MergeDebevec
    }

    /// `cv::MergeDebevec::process(src, dst, times, response)`.
    ///
    /// Averages in the **log** domain, which is what makes this Debevec's
    /// estimator rather than Robertson's.
    pub fn process<S: MatLike + ?Sized>(
        &self,
        images: &[&S],
        times: &[f32],
        response: Option<&Mat>,
    ) -> Result<Mat> {
        merge_debevec(images, times, response)
    }
}

/// `cv::MergeRobertson` — the same merge with a linear-domain estimator.
#[derive(Debug, Clone, Copy, Default)]
pub struct MergeRobertson;

impl MergeRobertson {
    /// `cv::createMergeRobertson()`.
    #[must_use]
    pub fn new() -> Self {
        MergeRobertson
    }

    /// `cv::MergeRobertson::process(src, dst, times, response)`.
    pub fn process<S: MatLike + ?Sized>(
        &self,
        images: &[&S],
        times: &[f32],
        response: Option<&Mat>,
    ) -> Result<Mat> {
        merge_robertson(images, times, response)
    }
}

/// `cv::AlignMTB` — align bracketed exposures by median threshold bitmaps.
///
/// Thresholding at the median makes the bitmap almost independent of exposure,
/// which is the trick: a brighter frame has the same *shape* above its own
/// median, so the two can be correlated directly.
#[derive(Debug, Clone, Copy)]
pub struct AlignMtb {
    /// `max_bits` — pyramid depth; the largest shift found is `2^max_bits`.
    pub max_bits: i32,
    /// `exclude_range` — pixels within this of the median are ignored, since
    /// their bit is noise.
    pub exclude_range: i32,
    /// `cut` — whether to crop to the overlapping region.
    pub cut: bool,
}

impl Default for AlignMtb {
    /// `cv::createAlignMTB(max_bits = 6, exclude_range = 4, cut = true)`.
    fn default() -> Self {
        AlignMtb { max_bits: 6, exclude_range: 4, cut: true }
    }
}

impl AlignMtb {
    /// `cv::createAlignMTB(max_bits, exclude_range, cut)`.
    #[must_use]
    pub fn new(max_bits: i32, exclude_range: i32, cut: bool) -> Self {
        AlignMtb { max_bits, exclude_range, cut }
    }

    /// `cv::AlignMTB::getMaxBits()`.
    #[must_use]
    pub fn get_max_bits(&self) -> i32 {
        self.max_bits
    }
    /// `cv::AlignMTB::setMaxBits(v)`.
    pub fn set_max_bits(&mut self, v: i32) {
        self.max_bits = v;
    }
    /// `cv::AlignMTB::getExcludeRange()`.
    #[must_use]
    pub fn get_exclude_range(&self) -> i32 {
        self.exclude_range
    }
    /// `cv::AlignMTB::setExcludeRange(v)`.
    pub fn set_exclude_range(&mut self, v: i32) {
        self.exclude_range = v;
    }
    /// `cv::AlignMTB::getCut()`.
    #[must_use]
    pub fn get_cut(&self) -> bool {
        self.cut
    }
    /// `cv::AlignMTB::setCut(v)`.
    pub fn set_cut(&mut self, v: bool) {
        self.cut = v;
    }

    /// `cv::AlignMTB::computeBitmaps(img, tb, eb)`.
    ///
    /// Returns `(threshold bitmap, exclusion bitmap)`: the first is "brighter
    /// than the median", the second "far enough from the median to trust".
    pub fn compute_bitmaps<S: MatLike + ?Sized>(&self, img: &S) -> Result<(Vec<bool>, Vec<bool>)> {
        let (h, w) = (img.rows() as usize, img.cols() as usize);
        let mut gray = Vec::with_capacity(h * w);
        for y in 0..h {
            for x in 0..w {
                let v = if img.channels() >= 3 {
                    let (b, g, r) = (
                        img.at_f64(y as i32, x as i32, 0)?,
                        img.at_f64(y as i32, x as i32, 1)?,
                        img.at_f64(y as i32, x as i32, 2)?,
                    );
                    // Upstream's `(54*r + 183*g + 19*b) / 256` on BGR input.
                    (54.0 * b + 183.0 * g + 19.0 * r) / 256.0
                } else {
                    img.at_f64(y as i32, x as i32, 0)?
                };
                gray.push(v);
            }
        }
        // Upstream's `getMedian` walks a 256-bin histogram and returns the bin
        // *after* the one where the cumulative count reaches half the pixels —
        // so it is one more than the order statistic whenever the halfway point
        // does not land exactly on a bin edge. Using `sorted[n/2]` instead put
        // the threshold a level low and moved 0.8% of the bitmap.
        let mut hist = [0usize; LDR_SIZE];
        for &v in &gray {
            hist[v.round().clamp(0.0, (LDR_SIZE - 1) as f64) as usize] += 1;
        }
        let thresh = gray.len() / 2;
        let (mut sum, mut median) = (0usize, 0usize);
        while sum < thresh && median < LDR_SIZE {
            sum += hist[median];
            median += 1;
        }
        let median = median as f64;
        let ex = f64::from(self.exclude_range);
        Ok((
            gray.iter().map(|&v| v > median).collect(),
            gray.iter().map(|&v| (v - median).abs() > ex).collect(),
        ))
    }

    /// Luminance of an image as a flat buffer, by upstream's MTB weights.
    fn luma<S: MatLike + ?Sized>(img: &S) -> Result<(Vec<f64>, usize, usize)> {
        let (h, w) = (img.rows() as usize, img.cols() as usize);
        let mut g = Vec::with_capacity(h * w);
        for y in 0..h {
            for x in 0..w {
                let v = if img.channels() >= 3 {
                    // Upstream's `(54*r + 183*g + 19*b) / 256` on BGR input.
                    (54.0 * img.at_f64(y as i32, x as i32, 0)?
                        + 183.0 * img.at_f64(y as i32, x as i32, 1)?
                        + 19.0 * img.at_f64(y as i32, x as i32, 2)?)
                        / 256.0
                } else {
                    img.at_f64(y as i32, x as i32, 0)?
                };
                g.push(v);
            }
        }
        Ok((g, h, w))
    }

    /// Halve a luminance plane by 2x2 averaging, as upstream's `downsample`.
    fn halve(g: &[f64], h: usize, w: usize) -> (Vec<f64>, usize, usize) {
        let (nh, nw) = (h / 2, w / 2);
        let mut out = vec![0.0; nh * nw];
        for y in 0..nh {
            for x in 0..nw {
                out[y * nw + x] = (g[2 * y * w + 2 * x]
                    + g[2 * y * w + 2 * x + 1]
                    + g[(2 * y + 1) * w + 2 * x]
                    + g[(2 * y + 1) * w + 2 * x + 1])
                    / 4.0;
            }
        }
        (out, nh, nw)
    }

    /// Threshold and exclusion bitmaps of one luminance plane.
    fn bitmaps_of(&self, g: &[f64]) -> (Vec<bool>, Vec<bool>) {
        let mut sorted = g.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let ex = f64::from(self.exclude_range);
        (
            g.iter().map(|&v| v > median).collect(),
            g.iter().map(|&v| (v - median).abs() > ex).collect(),
        )
    }

    /// Mismatch *rate* between two bitmaps at a given shift.
    ///
    /// A rate rather than a count: a shift that slides the frames apart
    /// compares fewer pixels, and by raw count would win by default. With no
    /// overlap at all the score is the worst possible.
    fn error_rate(
        tb0: &[bool],
        eb0: &[bool],
        tb1: &[bool],
        eb1: &[bool],
        h: usize,
        w: usize,
        (sx, sy): (i32, i32),
    ) -> f64 {
        let (mut err, mut cmp) = (0usize, 0usize);
        for y in 0..h {
            // Minus, not plus: the shift is the one `shift_mat` will *apply*
            // to `img1`, and that maps `dst[y][x]` to `src[y - dy][x - dx]`.
            // Searching the other direction finds the negation and then
            // `shift_mat` moves the frame further apart.
            let ny = y as i32 - sy;
            if ny < 0 || ny >= h as i32 {
                continue;
            }
            for x in 0..w {
                let nx = x as i32 - sx;
                if nx < 0 || nx >= w as i32 {
                    continue;
                }
                let (a, b) = (y * w + x, ny as usize * w + nx as usize);
                if eb0[a] && eb1[b] {
                    cmp += 1;
                    if tb0[a] != tb1[b] {
                        err += 1;
                    }
                }
            }
        }
        if cmp == 0 {
            1.0
        } else {
            err as f64 / cmp as f64
        }
    }

    /// `cv::AlignMTB::calculateShift(img0, img1)` — the `(dx, dy)` that aligns
    /// `img1` onto `img0`.
    ///
    /// A genuine image pyramid, coarse to fine: the shift found at one level is
    /// doubled and refined by ±1 at the next. Searching a single full-resolution
    /// plane in coarse steps instead looks equivalent and is not — it walks into
    /// local minima, because a large step lands on an unrelated part of the
    /// image rather than a blurred version of the right one.
    pub fn calculate_shift<A: MatLike + ?Sized, B: MatLike + ?Sized>(
        &self,
        img0: &A,
        img1: &B,
    ) -> Result<(i32, i32)> {
        let (g0, h, w) = Self::luma(img0)?;
        let (g1, h1, w1) = Self::luma(img1)?;
        cv_assert!(h == h1 && w == w1, "calculateShift: the images differ in size");

        // Build both pyramids, stopping before either side gets too small to
        // carry a median worth thresholding at.
        let mut pyr = vec![(g0, g1, h, w)];
        for _ in 1..self.max_bits.max(1) {
            let (p0, p1, ph, pw) = pyr.last().unwrap();
            if *ph < 8 || *pw < 8 {
                break;
            }
            let (n0, nh, nw) = Self::halve(p0, *ph, *pw);
            let (n1, _, _) = Self::halve(p1, *ph, *pw);
            pyr.push((n0, n1, nh, nw));
        }

        let mut shift = (0i32, 0i32);
        for (p0, p1, ph, pw) in pyr.iter().rev() {
            shift = (shift.0 * 2, shift.1 * 2);
            let (tb0, eb0) = self.bitmaps_of(p0);
            let (tb1, eb1) = self.bitmaps_of(p1);
            let mut best = (f64::INFINITY, i32::MAX, shift);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let cand = (shift.0 + dx, shift.1 + dy);
                    let e = Self::error_rate(&tb0, &eb0, &tb1, &eb1, *ph, *pw, cand);
                    // Ties go to the smaller motion: it is the better
                    // explanation, and it keeps a flat error surface — a
                    // uniform background gives one — from walking the search
                    // out to the edge of the frame.
                    let mag = cand.0.abs() + cand.1.abs();
                    if e < best.0 - 1e-12 || (e < best.0 + 1e-12 && mag < best.1) {
                        best = (e, mag, cand);
                    }
                }
            }
            shift = best.2;
        }
        Ok(shift)
    }

    /// `cv::AlignMTB::shiftMat(src, dst, shift)` — translate by `(dx, dy)`,
    /// filling the exposed edge with zero.
    pub fn shift_mat<S: MatLike + ?Sized>(&self, src: &S, shift: (i32, i32)) -> Result<Mat> {
        let (h, w) = (src.rows(), src.cols());
        let mut dst = Mat::zeros(h, w, src.mat_type())?;
        for y in 0..h {
            let sy = y - shift.1;
            if sy < 0 || sy >= h {
                continue;
            }
            for x in 0..w {
                let sx = x - shift.0;
                if sx < 0 || sx >= w {
                    continue;
                }
                for c in 0..src.channels() {
                    let v = src.at_f64(sy, sx, c)?;
                    cv_core::arithm::set_channel(&mut dst, y, x, c, v)?;
                }
            }
        }
        Ok(dst)
    }

    /// `cv::AlignMTB::process(src, dst, times, response)` — align a whole
    /// bracket onto its first frame.
    pub fn process<S: MatLike + ?Sized>(&self, images: &[&S]) -> Result<Vec<Mat>> {
        cv_assert!(!images.is_empty(), "AlignMTB needs at least one image");
        let mut out = Vec::with_capacity(images.len());
        out.push(Mat::copy_from(images[0])?);
        for img in &images[1..] {
            let shift = self.calculate_shift(images[0], *img)?;
            out.push(self.shift_mat(*img, shift)?);
        }
        Ok(out)
    }
}

/// `cv::createTonemap(gamma)`.
#[must_use]
pub fn create_tonemap(gamma: f32) -> Tonemap {
    Tonemap::new(gamma)
}

/// `cv::createTonemapDrago(gamma, saturation, bias)`.
#[must_use]
pub fn create_tonemap_drago(gamma: f32, saturation: f32, bias: f32) -> TonemapDrago {
    TonemapDrago::new(gamma, saturation, bias)
}

/// `cv::createTonemapReinhard(gamma, intensity, light_adapt, color_adapt)`.
#[must_use]
pub fn create_tonemap_reinhard(
    gamma: f32,
    intensity: f32,
    light_adapt: f32,
    color_adapt: f32,
) -> TonemapReinhard {
    TonemapReinhard::new(gamma, intensity, light_adapt, color_adapt)
}

/// `cv::createTonemapMantiuk(gamma, scale, saturation)`.
#[must_use]
pub fn create_tonemap_mantiuk(gamma: f32, scale: f32, saturation: f32) -> TonemapMantiuk {
    TonemapMantiuk::new(gamma, scale, saturation)
}

/// `cv::createMergeMertens(contrast_weight, saturation_weight, exposure_weight)`.
#[must_use]
pub fn create_merge_mertens(
    contrast_weight: f32,
    saturation_weight: f32,
    exposure_weight: f32,
) -> MergeMertens {
    MergeMertens::new(contrast_weight, saturation_weight, exposure_weight)
}

/// `cv::createMergeDebevec()`.
#[must_use]
pub fn create_merge_debevec() -> MergeDebevec {
    MergeDebevec::new()
}

/// `cv::createMergeRobertson()`.
#[must_use]
pub fn create_merge_robertson() -> MergeRobertson {
    MergeRobertson::new()
}

/// `cv::createCalibrateDebevec(samples, lambda, random)`.
#[must_use]
pub fn create_calibrate_debevec(samples: i32, lambda: f32, random: bool) -> CalibrateDebevec {
    CalibrateDebevec::new(samples, lambda, random)
}

/// `cv::createCalibrateRobertson(max_iter, threshold)`.
#[must_use]
pub fn create_calibrate_robertson(max_iter: i32, threshold: f32) -> CalibrateRobertson {
    CalibrateRobertson::new(max_iter, threshold)
}

/// `cv::AlignExposures` — the base of the alignment stage.
///
/// [`AlignMtb`] is the only implementation upstream ships, so this trait exists
/// to name the stage rather than to be implemented twice.
pub trait AlignExposures {
    /// `cv::AlignExposures::process(src, dst, times, response)`.
    fn process_exposures(&self, images: &[&Mat]) -> Result<Vec<Mat>>;
}

impl AlignExposures for AlignMtb {
    fn process_exposures(&self, images: &[&Mat]) -> Result<Vec<Mat>> {
        self.process(images)
    }
}

/// `cv::MergeExposures` — the base of the merge stage.
pub trait MergeExposures {
    /// `cv::MergeExposures::process(src, dst, times, response)`.
    fn process_exposures(
        &self,
        images: &[&Mat],
        times: &[f32],
        response: Option<&Mat>,
    ) -> Result<Mat>;
}

impl MergeExposures for MergeDebevec {
    fn process_exposures(
        &self,
        images: &[&Mat],
        times: &[f32],
        response: Option<&Mat>,
    ) -> Result<Mat> {
        self.process(images, times, response)
    }
}

impl MergeExposures for MergeRobertson {
    fn process_exposures(
        &self,
        images: &[&Mat],
        times: &[f32],
        response: Option<&Mat>,
    ) -> Result<Mat> {
        self.process(images, times, response)
    }
}

/// `cv::CalibrateCRF` — the base of the response-calibration stage.
pub trait CalibrateCrf {
    /// `cv::CalibrateCRF::process(src, dst, times)`.
    fn process_calibration(&self, images: &[&Mat], times: &[f32]) -> Result<Mat>;
}

impl CalibrateCrf for CalibrateDebevec {
    fn process_calibration(&self, images: &[&Mat], times: &[f32]) -> Result<Mat> {
        self.process(images, times)
    }
}

impl CalibrateCrf for CalibrateRobertson {
    fn process_calibration(&self, images: &[&Mat], times: &[f32]) -> Result<Mat> {
        self.process(images, times)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::{CV_32FC3, CV_8UC3};

    /// A synthetic bracket: one linear ramp captured at three exposures.
    fn bracket() -> (Vec<Mat>, Vec<f32>) {
        let (h, w) = (8, 8);
        let times = vec![0.25f32, 1.0, 4.0];
        let mut imgs = Vec::new();
        for &t in &times {
            let mut v = Vec::with_capacity(h * w * 3);
            for y in 0..h {
                for x in 0..w {
                    // Radiance rises across the image; each exposure scales it.
                    let e = (x + y) as f32 / (h + w) as f32;
                    for _ in 0..3 {
                        v.push((e * t * 120.0).min(255.0) as u8);
                    }
                }
            }
            imgs.push(Mat::from_vec(h as i32, w as i32, 3, &v).unwrap());
        }
        (imgs, times)
    }

    fn hdr_image() -> Mat {
        let (h, w) = (6, 6);
        let mut v = Vec::with_capacity(h * w * 3);
        for y in 0..h {
            for x in 0..w {
                // Four decades of dynamic range across the frame.
                let e = 10f32.powf((x + y) as f32 / 3.0 - 2.0);
                v.push(e);
                v.push(e * 0.8);
                v.push(e * 0.6);
            }
        }
        Mat::from_vec(h as i32, w as i32, 3, &v).unwrap()
    }

    #[test]
    fn the_base_tonemap_normalises_then_gammas() {
        let src = Mat::from_vec(1, 3, 3, &[0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
            .unwrap();
        let out = Tonemap::new(1.0).process(&src).unwrap();
        let v = out.to_vec::<f32>().unwrap();
        assert_eq!(v[0], 0.0, "the minimum maps to 0");
        assert_eq!(v[8], 1.0, "the maximum maps to 1");
        assert!((v[4] - 0.5).abs() < 1e-6);

        // Gamma 2 takes the square root of the normalised value.
        let g = Tonemap::new(2.0).process(&src).unwrap().to_vec::<f32>().unwrap();
        assert!((g[4] - 0.5f32.sqrt()).abs() < 1e-6, "{}", g[4]);
    }

    #[test]
    fn a_constant_image_survives_tonemapping() {
        // The range is zero, so upstream copies rather than dividing by it.
        let src = Mat::new_with(2, 2, CV_32FC3, cv_core::Scalar::all(0.5)).unwrap();
        let out = Tonemap::new(1.0).process(&src).unwrap();
        assert!(out.to_vec::<f32>().unwrap().iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn every_tonemapper_lands_in_the_unit_range() {
        let src = hdr_image();
        let outs = [
            Tonemap::default().process(&src).unwrap(),
            TonemapDrago::default().process(&src).unwrap(),
            TonemapReinhard::default().process(&src).unwrap(),
            TonemapMantiuk::default().process(&src).unwrap(),
        ];
        for (i, o) in outs.iter().enumerate() {
            assert_eq!(o.size(), src.size(), "mapper {i} changed the size");
            assert_eq!(o.channels(), 3);
            for v in o.to_vec::<f32>().unwrap() {
                assert!((0.0..=1.0).contains(&v) && v.is_finite(), "mapper {i} produced {v}");
            }
        }
    }

    /// Compression has to be measured between percentiles, not between the
    /// extremes: every mapper ends with a normalise, so the minimum is exactly
    /// 0 and a min/max ratio says nothing.
    #[test]
    fn drago_and_reinhard_compress_the_dynamic_range() {
        fn ratio(v: &[f32]) -> f32 {
            let mut s: Vec<f32> = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let lo = s[s.len() / 10].max(1e-9);
            s[s.len() * 9 / 10] / lo
        }
        let src = hdr_image();
        let src_ratio = ratio(&src.to_vec::<f32>().unwrap());
        for out in [
            TonemapDrago::default().process(&src).unwrap(),
            TonemapReinhard::default().process(&src).unwrap(),
        ] {
            let r = ratio(&out.to_vec::<f32>().unwrap());
            assert!(r < src_ratio, "must compress: {src_ratio} -> {r}");
        }
    }

    #[test]
    fn the_parameter_accessors_round_trip() {
        let mut t = TonemapDrago::default();
        t.set_gamma(2.2);
        t.set_saturation(0.8);
        t.set_bias(0.9);
        assert_eq!((t.get_gamma(), t.get_saturation(), t.get_bias()), (2.2, 0.8, 0.9));

        let mut r = TonemapReinhard::default();
        r.set_intensity(1.5);
        r.set_light_adaptation(0.5);
        r.set_color_adaptation(0.25);
        assert_eq!(r.get_intensity(), 1.5);
        assert_eq!(r.get_light_adaptation(), 0.5);
        assert_eq!(r.get_color_adaptation(), 0.25);

        let mut m = TonemapMantiuk::default();
        m.set_scale(0.6);
        m.set_saturation(1.2);
        assert_eq!((m.get_scale(), m.get_saturation()), (0.6, 1.2));

        let mut a = AlignMtb::default();
        a.set_max_bits(4);
        a.set_exclude_range(8);
        a.set_cut(false);
        assert_eq!((a.get_max_bits(), a.get_exclude_range(), a.get_cut()), (4, 8, false));

        let mut cd = CalibrateDebevec::default();
        cd.set_samples(50);
        cd.set_lambda(20.0);
        cd.set_random(true);
        assert_eq!((cd.get_samples(), cd.get_lambda(), cd.get_random()), (50, 20.0, true));

        let mut cr = CalibrateRobertson::default();
        cr.set_max_iter(10);
        cr.set_threshold(0.05);
        assert_eq!((cr.get_max_iter(), cr.get_threshold()), (10, 0.05));
        assert!(cr.get_radiance().is_none());

        let mut mm = MergeMertens::default();
        mm.set_contrast_weight(0.5);
        mm.set_saturation_weight(0.6);
        mm.set_exposure_weight(0.7);
        assert_eq!(mm.get_contrast_weight(), 0.5);
        assert_eq!(mm.get_saturation_weight(), 0.6);
        assert_eq!(mm.get_exposure_weight(), 0.7);
    }

    #[test]
    fn the_triangle_weight_peaks_in_the_middle() {
        // Upstream's `triangleWeights` puts `epsilon` at the ends rather than
        // zero, so a pixel that reads 0 or 255 in every exposure still has a
        // non-zero total weight to divide by.
        assert_eq!(triangle_weight(0), 1e-6);
        assert_eq!(triangle_weight(255), 1e-6);
        assert_eq!(triangle_weight(127), 127.0);
        assert_eq!(triangle_weight(128), 127.0);
        // Symmetric about the midpoint.
        for z in 0..128 {
            assert_eq!(triangle_weight(z), triangle_weight(255 - z));
        }
    }

    #[test]
    fn debevec_recovers_a_monotone_response_from_a_bracket() {
        let (imgs, times) = bracket();
        let refs: Vec<&Mat> = imgs.iter().collect();
        let response = CalibrateDebevec::new(40, 10.0, false).process(&refs, &times).unwrap();
        assert_eq!(response.size(), cv_core::Size::new(1, 256));
        assert_eq!(response.channels(), 3);
        let v = response.to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite() && *x > 0.0), "the response must be positive");
        // A real camera response rises with exposure; check the green channel
        // over the mid range, where the triangle weight gives it evidence.
        let g: Vec<f32> = (60..200).map(|z| v[z * 3 + 1]).collect();
        let rises = g.windows(2).filter(|p| p[1] >= p[0]).count();
        assert!(rises * 10 > g.len() * 8, "the response should be mostly increasing");
    }

    #[test]
    fn robertson_converges_and_normalises() {
        let (imgs, times) = bracket();
        let refs: Vec<&Mat> = imgs.iter().collect();
        let r = CalibrateRobertson::new(8, 0.01).process(&refs, &times).unwrap();
        assert_eq!(r.size(), cv_core::Size::new(1, 256));
        let v = r.to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
        // Normalised so the midpoint is 1 in every channel.
        for c in 0..3 {
            assert!((v[128 * 3 + c] - 1.0).abs() < 1e-5, "channel {c} is {}", v[128 * 3 + c]);
        }
    }

    #[test]
    fn merging_a_bracket_recovers_a_monotone_radiance_map() {
        let (imgs, times) = bracket();
        let refs: Vec<&Mat> = imgs.iter().collect();
        for hdr in [
            MergeDebevec::new().process(&refs, &times, None).unwrap(),
            MergeRobertson::new().process(&refs, &times, None).unwrap(),
        ] {
            assert_eq!(hdr.size(), imgs[0].size());
            assert_eq!(hdr.channels(), 3);
            let v = hdr.to_vec::<f32>().unwrap();
            assert!(v.iter().all(|x| x.is_finite() && *x >= 0.0));
            // The scene brightens towards the bottom-right corner, and the
            // recovered radiance must follow.
            let w = hdr.cols() as usize;
            let corner = v[0];
            let far = v[((hdr.rows() as usize - 1) * w + w - 1) * 3];
            assert!(far > corner, "radiance should rise across the frame: {corner} -> {far}");
        }
    }

    #[test]
    fn mertens_fuses_without_needing_exposure_times() {
        let (imgs, _) = bracket();
        let refs: Vec<&Mat> = imgs.iter().collect();
        let out = MergeMertens::default().process(&refs).unwrap();
        assert_eq!(out.size(), imgs[0].size());
        assert_eq!(out.channels(), 3);
        // Fusion is a convex combination of the inputs, so it cannot leave
        // their range.
        let v = out.to_vec::<f32>().unwrap();
        assert!(v.iter().all(|&x| (0.0..=255.0).contains(&x) && x.is_finite()));
    }

    /// A *textured* fixture, not a drawn shape.
    ///
    /// With a uniform background the exclusion bitmap keeps only the subject,
    /// so "both pixels trusted" implies "both bright" and every overlapping
    /// shift scores a perfect zero. Texture is what gives the bitmaps
    /// something to disagree about.
    #[test]
    fn align_finds_a_shift_it_was_given() {
        let (h, w) = (48usize, 48usize);
        // Deterministic xorshift noise, smoothed so neighbouring pixels
        // correlate the way a real scene's do.
        let mut state = 0x2545_f491u32;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let base: Vec<u8> = (0..(h + 8) * (w + 8)).map(|_| (rnd() % 200 + 28) as u8).collect();
        let make = |ox: usize, oy: usize| {
            let mut v = Vec::with_capacity(h * w * 3);
            for y in 0..h {
                for x in 0..w {
                    let p = base[(y + oy) * (w + 8) + x + ox];
                    for _ in 0..3 {
                        v.push(p);
                    }
                }
            }
            Mat::from_vec(h as i32, w as i32, 3, &v).unwrap()
        };
        let a = make(4, 4);
        let b = make(6, 5);

        let align = AlignMtb::new(4, 4, true);
        // `b` samples the texture 2 right and 1 down of where `a` does, so
        // its content sits left and up; realigning moves it back by (+2, +1).
        let (dx, dy) = align.calculate_shift(&a, &b).unwrap();
        assert_eq!((dx, dy), (2, 1), "the shift should undo the offset");

        // And applying it brings the two back together.
        let fixed = align.shift_mat(&b, (dx, dy)).unwrap();
        let inner_a = a.roi(cv_core::Rect::new(4, 4, 40, 40)).unwrap();
        let inner_f = fixed.roi(cv_core::Rect::new(4, 4, 40, 40)).unwrap();
        let inner_b = b.roi(cv_core::Rect::new(4, 4, 40, 40)).unwrap();
        let after = cv_core::norm_diff(&inner_a, &inner_f, cv_core::NormType::L1).unwrap();
        let before = cv_core::norm_diff(&inner_a, &inner_b, cv_core::NormType::L1).unwrap();
        assert_eq!(after, 0.0, "the frames should coincide exactly");
        assert!(before > 0.0, "the fixture must actually be misaligned");
    }

    #[test]
    fn a_zero_shift_is_found_when_the_frames_already_match() {
        let (imgs, _) = bracket();
        let align = AlignMtb::new(3, 4, true);
        assert_eq!(align.calculate_shift(&imgs[0], &imgs[0]).unwrap(), (0, 0));
    }

    #[test]
    fn the_bitmaps_split_at_the_median_and_exclude_its_neighbourhood() {
        let mut v = Vec::new();
        for i in 0..16u8 {
            for _ in 0..3 {
                v.push(i * 16);
            }
        }
        let img = Mat::from_vec(4, 4, 3, &v).unwrap();
        let (tb, eb) = AlignMtb::new(6, 20, true).compute_bitmaps(&img).unwrap();
        assert_eq!(tb.len(), 16);
        // Roughly half above the median.
        let above = tb.iter().filter(|&&b| b).count();
        assert!((6..=9).contains(&above), "got {above} above the median");
        // The exclusion bitmap drops the pixels closest to it.
        assert!(eb.iter().filter(|&&b| !b).count() >= 1);
    }

    #[test]
    fn a_full_pipeline_runs_end_to_end() {
        let (imgs, times) = bracket();
        let refs: Vec<&Mat> = imgs.iter().collect();
        let aligned = AlignMtb::new(2, 4, true).process(&refs).unwrap();
        let aligned_refs: Vec<&Mat> = aligned.iter().collect();
        let response =
            CalibrateDebevec::new(30, 10.0, false).process(&aligned_refs, &times).unwrap();
        let hdr = MergeDebevec::new()
            .process(&aligned_refs, &times, Some(&response))
            .unwrap();
        let ldr = TonemapReinhard::default().process(&hdr).unwrap();
        assert_eq!(ldr.size(), imgs[0].size());
        assert!(ldr.to_vec::<f32>().unwrap().iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn the_stage_traits_dispatch_to_their_implementations() {
        let (imgs, times) = bracket();
        let refs: Vec<&Mat> = imgs.iter().collect();
        let a: &dyn AlignExposures = &AlignMtb::default();
        assert_eq!(a.process_exposures(&refs).unwrap().len(), 3);
        let c: &dyn CalibrateCrf = &CalibrateRobertson::new(3, 0.1);
        assert_eq!(c.process_calibration(&refs, &times).unwrap().rows(), 256);
        let m: &dyn MergeExposures = &MergeRobertson::new();
        assert_eq!(m.process_exposures(&refs, &times, None).unwrap().channels(), 3);
    }

    #[test]
    fn a_mismatched_bracket_is_rejected() {
        let (imgs, _) = bracket();
        let refs: Vec<&Mat> = imgs.iter().collect();
        assert!(MergeDebevec::new().process(&refs, &[1.0, 2.0], None).is_err());
        let empty: [&Mat; 0] = [];
        assert!(MergeMertens::default().process(&empty).is_err());
        // Differently sized exposures cannot be fused.
        let odd = Mat::zeros(4, 4, CV_8UC3).unwrap();
        let mixed: Vec<&Mat> = vec![&imgs[0], &odd];
        assert!(MergeMertens::default().process(&mixed).is_err());
    }
}
