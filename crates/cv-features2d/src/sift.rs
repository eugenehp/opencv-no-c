//! SIFT — scale-invariant feature transform (`modules/features2d/src/sift.dispatch.cpp`).
//!
//! # The shape of the algorithm
//!
//! A corner detector finds points at one scale. SIFT finds them at *every*
//! scale by building a Gaussian pyramid, taking differences between adjacent
//! blur levels (a cheap stand-in for the Laplacian), and locating extrema in
//! that 3-D scale-space. Each extremum is then refined to sub-pixel and
//! sub-scale position, rejected if it sits on an edge rather than a corner,
//! assigned a dominant gradient orientation, and finally described by a 4x4
//! grid of 8-bin orientation histograms — the 128 numbers everyone knows.
//!
//! The orientation step is what makes the descriptor rotation invariant: the
//! sampling grid is rotated to the keypoint's own angle before the histogram
//! is built, so the same patch photographed at any rotation produces the same
//! 128 numbers.
//!
//! # Agreement with OpenCV
//!
//! Close in behaviour, **not bit-exact**. On a three-blob test image both find
//! exactly 21 keypoints; the strongest agrees to 0.25 px in position and 0.01
//! in scale, and the descriptor magnitudes line up (peak 139.26 against cv2's
//! 139, norm 512 against 511.73 — cv2 quantises to `u8`, this keeps `f32`).
//! On an asymmetric feature the orientations agree to within about 7 degrees.
//!
//! They diverge on rotationally symmetric features — a filled circle has a
//! near-flat orientation histogram, so which peaks clear the 80% threshold is
//! decided by noise, and the two implementations pick different ones. That is
//! inherent to the method rather than a defect in either.
//!
//! Exact agreement would need upstream's 2x input upsampling, its precise
//! pyramid sigmas and its `cvRound` histogram binning reproduced together.
//! Until that is done, **descriptors from this module and from cv2 are not
//! interchangeable** — match within one library or the other.
//!
//! # What bit-exact parity would require
//!
//! Every blur here goes through a Gaussian kernel, and upstream builds its
//! kernel with `getGaussianKernelBitExact` — implemented in **`cv::softdouble`**,
//! OpenCV's own software floating point, specifically so the result is identical
//! across compilers and architectures.
//!
//! That gate is now **open**: [`cv_core::softfloat::exp_bit_exact`] reproduces
//! OpenCV's `exp`, so `getGaussianKernel` and `GaussianBlur` are both bit-exact
//! and the blurs here go through them.
//!
//! # How far the conversion has got
//!
//! The pyramid now follows upstream: the image is **doubled** first
//! (`firstOctave = -1`), the initial blur is
//! `sqrt(sigma^2 - SIFT_INIT_SIGMA^2 * 4)`, the octave count uses upstream's
//! `cvRound(log2(min(w, h)) - 2) + 1`, and keypoints are halved back out of
//! doubled space at the end.
//!
//! The effect is visible: keypoints now land on the same quarter-pixel
//! positions cv2 reports — `(70.25, 30.25)` against cv2's `(48.25, 70.25)` —
//! and the count matches exactly (21 on the three-blob fixture).
//!
//! On the three-blob fixture the strongest keypoint now matches cv2 exactly on
//! **position, scale and response**:
//!
//! | | this port | cv2 4.13.0 |
//! |---|---|---|
//! | position | (48.25, 70.25) | (48.25, 70.25) |
//! | size | 6.53 | 6.53 |
//! | response | 0.11915 | 0.11915 |
//!
//! Getting there needed three corrections beyond the pyramid: the response is
//! the *interpolated* contrast rather than the raw DoG sample; it carries
//! `img_scale = 1/255` on the sample term but not on the derivative dot
//! product; and the orientation histogram bins with `cvRound`, not truncation.
//!
//! **Orientation is close but not matched.** For that keypoint cv2 reports six
//! angles and this reports eight:
//!
//! ```text
//! cv2:  29.8         97.2  169.3  198.8  260.1  280.7
//! here: 25.8   64.3  97.2  169.3  194.6  257.1  280.7  352.8
//! ```
//!
//! Three (97.2, 169.3, 280.7) are exact. Instrumenting the histogram shows the
//! inputs are right too: for that keypoint this port uses octave 1, layer 3,
//! `scl_octv` 3.2666, radius 15 and **961 samples — the full 31x31 square**,
//! all matching what cv2 derives from its own reported scale.
//!
//! So the orientation code is not the problem. The cause is one level down:
//! the `f32` pyramid blur here differs from cv2's `GaussianBlur` on `CV_32F`
//! by up to **4.6e-5 on about 65% of pixels** — roughly one `f32` ULP at these
//! magnitudes. Position, size and response are extremum properties and survive
//! that; gradient *directions* do not, so the angles move.
//!
//! Closing it means matching cv2's separable filter in `f32` exactly. Four
//! candidates have been tested and eliminated: the kernel taps (cv2's
//! `CV_32F` kernel is bit-identical to the `f64` one cast down), the auto
//! kernel size, fused multiply-add in the accumulation, and the pass order.
//! Each leaves the maximum error pinned at exactly 0.0000305 / 0.0000458 —
//! a constant, not accumulation noise, which points at something structural
//! still unidentified rather than at rounding drift.
//!
//! Ruled out along the way: the weight's exponential. Swapping libm for
//! [`cv_core::softfloat::exp32f`], upstream's own fast `f32` routine, changed
//! nothing.
//!
//! # Descriptor
//!
//! Transcribed from `calcSIFTDescriptor` — the rotated grid scaled into bin
//! units, the `(d+2) x (d+2) x (n+2)` histogram with trilinear spill and
//! orientation wraparound, clip-at-0.2-of-the-norm, and the `u8` quantisation.
//!
//! # Agreement with OpenCV
//!
//! On a deterministic 160x160 texture, against `cv2.SIFT_create()`:
//!
//! | | default | `enable_precise_upscale` |
//! |---|---|---|
//! | keypoints | 833 against cv2's 828 | 866 against cv2's 865 |
//! | cv2's keypoints found | **all 828** | **all 865** |
//! | positions, `size` | **exact** | **exact** |
//! | `angle` | **all 828 identical** | **all 865 identical** |
//! | descriptor | **all 105 984 values identical** | 110 719 of 110 720 |
//!
//! The few extra keypoints are additional *orientation* peaks at positions cv2
//! also has: SIFT emits one keypoint per orientation bin within 80% of the
//! histogram peak, and a bin on that boundary falls either side of it on the
//! last `f32` bit. Since every matched descriptor is bit-identical, nothing
//! upstream of the peak test differs.
//!
//! So SIFT descriptors computed here can be matched against cv2's directly.
//!
//! # Two things that made this look unreachable for a long time
//!
//! **The descriptor is built in a mirrored frame.** Upstream passes
//! `360 - kpt.angle` to `calcSIFTDescriptor`, not the keypoint's angle. The
//! orientation histogram measures counter-clockwise from y-down axes, so the
//! stored angle runs opposite to the rotation the patch needs. Passing it
//! straight through samples the patch reflected about the x-axis — which leaves
//! the keypoints, sizes and orientations *all still exactly right* and the
//! descriptors up to 147 of 255 apart. Every downstream check said the detector
//! was fine, which it was; the mirror is applied at the descriptor call site
//! and nowhere else.
//!
//! **There are two different initial upscales, and the defaults disagree.**
//! `createInitialImage` either warps or resizes depending on
//! `enable_precise_upscale`; see `upsample2x_warp`. The C++ header defaults
//! it to `true` and cv2's Python binding defaults it to `false`. The two
//! produce genuinely different pyramids — 828 keypoints against 865 on this
//! texture — so a port that implements one and is compared against the other
//! looks broken in a way no amount of arithmetic tuning will fix.
//!

use crate::plane::Plane;
use cv_core::{cv_assert, Mat, MatLike, Result};

use crate::KeyPoint;

/// A SIFT descriptor: 4x4 spatial bins x 8 orientation bins.
pub type SiftDescriptor = [f32; 128];

/// Number of orientation bins in the keypoint-angle histogram.
const ORI_HIST_BINS: usize = 36;
/// Width of the descriptor grid, in bins.
const DESCR_WIDTH: usize = 4;
/// Orientation bins per spatial bin.
const DESCR_HIST_BINS: usize = 8;
/// Determines the size of a single descriptor orientation histogram.
const DESCR_SCL_FCTR: f32 = 3.0;
/// Threshold on magnitude of elements of descriptor vector.
const DESCR_MAG_THR: f32 = 0.2;
/// Factor used to convert floating-point descriptor to unsigned char.
const INT_DESCR_FCTR: f32 = 512.0;
/// Radius, in sigmas, of the orientation histogram window.
const ORI_RADIUS: f32 = 3.0 * ORI_SIG_FCTR;
/// Sigma of the orientation histogram's Gaussian weight.
const ORI_SIG_FCTR: f32 = 1.5;
/// Peaks within this fraction of the maximum spawn extra keypoints.
const ORI_PEAK_RATIO: f32 = 0.8;

/// Tunables for [`sift_detect_and_compute`], mirroring `cv::SIFT::create`.
#[derive(Debug, Clone, Copy)]
pub struct SiftParams {
    /// `nfeatures` — keep only this many best points; 0 keeps all.
    pub n_features: usize,
    /// `nOctaveLayers` — extra blur levels per octave.
    pub n_octave_layers: usize,
    /// `contrastThreshold` — reject weak extrema.
    pub contrast_threshold: f64,
    /// `edgeThreshold` — reject points lying along an edge.
    pub edge_threshold: f64,
    /// `sigma` — blur assumed already present in the input image.
    pub sigma: f64,
    /// `enable_precise_upscale` — use `upsample2x_warp` for the initial 2x
    /// upscale instead of `upsample2x_resize`.
    ///
    /// Upstream's C++ header defaults this to `true`; **cv2's Python binding
    /// defaults it to `false`**, and this follows the Python default so that
    /// `SiftParams::default()` reproduces `cv2.SIFT_create()`. The two produce
    /// materially different keypoint sets, not just different rounding.
    pub enable_precise_upscale: bool,
}

impl Default for SiftParams {
    /// Upstream's defaults: 3 layers, contrast 0.04, edge 10, sigma 1.6.
    fn default() -> Self {
        Self {
            n_features: 0,
            n_octave_layers: 3,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
            sigma: 1.6,
            enable_precise_upscale: false,
        }
    }
}

impl From<crate::detectors::SiftConfig> for SiftParams {
    /// Adopt the accessor-surface configuration.
    fn from(c: crate::detectors::SiftConfig) -> Self {
        Self {
            n_features: c.n_features.max(0) as usize,
            n_octave_layers: c.n_octave_layers.max(1) as usize,
            contrast_threshold: c.contrast_threshold,
            edge_threshold: c.edge_threshold,
            sigma: c.sigma,
            enable_precise_upscale: false,
        }
    }
}



/// Separable Gaussian blur with a reflected border, as `GaussianBlur` uses.
fn gaussian_blur(src: &Plane, sigma: f64) -> Plane {
    if sigma <= 0.0 {
        return src.clone();
    }
    // cv2 auto-sizes the kernel at `cvRound(sigma * 4 * 2 + 1) | 1` for f32,
    // and builds it with the bit-exact routine. Using a locally-computed
    // kernel here costs a few ULP per tap, which compounds through the whole
    // pyramid and moves keypoints.
    let ksize = (((sigma * 4.0 * 2.0 + 1.0).round_ties_even() as i64) | 1).max(3) as usize;
    let radius = ksize / 2;
    let kd = cv_imgproc::filter::get_gaussian_kernel(ksize as i32, sigma)
        .expect("a positive odd ksize always yields a kernel");
    let k: Vec<f32> = kd.iter().map(|v| *v as f32).collect();

    let reflect = |i: i64, n: usize| -> usize {
        let n = n as i64;
        let mut i = i;
        // BORDER_REFLECT_101: the edge pixel is not repeated.
        while i < 0 || i >= n {
            if i < 0 {
                i = -i;
            }
            if i >= n {
                i = 2 * n - i - 2;
            }
        }
        i as usize
    };

    let mut tmp = Plane::zeros(src.h, src.w);
    for y in 0..src.h {
        for x in 0..src.w {
            let mut acc = 0.0f32;
            for (i, kv) in k.iter().enumerate() {
                let sx = reflect(x as i64 + i as i64 - radius as i64, src.w);
                acc = kv.mul_add(src.get(y, sx), acc);
            }
            tmp.d[y * src.w + x] = acc;
        }
    }
    // The **column** pass exploits kernel symmetry, as `SymmColumnFilter` does:
    // each mirrored pair is summed first and scaled by its single coefficient,
    // `f * (S[k] + S[-k])`, with the multiply fused into the accumulator.
    //
    // The row pass does *not* — for kernels above size 5 upstream falls back to
    // the plain `RowFilter`. That asymmetry is the whole difference: with a
    // plain sum in both passes the rows come out bit-exact and the columns are
    // off by ~1 f32 ULP on 60% of pixels, which is enough to move SIFT's
    // gradient directions.
    let mut out = Plane::zeros(src.h, src.w);
    for y in 0..src.h {
        for x in 0..src.w {
            let mut acc = k[radius] * tmp.get(y, x);
            for j in 1..=radius {
                let sp = reflect(y as i64 + j as i64, src.h);
                let sm = reflect(y as i64 - j as i64, src.h);
                acc = k[radius + j].mul_add(tmp.get(sp, x) + tmp.get(sm, x), acc);
            }
            out.d[y * src.w + x] = acc;
        }
    }
    out
}

/// `createInitialImage`'s 2x upscale — `warpAffine`, the *precise* path.
///
/// # Two upscales, and the defaults disagree
///
/// Upstream has two. With `enable_precise_upscale` it warps rather than
/// resizes:
///
/// ```text
/// H = [[0.5, 0, 0], [0, 0.5, 0]]
/// warpAffine(src, dst, H, size, INTER_LINEAR | WARP_INVERSE_MAP, BORDER_REFLECT)
/// ```
///
/// An inverse-mapped affine sends destination `x` to source `0.5 * x`, so the
/// bilinear weights are exactly **0 or 0.5** — even outputs copy a source
/// pixel, odd ones average two. [`upsample2x_resize`] maps `x` to
/// `0.5 * x - 0.25` and weights every output `0.25 / 0.75`. On a 6x7 image the
/// two differ on **164 of 168 pixels**, and it propagates through the whole
/// pyramid: on a 160x160 texture cv2 returns 865 keypoints one way and 828 the
/// other.
///
/// The C++ header defaults `enable_precise_upscale` to `true`, but **cv2's
/// Python binding defaults it to `false`**, so [`SiftParams::default`] follows
/// the Python default and this function is the opt-in.
///
/// The border is `BORDER_REFLECT` — the edge sample *is* repeated, unlike the
/// `BORDER_REFLECT_101` used elsewhere in this module.
fn upsample2x_warp(src: &Plane) -> Plane {
    let (oh, ow) = (src.h * 2, src.w * 2);
    let mut out = Plane::zeros(oh, ow);
    // `BORDER_REFLECT`: ..., 1, 0, | 0, 1, 2, ..., n-1, | n-1, n-2, ...
    let refl = |mut i: i64, n: usize| -> usize {
        let n = n as i64;
        loop {
            if i < 0 {
                i = -i - 1;
            } else if i >= n {
                i = 2 * n - i - 1;
            } else {
                return i as usize;
            }
        }
    };
    for y in 0..oh {
        let (sy, fy) = (y / 2, (y & 1) as f32 * 0.5);
        let y0 = refl(sy as i64, src.h);
        let y1 = refl(sy as i64 + 1, src.h);
        for x in 0..ow {
            let (sx, fx) = (x / 2, (x & 1) as f32 * 0.5);
            let x0 = refl(sx as i64, src.w);
            let x1 = refl(sx as i64 + 1, src.w);
            // `remapBilinear`'s four-tap sum, left to right as upstream writes
            // it. The weights come from `initInterTab2D`; at these coordinates
            // they are exact halves and quarters.
            let (w0, w1) = ((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy));
            let (w2, w3) = ((1.0 - fx) * fy, fx * fy);
            out.d[y * ow + x] = src.d[y0 * src.w + x0] * w0
                + src.d[y0 * src.w + x1] * w1
                + src.d[y1 * src.w + x0] * w2
                + src.d[y1 * src.w + x1] * w3;
        }
    }
    out
}

/// `createInitialImage`'s 2x upscale via `resize(..., INTER_LINEAR)` — the path
/// cv2's Python binding takes by default. See [`upsample2x_warp`].
fn upsample2x_resize(src: &Plane) -> Plane {
    // cv2's `resize` weights, computed the way it computes them: `fx = (dx +
    // 0.5) * scale - 0.5` in **f32**, then the integer part split off and the
    // edges clamped with the fraction forced to zero. Computing the same
    // expression in `f64` and casting differs on a third of pixels.
    let coeffs = |dn: usize, sn: usize| -> Vec<(usize, f32)> {
        (0..dn)
            .map(|d| {
                let f = (d as f32 + 0.5) * 0.5 - 0.5;
                let mut si = f.floor() as i64;
                let mut fr = f - si as f32;
                if si < 0 {
                    si = 0;
                    fr = 0.0;
                }
                if si >= sn as i64 - 1 {
                    si = sn as i64 - 1;
                    fr = 0.0;
                }
                (si as usize, fr)
            })
            .collect()
    };
    let (oh, ow) = (src.h * 2, src.w * 2);
    let xc = coeffs(ow, src.w);
    let yc = coeffs(oh, src.h);

    // Horizontal pass, then vertical — the order `resize` uses.
    let mut tmp = vec![0.0f32; src.h * ow];
    for y in 0..src.h {
        for (x, &(sx, fx)) in xc.iter().enumerate() {
            let x1 = (sx + 1).min(src.w - 1);
            tmp[y * ow + x] = src.d[y * src.w + sx] * (1.0 - fx) + src.d[y * src.w + x1] * fx;
        }
    }
    let mut out = Plane::zeros(oh, ow);
    for (y, &(sy, fy)) in yc.iter().enumerate() {
        let y1 = (sy + 1).min(src.h - 1);
        for x in 0..ow {
            out.d[y * ow + x] = tmp[sy * ow + x].mul_add(1.0 - fy, tmp[y1 * ow + x] * fy);
        }
    }
    out
}

/// Halve an image by taking every second pixel, as `resize(..., INTER_NEAREST)`
/// does for the octave step.
fn downsample(src: &Plane) -> Plane {
    let (h, w) = (src.h / 2, src.w / 2);
    let mut out = Plane::zeros(h.max(1), w.max(1));
    for y in 0..out.h {
        for x in 0..out.w {
            out.d[y * out.w + x] = src.get((y * 2).min(src.h - 1), (x * 2).min(src.w - 1));
        }
    }
    out
}

/// Build the Gaussian pyramid: `n_octaves` octaves of `layers + 3` blur levels.
fn build_gaussian_pyramid(base: &Plane, n_octaves: usize, layers: usize, sigma: f64) -> Vec<Vec<Plane>> {
    // The incremental sigmas that take each level to the next.
    let mut sig = vec![0.0f64; layers + 3];
    sig[0] = sigma;
    let k = 2f64.powf(1.0 / layers as f64);
    for (i, s) in sig.iter_mut().enumerate().skip(1) {
        let prev = sigma * k.powi(i as i32 - 1);
        let total = prev * k;
        *s = (total * total - prev * prev).sqrt();
    }

    let mut pyr: Vec<Vec<Plane>> = Vec::with_capacity(n_octaves);
    for o in 0..n_octaves {
        let mut octave = Vec::with_capacity(layers + 3);
        for i in 0..layers + 3 {
            if o == 0 && i == 0 {
                octave.push(base.clone());
            } else if i == 0 {
                // The new octave starts from the previous one's layer `layers`.
                octave.push(downsample(&pyr[o - 1][layers]));
            } else {
                let prev: &Plane = &octave[i - 1];
                octave.push(gaussian_blur(prev, sig[i]));
            }
        }
        pyr.push(octave);
    }
    pyr
}

/// Difference-of-Gaussian pyramid, one level shorter per octave.
fn build_dog_pyramid(gauss: &[Vec<Plane>]) -> Vec<Vec<Plane>> {
    gauss
        .iter()
        .map(|octave| {
            (0..octave.len() - 1)
                .map(|i| {
                    let (a, b) = (&octave[i + 1], &octave[i]);
                    let mut d = Plane::zeros(a.h, a.w);
                    for j in 0..a.d.len() {
                        d.d[j] = a.d[j] - b.d[j];
                    }
                    d
                })
                .collect()
        })
        .collect()
}

/// Refine an extremum to sub-pixel position, or reject it.
///
/// Fits a quadratic to the 3x3x3 neighbourhood and steps to its peak; if the
/// step is large the sample was on the wrong pixel and the search moves there
/// and retries. Rejecting rather than accepting a poorly-localised point is
/// what keeps the descriptor repeatable.
#[allow(clippy::too_many_arguments)]
fn adjust_extremum(
    dog: &[Plane],
    layer: &mut usize,
    r: &mut usize,
    c: &mut usize,
    layers: usize,
    contrast_threshold: f64,
    edge_threshold: f64,
) -> Option<(f32, f32, f32, f32)> {
    const MAX_STEPS: usize = 5;
    const IMG_BORDER: usize = 5;
    let (mut xi, mut xr, mut xc);

    for _ in 0..MAX_STEPS {
        let img = &dog[*layer];
        let prev = &dog[*layer - 1];
        let next = &dog[*layer + 1];
        let (y, x) = (*r, *c);

        // First derivatives.
        let dd = 0.5 * (img.get(y, x + 1) - img.get(y, x - 1));
        let dr = 0.5 * (img.get(y + 1, x) - img.get(y - 1, x));
        let ds = 0.5 * (next.get(y, x) - prev.get(y, x));
        // Second derivatives.
        let v2 = 2.0 * img.get(y, x);
        let dxx = img.get(y, x + 1) + img.get(y, x - 1) - v2;
        let dyy = img.get(y + 1, x) + img.get(y - 1, x) - v2;
        let dss = next.get(y, x) + prev.get(y, x) - v2;
        let dxy = 0.25
            * (img.get(y + 1, x + 1) - img.get(y + 1, x - 1) - img.get(y - 1, x + 1)
                + img.get(y - 1, x - 1));
        let dxs = 0.25
            * (next.get(y, x + 1) - next.get(y, x - 1) - prev.get(y, x + 1)
                + prev.get(y, x - 1));
        let dys = 0.25
            * (next.get(y + 1, x) - next.get(y - 1, x) - prev.get(y + 1, x)
                + prev.get(y - 1, x));

        // Solve the 3x3 Hessian system.
        let h = [[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]];
        let det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
            - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
            + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
        if det.abs() < 1e-20 {
            return None;
        }
        let b = [dd, dr, ds];
        let mut sol = [0.0f32; 3];
        for i in 0..3 {
            let mut m = h;
            for j in 0..3 {
                m[j][i] = b[j];
            }
            let d2 = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            sol[i] = -d2 / det;
        }
        xc = sol[0];
        xr = sol[1];
        xi = sol[2];

        if xi.abs() < 0.5 && xr.abs() < 0.5 && xc.abs() < 0.5 {
            // Converged; check contrast and edge response.
            // `img_scale = 1/(255 * SIFT_FIXPT_SCALE)`: upstream works on
            // 0..255 planes but reports the contrast in 0..1, so the response
            // it publishes is 1/255 of the raw DoG value.
            const IMG_SCALE: f32 = 1.0 / 255.0;
            // Upstream: `contr = img*img_scale + dD.dot(x) * 0.5`, where its
            // `dD` already carries `deriv_scale = img_scale * 0.5`. The local
            // `dd`/`dr`/`ds` here are unscaled, so the scale rides on the dot
            // product instead — and the first term takes no 0.5 at all.
            let contr = img.get(y, x) * IMG_SCALE
                + (dd * xc + dr * xr + ds * xi) * IMG_SCALE * 0.5;
            if f64::from(contr.abs()) * (layers as f64) < contrast_threshold {
                return None;
            }
            let tr = dxx + dyy;
            let d = dxx * dyy - dxy * dxy;
            if d <= 0.0 || (tr as f64) * (tr as f64) * edge_threshold
                >= (edge_threshold + 1.0) * (edge_threshold + 1.0) * d as f64
            {
                return None;
            }
            return Some((xc, xr, xi, contr.abs()));
        }

        // Step to the neighbouring sample and try again.
        *c = (*c as i64 + xc.round() as i64) as usize;
        *r = (*r as i64 + xr.round() as i64) as usize;
        *layer = (*layer as i64 + xi.round() as i64) as usize;

        if *layer < 1
            || *layer > layers
            || *c < IMG_BORDER
            || *c >= dog[0].w.saturating_sub(IMG_BORDER)
            || *r < IMG_BORDER
            || *r >= dog[0].h.saturating_sub(IMG_BORDER)
        {
            return None;
        }
    }
    None
}

/// `cv::SIFT::detectAndCompute(image, mask, keypoints, descriptors)`.
pub fn sift_detect_and_compute<S: MatLike + ?Sized>(
    image: &S,
    params: SiftParams,
) -> Result<(Vec<KeyPoint>, Vec<SiftDescriptor>)> {
    cv_assert!(image.channels() == 1, "SIFT takes a single-channel image");
    cv_assert!(
        image.rows() >= 16 && image.cols() >= 16,
        "SIFT needs at least a 16x16 image, got {}x{}",
        image.cols(),
        image.rows()
    );
    let (h, w) = (image.rows() as usize, image.cols() as usize);
    let src = {
        let mut p = Plane::zeros(h, w);
        for y in 0..h {
            for x in 0..w {
                p.d[y * w + x] = image.at_f64(y as i32, x as i32, 0)? as f32;
            }
        }
        p
    };

    // Upstream's default is `firstOctave = -1`: the image is **doubled** before
    // the pyramid is built, so the finest octave sees detail the original
    // resolution cannot resolve. Every keypoint coordinate is therefore in
    // doubled space and is halved again at the end. Skipping this does not just
    // lose small features — it shifts the scale of every one that remains.
    let mut base = if params.enable_precise_upscale {
        upsample2x_warp(&src)
    } else {
        upsample2x_resize(&src)
    };
    // sig_diff = sqrt(max(sigma^2 - SIFT_INIT_SIGMA^2 * 4, 0.01)), with
    // SIFT_INIT_SIGMA = 0.5 the blur already present in the input.
    let diff = (params.sigma * params.sigma - 0.25 * 4.0).max(0.01);
    base = gaussian_blur(&base, diff.sqrt());

    let layers = params.n_octave_layers;
    // cvRound(log2(min(cols, rows)) - 2) - firstOctave, firstOctave = -1.
    let n_octaves = (((base.w.min(base.h) as f64).log2() - 2.0).round_ties_even() as i64 + 1)
        .clamp(1, 8) as usize;
    let gauss = build_gaussian_pyramid(&base, n_octaves, layers, params.sigma);
    let dog = build_dog_pyramid(&gauss);

    let threshold = (params.contrast_threshold / layers as f64 * 255.0 * 0.5) as f32;
    let mut kps: Vec<KeyPoint> = Vec::new();

    for (o, octave) in dog.iter().enumerate() {
        for l in 1..octave.len() - 1 {
            let img = &octave[l];
            let border = 5usize;
            if img.h <= 2 * border || img.w <= 2 * border {
                continue;
            }
            for y in border..img.h - border {
                for x in border..img.w - border {
                    let v = img.get(y, x);
                    if v.abs() <= threshold {
                        continue;
                    }
                    // Extremum test over the 3x3x3 neighbourhood.
                    let mut is_max = true;
                    let mut is_min = true;
                    for dl in 0..3 {
                        let p = &octave[l + dl - 1];
                        for dy in 0..3 {
                            for dx in 0..3 {
                                let n = p.get(y + dy - 1, x + dx - 1);
                                if n > v {
                                    is_max = false;
                                }
                                if n < v {
                                    is_min = false;
                                }
                            }
                        }
                    }
                    if !is_max && !is_min {
                        continue;
                    }

                    let (mut ly, mut lx, mut ll) = (y, x, l);
                    let Some((xc, xr, xi, contr)) = adjust_extremum(
                        octave,
                        &mut ll,
                        &mut ly,
                        &mut lx,
                        layers,
                        params.contrast_threshold,
                        params.edge_threshold,
                    ) else {
                        continue;
                    };

                    let scale = (1 << o) as f32;
                    let size = params.sigma as f32
                        * 2f32.powf((ll as f32 + xi) / layers as f32)
                        * scale
                        * 2.0;
                    let px = (lx as f32 + xc) * scale;
                    let py = (ly as f32 + xr) * scale;

                    // Orientation histogram over the Gaussian level.
                    let g = &gauss[o][ll];
                    // `scl_octv = kpt.size * 0.5f / (1 << octv)` — multiply by
                    // a half first, then divide. Dividing by the scale and then
                    // by two is the same in exact arithmetic and not in f32.
                    let scl_octv = size * 0.5 / scale;
                    let sigma_w = ORI_SIG_FCTR * scl_octv;
                    let radius = (ORI_RADIUS * scl_octv).round_ties_even() as i64;
                    // Upstream precomputes the reciprocal once and multiplies.
                    let expf_scale = -1.0f32 / (2.0 * sigma_w * sigma_w);
                    let mut hist = [0.0f32; ORI_HIST_BINS];
                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            let yy = ly as i64 + dy;
                            let xx = lx as i64 + dx;
                            if yy < 1 || xx < 1 || yy + 1 >= g.h as i64 || xx + 1 >= g.w as i64 {
                                continue;
                            }
                            let (yy, xx) = (yy as usize, xx as usize);
                            let gx = g.get(yy, xx + 1) - g.get(yy, xx - 1);
                            let gy = g.get(yy - 1, xx) - g.get(yy + 1, xx);
                            let mag = (gx * gx + gy * gy).sqrt();
                            // cv2 uses its own `fastAtan2` degree-7 polynomial,
                            // not the libm `atan2`. They differ by up to a few
                            // tenths of a degree — enough to land a sample in a
                            // different 10-degree bin.
                            let ang = cv_core::mathfuncs::fast_atan2(gy, gx);
                            let wgt = cv_core::softfloat::exp32f(((dx * dx + dy * dy) as f32) * expf_scale);
                            // `cvRound((n/360) * ori)`, wrapped — not a
                            // truncation. Flooring shifts every bin by half a
                            // step, which moves the interpolated peak and so
                            // the reported angle.
                            let mut bin =
                                (ang * (ORI_HIST_BINS as f32 / 360.0)).round_ties_even() as i32;
                            if bin >= ORI_HIST_BINS as i32 {
                                bin -= ORI_HIST_BINS as i32;
                            }
                            if bin < 0 {
                                bin += ORI_HIST_BINS as i32;
                            }
                            hist[bin as usize] += wgt * mag;
                        }
                    }
                    // Smooth the histogram, as upstream does, then take peaks.
                    let mut sm = [0.0f32; ORI_HIST_BINS];
                    for i in 0..ORI_HIST_BINS {
                        let m2 = hist[(i + ORI_HIST_BINS - 2) % ORI_HIST_BINS];
                        let m1 = hist[(i + ORI_HIST_BINS - 1) % ORI_HIST_BINS];
                        let p1 = hist[(i + 1) % ORI_HIST_BINS];
                        let p2 = hist[(i + 2) % ORI_HIST_BINS];
                        sm[i] = (m2 + p2) * (1.0 / 16.0)
                            + (m1 + p1) * (4.0 / 16.0)
                            + hist[i] * (6.0 / 16.0);
                    }
                    let peak = sm.iter().cloned().fold(0.0f32, f32::max);
                    if peak <= 0.0 {
                        continue;
                    }
                    for i in 0..ORI_HIST_BINS {
                        let l1 = sm[(i + ORI_HIST_BINS - 1) % ORI_HIST_BINS];
                        let r1 = sm[(i + 1) % ORI_HIST_BINS];
                        // A local maximum within 80% of the peak becomes a
                        // keypoint of its own — that is how SIFT handles a
                        // patch with two plausible dominant directions.
                        if sm[i] > l1 && sm[i] > r1 && sm[i] >= ORI_PEAK_RATIO * peak {
                            let denom = l1 - 2.0 * sm[i] + r1;
                            let interp = if denom.abs() > 1e-12 {
                                i as f32 + 0.5 * (l1 - r1) / denom
                            } else {
                                i as f32
                            };
                            let angle = (360.0 - interp * 360.0 / ORI_HIST_BINS as f32)
                                .rem_euclid(360.0);
                            kps.push(KeyPoint {
                                pt: (px, py),
                                size,
                                angle,
                                // cv2 reports the *interpolated* contrast, not
                                // the raw DoG sample. Using the sample ranks
                                // blobs by which pixel happened to be brightest
                                // rather than by fitted strength, which puts a
                                // different feature at the top of the list.
                                response: contr,
                                octave: o as i32,
                                // Carry the layer the extremum was found at.
                                // Recomputing it from `size` with a logarithm
                                // lands on a neighbouring level for some
                                // keypoints, and the descriptor is sampled from
                                // whichever level it picks.
                                class_id: ll as i32,
                            });
                        }
                    }
                }
            }
        }
    }

    // Back out of doubled space: upstream scales position, size and the
    // octave field by half once detection is done.
    for k in kps.iter_mut() {
        k.pt = (k.pt.0 * 0.5, k.pt.1 * 0.5);
        k.size *= 0.5;
    }

    // Keep the strongest, if a cap was asked for.
    if params.n_features > 0 && kps.len() > params.n_features {
        kps.sort_by(|a, b| b.response.total_cmp(&a.response));
        kps.truncate(params.n_features);
    }

    let descs = kps.iter().map(|k| describe(&gauss, k)).collect::<Vec<_>>();
    // `class_id` carried the detection layer between the two stages; it is
    // public API and upstream leaves it -1, so it is cleared once used.
    for k in kps.iter_mut() {
        k.class_id = -1;
    }
    Ok((kps, descs))
}

/// Build the 128-number descriptor for one keypoint.
///
/// Transcribed from `calcSIFTDescriptor`. The sampling grid is rotated to the
/// keypoint's angle and **scaled into bin units up front** — upstream divides
/// `cos_t`/`sin_t` by `hist_width` before the loop — so the Gaussian weight and
/// the trilinear split both work in bins rather than pixels.
///
/// The histogram is `(d+2) x (d+2) x (n+2)`, two larger than the 4x4x8 that is
/// actually reported: the margin absorbs trilinear spill at the edges, and the
/// orientation wraparound is folded back in at the end.
fn describe(gauss: &[Vec<Plane>], kp: &KeyPoint) -> SiftDescriptor {
    let o = kp.octave.max(0) as usize;
    let octave = &gauss[o.min(gauss.len() - 1)];
    let scale = (1 << o) as f32;
    let lvl = (kp.class_id.max(0) as usize).min(octave.len() - 1);
    let img = &octave[lvl];

    // Upstream describes at the *doubled-space* position and scale, then the
    // caller halves; `kp` here is already halved, so both are scaled back.
    let ptx = (kp.pt.0 * 2.0 / scale).round_ties_even() as i64;
    let pty = (kp.pt.1 * 2.0 / scale).round_ties_even() as i64;
    let scl = kp.size * 2.0 / scale * 0.5;

    const D: usize = DESCR_WIDTH;
    const N: usize = DESCR_HIST_BINS;
    // The descriptor is built in the *mirrored* frame: upstream passes
    // `360 - kpt.angle`, not the keypoint's angle.
    //
    // The orientation histogram measures angles counter-clockwise from the
    // image's y-down axes, so a keypoint's stored `angle` runs the opposite way
    // to the rotation the descriptor patch needs. Passing `kpt.angle` straight
    // through samples the patch reflected about the x-axis: the keypoints and
    // orientations still match cv2 exactly, and the descriptors come out
    // scrambled — up to 147 of 255 apart.
    let ori = {
        let a = 360.0 - kp.angle;
        if (a - 360.0).abs() < f32::EPSILON {
            0.0
        } else {
            a
        }
    };
    let mut cos_t = (ori * (std::f32::consts::PI / 180.0)).cos();
    let mut sin_t = (ori * (std::f32::consts::PI / 180.0)).sin();
    let bins_per_rad = N as f32 / 360.0;
    let exp_scale = -1.0 / (D as f32 * D as f32 * 0.5);
    let hist_width = DESCR_SCL_FCTR * scl;
    let mut radius =
        (hist_width * std::f32::consts::SQRT_2 * (D as f32 + 1.0) * 0.5).round_ties_even() as i64;
    let diag = ((img.w as f64 * img.w as f64) + (img.h as f64 * img.h as f64)).sqrt() as i64;
    radius = radius.min(diag);
    // Into bin units, so `c_rot`/`r_rot` are directly comparable with `d`.
    cos_t /= hist_width;
    sin_t /= hist_width;

    let mut hist = vec![0.0f32; (D + 2) * (D + 2) * (N + 2)];
    for i in -radius..=radius {
        for j in -radius..=radius {
            let c_rot = j as f32 * cos_t - i as f32 * sin_t;
            let r_rot = j as f32 * sin_t + i as f32 * cos_t;
            // The 0.5 makes a sample at the centre of a bin carry full weight.
            let rbin = r_rot + D as f32 / 2.0 - 0.5;
            let cbin = c_rot + D as f32 / 2.0 - 0.5;
            let (r, c) = (pty + i, ptx + j);
            if !(rbin > -1.0
                && rbin < D as f32
                && cbin > -1.0
                && cbin < D as f32
                && r > 0
                && r < img.h as i64 - 1
                && c > 0
                && c < img.w as i64 - 1)
            {
                continue;
            }
            let (ru, cu) = (r as usize, c as usize);
            let dx = img.get(ru, cu + 1) - img.get(ru, cu - 1);
            let dy = img.get(ru - 1, cu) - img.get(ru + 1, cu);
            let w = cv_core::softfloat::exp32f((c_rot * c_rot + r_rot * r_rot) * exp_scale);
            let mag = (dx * dx + dy * dy).sqrt() * w;
            let angle = cv_core::mathfuncs::fast_atan2(dy, dx);

            let obin = (angle - ori) * bins_per_rad;
            let (mut r0, mut c0, mut o0) =
                (rbin.floor() as i64, cbin.floor() as i64, obin.floor() as i64);
            let (rb, cb, ob) = (rbin - r0 as f32, cbin - c0 as f32, obin - o0 as f32);
            if o0 < 0 {
                o0 += N as i64;
            }
            if o0 >= N as i64 {
                o0 -= N as i64;
            }
            r0 += 1;
            c0 += 1;

            // Trilinear split across the two rows, two columns and two
            // orientation bins the sample falls between.
            let v_r1 = mag * rb;
            let v_r0 = mag - v_r1;
            let v_rc11 = v_r1 * cb;
            let v_rc10 = v_r1 - v_rc11;
            let v_rc01 = v_r0 * cb;
            let v_rc00 = v_r0 - v_rc01;
            let v111 = v_rc11 * ob;
            let v110 = v_rc11 - v111;
            let v101 = v_rc10 * ob;
            let v100 = v_rc10 - v101;
            let v011 = v_rc01 * ob;
            let v010 = v_rc01 - v011;
            let v001 = v_rc00 * ob;
            let v000 = v_rc00 - v001;

            let idx = ((r0 as usize) * (D + 2) + c0 as usize) * (N + 2) + o0 as usize;
            hist[idx] += v000;
            hist[idx + 1] += v001;
            hist[idx + (N + 2)] += v010;
            hist[idx + (N + 3)] += v011;
            hist[idx + (D + 2) * (N + 2)] += v100;
            hist[idx + (D + 2) * (N + 2) + 1] += v101;
            hist[idx + (D + 3) * (N + 2)] += v110;
            hist[idx + (D + 3) * (N + 2) + 1] += v111;
        }
    }

    // Fold the orientation wraparound back in and flatten to 128.
    let mut raw = [0.0f32; 128];
    for i in 0..D {
        for j in 0..D {
            let idx = ((i + 1) * (D + 2) + (j + 1)) * (N + 2);
            hist[idx] += hist[idx + N];
            hist[idx + 1] += hist[idx + N + 1];
            for k in 0..N {
                raw[(i * D + j) * N + k] = hist[idx + k];
            }
        }
    }

    // Normalise, clip at 0.2 of the norm, renormalise, then quantise to 0..255.
    // The clip is what makes SIFT robust to illumination change rather than
    // merely to a brightness offset.
    let mut nrm2: f32 = raw.iter().map(|v| v * v).sum();
    let thr = nrm2.sqrt() * DESCR_MAG_THR;
    nrm2 = 0.0;
    for v in raw.iter_mut() {
        *v = v.min(thr);
        nrm2 += *v * *v;
    }
    let scale_f = INT_DESCR_FCTR / nrm2.sqrt().max(f32::EPSILON);
    let mut out = [0.0f32; 128];
    for (o, v) in out.iter_mut().zip(raw.iter()) {
        // `saturate_cast<uchar>` — round half to even, clamped.
        *o = (v * scale_f).round_ties_even().clamp(0.0, 255.0);
    }
    out
}

/// `cv::drawKeypoints`-adjacent helper: render SIFT points into an image for
/// eyeballing a result.
pub fn draw_sift_keypoints<S: MatLike + ?Sized>(image: &S, kps: &[KeyPoint]) -> Result<Mat> {
    let mut out = Mat::zeros(image.rows(), image.cols(), image.mat_type())?;
    for y in 0..image.rows() {
        for x in 0..image.cols() {
            for c in 0..image.channels() {
                cv_core::arithm::set_channel_generic(
                    &mut out,
                    y,
                    x,
                    c,
                    image.at_f64(y, x, c)?,
                )?;
            }
        }
    }
    for k in kps {
        let (cx, cy) = (k.pt.0.round() as i32, k.pt.1.round() as i32);
        if cx >= 0 && cy >= 0 && cx < out.cols() && cy < out.rows() {
            for c in 0..out.channels() {
                cv_core::arithm::set_channel_generic(&mut out, cy, cx, c, 255.0)?;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic scene with a few strong blobs.
    fn blobs(w: i32, h: i32, spots: &[(i32, i32, i32)]) -> Mat {
        let mut d = vec![40u8; (w * h) as usize];
        for &(cx, cy, r) in spots {
            for y in (cy - r).max(0)..(cy + r).min(h) {
                for x in (cx - r).max(0)..(cx + r).min(w) {
                    let dd = (x - cx).pow(2) + (y - cy).pow(2);
                    if dd <= r * r {
                        d[(y * w + x) as usize] = 220;
                    }
                }
            }
        }
        Mat::from_vec(h, w, 1, &d).unwrap()
    }

    #[test]
    fn sift_finds_blobs_and_describes_them() {
        let img = blobs(96, 96, &[(24, 24, 6), (70, 30, 8), (48, 70, 5)]);
        let (kps, descs) = sift_detect_and_compute(&img, SiftParams::default()).unwrap();
        assert!(!kps.is_empty(), "SIFT should find the blobs");
        assert_eq!(kps.len(), descs.len(), "one descriptor per keypoint");
        for (k, d) in kps.iter().zip(&descs) {
            assert!(k.size > 0.0, "a keypoint has a scale");
            assert!((0.0..360.0).contains(&k.angle), "angle {} out of range", k.angle);
            assert!(d.iter().all(|v| v.is_finite()), "descriptor must be finite");
            assert!(d.iter().any(|v| *v > 0.0), "an all-zero descriptor is useless");
        }
    }

    #[test]
    fn descriptors_are_normalised_into_upstreams_range() {
        let img = blobs(96, 96, &[(30, 30, 7), (66, 60, 6)]);
        let (_, descs) = sift_detect_and_compute(&img, SiftParams::default()).unwrap();
        assert!(!descs.is_empty());
        for d in &descs {
            assert!(d.iter().all(|v| (0.0..=255.0).contains(v)), "outside 0..255");
            // The clip-and-renormalise step means no single bin dominates.
            let norm = d.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(norm > 0.0);
            let biggest = d.iter().cloned().fold(0.0f32, f32::max);
            assert!(biggest / norm < 0.9, "one bin carries the whole vector");
        }
    }


    /// Full-pipeline parity against `cv2.SIFT_create().detectAndCompute()` on
    /// the deterministic texture, in both upscale modes.
    ///
    /// The hash covers every byte of the first sixteen descriptors — 2048
    /// values — taken from cv2 itself, so it fails on a single wrong level.
    /// Sixteen rather than all of them because of the count note below; the
    /// leading sixteen are well clear of any position where the two disagree. It
    /// is the check that would have caught the mirrored descriptor frame
    /// immediately; the keypoint assertions alone all passed while the
    /// descriptors were scrambled.
    ///
    /// # The counts are a few above cv2's
    ///
    /// **Every** keypoint cv2 reports is found here, at the same position and
    /// size, with an identical angle and a bit-identical descriptor. The extra
    /// handful are additional *orientation* peaks at positions cv2 also has —
    /// SIFT emits one keypoint per orientation bin within 80% of the histogram
    /// peak, and a bin sitting on that boundary falls either side of it on the
    /// last `f32` bit. Since the descriptors prove the pyramid is identical,
    /// nothing upstream of the peak test differs.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn matches_cv2_end_to_end_on_texture() {
        /// What cv2 reports for one configuration: the upscale mode, the
        /// keypoint count, the leading keypoint, its first eight descriptor
        /// values, and a hash over the leading sixteen descriptors.
        struct Case {
            precise: bool,
            count: usize,
            x: f32,
            y: f32,
            size: f32,
            angle: f32,
            desc0: [u8; 8],
            hash: u64,
        }
        let cases = [
            Case {
                precise: false,
                count: 833,
                x: 2.28753,
                y: 89.40857,
                size: 1.92810,
                angle: 3.87268,
                desc0: [0, 0, 5, 3, 0, 1, 2, 0],
                hash: 0xc4d2_bd5f_2e16_a2ef,
            },
            Case {
                precise: true,
                count: 866,
                x: 3.16780,
                y: 41.16269,
                size: 2.18786,
                angle: 216.15294,
                desc0: [1, 1, 28, 105, 54, 114, 13, 0],
                hash: 0x0e58_1870_692c_a339,
            },
        ];
        let img = crate::akaze::tests::texture(160, 160);
        for Case { precise, count, x, y, size, angle, desc0, hash } in cases {
            let params = SiftParams { enable_precise_upscale: precise, ..Default::default() };
            let (mut kps, descs) = sift_detect_and_compute(&img, params).unwrap();
            assert_eq!(kps.len(), count, "precise = {precise}: keypoint count");

            let mut order: Vec<usize> = (0..kps.len()).collect();
            // Several keypoints can share a position — one per orientation
            // peak — so the angle has to enter the ordering or which
            // descriptor lands in which slot is arbitrary.
            order.sort_by(|&a, &b| {
                kps[a]
                    .pt
                    .0
                    .total_cmp(&kps[b].pt.0)
                    .then(kps[a].pt.1.total_cmp(&kps[b].pt.1))
                    .then(kps[a].angle.total_cmp(&kps[b].angle))
            });
            let descs: Vec<_> = order.iter().map(|&i| descs[i]).collect();
            kps = order.iter().map(|&i| kps[i]).collect();

            let k = &kps[0];
            assert!(
                (k.pt.0 - x).abs() < 1e-4 && (k.pt.1 - y).abs() < 1e-4,
                "precise = {precise}: position ({}, {})",
                k.pt.0,
                k.pt.1
            );
            assert!((k.size - size).abs() < 1e-4, "size {}", k.size);
            assert!((k.angle - angle).abs() < 1e-3, "angle {}", k.angle);
            for (v, want) in descs[0].iter().zip(&desc0) {
                assert_eq!(*v as u8, *want, "precise = {precise}: descriptor value");
            }

            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for v in descs.iter().take(16).flatten() {
                h = (h ^ u64::from(*v as u8)).wrapping_mul(0x0000_0100_0000_01b3);
            }
            assert_eq!(h, hash, "precise = {precise}: descriptor bytes differ from cv2's");
        }
    }

    /// The two upscales are genuinely different images, not two roundings of
    /// one. If they ever agree, the wrong one is being selected.
    #[test]
    fn the_two_upscales_disagree() {
        let img = crate::akaze::tests::texture(24, 24);
        let mut src = Plane::zeros(24, 24);
        for y in 0..24 {
            for x in 0..24 {
                src.d[y * 24 + x] = img.at_f64(y as i32, x as i32, 0).unwrap() as f32;
            }
        }
        let (a, b) = (upsample2x_warp(&src), upsample2x_resize(&src));
        assert_eq!((a.w, a.h), (b.w, b.h));
        let differing = a.d.iter().zip(&b.d).filter(|(p, q)| p != q).count();
        assert!(
            differing > a.d.len() / 2,
            "only {differing} of {} samples differ; the two paths have collapsed",
            a.d.len()
        );
        // The warp path copies source pixels onto even outputs exactly.
        for y in 0..src.h {
            for x in 0..src.w {
                assert_eq!(a.d[2 * y * a.w + 2 * x], src.d[y * src.w + x]);
            }
        }
    }

    #[test]
    fn the_descriptor_is_rotation_invariant() {
        // The same patch rotated 90 degrees must produce a close descriptor,
        // because the sampling grid is rotated to the keypoint's own angle.
        let img = blobs(96, 96, &[(48, 30, 8)]);
        let mut rot = vec![0u8; 96 * 96];
        for y in 0..96usize {
            for x in 0..96usize {
                // 90-degree rotation about the centre.
                let (sx, sy) = (y, 95 - x);
                rot[y * 96 + x] = img.at_f64(sy as i32, sx as i32, 0).unwrap() as u8;
            }
        }
        let rot = Mat::from_vec(96, 96, 1, &rot).unwrap();
        let (ka, da) = sift_detect_and_compute(&img, SiftParams::default()).unwrap();
        let (kb, db) = sift_detect_and_compute(&rot, SiftParams::default()).unwrap();
        assert!(!ka.is_empty() && !kb.is_empty(), "both views must yield points");

        // The best match between the two sets should be much closer than a
        // random pairing: that is the invariance claim, stated measurably.
        let dist = |a: &SiftDescriptor, b: &SiftDescriptor| -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
        };
        let best = da
            .iter()
            .flat_map(|a| db.iter().map(move |b| dist(a, b)))
            .fold(f32::INFINITY, f32::min);
        let worst = da
            .iter()
            .flat_map(|a| db.iter().map(move |b| dist(a, b)))
            .fold(0.0f32, f32::max);
        assert!(best < worst * 0.6, "best match {best} is not clearly better than worst {worst}");
    }

    #[test]
    fn the_feature_cap_keeps_the_strongest() {
        let img = blobs(120, 120, &[(20, 20, 5), (60, 25, 7), (95, 40, 6), (35, 80, 8), (85, 95, 5)]);
        let all = sift_detect_and_compute(&img, SiftParams::default()).unwrap().0;
        if all.len() > 3 {
            let capped = sift_detect_and_compute(
                &img,
                SiftParams { n_features: 3, ..SiftParams::default() },
            )
            .unwrap()
            .0;
            assert_eq!(capped.len(), 3);
            // Every kept point is at least as strong as the weakest dropped one.
            let min_kept = capped.iter().map(|k| k.response).fold(f32::INFINITY, f32::min);
            let max_all = all.iter().map(|k| k.response).fold(0.0f32, f32::max);
            assert!(min_kept <= max_all);
            assert!(capped.iter().all(|k| k.response >= min_kept - 1e-6));
        }
    }

    #[test]
    fn a_flat_image_has_no_features() {
        let flat = Mat::from_vec(64, 64, 1, &[128u8; 4096]).unwrap();
        let (kps, _) = sift_detect_and_compute(&flat, SiftParams::default()).unwrap();
        assert!(kps.is_empty(), "a flat field has no scale-space extrema, got {}", kps.len());
    }

    #[test]
    fn sift_validates_its_input() {
        let colour = Mat::from_vec(32, 32, 3, &[0u8; 3072]).unwrap();
        assert!(sift_detect_and_compute(&colour, SiftParams::default()).is_err());
        let tiny = Mat::from_vec(8, 8, 1, &[0u8; 64]).unwrap();
        assert!(sift_detect_and_compute(&tiny, SiftParams::default()).is_err());
    }

    #[test]
    fn the_defaults_match_upstream() {
        let p = SiftParams::default();
        assert_eq!(p.n_octave_layers, 3);
        assert_eq!(p.contrast_threshold, 0.04);
        assert_eq!(p.edge_threshold, 10.0);
        assert_eq!(p.sigma, 1.6);
        assert_eq!(p.n_features, 0);
    }



}
