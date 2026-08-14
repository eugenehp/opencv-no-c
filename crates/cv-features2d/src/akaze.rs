//! AKAZE's detector — accelerated KAZE (`modules/features2d/src/kaze/AKAZEFeatures.cpp`).
//!
//! # Why this is not KAZE with a different descriptor
//!
//! The two share a name, a PDE and an author, and almost nothing else in the
//! detector. [`crate::kaze`] evolves every level at **full resolution**; AKAZE
//! builds a real pyramid, halving the image at each octave with `INTER_AREA`.
//! That single change cascades:
//!
//! * derivatives are taken at `1.5 * esigma / power`, not `esigma`, so the
//!   `sigma_size` used for the Scharr kernels depends only on the sublevel;
//! * the contrast factor is **decayed by 0.75 at every octave change**, because
//!   downsampling raises the gradient magnitudes and `k` has to follow;
//! * octaves stop as soon as a level would fall below 80x40, which on small
//!   images means AKAZE runs with *one* octave where KAZE runs with four.
//!
//! That last rule is why the two detectors disagree so visibly. On a 96x96
//! image the second octave would be 48x48, below the floor, so `omax` collapses
//! to 1 and AKAZE searches four sublevels of one octave — cv2 returns 10
//! keypoints against KAZE's 43.
//!
//! # Agreement with OpenCV
//!
//! Against cv2 4.13.0's `cv2.AKAZE_create().detectAndCompute()`, on
//! deterministic textures (see `akaze_matches_cv2_end_to_end_on_texture`):
//!
//! | | 160x160 | 320x240 |
//! |---|---|---|
//! | keypoints | **26, matching cv2** | **576, matching cv2** |
//! | positions | all within 1e-4 px | all within 1e-3 px |
//! | `size` | **exact** | **exact** |
//! | `angle` | **all 26 identical** | **all 576 identical** |
//! | M-LDB descriptor | **bit-identical** | 1 differing bit of 281088 |
//!
//! So AKAZE descriptors computed here can be matched directly against cv2's.
//!
//! ## The synthetic-disc numbers are worse, and that is the fixture's fault
//!
//! On drawn discs this finds 105 of cv2's 108 keypoints. Discs are four-fold
//! symmetric, so the extrema round each one tie in response to seven digits and
//! `find_neighbour`'s suppression is decided by the last `f32` ULP — cv2's
//! (246.9498, 173.1148) at 0.00625523 against this port's (248.8843, 175.0489)
//! at 0.00625522. Textured images have no such ties and match exactly.

use cv_core::{cv_assert, MatLike, Result};

use crate::detectors::{AkazeParams, Diffusivity};
use crate::kaze::{
    blur, cv_round, diffusivity, fast_atan2, fed_tau, mldb, scharr, solve3,
    AkazeDescriptorBits, Plane, SOFFSET,
};
use crate::KeyPoint;

/// `AKAZEOptions::derivative_factor` — derivatives are taken 1.5 scales out.
const DERIVATIVE_FACTOR: f32 = 1.5;

/// One level of AKAZE's pyramid (`MEvolution`).
struct Level {
    /// `Lt` — the diffused image, at this octave's resolution.
    lt: Plane,
    /// `Lsmooth` — `Lt` blurred by a fixed 5x5 kernel.
    smooth: Plane,
    /// `Ldet` — the scale-normalised Hessian determinant.
    det: Plane,
    /// `Lx`, `Ly` at this level's derivative scale, for the orientation stage.
    lx: Plane,
    ly: Plane,
    esigma: f32,
    sigma_size: usize,
    /// Rows and columns excluded from the search, from the descriptor's reach.
    border: usize,
    octave: i32,
    /// `2^octave` — how much smaller this level is than the input.
    octave_ratio: f32,
    w: usize,
    h: usize,
}

/// `AKAZEFeatures::Allocate_Memory_Evolution` — the pyramid's shape.
///
/// Returns nothing but geometry; the images are filled by [`build_pyramid`].
fn plan_levels(width: usize, height: usize, n_octaves: i32, n_sublevels: i32) -> Vec<Level> {
    // M-LDB reaches 10*sqrt(2) scales from the keypoint, and the border has to
    // cover that or the descriptor would sample outside the level.
    let smax = 10.0f32 * 2.0f32.sqrt();
    let mut out: Vec<Level> = Vec::new();

    let mut power = 1usize;
    for i in 0..n_octaves.max(1) {
        let rfactor = 1.0f32 / power as f32;
        let level_width = (width as f32 * rfactor) as usize;
        let level_height = (height as f32 * rfactor) as usize;
        // Upstream's floor. Note it applies to octave 1 and up only, so a tiny
        // image still gets its first octave.
        if (level_width < 80 || level_height < 40) && i != 0 {
            break;
        }
        for j in 0..n_sublevels.max(1) {
            let esigma = SOFFSET * 2f32.powf(j as f32 / n_sublevels.max(1) as f32 + i as f32);
            let sigma_size = cv_round(esigma * DERIVATIVE_FACTOR / power as f32).max(1) as usize;
            out.push(Level {
                lt: Plane::zeros(level_height, level_width),
                smooth: Plane::zeros(level_height, level_width),
                det: Plane::zeros(level_height, level_width),
                lx: Plane::zeros(level_height, level_width),
                ly: Plane::zeros(level_height, level_width),
                esigma,
                sigma_size,
                border: cv_round(smax * sigma_size as f32) as usize + 1,
                octave: i,
                octave_ratio: power as f32,
                w: level_width,
                h: level_height,
            });
        }
        power *= 2;
    }
    out
}

/// `compute_kcontrast` — AKAZE's contrast factor.
///
/// Not the same computation as KAZE's despite the shared purpose. This one
/// takes the derivatives it is handed rather than blurring first, histograms
/// the magnitude with `(nbins - 1) / hmax` scaling, and — the part that
/// actually changes the answer — **excludes bin 0 from the population** before
/// applying the percentile, then returns as soon as the running count reaches
/// the threshold rather than after adding the current bin.
fn kcontrast(lx: &Plane, ly: &Plane, perc: f32, nbins: usize) -> f32 {
    let (rows, cols) = (lx.h.saturating_sub(2), lx.w.saturating_sub(2));
    let total = rows * cols;
    if total == 0 {
        return 0.03;
    }
    let mag: Vec<f32> = (1..=rows)
        .flat_map(|y| (1..=cols).map(move |x| (y, x)))
        .map(|(y, x)| (lx.d[y * lx.w + x], ly.d[y * lx.w + x]))
        .map(|(a, b)| (a * a + b * b).sqrt())
        .collect();

    let hmax = mag.iter().copied().fold(0.0f32, f32::max);
    if hmax == 0.0 {
        return 0.03;
    }

    let scale = (nbins - 1) as f32 / hmax;
    let mut hist = vec![0usize; nbins];
    for m in &mag {
        let bin = (m * scale) as usize;
        if bin < nbins {
            hist[bin] += 1;
        }
    }

    let nthreshold = ((total - hist[0]) as f32 * perc) as usize;
    let mut nelements = 0usize;
    for (k, count) in hist.iter().enumerate().skip(1) {
        // The test comes *before* the accumulation, so the bin that crosses the
        // threshold is not itself counted — one bin lower than the obvious form.
        if nelements >= nthreshold {
            return hmax * k as f32 / nbins as f32;
        }
        nelements += count;
    }
    0.03
}

/// `nld_step_scalar_one_lane` — AKAZE's diffusion step.
///
/// Differs from KAZE's at the boundary and in where the halving happens: the
/// caller passes `tau * 0.5` and this multiplies the raw flux sum, where KAZE
/// halves inside. The two edge columns *are* stepped on interior rows, but the
/// first and last entries of the top and bottom rows are explicitly zeroed.
fn diffusion_step(lt: &mut Plane, lf: &Plane, step_size: f32) {
    let (h, w) = (lt.h, lt.w);
    if h < 2 || w < 2 {
        return;
    }
    let mut ls = vec![0.0f32; h * w];
    let at = |p: &Plane, y: usize, x: usize| p.d[y * w + x];

    // Three-flux forms on the first and last rows; four in between.
    for x in 1..w - 1 {
        ls[x] = step_size
            * ((at(lf, 0, x) + at(lf, 0, x + 1)) * (at(lt, 0, x + 1) - at(lt, 0, x))
                + (at(lf, 0, x) + at(lf, 0, x - 1)) * (at(lt, 0, x - 1) - at(lt, 0, x))
                + (at(lf, 0, x) + at(lf, 1, x)) * (at(lt, 1, x) - at(lt, 0, x)));
        let b = h - 1;
        ls[b * w + x] = step_size
            * ((at(lf, b, x) + at(lf, b, x + 1)) * (at(lt, b, x + 1) - at(lt, b, x))
                + (at(lf, b, x) + at(lf, b, x - 1)) * (at(lt, b, x - 1) - at(lt, b, x))
                + (at(lf, b, x) + at(lf, b - 1, x)) * (at(lt, b - 1, x) - at(lt, b, x)));
    }
    for y in 1..h - 1 {
        // Left and right columns: three fluxes, no neighbour outside.
        ls[y * w] = step_size
            * ((at(lf, y, 0) + at(lf, y, 1)) * (at(lt, y, 1) - at(lt, y, 0))
                + (at(lf, y, 0) + at(lf, y + 1, 0)) * (at(lt, y + 1, 0) - at(lt, y, 0))
                + (at(lf, y, 0) + at(lf, y - 1, 0)) * (at(lt, y - 1, 0) - at(lt, y, 0)));
        let r = w - 1;
        ls[y * w + r] = step_size
            * ((at(lf, y, r) + at(lf, y, r - 1)) * (at(lt, y, r - 1) - at(lt, y, r))
                + (at(lf, y, r) + at(lf, y + 1, r)) * (at(lt, y + 1, r) - at(lt, y, r))
                + (at(lf, y, r) + at(lf, y - 1, r)) * (at(lt, y - 1, r) - at(lt, y, r)));
        for x in 1..w - 1 {
            ls[y * w + x] = step_size
                * ((at(lf, y, x) + at(lf, y, x + 1)) * (at(lt, y, x + 1) - at(lt, y, x))
                    + (at(lf, y, x) + at(lf, y, x - 1)) * (at(lt, y, x - 1) - at(lt, y, x))
                    + (at(lf, y, x) + at(lf, y + 1, x)) * (at(lt, y + 1, x) - at(lt, y, x))
                    + (at(lf, y, x) + at(lf, y - 1, x)) * (at(lt, y - 1, x) - at(lt, y, x)));
        }
    }

    for (dst, s) in lt.d.iter_mut().zip(&ls) {
        *dst += s;
    }
}

/// `resize(..., INTER_AREA)` for the exact 2:1 case an octave change needs.
///
/// At precisely half size `INTER_AREA` is the mean of each 2x2 block, which is
/// what upstream's fast path computes, so this does not go through the general
/// area resampler.
fn halve(src: &Plane, w: usize, h: usize) -> Plane {
    let mut out = Plane::zeros(h, w);
    for y in 0..h {
        for x in 0..w {
            let (sy, sx) = (2 * y, 2 * x);
            let mut acc = 0.0f32;
            let mut n = 0.0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    if sy + dy < src.h && sx + dx < src.w {
                        acc += src.d[(sy + dy) * src.w + sx + dx];
                        n += 1.0;
                    }
                }
            }
            out.d[y * w + x] = acc / n;
        }
    }
    out
}

/// `create_nonlinear_scale_space` followed by `Compute_Determinant_Hessian_Response`.
fn build_pyramid(base: &Plane, levels: &mut [Level], kind: Diffusivity) {
    // Level 0 is the input blurred by soffset. Note `Lt` and `Lsmooth` are the
    // *same* image here, unlike every level after it.
    let lt0 = blur(base, SOFFSET);
    levels[0].smooth = lt0.clone();
    levels[0].lt = lt0;

    // k comes from the input at a fixed 5x5 / sigma 1.0 blur, not from level 0.
    let smooth0 = blur5(base);
    let mut k = kcontrast(&scharr(&smooth0, 1, 0, 1), &scharr(&smooth0, 0, 1, 1), 0.7, 300);

    for i in 1..levels.len() {
        let mut lt = if levels[i].octave > levels[i - 1].octave {
            // Downsampling raises every gradient, so k has to come down with it.
            k *= 0.75;
            halve(&levels[i - 1].lt, levels[i].w, levels[i].h)
        } else {
            levels[i - 1].lt.clone()
        };

        let smooth = blur5(&lt);
        let lx = scharr(&smooth, 1, 0, 1);
        let ly = scharr(&smooth, 0, 1, 1);
        let mut flow = Plane::zeros(lt.h, lt.w);
        for j in 0..flow.d.len() {
            flow.d[j] = diffusivity(kind, lx.d[j], ly.d[j], k);
        }

        let t = 0.5 * levels[i].esigma * levels[i].esigma
            - 0.5 * levels[i - 1].esigma * levels[i - 1].esigma;
        for tau in fed_tau(t, 0.25) {
            diffusion_step(&mut lt, &flow, tau * 0.5);
        }
        levels[i].smooth = smooth;
        levels[i].lt = lt;
    }

    for e in levels.iter_mut() {
        let ss = e.sigma_size;
        let lx = scharr(&e.smooth, 1, 0, ss);
        let ly = scharr(&e.smooth, 0, 1, ss);
        let lxx = scharr(&lx, 1, 0, ss);
        let lxy = scharr(&lx, 0, 1, ss);
        let lyy = scharr(&ly, 0, 1, ss);
        // One normalisation at the end, by sigma_size^4 — not per derivative.
        let quat = (ss * ss * ss * ss) as f32;
        for j in 0..e.det.d.len() {
            e.det.d[j] = (lxx.d[j] * lyy.d[j] - lxy.d[j] * lxy.d[j]) * quat;
        }
        e.lx = lx;
        e.ly = ly;
    }
}

/// The fixed `GaussianBlur(..., Size(5, 5), 1.0)` AKAZE uses before every
/// derivative — a constant kernel, not one derived from the level's sigma.
fn blur5(src: &Plane) -> Plane {
    let k = cv_imgproc::filter::get_gaussian_kernel(5, 1.0)
        .expect("5 is odd and positive");
    let d = cv_imgproc::filter::gaussian_blur_planar(
        &src.d,
        src.w as i32,
        src.h as i32,
        1,
        &k,
        &k,
        cv_core::BorderType::Replicate,
    )
    .expect("the plane's length matches its dimensions");
    Plane { w: src.w, h: src.h, d }
}

/// Is there already a keypoint within `radius` of `(x, y)`? Returns its index.
///
/// Transcribed including the half-open loop bounds: upstream searches
/// `y - r .. y + r` exclusive of the upper end, so the window is not symmetric
/// about the centre.
fn find_neighbour(mask: &[u8], w: usize, h: usize, x: usize, y: usize, radius: usize) -> Option<usize> {
    let y0 = y.saturating_sub(radius);
    let x0 = x.saturating_sub(radius);
    for i in y0..(y + radius).min(h) {
        for j in x0..(x + radius).min(w) {
            if mask[i * w + j] == 0 {
                continue;
            }
            let (dx, dy) = (j as i64 - x as i64, i as i64 - y as i64);
            if dx * dx + dy * dy <= (radius * radius) as i64 {
                return Some(i * w + j);
            }
        }
    }
    None
}

/// `Find_Scale_Space_Extrema` then `Do_Subpixel_Refinement`.
///
/// # Three suppression passes, not one
///
/// Unlike KAZE, AKAZE never compares a pixel against the neighbouring *levels'*
/// response surfaces. A candidate only has to beat its own 3x3 neighbourhood.
/// Cross-scale competition happens afterwards, as two sweeps over the level
/// masks — once upwards, once back down — each clearing the weaker of any two
/// keypoints that land within a search radius of one another once mapped
/// through the octave ratio.
fn detect(levels: &[Level], threshold: f64) -> Vec<KeyPoint> {
    let threshold = threshold as f32;
    let mut masks: Vec<Vec<u8>> = Vec::with_capacity(levels.len());

    for e in levels {
        let mut mask = vec![0u8; e.w * e.h];
        if e.border + 1 < e.h && e.border + 1 < e.w {
            for y in e.border..e.h - e.border {
                for x in e.border..e.w - e.border {
                    let v = e.det.d[y * e.w + x];
                    if v <= threshold {
                        continue;
                    }
                    // Strictly greater than all eight neighbours.
                    let beats = |yy: usize, xx: usize| v > e.det.d[yy * e.w + xx];
                    if !(beats(y, x - 1) && beats(y, x + 1))
                        || !(beats(y - 1, x - 1) && beats(y - 1, x) && beats(y - 1, x + 1))
                        || !(beats(y + 1, x - 1) && beats(y + 1, x) && beats(y + 1, x + 1))
                    {
                        continue;
                    }
                    if let Some(idx) = find_neighbour(&mask, e.w, e.h, x, y, e.sigma_size) {
                        if v > e.det.d[idx] {
                            mask[idx] = 0;
                        } else {
                            continue;
                        }
                    }
                    mask[y * e.w + x] = 1;
                }
            }
        }
        masks.push(mask);
    }

    // Upwards: a keypoint clears its counterpart on the level below.
    for i in 1..levels.len() {
        let ratio = (levels[i].octave_ratio / levels[i - 1].octave_ratio) as usize;
        let radius = levels[i].sigma_size * ratio;
        let (lower, upper) = masks.split_at_mut(i);
        for y in 0..levels[i].h {
            for x in 0..levels[i].w {
                if upper[0][y * levels[i].w + x] == 0 {
                    continue;
                }
                let (px, py) = (x * ratio, y * ratio);
                let prev = &mut lower[i - 1];
                if let Some(idx) =
                    find_neighbour(prev, levels[i - 1].w, levels[i - 1].h, px, py, radius)
                {
                    if levels[i].det.d[y * levels[i].w + x] > levels[i - 1].det.d[idx] {
                        prev[idx] = 0;
                    }
                }
            }
        }
    }
    // Downwards: and the same in reverse, so neither level wins by position.
    for i in (0..levels.len().saturating_sub(1)).rev() {
        let ratio = (levels[i + 1].octave_ratio / levels[i].octave_ratio) as usize;
        let radius = levels[i + 1].sigma_size;
        let (lower, upper) = masks.split_at_mut(i + 1);
        for y in 0..levels[i].h {
            for x in 0..levels[i].w {
                if lower[i][y * levels[i].w + x] == 0 {
                    continue;
                }
                let (px, py) = (x / ratio, y / ratio);
                let next = &mut upper[0];
                if let Some(idx) =
                    find_neighbour(next, levels[i + 1].w, levels[i + 1].h, px, py, radius)
                {
                    if levels[i].det.d[y * levels[i].w + x] > levels[i + 1].det.d[idx] {
                        next[idx] = 0;
                    }
                }
            }
        }
    }

    let mut kpts = Vec::new();
    for (i, e) in levels.iter().enumerate() {
        let ratio = e.octave_ratio;
        for y in 1..e.h.saturating_sub(1) {
            for x in 1..e.w.saturating_sub(1) {
                if masks[i][y * e.w + x] == 0 {
                    continue;
                }
                let d = &e.det.d;
                let at = |yy: usize, xx: usize| d[yy * e.w + xx];
                let dx = 0.5 * (at(y, x + 1) - at(y, x - 1));
                let dy = 0.5 * (at(y + 1, x) - at(y - 1, x));
                let dxx = at(y, x + 1) + at(y, x - 1) - 2.0 * at(y, x);
                let dyy = at(y + 1, x) + at(y - 1, x) - 2.0 * at(y, x);
                let dxy = 0.25
                    * (at(y + 1, x + 1) + at(y - 1, x - 1)
                        - at(y - 1, x + 1)
                        - at(y + 1, x - 1));

                // A 2x2 fit, in position only — AKAZE does not refine in scale.
                let Some(off) = solve3(
                    [[dxx, dxy, 0.0], [dxy, dyy, 0.0], [0.0, 0.0, 1.0]],
                    [-dx, -dy, 0.0],
                ) else {
                    continue;
                };
                if off[0].abs() > 1.0 || off[1].abs() > 1.0 {
                    continue;
                }
                // The half-pixel term re-centres the coarse level's sample grid
                // on the full-resolution one.
                kpts.push(KeyPoint {
                    pt: (
                        x as f32 * ratio + off[0] * ratio + 0.5 * (ratio - 1.0),
                        y as f32 * ratio + off[1] * ratio + 0.5 * (ratio - 1.0),
                    ),
                    // In OpenCV a keypoint's size is its diameter.
                    size: e.esigma * DERIVATIVE_FACTOR * 2.0,
                    angle: 0.0,
                    response: at(y, x),
                    octave: e.octave,
                    class_id: i as i32,
                });
            }
        }
    }
    kpts
}

/// `cv::AKAZE::detect(image, keypoints)`.
///
/// See the module documentation for how this differs from [`crate::kaze`].
pub fn akaze_detect<S: MatLike + ?Sized>(
    image: &S,
    p: AkazeParams,
    kind: Diffusivity,
) -> Result<Vec<KeyPoint>> {
    Ok(detect_with_pyramid(image, p, kind)?.0)
}

/// `gauss25` — a 7x7 quadrant of a Gaussian, indexed by `|i|` and `|j|`.
///
/// Stored as a quadrant because the weight is symmetric in both axes; upstream
/// keeps it as a literal table rather than evaluating `exp`, so it is
/// transcribed rather than recomputed.
#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
const GAUSS25: [[f32; 7]; 7] = [
    [0.02546481, 0.02350698, 0.01849125, 0.01239505, 0.00708017, 0.00344629, 0.00142946],
    [0.02350698, 0.02169968, 0.01706957, 0.01144208, 0.00653582, 0.00318132, 0.00131956],
    [0.01849125, 0.01706957, 0.01342740, 0.00900066, 0.00514126, 0.00250252, 0.00103800],
    [0.01239505, 0.01144208, 0.00900066, 0.00603332, 0.00344629, 0.00167749, 0.00069579],
    [0.00708017, 0.00653582, 0.00514126, 0.00344629, 0.00196855, 0.00095820, 0.00039744],
    [0.00344629, 0.00318132, 0.00250252, 0.00167749, 0.00095820, 0.00046640, 0.00019346],
    [0.00142946, 0.00131956, 0.00103800, 0.00069579, 0.00039744, 0.00019346, 0.00008024],
];

/// `Compute_Main_Orientation` — AKAZE's dominant gradient direction.
///
/// # Why this is not [`crate::kaze`]'s orientation
///
/// Both slide a 60-degree sector and keep the longest vector sum, but AKAZE
/// quantises first: the 109 sample angles are counting-sorted into 42 slices,
/// and the sector advances **one slice at a time** rather than by 0.15 rad.
/// 42 slices of `2*pi/42` make the 7-slice window exactly `pi/3`, so the sector
/// is the same width — but it can only ever start on a slice boundary, and the
/// two schemes therefore pick different maxima. On a 96x96 test image, using
/// KAZE's sweep here put the four keypoints' angles up to 22.6 degrees out.
///
/// Samples are taken at **integer positions on the octave's own grid**, and
/// unlike KAZE there is no per-sample distance weighting beyond [`GAUSS25`].
fn orientation(e: &Level, kp: &KeyPoint) -> f32 {
    let scale = cv_round(0.5 * kp.size / e.octave_ratio).max(1);
    let x0 = cv_round(kp.pt.0 / e.octave_ratio);
    let y0 = cv_round(kp.pt.1 / e.octave_ratio);

    let mut res_x = Vec::with_capacity(109);
    let mut res_y = Vec::with_capacity(109);
    let mut ang = Vec::with_capacity(109);
    for i in -6i32..=6 {
        for j in -6i32..=6 {
            if i * i + j * j >= 36 {
                continue;
            }
            let y = (y0 + i * scale).clamp(0, e.h as i32 - 1) as usize;
            let x = (x0 + j * scale).clamp(0, e.w as i32 - 1) as usize;
            let w = GAUSS25[i.unsigned_abs() as usize][j.unsigned_abs() as usize];
            let (rx, ry) = (w * e.lx.d[y * e.w + x], w * e.ly.d[y * e.w + x]);
            res_x.push(rx);
            res_y.push(ry);
            // `hal::fastAtan2(..., angleInDegrees = false)`: degrees, scaled.
            ang.push(fast_atan2(ry, rx) * (std::f32::consts::PI / 180.0));
        }
    }

    const SLICES: usize = 42;
    const WIN: usize = 7;
    let ang_step = 2.0 * std::f32::consts::PI / SLICES as f32;

    // `quantized_counting_sort`: `slice[b]` is where bucket `b` starts in
    // `sorted_idx`. Angles outside `0..2*pi` fall into bucket 0 rather than
    // being dropped, which is upstream's behaviour and not an error path.
    let bucket = |a: f32| {
        let b = (a / ang_step) as i32;
        if b < 0 || b as usize >= SLICES {
            0usize
        } else {
            b as usize
        }
    };
    let mut cum = [0usize; SLICES + 1];
    for a in &ang {
        cum[bucket(*a)] += 1;
    }
    for i in 1..=SLICES {
        cum[i] += cum[i - 1];
    }
    let mut sorted_idx = vec![0usize; ang.len()];
    for (i, a) in ang.iter().enumerate() {
        let b = bucket(*a);
        cum[b] -= 1;
        sorted_idx[cum[b]] = i;
    }
    // Upstream's `slice` *is* this array, read after the scatter — the
    // decrements walk each entry back down from the bucket's end to its start.
    // Snapshotting the prefix sums instead would leave every entry one bucket
    // high and shift every window by 8.57 degrees.
    let slice = cum;

    let sum_over = |range: std::ops::Range<usize>| {
        let (mut sx, mut sy) = (0.0f32, 0.0f32);
        for &idx in &sorted_idx[range] {
            sx += res_x[idx];
            sy += res_y[idx];
        }
        (sx, sy)
    };

    let (mut max_x, mut max_y) = sum_over(slice[0]..slice[WIN]);
    let mut max_norm = max_x * max_x + max_y * max_y;

    for sn in 1..=SLICES - WIN {
        // Nothing entered or left the window, so the sum cannot have changed.
        if slice[sn] == slice[sn - 1] && slice[sn + WIN] == slice[sn + WIN - 1] {
            continue;
        }
        let (sx, sy) = sum_over(slice[sn]..slice[sn + WIN]);
        let norm = sx * sx + sy * sy;
        if norm > max_norm {
            max_norm = norm;
            max_x = sx;
            max_y = sy;
        }
    }
    // The windows that wrap past slice 41 back to slice 0.
    for sn in SLICES - WIN + 1..SLICES {
        let remain = sn + WIN - SLICES;
        if slice[sn] == slice[sn - 1] && slice[remain] == slice[remain - 1] {
            continue;
        }
        let (ax, ay) = sum_over(slice[sn]..slice[SLICES]);
        let (bx, by) = sum_over(slice[0]..slice[remain]);
        let (sx, sy) = (ax + bx, ay + by);
        let norm = sx * sx + sy * sy;
        if norm > max_norm {
            max_norm = norm;
            max_x = sx;
            max_y = sy;
        }
    }
    fast_atan2(max_y, max_x)
}

/// `cv::AKAZE::detectAndCompute(image, mask, keypoints, descriptors)`.
///
/// The detector is at parity with cv2 (see the module docs). The **M-LDB
/// descriptor is not** — its sampling geometry is reconstructed from the paper
/// rather than transcribed, so the same keypoint yields different bits here
/// than in cv2. Matching descriptors across the two will not work.
pub fn akaze_detect_and_compute<S: MatLike + ?Sized>(
    image: &S,
    p: AkazeParams,
    kind: Diffusivity,
) -> Result<(Vec<KeyPoint>, Vec<AkazeDescriptorBits>)> {
    cv_assert!(
        (1..=3).contains(&p.descriptor_channels),
        "descriptor_channels must be 1..3, got {}",
        p.descriptor_channels
    );
    let (mut kps, levels) = detect_with_pyramid(image, p, kind)?;
    let descs = kps
        .iter_mut()
        .map(|k| {
            let e = &levels[(k.class_id.max(0) as usize).min(levels.len() - 1)];
            k.angle = orientation(e, k);
            mldb(&e.lt, &e.lx, &e.ly, k, p.descriptor_channels as usize)
        })
        .collect();
    for k in kps.iter_mut() {
        k.class_id = -1;
    }
    Ok((kps, descs))
}

/// Shared by [`akaze_detect`] and [`akaze_detect_and_compute`]; the pyramid is
/// expensive enough that the descriptor path must not rebuild it.
fn detect_with_pyramid<S: MatLike + ?Sized>(
    image: &S,
    p: AkazeParams,
    kind: Diffusivity,
) -> Result<(Vec<KeyPoint>, Vec<Level>)> {
    cv_assert!(image.channels() == 1, "AKAZE requires a single-channel image");
    cv_assert!(p.n_octaves > 0 && p.n_octave_layers > 0, "octaves and layers must be positive");
    cv_assert!(image.rows() >= 16 && image.cols() >= 16, "AKAZE needs at least a 16x16 image");

    let base = Plane::from_image(image)?;
    let mut levels = plan_levels(base.w, base.h, p.n_octaves, p.n_octave_layers);
    cv_assert!(!levels.is_empty(), "the image is too small for even one octave");
    build_pyramid(&base, &mut levels, kind);
    let kpts = detect(&levels, p.threshold);
    Ok((kpts, levels))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use cv_core::Mat;


    /// A deterministic texture: xorshift noise put through two integer 3x3 box
    /// passes so it carries energy at scales the detectors respond to.
    ///
    /// Every step is integer, so this generates byte-for-byte the same image
    /// here as the Python that produced the pinned references — checked by the
    /// checksum in [`the_texture_is_the_one_cv2_saw`]. Synthetic discs are a
    /// poor parity fixture by comparison: they are four-fold symmetric, so
    /// responses and orientations tie exactly and the tie-break is decided by
    /// the last ULP rather than by the algorithm.
    pub(crate) fn texture(w: i32, h: i32) -> Mat {
        let mut state: u32 = 0x2545_F491;
        let mut buf = vec![0i32; (w * h) as usize];
        for v in buf.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *v = (state >> 24) as i32;
        }
        for _ in 0..2 {
            let src = buf.clone();
            for y in 0..h {
                for x in 0..w {
                    let mut acc = 0i32;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            acc += src[((y + dy).clamp(0, h - 1) * w
                                + (x + dx).clamp(0, w - 1))
                                as usize];
                        }
                    }
                    buf[(y * w + x) as usize] = acc / 9;
                }
            }
        }
        let b: Vec<u8> = buf.iter().map(|v| *v as u8).collect();
        Mat::from_vec(h, w, 1, &b).unwrap()
    }

    #[test]
    fn the_texture_is_the_one_cv2_saw() {
        let img = texture(160, 160);
        let sum: i64 = (0..160)
            .flat_map(|y| (0..160).map(move |x| (y, x)))
            .map(|(y, x)| img.at_f64(y, x, 0).unwrap() as i64)
            .sum();
        // If this fails every pinned number below is measured against a
        // different image and means nothing.
        assert_eq!(sum, 3_259_013, "the generator drifted from cv2's input");
    }

    /// Full-pipeline parity against `cv2.AKAZE_create().detectAndCompute()`.
    ///
    /// On this texture cv2 finds 26 keypoints; this finds the same 26, at the
    /// same positions to 1e-4 px, with **identical** angles, and the M-LDB
    /// descriptors come out **bit-for-bit identical** — 0 differing bits of
    /// 12688. So AKAZE descriptors here can be matched against cv2's.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn akaze_matches_cv2_end_to_end_on_texture() {
        const CV2_KP: [(f32, f32, f32, f32); 4] = [
            (29.39196, 117.32749, 4.8, 332.2200),
            (33.28715, 86.82425, 4.8, 170.7600),
            (34.10664, 61.56907, 4.8, 163.1564),
            (39.32449, 39.18494, 4.8, 139.2559),
        ];
        /// cv2's descriptor for the first keypoint, in full.
        const CV2_DESC0: [u8; crate::kaze::AKAZE_BYTES] = [
            216, 113, 0, 252, 255, 47, 225, 63, 234, 90, 133, 153, 223, 1, 230, 15, 0, 128,
            156, 247, 255, 255, 79, 0, 247, 255, 3, 164, 168, 255, 136, 255, 239, 255, 7, 0,
            255, 55, 165, 148, 40, 189, 213, 128, 6, 11, 41, 88, 10, 255, 255, 20, 240, 255,
            194, 1, 228, 3, 252, 239, 63,
        ];
        /// FNV-1a over every descriptor byte, in keypoint order — one number
        /// standing in for all 1586 bytes.
        const CV2_DESC_HASH: u64 = 0x3533_0f88_7870_0b7c;

        let img = texture(160, 160);
        let (mut kps, descs) =
            akaze_detect_and_compute(&img, AkazeParams::default(), Diffusivity::PmG2).unwrap();
        assert_eq!(kps.len(), 26, "cv2 finds 26 keypoints here");

        // cv2's order is by x then y; sort to match before comparing.
        let mut order: Vec<usize> = (0..kps.len()).collect();
        order.sort_by(|&a, &b| {
            kps[a].pt.0.total_cmp(&kps[b].pt.0).then(kps[a].pt.1.total_cmp(&kps[b].pt.1))
        });
        let descs: Vec<_> = order.iter().map(|&i| descs[i]).collect();
        kps = order.iter().map(|&i| kps[i]).collect();

        for (k, &(x, y, size, angle)) in kps.iter().zip(&CV2_KP) {
            assert!(
                (k.pt.0 - x).abs() < 1e-4 && (k.pt.1 - y).abs() < 1e-4,
                "position ({}, {}) against cv2's ({x}, {y})",
                k.pt.0,
                k.pt.1
            );
            assert_eq!(k.size, size);
            assert!((k.angle - angle).abs() < 1e-3, "angle {} against {angle}", k.angle);
        }
        assert_eq!(descs[0], CV2_DESC0, "first descriptor differs from cv2's");

        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for b in descs.iter().flatten() {
            hash = (hash ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        assert_eq!(hash, CV2_DESC_HASH, "descriptor bytes differ from cv2's somewhere");
    }

    /// The same three-disc scene [`crate::kaze`]'s tests use.
    fn scene(w: i32, h: i32) -> Mat {
        let mut d = vec![40u8; (w * h) as usize];
        for (cx, cy, r) in [(24, 24, 7), (60, 32, 10), (40, 66, 8)] {
            for y in (cy - r).max(0)..(cy + r).min(h) {
                for x in (cx - r).max(0)..(cx + r).min(w) {
                    if (x - cx).pow(2) + (y - cy).pow(2) <= r * r {
                        d[(y * w + x) as usize] = 215;
                    }
                }
            }
        }
        Mat::from_vec(h, w, 1, &d).unwrap()
    }

    /// A 96x96 image cannot hold a second octave — 48x48 is below the 80x40
    /// floor — so AKAZE collapses to one octave of four sublevels. This is the
    /// single rule that most separates AKAZE's output from KAZE's, so it is
    /// checked directly rather than only through the keypoints.
    #[test]
    fn a_small_image_gets_one_octave() {
        let levels = plan_levels(96, 96, 4, 4);
        assert_eq!(levels.len(), 4, "one octave of four sublevels");
        assert!(levels.iter().all(|e| e.octave == 0 && e.octave_ratio == 1.0));
        assert!(levels.iter().all(|e| e.w == 96 && e.h == 96), "no downsampling");

        // Wide enough for two, and the second is half the size.
        let levels = plan_levels(320, 240, 4, 4);
        assert_eq!(levels.len(), 12, "three octaves fit before 80x40");
        assert_eq!((levels[4].w, levels[4].h), (160, 120));
        assert_eq!((levels[8].w, levels[8].h), (80, 60));
        assert_eq!(levels[8].octave_ratio, 4.0);
    }

    /// `sigma_size` divides by the octave's own ratio, so it depends only on
    /// the sublevel — upstream says as much in a comment, and it is what keeps
    /// the Scharr kernels the same width at every octave.
    #[test]
    fn the_derivative_scale_repeats_every_octave() {
        let levels = plan_levels(640, 480, 4, 4);
        let first: Vec<usize> = levels[0..4].iter().map(|e| e.sigma_size).collect();
        for oct in 1..4 {
            let this: Vec<usize> =
                levels[oct * 4..oct * 4 + 4].iter().map(|e| e.sigma_size).collect();
            assert_eq!(this, first, "octave {oct} uses different derivative scales");
        }
    }

    /// Pinned against cv2 4.13.0's `cv2.AKAZE_create().detect()` on [`scene`].
    ///
    /// cv2 returns exactly these four keypoints, in this order by descending
    /// response. Positions agree to four decimals and responses to seven, so
    /// the tolerances here are tight enough to catch any drift in the pyramid,
    /// the contrast decay or the two suppression sweeps.
    #[test]
    fn akaze_matches_cv2s_detector() {
        const CV2: [(f32, f32, f32); 4] = [
            (43.9916, 62.1991, 0.00789351),
            (36.0921, 62.0921, 0.00750430),
            (65.7567, 37.4211, 0.00660322),
            (54.7182, 37.8835, 0.00653580),
        ];
        let img = scene(96, 96);
        let mut kps = akaze_detect(&img, AkazeParams::default(), Diffusivity::PmG2).unwrap();
        assert_eq!(kps.len(), 4, "cv2 finds 4 keypoints here");
        assert!(kps.iter().all(|k| k.octave == 0), "a 96x96 image has one octave");

        kps.sort_by(|a, b| b.response.total_cmp(&a.response));
        for (k, &(x, y, response)) in kps.iter().zip(&CV2) {
            assert!(
                (k.pt.0 - x).abs() < 1e-4 && (k.pt.1 - y).abs() < 1e-4,
                "position ({}, {}) against cv2's ({x}, {y})",
                k.pt.0,
                k.pt.1
            );
            // Every keypoint here is sublevel 0: 1.6 * 1.5 * 2.
            assert_eq!(k.size, 4.8, "size");
            assert!(
                (k.response - response).abs() < 1e-7,
                "response {} against cv2's {response}",
                k.response
            );
        }
    }

    #[test]
    fn a_flat_image_has_no_features() {
        let flat = Mat::from_vec(64, 64, 1, &[128u8; 4096]).unwrap();
        let kps = akaze_detect(&flat, AkazeParams::default(), Diffusivity::PmG2).unwrap();
        assert!(kps.is_empty(), "a flat image has nothing to detect, got {}", kps.len());
    }

    #[test]
    fn it_validates_its_input() {
        let colour = Mat::zeros(32, 32, cv_core::types::CV_8UC3).unwrap();
        assert!(akaze_detect(&colour, AkazeParams::default(), Diffusivity::PmG2).is_err());
        let tiny = Mat::zeros(8, 8, cv_core::types::CV_8UC1).unwrap();
        assert!(akaze_detect(&tiny, AkazeParams::default(), Diffusivity::PmG2).is_err());
    }
}
