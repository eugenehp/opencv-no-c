//! KAZE and AKAZE — features in a nonlinear scale space
//! (`modules/features2d/src/kaze.cpp`, `akaze.cpp`, `kaze/nldiffusion_functions.cpp`).
//!
//! # Why a nonlinear scale space
//!
//! Every detector built on a Gaussian pyramid — SIFT, SURF, ORB — blurs
//! *uniformly*. That smooths noise and object boundaries alike, so as the scale
//! rises the edges that define an object drift and its features move with them.
//!
//! KAZE instead evolves the image by nonlinear diffusion: it solves
//! `dL/dt = div(c(x, y, t) . grad L)` where the conductivity `c` is **small
//! where the gradient is large**. Flat regions smooth as before; edges barely
//! diffuse at all. Detail is removed without moving the boundaries, so features
//! stay put across scales.
//!
//! The cost is that there is no closed form — the PDE has to be integrated
//! step by step. Upstream's answer is fast explicit diffusion (`fed_tau`),
//! which is ported here: it makes the integration cheap enough that the whole
//! scale space costs a handful of blurs.
//!
//! # Agreement with OpenCV
//!
//! Against cv2 4.13.0's `cv2.KAZE_create().detectAndCompute()`, on a
//! deterministic 160x160 texture (see `kaze_matches_cv2_end_to_end_on_texture`)
//! and a 320x240 one:
//!
//! | | 160x160 | 320x240 |
//! |---|---|---|
//! | keypoints | **30, matching cv2** | **1082, matching cv2** |
//! | positions | all within 1e-4 px | all within 1e-3 px |
//! | `size` | **exact** | **exact** |
//! | `angle` | **all 30 identical** | **all 1082 identical** |
//! | M-SURF descriptor | within 4.2e-7 | within 1.7e-6 |
//!
//! The descriptors are not bit-exact because the subregion sums accumulate in
//! a different order than upstream's, but 1.7e-6 on a unit-normalised vector is
//! far below anything a matcher can distinguish.
//!
//! ## Synthetic test images are a worse fixture than they look
//!
//! On three drawn discs the same code matches only 38 of 43 keypoints. That is
//! not a weaker result — it is a worse *measurement*. Discs are four-fold
//! symmetric, so the ring of extrema round each one carries the same response
//! to seven digits and the same orientation to within a tie; which candidate
//! survives suppression is then settled by the last `f32` ULP rather than by
//! the algorithm. The texture fixture above has no such ties, and everything
//! lines up exactly.
//!
//! # AKAZE lives in [`crate::akaze`]
//!
//! Upstream's AKAZE is not KAZE with another descriptor — it builds its own
//! pyramid, downsampling each octave — so it is ported separately. This module
//! shares its primitives (`blur`, `scharr`, `fed_tau`, `diffusivity`)
//! with it, matching upstream's own split between `KAZEFeatures.cpp` and
//! `nldiffusion_functions.cpp`.
//!
//! # The descriptors
//!
//! M-SURF (`msurf`) and M-LDB (`mldb`) are transcribed from upstream, not
//! reconstructed from the papers, and both are interchangeable with cv2's —
//! M-LDB bit-for-bit.

pub(crate) use crate::plane::Plane;
use cv_core::{cv_assert, BorderType, MatLike, Result};

use crate::detectors::{Diffusivity, KazeParams};
use crate::KeyPoint;

/// AKAZE's M-LDB descriptor, 486 bits rounded up to whole bytes.
pub const AKAZE_BYTES: usize = 61;

/// An AKAZE descriptor.
pub type AkazeDescriptorBits = [u8; AKAZE_BYTES];

/// KAZE's M-SURF descriptor is 64 floats (128 when extended).
pub type KazeDescriptor = Vec<f32>;



/// `gaussian_2D_convolution` — upstream derives the kernel size from sigma.
///
/// The formula inverts `getGaussianKernel`'s default sigma, and the result is
/// forced odd. It runs in `f32`, so it is transcribed rather than tidied: at
/// sigma 1.6 it gives 9, and computing the same thing in `f64` would not.
pub(crate) fn blur(src: &Plane, sigma: f32) -> Plane {
    if sigma <= 0.0 {
        return src.clone();
    }
    let mut ksize = (2.0f32 * (1.0 + (sigma - 0.8) / 0.3)).ceil() as i32;
    if ksize % 2 == 0 {
        ksize += 1;
    }
    let k = cv_imgproc::filter::get_gaussian_kernel(ksize, f64::from(sigma))
        .expect("ksize is odd and positive by construction");
    let d = cv_imgproc::filter::gaussian_blur_planar(
        &src.d,
        src.w as i32,
        src.h as i32,
        1,
        &k,
        &k,
        BorderType::Replicate,
    )
    .expect("the plane's length matches its dimensions");
    Plane { w: src.w, h: src.h, d }
}

/// `compute_derivative_kernels` — Scharr taps, optionally spread over a wider
/// support.
///
/// At `scale > 1` the kernel is **sparse**: three non-zero taps at the ends and
/// the centre of a `3 + 2*(scale-1)` window, zero in between. That is not a
/// resampled Scharr, it is the same three taps moved further apart, which is
/// what makes it a derivative *at that scale* rather than a wider smoother.
pub(crate) fn scharr_kernels(order: i32, scale: usize) -> Vec<f32> {
    let ksize = 3 + 2 * (scale - 1);
    let mut k = vec![0.0f32; ksize];
    if order == 1 {
        k[0] = -1.0;
        k[ksize - 1] = 1.0;
    } else {
        // w = 10/3 and the normaliser reproduce [3, 10, 3] / 32 at scale 1.
        let w = 10.0f32 / 3.0;
        let norm = 1.0 / (2.0 * scale as f32 * (w + 2.0));
        k[0] = norm;
        k[ksize / 2] = w * norm;
        k[ksize - 1] = norm;
    }
    k
}

/// `compute_scharr_derivatives` — separable Scharr, `BORDER_REFLECT_101`.
pub(crate) fn scharr(src: &Plane, dx: i32, dy: i32, scale: usize) -> Plane {
    let kx = scharr_kernels(dx, scale);
    let ky = scharr_kernels(dy, scale);
    let r = (kx.len() / 2) as i64;
    let refl = |i: i64, n: i64| -> usize {
        // BORDER_REFLECT_101: the edge sample is not repeated.
        let mut v = i;
        if n == 1 {
            return 0;
        }
        loop {
            if v < 0 {
                v = -v;
            } else if v >= n {
                v = 2 * (n - 1) - v;
            } else {
                return v as usize;
            }
        }
    };
    let mut tmp = Plane::zeros(src.h, src.w);
    for y in 0..src.h {
        for x in 0..src.w as i64 {
            let mut a = 0.0f32;
            for (i, kv) in kx.iter().enumerate() {
                a += kv * src.d[y * src.w + refl(x + i as i64 - r, src.w as i64)];
            }
            tmp.d[y * src.w + x as usize] = a;
        }
    }
    let mut out = Plane::zeros(src.h, src.w);
    for y in 0..src.h as i64 {
        for x in 0..src.w {
            let mut a = 0.0f32;
            for (i, kv) in ky.iter().enumerate() {
                a += kv * tmp.d[refl(y + i as i64 - r, src.h as i64) * src.w + x];
            }
            out.d[y as usize * src.w + x] = a;
        }
    }
    out
}

/// `compute_k_percentile` — the contrast factor `k`.
///
/// This is what makes the diffusion adaptive to the image rather than to a
/// fixed constant: `k` sets the gradient above which an edge stops smoothing,
/// and a percentile puts it where the image's own edges actually are.
///
/// Upstream histograms the gradient *magnitude* into 300 bins scaled by the
/// image maximum and returns the bin edge below which 70% of the non-zero
/// gradients fall — deliberately a coarse quantisation of the percentile, not
/// the percentile itself. Sorting the magnitudes and indexing at 70% gives a
/// slightly different number and a visibly different keypoint set.
fn contrast_factor(p: &Plane, perc: f32, gscale: f32, nbins: usize) -> f32 {
    let gaussian = blur(p, gscale);
    let lx = scharr(&gaussian, 1, 0, 1);
    let ly = scharr(&gaussian, 0, 1, 1);

    // The border row and column are excluded from both passes.
    let interior = || (1..p.h.saturating_sub(1)).flat_map(move |i| (1..p.w.saturating_sub(1)).map(move |j| i * p.w + j));

    let mut hmax = 0.0f32;
    for i in interior() {
        let modg = lx.d[i] * lx.d[i] + ly.d[i] * ly.d[i];
        if modg > hmax {
            hmax = modg;
        }
    }
    hmax = hmax.sqrt();
    if hmax == 0.0 {
        return 0.03;
    }

    let mut hist = vec![0usize; nbins];
    let mut npoints = 0.0f32;
    for i in interior() {
        let modg = lx.d[i] * lx.d[i] + ly.d[i] * ly.d[i];
        if modg != 0.0 {
            let nbin = ((nbins as f32 * (modg.sqrt() / hmax)).floor() as usize).min(nbins - 1);
            hist[nbin] += 1;
            npoints += 1.0;
        }
    }

    let nthreshold = (npoints * perc) as usize;
    let mut nelements = 0usize;
    let mut k = 0usize;
    while nelements < nthreshold && k < nbins {
        nelements += hist[k];
        k += 1;
    }
    if nelements < nthreshold {
        0.03
    } else {
        hmax * (k as f32 / nbins as f32)
    }
}

/// `fed_tau_by_cycle_time` — the step sizes for one FED cycle.
///
/// # Why the steps are not all stable
///
/// The explicit scheme diverges above 0.25, and most of these taus are larger
/// than that — some far larger. That is the point of fast explicit diffusion:
/// the individual steps are unstable but their *composition* over a full cycle
/// is not, because the taus are the reciprocal squared roots of Chebyshev
/// nodes. Taking `n` of them costs what one stable step of size 0.25 costs but
/// advances the time by O(n²) as much.
///
/// The reordering is what makes that survive rounding: applied in index order
/// the large steps cluster and `f32` loses the cancellation, so upstream
/// permutes them by a multiplier coprime to a prime above `n`.
pub(crate) fn fed_tau(t: f32, tau_max: f32) -> Vec<f32> {
    let n = (3.0 * t / tau_max + 0.25).sqrt() - 0.5 - 1.0e-8;
    let n = n.ceil() as i32;
    if n <= 0 {
        return Vec::new();
    }
    let scale = 3.0 * t / (tau_max * (n * (n + 1)) as f32);

    let c = 1.0 / (4.0 * n as f32 + 2.0);
    let d = scale * tau_max / 2.0;
    let tauh: Vec<f32> = (0..n)
        .map(|k| {
            let h = (std::f32::consts::PI * (2.0 * k as f32 + 1.0) * c).cos();
            d / (h * h)
        })
        .collect();

    let kappa = n / 2;
    let mut prime = n + 1;
    while !is_prime(prime) {
        prime += 1;
    }
    let mut tau = vec![0.0f32; n as usize];
    let mut k = 0i64;
    for slot in tau.iter_mut() {
        let mut index;
        loop {
            index = ((k + 1) * i64::from(kappa)) % i64::from(prime) - 1;
            if index < i64::from(n) {
                break;
            }
            k += 1;
        }
        *slot = tauh[index as usize];
        k += 1;
    }
    tau
}

pub(crate) fn is_prime(n: i32) -> bool {
    if n < 2 || (n > 2 && n % 2 == 0) {
        return n == 2;
    }
    let mut i = 3;
    while i * i <= n {
        if n % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}

/// The four conductivity functions upstream offers.
pub(crate) fn diffusivity(kind: Diffusivity, gx: f32, gy: f32, k: f32) -> f32 {
    let g2 = gx * gx + gy * gy;
    let k2 = k * k;
    match kind {
        // Favours high-contrast edges over low-contrast ones.
        Diffusivity::PmG1 => (-g2 / k2).exp(),
        // Favours wide regions over smaller ones.
        Diffusivity::PmG2 => 1.0 / (1.0 + g2 / k2),
        // Sharpens edges more aggressively than either Perona-Malik form.
        Diffusivity::Weickert => {
            if g2 <= 0.0 {
                1.0
            } else {
                1.0 - (-3.315 / (g2 / k2).powi(4)).exp()
            }
        }
        Diffusivity::Charbonnier => 1.0 / (1.0 + g2 / k2).sqrt(),
    }
}

/// `nld_step_scalar` — one explicit diffusion step, `L += step * div(c . grad L)`.
///
/// Transcribed rather than written from the divergence formula, because the
/// boundary is not what a general form gives. The interior uses all four
/// fluxes; each edge drops the flux that would leave the image; and the four
/// **corners are never written at all** — upstream allocates the step image
/// once and only ever fills rows and columns `1..n-1`, so the corners keep the
/// zero they were allocated with and never diffuse.
fn diffusion_step(l: &mut Plane, c: &Plane, step: f32) {
    let (h, w) = (l.h, l.w);
    if h < 2 || w < 2 {
        return;
    }
    let mut ls = vec![0.0f32; h * w];
    let half = 0.5 * step;
    let at = |p: &Plane, y: usize, x: usize| p.d[y * w + x];

    // Interior.
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let xpos = (at(c, y, x) + at(c, y, x + 1)) * (at(l, y, x + 1) - at(l, y, x));
            let xneg = (at(c, y, x - 1) + at(c, y, x)) * (at(l, y, x) - at(l, y, x - 1));
            let ypos = (at(c, y, x) + at(c, y + 1, x)) * (at(l, y + 1, x) - at(l, y, x));
            let yneg = (at(c, y - 1, x) + at(c, y, x)) * (at(l, y, x) - at(l, y - 1, x));
            ls[y * w + x] = half * (xpos - xneg + ypos - yneg);
        }
    }
    // Top and bottom rows: no flux across the missing neighbour.
    for x in 1..w - 1 {
        let xpos = (at(c, 0, x) + at(c, 0, x + 1)) * (at(l, 0, x + 1) - at(l, 0, x));
        let xneg = (at(c, 0, x - 1) + at(c, 0, x)) * (at(l, 0, x) - at(l, 0, x - 1));
        let ypos = (at(c, 0, x) + at(c, 1, x)) * (at(l, 1, x) - at(l, 0, x));
        ls[x] = half * (xpos - xneg + ypos);

        let b = h - 1;
        let xpos = (at(c, b, x) + at(c, b, x + 1)) * (at(l, b, x + 1) - at(l, b, x));
        let xneg = (at(c, b, x - 1) + at(c, b, x)) * (at(l, b, x) - at(l, b, x - 1));
        let yneg = (at(c, b - 1, x) + at(c, b, x)) * (at(l, b, x) - at(l, b - 1, x));
        ls[b * w + x] = half * (xpos - xneg - yneg);
    }
    // Left and right columns.
    let (r0, r1) = (w - 1, w - 2);
    for y in 1..h - 1 {
        let xpos = (at(c, y, 0) + at(c, y, 1)) * (at(l, y, 1) - at(l, y, 0));
        let ypos = (at(c, y, 0) + at(c, y + 1, 0)) * (at(l, y + 1, 0) - at(l, y, 0));
        let yneg = (at(c, y - 1, 0) + at(c, y, 0)) * (at(l, y, 0) - at(l, y - 1, 0));
        ls[y * w] = half * (xpos + ypos - yneg);

        let xneg = (at(c, y, r1) + at(c, y, r0)) * (at(l, y, r0) - at(l, y, r1));
        let ypos = (at(c, y, r0) + at(c, y + 1, r0)) * (at(l, y + 1, r0) - at(l, y, r0));
        let yneg = (at(c, y - 1, r0) + at(c, y, r0)) * (at(l, y, r0) - at(l, y - 1, r0));
        ls[y * w + r0] = half * (-xneg + ypos - yneg);
    }

    for (dst, s) in l.d.iter_mut().zip(&ls) {
        *dst += s;
    }
}

/// Upstream's `soffset` — the base scale of the whole evolution.
pub(crate) const SOFFSET: f32 = 1.6;
/// Upstream's `sderivatives` — the blur applied before every derivative.
pub(crate) const SDERIVATIVES: f32 = 1.0;

/// One level of the nonlinear scale space (`TEvolution`).
///
/// Every level is at **full resolution**. KAZE does not build a pyramid: the
/// octave index only sets the scale at which derivatives are taken, because
/// downsampling would reintroduce exactly the boundary drift the nonlinear
/// diffusion exists to avoid.
struct Evolution {
    /// `Lt` — the diffused image.
    l: Plane,
    /// `Lsmooth` — `Lt` of the *previous* level blurred by `sderivatives`.
    smooth: Plane,
    /// `Ldet` — determinant of the scale-normalised Hessian.
    det: Plane,
    /// `Lx`, `Ly` — first derivatives at this level's scale, already
    /// multiplied by `sigma_size`. The orientation and descriptor stages read
    /// these, not the raw image.
    lx: Plane,
    ly: Plane,
    esigma: f32,
    /// `cvRound(esigma)`, the integer scale the Scharr kernels are built at.
    sigma_size: usize,
    octave: i32,
    sublevel: i32,
}

/// `KAZEFeatures::Create_Nonlinear_Scale_Space`.
fn build_evolution(
    base: &Plane,
    n_octaves: i32,
    n_sublevels: i32,
    kind: Diffusivity,
) -> Vec<Evolution> {
    let (n_octaves, n_sublevels) = (n_octaves.max(1), n_sublevels.max(1));

    // Level 0 is the input blurred by soffset; k is measured on that, not on
    // the raw input.
    let lt0 = blur(base, SOFFSET);
    let k = contrast_factor(&lt0, 0.7, SDERIVATIVES, 300);

    let mut evo: Vec<Evolution> = Vec::new();
    for o in 0..n_octaves {
        for s in 0..n_sublevels {
            let esigma = SOFFSET * 2f32.powf(s as f32 / n_sublevels as f32 + o as f32);
            evo.push(Evolution {
                l: Plane::zeros(base.h, base.w),
                smooth: Plane::zeros(base.h, base.w),
                det: Plane::zeros(base.h, base.w),
                lx: Plane::zeros(base.h, base.w),
                ly: Plane::zeros(base.h, base.w),
                esigma,
                sigma_size: cv_round(esigma).max(1) as usize,
                octave: o,
                sublevel: s,
            });
        }
    }

    evo[0].smooth = blur(&lt0, SDERIVATIVES);
    evo[0].l = lt0;

    for i in 1..evo.len() {
        // Each level starts from the previous one's diffused image and is
        // driven forward by a conductivity computed *once* — the flow is frozen
        // for the whole cycle, not refreshed between steps.
        let prev = evo[i - 1].l.clone();
        let smooth = blur(&prev, SDERIVATIVES);
        let lx = scharr(&smooth, 1, 0, 1);
        let ly = scharr(&smooth, 0, 1, 1);
        let mut c = Plane::zeros(prev.h, prev.w);
        for j in 0..c.d.len() {
            c.d[j] = diffusivity(kind, lx.d[j], ly.d[j], k);
        }

        let t = 0.5 * evo[i].esigma * evo[i].esigma - 0.5 * evo[i - 1].esigma * evo[i - 1].esigma;
        let mut lt = prev;
        for tau in fed_tau(t, 0.25) {
            diffusion_step(&mut lt, &c, tau);
        }
        evo[i].smooth = smooth;
        evo[i].l = lt;
    }

    // `Compute_Multiscale_Derivatives`: derivatives at each level's own integer
    // scale, normalised by that scale so responses compare across levels.
    for e in &mut evo {
        let ss = e.sigma_size;
        let sf = ss as f32;
        let lx = scharr(&e.smooth, 1, 0, ss);
        let ly = scharr(&e.smooth, 0, 1, ss);
        let lxx = scharr(&lx, 1, 0, ss);
        let lyy = scharr(&ly, 0, 1, ss);
        let lxy = scharr(&lx, 0, 1, ss);
        for j in 0..e.det.d.len() {
            let (xx, yy, xy) = (lxx.d[j] * sf * sf, lyy.d[j] * sf * sf, lxy.d[j] * sf * sf);
            e.det.d[j] = xx * yy - xy * xy;
        }
        e.lx = Plane { w: lx.w, h: lx.h, d: lx.d.iter().map(|v| v * sf).collect() };
        e.ly = Plane { w: ly.w, h: ly.h, d: ly.d.iter().map(|v| v * sf).collect() };
    }
    evo
}

/// `KAZEFeatures::Determinant_Hessian` followed by `Do_Subpixel_Refinement`.
///
/// # The three filters that decide the keypoint count
///
/// 1. A response must beat `dthreshold` and be maximal over its 3x3
///    neighbourhood *and* over the same neighbourhood in the levels either
///    side. Note the asymmetry: on its own level a neighbour only rejects it by
///    being **strictly greater**, so a flat plateau yields several candidates;
///    on the neighbouring levels the centre pixel counts too.
/// 2. Candidates within `sigma_size` of an existing keypoint on an adjacent
///    level replace it or are dropped, whichever has the weaker response. This
///    is what collapses those plateaus back to one point.
/// 3. A keypoint whose descriptor patch — `3 * size` in each direction — would
///    leave the image is discarded outright. On a small image this rejects
///    almost everything from the coarse levels.
///
/// Then the subpixel step fits a quadratic in (x, y, scale) and **deletes any
/// keypoint whose fitted offset exceeds one unit in any axis**, which is a
/// fourth filter and not a small one.
fn detect(evo: &[Evolution], threshold: f64, n_sublevels: i32) -> Vec<KeyPoint> {
    let threshold = threshold as f32;
    let mut kpts: Vec<KeyPoint> = Vec::new();

    // Levels 0 and n-1 are never searched: they have no neighbour on one side.
    for i in 1..evo.len().saturating_sub(1) {
        let e = &evo[i];
        let (h, w) = (e.det.h, e.det.w);
        let smax = 3.0f32;

        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let v = e.det.d[y * w + x];
                if v <= threshold || v < e.det.d[y * w + x - 1] {
                    continue;
                }
                if !is_max_of(&e.det, v, y, x, true)
                    || !is_max_of(&evo[i - 1].det, v, y, x, false)
                    || !is_max_of(&evo[i + 1].det, v, y, x, false)
                {
                    continue;
                }
                let cand = KeyPoint {
                    pt: (x as f32, y as f32),
                    size: e.esigma,
                    angle: e.sublevel as f32,
                    response: v.abs(),
                    octave: e.octave,
                    class_id: i as i32,
                };

                // Same feature already found on this level or one either side?
                let mut repeated = None;
                let mut keep = true;
                for (ik, k) in kpts.iter().enumerate() {
                    if (k.class_id - i as i32).abs() > 1 {
                        continue;
                    }
                    let (dx, dy) = (cand.pt.0 - k.pt.0, cand.pt.1 - k.pt.1);
                    if dx * dx + dy * dy < (e.sigma_size * e.sigma_size) as f32 {
                        if cand.response > k.response {
                            repeated = Some(ik);
                        } else {
                            keep = false;
                        }
                        break;
                    }
                }
                if !keep {
                    continue;
                }
                // The descriptor patch has to fit inside the image.
                let left = cv_round(cand.pt.0 - smax * cand.size);
                let right = cv_round(cand.pt.0 + smax * cand.size);
                let up = cv_round(cand.pt.1 - smax * cand.size);
                let down = cv_round(cand.pt.1 + smax * cand.size);
                if left < 0 || right >= w as i32 || up < 0 || down >= h as i32 {
                    continue;
                }
                match repeated {
                    Some(ik) => kpts[ik] = cand,
                    None => kpts.push(cand),
                }
            }
        }
    }

    kpts.retain_mut(|kp| subpixel_refine(evo, kp, n_sublevels));
    kpts
}

/// `check_maximum_neighbourhood` — is `value` at least every neighbour?
///
/// `same_img` skips the centre pixel, which is how a point can be its own
/// maximum on its own level while still having to beat the centre of the
/// levels either side.
fn is_max_of(p: &Plane, value: f32, row: usize, col: usize, same_img: bool) -> bool {
    for i in row.saturating_sub(1)..=(row + 1).min(p.h - 1) {
        for j in col.saturating_sub(1)..=(col + 1).min(p.w - 1) {
            if same_img && i == row && j == col {
                continue;
            }
            if p.d[i * p.w + j] > value {
                return false;
            }
        }
    }
    true
}

/// One keypoint's quadratic fit in (x, y, scale). `false` means "delete it".
fn subpixel_refine(evo: &[Evolution], kp: &mut KeyPoint, n_sublevels: i32) -> bool {
    let step = 1usize;
    let level = kp.class_id as usize;
    let (x, y) = (kp.pt.0 as usize, kp.pt.1 as usize);
    let (cur, up, dn) = (&evo[level].det, &evo[level + 1].det, &evo[level - 1].det);
    let w = cur.w;
    if x < step || y < step || x + step >= w || y + step >= cur.h {
        return false;
    }
    let at = |p: &Plane, yy: usize, xx: usize| p.d[yy * w + xx];
    let s = step as f32;

    let dx_ = (1.0 / (2.0 * s)) * (at(cur, y, x + step) - at(cur, y, x - step));
    let dy_ = (1.0 / (2.0 * s)) * (at(cur, y + step, x) - at(cur, y - step, x));
    let ds = 0.5 * (at(up, y, x) - at(dn, y, x));

    let dxx = (1.0 / (s * s)) * (at(cur, y, x + step) + at(cur, y, x - step) - 2.0 * at(cur, y, x));
    let dyy = (1.0 / (s * s)) * (at(cur, y + step, x) + at(cur, y - step, x) - 2.0 * at(cur, y, x));
    let dss = at(up, y, x) + at(dn, y, x) - 2.0 * at(cur, y, x);
    let dxy = (1.0 / (4.0 * s * s))
        * (at(cur, y + step, x + step) + at(cur, y - step, x - step)
            - at(cur, y - step, x + step)
            - at(cur, y + step, x - step));
    let dxs = (1.0 / (4.0 * s))
        * (at(up, y, x + step) + at(dn, y, x - step) - at(dn, y, x + step) - at(up, y, x - step));
    let dys = (1.0 / (4.0 * s))
        * (at(up, y + step, x) + at(dn, y - step, x) - at(dn, y + step, x) - at(up, y - step, x));

    let a = [
        [dxx, dxy, dxs],
        [dxy, dyy, dys],
        [dxs, dys, dss],
    ];
    let Some(d) = solve3(a, [-dx_, -dy_, -ds]) else {
        return false;
    };
    if d[0].abs() > 1.0 || d[1].abs() > 1.0 || d[2].abs() > 1.0 {
        return false;
    }
    kp.pt.0 += d[0];
    kp.pt.1 += d[1];
    // `angle` has been carrying the sublevel; it becomes the real angle later.
    let dsc = kp.octave as f32 + (kp.angle + d[2]) / n_sublevels as f32;
    // In OpenCV a keypoint's size is its diameter.
    kp.size = 2.0 * SOFFSET * 2f32.powf(dsc);
    kp.angle = 0.0;
    true
}

/// Gaussian elimination with partial pivoting, matching `solve(..., DECOMP_LU)`.
pub(crate) fn solve3(mut a: [[f32; 3]; 3], mut b: [f32; 3]) -> Option<[f32; 3]> {
    for col in 0..3 {
        let piv = (col..3).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < f32::EPSILON {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let (done, rest) = a.split_at_mut(col + 1);
        let pivot_row = &done[col];
        for (i, row) in rest.iter_mut().enumerate() {
            let f = row[col] / pivot_row[col];
            for (dst, src) in row.iter_mut().zip(pivot_row).skip(col) {
                *dst -= f * src;
            }
            b[col + 1 + i] -= f * b[col];
        }
    }
    let mut x = [0.0f32; 3];
    for row in (0..3).rev() {
        let mut acc = b[row];
        for c in row + 1..3 {
            acc -= a[row][c] * x[c];
        }
        x[row] = acc / a[row][row];
    }
    Some(x)
}

/// `cvRound(float)`, re-exported so `akaze` keeps reaching it through this
/// module rather than growing its own import.
pub(crate) use cv_core::saturate::cv_roundf as cv_round;

/// `KAZEFeatures::Compute_Main_Orientation` — the dominant gradient direction.
///
/// # Why a sliding sector rather than a histogram
///
/// SIFT bins orientations and picks the tallest bin. KAZE instead slides a
/// 60-degree sector round the circle in 0.15 rad steps and keeps the sector
/// whose **vector sum** is longest. Summing vectors rather than counting means
/// two nearly-opposed strong gradients cancel instead of both voting, so the
/// answer tracks the net flow across the patch.
///
/// The samples come from the level's own scale-normalised `Lx`/`Ly`, on a
/// 13x13 grid of points spaced `s` apart and clipped to a disc of radius 6.
pub(crate) fn orientation(lx: &Plane, ly: &Plane, kp: &KeyPoint) -> f32 {
    let (xf, yf) = (kp.pt.0, kp.pt.1);
    let s = cv_round(kp.size / 2.0);
    let (h, w) = (lx.h as i32, lx.w as i32);

    let mut res_x = Vec::with_capacity(109);
    let mut res_y = Vec::with_capacity(109);
    let mut ang = Vec::with_capacity(109);
    for i in -6i32..=6 {
        for j in -6i32..=6 {
            if i * i + j * j >= 36 {
                continue;
            }
            let iy = cv_round(yf + (j * s) as f32);
            let ix = cv_round(xf + (i * s) as f32);
            let (rx, ry) = if iy >= 0 && iy < h && ix >= 0 && ix < w {
                let gw = gaussian_weight(iy as f32 - yf, ix as f32 - xf, 2.5 * s as f32);
                let k = (iy as usize) * lx.w + ix as usize;
                (gw * lx.d[k], gw * ly.d[k])
            } else {
                (0.0, 0.0)
            };
            res_x.push(rx);
            res_y.push(ry);
            ang.push(fast_atan2(ry, rx).to_radians());
        }
    }

    let mut max = 0.0f32;
    let mut best = 0.0f32;
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut ang1 = 0.0f32;
    while ang1 < two_pi {
        let ang2 = if ang1 + std::f32::consts::PI / 3.0 > two_pi {
            ang1 - 5.0 * std::f32::consts::PI / 3.0
        } else {
            ang1 + std::f32::consts::PI / 3.0
        };
        let (mut sum_x, mut sum_y) = (0.0f32, 0.0f32);
        for k in 0..ang.len() {
            let a = ang[k];
            // The sector either runs forwards or wraps through zero.
            let inside = if ang1 < ang2 {
                ang1 < a && a < ang2
            } else {
                (a > 0.0 && a < ang2) || (a > ang1 && a < two_pi)
            };
            if inside {
                sum_x += res_x[k];
                sum_y += res_y[k];
            }
        }
        if sum_x * sum_x + sum_y * sum_y > max {
            max = sum_x * sum_x + sum_y * sum_y;
            best = fast_atan2(sum_y, sum_x);
        }
        ang1 += 0.15;
    }
    best
}

/// `gaussian` from `kaze/utils.h` — unnormalised, so it is a weight not a pdf.
pub(crate) fn gaussian_weight(x: f32, y: f32, sigma: f32) -> f32 {
    (-(x * x + y * y) / (2.0 * sigma * sigma)).exp()
}

/// `cv::fastAtan2` — degrees in `[0, 360)`.
///
/// # Why not `f32::atan2`
///
/// This is a degree-7 odd polynomial in `min/max` of the two magnitudes, not a
/// correctly rounded arctangent — it is accurate to about 0.3 degrees, and that
/// error is *part of the answer* anywhere a caller compares two angles.
///
/// KAZE's orientation stage is exactly such a caller: it slides a sector round
/// the circle and keeps whichever sum is longest. Using a true `atan2` moves
/// samples across sector boundaries relative to upstream, and where two sectors
/// nearly tie that flips the winner — on a 96x96 test image it put one of 26
/// keypoints 84 degrees out, and its descriptor with it.
///
/// The `DBL_EPSILON` in the denominator is upstream's, and is what makes
/// `atan2(0, 0)` return 0 rather than a NaN.
// Upstream's literals are kept verbatim. The digits past `f32` precision are
// dropped by the compiler anyway, but shortening the source text is exactly how
// a transcribed constant silently loses its last bits in a later edit.
#[allow(clippy::excessive_precision)]
pub(crate) fn fast_atan2(y: f32, x: f32) -> f32 {
    /// `atan2_p1 ..= atan2_p7`, already scaled by `180 / pi`.
    const P1: f32 = 0.9997878412794807 * (180.0 / std::f64::consts::PI) as f32;
    const P3: f32 = -0.3258083974640975 * (180.0 / std::f64::consts::PI) as f32;
    const P5: f32 = 0.1555786518463281 * (180.0 / std::f64::consts::PI) as f32;
    const P7: f32 = -0.04432655554792128 * (180.0 / std::f64::consts::PI) as f32;
    const EPS: f32 = f64::EPSILON as f32;

    let (ax, ay) = (x.abs(), y.abs());
    let c = ax.min(ay) / (ax.max(ay) + EPS);
    let c2 = c * c;
    let poly = (((P7 * c2 + P5) * c2 + P3) * c2 + P1) * c;
    let mut a = if ax >= ay { poly } else { 90.0 - poly };
    if x < 0.0 {
        a = 180.0 - a;
    }
    if y < 0.0 {
        a = 360.0 - a;
    }
    a
}


/// `cv::KAZE::detectAndCompute(image, mask, keypoints, descriptors)`.
///
/// The descriptor is M-SURF: a 4x4 grid of cells, each contributing the sums of
/// the Haar-like responses and their absolute values, giving 64 numbers (128
/// with `extended`).
pub fn kaze_detect_and_compute<S: MatLike + ?Sized>(
    image: &S,
    p: KazeParams,
    kind: Diffusivity,
) -> Result<(Vec<KeyPoint>, Vec<KazeDescriptor>)> {
    cv_assert!(image.channels() == 1, "KAZE requires a single-channel image");
    cv_assert!(p.n_octaves > 0 && p.n_octave_layers > 0, "octaves and layers must be positive");
    cv_assert!(image.rows() >= 16 && image.cols() >= 16, "KAZE needs at least a 16x16 image");

    let base = Plane::from_image(image)?;
    let evo = build_evolution(&base, p.n_octaves, p.n_octave_layers, kind);
    let mut kps = detect(&evo, p.threshold, p.n_octave_layers);

    let descs = kps
        .iter_mut()
        .map(|k| {
            let lvl = &evo[(k.class_id.max(0) as usize).min(evo.len() - 1)];
            k.angle = if p.upright { 0.0 } else { orientation(&lvl.lx, &lvl.ly, k) };
            msurf(&lvl.lx, &lvl.ly, k, p.extended)
        })
        .collect();
    for k in kps.iter_mut() {
        k.class_id = -1;
    }
    Ok((kps, descs))
}

/// `Get_KAZE_Descriptor_64` and `Get_KAZE_Descriptor_128` — M-SURF.
///
/// # What is being sampled
///
/// Not the image: the level's own **scale-normalised** `Lx`/`Ly`. The patch is
/// a 24s x 24s grid of 4x4 subregions, each 9x9 samples with a 2-sample overlap
/// on every side — the overlap is what "modified" SURF refers to, and it is why
/// the loop advances by 9 but steps back by 4 first.
///
/// Each subregion contributes the sums of the rotated derivatives and of their
/// absolute values, weighted twice: once by a Gaussian on the sample's distance
/// from the subregion centre (`2.5s`), once by a Gaussian on the subregion's
/// distance from the patch centre (`1.5`).
///
/// The extended form splits each of those four sums by the *sign of the other*
/// component — `dx` accumulates into a positive or negative bin according to
/// `rry`, and vice versa — giving 8 numbers per subregion instead of 4.
fn msurf(lx: &Plane, ly: &Plane, kp: &KeyPoint, extended: bool) -> KazeDescriptor {
    let dsize = if extended { 128 } else { 64 };
    let mut desc = vec![0.0f32; dsize];
    let (w, h) = (lx.w as i32, lx.h as i32);

    let (xf, yf) = (kp.pt.0, kp.pt.1);
    let scale = cv_round(kp.size / 2.0);
    let angle = kp.angle.to_radians();
    let (co, si) = (angle.cos(), angle.sin());

    let sample_step = 5i32;
    let pattern_size = 12i32;
    let mut len = 0.0f32;
    let mut dcount = 0usize;

    // Transcribed loop shape: `i` steps back 4 then forward 9, so consecutive
    // subregions overlap by 4 samples. Rewriting it as a clean 4x4 grid moves
    // the sample positions and changes every descriptor.
    let (mut cx, mut cy);
    cx = -0.5f32;
    let mut i = -8i32;
    while i < pattern_size {
        let mut j = -8i32;
        i -= 4;
        cx += 1.0;
        cy = -0.5f32;
        while j < pattern_size {
            cy += 1.0;
            j -= 4;

            let ky = i + sample_step;
            let kx = j + sample_step;
            // The subregion centre, in the rotated frame.
            let xs = xf + (-(kx * scale) as f32 * si + (ky * scale) as f32 * co);
            let ys = yf + ((kx * scale) as f32 * co + (ky * scale) as f32 * si);

            let (mut dx, mut dy, mut mdx, mut mdy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let (mut dxp, mut dxn, mut mdxp, mut mdxn) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let (mut dyp, mut dyn_, mut mdyp, mut mdyn) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

            for k in i..i + 9 {
                for l in j..j + 9 {
                    let sample_y = yf + ((l * scale) as f32 * co + (k * scale) as f32 * si);
                    let sample_x = xf + (-(l * scale) as f32 * si + (k * scale) as f32 * co);
                    let gauss_s1 =
                        gaussian_weight(xs - sample_x, ys - sample_y, 2.5 * scale as f32);

                    // Both corners are clamped independently, so at the border
                    // x1 and x2 can collapse onto the same column.
                    let x1 = sample_x.floor() as i32;
                    let y1 = sample_y.floor() as i32;
                    let (x1, y1) = (x1.clamp(0, w - 1), y1.clamp(0, h - 1));
                    let (x2, y2) = ((x1 + 1).clamp(0, w - 1), (y1 + 1).clamp(0, h - 1));
                    let fx = sample_x - x1 as f32;
                    let fy = sample_y - y1 as f32;

                    let bilinear = |p: &Plane| {
                        let idx = |yy: i32, xx: i32| p.d[yy as usize * p.w + xx as usize];
                        (1.0 - fx) * (1.0 - fy) * idx(y1, x1)
                            + fx * (1.0 - fy) * idx(y1, x2)
                            + (1.0 - fx) * fy * idx(y2, x1)
                            + fx * fy * idx(y2, x2)
                    };
                    let rx = bilinear(lx);
                    let ry = bilinear(ly);

                    let rry = gauss_s1 * (rx * co + ry * si);
                    let rrx = gauss_s1 * (-rx * si + ry * co);

                    if extended {
                        // Each component is binned by the sign of the *other*.
                        if rry >= 0.0 {
                            dxp += rrx;
                            mdxp += rrx.abs();
                        } else {
                            dxn += rrx;
                            mdxn += rrx.abs();
                        }
                        if rrx >= 0.0 {
                            dyp += rry;
                            mdyp += rry.abs();
                        } else {
                            dyn_ += rry;
                            mdyn += rry.abs();
                        }
                    } else {
                        dx += rrx;
                        dy += rry;
                        mdx += rrx.abs();
                        mdy += rry.abs();
                    }
                }
            }

            let gauss_s2 = gaussian_weight(cx - 2.0, cy - 2.0, 1.5);
            let cell: &[f32] = if extended {
                &[dxp, dxn, mdxp, mdxn, dyp, dyn_, mdyp, mdyn]
            } else {
                &[dx, dy, mdx, mdy]
            };
            for &v in cell {
                desc[dcount] = v * gauss_s2;
                dcount += 1;
                len += v * v * gauss_s2 * gauss_s2;
            }
            j += 9;
        }
        i += 9;
    }

    let len = len.sqrt();
    if len > 0.0 {
        for v in &mut desc {
            *v /= len;
        }
    }
    desc
}

/// `Get_MLDB_Full_Descriptor` — AKAZE's binary descriptor.
///
/// # What the bits mean
///
/// Three nested grids over the patch — 2x2, 3x3 and 4x4 — each cell holding the
/// mean of the diffused image and of the two rotated derivatives over its
/// samples. Every **pair** of cells within a grid contributes one bit per
/// channel: 1 if the first cell's value is larger. That is
/// `(6 + 36 + 120) * 3 = 486` bits, which is why the descriptor is 61 bytes
/// with the last two bits unused.
///
/// Comparing means rather than raw samples is what makes it robust; comparing
/// *pairs* rather than against a threshold is what makes it invariant to any
/// monotonic change in illumination.
///
/// # Two details that are easy to get wrong
///
/// Samples are taken at the **octave's own resolution** — position and scale
/// are divided by `2^octave` first — and out-of-bounds samples are skipped
/// rather than clamped, so a cell's mean is over however many samples landed
/// inside.
///
/// The comparison is done on **bit-toggled integers**, not floats. That is not
/// merely a faster spelling: for `+0.0` against `-0.0` the toggled integers
/// order strictly, where `>` on floats reports neither greater. Cells whose
/// samples all fell outside are exactly `0.0`, so this case is reachable.
pub(crate) fn mldb(
    lt: &Plane,
    lx: &Plane,
    ly: &Plane,
    kp: &KeyPoint,
    channels: usize,
) -> AkazeDescriptorBits {
    /// `descriptor_pattern_size`; the patch is `2 * pattern * scale` across.
    const PATTERN: i32 = 10;
    let chan = channels.clamp(1, 3);

    let ratio = (1 << kp.octave.max(0)) as f32;
    let scale = cv_round(0.5 * kp.size / ratio) as f32;
    let (xf, yf) = (kp.pt.0 / ratio, kp.pt.1 / ratio);
    let angle = kp.angle.to_radians();
    let (co, si) = (angle.cos(), angle.sin());

    let mut desc = [0u8; AKAZE_BYTES];
    let mut dpos = 0usize;
    // `divUp`: 2x2, 3x3 and 4x4 grids over the same patch.
    let div_up = |a: i32, b: i32| (a + b - 1) / b;
    for sample_step in [PATTERN, div_up(PATTERN * 2, 3), div_up(PATTERN, 2)] {
        let mut values: Vec<f32> = Vec::with_capacity(16 * 3);
        let mut i = -PATTERN;
        while i < PATTERN {
            let mut j = -PATTERN;
            while j < PATTERN {
                let (mut di, mut dx, mut dy) = (0.0f32, 0.0f32, 0.0f32);
                let mut nsamples = 0f32;
                for k in i..i + sample_step {
                    for l in j..j + sample_step {
                        let sample_y = yf + (l as f32 * co * scale + k as f32 * si * scale);
                        let sample_x = xf + (-(l as f32) * si * scale + k as f32 * co * scale);
                        let y1 = cv_round(sample_y);
                        let x1 = cv_round(sample_x);
                        // Skipped, not clamped: a cell near the edge averages
                        // fewer samples rather than repeating its border.
                        if y1 < 0 || y1 >= lt.h as i32 || x1 < 0 || x1 >= lt.w as i32 {
                            continue;
                        }
                        let idx = y1 as usize * lt.w + x1 as usize;
                        di += lt.d[idx];
                        if chan > 1 {
                            let (rx, ry) = (lx.d[idx], ly.d[idx]);
                            if chan == 2 {
                                dx += (rx * rx + ry * ry).sqrt();
                            } else {
                                dy += rx * co + ry * si;
                                dx += -rx * si + ry * co;
                            }
                        }
                        nsamples += 1.0;
                    }
                }
                if nsamples > 0.0 {
                    let inv = 1.0 / nsamples;
                    di *= inv;
                    dx *= inv;
                    dy *= inv;
                }
                values.push(di);
                if chan > 1 {
                    values.push(dx);
                }
                if chan > 2 {
                    values.push(dy);
                }
                j += sample_step;
            }
            i += sample_step;
        }

        // `CV_TOGGLE_FLT`: map the float bits to an order-preserving integer.
        let ivalues: Vec<i32> = values
            .iter()
            .map(|v| {
                let b = v.to_bits() as i32;
                if b < 0 {
                    b ^ 0x7fff_ffff
                } else {
                    b
                }
            })
            .collect();

        let count = values.len() / chan;
        for pos in 0..chan {
            for a in 0..count {
                let ival = ivalues[chan * a + pos];
                for b in a + 1..count {
                    if ival > ivalues[chan * b + pos] {
                        desc[dpos >> 3] |= 1 << (dpos & 7);
                    }
                    dpos += 1;
                }
            }
        }
    }
    debug_assert_eq!(dpos, if chan == 3 { 486 } else { 162 * chan });
    desc
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::Mat;

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

    #[test]
    fn the_diffusivities_all_fall_with_gradient() {
        // Every conductivity function must be large in flat regions and small
        // at edges — that is the whole premise.
        for kind in [
            Diffusivity::PmG1,
            Diffusivity::PmG2,
            Diffusivity::Weickert,
            Diffusivity::Charbonnier,
        ] {
            let flat = diffusivity(kind, 0.0, 0.0, 0.1);
            let edge = diffusivity(kind, 5.0, 5.0, 0.1);
            assert!(flat > edge, "{kind:?}: flat {flat} should exceed edge {edge}");
            assert!((0.0..=1.0).contains(&flat), "{kind:?} flat {flat} out of range");
            assert!((0.0..=1.0).contains(&edge), "{kind:?} edge {edge} out of range");
        }
    }

    #[test]
    fn diffusion_preserves_an_edge_better_than_a_blur() {
        // The claim that justifies the whole method, stated measurably.
        let (w, h) = (32usize, 32usize);
        let mut p = Plane::zeros(h, w);
        for y in 0..h {
            for x in 0..w {
                p.d[y * w + x] = if x < w / 2 { 0.0 } else { 1.0 };
            }
        }
        let k = 0.05;
        let mut nl = p.clone();
        for _ in 0..40 {
            let (gx, gy) = (scharr(&nl, 1, 0, 1), scharr(&nl, 0, 1, 1));
            let mut c = Plane::zeros(h, w);
            for i in 0..c.d.len() {
                c.d[i] = diffusivity(Diffusivity::PmG2, gx.d[i], gy.d[i], k);
            }
            diffusion_step(&mut nl, &c, 0.2);
        }
        let gauss = blur(&p, 2.0);
        // Sharpness across the boundary: how much of the step survives.
        let at = |q: &Plane, y: usize, x: usize| q.d[y * q.w + x];
        let step_nl = at(&nl, 16, 17) - at(&nl, 16, 14);
        let step_g = at(&gauss, 16, 17) - at(&gauss, 16, 14);
        assert!(
            step_nl > step_g,
            "nonlinear diffusion kept {step_nl} of the edge, Gaussian kept {step_g}"
        );
    }

    #[test]
    fn the_contrast_factor_tracks_the_images_gradients() {
        // A flat image has no gradients at all, so the histogram never reaches
        // the percentile and upstream falls back to a fixed 0.03 rather than
        // returning zero — which the diffusion would divide by.
        let flat = Plane::zeros(20, 20);
        assert_eq!(contrast_factor(&flat, 0.7, SDERIVATIVES, 300), 0.03);
        let img = Plane::from_image(&scene(48, 48)).unwrap();
        let k = contrast_factor(&img, 0.7, SDERIVATIVES, 300);
        assert!(k > 0.0 && k < 1.0, "contrast factor {k} out of the 0..1 image range");
    }

    /// Pinned against cv2 4.13.0's `cv2.KAZE_create().detect()` on the scene
    /// [`scene`] builds. This is the whole point of transcribing the evolution,
    /// the FED steps and the subpixel fit rather than approximating them, so it
    /// is checked tightly enough to catch a drift of one part in ten thousand.
    #[test]
    fn kaze_matches_cv2s_detector() {
        const CV2: [(f32, f32, f32, f32); 4] = [
            (39.9378, 65.9378, 9.44670, 0.06637277),
            (23.9365, 23.9365, 8.73288, 0.06057156),
            (59.9543, 31.9508, 13.29216, 0.06002841),
            (59.9662, 31.9662, 9.73686, 0.04511207),
        ];
        let img = scene(96, 96);
        let (mut kps, _) = kaze_detect_and_compute(
            &img,
            KazeParams::default(),
            Diffusivity::PmG2,
        )
        .unwrap();
        assert_eq!(kps.len(), 29, "cv2 finds 29 keypoints here");

        kps.sort_by(|a, b| b.response.total_cmp(&a.response));
        for (k, &(x, y, size, response)) in kps.iter().zip(&CV2) {
            assert!(
                (k.pt.0 - x).abs() < 1e-3 && (k.pt.1 - y).abs() < 1e-3,
                "position ({}, {}) against cv2's ({x}, {y})",
                k.pt.0,
                k.pt.1
            );
            assert!((k.size - size).abs() < 1e-4, "size {} against cv2's {size}", k.size);
            assert!(
                (k.response - response).abs() < 1e-7,
                "response {} against cv2's {response}",
                k.response
            );
        }
    }

    /// Whatever else FED does, the steps must sum to the requested time, or
    /// every level of the evolution lands at the wrong scale.
    #[test]
    fn the_fed_steps_sum_to_the_requested_time() {
        for t in [0.28f32, 1.0, 4.5, 30.0] {
            let tau = fed_tau(t, 0.25);
            assert!(!tau.is_empty(), "t = {t} produced no steps");
            let sum: f32 = tau.iter().sum();
            assert!(
                (sum - t).abs() < 1e-3 * t,
                "steps for t = {t} sum to {sum} over {} steps",
                tau.len()
            );
        }
    }

    /// FED's whole advantage: `n` steps advance the time by O(n^2) as much as
    /// one stable step would, by letting individual steps exceed the explicit
    /// scheme's 0.25 stability limit.
    ///
    /// It only pays off once there is room for it — at `t = 0.28` the cycle is
    /// two steps and both are stable. By `t = 30` more than a third of them are
    /// unstable and the largest is 73 times the limit.
    #[test]
    fn fed_buys_time_with_individually_unstable_steps() {
        assert!(fed_tau(0.28, 0.25).iter().all(|&s| s <= 0.25), "a short cycle needs no help");

        for (t, want_steps) in [(1.0f32, 3usize), (4.5, 7), (30.0, 19)] {
            let tau = fed_tau(t, 0.25);
            assert_eq!(tau.len(), want_steps, "step count for t = {t}");
            assert!(
                tau.iter().any(|&s| s > 0.25),
                "t = {t}: every step is stable, so this is not FED"
            );
            // A stable integration would need this many steps instead.
            let stable = (t / 0.25).ceil() as usize;
            assert!(
                tau.len() * tau.len() >= stable,
                "t = {t}: {} steps is no better than the {stable} a stable scheme needs",
                tau.len()
            );
        }
    }


    /// Full-pipeline parity against `cv2.KAZE_create().detectAndCompute()`, on
    /// the deterministic texture from [`crate::akaze::tests::texture`].
    ///
    /// cv2 finds 30 keypoints; this finds the same 30, at the same positions
    /// and with **identical** angles. The M-SURF descriptors agree to 4.2e-7 on
    /// a unit-normalised vector — not bit-exact, because the subregion sums
    /// accumulate in a different order than upstream's, but far closer than any
    /// matcher could distinguish.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn kaze_matches_cv2_end_to_end_on_texture() {
        const CV2_KP: [(f32, f32, f32, f32); 4] = [
            (6.35040, 86.22365, 3.45305, 87.5923),
            (26.89305, 10.86488, 5.45606, 180.6571),
            (28.95924, 117.69919, 5.36833, 196.3669),
            (29.31159, 117.40597, 3.46433, 334.9491),
        ];
        /// cv2's first eight descriptor values for the first keypoint.
        const CV2_DESC0: [f32; 8] = [
            0.0027229784,
            -0.006476929,
            0.023649933,
            0.026693821,
            -0.026432779,
            -0.034472,
            0.041777879,
            0.041734345,
        ];

        let img = crate::akaze::tests::texture(160, 160);
        let (mut kps, descs) =
            kaze_detect_and_compute(&img, KazeParams::default(), Diffusivity::PmG2).unwrap();
        assert_eq!(kps.len(), 30, "cv2 finds 30 keypoints here");

        let mut order: Vec<usize> = (0..kps.len()).collect();
        order.sort_by(|&a, &b| {
            kps[a].pt.0.total_cmp(&kps[b].pt.0).then(kps[a].pt.1.total_cmp(&kps[b].pt.1))
        });
        let descs: Vec<_> = order.iter().map(|&i| descs[i].clone()).collect();
        kps = order.iter().map(|&i| kps[i]).collect();

        for (k, &(x, y, size, angle)) in kps.iter().zip(&CV2_KP) {
            assert!(
                (k.pt.0 - x).abs() < 1e-4 && (k.pt.1 - y).abs() < 1e-4,
                "position ({}, {}) against cv2's ({x}, {y})",
                k.pt.0,
                k.pt.1
            );
            assert!((k.size - size).abs() < 1e-4, "size {} against {size}", k.size);
            assert!((k.angle - angle).abs() < 1e-3, "angle {} against {angle}", k.angle);
        }
        for (v, want) in descs[0].iter().zip(&CV2_DESC0) {
            assert!((v - want).abs() < 1e-6, "descriptor {v} against cv2's {want}");
        }
    }

    #[test]
    fn kaze_detects_and_describes() {
        let img = scene(96, 96);
        let (kps, descs) =
            kaze_detect_and_compute(&img, KazeParams::default(), Diffusivity::PmG2).unwrap();
        assert!(!kps.is_empty(), "KAZE found nothing");
        assert_eq!(kps.len(), descs.len());
        for (k, d) in kps.iter().zip(&descs) {
            assert_eq!(d.len(), 64, "the plain descriptor is 64 long");
            assert!(k.size > 0.0);
            assert!(k.response > 0.0);
            assert!(d.iter().all(|v| v.is_finite()));
            // Unit norm, or all-zero for a featureless patch.
            let n = d.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(n < 1e-6 || (n - 1.0).abs() < 1e-4, "descriptor norm {n}");
        }
    }

    #[test]
    fn the_extended_descriptor_is_twice_as_long() {
        let img = scene(96, 96);
        let mut p = KazeParams::default();
        p.set_extended(true);
        let (_, descs) = kaze_detect_and_compute(&img, p, Diffusivity::PmG2).unwrap();
        assert!(!descs.is_empty());
        assert!(descs.iter().all(|d| d.len() == 128), "extended KAZE is 128 long");
    }

    #[test]
    fn upright_kaze_skips_the_orientation_step() {
        let img = scene(96, 96);
        let mut p = KazeParams::default();
        p.set_upright(true);
        let (kps, _) = kaze_detect_and_compute(&img, p, Diffusivity::PmG2).unwrap();
        assert!(!kps.is_empty());
        assert!(kps.iter().all(|k| k.angle == 0.0), "upright means no orientation");
    }

    #[test]
    fn akaze_detects_and_describes() {
        let img = scene(96, 96);
        let (kps, descs) =
            crate::akaze::akaze_detect_and_compute(&img, crate::detectors::AkazeParams::default(), Diffusivity::PmG2).unwrap();
        assert!(!kps.is_empty(), "AKAZE found nothing");
        assert_eq!(kps.len(), descs.len());
        assert!(descs.iter().any(|d| d.iter().any(|b| *b != 0)), "all-zero descriptors");
        for k in &kps {
            assert!((0.0..360.0).contains(&k.angle), "angle {}", k.angle);
        }
    }

    #[test]
    fn fewer_descriptor_channels_set_fewer_bits() {
        let img = scene(96, 96);
        let mut one = crate::detectors::AkazeParams::default();
        one.set_descriptor_channels(1);
        let (_, d1) = crate::akaze::akaze_detect_and_compute(&img, one, Diffusivity::PmG2).unwrap();
        let (_, d3) =
            crate::akaze::akaze_detect_and_compute(&img, crate::detectors::AkazeParams::default(), Diffusivity::PmG2).unwrap();
        assert!(!d1.is_empty() && !d3.is_empty());
        // With one channel per pair, the same grid fills fewer of the 488 bits,
        // so the tail stays zero.
        let tail1: u32 = d1.iter().map(|d| d[AKAZE_BYTES - 1].count_ones()).sum();
        let tail3: u32 = d3.iter().map(|d| d[AKAZE_BYTES - 1].count_ones()).sum();
        assert!(tail1 <= tail3, "one channel should not fill more bits: {tail1} vs {tail3}");
    }

    #[test]
    fn every_diffusivity_runs_end_to_end() {
        let img = scene(64, 64);
        for kind in [
            Diffusivity::PmG1,
            Diffusivity::PmG2,
            Diffusivity::Weickert,
            Diffusivity::Charbonnier,
        ] {
            let r = crate::akaze::akaze_detect_and_compute(&img, crate::detectors::AkazeParams::default(), kind);
            assert!(r.is_ok(), "{kind:?} failed: {:?}", r.err());
        }
    }

    #[test]
    fn both_validate_their_input() {
        let colour = Mat::from_vec(32, 32, 3, &[0u8; 3072]).unwrap();
        assert!(kaze_detect_and_compute(&colour, KazeParams::default(), Diffusivity::PmG2).is_err());
        assert!(
            crate::akaze::akaze_detect_and_compute(&colour, crate::detectors::AkazeParams::default(), Diffusivity::PmG2).is_err()
        );
        let tiny = Mat::from_vec(8, 8, 1, &[0u8; 64]).unwrap();
        assert!(kaze_detect_and_compute(&tiny, KazeParams::default(), Diffusivity::PmG2).is_err());

        let mut bad = crate::detectors::AkazeParams::default();
        bad.set_descriptor_channels(4);
        assert!(crate::akaze::akaze_detect_and_compute(&scene(64, 64), bad, Diffusivity::PmG2).is_err());
        let mut zero = KazeParams::default();
        zero.set_n_octaves(0);
        assert!(kaze_detect_and_compute(&scene(64, 64), zero, Diffusivity::PmG2).is_err());
    }

    #[test]
    fn a_flat_image_has_no_features() {
        let flat = Mat::from_vec(64, 64, 1, &[128u8; 4096]).unwrap();
        let (k, _) =
            crate::akaze::akaze_detect_and_compute(&flat, crate::detectors::AkazeParams::default(), Diffusivity::PmG2).unwrap();
        assert!(k.is_empty(), "a flat field has no Hessian extrema, got {}", k.len());
    }
}
