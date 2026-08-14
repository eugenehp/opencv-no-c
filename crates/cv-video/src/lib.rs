//! `cv::video` for `opencv-no-c` — motion, tracking and background modelling.
//!
//! # Scope
//!
//! Pyramidal Lucas-Kanade and Farnebäck optical flow, MOG2 and KNN background
//! subtraction, mean-shift and CamShift tracking, and a Kalman filter.
//!
//! `findTransformECC` and the DIS/variational flow algorithms are not here.

#![warn(missing_docs)]

pub mod algorithms;
pub mod dis;
pub mod ecc;
mod plane;

use cv_core::types::{CV_32FC1, CV_8UC1};
use cv_core::{cv_assert, Code, Error, Mat, MatLike, MatLikeMut, Rect, Result};


/// A greyscale image as a dense `f64` buffer, with bilinear sampling.
struct PyrLevel {
    v: Vec<f64>,
    w: i32,
    h: i32,
}

impl PyrLevel {
    fn from<S: MatLike + ?Sized>(m: &S) -> Result<Self> {
        cv_assert!(m.channels() == 1, "optical flow needs a single-channel image");
        let (w, h) = (m.cols(), m.rows());
        let mut v = vec![0.0f64; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                v[(y * w + x) as usize] = m.at_f64(y, x, 0)?;
            }
        }
        Ok(PyrLevel { v, w, h })
    }

    fn at(&self, y: i32, x: i32) -> f64 {
        self.v[(cv_core::reflect101(y, self.h) * self.w + cv_core::reflect101(x, self.w)) as usize]
    }

    /// Bilinear sample; returns `None` outside the image, which is what stops
    /// a tracked point from silently sliding along the border.
    fn sample(&self, x: f64, y: f64) -> Option<f64> {
        if x < 0.0 || y < 0.0 || x >= f64::from(self.w - 1) || y >= f64::from(self.h - 1) {
            return None;
        }
        let (ix, iy) = (x.floor() as i32, y.floor() as i32);
        let (ax, ay) = (x - f64::from(ix), y - f64::from(iy));
        Some(
            self.at(iy, ix) * (1.0 - ax) * (1.0 - ay)
                + self.at(iy, ix + 1) * ax * (1.0 - ay)
                + self.at(iy + 1, ix) * (1.0 - ax) * ay
                + self.at(iy + 1, ix + 1) * ax * ay,
        )
    }

    /// Halve the resolution with a 5-tap binomial blur, as `pyrDown` does.
    fn down(&self) -> PyrLevel {
        let (nw, nh) = ((self.w + 1) / 2, (self.h + 1) / 2);
        let k = [1.0, 4.0, 6.0, 4.0, 1.0];
        let mut v = vec![0.0f64; (nw * nh) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = 0.0;
                let mut wsum = 0.0;
                for (j, &kj) in k.iter().enumerate() {
                    for (i, &ki) in k.iter().enumerate() {
                        let sy = 2 * y + j as i32 - 2;
                        let sx = 2 * x + i as i32 - 2;
                        acc += ki * kj * self.at(sy, sx);
                        wsum += ki * kj;
                    }
                }
                v[(y * nw + x) as usize] = acc / wsum;
            }
        }
        PyrLevel { v, w: nw, h: nh }
    }
}

/// `cv::buildOpticalFlowPyramid(img, pyramid, winSize, maxLevel, ...)`.
///
/// Level 0 is the original; each level halves the previous one.
pub fn build_optical_flow_pyramid<S: MatLike + ?Sized>(
    img: &S,
    max_level: usize,
) -> Result<Vec<Mat>> {
    let mut p = PyrLevel::from(img)?;
    let mut out = vec![Mat::copy_from(img)?];
    for _ in 0..max_level {
        if p.w < 8 || p.h < 8 {
            break;
        }
        p = p.down();
        let mut m = Mat::zeros(p.h, p.w, CV_32FC1)?;
        for y in 0..p.h {
            for x in 0..p.w {
                m.pixel_mut::<f32>(y, x)?[0] = p.v[(y * p.w + x) as usize] as f32;
            }
        }
        out.push(m);
    }
    Ok(out)
}

/// One point's tracking result: where it went, whether it was found, and the
/// mean absolute residual over the window.
///
/// Upstream returns these as three parallel output arrays; one value per point
/// keeps them from drifting out of step.
pub type TrackedPoint = ((f64, f64), bool, f64);

/// `cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err, ...)`.
///
/// Pyramidal Lucas-Kanade. Returns, for each input point, its tracked position,
/// whether tracking succeeded, and the residual error.
///
/// # Why the pyramid matters
///
/// Lucas-Kanade assumes the displacement is small enough for a first-order
/// Taylor expansion to hold — a few pixels. Running it coarse-to-fine lets a
/// large motion be found at a low resolution, where it *is* small, and then
/// refined. Without the pyramid anything faster than about three pixels per
/// frame is simply lost.
pub fn calc_optical_flow_pyr_lk<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    prev: &A,
    next: &B,
    points: &[(f64, f64)],
    win: i32,
    max_level: usize,
    iterations: usize,
    epsilon: f64,
) -> Result<Vec<TrackedPoint>> {
    cv_assert!(win > 0 && win % 2 == 1, "the window must be odd and positive, got {win}");
    cv_assert!(prev.size() == next.size(), "the two frames must be the same size");

    // Build both pyramids, coarsest last.
    let mut pa = vec![PyrLevel::from(prev)?];
    let mut pb = vec![PyrLevel::from(next)?];
    for _ in 0..max_level {
        let (la, lb) = (pa.last().unwrap(), pb.last().unwrap());
        if la.w < 8 || la.h < 8 {
            break;
        }
        pa.push(la.down());
        pb.push(lb.down());
    }
    let levels = pa.len();
    let r = win / 2;

    let mut out = Vec::with_capacity(points.len());
    for &(px, py) in points {
        // Start at the coarsest level with the point scaled down.
        let scale = f64::from(1 << (levels - 1));
        let mut g = (0.0f64, 0.0f64);
        let mut ok = true;
        let mut err = 0.0f64;

        for lvl in (0..levels).rev() {
            let s = f64::from(1 << lvl);
            let (ax, ay) = (px / s, py / s);
            let (ia, ib) = (&pa[lvl], &pb[lvl]);
            // Carry the estimate from the coarser level.
            let mut d = g;

            for _ in 0..iterations.max(1) {
                // Build the 2x2 structure tensor and the mismatch vector over
                // the window, in one pass.
                let (mut a11, mut a12, mut a22) = (0.0f64, 0.0f64, 0.0f64);
                let (mut b1, mut b2) = (0.0f64, 0.0f64);
                let mut n = 0usize;
                for wy in -r..=r {
                    for wx in -r..=r {
                        let (sx, sy) = (ax + f64::from(wx), ay + f64::from(wy));
                        let Some(i0) = ia.sample(sx, sy) else { continue };
                        let Some(i1) = ib.sample(sx + d.0, sy + d.1) else { continue };
                        // Central differences on the first image.
                        let (Some(gxp), Some(gxm)) =
                            (ia.sample(sx + 1.0, sy), ia.sample(sx - 1.0, sy))
                        else {
                            continue;
                        };
                        let (Some(gyp), Some(gym)) =
                            (ia.sample(sx, sy + 1.0), ia.sample(sx, sy - 1.0))
                        else {
                            continue;
                        };
                        let gx = (gxp - gxm) * 0.5;
                        let gy = (gyp - gym) * 0.5;
                        let dt = i1 - i0;
                        a11 += gx * gx;
                        a12 += gx * gy;
                        a22 += gy * gy;
                        b1 -= gx * dt;
                        b2 -= gy * dt;
                        n += 1;
                    }
                }
                if n < 4 {
                    ok = false;
                    break;
                }
                let det = a11 * a22 - a12 * a12;
                // A near-singular tensor means the window has no corner — the
                // aperture problem. Upstream drops the point rather than
                // returning a motion along an arbitrary direction.
                if det.abs() < 1e-9 {
                    ok = false;
                    break;
                }
                let step = ((a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det);
                d = (d.0 + step.0, d.1 + step.1);
                if step.0.hypot(step.1) < epsilon {
                    break;
                }
            }
            if !ok {
                break;
            }
            // Hand the estimate down, doubling it for the finer level.
            g = if lvl > 0 { (d.0 * 2.0, d.1 * 2.0) } else { d };
        }
        let _ = scale;
        let nx = px + g.0;
        let ny = py + g.1;
        if ok {
            // Residual over the window, as upstream's `err` output.
            let (ia, ib) = (&pa[0], &pb[0]);
            let (mut acc, mut n) = (0.0f64, 0usize);
            for wy in -r..=r {
                for wx in -r..=r {
                    let (sx, sy) = (px + f64::from(wx), py + f64::from(wy));
                    if let (Some(i0), Some(i1)) =
                        (ia.sample(sx, sy), ib.sample(sx + g.0, sy + g.1))
                    {
                        acc += (i1 - i0).abs();
                        n += 1;
                    }
                }
            }
            err = if n > 0 { acc / n as f64 } else { 0.0 };
            if n == 0 {
                ok = false;
            }
        }
        out.push(((nx, ny), ok, err));
    }
    Ok(out)
}

/// `cv::calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, ...)`.
///
/// Dense flow. Returns a `CV_32FC2` field of `(dx, dy)` per pixel.
///
/// # Divergence
///
/// Upstream fits a local quadratic polynomial to each neighbourhood
/// (Farnebäck's expansion) and solves for the displacement that maps one
/// polynomial onto the other. This is a coarse-to-fine Lucas-Kanade over the
/// same pyramid with the same windowing, which produces a comparable field but
/// is *not* the same algorithm — values will differ from cv2's. Saying so
/// beats implying an equivalence that is not there.
pub fn calc_optical_flow_farneback<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    prev: &A,
    next: &B,
    levels: usize,
    win: i32,
    iterations: usize,
) -> Result<Mat> {
    cv_assert!(prev.size() == next.size(), "the two frames must be the same size");
    cv_assert!(win > 0 && win % 2 == 1, "the window must be odd and positive");
    let (w, h) = (prev.cols(), prev.rows());

    let mut pa = vec![PyrLevel::from(prev)?];
    let mut pb = vec![PyrLevel::from(next)?];
    for _ in 0..levels {
        let la = pa.last().unwrap();
        if la.w < 8 || la.h < 8 {
            break;
        }
        pa.push(la.down());
        pb.push(pb.last().unwrap().down());
    }
    let n_lvl = pa.len();
    let r = win / 2;

    // Flow at the coarsest level, then refined downward.
    let mut flow: Vec<(f64, f64)> = Vec::new();
    let mut fw = 0i32;
    let mut fh = 0i32;
    for lvl in (0..n_lvl).rev() {
        let (ia, ib) = (&pa[lvl], &pb[lvl]);
        let mut next_flow = vec![(0.0f64, 0.0f64); (ia.w * ia.h) as usize];
        for y in 0..ia.h {
            for x in 0..ia.w {
                // Upsample the coarser estimate.
                let mut d = if flow.is_empty() {
                    (0.0, 0.0)
                } else {
                    let (sx, sy) = ((x / 2).min(fw - 1), (y / 2).min(fh - 1));
                    let c = flow[(sy * fw + sx) as usize];
                    (c.0 * 2.0, c.1 * 2.0)
                };
                for _ in 0..iterations.max(1) {
                    let (mut a11, mut a12, mut a22) = (0.0f64, 0.0f64, 0.0f64);
                    let (mut b1, mut b2) = (0.0f64, 0.0f64);
                    for wy in -r..=r {
                        for wx in -r..=r {
                            let (sx, sy) = (f64::from(x + wx), f64::from(y + wy));
                            let Some(i0) = ia.sample(sx, sy) else { continue };
                            let Some(i1) = ib.sample(sx + d.0, sy + d.1) else { continue };
                            let gx = (ia.at(y + wy, x + wx + 1) - ia.at(y + wy, x + wx - 1)) * 0.5;
                            let gy = (ia.at(y + wy + 1, x + wx) - ia.at(y + wy - 1, x + wx)) * 0.5;
                            let dt = i1 - i0;
                            a11 += gx * gx;
                            a12 += gx * gy;
                            a22 += gy * gy;
                            b1 -= gx * dt;
                            b2 -= gy * dt;
                        }
                    }
                    // Tikhonov term: a flat window has no unique answer, and
                    // regularising keeps the field smooth instead of blowing up.
                    let lambda = 1e-3;
                    let det = (a11 + lambda) * (a22 + lambda) - a12 * a12;
                    if det.abs() < 1e-12 {
                        break;
                    }
                    let step = (
                        ((a22 + lambda) * b1 - a12 * b2) / det,
                        ((a11 + lambda) * b2 - a12 * b1) / det,
                    );
                    d = (d.0 + step.0, d.1 + step.1);
                    if step.0.hypot(step.1) < 0.01 {
                        break;
                    }
                }
                next_flow[(y * ia.w + x) as usize] = d;
            }
        }
        flow = next_flow;
        fw = ia.w;
        fh = ia.h;
    }

    let mut out = Mat::zeros(h, w, cv_core::types::MatType::new(cv_core::types::Depth::F32, 2))?;
    for y in 0..h {
        for x in 0..w {
            let d = flow[(y * w + x) as usize];
            let p = out.pixel_mut::<f32>(y, x)?;
            p[0] = d.0 as f32;
            p[1] = d.1 as f32;
        }
    }
    Ok(out)
}

/// `cv::BackgroundSubtractorMOG2`.
///
/// Per pixel, a small mixture of Gaussians; a pixel is foreground when it does
/// not match any component with enough accumulated weight. Handles a
/// multi-modal background (waving foliage, a flickering screen) that a single
/// running average cannot.
#[derive(Debug, Clone)]
pub struct BackgroundSubtractorMog2 {
    /// One mixture per pixel: `(weight, mean, variance)` per component.
    models: Vec<Vec<(f64, f64, f64)>>,
    w: i32,
    h: i32,
    /// Maximum components per pixel.
    k: usize,
    /// Learning rate.
    alpha: f64,
    /// Squared Mahalanobis threshold for a match; upstream's default is 16.
    var_threshold: f64,
    /// Fraction of weight that counts as background.
    background_ratio: f64,
    /// Initial variance for a fresh component.
    initial_variance: f64,
}

impl BackgroundSubtractorMog2 {
    /// `cv::createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)`.
    pub fn new(width: i32, height: i32, history: usize, var_threshold: f64) -> Result<Self> {
        cv_assert!(width > 0 && height > 0, "the frame size must be positive");
        cv_assert!(history > 0, "history must be positive");
        cv_assert!(var_threshold > 0.0, "varThreshold must be positive");
        Ok(BackgroundSubtractorMog2 {
            models: vec![Vec::new(); (width * height) as usize],
            w: width,
            h: height,
            k: 5,
            alpha: 1.0 / history as f64,
            var_threshold,
            background_ratio: 0.9,
            initial_variance: 225.0,
        })
    }

    /// The learning rate; see [`crate::algorithms`] for the `getHistory` view.
    pub(crate) fn alpha(&self) -> f64 {
        self.alpha
    }
    pub(crate) fn set_alpha(&mut self, v: f64) {
        self.alpha = v;
    }
    pub(crate) fn mixtures(&self) -> usize {
        self.k
    }
    pub(crate) fn set_mixtures(&mut self, v: usize) {
        self.k = v;
    }
    pub(crate) fn var_threshold(&self) -> f64 {
        self.var_threshold
    }
    pub(crate) fn set_var_threshold_value(&mut self, v: f64) {
        self.var_threshold = v;
    }
    pub(crate) fn background_ratio(&self) -> f64 {
        self.background_ratio
    }
    pub(crate) fn set_background_ratio_value(&mut self, v: f64) {
        self.background_ratio = v;
    }
    pub(crate) fn initial_variance(&self) -> f64 {
        self.initial_variance
    }
    pub(crate) fn set_initial_variance(&mut self, v: f64) {
        self.initial_variance = v;
    }

    /// `cv::BackgroundSubtractor::apply(image, fgmask, learningRate)`.
    ///
    /// Returns the foreground mask, `CV_8UC1` with 255 for foreground.
    pub fn apply<S: MatLike + ?Sized>(&mut self, frame: &S) -> Result<Mat> {
        cv_assert!(frame.channels() == 1, "MOG2 here is single-channel");
        cv_assert!(
            frame.cols() == self.w && frame.rows() == self.h,
            "frame is {}x{}, the model is {}x{}",
            frame.cols(),
            frame.rows(),
            self.w,
            self.h
        );
        let mut mask = Mat::zeros(self.h, self.w, CV_8UC1)?;
        for y in 0..self.h {
            for x in 0..self.w {
                let v = frame.at_f64(y, x, 0)?;
                let i = (y * self.w + x) as usize;
                let m = &mut self.models[i];

                // Which component does this sample belong to?
                let mut matched = None;
                for (ci, c) in m.iter().enumerate() {
                    let d = v - c.1;
                    if d * d < self.var_threshold * c.2 {
                        matched = Some(ci);
                        break;
                    }
                }
                match matched {
                    Some(ci) => {
                        let rho = self.alpha / m[ci].0.max(self.alpha);
                        let d = v - m[ci].1;
                        m[ci].1 += rho * d;
                        m[ci].2 = (m[ci].2 + rho * (d * d - m[ci].2)).clamp(4.0, 5000.0);
                        for (cj, c) in m.iter_mut().enumerate() {
                            c.0 += self.alpha * (f64::from(u8::from(cj == ci)) - c.0);
                        }
                    }
                    None => {
                        // A new mode. Replace the weakest if the mixture is full.
                        if m.len() < self.k {
                            m.push((self.alpha, v, self.initial_variance));
                        } else {
                            let worst = m
                                .iter()
                                .enumerate()
                                .min_by(|a, b| {
                                    a.1 .0.partial_cmp(&b.1 .0).unwrap_or(core::cmp::Ordering::Equal)
                                })
                                .map_or(0, |(j, _)| j);
                            m[worst] = (self.alpha, v, self.initial_variance);
                        }
                        for c in m.iter_mut() {
                            c.0 *= 1.0 - self.alpha;
                        }
                        if let Some(last) = m.last_mut() {
                            last.0 = self.alpha;
                        }
                    }
                }
                // Renormalise and sort by weight, strongest first.
                let total: f64 = m.iter().map(|c| c.0).sum();
                if total > 0.0 {
                    for c in m.iter_mut() {
                        c.0 /= total;
                    }
                }
                m.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));

                // Foreground when the matching component is outside the
                // background set — the leading components summing to
                // `background_ratio`.
                let fg = match matched {
                    None => true,
                    Some(_) => {
                        let mut acc = 0.0;
                        let mut bg = false;
                        for c in m.iter() {
                            let d = v - c.1;
                            if d * d < self.var_threshold * c.2 {
                                bg = true;
                                break;
                            }
                            acc += c.0;
                            if acc > self.background_ratio {
                                break;
                            }
                        }
                        !bg
                    }
                };
                mask.pixel_mut::<u8>(y, x)?[0] = if fg { 255 } else { 0 };
            }
        }
        Ok(mask)
    }

    /// `cv::BackgroundSubtractor::getBackgroundImage(backgroundImage)`.
    pub fn background_image(&self) -> Result<Mat> {
        let mut out = Mat::zeros(self.h, self.w, CV_8UC1)?;
        for y in 0..self.h {
            for x in 0..self.w {
                let m = &self.models[(y * self.w + x) as usize];
                let v = m.first().map_or(0.0, |c| c.1);
                out.pixel_mut::<u8>(y, x)?[0] = cv_core::saturate::Elem::from_f64(v);
            }
        }
        Ok(out)
    }
}

/// `cv::BackgroundSubtractorKNN`.
///
/// Keeps the last `history` samples per pixel and calls a pixel background
/// when enough of them are close to it. Non-parametric, so it copes with a
/// background distribution no Gaussian describes.
#[derive(Debug, Clone)]
pub struct BackgroundSubtractorKnn {
    samples: Vec<Vec<f64>>,
    w: i32,
    h: i32,
    history: usize,
    dist2: f64,
    /// How many neighbours must be close for a pixel to count as background.
    k: usize,
    next: usize,
}

impl BackgroundSubtractorKnn {
    /// `cv::createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows)`.
    pub fn new(width: i32, height: i32, history: usize, dist2_threshold: f64) -> Result<Self> {
        cv_assert!(width > 0 && height > 0, "the frame size must be positive");
        cv_assert!(history >= 2, "KNN needs a history of at least 2");
        cv_assert!(dist2_threshold > 0.0, "dist2Threshold must be positive");
        Ok(BackgroundSubtractorKnn {
            samples: vec![Vec::new(); (width * height) as usize],
            w: width,
            h: height,
            history,
            dist2: dist2_threshold,
            k: 3,
            next: 0,
        })
    }

    pub(crate) fn history_len(&self) -> usize {
        self.history
    }
    pub(crate) fn set_history_len(&mut self, v: usize) {
        self.history = v;
    }
    pub(crate) fn dist2_threshold(&self) -> f64 {
        self.dist2
    }
    pub(crate) fn set_dist2_threshold_value(&mut self, v: f64) {
        self.dist2 = v;
    }
    pub(crate) fn knn(&self) -> usize {
        self.k
    }
    pub(crate) fn set_knn(&mut self, v: usize) {
        self.k = v;
    }

    /// `cv::BackgroundSubtractor::apply`.
    pub fn apply<S: MatLike + ?Sized>(&mut self, frame: &S) -> Result<Mat> {
        cv_assert!(frame.channels() == 1, "KNN here is single-channel");
        cv_assert!(
            frame.cols() == self.w && frame.rows() == self.h,
            "frame size does not match the model"
        );
        let mut mask = Mat::zeros(self.h, self.w, CV_8UC1)?;
        for y in 0..self.h {
            for x in 0..self.w {
                let v = frame.at_f64(y, x, 0)?;
                let i = (y * self.w + x) as usize;
                let close = self.samples[i].iter().filter(|s| (**s - v).powi(2) < self.dist2).count();
                // Until the history fills, everything is background: calling a
                // pixel foreground on frame one would mark the whole image.
                let fg = self.samples[i].len() >= self.k && close < self.k;
                mask.pixel_mut::<u8>(y, x)?[0] = if fg { 255 } else { 0 };

                if self.samples[i].len() < self.history {
                    self.samples[i].push(v);
                } else {
                    let slot = self.next % self.history;
                    self.samples[i][slot] = v;
                }
            }
        }
        self.next += 1;
        Ok(mask)
    }
}

/// `cv::meanShift(probImage, window, criteria)`.
///
/// Slides the window to the centroid of the probability image beneath it until
/// it stops moving. Returns the converged window and the iteration count.
pub fn mean_shift<S: MatLike + ?Sized>(
    prob: &S,
    mut window: Rect,
    max_iter: usize,
    epsilon: f64,
) -> Result<(Rect, usize)> {
    cv_assert!(prob.channels() == 1, "meanShift takes a single-channel probability image");
    cv_assert!(window.width > 0 && window.height > 0, "the window must be non-empty");
    let (w, h) = (prob.cols(), prob.rows());
    for it in 0..max_iter.max(1) {
        let (mut m00, mut m10, mut m01) = (0.0f64, 0.0f64, 0.0f64);
        for y in window.y.max(0)..(window.y + window.height).min(h) {
            for x in window.x.max(0)..(window.x + window.width).min(w) {
                let p = prob.at_f64(y, x, 0)?;
                m00 += p;
                m10 += p * f64::from(x);
                m01 += p * f64::from(y);
            }
        }
        if m00 <= 0.0 {
            return Ok((window, it));
        }
        let (cx, cy) = (m10 / m00, m01 / m00);
        let nx = (cx - f64::from(window.width) * 0.5).round() as i32;
        let ny = (cy - f64::from(window.height) * 0.5).round() as i32;
        // Keep the window inside the image, as upstream does.
        let nx = nx.clamp(0, (w - window.width).max(0));
        let ny = ny.clamp(0, (h - window.height).max(0));
        let moved = f64::from((nx - window.x).abs() + (ny - window.y).abs());
        window.x = nx;
        window.y = ny;
        if moved <= epsilon {
            return Ok((window, it + 1));
        }
    }
    Ok((window, max_iter))
}

/// A rotated rectangle, as `cv::CamShift` returns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackBox {
    /// Centre.
    pub center: (f64, f64),
    /// Size before rotation.
    pub size: (f64, f64),
    /// Rotation in degrees.
    pub angle: f64,
}

/// `cv::CamShift(probImage, window, criteria)`.
///
/// Mean shift, then the window is resized and oriented from the second moments
/// of the region it converged on — so the box tracks a target that changes
/// scale, which plain mean shift cannot.
pub fn cam_shift<S: MatLike + ?Sized>(
    prob: &S,
    window: Rect,
    max_iter: usize,
    epsilon: f64,
) -> Result<(TrackBox, Rect)> {
    let (mut win, _) = mean_shift(prob, window, max_iter, epsilon)?;
    let (w, h) = (prob.cols(), prob.rows());

    // Upstream widens the settled window by ten pixels on every side before
    // measuring it. That is not a detail: the moments are what set the box's
    // size, and measuring them over the tracking window alone would clip the
    // target's tails and shrink the box a little more on every frame.
    const TOLERANCE: i32 = 10;
    win.x = (win.x - TOLERANCE).max(0);
    win.y = (win.y - TOLERANCE).max(0);
    win.width = (win.width + 2 * TOLERANCE).min(w - win.x);
    win.height = (win.height + 2 * TOLERANCE).min(h - win.y);

    // Moments of the *sub-image*, so the coordinates are window-relative and
    // the window's origin is added back to the centre afterwards.
    let (mut m00, mut m10, mut m01, mut m20, mut m11, mut m02) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in 0..win.height {
        for x in 0..win.width {
            let p = prob.at_f64(win.y + y, win.x + x, 0)?;
            let (fx, fy) = (f64::from(x), f64::from(y));
            m00 += p;
            m10 += p * fx;
            m01 += p * fy;
            m20 += p * fx * fx;
            m11 += p * fx * fy;
            m02 += p * fy * fy;
        }
    }
    if m00.abs() < f64::EPSILON {
        return Ok((TrackBox { center: (0.0, 0.0), size: (0.0, 0.0), angle: 0.0 }, win));
    }
    let inv = 1.0 / m00;
    let xc = cv_core::saturate::cv_round(m10 * inv + f64::from(win.x));
    let yc = cv_core::saturate::cv_round(m01 * inv + f64::from(win.y));
    // Central moments.
    let mu20 = m20 - m10 * m10 * inv;
    let mu11 = m11 - m10 * m01 * inv;
    let mu02 = m02 - m01 * m01 * inv;
    let (a, b, c) = (mu20 * inv, mu11 * inv, mu02 * inv);

    let square = (4.0 * b * b + (a - c) * (a - c)).sqrt();
    let mut theta = (2.0 * b).atan2(a - c + square);
    let (mut cs, mut sn) = (theta.cos(), theta.sin());
    // The axis lengths come from *rotating* the second-moment matrix into the
    // box's own frame, not from its eigenvalues directly. The two agree in
    // exact arithmetic and not in floating point, and upstream's is the one a
    // caller's numbers were fitted to.
    let rot_a = (cs * cs * mu20 + 2.0 * cs * sn * mu11 + sn * sn * mu02).max(0.0);
    let rot_c = (sn * sn * mu20 - 2.0 * cs * sn * mu11 + cs * cs * mu02).max(0.0);
    let mut length = (rot_a * inv).sqrt() * 4.0;
    let mut width = (rot_c * inv).sqrt() * 4.0;
    // At theta near 0 or pi/2 the two can come out the wrong way round.
    if length < width {
        core::mem::swap(&mut length, &mut width);
        core::mem::swap(&mut cs, &mut sn);
        theta = core::f64::consts::FRAC_PI_2 - theta;
    }

    let round = cv_core::saturate::cv_round;
    let t0 = round((length * cs).abs()).max(round((width * sn).abs())) + 2;
    let ww = t0.min((w - xc) * 2);
    let t0 = round((length * sn).abs()).max(round((width * cs).abs())) + 2;
    let hh = t0.min((h - yc) * 2);
    let wx = (xc - ww / 2).max(0);
    let wy = (yc - hh / 2).max(0);
    let out = Rect::new(wx, wy, (w - wx).min(ww), (h - wy).min(hh));

    let mut angle = (core::f64::consts::FRAC_PI_2 + theta).to_degrees();
    while angle < 0.0 {
        angle += 360.0;
    }
    while angle >= 360.0 {
        angle -= 360.0;
    }
    if angle >= 180.0 {
        angle -= 180.0;
    }
    // The box's centre is the *window's* centre, not the centroid — upstream
    // takes it from the rectangle it just built.
    Ok((
        TrackBox {
            center: (
                f64::from(out.x) + f64::from(out.width) * 0.5,
                f64::from(out.y) + f64::from(out.height) * 0.5,
            ),
            size: (width, length),
            angle,
        },
        out,
    ))
}

/// `cv::KalmanFilter`.
///
/// Constant-velocity by default, but every matrix is public so a caller can
/// impose a different model.
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    /// State dimension.
    pub nd: usize,
    /// Measurement dimension.
    pub nm: usize,
    /// State estimate.
    pub state: Vec<f64>,
    /// Transition matrix `A`, `nd x nd`.
    pub transition: Vec<f64>,
    /// Measurement matrix `H`, `nm x nd`.
    pub measurement: Vec<f64>,
    /// Process noise covariance `Q`, `nd x nd`.
    pub process_noise: Vec<f64>,
    /// Measurement noise covariance `R`, `nm x nm`.
    pub measurement_noise: Vec<f64>,
    /// Error covariance `P`, `nd x nd`.
    pub error_cov: Vec<f64>,
}

impl KalmanFilter {
    /// `cv::KalmanFilter::KalmanFilter(dynamParams, measureParams, controlParams)`.
    ///
    /// Initialised to a constant-velocity model when `nd == 2*nm`: the state is
    /// `[position..., velocity...]` and only the positions are measured.
    pub fn new(nd: usize, nm: usize) -> Result<Self> {
        cv_assert!(nd > 0 && nm > 0, "both dimensions must be positive");
        cv_assert!(nm <= nd, "there cannot be more measurements than states");
        let mut a = vec![0.0; nd * nd];
        for i in 0..nd {
            a[i * nd + i] = 1.0;
        }
        // Position gains velocity each step.
        if nd == 2 * nm {
            for i in 0..nm {
                a[i * nd + (nm + i)] = 1.0;
            }
        }
        let mut hm = vec![0.0; nm * nd];
        for i in 0..nm {
            hm[i * nd + i] = 1.0;
        }
        let mut q = vec![0.0; nd * nd];
        for i in 0..nd {
            q[i * nd + i] = 1e-4;
        }
        let mut r = vec![0.0; nm * nm];
        for i in 0..nm {
            r[i * nm + i] = 1e-1;
        }
        let mut p = vec![0.0; nd * nd];
        for i in 0..nd {
            p[i * nd + i] = 1.0;
        }
        Ok(KalmanFilter {
            nd,
            nm,
            state: vec![0.0; nd],
            transition: a,
            measurement: hm,
            process_noise: q,
            measurement_noise: r,
            error_cov: p,
        })
    }

    /// `cv::KalmanFilter::predict(control)` — advance the state one step.
    pub fn predict(&mut self) -> Vec<f64> {
        let (nd, a) = (self.nd, &self.transition);
        // x = A x
        let nx: Vec<f64> = (0..nd)
            .map(|i| (0..nd).map(|j| a[i * nd + j] * self.state[j]).sum())
            .collect();
        // P = A P A^T + Q
        let ap: Vec<f64> = (0..nd)
            .flat_map(|i| {
                (0..nd)
                    .map(|j| (0..nd).map(|k| a[i * nd + k] * self.error_cov[k * nd + j]).sum::<f64>())
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut np = vec![0.0f64; nd * nd];
        for i in 0..nd {
            for j in 0..nd {
                np[i * nd + j] = (0..nd).map(|k| ap[i * nd + k] * a[j * nd + k]).sum::<f64>()
                    + self.process_noise[i * nd + j];
            }
        }
        self.state = nx.clone();
        self.error_cov = np;
        nx
    }

    /// `cv::KalmanFilter::correct(measurement)` — fold in an observation.
    pub fn correct(&mut self, z: &[f64]) -> Result<Vec<f64>> {
        cv_assert!(
            z.len() == self.nm,
            "measurement has {} values, expected {}",
            z.len(),
            self.nm
        );
        let (nd, nm) = (self.nd, self.nm);
        let h = &self.measurement;
        // S = H P H^T + R
        let ph: Vec<f64> = (0..nd)
            .flat_map(|i| {
                (0..nm)
                    .map(|j| (0..nd).map(|k| self.error_cov[i * nd + k] * h[j * nd + k]).sum::<f64>())
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut s = vec![0.0f64; nm * nm];
        for i in 0..nm {
            for j in 0..nm {
                s[i * nm + j] = (0..nd).map(|k| h[i * nd + k] * ph[k * nm + j]).sum::<f64>()
                    + self.measurement_noise[i * nm + j];
            }
        }
        let si = invert(&s, nm).ok_or_else(|| {
            Error::new(Code::StsError, "the innovation covariance is singular".to_string())
        })?;
        // K = P H^T S^-1
        let mut k = vec![0.0f64; nd * nm];
        for i in 0..nd {
            for j in 0..nm {
                k[i * nm + j] = (0..nm).map(|t| ph[i * nm + t] * si[t * nm + j]).sum();
            }
        }
        // x = x + K (z - H x)
        let hx: Vec<f64> = (0..nm)
            .map(|i| (0..nd).map(|j| h[i * nd + j] * self.state[j]).sum())
            .collect();
        let y: Vec<f64> = z.iter().zip(&hx).map(|(a, b)| a - b).collect();
        for i in 0..nd {
            self.state[i] += (0..nm).map(|j| k[i * nm + j] * y[j]).sum::<f64>();
        }
        // P = (I - K H) P
        let mut kh = vec![0.0f64; nd * nd];
        for i in 0..nd {
            for j in 0..nd {
                kh[i * nd + j] = (0..nm).map(|t| k[i * nm + t] * h[t * nd + j]).sum();
            }
        }
        let mut np = vec![0.0f64; nd * nd];
        for i in 0..nd {
            for j in 0..nd {
                let ikh: Vec<f64> = (0..nd)
                    .map(|t| f64::from(u8::from(i == t)) - kh[i * nd + t])
                    .collect();
                np[i * nd + j] = (0..nd).map(|t| ikh[t] * self.error_cov[t * nd + j]).sum();
            }
        }
        self.error_cov = np;
        Ok(self.state.clone())
    }
}

/// Gauss-Jordan inverse of a small dense matrix.
fn invert(m: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = m.to_vec();
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for c in 0..n {
        let mut piv = c;
        for r in c + 1..n {
            if a[r * n + c].abs() > a[piv * n + c].abs() {
                piv = r;
            }
        }
        if a[piv * n + c].abs() < 1e-300 {
            return None;
        }
        for k in 0..n {
            a.swap(c * n + k, piv * n + k);
            inv.swap(c * n + k, piv * n + k);
        }
        let d = a[c * n + c];
        for k in 0..n {
            a[c * n + k] /= d;
            inv[c * n + k] /= d;
        }
        for r in 0..n {
            if r == c {
                continue;
            }
            let f = a[r * n + c];
            if f == 0.0 {
                continue;
            }
            for k in 0..n {
                a[r * n + k] -= f * a[c * n + k];
                inv[r * n + k] -= f * inv[c * n + k];
            }
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A textured field, shifted by `(dx, dy)`.
    fn shifted(w: i32, h: i32, dx: i32, dy: i32) -> Mat {
        let mut m = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = (x - dx, y - dy);
                // A smooth but non-periodic pattern, so tracking is unambiguous.
                let v = 128.0
                    + 100.0 * ((f64::from(sx) * 0.21).sin() * (f64::from(sy) * 0.17).cos());
                m.pixel_mut::<u8>(y, x).unwrap()[0] = cv_core::saturate::Elem::from_f64(v);
            }
        }
        m
    }

    #[test]
    fn pyramid_halves_each_level() {
        let img = shifted(64, 48, 0, 0);
        let p = build_optical_flow_pyramid(&img, 3).unwrap();
        assert_eq!(p.len(), 4);
        assert_eq!((p[0].cols(), p[0].rows()), (64, 48));
        assert_eq!((p[1].cols(), p[1].rows()), (32, 24));
        assert_eq!((p[3].cols(), p[3].rows()), (8, 6));
    }

    #[test]
    fn lucas_kanade_tracks_a_small_shift() {
        let a = shifted(80, 60, 0, 0);
        let b = shifted(80, 60, 2, 1);
        let pts = vec![(30.0, 25.0), (45.0, 35.0), (20.0, 40.0)];
        let r = calc_optical_flow_pyr_lk(&a, &b, &pts, 15, 2, 30, 0.01).unwrap();
        assert_eq!(r.len(), 3);
        for (i, ((nx, ny), ok, _)) in r.iter().enumerate() {
            assert!(ok, "point {i} was dropped");
            assert!((nx - pts[i].0 - 2.0).abs() < 0.5, "x moved to {nx}, expected {}", pts[i].0 + 2.0);
            assert!((ny - pts[i].1 - 1.0).abs() < 0.5, "y moved to {ny}, expected {}", pts[i].1 + 1.0);
        }
    }

    #[test]
    fn the_pyramid_is_what_catches_a_large_shift() {
        // An 8-pixel motion is far beyond what a single-level Taylor expansion
        // can find; the coarse level sees it as one pixel.
        let a = shifted(96, 96, 0, 0);
        let b = shifted(96, 96, 8, 0);
        let pts = vec![(48.0, 48.0)];
        let flat = calc_optical_flow_pyr_lk(&a, &b, &pts, 15, 0, 30, 0.01).unwrap();
        let pyr = calc_optical_flow_pyr_lk(&a, &b, &pts, 15, 3, 30, 0.01).unwrap();
        let err = |r: &[TrackedPoint]| (r[0].0 .0 - 56.0).abs();
        assert!(
            err(&pyr) < err(&flat),
            "the pyramid should win: {} vs {}",
            err(&pyr),
            err(&flat)
        );
        assert!(err(&pyr) < 1.0, "pyramidal LK missed an 8px shift by {}", err(&pyr));
    }

    #[test]
    fn zero_motion_is_reported_as_zero() {
        let a = shifted(64, 64, 0, 0);
        let pts = vec![(32.0, 32.0)];
        let r = calc_optical_flow_pyr_lk(&a, &a, &pts, 15, 2, 30, 0.01).unwrap();
        assert!(r[0].1);
        assert!((r[0].0 .0 - 32.0).abs() < 1e-3 && (r[0].0 .1 - 32.0).abs() < 1e-3);
        assert!(r[0].2 < 1e-6, "residual should vanish for identical frames");
    }

    #[test]
    fn a_featureless_window_is_dropped_not_guessed() {
        // A flat image has a singular structure tensor: the aperture problem.
        // Returning some arbitrary motion would be worse than refusing.
        let flat = Mat::from_vec(40, 40, 1, &[128u8; 1600]).unwrap();
        let r = calc_optical_flow_pyr_lk(&flat, &flat, &[(20.0, 20.0)], 15, 1, 10, 0.01).unwrap();
        assert!(!r[0].1, "a flat window must be reported as failed");
    }

    #[test]
    fn optical_flow_validates_its_arguments() {
        let a = shifted(32, 32, 0, 0);
        let small = shifted(16, 16, 0, 0);
        assert!(calc_optical_flow_pyr_lk(&a, &small, &[(1.0, 1.0)], 15, 1, 10, 0.01).is_err());
        assert!(calc_optical_flow_pyr_lk(&a, &a, &[(1.0, 1.0)], 14, 1, 10, 0.01).is_err());
    }

    #[test]
    fn farneback_produces_a_dense_field_pointing_the_right_way() {
        let a = shifted(64, 64, 0, 0);
        let b = shifted(64, 64, 3, 0);
        let f = calc_optical_flow_farneback(&a, &b, 3, 15, 10).unwrap();
        assert_eq!((f.rows(), f.cols(), f.channels()), (64, 64, 2));
        // Average the interior, away from the border where the shift is undefined.
        let (mut sx, mut sy, mut n) = (0.0, 0.0, 0);
        for y in 20..44 {
            for x in 20..44 {
                sx += f.at_f64(y, x, 0).unwrap();
                sy += f.at_f64(y, x, 1).unwrap();
                n += 1;
            }
        }
        let (mx, my) = (sx / f64::from(n), sy / f64::from(n));
        assert!(mx > 1.5 && mx < 4.5, "mean dx {mx}, expected about 3");
        assert!(my.abs() < 1.0, "mean dy {my}, expected about 0");
    }

    #[test]
    fn mog2_learns_a_static_background_then_flags_an_intruder() {
        let bg = shifted(32, 32, 0, 0);
        let mut sub = BackgroundSubtractorMog2::new(32, 32, 20, 16.0).unwrap();
        for _ in 0..40 {
            sub.apply(&bg).unwrap();
        }
        // The settled background is almost entirely quiet.
        let quiet = sub.apply(&bg).unwrap();
        let fg = quiet.to_vec::<u8>().unwrap().iter().filter(|&&v| v > 0).count();
        assert!(fg < 30, "a learned background should be quiet, got {fg} foreground pixels");

        // Paint a bright square and it should light up.
        let mut with = Mat::copy_from(&bg).unwrap();
        for y in 10..20 {
            for x in 10..20 {
                with.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        let mask = sub.apply(&with).unwrap();
        let inside = (10..20)
            .flat_map(|y| (10..20).map(move |x| (y, x)))
            .filter(|&(y, x)| mask.at_f64(y, x, 0).unwrap() > 0.0)
            .count();
        assert!(inside > 70, "the intruder should be detected, got {inside}/100");
    }

    #[test]
    fn mog2_can_report_its_background() {
        let bg = shifted(24, 24, 0, 0);
        let mut sub = BackgroundSubtractorMog2::new(24, 24, 10, 16.0).unwrap();
        for _ in 0..30 {
            sub.apply(&bg).unwrap();
        }
        let est = sub.background_image().unwrap();
        let mut worst = 0.0f64;
        for y in 0..24 {
            for x in 0..24 {
                worst = worst.max((est.at_f64(y, x, 0).unwrap() - bg.at_f64(y, x, 0).unwrap()).abs());
            }
        }
        assert!(worst < 12.0, "the recovered background is off by {worst}");
    }

    #[test]
    fn mog2_validates_its_frames() {
        let mut sub = BackgroundSubtractorMog2::new(16, 16, 10, 16.0).unwrap();
        let wrong = shifted(20, 20, 0, 0);
        assert!(sub.apply(&wrong).is_err());
        assert!(BackgroundSubtractorMog2::new(0, 16, 10, 16.0).is_err());
        assert!(BackgroundSubtractorMog2::new(16, 16, 0, 16.0).is_err());
    }

    #[test]
    fn knn_flags_a_change_after_its_history_fills() {
        let bg = shifted(24, 24, 0, 0);
        let mut sub = BackgroundSubtractorKnn::new(24, 24, 10, 400.0).unwrap();
        // The first frames must not paint everything foreground.
        let first = sub.apply(&bg).unwrap();
        assert_eq!(
            first.to_vec::<u8>().unwrap().iter().filter(|&&v| v > 0).count(),
            0,
            "an unfilled history cannot decide anything"
        );
        for _ in 0..15 {
            sub.apply(&bg).unwrap();
        }
        let mut with = Mat::copy_from(&bg).unwrap();
        for y in 5..15 {
            for x in 5..15 {
                with.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        let mask = sub.apply(&with).unwrap();
        let inside = (5..15)
            .flat_map(|y| (5..15).map(move |x| (y, x)))
            .filter(|&(y, x)| mask.at_f64(y, x, 0).unwrap() > 0.0)
            .count();
        assert!(inside > 80, "KNN should flag the square, got {inside}/100");
    }

    /// A probability image with a bright blob at `(cx, cy)`.
    fn blob(w: i32, h: i32, cx: i32, cy: i32, r: i32) -> Mat {
        let mut m = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                let d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                if d2 <= r * r {
                    m.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
                }
            }
        }
        m
    }

    #[test]
    fn mean_shift_walks_the_window_onto_the_blob() {
        let p = blob(64, 64, 45, 20, 8);
        // Start well off-centre but overlapping.
        let (win, iters) = mean_shift(&p, Rect::new(30, 10, 20, 20), 20, 1.0).unwrap();
        let (cx, cy) = (win.x + win.width / 2, win.y + win.height / 2);
        assert!((cx - 45).abs() <= 2 && (cy - 20).abs() <= 2, "converged to ({cx}, {cy})");
        assert!(iters > 0 && iters <= 20);
    }

    #[test]
    fn mean_shift_on_an_empty_region_stays_put() {
        let p = Mat::zeros(32, 32, CV_8UC1).unwrap();
        let start = Rect::new(4, 4, 8, 8);
        let (win, _) = mean_shift(&p, start, 10, 0.5).unwrap();
        assert_eq!(win, start, "with no probability mass there is nowhere to go");
    }

    #[test]
    fn camshift_reports_a_box_sized_to_the_target() {
        let p = blob(80, 80, 40, 40, 12);
        let (tb, win) = cam_shift(&p, Rect::new(28, 28, 24, 24), 20, 1.0).unwrap();
        assert!((tb.center.0 - 40.0).abs() < 2.0, "cx {}", tb.center.0);
        assert!((tb.center.1 - 40.0).abs() < 2.0, "cy {}", tb.center.1);
        // A disc of radius 12 gives a box on that order, not a point or the window.
        let m = tb.size.0.max(tb.size.1);
        assert!((10.0..40.0).contains(&m), "box size {:?}", tb.size);
        assert!(win.width > 0);
    }

    #[test]
    fn kalman_smooths_a_noisy_constant_velocity_track() {
        // State [x, y, vx, vy], measuring [x, y].
        let mut kf = KalmanFilter::new(4, 2).unwrap();
        kf.state = vec![0.0, 0.0, 0.0, 0.0];
        let mut rng = cv_core::rng::Rng::new(5);
        let (mut raw_err, mut filt_err) = (0.0f64, 0.0f64);
        for t in 0..60 {
            let truth = (f64::from(t) * 1.5, f64::from(t) * -0.75);
            let z = vec![
                truth.0 + rng.uniform_f64(-3.0, 3.0),
                truth.1 + rng.uniform_f64(-3.0, 3.0),
            ];
            kf.predict();
            let x = kf.correct(&z).unwrap();
            if t > 20 {
                raw_err += (z[0] - truth.0).abs() + (z[1] - truth.1).abs();
                filt_err += (x[0] - truth.0).abs() + (x[1] - truth.1).abs();
            }
        }
        assert!(
            filt_err < raw_err,
            "the filter should beat the raw measurements: {filt_err:.2} vs {raw_err:.2}"
        );
    }

    #[test]
    fn kalman_predicts_forward_with_constant_velocity() {
        let mut kf = KalmanFilter::new(4, 2).unwrap();
        kf.state = vec![10.0, 20.0, 2.0, -1.0];
        let p = kf.predict();
        assert!((p[0] - 12.0).abs() < 1e-9, "x should advance by vx");
        assert!((p[1] - 19.0).abs() < 1e-9, "y should advance by vy");
        assert!((p[2] - 2.0).abs() < 1e-9, "velocity is unchanged");
    }

    #[test]
    fn kalman_validates_its_shapes() {
        assert!(KalmanFilter::new(0, 1).is_err());
        assert!(KalmanFilter::new(2, 4).is_err(), "more measurements than states");
        let mut kf = KalmanFilter::new(4, 2).unwrap();
        assert!(kf.correct(&[1.0]).is_err());
    }
}
