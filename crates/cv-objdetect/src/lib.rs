//! `cv::objdetect` for `opencv-no-c` — HOG, cascades and detection grouping.
//!
//! # Scope
//!
//! `HOGDescriptor::compute` and `detectMultiScale` with a linear classifier,
//! `groupRectangles`, an integral-image Haar cascade evaluator, and QR finder
//! pattern location.
//!
//! Full QR *decoding* — format/version bits, Reed-Solomon error correction,
//! de-interleaving — is not here. Finding the symbol is the computer-vision
//! half; decoding it is a coding-theory library that happens to live in the
//! same OpenCV module.

#![warn(missing_docs)]

pub mod constants;
pub mod aruco;
pub mod aruco_data;
pub mod boards;
pub mod codes;

use cv_core::{cv_assert, Mat, MatLike, Rect, Result};

/// Configuration of a [`HogDescriptor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HogParams {
    /// Detection window, in pixels.
    pub win: (i32, i32),
    /// Block size, in pixels.
    pub block: (i32, i32),
    /// Block stride, in pixels.
    pub block_stride: (i32, i32),
    /// Cell size, in pixels.
    pub cell: (i32, i32),
    /// Orientation bins over 0..180 degrees.
    pub nbins: usize,
}

impl Default for HogParams {
    /// Upstream's defaults: a 64x128 window, 16x16 blocks, 8x8 cells, 9 bins.
    fn default() -> Self {
        HogParams {
            win: (64, 128),
            block: (16, 16),
            block_stride: (8, 8),
            cell: (8, 8),
            nbins: 9,
        }
    }
}

impl HogParams {
    /// Length of the descriptor this configuration produces.
    ///
    /// `cv::HOGDescriptor::getDescriptorSize()`.
    #[must_use]
    pub fn descriptor_size(&self) -> usize {
        let bx = (self.win.0 - self.block.0) / self.block_stride.0 + 1;
        let by = (self.win.1 - self.block.1) / self.block_stride.1 + 1;
        let cx = self.block.0 / self.cell.0;
        let cy = self.block.1 / self.cell.1;
        (bx * by * cx * cy) as usize * self.nbins
    }

    /// Whether the sizes divide cleanly, which upstream also requires.
    fn validate(&self) -> Result<()> {
        cv_assert!(self.nbins > 0, "nbins must be positive");
        cv_assert!(self.cell.0 > 0 && self.cell.1 > 0, "the cell size must be positive");
        cv_assert!(
            self.block.0 % self.cell.0 == 0 && self.block.1 % self.cell.1 == 0,
            "the block must be a whole number of cells"
        );
        cv_assert!(
            self.block_stride.0 > 0 && self.block_stride.1 > 0,
            "the block stride must be positive"
        );
        cv_assert!(
            (self.win.0 - self.block.0) % self.block_stride.0 == 0
                && (self.win.1 - self.block.1) % self.block_stride.1 == 0,
            "the window must be a whole number of block strides"
        );
        Ok(())
    }
}

/// `cv::HOGDescriptor`.
#[derive(Debug, Clone)]
pub struct HogDescriptor {
    /// Geometry.
    pub params: HogParams,
    /// The linear SVM's coefficients, empty until `setSVMDetector`.
    pub(crate) svm: Vec<f32>,
    /// `winSigma`; negative means "derive it from the block size".
    pub(crate) win_sigma: f64,
}

impl HogDescriptor {
    /// `cv::HOGDescriptor::HOGDescriptor(...)`.
    pub fn new(params: HogParams) -> Result<Self> {
        params.validate()?;
        Ok(HogDescriptor { params, svm: Vec::new(), win_sigma: -1.0 })
    }

    /// With upstream's default geometry.
    pub fn default_people() -> Result<Self> {
        Self::new(HogParams::default())
    }

    /// `cv::HOGDescriptor::compute(img, descriptors, winStride, padding, locations)`.
    ///
    /// The window is taken from `(x0, y0)`; the image must contain it.
    ///
    /// # Normalisation
    ///
    /// Blocks are L2-Hys normalised: L2, clip at 0.2, L2 again. The clip is
    /// what makes HOG robust to local contrast changes, and leaving it out
    /// gives a descriptor that still looks reasonable and matches badly.
    pub fn compute_at<S: MatLike + ?Sized>(&self, img: &S, x0: i32, y0: i32) -> Result<Vec<f32>> {
        cv_assert!(img.channels() == 1, "HOG here takes a single-channel image");
        let p = &self.params;
        cv_assert!(
            x0 >= 0 && y0 >= 0 && x0 + p.win.0 <= img.cols() && y0 + p.win.1 <= img.rows(),
            "the {}x{} window at ({x0}, {y0}) does not fit the image",
            p.win.0,
            p.win.1
        );

        // The gradient, as `computeGradient` builds it.
        //
        // Three things here are easy to leave out and all three move every
        // bin: the **gamma** LUT (`sqrt` of the intensity, on by default), the
        // **-0.5** offset that centres a bin on its nominal angle, and
        // reflect-101 at the image border rather than clamping.
        let (w, h) = (p.win.0, p.win.1);
        let lut: Vec<f32> = (0..256).map(|i| (i as f32).sqrt()).collect();
        let refl = |i: i32, n: i32| -> i32 {
            let mut i = i;
            loop {
                if i < 0 {
                    i = -i;
                } else if i >= n {
                    i = 2 * (n - 1) - i;
                } else {
                    return i;
                }
            }
        };
        let angle_scale = p.nbins as f32 / core::f32::consts::PI;
        let n = (w * h) as usize;
        let (mut g0, mut g1) = (vec![0.0f32; n], vec![0.0f32; n]);
        let (mut q0, mut q1) = (vec![0usize; n], vec![0usize; n]);
        for y in 0..h {
            for x in 0..w {
                let at = |dy: i32, dx: i32| -> Result<f32> {
                    let sy = refl(y0 + y + dy, img.rows());
                    let sx = refl(x0 + x + dx, img.cols());
                    Ok(lut[img.at_f64(sy, sx, 0)? as usize])
                };
                let dx = at(0, 1)? - at(0, -1)?;
                let dy = at(1, 0)? - at(-1, 0)?;
                let mag = dx.hypot(dy);
                // `cartToPolar` in radians, then scaled by `nbins / PI` — an
                // unsigned gradient folds the upper half turn onto the lower
                // by the wrap below, not by taking an absolute value.
                let mut a = dy.atan2(dx);
                if a < 0.0 {
                    a += 2.0 * core::f32::consts::PI;
                }
                let a = a * angle_scale - 0.5;
                let hidx = a.floor();
                let frac = a - hidx;
                let mut hidx = hidx as i32;
                let i = (y * w + x) as usize;
                g0[i] = mag * (1.0 - frac);
                g1[i] = mag * frac;
                if hidx < 0 {
                    hidx += p.nbins as i32;
                } else if hidx >= p.nbins as i32 {
                    hidx -= p.nbins as i32;
                }
                q0[i] = hidx as usize;
                q1[i] = if hidx + 1 < p.nbins as i32 { hidx as usize + 1 } else { 0 };
            }
        }

        // Per-block histograms with trilinear weights. The cells are *not*
        // binned once and shared: each block weights the pixels by a Gaussian
        // centred on itself and spreads each pixel over the four cells it lies
        // between, so a cell contributes differently to every block it belongs
        // to. Binning each cell once and reusing it is much simpler and gives
        // a different descriptor.
        let sigma = if self.win_sigma >= 0.0 {
            self.win_sigma
        } else {
            f64::from(p.block.0 + p.block.1) / 8.0
        } as f32;
        let gscale = 1.0 / (2.0 * sigma * sigma);
        let (bw, bh) = (p.block.0, p.block.1);
        let (ncx, ncy) = (bw / p.cell.0, bh / p.cell.1);
        let bx = (w - bw) / p.block_stride.0 + 1;
        let by = (h - bh) / p.block_stride.1 + 1;
        let mut out = Vec::with_capacity(self.params.descriptor_size());
        // Blocks come out **column-major**: upstream indexes `blockData` as
        // `j * nblocks.height + i` with `j` the x index, so the descriptor
        // walks down each column of blocks before moving right. Emitting them
        // row-major gives the same 3780 numbers in the wrong order, which a
        // trained SVM reads as noise.
        for bxi in 0..bx {
            for byi in 0..by {
                let (ox, oy) = (bxi * p.block_stride.0, byi * p.block_stride.1);
                let mut hist = vec![0.0f32; (ncx * ncy) as usize * p.nbins];
                // Upstream walks the block column-major and indexes cells as
                // `(cx * ncells.height + cy) * nbins`.
                for j in 0..bw {
                    for i2 in 0..bh {
                        let px = ox + j;
                        let py = oy + i2;
                        let src = (py * w + px) as usize;
                        let (di, dj) = (i2 as f32 - bh as f32 * 0.5, j as f32 - bw as f32 * 0.5);
                        let gw = (-(di * di + dj * dj) * gscale).exp();

                        let mut cell_x = (j as f32 + 0.5) / p.cell.0 as f32 - 0.5;
                        let mut cell_y = (i2 as f32 + 0.5) / p.cell.1 as f32 - 0.5;
                        let icx0 = cell_x.floor() as i32;
                        let icy0 = cell_y.floor() as i32;
                        cell_x -= icx0 as f32;
                        cell_y -= icy0 as f32;
                        let (icx1, icy1) = (icx0 + 1, icy0 + 1);

                        let mut add = |cx: i32, cy: i32, sw: f32| {
                            if cx < 0 || cy < 0 || cx >= ncx || cy >= ncy {
                                return;
                            }
                            let base = ((cx * ncy + cy) as usize) * p.nbins;
                            let wgt = gw * sw;
                            hist[base + q0[src]] += g0[src] * wgt;
                            hist[base + q1[src]] += g1[src] * wgt;
                        };
                        add(icx0, icy0, (1.0 - cell_x) * (1.0 - cell_y));
                        add(icx1, icy0, cell_x * (1.0 - cell_y));
                        add(icx0, icy1, (1.0 - cell_x) * cell_y);
                        add(icx1, icy1, cell_x * cell_y);
                    }
                }
                let mut block: Vec<f64> = hist.iter().map(|v| f64::from(*v)).collect();
                l2_hys(&mut block);
                out.extend(block.into_iter().map(|v| v as f32));
            }
        }
        Ok(out)
    }

    /// `cv::HOGDescriptor::detectMultiScale(img, foundLocations, ...)`.
    ///
    /// Slides the window at each scale and keeps positions where
    /// `weights . descriptor + bias` exceeds `threshold`. The classifier is
    /// the caller's; upstream ships a trained people detector, which is data
    /// this port does not carry.
    #[allow(clippy::too_many_arguments)]
    pub fn detect_multi_scale<S: MatLike + ?Sized>(
        &self,
        img: &S,
        weights: &[f32],
        bias: f32,
        threshold: f32,
        win_stride: (i32, i32),
        scale_step: f64,
        levels: usize,
    ) -> Result<Vec<(Rect, f32)>> {
        cv_assert!(
            weights.len() == self.params.descriptor_size(),
            "the classifier has {} weights but the descriptor is {} long",
            weights.len(),
            self.params.descriptor_size()
        );
        cv_assert!(scale_step > 1.0, "scaleStep must exceed 1, got {scale_step}");
        cv_assert!(win_stride.0 > 0 && win_stride.1 > 0, "winStride must be positive");

        let mut found = Vec::new();
        let mut scale = 1.0f64;
        for _ in 0..levels.max(1) {
            let (sw, sh) = (
                (img.cols() as f64 / scale).round() as i32,
                (img.rows() as f64 / scale).round() as i32,
            );
            if sw < self.params.win.0 || sh < self.params.win.1 {
                break;
            }
            let level: Mat = if scale == 1.0 {
                Mat::copy_from(img)?
            } else {
                cv_imgproc::resize(
                    img,
                    cv_core::Size::new(sw, sh),
                    0.0,
                    0.0,
                    cv_imgproc::Interpolation::Linear,
                )?
            };
            let mut y = 0;
            while y + self.params.win.1 <= sh {
                let mut x = 0;
                while x + self.params.win.0 <= sw {
                    let d = self.compute_at(&level, x, y)?;
                    let score: f32 =
                        d.iter().zip(weights).map(|(a, b)| a * b).sum::<f32>() + bias;
                    if score > threshold {
                        found.push((
                            Rect::new(
                                (x as f64 * scale).round() as i32,
                                (y as f64 * scale).round() as i32,
                                (self.params.win.0 as f64 * scale).round() as i32,
                                (self.params.win.1 as f64 * scale).round() as i32,
                            ),
                            score,
                        ));
                    }
                    x += win_stride.0;
                }
                y += win_stride.1;
            }
            scale *= scale_step;
        }
        Ok(found)
    }
}

/// L2-Hys normalisation, in place.
fn l2_hys(v: &mut [f64]) {
    let n = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    // The two epsilons are *different* and neither is decorative. The first is
    // `blockHistogramSize * 0.1` — 3.6 for a 36-bin block — which is large
    // enough to damp a block whose gradients are all tiny, and using a token
    // 1e-6 instead leaves every value a few percent high. The second really is
    // 1e-3.
    let e = n(v) + v.len() as f64 * 0.1;
    for x in v.iter_mut() {
        *x = (*x / e).min(0.2);
    }
    let e2 = n(v) + 1e-3;
    for x in v.iter_mut() {
        *x /= e2;
    }
}

/// `cv::groupRectangles(rectList, groupThreshold, eps)`.
///
/// `saturate_cast<int>(sum * (1.f / n))` — an `f32` reciprocal and half-to-even
/// rounding, as `groupRectangles` averages a cluster.
fn avg(values: impl Iterator<Item = i32>, n: f32) -> i32 {
    let sum: i32 = values.sum();
    let v = sum as f32 * (1.0f32 / n);
    let r = v.round();
    // `f32::round` is half-away-from-zero; correct the ties.
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        (r - v.signum()) as i32
    } else {
        r as i32
    }
}

/// Clusters overlapping detections and keeps a cluster only when it has more
/// than `group_threshold` members — the count is what suppresses one-off false
/// positives, so a threshold of 0 returns everything unclustered.
#[must_use]
pub fn group_rectangles(rects: &[Rect], group_threshold: usize, eps: f64) -> Vec<Rect> {
    if rects.is_empty() {
        return Vec::new();
    }
    if group_threshold == 0 {
        return rects.to_vec();
    }
    // Union-find over "similar enough" pairs.
    let n = rects.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    let similar = |a: &Rect, b: &Rect| -> bool {
        let d = eps * f64::from(a.width.min(b.width) + a.height.min(b.height)) * 0.5;
        f64::from((a.x - b.x).abs()) <= d
            && f64::from((a.y - b.y).abs()) <= d
            && f64::from((a.width - b.width).abs()) <= d
            && f64::from((a.height - b.height).abs()) <= d
    };
    for (i, ri) in rects.iter().enumerate() {
        for (j, rj) in rects.iter().enumerate().skip(i + 1) {
            if similar(ri, rj) {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[b.max(a)] = a.min(b);
                }
            }
        }
    }
    let mut groups: std::collections::HashMap<usize, Vec<&Rect>> = std::collections::HashMap::new();
    for (i, rect) in rects.iter().enumerate() {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(rect);
    }
    let mut out: Vec<Rect> = groups
        .into_values()
        .filter(|g| g.len() > group_threshold)
        .map(|g| {
            let k = g.len() as f64;
            Rect::new(
                // Upstream sums as `int`, then scales by an `f32` reciprocal
                // and takes `saturate_cast<int>` — which is `cvRound`,
                // half-to-**even**. `.round()` is half-away-from-zero, so a
                // group averaging exactly 60.5 comes out 61 here and 60 there.
                avg(g.iter().map(|r| r.x), k as f32),
                avg(g.iter().map(|r| r.y), k as f32),
                avg(g.iter().map(|r| r.width), k as f32),
                avg(g.iter().map(|r| r.height), k as f32),
            )
        })
        .collect();
    out.sort_by_key(|r| (r.y, r.x, r.width, r.height));
    out
}

/// A Haar-like feature: rectangles with weights, evaluated over an integral image.
#[derive(Debug, Clone, PartialEq)]
pub struct HaarFeature {
    /// `(rect, weight)` pairs, relative to the window's top-left corner.
    pub rects: Vec<(Rect, f64)>,
    /// Threshold on the weighted sum.
    pub threshold: f64,
    /// Value returned when the sum is below the threshold.
    pub left: f64,
    /// Value returned when it is at or above.
    pub right: f64,
}

/// One boosting stage: features summed against a stage threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct CascadeStage {
    /// Weak classifiers.
    pub features: Vec<HaarFeature>,
    /// The stage passes when the sum reaches this.
    pub threshold: f64,
}

/// `cv::CascadeClassifier` — a Viola-Jones attentional cascade.
///
/// The XML loader is not here; a cascade is constructed from stages directly.
/// That keeps the evaluator honest — it either runs a real cascade or it does
/// not, with no half-parsed model in between.
#[derive(Debug, Clone, PartialEq)]
pub struct CascadeClassifier {
    /// Stages, evaluated in order; the first failure rejects the window.
    pub stages: Vec<CascadeStage>,
    /// Window size the stages were trained at.
    pub win: (i32, i32),
}

impl CascadeClassifier {
    /// Build a classifier from its stages.
    pub fn new(stages: Vec<CascadeStage>, win: (i32, i32)) -> Result<Self> {
        cv_assert!(!stages.is_empty(), "a cascade needs at least one stage");
        cv_assert!(win.0 > 0 && win.1 > 0, "the window size must be positive");
        Ok(CascadeClassifier { stages, win })
    }

    /// Evaluate one window against the whole cascade.
    ///
    /// `integral` is the `(rows+1) x (cols+1)` summed-area table.
    pub fn passes<I: MatLike + ?Sized>(&self, integral: &I, x0: i32, y0: i32, scale: f64) -> Result<bool> {
        let area = |r: &Rect| -> Result<f64> {
            let (rx, ry) = (
                x0 + (f64::from(r.x) * scale).round() as i32,
                y0 + (f64::from(r.y) * scale).round() as i32,
            );
            let (rw, rh) = (
                (f64::from(r.width) * scale).round() as i32,
                (f64::from(r.height) * scale).round() as i32,
            );
            let (x1, y1) = (rx + rw, ry + rh);
            if rx < 0 || ry < 0 || x1 >= integral.cols() || y1 >= integral.rows() {
                return Ok(0.0);
            }
            Ok(integral.at_f64(y1, x1, 0)? + integral.at_f64(ry, rx, 0)?
                - integral.at_f64(ry, x1, 0)?
                - integral.at_f64(y1, rx, 0)?)
        };
        for stage in &self.stages {
            let mut sum = 0.0;
            for f in &stage.features {
                let mut v = 0.0;
                for (r, w) in &f.rects {
                    v += w * area(r)?;
                }
                sum += if v < f.threshold { f.left } else { f.right };
            }
            if sum < stage.threshold {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// `cv::CascadeClassifier::detectMultiScale(image, objects, scaleFactor, minNeighbors, ...)`.
    pub fn detect_multi_scale<S: MatLike + ?Sized>(
        &self,
        img: &S,
        scale_factor: f64,
        min_neighbours: usize,
        step: i32,
    ) -> Result<Vec<Rect>> {
        cv_assert!(scale_factor > 1.0, "scaleFactor must exceed 1");
        cv_assert!(step > 0, "the search step must be positive");
        //  returns (sum, squared-sum); the cascade only needs the sum.
        let (integral, _sqsum) = cv_imgproc::integral(img)?;
        let mut hits = Vec::new();
        let mut scale = 1.0f64;
        loop {
            let (ww, wh) = (
                (f64::from(self.win.0) * scale).round() as i32,
                (f64::from(self.win.1) * scale).round() as i32,
            );
            if ww > img.cols() || wh > img.rows() {
                break;
            }
            let mut y = 0;
            while y + wh <= img.rows() {
                let mut x = 0;
                while x + ww <= img.cols() {
                    if self.passes(&integral, x, y, scale)? {
                        hits.push(Rect::new(x, y, ww, wh));
                    }
                    x += step;
                }
                y += step;
            }
            scale *= scale_factor;
        }
        Ok(group_rectangles(&hits, min_neighbours, 0.2))
    }
}

/// A located QR finder pattern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FinderPattern {
    /// Centre of the pattern.
    pub center: (f64, f64),
    /// Width of one module, estimated from the 1:1:3:1:1 run.
    pub module_size: f64,
}

/// `cv::QRCodeDetector::detect(img, points)` — the finder-pattern half.
///
/// Scans rows and columns for the 1:1:3:1:1 dark/light run ratio that the
/// three corner squares of a QR symbol produce, then keeps candidates
/// confirmed in both directions.
///
/// Returns the patterns found. Locating three of them gives the symbol's
/// corners; decoding the payload is a separate job this port does not do.
pub fn find_qr_finder_patterns<S: MatLike + ?Sized>(
    img: &S,
    threshold: f64,
) -> Result<Vec<FinderPattern>> {
    cv_assert!(img.channels() == 1, "QR detection needs a single-channel image");
    let (h, w) = (img.rows(), img.cols());
    let mut candidates: Vec<(f64, f64, f64)> = Vec::new();

    // Horizontal scan for the 1:1:3:1:1 pattern.
    for y in 0..h {
        let mut runs: Vec<(bool, i32, i32)> = Vec::new(); // (dark, start, len)
        let mut cur_dark = img.at_f64(y, 0, 0)? < threshold;
        let mut start = 0;
        for x in 1..=w {
            let dark = if x < w { img.at_f64(y, x, 0)? < threshold } else { !cur_dark };
            if dark != cur_dark {
                runs.push((cur_dark, start, x - start));
                cur_dark = dark;
                start = x;
            }
        }
        for k in 0..runs.len().saturating_sub(4) {
            let win = &runs[k..k + 5];
            if !win[0].0 || win[1].0 || !win[2].0 || win[3].0 || !win[4].0 {
                continue; // must be dark, light, dark, light, dark
            }
            let unit = f64::from(win[0].2 + win[1].2 + win[3].2 + win[4].2) / 4.0;
            if unit < 1.0 {
                continue;
            }
            let ok = [win[0].2, win[1].2, win[3].2, win[4].2]
                .iter()
                .all(|&l| (f64::from(l) - unit).abs() <= unit * 0.6)
                && (f64::from(win[2].2) - 3.0 * unit).abs() <= unit * 1.2;
            if ok {
                let cx = f64::from(win[2].1) + f64::from(win[2].2) * 0.5;
                candidates.push((cx, f64::from(y), unit));
            }
        }
    }

    // Confirm vertically and merge nearby candidates.
    let mut out: Vec<FinderPattern> = Vec::new();
    for (cx, cy, unit) in candidates {
        let x = cx.round() as i32;
        if x < 0 || x >= w {
            continue;
        }
        // Count the dark run through the centre vertically.
        let (mut up, mut down) = (0, 0);
        let mut yy = cy.round() as i32;
        while yy >= 0 && img.at_f64(yy, x, 0)? < threshold {
            up += 1;
            yy -= 1;
        }
        let mut yy = cy.round() as i32 + 1;
        while yy < h && img.at_f64(yy, x, 0)? < threshold {
            down += 1;
            yy += 1;
        }
        let vert = f64::from(up + down);
        if (vert - 3.0 * unit).abs() > unit * 1.5 {
            continue;
        }
        let centre = (cx, cy.round() - f64::from(up) + vert * 0.5);
        // Merge with an existing pattern if they overlap.
        if let Some(p) = out.iter_mut().find(|p| {
            (p.center.0 - centre.0).abs() < unit * 2.0 && (p.center.1 - centre.1).abs() < unit * 2.0
        }) {
            p.center = ((p.center.0 + centre.0) * 0.5, (p.center.1 + centre.1) * 0.5);
            p.module_size = (p.module_size + unit) * 0.5;
        } else {
            out.push(FinderPattern { center: centre, module_size: unit });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::CV_8UC1;
    use cv_core::MatLikeMut;

    fn gradient(w: i32, h: i32) -> Mat {
        let mut m = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                m.pixel_mut::<u8>(y, x).unwrap()[0] = ((x * 3 + y * 5) % 256) as u8;
            }
        }
        m
    }

    #[test]
    fn descriptor_size_matches_upstreams_default() {
        // cv2.HOGDescriptor().getDescriptorSize() == 3780.
        assert_eq!(HogParams::default().descriptor_size(), 3780);
    }

    #[test]
    fn bad_geometry_is_rejected() {
        // Not a whole number of 8x8 cells.
        let p = HogParams { block: (15, 16), ..Default::default() };
        assert!(HogDescriptor::new(p).is_err());
        let q = HogParams { nbins: 0, ..Default::default() };
        assert!(HogDescriptor::new(q).is_err());
    }

    #[test]
    fn compute_returns_the_declared_length() {
        let hog = HogDescriptor::default_people().unwrap();
        let img = gradient(128, 200);
        let d = hog.compute_at(&img, 0, 0).unwrap();
        assert_eq!(d.len(), hog.params.descriptor_size());
        assert!(d.iter().all(|v| v.is_finite()), "descriptors must be finite");
    }

    #[test]
    fn blocks_are_l2_hys_normalised() {
        let hog = HogDescriptor::default_people().unwrap();
        let img = gradient(128, 200);
        let d = hog.compute_at(&img, 0, 0).unwrap();
        let per_block = (hog.params.block.0 / hog.params.cell.0
            * hog.params.block.1
            / hog.params.cell.1) as usize
            * hog.params.nbins;
        for chunk in d.chunks(per_block) {
            let n: f32 = chunk.iter().map(|v| v * v).sum::<f32>().sqrt();
            // After the clip-and-renormalise the norm is *just under* 1, by
            // `1e-3 / ||v||` — upstream's second epsilon is 1e-3, not a token
            // 1e-6, so how far under depends on the block. A flat block
            // normalises to 0.
            assert!(
                !(1e-3..=0.98).contains(&n) && n < 1.0001,
                "block norm {n} is neither near-1 nor 0"
            );

        }
    }

    #[test]
    fn l2_hys_clip_stops_one_bin_dominating() {
        // The clip happens *between* the two normalisations, so final values
        // can exceed 0.2 — asserting otherwise tests the wrong thing. What the
        // clip actually guarantees is that a single huge bin cannot take all
        // the energy.
        let raw = vec![100.0f64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut clipped = raw.clone();
        l2_hys(&mut clipped);
        let norm: f64 = clipped.iter().map(|v| v * v).sum::<f64>().sqrt();
        // Just under 1, not exactly: the `+1e-3` epsilon in the second
        // denominator keeps an all-zero block from dividing by zero, and it
        // costs `1e-3 / ||v||` of norm — about half a percent on a block this
        // small, which is upstream's behaviour and not a rounding slip.
        assert!((0.98..=1.0).contains(&norm), "L2-Hys output is near-unit, got {norm}");

        // Plain L2, no clip, for comparison.
        let n: f64 = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
        let plain: Vec<f64> = raw.iter().map(|v| v / n).collect();
        assert!(
            clipped[1] > plain[1] * 4.0,
            "the clip should hand the small bins several times more weight: {} vs {}",
            clipped[1],
            plain[1]
        );
        assert!(clipped[0] < plain[0], "and take some away from the spike");
    }

    #[test]
    fn a_window_outside_the_image_is_rejected() {
        let hog = HogDescriptor::default_people().unwrap();
        let img = gradient(64, 64);
        assert!(hog.compute_at(&img, 0, 0).is_err(), "a 64x128 window cannot fit 64x64");
    }

    #[test]
    fn identical_patches_give_identical_descriptors() {
        // A tiled image: two windows one tile apart see the same thing.
        let hog = HogDescriptor::new(HogParams {
            win: (32, 32),
            block: (16, 16),
            block_stride: (8, 8),
            cell: (8, 8),
            nbins: 9,
        })
        .unwrap();
        let mut img = Mat::zeros(64, 96, CV_8UC1).unwrap();
        for y in 0..64 {
            for x in 0..96 {
                img.pixel_mut::<u8>(y, x).unwrap()[0] = if (x % 16 < 8) == (y % 16 < 8) { 220 } else { 30 };
            }
        }
        let a = hog.compute_at(&img, 16, 16).unwrap();
        let b = hog.compute_at(&img, 32, 16).unwrap();
        for (p, q) in a.iter().zip(b.iter()) {
            assert!((p - q).abs() < 1e-5, "a periodic image must give periodic descriptors");
        }
    }

    #[test]
    fn detect_multi_scale_finds_what_the_classifier_likes() {
        let hog = HogDescriptor::new(HogParams {
            win: (32, 32),
            block: (16, 16),
            block_stride: (16, 16),
            cell: (8, 8),
            nbins: 9,
        })
        .unwrap();
        let img = gradient(64, 64);
        let n = hog.params.descriptor_size();
        // A classifier that accepts everything, then one that accepts nothing.
        let all = hog
            .detect_multi_scale(&img, &vec![0.0; n], 1.0, 0.0, (16, 16), 1.5, 3)
            .unwrap();
        assert!(!all.is_empty(), "a permissive classifier should fire");
        let none = hog
            .detect_multi_scale(&img, &vec![0.0; n], -1.0, 0.0, (16, 16), 1.5, 3)
            .unwrap();
        assert!(none.is_empty(), "a rejecting classifier should not fire");
    }

    #[test]
    fn detect_multi_scale_validates_its_arguments() {
        let hog = HogDescriptor::default_people().unwrap();
        let img = gradient(128, 200);
        assert!(hog.detect_multi_scale(&img, &[0.0; 3], 0.0, 0.0, (8, 8), 1.05, 2).is_err());
        let n = hog.params.descriptor_size();
        assert!(hog.detect_multi_scale(&img, &vec![0.0; n], 0.0, 0.0, (8, 8), 1.0, 2).is_err());
        assert!(hog.detect_multi_scale(&img, &vec![0.0; n], 0.0, 0.0, (0, 8), 1.05, 2).is_err());
    }

    #[test]
    fn group_rectangles_merges_a_cluster() {
        let r = |x, y| Rect::new(x, y, 40, 40);
        let rects = vec![r(10, 10), r(12, 11), r(11, 13), r(200, 200)];
        // Threshold 1 keeps clusters with more than one member.
        let g = group_rectangles(&rects, 1, 0.2);
        assert_eq!(g.len(), 1, "the lone rectangle should be dropped, got {g:?}");
        assert!((g[0].x - 11).abs() <= 1 && (g[0].y - 11).abs() <= 1, "{:?}", g[0]);
    }

    #[test]
    fn group_rectangles_threshold_zero_passes_everything_through() {
        let rects = vec![Rect::new(0, 0, 10, 10), Rect::new(50, 50, 10, 10)];
        assert_eq!(group_rectangles(&rects, 0, 0.2), rects);
        assert!(group_rectangles(&[], 1, 0.2).is_empty());
    }

    #[test]
    fn group_rectangles_keeps_distinct_clusters_apart() {
        let r = |x, y| Rect::new(x, y, 30, 30);
        let rects = vec![r(0, 0), r(2, 1), r(100, 100), r(102, 101)];
        let g = group_rectangles(&rects, 1, 0.2);
        assert_eq!(g.len(), 2, "two clusters, got {g:?}");
    }

    #[test]
    fn a_cascade_accepts_and_rejects_by_its_threshold() {
        // One feature over a bright square. The rejecting window is at (0,0)
        // and the square starts at 30, so it genuinely sees nothing — an
        // earlier version put the square at 10 and the "dark" window overlapped
        // it, which is why this fixture is spaced out.
        let mut img = Mat::zeros(60, 60, CV_8UC1).unwrap();
        for y in 30..50 {
            for x in 30..50 {
                img.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        let feat = HaarFeature {
            rects: vec![(Rect::new(0, 0, 20, 20), 1.0)],
            threshold: 1000.0,
            left: -1.0,
            right: 1.0,
        };
        let easy = CascadeClassifier::new(
            vec![CascadeStage { features: vec![feat.clone()], threshold: 0.0 }],
            (20, 20),
        )
        .unwrap();
        let (integral, _) = cv_imgproc::integral(&img).unwrap();
        assert!(easy.passes(&integral, 30, 30, 1.0).unwrap(), "the bright window should pass");
        assert!(!easy.passes(&integral, 0, 0, 1.0).unwrap(), "the empty corner should not");
    }

    #[test]
    fn cascade_detect_multi_scale_locates_a_bright_square() {
        let mut img = Mat::zeros(60, 60, CV_8UC1).unwrap();
        for y in 20..40 {
            for x in 20..40 {
                img.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        let stage = CascadeStage {
            features: vec![HaarFeature {
                rects: vec![(Rect::new(0, 0, 20, 20), 1.0)],
                threshold: 60000.0,
                left: -1.0,
                right: 1.0,
            }],
            threshold: 0.0,
        };
        let cc = CascadeClassifier::new(vec![stage], (20, 20)).unwrap();
        let hits = cc.detect_multi_scale(&img, 1.5, 0, 4).unwrap();
        assert!(!hits.is_empty(), "the square was not found");
        assert!(
            hits.iter().any(|r| (r.x - 20).abs() <= 6 && (r.y - 20).abs() <= 6),
            "no detection near the square: {hits:?}"
        );
    }

    #[test]
    fn an_empty_cascade_is_rejected() {
        assert!(CascadeClassifier::new(vec![], (20, 20)).is_err());
    }

    #[test]
    fn qr_finder_patterns_are_located() {
        // Three 7x7 finder patterns at 3x module scale, in the usual corners.
        let m = 3; // module size in pixels
        let sz = 7 * m;
        let mut img = Mat::from_vec(120, 120, 1, &[255u8; 14400]).unwrap();
        let draw = |img: &mut Mat, ox: i32, oy: i32| {
            for r in 0..7 {
                for c in 0..7 {
                    // 7x7 finder: dark ring, light ring, 3x3 dark centre.
                    let dark = r == 0
                        || r == 6
                        || c == 0
                        || c == 6
                        || ((2..=4).contains(&r) && (2..=4).contains(&c));
                    for dy in 0..m {
                        for dx in 0..m {
                            img.pixel_mut::<u8>(oy + r * m + dy, ox + c * m + dx).unwrap()[0] =
                                if dark { 0 } else { 255 };
                        }
                    }
                }
            }
        };
        draw(&mut img, 10, 10);
        draw(&mut img, 120 - 10 - sz, 10);
        draw(&mut img, 10, 120 - 10 - sz);

        let pats = find_qr_finder_patterns(&img, 128.0).unwrap();
        assert!(pats.len() >= 3, "expected three finder patterns, got {}: {pats:?}", pats.len());
        let near = |x: f64, y: f64| {
            pats.iter()
                .any(|p| (p.center.0 - x).abs() < 4.0 && (p.center.1 - y).abs() < 4.0)
        };
        let c = f64::from(sz) / 2.0;
        assert!(near(10.0 + c, 10.0 + c), "top-left missing: {pats:?}");
        assert!(near(120.0 - 10.0 - c, 10.0 + c), "top-right missing: {pats:?}");
        assert!(near(10.0 + c, 120.0 - 10.0 - c), "bottom-left missing: {pats:?}");
        for p in &pats {
            assert!((p.module_size - f64::from(m)).abs() < 1.5, "module size {}", p.module_size);
        }
    }

    #[test]
    fn a_blank_image_has_no_finder_patterns() {
        let img = Mat::from_vec(60, 60, 1, &[255u8; 3600]).unwrap();
        assert!(find_qr_finder_patterns(&img, 128.0).unwrap().is_empty());
    }

    #[test]
    fn qr_detection_rejects_a_colour_image() {
        let img = Mat::zeros(20, 20, cv_core::types::CV_8UC3).unwrap();
        assert!(find_qr_finder_patterns(&img, 128.0).is_err());
    }
}
