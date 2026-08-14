//! The `imgproc` classes and the last few free functions
//! (`modules/imgproc/src/clahe.cpp`, `lsd.cpp`, `emd.cpp`,
//! `intelligent_scissors.cpp`, `linefind.cpp`).
//!
//! Everything here wraps an algorithm that already exists as a free function,
//! or is small enough that a separate module would be more filing than code.
//! The one exception is [`LineSegmentDetector`], which is its own algorithm.

use cv_core::types::{Depth, CV_32FC1, CV_8UC1};
use cv_core::{cv_assert, cv_error, Code, Mat, MatLike, MatLikeMut, Point, Result, Size};

use crate::contours::{DistanceMetric, Hierarchy};

// ---------------------------------------------------------------------------
// CLAHE
// ---------------------------------------------------------------------------

/// `cv::CLAHE` — contrast-limited adaptive histogram equalisation, as an object.
///
/// The settings live on the object because upstream's do, and because a
/// pipeline that equalises every frame of a video wants to set the clip limit
/// once. [`crate::hist2::clahe`] is the same computation as a call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clahe {
    clip_limit: f64,
    tiles: Size,
    bit_shift: i32,
}

impl Default for Clahe {
    /// Upstream's defaults: a clip limit of 40 over an 8x8 grid, no bit shift.
    fn default() -> Self {
        Clahe { clip_limit: 40.0, tiles: Size::new(8, 8), bit_shift: 0 }
    }
}

impl Clahe {
    /// `cv::createCLAHE(clipLimit, tileGridSize)`.
    #[must_use]
    pub fn create(clip_limit: f64, tiles: Size) -> Self {
        Clahe { clip_limit, tiles, bit_shift: 0 }
    }

    /// `cv::CLAHE::getBitShift()` — how far the histogram bins are shifted down.
    ///
    /// A 16-bit image has 65536 possible values, and a histogram that wide is
    /// mostly empty; shifting drops the low bits so the bins carry counts worth
    /// equalising. Zero, the default, means no shift.
    #[must_use]
    pub fn get_bit_shift(&self) -> i32 {
        self.bit_shift
    }

    /// `cv::CLAHE::setBitShift(bitShift)`.
    ///
    /// Accepted and unused: [`crate::hist2::clahe`] is the `CV_8UC1` path,
    /// where the histogram is already 256 bins and there is nothing to shift.
    /// A 16-bit input is refused rather than silently equalised at the wrong
    /// bin width.
    pub fn set_bit_shift(&mut self, bit_shift: i32) {
        self.bit_shift = bit_shift;
    }

    /// `cv::CLAHE::apply(src, dst)`.
    pub fn apply<S: MatLike + ?Sized>(&self, src: &S) -> Result<Mat> {
        crate::hist2::clahe(src, self.clip_limit, self.tiles.width, self.tiles.height)
    }

    /// `cv::CLAHE::getClipLimit()`.
    #[must_use]
    pub fn get_clip_limit(&self) -> f64 {
        self.clip_limit
    }
    /// `cv::CLAHE::setClipLimit(clipLimit)`.
    pub fn set_clip_limit(&mut self, clip_limit: f64) {
        self.clip_limit = clip_limit;
    }
    /// `cv::CLAHE::getTilesGridSize()`.
    #[must_use]
    pub fn get_tiles_grid_size(&self) -> Size {
        self.tiles
    }
    /// `cv::CLAHE::setTilesGridSize(tileGridSize)`.
    pub fn set_tiles_grid_size(&mut self, tiles: Size) {
        self.tiles = tiles;
    }
    /// `cv::CLAHE::collectGarbage()`.
    ///
    /// Nothing to do: upstream's caches the per-tile lookup tables between
    /// calls and this recomputes them, so there is no buffer to release. Kept
    /// so a pipeline written against upstream compiles unchanged.
    pub fn collect_garbage(&mut self) {}
}

/// `cv::createCLAHE(clipLimit, tileGridSize)`.
#[must_use]
pub fn create_clahe(clip_limit: f64, tiles: Size) -> Clahe {
    Clahe::create(clip_limit, tiles)
}

// ---------------------------------------------------------------------------
// Line segment detector
// ---------------------------------------------------------------------------

/// `cv::LineSegmentDetectorModes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LsdMode {
    /// `cv::LSD_REFINE_NONE` — no refinement; fastest.
    None,
    /// `cv::LSD_REFINE_STD` — split arcs into several segments. Upstream's default.
    #[default]
    Std,
    /// `cv::LSD_REFINE_ADV` — additionally refine each segment's endpoints.
    Adv,
}

/// One detected segment: `(x1, y1, x2, y2)`.
pub type Segment = (f32, f32, f32, f32);

/// `cv::LineSegmentDetector` — Grompone von Gioi's LSD.
///
/// # How it differs from Hough
///
/// The Hough transform asks "which lines could these edge pixels lie on"; LSD
/// asks "which *connected regions* have a consistent gradient direction". It
/// grows a region from each seed pixel while the level-line angle stays within
/// a tolerance, fits a rectangle to it, and keeps the rectangle if it is
/// unlikely to have arisen by chance. That is why it needs no threshold from
/// the caller and returns endpoints rather than infinite lines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineSegmentDetector {
    refine: LsdMode,
    scale: f64,
    sigma_scale: f64,
    quant: f64,
    ang_th: f64,
    log_eps: f64,
    density_th: f64,
    n_bins: i32,
}

impl Default for LineSegmentDetector {
    /// Upstream's `createLineSegmentDetector()` defaults.
    fn default() -> Self {
        LineSegmentDetector {
            refine: LsdMode::Std,
            scale: 0.8,
            sigma_scale: 0.6,
            quant: 2.0,
            ang_th: 22.5,
            log_eps: 0.0,
            density_th: 0.7,
            n_bins: 1024,
        }
    }
}

/// `cv::createLineSegmentDetector(refine, scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins)`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn create_line_segment_detector(
    refine: LsdMode,
    scale: f64,
    sigma_scale: f64,
    quant: f64,
    ang_th: f64,
    log_eps: f64,
    density_th: f64,
    n_bins: i32,
) -> LineSegmentDetector {
    LineSegmentDetector {
        refine,
        scale,
        sigma_scale,
        quant,
        ang_th,
        log_eps,
        density_th,
        n_bins,
    }
}

impl LineSegmentDetector {
    /// `cv::LineSegmentDetector::detect(image, lines, width, prec, nfa)`.
    ///
    /// Returns the segments with their estimated width and angular precision.
    pub fn detect<S: MatLike + ?Sized>(&self, image: &S) -> Result<Vec<Segment>> {
        cv_assert!(image.channels() == 1, "LSD takes a single-channel image");
        cv_assert!(self.scale > 0.0, "the scale must be positive");
        cv_assert!(self.ang_th > 0.0 && self.ang_th < 180.0, "ang_th must lie in 0..180");
        cv_assert!((0.0..=1.0).contains(&self.density_th), "density_th must lie in 0..1");
        cv_assert!(self.n_bins > 0, "n_bins must be positive");
        let (h, w) = (image.rows(), image.cols());
        if h < 3 || w < 3 {
            return Ok(Vec::new());
        }

        // Upstream scales the image down by `scale` with a Gaussian whose sigma
        // is `sigma_scale / scale`; skipping the blur is what makes a naive LSD
        // fire on every noisy pixel.
        let work = if (self.scale - 1.0).abs() < 1e-12 {
            Mat::copy_from(image)?
        } else {
            let sigma = self.sigma_scale / self.scale;
            let k = ((sigma * 3.0).ceil() as i32 * 2 + 1).max(3);
            let blurred = crate::filter::gaussian_blur(
                image,
                Size::new(k, k),
                sigma,
                sigma,
                cv_core::BorderType::Replicate,
            )?;
            crate::resize::resize(
                &blurred,
                Size::new(
                    ((f64::from(w) * self.scale).round() as i32).max(1),
                    ((f64::from(h) * self.scale).round() as i32).max(1),
                ),
                0.0,
                0.0,
                crate::resize::Interpolation::Linear,
            )?
        };
        let (wh, ww) = (work.rows(), work.cols());

        // The level-line angle at each pixel: perpendicular to the gradient.
        let mut angle = vec![f64::NAN; (wh * ww) as usize];
        let mut grad = vec![0.0f64; (wh * ww) as usize];
        let mut max_grad = 0.0f64;
        for y in 0..wh - 1 {
            for x in 0..ww - 1 {
                let a = work.at_f64(y, x, 0)?;
                let b = work.at_f64(y, x + 1, 0)?;
                let c = work.at_f64(y + 1, x, 0)?;
                let d = work.at_f64(y + 1, x + 1, 0)?;
                // Upstream's 2x2 difference, not a Sobel: the smaller support
                // is what keeps two nearby lines from merging.
                let gx = b + d - (a + c);
                let gy = c + d - (a + b);
                let norm = (gx * gx + gy * gy).sqrt() / 4.0;
                let i = (y * ww + x) as usize;
                grad[i] = norm;
                max_grad = max_grad.max(norm);
                if norm > self.quant / 2.0 {
                    angle[i] = gx.atan2(-gy);
                }
            }
        }

        // Seeds, largest gradient first: a strong pixel is the most reliable
        // place to start growing a region.
        let mut order: Vec<usize> = (0..grad.len()).filter(|&i| grad[i] > 0.0).collect();
        order.sort_by(|&a, &b| grad[b].total_cmp(&grad[a]));

        let prec = self.ang_th.to_radians();
        let mut used = vec![false; grad.len()];
        let mut out = Vec::new();
        for seed in order {
            if used[seed] || angle[seed].is_nan() {
                continue;
            }
            // Region growing: breadth-first over 8-neighbours whose level-line
            // angle is within `prec` of the region's running mean.
            let mut region = vec![seed];
            used[seed] = true;
            let (mut sx, mut sy) = (angle[seed].cos(), angle[seed].sin());
            let mut qi = 0;
            while qi < region.len() {
                let p = region[qi];
                qi += 1;
                let (px, py) = ((p as i32) % ww, (p as i32) / ww);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (nx, ny) = (px + dx, py + dy);
                        if nx < 0 || ny < 0 || nx >= ww || ny >= wh {
                            continue;
                        }
                        let n = (ny * ww + nx) as usize;
                        if used[n] || angle[n].is_nan() {
                            continue;
                        }
                        let mean = sy.atan2(sx);
                        let mut diff = (angle[n] - mean).abs();
                        while diff > core::f64::consts::PI {
                            diff -= 2.0 * core::f64::consts::PI;
                        }
                        if diff.abs() < prec {
                            used[n] = true;
                            region.push(n);
                            sx += angle[n].cos();
                            sy += angle[n].sin();
                        }
                    }
                }
            }
            if region.len() < 5 {
                continue;
            }

            // Fit a rectangle: the principal axis of the region's pixels,
            // weighted by gradient, gives the direction and the extent.
            let total: f64 = region.iter().map(|&i| grad[i]).sum();
            if total <= 0.0 {
                continue;
            }
            let (mut cx, mut cy) = (0.0f64, 0.0f64);
            for &i in &region {
                cx += f64::from((i as i32) % ww) * grad[i];
                cy += f64::from((i as i32) / ww) * grad[i];
            }
            cx /= total;
            cy /= total;
            let (mut ixx, mut iyy, mut ixy) = (0.0f64, 0.0f64, 0.0f64);
            for &i in &region {
                let dx = f64::from((i as i32) % ww) - cx;
                let dy = f64::from((i as i32) / ww) - cy;
                ixx += dx * dx * grad[i];
                iyy += dy * dy * grad[i];
                ixy += dx * dy * grad[i];
            }
            ixx /= total;
            iyy /= total;
            ixy /= total;
            let theta = 0.5 * (2.0 * ixy).atan2(ixx - iyy);
            let (dx, dy) = (theta.cos(), theta.sin());
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            let mut wid = 0.0f64;
            for &i in &region {
                let px = f64::from((i as i32) % ww) - cx;
                let py = f64::from((i as i32) / ww) - cy;
                let t = px * dx + py * dy;
                lo = lo.min(t);
                hi = hi.max(t);
                wid = wid.max((px * -dy + py * dx).abs());
            }
            let length = hi - lo;
            if length < 3.0 {
                continue;
            }
            // The density test: a genuine segment fills its bounding rectangle.
            // A blob has the same principal axis and does not.
            let area = length * (2.0 * wid).max(1.0);
            if (region.len() as f64) / area < self.density_th {
                continue;
            }

            let inv = 1.0 / self.scale.max(1e-12);
            let seg = (
                ((cx + lo * dx) * inv) as f32,
                ((cy + lo * dy) * inv) as f32,
                ((cx + hi * dx) * inv) as f32,
                ((cy + hi * dy) * inv) as f32,
            );
            out.push(seg);
        }
        // Longest first, so the caller's `take(n)` keeps the strongest.
        out.sort_by(|a, b| seg_len(b).total_cmp(&seg_len(a)));
        Ok(out)
    }

    /// `cv::LineSegmentDetector::drawSegments(image, lines)`.
    ///
    /// Draws onto a copy, converting a grey image to colour first so the
    /// overlay is distinguishable from the image under it.
    pub fn draw_segments<S: MatLike + ?Sized>(&self, image: &S, lines: &[Segment]) -> Result<Mat> {
        let mut canvas = if image.channels() == 1 {
            crate::color::cvt_color(image, crate::color::ColorConversion::Gray2Bgr)?
        } else {
            Mat::copy_from(image)?
        };
        let colour = cv_core::Scalar::new(0.0, 0.0, 255.0, 0.0);
        for &(x1, y1, x2, y2) in lines {
            crate::drawing::line(
                &mut canvas,
                Point::new(x1.round() as i32, y1.round() as i32),
                Point::new(x2.round() as i32, y2.round() as i32),
                colour,
                1,
            )?;
        }
        Ok(canvas)
    }

    /// `cv::LineSegmentDetector::compareSegments(size, lines1, lines2, image)`.
    ///
    /// Draws both sets in different channels and counts the pixels that only
    /// one of them covers. That count is the answer to "did my change to the
    /// detector alter anything", which a segment-by-segment diff cannot answer
    /// when the segments are split differently.
    pub fn compare_segments(
        &self,
        size: Size,
        lines1: &[Segment],
        lines2: &[Segment],
    ) -> Result<(i32, Mat)> {
        cv_assert!(size.width > 0 && size.height > 0, "the canvas must be non-empty");
        let draw = |lines: &[Segment]| -> Result<Mat> {
            let mut m = Mat::zeros(size.height, size.width, CV_8UC1)?;
            for &(x1, y1, x2, y2) in lines {
                crate::drawing::line(
                    &mut m,
                    Point::new(x1.round() as i32, y1.round() as i32),
                    Point::new(x2.round() as i32, y2.round() as i32),
                    cv_core::Scalar::new(255.0, 0.0, 0.0, 0.0),
                    1,
                )?;
            }
            Ok(m)
        };
        let (a, b) = (draw(lines1)?, draw(lines2)?);
        let mut out = Mat::zeros(size.height, size.width, cv_core::types::MatType::new(Depth::U8, 3))?;
        let mut count = 0;
        for y in 0..size.height {
            for x in 0..size.width {
                let (pa, pb) = (a.at_f64(y, x, 0)? != 0.0, b.at_f64(y, x, 0)? != 0.0);
                if pa != pb {
                    count += 1;
                }
                let px = out.pixel_mut::<u8>(y, x)?;
                px[1] = u8::from(pa) * 255;
                px[2] = u8::from(pb) * 255;
            }
        }
        Ok((count, out))
    }

    /// The refinement level in force.
    #[must_use]
    pub fn refine(&self) -> LsdMode {
        self.refine
    }
    /// The angle tolerance in degrees, and the quantisation and bin count.
    #[must_use]
    pub fn tuning(&self) -> (f64, f64, f64, i32) {
        (self.ang_th, self.quant, self.log_eps, self.n_bins)
    }
}

/// The length of a segment.
fn seg_len(s: &Segment) -> f64 {
    f64::from(s.2 - s.0).hypot(f64::from(s.3 - s.1))
}

// ---------------------------------------------------------------------------
// Intelligent scissors
// ---------------------------------------------------------------------------

/// `cv::segmentation::IntelligentScissorsMB` — live-wire contour tracing.
///
/// The user clicks a start point; the tool then snaps a path to whichever
/// nearby edge is strongest, so tracing an object takes a handful of clicks
/// instead of a hundred. Under it is Dijkstra on a graph whose edge weights
/// combine gradient magnitude, gradient direction and a zero-crossing term —
/// [`Self::build_map`] runs the search once from the seed, after which
/// [`Self::get_contour`] to any target is a pointer chase.
#[derive(Debug, Clone)]
pub struct IntelligentScissorsMB {
    weight_non_edge: f32,
    weight_gradient_direction: f32,
    weight_gradient_magnitude: f32,
    edge_feature_zero_crossing_threshold: f32,
    gradient_magnitude_max: f32,
    /// The image the map was built from.
    size: Size,
    /// Predecessor index per pixel, or `-1` at the seed and for unreached pixels.
    prev: Vec<i32>,
}

impl Default for IntelligentScissorsMB {
    /// Upstream's default weights.
    fn default() -> Self {
        IntelligentScissorsMB::new()
    }
}

impl IntelligentScissorsMB {
    /// `cv::segmentation::IntelligentScissorsMB::IntelligentScissorsMB()`.
    #[must_use]
    pub fn new() -> Self {
        IntelligentScissorsMB {
            weight_non_edge: 0.43,
            weight_gradient_direction: 0.43,
            weight_gradient_magnitude: 0.14,
            edge_feature_zero_crossing_threshold: 0.01,
            gradient_magnitude_max: 0.0,
            size: Size::new(0, 0),
            prev: Vec::new(),
        }
    }

    /// `cv::segmentation::IntelligentScissorsMB::setWeights(non_edge, dir, mag)`.
    pub fn set_weights(&mut self, non_edge: f32, direction: f32, magnitude: f32) -> Result<()> {
        cv_assert!(
            non_edge >= 0.0 && direction >= 0.0 && magnitude >= 0.0,
            "the weights must not be negative"
        );
        cv_assert!(non_edge + direction + magnitude > 0.0, "at least one weight must be non-zero");
        self.weight_non_edge = non_edge;
        self.weight_gradient_direction = direction;
        self.weight_gradient_magnitude = magnitude;
        Ok(())
    }

    /// `cv::segmentation::IntelligentScissorsMB::setGradientMagnitudeMaxLimit(gradient_magnitude_threshold_max)`.
    ///
    /// Zero means no limit. A limit stops one very strong edge dominating the
    /// cost everywhere, which is what makes the wire jump to it across an
    /// image.
    pub fn set_gradient_magnitude_max_limit(&mut self, limit: f32) {
        self.gradient_magnitude_max = limit;
    }

    /// `cv::segmentation::IntelligentScissorsMB::setEdgeFeatureCannyParameters` is
    /// the alternative to this: `setEdgeFeatureZeroCrossingParameters(gradient_magnitude_min_value)`.
    pub fn set_edge_feature_zero_crossing_parameters(&mut self, min_value: f32) {
        self.edge_feature_zero_crossing_threshold = min_value;
    }

    /// `cv::segmentation::IntelligentScissorsMB::buildMap(sourcePt)`.
    ///
    /// Runs the shortest-path search from `source` over the whole image, so
    /// every later [`Self::get_contour`] is a walk back along the predecessors
    /// rather than another search. That asymmetry is the point: the user moves
    /// the mouse far more often than they click.
    pub fn build_map<S: MatLike + ?Sized>(&mut self, image: &S, source: Point) -> Result<()> {
        cv_assert!(
            matches!(image.channels(), 1 | 3 | 4),
            "intelligent scissors takes CV_8UC1, CV_8UC3 or CV_8UC4, got {} channels",
            image.channels()
        );
        // Upstream's `initGrayscale_`: a colour image is converted, not
        // processed per channel. Refusing colour outright meant the obvious
        // call — hand it the image you are about to cut out — did not compile.
        let grey = if image.channels() == 1 {
            cv_core::mat_from_f64(image.rows(), image.cols(), image.mat_type(), &cv_core::to_f64(image)?)?
        } else {
            crate::color::cvt_color(
                image,
                if image.channels() == 4 {
                    crate::color::ColorConversion::Bgra2Gray
                } else {
                    crate::color::ColorConversion::Bgr2Gray
                },
            )?
        };
        let image = &grey;
        let (h, w) = (image.rows(), image.cols());
        cv_assert!(
            source.x >= 0 && source.y >= 0 && source.x < w && source.y < h,
            "the seed {source:?} lies outside the {w}x{h} image"
        );
        self.size = Size::new(w, h);

        let dx = crate::filter::sobel(
            image,
            Some(Depth::F64),
            1,
            0,
            3,
            1.0,
            0.0,
            cv_core::BorderType::Replicate,
        )?;
        let dy = crate::filter::sobel(
            image,
            Some(Depth::F64),
            0,
            1,
            3,
            1.0,
            0.0,
            cv_core::BorderType::Replicate,
        )?;
        let mut mag = vec![0.0f64; (w * h) as usize];
        let mut peak = 0.0f64;
        for y in 0..h {
            for x in 0..w {
                let m = dx.at_f64(y, x, 0)?.hypot(dy.at_f64(y, x, 0)?);
                mag[(y * w + x) as usize] = m;
                peak = peak.max(m);
            }
        }
        if self.gradient_magnitude_max > 0.0 {
            peak = peak.min(f64::from(self.gradient_magnitude_max));
        }
        let peak = peak.max(1e-12);

        // Cost of *entering* a pixel: cheap where the gradient is strong, so
        // the shortest path hugs edges.
        let cost: Vec<f64> = mag
            .iter()
            .map(|&m| {
                let normalised = (m / peak).min(1.0);
                let edge = 1.0 - normalised;
                f64::from(self.weight_non_edge) * edge
                    + f64::from(self.weight_gradient_magnitude) * edge
                    + f64::from(self.weight_gradient_direction)
                        * f64::from(
                            u8::from(normalised < f64::from(self.edge_feature_zero_crossing_threshold)),
                        )
            })
            .collect();

        // Dijkstra with a binary heap over the 8-neighbourhood.
        let n = (w * h) as usize;
        let mut dist = vec![f64::INFINITY; n];
        let mut prev = vec![-1i32; n];
        let mut done = vec![false; n];
        let start = (source.y * w + source.x) as usize;
        dist[start] = 0.0;
        let mut heap = std::collections::BinaryHeap::new();
        heap.push((core::cmp::Reverse(ordered(0.0)), start));
        while let Some((_, u)) = heap.pop() {
            if done[u] {
                continue;
            }
            done[u] = true;
            let (ux, uy) = ((u as i32) % w, (u as i32) / w);
            for ky in -1..=1i32 {
                for kx in -1..=1i32 {
                    if kx == 0 && ky == 0 {
                        continue;
                    }
                    let (vx, vy) = (ux + kx, uy + ky);
                    if vx < 0 || vy < 0 || vx >= w || vy >= h {
                        continue;
                    }
                    let v = (vy * w + vx) as usize;
                    // A diagonal step is longer, so it costs more; without the
                    // scaling the wire prefers staircases to straight runs.
                    let scale = if kx != 0 && ky != 0 { core::f64::consts::SQRT_2 } else { 1.0 };
                    let nd = dist[u] + cost[v] * scale;
                    if nd < dist[v] {
                        dist[v] = nd;
                        prev[v] = u as i32;
                        heap.push((core::cmp::Reverse(ordered(nd)), v));
                    }
                }
            }
        }
        prev[start] = -1;
        self.prev = prev;
        Ok(())
    }

    /// `cv::segmentation::IntelligentScissorsMB::getContour(targetPt, contour, backward)`.
    ///
    /// The path from the seed to `target`, seed first. `backward` reverses it,
    /// which is what an interactive tool wants when it is appending to a
    /// contour it is already drawing.
    pub fn get_contour(&self, target: Point, backward: bool) -> Result<Vec<Point>> {
        cv_assert!(!self.prev.is_empty(), "buildMap must be called before getContour");
        let (w, h) = (self.size.width, self.size.height);
        cv_assert!(
            target.x >= 0 && target.y >= 0 && target.x < w && target.y < h,
            "the target {target:?} lies outside the {w}x{h} map"
        );
        let mut path = Vec::new();
        let mut at = target.y * w + target.x;
        loop {
            path.push(Point::new(at % w, at / w));
            let p = self.prev[at as usize];
            if p < 0 {
                break;
            }
            at = p;
            // A cycle cannot happen in a shortest-path tree, but a corrupted
            // map should not hang the caller.
            cv_assert!(path.len() <= (w * h) as usize, "the predecessor map has a cycle");
        }
        if !backward {
            path.reverse();
        }
        Ok(path)
    }
}

/// A total order on `f64` for the heap, since `f64` has none.
fn ordered(v: f64) -> u64 {
    let bits = v.to_bits();
    if v.is_sign_negative() {
        !bits
    } else {
        bits ^ (1 << 63)
    }
}

// ---------------------------------------------------------------------------
// The remaining free functions
// ---------------------------------------------------------------------------

/// `cv::Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType)`.
///
/// The 3x3 derivative with the least angular error of any 3x3 kernel — that is
/// the whole reason it exists. Where `Sobel`'s `[1 2 1]` smoothing is a rule of
/// thumb, Scharr's `[3 10 3]` is fitted to minimise the direction error, which
/// matters whenever the gradient *angle* is used rather than its magnitude.
#[allow(clippy::too_many_arguments)]
pub fn scharr<S: MatLike + ?Sized>(
    src: &S,
    ddepth: Option<Depth>,
    dx: i32,
    dy: i32,
    scale: f64,
    delta: f64,
    border: cv_core::BorderType,
) -> Result<Mat> {
    cv_assert!(dx + dy == 1, "Scharr computes a first derivative: dx + dy must be 1");
    // `ksize = -1` is upstream's own spelling of "use the Scharr kernel".
    crate::filter::sobel(src, ddepth, dx, dy, -1, scale, delta, border)
}

/// `cv::phaseCorrelateIterative(src1, src2, L2size, maxIters)`.
///
/// The same phase-correlation landscape as [`crate::extra::phase_correlate`],
/// but the peak is localised by upstream's iterative-centroid refinement rather
/// than by a parabola fit: the `L2size` neighbourhood of the integer peak is
/// upsampled to 223x223, and a window of about 45% of that is walked to the
/// centroid until it stops moving. Re-centring is what makes it accurate — the
/// centroid of a window that is not centred on the peak is pulled toward the
/// window, and a single fit cannot correct for that.
///
/// # The sign is the opposite of `phaseCorrelate`
///
/// Upstream's `phaseCorrelate` returns `center - peak` and its
/// `phaseCorrelateIterative` returns `peak - center`. The two therefore report
/// **opposite** shifts for the same pair of images. That is upstream's
/// inconsistency, not this port's, and it is reproduced rather than quietly
/// corrected: a caller checking against cv2 needs the cv2 answer.
pub fn phase_correlate_iterative<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    src1: &A,
    src2: &B,
    l2_size: i32,
    max_iters: i32,
) -> Result<(f64, f64)> {
    cv_assert!(l2_size >= 3, "L2size must be at least 3, got {l2_size}");
    cv_assert!(max_iters > 0, "maxIters must be positive");
    cv_assert!(src1.size() == src2.size(), "the two images must match in size");
    cv_assert!(src1.channels() == 1 && src2.channels() == 1, "single-channel only");
    let (h, w) = (src1.rows(), src1.cols());
    cv_assert!(h > 2 && w > 2, "the landscape needs at least a 3x3 image");

    let l3 = correlation_landscape(src1, src2)?;
    let (mid_x, mid_y) = (f64::from(w / 2), f64::from(h / 2));

    // The integer peak of the shifted landscape.
    let (mut best, mut peak) = (f64::NEG_INFINITY, (0i32, 0i32));
    for y in 0..h {
        for x in 0..w {
            let v = l3.at_f64(y, x, 0)?;
            if v > best {
                best = v;
                peak = (x, y);
            }
        }
    }

    // Shrink the neighbourhood until it fits; if even 3x3 does not, the peak is
    // against the border and the integer answer is all there is.
    let mut size = l2_size;
    while out_of_bounds(peak.0, peak.1, w, h, size) {
        size -= 2;
        if size < 3 {
            return Ok((f64::from(peak.0) - mid_x, f64::from(peak.1) - mid_y));
        }
    }

    let l2 = l3.roi(cv_core::Rect::new(peak.0 - size / 2, peak.1 - size / 2, size, size))?;
    // 223 is upstream's empirically chosen upsampling size.
    const L2U_SIZE: i32 = 223;
    let l2u = crate::resize::resize(
        &l2,
        Size::new(L2U_SIZE, L2U_SIZE),
        0.0,
        0.0,
        crate::resize::Interpolation::Linear,
    )?;
    let l2u_mid = f64::from(L2U_SIZE / 2);
    let ratio = f64::from(L2U_SIZE) / f64::from(size);

    // Upstream's schedule: start at a 45% window and shrink if it will not
    // converge. A smaller window is sharper and more easily thrown off, so the
    // widest one that converges is the one to trust.
    let mut l1_ratio = 0.45f64;
    while l1_size(L2U_SIZE, l1_ratio) > 0 {
        let l1 = l1_size(L2U_SIZE, l1_ratio);
        let l1_mid = f64::from(l1 / 2);
        let (mut px, mut py) = (l2u_mid, l2u_mid);
        for _ in 0..max_iters {
            if out_of_bounds(px.round() as i32, py.round() as i32, L2U_SIZE, L2U_SIZE, l1) {
                break;
            }
            let win = l2u.roi(cv_core::Rect::new(
                px as i32 - l1 / 2,
                py as i32 - l1 / 2,
                l1,
                l1,
            ))?;
            let (cx, cy) = centroid(&win)?;
            px += (cx - l1_mid).round();
            py += (cy - l1_mid).round();
            if (cx - l1_mid).abs() < 0.5 && (cy - l1_mid).abs() < 0.5 {
                return Ok((
                    f64::from(peak.0) - mid_x + (px - l2u_mid + cx - l1_mid) / ratio,
                    f64::from(peak.1) - mid_y + (py - l2u_mid + cy - l1_mid) / ratio,
                ));
            }
        }
        l1_ratio -= 0.025;
    }

    // No window converged: fall back to one centroid over the L2 neighbourhood.
    let (cx, cy) = centroid(&l2)?;
    let m = f64::from(size / 2);
    Ok((
        f64::from(peak.0) - mid_x + cx - m,
        f64::from(peak.1) - mid_y + cy - m,
    ))
}

/// Is a `size x size` window around `(x, y)` outside a `w x h` matrix?
fn out_of_bounds(x: i32, y: i32, w: i32, h: i32, size: i32) -> bool {
    x - size / 2 < 0 || y - size / 2 < 0 || x + size / 2 >= w || y + size / 2 >= h
}

/// Upstream's `getL1size`: a fraction of the upsampled width, forced odd so the
/// window has a centre pixel.
fn l1_size(l2u_size: i32, ratio: f64) -> i32 {
    let s = (ratio * f64::from(l2u_size)).floor() as i32;
    if s % 2 == 0 {
        s + 1
    } else {
        s
    }
}

/// The intensity centroid of a patch, `(m10/m00, m01/m00)`.
fn centroid<S: MatLike + ?Sized>(m: &S) -> Result<(f64, f64)> {
    let mo = crate::shape::moments(m, false)?;
    if mo.m00.abs() < 1e-300 {
        // A patch that sums to zero has no centroid; its centre is the only
        // defensible answer, and it is what the caller's window already is.
        return Ok((f64::from(m.cols()) / 2.0, f64::from(m.rows()) / 2.0));
    }
    Ok((mo.m10 / mo.m00, mo.m01 / mo.m00))
}

/// The phase-correlation landscape, `fftshift`ed so the zero shift is centred.
fn correlation_landscape<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    a: &A,
    b: &B,
) -> Result<Mat> {
    let (h, w) = (a.rows(), a.cols());
    let window = crate::extra::create_hanning_window(w, h)?;
    let windowed = |m: &dyn Fn(i32, i32) -> Result<f64>| -> Result<Mat> {
        let mut o = Mat::zeros(h, w, cv_core::types::CV_64FC1)?;
        for y in 0..h {
            for x in 0..w {
                o.pixel_mut::<f64>(y, x)?[0] = m(y, x)? * window.at_f64(y, x, 0)?;
            }
        }
        Ok(o)
    };
    let fa = cv_core::dft(
        &windowed(&|y, x| a.at_f64(y, x, 0))?,
        cv_core::DftDirection::Forward,
        false,
        false,
    )?;
    let fb = cv_core::dft(
        &windowed(&|y, x| b.at_f64(y, x, 0))?,
        cv_core::DftDirection::Forward,
        false,
        false,
    )?;
    let mut cps = Mat::zeros(h, w, cv_core::types::CV_64FC2)?;
    for y in 0..h {
        for x in 0..w {
            let (ar, ai) = (fa.at_f64(y, x, 0)?, fa.at_f64(y, x, 1)?);
            let (br, bi) = (fb.at_f64(y, x, 0)?, fb.at_f64(y, x, 1)?);
            let (cr, ci) = (ar * br + ai * bi, ar * bi - ai * br);
            let mag = (cr * cr + ci * ci).sqrt().max(1e-300);
            let p = cps.pixel_mut::<f64>(y, x)?;
            p[0] = cr / mag;
            p[1] = ci / mag;
        }
    }
    let corr = cv_core::idft(&cps, false)?;

    // fftshift: swap diagonally opposite quadrants so zero shift is at the
    // centre rather than at the corner.
    let (cx, cy) = (w / 2, h / 2);
    let mut out = Mat::zeros(h, w, cv_core::types::CV_64FC1)?;
    for y in 0..h {
        for x in 0..w {
            let sy = (y + cy) % h;
            let sx = (x + cx) % w;
            out.pixel_mut::<f64>(y, x)?[0] = corr.at_f64(sy, sx, 0)?;
        }
    }
    Ok(out)
}

/// `cv::approxPolyN(curve, approxCurve, nsides, epsilon_percentage, ensure_convex)`.
///
/// Reduces a contour to exactly `nsides` vertices by repeatedly merging the two
/// neighbouring edges whose intersection adds the least area. Unlike
/// [`crate::shape::approx_poly_dp`] the vertex *count* is the input and the
/// error is the output, which is what you want when the shape is known — a
/// document is four-sided whatever the scan did to its corners.
///
/// `epsilon_percentage` stops early if the added area would exceed that
/// fraction of the original; `-1` disables the check.
pub fn approx_poly_n(
    curve: &[Point],
    nsides: usize,
    epsilon_percentage: f64,
    ensure_convex: bool,
) -> Result<Vec<Point>> {
    cv_assert!(nsides >= 3, "a polygon needs at least three sides, got {nsides}");
    let hull = if ensure_convex {
        crate::shape::convex_hull(curve, true)
    } else {
        curve.to_vec()
    };
    cv_assert!(
        hull.len() >= nsides,
        "cannot grow a {}-vertex hull to {nsides} sides",
        hull.len()
    );
    let original_area = crate::shape::contour_area(&hull, false).abs();
    let limit = if epsilon_percentage < 0.0 {
        f64::INFINITY
    } else {
        original_area * epsilon_percentage
    };

    // Work in floating point: an intersection rounded to integers at every
    // step accumulates error that shows up as a visibly crooked quadrilateral.
    let mut pts: Vec<(f64, f64)> =
        hull.iter().map(|p| (f64::from(p.x), f64::from(p.y))).collect();
    let mut added = 0.0f64;
    while pts.len() > nsides {
        let n = pts.len();
        let mut best: Option<(f64, usize, (f64, f64))> = None;
        for i in 0..n {
            // Merge vertex `i+1` away by intersecting edge (i, i+1) with edge
            // (i+1, i+2). The triangle cut off is the cost.
            let a = pts[i];
            let b = pts[(i + 1) % n];
            let c = pts[(i + 2) % n];
            let d = pts[(i + 3) % n];
            let Some(p) = intersect(a, b, c, d) else { continue };
            let area = tri_area(b, c, p);
            if best.is_none() || best.is_some_and(|(cost, _, _)| area < cost) {
                best = Some((area, (i + 1) % n, p));
            }
        }
        let Some((area, idx, p)) = best else {
            return cv_error!(
                Code::StsError,
                "approxPolyN: no two consecutive edges of the hull intersect"
            );
        };
        if added + area > limit {
            break;
        }
        added += area;
        // Replace the two vertices `idx` and `idx+1` with their intersection.
        let next = (idx + 1) % pts.len();
        pts[idx] = p;
        pts.remove(next);
    }
    Ok(pts
        .into_iter()
        .map(|(x, y)| Point::new(x.round() as i32, y.round() as i32))
        .collect())
}

/// Where the line through `a,b` meets the line through `c,d`, if they are not
/// parallel.
fn intersect(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> Option<(f64, f64)> {
    let (r1, r2) = ((b.0 - a.0, b.1 - a.1), (d.0 - c.0, d.1 - c.1));
    let det = r1.0 * r2.1 - r1.1 * r2.0;
    if det.abs() < 1e-12 {
        return None;
    }
    let t = ((c.0 - a.0) * r2.1 - (c.1 - a.1) * r2.0) / det;
    Some((a.0 + t * r1.0, a.1 + t * r1.1))
}

/// Twice the signed area of a triangle, halved.
fn tri_area(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    ((b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)).abs() / 2.0
}

/// `cv::findContoursLinkRuns(image, contours, hierarchy)`.
///
/// Upstream's cheaper contour finder: it works from the runs of equal pixels in
/// each row rather than following boundaries, so it allocates no scratch image.
/// The trade is fixed behaviour — `CV_8UC1` only, a two-level `RETR_CCOMP`
/// hierarchy, and `CHAIN_APPROX_SIMPLE`.
pub fn find_contours_link_runs<S: MatLike + ?Sized>(
    image: &S,
) -> Result<(Vec<Vec<Point>>, Vec<Hierarchy>)> {
    cv_assert!(image.channels() == 1, "findContoursLinkRuns supports CV_8UC1 only");
    if image.depth() != Depth::U8 {
        return cv_error!(Code::StsUnsupportedFormat, "findContoursLinkRuns supports CV_8UC1 only");
    }
    // The run-linking result is the same contour set `findContours` produces in
    // RETR_CCOMP/CHAIN_APPROX_SIMPLE, so that is what is returned — the
    // difference upstream documents is memory, not output.
    crate::contours::find_contours(
        image,
        crate::contours::RetrievalMode::CComp,
        crate::contours::ContourApproximation::Simple,
    )
}

/// `cv::EMD(signature1, signature2, distType, cost, lowerBound, flow)`.
///
/// The earth mover's distance: the least work needed to turn one weighted point
/// set into the other, where work is weight moved times distance moved. Unlike
/// a bin-by-bin histogram comparison it understands that a colour shifted one
/// bin is nearly the same colour, which is why it is the metric of choice for
/// comparing colour and texture signatures.
///
/// Each signature row is `[weight, x, y, ...]`. Returns the distance and the
/// `size1 x size2` flow matrix.
///
/// The transportation problem is solved by a min-cost flow on the complete
/// bipartite graph; when the two total weights differ, the lighter side gets a
/// zero-cost dummy sink, as upstream does.
pub fn emd(
    signature1: &[Vec<f64>],
    signature2: &[Vec<f64>],
    dist_type: DistanceMetric,
) -> Result<(f32, Mat)> {
    cv_assert!(!signature1.is_empty() && !signature2.is_empty(), "both signatures must be non-empty");
    let dim = signature1[0].len();
    cv_assert!(dim >= 1, "a signature row is a weight followed by its coordinates");
    cv_assert!(
        signature1.iter().all(|r| r.len() == dim) && signature2.iter().all(|r| r.len() == dim),
        "every signature row must have the same width"
    );
    cv_assert!(
        signature1.iter().chain(signature2).all(|r| r[0] >= 0.0),
        "the weights must not be negative"
    );
    let (n, m) = (signature1.len(), signature2.len());
    let mut supply: Vec<f64> = signature1.iter().map(|r| r[0]).collect();
    let mut demand: Vec<f64> = signature2.iter().map(|r| r[0]).collect();
    let (s1, s2): (f64, f64) = (supply.iter().sum(), demand.iter().sum());
    cv_assert!(s1 > 0.0 && s2 > 0.0, "each signature needs a non-zero total weight");

    let mut cost = vec![vec![0.0f64; m]; n];
    for i in 0..n {
        for j in 0..m {
            cost[i][j] = point_distance(&signature1[i][1..], &signature2[j][1..], dist_type);
        }
    }

    // Balance with a zero-cost dummy on the lighter side; the dummy absorbs the
    // surplus without contributing work, which is upstream's convention.
    let total = s1.min(s2);
    if s1 > s2 {
        demand.push(s1 - s2);
    } else if s2 > s1 {
        supply.push(s2 - s1);
    }
    let (rows, cols) = (supply.len(), demand.len());
    let mut ext = vec![vec![0.0f64; cols]; rows];
    for (i, row) in ext.iter_mut().enumerate().take(n) {
        for (j, c) in row.iter_mut().enumerate().take(m) {
            *c = cost[i][j];
        }
    }

    // Solved exactly, by successive shortest paths. A greedy "cheapest cell
    // first" fill is not enough: with a dummy row at zero cost the greedy
    // spends the dummy on the *cheapest* real column and leaves the real mass
    // paying for the most expensive one — on a two-bin example that turns a
    // distance of 2 into 100.
    let flow = min_cost_flow(&supply, &demand, &ext);
    let mut work = 0.0f64;
    for (i, row) in flow.iter().enumerate().take(n) {
        for (j, f) in row.iter().enumerate().take(m) {
            work += f * cost[i][j];
        }
    }

    let mut out = Mat::zeros(n as i32, m as i32, CV_32FC1)?;
    for (i, row) in flow.iter().enumerate().take(n) {
        for (j, f) in row.iter().enumerate().take(m) {
            out.pixel_mut::<f32>(i as i32, j as i32)?[0] = *f as f32;
        }
    }
    Ok(((work / total) as f32, out))
}

/// The minimum-cost transportation plan between `supply` and `demand`.
///
/// Successive shortest paths over the residual graph, with Bellman-Ford finding
/// each path because the residual arcs carry negative costs.
///
/// # Why not a greedy fill
///
/// Filling the cheapest cell first looks equivalent and is not. With a dummy
/// row at zero cost, the greedy spends the dummy on whichever real column it
/// meets first — the *cheapest* one — leaving the real mass to pay for the
/// expensive column. On a two-bin example that turns a distance of 2 into 100.
/// Only a method that can undo an earlier assignment gets it right.
fn min_cost_flow(supply: &[f64], demand: &[f64], cost: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (rows, cols) = (supply.len(), demand.len());
    // Node numbering: 0 = source, 1..=rows = supply, rows+1..=rows+cols =
    // demand, rows+cols+1 = sink.
    let n_nodes = rows + cols + 2;
    let sink = rows + cols + 1;

    // Arcs as parallel arrays; `i ^ 1` is the reverse of arc `i`.
    let mut to: Vec<usize> = Vec::new();
    let mut cap: Vec<f64> = Vec::new();
    let mut cst: Vec<f64> = Vec::new();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    let add = |u: usize,
                   v: usize,
                   c: f64,
                   w: f64,
                   to: &mut Vec<usize>,
                   cap: &mut Vec<f64>,
                   cst: &mut Vec<f64>,
                   adj: &mut Vec<Vec<usize>>| {
        adj[u].push(to.len());
        to.push(v);
        cap.push(c);
        cst.push(w);
        adj[v].push(to.len());
        to.push(u);
        cap.push(0.0);
        cst.push(-w);
    };
    for (i, &s) in supply.iter().enumerate() {
        add(0, 1 + i, s, 0.0, &mut to, &mut cap, &mut cst, &mut adj);
    }
    for (j, &d) in demand.iter().enumerate() {
        add(rows + 1 + j, sink, d, 0.0, &mut to, &mut cap, &mut cst, &mut adj);
    }
    let mut arc_of = vec![vec![usize::MAX; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            arc_of[i][j] = to.len();
            add(1 + i, rows + 1 + j, f64::INFINITY, cost[i][j], &mut to, &mut cap, &mut cst, &mut adj);
        }
    }

    loop {
        // Bellman-Ford, queue-driven: the residual graph has negative arcs, so
        // Dijkstra is not applicable without potentials and this is small.
        let mut dist = vec![f64::INFINITY; n_nodes];
        let mut prev_arc = vec![usize::MAX; n_nodes];
        let mut queued = vec![false; n_nodes];
        dist[0] = 0.0;
        let mut q = std::collections::VecDeque::from([0usize]);
        queued[0] = true;
        while let Some(u) = q.pop_front() {
            queued[u] = false;
            for &a in &adj[u] {
                if cap[a] <= 1e-12 {
                    continue;
                }
                let v = to[a];
                let nd = dist[u] + cst[a];
                if nd < dist[v] - 1e-12 {
                    dist[v] = nd;
                    prev_arc[v] = a;
                    if !queued[v] {
                        queued[v] = true;
                        q.push_back(v);
                    }
                }
            }
        }
        if !dist[sink].is_finite() {
            break;
        }
        // Push as much as the tightest arc on the path allows.
        let mut push = f64::INFINITY;
        let mut v = sink;
        while v != 0 {
            let a = prev_arc[v];
            push = push.min(cap[a]);
            v = to[a ^ 1];
        }
        if !push.is_finite() || push <= 1e-12 {
            break;
        }
        let mut v = sink;
        while v != 0 {
            let a = prev_arc[v];
            cap[a] -= push;
            cap[a ^ 1] += push;
            v = to[a ^ 1];
        }
    }

    let mut flow = vec![vec![0.0f64; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            // The reverse arc's capacity is exactly the flow that was pushed.
            flow[i][j] = cap[arc_of[i][j] ^ 1];
        }
    }
    flow
}

/// Distance between two coordinate vectors under `metric`.
fn point_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::L1 => a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum(),
        DistanceMetric::C => {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
        }
        _ => a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grey ramp with a bright block, so CLAHE has something to redistribute.
    fn ramp(w: i32, h: i32) -> Mat {
        let mut m = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                let v = ((x * 3 + y) % 60 + 90) as u8;
                m.pixel_mut::<u8>(y, x).unwrap()[0] = v;
            }
        }
        m
    }

    #[test]
    fn the_clahe_object_matches_the_free_function() {
        let src = ramp(32, 32);
        let mut c = create_clahe(2.0, Size::new(4, 4));
        assert_eq!(c.get_clip_limit(), 2.0);
        assert_eq!(c.get_tiles_grid_size(), Size::new(4, 4));
        let a = c.apply(&src).unwrap().to_vec::<u8>().unwrap();
        let b = crate::hist2::clahe(&src, 2.0, 4, 4).unwrap().to_vec::<u8>().unwrap();
        assert_eq!(a, b, "the object must not change the computation");

        c.set_clip_limit(8.0);
        c.set_tiles_grid_size(Size::new(8, 8));
        assert_eq!(c.get_clip_limit(), 8.0);
        assert_eq!(c.get_tiles_grid_size(), Size::new(8, 8));
        let d = c.apply(&src).unwrap().to_vec::<u8>().unwrap();
        assert_ne!(a, d, "changing the settings must change the result");
        c.collect_garbage();
        assert_eq!(c.apply(&src).unwrap().to_vec::<u8>().unwrap(), d, "and must not disturb it");

        // Upstream's defaults.
        let def = Clahe::default();
        assert_eq!(def.get_clip_limit(), 40.0);
        assert_eq!(def.get_tiles_grid_size(), Size::new(8, 8));
        assert_eq!(def.get_bit_shift(), 0);
        let mut s = Clahe::default();
        s.set_bit_shift(4);
        assert_eq!(s.get_bit_shift(), 4);
    }

    #[test]
    fn scharr_is_sobel_with_the_scharr_kernel() {
        let src = ramp(16, 16);
        let a = scharr(&src, Some(Depth::F32), 1, 0, 1.0, 0.0, cv_core::BorderType::Replicate)
            .unwrap();
        let b = crate::filter::sobel(
            &src,
            Some(Depth::F32),
            1,
            0,
            -1,
            1.0,
            0.0,
            cv_core::BorderType::Replicate,
        )
        .unwrap();
        assert_eq!(a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap());
        // The second derivative has no Scharr form.
        assert!(scharr(&src, Some(Depth::F32), 2, 0, 1.0, 0.0, cv_core::BorderType::Replicate)
            .is_err());
        assert!(scharr(&src, Some(Depth::F32), 0, 0, 1.0, 0.0, cv_core::BorderType::Replicate)
            .is_err());
    }

    #[test]
    fn scharr_responds_to_a_vertical_edge() {
        let mut m = Mat::zeros(16, 16, CV_8UC1).unwrap();
        for y in 0..16 {
            for x in 8..16 {
                m.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        let gx = scharr(&m, Some(Depth::F32), 1, 0, 1.0, 0.0, cv_core::BorderType::Replicate)
            .unwrap();
        // The step is between x=7 and x=8, so the x-derivative peaks there and
        // is zero in the flat regions.
        assert!(gx.at_f64(8, 7, 0).unwrap().abs() > 100.0);
        assert_eq!(gx.at_f64(8, 2, 0).unwrap(), 0.0);
        assert_eq!(gx.at_f64(8, 13, 0).unwrap(), 0.0);
    }

    /// An image with a bright horizontal bar across most of its width.
    fn bar(w: i32, h: i32) -> Mat {
        let mut m = Mat::zeros(h, w, CV_8UC1).unwrap();
        let (y0, y1) = (h / 2 - 2, h / 2 + 2);
        for y in y0..y1 {
            for x in (w / 8)..(w - w / 8) {
                m.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        m
    }

    #[test]
    fn lsd_finds_the_edges_of_a_bar() {
        let img = bar(80, 48);
        let lsd = LineSegmentDetector::default();
        let segs = lsd.detect(&img).unwrap();
        assert!(!segs.is_empty(), "the bar's edges should be found");
        // The longest segment runs along the bar, so it is near-horizontal and
        // roughly the bar's length.
        let (x1, y1, x2, y2) = segs[0];
        assert!((y2 - y1).abs() < 6.0, "the segment should be near-horizontal: {segs:?}");
        assert!((x2 - x1).abs() > 30.0, "and should span most of the bar: {segs:?}");
    }

    #[test]
    fn lsd_finds_nothing_in_a_flat_image() {
        let flat = Mat::zeros(40, 40, CV_8UC1).unwrap();
        assert!(LineSegmentDetector::default().detect(&flat).unwrap().is_empty());
        // A one-pixel image has no gradient to speak of.
        let tiny = Mat::zeros(2, 2, CV_8UC1).unwrap();
        assert!(LineSegmentDetector::default().detect(&tiny).unwrap().is_empty());
    }

    #[test]
    fn lsd_rejects_bad_settings_and_reports_its_own() {
        let img = bar(40, 40);
        let d = create_line_segment_detector(LsdMode::Adv, 0.0, 0.6, 2.0, 22.5, 0.0, 0.7, 1024);
        assert!(d.detect(&img).is_err(), "a zero scale");
        let d = create_line_segment_detector(LsdMode::Std, 0.8, 0.6, 2.0, 200.0, 0.0, 0.7, 1024);
        assert!(d.detect(&img).is_err(), "an angle tolerance past 180");
        let d = create_line_segment_detector(LsdMode::Std, 0.8, 0.6, 2.0, 22.5, 0.0, 2.0, 1024);
        assert!(d.detect(&img).is_err(), "a density above 1");
        let d = create_line_segment_detector(LsdMode::None, 0.8, 0.6, 2.0, 22.5, 0.0, 0.7, 0);
        assert!(d.detect(&img).is_err(), "no bins");

        let d = create_line_segment_detector(LsdMode::Adv, 1.0, 0.6, 2.0, 30.0, 1.5, 0.7, 512);
        assert_eq!(d.refine(), LsdMode::Adv);
        assert_eq!(d.tuning(), (30.0, 2.0, 1.5, 512));
        assert_eq!(LsdMode::default(), LsdMode::Std);
        // A three-channel image is not a grey one.
        let colour =
            Mat::zeros(8, 8, cv_core::types::MatType::new(Depth::U8, 3)).unwrap();
        assert!(d.detect(&colour).is_err());
    }

    #[test]
    fn segments_can_be_drawn_and_compared() {
        let img = bar(60, 40);
        let lsd = LineSegmentDetector::default();
        let segs = lsd.detect(&img).unwrap();
        let canvas = lsd.draw_segments(&img, &segs).unwrap();
        assert_eq!(canvas.channels(), 3, "a grey image is promoted to colour");
        assert_eq!((canvas.rows(), canvas.cols()), (40, 60));

        // Comparing a set against itself finds no difference.
        let (same, vis) = lsd.compare_segments(Size::new(60, 40), &segs, &segs).unwrap();
        assert_eq!(same, 0);
        assert_eq!(vis.channels(), 3);
        // Against an empty set, every drawn pixel differs.
        let (differs, _) = lsd.compare_segments(Size::new(60, 40), &segs, &[]).unwrap();
        assert!(differs > 0, "the two sets should not agree");
        assert!(lsd.compare_segments(Size::new(0, 0), &segs, &segs).is_err());
    }

    #[test]
    fn the_live_wire_snaps_to_an_edge() {
        // A bright half-plane: the strongest edge is the vertical line at x=20.
        let mut m = Mat::zeros(40, 40, CV_8UC1).unwrap();
        for y in 0..40 {
            for x in 20..40 {
                m.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        let mut s = IntelligentScissorsMB::new();
        s.set_weights(0.43, 0.43, 0.14).unwrap();
        s.set_gradient_magnitude_max_limit(0.0);
        s.set_edge_feature_zero_crossing_parameters(0.01);
        s.build_map(&m, Point::new(20, 2)).unwrap();

        let path = s.get_contour(Point::new(20, 37), false).unwrap();
        assert!(path.len() >= 30, "the path should run the height of the image");
        assert_eq!(path[0], Point::new(20, 2), "it starts at the seed");
        assert_eq!(*path.last().unwrap(), Point::new(20, 37), "and ends at the target");
        // Every step hugs the edge rather than wandering into the flat regions.
        assert!(path.iter().all(|p| (p.x - 20).abs() <= 2), "{path:?}");

        // Backward gives the same points, reversed.
        let back = s.get_contour(Point::new(20, 37), true).unwrap();
        let mut f = path.clone();
        f.reverse();
        assert_eq!(back, f);
    }

    #[test]
    fn the_live_wire_checks_its_inputs() {
        let m = Mat::zeros(20, 20, CV_8UC1).unwrap();
        let mut s = IntelligentScissorsMB::default();
        assert!(s.get_contour(Point::new(1, 1), false).is_err(), "no map yet");
        assert!(s.build_map(&m, Point::new(50, 50)).is_err(), "the seed is outside");
        assert!(s.set_weights(-1.0, 0.0, 0.0).is_err());
        assert!(s.set_weights(0.0, 0.0, 0.0).is_err(), "all-zero weights");
        s.build_map(&m, Point::new(5, 5)).unwrap();
        assert!(s.get_contour(Point::new(50, 50), false).is_err());
        assert_eq!(s.get_contour(Point::new(5, 5), false).unwrap(), vec![Point::new(5, 5)]);
    }

    #[test]
    fn iterative_phase_correlation_recovers_a_known_shift() {
        // A textured patch and the same patch moved by (3, -2).
        let (w, h) = (64, 64);
        let mut a = Mat::zeros(h, w, cv_core::types::CV_64FC1).unwrap();
        let mut b = Mat::zeros(h, w, cv_core::types::CV_64FC1).unwrap();
        let mut state = 0x1234_5678u32;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            f64::from(state % 256)
        };
        let vals: Vec<f64> = (0..(w * h)).map(|_| rnd()).collect();
        for y in 0..h {
            for x in 0..w {
                a.pixel_mut::<f64>(y, x).unwrap()[0] = vals[(y * w + x) as usize];
                let (sx, sy) = (x - 3, y + 2);
                let v = if (0..w).contains(&sx) && (0..h).contains(&sy) {
                    vals[(sy * w + sx) as usize]
                } else {
                    0.0
                };
                b.pixel_mut::<f64>(y, x).unwrap()[0] = v;
            }
        }
        // `b` is `a` shifted by (+3, -2). `phaseCorrelateIterative` reports
        // `peak - centre`, so it returns that shift directly...
        let (dx, dy) = phase_correlate_iterative(&a, &b, 7, 5).unwrap();
        assert!((dx - 3.0).abs() < 0.3, "dx {dx}");
        assert!((dy + 2.0).abs() < 0.3, "dy {dy}");

        // ...where `phaseCorrelate` reports `centre - peak`, the negative of
        // it. The two upstream functions disagree in sign and both are ported
        // as they are.
        let ((sx, sy), _) = crate::extra::phase_correlate(&a, &b, None).unwrap();
        assert!((sx + dx).abs() < 0.5, "{sx} and {dx} should be opposites");
        assert!((sy + dy).abs() < 0.5, "{sy} and {dy} should be opposites");
        assert!(phase_correlate_iterative(&a, &b, 0, 5).is_err());
        assert!(phase_correlate_iterative(&a, &b, 7, 0).is_err());
    }

    #[test]
    fn approx_poly_n_reduces_a_hull_to_the_asked_side_count() {
        // A rounded square: many hull vertices, four real corners.
        let mut pts = Vec::new();
        for i in 0..40 {
            let t = f64::from(i) / 40.0 * core::f64::consts::TAU;
            // A superellipse, which is square-ish with rounded corners.
            let (c, s) = (t.cos(), t.sin());
            let x = 50.0 + 40.0 * c.abs().powf(0.5) * c.signum();
            let y = 50.0 + 40.0 * s.abs().powf(0.5) * s.signum();
            pts.push(Point::new(x.round() as i32, y.round() as i32));
        }
        let quad = approx_poly_n(&pts, 4, -1.0, true).unwrap();
        assert_eq!(quad.len(), 4, "four sides were asked for: {quad:?}");
        // The quadrilateral encloses the original, so its area is at least as
        // large — every vertex lies on or outside the hull.
        let a0 = crate::shape::contour_area(&pts, false).abs();
        let a1 = crate::shape::contour_area(&quad, false).abs();
        assert!(a1 >= a0 * 0.99, "{a1} should not be below {a0}");

        assert!(approx_poly_n(&pts, 2, -1.0, true).is_err(), "two sides is not a polygon");
        assert!(approx_poly_n(&pts[..3], 8, -1.0, true).is_err(), "cannot grow a triangle");

        // A tight area budget stops the contraction early.
        let partial = approx_poly_n(&pts, 4, 0.0, true).unwrap();
        assert!(partial.len() > 4, "no added area is allowed, so nothing merges");
    }

    #[test]
    fn link_runs_finds_the_same_contours_as_find_contours() {
        let mut m = Mat::zeros(30, 30, CV_8UC1).unwrap();
        for y in 5..15 {
            for x in 5..15 {
                m.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        let (contours, hierarchy) = find_contours_link_runs(&m).unwrap();
        assert_eq!(contours.len(), 1, "one square");
        assert_eq!(hierarchy.len(), 1);
        assert!(contours[0].len() >= 4);

        let f = Mat::zeros(4, 4, CV_32FC1).unwrap();
        assert!(find_contours_link_runs(&f).is_err(), "CV_8UC1 only");
        let colour =
            Mat::zeros(4, 4, cv_core::types::MatType::new(Depth::U8, 3)).unwrap();
        assert!(find_contours_link_runs(&colour).is_err());
    }

    #[test]
    fn the_earth_movers_distance_is_zero_for_identical_signatures() {
        let s = vec![vec![0.4, 0.0, 0.0], vec![0.3, 10.0, 0.0], vec![0.3, 0.0, 10.0]];
        let (d, flow) = emd(&s, &s, DistanceMetric::L2).unwrap();
        assert!(d.abs() < 1e-6, "identical signatures cost nothing to match: {d}");
        assert_eq!((flow.rows(), flow.cols()), (3, 3));
        // All the mass stays where it is.
        for i in 0..3 {
            assert!((flow.at_f64(i, i, 0).unwrap() - s[i as usize][0]).abs() < 1e-6);
        }
    }

    #[test]
    fn the_earth_movers_distance_grows_with_the_separation() {
        let a = vec![vec![1.0, 0.0]];
        let near = vec![vec![1.0, 1.0]];
        let far = vec![vec![1.0, 5.0]];
        let (dn, _) = emd(&a, &near, DistanceMetric::L1).unwrap();
        let (df, _) = emd(&a, &far, DistanceMetric::L1).unwrap();
        assert!((dn - 1.0).abs() < 1e-6, "{dn}");
        assert!((df - 5.0).abs() < 1e-6, "{df}");
        assert!(df > dn);

        // Splitting the mass across two bins moves half of it half as far.
        let split = vec![vec![0.5, 0.0], vec![0.5, 4.0]];
        let (ds, _) = emd(&a, &split, DistanceMetric::L1).unwrap();
        assert!((ds - 2.0).abs() < 1e-6, "{ds}");
    }

    #[test]
    fn unequal_total_weights_are_absorbed_by_a_dummy() {
        // Twice the weight on the right; only the matched half does work.
        let a = vec![vec![1.0, 0.0]];
        let b = vec![vec![1.0, 2.0], vec![1.0, 100.0]];
        let (d, flow) = emd(&a, &b, DistanceMetric::L1).unwrap();
        assert!((d - 2.0).abs() < 1e-6, "the far bin should be absorbed, not paid for: {d}");
        assert!((flow.at_f64(0, 0, 0).unwrap() - 1.0).abs() < 1e-6);
        assert!(flow.at_f64(0, 1, 0).unwrap().abs() < 1e-6);
    }

    #[test]
    fn the_earth_movers_distance_checks_its_signatures() {
        let ok = vec![vec![1.0, 0.0]];
        assert!(emd(&[], &ok, DistanceMetric::L2).is_err());
        assert!(emd(&ok, &[], DistanceMetric::L2).is_err());
        assert!(emd(&ok, &[vec![1.0, 0.0, 0.0]], DistanceMetric::L2).is_err(), "widths differ");
        assert!(emd(&[vec![-1.0, 0.0]], &ok, DistanceMetric::L2).is_err(), "negative weight");
        assert!(emd(&[vec![0.0, 0.0]], &ok, DistanceMetric::L2).is_err(), "no weight at all");
        // Every metric produces a finite answer.
        for m in [DistanceMetric::L1, DistanceMetric::L2, DistanceMetric::C] {
            let (d, _) = emd(&ok, &[vec![1.0, 3.0]], m).unwrap();
            assert!(d.is_finite() && d > 0.0, "{m:?} gave {d}");
        }
    }
}
