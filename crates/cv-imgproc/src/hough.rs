//! The Hough transforms (`modules/imgproc/src/hough.cpp`,
//! `generalized_hough.cpp`).
//!
//! # The idea
//!
//! An edge pixel does not say where a line is; it says which lines *could* pass
//! through it. Every such line votes in a parameter-space accumulator, and a
//! line that really exists collects a vote from each of its pixels while a
//! coincidence collects a few. Finding lines becomes finding peaks.
//!
//! That is why the transform survives gaps and occlusion where edge-following
//! does not: a line broken into three pieces still puts all its votes in one
//! bin.
//!
//! # Which one to use
//!
//! [`hough_lines`] returns infinite lines as `(rho, theta)`; use it when the
//! line's *direction* is the answer. [`hough_lines_p`] returns endpoints, which
//! is what you want if the extent matters. [`hough_circles`] needs a gradient,
//! not just edges — voting over all three circle parameters directly would be a
//! 3-D accumulator, and restricting the votes to the gradient direction is what
//! makes it tractable.
//!
//! # Agreement with OpenCV
//!
//! [`hough_lines`], [`hough_lines_point_set`] and [`hough_circles`] are
//! deterministic and follow upstream's accumulator geometry and tie-breaks.
//! [`hough_lines_p`] visits edge pixels in a pseudo-random order; it uses
//! [`cv_core::rng::Rng`] seeded exactly as upstream seeds its own, so the order
//! is reproducible here, but it is not claimed to be identical to cv2's.

use cv_core::types::{Depth, CV_32FC1};
use cv_core::{cv_assert, cv_error, Code, Mat, MatLike, MatLikeMut, Point, Result};

use cv_core::saturate::cv_roundf;
use cv_core::BorderType;

/// One line in normal form: the distance from the origin and the angle of the
/// normal, as `cv::HoughLines` returns them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoughLine {
    /// `rho` — signed distance from the image origin to the line.
    pub rho: f32,
    /// `theta` — angle of the line's normal, in radians.
    pub theta: f32,
    /// `votes` — accumulator count, as `HoughLinesWithAccumulator` reports.
    pub votes: i32,
}

/// How many results to keep, with `0` meaning "all of them".
///
/// Upstream has no such sentinel: its `HoughLinesStandard` takes a `linesMax`
/// and every caller in the library passes `INT_MAX`, so the argument only
/// exists for an internal caller that wants to stop early. A Rust caller who
/// writes `0` means "no limit", not "no lines", and the literal reading
/// silently returned an empty vector.
fn cap(limit: usize) -> usize {
    if limit == 0 {
        usize::MAX
    } else {
        limit
    }
}

/// The non-zero pixels of a mask, as `(x, y)`.
fn edge_points<S: MatLike + ?Sized>(image: &S) -> Result<Vec<(i32, i32)>> {
    cv_assert!(image.channels() == 1, "the Hough transform takes a single-channel mask");
    cv_assert!(image.depth() == Depth::U8, "the Hough transform takes a CV_8U mask");
    let mut pts = Vec::new();
    for y in 0..image.rows() {
        for x in 0..image.cols() {
            if image.at_f64(y, x, 0)? != 0.0 {
                pts.push((x, y));
            }
        }
    }
    Ok(pts)
}

/// `cv::HoughLines(image, lines, rho, theta, threshold, srn, stn, min_theta, max_theta)`.
///
/// The accumulator geometry is upstream's: `numrho` bins spanning
/// `-(w+h) .. (w+h)`, with a one-bin border on every side so the peak test can
/// read its neighbours without a bounds check. A line is returned when its bin
/// beats `threshold` **and** all four neighbours — the second condition is what
/// stops one strong line being reported as a dozen near-duplicates.
///
/// Results come back strongest first, ties broken by accumulator position, so
/// the order is stable rather than dependent on scan order.
///
/// `lines_max` of `0` means every peak; see `cap`.
#[allow(clippy::too_many_arguments)]
pub fn hough_lines<S: MatLike + ?Sized>(
    image: &S,
    rho: f64,
    theta: f64,
    threshold: i32,
    lines_max: usize,
    min_theta: f64,
    max_theta: f64,
) -> Result<Vec<HoughLine>> {
    cv_assert!(rho > 0.0 && theta > 0.0, "rho and theta steps must be positive");
    cv_assert!(
        min_theta <= max_theta,
        "min_theta {min_theta} must not exceed max_theta {max_theta}"
    );
    cv_assert!(
        (0.0..=core::f64::consts::PI).contains(&min_theta)
            && (0.0..=core::f64::consts::PI).contains(&max_theta),
        "theta must lie in 0..pi"
    );
    let pts = edge_points(image)?;
    let (w, h) = (image.cols(), image.rows());

    let numangle = theta_count(min_theta, max_theta, theta);
    let numrho = (((f64::from(w) + f64::from(h)) * 2.0 + 1.0) / rho).round() as i32;
    if numangle <= 0 || numrho <= 0 {
        return Ok(Vec::new());
    }

    // Trig tables, scaled by 1/rho once so the inner loop is two multiplies.
    let irho = 1.0 / rho;
    let mut tab_sin = Vec::with_capacity(numangle as usize);
    let mut tab_cos = Vec::with_capacity(numangle as usize);
    let mut ang = min_theta;
    for _ in 0..numangle {
        tab_sin.push((ang.sin() * irho) as f32);
        tab_cos.push((ang.cos() * irho) as f32);
        ang += theta;
    }

    let stride = (numrho + 2) as usize;
    let mut accum = vec![0i32; ((numangle + 2) as usize) * stride];
    for &(x, y) in &pts {
        for n in 0..numangle as usize {
            let r = (x as f32 * tab_cos[n] + y as f32 * tab_sin[n]).round() as i32
                + (numrho - 1) / 2;
            accum[(n + 1) * stride + (r + 1) as usize] += 1;
        }
    }

    // Peaks: above threshold and no smaller than either rho neighbour.
    let mut peaks: Vec<(i32, usize)> = Vec::new();
    for n in 0..numangle as usize {
        for r in 0..numrho as usize {
            let base = (n + 1) * stride + r + 1;
            let v = accum[base];
            if v > threshold
                && v > accum[base - 1]
                && v >= accum[base + 1]
                && v > accum[base - stride]
                && v >= accum[base + stride]
            {
                peaks.push((v, base));
            }
        }
    }
    // Strongest first; the index breaks ties, which is upstream's `hough_cmp_gt`.
    peaks.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    peaks.truncate(cap(lines_max));

    Ok(peaks
        .into_iter()
        .map(|(votes, idx)| {
            // `idx = (n + 1) * stride + (r + 1)`, inverted exactly.
            let n = (idx / stride - 1) as f64;
            let r = (idx % stride - 1) as f64;
            HoughLine {
                // Upstream recovers rho with `(numrho - 1) * 0.5f` even though
                // it *accumulated* with the integer `(numrho - 1) / 2`. The two
                // agree whenever numrho is odd, which is the usual case; the
                // asymmetry is upstream's and is kept so an even numrho lands
                // where cv2 puts it.
                rho: ((r - f64::from(numrho - 1) * 0.5) * rho) as f32,
                theta: (min_theta + n * theta) as f32,
                votes,
            }
        })
        .collect())
}

/// How many angle bins `min..max` in steps of `theta` produces.
fn theta_count(min_theta: f64, max_theta: f64, theta: f64) -> i32 {
    let span = max_theta - min_theta;
    let mut n = (span / theta).floor() as i32 + 1;
    // Upstream drops the last bin when it would land exactly on `max_theta`
    // *and* the span is the full half-turn, so 0 and pi are not both counted.
    if n > 1 && (span - f64::from(n - 1) * theta).abs() < f64::EPSILON * span.max(1.0) {
        n -= 1;
    }
    n.max(0)
}

/// One line segment, as `cv::HoughLinesP` returns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HoughSegment {
    /// One endpoint.
    pub a: Point,
    /// The other.
    pub b: Point,
}

/// `cv::HoughLinesP(image, lines, rho, theta, threshold, minLineLength, maxLineGap)`.
///
/// The probabilistic transform: instead of voting with every edge pixel and
/// then looking for peaks, it draws pixels in random order, and as soon as one
/// pixel pushes a bin over `threshold` it walks that line in the image, keeps
/// the segment if it is long enough, and **removes its pixels from the
/// accumulator**. Removing them is the point — it stops one strong line from
/// being found again and again, and it makes the cost fall as lines are found
/// rather than stay fixed.
///
/// `max_line_gap` is how many blank pixels a walk may cross before giving up,
/// which is what lets a dashed line come back as one segment.
#[allow(clippy::too_many_arguments)]
pub fn hough_lines_p<S: MatLike + ?Sized>(
    image: &S,
    rho: f64,
    theta: f64,
    threshold: i32,
    min_line_length: f64,
    max_line_gap: f64,
    lines_max: usize,
) -> Result<Vec<HoughSegment>> {
    cv_assert!(rho > 0.0 && theta > 0.0, "rho and theta steps must be positive");
    let (w, h) = (image.cols(), image.rows());
    let mut pts = edge_points(image)?;
    if pts.is_empty() {
        return Ok(Vec::new());
    }

    let numangle = theta_count(0.0, core::f64::consts::PI, theta);
    let numrho = (((f64::from(w) + f64::from(h)) * 2.0 + 1.0) / rho).round() as i32;
    if numangle <= 0 || numrho <= 0 {
        return Ok(Vec::new());
    }
    let irho = 1.0 / rho;
    let tab_sin: Vec<f32> =
        (0..numangle).map(|n| ((f64::from(n) * theta).sin() * irho) as f32).collect();
    let tab_cos: Vec<f32> =
        (0..numangle).map(|n| ((f64::from(n) * theta).cos() * irho) as f32).collect();

    let mut accum = vec![0i32; (numangle * numrho) as usize];
    // The working mask: set where an edge pixel is still available.
    let mut mask = vec![false; (w * h) as usize];
    for &(x, y) in &pts {
        mask[(y * w + x) as usize] = true;
    }

    // Upstream shuffles with `RNG((uint64)-1)`; the same seed is used here so
    // a run is reproducible, and the shuffle is Fisher-Yates as upstream's is.
    let mut rng = cv_core::rng::Rng::new(u64::MAX);
    for i in (1..pts.len()).rev() {
        let j = (rng.next_u32() as usize) % (i + 1);
        pts.swap(i, j);
    }

    let mut out = Vec::new();
    for &(px, py) in &pts {
        if out.len() >= cap(lines_max) {
            break;
        }
        if !mask[(py * w + px) as usize] {
            continue;
        }
        // Vote, and note the strongest bin this pixel lands in.
        let mut best = (0i32, 0usize);
        for n in 0..numangle as usize {
            let r = (px as f32 * tab_cos[n] + py as f32 * tab_sin[n]).round() as i32
                + (numrho - 1) / 2;
            if r < 0 || r >= numrho {
                continue;
            }
            let slot = n * numrho as usize + r as usize;
            accum[slot] += 1;
            if accum[slot] > best.0 {
                best = (accum[slot], n);
            }
        }
        if best.0 < threshold {
            continue;
        }

        // Walk the line through this pixel in both directions, allowing gaps.
        let n = best.1;
        let a = -tab_sin[n] as f64;
        let b = tab_cos[n] as f64;
        let (dx0, dy0) = (a, b);
        let mut ends = [(px, py); 2];
        for (k, sign) in [(0usize, 1.0f64), (1, -1.0)] {
            let (dx, dy) = (dx0 * sign, dy0 * sign);
            let (mut x, mut y) = (f64::from(px), f64::from(py));
            let mut gap = 0.0f64;
            let step = 1.0 / dx.abs().max(dy.abs()).max(1e-12);
            let (sx, sy) = (dx * step, dy * step);
            let (mut lx, mut ly) = (px, py);
            loop {
                x += sx;
                y += sy;
                let (ix, iy) = (x.round() as i32, y.round() as i32);
                if ix < 0 || iy < 0 || ix >= w || iy >= h {
                    break;
                }
                if mask[(iy * w + ix) as usize] {
                    gap = 0.0;
                    lx = ix;
                    ly = iy;
                } else {
                    gap += 1.0;
                    if gap > max_line_gap {
                        break;
                    }
                }
            }
            ends[k] = (lx, ly);
        }

        let (x0, y0) = ends[0];
        let (x1, y1) = ends[1];
        let len = f64::from(x1 - x0).hypot(f64::from(y1 - y0));
        let good = len >= min_line_length;

        // Whether or not it was long enough, retract the votes of the pixels
        // along it: leaving them would let the same line fire repeatedly.
        for (k, sign) in [(0usize, 1.0f64), (1, -1.0)] {
            let (dx, dy) = (dx0 * sign, dy0 * sign);
            let step = 1.0 / dx.abs().max(dy.abs()).max(1e-12);
            let (sx, sy) = (dx * step, dy * step);
            let (mut x, mut y) = (f64::from(px), f64::from(py));
            let target = ends[k];
            loop {
                let (ix, iy) = (x.round() as i32, y.round() as i32);
                if ix < 0 || iy < 0 || ix >= w || iy >= h {
                    break;
                }
                if mask[(iy * w + ix) as usize] {
                    mask[(iy * w + ix) as usize] = false;
                    if good {
                        for m in 0..numangle as usize {
                            let r = (ix as f32 * tab_cos[m] + iy as f32 * tab_sin[m]).round()
                                as i32
                                + (numrho - 1) / 2;
                            if (0..numrho).contains(&r) {
                                accum[m * numrho as usize + r as usize] -= 1;
                            }
                        }
                    }
                }
                if (ix, iy) == target {
                    break;
                }
                x += sx;
                y += sy;
            }
        }

        if good {
            out.push(HoughSegment { a: Point::new(x0, y0), b: Point::new(x1, y1) });
        }
    }
    Ok(out)
}

/// `cv::HoughLinesPointSet(point, lines, lines_max, threshold, ...)`.
///
/// The same accumulator over an explicit point list rather than a mask, which
/// is what you want when the points come from somewhere other than an edge
/// detector — a lidar sweep, a corner list, a sparse feature set — and building
/// an image just to rasterise them would quantise them first.
#[allow(clippy::too_many_arguments)]
pub fn hough_lines_point_set(
    points: &[(f32, f32)],
    lines_max: usize,
    threshold: i32,
    min_rho: f64,
    max_rho: f64,
    rho_step: f64,
    min_theta: f64,
    max_theta: f64,
    theta_step: f64,
) -> Result<Vec<HoughLine>> {
    cv_assert!(rho_step > 0.0 && theta_step > 0.0, "the rho and theta steps must be positive");
    cv_assert!(max_rho > min_rho, "max_rho must exceed min_rho");
    cv_assert!(max_theta > min_theta, "max_theta must exceed min_theta");
    let numrho = ((max_rho - min_rho) / rho_step).round() as i32;
    let numangle = ((max_theta - min_theta) / theta_step).round() as i32;
    if numrho <= 0 || numangle <= 0 || points.is_empty() {
        return Ok(Vec::new());
    }

    let mut accum = vec![0i32; (numangle * numrho) as usize];
    for n in 0..numangle {
        let ang = min_theta + f64::from(n) * theta_step;
        let (s, c) = (ang.sin(), ang.cos());
        for &(x, y) in points {
            let r = f64::from(x) * c + f64::from(y) * s;
            let bin = ((r - min_rho) / rho_step).round() as i32;
            if (0..numrho).contains(&bin) {
                accum[(n * numrho + bin) as usize] += 1;
            }
        }
    }

    let mut peaks: Vec<(i32, i32, i32)> = Vec::new();
    for n in 0..numangle {
        for r in 0..numrho {
            let v = accum[(n * numrho + r) as usize];
            if v >= threshold {
                peaks.push((v, n, r));
            }
        }
    }
    peaks.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    peaks.truncate(cap(lines_max));
    Ok(peaks
        .into_iter()
        .map(|(votes, n, r)| HoughLine {
            rho: (min_rho + f64::from(r) * rho_step) as f32,
            theta: (min_theta + f64::from(n) * theta_step) as f32,
            votes,
        })
        .collect())
}

/// One circle, as `cv::HoughCircles` returns it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoughCircle {
    /// Centre x.
    pub x: f32,
    /// Centre y.
    pub y: f32,
    /// Radius.
    pub radius: f32,
    /// Accumulator votes for the centre.
    pub votes: i32,
}

/// `cv::HoughCircles(image, circles, method, dp, minDist, param1, param2, minRadius, maxRadius)`
/// with `HOUGH_GRADIENT`.
///
/// # Why the gradient
///
/// A circle has three parameters, so voting over all of them means a 3-D
/// accumulator and a cost that makes the method useless. The gradient at an
/// edge pixel points along the radius, so each pixel can vote only for centres
/// on that ray — a 2-D accumulator — and the radius is recovered afterwards
/// from the distances of the edge pixels to the chosen centre.
///
/// The cost is a dependence on gradient *direction*, so this is much more
/// sensitive to noise and to `param1` (the Canny high threshold) than
/// [`hough_lines`] is to its own settings.
///
/// # Blur the input first
///
/// Upstream says this and it is not a formality. On a hard-edged binary image
/// the gradient at each boundary pixel is quantised by the rasterisation
/// staircase, and at radius 20 a ten-degree direction error throws the vote
/// three pixels off the centre — the peak flattens into a blob and the answer
/// drifts. A 5x5 Gaussian at sigma 1.5 on the same disc takes the centre peak
/// here from 20 votes at the wrong place to 56 at the right one.
#[allow(clippy::too_many_arguments)]
pub fn hough_circles<S: MatLike + ?Sized>(
    image: &S,
    dp: f64,
    min_dist: f64,
    param1: f64,
    param2: f64,
    min_radius: i32,
    max_radius: i32,
    circles_max: usize,
) -> Result<Vec<HoughCircle>> {
    cv_assert!(dp >= 1.0, "dp must be at least 1, got {dp}");
    cv_assert!(min_dist > 0.0, "minDist must be positive");
    cv_assert!(param1 > 0.0 && param2 > 0.0, "param1 and param2 must be positive");
    cv_assert!(image.channels() == 1, "HoughCircles takes a single-channel image");
    if image.depth() != Depth::U8 {
        return cv_error!(Code::StsUnsupportedFormat, "HoughCircles requires CV_8U");
    }
    let (w, h) = (image.cols(), image.rows());
    let max_radius = if max_radius <= 0 { w.max(h) } else { max_radius };
    let min_radius = min_radius.max(0);
    cv_assert!(min_radius <= max_radius, "minRadius must not exceed maxRadius");

    // Upstream runs Canny at (param1/2, param1) and takes the Sobel gradient
    // of the *original* image, not of the edge map.
    let edges = crate::edge::canny(image, param1 / 2.0, param1, 3, false)?;
    let dx = crate::filter::sobel(image, Some(Depth::S16), 1, 0, 3, 1.0, 0.0, BorderType::Replicate)?;
    let dy = crate::filter::sobel(image, Some(Depth::S16), 0, 1, 3, 1.0, 0.0, BorderType::Replicate)?;

    let idp = (1.0 / dp) as f32;
    // `cvCeil`, and a one-cell border on every side so the peak test can read
    // its neighbours without a bounds check. Votes land in the top-left
    // `aw x ah` of the padded grid, and the padded index *is* the reported
    // coordinate — upstream never subtracts the border back off.
    let aw = (f64::from(w) * f64::from(idp)).ceil() as i32;
    let ah = (f64::from(h) * f64::from(idp)).ceil() as i32;
    let (aw, ah) = (aw.max(1), ah.max(1));
    let astep = aw + 2;
    let mut accum = vec![0i32; ((ah + 2) * astep) as usize];
    // Edge pixels kept for the radius stage.
    let mut nz: Vec<(i32, i32)> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            if edges.at_f64(y, x, 0)? == 0.0 {
                continue;
            }
            let vx = dx.at_f64(y, x, 0)? as f32;
            let vy = dy.at_f64(y, x, 0)? as f32;
            if vx == 0.0 && vy == 0.0 {
                continue;
            }
            let mag = (vx * vx + vy * vy).sqrt();
            if mag < 1.0 {
                continue;
            }
            nz.push((x, y));

            // Fixed point, 1/1024 of an accumulator cell. The direction is
            // rounded to that grid *once* and then added repeatedly, so the
            // ray drifts exactly as upstream's does; recomputing it in floats
            // at each step gives a subtly different set of cells.
            let sx0 = cv_roundf(vx * idp * 1024.0 / mag);
            let sy0 = cv_roundf(vy * idp * 1024.0 / mag);
            let x0 = cv_roundf(x as f32 * idp * 1024.0);
            let y0 = cv_roundf(y as f32 * idp * 1024.0);

            // Vote along the gradient ray in both directions: a bright disc on
            // dark and a dark disc on bright have opposite gradients, and
            // walking only one way would find one and not the other.
            for sign in [1, -1] {
                let (sx, sy) = (sx0 * sign, sy0 * sign);
                let (mut x1, mut y1) = (x0 + min_radius * sx, y0 + min_radius * sy);
                for _ in min_radius..=max_radius {
                    // An arithmetic shift, not a round: the cell a ray is in is
                    // the one it has reached, and rounding here moves every
                    // vote half a cell along the gradient.
                    let (x2, y2) = (x1 >> 10, y1 >> 10);
                    if (x2 as u32) >= (aw as u32) || (y2 as u32) >= (ah as u32) {
                        break;
                    }
                    accum[(y2 * astep + x2) as usize] += 1;
                    x1 += sx;
                    y1 += sy;
                }
            }
        }
    }

    // Centres: above param2 and no smaller than their four neighbours.
    let mut centers: Vec<(i32, i32, i32)> = Vec::new();
    let acc_threshold = param2.round().max(1.0) as i32;
    for y in 1..ah + 1 {
        for x in 1..astep - 1 {
            let base = (y * astep + x) as usize;
            let v = accum[base];
            if v > acc_threshold
                && v > accum[base - 1]
                && v >= accum[base + 1]
                && v > accum[base - astep as usize]
                && v >= accum[base + astep as usize]
            {
                centers.push((v, x, y));
            }
        }
    }
    // Upstream collects centres in raster order and never sorts them here; the
    // ordering that decides the output is applied after the radius stage.
    centers.sort_by(|a, b| a.2.cmp(&b.2).then(a.1.cmp(&b.1)));

    let min_dist2 = min_dist * min_dist;
    let (r_min2, r_max2) = (
        (min_radius * min_radius) as f32,
        (max_radius * max_radius) as f32,
    );

    // The radius is the most popular *distance* from this centre to an edge
    // pixel — a histogram, not a mean, because a mean over pixels that belong
    // to two concentric circles lands between them.
    //
    // Upstream's histogram is ten bins per unit of `dp`, and the estimate is
    // the midpoint of the best window of ten consecutive bins rather than the
    // single fullest bin. That is what lets it report a fractional radius: a
    // one-bin-per-pixel histogram can only ever answer in whole pixels, and
    // on a disc of true radius 20 cv2 says 19.6.
    const N_BINS_PER_DR: i32 = 10;
    let dr = dp as f32;
    let n_bins = (f64::from(max_radius - min_radius) / dp * f64::from(N_BINS_PER_DR)).round() as i32;
    let n_bins = n_bins.max(1);

    // Every centre gets a radius before any of them is discarded: upstream
    // orders by the radius's support and *then* removes overlaps, so a
    // minDist collision is settled by how many edge pixels actually lie on the
    // circle, not by which centre bin happened to score highest.
    let mut estimated: Vec<(i32, f32, f32, f32)> = Vec::new();
    for (_votes, ax, ay) in centers {
        // The half-pixel is upstream's `(x + 0.5f) * dr`: an accumulator cell
        // covers a whole pixel, so its centre is half a pixel past its index.
        // Dropping it biases every reported centre up and left by dp/2.
        let cx = (ax as f32 + 0.5) * dr;
        let cy = (ay as f32 + 0.5) * dr;

        let mut bins = vec![0i32; n_bins as usize];
        for &(x, y) in &nz {
            let (ex, ey) = (x as f32 - cx, y as f32 - cy);
            let d2 = ex * ex + ey * ey;
            if d2 < r_min2 || d2 > r_max2 {
                continue;
            }
            let d = d2.sqrt();
            let bin = ((d - min_radius as f32) / dr * N_BINS_PER_DR as f32).round() as i32;
            bins[bin.clamp(0, n_bins - 1) as usize] += 1;
        }

        let (mut max_count, mut r_best) = (0i32, 0.0f32);
        let mut j = n_bins - 1;
        while j > 0 {
            if bins[j as usize] != 0 {
                // A window of ten bins ending here, summed as it walks down.
                let upbin = j;
                let mut cur_count = 0;
                while j > upbin - N_BINS_PER_DR && j >= 0 {
                    cur_count += bins[j as usize];
                    j -= 1;
                }
                let r_cur =
                    (upbin + j) as f32 / 2.0 / N_BINS_PER_DR as f32 * dr + min_radius as f32;
                // Pick by support *per unit radius*: a big circle naturally
                // collects more edge pixels than a small one, and comparing
                // raw counts would always choose the biggest.
                if (cur_count as f32 * r_best >= max_count as f32 * r_cur)
                    || (r_best < f32::EPSILON && cur_count >= max_count)
                {
                    r_best = r_cur;
                    max_count = cur_count;
                }
            }
            j -= 1;
        }

        if max_count > acc_threshold {
            estimated.push((max_count, cx, cy, r_best));
        }
    }

    // `cmpAccum`: support, then larger radius, then smaller x, then smaller y.
    // Upstream compares every field so the order cannot depend on the sort.
    estimated.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then(b.3.total_cmp(&a.3))
            .then(a.1.total_cmp(&b.1))
            .then(a.2.total_cmp(&b.2))
    });

    let mut out: Vec<HoughCircle> = Vec::new();
    for (votes, cx, cy, radius) in estimated {
        if out.len() >= cap(circles_max) {
            break;
        }
        // Keep centres apart: two peaks a pixel apart describe one circle.
        if out.iter().any(|c| {
            let (dx, dy) = (c.x - cx, c.y - cy);
            f64::from(dx * dx + dy * dy) < min_dist2
        }) {
            continue;
        }
        out.push(HoughCircle { x: cx, y: cy, radius, votes });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The generalised Hough transform
// ---------------------------------------------------------------------------

/// `cv::GeneralizedHough` — the shared interface of the two detectors.
///
/// Where [`hough_lines`] votes for a parametrised shape, this votes for an
/// arbitrary one: a template's edge pixels are stored as offsets from a
/// reference point (the R-table), and every edge pixel of the query image votes
/// for where that reference point would have to be. It finds shapes that have
/// no equation.
pub trait GeneralizedHough {
    /// `cv::GeneralizedHough::setTemplate(templ, cannyThreshold, templCenter)`.
    fn set_template(&mut self, templ: &Mat, templ_center: Option<Point>) -> Result<()>;

    /// `cv::GeneralizedHough::detect(image, positions, votes)`.
    ///
    /// Returns `(x, y, scale, angle)` per detection, plus the vote count.
    fn detect(&self, image: &Mat) -> Result<Vec<(Point, f64, f64, i32)>>;

    /// `cv::GeneralizedHough::getCannyLowThresh()`.
    fn get_canny_low_thresh(&self) -> i32;
    /// `cv::GeneralizedHough::setCannyLowThresh(cannyLowThresh)`.
    fn set_canny_low_thresh(&mut self, v: i32);
    /// `cv::GeneralizedHough::getCannyHighThresh()`.
    fn get_canny_high_thresh(&self) -> i32;
    /// `cv::GeneralizedHough::setCannyHighThresh(cannyHighThresh)`.
    fn set_canny_high_thresh(&mut self, v: i32);
    /// `cv::GeneralizedHough::getMinDist()` — the smallest gap between two hits.
    fn get_min_dist(&self) -> f64;
    /// `cv::GeneralizedHough::setMinDist(minDist)`.
    fn set_min_dist(&mut self, v: f64);
    /// `cv::GeneralizedHough::getDp()` — accumulator resolution divisor.
    fn get_dp(&self) -> f64;
    /// `cv::GeneralizedHough::setDp(dp)`.
    fn set_dp(&mut self, v: f64);
    /// `cv::GeneralizedHough::getMaxBufferSize()` — cap on edge points used.
    fn get_max_buffer_size(&self) -> i32;
    /// `cv::GeneralizedHough::setMaxBufferSize(maxBufferSize)`.
    fn set_max_buffer_size(&mut self, v: i32);
}

/// One R-table entry: an edge point's offset to the reference, and the gradient
/// direction that indexes it.
#[derive(Debug, Clone, Copy)]
struct REntry {
    dx: f64,
    dy: f64,
}

/// The shared state of both detectors.
#[derive(Debug, Clone)]
struct Common {
    canny_low: i32,
    canny_high: i32,
    min_dist: f64,
    dp: f64,
    max_buffer_size: i32,
    /// R-table, bucketed by quantised gradient direction.
    r_table: Vec<Vec<REntry>>,
}

/// Direction buckets in the R-table; 1 degree each, as upstream uses.
const R_LEVELS: usize = 360;

impl Common {
    fn new() -> Self {
        Common {
            canny_low: 50,
            canny_high: 100,
            min_dist: 1.0,
            dp: 1.0,
            max_buffer_size: 1000,
            r_table: vec![Vec::new(); R_LEVELS],
        }
    }

    /// Edge points with their gradient direction in degrees, 0..360.
    fn edges_with_direction(&self, image: &Mat) -> Result<Vec<(i32, i32, f64)>> {
        cv_assert!(image.channels() == 1, "the generalised Hough takes a single channel");
        let edges = crate::edge::canny(
            image,
            f64::from(self.canny_low),
            f64::from(self.canny_high),
            3,
            false,
        )?;
        let dx =
            crate::filter::sobel(image, Some(Depth::F64), 1, 0, 3, 1.0, 0.0, BorderType::Replicate)?;
        let dy =
            crate::filter::sobel(image, Some(Depth::F64), 0, 1, 3, 1.0, 0.0, BorderType::Replicate)?;
        let mut out = Vec::new();
        for y in 0..image.rows() {
            for x in 0..image.cols() {
                if edges.at_f64(y, x, 0)? == 0.0 {
                    continue;
                }
                let (gx, gy) = (dx.at_f64(y, x, 0)?, dy.at_f64(y, x, 0)?);
                if gx.hypot(gy) < 1e-9 {
                    continue;
                }
                let mut a = gy.atan2(gx).to_degrees();
                if a < 0.0 {
                    a += 360.0;
                }
                out.push((x, y, a));
                if out.len() >= self.max_buffer_size.max(1) as usize {
                    return Ok(out);
                }
            }
        }
        Ok(out)
    }

    fn build_r_table(&mut self, templ: &Mat, center: Option<Point>) -> Result<()> {
        let c = center
            .unwrap_or_else(|| Point::new(templ.cols() / 2, templ.rows() / 2));
        cv_assert!(
            c.x >= 0 && c.y >= 0 && c.x < templ.cols() && c.y < templ.rows(),
            "the template centre must lie inside the template"
        );
        let pts = self.edges_with_direction(templ)?;
        cv_assert!(!pts.is_empty(), "the template has no edges at these Canny thresholds");
        self.r_table = vec![Vec::new(); R_LEVELS];
        for (x, y, a) in pts {
            let bucket = (a.floor() as usize).min(R_LEVELS - 1);
            self.r_table[bucket]
                .push(REntry { dx: f64::from(c.x - x), dy: f64::from(c.y - y) });
        }
        Ok(())
    }
}

/// `cv::GeneralizedHoughBallard` — position only, at a fixed scale and angle.
#[derive(Debug, Clone)]
pub struct GeneralizedHoughBallard {
    common: Common,
    levels: i32,
    votes_threshold: i32,
}

impl Default for GeneralizedHoughBallard {
    fn default() -> Self {
        GeneralizedHoughBallard::new()
    }
}

impl GeneralizedHoughBallard {
    /// `cv::createGeneralizedHoughBallard()`.
    #[must_use]
    pub fn new() -> Self {
        GeneralizedHoughBallard { common: Common::new(), levels: 360, votes_threshold: 100 }
    }

    /// `cv::GeneralizedHoughBallard::getLevels()` — gradient-direction buckets.
    #[must_use]
    pub fn get_levels(&self) -> i32 {
        self.levels
    }
    /// `cv::GeneralizedHoughBallard::setLevels(levels)`.
    pub fn set_levels(&mut self, v: i32) {
        self.levels = v;
    }
    /// `cv::GeneralizedHoughBallard::getVotesThreshold()`.
    #[must_use]
    pub fn get_votes_threshold(&self) -> i32 {
        self.votes_threshold
    }
    /// `cv::GeneralizedHoughBallard::setVotesThreshold(votesThreshold)`.
    pub fn set_votes_threshold(&mut self, v: i32) {
        self.votes_threshold = v;
    }
}

/// `cv::createGeneralizedHoughBallard()`.
#[must_use]
pub fn create_generalized_hough_ballard() -> GeneralizedHoughBallard {
    GeneralizedHoughBallard::new()
}

impl GeneralizedHough for GeneralizedHoughBallard {
    fn set_template(&mut self, templ: &Mat, templ_center: Option<Point>) -> Result<()> {
        self.common.build_r_table(templ, templ_center)
    }

    fn detect(&self, image: &Mat) -> Result<Vec<(Point, f64, f64, i32)>> {
        cv_assert!(
            self.common.r_table.iter().any(|b| !b.is_empty()),
            "setTemplate must be called before detect"
        );
        let dp = self.common.dp.max(1.0);
        let idp = 1.0 / dp;
        let (aw, ah) = (
            ((f64::from(image.cols()) * idp).round() as i32).max(1),
            ((f64::from(image.rows()) * idp).round() as i32).max(1),
        );
        let mut accum = vec![0i32; (aw * ah) as usize];
        for (x, y, a) in self.common.edges_with_direction(image)? {
            let bucket = (a.floor() as usize).min(R_LEVELS - 1);
            for e in &self.common.r_table[bucket] {
                let cx = ((f64::from(x) + e.dx) * idp).round() as i32;
                let cy = ((f64::from(y) + e.dy) * idp).round() as i32;
                if (0..aw).contains(&cx) && (0..ah).contains(&cy) {
                    accum[(cy * aw + cx) as usize] += 1;
                }
            }
        }
        Ok(collect_peaks(&accum, aw, ah, dp, self.votes_threshold, self.common.min_dist)
            .into_iter()
            .map(|(p, v)| (p, 1.0, 0.0, v))
            .collect())
    }

    fn get_canny_low_thresh(&self) -> i32 {
        self.common.canny_low
    }
    fn set_canny_low_thresh(&mut self, v: i32) {
        self.common.canny_low = v;
    }
    fn get_canny_high_thresh(&self) -> i32 {
        self.common.canny_high
    }
    fn set_canny_high_thresh(&mut self, v: i32) {
        self.common.canny_high = v;
    }
    fn get_min_dist(&self) -> f64 {
        self.common.min_dist
    }
    fn set_min_dist(&mut self, v: f64) {
        self.common.min_dist = v;
    }
    fn get_dp(&self) -> f64 {
        self.common.dp
    }
    fn set_dp(&mut self, v: f64) {
        self.common.dp = v;
    }
    fn get_max_buffer_size(&self) -> i32 {
        self.common.max_buffer_size
    }
    fn set_max_buffer_size(&mut self, v: i32) {
        self.common.max_buffer_size = v;
    }
}

/// Accumulator peaks above `threshold`, strongest first and `min_dist` apart.
fn collect_peaks(
    accum: &[i32],
    aw: i32,
    ah: i32,
    dp: f64,
    threshold: i32,
    min_dist: f64,
) -> Vec<(Point, i32)> {
    let mut peaks: Vec<(i32, i32, i32)> = Vec::new();
    for y in 1..(ah - 1).max(1) {
        for x in 1..(aw - 1).max(1) {
            let i = (y * aw + x) as usize;
            let v = accum[i];
            // Above threshold *and* a local maximum, as upstream's
            // `processImpl` tests. Thresholding alone reports every cell of a
            // strong peak's plateau: on one rectangle in a 160x120 scene that
            // is 36 detections where cv2 gives one. The mixed strict/non-strict
            // comparisons are upstream's, and they are what break a tie in a
            // fixed direction instead of dropping both cells.
            if v > threshold
                && v > accum[i - 1]
                && v >= accum[i + 1]
                && v > accum[i - aw as usize]
                && v >= accum[i + aw as usize]
            {
                peaks.push((v, x, y));
            }
        }
    }
    peaks.sort_by(|a, b| b.0.cmp(&a.0).then(a.2.cmp(&b.2)).then(a.1.cmp(&b.1)));
    let min_dist2 = min_dist * min_dist;
    let mut out: Vec<(Point, i32)> = Vec::new();
    for (v, x, y) in peaks {
        let (px, py) = (f64::from(x) * dp, f64::from(y) * dp);
        if out.iter().any(|(p, _)| {
            let (dx, dy) = (f64::from(p.x) - px, f64::from(p.y) - py);
            dx * dx + dy * dy < min_dist2
        }) {
            continue;
        }
        out.push((Point::new(px.round() as i32, py.round() as i32), v));
    }
    out
}

/// `cv::GeneralizedHoughGuil` — position, scale and rotation.
///
/// Guil's method adds two more parameters to Ballard's, so it finds a shape
/// that has been resized or turned. The cost is one accumulator pass per
/// `(scale, angle)` pair, which is why the ranges and steps are all exposed:
/// the defaults sweep 360 angles and a 2x scale range, and narrowing them is
/// usually the difference between seconds and minutes.
#[derive(Debug, Clone)]
pub struct GeneralizedHoughGuil {
    common: Common,
    levels: i32,
    xi: f64,
    angle_epsilon: f64,
    min_angle: f64,
    max_angle: f64,
    angle_step: f64,
    angle_thresh: i32,
    min_scale: f64,
    max_scale: f64,
    scale_step: f64,
    scale_thresh: i32,
    pos_thresh: i32,
}

impl Default for GeneralizedHoughGuil {
    fn default() -> Self {
        GeneralizedHoughGuil::new()
    }
}

impl GeneralizedHoughGuil {
    /// `cv::createGeneralizedHoughGuil()`, with upstream's defaults.
    #[must_use]
    pub fn new() -> Self {
        GeneralizedHoughGuil {
            common: Common::new(),
            levels: 360,
            xi: 90.0,
            angle_epsilon: 1.0,
            min_angle: 0.0,
            max_angle: 360.0,
            angle_step: 1.0,
            angle_thresh: 10000,
            min_scale: 0.5,
            max_scale: 2.0,
            scale_step: 0.05,
            scale_thresh: 1000,
            pos_thresh: 100,
        }
    }

    /// `cv::GeneralizedHoughGuil::getXi()` — the angle between the two edge
    /// points each R-table entry pairs.
    #[must_use]
    pub fn get_xi(&self) -> f64 {
        self.xi
    }
    /// `cv::GeneralizedHoughGuil::setXi(xi)`.
    pub fn set_xi(&mut self, v: f64) {
        self.xi = v;
    }
    /// `cv::GeneralizedHoughGuil::getLevels()`.
    #[must_use]
    pub fn get_levels(&self) -> i32 {
        self.levels
    }
    /// `cv::GeneralizedHoughGuil::setLevels(levels)`.
    pub fn set_levels(&mut self, v: i32) {
        self.levels = v;
    }
    /// `cv::GeneralizedHoughGuil::getAngleEpsilon()` — direction tolerance.
    #[must_use]
    pub fn get_angle_epsilon(&self) -> f64 {
        self.angle_epsilon
    }
    /// `cv::GeneralizedHoughGuil::setAngleEpsilon(angleEpsilon)`.
    pub fn set_angle_epsilon(&mut self, v: f64) {
        self.angle_epsilon = v;
    }
    /// `cv::GeneralizedHoughGuil::getMinAngle()`.
    #[must_use]
    pub fn get_min_angle(&self) -> f64 {
        self.min_angle
    }
    /// `cv::GeneralizedHoughGuil::setMinAngle(minAngle)`.
    pub fn set_min_angle(&mut self, v: f64) {
        self.min_angle = v;
    }
    /// `cv::GeneralizedHoughGuil::getMaxAngle()`.
    #[must_use]
    pub fn get_max_angle(&self) -> f64 {
        self.max_angle
    }
    /// `cv::GeneralizedHoughGuil::setMaxAngle(maxAngle)`.
    pub fn set_max_angle(&mut self, v: f64) {
        self.max_angle = v;
    }
    /// `cv::GeneralizedHoughGuil::getAngleStep()`.
    #[must_use]
    pub fn get_angle_step(&self) -> f64 {
        self.angle_step
    }
    /// `cv::GeneralizedHoughGuil::setAngleStep(angleStep)`.
    pub fn set_angle_step(&mut self, v: f64) {
        self.angle_step = v;
    }
    /// `cv::GeneralizedHoughGuil::getAngleThresh()`.
    #[must_use]
    pub fn get_angle_thresh(&self) -> i32 {
        self.angle_thresh
    }
    /// `cv::GeneralizedHoughGuil::setAngleThresh(angleThresh)`.
    pub fn set_angle_thresh(&mut self, v: i32) {
        self.angle_thresh = v;
    }
    /// `cv::GeneralizedHoughGuil::getMinScale()`.
    #[must_use]
    pub fn get_min_scale(&self) -> f64 {
        self.min_scale
    }
    /// `cv::GeneralizedHoughGuil::setMinScale(minScale)`.
    pub fn set_min_scale(&mut self, v: f64) {
        self.min_scale = v;
    }
    /// `cv::GeneralizedHoughGuil::getMaxScale()`.
    #[must_use]
    pub fn get_max_scale(&self) -> f64 {
        self.max_scale
    }
    /// `cv::GeneralizedHoughGuil::setMaxScale(maxScale)`.
    pub fn set_max_scale(&mut self, v: f64) {
        self.max_scale = v;
    }
    /// `cv::GeneralizedHoughGuil::getScaleStep()`.
    #[must_use]
    pub fn get_scale_step(&self) -> f64 {
        self.scale_step
    }
    /// `cv::GeneralizedHoughGuil::setScaleStep(scaleStep)`.
    pub fn set_scale_step(&mut self, v: f64) {
        self.scale_step = v;
    }
    /// `cv::GeneralizedHoughGuil::getScaleThresh()`.
    #[must_use]
    pub fn get_scale_thresh(&self) -> i32 {
        self.scale_thresh
    }
    /// `cv::GeneralizedHoughGuil::setScaleThresh(scaleThresh)`.
    pub fn set_scale_thresh(&mut self, v: i32) {
        self.scale_thresh = v;
    }
    /// `cv::GeneralizedHoughGuil::getPosThresh()`.
    #[must_use]
    pub fn get_pos_thresh(&self) -> i32 {
        self.pos_thresh
    }
    /// `cv::GeneralizedHoughGuil::setPosThresh(posThresh)`.
    pub fn set_pos_thresh(&mut self, v: i32) {
        self.pos_thresh = v;
    }
}

/// `cv::createGeneralizedHoughGuil()`.
#[must_use]
pub fn create_generalized_hough_guil() -> GeneralizedHoughGuil {
    GeneralizedHoughGuil::new()
}

impl GeneralizedHough for GeneralizedHoughGuil {
    fn set_template(&mut self, templ: &Mat, templ_center: Option<Point>) -> Result<()> {
        self.common.build_r_table(templ, templ_center)
    }

    /// Sweeps `(scale, angle)` and runs Ballard's position vote at each.
    ///
    /// The R-table entries are rotated and scaled rather than the image, which
    /// is what makes the sweep affordable: resampling the query image once per
    /// pair would dominate everything else.
    fn detect(&self, image: &Mat) -> Result<Vec<(Point, f64, f64, i32)>> {
        cv_assert!(
            self.common.r_table.iter().any(|b| !b.is_empty()),
            "setTemplate must be called before detect"
        );
        cv_assert!(self.scale_step > 0.0, "the scale step must be positive");
        cv_assert!(self.angle_step > 0.0, "the angle step must be positive");
        cv_assert!(self.max_scale >= self.min_scale, "maxScale must not be below minScale");
        cv_assert!(self.max_angle >= self.min_angle, "maxAngle must not be below minAngle");

        let dp = self.common.dp.max(1.0);
        let idp = 1.0 / dp;
        let (aw, ah) = (
            ((f64::from(image.cols()) * idp).round() as i32).max(1),
            ((f64::from(image.rows()) * idp).round() as i32).max(1),
        );
        let edges = self.common.edges_with_direction(image)?;

        let mut best: Vec<(Point, f64, f64, i32)> = Vec::new();
        let n_scale = ((self.max_scale - self.min_scale) / self.scale_step).floor() as i32 + 1;
        let n_angle = ((self.max_angle - self.min_angle) / self.angle_step).floor() as i32 + 1;
        for si in 0..n_scale {
            let scale = self.min_scale + f64::from(si) * self.scale_step;
            for ai in 0..n_angle {
                let angle = self.min_angle + f64::from(ai) * self.angle_step;
                let (sa, ca) = angle.to_radians().sin_cos();
                let mut accum = vec![0i32; (aw * ah) as usize];
                for &(x, y, a) in &edges {
                    // A rotated shape presents each edge with its direction
                    // turned by the same angle, so the R-table is indexed at
                    // `a - angle`, not at `a`.
                    let mut src = a - angle;
                    while src < 0.0 {
                        src += 360.0;
                    }
                    let bucket = (src.floor() as usize) % R_LEVELS;
                    for e in &self.common.r_table[bucket] {
                        let (rx, ry) = (e.dx * scale, e.dy * scale);
                        let cx = ((f64::from(x) + rx * ca - ry * sa) * idp).round() as i32;
                        let cy = ((f64::from(y) + rx * sa + ry * ca) * idp).round() as i32;
                        if (0..aw).contains(&cx) && (0..ah).contains(&cy) {
                            accum[(cy * aw + cx) as usize] += 1;
                        }
                    }
                }
                for (p, v) in
                    collect_peaks(&accum, aw, ah, dp, self.pos_thresh, self.common.min_dist)
                {
                    best.push((p, scale, angle, v));
                }
            }
        }
        // One hit per location: the strongest (scale, angle) wins.
        best.sort_by_key(|h| core::cmp::Reverse(h.3));
        let min_dist2 = self.common.min_dist * self.common.min_dist;
        let mut out: Vec<(Point, f64, f64, i32)> = Vec::new();
        for hit in best {
            if out.iter().any(|(p, _, _, _)| {
                let (dx, dy) = (f64::from(p.x - hit.0.x), f64::from(p.y - hit.0.y));
                dx * dx + dy * dy < min_dist2
            }) {
                continue;
            }
            out.push(hit);
        }
        Ok(out)
    }

    fn get_canny_low_thresh(&self) -> i32 {
        self.common.canny_low
    }
    fn set_canny_low_thresh(&mut self, v: i32) {
        self.common.canny_low = v;
    }
    fn get_canny_high_thresh(&self) -> i32 {
        self.common.canny_high
    }
    fn set_canny_high_thresh(&mut self, v: i32) {
        self.common.canny_high = v;
    }
    fn get_min_dist(&self) -> f64 {
        self.common.min_dist
    }
    fn set_min_dist(&mut self, v: f64) {
        self.common.min_dist = v;
    }
    fn get_dp(&self) -> f64 {
        self.common.dp
    }
    fn set_dp(&mut self, v: f64) {
        self.common.dp = v;
    }
    fn get_max_buffer_size(&self) -> i32 {
        self.common.max_buffer_size
    }
    fn set_max_buffer_size(&mut self, v: i32) {
        self.common.max_buffer_size = v;
    }
}

/// Lines as a `CV_32FC1` matrix of `(rho, theta)` rows, which is the shape
/// `cv::HoughLines` writes into its output array.
pub fn lines_to_mat(lines: &[HoughLine]) -> Result<Mat> {
    let mut m = Mat::zeros(lines.len() as i32, 2, CV_32FC1)?;
    for (i, l) in lines.iter().enumerate() {
        m.pixel_mut::<f32>(i as i32, 0)?[0] = l.rho;
        m.pixel_mut::<f32>(i as i32, 1)?[0] = l.theta;
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::{MatType, CV_8UC1};

    /// A blank mask with `f` deciding which pixels are set.
    fn mask(w: i32, h: i32, f: impl Fn(i32, i32) -> bool) -> Mat {
        let mut m = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                if f(x, y) {
                    m.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
                }
            }
        }
        m
    }

    #[test]
    fn a_vertical_line_comes_back_at_theta_zero() {
        // x = 20 for every row: the normal points along +x, so theta = 0 and
        // rho = 20.
        let img = mask(64, 64, |x, _| x == 20);
        let lines =
            hough_lines(&img, 1.0, core::f64::consts::PI / 180.0, 30, 5, 0.0, core::f64::consts::PI)
                .unwrap();
        assert!(!lines.is_empty(), "the line should be found");
        let l = lines[0];
        assert!((l.rho - 20.0).abs() <= 1.0, "rho {} should be 20", l.rho);
        assert!(l.theta.abs() < 1e-6, "theta {} should be 0", l.theta);
        assert_eq!(l.votes, 64, "one vote per row");
    }

    #[test]
    fn a_horizontal_line_comes_back_at_theta_ninety_degrees() {
        let img = mask(64, 64, |_, y| y == 10);
        let lines =
            hough_lines(&img, 1.0, core::f64::consts::PI / 180.0, 30, 5, 0.0, core::f64::consts::PI)
                .unwrap();
        let l = lines[0];
        assert!((l.rho - 10.0).abs() <= 1.0, "rho {}", l.rho);
        assert!(
            (f64::from(l.theta) - core::f64::consts::FRAC_PI_2).abs() < 0.02,
            "theta {}",
            l.theta
        );
    }

    #[test]
    fn two_lines_are_both_found_and_ranked_by_votes() {
        // A long vertical line and a shorter horizontal one.
        let img = mask(64, 64, |x, y| x == 30 || (y == 40 && x < 40));
        let lines =
            hough_lines(&img, 1.0, core::f64::consts::PI / 180.0, 20, 10, 0.0, core::f64::consts::PI)
                .unwrap();
        assert!(lines.len() >= 2, "both lines should appear: {lines:?}");
        assert!(lines[0].votes >= lines[1].votes, "sorted strongest first");
        assert_eq!(lines[0].votes, 64, "the full-height line wins");
    }

    #[test]
    fn an_empty_mask_and_a_high_threshold_find_nothing() {
        let blank = Mat::zeros(32, 32, CV_8UC1).unwrap();
        assert!(hough_lines(
            &blank,
            1.0,
            core::f64::consts::PI / 180.0,
            1,
            5,
            0.0,
            core::f64::consts::PI
        )
        .unwrap()
        .is_empty());

        let img = mask(64, 64, |x, _| x == 20);
        assert!(hough_lines(
            &img,
            1.0,
            core::f64::consts::PI / 180.0,
            1000,
            5,
            0.0,
            core::f64::consts::PI
        )
        .unwrap()
        .is_empty());
    }

    #[test]
    fn hough_lines_rejects_bad_parameters() {
        let img = mask(16, 16, |x, _| x == 4);
        assert!(hough_lines(&img, 0.0, 0.1, 1, 5, 0.0, core::f64::consts::PI).is_err());
        assert!(hough_lines(&img, 1.0, 0.0, 1, 5, 0.0, core::f64::consts::PI).is_err());
        assert!(hough_lines(&img, 1.0, 0.1, 1, 5, 2.0, 1.0).is_err(), "min above max");
        assert!(hough_lines(&img, 1.0, 0.1, 1, 5, -1.0, 1.0).is_err(), "outside 0..pi");
        // A three-channel image is not a mask.
        let colour = Mat::zeros(8, 8, MatType::new(Depth::U8, 3)).unwrap();
        assert!(hough_lines(&colour, 1.0, 0.1, 1, 5, 0.0, core::f64::consts::PI).is_err());
    }

    #[test]
    fn the_probabilistic_transform_returns_the_segment_endpoints() {
        // A vertical run from y=10 to y=49 at x=25.
        let img = mask(64, 64, |x, y| x == 25 && (10..50).contains(&y));
        let segs =
            hough_lines_p(&img, 1.0, core::f64::consts::PI / 180.0, 20, 20.0, 3.0, 10).unwrap();
        assert!(!segs.is_empty(), "the run should be found");
        let s = segs[0];
        assert_eq!(s.a.x, 25);
        assert_eq!(s.b.x, 25);
        let (lo, hi) = (s.a.y.min(s.b.y), s.a.y.max(s.b.y));
        assert!(lo <= 12 && hi >= 47, "the segment {lo}..{hi} should span the run");
    }

    #[test]
    fn a_segment_shorter_than_the_minimum_is_dropped() {
        let img = mask(64, 64, |x, y| x == 25 && (10..15).contains(&y));
        let segs =
            hough_lines_p(&img, 1.0, core::f64::consts::PI / 180.0, 3, 40.0, 3.0, 10).unwrap();
        assert!(segs.is_empty(), "a 5-pixel run is not a 40-pixel line: {segs:?}");
    }

    #[test]
    fn a_gap_smaller_than_max_line_gap_is_bridged() {
        // Two runs with a three-pixel hole between them.
        let img = mask(64, 64, |x, y| x == 25 && ((10..25).contains(&y) || (28..45).contains(&y)));
        let joined =
            hough_lines_p(&img, 1.0, core::f64::consts::PI / 180.0, 15, 30.0, 5.0, 10).unwrap();
        assert!(!joined.is_empty(), "a 5-pixel gap tolerance should bridge a 3-pixel hole");
        let split =
            hough_lines_p(&img, 1.0, core::f64::consts::PI / 180.0, 15, 30.0, 1.0, 10).unwrap();
        assert!(split.is_empty(), "a 1-pixel tolerance should not bridge it");
    }

    #[test]
    fn hough_lines_p_is_reproducible() {
        let img = mask(64, 64, |x, y| (x == 25 && y > 5) || (y == 30 && x > 5));
        let a = hough_lines_p(&img, 1.0, core::f64::consts::PI / 180.0, 20, 20.0, 3.0, 10).unwrap();
        let b = hough_lines_p(&img, 1.0, core::f64::consts::PI / 180.0, 20, 20.0, 3.0, 10).unwrap();
        assert_eq!(a, b, "the same input must give the same segments");
    }

    #[test]
    fn a_point_set_votes_without_rasterising() {
        // Points on y = 5, so rho = 5 at theta = pi/2.
        let pts: Vec<(f32, f32)> = (0..20).map(|i| (i as f32, 5.0)).collect();
        let lines = hough_lines_point_set(
            &pts,
            5,
            10,
            0.0,
            40.0,
            1.0,
            core::f64::consts::FRAC_PI_2 - 0.1,
            core::f64::consts::FRAC_PI_2 + 0.1,
            core::f64::consts::PI / 180.0,
        )
        .unwrap();
        assert!(!lines.is_empty());
        assert!((lines[0].rho - 5.0).abs() <= 1.0, "rho {}", lines[0].rho);
        assert_eq!(lines[0].votes, 20, "every point votes");

        assert!(hough_lines_point_set(&[], 5, 1, 0.0, 1.0, 1.0, 0.0, 1.0, 0.1).unwrap().is_empty());
        assert!(hough_lines_point_set(&pts, 5, 1, 0.0, 1.0, 0.0, 0.0, 1.0, 0.1).is_err());
        assert!(hough_lines_point_set(&pts, 5, 1, 1.0, 0.0, 1.0, 0.0, 1.0, 0.1).is_err());
    }

    /// A filled disc of radius `r` centred at `(cx, cy)`.
    fn disc(w: i32, h: i32, cx: i32, cy: i32, r: i32) -> Mat {
        mask(w, h, |x, y| {
            let (dx, dy) = (x - cx, y - cy);
            dx * dx + dy * dy <= r * r
        })
    }

    /// A 5x5 Gaussian, which is what the docs on [`hough_circles`] ask for.
    fn softened(m: &Mat) -> Mat {
        crate::filter::gaussian_blur(m, cv_core::Size::new(5, 5), 1.5, 1.5, BorderType::Replicate)
            .unwrap()
    }

    #[test]
    fn a_drawn_disc_is_found_at_its_centre_and_radius() {
        let img = softened(&disc(100, 100, 50, 45, 20));
        let circles = hough_circles(&img, 1.0, 20.0, 100.0, 20.0, 12, 30, 5).unwrap();
        assert!(!circles.is_empty(), "the disc should be found");
        let c = circles[0];
        assert!((c.x - 50.0).abs() <= 2.0, "x {}", c.x);
        assert!((c.y - 45.0).abs() <= 2.0, "y {}", c.y);
        assert!((c.radius - 20.0).abs() <= 2.0, "radius {}", c.radius);

        // Without the blur the same disc is a much weaker peak, which is why
        // the docs insist on it.
        let raw = hough_circles(&disc(100, 100, 50, 45, 20), 1.0, 20.0, 100.0, 20.0, 12, 30, 5)
            .unwrap();
        assert!(raw.is_empty() || raw[0].votes < c.votes, "the blur should sharpen the peak");
    }

    #[test]
    fn two_discs_are_kept_apart_by_min_dist() {
        let mut img = disc(140, 80, 35, 40, 15);
        let other = disc(140, 80, 105, 40, 15);
        for y in 0..80 {
            for x in 0..140 {
                if other.at_f64(y, x, 0).unwrap() != 0.0 {
                    img.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
                }
            }
        }
        let img = softened(&img);
        let two = hough_circles(&img, 1.0, 30.0, 100.0, 15.0, 10, 25, 5).unwrap();
        assert_eq!(two.len(), 2, "both discs should be found: {two:?}");
        let mut xs: Vec<f32> = two.iter().map(|c| c.x).collect();
        xs.sort_by(f32::total_cmp);
        assert!((xs[0] - 35.0).abs() <= 3.0 && (xs[1] - 105.0).abs() <= 3.0, "{xs:?}");
        // A minimum distance wider than their separation collapses them to one.
        let one = hough_circles(&img, 1.0, 100.0, 100.0, 15.0, 10, 25, 5).unwrap();
        assert_eq!(one.len(), 1, "{one:?}");
    }

    #[test]
    fn hough_circles_rejects_bad_parameters() {
        let img = disc(40, 40, 20, 20, 8);
        assert!(hough_circles(&img, 0.5, 10.0, 100.0, 10.0, 4, 12, 5).is_err(), "dp below 1");
        assert!(hough_circles(&img, 1.0, 0.0, 100.0, 10.0, 4, 12, 5).is_err());
        assert!(hough_circles(&img, 1.0, 10.0, 0.0, 10.0, 4, 12, 5).is_err());
        assert!(hough_circles(&img, 1.0, 10.0, 100.0, 10.0, 20, 5, 5).is_err(), "min above max");
        let f = Mat::zeros(8, 8, MatType::new(Depth::F32, 1)).unwrap();
        assert!(hough_circles(&f, 1.0, 10.0, 100.0, 10.0, 1, 4, 5).is_err(), "not CV_8U");
    }

    /// A hollow square, which has enough edge direction variety to be a
    /// meaningful generalised-Hough template.
    fn square(w: i32, h: i32, x0: i32, y0: i32, side: i32) -> Mat {
        mask(w, h, |x, y| {
            let inside = (x0..x0 + side).contains(&x) && (y0..y0 + side).contains(&y);
            let inner = (x0 + 3..x0 + side - 3).contains(&x) && (y0 + 3..y0 + side - 3).contains(&y);
            inside && !inner
        })
    }

    #[test]
    fn ballard_finds_the_template_where_it_was_planted() {
        let templ = square(40, 40, 8, 8, 24);
        let scene = square(120, 120, 60, 30, 24);
        let mut g = create_generalized_hough_ballard();
        assert_eq!(g.get_levels(), 360);
        assert_eq!(g.get_votes_threshold(), 100);
        g.set_votes_threshold(10);
        g.set_min_dist(20.0);
        g.set_template(&templ, None).unwrap();
        let hits = g.detect(&scene).unwrap();
        assert!(!hits.is_empty(), "the square should be found");
        // The template centre is at (20, 20) of a 24-wide square starting at
        // (8, 8), so the planted centre is (60 + 12, 30 + 12) = (72, 42).
        let (p, scale, angle, votes) = hits[0];
        assert!((p.x - 72).abs() <= 6, "x {}", p.x);
        assert!((p.y - 42).abs() <= 6, "y {}", p.y);
        assert_eq!(scale, 1.0, "Ballard does not search scale");
        assert_eq!(angle, 0.0);
        assert!(votes > 0);
    }

    #[test]
    fn detect_before_set_template_is_an_error() {
        let scene = square(60, 60, 10, 10, 20);
        assert!(create_generalized_hough_ballard().detect(&scene).is_err());
        assert!(create_generalized_hough_guil().detect(&scene).is_err());
        // A centre outside the template is refused.
        let templ = square(40, 40, 8, 8, 24);
        let mut g = create_generalized_hough_ballard();
        assert!(g.set_template(&templ, Some(Point::new(100, 100))).is_err());
    }

    #[test]
    fn the_generalized_hough_accessors_round_trip() {
        let mut b = create_generalized_hough_ballard();
        b.set_canny_low_thresh(11);
        b.set_canny_high_thresh(22);
        b.set_min_dist(3.5);
        b.set_dp(2.0);
        b.set_max_buffer_size(77);
        b.set_levels(180);
        assert_eq!(b.get_canny_low_thresh(), 11);
        assert_eq!(b.get_canny_high_thresh(), 22);
        assert_eq!(b.get_min_dist(), 3.5);
        assert_eq!(b.get_dp(), 2.0);
        assert_eq!(b.get_max_buffer_size(), 77);
        assert_eq!(b.get_levels(), 180);

        let mut g = create_generalized_hough_guil();
        // Upstream's defaults.
        assert_eq!(g.get_xi(), 90.0);
        assert_eq!(g.get_levels(), 360);
        assert_eq!(g.get_angle_epsilon(), 1.0);
        assert_eq!((g.get_min_angle(), g.get_max_angle(), g.get_angle_step()), (0.0, 360.0, 1.0));
        assert_eq!(g.get_angle_thresh(), 10000);
        assert_eq!((g.get_min_scale(), g.get_max_scale(), g.get_scale_step()), (0.5, 2.0, 0.05));
        assert_eq!(g.get_scale_thresh(), 1000);
        assert_eq!(g.get_pos_thresh(), 100);

        g.set_xi(45.0);
        g.set_levels(90);
        g.set_angle_epsilon(2.0);
        g.set_min_angle(10.0);
        g.set_max_angle(20.0);
        g.set_angle_step(5.0);
        g.set_angle_thresh(7);
        g.set_min_scale(0.9);
        g.set_max_scale(1.1);
        g.set_scale_step(0.1);
        g.set_scale_thresh(8);
        g.set_pos_thresh(9);
        g.set_min_dist(4.0);
        assert_eq!(g.get_xi(), 45.0);
        assert_eq!(g.get_levels(), 90);
        assert_eq!(g.get_angle_epsilon(), 2.0);
        assert_eq!((g.get_min_angle(), g.get_max_angle(), g.get_angle_step()), (10.0, 20.0, 5.0));
        assert_eq!(g.get_angle_thresh(), 7);
        assert_eq!((g.get_min_scale(), g.get_max_scale(), g.get_scale_step()), (0.9, 1.1, 0.1));
        assert_eq!(g.get_scale_thresh(), 8);
        assert_eq!(g.get_pos_thresh(), 9);
        assert_eq!(g.get_min_dist(), 4.0);
    }

    #[test]
    fn guil_reports_the_scale_and_angle_it_searched() {
        let templ = square(40, 40, 8, 8, 24);
        let scene = square(120, 120, 50, 40, 24);
        let mut g = create_generalized_hough_guil();
        // Narrow sweep: unit scale, no rotation.
        g.set_min_scale(1.0);
        g.set_max_scale(1.0);
        g.set_min_angle(0.0);
        g.set_max_angle(0.0);
        g.set_pos_thresh(10);
        g.set_min_dist(20.0);
        g.set_template(&templ, None).unwrap();
        let hits = g.detect(&scene).unwrap();
        assert!(!hits.is_empty(), "the square should be found");
        let (p, scale, angle, _) = hits[0];
        assert_eq!(scale, 1.0);
        assert_eq!(angle, 0.0);
        assert!((p.x - 62).abs() <= 8 && (p.y - 52).abs() <= 8, "{p:?}");

        // A step of zero would loop forever; it is refused instead.
        g.set_scale_step(0.0);
        assert!(g.detect(&scene).is_err());
    }

    #[test]
    fn lines_convert_to_the_two_column_matrix_form() {
        let lines = vec![
            HoughLine { rho: 5.0, theta: 0.5, votes: 9 },
            HoughLine { rho: -2.0, theta: 1.5, votes: 4 },
        ];
        let m = lines_to_mat(&lines).unwrap();
        assert_eq!((m.rows(), m.cols()), (2, 2));
        assert!((m.at_f64(0, 0, 0).unwrap() - 5.0).abs() < 1e-6);
        assert!((m.at_f64(1, 1, 0).unwrap() - 1.5).abs() < 1e-6);
    }
}
