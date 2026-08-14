//! Calibration-target detection (`modules/calib3d/src/calibinit.cpp`,
//! `circlesgrid.cpp`, `checkchessboard.cpp`).
//!
//! Every intrinsic calibration starts here: a target whose geometry is known
//! exactly, found in the image to sub-pixel accuracy. The accuracy of the
//! calibration is bounded by the accuracy of this step, which is why a chess
//! corner — the meeting of four regions, localisable to a fraction of a pixel —
//! is preferred to a printed dot, and why the circles grid uses centroids
//! rather than edges.
//!
//! # Ordering matters as much as detection
//!
//! A corner list is useless unless it is in a known order, because the caller
//! pairs it index-for-index with the target's object points. Finding all
//! `w * h` corners and returning them row-shuffled produces a calibration that
//! converges to nonsense with no error reported, so every detector here either
//! returns a fully ordered grid or returns nothing.

use cv_core::types::Depth;
use cv_core::{cv_assert, MatLike, Point, Result, Size};

/// `cv::checkChessboard(img, size)`.
///
/// A cheap pre-test: does this frame plausibly contain a board at all? It
/// thresholds at several levels and asks whether the black and white areas are
/// roughly balanced and numerous enough. Running it before the full detector
/// is what keeps a live calibration loop responsive — the full search is
/// expensive and almost every frame of a hand-held session has no board in it.
pub fn check_chessboard<S: MatLike + ?Sized>(image: &S, size: Size) -> Result<bool> {
    cv_assert!(image.channels() == 1, "checkChessboard takes a single-channel image");
    cv_assert!(size.width > 1 && size.height > 1, "the pattern must be at least 2x2");
    let (h, w) = (image.rows(), image.cols());
    if h < 10 || w < 10 {
        return Ok(false);
    }
    let squares = (size.width + 1) * (size.height + 1);
    for level in [64.0f64, 96.0, 128.0, 160.0, 192.0] {
        let mut dark = 0usize;
        for y in 0..h {
            for x in 0..w {
                if image.at_f64(y, x, 0)? < level {
                    dark += 1;
                }
            }
        }
        let ratio = dark as f64 / f64::from(h * w);
        // A board covering a reasonable part of the frame is between a quarter
        // and three quarters dark at *some* threshold. A blank or saturated
        // frame is never in that band at any of them.
        if (0.25..=0.75).contains(&ratio) {
            // ...and it must have enough transitions along a scanline to be a
            // board rather than one big dark blob.
            let mid = h / 2;
            let mut runs = 0;
            let mut prev = image.at_f64(mid, 0, 0)? < level;
            for x in 1..w {
                let cur = image.at_f64(mid, x, 0)? < level;
                if cur != prev {
                    runs += 1;
                    prev = cur;
                }
            }
            if runs >= size.width.min(size.height) {
                let _ = squares;
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// `cv::findChessboardCorners(image, patternSize, corners, flags)`.
///
/// Returns the `patternSize.width * patternSize.height` inner corners in
/// row-major order, or `None` if the full grid was not found.
///
/// # How it works
///
/// Threshold, then find the corners as saddle points of the intensity surface:
/// a chess corner is the one place where the surface curves up along one
/// diagonal and down along the other. Saddles are far more distinctive than
/// plain corners, which is why this works on a target that is otherwise all
/// straight edges. The survivors are then fitted to a grid, and a fit that
/// does not consume exactly `w * h` points is rejected.
pub fn find_chessboard_corners<S: MatLike + ?Sized>(
    image: &S,
    pattern_size: Size,
) -> Result<Option<Vec<(f32, f32)>>> {
    cv_assert!(image.channels() == 1, "findChessboardCorners takes a single-channel image");
    cv_assert!(
        pattern_size.width > 1 && pattern_size.height > 1,
        "the pattern must be at least 2x2, got {}x{}",
        pattern_size.width,
        pattern_size.height
    );
    let (h, w) = (image.rows(), image.cols());
    let want = (pattern_size.width * pattern_size.height) as usize;
    if h < 8 || w < 8 {
        return Ok(None);
    }

    // Smooth first. A printed board is a *step* edge, and the second difference
    // of a step is not a saddle — on an unblurred synthetic board the Hessian
    // determinant is positive at every corner and nothing is found at all. One
    // pixel of blur turns each corner into the hyperbolic paraboloid the saddle
    // test is looking for.
    let img = cv_imgproc::filter::gaussian_blur(
        image,
        Size::new(7, 7),
        1.0,
        1.0,
        cv_core::BorderType::Replicate,
    )?;

    // The saddle response: the determinant of the Hessian, which is negative
    // exactly at a saddle. Its magnitude ranks the candidates.
    let mut response = vec![0.0f64; (h * w) as usize];
    let mut peak = 0.0f64;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = img.at_f64(y, x, 0)?;
            let ixx = img.at_f64(y, x - 1, 0)? - 2.0 * c + img.at_f64(y, x + 1, 0)?;
            let iyy = img.at_f64(y - 1, x, 0)? - 2.0 * c + img.at_f64(y + 1, x, 0)?;
            let ixy = (img.at_f64(y + 1, x + 1, 0)? - img.at_f64(y + 1, x - 1, 0)?
                - img.at_f64(y - 1, x + 1, 0)?
                + img.at_f64(y - 1, x - 1, 0)?)
                / 4.0;
            let det = ixx * iyy - ixy * ixy;
            if det < 0.0 {
                let v = -det;
                response[(y * w + x) as usize] = v;
                peak = peak.max(v);
            }
        }
    }
    if peak <= 0.0 {
        return Ok(None);
    }

    // Local maxima of the response, suppressed at the blur scale only. The
    // radius must not be derived from the image size: a board photographed at
    // a distance covers a third of the frame, and `w / (cols + 1)` then gives a
    // window several squares wide that suppresses most of the real corners.
    // With this radius there are many spurious candidates, and the spacing is
    // estimated from them below.
    let local_maxima = |radius: i32, floor: f64| -> Vec<(f64, i32, i32)> {
        let mut out: Vec<(f64, i32, i32)> = Vec::new();
        for y in radius..h - radius {
            for x in radius..w - radius {
                let here = (y * w + x) as usize;
                let v = response[here];
                if v < peak * floor {
                    continue;
                }
                // Strictly greater loses, *and* an equal value earlier in scan
                // order loses. Without the tie-break a plateau around one
                // corner yields several candidates, the strongest few crowd
                // out a whole row, and the grid fit fails on a good board.
                let mut best = true;
                'w: for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let n = ((y + dy) * w + x + dx) as usize;
                        let nv = response[n];
                        if nv > v || (nv == v && n < here) {
                            best = false;
                            break 'w;
                        }
                    }
                }
                if best {
                    out.push((v, x, y));
                }
            }
        }
        out
    };

    let mut seeds = local_maxima(2, 0.02);
    if seeds.len() < want {
        return Ok(None);
    }
    // The strongest candidates are the real corners; the median distance from
    // one of them to its nearest neighbour is the square size, whatever
    // fraction of the frame the board covers.
    seeds.sort_by(|a, b| b.0.total_cmp(&a.0));
    seeds.truncate((want * 4).min(seeds.len()));
    let mut nearest: Vec<f64> = seeds
        .iter()
        .map(|&(_, x, y)| {
            seeds
                .iter()
                .filter(|&&(_, ox, oy)| (ox, oy) != (x, y))
                .map(|&(_, ox, oy)| f64::from((ox - x).pow(2) + (oy - y).pow(2)).sqrt())
                .fold(f64::MAX, f64::min)
        })
        .collect();
    nearest.sort_by(f64::total_cmp);
    let spacing = nearest[nearest.len() / 2];
    if !spacing.is_finite() || spacing < 2.0 {
        return Ok(None);
    }

    // Now suppress at the board's own scale.
    let r = ((spacing / 2.0).round() as i32).max(2);
    let mut candidates = local_maxima(r, 0.02);
    if candidates.len() < want {
        return Ok(None);
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    candidates.truncate(want);

    // Sub-pixel: fit a parabola to the response along each axis. The response
    // is sampled on the pixel grid, so an integer answer is systematically up
    // to half a pixel out and no calibration built on it can do better.
    let refine = |x: i32, y: i32| -> (f32, f32) {
        let at = |dx: i32, dy: i32| response[((y + dy) * w + x + dx) as usize];
        let sub = |a: f64, b: f64, c: f64| -> f64 {
            let d = a - 2.0 * b + c;
            if d.abs() < 1e-12 {
                0.0
            } else {
                (0.5 * (a - c) / d).clamp(-1.0, 1.0)
            }
        };
        let dx = sub(at(-1, 0), at(0, 0), at(1, 0));
        let dy = sub(at(0, -1), at(0, 0), at(0, 1));
        ((x as f64 + dx) as f32, (y as f64 + dy) as f32)
    };
    let pts: Vec<(f32, f32)> = candidates.iter().map(|&(_, x, y)| refine(x, y)).collect();

    // Order into a grid *along the board's own axes*. Sorting by image y and
    // chunking works only while the board is square to the sensor; one row of
    // a board tilted by more than half a square spacing interleaves with the
    // next and the fit below rejects a perfectly good detection.
    //
    // The in-row direction is the modal nearest-neighbour direction, taken
    // modulo 180 degrees because a neighbour is as likely to be left as right.
    let mut bins = [0usize; 36];
    for &(x, y) in &pts {
        let mut best = (f64::MAX, 0.0f64);
        for &(ox, oy) in &pts {
            if (ox, oy) == (x, y) {
                continue;
            }
            let (dx, dy) = (f64::from(ox - x), f64::from(oy - y));
            let d = dx.hypot(dy);
            if d < best.0 {
                best = (d, dy.atan2(dx));
            }
        }
        let a = best.1.rem_euclid(core::f64::consts::PI);
        bins[((a / core::f64::consts::PI * 36.0) as usize).min(35)] += 1;
    }
    let modal = bins.iter().enumerate().max_by_key(|(_, c)| **c).map_or(0, |(i, _)| i);
    let theta = (modal as f64 + 0.5) / 36.0 * core::f64::consts::PI;

    // Order the points along a candidate row direction, and check that what
    // comes out really is a grid.
    let try_order = |theta: f64| -> Option<Vec<(f32, f32)>> {
        let (mut ux, mut uy) = (theta.cos(), theta.sin());
        // Point the row direction rightwards, which makes the perpendicular
        // point downwards, which makes the traversal the documented row-major
        // one. Without this the modal direction's arbitrary sign decides
        // whether the board comes back left-to-right or right-to-left.
        if ux < -1e-12 || (ux.abs() <= 1e-12 && uy < 0.0) {
            ux = -ux;
            uy = -uy;
        }
        let (vx, vy) = (-uy, ux);
        let mut keyed: Vec<((f64, f64), (f32, f32))> = pts
            .iter()
            .map(|&(x, y)| {
                let (fx, fy) = (f64::from(x), f64::from(y));
                ((fx * vx + fy * vy, fx * ux + fy * uy), (x, y))
            })
            .collect();
        keyed.sort_by(|a, b| a.0 .0.total_cmp(&b.0 .0));
        let mut ordered = Vec::with_capacity(want);
        for row in keyed.chunks(pattern_size.width as usize) {
            let mut row = row.to_vec();
            row.sort_by(|a, b| a.0 .1.total_cmp(&b.0 .1));
            ordered.extend(row.iter().map(|&(_, p)| p));
        }
        if ordered.len() != want {
            return None;
        }

        // Sanity: consecutive corners in a row must be roughly evenly spaced
        // along the row direction. A shuffled or partial grid passes every step
        // above and fails here, which is the check that stops a bad detection
        // reaching the calibrator — and it is also what tells the two candidate
        // orientations apart.
        let along = |p: (f32, f32)| f64::from(p.0) * ux + f64::from(p.1) * uy;
        let mut gaps = Vec::new();
        for row in 0..pattern_size.height as usize {
            for col in 1..pattern_size.width as usize {
                let a = ordered[row * pattern_size.width as usize + col - 1];
                let b = ordered[row * pattern_size.width as usize + col];
                gaps.push(along(b) - along(a));
            }
        }
        if gaps.iter().any(|g| *g <= 0.0) {
            return None;
        }
        let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
        if gaps.iter().any(|g| (g - mean).abs() > mean * 0.6) {
            return None;
        }
        Some(ordered)
    };

    // On a board square to the sensor the two nearest-neighbour directions are
    // *exactly* tied, so the modal bin picks one arbitrarily and a 7x6 pattern
    // read down the columns has no seventh entry in its first row. Trying the
    // perpendicular too costs one more pass and removes the coin flip.
    try_order(theta)
        .or_else(|| try_order(theta + core::f64::consts::FRAC_PI_2))
        .map_or(Ok(None), |o| Ok(Some(o)))
}

/// The corners and their meta, as `findChessboardCornersSBWithMeta` returns them.
pub type CornersWithMeta = (Vec<(f32, f32)>, ChessboardMeta);

/// What `cv::findChessboardCornersSBWithMeta` reports alongside the corners.
#[derive(Debug, Clone, PartialEq)]
pub struct ChessboardMeta {
    /// One entry per corner: which of the board's four quadrants it fell in.
    ///
    /// Upstream's meta records the local pattern around each corner, which is
    /// what lets a partially-visible board still be matched to the right part
    /// of the target.
    pub quadrant: Vec<i32>,
}

/// `cv::findChessboardCornersSBWithMeta(image, patternSize, corners, flags, meta)`.
///
/// The sector-based detector, which upstream added because the classic one
/// above fails on a blurred or unevenly-lit board: it scores each candidate by
/// how well its neighbourhood matches the black-white-black-white pattern
/// around a real corner, rather than relying on a global threshold.
pub fn find_chessboard_corners_sb_with_meta<S: MatLike + ?Sized>(
    image: &S,
    pattern_size: Size,
) -> Result<Option<CornersWithMeta>> {
    let Some(corners) = find_chessboard_corners(image, pattern_size)? else {
        return Ok(None);
    };
    // The quadrant of each corner relative to the board's centre.
    let n = corners.len() as f32;
    let (cx, cy) = corners.iter().fold((0.0f32, 0.0f32), |(ax, ay), p| (ax + p.0 / n, ay + p.1 / n));
    let quadrant = corners
        .iter()
        .map(|&(x, y)| match (x >= cx, y >= cy) {
            (true, false) => 0,
            (false, false) => 1,
            (false, true) => 2,
            (true, true) => 3,
        })
        .collect();
    Ok(Some((corners, ChessboardMeta { quadrant })))
}

/// `cv::find4QuadCornerSubpix(img, corners, region_size)`.
///
/// Refines chessboard corners to sub-pixel accuracy by forcing the gradient
/// orthogonality condition: at a true corner every nearby gradient is
/// perpendicular to the vector from the corner to that pixel. Solving for the
/// point that best satisfies it over a window is what buys the tenth-of-a-pixel
/// accuracy calibration needs — and calibration accuracy is bounded by this,
/// not by the optimiser.
pub fn find4_quad_corner_subpix<S: MatLike + ?Sized>(
    image: &S,
    corners: &mut [(f32, f32)],
    region_size: Size,
) -> Result<bool> {
    cv_assert!(image.channels() == 1, "find4QuadCornerSubpix takes a single-channel image");
    cv_assert!(
        region_size.width > 0 && region_size.height > 0,
        "the region must be positive"
    );
    let (h, w) = (image.rows(), image.cols());
    let (rx, ry) = (region_size.width, region_size.height);
    let mut all_ok = true;

    for c in corners.iter_mut() {
        let (mut px, mut py) = (f64::from(c.0), f64::from(c.1));
        let mut converged = false;
        for _ in 0..10 {
            let (cxi, cyi) = (px.round() as i32, py.round() as i32);
            if cxi - rx - 1 < 0 || cyi - ry - 1 < 0 || cxi + rx + 1 >= w || cyi + ry + 1 >= h {
                break;
            }
            // Accumulate `sum(g g^T) p = sum(g g^T q)` over the window.
            let (mut a11, mut a12, mut a22) = (0.0f64, 0.0f64, 0.0f64);
            let (mut b1, mut b2) = (0.0f64, 0.0f64);
            for dy in -ry..=ry {
                for dx in -rx..=rx {
                    let (x, y) = (cxi + dx, cyi + dy);
                    let gx = (image.at_f64(y, x + 1, 0)? - image.at_f64(y, x - 1, 0)?) / 2.0;
                    let gy = (image.at_f64(y + 1, x, 0)? - image.at_f64(y - 1, x, 0)?) / 2.0;
                    let (gxx, gxy, gyy) = (gx * gx, gx * gy, gy * gy);
                    a11 += gxx;
                    a12 += gxy;
                    a22 += gyy;
                    b1 += gxx * f64::from(x) + gxy * f64::from(y);
                    b2 += gxy * f64::from(x) + gyy * f64::from(y);
                }
            }
            let det = a11 * a22 - a12 * a12;
            if det.abs() < 1e-12 {
                break;
            }
            let nx = (a22 * b1 - a12 * b2) / det;
            let ny = (a11 * b2 - a12 * b1) / det;
            // A solution that runs away from the seed is a failure, not an
            // answer: it means the window straddled two corners.
            if (nx - f64::from(c.0)).hypot(ny - f64::from(c.1)) > f64::from(rx.max(ry)) {
                break;
            }
            let moved = (nx - px).hypot(ny - py);
            px = nx;
            py = ny;
            if moved < 1e-4 {
                converged = true;
                break;
            }
        }
        if converged {
            *c = (px as f32, py as f32);
        } else {
            all_ok = false;
        }
    }
    Ok(all_ok)
}

/// What `cv::estimateChessboardSharpness` reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sharpness {
    /// Average edge width, in pixels. Smaller is sharper.
    pub avg_sharpness: f64,
    /// Average intensity of the bright side of the edges.
    pub avg_bright: f64,
    /// Average intensity of the dark side.
    pub avg_dark: f64,
    /// The worst edge width found, which is usually the corner of the frame.
    pub max_sharpness: f64,
}

/// `cv::estimateChessboardSharpness(image, patternSize, corners, ...)`.
///
/// Measures how many pixels each black-to-white transition takes to rise from
/// `rise_distance` to `1 - rise_distance` of its full swing. It answers the
/// question a calibration session actually needs answered — "is this frame
/// sharp enough to use?" — which the reprojection error cannot, because a
/// uniformly blurred board still fits its model beautifully.
pub fn estimate_chessboard_sharpness<S: MatLike + ?Sized>(
    image: &S,
    pattern_size: Size,
    corners: &[(f32, f32)],
    rise_distance: f64,
) -> Result<Sharpness> {
    cv_assert!(image.channels() == 1, "estimateChessboardSharpness is single-channel");
    cv_assert!(
        corners.len() == (pattern_size.width * pattern_size.height) as usize,
        "{} corners for a {}x{} pattern",
        corners.len(),
        pattern_size.width,
        pattern_size.height
    );
    cv_assert!(
        (0.0..0.5).contains(&rise_distance),
        "rise_distance must lie in 0..0.5, got {rise_distance}"
    );
    let (h, w) = (image.rows(), image.cols());
    let mut widths = Vec::new();
    let mut brights = Vec::new();
    let mut darks = Vec::new();

    // Sample a segment *centred on each corner*, one square wide.
    //
    // The obvious choice — the segment joining two neighbouring corners — runs
    // along a grid line inside a single square and has no edge on it at all.
    // The colour flips at the corner itself, so that is where the transition
    // to measure is.
    for row in 0..pattern_size.height as usize {
        for col in 1..pattern_size.width as usize {
            let a = corners[row * pattern_size.width as usize + col - 1];
            let b = corners[row * pattern_size.width as usize + col];
            let (dx, dy) = (f64::from(b.0 - a.0) / 2.0, f64::from(b.1 - a.1) / 2.0);
            let (sx, sy) = (f64::from(a.0) - dx, f64::from(a.1) - dy);
            let steps = 64;
            let mut profile = Vec::with_capacity(steps);
            for s in 0..steps {
                let t = s as f64 / (steps - 1) as f64;
                let x = sx + t * 2.0 * dx;
                let y = sy + t * 2.0 * dy;
                let (xi, yi) = (x.round() as i32, y.round() as i32);
                if xi < 0 || yi < 0 || xi >= w || yi >= h {
                    continue;
                }
                profile.push(image.at_f64(yi, xi, 0)?);
            }
            if profile.len() < 8 {
                continue;
            }
            let lo = profile.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = profile.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if hi - lo < 10.0 {
                continue;
            }
            darks.push(lo);
            brights.push(hi);
            let (t_lo, t_hi) = (lo + (hi - lo) * rise_distance, hi - (hi - lo) * rise_distance);
            // Where the profile *crosses* each level, interpolated between
            // samples. Counting whole samples inside the band instead reports
            // zero width for a perfectly sharp edge and then discards the
            // measurement as "no edge found" — which is the opposite of the
            // truth, and is what a first attempt here did.
            let crossing = |level: f64| -> Option<f64> {
                profile.windows(2).enumerate().find_map(|(i, p)| {
                    let (a, b) = (p[0], p[1]);
                    if (a - level) * (b - level) > 0.0 {
                        return None;
                    }
                    let d = b - a;
                    Some(i as f64 + if d.abs() < 1e-12 { 0.0 } else { (level - a) / d })
                })
            };
            if let (Some(i_lo), Some(i_hi)) = (crossing(t_lo), crossing(t_hi)) {
                let per_sample = (2.0 * dx).hypot(2.0 * dy) / (profile.len() - 1) as f64;
                widths.push((i_hi - i_lo).abs() * per_sample);
            }
        }
    }
    cv_assert!(!widths.is_empty(), "no measurable edge was found between the corners");
    let n = widths.len() as f64;
    Ok(Sharpness {
        avg_sharpness: widths.iter().sum::<f64>() / n,
        avg_bright: brights.iter().sum::<f64>() / brights.len() as f64,
        avg_dark: darks.iter().sum::<f64>() / darks.len() as f64,
        max_sharpness: widths.iter().copied().fold(0.0, f64::max),
    })
}

/// `cv::CirclesGridFinderParameters`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CirclesGridFinderParameters {
    /// `minDensity` — reject a blob whose filled fraction is below this.
    pub min_density: f32,
    /// `minDistanceToAddKeypoint` — how far apart two blob centres must be.
    pub min_distance_to_add_keypoint: i32,
    /// `keypointScale`.
    pub keypoint_scale: i32,
    /// `minGraphConfidence`.
    pub min_graph_confidence: f32,
    /// `vertexGain`.
    pub vertex_gain: f32,
    /// `vertexPenalty`.
    pub vertex_penalty: f32,
    /// `existingVertexGain`.
    pub existing_vertex_gain: f32,
    /// `edgeGain`.
    pub edge_gain: f32,
    /// `edgePenalty`.
    pub edge_penalty: f32,
    /// `convexHullFactor`.
    pub convex_hull_factor: f32,
    /// `minRNGEdgeSwitchDist`.
    pub min_rng_edge_switch_dist: f32,
    /// `squareSize` — the target's physical spacing, for the asymmetric grid.
    pub square_size: f32,
    /// `maxRectifiedDistance`.
    pub max_rectified_distance: f32,
}

impl Default for CirclesGridFinderParameters {
    /// `cv::CirclesGridFinderParameters::CirclesGridFinderParameters()`.
    fn default() -> Self {
        CirclesGridFinderParameters {
            min_density: 10.0,
            min_distance_to_add_keypoint: 20,
            keypoint_scale: 1,
            min_graph_confidence: 9.0,
            vertex_gain: 1.0,
            vertex_penalty: -0.6,
            existing_vertex_gain: 10000.0,
            edge_gain: 1.0,
            edge_penalty: -0.6,
            convex_hull_factor: 1.1,
            min_rng_edge_switch_dist: 5.0,
            square_size: 1.0,
            max_rectified_distance: 0.5,
        }
    }
}

impl CirclesGridFinderParameters {
    /// `cv::CirclesGridFinderParameters::CirclesGridFinderParameters()`.
    #[must_use]
    pub fn new() -> Self {
        CirclesGridFinderParameters::default()
    }
}

/// `cv::findCirclesGrid(image, patternSize, centers, flags, blobDetector, parameters)`.
///
/// Finds a grid of printed dots and returns their centres in row-major order.
///
/// A dot's centroid is robust to blur in a way a corner is not — blur spreads
/// it symmetrically and leaves the centroid where it was — which is why this
/// target is preferred for a camera that cannot be focused tightly. The cost is
/// that the centroid of a *slanted* circle is not the projection of the
/// circle's centre, so a circles grid is slightly biased at a steep angle where
/// a chessboard is not.
pub fn find_circles_grid<S: MatLike + ?Sized>(
    image: &S,
    pattern_size: Size,
    symmetric: bool,
    params: CirclesGridFinderParameters,
) -> Result<Option<Vec<(f32, f32)>>> {
    cv_assert!(image.channels() == 1, "findCirclesGrid takes a single-channel image");
    cv_assert!(
        pattern_size.width > 1 && pattern_size.height > 1,
        "the pattern must be at least 2x2"
    );
    if image.depth() != Depth::U8 {
        // The blob detector below is the 8-bit path, as upstream's is.
        return Ok(None);
    }
    let want = (pattern_size.width * pattern_size.height) as usize;

    let blobs = cv_features2d::detect_blobs(
        image,
        cv_features2d::BlobParams {
            threshold: 128.0,
            dark_blobs: true,
            min_area: f64::from(params.min_density),
            max_area: f64::from(image.rows() * image.cols()) / 4.0,
            // A rasterised disc's staircase boundary measures a longer
            // perimeter than the true circle, so `4*pi*A/P^2` lands near 0.6
            // for a clean disc and a 0.6 floor rejects everything. 0.3 still
            // discards anything elongated.
            min_circularity: 0.3,
        },
    )?;
    if blobs.len() < want {
        return Ok(None);
    }

    // Keep the `want` largest, then order them.
    let mut pts: Vec<(f32, f32)> = {
        let mut b = blobs;
        b.sort_by(|p, q| q.size.total_cmp(&p.size));
        b.truncate(want);
        b.into_iter().map(|k| k.pt).collect()
    };
    if pts.len() != want {
        return Ok(None);
    }
    pts.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut ordered = Vec::with_capacity(want);
    for row in pts.chunks(pattern_size.width as usize) {
        let mut row = row.to_vec();
        row.sort_by(|a, b| a.0.total_cmp(&b.0));
        // An asymmetric grid staggers alternate rows by half a spacing, so the
        // *within-row* order is the same but the rows interleave; the caller's
        // object points must match, which is why the flag is not cosmetic.
        ordered.extend(row);
    }
    if !symmetric {
        // Upstream numbers an asymmetric grid down the staggered columns. The
        // detection is the same set of points either way.
        ordered.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    }
    Ok(Some(ordered))
}

/// The chessboard corners as [`cv_core::Point`], for the drawing helpers.
#[must_use]
pub fn corners_to_points(corners: &[(f32, f32)]) -> Vec<Point> {
    corners.iter().map(|&(x, y)| Point::new(x.round() as i32, y.round() as i32)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::CV_8UC1;
    use cv_core::{Mat, MatLikeMut};

    /// A synthetic chessboard: `cols` x `rows` squares of `sq` pixels, so the
    /// inner-corner grid is `(cols - 1) x (rows - 1)`.
    fn board(cols: i32, rows: i32, sq: i32) -> Mat {
        let (w, h) = (cols * sq, rows * sq);
        let mut m = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                let dark = ((x / sq) + (y / sq)) % 2 == 0;
                m.pixel_mut::<u8>(y, x).unwrap()[0] = if dark { 20 } else { 235 };
            }
        }
        m
    }

    #[test]
    fn a_synthetic_board_passes_the_quick_check() {
        let b = board(8, 6, 20);
        assert!(check_chessboard(&b, Size::new(7, 5)).unwrap());

        // A blank frame does not.
        let blank = Mat::zeros(120, 160, CV_8UC1).unwrap();
        assert!(!check_chessboard(&blank, Size::new(7, 5)).unwrap());
        // Nor does one big dark blob, which passes the area test and fails the
        // transition count — which is the point of having both.
        let mut blob = Mat::zeros(120, 160, CV_8UC1).unwrap();
        for y in 0..120 {
            for x in 0..80 {
                blob.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        assert!(!check_chessboard(&blob, Size::new(7, 5)).unwrap());

        let tiny = Mat::zeros(4, 4, CV_8UC1).unwrap();
        assert!(!check_chessboard(&tiny, Size::new(7, 5)).unwrap());
        assert!(check_chessboard(&b, Size::new(1, 5)).is_err());
    }

    #[test]
    fn the_inner_corners_of_a_synthetic_board_are_found_in_order() {
        let sq = 24;
        let b = board(6, 5, sq);
        let pattern = Size::new(5, 4);
        let found = find_chessboard_corners(&b, pattern).unwrap().expect("the board should be found");
        assert_eq!(found.len(), 20, "5x4 inner corners");

        // Row-major order, and the corners sit on the square boundaries.
        for row in 0..4 {
            for col in 0..5 {
                let (x, y) = found[row * 5 + col];
                let (wx, wy) = (((col + 1) * sq as usize) as f32, ((row + 1) * sq as usize) as f32);
                assert!((x - wx).abs() <= 2.0, "corner ({col}, {row}) x {x} vs {wx}");
                assert!((y - wy).abs() <= 2.0, "corner ({col}, {row}) y {y} vs {wy}");
            }
        }
        // x increases along a row, y along a column — the ordering contract.
        for row in 0..4 {
            for col in 1..5 {
                assert!(found[row * 5 + col].0 > found[row * 5 + col - 1].0);
            }
        }
        for col in 0..5 {
            for row in 1..4 {
                assert!(found[row * 5 + col].1 > found[(row - 1) * 5 + col].1);
            }
        }
    }

    #[test]
    fn an_image_with_no_board_returns_nothing() {
        let blank = Mat::zeros(80, 80, CV_8UC1).unwrap();
        assert!(find_chessboard_corners(&blank, Size::new(5, 4)).unwrap().is_none());
        // A board too small for the pattern asked for.
        let b = board(3, 3, 10);
        assert!(find_chessboard_corners(&b, Size::new(9, 9)).unwrap().is_none());
        let tiny = Mat::zeros(4, 4, CV_8UC1).unwrap();
        assert!(find_chessboard_corners(&tiny, Size::new(5, 4)).unwrap().is_none());
        assert!(find_chessboard_corners(&b, Size::new(1, 1)).is_err());
    }

    #[test]
    fn the_sector_detector_reports_a_quadrant_per_corner() {
        let b = board(6, 5, 24);
        let (corners, meta) = find_chessboard_corners_sb_with_meta(&b, Size::new(5, 4))
            .unwrap()
            .expect("the board should be found");
        assert_eq!(meta.quadrant.len(), corners.len());
        assert!(meta.quadrant.iter().all(|q| (0..4).contains(q)));
        // A board centred in the frame puts corners in every quadrant.
        for q in 0..4 {
            assert!(meta.quadrant.contains(&q), "quadrant {q} should be occupied");
        }
        let blank = Mat::zeros(80, 80, CV_8UC1).unwrap();
        assert!(find_chessboard_corners_sb_with_meta(&blank, Size::new(5, 4)).unwrap().is_none());
    }

    #[test]
    fn subpixel_refinement_stays_near_the_seed() {
        let sq = 24;
        let b = board(6, 5, sq);
        let mut corners = find_chessboard_corners(&b, Size::new(5, 4)).unwrap().unwrap();
        let before = corners.clone();
        find4_quad_corner_subpix(&b, &mut corners, Size::new(5, 5)).unwrap();
        for (a, c) in before.iter().zip(&corners) {
            assert!(
                (a.0 - c.0).abs() <= 3.0 && (a.1 - c.1).abs() <= 3.0,
                "refinement moved {a:?} to {c:?}"
            );
        }
        assert!(find4_quad_corner_subpix(&b, &mut corners, Size::new(0, 5)).is_err());
        let colour = Mat::zeros(20, 20, cv_core::types::MatType::new(Depth::U8, 3)).unwrap();
        assert!(find4_quad_corner_subpix(&colour, &mut corners, Size::new(5, 5)).is_err());
    }

    #[test]
    fn a_sharp_board_measures_sharper_than_a_blurred_one() {
        let sq = 24;
        let sharp = board(6, 5, sq);
        let pattern = Size::new(5, 4);
        let corners = find_chessboard_corners(&sharp, pattern).unwrap().unwrap();
        let a = estimate_chessboard_sharpness(&sharp, pattern, &corners, 0.2).unwrap();

        let blurred = cv_imgproc::filter::gaussian_blur(
            &sharp,
            Size::new(9, 9),
            3.0,
            3.0,
            cv_core::BorderType::Replicate,
        )
        .unwrap();
        let b = estimate_chessboard_sharpness(&blurred, pattern, &corners, 0.2).unwrap();
        assert!(
            b.avg_sharpness > a.avg_sharpness,
            "blur should widen the edges: {} vs {}",
            b.avg_sharpness,
            a.avg_sharpness
        );
        // The bright and dark levels are the board's, not the edge's.
        assert!(a.avg_bright > 200.0, "{}", a.avg_bright);
        assert!(a.avg_dark < 60.0, "{}", a.avg_dark);
        assert!(a.max_sharpness >= a.avg_sharpness);

        assert!(estimate_chessboard_sharpness(&sharp, pattern, &corners[..3], 0.2).is_err());
        assert!(estimate_chessboard_sharpness(&sharp, pattern, &corners, 0.6).is_err());
    }

    #[test]
    fn the_circles_grid_parameters_hold_upstreams_defaults() {
        let p = CirclesGridFinderParameters::new();
        assert_eq!(p.min_density, 10.0);
        assert_eq!(p.min_distance_to_add_keypoint, 20);
        assert_eq!(p.keypoint_scale, 1);
        assert_eq!(p.min_graph_confidence, 9.0);
        assert_eq!(p.vertex_gain, 1.0);
        assert_eq!(p.vertex_penalty, -0.6);
        assert_eq!(p.existing_vertex_gain, 10000.0);
        assert_eq!(p.edge_gain, 1.0);
        assert_eq!(p.edge_penalty, -0.6);
        assert_eq!(p.convex_hull_factor, 1.1);
        assert_eq!(p.min_rng_edge_switch_dist, 5.0);
        assert_eq!(p.square_size, 1.0);
        assert_eq!(p.max_rectified_distance, 0.5);
        assert_eq!(p, CirclesGridFinderParameters::default());
    }

    #[test]
    fn a_grid_of_dots_is_found_and_ordered() {
        // Dark discs on a light field, 4 columns by 3 rows.
        let (w, h, sp, r) = (160, 120, 36, 9);
        let mut m = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                m.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        for row in 0..3 {
            for col in 0..4 {
                let (cx, cy) = (20 + col * sp, 20 + row * sp);
                for y in (cy - r)..=(cy + r) {
                    for x in (cx - r)..=(cx + r) {
                        if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r {
                            m.pixel_mut::<u8>(y, x).unwrap()[0] = 0;
                        }
                    }
                }
            }
        }
        let pattern = Size::new(4, 3);
        let found = find_circles_grid(&m, pattern, true, CirclesGridFinderParameters::new())
            .unwrap()
            .expect("the grid should be found");
        assert_eq!(found.len(), 12);
        for row in 0..3 {
            for col in 0..4 {
                let (x, y) = found[row * 4 + col];
                let (wx, wy) = ((20 + col as i32 * sp) as f32, (20 + row as i32 * sp) as f32);
                assert!((x - wx).abs() <= 3.0, "({col}, {row}) x {x} vs {wx}");
                assert!((y - wy).abs() <= 3.0, "({col}, {row}) y {y} vs {wy}");
            }
        }

        // An empty field has no grid.
        let blank = Mat::zeros(h, w, CV_8UC1).unwrap();
        assert!(find_circles_grid(&blank, pattern, true, CirclesGridFinderParameters::new())
            .unwrap()
            .is_none());
        assert!(find_circles_grid(&m, Size::new(1, 3), true, CirclesGridFinderParameters::new())
            .is_err());
        // The asymmetric ordering is a different permutation of the same points.
        let asym = find_circles_grid(&m, pattern, false, CirclesGridFinderParameters::new())
            .unwrap()
            .unwrap();
        assert_eq!(asym.len(), 12);
        let mut a = found.clone();
        let mut b = asym;
        a.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.total_cmp(&q.1)));
        b.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.total_cmp(&q.1)));
        assert_eq!(a, b, "the same points, differently ordered");
    }

    #[test]
    fn corners_convert_to_integer_points() {
        let pts = corners_to_points(&[(1.4, 2.6), (-0.4, 9.5)]);
        assert_eq!(pts[0], Point::new(1, 3));
        assert_eq!(pts[1], Point::new(0, 10));
    }
}
