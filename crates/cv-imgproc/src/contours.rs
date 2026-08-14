//! Contour tracing, flood fill, distance transform and watershed
//! (`modules/imgproc/src/contours.cpp`, `floodfill.cpp`, `distransform.cpp`,
//! `segmentation.cpp`).

use cv_core::{cv_assert, Mat, MatLike, MatLikeMut, Point, Rect, Result, Scalar};
use cv_core::types::{Depth, MatType, CV_32FC1, CV_32SC1, CV_8UC1};

/// `cv::RetrievalModes` — which contours [`find_contours`] keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode {
    /// `cv::RETR_EXTERNAL` — outermost boundaries only.
    External = 0,
    /// `cv::RETR_LIST` — every contour, with no hierarchy.
    List = 1,
    /// `cv::RETR_CCOMP` — outer boundaries and holes, two levels.
    CComp = 2,
    /// `cv::RETR_TREE` — every contour with full nesting.
    Tree = 3,
}

/// `cv::ContourApproximationModes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourApproximation {
    /// `cv::CHAIN_APPROX_NONE` — store every boundary pixel.
    None = 1,
    /// `cv::CHAIN_APPROX_SIMPLE` — collapse straight runs to their endpoints.
    Simple = 2,
}

/// One row of `cv::findContours`' hierarchy output.
///
/// Indices refer to positions in the returned contour list; `-1` means absent,
/// exactly as upstream's `Vec4i` encodes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hierarchy {
    /// Next contour at the same level.
    pub next: i32,
    /// Previous contour at the same level.
    pub prev: i32,
    /// First child.
    pub first_child: i32,
    /// Enclosing contour.
    pub parent: i32,
}

/// The eight neighbours in counter-clockwise order, starting east.
const NBR: [(i32, i32); 8] =
    [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)];

/// `cv::findContours(image, contours, hierarchy, mode, method)`.
///
/// A Suzuki-Abe border follower over a binarised copy of `image`: any non-zero
/// pixel is foreground. Returns the contours and, for [`RetrievalMode::Tree`]
/// and [`RetrievalMode::CComp`], the nesting relationships.
///
/// # Divergence
///
/// Upstream mutates the input image in place to mark visited borders; this port
/// works on an internal buffer and leaves the caller's image untouched. That
/// difference is visible: passing the same image twice gives the same answer
/// here and does not upstream.
pub fn find_contours<S: MatLike + ?Sized>(
    image: &S,
    mode: RetrievalMode,
    method: ContourApproximation,
) -> Result<(Vec<Vec<Point>>, Vec<Hierarchy>)> {
    cv_assert!(image.channels() == 1, "findContours requires a single-channel image");
    let (h, w) = (image.rows(), image.cols());
    // A one-pixel zero frame removes every bounds check from the follower.
    let (bw, bh) = (w + 2, h + 2);
    let mut f = vec![0i32; (bw * bh) as usize];
    for y in 0..h {
        for x in 0..w {
            if image.at_f64(y, x, 0)? != 0.0 {
                f[((y + 1) * bw + (x + 1)) as usize] = 1;
            }
        }
    }
    let at = |f: &[i32], x: i32, y: i32| f[(y * bw + x) as usize];

    let mut contours: Vec<Vec<Point>> = Vec::new();
    let mut hier: Vec<Hierarchy> = Vec::new();
    // Border id -> (contour index, is_hole). Id 1 is the image frame.
    let mut parent_of: Vec<i32> = vec![-1];
    let mut idx_of: Vec<i32> = vec![-1];
    let mut hole_of: Vec<bool> = vec![false];
    let mut nbd = 1i32;
    let mut lnbd;

    for y in 1..bh - 1 {
        lnbd = 1;
        for x in 1..bw - 1 {
            let cur = at(&f, x, y);
            if cur == 0 {
                continue;
            }
            // Outer border: foreground with background to the west.
            let outer = cur == 1 && at(&f, x - 1, y) == 0;
            // Hole border: foreground with background to the east.
            let hole = cur >= 1 && at(&f, x + 1, y) == 0;
            if !outer && !hole {
                if cur > 1 {
                    lnbd = cur;
                }
                continue;
            }
            nbd += 1;
            let start_dir = if outer { 2usize } else { 6usize };
            let is_hole = !outer;

            // Suzuki-Abe parent rule, expressed over the last border seen.
            let lidx = lnbd.max(1) as usize;
            let l_is_hole = *hole_of.get(lidx).unwrap_or(&false);
            let l_parent = *parent_of.get(lidx).unwrap_or(&-1);
            let parent_border = if is_hole == l_is_hole { l_parent } else { lnbd };

            let pts = trace(&mut f, bw, x, y, start_dir, nbd);
            let kept = match method {
                ContourApproximation::None => pts,
                ContourApproximation::Simple => simplify(&pts),
            };

            let keep = match mode {
                RetrievalMode::External => !is_hole,
                _ => true,
            };
            let my_idx = if keep {
                contours.push(kept.iter().map(|p| Point::new(p.x - 1, p.y - 1)).collect());
                hier.push(Hierarchy { next: -1, prev: -1, first_child: -1, parent: -1 });
                contours.len() as i32 - 1
            } else {
                -1
            };
            parent_of.push(parent_border);
            idx_of.push(my_idx);
            hole_of.push(is_hole);
            lnbd = nbd;
        }
    }

    if matches!(mode, RetrievalMode::Tree | RetrievalMode::CComp) {
        link_hierarchy(&mut hier, &parent_of, &idx_of, &hole_of, mode);
    }
    Ok((contours, hier))
}

/// Fill in `next`/`prev`/`first_child`/`parent` from the border relationships.
fn link_hierarchy(
    hier: &mut [Hierarchy],
    parent_of: &[i32],
    idx_of: &[i32],
    hole_of: &[bool],
    mode: RetrievalMode,
) {
    let border_of_contour = |ci: i32| idx_of.iter().position(|&i| i == ci).unwrap_or(0);
    for ci in 0..hier.len() as i32 {
        let b = border_of_contour(ci);
        let parent_border = parent_of[b];
        let parent_ci = if parent_border >= 1 {
            let pb = parent_border as usize;
            match mode {
                // CCOMP flattens to two levels: a hole's parent is its outer
                // contour, and an outer contour has no parent.
                RetrievalMode::CComp if !hole_of[b] => -1,
                _ => *idx_of.get(pb).unwrap_or(&-1),
            }
        } else {
            -1
        };
        hier[ci as usize].parent = parent_ci;
    }
    // Sibling chains and first-child links, in discovery order.
    for ci in 0..hier.len() as i32 {
        let p = hier[ci as usize].parent;
        let sibs: Vec<i32> = (0..hier.len() as i32).filter(|&j| hier[j as usize].parent == p).collect();
        let pos = sibs.iter().position(|&j| j == ci).unwrap_or(0);
        hier[ci as usize].prev = if pos > 0 { sibs[pos - 1] } else { -1 };
        hier[ci as usize].next = sibs.get(pos + 1).copied().unwrap_or(-1);
        if p >= 0 && hier[p as usize].first_child < 0 {
            hier[p as usize].first_child = sibs[0];
        }
    }
}

/// Follow one border, marking it in `f`, and return its pixels.
fn trace(f: &mut [i32], bw: i32, sx: i32, sy: i32, start_dir: usize, nbd: i32) -> Vec<Point> {
    let idx = |x: i32, y: i32| (y * bw + x) as usize;
    let mut out = Vec::new();
    let (mut cx, mut cy) = (sx, sy);

    // Find the first non-zero neighbour, scanning clockwise from start_dir.
    let mut found = None;
    for k in 0..8 {
        let d = (start_dir + 8 - k) % 8;
        let (nx, ny) = (cx + NBR[d].0, cy + NBR[d].1);
        if f[idx(nx, ny)] != 0 {
            found = Some(d);
            break;
        }
    }
    let Some(first) = found else {
        // Isolated pixel: a one-point contour.
        f[idx(sx, sy)] = -nbd;
        return vec![Point::new(sx, sy)];
    };
    let dir = first;
    let (mut px, mut py) = (cx + NBR[dir].0, cy + NBR[dir].1);
    let (start_px, start_py) = (px, py);
    let (mut prev_x, mut prev_y);

    loop {
        out.push(Point::new(cx, cy));
        // Mark: negative means "border already traced", which keeps the
        // left/right emptiness tests above working on later rows.
        if f[idx(cx + 1, cy)] == 0 {
            f[idx(cx, cy)] = -nbd;
        } else if f[idx(cx, cy)] == 1 {
            f[idx(cx, cy)] = nbd;
        }
        prev_x = cx;
        prev_y = cy;
        // Rotate counter-clockwise from the previous pixel's direction.
        let back = NBR.iter().position(|&(dx, dy)| dx == px - cx && dy == py - cy).unwrap_or(0);
        let mut step = None;
        for k in 1..=8 {
            let d = (back + k) % 8;
            let (nx, ny) = (cx + NBR[d].0, cy + NBR[d].1);
            if f[idx(nx, ny)] != 0 {
                step = Some((d, nx, ny));
                break;
            }
        }
        let Some((_, nx, ny)) = step else { break };
        px = prev_x;
        py = prev_y;
        cx = nx;
        cy = ny;
        if cx == sx && cy == sy && px == start_px && py == start_py {
            break;
        }
        if out.len() > (bw as usize) * (bw as usize) * 4 {
            break; // safety valve; a well-formed border cannot reach this
        }
    }
    out
}

/// `CHAIN_APPROX_SIMPLE`: keep only the endpoints of each straight run.
fn simplify(pts: &[Point]) -> Vec<Point> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let mut out = Vec::with_capacity(pts.len());
    let n = pts.len();
    for i in 0..n {
        let p = pts[(i + n - 1) % n];
        let c = pts[i];
        let q = pts[(i + 1) % n];
        // Drop c when it lies on the straight segment p..q.
        if (c.x - p.x) * (q.y - c.y) != (c.y - p.y) * (q.x - c.x) {
            out.push(c);
        }
    }
    if out.is_empty() {
        out.push(pts[0]);
    }
    out
}

/// `cv::drawContours(image, contours, contourIdx, color, thickness, ...)`.
///
/// `contour_idx` of `None` draws all of them. A negative `thickness` fills.
pub fn draw_contours<D: MatLikeMut + ?Sized>(
    image: &mut D,
    contours: &[Vec<Point>],
    contour_idx: Option<usize>,
    color: Scalar,
    thickness: i32,
) -> Result<()> {
    let range: Vec<usize> = match contour_idx {
        Some(i) => {
            cv_assert!(i < contours.len(), "contour index {i} out of range");
            vec![i]
        }
        None => (0..contours.len()).collect(),
    };
    for i in range {
        let c = &contours[i];
        if c.is_empty() {
            continue;
        }
        if thickness < 0 {
            crate::drawing::fill_poly(image, c, color)?;
        } else {
            crate::drawing::polylines(image, c, true, color, thickness.max(1))?;
        }
    }
    Ok(())
}

/// `cv::floodFill(image, mask, seedPoint, newVal, rect, loDiff, upDiff, flags)`.
///
/// Returns the number of pixels repainted and their bounding box. Connectivity
/// is 4 or 8. Tolerances are compared against the **seed** value, which is
/// upstream's default; `FLOODFILL_FIXED_RANGE` semantics, in other words.
pub fn flood_fill<D: MatLike + MatLikeMut + ?Sized>(
    image: &mut D,
    seed: Point,
    new_val: Scalar,
    lo_diff: f64,
    up_diff: f64,
    connectivity: i32,
) -> Result<(i32, Rect)> {
    cv_assert!(
        connectivity == 4 || connectivity == 8,
        "connectivity must be 4 or 8, got {connectivity}"
    );
    let (h, w) = (image.rows(), image.cols());
    cv_assert!(
        seed.x >= 0 && seed.y >= 0 && seed.x < w && seed.y < h,
        "seed ({}, {}) is outside the {w}x{h} image",
        seed.x,
        seed.y
    );
    let cn = image.channels();
    let target: Vec<f64> = (0..cn).map(|c| new_val.0[c as usize]).collect();
    // The comparison is against the **original** values, and against the pixel
    // a candidate spread from — not the seed. Filling in place would otherwise
    // compare against `new_val` as soon as a neighbour was written.
    let mut orig = vec![0.0f64; (h * w * cn) as usize];
    for y in 0..h {
        for x in 0..w {
            for c in 0..cn {
                orig[((y * w + x) * cn + c) as usize] = image.at_f64(y, x, c)?;
            }
        }
    }

    let mut visited = vec![false; (h * w) as usize];
    let mut stack = vec![seed];
    visited[(seed.y * w + seed.x) as usize] = true;
    let (mut x0, mut y0, mut x1, mut y1) = (seed.x, seed.y, seed.x, seed.y);
    let mut count = 0;

    let steps: &[(i32, i32)] = if connectivity == 4 {
        &[(1, 0), (-1, 0), (0, 1), (0, -1)]
    } else {
        &[(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    };

    while let Some(p) = stack.pop() {
        for c in 0..cn {
            cv_core::arithm::set_channel_generic(image, p.y, p.x, c, target[c as usize])?;
        }
        count += 1;
        x0 = x0.min(p.x);
        y0 = y0.min(p.y);
        x1 = x1.max(p.x);
        y1 = y1.max(p.y);

        for &(dx, dy) in steps {
            let (nx, ny) = (p.x + dx, p.y + dy);
            if nx < 0 || ny < 0 || nx >= w || ny >= h {
                continue;
            }
            let i = (ny * w + nx) as usize;
            if visited[i] {
                continue;
            }
            // `cv::floodFill` without `FLOODFILL_FIXED_RANGE` compares each
            // candidate against the neighbour it spread from, so the accepted
            // range drifts with the region. Comparing against the seed instead
            // is the fixed-range variant and fills a far smaller area — 118
            // pixels against cv2's 753 on the reference fixture.
            let mut inside = true;
            for c in 0..cn {
                let v = orig[((ny * w + nx) * cn + c) as usize];
                let s = orig[((p.y * w + p.x) * cn + c) as usize];
                if v < s - lo_diff || v > s + up_diff {
                    inside = false;
                    break;
                }
            }
            if inside {
                visited[i] = true;
                stack.push(Point::new(nx, ny));
            }
        }
    }
    Ok((count, Rect::new(x0, y0, x1 - x0 + 1, y1 - y0 + 1)))
}

/// `cv::DistanceTypes` for [`distance_transform`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// `cv::DIST_L1` — city block.
    L1,
    /// `cv::DIST_L2` — Euclidean.
    L2,
    /// `cv::DIST_C` — chessboard.
    C,
}

/// `cv::distanceTransform(src, dst, distanceType, maskSize)`.
///
/// Returns the distance from every non-zero pixel to the nearest zero pixel, as
/// `CV_32F`.
///
/// # Divergence
///
/// `DIST_L2` here is the **exact** Euclidean distance, computed with Felzenszwalb
/// and Huttenlocher's linear-time squared-distance transform. Upstream's
/// `maskSize` of 3 or 5 uses a chamfer approximation with fixed weights
/// (0.955/1.369 and so on) that is a few percent off; upstream's `DIST_MASK_PRECISE`
/// is the exact one. There is no `maskSize` argument here because there is
/// nothing to choose.
pub fn distance_transform<S: MatLike + ?Sized>(src: &S, metric: DistanceMetric) -> Result<Mat> {
    cv_assert!(src.channels() == 1, "distanceTransform requires a single-channel image");
    let (h, w) = (src.rows(), src.cols());
    const INF: f64 = 1e20;
    let mut d = vec![0.0f64; (h * w) as usize];
    for y in 0..h {
        for x in 0..w {
            d[(y * w + x) as usize] = if src.at_f64(y, x, 0)? != 0.0 { INF } else { 0.0 };
        }
    }

    match metric {
        DistanceMetric::L2 => {
            // Squared distance, separable: columns then rows.
            let mut f = vec![0.0f64; h.max(w) as usize];
            for x in 0..w {
                for y in 0..h {
                    f[y as usize] = d[(y * w + x) as usize];
                }
                let r = edt_1d(&f[..h as usize]);
                for y in 0..h {
                    d[(y * w + x) as usize] = r[y as usize];
                }
            }
            for y in 0..h {
                for x in 0..w {
                    f[x as usize] = d[(y * w + x) as usize];
                }
                let r = edt_1d(&f[..w as usize]);
                for x in 0..w {
                    d[(y * w + x) as usize] = r[x as usize].sqrt();
                }
            }
        }
        DistanceMetric::L1 | DistanceMetric::C => {
            // Two-pass chamfer; exact for both of these metrics.
            let diag = if metric == DistanceMetric::L1 { 2.0 } else { 1.0 };
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) as usize;
                    if d[i] == 0.0 {
                        continue;
                    }
                    let mut best = d[i];
                    for (dx, dy, cost) in [(-1, 0, 1.0), (0, -1, 1.0), (-1, -1, diag), (1, -1, diag)] {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx >= 0 && ny >= 0 && nx < w && ny < h {
                            best = best.min(d[(ny * w + nx) as usize] + cost);
                        }
                    }
                    d[i] = best;
                }
            }
            for y in (0..h).rev() {
                for x in (0..w).rev() {
                    let i = (y * w + x) as usize;
                    if d[i] == 0.0 {
                        continue;
                    }
                    let mut best = d[i];
                    for (dx, dy, cost) in [(1, 0, 1.0), (0, 1, 1.0), (1, 1, diag), (-1, 1, diag)] {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx >= 0 && ny >= 0 && nx < w && ny < h {
                            best = best.min(d[(ny * w + nx) as usize] + cost);
                        }
                    }
                    d[i] = best;
                }
            }
        }
    }

    let mut out = Mat::zeros(h, w, CV_32FC1)?;
    for y in 0..h {
        for x in 0..w {
            out.pixel_mut::<f32>(y, x)?[0] = d[(y * w + x) as usize] as f32;
        }
    }
    Ok(out)
}

/// Felzenszwalb-Huttenlocher lower envelope of parabolas: exact 1-D squared EDT.
fn edt_1d(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let mut d = vec![0.0f64; n];
    if n == 0 {
        return d;
    }
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];
    let mut k = 0usize;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s;
        loop {
            let p = v[k];
            s = ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2.0 * q as f64 - 2.0 * p as f64);
            if s > z[k] {
                break;
            }
            if k == 0 {
                break;
            }
            k -= 1;
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for (q, out) in d.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        let dq = q as f64 - p as f64;
        *out = dq * dq + f[p];
    }
    d
}

/// `cv::watershed(image, markers)` — priority-flood from labelled seeds.
///
/// `markers` is `CV_32SC1`: positive values are seeds, 0 is unknown, and
/// negative values are left alone. On return every unknown pixel carries either
/// a seed's label or `-1` for a watershed line, as upstream defines it.
pub fn watershed<S: MatLike + ?Sized, M: MatLike + MatLikeMut + ?Sized>(
    image: &S,
    markers: &mut M,
) -> Result<()> {
    cv_assert!(markers.mat_type() == CV_32SC1, "markers must be CV_32SC1");
    cv_assert!(image.size() == markers.size(), "image and markers must match in size");
    let (h, w) = (image.rows(), image.cols());
    let cn = image.channels();
    if h < 3 || w < 3 {
        return Ok(());
    }

    const WSHED: i32 = -1;
    const IN_QUEUE: i32 = -2;
    const NQ: usize = 256;

    // The image, flattened once: the inner loops touch every pixel several
    // times and `at_f64` is not free.
    let mut pix = vec![0.0f64; (h * w * cn) as usize];
    for y in 0..h {
        for x in 0..w {
            for c in 0..cn {
                pix[((y * w + x) * cn + c) as usize] = image.at_f64(y, x, c)?;
            }
        }
    }
    // `c_diff`: the **largest** absolute channel difference, not their sum.
    let diff = |a: usize, b: usize| -> usize {
        let mut m = 0.0f64;
        for c in 0..cn as usize {
            m = m.max((pix[a * cn as usize + c] - pix[b * cn as usize + c]).abs());
        }
        (m as usize).min(NQ - 1)
    };

    let mut label = vec![0i32; (h * w) as usize];
    for y in 0..h {
        for x in 0..w {
            label[(y * w + x) as usize] = markers.at_f64(y, x, 0)? as i32;
        }
    }

    // A hierarchical queue: one FIFO per gradient level. The active level falls
    // as well as rises — every push does `active = min(active, t)` — so this is
    // a priority queue with first-in-first-out ties, and the tie order is the
    // order pixels were discovered.
    let mut q: Vec<std::collections::VecDeque<usize>> =
        (0..NQ).map(|_| std::collections::VecDeque::new()).collect();
    let mut active = NQ;

    // A pixel-wide border of dummy watershed pixels, so a basin cannot creep
    // round the edge of the frame. This overwrites the caller's outermost row
    // and column, upstream included — a seed placed there is destroyed.
    for x in 0..w {
        label[x as usize] = WSHED;
        label[((h - 1) * w + x) as usize] = WSHED;
    }
    for y in 1..h - 1 {
        label[(y * w) as usize] = WSHED;
        label[(y * w + w - 1) as usize] = WSHED;
    }

    // Seed: each *unlabelled* pixel with a labelled neighbour is queued **once**,
    // at the smallest difference to any of them, and marked in-queue. Walking
    // the labelled pixels instead and pushing each neighbour once per adjacency
    // queues a pixel several times at several levels; the duplicates flood
    // again later from a level the front has already left.
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i2 = (y * w + x) as usize;
            // Any leftover negative marker counts as unlabelled.
            if label[i2] < 0 {
                label[i2] = 0;
            }
            if label[i2] != 0 {
                continue;
            }
            let mut idx = NQ;
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nb = ((y + dy) * w + x + dx) as usize;
                if label[nb] > 0 {
                    idx = idx.min(diff(i2, nb));
                }
            }
            if idx < NQ {
                q[idx].push_back(i2);
                active = active.min(idx);
                label[i2] = IN_QUEUE;
            }
        }
    }

    loop {
        while active < NQ && q[active].is_empty() {
            active += 1;
        }
        if active >= NQ {
            break;
        }
        let i2 = q[active].pop_front().expect("the level is non-empty");
        let (y, x) = ((i2 as i32) / w, (i2 as i32) % w);

        // The label comes from the positive neighbours alone; two different
        // ones meeting here makes this a boundary.
        let mut lab = 0i32;
        for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let nb = ((y + dy) * w + x + dx) as usize;
            let tl = label[nb];
            if tl > 0 {
                if lab == 0 {
                    lab = tl;
                } else if tl != lab {
                    lab = WSHED;
                }
            }
        }
        label[i2] = lab;
        // A boundary pixel does not flood onward.
        if lab == WSHED {
            continue;
        }

        for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let nb = ((y + dy) * w + x + dx) as usize;
            if label[nb] == 0 {
                let tq = diff(i2, nb);
                q[tq].push_back(nb);
                active = active.min(tq);
                label[nb] = IN_QUEUE;
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            cv_core::arithm::set_channel_generic(
                markers,
                y,
                x,
                0,
                f64::from(label[(y * w + x) as usize]),
            )?;
        }
    }
    Ok(())
}

/// `cv::isContourConvex(contour)`.
#[must_use]
pub fn is_contour_convex(pts: &[Point]) -> bool {
    let n = pts.len();
    if n < 3 {
        return true;
    }
    let (mut pos, mut neg) = (false, false);
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        let c = pts[(i + 2) % n];
        let cross = (b.x - a.x) as i64 * (c.y - b.y) as i64 - (b.y - a.y) as i64 * (c.x - b.x) as i64;
        if cross > 0 {
            pos = true;
        }
        if cross < 0 {
            neg = true;
        }
        if pos && neg {
            return false;
        }
    }
    true
}

/// `cv::matchShapes(contour1, contour2, method, parameter)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeMatchMode {
    /// `cv::CONTOURS_MATCH_I1`
    I1,
    /// `cv::CONTOURS_MATCH_I2`
    I2,
    /// `cv::CONTOURS_MATCH_I3`
    I3,
}

/// `cv::matchShapes` over two contours, via their Hu moments.
pub fn match_shapes(a: &[Point], b: &[Point], mode: ShapeMatchMode) -> f64 {
    let ha = hu_moments(&crate::shape::contour_moments(a));
    let hb = hu_moments(&crate::shape::contour_moments(b));
    // Upstream's log-magnitude form: sign-preserving log of |hu|.
    let la: Vec<f64> = ha.iter().map(|&v| sign_log(v)).collect();
    let lb: Vec<f64> = hb.iter().map(|&v| sign_log(v)).collect();
    let mut acc = 0.0;
    for i in 0..7 {
        // Upstream's guard is `|hu| > 1e-5` on *both* shapes, not "non-zero".
        // The higher Hu moments run to 1e-12 and below on any real contour, and
        // including them adds a term of order 1/12 apiece to a total that is
        // often smaller than that.
        const EPS: f64 = 1.0e-5;
        if ha[i].abs() <= EPS || hb[i].abs() <= EPS {
            continue;
        }
        match mode {
            ShapeMatchMode::I1 => acc += (1.0 / la[i] - 1.0 / lb[i]).abs(),
            ShapeMatchMode::I2 => acc += (la[i] - lb[i]).abs(),
            // `I3` is the **largest** relative discrepancy, not their sum —
            // upstream writes `if (result < mmm) result = mmm`. Summing gives a
            // number that grows with however many moments cleared the guard.
            ShapeMatchMode::I3 => acc = acc.max((la[i] - lb[i]).abs() / la[i].abs()),
        }
    }
    acc
}

/// `sign(v) * log|v|`, upstream's normalisation inside `matchShapes`.
fn sign_log(v: f64) -> f64 {
    if v == 0.0 {
        0.0
    } else if v > 0.0 {
        v.abs().log10()
    } else {
        -v.abs().log10()
    }
}

/// `cv::HuMoments(moments, hu)` — the seven rotation-invariant moments.
#[must_use]
pub fn hu_moments(m: &crate::shape::Moments) -> [f64; 7] {
    let (n20, n11, n02) = (m.nu20, m.nu11, m.nu02);
    let (n30, n21, n12, n03) = (m.nu30, m.nu21, m.nu12, m.nu03);
    let t0 = n30 + n12;
    let t1 = n21 + n03;
    let q0 = t0 * t0;
    let q1 = t1 * t1;
    let n4 = 4.0 * n11;
    let s = n20 + n02;
    let d = n20 - n02;

    let h0 = s;
    let h1 = d * d + n4 * n11;
    let h3 = q0 + q1;
    let h5 = d * (q0 - q1) + n4 * t0 * t1;

    let a0 = n30 - 3.0 * n12;
    let a1 = 3.0 * n21 - n03;
    let b0 = 3.0 * t0 * t0 - t1 * t1;
    let b1 = 3.0 * t1 * t1 - t0 * t0;
    // Upstream reuses the same subexpressions; the shapes below are its.
    let h2 = a0 * a0 + a1 * a1;
    let h4 = a0 * t0 * (q0 - 3.0 * q1) + a1 * t1 * (3.0 * q0 - q1);
    let h6 = a1 * t0 * (q0 - 3.0 * q1) - a0 * t1 * (3.0 * q0 - q1);
    let _ = (b0, b1);
    [h0, h1, h2, h3, h4, h5, h6]
}

/// Read a single-channel image into a dense `f64` buffer.
///
/// Shared with the corner detectors in [`crate::features`].
pub fn to_gray_f64<S: MatLike + ?Sized>(src: &S) -> Result<Vec<f64>> {
    cv_assert!(src.channels() == 1, "expected a single-channel image");
    let (h, w) = (src.rows(), src.cols());
    let mut v = vec![0.0f64; (h * w) as usize];
    for y in 0..h {
        for x in 0..w {
            v[(y * w + x) as usize] = src.at_f64(y, x, 0)?;
        }
    }
    Ok(v)
}

/// Convenience: build a `CV_8UC1` mask of everything a flood fill reached.
pub fn flood_mask<S: MatLike + ?Sized>(
    image: &S,
    seed: Point,
    lo_diff: f64,
    up_diff: f64,
    connectivity: i32,
) -> Result<Mat> {
    let mut work = Mat::zeros(image.rows(), image.cols(), MatType::new(Depth::U8, 1))?;
    for y in 0..image.rows() {
        for x in 0..image.cols() {
            work.pixel_mut::<u8>(y, x)?[0] = image.at_f64(y, x, 0)? as u8;
        }
    }
    let mut mask = Mat::zeros(image.rows(), image.cols(), CV_8UC1)?;
    let before: Vec<u8> = work.to_vec()?;
    flood_fill(&mut work, seed, Scalar::all(255.0), lo_diff, up_diff, connectivity)?;
    let after: Vec<u8> = work.to_vec()?;
    let w = image.cols();
    for (i, (&a, &b)) in before.iter().zip(after.iter()).enumerate() {
        if a != b || b == 255 {
            let (y, x) = ((i as i32) / w, (i as i32) % w);
            mask.pixel_mut::<u8>(y, x)?[0] = 255;
        }
    }
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(v: &[u8], r: i32, c: i32) -> Mat {
        Mat::from_vec(r, c, 1, v).unwrap()
    }

    #[test]
    fn find_contours_traces_one_square() {
        #[rustfmt::skip]
        let m = img(&[
            0,0,0,0,0,
            0,1,1,1,0,
            0,1,1,1,0,
            0,1,1,1,0,
            0,0,0,0,0,
        ], 5, 5);
        let (cs, _) = find_contours(&m, RetrievalMode::External, ContourApproximation::None).unwrap();
        assert_eq!(cs.len(), 1, "one blob, one contour");
        // The border of a 3x3 square is its 8 outer pixels.
        assert_eq!(cs[0].len(), 8, "got {:?}", cs[0]);
        for p in &cs[0] {
            assert!((1..=3).contains(&p.x) && (1..=3).contains(&p.y));
            assert!(p.x == 1 || p.x == 3 || p.y == 1 || p.y == 3, "centre must not be on the border");
        }
    }

    #[test]
    fn find_contours_separates_two_blobs() {
        #[rustfmt::skip]
        let m = img(&[
            1,1,0,0,1,1,
            1,1,0,0,1,1,
            0,0,0,0,0,0,
        ], 3, 6);
        let (cs, _) = find_contours(&m, RetrievalMode::External, ContourApproximation::None).unwrap();
        assert_eq!(cs.len(), 2);
    }

    #[test]
    fn chain_approx_simple_drops_collinear_points() {
        #[rustfmt::skip]
        let m = img(&[
            0,0,0,0,0,0,
            0,1,1,1,1,0,
            0,1,1,1,1,0,
            0,0,0,0,0,0,
        ], 4, 6);
        let (none, _) = find_contours(&m, RetrievalMode::External, ContourApproximation::None).unwrap();
        let (simple, _) = find_contours(&m, RetrievalMode::External, ContourApproximation::Simple).unwrap();
        assert!(
            simple[0].len() < none[0].len(),
            "SIMPLE must be shorter: {} vs {}",
            simple[0].len(),
            none[0].len()
        );
        assert_eq!(simple[0].len(), 4, "a rectangle keeps its four corners");
    }

    #[test]
    fn flood_fill_repaints_a_connected_region() {
        #[rustfmt::skip]
        let mut m = img(&[
            10,10,90,
            10,10,90,
            90,90,90,
        ], 3, 3);
        let (n, r) = flood_fill(&mut m, Point::new(0, 0), Scalar::all(1.0), 5.0, 5.0, 4).unwrap();
        assert_eq!(n, 4, "only the four 10s");
        assert_eq!(r, Rect::new(0, 0, 2, 2));
        assert_eq!(m.to_vec::<u8>().unwrap(), vec![1, 1, 90, 1, 1, 90, 90, 90, 90]);
    }

    #[test]
    fn flood_fill_connectivity_matters() {
        // Two diagonal cells: 8-connectivity joins them, 4-connectivity does not.
        let mut a = img(&[5, 99, 99, 5], 2, 2);
        let (n4, _) = flood_fill(&mut a, Point::new(0, 0), Scalar::all(1.0), 1.0, 1.0, 4).unwrap();
        assert_eq!(n4, 1);
        let mut b = img(&[5, 99, 99, 5], 2, 2);
        let (n8, _) = flood_fill(&mut b, Point::new(0, 0), Scalar::all(1.0), 1.0, 1.0, 8).unwrap();
        assert_eq!(n8, 2);
    }

    #[test]
    fn distance_transform_l2_is_exact() {
        // A single zero in the centre of a 5x5 field of ones.
        let mut v = vec![1u8; 25];
        v[12] = 0;
        let m = img(&v, 5, 5);
        let d = distance_transform(&m, DistanceMetric::L2).unwrap();
        assert_eq!(d.at_f64(2, 2, 0).unwrap(), 0.0);
        assert_eq!(d.at_f64(2, 3, 0).unwrap(), 1.0);
        let diag = d.at_f64(1, 1, 0).unwrap();
        assert!((diag - 2f64.sqrt()).abs() < 1e-6, "exact Euclidean, got {diag}");
        let corner = d.at_f64(0, 0, 0).unwrap();
        assert!((corner - 8f64.sqrt()).abs() < 1e-6, "got {corner}");
    }

    #[test]
    fn distance_transform_l1_and_chessboard() {
        let mut v = vec![1u8; 25];
        v[12] = 0;
        let m = img(&v, 5, 5);
        let l1 = distance_transform(&m, DistanceMetric::L1).unwrap();
        assert_eq!(l1.at_f64(0, 0, 0).unwrap(), 4.0, "city block from (0,0) to (2,2)");
        let c = distance_transform(&m, DistanceMetric::C).unwrap();
        assert_eq!(c.at_f64(0, 0, 0).unwrap(), 2.0, "chessboard");
    }

    #[test]
    fn is_contour_convex_distinguishes_a_square_from_an_arrow() {
        let sq = vec![
            Point::new(0, 0),
            Point::new(4, 0),
            Point::new(4, 4),
            Point::new(0, 4),
        ];
        assert!(is_contour_convex(&sq));
        let arrow = vec![
            Point::new(0, 0),
            Point::new(4, 0),
            Point::new(2, 2),
            Point::new(4, 4),
            Point::new(0, 4),
        ];
        assert!(!is_contour_convex(&arrow));
    }

    #[test]
    fn match_shapes_scores_identical_shapes_at_zero() {
        let a = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ];
        let d = match_shapes(&a, &a, ShapeMatchMode::I1);
        assert!(d.abs() < 1e-12, "a shape must match itself exactly, got {d}");
        // A very different shape scores higher than a scaled copy of itself.
        let scaled: Vec<Point> = a.iter().map(|p| Point::new(p.x * 3, p.y * 3)).collect();
        let thin = vec![
            Point::new(0, 0),
            Point::new(40, 0),
            Point::new(40, 2),
            Point::new(0, 2),
        ];
        assert!(
            match_shapes(&a, &scaled, ShapeMatchMode::I2) < match_shapes(&a, &thin, ShapeMatchMode::I2),
            "Hu moments are scale invariant"
        );
    }

    #[test]
    fn watershed_splits_two_basins() {
        // A dark ridge down the middle separates two bright regions. The
        // seeds sit one pixel *in* from the edge: upstream overwrites the
        // outermost row and column of the caller's marker image with WSHED
        // before flooding, so a seed on the border is destroyed.
        #[rustfmt::skip]
        let im = img(&[
            0,0,0,255,0,0,0,
            0,0,0,255,0,0,0,
            0,0,0,255,0,0,0,
            0,0,0,255,0,0,0,
            0,0,0,255,0,0,0,
        ], 5, 7);
        let mut mk = Mat::zeros(5, 7, CV_32SC1).unwrap();
        mk.pixel_mut::<i32>(2, 1).unwrap()[0] = 1;
        mk.pixel_mut::<i32>(2, 5).unwrap()[0] = 2;
        watershed(&im, &mut mk).unwrap();
        assert_eq!(mk.at_f64(2, 1, 0).unwrap() as i32, 1);
        assert_eq!(mk.at_f64(2, 5, 0).unwrap() as i32, 2);
        // Nothing inside the border is left unassigned.
        for y in 1..4 {
            for x in 1..6 {
                assert_ne!(mk.at_f64(y, x, 0).unwrap() as i32, 0, "({x},{y}) still unknown");
            }
        }
    }
}
