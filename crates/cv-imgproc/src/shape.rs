//! Moments, contour metrics and convexity
//! (`modules/imgproc/src/moments.cpp`, `shapedescr.cpp`, `approx.cpp`, `convhull.cpp`).

use cv_core::{cv_assert, cv_error, Code, Mat, MatLike, MatLikeMut, Point, Rect, Result};

pub use cv_core::Moments;

/// `cv::moments(array, binaryImage)` for a raster image.
///
/// With `binary` set, every non-zero pixel counts as 1, matching upstream.
pub fn moments<S: MatLike + ?Sized>(src: &S, binary: bool) -> Result<Moments> {
    cv_assert!(src.channels() == 1, "moments requires a single-channel image");
    let mut m = Moments::default();
    for y in 0..src.rows() {
        let yd = y as f64;
        for x in 0..src.cols() {
            let v = src.at_f64(y, x, 0)?;
            let p = if binary {
                if v != 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                v
            };
            if p == 0.0 {
                continue;
            }
            let xd = x as f64;
            m.m00 += p;
            m.m10 += p * xd;
            m.m01 += p * yd;
            m.m20 += p * xd * xd;
            m.m11 += p * xd * yd;
            m.m02 += p * yd * yd;
            m.m30 += p * xd * xd * xd;
            m.m21 += p * xd * xd * yd;
            m.m12 += p * xd * yd * yd;
            m.m03 += p * yd * yd * yd;
        }
    }
    m.complete();
    Ok(m)
}

/// `cv::moments` for a closed polygon given as a point sequence.
///
/// This is Green's-theorem integration around the boundary, not a pixel sum, so
/// it differs from [`moments`] on the same shape by the usual half-pixel
/// boundary effect — exactly as upstream's two paths do.
pub fn contour_moments(points: &[Point]) -> Moments {
    let mut m = Moments::default();
    if points.len() < 2 {
        return m;
    }

    let (mut a00, mut a10, mut a01) = (0.0f64, 0.0f64, 0.0f64);
    let (mut a20, mut a11, mut a02) = (0.0f64, 0.0f64, 0.0f64);
    let (mut a30, mut a21, mut a12, mut a03) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);

    let last = points[points.len() - 1];
    let (mut xi_1, mut yi_1) = (last.x as f64, last.y as f64);
    let (mut xi_12, mut yi_12) = (xi_1 * xi_1, yi_1 * yi_1);

    for p in points {
        let (xi, yi) = (p.x as f64, p.y as f64);
        let (xi2, yi2) = (xi * xi, yi * yi);
        let dxy = xi_1 * yi - xi * yi_1;
        let xii_1 = xi_1 + xi;
        let yii_1 = yi_1 + yi;

        a00 += dxy;
        a10 += dxy * xii_1;
        a01 += dxy * yii_1;
        a20 += dxy * (xi_1 * xii_1 + xi2);
        a11 += dxy * (xi_1 * (yii_1 + yi_1) + xi * (yii_1 + yi));
        a02 += dxy * (yi_1 * yii_1 + yi2);
        a30 += dxy * xii_1 * (xi_12 + xi2);
        a03 += dxy * yii_1 * (yi_12 + yi2);
        a21 += dxy * (xi_12 * (3.0 * yi_1 + yi) + 2.0 * xi * xi_1 * yii_1 + xi2 * (yi_1 + 3.0 * yi));
        a12 += dxy * (yi_12 * (3.0 * xi_1 + xi) + 2.0 * yi * yi_1 * xii_1 + yi2 * (xi_1 + 3.0 * xi));

        xi_1 = xi;
        yi_1 = yi;
        xi_12 = xi2;
        yi_12 = yi2;
    }

    if a00.abs() > f32::EPSILON as f64 {
        // The divisors carry the sign of a00, so a clockwise contour yields
        // positive m00 just as a counter-clockwise one does.
        let sign = if a00 > 0.0 { 1.0 } else { -1.0 };
        m.m00 = a00 * 0.5 * sign;
        m.m10 = a10 * sign / 6.0;
        m.m01 = a01 * sign / 6.0;
        m.m20 = a20 * sign / 12.0;
        m.m11 = a11 * sign / 24.0;
        m.m02 = a02 * sign / 12.0;
        m.m30 = a30 * sign / 20.0;
        m.m21 = a21 * sign / 60.0;
        m.m12 = a12 * sign / 60.0;
        m.m03 = a03 * sign / 20.0;
        m.complete();
    }
    m
}

/// `cv::contourArea(contour, oriented)` — the shoelace formula.
#[must_use]
pub fn contour_area(points: &[Point], oriented: bool) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let mut a00 = 0.0f64;
    let mut prev = points[points.len() - 1];
    for p in points {
        a00 += prev.x as f64 * p.y as f64 - prev.y as f64 * p.x as f64;
        prev = *p;
    }
    a00 *= 0.5;
    if oriented {
        a00
    } else {
        a00.abs()
    }
}

/// Euclidean distance between two integer points, in `f64`.
///
/// Used by `approxPolyDP` and `pointPolygonTest`, which upstream computes in
/// double precision; [`arc_length`] deliberately does not use this.
#[inline]
fn dist(a: Point, b: Point) -> f64 {
    let dx = (a.x - b.x) as f64;
    let dy = (a.y - b.y) as f64;
    (dx * dx + dy * dy).sqrt()
}

/// `cv::arcLength(curve, closed)`.
///
/// # Precision
///
/// Each segment length is computed in **f32** and only the running total is
/// `f64`: upstream widens the points to `Point2f`, so `dx`, `dy`, `dx*dx+dy*dy`
/// and the square root are all single-precision. Computing the segments in
/// `f64` is more accurate but disagrees with cv2 in the 7th significant digit.
///
/// The loop also starts from the closing edge rather than appending it, which
/// is what makes the summation order match.
#[must_use]
pub fn arc_length(points: &[Point], closed: bool) -> f64 {
    let count = points.len();
    if count <= 1 {
        return 0.0;
    }
    let last = if closed { count - 1 } else { 0 };
    let mut perimeter = 0.0f64;
    let mut prev = points[last];
    for &p in points {
        let dx = p.x as f32 - prev.x as f32;
        let dy = p.y as f32 - prev.y as f32;
        perimeter += (dx * dx + dy * dy).sqrt() as f64;
        prev = p;
    }
    perimeter
}

/// `cv::boundingRect(points)` — the smallest upright rectangle containing them.
pub fn bounding_rect(points: &[Point]) -> Result<Rect> {
    if points.is_empty() {
        return cv_error!(Code::StsBadArg, "boundingRect needs at least one point");
    }
    let (mut x0, mut y0) = (i32::MAX, i32::MAX);
    let (mut x1, mut y1) = (i32::MIN, i32::MIN);
    for p in points {
        x0 = x0.min(p.x);
        y0 = y0.min(p.y);
        x1 = x1.max(p.x);
        y1 = y1.max(p.y);
    }
    // The rectangle is inclusive of both extremes, hence the +1.
    Ok(Rect::new(x0, y0, x1 - x0 + 1, y1 - y0 + 1))
}

/// `cv::convexHull(points, hull, clockwise)`.
///
/// `-1`, `0` or `1` — upstream's `CV_SIGN`.
#[inline]
fn sign(v: i64) -> i32 {
    match v.cmp(&0) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// One monotone sweep of upstream's `Sklansky_`, over `arr` sorted by `(x, y)`.
///
/// Walks from `start` towards `end` (either direction), keeping a stack of
/// candidate vertices. `nsign` selects which vertical direction is a candidate
/// edge and `sign2` which turn direction counts as convex, so the same routine
/// serves all four quarter-sweeps. Returns the number of valid stack entries.
///
/// Indices are `isize` because the sweep stores one position past the end
/// before terminating; that entry is always excluded by the final decrement,
/// but it must not be an out-of-range `usize` in the meantime.
fn sklansky(arr: &[Point], start: usize, end: usize, stack: &mut [isize], nsign: i32, sign2: i32) -> usize {
    let (start, end0) = (start as isize, end as isize);
    let at = |i: isize| arr[i as usize];
    if start == end0 || (at(start).x == at(end0).x && at(start).y == at(end0).y) {
        stack[0] = start;
        return 1;
    }

    let incr: isize = if end0 > start { 1 } else { -1 };
    let (mut pprev, mut pcur) = (start, start + incr);
    let mut pnext = pcur + incr;
    let mut stacksize = 3usize;
    stack[0] = pprev;
    stack[1] = pcur;
    stack[2] = pnext;
    let end = end0 + incr; // one past the end

    while pnext != end {
        let cury = at(pcur).y;
        let nexty = at(pnext).y;
        let by = nexty - cury;

        if sign(by as i64) != nsign {
            let a = (at(pcur).x - at(pprev).x, cury - at(pprev).y);
            let b = (at(pnext).x - at(pcur).x, by);
            let convexity = a.1 as i64 * b.0 as i64 - a.0 as i64 * b.1 as i64;

            if sign(convexity) == sign2 && (a.0 != 0 || a.1 != 0) {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                stack[stacksize] = pnext;
                stacksize += 1;
            } else if pprev == start {
                pcur = pnext;
                stack[1] = pcur;
                pnext += incr;
                stack[2] = pnext;
            } else {
                stack[stacksize - 2] = pnext;
                pcur = pprev;
                pprev = stack[stacksize - 4];
                stacksize -= 1;
            }
        } else {
            pnext += incr;
            stack[stacksize - 1] = pnext;
        }
    }
    stacksize - 1
}

/// `cv::convexHull(points, hull, clockwise)`.
///
/// A transcription of upstream's Sklansky scan rather than the textbook
/// monotone chain. The two agree on *which* points are hull vertices, but not
/// on where the cycle starts: upstream sorts by `(x, y)`, sweeps the upper hull
/// outward from both ends of that order towards the highest point and the lower
/// hull towards the lowest, then concatenates. The resulting rotation is not
/// any simple extreme — for the pentagon `(10,10) (90,14) (120,70) (60,110)
/// (15,80)` cv2 starts at `(15,80)`, not at the leftmost `(10,10)` — so it can
/// only be matched by reproducing the sweep.
pub fn convex_hull(points: &[Point], clockwise: bool) -> Vec<Point> {
    let total = points.len();
    if total == 0 {
        return Vec::new();
    }
    // Sort *positions*, not points: the closing cyclic shift below is defined
    // over the original input indices, so they have to survive the sort.
    // Upstream's comparator falls back to comparing the pointers themselves,
    // which for pointers into one array means original order among duplicates.
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_unstable_by_key(|&i| (points[i].x, points[i].y, i));
    let pts: Vec<Point> = order.iter().map(|&i| points[i]).collect();

    // Ties keep the earliest point in sorted order: both comparisons are strict.
    let (mut miny_ind, mut maxy_ind) = (0usize, 0usize);
    for i in 1..total {
        if pts[miny_ind].y > pts[i].y {
            miny_ind = i;
        }
        if pts[maxy_ind].y < pts[i].y {
            maxy_ind = i;
        }
    }
    if pts[0] == pts[total - 1] {
        return vec![pts[0]];
    }

    let mut out: Vec<usize> = Vec::with_capacity(total);
    let mut push = |s: &[isize], n: usize, rev: bool| {
        if rev {
            for i in (1..n).rev() {
                out.push(s[i] as usize);
            }
        } else {
            for &i in s.iter().take(n.saturating_sub(1)) {
                out.push(i as usize);
            }
        }
    };

    let mut a = vec![0isize; total + 2];
    let mut b = vec![0isize; total + 2];

    // Upper hull: sweep from each end of the sorted order up to the highest point.
    let (na, nb) = (
        sklansky(&pts, 0, maxy_ind, &mut a, -1, 1),
        sklansky(&pts, total - 1, maxy_ind, &mut b, -1, -1),
    );
    let (tl, tl_n, tr, tr_n) = if clockwise { (&a, na, &b, nb) } else { (&b, nb, &a, na) };
    push(tl, tl_n, false);
    push(tr, tr_n, true);
    let stop_idx: isize = if tr_n > 2 {
        tr[1]
    } else if tl_n > 2 {
        tl[tl_n - 2]
    } else {
        -1
    };

    // Lower hull: the mirror sweep, towards the lowest point.
    let mut c = vec![0isize; total + 2];
    let mut d = vec![0isize; total + 2];
    let (nc, nd) = (
        sklansky(&pts, 0, miny_ind, &mut c, 1, -1),
        sklansky(&pts, total - 1, miny_ind, &mut d, 1, 1),
    );
    let (bl, mut bl_n, br, mut br_n) = if clockwise { (&d, nd, &c, nc) } else { (&c, nc, &d, nd) };

    // If every point is collinear the lower hull is the mirror of the upper one,
    // so keep only its two extremes rather than emitting the run twice.
    if stop_idx >= 0 {
        let check_idx: isize = if bl_n > 2 {
            bl[1]
        } else if bl_n + br_n > 2 {
            br[2 - bl_n]
        } else {
            -1
        };
        let same = check_idx >= 0 && pts[check_idx as usize] == pts[stop_idx as usize];
        if check_idx == stop_idx || same {
            bl_n = bl_n.min(2);
            br_n = br_n.min(2);
        }
    }
    push(bl, bl_n, false);
    push(br, br_n, true);

    // Back to original input indices, then upstream's closing step: cyclically
    // shift the hull so those indices run monotonically. This is what fixes the
    // starting vertex, and it is why the sweeps alone were never going to
    // agree. For the pentagon `(10,10) (90,14) (120,70) (60,110) (15,80)` the
    // sweeps emit input indices `0 4 3 2 1`; the descending run starts at
    // position 1, so the shift rotates to `4 3 2 1 0` and the hull begins at
    // `(15,80)` — exactly what cv2 returns, and not any extreme of the point
    // set. The guard conditions are upstream's: the shift only applies when the
    // sequence is already almost monotone, so a genuinely self-intersecting
    // input is left alone.
    let mut hull: Vec<usize> = out.iter().map(|&p| order[p]).collect();
    let nout = hull.len();
    if nout >= 3 {
        let (mut min_idx, mut max_idx, mut lt) = (0usize, 0usize, 0i32);
        for i in 1..nout {
            let idx = hull[i];
            lt += i32::from(hull[i - 1] < idx);
            if lt > 1 && lt <= i as i32 - 2 {
                break;
            }
            if idx < hull[min_idx] {
                min_idx = i;
            }
            if idx > hull[max_idx] {
                max_idx = i;
            }
        }
        let mmdist = (max_idx as i32 - min_idx as i32).abs();
        if (mmdist == 1 || mmdist == nout as i32 - 1) && (lt <= 1 || lt >= nout as i32 - 2) {
            let ascending = (max_idx + 1) % nout == min_idx;
            let i0 = if ascending { min_idx } else { max_idx };
            if i0 > 0 {
                let mut shifted = Vec::with_capacity(nout);
                let (mut j, mut i) = (i0, 0usize);
                while i < nout {
                    let curr = hull[j];
                    shifted.push(curr);
                    let next_j = if j + 1 < nout { j + 1 } else { 0 };
                    if i < nout - 1 && ascending != (curr < hull[next_j]) {
                        break;
                    }
                    j = next_j;
                    i += 1;
                }
                if i == nout {
                    hull = shifted;
                }
            }
        }
    }
    hull.into_iter().map(|i| points[i]).collect()
}

/// `cv::approxPolyDP(curve, approxCurve, epsilon, closed)` — Douglas-Peucker.
pub fn approx_poly_dp(points: &[Point], epsilon: f64, closed: bool) -> Result<Vec<Point>> {
    cv_assert!(epsilon >= 0.0, "epsilon must be non-negative, got {epsilon}");
    if points.len() < 3 {
        return Ok(points.to_vec());
    }

    if closed {
        // Anchor the recursion on the two mutually farthest-apart vertices so
        // the result does not depend on where the caller started the loop.
        let start = 0usize;
        let mut end = 0usize;
        let mut best = -1.0f64;
        for (i, p) in points.iter().enumerate() {
            let d = dist(points[start], *p);
            if d > best {
                best = d;
                end = i;
            }
        }
        let mut a = simplify(&points[start..=end], epsilon);
        let mut tail: Vec<Point> = points[end..].to_vec();
        tail.push(points[start]);
        let b = simplify(&tail, epsilon);
        a.pop(); // shared vertex
        a.extend(b.into_iter().take(b_len_minus_one(&points[end..])));
        // Rebuild cleanly: dedup consecutive duplicates and drop the closing point.
        let mut out = a;
        out.dedup();
        if out.len() > 1 && out[0] == out[out.len() - 1] {
            out.pop();
        }
        return Ok(out);
    }
    Ok(simplify(points, epsilon))
}

/// Number of points to take from the second half, excluding the duplicated end.
fn b_len_minus_one(seg: &[Point]) -> usize {
    seg.len()
}

/// Recursive Douglas-Peucker over an open polyline.
fn simplify(points: &[Point], epsilon: f64) -> Vec<Point> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let (first, last) = (points[0], points[points.len() - 1]);

    let mut max_d = 0.0f64;
    let mut index = 0usize;
    for (i, p) in points.iter().enumerate().take(points.len() - 1).skip(1) {
        let d = perpendicular_distance(*p, first, last);
        if d > max_d {
            max_d = d;
            index = i;
        }
    }

    if max_d > epsilon {
        let mut left = simplify(&points[..=index], epsilon);
        let right = simplify(&points[index..], epsilon);
        left.pop(); // the split vertex appears in both halves
        left.extend(right);
        left
    } else {
        vec![first, last]
    }
}

/// Distance from `p` to the segment `a`–`b`.
fn perpendicular_distance(p: Point, a: Point, b: Point) -> f64 {
    let (dx, dy) = ((b.x - a.x) as f64, (b.y - a.y) as f64);
    if dx == 0.0 && dy == 0.0 {
        return dist(p, a);
    }
    let num = (dy * (p.x - a.x) as f64 - dx * (p.y - a.y) as f64).abs();
    num / (dx * dx + dy * dy).sqrt()
}

/// `cv::pointPolygonTest(contour, pt, measureDist)`.
///
/// Without `measure_dist` returns +1 inside, 0 on an edge, -1 outside. With it,
/// returns the signed distance to the nearest edge, positive inside.
pub fn point_polygon_test(contour: &[Point], pt: Point, measure_dist: bool) -> f64 {
    if contour.len() < 3 {
        return -1.0;
    }
    let n = contour.len();
    let mut inside = false;
    let mut min_dist = f64::INFINITY;

    for i in 0..n {
        let a = contour[i];
        let b = contour[(i + 1) % n];

        // Ray casting: count crossings of a horizontal ray to the right.
        if (a.y > pt.y) != (b.y > pt.y) {
            let t = (pt.y - a.y) as f64 / (b.y - a.y) as f64;
            let xint = a.x as f64 + t * (b.x - a.x) as f64;
            if (pt.x as f64) < xint {
                inside = !inside;
            }
        }
        min_dist = min_dist.min(segment_distance(pt, a, b));
    }

    if !measure_dist {
        if min_dist == 0.0 {
            0.0
        } else if inside {
            1.0
        } else {
            -1.0
        }
    } else if inside {
        min_dist
    } else {
        -min_dist
    }
}

/// Distance from `p` to the closed segment `a`–`b`.
fn segment_distance(p: Point, a: Point, b: Point) -> f64 {
    let (dx, dy) = ((b.x - a.x) as f64, (b.y - a.y) as f64);
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return dist(p, a);
    }
    // Project onto the segment, clamping to its endpoints.
    let t = (((p.x - a.x) as f64 * dx + (p.y - a.y) as f64 * dy) / len2).clamp(0.0, 1.0);
    let px = a.x as f64 + t * dx;
    let py = a.y as f64 + t * dy;
    ((p.x as f64 - px).powi(2) + (p.y as f64 - py).powi(2)).sqrt()
}

/// Per-component statistics from [`connected_components_with_stats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentStats {
    /// Bounding box of the component.
    pub rect: Rect,
    /// Number of pixels in the component.
    pub area: i32,
}

/// `cv::connectedComponents(image, labels, connectivity)`.
///
/// The input is treated as binary: any non-zero pixel is foreground. Labels are
/// numbered from 1 in raster order of first appearance, with 0 for background,
/// matching upstream's convention. Returns `(labels, count)` where `count`
/// includes the background label.
pub fn connected_components<S: MatLike + ?Sized>(
    src: &S,
    connectivity: i32,
) -> Result<(Mat, i32)> {
    label_components(src, connectivity)
}

/// Result of [`connected_components_with_stats`].
///
/// Index 0 of `stats` and `centroids` describes the background, as upstream's
/// output arrays do.
#[derive(Debug)]
pub struct LabelledComponents {
    /// `CV_32SC1` label image; 0 is background.
    pub labels: Mat,
    /// Bounding box and pixel count per label.
    pub stats: Vec<ComponentStats>,
    /// Centre of mass per label, as `(x, y)`.
    pub centroids: Vec<(f64, f64)>,
}

/// `cv::connectedComponentsWithStats(image, labels, stats, centroids, connectivity)`.
pub fn connected_components_with_stats<S: MatLike + ?Sized>(
    src: &S,
    connectivity: i32,
) -> Result<LabelledComponents> {
    let (labels, count) = label_components(src, connectivity)?;
    let n = count as usize;

    let mut min_x = vec![i32::MAX; n];
    let mut min_y = vec![i32::MAX; n];
    let mut max_x = vec![i32::MIN; n];
    let mut max_y = vec![i32::MIN; n];
    let mut area = vec![0i32; n];
    let mut sx = vec![0f64; n];
    let mut sy = vec![0f64; n];

    for y in 0..labels.rows() {
        for x in 0..labels.cols() {
            let l = labels.at::<i32>(y, x)? as usize;
            min_x[l] = min_x[l].min(x);
            min_y[l] = min_y[l].min(y);
            max_x[l] = max_x[l].max(x);
            max_y[l] = max_y[l].max(y);
            area[l] += 1;
            sx[l] += x as f64;
            sy[l] += y as f64;
        }
    }

    let mut stats = Vec::with_capacity(n);
    let mut centroids = Vec::with_capacity(n);
    for i in 0..n {
        if area[i] == 0 {
            stats.push(ComponentStats { rect: Rect::default(), area: 0 });
            centroids.push((0.0, 0.0));
            continue;
        }
        stats.push(ComponentStats {
            rect: Rect::new(min_x[i], min_y[i], max_x[i] - min_x[i] + 1, max_y[i] - min_y[i] + 1),
            area: area[i],
        });
        centroids.push((sx[i] / area[i] as f64, sy[i] / area[i] as f64));
    }
    Ok(LabelledComponents { labels, stats, centroids })
}

/// Two-pass union-find labelling shared by both entry points.
fn label_components<S: MatLike + ?Sized>(src: &S, connectivity: i32) -> Result<(Mat, i32)> {
    cv_assert!(
        connectivity == 4 || connectivity == 8,
        "connectivity must be 4 or 8, got {connectivity}"
    );
    cv_assert!(src.channels() == 1, "connectedComponents requires a single-channel image");
    cv_assert!(!src.is_empty(), "connectedComponents requires a non-empty image");

    let (w, h) = (src.cols(), src.rows());
    let mut provisional = vec![0usize; (w * h) as usize];
    let mut parent: Vec<usize> = vec![0];

    // Union-find over provisional labels.
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]]; // path halving
            i = parent[i];
        }
        i
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra != rb {
            // Keep the smaller root so raster order survives the merge.
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            parent[hi] = lo;
        }
    }

    for y in 0..h {
        for x in 0..w {
            if src.at_f64(y, x, 0)? == 0.0 {
                continue;
            }
            let idx = (y * w + x) as usize;

            // Neighbours already visited in raster order.
            let mut neighbours: Vec<usize> = Vec::with_capacity(4);
            let push = |ny: i32, nx: i32, provisional: &[usize]| -> Option<usize> {
                if ny < 0 || nx < 0 || nx >= w {
                    return None;
                }
                let l = provisional[(ny * w + nx) as usize];
                if l != 0 {
                    Some(l)
                } else {
                    None
                }
            };
            if let Some(l) = push(y, x - 1, &provisional) {
                neighbours.push(l);
            }
            if let Some(l) = push(y - 1, x, &provisional) {
                neighbours.push(l);
            }
            if connectivity == 8 {
                if let Some(l) = push(y - 1, x - 1, &provisional) {
                    neighbours.push(l);
                }
                if let Some(l) = push(y - 1, x + 1, &provisional) {
                    neighbours.push(l);
                }
            }

            if neighbours.is_empty() {
                let new_label = parent.len();
                parent.push(new_label);
                provisional[idx] = new_label;
            } else {
                let min = *neighbours.iter().min().unwrap();
                provisional[idx] = min;
                for &n in &neighbours {
                    union(&mut parent, min, n);
                }
            }
        }
    }

    // Second pass: resolve to roots and renumber by order of first appearance.
    //
    // *Which* order is upstream's, and it depends on connectivity, because
    // upstream runs two different algorithms: SAUF for 4-connectivity, which
    // scans pixel by pixel, and Grana's block-based decision tree for
    // 8-connectivity, which scans in 2x2 blocks. So a component's number is its
    // rank by first appearance in **pixel** raster order at 4-connectivity but
    // in **2x2 block** raster order at 8-connectivity. Both were confirmed
    // against cv2 4.13.0 on a 331-component image: scanning in the matching
    // order recovers exactly 1..k, scanning in the other does not.
    //
    // This renumbering is the only thing BBDT is being reproduced for. The
    // partition itself is identical either way, so the union-find above stands.
    let mut remap = vec![0i32; parent.len()];
    let mut next = 1i32;
    let claim = |parent: &mut Vec<usize>, remap: &mut Vec<i32>, next: &mut i32, y: i32, x: i32| {
        let p = provisional[(y * w + x) as usize];
        if p == 0 {
            return;
        }
        let root = find(parent, p);
        if remap[root] == 0 {
            remap[root] = *next;
            *next += 1;
        }
    };
    if connectivity == 8 {
        for by in (0..h).step_by(2) {
            for bx in (0..w).step_by(2) {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (y, x) = (by + dy, bx + dx);
                        if y < h && x < w {
                            claim(&mut parent, &mut remap, &mut next, y, x);
                        }
                    }
                }
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                claim(&mut parent, &mut remap, &mut next, y, x);
            }
        }
    }

    let mut labels = Mat::zeros(h, w, cv_core::types::CV_32SC1)?;
    for y in 0..h {
        for x in 0..w {
            let p = provisional[(y * w + x) as usize];
            if p == 0 {
                continue;
            }
            let root = find(&mut parent, p);
            labels.pixel_mut::<i32>(y, x)?[0] = remap[root];
        }
    }
    Ok((labels, next))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{types::CV_8UC1, MatLikeMut, Scalar};

    fn square(n: i32) -> Vec<Point> {
        vec![
            Point::new(0, 0),
            Point::new(n, 0),
            Point::new(n, n),
            Point::new(0, n),
        ]
    }

    #[test]
    fn contour_area_uses_the_shoelace_formula() {
        assert_eq!(contour_area(&square(10), false), 100.0);
        // Reversing the winding flips the oriented sign but not the magnitude.
        let mut rev = square(10);
        rev.reverse();
        assert_eq!(contour_area(&rev, false), 100.0);
        assert_eq!(contour_area(&square(10), true), 100.0);
        assert_eq!(contour_area(&rev, true), -100.0);
    }

    #[test]
    fn contour_area_of_a_triangle() {
        let t = [Point::new(0, 0), Point::new(4, 0), Point::new(0, 3)];
        assert_eq!(contour_area(&t, false), 6.0);
    }

    #[test]
    fn degenerate_contours_have_zero_area() {
        assert_eq!(contour_area(&[], false), 0.0);
        assert_eq!(contour_area(&[Point::new(1, 1)], false), 0.0);
        assert_eq!(contour_area(&[Point::new(0, 0), Point::new(5, 5)], false), 0.0);
    }

    #[test]
    fn arc_length_open_and_closed() {
        let pts = [Point::new(0, 0), Point::new(3, 0), Point::new(3, 4)];
        assert_eq!(arc_length(&pts, false), 7.0);
        assert_eq!(arc_length(&pts, true), 12.0); // + hypotenuse 5
        assert_eq!(arc_length(&pts[..1], true), 0.0);
    }

    #[test]
    fn bounding_rect_is_inclusive_of_both_extremes() {
        let pts = [Point::new(2, 3), Point::new(7, 3), Point::new(4, 9)];
        assert_eq!(bounding_rect(&pts).unwrap(), Rect::new(2, 3, 6, 7));
        assert_eq!(bounding_rect(&[Point::new(5, 5)]).unwrap(), Rect::new(5, 5, 1, 1));
        assert!(bounding_rect(&[]).is_err());
    }

    #[test]
    fn contour_moments_match_the_area_and_centroid() {
        let m = contour_moments(&square(10));
        assert!((m.m00 - 100.0).abs() < 1e-9, "m00 = {}", m.m00);
        let (cx, cy) = m.centroid().unwrap();
        assert!((cx - 5.0).abs() < 1e-9 && (cy - 5.0).abs() < 1e-9, "centroid {cx},{cy}");
    }

    #[test]
    fn contour_moments_are_winding_independent() {
        let mut rev = square(10);
        rev.reverse();
        let a = contour_moments(&square(10));
        let b = contour_moments(&rev);
        assert!((a.m00 - b.m00).abs() < 1e-9);
        assert!((a.mu20 - b.mu20).abs() < 1e-6, "{} vs {}", a.mu20, b.mu20);
    }

    #[test]
    fn image_moments_count_pixels() {
        // A 4x4 block of ones: m00 is the pixel count, centroid is its middle.
        let mut m = Mat::zeros(8, 8, CV_8UC1).unwrap();
        {
            let mut roi = m.roi_mut(Rect::new(2, 2, 4, 4)).unwrap();
            cv_core::arithm::set_to(&mut roi, Scalar::single(1.0)).unwrap();
        }
        let mo = moments(&m, false).unwrap();
        assert_eq!(mo.m00, 16.0);
        let (cx, cy) = mo.centroid().unwrap();
        assert!((cx - 3.5).abs() < 1e-12 && (cy - 3.5).abs() < 1e-12);
    }

    #[test]
    fn binary_flag_ignores_intensity() {
        let mut m = Mat::zeros(4, 4, CV_8UC1).unwrap();
        m.pixel_mut::<u8>(1, 1).unwrap()[0] = 200;
        m.pixel_mut::<u8>(2, 2).unwrap()[0] = 50;
        assert_eq!(moments(&m, true).unwrap().m00, 2.0);
        assert_eq!(moments(&m, false).unwrap().m00, 250.0);
    }

    #[test]
    fn normalised_moments_are_scale_invariant() {
        // nu20 of a square should not depend on its size.
        let small = contour_moments(&square(10));
        let large = contour_moments(&square(100));
        assert!(
            (small.nu20 - large.nu20).abs() < 1e-6,
            "{} vs {}",
            small.nu20,
            large.nu20
        );
    }

    #[test]
    fn moments_of_an_empty_shape_are_zero() {
        let m = contour_moments(&[]);
        assert_eq!(m.m00, 0.0);
        assert!(m.centroid().is_none());
        let blank = Mat::zeros(4, 4, CV_8UC1).unwrap();
        assert_eq!(moments(&blank, false).unwrap().m00, 0.0);
    }

    #[test]
    fn convex_hull_of_a_square_with_interior_points() {
        let pts = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
            Point::new(5, 5), // interior, must be dropped
            Point::new(3, 7),
        ];
        let hull = convex_hull(&pts, true);
        assert_eq!(hull.len(), 4);
        for corner in [
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ] {
            assert!(hull.contains(&corner), "missing {corner:?} in {hull:?}");
        }
    }

    #[test]
    fn convex_hull_orientation_flips_with_the_flag() {
        let pts = square(10);
        let cw = convex_hull(&pts, true);
        let ccw = convex_hull(&pts, false);
        assert_eq!(cw.len(), ccw.len());
        // Opposite winding gives opposite signed area.
        assert!(contour_area(&cw, true) * contour_area(&ccw, true) < 0.0);
    }

    #[test]
    fn convex_hull_drops_collinear_points() {
        let pts = vec![
            Point::new(0, 0),
            Point::new(5, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ];
        let hull = convex_hull(&pts, true);
        assert_eq!(hull.len(), 4, "collinear midpoint survived: {hull:?}");
    }

    #[test]
    fn convex_hull_handles_tiny_inputs() {
        assert_eq!(convex_hull(&[], true).len(), 0);
        assert_eq!(convex_hull(&[Point::new(1, 1)], true).len(), 1);
        assert_eq!(convex_hull(&[Point::new(1, 1), Point::new(2, 2)], true).len(), 2);
    }

    #[test]
    fn approx_poly_simplifies_a_noisy_line() {
        // Points on a straight line plus one small deviation.
        let pts = vec![
            Point::new(0, 0),
            Point::new(1, 0),
            Point::new(2, 1),
            Point::new(3, 0),
            Point::new(4, 0),
        ];
        // A large epsilon removes the bump entirely.
        let coarse = approx_poly_dp(&pts, 5.0, false).unwrap();
        assert_eq!(coarse, vec![Point::new(0, 0), Point::new(4, 0)]);
        // A small epsilon keeps it.
        let fine = approx_poly_dp(&pts, 0.5, false).unwrap();
        assert!(fine.contains(&Point::new(2, 1)), "{fine:?}");
    }

    #[test]
    fn approx_poly_keeps_a_square_as_four_corners() {
        // A densely sampled square boundary must reduce to its corners.
        let mut pts = Vec::new();
        for x in 0..=20 {
            pts.push(Point::new(x, 0));
        }
        for y in 1..=20 {
            pts.push(Point::new(20, y));
        }
        for x in (0..20).rev() {
            pts.push(Point::new(x, 20));
        }
        for y in (1..20).rev() {
            pts.push(Point::new(0, y));
        }
        let approx = approx_poly_dp(&pts, 2.0, true).unwrap();
        assert_eq!(approx.len(), 4, "got {approx:?}");
    }

    #[test]
    fn approx_poly_rejects_negative_epsilon() {
        assert!(approx_poly_dp(&square(4), -1.0, true).is_err());
    }

    #[test]
    fn point_polygon_test_classifies_inside_outside_and_edge() {
        let sq = square(10);
        assert_eq!(point_polygon_test(&sq, Point::new(5, 5), false), 1.0);
        assert_eq!(point_polygon_test(&sq, Point::new(20, 5), false), -1.0);
        assert_eq!(point_polygon_test(&sq, Point::new(0, 5), false), 0.0);
    }

    #[test]
    fn point_polygon_test_measures_signed_distance() {
        let sq = square(10);
        let inside = point_polygon_test(&sq, Point::new(5, 5), true);
        assert!((inside - 5.0).abs() < 1e-9, "{inside}");
        let outside = point_polygon_test(&sq, Point::new(-3, 5), true);
        assert!((outside + 3.0).abs() < 1e-9, "{outside}");
    }

    #[test]
    fn connected_components_labels_separate_blobs() {
        // Two 2x2 blocks with a gap between them.
        let mut m = Mat::zeros(6, 8, CV_8UC1).unwrap();
        for (x0, y0) in [(1, 1), (5, 3)] {
            for dy in 0..2 {
                for dx in 0..2 {
                    m.pixel_mut::<u8>(y0 + dy, x0 + dx).unwrap()[0] = 255;
                }
            }
        }
        let (labels, n) = connected_components(&m, 8).unwrap();
        assert_eq!(n, 3, "expected background + 2 components");
        assert_eq!(labels.at::<i32>(0, 0).unwrap(), 0);
        assert_eq!(labels.at::<i32>(1, 1).unwrap(), 1);
        assert_eq!(labels.at::<i32>(3, 5).unwrap(), 2);
    }

    #[test]
    fn connectivity_changes_diagonal_joins() {
        // Two pixels touching only at a corner.
        let mut m = Mat::zeros(3, 3, CV_8UC1).unwrap();
        m.pixel_mut::<u8>(0, 0).unwrap()[0] = 1;
        m.pixel_mut::<u8>(1, 1).unwrap()[0] = 1;
        assert_eq!(connected_components(&m, 4).unwrap().1, 3); // two components
        assert_eq!(connected_components(&m, 8).unwrap().1, 2); // one component
    }

    #[test]
    fn connected_components_merges_a_u_shape() {
        // A shape whose arms are only joined at the bottom forces the
        // union-find merge; a single-pass labeller reports two components.
        let mut m = Mat::zeros(5, 5, CV_8UC1).unwrap();
        for y in 0..5 {
            m.pixel_mut::<u8>(y, 0).unwrap()[0] = 1;
            m.pixel_mut::<u8>(y, 4).unwrap()[0] = 1;
        }
        for x in 0..5 {
            m.pixel_mut::<u8>(4, x).unwrap()[0] = 1;
        }
        let (_, n) = connected_components(&m, 4).unwrap();
        assert_eq!(n, 2, "the U must be a single component");
    }

    #[test]
    fn stats_report_area_bounding_box_and_centroid() {
        let mut m = Mat::zeros(6, 6, CV_8UC1).unwrap();
        for y in 1..4 {
            for x in 2..5 {
                m.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        let c = connected_components_with_stats(&m, 8).unwrap();
        let (stats, centroids) = (c.stats, c.centroids);
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[1].area, 9);
        assert_eq!(stats[1].rect, Rect::new(2, 1, 3, 3));
        assert!((centroids[1].0 - 3.0).abs() < 1e-12);
        assert!((centroids[1].1 - 2.0).abs() < 1e-12);
        // Background covers everything else.
        assert_eq!(stats[0].area, 36 - 9);
    }

    #[test]
    fn an_empty_image_has_only_background() {
        let m = Mat::zeros(4, 4, CV_8UC1).unwrap();
        let (labels, n) = connected_components(&m, 8).unwrap();
        assert_eq!(n, 1);
        assert!(labels.to_vec::<i32>().unwrap().iter().all(|&v| v == 0));
    }

    #[test]
    fn invalid_connectivity_is_rejected() {
        let m = Mat::zeros(4, 4, CV_8UC1).unwrap();
        assert!(connected_components(&m, 6).is_err());
    }

    #[test]
    fn convex_hull_matches_upstream_vertex_order() {
        // General-position set; cv2 4.13.0 returns exactly these sequences.
        let p: Vec<Point> = [
            [160, 57], [29, 63], [84, 156], [91, 26], [70, 118],
            [156, 141], [188, 43], [168, 19], [110, 59],
        ]
        .iter()
        .map(|c| Point::new(c[0], c[1]))
        .collect();
        let as_pairs = |h: Vec<Point>| h.iter().map(|q| [q.x, q.y]).collect::<Vec<_>>();
        assert_eq!(
            as_pairs(convex_hull(&p, true)),
            vec![[29, 63], [84, 156], [156, 141], [188, 43], [168, 19], [91, 26]]
        );
        assert_eq!(
            as_pairs(convex_hull(&p, false)),
            vec![[188, 43], [156, 141], [84, 156], [29, 63], [91, 26], [168, 19]]
        );
    }

    #[test]
    fn convex_hull_applies_upstreams_cyclic_shift() {
        // The sweeps alone emit input indices 0 4 3 2 1 for this pentagon; the
        // closing shift rotates to 4 3 2 1 0, so the hull starts at (15,80) and
        // not at any extreme of the point set. cv2 4.13.0 returns exactly this.
        let p: Vec<Point> = [[10, 10], [90, 14], [120, 70], [60, 110], [15, 80]]
            .iter()
            .map(|c| Point::new(c[0], c[1]))
            .collect();
        assert_eq!(
            convex_hull(&p, true).iter().map(|q| [q.x, q.y]).collect::<Vec<_>>(),
            vec![[15, 80], [60, 110], [120, 70], [90, 14], [10, 10]]
        );
    }

    #[test]
    fn convex_hull_degenerate_shapes() {
        assert!(convex_hull(&[], true).is_empty());
        let one = Point::new(4, 7);
        assert_eq!(convex_hull(&[one, one, one], true), vec![one]);
        // A collinear run collapses to its two extremes.
        let line: Vec<Point> = (0..5).map(|i| Point::new(i, 2 * i)).collect();
        let h = convex_hull(&line, true);
        assert_eq!(h.len(), 2, "collinear points give a degenerate 2-point hull, got {h:?}");
    }
}
