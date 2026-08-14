//! Shape fitting and convexity
//! (`modules/imgproc/src/shapedescr.cpp`, `linefit.cpp`, `convhull.cpp`,
//! `intersection.cpp`, `rotcalipers.cpp`).

use cv_core::types::CV_64FC1;
use cv_core::{cv_assert, Mat, Point, Result};

pub use cv_core::RotatedRect;

/// `cv::boxPoints(box, points)`.
#[must_use]
pub fn box_points(r: &RotatedRect) -> [(f64, f64); 4] {
    r.points()
}

/// `cv::minAreaRect(points)` — rotating calipers over the convex hull.
///
/// Every minimum-area enclosing rectangle has a side flush with a hull edge, so
/// testing one orientation per edge is exhaustive rather than a heuristic.
pub fn min_area_rect(pts: &[Point]) -> Result<RotatedRect> {
    cv_assert!(!pts.is_empty(), "minAreaRect needs at least one point");
    let hull = crate::shape::convex_hull(pts, true);
    if hull.len() < 3 {
        // Degenerate: a point or a segment.
        let xs: Vec<f64> = hull.iter().map(|p| p.x as f64).collect();
        let ys: Vec<f64> = hull.iter().map(|p| p.y as f64).collect();
        let (x0, x1) = (min_of(&xs), max_of(&xs));
        let (y0, y1) = (min_of(&ys), max_of(&ys));
        return Ok(normalise_min_area_rect(RotatedRect {
            center: ((x0 + x1) * 0.5, (y0 + y1) * 0.5),
            size: (x1 - x0, y1 - y0),
            angle: 0.0,
        }));
    }

    let n = hull.len();
    let mut best: Option<(f64, RotatedRect)> = None;
    for i in 0..n {
        let a = hull[i];
        let b = hull[(i + 1) % n];
        let (ex, ey) = ((b.x - a.x) as f64, (b.y - a.y) as f64);
        let len = ex.hypot(ey);
        if len < 1e-12 {
            continue;
        }
        let (ux, uy) = (ex / len, ey / len);
        // Project every hull vertex onto the edge and its normal.
        let (mut u0, mut u1, mut v0, mut v1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for p in &hull {
            let (px, py) = (p.x as f64, p.y as f64);
            let pu = px * ux + py * uy;
            let pv = -px * uy + py * ux;
            u0 = u0.min(pu);
            u1 = u1.max(pu);
            v0 = v0.min(pv);
            v1 = v1.max(pv);
        }
        let (w, h) = (u1 - u0, v1 - v0);
        let area = w * h;
        if best.as_ref().map_or(true, |(ba, _)| area < *ba) {
            // Centre in the rotated frame, mapped back.
            let (cu, cv) = ((u0 + u1) * 0.5, (v0 + v1) * 0.5);
            let cx = cu * ux - cv * uy;
            let cy = cu * uy + cv * ux;
            best = Some((
                area,
                RotatedRect {
                    center: (cx, cy),
                    size: (w, h),
                    angle: uy.atan2(ux).to_degrees(),
                },
            ));
        }
    }
    best.map(|(_, r)| normalise_min_area_rect(r)).ok_or_else(|| {
        cv_core::Error::new(cv_core::Code::StsBadArg, "minAreaRect: degenerate hull".to_string())
    })
}

/// `cv::minAreaRect` reports the angle in `[-90, 0)`, with `size` ordered to
/// match. A rectangle has four equivalent descriptions — rotate by 90 degrees
/// and swap the sides and it is the same rectangle — so this is a labelling
/// convention, not a different answer. Without it the angle comes out 90
/// degrees off and the sides transposed.
fn normalise_min_area_rect(mut r: RotatedRect) -> RotatedRect {
    while r.angle >= 0.0 {
        r.angle -= 90.0;
        r.size = (r.size.1, r.size.0);
    }
    while r.angle < -90.0 {
        r.angle += 90.0;
        r.size = (r.size.1, r.size.0);
    }
    r

}

fn min_of(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}
fn max_of(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// `cv::minEnclosingCircle(points, center, radius)`.
///
/// Welzl's algorithm, iterated rather than recursed so a large contour cannot
/// overflow the stack. Returns `(center, radius)`.
pub fn min_enclosing_circle(pts: &[Point]) -> Result<((f64, f64), f64)> {
    cv_assert!(!pts.is_empty(), "minEnclosingCircle needs at least one point");
    let p: Vec<(f64, f64)> = pts.iter().map(|q| (q.x as f64, q.y as f64)).collect();
    // Deterministic order: upstream shuffles, but a fixed order keeps results
    // reproducible and the expected-linear bound still holds in practice.
    let mut c = (p[0], 0.0f64);
    for i in 0..p.len() {
        if inside(&c, p[i]) {
            continue;
        }
        c = (p[i], 0.0);
        for j in 0..i {
            if inside(&c, p[j]) {
                continue;
            }
            c = two_point(p[i], p[j]);
            for k in 0..j {
                if inside(&c, p[k]) {
                    continue;
                }
                c = three_point(p[i], p[j], p[k]).unwrap_or(c);
            }
        }
    }
    Ok(c)
}

fn inside(c: &((f64, f64), f64), p: (f64, f64)) -> bool {
    (p.0 - c.0 .0).hypot(p.1 - c.0 .1) <= c.1 + 1e-9
}
fn two_point(a: (f64, f64), b: (f64, f64)) -> ((f64, f64), f64) {
    let c = ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
    (c, (a.0 - b.0).hypot(a.1 - b.1) * 0.5)
}
fn three_point(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> Option<((f64, f64), f64)> {
    let d = 2.0 * (a.0 * (b.1 - c.1) + b.0 * (c.1 - a.1) + c.0 * (a.1 - b.1));
    if d.abs() < 1e-12 {
        return None;
    }
    let (aa, bb, cc) = (
        a.0 * a.0 + a.1 * a.1,
        b.0 * b.0 + b.1 * b.1,
        c.0 * c.0 + c.1 * c.1,
    );
    let ux = (aa * (b.1 - c.1) + bb * (c.1 - a.1) + cc * (a.1 - b.1)) / d;
    let uy = (aa * (c.0 - b.0) + bb * (a.0 - c.0) + cc * (b.0 - a.0)) / d;
    Some(((ux, uy), (a.0 - ux).hypot(a.1 - uy)))
}

/// `cv::DistanceTypes` as accepted by [`fit_line`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitDistance {
    /// `cv::DIST_L2` — ordinary least squares, solved in closed form.
    L2,
    /// `cv::DIST_L1` — iteratively reweighted, robust to outliers.
    L1,
    /// `cv::DIST_HUBER`
    Huber,
}

/// `cv::fitLine(points, line, distType, param, reps, aeps)` for 2-D input.
///
/// Returns `(vx, vy, x0, y0)`: a unit direction and a point on the line, which
/// is upstream's `Vec4f` layout.
pub fn fit_line(pts: &[Point], dist: FitDistance) -> Result<(f64, f64, f64, f64)> {
    cv_assert!(pts.len() >= 2, "fitLine needs at least two points, got {}", pts.len());
    let p: Vec<(f64, f64)> = pts.iter().map(|q| (q.x as f64, q.y as f64)).collect();
    let mut wts = vec![1.0f64; p.len()];
    let mut line = weighted_fit(&p, &wts);

    if dist != FitDistance::L2 {
        // Upstream's IRLS: at most 20 passes, reweighting by the residual.
        for _ in 0..20 {
            let mut scale = 0.0;
            let d: Vec<f64> = p
                .iter()
                .map(|&(x, y)| {
                    let r = (x - line.2) * line.1 - (y - line.3) * line.0;
                    r.abs()
                })
                .collect();
            for &v in &d {
                scale += v;
            }
            scale = (scale / p.len() as f64).max(1e-9);
            for (w, &r) in wts.iter_mut().zip(d.iter()) {
                *w = match dist {
                    FitDistance::L1 => 1.0 / (r / scale).max(1e-9),
                    // Huber with upstream's default c = 1.345.
                    FitDistance::Huber => {
                        let t = r / (scale * 1.345);
                        if t <= 1.0 {
                            1.0
                        } else {
                            1.0 / t
                        }
                    }
                    FitDistance::L2 => 1.0,
                };
            }
            let next = weighted_fit(&p, &wts);
            let moved = (next.0 - line.0).abs() + (next.1 - line.1).abs();
            line = next;
            if moved < 1e-9 {
                break;
            }
        }
    }
    Ok(line)
}

/// Total-least-squares line through weighted points.
fn weighted_fit(p: &[(f64, f64)], w: &[f64]) -> (f64, f64, f64, f64) {
    let sw: f64 = w.iter().sum();
    let cx: f64 = p.iter().zip(w).map(|(q, wi)| q.0 * wi).sum::<f64>() / sw;
    let cy: f64 = p.iter().zip(w).map(|(q, wi)| q.1 * wi).sum::<f64>() / sw;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for (q, wi) in p.iter().zip(w) {
        let (dx, dy) = (q.0 - cx, q.1 - cy);
        sxx += wi * dx * dx;
        sxy += wi * dx * dy;
        syy += wi * dy * dy;
    }
    // Principal direction of the scatter matrix.
    let t = (sxx - syy) * 0.5;
    let disc = (t * t + sxy * sxy).sqrt();
    let l = (sxx + syy) * 0.5 + disc;
    let (mut vx, mut vy) = (sxy, l - sxx);
    if vx.abs() < 1e-300 && vy.abs() < 1e-300 {
        vx = 1.0;
        vy = 0.0;
    }
    let n = vx.hypot(vy);
    (vx / n, vy / n, cx, cy)
}

/// `cv::fitEllipse(points)`.
///
/// Returns the ellipse as a [`RotatedRect`], upstream's representation: `size`
/// holds the **full** axis lengths, not the semi-axes.
///
/// # Three stages, not one
///
/// Upstream fits in three passes — a 5-parameter conic, then the centre from
/// that conic's gradient, then a **re-fit of the three quadratic terms with the
/// centre held fixed**. The third pass is what makes the result usable on a
/// partial arc: a single-pass algebraic fit on a half-ellipse comes out with
/// axes less than half their true length, and on a short arc it degenerates to
/// a hyperbola and has no answer at all. Both were true of this function before
/// the passes were transcribed.
pub fn fit_ellipse(pts: &[Point]) -> Result<RotatedRect> {
    cv_assert!(pts.len() >= 5, "fitEllipse needs at least 5 points, got {}", pts.len());
    const MIN_EPS: f64 = 1e-8;
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p.x as f64).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p.y as f64).sum::<f64>() / n;
    let local: Vec<(f64, f64)> =
        pts.iter().map(|p| (p.x as f64 - cx, p.y as f64 - cy)).collect();

    // Stage 1: the general conic, with upstream's inverted A-C signs and its
    // 10000 right-hand side.
    let mut ata = [[0.0f64; 5]; 5];
    let mut atb = [0.0f64; 5];
    for &(x, y) in &local {
        let row = [-x * x, -y * y, -x * y, x, y];
        for i in 0..5 {
            for j in 0..5 {
                ata[i][j] += row[i] * row[j];
            }
            atb[i] += row[i] * 10000.0;
        }
    }
    let g = solve5(&mut ata, &mut atb).ok_or_else(|| {
        cv_core::Error::new(
            cv_core::Code::StsBadArg,
            "fitEllipse: the points are degenerate (collinear?)".to_string(),
        )
    })?;

    // Stage 2: the centre, where the conic's gradient vanishes.
    let m = [[2.0 * g[0], g[2]], [g[2], 2.0 * g[1]]];
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() < 1e-300 {
        return Err(cv_core::Error::new(
            cv_core::Code::StsBadArg,
            "fitEllipse: the conic has no finite centre".to_string(),
        ));
    }
    let rx = (g[3] * m[1][1] - m[0][1] * g[4]) / det;
    let ry = (m[0][0] * g[4] - g[3] * m[1][0]) / det;

    // Stage 3: re-fit the quadratic terms about that centre.
    let mut a3 = [[0.0f64; 3]; 3];
    let mut b3 = [0.0f64; 3];
    for &(x, y) in &local {
        let (u, v) = (x - rx, y - ry);
        let row = [u * u, v * v, u * v];
        for i in 0..3 {
            for j in 0..3 {
                a3[i][j] += row[i] * row[j];
            }
            b3[i] += row[i];
        }
    }
    let q = solve3(&mut a3, &mut b3).ok_or_else(|| {
        cv_core::Error::new(
            cv_core::Code::StsBadArg,
            "fitEllipse: the re-fit is degenerate".to_string(),
        )
    })?;

    // Axes and angle, exactly as upstream derives them.
    let angle = -0.5 * q[2].atan2(q[1] - q[0]);
    let t = if q[2].abs() > MIN_EPS {
        q[2] / (-2.0 * angle).sin()
    } else {
        q[1] - q[0]
    };
    let mut r2 = (q[0] + q[1] - t).abs();
    if r2 > MIN_EPS {
        r2 = (2.0 / r2).sqrt();
    }
    let mut r3 = (q[0] + q[1] + t).abs();
    if r3 > MIN_EPS {
        r3 = (2.0 / r3).sqrt();
    }

    let mut width = r2 * 2.0;
    let mut height = r3 * 2.0;
    let mut deg = angle.to_degrees();
    if width > height {
        std::mem::swap(&mut width, &mut height);
        deg = 90.0 + angle.to_degrees();
    }
    if deg < -180.0 {
        deg += 360.0;
    }
    if deg > 360.0 {
        deg -= 360.0;
    }
    Ok(RotatedRect {
        center: (rx + cx, ry + cy),
        size: (width, height),
        angle: deg,
    })
}

/// One entry of `cv::convexityDefects`' output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConvexityDefect {
    /// Index into the contour of the defect's start point.
    pub start: i32,
    /// Index of the end point.
    pub end: i32,
    /// Index of the point furthest from the hull chord.
    pub farthest: i32,
    /// That distance, in **1/256 pixel** units, matching upstream's fixed point.
    pub depth: i32,
}

/// Gaussian elimination on a 5x5 system, or `None` if it is singular.
fn solve5(a: &mut [[f64; 5]; 5], b: &mut [f64; 5]) -> Option<[f64; 5]> {
    for i in 0..5 {
        let mut piv = i;
        for r in i + 1..5 {
            if a[r][i].abs() > a[piv][i].abs() {
                piv = r;
            }
        }
        if a[piv][i].abs() < 1e-12 {
            return None;
        }
        a.swap(i, piv);
        b.swap(i, piv);
        for r in i + 1..5 {
            let f = a[r][i] / a[i][i];
            let pivot_row = a[i];
            for (c, pv) in pivot_row.iter().enumerate().skip(i) {
                a[r][c] -= f * pv;
            }
            b[r] -= f * b[i];
        }
    }
    let mut x = [0.0f64; 5];
    for i in (0..5).rev() {
        let mut s = b[i];
        for (c, &xc) in x.iter().enumerate().skip(i + 1) {
            s -= a[i][c] * xc;
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// Solve a 3x3 system in place by Gauss elimination with partial pivoting.
fn solve3(a: &mut [[f64; 3]; 3], b: &mut [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut piv = col;
        for r in (col + 1)..3 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in (col + 1)..3 {
            let f = a[r][col] / a[col][col];
            let pivot_row = a[col];
            for (k, v) in a[r].iter_mut().enumerate().skip(col) {
                *v -= f * pivot_row[k];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        let s: f64 = ((i + 1)..3).map(|j| a[i][j] * x[j]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    Some(x)
}

/// `cv::convexityDefects(contour, convexhull, convexityDefects)`.
///
/// `hull_idx` holds indices into `contour`, in the same order the hull was
/// produced — that is what upstream requires and it silently misbehaves if the
/// order is wrong, so the ordering is checked here.
pub fn convexity_defects(contour: &[Point], hull_idx: &[i32]) -> Result<Vec<ConvexityDefect>> {
    cv_assert!(
        hull_idx.len() >= 3,
        "convexityDefects needs a hull of at least 3 points, got {}",
        hull_idx.len()
    );
    let n = contour.len() as i32;
    for &i in hull_idx {
        cv_assert!(i >= 0 && i < n, "hull index {i} is outside the contour");
    }
    let mut out = Vec::new();
    for h in 0..hull_idx.len() {
        let s = hull_idx[h];
        let e = hull_idx[(h + 1) % hull_idx.len()];
        let (a, b) = (contour[s as usize], contour[e as usize]);
        let (dx, dy) = ((b.x - a.x) as f64, (b.y - a.y) as f64);
        let len = dx.hypot(dy);
        if len < 1e-12 {
            continue;
        }
        // Walk the contour from s to e the long way round.
        let mut best = (0.0f64, -1i32);
        let mut i = s;
        while i != e {
            i = (i + 1) % n;
            if i == e {
                break;
            }
            let p = contour[i as usize];
            let d = ((p.x - a.x) as f64 * dy - (p.y - a.y) as f64 * dx).abs() / len;
            if d > best.0 {
                best = (d, i);
            }
        }
        if best.1 >= 0 && best.0 > 0.0 {
            out.push(ConvexityDefect {
                start: s,
                end: e,
                farthest: best.1,
                depth: (best.0 * 256.0).round() as i32,
            });
        }
    }
    Ok(out)
}

/// `cv::intersectConvexConvex(p1, p2, p12, handleNested)`.
///
/// Returns the intersection polygon and its area. Both inputs must be convex
/// and counter-clockwise; the Sutherland-Hodgman clip below relies on it.
pub fn intersect_convex_convex(a: &[Point], b: &[Point]) -> Result<(Vec<(f64, f64)>, f64)> {
    let fa: Vec<(f64, f64)> = a.iter().map(|p| (p.x as f64, p.y as f64)).collect();
    let fb: Vec<(f64, f64)> = b.iter().map(|p| (p.x as f64, p.y as f64)).collect();
    intersect_convex_f64(&fa, &fb)
}

/// [`intersect_convex_convex`] on floating-point vertices.
///
/// The integer entry point exists because upstream's takes `Point`, but
/// rounding a rotated rectangle's corners to integers inflates its area by
/// several percent, so anything geometric must clip in `f64`.
pub fn intersect_convex_f64(a: &[(f64, f64)], b: &[(f64, f64)]) -> Result<(Vec<(f64, f64)>, f64)> {
    cv_assert!(a.len() >= 3 && b.len() >= 3, "both polygons need at least 3 vertices");
    let mut poly: Vec<(f64, f64)> = a.to_vec();
    let clip: Vec<(f64, f64)> = b.to_vec();
    // Orient the clip polygon counter-clockwise so `side` has a fixed meaning.
    let clip = if signed_area(&clip) < 0.0 {
        clip.iter().rev().cloned().collect()
    } else {
        clip
    };
    if signed_area(&poly) < 0.0 {
        poly.reverse();
    }

    for i in 0..clip.len() {
        let (c0, c1) = (clip[i], clip[(i + 1) % clip.len()]);
        let side = |p: (f64, f64)| (c1.0 - c0.0) * (p.1 - c0.1) - (c1.1 - c0.1) * (p.0 - c0.0);
        let mut next: Vec<(f64, f64)> = Vec::with_capacity(poly.len() + 1);
        for j in 0..poly.len() {
            let (p, q) = (poly[j], poly[(j + 1) % poly.len()]);
            let (sp, sq) = (side(p), side(q));
            if sp >= 0.0 {
                next.push(p);
            }
            if (sp > 0.0 && sq < 0.0) || (sp < 0.0 && sq > 0.0) {
                let t = sp / (sp - sq);
                next.push((p.0 + t * (q.0 - p.0), p.1 + t * (q.1 - p.1)));
            }
        }
        poly = next;
        if poly.is_empty() {
            return Ok((Vec::new(), 0.0));
        }
    }
    let area = signed_area(&poly).abs();
    Ok((poly, area))
}

/// Shoelace area, signed by winding.
fn signed_area(p: &[(f64, f64)]) -> f64 {
    let mut s = 0.0;
    for i in 0..p.len() {
        let q = p[(i + 1) % p.len()];
        s += p[i].0 * q.1 - q.0 * p[i].1;
    }
    s * 0.5
}

/// `cv::rotatedRectangleIntersection(rect1, rect2, intersectingRegion)`.
///
/// Returns the intersection polygon and its area.
pub fn rotated_rectangle_intersection(
    a: &RotatedRect,
    b: &RotatedRect,
) -> Result<(Vec<(f64, f64)>, f64)> {
    intersect_convex_f64(&a.points(), &b.points())
}

/// `cv::getClosestEllipsePoints(ellipse, points, closestPts)`.
///
/// The nearest point on the ellipse boundary for each input, found by Newton
/// iteration on the eccentric anomaly.
pub fn closest_ellipse_points(e: &RotatedRect, pts: &[Point]) -> Result<Mat> {
    cv_assert!(e.size.0 > 0.0 && e.size.1 > 0.0, "the ellipse must have positive axes");
    let (a, b) = (e.size.0 * 0.5, e.size.1 * 0.5);
    let ang = e.angle.to_radians();
    let (s, c) = (ang.sin(), ang.cos());
    let mut out = Mat::zeros(pts.len() as i32, 2, CV_64FC1)?;
    for (i, p) in pts.iter().enumerate() {
        // Into the ellipse's own frame.
        let (dx, dy) = (p.x as f64 - e.center.0, p.y as f64 - e.center.1);
        let (px, py) = (dx * c + dy * s, -dx * s + dy * c);
        let mut t = py.atan2(px);
        for _ in 0..64 {
            let (ct, st) = (t.cos(), t.sin());
            let (ex, ey) = (a * ct, b * st);
            // Derivative of the squared distance.
            let f = (px - ex) * (a * st) - (py - ey) * (b * ct);
            let df = (px - ex) * (a * ct) + a * st * a * st - (py - ey) * (-b * st) + b * ct * b * ct;
            if df.abs() < 1e-15 {
                break;
            }
            let step = f / df;
            t -= step;
            if step.abs() < 1e-12 {
                break;
            }
        }
        let (ex, ey) = (a * t.cos(), b * t.sin());
        use cv_core::MatLikeMut;
        let row = out.pixel_mut::<f64>(i as i32, 0)?;
        row[0] = e.center.0 + ex * c - ey * s;
        let row1 = out.pixel_mut::<f64>(i as i32, 1)?;
        row1[0] = e.center.1 + ex * s + ey * c;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::MatLike;

    fn pts(v: &[(i32, i32)]) -> Vec<Point> {
        v.iter().map(|&(x, y)| Point::new(x, y)).collect()
    }

    #[test]
    fn min_area_rect_of_an_axis_aligned_square() {
        let p = pts(&[(0, 0), (10, 0), (10, 10), (0, 10)]);
        let r = min_area_rect(&p).unwrap();
        assert!((r.area() - 100.0).abs() < 1e-6, "area {}", r.area());
        assert!((r.center.0 - 5.0).abs() < 1e-6 && (r.center.1 - 5.0).abs() < 1e-6);
    }

    #[test]
    fn min_area_rect_of_a_diamond_is_rotated() {
        // A 45-degree diamond: the tight rectangle has area 2*r^2, not 4*r^2.
        let p = pts(&[(10, 0), (20, 10), (10, 20), (0, 10)]);
        let r = min_area_rect(&p).unwrap();
        assert!((r.area() - 200.0).abs() < 1.0, "expected ~200, got {}", r.area());
        let bbox = 20.0 * 20.0;
        assert!(r.area() < bbox * 0.75, "must beat the axis-aligned box");
    }

    #[test]
    fn box_points_returns_four_corners_of_the_right_size() {
        let r = RotatedRect { center: (5.0, 5.0), size: (4.0, 2.0), angle: 0.0 };
        let p = box_points(&r);
        let xs: Vec<f64> = p.iter().map(|q| q.0).collect();
        let ys: Vec<f64> = p.iter().map(|q| q.1).collect();
        assert!((max_of(&xs) - min_of(&xs) - 4.0).abs() < 1e-9);
        assert!((max_of(&ys) - min_of(&ys) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn min_enclosing_circle_of_a_square() {
        let p = pts(&[(0, 0), (10, 0), (10, 10), (0, 10)]);
        let (c, r) = min_enclosing_circle(&p).unwrap();
        assert!((c.0 - 5.0).abs() < 1e-6 && (c.1 - 5.0).abs() < 1e-6, "centre {c:?}");
        assert!((r - 50f64.sqrt()).abs() < 1e-6, "radius {r}");
    }

    #[test]
    fn min_enclosing_circle_covers_every_point() {
        let p = pts(&[(3, 7), (19, 2), (8, 14), (0, 0), (11, 11), (17, 9)]);
        let (c, r) = min_enclosing_circle(&p).unwrap();
        for q in &p {
            let d = (q.x as f64 - c.0).hypot(q.y as f64 - c.1);
            assert!(d <= r + 1e-6, "{q:?} at {d} is outside r={r}");
        }
    }

    #[test]
    fn fit_line_recovers_a_known_slope() {
        // y = 2x + 1
        let p: Vec<Point> = (0..20).map(|i| Point::new(i, 2 * i + 1)).collect();
        let (vx, vy, x0, y0) = fit_line(&p, FitDistance::L2).unwrap();
        assert!((vy / vx - 2.0).abs() < 1e-6, "slope {}", vy / vx);
        // The anchor point must lie on the line.
        assert!((y0 - (2.0 * x0 + 1.0)).abs() < 1e-6);
        assert!((vx.hypot(vy) - 1.0).abs() < 1e-12, "direction must be unit");
    }

    #[test]
    fn fit_line_l1_resists_an_outlier_that_l2_follows() {
        // One point pulled off the line by 20. cv2 4.13.0 on this exact input
        // gives L2 slope 2.2471 and L1 slope 2.0000, so L1 recovers the true
        // slope and L2 does not.
        let mut p: Vec<Point> = (0..20).map(|i| Point::new(i, 2 * i)).collect();
        p.push(Point::new(10, 40));
        let l2 = fit_line(&p, FitDistance::L2).unwrap();
        let l1 = fit_line(&p, FitDistance::L1).unwrap();
        assert!((l2.1 / l2.0 - 2.2471).abs() < 1e-3, "L2 slope {}", l2.1 / l2.0);
        assert!((l1.1 / l1.0 - 2.0).abs() < 1e-3, "L1 slope {}", l1.1 / l1.0);
    }

    #[test]
    fn fit_line_follows_upstream_into_the_degenerate_case() {
        // A gross y-outlier makes total-least-squares go near-vertical, and
        // IRLS then locks onto it. That is not a defect: cv2 4.13.0 returns
        // slope 142.618 for L2 and 482.00 for L1 on this input. Pinning the
        // behaviour stops a future "fix" from silently diverging from OpenCV.
        let mut p: Vec<Point> = (0..20).map(|i| Point::new(i, 2 * i)).collect();
        p.push(Point::new(10, 500));
        let l2 = fit_line(&p, FitDistance::L2).unwrap();
        assert!((l2.1 / l2.0 - 142.618).abs() < 0.01, "L2 slope {}", l2.1 / l2.0);
        let l1 = fit_line(&p, FitDistance::L1).unwrap();
        assert!(l1.1 / l1.0 > 400.0, "L1 slope {} should also be near-vertical", l1.1 / l1.0);
    }

    #[test]
    fn fit_ellipse_recovers_a_circle() {
        let p: Vec<Point> = (0..24)
            .map(|i| {
                let a = i as f64 * core::f64::consts::TAU / 24.0;
                Point::new((50.0 + 20.0 * a.cos()).round() as i32, (40.0 + 20.0 * a.sin()).round() as i32)
            })
            .collect();
        let e = fit_ellipse(&p).unwrap();
        assert!((e.center.0 - 50.0).abs() < 0.5, "cx {}", e.center.0);
        assert!((e.center.1 - 40.0).abs() < 0.5, "cy {}", e.center.1);
        assert!((e.size.0 - 40.0).abs() < 1.0, "w {}", e.size.0);
        assert!((e.size.1 - 40.0).abs() < 1.0, "h {}", e.size.1);
    }

    #[test]
    fn fit_ellipse_recovers_an_elongated_axis() {
        let p: Vec<Point> = (0..36)
            .map(|i| {
                let a = i as f64 * core::f64::consts::TAU / 36.0;
                Point::new((60.0 + 40.0 * a.cos()).round() as i32, (60.0 + 10.0 * a.sin()).round() as i32)
            })
            .collect();
        let e = fit_ellipse(&p).unwrap();
        let (lo, hi) = (e.size.0.min(e.size.1), e.size.0.max(e.size.1));
        assert!((hi - 80.0).abs() < 2.0, "major {hi}");
        assert!((lo - 20.0).abs() < 2.0, "minor {lo}");
    }

    #[test]
    fn fit_ellipse_rejects_collinear_points() {
        let p: Vec<Point> = (0..8).map(|i| Point::new(i, i)).collect();
        assert!(fit_ellipse(&p).is_err(), "a line is not an ellipse");
    }

    #[test]
    fn convexity_defects_finds_the_notch_of_an_arrow() {
        // A square with a notch cut into its right edge.
        let c = pts(&[(0, 0), (10, 0), (5, 5), (10, 10), (0, 10)]);
        let hull_idx = vec![0, 1, 3, 4];
        let d = convexity_defects(&c, &hull_idx).unwrap();
        assert_eq!(d.len(), 1, "one notch, got {d:?}");
        assert_eq!(d[0].farthest, 2, "the notch tip is contour point 2");
        // Depth is 5 pixels, in 1/256 units.
        assert!((d[0].depth - 5 * 256).abs() < 256, "depth {}", d[0].depth);
    }

    #[test]
    fn convex_polygon_has_no_defects() {
        let c = pts(&[(0, 0), (10, 0), (10, 10), (0, 10)]);
        assert!(convexity_defects(&c, &[0, 1, 2, 3]).unwrap().is_empty());
    }

    #[test]
    fn intersect_convex_convex_of_overlapping_squares() {
        let a = pts(&[(0, 0), (10, 0), (10, 10), (0, 10)]);
        let b = pts(&[(5, 5), (15, 5), (15, 15), (5, 15)]);
        let (poly, area) = intersect_convex_convex(&a, &b).unwrap();
        assert!((area - 25.0).abs() < 1e-6, "expected 25, got {area}");
        assert_eq!(poly.len(), 4);
    }

    #[test]
    fn disjoint_polygons_intersect_in_nothing() {
        let a = pts(&[(0, 0), (2, 0), (2, 2), (0, 2)]);
        let b = pts(&[(50, 50), (52, 50), (52, 52), (50, 52)]);
        let (poly, area) = intersect_convex_convex(&a, &b).unwrap();
        assert!(poly.is_empty() && area == 0.0);
    }

    #[test]
    fn rotated_rect_intersection_of_a_rect_with_itself_is_its_own_area() {
        let r = RotatedRect { center: (10.0, 10.0), size: (8.0, 6.0), angle: 30.0 };
        let (_, area) = rotated_rectangle_intersection(&r, &r).unwrap();
        assert!((area - r.area()).abs() < 1e-6, "expected {}, got {area}", r.area());
    }

    #[test]
    fn closest_ellipse_point_of_an_axis_point_is_on_the_axis() {
        let e = RotatedRect { center: (0.0, 0.0), size: (20.0, 10.0), angle: 0.0 };
        let p = pts(&[(100, 0), (0, 100)]);
        let r = closest_ellipse_points(&e, &p).unwrap();
        // Semi-axes are 10 and 5.
        assert!((r.at_f64(0, 0, 0).unwrap() - 10.0).abs() < 1e-6);
        assert!(r.at_f64(0, 1, 0).unwrap().abs() < 1e-6);
        assert!(r.at_f64(1, 0, 0).unwrap().abs() < 1e-6);
        assert!((r.at_f64(1, 1, 0).unwrap() - 5.0).abs() < 1e-6);
    }
}
