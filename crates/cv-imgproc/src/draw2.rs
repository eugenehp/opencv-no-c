//! Ellipses, markers and polygon fills (`modules/imgproc/src/drawing.cpp`).
//!
//! The primitives in [`crate::drawing`] cover lines, rectangles, circles and
//! generic polygons; this module adds the ones that need their own geometry.

use cv_core::{cv_assert, MatLikeMut, Point, Rect, Result, Scalar};

use crate::drawing::{fill_poly, line, polylines};

/// `cv::ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta, pts)`.
///
/// Samples the arc every `delta` degrees. Upstream's rounding is reproduced:
/// the sampled angle is `arc_start + i*delta` clamped to `arc_end`, and the
/// final point is always emitted so a full ellipse closes exactly.
pub fn ellipse_2_poly(
    center: Point,
    axes: (i32, i32),
    angle: f64,
    arc_start: f64,
    arc_end: f64,
    delta: i32,
) -> Result<Vec<Point>> {
    cv_assert!(delta > 0, "delta must be positive, got {delta}");
    cv_assert!(axes.0 >= 0 && axes.1 >= 0, "axes must be non-negative");
    let (mut start, mut end) = (arc_start, arc_end);
    if end < start {
        core::mem::swap(&mut start, &mut end);
    }
    // Upstream normalises a full sweep to exactly 360 degrees.
    if (end - start) > 360.0 {
        start = 0.0;
        end = 360.0;
    }
    let a = angle.to_radians();
    let (sa, ca) = (a.sin(), a.cos());
    let (rx, ry) = (axes.0 as f64, axes.1 as f64);

    let mut pts = Vec::new();
    let mut t = start;
    loop {
        let done = t >= end;
        let tt = if done { end } else { t };
        let r = tt.to_radians();
        let (x, y) = (rx * r.cos(), ry * r.sin());
        pts.push(Point::new(
            center.x + (x * ca - y * sa).round() as i32,
            center.y + (x * sa + y * ca).round() as i32,
        ));
        if done {
            break;
        }
        t += delta as f64;
    }
    Ok(pts)
}

/// `cv::ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)`.
///
/// A negative `thickness` fills the ellipse (upstream's `FILLED`).
#[allow(clippy::too_many_arguments)]
pub fn ellipse<M: MatLikeMut + ?Sized>(
    img: &mut M,
    center: Point,
    axes: (i32, i32),
    angle: f64,
    start_angle: f64,
    end_angle: f64,
    color: Scalar,
    thickness: i32,
) -> Result<()> {
    // 5 degrees is upstream's sampling step for a drawn ellipse.
    let pts = ellipse_2_poly(center, axes, angle, start_angle, end_angle, 5)?;
    if pts.len() < 2 {
        return Ok(());
    }
    let full = (end_angle - start_angle).abs() >= 360.0;
    if thickness < 0 {
        if full {
            fill_poly(img, &pts, color)
        } else {
            // A partial arc fills as a pie slice, with the centre closing it.
            let mut wedge = pts.clone();
            wedge.push(center);
            fill_poly(img, &wedge, color)
        }
    } else {
        polylines(img, &pts, full, color, thickness.max(1))
    }
}

/// `cv::fillConvexPoly(img, points, color)`.
///
/// Distinct from [`fill_poly`]: it assumes convexity (or at worst a shape whose
/// horizontal lines cross the boundary at most twice) and so fills between the
/// leftmost and rightmost crossing of every scanline rather than running an
/// even-odd parity test. On a convex polygon the two agree; on a self-crossing
/// one they do not, which is exactly why upstream keeps both.
pub fn fill_convex_poly<M: MatLikeMut + ?Sized>(
    img: &mut M,
    pts: &[Point],
    color: Scalar,
) -> Result<()> {
    if pts.len() < 3 {
        return Ok(());
    }
    let y0 = pts.iter().map(|p| p.y).min().unwrap().max(0);
    let y1 = pts.iter().map(|p| p.y).max().unwrap().min(img.rows() - 1);
    let n = pts.len();
    for y in y0..=y1 {
        let (mut lo, mut hi) = (i32::MAX, i32::MIN);
        for i in 0..n {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            if a.y == b.y {
                // A horizontal edge on this row still bounds the span.
                if a.y == y {
                    lo = lo.min(a.x.min(b.x));
                    hi = hi.max(a.x.max(b.x));
                }
                continue;
            }
            let (top, bot) = if a.y < b.y { (a, b) } else { (b, a) };
            if y < top.y || y > bot.y {
                continue;
            }
            let t = (y - top.y) as f64 / (bot.y - top.y) as f64;
            let x = (top.x as f64 + t * (bot.x - top.x) as f64).round() as i32;
            lo = lo.min(x);
            hi = hi.max(x);
        }
        if lo > hi {
            continue;
        }
        for x in lo.max(0)..=hi.min(img.cols() - 1) {
            put(img, y, x, color)?;
        }
    }
    Ok(())
}

/// Write one pixel, saturating per channel.
fn put<M: MatLikeMut + ?Sized>(img: &mut M, y: i32, x: i32, color: Scalar) -> Result<()> {
    for c in 0..img.channels() {
        cv_core::arithm::set_channel_generic(img, y, x, c, color.0[c.min(3) as usize])?;
    }
    Ok(())
}

/// `cv::clipLine(imgRect, pt1, pt2)`.
///
/// Returns the clipped endpoints, or `None` when the segment misses the
/// rectangle entirely. The rectangle is inclusive of `x..x+width-1`, as
/// upstream's `Rect` overload is.
#[must_use]
pub fn clip_line(rect: Rect, p1: Point, p2: Point) -> Option<(Point, Point)> {
    // Liang-Barsky.
    let (x0, y0) = (rect.x as f64, rect.y as f64);
    let (x1, y1) = ((rect.x + rect.width - 1) as f64, (rect.y + rect.height - 1) as f64);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    let dx = (p2.x - p1.x) as f64;
    let dy = (p2.y - p1.y) as f64;

    for (p, q) in [
        (-dx, p1.x as f64 - x0),
        (dx, x1 - p1.x as f64),
        (-dy, p1.y as f64 - y0),
        (dy, y1 - p1.y as f64),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None; // parallel and outside
            }
            continue;
        }
        let r = q / p;
        if p < 0.0 {
            if r > t1 {
                return None;
            }
            t0 = t0.max(r);
        } else {
            if r < t0 {
                return None;
            }
            t1 = t1.min(r);
        }
    }
    Some((
        Point::new(
            (p1.x as f64 + t0 * dx).round() as i32,
            (p1.y as f64 + t0 * dy).round() as i32,
        ),
        Point::new(
            (p1.x as f64 + t1 * dx).round() as i32,
            (p1.y as f64 + t1 * dy).round() as i32,
        ),
    ))
}

/// `cv::arrowedLine(img, pt1, pt2, color, thickness, line_type, shift, tipLength)`.
///
/// `tip_length` is a fraction of the shaft length, as upstream defines it.
pub fn arrowed_line<M: MatLikeMut + ?Sized>(
    img: &mut M,
    p1: Point,
    p2: Point,
    color: Scalar,
    thickness: i32,
    tip_length: f64,
) -> Result<()> {
    line(img, p1, p2, color, thickness)?;
    let dx = (p1.x - p2.x) as f64;
    let dy = (p1.y - p2.y) as f64;
    let len = dx.hypot(dy);
    if len < 1e-9 {
        return Ok(());
    }
    let tip = tip_length * len;
    let angle = dy.atan2(dx);
    // Upstream's barbs sit at +/- pi/4 from the shaft.
    for s in [-1.0f64, 1.0] {
        let a = angle + s * core::f64::consts::FRAC_PI_4;
        let q = Point::new(
            p2.x + (tip * a.cos()).round() as i32,
            p2.y + (tip * a.sin()).round() as i32,
        );
        line(img, p2, q, color, thickness)?;
    }
    Ok(())
}

/// `cv::MarkerTypes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerType {
    /// `cv::MARKER_CROSS`
    Cross,
    /// `cv::MARKER_TILTED_CROSS`
    TiltedCross,
    /// `cv::MARKER_STAR` — both crosses together.
    Star,
    /// `cv::MARKER_DIAMOND`
    Diamond,
    /// `cv::MARKER_SQUARE`
    Square,
    /// `cv::MARKER_TRIANGLE_UP`
    TriangleUp,
    /// `cv::MARKER_TRIANGLE_DOWN`
    TriangleDown,
}

/// `cv::drawMarker(img, position, color, markerType, markerSize, thickness)`.
pub fn draw_marker<M: MatLikeMut + ?Sized>(
    img: &mut M,
    p: Point,
    color: Scalar,
    kind: MarkerType,
    size: i32,
    thickness: i32,
) -> Result<()> {
    cv_assert!(size > 0, "marker size must be positive, got {size}");
    let r = size / 2;
    let t = thickness.max(1);
    let cross = |img: &mut M| -> Result<()> {
        line(img, Point::new(p.x - r, p.y), Point::new(p.x + r, p.y), color, t)?;
        line(img, Point::new(p.x, p.y - r), Point::new(p.x, p.y + r), color, t)
    };
    let tilted = |img: &mut M| -> Result<()> {
        line(img, Point::new(p.x - r, p.y - r), Point::new(p.x + r, p.y + r), color, t)?;
        line(img, Point::new(p.x - r, p.y + r), Point::new(p.x + r, p.y - r), color, t)
    };
    match kind {
        MarkerType::Cross => cross(img),
        MarkerType::TiltedCross => tilted(img),
        MarkerType::Star => {
            cross(img)?;
            tilted(img)
        }
        MarkerType::Diamond => polylines(
            img,
            &[
                Point::new(p.x, p.y - r),
                Point::new(p.x + r, p.y),
                Point::new(p.x, p.y + r),
                Point::new(p.x - r, p.y),
            ],
            true,
            color,
            t,
        ),
        MarkerType::Square => polylines(
            img,
            &[
                Point::new(p.x - r, p.y - r),
                Point::new(p.x + r, p.y - r),
                Point::new(p.x + r, p.y + r),
                Point::new(p.x - r, p.y + r),
            ],
            true,
            color,
            t,
        ),
        MarkerType::TriangleUp => polylines(
            img,
            &[
                Point::new(p.x, p.y - r),
                Point::new(p.x + r, p.y + r),
                Point::new(p.x - r, p.y + r),
            ],
            true,
            color,
            t,
        ),
        MarkerType::TriangleDown => polylines(
            img,
            &[
                Point::new(p.x, p.y + r),
                Point::new(p.x + r, p.y - r),
                Point::new(p.x - r, p.y - r),
            ],
            true,
            color,
            t,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::{CV_8UC1, CV_8UC3};
    use cv_core::{Mat, MatLike};

    fn canvas() -> Mat {
        Mat::zeros(40, 40, CV_8UC1).unwrap()
    }
    fn lit(m: &Mat) -> usize {
        m.to_vec::<u8>().unwrap().iter().filter(|&&v| v > 0).count()
    }

    #[test]
    fn ellipse_2_poly_closes_a_full_sweep() {
        let p = ellipse_2_poly(Point::new(0, 0), (10, 5), 0.0, 0.0, 360.0, 90).unwrap();
        // 0, 90, 180, 270, 360 -> five points, first and last coincide.
        assert_eq!(p.len(), 5);
        assert_eq!(p[0], p[4]);
        assert_eq!(p[0], Point::new(10, 0));
        assert_eq!(p[1], Point::new(0, 5));
        assert_eq!(p[2], Point::new(-10, 0));
    }

    #[test]
    fn ellipse_2_poly_respects_rotation() {
        // A 90-degree rotation swaps the axes.
        let p = ellipse_2_poly(Point::new(0, 0), (10, 5), 90.0, 0.0, 0.0, 30).unwrap();
        assert_eq!(p[0], Point::new(0, 10));
    }

    #[test]
    fn ellipse_2_poly_partial_arc_is_shorter() {
        let full = ellipse_2_poly(Point::new(0, 0), (8, 8), 0.0, 0.0, 360.0, 10).unwrap();
        let half = ellipse_2_poly(Point::new(0, 0), (8, 8), 0.0, 0.0, 180.0, 10).unwrap();
        assert!(half.len() < full.len());
    }

    #[test]
    fn a_filled_ellipse_covers_more_than_its_outline() {
        let mut a = canvas();
        ellipse(&mut a, Point::new(20, 20), (10, 6), 0.0, 0.0, 360.0, Scalar::all(255.0), 1)
            .unwrap();
        let mut b = canvas();
        ellipse(&mut b, Point::new(20, 20), (10, 6), 0.0, 0.0, 360.0, Scalar::all(255.0), -1)
            .unwrap();
        assert!(lit(&b) > lit(&a) * 2, "outline {} vs fill {}", lit(&a), lit(&b));
        // The centre is inside the fill and outside the outline.
        assert_eq!(b.at_f64(20, 20, 0).unwrap(), 255.0);
        assert_eq!(a.at_f64(20, 20, 0).unwrap(), 0.0);
    }

    #[test]
    fn fill_convex_poly_fills_a_triangle() {
        let mut m = canvas();
        fill_convex_poly(
            &mut m,
            &[Point::new(5, 5), Point::new(35, 5), Point::new(20, 30)],
            Scalar::all(255.0),
        )
        .unwrap();
        // A point well inside is set; a point outside is not.
        assert_eq!(m.at_f64(10, 20, 0).unwrap(), 255.0);
        assert_eq!(m.at_f64(35, 2, 0).unwrap(), 0.0);
        assert!(lit(&m) > 300, "a 30x25 triangle should cover ~375 px, got {}", lit(&m));
    }

    #[test]
    fn fill_convex_poly_and_fill_poly_agree_on_a_convex_shape() {
        let pts = [Point::new(5, 5), Point::new(30, 8), Point::new(25, 30), Point::new(8, 26)];
        let mut a = canvas();
        fill_convex_poly(&mut a, &pts, Scalar::all(255.0)).unwrap();
        let mut b = canvas();
        fill_poly(&mut b, &pts, Scalar::all(255.0)).unwrap();
        let (va, vb) = (a.to_vec::<u8>().unwrap(), b.to_vec::<u8>().unwrap());
        let differ = va.iter().zip(vb.iter()).filter(|(x, y)| x != y).count();
        // Boundary rounding can differ by a pixel or two; the interiors must not.
        assert!(differ * 20 < lit(&a), "{differ} pixels differ out of {}", lit(&a));
    }

    #[test]
    fn fill_convex_poly_differs_from_fill_poly_on_a_bowtie() {
        // A self-crossing quad: even-odd leaves the crossing empty, the convex
        // fill spans it. This is the case the two functions exist to separate.
        let pts = [Point::new(5, 5), Point::new(30, 30), Point::new(30, 5), Point::new(5, 30)];
        let mut a = canvas();
        fill_convex_poly(&mut a, &pts, Scalar::all(255.0)).unwrap();
        let mut b = canvas();
        fill_poly(&mut b, &pts, Scalar::all(255.0)).unwrap();
        assert!(lit(&a) > lit(&b), "convex {} should exceed even-odd {}", lit(&a), lit(&b));
    }

    #[test]
    fn clip_line_trims_to_the_rectangle() {
        let r = Rect::new(0, 0, 10, 10); // covers x,y in 0..=9
        let (a, b) = clip_line(r, Point::new(-5, 5), Point::new(20, 5)).unwrap();
        assert_eq!(a, Point::new(0, 5));
        assert_eq!(b, Point::new(9, 5));
    }

    #[test]
    fn clip_line_keeps_an_inside_segment_whole() {
        let r = Rect::new(0, 0, 10, 10);
        let (a, b) = clip_line(r, Point::new(2, 2), Point::new(7, 8)).unwrap();
        assert_eq!((a, b), (Point::new(2, 2), Point::new(7, 8)));
    }

    #[test]
    fn clip_line_rejects_a_missing_segment() {
        let r = Rect::new(0, 0, 10, 10);
        assert!(clip_line(r, Point::new(20, 20), Point::new(30, 30)).is_none());
        assert!(clip_line(r, Point::new(-1, 20), Point::new(11, 20)).is_none());
    }

    #[test]
    fn arrowed_line_draws_more_than_a_plain_line() {
        let mut a = canvas();
        line(&mut a, Point::new(5, 20), Point::new(35, 20), Scalar::all(255.0), 1).unwrap();
        let mut b = canvas();
        arrowed_line(&mut b, Point::new(5, 20), Point::new(35, 20), Scalar::all(255.0), 1, 0.3)
            .unwrap();
        assert!(lit(&b) > lit(&a), "the barbs must add pixels");
    }

    #[test]
    fn markers_draw_and_star_is_the_union_of_two_crosses() {
        let count = |k: MarkerType| {
            let mut m = canvas();
            draw_marker(&mut m, Point::new(20, 20), Scalar::all(255.0), k, 11, 1).unwrap();
            lit(&m)
        };
        let (c, t, s) = (count(MarkerType::Cross), count(MarkerType::TiltedCross), count(MarkerType::Star));
        assert!(c > 0 && t > 0);
        assert!(s > c && s > t, "star {s} should exceed cross {c} and tilted {t}");
        for k in [
            MarkerType::Diamond,
            MarkerType::Square,
            MarkerType::TriangleUp,
            MarkerType::TriangleDown,
        ] {
            assert!(count(k) > 0, "{k:?} drew nothing");
        }
    }

    #[test]
    fn drawing_works_on_a_colour_image() {
        let mut m = Mat::zeros(20, 20, CV_8UC3).unwrap();
        ellipse(&mut m, Point::new(10, 10), (5, 5), 0.0, 0.0, 360.0, Scalar::new(10.0, 20.0, 30.0, 0.0), -1)
            .unwrap();
        assert_eq!(m.at_f64(10, 10, 0).unwrap(), 10.0);
        assert_eq!(m.at_f64(10, 10, 1).unwrap(), 20.0);
        assert_eq!(m.at_f64(10, 10, 2).unwrap(), 30.0);
    }

    #[test]
    fn a_zero_size_marker_is_rejected() {
        let mut m = canvas();
        assert!(draw_marker(&mut m, Point::new(5, 5), Scalar::all(255.0), MarkerType::Cross, 0, 1).is_err());
    }
}
