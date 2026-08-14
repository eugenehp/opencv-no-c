//! Drawing primitives (`modules/imgproc/src/drawing.cpp`).
//!
//! Only the 8-connected (`LINE_8`) variants are ported; upstream's anti-aliased
//! `LINE_AA` path uses a separate sub-pixel rasteriser that is not reproduced
//! here. Shapes are clipped to the image rather than erroring on out-of-range
//! coordinates, matching upstream.

use cv_core::{cv_assert, MatLikeMut, Point, Rect, Result, Scalar};

/// Write `color` into one pixel, saturating per channel, ignoring out-of-range
/// coordinates the way upstream's clipped rasterisers do.
fn plot<M: MatLikeMut + ?Sized>(img: &mut M, x: i32, y: i32, color: Scalar) -> Result<()> {
    if x < 0 || y < 0 || x >= img.cols() || y >= img.rows() {
        return Ok(());
    }
    let cn = img.channels();
    for c in 0..cn {
        set_one(img, y, x, c, color.0[(c as usize).min(3)])?;
    }
    Ok(())
}

/// Depth-generic single-channel store.
fn set_one<M: MatLikeMut + ?Sized>(m: &mut M, y: i32, x: i32, c: i32, v: f64) -> Result<()> {
    use cv_core::saturate::Elem;
    use cv_core::Depth;
    let i = c as usize;
    match m.depth() {
        Depth::U8 => m.pixel_mut::<u8>(y, x)?[i] = u8::from_f64(v),
        Depth::S8 => m.pixel_mut::<i8>(y, x)?[i] = i8::from_f64(v),
        Depth::U16 => m.pixel_mut::<u16>(y, x)?[i] = u16::from_f64(v),
        Depth::S16 => m.pixel_mut::<i16>(y, x)?[i] = i16::from_f64(v),
        Depth::S32 => m.pixel_mut::<i32>(y, x)?[i] = i32::from_f64(v),
        Depth::F32 => m.pixel_mut::<f32>(y, x)?[i] = v as f32,
        Depth::F64 => m.pixel_mut::<f64>(y, x)?[i] = v,
        Depth::F16 => {
            return cv_core::cv_error!(cv_core::Code::StsNotImplemented, "CV_16F is not ported yet")
        }
    }
    Ok(())
}

/// `cv::line(img, pt1, pt2, color, thickness)` — 8-connected Bresenham (`LINE_8`).
///
/// `thickness` above 1 is drawn by stamping a square brush, which matches
/// upstream closely for thin lines but not its exact polygon fill for thick
/// ones.
pub fn line<M: MatLikeMut + ?Sized>(
    img: &mut M,
    pt1: Point,
    pt2: Point,
    color: Scalar,
    thickness: i32,
) -> Result<()> {
    cv_assert!(thickness > 0, "thickness must be positive, got {thickness}");

    let (mut x, mut y) = (pt1.x, pt1.y);
    let dx = (pt2.x - pt1.x).abs();
    let dy = -(pt2.y - pt1.y).abs();
    let sx = if pt1.x < pt2.x { 1 } else { -1 };
    let sy = if pt1.y < pt2.y { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        stamp(img, x, y, color, thickness)?;
        if x == pt2.x && y == pt2.y {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
    Ok(())
}

/// Draw a `thickness`-wide square brush centred on `(x, y)`.
fn stamp<M: MatLikeMut + ?Sized>(
    img: &mut M,
    x: i32,
    y: i32,
    color: Scalar,
    thickness: i32,
) -> Result<()> {
    if thickness <= 1 {
        return plot(img, x, y, color);
    }
    let r = thickness / 2;
    for oy in -r..=r {
        for ox in -r..=r {
            plot(img, x + ox, y + oy, color)?;
        }
    }
    Ok(())
}

/// `cv::rectangle(img, rec, color, thickness)`.
///
/// A negative `thickness` (upstream's `FILLED`) fills the rectangle.
pub fn rectangle<M: MatLikeMut + ?Sized>(
    img: &mut M,
    rect: Rect,
    color: Scalar,
    thickness: i32,
) -> Result<()> {
    if rect.empty() {
        return Ok(());
    }
    let (x0, y0) = (rect.x, rect.y);
    let (x1, y1) = (rect.x + rect.width - 1, rect.y + rect.height - 1);

    if thickness < 0 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                plot(img, x, y, color)?;
            }
        }
        return Ok(());
    }
    line(img, Point::new(x0, y0), Point::new(x1, y0), color, thickness)?;
    line(img, Point::new(x1, y0), Point::new(x1, y1), color, thickness)?;
    line(img, Point::new(x1, y1), Point::new(x0, y1), color, thickness)?;
    line(img, Point::new(x0, y1), Point::new(x0, y0), color, thickness)
}

/// `cv::circle(img, center, radius, color, thickness)`.
///
/// A negative `thickness` fills the disc. The outline uses the midpoint
/// algorithm, as upstream's `Circle` does.
pub fn circle<M: MatLikeMut + ?Sized>(
    img: &mut M,
    center: Point,
    radius: i32,
    color: Scalar,
    thickness: i32,
) -> Result<()> {
    cv_assert!(radius >= 0, "radius must be non-negative, got {radius}");
    if radius == 0 {
        return plot(img, center.x, center.y, color);
    }

    if thickness < 0 {
        // Filled: scan each row of the bounding box against the radius.
        let r2 = radius * radius;
        for dy in -radius..=radius {
            let dx = ((r2 - dy * dy) as f64).sqrt() as i32;
            for x in (center.x - dx)..=(center.x + dx) {
                plot(img, x, center.y + dy, color)?;
            }
        }
        return Ok(());
    }

    let (mut x, mut y) = (0i32, radius);
    let mut err = 3 - 2 * radius;
    while x <= y {
        for (px, py) in [
            (x, y),
            (y, x),
            (-x, y),
            (-y, x),
            (x, -y),
            (y, -x),
            (-x, -y),
            (-y, -x),
        ] {
            stamp(img, center.x + px, center.y + py, color, thickness)?;
        }
        if err > 0 {
            y -= 1;
            err += 4 * (x - y) + 10;
        } else {
            err += 4 * x + 6;
        }
        x += 1;
    }
    Ok(())
}

/// `cv::polylines(img, pts, isClosed, color, thickness)`.
pub fn polylines<M: MatLikeMut + ?Sized>(
    img: &mut M,
    pts: &[Point],
    is_closed: bool,
    color: Scalar,
    thickness: i32,
) -> Result<()> {
    if pts.len() < 2 {
        return Ok(());
    }
    for w in pts.windows(2) {
        line(img, w[0], w[1], color, thickness)?;
    }
    if is_closed {
        line(img, pts[pts.len() - 1], pts[0], color, thickness)?;
    }
    Ok(())
}

/// `cv::fillPoly(img, pts, color)` — even-odd scanline fill.
pub fn fill_poly<M: MatLikeMut + ?Sized>(
    img: &mut M,
    pts: &[Point],
    color: Scalar,
) -> Result<()> {
    if pts.len() < 3 {
        return Ok(());
    }
    let min_y = pts.iter().map(|p| p.y).min().unwrap().max(0);
    let max_y = pts.iter().map(|p| p.y).max().unwrap().min(img.rows() - 1);

    let mut crossings: Vec<i32> = Vec::new();
    for y in min_y..=max_y {
        crossings.clear();
        for i in 0..pts.len() {
            let a = pts[i];
            let b = pts[(i + 1) % pts.len()];
            // Half-open in y so shared vertices are counted once.
            let (lo, hi) = if a.y < b.y { (a, b) } else { (b, a) };
            if y >= lo.y && y < hi.y {
                let t = (y - lo.y) as f64 / (hi.y - lo.y) as f64;
                crossings.push(cv_core::cv_round(lo.x as f64 + t * (hi.x - lo.x) as f64));
            }
        }
        crossings.sort_unstable();
        for pair in crossings.chunks_exact(2) {
            for x in pair[0]..=pair[1] {
                plot(img, x, y, color)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{count_non_zero, types::CV_8UC1, types::CV_8UC3, Mat, MatLike};

    fn blank(n: i32) -> Mat {
        Mat::zeros(n, n, CV_8UC1).unwrap()
    }

    fn white() -> Scalar {
        Scalar::single(255.0)
    }

    #[test]
    fn horizontal_line_sets_exactly_its_span() {
        let mut m = blank(9);
        line(&mut m, Point::new(2, 4), Point::new(6, 4), white(), 1).unwrap();
        assert_eq!(count_non_zero(&m).unwrap(), 5);
        for x in 2..=6 {
            assert_eq!(m.at::<u8>(4, x).unwrap(), 255);
        }
    }

    #[test]
    fn vertical_line_sets_exactly_its_span() {
        let mut m = blank(9);
        line(&mut m, Point::new(3, 1), Point::new(3, 7), white(), 1).unwrap();
        assert_eq!(count_non_zero(&m).unwrap(), 7);
    }

    #[test]
    fn diagonal_line_is_eight_connected() {
        let mut m = blank(8);
        line(&mut m, Point::new(0, 0), Point::new(7, 7), white(), 1).unwrap();
        assert_eq!(count_non_zero(&m).unwrap(), 8);
        for i in 0..8 {
            assert_eq!(m.at::<u8>(i, i).unwrap(), 255);
        }
    }

    #[test]
    fn line_is_direction_independent() {
        let mut a = blank(9);
        let mut b = blank(9);
        line(&mut a, Point::new(1, 2), Point::new(7, 5), white(), 1).unwrap();
        line(&mut b, Point::new(7, 5), Point::new(1, 2), white(), 1).unwrap();
        // Bresenham can differ on ties, but the endpoints and count must match.
        assert_eq!(count_non_zero(&a).unwrap(), count_non_zero(&b).unwrap());
        assert_eq!(a.at::<u8>(2, 1).unwrap(), 255);
        assert_eq!(b.at::<u8>(2, 1).unwrap(), 255);
    }

    #[test]
    fn line_clips_to_the_image() {
        let mut m = blank(5);
        // Entirely outside; must not panic and must leave the image blank.
        line(&mut m, Point::new(-20, -20), Point::new(-10, -10), white(), 1).unwrap();
        assert_eq!(count_non_zero(&m).unwrap(), 0);
        // Partially outside.
        line(&mut m, Point::new(-3, 2), Point::new(3, 2), white(), 1).unwrap();
        assert_eq!(count_non_zero(&m).unwrap(), 4);
    }

    #[test]
    fn single_point_line_sets_one_pixel() {
        let mut m = blank(5);
        line(&mut m, Point::new(2, 2), Point::new(2, 2), white(), 1).unwrap();
        assert_eq!(count_non_zero(&m).unwrap(), 1);
    }

    #[test]
    fn rectangle_outline_has_a_hollow_interior() {
        let mut m = blank(9);
        rectangle(&mut m, Rect::new(2, 2, 5, 5), white(), 1).unwrap();
        // Perimeter of a 5x5 box is 16 pixels.
        assert_eq!(count_non_zero(&m).unwrap(), 16);
        assert_eq!(m.at::<u8>(4, 4).unwrap(), 0);
        assert_eq!(m.at::<u8>(2, 2).unwrap(), 255);
        assert_eq!(m.at::<u8>(6, 6).unwrap(), 255);
    }

    #[test]
    fn filled_rectangle_covers_its_area() {
        let mut m = blank(9);
        rectangle(&mut m, Rect::new(2, 3, 4, 2), white(), -1).unwrap();
        assert_eq!(count_non_zero(&m).unwrap(), 8);
        assert_eq!(m.at::<u8>(3, 2).unwrap(), 255);
        assert_eq!(m.at::<u8>(4, 5).unwrap(), 255);
        assert_eq!(m.at::<u8>(5, 5).unwrap(), 0);
    }

    #[test]
    fn empty_rectangle_draws_nothing() {
        let mut m = blank(5);
        rectangle(&mut m, Rect::new(1, 1, 0, 3), white(), -1).unwrap();
        assert_eq!(count_non_zero(&m).unwrap(), 0);
    }

    #[test]
    fn filled_circle_is_roughly_pi_r_squared() {
        let mut m = blank(41);
        circle(&mut m, Point::new(20, 20), 10, white(), -1).unwrap();
        let n = count_non_zero(&m).unwrap() as f64;
        let expected = core::f64::consts::PI * 100.0;
        assert!((n - expected).abs() / expected < 0.1, "area {n} vs {expected}");
    }

    #[test]
    fn circle_outline_is_hollow_and_symmetric() {
        let mut m = blank(41);
        circle(&mut m, Point::new(20, 20), 10, white(), 1).unwrap();
        assert_eq!(m.at::<u8>(20, 20).unwrap(), 0, "centre should be hollow");
        // Symmetric about both axes.
        assert_eq!(m.at::<u8>(10, 20).unwrap(), m.at::<u8>(30, 20).unwrap());
        assert_eq!(m.at::<u8>(20, 10).unwrap(), m.at::<u8>(20, 30).unwrap());
        assert_eq!(m.at::<u8>(10, 20).unwrap(), 255);
    }

    #[test]
    fn zero_radius_circle_is_one_pixel() {
        let mut m = blank(5);
        circle(&mut m, Point::new(2, 2), 0, white(), -1).unwrap();
        assert_eq!(count_non_zero(&m).unwrap(), 1);
    }

    #[test]
    fn circle_clips_at_the_border() {
        let mut m = blank(10);
        circle(&mut m, Point::new(0, 0), 5, white(), -1).unwrap();
        // A quarter disc, no panic.
        assert!(count_non_zero(&m).unwrap() > 0);
        assert_eq!(m.at::<u8>(0, 0).unwrap(), 255);
    }

    #[test]
    fn polylines_closed_adds_the_final_segment() {
        let pts = [Point::new(1, 1), Point::new(5, 1), Point::new(5, 5)];
        let mut open = blank(9);
        let mut closed = blank(9);
        polylines(&mut open, &pts, false, white(), 1).unwrap();
        polylines(&mut closed, &pts, true, white(), 1).unwrap();
        assert!(count_non_zero(&closed).unwrap() > count_non_zero(&open).unwrap());
        assert_eq!(closed.at::<u8>(3, 3).unwrap(), 255);
        assert_eq!(open.at::<u8>(3, 3).unwrap(), 0);
    }

    #[test]
    fn fill_poly_fills_a_triangle() {
        let mut m = blank(11);
        let pts = [Point::new(1, 1), Point::new(9, 1), Point::new(1, 9)];
        fill_poly(&mut m, &pts, white()).unwrap();
        // Interior point is set, exterior is not.
        assert_eq!(m.at::<u8>(2, 2).unwrap(), 255);
        assert_eq!(m.at::<u8>(8, 8).unwrap(), 0);
        assert!(count_non_zero(&m).unwrap() > 20);
    }

    #[test]
    fn fill_poly_fills_a_rectangle_exactly() {
        let mut m = blank(9);
        let pts = [
            Point::new(2, 2),
            Point::new(5, 2),
            Point::new(5, 6),
            Point::new(2, 6),
        ];
        fill_poly(&mut m, &pts, white()).unwrap();
        // Rows 2..5 inclusive of the half-open span, columns 2..=5.
        for y in 2..6 {
            for x in 2..=5 {
                assert_eq!(m.at::<u8>(y, x).unwrap(), 255, "at {y},{x}");
            }
        }
    }

    #[test]
    fn degenerate_polygons_draw_nothing() {
        let mut m = blank(5);
        fill_poly(&mut m, &[Point::new(1, 1), Point::new(3, 3)], white()).unwrap();
        assert_eq!(count_non_zero(&m).unwrap(), 0);
    }

    #[test]
    fn colour_is_written_per_channel() {
        let mut m = Mat::zeros(5, 5, CV_8UC3).unwrap();
        line(&mut m, Point::new(0, 2), Point::new(4, 2), Scalar::new(10.0, 20.0, 30.0, 0.0), 1)
            .unwrap();
        assert_eq!(m.pixel::<u8>(2, 0).unwrap(), &[10, 20, 30]);
        assert_eq!(m.pixel::<u8>(0, 0).unwrap(), &[0, 0, 0]);
    }

    #[test]
    fn colour_saturates_per_channel() {
        let mut m = Mat::zeros(3, 3, CV_8UC3).unwrap();
        rectangle(&mut m, Rect::new(0, 0, 3, 3), Scalar::new(-5.0, 300.0, 128.0, 0.0), -1).unwrap();
        assert_eq!(m.pixel::<u8>(1, 1).unwrap(), &[0, 255, 128]);
    }

    #[test]
    fn thick_line_covers_more_than_a_thin_one() {
        let mut thin = blank(15);
        let mut thick = blank(15);
        line(&mut thin, Point::new(2, 7), Point::new(12, 7), white(), 1).unwrap();
        line(&mut thick, Point::new(2, 7), Point::new(12, 7), white(), 3).unwrap();
        assert!(count_non_zero(&thick).unwrap() > count_non_zero(&thin).unwrap());
        assert_eq!(thick.at::<u8>(6, 7).unwrap(), 255);
    }

    #[test]
    fn drawing_works_through_a_mutable_roi() {
        let mut m = Mat::zeros(10, 10, CV_8UC1).unwrap();
        {
            let mut roi = m.roi_mut(Rect::new(2, 2, 5, 5)).unwrap();
            rectangle(&mut roi, Rect::new(0, 0, 5, 5), white(), -1).unwrap();
        }
        assert_eq!(count_non_zero(&m).unwrap(), 25);
        assert_eq!(m.at::<u8>(2, 2).unwrap(), 255);
        assert_eq!(m.at::<u8>(1, 1).unwrap(), 0);
        assert_eq!(m.at::<u8>(7, 7).unwrap(), 0);
    }
}
