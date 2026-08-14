//! Morphological operations (`modules/imgproc/src/morph.dispatch.cpp`).

use cv_core::{
    cv_assert, cv_error, BorderType, Code, Mat, MatLike, MatLikeMut, Point, Result, Scalar, Size,
};


/// `cv::MorphShapes`, with upstream's numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MorphShape {
    /// `MORPH_RECT` — every element set.
    Rect = 0,
    /// `MORPH_CROSS` — the anchor row and column.
    Cross = 1,
    /// `MORPH_ELLIPSE` — a filled ellipse inscribed in the box.
    Ellipse = 2,
    /// `MORPH_DIAMOND` — a filled diamond (L1 ball).
    Diamond = 3,
}

/// `cv::MorphTypes`, with upstream's numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MorphType {
    /// `MORPH_ERODE` — local minimum.
    Erode = 0,
    /// `MORPH_DILATE` — local maximum.
    Dilate = 1,
    /// `MORPH_OPEN` — erode then dilate.
    Open = 2,
    /// `MORPH_CLOSE` — dilate then erode.
    Close = 3,
    /// `MORPH_GRADIENT` — dilate minus erode.
    Gradient = 4,
    /// `MORPH_TOPHAT` — source minus opening.
    TopHat = 5,
    /// `MORPH_BLACKHAT` — closing minus source.
    BlackHat = 6,
}

/// `cv::getStructuringElement(shape, ksize, anchor)`.
///
/// A direct transcription, including the ellipse's per-row half-width
/// `dx = saturate_cast<int>(c * sqrt((r*r - dy*dy) * inv_r2))`, which is not
/// the same as testing `(x/c)^2 + (y/r)^2 <= 1` pointwise.
pub fn get_structuring_element(shape: MorphShape, ksize: Size, anchor: Point) -> Result<Mat> {
    cv_assert!(ksize.width > 0 && ksize.height > 0, "structuring element size must be positive");
    let anchor = Point::new(
        if anchor.x < 0 { ksize.width / 2 } else { anchor.x },
        if anchor.y < 0 { ksize.height / 2 } else { anchor.y },
    );
    // A 1x1 element is a rectangle whatever shape was asked for.
    let shape = if ksize.width == 1 && ksize.height == 1 { MorphShape::Rect } else { shape };

    let r = ksize.height / 2;
    let c = ksize.width / 2;
    let inv_r2 = if r != 0 { 1.0 / (r as f64 * r as f64) } else { 0.0 };

    let mut elem = Mat::zeros(ksize.height, ksize.width, cv_core::types::CV_8UC1)?;
    for i in 0..ksize.height {
        let (j1, j2) = match shape {
            MorphShape::Rect => (0, ksize.width),
            MorphShape::Cross => {
                if i == anchor.y {
                    (0, ksize.width)
                } else {
                    (anchor.x, anchor.x + 1)
                }
            }
            MorphShape::Diamond => {
                let dy = (i - r).abs();
                if dy <= r {
                    let dx = r - dy;
                    ((c - dx).max(0), (c + dx + 1).min(ksize.width))
                } else {
                    (0, 0)
                }
            }
            MorphShape::Ellipse => {
                let dy = i - r;
                if dy.abs() <= r {
                    let inner = (r * r - dy * dy) as f64 * inv_r2;
                    let dx = cv_core::cv_round(c as f64 * inner.sqrt());
                    ((c - dx).max(0), (c + dx + 1).min(ksize.width))
                } else {
                    (0, 0)
                }
            }
        };
        let row = elem.row_as_mut::<u8>(i)?;
        for j in j1..j2 {
            row[j as usize] = 1;
        }
    }
    Ok(elem)
}

/// Run a min or max over the structuring element's support.
fn morph_pass<S: MatLike + ?Sized, K: MatLike + ?Sized>(
    src: &S,
    kernel: &K,
    anchor: Point,
    dilate: bool,
    border: BorderType,
    border_value: Option<Scalar>,
) -> Result<Mat> {
    cv_assert!(!src.is_empty(), "morphology requires a non-empty source");
    cv_assert!(!kernel.is_empty(), "structuring element must be non-empty");
    let (kw, kh) = (kernel.cols(), kernel.rows());
    let a = Point::new(
        if anchor.x < 0 { kw / 2 } else { anchor.x },
        if anchor.y < 0 { kh / 2 } else { anchor.y },
    );

    let cn = src.channels() as usize;
    let (rows, cols) = (src.rows(), src.cols());

    // Under a constant border the outside must act as the identity for the
    // reduction: the largest representable value for a min (erode), the
    // smallest for a max (dilate). That is what upstream's
    // `morphologyDefaultBorderValue` encodes, and it lets the border be baked
    // into a padded copy instead of tested per tap.
    let depth = src.depth();
    let identity = border_value.map(|s| s.0[0]).unwrap_or(if dilate {
        depth.min_value().max(-1e300)
    } else {
        depth.max_value().min(1e300)
    });

    let padded = cv_core::copy_make_border(
        src,
        a.y,
        kh - 1 - a.y,
        a.x,
        kw - 1 - a.x,
        border,
        Scalar::all(identity),
    )?;
    let pw = padded.cols() as usize;
    let buf = cv_core::to_f64(&padded)?;

    // Flat offsets of the set elements, computed once.
    let mut offsets: Vec<usize> = Vec::new();
    for ky in 0..kh as usize {
        for kx in 0..kw as usize {
            if kernel.at_f64(ky as i32, kx as i32, 0)? != 0.0 {
                offsets.push((ky * pw + kx) * cn);
            }
        }
    }
    if offsets.is_empty() {
        return cv_error!(Code::StsBadArg, "structuring element is all zeros");
    }

    let width = cols as usize * cn;
    let mut out = vec![0.0f64; rows as usize * width];
    for y in 0..rows as usize {
        let row_base = y * pw * cn;
        let dst_row = &mut out[y * width..(y + 1) * width];
        for (i, o) in dst_row.iter_mut().enumerate() {
            let base = row_base + i;
            let mut acc = buf[base + offsets[0]];
            for &off in &offsets[1..] {
                let v = buf[base + off];
                acc = if dilate { acc.max(v) } else { acc.min(v) };
            }
            *o = acc;
        }
    }

    cv_core::mat_from_f64(rows, cols, src.mat_type(), &out)
}

/// `cv::erode(src, dst, kernel, anchor, iterations, borderType, borderValue)`.
pub fn erode<S: MatLike + ?Sized, K: MatLike + ?Sized>(
    src: &S,
    kernel: &K,
    anchor: Point,
    iterations: i32,
    border: BorderType,
) -> Result<Mat> {
    cv_assert!(iterations >= 0, "iterations must be non-negative");
    let mut cur = Mat::copy_from(src)?;
    for _ in 0..iterations {
        cur = morph_pass(&cur, kernel, anchor, false, border, None)?;
    }
    Ok(cur)
}

/// `cv::dilate(src, dst, kernel, anchor, iterations, borderType, borderValue)`.
pub fn dilate<S: MatLike + ?Sized, K: MatLike + ?Sized>(
    src: &S,
    kernel: &K,
    anchor: Point,
    iterations: i32,
    border: BorderType,
) -> Result<Mat> {
    cv_assert!(iterations >= 0, "iterations must be non-negative");
    let mut cur = Mat::copy_from(src)?;
    for _ in 0..iterations {
        cur = morph_pass(&cur, kernel, anchor, true, border, None)?;
    }
    Ok(cur)
}

/// `cv::morphologyEx(src, dst, op, kernel, anchor, iterations, borderType, borderValue)`.
pub fn morphology_ex<S: MatLike + ?Sized, K: MatLike + ?Sized>(
    src: &S,
    op: MorphType,
    kernel: &K,
    anchor: Point,
    iterations: i32,
    border: BorderType,
) -> Result<Mat> {
    use cv_core::subtract;
    match op {
        MorphType::Erode => erode(src, kernel, anchor, iterations, border),
        MorphType::Dilate => dilate(src, kernel, anchor, iterations, border),
        MorphType::Open => {
            let e = erode(src, kernel, anchor, iterations, border)?;
            dilate(&e, kernel, anchor, iterations, border)
        }
        MorphType::Close => {
            let d = dilate(src, kernel, anchor, iterations, border)?;
            erode(&d, kernel, anchor, iterations, border)
        }
        MorphType::Gradient => {
            let d = dilate(src, kernel, anchor, iterations, border)?;
            let e = erode(src, kernel, anchor, iterations, border)?;
            subtract(&d, &e)
        }
        MorphType::TopHat => {
            let opened = morphology_ex(src, MorphType::Open, kernel, anchor, iterations, border)?;
            subtract(src, &opened)
        }
        MorphType::BlackHat => {
            let closed = morphology_ex(src, MorphType::Close, kernel, anchor, iterations, border)?;
            subtract(&closed, src)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::CV_8UC1;

    fn u8m(rows: i32, cols: i32, v: &[u8]) -> Mat {
        Mat::from_vec(rows, cols, 1, v).unwrap()
    }

    fn rect3() -> Mat {
        get_structuring_element(MorphShape::Rect, Size::new(3, 3), Point::new(-1, -1)).unwrap()
    }

    #[test]
    fn shape_and_type_values_match_opencv() {
        assert_eq!(MorphShape::Rect as i32, 0);
        assert_eq!(MorphShape::Cross as i32, 1);
        assert_eq!(MorphShape::Ellipse as i32, 2);
        assert_eq!(MorphType::Erode as i32, 0);
        assert_eq!(MorphType::BlackHat as i32, 6);
    }

    #[test]
    fn rect_element_is_all_ones() {
        let e = rect3();
        assert_eq!(e.to_vec::<u8>().unwrap(), vec![1; 9]);
    }

    #[test]
    fn cross_element_has_an_anchor_row_and_column() {
        let e = get_structuring_element(MorphShape::Cross, Size::new(3, 3), Point::new(-1, -1))
            .unwrap();
        assert_eq!(e.to_vec::<u8>().unwrap(), vec![0, 1, 0, 1, 1, 1, 0, 1, 0]);
    }

    #[test]
    fn ellipse_5x5_matches_upstream() {
        // Upstream's per-row half-width computation gives a full 5-wide middle
        // row and 3-wide neighbours, not a pointwise ellipse test.
        let e = get_structuring_element(MorphShape::Ellipse, Size::new(5, 5), Point::new(-1, -1))
            .unwrap();
        assert_eq!(
            e.to_vec::<u8>().unwrap(),
            vec![
                0, 0, 1, 0, 0, //
                1, 1, 1, 1, 1, //
                1, 1, 1, 1, 1, //
                1, 1, 1, 1, 1, //
                0, 0, 1, 0, 0, //
            ]
        );
    }

    #[test]
    fn diamond_element_is_an_l1_ball() {
        let e = get_structuring_element(MorphShape::Diamond, Size::new(5, 5), Point::new(-1, -1))
            .unwrap();
        assert_eq!(
            e.to_vec::<u8>().unwrap(),
            vec![
                0, 0, 1, 0, 0, //
                0, 1, 1, 1, 0, //
                1, 1, 1, 1, 1, //
                0, 1, 1, 1, 0, //
                0, 0, 1, 0, 0, //
            ]
        );
    }

    #[test]
    fn one_by_one_is_always_rectangular() {
        let e = get_structuring_element(MorphShape::Ellipse, Size::new(1, 1), Point::new(-1, -1))
            .unwrap();
        assert_eq!(e.to_vec::<u8>().unwrap(), vec![1]);
    }

    #[test]
    fn erode_shrinks_a_bright_blob() {
        let m = u8m(5, 5, &[
            0, 0, 0, 0, 0, //
            0, 255, 255, 255, 0, //
            0, 255, 255, 255, 0, //
            0, 255, 255, 255, 0, //
            0, 0, 0, 0, 0, //
        ]);
        let e = erode(&m, &rect3(), Point::new(-1, -1), 1, BorderType::Replicate).unwrap();
        // Only the centre survives a 3x3 erosion of a 3x3 blob.
        assert_eq!(e.at::<u8>(2, 2).unwrap(), 255);
        assert_eq!(e.at::<u8>(1, 1).unwrap(), 0);
        assert_eq!(cv_core::count_non_zero(&e).unwrap(), 1);
    }

    #[test]
    fn dilate_grows_a_single_pixel() {
        let mut data = vec![0u8; 25];
        data[12] = 255;
        let m = u8m(5, 5, &data);
        let d = dilate(&m, &rect3(), Point::new(-1, -1), 1, BorderType::Replicate).unwrap();
        assert_eq!(cv_core::count_non_zero(&d).unwrap(), 9);
        assert_eq!(d.at::<u8>(1, 1).unwrap(), 255);
        assert_eq!(d.at::<u8>(0, 0).unwrap(), 0);
    }

    #[test]
    fn erode_and_dilate_are_dual_on_a_flat_image() {
        let m = Mat::new_with(5, 5, CV_8UC1, Scalar::single(100.0)).unwrap();
        for out in [
            erode(&m, &rect3(), Point::new(-1, -1), 1, BorderType::Replicate).unwrap(),
            dilate(&m, &rect3(), Point::new(-1, -1), 1, BorderType::Replicate).unwrap(),
        ] {
            assert!(out.to_vec::<u8>().unwrap().iter().all(|&v| v == 100));
        }
    }

    #[test]
    fn zero_iterations_is_a_copy() {
        let m = u8m(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let e = erode(&m, &rect3(), Point::new(-1, -1), 0, BorderType::Replicate).unwrap();
        assert_eq!(e.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn opening_removes_small_specks() {
        let mut data = vec![0u8; 49];
        data[3 * 7 + 3] = 255; // a lone pixel
        let m = u8m(7, 7, &data);
        let opened =
            morphology_ex(&m, MorphType::Open, &rect3(), Point::new(-1, -1), 1, BorderType::Replicate)
                .unwrap();
        assert_eq!(cv_core::count_non_zero(&opened).unwrap(), 0);
    }

    #[test]
    fn closing_fills_small_holes() {
        let mut data = vec![255u8; 49];
        data[3 * 7 + 3] = 0; // a lone hole
        let m = u8m(7, 7, &data);
        let closed = morphology_ex(
            &m,
            MorphType::Close,
            &rect3(),
            Point::new(-1, -1),
            1,
            BorderType::Replicate,
        )
        .unwrap();
        assert_eq!(cv_core::count_non_zero(&closed).unwrap(), 49);
    }

    #[test]
    fn gradient_outlines_an_edge() {
        let mut data = vec![0u8; 25];
        data[12] = 255;
        let m = u8m(5, 5, &data);
        let g = morphology_ex(
            &m,
            MorphType::Gradient,
            &rect3(),
            Point::new(-1, -1),
            1,
            BorderType::Replicate,
        )
        .unwrap();
        // dilate(9 pixels) - erode(0 pixels) = a 3x3 ring plus centre.
        assert_eq!(cv_core::count_non_zero(&g).unwrap(), 9);
    }

    #[test]
    fn tophat_recovers_a_bright_speck() {
        // Opening deletes a lone bright pixel, so src - opening restores it.
        let mut data = vec![0u8; 49];
        data[3 * 7 + 3] = 200;
        let m = u8m(7, 7, &data);
        let t =
            morphology_ex(&m, MorphType::TopHat, &rect3(), Point::new(-1, -1), 1, BorderType::Replicate)
                .unwrap();
        assert_eq!(t.at::<u8>(3, 3).unwrap(), 200);
        assert_eq!(cv_core::count_non_zero(&t).unwrap(), 1);
    }

    #[test]
    fn blackhat_recovers_a_dark_hole() {
        // Closing fills a lone dark pixel, so closing - src restores it.
        let mut data = vec![255u8; 49];
        data[3 * 7 + 3] = 55;
        let m = u8m(7, 7, &data);
        let b = morphology_ex(
            &m,
            MorphType::BlackHat,
            &rect3(),
            Point::new(-1, -1),
            1,
            BorderType::Replicate,
        )
        .unwrap();
        assert_eq!(b.at::<u8>(3, 3).unwrap(), 200);
        assert_eq!(cv_core::count_non_zero(&b).unwrap(), 1);
    }

    #[test]
    fn multiple_iterations_compound() {
        let mut data = vec![0u8; 81];
        for y in 2..7 {
            for x in 2..7 {
                data[y * 9 + x] = 255;
            }
        }
        let m = u8m(9, 9, &data);
        let once = erode(&m, &rect3(), Point::new(-1, -1), 1, BorderType::Replicate).unwrap();
        let twice = erode(&m, &rect3(), Point::new(-1, -1), 2, BorderType::Replicate).unwrap();
        assert_eq!(cv_core::count_non_zero(&once).unwrap(), 9); // 3x3
        assert_eq!(cv_core::count_non_zero(&twice).unwrap(), 1); // 1x1
    }

    #[test]
    fn an_empty_element_is_rejected() {
        let zeros = Mat::zeros(3, 3, CV_8UC1).unwrap();
        let m = u8m(3, 3, &[0; 9]);
        assert!(erode(&m, &zeros, Point::new(-1, -1), 1, BorderType::Replicate).is_err());
    }
}
