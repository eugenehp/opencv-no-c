//! Edge detection (`modules/imgproc/src/deriv.cpp`, `canny.cpp`).

use cv_core::{cv_assert, cv_error, BorderType, Code, Depth, Mat, MatLike, MatLikeMut, Point, Result, Size};

use crate::filter::{filter_2d, get_deriv_kernels, sep_filter_2d, sobel};

/// `cv::Laplacian(src, dst, ddepth, ksize, scale, delta, borderType)`.
///
/// For `ksize` 1 and 3 upstream uses two hard-coded 3x3 kernels rather than
/// deriving them; both are reproduced exactly. Larger apertures sum the
/// separable second derivatives along each axis.
pub fn laplacian<S: MatLike + ?Sized>(
    src: &S,
    ddepth: Option<Depth>,
    ksize: i32,
    scale: f64,
    delta: f64,
    border: BorderType,
) -> Result<Mat> {
    cv_assert!(ksize > 0 && ksize % 2 == 1, "Laplacian ksize must be odd and positive");

    if ksize == 1 || ksize == 3 {
        // The two tabulated kernels from upstream's `K[2][9]`.
        let k: [f64; 9] = if ksize == 1 {
            [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0]
        } else {
            [2.0, 0.0, 2.0, 0.0, -8.0, 0.0, 2.0, 0.0, 2.0]
        };
        let scaled: Vec<f64> = k.iter().map(|v| v * scale).collect();
        return filter_2d(src, &scaled, Size::new(3, 3), ddepth, Point::new(-1, -1), delta, border);
    }

    // Laplacian = d2/dx2 + d2/dy2, each computed separably.
    let (kx2, ky2) = get_deriv_kernels(2, 0, ksize)?;
    let (kx0, ky0) = get_deriv_kernels(0, 2, ksize)?;
    let sx: Vec<f64> = kx2.iter().map(|v| v * scale).collect();
    let sy: Vec<f64> = kx0.iter().map(|v| v * scale).collect();

    let target = ddepth.unwrap_or(Depth::F64);
    let dxx = sep_filter_2d(src, &sx, &ky2, Some(Depth::F64), Point::new(-1, -1), 0.0, border)?;
    let dyy = sep_filter_2d(src, &sy, &ky0, Some(Depth::F64), Point::new(-1, -1), 0.0, border)?;
    let sum = cv_core::add(&dxx, &dyy)?;
    cv_core::convert_to(&sum, target, 1.0, delta)
}

/// `cv::Canny(image, edges, threshold1, threshold2, apertureSize, L2gradient)`.
///
/// `CV_8U` input, `CV_8UC1` output with values 0 or 255.
///
/// The non-maximum suppression here is upstream's integer formulation, not an
/// `atan2`-based one: the gradient direction is classified by comparing
/// `|dy| << 15` against `|dx| * 13573` (`tan(22.5°)` in Q15). Note the
/// asymmetric comparisons — horizontal and vertical directions use `>` on one
/// side and `>=` on the other, while the diagonal uses `>` on both — which
/// decides ties between equal-magnitude neighbours.
pub fn canny<S: MatLike + ?Sized>(
    src: &S,
    threshold1: f64,
    threshold2: f64,
    aperture_size: i32,
    l2_gradient: bool,
) -> Result<Mat> {
    if src.depth() != Depth::U8 {
        return cv_error!(Code::StsUnsupportedFormat, "Canny requires CV_8U, got {}", src.depth().name());
    }
    if aperture_size % 2 == 0 || !(3..=7).contains(&aperture_size) {
        return cv_error!(Code::StsBadFlag, "aperture size should be odd between 3 and 7");
    }
    cv_assert!(!src.is_empty(), "Canny requires a non-empty source");

    let (mut low_thresh, mut high_thresh) = (threshold1, threshold2);
    // Upstream rescales for the widest aperture, whose kernel gain is 16x.
    if aperture_size == 7 {
        low_thresh /= 16.0;
        high_thresh /= 16.0;
    }
    if low_thresh > high_thresh {
        core::mem::swap(&mut low_thresh, &mut high_thresh);
    }
    if l2_gradient {
        // Thresholds are compared against the *squared* L2 magnitude, so they
        // are squared here rather than the magnitude being square-rooted.
        low_thresh = low_thresh.min(32767.0);
        high_thresh = high_thresh.min(32767.0);
        if low_thresh > 0.0 {
            low_thresh *= low_thresh;
        }
        if high_thresh > 0.0 {
            high_thresh *= high_thresh;
        }
    }
    let low = cv_core::cv_floor(low_thresh) as i64;
    let high = cv_core::cv_floor(high_thresh) as i64;

    let (w, h) = (src.cols(), src.rows());
    // Gradients are CV_16S upstream; the derivative kernels can overflow that
    // for large apertures, so accumulate in i32 and clamp like saturate_cast.
    let dx = sobel(src, Some(Depth::S16), 1, 0, aperture_size, 1.0, 0.0, BorderType::Replicate)?;
    let dy = sobel(src, Some(Depth::S16), 0, 1, aperture_size, 1.0, 0.0, BorderType::Replicate)?;

    // Per-channel maximum magnitude, as upstream does for colour input.
    let cn = src.channels();
    let mut mag = vec![0i64; (w * h) as usize];
    let mut gx = vec![0i32; (w * h) as usize];
    let mut gy = vec![0i32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let mut best = -1i64;
            let (mut bx, mut by) = (0i32, 0i32);
            for c in 0..cn {
                let a = dx.at_f64(y, x, c)? as i32;
                let b = dy.at_f64(y, x, c)? as i32;
                let m = if l2_gradient {
                    a as i64 * a as i64 + b as i64 * b as i64
                } else {
                    (a.abs() + b.abs()) as i64
                };
                if m > best {
                    best = m;
                    bx = a;
                    by = b;
                }
            }
            let i = (y * w + x) as usize;
            mag[i] = best;
            gx[i] = bx;
            gy[i] = by;
        }
    }

    /// `tan(22.5°)` in Q15 — upstream's `TG22`.
    const TG22: i64 = 13573;

    // map: 0 = suppressed candidate, 1 = not an edge, 2 = confirmed edge.
    let mut map = vec![1u8; (w * h) as usize];
    let mut stack: Vec<usize> = Vec::new();
    let at = |y: i32, x: i32| (y * w + x) as usize;

    // Upstream runs the suppression over a magnitude buffer with a zeroed
    // one-pixel margin, so border pixels *are* candidates and simply compare
    // against 0 on the outside. Skipping them instead loses every edge that
    // touches the frame — measured against a real OpenCV build, that was 280
    // missing edge pixels on a 160x120 image, all of them on row/column 0 or
    // the last row/column.
    let mag_at = |y: i32, x: i32| -> i64 {
        if y < 0 || x < 0 || y >= h || x >= w {
            0
        } else {
            mag[at(y, x)]
        }
    };

    for y in 0..h {
        for x in 0..w {
            let i = at(y, x);
            let m = mag[i];
            if m <= low {
                continue;
            }
            let (xs, ys) = (gx[i], gy[i]);
            let ax = xs.unsigned_abs() as i64;
            let ay = (ys.unsigned_abs() as i64) << 15;
            let tg22x = ax * TG22;

            let keep = if ay < tg22x {
                // Predominantly horizontal gradient: compare left and right.
                m > mag_at(y, x - 1) && m >= mag_at(y, x + 1)
            } else {
                let tg67x = tg22x + (ax << 16);
                if ay > tg67x {
                    // Predominantly vertical: compare up and down.
                    m > mag_at(y - 1, x) && m >= mag_at(y + 1, x)
                } else {
                    // Diagonal: the sign product picks which diagonal.
                    let s = if (xs ^ ys) < 0 { -1 } else { 1 };
                    m > mag_at(y - 1, x - s) && m > mag_at(y + 1, x + s)
                }
            };
            if keep {
                if m > high {
                    map[i] = 2;
                    stack.push(i);
                } else {
                    map[i] = 0;
                }
            }
        }
    }

    // Hysteresis: grow confirmed edges into adjacent weak candidates.
    while let Some(i) = stack.pop() {
        let (y, x) = ((i as i32) / w, (i as i32) % w);
        for dy2 in -1..=1i32 {
            for dx2 in -1..=1i32 {
                if dx2 == 0 && dy2 == 0 {
                    continue;
                }
                let (ny, nx) = (y + dy2, x + dx2);
                if ny < 0 || nx < 0 || ny >= h || nx >= w {
                    continue;
                }
                let j = at(ny, nx);
                if map[j] == 0 {
                    map[j] = 2;
                    stack.push(j);
                }
            }
        }
    }

    let mut dst = Mat::zeros(h, w, cv_core::types::CV_8UC1)?;
    for y in 0..h {
        let row = dst.row_as_mut::<u8>(y)?;
        for x in 0..w {
            row[x as usize] = if map[at(y, x)] == 2 { 255 } else { 0 };
        }
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{count_non_zero, types::CV_8UC1, Scalar};

    fn u8m(rows: i32, cols: i32, v: &[u8]) -> Mat {
        Mat::from_vec(rows, cols, 1, v).unwrap()
    }

    /// A 16x16 image, dark on the left half and bright on the right.
    fn vertical_edge() -> Mat {
        let mut data = vec![0u8; 256];
        for y in 0..16 {
            for x in 8..16 {
                data[y * 16 + x] = 255;
            }
        }
        u8m(16, 16, &data)
    }

    #[test]
    fn laplacian_of_a_flat_image_is_zero() {
        let m = Mat::new_with(7, 7, CV_8UC1, Scalar::single(90.0)).unwrap();
        for k in [1, 3, 5] {
            let l = laplacian(&m, Some(Depth::F32), k, 1.0, 0.0, BorderType::Reflect101).unwrap();
            assert!(l.to_vec::<f32>().unwrap().iter().all(|&v| v.abs() < 1e-4), "ksize {k}");
        }
    }

    #[test]
    fn laplacian_ksize1_uses_the_four_neighbour_kernel() {
        // With the [[0,1,0],[1,-4,1],[0,1,0]] kernel, a lone bright pixel gives
        // -4 * value at its centre.
        let mut data = vec![0u8; 25];
        data[12] = 10;
        let m = u8m(5, 5, &data);
        let l = laplacian(&m, Some(Depth::F32), 1, 1.0, 0.0, BorderType::Constant).unwrap();
        assert_eq!(l.at::<f32>(2, 2).unwrap(), -40.0);
        assert_eq!(l.at::<f32>(2, 1).unwrap(), 10.0);
        assert_eq!(l.at::<f32>(1, 1).unwrap(), 0.0);
    }

    #[test]
    fn laplacian_ksize3_uses_the_eight_neighbour_kernel() {
        // [[2,0,2],[0,-8,0],[2,0,2]] gives -8 at the centre and 2 diagonally.
        let mut data = vec![0u8; 25];
        data[12] = 10;
        let m = u8m(5, 5, &data);
        let l = laplacian(&m, Some(Depth::F32), 3, 1.0, 0.0, BorderType::Constant).unwrap();
        assert_eq!(l.at::<f32>(2, 2).unwrap(), -80.0);
        assert_eq!(l.at::<f32>(1, 1).unwrap(), 20.0);
        assert_eq!(l.at::<f32>(2, 1).unwrap(), 0.0);
    }

    #[test]
    fn laplacian_applies_scale_and_delta() {
        let mut data = vec![0u8; 25];
        data[12] = 10;
        let m = u8m(5, 5, &data);
        let l = laplacian(&m, Some(Depth::F32), 1, 0.5, 100.0, BorderType::Constant).unwrap();
        assert_eq!(l.at::<f32>(2, 2).unwrap(), 100.0 - 20.0);
    }

    #[test]
    fn canny_finds_a_vertical_edge() {
        let m = vertical_edge();
        let e = canny(&m, 50.0, 150.0, 3, false).unwrap();
        assert_eq!(e.mat_type(), CV_8UC1);
        assert_eq!(e.size(), m.size());
        // Every set pixel is exactly 255.
        assert!(e.to_vec::<u8>().unwrap().iter().all(|&v| v == 0 || v == 255));
        // The edge runs down a single column of the interior.
        assert!(count_non_zero(&e).unwrap() >= 10, "found {}", count_non_zero(&e).unwrap());
        for y in 1..15 {
            assert_eq!(e.at::<u8>(y, 7).unwrap(), 255, "row {y}");
        }
    }

    #[test]
    fn canny_suppresses_non_maxima_to_a_thin_line() {
        let m = vertical_edge();
        let e = canny(&m, 50.0, 150.0, 3, false).unwrap();
        // Exactly one edge pixel per interior row: NMS thinned the response.
        for y in 1..15 {
            let row: Vec<u8> = (0..16).map(|x| e.at::<u8>(y, x).unwrap()).collect();
            let set = row.iter().filter(|&&v| v == 255).count();
            assert_eq!(set, 1, "row {y} had {set} edge pixels: {row:?}");
        }
    }

    #[test]
    fn canny_of_a_flat_image_is_empty() {
        let m = Mat::new_with(16, 16, CV_8UC1, Scalar::single(128.0)).unwrap();
        let e = canny(&m, 50.0, 150.0, 3, false).unwrap();
        assert_eq!(count_non_zero(&e).unwrap(), 0);
    }

    #[test]
    fn canny_thresholds_are_swapped_when_reversed() {
        let m = vertical_edge();
        let a = canny(&m, 50.0, 150.0, 3, false).unwrap();
        let b = canny(&m, 150.0, 50.0, 3, false).unwrap();
        assert_eq!(a.to_vec::<u8>().unwrap(), b.to_vec::<u8>().unwrap());
    }

    #[test]
    fn canny_high_threshold_suppresses_everything() {
        let m = vertical_edge();
        let e = canny(&m, 5000.0, 10000.0, 3, false).unwrap();
        assert_eq!(count_non_zero(&e).unwrap(), 0);
    }

    #[test]
    fn canny_l2_gradient_squares_the_thresholds() {
        // With L2gradient the magnitude is |g|^2, so a threshold of 300 must
        // still admit an edge whose L2 magnitude is ~1020. Squaring the
        // threshold (not rooting the magnitude) is what makes this work.
        let m = vertical_edge();
        let e = canny(&m, 300.0, 600.0, 3, true).unwrap();
        assert!(count_non_zero(&e).unwrap() > 0, "L2 gradient found no edges");
    }

    #[test]
    fn canny_hysteresis_links_weak_pixels_to_strong_ones() {
        // A gradient ramp produces a range of edge strengths; a wide
        // low/high band must connect them into one contour.
        let mut data = vec![0u8; 256];
        for y in 0..16 {
            for x in 8..16 {
                data[y * 16 + x] = (60 + y * 12).min(255) as u8;
            }
        }
        let m = u8m(16, 16, &data);
        let strict = canny(&m, 200.0, 400.0, 3, false).unwrap();
        let loose = canny(&m, 40.0, 400.0, 3, false).unwrap();
        // Lowering only the low threshold can only add pixels.
        assert!(count_non_zero(&loose).unwrap() >= count_non_zero(&strict).unwrap());
    }

    #[test]
    fn canny_rejects_bad_apertures_and_depths() {
        let m = vertical_edge();
        assert_eq!(canny(&m, 50.0, 150.0, 4, false).unwrap_err().code, Code::StsBadFlag);
        assert_eq!(canny(&m, 50.0, 150.0, 9, false).unwrap_err().code, Code::StsBadFlag);
        let f = Mat::zeros(4, 4, cv_core::types::CV_32FC1).unwrap();
        assert_eq!(canny(&f, 50.0, 150.0, 3, false).unwrap_err().code, Code::StsUnsupportedFormat);
    }

    #[test]
    fn canny_finds_a_horizontal_edge_too() {
        let mut data = vec![0u8; 256];
        for y in 8..16 {
            for x in 0..16 {
                data[y * 16 + x] = 255;
            }
        }
        let m = u8m(16, 16, &data);
        let e = canny(&m, 50.0, 150.0, 3, false).unwrap();
        for x in 1..15 {
            assert_eq!(e.at::<u8>(7, x).unwrap(), 255, "col {x}");
        }
    }
}
