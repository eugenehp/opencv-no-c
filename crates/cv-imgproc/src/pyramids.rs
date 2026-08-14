//! Image pyramids (`modules/imgproc/src/pyramids.cpp`).
//!
//! Both directions use the same separable 5-tap binomial `[1, 4, 6, 4, 1]/16`
//! with `BORDER_REFLECT_101`. `CV_8U` accumulates in integers and finishes with
//! `(sum + 128) >> 8`, matching upstream's fixed-point reduction; other depths
//! accumulate in `f64`.

use cv_core::{border_interpolate, cv_assert, BorderType, Depth, Mat, MatLike, Result};

use crate::color::set_channel;

/// The 5-tap binomial kernel, unnormalised.
const K: [i32; 5] = [1, 4, 6, 4, 1];
/// `sum(K) * sum(K)` — the normalising divisor for a full 2-D pass.
const K_SUM2: i32 = 256;

/// `cv::pyrDown(src, dst, dstsize, borderType)`.
///
/// Output size is `((cols + 1) / 2, (rows + 1) / 2)`, as upstream computes it.
pub fn pyr_down<S: MatLike + ?Sized>(src: &S, border: BorderType) -> Result<Mat> {
    cv_assert!(!src.is_empty(), "pyrDown requires a non-empty source");
    let (sw, sh) = (src.cols(), src.rows());
    let (dw, dh) = ((sw + 1) / 2, (sh + 1) / 2);
    let cn = src.channels();
    let is_u8 = src.depth() == Depth::U8;
    let mut dst = Mat::zeros(dh, dw, src.mat_type())?;

    for dy in 0..dh {
        for dx in 0..dw {
            for c in 0..cn {
                let mut acc_i = 0i32;
                let mut acc_f = 0.0f64;
                for (ky, &wy) in K.iter().enumerate() {
                    let sy = border_interpolate(2 * dy + ky as i32 - 2, sh, border)?;
                    for (kx, &wx) in K.iter().enumerate() {
                        let sx = border_interpolate(2 * dx + kx as i32 - 2, sw, border)?;
                        let v = src.at_f64(sy, sx, c)?;
                        if is_u8 {
                            acc_i += wy * wx * v as i32;
                        } else {
                            acc_f += (wy * wx) as f64 * v;
                        }
                    }
                }
                let out = if is_u8 {
                    ((acc_i + K_SUM2 / 2) >> 8) as f64
                } else {
                    acc_f / K_SUM2 as f64
                };
                set_channel(&mut dst, dy, dx, c, out)?;
            }
        }
    }
    Ok(dst)
}

/// `cv::pyrUp(src, dst, dstsize, borderType)`.
///
/// Output size is `(cols * 2, rows * 2)`. The source is conceptually
/// zero-stuffed to twice its size and convolved with the same kernel scaled by
/// four, so only taps of matching parity contribute to each output pixel.
pub fn pyr_up<S: MatLike + ?Sized>(src: &S, _border: BorderType) -> Result<Mat> {
    cv_assert!(!src.is_empty(), "pyrUp requires a non-empty source");
    let (sw, sh) = (src.cols() as usize, src.rows() as usize);
    let cn = src.channels() as usize;
    let buf = cv_core::to_f64(src)?;
    let integral = !src.depth().is_float();

    // Each source sample expands to two outputs: an "even" one aligned with it,
    // smoothed by `[1, 6, 1]`, and an "odd" one halfway to the next sample,
    // `[4, 4]`. Both denominators are 8, so a full 2-D pass has weight 64 and
    // the classic factor of four restores the level.
    //
    // The edges are **asymmetric**, which is the part that is easy to get
    // wrong: the previous neighbour reflects (`BORDER_REFLECT_101`), but the
    // next one *clamps*. Reflecting on both sides puts the last two rows and
    // columns off by as much as 85 levels — verified against a real cv2 build,
    // where clamping the forward tap is exact on every shape tried, including
    // degenerate 1xN and Nx1 inputs.
    let prev = |i: usize, n: usize| -> usize {
        if n == 1 {
            0
        } else if i == 0 {
            1
        } else {
            i - 1
        }
    };
    let next = |i: usize, n: usize| -> usize { (i + 1).min(n - 1) };

    // Horizontal pass: sw source columns become 2*sw output columns.
    let dw = sw * 2;
    let mut hz = vec![0.0f64; sh * dw * cn];
    for y in 0..sh {
        let row = &buf[y * sw * cn..(y + 1) * sw * cn];
        for x in 0..sw {
            let (lo, hi) = (prev(x, sw), next(x, sw));
            for c in 0..cn {
                let base = (y * dw + 2 * x) * cn + c;
                hz[base] = row[lo * cn + c] + 6.0 * row[x * cn + c] + row[hi * cn + c];
                hz[base + cn] = 4.0 * row[x * cn + c] + 4.0 * row[hi * cn + c];
            }
        }
    }

    // Vertical pass, then the single rounding step.
    let dh = sh * 2;
    let width = dw * cn;
    let mut out = vec![0.0f64; dh * width];
    for y in 0..sh {
        let (lo, hi) = (prev(y, sh), next(y, sh));
        for i in 0..width {
            let even = hz[lo * width + i] + 6.0 * hz[y * width + i] + hz[hi * width + i];
            let odd = 4.0 * hz[y * width + i] + 4.0 * hz[hi * width + i];
            out[2 * y * width + i] = finish(even, integral);
            out[(2 * y + 1) * width + i] = finish(odd, integral);
        }
    }
    cv_core::mat_from_f64(dh as i32, dw as i32, src.mat_type(), &out)
}

/// `cv::pyrUp(src, dst, dstsize, borderType)` with an explicit output size.
///
/// `pyrUp` doubles, so it cannot land on an odd dimension; `pyrDown` rounds
/// *up*, so a 121-row image becomes 61 and doubling gives 122. Upstream takes
/// the wanted size and crops, which is what makes a Laplacian pyramid
/// reconstruct exactly on an image whose size is not a power of two. Without
/// this the level below cannot be subtracted at all.
pub fn pyr_up_to<S: MatLike + ?Sized>(
    src: &S,
    rows: i32,
    cols: i32,
    border: BorderType,
) -> Result<Mat> {
    let full = pyr_up(src, border)?;
    cv_assert!(
        rows <= full.rows() && cols <= full.cols() && rows > 0 && cols > 0,
        "pyrUp to {rows}x{cols} is not reachable from {}x{}",
        src.rows(),
        src.cols()
    );
    if rows == full.rows() && cols == full.cols() {
        return Ok(full);
    }
    let cn = full.channels() as usize;
    let mut out = vec![0.0f64; (rows * cols) as usize * cn];
    for y in 0..rows {
        for x in 0..cols {
            for c in 0..cn {
                out[((y * cols + x) as usize) * cn + c] = full.at_f64(y, x, c as i32)?;
            }
        }
    }
    cv_core::mat_from_f64(rows, cols, full.mat_type(), &out)
}

/// Scale a `pyrUp` accumulator (weight 64) back to a pixel value.
///
/// Integer depths use upstream's `(acc * 4 + 128) >> 8`, which rounds half up;
/// float depths divide exactly.
#[inline]
fn finish(acc: f64, integral: bool) -> f64 {
    if integral {
        (((acc as i64) * 4 + 128) >> 8) as f64
    } else {
        acc * 4.0 / 256.0
    }
}

/// `cv::buildPyramid(src, dst, maxlevel, borderType)`.
///
/// Returns `maxlevel + 1` images, the first being a copy of `src`.
pub fn build_pyramid<S: MatLike + ?Sized>(
    src: &S,
    maxlevel: i32,
    border: BorderType,
) -> Result<Vec<Mat>> {
    cv_assert!(maxlevel >= 0, "maxlevel must be non-negative");
    let mut out = vec![Mat::copy_from(src)?];
    for i in 0..maxlevel {
        let next = pyr_down(&out[i as usize], border)?;
        out.push(next);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{types::CV_8UC3, Scalar, Size};

    #[test]
    fn pyr_down_halves_the_size_rounding_up() {
        for (w, h, ew, eh) in [(8, 8, 4, 4), (7, 9, 4, 5), (1, 1, 1, 1), (5, 3, 3, 2)] {
            let m = Mat::zeros(h, w, cv_core::types::CV_8UC1).unwrap();
            let d = pyr_down(&m, BorderType::Reflect101).unwrap();
            assert_eq!(d.size(), Size::new(ew, eh), "for {w}x{h}");
        }
    }

    #[test]
    fn pyr_down_preserves_a_flat_image() {
        let m = Mat::new_with(8, 8, CV_8UC3, Scalar::new(30.0, 60.0, 90.0, 0.0)).unwrap();
        let d = pyr_down(&m, BorderType::Reflect101).unwrap();
        for y in 0..d.rows() {
            for x in 0..d.cols() {
                assert_eq!(d.pixel::<u8>(y, x).unwrap(), &[30, 60, 90], "at {y},{x}");
            }
        }
    }

    #[test]
    fn pyr_up_doubles_the_size() {
        let m = Mat::zeros(3, 5, cv_core::types::CV_8UC1).unwrap();
        let u = pyr_up(&m, BorderType::Reflect101).unwrap();
        assert_eq!(u.size(), Size::new(10, 6));
    }

    #[test]
    fn pyr_up_preserves_a_flat_image() {
        // The x4 gain must exactly compensate the zero-stuffing; getting this
        // wrong shows up as a uniform brightness shift.
        let m = Mat::new_with(4, 4, CV_8UC3, Scalar::new(77.0, 128.0, 200.0, 0.0)).unwrap();
        let u = pyr_up(&m, BorderType::Reflect101).unwrap();
        for y in 0..u.rows() {
            for x in 0..u.cols() {
                assert_eq!(u.pixel::<u8>(y, x).unwrap(), &[77, 128, 200], "at {y},{x}");
            }
        }
    }

    #[test]
    fn pyr_down_smooths_an_impulse() {
        let mut data = vec![0u8; 64];
        data[4 * 8 + 4] = 255;
        let m = Mat::from_vec(8, 8, 1, &data).unwrap();
        let d = pyr_down(&m, BorderType::Reflect101).unwrap();
        // Energy spreads over neighbours; the peak lands at (2,2).
        let mm = cv_core::min_max_loc::<Mat, Mat>(&d, None).unwrap();
        assert_eq!(mm.max_loc, cv_core::Point::new(2, 2));
        assert!(mm.max_val < 255.0 && mm.max_val > 0.0);
    }

    #[test]
    fn pyr_down_then_up_returns_to_the_original_size() {
        let m = Mat::new_with(8, 8, cv_core::types::CV_8UC1, Scalar::single(120.0)).unwrap();
        let d = pyr_down(&m, BorderType::Reflect101).unwrap();
        let u = pyr_up(&d, BorderType::Reflect101).unwrap();
        assert_eq!(u.size(), m.size());
        assert!(u.to_vec::<u8>().unwrap().iter().all(|&v| v == 120));
    }

    #[test]
    fn float_pyramids_use_the_float_path() {
        let m = Mat::new_with(4, 4, cv_core::types::CV_32FC1, Scalar::single(0.25)).unwrap();
        let d = pyr_down(&m, BorderType::Reflect101).unwrap();
        assert_eq!(d.depth(), Depth::F32);
        for v in d.to_vec::<f32>().unwrap() {
            assert!((v - 0.25).abs() < 1e-6, "{v}");
        }
    }

    #[test]
    fn build_pyramid_returns_maxlevel_plus_one() {
        let m = Mat::zeros(16, 16, cv_core::types::CV_8UC1).unwrap();
        let p = build_pyramid(&m, 3, BorderType::Reflect101).unwrap();
        assert_eq!(p.len(), 4);
        assert_eq!(p[0].size(), Size::new(16, 16));
        assert_eq!(p[1].size(), Size::new(8, 8));
        assert_eq!(p[2].size(), Size::new(4, 4));
        assert_eq!(p[3].size(), Size::new(2, 2));
    }
}
