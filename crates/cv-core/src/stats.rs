//! Reductions and statistics (`modules/core/src/stat.dispatch.cpp`, `minmax.cpp`, `norm.cpp`).

use crate::arithm::convert_to;
use crate::error::{Code, Result};
use crate::geometry::{Point, Scalar};
use crate::mat::{Mat, MatLike};
use crate::types::Depth;
use crate::{cv_assert, cv_error};

/// `cv::NormTypes`, with upstream's numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NormType {
    /// `NORM_INF` — maximum absolute value.
    Inf = 1,
    /// `NORM_L1` — sum of absolute values.
    L1 = 2,
    /// `NORM_L2` — Euclidean norm.
    L2 = 4,
    /// `NORM_L2SQR` — squared Euclidean norm.
    L2Sqr = 5,
    /// `NORM_HAMMING` — number of differing bits; `CV_8U` input only.
    Hamming = 6,
    /// `NORM_HAMMING2` — number of differing *bit pairs*; `CV_8U` input only.
    ///
    /// Each two bits count as one unit, which is what a descriptor packing two
    /// bits per element (BRIEF's variants, ORB's `WTA_K` of 3 or 4) needs.
    Hamming2 = 7,
}

/// `NORM_TYPE_MASK` — the bits of a flags word that hold the [`NormType`].
pub const NORM_TYPE_MASK: i32 = 7;
/// `NORM_RELATIVE` — divide by the norm of the second operand; see [`norm_relative`].
pub const NORM_RELATIVE: i32 = 8;
/// `NORM_MINMAX` — rescale into a range; see [`normalize_minmax`].
pub const NORM_MINMAX: i32 = 32;

impl NormType {
    /// Is this one of the bit-counting norms, which only accept `CV_8U`?
    #[must_use]
    fn is_hamming(self) -> bool {
        matches!(self, NormType::Hamming | NormType::Hamming2)
    }
}

/// Bits set in `v`, counting each two-bit cell as one unit.
///
/// Upstream precomputes this as `popCountTable2`; the arithmetic is cheap
/// enough to do directly and leaves nothing to get out of step with the table.
#[inline]
fn popcount2(v: u8) -> u32 {
    // Collapse each pair to 1 when either of its bits is set, then count.
    ((v | (v >> 1)) & 0x55).count_ones()
}

/// Hamming distance between two byte strings at the given cell size.
fn hamming(a: &[u8], b: Option<&[u8]>, two_bit: bool) -> f64 {
    let bits = |i: usize, x: u8| -> u32 {
        let v = match b {
            Some(other) => x ^ other[i],
            None => x,
        };
        if two_bit {
            popcount2(v)
        } else {
            v.count_ones()
        }
    };
    f64::from(a.iter().enumerate().map(|(i, &x)| bits(i, x)).sum::<u32>())
}

/// Result of [`min_max_loc`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinMaxLoc {
    /// Smallest value found.
    pub min_val: f64,
    /// Largest value found.
    pub max_val: f64,
    /// Location of the first occurrence of `min_val`, in scan order.
    pub min_loc: Point,
    /// Location of the first occurrence of `max_val`, in scan order.
    pub max_loc: Point,
}

/// `cv::minMaxLoc(src, ...)` — single-channel only, as upstream.
///
/// With a mask, only pixels where the mask is non-zero are considered; if the
/// mask selects nothing, this returns `StsBadArg` rather than reporting
/// meaningless extrema.
pub fn min_max_loc<S: MatLike + ?Sized, M: MatLike + ?Sized>(
    src: &S,
    mask: Option<&M>,
) -> Result<MinMaxLoc> {
    cv_assert!(src.channels() == 1, "minMaxLoc requires a single-channel matrix");
    cv_assert!(!src.is_empty(), "minMaxLoc requires a non-empty matrix");
    if let Some(m) = mask {
        cv_assert!(m.mat_type() == crate::types::CV_8UC1, "mask must be CV_8UC1");
        cv_assert!(m.size() == src.size(), "mask size must match the source");
    }

    let mut out: Option<MinMaxLoc> = None;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            if let Some(m) = mask {
                if m.at::<u8>(y, x)? == 0 {
                    continue;
                }
            }
            let v = src.at_f64(y, x, 0)?;
            // NaN never compares greater or less, so it is skipped — matching
            // upstream, which uses plain `<` / `>` comparisons.
            match &mut out {
                None => {
                    out = Some(MinMaxLoc {
                        min_val: v,
                        max_val: v,
                        min_loc: Point::new(x, y),
                        max_loc: Point::new(x, y),
                    })
                }
                Some(acc) => {
                    if v < acc.min_val {
                        acc.min_val = v;
                        acc.min_loc = Point::new(x, y);
                    }
                    if v > acc.max_val {
                        acc.max_val = v;
                        acc.max_loc = Point::new(x, y);
                    }
                }
            }
        }
    }
    out.ok_or_else(|| {
        crate::Error::new(Code::StsBadArg, "minMaxLoc: mask selected no elements".to_string())
    })
}

/// `cv::sum(src)` — per-channel sum, accumulated in `f64`.
pub fn sum<S: MatLike + ?Sized>(src: &S) -> Result<Scalar> {
    let cn = src.channels();
    cv_assert!(cn <= 4, "sum() supports at most 4 channels");
    let mut acc = [0.0f64; 4];
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            for c in 0..cn {
                acc[c as usize] += src.at_f64(y, x, c)?;
            }
        }
    }
    Ok(Scalar(acc))
}

/// `cv::countNonZero(src)` — single-channel only, as upstream.
pub fn count_non_zero<S: MatLike + ?Sized>(src: &S) -> Result<i32> {
    cv_assert!(src.channels() == 1, "countNonZero requires a single-channel matrix");
    let mut n = 0;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            if src.at_f64(y, x, 0)? != 0.0 {
                n += 1;
            }
        }
    }
    Ok(n)
}

/// `cv::mean(src, mask)` — per-channel mean over the selected pixels.
pub fn mean<S: MatLike + ?Sized, M: MatLike + ?Sized>(src: &S, mask: Option<&M>) -> Result<Scalar> {
    let (s, n) = sum_and_count(src, mask)?;
    if n == 0 {
        return Ok(Scalar::default());
    }
    let mut out = [0.0; 4];
    for c in 0..4 {
        out[c] = s[c] / n as f64;
    }
    Ok(Scalar(out))
}

/// `cv::meanStdDev(src, mean, stddev, mask)`.
///
/// The deviation is the population (not sample) standard deviation, matching
/// upstream's divide-by-N.
pub fn mean_std_dev<S: MatLike + ?Sized, M: MatLike + ?Sized>(
    src: &S,
    mask: Option<&M>,
) -> Result<(Scalar, Scalar)> {
    let cn = src.channels();
    cv_assert!(cn <= 4, "meanStdDev() supports at most 4 channels");
    let (s, n) = sum_and_count(src, mask)?;
    if n == 0 {
        return Ok((Scalar::default(), Scalar::default()));
    }
    let mut mu = [0.0f64; 4];
    for c in 0..4 {
        mu[c] = s[c] / n as f64;
    }
    let mut sq = [0.0f64; 4];
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            if !mask_allows(mask, y, x)? {
                continue;
            }
            for c in 0..cn as usize {
                let d = src.at_f64(y, x, c as i32)? - mu[c];
                sq[c] += d * d;
            }
        }
    }
    let mut sd = [0.0f64; 4];
    for c in 0..4 {
        sd[c] = (sq[c] / n as f64).sqrt();
    }
    Ok((Scalar(mu), Scalar(sd)))
}

/// Per-channel sum plus the number of selected pixels.
fn sum_and_count<S: MatLike + ?Sized, M: MatLike + ?Sized>(
    src: &S,
    mask: Option<&M>,
) -> Result<([f64; 4], usize)> {
    let cn = src.channels();
    cv_assert!(cn <= 4, "statistics support at most 4 channels");
    if let Some(m) = mask {
        cv_assert!(m.mat_type() == crate::types::CV_8UC1, "mask must be CV_8UC1");
        cv_assert!(m.size() == src.size(), "mask size must match the source");
    }
    let mut acc = [0.0f64; 4];
    let mut n = 0usize;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            if !mask_allows(mask, y, x)? {
                continue;
            }
            n += 1;
            for c in 0..cn {
                acc[c as usize] += src.at_f64(y, x, c)?;
            }
        }
    }
    Ok((acc, n))
}

#[inline]
fn mask_allows<M: MatLike + ?Sized>(mask: Option<&M>, y: i32, x: i32) -> Result<bool> {
    Ok(match mask {
        None => true,
        Some(m) => m.at::<u8>(y, x)? != 0,
    })
}

/// `cv::norm(src, normType)`.
pub fn norm<S: MatLike + ?Sized>(src: &S, norm_type: NormType) -> Result<f64> {
    norm_masked::<S, Mat>(src, norm_type, None)
}

/// `cv::norm(src, normType, mask)` — the norm over the masked-in elements only.
pub fn norm_masked<S: MatLike + ?Sized, M: MatLike + ?Sized>(
    src: &S,
    norm_type: NormType,
    mask: Option<&M>,
) -> Result<f64> {
    if norm_type.is_hamming() {
        cv_assert!(src.depth() == Depth::U8, "NORM_HAMMING requires a CV_8U matrix");
        cv_assert!(mask.is_none(), "NORM_HAMMING does not take a mask");
        let mut total = 0.0;
        for y in 0..src.rows() {
            total += hamming(src.row_as::<u8>(y)?, None, norm_type == NormType::Hamming2);
        }
        return Ok(total);
    }
    let mut inf = 0.0f64;
    let mut l1 = 0.0f64;
    let mut l2 = 0.0f64;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            if !mask_allows(mask, y, x)? {
                continue;
            }
            for c in 0..src.channels() {
                let v = src.at_f64(y, x, c)?.abs();
                inf = inf.max(v);
                l1 += v;
                l2 += v * v;
            }
        }
    }
    Ok(match norm_type {
        NormType::Inf => inf,
        NormType::L1 => l1,
        NormType::L2 => l2.sqrt(),
        NormType::L2Sqr => l2,
        NormType::Hamming | NormType::Hamming2 => unreachable!("handled above"),
    })
}

/// `cv::norm(src1, src2, normType)` — norm of the difference.
pub fn norm_diff<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    a: &A,
    b: &B,
    norm_type: NormType,
) -> Result<f64> {
    norm_diff_masked::<A, B, Mat>(a, b, norm_type, None)
}

/// `cv::norm(src1, src2, normType, mask)`.
pub fn norm_diff_masked<A: MatLike + ?Sized, B: MatLike + ?Sized, M: MatLike + ?Sized>(
    a: &A,
    b: &B,
    norm_type: NormType,
    mask: Option<&M>,
) -> Result<f64> {
    if a.size() != b.size() || a.mat_type() != b.mat_type() {
        return cv_error!(Code::StsUnmatchedSizes, "norm() operands must match in size and type");
    }
    if norm_type.is_hamming() {
        cv_assert!(a.depth() == Depth::U8, "NORM_HAMMING requires CV_8U matrices");
        cv_assert!(mask.is_none(), "NORM_HAMMING does not take a mask");
        let mut total = 0.0;
        for y in 0..a.rows() {
            total += hamming(
                a.row_as::<u8>(y)?,
                Some(b.row_as::<u8>(y)?),
                norm_type == NormType::Hamming2,
            );
        }
        return Ok(total);
    }
    let mut inf = 0.0f64;
    let mut l1 = 0.0f64;
    let mut l2 = 0.0f64;
    for y in 0..a.rows() {
        for x in 0..a.cols() {
            if !mask_allows(mask, y, x)? {
                continue;
            }
            for c in 0..a.channels() {
                let v = (a.at_f64(y, x, c)? - b.at_f64(y, x, c)?).abs();
                inf = inf.max(v);
                l1 += v;
                l2 += v * v;
            }
        }
    }
    Ok(match norm_type {
        NormType::Inf => inf,
        NormType::L1 => l1,
        NormType::L2 => l2.sqrt(),
        NormType::L2Sqr => l2,
        NormType::Hamming | NormType::Hamming2 => unreachable!("handled above"),
    })
}

/// `cv::norm(src1, src2, normType | NORM_RELATIVE)`.
///
/// The difference norm divided by the norm of `src2`, so the result is a
/// scale-free error measure. Upstream carries this as the [`NORM_RELATIVE`]
/// bit in the same flags word rather than a separate call.
pub fn norm_relative<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    a: &A,
    b: &B,
    norm_type: NormType,
) -> Result<f64> {
    let d = norm_diff(a, b, norm_type)?;
    Ok(d / norm(b, norm_type)?)
}

/// `cv::normalize(src, dst, alpha, beta, NORM_MINMAX, dtype)`.
///
/// Linearly rescales so the minimum maps to `alpha` and the maximum to `beta`.
/// A constant input maps entirely to `alpha`, as upstream does when the range
/// collapses.
pub fn normalize_minmax<S: MatLike + ?Sized>(
    src: &S,
    alpha: f64,
    beta: f64,
    depth: Option<Depth>,
) -> Result<Mat> {
    cv_assert!(src.channels() == 1, "normalize(NORM_MINMAX) requires a single-channel matrix");
    let mm = min_max_loc::<S, Mat>(src, None)?;
    let (lo, hi) = (alpha.min(beta), alpha.max(beta));
    let range = mm.max_val - mm.min_val;
    let (scale, shift) = if range == 0.0 {
        (0.0, lo)
    } else {
        let s = (hi - lo) / range;
        (s, lo - mm.min_val * s)
    };
    convert_to(src, depth.unwrap_or_else(|| src.depth()), scale, shift)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CV_8UC1;

    fn u8m(rows: i32, cols: i32, v: &[u8]) -> Mat {
        Mat::from_vec(rows, cols, 1, v).unwrap()
    }

    /// Reference values from `cv2.norm` on the same bytes.
    #[test]
    fn hamming_norms_count_bits_and_bit_pairs() {
        let a = u8m(1, 4, &[0b1011_0100, 0b0000_1111, 0b1111_1111, 0b0000_0000]);
        let b = u8m(1, 4, &[0b1011_0000, 0b1010_1010, 0b0000_0000, 0b1100_0011]);
        assert_eq!(norm(&a, NormType::Hamming).unwrap(), 16.0);
        assert_eq!(norm(&a, NormType::Hamming2).unwrap(), 9.0);
        assert_eq!(norm_diff(&a, &b, NormType::Hamming).unwrap(), 17.0);
        assert_eq!(norm_diff(&a, &b, NormType::Hamming2).unwrap(), 11.0);
    }

    #[test]
    fn hamming_rejects_a_non_byte_matrix() {
        let f = Mat::from_vec(1, 2, 1, &[1.0f64, 2.0]).unwrap();
        assert!(norm(&f, NormType::Hamming).is_err());
    }

    #[test]
    fn a_mask_restricts_the_norm_to_its_set_elements() {
        let f = Mat::from_vec(1, 4, 1, &[1.0f64, -2.0, 3.0, -4.0]).unwrap();
        let m = u8m(1, 4, &[1, 0, 1, 0]);
        assert_eq!(norm_masked(&f, NormType::L1, Some(&m)).unwrap(), 4.0);
        assert_eq!(norm_masked(&f, NormType::Inf, Some(&m)).unwrap(), 3.0);
        let l2 = norm_masked(&f, NormType::L2, Some(&m)).unwrap();
        assert!((l2 - 10.0f64.sqrt()).abs() < 1e-15, "{l2}");
    }

    #[test]
    fn relative_norm_divides_by_the_second_operand() {
        let a = Mat::from_vec(1, 4, 1, &[180.0f64, 15.0, 255.0, 0.0]).unwrap();
        let b = Mat::from_vec(1, 4, 1, &[176.0f64, 170.0, 0.0, 195.0]).unwrap();
        let r = norm_relative(&a, &b, NormType::L2).unwrap();
        assert!((r - 1.139_367_519_123_155_4).abs() < 1e-12, "{r}");
    }

    #[test]
    fn min_max_loc_reports_first_occurrence() {
        let m = u8m(2, 3, &[5, 1, 7, 1, 9, 9]);
        let r = min_max_loc::<Mat, Mat>(&m, None).unwrap();
        assert_eq!(r.min_val, 1.0);
        assert_eq!(r.max_val, 9.0);
        assert_eq!(r.min_loc, Point::new(1, 0));
        assert_eq!(r.max_loc, Point::new(1, 1));
    }

    #[test]
    fn min_max_loc_honours_mask() {
        let m = u8m(1, 4, &[5, 1, 7, 9]);
        let mask = u8m(1, 4, &[0, 0, 255, 255]);
        let r = min_max_loc(&m, Some(&mask)).unwrap();
        assert_eq!(r.min_val, 7.0);
        assert_eq!(r.max_val, 9.0);
        assert_eq!(r.min_loc, Point::new(2, 0));
    }

    #[test]
    fn min_max_loc_with_empty_mask_errors() {
        let m = u8m(1, 3, &[1, 2, 3]);
        let mask = u8m(1, 3, &[0, 0, 0]);
        assert!(min_max_loc(&m, Some(&mask)).is_err());
    }

    #[test]
    fn sum_and_mean_are_per_channel() {
        let m = Mat::from_vec(1, 2, 3, &[1u8, 2, 3, 5, 6, 7]).unwrap();
        let s = sum(&m).unwrap();
        assert_eq!([s[0], s[1], s[2]], [6.0, 8.0, 10.0]);
        let mu = mean::<Mat, Mat>(&m, None).unwrap();
        assert_eq!([mu[0], mu[1], mu[2]], [3.0, 4.0, 5.0]);
    }

    #[test]
    fn mean_std_dev_uses_population_divisor() {
        // values 2,4,4,4,5,5,7,9 -> mean 5, population sd 2
        let m = u8m(1, 8, &[2, 4, 4, 4, 5, 5, 7, 9]);
        let (mu, sd) = mean_std_dev::<Mat, Mat>(&m, None).unwrap();
        assert!((mu[0] - 5.0).abs() < 1e-12);
        assert!((sd[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn count_non_zero_counts_pixels() {
        let m = u8m(2, 2, &[0, 3, 0, 7]);
        assert_eq!(count_non_zero(&m).unwrap(), 2);
    }

    #[test]
    fn norms_match_definitions() {
        let m = Mat::from_vec(1, 3, 1, &[3.0f32, -4.0, 0.0]).unwrap();
        assert_eq!(norm(&m, NormType::Inf).unwrap(), 4.0);
        assert_eq!(norm(&m, NormType::L1).unwrap(), 7.0);
        assert_eq!(norm(&m, NormType::L2).unwrap(), 5.0);
        assert_eq!(norm(&m, NormType::L2Sqr).unwrap(), 25.0);
    }

    #[test]
    fn norm_diff_measures_distance() {
        let a = Mat::from_vec(1, 2, 1, &[1.0f32, 2.0]).unwrap();
        let b = Mat::from_vec(1, 2, 1, &[4.0f32, 6.0]).unwrap();
        assert_eq!(norm_diff(&a, &b, NormType::L2).unwrap(), 5.0);
        assert_eq!(norm_diff(&a, &b, NormType::L1).unwrap(), 7.0);
    }

    #[test]
    fn normalize_minmax_spans_the_target_range() {
        let m = u8m(1, 5, &[10, 20, 30, 40, 50]);
        let n = normalize_minmax(&m, 0.0, 255.0, None).unwrap();
        let v = n.to_vec::<u8>().unwrap();
        assert_eq!(v[0], 0);
        assert_eq!(v[4], 255);
        assert_eq!(v[2], 128); // 20/40 * 255 = 127.5 -> 128 (half-even from .5)
    }

    #[test]
    fn normalize_constant_input_does_not_divide_by_zero() {
        let m = u8m(1, 3, &[7, 7, 7]);
        let n = normalize_minmax(&m, 0.0, 255.0, None).unwrap();
        assert_eq!(n.to_vec::<u8>().unwrap(), vec![0, 0, 0]);
    }

    #[test]
    fn stats_accept_strided_views() {
        let data: Vec<u8> = (0..25).collect();
        let m = Mat::from_vec(5, 5, 1, &data).unwrap();
        let v = m.roi(crate::Rect::new(1, 1, 2, 2)).unwrap();
        assert_eq!(v.mat_type(), CV_8UC1);
        let r = min_max_loc::<_, Mat>(&v, None).unwrap();
        assert_eq!(r.min_val, 6.0);
        assert_eq!(r.max_val, 12.0);
        assert_eq!(sum(&v).unwrap()[0], 6.0 + 7.0 + 11.0 + 12.0);
    }
}
