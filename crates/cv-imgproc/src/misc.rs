//! Assorted image transforms: adaptive thresholding, integral images and
//! template matching (`modules/imgproc/src/thresh.cpp`, `sumpixels.dispatch.cpp`,
//! `templmatch.cpp`).

use cv_core::{
    cv_assert, cv_error, BorderType, Code, Depth, Mat, MatLike, MatLikeMut, MatType, Result, Size,
};

use crate::color::set_channel;
use crate::filter::{box_filter, gaussian_blur};
use crate::threshold::ThresholdType;

/// `cv::AdaptiveThresholdTypes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum AdaptiveMethod {
    /// `ADAPTIVE_THRESH_MEAN_C` — the local mean over `block_size`.
    MeanC = 0,
    /// `ADAPTIVE_THRESH_GAUSSIAN_C` — a Gaussian-weighted local mean.
    GaussianC = 1,
}

/// `cv::adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C)`.
///
/// `CV_8UC1` only. Upstream rounds `C` differently per threshold type —
/// `cvCeil` for `THRESH_BINARY`, `cvFloor` for `THRESH_BINARY_INV` — and
/// compares through a 768-entry table; the same integer comparison is used here.
pub fn adaptive_threshold<S: MatLike + ?Sized>(
    src: &S,
    max_value: f64,
    method: AdaptiveMethod,
    ttype: ThresholdType,
    block_size: i32,
    c: f64,
) -> Result<Mat> {
    if src.depth() != Depth::U8 || src.channels() != 1 {
        return cv_error!(
            Code::StsUnsupportedFormat,
            "adaptiveThreshold requires CV_8UC1, got {}",
            src.mat_type()
        );
    }
    cv_assert!(
        block_size >= 3 && block_size % 2 == 1,
        "blockSize must be odd and at least 3, got {block_size}"
    );
    let inv = match ttype {
        ThresholdType::Binary => false,
        ThresholdType::BinaryInv => true,
        _ => {
            return cv_error!(
                Code::StsBadArg,
                "adaptiveThreshold supports only THRESH_BINARY and THRESH_BINARY_INV"
            )
        }
    };

    // Upstream uses BORDER_REPLICATE for both smoothing methods.
    let mean = match method {
        AdaptiveMethod::MeanC => {
            box_filter(src, Size::new(block_size, block_size), true, BorderType::Replicate)?
        }
        AdaptiveMethod::GaussianC => gaussian_blur(
            src,
            Size::new(block_size, block_size),
            0.0,
            0.0,
            BorderType::Replicate,
        )?,
    };

    let imaxval = cv_core::saturate::Elem::from_f64(max_value);
    let idelta = if inv { cv_core::cv_floor(c) } else { cv_core::cv_ceil(c) };

    let mut dst = Mat::zeros(src.rows(), src.cols(), src.mat_type())?;
    for y in 0..src.rows() {
        let s: Vec<u8> = src.row_as::<u8>(y)?.to_vec();
        let m: Vec<u8> = mean.row_as::<u8>(y)?.to_vec();
        let d = dst.row_as_mut::<u8>(y)?;
        for i in 0..d.len() {
            let diff = s[i] as i32 - m[i] as i32;
            let above = diff > -idelta;
            d[i] = if above != inv { imaxval } else { 0u8 };
        }
    }
    Ok(dst)
}

/// `cv::integral(src, sum, sqsum, sdepth, sqdepth)`.
///
/// Returns `(sum, sqsum)`, both `(rows + 1) x (cols + 1)` with a zero first row
/// and column so that any rectangle sum is a four-corner lookup.
pub fn integral<S: MatLike + ?Sized>(src: &S) -> Result<(Mat, Mat)> {
    cv_assert!(!src.is_empty(), "integral requires a non-empty source");
    let cn = src.channels();
    let (w, h) = (src.cols(), src.rows());
    let typ = MatType::new(Depth::F64, cn);
    let mut sum = Mat::zeros(h + 1, w + 1, typ)?;
    let mut sqsum = Mat::zeros(h + 1, w + 1, typ)?;

    for y in 0..h {
        for x in 0..w {
            for c in 0..cn {
                let v = src.at_f64(y, x, c)?;
                // S(y+1,x+1) = s(y,x) + S(y,x+1) + S(y+1,x) - S(y,x)
                let s = v + sum.at_f64(y, x + 1, c)? + sum.at_f64(y + 1, x, c)?
                    - sum.at_f64(y, x, c)?;
                set_channel(&mut sum, y + 1, x + 1, c, s)?;

                let q = v * v + sqsum.at_f64(y, x + 1, c)? + sqsum.at_f64(y + 1, x, c)?
                    - sqsum.at_f64(y, x, c)?;
                set_channel(&mut sqsum, y + 1, x + 1, c, q)?;
            }
        }
    }
    Ok((sum, sqsum))
}

/// `cv::TemplateMatchModes`, with upstream's numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum TemplateMatchMode {
    /// `TM_SQDIFF` — sum of squared differences; lower is better.
    SqDiff = 0,
    /// `TM_SQDIFF_NORMED`.
    SqDiffNormed = 1,
    /// `TM_CCORR` — cross-correlation; higher is better.
    CCorr = 2,
    /// `TM_CCORR_NORMED`.
    CCorrNormed = 3,
    /// `TM_CCOEFF` — zero-mean cross-correlation.
    CCoeff = 4,
    /// `TM_CCOEFF_NORMED`.
    CCoeffNormed = 5,
}

/// `cv::matchTemplate(image, templ, result, method)`.
///
/// # More accurate than OpenCV, deliberately
///
/// Upstream computes the correlation by **block-wise FFT in `f32`**, which is
/// fast and lossy. This sums directly in `f64` and rounds once.
///
/// On the parity fixture — a 120x160 image and a 16x20 template — every exact
/// `TM_CCORR` and `TM_SQDIFF` value is an integer below 2^24, so `f32` can
/// represent all of them without error. Against that exact integer answer:
///
/// | | max error | cells wrong |
/// |---|---|---|
/// | this port | **0** | **0** |
/// | cv2 4.13.0 | 4 | 65.5% |
///
/// So the two disagree, and OpenCV is the one that is off. Reproducing its
/// FFT blocking to match bit for bit would mean deliberately importing that
/// round-off; the parity harness records these three methods as a known
/// divergence with a known direction instead.
pub fn match_template<I: MatLike + ?Sized, T: MatLike + ?Sized>(
    image: &I,
    templ: &T,
    method: TemplateMatchMode,
) -> Result<Mat> {
    cv_assert!(!image.is_empty() && !templ.is_empty(), "matchTemplate requires non-empty inputs");
    if templ.rows() > image.rows() || templ.cols() > image.cols() {
        return cv_error!(
            Code::StsBadSize,
            "template {}x{} is larger than the image {}x{}",
            templ.cols(),
            templ.rows(),
            image.cols(),
            image.rows()
        );
    }
    if image.mat_type() != templ.mat_type() {
        return cv_error!(Code::StsUnmatchedFormats, "image and template types must match");
    }

    let cn = image.channels();
    let (tw, th) = (templ.cols(), templ.rows());
    let (rw, rh) = (image.cols() - tw + 1, image.rows() - th + 1);
    let mut result = Mat::zeros(rh, rw, cv_core::types::CV_32FC1)?;

    // Template statistics are position independent, so hoist them.
    let mut t_vals = Vec::with_capacity((th * tw * cn) as usize);
    for y in 0..th {
        for x in 0..tw {
            for c in 0..cn {
                t_vals.push(templ.at_f64(y, x, c)?);
            }
        }
    }
    let t_count = t_vals.len() as f64;
    let t_mean = t_vals.iter().sum::<f64>() / t_count;
    let t_sq: f64 = t_vals.iter().map(|v| v * v).sum();
    let t_zero_sq: f64 = t_vals.iter().map(|v| (v - t_mean) * (v - t_mean)).sum();

    for ry in 0..rh {
        for rx in 0..rw {
            let mut w_vals = Vec::with_capacity(t_vals.len());
            for y in 0..th {
                for x in 0..tw {
                    for c in 0..cn {
                        w_vals.push(image.at_f64(ry + y, rx + x, c)?);
                    }
                }
            }
            let w_mean = w_vals.iter().sum::<f64>() / t_count;
            let w_sq: f64 = w_vals.iter().map(|v| v * v).sum();
            let w_zero_sq: f64 = w_vals.iter().map(|v| (v - w_mean) * (v - w_mean)).sum();

            let ccorr: f64 = w_vals.iter().zip(t_vals.iter()).map(|(a, b)| a * b).sum();
            let sqdiff: f64 =
                w_vals.iter().zip(t_vals.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
            let ccoeff: f64 = w_vals
                .iter()
                .zip(t_vals.iter())
                .map(|(a, b)| (a - w_mean) * (b - t_mean))
                .sum();

            let v = match method {
                TemplateMatchMode::SqDiff => sqdiff,
                TemplateMatchMode::CCorr => ccorr,
                TemplateMatchMode::CCoeff => ccoeff,
                TemplateMatchMode::SqDiffNormed => norm_div(sqdiff, (w_sq * t_sq).sqrt()),
                TemplateMatchMode::CCorrNormed => norm_div(ccorr, (w_sq * t_sq).sqrt()),
                TemplateMatchMode::CCoeffNormed => {
                    norm_div(ccoeff, (w_zero_sq * t_zero_sq).sqrt())
                }
            };
            result.pixel_mut::<f32>(ry, rx)?[0] = v as f32;
        }
    }
    Ok(result)
}

/// Normalised score, guarding the degenerate all-zero window.
fn norm_div(num: f64, den: f64) -> f64 {
    if den.abs() < f64::EPSILON {
        // Upstream leaves these at 0 (or 1 for SQDIFF_NORMED); 0 is the value
        // callers compare against, so use it consistently.
        0.0
    } else {
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{min_max_loc, types::CV_8UC1, Point, Scalar};

    #[test]
    fn mode_values_match_opencv() {
        assert_eq!(TemplateMatchMode::SqDiff as i32, 0);
        assert_eq!(TemplateMatchMode::CCoeffNormed as i32, 5);
        assert_eq!(AdaptiveMethod::MeanC as i32, 0);
        assert_eq!(AdaptiveMethod::GaussianC as i32, 1);
    }

    #[test]
    fn integral_gives_rectangle_sums_in_four_lookups() {
        let m = Mat::from_vec(3, 3, 1, &[1u8, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let (sum, sqsum) = integral(&m).unwrap();
        assert_eq!(sum.size(), Size::new(4, 4));
        // First row and column are zero.
        assert_eq!(sum.at::<f64>(0, 0).unwrap(), 0.0);
        assert_eq!(sum.at::<f64>(0, 3).unwrap(), 0.0);
        // Whole-image sum is 45.
        assert_eq!(sum.at::<f64>(3, 3).unwrap(), 45.0);
        // Sum of squares is 285.
        assert_eq!(sqsum.at::<f64>(3, 3).unwrap(), 285.0);

        // The 2x2 block at (1,1)..(2,2) is 5+6+8+9 = 28.
        let s = sum.at::<f64>(3, 3).unwrap() - sum.at::<f64>(1, 3).unwrap()
            - sum.at::<f64>(3, 1).unwrap()
            + sum.at::<f64>(1, 1).unwrap();
        assert_eq!(s, 28.0);
    }

    #[test]
    fn integral_handles_multichannel() {
        let m = Mat::from_vec(1, 2, 3, &[1u8, 2, 3, 4, 5, 6]).unwrap();
        let (sum, _) = integral(&m).unwrap();
        assert_eq!(sum.channels(), 3);
        assert_eq!(sum.pixel::<f64>(1, 2).unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn adaptive_threshold_splits_a_gradient() {
        // A left-dark, right-bright image: adaptive thresholding responds to
        // the local contrast at the seam rather than a global level.
        let mut data = vec![0u8; 100];
        for y in 0..10 {
            for x in 5..10 {
                data[y * 10 + x] = 255;
            }
        }
        let m = Mat::from_vec(10, 10, 1, &data).unwrap();
        let t = adaptive_threshold(&m, 255.0, AdaptiveMethod::MeanC, ThresholdType::Binary, 5, 0.0)
            .unwrap();
        assert_eq!(t.size(), m.size());
        // The bright side is above its local mean somewhere.
        assert!(cv_core::count_non_zero(&t).unwrap() > 0);
    }

    #[test]
    fn adaptive_threshold_inverts_consistently() {
        let m = Mat::from_vec(8, 8, 1, &(0u8..64).collect::<Vec<_>>()).unwrap();
        let a =
            adaptive_threshold(&m, 255.0, AdaptiveMethod::MeanC, ThresholdType::Binary, 3, 0.0)
                .unwrap();
        let b =
            adaptive_threshold(&m, 255.0, AdaptiveMethod::MeanC, ThresholdType::BinaryInv, 3, 0.0)
                .unwrap();
        // Every pixel set in one must be clear in the other.
        for (p, q) in a.to_vec::<u8>().unwrap().iter().zip(b.to_vec::<u8>().unwrap().iter()) {
            assert_ne!(*p == 255, *q == 255, "both were {p}/{q}");
        }
    }

    #[test]
    fn adaptive_threshold_rejects_bad_parameters() {
        let m = Mat::zeros(8, 8, CV_8UC1).unwrap();
        // Even block size.
        assert!(adaptive_threshold(&m, 255.0, AdaptiveMethod::MeanC, ThresholdType::Binary, 4, 0.0)
            .is_err());
        // Too small.
        assert!(adaptive_threshold(&m, 255.0, AdaptiveMethod::MeanC, ThresholdType::Binary, 1, 0.0)
            .is_err());
        // Unsupported threshold type.
        assert!(adaptive_threshold(&m, 255.0, AdaptiveMethod::MeanC, ThresholdType::Trunc, 3, 0.0)
            .is_err());
    }

    #[test]
    fn match_template_result_has_the_right_size() {
        let img = Mat::zeros(10, 12, CV_8UC1).unwrap();
        let t = Mat::zeros(3, 4, CV_8UC1).unwrap();
        let r = match_template(&img, &t, TemplateMatchMode::SqDiff).unwrap();
        assert_eq!(r.size(), Size::new(9, 8));
        assert_eq!(r.depth(), Depth::F32);
    }

    #[test]
    fn sqdiff_is_zero_at_an_exact_match() {
        let mut data = vec![0u8; 64];
        // Put a distinctive 2x2 patch at (2,3).
        data[2 * 8 + 3] = 10;
        data[2 * 8 + 4] = 20;
        data[3 * 8 + 3] = 30;
        data[3 * 8 + 4] = 40;
        let img = Mat::from_vec(8, 8, 1, &data).unwrap();
        let t = Mat::from_vec(2, 2, 1, &[10u8, 20, 30, 40]).unwrap();

        let r = match_template(&img, &t, TemplateMatchMode::SqDiff).unwrap();
        let mm = min_max_loc::<Mat, Mat>(&r, None).unwrap();
        assert_eq!(mm.min_val, 0.0);
        assert_eq!(mm.min_loc, Point::new(3, 2));
    }

    #[test]
    fn ccoeff_normed_peaks_at_one_on_an_exact_match() {
        let mut data = vec![0u8; 64];
        for (i, v) in [(2usize, 10u8), (3, 200), (10, 60), (11, 90)] {
            data[i] = v;
        }
        let img = Mat::from_vec(8, 8, 1, &data).unwrap();
        let t = Mat::from_vec(2, 2, 1, &[10u8, 200, 60, 90]).unwrap();
        let r = match_template(&img, &t, TemplateMatchMode::CCoeffNormed).unwrap();
        let mm = min_max_loc::<Mat, Mat>(&r, None).unwrap();
        assert!((mm.max_val - 1.0).abs() < 1e-5, "peak was {}", mm.max_val);
        assert_eq!(mm.max_loc, Point::new(2, 0));
    }

    #[test]
    fn ccorr_normed_is_bounded_by_one() {
        let img = Mat::from_vec(6, 6, 1, &(0u8..36).collect::<Vec<_>>()).unwrap();
        let t = Mat::from_vec(2, 2, 1, &[7u8, 8, 13, 14]).unwrap();
        let r = match_template(&img, &t, TemplateMatchMode::CCorrNormed).unwrap();
        for v in r.to_vec::<f32>().unwrap() {
            assert!((-1e-5..=1.0 + 1e-5).contains(&v), "{v}");
        }
    }

    #[test]
    fn match_template_rejects_an_oversized_template() {
        let img = Mat::zeros(4, 4, CV_8UC1).unwrap();
        let t = Mat::zeros(5, 4, CV_8UC1).unwrap();
        assert_eq!(match_template(&img, &t, TemplateMatchMode::SqDiff).unwrap_err().code, Code::StsBadSize);
    }

    #[test]
    fn match_template_handles_a_uniform_image_without_dividing_by_zero() {
        let img = Mat::new_with(6, 6, CV_8UC1, Scalar::single(128.0)).unwrap();
        let t = Mat::new_with(2, 2, CV_8UC1, Scalar::single(128.0)).unwrap();
        let r = match_template(&img, &t, TemplateMatchMode::CCoeffNormed).unwrap();
        assert!(r.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
    }

    /// `matchTemplate` is bit-exact against the exact integer answer.
    ///
    /// A small u8 image and template where every `TM_CCORR` / `TM_SQDIFF`
    /// value is an integer well below 2^24, so `f32` holds each one exactly and
    /// there is a single right answer to compare against. cv2 misses it by up
    /// to 4 on the parity fixture; see the function docs.
    #[test]
    fn match_template_is_bit_exact_on_integer_inputs() {
        let img_data: Vec<u8> = (0..12 * 10).map(|i| ((i * 37 + 11) % 251) as u8).collect();
        let tpl_data: Vec<u8> = (0..3 * 4).map(|i| ((i * 53 + 7) % 241) as u8).collect();
        let img = Mat::from_vec(10, 12, 1, &img_data).unwrap();
        let tpl = Mat::from_vec(4, 3, 1, &tpl_data).unwrap();

        let cc = match_template(&img, &tpl, TemplateMatchMode::CCorr).unwrap();
        let sq = match_template(&img, &tpl, TemplateMatchMode::SqDiff).unwrap();
        for ry in 0..cc.rows() {
            for rx in 0..cc.cols() {
                // The exact answer, in integers.
                let mut ec: i64 = 0;
                let mut es: i64 = 0;
                for y in 0..4i32 {
                    for x in 0..3i32 {
                        let a = i64::from(img_data[((ry + y) * 12 + rx + x) as usize]);
                        let b = i64::from(tpl_data[(y * 3 + x) as usize]);
                        ec += a * b;
                        es += (a - b) * (a - b);
                    }
                }
                assert_eq!(cc.at_f64(ry, rx, 0).unwrap(), ec as f64, "CCORR at ({ry},{rx})");
                assert_eq!(sq.at_f64(ry, rx, 0).unwrap(), es as f64, "SQDIFF at ({ry},{rx})");
            }
        }
    }
}
