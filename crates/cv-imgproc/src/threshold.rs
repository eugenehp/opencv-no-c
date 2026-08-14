//! `cv::threshold` and its automatic threshold selectors
//! (`modules/imgproc/src/thresh.cpp`).

use cv_core::{cv_assert, cv_error, Code, Depth, Mat, MatLike, Result};

/// `cv::ThresholdTypes` — the base operations, with upstream's values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ThresholdType {
    /// `THRESH_BINARY` — `v > thresh ? maxval : 0`.
    Binary = 0,
    /// `THRESH_BINARY_INV` — `v > thresh ? 0 : maxval`.
    BinaryInv = 1,
    /// `THRESH_TRUNC` — `v > thresh ? thresh : v`.
    Trunc = 2,
    /// `THRESH_TOZERO` — `v > thresh ? v : 0`.
    ToZero = 3,
    /// `THRESH_TOZERO_INV` — `v > thresh ? 0 : v`.
    ToZeroInv = 4,
}

/// How the threshold value is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdMethod {
    /// Use the caller-supplied value.
    Manual,
    /// `THRESH_OTSU` — maximise between-class variance. `CV_8UC1` only.
    Otsu,
    /// `THRESH_TRIANGLE` — the triangle/Zack method. `CV_8UC1` only.
    Triangle,
}

/// Result of [`threshold`]: the output image plus the threshold actually used.
#[derive(Debug)]
pub struct Threshold {
    /// Thresholded image.
    pub dst: Mat,
    /// The threshold applied — the computed value for Otsu/Triangle.
    pub thresh: f64,
}

/// `cv::threshold(src, dst, thresh, maxval, type)`.
///
/// Returns the applied threshold alongside the image, mirroring upstream's
/// `double` return value, which matters for `THRESH_OTSU`/`THRESH_TRIANGLE`.
pub fn threshold<S: MatLike + ?Sized>(
    src: &S,
    thresh: f64,
    maxval: f64,
    ttype: ThresholdType,
    method: ThresholdMethod,
) -> Result<Threshold> {
    cv_assert!(!src.is_empty(), "threshold requires a non-empty source");

    let thresh = match method {
        ThresholdMethod::Manual => thresh,
        ThresholdMethod::Otsu | ThresholdMethod::Triangle => {
            if src.depth() != Depth::U8 || src.channels() != 1 {
                return cv_error!(
                    Code::StsUnsupportedFormat,
                    "automatic thresholding requires CV_8UC1, got {}",
                    src.mat_type()
                );
            }
            let hist = histogram_8u(src)?;
            match method {
                ThresholdMethod::Otsu => otsu_threshold(&hist, src.total()),
                _ => triangle_threshold(&mut hist.clone()),
            }
        }
    };

    // One bulk widen, one tight loop, one bulk store — instead of two
    // depth-dispatching accessor calls per channel per pixel.
    let mut buf = cv_core::to_f64(src)?;
    for v in buf.iter_mut() {
        let x = *v;
        *v = match ttype {
            ThresholdType::Binary => {
                if x > thresh {
                    maxval
                } else {
                    0.0
                }
            }
            ThresholdType::BinaryInv => {
                if x > thresh {
                    0.0
                } else {
                    maxval
                }
            }
            ThresholdType::Trunc => {
                if x > thresh {
                    thresh
                } else {
                    x
                }
            }
            ThresholdType::ToZero => {
                if x > thresh {
                    x
                } else {
                    0.0
                }
            }
            ThresholdType::ToZeroInv => {
                if x > thresh {
                    0.0
                } else {
                    x
                }
            }
        };
    }
    let dst = cv_core::mat_from_f64(src.rows(), src.cols(), src.mat_type(), &buf)?;
    Ok(Threshold { dst, thresh })
}

/// 256-bin histogram of a `CV_8UC1` image.
fn histogram_8u<S: MatLike + ?Sized>(src: &S) -> Result<Vec<i32>> {
    let mut h = vec![0i32; 256];
    for y in 0..src.rows() {
        for &v in src.row_as::<u8>(y)? {
            h[v as usize] += 1;
        }
    }
    Ok(h)
}

/// `getThreshVal_Otsu_8u` — maximise between-class variance.
fn otsu_threshold(h: &[i32], total: usize) -> f64 {
    const N: usize = 256;
    let scale = 1.0 / total as f64;

    let mut mu: f64 = h.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();
    mu *= scale;

    let mut mu1 = 0.0f64;
    let mut q1 = 0.0f64;
    let mut max_sigma = 0.0f64;
    let mut max_val = 0.0f64;
    // f32 epsilon, matching upstream's FLT_EPSILON guard.
    let eps = f32::EPSILON as f64;

    for (i, &count) in h.iter().enumerate().take(N) {
        let p_i = count as f64 * scale;
        mu1 *= q1;
        q1 += p_i;
        let q2 = 1.0 - q1;
        if q1.min(q2) < eps || q1.max(q2) > 1.0 - eps {
            continue;
        }
        mu1 = (mu1 + i as f64 * p_i) / q1;
        let mu2 = (mu - q1 * mu1) / q2;
        let sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
        if sigma > max_sigma {
            max_sigma = sigma;
            max_val = i as f64;
        }
    }
    max_val
}

/// `getThreshVal_Triangle_8u` — the Zack triangle method.
fn triangle_threshold(h: &mut [i32]) -> f64 {
    const N: i32 = 256;
    let n = N as usize;

    let mut left_bound = 0usize;
    let mut right_bound = 0usize;
    let mut max_ind = 0usize;
    let mut max = 0i32;

    if let Some(i) = h.iter().position(|&v| v > 0) {
        left_bound = i;
    }
    left_bound = left_bound.saturating_sub(1);
    if let Some(i) = h[1..n].iter().rposition(|&v| v > 0) {
        right_bound = i + 1;
    }
    if right_bound < n - 1 {
        right_bound += 1;
    }
    for (i, &v) in h.iter().enumerate().take(n) {
        if v > max {
            max = v;
            max_ind = i;
        }
    }

    // Work on the side with the longer tail; flip if it is the left one.
    let mut is_flipped = false;
    if (max_ind as i32 - left_bound as i32) < (right_bound as i32 - max_ind as i32) {
        is_flipped = true;
        h[..n].reverse();
        left_bound = n - 1 - right_bound;
        max_ind = n - 1 - max_ind;
    }

    let mut thresh = left_bound;
    if max_ind > left_bound {
        let a = max as f64;
        let b = left_bound as f64 - max_ind as f64;
        let mut dist = 0.0f64;
        for (i, &hv) in h.iter().enumerate().take(max_ind + 1).skip(left_bound + 1) {
            let tempdist = a * i as f64 + b * hv as f64;
            if tempdist > dist {
                dist = tempdist;
                thresh = i;
            }
        }
        // `thresh` is at least `left_bound + 1` here, so this cannot underflow.
        thresh -= 1;
    }

    if is_flipped {
        thresh = n - 1 - thresh;
    }
    thresh as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::MatType;

    fn u8m(cols: i32, v: &[u8]) -> Mat {
        Mat::from_vec(1, cols, 1, v).unwrap()
    }

    #[test]
    fn type_values_match_opencv() {
        assert_eq!(ThresholdType::Binary as i32, 0);
        assert_eq!(ThresholdType::BinaryInv as i32, 1);
        assert_eq!(ThresholdType::Trunc as i32, 2);
        assert_eq!(ThresholdType::ToZero as i32, 3);
        assert_eq!(ThresholdType::ToZeroInv as i32, 4);
    }

    #[test]
    fn binary_is_strictly_greater_than() {
        let m = u8m(4, &[0, 128, 129, 255]);
        let r = threshold(&m, 128.0, 255.0, ThresholdType::Binary, ThresholdMethod::Manual).unwrap();
        // 128 is NOT above the threshold; the comparison is strict.
        assert_eq!(r.dst.to_vec::<u8>().unwrap(), vec![0, 0, 255, 255]);
    }

    #[test]
    fn all_five_types_behave_as_documented() {
        let m = u8m(3, &[50, 100, 200]);
        let t = 100.0;
        let got = |ty| threshold(&m, t, 255.0, ty, ThresholdMethod::Manual).unwrap().dst.to_vec::<u8>().unwrap();
        assert_eq!(got(ThresholdType::Binary), vec![0, 0, 255]);
        assert_eq!(got(ThresholdType::BinaryInv), vec![255, 255, 0]);
        assert_eq!(got(ThresholdType::Trunc), vec![50, 100, 100]);
        assert_eq!(got(ThresholdType::ToZero), vec![0, 0, 200]);
        assert_eq!(got(ThresholdType::ToZeroInv), vec![50, 100, 0]);
    }

    #[test]
    fn otsu_splits_a_clean_bimodal_histogram() {
        // Half the pixels at 20, half at 200: the optimal split lies between.
        let mut data = vec![20u8; 50];
        data.extend(vec![200u8; 50]);
        let m = u8m(100, &data);
        let r = threshold(&m, 0.0, 255.0, ThresholdType::Binary, ThresholdMethod::Otsu).unwrap();
        assert!((20.0..200.0).contains(&r.thresh), "otsu picked {}", r.thresh);
        assert_eq!(r.dst.to_vec::<u8>().unwrap()[0], 0);
        assert_eq!(r.dst.to_vec::<u8>().unwrap()[99], 255);
    }

    #[test]
    fn otsu_returns_the_applied_threshold() {
        let mut data = vec![0u8; 10];
        data.extend(vec![255u8; 10]);
        let m = u8m(20, &data);
        let r = threshold(&m, 999.0, 255.0, ThresholdType::Binary, ThresholdMethod::Otsu).unwrap();
        // The supplied 999 is ignored in favour of the computed value.
        assert!(r.thresh < 255.0);
    }

    #[test]
    fn triangle_runs_on_a_skewed_histogram() {
        let mut data = vec![10u8; 90];
        data.extend(vec![180u8; 10]);
        let m = u8m(100, &data);
        let r = threshold(&m, 0.0, 255.0, ThresholdType::Binary, ThresholdMethod::Triangle).unwrap();
        assert!((0.0..=255.0).contains(&r.thresh));
    }

    #[test]
    fn automatic_methods_reject_non_8uc1() {
        let m = Mat::zeros(2, 2, MatType::new(Depth::F32, 1)).unwrap();
        let e = threshold(&m, 0.0, 1.0, ThresholdType::Binary, ThresholdMethod::Otsu).unwrap_err();
        assert_eq!(e.code, Code::StsUnsupportedFormat);

        let m3 = Mat::zeros(2, 2, MatType::new(Depth::U8, 3)).unwrap();
        assert!(threshold(&m3, 0.0, 1.0, ThresholdType::Binary, ThresholdMethod::Otsu).is_err());
    }

    #[test]
    fn float_images_threshold_without_saturation() {
        let m = Mat::from_vec(1, 3, 1, &[0.1f32, 0.5, 0.9]).unwrap();
        let r = threshold(&m, 0.5, 1.0, ThresholdType::Binary, ThresholdMethod::Manual).unwrap();
        assert_eq!(r.dst.to_vec::<f32>().unwrap(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn multichannel_thresholds_each_channel() {
        let m = Mat::from_vec(1, 1, 3, &[10u8, 100, 200]).unwrap();
        let r = threshold(&m, 50.0, 255.0, ThresholdType::Binary, ThresholdMethod::Manual).unwrap();
        assert_eq!(r.dst.pixel::<u8>(0, 0).unwrap(), &[0, 255, 255]);
    }
}
