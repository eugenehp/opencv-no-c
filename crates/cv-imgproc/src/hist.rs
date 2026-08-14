//! Histograms (`modules/imgproc/src/histogram.cpp`).

use cv_core::{cv_assert, cv_error, Code, Depth, Mat, MatLike, MatLikeMut, Result};

/// `cv::calcHist` for the single-channel, uniform, no-mask case.
///
/// Returns a `hist_size x 1` `CV_32F` matrix. Values outside `range` are
/// dropped, and the bin index is `floor((v - lo) * hist_size / (hi - lo))`
/// clamped to the last bin, matching upstream's uniform binning.
pub fn calc_hist<S: MatLike + ?Sized, M: MatLike + ?Sized>(
    src: &S,
    channel: i32,
    hist_size: i32,
    range: (f32, f32),
    mask: Option<&M>,
) -> Result<Mat> {
    cv_assert!(hist_size > 0, "hist_size must be positive, got {hist_size}");
    cv_assert!(
        channel >= 0 && channel < src.channels(),
        "channel {channel} out of range 0..{}",
        src.channels()
    );
    let (lo, hi) = (range.0 as f64, range.1 as f64);
    if hi <= lo || hi.is_nan() || lo.is_nan() {
        return cv_error!(Code::StsBadArg, "range must be increasing, got {lo}..{hi}");
    }
    if let Some(m) = mask {
        cv_assert!(m.size() == src.size(), "mask size must match the source");
        cv_assert!(m.mat_type() == cv_core::types::CV_8UC1, "mask must be CV_8UC1");
    }

    let scale = hist_size as f64 / (hi - lo);
    let mut counts = vec![0f32; hist_size as usize];
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            if let Some(m) = mask {
                if m.at::<u8>(y, x)? == 0 {
                    continue;
                }
            }
            let v = src.at_f64(y, x, channel)?;
            if v < lo || v >= hi {
                continue;
            }
            let idx = ((v - lo) * scale) as i32;
            counts[idx.clamp(0, hist_size - 1) as usize] += 1.0;
        }
    }
    Mat::from_vec(hist_size, 1, 1, &counts)
}

/// `cv::equalizeHist(src, dst)` — `CV_8UC1` only, as upstream.
///
/// A transcription of the upstream lookup-table construction, including the
/// detail that the first populated bin maps to 0 and the scale divides by
/// `total - hist[first]` rather than by `total`. Dividing by `total` is the
/// common mistake and shifts every output level.
pub fn equalize_hist<S: MatLike + ?Sized>(src: &S) -> Result<Mat> {
    if src.depth() != Depth::U8 || src.channels() != 1 {
        return cv_error!(
            Code::StsUnsupportedFormat,
            "equalizeHist requires CV_8UC1, got {}",
            src.mat_type()
        );
    }
    cv_assert!(!src.is_empty(), "equalizeHist requires a non-empty source");
    const HIST_SZ: usize = 256;

    let mut hist = [0i32; HIST_SZ];
    for y in 0..src.rows() {
        for &v in src.row_as::<u8>(y)? {
            hist[v as usize] += 1;
        }
    }
    let total = src.total() as i32;

    // First populated bin.
    let first = hist.iter().position(|&c| c > 0).unwrap_or(0);

    let mut dst = Mat::zeros(src.rows(), src.cols(), src.mat_type())?;
    if hist[first] == total {
        // A constant image maps entirely onto its own level.
        cv_core::arithm::set_to(&mut dst, cv_core::Scalar::single(first as f64))?;
        return Ok(dst);
    }

    let scale = (HIST_SZ as f32 - 1.0) / (total - hist[first]) as f32;
    let mut lut = [0u8; HIST_SZ];
    let mut sum = 0i32;
    for (i, slot) in lut.iter_mut().enumerate().skip(first + 1) {
        sum += hist[i];
        *slot = cv_core::saturate::Elem::from_f64((sum as f32 * scale) as f64);
    }

    for y in 0..src.rows() {
        let row: Vec<u8> = src.row_as::<u8>(y)?.to_vec();
        let d = dst.row_as_mut::<u8>(y)?;
        for (o, &v) in d.iter_mut().zip(row.iter()) {
            *o = lut[v as usize];
        }
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::{CV_32FC1, CV_8UC1};

    fn u8m(cols: i32, v: &[u8]) -> Mat {
        Mat::from_vec(1, cols, 1, v).unwrap()
    }

    #[test]
    fn calc_hist_counts_into_uniform_bins() {
        let m = u8m(6, &[0, 10, 10, 200, 255, 255]);
        let h = calc_hist::<Mat, Mat>(&m, 0, 256, (0.0, 256.0), None).unwrap();
        assert_eq!(h.size(), cv_core::Size::new(1, 256));
        let v = h.to_vec::<f32>().unwrap();
        assert_eq!(v[0], 1.0);
        assert_eq!(v[10], 2.0);
        assert_eq!(v[200], 1.0);
        assert_eq!(v[255], 2.0);
        assert_eq!(v.iter().sum::<f32>(), 6.0);
    }

    #[test]
    fn calc_hist_supports_coarse_binning() {
        let m = u8m(4, &[0, 63, 64, 255]);
        let h = calc_hist::<Mat, Mat>(&m, 0, 4, (0.0, 256.0), None).unwrap();
        assert_eq!(h.to_vec::<f32>().unwrap(), vec![2.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn calc_hist_drops_values_outside_the_range() {
        let m = u8m(4, &[5, 50, 100, 200]);
        let h = calc_hist::<Mat, Mat>(&m, 0, 2, (50.0, 150.0), None).unwrap();
        // 5 and 200 fall outside; 50 and 100 land in bin 0 and 1.
        assert_eq!(h.to_vec::<f32>().unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn calc_hist_honours_a_mask() {
        let m = u8m(4, &[10, 10, 20, 20]);
        let mask = u8m(4, &[255, 0, 255, 0]);
        let h = calc_hist(&m, 0, 256, (0.0, 256.0), Some(&mask)).unwrap();
        let v = h.to_vec::<f32>().unwrap();
        assert_eq!(v[10], 1.0);
        assert_eq!(v[20], 1.0);
    }

    #[test]
    fn calc_hist_selects_the_requested_channel() {
        let m = Mat::from_vec(1, 2, 3, &[1u8, 2, 3, 1, 2, 3]).unwrap();
        let h = calc_hist::<Mat, Mat>(&m, 2, 256, (0.0, 256.0), None).unwrap();
        assert_eq!(h.to_vec::<f32>().unwrap()[3], 2.0);
    }

    #[test]
    fn calc_hist_rejects_a_bad_range() {
        let m = u8m(2, &[0, 1]);
        assert!(calc_hist::<Mat, Mat>(&m, 0, 4, (10.0, 10.0), None).is_err());
    }

    #[test]
    fn equalize_hist_spreads_a_narrow_range_to_full_scale() {
        let m = u8m(4, &[100, 101, 102, 103]);
        let e = equalize_hist(&m).unwrap();
        let v = e.to_vec::<u8>().unwrap();
        assert_eq!(v[0], 0);
        assert_eq!(v[3], 255);
        // Monotonic non-decreasing.
        assert!(v.windows(2).all(|w| w[0] <= w[1]), "{v:?}");
    }

    #[test]
    fn equalize_hist_of_a_constant_image_is_that_constant() {
        let m = u8m(5, &[42; 5]);
        let e = equalize_hist(&m).unwrap();
        assert_eq!(e.to_vec::<u8>().unwrap(), vec![42; 5]);
    }

    #[test]
    fn equalize_hist_maps_the_lowest_level_to_zero() {
        // The upstream formula pins the first populated bin to 0 and divides by
        // (total - hist[first]); dividing by total instead shifts everything.
        let m = u8m(8, &[0, 0, 0, 0, 255, 255, 255, 255]);
        let e = equalize_hist(&m).unwrap();
        let v = e.to_vec::<u8>().unwrap();
        assert_eq!(&v[0..4], &[0, 0, 0, 0]);
        assert_eq!(&v[4..8], &[255, 255, 255, 255]);
    }

    #[test]
    fn equalize_hist_preserves_shape() {
        let m = Mat::from_vec(3, 4, 1, &(0u8..12).collect::<Vec<_>>()).unwrap();
        let e = equalize_hist(&m).unwrap();
        assert_eq!(e.size(), m.size());
        assert_eq!(e.mat_type(), CV_8UC1);
    }

    #[test]
    fn equalize_hist_rejects_non_8uc1() {
        let f = Mat::zeros(2, 2, CV_32FC1).unwrap();
        assert_eq!(equalize_hist(&f).unwrap_err().code, Code::StsUnsupportedFormat);
        let c3 = Mat::zeros(2, 2, cv_core::types::CV_8UC3).unwrap();
        assert!(equalize_hist(&c3).is_err());
    }
}
