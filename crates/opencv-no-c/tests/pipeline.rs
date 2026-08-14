//! End-to-end tests across crate boundaries.
//!
//! These exercise realistic multi-stage pipelines rather than single functions,
//! which is where mismatched conventions between modules (channel order, border
//! handling, saturation) tend to surface.

use opencv_no_c::core::{
    absdiff, count_non_zero, min_max_loc, norm, split, BorderType, Depth, NormType, Rect, Scalar,
};
use opencv_no_c::core as cv_core;
use opencv_no_c::imgproc::{
    blur, cvt_color, gaussian_blur, median_blur, resize, threshold, ColorConversion, Interpolation,
    ThresholdMethod, ThresholdType,
};
use opencv_no_c::prelude::*;
use opencv_no_c::Result;

/// A 64x64 BGR image with a bright square in the middle.
fn test_image() -> Result<Mat> {
    let mut img = Mat::new_with(64, 64, types::CV_8UC3, Scalar::new(20.0, 30.0, 40.0, 0.0))?;
    let mut roi = img.roi_mut(Rect::new(16, 16, 32, 32))?;
    opencv_no_c::core::arithm::set_to(&mut roi, Scalar::new(200.0, 210.0, 220.0, 0.0))?;
    Ok(img)
}

#[test]
fn bgr_to_gray_to_threshold_isolates_the_square() -> Result<()> {
    let img = test_image()?;
    let gray = cvt_color(&img, ColorConversion::Bgr2Gray)?;
    assert_eq!(gray.channels(), 1);

    let t = threshold(&gray, 0.0, 255.0, ThresholdType::Binary, ThresholdMethod::Otsu)?;
    // The square is 32x32 = 1024 pixels out of 64x64 = 4096.
    assert_eq!(count_non_zero(&t.dst)?, 1024);
    Ok(())
}

#[test]
fn resize_roundtrip_preserves_a_flat_region() -> Result<()> {
    let img = test_image()?;
    let small = resize(&img, Size::new(32, 32), 0.0, 0.0, Interpolation::Area)?;
    assert_eq!(small.size(), Size::new(32, 32));

    // The centre of the square is far from any edge, so area averaging leaves
    // it untouched.
    assert_eq!(small.pixel::<u8>(16, 16)?, &[200, 210, 220]);
    // As does the flat background corner.
    assert_eq!(small.pixel::<u8>(0, 0)?, &[20, 30, 40]);
    Ok(())
}

#[test]
fn blurring_a_flat_image_is_a_no_op_across_all_filters() -> Result<()> {
    let flat = Mat::new_with(32, 32, types::CV_8UC3, Scalar::new(77.0, 88.0, 99.0, 0.0))?;
    for out in [
        blur(&flat, Size::new(5, 5), BorderType::Reflect101)?,
        gaussian_blur(&flat, Size::new(5, 5), 1.4, 0.0, BorderType::Reflect101)?,
        median_blur(&flat, 5)?,
    ] {
        assert_eq!(norm(&absdiff(&flat, &out)?, NormType::Inf)?, 0.0);
    }
    Ok(())
}

#[test]
fn hsv_roundtrip_stays_within_one_level() -> Result<()> {
    let img = test_image()?;
    let hsv = cvt_color(&img, ColorConversion::Bgr2Hsv)?;
    let back = cvt_color(&hsv, ColorConversion::Hsv2Bgr)?;

    // 8-bit HSV quantises hue to 0..180, so a perfect roundtrip is not
    // expected; upstream's own tests allow a small tolerance here too.
    let diff = absdiff(&img, &back)?;
    let worst = norm(&diff, NormType::Inf)?;
    assert!(worst <= 2.0, "worst channel error was {worst}");
    Ok(())
}

#[test]
fn split_recovers_each_channel_of_a_converted_image() -> Result<()> {
    let img = test_image()?;
    let ycc = cvt_color(&img, ColorConversion::Bgr2YCrCb)?;
    let planes = split(&ycc)?;
    assert_eq!(planes.len(), 3);

    // The luma plane must be brighter inside the square than outside.
    let inside = planes[0].at::<u8>(32, 32)?;
    let outside = planes[0].at::<u8>(0, 0)?;
    assert!(inside > outside, "{inside} should exceed {outside}");
    Ok(())
}

#[test]
fn operations_compose_over_a_borrowed_roi() -> Result<()> {
    let img = test_image()?;
    // Take a view spanning both the background and the square, and confirm the
    // pipeline reads through the stride correctly.
    let view = img.roi(Rect::new(8, 8, 32, 32))?;
    assert!(!view.is_continuous());

    let gray = cvt_color(&view, ColorConversion::Bgr2Gray)?;
    assert_eq!(gray.size(), Size::new(32, 32));

    let mm = min_max_loc::<Mat, Mat>(&gray, None)?;
    // Background luma ~= 31, square luma ~= 209; both must appear.
    assert!(mm.min_val < 60.0, "min was {}", mm.min_val);
    assert!(mm.max_val > 180.0, "max was {}", mm.max_val);
    Ok(())
}

#[test]
fn float_pipeline_preserves_precision() -> Result<()> {
    let src = Mat::from_vec(4, 4, 1, &(0..16).map(|i| i as f32 / 16.0).collect::<Vec<_>>())?;
    let big = resize(&src, Size::new(8, 8), 0.0, 0.0, Interpolation::Linear)?;
    assert_eq!(big.depth(), Depth::F32);

    let vals = big.to_vec::<f32>()?;
    assert!(vals.iter().all(|v| (0.0..=1.0).contains(v)), "values escaped [0,1]");
    // Interpolation must not collapse the gradient.
    assert!(vals.iter().cloned().fold(f32::MIN, f32::max) > 0.8);
    Ok(())
}

#[test]
fn version_constants_agree_across_crates() {
    assert_eq!(opencv_no_c::CV_VERSION, "4.15.0-dev");
    assert_eq!(opencv_no_c::CV_VERSION, opencv_no_c::core::CV_VERSION);
    assert_eq!(opencv_no_c::CV_UPSTREAM_COMMIT.len(), 40);
}

#[test]
fn full_pipeline_from_file_to_file() -> Result<()> {
    use opencv_no_c::imgcodecs::{imdecode, imencode, ImreadMode};
    use opencv_no_c::imgproc::{canny, get_structuring_element, dilate, MorphShape};
    use cv_core::Point;

    // Encode a synthetic scene, decode it back, and run a real pipeline over it.
    let scene = {
        let mut m = Mat::new_with(48, 48, types::CV_8UC3, Scalar::new(30.0, 30.0, 30.0, 0.0))?;
        let mut roi = m.roi_mut(Rect::new(12, 12, 24, 24))?;
        opencv_no_c::core::arithm::set_to(&mut roi, Scalar::new(220.0, 220.0, 220.0, 0.0))?;
        m
    };

    let encoded = imencode(".bmp", &scene)?;
    let decoded = imdecode(&encoded, ImreadMode::Grayscale)?;
    assert_eq!(decoded.size(), Size::new(48, 48));
    assert_eq!(decoded.channels(), 1);

    let edges = canny(&decoded, 50.0, 150.0, 3, false)?;
    let before = opencv_no_c::core::count_non_zero(&edges)?;
    assert!(before > 0, "canny found no edges in the decoded image");

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(3, 3), Point::new(-1, -1))?;
    let thick = dilate(&edges, &kernel, Point::new(-1, -1), 1, BorderType::Replicate)?;
    assert!(opencv_no_c::core::count_non_zero(&thick)? > before);

    // Round-trip the result back out as a PGM.
    let out = imencode(".pgm", &thick)?;
    let reread = imdecode(&out, ImreadMode::Unchanged)?;
    assert_eq!(reread.to_vec::<u8>()?, thick.to_vec::<u8>()?);
    Ok(())
}
