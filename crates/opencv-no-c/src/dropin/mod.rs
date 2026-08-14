//! Upstream's names, and errors that unwind instead of being returned.
//!
//! Enabled by the `dropin` feature. Every function here is a thin forward to
//! the snake_case original with one difference: an `Err` becomes a **panic
//! carrying the [`Error`]**, so the failure propagates the way a
//! `cv::Exception` does rather than having to be threaded through a `Result`.
//!
//! ```
//! # #[cfg(feature = "dropin")] {
//! use opencv_no_c::dropin::*;
//! use opencv_no_c::core::{types::CV_8UC3, Mat};
//! use opencv_no_c::imgproc::color::ColorConversion;
//!
//! let src = Mat::zeros(64, 64, CV_8UC3).unwrap();
//! let gray = cvtColor(&src, ColorConversion::Bgr2Gray);   // no `?`, no unwrap
//! let small = resize(&gray, Size::new(32, 32), 0.0, 0.0, Interpolation::Area);
//! assert_eq!(small.rows(), 32);
//! # }
//! ```
//!
//! # Catching
//!
//! The panic payload is the `Error` itself, so it can be recovered:
//!
//! ```
//! # #[cfg(feature = "dropin")] {
//! use opencv_no_c::dropin::{catch, medianBlur};
//! use opencv_no_c::core::{types::CV_8UC1, Mat};
//!
//! let src = Mat::zeros(8, 8, CV_8UC1).unwrap();
//! // An even kernel is not a valid median window.
//! let e = catch(|| medianBlur(&src, 4)).unwrap_err();
//! assert!(!e.msg.is_empty());
//! # }
//! ```
//!
//! # What this layer is and is not
//!
//! It gives upstream's spelling and upstream's failure model. It does **not**
//! make C++ or `cv2` source compile against this crate: argument order and
//! defaulted parameters differ, output arrays are returned rather than filled
//! through an `OutputArray`, and Rust has no overloading. Treat it as a
//! translation aid that removes the two most mechanical differences, not as a
//! source-compatible shim.
//!
//! Pair it with the `compat` feature for `cv::Mat`'s storage semantics, and
//! with `abi` for a C-callable surface.

#![allow(non_snake_case)]

use core::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};

use cv_core::Error;

pub use cv_core::{
    types::{self, Depth, MatType},
    BorderType, Mat, MatLike, MatLikeMut, Point, Rect, Scalar, Size,
};
pub use cv_imgproc::color::ColorConversion;
pub use cv_imgproc::resize::Interpolation;

#[cfg(feature = "abi")]
pub mod abi;

/// Turn an `Err` into a panic carrying the [`Error`], the way `CV_Error`
/// raises a `cv::Exception`.
///
/// # Panics
///
/// On `Err`, always. That is the point.
#[inline]
pub fn throw<T>(r: cv_core::Result<T>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => std::panic::panic_any(e),
    }
}

/// Run `f`, converting a thrown [`Error`] back into an `Err` — `try`/`catch`
/// around a block of this module's calls.
///
/// A panic that is **not** an `Error` is re-raised untouched, so a genuine bug
/// is never silently turned into a library error.
///
/// # Errors
///
/// The `Error` a call inside `f` threw.
///
/// # Panics
///
/// Re-panics if the payload was not an `Error`.
pub fn catch<T, F: FnOnce() -> T>(f: F) -> cv_core::Result<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => Ok(v),
        Err(payload) => Err(downcast_or_resume(payload)),
    }
}

/// Recover the `Error` from a panic payload, or re-raise what was not ours.
fn downcast_or_resume(payload: Box<dyn Any + Send>) -> Error {
    match payload.downcast::<Error>() {
        Ok(e) => *e,
        Err(other) => std::panic::resume_unwind(other),
    }
}

/// Declare a throwing wrapper: upstream's name, the same arguments, the value
/// unwrapped.
macro_rules! throwing {
    ($(
        $(#[$meta:meta])*
        fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) -> $ret:ty = $path:path;
    )*) => {
        $(
            $(#[$meta])*
            #[inline]
            #[must_use]
            // Upstream's own signatures are this wide; narrowing them here
            // would defeat the purpose of matching them.
            #[allow(clippy::too_many_arguments)]
            pub fn $name ( $($arg : $ty),* ) -> $ret {
                throw($path($($arg),*))
            }
        )*
    };
}

throwing! {
    // --- core: element-wise arithmetic and logic.
    /// `cv::add(src1, src2, dst)`.
    fn add(a: &impl MatLike, b: &impl MatLike) -> Mat = cv_core::arithm::add;
    /// `cv::subtract(src1, src2, dst)`.
    fn subtract(a: &impl MatLike, b: &impl MatLike) -> Mat = cv_core::arithm::subtract;
    /// `cv::multiply(src1, src2, dst, scale)`.
    fn multiply(a: &impl MatLike, b: &impl MatLike, scale: f64) -> Mat = cv_core::arithm::multiply;
    /// `cv::divide(src1, src2, dst, scale)`.
    fn divide(a: &impl MatLike, b: &impl MatLike, scale: f64) -> Mat = cv_core::arithm::divide;
    /// `cv::absdiff(src1, src2, dst)`.
    fn absdiff(a: &impl MatLike, b: &impl MatLike) -> Mat = cv_core::arithm::absdiff;
    /// `cv::bitwise_and(src1, src2, dst)`.
    fn bitwise_and(a: &impl MatLike, b: &impl MatLike) -> Mat = cv_core::arithm::bitwise_and;
    /// `cv::bitwise_or(src1, src2, dst)`.
    fn bitwise_or(a: &impl MatLike, b: &impl MatLike) -> Mat = cv_core::arithm::bitwise_or;
    /// `cv::bitwise_xor(src1, src2, dst)`.
    fn bitwise_xor(a: &impl MatLike, b: &impl MatLike) -> Mat = cv_core::arithm::bitwise_xor;
    /// `cv::bitwise_not(src, dst)`.
    fn bitwise_not(src: &impl MatLike) -> Mat = cv_core::arithm::bitwise_not;
    /// `cv::inRange(src, lowerb, upperb, dst)`.
    fn inRange(src: &impl MatLike, lower: Scalar, upper: Scalar) -> Mat = cv_core::arithm::in_range;

    // --- core: layout.
    /// `cv::split(src, mv)`.
    fn split(src: &impl MatLike) -> Vec<Mat> = cv_core::arithm::split;
    /// `cv::merge(mv, dst)`.
    fn merge(planes: &[Mat]) -> Mat = cv_core::arithm::merge;
    /// `cv::flip(src, dst, flipCode)`.
    fn flip(src: &impl MatLike, flip_code: i32) -> Mat = cv_core::arithm::flip;
    /// `cv::transpose(src, dst)`.
    fn transpose(src: &impl MatLike) -> Mat = cv_core::arithm::transpose;
    /// `cv::repeat(src, ny, nx, dst)`.
    fn repeat(src: &impl MatLike, ny: i32, nx: i32) -> Mat = cv_core::manip::repeat;
    /// `cv::hconcat(src, dst)`.
    fn hconcat(parts: &[Mat]) -> Mat = cv_core::manip::hconcat;
    /// `cv::vconcat(src, dst)`.
    fn vconcat(parts: &[Mat]) -> Mat = cv_core::manip::vconcat;
    /// `cv::copyMakeBorder(src, dst, top, bottom, left, right, borderType, value)`.
    fn copyMakeBorder(
        src: &impl MatLike,
        top: i32,
        bottom: i32,
        left: i32,
        right: i32,
        border_type: BorderType,
        value: Scalar,
    ) -> Mat = cv_core::border::copy_make_border;

    // --- core: statistics and maths.
    /// `cv::countNonZero(src)`.
    fn countNonZero(src: &impl MatLike) -> i32 = cv_core::stats::count_non_zero;
    /// `cv::LUT(src, lut, dst)`.
    fn LUT(src: &impl MatLike, table: &impl MatLike) -> Mat = cv_core::mathfuncs::lut;
    /// `cv::magnitude(x, y, magnitude)`.
    fn magnitude(x: &impl MatLike, y: &impl MatLike) -> Mat = cv_core::mathfuncs::magnitude;
    /// `cv::phase(x, y, angle, angleInDegrees)`.
    fn phase(x: &impl MatLike, y: &impl MatLike, angle_in_degrees: bool) -> Mat =
        cv_core::mathfuncs::phase;
    /// `cv::cartToPolar(x, y, magnitude, angle, angleInDegrees)`.
    fn cartToPolar(x: &impl MatLike, y: &impl MatLike, angle_in_degrees: bool) -> (Mat, Mat) =
        cv_core::mathfuncs::cart_to_polar;
    /// `cv::polarToCart(magnitude, angle, x, y, angleInDegrees)`.
    fn polarToCart(mag: &impl MatLike, angle: &impl MatLike, angle_in_degrees: bool) -> (Mat, Mat) =
        cv_core::mathfuncs::polar_to_cart;
    /// `cv::sqrt(src, dst)`.
    fn sqrt(src: &impl MatLike) -> Mat = cv_core::mathfuncs::sqrt;
    /// `cv::exp(src, dst)`.
    fn exp(src: &impl MatLike) -> Mat = cv_core::mathfuncs::exp;
    /// `cv::log(src, dst)`.
    fn log(src: &impl MatLike) -> Mat = cv_core::mathfuncs::log;
    /// `cv::pow(src, power, dst)`.
    fn pow(src: &impl MatLike, power: f64) -> Mat = cv_core::mathfuncs::pow;
    /// `cv::determinant(mtx)`.
    fn determinant(m: &impl MatLike) -> f64 = cv_core::matmul::determinant;

    // --- imgproc: colour, geometry, filters.
    /// `cv::cvtColor(src, dst, code)`.
    fn cvtColor(src: &impl MatLike, code: ColorConversion) -> Mat = cv_imgproc::color::cvt_color;
    /// `cv::resize(src, dst, dsize, fx, fy, interpolation)`.
    fn resize(
        src: &impl MatLike,
        dsize: Size,
        fx: f64,
        fy: f64,
        interpolation: Interpolation,
    ) -> Mat = cv_imgproc::resize::resize;
    /// `cv::GaussianBlur(src, dst, ksize, sigmaX, sigmaY, borderType)`.
    fn GaussianBlur(
        src: &impl MatLike,
        ksize: Size,
        sigma_x: f64,
        sigma_y: f64,
        border: BorderType,
    ) -> Mat = cv_imgproc::filter::gaussian_blur;
    /// `cv::medianBlur(src, dst, ksize)`.
    fn medianBlur(src: &impl MatLike, ksize: i32) -> Mat = cv_imgproc::filter::median_blur;
    /// `cv::blur(src, dst, ksize, anchor, borderType)`.
    fn blur(src: &impl MatLike, ksize: Size, border: BorderType) -> Mat = cv_imgproc::filter::blur;
    /// `cv::bilateralFilter(src, dst, d, sigmaColor, sigmaSpace)`.
    fn bilateralFilter(src: &impl MatLike, d: i32, sigma_color: f64, sigma_space: f64) -> Mat =
        cv_imgproc::hist2::bilateral_filter;
    /// `cv::Canny(image, edges, threshold1, threshold2, apertureSize, L2gradient)`.
    fn Canny(
        src: &impl MatLike,
        threshold1: f64,
        threshold2: f64,
        aperture_size: i32,
        l2_gradient: bool,
    ) -> Mat = cv_imgproc::edge::canny;
    /// `cv::Laplacian(src, dst, ddepth, ksize, scale, delta, borderType)`.
    fn Laplacian(
        src: &impl MatLike,
        ddepth: Option<Depth>,
        ksize: i32,
        scale: f64,
        delta: f64,
        border: BorderType,
    ) -> Mat = cv_imgproc::edge::laplacian;
    /// `cv::Sobel(src, dst, ddepth, dx, dy, ksize, scale, delta, borderType)`.
    fn Sobel(
        src: &impl MatLike,
        ddepth: Option<Depth>,
        dx: i32,
        dy: i32,
        ksize: i32,
        scale: f64,
        delta: f64,
        border: BorderType,
    ) -> Mat = cv_imgproc::filter::sobel;
    /// `cv::Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType)`.
    fn Scharr(
        src: &impl MatLike,
        ddepth: Option<Depth>,
        dx: i32,
        dy: i32,
        scale: f64,
        delta: f64,
        border: BorderType,
    ) -> Mat = cv_imgproc::objects::scharr;
    /// `cv::pyrDown(src, dst, dstsize, borderType)`.
    fn pyrDown(src: &impl MatLike, border: BorderType) -> Mat = cv_imgproc::pyramids::pyr_down;
    /// `cv::pyrUp(src, dst, dstsize, borderType)`.
    fn pyrUp(src: &impl MatLike, border: BorderType) -> Mat = cv_imgproc::pyramids::pyr_up;
    /// `cv::equalizeHist(src, dst)`.
    fn equalizeHist(src: &impl MatLike) -> Mat = cv_imgproc::hist::equalize_hist;
    /// `cv::integral(src, sum, sqsum)`.
    fn integral(src: &impl MatLike) -> (Mat, Mat) = cv_imgproc::misc::integral;
    /// `cv::distanceTransform(src, dst, distanceType, maskSize)`.
    fn distanceTransform(
        src: &impl MatLike,
        metric: cv_imgproc::contours::DistanceMetric,
    ) -> Mat = cv_imgproc::contours::distance_transform;
    /// `cv::applyColorMap(src, dst, colormap)`.
    fn applyColorMap(src: &impl MatLike, cm: cv_imgproc::hist2::ColorMap) -> Mat =
        cv_imgproc::hist2::apply_color_map;
    /// `cv::connectedComponents(image, labels, connectivity)`.
    fn connectedComponents(src: &impl MatLike, connectivity: i32) -> (Mat, i32) =
        cv_imgproc::shape::connected_components;

    // --- imgproc: morphology.
    /// `cv::getStructuringElement(shape, ksize, anchor)`.
    fn getStructuringElement(
        shape: cv_imgproc::morph::MorphShape,
        ksize: Size,
        anchor: Point,
    ) -> Mat = cv_imgproc::morph::get_structuring_element;
    /// `cv::erode(src, dst, kernel, anchor, iterations, borderType)`.
    fn erode(
        src: &impl MatLike,
        kernel: &impl MatLike,
        anchor: Point,
        iterations: i32,
        border: BorderType,
    ) -> Mat = cv_imgproc::morph::erode;
    /// `cv::dilate(src, dst, kernel, anchor, iterations, borderType)`.
    fn dilate(
        src: &impl MatLike,
        kernel: &impl MatLike,
        anchor: Point,
        iterations: i32,
        border: BorderType,
    ) -> Mat = cv_imgproc::morph::dilate;
    /// `cv::morphologyEx(src, dst, op, kernel, anchor, iterations, borderType)`.
    fn morphologyEx(
        src: &impl MatLike,
        op: cv_imgproc::morph::MorphType,
        kernel: &impl MatLike,
        anchor: Point,
        iterations: i32,
        border: BorderType,
    ) -> Mat = cv_imgproc::morph::morphology_ex;

    // --- imgproc: warps.
    /// `cv::warpAffine(src, dst, M, dsize, flags, borderMode, borderValue)`.
    fn warpAffine(
        src: &impl MatLike,
        m: cv_imgproc::warp::Affine2x3,
        dsize: Size,
        interpolation: Interpolation,
        inverse_map: bool,
        border: BorderType,
        border_value: Scalar,
    ) -> Mat = cv_imgproc::warp::warp_affine;
    /// `cv::warpPerspective(src, dst, M, dsize, flags, borderMode, borderValue)`.
    fn warpPerspective(
        src: &impl MatLike,
        m: cv_imgproc::warp::Perspective3x3,
        dsize: Size,
        interpolation: Interpolation,
        inverse_map: bool,
        border: BorderType,
        border_value: Scalar,
    ) -> Mat = cv_imgproc::warp::warp_perspective;
    /// `cv::remap(src, dst, map1, map2, interpolation, borderMode, borderValue)`.
    fn remap(
        src: &impl MatLike,
        map_x: &impl MatLike,
        map_y: &impl MatLike,
        interpolation: Interpolation,
        border: BorderType,
        border_value: Scalar,
    ) -> Mat = cv_imgproc::warp::remap;

    // --- imgproc: contours and shape.
    /// `cv::findContours(image, contours, hierarchy, mode, method)`.
    fn findContours(
        image: &impl MatLike,
        mode: cv_imgproc::contours::RetrievalMode,
        method: cv_imgproc::contours::ContourApproximation,
    ) -> (Vec<Vec<Point>>, Vec<cv_imgproc::contours::Hierarchy>) =
        cv_imgproc::contours::find_contours;
    /// `cv::boundingRect(array)`.
    fn boundingRect(points: &[Point]) -> Rect = cv_imgproc::shape::bounding_rect;
    /// `cv::approxPolyDP(curve, approxCurve, epsilon, closed)`.
    fn approxPolyDP(points: &[Point], epsilon: f64, closed: bool) -> Vec<Point> =
        cv_imgproc::shape::approx_poly_dp;
    /// `cv::minAreaRect(points)`.
    fn minAreaRect(pts: &[Point]) -> cv_imgproc::fitting::RotatedRect =
        cv_imgproc::fitting::min_area_rect;

    // --- imgcodecs.
    /// `cv::imdecode(buf, flags)`.
    fn imdecode(data: &[u8], mode: cv_imgcodecs::ImreadMode) -> Mat = cv_imgcodecs::imdecode;
    /// `cv::imencode(ext, img, buf, params)`.
    fn imencode(ext: &str, img: &impl MatLike) -> Vec<u8> = cv_imgcodecs::imencode;
}

// Functions upstream spells the same as this port, or that already return a
// plain value, are re-exported rather than wrapped.
pub use cv_imgproc::shape::{arc_length as arcLength, contour_area as contourArea};
pub use cv_imgproc::warp::get_rotation_matrix_2d as getRotationMatrix2D;

/// `cv::threshold(src, dst, thresh, maxval, type)` — returns the image and the
/// threshold actually applied, which is what upstream's `double` return is for.
///
/// # Panics
///
/// If the operation fails; see the module docs.
#[must_use]
pub fn threshold(
    src: &impl MatLike,
    thresh: f64,
    maxval: f64,
    ttype: cv_imgproc::threshold::ThresholdType,
    method: cv_imgproc::threshold::ThresholdMethod,
) -> (Mat, f64) {
    let t = throw(cv_imgproc::threshold::threshold(src, thresh, maxval, ttype, method));
    (t.dst, t.thresh)
}

/// `cv::minMaxLoc(src, minVal, maxVal, minLoc, maxLoc, mask)`.
///
/// # Panics
///
/// If the operation fails; see the module docs.
#[must_use]
pub fn minMaxLoc(src: &impl MatLike) -> cv_core::stats::MinMaxLoc {
    throw(cv_core::stats::min_max_loc::<_, Mat>(src, None))
}

/// `cv::meanStdDev(src, mean, stddev, mask)`.
///
/// # Panics
///
/// If the operation fails; see the module docs.
#[must_use]
pub fn meanStdDev(src: &impl MatLike) -> (Scalar, Scalar) {
    throw(cv_core::stats::mean_std_dev::<_, Mat>(src, None))
}

/// `cv::Mat::convertTo(dst, rtype, alpha, beta)`.
///
/// # Panics
///
/// If the operation fails; see the module docs.
#[must_use]
pub fn convertTo(src: &impl MatLike, depth: Depth, alpha: f64, beta: f64) -> Mat {
    throw(cv_core::arithm::convert_to(src, depth, alpha, beta))
}

/// `cv::Mat::setTo(value)`.
///
/// # Panics
///
/// If the operation fails; see the module docs.
pub fn setTo(dst: &mut impl MatLikeMut, value: Scalar) {
    throw(cv_core::arithm::set_to(dst, value));
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::{CV_8UC1, CV_8UC3};

    #[test]
    fn a_pipeline_reads_the_way_the_cpp_does() {
        let src = Mat::zeros(64, 48, CV_8UC3).unwrap();
        let gray = cvtColor(&src, ColorConversion::Bgr2Gray);
        let small = resize(&gray, Size::new(24, 32), 0.0, 0.0, Interpolation::Area);
        let blurred = GaussianBlur(&small, Size::new(3, 3), 0.0, 0.0, BorderType::Reflect101);
        let edges = Canny(&blurred, 50.0, 150.0, 3, false);
        assert_eq!((edges.rows(), edges.cols()), (32, 24));
    }

    #[test]
    fn a_failure_unwinds_carrying_the_error() {
        let src = Mat::zeros(8, 8, CV_8UC1).unwrap();
        let e = catch(|| medianBlur(&src, 4)).unwrap_err();
        assert!(!e.msg.is_empty(), "the thrown error kept its message");
    }

    #[test]
    fn catch_returns_the_value_when_nothing_throws() {
        let src = Mat::zeros(4, 4, CV_8UC1).unwrap();
        let n = catch(|| countNonZero(&src)).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn a_panic_that_is_not_an_error_is_not_swallowed() {
        // A genuine bug must not come back as a library error.
        let caught = std::panic::catch_unwind(|| {
            let _ = catch(|| -> i32 { panic!("not a cv::Exception") });
        });
        assert!(caught.is_err(), "a foreign panic was absorbed");
    }

    #[test]
    fn the_throwing_wrappers_agree_with_the_returning_originals() {
        let a = Mat::zeros(6, 6, CV_8UC1).unwrap();
        let mut b = Mat::zeros(6, 6, CV_8UC1).unwrap();
        cv_core::arithm::set_to(&mut b, Scalar::all(3.0)).unwrap();
        let viaThrowing = add(&a, &b);
        let viaResult = cv_core::arithm::add(&a, &b).unwrap();
        assert_eq!(
            cv_core::to_f64(&viaThrowing).unwrap(),
            cv_core::to_f64(&viaResult).unwrap()
        );
    }
}
