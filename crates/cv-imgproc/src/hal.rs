//! The `imgproc` HAL's flag numbering
//! (`modules/imgproc/include/opencv2/imgproc/hal/interface.h`).
//!
//! A hardware-acceleration layer is compiled against these numbers rather than
//! against the `cv::` enums, and the two are **not** always the same: the HAL
//! threshold and morphology values happen to coincide with `cv::ThresholdTypes`
//! and `cv::MorphTypes`, while `CV_HAL_INTER_*` adds `LINEAR_EXACT` and
//! `NEAREST_EXACT` in positions the public `cv::InterpolationFlags` numbers
//! differently.
//!
//! They are declared as their own constants for that reason. Reading a HAL
//! flag as a `cv::` one is the kind of mistake that produces a plausible image
//! computed with the wrong interpolation.

/// `CV_IMGPROC_HAL_HPP` — the header's include guard, and the marker a build
/// checks for when deciding whether the HAL interface is available.
pub const CV_IMGPROC_HAL_HPP: bool = true;

/// `CV_HAL_INTER_NEAREST`
pub const CV_HAL_INTER_NEAREST: i32 = 0;
/// `CV_HAL_INTER_LINEAR`
pub const CV_HAL_INTER_LINEAR: i32 = 1;
/// `CV_HAL_INTER_CUBIC`
pub const CV_HAL_INTER_CUBIC: i32 = 2;
/// `CV_HAL_INTER_AREA`
pub const CV_HAL_INTER_AREA: i32 = 3;
/// `CV_HAL_INTER_LANCZOS4`
pub const CV_HAL_INTER_LANCZOS4: i32 = 4;
/// `CV_HAL_INTER_LINEAR_EXACT`
pub const CV_HAL_INTER_LINEAR_EXACT: i32 = 5;
/// `CV_HAL_INTER_NEAREST_EXACT`
pub const CV_HAL_INTER_NEAREST_EXACT: i32 = 6;
/// `CV_HAL_INTER_MAX` — one past the last interpolation mode.
pub const CV_HAL_INTER_MAX: i32 = 7;
/// `CV_HAL_WARP_FILL_OUTLIERS`
pub const CV_HAL_WARP_FILL_OUTLIERS: i32 = 8;
/// `CV_HAL_WARP_INVERSE_MAP`
pub const CV_HAL_WARP_INVERSE_MAP: i32 = 16;
/// `CV_HAL_WARP_RELATIVE_MAP`
pub const CV_HAL_WARP_RELATIVE_MAP: i32 = 32;

/// `CV_HAL_MORPH_ERODE`
pub const CV_HAL_MORPH_ERODE: i32 = 0;
/// `CV_HAL_MORPH_DILATE`
pub const CV_HAL_MORPH_DILATE: i32 = 1;

/// `CV_HAL_THRESH_BINARY`
pub const CV_HAL_THRESH_BINARY: i32 = 0;
/// `CV_HAL_THRESH_BINARY_INV`
pub const CV_HAL_THRESH_BINARY_INV: i32 = 1;
/// `CV_HAL_THRESH_TRUNC`
pub const CV_HAL_THRESH_TRUNC: i32 = 2;
/// `CV_HAL_THRESH_TOZERO`
pub const CV_HAL_THRESH_TOZERO: i32 = 3;
/// `CV_HAL_THRESH_TOZERO_INV`
pub const CV_HAL_THRESH_TOZERO_INV: i32 = 4;
/// `CV_HAL_THRESH_MASK` — the bits that hold the type, clearing the modifiers.
pub const CV_HAL_THRESH_MASK: i32 = 7;
/// `CV_HAL_THRESH_OTSU` — a modifier bit, not a type.
pub const CV_HAL_THRESH_OTSU: i32 = 8;
/// `CV_HAL_THRESH_TRIANGLE` — likewise.
pub const CV_HAL_THRESH_TRIANGLE: i32 = 16;

/// `CV_HAL_ADAPTIVE_THRESH_MEAN_C`
pub const CV_HAL_ADAPTIVE_THRESH_MEAN_C: i32 = 0;
/// `CV_HAL_ADAPTIVE_THRESH_GAUSSIAN_C`
pub const CV_HAL_ADAPTIVE_THRESH_GAUSSIAN_C: i32 = 1;

/// `CV_RGB(r, g, b)` — a [`cv_core::Scalar`] in **BGR** order.
///
/// The whole point of the macro: OpenCV's drawing functions take BGR, and
/// writing `Scalar(r, g, b)` by hand is the single most common way to get a red
/// line where a blue one was wanted. The alpha channel is zero, as upstream's
/// is.
#[must_use]
pub fn cv_rgb(r: f64, g: f64, b: f64) -> cv_core::Scalar {
    cv_core::Scalar::new(b, g, r, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hal_flags_hold_upstreams_numbering() {
        assert_eq!(
            [
                CV_HAL_INTER_NEAREST,
                CV_HAL_INTER_LINEAR,
                CV_HAL_INTER_CUBIC,
                CV_HAL_INTER_AREA,
                CV_HAL_INTER_LANCZOS4,
                CV_HAL_INTER_LINEAR_EXACT,
                CV_HAL_INTER_NEAREST_EXACT,
                CV_HAL_INTER_MAX,
            ],
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
        // The warp flags are bits above the mode, so they can be combined.
        assert_eq!(CV_HAL_WARP_FILL_OUTLIERS, 8);
        assert_eq!(CV_HAL_WARP_INVERSE_MAP, 16);
        assert_eq!(CV_HAL_WARP_RELATIVE_MAP, 32);
        assert_eq!(CV_HAL_INTER_MAX & CV_HAL_WARP_INVERSE_MAP, 0, "mode and flags do not overlap");

        assert_eq!((CV_HAL_MORPH_ERODE, CV_HAL_MORPH_DILATE), (0, 1));

        assert_eq!(
            [
                CV_HAL_THRESH_BINARY,
                CV_HAL_THRESH_BINARY_INV,
                CV_HAL_THRESH_TRUNC,
                CV_HAL_THRESH_TOZERO,
                CV_HAL_THRESH_TOZERO_INV,
            ],
            [0, 1, 2, 3, 4]
        );
        // The mask covers every type and none of the modifiers.
        assert_eq!(CV_HAL_THRESH_TOZERO_INV & CV_HAL_THRESH_MASK, CV_HAL_THRESH_TOZERO_INV);
        assert_eq!(CV_HAL_THRESH_OTSU & CV_HAL_THRESH_MASK, 0);
        assert_eq!(CV_HAL_THRESH_TRIANGLE & CV_HAL_THRESH_MASK, 0);

        assert_eq!((CV_HAL_ADAPTIVE_THRESH_MEAN_C, CV_HAL_ADAPTIVE_THRESH_GAUSSIAN_C), (0, 1));
    }

    // The guard constant is what a build tests to decide whether the HAL
    // interface is present at all, so it is checked at compile time.
    const _: () = assert!(CV_IMGPROC_HAL_HPP);

    #[test]
    fn cv_rgb_swaps_into_bgr_order() {
        let red = cv_rgb(255.0, 0.0, 0.0);
        assert_eq!(red.0, [0.0, 0.0, 255.0, 0.0], "red lands in the third channel");
        let green = cv_rgb(0.0, 255.0, 0.0);
        assert_eq!(green.0, [0.0, 255.0, 0.0, 0.0]);
        let blue = cv_rgb(0.0, 0.0, 255.0);
        assert_eq!(blue.0, [255.0, 0.0, 0.0, 0.0]);
    }
}
