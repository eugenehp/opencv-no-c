//! Flag and mode constants of `cv::imgproc`
//! (`modules/imgproc/include/opencv2/imgproc.hpp`).
//!
//! The families that the typed enums elsewhere in this crate do not already
//! cover: Hough variants, connected-component algorithms, marker and line
//! types for the drawing calls, grab-cut classes, template-match modes,
//! histogram comparison metrics, and the remaining colour-conversion codes.
//!
//! Generated from the upstream header, so every value is the one `cv2`
//! reports. Where this crate has a typed enum for a family — `Interpolation`,
//! `BorderType`, `ThresholdType` — that enum is what new code should use;
//! these integers are for translating existing code and for the flags words
//! that arrive as `i32`.

/// `FILTER_SCHARR`.
pub const FILTER_SCHARR: i32 = -1;
/// `MORPH_HITMISS` — "hit or miss".
pub const MORPH_HITMISS: i32 = 7;
/// `INTER_CUBIC`.
pub const INTER_CUBIC: i32 = 2;
/// `INTER_LANCZOS4`.
pub const INTER_LANCZOS4: i32 = 4;
/// `INTER_NEAREST_EXACT`.
pub const INTER_NEAREST_EXACT: i32 = 6;
/// `INTER_MAX`.
pub const INTER_MAX: i32 = 7;
/// `WARP_FILL_OUTLIERS`.
pub const WARP_FILL_OUTLIERS: i32 = 8;
/// `WARP_RELATIVE_MAP`.
pub const WARP_RELATIVE_MAP: i32 = 32;
/// `INTER_BITS`.
pub const INTER_BITS: i32 = 5;
/// `INTER_BITS2`.
pub const INTER_BITS2: i32 = 10;
/// `INTER_TAB_SIZE`.
pub const INTER_TAB_SIZE: i32 = 32;
/// `INTER_TAB_SIZE2`.
pub const INTER_TAB_SIZE2: i32 = 1024;
/// `DIST_USER` — User defined distance.
pub const DIST_USER: i32 = -1;
/// `DIST_L12` — L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1)).
pub const DIST_L12: i32 = 4;
/// `DIST_FAIR` — distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998.
pub const DIST_FAIR: i32 = 5;
/// `DIST_WELSCH` — distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846.
pub const DIST_WELSCH: i32 = 6;
/// `THRESH_MASK`.
pub const THRESH_MASK: i32 = 7;
/// `THRESH_DRYRUN` — flag, compute threshold only (useful for OTSU/TRIANGLE)
///   but does not actually run thresholding.
pub const THRESH_DRYRUN: i32 = 128;
/// `GC_INIT_WITH_MASK`.
pub const GC_INIT_WITH_MASK: i32 = 1;
/// `GC_EVAL`.
pub const GC_EVAL: i32 = 2;
/// `GC_EVAL_FREEZE_MODEL`.
pub const GC_EVAL_FREEZE_MODEL: i32 = 3;
/// `FLOODFILL_MASK_ONLY`.
pub const FLOODFILL_MASK_ONLY: i32 = 131072;
/// `RETR_FLOODFILL`.
pub const RETR_FLOODFILL: i32 = 4;
/// `CHAIN_APPROX_TC89_L1`.
pub const CHAIN_APPROX_TC89_L1: i32 = 3;
/// `CHAIN_APPROX_TC89_KCOS`.
pub const CHAIN_APPROX_TC89_KCOS: i32 = 4;
/// `HOUGH_STANDARD`.
pub const HOUGH_STANDARD: i32 = 0;
/// `HOUGH_PROBABILISTIC`.
pub const HOUGH_PROBABILISTIC: i32 = 1;
/// `HOUGH_MULTI_SCALE`.
pub const HOUGH_MULTI_SCALE: i32 = 2;
/// `HOUGH_GRADIENT` — basically *21HT*, described in Yuen90.
pub const HOUGH_GRADIENT: i32 = 3;
/// `HOUGH_GRADIENT_ALT` — variation of HOUGH_GRADIENT to get better accuracy.
pub const HOUGH_GRADIENT_ALT: i32 = 4;
/// `LSD_REFINE_NONE` — No refinement applied.
pub const LSD_REFINE_NONE: i32 = 0;
/// `LSD_REFINE_STD` — Standard refinement is applied. E.g. breaking arches
///   into smaller straighter line approximations.
pub const LSD_REFINE_STD: i32 = 1;
/// `LSD_REFINE_ADV` — Advanced refinement. Number of false alarms is
///   calculated, lines are.
pub const LSD_REFINE_ADV: i32 = 2;
/// `HISTCMP_HELLINGER` — Synonym for HISTCMP_BHATTACHARYYA.
pub const HISTCMP_HELLINGER: i32 = 3;
/// `COLOR_RGB2BGR565` — `CV_8U`.
pub const COLOR_RGB2BGR565: i32 = 13;
/// `COLOR_BGR5652RGB` — `CV_8U`.
pub const COLOR_BGR5652RGB: i32 = 15;
/// `COLOR_BGRA2BGR565` — `CV_8U`.
pub const COLOR_BGRA2BGR565: i32 = 16;
/// `COLOR_RGBA2BGR565` — `CV_8U`.
pub const COLOR_RGBA2BGR565: i32 = 17;
/// `COLOR_BGR5652BGRA` — `CV_8U`.
pub const COLOR_BGR5652BGRA: i32 = 18;
/// `COLOR_BGR5652RGBA` — `CV_8U`.
pub const COLOR_BGR5652RGBA: i32 = 19;
/// `COLOR_GRAY2BGR565` — `CV_8U` convert between grayscale to BGR565 (16-bit
///   images).
pub const COLOR_GRAY2BGR565: i32 = 20;
/// `COLOR_BGR5652GRAY` — `CV_8U`.
pub const COLOR_BGR5652GRAY: i32 = 21;
/// `COLOR_RGB2BGR555` — `CV_8U`.
pub const COLOR_RGB2BGR555: i32 = 23;
/// `COLOR_BGR5552RGB` — `CV_8U`.
pub const COLOR_BGR5552RGB: i32 = 25;
/// `COLOR_BGRA2BGR555` — `CV_8U`.
pub const COLOR_BGRA2BGR555: i32 = 26;
/// `COLOR_RGBA2BGR555` — `CV_8U`.
pub const COLOR_RGBA2BGR555: i32 = 27;
/// `COLOR_BGR5552BGRA` — `CV_8U`.
pub const COLOR_BGR5552BGRA: i32 = 28;
/// `COLOR_BGR5552RGBA` — `CV_8U`.
pub const COLOR_BGR5552RGBA: i32 = 29;
/// `COLOR_GRAY2BGR555` — `CV_8U` convert between grayscale and BGR555 (16-bit
///   images).
pub const COLOR_GRAY2BGR555: i32 = 30;
/// `COLOR_BGR5552GRAY` — `CV_8U`.
pub const COLOR_BGR5552GRAY: i32 = 31;
/// `COLOR_BGR2HSV_FULL` — [8U/32F] convert RGB/BGR to HSV (hue saturation
///   value) with H range 0..255 if 8 bit image, color_convert_rgb_hsv "color
///   conversions".
pub const COLOR_BGR2HSV_FULL: i32 = 66;
/// `COLOR_RGB2HSV_FULL` — [8U/32F].
pub const COLOR_RGB2HSV_FULL: i32 = 67;
/// `COLOR_BGR2HLS_FULL` — [8U/32F] convert RGB/BGR to HLS (hue lightness
///   saturation) with H range 0..255 if 8 bit image, color_convert_rgb_hls
///   "color conversions".
pub const COLOR_BGR2HLS_FULL: i32 = 68;
/// `COLOR_RGB2HLS_FULL` — [8U/32F].
pub const COLOR_RGB2HLS_FULL: i32 = 69;
/// `COLOR_HSV2BGR_FULL` — [8U/32F] backward conversions HSV to RGB/BGR with H
///   range 0..255 if 8 bit image.
pub const COLOR_HSV2BGR_FULL: i32 = 70;
/// `COLOR_HSV2RGB_FULL` — [8U/32F].
pub const COLOR_HSV2RGB_FULL: i32 = 71;
/// `COLOR_HLS2BGR_FULL` — [8U/32F] backward conversions HLS to RGB/BGR with H
///   range 0..255 if 8 bit image.
pub const COLOR_HLS2BGR_FULL: i32 = 72;
/// `COLOR_HLS2RGB_FULL` — [8U/32F].
pub const COLOR_HLS2RGB_FULL: i32 = 73;
/// `COLOR_RGB2YUV_UYVY` — `CV_8U` convert between RGB and YUV UYVU, YUV is 4:2:2
///   and interleaved as U/Y1/V/Y2, see color_convert_rgb_yuv_42x.
pub const COLOR_RGB2YUV_UYVY: i32 = 143;
/// `COLOR_BGR2YUV_UYVY` — `CV_8U` convert between BGR and YUV UYVU, YUV is 4:2:2
///   and interleaved as U/Y1/V/Y2, see color_convert_rgb_yuv_42x.
pub const COLOR_BGR2YUV_UYVY: i32 = 144;
/// `COLOR_RGB2YUV_Y422` — synonym to UYVY.
pub const COLOR_RGB2YUV_Y422: i32 = 143;
/// `COLOR_BGR2YUV_Y422` — synonym to UYVY.
pub const COLOR_BGR2YUV_Y422: i32 = 144;
/// `COLOR_RGB2YUV_UYNV` — synonym to UYVY.
pub const COLOR_RGB2YUV_UYNV: i32 = 143;
/// `COLOR_BGR2YUV_UYNV` — synonym to UYVY.
pub const COLOR_BGR2YUV_UYNV: i32 = 144;
/// `COLOR_RGBA2YUV_UYVY` — `CV_8U` convert between RGBA and YUV UYVU, YUV is
///   4:2:2 and interleaved as U/Y1/V/Y2, see color_convert_rgb_yuv_42x.
pub const COLOR_RGBA2YUV_UYVY: i32 = 145;
/// `COLOR_BGRA2YUV_UYVY` — `CV_8U` convert between BGRA and YUV UYVU, YUV is
///   4:2:2 and interleaved as U/Y1/V/Y2, see color_convert_rgb_yuv_42x.
pub const COLOR_BGRA2YUV_UYVY: i32 = 146;
/// `COLOR_RGBA2YUV_Y422` — synonym to UYVY.
pub const COLOR_RGBA2YUV_Y422: i32 = 145;
/// `COLOR_BGRA2YUV_Y422` — synonym to UYVY.
pub const COLOR_BGRA2YUV_Y422: i32 = 146;
/// `COLOR_RGBA2YUV_UYNV` — synonym to UYVY.
pub const COLOR_RGBA2YUV_UYNV: i32 = 145;
/// `COLOR_BGRA2YUV_UYNV` — synonym to UYVY.
pub const COLOR_BGRA2YUV_UYNV: i32 = 146;
/// `COLOR_RGB2YUV_YUY2` — `CV_8U` convert between RGB and YUV YUY2, YUV is 4:2:2
///   and interleaved as Y1/U/Y2/V, see color_convert_rgb_yuv_42x.
pub const COLOR_RGB2YUV_YUY2: i32 = 147;
/// `COLOR_BGR2YUV_YUY2` — `CV_8U` convert between BGR and YUV YUY2, YUV is 4:2:2
///   and interleaved as Y1/U/Y2/V, see color_convert_rgb_yuv_42x.
pub const COLOR_BGR2YUV_YUY2: i32 = 148;
/// `COLOR_RGB2YUV_YVYU` — `CV_8U` convert between RGB and YUV YVYU, YUV is 4:2:2
///   and interleaved as Y1/V/Y2/U, see color_convert_rgb_yuv_42x.
pub const COLOR_RGB2YUV_YVYU: i32 = 149;
/// `COLOR_BGR2YUV_YVYU` — `CV_8U` convert between BGR and YUV YVYU, YUV is 4:2:2
///   and interleaved as Y1/V/Y2/U, see color_convert_rgb_yuv_42x.
pub const COLOR_BGR2YUV_YVYU: i32 = 150;
/// `COLOR_RGB2YUV_YUYV` — synonym to YUY2.
pub const COLOR_RGB2YUV_YUYV: i32 = 147;
/// `COLOR_BGR2YUV_YUYV` — synonym to YUY2.
pub const COLOR_BGR2YUV_YUYV: i32 = 148;
/// `COLOR_RGB2YUV_YUNV` — synonym to YUY2.
pub const COLOR_RGB2YUV_YUNV: i32 = 147;
/// `COLOR_BGR2YUV_YUNV` — synonym to YUY2.
pub const COLOR_BGR2YUV_YUNV: i32 = 148;
/// `COLOR_RGBA2YUV_YUY2` — `CV_8U` convert between RGBA and YUV YUY2, YUV is
///   4:2:2 and interleaved as Y1/U/Y2/V, see color_convert_rgb_yuv_42x.
pub const COLOR_RGBA2YUV_YUY2: i32 = 151;
/// `COLOR_BGRA2YUV_YUY2` — `CV_8U` convert between BGRA and YUV YUY2, YUV is
///   4:2:2 and interleaved as Y1/U/Y2/V, see color_convert_rgb_yuv_42x.
pub const COLOR_BGRA2YUV_YUY2: i32 = 152;
/// `COLOR_RGBA2YUV_YVYU` — `CV_8U` convert between RGBA and YUV YVYU, YUV is
///   4:2:2 and interleaved as Y1/V/Y2/U, see color_convert_rgb_yuv_42x.
pub const COLOR_RGBA2YUV_YVYU: i32 = 153;
/// `COLOR_BGRA2YUV_YVYU` — `CV_8U` convert between BGRA and YUV YVYU, YUV is
///   4:2:2 and interleaved as Y1/V/Y2/U, see color_convert_rgb_yuv_42x.
pub const COLOR_BGRA2YUV_YVYU: i32 = 154;
/// `COLOR_RGBA2YUV_YUYV` — synonym to YUY2.
pub const COLOR_RGBA2YUV_YUYV: i32 = 151;
/// `COLOR_BGRA2YUV_YUYV` — synonym to YUY2.
pub const COLOR_BGRA2YUV_YUYV: i32 = 152;
/// `COLOR_RGBA2YUV_YUNV` — synonym to YUY2.
pub const COLOR_RGBA2YUV_YUNV: i32 = 151;
/// `COLOR_BGRA2YUV_YUNV` — synonym to YUY2.
pub const COLOR_BGRA2YUV_YUNV: i32 = 152;
/// `COLOR_COLORCVT_MAX`.
pub const COLOR_COLORCVT_MAX: i32 = 155;
/// `INTERSECT_NONE` — No intersection.
pub const INTERSECT_NONE: i32 = 0;
/// `INTERSECT_PARTIAL` — There is a partial intersection.
pub const INTERSECT_PARTIAL: i32 = 1;
/// `INTERSECT_FULL` — One of the rectangle is fully enclosed in the other.
pub const INTERSECT_FULL: i32 = 2;
/// `LINE_4` — 4-connected line.
pub const LINE_4: i32 = 4;
/// `LINE_AA` — antialiased line.
pub const LINE_AA: i32 = 16;
/// `FONT_HERSHEY_PLAIN` — small size sans-serif font.
pub const FONT_HERSHEY_PLAIN: i32 = 1;
/// `FONT_HERSHEY_DUPLEX` — normal size sans-serif font (more complex than
///   FONT_HERSHEY_SIMPLEX).
pub const FONT_HERSHEY_DUPLEX: i32 = 2;
/// `FONT_HERSHEY_COMPLEX` — normal size serif font.
pub const FONT_HERSHEY_COMPLEX: i32 = 3;
/// `FONT_HERSHEY_TRIPLEX` — normal size serif font (more complex than
///   FONT_HERSHEY_COMPLEX).
pub const FONT_HERSHEY_TRIPLEX: i32 = 4;
/// `FONT_HERSHEY_COMPLEX_SMALL` — smaller version of FONT_HERSHEY_COMPLEX.
pub const FONT_HERSHEY_COMPLEX_SMALL: i32 = 5;
/// `FONT_HERSHEY_SCRIPT_SIMPLEX` — hand-writing style font.
pub const FONT_HERSHEY_SCRIPT_SIMPLEX: i32 = 6;
/// `FONT_HERSHEY_SCRIPT_COMPLEX` — more complex variant of
///   FONT_HERSHEY_SCRIPT_SIMPLEX.
pub const FONT_HERSHEY_SCRIPT_COMPLEX: i32 = 7;
/// `FONT_ITALIC` — flag for italic font.
pub const FONT_ITALIC: i32 = 16;
