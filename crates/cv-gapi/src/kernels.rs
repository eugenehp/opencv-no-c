//! The standard kernel set: every operation a graph can name
//! (`modules/gapi/include/opencv2/gapi/core.hpp`, `imgproc.hpp`, `video.hpp`,
//! `stereo.hpp`, `infer/parsers.hpp`).
//!
//! # These record, they do not compute
//!
//! Each function here appends one node to a [`crate::GComputation`] and returns
//! a handle to its result. Nothing is evaluated until the graph is compiled and
//! applied — which is the whole reason G-API exists, and the thing that most
//! surprises a reader coming from `imgproc`, where the same names run
//! immediately.
//!
//! # Recorded, not yet bound
//!
//! These nodes join the graph like any other: their id, operands and position
//! are real, and the graph's shape — which most of G-API's surface operates on
//! — is fully usable. What they do not yet have is a CPU *binding*, so
//! compiling a graph that contains one fails and names the kernel.
//!
//! That is deliberate. Evaluating an unbound node as a pass-through would
//! produce an image that is subtly not what the graph says, and no amount of
//! looking at the output would reveal it. The built-in operations on
//! [`crate::GComputation`] — blur, Sobel, Canny, threshold and the rest — are
//! the set that does run today.
//!
//! # Naming
//!
//! Upstream distinguishes an operation taking two matrices from the same
//! operation taking a matrix and a constant by a `C` suffix — `sub` and
//! `subC` — and the reversed-operand forms by an `R` — `divRC` is
//! `constant / matrix`. Those are three genuinely different kernels, and
//! collapsing them would make a graph's arity ambiguous.

use cv_core::{cv_assert, Result};

use crate::{GComputation, GMat, GScalar};

/// One recorded operation's name and arity, as the graph stores it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelNode {
    /// The upstream operation id, e.g. `org.opencv.core.math.add`.
    pub id: String,
    /// How many protocol inputs it consumes.
    pub arity: usize,
}

/// Record a unary node and hand back its result.
fn unary(g: &mut GComputation, id: &str, src: GMat) -> GMat {
    g.record(id, &[src])
}

/// Record a binary node.
fn binary(g: &mut GComputation, id: &str, a: GMat, b: GMat) -> GMat {
    g.record(id, &[a, b])
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

/// `cv::gapi::addWeighted(src1, alpha, src2, beta, gamma, ddepth)`.
///
/// `alpha*src1 + beta*src2 + gamma`, in one pass. Worth its own kernel rather
/// than three: the intermediate of `alpha*src1` would otherwise be a full
/// buffer, and on a streaming graph that buffer is per frame.
pub fn add_weighted(
    g: &mut GComputation,
    src1: GMat,
    _alpha: f64,
    src2: GMat,
    _beta: f64,
    _gamma: f64,
) -> GMat {
    binary(g, "org.opencv.core.math.addWeighted", src1, src2)
}

/// `cv::gapi::subC(src, c, ddepth)` — matrix minus constant.
pub fn sub_c(g: &mut GComputation, src: GMat, _c: GScalar) -> GMat {
    unary(g, "org.opencv.core.math.subC", src)
}

/// `cv::gapi::subRC(c, src, ddepth)` — constant minus matrix.
///
/// Not the same as [`sub_c`] with a negated constant, because saturation is
/// applied after the subtraction: `10 - 200` clamps to 0 in `u8`, and
/// `200 - 10` does not.
pub fn sub_rc(g: &mut GComputation, _c: GScalar, src: GMat) -> GMat {
    unary(g, "org.opencv.core.math.subRC", src)
}

/// `cv::gapi::absDiffC(src, c)`.
pub fn abs_diff_c(g: &mut GComputation, src: GMat, _c: GScalar) -> GMat {
    unary(g, "org.opencv.core.matrixop.absdiffC", src)
}

/// `cv::gapi::div(src1, src2, scale, ddepth)`.
///
/// Division by zero yields zero, as upstream's does — the alternative is an
/// infinity that then saturates to the type's maximum, which looks like a
/// bright pixel rather than like missing data.
pub fn div(g: &mut GComputation, src1: GMat, src2: GMat, _scale: f64) -> GMat {
    binary(g, "org.opencv.core.math.div", src1, src2)
}

/// `cv::gapi::divC(src, divisor, scale, ddepth)`.
pub fn div_c(g: &mut GComputation, src: GMat, _divisor: GScalar, _scale: f64) -> GMat {
    unary(g, "org.opencv.core.math.divC", src)
}

/// `cv::gapi::divRC(divident, src, scale, ddepth)` — constant over matrix.
pub fn div_rc(g: &mut GComputation, _divident: GScalar, src: GMat, _scale: f64) -> GMat {
    unary(g, "org.opencv.core.math.divRC", src)
}

/// `cv::gapi::max(src1, src2)`.
pub fn max(g: &mut GComputation, src1: GMat, src2: GMat) -> GMat {
    binary(g, "org.opencv.core.matrixop.max", src1, src2)
}

/// `cv::gapi::min(src1, src2)`.
pub fn min(g: &mut GComputation, src1: GMat, src2: GMat) -> GMat {
    binary(g, "org.opencv.core.matrixop.min", src1, src2)
}

/// `cv::gapi::bitwise_xor(src1, src2)`.
pub fn bitwise_xor(g: &mut GComputation, src1: GMat, src2: GMat) -> GMat {
    binary(g, "org.opencv.core.pixelwise.bitwise_xor", src1, src2)
}

/// `cv::gapi::mask(src, mask)` — keep the pixels the mask selects.
pub fn mask(g: &mut GComputation, src: GMat, m: GMat) -> GMat {
    binary(g, "org.opencv.core.pixelwise.mask", src, m)
}

/// `cv::gapi::select(src1, src2, mask)` — pick per pixel from two sources.
pub fn select(g: &mut GComputation, src1: GMat, src2: GMat, m: GMat) -> GMat {
    g.record("org.opencv.core.pixelwise.select", &[src1, src2, m])
}

/// `cv::gapi::convertTo(src, rdepth, alpha, beta)`.
pub fn convert_to(g: &mut GComputation, src: GMat, _rdepth: i32, _alpha: f64, _beta: f64) -> GMat {
    unary(g, "org.opencv.core.transform.convertTo", src)
}

/// `cv::gapi::LUT(src, lut)` — a per-element table lookup.
pub fn lut(g: &mut GComputation, src: GMat, _lut: &[u8]) -> GMat {
    unary(g, "org.opencv.core.transform.LUT", src)
}

/// `cv::gapi::sqrt(src)`.
pub fn sqrt(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.core.math.sqrt", src)
}

/// `cv::gapi::phase(x, y, angleInDegrees)` — the angle of each `(x, y)` pair.
pub fn phase(g: &mut GComputation, x: GMat, y: GMat, _angle_in_degrees: bool) -> GMat {
    binary(g, "org.opencv.core.math.phase", x, y)
}

/// `cv::gapi::cartToPolar(x, y, angleInDegrees)`.
pub fn cart_to_polar(
    g: &mut GComputation,
    x: GMat,
    y: GMat,
    _angle_in_degrees: bool,
) -> (GMat, GMat) {
    let magnitude = binary(g, "org.opencv.core.math.cartToPolar", x, y);
    let angle = binary(g, "org.opencv.core.math.cartToPolar", x, y);
    (magnitude, angle)
}

/// `cv::gapi::polarToCart(magnitude, angle, angleInDegrees)`.
pub fn polar_to_cart(
    g: &mut GComputation,
    magnitude: GMat,
    angle: GMat,
    _angle_in_degrees: bool,
) -> (GMat, GMat) {
    let x = binary(g, "org.opencv.core.math.polarToCart", magnitude, angle);
    let y = binary(g, "org.opencv.core.math.polarToCart", magnitude, angle);
    (x, y)
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------


/// `cv::gapi::cmpGT(src1, src2)` — 255 where the first exceeds the second.
pub fn cmp_gt(g: &mut GComputation, src1: GMat, src2: GMat) -> GMat {
    binary(g, "org.opencv.core.pixelwise.compare.cmpGT", src1, src2)
}

/// `cv::gapi::cmpGE(src1, src2)`.
pub fn cmp_ge(g: &mut GComputation, src1: GMat, src2: GMat) -> GMat {
    binary(g, "org.opencv.core.pixelwise.compare.cmpGE", src1, src2)
}

/// `cv::gapi::cmpLE(src1, src2)`.
pub fn cmp_le(g: &mut GComputation, src1: GMat, src2: GMat) -> GMat {
    binary(g, "org.opencv.core.pixelwise.compare.cmpLE", src1, src2)
}

/// `cv::gapi::cmpLT(src1, src2)`.
pub fn cmp_lt(g: &mut GComputation, src1: GMat, src2: GMat) -> GMat {
    binary(g, "org.opencv.core.pixelwise.compare.cmpLT", src1, src2)
}

/// `cv::gapi::cmpEQ(src1, src2)`.
pub fn cmp_eq(g: &mut GComputation, src1: GMat, src2: GMat) -> GMat {
    binary(g, "org.opencv.core.pixelwise.compare.cmpEQ", src1, src2)
}

/// `cv::gapi::cmpNE(src1, src2)`.
pub fn cmp_ne(g: &mut GComputation, src1: GMat, src2: GMat) -> GMat {
    binary(g, "org.opencv.core.pixelwise.compare.cmpNE", src1, src2)
}


/// `cv::gapi::inRange(src, threshLow, threshUp)`.
pub fn in_range(g: &mut GComputation, src: GMat, _low: GScalar, _up: GScalar) -> GMat {
    unary(g, "org.opencv.core.pixelwise.inRange", src)
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// `cv::gapi::sum(src)`.
pub fn sum(g: &mut GComputation, src: GMat) -> GScalar {
    g.record_scalar("org.opencv.core.matrixop.sum", src)
}

/// `cv::gapi::mean(src)`.
pub fn mean(g: &mut GComputation, src: GMat) -> GScalar {
    g.record_scalar("org.opencv.core.math.mean", src)
}

/// `cv::gapi::countNonZero(src)`.
pub fn count_non_zero(g: &mut GComputation, src: GMat) -> GScalar {
    g.record_scalar("org.opencv.core.matrixop.countNonZero", src)
}

/// `cv::gapi::normL1(src)`.
pub fn norm_l1(g: &mut GComputation, src: GMat) -> GScalar {
    g.record_scalar("org.opencv.core.matrixop.norm.normL1", src)
}

/// `cv::gapi::normL2(src)`.
pub fn norm_l2(g: &mut GComputation, src: GMat) -> GScalar {
    g.record_scalar("org.opencv.core.matrixop.norm.normL2", src)
}

/// `cv::gapi::normInf(src)`.
pub fn norm_inf(g: &mut GComputation, src: GMat) -> GScalar {
    g.record_scalar("org.opencv.core.matrixop.norm.normInf", src)
}

/// `cv::gapi::integral(src, sdepth, sqdepth)`.
pub fn integral(g: &mut GComputation, src: GMat, _sdepth: i32, _sqdepth: i32) -> (GMat, GMat) {
    let sum = unary(g, "org.opencv.core.matrixop.integral", src);
    let sqsum = unary(g, "org.opencv.core.matrixop.integral", src);
    (sum, sqsum)
}

/// `cv::gapi::normalize(src, a, b, norm_type, ddepth)`.
pub fn normalize(g: &mut GComputation, src: GMat, _a: f64, _b: f64, _norm_type: i32) -> GMat {
    unary(g, "org.opencv.core.normalize", src)
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/// `cv::gapi::transpose(src)`.
pub fn transpose(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.core.transpose", src)
}

/// `cv::gapi::concatHor(src1, src2)`.
pub fn concat_hor(g: &mut GComputation, src1: GMat, src2: GMat) -> GMat {
    binary(g, "org.opencv.imgproc.transform.concatHor", src1, src2)
}

/// `cv::gapi::concatVert(src1, src2)`.
pub fn concat_vert(g: &mut GComputation, src1: GMat, src2: GMat) -> GMat {
    binary(g, "org.opencv.imgproc.transform.concatVert", src1, src2)
}

/// `cv::gapi::crop(src, rect)`.
pub fn crop(g: &mut GComputation, src: GMat, _rect: crate::graph::Rect) -> GMat {
    unary(g, "org.opencv.core.transform.crop", src)
}

/// `cv::gapi::resizeP(src, dsize, interpolation)` — resize a planar image.
///
/// Distinct from `resize` because a planar buffer's rows are not adjacent: the
/// interleaved kernel would blend across a plane boundary and mix channels.
pub fn resize_p(g: &mut GComputation, src: GMat, _dsize: (i32, i32), _interpolation: i32) -> GMat {
    unary(g, "org.opencv.imgproc.transform.resizeP", src)
}

/// `cv::gapi::remap(src, map1, map2, interpolation, borderMode, borderValue)`.
pub fn remap(g: &mut GComputation, src: GMat, _map1: &[f32], _map2: &[f32]) -> GMat {
    unary(g, "org.opencv.core.transform.remap", src)
}

/// `cv::gapi::warpAffine(src, M, dsize, flags, borderMode, borderValue)`.
pub fn warp_affine(g: &mut GComputation, src: GMat, _m: [[f64; 3]; 2], _dsize: (i32, i32)) -> GMat {
    unary(g, "org.opencv.core.warpAffine", src)
}

/// `cv::gapi::warpPerspective(src, M, dsize, flags, borderMode, borderValue)`.
pub fn warp_perspective(
    g: &mut GComputation,
    src: GMat,
    _m: [[f64; 3]; 3],
    _dsize: (i32, i32),
) -> GMat {
    unary(g, "org.opencv.core.warpPerspective", src)
}

// ---------------------------------------------------------------------------
// Filtering and morphology
// ---------------------------------------------------------------------------

/// `cv::gapi::boxFilter(src, dtype, ksize, anchor, normalize, borderType, borderValue)`.
pub fn box_filter(g: &mut GComputation, src: GMat, _ksize: (i32, i32), _normalize: bool) -> GMat {
    unary(g, "org.opencv.imgproc.filters.boxfilter", src)
}

/// `cv::gapi::sepFilter(src, ddepth, kernelX, kernelY, anchor, delta, borderType, borderValue)`.
pub fn sep_filter(g: &mut GComputation, src: GMat, _kx: &[f32], _ky: &[f32]) -> GMat {
    unary(g, "org.opencv.imgproc.filters.sepfilter", src)
}

/// `cv::gapi::bilateralFilter(src, d, sigmaColor, sigmaSpace, borderType)`.
pub fn bilateral_filter(
    g: &mut GComputation,
    src: GMat,
    _d: i32,
    _sigma_color: f64,
    _sigma_space: f64,
) -> GMat {
    unary(g, "org.opencv.imgproc.filters.bilateralfilter", src)
}

/// `cv::gapi::erode3x3(src, iterations, borderType, borderValue)`.
///
/// A separate kernel from the general `erode` because a 3x3 structuring element
/// is the overwhelmingly common case and admits a much tighter inner loop.
pub fn erode3x3(g: &mut GComputation, src: GMat, _iterations: i32) -> GMat {
    unary(g, "org.opencv.imgproc.filters.erode3x3", src)
}

/// `cv::gapi::dilate3x3(src, iterations, borderType, borderValue)`.
pub fn dilate3x3(g: &mut GComputation, src: GMat, _iterations: i32) -> GMat {
    unary(g, "org.opencv.imgproc.filters.dilate3x3", src)
}

/// `cv::gapi::morphologyEx(src, op, kernel, anchor, iterations, borderType, borderValue)`.
pub fn morphology_ex(g: &mut GComputation, src: GMat, _op: i32, _ksize: (i32, i32)) -> GMat {
    unary(g, "org.opencv.imgproc.filters.morphologyEx", src)
}

/// `cv::gapi::SobelXY(src, ddepth, order, ksize, ...)`, also spelled
/// `cv::gapi::sobelXY`.
///
/// Both derivatives from one pass. The x and y Sobels share their smoothing
/// stage, so computing them together costs appreciably less than twice one.
pub fn sobel_xy(g: &mut GComputation, src: GMat, _order: i32, _ksize: i32) -> (GMat, GMat) {
    let dx = unary(g, "org.opencv.imgproc.filters.sobelxy", src);
    let dy = unary(g, "org.opencv.imgproc.filters.sobelxy", src);
    (dx, dy)
}

// ---------------------------------------------------------------------------
// Colour conversion
// ---------------------------------------------------------------------------


/// `cv::gapi::BGR2RGB(src)`.
pub fn bgr2rgb(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.bgr2rgb", src)
}

/// `cv::gapi::RGB2Gray(src)`.
///
/// Distinct from `BGR2Gray`: the luma weights are applied to different
/// channels, and swapping them tints the result.
pub fn rgb2gray(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.rgb2gray", src)
}

/// `cv::gapi::RGB2HSV(src)`.
pub fn rgb2hsv(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.rgb2hsv", src)
}

/// `cv::gapi::RGB2Lab(src)`.
pub fn rgb2lab(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.rgb2lab", src)
}

/// `cv::gapi::BGR2LUV(src)`.
pub fn bgr2luv(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.bgr2luv", src)
}

/// `cv::gapi::LUV2BGR(src)`.
pub fn luv2bgr(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.luv2bgr", src)
}

/// `cv::gapi::BGR2YUV(src)`.
pub fn bgr2yuv(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.bgr2yuv", src)
}

/// `cv::gapi::YUV2BGR(src)`.
pub fn yuv2bgr(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.yuv2bgr", src)
}

/// `cv::gapi::RGB2YUV(src)`.
pub fn rgb2yuv(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.rgb2yuv", src)
}

/// `cv::gapi::YUV2RGB(src)`.
pub fn yuv2rgb(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.yuv2rgb", src)
}

/// `cv::gapi::RGB2YUV422(src)`.
pub fn rgb2yuv422(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.rgb2yuv422", src)
}

/// `cv::gapi::BGR2I420(src)`.
///
/// I420 is planar 4:2:0 — a full-size luma plane then two quarter-size
/// chroma planes. That is what a video encoder wants, and why the
/// conversion exists as its own kernel rather than through BGR2YUV.
pub fn bgr2i420(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.bgr2i420", src)
}

/// `cv::gapi::RGB2I420(src)`.
pub fn rgb2i420(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.rgb2i420", src)
}

/// `cv::gapi::I4202BGR(src)`.
pub fn i4202bgr(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.i4202bgr", src)
}

/// `cv::gapi::I4202RGB(src)`.
pub fn i4202rgb(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.i4202rgb", src)
}

/// `cv::gapi::BayerGR2RGB(src_gr)`.
pub fn bayer_gr2rgb(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.bayergr2rgb", src)
}

/// `cv::gapi::NV12toRGB(src_y, src_uv)` — the single-argument form.
pub fn nv12to_rgb(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.nv12torgb", src)
}

/// `cv::gapi::NV12toBGR(src_y, src_uv)`.
pub fn nv12to_bgr(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.nv12tobgr", src)
}

/// `cv::gapi::NV12toGray(src_y, src_uv)`.
///
/// Free, in the sense that matters: NV12's luma plane *is* the greyscale
/// image, so this is a plane selection rather than a conversion.
pub fn nv12to_gray(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.nv12togray", src)
}

/// `cv::gapi::NV12toRGBp(src_y, src_uv)` — to a planar RGB.
pub fn nv12to_rgbp(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.nv12torgbp", src)
}

/// `cv::gapi::NV12toBGRp(src_y, src_uv)`.
pub fn nv12to_bgrp(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.colorconvert.nv12tobgrp", src)
}


/// `cv::gapi::Y(frame)` — the luma plane of a media frame.
pub fn y(g: &mut GComputation, frame: GMat) -> GMat {
    unary(g, "org.opencv.streaming.Y", frame)
}

/// `cv::gapi::UV(frame)` — the interleaved chroma plane.
pub fn uv(g: &mut GComputation, frame: GMat) -> GMat {
    unary(g, "org.opencv.streaming.UV", frame)
}

// ---------------------------------------------------------------------------
// Shape and feature operations
// ---------------------------------------------------------------------------

/// `cv::gapi::boundingRect(src)`.
pub fn bounding_rect(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.shape.boundingRectMat", src)
}

/// `cv::gapi::fitLine2D(src)`.
pub fn fit_line2d(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.shape.fitLine2DMat", src)
}

/// `cv::gapi::fitLine3D(src)`.
pub fn fit_line3d(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.imgproc.shape.fitLine3DMat", src)
}

/// `cv::gapi::goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance, ...)`.
pub fn good_features_to_track(
    g: &mut GComputation,
    image: GMat,
    _max_corners: i32,
    _quality_level: f64,
    _min_distance: f64,
) -> GMat {
    unary(g, "org.opencv.imgproc.feature.goodFeaturesToTrack", image)
}

/// `cv::gapi::stereo(left, right, of)` — a disparity or depth map.
pub fn stereo(g: &mut GComputation, left: GMat, right: GMat) -> GMat {
    binary(g, "org.opencv.stereo", left, right)
}

/// `cv::gapi::video::BackgroundSubtractor(src, bsParams)`.
pub fn background_subtractor(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.video.BackgroundSubtractor", src)
}

/// `cv::gapi::KalmanFilter(measurement, haveMeasurement, control, kfParams)`.
pub fn kalman_filter(g: &mut GComputation, measurement: GMat) -> GMat {
    unary(g, "org.opencv.video.KalmanFilter", measurement)
}

/// `cv::gapi::parseSSD(in, inSz, confidenceThreshold, filterLabel)`.
///
/// Decodes an SSD head's `[1, 1, N, 7]` output into boxes and labels. It is a
/// graph node rather than a post-processing call so the decode runs on the same
/// backend as the inference, which on an accelerator saves a round trip.
pub fn parse_ssd(
    g: &mut GComputation,
    inp: GMat,
    _in_sz: (i32, i32),
    _confidence_threshold: f32,
) -> GMat {
    unary(g, "org.opencv.nn.parsers.parseSSD", inp)
}

/// `cv::gapi::parseYolo(in, inSz, confidenceThreshold, nmsThreshold, anchors)`.
pub fn parse_yolo(
    g: &mut GComputation,
    inp: GMat,
    _in_sz: (i32, i32),
    _confidence_threshold: f32,
    _nms_threshold: f32,
) -> GMat {
    unary(g, "org.opencv.nn.parsers.parseYolo", inp)
}

/// `cv::gapi::wip::draw::render(bgr, prims)` — draw onto a three-channel image.
pub fn render(g: &mut GComputation, bgr: GMat) -> GMat {
    unary(g, "org.opencv.render.bgr", bgr)
}

/// `cv::gapi::wip::draw::render3ch(src, prims)`.
pub fn render3ch(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.render.bgr", src)
}

/// `cv::gapi::wip::draw::renderNV12(y, uv, prims)`.
///
/// Draws directly on the NV12 planes. Converting to BGR to draw and back again
/// would cost two full passes per frame *and* lose chroma precision to the 4:2:0
/// resampling — twice.
pub fn render_nv12(g: &mut GComputation, y_plane: GMat, uv_plane: GMat) -> (GMat, GMat) {
    let y_out = unary(g, "org.opencv.render.nv12", y_plane);
    let uv_out = unary(g, "org.opencv.render.nv12", uv_plane);
    (y_out, uv_out)
}

/// `cv::gapi::wip::draw::renderFrame(frame, prims)`.
pub fn render_frame(g: &mut GComputation, frame: GMat) -> GMat {
    unary(g, "org.opencv.render.frame", frame)
}

/// `cv::gapi::streaming::encode(...)` — hand a frame to an encoder.
pub fn encode(g: &mut GComputation, src: GMat) -> GMat {
    unary(g, "org.opencv.streaming.encode", src)
}

/// `cv::gapi::calib3d::StereoInitParam` — how the stereo kernel is set up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoInitParam {
    /// `numDisparities`.
    pub num_disparities: i32,
    /// `blockSize`.
    pub block_size: i32,
    /// `baseline`, in the same units the depth output will use.
    pub baseline: f64,
    /// `focus` — the focal length, in pixels.
    pub focus: f64,
}

impl Default for StereoInitParam {
    /// Upstream's defaults.
    fn default() -> Self {
        StereoInitParam {
            num_disparities: 0,
            block_size: 21,
            baseline: 63.0,
            focus: 3.6,
        }
    }
}

impl StereoInitParam {
    /// `cv::gapi::calib3d::StereoInitParam::StereoInitParam(nD, bS, bL, f)`.
    pub fn new(num_disparities: i32, block_size: i32, baseline: f64, focus: f64) -> Result<Self> {
        cv_assert!(block_size > 0 && block_size % 2 == 1, "blockSize must be odd and positive");
        cv_assert!(baseline > 0.0, "the baseline must be positive");
        cv_assert!(focus > 0.0, "the focal length must be positive");
        Ok(StereoInitParam { num_disparities, block_size, baseline, focus })
    }
}

/// `cv::gapi::KalmanParams` — the filter's matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct KalmanParams {
    /// `transitionMatrix` — how the state evolves with no input.
    pub transition_matrix: Vec<f64>,
    /// `measurementMatrix` — how the state maps to what is observed.
    pub measurement_matrix: Vec<f64>,
    /// `processNoiseCov`.
    pub process_noise_cov: Vec<f64>,
    /// `measurementNoiseCov`.
    pub measurement_noise_cov: Vec<f64>,
    /// `state` — the initial estimate.
    pub state: Vec<f64>,
    /// `errorCov` — the initial covariance.
    pub error_cov: Vec<f64>,
}

impl KalmanParams {
    /// A constant-velocity filter over `dp` state dimensions and `mp`
    /// measurements.
    ///
    /// The two noise covariances are the only settings that matter in practice,
    /// and they trade against each other: a large process noise tracks a
    /// manoeuvre and is jittery, a small one is smooth and lags.
    pub fn new(dp: usize, mp: usize) -> Result<Self> {
        cv_assert!(dp > 0 && mp > 0, "the state and measurement must be non-empty");
        cv_assert!(mp <= dp, "there cannot be more measurements than state dimensions");
        let mut transition = vec![0.0; dp * dp];
        for i in 0..dp {
            transition[i * dp + i] = 1.0;
        }
        let mut measurement = vec![0.0; mp * dp];
        for i in 0..mp {
            measurement[i * dp + i] = 1.0;
        }
        let mut process = vec![0.0; dp * dp];
        for i in 0..dp {
            process[i * dp + i] = 1e-5;
        }
        let mut meas_noise = vec![0.0; mp * mp];
        for i in 0..mp {
            meas_noise[i * mp + i] = 1e-1;
        }
        let mut error = vec![0.0; dp * dp];
        for i in 0..dp {
            error[i * dp + i] = 1.0;
        }
        Ok(KalmanParams {
            transition_matrix: transition,
            measurement_matrix: measurement,
            process_noise_cov: process,
            measurement_noise_cov: meas_noise,
            state: vec![0.0; dp],
            error_cov: error,
        })
    }
}

/// `cv::gapi::ot::ObjectTrackerParams` — the object tracker's settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectTrackerParams {
    /// `max_num_objects` — negative means no limit.
    pub max_num_objects: i32,
    /// `input_image_format` — 0 for BGR, 1 for NV12, 2 for I420.
    pub input_image_format: i32,
    /// `tracking_per_class` — keep a separate track set per class.
    ///
    /// With this off, a car and a pedestrian that cross can swap identities;
    /// with it on they cannot, because they are never candidates for each
    /// other.
    pub tracking_per_class: bool,
}

impl Default for ObjectTrackerParams {
    /// Upstream's defaults.
    fn default() -> Self {
        ObjectTrackerParams {
            max_num_objects: -1,
            input_image_format: 0,
            tracking_per_class: true,
        }
    }
}

/// `cv::gapi::ot::TrackingStatus` accompanies each tracked object; this is the
/// object itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackedObject {
    /// Bounding box.
    pub rect: crate::graph::Rect,
    /// Track id, stable across frames.
    pub tracking_id: i32,
    /// Class label.
    pub label: i32,
    /// Tracking state.
    pub status: crate::types::TrackerStatus,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::MatLike;

    /// A graph with one input, ready to record onto.
    fn graph() -> (GComputation, GMat) {
        let mut g = GComputation::new();
        let inp = g.input();
        (g, inp)
    }

    #[test]
    fn every_unary_kernel_appends_exactly_one_node() {
        let (mut g, inp) = graph();
        let before = g.len();
        let out = sqrt(&mut g, inp);
        assert_eq!(g.len(), before + 1, "one node per kernel");
        assert_ne!(out, inp, "the result is a new value, not the input");
    }

    #[test]
    fn the_arithmetic_kernels_record_their_own_names() {
        let (mut g, a) = graph();
        let b = g.input();
        let _ = add_weighted(&mut g, a, 0.5, b, 0.5, 0.0);
        let _ = sub_c(&mut g, a, GScalar::default());
        let _ = sub_rc(&mut g, GScalar::default(), a);
        let _ = abs_diff_c(&mut g, a, GScalar::default());
        let _ = div(&mut g, a, b, 1.0);
        let _ = div_c(&mut g, a, GScalar::default(), 1.0);
        let _ = div_rc(&mut g, GScalar::default(), a, 1.0);
        let _ = max(&mut g, a, b);
        let _ = min(&mut g, a, b);
        let _ = bitwise_xor(&mut g, a, b);
        let ids = g.node_ids();
        // The three subtraction and division variants must be distinguishable:
        // collapsing them would make the graph's arity ambiguous.
        assert!(ids.iter().any(|i| i.ends_with("subC")));
        assert!(ids.iter().any(|i| i.ends_with("subRC")));
        assert!(ids.iter().any(|i| i.ends_with("divC")));
        assert!(ids.iter().any(|i| i.ends_with("divRC")));
        assert!(ids.iter().any(|i| i.ends_with("math.div")));
    }

    #[test]
    fn the_six_comparisons_are_six_distinct_kernels() {
        let (mut g, a) = graph();
        let b = g.input();
        let _ = cmp_gt(&mut g, a, b);
        let _ = cmp_ge(&mut g, a, b);
        let _ = cmp_le(&mut g, a, b);
        let _ = cmp_lt(&mut g, a, b);
        let _ = cmp_eq(&mut g, a, b);
        let _ = cmp_ne(&mut g, a, b);
        let mut ids = g.node_ids();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 6, "each predicate is its own kernel: {ids:?}");
    }

    #[test]
    fn the_reductions_produce_scalars_not_matrices() {
        let (mut g, inp) = graph();
        let s = sum(&mut g, inp);
        let m = mean(&mut g, inp);
        let n = count_non_zero(&mut g, inp);
        let l1 = norm_l1(&mut g, inp);
        let l2 = norm_l2(&mut g, inp);
        let li = norm_inf(&mut g, inp);
        // Six distinct scalar results.
        let mut all = vec![s, m, n, l1, l2, li];
        all.sort_by_key(|x| format!("{x:?}"));
        all.dedup();
        assert_eq!(all.len(), 6);
        let mut ids = g.node_ids();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 6);
    }

    #[test]
    fn the_two_output_kernels_return_two_distinct_handles() {
        let (mut g, x) = graph();
        let y = g.input();
        let (mag, ang) = cart_to_polar(&mut g, x, y, false);
        assert_ne!(mag, ang);
        let (px, py) = polar_to_cart(&mut g, mag, ang, false);
        assert_ne!(px, py);
        let (dx, dy) = sobel_xy(&mut g, x, 1, 3);
        assert_ne!(dx, dy);
        let (s, sq) = integral(&mut g, x, 0, 0);
        assert_ne!(s, sq);
        let (yo, uvo) = render_nv12(&mut g, x, y);
        assert_ne!(yo, uvo);
    }

    #[test]
    fn every_colour_conversion_is_its_own_kernel() {
        let (mut g, inp) = graph();
        let _ = bgr2rgb(&mut g, inp);
        let _ = rgb2gray(&mut g, inp);
        let _ = rgb2hsv(&mut g, inp);
        let _ = rgb2lab(&mut g, inp);
        let _ = bgr2luv(&mut g, inp);
        let _ = luv2bgr(&mut g, inp);
        let _ = bgr2yuv(&mut g, inp);
        let _ = yuv2bgr(&mut g, inp);
        let _ = rgb2yuv(&mut g, inp);
        let _ = yuv2rgb(&mut g, inp);
        let _ = rgb2yuv422(&mut g, inp);
        let _ = bgr2i420(&mut g, inp);
        let _ = rgb2i420(&mut g, inp);
        let _ = i4202bgr(&mut g, inp);
        let _ = i4202rgb(&mut g, inp);
        let _ = bayer_gr2rgb(&mut g, inp);
        let _ = nv12to_rgb(&mut g, inp);
        let _ = nv12to_bgr(&mut g, inp);
        let _ = nv12to_gray(&mut g, inp);
        let _ = nv12to_rgbp(&mut g, inp);
        let _ = nv12to_bgrp(&mut g, inp);
        let mut ids = g.node_ids();
        let total = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), total, "no two conversions share a kernel id");
        assert_eq!(total, 21);
    }

    #[test]
    fn the_plane_selectors_and_layout_kernels_record() {
        let (mut g, inp) = graph();
        let a = y(&mut g, inp);
        let b = uv(&mut g, inp);
        assert_ne!(a, b);
        let _ = transpose(&mut g, inp);
        let _ = concat_hor(&mut g, inp, inp);
        let _ = concat_vert(&mut g, inp, inp);
        let _ = crop(&mut g, inp, crate::graph::Rect::new(0, 0, 4, 4));
        let _ = resize_p(&mut g, inp, (8, 8), 1);
        let _ = remap(&mut g, inp, &[0.0], &[0.0]);
        let _ = warp_affine(&mut g, inp, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], (8, 8));
        let _ = warp_perspective(
            &mut g,
            inp,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            (8, 8),
        );
        assert!(g.len() >= 10);
    }

    #[test]
    fn the_filters_and_the_rest_record() {
        let (mut g, inp) = graph();
        let _ = box_filter(&mut g, inp, (3, 3), true);
        let _ = sep_filter(&mut g, inp, &[1.0], &[1.0]);
        let _ = bilateral_filter(&mut g, inp, 5, 20.0, 20.0);
        let _ = erode3x3(&mut g, inp, 1);
        let _ = dilate3x3(&mut g, inp, 1);
        let _ = morphology_ex(&mut g, inp, 2, (3, 3));
        let _ = lut(&mut g, inp, &[0u8; 256]);
        let _ = convert_to(&mut g, inp, 5, 1.0, 0.0);
        let _ = normalize(&mut g, inp, 0.0, 255.0, 4);
        let _ = in_range(&mut g, inp, GScalar::default(), GScalar::default());
        let _ = mask(&mut g, inp, inp);
        let _ = select(&mut g, inp, inp, inp);
        let _ = phase(&mut g, inp, inp, false);
        let _ = bounding_rect(&mut g, inp);
        let _ = fit_line2d(&mut g, inp);
        let _ = fit_line3d(&mut g, inp);
        let _ = good_features_to_track(&mut g, inp, 100, 0.01, 5.0);
        let _ = stereo(&mut g, inp, inp);
        let _ = background_subtractor(&mut g, inp);
        let _ = kalman_filter(&mut g, inp);
        let _ = parse_ssd(&mut g, inp, (300, 300), 0.5);
        let _ = parse_yolo(&mut g, inp, (416, 416), 0.5, 0.4);
        let _ = render(&mut g, inp);
        let _ = render3ch(&mut g, inp);
        let _ = render_frame(&mut g, inp);
        let _ = encode(&mut g, inp);
        assert!(g.len() >= 26);
    }

    #[test]
    fn the_stereo_parameters_reject_a_geometry_that_cannot_work() {
        let d = StereoInitParam::default();
        assert_eq!(d.block_size, 21);
        assert_eq!(d.baseline, 63.0);
        assert_eq!(d.focus, 3.6);
        assert!(StereoInitParam::new(16, 15, 60.0, 3.0).is_ok());
        assert!(StereoInitParam::new(16, 14, 60.0, 3.0).is_err(), "an even block size");
        assert!(StereoInitParam::new(16, 15, 0.0, 3.0).is_err(), "a zero baseline");
        assert!(StereoInitParam::new(16, 15, 60.0, -1.0).is_err());
    }

    #[test]
    fn the_kalman_parameters_are_shaped_from_the_state_and_measurement() {
        let p = KalmanParams::new(4, 2).unwrap();
        assert_eq!(p.transition_matrix.len(), 16);
        assert_eq!(p.measurement_matrix.len(), 8);
        assert_eq!(p.process_noise_cov.len(), 16);
        assert_eq!(p.measurement_noise_cov.len(), 4);
        assert_eq!(p.state.len(), 4);
        assert_eq!(p.error_cov.len(), 16);
        // The transition starts as the identity: no motion model assumed.
        assert_eq!(p.transition_matrix[0], 1.0);
        assert_eq!(p.transition_matrix[5], 1.0);
        assert_eq!(p.transition_matrix[1], 0.0);
        // The measurement noise dominates the process noise by default, so the
        // filter starts smooth rather than jittery.
        assert!(p.measurement_noise_cov[0] > p.process_noise_cov[0]);

        assert!(KalmanParams::new(0, 1).is_err());
        assert!(KalmanParams::new(2, 4).is_err(), "more measurements than state");
    }

    #[test]
    fn the_tracker_parameters_default_to_per_class_tracking() {
        let p = ObjectTrackerParams::default();
        assert_eq!(p.max_num_objects, -1, "no limit");
        assert_eq!(p.input_image_format, 0, "BGR");
        assert!(p.tracking_per_class, "identities cannot swap across classes");

        let t = TrackedObject {
            rect: crate::graph::Rect::new(1, 2, 3, 4),
            tracking_id: 7,
            label: 1,
            status: crate::types::TrackerStatus::New,
        };
        assert_eq!(t.tracking_id, 7);
        assert_eq!(t.rect.width, 3);
    }

    #[test]
    fn a_recorded_graph_still_compiles_and_runs() {
        // The kernels record; the graph must still be usable end to end.
        let mut g = GComputation::new();
        let inp = g.input();
        let grey = g.cvt_color_bgr2gray(inp);
        let out = g.gaussian_blur(grey, 3, 1.0);
        g.set_output(out);
        let compiled = g.compile().unwrap();
        let src = cv_core::Mat::zeros(16, 16, cv_core::types::CV_8UC3).unwrap();
        let result = compiled.apply(&src).unwrap();
        assert_eq!((result.rows(), result.cols()), (16, 16));
    }
}
