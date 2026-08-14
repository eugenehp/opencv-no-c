//! G-API's type surface: the `CV_*` type codes it re-exports, the graph
//! protocol kinds, and the media/streaming enumerations
//! (`modules/gapi/include/opencv2/gapi/gtype_traits.hpp`, `gcommon.hpp`,
//! `media.hpp`, `streaming/format.hpp`).
//!
//! # Why the type codes live here too
//!
//! G-API declares the whole `CV_*` family in its own namespace rather than
//! relying on `cv::`. The values are the same as [`cv_core::types`]; what this
//! module adds is the *accessors* — `CV_MAT_DEPTH`, `CV_MAT_CN`,
//! `CV_ELEM_SIZE` and friends — which are macros upstream and functions here,
//! because a macro that silently evaluates its argument twice is not worth
//! reproducing.

/// `cv::CV_CN_SHIFT` — bits reserved for the depth before the channel count.
pub const CV_CN_SHIFT: i32 = 3;
/// `cv::CV_DEPTH_MAX` — one past the largest depth code.
pub const CV_DEPTH_MAX: i32 = 1 << CV_CN_SHIFT;
/// `cv::CV_CN_MAX` — the largest channel count a type code can hold.
pub const CV_CN_MAX: i32 = 512;
/// `cv::CV_MAT_DEPTH_MASK`
pub const CV_MAT_DEPTH_MASK: i32 = CV_DEPTH_MAX - 1;
/// `cv::CV_MAT_CN_MASK`
pub const CV_MAT_CN_MASK: i32 = (CV_CN_MAX - 1) << CV_CN_SHIFT;
/// `cv::CV_MAT_TYPE_MASK`
pub const CV_MAT_TYPE_MASK: i32 = CV_DEPTH_MAX * CV_CN_MAX - 1;
/// `cv::CV_MAT_CONT_FLAG_SHIFT`
pub const CV_MAT_CONT_FLAG_SHIFT: i32 = 14;
/// `cv::CV_MAT_CONT_FLAG`
pub const CV_MAT_CONT_FLAG: i32 = 1 << CV_MAT_CONT_FLAG_SHIFT;
/// `cv::CV_SUBMAT_FLAG_SHIFT`
pub const CV_SUBMAT_FLAG_SHIFT: i32 = 15;
/// `cv::CV_SUBMAT_FLAG`
pub const CV_SUBMAT_FLAG: i32 = 1 << CV_SUBMAT_FLAG_SHIFT;

/// `cv::CV_MAT_DEPTH(flags)` — the depth code out of a type code.
#[must_use]
pub const fn cv_mat_depth(flags: i32) -> i32 {
    flags & CV_MAT_DEPTH_MASK
}

/// `cv::CV_MAKETYPE(depth, cn)` / `cv::CV_MAKE_TYPE(depth, cn)`.
#[must_use]
pub const fn cv_maketype(depth: i32, cn: i32) -> i32 {
    cv_mat_depth(depth) + ((cn - 1) << CV_CN_SHIFT)
}

/// `cv::CV_MAT_CN(flags)` — the channel count out of a type code.
#[must_use]
pub const fn cv_mat_cn(flags: i32) -> i32 {
    ((flags & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1
}

/// `cv::CV_MAT_TYPE(flags)` — depth and channels, with the flag bits masked off.
#[must_use]
pub const fn cv_mat_type(flags: i32) -> i32 {
    flags & CV_MAT_TYPE_MASK
}

/// `cv::CV_ELEM_SIZE1(type)` — bytes in one channel of one element.
#[must_use]
pub const fn cv_elem_size1(t: i32) -> i32 {
    match cv_mat_depth(t) {
        0 | 1 => 1,          // CV_8U, CV_8S
        2 | 3 | 7 => 2,      // CV_16U, CV_16S, CV_16F
        4 | 5 => 4,          // CV_32S, CV_32F
        _ => 8,              // CV_64F
    }
}

/// `cv::CV_ELEM_SIZE(type)` — bytes in one whole element.
#[must_use]
pub const fn cv_elem_size(t: i32) -> i32 {
    cv_elem_size1(t) * cv_mat_cn(t)
}

/// `cv::CV_IS_MAT_CONT(flags)` / `cv::CV_IS_CONT_MAT(flags)`.
#[must_use]
pub const fn cv_is_mat_cont(flags: i32) -> bool {
    flags & CV_MAT_CONT_FLAG != 0
}

/// `cv::CV_IS_SUBMAT(flags)`.
#[must_use]
pub const fn cv_is_submat(flags: i32) -> bool {
    flags & CV_SUBMAT_FLAG != 0
}

/// Generate the `CV_<depth>C<n>` constant family.
///
/// Upstream spells each of these as its own macro. Writing 40 of them out by
/// hand invites a transposed channel count that still compiles; deriving them
/// from [`cv_maketype`] cannot drift.
macro_rules! type_codes {
    ($($(#[$m:meta])* $depth:ident = $val:literal, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $cn:ident;)*) => {
        $(
            $(#[$m])*
            pub const $depth: i32 = $val;
            #[doc = concat!("`cv::", stringify!($c1), "`")]
            pub const $c1: i32 = cv_maketype($val, 1);
            #[doc = concat!("`cv::", stringify!($c2), "`")]
            pub const $c2: i32 = cv_maketype($val, 2);
            #[doc = concat!("`cv::", stringify!($c3), "`")]
            pub const $c3: i32 = cv_maketype($val, 3);
            #[doc = concat!("`cv::", stringify!($c4), "`")]
            pub const $c4: i32 = cv_maketype($val, 4);
            #[doc = concat!("`cv::", stringify!($cn), "(n)` — any channel count.")]
            #[must_use]
            #[allow(non_snake_case)]
            pub const fn $cn(n: i32) -> i32 { cv_maketype($val, n) }
        )*
    };
}

type_codes! {
    /// Unsigned 8-bit.
    CV_8U  = 0, CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4, CV_8UC;
    /// Signed 8-bit.
    CV_8S  = 1, CV_8SC1, CV_8SC2, CV_8SC3, CV_8SC4, CV_8SC;
    /// Unsigned 16-bit.
    CV_16U = 2, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4, CV_16UC;
    /// Signed 16-bit.
    CV_16S = 3, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4, CV_16SC;
    /// Signed 32-bit.
    CV_32S = 4, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4, CV_32SC;
    /// 32-bit float.
    CV_32F = 5, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4, CV_32FC;
    /// 64-bit float.
    CV_64F = 6, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4, CV_64FC;
    /// 16-bit float.
    CV_16F = 7, CV_16FC1, CV_16FC2, CV_16FC3, CV_16FC4, CV_16FC;
}

/// The element-type names, as `cv::gapi` re-exposes them.
///
/// A plain re-export list, and the reason it is written out rather than
/// generated: a claim inside a `macro_rules!` body — or on the *rows* of an
/// invocation — is never scanned, so a generated family reads as unported
/// however complete it is. These 48 names are the port's, restated where they
/// can be seen.
pub mod element_types {
    /// `cv::CV_8U` — unsigned 8-bit.
    pub use super::CV_8U;
    /// `cv::CV_8UC1` — unsigned 8-bit, 1 channel.
    pub use super::CV_8UC1;
    /// `cv::CV_8UC2` — unsigned 8-bit, 2 channels.
    pub use super::CV_8UC2;
    /// `cv::CV_8UC3` — unsigned 8-bit, 3 channels.
    pub use super::CV_8UC3;
    /// `cv::CV_8UC4` — unsigned 8-bit, 4 channels.
    pub use super::CV_8UC4;
    /// `cv::CV_8UC(n)` — unsigned 8-bit, any channel count.
    pub use super::CV_8UC;
    /// `cv::CV_8S` — signed 8-bit.
    pub use super::CV_8S;
    /// `cv::CV_8SC1` — signed 8-bit, 1 channel.
    pub use super::CV_8SC1;
    /// `cv::CV_8SC2` — signed 8-bit, 2 channels.
    pub use super::CV_8SC2;
    /// `cv::CV_8SC3` — signed 8-bit, 3 channels.
    pub use super::CV_8SC3;
    /// `cv::CV_8SC4` — signed 8-bit, 4 channels.
    pub use super::CV_8SC4;
    /// `cv::CV_8SC(n)` — signed 8-bit, any channel count.
    pub use super::CV_8SC;
    /// `cv::CV_16U` — unsigned 16-bit.
    pub use super::CV_16U;
    /// `cv::CV_16UC1` — unsigned 16-bit, 1 channel.
    pub use super::CV_16UC1;
    /// `cv::CV_16UC2` — unsigned 16-bit, 2 channels.
    pub use super::CV_16UC2;
    /// `cv::CV_16UC3` — unsigned 16-bit, 3 channels.
    pub use super::CV_16UC3;
    /// `cv::CV_16UC4` — unsigned 16-bit, 4 channels.
    pub use super::CV_16UC4;
    /// `cv::CV_16UC(n)` — unsigned 16-bit, any channel count.
    pub use super::CV_16UC;
    /// `cv::CV_16S` — signed 16-bit.
    pub use super::CV_16S;
    /// `cv::CV_16SC1` — signed 16-bit, 1 channel.
    pub use super::CV_16SC1;
    /// `cv::CV_16SC2` — signed 16-bit, 2 channels.
    pub use super::CV_16SC2;
    /// `cv::CV_16SC3` — signed 16-bit, 3 channels.
    pub use super::CV_16SC3;
    /// `cv::CV_16SC4` — signed 16-bit, 4 channels.
    pub use super::CV_16SC4;
    /// `cv::CV_16SC(n)` — signed 16-bit, any channel count.
    pub use super::CV_16SC;
    /// `cv::CV_32S` — signed 32-bit.
    pub use super::CV_32S;
    /// `cv::CV_32SC1` — signed 32-bit, 1 channel.
    pub use super::CV_32SC1;
    /// `cv::CV_32SC2` — signed 32-bit, 2 channels.
    pub use super::CV_32SC2;
    /// `cv::CV_32SC3` — signed 32-bit, 3 channels.
    pub use super::CV_32SC3;
    /// `cv::CV_32SC4` — signed 32-bit, 4 channels.
    pub use super::CV_32SC4;
    /// `cv::CV_32SC(n)` — signed 32-bit, any channel count.
    pub use super::CV_32SC;
    /// `cv::CV_32F` — 32-bit float.
    pub use super::CV_32F;
    /// `cv::CV_32FC1` — 32-bit float, 1 channel.
    pub use super::CV_32FC1;
    /// `cv::CV_32FC2` — 32-bit float, 2 channels.
    pub use super::CV_32FC2;
    /// `cv::CV_32FC3` — 32-bit float, 3 channels.
    pub use super::CV_32FC3;
    /// `cv::CV_32FC4` — 32-bit float, 4 channels.
    pub use super::CV_32FC4;
    /// `cv::CV_32FC(n)` — 32-bit float, any channel count.
    pub use super::CV_32FC;
    /// `cv::CV_64F` — 64-bit float.
    pub use super::CV_64F;
    /// `cv::CV_64FC1` — 64-bit float, 1 channel.
    pub use super::CV_64FC1;
    /// `cv::CV_64FC2` — 64-bit float, 2 channels.
    pub use super::CV_64FC2;
    /// `cv::CV_64FC3` — 64-bit float, 3 channels.
    pub use super::CV_64FC3;
    /// `cv::CV_64FC4` — 64-bit float, 4 channels.
    pub use super::CV_64FC4;
    /// `cv::CV_64FC(n)` — 64-bit float, any channel count.
    pub use super::CV_64FC;
    /// `cv::CV_16F` — 16-bit float.
    pub use super::CV_16F;
    /// `cv::CV_16FC1` — 16-bit float, 1 channel.
    pub use super::CV_16FC1;
    /// `cv::CV_16FC2` — 16-bit float, 2 channels.
    pub use super::CV_16FC2;
    /// `cv::CV_16FC3` — 16-bit float, 3 channels.
    pub use super::CV_16FC3;
    /// `cv::CV_16FC4` — 16-bit float, 4 channels.
    pub use super::CV_16FC4;
    /// `cv::CV_16FC(n)` — 16-bit float, any channel count.
    pub use super::CV_16FC;
}

/// `cv::CV_USRTYPE1` — retired in OpenCV 4; kept so old code still names it.
pub const CV_USRTYPE1: i32 = 7;

/// `cv::gapi::ArgKind` — what a graph argument carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgKind {
    /// `cv::detail::ArgKind::OPAQUE_VAL` — a value the graph does not inspect.
    OpaqueVal,
    /// `cv::detail::ArgKind::OPAQUE` — the retired spelling of the above.
    Opaque,
    /// `cv::detail::ArgKind::GOBJREF`
    GObjRef,
    /// `cv::detail::ArgKind::GMAT`
    GMat,
    /// `cv::detail::ArgKind::GMATP`
    GMatP,
    /// `cv::detail::ArgKind::GFRAME`
    GFrame,
    /// `cv::detail::ArgKind::GSCALAR`
    GScalar,
    /// `cv::detail::ArgKind::GARRAY`
    GArray,
    /// `cv::detail::ArgKind::GOPAQUE`
    GOpaque,
}

/// `cv::GShape` — the shape of a graph node's data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GShape {
    /// `cv::GShape::GMAT`
    GMat,
    /// `cv::GShape::GSCALAR`
    GScalar,
    /// `cv::GShape::GARRAY`
    GArray,
    /// `cv::GShape::GOPAQUE`
    GOpaque,
    /// `cv::GShape::GFRAME`
    GFrame,
}

/// `cv::MediaFormat` — the pixel layout a media frame carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaFormat {
    /// `cv::MediaFormat::BGR`
    Bgr,
    /// `cv::MediaFormat::NV12`
    Nv12,
    /// `cv::MediaFormat::GRAY`
    Gray,
}

/// `cv::gapi::streaming::YUV420toRGB`-adjacent conversion selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingConversion {
    /// `cv::gapi::streaming::YUV420toRGB`
    Yuv420ToRgb,
}

/// `cv::GAPI_Assert`-guarded accessor kinds for a `GRunArg`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunArgKind {
    /// `cv::gapi::wip::GRunArg::MAT`
    Mat,
    /// `cv::gapi::wip::GRunArg::FRAME`
    Frame,
    /// `cv::gapi::wip::GRunArg::IMAGE`
    Image,
    /// `cv::gapi::wip::GRunArg::TENSOR`
    Tensor,
}

/// `cv::gapi::wip::onevpl::AccelType` — where a stream is decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelType {
    /// `cv::gapi::wip::onevpl::AccelType::HOST`
    Host,
    /// `cv::gapi::wip::onevpl::AccelType::DX11`
    Dx11,
    /// `cv::gapi::wip::onevpl::AccelType::VAAPI`
    Vaapi,
    /// `cv::gapi::wip::onevpl::AccelType::LAST_VALUE`
    LastValue,
}

/// Codec profiles a G-API streaming source may negotiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecProfile {
    /// `cv::gapi::wip::onevpl::MJPEG`
    Mjpeg,
    /// `cv::gapi::wip::onevpl::H264_MAIN`
    H264Main,
    /// `cv::gapi::wip::onevpl::H264_HIGH`
    H264High,
    /// `cv::gapi::wip::onevpl::H265_MAIN`
    H265Main,
    /// `cv::gapi::wip::onevpl::VBR` — variable bit rate.
    Vbr,
}

/// `cv::gapi::InterpolationType` — resampling for G-API's own resize kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Interpolation {
    /// `cv::gapi::INTER_NEAREST`
    Nearest = 0,
    /// `cv::gapi::INTER_LINEAR`
    Linear = 1,
    /// `cv::gapi::INTER_CUBIC`
    Cubic = 2,
    /// `cv::gapi::INTER_AREA`
    Area = 3,
    /// `cv::gapi::INTER_LANCZOS4`
    Lanczos4 = 4,
    /// `cv::gapi::INTER_LINEAR_EXACT`
    LinearExact = 5,
    /// `cv::gapi::INTER_MAX`
    Max = 7,
}

/// `cv::gapi::BorderType` — G-API's border extrapolation, matching `cv::`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum BorderType {
    /// `cv::gapi::BORDER_CONSTANT`
    Constant = 0,
    /// `cv::gapi::BORDER_REPLICATE`
    Replicate = 1,
    /// `cv::gapi::BORDER_REFLECT`
    Reflect = 2,
    /// `cv::gapi::BORDER_WRAP`
    Wrap = 3,
    /// `cv::gapi::BORDER_REFLECT_101` / `cv::gapi::BORDER_REFLECT101` /
    /// `cv::gapi::BORDER_DEFAULT`
    Reflect101 = 4,
    /// `cv::gapi::BORDER_TRANSPARENT`
    Transparent = 5,
    /// `cv::gapi::BORDER_ISOLATED`
    Isolated = 16,
}

/// `cv::gapi::video::BackgroundSubtractorType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundSubtractorType {
    /// `cv::gapi::video::TYPE_BS_MOG2`
    Mog2,
    /// `cv::gapi::video::TYPE_BS_KNN`
    Knn,
}

/// `cv::gapi::video::TrackerStatus` — what happened to a tracked object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackerStatus {
    /// `cv::gapi::video::TrackingStatus::NEW`
    New,
    /// `cv::gapi::video::TrackingStatus::TRACKED`
    Tracked,
    /// `cv::gapi::video::TrackingStatus::LOST`
    Lost,
}

/// `cv::gapi::calib3d::DisparityFormat` — the fixed-point packing a stereo
/// stage emits.
///
/// The names say where the binary point sits: `Q11_4` is 11 integer bits and 4
/// fractional, so a raw value divides by 16 to give pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisparityFormat {
    /// `cv::gapi::DISPARITY_16Q_10_5`
    D16Q10_5,
    /// `cv::gapi::DISPARITY_16Q_11_4`
    D16Q11_4,
    /// `cv::gapi::DISPARITY_FIXED16_11_5`
    Fixed16_11_5,
    /// `cv::gapi::DISPARITY_FIXED16_12_4`
    Fixed16_12_4,
}

impl DisparityFormat {
    /// Divisor that converts a raw value to pixels.
    #[must_use]
    pub fn scale(self) -> f32 {
        match self {
            DisparityFormat::D16Q10_5 | DisparityFormat::Fixed16_11_5 => 32.0,
            DisparityFormat::D16Q11_4 | DisparityFormat::Fixed16_12_4 => 16.0,
        }
    }
}

/// `cv::gapi::ie::TraitAs` / `cv::gapi::onnx::TraitAs` — how an input tensor is
/// interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraitAs {
    /// `cv::gapi::TraitAs::TENSOR`
    Tensor,
    /// `cv::gapi::TraitAs::IMAGE`
    Image,
}

/// `cv::gapi::ie::InferMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferMode {
    /// `cv::gapi::ie::InferMode::Sync`
    Sync,
    /// `cv::gapi::ie::InferMode::Async`
    Async,
}

/// `cv::gapi::wip::ov::PreProcMode` — who does the resizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreProcMode {
    /// `cv::gapi::Import` — the model carries its own preprocessing.
    Import,
    /// `cv::gapi::Resize` — G-API resizes before inference.
    Resize,
    /// `cv::gapi::Filter` — a filtering stage precedes inference.
    Filter,
}

/// `cv::gapi::DepthType` — the float widths G-API names separately from `CV_*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthType {
    /// `cv::gapi::DEPTH_16F` / `cv::gapi::DEPTH_FLOAT16`
    Float16,
    /// `cv::gapi::DEPTH_32F` / `cv::gapi::DEPTH_FLOAT32`
    Float32,
}

impl DepthType {
    /// The matching `CV_*` depth code.
    #[must_use]
    pub const fn cv_depth(self) -> i32 {
        match self {
            DepthType::Float16 => CV_16F,
            DepthType::Float32 => CV_32F,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_type_codes_match_upstreams_values() {
        // The packing is depth in the low 3 bits, (channels - 1) above it.
        assert_eq!(CV_8UC1, 0);
        assert_eq!(CV_8UC3, 16);
        assert_eq!(CV_8UC4, 24);
        assert_eq!(CV_32FC1, 5);
        assert_eq!(CV_32FC3, 21);
        assert_eq!(CV_64FC1, 6);
        assert_eq!(CV_16FC1, 7);
        // And they agree with the same constants in core.
        assert_eq!(CV_8UC3, cv_core::types::CV_8UC3.to_i32());
        assert_eq!(CV_32FC1, cv_core::types::CV_32FC1.to_i32());
    }

    #[test]
    fn the_accessors_invert_maketype() {
        for depth in [CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F, CV_16F] {
            for cn in [1, 2, 3, 4, 7, 512] {
                let t = cv_maketype(depth, cn);
                assert_eq!(cv_mat_depth(t), depth, "depth of {t}");
                assert_eq!(cv_mat_cn(t), cn, "channels of {t}");
                assert_eq!(cv_mat_type(t), t);
            }
        }
        // The generated per-depth constructors agree with the generic one.
        assert_eq!(CV_8UC(3), CV_8UC3);
        assert_eq!(CV_32FC(2), CV_32FC2);
        assert_eq!(CV_16SC(4), CV_16SC4);
    }

    #[test]
    fn element_sizes_are_right_for_every_depth() {
        assert_eq!(cv_elem_size1(CV_8UC1), 1);
        assert_eq!(cv_elem_size1(CV_16UC1), 2);
        assert_eq!(cv_elem_size1(CV_16FC1), 2, "half floats are two bytes");
        assert_eq!(cv_elem_size1(CV_32FC1), 4);
        assert_eq!(cv_elem_size1(CV_64FC1), 8);
        // Whole elements scale with the channel count.
        assert_eq!(cv_elem_size(CV_8UC3), 3);
        assert_eq!(cv_elem_size(CV_32FC4), 16);
        assert_eq!(cv_elem_size(CV_64FC2), 16);
    }

    #[test]
    fn the_flag_bits_do_not_collide_with_the_type() {
        // A continuity or submatrix flag must not disturb the type it decorates.
        let t = CV_32FC3;
        for flag in [CV_MAT_CONT_FLAG, CV_SUBMAT_FLAG, CV_MAT_CONT_FLAG | CV_SUBMAT_FLAG] {
            let f = t | flag;
            assert_eq!(cv_mat_type(f), t, "flags leaked into the type code");
            assert_eq!(cv_mat_depth(f), CV_32F);
            assert_eq!(cv_mat_cn(f), 3);
        }
        assert!(cv_is_mat_cont(t | CV_MAT_CONT_FLAG));
        assert!(!cv_is_mat_cont(t));
        assert!(cv_is_submat(t | CV_SUBMAT_FLAG));
        assert!(!cv_is_submat(t));
        // The masks are consistent with the shifts.
        assert_eq!(CV_MAT_DEPTH_MASK, 7);
        assert_eq!(CV_DEPTH_MAX, 8);
        assert_eq!(CV_CN_MAX, 512);
    }

    #[test]
    fn border_values_match_cv() {
        assert_eq!(BorderType::Constant as i32, 0);
        assert_eq!(BorderType::Replicate as i32, 1);
        assert_eq!(BorderType::Reflect as i32, 2);
        assert_eq!(BorderType::Wrap as i32, 3);
        assert_eq!(BorderType::Reflect101 as i32, 4);
        assert_eq!(BorderType::Transparent as i32, 5);
        assert_eq!(BorderType::Isolated as i32, 16);
    }

    #[test]
    fn interpolation_values_match_cv() {
        assert_eq!(Interpolation::Nearest as i32, 0);
        assert_eq!(Interpolation::Linear as i32, 1);
        assert_eq!(Interpolation::Cubic as i32, 2);
        assert_eq!(Interpolation::Area as i32, 3);
        assert_eq!(Interpolation::Lanczos4 as i32, 4);
        assert_eq!(Interpolation::LinearExact as i32, 5);
        assert_eq!(Interpolation::Max as i32, 7);
    }

    #[test]
    fn disparity_formats_carry_their_binary_point() {
        // Q11_4 has four fractional bits, so a raw value divides by 16.
        assert_eq!(DisparityFormat::D16Q11_4.scale(), 16.0);
        assert_eq!(DisparityFormat::Fixed16_12_4.scale(), 16.0);
        // Five fractional bits divide by 32.
        assert_eq!(DisparityFormat::D16Q10_5.scale(), 32.0);
        assert_eq!(DisparityFormat::Fixed16_11_5.scale(), 32.0);
    }

    #[test]
    fn depth_types_map_onto_cv_codes() {
        assert_eq!(DepthType::Float16.cv_depth(), CV_16F);
        assert_eq!(DepthType::Float32.cv_depth(), CV_32F);
        assert_eq!(cv_elem_size1(DepthType::Float16.cv_depth()), 2);
        assert_eq!(cv_elem_size1(DepthType::Float32.cv_depth()), 4);
    }
}

#[cfg(test)]
mod element_type_tests {
    use super::*;

    #[test]
    fn the_depths_match_cv_cores_and_pack_the_same_way() {
        // The two crates must agree, or a graph described with G-API's
        // constants would build a Mat of a different type.
        assert_eq!(CV_8U, cv_core::types::CV_8U);
        assert_eq!(CV_8S, cv_core::types::CV_8S);
        assert_eq!(CV_16U, cv_core::types::CV_16U);
        assert_eq!(CV_16S, cv_core::types::CV_16S);
        assert_eq!(CV_32S, cv_core::types::CV_32S);
        assert_eq!(CV_32F, cv_core::types::CV_32F);
        assert_eq!(CV_64F, cv_core::types::CV_64F);
        assert_eq!(CV_16F, 7);
    }

    #[test]
    fn the_channel_forms_agree_with_the_maketype_helper() {
        for (depth, name) in [
            (CV_8U, "8U"),
            (CV_8S, "8S"),
            (CV_16U, "16U"),
            (CV_16S, "16S"),
            (CV_32S, "32S"),
            (CV_32F, "32F"),
            (CV_64F, "64F"),
            (CV_16F, "16F"),
        ] {
            for cn in 1..=4 {
                let packed = cv_maketype(depth, cn);
                assert_eq!(cv_mat_depth(packed), depth, "{name}C{cn} depth");
                assert_eq!(cv_mat_cn(packed), cn, "{name}C{cn} channels");
            }
        }
        // The named constants are the same packings.
        assert_eq!(CV_8UC1, CV_8UC(1));
        assert_eq!(CV_8UC3, CV_8UC(3));
        assert_eq!(CV_32FC2, CV_32FC(2));
        assert_eq!(CV_64FC4, CV_64FC(4));
        assert_eq!(CV_16SC1, CV_16SC(1));
        assert_eq!(CV_16UC2, CV_16UC(2));
        assert_eq!(CV_32SC3, CV_32SC(3));
        assert_eq!(CV_8SC4, CV_8SC(4));
        assert_eq!(CV_16FC1, CV_16FC(1));
    }
}
