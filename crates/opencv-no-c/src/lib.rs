//! A pure-Rust port of OpenCV — no C, no C++, no `build.rs` linking step.
//!
//! Targets upstream **OpenCV 4.15.0-dev**
//! (`77dfa297d08fdecdc509fc01ad92a2e9ec776a57`).
//!
//! This crate is a facade: it re-exports the ported module crates under names
//! that mirror OpenCV's own module layout, so `use opencv_no_c::imgproc::*` sits
//! where `#include <opencv2/imgproc.hpp>` would.
//!
//! ```
//! use opencv_no_c::prelude::*;
//! # #[cfg(feature = "imgproc")]
//! use opencv_no_c::imgproc::{cvt_color, resize, ColorConversion, Interpolation};
//!
//! # #[cfg(feature = "imgproc")]
//! # fn main() -> opencv_no_c::Result<()> {
//! let img = Mat::new_with(64, 64, types::CV_8UC3, Scalar::new(20.0, 60.0, 200.0, 0.0))?;
//! let gray = cvt_color(&img, ColorConversion::Bgr2Gray)?;
//! let small = resize(&gray, Size::new(32, 32), 0.0, 0.0, Interpolation::Area)?;
//!
//! assert_eq!(small.size(), Size::new(32, 32));
//! assert_eq!(small.channels(), 1);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "imgproc"))]
//! # fn main() {}
//! ```
//!
//! # Coverage
//!
//! `cargo xtask report` prints per-module parity against the upstream header
//! surface. Symbol coverage is not numerical agreement; the README separates
//! the two.
//!
//! # Features
//!
//! One feature per OpenCV module (`imgproc`, `imgcodecs`, `videoio`,
//! `highgui`, `video`, `calib3d`, `features2d`, `objdetect`, `dnn`, `ml`,
//! `flann`, `photo`, `stitching`, `gapi`), all on by default through `full`.
//! `core` is always present. To take only what you use:
//!
//! ```toml
//! opencv-no-c = { version = "0.1", default-features = false, features = ["imgproc", "imgcodecs"] }
//! ```
//!
//! Optional extras: `dropin`, `abi`, `compat`, `gui`, `rlx`, `rlx-all`, `onnx`.
//!
//! # Where this intentionally differs from C++
//!
//! - Errors are returned as [`Result`], not thrown.
//! - Matrices own their pixels; sharing is explicit via
//!   [`MatView`](core::MatView) / [`MatViewMut`](core::MatViewMut) rather than
//!   reference counting. See the `cv_core::mat` module docs for why.
//!
//! Numerics are not a divergence: the fixed-point kernels reproduce upstream's
//! integer arithmetic exactly rather than approximating it in floating point.

#![doc(html_root_url = "https://docs.rs/opencv-no-c/0.1.0")]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "dropin")]
#[cfg_attr(docsrs, doc(cfg(feature = "dropin")))]
pub mod dropin;

/// The `core` module: [`Mat`](cv_core::Mat), the type system, arithmetic and statistics.
pub mod core {
    pub use cv_core::*;
}

/// The `imgproc` module: colour conversion, geometric transforms and filters.
#[cfg(feature = "imgproc")]
#[cfg_attr(docsrs, doc(cfg(feature = "imgproc")))]
pub mod imgproc {
    pub use cv_imgproc::*;
}

/// The `imgcodecs` module: reading and writing image files.
///
/// Only BMP and PNM are decoded; formats needing a compression codec (PNG,
/// JPEG, TIFF, WebP) report `StsUnsupportedFormat` rather than failing quietly.
#[cfg(feature = "imgcodecs")]
#[cfg_attr(docsrs, doc(cfg(feature = "imgcodecs")))]
pub mod imgcodecs {
    pub use cv_imgcodecs::*;
}

/// `cv::calib3d` — Rodrigues, homography, PnP, epipolar geometry.
#[cfg(feature = "calib3d")]
#[cfg_attr(docsrs, doc(cfg(feature = "calib3d")))]
pub mod calib3d {
    pub use cv_calib3d::*;
}

/// `cv::photo` — denoising, inpainting, edge-preserving stylisation.
#[cfg(feature = "photo")]
#[cfg_attr(docsrs, doc(cfg(feature = "photo")))]
pub mod photo {
    pub use cv_photo::*;
}

/// `cv::dnn` — blob conversion, non-maximum suppression and a layer graph.
///
/// The heavy layers execute on RLX when the `rlx` feature is on; see the
/// [`cv_dnn`] crate docs.
#[cfg(feature = "dnn")]
#[cfg_attr(docsrs, doc(cfg(feature = "dnn")))]
pub mod dnn {
    pub use cv_dnn::*;
}

/// `cv::features2d` — FAST/ORB keypoints, descriptors and matching.
#[cfg(feature = "features2d")]
#[cfg_attr(docsrs, doc(cfg(feature = "features2d")))]
pub mod features2d {
    pub use cv_features2d::*;
}

/// `cv::flann` — KD-tree, LSH and linear nearest-neighbour indices.
#[cfg(feature = "flann")]
#[cfg_attr(docsrs, doc(cfg(feature = "flann")))]
pub mod flann {
    pub use cv_flann::*;
}

/// `cv::ml` — KNN, Bayes, SVM, trees, MLP and EM.
#[cfg(feature = "ml")]
#[cfg_attr(docsrs, doc(cfg(feature = "ml")))]
pub mod ml {
    pub use cv_ml::*;
}

/// `cv::objdetect` — HOG, Haar cascades, rectangle grouping, QR finder patterns.
#[cfg(feature = "objdetect")]
#[cfg_attr(docsrs, doc(cfg(feature = "objdetect")))]
pub mod objdetect {
    pub use cv_objdetect::*;
}

/// `cv::stitching` — warpers, blenders and a panorama pipeline.
#[cfg(feature = "stitching")]
#[cfg_attr(docsrs, doc(cfg(feature = "stitching")))]
pub mod stitching {
    pub use cv_stitching::*;
}

/// `cv::video` — optical flow, background subtraction, tracking, Kalman.
#[cfg(feature = "video")]
#[cfg_attr(docsrs, doc(cfg(feature = "video")))]
pub mod video {
    pub use cv_video::*;
}

/// `cv::videoio` — AVI, Y4M and image-sequence capture and writing.
#[cfg(feature = "videoio")]
#[cfg_attr(docsrs, doc(cfg(feature = "videoio")))]
pub mod videoio {
    pub use cv_videoio::*;
}

/// `cv::highgui` — windows, keys, trackbars and ROI selection.
///
/// Needs the `gui` feature for a real backend; without it the API still
/// exists and reports that none was compiled in.
#[cfg(feature = "highgui")]
#[cfg_attr(docsrs, doc(cfg(feature = "highgui")))]
pub mod highgui {
    pub use cv_highgui::*;
}

/// `cv::gapi` — a deferred image-processing graph, compiled once per pipeline.
#[cfg(feature = "gapi")]
#[cfg_attr(docsrs, doc(cfg(feature = "gapi")))]
pub mod gapi {
    pub use cv_gapi::*;
}

pub use cv_core::{Code, Error, Result};

/// The types and traits most code needs in scope.
pub mod prelude {
    pub use cv_core::types;
    pub use cv_core::{
        BorderType, Depth, Mat, MatLike, MatLikeMut, MatType, MatView, MatViewMut, Point, Range,
        Rect, Scalar, Size,
    };
}

/// The upstream OpenCV version this port tracks.
pub const CV_VERSION: &str = cv_core::CV_VERSION;
/// Upstream commit the ported semantics were read from.
pub const CV_UPSTREAM_COMMIT: &str = cv_core::CV_UPSTREAM_COMMIT;
