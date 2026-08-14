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
//! use opencv_no_c::imgproc::{cvt_color, resize, ColorConversion, Interpolation};
//!
//! # fn main() -> opencv_no_c::Result<()> {
//! let img = Mat::new_with(64, 64, types::CV_8UC3, Scalar::new(20.0, 60.0, 200.0, 0.0))?;
//! let gray = cvt_color(&img, ColorConversion::Bgr2Gray)?;
//! let small = resize(&gray, Size::new(32, 32), 0.0, 0.0, Interpolation::Area)?;
//!
//! assert_eq!(small.size(), Size::new(32, 32));
//! assert_eq!(small.channels(), 1);
//! # Ok(())
//! # }
//! ```
//!
//! # Coverage
//!
//! This is an in-progress port, not a finished one. `cargo xtask report` prints
//! the current per-module parity against the upstream header surface; run it
//! before assuming a given function exists. `core` and `imgproc` are the only
//! modules with implementations today.
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

#[cfg(feature = "dropin")]
pub mod dropin;

/// The `core` module: [`Mat`](cv_core::Mat), the type system, arithmetic and statistics.
pub mod core {
    pub use cv_core::*;
}

/// The `imgproc` module: colour conversion, geometric transforms and filters.
pub mod imgproc {
    pub use cv_imgproc::*;
}

/// The `imgcodecs` module: reading and writing image files.
///
/// Only BMP and PNM are decoded; formats needing a compression codec (PNG,
/// JPEG, TIFF, WebP) report `StsUnsupportedFormat` rather than failing quietly.
pub mod imgcodecs {
    pub use cv_imgcodecs::*;
}

/// `cv::calib3d` — Rodrigues, homography, PnP, epipolar geometry.
pub mod calib3d {
    pub use cv_calib3d::*;
}

/// `cv::photo` — denoising, inpainting, edge-preserving stylisation.
pub mod photo {
    pub use cv_photo::*;
}

/// `cv::dnn` — blob conversion, non-maximum suppression and a layer graph.
///
/// The heavy layers execute on RLX when the `rlx` feature is on; see the
/// [`cv_dnn`] crate docs.
pub mod dnn {
    pub use cv_dnn::*;
}

/// `cv::features2d` — FAST/ORB keypoints, descriptors and matching.
pub mod features2d {
    pub use cv_features2d::*;
}

/// `cv::flann` — KD-tree, LSH and linear nearest-neighbour indices.
pub mod flann {
    pub use cv_flann::*;
}

/// `cv::ml` — KNN, Bayes, SVM, trees, MLP and EM.
pub mod ml {
    pub use cv_ml::*;
}

/// `cv::objdetect` — HOG, Haar cascades, rectangle grouping, QR finder patterns.
pub mod objdetect {
    pub use cv_objdetect::*;
}

/// `cv::stitching` — warpers, blenders and a panorama pipeline.
pub mod stitching {
    pub use cv_stitching::*;
}

/// `cv::video` — optical flow, background subtraction, tracking, Kalman.
pub mod video {
    pub use cv_video::*;
}

/// `cv::videoio` — AVI, Y4M and image-sequence capture and writing.
pub mod videoio {
    pub use cv_videoio::*;
}

/// `cv::highgui` — windows, keys, trackbars and ROI selection.
///
/// Needs the `gui` feature for a real backend; without it the API still
/// exists and reports that none was compiled in.
pub mod highgui {
    pub use cv_highgui::*;
}

/// `cv::gapi` — a deferred image-processing graph, compiled once per pipeline.
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
