//! Status and mode constants of `cv::Stitcher` and its stages
//! (`modules/stitching/include/opencv2/stitching*.hpp`).
//!
//! The stitcher's return statuses, its two operating modes, and the wave
//! correction kinds. Generated from the upstream headers.

/// `VORONOI_SEAM`.
pub const VORONOI_SEAM: i32 = 1;
/// `FIRST`.
pub const FIRST: i32 = 1;
/// `SECOND`.
pub const SECOND: i32 = 2;
/// `INTERS`.
pub const INTERS: i32 = 4;
/// `INTERS_FIRST`.
pub const INTERS_FIRST: i32 = 5;
/// `INTERS_SECOND`.
pub const INTERS_SECOND: i32 = 6;
