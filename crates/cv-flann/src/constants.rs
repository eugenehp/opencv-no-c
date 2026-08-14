//! Index, distance and search constants of `cv::flann`
//! (`modules/flann/include/opencv2/flann/defines.h`).
//!
//! The algorithm selectors an index is built with, the centre-picking
//! strategies for the k-means tree, the distance metrics, and the checks
//! settings. Generated from the upstream header.

/// `KDTREE_SINGLE`.
pub const KDTREE_SINGLE: i32 = 4;
/// `FLANN_LOG_NONE`.
pub const FLANN_LOG_NONE: i32 = 0;
/// `FLANN_LOG_FATAL`.
pub const FLANN_LOG_FATAL: i32 = 1;
/// `FLANN_LOG_ERROR`.
pub const FLANN_LOG_ERROR: i32 = 2;
/// `FLANN_LOG_WARN`.
pub const FLANN_LOG_WARN: i32 = 3;
/// `FLANN_LOG_INFO`.
pub const FLANN_LOG_INFO: i32 = 4;
/// `MINKOWSKI`.
pub const MINKOWSKI: i32 = 3;
/// `MAX_DIST`.
pub const MAX_DIST: i32 = 4;
/// `HIST_INTERSECT`.
pub const HIST_INTERSECT: i32 = 5;
/// `HELLINGER`.
pub const HELLINGER: i32 = 6;
/// `KULLBACK_LEIBLER`.
pub const KULLBACK_LEIBLER: i32 = 8;
/// `LAST_VALUE_FLANN_INDEX_TYPE`.
pub const LAST_VALUE_FLANN_INDEX_TYPE: i32 = 2;
