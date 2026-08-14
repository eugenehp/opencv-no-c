//! Constants of `cv::gapi`
//! (`modules/gapi/include/opencv2/gapi/**`).
//!
//! Border and interpolation modes the graph kernels take, and the queue and
//! backend selectors of the streaming executor. Generated from the upstream
//! headers.

/// `CV_UNKNOWN` — Unknown, generic, opaque-to-GAPI data type unsupported in
///   graph seriallization.
pub const CV_UNKNOWN: i32 = 0;
/// `CV_BOOL` — bool user G-API data.
pub const CV_BOOL: i32 = 1;
/// `CV_INT` — int user G-API data.
pub const CV_INT: i32 = 2;
/// `CV_INT64` — int64_t user G-API data.
pub const CV_INT64: i32 = 3;
/// `CV_DOUBLE` — double user G-API data.
pub const CV_DOUBLE: i32 = 4;
/// `CV_FLOAT` — float user G-API data.
pub const CV_FLOAT: i32 = 5;
/// `CV_UINT64` — uint64_t user G-API data.
pub const CV_UINT64: i32 = 6;
/// `CV_STRING` — std::string user G-API data.
pub const CV_STRING: i32 = 7;
/// `CV_POINT` — cv::Point user G-API data.
pub const CV_POINT: i32 = 8;
/// `CV_POINT2F` — cv::Point2f user G-API data.
pub const CV_POINT2F: i32 = 9;
/// `CV_POINT3F` — cv::Point3f user G-API data.
pub const CV_POINT3F: i32 = 10;
/// `CV_SIZE` — cv::Size user G-API data.
pub const CV_SIZE: i32 = 11;
/// `CV_RECT` — cv::Rect user G-API data.
pub const CV_RECT: i32 = 12;
/// `CV_SCALAR` — cv::Scalar user G-API data.
pub const CV_SCALAR: i32 = 13;
/// `CV_MAT` — cv::Mat user G-API data.
pub const CV_MAT: i32 = 14;
/// `CV_DRAW_PRIM` — cv::gapi::wip::draw::Prim user G-API data.
pub const CV_DRAW_PRIM: i32 = 15;
