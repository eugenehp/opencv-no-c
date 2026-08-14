//! Constants of `cv::objdetect`
//! (`modules/objdetect/include/opencv2/objdetect*.hpp`).
//!
//! Cascade-classifier flags, HOG descriptor modes, QR code encoding and
//! correction levels, ArUco corner-refinement methods and the predefined
//! dictionary ids. Generated from the upstream headers.
//!
//! The ArUco dictionary ids pair with [`crate::aruco::Dictionary::predefined`],
//! which unpacks the real marker tables, so an id here selects the same
//! dictionary `cv2` would.

/// `CASCADE_SCALE_IMAGE`.
pub const CASCADE_SCALE_IMAGE: i32 = 2;
/// `CASCADE_FIND_BIGGEST_OBJECT`.
pub const CASCADE_FIND_BIGGEST_OBJECT: i32 = 4;
/// `CASCADE_DO_ROUGH_SEARCH`.
pub const CASCADE_DO_ROUGH_SEARCH: i32 = 8;
/// `MODE_AUTO`.
pub const MODE_AUTO: i32 = -1;
/// `MODE_NUMERIC` — 0b0001.
pub const MODE_NUMERIC: i32 = 1;
/// `MODE_ALPHANUMERIC` — 0b0010.
pub const MODE_ALPHANUMERIC: i32 = 2;
/// `MODE_BYTE` — 0b0100.
pub const MODE_BYTE: i32 = 4;
/// `MODE_ECI` — 0b0111.
pub const MODE_ECI: i32 = 7;
/// `MODE_KANJI` — 0b1000.
pub const MODE_KANJI: i32 = 8;
/// `MODE_STRUCTURED_APPEND` — 0b0011.
pub const MODE_STRUCTURED_APPEND: i32 = 3;
/// `CORRECT_LEVEL_L`.
pub const CORRECT_LEVEL_L: i32 = 0;
/// `CORRECT_LEVEL_M`.
pub const CORRECT_LEVEL_M: i32 = 1;
/// `CORRECT_LEVEL_Q`.
pub const CORRECT_LEVEL_Q: i32 = 2;
/// `CORRECT_LEVEL_H`.
pub const CORRECT_LEVEL_H: i32 = 3;
/// `ECI_SHIFT_JIS`.
pub const ECI_SHIFT_JIS: i32 = 20;
/// `ECI_UTF8`.
pub const ECI_UTF8: i32 = 26;
/// `CORNER_REFINE_NONE` — Tag and corners detection based on the ArUco
///   approach.
pub const CORNER_REFINE_NONE: i32 = 0;
/// `CORNER_REFINE_SUBPIX` — ArUco approach and refine the corners locations
///   using corner subpixel accuracy.
pub const CORNER_REFINE_SUBPIX: i32 = 1;
/// `CORNER_REFINE_CONTOUR` — ArUco approach and refine the corners locations
///   using the contour-points line fitting.
pub const CORNER_REFINE_CONTOUR: i32 = 2;
/// `CORNER_REFINE_APRILTAG` — Tag and corners detection based on the AprilTag
///   2 approach wang2016iros.
pub const CORNER_REFINE_APRILTAG: i32 = 3;
/// `DICT_4X4_250` — 4x4 bits, minimum hamming distance between any two codes
///   = 3, 250 codes.
pub const DICT_4X4_250: i32 = 2;
/// `DICT_4X4_1000` — 4x4 bits, minimum hamming distance between any two codes
///   = 2, 1000 codes.
pub const DICT_4X4_1000: i32 = 3;
/// `DICT_5X5_100` — 5x5 bits, minimum hamming distance between any two codes
///   = 7, 100 codes.
pub const DICT_5X5_100: i32 = 5;
/// `DICT_5X5_250` — 5x5 bits, minimum hamming distance between any two codes
///   = 6, 250 codes.
pub const DICT_5X5_250: i32 = 6;
/// `DICT_5X5_1000` — 5x5 bits, minimum hamming distance between any two codes
///   = 5, 1000 codes.
pub const DICT_5X5_1000: i32 = 7;
/// `DICT_6X6_100` — 6x6 bits, minimum hamming distance between any two codes
///   = 12, 100 codes.
pub const DICT_6X6_100: i32 = 9;
/// `DICT_6X6_250` — 6x6 bits, minimum hamming distance between any two codes
///   = 11, 250 codes.
pub const DICT_6X6_250: i32 = 10;
/// `DICT_6X6_1000` — 6x6 bits, minimum hamming distance between any two codes
///   = 9, 1000 codes.
pub const DICT_6X6_1000: i32 = 11;
/// `DICT_7X7_100` — 7x7 bits, minimum hamming distance between any two codes
///   = 18, 100 codes.
pub const DICT_7X7_100: i32 = 13;
/// `DICT_7X7_250` — 7x7 bits, minimum hamming distance between any two codes
///   = 17, 250 codes.
pub const DICT_7X7_250: i32 = 14;
/// `DICT_7X7_1000` — 7x7 bits, minimum hamming distance between any two codes
///   = 14, 1000 codes.
pub const DICT_7X7_1000: i32 = 15;
/// `DICT_ARUCO_ORIGINAL` — 6x6 bits, minimum hamming distance between any two
///   codes = 3, 1024 codes.
pub const DICT_ARUCO_ORIGINAL: i32 = 16;
/// `DETECTED_NOT_SHOWN_YET`.
pub const DETECTED_NOT_SHOWN_YET: i32 = 0;
/// `DETECTED`.
pub const DETECTED: i32 = 1;
/// `DETECTED_TEMPORARY_LOST`.
pub const DETECTED_TEMPORARY_LOST: i32 = 2;
/// `WRONG_OBJECT`.
pub const WRONG_OBJECT: i32 = 3;
