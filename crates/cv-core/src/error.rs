//! Error codes mirroring `cv::Error::Code` (`modules/core/include/opencv2/core/base.hpp`).

use core::fmt;

/// Status codes, numerically identical to `cv::Error::Code`.
///
/// The whole upstream enumeration is present, including the codes this port
/// never raises. A caller that matches on a code from OpenCV — read out of a
/// serialised exception, say, or compared against `cv2.error.code` — needs the
/// full mapping to be meaningful, and the discriminants match upstream so the
/// integers can be compared directly.
///
/// The four GPU/OpenGL/OpenCL families are listed for that reason alone: this
/// port has no such backends, so nothing here will ever return them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
#[non_exhaustive]
pub enum Code {
    /// `cv::Error::StsOk` — everything is fine.
    StsOk = 0,
    /// `cv::Error::StsBackTrace` — pseudo-error marking a stack unwind.
    StsBackTrace = -1,
    /// `cv::Error::StsError` — unspecified error.
    StsError = -2,
    /// `cv::Error::StsInternal` — internal error, a bad state was reached.
    StsInternal = -3,
    /// `cv::Error::StsNoMem` — insufficient memory.
    StsNoMem = -4,
    /// `cv::Error::StsBadArg` — an argument is bad.
    StsBadArg = -5,
    /// `cv::Error::StsBadFunc` — unsupported function.
    StsBadFunc = -6,
    /// `cv::Error::StsNoConv` — an iterative algorithm failed to converge.
    StsNoConv = -7,
    /// `cv::Error::StsAutoTrace` — tracing.
    StsAutoTrace = -8,
    /// `cv::Error::HeaderIsNull` — image header is null.
    HeaderIsNull = -9,
    /// `cv::Error::BadImageSize` — image size is invalid.
    BadImageSize = -10,
    /// `cv::Error::BadOffset` — offset is invalid.
    BadOffset = -11,
    /// `cv::Error::BadDataPtr`
    BadDataPtr = -12,
    /// `cv::Error::BadStep` — image step is wrong, as for a non-continuous matrix.
    BadStep = -13,
    /// `cv::Error::BadModelOrChSeq`
    BadModelOrChSeq = -14,
    /// `cv::Error::BadNumChannels` — for example a single-channel-only function.
    BadNumChannels = -15,
    /// `cv::Error::BadNumChannel1U`
    BadNumChannel1U = -16,
    /// `cv::Error::BadDepth` — input depth is not supported by the function.
    BadDepth = -17,
    /// `cv::Error::BadAlphaChannel`
    BadAlphaChannel = -18,
    /// `cv::Error::BadOrder` — number of dimensions is out of range.
    BadOrder = -19,
    /// `cv::Error::BadOrigin` — incorrect input origin.
    BadOrigin = -20,
    /// `cv::Error::BadAlign` — incorrect input alignment.
    BadAlign = -21,
    /// `cv::Error::BadCallBack`
    BadCallBack = -22,
    /// `cv::Error::BadTileSize`
    BadTileSize = -23,
    /// `cv::Error::BadCOI` — the input channel of interest is not supported.
    BadCOI = -24,
    /// `cv::Error::BadROISize` — incorrect input region of interest.
    BadROISize = -25,
    /// `cv::Error::MaskIsTiled`
    MaskIsTiled = -26,
    /// `cv::Error::StsNullPtr` — null pointer.
    StsNullPtr = -27,
    /// `cv::Error::StsVecLengthErr` — incorrect vector length.
    StsVecLengthErr = -28,
    /// `cv::Error::StsFilterStructContentErr` — incorrect filter structure content.
    StsFilterStructContentErr = -29,
    /// `cv::Error::StsKernelStructContentErr` — incorrect transform kernel content.
    StsKernelStructContentErr = -30,
    /// `cv::Error::StsFilterOffsetErr` — incorrect filter offset value.
    StsFilterOffsetErr = -31,
    /// `cv::Error::StsBadSize` — the input/output structure size is incorrect.
    StsBadSize = -201,
    /// `cv::Error::StsDivByZero` — division by zero.
    StsDivByZero = -202,
    /// `cv::Error::StsInplaceNotSupported` — in-place operation is not supported.
    StsInplaceNotSupported = -203,
    /// `cv::Error::StsObjectNotFound` — the request cannot be completed.
    StsObjectNotFound = -204,
    /// `cv::Error::StsUnmatchedFormats` — formats of input/output arrays differ.
    StsUnmatchedFormats = -205,
    /// `cv::Error::StsBadFlag` — flag is wrong or not supported.
    StsBadFlag = -206,
    /// `cv::Error::StsBadPoint` — bad point.
    StsBadPoint = -207,
    /// `cv::Error::StsBadMask` — bad mask format, neither `8UC1` nor `8SC1`.
    StsBadMask = -208,
    /// `cv::Error::StsUnmatchedSizes` — sizes of input/output structures differ.
    StsUnmatchedSizes = -209,
    /// `cv::Error::StsUnsupportedFormat` — the data format is not supported.
    StsUnsupportedFormat = -210,
    /// `cv::Error::StsOutOfRange` — some parameter is out of range.
    StsOutOfRange = -211,
    /// `cv::Error::StsParseError` — invalid syntax in a parsed file.
    StsParseError = -212,
    /// `cv::Error::StsNotImplemented` — the requested feature is not implemented.
    StsNotImplemented = -213,
    /// `cv::Error::StsBadMemBlock` — an allocated block has been corrupted.
    StsBadMemBlock = -214,
    /// `cv::Error::StsAssert` — assertion failed.
    StsAssert = -215,
    /// `cv::Error::GpuNotSupported` — no CUDA support. Never raised here.
    GpuNotSupported = -216,
    /// `cv::Error::GpuApiCallError` — GPU API call error. Never raised here.
    GpuApiCallError = -217,
    /// `cv::Error::OpenGlNotSupported` — no OpenGL support. Never raised here.
    OpenGlNotSupported = -218,
    /// `cv::Error::OpenGlApiCallError` — OpenGL API call error. Never raised here.
    OpenGlApiCallError = -219,
    /// `cv::Error::OpenCLApiCallError` — OpenCL API call error. Never raised here.
    OpenCLApiCallError = -220,
    /// `cv::Error::OpenCLDoubleNotSupported`. Never raised here.
    OpenCLDoubleNotSupported = -221,
    /// `cv::Error::OpenCLInitError` — OpenCL initialisation error. Never raised here.
    OpenCLInitError = -222,
    /// `cv::Error::OpenCLNoAMDBlasFft`. Never raised here.
    OpenCLNoAMDBlasFft = -223,
}

/// The equivalent of `cv::Exception`.
///
/// Upstream OpenCV throws; this port returns [`Result`] instead, which is the
/// single intentional, pervasive API divergence from the C++ surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// Numeric status code, identical to the value OpenCV would set.
    pub code: Code,
    /// Human-readable description.
    pub msg: String,
}

impl Error {
    /// Construct an error with the given code and message.
    pub fn new(code: Code, msg: impl Into<String>) -> Self {
        Self { code, msg: msg.into() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenCV({:?}): {}", self.code, self.msg)
    }
}

impl std::error::Error for Error {}

/// Result alias used throughout the port.
pub type Result<T> = core::result::Result<T, Error>;

/// Mirrors `CV_Error` — build an [`Err`] with a code and message.
#[macro_export]
macro_rules! cv_error {
    ($code:expr, $($arg:tt)*) => {
        Err($crate::Error::new($code, format!($($arg)*)))
    };
}

/// Mirrors `CV_Assert` — return `StsAssert` when the condition does not hold.
#[macro_export]
macro_rules! cv_assert {
    ($cond:expr) => {
        if !($cond) {
            return Err($crate::Error::new(
                $crate::Code::StsAssert,
                concat!("Assertion failed: ", stringify!($cond)),
            ));
        }
    };
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            return Err($crate::Error::new($crate::Code::StsAssert, format!($($arg)*)));
        }
    };
}

/// Mirrors `CV_Assert_N` — check several conditions, naming the one that failed.
///
/// Upstream needs ten numbered expansion macros (`CV_Assert_1` … `CV_Assert_10`)
/// to reach a variadic assert and caps out at ten conditions. Rust's macro
/// repetition does it directly, with no limit.
#[macro_export]
macro_rules! cv_assert_n {
    ($($cond:expr),+ $(,)?) => {
        $(
            if !($cond) {
                return Err($crate::Error::new(
                    $crate::Code::StsAssert,
                    concat!("Assertion failed: ", stringify!($cond)),
                ));
            }
        )+
    };
}

/// Mirrors `CV_ErrorNoReturn` — upstream's deprecated alias for `CV_Error`.
///
/// It existed when `CV_Error` could return on some compilers; it cannot now, so
/// upstream defines the two identically and so does this.
#[macro_export]
macro_rules! cv_error_no_return {
    ($code:expr, $($arg:tt)*) => {
        $crate::cv_error!($code, $($arg)*)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_codes_match_upstreams_values() {
        // Spot checks across the three ranges: the small negatives, the -20x
        // block, and the backend codes this port never raises.
        assert_eq!(Code::StsOk as i32, 0);
        assert_eq!(Code::StsInternal as i32, -3);
        assert_eq!(Code::StsFilterOffsetErr as i32, -31);
        assert_eq!(Code::StsBadSize as i32, -201);
        assert_eq!(Code::StsAssert as i32, -215);
        assert_eq!(Code::OpenCLNoAMDBlasFft as i32, -223);
    }

    #[test]
    fn assert_n_names_the_condition_that_failed() {
        fn check(a: i32, b: i32, c: i32) -> Result<()> {
            cv_assert_n!(a > 0, b > 0, c > 0);
            Ok(())
        }
        assert!(check(1, 2, 3).is_ok());
        // The first failing condition is the one reported, in order.
        let e = check(1, -1, -1).unwrap_err();
        assert_eq!(e.code, Code::StsAssert);
        assert!(e.msg.contains("b > 0"), "{}", e.msg);
        assert!(!e.msg.contains("c > 0"), "later conditions are not evaluated: {}", e.msg);
    }

    #[test]
    fn error_no_return_is_the_same_as_cv_error() {
        fn f() -> Result<()> {
            cv_error_no_return!(Code::StsBadArg, "bad {}", 7)
        }
        let e = f().unwrap_err();
        assert_eq!(e.code, Code::StsBadArg);
        assert_eq!(e.msg, "bad 7");
    }

    #[test]
    fn display_includes_the_code_and_message() {
        let e = Error::new(Code::StsBadSize, "too small");
        assert_eq!(format!("{e}"), "OpenCV(StsBadSize): too small");
    }
}
