//! A C-callable surface, so a C or C++ caller can link against this crate.
//!
//! Enabled by the `abi` feature, which implies `dropin`.
//!
//! # This links no C
//!
//! `extern "C"` is a **calling convention**, not a dependency: these symbols
//! are emitted by rustc, and nothing here compiles or links a line of C. The
//! project's "no C" premise is about not depending on a C toolchain or a C
//! library, and exporting a stable ABI does neither.
//!
//! # What it is not
//!
//! It is **not** OpenCV's C++ ABI. That cannot be reproduced from Rust at all:
//! it depends on Itanium name mangling, vtable layout, and the memory layout
//! of `std::vector`, `std::string` and `cv::Mat` as a particular libstdc++
//! emitted them. Code calling `cv::cvtColor` will not link against this. What
//! is achievable — and what this is — is a stable C ABI in the shape OpenCV's
//! own C API had: opaque handles, integer status codes, and an out-parameter
//! for the result.
//!
//! Symbols are prefixed `cvno_` rather than `cv`, deliberately: a program that
//! links both this and a real `libopencv_core` must not get duplicate
//! definitions, and a silently-resolved wrong one would be worse than a link
//! error.
//!
//! # Contract
//!
//! * Every fallible entry point returns `i32`: `0` on success, otherwise
//!   upstream's `cv::Error::Code` unchanged. Those are negative, so a caller
//!   can test `rc < 0` or compare against the same constants OpenCV uses —
//!   `StsBadArg` is `-5` here exactly as it is there.
//! * On failure, `cvno_last_error` returns the message. It is thread-local and
//!   valid until the next failing call **on the same thread**.
//! * Handles from `cvno_mat_*` and buffers from `cvno_imencode` are owned by
//!   the caller and must be released with the matching `_release`.
//! * A panic is caught at the boundary and turned into a status code. Unwinding
//!   into C is undefined behaviour, so nothing is allowed to escape.
//!
//! ```c
//! CvnoMat* img = NULL;
//! if (cvno_imdecode(bytes, len, 1 /* IMREAD_COLOR */, &img) != 0) {
//!     fprintf(stderr, "%s\n", cvno_last_error());
//!     return 1;
//! }
//! CvnoMat* gray = NULL;
//! cvno_cvt_color(img, 6 /* COLOR_BGR2GRAY */, &gray);
//! printf("%dx%d\n", cvno_mat_cols(gray), cvno_mat_rows(gray));
//! cvno_mat_release(gray);
//! cvno_mat_release(img);
//! ```

use core::ffi::{c_char, c_int, c_uchar};
use std::cell::RefCell;
use std::ffi::CString;
use std::panic::{catch_unwind, AssertUnwindSafe};

use cv_core::{Code, Error, Mat, MatLike};

/// An owned matrix, opaque to the caller.
#[derive(Debug)]
#[repr(C)]
pub struct CvnoMat {
    inner: Mat,
}

/// An owned byte buffer, opaque to the caller.
#[derive(Debug)]
#[repr(C)]
pub struct CvnoBuffer {
    inner: Vec<u8>,
}

thread_local! {
    /// The last failure on this thread, kept alive for `cvno_last_error`.
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").expect("empty is valid"));
}

/// Record a message and hand back the status code for it.
fn fail(e: &Error) -> c_int {
    let msg = CString::new(e.msg.replace('\0', " ")).unwrap_or_else(|_| {
        CString::new("error message contained an interior NUL").expect("literal is valid")
    });
    LAST_ERROR.with(|c| *c.borrow_mut() = msg);
    // Upstream's codes are already negative; passing one through unchanged
    // lets a caller compare against `cv::Error::StsBadArg` and friends.
    e.code as i32
}

/// Run `f` at the ABI boundary: catch everything, report a status code.
///
/// Nothing unwinds past this point, because unwinding into C is undefined.
fn guard<F: FnOnce() -> cv_core::Result<()>>(f: F) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => fail(&e),
        Err(payload) => {
            let e = match payload.downcast::<Error>() {
                Ok(e) => *e,
                Err(_) => Error::new(Code::StsInternal, "a panic crossed the ABI boundary"),
            };
            fail(&e)
        }
    }
}

/// Borrow a handle, or report a null argument.
///
/// # Safety
///
/// `p` must be null or a live handle from this module.
unsafe fn as_mat<'a>(p: *const CvnoMat) -> cv_core::Result<&'a Mat> {
    if p.is_null() {
        return Err(Error::new(Code::StsNullPtr, "matrix argument was null"));
    }
    // SAFETY: checked non-null; the caller guarantees it is a live handle.
    Ok(unsafe { &(*p).inner })
}

/// Hand an owned matrix back through an out-parameter.
///
/// # Safety
///
/// `out` must be null or point at a writable pointer slot.
unsafe fn give(out: *mut *mut CvnoMat, m: Mat) -> cv_core::Result<()> {
    if out.is_null() {
        return Err(Error::new(Code::StsNullPtr, "output argument was null"));
    }
    // SAFETY: checked non-null; the caller guarantees it is writable.
    unsafe { *out = Box::into_raw(Box::new(CvnoMat { inner: m })) };
    Ok(())
}

/// The message from the last failing call on this thread.
///
/// Valid until the next failing call on this thread. Never null.
#[no_mangle]
pub extern "C" fn cvno_last_error() -> *const c_char {
    LAST_ERROR.with(|c| c.borrow().as_ptr())
}

/// This crate's version, as a NUL-terminated string.
#[no_mangle]
pub extern "C" fn cvno_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// `cv::Mat::zeros(rows, cols, type)`. Returns null on failure.
#[no_mangle]
pub extern "C" fn cvno_mat_zeros(rows: c_int, cols: c_int, ty: c_int) -> *mut CvnoMat {
    let made = catch_unwind(|| {
        let ty = cv_core::types::MatType::from_i32(ty)?;
        Mat::zeros(rows, cols, ty)
    });
    match made {
        Ok(Ok(m)) => Box::into_raw(Box::new(CvnoMat { inner: m })),
        Ok(Err(e)) => {
            fail(&e);
            core::ptr::null_mut()
        }
        Err(_) => {
            fail(&Error::new(Code::StsInternal, "a panic crossed the ABI boundary"));
            core::ptr::null_mut()
        }
    }
}

/// Release a handle. Null is accepted and ignored, as `free` is.
///
/// # Safety
///
/// `m` must be null or a handle from this module, not yet released.
#[no_mangle]
pub unsafe extern "C" fn cvno_mat_release(m: *mut CvnoMat) {
    if !m.is_null() {
        // SAFETY: checked non-null; the caller guarantees single ownership.
        drop(unsafe { Box::from_raw(m) });
    }
}

/// Row count, or `-1` for a null handle.
///
/// # Safety
///
/// `m` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn cvno_mat_rows(m: *const CvnoMat) -> c_int {
    // SAFETY: delegated to the caller's contract.
    unsafe { as_mat(m) }.map_or(-1, MatLike::rows)
}

/// Column count, or `-1` for a null handle.
///
/// # Safety
///
/// `m` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn cvno_mat_cols(m: *const CvnoMat) -> c_int {
    // SAFETY: delegated to the caller's contract.
    unsafe { as_mat(m) }.map_or(-1, MatLike::cols)
}

/// `Mat::type()` as upstream's packed integer, or `-1` for a null handle.
///
/// # Safety
///
/// `m` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn cvno_mat_type(m: *const CvnoMat) -> c_int {
    // SAFETY: delegated to the caller's contract.
    unsafe { as_mat(m) }.map_or(-1, |m| m.mat_type().to_i32())
}

/// Row stride in bytes, or `0` for a null handle.
///
/// # Safety
///
/// `m` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn cvno_mat_step(m: *const CvnoMat) -> usize {
    // SAFETY: delegated to the caller's contract.
    unsafe { as_mat(m) }.map_or(0, MatLike::step)
}

/// A read-only pointer to the pixels, or null.
///
/// The pointer is valid until the handle is released. The buffer is
/// `rows * step` bytes.
///
/// # Safety
///
/// `m` must be null or a live handle.
#[no_mangle]
pub unsafe extern "C" fn cvno_mat_data(m: *const CvnoMat) -> *const c_uchar {
    // SAFETY: delegated to the caller's contract.
    match unsafe { as_mat(m) } {
        Ok(m) => m.as_bytes().as_ptr(),
        Err(_) => core::ptr::null(),
    }
}

/// `cv::imdecode(buf, flags)`.
///
/// # Safety
///
/// `data` must point at `len` readable bytes; `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn cvno_imdecode(
    data: *const c_uchar,
    len: usize,
    flags: c_int,
    out: *mut *mut CvnoMat,
) -> c_int {
    guard(|| {
        if data.is_null() {
            return Err(Error::new(Code::StsNullPtr, "input buffer was null"));
        }
        // SAFETY: the caller guarantees `len` readable bytes at `data`.
        let bytes = unsafe { core::slice::from_raw_parts(data, len) };
        let mode = imread_mode(flags)?;
        let m = cv_imgcodecs::imdecode(bytes, mode)?;
        // SAFETY: `out` is checked inside `give`.
        unsafe { give(out, m) }
    })
}

/// `cv::imencode(ext, img, buf)`. `ext` is a NUL-terminated extension such as
/// `".png"`. Release the buffer with `cvno_buffer_release`.
///
/// # Safety
///
/// `ext` must be a NUL-terminated string; `src` a live handle; `out` writable.
#[no_mangle]
pub unsafe extern "C" fn cvno_imencode(
    ext: *const c_char,
    src: *const CvnoMat,
    out: *mut *mut CvnoBuffer,
) -> c_int {
    guard(|| {
        if ext.is_null() || out.is_null() {
            return Err(Error::new(Code::StsNullPtr, "extension or output was null"));
        }
        // SAFETY: the caller guarantees a NUL-terminated string.
        let ext = unsafe { core::ffi::CStr::from_ptr(ext) }
            .to_str()
            .map_err(|_| Error::new(Code::StsBadArg, "extension was not UTF-8"))?;
        // SAFETY: delegated to the caller's contract.
        let m = unsafe { as_mat(src) }?;
        let bytes = cv_imgcodecs::imencode(ext, m)?;
        // SAFETY: `out` checked non-null above.
        unsafe { *out = Box::into_raw(Box::new(CvnoBuffer { inner: bytes })) };
        Ok(())
    })
}

/// Bytes of an encoded buffer, or null.
///
/// # Safety
///
/// `b` must be null or a live buffer handle.
#[no_mangle]
pub unsafe extern "C" fn cvno_buffer_data(b: *const CvnoBuffer) -> *const c_uchar {
    if b.is_null() {
        return core::ptr::null();
    }
    // SAFETY: checked non-null; the caller guarantees a live handle.
    unsafe { (*b).inner.as_ptr() }
}

/// Length of an encoded buffer, or `0`.
///
/// # Safety
///
/// `b` must be null or a live buffer handle.
#[no_mangle]
pub unsafe extern "C" fn cvno_buffer_len(b: *const CvnoBuffer) -> usize {
    if b.is_null() {
        return 0;
    }
    // SAFETY: checked non-null; the caller guarantees a live handle.
    unsafe { (*b).inner.len() }
}

/// Release an encoded buffer. Null is accepted.
///
/// # Safety
///
/// `b` must be null or a buffer handle, not yet released.
#[no_mangle]
pub unsafe extern "C" fn cvno_buffer_release(b: *mut CvnoBuffer) {
    if !b.is_null() {
        // SAFETY: checked non-null; the caller guarantees single ownership.
        drop(unsafe { Box::from_raw(b) });
    }
}

/// `cv::cvtColor(src, dst, code)`, with upstream's numeric `code`.
///
/// # Safety
///
/// `src` must be a live handle; `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn cvno_cvt_color(
    src: *const CvnoMat,
    code: c_int,
    out: *mut *mut CvnoMat,
) -> c_int {
    guard(|| {
        // SAFETY: delegated to the caller's contract.
        let m = unsafe { as_mat(src) }?;
        let code = cv_imgproc::color::ColorConversion::from_i32(code)
            .ok_or_else(|| Error::new(Code::StsBadArg, format!("unknown conversion code {code}")))?;
        let dst = cv_imgproc::color::cvt_color(m, code)?;
        // SAFETY: `out` is checked inside `give`.
        unsafe { give(out, dst) }
    })
}

/// `cv::resize(src, dst, dsize, fx, fy, interpolation)`.
///
/// # Safety
///
/// `src` must be a live handle; `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn cvno_resize(
    src: *const CvnoMat,
    width: c_int,
    height: c_int,
    fx: f64,
    fy: f64,
    interpolation: c_int,
    out: *mut *mut CvnoMat,
) -> c_int {
    guard(|| {
        // SAFETY: delegated to the caller's contract.
        let m = unsafe { as_mat(src) }?;
        let interp = interpolation_from_int(interpolation)?;
        let dst = cv_imgproc::resize::resize(
            m,
            cv_core::Size::new(width, height),
            fx,
            fy,
            interp,
        )?;
        // SAFETY: `out` is checked inside `give`.
        unsafe { give(out, dst) }
    })
}

/// `cv::GaussianBlur(src, dst, ksize, sigmaX, sigmaY)` with `BORDER_REFLECT_101`.
///
/// # Safety
///
/// `src` must be a live handle; `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn cvno_gaussian_blur(
    src: *const CvnoMat,
    kwidth: c_int,
    kheight: c_int,
    sigma_x: f64,
    sigma_y: f64,
    out: *mut *mut CvnoMat,
) -> c_int {
    guard(|| {
        // SAFETY: delegated to the caller's contract.
        let m = unsafe { as_mat(src) }?;
        let dst = cv_imgproc::filter::gaussian_blur(
            m,
            cv_core::Size::new(kwidth, kheight),
            sigma_x,
            sigma_y,
            cv_core::BorderType::Reflect101,
        )?;
        // SAFETY: `out` is checked inside `give`.
        unsafe { give(out, dst) }
    })
}

/// `cv::Canny(image, edges, threshold1, threshold2, apertureSize, L2gradient)`.
///
/// # Safety
///
/// `src` must be a live handle; `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn cvno_canny(
    src: *const CvnoMat,
    threshold1: f64,
    threshold2: f64,
    aperture_size: c_int,
    l2_gradient: c_int,
    out: *mut *mut CvnoMat,
) -> c_int {
    guard(|| {
        // SAFETY: delegated to the caller's contract.
        let m = unsafe { as_mat(src) }?;
        let dst = cv_imgproc::edge::canny(
            m,
            threshold1,
            threshold2,
            aperture_size,
            l2_gradient != 0,
        )?;
        // SAFETY: `out` is checked inside `give`.
        unsafe { give(out, dst) }
    })
}

/// `cv::threshold(src, dst, thresh, maxval, type)`. `applied` receives the
/// threshold actually used, which differs from `thresh` under Otsu.
///
/// # Safety
///
/// `src` must be a live handle; `out` and `applied` must be writable or null.
#[no_mangle]
pub unsafe extern "C" fn cvno_threshold(
    src: *const CvnoMat,
    thresh: f64,
    maxval: f64,
    ttype: c_int,
    out: *mut *mut CvnoMat,
    applied: *mut f64,
) -> c_int {
    guard(|| {
        // SAFETY: delegated to the caller's contract.
        let m = unsafe { as_mat(src) }?;
        // Upstream packs THRESH_OTSU (8) and THRESH_TRIANGLE (16) into the
        // same integer as the type, so they are masked out here the same way.
        let method = match ttype & !7 {
            0 => cv_imgproc::threshold::ThresholdMethod::Manual,
            8 => cv_imgproc::threshold::ThresholdMethod::Otsu,
            16 => cv_imgproc::threshold::ThresholdMethod::Triangle,
            other => {
                return Err(Error::new(
                    Code::StsBadArg,
                    format!("unknown threshold method bits {other}"),
                ))
            }
        };
        let ty = threshold_type_from_int(ttype & 7)?;
        let t = cv_imgproc::threshold::threshold(m, thresh, maxval, ty, method)?;
        if !applied.is_null() {
            // SAFETY: checked non-null; the caller guarantees it is writable.
            unsafe { *applied = t.thresh };
        }
        // SAFETY: `out` is checked inside `give`.
        unsafe { give(out, t.dst) }
    })
}

/// `IMREAD_*` as upstream numbers them.
fn imread_mode(flags: c_int) -> cv_core::Result<cv_imgcodecs::ImreadMode> {
    Ok(match flags {
        -1 => cv_imgcodecs::ImreadMode::Unchanged,
        0 => cv_imgcodecs::ImreadMode::Grayscale,
        1 => cv_imgcodecs::ImreadMode::Color,
        other => {
            return Err(Error::new(Code::StsBadArg, format!("unknown imread flag {other}")))
        }
    })
}

/// `INTER_*` as upstream numbers them.
fn interpolation_from_int(v: c_int) -> cv_core::Result<cv_imgproc::resize::Interpolation> {
    use cv_imgproc::resize::Interpolation as I;
    Ok(match v {
        0 => I::Nearest,
        1 => I::Linear,
        3 => I::Area,
        5 => I::LinearExact,
        other => {
            return Err(Error::new(Code::StsBadArg, format!("unknown interpolation {other}")))
        }
    })
}

/// `THRESH_*` as upstream numbers them, with the method bits already masked off.
fn threshold_type_from_int(v: c_int) -> cv_core::Result<cv_imgproc::threshold::ThresholdType> {
    use cv_imgproc::threshold::ThresholdType as T;
    Ok(match v {
        0 => T::Binary,
        1 => T::BinaryInv,
        2 => T::Trunc,
        3 => T::ToZero,
        4 => T::ToZeroInv,
        other => {
            return Err(Error::new(Code::StsBadArg, format!("unknown threshold type {other}")))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ABI is exercised the way C would, through raw pointers.
    #[test]
    fn a_c_style_pipeline_runs_end_to_end() {
        // CV_8UC3 is 16 in upstream's packed encoding.
        let src = cvno_mat_zeros(32, 48, 16);
        assert!(!src.is_null());
        // SAFETY: `src` is a live handle from this module.
        unsafe {
            assert_eq!(cvno_mat_rows(src), 32);
            assert_eq!(cvno_mat_cols(src), 48);
            assert_eq!(cvno_mat_type(src), 16);
            assert!(!cvno_mat_data(src).is_null());

            let mut gray: *mut CvnoMat = core::ptr::null_mut();
            assert_eq!(cvno_cvt_color(src, 6, &mut gray), 0, "COLOR_BGR2GRAY");
            assert_eq!(cvno_mat_type(gray), 0, "CV_8UC1");

            let mut small: *mut CvnoMat = core::ptr::null_mut();
            assert_eq!(cvno_resize(gray, 24, 16, 0.0, 0.0, 3, &mut small), 0);
            assert_eq!((cvno_mat_rows(small), cvno_mat_cols(small)), (16, 24));

            let mut edges: *mut CvnoMat = core::ptr::null_mut();
            assert_eq!(cvno_canny(small, 50.0, 150.0, 3, 0, &mut edges), 0);

            cvno_mat_release(edges);
            cvno_mat_release(small);
            cvno_mat_release(gray);
            cvno_mat_release(src);
            cvno_mat_release(core::ptr::null_mut());
        }
    }

    #[test]
    fn a_failure_sets_a_status_and_a_message() {
        let src = cvno_mat_zeros(8, 8, 0);
        // SAFETY: `src` is a live handle.
        unsafe {
            let mut out: *mut CvnoMat = core::ptr::null_mut();
            let rc = cvno_cvt_color(src, 9999, &mut out);
            assert_eq!(rc, Code::StsBadArg as i32, "the status is upstream's own code");
            assert!(out.is_null(), "nothing is handed back on failure");
            let msg = core::ffi::CStr::from_ptr(cvno_last_error()).to_str().unwrap();
            assert!(msg.contains("9999"), "the message names the bad code: {msg}");
            cvno_mat_release(src);
        }
    }

    #[test]
    fn null_arguments_are_reported_not_dereferenced() {
        // SAFETY: passing null is exactly what is under test.
        unsafe {
            let mut out: *mut CvnoMat = core::ptr::null_mut();
            assert!(cvno_cvt_color(core::ptr::null(), 6, &mut out) != 0);
            assert_eq!(cvno_mat_rows(core::ptr::null()), -1);
            assert_eq!(cvno_mat_cols(core::ptr::null()), -1);
            assert_eq!(cvno_mat_step(core::ptr::null()), 0);
            assert!(cvno_mat_data(core::ptr::null()).is_null());
            assert_eq!(cvno_buffer_len(core::ptr::null()), 0);
            assert!(cvno_buffer_data(core::ptr::null()).is_null());
        }
    }

    #[test]
    fn encoding_round_trips_through_the_buffer_handles() {
        let src = cvno_mat_zeros(12, 10, 0);
        // SAFETY: live handles throughout.
        unsafe {
            let ext = CString::new(".png").unwrap();
            let mut buf: *mut CvnoBuffer = core::ptr::null_mut();
            assert_eq!(cvno_imencode(ext.as_ptr(), src, &mut buf), 0);
            let len = cvno_buffer_len(buf);
            assert!(len > 8, "an encoded PNG is not empty");

            let mut back: *mut CvnoMat = core::ptr::null_mut();
            assert_eq!(cvno_imdecode(cvno_buffer_data(buf), len, 0, &mut back), 0);
            assert_eq!((cvno_mat_rows(back), cvno_mat_cols(back)), (12, 10));

            cvno_mat_release(back);
            cvno_buffer_release(buf);
            cvno_buffer_release(core::ptr::null_mut());
            cvno_mat_release(src);
        }
    }

    #[test]
    fn the_version_string_is_nul_terminated() {
        // SAFETY: `cvno_version` returns a static NUL-terminated literal.
        let v = unsafe { core::ffi::CStr::from_ptr(cvno_version()) }.to_str().unwrap();
        assert_eq!(v, env!("CARGO_PKG_VERSION"));
    }
}
