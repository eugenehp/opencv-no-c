//! `cv::instr` — the instrumentation facility
//! (`modules/core/include/opencv2/core/utils/instrumentation.hpp`,
//! `modules/core/include/opencv2/core/private.hpp`).
//!
//! Where `crate::trace` records *what ran and for how long*, instrumentation
//! records *which implementation ran it* — plain C++, IPP, or OpenCL — so a
//! profile can show that a call fell back off its accelerated path. Upstream
//! sprinkles `CV_INSTRUMENT_REGION()` through every function for this.
//!
//! This port has exactly one implementation of everything, so every region is
//! [`Impl::Plain`]. The facility is still worth having: it tells a caller that,
//! which is a real answer to a real question, and the macros are what a
//! translation from C++ needs to compile.
//!
//! Regions here delegate to [`crate::trace::Region`], so instrumentation and
//! tracing share one nesting depth rather than counting separately.

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use crate::trace::Region;

/// `cv::instr::TYPE` — what kind of thing a region covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum Type {
    /// `TYPE_GENERAL` — an exported API function.
    #[default]
    General = 0,
    /// `TYPE_MARKER` — an information marker, not a span.
    Marker = 1,
    /// `TYPE_WRAPPER` — a wrapper around an implementation.
    Wrapper = 2,
    /// `TYPE_FUN` — a plain function call.
    Fun = 3,
}

/// `cv::instr::IMPL` — which code path handled a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum Impl {
    /// `IMPL_PLAIN` — the built-in implementation. Always this one here.
    #[default]
    Plain = 0,
    /// `IMPL_IPP` — Intel IPP. Never reached; this port links no IPP.
    Ipp = 1,
    /// `IMPL_OPENCL` — OpenCL. Never reached; see [`crate::ocl`].
    OpenCL = 2,
}

/// `cv::instr::FLAGS` — how the collected tree is assembled.
pub mod flags {
    /// `FLAGS_NONE`
    pub const FLAGS_NONE: i32 = 0;
    /// `FLAGS_MAPPING` — record the source location of each region.
    pub const FLAGS_MAPPING: i32 = 0x01;
    /// `FLAGS_EXPAND_SAME_NAMES` — keep repeated names as separate nodes
    /// instead of merging them.
    pub const FLAGS_EXPAND_SAME_NAMES: i32 = 0x02;
}

/// `cv::utils::trace::details::CODE_PATH` — which path a traced region ran.
///
/// The same distinction [`Impl`] draws, under the name the trace format uses.
pub mod code_path {
    /// `CODE_PATH_PLAIN` — the built-in implementation. Always this one here.
    pub const CODE_PATH_PLAIN: i32 = 0;
    /// `CODE_PATH_IPP` — Intel IPP. Never reached.
    pub const CODE_PATH_IPP: i32 = 1;
    /// `CODE_PATH_OPENCL` — OpenCL. Never reached.
    pub const CODE_PATH_OPENCL: i32 = 2;
    /// `CODE_PATH_OPENVX` — OpenVX. Never reached.
    pub const CODE_PATH_OPENVX: i32 = 3;
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static FLAGS: AtomicI32 = AtomicI32::new(flags::FLAGS_NONE);

/// `cv::instr::useInstrumentation()`.
#[must_use]
pub fn use_instrumentation() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// `cv::instr::setUseInstrumentation(flag)`.
pub fn set_use_instrumentation(flag: bool) {
    ENABLED.store(flag, Ordering::Relaxed);
    // Regions are recorded through the tracer, so enabling one enables it.
    crate::trace::set_trace_enabled(flag);
}

/// `cv::instr::getFlags()`.
#[must_use]
pub fn get_flags() -> i32 {
    FLAGS.load(Ordering::Relaxed)
}

/// `cv::instr::setFlags(modeFlags)`.
pub fn set_flags(mode_flags: i32) {
    FLAGS.store(mode_flags, Ordering::Relaxed);
}

/// `cv::instr::resetTrace()`.
pub fn reset_trace() {
    FLAGS.store(flags::FLAGS_NONE, Ordering::Relaxed);
}

/// `cv::instr::getTrace()` — the collected region tree.
///
/// `None`: regions are written out as they close rather than accumulated into
/// a tree, so there is no `InstrNode` root to hand back. `crate::trace`'s
/// output carries the same information in the order it happened.
#[must_use]
pub fn get_trace() -> Option<&'static ()> {
    None
}

/// Open an instrumented region. The [`Region`] closes it when dropped.
#[must_use]
pub fn region(name: &'static str, ty: Type, implementation: Impl) -> Region {
    let _ = (ty, implementation);
    Region::new(name, None, 0)
}

/// `CV_INSTRUMENT_REGION()` — instrument the enclosing function.
///
/// Bind the result: `let _r = cv_instrument_region!();`. Dropping it
/// immediately would close the region before the body runs.
#[macro_export]
macro_rules! cv_instrument_region {
    () => {
        $crate::trace::Region::new(
            $crate::cv_func!(),
            Some(concat!(file!(), ":", line!())),
            $crate::trace::region_flag::REGION_FLAG_FUNCTION,
        )
    };
}

/// `CV_INSTRUMENT_REGION_NAME(name)` — instrument a named block.
#[macro_export]
macro_rules! cv_instrument_region_name {
    ($name:expr) => {
        $crate::instr::region($name, $crate::instr::Type::General, $crate::instr::Impl::Plain)
    };
}

/// `CV_INSTRUMENT_REGION_META(name, ...)` — a region with a custom type and impl.
#[macro_export]
macro_rules! cv_instrument_region_meta {
    ($name:expr, $ty:expr, $impl:expr) => {
        $crate::instr::region($name, $ty, $impl)
    };
}

/// `CV_INSTRUMENT_REGION_CUSTOM_META(...)` — upstream's alias for the above.
#[macro_export]
macro_rules! cv_instrument_region_custom_meta {
    ($name:expr, $ty:expr, $impl:expr) => {
        $crate::cv_instrument_region_meta!($name, $ty, $impl)
    };
}

/// `CV_INSTRUMENT_REGION_MT_FORK()` — marks a parallel fork point.
///
/// A no-op region: this port's kernels are single-threaded, so nothing forks.
#[macro_export]
macro_rules! cv_instrument_region_mt_fork {
    () => {
        $crate::instr::region("mt_fork", $crate::instr::Type::Marker, $crate::instr::Impl::Plain)
    };
}

/// `CV_INSTRUMENT_REGION_IPP()` — an IPP-handled region. Never reached here.
#[macro_export]
macro_rules! cv_instrument_region_ipp {
    () => {
        $crate::instr::region("ipp", $crate::instr::Type::Wrapper, $crate::instr::Impl::Ipp)
    };
}

/// `CV_INSTRUMENT_REGION_OPENCL()` — an OpenCL region. Never reached here.
#[macro_export]
macro_rules! cv_instrument_region_opencl {
    () => {
        $crate::instr::region("opencl", $crate::instr::Type::Wrapper, $crate::instr::Impl::OpenCL)
    };
}

/// `CV_INSTRUMENT_REGION_OPENCL_RUN(name)`.
#[macro_export]
macro_rules! cv_instrument_region_opencl_run {
    ($name:expr) => {
        $crate::instr::region($name, $crate::instr::Type::Fun, $crate::instr::Impl::OpenCL)
    };
}

/// `CV_INSTRUMENT_REGION_OPENCL_COMPILE(name)`.
#[macro_export]
macro_rules! cv_instrument_region_opencl_compile {
    ($name:expr) => {
        $crate::instr::region($name, $crate::instr::Type::Marker, $crate::instr::Impl::OpenCL)
    };
}

/// `CV_INSTRUMENT_MARK_META(impl, name)` — record a point, not a span.
#[macro_export]
macro_rules! cv_instrument_mark_meta {
    ($impl:expr, $name:expr) => {{
        let _marker = $crate::instr::region($name, $crate::instr::Type::Marker, $impl);
    }};
}

/// `CV_INSTRUMENT_MARK_IPP(name)`.
#[macro_export]
macro_rules! cv_instrument_mark_ipp {
    ($name:expr) => {
        $crate::cv_instrument_mark_meta!($crate::instr::Impl::Ipp, $name)
    };
}

/// `CV_INSTRUMENT_MARK_OPENCL(name)`.
#[macro_export]
macro_rules! cv_instrument_mark_opencl {
    ($name:expr) => {
        $crate::cv_instrument_mark_meta!($crate::instr::Impl::OpenCL, $name)
    };
}

/// `CV_INSTRUMENT_FUN_IPP(func, ...)` — call an IPP function under instrumentation.
///
/// Expands to the failure code, since there is no IPP function to call.
#[macro_export]
macro_rules! cv_instrument_fun_ipp {
    ($($t:tt)*) => {
        -1
    };
}

/// `CV_IPP_CHECK_COND` — is IPP in use? Always false; see [`crate::utility::use_ipp`].
pub const CV_IPP_CHECK_COND: bool = false;

/// `CV_IPP_CHECK()` — guard a block on IPP being in use.
///
/// The block never runs. Upstream's expands to `if (cv::ipp::useIPP())`, which
/// is the same thing once `useIPP()` is constant-false.
#[macro_export]
macro_rules! cv_ipp_check {
    ($($body:tt)*) => {
        if $crate::instr::CV_IPP_CHECK_COND {
            $($body)*
        }
    };
}

/// `CV_IPP_RUN(condition, func, ...)` — try IPP, fall through when it declines.
///
/// Always falls through: expands to nothing, so the caller's own
/// implementation runs. That is exactly what upstream's non-`HAVE_IPP` build
/// does.
#[macro_export]
macro_rules! cv_ipp_run {
    ($($t:tt)*) => {};
}

/// `CV_IPP_RUN_FAST(func, ...)` — as [`cv_ipp_run`], with the condition implied.
#[macro_export]
macro_rules! cv_ipp_run_fast {
    ($($t:tt)*) => {};
}

/// `CV_OCL_RUN(condition, func, ...)` — try the OpenCL path, fall through.
///
/// Always falls through; see [`crate::cv_ipp_run`] for the reasoning.
#[macro_export]
macro_rules! cv_ocl_run {
    ($($t:tt)*) => {};
}

/// `CV_OVX_RUN(condition, func, ...)` — try the OpenVX path, fall through.
#[macro_export]
macro_rules! cv_ovx_run {
    ($($t:tt)*) => {};
}

/// `CV_IPP_MALLOC(size)` — allocate through IPP.
///
/// Null: there is no IPP allocator. Upstream's callers check for null and fall
/// back to `cv::fastMalloc`, which is the path this forces.
#[macro_export]
macro_rules! cv_ipp_malloc {
    ($size:expr) => {{
        let _ = $size;
        std::ptr::null_mut::<u8>()
    }};
}

/// `CV_REQUIRE_OPENCL_1_2_ERROR` — the error raised when a call needs OpenCL 1.2.
pub const CV_REQUIRE_OPENCL_1_2_ERROR: &str = "OpenCL 1.2 is required, and this build has no OpenCL";

/// `CV_INSTRUMENT_GET_RETURN_ADDRESS` — the caller's return address.
///
/// Upstream reads it with a compiler builtin to attribute a region to its
/// caller. Rust has no stable equivalent, and [`crate::cv_func!`] gives the
/// enclosing function's name directly, so this reports null.
#[macro_export]
macro_rules! cv_instrument_get_return_address {
    () => {
        std::ptr::null::<u8>()
    };
}

/// `CV_INSTRUMENT_FUN_RT_META(...)` — instrument a call and keep its return value.
#[macro_export]
macro_rules! cv_instrument_fun_rt_meta {
    ($ty:expr, $impl:expr, $name:expr, $call:expr) => {{
        let _region = $crate::instr::region($name, $ty, $impl);
        $call
    }};
}

/// `CV_INSTRUMENT_FUN_RV_META(...)` — instrument a call returning void.
#[macro_export]
macro_rules! cv_instrument_fun_rv_meta {
    ($ty:expr, $impl:expr, $name:expr, $call:expr) => {{
        let _region = $crate::instr::region($name, $ty, $impl);
        $call;
    }};
}

/// `CV_IMPL_IPP` — the implementation tag for an IPP path.
pub const CV_IMPL_IPP: Impl = Impl::Ipp;
/// `CV_IMPL_PLAIN` — the implementation tag for the built-in path.
pub const CV_IMPL_PLAIN: Impl = Impl::Plain;
/// `CV_IMPL_OCL` — the implementation tag for an OpenCL path.
pub const CV_IMPL_OCL: Impl = Impl::OpenCL;
/// `CV_IMPL_MT` — marks a multi-threaded region. Never set: kernels here are
/// single-threaded.
pub const CV_IMPL_MT: i32 = 1 << 30;

/// `CV_IMPL_ADD(impl)` — record that a path was taken.
///
/// A no-op: with one implementation there is nothing to distinguish.
#[macro_export]
macro_rules! cv_impl_add {
    ($impl:expr) => {{
        let _ = $impl;
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Instrumentation state is process-global.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn enum_values_match_upstream() {
        assert_eq!(Type::General as i32, 0);
        assert_eq!(Type::Marker as i32, 1);
        assert_eq!(Type::Wrapper as i32, 2);
        assert_eq!(Type::Fun as i32, 3);
        assert_eq!(Impl::Plain as i32, 0);
        assert_eq!(Impl::Ipp as i32, 1);
        assert_eq!(Impl::OpenCL as i32, 2);
        assert_eq!(flags::FLAGS_NONE, 0);
        assert_eq!(flags::FLAGS_MAPPING, 1);
        assert_eq!(flags::FLAGS_EXPAND_SAME_NAMES, 2);
    }

    #[test]
    fn flags_and_the_enable_switch_round_trip() {
        let _g = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prev = use_instrumentation();
        set_flags(flags::FLAGS_MAPPING | flags::FLAGS_EXPAND_SAME_NAMES);
        assert_eq!(get_flags(), 3);
        reset_trace();
        assert_eq!(get_flags(), flags::FLAGS_NONE);

        set_use_instrumentation(true);
        assert!(use_instrumentation());
        // Enabling instrumentation enables the tracer it records through.
        assert!(crate::trace::trace_enabled());
        set_use_instrumentation(false);
        assert!(!use_instrumentation());
        assert!(get_trace().is_none());
        set_use_instrumentation(prev);
    }

    #[test]
    fn the_region_macros_compile_and_nest() {
        let _g = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prev = use_instrumentation();
        set_use_instrumentation(false);
        {
            let _f = cv_instrument_region!();
            let _n = cv_instrument_region_name!("named");
            let _m = cv_instrument_region_meta!("meta", Type::Fun, Impl::Plain);
            let _c = cv_instrument_region_custom_meta!("custom", Type::Wrapper, Impl::Plain);
            let _k = cv_instrument_region_mt_fork!();
            let _i = cv_instrument_region_ipp!();
            let _o = cv_instrument_region_opencl!();
            let _r = cv_instrument_region_opencl_run!("run");
            let _p = cv_instrument_region_opencl_compile!("compile");
            cv_instrument_mark_ipp!("mark");
            cv_instrument_mark_opencl!("mark");
            cv_impl_add!(Impl::Plain);
        }
        set_use_instrumentation(prev);
    }

    #[test]
    fn the_ipp_path_is_never_taken() {
        const _: () = assert!(!CV_IPP_CHECK_COND);
        let mut ran = false;
        cv_ipp_check! {
            ran = true;
        }
        assert!(!ran, "the IPP branch must not run");
        // The run macros expand to nothing, so the caller's own code follows.
        cv_ipp_run!(true, some_ipp_call());
        cv_ipp_run_fast!(some_ipp_call());
        assert_eq!(cv_instrument_fun_ipp!(anything), -1);
        assert_eq!(CV_IMPL_IPP, Impl::Ipp);
        assert_eq!(CV_IMPL_PLAIN, Impl::Plain);
        assert_eq!(CV_IMPL_OCL, Impl::OpenCL);
    }
}
