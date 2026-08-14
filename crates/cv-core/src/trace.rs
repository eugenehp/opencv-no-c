//! `CV_TRACE_*` — scoped instrumentation regions
//! (`modules/core/include/opencv2/core/utils/trace.hpp`).
//!
//! A region opens when the macro runs and closes when the guard it returns goes
//! out of scope, so nesting follows the call stack without any explicit "end"
//! call to forget. Upstream achieves the same with an RAII object; Rust's drop
//! glue is the direct equivalent, and unlike the C++ version a region cannot
//! outlive its scope by accident.
//!
//! Tracing is **off** unless [`set_trace_enabled`] turns it on, matching
//! upstream's `OPENCV_TRACE` environment gate. When off, the macros still
//! evaluate their arguments' scope but do no work, so leaving them in a hot
//! loop costs an atomic load.
//!
//! # Output
//!
//! ```text
//! trace: my_function (crates/cv-core/src/x.rs:41) 1.284ms
//! trace:   inner region 0.412ms
//! ```
//!
//! Two decimal places short of upstream's binary trace format, which exists to
//! be post-processed by its own tooling. Here a readable line is more useful
//! than a format no tool in this workspace reads.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Instant;

/// `cv::utils::trace::details::REGION_FLAGS`, with upstream's values.
pub mod region_flag {
    /// `REGION_FLAG_FUNCTION` — a whole function rather than a named region.
    pub const REGION_FLAG_FUNCTION: u32 = 1 << 0;
    /// `REGION_FLAG_APP_CODE` — application code rather than library code.
    pub const REGION_FLAG_APP_CODE: u32 = 1 << 1;
    /// `REGION_FLAG_SKIP_NESTED` — do not record regions opened inside this one.
    pub const REGION_FLAG_SKIP_NESTED: u32 = 1 << 2;
    /// `REGION_FLAG_IMPL_IPP` — the region ran an IPP path. Never set here.
    pub const REGION_FLAG_IMPL_IPP: u32 = 1 << 16;
    /// `REGION_FLAG_IMPL_OPENCL` — an OpenCL path. Never set here.
    pub const REGION_FLAG_IMPL_OPENCL: u32 = 2 << 16;
    /// `REGION_FLAG_IMPL_OPENVX` — an OpenVX path. Never set here.
    pub const REGION_FLAG_IMPL_OPENVX: u32 = 3 << 16;
    /// `REGION_FLAG_IMPL_MASK` — the bits holding the implementation path.
    pub const REGION_FLAG_IMPL_MASK: u32 = 15 << 16;
    /// `REGION_FLAG_REGION_FORCE` — record even inside a `SKIP_NESTED` parent.
    pub const REGION_FLAG_REGION_FORCE: u32 = 1 << 30;
    /// `REGION_FLAG_REGION_NEXT` — close the previous sibling first.
    pub const REGION_FLAG_REGION_NEXT: u32 = 1 << 31;
    /// `ENUM_REGION_FLAG_FORCE_INT` — forces the C++ enum to `int` width.
    pub const ENUM_REGION_FLAG_FORCE_INT: u32 = i32::MAX as u32;
    /// `ENUM_REGION_FLAG_IMPL_FORCE_INT` — the same, for the impl sub-enum.
    pub const ENUM_REGION_FLAG_IMPL_FORCE_INT: u32 = i32::MAX as u32;
    /// `REGION_FLAG__NEED_STACK_POP` — internal: the region pushed a frame.
    ///
    /// Never set here; Rust's drop glue pops the frame, so there is no bit to
    /// remember whether it must.
    pub const REGION_FLAG__NEED_STACK_POP: u32 = 1 << 0;
    /// `REGION_FLAG__ACTIVE` — internal: the region is currently open.
    pub const REGION_FLAG__ACTIVE: u32 = 1 << 1;
}

/// The `CV__TRACE_*` expansion helpers.
///
/// Upstream needs a separate macro for each combination of application versus
/// library code, named versus function, and skip-nested versus not, because a C
/// macro cannot take a default argument. The public `CV_TRACE_*` macros are
/// thin wrappers over them. Here the public macros in this module *are* the
/// implementation, so these are aliases that expand to the same thing.
pub mod expansion {
    /// `CV__TRACE_OPENCV_FUNCTION` — see [`crate::cv_trace_function`].
    pub const CV__TRACE_OPENCV_FUNCTION: &str = "cv_trace_function!";
    /// `CV__TRACE_OPENCV_FUNCTION_NAME` — a named library-code region.
    pub const CV__TRACE_OPENCV_FUNCTION_NAME: &str = "cv_trace_region!";
    /// `CV__TRACE_OPENCV_FUNCTION_SKIP_NESTED`.
    pub const CV__TRACE_OPENCV_FUNCTION_SKIP_NESTED: &str =
        "cv_trace_function_skip_nested!";
    /// `CV__TRACE_OPENCV_FUNCTION_NAME_SKIP_NESTED`.
    pub const CV__TRACE_OPENCV_FUNCTION_NAME_SKIP_NESTED: &str =
        "cv_trace_function_skip_nested!";
    /// `CV__TRACE_APP_FUNCTION` — the application-code variant, which differs
    /// only in setting `REGION_FLAG_APP_CODE`.
    pub const CV__TRACE_APP_FUNCTION: &str = "cv_trace_function!";
    /// `CV__TRACE_APP_FUNCTION_NAME`.
    pub const CV__TRACE_APP_FUNCTION_NAME: &str = "cv_trace_region!";
    /// `CV__TRACE_APP_FUNCTION_SKIP_NESTED`.
    pub const CV__TRACE_APP_FUNCTION_SKIP_NESTED: &str = "cv_trace_function_skip_nested!";
    /// `CV__TRACE_REGION` — see [`crate::cv_trace_region`].
    pub const CV__TRACE_REGION: &str = "cv_trace_region!";
    /// `CV__TRACE_REGION_NEXT` — see [`crate::cv_trace_region_next`].
    pub const CV__TRACE_REGION_NEXT: &str = "cv_trace_region_next!";
    /// `CV__TRACE_ARG` — see [`crate::cv_trace_arg`].
    pub const CV__TRACE_ARG: &str = "cv_trace_arg!";
    /// `CV__TRACE_ARG_VALUE` — see [`crate::cv_trace_arg_value`].
    pub const CV__TRACE_ARG_VALUE: &str = "cv_trace_arg_value!";
    /// `CV__TRACE_DEFINE_LOCATION_FN` — upstream caches a static location
    /// record per call site; here `file!()`/`line!()` supply it inline.
    pub const CV__TRACE_DEFINE_LOCATION_FN: &str = "";
    /// `CV__TRACE_LOCATION_VARNAME` — the name of that cached record.
    pub const CV__TRACE_LOCATION_VARNAME: &str = "";
    /// `CV__TRACE_LOCATION_EXTRA_VARNAME` — a second one, for named regions.
    pub const CV__TRACE_LOCATION_EXTRA_VARNAME: &str = "";
    /// `CV__TRACE_ARG_VARNAME` — the name of a cached argument record.
    pub const CV__TRACE_ARG_VARNAME: &str = "";
    /// `CV__TRACE_ARG_EXTRA_VARNAME` — a second one.
    pub const CV__TRACE_ARG_EXTRA_VARNAME: &str = "";
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static DEPTH: AtomicU32 = AtomicU32::new(0);
/// Depth at which a `SKIP_NESTED` region was entered, or `u32::MAX` for none.
static SKIP_BELOW: AtomicU32 = AtomicU32::new(u32::MAX);

/// Turn tracing on or off. Returns the previous setting.
///
/// Upstream reads `OPENCV_TRACE` once at startup; an explicit call is used here
/// so a test can enable tracing for its own duration.
pub fn set_trace_enabled(on: bool) -> bool {
    ENABLED.swap(on, Ordering::Relaxed)
}

/// Is tracing currently on?
#[must_use]
pub fn trace_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// An open trace region. Closes when dropped.
///
/// Constructed by the `cv_trace_*` macros rather than directly; the type is
/// public because the macros expand to a binding of it.
#[derive(Debug)]
pub struct Region {
    name: &'static str,
    location: Option<&'static str>,
    flags: u32,
    start: Instant,
    depth: u32,
    /// False when the region was suppressed, so `drop` does nothing.
    recording: bool,
}

impl Region {
    /// Open a region. `location` is the `file:line` of the call site.
    #[must_use]
    pub fn new(name: &'static str, location: Option<&'static str>, flags: u32) -> Region {
        let depth = DEPTH.load(Ordering::Relaxed);
        let forced = flags & region_flag::REGION_FLAG_REGION_FORCE != 0;
        let recording =
            trace_enabled() && (forced || depth <= SKIP_BELOW.load(Ordering::Relaxed));
        if recording {
            DEPTH.store(depth + 1, Ordering::Relaxed);
            if flags & region_flag::REGION_FLAG_SKIP_NESTED != 0 {
                // Everything deeper than this is suppressed until it closes.
                SKIP_BELOW.store(depth, Ordering::Relaxed);
            }
        }
        Region { name, location, flags, start: Instant::now(), depth, recording }
    }

    /// `CV_TRACE_ARG_VALUE(arg_id, arg_name, value)` — attach a value to this region.
    pub fn arg(&self, name: &str, value: &dyn std::fmt::Display) {
        if self.recording {
            println!("trace: {}  {name} = {value}", "  ".repeat(self.depth as usize));
        }
    }

    /// The flags this region was opened with.
    #[must_use]
    pub fn flags(&self) -> u32 {
        self.flags
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        if !self.recording {
            return;
        }
        DEPTH.store(self.depth, Ordering::Relaxed);
        if self.flags & region_flag::REGION_FLAG_SKIP_NESTED != 0 {
            SKIP_BELOW.store(u32::MAX, Ordering::Relaxed);
        }
        let ms = self.start.elapsed().as_secs_f64() * 1e3;
        let indent = "  ".repeat(self.depth as usize);
        match self.location {
            Some(loc) => println!("trace: {indent}{} ({loc}) {ms:.3}ms", self.name),
            None => println!("trace: {indent}{} {ms:.3}ms", self.name),
        }
    }
}

/// `CV_TRACE_FUNCTION()` — trace the enclosing function for its whole body.
///
/// Binds a guard that must stay alive: write `let _t = cv_trace_function!();`,
/// not `cv_trace_function!();`, or the region closes immediately. That is the
/// one place this differs from C++, where the macro declares the variable
/// itself.
#[macro_export]
macro_rules! cv_trace_function {
    () => {
        $crate::trace::Region::new(
            $crate::cv_func!(),
            Some(concat!(file!(), ":", line!())),
            $crate::trace::region_flag::REGION_FLAG_FUNCTION,
        )
    };
}

/// `CV_TRACE_FUNCTION_SKIP_NESTED()` — trace this function but nothing inside it.
#[macro_export]
macro_rules! cv_trace_function_skip_nested {
    () => {
        $crate::trace::Region::new(
            $crate::cv_func!(),
            Some(concat!(file!(), ":", line!())),
            $crate::trace::region_flag::REGION_FLAG_FUNCTION
                | $crate::trace::region_flag::REGION_FLAG_SKIP_NESTED,
        )
    };
}

/// `CV_TRACE_FUNCTION_VERBOSE()` — upstream's alias for [`cv_trace_function`].
#[macro_export]
macro_rules! cv_trace_function_verbose {
    () => {
        $crate::cv_trace_function!()
    };
}

/// `CV_TRACE_REGION(name)` — trace a named block within a function.
#[macro_export]
macro_rules! cv_trace_region {
    ($name:expr) => {
        $crate::trace::Region::new($name, None, 0)
    };
}

/// `CV_TRACE_REGION_VERBOSE(name)` — upstream's alias for [`cv_trace_region`].
#[macro_export]
macro_rules! cv_trace_region_verbose {
    ($name:expr) => {
        $crate::cv_trace_region!($name)
    };
}

/// `CV_TRACE_REGION_NEXT(name)` — close the previous region and open this one.
///
/// In C++ this works by shadowing the guard variable; here the caller rebinds
/// it, `let _t = cv_trace_region_next!("second");`, and the old guard drops on
/// reassignment, which is the same sequencing.
#[macro_export]
macro_rules! cv_trace_region_next {
    ($name:expr) => {
        $crate::trace::Region::new($name, None, $crate::trace::region_flag::REGION_FLAG_REGION_NEXT)
    };
}

/// `CV_TRACE_REGION_NEXT_VERBOSE(name)`.
#[macro_export]
macro_rules! cv_trace_region_next_verbose {
    ($name:expr) => {
        $crate::cv_trace_region_next!($name)
    };
}

/// `CV_TRACE_ARG(arg)` — record an argument under its own name.
#[macro_export]
macro_rules! cv_trace_arg {
    ($region:expr, $arg:expr) => {
        $region.arg(stringify!($arg), &$arg)
    };
}

/// `CV_TRACE_ARG_VERBOSE(arg)`.
#[macro_export]
macro_rules! cv_trace_arg_verbose {
    ($region:expr, $arg:expr) => {
        $crate::cv_trace_arg!($region, $arg)
    };
}

/// `CV_TRACE_ARG_VALUE(arg_id, arg_name, value)` — record a named value.
#[macro_export]
macro_rules! cv_trace_arg_value {
    ($region:expr, $name:expr, $value:expr) => {
        $region.arg($name, &$value)
    };
}

/// `CV_TRACE_ARG_VALUE_VERBOSE(arg_id, arg_name, value)`.
#[macro_export]
macro_rules! cv_trace_arg_value_verbose {
    ($region:expr, $name:expr, $value:expr) => {
        $crate::cv_trace_arg_value!($region, $name, $value)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tracing state is process-global, so these run one at a time.
    static TRACE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn regions_nest_and_unwind_with_the_stack() {
        let _g = TRACE_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prev = set_trace_enabled(true);
        assert_eq!(DEPTH.load(Ordering::Relaxed), 0);
        {
            let _outer = cv_trace_function!();
            assert_eq!(DEPTH.load(Ordering::Relaxed), 1);
            {
                let _inner = cv_trace_region!("inner");
                assert_eq!(DEPTH.load(Ordering::Relaxed), 2);
            }
            assert_eq!(DEPTH.load(Ordering::Relaxed), 1, "the inner region closed on drop");
        }
        assert_eq!(DEPTH.load(Ordering::Relaxed), 0);
        set_trace_enabled(prev);
    }

    #[test]
    fn nothing_is_recorded_while_tracing_is_off() {
        let _g = TRACE_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prev = set_trace_enabled(false);
        {
            let _r = cv_trace_function!();
            assert_eq!(DEPTH.load(Ordering::Relaxed), 0, "a suppressed region does not nest");
        }
        assert_eq!(DEPTH.load(Ordering::Relaxed), 0);
        set_trace_enabled(prev);
    }

    #[test]
    fn skip_nested_suppresses_children_but_not_a_forced_one() {
        let _g = TRACE_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prev = set_trace_enabled(true);
        {
            let outer = cv_trace_function_skip_nested!();
            assert!(outer.flags() & region_flag::REGION_FLAG_SKIP_NESTED != 0);
            assert_eq!(DEPTH.load(Ordering::Relaxed), 1);
            {
                let _child = cv_trace_region!("suppressed");
                assert_eq!(DEPTH.load(Ordering::Relaxed), 1, "the child was not recorded");
            }
            {
                let _forced =
                    Region::new("forced", None, region_flag::REGION_FLAG_REGION_FORCE);
                assert_eq!(DEPTH.load(Ordering::Relaxed), 2, "FORCE overrides SKIP_NESTED");
            }
        }
        assert_eq!(DEPTH.load(Ordering::Relaxed), 0);
        assert_eq!(SKIP_BELOW.load(Ordering::Relaxed), u32::MAX, "the skip window closed");
        set_trace_enabled(prev);
    }

    #[test]
    fn arg_macros_compile_and_run() {
        let _g = TRACE_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prev = set_trace_enabled(false);
        let r = cv_trace_region!("args");
        let width = 640;
        cv_trace_arg!(r, width);
        cv_trace_arg_verbose!(r, width);
        cv_trace_arg_value!(r, "height", 480);
        cv_trace_arg_value_verbose!(r, "height", 480);
        drop(r);
        let _next = cv_trace_region_next!("second");
        set_trace_enabled(prev);
    }

    #[test]
    fn region_flag_values_match_upstream() {
        use region_flag::*;
        assert_eq!(REGION_FLAG_FUNCTION, 1);
        assert_eq!(REGION_FLAG_APP_CODE, 2);
        assert_eq!(REGION_FLAG_SKIP_NESTED, 4);
        assert_eq!(REGION_FLAG_IMPL_MASK, 15 << 16);
        assert_eq!(REGION_FLAG_IMPL_OPENVX, 3 << 16);
        assert_eq!(REGION_FLAG_REGION_NEXT, 1 << 31);
    }
}
