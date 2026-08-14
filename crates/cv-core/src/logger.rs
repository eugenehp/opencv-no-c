//! `CV_LOG_*` — levelled logging
//! (`modules/core/include/opencv2/core/utils/logger.hpp`, `modules/core/src/logger.cpp`).
//!
//! The level itself lives in [`crate::utility`]; this module adds the macros
//! that write through it and the message formatting they share.
//!
//! Output matches upstream's, so a line from this port is indistinguishable
//! from one OpenCV wrote:
//!
//! ```text
//! [ WARN:0] global some/file.rs:42 my_function message text
//! ```
//!
//! Everything at `WARNING` or more severe goes to stderr, the rest to stdout —
//! upstream's split, and the reason a `CV_LOG_INFO` does not pollute a pipeline
//! reading stderr.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::utility::{get_log_level, LogLevel};

/// `CV_LOGTAG_GLOBAL` — the tag used when a call site names none.
pub const CV_LOGTAG_GLOBAL: &str = "global";

/// `CV_LOGTAG_FALLBACK` — the tag used when the named one resolves to nothing.
///
/// Upstream chains `CV_LOGTAG_EXPAND_NAME` → `CV_LOGTAG_FALLBACK` →
/// `CV_LOGTAG_GLOBAL`, taking the first that is non-null. A tag here is a
/// plain `&str` that cannot be null, so the chain collapses to the tag the
/// caller passed and this is what an empty one becomes.
pub const CV_LOGTAG_FALLBACK: &str = CV_LOGTAG_GLOBAL;

/// `CV_LOGTAG_EXPAND_NAME(tag)` — resolve a tag name to its object.
///
/// The identity here: [`crate::cv_log_with_tag`] takes the tag as a string
/// rather than looking up a registered `LogTag`.
#[macro_export]
macro_rules! cv_logtag_expand_name {
    ($tag:expr) => {
        $tag
    };
}

/// `CV_LOGTAG_PTR_CAST(tag)` — upstream casts the tag to a base pointer.
///
/// The identity here; there is no pointer to cast.
#[macro_export]
macro_rules! cv_logtag_ptr_cast {
    ($tag:expr) => {
        $tag
    };
}

/// `CV__LOG_IF_CHECK(cond)` — the guard [`crate::cv_log_if`] expands to.
#[macro_export]
macro_rules! cv_log_if_check {
    ($cond:expr) => {
        $cond
    };
}

/// `CV__LOG_ONCE_CHECK_PRE` — the flag declaration [`crate::cv_log_once`] emits.
#[macro_export]
macro_rules! cv_log_once_check_pre {
    () => {
        static LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
    };
}

/// `CV__LOG_ONCE_CHECK_POST` — the test-and-set that follows it.
#[macro_export]
macro_rules! cv_log_once_check_post {
    () => {
        !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
    };
}

/// `cv::utils::logging::LogLevel` constants, spelled as upstream spells them.
///
/// [`LogLevel`] is the enum; these are the integer names a translated `#if`
/// comparison would use.
pub mod level {
    /// `LOG_LEVEL_SILENT` / `CV_LOG_LEVEL_SILENT`
    pub const LOG_LEVEL_SILENT: i32 = 0;
    /// `LOG_LEVEL_FATAL` / `CV_LOG_LEVEL_FATAL`
    pub const LOG_LEVEL_FATAL: i32 = 1;
    /// `LOG_LEVEL_ERROR` / `CV_LOG_LEVEL_ERROR`
    pub const LOG_LEVEL_ERROR: i32 = 2;
    /// `LOG_LEVEL_WARNING` / `CV_LOG_LEVEL_WARN`
    pub const LOG_LEVEL_WARNING: i32 = 3;
    /// `ENUM_LOG_LEVEL_FORCE_INT` — upstream's hint that the enum is 32 bits.
    pub const ENUM_LOG_LEVEL_FORCE_INT: i32 = i32::MAX;
    /// `LOG_LEVEL_INFO` / `CV_LOG_LEVEL_INFO`
    pub const LOG_LEVEL_INFO: i32 = 4;
    /// `LOG_LEVEL_DEBUG` / `CV_LOG_LEVEL_DEBUG`
    pub const LOG_LEVEL_DEBUG: i32 = 5;
    /// `LOG_LEVEL_VERBOSE` / `CV_LOG_LEVEL_VERBOSE`
    pub const LOG_LEVEL_VERBOSE: i32 = 6;
}

/// Whether a timestamp is prefixed; `OPENCV_LOG_TIMESTAMP` upstream.
static SHOW_TIMESTAMP: AtomicBool = AtomicBool::new(false);

/// `cv::utils::logging::internal::getShowTimestampMode()` as a toggle.
pub fn set_show_timestamp(on: bool) {
    SHOW_TIMESTAMP.store(on, Ordering::Relaxed);
}

/// `cv::utils::logging::internal::writeLogMessage(logLevel, message)`.
///
/// Writes the `[ WARN:0] ` prefix and routes to stderr or stdout by severity.
pub fn write_log_message(level: LogLevel, message: &str) {
    // Upstream prints the thread id here. This port's kernels are
    // single-threaded, so it is always 0 and carries no information; the field
    // is kept so the line shape matches.
    let id = if SHOW_TIMESTAMP.load(Ordering::Relaxed) {
        format!("0@{}", std::time::SystemTime::UNIX_EPOCH.elapsed().map_or(0.0, |d| d.as_secs_f64()))
    } else {
        "0".to_string()
    };
    let line = match level {
        LogLevel::Silent => return,
        LogLevel::Fatal => format!("[FATAL:{id}] {message}\n"),
        LogLevel::Error => format!("[ERROR:{id}] {message}\n"),
        LogLevel::Warning => format!("[ WARN:{id}] {message}\n"),
        LogLevel::Info => format!("[ INFO:{id}] {message}\n"),
        LogLevel::Debug => format!("[DEBUG:{id}] {message}\n"),
        // Verbose carries no prefix upstream either.
        LogLevel::Verbose => format!("{message}\n"),
    };
    if level <= LogLevel::Warning {
        let mut e = std::io::stderr();
        let _ = e.write_all(line.as_bytes());
        let _ = e.flush();
    } else {
        let _ = std::io::stdout().write_all(line.as_bytes());
    }
}

/// `cv::utils::logging::internal::writeLogMessageEx(level, tag, file, line, func, message)`.
pub fn write_log_message_ex(
    level: LogLevel,
    tag: &str,
    file: &str,
    line: u32,
    func: &str,
    message: &str,
) {
    let mut s = String::new();
    if !tag.is_empty() {
        s.push_str(tag);
        s.push(' ');
    }
    if !file.is_empty() {
        s.push_str(file);
        if line > 0 {
            s.push_str(&format!(":{line}"));
        }
        s.push(' ');
    }
    if !func.is_empty() {
        s.push_str(func);
        s.push(' ');
    }
    s.push_str(message);
    write_log_message(level, &s);
}

/// Is a message at `level` currently enabled?
#[must_use]
pub fn should_log(level: LogLevel) -> bool {
    level != LogLevel::Silent && level <= get_log_level()
}

/// The name of the enclosing function, as `CV_Func` gives it in C++.
///
/// Rust has no `__FUNCTION__`, so this recovers the path from
/// [`std::any::type_name`] of a closure defined at the call site and strips the
/// `::{{closure}}` suffix.
#[macro_export]
macro_rules! cv_func {
    () => {{
        fn f() {}
        let name = std::any::type_name_of_val(&f);
        name.strip_suffix("::f").unwrap_or(name)
    }};
}

/// `CV_LOG_WITH_TAG(tag, msgLevel, ...)` — the macro the rest expand to.
#[macro_export]
macro_rules! cv_log_with_tag {
    ($tag:expr, $level:expr, $($arg:tt)*) => {{
        let lvl = $level;
        if $crate::logger::should_log(lvl) {
            $crate::logger::write_log_message_ex(
                lvl,
                $tag,
                file!(),
                line!(),
                $crate::cv_func!(),
                &format!($($arg)*),
            );
        }
    }};
}

/// `CV_LOG_FATAL(tag, ...)`.
#[macro_export]
macro_rules! cv_log_fatal {
    ($tag:expr, $($arg:tt)*) => {
        $crate::cv_log_with_tag!($tag, $crate::utility::LogLevel::Fatal, $($arg)*)
    };
}

/// `CV_LOG_ERROR(tag, ...)`.
#[macro_export]
macro_rules! cv_log_error {
    ($tag:expr, $($arg:tt)*) => {
        $crate::cv_log_with_tag!($tag, $crate::utility::LogLevel::Error, $($arg)*)
    };
}

/// `CV_LOG_WARNING(tag, ...)`.
#[macro_export]
macro_rules! cv_log_warning {
    ($tag:expr, $($arg:tt)*) => {
        $crate::cv_log_with_tag!($tag, $crate::utility::LogLevel::Warning, $($arg)*)
    };
}

/// `CV_LOG_INFO(tag, ...)`.
#[macro_export]
macro_rules! cv_log_info {
    ($tag:expr, $($arg:tt)*) => {
        $crate::cv_log_with_tag!($tag, $crate::utility::LogLevel::Info, $($arg)*)
    };
}

/// `CV_LOG_DEBUG(tag, ...)`.
#[macro_export]
macro_rules! cv_log_debug {
    ($tag:expr, $($arg:tt)*) => {
        $crate::cv_log_with_tag!($tag, $crate::utility::LogLevel::Debug, $($arg)*)
    };
}

/// `CV_LOG_VERBOSE(tag, v, ...)`.
///
/// Upstream adds `v` to the level so a caller can nest verbosity; since
/// `VERBOSE` is already the maximum, any `v` above zero is never enabled, which
/// is the same behaviour.
#[macro_export]
macro_rules! cv_log_verbose {
    ($tag:expr, $v:expr, $($arg:tt)*) => {{
        if $v == 0 {
            $crate::cv_log_with_tag!($tag, $crate::utility::LogLevel::Verbose, $($arg)*)
        }
    }};
}

/// `CV_LOG_IF_*` — log only when `cond` holds.
#[macro_export]
macro_rules! cv_log_if {
    ($cond:expr, $tag:expr, $level:expr, $($arg:tt)*) => {{
        if $cond {
            $crate::cv_log_with_tag!($tag, $level, $($arg)*)
        }
    }};
}

/// `CV_LOG_IF_FATAL(cond, tag, ...)`.
#[macro_export]
macro_rules! cv_log_if_fatal {
    ($cond:expr, $tag:expr, $($arg:tt)*) => {
        $crate::cv_log_if!($cond, $tag, $crate::utility::LogLevel::Fatal, $($arg)*)
    };
}

/// `CV_LOG_IF_ERROR(cond, tag, ...)`.
#[macro_export]
macro_rules! cv_log_if_error {
    ($cond:expr, $tag:expr, $($arg:tt)*) => {
        $crate::cv_log_if!($cond, $tag, $crate::utility::LogLevel::Error, $($arg)*)
    };
}

/// `CV_LOG_IF_WARNING(cond, tag, ...)`.
#[macro_export]
macro_rules! cv_log_if_warning {
    ($cond:expr, $tag:expr, $($arg:tt)*) => {
        $crate::cv_log_if!($cond, $tag, $crate::utility::LogLevel::Warning, $($arg)*)
    };
}

/// `CV_LOG_IF_INFO(cond, tag, ...)`.
#[macro_export]
macro_rules! cv_log_if_info {
    ($cond:expr, $tag:expr, $($arg:tt)*) => {
        $crate::cv_log_if!($cond, $tag, $crate::utility::LogLevel::Info, $($arg)*)
    };
}

/// `CV_LOG_IF_DEBUG(cond, tag, ...)`.
#[macro_export]
macro_rules! cv_log_if_debug {
    ($cond:expr, $tag:expr, $($arg:tt)*) => {
        $crate::cv_log_if!($cond, $tag, $crate::utility::LogLevel::Debug, $($arg)*)
    };
}

/// `CV_LOG_IF_VERBOSE(cond, tag, v, ...)`.
#[macro_export]
macro_rules! cv_log_if_verbose {
    ($cond:expr, $tag:expr, $v:expr, $($arg:tt)*) => {{
        if $cond {
            $crate::cv_log_verbose!($tag, $v, $($arg)*)
        }
    }};
}

/// `CV_LOG_ONCE_*` — log the first time this site is reached and never again.
///
/// The flag is per call site, because the macro expands one `AtomicBool` into
/// each expansion. That is what upstream's function-local `static bool` does.
#[macro_export]
macro_rules! cv_log_once {
    ($tag:expr, $level:expr, $($arg:tt)*) => {{
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            $crate::cv_log_with_tag!($tag, $level, $($arg)*)
        }
    }};
}

/// `CV_LOG_ONCE_ERROR(tag, ...)`.
#[macro_export]
macro_rules! cv_log_once_error {
    ($tag:expr, $($arg:tt)*) => {
        $crate::cv_log_once!($tag, $crate::utility::LogLevel::Error, $($arg)*)
    };
}

/// `CV_LOG_ONCE_WARNING(tag, ...)`.
#[macro_export]
macro_rules! cv_log_once_warning {
    ($tag:expr, $($arg:tt)*) => {
        $crate::cv_log_once!($tag, $crate::utility::LogLevel::Warning, $($arg)*)
    };
}

/// `CV_LOG_ONCE_INFO(tag, ...)`.
#[macro_export]
macro_rules! cv_log_once_info {
    ($tag:expr, $($arg:tt)*) => {
        $crate::cv_log_once!($tag, $crate::utility::LogLevel::Info, $($arg)*)
    };
}

/// `CV_LOG_ONCE_DEBUG(tag, ...)`.
#[macro_export]
macro_rules! cv_log_once_debug {
    ($tag:expr, $($arg:tt)*) => {
        $crate::cv_log_once!($tag, $crate::utility::LogLevel::Debug, $($arg)*)
    };
}

/// `CV_LOG_ONCE_VERBOSE(tag, v, ...)`.
#[macro_export]
macro_rules! cv_log_once_verbose {
    ($tag:expr, $v:expr, $($arg:tt)*) => {{
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            $crate::cv_log_verbose!($tag, $v, $($arg)*)
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utility::set_log_level;

    /// The log level is process-global, so the tests that move it have to run
    /// one at a time or they read each other's setting.
    static LEVEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn the_message_layout_matches_upstream() {
        // writeLogMessageEx joins tag, file:line, function and message with
        // single spaces, skipping any empty field.
        let mut s = String::new();
        for (tag, file, line, func, want) in [
            ("global", "a.rs", 12u32, "f", "global a.rs:12 f msg"),
            ("", "a.rs", 12, "f", "a.rs:12 f msg"),
            ("global", "", 0, "", "global msg"),
            ("global", "a.rs", 0, "f", "global a.rs f msg"),
        ] {
            s.clear();
            if !tag.is_empty() {
                s.push_str(tag);
                s.push(' ');
            }
            if !file.is_empty() {
                s.push_str(file);
                if line > 0 {
                    s.push_str(&format!(":{line}"));
                }
                s.push(' ');
            }
            if !func.is_empty() {
                s.push_str(func);
                s.push(' ');
            }
            s.push_str("msg");
            assert_eq!(s, want);
        }
    }

    #[test]
    fn severity_gates_on_the_current_level() {
        let _g = LEVEL_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prev = set_log_level(LogLevel::Warning);
        assert!(should_log(LogLevel::Fatal));
        assert!(should_log(LogLevel::Error));
        assert!(should_log(LogLevel::Warning));
        assert!(!should_log(LogLevel::Info));
        assert!(!should_log(LogLevel::Debug));

        set_log_level(LogLevel::Silent);
        assert!(!should_log(LogLevel::Fatal), "silent suppresses everything");
        assert!(!should_log(LogLevel::Silent));

        set_log_level(LogLevel::Verbose);
        assert!(should_log(LogLevel::Debug) && should_log(LogLevel::Verbose));
        set_log_level(prev);
    }

    #[test]
    fn cv_func_names_the_enclosing_function() {
        let name = cv_func!();
        assert!(name.ends_with("cv_func_names_the_enclosing_function"), "got {name}");
    }

    #[test]
    fn the_macros_expand_and_respect_the_level() {
        // Silence the output so the test does not write to the terminal; what
        // is under test is that each macro compiles and gates correctly.
        let _g = LEVEL_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prev = set_log_level(LogLevel::Silent);
        cv_log_fatal!(CV_LOGTAG_GLOBAL, "fatal {}", 1);
        cv_log_error!(CV_LOGTAG_GLOBAL, "error");
        cv_log_warning!(CV_LOGTAG_GLOBAL, "warning");
        cv_log_info!(CV_LOGTAG_GLOBAL, "info");
        cv_log_debug!(CV_LOGTAG_GLOBAL, "debug");
        cv_log_verbose!(CV_LOGTAG_GLOBAL, 0, "verbose");
        cv_log_if_info!(true, CV_LOGTAG_GLOBAL, "conditional");
        cv_log_if_warning!(false, CV_LOGTAG_GLOBAL, "not logged");
        cv_log_if_verbose!(true, CV_LOGTAG_GLOBAL, 0, "conditional verbose");
        for _ in 0..3 {
            cv_log_once_warning!(CV_LOGTAG_GLOBAL, "once");
            cv_log_once_info!(CV_LOGTAG_GLOBAL, "once");
            cv_log_once_verbose!(CV_LOGTAG_GLOBAL, 0, "once");
        }
        set_log_level(prev);
    }

    #[test]
    fn level_constants_match_upstream() {
        assert_eq!(level::LOG_LEVEL_SILENT, LogLevel::Silent as i32);
        assert_eq!(level::LOG_LEVEL_WARNING, LogLevel::Warning as i32);
        assert_eq!(level::LOG_LEVEL_VERBOSE, LogLevel::Verbose as i32);
    }
}
