//! `CV_Check*` — argument validation with a decoded failure message
//! (`modules/core/include/opencv2/core/check.hpp`, `modules/core/src/check.cpp`).
//!
//! These do what [`crate::cv_assert`] does, but the message names both operands
//! and prints their values, which is the difference between "Assertion failed:
//! `src.depth() == dst.depth()`" and being told which two depths they actually
//! were. `CV_CheckTypeEQ` and `CV_CheckDepthEQ` go further and decode the
//! integers into `CV_8UC3`-style names.
//!
//! Upstream throws; every macro here `return`s an `Err` instead, matching the
//! convention the rest of this crate uses. That means they can only be written
//! inside a function returning [`crate::Result`] — the same restriction
//! [`crate::cv_assert`] already imposes.
//!
//! The message text is reproduced exactly, so a failure here reads the same as
//! one from OpenCV:
//!
//! ```text
//! bad depth (expected: 'a == b'), where
//!     'a' is 5 (CV_32F)
//! must be equal to
//!     'b' is 0 (CV_8U)
//! ```

use crate::types::{cv_mat_cn, cv_mat_depth};

/// `cv::detail::TestOp` — which comparison a check performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum TestOp {
    /// `TEST_CUSTOM` — an arbitrary predicate rather than a binary comparison.
    Custom = 0,
    /// `TEST_EQ`
    Eq = 1,
    /// `TEST_NE`
    Ne = 2,
    /// `TEST_LE`
    Le = 3,
    /// `TEST_LT`
    Lt = 4,
    /// `TEST_GE`
    Ge = 5,
    /// `TEST_GT`
    Gt = 6,
}

/// `CV__LAST_TEST_OP` — one past the last [`TestOp`].
pub const CV_LAST_TEST_OP: i32 = 7;

/// The `CV__*` expansion helpers of the check family.
///
/// Upstream needs a macro per comparison because a C macro cannot take an
/// operator as a parameter; [`crate::cv_check_op`] takes a [`TestOp`] value
/// instead, so these name what each one tested rather than generating code.
pub mod expansion {
    /// `CV__TEST_EQ(v1, v2)`.
    #[must_use]
    pub fn cv_test_eq<T: PartialOrd>(a: &T, b: &T) -> bool {
        a == b
    }
    /// `CV__TEST_NE(v1, v2)`.
    #[must_use]
    pub fn cv_test_ne<T: PartialOrd>(a: &T, b: &T) -> bool {
        a != b
    }
    /// `CV__TEST_LE(v1, v2)`.
    #[must_use]
    pub fn cv_test_le<T: PartialOrd>(a: &T, b: &T) -> bool {
        a <= b
    }
    /// `CV__TEST_LT(v1, v2)`.
    #[must_use]
    pub fn cv_test_lt<T: PartialOrd>(a: &T, b: &T) -> bool {
        a < b
    }
    /// `CV__TEST_GE(v1, v2)`.
    #[must_use]
    pub fn cv_test_ge<T: PartialOrd>(a: &T, b: &T) -> bool {
        a >= b
    }
    /// `CV__TEST_GT(v1, v2)`.
    #[must_use]
    pub fn cv_test_gt<T: PartialOrd>(a: &T, b: &T) -> bool {
        a > b
    }
    /// `CV__CHECK(...)` — see [`crate::cv_check_op`].
    pub const CV__CHECK: &str = "cv_check_op!";
    /// `CV__CHECK_CUSTOM_TEST(...)` — see [`crate::cv_check_custom`].
    pub const CV__CHECK_CUSTOM_TEST: &str = "cv_check_custom!";
    /// `CV__DEFINE_CHECK_CONTEXT(...)` — upstream caches a static context per
    /// call site; here the message is built from `stringify!` at the point of
    /// failure, so nothing is cached.
    pub const CV__DEFINE_CHECK_CONTEXT: &str = "";
    /// `CV__CHECK_LOCATION_VARNAME(id)` — the name of that cached context.
    pub const CV__CHECK_LOCATION_VARNAME: &str = "";
}

impl TestOp {
    /// The operator as it appears in the "expected:" clause.
    #[must_use]
    pub fn math(self) -> &'static str {
        match self {
            TestOp::Custom => "???",
            TestOp::Eq => "==",
            TestOp::Ne => "!=",
            TestOp::Le => "<=",
            TestOp::Lt => "<",
            TestOp::Ge => ">=",
            TestOp::Gt => ">",
        }
    }

    /// The phrase used on the "must be ..." line.
    #[must_use]
    pub fn phrase(self) -> &'static str {
        match self {
            TestOp::Custom => "{custom check}",
            TestOp::Eq => "equal to",
            TestOp::Ne => "not equal to",
            TestOp::Le => "less than or equal to",
            TestOp::Lt => "less than",
            TestOp::Ge => "greater than or equal to",
            TestOp::Gt => "greater than",
        }
    }
}

/// `cv::depthToString(depth)` — `CV_8U`, or `<invalid depth>`.
#[must_use]
pub fn depth_to_string(depth: i32) -> &'static str {
    match depth {
        0 => "CV_8U",
        1 => "CV_8S",
        2 => "CV_16U",
        3 => "CV_16S",
        4 => "CV_32S",
        5 => "CV_32F",
        6 => "CV_64F",
        7 => "CV_16F",
        _ => "<invalid depth>",
    }
}

/// `cv::typeToString(type)` — `CV_8UC3`, or `<invalid type>`.
#[must_use]
pub fn type_to_string(ty: i32) -> String {
    let depth = cv_mat_depth(ty);
    if (0..=7).contains(&depth) {
        format!("{}C{}", depth_to_string(depth), cv_mat_cn(ty))
    } else {
        "<invalid type>".to_string()
    }
}

/// Build the message `check_failed_auto` produces for a binary comparison.
///
/// Public because the macros below expand to a call to it; a caller has no
/// reason to invoke it directly.
#[must_use]
pub fn check_failed_message(
    message: &str,
    op: TestOp,
    p1_str: &str,
    v1: &str,
    p2_str: &str,
    v2: &str,
) -> String {
    let mut s = format!(
        "{message} (expected: '{p1_str} {} {p2_str}'), where\n    '{p1_str}' is {v1}\n",
        op.math()
    );
    if op != TestOp::Custom {
        s.push_str(&format!("must be {}\n", op.phrase()));
    }
    s.push_str(&format!("    '{p2_str}' is {v2}"));
    s
}

/// Build the message the custom-predicate checks produce.
#[must_use]
pub fn check_failed_custom_message(
    message: &str,
    p_str: &str,
    v: &str,
    test_expr_str: &str,
) -> String {
    format!("{message}:\n    '{test_expr_str}'\nwhere\n    '{p_str}' is {v}")
}

/// The comparison half of the `CV_Check*` family, shared by all six operators.
///
/// Takes the operands twice — once as expressions and once stringified — which
/// is what lets the message name them the way the caller wrote them.
#[macro_export]
macro_rules! cv_check_op {
    ($op:expr, $v1:expr, $v2:expr, $msg:expr, $fmt:expr) => {{
        let (a, b) = ($v1, $v2);
        let ok = match $op {
            $crate::check::TestOp::Eq => a == b,
            $crate::check::TestOp::Ne => a != b,
            $crate::check::TestOp::Le => a <= b,
            $crate::check::TestOp::Lt => a < b,
            $crate::check::TestOp::Ge => a >= b,
            $crate::check::TestOp::Gt => a > b,
            $crate::check::TestOp::Custom => true,
        };
        if !ok {
            return Err($crate::Error::new(
                $crate::Code::StsError,
                $crate::check::check_failed_message(
                    $msg,
                    $op,
                    stringify!($v1),
                    &$fmt(a),
                    stringify!($v2),
                    &$fmt(b),
                ),
            ));
        }
    }};
}

/// Render an operand with `Display`, which is what the plain checks use.
#[doc(hidden)]
pub fn plain<T: std::fmt::Display>(v: T) -> String {
    v.to_string()
}

/// Render an integer as `5 (CV_32F)`, for the depth-aware checks.
#[doc(hidden)]
pub fn with_depth(v: i32) -> String {
    format!("{v} ({})", depth_to_string(v))
}

/// Render an integer as `16 (CV_8UC3)`, for the type-aware checks.
#[doc(hidden)]
pub fn with_type(v: i32) -> String {
    format!("{v} ({})", type_to_string(v))
}

/// `CV_CheckEQ(v1, v2, msg)`.
#[macro_export]
macro_rules! cv_check_eq {
    ($v1:expr, $v2:expr, $msg:expr) => {
        $crate::cv_check_op!($crate::check::TestOp::Eq, $v1, $v2, $msg, $crate::check::plain)
    };
}

/// `CV_CheckNE(v1, v2, msg)`.
#[macro_export]
macro_rules! cv_check_ne {
    ($v1:expr, $v2:expr, $msg:expr) => {
        $crate::cv_check_op!($crate::check::TestOp::Ne, $v1, $v2, $msg, $crate::check::plain)
    };
}

/// `CV_CheckLE(v1, v2, msg)`.
#[macro_export]
macro_rules! cv_check_le {
    ($v1:expr, $v2:expr, $msg:expr) => {
        $crate::cv_check_op!($crate::check::TestOp::Le, $v1, $v2, $msg, $crate::check::plain)
    };
}

/// `CV_CheckLT(v1, v2, msg)`.
#[macro_export]
macro_rules! cv_check_lt {
    ($v1:expr, $v2:expr, $msg:expr) => {
        $crate::cv_check_op!($crate::check::TestOp::Lt, $v1, $v2, $msg, $crate::check::plain)
    };
}

/// `CV_CheckGE(v1, v2, msg)`.
#[macro_export]
macro_rules! cv_check_ge {
    ($v1:expr, $v2:expr, $msg:expr) => {
        $crate::cv_check_op!($crate::check::TestOp::Ge, $v1, $v2, $msg, $crate::check::plain)
    };
}

/// `CV_CheckGT(v1, v2, msg)`.
#[macro_export]
macro_rules! cv_check_gt {
    ($v1:expr, $v2:expr, $msg:expr) => {
        $crate::cv_check_op!($crate::check::TestOp::Gt, $v1, $v2, $msg, $crate::check::plain)
    };
}

/// `CV_CheckTypeEQ(t1, t2, msg)` — reports both types by name.
#[macro_export]
macro_rules! cv_check_type_eq {
    ($v1:expr, $v2:expr, $msg:expr) => {
        $crate::cv_check_op!($crate::check::TestOp::Eq, $v1, $v2, $msg, $crate::check::with_type)
    };
}

/// `CV_CheckDepthEQ(d1, d2, msg)` — reports both depths by name.
#[macro_export]
macro_rules! cv_check_depth_eq {
    ($v1:expr, $v2:expr, $msg:expr) => {
        $crate::cv_check_op!($crate::check::TestOp::Eq, $v1, $v2, $msg, $crate::check::with_depth)
    };
}

/// `CV_CheckChannelsEQ(c1, c2, msg)`.
#[macro_export]
macro_rules! cv_check_channels_eq {
    ($v1:expr, $v2:expr, $msg:expr) => {
        $crate::cv_check_op!($crate::check::TestOp::Eq, $v1, $v2, $msg, $crate::check::plain)
    };
}

/// The custom-predicate half of the family.
#[macro_export]
macro_rules! cv_check_custom {
    ($v:expr, $test:expr, $msg:expr, $fmt:expr) => {{
        let value = $v;
        if !($test) {
            return Err($crate::Error::new(
                $crate::Code::StsError,
                $crate::check::check_failed_custom_message(
                    $msg,
                    stringify!($v),
                    &$fmt(value),
                    stringify!($test),
                ),
            ));
        }
    }};
}

/// `CV_Check(v, test_expr, msg)` — an arbitrary predicate on `v`.
#[macro_export]
macro_rules! cv_check {
    ($v:expr, $test:expr, $msg:expr) => {
        $crate::cv_check_custom!($v, $test, $msg, $crate::check::plain)
    };
}

/// `CV_CheckType(t, test_expr, msg)` — for example `t == CV_8UC1 || t == CV_8UC3`.
#[macro_export]
macro_rules! cv_check_type {
    ($v:expr, $test:expr, $msg:expr) => {
        $crate::cv_check_custom!($v, $test, $msg, $crate::check::with_type)
    };
}

/// `CV_CheckDepth(d, test_expr, msg)` — for example `d == CV_32F || d == CV_64F`.
#[macro_export]
macro_rules! cv_check_depth {
    ($v:expr, $test:expr, $msg:expr) => {
        $crate::cv_check_custom!($v, $test, $msg, $crate::check::with_depth)
    };
}

/// `CV_CheckChannels(c, test_expr, msg)` — for example `c == 1 || c == 3`.
#[macro_export]
macro_rules! cv_check_channels {
    ($v:expr, $test:expr, $msg:expr) => {
        $crate::cv_check_custom!($v, $test, $msg, $crate::check::plain)
    };
}

// `cv_check_true` and `cv_check_false` cannot delegate to `cv_check_custom`:
// the predicate they would pass is the value itself, and macro hygiene keeps a
// binding made inside one macro invisible to an expression substituted from
// another. They expand directly instead.

/// `CV_CheckTrue(v, msg)`.
#[macro_export]
macro_rules! cv_check_true {
    ($v:expr, $msg:expr) => {{
        let value = $v;
        if !value {
            return Err($crate::Error::new(
                $crate::Code::StsError,
                $crate::check::check_failed_custom_message(
                    $msg,
                    stringify!($v),
                    &$crate::check::plain(value),
                    "",
                ),
            ));
        }
    }};
}

/// `CV_CheckFalse(v, msg)`.
#[macro_export]
macro_rules! cv_check_false {
    ($v:expr, $msg:expr) => {{
        let value = $v;
        if value {
            return Err($crate::Error::new(
                $crate::Code::StsError,
                $crate::check::check_failed_custom_message(
                    $msg,
                    stringify!($v),
                    &$crate::check::plain(value),
                    "",
                ),
            ));
        }
    }};
}

/// `CV_DbgCheck(v, test_expr, msg)` — [`cv_check`] in a debug build only.
///
/// Upstream compiles the `CV_DbgCheck*` family away under `NDEBUG`; the Rust
/// equivalent is `debug_assertions`, which is off in release for the same
/// reason.
#[macro_export]
macro_rules! cv_dbg_check {
    ($v:expr, $test:expr, $msg:expr) => {
        if cfg!(debug_assertions) {
            $crate::cv_check!($v, $test, $msg)
        }
    };
}

/// `CV_DbgCheckEQ(v1, v2, msg)`.
#[macro_export]
macro_rules! cv_dbg_check_eq {
    ($v1:expr, $v2:expr, $msg:expr) => {
        if cfg!(debug_assertions) {
            $crate::cv_check_eq!($v1, $v2, $msg)
        }
    };
}

/// `CV_DbgCheckNE(v1, v2, msg)`.
#[macro_export]
macro_rules! cv_dbg_check_ne {
    ($v1:expr, $v2:expr, $msg:expr) => {
        if cfg!(debug_assertions) {
            $crate::cv_check_ne!($v1, $v2, $msg)
        }
    };
}

/// `CV_DbgCheckLE(v1, v2, msg)`.
#[macro_export]
macro_rules! cv_dbg_check_le {
    ($v1:expr, $v2:expr, $msg:expr) => {
        if cfg!(debug_assertions) {
            $crate::cv_check_le!($v1, $v2, $msg)
        }
    };
}

/// `CV_DbgCheckLT(v1, v2, msg)`.
#[macro_export]
macro_rules! cv_dbg_check_lt {
    ($v1:expr, $v2:expr, $msg:expr) => {
        if cfg!(debug_assertions) {
            $crate::cv_check_lt!($v1, $v2, $msg)
        }
    };
}

/// `CV_DbgCheckGE(v1, v2, msg)`.
#[macro_export]
macro_rules! cv_dbg_check_ge {
    ($v1:expr, $v2:expr, $msg:expr) => {
        if cfg!(debug_assertions) {
            $crate::cv_check_ge!($v1, $v2, $msg)
        }
    };
}

/// `CV_DbgCheckGT(v1, v2, msg)`.
#[macro_export]
macro_rules! cv_dbg_check_gt {
    ($v1:expr, $v2:expr, $msg:expr) => {
        if cfg!(debug_assertions) {
            $crate::cv_check_gt!($v1, $v2, $msg)
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::error::Result;
    use crate::types::{CV_32F, CV_8U, CV_8UC3};

    #[test]
    fn depth_and_type_decode_to_upstreams_names() {
        assert_eq!(super::depth_to_string(CV_8U), "CV_8U");
        assert_eq!(super::depth_to_string(CV_32F), "CV_32F");
        assert_eq!(super::depth_to_string(99), "<invalid depth>");
        assert_eq!(super::type_to_string(CV_8UC3.to_i32()), "CV_8UC3");
        // `CV_MAT_DEPTH` masks to three bits, so the depth it extracts is
        // always 0..=7 and the `<invalid type>` branch cannot be reached —
        // upstream decodes -1 the same way rather than rejecting it.
        assert_eq!(super::type_to_string(-1), "CV_16FC512");
    }

    fn eq_check(a: i32, b: i32) -> Result<()> {
        cv_check_eq!(a, b, "sizes must match");
        Ok(())
    }

    /// The layout is upstream's, line for line.
    #[test]
    fn a_failed_comparison_names_both_operands() {
        assert!(eq_check(3, 3).is_ok());
        let e = eq_check(3, 5).unwrap_err();
        assert_eq!(
            e.msg,
            "sizes must match (expected: 'a == b'), where\n    'a' is 3\nmust be equal to\n    'b' is 5"
        );
        assert_eq!(e.code, crate::Code::StsError);
    }

    #[test]
    fn every_operator_reports_its_own_phrase() {
        fn ne(a: i32, b: i32) -> Result<()> {
            cv_check_ne!(a, b, "m");
            Ok(())
        }
        fn lt(a: i32, b: i32) -> Result<()> {
            cv_check_lt!(a, b, "m");
            Ok(())
        }
        fn ge(a: i32, b: i32) -> Result<()> {
            cv_check_ge!(a, b, "m");
            Ok(())
        }
        assert!(ne(1, 1).unwrap_err().msg.contains("must be not equal to"));
        assert!(lt(5, 1).unwrap_err().msg.contains("must be less than"));
        assert!(ge(1, 5).unwrap_err().msg.contains("must be greater than or equal to"));
        assert!(ne(1, 2).is_ok() && lt(1, 5).is_ok() && ge(5, 1).is_ok());
    }

    #[test]
    fn depth_and_type_checks_decode_their_values() {
        fn d(a: i32, b: i32) -> Result<()> {
            cv_check_depth_eq!(a, b, "bad depth");
            Ok(())
        }
        let e = d(CV_32F, CV_8U).unwrap_err();
        assert!(e.msg.contains("'a' is 5 (CV_32F)"), "{}", e.msg);
        assert!(e.msg.contains("'b' is 0 (CV_8U)"), "{}", e.msg);

        fn t(a: i32, b: i32) -> Result<()> {
            cv_check_type_eq!(a, b, "bad type");
            Ok(())
        }
        let e = t(CV_8UC3.to_i32(), 0).unwrap_err();
        assert!(e.msg.contains("(CV_8UC3)") && e.msg.contains("(CV_8UC1)"), "{}", e.msg);
    }

    #[test]
    fn a_custom_predicate_prints_the_expression_it_tested() {
        fn f(depth: i32) -> Result<()> {
            cv_check_depth!(depth, depth == CV_32F || depth == CV_8U, "unsupported depth");
            Ok(())
        }
        assert!(f(CV_32F).is_ok() && f(CV_8U).is_ok());
        let e = f(2).unwrap_err();
        assert_eq!(
            e.msg,
            "unsupported depth:\n    'depth == CV_32F || depth == CV_8U'\nwhere\n    'depth' is 2 (CV_16U)"
        );
    }

    #[test]
    fn true_and_false_checks_test_the_value_itself() {
        fn t(v: bool) -> Result<()> {
            cv_check_true!(v, "must hold");
            Ok(())
        }
        fn f(v: bool) -> Result<()> {
            cv_check_false!(v, "must not hold");
            Ok(())
        }
        assert!(t(true).is_ok() && f(false).is_ok());
        assert!(t(false).unwrap_err().msg.contains("must hold"));
        assert!(f(true).unwrap_err().msg.contains("must not hold"));
    }

    #[test]
    fn the_debug_variants_follow_the_build_profile() {
        fn d(a: i32, b: i32) -> Result<()> {
            cv_dbg_check_eq!(a, b, "m");
            Ok(())
        }
        assert!(d(1, 1).is_ok());
        // Under `cargo test` debug assertions are on, so this must fire; in a
        // release build the check compiles away and the call succeeds.
        assert_eq!(d(1, 2).is_err(), cfg!(debug_assertions));
    }

    #[test]
    fn checks_work_on_floats_and_sizes_too() {
        fn f(a: f64, b: f64) -> Result<()> {
            cv_check_le!(a, b, "m");
            Ok(())
        }
        assert!(f(1.5, 2.5).is_ok());
        assert!(f(2.5, 1.5).unwrap_err().msg.contains("'a' is 2.5"));

        fn s(a: crate::Size, b: crate::Size) -> Result<()> {
            cv_check_eq!(format!("{a:?}"), format!("{b:?}"), "sizes differ");
            Ok(())
        }
        assert!(s(crate::Size::new(1, 2), crate::Size::new(1, 2)).is_ok());
        assert!(s(crate::Size::new(1, 2), crate::Size::new(3, 4)).is_err());
    }
}
