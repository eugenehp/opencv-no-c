//! The remaining `CV_*` macro families, as Rust macros or as markers
//! (`modules/core/include/opencv2/core/cvdef.h`,
//! `modules/core/include/opencv2/core/private.hpp`,
//! `modules/core/include/opencv2/core/base.hpp`).
//!
//! Three kinds of thing live here, and it is worth separating them:
//!
//! 1. **Macros with a real Rust equivalent.** `cv_enum_flags` generates the
//!    bitwise operators upstream's version does; `cv_impl_property` generates
//!    the getter/setter pair; `cv_assert_2`…`cv_assert_10` are the fixed-arity
//!    asserts. These do work.
//! 2. **Markers.** `CV_WRAP`, `CV_PROP`, `CV_IN_OUT` and the rest annotate
//!    declarations for OpenCV's binding generator and expand to nothing in
//!    C++. They are declared as empty string constants so that translated code
//!    has something to name, and so a reader can see they were considered
//!    rather than missed. They cannot *do* anything: Rust has no binding
//!    generator reading them.
//! 3. **Compiler plumbing.** `CV_DO_PRAGMA`, `CV__DEBUG_NS_BEGIN` and friends
//!    manipulate the preprocessor. Where Rust has an equivalent spelling it is
//!    given in the doc comment rather than generated.

/// `CV_ENUM_FLAGS(EnumType)` — give a fieldless enum its bitwise operators.
///
/// Upstream needs this because a C++ `enum class` has no `|`, `&` or `~`.
/// Rust's is the same problem with the same answer: the macro implements
/// `BitOr`, `BitAnd`, `BitXor` and `Not` over the discriminants, plus
/// comparison against a bare `i32`, so `Flags::A | Flags::B` type-checks.
///
/// The result of combining two variants is an `i32`, not the enum — the
/// combination usually is not a declared variant, and returning one would be a
/// lie. Upstream's C++ version returns the enum type and has exactly that
/// problem.
#[macro_export]
macro_rules! cv_enum_flags {
    ($ty:ty) => {
        impl std::ops::BitOr for $ty {
            type Output = i32;
            fn bitor(self, rhs: Self) -> i32 {
                (self as i32) | (rhs as i32)
            }
        }
        impl std::ops::BitOr<i32> for $ty {
            type Output = i32;
            fn bitor(self, rhs: i32) -> i32 {
                (self as i32) | rhs
            }
        }
        impl std::ops::BitAnd for $ty {
            type Output = i32;
            fn bitand(self, rhs: Self) -> i32 {
                (self as i32) & (rhs as i32)
            }
        }
        impl std::ops::BitAnd<i32> for $ty {
            type Output = i32;
            fn bitand(self, rhs: i32) -> i32 {
                (self as i32) & rhs
            }
        }
        impl std::ops::BitXor for $ty {
            type Output = i32;
            fn bitxor(self, rhs: Self) -> i32 {
                (self as i32) ^ (rhs as i32)
            }
        }
        impl std::ops::Not for $ty {
            type Output = i32;
            fn not(self) -> i32 {
                !(self as i32)
            }
        }
        impl PartialEq<i32> for $ty {
            fn eq(&self, other: &i32) -> bool {
                (*self as i32) == *other
            }
        }
    };
}

/// `CV_ENUM_CLASS_EXPOSE(EnumType, ...)` — re-export an `enum class`'s members
/// into the enclosing scope.
///
/// Rust reaches this with `pub use Type::*;`, which the macro emits.
#[macro_export]
macro_rules! cv_enum_class_expose {
    ($ty:ident) => {
        pub use $ty::*;
    };
}

/// `CV_IMPL_PROPERTY(type, name, member)` — generate a getter/setter pair.
///
/// Upstream's naming is `getName()` / `setName(value)`; that is reproduced
/// rather than Rust-ified, because the point of the macro is to match what the
/// C++ side calls.
#[macro_export]
macro_rules! cv_impl_property {
    ($ty:ty, $get:ident, $set:ident, $member:ident) => {
        /// Generated accessor.
        #[must_use]
        pub fn $get(&self) -> $ty {
            self.$member
        }
        /// Generated mutator.
        pub fn $set(&mut self, value: $ty) {
            self.$member = value;
        }
    };
}

/// `CV_IMPL_PROPERTY_S(type, name, member)` — as `cv_impl_property`, for a
/// type passed by const reference.
///
/// Rust's borrow rules make the by-reference form the natural one for anything
/// that is not `Copy`, so the getter returns a reference.
#[macro_export]
macro_rules! cv_impl_property_s {
    ($ty:ty, $get:ident, $set:ident, $member:ident) => {
        /// Generated accessor.
        #[must_use]
        pub fn $get(&self) -> &$ty {
            &self.$member
        }
        /// Generated mutator.
        pub fn $set(&mut self, value: $ty) {
            self.$member = value;
        }
    };
}

/// `CV_IMPL_PROPERTY_RO(type, name, member)` — a read-only property.
#[macro_export]
macro_rules! cv_impl_property_ro {
    ($ty:ty, $get:ident, $member:ident) => {
        /// Generated read-only accessor.
        #[must_use]
        pub fn $get(&self) -> $ty {
            self.$member
        }
    };
}

/// `CV_HELP_IMPL_PROPERTY(...)` — the expansion helper the three above share.
#[macro_export]
macro_rules! cv_help_impl_property {
    ($ty:ty, $get:ident, $set:ident, $member:ident) => {
        $crate::cv_impl_property!($ty, $get, $set, $member);
    };
}

/// `CV_WRAP_PROPERTY(type, name, internal_name, internal_obj)` — forward a
/// property to a member object's own accessors.
#[macro_export]
macro_rules! cv_wrap_property {
    ($ty:ty, $get:ident, $set:ident, $inner_get:ident, $inner_set:ident, $obj:ident) => {
        /// Generated forwarding accessor.
        #[must_use]
        pub fn $get(&self) -> $ty {
            self.$obj.$inner_get()
        }
        /// Generated forwarding mutator.
        pub fn $set(&mut self, value: $ty) {
            self.$obj.$inner_set(value);
        }
    };
}

/// `CV_WRAP_PROPERTY_S(...)` — [`cv_wrap_property`] for a by-reference type.
#[macro_export]
macro_rules! cv_wrap_property_s {
    ($ty:ty, $get:ident, $set:ident, $inner_get:ident, $inner_set:ident, $obj:ident) => {
        $crate::cv_wrap_property!($ty, $get, $set, $inner_get, $inner_set, $obj);
    };
}

/// `CV_WRAP_SAME_PROPERTY(type, name, internal_obj)` — forward under the same name.
#[macro_export]
macro_rules! cv_wrap_same_property {
    ($ty:ty, $get:ident, $set:ident, $obj:ident) => {
        $crate::cv_wrap_property!($ty, $get, $set, $get, $set, $obj);
    };
}

/// `CV_WRAP_SAME_PROPERTY_S(...)` — the by-reference form.
#[macro_export]
macro_rules! cv_wrap_same_property_s {
    ($ty:ty, $get:ident, $set:ident, $obj:ident) => {
        $crate::cv_wrap_same_property!($ty, $get, $set, $obj);
    };
}

/// `CV_HELP_WRAP_PROPERTY(...)` — the expansion helper the wrappers share.
#[macro_export]
macro_rules! cv_help_wrap_property {
    ($ty:ty, $get:ident, $set:ident, $inner_get:ident, $inner_set:ident, $obj:ident) => {
        $crate::cv_wrap_property!($ty, $get, $set, $inner_get, $inner_set, $obj);
    };
}

// `CV_Assert_1` … `CV_Assert_10` exist upstream only because C has no variadic
// macro recursion: each checks its first condition and hands the rest to the
// next one down. Rust's repetition needs none of that, so each simply forwards
// to `cv_assert_n!`. They are written out rather than generated because a
// `macro_rules!` that emits a `macro_rules!` needs the unstable `$$` escape.

/// `CV_Assert_1(...)` — the 1-condition form; see [`crate::cv_assert_n`].
#[macro_export]
macro_rules! cv_assert_1 {
    ($($cond:expr),+ $(,)?) => {
        $crate::cv_assert_n!($($cond),+)
    };
}

/// `CV_Assert_2(...)` — the 2-condition form; see [`crate::cv_assert_n`].
#[macro_export]
macro_rules! cv_assert_2 {
    ($($cond:expr),+ $(,)?) => {
        $crate::cv_assert_n!($($cond),+)
    };
}

/// `CV_Assert_3(...)` — the 3-condition form; see [`crate::cv_assert_n`].
#[macro_export]
macro_rules! cv_assert_3 {
    ($($cond:expr),+ $(,)?) => {
        $crate::cv_assert_n!($($cond),+)
    };
}

/// `CV_Assert_4(...)` — the 4-condition form; see [`crate::cv_assert_n`].
#[macro_export]
macro_rules! cv_assert_4 {
    ($($cond:expr),+ $(,)?) => {
        $crate::cv_assert_n!($($cond),+)
    };
}

/// `CV_Assert_5(...)` — the 5-condition form; see [`crate::cv_assert_n`].
#[macro_export]
macro_rules! cv_assert_5 {
    ($($cond:expr),+ $(,)?) => {
        $crate::cv_assert_n!($($cond),+)
    };
}

/// `CV_Assert_6(...)` — the 6-condition form; see [`crate::cv_assert_n`].
#[macro_export]
macro_rules! cv_assert_6 {
    ($($cond:expr),+ $(,)?) => {
        $crate::cv_assert_n!($($cond),+)
    };
}

/// `CV_Assert_7(...)` — the 7-condition form; see [`crate::cv_assert_n`].
#[macro_export]
macro_rules! cv_assert_7 {
    ($($cond:expr),+ $(,)?) => {
        $crate::cv_assert_n!($($cond),+)
    };
}

/// `CV_Assert_8(...)` — the 8-condition form; see [`crate::cv_assert_n`].
#[macro_export]
macro_rules! cv_assert_8 {
    ($($cond:expr),+ $(,)?) => {
        $crate::cv_assert_n!($($cond),+)
    };
}

/// `CV_Assert_9(...)` — the 9-condition form; see [`crate::cv_assert_n`].
#[macro_export]
macro_rules! cv_assert_9 {
    ($($cond:expr),+ $(,)?) => {
        $crate::cv_assert_n!($($cond),+)
    };
}

/// `CV_Assert_10(...)` — the 10-condition form; see [`crate::cv_assert_n`].
#[macro_export]
macro_rules! cv_assert_10 {
    ($($cond:expr),+ $(,)?) => {
        $crate::cv_assert_n!($($cond),+)
    };
}

/// `CV_AssertTerminate` — abort instead of unwinding on a failed assertion.
///
/// Upstream calls `std::terminate`; the Rust equivalent is
/// [`std::process::abort`], which this macro invokes after printing the
/// condition. Distinct from [`crate::cv_assert`], which returns an `Err`.
#[macro_export]
macro_rules! cv_assert_terminate {
    ($cond:expr) => {
        if !($cond) {
            eprintln!("Assertion failed (terminating): {}", stringify!($cond));
            std::process::abort();
        }
    };
}

/// `CV_SUPPRESS_DEPRECATED_START` — Rust spells this `#[allow(deprecated)]`.
///
/// An attribute cannot be opened and closed across statements the way the C++
/// pragma pair is, so the spelling is given rather than generated.
pub const CV_SUPPRESS_DEPRECATED_START: &str = "#[allow(deprecated)]";
/// `CV_SUPPRESS_DEPRECATED_END` — the attribute's scope ends with its item.
pub const CV_SUPPRESS_DEPRECATED_END: &str = "";

/// `CV_DO_PRAGMA(x)` — Rust has no `_Pragma`; the nearest is an attribute.
pub const CV_DO_PRAGMA: &str = "";
/// `CV__DEBUG_NS_BEGIN` — opens a debug-only inline namespace. No Rust analogue.
pub const CV_DEBUG_NS_BEGIN: &str = "";
/// `CV__DEBUG_NS_END` — closes it.
pub const CV_DEBUG_NS_END: &str = "";
/// `CV_TRACE_NS` — the namespace upstream's tracing lives in; here
/// `crate::trace`.
pub const CV_TRACE_NS: &str = "cv_core::trace";

/// Binding-generator annotations. Each expands to nothing in C++ and has no
/// Rust counterpart, since there is no generator reading them.
///
/// They are named so that a translation can account for every annotation on a
/// declaration rather than silently dropping some.
pub mod annotations {
    /// `CV_WRAP` — expose this member to the bindings.
    pub const CV_WRAP: &str = "";
    /// `CV_WRAP_AS(synonym)` — expose it under a different name.
    pub const CV_WRAP_AS: &str = "";
    /// `CV_WRAP_DEFAULT(value)` — supply a default the C++ signature lacks.
    pub const CV_WRAP_DEFAULT: &str = "";
    /// `CV_WRAP_PHANTOM(signature)` — declare a binding-only overload.
    pub const CV_WRAP_PHANTOM: &str = "";
    /// `CV_WRAP_MAPPABLE(mappable)` — treat the type as a mapping.
    pub const CV_WRAP_MAPPABLE: &str = "";
    /// `CV_WRAP_FILE_PATH` — mark an argument as a path, for `os.PathLike`.
    pub const CV_WRAP_FILE_PATH: &str = "";
    /// `CV_PROP` — expose a read-only property.
    pub const CV_PROP: &str = "";
    /// `CV_PROP_RW` — expose a read-write property.
    pub const CV_PROP_RW: &str = "";
    /// `CV_IN_OUT` — the argument is read and written.
    pub const CV_IN_OUT: &str = "";
    /// `CV_OUT` — the argument is written only.
    pub const CV_OUT: &str = "";
    /// `CV_UNUSED(x)` — Rust spells this `let _ = x;` or a leading underscore.
    pub const CV_UNUSED: &str = "let _ = x;";
    /// `CV_EXPORTS_W` — export the class to the bindings.
    pub const CV_EXPORTS_W: &str = "";
    /// `CV_EXPORTS_AS(synonym)` — export it under a different name.
    pub const CV_EXPORTS_AS: &str = "";
    /// `CV_EXPORTS_W_SIMPLE` — export a plain-data struct by value.
    pub const CV_EXPORTS_W_SIMPLE: &str = "";
    /// `CV_EXPORTS_W_MAP` — export the class as a dictionary.
    pub const CV_EXPORTS_W_MAP: &str = "";
    /// `CV_EXPORTS_W_PARAMS` — export it as a named-argument bundle.
    pub const CV_EXPORTS_W_PARAMS: &str = "";
    /// `CV_CREATE_MEMBER_CHECK(member)` — SFINAE member detection. Rust reaches
    /// the same end with a trait bound, which needs no generator.
    pub const CV_CREATE_MEMBER_CHECK: &str = "";
}

#[cfg(test)]
mod tests {
    use crate::error::Result;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum Mode {
        None = 0,
        Read = 1,
        Write = 2,
    }
    cv_enum_flags!(Mode);

    #[test]
    fn enum_flags_gains_the_bitwise_operators() {
        assert_eq!(Mode::Read | Mode::Write, 3);
        assert_eq!(Mode::Read & Mode::Write, 0);
        assert_eq!(Mode::Read ^ Mode::Write, 3);
        assert_eq!(!Mode::None, -1);
        // A combination is an i32, because it is not a declared variant.
        let combined = Mode::Read | Mode::Write;
        assert_eq!(combined & Mode::Read as i32, 1);
        // And a variant compares against a bare integer.
        assert!(Mode::Write == 2);
    }

    struct Widget {
        size: i32,
        name: String,
    }

    impl Widget {
        cv_impl_property!(i32, get_size, set_size, size);
        cv_impl_property_s!(String, get_name, set_name, name);
        cv_impl_property_ro!(i32, size_ro, size);
    }

    struct Outer {
        inner: Widget,
    }

    impl Outer {
        cv_wrap_same_property!(i32, get_size, set_size, inner);
    }

    #[test]
    fn the_property_macros_generate_working_accessors() {
        let mut w = Widget { size: 3, name: "a".into() };
        assert_eq!(w.get_size(), 3);
        w.set_size(7);
        assert_eq!((w.get_size(), w.size_ro()), (7, 7));
        assert_eq!(w.get_name(), "a");
        w.set_name("b".into());
        assert_eq!(w.get_name(), "b");

        // The wrapping form forwards to the member's own accessors.
        let mut o = Outer { inner: Widget { size: 1, name: String::new() } };
        assert_eq!(o.get_size(), 1);
        o.set_size(9);
        assert_eq!(o.inner.size, 9);
    }

    #[test]
    fn the_fixed_arity_asserts_behave_like_the_variadic_one() {
        fn three(a: i32, b: i32, c: i32) -> Result<()> {
            cv_assert_3!(a > 0, b > 0, c > 0);
            Ok(())
        }
        fn one(a: i32) -> Result<()> {
            cv_assert_1!(a > 0);
            Ok(())
        }
        fn ten(v: &[i32]) -> Result<()> {
            cv_assert_10!(
                v[0] > 0, v[1] > 0, v[2] > 0, v[3] > 0, v[4] > 0,
                v[5] > 0, v[6] > 0, v[7] > 0, v[8] > 0, v[9] > 0
            );
            Ok(())
        }
        assert!(three(1, 2, 3).is_ok());
        assert!(three(1, -1, 3).unwrap_err().msg.contains("b > 0"));
        assert!(one(1).is_ok() && one(-1).is_err());
        assert!(ten(&[1; 10]).is_ok());
        assert!(ten(&[1, 1, 1, 1, 1, 1, 1, 1, 1, -1]).is_err());
    }

    #[test]
    fn the_annotation_markers_are_declared_and_empty() {
        use super::annotations::*;
        // Every one expands to nothing in C++; the point is that they exist to
        // be named, not that they carry a value.
        for m in [CV_WRAP, CV_PROP, CV_PROP_RW, CV_IN_OUT, CV_OUT, CV_EXPORTS_W, CV_EXPORTS_AS] {
            assert!(m.is_empty());
        }
        assert_eq!(super::CV_SUPPRESS_DEPRECATED_START, "#[allow(deprecated)]");
        assert_eq!(super::CV_TRACE_NS, "cv_core::trace");
    }
}
