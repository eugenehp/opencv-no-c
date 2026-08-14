//! `V_TypeTraits` — the scalar type table behind the universal intrinsics
//! (`modules/core/include/opencv2/core/hal/intrin.hpp`,
//! `modules/core/include/opencv2/core/hal/simd_utils.impl.hpp`).
//!
//! Upstream's SIMD kernels are written once, generically, over a scalar type.
//! To do that they need to know, for a given `T`: what integer has the same
//! width, what type an absolute value lands in, what type a product needs to
//! avoid overflow, what type a whole-vector sum needs. `V_TypeTraits<T>`
//! answers all of that, and `CV_INTRIN_DEF_TYPE_TRAITS` fills the table in.
//!
//! The vector types are not ported — this port runs scalar kernels for exact
//! parity, so there is no `v_uint8x16` to specialise over. **The table is**,
//! because it is not about vectors at all: it is the answer to "how wide must
//! my accumulator be?", which every kernel here decides too, currently by hand
//! at each site.
//!
//! # Example
//!
//! ```
//! use cv_core::intrin::VTypeTraits;
//!
//! // Summing bytes needs a u32 accumulator, and widening a product needs u16.
//! fn mean(xs: &[u8]) -> f64 {
//!     let total: <u8 as VTypeTraits>::SumType = xs.iter().map(|&x| u32::from(x)).sum();
//!     f64::from(total) / xs.len() as f64
//! }
//! assert_eq!(mean(&[1, 2, 3, 250]), 64.0);
//! ```
//!
//! # Reinterpretation
//!
//! [`VTypeTraits::reinterpret_int`] and [`VTypeTraits::reinterpret_from_int`]
//! are upstream's bit-cast pair, written there as a `union` because C++ had no
//! `bit_cast` when it was added. They are how `patchNaNs` and the float sorts
//! reach a float's bits; see [`crate::cvdef::cv_toggle_flt`] for what is
//! usually done with them next.

/// `cv::V_TypeTraits<T>` — the companion types of one scalar.
///
/// `WType` and `QType` are the "wide" and "quad" widenings: one and two
/// doublings. A type with nowhere left to double — anything 64-bit — sets them
/// to itself rather than to a fictional 128-bit type, which is upstream's
/// `void` in a form Rust can name.
pub trait VTypeTraits: Copy {
    /// `value_type` — the type itself.
    type ValueType: Copy;
    /// `int_type` — a signed integer of the same width, for bit access.
    type IntType: Copy;
    /// `uint_type` — an unsigned integer of the same width.
    type UintType: Copy;
    /// `abs_type` — what `|x|` lands in. Unsigned for the signed integers,
    /// since `|i8::MIN|` does not fit an `i8`.
    type AbsType: Copy;
    /// `w_type` — twice the width, for a product or a partial sum.
    type WType: Copy;
    /// `q_type` — four times the width, where such a type exists.
    type QType: Copy;
    /// `sum_type` — what a whole-vector sum lands in.
    type SumType: Copy;

    /// `V_TypeTraits::reinterpret_int(x)` — the value's bits as an integer.
    fn reinterpret_int(self) -> Self::IntType;
    /// `V_TypeTraits::reinterpret_from_int(x)` — the inverse.
    fn reinterpret_from_int(v: Self::IntType) -> Self;
}

/// `CV_INTRIN_DEF_TYPE_TRAITS(type, int, uint, abs, w, q, sum)` — one row of
/// the table, for a type that has a quad-width companion.
#[macro_export]
macro_rules! cv_intrin_def_type_traits {
    ($ty:ty, $int:ty, $uint:ty, $abs:ty, $w:ty, $q:ty, $sum:ty) => {
        impl $crate::intrin::VTypeTraits for $ty {
            type ValueType = $ty;
            type IntType = $int;
            type UintType = $uint;
            type AbsType = $abs;
            type WType = $w;
            type QType = $q;
            type SumType = $sum;

            fn reinterpret_int(self) -> $int {
                // Upstream punts through a union; `to_ne_bytes` is the same
                // reinterpretation with the endianness made explicit.
                <$int>::from_ne_bytes(self.to_ne_bytes())
            }
            fn reinterpret_from_int(v: $int) -> $ty {
                <$ty>::from_ne_bytes(v.to_ne_bytes())
            }
        }
    };
}

/// `CV_INTRIN_DEF_TYPE_TRAITS_NO_Q_TYPE(type, int, uint, abs, w, sum)` — a row
/// for a type with no quad-width companion, which sets `q_type` to itself.
#[macro_export]
macro_rules! cv_intrin_def_type_traits_no_q_type {
    ($ty:ty, $int:ty, $uint:ty, $abs:ty, $w:ty, $sum:ty) => {
        $crate::cv_intrin_def_type_traits!($ty, $int, $uint, $abs, $w, $ty, $sum);
    };
}

// The table, transcribed from `intrin.hpp` lines 164-173. Upstream's `w_type`
// for the 64-bit rows is `void`; Rust has no such type in this position, so
// they widen to themselves — the meaning is the same, "there is nowhere left
// to go".
cv_intrin_def_type_traits!(u8, i8, u8, u8, u16, u32, u32);
cv_intrin_def_type_traits!(i8, i8, u8, u8, i16, i32, i32);
cv_intrin_def_type_traits!(u16, i16, u16, u16, u32, u64, u32);
cv_intrin_def_type_traits!(i16, i16, u16, u16, i32, i64, i32);
cv_intrin_def_type_traits_no_q_type!(u32, i32, u32, u32, u64, u32);
cv_intrin_def_type_traits_no_q_type!(i32, i32, u32, u32, i64, i32);
cv_intrin_def_type_traits_no_q_type!(f32, i32, u32, f32, f64, f32);
cv_intrin_def_type_traits_no_q_type!(u64, i64, u64, u64, u64, u64);
cv_intrin_def_type_traits_no_q_type!(i64, i64, u64, u64, i64, i64);
cv_intrin_def_type_traits_no_q_type!(f64, i64, u64, f64, f64, f64);

/// `CV_DEF_REG_TRAITS(prefix, _reg, lane_type, suffix, _u_reg, _w_reg, _q_reg, ...)`
/// — the register-level companion of [`cv_intrin_def_type_traits`].
///
/// Upstream uses it to record, for each vector *register* type, its lane type
/// and its unsigned, wide and quad register counterparts. There are no
/// register types here, so this records only the lane type, which is the part
/// that survives without them.
#[macro_export]
macro_rules! cv_def_reg_traits {
    ($name:ident, $lane:ty) => {
        /// Generated register traits: the lane type of a vector register.
        pub type $name = $lane;
    };
}

/// `CV_INTRIN_DEF_TYPE2VEC_TRAITS(type, vec_type)` — map a scalar to the
/// vector register that holds it.
///
/// Every mapping is to the scalar itself: with no vector types, a "vector of
/// `T`" is one `T`. The macro exists so a translated kernel still names its
/// element type through the same indirection.
#[macro_export]
macro_rules! cv_intrin_def_type2vec_traits {
    ($name:ident, $ty:ty) => {
        /// Generated `Type2Vec_Traits`: the vector type holding this scalar.
        pub type $name = $ty;
    };
}

/// `CV_INTRIN_DEF_TYPE2VEC128_TRAITS(type, vec_type)` — the 128-bit form.
#[macro_export]
macro_rules! cv_intrin_def_type2vec128_traits {
    ($name:ident, $ty:ty) => {
        $crate::cv_intrin_def_type2vec_traits!($name, $ty);
    };
}

/// `CV_INTRIN_DEF_TYPE2VEC256_TRAITS(type, vec_type)` — the 256-bit form.
#[macro_export]
macro_rules! cv_intrin_def_type2vec256_traits {
    ($name:ident, $ty:ty) => {
        $crate::cv_intrin_def_type2vec_traits!($name, $ty);
    };
}

/// `CV_INTRIN_DEF_TYPE2VEC512_TRAITS(type, vec_type)` — the 512-bit form.
#[macro_export]
macro_rules! cv_intrin_def_type2vec512_traits {
    ($name:ident, $ty:ty) => {
        $crate::cv_intrin_def_type2vec_traits!($name, $ty);
    };
}

/// `CV__HAL_INTRIN_IMPL_BIN_OP(bin_op)` — implement an arithmetic operator
/// across a list of types.
///
/// Upstream generates `operator+`, `-`, `*`, `/` for every vector register
/// type from one definition. The Rust equivalent takes the trait, the method
/// and the type list, and emits one impl per type.
#[macro_export]
macro_rules! cv_hal_intrin_impl_bin_op {
    ($trait:ident, $method:ident, $op:tt, $($ty:ty),+ $(,)?) => {
        $(
            impl $trait for $ty {
                fn $method(self, rhs: Self) -> Self {
                    self $op rhs
                }
            }
        )+
    };
}

/// `CV__HAL_INTRIN_IMPL_BIT_OP(bit_op)` — the same, for a bitwise operator.
///
/// Separate upstream because the float types need the operation applied to
/// their bit pattern rather than their value; that split is preserved here by
/// routing through [`VTypeTraits::reinterpret_int`].
#[macro_export]
macro_rules! cv_hal_intrin_impl_bit_op {
    ($trait:ident, $method:ident, $op:tt, $($ty:ty),+ $(,)?) => {
        $(
            impl $trait for $ty {
                fn $method(self, rhs: Self) -> Self {
                    use $crate::intrin::VTypeTraits;
                    let (a, b) = (self.reinterpret_int(), rhs.reinterpret_int());
                    <$ty>::reinterpret_from_int(a $op b)
                }
            }
        )+
    };
}

/// `CV__HAL_INTRIN_EXPAND_WITH_INTEGER_TYPES(macro, ...)` — apply a macro to
/// every integer scalar.
#[macro_export]
macro_rules! cv_hal_intrin_expand_with_integer_types {
    // With leading arguments of its own, and without.
    ($mac:ident, $($arg:tt)+) => {
        $mac!($($arg)+, u8, i8, u16, i16, u32, i32, u64, i64)
    };
    ($mac:ident $(,)?) => {
        $mac!(u8, i8, u16, i16, u32, i32, u64, i64)
    };
}

/// `CV__HAL_INTRIN_EXPAND_WITH_FP_TYPES(macro, ...)` — apply it to the floats.
#[macro_export]
macro_rules! cv_hal_intrin_expand_with_fp_types {
    // With leading arguments of its own, and without.
    ($mac:ident, $($arg:tt)+) => {
        $mac!($($arg)+, f32, f64)
    };
    ($mac:ident $(,)?) => {
        $mac!(f32, f64)
    };
}

/// `CV__HAL_INTRIN_EXPAND_WITH_ALL_TYPES(macro, ...)` — apply it to both.
#[macro_export]
macro_rules! cv_hal_intrin_expand_with_all_types {
    // With leading arguments of its own, and without.
    ($mac:ident, $($arg:tt)+) => {
        $mac!($($arg)+, u8, i8, u16, i16, u32, i32, u64, i64, f32, f64)
    };
    ($mac:ident $(,)?) => {
        $mac!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64)
    };
}

/// `CV_CUDEV_IMPLEMENT_VEC_UNARY_OP(op, ...)` — implement a unary operator
/// element-wise over a fixed-size vector.
///
/// The CUDA device headers use these to give `uchar4` and friends their
/// arithmetic. `[T; N]` is the Rust equivalent of those types, so the macros
/// generate free functions over an array.
#[macro_export]
macro_rules! cv_cudev_implement_vec_unary_op {
    ($name:ident, $ty:ty, $n:expr, $op:tt) => {
        /// Generated element-wise unary operator.
        #[must_use]
        pub fn $name(a: [$ty; $n]) -> [$ty; $n] {
            let mut out = a;
            for v in &mut out {
                *v = $op *v;
            }
            out
        }
    };
}

/// `CV_CUDEV_IMPLEMENT_VEC_BINARY_OP(op, ...)` — the binary form.
#[macro_export]
macro_rules! cv_cudev_implement_vec_binary_op {
    ($name:ident, $ty:ty, $n:expr, $op:tt) => {
        /// Generated element-wise binary operator.
        #[must_use]
        #[allow(clippy::assign_op_pattern)]
        pub fn $name(a: [$ty; $n], b: [$ty; $n]) -> [$ty; $n] {
            let mut out = a;
            for (o, r) in out.iter_mut().zip(b) {
                *o = *o $op r;
            }
            out
        }
    };
}

/// `CV_CUDEV_IMPLEMENT_VEC_UNARY_FUNC(func, ...)` — apply a named function
/// element-wise.
#[macro_export]
macro_rules! cv_cudev_implement_vec_unary_func {
    ($name:ident, $ty:ty, $n:expr, $f:expr) => {
        /// Generated element-wise unary function.
        #[must_use]
        pub fn $name(a: [$ty; $n]) -> [$ty; $n] {
            let mut out = a;
            for v in &mut out {
                *v = $f(*v);
            }
            out
        }
    };
}

/// `CV_CUDEV_IMPLEMENT_VEC_BINARY_FUNC(func, ...)` — the binary form.
#[macro_export]
macro_rules! cv_cudev_implement_vec_binary_func {
    ($name:ident, $ty:ty, $n:expr, $f:expr) => {
        /// Generated element-wise binary function.
        #[must_use]
        pub fn $name(a: [$ty; $n], b: [$ty; $n]) -> [$ty; $n] {
            let mut out = a;
            for (o, r) in out.iter_mut().zip(b) {
                *o = $f(*o, r);
            }
            out
        }
    };
}

/// `CV_CUDEV_IMPLEMENT_SCALAR_BINARY_OP(op, ...)` — a vector against a scalar,
/// broadcast.
#[macro_export]
macro_rules! cv_cudev_implement_scalar_binary_op {
    ($name:ident, $ty:ty, $n:expr, $op:tt) => {
        /// Generated broadcast binary operator.
        #[must_use]
        #[allow(clippy::assign_op_pattern)]
        pub fn $name(a: [$ty; $n], s: $ty) -> [$ty; $n] {
            let mut out = a;
            for v in &mut out {
                *v = *v $op s;
            }
            out
        }
    };
}

/// `CV_CUDEV_IMPLEMENT_SCALAR_BINARY_FUNC(func, ...)` — the function form.
#[macro_export]
macro_rules! cv_cudev_implement_scalar_binary_func {
    ($name:ident, $ty:ty, $n:expr, $f:expr) => {
        /// Generated broadcast binary function.
        #[must_use]
        pub fn $name(a: [$ty; $n], s: $ty) -> [$ty; $n] {
            let mut out = a;
            for v in &mut out {
                *v = $f(*v, s);
            }
            out
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The widths are what upstream's table says, and they are the widths a
    /// kernel actually needs.
    #[test]
    fn the_traits_table_matches_upstream() {
        // uchar: sums into unsigned, widens to ushort, quads to unsigned.
        let w: <u8 as VTypeTraits>::WType = u16::from(200u8) * 2;
        assert_eq!(w, 400, "a u8 product must not overflow its w_type");
        let s: <u8 as VTypeTraits>::SumType = 255u32 * 1_000_000;
        assert_eq!(s, 255_000_000, "a u8 sum needs 32 bits");

        // schar's abs_type is *unsigned*, because |i8::MIN| does not fit i8.
        let a: <i8 as VTypeTraits>::AbsType = i8::MIN.unsigned_abs();
        assert_eq!(a, 128u8);

        // float widens to double and sums as float, per upstream's row.
        let fw: <f32 as VTypeTraits>::WType = 1.5f64;
        assert_eq!(fw, 1.5);
    }

    #[test]
    fn the_64_bit_rows_widen_to_themselves() {
        // Upstream writes `void` for these; there is nowhere left to double.
        let w: <u64 as VTypeTraits>::WType = u64::MAX;
        assert_eq!(w, u64::MAX);
        let d: <f64 as VTypeTraits>::WType = f64::MAX;
        assert_eq!(d, f64::MAX);
        // And a type with no quad companion sets q_type to itself.
        let q: <i32 as VTypeTraits>::QType = -7i32;
        assert_eq!(q, -7);
    }

    #[test]
    fn reinterpretation_round_trips_through_the_bits() {
        for v in [0.0f32, 1.0, -1.5, f32::MIN, f32::MAX, f32::EPSILON] {
            assert_eq!(f32::reinterpret_from_int(v.reinterpret_int()), v);
        }
        // And it really is the bit pattern, not a numeric conversion.
        assert_eq!(1.0f32.reinterpret_int(), 0x3f80_0000u32 as i32);
        assert_eq!((-1.0f64).reinterpret_int(), u64::from_ne_bytes((-1.0f64).to_ne_bytes()) as i64);
        assert_eq!(255u8.reinterpret_int(), -1i8);
    }

    // The op-generating macros, exercised the way upstream uses them.
    trait Add2 {
        fn add2(self, rhs: Self) -> Self;
    }
    cv_hal_intrin_impl_bin_op!(Add2, add2, +, u8, i16, f32);

    trait Xor2 {
        fn xor2(self, rhs: Self) -> Self;
    }
    cv_hal_intrin_impl_bit_op!(Xor2, xor2, ^, u8, i16, f32);

    #[test]
    fn the_op_expansion_macros_generate_working_impls() {
        assert_eq!(3u8.add2(4), 7);
        assert_eq!((-3i16).add2(4), 1);
        assert_eq!(1.5f32.add2(2.5), 4.0);

        assert_eq!(0b1100u8.xor2(0b1010), 0b0110);
        // On a float the operation applies to the bit pattern, so xor with
        // itself is +0.0 whatever the input was.
        assert_eq!(1.5f32.xor2(1.5), 0.0);
    }

    macro_rules! count_types {
        ($($ty:ty),+) => { [$(std::mem::size_of::<$ty>()),+].len() };
    }

    #[test]
    fn the_type_list_macros_expand_to_the_right_sets() {
        assert_eq!(cv_hal_intrin_expand_with_integer_types!(count_types), 8);
        assert_eq!(cv_hal_intrin_expand_with_fp_types!(count_types), 2);
        assert_eq!(cv_hal_intrin_expand_with_all_types!(count_types), 10);
    }

    cv_cudev_implement_vec_unary_op!(neg4, i32, 4, -);
    cv_cudev_implement_vec_binary_op!(add4, i32, 4, +);
    cv_cudev_implement_scalar_binary_op!(scale4, i32, 4, *);
    cv_cudev_implement_vec_unary_func!(abs4, i32, 4, i32::abs);
    cv_cudev_implement_vec_binary_func!(max4, i32, 4, i32::max);
    cv_cudev_implement_scalar_binary_func!(min4s, i32, 4, i32::min);

    #[test]
    fn the_cudev_macros_generate_element_wise_operations() {
        assert_eq!(neg4([1, -2, 3, -4]), [-1, 2, -3, 4]);
        assert_eq!(add4([1, 2, 3, 4], [10, 20, 30, 40]), [11, 22, 33, 44]);
        assert_eq!(scale4([1, 2, 3, 4], 3), [3, 6, 9, 12]);
        assert_eq!(abs4([1, -2, 3, -4]), [1, 2, 3, 4]);
        assert_eq!(max4([1, 9, 3, 9], [5, 2, 5, 2]), [5, 9, 5, 9]);
        assert_eq!(min4s([1, 9, 3, 9], 5), [1, 5, 3, 5]);
    }

    cv_intrin_def_type2vec_traits!(VUint8, u8);
    cv_def_reg_traits!(LaneU8, u8);

    #[test]
    fn the_vector_type_aliases_resolve_to_the_scalar() {
        // With no register types, a "vector of u8" is one u8.
        let v: VUint8 = 7;
        let l: LaneU8 = 7;
        assert_eq!(v, l);
    }
}
