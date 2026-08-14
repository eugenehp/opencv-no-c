//! `cvRound` and `saturate_cast`, matching `core/include/opencv2/core/saturate.hpp`.
//!
//! Rounding parity matters more than it looks: upstream `cvRound` lowers to
//! `_mm_cvtss_si32` / `lrintf` / `vcvtn_s32_f32`, all of which round half to
//! **even** under the default FP environment. Using Rust's `f32::round` (half
//! away from zero) would put every `.5` boundary pixel one step off.

use crate::types::Depth;

/// `cvRound(double)` — round half to even.
#[inline]
#[must_use]
pub fn cv_round(value: f64) -> i32 {
    value.round_ties_even() as i32
}

/// `cvRound(float)` — round half to even.
#[inline]
#[must_use]
pub fn cv_roundf(value: f32) -> i32 {
    value.round_ties_even() as i32
}

/// `cvFloor` — largest integer not greater than `value`.
#[inline]
#[must_use]
pub fn cv_floor(value: f64) -> i32 {
    value.floor() as i32
}

/// `cvCeil` — smallest integer not less than `value`.
#[inline]
#[must_use]
pub fn cv_ceil(value: f64) -> i32 {
    value.ceil() as i32
}

/// A primitive channel type that a [`crate::Mat`] can hold.
///
/// The `from_*` conversions are the `saturate_cast<T>` family: they clamp to the
/// destination range and round half-to-even for float sources.
pub trait Elem: Copy + PartialOrd + Send + Sync + 'static {
    /// The depth code this type occupies in a [`crate::MatType`].
    const DEPTH: Depth;
    /// Neutral zero value.
    const ZERO: Self;

    /// `saturate_cast<Self>(f64)`.
    fn from_f64(v: f64) -> Self;
    /// `saturate_cast<Self>(i64)` — used by integer accumulation paths.
    fn from_i64(v: i64) -> Self;
    /// Widen to `f64` losslessly (all supported depths fit).
    fn to_f64(self) -> f64;
    /// Widen to `i64`; float depths truncate toward zero as C++ casts do.
    fn to_i64(self) -> i64;
}

/// Clamp an `f64` into an integral range with `cvRound` semantics.
#[inline]
fn round_clamp(v: f64, lo: f64, hi: f64) -> f64 {
    // NaN must land on 0, matching the C++ behaviour where the comparison
    // chain fails through to the initial value of an int register. OpenCV's
    // SIMD paths produce 0 for NaN on both SSE and NEON.
    if v.is_nan() {
        return 0.0;
    }
    let r = v.round_ties_even();
    if r < lo {
        lo
    } else if r > hi {
        hi
    } else {
        r
    }
}

macro_rules! impl_int_elem {
    ($t:ty, $depth:ident) => {
        impl Elem for $t {
            const DEPTH: Depth = Depth::$depth;
            const ZERO: Self = 0;

            #[inline]
            fn from_f64(v: f64) -> Self {
                round_clamp(v, <$t>::MIN as f64, <$t>::MAX as f64) as $t
            }

            #[inline]
            fn from_i64(v: i64) -> Self {
                v.clamp(<$t>::MIN as i64, <$t>::MAX as i64) as $t
            }

            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }

            #[inline]
            fn to_i64(self) -> i64 {
                self as i64
            }
        }
    };
}

impl_int_elem!(u8, U8);
impl_int_elem!(i8, S8);
impl_int_elem!(u16, U16);
impl_int_elem!(i16, S16);
impl_int_elem!(i32, S32);

impl Elem for f32 {
    const DEPTH: Depth = Depth::F32;
    const ZERO: Self = 0.0;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }

    #[inline]
    fn from_i64(v: i64) -> Self {
        v as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn to_i64(self) -> i64 {
        self as i64
    }
}

impl Elem for f64 {
    const DEPTH: Depth = Depth::F64;
    const ZERO: Self = 0.0;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }

    #[inline]
    fn from_i64(v: i64) -> Self {
        v as f64
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn to_i64(self) -> i64 {
        self as i64
    }
}

/// Free-function spelling of `saturate_cast<D>(src)`.
#[inline]
#[must_use]
pub fn saturate_cast<D: Elem>(v: f64) -> D {
    D::from_f64(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_is_half_to_even() {
        // The cases that separate round-half-even from round-half-away.
        assert_eq!(cv_round(0.5), 0);
        assert_eq!(cv_round(1.5), 2);
        assert_eq!(cv_round(2.5), 2);
        assert_eq!(cv_round(3.5), 4);
        assert_eq!(cv_round(-0.5), 0);
        assert_eq!(cv_round(-1.5), -2);
        assert_eq!(cv_round(-2.5), -2);
        assert_eq!(cv_round(2.4), 2);
        assert_eq!(cv_round(2.6), 3);
    }

    #[test]
    fn saturation_clamps_at_type_bounds() {
        assert_eq!(u8::from_f64(300.0), 255);
        assert_eq!(u8::from_f64(-4.0), 0);
        assert_eq!(u8::from_f64(254.5), 254); // ties-to-even
        assert_eq!(u8::from_f64(253.5), 254);
        assert_eq!(i8::from_f64(-200.0), -128);
        assert_eq!(i16::from_f64(40000.0), 32767);
        assert_eq!(u16::from_f64(-1.0), 0);
        assert_eq!(i32::from_f64(3e18), i32::MAX);
    }

    #[test]
    fn nan_saturates_to_zero() {
        assert_eq!(u8::from_f64(f64::NAN), 0);
        assert_eq!(i32::from_f64(f64::NAN), 0);
    }

    #[test]
    fn infinities_clamp() {
        assert_eq!(u8::from_f64(f64::INFINITY), 255);
        assert_eq!(u8::from_f64(f64::NEG_INFINITY), 0);
    }
}
