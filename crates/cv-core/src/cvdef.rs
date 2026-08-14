//! The scalar half of `modules/core/include/opencv2/core/cvdef.h`.
//!
//! Upstream's `cvdef.h` is mostly compiler plumbing — export attributes,
//! pragma wrappers, SIMD dispatch — none of which has a Rust counterpart. What
//! does port is the handful of numeric constants and small helpers that ported
//! algorithms actually read, so they live here rather than being re-derived at
//! each call site. The type-encoding half of the header is in [`crate::types`].
//!
//! `CV_PI` is spelled to upstream's full 31 digits. That matters: the literal
//! rounds to a `double` one ulp away from [`std::f64::consts::PI`] on no
//! platform, but writing it out keeps the provenance checkable rather than
//! assumed.

/// `CV_PI`.
#[allow(clippy::approx_constant)]
pub const CV_PI: f64 = 3.1415926535897932384626433832795;
/// `CV_PI_F` — `CV_PI` narrowed to `f32`.
///
/// Upstream defines this as `((float)CV_PI)` except on CUDA device code, where
/// it is the literal `3.14159265f`. The two agree to the last bit, so there is
/// one constant here.
#[allow(clippy::approx_constant)]
pub const CV_PI_F: f32 = std::f32::consts::PI;
/// `CV_2PI`.
#[allow(clippy::approx_constant)]
pub const CV_2PI: f64 = 6.283185307179586476925286766559;
/// `CV_LOG2` — the natural logarithm of 2.
#[allow(clippy::approx_constant)]
pub const CV_LOG2: f64 = 0.69314718055994530941723212145818;

/// `CV_VERSION_MAJOR`.
pub const CV_VERSION_MAJOR: i32 = 4;
/// `CV_VERSION_MINOR`.
pub const CV_VERSION_MINOR: i32 = 15;
/// `CV_VERSION_REVISION`.
pub const CV_VERSION_REVISION: i32 = 0;
/// `CV_VERSION_STATUS`.
pub const CV_VERSION_STATUS: &str = "-dev";

/// `CV_MAJOR_VERSION` — upstream's older alias for [`CV_VERSION_MAJOR`].
pub const CV_MAJOR_VERSION: i32 = CV_VERSION_MAJOR;
/// `CV_MINOR_VERSION` — upstream's older alias for [`CV_VERSION_MINOR`].
pub const CV_MINOR_VERSION: i32 = CV_VERSION_MINOR;
/// `CV_SUBMINOR_VERSION` — upstream's older alias for [`CV_VERSION_REVISION`].
pub const CV_SUBMINOR_VERSION: i32 = CV_VERSION_REVISION;

/// `CV_CMP(a, b)` — the three-way sign of `a - b`, without subtracting.
///
/// Upstream writes `((a) > (b)) - ((a) < (b))` precisely so it never overflows
/// for integer arguments near the extremes; the same reasoning applies here.
#[inline]
#[must_use]
pub fn cv_cmp<T: PartialOrd>(a: T, b: T) -> i32 {
    i32::from(a > b) - i32::from(a < b)
}

/// `CV_SIGN(a)`.
#[inline]
#[must_use]
pub fn cv_sign<T: PartialOrd + Default>(a: T) -> i32 {
    cv_cmp(a, T::default())
}

/// `CV_IMIN(a, b)` — branchless integer minimum.
#[inline]
#[must_use]
pub const fn cv_imin(a: i32, b: i32) -> i32 {
    if a < b {
        a
    } else {
        b
    }
}

/// `CV_IMAX(a, b)` — branchless integer maximum.
#[inline]
#[must_use]
pub const fn cv_imax(a: i32, b: i32) -> i32 {
    if a > b {
        a
    } else {
        b
    }
}

/// `CV_FOURCC_MACRO(c1, c2, c3, c4)` — pack four characters into a codec tag.
///
/// The same four bytes `VideoWriter` takes; little-endian, so `'M','J','P','G'`
/// packs with `M` in the low byte.
#[inline]
#[must_use]
pub const fn cv_fourcc_macro(c1: u8, c2: u8, c3: u8, c4: u8) -> i32 {
    (c1 as i32) + ((c2 as i32) << 8) + ((c3 as i32) << 16) + ((c4 as i32) << 24)
}

/// `CV_STRONG_ALIGNMENT` — does the target fault on an unaligned load?
///
/// Zero on x86 and AArch64, which both permit unaligned access. It is `1` only
/// on targets where upstream must copy through an aligned buffer; nothing here
/// does unaligned loads at all, so it is informational.
pub const CV_STRONG_ALIGNMENT: i32 = 0;

/// `CV_ND` — marks an argument that should be parsed as N-dimensional.
///
/// A binding-generator annotation upstream, expanding to nothing. Declared as
/// the empty string so a translation has something to name.
pub const CV_ND: &str = "";

/// `CV_ALWAYS_INLINE` — Rust spells this `#[inline(always)]`.
///
/// A marker constant rather than a macro: an attribute cannot be produced by a
/// `macro_rules!` in the position C++ uses this one.
pub const CV_ALWAYS_INLINE: &str = "#[inline(always)]";

/// `CV_TOGGLE_FLT(x)` — map a float's bits so integer order matches float order.
///
/// IEEE-754 floats already compare correctly as signed integers when both are
/// positive, but a negative float has its magnitude bits ascending where the
/// value descends. Flipping the low 31 bits of a negative pattern repairs
/// that, which is what lets `patchNaNs` and the float sorts work on the
/// integer representation. Its own inverse.
#[inline]
#[must_use]
pub const fn cv_toggle_flt(x: u32) -> u32 {
    x ^ if (x as i32) < 0 { 0x7fff_ffff } else { 0 }
}

/// `CV_TOGGLE_DBL(x)` — the `f64` form of [`cv_toggle_flt`].
#[inline]
#[must_use]
pub const fn cv_toggle_dbl(x: u64) -> u64 {
    x ^ if (x as i64) < 0 { 0x7fff_ffff_ffff_ffff } else { 0 }
}

/// `CV_DESCALE(x, n)` — divide by `2^n`, rounding to nearest.
///
/// The fixed-point workhorse of the colour conversions: adding half the divisor
/// before shifting is what makes the shift round rather than truncate.
#[inline]
#[must_use]
pub const fn cv_descale(x: i32, n: i32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// Fixed-point weights and shifts shared by the colour conversions.
///
/// Upstream declares these in an unnamed enum reached from several headers.
/// They are here rather than in `cv-imgproc` because that is where the include
/// tree puts them, and because `CV_DESCALE` above is meaningless without the
/// shift counts that go with it.
pub mod color {
    /// `yuv_shift` — fractional bits of the RGB↔YUV integer coefficients.
    pub const YUV_SHIFT: i32 = 14;
    /// `xyz_shift` — fractional bits of the RGB↔XYZ integer coefficients.
    pub const XYZ_SHIFT: i32 = 12;
    /// `gray_shift` — fractional bits of the 15-bit greyscale weights.
    pub const GRAY_SHIFT: i32 = 15;
    /// `lab_shift` — the same as [`XYZ_SHIFT`].
    pub const LAB_SHIFT: i32 = XYZ_SHIFT;
    /// `gamma_shift` — extra fractional bits of the 8-bit gamma table index.
    pub const GAMMA_SHIFT: i32 = 3;
    /// `lab_shift2` — [`LAB_SHIFT`] plus [`GAMMA_SHIFT`].
    pub const LAB_SHIFT2: i32 = LAB_SHIFT + GAMMA_SHIFT;

    /// `R2Y` — the red weight of luma, at [`YUV_SHIFT`] fractional bits.
    pub const R2Y: i32 = 4899;
    /// `G2Y` — the green weight of luma.
    pub const G2Y: i32 = 9617;
    /// `B2Y` — the blue weight of luma.
    pub const B2Y: i32 = 1868;

    /// `RY15` — the red weight at 15 bits: `R2YF * 32768 + 0.5`.
    pub const RY15: i32 = 9798;
    /// `GY15` — the green weight at 15 bits.
    pub const GY15: i32 = 19235;
    /// `BY15` — the blue weight at 15 bits.
    pub const BY15: i32 = 3735;

    /// `BLOCK_SIZE` — rows a colour kernel processes per tile.
    pub const BLOCK_SIZE: i32 = 256;
    /// `LAB_CBRT_TAB_SIZE` — entries in the float cube-root table.
    pub const LAB_CBRT_TAB_SIZE: i32 = 1024;
    /// `GAMMA_TAB_SIZE` — entries in the float sRGB gamma table.
    pub const GAMMA_TAB_SIZE: i32 = 1024;
    /// `LAB_CBRT_TAB_SIZE_B` — entries in the 8-bit cube-root table.
    pub const LAB_CBRT_TAB_SIZE_B: i32 = 256 * 3 / 2 * (1 << GAMMA_SHIFT);
}

/// `CV_SWAP(a, b, t)` — exchange two values.
///
/// Upstream needs the explicit temporary because it is a C macro; Rust has
/// [`std::mem::swap`], so the third argument disappears.
#[inline]
pub fn cv_swap<T>(a: &mut T, b: &mut T) {
    std::mem::swap(a, b);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_upstream() {
        // The literals above are what a C compiler would fold; check they land
        // on the same double the standard library derives.
        assert_eq!(CV_PI, std::f64::consts::PI);
        assert_eq!(CV_2PI, 2.0 * std::f64::consts::PI);
        assert_eq!(CV_LOG2, std::f64::consts::LN_2);
    }

    #[test]
    fn version_matches_the_tracked_release() {
        assert_eq!(
            format!("{CV_VERSION_MAJOR}.{CV_VERSION_MINOR}.{CV_VERSION_REVISION}{CV_VERSION_STATUS}"),
            crate::CV_VERSION
        );
        assert_eq!(CV_MAJOR_VERSION, 4);
        assert_eq!(CV_SUBMINOR_VERSION, 0);
    }

    #[test]
    fn cmp_and_sign_are_three_valued() {
        assert_eq!(cv_cmp(3, 5), -1);
        assert_eq!(cv_cmp(5, 5), 0);
        assert_eq!(cv_cmp(7, 5), 1);
        assert_eq!(cv_sign(-2.5_f64), -1);
        assert_eq!(cv_sign(0.0_f64), 0);
        assert_eq!(cv_sign(2.5_f64), 1);
    }

    #[test]
    fn imin_imax_survive_the_extremes() {
        // The upstream bit-twiddling form exists to avoid overflow here.
        assert_eq!(cv_imin(i32::MIN, i32::MAX), i32::MIN);
        assert_eq!(cv_imax(i32::MIN, i32::MAX), i32::MAX);
        assert_eq!(cv_imin(4, -4), -4);
        assert_eq!(cv_imax(4, -4), 4);
    }

    #[test]
    fn fourcc_packs_little_endian() {
        // 'MJPG' with M in the low byte, which is what VideoWriter expects.
        let mjpg = cv_fourcc_macro(b'M', b'J', b'P', b'G');
        assert_eq!(mjpg & 0xFF, i32::from(b'M'));
        assert_eq!((mjpg >> 24) & 0xFF, i32::from(b'G'));
        assert_eq!(cv_fourcc_macro(b'a', b'b', b'c', b'd'), 0x6463_6261);
        // High bytes are masked to 8 bits, so a wide value cannot bleed across.
        assert_eq!(cv_fourcc_macro(0xFF, 0, 0, 0), 0xFF);
    }

    #[test]
    fn swap_exchanges() {
        let (mut a, mut b) = (1, 2);
        cv_swap(&mut a, &mut b);
        assert_eq!((a, b), (2, 1));
    }
}
