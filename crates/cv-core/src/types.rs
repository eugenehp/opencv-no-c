//! Element-type encoding, bit-identical to `modules/core/include/opencv2/core/hal/interface.h`.
//!
//! A type is packed as `depth | ((channels - 1) << CV_CN_SHIFT)`, exactly as
//! `CV_MAKETYPE` does upstream, so a [`MatType`] round-trips through OpenCV's
//! integer representation without translation.

use crate::error::{Code, Error, Result};

/// `CV_CN_SHIFT` — bit position where the channel count starts.
pub const CV_CN_SHIFT: i32 = 3;
/// `CV_DEPTH_MAX` — one past the highest depth value.
pub const CV_DEPTH_MAX: i32 = 1 << CV_CN_SHIFT;
/// `CV_MAT_DEPTH_MASK` — mask isolating the depth bits.
pub const CV_MAT_DEPTH_MASK: i32 = CV_DEPTH_MAX - 1;
/// `CV_MAT_CN_MASK` — mask isolating the channel bits.
pub const CV_MAT_CN_MASK: i32 = (CV_CN_MAX - 1) << CV_CN_SHIFT;
/// `CV_CN_MAX` — maximum number of channels.
pub const CV_CN_MAX: i32 = 512;

/// `CV_8U` — 8-bit unsigned integer.
pub const CV_8U: i32 = 0;
/// `CV_8S` — 8-bit signed integer.
pub const CV_8S: i32 = 1;
/// `CV_16U` — 16-bit unsigned integer.
pub const CV_16U: i32 = 2;
/// `CV_16S` — 16-bit signed integer.
pub const CV_16S: i32 = 3;
/// `CV_32S` — 32-bit signed integer.
pub const CV_32S: i32 = 4;
/// `CV_32F` — 32-bit float.
pub const CV_32F: i32 = 5;
/// `CV_64F` — 64-bit float.
pub const CV_64F: i32 = 6;
/// `CV_16F` — 16-bit (half) float.
pub const CV_16F: i32 = 7;
/// `CV_USRTYPE1` — a deprecated spelling upstream keeps for source
/// compatibility. It used to mean "a user-defined type"; it now aliases
/// [`CV_16F`], which is what took its slot.
#[deprecated(note = "upstream deprecated this; use CV_16F")]
pub const CV_USRTYPE1: i32 = CV_16F;

/// `CV_MAT_DEPTH(flags)`.
#[inline]
#[must_use]
pub const fn cv_mat_depth(flags: i32) -> i32 {
    flags & CV_MAT_DEPTH_MASK
}

/// `CV_MAKETYPE(depth, cn)`.
#[inline]
#[must_use]
pub const fn cv_maketype(depth: i32, cn: i32) -> i32 {
    cv_mat_depth(depth) + ((cn - 1) << CV_CN_SHIFT)
}

/// `CV_MAT_CN(flags)`.
#[inline]
#[must_use]
pub const fn cv_mat_cn(flags: i32) -> i32 {
    ((flags & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1
}

/// `CV_MAKE_TYPE(depth, cn)` — upstream's alias for `CV_MAKETYPE`.
#[inline]
#[must_use]
pub const fn cv_make_type(depth: i32, cn: i32) -> i32 {
    cv_maketype(depth, cn)
}

/// `CV_MAT_TYPE_MASK` — the depth and channel bits together.
pub const CV_MAT_TYPE_MASK: i32 = CV_DEPTH_MAX * CV_CN_MAX - 1;

/// `CV_MAT_TYPE(flags)` — strip everything but depth and channel count.
#[inline]
#[must_use]
pub const fn cv_mat_type(flags: i32) -> i32 {
    flags & CV_MAT_TYPE_MASK
}

/// `CV_MAT_CONT_FLAG_SHIFT` — bit position of the continuity flag.
pub const CV_MAT_CONT_FLAG_SHIFT: i32 = 14;
/// `CV_MAT_CONT_FLAG` — set when rows are stored without inter-row padding.
pub const CV_MAT_CONT_FLAG: i32 = 1 << CV_MAT_CONT_FLAG_SHIFT;
/// `CV_SUBMAT_FLAG_SHIFT` — bit position of the submatrix flag.
pub const CV_SUBMAT_FLAG_SHIFT: i32 = 15;
/// `CV_SUBMAT_FLAG` — set when the header views part of a larger buffer.
pub const CV_SUBMAT_FLAG: i32 = 1 << CV_SUBMAT_FLAG_SHIFT;

/// `CV_IS_MAT_CONT(flags)`.
#[inline]
#[must_use]
pub const fn cv_is_mat_cont(flags: i32) -> bool {
    flags & CV_MAT_CONT_FLAG != 0
}

/// `CV_IS_CONT_MAT(flags)` — upstream's alias for `CV_IS_MAT_CONT`.
#[inline]
#[must_use]
pub const fn cv_is_cont_mat(flags: i32) -> bool {
    cv_is_mat_cont(flags)
}

/// `CV_IS_SUBMAT(flags)`.
#[inline]
#[must_use]
pub const fn cv_is_submat(flags: i32) -> bool {
    flags & CV_SUBMAT_FLAG != 0
}

/// `CV_MAX_DIM` — the most dimensions an array may have.
pub const CV_MAX_DIM: i32 = 32;

/// `CV_ELEM_SIZE1(type)` — bytes in one channel of one element.
///
/// Upstream packs the eight sizes into the nibbles of `0x28442211` and shifts;
/// keeping that form means a new depth cannot be added here without also
/// updating the constant, exactly as upstream.
#[inline]
#[must_use]
pub const fn cv_elem_size1(ty: i32) -> i32 {
    (0x2844_2211 >> (cv_mat_depth(ty) * 4)) & 15
}

/// `CV_ELEM_SIZE(type)` — bytes in one whole element.
#[inline]
#[must_use]
pub const fn cv_elem_size(ty: i32) -> i32 {
    cv_mat_cn(ty) * cv_elem_size1(ty)
}

/// Element depth — the scalar type of one channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i32)]
pub enum Depth {
    /// `CV_8U`
    U8 = CV_8U,
    /// `CV_8S`
    S8 = CV_8S,
    /// `CV_16U`
    U16 = CV_16U,
    /// `CV_16S`
    S16 = CV_16S,
    /// `CV_32S`
    S32 = CV_32S,
    /// `CV_32F`
    F32 = CV_32F,
    /// `CV_64F`
    F64 = CV_64F,
    /// `CV_16F`
    F16 = CV_16F,
}

impl Depth {
    /// This depth as a one-bit set, for testing against a `DEPTH_MASK_*`.
    ///
    /// Pairs with [`crate::mat::depth_mask`]: `DEPTH_MASK_FLT & d.mask() != 0`
    /// asks "is `d` a float depth?" in the form upstream's `CV_CheckDepth`
    /// guards are written.
    #[inline]
    #[must_use]
    pub const fn mask(self) -> i32 {
        1 << (self as i32)
    }

    /// Decode a depth from its OpenCV integer value.
    pub const fn from_i32(v: i32) -> Result<Self> {
        Ok(match v {
            CV_8U => Depth::U8,
            CV_8S => Depth::S8,
            CV_16U => Depth::U16,
            CV_16S => Depth::S16,
            CV_32S => Depth::S32,
            CV_32F => Depth::F32,
            CV_64F => Depth::F64,
            CV_16F => Depth::F16,
            _ => {
                return Err(Error {
                    code: Code::StsUnsupportedFormat,
                    msg: String::new(),
                })
            }
        })
    }

    /// Size in bytes of one channel value — `CV_ELEM_SIZE1`.
    #[inline]
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Depth::U8 | Depth::S8 => 1,
            Depth::U16 | Depth::S16 | Depth::F16 => 2,
            Depth::S32 | Depth::F32 => 4,
            Depth::F64 => 8,
        }
    }

    /// Whether this depth is a floating-point format.
    #[inline]
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Depth::F32 | Depth::F64 | Depth::F16)
    }

    /// Inclusive lower bound of the representable range, as `f64`.
    #[must_use]
    pub const fn min_value(self) -> f64 {
        match self {
            Depth::U8 | Depth::U16 => 0.0,
            Depth::S8 => -128.0,
            Depth::S16 => -32768.0,
            Depth::S32 => -2147483648.0,
            Depth::F32 | Depth::F64 | Depth::F16 => f64::NEG_INFINITY,
        }
    }

    /// Inclusive upper bound of the representable range, as `f64`.
    #[must_use]
    pub const fn max_value(self) -> f64 {
        match self {
            Depth::U8 => 255.0,
            Depth::S8 => 127.0,
            Depth::U16 => 65535.0,
            Depth::S16 => 32767.0,
            Depth::S32 => 2147483647.0,
            Depth::F32 | Depth::F64 | Depth::F16 => f64::INFINITY,
        }
    }

    /// The upstream spelling of this depth, e.g. `"CV_8U"`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Depth::U8 => "CV_8U",
            Depth::S8 => "CV_8S",
            Depth::U16 => "CV_16U",
            Depth::S16 => "CV_16S",
            Depth::S32 => "CV_32S",
            Depth::F32 => "CV_32F",
            Depth::F64 => "CV_64F",
            Depth::F16 => "CV_16F",
        }
    }
}

/// A packed element type: depth plus channel count.
///
/// The wire format is identical to OpenCV's `int` type codes, so
/// `MatType::new(Depth::U8, 3).to_i32() == CV_8UC3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatType(i32);

impl MatType {
    /// Build a type from a depth and channel count (1..=512).
    ///
    /// # Panics
    /// Panics if `channels` is outside `1..=CV_CN_MAX`.
    #[inline]
    #[must_use]
    pub const fn new(depth: Depth, channels: i32) -> Self {
        assert!(channels >= 1 && channels <= CV_CN_MAX, "channels out of range");
        MatType(cv_maketype(depth as i32, channels))
    }

    /// Decode from OpenCV's integer type code.
    pub fn from_i32(v: i32) -> Result<Self> {
        let cn = cv_mat_cn(v);
        let depth = Depth::from_i32(cv_mat_depth(v)).map_err(|_| {
            Error::new(Code::StsUnsupportedFormat, format!("unsupported depth in type {v}"))
        })?;
        if !(1..=CV_CN_MAX).contains(&cn) {
            return crate::cv_error!(Code::StsOutOfRange, "channel count {cn} out of range");
        }
        Ok(MatType::new(depth, cn))
    }

    /// The OpenCV integer type code.
    #[inline]
    #[must_use]
    pub const fn to_i32(self) -> i32 {
        self.0
    }

    /// Channel depth — `CV_MAT_DEPTH`.
    #[inline]
    #[must_use]
    pub const fn depth(self) -> Depth {
        // SAFETY-free: the constructor validated the depth, and the mask keeps
        // it in 0..8, so the match below is total.
        match cv_mat_depth(self.0) {
            CV_8U => Depth::U8,
            CV_8S => Depth::S8,
            CV_16U => Depth::U16,
            CV_16S => Depth::S16,
            CV_32S => Depth::S32,
            CV_32F => Depth::F32,
            CV_64F => Depth::F64,
            _ => Depth::F16,
        }
    }

    /// Channel count — `CV_MAT_CN`.
    #[inline]
    #[must_use]
    pub const fn channels(self) -> i32 {
        cv_mat_cn(self.0)
    }

    /// Bytes per element across all channels — `CV_ELEM_SIZE`.
    #[inline]
    #[must_use]
    pub const fn elem_size(self) -> usize {
        self.depth().size() * self.channels() as usize
    }

    /// Bytes per single channel value — `CV_ELEM_SIZE1`.
    #[inline]
    #[must_use]
    pub const fn elem_size1(self) -> usize {
        self.depth().size()
    }

    /// Same depth, different channel count.
    #[inline]
    #[must_use]
    pub const fn with_channels(self, channels: i32) -> Self {
        MatType::new(self.depth(), channels)
    }

    /// Same channel count, different depth.
    #[inline]
    #[must_use]
    pub const fn with_depth(self, depth: Depth) -> Self {
        MatType::new(depth, self.channels())
    }
}

impl core::fmt::Display for MatType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}C{}", self.depth().name(), self.channels())
    }
}

// The `CV_<depth>C<n>` constants are spelled out rather than macro-generated so
// that each one carries its own upstream claim where a reader — and the parity
// scanner — will see it.

/// `CV_8UC1`
pub const CV_8UC1: MatType = MatType::new(Depth::U8, 1);
/// `CV_8UC2`
pub const CV_8UC2: MatType = MatType::new(Depth::U8, 2);
/// `CV_8UC3`
pub const CV_8UC3: MatType = MatType::new(Depth::U8, 3);
/// `CV_8UC4`
pub const CV_8UC4: MatType = MatType::new(Depth::U8, 4);

/// `CV_8SC1`
pub const CV_8SC1: MatType = MatType::new(Depth::S8, 1);
/// `CV_8SC2`
pub const CV_8SC2: MatType = MatType::new(Depth::S8, 2);
/// `CV_8SC3`
pub const CV_8SC3: MatType = MatType::new(Depth::S8, 3);
/// `CV_8SC4`
pub const CV_8SC4: MatType = MatType::new(Depth::S8, 4);

/// `CV_16UC1`
pub const CV_16UC1: MatType = MatType::new(Depth::U16, 1);
/// `CV_16UC2`
pub const CV_16UC2: MatType = MatType::new(Depth::U16, 2);
/// `CV_16UC3`
pub const CV_16UC3: MatType = MatType::new(Depth::U16, 3);
/// `CV_16UC4`
pub const CV_16UC4: MatType = MatType::new(Depth::U16, 4);

/// `CV_16SC1`
pub const CV_16SC1: MatType = MatType::new(Depth::S16, 1);
/// `CV_16SC2`
pub const CV_16SC2: MatType = MatType::new(Depth::S16, 2);
/// `CV_16SC3`
pub const CV_16SC3: MatType = MatType::new(Depth::S16, 3);
/// `CV_16SC4`
pub const CV_16SC4: MatType = MatType::new(Depth::S16, 4);

/// `CV_32SC1`
pub const CV_32SC1: MatType = MatType::new(Depth::S32, 1);
/// `CV_32SC2`
pub const CV_32SC2: MatType = MatType::new(Depth::S32, 2);
/// `CV_32SC3`
pub const CV_32SC3: MatType = MatType::new(Depth::S32, 3);
/// `CV_32SC4`
pub const CV_32SC4: MatType = MatType::new(Depth::S32, 4);

/// `CV_32FC1`
pub const CV_32FC1: MatType = MatType::new(Depth::F32, 1);
/// `CV_32FC2`
pub const CV_32FC2: MatType = MatType::new(Depth::F32, 2);
/// `CV_32FC3`
pub const CV_32FC3: MatType = MatType::new(Depth::F32, 3);
/// `CV_32FC4`
pub const CV_32FC4: MatType = MatType::new(Depth::F32, 4);

/// `CV_64FC1`
pub const CV_64FC1: MatType = MatType::new(Depth::F64, 1);
/// `CV_64FC2`
pub const CV_64FC2: MatType = MatType::new(Depth::F64, 2);
/// `CV_64FC3`
pub const CV_64FC3: MatType = MatType::new(Depth::F64, 3);
/// `CV_64FC4`
pub const CV_64FC4: MatType = MatType::new(Depth::F64, 4);

/// `CV_16FC1`
pub const CV_16FC1: MatType = MatType::new(Depth::F16, 1);
/// `CV_16FC2`
pub const CV_16FC2: MatType = MatType::new(Depth::F16, 2);
/// `CV_16FC3`
pub const CV_16FC3: MatType = MatType::new(Depth::F16, 3);
/// `CV_16FC4`
pub const CV_16FC4: MatType = MatType::new(Depth::F16, 4);

/// `CV_8UC(n)` — 8-bit unsigned with an arbitrary channel count.
#[inline]
#[must_use]
pub const fn cv_8uc(cn: i32) -> MatType {
    MatType::new(Depth::U8, cn)
}
/// `CV_8SC(n)` — 8-bit signed with an arbitrary channel count.
#[inline]
#[must_use]
pub const fn cv_8sc(cn: i32) -> MatType {
    MatType::new(Depth::S8, cn)
}
/// `CV_16UC(n)` — 16-bit unsigned with an arbitrary channel count.
#[inline]
#[must_use]
pub const fn cv_16uc(cn: i32) -> MatType {
    MatType::new(Depth::U16, cn)
}
/// `CV_16SC(n)` — 16-bit signed with an arbitrary channel count.
#[inline]
#[must_use]
pub const fn cv_16sc(cn: i32) -> MatType {
    MatType::new(Depth::S16, cn)
}
/// `CV_32SC(n)` — 32-bit signed with an arbitrary channel count.
#[inline]
#[must_use]
pub const fn cv_32sc(cn: i32) -> MatType {
    MatType::new(Depth::S32, cn)
}
/// `CV_32FC(n)` — 32-bit float with an arbitrary channel count.
#[inline]
#[must_use]
pub const fn cv_32fc(cn: i32) -> MatType {
    MatType::new(Depth::F32, cn)
}
/// `CV_64FC(n)` — 64-bit float with an arbitrary channel count.
#[inline]
#[must_use]
pub const fn cv_64fc(cn: i32) -> MatType {
    MatType::new(Depth::F64, cn)
}
/// `CV_16FC(n)` — half float with an arbitrary channel count.
#[inline]
#[must_use]
pub const fn cv_16fc(cn: i32) -> MatType {
    MatType::new(Depth::F16, cn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_codes_match_opencv() {
        // Values taken from OpenCV's CV_MAKETYPE expansion.
        assert_eq!(CV_8UC1.to_i32(), 0);
        assert_eq!(CV_8UC3.to_i32(), 16);
        assert_eq!(CV_8UC4.to_i32(), 24);
        assert_eq!(CV_32FC1.to_i32(), 5);
        assert_eq!(CV_32FC3.to_i32(), 21);
        assert_eq!(CV_64FC4.to_i32(), 30);
        assert_eq!(CV_16UC2.to_i32(), 10);
    }

    #[test]
    fn roundtrip_through_i32() {
        for depth in [Depth::U8, Depth::S8, Depth::U16, Depth::S16, Depth::S32, Depth::F32, Depth::F64, Depth::F16] {
            for cn in [1, 2, 3, 4, 7, 64, 512] {
                let t = MatType::new(depth, cn);
                let back = MatType::from_i32(t.to_i32()).unwrap();
                assert_eq!(t, back);
                assert_eq!(back.depth(), depth);
                assert_eq!(back.channels(), cn);
            }
        }
    }

    #[test]
    fn elem_sizes() {
        assert_eq!(CV_8UC3.elem_size(), 3);
        assert_eq!(CV_16SC2.elem_size(), 4);
        assert_eq!(CV_64FC4.elem_size(), 32);
        assert_eq!(CV_32FC3.elem_size1(), 4);
    }
}
