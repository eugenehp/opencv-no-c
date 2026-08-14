//! Border extrapolation — `borderInterpolate` and `copyMakeBorder`
//! (`modules/core/src/copy.cpp`).

use crate::error::{Code, Result};
use crate::geometry::Scalar;
use crate::mat::{Mat, MatLike, MatLikeMut};
use crate::types::Depth;
use crate::{cv_assert, cv_error};

/// `cv::BorderTypes`, with upstream's numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum BorderType {
    /// `BORDER_CONSTANT` — `iiiiii|abcdefgh|iiiiiii` with a specified constant `i`.
    Constant = 0,
    /// `BORDER_REPLICATE` — `aaaaaa|abcdefgh|hhhhhhh`
    Replicate = 1,
    /// `BORDER_REFLECT` — `fedcba|abcdefgh|hgfedcb`
    Reflect = 2,
    /// `BORDER_WRAP` — `cdefgh|abcdefgh|abcdefg`
    Wrap = 3,
    /// `BORDER_REFLECT101` / `BORDER_REFLECT_101` — `gfedcb|abcdefgh|gfedcba`.
    ///
    /// This is `BORDER_DEFAULT`.
    Reflect101 = 4,
    /// `BORDER_TRANSPARENT` — leave outliers untouched; only meaningful to
    /// remapping operations.
    Transparent = 5,
}

/// `BORDER_ISOLATED` — do not look outside the ROI into the parent matrix.
///
/// Upstream or-s this into the border argument. It cannot arise here: a
/// [`crate::MatView`] borrows only its own window, so no kernel can read past
/// it in the first place, which is the behaviour `BORDER_ISOLATED` requests.
pub const BORDER_ISOLATED: i32 = 16;

impl BorderType {
    /// `BORDER_DEFAULT`.
    pub const DEFAULT: BorderType = BorderType::Reflect101;
    /// `BORDER_REFLECT_101`, spelled as upstream's alias.
    pub const REFLECT_101: BorderType = BorderType::Reflect101;
}

/// `BORDER_REFLECT_101` on its own, without the `Result` or the enum dispatch.
///
/// This is the mapping every separable kernel in the port needs in its inner
/// loop, where `border_interpolate`'s error path is dead weight. A length of
/// one has to be special-cased: the fold would otherwise map 0 to 0 forever.
#[inline]
#[must_use]
pub fn reflect101(mut p: i32, len: i32) -> i32 {
    if len == 1 {
        return 0;
    }
    loop {
        if p < 0 {
            p = -p;
        } else if p >= len {
            p = 2 * (len - 1) - p;
        } else {
            return p;
        }
    }
}

/// `cv::borderInterpolate(p, len, borderType)`.
///
/// Maps an out-of-range coordinate `p` onto `[0, len)`. Returns `-1` for
/// [`BorderType::Constant`], signalling "use the constant value" — the same
/// sentinel upstream uses.
///
/// This is a direct transcription of `cv::borderInterpolate`, including the
/// iterative reflection loop that handles coordinates several widths outside
/// the image.
pub fn border_interpolate(p: i32, len: i32, border_type: BorderType) -> Result<i32> {
    cv_assert!(len > 0, "borderInterpolate requires len > 0");

    let mut p = p;
    if (p as u32) < (len as u32) {
        return Ok(p);
    }

    match border_type {
        BorderType::Replicate => {
            p = if p < 0 { 0 } else { len - 1 };
        }
        BorderType::Reflect | BorderType::Reflect101 => {
            if border_type == BorderType::Reflect101 {
                return Ok(reflect101(p, len));
            }
            if len == 1 {
                return Ok(0);
            }
            loop {
                p = if p < 0 { -p - 1 } else { len - 1 - (p - len) };
                if (p as u32) < (len as u32) {
                    break;
                }
            }
        }
        BorderType::Wrap => {
            if p < 0 {
                p -= ((p - len + 1) / len) * len;
            }
            if p >= len {
                p %= len;
            }
        }
        BorderType::Constant => p = -1,
        BorderType::Transparent => {
            return cv_error!(Code::StsBadArg, "BORDER_TRANSPARENT has no borderInterpolate mapping")
        }
    }
    Ok(p)
}

/// `cv::copyMakeBorder(src, dst, top, bottom, left, right, borderType, value)`.
///
/// Returns a new matrix of size `(rows + top + bottom) x (cols + left + right)`.
pub fn copy_make_border<S: MatLike + ?Sized>(
    src: &S,
    top: i32,
    bottom: i32,
    left: i32,
    right: i32,
    border_type: BorderType,
    value: Scalar,
) -> Result<Mat> {
    cv_assert!(top >= 0 && bottom >= 0 && left >= 0 && right >= 0, "border widths must be non-negative");
    cv_assert!(!src.is_empty(), "copyMakeBorder requires a non-empty source");
    if border_type == BorderType::Transparent {
        return cv_error!(Code::StsBadArg, "BORDER_TRANSPARENT is not valid for copyMakeBorder");
    }

    let (sr, sc) = (src.rows(), src.cols());
    let typ = src.mat_type();
    let esz = typ.elem_size();
    let mut dst = Mat::zeros(sr + top + bottom, sc + left + right, typ)?;

    // Constant borders are filled from `value`; the pattern is one element wide
    // and repeats across the row.
    let const_elem = if border_type == BorderType::Constant {
        Some(scalar_to_bytes(value, typ)?)
    } else {
        None
    };

    for dy in 0..dst.rows() {
        let sy = border_interpolate(dy - top, sr, border_type)?;
        // Precompute the source row once; a constant border has no source row.
        let src_row: Option<Vec<u8>> =
            if sy >= 0 { Some(src.row_bytes(sy)?.to_vec()) } else { None };

        let dcols = dst.cols();
        let drow = dst.row_bytes_mut(dy)?;
        for dx in 0..dcols {
            let out = &mut drow[dx as usize * esz..(dx as usize + 1) * esz];
            let sx = border_interpolate(dx - left, sc, border_type)?;
            match (&src_row, sx) {
                (Some(row), x) if x >= 0 => {
                    out.copy_from_slice(&row[x as usize * esz..(x as usize + 1) * esz]);
                }
                _ => {
                    // Outside in at least one axis with a constant border.
                    let c = const_elem.as_ref().expect("non-constant border yielded sentinel -1");
                    out.copy_from_slice(c);
                }
            }
        }
    }
    Ok(dst)
}

/// Encode a [`Scalar`] as one element's worth of bytes for `typ`, saturating
/// each channel. Channels beyond the scalar's four are filled with zero, as
/// upstream's `scalarToRawData` does.
pub(crate) fn scalar_to_bytes(value: Scalar, typ: crate::types::MatType) -> Result<Vec<u8>> {
    let cn = typ.channels() as usize;
    let mut out = Vec::with_capacity(typ.elem_size());
    for c in 0..cn {
        let v = if c < 4 { value.0[c] } else { 0.0 };
        push_saturated(&mut out, v, typ.depth())?;
    }
    Ok(out)
}

/// Append `v`, saturated into `depth`, to `out` in native byte order.
fn push_saturated(out: &mut Vec<u8>, v: f64, depth: Depth) -> Result<()> {
    use crate::saturate::Elem;
    match depth {
        Depth::U8 => out.extend_from_slice(&u8::from_f64(v).to_ne_bytes()),
        Depth::S8 => out.extend_from_slice(&i8::from_f64(v).to_ne_bytes()),
        Depth::U16 => out.extend_from_slice(&u16::from_f64(v).to_ne_bytes()),
        Depth::S16 => out.extend_from_slice(&i16::from_f64(v).to_ne_bytes()),
        Depth::S32 => out.extend_from_slice(&i32::from_f64(v).to_ne_bytes()),
        Depth::F32 => out.extend_from_slice(&(v as f32).to_ne_bytes()),
        Depth::F64 => out.extend_from_slice(&v.to_ne_bytes()),
        Depth::F16 => {
            return cv_error!(Code::StsNotImplemented, "CV_16F scalars are not ported yet")
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CV_8UC1;

    /// Expected values are the documented patterns in `core/base.hpp`, read
    /// against an 8-wide row `abcdefgh` = indices 0..8.
    #[test]
    fn replicate_matches_documented_pattern() {
        let f = |p| border_interpolate(p, 8, BorderType::Replicate).unwrap();
        assert_eq!((-3..0).map(f).collect::<Vec<_>>(), vec![0, 0, 0]);
        assert_eq!((8..11).map(f).collect::<Vec<_>>(), vec![7, 7, 7]);
    }

    #[test]
    fn reflect_matches_documented_pattern() {
        // fedcba|abcdefgh|hgfedcb
        let f = |p| border_interpolate(p, 8, BorderType::Reflect).unwrap();
        assert_eq!((-6..0).map(f).collect::<Vec<_>>(), vec![5, 4, 3, 2, 1, 0]);
        assert_eq!((8..15).map(f).collect::<Vec<_>>(), vec![7, 6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn reflect101_matches_documented_pattern() {
        // gfedcb|abcdefgh|gfedcba
        let f = |p| border_interpolate(p, 8, BorderType::Reflect101).unwrap();
        assert_eq!((-5..0).map(f).collect::<Vec<_>>(), vec![5, 4, 3, 2, 1]);
        assert_eq!((8..15).map(f).collect::<Vec<_>>(), vec![6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn wrap_matches_documented_pattern() {
        // cdefgh|abcdefgh|abcdefg
        let f = |p| border_interpolate(p, 8, BorderType::Wrap).unwrap();
        assert_eq!((-6..0).map(f).collect::<Vec<_>>(), vec![2, 3, 4, 5, 6, 7]);
        assert_eq!((8..15).map(f).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn constant_returns_sentinel() {
        assert_eq!(border_interpolate(-1, 8, BorderType::Constant).unwrap(), -1);
        assert_eq!(border_interpolate(9, 8, BorderType::Constant).unwrap(), -1);
        assert_eq!(border_interpolate(3, 8, BorderType::Constant).unwrap(), 3);
    }

    #[test]
    fn reflection_converges_far_outside_the_image() {
        // The upstream loop iterates; a single-step implementation returns
        // out-of-range values here.
        for p in -40..40 {
            for len in [1, 2, 3, 5, 8] {
                for bt in [BorderType::Reflect, BorderType::Reflect101, BorderType::Wrap, BorderType::Replicate] {
                    if len == 1 && bt == BorderType::Reflect101 {
                        continue; // degenerate: upstream short-circuits to 0
                    }
                    let r = border_interpolate(p, len, bt).unwrap();
                    assert!((0..len).contains(&r), "p={p} len={len} {bt:?} -> {r}");
                }
            }
        }
    }

    #[test]
    fn len_one_reflect_is_zero() {
        assert_eq!(border_interpolate(-5, 1, BorderType::Reflect101).unwrap(), 0);
        assert_eq!(border_interpolate(5, 1, BorderType::Reflect).unwrap(), 0);
    }

    #[test]
    fn copy_make_border_constant() {
        let m = Mat::from_vec(2, 2, 1, &[1u8, 2, 3, 4]).unwrap();
        let b = copy_make_border(&m, 1, 1, 1, 1, BorderType::Constant, Scalar::single(9.0)).unwrap();
        assert_eq!(b.size(), crate::Size::new(4, 4));
        assert_eq!(
            b.to_vec::<u8>().unwrap(),
            vec![9, 9, 9, 9, 9, 1, 2, 9, 9, 3, 4, 9, 9, 9, 9, 9]
        );
    }

    #[test]
    fn copy_make_border_replicate() {
        let m = Mat::from_vec(2, 2, 1, &[1u8, 2, 3, 4]).unwrap();
        let b = copy_make_border(&m, 1, 1, 1, 1, BorderType::Replicate, Scalar::default()).unwrap();
        assert_eq!(
            b.to_vec::<u8>().unwrap(),
            vec![1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]
        );
    }

    #[test]
    fn copy_make_border_reflect101_on_3x3() {
        let m = Mat::from_vec(3, 3, 1, &[1u8, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let b = copy_make_border(&m, 1, 1, 1, 1, BorderType::Reflect101, Scalar::default()).unwrap();
        // Row/col -1 maps to index 1, row/col 3 maps to index 1.
        assert_eq!(
            b.to_vec::<u8>().unwrap(),
            vec![
                5, 4, 5, 6, 5, //
                2, 1, 2, 3, 2, //
                5, 4, 5, 6, 5, //
                8, 7, 8, 9, 8, //
                5, 4, 5, 6, 5, //
            ]
        );
    }

    #[test]
    fn copy_make_border_preserves_multichannel() {
        let m = Mat::from_vec(1, 2, 3, &[1u8, 2, 3, 4, 5, 6]).unwrap();
        let b = copy_make_border(&m, 0, 0, 1, 0, BorderType::Constant, Scalar::new(7.0, 8.0, 9.0, 0.0)).unwrap();
        assert_eq!(b.cols(), 3);
        assert_eq!(b.pixel::<u8>(0, 0).unwrap(), &[7, 8, 9]);
        assert_eq!(b.pixel::<u8>(0, 1).unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn zero_width_border_is_a_copy() {
        let m = Mat::zeros(3, 3, CV_8UC1).unwrap();
        let b = copy_make_border(&m, 0, 0, 0, 0, BorderType::Reflect101, Scalar::default()).unwrap();
        assert_eq!(b.size(), m.size());
    }
}
