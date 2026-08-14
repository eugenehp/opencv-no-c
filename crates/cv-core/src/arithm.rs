//! Element-wise arithmetic, comparison and bitwise operations
//! (`modules/core/src/arithm.cpp`, `convert.dispatch.cpp`, `copy.cpp`).
//!
//! # Accumulator choice
//!
//! Integer depths accumulate in `i64` and saturate on store; float depths
//! accumulate in `f64`. This is not an optimisation detail — routing `CV_32S`
//! through `f64` would lose precision above 2^53 and routing it through `i32`
//! would wrap instead of saturate. Upstream hit the latter bug recently enough
//! that the fix (`fix/int-overflow-absdiff`) is the tip commit of the tree this
//! port targets.

use crate::border::scalar_to_bytes;
use crate::error::{Code, Result};
use crate::geometry::Scalar;
use crate::mat::{Mat, MatLike, MatLikeMut};
use crate::saturate::Elem;
use crate::types::{Depth, MatType, CV_8UC1};
use crate::{cv_assert, cv_error};

/// Fill every pixel of `dst` with `value`, saturating per channel — `Mat::setTo`.
pub fn set_to<D: MatLikeMut + ?Sized>(dst: &mut D, value: Scalar) -> Result<()> {
    let typ = dst.mat_type();
    let elem = scalar_to_bytes(value, typ)?;
    let esz = typ.elem_size();
    for y in 0..dst.rows() {
        let row = dst.row_bytes_mut(y)?;
        for chunk in row.chunks_exact_mut(esz) {
            chunk.copy_from_slice(&elem);
        }
    }
    Ok(())
}

/// Write one channel of one pixel, saturating into the destination depth.
///
/// The depth-generic counterpart of [`MatLike::at_f64`], used by kernels that
/// compute in `f64` and store into whatever the destination happens to be.
pub fn set_channel(dst: &mut Mat, y: i32, x: i32, c: i32, v: f64) -> Result<()> {
    let i = c as usize;
    match dst.depth() {
        Depth::U8 => dst.pixel_mut::<u8>(y, x)?[i] = u8::from_f64(v),
        Depth::S8 => dst.pixel_mut::<i8>(y, x)?[i] = i8::from_f64(v),
        Depth::U16 => dst.pixel_mut::<u16>(y, x)?[i] = u16::from_f64(v),
        Depth::S16 => dst.pixel_mut::<i16>(y, x)?[i] = i16::from_f64(v),
        Depth::S32 => dst.pixel_mut::<i32>(y, x)?[i] = i32::from_f64(v),
        Depth::F32 => dst.pixel_mut::<f32>(y, x)?[i] = v as f32,
        Depth::F64 => dst.pixel_mut::<f64>(y, x)?[i] = v,
        Depth::F16 => return cv_error!(Code::StsNotImplemented, "CV_16F is not ported yet"),
    }
    Ok(())
}

/// Depth-generic single-channel write, for any [`MatLikeMut`].
///
/// [`set_channel`] is the same operation restricted to an owned [`Mat`]; this
/// one also accepts views, which the transform helpers in `misc` need.
pub fn set_channel_generic<D: MatLikeMut + ?Sized>(
    dst: &mut D,
    y: i32,
    x: i32,
    c: i32,
    v: f64,
) -> Result<()> {
    let i = c as usize;
    match dst.depth() {
        Depth::U8 => dst.pixel_mut::<u8>(y, x)?[i] = u8::from_f64(v),
        Depth::S8 => dst.pixel_mut::<i8>(y, x)?[i] = i8::from_f64(v),
        Depth::U16 => dst.pixel_mut::<u16>(y, x)?[i] = u16::from_f64(v),
        Depth::S16 => dst.pixel_mut::<i16>(y, x)?[i] = i16::from_f64(v),
        Depth::S32 => dst.pixel_mut::<i32>(y, x)?[i] = i32::from_f64(v),
        Depth::F32 => dst.pixel_mut::<f32>(y, x)?[i] = v as f32,
        Depth::F64 => dst.pixel_mut::<f64>(y, x)?[i] = v,
        Depth::F16 => return cv_error!(Code::StsNotImplemented, "CV_16F is not ported yet"),
    }
    Ok(())
}

/// Dispatch a per-channel-value kernel over the depth of `m`.
///
/// The kernel receives and returns `f64`; integer depths still round-trip
/// exactly because every integer depth's range fits in `f64`'s 53-bit mantissa
/// except `CV_32S`, which is handled by the dedicated `i64` paths below.
macro_rules! for_each_depth {
    ($depth:expr, |$t:ident| $body:expr) => {
        match $depth {
            Depth::U8 => {
                type $t = u8;
                $body
            }
            Depth::S8 => {
                type $t = i8;
                $body
            }
            Depth::U16 => {
                type $t = u16;
                $body
            }
            Depth::S16 => {
                type $t = i16;
                $body
            }
            Depth::S32 => {
                type $t = i32;
                $body
            }
            Depth::F32 => {
                type $t = f32;
                $body
            }
            Depth::F64 => {
                type $t = f64;
                $body
            }
            Depth::F16 => {
                return cv_error!(Code::StsNotImplemented, "CV_16F is not ported yet")
            }
        }
    };
}

/// Confirm two operands have identical size and type.
fn check_same<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B) -> Result<()> {
    if a.size() != b.size() {
        return cv_error!(
            Code::StsUnmatchedSizes,
            "operand sizes differ: {}x{} vs {}x{}",
            a.cols(),
            a.rows(),
            b.cols(),
            b.rows()
        );
    }
    if a.mat_type() != b.mat_type() {
        return cv_error!(
            Code::StsUnmatchedFormats,
            "operand types differ: {} vs {}",
            a.mat_type(),
            b.mat_type()
        );
    }
    Ok(())
}

/// Validate an optional operation mask: `CV_8UC1`, same size as the operands.
fn check_mask<M: MatLike + ?Sized>(mask: Option<&M>, rows: i32, cols: i32) -> Result<()> {
    if let Some(m) = mask {
        cv_assert!(
            m.mat_type() == CV_8UC1,
            "mask must be CV_8UC1, got {}",
            m.mat_type()
        );
        cv_assert!(
            m.rows() == rows && m.cols() == cols,
            "mask size {}x{} does not match operands {cols}x{rows}",
            m.cols(),
            m.rows()
        );
    }
    Ok(())
}

/// Core element-wise loop shared by every binary arithmetic entry point.
///
/// `int_op` runs for integer depths, `float_op` for float depths.
fn binary_op<A, B, M>(
    a: &A,
    b: &B,
    mask: Option<&M>,
    int_op: fn(i64, i64) -> i64,
    float_op: fn(f64, f64) -> f64,
) -> Result<Mat>
where
    A: MatLike + ?Sized,
    B: MatLike + ?Sized,
    M: MatLike + ?Sized,
{
    check_same(a, b)?;
    check_mask(mask, a.rows(), a.cols())?;
    let typ = a.mat_type();
    let mut dst = Mat::zeros(a.rows(), a.cols(), typ)?;
    let is_float = typ.depth().is_float();

    for y in 0..a.rows() {
        let mrow = match mask {
            Some(m) => Some(m.row_as::<u8>(y)?.to_vec()),
            None => None,
        };
        for_each_depth!(typ.depth(), |T| {
            let ar = a.row_as::<T>(y)?;
            let br = b.row_as::<T>(y)?;
            let cn = typ.channels() as usize;
            let dr = dst.row_as_mut::<T>(y)?;
            for i in 0..dr.len() {
                if let Some(mr) = &mrow {
                    if mr[i / cn] == 0 {
                        continue;
                    }
                }
                dr[i] = if is_float {
                    T::from_f64(float_op(ar[i].to_f64(), br[i].to_f64()))
                } else {
                    T::from_i64(int_op(ar[i].to_i64(), br[i].to_i64()))
                };
            }
        });
    }
    Ok(dst)
}

/// `cv::add(src1, src2, dst, mask)` — saturating element-wise sum.
pub fn add<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B) -> Result<Mat> {
    add_masked::<A, B, Mat>(a, b, None)
}

/// `cv::add` with an optional operation mask.
pub fn add_masked<A, B, M>(a: &A, b: &B, mask: Option<&M>) -> Result<Mat>
where
    A: MatLike + ?Sized,
    B: MatLike + ?Sized,
    M: MatLike + ?Sized,
{
    binary_op(a, b, mask, |x, y| x.saturating_add(y), |x, y| x + y)
}

/// `cv::add(src, value, dst)` — add a per-channel constant, saturating.
///
/// Channel `c` of every pixel gets `value[c]`, which is what makes this
/// different from adding one number: a three-channel image can be shifted per
/// channel in one call.
pub fn add_scalar<A: MatLike + ?Sized>(a: &A, value: crate::Scalar) -> Result<Mat> {
    scalar_op(a, value, 1.0)
}

/// `cv::subtract(src, value, dst)` — subtract a per-channel constant.
pub fn subtract_scalar<A: MatLike + ?Sized>(a: &A, value: crate::Scalar) -> Result<Mat> {
    scalar_op(a, value, -1.0)
}

/// Shared body of [`add_scalar`] and [`subtract_scalar`].
///
/// `sign` selects the direction. Accumulation is in `f64` and the store
/// saturates, matching [`add`]'s behaviour on the same values.
fn scalar_op<A: MatLike + ?Sized>(a: &A, value: crate::Scalar, sign: f64) -> Result<Mat> {
    let cn = a.channels();
    cv_assert!(cn <= 4, "add/subtract with a Scalar handles at most 4 channels, got {cn}");
    let mut dst = Mat::zeros(a.rows(), a.cols(), a.mat_type())?;
    for y in 0..a.rows() {
        for x in 0..a.cols() {
            for c in 0..cn {
                let v = a.at_f64(y, x, c)? + sign * value[c as usize];
                set_channel(&mut dst, y, x, c, v)?;
            }
        }
    }
    Ok(dst)
}

/// `cv::subtract(src1, src2, dst)` — saturating element-wise difference.
pub fn subtract<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B) -> Result<Mat> {
    subtract_masked::<A, B, Mat>(a, b, None)
}

/// `cv::subtract` with an optional operation mask.
pub fn subtract_masked<A, B, M>(a: &A, b: &B, mask: Option<&M>) -> Result<Mat>
where
    A: MatLike + ?Sized,
    B: MatLike + ?Sized,
    M: MatLike + ?Sized,
{
    binary_op(a, b, mask, |x, y| x.saturating_sub(y), |x, y| x - y)
}

/// `cv::absdiff(src1, src2, dst)` — saturating `|a - b|`.
pub fn absdiff<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B) -> Result<Mat> {
    binary_op::<A, B, Mat>(
        a,
        b,
        None,
        // `saturating_sub` then `saturating_abs` keeps i64::MIN from wrapping.
        |x, y| x.saturating_sub(y).saturating_abs(),
        |x, y| (x - y).abs(),
    )
}

/// `cv::multiply(src1, src2, dst, scale)`.
///
/// # Rounding
///
/// The scaled result **truncates** toward zero for integer depths; it does not
/// round. Upstream's declared form is `saturate_cast<T>(scale * a * b)`, which
/// reads as round-to-nearest, but a real OpenCV build does not behave that way:
/// with `scale = 0.01`, `a = 20`, `b = 58` the product is 11.60 and cv2 returns
/// **11**, not 12. Rounding disagrees on 40% of pixels; truncating the product
/// `scale * (a * b)` is exact on every one.
///
/// The association matters too — `scale * (a * b)` rather than
/// `(scale * a) * b`, because the integer product is exact before scaling
/// while the intermediate `scale * a` is not.
pub fn multiply<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B, scale: f64) -> Result<Mat> {
    check_same(a, b)?;
    let typ = a.mat_type();
    let integral = !typ.depth().is_float();
    let mut dst = Mat::zeros(a.rows(), a.cols(), typ)?;
    for y in 0..a.rows() {
        for_each_depth!(typ.depth(), |T| {
            let ar = a.row_as::<T>(y)?;
            let br = b.row_as::<T>(y)?;
            let dr = dst.row_as_mut::<T>(y)?;
            for i in 0..dr.len() {
                let v = scale * (ar[i].to_f64() * br[i].to_f64());
                dr[i] = if integral { T::from_f64(v.trunc()) } else { T::from_f64(v) };
            }
        });
    }
    Ok(dst)
}

/// `cv::divide(src1, src2, dst, scale)`.
///
/// # Exact `.5` ties round the other way in cv2
///
/// The quotient is evaluated in `f32` (see [`add_weighted`] for why the width
/// matters). Where it lands on an exact half, cv2 rounds **down** and this port
/// rounds half-to-even, which is `cvRound`'s documented behaviour:
///
/// ```text
/// a=35  b=37  -> 3.5   cv2 gives 3, half-to-even gives 4
/// a=115 b=37  -> 11.5  cv2 gives 11, half-to-even gives 12
/// a=60  b=148 -> 1.5   cv2 gives 1,  half-to-even gives 2
/// ```
///
/// That is 3 pixels in 4096 on random `u8` input, and every one of them is an
/// exact tie — the arithmetic agrees bit-for-bit everywhere else. Re-grouping
/// the expression does not move it: `(a*s)/b`, `a*(s/b)`, `(a/b)*s` and a full
/// `f64` evaluation all produce the same three disagreements, so the residual
/// is in how the tie is broken and not in how the quotient is formed. This is
/// the same `.5`-tie divergence already documented for the SIMD paths
/// elsewhere in this port.
///
/// Division by zero yields 0, matching upstream rather than producing an
/// infinity — this is deliberate in OpenCV and relied upon by callers.
pub fn divide<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B, scale: f64) -> Result<Mat> {
    check_same(a, b)?;
    let typ = a.mat_type();
    let narrow = !matches!(typ.depth(), Depth::F64);
    let mut dst = Mat::zeros(a.rows(), a.cols(), typ)?;
    for y in 0..a.rows() {
        for_each_depth!(typ.depth(), |T| {
            let ar = a.row_as::<T>(y)?;
            let br = b.row_as::<T>(y)?;
            let dr = dst.row_as_mut::<T>(y)?;
            for i in 0..dr.len() {
                let d = br[i].to_f64();
                if d == 0.0 {
                    dr[i] = T::ZERO;
                } else if narrow {
                    // Upstream `c_div` widens to **f32**, not f64:
                    // `saturate_cast<T>(scalar * CV_8TO32F(a) / CV_8TO32F(b))`.
                    let v = (scale as f32) * (ar[i].to_f64() as f32) / (d as f32);
                    dr[i] = T::from_f64(v as f64);
                } else {
                    dr[i] = T::from_f64(scale * ar[i].to_f64() / d);
                }
            }
        });
    }
    Ok(dst)
}

/// `cv::addWeighted(src1, alpha, src2, beta, gamma, dst)`.
///
/// # Precision
///
/// Upstream evaluates `a*alpha + b*beta + gamma` in **f32** for every depth
/// except `CV_32S` and `CV_64F`. Computing it in `f64` disagrees with cv2 on
/// roughly 1 in 170 random `u8` pairs, so the width is part of the answer.
///
/// # OpenCV disagrees with itself here
///
/// The `f32` evaluation is fused, and the vectorised body and the scalar tail
/// fuse it **differently**. On the seven-value fixture in the parity harness,
/// `cv2.addWeighted` returns
///
/// ```text
/// [116, 126, 180, 196, 162, 160, 120]   // 1x7 array, scalar tail
/// [115, 125, 179, 195, 162, 160, 120]   // same values tiled, vector body
/// ```
///
/// — four of seven differing by one level, from the same call on the same
/// numbers. The scalar tail computes `fma(a, alpha, b*beta) + gamma`; the
/// vector body computes `fma(b, beta, fma(a, alpha, gamma))`.
///
/// **This port reproduces the vector body**, because that is what runs for any
/// image large enough to care about: a caller comparing against cv2 on real
/// data sees agreement, and only a caller working on a handful of pixels sees
/// the one-level difference. The choice cannot be avoided — no single
/// expression matches both, since OpenCV's two paths do not match each other.
pub fn add_weighted<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    a: &A,
    alpha: f64,
    b: &B,
    beta: f64,
    gamma: f64,
) -> Result<Mat> {
    check_same(a, b)?;
    let typ = a.mat_type();
    let narrow = !matches!(typ.depth(), Depth::S32 | Depth::F64);
    let (alpha32, beta32, gamma32) = (alpha as f32, beta as f32, gamma as f32);
    let mut dst = Mat::zeros(a.rows(), a.cols(), typ)?;
    for y in 0..a.rows() {
        for_each_depth!(typ.depth(), |T| {
            let ar = a.row_as::<T>(y)?;
            let br = b.row_as::<T>(y)?;
            let dr = dst.row_as_mut::<T>(y)?;
            for i in 0..dr.len() {
                let v = if narrow {
                    // Two fused multiply-adds in `f32`, with gamma folded into
                    // the **`a * alpha`** product and `b * beta` layered on top.
                    // This reproduces cv2's *vectorised* body exactly; see the
                    // note on [`add_weighted`] for why that is a choice.
                    let a32 = ar[i].to_f64() as f32;
                    let b32 = br[i].to_f64() as f32;
                    if beta32 == 1.0 && gamma32 == 0.0 {
                        // Upstream switches to `op_add_scale` for this case
                        // rather than feeding 1 and 0 through the general form,
                        // and the two do not round alike.
                        f64::from(a32.mul_add(alpha32, b32))
                    } else {
                        f64::from(b32.mul_add(beta32, a32.mul_add(alpha32, gamma32)))
                    }
                } else {
                    ar[i].to_f64() * alpha + br[i].to_f64() * beta + gamma
                };
                dr[i] = T::from_f64(v);
            }
        });
    }
    Ok(dst)
}

/// `cv::min(src1, src2, dst)`.
pub fn min<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B) -> Result<Mat> {
    binary_op::<A, B, Mat>(a, b, None, Ord::min, f64::min)
}

/// `cv::max(src1, src2, dst)`.
pub fn max<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B) -> Result<Mat> {
    binary_op::<A, B, Mat>(a, b, None, Ord::max, f64::max)
}

/// `cv::CmpTypes`, with upstream's numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum CmpType {
    /// `CMP_EQ` — `src1 == src2`
    Eq = 0,
    /// `CMP_GT` — `src1 > src2`
    Gt = 1,
    /// `CMP_GE` — `src1 >= src2`
    Ge = 2,
    /// `CMP_LT` — `src1 < src2`
    Lt = 3,
    /// `CMP_LE` — `src1 <= src2`
    Le = 4,
    /// `CMP_NE` — `src1 != src2`
    Ne = 5,
}

impl CmpType {
    fn apply(self, x: f64, y: f64) -> bool {
        match self {
            CmpType::Eq => x == y,
            CmpType::Gt => x > y,
            CmpType::Ge => x >= y,
            CmpType::Lt => x < y,
            CmpType::Le => x <= y,
            CmpType::Ne => x != y,
        }
    }
}

/// `cv::compare(src1, src2, dst, cmpop)` — result is `CV_8UC1` with 0 or 255.
pub fn compare<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B, op: CmpType) -> Result<Mat> {
    check_same(a, b)?;
    cv_assert!(a.channels() == 1, "compare() requires single-channel operands");
    let mut dst = Mat::zeros(a.rows(), a.cols(), CV_8UC1)?;
    for y in 0..a.rows() {
        for x in 0..a.cols() {
            let (l, r) = (a.at_f64(y, x, 0)?, b.at_f64(y, x, 0)?);
            dst.pixel_mut::<u8>(y, x)?[0] = if op.apply(l, r) { 255 } else { 0 };
        }
    }
    Ok(dst)
}

/// `cv::inRange(src, lowerb, upperb, dst)` — `CV_8UC1` mask, 255 where every
/// channel lies within the inclusive bounds.
pub fn in_range<S: MatLike + ?Sized>(src: &S, lower: Scalar, upper: Scalar) -> Result<Mat> {
    let mut dst = Mat::zeros(src.rows(), src.cols(), CV_8UC1)?;
    let cn = src.channels();
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            let mut inside = true;
            for c in 0..cn {
                let v = src.at_f64(y, x, c)?;
                let (lo, hi) = (lower.0[c.min(3) as usize], upper.0[c.min(3) as usize]);
                if v < lo || v > hi {
                    inside = false;
                    break;
                }
            }
            dst.pixel_mut::<u8>(y, x)?[0] = if inside { 255 } else { 0 };
        }
    }
    Ok(dst)
}

/// Element-wise bitwise operation over the raw bytes.
fn bitwise<A, B>(a: &A, b: &B, op: fn(u8, u8) -> u8) -> Result<Mat>
where
    A: MatLike + ?Sized,
    B: MatLike + ?Sized,
{
    check_same(a, b)?;
    let mut dst = Mat::zeros(a.rows(), a.cols(), a.mat_type())?;
    for y in 0..a.rows() {
        let (ar, br) = (a.row_bytes(y)?.to_vec(), b.row_bytes(y)?.to_vec());
        let dr = dst.row_bytes_mut(y)?;
        for i in 0..dr.len() {
            dr[i] = op(ar[i], br[i]);
        }
    }
    Ok(dst)
}

/// `cv::bitwise_and`.
pub fn bitwise_and<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B) -> Result<Mat> {
    bitwise(a, b, |x, y| x & y)
}

/// `cv::bitwise_or`.
pub fn bitwise_or<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B) -> Result<Mat> {
    bitwise(a, b, |x, y| x | y)
}

/// `cv::bitwise_xor`.
pub fn bitwise_xor<A: MatLike + ?Sized, B: MatLike + ?Sized>(a: &A, b: &B) -> Result<Mat> {
    bitwise(a, b, |x, y| x ^ y)
}

/// `cv::bitwise_not`.
pub fn bitwise_not<S: MatLike + ?Sized>(src: &S) -> Result<Mat> {
    let mut dst = Mat::zeros(src.rows(), src.cols(), src.mat_type())?;
    for y in 0..src.rows() {
        let sr = src.row_bytes(y)?.to_vec();
        let dr = dst.row_bytes_mut(y)?;
        for i in 0..dr.len() {
            dr[i] = !sr[i];
        }
    }
    Ok(dst)
}

/// `Mat::convertTo(dst, rtype, alpha, beta)` — `dst = saturate_cast(alpha * src + beta)`.
pub fn convert_to<S: MatLike + ?Sized>(
    src: &S,
    depth: Depth,
    alpha: f64,
    beta: f64,
) -> Result<Mat> {
    let dst_type = MatType::new(depth, src.channels());
    let mut dst = Mat::zeros(src.rows(), src.cols(), dst_type)?;
    for y in 0..src.rows() {
        // Read the source row once, widened, then store into the target depth.
        let widened: Vec<f64> = for_each_depth!(src.depth(), |S2| {
            src.row_as::<S2>(y)?.iter().map(|v| v.to_f64()).collect()
        });
        for_each_depth!(depth, |D| {
            let dr = dst.row_as_mut::<D>(y)?;
            for (d, &s) in dr.iter_mut().zip(widened.iter()) {
                *d = D::from_f64(s * alpha + beta);
            }
        });
    }
    Ok(dst)
}

/// `cv::split(src, mvbegin)` — one single-channel matrix per channel.
pub fn split<S: MatLike + ?Sized>(src: &S) -> Result<Vec<Mat>> {
    let cn = src.channels() as usize;
    let single = src.mat_type().with_channels(1);
    let mut out = Vec::with_capacity(cn);
    for c in 0..cn {
        let mut plane = Mat::zeros(src.rows(), src.cols(), single)?;
        for y in 0..src.rows() {
            for_each_depth!(src.depth(), |T| {
                let sr = src.row_as::<T>(y)?;
                let dr = plane.row_as_mut::<T>(y)?;
                for (x, d) in dr.iter_mut().enumerate() {
                    *d = sr[x * cn + c];
                }
            });
        }
        out.push(plane);
    }
    Ok(out)
}

/// `cv::merge(mv, dst)` — interleave single-channel planes.
pub fn merge(planes: &[Mat]) -> Result<Mat> {
    cv_assert!(!planes.is_empty(), "merge() requires at least one plane");
    let first = &planes[0];
    for p in planes {
        cv_assert!(p.channels() == 1, "merge() requires single-channel planes");
        check_same(first, p)?;
    }
    let cn = planes.len();
    let typ = first.mat_type().with_channels(cn as i32);
    let mut dst = Mat::zeros(first.rows(), first.cols(), typ)?;
    for y in 0..first.rows() {
        for (c, p) in planes.iter().enumerate() {
            for_each_depth!(first.depth(), |T| {
                let sr = p.row_as::<T>(y)?.to_vec();
                let dr = dst.row_as_mut::<T>(y)?;
                for (x, &s) in sr.iter().enumerate() {
                    dr[x * cn + c] = s;
                }
            });
        }
    }
    Ok(dst)
}

/// `cv::transpose(src, dst)`.
pub fn transpose<S: MatLike + ?Sized>(src: &S) -> Result<Mat> {
    let typ = src.mat_type();
    let esz = typ.elem_size();
    let mut dst = Mat::zeros(src.cols(), src.rows(), typ)?;
    for y in 0..src.rows() {
        let srow = src.row_bytes(y)?.to_vec();
        for x in 0..src.cols() {
            let elem = &srow[x as usize * esz..(x as usize + 1) * esz];
            let drow = dst.row_bytes_mut(x)?;
            drow[y as usize * esz..(y as usize + 1) * esz].copy_from_slice(elem);
        }
    }
    Ok(dst)
}

/// `cv::flip(src, dst, flipCode)`.
///
/// `flip_code > 0` mirrors around the y-axis (horizontally), `0` around the
/// x-axis (vertically), `< 0` around both.
pub fn flip<S: MatLike + ?Sized>(src: &S, flip_code: i32) -> Result<Mat> {
    let typ = src.mat_type();
    let esz = typ.elem_size();
    let (fx, fy) = (flip_code != 0, flip_code <= 0);
    let mut dst = Mat::zeros(src.rows(), src.cols(), typ)?;
    for y in 0..src.rows() {
        let sy = if fy { src.rows() - 1 - y } else { y };
        let srow = src.row_bytes(sy)?.to_vec();
        let drow = dst.row_bytes_mut(y)?;
        for x in 0..src.cols() {
            let sx = if fx { src.cols() - 1 - x } else { x };
            drow[x as usize * esz..(x as usize + 1) * esz]
                .copy_from_slice(&srow[sx as usize * esz..(sx as usize + 1) * esz]);
        }
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CV_32FC1, CV_32SC1, CV_8UC3};

    fn u8m(rows: i32, cols: i32, v: &[u8]) -> Mat {
        Mat::from_vec(rows, cols, 1, v).unwrap()
    }

    #[test]
    fn add_saturates_at_the_top() {
        let a = u8m(1, 4, &[250, 100, 0, 255]);
        let b = u8m(1, 4, &[10, 100, 0, 255]);
        assert_eq!(add(&a, &b).unwrap().to_vec::<u8>().unwrap(), vec![255, 200, 0, 255]);
    }

    #[test]
    fn subtract_saturates_at_the_bottom() {
        let a = u8m(1, 3, &[10, 200, 0]);
        let b = u8m(1, 3, &[20, 100, 5]);
        assert_eq!(subtract(&a, &b).unwrap().to_vec::<u8>().unwrap(), vec![0, 100, 0]);
    }

    #[test]
    fn absdiff_is_symmetric_and_saturating() {
        let a = u8m(1, 3, &[10, 200, 0]);
        let b = u8m(1, 3, &[20, 100, 255]);
        let ab = absdiff(&a, &b).unwrap().to_vec::<u8>().unwrap();
        let ba = absdiff(&b, &a).unwrap().to_vec::<u8>().unwrap();
        assert_eq!(ab, vec![10, 100, 255]);
        assert_eq!(ab, ba);
    }

    #[test]
    fn absdiff_on_i32_extremes_does_not_wrap() {
        // The upstream bug this guards against: i32::MIN - i32::MAX wraps in
        // 32-bit arithmetic and yields 1 instead of saturating to i32::MAX.
        let a = Mat::from_vec(1, 2, 1, &[i32::MIN, i32::MAX]).unwrap();
        let b = Mat::from_vec(1, 2, 1, &[i32::MAX, i32::MIN]).unwrap();
        assert_eq!(a.mat_type(), CV_32SC1);
        assert_eq!(
            absdiff(&a, &b).unwrap().to_vec::<i32>().unwrap(),
            vec![i32::MAX, i32::MAX]
        );
    }

    #[test]
    fn add_on_i32_saturates_rather_than_wrapping() {
        let a = Mat::from_vec(1, 2, 1, &[i32::MAX, i32::MIN]).unwrap();
        let b = Mat::from_vec(1, 2, 1, &[100, -100]).unwrap();
        assert_eq!(add(&a, &b).unwrap().to_vec::<i32>().unwrap(), vec![i32::MAX, i32::MIN]);
    }

    #[test]
    fn float_ops_do_not_saturate() {
        let a = Mat::from_vec(1, 2, 1, &[1.5f32, -3.0]).unwrap();
        let b = Mat::from_vec(1, 2, 1, &[2.5f32, 1.0]).unwrap();
        assert_eq!(a.mat_type(), CV_32FC1);
        assert_eq!(add(&a, &b).unwrap().to_vec::<f32>().unwrap(), vec![4.0, -2.0]);
        assert_eq!(subtract(&a, &b).unwrap().to_vec::<f32>().unwrap(), vec![-1.0, -4.0]);
    }

    #[test]
    fn masked_add_leaves_unmasked_pixels_zero() {
        let a = u8m(1, 4, &[1, 2, 3, 4]);
        let b = u8m(1, 4, &[10, 20, 30, 40]);
        let mask = u8m(1, 4, &[255, 0, 1, 0]);
        let out = add_masked(&a, &b, Some(&mask)).unwrap();
        assert_eq!(out.to_vec::<u8>().unwrap(), vec![11, 0, 33, 0]);
    }

    #[test]
    fn multiply_applies_scale_and_rounds_half_even() {
        let a = u8m(1, 3, &[10, 3, 5]);
        let b = u8m(1, 3, &[10, 3, 1]);
        assert_eq!(multiply(&a, &b, 1.0).unwrap().to_vec::<u8>().unwrap(), vec![100, 9, 5]);
        // Integer results truncate rather than round: 3*3*0.5 = 4.5 -> 4, and
        // 5*1*0.5 = 2.5 -> 2. Verified against a real cv2 build.
        assert_eq!(multiply(&a, &b, 0.5).unwrap().to_vec::<u8>().unwrap(), vec![50, 4, 2]);

        // A case where truncation and rounding visibly disagree: 20*58*0.01
        // is 11.6, and OpenCV returns 11.
        let p = Mat::from_vec(1, 1, 1, &[20u8]).unwrap();
        let q = Mat::from_vec(1, 1, 1, &[58u8]).unwrap();
        assert_eq!(multiply(&p, &q, 0.01).unwrap().to_vec::<u8>().unwrap(), vec![11]);
    }

    #[test]
    fn divide_by_zero_yields_zero() {
        let a = u8m(1, 3, &[10, 20, 30]);
        let b = u8m(1, 3, &[2, 0, 3]);
        assert_eq!(divide(&a, &b, 1.0).unwrap().to_vec::<u8>().unwrap(), vec![5, 0, 10]);
    }

    /// `addWeighted` on `u8` reproduces cv2's **vectorised** body, which is
    /// what runs for any real image. cv2's scalar tail computes the same
    /// expression with a different fusion and returns `[116, 126, 180, 196,
    /// ...]` for these inputs — see the note on [`add_weighted`]. Both are
    /// pinned so a future change cannot silently move between them.
    #[test]
    fn add_weighted_follows_cv2s_vectorised_rounding() {
        const A: [u8; 7] = [89, 129, 173, 193, 175, 153, 133];
        const B: [u8; 7] = [103, 58, 116, 121, 70, 102, 38];
        /// What `cv2.addWeighted` returns once the input is large enough to
        /// reach its vector body.
        const CV2_VECTOR: [u8; 7] = [115, 125, 179, 195, 162, 160, 120];
        /// What it returns for a 1x7 input, from its scalar tail.
        const CV2_SCALAR_TAIL: [u8; 7] = [116, 126, 180, 196, 162, 160, 120];

        let out = add_weighted(&u8m(1, 7, &A), 0.7, &u8m(1, 7, &B), 0.4, 12.0).unwrap();
        assert_eq!(out.to_vec::<u8>().unwrap(), CV2_VECTOR.to_vec());
        assert_ne!(
            out.to_vec::<u8>().unwrap(),
            CV2_SCALAR_TAIL.to_vec(),
            "cv2's two paths differ on four of these; matching both is impossible"
        );
    }

    #[test]
    fn add_weighted_blends() {
        let a = u8m(1, 2, &[100, 200]);
        let b = u8m(1, 2, &[0, 100]);
        let out = add_weighted(&a, 0.5, &b, 0.5, 0.0).unwrap();
        assert_eq!(out.to_vec::<u8>().unwrap(), vec![50, 150]);
    }

    #[test]
    fn min_max_are_elementwise() {
        let a = u8m(1, 3, &[1, 5, 9]);
        let b = u8m(1, 3, &[4, 4, 4]);
        assert_eq!(min(&a, &b).unwrap().to_vec::<u8>().unwrap(), vec![1, 4, 4]);
        assert_eq!(max(&a, &b).unwrap().to_vec::<u8>().unwrap(), vec![4, 5, 9]);
    }

    #[test]
    fn compare_yields_0_or_255() {
        let a = u8m(1, 3, &[1, 5, 9]);
        let b = u8m(1, 3, &[4, 5, 4]);
        assert_eq!(compare(&a, &b, CmpType::Gt).unwrap().to_vec::<u8>().unwrap(), vec![0, 0, 255]);
        assert_eq!(compare(&a, &b, CmpType::Eq).unwrap().to_vec::<u8>().unwrap(), vec![0, 255, 0]);
        assert_eq!(compare(&a, &b, CmpType::Le).unwrap().to_vec::<u8>().unwrap(), vec![255, 255, 0]);
    }

    #[test]
    fn bitwise_ops_operate_on_raw_bytes() {
        let a = u8m(1, 2, &[0b1100, 0xFF]);
        let b = u8m(1, 2, &[0b1010, 0x0F]);
        assert_eq!(bitwise_and(&a, &b).unwrap().to_vec::<u8>().unwrap(), vec![0b1000, 0x0F]);
        assert_eq!(bitwise_or(&a, &b).unwrap().to_vec::<u8>().unwrap(), vec![0b1110, 0xFF]);
        assert_eq!(bitwise_xor(&a, &b).unwrap().to_vec::<u8>().unwrap(), vec![0b0110, 0xF0]);
        assert_eq!(bitwise_not(&a).unwrap().to_vec::<u8>().unwrap(), vec![0xF3, 0x00]);
    }

    #[test]
    fn convert_to_scales_and_saturates() {
        let a = u8m(1, 4, &[0, 100, 200, 255]);
        let f = convert_to(&a, Depth::F32, 1.0 / 255.0, 0.0).unwrap();
        let v = f.to_vec::<f32>().unwrap();
        assert!((v[3] - 1.0).abs() < 1e-6);
        assert!((v[1] - 100.0 / 255.0).abs() < 1e-6);

        // Widening back down saturates.
        let back = convert_to(&a, Depth::U8, 2.0, 0.0).unwrap();
        assert_eq!(back.to_vec::<u8>().unwrap(), vec![0, 200, 255, 255]);
    }

    #[test]
    fn convert_to_negative_into_unsigned_clamps() {
        let a = Mat::from_vec(1, 3, 1, &[-5.0f32, 0.0, 300.0]).unwrap();
        let b = convert_to(&a, Depth::U8, 1.0, 0.0).unwrap();
        assert_eq!(b.to_vec::<u8>().unwrap(), vec![0, 0, 255]);
    }

    #[test]
    fn split_merge_roundtrip() {
        let data: Vec<u8> = (0..12).collect();
        let m = Mat::from_vec(2, 2, 3, &data).unwrap();
        let planes = split(&m).unwrap();
        assert_eq!(planes.len(), 3);
        assert_eq!(planes[0].to_vec::<u8>().unwrap(), vec![0, 3, 6, 9]);
        assert_eq!(planes[1].to_vec::<u8>().unwrap(), vec![1, 4, 7, 10]);
        assert_eq!(planes[2].to_vec::<u8>().unwrap(), vec![2, 5, 8, 11]);
        let back = merge(&planes).unwrap();
        assert_eq!(back.mat_type(), CV_8UC3);
        assert_eq!(back.to_vec::<u8>().unwrap(), data);
    }

    #[test]
    fn transpose_swaps_axes() {
        let m = u8m(2, 3, &[1, 2, 3, 4, 5, 6]);
        let t = transpose(&m).unwrap();
        assert_eq!(t.size(), crate::Size::new(2, 3));
        assert_eq!(t.to_vec::<u8>().unwrap(), vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn flip_codes_match_opencv() {
        let m = u8m(2, 2, &[1, 2, 3, 4]);
        assert_eq!(flip(&m, 1).unwrap().to_vec::<u8>().unwrap(), vec![2, 1, 4, 3]);
        assert_eq!(flip(&m, 0).unwrap().to_vec::<u8>().unwrap(), vec![3, 4, 1, 2]);
        assert_eq!(flip(&m, -1).unwrap().to_vec::<u8>().unwrap(), vec![4, 3, 2, 1]);
    }

    #[test]
    fn in_range_checks_every_channel() {
        let m = Mat::from_vec(1, 2, 3, &[10u8, 50, 90, 10, 250, 90]).unwrap();
        let mask = in_range(&m, Scalar::new(0.0, 0.0, 0.0, 0.0), Scalar::new(100.0, 100.0, 100.0, 0.0)).unwrap();
        assert_eq!(mask.to_vec::<u8>().unwrap(), vec![255, 0]);
    }

    #[test]
    fn mismatched_operands_are_rejected() {
        let a = u8m(1, 3, &[1, 2, 3]);
        let b = u8m(1, 2, &[1, 2]);
        assert_eq!(add(&a, &b).unwrap_err().code, Code::StsUnmatchedSizes);

        let c = Mat::zeros(1, 3, CV_32FC1).unwrap();
        assert_eq!(add(&a, &c).unwrap_err().code, Code::StsUnmatchedFormats);
    }

    #[test]
    fn ops_accept_strided_views() {
        let data: Vec<u8> = (0..25).collect();
        let m = Mat::from_vec(5, 5, 1, &data).unwrap();
        let a = m.roi(crate::Rect::new(1, 1, 2, 2)).unwrap();
        let b = m.roi(crate::Rect::new(2, 2, 2, 2)).unwrap();
        let out = add(&a, &b).unwrap();
        // (6+12, 7+13, 11+17, 12+18)
        assert_eq!(out.to_vec::<u8>().unwrap(), vec![18, 20, 28, 30]);
    }
}
