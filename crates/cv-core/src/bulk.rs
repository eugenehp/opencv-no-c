//! Bulk conversion between matrices and flat `f64` buffers.
//!
//! The depth-generic accessors [`MatLike::at_f64`] and [`crate::set_channel`]
//! branch on the element type *per call*. That is fine at the edges of an
//! algorithm and ruinous inside one: a 7x7 `filter2D` calls `at_f64` 49 times
//! per pixel per channel, and each call re-derives a typed row slice.
//!
//! These helpers pay the depth dispatch **once** for a whole matrix, then run a
//! tight monomorphised loop. Kernels that need random access convert up front,
//! work in `f64`, and convert back — which is both faster and easier to keep
//! numerically identical to the naive version, since the arithmetic still
//! happens in `f64`.

use crate::error::{Code, Result};
use crate::mat::{Mat, MatLike, MatLikeMut};
use crate::saturate::Elem;
use crate::types::Depth;
use crate::{cv_error, Error};

/// Flatten a matrix into row-major, channel-interleaved `f64`.
///
/// The buffer has `rows * cols * channels` entries.
pub fn to_f64<M: MatLike + ?Sized>(m: &M) -> Result<Vec<f64>> {
    let n = m.total() * m.channels() as usize;
    let mut out = vec![0.0f64; n];
    let width = m.cols() as usize * m.channels() as usize;

    /// Copy one depth's rows into the output, widening as we go.
    macro_rules! fill {
        ($t:ty) => {{
            for y in 0..m.rows() {
                let row = m.row_as::<$t>(y)?;
                let dst = &mut out[y as usize * width..(y as usize + 1) * width];
                for (d, s) in dst.iter_mut().zip(row.iter()) {
                    *d = *s as f64;
                }
            }
        }};
    }

    match m.depth() {
        Depth::U8 => fill!(u8),
        Depth::S8 => fill!(i8),
        Depth::U16 => fill!(u16),
        Depth::S16 => fill!(i16),
        Depth::S32 => fill!(i32),
        Depth::F32 => fill!(f32),
        Depth::F64 => fill!(f64),
        Depth::F16 => return cv_error!(Code::StsNotImplemented, "CV_16F is not ported yet"),
    }
    Ok(out)
}

/// Write a flat `f64` buffer into an existing matrix, saturating per element.
///
/// `buf` must hold exactly `rows * cols * channels` values.
pub fn from_f64<M: MatLikeMut + ?Sized>(dst: &mut M, buf: &[f64]) -> Result<()> {
    let width = dst.cols() as usize * dst.channels() as usize;
    let want = dst.rows() as usize * width;
    if buf.len() != want {
        return Err(Error::new(
            Code::StsUnmatchedSizes,
            format!("buffer has {} values, matrix needs {want}", buf.len()),
        ));
    }

    /// Store one depth's rows, saturating on the way down.
    macro_rules! store {
        ($t:ty) => {{
            for y in 0..dst.rows() {
                let src = &buf[y as usize * width..(y as usize + 1) * width];
                let row = dst.row_as_mut::<$t>(y)?;
                for (d, s) in row.iter_mut().zip(src.iter()) {
                    *d = <$t>::from_f64(*s);
                }
            }
        }};
    }

    match dst.depth() {
        Depth::U8 => store!(u8),
        Depth::S8 => store!(i8),
        Depth::U16 => store!(u16),
        Depth::S16 => store!(i16),
        Depth::S32 => store!(i32),
        Depth::F32 => store!(f32),
        Depth::F64 => store!(f64),
        Depth::F16 => return cv_error!(Code::StsNotImplemented, "CV_16F is not ported yet"),
    }
    Ok(())
}

/// Build a fresh matrix of `typ` from a flat `f64` buffer.
pub fn mat_from_f64(rows: i32, cols: i32, typ: crate::types::MatType, buf: &[f64]) -> Result<Mat> {
    let mut m = Mat::zeros(rows, cols, typ)?;
    from_f64(&mut m, buf)?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CV_32FC1, CV_8UC1, CV_8UC3};
    use crate::Rect;

    #[test]
    fn roundtrip_preserves_u8() {
        let data: Vec<u8> = (0..24).collect();
        let m = Mat::from_vec(2, 4, 3, &data).unwrap();
        let buf = to_f64(&m).unwrap();
        assert_eq!(buf.len(), 24);
        assert_eq!(buf[0], 0.0);
        assert_eq!(buf[23], 23.0);
        let back = mat_from_f64(2, 4, CV_8UC3, &buf).unwrap();
        assert_eq!(back.to_vec::<u8>().unwrap(), data);
    }

    #[test]
    fn roundtrip_preserves_f32() {
        let data: Vec<f32> = vec![-1.5, 0.0, 2.25, 1e6];
        let m = Mat::from_vec(2, 2, 1, &data).unwrap();
        let back = mat_from_f64(2, 2, CV_32FC1, &to_f64(&m).unwrap()).unwrap();
        assert_eq!(back.to_vec::<f32>().unwrap(), data);
    }

    #[test]
    fn from_f64_saturates_like_the_scalar_path() {
        let buf = vec![-5.0, 0.4, 254.5, 300.0];
        let m = mat_from_f64(1, 4, CV_8UC1, &buf).unwrap();
        // Same rounding as saturate_cast: half-to-even, clamped.
        assert_eq!(m.to_vec::<u8>().unwrap(), vec![0, 0, 254, 255]);
    }

    #[test]
    fn agrees_with_the_generic_accessors() {
        // The whole point is to be a faster equivalent, so it must agree
        // element for element with at_f64.
        let data: Vec<i16> = (-30..30).collect();
        let m = Mat::from_vec(5, 4, 3, &data).unwrap();
        let buf = to_f64(&m).unwrap();
        let cn = m.channels() as usize;
        for y in 0..m.rows() {
            for x in 0..m.cols() {
                for c in 0..m.channels() {
                    let want = m.at_f64(y, x, c).unwrap();
                    let got = buf[(y as usize * 4 + x as usize) * cn + c as usize];
                    assert_eq!(got, want, "at {y},{x},{c}");
                }
            }
        }
    }

    #[test]
    fn works_on_a_strided_view() {
        let data: Vec<u8> = (0..25).collect();
        let m = Mat::from_vec(5, 5, 1, &data).unwrap();
        let v = m.roi(Rect::new(1, 1, 3, 2)).unwrap();
        assert_eq!(to_f64(&v).unwrap(), vec![6.0, 7.0, 8.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn wrong_length_is_rejected() {
        let mut m = Mat::zeros(2, 2, CV_8UC1).unwrap();
        assert_eq!(
            from_f64(&mut m, &[1.0, 2.0]).unwrap_err().code,
            Code::StsUnmatchedSizes
        );
    }
}
