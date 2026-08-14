//! A contiguous single-channel scratch buffer.
//!
//! Both the ECC solver and dense inverse search spend their whole run indexing
//! one image many times per pixel, which `Mat::at_f64` cannot serve: it is
//! fallible, it dispatches on depth, and it costs more than the arithmetic
//! around it. Each had grown its own flat buffer with the same three fields.
//! This is that buffer, once, with the element type left open because ECC
//! works in `f64` and inverse search in `f32` — and neither can be moved to
//! the other without changing what they compute.

use cv_core::{Mat, MatLike, Result};

/// A row-major `h * w` plane of `T`.
#[derive(Debug, Clone)]
pub(crate) struct Plane<T> {
    /// Rows.
    pub(crate) h: usize,
    /// Columns.
    pub(crate) w: usize,
    /// The pixels, row-major.
    pub(crate) d: Vec<T>,
}

impl<T: Copy + Default> Plane<T> {
    /// An `h * w` plane of zeros.
    pub(crate) fn zeros(h: usize, w: usize) -> Self {
        Plane { h, w, d: vec![T::default(); h * w] }
    }

    /// The pixel at `(y, x)`. Panics out of range rather than folding a
    /// border: every caller here has already clamped or padded.
    pub(crate) fn at(&self, y: usize, x: usize) -> T {
        self.d[y * self.w + x]
    }
}

impl Plane<f32> {
    /// Read a single-channel image into `f32`.
    pub(crate) fn from_mat<S: MatLike + ?Sized>(m: &S) -> Result<Self> {
        let (h, w) = (m.rows() as usize, m.cols() as usize);
        let mut p = Plane::zeros(h, w);
        for y in 0..h {
            for x in 0..w {
                p.d[y * w + x] = m.at_f64(y as i32, x as i32, 0)? as f32;
            }
        }
        Ok(p)
    }

    /// Back to a `CV_32FC1` image.
    pub(crate) fn to_mat(&self) -> Result<Mat> {
        Mat::from_vec(self.h as i32, self.w as i32, 1, &self.d)
    }
}

impl Plane<f64> {
    /// Read a single-channel image into `f64`, without the narrowing that
    /// [`Plane::<f32>::from_mat`] applies.
    pub(crate) fn from_mat<S: MatLike + ?Sized>(m: &S) -> Result<Self> {
        let (h, w) = (m.rows() as usize, m.cols() as usize);
        let mut p = Plane::zeros(h, w);
        for y in 0..h {
            for x in 0..w {
                p.d[y * w + x] = m.at_f64(y as i32, x as i32, 0)?;
            }
        }
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mat_survives_the_round_trip_through_f32() {
        let m = Mat::from_vec(2, 3, 1, &[1.5f32, -2.0, 3.25, 0.0, 7.5, -0.5]).unwrap();
        let p = Plane::<f32>::from_mat(&m).unwrap();
        assert_eq!((p.h, p.w), (2, 3));
        assert_eq!(p.at(1, 2), -0.5);
        assert_eq!(cv_core::to_f64(&p.to_mat().unwrap()).unwrap(), cv_core::to_f64(&m).unwrap());
    }

    #[test]
    fn the_f64_reader_does_not_narrow() {
        // A value that is not representable in f32: the two readers must
        // disagree here, which is the whole reason both exist.
        let v = 1.0 + 2f64.powi(-40);
        let m = Mat::from_vec(1, 1, 1, &[v]).unwrap();
        assert_eq!(Plane::<f64>::from_mat(&m).unwrap().at(0, 0), v);
        assert_eq!(Plane::<f32>::from_mat(&m).unwrap().at(0, 0), 1.0f32);
    }
}
