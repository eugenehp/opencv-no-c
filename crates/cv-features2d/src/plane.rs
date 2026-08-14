//! The `f32` scratch buffer the scale-space detectors are built on.
//!
//! KAZE, AKAZE and SIFT all build a pyramid of single-channel `f32` images and
//! then index them several times per pixel. `Mat::at_f64` is the wrong shape
//! for that — fallible, depth-dispatched, and more expensive than the
//! arithmetic it feeds — so each had grown the same three-field buffer
//! separately. This is that buffer, once.

use cv_core::{MatLike, Result};

/// A row-major `h * w` plane of `f32`.
#[derive(Debug, Clone)]
pub(crate) struct Plane {
    /// Columns.
    pub(crate) w: usize,
    /// Rows.
    pub(crate) h: usize,
    /// The pixels, row-major.
    pub(crate) d: Vec<f32>,
}

impl Plane {
    /// An `h * w` plane of zeros.
    pub(crate) fn zeros(h: usize, w: usize) -> Self {
        Plane { w, h, d: vec![0.0; w * h] }
    }

    /// The pixel at `(y, x)`. Panics out of range: the callers here either
    /// clamp first or stay inside a margin.
    pub(crate) fn get(&self, y: usize, x: usize) -> f32 {
        self.d[y * self.w + x]
    }

    /// Read a single-channel image, **scaled into `0..1`**, which is the range
    /// KAZE's diffusion constants are written for.
    pub(crate) fn from_image<S: MatLike + ?Sized>(img: &S) -> Result<Self> {
        let (h, w) = (img.rows() as usize, img.cols() as usize);
        let mut p = Plane::zeros(h, w);
        for y in 0..h {
            for x in 0..w {
                p.d[y * w + x] = img.at_f64(y as i32, x as i32, 0)? as f32 / 255.0;
            }
        }
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::Mat;

    #[test]
    fn from_image_scales_into_the_unit_range() {
        let m = Mat::from_vec(1, 3, 1, &[0u8, 128, 255]).unwrap();
        let p = Plane::from_image(&m).unwrap();
        assert_eq!(p.get(0, 0), 0.0);
        assert_eq!(p.get(0, 2), 1.0);
        assert!((p.get(0, 1) - 128.0 / 255.0).abs() < 1e-7);
    }

    #[test]
    fn indexing_is_row_major() {
        let mut p = Plane::zeros(2, 3);
        p.d[3 + 2] = 9.0;
        assert_eq!(p.get(1, 2), 9.0);
        assert_eq!((p.h, p.w), (2, 3));
    }
}
