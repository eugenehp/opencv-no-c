//! Image <-> blob conversion (`modules/dnn/src/dnn.cpp`, `blob.cpp`).
//!
//! A *blob* is the NCHW `f32` tensor every network layer consumes: batch,
//! channel, row, column, in that nesting order. Images are HWC and usually
//! `CV_8U`, so this is where the transpose and the type change happen.

use cv_core::{cv_assert, MatLike, Result, Scalar};

/// A dense NCHW tensor.
///
/// Upstream returns a 4-D `cv::Mat`; this port uses an explicit struct because
/// a 4-D `Mat` would need N-dimensional support that the rest of the workspace
/// deliberately does not have.
#[derive(Debug, Clone, PartialEq)]
pub struct Blob {
    /// Batch, channels, height, width.
    pub shape: [usize; 4],
    /// `shape.iter().product()` values, NCHW order.
    pub data: Vec<f32>,
}

impl Blob {
    /// A zero-filled blob of the given shape.
    #[must_use]
    pub fn zeros(n: usize, c: usize, h: usize, w: usize) -> Self {
        Blob { shape: [n, c, h, w], data: vec![0.0; n * c * h * w] }
    }

    /// Wrap an existing buffer, checking it against the shape.
    pub fn new(shape: [usize; 4], data: Vec<f32>) -> Result<Self> {
        let want: usize = shape.iter().product();
        cv_assert!(data.len() == want, "blob data is {} values, expected {want}", data.len());
        Ok(Blob { shape, data })
    }

    /// Batch size.
    #[must_use]
    pub fn batch(&self) -> usize {
        self.shape[0]
    }
    /// Channel count.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.shape[1]
    }
    /// Row count.
    #[must_use]
    pub fn height(&self) -> usize {
        self.shape[2]
    }
    /// Column count.
    #[must_use]
    pub fn width(&self) -> usize {
        self.shape[3]
    }
    /// Values per image plane.
    #[must_use]
    pub fn plane(&self) -> usize {
        self.shape[2] * self.shape[3]
    }

    /// Value at `(n, c, y, x)`.
    #[must_use]
    pub fn at(&self, n: usize, c: usize, y: usize, x: usize) -> f32 {
        self.data[((n * self.shape[1] + c) * self.shape[2] + y) * self.shape[3] + x]
    }

    /// Mutable value at `(n, c, y, x)`.
    pub fn at_mut(&mut self, n: usize, c: usize, y: usize, x: usize) -> &mut f32 {
        let i = ((n * self.shape[1] + c) * self.shape[2] + y) * self.shape[3] + x;
        &mut self.data[i]
    }
}

/// How [`blob_from_image`] fits a source image to the requested size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobResize {
    /// Stretch to exactly `size`, ignoring aspect ratio — upstream's `crop=false`.
    Stretch,
    /// Scale so the short side fits, then centre-crop — upstream's `crop=true`.
    CenterCrop,
}

/// `cv::dnn::blobFromImage(image, scalefactor, size, mean, swapRB, crop)`.
///
/// # Order of operations
///
/// Upstream applies these in a specific order and getting it wrong is a silent
/// accuracy bug rather than a crash: **swap R and B first, then subtract the
/// mean, then multiply by the scale factor**. With `mean = (1,2,3)`,
/// `scale = 1/255`, `swapRB = true` and a BGR pixel of `(0,1,2)`, cv2 4.13.0
/// returns `(0.00392157, -0.00392157, -0.01176471)`: the channels are swapped
/// to `(2,1,0)`, the mean is taken off to give `(1,-1,-3)`, and only then is
/// the scale applied. Subtracting the mean after scaling would give a
/// completely different tensor.
pub fn blob_from_image<S: MatLike + ?Sized>(
    image: &S,
    scale_factor: f64,
    size: Option<(i32, i32)>,
    mean: Scalar,
    swap_rb: bool,
    crop: BlobResize,
) -> Result<Blob> {
    blob_from_images(std::slice::from_ref(&image), scale_factor, size, mean, swap_rb, crop)
}

/// `cv::dnn::blobFromImages(images, ...)` — the batched form.
///
/// Every image must share a channel count; sizes may differ because each is
/// resized to `size` first.
pub fn blob_from_images<S: MatLike + ?Sized>(
    images: &[&S],
    scale_factor: f64,
    size: Option<(i32, i32)>,
    mean: Scalar,
    swap_rb: bool,
    crop: BlobResize,
) -> Result<Blob> {
    cv_assert!(!images.is_empty(), "blobFromImages needs at least one image");
    let cn = images[0].channels() as usize;
    for (i, im) in images.iter().enumerate() {
        cv_assert!(
            im.channels() as usize == cn,
            "image {i} has {} channels, expected {cn}",
            im.channels()
        );
    }
    let (tw, th) = match size {
        Some((w, h)) => {
            cv_assert!(w > 0 && h > 0, "blob size must be positive, got {w}x{h}");
            (w, h)
        }
        None => (images[0].cols(), images[0].rows()),
    };

    let n = images.len();
    let mut blob = Blob::zeros(n, cn, th as usize, tw as usize);
    for (bi, im) in images.iter().enumerate() {
        let resized = fit(*im, tw, th, crop)?;
        for y in 0..th {
            for x in 0..tw {
                for c in 0..cn {
                    // swapRB exchanges channel 0 and channel 2 of a 3- or
                    // 4-channel image; it leaves alpha and grayscale alone.
                    let src_c = if swap_rb && cn >= 3 && (c == 0 || c == 2) { 2 - c } else { c };
                    let v = resized.at_f64(y, x, src_c as i32)?;
                    let m = mean.0[c.min(3)];
                    *blob.at_mut(bi, c, y as usize, x as usize) = ((v - m) * scale_factor) as f32;
                }
            }
        }
    }
    Ok(blob)
}

/// Resize one image to `(tw, th)` under the chosen fit rule.
fn fit<S: MatLike + ?Sized>(
    im: &S,
    tw: i32,
    th: i32,
    crop: BlobResize,
) -> Result<cv_core::Mat> {
    use cv_core::Size;
    use cv_imgproc::resize::{resize, Interpolation};
    match crop {
        BlobResize::Stretch => {
            if im.cols() == tw && im.rows() == th {
                cv_core::Mat::copy_from(im)
            } else {
                resize(im, Size::new(tw, th), 0.0, 0.0, Interpolation::Linear)
            }
        }
        BlobResize::CenterCrop => {
            // Scale so the *short* side reaches the target, then take the middle.
            let sx = tw as f64 / im.cols() as f64;
            let sy = th as f64 / im.rows() as f64;
            let s = sx.max(sy);
            let (rw, rh) = (
                (im.cols() as f64 * s).round().max(tw as f64) as i32,
                (im.rows() as f64 * s).round().max(th as f64) as i32,
            );
            let scaled = resize(im, Size::new(rw, rh), 0.0, 0.0, Interpolation::Linear)?;
            let (x0, y0) = ((rw - tw) / 2, (rh - th) / 2);
            let mut out = cv_core::Mat::zeros(th, tw, im.mat_type())?;
            for y in 0..th {
                for x in 0..tw {
                    for c in 0..im.channels() {
                        let v = scaled.at_f64(y0 + y, x0 + x, c)?;
                        cv_core::arithm::set_channel_generic(&mut out, y, x, c, v)?;
                    }
                }
            }
            Ok(out)
        }
    }
}

/// `cv::dnn::imagesFromBlob(blob_, images_)` — the inverse of [`blob_from_images`].
///
/// No scale or mean is undone; upstream does not either. The result is
/// `CV_32F` with `blob.channels()` channels.
pub fn images_from_blob(blob: &Blob) -> Result<Vec<cv_core::Mat>> {
    use cv_core::types::{Depth, MatType};
    let mut out = Vec::with_capacity(blob.batch());
    for n in 0..blob.batch() {
        let mut m = cv_core::Mat::zeros(
            blob.height() as i32,
            blob.width() as i32,
            MatType::new(Depth::F32, blob.channels() as i32),
        )?;
        for y in 0..blob.height() {
            for x in 0..blob.width() {
                for c in 0..blob.channels() {
                    cv_core::arithm::set_channel_generic(
                        &mut m,
                        y as i32,
                        x as i32,
                        c as i32,
                        f64::from(blob.at(n, c, y, x)),
                    )?;
                }
            }
        }
        out.push(m);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::Mat;

    /// A 3x3 BGR image whose pixel (y,x) channel c is `(y*3 + x)*3 + c`.
    fn ramp() -> Mat {
        let v: Vec<u8> = (0..27).map(|i| i as u8).collect();
        Mat::from_vec(3, 3, 3, &v).unwrap()
    }

    #[test]
    fn blob_layout_is_nchw() {
        let b = blob_from_image(&ramp(), 1.0, None, Scalar::all(0.0), false, BlobResize::Stretch)
            .unwrap();
        assert_eq!(b.shape, [1, 3, 3, 3]);
        // cv2 4.13.0: blob[0,:,0,0] == [0, 1, 2], the first pixel's channels.
        assert_eq!(b.at(0, 0, 0, 0), 0.0);
        assert_eq!(b.at(0, 1, 0, 0), 1.0);
        assert_eq!(b.at(0, 2, 0, 0), 2.0);
        // Second pixel of the first row is 3,4,5.
        assert_eq!(b.at(0, 0, 0, 1), 3.0);
        assert_eq!(b.at(0, 2, 0, 1), 5.0);
    }

    #[test]
    fn swap_rb_then_mean_then_scale() {
        // Pinned to cv2 4.13.0: blobFromImage(im, 1/255, (3,3), (1,2,3), True, False)
        // gives [0.00392157, -0.00392157, -0.01176471] at pixel (0,0).
        let b = blob_from_image(
            &ramp(),
            1.0 / 255.0,
            Some((3, 3)),
            Scalar::new(1.0, 2.0, 3.0, 0.0),
            true,
            BlobResize::Stretch,
        )
        .unwrap();
        let got = [b.at(0, 0, 0, 0), b.at(0, 1, 0, 0), b.at(0, 2, 0, 0)];
        let want = [0.003_921_57_f32, -0.003_921_57, -0.011_764_71];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "got {got:?}, cv2 gives {want:?}");
        }
    }

    #[test]
    fn mean_is_not_applied_after_the_scale() {
        // The distinguishing case: if the order were (v*scale - mean) the first
        // channel would be 0*0.5 - 1 = -1, not (0 - 1)*0.5 = -0.5.
        let b = blob_from_image(
            &ramp(),
            0.5,
            None,
            Scalar::new(1.0, 0.0, 0.0, 0.0),
            false,
            BlobResize::Stretch,
        )
        .unwrap();
        assert_eq!(b.at(0, 0, 0, 0), -0.5);
    }

    #[test]
    fn batched_blob_stacks_along_n() {
        let a = ramp();
        let b = ramp();
        let blob =
            blob_from_images(&[&a, &b], 1.0, None, Scalar::all(0.0), false, BlobResize::Stretch)
                .unwrap();
        assert_eq!(blob.shape, [2, 3, 3, 3]);
        assert_eq!(blob.at(0, 0, 0, 0), blob.at(1, 0, 0, 0));
    }

    #[test]
    fn grayscale_is_single_channel_and_ignores_swap_rb() {
        let g = Mat::from_vec(2, 2, 1, &[10u8, 20, 30, 40]).unwrap();
        let b = blob_from_image(&g, 1.0, None, Scalar::all(0.0), true, BlobResize::Stretch).unwrap();
        assert_eq!(b.shape, [1, 1, 2, 2]);
        assert_eq!(b.at(0, 0, 1, 1), 40.0);
    }

    #[test]
    fn center_crop_keeps_the_middle_of_a_wide_image() {
        // 4 wide, 2 tall, cropped to 2x2: the two middle columns survive.
        let m = Mat::from_vec(2, 4, 1, &[1u8, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let b =
            blob_from_image(&m, 1.0, Some((2, 2)), Scalar::all(0.0), false, BlobResize::CenterCrop)
                .unwrap();
        assert_eq!(b.shape, [1, 1, 2, 2]);
        assert_eq!(b.at(0, 0, 0, 0), 2.0);
        assert_eq!(b.at(0, 0, 0, 1), 3.0);
        assert_eq!(b.at(0, 0, 1, 0), 6.0);
    }

    #[test]
    fn stretch_resizes_to_the_requested_size() {
        let b = blob_from_image(&ramp(), 1.0, Some((6, 6)), Scalar::all(0.0), false, BlobResize::Stretch)
            .unwrap();
        assert_eq!(b.shape, [1, 3, 6, 6]);
    }

    #[test]
    fn images_from_blob_round_trips() {
        let b = blob_from_image(&ramp(), 1.0, None, Scalar::all(0.0), false, BlobResize::Stretch)
            .unwrap();
        let ims = images_from_blob(&b).unwrap();
        assert_eq!(ims.len(), 1);
        assert_eq!((ims[0].rows(), ims[0].cols(), ims[0].channels()), (3, 3, 3));
        for c in 0..3 {
            assert_eq!(ims[0].at_f64(0, 0, c).unwrap(), f64::from(c));
        }
    }

    #[test]
    fn mismatched_channel_counts_are_rejected() {
        let a = ramp();
        let g = Mat::from_vec(3, 3, 1, &[0u8; 9]).unwrap();
        // `MatLike` is not dyn-compatible, so the mismatch is expressed with
        // two same-typed values whose channel counts differ.
        let e = blob_from_images(&[&a, &g], 1.0, None, Scalar::all(0.0), false, BlobResize::Stretch);
        assert!(e.is_err(), "a 3-channel and a 1-channel image cannot share a blob");
    }
}
