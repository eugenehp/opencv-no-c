//! Seam finding, exposure compensation and blending — the compositing half of
//! the stitching pipeline (`modules/stitching/src/blenders.cpp`,
//! `exposure_compensate.cpp`, `seam_finders.cpp`).
//!
//! # Why three separate stages
//!
//! Once the images are warped into a common frame, three different artefacts
//! remain and each needs its own fix:
//!
//! * the overlap is **brighter in one image than the other**, because the
//!   camera metered them differently — [`ExposureCompensator`];
//! * a moving object or parallax makes the two disagree about *content*, so
//!   the cut has to go somewhere they agree — `SeamFinder`;
//! * even a perfect cut is visible as a hard edge — [`Blender`].
//!
//! Applying only the last is the common mistake: feathering across an exposure
//! step produces a soft bright band instead of a hard one.

use cv_core::{cv_assert, Mat, MatLike, MatLikeMut, Rect, Result};

/// `cv::detail::Blender` kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlenderType {
    /// `cv::detail::Blender::NO` — hard overwrite, for checking alignment.
    No,
    /// `cv::detail::Blender::FEATHER` — weight by distance from the seam.
    #[default]
    Feather,
    /// `cv::detail::Blender::MULTI_BAND` — blend each frequency band over a
    /// different width, so detail stays sharp while illumination blends wide.
    MultiBand,
}

/// `cv::detail::ExposureCompensator` kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExposureCompensatorType {
    /// `cv::detail::ExposureCompensator::NO`
    No,
    /// `cv::detail::ExposureCompensator::GAIN` — one scalar per image.
    #[default]
    Gain,
    /// `cv::detail::ExposureCompensator::GAIN_BLOCKS` — a gain per block, for
    /// vignetting that one scalar cannot describe.
    GainBlocks,
    /// `cv::detail::ExposureCompensator::CHANNELS` — a gain per colour channel.
    Channels,
    /// `cv::detail::ExposureCompensator::CHANNELS_BLOCKS`
    ChannelsBlocks,
}

/// `cv::detail::SeamFinder` kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeamFinderType {
    /// `cv::detail::SeamFinder::NO`
    No,
    /// `cv::detail::VoronoiSeamFinder` — cut at the midline. Ignores content,
    /// so a moving object can be sliced in half.
    #[default]
    Voronoi,
    /// `cv::detail::DpSeamFinder` with `COLOR`.
    DpColor,
    /// `cv::detail::DpSeamFinder` with `COLOR_GRAD`.
    DpColorGrad,
}

/// `cv::detail::WaveCorrectKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaveCorrectKind {
    /// `cv::detail::WAVE_CORRECT_HORIZ`
    #[default]
    Horiz,
    /// `cv::detail::WAVE_CORRECT_VERT`
    Vert,
    /// `cv::detail::WAVE_CORRECT_AUTO`
    Auto,
}

/// `cv::Stitcher::Mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StitcherMode {
    /// `cv::Stitcher::PANORAMA` — assumes rotation about the camera centre.
    #[default]
    Panorama,
    /// `cv::Stitcher::SCANS` — affine, for flatbed-style captures.
    Scans,
}

/// `cv::Stitcher::Status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StitcherStatus {
    /// `cv::Stitcher::OK`
    Ok,
    /// `cv::Stitcher::ERR_NEED_MORE_IMGS`
    NeedMoreImgs,
    /// `cv::Stitcher::ERR_HOMOGRAPHY_EST_FAIL`
    HomographyEstFail,
    /// `cv::Stitcher::ERR_CAMERA_PARAMS_ADJUST_FAIL`
    CameraParamsAdjustFail,
}

/// `cv::detail::ExposureCompensator::feed(corners, images, masks)` and
/// `apply(index, corner, image, mask)`.
///
/// Solves for one gain per image so that, wherever two overlap, their scaled
/// intensities agree. Upstream's least-squares system is the same idea; this
/// normalises so the mean gain is 1, which keeps the panorama's overall
/// brightness where the inputs put it rather than drifting with image order.
#[derive(Debug, Clone, PartialEq)]
pub struct ExposureCompensator {
    kind: ExposureCompensatorType,
    gains: Vec<f64>,
}

impl ExposureCompensator {
    /// A compensator of the given kind, before any images are fed.
    #[must_use]
    pub fn new(kind: ExposureCompensatorType) -> Self {
        Self { kind, gains: Vec::new() }
    }

    /// `cv::detail::ExposureCompensator::getMatGains(umv)`.
    #[must_use]
    pub fn gains(&self) -> &[f64] {
        &self.gains
    }

    /// Replace the solved gains.
    ///
    /// Used by [`crate::pipeline::ExposureCompensatorTrait::set_mat_gains`], so
    /// a caller who has gains from an earlier run can install them instead of
    /// re-solving — which is what keeps a video panorama from flickering.
    pub fn set_gains(&mut self, gains: Vec<f64>) {
        self.gains = gains;
    }

    /// `cv::detail::ExposureCompensator::feed(corners, images, masks)`.
    pub fn feed(&mut self, corners: &[(i32, i32)], images: &[&Mat]) -> Result<()> {
        cv_assert!(
            corners.len() == images.len(),
            "feed got {} corners for {} images",
            corners.len(),
            images.len()
        );
        let n = images.len();
        cv_assert!(n > 0, "at least one image is needed");
        if self.kind == ExposureCompensatorType::No {
            self.gains = vec![1.0; n];
            return Ok(());
        }

        // Mean intensity over each pair's overlap, and the overlap size.
        // `N(i, i)` is the image's own pixel count, which is what carries the
        // `beta` regularisation for an image that overlaps nothing.
        let mut i_mean = vec![vec![0.0f64; n]; n];
        let mut i_count = vec![vec![0usize; n]; n];
        for a in 0..n {
            for b in 0..n {
                let (sum, cnt) = overlap_mean(corners[a], images[a], corners[b], images[b])?;
                // `max(1, count)`, upstream's, so a pair with no overlap still
                // contributes its `beta` term rather than dividing by zero.
                i_count[a][b] = cnt.max(1);
                i_mean[a][b] = sum / i_count[a][b] as f64;
            }
        }

        // Upstream's system, with its constants:
        //   b(i)   += beta * N(i, j)
        //   A(i,i) += beta * N(i, j) + 2 alpha I(i,j)^2 N(i,j)
        //   A(i,j) -= 2 alpha I(i,j) I(j,i) N(i,j)
        // `beta` is what pulls the solution towards 1 — it is the same idea as
        // the unit-mean tie-break this used to apply afterwards, but it is
        // *inside* the least-squares problem and weighted by the overlap size,
        // so it does not simply rescale the answer. Normalising the solution to
        // unit mean afterwards is therefore wrong: cv2's gains here are
        // [1.149, 0.754], whose mean is 0.95.
        const ALPHA: f64 = 0.01;
        const BETA: f64 = 100.0;
        let mut a_mat = vec![0.0f64; n * n];
        let mut rhs = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                let w = i_count[i][j] as f64;
                rhs[i] += BETA * w;
                a_mat[i * n + i] += BETA * w;
                if i != j {
                    a_mat[i * n + i] += 2.0 * ALPHA * i_mean[i][j] * i_mean[i][j] * w;
                    a_mat[i * n + j] -= 2.0 * ALPHA * i_mean[i][j] * i_mean[j][i] * w;
                }
            }
        }
        self.gains = solve_dense(&mut a_mat, &mut rhs, n).unwrap_or_else(|| vec![1.0; n]);
        Ok(())
    }

    /// `cv::detail::ExposureCompensator::apply(index, corner, image, mask)`.
    pub fn apply(&self, index: usize, image: &Mat) -> Result<Mat> {
        cv_assert!(index < self.gains.len(), "no gain for image {index}; call feed first");
        let g = self.gains[index];
        let mut out = Mat::zeros(image.rows(), image.cols(), image.mat_type())?;
        for y in 0..image.rows() {
            for x in 0..image.cols() {
                for c in 0..image.channels() {
                    let v = (image.at_f64(y, x, c)? * g).clamp(0.0, 255.0);
                    cv_core::arithm::set_channel_generic(&mut out, y, x, c, v)?;
                }
            }
        }
        Ok(out)
    }
}

/// Mean intensity of `a` over its overlap with `b`, and the pixel count.
fn overlap_mean(
    ca: (i32, i32),
    a: &Mat,
    cb: (i32, i32),
    b: &Mat,
) -> Result<(f64, usize)> {
    let ra = Rect::new(ca.0, ca.1, a.cols(), a.rows());
    let rb = Rect::new(cb.0, cb.1, b.cols(), b.rows());
    let x0 = ra.x.max(rb.x);
    let y0 = ra.y.max(rb.y);
    let x1 = (ra.x + ra.width).min(rb.x + rb.width);
    let y1 = (ra.y + ra.height).min(rb.y + rb.height);
    if x1 <= x0 || y1 <= y0 {
        return Ok((0.0, 0));
    }
    let mut sum = 0.0;
    let mut n = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            // Upstream sums `norm(Vec3b)` — the pixel's *L2 norm across
            // channels*, not its mean. On a mid-grey pixel the two differ by
            // a factor of sqrt(3), and since the gain system is quadratic in
            // `I` that factor does not cancel.
            let mut px = 0.0;
            for c in 0..a.channels() {
                let v = a.at_f64(y - ca.1, x - ca.0, c)?;
                px += v * v;
            }
            sum += px.sqrt();
            n += 1;
        }
    }
    Ok((sum, n))
}

fn solve_dense(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r * n + col].abs() > a[piv * n + col].abs() {
                piv = r;
            }
        }
        if a[piv * n + col].abs() < 1e-12 {
            return None;
        }
        for k in 0..n {
            a.swap(col * n + k, piv * n + k);
        }
        b.swap(col, piv);
        for r in (col + 1)..n {
            let f = a[r * n + col] / a[col * n + col];
            for k in col..n {
                a[r * n + k] -= f * a[col * n + k];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let s: f64 = ((i + 1)..n).map(|j| a[i * n + j] * x[j]).sum();
        x[i] = (b[i] - s) / a[i * n + i];
    }
    Some(x)
}

/// `cv::detail::Blender` — composes warped images into one canvas.
/// `cv::detail::Blender::WEIGHT_EPS`.
const WEIGHT_EPS: f64 = 1e-5;

/// `cv::detail::Blender` — compose warped images onto one canvas.
#[derive(Debug, Clone)]
pub struct Blender {
    kind: BlenderType,
    /// Feather width, in pixels.
    sharpness: f64,
    /// Octaves the multi-band blender splits each image into.
    num_bands: i32,
    /// Images fed so far, with their masks and top-left corners. The blend
    /// happens in `blend`, because a multi-band blend cannot be accumulated
    /// pixel by pixel — it needs each image's whole pyramid.
    fed: Vec<(Mat, Mat, (i32, i32))>,
    canvas: Rect,
}

impl Blender {
    /// `cv::detail::Blender::createDefault(type, try_gpu)`.
    #[must_use]
    pub fn new(kind: BlenderType) -> Self {
        Self {
            kind,
            sharpness: 0.02,
            num_bands: 5,
            fed: Vec::new(),
            canvas: Rect::new(0, 0, 0, 0),
        }
    }

    /// `cv::detail::FeatherBlender::setSharpness(val)`.
    pub fn set_sharpness(&mut self, v: f64) -> &mut Self {
        self.sharpness = v;
        self
    }

    /// `cv::detail::MultiBandBlender::numBands()`.
    ///
    /// How many octaves the multi-band blender splits each image into. More
    /// bands blend illumination over a wider region while keeping detail
    /// sharp; past about five the coarsest band is a handful of pixels and
    /// there is nothing left to gain.
    #[must_use]
    pub fn num_bands(&self) -> i32 {
        self.num_bands
    }

    /// `cv::detail::MultiBandBlender::setNumBands(val)`.
    pub fn set_num_bands(&mut self, val: i32) -> &mut Self {
        self.num_bands = val;
        self
    }

    /// `cv::detail::FeatherBlender::sharpness()`.
    #[must_use]
    pub fn sharpness(&self) -> f64 {
        self.sharpness
    }

    /// `cv::detail::Blender::prepare(dst_roi)`.
    pub fn prepare(&mut self, roi: Rect) -> Result<()> {
        cv_assert!(roi.width > 0 && roi.height > 0, "the canvas must be non-empty");
        self.canvas = roi;
        self.fed.clear();
        Ok(())
    }

    /// `cv::detail::Blender::feed(img, mask, tl)` with a fully-valid image.
    ///
    /// Equivalent to passing an all-255 mask, which is what a warped image
    /// covering its whole rectangle has.
    pub fn feed(&mut self, img: &Mat, tl: (i32, i32)) -> Result<()> {
        cv_assert!(
            self.canvas.width > 0 && self.canvas.height > 0,
            "call prepare before feed"
        );
        cv_assert!(img.channels() == 3, "the blender composes 3-channel images");
        let mut mask = Mat::zeros(img.rows(), img.cols(), cv_core::types::CV_8UC1)?;
        for y in 0..img.rows() {
            for x in 0..img.cols() {
                mask.pixel_mut::<u8>(y, x)?[0] = 255;
            }
        }
        self.feed_with_mask(img, &mask, tl)
    }

    /// `cv::detail::Blender::feed(img, mask, tl)`.
    ///
    /// The mask is what the weighting is built from — not the image rectangle.
    /// A warped image is a quadrilateral inside a rectangle, and the pixels
    /// outside it must not vote.
    pub fn feed_with_mask(&mut self, img: &Mat, mask: &Mat, tl: (i32, i32)) -> Result<()> {
        cv_assert!(
            self.canvas.width > 0 && self.canvas.height > 0,
            "call prepare before feed"
        );
        cv_assert!(img.channels() == 3, "the blender composes 3-channel images");
        cv_assert!(mask.channels() == 1, "the mask is single-channel");
        cv_assert!(
            mask.rows() == img.rows() && mask.cols() == img.cols(),
            "the mask must match the image"
        );
        self.fed.push((Mat::copy_from(img)?, Mat::copy_from(mask)?, tl));
        Ok(())
    }

    /// Lay one fed image and its weight onto canvas-sized planes.
    fn on_canvas(&self, img: &Mat, mask: &Mat, tl: (i32, i32)) -> Result<(Vec<f64>, Vec<f64>)> {
        let (cw, ch) = (self.canvas.width as usize, self.canvas.height as usize);
        self.on_canvas_sized(img, mask, tl, ch, cw)
    }

    /// The same onto a canvas expanded to `ch x cw`.
    fn on_canvas_sized(
        &self,
        img: &Mat,
        mask: &Mat,
        tl: (i32, i32),
        ch: usize,
        cw: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut plane = vec![0.0f64; cw * ch * 3];
        let mut wgt = vec![0.0f64; cw * ch];
        let wm = crate::detail::create_weight_map(mask, self.sharpness)?;
        // Outside the image the plane is filled by *replication*, not zeros.
        // A multi-band blend takes a pyramid of this plane, and a zero border
        // is a cliff the coarse levels smear back over the seam — which shows
        // up as one image's content bleeding into a region only the other
        // covers. The weight is still zero out there, so nothing is voted for
        // that should not be.
        for y in 0..ch as i32 {
            for x in 0..cw as i32 {
                let sy = (y + self.canvas.y - tl.1).clamp(0, img.rows() - 1);
                let sx = (x + self.canvas.x - tl.0).clamp(0, img.cols() - 1);
                let i = y as usize * cw + x as usize;
                for c in 0..3 {
                    plane[i * 3 + c] = img.at_f64(sy, sx, c as i32)?;
                }
            }
        }
        for y in 0..img.rows() {
            for x in 0..img.cols() {
                let (dx, dy) = (tl.0 + x - self.canvas.x, tl.1 + y - self.canvas.y);
                if dx < 0 || dy < 0 || dx as usize >= cw || dy as usize >= ch {
                    continue;
                }
                let i = dy as usize * cw + dx as usize;
                wgt[i] = wm.at_f64(y, x, 0)?;
                for c in 0..3 {
                    plane[i * 3 + c] = img.at_f64(y, x, c as i32)?;
                }
            }
        }
        Ok((plane, wgt))
    }

    /// `cv::detail::Blender::blend(dst, dst_mask)`.
    ///
    /// Returns the composed image and the coverage mask. A pixel no image
    /// reached is left black and marked absent, rather than divided by a zero
    /// weight.
    pub fn blend(&self) -> Result<(Mat, Mat)> {
        let (cw, ch) = (self.canvas.width as usize, self.canvas.height as usize);
        cv_assert!(cw > 0 && ch > 0, "call prepare before blend");
        let mut acc = vec![0.0f64; cw * ch * 3];
        let mut wsum = vec![0.0f64; cw * ch];

        if self.kind == BlenderType::MultiBand {

            // Blend each frequency band over a different width: the coarse
            // bands blend across the whole overlap, so an exposure step
            // disappears, while the fine bands blend over a few pixels, so
            // detail stays sharp. Averaging every frequency over the same
            // width — which is what a feather blend does — either leaves a
            // visible step or smears the texture, and cannot do both.
            let bands = self.num_bands.max(1) as usize;
            // Upstream expands the canvas so each side divides by `2^bands`
            // and crops back at the end. Measured here it makes the result
            // *worse* against cv2 (max error 17 -> 42), so the canvas is used
            // as given; `pyr_up_to` already reconstructs an odd level exactly.
            let (ew, eh) = (cw, ch);
            let mut band_acc: Vec<Vec<f64>> = Vec::new();
            let mut band_w: Vec<Vec<f64>> = Vec::new();
            let mut sizes: Vec<(usize, usize)> = Vec::new();

            for (img, mask, tl) in &self.fed {
                let (plane, wgt) = self.on_canvas_sized(img, mask, *tl, eh, ew)?;
                let lap = laplacian_pyramid(&plane, eh, ew, 3, bands)?;
                let gw = gaussian_pyramid(&wgt, eh, ew, 1, bands)?;
                if band_acc.is_empty() {
                    for (l, _) in lap.iter().enumerate() {
                        sizes.push((lap[l].1, lap[l].2));
                        band_acc.push(vec![0.0; lap[l].1 * lap[l].2 * 3]);
                        band_w.push(vec![0.0; lap[l].1 * lap[l].2]);
                    }
                }
                // Accumulate only inside this image's own rectangle, scaled
                // to the level. `pyrDown` of a 0/1 weight spreads it a little
                // further at every level, so without this restriction one
                // image's coarse band votes across a region only the other
                // covers — which shows up as its content bleeding in.
                let rx0 = tl.0 - self.canvas.x;
                let ry0 = tl.1 - self.canvas.y;
                for l in 0..lap.len() {
                    let (v, bh, bw) = &lap[l];
                    let (g, ..) = &gw[l];
                    let step: i32 = 1 << l;
                    let (lx0, ly0) = ((rx0 / step).max(0), (ry0 / step).max(0));
                    let lx1 = ((rx0 + img.cols() + step - 1) / step).min(*bw as i32);
                    let ly1 = ((ry0 + img.rows() + step - 1) / step).min(*bh as i32);
                    for y in ly0..ly1 {
                        for x in lx0..lx1 {
                            let i = (y * *bw as i32 + x) as usize;
                            band_w[l][i] += g[i];
                            for c in 0..3 {
                                band_acc[l][i * 3 + c] += v[i * 3 + c] * g[i];
                            }
                        }
                    }
                }
            }
            if band_acc.is_empty() {
                return self.finish(&acc, &wsum);
            }
            // Normalise each band, then collapse.
            for l in 0..band_acc.len() {
                for i in 0..band_w[l].len() {
                    let w = band_w[l][i];
                    if w > 1e-12 {
                        for c in 0..3 {
                            band_acc[l][i * 3 + c] /= w;
                        }
                    }
                }
            }
            // Collapse, then crop back to the requested canvas.
            let full = collapse_pyramid(&band_acc, &sizes, 3)?;
            for y in 0..ch {
                for x in 0..cw {
                    for c in 0..3 {
                        acc[(y * cw + x) * 3 + c] = full[(y * ew + x) * 3 + c];
                    }
                }
            }
            // Coverage: any image that reached the pixel.
            for (img, mask, tl) in &self.fed {
                let (_, wgt) = self.on_canvas(img, mask, *tl)?;
                for i in 0..cw * ch {
                    wsum[i] += wgt[i];
                }
            }
            return self.finish_multiband(&acc, &wsum);
        }

        for (img, mask, tl) in &self.fed {
            let (plane, wgt) = self.on_canvas(img, mask, *tl)?;
            for i in 0..cw * ch {
                let w = if self.kind == BlenderType::No {
                    if wgt[i] > 0.0 {
                        // A hard overwrite: the last image to cover a pixel wins.
                        wsum[i] = 0.0;
                        for c in 0..3 {
                            acc[i * 3 + c] = 0.0;
                        }
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    wgt[i]
                };
                if w <= 0.0 {
                    continue;
                }
                wsum[i] += w;
                for c in 0..3 {
                    acc[i * 3 + c] += plane[i * 3 + c] * w;
                }
            }
        }
        self.finish(&acc, &wsum)
    }

    /// Divide by the accumulated weight and pack into a `CV_8UC3`.
    fn finish(&self, acc: &[f64], wsum: &[f64]) -> Result<(Mat, Mat)> {
        let (w, h) = (self.canvas.width, self.canvas.height);
        let mut out = Mat::zeros(h, w, cv_core::types::MatType::new(cv_core::types::Depth::U8, 3))?;
        let mut mask =
            Mat::zeros(h, w, cv_core::types::MatType::new(cv_core::types::Depth::U8, 1))?;
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                if wsum[i] <= 0.0 {
                    continue;
                }
                mask.pixel_mut::<u8>(y, x)?[0] = 255;
                for c in 0..3 {
                    // `static_cast<short>` truncates toward zero, and the
                    // denominator carries upstream's `WEIGHT_EPS`. Rounding
                    // instead is a level high on essentially every pixel.
                    let v = (acc[i * 3 + c] / (wsum[i] + WEIGHT_EPS)).clamp(0.0, 255.0);
                    out.pixel_mut::<u8>(y, x)?[c] = v.trunc() as u8;
                }
            }
        }
        Ok((out, mask))
    }

    /// The same, for a pyramid result that is already normalised.
    fn finish_multiband(&self, acc: &[f64], wsum: &[f64]) -> Result<(Mat, Mat)> {
        let (w, h) = (self.canvas.width, self.canvas.height);
        let mut out = Mat::zeros(h, w, cv_core::types::MatType::new(cv_core::types::Depth::U8, 3))?;
        let mut mask =
            Mat::zeros(h, w, cv_core::types::MatType::new(cv_core::types::Depth::U8, 1))?;
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                if wsum[i] <= 0.0 {
                    continue;
                }
                mask.pixel_mut::<u8>(y, x)?[0] = 255;
                for c in 0..3 {
                    out.pixel_mut::<u8>(y, x)?[c] =
                        acc[i * 3 + c].clamp(0.0, 255.0).trunc() as u8;
                }
            }
        }
        Ok((out, mask))
    }
}

/// A Gaussian pyramid of a flat plane, as `(data, rows, cols)` per level.
fn gaussian_pyramid(
    v: &[f64],
    h: usize,
    w: usize,
    cn: usize,
    levels: usize,
) -> Result<Vec<(Vec<f64>, usize, usize)>> {
    let mut out = vec![(v.to_vec(), h, w)];
    for _ in 0..levels {
        let (ref cur, ch, cw) = out[out.len() - 1];
        if ch < 2 || cw < 2 {
            break;
        }
        let m = cv_core::mat_from_f64(
            ch as i32,
            cw as i32,
            cv_core::types::MatType::new(cv_core::types::Depth::F64, cn as i32),
            cur,
        )?;
        let d = cv_imgproc::pyramids::pyr_down(&m, cv_core::BorderType::Reflect101)?;
        out.push((cv_core::to_f64(&d)?, d.rows() as usize, d.cols() as usize));
    }
    Ok(out)
}

/// A Laplacian pyramid: each level minus the upsampled level above it, with the
/// coarsest level kept whole so the pyramid can be collapsed exactly.
fn laplacian_pyramid(
    v: &[f64],
    h: usize,
    w: usize,
    cn: usize,
    levels: usize,
) -> Result<Vec<(Vec<f64>, usize, usize)>> {
    let g = gaussian_pyramid(v, h, w, cn, levels)?;
    let mut out = g.clone();
    for l in 0..g.len() - 1 {
        let (ref up, uh, uw) = g[l + 1];
        let m = cv_core::mat_from_f64(
            uh as i32,
            uw as i32,
            cv_core::types::MatType::new(cv_core::types::Depth::F64, cn as i32),
            up,
        )?;
        let e = cv_imgproc::pyramids::pyr_up_to(
            &m,
            g[l].1 as i32,
            g[l].2 as i32,
            cv_core::BorderType::Reflect101,
        )?;
        let ev = cv_core::to_f64(&e)?;
        for (i, o) in out[l].0.iter_mut().enumerate() {
            *o -= ev[i];
        }
    }
    Ok(out)
}

/// Sum a Laplacian pyramid back into one plane.
fn collapse_pyramid(
    bands: &[Vec<f64>],
    sizes: &[(usize, usize)],
    cn: usize,
) -> Result<Vec<f64>> {
    let mut acc = bands[bands.len() - 1].clone();
    let (mut ah, mut aw) = sizes[sizes.len() - 1];
    for l in (0..bands.len() - 1).rev() {
        let m = cv_core::mat_from_f64(
            ah as i32,
            aw as i32,
            cv_core::types::MatType::new(cv_core::types::Depth::F64, cn as i32),
            &acc,
        )?;
        let (th, tw) = sizes[l];
        let up = cv_imgproc::pyramids::pyr_up_to(
            &m,
            th as i32,
            tw as i32,
            cv_core::BorderType::Reflect101,
        )?;
        let uv = cv_core::to_f64(&up)?;
        acc = bands[l].iter().zip(&uv).map(|(a, b)| a + b).collect();
        (ah, aw) = (th, tw);
    }
    Ok(acc)
}

/// `cv::detail::VoronoiSeamFinder::find(src, corners, masks)`.
///
/// Assigns every canvas pixel to whichever image's centre is nearest, so the
/// cut runs down the middle of each overlap. Returns one mask per image.
pub fn find_seams_voronoi(
    corners: &[(i32, i32)],
    sizes: &[(i32, i32)],
    canvas: Rect,
) -> Result<Vec<Mat>> {
    cv_assert!(corners.len() == sizes.len(), "corner and size counts must match");
    cv_assert!(!corners.is_empty(), "at least one image is needed");
    cv_assert!(canvas.width > 0 && canvas.height > 0, "the canvas must be non-empty");

    let centres: Vec<(f64, f64)> = corners
        .iter()
        .zip(sizes)
        .map(|(c, s)| (f64::from(c.0) + f64::from(s.0) / 2.0, f64::from(c.1) + f64::from(s.1) / 2.0))
        .collect();

    let mut masks = Vec::with_capacity(corners.len());
    for (i, (c, s)) in corners.iter().zip(sizes).enumerate() {
        let mut m = Mat::zeros(s.1, s.0, cv_core::types::MatType::new(cv_core::types::Depth::U8, 1))?;
        for y in 0..s.1 {
            for x in 0..s.0 {
                let (gx, gy) = (f64::from(c.0 + x), f64::from(c.1 + y));
                if gx < f64::from(canvas.x)
                    || gy < f64::from(canvas.y)
                    || gx >= f64::from(canvas.x + canvas.width)
                    || gy >= f64::from(canvas.y + canvas.height)
                {
                    continue;
                }
                // Nearest centre wins; ties go to the lower index so exactly
                // one mask claims each pixel.
                let mut best = usize::MAX;
                let mut bd = f64::MAX;
                for (j, ct) in centres.iter().enumerate() {
                    let d = (gx - ct.0).powi(2) + (gy - ct.1).powi(2);
                    if d < bd - 1e-9 {
                        bd = d;
                        best = j;
                    }
                }
                if best == i {
                    m.pixel_mut::<u8>(y, x)?[0] = 255;
                }
            }
        }
        masks.push(m);
    }
    Ok(masks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: i32, h: i32, v: u8) -> Mat {
        Mat::from_vec(h, w, 3, &vec![v; (w * h * 3) as usize]).unwrap()
    }

    #[test]
    fn the_defaults_match_upstream() {
        assert_eq!(BlenderType::default(), BlenderType::Feather);
        assert_eq!(ExposureCompensatorType::default(), ExposureCompensatorType::Gain);
        assert_eq!(SeamFinderType::default(), SeamFinderType::Voronoi);
        assert_eq!(WaveCorrectKind::default(), WaveCorrectKind::Horiz);
        assert_eq!(StitcherMode::default(), StitcherMode::Panorama);
        assert_eq!(Blender::new(BlenderType::Feather).sharpness(), 0.02);
    }

    #[test]
    fn exposure_gains_equalise_a_brightness_step() {
        // Two overlapping images of the same scene, one twice as bright.
        let a = flat(40, 20, 60);
        let b = flat(40, 20, 120);
        let mut c = ExposureCompensator::new(ExposureCompensatorType::Gain);
        c.feed(&[(0, 0), (20, 0)], &[&a, &b]).unwrap();
        let g = c.gains();
        assert_eq!(g.len(), 2);
        // The darker image gets the larger gain. Upstream does *not* normalise
        // the solution to unit mean — `beta` inside the least-squares problem
        // is what anchors the scale, and it is weighted by the overlap size.
        assert!(g[0] > g[1], "gains {g:?}: the darker image should be lifted");
        assert!(g.iter().all(|v| (0.5..2.0).contains(v)), "gains {g:?} are not near unity");
        // Applying them brings the two much closer together.
        let ga = c.apply(0, &a).unwrap().at_f64(0, 0, 0).unwrap();
        let gb = c.apply(1, &b).unwrap().at_f64(0, 0, 0).unwrap();
        assert!((ga - gb).abs() < (60.0f64 - 120.0).abs(), "{ga} vs {gb} is no closer");
    }

    #[test]
    fn the_no_compensator_leaves_everything_alone() {
        let a = flat(10, 10, 80);
        let mut c = ExposureCompensator::new(ExposureCompensatorType::No);
        c.feed(&[(0, 0)], &[&a]).unwrap();
        assert_eq!(c.gains(), &[1.0]);
        assert_eq!(c.apply(0, &a).unwrap().at_f64(3, 3, 0).unwrap(), 80.0);
    }

    #[test]
    fn exposure_validates_its_input() {
        let a = flat(8, 8, 100);
        let mut c = ExposureCompensator::new(ExposureCompensatorType::Gain);
        assert!(c.feed(&[(0, 0), (5, 0)], &[&a]).is_err(), "counts must match");
        assert!(c.feed(&[], &[]).is_err(), "an empty set is an error");
        // apply before feed has no gain to use.
        let fresh = ExposureCompensator::new(ExposureCompensatorType::Gain);
        assert!(fresh.apply(0, &a).is_err());
    }

    #[test]
    fn blending_two_tiles_covers_the_canvas() {
        let a = flat(20, 20, 50);
        let b = flat(20, 20, 200);
        let mut bl = Blender::new(BlenderType::Feather);
        bl.prepare(Rect::new(0, 0, 32, 20)).unwrap();
        bl.feed(&a, (0, 0)).unwrap();
        bl.feed(&b, (12, 0)).unwrap();
        let (out, mask) = bl.blend().unwrap();
        assert_eq!((out.rows(), out.cols()), (20, 32));
        // Everything both images reached is covered.
        for y in 2..18 {
            for x in 2..30 {
                assert_eq!(mask.at_f64(y, x, 0).unwrap(), 255.0, "gap at ({y},{x})");
            }
        }
        // Away from the overlap each side keeps its own value; in the overlap
        // the result lies between them.
        let left = out.at_f64(10, 4, 0).unwrap();
        let right = out.at_f64(10, 28, 0).unwrap();
        let mid = out.at_f64(10, 16, 0).unwrap();
        assert!(left < 70.0, "left should stay dark, got {left}");
        assert!(right > 180.0, "right should stay bright, got {right}");
        assert!(mid > left && mid < right, "the seam should blend: {left} / {mid} / {right}");
    }

    #[test]
    fn the_no_blender_overwrites_rather_than_mixing() {
        let a = flat(20, 20, 50);
        let b = flat(20, 20, 200);
        let mut bl = Blender::new(BlenderType::No);
        bl.prepare(Rect::new(0, 0, 32, 20)).unwrap();
        bl.feed(&a, (0, 0)).unwrap();
        bl.feed(&b, (12, 0)).unwrap();
        let (out, _) = bl.blend().unwrap();
        // In the overlap the second image wins outright — no intermediate
        // value. 199 rather than 200 because upstream divides by
        // `weight + WEIGHT_EPS` and truncates, which costs the last level.
        let mid = out.at_f64(10, 16, 0).unwrap();
        assert!((mid - 199.0).abs() < 1.0, "expected a hard overwrite, got {mid}");
    }

    #[test]
    fn the_blender_validates_its_sequence() {
        let a = flat(8, 8, 100);
        let mut bl = Blender::new(BlenderType::Feather);
        // feed and blend before prepare are errors, not silent no-ops.
        assert!(bl.feed(&a, (0, 0)).is_err());
        assert!(bl.blend().is_err());
        assert!(bl.prepare(Rect::new(0, 0, 0, 10)).is_err());
        bl.prepare(Rect::new(0, 0, 10, 10)).unwrap();
        let grey = Mat::from_vec(8, 8, 1, &[100u8; 64]).unwrap();
        assert!(bl.feed(&grey, (0, 0)).is_err(), "the blender needs 3 channels");
    }

    #[test]
    fn voronoi_seams_partition_the_overlap_exactly_once() {
        let corners = [(0, 0), (16, 0)];
        let sizes = [(24, 16), (24, 16)];
        let canvas = Rect::new(0, 0, 40, 16);
        let masks = find_seams_voronoi(&corners, &sizes, canvas).unwrap();
        assert_eq!(masks.len(), 2);
        // Every canvas pixel both images cover is claimed by exactly one.
        for y in 0..16 {
            for x in 16..24 {
                let a = masks[0].at_f64(y, x, 0).unwrap() > 0.0;
                let b = masks[1].at_f64(y, x - 16, 0).unwrap() > 0.0;
                assert!(a ^ b, "({y},{x}) claimed by {a}/{b}, should be exactly one");
            }
        }
        // The left image keeps its far edge, the right keeps its own.
        assert_eq!(masks[0].at_f64(8, 2, 0).unwrap(), 255.0);
        assert_eq!(masks[1].at_f64(8, 22, 0).unwrap(), 255.0);
    }

    #[test]
    fn seam_finding_validates_its_input() {
        assert!(find_seams_voronoi(&[(0, 0)], &[], Rect::new(0, 0, 10, 10)).is_err());
        assert!(find_seams_voronoi(&[], &[], Rect::new(0, 0, 10, 10)).is_err());
        assert!(find_seams_voronoi(&[(0, 0)], &[(5, 5)], Rect::new(0, 0, 0, 10)).is_err());
    }
}
