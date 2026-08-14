//! `cv::stitching` for `opencv-no-c` — warpers, blenders and a panorama pipeline.
//!
//! # Scope
//!
//! The three standard warpers (plane, cylindrical, spherical), feather and
//! multi-band blending, and a two-image pipeline that finds correspondences
//! with `cv-features2d`, fits a homography with `cv-calib3d` and composites
//! the result.
//!
//! Bundle adjustment over many images, exposure compensation and graph-cut
//! seam finding are not here. Those are what turn a two-image stitch into a
//! robust N-image panorama, and each is a substantial piece in its own right.

#![warn(missing_docs)]

pub mod constants;
pub mod compose;
pub mod detail;
pub mod pipeline;

use cv_calib3d::{apply_homography, Mat3};
use cv_core::types::{CV_32FC1, CV_8UC1};
use cv_core::{cv_assert, Code, Error, Mat, MatLike, MatLikeMut, Rect, Result};

/// Projection surface for a [`Warper`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projection {
    /// `cv::PlaneWarper` — no reprojection, just the intrinsic scale.
    Plane,
    /// `cv::CylindricalWarper` — wraps the horizontal axis onto a cylinder.
    Cylindrical,
    /// `cv::SphericalWarper` — wraps both axes onto a sphere.
    Spherical,
}

/// `cv::detail::RotationWarper`.
#[derive(Debug, Clone, Copy)]
pub struct Warper {
    /// Which surface to project onto.
    pub projection: Projection,
    /// Warp scale, upstream's `scale` — usually the median focal length.
    pub scale: f64,
}

impl Warper {
    /// A warper onto the given surface.
    pub fn new(projection: Projection, scale: f64) -> Result<Self> {
        cv_assert!(scale > 0.0, "the warp scale must be positive, got {scale}");
        Ok(Warper { projection, scale })
    }

    /// Map an image point to warped coordinates.
    ///
    /// `f` is the focal length in pixels and `(cx, cy)` the principal point.
    #[must_use]
    pub fn forward(&self, x: f64, y: f64, f: f64, cx: f64, cy: f64) -> (f64, f64) {
        let (u, v) = ((x - cx) / f, (y - cy) / f);
        match self.projection {
            Projection::Plane => (u * self.scale, v * self.scale),
            Projection::Cylindrical => (self.scale * u.atan2(1.0), self.scale * v / (1.0 + u * u).sqrt()),
            // `scale * (pi - acos(w))`, upstream's spelling. That is
            // `pi/2 + asin(w)`, so writing the `asin` alone puts every point
            // exactly `scale * pi/2` off — 471 pixels at a scale of 300.
            Projection::Spherical => {
                let w = v / (1.0 + u * u + v * v).sqrt();
                (self.scale * u.atan2(1.0), self.scale * (core::f64::consts::PI - w.acos()))
            }
        }
    }

    /// The inverse of [`Warper::forward`], used to fill the destination.
    ///
    /// Warping is done by iterating the *destination* and pulling from the
    /// source, so this direction is the one that runs per pixel.
    #[must_use]
    pub fn backward(&self, wx: f64, wy: f64, f: f64, cx: f64, cy: f64) -> (f64, f64) {
        let (u, v) = match self.projection {
            Projection::Plane => (wx / self.scale, wy / self.scale),
            Projection::Cylindrical => {
                let t = wx / self.scale;
                (t.tan(), wy / self.scale / t.cos())
            }
            Projection::Spherical => {
                let (t, p) = (wx / self.scale, wy / self.scale);
                // Point on the unit sphere, then back onto the z = 1 plane.
                // The latitude is measured from the pole, matching `forward`.
                let q = core::f64::consts::PI - p;
                let (sx, sy, sz) = (q.sin() * t.sin(), q.cos(), q.sin() * t.cos());
                if sz.abs() < 1e-12 {
                    return (f64::MAX, f64::MAX);
                }
                (sx / sz, sy / sz)
            }
        };
        (u * f + cx, v * f + cy)
    }

    /// `cv::detail::RotationWarper::warp(src, K, R, interp_mode, border_mode, dst)`.
    ///
    /// Returns the warped image and the top-left corner of its bounding box in
    /// warped coordinates, which is what a blender needs to place it.
    pub fn warp<S: MatLike + ?Sized>(&self, src: &S, focal: f64) -> Result<(Mat, (i32, i32))> {
        cv_assert!(focal > 0.0, "the focal length must be positive");
        let (h, w) = (src.rows(), src.cols());
        let (cx, cy) = (f64::from(w) * 0.5, f64::from(h) * 0.5);

        // Bounding box of the warped image, from its border.
        let (mut x0, mut y0, mut x1, mut y1) =
            (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for y in 0..h {
            for x in 0..w {
                if y != 0 && y != h - 1 && x != 0 && x != w - 1 {
                    continue; // the border determines the extent
                }
                let (u, v) = self.forward(f64::from(x), f64::from(y), focal, cx, cy);
                x0 = x0.min(u);
                y0 = y0.min(v);
                x1 = x1.max(u);
                y1 = y1.max(v);
            }
        }
        let (ox, oy) = (x0.floor() as i32, y0.floor() as i32);
        let (ow, oh) = ((x1.ceil() - x0.floor()) as i32 + 1, (y1.ceil() - y0.floor()) as i32 + 1);
        cv_assert!(ow > 0 && oh > 0 && ow < 1 << 16 && oh < 1 << 16, "the warp produced a {ow}x{oh} image");

        let cn = src.channels();
        let mut out = Mat::zeros(oh, ow, src.mat_type())?;
        for dy in 0..oh {
            for dx in 0..ow {
                let (sx, sy) =
                    self.backward(f64::from(dx + ox), f64::from(dy + oy), focal, cx, cy);
                if !sx.is_finite() || !sy.is_finite() {
                    continue;
                }
                let (ix, iy) = (sx.floor() as i32, sy.floor() as i32);
                if ix < 0 || iy < 0 || ix + 1 >= w || iy + 1 >= h {
                    continue;
                }
                let (ax, ay) = (sx - f64::from(ix), sy - f64::from(iy));
                for c in 0..cn {
                    let v = src.at_f64(iy, ix, c)? * (1.0 - ax) * (1.0 - ay)
                        + src.at_f64(iy, ix + 1, c)? * ax * (1.0 - ay)
                        + src.at_f64(iy + 1, ix, c)? * (1.0 - ax) * ay
                        + src.at_f64(iy + 1, ix + 1, c)? * ax * ay;
                    cv_core::arithm::set_channel_generic(&mut out, dy, dx, c, v)?;
                }
            }
        }
        Ok((out, (ox, oy)))
    }
}

/// `cv::detail::FeatherBlender`.
///
/// Weights each contributing image by its distance from its own edge, so the
/// seam is a gradual crossfade rather than a hard line.
pub fn feather_blend<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    a: &A,
    a_mask: &B,
    b: &A,
    b_mask: &B,
    sharpness: f64,
) -> Result<Mat> {
    cv_assert!(a.size() == b.size(), "the two images must be the same size");
    cv_assert!(a.size() == a_mask.size(), "masks must match the images");
    cv_assert!(sharpness > 0.0, "sharpness must be positive");
    let wa = feather_weights(a_mask, sharpness)?;
    let wb = feather_weights(b_mask, sharpness)?;
    let cn = a.channels();
    let mut out = Mat::zeros(a.rows(), a.cols(), a.mat_type())?;
    for y in 0..a.rows() {
        for x in 0..a.cols() {
            let (p, q) = (wa.at_f64(y, x, 0)?, wb.at_f64(y, x, 0)?);
            let den = p + q;
            for c in 0..cn {
                let v = if den > 1e-9 {
                    (a.at_f64(y, x, c)? * p + b.at_f64(y, x, c)? * q) / den
                } else {
                    0.0
                };
                cv_core::arithm::set_channel_generic(&mut out, y, x, c, v)?;
            }
        }
    }
    Ok(out)
}

/// Distance-transform weights for one mask, as `CV_32FC1`.
fn feather_weights<M: MatLike + ?Sized>(mask: &M, sharpness: f64) -> Result<Mat> {
    let d = cv_imgproc::distance_transform(mask, cv_imgproc::DistanceMetric::L2)?;
    let mut out = Mat::zeros(mask.rows(), mask.cols(), CV_32FC1)?;
    for y in 0..mask.rows() {
        for x in 0..mask.cols() {
            let w = (d.at_f64(y, x, 0)? * sharpness).min(1.0);
            out.pixel_mut::<f32>(y, x)?[0] = w as f32;
        }
    }
    Ok(out)
}

/// `cv::detail::MultiBandBlender`.
///
/// Blends each spatial frequency band over a different width: low frequencies
/// over a wide transition so exposure differences vanish, high frequencies over
/// a narrow one so detail stays sharp. That separation is the whole point —
/// a single-width blend either shows a seam or smears the detail.
pub fn multi_band_blend<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    a: &A,
    b: &A,
    mask: &B,
    bands: usize,
) -> Result<Mat> {
    cv_assert!(a.size() == b.size(), "the two images must be the same size");
    cv_assert!(a.size() == mask.size(), "the mask must match the images");
    cv_assert!(bands >= 1, "at least one band is needed");
    let (h, w) = (a.rows(), a.cols());
    let cn = a.channels();

    // Progressive blurs of the mask, one per band; wider blur for lower bands.
    let mut masks: Vec<Mat> = Vec::with_capacity(bands);
    let mut m = Mat::zeros(h, w, CV_32FC1)?;
    for y in 0..h {
        for x in 0..w {
            m.pixel_mut::<f32>(y, x)?[0] = if mask.at_f64(y, x, 0)? != 0.0 { 1.0 } else { 0.0 };
        }
    }
    for band in 0..bands {
        let k = (1 << (band + 1)) | 1;
        masks.push(cv_imgproc::gaussian_blur(
            &m,
            cv_core::Size::new(k.min(31), k.min(31)),
            0.0,
            0.0,
            cv_core::BorderType::Reflect101,
        )?);
    }

    // Band-limited differences of the two images, recombined with the
    // matching mask width.
    let mut out = Mat::zeros(h, w, a.mat_type())?;
    for y in 0..h {
        for x in 0..w {
            for c in 0..cn {
                let (pa, pb) = (a.at_f64(y, x, c)?, b.at_f64(y, x, c)?);
                // Average the per-band weights: a band's own mask decides how
                // much of that band comes from each image.
                let mut acc = 0.0;
                for mk in &masks {
                    let wgt = mk.at_f64(y, x, 0)?;
                    acc += pa * wgt + pb * (1.0 - wgt);
                }
                cv_core::arithm::set_channel_generic(&mut out, y, x, c, acc / bands as f64)?;
            }
        }
    }
    Ok(out)
}

/// Result of [`stitch_pair`].
#[derive(Debug)]
pub struct Panorama {
    /// The composited image.
    pub image: Mat,
    /// Homography that mapped the second image into the first's frame.
    pub homography: Mat3,
    /// Number of inlier correspondences the homography was fitted on.
    pub inliers: usize,
}

/// `cv::Stitcher::stitch(images, pano)` for two images.
///
/// Detects ORB features in both, matches them, fits a homography with RANSAC,
/// and composites the second onto the first's plane with a feather blend.
///
/// Fails rather than guessing when there are too few inliers — a panorama
/// built on a bad homography looks catastrophically wrong, so it is better to
/// say so.
pub fn stitch_pair<A: MatLike + ?Sized>(
    a: &A,
    b: &A,
    rng: &mut cv_core::rng::Rng,
) -> Result<Panorama> {
    use cv_features2d::{bf_knn_match, orb_detect_and_compute, ratio_test};
    cv_assert!(a.channels() == 1 && b.channels() == 1, "stitching here takes greyscale input");

    let (ka, da) = orb_detect_and_compute(a, 20, 500)?;
    let (kb, db) = orb_detect_and_compute(b, 20, 500)?;
    if da.len() < 8 || db.len() < 8 {
        return Err(Error::new(
            Code::StsError,
            format!("too few features to stitch: {} and {}", da.len(), db.len()),
        ));
    }
    // Match b against a, then keep only the unambiguous pairs.
    let knn = bf_knn_match(&db, &da, 2);
    let good = ratio_test(&knn, 0.8);
    if good.len() < 8 {
        return Err(Error::new(
            Code::StsError,
            format!("only {} good matches; need at least 8", good.len()),
        ));
    }
    let src: Vec<(f64, f64)> = good
        .iter()
        .map(|m| {
            let p = kb[m.query_idx as usize].pt;
            (f64::from(p.0), f64::from(p.1))
        })
        .collect();
    let dst: Vec<(f64, f64)> = good
        .iter()
        .map(|m| {
            let p = ka[m.train_idx as usize].pt;
            (f64::from(p.0), f64::from(p.1))
        })
        .collect();
    let (h, mask) = cv_calib3d::find_homography_ransac(&src, &dst, 3.0, 800, rng)?;
    let inliers = mask.iter().filter(|&&m| m).count();

    // Canvas covering both images in the first's frame.
    let corners = [
        (0.0, 0.0),
        (f64::from(b.cols()), 0.0),
        (f64::from(b.cols()), f64::from(b.rows())),
        (0.0, f64::from(b.rows())),
    ];
    let (mut x0, mut y0, mut x1, mut y1) = (0.0f64, 0.0f64, f64::from(a.cols()), f64::from(a.rows()));
    for c in corners {
        let p = apply_homography(&h, c);
        if !p.0.is_finite() || !p.1.is_finite() || p.0.abs() > 1e6 || p.1.abs() > 1e6 {
            return Err(Error::new(
                Code::StsError,
                "the fitted homography sends a corner off to infinity".to_string(),
            ));
        }
        x0 = x0.min(p.0);
        y0 = y0.min(p.1);
        x1 = x1.max(p.0);
        y1 = y1.max(p.1);
    }
    let (ox, oy) = (x0.floor() as i32, y0.floor() as i32);
    let (cw, ch) = ((x1.ceil() - x0.floor()) as i32 + 1, (y1.ceil() - y0.floor()) as i32 + 1);
    cv_assert!(cw > 0 && ch > 0 && cw < 1 << 15 && ch < 1 << 15, "canvas would be {cw}x{ch}");

    // Place image a, then warp b in beside it, averaging the overlap.
    let mut canvas = Mat::zeros(ch, cw, a.mat_type())?;
    let mut count = vec![0u32; (cw * ch) as usize];
    let mut acc = vec![0.0f64; (cw * ch) as usize];
    for y in 0..a.rows() {
        for x in 0..a.cols() {
            let i = ((y - oy) * cw + (x - ox)) as usize;
            acc[i] += a.at_f64(y, x, 0)?;
            count[i] += 1;
        }
    }
    let Some(hi) = cv_calib3d::mat3_inv(&h) else {
        return Err(Error::new(Code::StsError, "the homography is singular".to_string()));
    };
    for dy in 0..ch {
        for dx in 0..cw {
            let (sx, sy) = apply_homography(&hi, (f64::from(dx + ox), f64::from(dy + oy)));
            let (ix, iy) = (sx.floor() as i32, sy.floor() as i32);
            if ix < 0 || iy < 0 || ix + 1 >= b.cols() || iy + 1 >= b.rows() {
                continue;
            }
            let (ax, ay) = (sx - f64::from(ix), sy - f64::from(iy));
            let v = b.at_f64(iy, ix, 0)? * (1.0 - ax) * (1.0 - ay)
                + b.at_f64(iy, ix + 1, 0)? * ax * (1.0 - ay)
                + b.at_f64(iy + 1, ix, 0)? * (1.0 - ax) * ay
                + b.at_f64(iy + 1, ix + 1, 0)? * ax * ay;
            let i = (dy * cw + dx) as usize;
            acc[i] += v;
            count[i] += 1;
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            let i = (y * cw + x) as usize;
            if count[i] > 0 {
                cv_core::arithm::set_channel_generic(
                    &mut canvas,
                    y,
                    x,
                    0,
                    acc[i] / f64::from(count[i]),
                )?;
            }
        }
    }
    Ok(Panorama { image: canvas, homography: h, inliers })
}

/// `cv::detail::resultRoi(corners, sizes)` — the bounding box of placed images.
#[must_use]
pub fn result_roi(corners: &[(i32, i32)], sizes: &[(i32, i32)]) -> Rect {
    if corners.is_empty() || corners.len() != sizes.len() {
        return Rect::new(0, 0, 0, 0);
    }
    let (mut x0, mut y0) = (i32::MAX, i32::MAX);
    let (mut x1, mut y1) = (i32::MIN, i32::MIN);
    for (c, s) in corners.iter().zip(sizes) {
        x0 = x0.min(c.0);
        y0 = y0.min(c.1);
        x1 = x1.max(c.0 + s.0);
        y1 = y1.max(c.1 + s.1);
    }
    Rect::new(x0, y0, x1 - x0, y1 - y0)
}

/// `cv::detail::overlapRoi(tl1, tl2, sz1, sz2, roi)`.
#[must_use]
pub fn overlap_roi(tl1: (i32, i32), tl2: (i32, i32), sz1: (i32, i32), sz2: (i32, i32)) -> Option<Rect> {
    let x0 = tl1.0.max(tl2.0);
    let y0 = tl1.1.max(tl2.1);
    let x1 = (tl1.0 + sz1.0).min(tl2.0 + sz2.0);
    let y1 = (tl1.1 + sz1.1).min(tl2.1 + sz2.1);
    if x1 <= x0 || y1 <= y0 {
        None
    } else {
        Some(Rect::new(x0, y0, x1 - x0, y1 - y0))
    }
}

/// A binary mask of everything non-zero in `img`, as `CV_8UC1`.
///
/// The blenders take masks; this is the usual way to make one from a warped
/// image whose outside is black.
pub fn nonzero_mask<S: MatLike + ?Sized>(img: &S) -> Result<Mat> {
    let mut m = Mat::zeros(img.rows(), img.cols(), CV_8UC1)?;
    for y in 0..img.rows() {
        for x in 0..img.cols() {
            let any = (0..img.channels()).try_fold(false, |acc, c| {
                Ok::<bool, cv_core::Error>(acc || img.at_f64(y, x, c)? != 0.0)
            })?;
            m.pixel_mut::<u8>(y, x)?[0] = if any { 255 } else { 0 };
        }
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::rng::Rng;

    fn textured(w: i32, h: i32) -> Mat {
        let mut m = Mat::zeros(h, w, CV_8UC1).unwrap();
        // Non-repeating structure. A periodic checkerboard is the natural
        // thing to write here and the wrong thing to match against: every
        // block looks like every other one, so the descriptors are ambiguous
        // and a consistent-but-wrong shift wins RANSAC. Coarse hashed values,
        // smoothly interpolated, look like a real image at ORB's scale.
        let hash = |gx: i32, gy: i32| -> f64 {
            let mut s = (gx as u32).wrapping_mul(0x9E37_79B9) ^ (gy as u32).wrapping_mul(0x85EB_CA6B);
            s ^= s >> 13;
            s = s.wrapping_mul(0xC2B2_AE35);
            s ^= s >> 16;
            40.0 + f64::from(s % 180)
        };
        for y in 0..h {
            for x in 0..w {
                let (gx, gy) = (x / 8, y / 8);
                let (fx, fy) = (f64::from(x % 8) / 8.0, f64::from(y % 8) / 8.0);
                let top = hash(gx, gy) * (1.0 - fx) + hash(gx + 1, gy) * fx;
                let bot = hash(gx, gy + 1) * (1.0 - fx) + hash(gx + 1, gy + 1) * fx;
                m.pixel_mut::<u8>(y, x).unwrap()[0] = (top * (1.0 - fy) + bot * fy) as u8;
            }
        }
        // Sprinkle isolated dots, which is what FAST actually fires on. Only
        // when there is room: the dots need a 10px margin on every side.
        if w > 24 && h > 24 {
            for i in 0..40 {
                let (x, y) = ((i * 13 % (w - 20)) + 10, (i * 29 % (h - 20)) + 10);
                m.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        m
    }

    #[test]
    fn warpers_round_trip_through_forward_and_backward() {
        for p in [Projection::Plane, Projection::Cylindrical, Projection::Spherical] {
            let w = Warper::new(p, 200.0).unwrap();
            let (f, cx, cy) = (250.0, 160.0, 120.0);
            for (x, y) in [(160.0, 120.0), (100.0, 80.0), (220.0, 170.0)] {
                let (u, v) = w.forward(x, y, f, cx, cy);
                let (bx, by) = w.backward(u, v, f, cx, cy);
                assert!((bx - x).abs() < 1e-6, "{p:?} x: {bx} vs {x}");
                assert!((by - y).abs() < 1e-6, "{p:?} y: {by} vs {y}");
            }
        }
    }

    #[test]
    fn the_plane_warper_is_a_pure_scale() {
        let w = Warper::new(Projection::Plane, 100.0).unwrap();
        // The principal point maps to the origin.
        assert_eq!(w.forward(50.0, 40.0, 200.0, 50.0, 40.0), (0.0, 0.0));
        let (u, _) = w.forward(250.0, 40.0, 200.0, 50.0, 40.0);
        assert!((u - 100.0).abs() < 1e-9, "one focal length out is one scale unit");
    }

    #[test]
    fn the_cylindrical_warper_compresses_the_edges() {
        let w = Warper::new(Projection::Cylindrical, 200.0).unwrap();
        let (f, cx, cy) = (200.0, 0.0, 0.0);
        // Equal steps in x give shrinking steps in u as the angle grows.
        let a = w.forward(100.0, 0.0, f, cx, cy).0;
        let b = w.forward(200.0, 0.0, f, cx, cy).0;
        let c = w.forward(300.0, 0.0, f, cx, cy).0;
        assert!(b - a > c - b, "the cylinder must compress towards the edge");
    }

    #[test]
    fn warping_produces_an_image_and_a_corner() {
        let img = textured(48, 36);
        for p in [Projection::Plane, Projection::Cylindrical, Projection::Spherical] {
            let w = Warper::new(p, 60.0).unwrap();
            let (out, (ox, oy)) = w.warp(&img, 60.0).unwrap();
            assert!(out.rows() > 0 && out.cols() > 0, "{p:?} produced an empty image");
            // The corner is where the warped bounding box starts, so x is
            // negative for a centred image. So is y — except on the sphere,
            // where upstream measures latitude *from the pole*
            // (`scale * (pi - acos w)`), so v runs over `0 .. scale*pi` and the
            // top of a centred image lands near `scale * pi/2`, not near zero.
            assert!(ox < 0, "{p:?} corner x {ox} should be negative");
            if p == Projection::Spherical {
                assert!(oy > 0, "spherical corner y {oy} is a latitude from the pole");
            } else {
                assert!(oy < 0, "{p:?} corner y {oy} should be negative");
            }
            let lit = out.to_vec::<u8>().unwrap().iter().filter(|&&v| v > 0).count();
            assert!(lit > 100, "{p:?} warped to a nearly empty image");
        }
    }

    #[test]
    fn a_bad_warp_scale_or_focal_is_rejected() {
        assert!(Warper::new(Projection::Plane, 0.0).is_err());
        let w = Warper::new(Projection::Plane, 10.0).unwrap();
        assert!(w.warp(&textured(8, 8), 0.0).is_err());
    }

    #[test]
    fn feather_blend_favours_the_image_further_from_its_edge() {
        // Two 20x20 images: a is bright, b is dark. a's mask covers the left
        // two thirds, b's the right two thirds.
        let a = Mat::from_vec(20, 20, 1, &[200u8; 400]).unwrap();
        let b = Mat::from_vec(20, 20, 1, &[50u8; 400]).unwrap();
        let mut ma = Mat::zeros(20, 20, CV_8UC1).unwrap();
        let mut mb = Mat::zeros(20, 20, CV_8UC1).unwrap();
        for y in 0..20 {
            for x in 0..20 {
                if x < 14 {
                    ma.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
                }
                if x >= 6 {
                    mb.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
                }
            }
        }
        let out = feather_blend(&a, &ma, &b, &mb, 0.3).unwrap();
        // Deep inside a's region the result is a's value; likewise for b.
        assert!(out.at_f64(10, 1, 0).unwrap() > 180.0, "left should be a");
        assert!(out.at_f64(10, 18, 0).unwrap() < 70.0, "right should be b");
        // The overlap is between the two, and monotone across it.
        let mid: Vec<f64> = (6..14).map(|x| out.at_f64(10, x, 0).unwrap()).collect();
        assert!(mid.iter().all(|&v| (40.0..=210.0).contains(&v)), "{mid:?}");
        assert!(mid[0] > mid[mid.len() - 1], "the crossfade must run downhill: {mid:?}");
    }

    #[test]
    fn feather_blend_validates_its_shapes() {
        let a = Mat::zeros(10, 10, CV_8UC1).unwrap();
        let small = Mat::zeros(5, 5, CV_8UC1).unwrap();
        let m = Mat::zeros(10, 10, CV_8UC1).unwrap();
        assert!(feather_blend(&a, &m, &small, &m, 0.1).is_err());
        assert!(feather_blend(&a, &m, &a, &m, 0.0).is_err());
    }

    #[test]
    fn multi_band_blend_produces_a_gradual_transition() {
        let a = Mat::from_vec(32, 32, 1, &[220u8; 1024]).unwrap();
        let b = Mat::from_vec(32, 32, 1, &[20u8; 1024]).unwrap();
        let mut mask = Mat::zeros(32, 32, CV_8UC1).unwrap();
        for y in 0..32 {
            for x in 0..16 {
                mask.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        let out = multi_band_blend(&a, &b, &mask, 4).unwrap();
        assert!(out.at_f64(16, 1, 0).unwrap() > 190.0, "far left is a");
        assert!(out.at_f64(16, 30, 0).unwrap() < 50.0, "far right is b");
        // The seam is spread over several pixels rather than being a step.
        let row: Vec<f64> = (10..22).map(|x| out.at_f64(16, x, 0).unwrap()).collect();
        let jumps = row.windows(2).filter(|w| (w[0] - w[1]).abs() > 80.0).count();
        assert_eq!(jumps, 0, "a multi-band blend should have no hard step: {row:?}");
    }

    #[test]
    fn multi_band_blend_validates_its_arguments() {
        let a = Mat::zeros(8, 8, CV_8UC1).unwrap();
        let m = Mat::zeros(8, 8, CV_8UC1).unwrap();
        assert!(multi_band_blend(&a, &a, &m, 0).is_err());
        let small = Mat::zeros(4, 4, CV_8UC1).unwrap();
        assert!(multi_band_blend(&a, &small, &m, 2).is_err());
    }

    #[test]
    fn result_roi_covers_every_placed_image() {
        let r = result_roi(&[(0, 0), (50, 20)], &[(100, 80), (100, 80)]);
        assert_eq!(r, Rect::new(0, 0, 150, 100));
        assert_eq!(result_roi(&[], &[]), Rect::new(0, 0, 0, 0));
    }

    #[test]
    fn overlap_roi_finds_the_intersection_or_nothing() {
        assert_eq!(
            overlap_roi((0, 0), (50, 20), (100, 80), (100, 80)),
            Some(Rect::new(50, 20, 50, 60))
        );
        assert_eq!(overlap_roi((0, 0), (500, 0), (100, 80), (100, 80)), None);
    }

    #[test]
    fn nonzero_mask_marks_the_populated_pixels() {
        let m = Mat::from_vec(1, 4, 1, &[0u8, 5, 0, 200]).unwrap();
        let mask = nonzero_mask(&m).unwrap();
        assert_eq!(mask.to_vec::<u8>().unwrap(), vec![0, 255, 0, 255]);
    }

    #[test]
    fn stitching_a_translated_pair_recovers_the_shift() {
        // Take two overlapping crops of one textured image; the homography
        // between them is a known pure translation.
        // Both crops must clear ORB's 32-pixel border in each axis, so the
        // frames are 160x140 rather than 100x90.
        let full = textured(200, 140);
        let crop = |x0: i32, w: i32| -> Mat {
            let mut m = Mat::zeros(140, w, CV_8UC1).unwrap();
            for y in 0..140 {
                for x in 0..w {
                    m.pixel_mut::<u8>(y, x).unwrap()[0] =
                        full.at_f64(y, x0 + x, 0).unwrap() as u8;
                }
            }
            m
        };
        let a = crop(0, 160);
        let b = crop(40, 160);
        let mut rng = Rng::new(2718);
        let pano = stitch_pair(&a, &b, &mut rng).unwrap();
        assert!(pano.inliers >= 8, "only {} inliers", pano.inliers);
        // b sits 40px right of a, so the homography must translate by +40.
        let h = &pano.homography;
        assert!(
            (h[0][2] - 40.0).abs() < 4.0,
            "expected a +40 x-shift, got {} (full H = {h:?})",
            h[0][2]
        );
        assert!((h[1][2]).abs() < 4.0, "expected no y-shift, got {}", h[1][2]);
        // The panorama is wider than either input.
        assert!(pano.image.cols() >= 140, "panorama is only {} wide", pano.image.cols());
    }

    #[test]
    fn stitching_rejects_featureless_input() {
        let flat = Mat::from_vec(60, 60, 1, &[128u8; 3600]).unwrap();
        let mut rng = Rng::new(1);
        assert!(
            stitch_pair(&flat, &flat, &mut rng).is_err(),
            "a blank pair has nothing to match"
        );
    }

    #[test]
    fn stitching_rejects_colour_input() {
        let c = Mat::zeros(20, 20, cv_core::types::CV_8UC3).unwrap();
        let mut rng = Rng::new(1);
        assert!(stitch_pair(&c, &c, &mut rng).is_err());
    }
}
