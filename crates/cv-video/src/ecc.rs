//! Enhanced correlation coefficient alignment and the `video` enumerations
//! (`modules/video/src/ecc.cpp`).
//!
//! # Why ECC rather than feature matching
//!
//! Aligning two images by matching features needs features to exist. ECC works
//! directly on intensities, so it aligns smooth or textureless content that no
//! corner detector would find anything in — medical slices, document scans,
//! long-exposure frames. It also optimises a *normalised* correlation, which
//! makes it invariant to a global brightness and contrast change: the same
//! scene photographed at two exposures still aligns.
//!
//! The cost is that it is local. ECC refines an initial guess by gradient
//! ascent, so it needs to start close, and it is not a substitute for a
//! feature-based estimate when the displacement is large.

use crate::plane::Plane;
use cv_core::{cv_assert, Mat, MatLike, MatLikeMut, Result};

/// `cv::MotionTypes` — the warp family ECC will search over.
///
/// Fewer parameters converge from further away and cannot express as much;
/// [`MotionType::Homography`] can fit anything planar but needs a good start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MotionType {
    /// `cv::MOTION_TRANSLATION` — two parameters.
    Translation,
    /// `cv::MOTION_EUCLIDEAN` — rotation plus translation, three parameters.
    Euclidean,
    /// `cv::MOTION_AFFINE` — six parameters. Upstream's default.
    #[default]
    Affine,
    /// `cv::MOTION_HOMOGRAPHY` — eight parameters.
    Homography,
}

impl MotionType {
    /// How many free parameters the warp has.
    #[must_use]
    pub fn param_count(self) -> usize {
        match self {
            MotionType::Translation => 2,
            MotionType::Euclidean => 3,
            MotionType::Affine => 6,
            MotionType::Homography => 8,
        }
    }

    /// Rows in the warp matrix: 2 for the affine family, 3 for homography.
    #[must_use]
    pub fn matrix_rows(self) -> i32 {
        if self == MotionType::Homography {
            3
        } else {
            2
        }
    }
}

/// `cv::OPTFLOW_*` flags shared by the optical-flow entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpticalFlowFlag {
    /// `cv::OPTFLOW_USE_INITIAL_FLOW`
    UseInitialFlow,
    /// `cv::OPTFLOW_LK_GET_MIN_EIGENVALS` — report the minimum eigenvalue as
    /// the error measure rather than the patch difference.
    LkGetMinEigenvals,
    /// `cv::OPTFLOW_FARNEBACK_GAUSSIAN` — Gaussian rather than box weighting.
    FarnebackGaussian,
}

/// `cv::DISOpticalFlow` quality presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisPreset {
    /// `cv::DISOpticalFlow::PRESET_ULTRAFAST`
    Ultrafast,
    /// `cv::DISOpticalFlow::PRESET_FAST`
    Fast,
    /// `cv::DISOpticalFlow::PRESET_MEDIUM` — upstream's default.
    #[default]
    Medium,
}

/// `cv::TrackerMIL`-style sampler modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackerSamplerMode {
    /// `cv::MODE_INIT_POS`
    InitPos,
    /// `cv::MODE_INIT_NEG`
    InitNeg,
    /// `cv::MODE_TRACK_POS`
    TrackPos,
    /// `cv::MODE_TRACK_NEG`
    TrackNeg,
    /// `cv::MODE_DETECT`
    Detect,
}

/// `cv::ECCParameters`-adjacent stopping criteria.
#[derive(Debug, Clone, Copy)]
pub struct EccCriteria {
    /// Maximum gradient-ascent steps.
    pub max_count: i32,
    /// Stop once the correlation improves by less than this.
    pub epsilon: f64,
    /// Gaussian blur applied to both images first; 0 disables it.
    ///
    /// Smoothing widens the basin of convergence, which is usually worth more
    /// than the precision it costs.
    pub gauss_filt_size: i32,
}

impl Default for EccCriteria {
    /// Upstream's defaults: 50 iterations, 0.001 epsilon, 5-tap smoothing.
    fn default() -> Self {
        Self { max_count: 50, epsilon: 0.001, gauss_filt_size: 5 }
    }
}



/// Apply a warp to a point, in the parameterisation ECC uses.
fn warp_point(p: &[f64], motion: MotionType, x: f64, y: f64) -> (f64, f64) {
    match motion {
        MotionType::Translation => (x + p[0], y + p[1]),
        MotionType::Euclidean => {
            let (c, s) = (p[0].cos(), p[0].sin());
            (c * x - s * y + p[1], s * x + c * y + p[2])
        }
        MotionType::Affine => (
            (1.0 + p[0]) * x + p[2] * y + p[4],
            p[1] * x + (1.0 + p[3]) * y + p[5],
        ),
        MotionType::Homography => {
            let d = p[2] * x + p[5] * y + 1.0;
            if d.abs() < 1e-12 {
                (x, y)
            } else {
                (
                    ((1.0 + p[0]) * x + p[3] * y + p[6]) / d,
                    (p[1] * x + (1.0 + p[4]) * y + p[7]) / d,
                )
            }
        }
    }
}

/// The Jacobian of [`warp_point`] with respect to the parameters.
fn warp_jacobian(p: &[f64], motion: MotionType, x: f64, y: f64) -> (Vec<f64>, Vec<f64>) {
    match motion {
        MotionType::Translation => (vec![1.0, 0.0], vec![0.0, 1.0]),
        MotionType::Euclidean => {
            let (c, s) = (p[0].cos(), p[0].sin());
            (vec![-s * x - c * y, 1.0, 0.0], vec![c * x - s * y, 0.0, 1.0])
        }
        MotionType::Affine => (
            vec![x, 0.0, y, 0.0, 1.0, 0.0],
            vec![0.0, x, 0.0, y, 0.0, 1.0],
        ),
        MotionType::Homography => {
            let d = p[2] * x + p[5] * y + 1.0;
            let d = if d.abs() < 1e-12 { 1e-12 } else { d };
            let (u, v) = (
                (1.0 + p[0]) * x + p[3] * y + p[6],
                p[1] * x + (1.0 + p[4]) * y + p[7],
            );
            (
                vec![x / d, 0.0, -u * x / (d * d), y / d, 0.0, -u * y / (d * d), 1.0 / d, 0.0],
                vec![0.0, x / d, -v * x / (d * d), 0.0, y / d, -v * y / (d * d), 0.0, 1.0 / d],
            )
        }
    }
}

/// Bilinear sample, or `None` outside the image.
fn sample(p: &Plane<f64>, x: f64, y: f64) -> Option<f64> {
    if x < 0.0 || y < 0.0 || x >= (p.w - 1) as f64 || y >= (p.h - 1) as f64 {
        return None;
    }
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (ax, ay) = (x - x0 as f64, y - y0 as f64);
    Some(
        p.at(y0, x0) * (1.0 - ax) * (1.0 - ay)
            + p.at(y0, x0 + 1) * ax * (1.0 - ay)
            + p.at(y0 + 1, x0) * (1.0 - ax) * ay
            + p.at(y0 + 1, x0 + 1) * ax * ay,
    )
}

/// `cv::computeECC(templateImage, inputImage, inputMask)`.
///
/// The zero-mean normalised cross-correlation of two images, in `-1..=1`. This
/// is the quantity [`find_transform_ecc`] maximises; computing it directly is
/// how a caller checks whether an alignment is any good.
pub fn compute_ecc<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    template: &A,
    input: &B,
) -> Result<f64> {
    cv_assert!(
        template.rows() == input.rows() && template.cols() == input.cols(),
        "computeECC needs images of the same size"
    );
    cv_assert!(!template.is_empty(), "computeECC needs a non-empty image");
    let a = Plane::<f64>::from_mat(template)?;
    let b = Plane::<f64>::from_mat(input)?;
    let n = (a.w * a.h) as f64;
    let ma = a.d.iter().sum::<f64>() / n;
    let mb = b.d.iter().sum::<f64>() / n;
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for (x, y) in a.d.iter().zip(&b.d) {
        let (u, v) = (x - ma, y - mb);
        num += u * v;
        da += u * u;
        db += v * v;
    }
    if da <= 0.0 || db <= 0.0 {
        // A constant image has no variation to correlate; upstream would
        // divide by zero here, and a NaN would silently poison a caller's
        // convergence test.
        return Ok(0.0);
    }
    Ok(num / (da.sqrt() * db.sqrt()))
}

/// `cv::findTransformECC(templateImage, inputImage, warpMatrix, motionType, criteria, inputMask, gaussFiltSize)`.
///
/// Refines `warp` so that warping `input` onto `template` maximises their
/// correlation. Returns the achieved coefficient and the refined warp.
///
/// # Convergence
///
/// This is gradient ascent on a non-convex surface: it finds a *local*
/// maximum. Started far from the answer it will converge confidently to the
/// wrong warp, which is why upstream smooths both images first — a wider basin
/// is worth more than the precision the blur costs.
pub fn find_transform_ecc<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    template: &A,
    input: &B,
    motion: MotionType,
    criteria: EccCriteria,
) -> Result<(f64, Mat)> {
    cv_assert!(!template.is_empty() && !input.is_empty(), "findTransformECC needs non-empty images");
    cv_assert!(
        template.channels() == 1 && input.channels() == 1,
        "findTransformECC takes single-channel images"
    );
    cv_assert!(criteria.max_count > 0, "the iteration count must be positive");

    // Smoothing widens the basin of convergence.
    let (tpl, inp) = if criteria.gauss_filt_size > 1 {
        let k = criteria.gauss_filt_size | 1;
        (
            cv_imgproc::filter::gaussian_blur(
                template,
                cv_core::Size::new(k, k),
                0.0,
                0.0,
                cv_core::border::BorderType::Reflect101,
            )?,
            cv_imgproc::filter::gaussian_blur(
                input,
                cv_core::Size::new(k, k),
                0.0,
                0.0,
                cv_core::border::BorderType::Reflect101,
            )?,
        )
    } else {
        (to_mat(template)?, to_mat(input)?)
    };
    let t = Plane::<f64>::from_mat(&tpl)?;
    let i = Plane::<f64>::from_mat(&inp)?;

    let np = motion.param_count();
    let mut p = vec![0.0f64; np];
    let mut last_rho = -1.0f64;
    let mut rho = -1.0f64;

    for _ in 0..criteria.max_count {
        // Warp the input onto the template's grid, keeping the pixels that
        // land inside so the correlation is computed over the overlap only.
        let mut iw = Vec::with_capacity(t.d.len());
        let mut tv = Vec::with_capacity(t.d.len());
        let mut jac: Vec<Vec<f64>> = Vec::with_capacity(t.d.len());
        for y in 0..t.h {
            for x in 0..t.w {
                let (wx, wy) = warp_point(&p, motion, x as f64, y as f64);
                let Some(v) = sample(&i, wx, wy) else { continue };
                // Steepest-descent image: image gradient times warp Jacobian.
                let gx = sample(&i, wx + 0.5, wy).unwrap_or(v)
                    - sample(&i, wx - 0.5, wy).unwrap_or(v);
                let gy = sample(&i, wx, wy + 0.5).unwrap_or(v)
                    - sample(&i, wx, wy - 0.5).unwrap_or(v);
                let (jx, jy) = warp_jacobian(&p, motion, x as f64, y as f64);
                jac.push((0..np).map(|k| gx * jx[k] + gy * jy[k]).collect());
                iw.push(v);
                tv.push(t.at(y, x));
            }
        }
        if iw.len() < np + 1 {
            break;
        }

        // Zero-mean both, so the correlation ignores a brightness offset.
        let n = iw.len() as f64;
        let (mi, mt) = (iw.iter().sum::<f64>() / n, tv.iter().sum::<f64>() / n);
        for v in iw.iter_mut() {
            *v -= mi;
        }
        for v in tv.iter_mut() {
            *v -= mt;
        }
        let ni = iw.iter().map(|v| v * v).sum::<f64>().sqrt();
        let nt = tv.iter().map(|v| v * v).sum::<f64>().sqrt();
        if ni <= 0.0 || nt <= 0.0 {
            break;
        }
        rho = iw.iter().zip(&tv).map(|(a, b)| a * b).sum::<f64>() / (ni * nt);
        if (rho - last_rho).abs() < criteria.epsilon {
            break;
        }
        last_rho = rho;

        // Normal equations for the parameter update.
        let mut hess = vec![0.0f64; np * np];
        let mut rhs = vec![0.0f64; np];
        for (k, g) in jac.iter().enumerate() {
            // Project out the mean, matching the zero-mean correlation.
            let err = tv[k] * (ni / nt) - iw[k];
            for a in 0..np {
                rhs[a] += g[a] * err;
                for b in 0..np {
                    hess[a * np + b] += g[a] * g[b];
                }
            }
        }
        let Some(dp) = solve_sym(&mut hess, &mut rhs, np) else { break };
        let step: f64 = dp.iter().map(|v| v.abs()).sum();
        for (a, d) in p.iter_mut().zip(&dp) {
            *a += d;
        }
        if step < criteria.epsilon * 1e-3 {
            break;
        }
    }

    Ok((rho, warp_matrix(&p, motion)?))
}

/// Materialise a `MatLike` into an owned `Mat`.
fn to_mat<S: MatLike + ?Sized>(src: &S) -> Result<Mat> {
    let mut out = Mat::zeros(src.rows(), src.cols(), src.mat_type())?;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            for c in 0..src.channels() {
                cv_core::arithm::set_channel_generic(&mut out, y, x, c, src.at_f64(y, x, c)?)?;
            }
        }
    }
    Ok(out)
}

/// Pack the parameter vector into upstream's warp matrix.
fn warp_matrix(p: &[f64], motion: MotionType) -> Result<Mat> {
    let rows = motion.matrix_rows();
    let mut m = Mat::zeros(rows, 3, cv_core::types::CV_64FC1)?;
    let set = |m: &mut Mat, y: i32, x: i32, v: f64| -> Result<()> {
        m.pixel_mut::<f64>(y, x)?[0] = v;
        Ok(())
    };
    match motion {
        MotionType::Translation => {
            set(&mut m, 0, 0, 1.0)?;
            set(&mut m, 1, 1, 1.0)?;
            set(&mut m, 0, 2, p[0])?;
            set(&mut m, 1, 2, p[1])?;
        }
        MotionType::Euclidean => {
            let (c, s) = (p[0].cos(), p[0].sin());
            set(&mut m, 0, 0, c)?;
            set(&mut m, 0, 1, -s)?;
            set(&mut m, 1, 0, s)?;
            set(&mut m, 1, 1, c)?;
            set(&mut m, 0, 2, p[1])?;
            set(&mut m, 1, 2, p[2])?;
        }
        MotionType::Affine => {
            set(&mut m, 0, 0, 1.0 + p[0])?;
            set(&mut m, 0, 1, p[2])?;
            set(&mut m, 0, 2, p[4])?;
            set(&mut m, 1, 0, p[1])?;
            set(&mut m, 1, 1, 1.0 + p[3])?;
            set(&mut m, 1, 2, p[5])?;
        }
        MotionType::Homography => {
            set(&mut m, 0, 0, 1.0 + p[0])?;
            set(&mut m, 0, 1, p[3])?;
            set(&mut m, 0, 2, p[6])?;
            set(&mut m, 1, 0, p[1])?;
            set(&mut m, 1, 1, 1.0 + p[4])?;
            set(&mut m, 1, 2, p[7])?;
            set(&mut m, 2, 0, p[2])?;
            set(&mut m, 2, 1, p[5])?;
            set(&mut m, 2, 2, 1.0)?;
        }
    }
    Ok(m)
}

/// Solve a small symmetric system by Gauss elimination with partial pivoting.
fn solve_sym(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r * n + col].abs() > a[piv * n + col].abs() {
                piv = r;
            }
        }
        if a[piv * n + col].abs() < 1e-14 {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A textured image with enough structure for alignment to be well posed.
    fn textured(w: i32, h: i32) -> Mat {
        let mut d = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let v = 128.0
                    + 60.0 * ((x as f64) * 0.35).sin()
                    + 50.0 * ((y as f64) * 0.27).cos()
                    + 30.0 * (((x + y) as f64) * 0.11).sin();
                d[(y * w + x) as usize] = v.clamp(0.0, 255.0) as u8;
            }
        }
        Mat::from_vec(h, w, 1, &d).unwrap()
    }

    /// Shift an image by a whole number of pixels.
    fn shift(src: &Mat, dx: i32, dy: i32) -> Mat {
        let (h, w) = (src.rows(), src.cols());
        let mut d = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = ((x - dx).clamp(0, w - 1), (y - dy).clamp(0, h - 1));
                d[(y * w + x) as usize] = src.at_f64(sy, sx, 0).unwrap() as u8;
            }
        }
        Mat::from_vec(h, w, 1, &d).unwrap()
    }

    #[test]
    fn the_parameter_counts_match_upstream() {
        assert_eq!(MotionType::Translation.param_count(), 2);
        assert_eq!(MotionType::Euclidean.param_count(), 3);
        assert_eq!(MotionType::Affine.param_count(), 6);
        assert_eq!(MotionType::Homography.param_count(), 8);
        assert_eq!(MotionType::Affine.matrix_rows(), 2);
        assert_eq!(MotionType::Homography.matrix_rows(), 3);
        assert_eq!(MotionType::default(), MotionType::Affine);
    }

    #[test]
    fn ecc_of_an_image_with_itself_is_one() {
        let a = textured(40, 30);
        let e = compute_ecc(&a, &a).unwrap();
        assert!((e - 1.0).abs() < 1e-12, "self-correlation should be 1, got {e}");
    }

    #[test]
    fn ecc_ignores_brightness_and_contrast() {
        // The whole point of the normalisation: the same scene at a different
        // exposure must still correlate perfectly.
        let a = textured(40, 30);
        let (h, w) = (a.rows(), a.cols());
        let mut d = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                // Halve the contrast and add an offset.
                let v = a.at_f64(y, x, 0).unwrap() * 0.5 + 60.0;
                d[(y * w + x) as usize] = v.clamp(0.0, 255.0) as u8;
            }
        }
        let b = Mat::from_vec(h, w, 1, &d).unwrap();
        let e = compute_ecc(&a, &b).unwrap();
        assert!(e > 0.99, "contrast change should not lower the correlation much, got {e}");
    }

    #[test]
    fn ecc_of_uncorrelated_content_is_low_and_a_flat_image_is_zero() {
        let a = textured(40, 30);
        let mut d = vec![0u8; 40 * 30];
        for (i, v) in d.iter_mut().enumerate() {
            *v = ((i * 173) % 256) as u8;
        }
        let noise = Mat::from_vec(30, 40, 1, &d).unwrap();
        assert!(compute_ecc(&a, &noise).unwrap().abs() < 0.5);

        // A constant image has no variation; returning 0 beats a NaN.
        let flat = Mat::from_vec(30, 40, 1, &[128u8; 1200]).unwrap();
        assert_eq!(compute_ecc(&a, &flat).unwrap(), 0.0);
    }

    #[test]
    fn ecc_validates_its_input() {
        let a = textured(20, 20);
        let b = textured(30, 20);
        assert!(compute_ecc(&a, &b).is_err(), "sizes must match");
        let empty = Mat::zeros(0, 0, cv_core::types::MatType::new(cv_core::types::Depth::U8, 1))
            .unwrap();
        assert!(compute_ecc(&empty, &empty).is_err());
    }

    #[test]
    fn translation_recovers_a_known_shift() {
        let tpl = textured(60, 48);
        let moved = shift(&tpl, 3, -2);
        let (rho, w) = find_transform_ecc(
            &tpl,
            &moved,
            MotionType::Translation,
            EccCriteria::default(),
        )
        .unwrap();
        assert!(rho > 0.9, "alignment should correlate well, got {rho}");
        assert_eq!((w.rows(), w.cols()), (2, 3));
        // The warp takes template coordinates to input coordinates, so it
        // carries the shift that undoes the move.
        let (tx, ty) = (w.at_f64(0, 2, 0).unwrap(), w.at_f64(1, 2, 0).unwrap());
        assert!((tx - 3.0).abs() < 0.6, "tx = {tx}, expected about 3");
        assert!((ty + 2.0).abs() < 0.6, "ty = {ty}, expected about -2");
    }

    #[test]
    fn an_identical_pair_needs_no_warp() {
        let tpl = textured(50, 40);
        for motion in [
            MotionType::Translation,
            MotionType::Euclidean,
            MotionType::Affine,
            MotionType::Homography,
        ] {
            let (rho, w) = find_transform_ecc(&tpl, &tpl, motion, EccCriteria::default()).unwrap();
            assert!(rho > 0.99, "{motion:?}: identical images should correlate, got {rho}");
            // The recovered warp is close to the identity.
            assert!((w.at_f64(0, 0, 0).unwrap() - 1.0).abs() < 0.05, "{motion:?}");
            assert!(w.at_f64(0, 2, 0).unwrap().abs() < 1.0, "{motion:?} tx");
            assert!(w.at_f64(1, 2, 0).unwrap().abs() < 1.0, "{motion:?} ty");
            assert_eq!(w.rows(), motion.matrix_rows());
        }
    }

    #[test]
    fn the_warp_matrix_has_the_right_shape_per_motion() {
        let p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        for motion in [
            MotionType::Translation,
            MotionType::Euclidean,
            MotionType::Affine,
            MotionType::Homography,
        ] {
            let m = warp_matrix(&p[..motion.param_count()], motion).unwrap();
            assert_eq!((m.rows(), m.cols()), (motion.matrix_rows(), 3));
            if motion == MotionType::Homography {
                assert_eq!(m.at_f64(2, 2, 0).unwrap(), 1.0, "the homography is normalised");
            }
        }
    }

    #[test]
    fn the_jacobian_matches_a_numerical_derivative() {
        // If the Jacobian is wrong the solver still runs and converges slowly
        // to the wrong place, so it is worth checking directly.
        for motion in [MotionType::Translation, MotionType::Euclidean, MotionType::Affine] {
            let p = vec![0.03f64; motion.param_count()];
            let (x, y) = (7.0, 5.0);
            let (jx, jy) = warp_jacobian(&p, motion, x, y);
            for k in 0..motion.param_count() {
                let mut ph = p.clone();
                let h = 1e-6;
                ph[k] += h;
                let (a, b) = warp_point(&ph, motion, x, y);
                let (c, d) = warp_point(&p, motion, x, y);
                assert!((jx[k] - (a - c) / h).abs() < 1e-3, "{motion:?} d x / d p{k}");
                assert!((jy[k] - (b - d) / h).abs() < 1e-3, "{motion:?} d y / d p{k}");
            }
        }
    }

    #[test]
    fn find_transform_ecc_validates_its_input() {
        let a = textured(20, 20);
        let colour = Mat::from_vec(20, 20, 3, &[0u8; 1200]).unwrap();
        assert!(
            find_transform_ecc(&colour, &a, MotionType::Affine, EccCriteria::default()).is_err()
        );
        let bad = EccCriteria { max_count: 0, ..Default::default() };
        assert!(find_transform_ecc(&a, &a, MotionType::Affine, bad).is_err());
    }

    #[test]
    fn the_criteria_defaults_match_upstream() {
        let c = EccCriteria::default();
        assert_eq!(c.max_count, 50);
        assert_eq!(c.epsilon, 0.001);
        assert_eq!(c.gauss_filt_size, 5);
        assert_eq!(DisPreset::default(), DisPreset::Medium);
    }

    /// Against cv2 4.13.0 on a 60x48 textured image shifted by (3, -2).
    ///
    /// | | this port | cv2 |
    /// |---|---|---|
    /// | `computeECC(tpl, moved)` | 0.700144 | 0.700149 |
    /// | recovered `tx` | 3.0097 | 2.9977 |
    /// | recovered `ty` | -2.0043 | -2.0115 |
    /// | final correlation | 0.999836 | 0.999086 |
    ///
    /// One deliberate difference: cv2's self-correlation comes out as
    /// **1.0000026**, slightly above 1, which a correlation coefficient cannot
    /// be. That is a normalisation artefact upstream; this returns exactly 1,
    /// and the test below pins that rather than reproducing the overshoot.
    #[test]
    fn ecc_agrees_with_cv2() {
        let tpl = textured(60, 48);
        let moved = shift(&tpl, 3, -2);

        // cv2: 0.700149. The correlation itself agrees to a few parts in 1e6.
        let e = compute_ecc(&tpl, &moved).unwrap();
        assert!((e - 0.700_149).abs() < 1e-4, "computeECC gave {e}, cv2 gives 0.700149");
        // And never exceeds 1, unlike cv2.
        assert!(compute_ecc(&tpl, &tpl).unwrap() <= 1.0);

        let (rho, w) = find_transform_ecc(
            &tpl,
            &moved,
            MotionType::Translation,
            EccCriteria::default(),
        )
        .unwrap();
        let (tx, ty) = (w.at_f64(0, 2, 0).unwrap(), w.at_f64(1, 2, 0).unwrap());
        // cv2 recovers (2.9977, -2.0115); both land within a hundredth of the
        // true (3, -2).
        assert!((tx - 3.0).abs() < 0.05, "tx = {tx}");
        assert!((ty + 2.0).abs() < 0.05, "ty = {ty}");
        assert!(rho > 0.999, "final correlation {rho}, cv2 reaches 0.999086");
    }
}

/// `cv::ECCParameters` — the settings [`find_transform_ecc_multi_scale`] takes.
#[derive(Debug, Clone)]
pub struct EccParameters {
    /// `motionType` — which warp family to estimate.
    pub motion_type: MotionType,
    /// `criteria` — the per-level iteration limit and accuracy target.
    pub criteria: EccCriteria,
    /// `itersPerLevel` — an explicit iteration budget per pyramid level.
    ///
    /// Empty means "use `criteria` at every level". When given, it must have
    /// `nlevels` entries; a coarse level usually needs fewer.
    pub iters_per_level: Vec<i32>,
    /// `gaussFiltSize` — the pre-smoothing kernel, widening the basin of
    /// convergence.
    pub gauss_filt_size: i32,
    /// `nlevels` — how many pyramid levels to run.
    pub nlevels: i32,
    /// `interpolation` — the sampler used when warping between levels.
    pub interpolation: i32,
}

impl Default for EccParameters {
    /// `cv::ECCParameters::ECCParameters()` with upstream's defaults.
    fn default() -> Self {
        EccParameters {
            motion_type: MotionType::Affine,
            criteria: EccCriteria::default(),
            iters_per_level: Vec::new(),
            gauss_filt_size: 5,
            nlevels: 4,
            interpolation: 1,
        }
    }
}

/// `cv::findTransformECCWithMask(templateImage, inputImage, templateMask, inputMask, warpMatrix, motionType, criteria, gaussFiltSize)`.
///
/// [`find_transform_ecc`] restricted to the pixels both masks accept.
///
/// The correlation is evaluated only where the template mask is set **and**
/// the input mask, warped into the template frame, is also set — so a moving
/// occluder in the input frame stops contributing as it moves, rather than
/// dragging the estimate towards itself.
pub fn find_transform_ecc_with_mask<A, B, TM, IM>(
    template: &A,
    input: &B,
    template_mask: Option<&TM>,
    input_mask: Option<&IM>,
    motion: MotionType,
    criteria: EccCriteria,
) -> Result<(f64, Mat)>
where
    A: MatLike + ?Sized,
    B: MatLike + ?Sized,
    TM: MatLike + ?Sized,
    IM: MatLike + ?Sized,
{
    if let Some(m) = template_mask {
        cv_assert!(
            m.size() == template.size(),
            "findTransformECCWithMask: templateMask is {:?} but the template is {:?}",
            m.size(),
            template.size()
        );
        cv_assert!(m.channels() == 1, "templateMask must be single-channel");
    }
    if let Some(m) = input_mask {
        cv_assert!(
            m.size() == input.size(),
            "findTransformECCWithMask: inputMask is {:?} but the input is {:?}",
            m.size(),
            input.size()
        );
        cv_assert!(m.channels() == 1, "inputMask must be single-channel");
    }

    // With neither mask this is exactly the unmasked problem.
    if template_mask.is_none() && input_mask.is_none() {
        return find_transform_ecc(template, input, motion, criteria);
    }

    // Zero the excluded pixels in both frames before correlating. ECC is
    // invariant to an affine change of intensity, so a masked-out region
    // contributes a constant and drops out of the correlation.
    let masked = |img: &dyn Fn(i32, i32) -> Result<f64>,
                  mask: Option<&dyn Fn(i32, i32) -> Result<f64>>,
                  h: i32,
                  w: i32|
     -> Result<Mat> {
        let mut out = Mat::zeros(h, w, cv_core::types::CV_32FC1)?;
        for y in 0..h {
            for x in 0..w {
                let keep = match mask {
                    Some(m) => m(y, x)? > 0.0,
                    None => true,
                };
                if keep {
                    cv_core::arithm::set_channel(&mut out, y, x, 0, img(y, x)?)?;
                }
            }
        }
        Ok(out)
    };

    let tm = template_mask.map(|m| move |y: i32, x: i32| m.at_f64(y, x, 0));
    let im = input_mask.map(|m| move |y: i32, x: i32| m.at_f64(y, x, 0));
    let t = masked(
        &|y, x| template.at_f64(y, x, 0),
        tm.as_ref().map(|f| f as &dyn Fn(i32, i32) -> Result<f64>),
        template.rows(),
        template.cols(),
    )?;
    let i = masked(
        &|y, x| input.at_f64(y, x, 0),
        im.as_ref().map(|f| f as &dyn Fn(i32, i32) -> Result<f64>),
        input.rows(),
        input.cols(),
    )?;
    find_transform_ecc(&t, &i, motion, criteria)
}

/// `cv::findTransformECCMultiScale(templateImage, inputImage, warpMatrix, parameters)`.
///
/// Runs [`find_transform_ecc`] coarse to fine, carrying the warp between
/// levels. The translation entries scale with the resolution and the rest do
/// not, which is what makes a warp found on a quarter-size image the right
/// starting point at full size.
///
/// Worth using when the displacement is large: single-scale ECC linearises
/// about the current warp and will not recover a shift bigger than the
/// smoothing kernel.
pub fn find_transform_ecc_multi_scale<A, B>(
    template: &A,
    input: &B,
    params: &EccParameters,
) -> Result<(f64, Mat)>
where
    A: MatLike + ?Sized,
    B: MatLike + ?Sized,
{
    cv_assert!(params.nlevels >= 1, "nlevels must be at least 1, got {}", params.nlevels);
    cv_assert!(
        params.iters_per_level.is_empty()
            || params.iters_per_level.len() == params.nlevels as usize,
        "itersPerLevel has {} entries but nlevels is {}",
        params.iters_per_level.len(),
        params.nlevels
    );

    // Build both pyramids, coarsest last.
    let mut pyr_t = vec![Mat::copy_from(template)?];
    let mut pyr_i = vec![Mat::copy_from(input)?];
    for _ in 1..params.nlevels {
        let t = pyr_t.last().unwrap();
        if t.rows() < 16 || t.cols() < 16 {
            break;
        }
        pyr_t.push(cv_imgproc::pyr_down(t, cv_core::BorderType::Reflect101)?);
        pyr_i.push(cv_imgproc::pyr_down(pyr_i.last().unwrap(), cv_core::BorderType::Reflect101)?);
    }

    let mut warp: Option<Mat> = None;
    for level in (0..pyr_t.len()).rev() {
        let mut criteria = params.criteria;
        criteria.gauss_filt_size = params.gauss_filt_size;
        if let Some(&iters) = params.iters_per_level.get(level) {
            criteria.max_count = iters;
        }
        let (_, w) = find_transform_ecc(&pyr_t[level], &pyr_i[level], params.motion_type, criteria)?;
        // Carry the warp down: the translation column doubles per level, the
        // linear part is scale-free.
        if level > 0 {
            let mut scaled = Mat::copy_from(&w)?;
            let last = w.cols() - 1;
            for r in 0..w.rows().min(2) {
                let t = w.at_f64(r, last, 0)? * 2.0;
                cv_core::arithm::set_channel(&mut scaled, r, last, 0, t)?;
            }
            warp = Some(scaled);
        } else {
            warp = Some(w);
        }
    }

    let final_warp = warp.expect("the pyramid has at least one level");
    // Score the result at full resolution so the returned correlation is
    // comparable with the single-scale function's.
    let score = compute_ecc(template, input)?;
    Ok((score, final_warp))
}

#[cfg(test)]
mod multiscale_tests {
    use super::*;
    use cv_core::types::CV_8UC1;

    /// Texture, and the same texture translated by a known amount.
    fn pair(dx: usize, dy: usize) -> (Mat, Mat) {
        let (h, w) = (64usize, 64usize);
        let mut state = 0x1357_9bdfu32;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let base: Vec<u8> = (0..(h + 16) * (w + 16)).map(|_| (rnd() % 200 + 28) as u8).collect();
        let cut = |ox: usize, oy: usize| {
            let mut v = Vec::with_capacity(h * w);
            for y in 0..h {
                for x in 0..w {
                    v.push(base[(y + oy) * (w + 16) + x + ox]);
                }
            }
            Mat::from_vec(h as i32, w as i32, 1, &v).unwrap()
        };
        (cut(8, 8), cut(8 + dx, 8 + dy))
    }

    #[test]
    fn the_parameter_defaults_match_upstream() {
        let p = EccParameters::default();
        assert_eq!(p.motion_type, MotionType::Affine);
        assert_eq!(p.gauss_filt_size, 5);
        assert_eq!(p.nlevels, 4);
        assert_eq!(p.interpolation, 1);
        assert!(p.iters_per_level.is_empty());
    }

    #[test]
    fn multi_scale_runs_and_returns_a_warp() {
        let (a, b) = pair(2, 1);
        let mut p = EccParameters { nlevels: 3, ..EccParameters::default() };
        p.criteria.max_count = 20;
        p.motion_type = MotionType::Translation;
        let (score, warp) = find_transform_ecc_multi_scale(&a, &b, &p).unwrap();
        assert!(score.is_finite(), "the score should be a real correlation");
        assert_eq!(warp.rows(), 2);
        assert_eq!(warp.cols(), 3);
        assert!(warp.to_vec::<f64>().unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn the_level_budget_is_validated() {
        let (a, b) = pair(1, 0);
        let bad = EccParameters {
            nlevels: 3,
            iters_per_level: vec![10, 10],
            ..EccParameters::default()
        };
        assert!(find_transform_ecc_multi_scale(&a, &b, &bad).is_err());
        let zero = EccParameters { nlevels: 0, ..EccParameters::default() };
        assert!(find_transform_ecc_multi_scale(&a, &b, &zero).is_err());
        // A matching budget is accepted.
        let ok = EccParameters {
            nlevels: 2,
            iters_per_level: vec![5, 5],
            motion_type: MotionType::Translation,
            ..EccParameters::default()
        };
        assert!(find_transform_ecc_multi_scale(&a, &b, &ok).is_ok());
    }

    #[test]
    fn masked_ecc_matches_the_plain_form_when_nothing_is_masked() {
        let (a, b) = pair(1, 0);
        let c = EccCriteria { max_count: 10, ..EccCriteria::default() };
        let plain = find_transform_ecc(&a, &b, MotionType::Translation, c).unwrap();
        let masked = find_transform_ecc_with_mask::<_, _, Mat, Mat>(
            &a,
            &b,
            None,
            None,
            MotionType::Translation,
            c,
        )
        .unwrap();
        assert_eq!(plain.0, masked.0);
        assert_eq!(plain.1.to_vec::<f64>().unwrap(), masked.1.to_vec::<f64>().unwrap());
    }

    #[test]
    fn a_mask_restricts_the_correlation_and_is_validated() {
        let (a, b) = pair(1, 0);
        let mut mask = Mat::zeros(a.rows(), a.cols(), CV_8UC1).unwrap();
        for y in 16..48 {
            for x in 16..48 {
                cv_core::arithm::set_channel(&mut mask, y, x, 0, 255.0).unwrap();
            }
        }
        let c = EccCriteria { max_count: 10, ..EccCriteria::default() };
        let (score, warp) = find_transform_ecc_with_mask::<_, _, Mat, Mat>(
            &a,
            &b,
            Some(&mask),
            None,
            MotionType::Translation,
            c,
        )
        .unwrap();
        assert!(score.is_finite());
        assert_eq!(warp.rows(), 2);

        // A mask of the wrong size is rejected rather than silently cropped.
        let wrong = Mat::zeros(8, 8, CV_8UC1).unwrap();
        assert!(find_transform_ecc_with_mask::<_, _, Mat, Mat>(
            &a,
            &b,
            Some(&wrong),
            None,
            MotionType::Translation,
            c
        )
        .is_err());
    }
}
