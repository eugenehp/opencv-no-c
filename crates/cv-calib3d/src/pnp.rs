//! Pose from point correspondences, and the calibration flag surface
//! (`modules/calib3d/src/solvepnp.cpp`, `five-point.cpp`, `calibration.cpp`).

use cv_core::{cv_assert, Result};

use crate::mat3;
use crate::{mat3_mul, project_points, Mat3};

/// `cv::SolvePnPMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolvePnpMethod {
    /// `cv::SOLVEPNP_ITERATIVE` — Levenberg-Marquardt from a linear guess.
    #[default]
    Iterative,
    /// `cv::SOLVEPNP_EPNP`
    Epnp,
    /// `cv::SOLVEPNP_P3P` — exactly three points plus one to disambiguate.
    P3p,
    /// `cv::SOLVEPNP_DLS`
    Dls,
    /// `cv::SOLVEPNP_UPNP`
    Upnp,
    /// `cv::SOLVEPNP_AP3P`
    Ap3p,
    /// `cv::SOLVEPNP_IPPE` — for planar objects.
    Ippe,
    /// `cv::SOLVEPNP_IPPE_SQUARE` — for a square marker.
    IppeSquare,
    /// `cv::SOLVEPNP_SQPNP`
    SqPnp,
}

/// `cv::CALIB_*` flags for `calibrateCamera` and `stereoCalibrate`.
///
/// Each fixes or enables part of the model. Fixing a parameter you have no
/// data to constrain is usually better than letting the optimiser wander:
/// a short calibration sequence cannot separate focal length from distance,
/// and leaving both free produces a confident, wrong answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibFlag {
    /// `cv::CALIB_USE_INTRINSIC_GUESS`
    UseIntrinsicGuess,
    /// `cv::CALIB_FIX_ASPECT_RATIO`
    FixAspectRatio,
    /// `cv::CALIB_FIX_PRINCIPAL_POINT`
    FixPrincipalPoint,
    /// `cv::CALIB_ZERO_TANGENT_DIST`
    ZeroTangentDist,
    /// `cv::CALIB_FIX_FOCAL_LENGTH`
    FixFocalLength,
    /// `cv::CALIB_FIX_K1`
    FixK1,
    /// `cv::CALIB_FIX_K2`
    FixK2,
    /// `cv::CALIB_FIX_K3`
    FixK3,
    /// `cv::CALIB_FIX_K4`
    FixK4,
    /// `cv::CALIB_FIX_K5`
    FixK5,
    /// `cv::CALIB_FIX_K6`
    FixK6,
    /// `cv::CALIB_RATIONAL_MODEL` — the six-term radial model.
    RationalModel,
    /// `cv::CALIB_THIN_PRISM_MODEL`
    ThinPrismModel,
    /// `cv::CALIB_FIX_S1_S2_S3_S4`
    FixS1S2S3S4,
    /// `cv::CALIB_TILTED_MODEL`
    TiltedModel,
    /// `cv::CALIB_FIX_TAUX_TAUY`
    FixTauxTauy,
    /// `cv::CALIB_FIX_INTRINSIC` — stereo only; trust each camera's intrinsics.
    FixIntrinsic,
    /// `cv::CALIB_SAME_FOCAL_LENGTH`
    SameFocalLength,
    /// `cv::CALIB_FIX_TANGENT_DIST`
    FixTangentDist,
    /// `cv::CALIB_FIX_SKEW` — fisheye only.
    FixSkew,
    /// `cv::CALIB_RECOMPUTE_EXTRINSIC` — fisheye only.
    RecomputeExtrinsic,
    /// `cv::CALIB_CHECK_COND` — fisheye only.
    CheckCond,
}

/// `cv::CALIB_CB_*` flags for chessboard and circle-grid detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibCbFlag {
    /// `cv::CALIB_CB_ADAPTIVE_THRESH`
    AdaptiveThresh,
    /// `cv::CALIB_CB_NORMALIZE_IMAGE`
    NormalizeImage,
    /// `cv::CALIB_CB_FILTER_QUADS`
    FilterQuads,
    /// `cv::CALIB_CB_FAST_CHECK`
    FastCheck,
    /// `cv::CALIB_CB_EXHAUSTIVE`
    Exhaustive,
    /// `cv::CALIB_CB_ACCURACY`
    Accuracy,
    /// `cv::CALIB_CB_LARGER`
    Larger,
    /// `cv::CALIB_CB_MARKER`
    Marker,
    /// `cv::CALIB_CB_PLAIN`
    Plain,
    /// `cv::CALIB_CB_SYMMETRIC_GRID`
    SymmetricGrid,
    /// `cv::CALIB_CB_ASYMMETRIC_GRID`
    AsymmetricGrid,
    /// `cv::CALIB_CB_CLUSTERING`
    Clustering,
}

/// `cv::HandEyeCalibrationMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HandEyeMethod {
    /// `cv::CALIB_HAND_EYE_TSAI`
    #[default]
    Tsai,
    /// `cv::CALIB_HAND_EYE_PARK`
    Park,
    /// `cv::CALIB_HAND_EYE_HORAUD`
    Horaud,
    /// `cv::CALIB_HAND_EYE_ANDREFF`
    Andreff,
    /// `cv::CALIB_HAND_EYE_DANIILIDIS`
    Daniilidis,
}

/// `cv::RobotWorldHandEyeCalibrationMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RobotWorldHandEyeMethod {
    /// `cv::CALIB_ROBOT_WORLD_HAND_EYE_SHAH`
    #[default]
    Shah,
    /// `cv::CALIB_ROBOT_WORLD_HAND_EYE_LI`
    Li,
}

/// Mean reprojection error of a pose, in pixels.
fn reprojection_error(
    object: &[[f64; 3]],
    image: &[(f64, f64)],
    rvec: [f64; 3],
    tvec: [f64; 3],
    k: &Mat3,
    dist: [f64; 5],
) -> f64 {
    let proj = project_points(object, rvec, tvec, k, dist);
    let s: f64 = proj
        .iter()
        .zip(image)
        .map(|(p, q)| (p.0 - q.0).powi(2) + (p.1 - q.1).powi(2))
        .sum();
    (s / image.len() as f64).sqrt()
}

/// `cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, flags)`.
///
/// Returns `(rvec, tvec)` taking a point from object coordinates into the
/// camera's frame.
///
/// # How it works
///
/// Seeds from [`crate::solve_pnp`]'s DLT solution, then runs
/// Levenberg-Marquardt on the reprojection error with numerical derivatives.
/// Unlike the DLT alone this accounts for lens distortion, which is what makes
/// it usable on a real camera.
///
/// Upstream offers several closed-form solvers as well; [`SolvePnpMethod`]
/// names them all, and anything other than [`SolvePnpMethod::Iterative`] is
/// refused rather than silently substituted — a caller who asked for P3P and
/// got iterative would have no way to tell.
pub fn solve_pnp_refine(
    object: &[[f64; 3]],
    image: &[(f64, f64)],
    k: &Mat3,
    dist: [f64; 5],
    method: SolvePnpMethod,
) -> Result<([f64; 3], [f64; 3])> {
    cv_assert!(
        object.len() == image.len(),
        "solvePnP needs matching counts, got {} and {}",
        object.len(),
        image.len()
    );
    cv_assert!(object.len() >= 4, "solvePnP needs at least 4 points, got {}", object.len());
    cv_assert!(
        method == SolvePnpMethod::Iterative,
        "only SOLVEPNP_ITERATIVE is implemented; {method:?} is named but not available"
    );

    // A crude but well-conditioned start: place the object along the optical
    // axis at a depth implied by its apparent size.
    let centroid = {
        let n = object.len() as f64;
        [
            object.iter().map(|p| p[0]).sum::<f64>() / n,
            object.iter().map(|p| p[1]).sum::<f64>() / n,
            object.iter().map(|p| p[2]).sum::<f64>() / n,
        ]
    };
    let obj_spread = (object
        .iter()
        .map(|p| {
            (p[0] - centroid[0]).powi(2) + (p[1] - centroid[1]).powi(2) + (p[2] - centroid[2]).powi(2)
        })
        .sum::<f64>()
        / object.len() as f64)
        .sqrt()
        .max(1e-9);
    let (icx, icy) = {
        let n = image.len() as f64;
        (image.iter().map(|p| p.0).sum::<f64>() / n, image.iter().map(|p| p.1).sum::<f64>() / n)
    };
    let img_spread = (image
        .iter()
        .map(|p| (p.0 - icx).powi(2) + (p.1 - icy).powi(2))
        .sum::<f64>()
        / image.len() as f64)
        .sqrt()
        .max(1e-9);
    // Apparent size shrinks with distance, so depth ~ f * size / apparent size.
    let z = k[0][0] * obj_spread / img_spread;
    let mut p = match crate::solve_pnp(object, image, k) {
        // The DLT ignores distortion but lands close enough to refine from.
        Ok(pose) => [
            pose.rvec[0], pose.rvec[1], pose.rvec[2],
            pose.tvec[0], pose.tvec[1], pose.tvec[2],
        ],
        // Fall back to placing the object along the optical axis at the depth
        // its apparent size implies.
        Err(_) => [
            0.0,
            0.0,
            0.0,
            (icx - k[0][2]) / k[0][0] * z - centroid[0],
            (icy - k[1][2]) / k[1][1] * z - centroid[1],
            z - centroid[2],
        ],
    };

    let cost = |p: &[f64; 6]| {
        reprojection_error(object, image, [p[0], p[1], p[2]], [p[3], p[4], p[5]], k, dist)
    };
    let mut best = cost(&p);
    let mut lambda = 1e-3f64;

    // Levenberg-Marquardt with numerical Jacobians. The pose has six
    // parameters and the residual is cheap, so a finite difference is
    // simpler than deriving the analytic form and no less accurate here.
    for _ in 0..200 {
        let mut jac = [[0.0f64; 6]; 6];
        let mut grad = [0.0f64; 6];
        let h = 1e-6;
        let mut resid = Vec::with_capacity(image.len() * 2);
        let proj = project_points(object, [p[0], p[1], p[2]], [p[3], p[4], p[5]], k, dist);
        for (a, b) in proj.iter().zip(image) {
            resid.push(b.0 - a.0);
            resid.push(b.1 - a.1);
        }
        let mut cols: Vec<Vec<f64>> = Vec::with_capacity(6);
        for i in 0..6 {
            let mut q = *p_ref(&p);
            q[i] += h;
            let pj = project_points(object, [q[0], q[1], q[2]], [q[3], q[4], q[5]], k, dist);
            let mut col = Vec::with_capacity(resid.len());
            for (a, b) in pj.iter().zip(&proj) {
                col.push((a.0 - b.0) / h);
                col.push((a.1 - b.1) / h);
            }
            cols.push(col);
        }
        for (i, row) in jac.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = cols[i].iter().zip(&cols[j]).map(|(a, b)| a * b).sum();
            }
        }
        for (i, g) in grad.iter_mut().enumerate() {
            *g = cols[i].iter().zip(&resid).map(|(a, b)| a * b).sum();
        }

        let mut improved = false;
        for _ in 0..12 {
            let mut a = jac;
            for (i, row) in a.iter_mut().enumerate() {
                row[i] *= 1.0 + lambda;
            }
            let Some(d) = solve6(&mut a, &mut grad.clone()) else { break };
            let mut q = p;
            for (t, v) in q.iter_mut().zip(&d) {
                *t += v;
            }
            let c = cost(&q);
            if c < best {
                best = c;
                p = q;
                lambda = (lambda * 0.3).max(1e-12);
                improved = true;
                break;
            }
            lambda *= 10.0;
            if lambda > 1e12 {
                break;
            }
        }
        if !improved {
            break;
        }
    }
    Ok(([p[0], p[1], p[2]], [p[3], p[4], p[5]]))
}

fn p_ref(p: &[f64; 6]) -> &[f64; 6] {
    p
}

/// `cv::solvePnPRansac(...)`.
///
/// Fits the pose on random minimal subsets and keeps the one with the most
/// inliers, so a handful of wrong correspondences cannot drag the answer.
/// Returns `(rvec, tvec, inlier mask)`.
pub fn solve_pnp_ransac(
    object: &[[f64; 3]],
    image: &[(f64, f64)],
    k: &Mat3,
    dist: [f64; 5],
    reproj_threshold: f64,
    iterations: usize,
    rng: &mut cv_core::rng::Rng,
) -> Result<([f64; 3], [f64; 3], Vec<bool>)> {
    cv_assert!(object.len() == image.len(), "point counts must match");
    cv_assert!(object.len() >= 4, "solvePnPRansac needs at least 4 points");
    cv_assert!(reproj_threshold > 0.0, "the threshold must be positive");
    let n = object.len();
    let mut best: (usize, Vec<bool>) = (0, vec![false; n]);

    for _ in 0..iterations.max(1) {
        let mut pick = [0usize; 4];
        for i in 0..4 {
            loop {
                let c = (rng.next_u32() as usize) % n;
                if !pick[..i].contains(&c) {
                    pick[i] = c;
                    break;
                }
            }
        }
        let o: Vec<[f64; 3]> = pick.iter().map(|&i| object[i]).collect();
        let m: Vec<(f64, f64)> = pick.iter().map(|&i| image[i]).collect();
        let Ok((r, t)) = solve_pnp_refine(&o, &m, k, dist, SolvePnpMethod::Iterative) else { continue };
        let proj = project_points(object, r, t, k, dist);
        let mask: Vec<bool> = proj
            .iter()
            .zip(image)
            .map(|(a, b)| {
                ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt() <= reproj_threshold
            })
            .collect();
        let c = mask.iter().filter(|&&b| b).count();
        if c > best.0 {
            best = (c, mask);
        }
    }
    cv_assert!(best.0 >= 4, "RANSAC found only {} inliers", best.0);

    // Refit on the consensus set.
    let o: Vec<[f64; 3]> = (0..n).filter(|&i| best.1[i]).map(|i| object[i]).collect();
    let m: Vec<(f64, f64)> = (0..n).filter(|&i| best.1[i]).map(|i| image[i]).collect();
    let (r, t) = solve_pnp_refine(&o, &m, k, dist, SolvePnpMethod::Iterative)?;
    Ok((r, t, best.1))
}

/// `cv::decomposeEssentialMat(E, R1, R2, t)`.
///
/// An essential matrix admits four pose candidates — two rotations and a
/// translation sign. Only a cheirality test against real correspondences picks
/// the right one, so all four are returned rather than one guessed.
pub fn decompose_essential_mat(e: &Mat3) -> Result<(Mat3, Mat3, [f64; 3])> {
    // SVD of E; the rotations follow from U W V^T and U W^T V^T.
    let m = cv_core::Mat::from_vec(
        3,
        3,
        1,
        &[e[0][0], e[0][1], e[0][2], e[1][0], e[1][1], e[1][2], e[2][0], e[2][1], e[2][2]],
    )?;
    let svd = cv_core::linalg2::svd_decomp(&m)?;
    let get = |mat: &cv_core::Mat, r: i32, c: i32| -> f64 {
        cv_core::MatLike::at_f64(mat, r, c, 0).unwrap_or(0.0)
    };
    let mut u = [[0.0f64; 3]; 3];
    let mut vt = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            u[i][j] = get(&svd.u, i as i32, j as i32);
            vt[i][j] = get(&svd.vt, i as i32, j as i32);
        }
    }
    // W is the 90-degree rotation about z.
    let w = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let wt = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let mut r1 = mat3_mul(&mat3_mul(&u, &w), &vt);
    let mut r2 = mat3_mul(&mat3_mul(&u, &wt), &vt);
    // A rotation has determinant +1; the SVD does not guarantee the sign.
    for r in [&mut r1, &mut r2] {
        let d = mat3::det3(r);
        if d < 0.0 {
            for row in r.iter_mut() {
                for v in row.iter_mut() {
                    *v = -*v;
                }
            }
        }
    }
    // The translation is U's last column, up to sign and scale.
    let t = [u[0][2], u[1][2], u[2][2]];
    Ok((r1, r2, t))
}


/// `cv::recoverPose(E, points1, points2, cameraMatrix, R, t, mask)`.
///
/// Picks among [`decompose_essential_mat`]'s four candidates the one that puts
/// the most points in front of both cameras — the cheirality test. Returns the
/// pose and the count that passed.
pub fn recover_pose(
    e: &Mat3,
    p1: &[(f64, f64)],
    p2: &[(f64, f64)],
    k: &Mat3,
) -> Result<(Mat3, [f64; 3], usize)> {
    cv_assert!(p1.len() == p2.len(), "point counts must match");
    cv_assert!(!p1.is_empty(), "recoverPose needs correspondences");
    let (r1, r2, t) = decompose_essential_mat(e)?;
    let nt = [-t[0], -t[1], -t[2]];

    let mut best = (0usize, r1, t);
    for r in [r1, r2] {
        for tv in [t, nt] {
            let mut good = 0usize;
            for (a, b) in p1.iter().zip(p2) {
                // Normalised rays.
                let x1 = [(a.0 - k[0][2]) / k[0][0], (a.1 - k[1][2]) / k[1][1], 1.0];
                let x2 = [(b.0 - k[0][2]) / k[0][0], (b.1 - k[1][2]) / k[1][1], 1.0];
                if let Some((z1, z2)) = triangulate_depths(&r, tv, x1, x2) {
                    if z1 > 0.0 && z2 > 0.0 {
                        good += 1;
                    }
                }
            }
            if good > best.0 {
                best = (good, r, tv);
            }
        }
    }
    Ok((best.1, best.2, best.0))
}

/// Depths of a correspondence in both cameras, by least squares on the two rays.
fn triangulate_depths(r: &Mat3, t: [f64; 3], x1: [f64; 3], x2: [f64; 3]) -> Option<(f64, f64)> {
    // x1 * z1 = R^T (x2 * z2 - t), solved for (z1, z2).
    let mut rtx2 = [0.0f64; 3];
    for i in 0..3 {
        rtx2[i] = (0..3).map(|j| r[j][i] * x2[j]).sum();
    }
    let mut rtt = [0.0f64; 3];
    for i in 0..3 {
        rtt[i] = (0..3).map(|j| r[j][i] * t[j]).sum();
    }
    // Least squares for [z1, -z2] against -R^T t.
    let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..3 {
        a11 += x1[i] * x1[i];
        a12 += x1[i] * -rtx2[i];
        a22 += rtx2[i] * rtx2[i];
        b1 += x1[i] * -rtt[i];
        b2 += -rtx2[i] * -rtt[i];
    }
    let det = a11 * a22 - a12 * a12;
    if det.abs() < 1e-12 {
        return None;
    }
    let z1 = (a22 * b1 - a12 * b2) / det;
    let z2 = (a11 * b2 - a12 * b1) / det;
    Some((z1, z2))
}

/// `cv::calibrationMatrixValues(cameraMatrix, imageSize, apertureWidth, apertureHeight, ...)`.
///
/// Turns a camera matrix into the numbers a lens is actually sold by: field of
/// view, focal length in millimetres, principal point in millimetres and the
/// pixel aspect ratio. Needs the physical sensor size, which the matrix alone
/// does not contain.
pub fn calibration_matrix_values(
    k: &Mat3,
    image_width: i32,
    image_height: i32,
    aperture_width: f64,
    aperture_height: f64,
) -> Result<CalibrationValues> {
    cv_assert!(image_width > 0 && image_height > 0, "the image size must be positive");
    cv_assert!(
        aperture_width > 0.0 && aperture_height > 0.0,
        "the sensor size must be positive to convert pixels to millimetres"
    );
    let (mx, my) = (f64::from(image_width) / aperture_width, f64::from(image_height) / aperture_height);
    // Upstream sums the two *half*-angles measured from the actual principal
    // point — `atan2(cx, fx) + atan2(width - cx, fx)` — rather than assuming
    // the principal point is centred and taking `2 * atan(width / (2 * fx))`.
    // The two agree only when `cx` is exactly `width / 2`; with the usual
    // `(w - 1) / 2` convention they never are.
    let (w, h) = (f64::from(image_width), f64::from(image_height));
    let fovx = (k[0][2].atan2(k[0][0]) + (w - k[0][2]).atan2(k[0][0])).to_degrees();
    let fovy = (k[1][2].atan2(k[1][1]) + (h - k[1][2]).atan2(k[1][1])).to_degrees();
    Ok(CalibrationValues {
        fovx,
        fovy,
        focal_length: k[0][0] / mx,
        principal_point: (k[0][2] / mx, k[1][2] / my),
        aspect_ratio: k[1][1] / k[0][0],
    })
}

/// What [`calibration_matrix_values`] reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationValues {
    /// Horizontal field of view, degrees.
    pub fovx: f64,
    /// Vertical field of view, degrees.
    pub fovy: f64,
    /// Focal length in the same units as the aperture.
    pub focal_length: f64,
    /// Principal point in the same units as the aperture.
    pub principal_point: (f64, f64),
    /// `fy / fx`.
    pub aspect_ratio: f64,
}

/// Solve a 6x6 system in place.
fn solve6(a: &mut [[f64; 6]; 6], b: &mut [f64; 6]) -> Option<[f64; 6]> {
    for col in 0..6 {
        let mut piv = col;
        for r in (col + 1)..6 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in (col + 1)..6 {
            let f = a[r][col] / a[col][col];
            let pivot = a[col];
            for (k, v) in a[r].iter_mut().enumerate().skip(col) {
                *v -= f * pivot[k];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0f64; 6];
    for i in (0..6).rev() {
        let s: f64 = ((i + 1)..6).map(|j| a[i][j] * x[j]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{camera_matrix, rodrigues_to_matrix, rodrigues_to_vector};

    /// A non-planar object, so the pose is well determined.
    fn cube() -> Vec<[f64; 3]> {
        let mut v = Vec::new();
        for z in [0.0, 1.0] {
            for y in [0.0, 1.0] {
                for x in [0.0, 1.0] {
                    v.push([x, y, z]);
                }
            }
        }
        v
    }

    #[test]
    fn solve_pnp_recovers_a_known_pose() {
        let k = camera_matrix(600.0, 600.0, 320.0, 240.0);
        let obj = cube();
        let rvec = [0.12, -0.25, 0.08];
        let tvec = [0.4, -0.3, 7.0];
        let img = project_points(&obj, rvec, tvec, &k, [0.0; 5]);

        let (r, t) = solve_pnp_refine(&obj, &img, &k, [0.0; 5], SolvePnpMethod::Iterative).unwrap();
        // Compare by reprojection: the parameterisation is not unique, the
        // resulting projection is.
        let err = reprojection_error(&obj, &img, r, t, &k, [0.0; 5]);
        assert!(err < 0.01, "reprojection error {err} px");
        for i in 0..3 {
            assert!((t[i] - tvec[i]).abs() < 0.05, "t[{i}] = {}, expected {}", t[i], tvec[i]);
        }
    }

    #[test]
    fn solve_pnp_works_with_distortion() {
        let k = camera_matrix(500.0, 520.0, 300.0, 250.0);
        let dist = [-0.2, 0.05, 0.001, -0.001, 0.0];
        let obj = cube();
        let (rvec, tvec) = ([0.05, 0.3, -0.1], [0.2, 0.1, 6.0]);
        let img = project_points(&obj, rvec, tvec, &k, dist);
        let (r, t) = solve_pnp_refine(&obj, &img, &k, dist, SolvePnpMethod::Iterative).unwrap();
        let err = reprojection_error(&obj, &img, r, t, &k, dist);
        assert!(err < 0.05, "reprojection error {err} px with distortion");
    }

    #[test]
    fn solve_pnp_validates_and_refuses_unimplemented_methods() {
        let k = camera_matrix(600.0, 600.0, 320.0, 240.0);
        let obj = cube();
        let img = project_points(&obj, [0.0; 3], [0.0, 0.0, 6.0], &k, [0.0; 5]);
        assert!(solve_pnp_refine(&obj[..3], &img[..3], &k, [0.0; 5], SolvePnpMethod::Iterative).is_err());
        assert!(solve_pnp_refine(&obj, &img[..4], &k, [0.0; 5], SolvePnpMethod::Iterative).is_err());
        // A named-but-absent method is refused, not silently substituted.
        let e = solve_pnp_refine(&obj, &img, &k, [0.0; 5], SolvePnpMethod::Epnp).unwrap_err();
        assert!(format!("{e}").contains("Epnp"), "the error should name it: {e}");
    }

    #[test]
    fn ransac_rejects_outliers() {
        let k = camera_matrix(600.0, 600.0, 320.0, 240.0);
        let obj: Vec<[f64; 3]> = (0..20)
            .map(|i| {
                let f = i as f64;
                [(i % 4) as f64, (i / 4) as f64, (f * 0.31).sin()]
            })
            .collect();
        let (rvec, tvec) = ([0.1, 0.2, -0.05], [0.3, -0.2, 9.0]);
        let mut img = project_points(&obj, rvec, tvec, &k, [0.0; 5]);
        // Corrupt four correspondences.
        for i in (0..20).step_by(5) {
            img[i] = (img[i].0 + 80.0, img[i].1 - 60.0);
        }
        let mut rng = cv_core::rng::Rng::new(11);
        let (r, t, mask) =
            solve_pnp_ransac(&obj, &img, &k, [0.0; 5], 3.0, 400, &mut rng).unwrap();
        let inliers = mask.iter().filter(|&&b| b).count();
        assert_eq!(inliers, 16, "the four corrupted points must be rejected");
        for i in (0..20).step_by(5) {
            assert!(!mask[i], "point {i} was corrupted and should be an outlier");
        }
        // And the pose fits the surviving points.
        let good: Vec<[f64; 3]> = (0..20).filter(|&i| mask[i]).map(|i| obj[i]).collect();
        let gimg: Vec<(f64, f64)> = (0..20).filter(|&i| mask[i]).map(|i| img[i]).collect();
        assert!(reprojection_error(&good, &gimg, r, t, &k, [0.0; 5]) < 0.1);
    }

    #[test]
    fn an_essential_matrix_decomposes_into_rotations() {
        // Build E = [t]x R from a known pose.
        let r = rodrigues_to_matrix([0.05, 0.1, -0.02]);
        let t = [0.5, 0.1, 0.05];
        let tx = [
            [0.0, -t[2], t[1]],
            [t[2], 0.0, -t[0]],
            [-t[1], t[0], 0.0],
        ];
        let e = mat3_mul(&tx, &r);
        let (r1, r2, tv) = decompose_essential_mat(&e).unwrap();
        // Both candidates are genuine rotations.
        for c in [&r1, &r2] {
            assert!((mat3::det3(c) - 1.0).abs() < 1e-6, "determinant {}", mat3::det3(c));
            for (i, row) in c.iter().enumerate() {
                let n: f64 = row.iter().map(|v| v * v).sum();
                assert!((n - 1.0).abs() < 1e-6, "row {i} norm {n}");
            }
        }
        // The translation direction is recovered up to sign and scale.
        let n = (tv[0] * tv[0] + tv[1] * tv[1] + tv[2] * tv[2]).sqrt();
        assert!((n - 1.0).abs() < 1e-6, "the translation is a unit direction");
        let tn = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        let dot = (0..3).map(|i| tv[i] * t[i] / tn).sum::<f64>().abs();
        assert!(dot > 0.99, "direction cosine {dot} — should be parallel to t");
    }

    #[test]
    fn recover_pose_picks_the_candidate_in_front_of_both_cameras() {
        let k = camera_matrix(600.0, 600.0, 320.0, 240.0);
        let r = rodrigues_to_matrix([0.02, 0.06, -0.01]);
        let t = [0.6, 0.05, 0.02];
        let obj: Vec<[f64; 3]> = (0..14)
            .map(|i| {
                let f = i as f64;
                [(i % 4) as f64 - 1.5, (i / 4) as f64 - 1.0, 6.0 + (f * 1.7).sin() * 1.5]
            })
            .collect();
        let p1 = project_points(&obj, [0.0; 3], [0.0; 3], &k, [0.0; 5]);
        let p2 = project_points(&obj, rodrigues_to_vector(&r), t, &k, [0.0; 5]);
        let tx = [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]];
        let e = mat3_mul(&tx, &r);

        let (rr, _tt, good) = recover_pose(&e, &p1, &p2, &k).unwrap();
        assert!(good >= p1.len() - 1, "only {good} of {} passed cheirality", p1.len());
        // The recovered rotation matches the true one.
        for i in 0..3 {
            for j in 0..3 {
                assert!((rr[i][j] - r[i][j]).abs() < 1e-3, "R[{i}][{j}]");
            }
        }
    }

    #[test]
    fn calibration_matrix_values_converts_to_lens_units() {
        // A 640x480 sensor 6.4 x 4.8 mm: exactly 100 pixels per mm.
        let k = camera_matrix(600.0, 600.0, 320.0, 240.0);
        let v = calibration_matrix_values(&k, 640, 480, 6.4, 4.8).unwrap();
        assert!((v.focal_length - 6.0).abs() < 1e-9, "6 mm lens, got {}", v.focal_length);
        assert!((v.principal_point.0 - 3.2).abs() < 1e-9);
        assert!((v.principal_point.1 - 2.4).abs() < 1e-9);
        assert!((v.aspect_ratio - 1.0).abs() < 1e-12);
        // 2 * atan(320/600) = 56.14 degrees.
        assert!((v.fovx - 56.144_975).abs() < 1e-4, "fovx {}", v.fovx);
        assert!(v.fovy < v.fovx, "the shorter axis has the narrower field");

        assert!(calibration_matrix_values(&k, 0, 480, 6.4, 4.8).is_err());
        assert!(calibration_matrix_values(&k, 640, 480, 0.0, 4.8).is_err());
    }
}
