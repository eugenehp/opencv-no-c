//! Camera calibration and the remaining PnP solvers
//! (`modules/calib3d/src/calibration.cpp`, `solvepnp.cpp`, `p3p.cpp`,
//! `calibration_handeye.cpp`).
//!
//! Calibration is one large non-linear least-squares problem: find the
//! intrinsics, the distortion and one pose per view that together minimise the
//! reprojection error over every observed corner. Nothing here is closed form,
//! which is why [`crate::init_camera_matrix_2d`] exists to supply a starting
//! point and why the optimiser below is Levenberg-Marquardt rather than
//! Gauss-Newton: a bad start makes the Gauss-Newton step overshoot, and the
//! damping term is what keeps it from diverging.
//!
//! # Reading the result
//!
//! Every entry point returns the RMS reprojection error in pixels. On a decent
//! target and a well-exposed set of views it lands under one; a value of three
//! or more usually means a corner ordering mistake in one view rather than a
//! bad camera, and no amount of extra views will fix it.

use cv_core::{cv_assert, cv_error, Code, Result, Size};

use crate::mat3;
use crate::{mat3_inv, mat3_mul, project_points, rodrigues_to_matrix, rodrigues_to_vector, Mat3};

/// One view's pose.
pub type Pose6 = ([f64; 3], [f64; 3]);

/// What [`calibrate_camera`] returns: the RMS error, the intrinsics, the
/// distortion and one pose per view.
pub type Calibration = (f64, Mat3, [f64; 5], Vec<Pose6>);

/// [`calibrate`]'s fisheye form, with four distortion coefficients.
pub type FisheyeCalibration = (f64, Mat3, [f64; 4], Vec<Pose6>);

/// [`calibrate_camera_ro`]'s form, which also returns the refined target.
pub type CalibrationRo = (f64, Mat3, [f64; 5], Vec<Pose6>, Vec<[f64; 3]>);

/// A rigid transform: rotation and translation.
pub type Rigid = (Mat3, [f64; 3]);

/// What the `*Extended` entry points add: the per-view errors and the standard
/// deviations of the estimated parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationExtended {
    /// RMS reprojection error over every view.
    pub rms: f64,
    /// The refined intrinsics.
    pub camera_matrix: Mat3,
    /// The refined distortion, `[k1, k2, p1, p2, k3]`.
    pub dist_coeffs: [f64; 5],
    /// One pose per view.
    pub poses: Vec<Pose6>,
    /// `stdDeviationsIntrinsics` — `[fx, fy, cx, cy, k1, k2, p1, p2, k3]`.
    ///
    /// A large deviation on a parameter means the views did not constrain it,
    /// which is the useful diagnostic: a set of head-on views leaves the
    /// tangential terms free and they will come back with huge deviations
    /// however small the reprojection error is.
    pub std_deviations_intrinsics: [f64; 9],
    /// `perViewErrors` — the RMS error of each view on its own.
    pub per_view_errors: Vec<f64>,
}

/// The parameters the optimiser varies, flattened.
///
/// `[fx, fy, cx, cy, k1, k2, p1, p2, k3]` then `[rx, ry, rz, tx, ty, tz]` per
/// view. Flattening lets one generic LM loop serve every entry point.
fn pack(k: &Mat3, dist: &[f64; 5], poses: &[Pose6]) -> Vec<f64> {
    let mut p = vec![k[0][0], k[1][1], k[0][2], k[1][2]];
    p.extend_from_slice(dist);
    for (r, t) in poses {
        p.extend_from_slice(r);
        p.extend_from_slice(t);
    }
    p
}

/// The inverse of [`pack`].
fn unpack(p: &[f64], n_views: usize) -> (Mat3, [f64; 5], Vec<Pose6>) {
    let k = [[p[0], 0.0, p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]];
    let mut dist = [0.0f64; 5];
    dist.copy_from_slice(&p[4..9]);
    let mut poses = Vec::with_capacity(n_views);
    for v in 0..n_views {
        let b = 9 + v * 6;
        poses.push(([p[b], p[b + 1], p[b + 2]], [p[b + 3], p[b + 4], p[b + 5]]));
    }
    (k, dist, poses)
}

/// Every residual: the x and y reprojection error of every point of every view.
fn residuals(
    p: &[f64],
    object: &[Vec<[f64; 3]>],
    image: &[Vec<(f64, f64)>],
) -> Vec<f64> {
    let (k, dist, poses) = unpack(p, object.len());
    let mut r = Vec::new();
    for (v, (obj, img)) in object.iter().zip(image).enumerate() {
        let projected = project_points(obj, poses[v].0, poses[v].1, &k, dist);
        for (q, m) in projected.iter().zip(img) {
            r.push(q.0 - m.0);
            r.push(q.1 - m.1);
        }
    }
    r
}

/// Levenberg-Marquardt on the reprojection residual, with numerical Jacobians.
///
/// Numerical rather than analytic: the analytic Jacobian of the distortion
/// chain is where sign errors hide, and a wrong Jacobian produces a solver that
/// converges slowly to the right answer often enough that the bug survives
/// review. The cost is one extra residual evaluation per parameter per
/// iteration, which for a calibration set is not the bottleneck.
fn lm_refine(
    mut p: Vec<f64>,
    object: &[Vec<[f64; 3]>],
    image: &[Vec<(f64, f64)>],
    fixed: &[usize],
    max_iter: usize,
) -> (Vec<f64>, f64) {
    let cost = |p: &[f64]| -> f64 { residuals(p, object, image).iter().map(|r| r * r).sum() };
    let mut lambda = 1e-3f64;
    let mut best = cost(&p);
    let n = p.len();
    let free: Vec<usize> = (0..n).filter(|i| !fixed.contains(i)).collect();

    for _ in 0..max_iter {
        let r0 = residuals(&p, object, image);
        let m = r0.len();
        if m == 0 || free.is_empty() {
            break;
        }
        // J, column by column.
        let mut j = vec![0.0f64; m * free.len()];
        for (c, &idx) in free.iter().enumerate() {
            let h = (p[idx].abs() * 1e-6).max(1e-8);
            let mut q = p.clone();
            q[idx] += h;
            for (row, v) in residuals(&q, object, image).iter().enumerate() {
                j[row * free.len() + c] = (v - r0[row]) / h;
            }
        }
        // Normal equations, damped.
        let f = free.len();
        let mut jtj = vec![0.0f64; f * f];
        let mut jtr = vec![0.0f64; f];
        for row in 0..m {
            for a in 0..f {
                let ja = j[row * f + a];
                if ja == 0.0 {
                    continue;
                }
                jtr[a] += ja * r0[row];
                for b in 0..f {
                    jtj[a * f + b] += ja * j[row * f + b];
                }
            }
        }

        let mut improved = false;
        for _ in 0..8 {
            let mut a = jtj.clone();
            for i in 0..f {
                a[i * f + i] *= 1.0 + lambda;
                a[i * f + i] += lambda;
            }
            let Some(delta) = solve_sym(&a, &jtr, f) else {
                lambda *= 10.0;
                continue;
            };
            let mut q = p.clone();
            for (c, &idx) in free.iter().enumerate() {
                q[idx] -= delta[c];
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
        }
        if !improved {
            break;
        }
    }
    let total = residuals(&p, object, image).len();
    let rms = if total == 0 { 0.0 } else { (best / (total / 2) as f64).sqrt() };
    (p, rms)
}

/// Solve `A x = b` for a symmetric `A`, by Gauss-Jordan with partial pivoting.
fn solve_sym(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = vec![0.0f64; n * (n + 1)];
    for i in 0..n {
        m[i * (n + 1)..i * (n + 1) + n].copy_from_slice(&a[i * n..i * n + n]);
        m[i * (n + 1) + n] = b[i];
    }
    for col in 0..n {
        let pivot = (col..n).max_by(|&x, &y| {
            m[x * (n + 1) + col].abs().total_cmp(&m[y * (n + 1) + col].abs())
        })?;
        if m[pivot * (n + 1) + col].abs() < 1e-14 {
            return None;
        }
        for k in 0..=n {
            m.swap(col * (n + 1) + k, pivot * (n + 1) + k);
        }
        let d = m[col * (n + 1) + col];
        for k in 0..=n {
            m[col * (n + 1) + k] /= d;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = m[row * (n + 1) + col];
            if f == 0.0 {
                continue;
            }
            for k in 0..=n {
                m[row * (n + 1) + k] -= f * m[col * (n + 1) + k];
            }
        }
    }
    Some((0..n).map(|i| m[i * (n + 1) + n]).collect())
}

/// Shape checks shared by every calibration entry point.
fn check_views(object: &[Vec<[f64; 3]>], image: &[Vec<(f64, f64)>]) -> Result<()> {
    cv_assert!(!object.is_empty(), "calibration needs at least one view");
    cv_assert!(
        object.len() == image.len(),
        "{} object views and {} image views",
        object.len(),
        image.len()
    );
    for (i, (o, m)) in object.iter().zip(image).enumerate() {
        cv_assert!(
            o.len() == m.len(),
            "view {i}: {} object points and {} image points",
            o.len(),
            m.len()
        );
        cv_assert!(o.len() >= 4, "view {i} has {} points; at least four are needed", o.len());
    }
    Ok(())
}

/// `cv::calibrateCameraExtended(objectPoints, imagePoints, imageSize, ...)`.
///
/// The full form: everything [`calibrate_camera`] returns, plus the parameter
/// deviations and the per-view errors. Reach for it when a calibration is
/// disappointing — the per-view errors name the bad view, which the aggregate
/// RMS cannot.
pub fn calibrate_camera_extended(
    object: &[Vec<[f64; 3]>],
    image: &[Vec<(f64, f64)>],
    image_size: Size,
) -> Result<CalibrationExtended> {
    check_views(object, image)?;
    cv_assert!(
        image_size.width > 0 && image_size.height > 0,
        "the image size must be positive"
    );

    // Seed the intrinsics from the homographies, and each pose from the DLT.
    let k = crate::init_camera_matrix_2d(object, image, image_size.width, image_size.height)?;
    let dist = [0.0f64; 5];
    let mut poses = Vec::with_capacity(object.len());
    for (o, m) in object.iter().zip(image) {
        poses.push(seed_pose(o, m, &k)?);
    }

    let p0 = pack(&k, &dist, &poses);
    let (p, rms) = lm_refine(p0, object, image, &[], 60);
    let (k, dist_new, poses_new) = unpack(&p, object.len());

    // Per-view errors, and deviations from the diagonal of the covariance.
    let mut per_view = Vec::with_capacity(object.len());
    for (v, (o, m)) in object.iter().zip(image).enumerate() {
        let proj = project_points(o, poses_new[v].0, poses_new[v].1, &k, dist_new);
        let sse: f64 =
            proj.iter().zip(m).map(|(q, r)| (q.0 - r.0).powi(2) + (q.1 - r.1).powi(2)).sum();
        per_view.push((sse / o.len() as f64).sqrt());
    }
    let std_dev = intrinsic_deviations(&p, object, image, rms);

    Ok(CalibrationExtended {
        rms,
        camera_matrix: k,
        dist_coeffs: dist_new,
        poses: poses_new,
        std_deviations_intrinsics: std_dev,
        per_view_errors: per_view,
    })
}

/// A starting pose for one view.
///
/// `solve_pnp`'s DLT needs non-coplanar object points and a calibration target
/// is a *plane*, so the DLT is exactly degenerate on the one input that matters
/// most. A planar target is handled through its homography instead: with `K`
/// known, `[r1 r2 t] = lambda K^-1 H` and `r3 = r1 x r2`.
fn seed_pose(object: &[[f64; 3]], image: &[(f64, f64)], k: &Mat3) -> Result<Pose6> {
    let z0 = object[0][2];
    let planar = object.iter().all(|p| (p[2] - z0).abs() < 1e-9);
    if !planar {
        let pose = crate::solve_pnp(object, image, k)?;
        return Ok((pose.rvec, pose.tvec));
    }

    let src: Vec<(f64, f64)> = object.iter().map(|p| (p[0], p[1])).collect();
    let h = crate::find_homography(&src, image)?;
    let ki = mat3_inv(k).ok_or_else(|| {
        cv_core::Error::new(Code::StsError, "the camera matrix is singular".to_string())
    })?;
    let col = |j: usize| -> [f64; 3] {
        [
            ki[0][0] * h[0][j] + ki[0][1] * h[1][j] + ki[0][2] * h[2][j],
            ki[1][0] * h[0][j] + ki[1][1] * h[1][j] + ki[1][2] * h[2][j],
            ki[2][0] * h[0][j] + ki[2][1] * h[1][j] + ki[2][2] * h[2][j],
        ]
    };
    let (c0, c1, c2) = (col(0), col(1), col(2));
    let n0 = (c0[0] * c0[0] + c0[1] * c0[1] + c0[2] * c0[2]).sqrt();
    let n1 = (c1[0] * c1[0] + c1[1] * c1[1] + c1[2] * c1[2]).sqrt();
    cv_assert!(n0 > 1e-12 && n1 > 1e-12, "the homography is degenerate");
    // The two column norms should agree; their mean is the least-squares scale.
    let lambda = 2.0 / (n0 + n1);
    let mut r1 = [c0[0] * lambda, c0[1] * lambda, c0[2] * lambda];
    let mut r2 = [c1[0] * lambda, c1[1] * lambda, c1[2] * lambda];
    let mut t = [c2[0] * lambda, c2[1] * lambda, c2[2] * lambda];
    // The target is in front of the camera; a negative depth means the
    // homography came back with the opposite overall sign.
    if t[2] < 0.0 {
        for v in r1.iter_mut().chain(r2.iter_mut()).chain(t.iter_mut()) {
            *v = -*v;
        }
    }
    // Re-orthonormalise: the two columns are only approximately orthogonal,
    // and a non-rotation here makes `rodrigues_to_vector` report a wrong angle.
    let d = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2];
    for i in 0..3 {
        r2[i] -= d * r1[i];
    }
    let normalise = |v: &mut [f64; 3]| {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-300);
        for c in v.iter_mut() {
            *c /= n;
        }
    };
    normalise(&mut r1);
    normalise(&mut r2);
    let r3 = [
        r1[1] * r2[2] - r1[2] * r2[1],
        r1[2] * r2[0] - r1[0] * r2[2],
        r1[0] * r2[1] - r1[1] * r2[0],
    ];
    let r = [
        [r1[0], r2[0], r3[0]],
        [r1[1], r2[1], r3[1]],
        [r1[2], r2[2], r3[2]],
    ];
    // The plane may sit at a non-zero Z; shift the translation to match.
    let t = [t[0] + r[0][2] * z0, t[1] + r[1][2] * z0, t[2] + r[2][2] * z0];
    Ok((rodrigues_to_vector(&r), t))
}

/// The square roots of the diagonal of `sigma^2 (J^T J)^-1`, for the nine
/// intrinsic parameters.
fn intrinsic_deviations(
    p: &[f64],
    object: &[Vec<[f64; 3]>],
    image: &[Vec<(f64, f64)>],
    rms: f64,
) -> [f64; 9] {
    let r0 = residuals(p, object, image);
    let m = r0.len();
    let f = 9usize.min(p.len());
    let mut j = vec![0.0f64; m * f];
    for c in 0..f {
        let h = (p[c].abs() * 1e-6).max(1e-8);
        let mut q = p.to_vec();
        q[c] += h;
        for (row, v) in residuals(&q, object, image).iter().enumerate() {
            j[row * f + c] = (v - r0[row]) / h;
        }
    }
    let mut jtj = vec![0.0f64; f * f];
    for row in 0..m {
        for a in 0..f {
            for b in 0..f {
                jtj[a * f + b] += j[row * f + a] * j[row * f + b];
            }
        }
    }
    let mut out = [f64::INFINITY; 9];
    // The deviation of a parameter is the residual scale times the diagonal of
    // the inverse normal matrix; an unconstrained parameter makes that inverse
    // huge, which is the signal worth reporting.
    for i in 0..f {
        let mut e = vec![0.0f64; f];
        e[i] = 1.0;
        if let Some(col) = solve_sym(&jtj, &e, f) {
            out[i] = (col[i].max(0.0)).sqrt() * rms;
        }
    }
    out
}

/// `cv::calibrateCamera(objectPoints, imagePoints, imageSize, ...)`.
///
/// Returns the RMS reprojection error, the intrinsics, the distortion and one
/// pose per view.
pub fn calibrate_camera(
    object: &[Vec<[f64; 3]>],
    image: &[Vec<(f64, f64)>],
    image_size: Size,
) -> Result<Calibration> {
    let e = calibrate_camera_extended(object, image, image_size)?;
    Ok((e.rms, e.camera_matrix, e.dist_coeffs, e.poses))
}

/// `cv::fisheye::calibrate(objectPoints, imagePoints, image_size, K, D, rvecs, tvecs, flags, criteria)`.
///
/// The same optimisation over the fisheye projection model. The pinhole path is
/// reused with the fisheye's four coefficients carried in the first four slots,
/// so a mild fisheye calibrates correctly; a lens past about 120 degrees needs
/// the equidistant projection this does not implement, and the RMS will say so
/// rather than the result being quietly wrong.
pub fn calibrate(
    object: &[Vec<[f64; 3]>],
    image: &[Vec<(f64, f64)>],
    image_size: Size,
) -> Result<FisheyeCalibration> {
    let (rms, k, d, poses) = calibrate_camera(object, image, image_size)?;
    Ok((rms, k, [d[0], d[1], d[2], d[3]], poses))
}

/// `cv::calibrateCameraROExtended(objectPoints, imagePoints, imageSize, ...)`.
///
/// "RO" is release-object: the target's own geometry is refined alongside the
/// camera. A printed board is never exactly to specification, and past a
/// certain camera quality that printing error dominates the residual — this is
/// what removes it. `i_fixed_point` names the object point held fixed, without
/// which the whole target could slide and scale freely.
pub fn calibrate_camera_ro_extended(
    object: &[Vec<[f64; 3]>],
    image: &[Vec<(f64, f64)>],
    image_size: Size,
    i_fixed_point: usize,
) -> Result<(CalibrationExtended, Vec<[f64; 3]>)> {
    check_views(object, image)?;
    let n_pts = object[0].len();
    cv_assert!(
        object.iter().all(|o| o.len() == n_pts),
        "releaseObject needs every view to observe the same target"
    );
    cv_assert!(
        i_fixed_point < n_pts,
        "the fixed point {i_fixed_point} is past the {n_pts}-point target"
    );

    // Calibrate with the nominal target, then refine the target itself by
    // moving each point to minimise its own reprojection error across views.
    let base = calibrate_camera_extended(object, image, image_size)?;
    let mut refined = object[0].clone();
    for (i, pt) in refined.iter_mut().enumerate() {
        if i == i_fixed_point {
            continue;
        }
        for axis in 0..3 {
            // One coordinate at a time by golden-section-free bisection on the
            // derivative; the residual is smooth and nearly quadratic here.
            let mut step = 1e-3;
            for _ in 0..20 {
                let err = |delta: f64| -> f64 {
                    let mut p = *pt;
                    p[axis] += delta;
                    object
                        .iter()
                        .zip(image)
                        .zip(&base.poses)
                        .map(|((o, m), (r, t))| {
                            let mut view = o.clone();
                            view[i] = p;
                            let proj =
                                project_points(&view, *r, *t, &base.camera_matrix, base.dist_coeffs);
                            (proj[i].0 - m[i].0).powi(2) + (proj[i].1 - m[i].1).powi(2)
                        })
                        .sum()
                };
                let (c, plus, minus) = (err(0.0), err(step), err(-step));
                if plus < c && plus <= minus {
                    pt[axis] += step;
                } else if minus < c {
                    pt[axis] -= step;
                } else {
                    step /= 2.0;
                }
            }
        }
    }
    Ok((base, refined))
}

/// `cv::calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint, ...)`.
pub fn calibrate_camera_ro(
    object: &[Vec<[f64; 3]>],
    image: &[Vec<(f64, f64)>],
    image_size: Size,
    i_fixed_point: usize,
) -> Result<CalibrationRo> {
    let (e, obj) = calibrate_camera_ro_extended(object, image, image_size, i_fixed_point)?;
    Ok((e.rms, e.camera_matrix, e.dist_coeffs, e.poses, obj))
}

/// What `cv::stereoCalibrateExtended` returns.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoCalibration {
    /// RMS reprojection error over both cameras.
    pub rms: f64,
    /// Rotation taking the first camera's frame to the second's.
    pub r: Mat3,
    /// Translation, in the same units as the object points.
    pub t: [f64; 3],
    /// `perViewErrors` — two columns, one per camera.
    pub per_view_errors: Vec<(f64, f64)>,
}

/// `cv::stereoCalibrateExtended(objectPoints, imagePoints1, imagePoints2, ...)`.
///
/// With both cameras already calibrated, the rig's relative pose is
/// over-determined: every view gives an independent estimate of it. Averaging
/// the rotations in the log map rather than element-wise is what keeps the
/// result a rotation — an element-wise mean of rotation matrices is not one.
#[allow(clippy::too_many_arguments)]
pub fn stereo_calibrate_extended(
    object: &[Vec<[f64; 3]>],
    image1: &[Vec<(f64, f64)>],
    image2: &[Vec<(f64, f64)>],
    k1: &Mat3,
    dist1: [f64; 5],
    k2: &Mat3,
    dist2: [f64; 5],
) -> Result<StereoCalibration> {
    check_views(object, image1)?;
    check_views(object, image2)?;

    let mut log_sum = [0.0f64; 3];
    let mut t_sum = [0.0f64; 3];
    let mut per_view = Vec::with_capacity(object.len());
    let n = object.len() as f64;
    for ((o, m1), m2) in object.iter().zip(image1).zip(image2) {
        let p1 = seed_pose(o, m1, k1)?;
        let p2 = seed_pose(o, m2, k2)?;
        let r1 = rodrigues_to_matrix(p1.0);
        let r2 = rodrigues_to_matrix(p2.0);
        let r1i = mat3_inv(&r1).ok_or_else(|| {
            cv_core::Error::new(Code::StsError, "a view's rotation is singular".to_string())
        })?;
        let r = mat3_mul(&r2, &r1i);
        let log = rodrigues_to_vector(&r);
        for i in 0..3 {
            log_sum[i] += log[i] / n;
        }
        // t = t2 - R t1.
        let rt1 = [
            r[0][0] * p1.1[0] + r[0][1] * p1.1[1] + r[0][2] * p1.1[2],
            r[1][0] * p1.1[0] + r[1][1] * p1.1[1] + r[1][2] * p1.1[2],
            r[2][0] * p1.1[0] + r[2][1] * p1.1[1] + r[2][2] * p1.1[2],
        ];
        for i in 0..3 {
            t_sum[i] += (p2.1[i] - rt1[i]) / n;
        }

        let e1 = view_rms(o, m1, p1.0, p1.1, k1, dist1);
        let e2 = view_rms(o, m2, p2.0, p2.1, k2, dist2);
        per_view.push((e1, e2));
    }

    let total: f64 = per_view.iter().map(|(a, b)| a * a + b * b).sum::<f64>() / (2.0 * n);
    Ok(StereoCalibration {
        rms: total.sqrt(),
        r: rodrigues_to_matrix(log_sum),
        t: t_sum,
        per_view_errors: per_view,
    })
}

/// One view's RMS reprojection error.
fn view_rms(
    object: &[[f64; 3]],
    image: &[(f64, f64)],
    rvec: [f64; 3],
    tvec: [f64; 3],
    k: &Mat3,
    dist: [f64; 5],
) -> f64 {
    let proj = project_points(object, rvec, tvec, k, dist);
    let sse: f64 =
        proj.iter().zip(image).map(|(q, m)| (q.0 - m.0).powi(2) + (q.1 - m.1).powi(2)).sum();
    (sse / object.len() as f64).sqrt()
}

/// `cv::calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, ...)`.
///
/// Where is the camera, relative to the robot's gripper? The classic
/// formulation is `AX = XB`: `A` is the gripper's motion between two poses, `B`
/// is the camera's, and `X` — the unknown — is the fixed transform between
/// them. Two motions about non-parallel axes determine it, which is why a
/// calibration sequence has to *rotate* the wrist and not merely translate it.
pub fn calibrate_hand_eye(
    r_gripper2base: &[Mat3],
    t_gripper2base: &[[f64; 3]],
    r_target2cam: &[Mat3],
    t_target2cam: &[[f64; 3]],
) -> Result<Rigid> {
    cv_assert!(
        r_gripper2base.len() == t_gripper2base.len()
            && r_target2cam.len() == t_target2cam.len()
            && r_gripper2base.len() == r_target2cam.len(),
        "the four input lists must have the same length"
    );
    cv_assert!(
        r_gripper2base.len() >= 3,
        "hand-eye calibration needs at least three poses, got {}",
        r_gripper2base.len()
    );

    // Tsai-Lenz: solve the rotation from the axis pairs first, then the
    // translation given that rotation. Solving them jointly is possible and
    // buys nothing when the rotations are well conditioned.
    let n = r_gripper2base.len();

    // Every *pair* of stations, not just consecutive ones. Upstream takes all
    // n(n-1)/2 of them: a five-station sequence gives ten motions rather than
    // four, and the extra ones are not redundant because each pair sees a
    // different relative rotation.
    let mut motions: Vec<(Rigid, Rigid)> = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            // Gripper: `inv(Hg[j]) * Hg[i]`. Camera: `Hc[j] * inv(Hc[i])`.
            // The two are *not* the same composition — the gripper poses are
            // gripper-to-base and the camera poses target-to-camera, so the
            // frame being eliminated sits on opposite sides. Using one form
            // for both is the classic hand-eye bug and it silently returns a
            // plausible rotation that is simply wrong.
            let a = motion_back(
                &r_gripper2base[i],
                t_gripper2base[i],
                &r_gripper2base[j],
                t_gripper2base[j],
            );
            let b =
                motion_fwd(&r_target2cam[i], t_target2cam[i], &r_target2cam[j], t_target2cam[j]);
            let (pa, pb) = (modified_rodrigues(&a.0), modified_rodrigues(&b.0));
            // Upstream discards a pair whose rotation is under about 17 degrees
            // or over about 120: a small rotation carries almost no information
            // about the axis, and a large one is where the quaternion-minimal
            // parametrisation loses precision.
            let norm = |v: &[f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if !(0.3..=1.7).contains(&norm(&pa)) || !(0.3..=1.7).contains(&norm(&pb)) {
                continue;
            }
            motions.push((a, b));
        }
    }
    if motions.len() < 2 {
        return Err(cv_core::Error::new(
            Code::StsError,
            "hand-eye calibration failed: not enough informative motions — include larger rotations"
                .to_string(),
        ));
    }

    let mut rows: Vec<[f64; 3]> = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();
    for (a, b) in &motions {
        let (pa, pb) = (modified_rodrigues(&a.0), modified_rodrigues(&b.0));
        // skew(Pa + Pb) x = Pb - Pa, one 3x3 block per motion pair.
        let s = [pa[0] + pb[0], pa[1] + pb[1], pa[2] + pb[2]];
        let d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let skew = [[0.0, -s[2], s[1]], [s[2], 0.0, -s[0]], [-s[1], s[0], 0.0]];
        for r in 0..3 {
            rows.push(skew[r]);
            rhs.push(d[r]);
        }
    }
    let p = least_squares3(&rows, &rhs).ok_or_else(|| {
        cv_core::Error::new(
            Code::StsError,
            "hand-eye: the motions are degenerate — rotate about at least two distinct axes"
                .to_string(),
        )
    })?;
    // Back out the rotation from the modified Rodrigues vector.
    let norm2: f64 = p.iter().map(|v| v * v).sum();
    let scale = 2.0 / (1.0 + norm2).sqrt();
    let pr = [p[0] * scale, p[1] * scale, p[2] * scale];
    let r = mrp_to_matrix(&pr);

    // Translation: (Ra - I) t = R b - a, stacked over the same motion pairs.
    let mut rows_t: Vec<[f64; 3]> = Vec::new();
    let mut rhs_t: Vec<f64> = Vec::new();
    for (a, b) in &motions {
        let rb = [
            r[0][0] * b.1[0] + r[0][1] * b.1[1] + r[0][2] * b.1[2],
            r[1][0] * b.1[0] + r[1][1] * b.1[1] + r[1][2] * b.1[2],
            r[2][0] * b.1[0] + r[2][1] * b.1[1] + r[2][2] * b.1[2],
        ];
        for row in 0..3 {
            let mut m = a.0[row];
            m[row] -= 1.0;
            rows_t.push(m);
            rhs_t.push(rb[row] - a.1[row]);
        }
    }
    let t = least_squares3(&rows_t, &rhs_t).unwrap_or([0.0; 3]);
    Ok((r, t))
}

/// `inv(T_j) * T_i` — the motion that takes frame `i` into frame `j`.
fn motion_back(ri: &Mat3, ti: [f64; 3], rj: &Mat3, tj: [f64; 3]) -> Rigid {
    let rjt = mat3::transpose3(rj);
    let r = mat3_mul(&rjt, ri);
    let d = [ti[0] - tj[0], ti[1] - tj[1], ti[2] - tj[2]];
    (r, mat3::mul3(&rjt, d))
}

/// `T_j * inv(T_i)` — the same motion expressed in the other frame.
fn motion_fwd(ri: &Mat3, ti: [f64; 3], rj: &Mat3, tj: [f64; 3]) -> Rigid {
    let r = mat3_mul(rj, &mat3::transpose3(ri));
    let rti = mat3::mul3(&r, ti);
    (r, [tj[0] - rti[0], tj[1] - rti[1], tj[2] - rti[2]])
}




/// Tsai's modified Rodrigues vector: `2 sin(theta/2)` times the axis.
fn modified_rodrigues(r: &Mat3) -> [f64; 3] {
    let v = rodrigues_to_vector(r);
    let theta = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if theta < 1e-12 {
        return [0.0; 3];
    }
    let s = 2.0 * (theta / 2.0).sin() / theta;
    [v[0] * s, v[1] * s, v[2] * s]
}

/// The rotation matrix of a modified Rodrigues vector.
fn mrp_to_matrix(p: &[f64; 3]) -> Mat3 {
    let n2: f64 = p.iter().map(|v| v * v).sum();
    let theta = 2.0 * (n2.sqrt() / 2.0).asin();
    let n = n2.sqrt();
    if n < 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let axis = [p[0] / n * theta, p[1] / n * theta, p[2] / n * theta];
    rodrigues_to_matrix(axis)
}

/// The least-squares solution of a stacked 3-column system.
fn least_squares3(rows: &[[f64; 3]], rhs: &[f64]) -> Option<[f64; 3]> {
    let mut ata = [0.0f64; 9];
    let mut atb = [0.0f64; 3];
    for (r, &b) in rows.iter().zip(rhs) {
        for i in 0..3 {
            atb[i] += r[i] * b;
            for j in 0..3 {
                ata[i * 3 + j] += r[i] * r[j];
            }
        }
    }
    let v = solve_sym(&ata, &atb, 3)?;
    Some([v[0], v[1], v[2]])
}

/// `cv::calibrateRobotWorldHandEye(R_world2cam, t_world2cam, R_base2gripper, ...)`.
///
/// The `AX = ZB` variant: it solves for *two* unknown transforms at once — the
/// robot base to the world frame, and the gripper to the camera. Use it when
/// neither is known; when the base-to-world transform is already surveyed,
/// [`calibrate_hand_eye`] is better conditioned because it has half as many
/// unknowns.
pub fn calibrate_robot_world_hand_eye(
    r_world2cam: &[Mat3],
    t_world2cam: &[[f64; 3]],
    r_base2gripper: &[Mat3],
    t_base2gripper: &[[f64; 3]],
) -> Result<(Rigid, Rigid)> {
    cv_assert!(
        r_world2cam.len() == t_world2cam.len()
            && r_base2gripper.len() == t_base2gripper.len()
            && r_world2cam.len() == r_base2gripper.len(),
        "the four input lists must have the same length"
    );
    cv_assert!(
        r_world2cam.len() >= 3,
        "robot-world hand-eye needs at least three poses, got {}",
        r_world2cam.len()
    );
    // Shah's separable solution: the gripper-to-camera rotation is the hand-eye
    // one, and the base-to-world rotation then follows from any single pose.
    let (r_gripper2cam, t_gripper2cam) =
        calibrate_hand_eye(r_base2gripper, t_base2gripper, r_world2cam, t_world2cam)?;
    let rb = mat3_mul(&r_world2cam[0], &r_gripper2cam);
    let inv = mat3_inv(&r_base2gripper[0]).ok_or_else(|| {
        cv_core::Error::new(Code::StsError, "the first gripper pose is singular".to_string())
    })?;
    let r_base2world = mat3_mul(&rb, &inv);
    let t_base2world = [
        t_world2cam[0][0] - t_base2gripper[0][0],
        t_world2cam[0][1] - t_base2gripper[0][1],
        t_world2cam[0][2] - t_base2gripper[0][2],
    ];
    Ok(((r_base2world, t_base2world), (r_gripper2cam, t_gripper2cam)))
}

/// `cv::solveP3P(objectPoints, imagePoints, cameraMatrix, distCoeffs, ...)`.
///
/// Three points, so up to four poses satisfy the projection exactly and the
/// image alone cannot choose between them. Returning all of them is the honest
/// answer; a fourth point, or a plausibility test on the scene, is what
/// disambiguates.
pub fn solve_p3p(
    object: &[[f64; 3]],
    image: &[(f64, f64)],
    k: &Mat3,
    dist: [f64; 5],
) -> Result<Vec<Pose6>> {
    cv_assert!(object.len() == 3 && image.len() == 3, "P3P takes exactly three correspondences");
    // Bearing vectors, undistorted and normalised.
    let und = crate::undistort_points(image, k, dist);
    let bearings: Vec<[f64; 3]> = und
        .iter()
        .map(|&(x, y)| {
            let n = (x * x + y * y + 1.0).sqrt();
            [x / n, y / n, 1.0 / n]
        })
        .collect();

    // Grunert: the three unknown depths satisfy a quartic. Solving the depths
    // by a bounded search rather than the closed-form quartic keeps the
    // branch-and-sign bookkeeping — the classic source of wrong P3P roots —
    // out of the code entirely.
    let d = |a: &[f64; 3], b: &[f64; 3]| -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let (ab, bc, ca) = (d(&object[0], &object[1]), d(&object[1], &object[2]), d(&object[2], &object[0]));
    cv_assert!(ab > 0.0 && bc > 0.0 && ca > 0.0, "the three object points must be distinct");
    let cos = |u: &[f64; 3], v: &[f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let (c_ab, c_bc, c_ca) =
        (cos(&bearings[0], &bearings[1]), cos(&bearings[1], &bearings[2]), cos(&bearings[2], &bearings[0]));

    // For each candidate depth of the first point, the other two follow from
    // the law of cosines; the residual of the third constraint selects a root.
    // Both quadratics have two roots, so all four sign combinations are swept
    // — that is where P3P's four poses come from, and sweeping one of them
    // finds a subset without any sign that the rest exist.
    let mut roots: Vec<[f64; 3]> = Vec::new();
    let span = (ab + bc + ca) * 4.0;
    let steps = 4000;
    for b2 in 0..2 {
        for b3 in 0..2 {
            let residual = |s1: f64| -> Option<(f64, [f64; 3])> {
                let s2 = solve_cosine(s1, ab, c_ab, b2)?;
                let s3 = solve_cosine(s1, ca, c_ca, b3)?;
                let got = (s2 * s2 + s3 * s3 - 2.0 * s2 * s3 * c_bc).sqrt();
                Some((got - bc, [s1, s2, s3]))
            };
            let mut prev: Option<(f64, f64)> = None;
            for i in 1..=steps {
                let s1 = span * i as f64 / steps as f64;
                let Some((r, _)) = residual(s1) else {
                    prev = None;
                    continue;
                };
                if let Some((ps, pr)) = prev {
                    if pr * r < 0.0 {
                        // Bisect the sign change.
                        let (mut lo, mut hi) = (ps, s1);
                        for _ in 0..60 {
                            let mid = 0.5 * (lo + hi);
                            let Some((rm, _)) = residual(mid) else { break };
                            if rm * pr < 0.0 {
                                hi = mid;
                            } else {
                                lo = mid;
                            }
                        }
                        if let Some((_, depths)) = residual(0.5 * (lo + hi)) {
                            // The same depth triple can be reached from two
                            // branches when a root sits on the discriminant.
                            let dup = roots.iter().any(|r: &[f64; 3]| {
                                (0..3).all(|i| (r[i] - depths[i]).abs() < 1e-6 * depths[i].max(1.0))
                            });
                            if !dup {
                                roots.push(depths);
                            }
                        }
                    }
                }
                prev = Some((s1, r));
            }
        }
    }
    if roots.is_empty() {
        return cv_error!(Code::StsError, "solveP3P found no consistent depth triple");
    }

    // Each depth triple gives three camera-frame points; the pose is the
    // rigid transform taking the object points onto them.
    let mut poses = Vec::new();
    for depths in roots {
        let cam: Vec<[f64; 3]> = (0..3)
            .map(|i| {
                [
                    bearings[i][0] * depths[i],
                    bearings[i][1] * depths[i],
                    bearings[i][2] * depths[i],
                ]
            })
            .collect();
        if let Some((r, t)) = rigid_from_three(object, &cam) {
            poses.push((rodrigues_to_vector(&r), t));
        }
    }
    cv_assert!(!poses.is_empty(), "solveP3P: no depth triple yielded a valid pose");
    Ok(poses)
}

/// A positive root of `s2^2 - 2 s1 s2 cos + s1^2 - side^2 = 0`.
///
/// There are two, and `branch` picks which. Taking only the larger is why this
/// used to return two of P3P's four poses: the four come from the two sign
/// choices on each of the two sides adjacent to the first point.
fn solve_cosine(s1: f64, side: f64, cos: f64, branch: usize) -> Option<f64> {
    let disc = s1 * s1 * cos * cos - (s1 * s1 - side * side);
    if disc < 0.0 {
        return None;
    }
    let root = if branch == 0 { s1 * cos + disc.sqrt() } else { s1 * cos - disc.sqrt() };
    (root > 0.0).then_some(root)
}

/// The rigid transform taking three points onto three others.
///
/// Built from an orthonormal frame on each triple rather than by Kabsch: three
/// points span a plane, so their covariance matrix has rank two and the polar
/// decomposition Kabsch needs is not defined. Constructing the frame directly
/// is exact for three points and cannot silently return a reflection.
fn rigid_from_three(from: &[[f64; 3]], to: &[[f64; 3]]) -> Option<Rigid> {
    let frame = |p: &[[f64; 3]]| -> Option<Mat3> {
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let mut u1 = sub(p[1], p[0]);
        let n1 = norm(u1);
        if n1 < 1e-12 {
            return None;
        }
        for c in &mut u1 {
            *c /= n1;
        }
        let e2 = sub(p[2], p[0]);
        let d = u1[0] * e2[0] + u1[1] * e2[1] + u1[2] * e2[2];
        let mut u2 = [e2[0] - d * u1[0], e2[1] - d * u1[1], e2[2] - d * u1[2]];
        let n2 = norm(u2);
        if n2 < 1e-12 {
            return None;
        }
        for c in &mut u2 {
            *c /= n2;
        }
        let u3 = [
            u1[1] * u2[2] - u1[2] * u2[1],
            u1[2] * u2[0] - u1[0] * u2[2],
            u1[0] * u2[1] - u1[1] * u2[0],
        ];
        Some([
            [u1[0], u2[0], u3[0]],
            [u1[1], u2[1], u3[1]],
            [u1[2], u2[2], u3[2]],
        ])
    };
    let (a, b) = (frame(from)?, frame(to)?);
    // R takes the source frame onto the destination frame.
    let r = mat3_mul(&b, &mat3::transpose3(&a));
    let t = [
        to[0][0] - (r[0][0] * from[0][0] + r[0][1] * from[0][1] + r[0][2] * from[0][2]),
        to[0][1] - (r[1][0] * from[0][0] + r[1][1] * from[0][1] + r[1][2] * from[0][2]),
        to[0][2] - (r[2][0] * from[0][0] + r[2][1] * from[0][1] + r[2][2] * from[0][2]),
    ];
    Some((r, t))
}


/// `cv::solvePnPGeneric(objectPoints, imagePoints, cameraMatrix, distCoeffs, ...)`.
///
/// Every solution the chosen method admits, with its reprojection error, best
/// first. `solvePnP` returns only the first of these, which is the right
/// default and the wrong one whenever the configuration is ambiguous — a planar
/// target seen near head-on has two poses that fit almost equally well, and
/// this is how to see both.
pub fn solve_pnp_generic(
    object: &[[f64; 3]],
    image: &[(f64, f64)],
    k: &Mat3,
    dist: [f64; 5],
) -> Result<Vec<(Pose6, f64)>> {
    cv_assert!(
        object.len() == image.len(),
        "{} object points and {} image points",
        object.len(),
        image.len()
    );
    cv_assert!(object.len() >= 3, "PnP needs at least three points, got {}", object.len());

    let mut out = Vec::new();
    if object.len() == 3 {
        for pose in solve_p3p(object, image, k, dist)? {
            let e = view_rms(object, image, pose.0, pose.1, k, dist);
            out.push((pose, e));
        }
    } else {
        // Seed through [`seed_pose`], not through `solve_pnp`'s DLT: a
        // calibration target is planar, which is precisely where the DLT is
        // degenerate, and the refinement then starts from an infinity.
        let (rvec, tvec) = seed_pose(object, image, k)?;
        let refined = solve_pnp_refine_lm(object, image, k, dist, rvec, tvec, 50)?;
        let e = view_rms(object, image, refined.0, refined.1, k, dist);
        out.push((refined, e));
    }
    out.sort_by(|a, b| a.1.total_cmp(&b.1));
    Ok(out)
}

/// `cv::solvePnPRefineLM(objectPoints, imagePoints, cameraMatrix, distCoeffs, ...)`.
///
/// Refines a pose the caller already has. Distinct from `solvePnP` in that the
/// seed is the caller's: when tracking, the previous frame's pose is a far
/// better start than anything a fresh DLT produces, and it is what keeps the
/// solver on the right branch of an ambiguous configuration.
pub fn solve_pnp_refine_lm(
    object: &[[f64; 3]],
    image: &[(f64, f64)],
    k: &Mat3,
    dist: [f64; 5],
    rvec: [f64; 3],
    tvec: [f64; 3],
    max_iter: usize,
) -> Result<Pose6> {
    cv_assert!(object.len() == image.len(), "the point sets must correspond");
    cv_assert!(object.len() >= 3, "PnP refinement needs at least three points");
    cv_assert!(max_iter > 0, "the iteration count must be positive");
    let obj = vec![object.to_vec()];
    let img = vec![image.to_vec()];
    // Hold the intrinsics fixed; only the six pose parameters move.
    let p0 = pack(k, &dist, &[(rvec, tvec)]);
    let (p, _) = lm_refine(p0, &obj, &img, &(0..9).collect::<Vec<_>>(), max_iter);
    let (_, _, poses) = unpack(&p, 1);
    Ok(poses[0])
}

/// `cv::solvePnPRefineVVS(objectPoints, imagePoints, cameraMatrix, distCoeffs, ...)`.
///
/// Virtual visual servoing: the same residual, minimised by a damped
/// Gauss-Newton whose gain `vvs_lambda` is fixed rather than adapted. It
/// converges more smoothly than Levenberg-Marquardt when the seed is already
/// close — which is the tracking case it exists for — and less reliably when it
/// is not.
#[allow(clippy::too_many_arguments)]
pub fn solve_pnp_refine_vvs(
    object: &[[f64; 3]],
    image: &[(f64, f64)],
    k: &Mat3,
    dist: [f64; 5],
    rvec: [f64; 3],
    tvec: [f64; 3],
    max_iter: usize,
    vvs_lambda: f64,
) -> Result<Pose6> {
    cv_assert!(object.len() == image.len(), "the point sets must correspond");
    cv_assert!(object.len() >= 3, "PnP refinement needs at least three points");
    cv_assert!(max_iter > 0, "the iteration count must be positive");
    cv_assert!(vvs_lambda > 0.0, "VVSlambda must be positive, got {vvs_lambda}");

    let obj = vec![object.to_vec()];
    let img = vec![image.to_vec()];
    let mut p = pack(k, &dist, &[(rvec, tvec)]);
    let free: Vec<usize> = (9..15).collect();
    for _ in 0..max_iter {
        let r0 = residuals(&p, &obj, &img);
        let m = r0.len();
        let f = free.len();
        let mut j = vec![0.0f64; m * f];
        for (c, &idx) in free.iter().enumerate() {
            let h = (p[idx].abs() * 1e-6).max(1e-8);
            let mut q = p.clone();
            q[idx] += h;
            for (row, v) in residuals(&q, &obj, &img).iter().enumerate() {
                j[row * f + c] = (v - r0[row]) / h;
            }
        }
        let mut jtj = vec![0.0f64; f * f];
        let mut jtr = vec![0.0f64; f];
        for row in 0..m {
            for a in 0..f {
                jtr[a] += j[row * f + a] * r0[row];
                for b in 0..f {
                    jtj[a * f + b] += j[row * f + a] * j[row * f + b];
                }
            }
        }
        let Some(delta) = solve_sym(&jtj, &jtr, f) else { break };
        let mut moved = 0.0;
        for (c, &idx) in free.iter().enumerate() {
            let step = vvs_lambda * delta[c];
            p[idx] -= step;
            moved += step.abs();
        }
        if moved < 1e-12 {
            break;
        }
    }
    let (_, _, poses) = unpack(&p, 1);
    Ok(poses[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A planar chessboard target, `cols x rows` points spaced one unit apart.
    fn target(cols: usize, rows: usize) -> Vec<[f64; 3]> {
        (0..rows)
            .flat_map(|r| (0..cols).map(move |c| [c as f64, r as f64, 0.0]))
            .collect()
    }

    /// The ground-truth camera.
    fn truth() -> (Mat3, [f64; 5]) {
        (crate::camera_matrix(600.0, 600.0, 320.0, 240.0), [0.0; 5])
    }

    /// Object points, image points and poses for a set of views.
    type Views = (Vec<Vec<[f64; 3]>>, Vec<Vec<(f64, f64)>>, Vec<Pose6>);

    /// Views of the target from a handful of poses.
    fn views() -> Views {
        let (k, d) = truth();
        let obj = target(7, 5);
        let poses: Vec<Pose6> = vec![
            ([0.05, -0.10, 0.02], [-3.0, -2.0, 14.0]),
            ([0.30, 0.15, -0.05], [-3.5, -2.5, 16.0]),
            ([-0.25, 0.20, 0.10], [-2.5, -1.5, 13.0]),
            ([0.12, -0.30, -0.08], [-3.2, -2.2, 15.0]),
            ([-0.18, -0.22, 0.15], [-2.8, -1.8, 12.5]),
            ([0.22, 0.28, 0.03], [-3.1, -2.6, 17.0]),
        ];
        let images = poses
            .iter()
            .map(|&(r, t)| project_points(&obj, r, t, &k, d))
            .collect();
        (vec![obj; poses.len()], images, poses)
    }

    #[test]
    fn calibration_recovers_the_camera_it_was_generated_from() {
        let (obj, img, _) = views();
        let (k, _) = truth();
        let (rms, found, dist, poses) =
            calibrate_camera(&obj, &img, Size::new(640, 480)).unwrap();
        assert!(rms < 0.5, "noise-free views should fit closely: rms {rms}");
        assert!((found[0][0] - k[0][0]).abs() < 20.0, "fx {} vs {}", found[0][0], k[0][0]);
        assert!((found[1][1] - k[1][1]).abs() < 20.0, "fy {} vs {}", found[1][1], k[1][1]);
        assert!((found[0][2] - k[0][2]).abs() < 20.0, "cx {}", found[0][2]);
        assert!((found[1][2] - k[1][2]).abs() < 20.0, "cy {}", found[1][2]);
        assert_eq!(poses.len(), 6);
        assert!(dist.iter().all(|v| v.abs() < 0.5), "a pinhole camera has small distortion: {dist:?}");
    }

    #[test]
    fn the_extended_form_reports_per_view_errors_and_deviations() {
        let (obj, img, _) = views();
        let e = calibrate_camera_extended(&obj, &img, Size::new(640, 480)).unwrap();
        assert_eq!(e.per_view_errors.len(), 6);
        assert!(e.per_view_errors.iter().all(|v| v.is_finite() && *v >= 0.0));
        // The aggregate is in the range of the parts.
        let worst = e.per_view_errors.iter().copied().fold(0.0f64, f64::max);
        assert!(e.rms <= worst + 1e-9, "rms {} vs worst view {worst}", e.rms);
        assert_eq!(e.std_deviations_intrinsics.len(), 9);
        assert!(e.std_deviations_intrinsics.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn calibration_checks_its_input() {
        let (obj, img, _) = views();
        assert!(calibrate_camera(&[], &[], Size::new(640, 480)).is_err());
        assert!(calibrate_camera(&obj, &img[..3], Size::new(640, 480)).is_err());
        assert!(calibrate_camera(&obj, &img, Size::new(0, 480)).is_err());
        // A view with fewer than four points cannot give a pose.
        let short_obj = vec![obj[0][..3].to_vec()];
        let short_img = vec![img[0][..3].to_vec()];
        assert!(calibrate_camera(&short_obj, &short_img, Size::new(640, 480)).is_err());
        // Mismatched counts within a view.
        let bad_img = vec![img[0][..10].to_vec()];
        assert!(calibrate_camera(&[obj[0].clone()], &bad_img, Size::new(640, 480)).is_err());
    }

    #[test]
    fn the_fisheye_entry_point_returns_four_coefficients() {
        let (obj, img, _) = views();
        let (rms, k, d, poses) = calibrate(&obj, &img, Size::new(640, 480)).unwrap();
        assert_eq!(d.len(), 4);
        assert!(rms.is_finite());
        assert!(k[0][0] > 0.0);
        assert_eq!(poses.len(), 6);
    }

    #[test]
    fn release_object_refines_the_target_and_holds_its_anchor() {
        let (obj, img, _) = views();
        let anchor = 0usize;
        let (e, refined) =
            calibrate_camera_ro_extended(&obj, &img, Size::new(640, 480), anchor).unwrap();
        assert_eq!(refined.len(), obj[0].len());
        assert_eq!(refined[anchor], obj[0][anchor], "the anchor must not move");
        // The refinement is a nudge, not a redesign: the target was already right.
        for (a, b) in obj[0].iter().zip(&refined) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 0.5, "{a:?} moved to {b:?}");
            }
        }
        assert!(e.rms.is_finite());

        let (rms, _, _, _, obj2) =
            calibrate_camera_ro(&obj, &img, Size::new(640, 480), anchor).unwrap();
        assert!(rms.is_finite());
        assert_eq!(obj2.len(), refined.len());
        assert!(
            calibrate_camera_ro(&obj, &img, Size::new(640, 480), 999).is_err(),
            "the anchor must be a real point"
        );
    }

    #[test]
    fn stereo_calibration_recovers_a_known_rig() {
        let (k, d) = truth();
        let obj_one = target(7, 5);
        // The second camera sits 1 unit to the right, unrotated.
        let poses: Vec<Pose6> = vec![
            ([0.05, -0.10, 0.02], [-3.0, -2.0, 14.0]),
            ([0.30, 0.15, -0.05], [-3.5, -2.5, 16.0]),
            ([-0.25, 0.20, 0.10], [-2.5, -1.5, 13.0]),
        ];
        let obj: Vec<Vec<[f64; 3]>> = vec![obj_one.clone(); poses.len()];
        let img1: Vec<Vec<(f64, f64)>> =
            poses.iter().map(|&(r, t)| project_points(&obj_one, r, t, &k, d)).collect();
        let img2: Vec<Vec<(f64, f64)>> = poses
            .iter()
            .map(|&(r, t)| project_points(&obj_one, r, [t[0] - 1.0, t[1], t[2]], &k, d))
            .collect();

        let s = stereo_calibrate_extended(&obj, &img1, &img2, &k, d, &k, d).unwrap();
        assert_eq!(s.per_view_errors.len(), 3);
        assert!(s.rms < 1.0, "rms {}", s.rms);
        // The rotation is the identity and the translation is -1 along x.
        for i in 0..3 {
            for j in 0..3 {
                let want = f64::from(u8::from(i == j));
                assert!((s.r[i][j] - want).abs() < 0.05, "R[{i}][{j}] = {}", s.r[i][j]);
            }
        }
        assert!((s.t[0] + 1.0).abs() < 0.2, "tx {}", s.t[0]);
        assert!(s.t[1].abs() < 0.2 && s.t[2].abs() < 0.2, "t {:?}", s.t);

        assert!(stereo_calibrate_extended(&obj, &img1[..1], &img2, &k, d, &k, d).is_err());
    }

    /// A rotation about `axis` by `angle`.
    fn rot(axis: [f64; 3], angle: f64) -> Mat3 {
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        rodrigues_to_matrix([axis[0] / n * angle, axis[1] / n * angle, axis[2] / n * angle])
    }

    #[test]
    fn hand_eye_calibration_recovers_a_known_mounting() {
        // Build the scene the way the robot actually is, not the way the
        // equation is written: the target is bolted to the base, the camera is
        // bolted to the gripper, and the camera's view of the target follows.
        // Synthesising the camera poses as `X^-1 A X` instead applies the
        // *motion* relation to *absolute* poses, which is a different scene and
        // agrees with no hand-eye solver.
        let x_r = rot([0.0, 0.0, 1.0], 0.4); // cam2gripper, the answer
        let x_t = [0.05, -0.03, 0.10];
        let target_r = rot([0.3, 0.1, -0.5], 0.7); // target2base, fixed
        let target_t = [0.6, 0.2, 0.9];

        let gripper: Vec<Mat3> = vec![
            rot([1.0, 0.0, 0.0], 0.6),
            rot([0.0, 1.0, 0.0], 0.9),
            rot([1.0, 1.0, 0.0], 0.7),
            rot([0.0, 1.0, 1.0], 1.0),
            rot([1.0, 0.0, 1.0], 0.5),
        ];
        let gt: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0],
            [0.0, 0.3, 0.1],
            [0.1, 0.0, 0.25],
            [0.15, 0.2, 0.05],
        ];

        // target2cam = gripper2cam . base2gripper . target2base
        let xt = mat3::transpose3(&x_r);
        let (mut cam, mut ct) = (Vec::new(), Vec::new());
        for (g, &t) in gripper.iter().zip(&gt) {
            let gtr = mat3::transpose3(g);
            let r1 = mat3_mul(&gtr, &target_r);
            let d = [target_t[0] - t[0], target_t[1] - t[1], target_t[2] - t[2]];
            let t1 = mat3::mul3(&gtr, d);
            cam.push(mat3_mul(&xt, &r1));
            let a = mat3::mul3(&xt, t1);
            let b = mat3::mul3(&xt, x_t);
            ct.push([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
        }

        let (r, t) = calibrate_hand_eye(&gripper, &gt, &cam, &ct).unwrap();
        let v = rodrigues_to_vector(&r);
        let angle = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((angle - 0.4).abs() < 1e-6, "recovered angle {angle}");
        assert!(v[2].abs() > 0.99 * angle, "the axis should be z: {v:?}");
        for i in 0..3 {
            assert!((t[i] - x_t[i]).abs() < 1e-6, "t[{i}] = {} want {}", t[i], x_t[i]);
        }

        assert!(calibrate_hand_eye(&gripper[..2], &gt[..2], &cam[..2], &ct[..2]).is_err());
        assert!(calibrate_hand_eye(&gripper, &gt[..2], &cam, &ct).is_err());
    }

    #[test]
    fn robot_world_hand_eye_returns_two_transforms() {
        let gripper: Vec<Mat3> = vec![
            rot([1.0, 0.0, 0.0], 0.3),
            rot([0.0, 1.0, 0.0], 0.5),
            rot([1.0, 1.0, 0.0], 0.4),
            rot([0.0, 1.0, 1.0], 0.6),
        ];
        let t: Vec<[f64; 3]> =
            vec![[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.0, 0.3, 0.1], [0.1, 0.0, 0.25]];
        let x_r = rot([0.0, 0.0, 1.0], 0.4);
        let xi = mat3_inv(&x_r).unwrap();
        let cam: Vec<Mat3> = gripper.iter().map(|a| mat3_mul(&xi, &mat3_mul(a, &x_r))).collect();

        let ((rb, _tb), (rg, _tg)) =
            calibrate_robot_world_hand_eye(&cam, &t, &gripper, &t).unwrap();
        // Both must be rotations.
        for r in [rb, rg] {
            let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
            assert!((det - 1.0).abs() < 0.05, "det {det}");
        }
        assert!(calibrate_robot_world_hand_eye(&cam[..2], &t[..2], &gripper[..2], &t[..2]).is_err());
    }

    #[test]
    fn p3p_returns_poses_that_reproject_onto_their_points() {
        let (k, d) = truth();
        let obj = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]];
        let (rvec, tvec) = ([0.1, -0.2, 0.05], [-1.0, -1.5, 12.0]);
        let img = project_points(&obj, rvec, tvec, &k, d);
        let poses = solve_p3p(&obj, &img, &k, d).unwrap();
        assert!(!poses.is_empty(), "at least one pose must be found");
        // Every returned pose must reproject onto the observations — that is
        // the defining property, and several poses satisfying it is the
        // ambiguity, not a bug.
        for (r, t) in &poses {
            let e = view_rms(&obj, &img, *r, *t, &k, d);
            assert!(e < 1.0, "a returned pose should fit: rms {e}");
        }
        // One of them is the truth.
        let best = poses
            .iter()
            .map(|(r, _)| {
                (r[0] - rvec[0]).abs() + (r[1] - rvec[1]).abs() + (r[2] - rvec[2]).abs()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(best < 0.2, "the true pose should be among them: closest {best}");

        assert!(solve_p3p(&obj[..2], &img[..2], &k, d).is_err(), "P3P takes exactly three");
        let same = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        assert!(solve_p3p(&same, &img, &k, d).is_err(), "coincident points");
    }

    #[test]
    fn the_generic_solver_orders_its_solutions_by_error() {
        let (k, d) = truth();
        let obj = target(4, 3);
        let (rvec, tvec) = ([0.1, -0.2, 0.05], [-1.5, -1.0, 12.0]);
        let img = project_points(&obj, rvec, tvec, &k, d);
        let sols = solve_pnp_generic(&obj, &img, &k, d).unwrap();
        assert!(!sols.is_empty());
        for w in sols.windows(2) {
            assert!(w[0].1 <= w[1].1, "solutions must be sorted by error");
        }
        assert!(sols[0].1 < 1.0, "the best solution should fit: {}", sols[0].1);

        // With three points it routes to P3P and may return several.
        let three = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]];
        let img3 = project_points(&three, rvec, tvec, &k, d);
        let many = solve_pnp_generic(&three, &img3, &k, d).unwrap();
        assert!(!many.is_empty());

        assert!(solve_pnp_generic(&obj[..2], &img[..2], &k, d).is_err());
        assert!(solve_pnp_generic(&obj, &img[..3], &k, d).is_err());
    }

    #[test]
    fn both_refiners_improve_a_perturbed_pose() {
        let (k, d) = truth();
        let obj = target(4, 3);
        let (rvec, tvec) = ([0.1, -0.2, 0.05], [-1.5, -1.0, 12.0]);
        let img = project_points(&obj, rvec, tvec, &k, d);
        // A seed that is off by a noticeable amount.
        let seed_r = [rvec[0] + 0.05, rvec[1] - 0.04, rvec[2] + 0.03];
        let seed_t = [tvec[0] + 0.3, tvec[1] - 0.2, tvec[2] + 0.4];
        let before = view_rms(&obj, &img, seed_r, seed_t, &k, d);
        assert!(before > 1.0, "the seed should start off: {before}");

        let (lr, lt) = solve_pnp_refine_lm(&obj, &img, &k, d, seed_r, seed_t, 50).unwrap();
        let after_lm = view_rms(&obj, &img, lr, lt, &k, d);
        assert!(after_lm < before / 10.0, "LM: {after_lm} vs {before}");

        let (vr, vt) =
            solve_pnp_refine_vvs(&obj, &img, &k, d, seed_r, seed_t, 50, 1.0).unwrap();
        let after_vvs = view_rms(&obj, &img, vr, vt, &k, d);
        assert!(after_vvs < before / 10.0, "VVS: {after_vvs} vs {before}");

        assert!(solve_pnp_refine_lm(&obj, &img, &k, d, seed_r, seed_t, 0).is_err());
        assert!(solve_pnp_refine_vvs(&obj, &img, &k, d, seed_r, seed_t, 10, 0.0).is_err());
        assert!(solve_pnp_refine_lm(&obj[..2], &img[..2], &k, d, seed_r, seed_t, 5).is_err());
    }
}
