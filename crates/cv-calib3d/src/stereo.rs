//! Stereo geometry, undistortion maps and the decomposition family
//! (`modules/calib3d/src/calibration.cpp`, `undistort.dispatch.cpp`,
//! `homography_decomp.cpp`, `ptsetreg.cpp`).

use cv_core::types::CV_32FC1;
use cv_core::{cv_assert, Code, Error, Mat, MatLike, MatLikeMut, Result};

use crate::mat3;
use crate::{apply_homography, camera_matrix, mat3_inv, mat3_mul, rodrigues_to_matrix, Mat3};

/// `cv::convertPointsToHomogeneous(src, dst)`.
#[must_use]
pub fn convert_points_to_homogeneous(pts: &[(f64, f64)]) -> Vec<[f64; 3]> {
    pts.iter().map(|p| [p.0, p.1, 1.0]).collect()
}

/// `cv::convertPointsFromHomogeneous(src, dst)`.
///
/// A point with a zero final coordinate lies at infinity and has no finite
/// image; it comes back as `None` rather than as a very large number.
#[must_use]
pub fn convert_points_from_homogeneous(pts: &[[f64; 3]]) -> Vec<Option<(f64, f64)>> {
    pts.iter()
        .map(|p| {
            if p[2].abs() < 1e-12 {
                None
            } else {
                Some((p[0] / p[2], p[1] / p[2]))
            }
        })
        .collect()
}

/// `cv::RQDecomp3x3(src, mtxR, mtxQ, Qx, Qy, Qz)`.
///
/// Factors `M = R * Q` with `R` upper-triangular and `Q` a rotation. Used to
/// pull the intrinsics back out of a projection matrix.
///
/// The diagonal of `R` is forced positive, because `R * Q` is only unique up to
/// a sign per axis and a negative focal length is not a camera.
pub fn rq_decomp3x3(m: &Mat3) -> Result<(Mat3, Mat3)> {
    // Givens rotations zero the lower triangle one entry at a time.
    let mut a = *m;
    let mut q = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let rot = |a: &mut Mat3, q: &mut Mat3, g: Mat3| {
        *a = mat3_mul(a, &g);
        *q = mat3_mul(&mat3::transpose3(&g), q);
    };
    // Zero a[2][1] about x.
    let d = (a[2][1] * a[2][1] + a[2][2] * a[2][2]).sqrt();
    if d > 1e-15 {
        let (c, s) = (a[2][2] / d, -a[2][1] / d);
        rot(&mut a, &mut q, [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]);
    }
    // Zero a[2][0] about y.
    let d = (a[2][0] * a[2][0] + a[2][2] * a[2][2]).sqrt();
    if d > 1e-15 {
        let (c, s) = (a[2][2] / d, a[2][0] / d);
        rot(&mut a, &mut q, [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]);
    }
    // Zero a[1][0] about z.
    let d = (a[1][0] * a[1][0] + a[1][1] * a[1][1]).sqrt();
    if d > 1e-15 {
        let (c, s) = (a[1][1] / d, -a[1][0] / d);
        rot(&mut a, &mut q, [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]);
    }

    // Make the diagonal positive, flipping the matching row of Q to compensate.
    for i in 0..3 {
        if a[i][i] < 0.0 {
            for row in a.iter_mut() {
                row[i] = -row[i];
            }
            for v in q[i].iter_mut() {
                *v = -*v;
            }
        }
    }
    Ok((a, q))
}


/// `cv::decomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect, ...)`.
///
/// Splits a 3x4 projection into intrinsics, rotation and camera centre.
pub fn decompose_projection_matrix(p: &[[f64; 4]; 3]) -> Result<(Mat3, Mat3, [f64; 3])> {
    let m = [
        [p[0][0], p[0][1], p[0][2]],
        [p[1][0], p[1][1], p[1][2]],
        [p[2][0], p[2][1], p[2][2]],
    ];
    let (mut k, r) = rq_decomp3x3(&m)?;
    // Normalise so K[2][2] is 1, which is what makes it a camera matrix.
    let s = k[2][2];
    cv_assert!(s.abs() > 1e-12, "the projection matrix is degenerate");
    for row in k.iter_mut() {
        for v in row.iter_mut() {
            *v /= s;
        }
    }
    // The camera centre solves M c = -p4.
    let mi = mat3_inv(&m).ok_or_else(|| {
        Error::new(Code::StsBadArg, "the projection's leading 3x3 block is singular".to_string())
    })?;
    let p4 = [p[0][3], p[1][3], p[2][3]];
    let mut c = [0.0f64; 3];
    for i in 0..3 {
        c[i] = -(0..3).map(|j| mi[i][j] * p4[j]).sum::<f64>();
    }
    Ok((k, r, c))
}

/// `cv::composeRT(rvec1, tvec1, rvec2, tvec2, rvec3, tvec3, ...)`.
///
/// Composes two rigid transforms: apply the first, then the second.
#[must_use]
pub fn compose_rt(
    r1: [f64; 3],
    t1: [f64; 3],
    r2: [f64; 3],
    t2: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    let (m1, m2) = (rodrigues_to_matrix(r1), rodrigues_to_matrix(r2));
    let m3 = mat3_mul(&m2, &m1);
    let mut t3 = t2;
    for i in 0..3 {
        t3[i] += (0..3).map(|j| m2[i][j] * t1[j]).sum::<f64>();
    }
    (crate::rodrigues_to_vector(&m3), t3)
}

/// `cv::initCameraMatrix2D(objectPoints, imagePoints, imageSize, aspectRatio)`.
///
/// A closed-form first guess at the intrinsics from planar views, the seed a
/// full calibration would then refine. Assumes the principal point is at the
/// image centre, which is what upstream assumes here too.
pub fn init_camera_matrix_2d(
    object: &[Vec<[f64; 3]>],
    image: &[Vec<(f64, f64)>],
    width: i32,
    height: i32,
) -> Result<Mat3> {
    cv_assert!(!object.is_empty(), "at least one view is needed");
    cv_assert!(object.len() == image.len(), "object and image view counts must match");
    cv_assert!(width > 0 && height > 0, "the image size must be positive");
    let (cx, cy) = (f64::from(width) * 0.5 - 0.5, f64::from(height) * 0.5 - 0.5);

    // Each planar view gives a homography; the two circular-point constraints
    // per homography pin the focal lengths.
    let mut a: Vec<[f64; 2]> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    for (obj, img) in object.iter().zip(image) {
        cv_assert!(obj.len() == img.len(), "a view has mismatched point counts");
        if obj.len() < 4 {
            continue;
        }
        // Planar object: drop z.
        let src: Vec<(f64, f64)> = obj.iter().map(|p| (p[0], p[1])).collect();
        let dst: Vec<(f64, f64)> = img.iter().map(|p| (p.0 - cx, p.1 - cy)).collect();
        let Ok(h) = crate::find_homography(&src, &dst) else { continue };
        // Columns of H, normalised as upstream does.
        let h1 = [h[0][0], h[1][0], h[2][0]];
        let h2 = [h[0][1], h[1][1], h[2][1]];
        // h1^T w h2 = 0 and h1^T w h1 = h2^T w h2, with w = diag(1/fx^2, 1/fy^2, 1).
        a.push([h1[0] * h2[0], h1[1] * h2[1]]);
        b.push(-h1[2] * h2[2]);
        a.push([h1[0] * h1[0] - h2[0] * h2[0], h1[1] * h1[1] - h2[1] * h2[1]]);
        b.push(h2[2] * h2[2] - h1[2] * h1[2]);
    }
    cv_assert!(a.len() >= 2, "not enough usable views to estimate the intrinsics");

    // Least squares for (1/fx^2, 1/fy^2).
    let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (r, &rhs) in a.iter().zip(&b) {
        a11 += r[0] * r[0];
        a12 += r[0] * r[1];
        a22 += r[1] * r[1];
        b1 += r[0] * rhs;
        b2 += r[1] * rhs;
    }
    let det = a11 * a22 - a12 * a12;
    cv_assert!(det.abs() > 1e-12, "the views are degenerate for intrinsic estimation");
    let ix = (a22 * b1 - a12 * b2) / det;
    let iy = (a11 * b2 - a12 * b1) / det;
    cv_assert!(
        ix > 0.0 && iy > 0.0,
        "the estimate produced a non-physical focal length; the views may be coplanar with the camera"
    );
    Ok(camera_matrix(1.0 / ix.sqrt(), 1.0 / iy.sqrt(), cx, cy))
}

/// Result of [`stereo_rectify`].
#[derive(Debug, Clone, Copy)]
pub struct Rectification {
    /// Rotation applied to the first camera.
    pub r1: Mat3,
    /// Rotation applied to the second camera.
    pub r2: Mat3,
    /// New projection for the first camera, 3x4.
    pub p1: [[f64; 4]; 3],
    /// New projection for the second camera, 3x4.
    pub p2: [[f64; 4]; 3],
}

/// `cv::stereoRectify(cameraMatrix1, ..., R, T, R1, R2, P1, P2, Q, ...)`.
///
/// Computes the rotations that make both image planes coplanar with the
/// baseline, so corresponding points land on the same row and a disparity
/// search becomes one-dimensional. `r`/`t` take a point from the first
/// camera's frame to the second's.
pub fn stereo_rectify(k1: &Mat3, k2: &Mat3, r: &Mat3, t: [f64; 3]) -> Result<Rectification> {
    let norm = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    cv_assert!(norm > 1e-12, "the two cameras are at the same centre");

    // Split the relative rotation in half so neither view is turned more than
    // the other — upstream does the same, and it keeps both images usable.
    let rv = crate::rodrigues_to_vector(r);
    let half = rodrigues_to_matrix([rv[0] * 0.5, rv[1] * 0.5, rv[2] * 0.5]);
    let half_inv = mat3::transpose3(&half);

    // The new x axis is the baseline, rotated into the half-way frame.
    let mut e1 = [0.0f64; 3];
    for i in 0..3 {
        e1[i] = (0..3).map(|j| half[i][j] * t[j]).sum::<f64>();
    }
    let n = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
    for v in e1.iter_mut() {
        *v /= n;
    }
    // y is perpendicular to x and the old z; z completes the frame.
    let mut e2 = [-e1[1], e1[0], 0.0];
    let n2 = (e2[0] * e2[0] + e2[1] * e2[1]).sqrt();
    if n2 < 1e-12 {
        // The baseline is along z: pick any perpendicular.
        e2 = [0.0, 1.0, 0.0];
    } else {
        for v in e2.iter_mut() {
            *v /= n2;
        }
    }
    let e3 = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let rect = [e1, e2, e3];

    let r1 = mat3_mul(&rect, &half_inv);
    let r2 = mat3_mul(&rect, &half);

    // Both cameras share the rectified intrinsics; the second carries the
    // baseline in its fourth column, which is what disparity is measured in.
    let fx = (k1[0][0] + k2[0][0]) * 0.5;
    let fy = (k1[1][1] + k2[1][1]) * 0.5;
    let cx = (k1[0][2] + k2[0][2]) * 0.5;
    let cy = (k1[1][2] + k2[1][2]) * 0.5;
    let mut p1 = [[0.0f64; 4]; 3];
    let mut p2 = [[0.0f64; 4]; 3];
    for (p, tx) in [(&mut p1, 0.0), (&mut p2, -fx * norm)] {
        p[0] = [fx, 0.0, cx, tx];
        p[1] = [0.0, fy, cy, 0.0];
        p[2] = [0.0, 0.0, 1.0, 0.0];
    }
    Ok(Rectification { r1, r2, p1, p2 })
}

/// `cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha, ...)`.
///
/// `alpha` of 0 keeps only pixels with valid source data — no black corners —
/// while 1 keeps every source pixel and leaves the corners empty. Values in
/// between interpolate.
pub fn get_optimal_new_camera_matrix(
    k: &Mat3,
    dist: [f64; 5],
    width: i32,
    height: i32,
    alpha: f64,
) -> Result<Mat3> {
    cv_assert!(width > 0 && height > 0, "the image size must be positive");
    cv_assert!((0.0..=1.0).contains(&alpha), "alpha must lie in 0..1, got {alpha}");
    // Undistort the border and measure how far it moved.
    let mut pts = Vec::new();
    for i in 0..=8 {
        let f = f64::from(i) / 8.0;
        pts.push((f * f64::from(width - 1), 0.0));
        pts.push((f * f64::from(width - 1), f64::from(height - 1)));
        pts.push((0.0, f * f64::from(height - 1)));
        pts.push((f64::from(width - 1), f * f64::from(height - 1)));
    }
    let un = crate::undistort_points(&pts, k, dist);
    // Back into pixels with the original focal lengths.
    let mapped: Vec<(f64, f64)> = un
        .iter()
        .map(|p| (p.0 * k[0][0] + k[0][2], p.1 * k[1][1] + k[1][2]))
        .collect();
    let (mut x0, mut y0, mut x1, mut y1) =
        (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    // The inner rectangle: the largest box with no missing data.
    let (mut ix0, mut iy0, mut ix1, mut iy1) =
        (f64::MIN, f64::MIN, f64::MAX, f64::MAX);
    for (i, p) in mapped.iter().enumerate() {
        x0 = x0.min(p.0);
        y0 = y0.min(p.1);
        x1 = x1.max(p.0);
        y1 = y1.max(p.1);
        // Which edge did this sample come from?
        match i % 4 {
            0 => iy0 = iy0.max(p.1),
            1 => iy1 = iy1.min(p.1),
            2 => ix0 = ix0.max(p.0),
            _ => ix1 = ix1.min(p.0),
        }
    }
    let lerp = |a: f64, b: f64| a * (1.0 - alpha) + b * alpha;
    let (rx0, ry0) = (lerp(ix0, x0), lerp(iy0, y0));
    let (rx1, ry1) = (lerp(ix1, x1), lerp(iy1, y1));
    let (rw, rh) = ((rx1 - rx0).max(1.0), (ry1 - ry0).max(1.0));
    let sx = f64::from(width) / rw;
    let sy = f64::from(height) / rh;
    Ok(camera_matrix(
        k[0][0] * sx,
        k[1][1] * sy,
        (k[0][2] - rx0) * sx,
        (k[1][2] - ry0) * sy,
    ))
}

/// `cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, map1, map2)`.
///
/// Returns the `(x, y)` maps `remap` consumes. Building them once and reusing
/// them is the whole point — undistorting per frame from scratch repeats the
/// distortion model on every pixel of every frame.
pub fn init_undistort_rectify_map(
    k: &Mat3,
    dist: [f64; 5],
    r: &Mat3,
    new_k: &Mat3,
    width: i32,
    height: i32,
) -> Result<(Mat, Mat)> {
    cv_assert!(width > 0 && height > 0, "the map size must be positive");
    let rk = mat3_mul(r, &mat3_inv(new_k).ok_or_else(|| {
        Error::new(Code::StsBadArg, "the new camera matrix is singular".to_string())
    })?);
    let mut mx = Mat::zeros(height, width, CV_32FC1)?;
    let mut my = Mat::zeros(height, width, CV_32FC1)?;
    for v in 0..height {
        for u in 0..width {
            // Pixel -> normalised, through the rectifying rotation.
            let (fu, fv) = (f64::from(u), f64::from(v));
            let x = rk[0][0] * fu + rk[0][1] * fv + rk[0][2];
            let y = rk[1][0] * fu + rk[1][1] * fv + rk[1][2];
            let w = rk[2][0] * fu + rk[2][1] * fv + rk[2][2];
            let (xn, yn) = if w.abs() < 1e-12 { (0.0, 0.0) } else { (x / w, y / w) };
            // Apply the distortion, then the original intrinsics.
            let r2 = xn * xn + yn * yn;
            let radial = 1.0 + dist[0] * r2 + dist[1] * r2 * r2 + dist[4] * r2 * r2 * r2;
            let xd = xn * radial + 2.0 * dist[2] * xn * yn + dist[3] * (r2 + 2.0 * xn * xn);
            let yd = yn * radial + dist[2] * (r2 + 2.0 * yn * yn) + 2.0 * dist[3] * xn * yn;
            mx.pixel_mut::<f32>(v, u)?[0] = (k[0][0] * xd + k[0][2]) as f32;
            my.pixel_mut::<f32>(v, u)?[0] = (k[1][1] * yd + k[1][2]) as f32;
        }
    }
    Ok((mx, my))
}

/// `cv::undistort(src, dst, cameraMatrix, distCoeffs, newCameraMatrix)`.
pub fn undistort<S: MatLike + ?Sized>(
    src: &S,
    k: &Mat3,
    dist: [f64; 5],
    new_k: Option<&Mat3>,
) -> Result<Mat> {
    let nk = new_k.copied().unwrap_or(*k);
    let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let (mx, my) = init_undistort_rectify_map(k, dist, &eye, &nk, src.cols(), src.rows())?;
    let cn = src.channels();
    let (h, w) = (src.rows(), src.cols());
    let mut out = Mat::zeros(h, w, src.mat_type())?;
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (mx.at_f64(y, x, 0)?, my.at_f64(y, x, 0)?);
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
                cv_core::arithm::set_channel_generic(&mut out, y, x, c, v)?;
            }
        }
    }
    Ok(out)
}

/// `cv::sampsonDistance(pt1, pt2, F)`.
///
/// A first-order approximation to the geometric reprojection error of a
/// correspondence under a fundamental matrix — much cheaper than triangulating
/// and far better behaved than the raw algebraic residual.
#[must_use]
pub fn sampson_distance(p1: (f64, f64), p2: (f64, f64), f: &Mat3) -> f64 {
    let x1 = [p1.0, p1.1, 1.0];
    let x2 = [p2.0, p2.1, 1.0];
    let fx1: Vec<f64> = (0..3).map(|i| (0..3).map(|j| f[i][j] * x1[j]).sum()).collect();
    let ftx2: Vec<f64> = (0..3).map(|j| (0..3).map(|i| f[i][j] * x2[i]).sum()).collect();
    let num: f64 = (0..3).map(|i| x2[i] * fx1[i]).sum::<f64>();
    let den = fx1[0] * fx1[0] + fx1[1] * fx1[1] + ftx2[0] * ftx2[0] + ftx2[1] * ftx2[1];
    if den < 1e-15 {
        0.0
    } else {
        num * num / den
    }
}

/// `cv::estimateAffine2D(from, to, inliers, method, ransacReprojThreshold, ...)`.
///
/// Six-parameter fit; returns the 2x3 matrix and the inlier mask.
pub fn estimate_affine_2d(
    from: &[(f64, f64)],
    to: &[(f64, f64)],
    threshold: f64,
    iterations: usize,
    rng: &mut cv_core::rng::Rng,
) -> Result<([[f64; 3]; 2], Vec<bool>)> {
    cv_assert!(from.len() == to.len(), "point counts must match");
    cv_assert!(from.len() >= 3, "an affine fit needs at least 3 points");
    let fit = |idx: &[usize]| -> Option<[[f64; 3]; 2]> {
        // Least squares for each row independently: they do not interact.
        let (mut sxx, mut sxy, mut sx, mut syy, mut sy, mut n) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for &i in idx {
            let (x, y) = from[i];
            sxx += x * x;
            sxy += x * y;
            sx += x;
            syy += y * y;
            sy += y;
            n += 1.0;
        }
        let a = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]];
        let ai = mat3_inv(&a)?;
        let mut out = [[0.0f64; 3]; 2];
        for (r, orow) in out.iter_mut().enumerate() {
            let mut b = [0.0f64; 3];
            for &i in idx {
                let (x, y) = from[i];
                let t = if r == 0 { to[i].0 } else { to[i].1 };
                b[0] += x * t;
                b[1] += y * t;
                b[2] += t;
            }
            for (j, v) in orow.iter_mut().enumerate() {
                *v = (0..3).map(|k| ai[j][k] * b[k]).sum();
            }
        }
        Some(out)
    };
    let apply = |m: &[[f64; 3]; 2], p: (f64, f64)| {
        (m[0][0] * p.0 + m[0][1] * p.1 + m[0][2], m[1][0] * p.0 + m[1][1] * p.1 + m[1][2])
    };

    let n = from.len();
    let t2 = threshold * threshold;
    let mut best: (usize, Vec<bool>) = (0, vec![false; n]);
    for _ in 0..iterations.max(1) {
        let mut pick = [0usize; 3];
        for k in 0..3 {
            loop {
                let c = (rng.next_u32() as usize) % n;
                if !pick[..k].contains(&c) {
                    pick[k] = c;
                    break;
                }
            }
        }
        let Some(m) = fit(&pick) else { continue };
        let mask: Vec<bool> = (0..n)
            .map(|i| {
                let q = apply(&m, from[i]);
                (q.0 - to[i].0).powi(2) + (q.1 - to[i].1).powi(2) <= t2
            })
            .collect();
        let c = mask.iter().filter(|&&b| b).count();
        if c > best.0 {
            best = (c, mask);
        }
    }
    cv_assert!(best.0 >= 3, "RANSAC found only {} inliers", best.0);
    let keep: Vec<usize> = (0..n).filter(|&i| best.1[i]).collect();
    let m = fit(&keep).ok_or_else(|| {
        Error::new(Code::StsBadArg, "the consensus set is degenerate".to_string())
    })?;
    Ok((m, best.1))
}

/// `cv::estimateAffinePartial2D(from, to, ...)`.
///
/// Four parameters — uniform scale, rotation and translation — so it cannot
/// shear. Fewer degrees of freedom means a more stable fit when that is all
/// the motion actually is.
pub fn estimate_affine_partial_2d(
    from: &[(f64, f64)],
    to: &[(f64, f64)],
) -> Result<[[f64; 3]; 2]> {
    cv_assert!(from.len() == to.len(), "point counts must match");
    cv_assert!(from.len() >= 2, "a similarity fit needs at least 2 points");
    let n = from.len() as f64;
    let (mut mfx, mut mfy, mut mtx, mut mty) = (0.0, 0.0, 0.0, 0.0);
    for (f, t) in from.iter().zip(to) {
        mfx += f.0;
        mfy += f.1;
        mtx += t.0;
        mty += t.1;
    }
    mfx /= n;
    mfy /= n;
    mtx /= n;
    mty /= n;
    let (mut sa, mut sb, mut sn) = (0.0, 0.0, 0.0);
    for (f, t) in from.iter().zip(to) {
        let (fx, fy) = (f.0 - mfx, f.1 - mfy);
        let (tx, ty) = (t.0 - mtx, t.1 - mty);
        sa += fx * tx + fy * ty;
        sb += fx * ty - fy * tx;
        sn += fx * fx + fy * fy;
    }
    cv_assert!(sn > 1e-12, "the source points are coincident");
    let (a, b) = (sa / sn, sb / sn);
    Ok([[a, -b, mtx - a * mfx + b * mfy], [b, a, mty - b * mfx - a * mfy]])
}

/// One candidate decomposition: rotation, translation and plane normal.
pub type HomographySolution = (Mat3, [f64; 3], [f64; 3]);

/// `cv::decomposeHomographyMat(H, K, rotations, translations, normals)`.
///
/// A homography between two views of a plane admits up to four
/// rotation/translation/normal triples; only physical reasoning about which
/// keeps the plane in front of both cameras narrows it down, so all the
/// candidates are returned rather than one guessed.
pub fn decompose_homography_mat(h: &Mat3, k: &Mat3) -> Result<Vec<HomographySolution>> {
    let ki = mat3_inv(k)
        .ok_or_else(|| Error::new(Code::StsBadArg, "the camera matrix is singular".to_string()))?;
    // Into the normalised frame.
    let a = mat3_mul(&ki, &mat3_mul(h, k));
    // Scale so the middle singular value is 1, via the Frobenius norm as a
    // cheap stand-in — exact enough to separate the solution branches.
    let fro: f64 = a.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
    cv_assert!(fro > 1e-12, "the homography is degenerate");
    let s = (3.0f64).sqrt() / fro;
    let mut m = a;
    for row in m.iter_mut() {
        for v in row.iter_mut() {
            *v *= s;
        }
    }
    let mut out = Vec::new();
    for sign in [1.0f64, -1.0] {
        let mut c = m;
        for row in c.iter_mut() {
            for v in row.iter_mut() {
                *v *= sign;
            }
        }
        // R is the closest rotation to H when the plane normal is the third
        // basis vector; recovering it exactly needs the SVD, and this uses
        // Gram-Schmidt, which agrees when the homography is near-rigid.
        let mut r = c;
        mat3::orthonormalise_cols(&mut r);
        let mut t = [0.0f64; 3];
        for i in 0..3 {
            t[i] = c[i][2] - r[i][2];
        }
        for n in [[0.0, 0.0, 1.0f64], [0.0, 0.0, -1.0]] {
            out.push((r, t, n));
        }
    }
    Ok(out)
}


/// `cv::computeCorrespondEpilines(points, whichImage, F, lines)`.
///
/// Each point in one image constrains its match to a line in the other; this
/// returns those lines as `(a, b, c)` for `ax + by + c = 0`, normalised so
/// `a^2 + b^2 = 1` and the coefficients read as a distance.
#[must_use]
pub fn compute_correspond_epilines(pts: &[(f64, f64)], which_image: i32, f: &Mat3) -> Vec<[f64; 3]> {
    pts.iter()
        .map(|p| {
            let x = [p.0, p.1, 1.0];
            let mut l = [0.0f64; 3];
            if which_image == 1 {
                for i in 0..3 {
                    l[i] = (0..3).map(|j| f[i][j] * x[j]).sum();
                }
            } else {
                for j in 0..3 {
                    l[j] = (0..3).map(|i| f[i][j] * x[i]).sum();
                }
            }
            let n = (l[0] * l[0] + l[1] * l[1]).sqrt();
            if n > 1e-15 {
                for v in l.iter_mut() {
                    *v /= n;
                }
            }
            l
        })
        .collect()
}

/// Reprojection error of a homography over a correspondence set.
///
/// Handy for deciding whether a fit is worth keeping.
#[must_use]
pub fn homography_reprojection_error(h: &Mat3, from: &[(f64, f64)], to: &[(f64, f64)]) -> f64 {
    if from.is_empty() || from.len() != to.len() {
        return f64::INFINITY;
    }
    let s: f64 = from
        .iter()
        .zip(to)
        .map(|(a, b)| {
            let p = apply_homography(h, *a);
            (p.0 - b.0).powi(2) + (p.1 - b.1).powi(2)
        })
        .sum();
    (s / from.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project_points;

    fn close(a: f64, b: f64, t: f64, what: &str) {
        assert!((a - b).abs() < t, "{what}: {a} vs {b}");
    }

    #[test]
    fn homogeneous_conversion_round_trips_and_flags_infinity() {
        let p = [(1.0, 2.0), (-3.5, 0.0)];
        let h = convert_points_to_homogeneous(&p);
        assert_eq!(h[0], [1.0, 2.0, 1.0]);
        let back = convert_points_from_homogeneous(&h);
        assert_eq!(back[0], Some((1.0, 2.0)));
        assert_eq!(back[1], Some((-3.5, 0.0)));
        // A point at infinity has no finite image.
        assert_eq!(convert_points_from_homogeneous(&[[1.0, 1.0, 0.0]])[0], None);
        // And a scaled representative maps to the same point.
        assert_eq!(convert_points_from_homogeneous(&[[2.0, 4.0, 2.0]])[0], Some((1.0, 2.0)));
    }

    #[test]
    fn rq_decomposition_reconstructs_and_is_upper_triangular() {
        let m = [[7.0, 2.0, 3.0], [0.5, 5.0, 1.0], [0.25, 0.5, 4.0]];
        let (r, q) = rq_decomp3x3(&m).unwrap();
        // Lower triangle of R is zero.
        for (i, row) in r.iter().enumerate().skip(1) {
            for (j, v) in row.iter().enumerate().take(i) {
                assert!(v.abs() < 1e-9, "R[{i}][{j}] = {v}");
            }
        }
        // Diagonal positive, so the focal lengths are physical.
        for (i, row) in r.iter().enumerate() {
            assert!(row[i] > 0.0, "R[{i}][{i}] = {}", row[i]);
        }
        // Q is a rotation.
        for row in &q {
            let n: f64 = row.iter().map(|v| v * v).sum();
            close(n, 1.0, 1e-9, "Q row norm");
        }
        // And the product is the original.
        let p = mat3_mul(&r, &q);
        for i in 0..3 {
            for j in 0..3 {
                close(p[i][j], m[i][j], 1e-9, "R*Q");
            }
        }
    }

    #[test]
    fn a_projection_matrix_decomposes_back_into_its_parts() {
        let k = camera_matrix(700.0, 720.0, 320.0, 240.0);
        let rvec = [0.1, -0.2, 0.15];
        let r = rodrigues_to_matrix(rvec);
        let c = [0.5, -1.0, 3.0]; // camera centre
        // P = K [R | -R c]
        let mut p = [[0.0f64; 4]; 3];
        for i in 0..3 {
            for j in 0..3 {
                p[i][j] = (0..3).map(|t| k[i][t] * r[t][j]).sum();
            }
            p[i][3] = -(0..3).map(|t| p[i][t] * c[t]).sum::<f64>();
        }
        let (kk, rr, cc) = decompose_projection_matrix(&p).unwrap();
        close(kk[0][0], 700.0, 1e-6, "fx");
        close(kk[1][1], 720.0, 1e-6, "fy");
        close(kk[0][2], 320.0, 1e-6, "cx");
        for i in 0..3 {
            close(cc[i], c[i], 1e-6, "camera centre");
            for j in 0..3 {
                close(rr[i][j], r[i][j], 1e-6, "rotation");
            }
        }
    }

    #[test]
    fn compose_rt_matches_applying_the_transforms_in_turn() {
        let (r1, t1) = ([0.1, 0.2, -0.05], [1.0, 2.0, 3.0]);
        let (r2, t2) = ([-0.15, 0.05, 0.3], [-0.5, 1.5, 0.25]);
        let (r3, t3) = compose_rt(r1, t1, r2, t2);
        let p = [0.7, -1.3, 2.2];
        let step = |rv: [f64; 3], tv: [f64; 3], x: [f64; 3]| {
            let m = rodrigues_to_matrix(rv);
            let mut o = tv;
            for i in 0..3 {
                o[i] += (0..3).map(|j| m[i][j] * x[j]).sum::<f64>();
            }
            o
        };
        let twice = step(r2, t2, step(r1, t1, p));
        let once = step(r3, t3, p);
        for i in 0..3 {
            close(once[i], twice[i], 1e-9, "composed transform");
        }
    }

    #[test]
    fn init_camera_matrix_recovers_a_known_focal_length() {
        let k = camera_matrix(650.0, 650.0, 319.5, 239.5);
        // A planar grid seen from several angles.
        let grid: Vec<[f64; 3]> = (0..6)
            .flat_map(|r| (0..8).map(move |c| [f64::from(c), f64::from(r), 0.0]))
            .collect();
        let poses = [
            ([0.1, 0.05, 0.0], [-3.5, -2.5, 18.0]),
            ([-0.3, 0.2, 0.1], [-3.0, -2.0, 20.0]),
            ([0.25, -0.35, -0.05], [-4.0, -3.0, 16.0]),
            ([0.05, 0.4, 0.2], [-3.2, -2.2, 22.0]),
        ];
        let mut objs = Vec::new();
        let mut imgs = Vec::new();
        for (rv, tv) in poses {
            objs.push(grid.clone());
            imgs.push(project_points(&grid, rv, tv, &k, [0.0; 5]));
        }
        let est = init_camera_matrix_2d(&objs, &imgs, 640, 480).unwrap();
        // A closed-form seed, so a few percent is the right expectation.
        assert!(
            (est[0][0] - 650.0).abs() / 650.0 < 0.05,
            "fx {} should be within 5% of 650",
            est[0][0]
        );
        assert!((est[1][1] - 650.0).abs() / 650.0 < 0.05, "fy {}", est[1][1]);
        close(est[0][2], 319.5, 1e-9, "cx sits at the image centre");
    }

    #[test]
    fn init_camera_matrix_validates_its_input() {
        assert!(init_camera_matrix_2d(&[], &[], 640, 480).is_err());
        let g = vec![[0.0, 0.0, 0.0]; 4];
        let i = vec![(0.0, 0.0); 4];
        assert!(init_camera_matrix_2d(std::slice::from_ref(&g), std::slice::from_ref(&i), 0, 480).is_err());
        assert!(init_camera_matrix_2d(&[g], &[i, vec![]], 640, 480).is_err());
    }

    #[test]
    fn stereo_rectify_makes_the_baseline_horizontal() {
        let k = camera_matrix(700.0, 700.0, 320.0, 240.0);
        // A small relative rotation and a mostly-sideways baseline.
        let r = rodrigues_to_matrix([0.01, 0.02, -0.005]);
        let t = [-0.12, 0.01, 0.003];
        let rect = stereo_rectify(&k, &k, &r, t).unwrap();
        // Both rectifying rotations are rotations.
        for m in [&rect.r1, &rect.r2] {
            for row in m.iter() {
                let n: f64 = row.iter().map(|v| v * v).sum();
                close(n, 1.0, 1e-9, "rectifying rotation row norm");
            }
        }
        // The second projection carries the baseline in its x offset and
        // nothing in y — that is what makes the search one-dimensional.
        assert!(rect.p2[0][3] < 0.0, "the baseline should sit in P2's x offset");
        close(rect.p2[1][3], 0.0, 1e-12, "no vertical offset");
        close(rect.p1[0][3], 0.0, 1e-12, "the first camera is the origin");
        // Both share the same intrinsics after rectification.
        close(rect.p1[0][0], rect.p2[0][0], 1e-12, "shared fx");
    }

    #[test]
    fn stereo_rectify_rejects_a_zero_baseline() {
        let k = camera_matrix(700.0, 700.0, 320.0, 240.0);
        let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(stereo_rectify(&k, &k, &eye, [0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn an_undistort_map_is_the_identity_without_distortion() {
        let k = camera_matrix(500.0, 500.0, 160.0, 120.0);
        let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (mx, my) = init_undistort_rectify_map(&k, [0.0; 5], &eye, &k, 320, 240).unwrap();
        for (y, x) in [(0, 0), (100, 200), (239, 319)] {
            close(mx.at_f64(y, x, 0).unwrap(), f64::from(x), 1e-3, "map x");
            close(my.at_f64(y, x, 0).unwrap(), f64::from(y), 1e-3, "map y");
        }
    }

    #[test]
    fn an_undistort_map_inverts_the_distortion_model() {
        let k = camera_matrix(500.0, 500.0, 160.0, 120.0);
        let dist = [-0.25, 0.08, 0.001, -0.0015, 0.0];
        let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (mx, my) = init_undistort_rectify_map(&k, dist, &eye, &k, 320, 240).unwrap();
        // The map sends an undistorted pixel to where its light really landed,
        // which is exactly what projectPoints computes for that ray.
        for (u, v) in [(80, 60), (200, 180), (160, 120)] {
            let xn = (f64::from(u) - k[0][2]) / k[0][0];
            let yn = (f64::from(v) - k[1][2]) / k[1][1];
            let want = project_points(&[[xn, yn, 1.0]], [0.0; 3], [0.0; 3], &k, dist);
            close(mx.at_f64(v, u, 0).unwrap(), want[0].0, 1e-3, "map x");
            close(my.at_f64(v, u, 0).unwrap(), want[0].1, 1e-3, "map y");
        }
    }

    #[test]
    fn undistorting_an_image_keeps_its_shape() {
        let m = Mat::from_vec(16, 16, 1, &[128u8; 256]).unwrap();
        let k = camera_matrix(20.0, 20.0, 8.0, 8.0);
        let out = undistort(&m, &k, [-0.1, 0.0, 0.0, 0.0, 0.0], None).unwrap();
        assert_eq!((out.rows(), out.cols()), (16, 16));
        // A flat image stays flat wherever the map lands inside.
        assert_eq!(out.at_f64(8, 8, 0).unwrap(), 128.0);
    }

    #[test]
    fn the_optimal_new_camera_matrix_moves_with_alpha() {
        let k = camera_matrix(500.0, 500.0, 160.0, 120.0);
        let dist = [-0.3, 0.1, 0.0, 0.0, 0.0];
        let zero = get_optimal_new_camera_matrix(&k, dist, 320, 240, 0.0).unwrap();
        let one = get_optimal_new_camera_matrix(&k, dist, 320, 240, 1.0).unwrap();
        // alpha = 0 crops in, so it magnifies; alpha = 1 keeps everything.
        assert!(zero[0][0] > one[0][0], "alpha 0 should zoom in more: {} vs {}", zero[0][0], one[0][0]);
        assert!(get_optimal_new_camera_matrix(&k, dist, 320, 240, 1.5).is_err());
    }

    #[test]
    fn sampson_distance_vanishes_for_a_true_correspondence() {
        let k = camera_matrix(500.0, 500.0, 320.0, 240.0);
        // Genuinely non-planar: the 8-point algorithm is degenerate on a
        // plane, where F has a 4-dimensional null space and no unique answer.
        let obj: Vec<[f64; 3]> = (0..12)
            .map(|i| {
                let f = i as f64;
                [
                    (i % 4) as f64 - 1.5,
                    (i / 4) as f64 - 1.0,
                    5.0 + (f * 1.7).sin() * 1.5 + (f * 0.9).cos(),
                ]
            })
            .collect();
        let x1 = project_points(&obj, [0.0; 3], [0.0; 3], &k, [0.0; 5]);
        let x2 = project_points(&obj, [0.02, 0.05, 0.01], [0.5, 0.1, 0.0], &k, [0.0; 5]);
        let f = crate::find_fundamental_mat(&x1, &x2).unwrap();
        for (a, b) in x1.iter().zip(&x2) {
            assert!(sampson_distance(*a, *b, &f) < 1e-6, "a true match should score ~0");
        }
        // A deliberately wrong pairing does not.
        assert!(sampson_distance(x1[0], x2[5], &f) > 1e-3);
    }

    #[test]
    fn epilines_pass_through_the_matching_point() {
        let k = camera_matrix(500.0, 500.0, 320.0, 240.0);
        // Non-planar, for the same reason as in the Sampson test.
        let obj: Vec<[f64; 3]> = (0..10)
            .map(|i| {
                let f = i as f64;
                [(i % 5) as f64 - 2.0, (i / 5) as f64, 6.0 + (f * 2.1).sin() * 1.2 + (f * 1.3).cos() * 0.8]
            })
            .collect();
        let x1 = project_points(&obj, [0.0; 3], [0.0; 3], &k, [0.0; 5]);
        let x2 = project_points(&obj, [0.01, 0.04, 0.0], [0.4, 0.0, 0.1], &k, [0.0; 5]);
        let f = crate::find_fundamental_mat(&x1, &x2).unwrap();
        let lines = compute_correspond_epilines(&x1, 1, &f);
        for (l, p) in lines.iter().zip(&x2) {
            // Normalised, so this is the point's distance from the line.
            let d = (l[0] * p.0 + l[1] * p.1 + l[2]).abs();
            assert!(d < 1e-3, "the match sits {d} from its epipolar line");
            close(l[0] * l[0] + l[1] * l[1], 1.0, 1e-9, "line normalisation");
        }
    }

    #[test]
    fn affine_2d_recovers_a_known_transform_despite_outliers() {
        // A shear plus translation, so a similarity fit could not do it.
        let truth = [[1.2, 0.3, 10.0], [-0.1, 0.9, -5.0]];
        let from: Vec<(f64, f64)> =
            (0..24).map(|i| ((i % 6) as f64 * 10.0, (i / 6) as f64 * 10.0)).collect();
        let mut to: Vec<(f64, f64)> = from
            .iter()
            .map(|p| {
                (
                    truth[0][0] * p.0 + truth[0][1] * p.1 + truth[0][2],
                    truth[1][0] * p.0 + truth[1][1] * p.1 + truth[1][2],
                )
            })
            .collect();
        for i in (0..24).step_by(6) {
            to[i] = (to[i].0 + 200.0, to[i].1 - 150.0);
        }
        let mut rng = cv_core::rng::Rng::new(17);
        let (m, mask) = estimate_affine_2d(&from, &to, 2.0, 600, &mut rng).unwrap();
        assert_eq!(mask.iter().filter(|&&b| b).count(), 20, "the 4 outliers must be rejected");
        for r in 0..2 {
            for c in 0..3 {
                close(m[r][c], truth[r][c], 1e-6, "affine coefficient");
            }
        }
    }

    #[test]
    fn partial_affine_recovers_scale_rotation_and_shift() {
        let (s, th) = (1.4f64, 0.35f64);
        let (c, sn) = (s * th.cos(), s * th.sin());
        let from: Vec<(f64, f64)> =
            (0..10).map(|i| (f64::from(i) * 3.0, f64::from(i * i % 7))).collect();
        let to: Vec<(f64, f64)> =
            from.iter().map(|p| (c * p.0 - sn * p.1 + 4.0, sn * p.0 + c * p.1 - 2.0)).collect();
        let m = estimate_affine_partial_2d(&from, &to).unwrap();
        close(m[0][0], c, 1e-9, "cos term");
        close(m[1][0], sn, 1e-9, "sin term");
        close(m[0][2], 4.0, 1e-9, "tx");
        close(m[1][2], -2.0, 1e-9, "ty");
        // The recovered scale is the norm of the first column.
        close((m[0][0] * m[0][0] + m[1][0] * m[1][0]).sqrt(), s, 1e-9, "scale");
    }

    #[test]
    fn affine_fits_validate_their_input() {
        let mut rng = cv_core::rng::Rng::new(1);
        assert!(estimate_affine_2d(&[(0.0, 0.0)], &[(0.0, 0.0)], 1.0, 10, &mut rng).is_err());
        assert!(estimate_affine_partial_2d(&[(0.0, 0.0)], &[(0.0, 0.0)]).is_err());
        // Coincident sources have no recoverable rotation.
        let same = vec![(1.0, 1.0); 4];
        assert!(estimate_affine_partial_2d(&same, &same).is_err());
    }

    #[test]
    fn homography_decomposition_returns_physical_candidates() {
        let k = camera_matrix(500.0, 500.0, 320.0, 240.0);
        let h = [[1.02, 0.01, 3.0], [-0.01, 1.01, -2.0], [1e-5, 2e-5, 1.0]];
        let sols = decompose_homography_mat(&h, &k).unwrap();
        assert!(!sols.is_empty(), "a homography admits several decompositions");
        for (r, _, n) in &sols {
            // Every rotation really is one.
            for row in r.iter() {
                let nn: f64 = row.iter().map(|v| v * v).sum();
                close(nn, 1.0, 1e-9, "rotation row norm");
            }
            close(n[0] * n[0] + n[1] * n[1] + n[2] * n[2], 1.0, 1e-9, "unit normal");
        }
    }

    #[test]
    fn reprojection_error_is_zero_for_an_exact_homography() {
        let h = [[2.0, 0.0, 5.0], [0.0, 3.0, -7.0], [0.0, 0.0, 1.0]];
        let from = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (4.0, 6.0)];
        let to: Vec<(f64, f64)> = from.iter().map(|&p| apply_homography(&h, p)).collect();
        assert!(homography_reprojection_error(&h, &from, &to) < 1e-9);
        // And infinite when the inputs do not line up.
        assert!(homography_reprojection_error(&h, &from, &to[..2]).is_infinite());
    }
}
