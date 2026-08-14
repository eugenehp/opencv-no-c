//! `cv::calib3d` for `opencv-no-c` — rotations, homographies and pose.
//!
//! # Scope
//!
//! The geometry that does not need an optimiser: Rodrigues both ways,
//! `findHomography` (DLT, with and without RANSAC), `projectPoints`,
//! `solvePnP` by DLT, `findFundamentalMat` by the normalised eight-point
//! algorithm, and the triangulation and decomposition helpers around them.
//!
//! Camera *calibration* itself — `calibrateCamera`, `stereoCalibrate` — is not
//! here. Those are Levenberg-Marquardt refinements over a full reprojection
//! residual, and a version without the refinement would return numbers that
//! look plausible and are wrong, which is worse than not having it.

#![warn(missing_docs)]

use cv_core::types::CV_64FC1;
use cv_core::{cv_assert, Code, Error, Mat, MatLike, MatLikeMut, Result};

pub mod calibrate;
pub mod constants;
pub mod mat3;
pub mod patterns;
pub mod pnp;
pub mod stereo;
pub mod stereo_match;
pub mod usac;

pub use pnp::*;
pub use stereo::*;

/// A 3x3 matrix in row-major order, the shape most of this module passes around.
pub type Mat3 = [[f64; 3]; 3];

/// Read a `MatLike` into a dense row-major buffer.
fn dense<M: MatLike + ?Sized>(m: &M) -> Result<(Vec<f64>, usize, usize)> {
    cv_assert!(m.channels() == 1, "expected a single-channel matrix");
    let (h, w) = (m.rows() as usize, m.cols() as usize);
    let mut v = vec![0.0f64; h * w];
    for y in 0..h {
        for x in 0..w {
            v[y * w + x] = m.at_f64(y as i32, x as i32, 0)?;
        }
    }
    Ok((v, h, w))
}

/// Wrap a 3x3 array as a `CV_64FC1` matrix.
pub fn mat3_to_mat(m: &Mat3) -> Result<Mat> {
    let flat: Vec<f64> = m.iter().flat_map(|r| r.iter().copied()).collect();
    Mat::from_vec(3, 3, 1, &flat)
}

/// Read a 3x3 matrix out of a `MatLike`.
pub fn mat_to_mat3<M: MatLike + ?Sized>(m: &M) -> Result<Mat3> {
    let (v, h, w) = dense(m)?;
    cv_assert!(h == 3 && w == 3, "expected a 3x3 matrix, got {w}x{h}");
    Ok([
        [v[0], v[1], v[2]],
        [v[3], v[4], v[5]],
        [v[6], v[7], v[8]],
    ])
}

/// `cv::Rodrigues(src, dst)` — rotation vector to rotation matrix.
///
/// The vector's direction is the axis and its length the angle in radians.
/// Below `1e-12` the small-angle limit is the identity, which is upstream's
/// behaviour and avoids dividing by the norm.
#[must_use]
pub fn rodrigues_to_matrix(rvec: [f64; 3]) -> Mat3 {
    let theta = (rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]).sqrt();
    if theta < 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let (x, y, z) = (rvec[0] / theta, rvec[1] / theta, rvec[2] / theta);
    let (s, c) = (theta.sin(), theta.cos());
    let c1 = 1.0 - c;
    [
        [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
        [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
        [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
    ]
}

/// `cv::Rodrigues(src, dst)` — rotation matrix to rotation vector.
#[must_use]
pub fn rodrigues_to_vector(r: &Mat3) -> [f64; 3] {
    // The antisymmetric part gives the axis, the trace the angle.
    let (ax, ay, az) = (
        r[2][1] - r[1][2],
        r[0][2] - r[2][0],
        r[1][0] - r[0][1],
    );
    let s = (ax * ax + ay * ay + az * az).sqrt() * 0.5;
    let c = ((r[0][0] + r[1][1] + r[2][2] - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = s.atan2(c);
    if s < 1e-12 {
        if c > 0.0 {
            return [0.0, 0.0, 0.0]; // identity
        }
        // A half turn: the axis comes from the diagonal, whose signs the
        // antisymmetric part cannot supply.
        let d = [
            ((r[0][0] + 1.0) * 0.5).max(0.0).sqrt(),
            ((r[1][1] + 1.0) * 0.5).max(0.0).sqrt(),
            ((r[2][2] + 1.0) * 0.5).max(0.0).sqrt(),
        ];
        let big = if d[0] >= d[1] && d[0] >= d[2] {
            0
        } else if d[1] >= d[2] {
            1
        } else {
            2
        };
        let mut v = d;
        // Fix the relative signs from the off-diagonal entries.
        for k in 0..3 {
            if k != big {
                let off = r[big][k];
                if off < 0.0 {
                    v[k] = -v[k];
                }
            }
        }
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        return [
            v[0] / n * theta,
            v[1] / n * theta,
            v[2] / n * theta,
        ];
    }
    let k = theta / (2.0 * s);
    [ax * k, ay * k, az * k]
}

/// Solve `A x = 0` for the unit `x` minimising `|Ax|`, via the smallest
/// eigenvector of `A^T A`.
fn null_vector(a: &[Vec<f64>], n: usize) -> Option<Vec<f64>> {
    let mut ata = vec![0.0f64; n * n];
    for row in a {
        for i in 0..n {
            for j in 0..n {
                ata[i * n + j] += row[i] * row[j];
            }
        }
    }
    // Jacobi eigen-decomposition; the smallest eigenvalue's vector is the answer.
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                off += ata[i * n + j] * ata[i * n + j];
            }
        }
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = ata[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (ata[q * n + q] - ata[p * n + p]) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (kp, kq) = (ata[k * n + p], ata[k * n + q]);
                    ata[k * n + p] = c * kp - s * kq;
                    ata[k * n + q] = s * kp + c * kq;
                }
                for k in 0..n {
                    let (pk, qk) = (ata[p * n + k], ata[q * n + k]);
                    ata[p * n + k] = c * pk - s * qk;
                    ata[q * n + k] = s * pk + c * qk;
                }
                for k in 0..n {
                    let (kp, kq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * kp - s * kq;
                    v[k * n + q] = s * kp + c * kq;
                }
            }
        }
    }
    let mut best = 0usize;
    for i in 1..n {
        if ata[i * n + i] < ata[best * n + best] {
            best = i;
        }
    }
    let out: Vec<f64> = (0..n).map(|k| v[k * n + best]).collect();
    let norm = out.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-300 {
        None
    } else {
        Some(out.into_iter().map(|x| x / norm).collect())
    }
}

/// Hartley normalisation: centre the points and scale so the mean distance from
/// the origin is `sqrt(2)`.
///
/// Skipping this is the classic way to get a homography that looks fine on
/// synthetic data and falls apart on real pixel coordinates, because the
/// design matrix becomes badly conditioned once values reach the hundreds.
fn normalise(pts: &[(f64, f64)]) -> (Vec<(f64, f64)>, Mat3) {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let mean_d = pts
        .iter()
        .map(|p| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    let s = if mean_d > 1e-12 { 2f64.sqrt() / mean_d } else { 1.0 };
    let t = [[s, 0.0, -s * cx], [0.0, s, -s * cy], [0.0, 0.0, 1.0]];
    let out = pts.iter().map(|p| ((p.0 - cx) * s, (p.1 - cy) * s)).collect();
    (out, t)
}

/// 3x3 matrix product.
#[must_use]
pub fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [[0.0f64; 3]; 3];
    for (i, orow) in o.iter_mut().enumerate() {
        for (j, oij) in orow.iter_mut().enumerate() {
            *oij = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

/// 3x3 inverse, or `None` when singular.
#[must_use]
pub fn mat3_inv(m: &Mat3) -> Option<Mat3> {
    let d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if d.abs() < 1e-300 {
        return None;
    }
    let mut o = [[0.0f64; 3]; 3];
    for (i, orow) in o.iter_mut().enumerate() {
        for (j, oij) in orow.iter_mut().enumerate() {
            let (r0, r1) = ((j + 1) % 3, (j + 2) % 3);
            let (c0, c1) = ((i + 1) % 3, (i + 2) % 3);
            *oij = (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]) / d;
        }
    }
    Some(o)
}

/// Apply a homography to a point, dividing through by `w`.
#[must_use]
pub fn apply_homography(h: &Mat3, p: (f64, f64)) -> (f64, f64) {
    let x = h[0][0] * p.0 + h[0][1] * p.1 + h[0][2];
    let y = h[1][0] * p.0 + h[1][1] * p.1 + h[1][2];
    let w = h[2][0] * p.0 + h[2][1] * p.1 + h[2][2];
    if w.abs() < 1e-300 {
        (f64::MAX, f64::MAX)
    } else {
        (x / w, y / w)
    }
}

/// `cv::findHomography(srcPoints, dstPoints, method)` with `method = 0`.
///
/// The normalised direct linear transform over every correspondence. At least
/// four points are needed; more are least-squares fitted.
pub fn find_homography(src: &[(f64, f64)], dst: &[(f64, f64)]) -> Result<Mat3> {
    cv_assert!(
        src.len() == dst.len(),
        "findHomography got {} and {} points",
        src.len(),
        dst.len()
    );
    cv_assert!(src.len() >= 4, "findHomography needs at least 4 points, got {}", src.len());
    let (ns, ts) = normalise(src);
    let (nd, td) = normalise(dst);

    let mut a: Vec<Vec<f64>> = Vec::with_capacity(src.len() * 2);
    for (s, d) in ns.iter().zip(nd.iter()) {
        a.push(vec![-s.0, -s.1, -1.0, 0.0, 0.0, 0.0, d.0 * s.0, d.0 * s.1, d.0]);
        a.push(vec![0.0, 0.0, 0.0, -s.0, -s.1, -1.0, d.1 * s.0, d.1 * s.1, d.1]);
    }
    let h = null_vector(&a, 9)
        .ok_or_else(|| Error::new(Code::StsBadArg, "findHomography: degenerate points".to_string()))?;
    let hn = [
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], h[8]],
    ];
    // Undo the normalisation: H = Td^-1 * Hn * Ts.
    let tdi = mat3_inv(&td)
        .ok_or_else(|| Error::new(Code::StsBadArg, "findHomography: degenerate points".to_string()))?;
    let mut out = mat3_mul(&tdi, &mat3_mul(&hn, &ts));
    // The DLT is only the starting point: upstream follows it with 10
    // Levenberg-Marquardt iterations against the *forward reprojection error*,
    // which is what the homography is actually meant to minimise. The DLT
    // minimises an algebraic residual instead, and on a six-point fit that
    // difference reaches 0.11 in the matrix entries.
    // Fix the scale so H[2][2] is 1, as upstream returns it. This has to happen
    // **before** the refinement: `H` and `kH` describe the same map, so the
    // objective is scale-invariant, but the Levenberg-Marquardt *step* is not
    // — the damping is relative to `diag(JtJ)`, which moves with the scale.
    // Refining an unnormalised `H` stalls at a residual of 10.659 where the
    // normalised one reaches cv2's 10.624.
    if out[2][2].abs() > 1e-300 {
        let k = out[2][2];
        for row in out.iter_mut() {
            for v in row.iter_mut() {
                *v /= k;
            }
        }
    }
    refine_homography(&mut out, src, dst);
    if out[2][2].abs() > 1e-300 {
        let k = out[2][2];
        for row in out.iter_mut() {
            for v in row.iter_mut() {
                *v /= k;
            }
        }
    }
    Ok(out)
}

/// `cv::findHomography(..., RANSAC, ransacReprojThreshold)`.
///
/// Returns the homography and the inlier mask. `rng` drives the sampling, so
/// runs are reproducible; upstream uses its global RNG and is not.
pub fn find_homography_ransac(
    src: &[(f64, f64)],
    dst: &[(f64, f64)],
    threshold: f64,
    iterations: usize,
    rng: &mut cv_core::rng::Rng,
) -> Result<(Mat3, Vec<bool>)> {
    cv_assert!(src.len() == dst.len(), "point counts must match");
    cv_assert!(src.len() >= 4, "RANSAC needs at least 4 points");
    let n = src.len();
    let t2 = threshold * threshold;
    let mut best: (usize, Vec<bool>) = (0, vec![false; n]);

    for _ in 0..iterations.max(1) {
        // Four distinct correspondences.
        let mut pick = [0usize; 4];
        for k in 0..4 {
            loop {
                let c = (rng.next_u32() as usize) % n;
                if !pick[..k].contains(&c) {
                    pick[k] = c;
                    break;
                }
            }
        }
        let s: Vec<(f64, f64)> = pick.iter().map(|&i| src[i]).collect();
        let d: Vec<(f64, f64)> = pick.iter().map(|&i| dst[i]).collect();
        let Ok(h) = find_homography(&s, &d) else { continue };

        let mask: Vec<bool> = src
            .iter()
            .zip(dst.iter())
            .map(|(a, b)| {
                let p = apply_homography(&h, *a);
                let (dx, dy) = (p.0 - b.0, p.1 - b.1);
                dx * dx + dy * dy <= t2
            })
            .collect();
        let count = mask.iter().filter(|&&m| m).count();
        if count > best.0 {
            best = (count, mask);
        }
    }
    cv_assert!(best.0 >= 4, "RANSAC found only {} inliers", best.0);
    // Refit on the consensus set.
    let s: Vec<(f64, f64)> =
        src.iter().zip(&best.1).filter(|(_, &m)| m).map(|(p, _)| *p).collect();
    let d: Vec<(f64, f64)> =
        dst.iter().zip(&best.1).filter(|(_, &m)| m).map(|(p, _)| *p).collect();
    Ok((find_homography(&s, &d)?, best.1))
}

/// `cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints)`.
///
/// `dist` is `[k1, k2, p1, p2, k3]`, upstream's ordering. Pass zeros for a
/// pinhole camera.
pub fn project_points(
    object: &[[f64; 3]],
    rvec: [f64; 3],
    tvec: [f64; 3],
    camera: &Mat3,
    dist: [f64; 5],
) -> Vec<(f64, f64)> {
    let r = rodrigues_to_matrix(rvec);
    let (fx, fy) = (camera[0][0], camera[1][1]);
    let (cx, cy) = (camera[0][2], camera[1][2]);
    let skew = camera[0][1];
    object
        .iter()
        .map(|p| {
            // Into the camera frame.
            let mut c = [0.0f64; 3];
            for i in 0..3 {
                c[i] = r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2] + tvec[i];
            }
            if c[2].abs() < 1e-300 {
                return (f64::MAX, f64::MAX);
            }
            let (xn, yn) = (c[0] / c[2], c[1] / c[2]);
            // Brown-Conrady distortion.
            let r2 = xn * xn + yn * yn;
            let radial = 1.0 + dist[0] * r2 + dist[1] * r2 * r2 + dist[4] * r2 * r2 * r2;
            let xd = xn * radial + 2.0 * dist[2] * xn * yn + dist[3] * (r2 + 2.0 * xn * xn);
            let yd = yn * radial + dist[2] * (r2 + 2.0 * yn * yn) + 2.0 * dist[3] * xn * yn;
            (fx * xd + skew * yd + cx, fy * yd + cy)
        })
        .collect()
}

/// Result of [`solve_pnp`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    /// Rotation as a Rodrigues vector.
    pub rvec: [f64; 3],
    /// Translation.
    pub tvec: [f64; 3],
}

/// `cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec)`
/// with `flags = SOLVEPNP_DLT`-style direct linear transform.
///
/// Needs at least six non-coplanar correspondences. The rotation block is
/// re-orthogonalised, because the raw DLT solution is only approximately a
/// rotation and feeding a non-orthogonal matrix to [`rodrigues_to_vector`]
/// silently produces a wrong angle.
///
/// This is the *linear* solver alone. `cv::solvePnP` defaults to
/// `SOLVEPNP_ITERATIVE`, which is this seed followed by a Levenberg-Marquardt
/// minimisation of the reprojection error; to match it, pass the result to
/// [`crate::calibrate::solve_pnp_refine_lm`]. On a planar target the difference
/// is not cosmetic — the DLT is degenerate there, and on a 6x5 board at 0.55 m
/// the seed alone is off by 0.22 in the rotation vector where the refined pose
/// agrees with cv2 to 1e-6.
pub fn solve_pnp(
    object: &[[f64; 3]],
    image: &[(f64, f64)],
    camera: &Mat3,
) -> Result<Pose> {
    cv_assert!(
        object.len() == image.len(),
        "solvePnP got {} object points and {} image points",
        object.len(),
        image.len()
    );
    cv_assert!(object.len() >= 6, "the DLT solver needs at least 6 points, got {}", object.len());
    let ki = mat3_inv(camera)
        .ok_or_else(|| Error::new(Code::StsBadArg, "the camera matrix is singular".to_string()))?;

    // Normalised image coordinates.
    let norm: Vec<(f64, f64)> = image
        .iter()
        .map(|&(u, v)| {
            let x = ki[0][0] * u + ki[0][1] * v + ki[0][2];
            let y = ki[1][0] * u + ki[1][1] * v + ki[1][2];
            let w = ki[2][0] * u + ki[2][1] * v + ki[2][2];
            (x / w, y / w)
        })
        .collect();

    // 12-unknown DLT for [R|t].
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(object.len() * 2);
    for (p, q) in object.iter().zip(norm.iter()) {
        let (x, y, z) = (p[0], p[1], p[2]);
        a.push(vec![x, y, z, 1.0, 0.0, 0.0, 0.0, 0.0, -q.0 * x, -q.0 * y, -q.0 * z, -q.0]);
        a.push(vec![0.0, 0.0, 0.0, 0.0, x, y, z, 1.0, -q.1 * x, -q.1 * y, -q.1 * z, -q.1]);
    }
    let s = null_vector(&a, 12)
        .ok_or_else(|| Error::new(Code::StsBadArg, "solvePnP: degenerate configuration".to_string()))?;
    let mut r = [
        [s[0], s[1], s[2]],
        [s[4], s[5], s[6]],
        [s[8], s[9], s[10]],
    ];
    let mut t = [s[3], s[7], s[11]];

    // Scale so the rotation block has unit rows, and fix the sign so the points
    // are in front of the camera.
    let scale = (0..3)
        .map(|i| (r[i][0] * r[i][0] + r[i][1] * r[i][1] + r[i][2] * r[i][2]).sqrt())
        .sum::<f64>()
        / 3.0;
    if scale < 1e-300 {
        return Err(Error::new(Code::StsBadArg, "solvePnP: degenerate scale".to_string()));
    }
    for row in r.iter_mut() {
        for v in row.iter_mut() {
            *v /= scale;
        }
    }
    for v in t.iter_mut() {
        *v /= scale;
    }
    if t[2] < 0.0 {
        for row in r.iter_mut() {
            for v in row.iter_mut() {
                *v = -*v;
            }
        }
        for v in t.iter_mut() {
            *v = -*v;
        }
    }
    mat3::orthonormalise_rows(&mut r);
    Ok(Pose { rvec: rodrigues_to_vector(&r), tvec: t })
}


/// `cv::findFundamentalMat(points1, points2, FM_8POINT)`.
///
/// The normalised eight-point algorithm, with the rank-2 constraint enforced
/// by zeroing the smallest singular value.
pub fn find_fundamental_mat(p1: &[(f64, f64)], p2: &[(f64, f64)]) -> Result<Mat3> {
    cv_assert!(p1.len() == p2.len(), "point counts must match");
    cv_assert!(p1.len() >= 8, "the eight-point algorithm needs 8 points, got {}", p1.len());
    let (n1, t1) = normalise(p1);
    let (n2, t2) = normalise(p2);
    let a: Vec<Vec<f64>> = n1
        .iter()
        .zip(n2.iter())
        .map(|(a, b)| {
            vec![
                b.0 * a.0,
                b.0 * a.1,
                b.0,
                b.1 * a.0,
                b.1 * a.1,
                b.1,
                a.0,
                a.1,
                1.0,
            ]
        })
        .collect();
    let f = null_vector(&a, 9)
        .ok_or_else(|| Error::new(Code::StsBadArg, "findFundamentalMat: degenerate points".to_string()))?;
    let mut fm = [
        [f[0], f[1], f[2]],
        [f[3], f[4], f[5]],
        [f[6], f[7], f[8]],
    ];
    enforce_rank2(&mut fm);
    // Undo normalisation: F = T2^T * Fn * T1.
    let t2t = [
        [t2[0][0], t2[1][0], t2[2][0]],
        [t2[0][1], t2[1][1], t2[2][1]],
        [t2[0][2], t2[1][2], t2[2][2]],
    ];
    let mut out = mat3_mul(&t2t, &mat3_mul(&fm, &t1));
    // Normalise so the largest entry is 1, which is how upstream scales it.
    let m = out.iter().flatten().fold(0.0f64, |a, &b| a.max(b.abs()));
    if m > 1e-300 {
        for row in out.iter_mut() {
            for v in row.iter_mut() {
                *v /= m;
            }
        }
    }
    Ok(out)
}

/// Zero the smallest singular value, making the matrix rank 2.
///
/// Without this a fundamental matrix has no epipole and every epipolar line
/// computed from it is subtly wrong.
fn enforce_rank2(f: &mut Mat3) {
    // Symmetric eigen-decomposition of F^T F gives V; project out the
    // direction of the smallest eigenvalue.
    let mut fta = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            fta[i][j] = (0..3).map(|k| f[k][i] * f[k][j]).sum();
        }
    }
    let rows: Vec<Vec<f64>> = (0..3).map(|i| fta[i].to_vec()).collect();
    let Some(v) = null_vector(&rows, 3) else { return };
    // F <- F - (F v) v^T.
    let fv: Vec<f64> = (0..3).map(|i| (0..3).map(|k| f[i][k] * v[k]).sum()).collect();
    for i in 0..3 {
        for j in 0..3 {
            f[i][j] -= fv[i] * v[j];
        }
    }
}

/// `cv::triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2, points4D)`.
///
/// Each projection matrix is 3x4, row-major. Returns homogeneous points
/// normalised so `w = 1`, or `None` for a point at infinity.
pub fn triangulate_points(
    p1: &[[f64; 4]; 3],
    p2: &[[f64; 4]; 3],
    x1: &[(f64, f64)],
    x2: &[(f64, f64)],
) -> Result<Vec<Option<[f64; 3]>>> {
    cv_assert!(x1.len() == x2.len(), "point counts must match");
    let mut out = Vec::with_capacity(x1.len());
    for (a, b) in x1.iter().zip(x2.iter()) {
        let rows: Vec<Vec<f64>> = vec![
            (0..4).map(|k| a.0 * p1[2][k] - p1[0][k]).collect(),
            (0..4).map(|k| a.1 * p1[2][k] - p1[1][k]).collect(),
            (0..4).map(|k| b.0 * p2[2][k] - p2[0][k]).collect(),
            (0..4).map(|k| b.1 * p2[2][k] - p2[1][k]).collect(),
        ];
        out.push(null_vector(&rows, 4).and_then(|x| {
            if x[3].abs() < 1e-12 {
                None
            } else {
                Some([x[0] / x[3], x[1] / x[3], x[2] / x[3]])
            }
        }));
    }
    Ok(out)
}

/// `cv::getDefaultNewCameraMatrix` / a pinhole intrinsic matrix.
#[must_use]
pub fn camera_matrix(fx: f64, fy: f64, cx: f64, cy: f64) -> Mat3 {
    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
}

/// `cv::undistortPoints(src, dst, cameraMatrix, distCoeffs)`.
///
/// Iteratively inverts the Brown-Conrady model, as upstream does; five passes
/// is upstream's default and is plenty for realistic coefficients.
pub fn undistort_points(
    pts: &[(f64, f64)],
    camera: &Mat3,
    dist: [f64; 5],
) -> Vec<(f64, f64)> {
    let (fx, fy) = (camera[0][0], camera[1][1]);
    let (cx, cy) = (camera[0][2], camera[1][2]);
    pts.iter()
        .map(|&(u, v)| {
            let (x0, y0) = ((u - cx) / fx, (v - cy) / fy);
            let (mut x, mut y) = (x0, y0);
            for _ in 0..5 {
                let r2 = x * x + y * y;
                let radial = 1.0 + dist[0] * r2 + dist[1] * r2 * r2 + dist[4] * r2 * r2 * r2;
                let dx = 2.0 * dist[2] * x * y + dist[3] * (r2 + 2.0 * x * x);
                let dy = dist[2] * (r2 + 2.0 * y * y) + 2.0 * dist[3] * x * y;
                x = (x0 - dx) / radial;
                y = (y0 - dy) / radial;
            }
            (x, y)
        })
        .collect()
}

/// Convert a [`Mat3`] into an owned `CV_64FC1` [`Mat`], for interop with the
/// rest of the workspace.
pub fn to_mat(m: &Mat3) -> Result<Mat> {
    let mut out = Mat::zeros(3, 3, CV_64FC1)?;
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out.pixel_mut::<f64>(i as i32, j as i32)?[0] = v;
        }
    }
    Ok(out)
}

/// `HomographyRefineCallback` driven by `LMSolver`, 10 iterations.
///
/// Minimises `sum |H*src / w - dst|^2` over all nine entries. The parameter
/// vector is scale-redundant — `H` and `2H` describe the same map — so the
/// normal matrix is singular in one direction; the LM damping is what keeps
/// the step well-posed rather than any explicit gauge fixing.
fn refine_homography(h: &mut Mat3, src: &[(f64, f64)], dst: &[(f64, f64)]) {
    let mut p = [h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2]];
    let residual = |p: &[f64; 9]| -> f64 {
        let mut acc = 0.0;
        for (s, d) in src.iter().zip(dst) {
            let w = p[6] * s.0 + p[7] * s.1 + p[8];
            if w.abs() < 1e-300 {
                return f64::MAX;
            }
            let x = (p[0] * s.0 + p[1] * s.1 + p[2]) / w;
            let y = (p[3] * s.0 + p[4] * s.1 + p[5]) / w;
            acc += (x - d.0) * (x - d.0) + (y - d.1) * (y - d.1);
        }
        acc
    };

    let mut lambda = 1e-3;
    let mut best = residual(&p);
    for _ in 0..200 {
        // Normal equations from the analytic Jacobian.
        let mut jtj = [[0.0f64; 9]; 9];
        let mut jte = [0.0f64; 9];
        for (s, d) in src.iter().zip(dst) {
            let w = p[6] * s.0 + p[7] * s.1 + p[8];
            if w.abs() < 1e-300 {
                return;
            }
            let iw = 1.0 / w;
            let x = (p[0] * s.0 + p[1] * s.1 + p[2]) * iw;
            let y = (p[3] * s.0 + p[4] * s.1 + p[5]) * iw;
            let rows = [
                ([s.0 * iw, s.1 * iw, iw, 0.0, 0.0, 0.0,
                  -x * s.0 * iw, -x * s.1 * iw, -x * iw], x - d.0),
                ([0.0, 0.0, 0.0, s.0 * iw, s.1 * iw, iw,
                  -y * s.0 * iw, -y * s.1 * iw, -y * iw], y - d.1),
            ];
            for (j, e) in rows {
                for a in 0..9 {
                    jte[a] += j[a] * e;
                    for b in 0..9 {
                        jtj[a][b] += j[a] * j[b];
                    }
                }
            }
        }

        // Damped step, retried with more damping if it does not improve.
        let mut improved = false;
        for _ in 0..20 {
            let mut m = jtj;
            for (i, row) in m.iter_mut().enumerate() {
                row[i] += lambda * jtj[i][i].max(1e-12);
            }
            // `JtJ` is singular along the scale direction, so a small lambda
            // can leave the damped matrix numerically singular. Upstream's
            // solver degrades gracefully there; abandoning the refinement
            // instead leaves a half-refined H, which is what stalled this at a
            // residual of 10.659 against cv2's 10.624.
            let Some(step) = solve9(m, jte) else {
                lambda *= 10.0;
                continue;
            };
            let mut cand = p;
            for i in 0..9 {
                cand[i] -= step[i];
            }
            let r = residual(&cand);
            if !r.is_finite() {
                lambda *= 10.0;
                continue;
            }
            if r < best {
                best = r;
                p = cand;
                lambda = (lambda * 0.1).max(1e-14);
                improved = true;
                break;
            }
            lambda *= 10.0;
            if lambda > 1e12 {
                break;
            }
        }
        // No damping level improved on this point: it is a stationary point,
        // and further outer iterations would repeat the same search.
        if !improved {
            break;
        }
    }
    *h = [[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], p[8]]];
}

/// Gaussian elimination with partial pivoting for the 9x9 normal equations.
fn solve9(mut a: [[f64; 9]; 9], mut b: [f64; 9]) -> Option<[f64; 9]> {
    for col in 0..9 {
        let piv = (col..9).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let (done, rest) = a.split_at_mut(col + 1);
        let pivot_row = &done[col];
        for (i, row) in rest.iter_mut().enumerate() {
            let f = row[col] / pivot_row[col];
            for (dst, srcv) in row.iter_mut().zip(pivot_row).skip(col) {
                *dst -= f * srcv;
            }
            b[col + 1 + i] -= f * b[col];
        }
    }
    let mut x = [0.0f64; 9];
    for row in (0..9).rev() {
        let mut acc = b[row];
        for c in row + 1..9 {
            acc -= a[row][c] * x[c];
        }
        x[row] = acc / a[row][row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, t: f64, what: &str) {
        assert!((a - b).abs() < t, "{what}: {a} vs {b}");
    }

    #[test]
    fn rodrigues_round_trips() {
        for v in [
            [0.1, 0.2, 0.3],
            [0.0, 0.0, core::f64::consts::FRAC_PI_2],
            [1.0, -0.5, 0.25],
            [0.0, 0.0, 0.0],
        ] {
            let r = rodrigues_to_matrix(v);
            let back = rodrigues_to_vector(&r);
            for k in 0..3 {
                close(back[k], v[k], 1e-9, "rodrigues round trip");
            }
        }
    }

    #[test]
    fn rodrigues_of_zero_is_the_identity() {
        let r = rodrigues_to_matrix([0.0, 0.0, 0.0]);
        assert_eq!(r, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn rodrigues_quarter_turn_about_z() {
        let r = rodrigues_to_matrix([0.0, 0.0, core::f64::consts::FRAC_PI_2]);
        // (1,0,0) -> (0,1,0).
        close(r[0][0], 0.0, 1e-12, "r00");
        close(r[1][0], 1.0, 1e-12, "r10");
        close(r[0][1], -1.0, 1e-12, "r01");
    }

    #[test]
    fn rodrigues_handles_a_half_turn() {
        // A pi rotation has zero antisymmetric part, the degenerate branch.
        let v = [0.0, 0.0, core::f64::consts::PI];
        let r = rodrigues_to_matrix(v);
        let back = rodrigues_to_vector(&r);
        let n = (back[0].powi(2) + back[1].powi(2) + back[2].powi(2)).sqrt();
        close(n, core::f64::consts::PI, 1e-9, "half-turn angle");
        close(back[2].abs(), core::f64::consts::PI, 1e-9, "half-turn axis is z");
    }

    #[test]
    fn rotation_matrices_are_orthonormal() {
        let r = rodrigues_to_matrix([0.3, -0.7, 1.1]);
        for i in 0..3 {
            let n: f64 = (0..3).map(|k| r[i][k] * r[i][k]).sum();
            close(n, 1.0, 1e-12, "row norm");
            for j in i + 1..3 {
                let d: f64 = (0..3).map(|k| r[i][k] * r[j][k]).sum();
                close(d, 0.0, 1e-12, "row orthogonality");
            }
        }
    }

    #[test]
    fn homography_recovers_a_known_transform() {
        // A pure translation plus scale.
        let h_true = [[2.0, 0.0, 5.0], [0.0, 3.0, -7.0], [0.0, 0.0, 1.0]];
        let src = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (4.0, 6.0)];
        let dst: Vec<(f64, f64)> = src.iter().map(|&p| apply_homography(&h_true, p)).collect();
        let h = find_homography(&src, &dst).unwrap();
        for p in &src {
            let a = apply_homography(&h, *p);
            let b = apply_homography(&h_true, *p);
            close(a.0, b.0, 1e-6, "x");
            close(a.1, b.1, 1e-6, "y");
        }
    }

    #[test]
    fn homography_recovers_a_projective_transform() {
        let h_true = [[1.1, 0.2, 3.0], [-0.15, 0.9, -2.0], [0.001, 0.002, 1.0]];
        let src = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 80.0),
            (0.0, 80.0),
            (37.0, 22.0),
            (61.0, 70.0),
        ];
        let dst: Vec<(f64, f64)> = src.iter().map(|&p| apply_homography(&h_true, p)).collect();
        let h = find_homography(&src, &dst).unwrap();
        for p in &src {
            let a = apply_homography(&h, *p);
            let b = apply_homography(&h_true, *p);
            close(a.0, b.0, 1e-6, "x");
            close(a.1, b.1, 1e-6, "y");
        }
    }

    #[test]
    fn homography_needs_four_points() {
        let p = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
        assert!(find_homography(&p, &p).is_err());
    }

    #[test]
    fn ransac_ignores_gross_outliers() {
        let h_true = [[1.0, 0.0, 20.0], [0.0, 1.0, -10.0], [0.0, 0.0, 1.0]];
        let src: Vec<(f64, f64)> =
            (0..20).map(|i| ((i % 5) as f64 * 20.0, (i / 5) as f64 * 20.0)).collect();
        let mut dst: Vec<(f64, f64)> =
            src.iter().map(|&p| apply_homography(&h_true, p)).collect();
        // Corrupt a quarter of the correspondences.
        for i in (0..20).step_by(4) {
            dst[i] = (dst[i].0 + 500.0, dst[i].1 - 300.0);
        }
        let mut rng = cv_core::rng::Rng::new(4242);
        let (h, mask) = find_homography_ransac(&src, &dst, 2.0, 500, &mut rng).unwrap();
        assert_eq!(mask.iter().filter(|&&m| m).count(), 15, "the 5 outliers must be rejected");
        for i in (0..20).step_by(4) {
            assert!(!mask[i], "outlier {i} was accepted");
        }
        close(h[0][2], 20.0, 1e-6, "recovered tx");
        close(h[1][2], -10.0, 1e-6, "recovered ty");
    }

    #[test]
    fn project_points_of_a_pinhole_camera() {
        let k = camera_matrix(500.0, 500.0, 320.0, 240.0);
        // A point one unit ahead lands on the principal point.
        let p = project_points(&[[0.0, 0.0, 1.0]], [0.0; 3], [0.0; 3], &k, [0.0; 5]);
        close(p[0].0, 320.0, 1e-9, "cx");
        close(p[0].1, 240.0, 1e-9, "cy");
        // One unit right at one unit depth moves by fx.
        let q = project_points(&[[1.0, 0.0, 1.0]], [0.0; 3], [0.0; 3], &k, [0.0; 5]);
        close(q[0].0, 820.0, 1e-9, "cx + fx");
    }

    #[test]
    fn distortion_pushes_points_and_undistort_pulls_them_back() {
        let k = camera_matrix(500.0, 500.0, 320.0, 240.0);
        let dist = [-0.2, 0.05, 0.001, -0.002, 0.0];
        let obj = [[0.3, 0.2, 1.0], [-0.4, 0.35, 1.2]];
        let distorted = project_points(&obj, [0.0; 3], [0.0; 3], &k, dist);
        let undist = undistort_points(&distorted, &k, dist);
        for (u, o) in undist.iter().zip(obj.iter()) {
            close(u.0, o[0] / o[2], 1e-6, "undistorted x");
            close(u.1, o[1] / o[2], 1e-6, "undistorted y");
        }
    }

    #[test]
    fn solve_pnp_recovers_a_known_pose() {
        let k = camera_matrix(600.0, 600.0, 320.0, 240.0);
        let rvec = [0.15, -0.25, 0.35];
        let tvec = [0.4, -0.2, 6.0];
        // A non-coplanar object.
        let obj = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let img = project_points(&obj, rvec, tvec, &k, [0.0; 5]);
        let pose = solve_pnp(&obj, &img, &k).unwrap();
        for i in 0..3 {
            close(pose.rvec[i], rvec[i], 1e-5, "rvec");
            close(pose.tvec[i], tvec[i], 1e-4, "tvec");
        }
        // And it reprojects onto the same pixels.
        let back = project_points(&obj, pose.rvec, pose.tvec, &k, [0.0; 5]);
        for (a, b) in back.iter().zip(img.iter()) {
            close(a.0, b.0, 1e-3, "reprojected x");
            close(a.1, b.1, 1e-3, "reprojected y");
        }
    }

    #[test]
    fn solve_pnp_needs_six_points() {
        let k = camera_matrix(1.0, 1.0, 0.0, 0.0);
        let obj = [[0.0, 0.0, 1.0]; 5];
        let img = [(0.0, 0.0); 5];
        assert!(solve_pnp(&obj, &img, &k).is_err());
    }

    #[test]
    fn fundamental_matrix_satisfies_the_epipolar_constraint() {
        // Two cameras looking at a rigid cloud.
        let k = camera_matrix(500.0, 500.0, 320.0, 240.0);
        let obj: Vec<[f64; 3]> = (0..12)
            .map(|i| {
                let f = i as f64;
                [(f % 4.0) - 1.5, (f / 4.0).floor() - 1.0, 5.0 + (f * 0.37) % 2.0]
            })
            .collect();
        let x1 = project_points(&obj, [0.0; 3], [0.0; 3], &k, [0.0; 5]);
        let x2 = project_points(&obj, [0.02, 0.05, 0.01], [0.5, 0.1, 0.0], &k, [0.0; 5]);
        let f = find_fundamental_mat(&x1, &x2).unwrap();
        for (a, b) in x1.iter().zip(x2.iter()) {
            // x2^T F x1 should vanish.
            let fx = [
                f[0][0] * a.0 + f[0][1] * a.1 + f[0][2],
                f[1][0] * a.0 + f[1][1] * a.1 + f[1][2],
                f[2][0] * a.0 + f[2][1] * a.1 + f[2][2],
            ];
            let e = b.0 * fx[0] + b.1 * fx[1] + fx[2];
            assert!(e.abs() < 1e-6, "epipolar residual {e}");
        }
    }

    #[test]
    fn fundamental_matrix_is_rank_two() {
        let k = camera_matrix(500.0, 500.0, 320.0, 240.0);
        let obj: Vec<[f64; 3]> = (0..10)
            .map(|i| [(i % 3) as f64, (i / 3) as f64, 4.0 + (i as f64) * 0.3])
            .collect();
        let x1 = project_points(&obj, [0.0; 3], [0.0; 3], &k, [0.0; 5]);
        let x2 = project_points(&obj, [0.01, 0.03, 0.0], [0.4, 0.0, 0.1], &k, [0.0; 5]);
        let f = find_fundamental_mat(&x1, &x2).unwrap();
        let det = f[0][0] * (f[1][1] * f[2][2] - f[1][2] * f[2][1])
            - f[0][1] * (f[1][0] * f[2][2] - f[1][2] * f[2][0])
            + f[0][2] * (f[1][0] * f[2][1] - f[1][1] * f[2][0]);
        assert!(det.abs() < 1e-9, "a fundamental matrix must be singular, det = {det}");
    }

    #[test]
    fn triangulation_recovers_the_original_points() {
        let k = camera_matrix(500.0, 500.0, 320.0, 240.0);
        let obj = [[0.2, -0.1, 5.0], [-0.4, 0.3, 6.0], [1.0, 0.5, 4.5]];
        // Camera 1 at the origin, camera 2 shifted along x.
        let mut p1 = [[0.0f64; 4]; 3];
        let mut p2 = [[0.0f64; 4]; 3];
        for i in 0..3 {
            for j in 0..3 {
                p1[i][j] = k[i][j];
                p2[i][j] = k[i][j];
            }
        }
        let baseline = -1.0;
        for i in 0..3 {
            p2[i][3] = k[i][0] * baseline;
        }
        let x1: Vec<(f64, f64)> = obj
            .iter()
            .map(|p| {
                let z = k[2][0] * p[0] + k[2][1] * p[1] + k[2][2] * p[2];
                (
                    (k[0][0] * p[0] + k[0][1] * p[1] + k[0][2] * p[2]) / z,
                    (k[1][0] * p[0] + k[1][1] * p[1] + k[1][2] * p[2]) / z,
                )
            })
            .collect();
        let x2: Vec<(f64, f64)> = obj
            .iter()
            .map(|p| {
                let q = [p[0] + baseline, p[1], p[2]];
                let z = k[2][0] * q[0] + k[2][1] * q[1] + k[2][2] * q[2];
                (
                    (k[0][0] * q[0] + k[0][1] * q[1] + k[0][2] * q[2]) / z,
                    (k[1][0] * q[0] + k[1][1] * q[1] + k[1][2] * q[2]) / z,
                )
            })
            .collect();
        let out = triangulate_points(&p1, &p2, &x1, &x2).unwrap();
        for (got, want) in out.iter().zip(obj.iter()) {
            let g = got.expect("finite point");
            for k in 0..3 {
                close(g[k], want[k], 1e-6, "triangulated");
            }
        }
    }

    #[test]
    fn mat3_inverse_round_trips() {
        let m = [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]];
        let i = mat3_inv(&m).unwrap();
        let p = mat3_mul(&m, &i);
        for (r, prow) in p.iter().enumerate() {
            for (c, &v) in prow.iter().enumerate() {
                close(v, if r == c { 1.0 } else { 0.0 }, 1e-12, "M M^-1");
            }
        }
        assert!(mat3_inv(&[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]).is_none());
    }

    #[test]
    fn mat_conversion_round_trips() {
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let as_mat = to_mat(&m).unwrap();
        assert_eq!(mat_to_mat3(&as_mat).unwrap(), m);
        assert_eq!(mat_to_mat3(&mat3_to_mat(&m).unwrap()).unwrap(), m);
    }
}
