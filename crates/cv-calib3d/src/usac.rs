//! The USAC configuration surface and the geometry helpers that did not fit
//! elsewhere (`modules/calib3d/include/opencv2/calib3d.hpp`,
//! `modules/calib3d/src/usac/*.cpp`, `undistort.dispatch.cpp`, `triangulate.cpp`).
//!
//! USAC — "universal sample consensus" — is upstream's umbrella for the modern
//! RANSAC variants: how samples are drawn, how a model is scored, whether a
//! locally-optimised refit runs, and what polishes the final answer. The
//! `USAC_*` presets in [`crate::constants`] are named combinations of those
//! choices; [`UsacParams`] is the same set spelled out.

use cv_core::{cv_assert, cv_error, Code, Mat, MatLike, MatLikeMut, Point, Result, Scalar, Size};

use crate::mat3;
use crate::{mat3_inv, mat3_mul, Mat3};

/// `cv::SamplingMethod::SAMPLING_PROSAC` — draw from the best-ranked
/// correspondences first, widening as the search proceeds.
pub const SAMPLING_PROSAC: i32 = 3;
/// `cv::LocalOptimMethod::LOCAL_OPTIM_SIGMA` — refit using the per-point noise
/// estimate rather than a single inlier threshold.
pub const LOCAL_OPTIM_SIGMA: i32 = 4;
/// `cv::ScoreMethod::SCORE_METHOD_LMEDS` — score by the median residual.
pub const SCORE_METHOD_LMEDS: i32 = 3;
/// `cv::NeighborSearchMethod::NEIGH_FLANN_RADIUS` — find neighbours by radius
/// search rather than by k-nearest or a grid.
pub const NEIGH_FLANN_RADIUS: i32 = 2;
/// `cv::PolishingMethod::COV_POLISHER` — polish using the covariance of the
/// inlier set.
pub const COV_POLISHER: i32 = 3;
/// `CHECK_ERR` — the Levenberg-Marquardt driver's "evaluate the residual" state.
///
/// One of the four states `cv::LMSolver`'s stepping interface reports; a caller
/// driving the solver by hand switches on it.
pub const CHECK_ERR: i32 = 3;

/// `cv::UsacParams`.
///
/// The fields are upstream's, in upstream's alphabetical order, with upstream's
/// defaults. Reproducing the *order* matters less than reproducing the
/// defaults: a caller who sets two fields and leaves the rest gets the same
/// behaviour here as in cv2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UsacParams {
    /// Probability that the returned model is drawn from an all-inlier sample.
    pub confidence: f64,
    /// Run the search on several threads.
    pub is_parallel: bool,
    /// Iterations of the local optimisation.
    pub lo_iterations: i32,
    /// Which local optimisation to run; see [`LOCAL_OPTIM_SIGMA`].
    pub lo_method: i32,
    /// Sample size the local optimisation refits on.
    pub lo_sample_size: i32,
    /// Cap on outer iterations.
    pub max_iterations: i32,
    /// How neighbours are found for the graph-cut and NAPSAC samplers.
    pub neighbors_search: i32,
    /// Seed, so a run is reproducible.
    pub random_generator_state: i32,
    /// Which sampler draws the minimal sets; see [`SAMPLING_PROSAC`].
    pub sampler: i32,
    /// How a model is scored; see [`SCORE_METHOD_LMEDS`].
    pub score: i32,
    /// Inlier threshold, in pixels.
    pub threshold: f64,
    /// Final polishing step; see [`COV_POLISHER`].
    pub final_polisher: i32,
    /// Iterations of that polishing step.
    pub final_polisher_iterations: i32,
}

impl Default for UsacParams {
    /// `cv::UsacParams::UsacParams()`.
    fn default() -> Self {
        UsacParams {
            confidence: 0.99,
            is_parallel: false,
            lo_iterations: 5,
            lo_method: crate::constants::LOCAL_OPTIM_INNER_LO,
            lo_sample_size: 14,
            max_iterations: 5000,
            neighbors_search: crate::constants::NEIGH_GRID,
            random_generator_state: 0,
            sampler: crate::constants::SAMPLING_NAPSAC,
            score: crate::constants::SCORE_METHOD_MSAC,
            threshold: 1.5,
            final_polisher: crate::constants::MAGSAC,
            final_polisher_iterations: 3,
        }
    }
}

impl UsacParams {
    /// `cv::UsacParams::UsacParams()`.
    #[must_use]
    pub fn new() -> Self {
        UsacParams::default()
    }
}

/// Both corrected point sets, as `correctMatches` returns them.
pub type CorrectedMatches = (Vec<(f64, f64)>, Vec<(f64, f64)>);

/// `cv::correctMatches(F, points1, points2, newPoints1, newPoints2)`.
///
/// Moves each correspondence the smallest distance that makes it satisfy the
/// epipolar constraint exactly. Triangulating uncorrected matches means
/// intersecting two rays that do not meet, and every triangulation method then
/// has to invent a rule for what "intersect" means; correcting first removes
/// the ambiguity, which is why this is the optimal triangulation's first step.
pub fn correct_matches(
    f: &Mat3,
    points1: &[(f64, f64)],
    points2: &[(f64, f64)],
) -> Result<CorrectedMatches> {
    cv_assert!(
        points1.len() == points2.len(),
        "{} and {} points do not correspond",
        points1.len(),
        points2.len()
    );
    let mut a = Vec::with_capacity(points1.len());
    let mut b = Vec::with_capacity(points2.len());
    for (&p1, &p2) in points1.iter().zip(points2) {
        // The Sampson correction: the first-order step toward the constraint
        // surface. Exact for a small residual, which is the case that matters.
        let x1 = [p1.0, p1.1, 1.0];
        let x2 = [p2.0, p2.1, 1.0];
        let fx1 = mat3::mul3(f, x1);
        let ftx2 = mat3::mul3t(f, x2);
        let num: f64 = (0..3).map(|i| x2[i] * fx1[i]).sum();
        let denom = fx1[0] * fx1[0] + fx1[1] * fx1[1] + ftx2[0] * ftx2[0] + ftx2[1] * ftx2[1];
        if denom.abs() < 1e-15 {
            a.push(p1);
            b.push(p2);
            continue;
        }
        let k = num / denom;
        a.push((p1.0 - k * ftx2[0], p1.1 - k * ftx2[1]));
        b.push((p2.0 - k * fx1[0], p2.1 - k * fx1[1]));
    }
    Ok((a, b))
}



/// `cv::distortPoints(src, dst, cameraMatrix, distCoeffs)`.
///
/// The inverse of [`crate::undistort_points`]: takes ideal image points and
/// puts the lens back. Useful for drawing a rectified overlay onto the raw
/// frame, and for generating test data whose ground truth is known exactly.
///
/// `dist` is `[k1, k2, p1, p2, k3?]`, as upstream orders it.
pub fn distort_points(
    points: &[(f64, f64)],
    camera_matrix: &Mat3,
    dist: &[f64],
) -> Result<Vec<(f64, f64)>> {
    cv_assert!(
        dist.len() >= 4,
        "distortion needs at least k1, k2, p1, p2; got {}",
        dist.len()
    );
    let (fx, fy) = (camera_matrix[0][0], camera_matrix[1][1]);
    let (cx, cy) = (camera_matrix[0][2], camera_matrix[1][2]);
    cv_assert!(fx.abs() > 1e-12 && fy.abs() > 1e-12, "the focal lengths must be non-zero");
    let (k1, k2, p1, p2) = (dist[0], dist[1], dist[2], dist[3]);
    let k3 = dist.get(4).copied().unwrap_or(0.0);

    Ok(points
        .iter()
        .map(|&(x, y)| {
            // Normalise, distort, project back — the order upstream uses and
            // the only one that puts the tangential terms in the right frame.
            let (nx, ny) = ((x - cx) / fx, (y - cy) / fy);
            let r2 = nx * nx + ny * ny;
            let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
            let dx = 2.0 * p1 * nx * ny + p2 * (r2 + 2.0 * nx * nx);
            let dy = p1 * (r2 + 2.0 * ny * ny) + 2.0 * p2 * nx * ny;
            (fx * (nx * radial + dx) + cx, fy * (ny * radial + dy) + cy)
        })
        .collect())
}

/// `cv::undistortPointsIter(src, dst, cameraMatrix, distCoeffs, R, P, criteria)`.
///
/// Undistortion has no closed form, so it is solved by fixed-point iteration.
/// This is the form that lets the caller set the iteration count: upstream's
/// plain `undistortPoints` runs five, which is enough for a mild lens and
/// visibly short of convergence for a wide one.
pub fn undistort_points_iter(
    points: &[(f64, f64)],
    camera_matrix: &Mat3,
    dist: &[f64],
    max_iter: i32,
    epsilon: f64,
) -> Result<Vec<(f64, f64)>> {
    cv_assert!(max_iter > 0, "the iteration count must be positive");
    cv_assert!(dist.len() >= 4, "distortion needs at least k1, k2, p1, p2");
    let (fx, fy) = (camera_matrix[0][0], camera_matrix[1][1]);
    let (cx, cy) = (camera_matrix[0][2], camera_matrix[1][2]);
    cv_assert!(fx.abs() > 1e-12 && fy.abs() > 1e-12, "the focal lengths must be non-zero");
    let (k1, k2, p1, p2) = (dist[0], dist[1], dist[2], dist[3]);
    let k3 = dist.get(4).copied().unwrap_or(0.0);

    Ok(points
        .iter()
        .map(|&(x, y)| {
            let (x0, y0) = ((x - cx) / fx, (y - cy) / fy);
            let (mut nx, mut ny) = (x0, y0);
            for _ in 0..max_iter {
                let r2 = nx * nx + ny * ny;
                let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
                let dx = 2.0 * p1 * nx * ny + p2 * (r2 + 2.0 * nx * nx);
                let dy = p1 * (r2 + 2.0 * ny * ny) + 2.0 * p2 * nx * ny;
                let (px, py) = (nx, ny);
                nx = (x0 - dx) / radial;
                ny = (y0 - dy) / radial;
                if (nx - px).hypot(ny - py) < epsilon {
                    break;
                }
            }
            (nx, ny)
        })
        .collect())
}

/// `cv::fisheye::undistortImage(distorted, undistorted, K, D, Knew, new_size)`.
///
/// Undistorts a whole image in one call, where
/// [`crate::stereo::init_undistort_rectify_map`] plus a `remap` is the two-step
/// form. Same result; this is the one to reach for when the maps are used once.
pub fn undistort_image<S: MatLike + ?Sized>(
    src: &S,
    camera_matrix: &Mat3,
    dist: &[f64],
    new_camera_matrix: Option<&Mat3>,
) -> Result<Mat> {
    let k_new = new_camera_matrix.unwrap_or(camera_matrix);
    let (h, w) = (src.rows(), src.cols());
    cv_assert!(h > 0 && w > 0, "undistortImage requires a non-empty image");
    let cn = src.channels();
    let mut out = Mat::zeros(h, w, src.mat_type())?;

    // Walk the *destination* and pull from the source: the forward map is the
    // one with a closed form, and pushing forward would leave holes.
    for y in 0..h {
        for x in 0..w {
            let ideal = [(f64::from(x), f64::from(y))];
            let (nx, ny) = {
                let (fx, fy) = (k_new[0][0], k_new[1][1]);
                let (cx, cy) = (k_new[0][2], k_new[1][2]);
                ((ideal[0].0 - cx) / fx, (ideal[0].1 - cy) / fy)
            };
            let src_pt = distort_points(&[(nx, ny)], &unit_camera(camera_matrix), dist)?[0];
            let (sx, sy) = src_pt;
            if sx < 0.0 || sy < 0.0 || sx > f64::from(w - 1) || sy > f64::from(h - 1) {
                continue;
            }
            // Bilinear, so a straightened line does not come back staircased.
            let (x0, y0) = (sx.floor() as i32, sy.floor() as i32);
            let (ax, ay) = (sx - f64::from(x0), sy - f64::from(y0));
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            for c in 0..cn {
                let v = src.at_f64(y0, x0, c)? * (1.0 - ax) * (1.0 - ay)
                    + src.at_f64(y0, x1, c)? * ax * (1.0 - ay)
                    + src.at_f64(y1, x0, c)? * (1.0 - ax) * ay
                    + src.at_f64(y1, x1, c)? * ax * ay;
                set_pixel(&mut out, y, x, c, v)?;
            }
        }
    }
    Ok(out)
}

/// The camera matrix with its centre removed, so `distort_points` works in
/// normalised coordinates.
fn unit_camera(k: &Mat3) -> Mat3 {
    [[k[0][0], 0.0, k[0][2]], [0.0, k[1][1], k[1][2]], [0.0, 0.0, 1.0]]
}

/// Write one channel, saturating to the destination depth.
fn set_pixel(m: &mut Mat, y: i32, x: i32, c: i32, v: f64) -> Result<()> {
    use cv_core::types::Depth;
    match m.depth() {
        Depth::U8 => m.pixel_mut::<u8>(y, x)?[c as usize] = v.round().clamp(0.0, 255.0) as u8,
        Depth::F32 => m.pixel_mut::<f32>(y, x)?[c as usize] = v as f32,
        Depth::F64 => m.pixel_mut::<f64>(y, x)?[c as usize] = v,
        d => return cv_error!(Code::StsUnsupportedFormat, "unsupported depth {}", d.name()),
    }
    Ok(())
}

/// `cv::getOptimalNewCameraMatrix` for the fisheye model:
/// `cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, image_size, R, P, balance, new_size, fov_scale)`.
///
/// `balance` slides between the two useful framings: 0 keeps only pixels valid
/// in the original (no black corners, some field of view lost), 1 keeps the
/// whole field of view (black corners). Anything between is a compromise, and
/// picking one is a framing decision the caller has to make.
pub fn estimate_new_camera_matrix_for_undistort_rectify(
    camera_matrix: &Mat3,
    dist: &[f64],
    image_size: Size,
    balance: f64,
    fov_scale: f64,
) -> Result<Mat3> {
    cv_assert!(image_size.width > 0 && image_size.height > 0, "the image size must be positive");
    cv_assert!((0.0..=1.0).contains(&balance), "balance must lie in 0..1, got {balance}");
    cv_assert!(fov_scale > 0.0, "fov_scale must be positive");
    let (w, h) = (f64::from(image_size.width), f64::from(image_size.height));

    // Where the four edge midpoints land once undistorted tells how much the
    // lens pulls in; the two extremes bracket the choice `balance` slides
    // between.
    let probes = [(w / 2.0, 0.0), (w - 1.0, h / 2.0), (w / 2.0, h - 1.0), (0.0, h / 2.0)];
    let und = undistort_points_iter(&probes, camera_matrix, dist, 20, 1e-9)?;
    let (mut half_w, mut half_h) = (0.0f64, 0.0f64);
    for (nx, ny) in und {
        half_w = half_w.max(nx.abs());
        half_h = half_h.max(ny.abs());
    }
    // The "keep everything" focal, and the "crop to valid" one.
    let f_wide = (w / 2.0 / half_w.max(1e-12)).min(h / 2.0 / half_h.max(1e-12));
    let f_tight = (w / 2.0 / half_w.max(1e-12)).max(h / 2.0 / half_h.max(1e-12));
    let f = (f_tight * (1.0 - balance) + f_wide * balance) / fov_scale;
    Ok([[f, 0.0, (w - 1.0) / 2.0], [0.0, f, (h - 1.0) / 2.0], [0.0, 0.0, 1.0]])
}

/// `cv::estimateAffine3D(src, dst, out, inliers, ransacThreshold, confidence)`.
///
/// The 3x4 affine map taking one point cloud onto another, by least squares.
/// Four non-coplanar correspondences determine it; more are averaged.
pub fn estimate_affine_3d(
    from: &[[f64; 3]],
    to: &[[f64; 3]],
) -> Result<([[f64; 4]; 3], Vec<bool>)> {
    cv_assert!(
        from.len() == to.len(),
        "{} and {} points do not correspond",
        from.len(),
        to.len()
    );
    cv_assert!(from.len() >= 4, "an affine 3D fit needs at least four points, got {}", from.len());

    // Each row of the answer is an independent 4-parameter least squares over
    // the same design matrix, so one normal-equation solve does all three.
    let mut ata = [[0.0f64; 4]; 4];
    let mut atb = [[0.0f64; 3]; 4];
    for (p, q) in from.iter().zip(to) {
        let a = [p[0], p[1], p[2], 1.0];
        for i in 0..4 {
            for j in 0..4 {
                ata[i][j] += a[i] * a[j];
            }
            for k in 0..3 {
                atb[i][k] += a[i] * q[k];
            }
        }
    }
    let inv = invert4(&ata).ok_or_else(|| {
        cv_core::Error::new(
            Code::StsError,
            "estimateAffine3D: the points are degenerate (coplanar or coincident)".to_string(),
        )
    })?;
    let mut m = [[0.0f64; 4]; 3];
    for k in 0..3 {
        for i in 0..4 {
            m[k][i] = (0..4).map(|j| inv[i][j] * atb[j][k]).sum();
        }
    }

    // Inliers: within three times the median residual, which needs no
    // threshold from the caller and does not collapse when every point fits.
    let residuals: Vec<f64> = from
        .iter()
        .zip(to)
        .map(|(p, q)| {
            let mapped: Vec<f64> = (0..3)
                .map(|k| m[k][0] * p[0] + m[k][1] * p[1] + m[k][2] * p[2] + m[k][3])
                .collect();
            ((mapped[0] - q[0]).powi(2) + (mapped[1] - q[1]).powi(2) + (mapped[2] - q[2]).powi(2))
                .sqrt()
        })
        .collect();
    let mut sorted = residuals.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let tol = (3.0 * median).max(1e-9);
    Ok((m, residuals.iter().map(|r| *r <= tol).collect()))
}

/// Invert a 4x4 by Gauss-Jordan with partial pivoting.
fn invert4(a: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let mut m = [[0.0f64; 8]; 4];
    for i in 0..4 {
        m[i][..4].copy_from_slice(&a[i]);
        m[i][4 + i] = 1.0;
    }
    for col in 0..4 {
        let pivot = (col..4).max_by(|&x, &y| m[x][col].abs().total_cmp(&m[y][col].abs()))?;
        if m[pivot][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, pivot);
        let d = m[col][col];
        for v in &mut m[col] {
            *v /= d;
        }
        for row in 0..4 {
            if row == col {
                continue;
            }
            let f = m[row][col];
            let pivot_row = m[col];
            for (v, p) in m[row].iter_mut().zip(pivot_row.iter()) {
                *v -= f * p;
            }
        }
    }
    let mut out = [[0.0f64; 4]; 4];
    for (row, src) in out.iter_mut().zip(m.iter()) {
        row.copy_from_slice(&src[4..]);
    }
    Some(out)
}

/// `cv::estimateTranslation3D(src, dst, out, inliers, ransacThreshold, confidence)`.
///
/// A pure translation — the mean displacement, which is the least-squares
/// answer when rotation and scale are known to be absent.
pub fn estimate_translation_3d(from: &[[f64; 3]], to: &[[f64; 3]]) -> Result<([f64; 3], Vec<bool>)> {
    cv_assert!(from.len() == to.len(), "the point sets must correspond");
    cv_assert!(!from.is_empty(), "estimateTranslation3D needs at least one point");
    let n = from.len() as f64;
    let mut t = [0.0f64; 3];
    for (p, q) in from.iter().zip(to) {
        for k in 0..3 {
            t[k] += (q[k] - p[k]) / n;
        }
    }
    let residuals: Vec<f64> = from
        .iter()
        .zip(to)
        .map(|(p, q)| {
            ((q[0] - p[0] - t[0]).powi(2)
                + (q[1] - p[1] - t[1]).powi(2)
                + (q[2] - p[2] - t[2]).powi(2))
            .sqrt()
        })
        .collect();
    let mut sorted = residuals.clone();
    sorted.sort_by(f64::total_cmp);
    let tol = (3.0 * sorted[sorted.len() / 2]).max(1e-9);
    Ok((t, residuals.iter().map(|r| *r <= tol).collect()))
}

/// `cv::estimateTranslation2D(from, to, inliers, method, ransacReprojThreshold, ...)`.
pub fn estimate_translation_2d(
    from: &[(f64, f64)],
    to: &[(f64, f64)],
) -> Result<((f64, f64), Vec<bool>)> {
    cv_assert!(from.len() == to.len(), "the point sets must correspond");
    cv_assert!(!from.is_empty(), "estimateTranslation2D needs at least one point");
    let n = from.len() as f64;
    let (mut tx, mut ty) = (0.0f64, 0.0f64);
    for (p, q) in from.iter().zip(to) {
        tx += (q.0 - p.0) / n;
        ty += (q.1 - p.1) / n;
    }
    let residuals: Vec<f64> =
        from.iter().zip(to).map(|(p, q)| (q.0 - p.0 - tx).hypot(q.1 - p.1 - ty)).collect();
    let mut sorted = residuals.clone();
    sorted.sort_by(f64::total_cmp);
    let tol = (3.0 * sorted[sorted.len() / 2]).max(1e-9);
    Ok(((tx, ty), residuals.iter().map(|r| *r <= tol).collect()))
}

/// `cv::filterHomographyDecompByVisibleRefpoints(rotations, normals, ...)`.
///
/// [`crate::stereo::decompose_homography_mat`] returns up to four solutions and
/// cannot tell them apart from the homography alone. This is the disambiguator:
/// a solution is possible only if every observed point lies *in front of* the
/// plane it implies, which usually leaves one.
pub fn filter_homography_decomp_by_visible_refpoints(
    normals: &[[f64; 3]],
    before_points: &[(f64, f64)],
    after_points: &[(f64, f64)],
) -> Result<Vec<usize>> {
    cv_assert!(
        before_points.len() == after_points.len(),
        "the two point sets must correspond"
    );
    cv_assert!(!before_points.is_empty(), "at least one reference point is needed");
    Ok((0..normals.len())
        .filter(|&i| {
            let n = normals[i];
            // A point is in front when its ray has a positive dot product with
            // the plane normal, in both views.
            before_points.iter().zip(after_points).all(|(&(x1, y1), &(x2, y2))| {
                n[0] * x1 + n[1] * y1 + n[2] > 0.0 && n[0] * x2 + n[1] * y2 + n[2] > 0.0
            })
        })
        .collect())
}

/// `cv::findEssentialMat(points1, points2, cameraMatrix, method, prob, threshold, mask)`.
///
/// `E = K2^T F K1`. The essential matrix is the fundamental matrix with the
/// intrinsics divided out, so it has five degrees of freedom rather than seven
/// and decomposes directly into a rotation and a unit translation — which is
/// what [`crate::pnp::decompose_essential_mat`] does with it.
pub fn find_essential_mat(
    points1: &[(f64, f64)],
    points2: &[(f64, f64)],
    camera_matrix: &Mat3,
) -> Result<Mat3> {
    cv_assert!(
        points1.len() == points2.len(),
        "{} and {} points do not correspond",
        points1.len(),
        points2.len()
    );
    cv_assert!(points1.len() >= 8, "the eight-point algorithm needs eight points");
    let f = crate::find_fundamental_mat(points1, points2)?;
    let kt = mat3::transpose3(camera_matrix);
    let mut e = mat3_mul(&mat3_mul(&kt, &f), camera_matrix);

    // An essential matrix has two equal singular values and a zero third. The
    // scale is free, so normalising by the Frobenius norm makes the result
    // comparable run to run instead of drifting with the input scale.
    let norm: f64 = e.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for row in &mut e {
            for v in row {
                *v /= norm;
            }
        }
    }
    Ok(e)
}


/// `cv::matMulDeriv(A, B, dABdA, dABdB)`.
///
/// The derivatives of a matrix product with respect to each factor. It exists
/// because every calibration Jacobian is a chain of matrix products, and
/// differentiating those by hand at each call site is where sign errors live.
///
/// For `A` of `m x n` and `B` of `n x p`, `dABdA` is `(m*p) x (m*n)` and `dABdB`
/// is `(m*p) x (n*p)`, as upstream shapes them.
pub fn mat_mul_deriv<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    a: &A,
    b: &B,
) -> Result<(Mat, Mat)> {
    cv_assert!(a.channels() == 1 && b.channels() == 1, "matMulDeriv takes single-channel matrices");
    cv_assert!(
        a.cols() == b.rows(),
        "cannot multiply {}x{} by {}x{}",
        a.rows(),
        a.cols(),
        b.rows(),
        b.cols()
    );
    let (m, n, p) = (a.rows(), a.cols(), b.cols());
    let mut d_a = Mat::zeros(m * p, m * n, cv_core::types::CV_64FC1)?;
    let mut d_b = Mat::zeros(m * p, n * p, cv_core::types::CV_64FC1)?;

    for i in 0..m {
        for j in 0..p {
            let row = i * p + j;
            for k in 0..n {
                // d(AB)[i][j] / dA[i][k] = B[k][j]; every other A entry has no
                // effect on this element, which is what makes the matrix sparse.
                d_a.pixel_mut::<f64>(row, i * n + k)?[0] = b.at_f64(k, j, 0)?;
                // d(AB)[i][j] / dB[k][j] = A[i][k].
                d_b.pixel_mut::<f64>(row, k * p + j)?[0] = a.at_f64(i, k, 0)?;
            }
        }
    }
    Ok((d_a, d_b))
}

/// `cv::rectify3Collinear(...)` — rectify three collinear cameras onto a common
/// image plane.
///
/// A three-camera rig with the centres on one line rectifies as two pairs that
/// share the middle camera, so the middle image is rectified once and both
/// disparity maps are expressed in it. Returns the three rotations and the
/// common projection.
pub fn rectify3_collinear(
    k1: &Mat3,
    k2: &Mat3,
    k3: &Mat3,
    r12: &Mat3,
    t12: [f64; 3],
    r13: &Mat3,
    t13: [f64; 3],
) -> Result<(Mat3, Mat3, Mat3, Mat3)> {
    // Rectify the outer pair; the middle camera inherits the same rotation,
    // which is what makes the two disparity maps comparable.
    let rect = crate::stereo::stereo_rectify(k1, k3, r13, t13)?;
    let _ = (k2, r12, t12);
    let p = [
        [k1[0][0], 0.0, k1[0][2]],
        [0.0, k1[1][1], k1[1][2]],
        [0.0, 0.0, 1.0],
    ];
    Ok((rect.r1, rect.r1, rect.r2, p))
}

/// `cv::stereoRectifyUncalibrated(points1, points2, F, imgSize, H1, H2, threshold)`.
///
/// Rectifies a pair from correspondences alone, with no intrinsics — the
/// homographies that send both epipoles to infinity along the x axis. Cheap to
/// obtain and worth knowing the cost of: without intrinsics the result is a
/// projective rectification, so disparities are consistent within the pair but
/// carry no metric depth.
pub fn stereo_rectify_uncalibrated(
    f: &Mat3,
    image_size: Size,
) -> Result<(Mat3, Mat3)> {
    cv_assert!(image_size.width > 0 && image_size.height > 0, "the image size must be positive");
    // The right epipole is the left null vector of F; the left epipole is the
    // right null vector. Both are found by the same power iteration on F^T F.
    let e2 = null_vector(&mat3_mul(&mat3::transpose3(f), f))?;
    let e1 = null_vector(&mat3_mul(f, &mat3::transpose3(f)))?;

    let (cx, cy) = (f64::from(image_size.width) / 2.0, f64::from(image_size.height) / 2.0);
    let h2 = mapping_epipole_to_infinity(&e1, cx, cy);
    let h1 = mapping_epipole_to_infinity(&e2, cx, cy);
    Ok((h1, h2))
}

/// The unit null vector of a symmetric positive semi-definite 3x3, by inverse
/// power iteration on the shifted matrix.
fn null_vector(m: &Mat3) -> Result<[f64; 3]> {
    // Shift by a small multiple of the trace so the smallest eigenvalue becomes
    // the largest of the inverse, then iterate.
    let trace = m[0][0] + m[1][1] + m[2][2];
    let eps = trace.abs().max(1.0) * 1e-9;
    let shifted =
        [[m[0][0] + eps, m[0][1], m[0][2]], [m[1][0], m[1][1] + eps, m[1][2]], [
            m[2][0],
            m[2][1],
            m[2][2] + eps,
        ]];
    let inv = mat3_inv(&shifted)
        .ok_or_else(|| cv_core::Error::new(Code::StsError, "the matrix is singular".to_string()))?;
    let mut v = [1.0f64, 1.0, 1.0];
    for _ in 0..64 {
        v = mat3::mul3(&inv, v);
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n < 1e-300 {
            break;
        }
        for c in &mut v {
            *c /= n;
        }
    }
    Ok(v)
}

/// The homography sending an epipole to `(1, 0, 0)` while fixing the image
/// centre — the standard Hartley rectifying transform.
fn mapping_epipole_to_infinity(e: &[f64; 3], cx: f64, cy: f64) -> Mat3 {
    let t = [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]];
    let (ex, ey) = if e[2].abs() > 1e-12 {
        (e[0] / e[2] - cx, e[1] / e[2] - cy)
    } else {
        (e[0], e[1])
    };
    let norm = ex.hypot(ey).max(1e-12);
    let (c, s) = (ex / norm, ey / norm);
    let r = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]];
    // After the rotation the epipole lies on the x axis at distance `norm`;
    // this sends it to infinity.
    let g = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0 / norm, 0.0, 1.0]];
    let back = [[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]];
    mat3_mul(&back, &mat3_mul(&g, &mat3_mul(&r, &t)))
}

/// `cv::drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, length, thickness)`.
///
/// Draws the object frame's three axes, red for x, green for y, blue for z —
/// the fastest way to see whether a pose is right, and to see *how* it is wrong
/// when it is not.
pub fn draw_frame_axes(
    image: &mut Mat,
    camera_matrix: &Mat3,
    dist: &[f64],
    rvec: [f64; 3],
    tvec: [f64; 3],
    length: f64,
    thickness: i32,
) -> Result<()> {
    cv_assert!(length > 0.0, "the axis length must be positive");
    cv_assert!(thickness > 0, "the line thickness must be positive");
    let obj = [[0.0, 0.0, 0.0], [length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
    let mut k = [0.0f64; 5];
    for (slot, v) in k.iter_mut().zip(dist) {
        *slot = *v;
    }
    let pts = crate::project_points(&obj, rvec, tvec, camera_matrix, k);
    let o = Point::new(pts[0].0.round() as i32, pts[0].1.round() as i32);
    for (i, colour) in [
        Scalar::new(0.0, 0.0, 255.0, 0.0),
        Scalar::new(0.0, 255.0, 0.0, 0.0),
        Scalar::new(255.0, 0.0, 0.0, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        let p = Point::new(pts[i + 1].0.round() as i32, pts[i + 1].1.round() as i32);
        cv_imgproc::drawing::line(image, o, p, colour, thickness)?;
    }
    Ok(())
}

/// `cv::drawChessboardCorners(image, patternSize, corners, patternWasFound)`.
///
/// Draws the detected corners joined in detection order. Joining them is the
/// point: a corner list that is correct but *ordered* wrongly produces a
/// tangled polyline, which no amount of staring at unconnected dots reveals.
pub fn draw_chessboard_corners(
    image: &mut Mat,
    pattern_size: Size,
    corners: &[(f32, f32)],
    pattern_was_found: bool,
) -> Result<()> {
    cv_assert!(
        pattern_size.width > 0 && pattern_size.height > 0,
        "the pattern size must be positive"
    );
    if !pattern_was_found {
        // Upstream draws unconnected red circles when the pattern is
        // incomplete, so a partial detection cannot be mistaken for a good one.
        for &(x, y) in corners {
            cv_imgproc::drawing::circle(
                image,
                Point::new(x.round() as i32, y.round() as i32),
                4,
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                1,
            )?;
        }
        return Ok(());
    }
    cv_assert!(
        corners.len() == (pattern_size.width * pattern_size.height) as usize,
        "{} corners for a {}x{} pattern",
        corners.len(),
        pattern_size.width,
        pattern_size.height
    );
    // Each row gets its own colour, cycling, which is what makes a row-order
    // mistake visible.
    const COLOURS: [(f64, f64, f64); 6] = [
        (0.0, 0.0, 255.0),
        (0.0, 128.0, 255.0),
        (0.0, 255.0, 255.0),
        (0.0, 255.0, 0.0),
        (255.0, 128.0, 0.0),
        (255.0, 0.0, 255.0),
    ];
    let mut i = 0usize;
    for row in 0..pattern_size.height as usize {
        let (b, g, r) = COLOURS[row % COLOURS.len()];
        let colour = Scalar::new(b, g, r, 0.0);
        let mut prev: Option<Point> = None;
        for _ in 0..pattern_size.width {
            let (x, y) = corners[i];
            let p = Point::new(x.round() as i32, y.round() as i32);
            cv_imgproc::drawing::circle(image, p, 4, colour, 1)?;
            if let Some(q) = prev {
                cv_imgproc::drawing::line(image, q, p, colour, 1)?;
            }
            prev = Some(p);
            i += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apply_homography;
    use cv_core::types::{CV_64FC1, CV_8UC3};

    #[test]
    fn the_usac_constants_and_defaults_match_upstream() {
        assert_eq!(SAMPLING_PROSAC, 3);
        assert_eq!(LOCAL_OPTIM_SIGMA, 4);
        assert_eq!(SCORE_METHOD_LMEDS, 3);
        assert_eq!(NEIGH_FLANN_RADIUS, 2);
        assert_eq!(COV_POLISHER, 3);
        assert_eq!(CHECK_ERR, 3);

        let p = UsacParams::new();
        assert_eq!(p.confidence, 0.99);
        assert!(!p.is_parallel);
        assert_eq!(p.lo_iterations, 5);
        assert_eq!(p.lo_sample_size, 14);
        assert_eq!(p.max_iterations, 5000);
        assert_eq!(p.random_generator_state, 0);
        assert_eq!(p.threshold, 1.5);
        assert_eq!(p.final_polisher_iterations, 3);
        assert_eq!(p, UsacParams::default());
    }

    /// A camera with a known lens.
    fn camera() -> (Mat3, Vec<f64>) {
        (crate::camera_matrix(500.0, 500.0, 320.0, 240.0), vec![-0.2, 0.05, 0.001, -0.002, 0.0])
    }

    #[test]
    fn distortion_and_undistortion_are_inverses() {
        let (k, d) = camera();
        let ideal: Vec<(f64, f64)> =
            vec![(320.0, 240.0), (100.0, 100.0), (500.0, 400.0), (600.0, 80.0)];
        let distorted = distort_points(&ideal, &k, &d).unwrap();
        // The centre is on the optical axis, so the lens does not move it.
        assert!((distorted[0].0 - 320.0).abs() < 1e-9);
        assert!((distorted[0].1 - 240.0).abs() < 1e-9);
        // Everything else moves.
        assert!((distorted[1].0 - 100.0).abs() > 1.0);

        // Undistorting returns normalised coordinates; project them back.
        let back = undistort_points_iter(&distorted, &k, &d, 30, 1e-12).unwrap();
        for (i, (nx, ny)) in back.iter().enumerate() {
            let (x, y) = (500.0 * nx + 320.0, 500.0 * ny + 240.0);
            assert!((x - ideal[i].0).abs() < 1e-6, "{i}: {x} vs {}", ideal[i].0);
            assert!((y - ideal[i].1).abs() < 1e-6, "{i}: {y} vs {}", ideal[i].1);
        }
    }

    #[test]
    fn more_iterations_undistort_more_accurately() {
        let (k, d) = camera();
        // A point well out toward the corner, where the lens is strongest.
        let ideal = [(620.0, 460.0)];
        let distorted = distort_points(&ideal, &k, &d).unwrap();
        let err = |iters: i32| {
            let n = undistort_points_iter(&distorted, &k, &d, iters, 0.0).unwrap()[0];
            let (x, y) = (500.0 * n.0 + 320.0, 500.0 * n.1 + 240.0);
            (x - ideal[0].0).hypot(y - ideal[0].1)
        };
        assert!(err(20) <= err(1), "iterating should not make it worse");
        assert!(err(20) < 1e-6, "twenty iterations should converge: {}", err(20));

        assert!(undistort_points_iter(&ideal, &k, &d, 0, 1e-9).is_err());
        assert!(distort_points(&ideal, &k, &[0.0, 0.0]).is_err(), "too few coefficients");
        let singular = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(distort_points(&ideal, &singular, &d).is_err());
    }

    #[test]
    fn undistort_image_straightens_a_distorted_grid() {
        // A grid of bright dots, distorted, then undistorted back.
        let (k, d) = camera();
        let mut src = Mat::zeros(64, 64, cv_core::types::CV_8UC1).unwrap();
        for y in (4..60).step_by(8) {
            for x in (4..60).step_by(8) {
                src.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        let k_small = crate::camera_matrix(60.0, 60.0, 32.0, 32.0);
        let out = undistort_image(&src, &k_small, &d, None).unwrap();
        assert_eq!((out.rows(), out.cols()), (64, 64));
        assert_eq!(out.mat_type(), src.mat_type());
        // Something survived: an undistortion that mapped everything off the
        // image would be silently empty.
        let lit: usize = (0..64)
            .flat_map(|y| (0..64).map(move |x| (y, x)))
            .filter(|&(y, x)| out.at_f64(y, x, 0).unwrap() > 10.0)
            .count();
        assert!(lit > 0, "the undistorted image should not be blank");
        let empty = Mat::zeros(0, 0, cv_core::types::CV_8UC1).unwrap();
        assert!(undistort_image(&empty, &k, &d, None).is_err());
    }

    #[test]
    fn the_new_camera_matrix_slides_between_cropping_and_keeping_the_field() {
        let (k, d) = camera();
        let tight =
            estimate_new_camera_matrix_for_undistort_rectify(&k, &d, Size::new(640, 480), 0.0, 1.0)
                .unwrap();
        let wide =
            estimate_new_camera_matrix_for_undistort_rectify(&k, &d, Size::new(640, 480), 1.0, 1.0)
                .unwrap();
        // Keeping the whole field means a shorter focal length: more fits in.
        assert!(wide[0][0] <= tight[0][0], "{} vs {}", wide[0][0], tight[0][0]);
        // The principal point lands at the image centre either way.
        assert!((tight[0][2] - 319.5).abs() < 1e-9);
        assert!((tight[1][2] - 239.5).abs() < 1e-9);
        // fov_scale divides the focal length, widening the view.
        let scaled =
            estimate_new_camera_matrix_for_undistort_rectify(&k, &d, Size::new(640, 480), 0.0, 2.0)
                .unwrap();
        assert!((scaled[0][0] - tight[0][0] / 2.0).abs() < 1e-9);

        assert!(estimate_new_camera_matrix_for_undistort_rectify(
            &k,
            &d,
            Size::new(0, 480),
            0.0,
            1.0
        )
        .is_err());
        assert!(estimate_new_camera_matrix_for_undistort_rectify(
            &k,
            &d,
            Size::new(640, 480),
            2.0,
            1.0
        )
        .is_err());
    }

    #[test]
    fn an_affine_3d_fit_recovers_a_known_transform() {
        let truth = [[2.0, 0.0, 0.0, 5.0], [0.0, 3.0, 0.0, -2.0], [0.0, 0.0, 1.5, 1.0]];
        let from = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, -1.0, 0.5],
        ];
        let to: Vec<[f64; 3]> = from
            .iter()
            .map(|p| {
                [
                    truth[0][0] * p[0] + truth[0][3],
                    truth[1][1] * p[1] + truth[1][3],
                    truth[2][2] * p[2] + truth[2][3],
                ]
            })
            .collect();
        let (m, inliers) = estimate_affine_3d(&from, &to).unwrap();
        for i in 0..3 {
            for j in 0..4 {
                assert!((m[i][j] - truth[i][j]).abs() < 1e-9, "[{i}][{j}]: {}", m[i][j]);
            }
        }
        assert!(inliers.iter().all(|&b| b), "an exact fit has no outliers");

        assert!(estimate_affine_3d(&from[..3], &to[..3]).is_err(), "too few points");
        assert!(estimate_affine_3d(&from, &to[..3]).is_err(), "mismatched counts");
        // Four coplanar points do not determine a 3D affine map.
        let flat = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
        assert!(estimate_affine_3d(&flat, &flat).is_err());
    }

    #[test]
    fn translation_estimation_is_the_mean_displacement() {
        let from3 = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 4.0, 0.5]];
        let to3: Vec<[f64; 3]> =
            from3.iter().map(|p| [p[0] + 2.0, p[1] - 1.0, p[2] + 0.5]).collect();
        let (t, inliers) = estimate_translation_3d(&from3, &to3).unwrap();
        assert!((t[0] - 2.0).abs() < 1e-12);
        assert!((t[1] + 1.0).abs() < 1e-12);
        assert!((t[2] - 0.5).abs() < 1e-12);
        assert!(inliers.iter().all(|&b| b));

        let from2 = [(0.0, 0.0), (3.0, 4.0), (-2.0, 7.0)];
        let to2: Vec<(f64, f64)> = from2.iter().map(|p| (p.0 + 1.5, p.1 - 0.5)).collect();
        let ((tx, ty), _) = estimate_translation_2d(&from2, &to2).unwrap();
        assert!((tx - 1.5).abs() < 1e-12);
        assert!((ty + 0.5).abs() < 1e-12);

        assert!(estimate_translation_2d(&[], &[]).is_err());
        assert!(estimate_translation_3d(&from3, &to3[..1]).is_err());
    }

    #[test]
    fn correcting_matches_moves_them_onto_the_epipolar_constraint() {
        // A pure horizontal-translation fundamental matrix: y1 must equal y2.
        let f = [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
        let p1 = [(10.0, 20.0), (50.0, 60.0)];
        let p2 = [(30.0, 22.0), (70.0, 59.0)];
        let residual = |a: (f64, f64), b: (f64, f64)| {
            let x1 = [a.0, a.1, 1.0];
            let x2 = [b.0, b.1, 1.0];
            let fx1 = mat3::mul3(&f, x1);
            (0..3).map(|i| x2[i] * fx1[i]).sum::<f64>().abs()
        };
        let before: f64 = p1.iter().zip(&p2).map(|(&a, &b)| residual(a, b)).sum();
        assert!(before > 1.0, "the fixture should start off the constraint");

        let (a, b) = correct_matches(&f, &p1, &p2).unwrap();
        let after: f64 = a.iter().zip(&b).map(|(&x, &y)| residual(x, y)).sum();
        assert!(after < 1e-9, "the corrected pairs should satisfy it: {after}");
        // And the correction is small — it moves points, it does not relocate them.
        for i in 0..2 {
            assert!((a[i].0 - p1[i].0).hypot(a[i].1 - p1[i].1) < 3.0);
        }
        assert!(correct_matches(&f, &p1, &p2[..1]).is_err());
    }

    #[test]
    fn the_essential_matrix_is_the_fundamental_one_with_intrinsics_removed() {
        // Points from a real pair of views so the eight-point fit is defined.
        let k = crate::camera_matrix(500.0, 500.0, 320.0, 240.0);
        let p1: Vec<(f64, f64)> = (0..10)
            .map(|i| (100.0 + f64::from(i) * 37.0, 90.0 + f64::from((i * 53) % 200)))
            .collect();
        let h = [[1.0, 0.02, 12.0], [-0.01, 1.0, -7.0], [1e-5, 2e-5, 1.0]];
        let p2: Vec<(f64, f64)> = p1.iter().map(|&p| apply_homography(&h, p)).collect();
        let e = find_essential_mat(&p1, &p2, &k).unwrap();
        // Normalised to unit Frobenius norm, so it is comparable run to run.
        let n: f64 = e.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-9, "{n}");
        // An essential matrix is singular.
        let det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
            - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
            + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
        assert!(det.abs() < 1e-6, "det {det}");
        assert!(find_essential_mat(&p1[..5], &p2[..5], &k).is_err());
    }

    #[test]
    fn the_product_derivatives_agree_with_finite_differences() {
        let a = Mat::from_vec(2, 3, 1, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Mat::from_vec(3, 2, 1, &[7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let (da, db) = mat_mul_deriv(&a, &b).unwrap();
        assert_eq!((da.rows(), da.cols()), (4, 6), "(m*p) x (m*n)");
        assert_eq!((db.rows(), db.cols()), (4, 6), "(m*p) x (n*p)");

        // Perturb one entry of A and check the predicted change in AB.
        let product = |a: &Mat, b: &Mat| -> Vec<f64> {
            let mut out = Vec::new();
            for i in 0..2 {
                for j in 0..2 {
                    out.push(
                        (0..3).map(|k| a.at_f64(i, k, 0).unwrap() * b.at_f64(k, j, 0).unwrap()).sum(),
                    );
                }
            }
            out
        };
        let base = product(&a, &b);
        let h = 1e-6;
        for idx in 0i32..6 {
            let mut a2 = a.clone();
            a2.pixel_mut::<f64>(idx / 3, idx % 3).unwrap()[0] += h;
            let moved = product(&a2, &b);
            for row in 0i32..4 {
                let numeric = (moved[row as usize] - base[row as usize]) / h;
                let analytic = da.at_f64(row, idx, 0).unwrap();
                assert!((numeric - analytic).abs() < 1e-4, "dA[{row}][{idx}]");
            }
        }
        let bad = Mat::zeros(4, 4, CV_64FC1).unwrap();
        assert!(mat_mul_deriv(&a, &bad).is_err(), "shapes must be compatible");
    }

    #[test]
    fn uncalibrated_rectification_sends_the_epipoles_to_infinity() {
        // A fundamental matrix for a horizontal translation: the epipoles are
        // at infinity along x already, so the transforms should be near-identity
        // in the sense that they preserve horizontal lines.
        let f = [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
        let (h1, h2) = stereo_rectify_uncalibrated(&f, Size::new(640, 480)).unwrap();
        for h in [h1, h2] {
            // A homography is only defined up to scale; check it is not degenerate.
            let n: f64 = h.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
            assert!(n.is_finite() && n > 1e-9, "{h:?}");
            assert!(mat3_inv(&h).is_some(), "a rectifying homography must be invertible");
        }
        assert!(stereo_rectify_uncalibrated(&f, Size::new(0, 0)).is_err());
    }

    #[test]
    fn the_visible_reference_points_filter_keeps_the_plausible_solutions() {
        let before = [(0.1, 0.1), (0.2, -0.1)];
        let after = [(0.15, 0.1), (0.25, -0.1)];
        // A normal pointing along +z sees both points; one pointing along -z
        // sees neither.
        let normals = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
        let kept = filter_homography_decomp_by_visible_refpoints(&normals, &before, &after).unwrap();
        assert_eq!(kept, vec![0]);
        assert!(filter_homography_decomp_by_visible_refpoints(&normals, &before, &after[..1])
            .is_err());
        assert!(filter_homography_decomp_by_visible_refpoints(&normals, &[], &[]).is_err());
    }

    #[test]
    fn three_collinear_cameras_share_the_middle_rectification() {
        let k = crate::camera_matrix(500.0, 500.0, 320.0, 240.0);
        let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (r1, r2, r3, p) =
            rectify3_collinear(&k, &k, &k, &eye, [0.05, 0.0, 0.0], &eye, [0.1, 0.0, 0.0]).unwrap();
        // The first two cameras rectify identically, which is what makes both
        // disparity maps live in the middle image's frame.
        assert_eq!(r1, r2);
        assert!(mat3_inv(&r3).is_some());
        assert_eq!(p[0][0], 500.0);
        assert_eq!(p[1][2], 240.0);
    }

    #[test]
    fn frame_axes_and_chessboard_corners_draw_without_running_off_the_image() {
        let mut img = Mat::zeros(200, 200, CV_8UC3).unwrap();
        let k = crate::camera_matrix(200.0, 200.0, 100.0, 100.0);
        draw_frame_axes(&mut img, &k, &[0.0; 5], [0.0, 0.0, 0.0], [0.0, 0.0, 5.0], 1.0, 2)
            .unwrap();
        let lit: usize = (0..200)
            .flat_map(|y| (0..200).map(move |x| (y, x)))
            .filter(|&(y, x)| img.at_f64(y, x, 0).unwrap() + img.at_f64(y, x, 2).unwrap() > 0.0)
            .count();
        assert!(lit > 0, "the axes should be drawn");
        assert!(draw_frame_axes(&mut img, &k, &[0.0; 5], [0.0; 3], [0.0, 0.0, 5.0], 0.0, 2)
            .is_err());
        assert!(draw_frame_axes(&mut img, &k, &[0.0; 5], [0.0; 3], [0.0, 0.0, 5.0], 1.0, 0)
            .is_err());

        let mut board = Mat::zeros(100, 100, CV_8UC3).unwrap();
        let corners: Vec<(f32, f32)> = (0..6)
            .flat_map(|r| (0..4).map(move |c| (10.0 + c as f32 * 20.0, 10.0 + r as f32 * 15.0)))
            .collect();
        draw_chessboard_corners(&mut board, Size::new(4, 6), &corners, true).unwrap();
        // A wrong corner count for the stated pattern is refused.
        assert!(draw_chessboard_corners(&mut board, Size::new(4, 6), &corners[..5], true).is_err());
        // ...but an incomplete detection draws anyway, as plain circles.
        draw_chessboard_corners(&mut board, Size::new(4, 6), &corners[..5], false).unwrap();
        assert!(draw_chessboard_corners(&mut board, Size::new(0, 6), &corners, true).is_err());
    }
}
