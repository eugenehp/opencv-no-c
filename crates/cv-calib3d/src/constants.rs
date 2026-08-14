//! Flag constants of `cv::calib3d`
//! (`modules/calib3d/include/opencv2/calib3d.hpp`).
//!
//! Calibration flags (`CALIB_*`), robust-estimation methods (`RANSAC`, `LMEDS`,
//! `RHO`), PnP solver selectors (`SOLVEPNP_*`), stereo matcher and rectify
//! options, and the fisheye family. Generated from the upstream header.
//!
//! A flag naming a method this crate does not implement is still worth having:
//! it is what a calibration config or a translated call site carries, and
//! passing one gives a clear error instead of being silently ignored.

/// `RANSAC` — RANSAC algorithm.
pub const RANSAC: i32 = 8;
/// `RHO` — RHO algorithm.
pub const RHO: i32 = 16;
/// `USAC_DEFAULT` — USAC algorithm, default settings.
pub const USAC_DEFAULT: i32 = 32;
/// `USAC_PARALLEL` — USAC, parallel version.
pub const USAC_PARALLEL: i32 = 33;
/// `USAC_FM_8PTS` — USAC, fundamental matrix 8 points.
pub const USAC_FM_8PTS: i32 = 34;
/// `USAC_FAST` — USAC, fast settings.
pub const USAC_FAST: i32 = 35;
/// `USAC_ACCURATE` — USAC, accurate settings.
pub const USAC_ACCURATE: i32 = 36;
/// `USAC_PROSAC` — USAC, sorted points, runs PROSAC.
pub const USAC_PROSAC: i32 = 37;
/// `USAC_MAGSAC` — USAC, runs MAGSAC++.
pub const USAC_MAGSAC: i32 = 38;
/// `SOLVEPNP_MAX_COUNT` — Used for count.
pub const SOLVEPNP_MAX_COUNT: i32 = 9;
/// `CALIB_USE_QR` — use QR instead of SVD decomposition for solving. Faster
///   but potentially less precise.
pub const CALIB_USE_QR: i32 = 1048576;
/// `CALIB_ZERO_DISPARITY`.
pub const CALIB_ZERO_DISPARITY: i32 = 1024;
/// `CALIB_USE_LU` — use LU instead of SVD decomposition for solving. much
///   faster but potentially less precise.
pub const CALIB_USE_LU: i32 = 131072;
/// `CALIB_USE_EXTRINSIC_GUESS` — for stereoCalibrate.
pub const CALIB_USE_EXTRINSIC_GUESS: i32 = 4194304;
/// `CALIB_DISABLE_SCHUR_COMPLEMENT` — disable Schur complement (use Bouguet
///   calibration engine).
pub const CALIB_DISABLE_SCHUR_COMPLEMENT: i32 = 8388608;
/// `FM_8POINT` — 8-point algorithm.
pub const FM_8POINT: i32 = 2;
/// `FM_LMEDS` — least-median algorithm. 7-point algorithm is used.
pub const FM_LMEDS: i32 = 4;
/// `FM_RANSAC` — RANSAC algorithm. It needs at least 15 points. 7-point
///   algorithm is used.
pub const FM_RANSAC: i32 = 8;
/// `SAMPLING_PROGRESSIVE_NAPSAC`.
pub const SAMPLING_PROGRESSIVE_NAPSAC: i32 = 1;
/// `SAMPLING_NAPSAC`.
pub const SAMPLING_NAPSAC: i32 = 2;
/// `LOCAL_OPTIM_INNER_LO`.
pub const LOCAL_OPTIM_INNER_LO: i32 = 1;
/// `LOCAL_OPTIM_INNER_AND_ITER_LO`.
pub const LOCAL_OPTIM_INNER_AND_ITER_LO: i32 = 2;
/// `LOCAL_OPTIM_GC`.
pub const LOCAL_OPTIM_GC: i32 = 3;
/// `SCORE_METHOD_MSAC`.
pub const SCORE_METHOD_MSAC: i32 = 1;
/// `SCORE_METHOD_MAGSAC`.
pub const SCORE_METHOD_MAGSAC: i32 = 2;
/// `NEIGH_GRID`.
pub const NEIGH_GRID: i32 = 1;
/// `LSQ_POLISHER`.
pub const LSQ_POLISHER: i32 = 1;
/// `MAGSAC`.
pub const MAGSAC: i32 = 2;
/// `SYMMETRIC_GRID`.
pub const SYMMETRIC_GRID: i32 = 0;
/// `ASYMMETRIC_GRID`.
pub const ASYMMETRIC_GRID: i32 = 1;
/// `DISP_SCALE`.
pub const DISP_SCALE: i32 = 16;
/// `PREFILTER_NORMALIZED_RESPONSE` — Normalized response pre-filter.
pub const PREFILTER_NORMALIZED_RESPONSE: i32 = 0;
/// `PREFILTER_XSOBEL` — X-Sobel pre-filter.
pub const PREFILTER_XSOBEL: i32 = 1;
/// `PROJ_SPHERICAL_ORTHO`.
pub const PROJ_SPHERICAL_ORTHO: i32 = 0;
/// `PROJ_SPHERICAL_EQRECT`.
pub const PROJ_SPHERICAL_EQRECT: i32 = 1;
/// `STARTED`.
pub const STARTED: i32 = 1;
/// `CALC_J`.
pub const CALC_J: i32 = 2;
