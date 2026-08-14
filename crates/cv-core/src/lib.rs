//! Pure-Rust port of the OpenCV `core` module.
//!
//! Targets upstream **OpenCV 4.15.0-dev** (`77dfa297d08fdecdc509fc01ad92a2e9ec776a57`).
//! No C or C++ is linked: every routine here is written in Rust.
//!
//! # Mapping from the C++ API
//!
//! | upstream                    | here                                    |
//! |-----------------------------|-----------------------------------------|
//! | `cv::Mat`                   | [`Mat`], [`MatView`], [`MatViewMut`]    |
//! | `cv::UMat`                  | [`UMat`]                                |
//! | `CV_8UC3` and friends       | [`types::CV_8UC3`] ([`MatType`])        |
//! | `cv::saturate_cast<T>`      | [`saturate::Elem::from_f64`]            |
//! | `cv::Exception` (thrown)    | [`Error`] returned in a [`Result`]      |
//! | `CV_Assert` / `CV_CheckEQ`  | [`cv_assert!`] / [`cv_check_eq!`]       |
//!
//! Every ported item names its upstream counterpart in backticks in its own doc
//! comment. That is not decoration: `cargo run -p xtask -- report` counts those
//! claims against a scan of the upstream headers, so the parity number cannot
//! drift from what the code actually says.
//!
//! # Deliberate divergences
//!
//! Three, all pervasive, all load-bearing:
//!
//! 1. **Errors are returned, not thrown.** Every fallible function gives a
//!    [`Result`]; [`cv_assert!`] and the [`check`] family `return Err`.
//! 2. **Matrices own their pixels.** `cv::Mat` is reference-counted with
//!    shallow copy semantics, which is unrepresentable in safe Rust and is the
//!    largest source of aliasing bugs in OpenCV client code. See [`mat`].
//! 3. **One scalar implementation per function.** No SIMD dispatch, no IPP, no
//!    OpenCL — chosen so results are bit-exact against cv2 rather than fast.
//!    See [`build_config`].
//!
//! # The modules
//!
//! **Data.** [`mat`] and [`umat`] are the matrix types; [`types`] the element
//! encoding; [`geometry`] the small value types ([`Point`], [`Rect`],
//! [`Scalar`], [`RotatedRect`], [`KeyPoint`], [`Moments`], [`TermCriteria`]);
//! [`buffer`] the aligned allocation underneath.
//!
//! **Arithmetic and transforms.** [`arithm`] (element-wise, with saturation),
//! [`mathfuncs`] (`exp`, `log`, `pow`, polar/cartesian), [`manip`] (channel
//! and layout surgery), [`stats`] (norms, sums, extrema), [`metrics`],
//! [`bulk`], [`saturate`], [`softfloat`].
//!
//! **Linear algebra.** `matmul` (`gemm`, `invert`, `solve` with all five
//! decompositions), [`linalg`] (SVD, eigen, PCA, k-means), [`linalg2`],
//! [`solvers`] (polynomial roots, non-symmetric eigenvalues, `solveLP`),
//! `dft`, [`quaternion`].
//!
//! **Infrastructure.** [`error`], [`check`] (argument validation with decoded
//! messages), [`logger`], `trace` and [`instr`] (scoped profiling),
//! [`utility`] (timing, threading, CPU features), [`persistence`]
//! (FileStorage), `format` (`cv::format`), [`algorithm`], [`async_array`],
//! [`rng`], [`border`], [`cvdef`], [`macros`].
//!
//! **Backend surfaces.** [`cuda`], [`ocl`] and [`opengl`] reproduce the shape
//! those namespaces have in a build without the corresponding backend —
//! queries answer, operations report `GpuNotSupported`. [`hal`] and [`intrin`]
//! carry the HAL constants and the `V_TypeTraits` table. See each module's
//! header for why they exist at all.
//!
//! # Parity
//!
//! `core` is at 100% of the upstream symbol surface the header scanner finds.
//! Numeric agreement with cv2 is checked separately and continuously by
//! `cargo run -p opencv-no-c --example parity_check`, which diffs against
//! arrays produced by a real OpenCV build.

#![doc(html_root_url = "https://docs.rs/cv-core/0.1.0")]
#![warn(missing_docs)]

pub mod algorithm;
pub mod arithm;
pub mod async_array;
pub mod bindings_utils;
pub mod border;
pub mod buffer;
pub mod build_config;
pub mod bulk;
pub mod check;
#[cfg(feature = "compat")]
pub mod compat;
pub mod cuda;
pub mod cvdef;
pub mod dispatch;
pub mod dft;
pub mod error;
pub mod format;
pub mod geometry;
pub mod hal;
pub mod instr;
pub mod intrin;
pub mod linalg2;
pub mod linalg;
pub mod logger;
pub mod macros;
pub mod manip;
pub mod mat;
pub mod mathfuncs;
pub mod matmul;
pub mod metrics;
pub mod misc;
pub mod ocl;
pub mod opengl;
pub mod operations;
pub mod persistence;
pub mod quaternion;
pub mod rng;
pub mod saturate;
pub mod softfloat;
pub mod solvers;
pub mod stats;
pub mod trace;
pub mod types;
pub mod umat;
pub mod utility;

pub use arithm::{
    absdiff, add, add_masked, add_scalar, add_weighted, bitwise_and, bitwise_not, bitwise_or, bitwise_xor,
    compare, convert_to, divide, flip, in_range, max, merge, min, multiply, set_channel, split,
    subtract, subtract_masked, subtract_scalar, transpose, CmpType,
};
pub use border::{border_interpolate, copy_make_border, reflect101, BorderType};
pub use misc::{
    batch_distance, broadcast, check_range, complete_symm, convert_scale_abs, cube_root,
    cv_is_inf, cv_is_nan, cv_round64, cv_trunc, find_non_zero, has_non_zero, min_max_idx,
    normalize_minmax_to, patch_nans, perspective_transform, quantize_fp16, rand_shuffle,
    reduce_arg, rotate, round_to_index, set_rng_seed, transform, DistanceType, Extremes,
    RotateCode,
};
pub use persistence::{FileStorage, Format as StorageFormat, Node};
pub use dft::{
    idft_real,
    complex_magnitude, complex_type, dct, dft, get_optimal_dft_size, idct, idft, mul_spectrums,
    DftDirection,
};
pub use linalg::{
    calc_covar_matrix, eigen, kmeans, mahalanobis, pca_compute, svd, svd_back_subst, DataLayout,
    KMeans, KMeansFlags, Pca, Svd,
};
pub use bulk::{from_f64, mat_from_f64, to_f64};
pub use manip::{
    extract_channel, hconcat, insert_channel, mix_channels, reduce, repeat, sort, sort_idx,
    ReduceType, SortFlags,
};
pub use mathfuncs::{
    cart_to_polar, exp, log, lut, magnitude, phase, polar_to_cart, pow, sqrt,
};
pub use manip::vconcat;
pub use metrics::{
    cosine_distance, cosine_distances, cosine_similarities, cosine_similarity, l2_normalize,
    nearest_by_cosine,
};
pub use matmul::{
    determinant, gemm, invert, matmul, mul_transposed, scale_add, set_identity, solve,
    solve_normal, trace, Decomp, GemmFlags, DECOMP_NORMAL,
};
pub use cvdef::{
    cv_cmp, cv_imax, cv_imin, cv_sign, cv_swap, CV_2PI, CV_LOG2, CV_PI, CV_VERSION_MAJOR,
    CV_VERSION_MINOR, CV_VERSION_REVISION, CV_VERSION_STATUS,
};
pub use algorithm::{Algorithm, Param};
pub use async_array::{AsyncArray, AsyncPromise};
pub use error::{Code, Error, Result};
pub use quaternion::{Assume as QuatAssume, EulerAnglesType, Quat};
pub use format::{format, FormatType};
pub use solvers::{solve_lp, SolveLpResult};
pub use utility::{
    check_hardware_support, get_cpu_features_line, get_default_algorithm_hint,
    get_hardware_feature_name, AlgorithmHint, CpuFeature,
};
pub use geometry::{
    DMatch, KeyPoint, Moments, Point, Point2d, Point2f, Point_, Range, Rect, Rect2d, Rect2f,
    Rect_, RotatedRect, Scalar, Size, Size2d, Size2f, Size_, TermCriteria,
};
pub use mat::{Mat, MatLike, MatLikeMut, MatView, MatViewMut};
pub use umat::{UMat, UMatUsageFlags};
pub use rng::{randn, randu, random_mat, Distribution, Rng};
pub use saturate::{cv_ceil, cv_floor, cv_round, cv_roundf, saturate_cast, Elem};
pub use stats::{
    count_non_zero, mean, mean_std_dev, min_max_loc, norm, norm_diff, norm_diff_masked,
    norm_masked, norm_relative, normalize_minmax, sum, MinMaxLoc, NormType, NORM_MINMAX,
    NORM_RELATIVE, NORM_TYPE_MASK,
};
pub use types::{Depth, MatType};

/// `CV_VERSION` — the upstream OpenCV release this port tracks.
pub const CV_VERSION: &str = "4.15.0-dev";
/// Upstream commit the ported semantics were read from.
pub const CV_UPSTREAM_COMMIT: &str = "77dfa297d08fdecdc509fc01ad92a2e9ec776a57";
