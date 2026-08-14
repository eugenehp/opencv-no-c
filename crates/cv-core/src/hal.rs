//! `CV_HAL_*` — the constants of OpenCV's hardware abstraction layer
//! (`modules/core/include/opencv2/core/hal/interface.h`).
//!
//! The HAL is a C ABI for swapping in a vendor's optimised kernels. This port
//! implements no HAL and links no C, so there is nothing here but the integer
//! constants that define the interface. They are worth having: their values are
//! **not** the same as the `cv::` enums they shadow, and confusing the two is a
//! silent bug.
//!
//! The clearest example is SVD. `SVD::MODIFY_A` is 1 and `SVD::NO_UV` is 2;
//! [`svd::CV_HAL_SVD_NO_UV`] is 1 and [`svd::CV_HAL_SVD_MODIFY_A`] is 4. A
//! flags word passed from one convention to the other means something else
//! entirely.

/// `CV_SIMD*` — how wide a vector the build compiled for.
///
/// All zero, and `CV_SIMD_WIDTH` at the 128-bit baseline. Upstream's headers
/// set these from the compiler's `-m` flags so that `#if CV_SIMD` picks a
/// vector path; this port has one scalar implementation per function, chosen
/// for exact parity, so every such test must take the scalar branch.
///
/// They are constants rather than a runtime query on purpose: this is what the
/// *build* has, not what the CPU has. For the CPU, see
/// [`crate::utility::check_hardware_support`], which does report the truth.
pub mod simd {
    /// `CV_SIMD` — is any vector path compiled in?
    pub const CV_SIMD: i32 = 0;
    /// `CV_SIMD_WIDTH` — vector width in bytes; upstream's 128-bit baseline.
    pub const CV_SIMD_WIDTH: i32 = 16;
    /// `CV_SIMD_64F` — are `double` vectors available?
    pub const CV_SIMD_64F: i32 = 0;
    /// `CV_SIMD_FP16` — are half-float vectors available?
    pub const CV_SIMD_FP16: i32 = 0;
    /// `CV_SIMD_SCALABLE` — a length-agnostic ISA such as SVE or RVV.
    pub const CV_SIMD_SCALABLE: i32 = 0;
    /// `CV_SIMD_SCALABLE_64F`
    pub const CV_SIMD_SCALABLE_64F: i32 = 0;
    /// `CV_SIMD128`
    pub const CV_SIMD128: i32 = 0;
    /// `CV_SIMD128_64F`
    pub const CV_SIMD128_64F: i32 = 0;
    /// `CV_SIMD128_FP16`
    pub const CV_SIMD128_FP16: i32 = 0;
    /// `CV_SIMD128_CPP` — the portable C++ fallback implementation.
    pub const CV_SIMD128_CPP: i32 = 0;
    /// `CV_SIMD256`
    pub const CV_SIMD256: i32 = 0;
    /// `CV_SIMD256_64F`
    pub const CV_SIMD256_64F: i32 = 0;
    /// `CV_SIMD256_FP16`
    pub const CV_SIMD256_FP16: i32 = 0;
    /// `CV_SIMD512`
    pub const CV_SIMD512: i32 = 0;
    /// `CV_SIMD512_64F`
    pub const CV_SIMD512_64F: i32 = 0;
    /// `CV_SIMD512_FP16`
    pub const CV_SIMD512_FP16: i32 = 0;
    /// `CV_CPU_BASELINE_MODE` — is only the baseline ISA compiled in?
    pub const CV_CPU_BASELINE_MODE: i32 = 1;
}

/// `cv::hal::StoreMode` — how a vector store should reach memory.
///
/// Declared for translating C++; nothing here issues vector stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum StoreMode {
    /// `STORE_UNALIGNED`
    #[default]
    Unaligned = 0,
    /// `STORE_ALIGNED`
    Aligned = 1,
    /// `STORE_ALIGNED_NOCACHE` — aligned, and bypassing the cache.
    AlignedNocache = 2,
}

/// Return codes a HAL function may give.
pub mod error {
    /// `CV_HAL_ERROR_OK` — handled.
    pub const CV_HAL_ERROR_OK: i32 = 0;
    /// `CV_HAL_ERROR_NOT_IMPLEMENTED` — fall back to the built-in path.
    pub const CV_HAL_ERROR_NOT_IMPLEMENTED: i32 = 1;
    /// `CV_HAL_ERROR_UNKNOWN` — failed for some other reason.
    pub const CV_HAL_ERROR_UNKNOWN: i32 = -1;
}

/// Comparison operators. These *do* match [`crate::CmpType`]'s values.
pub mod cmp {
    /// `CV_HAL_CMP_EQ`
    pub const CV_HAL_CMP_EQ: i32 = 0;
    /// `CV_HAL_CMP_GT`
    pub const CV_HAL_CMP_GT: i32 = 1;
    /// `CV_HAL_CMP_GE`
    pub const CV_HAL_CMP_GE: i32 = 2;
    /// `CV_HAL_CMP_LT`
    pub const CV_HAL_CMP_LT: i32 = 3;
    /// `CV_HAL_CMP_LE`
    pub const CV_HAL_CMP_LE: i32 = 4;
    /// `CV_HAL_CMP_NE`
    pub const CV_HAL_CMP_NE: i32 = 5;
}

/// Border modes. These match [`crate::BorderType`]'s values.
pub mod border {
    /// `CV_HAL_BORDER_CONSTANT`
    pub const CV_HAL_BORDER_CONSTANT: i32 = 0;
    /// `CV_HAL_BORDER_REPLICATE`
    pub const CV_HAL_BORDER_REPLICATE: i32 = 1;
    /// `CV_HAL_BORDER_REFLECT`
    pub const CV_HAL_BORDER_REFLECT: i32 = 2;
    /// `CV_HAL_BORDER_WRAP`
    pub const CV_HAL_BORDER_WRAP: i32 = 3;
    /// `CV_HAL_BORDER_REFLECT_101`
    pub const CV_HAL_BORDER_REFLECT_101: i32 = 4;
    /// `CV_HAL_BORDER_TRANSPARENT`
    pub const CV_HAL_BORDER_TRANSPARENT: i32 = 5;
    /// `CV_HAL_BORDER_ISOLATED`
    pub const CV_HAL_BORDER_ISOLATED: i32 = 16;
}

/// DFT flags. The first five match [`crate::dft::flags`]; the rest are HAL-only
/// hints about the buffer layout that have no `cv::` counterpart.
pub mod dft {
    /// `CV_HAL_DFT_INVERSE`
    pub const CV_HAL_DFT_INVERSE: i32 = 1;
    /// `CV_HAL_DFT_SCALE`
    pub const CV_HAL_DFT_SCALE: i32 = 2;
    /// `CV_HAL_DFT_ROWS`
    pub const CV_HAL_DFT_ROWS: i32 = 4;
    /// `CV_HAL_DFT_COMPLEX_OUTPUT`
    pub const CV_HAL_DFT_COMPLEX_OUTPUT: i32 = 16;
    /// `CV_HAL_DFT_REAL_OUTPUT`
    pub const CV_HAL_DFT_REAL_OUTPUT: i32 = 32;
    /// `CV_HAL_DFT_TWO_STAGE` — the transform is split across two passes.
    pub const CV_HAL_DFT_TWO_STAGE: i32 = 64;
    /// `CV_HAL_DFT_STAGE_COLS` — this pass handles the columns.
    pub const CV_HAL_DFT_STAGE_COLS: i32 = 128;
    /// `CV_HAL_DFT_IS_CONTINUOUS` — the source has no row padding.
    pub const CV_HAL_DFT_IS_CONTINUOUS: i32 = 512;
    /// `CV_HAL_DFT_IS_INPLACE` — source and destination are the same buffer.
    pub const CV_HAL_DFT_IS_INPLACE: i32 = 1024;
}

/// SVD flags. **These do not match `cv::SVD::Flags`** — see the module header.
pub mod svd {
    /// `CV_HAL_SVD_NO_UV` — `cv::SVD::NO_UV` is 2, not 1.
    pub const CV_HAL_SVD_NO_UV: i32 = 1;
    /// `CV_HAL_SVD_SHORT_UV` — the thin factorisation; no `cv::` counterpart.
    pub const CV_HAL_SVD_SHORT_UV: i32 = 2;
    /// `CV_HAL_SVD_MODIFY_A` — `cv::SVD::MODIFY_A` is 1, not 4.
    pub const CV_HAL_SVD_MODIFY_A: i32 = 4;
    /// `CV_HAL_SVD_FULL_UV` — `cv::SVD::FULL_UV` is 4, not 8.
    pub const CV_HAL_SVD_FULL_UV: i32 = 8;
}

/// GEMM transposition flags. These match [`crate::GemmFlags`]'s values.
pub mod gemm {
    /// `CV_HAL_GEMM_1_T` — transpose the first operand.
    pub const CV_HAL_GEMM_1_T: i32 = 1;
    /// `CV_HAL_GEMM_2_T` — transpose the second operand.
    pub const CV_HAL_GEMM_2_T: i32 = 2;
    /// `CV_HAL_GEMM_3_T` — transpose the third operand.
    pub const CV_HAL_GEMM_3_T: i32 = 4;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_families_that_agree_with_the_cv_enums_do_agree() {
        assert_eq!(cmp::CV_HAL_CMP_EQ, crate::CmpType::Eq as i32);
        assert_eq!(cmp::CV_HAL_CMP_NE, crate::CmpType::Ne as i32);
        assert_eq!(border::CV_HAL_BORDER_REFLECT_101, crate::BorderType::Reflect101 as i32);
        assert_eq!(border::CV_HAL_BORDER_TRANSPARENT, crate::BorderType::Transparent as i32);
        assert_eq!(border::CV_HAL_BORDER_ISOLATED, crate::border::BORDER_ISOLATED);
        assert_eq!(dft::CV_HAL_DFT_INVERSE, crate::dft::flags::DFT_INVERSE);
        assert_eq!(dft::CV_HAL_DFT_ROWS, crate::dft::flags::DFT_ROWS);
        assert_eq!(dft::CV_HAL_DFT_REAL_OUTPUT, crate::dft::flags::DFT_REAL_OUTPUT);
    }

    /// The whole reason these constants are worth declaring separately.
    #[test]
    fn the_svd_flags_deliberately_disagree_with_the_cv_enum() {
        use crate::linalg::svd_flags;
        assert_eq!(svd::CV_HAL_SVD_MODIFY_A, 4);
        assert_eq!(svd_flags::MODIFY_A, 1);
        assert_eq!(svd::CV_HAL_SVD_NO_UV, 1);
        assert_eq!(svd_flags::NO_UV, 2);
        assert_eq!(svd::CV_HAL_SVD_FULL_UV, 8);
        assert_eq!(svd_flags::FULL_UV, 4);
        // The HAL has a fourth mode the cv:: enum has no name for.
        assert_eq!(svd::CV_HAL_SVD_SHORT_UV, 2);
    }

    #[test]
    fn error_and_gemm_codes_match_upstream() {
        assert_eq!(error::CV_HAL_ERROR_OK, 0);
        assert_eq!(error::CV_HAL_ERROR_NOT_IMPLEMENTED, 1);
        assert_eq!(error::CV_HAL_ERROR_UNKNOWN, -1);
        assert_eq!(gemm::CV_HAL_GEMM_1_T, 1);
        assert_eq!(gemm::CV_HAL_GEMM_3_T, 4);
        // The two-stage DFT hints have no cv:: counterpart at all.
        assert_eq!(dft::CV_HAL_DFT_TWO_STAGE, 64);
        assert_eq!(dft::CV_HAL_DFT_IS_INPLACE, 1024);
    }
}
