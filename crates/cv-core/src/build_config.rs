//! Build-configuration names from `cv_cpu_dispatch.h`, `intrin*.hpp` and
//! `fast_math.hpp`.
//!
//! Upstream compiles each SIMD kernel several times, once per instruction set,
//! into a namespace named after that set, and picks one at runtime. The names
//! here describe that machinery: which namespace a kernel landed in, whether a
//! dispatch table exists, whether a math builtin was available.
//!
//! **This port has none of it.** One scalar implementation per function,
//! chosen for exact parity, compiled once. So every switch below reports the
//! no-dispatch configuration — which is a real configuration upstream also
//! produces, with `CV_CPU_BASELINE_MODE` set and every `CV_SIMD*` at zero (see
//! [`crate::hal::simd`]).
//!
//! They are constants rather than macros because none of them can expand to
//! code here: there is no second namespace to open, no dispatch table to index.

/// `CV_CPU_OPTIMIZATION_NAMESPACE` — the namespace a dispatched kernel lands in.
///
/// `"baseline"`, since there is only ever the one build.
pub const CV_CPU_OPTIMIZATION_NAMESPACE: &str = "baseline";
/// `CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN` — opens it. Nothing to open.
pub const CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN: &str = "";
/// `CV_CPU_OPTIMIZATION_NAMESPACE_END` — closes it.
pub const CV_CPU_OPTIMIZATION_NAMESPACE_END: &str = "";

/// `CV_CPU_OPTIMIZATION_HAL_NAMESPACE` — the same, for HAL kernels.
pub const CV_CPU_OPTIMIZATION_HAL_NAMESPACE: &str = "hal_baseline";
/// `CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN`.
pub const CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN: &str = "";
/// `CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END`.
pub const CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END: &str = "";

/// `CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY` — compile declarations, not bodies.
///
/// False: there is one body and it is always compiled.
pub const CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY: bool = false;

/// `CV_CPU_DISPATCH(fn, args, mode...)` — call the best available variant.
///
/// Empty: with one variant there is nothing to dispatch between, so a
/// translated call site falls straight through to
/// [`CV_CPU_CALL_BASELINE`].
pub const CV_CPU_DISPATCH: &str = "";

/// `CV_CPU_CALL_BASELINE(fn, args)` — call the baseline variant directly.
pub const CV_CPU_CALL_BASELINE: &str = "";

/// `CV_AVX_GUARD` — emit `vzeroupper` on leaving an AVX function.
///
/// Empty. The guard exists to avoid the AVX-SSE transition penalty after
/// inline AVX code; nothing here emits AVX instructions.
pub const CV_AVX_GUARD: &str = "";

/// `CV_API_CALL` — the calling convention of an exported C entry point.
///
/// Empty: this crate exports no C ABI. Rust's equivalent would be
/// `extern "C"` on the function.
pub const CV_API_CALL: &str = "";

/// `CV_DISABLE_CLANG_ENUM_WARNINGS` — suppress `-Wdeprecated-enum-*`.
///
/// Empty. Rust's nearest is `#[allow(...)]` on the item.
pub const CV_DISABLE_CLANG_ENUM_WARNINGS: &str = "";

/// `CV_NEON_AARCH64` — is this the 64-bit NEON variant?
///
/// Set on `aarch64`, where the extra instructions upstream guards with it are
/// architecturally guaranteed. Reported truthfully even though no kernel here
/// uses them, for the same reason
/// [`crate::utility::check_hardware_support`] reports the real CPU.
pub const CV_NEON_AARCH64: i32 = if cfg!(target_arch = "aarch64") { 1 } else { 0 };

/// `CV_RVV_MAX_VLEN` — the widest RISC-V vector length upstream assumes, in bits.
pub const CV_RVV_MAX_VLEN: i32 = 1024;

/// `CV__SIMD_NATIVE_mm_extract_epi64` — is the 64-bit SSE4.1 extract native?
///
/// False; nothing here issues SSE.
pub const CV_SIMD_NATIVE_MM_EXTRACT_EPI64: bool = false;

/// `CV__SIMD_NAMESPACE` — the inline namespace the universal intrinsics live in.
pub const CV_SIMD_NAMESPACE: &str = "";
/// `CV__SIMD_FORWARD` — forward-declares a vector width before its definition.
pub const CV_SIMD_FORWARD: &str = "";

/// Whether the fast-math paths use a compiler builtin.
///
/// Upstream picks between a builtin, an SSE intrinsic and a portable fallback
/// for rounding and the NaN/infinity tests. This port uses Rust's
/// [`f64::round_ties_even`] and [`f64::is_nan`], which lower to the right
/// instruction without a per-compiler switch — so every one of these is false
/// and the ported behaviour is the portable branch.
///
/// See [`crate::cv_round`], [`crate::cv_is_nan`] and [`crate::cv_is_inf`] for
/// what actually runs.
pub mod fast_math {
    /// `CV__FASTMATH_ENABLE_GCC_MATH_BUILTINS`.
    pub const CV_FASTMATH_ENABLE_GCC_MATH_BUILTINS: bool = false;
    /// `CV__FASTMATH_ENABLE_CLANG_MATH_BUILTINS`.
    pub const CV_FASTMATH_ENABLE_CLANG_MATH_BUILTINS: bool = false;
    /// `CV_INLINE_ROUND_DBL` — is `cvRound(double)` inlined from a builtin?
    pub const CV_INLINE_ROUND_DBL: bool = false;
    /// `CV_INLINE_ROUND_FLT` — likewise for `cvRound(float)`.
    pub const CV_INLINE_ROUND_FLT: bool = false;
    /// `CV_INLINE_ISNAN_DBL` — is `cvIsNaN(double)` inlined from a builtin?
    pub const CV_INLINE_ISNAN_DBL: bool = false;
    /// `CV_INLINE_ISNAN_FLT`.
    pub const CV_INLINE_ISNAN_FLT: bool = false;
    /// `CV_INLINE_ISINF_DBL` — is `cvIsInf(double)` inlined from a builtin?
    pub const CV_INLINE_ISINF_DBL: bool = false;
    /// `CV_INLINE_ISINF_FLT`.
    pub const CV_INLINE_ISINF_FLT: bool = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dispatch_switches_describe_a_single_build() {
        const _: () = assert!(!CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY);
        assert_eq!(CV_CPU_OPTIMIZATION_NAMESPACE, "baseline");
        assert!(CV_CPU_DISPATCH.is_empty());
        assert!(CV_AVX_GUARD.is_empty());
        // Consistent with the SIMD width constants next door: no dispatch, and
        // baseline mode on.
        assert_eq!(crate::hal::simd::CV_SIMD, 0);
        assert_eq!(crate::hal::simd::CV_CPU_BASELINE_MODE, 1);
    }

    #[test]
    fn the_architecture_flags_follow_the_target() {
        assert_eq!(CV_NEON_AARCH64, i32::from(cfg!(target_arch = "aarch64")));
        assert_eq!(CV_RVV_MAX_VLEN, 1024);
        const _: () = assert!(!CV_SIMD_NATIVE_MM_EXTRACT_EPI64);
    }

    #[test]
    fn no_fast_math_builtin_is_claimed() {
        use fast_math::*;
        let switches = [
            CV_FASTMATH_ENABLE_GCC_MATH_BUILTINS,
            CV_FASTMATH_ENABLE_CLANG_MATH_BUILTINS,
            CV_INLINE_ROUND_DBL,
            CV_INLINE_ROUND_FLT,
            CV_INLINE_ISNAN_DBL,
            CV_INLINE_ISNAN_FLT,
            CV_INLINE_ISINF_DBL,
            CV_INLINE_ISINF_FLT,
        ];
        assert!(switches.iter().all(|b| !b), "no builtin is claimed");
        // And the portable path they select is the one that runs: half-to-even.
        assert_eq!(crate::cv_round(0.5), 0);
        assert_eq!(crate::cv_round(1.5), 2);
        assert!(crate::cv_is_nan(f64::NAN));
        assert!(crate::cv_is_inf(f64::INFINITY));
    }
}
