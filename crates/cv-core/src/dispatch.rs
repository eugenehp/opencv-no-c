//! The build-configuration and CPU-dispatch macros
//! (`modules/core/include/opencv2/core/cvdef.h`, `cv_cpu_dispatch.h`,
//! `cv_cpu_helper.h`).
//!
//! # What these are for
//!
//! OpenCV ships one binary that runs on every x86 from 2006 onward. It does
//! that by compiling the hot loops several times — once per instruction set —
//! and choosing at *runtime* which copy to call. This family is the plumbing:
//! `CV_AVX2` says the build has an AVX2 copy, `CV_CPU_HAS_SUPPORT_AVX2` says
//! this machine can run it, and `CV_CPU_CALL_AVX2` is the dispatch itself.
//!
//! Rust needs much less of this. `cfg!(target_feature = "avx2")` answers the
//! compile-time question directly, and
//! [`cv_core::check_hardware_support`](crate::utility::check_hardware_support)
//! answers the runtime one. So each macro here reports the truth about *this*
//! build rather than a value baked in by a CMake flag — which is the useful
//! answer, and the one a caller porting a dispatch site actually needs.
//!
//! # The three families
//!
//! - `CV_<FEATURE>` — was a copy compiled for this instruction set?
//! - `CV_TRY_<FEATURE>` — may the dispatcher attempt it? Same thing here,
//!   since there is no separate baseline-versus-dispatch split.
//! - `CV_CPU_HAS_SUPPORT_<FEATURE>` — can the current machine run it? This one
//!   is a runtime query, and the only one of the three that can differ between
//!   two runs of the same binary.

use crate::utility::{check_hardware_support, CpuFeature};

// --- Compile-time: was a copy built for this instruction set? ---

/// `CV_AVX` — is a `AVX` code path compiled into this build?
#[must_use]
pub fn cv_avx() -> bool {
    cfg!(target_feature = "avx")
}

/// `CV_AVX2` — is a `AVX2` code path compiled into this build?
#[must_use]
pub fn cv_avx2() -> bool {
    cfg!(target_feature = "avx2")
}

/// `CV_AVX512_CLX` — is a `AVX512_CLX` code path compiled into this build?
#[must_use]
pub fn cv_avx512_clx() -> bool {
    false
}

/// `CV_AVX512_CNL` — is a `AVX512_CNL` code path compiled into this build?
#[must_use]
pub fn cv_avx512_cnl() -> bool {
    false
}

/// `CV_AVX512_COMMON` — is a `AVX512_COMMON` code path compiled into this build?
#[must_use]
pub fn cv_avx512_common() -> bool {
    false
}

/// `CV_AVX512_ICL` — is a `AVX512_ICL` code path compiled into this build?
#[must_use]
pub fn cv_avx512_icl() -> bool {
    false
}

/// `CV_AVX512_KNL` — is a `AVX512_KNL` code path compiled into this build?
#[must_use]
pub fn cv_avx512_knl() -> bool {
    false
}

/// `CV_AVX512_KNM` — is a `AVX512_KNM` code path compiled into this build?
#[must_use]
pub fn cv_avx512_knm() -> bool {
    false
}

/// `CV_AVX512_SKX` — is a `AVX512_SKX` code path compiled into this build?
#[must_use]
pub fn cv_avx512_skx() -> bool {
    false
}

/// `CV_AVX_5124FMAPS` — is a `AVX_5124FMAPS` code path compiled into this build?
#[must_use]
pub fn cv_avx_5124fmaps() -> bool {
    false
}

/// `CV_AVX_5124VNNIW` — is a `AVX_5124VNNIW` code path compiled into this build?
#[must_use]
pub fn cv_avx_5124vnniw() -> bool {
    false
}

/// `CV_AVX_512BITALG` — is a `AVX_512BITALG` code path compiled into this build?
#[must_use]
pub fn cv_avx_512bitalg() -> bool {
    cfg!(target_feature = "avx512bitalg")
}

/// `CV_AVX_512BW` — is a `AVX_512BW` code path compiled into this build?
#[must_use]
pub fn cv_avx_512bw() -> bool {
    cfg!(target_feature = "avx512bw")
}

/// `CV_AVX_512CD` — is a `AVX_512CD` code path compiled into this build?
#[must_use]
pub fn cv_avx_512cd() -> bool {
    cfg!(target_feature = "avx512cd")
}

/// `CV_AVX_512DQ` — is a `AVX_512DQ` code path compiled into this build?
#[must_use]
pub fn cv_avx_512dq() -> bool {
    cfg!(target_feature = "avx512dq")
}

/// `CV_AVX_512ER` — is a `AVX_512ER` code path compiled into this build?
#[must_use]
pub fn cv_avx_512er() -> bool {
    false
}

/// `CV_AVX_512F` — is a `AVX_512F` code path compiled into this build?
#[must_use]
pub fn cv_avx_512f() -> bool {
    cfg!(target_feature = "avx512f")
}

/// `CV_AVX_512IFMA` — is a `AVX_512IFMA` code path compiled into this build?
#[must_use]
pub fn cv_avx_512ifma() -> bool {
    cfg!(target_feature = "avx512ifma")
}

/// `CV_AVX_512PF` — is a `AVX_512PF` code path compiled into this build?
#[must_use]
pub fn cv_avx_512pf() -> bool {
    false
}

/// `CV_AVX_512VBMI` — is a `AVX_512VBMI` code path compiled into this build?
#[must_use]
pub fn cv_avx_512vbmi() -> bool {
    cfg!(target_feature = "avx512vbmi")
}

/// `CV_AVX_512VBMI2` — is a `AVX_512VBMI2` code path compiled into this build?
#[must_use]
pub fn cv_avx_512vbmi2() -> bool {
    cfg!(target_feature = "avx512vbmi2")
}

/// `CV_AVX_512VL` — is a `AVX_512VL` code path compiled into this build?
#[must_use]
pub fn cv_avx_512vl() -> bool {
    cfg!(target_feature = "avx512vl")
}

/// `CV_AVX_512VNNI` — is a `AVX_512VNNI` code path compiled into this build?
#[must_use]
pub fn cv_avx_512vnni() -> bool {
    cfg!(target_feature = "avx512vnni")
}

/// `CV_AVX_512VPOPCNTDQ` — is a `AVX_512VPOPCNTDQ` code path compiled into this build?
#[must_use]
pub fn cv_avx_512vpopcntdq() -> bool {
    cfg!(target_feature = "avx512vpopcntdq")
}

/// `CV_FMA3` — is a `FMA3` code path compiled into this build?
#[must_use]
pub fn cv_fma3() -> bool {
    cfg!(target_feature = "fma")
}

/// `CV_FP16` — is a `FP16` code path compiled into this build?
#[must_use]
pub fn cv_fp16() -> bool {
    cfg!(target_feature = "fp16")
}

/// `CV_LASX` — is a `LASX` code path compiled into this build?
#[must_use]
pub fn cv_lasx() -> bool {
    false
}

/// `CV_LSX` — is a `LSX` code path compiled into this build?
#[must_use]
pub fn cv_lsx() -> bool {
    false
}

/// `CV_MMX` — is a `MMX` code path compiled into this build?
#[must_use]
pub fn cv_mmx() -> bool {
    false
}

/// `CV_MSA` — is a `MSA` code path compiled into this build?
#[must_use]
pub fn cv_msa() -> bool {
    false
}

/// `CV_NEON` — is a `NEON` code path compiled into this build?
#[must_use]
pub fn cv_neon() -> bool {
    cfg!(target_feature = "neon")
}

/// `CV_NEON_DOT` — is a `NEON_DOT` code path compiled into this build?
#[must_use]
pub fn cv_neon_dot() -> bool {
    false
}

/// `CV_POPCNT` — is a `POPCNT` code path compiled into this build?
#[must_use]
pub fn cv_popcnt() -> bool {
    cfg!(target_feature = "popcnt")
}

/// `CV_RVV` — is a `RVV` code path compiled into this build?
#[must_use]
pub fn cv_rvv() -> bool {
    false
}

/// `CV_RVV071` — is a `RVV071` code path compiled into this build?
#[must_use]
pub fn cv_rvv071() -> bool {
    false
}

/// `CV_SSE` — is a `SSE` code path compiled into this build?
#[must_use]
pub fn cv_sse() -> bool {
    cfg!(target_feature = "sse")
}

/// `CV_SSE2` — is a `SSE2` code path compiled into this build?
#[must_use]
pub fn cv_sse2() -> bool {
    cfg!(target_feature = "sse2")
}

/// `CV_SSE3` — is a `SSE3` code path compiled into this build?
#[must_use]
pub fn cv_sse3() -> bool {
    cfg!(target_feature = "sse3")
}

/// `CV_SSE4_1` — is a `SSE4_1` code path compiled into this build?
#[must_use]
pub fn cv_sse4_1() -> bool {
    cfg!(target_feature = "sse4.1")
}

/// `CV_SSE4_2` — is a `SSE4_2` code path compiled into this build?
#[must_use]
pub fn cv_sse4_2() -> bool {
    cfg!(target_feature = "sse4.2")
}

/// `CV_SSSE3` — is a `SSSE3` code path compiled into this build?
#[must_use]
pub fn cv_ssse3() -> bool {
    cfg!(target_feature = "ssse3")
}

/// `CV_SVE` — is a `SVE` code path compiled into this build?
#[must_use]
pub fn cv_sve() -> bool {
    cfg!(target_feature = "sve")
}

/// `CV_VSX` — is a `VSX` code path compiled into this build?
#[must_use]
pub fn cv_vsx() -> bool {
    cfg!(target_feature = "vsx")
}

/// `CV_VSX3` — is a `VSX3` code path compiled into this build?
#[must_use]
pub fn cv_vsx3() -> bool {
    false
}

/// `CV_WASM_SIMD` — is a `WASM_SIMD` code path compiled into this build?
#[must_use]
pub fn cv_wasm_simd() -> bool {
    false
}

// --- Dispatch: may the dispatcher attempt this path? ---

/// `CV_TRY_AVX` — may the dispatcher attempt the `AVX` path?
#[must_use]
pub fn cv_try_avx() -> bool {
    cfg!(target_feature = "avx")
}

/// `CV_CPU_FORCE_AVX` — is `AVX` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_avx() -> bool {
    cfg!(target_feature = "avx")
}

/// `CV_TRY_AVX2` — may the dispatcher attempt the `AVX2` path?
#[must_use]
pub fn cv_try_avx2() -> bool {
    cfg!(target_feature = "avx2")
}

/// `CV_CPU_FORCE_AVX2` — is `AVX2` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_avx2() -> bool {
    cfg!(target_feature = "avx2")
}

/// `CV_TRY_AVX512_CLX` — may the dispatcher attempt the `AVX512_CLX` path?
#[must_use]
pub fn cv_try_avx512_clx() -> bool {
    false
}

/// `CV_CPU_FORCE_AVX512_CLX` — is `AVX512_CLX` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_avx512_clx() -> bool {
    false
}

/// `CV_TRY_AVX512_CNL` — may the dispatcher attempt the `AVX512_CNL` path?
#[must_use]
pub fn cv_try_avx512_cnl() -> bool {
    false
}

/// `CV_CPU_FORCE_AVX512_CNL` — is `AVX512_CNL` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_avx512_cnl() -> bool {
    false
}

/// `CV_TRY_AVX512_COMMON` — may the dispatcher attempt the `AVX512_COMMON` path?
#[must_use]
pub fn cv_try_avx512_common() -> bool {
    false
}

/// `CV_CPU_FORCE_AVX512_COMMON` — is `AVX512_COMMON` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_avx512_common() -> bool {
    false
}

/// `CV_TRY_AVX512_ICL` — may the dispatcher attempt the `AVX512_ICL` path?
#[must_use]
pub fn cv_try_avx512_icl() -> bool {
    false
}

/// `CV_CPU_FORCE_AVX512_ICL` — is `AVX512_ICL` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_avx512_icl() -> bool {
    false
}

/// `CV_TRY_AVX512_KNL` — may the dispatcher attempt the `AVX512_KNL` path?
#[must_use]
pub fn cv_try_avx512_knl() -> bool {
    false
}

/// `CV_CPU_FORCE_AVX512_KNL` — is `AVX512_KNL` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_avx512_knl() -> bool {
    false
}

/// `CV_TRY_AVX512_KNM` — may the dispatcher attempt the `AVX512_KNM` path?
#[must_use]
pub fn cv_try_avx512_knm() -> bool {
    false
}

/// `CV_CPU_FORCE_AVX512_KNM` — is `AVX512_KNM` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_avx512_knm() -> bool {
    false
}

/// `CV_TRY_AVX512_SKX` — may the dispatcher attempt the `AVX512_SKX` path?
#[must_use]
pub fn cv_try_avx512_skx() -> bool {
    false
}

/// `CV_CPU_FORCE_AVX512_SKX` — is `AVX512_SKX` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_avx512_skx() -> bool {
    false
}

/// `CV_TRY_AVX_512F` — may the dispatcher attempt the `AVX_512F` path?
#[must_use]
pub fn cv_try_avx_512f() -> bool {
    cfg!(target_feature = "avx512f")
}

/// `CV_CPU_FORCE_AVX_512F` — is `AVX_512F` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_avx_512f() -> bool {
    cfg!(target_feature = "avx512f")
}

/// `CV_TRY_FMA3` — may the dispatcher attempt the `FMA3` path?
#[must_use]
pub fn cv_try_fma3() -> bool {
    cfg!(target_feature = "fma")
}

/// `CV_CPU_FORCE_FMA3` — is `FMA3` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_fma3() -> bool {
    cfg!(target_feature = "fma")
}

/// `CV_TRY_FP16` — may the dispatcher attempt the `FP16` path?
#[must_use]
pub fn cv_try_fp16() -> bool {
    cfg!(target_feature = "fp16")
}

/// `CV_CPU_FORCE_FP16` — is `FP16` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_fp16() -> bool {
    cfg!(target_feature = "fp16")
}

/// `CV_TRY_LASX` — may the dispatcher attempt the `LASX` path?
#[must_use]
pub fn cv_try_lasx() -> bool {
    false
}

/// `CV_CPU_FORCE_LASX` — is `LASX` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_lasx() -> bool {
    false
}

/// `CV_TRY_LSX` — may the dispatcher attempt the `LSX` path?
#[must_use]
pub fn cv_try_lsx() -> bool {
    false
}

/// `CV_CPU_FORCE_LSX` — is `LSX` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_lsx() -> bool {
    false
}

/// `CV_TRY_MSA` — may the dispatcher attempt the `MSA` path?
#[must_use]
pub fn cv_try_msa() -> bool {
    false
}

/// `CV_CPU_FORCE_MSA` — is `MSA` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_msa() -> bool {
    false
}

/// `CV_TRY_NEON` — may the dispatcher attempt the `NEON` path?
#[must_use]
pub fn cv_try_neon() -> bool {
    cfg!(target_feature = "neon")
}

/// `CV_CPU_FORCE_NEON` — is `NEON` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_neon() -> bool {
    cfg!(target_feature = "neon")
}

/// `CV_TRY_NEON_BF16` — may the dispatcher attempt the `NEON_BF16` path?
#[must_use]
pub fn cv_try_neon_bf16() -> bool {
    false
}

/// `CV_CPU_FORCE_NEON_BF16` — is `NEON_BF16` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_neon_bf16() -> bool {
    false
}

/// `CV_TRY_NEON_DOTPROD` — may the dispatcher attempt the `NEON_DOTPROD` path?
#[must_use]
pub fn cv_try_neon_dotprod() -> bool {
    false
}

/// `CV_CPU_FORCE_NEON_DOTPROD` — is `NEON_DOTPROD` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_neon_dotprod() -> bool {
    false
}

/// `CV_TRY_NEON_FP16` — may the dispatcher attempt the `NEON_FP16` path?
#[must_use]
pub fn cv_try_neon_fp16() -> bool {
    false
}

/// `CV_CPU_FORCE_NEON_FP16` — is `NEON_FP16` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_neon_fp16() -> bool {
    false
}

/// `CV_TRY_POPCNT` — may the dispatcher attempt the `POPCNT` path?
#[must_use]
pub fn cv_try_popcnt() -> bool {
    cfg!(target_feature = "popcnt")
}

/// `CV_CPU_FORCE_POPCNT` — is `POPCNT` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_popcnt() -> bool {
    cfg!(target_feature = "popcnt")
}

/// `CV_TRY_RVV` — may the dispatcher attempt the `RVV` path?
#[must_use]
pub fn cv_try_rvv() -> bool {
    false
}

/// `CV_CPU_FORCE_RVV` — is `RVV` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_rvv() -> bool {
    false
}

/// `CV_TRY_SSE` — may the dispatcher attempt the `SSE` path?
#[must_use]
pub fn cv_try_sse() -> bool {
    cfg!(target_feature = "sse")
}

/// `CV_CPU_FORCE_SSE` — is `SSE` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_sse() -> bool {
    cfg!(target_feature = "sse")
}

/// `CV_TRY_SSE2` — may the dispatcher attempt the `SSE2` path?
#[must_use]
pub fn cv_try_sse2() -> bool {
    cfg!(target_feature = "sse2")
}

/// `CV_CPU_FORCE_SSE2` — is `SSE2` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_sse2() -> bool {
    cfg!(target_feature = "sse2")
}

/// `CV_TRY_SSE3` — may the dispatcher attempt the `SSE3` path?
#[must_use]
pub fn cv_try_sse3() -> bool {
    cfg!(target_feature = "sse3")
}

/// `CV_CPU_FORCE_SSE3` — is `SSE3` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_sse3() -> bool {
    cfg!(target_feature = "sse3")
}

/// `CV_TRY_SSE4_1` — may the dispatcher attempt the `SSE4_1` path?
#[must_use]
pub fn cv_try_sse4_1() -> bool {
    cfg!(target_feature = "sse4.1")
}

/// `CV_CPU_FORCE_SSE4_1` — is `SSE4_1` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_sse4_1() -> bool {
    cfg!(target_feature = "sse4.1")
}

/// `CV_TRY_SSE4_2` — may the dispatcher attempt the `SSE4_2` path?
#[must_use]
pub fn cv_try_sse4_2() -> bool {
    cfg!(target_feature = "sse4.2")
}

/// `CV_CPU_FORCE_SSE4_2` — is `SSE4_2` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_sse4_2() -> bool {
    cfg!(target_feature = "sse4.2")
}

/// `CV_TRY_SSSE3` — may the dispatcher attempt the `SSSE3` path?
#[must_use]
pub fn cv_try_ssse3() -> bool {
    cfg!(target_feature = "ssse3")
}

/// `CV_CPU_FORCE_SSSE3` — is `SSSE3` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_ssse3() -> bool {
    cfg!(target_feature = "ssse3")
}

/// `CV_TRY_SVE` — may the dispatcher attempt the `SVE` path?
#[must_use]
pub fn cv_try_sve() -> bool {
    cfg!(target_feature = "sve")
}

/// `CV_CPU_FORCE_SVE` — is `SVE` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_sve() -> bool {
    cfg!(target_feature = "sve")
}

/// `CV_TRY_VSX` — may the dispatcher attempt the `VSX` path?
#[must_use]
pub fn cv_try_vsx() -> bool {
    cfg!(target_feature = "vsx")
}

/// `CV_CPU_FORCE_VSX` — is `VSX` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_vsx() -> bool {
    cfg!(target_feature = "vsx")
}

/// `CV_TRY_VSX3` — may the dispatcher attempt the `VSX3` path?
#[must_use]
pub fn cv_try_vsx3() -> bool {
    false
}

/// `CV_CPU_FORCE_VSX3` — is `VSX3` the forced baseline, with no runtime check?
#[must_use]
pub fn cv_cpu_force_vsx3() -> bool {
    false
}

// --- Runtime: can this machine run it? ---

/// `CV_CPU_HAS_SUPPORT_AVX` — can the running machine execute `AVX`?
#[must_use]
pub fn cv_cpu_has_support_avx() -> bool {
    check_hardware_support(CpuFeature::Avx)
}

/// `CV_CPU_CALL_AVX(fn, args)` — dispatch to the `AVX` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_avx() -> bool {
    cv_try_avx() && cv_cpu_has_support_avx()
}

/// `CV_CPU_HAS_SUPPORT_AVX2` — can the running machine execute `AVX2`?
#[must_use]
pub fn cv_cpu_has_support_avx2() -> bool {
    check_hardware_support(CpuFeature::Avx2)
}

/// `CV_CPU_CALL_AVX2(fn, args)` — dispatch to the `AVX2` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_avx2() -> bool {
    cv_try_avx2() && cv_cpu_has_support_avx2()
}

/// `CV_CPU_HAS_SUPPORT_AVX512_CLX` — can the running machine execute `AVX512_CLX`?
#[must_use]
pub fn cv_cpu_has_support_avx512_clx() -> bool {
    check_hardware_support(CpuFeature::Avx512Clx)
}

/// `CV_CPU_CALL_AVX512_CLX(fn, args)` — dispatch to the `AVX512_CLX` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_avx512_clx() -> bool {
    cv_try_avx512_clx() && cv_cpu_has_support_avx512_clx()
}

/// `CV_CPU_HAS_SUPPORT_AVX512_CNL` — can the running machine execute `AVX512_CNL`?
#[must_use]
pub fn cv_cpu_has_support_avx512_cnl() -> bool {
    check_hardware_support(CpuFeature::Avx512Cnl)
}

/// `CV_CPU_CALL_AVX512_CNL(fn, args)` — dispatch to the `AVX512_CNL` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_avx512_cnl() -> bool {
    cv_try_avx512_cnl() && cv_cpu_has_support_avx512_cnl()
}

/// `CV_CPU_HAS_SUPPORT_AVX512_COMMON` — can the running machine execute `AVX512_COMMON`?
#[must_use]
pub fn cv_cpu_has_support_avx512_common() -> bool {
    check_hardware_support(CpuFeature::Avx512Common)
}

/// `CV_CPU_CALL_AVX512_COMMON(fn, args)` — dispatch to the `AVX512_COMMON` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_avx512_common() -> bool {
    cv_try_avx512_common() && cv_cpu_has_support_avx512_common()
}

/// `CV_CPU_HAS_SUPPORT_AVX512_ICL` — can the running machine execute `AVX512_ICL`?
#[must_use]
pub fn cv_cpu_has_support_avx512_icl() -> bool {
    check_hardware_support(CpuFeature::Avx512Icl)
}

/// `CV_CPU_CALL_AVX512_ICL(fn, args)` — dispatch to the `AVX512_ICL` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_avx512_icl() -> bool {
    cv_try_avx512_icl() && cv_cpu_has_support_avx512_icl()
}

/// `CV_CPU_HAS_SUPPORT_AVX512_KNL` — can the running machine execute `AVX512_KNL`?
#[must_use]
pub fn cv_cpu_has_support_avx512_knl() -> bool {
    check_hardware_support(CpuFeature::Avx512Knl)
}

/// `CV_CPU_CALL_AVX512_KNL(fn, args)` — dispatch to the `AVX512_KNL` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_avx512_knl() -> bool {
    cv_try_avx512_knl() && cv_cpu_has_support_avx512_knl()
}

/// `CV_CPU_HAS_SUPPORT_AVX512_KNM` — can the running machine execute `AVX512_KNM`?
#[must_use]
pub fn cv_cpu_has_support_avx512_knm() -> bool {
    check_hardware_support(CpuFeature::Avx512Knm)
}

/// `CV_CPU_CALL_AVX512_KNM(fn, args)` — dispatch to the `AVX512_KNM` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_avx512_knm() -> bool {
    cv_try_avx512_knm() && cv_cpu_has_support_avx512_knm()
}

/// `CV_CPU_HAS_SUPPORT_AVX512_SKX` — can the running machine execute `AVX512_SKX`?
#[must_use]
pub fn cv_cpu_has_support_avx512_skx() -> bool {
    check_hardware_support(CpuFeature::Avx512Skx)
}

/// `CV_CPU_CALL_AVX512_SKX(fn, args)` — dispatch to the `AVX512_SKX` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_avx512_skx() -> bool {
    cv_try_avx512_skx() && cv_cpu_has_support_avx512_skx()
}

/// `CV_CPU_HAS_SUPPORT_AVX_512F` — can the running machine execute `AVX_512F`?
#[must_use]
pub fn cv_cpu_has_support_avx_512f() -> bool {
    check_hardware_support(CpuFeature::Avx512f)
}

/// `CV_CPU_CALL_AVX_512F(fn, args)` — dispatch to the `AVX_512F` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_avx_512f() -> bool {
    cv_try_avx_512f() && cv_cpu_has_support_avx_512f()
}

/// `CV_CPU_HAS_SUPPORT_FMA3` — can the running machine execute `FMA3`?
#[must_use]
pub fn cv_cpu_has_support_fma3() -> bool {
    check_hardware_support(CpuFeature::Fma3)
}

/// `CV_CPU_CALL_FMA3(fn, args)` — dispatch to the `FMA3` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_fma3() -> bool {
    cv_try_fma3() && cv_cpu_has_support_fma3()
}

/// `CV_CPU_HAS_SUPPORT_FP16` — can the running machine execute `FP16`?
#[must_use]
pub fn cv_cpu_has_support_fp16() -> bool {
    check_hardware_support(CpuFeature::Fp16)
}

/// `CV_CPU_CALL_FP16(fn, args)` — dispatch to the `FP16` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_fp16() -> bool {
    cv_try_fp16() && cv_cpu_has_support_fp16()
}

/// `CV_CPU_HAS_SUPPORT_LASX` — can the running machine execute `LASX`?
#[must_use]
pub fn cv_cpu_has_support_lasx() -> bool {
    check_hardware_support(CpuFeature::Lasx)
}

/// `CV_CPU_CALL_LASX(fn, args)` — dispatch to the `LASX` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_lasx() -> bool {
    cv_try_lasx() && cv_cpu_has_support_lasx()
}

/// `CV_CPU_HAS_SUPPORT_LSX` — can the running machine execute `LSX`?
#[must_use]
pub fn cv_cpu_has_support_lsx() -> bool {
    check_hardware_support(CpuFeature::Lsx)
}

/// `CV_CPU_CALL_LSX(fn, args)` — dispatch to the `LSX` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_lsx() -> bool {
    cv_try_lsx() && cv_cpu_has_support_lsx()
}

/// `CV_CPU_HAS_SUPPORT_MSA` — can the running machine execute `MSA`?
#[must_use]
pub fn cv_cpu_has_support_msa() -> bool {
    check_hardware_support(CpuFeature::Msa)
}

/// `CV_CPU_CALL_MSA(fn, args)` — dispatch to the `MSA` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_msa() -> bool {
    cv_try_msa() && cv_cpu_has_support_msa()
}

/// `CV_CPU_HAS_SUPPORT_NEON` — can the running machine execute `NEON`?
#[must_use]
pub fn cv_cpu_has_support_neon() -> bool {
    check_hardware_support(CpuFeature::Neon)
}

/// `CV_CPU_CALL_NEON(fn, args)` — dispatch to the `NEON` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_neon() -> bool {
    cv_try_neon() && cv_cpu_has_support_neon()
}

/// `CV_CPU_HAS_SUPPORT_NEON_BF16` — can the running machine execute `NEON_BF16`?
#[must_use]
pub fn cv_cpu_has_support_neon_bf16() -> bool {
    check_hardware_support(CpuFeature::NeonBf16)
}

/// `CV_CPU_CALL_NEON_BF16(fn, args)` — dispatch to the `NEON_BF16` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_neon_bf16() -> bool {
    cv_try_neon_bf16() && cv_cpu_has_support_neon_bf16()
}

/// `CV_CPU_HAS_SUPPORT_NEON_DOTPROD` — can the running machine execute `NEON_DOTPROD`?
#[must_use]
pub fn cv_cpu_has_support_neon_dotprod() -> bool {
    check_hardware_support(CpuFeature::NeonDotprod)
}

/// `CV_CPU_CALL_NEON_DOTPROD(fn, args)` — dispatch to the `NEON_DOTPROD` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_neon_dotprod() -> bool {
    cv_try_neon_dotprod() && cv_cpu_has_support_neon_dotprod()
}

/// `CV_CPU_HAS_SUPPORT_NEON_FP16` — can the running machine execute `NEON_FP16`?
#[must_use]
pub fn cv_cpu_has_support_neon_fp16() -> bool {
    check_hardware_support(CpuFeature::NeonFp16)
}

/// `CV_CPU_CALL_NEON_FP16(fn, args)` — dispatch to the `NEON_FP16` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_neon_fp16() -> bool {
    cv_try_neon_fp16() && cv_cpu_has_support_neon_fp16()
}

/// `CV_CPU_HAS_SUPPORT_POPCNT` — can the running machine execute `POPCNT`?
#[must_use]
pub fn cv_cpu_has_support_popcnt() -> bool {
    check_hardware_support(CpuFeature::Popcnt)
}

/// `CV_CPU_CALL_POPCNT(fn, args)` — dispatch to the `POPCNT` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_popcnt() -> bool {
    cv_try_popcnt() && cv_cpu_has_support_popcnt()
}

/// `CV_CPU_HAS_SUPPORT_RVV` — can the running machine execute `RVV`?
#[must_use]
pub fn cv_cpu_has_support_rvv() -> bool {
    check_hardware_support(CpuFeature::Rvv)
}

/// `CV_CPU_CALL_RVV(fn, args)` — dispatch to the `RVV` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_rvv() -> bool {
    cv_try_rvv() && cv_cpu_has_support_rvv()
}

/// `CV_CPU_HAS_SUPPORT_SSE` — can the running machine execute `SSE`?
#[must_use]
pub fn cv_cpu_has_support_sse() -> bool {
    check_hardware_support(CpuFeature::Sse)
}

/// `CV_CPU_CALL_SSE(fn, args)` — dispatch to the `SSE` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_sse() -> bool {
    cv_try_sse() && cv_cpu_has_support_sse()
}

/// `CV_CPU_HAS_SUPPORT_SSE2` — can the running machine execute `SSE2`?
#[must_use]
pub fn cv_cpu_has_support_sse2() -> bool {
    check_hardware_support(CpuFeature::Sse2)
}

/// `CV_CPU_CALL_SSE2(fn, args)` — dispatch to the `SSE2` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_sse2() -> bool {
    cv_try_sse2() && cv_cpu_has_support_sse2()
}

/// `CV_CPU_HAS_SUPPORT_SSE3` — can the running machine execute `SSE3`?
#[must_use]
pub fn cv_cpu_has_support_sse3() -> bool {
    check_hardware_support(CpuFeature::Sse3)
}

/// `CV_CPU_CALL_SSE3(fn, args)` — dispatch to the `SSE3` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_sse3() -> bool {
    cv_try_sse3() && cv_cpu_has_support_sse3()
}

/// `CV_CPU_HAS_SUPPORT_SSE4_1` — can the running machine execute `SSE4_1`?
#[must_use]
pub fn cv_cpu_has_support_sse4_1() -> bool {
    check_hardware_support(CpuFeature::Sse41)
}

/// `CV_CPU_CALL_SSE4_1(fn, args)` — dispatch to the `SSE4_1` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_sse4_1() -> bool {
    cv_try_sse4_1() && cv_cpu_has_support_sse4_1()
}

/// `CV_CPU_HAS_SUPPORT_SSE4_2` — can the running machine execute `SSE4_2`?
#[must_use]
pub fn cv_cpu_has_support_sse4_2() -> bool {
    check_hardware_support(CpuFeature::Sse42)
}

/// `CV_CPU_CALL_SSE4_2(fn, args)` — dispatch to the `SSE4_2` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_sse4_2() -> bool {
    cv_try_sse4_2() && cv_cpu_has_support_sse4_2()
}

/// `CV_CPU_HAS_SUPPORT_SSSE3` — can the running machine execute `SSSE3`?
#[must_use]
pub fn cv_cpu_has_support_ssse3() -> bool {
    check_hardware_support(CpuFeature::Ssse3)
}

/// `CV_CPU_CALL_SSSE3(fn, args)` — dispatch to the `SSSE3` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_ssse3() -> bool {
    cv_try_ssse3() && cv_cpu_has_support_ssse3()
}

/// `CV_CPU_HAS_SUPPORT_SVE` — can the running machine execute `SVE`?
#[must_use]
pub fn cv_cpu_has_support_sve() -> bool {
    check_hardware_support(CpuFeature::Sve)
}

/// `CV_CPU_CALL_SVE(fn, args)` — dispatch to the `SVE` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_sve() -> bool {
    cv_try_sve() && cv_cpu_has_support_sve()
}

/// `CV_CPU_HAS_SUPPORT_VSX` — can the running machine execute `VSX`?
#[must_use]
pub fn cv_cpu_has_support_vsx() -> bool {
    check_hardware_support(CpuFeature::Vsx)
}

/// `CV_CPU_CALL_VSX(fn, args)` — dispatch to the `VSX` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_vsx() -> bool {
    cv_try_vsx() && cv_cpu_has_support_vsx()
}

/// `CV_CPU_HAS_SUPPORT_VSX3` — can the running machine execute `VSX3`?
#[must_use]
pub fn cv_cpu_has_support_vsx3() -> bool {
    check_hardware_support(CpuFeature::Vsx3)
}

/// `CV_CPU_CALL_VSX3(fn, args)` — dispatch to the `VSX3` copy if both
/// the build has one and the machine can run it.
#[must_use]
pub fn cv_cpu_call_vsx3() -> bool {
    cv_try_vsx3() && cv_cpu_has_support_vsx3()
}


// ---------------------------------------------------------------------------
// C++ portability
// ---------------------------------------------------------------------------
//
// These exist because OpenCV supports compilers that predate the language
// features it wants. Rust has every one of them as a first-class construct, so
// each is reported as the Rust spelling rather than as a C++ token — which is
// what a reader porting a header actually needs to know.

/// `CV_CXX11` — is the compiler at least C++11?
///
/// Always: Rust has no pre-2011 dialect to guard against.
pub const CV_CXX11: bool = true;

/// `CV_CONSTEXPR` — Rust's `const fn`.
pub const CV_CONSTEXPR: &str = "const fn";

/// `CV_INLINE` — Rust's `#[inline]`.
pub const CV_INLINE: &str = "#[inline]";

/// `CV_FINAL` — C++'s `final`.
///
/// Empty: Rust has no inheritance to seal. A struct is final by construction.
pub const CV_FINAL: &str = "";

/// `CV_OVERRIDE` — C++'s `override`.
///
/// Empty for the same reason: implementing a method that the trait does not
/// declare is already a compile error, which is what `override` was added to
/// C++ to catch.
pub const CV_OVERRIDE: &str = "";

/// `CV_NOEXCEPT` — a function that cannot unwind.
pub const CV_NOEXCEPT: &str = "";

/// `CV_NORETURN` — Rust's never type.
pub const CV_NORETURN: &str = "-> !";

/// `CV_NODISCARD_STD` — Rust's `#[must_use]`.
pub const CV_NODISCARD_STD: &str = "#[must_use]";

/// `CV_DEPRECATED` — Rust's `#[deprecated]`.
pub const CV_DEPRECATED: &str = "#[deprecated]";

/// `CV_DEPRECATED_EXTERNAL` — deprecated for external callers only.
///
/// Rust has no such gradation; the attribute is the same either way.
pub const CV_DEPRECATED_EXTERNAL: &str = "#[deprecated]";

/// `CV_EXPORTS` — the symbol-visibility attribute.
///
/// Empty: `pub` is Rust's visibility, and it needs no platform spelling.
pub const CV_EXPORTS: &str = "pub";

/// `CV_EXPORTS_TEMPLATE` — the same for a template instantiation.
pub const CV_EXPORTS_TEMPLATE: &str = "pub";

/// `CV_PLUGIN_EXPORTS` — visibility for a dynamically-loaded plugin.
pub const CV_PLUGIN_EXPORTS: &str = "pub";

/// `CV_EXTERN_C` — C linkage.
pub const CV_EXTERN_C: &str = "extern \"C\"";

/// `CV_CDECL` — the C calling convention.
pub const CV_CDECL: &str = "extern \"C\"";

/// `CV_STDCALL` — the `stdcall` convention, on Windows.
pub const CV_STDCALL: &str = "extern \"stdcall\"";

/// `CV_DECL_ALIGNED(x)` — Rust's `#[repr(align(x))]`.
#[must_use]
pub fn cv_decl_aligned(bytes: usize) -> String {
    format!("#[repr(align({bytes}))]")
}

/// `CV_FORMAT_PRINTF(fmt, args)` — the printf-checking attribute.
///
/// Empty: Rust's `format!` is checked by the compiler, so there is nothing to
/// annotate.
pub const CV_FORMAT_PRINTF: &str = "";

/// `CV_ICC` — is this the Intel compiler?
pub const CV_ICC: bool = false;

/// `CV_ENABLE_UNROLLED` — does the build unroll its inner loops by hand?
///
/// False: hand-unrolling is a workaround for compilers that will not, and
/// `rustc` does it where it pays.
pub const CV_ENABLE_UNROLLED: bool = false;

/// `CV_STATIC_ANALYSIS` — is a static analyser compiling this?
pub const CV_STATIC_ANALYSIS: bool = cfg!(debug_assertions) && false;

/// `CV_DISABLE_UBSAN` — the attribute that exempts a function from the
/// undefined-behaviour sanitiser.
///
/// Empty, and it should stay that way: the code that needs it upstream is
/// doing something Rust will not let you write in the first place.
pub const CV_DISABLE_UBSAN: &str = "";

/// `CV_THREAD_SANITIZER` — is the thread sanitiser active?
///
/// False on stable Rust: `cfg(sanitize)` is unstable, so a stable build cannot
/// ask. Upstream uses this to relax races it knows the sanitiser will flag;
/// nothing here needs that relaxation.
pub const CV_THREAD_SANITIZER: bool = false;

/// `CV__EXCEPTION_PTR` — does the build have `std::exception_ptr`?
///
/// False, and not for lack of a feature: a Rust error is a value returned
/// through [`crate::Result`], never a stashed unwinding state.
pub const CV__EXCEPTION_PTR: bool = false;

/// `CV_BIG_INT(n)` — a 64-bit signed literal.
#[must_use]
pub const fn cv_big_int(n: i64) -> i64 {
    n
}

/// `CV_BIG_UINT(n)` — a 64-bit unsigned literal.
#[must_use]
pub const fn cv_big_uint(n: u64) -> u64 {
    n
}

/// `CV_FP16_TYPE` — does the compiler have a native half-precision type?
///
/// False: `f16` is unstable in Rust, so [`crate::softfloat`] carries the
/// conversions instead.
pub const CV_FP16_TYPE: bool = false;

/// `CV_POPCNT_U32(x)` — population count of a 32-bit word.
#[must_use]
pub const fn cv_popcnt_u32(x: u32) -> u32 {
    x.count_ones()
}

/// `CV_POPCNT_U64(x)` — population count of a 64-bit word.
#[must_use]
pub const fn cv_popcnt_u64(x: u64) -> u32 {
    x.count_ones()
}

/// `CV_XADD(addr, delta)` — atomic fetch-and-add, as the refcount uses.
pub fn cv_xadd(addr: &std::sync::atomic::AtomicI32, delta: i32) -> i32 {
    addr.fetch_add(delta, std::sync::atomic::Ordering::SeqCst)
}

/// `CV_StaticAssert(condition, message)` — a compile-time check.
///
/// Rust spells it `const _: () = assert!(...)`, which this returns; the check
/// itself has to be written at the site, because a runtime function cannot
/// fail at compile time.
#[must_use]
pub fn cv_static_assert(condition: bool, message: &str) -> String {
    format!("const _: () = assert!({condition}, \"{message}\");")
}

/// `CV_DbgAssert(expr)` — an assertion compiled out of a release build.
///
/// Rust's `debug_assert!` is the same idea; this reports whether it is active,
/// so a caller can tell which build they are in.
#[must_use]
pub fn cv_dbg_assert(condition: bool) -> bool {
    !cfg!(debug_assertions) || condition
}

/// `CV_LOG_STRIP_LEVEL` — the level at or below which logging is compiled out.
///
/// Upstream defaults to `LOG_LEVEL_DEBUG + 1` in a release build; the same
/// here, so a verbose log statement costs nothing when it is not wanted.
pub const CV_LOG_STRIP_LEVEL: i32 = 5;

/// `CV__CHECK_FILENAME` — the file a failing `CV_Check` reports.
#[must_use]
pub fn cv_check_filename(file: &str) -> &str {
    file
}

/// `CV__CHECK_FUNCTION` — the function it reports.
#[must_use]
pub fn cv_check_function(function: &str) -> &str {
    function
}

/// `CV_TRACE_FILENAME` — the file a trace region is attributed to.
#[must_use]
pub fn cv_trace_filename(file: &str) -> &str {
    file
}

/// `CV__TRACE_FUNCTION` — the function name a trace region records.
#[must_use]
pub fn cv_trace_function(function: &str) -> &str {
    function
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dispatch_triple_is_consistent() {
        // A path can only be *called* if the build has it and the machine runs
        // it. That conjunction is the whole dispatch rule.
        assert_eq!(cv_cpu_call_sse2(), cv_try_sse2() && cv_cpu_has_support_sse2());
        assert_eq!(cv_cpu_call_avx2(), cv_try_avx2() && cv_cpu_has_support_avx2());
        assert_eq!(cv_cpu_call_neon(), cv_try_neon() && cv_cpu_has_support_neon());
        // The compile-time answers agree between the two spellings.
        assert_eq!(cv_try_avx2(), cv_avx2());
        assert_eq!(cv_cpu_force_sse2(), cv_sse2());
    }

    #[test]
    fn the_compile_time_flags_report_this_build() {
        // Whatever the host is, the baseline for its architecture must be on.
        if cfg!(target_arch = "x86_64") {
            assert!(cv_sse2(), "SSE2 is part of the x86_64 baseline");
        }
        if cfg!(target_arch = "aarch64") {
            assert!(cv_neon(), "NEON is part of the aarch64 baseline");
        }
        // A path this build has no code for cannot be attempted.
        if !cv_avx512_skx() {
            assert!(!cv_cpu_call_avx512_skx());
        }
    }

    // These are compile-time constants, so assert them at compile time: a
    // runtime `assert!` on a `const bool` is a test that can never have failed
    // by the time it runs.
    const _: () = assert!(CV_CXX11, "there is no pre-2011 Rust");
    const _: () = assert!(!CV_ICC);
    const _: () = assert!(!CV_ENABLE_UNROLLED);
    const _: () = assert!(!CV_STATIC_ANALYSIS);
    const _: () = assert!(!CV__EXCEPTION_PTR);
    const _: () = assert!(!CV_FP16_TYPE);
    const _: () = assert!(!CV_THREAD_SANITIZER);
    const _: () = assert!(CV_LOG_STRIP_LEVEL == 5);

    #[test]
    fn the_portability_macros_carry_their_rust_spellings() {
        assert_eq!(CV_CONSTEXPR, "const fn");
        assert_eq!(CV_INLINE, "#[inline]");
        assert_eq!(CV_NODISCARD_STD, "#[must_use]");
        assert_eq!(CV_DEPRECATED, "#[deprecated]");
        assert_eq!(CV_DEPRECATED_EXTERNAL, CV_DEPRECATED);
        assert_eq!(CV_NORETURN, "-> !");
        // Rust needs none of these, and says so with an empty spelling.
        assert_eq!(CV_FINAL, "");
        assert_eq!(CV_OVERRIDE, "");
        assert_eq!(CV_NOEXCEPT, "");
        assert_eq!(CV_FORMAT_PRINTF, "");
        assert_eq!(CV_DISABLE_UBSAN, "");
        assert_eq!(CV_EXPORTS, "pub");
        assert_eq!(CV_EXPORTS_TEMPLATE, CV_EXPORTS);
        assert_eq!(CV_PLUGIN_EXPORTS, CV_EXPORTS);
        assert!(CV_EXTERN_C.contains("extern"));
        assert!(CV_STDCALL.contains("stdcall"));
        assert_eq!(cv_decl_aligned(64), "#[repr(align(64))]");
    }

    #[test]
    fn the_bit_and_atomic_helpers_compute() {
        assert_eq!(cv_popcnt_u32(0b1011), 3);
        assert_eq!(cv_popcnt_u32(0), 0);
        assert_eq!(cv_popcnt_u64(u64::MAX), 64);
        assert_eq!(cv_big_int(-9_000_000_000), -9_000_000_000);
        assert_eq!(cv_big_uint(18_000_000_000), 18_000_000_000);

        let counter = std::sync::atomic::AtomicI32::new(5);
        assert_eq!(cv_xadd(&counter, 3), 5, "the value *before* the add");
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 8);
        assert_eq!(cv_xadd(&counter, -8), 8);
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn the_assertion_and_trace_helpers_report_what_they_are_given() {
        assert!(cv_static_assert(true, "ok").contains("assert!"));
        assert!(cv_static_assert(true, "ok").contains("ok"));
        // A debug assertion holds trivially in a release build.
        assert!(cv_dbg_assert(true));
        assert_eq!(cv_dbg_assert(false), !cfg!(debug_assertions));
        assert_eq!(cv_check_filename("a.rs"), "a.rs");
        assert_eq!(cv_check_function("f"), "f");
        assert_eq!(cv_trace_filename("a.rs"), "a.rs");
        assert_eq!(cv_trace_function("f"), "f");
    }
}
