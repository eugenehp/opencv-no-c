//! Timing, threading and build introspection
//! (`modules/core/src/system.cpp`, `parallel.cpp`, `utils/logger.cpp`).
//!
//! These report on the port itself rather than transforming images, so several
//! answer differently from upstream by construction — [`get_build_information`]
//! describes a Rust workspace rather than a CMake configuration.
//!
//! [`check_hardware_support`] is the exception worth naming: it reports the
//! **host CPU**, as upstream documents, even though no kernel here dispatches
//! on the answer. A caller asking the question is usually deciding about its
//! own code, not ours.

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// `cv::getTickCount()` — a monotonic tick count.
///
/// The epoch is the first call in the process, not upstream's platform clock;
/// only differences are meaningful, which is all upstream promises either.
#[must_use]
pub fn get_tick_count() -> i64 {
    static START: OnceLock<Instant> = OnceLock::new();
    let s = START.get_or_init(Instant::now);
    s.elapsed().as_nanos() as i64
}

/// `cv::getTickFrequency()` — ticks per second, so `get_tick_count` divided by
/// this is seconds.
#[must_use]
pub fn get_tick_frequency() -> f64 {
    1e9
}

/// `cv::getCPUTickCount()`.
///
/// Upstream reads the cycle counter (`rdtsc`). Reaching that needs inline
/// assembly per architecture, so this returns the same nanosecond clock as
/// [`get_tick_count`] — monotonic and useful for deltas, but not a cycle count.
#[must_use]
pub fn get_cpu_tick_count() -> i64 {
    get_tick_count()
}

/// `cv::TickMeter` — accumulates timings across start/stop pairs.
#[derive(Debug, Clone)]
pub struct TickMeter {
    start: Option<Instant>,
    total_ns: u128,
    last_ns: u128,
    counter: i64,
}

impl Default for TickMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl TickMeter {
    /// A meter with nothing recorded yet.
    #[must_use]
    pub fn new() -> Self {
        Self { start: None, total_ns: 0, last_ns: 0, counter: 0 }
    }

    /// `TickMeter::start()`.
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    /// `TickMeter::stop()`.
    ///
    /// A stop without a matching start is ignored, as it is upstream, rather
    /// than recording a garbage interval.
    pub fn stop(&mut self) {
        if let Some(s) = self.start.take() {
            self.last_ns = s.elapsed().as_nanos();
            self.total_ns += self.last_ns;
            self.counter += 1;
        }
    }

    /// `TickMeter::reset()`.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// `TickMeter::getCounter()` — how many intervals were recorded.
    #[must_use]
    pub fn get_counter(&self) -> i64 {
        self.counter
    }

    /// `TickMeter::getTimeTicks()`.
    #[must_use]
    pub fn get_time_ticks(&self) -> i64 {
        self.total_ns as i64
    }

    /// `TickMeter::getTimeMicro()`.
    #[must_use]
    pub fn get_time_micro(&self) -> f64 {
        self.total_ns as f64 / 1e3
    }

    /// `TickMeter::getTimeMilli()`.
    #[must_use]
    pub fn get_time_milli(&self) -> f64 {
        self.total_ns as f64 / 1e6
    }

    /// `TickMeter::getTimeSec()`.
    #[must_use]
    pub fn get_time_sec(&self) -> f64 {
        self.total_ns as f64 / 1e9
    }

    /// `TickMeter::getAvgTimeSec()` — zero when nothing has been recorded.
    #[must_use]
    pub fn get_avg_time_sec(&self) -> f64 {
        if self.counter == 0 {
            0.0
        } else {
            self.get_time_sec() / self.counter as f64
        }
    }

    /// `TickMeter::getAvgTimeMilli()`.
    #[must_use]
    pub fn get_avg_time_milli(&self) -> f64 {
        self.get_avg_time_sec() * 1e3
    }

    /// `TickMeter::getFPS()` — the reciprocal of the average interval.
    #[must_use]
    pub fn get_fps(&self) -> f64 {
        let a = self.get_avg_time_sec();
        if a <= 0.0 {
            0.0
        } else {
            1.0 / a
        }
    }

    /// `TickMeter::getLastTimeTicks()`.
    #[must_use]
    pub fn get_last_time_ticks(&self) -> i64 {
        self.last_ns as i64
    }

    /// `TickMeter::getLastTimeMicro()`.
    #[must_use]
    pub fn get_last_time_micro(&self) -> f64 {
        self.last_ns as f64 / 1e3
    }

    /// `TickMeter::getLastTimeMilli()`.
    #[must_use]
    pub fn get_last_time_milli(&self) -> f64 {
        self.last_ns as f64 / 1e6
    }

    /// `TickMeter::getLastTimeSec()`.
    #[must_use]
    pub fn get_last_time_sec(&self) -> f64 {
        self.last_ns as f64 / 1e9
    }
}

static NUM_THREADS: AtomicI32 = AtomicI32::new(0);
static USE_OPTIMIZED: AtomicBool = AtomicBool::new(true);
static LOG_LEVEL: AtomicI32 = AtomicI32::new(3);

/// `cv::getNumberOfCPUs()`.
#[must_use]
pub fn get_number_of_cpus() -> i32 {
    std::thread::available_parallelism().map_or(1, |n| n.get() as i32)
}

/// `cv::getNumThreads()` — the cap set by [`set_num_threads`], or the CPU count.
#[must_use]
pub fn get_num_threads() -> i32 {
    let n = NUM_THREADS.load(Ordering::Relaxed);
    if n > 0 {
        n
    } else {
        get_number_of_cpus()
    }
}

/// `cv::setNumThreads(nthreads)`.
///
/// Zero or negative restores the default, matching upstream. This crate's
/// kernels are currently single-threaded, so the value is recorded and
/// reported but does not yet change how work is scheduled.
pub fn set_num_threads(nthreads: i32) {
    NUM_THREADS.store(nthreads.max(0), Ordering::Relaxed);
}

/// `cv::getThreadNum()` — always 0 here, since the kernels are single-threaded.
#[must_use]
pub fn get_thread_num() -> i32 {
    0
}

/// `cv::useOptimized()`.
#[must_use]
pub fn use_optimized() -> bool {
    USE_OPTIMIZED.load(Ordering::Relaxed)
}

/// `cv::setUseOptimized(onoff)`.
///
/// Upstream switches between SIMD and reference kernels. This port has one
/// scalar path chosen for exact parity, so the flag is recorded and reported
/// but selects nothing — which is deliberate: a second path would be a second
/// set of rounding behaviours to keep in agreement.
pub fn set_use_optimized(onoff: bool) {
    USE_OPTIMIZED.store(onoff, Ordering::Relaxed);
}

/// `cv::getVersionMajor()`.
#[must_use]
pub fn get_version_major() -> i32 {
    4
}

/// `cv::getVersionMinor()`.
#[must_use]
pub fn get_version_minor() -> i32 {
    15
}

/// `cv::getVersionRevision()`.
#[must_use]
pub fn get_version_revision() -> i32 {
    0
}

/// `cv::getVersionString()` — the upstream release this port tracks.
#[must_use]
pub fn get_version_string() -> String {
    format!("{}.{}.{}-dev", get_version_major(), get_version_minor(), get_version_revision())
}

/// `cv::getBuildInformation()`.
///
/// Describes this Rust workspace, not an OpenCV CMake configuration: the
/// fields upstream fills in are about a C++ build that does not exist here.
#[must_use]
pub fn get_build_information() -> String {
    format!(
        "\ngeneral configuration for opencv-no-c {}\n\
         \x20 Tracking OpenCV:             {}\n\
         \x20 Implementation:              pure Rust, no C or C++\n\
         \x20 Host:                        {} {}\n\
         \x20 CPUs available:              {}\n\
         \x20 Parallel framework:          none (single-threaded kernels)\n\
         \x20 SIMD dispatch:               none (scalar paths, for exact parity)\n",
        env!("CARGO_PKG_VERSION"),
        get_version_string(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        get_number_of_cpus(),
    )
}

/// `cv::ipp::useIPP()` — is Intel IPP acceleration in use?
///
/// Always false. IPP is a closed-source C library, which this port does not
/// link by construction; the query exists so code translated from C++ compiles
/// and takes its non-IPP branch.
#[must_use]
pub fn use_ipp() -> bool {
    false
}

/// `cv::ipp::setUseIPP(flag)` — accepted and ignored; see [`use_ipp`].
pub fn set_use_ipp(_flag: bool) {}

/// `cv::ipp::useIPP_NotExact()`.
///
/// Always false. Upstream's "not exact" mode allows IPP kernels whose results
/// differ from its own; this port has one implementation per function, chosen
/// for exact parity, so there is no inexact variant to enable.
#[must_use]
pub fn use_ipp_not_exact() -> bool {
    false
}

/// `cv::ipp::setUseIPP_NotExact(flag)` — accepted and ignored.
pub fn set_use_ipp_not_exact(_flag: bool) {}

/// `cv::ipp::getIppVersion()`.
#[must_use]
pub fn get_ipp_version() -> String {
    "disabled".to_string()
}

/// `cv::parallel::setParallelForBackend(backendName, propagateNumThreads)`.
///
/// Returns false for every name: the kernels here are single-threaded, so
/// there is no `parallel_for_` to redirect at TBB or OpenMP. Upstream also
/// returns false for a backend it was not built with.
#[must_use]
pub fn set_parallel_for_backend(_backend_name: &str, _propagate_num_threads: bool) -> bool {
    false
}

/// `cv::haveOpenVX()` — always false.
///
/// OpenVX is a C acceleration API, like IPP and OpenCL; this port links none of
/// them. The query exists so translated code takes its non-OpenVX branch.
#[must_use]
pub fn have_openvx() -> bool {
    false
}

/// `cv::useOpenVX()` — always false.
#[must_use]
pub fn use_openvx() -> bool {
    false
}

/// `cv::setUseOpenVX(flag)` — accepted and ignored; see [`have_openvx`].
pub fn set_use_openvx(_flag: bool) {}

/// `cv::ovx::getOpenVXContext()`.
///
/// `None`: there is no OpenVX runtime to hold a context.
#[must_use]
pub fn get_openvx_context() -> Option<&'static ()> {
    None
}

/// `cv::AlgorithmHint` — which of several implementations a function should pick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum AlgorithmHint {
    /// `ALGO_HINT_DEFAULT` — whatever [`get_default_algorithm_hint`] returns.
    #[default]
    Default = 0,
    /// `ALGO_HINT_ACCURATE` — prefer the exact implementation.
    Accurate = 1,
    /// `ALGO_HINT_APPROX` — accept an approximation for speed.
    Approx = 2,
}

/// `cv::getDefaultAlgorithmHint()`.
///
/// Always [`AlgorithmHint::Accurate`]. Upstream's default depends on a build
/// flag that trades accuracy for SIMD throughput; this port has one scalar
/// implementation per function and it is the accurate one, so there is nothing
/// for `ALGO_HINT_APPROX` to select.
#[must_use]
pub fn get_default_algorithm_hint() -> AlgorithmHint {
    AlgorithmHint::Accurate
}

/// `cv::CpuFeatures`, with upstream's numeric values.
///
/// The whole enumeration is present because a caller translating C++ needs the
/// names and values to line up, not just the subset this port could dispatch
/// on. [`check_hardware_support`] answers for every one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
#[allow(missing_docs)]
pub enum CpuFeature {
    /// `CPU_MMX` / `CV_CPU_MMX`
    Mmx = 1,
    /// `CPU_SSE` / `CV_CPU_SSE`
    Sse = 2,
    /// `CPU_SSE2` / `CV_CPU_SSE2`
    Sse2 = 3,
    /// `CPU_SSE3` / `CV_CPU_SSE3`
    Sse3 = 4,
    /// `CPU_SSSE3` / `CV_CPU_SSSE3`
    Ssse3 = 5,
    /// `CPU_SSE4_1` / `CV_CPU_SSE4_1`
    Sse41 = 6,
    /// `CPU_SSE4_2` / `CV_CPU_SSE4_2`
    Sse42 = 7,
    /// `CPU_POPCNT` / `CV_CPU_POPCNT`
    Popcnt = 8,
    /// `CPU_FP16` / `CV_CPU_FP16`
    Fp16 = 9,
    /// `CPU_AVX` / `CV_CPU_AVX`
    Avx = 10,
    /// `CPU_AVX2` / `CV_CPU_AVX2`
    Avx2 = 11,
    /// `CPU_FMA3` / `CV_CPU_FMA3`
    Fma3 = 12,
    /// `CPU_AVX_512F` / `CV_CPU_AVX_512F`
    Avx512f = 13,
    /// `CPU_AVX_512BW` / `CV_CPU_AVX_512BW`
    Avx512bw = 14,
    /// `CPU_AVX_512CD` / `CV_CPU_AVX_512CD`
    Avx512cd = 15,
    /// `CPU_AVX_512DQ` / `CV_CPU_AVX_512DQ`
    Avx512dq = 16,
    /// `CPU_AVX_512ER` / `CV_CPU_AVX_512ER`
    Avx512er = 17,
    /// `CPU_AVX_512IFMA` / `CPU_AVX_512IFMA512` — upstream shares the value.
    ///
    /// `CV_CPU_AVX_512IFMA`, `CV_CPU_AVX_512IFMA512` and `CV_AVX_512IFMA512`
    /// are all 18.
    Avx512ifma = 18,
    /// `CPU_AVX_512PF` / `CV_CPU_AVX_512PF`
    Avx512pf = 19,
    /// `CPU_AVX_512VBMI` / `CV_CPU_AVX_512VBMI`
    Avx512vbmi = 20,
    /// `CPU_AVX_512VL` / `CV_CPU_AVX_512VL`
    Avx512vl = 21,
    /// `CPU_AVX_512VBMI2` / `CV_CPU_AVX_512VBMI2`
    Avx512vbmi2 = 22,
    /// `CPU_AVX_512VNNI` / `CV_CPU_AVX_512VNNI`
    Avx512vnni = 23,
    /// `CPU_AVX_512BITALG` / `CV_CPU_AVX_512BITALG`
    Avx512bitalg = 24,
    /// `CPU_AVX_512VPOPCNTDQ` / `CV_CPU_AVX_512VPOPCNTDQ`
    Avx512vpopcntdq = 25,
    /// `CPU_AVX_5124VNNIW` / `CV_CPU_AVX_5124VNNIW`
    Avx5124vnniw = 26,
    /// `CPU_AVX_5124FMAPS` / `CV_CPU_AVX_5124FMAPS`
    Avx5124fmaps = 27,
    /// `CPU_AVX_VNNI` / `CV_CPU_AVX_VNNI`
    AvxVnni = 28,
    /// `CPU_NEON` / `CV_CPU_NEON`
    Neon = 100,
    /// `CPU_NEON_DOTPROD` / `CV_CPU_NEON_DOTPROD`
    NeonDotprod = 101,
    /// `CPU_NEON_FP16` / `CV_CPU_NEON_FP16`
    NeonFp16 = 102,
    /// `CPU_NEON_BF16` / `CV_CPU_NEON_BF16`
    NeonBf16 = 103,
    /// `CPU_SVE` / `CV_CPU_SVE`
    Sve = 104,
    /// `CPU_MSA` / `CV_CPU_MSA`
    Msa = 150,
    /// `CPU_RISCVV` / `CV_CPU_RISCVV`
    RiscvV = 170,
    /// `CPU_VSX` / `CV_CPU_VSX`
    Vsx = 200,
    /// `CPU_VSX3` / `CV_CPU_VSX3`
    Vsx3 = 201,
    /// `CPU_RVV` / `CV_CPU_RVV`
    Rvv = 210,
    /// `CPU_LSX` / `CV_CPU_LSX`
    Lsx = 230,
    /// `CPU_LASX` / `CV_CPU_LASX`
    Lasx = 231,
    /// `CPU_AVX512_SKX` / `CV_CPU_AVX512_SKX` — the Skylake-X AVX-512 feature group.
    Avx512Skx = 256,
    /// `CPU_AVX512_COMMON` / `CV_CPU_AVX512_COMMON` — the subset every AVX-512 part has.
    Avx512Common = 257,
    /// `CPU_AVX512_KNL` / `CV_CPU_AVX512_KNL` — Knights Landing.
    Avx512Knl = 258,
    /// `CPU_AVX512_KNM` / `CV_CPU_AVX512_KNM` — Knights Mill.
    Avx512Knm = 259,
    /// `CPU_AVX512_CNL` / `CV_CPU_AVX512_CNL` — Cannon Lake.
    Avx512Cnl = 260,
    /// `CPU_AVX512_CLX` / `CV_CPU_AVX512_CLX` — Cascade Lake.
    Avx512Clx = 261,
    /// `CPU_AVX512_ICL` / `CV_CPU_AVX512_ICL` — Ice Lake.
    Avx512Icl = 262,
}

/// `CPU_MAX_FEATURE` / `CV_CPU_MAX_FEATURE` — one past the highest feature id.
pub const CPU_MAX_FEATURE: i32 = 512;
/// `CV_HARDWARE_MAX_FEATURE` — the size of upstream's feature table.
pub const CV_HARDWARE_MAX_FEATURE: i32 = CPU_MAX_FEATURE;
/// `CV_CPU_NONE` — the absence of any feature requirement.
pub const CV_CPU_NONE: i32 = 0;

/// Every feature in the enumeration, in ascending order.
const ALL_FEATURES: &[CpuFeature] = &[
    CpuFeature::Mmx,
    CpuFeature::Sse,
    CpuFeature::Sse2,
    CpuFeature::Sse3,
    CpuFeature::Ssse3,
    CpuFeature::Sse41,
    CpuFeature::Sse42,
    CpuFeature::Popcnt,
    CpuFeature::Fp16,
    CpuFeature::Avx,
    CpuFeature::Avx2,
    CpuFeature::Fma3,
    CpuFeature::Avx512f,
    CpuFeature::Avx512bw,
    CpuFeature::Avx512cd,
    CpuFeature::Avx512dq,
    CpuFeature::Avx512er,
    CpuFeature::Avx512ifma,
    CpuFeature::Avx512pf,
    CpuFeature::Avx512vbmi,
    CpuFeature::Avx512vl,
    CpuFeature::Avx512vbmi2,
    CpuFeature::Avx512vnni,
    CpuFeature::Avx512bitalg,
    CpuFeature::Avx512vpopcntdq,
    CpuFeature::Avx5124vnniw,
    CpuFeature::Avx5124fmaps,
    CpuFeature::AvxVnni,
    CpuFeature::Neon,
    CpuFeature::NeonDotprod,
    CpuFeature::NeonFp16,
    CpuFeature::NeonBf16,
    CpuFeature::Sve,
    CpuFeature::Msa,
    CpuFeature::RiscvV,
    CpuFeature::Vsx,
    CpuFeature::Vsx3,
    CpuFeature::Rvv,
    CpuFeature::Lsx,
    CpuFeature::Lasx,
];

impl CpuFeature {
    /// `cv::getHardwareFeatureName(feature)`.
    ///
    /// The spellings are upstream's `HWFeatureNames` table, not the Rust
    /// variant names, so a diagnostic printed here reads the same as one from
    /// OpenCV.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            CpuFeature::Mmx => "MMX",
            CpuFeature::Sse => "SSE",
            CpuFeature::Sse2 => "SSE2",
            CpuFeature::Sse3 => "SSE3",
            CpuFeature::Ssse3 => "SSSE3",
            CpuFeature::Sse41 => "SSE4.1",
            CpuFeature::Sse42 => "SSE4.2",
            CpuFeature::Popcnt => "POPCNT",
            CpuFeature::Fp16 => "FP16",
            CpuFeature::Avx => "AVX",
            CpuFeature::Avx2 => "AVX2",
            CpuFeature::Fma3 => "FMA3",
            CpuFeature::Avx512f => "AVX512F",
            CpuFeature::Avx512bw => "AVX512BW",
            CpuFeature::Avx512cd => "AVX512CD",
            CpuFeature::Avx512dq => "AVX512DQ",
            CpuFeature::Avx512er => "AVX512ER",
            CpuFeature::Avx512ifma => "AVX512IFMA",
            CpuFeature::Avx512pf => "AVX512PF",
            CpuFeature::Avx512vbmi => "AVX512VBMI",
            CpuFeature::Avx512vl => "AVX512VL",
            CpuFeature::Avx512vbmi2 => "AVX512VBMI2",
            CpuFeature::Avx512vnni => "AVX512VNNI",
            CpuFeature::Avx512bitalg => "AVX512BITALG",
            CpuFeature::Avx512vpopcntdq => "AVX512VPOPCNTDQ",
            CpuFeature::Avx5124vnniw => "AVX5124VNNIW",
            CpuFeature::Avx5124fmaps => "AVX5124FMAPS",
            CpuFeature::AvxVnni => "AVX_VNNI",
            CpuFeature::Neon => "NEON",
            CpuFeature::NeonDotprod => "NEON_DOTPROD",
            CpuFeature::NeonFp16 => "NEON_FP16",
            CpuFeature::NeonBf16 => "NEON_BF16",
            CpuFeature::Sve => "SVE",
            CpuFeature::Msa => "MSA",
            CpuFeature::RiscvV => "RISCVV",
            CpuFeature::Vsx => "VSX",
            CpuFeature::Vsx3 => "VSX3",
            CpuFeature::Rvv => "RVV",
            CpuFeature::Lsx => "LSX",
            CpuFeature::Lasx => "LASX",
            CpuFeature::Avx512Skx => "AVX512-SKX",
            CpuFeature::Avx512Common => "AVX512-COMMON",
            CpuFeature::Avx512Knl => "AVX512-KNL",
            CpuFeature::Avx512Knm => "AVX512-KNM",
            CpuFeature::Avx512Cnl => "AVX512-CNL",
            CpuFeature::Avx512Clx => "AVX512-CLX",
            CpuFeature::Avx512Icl => "AVX512-ICL",
        }
    }
}

/// `cv::checkHardwareSupport(feature)` — does the **host CPU** have it?
///
/// This reports the hardware, exactly as upstream's documentation describes it.
/// It says nothing about this port taking advantage of the feature: every
/// kernel here is scalar, for exact parity. A caller choosing between two of
/// its *own* implementations still wants the true answer.
#[must_use]
pub fn check_hardware_support(feature: CpuFeature) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use CpuFeature as F;
        return match feature {
            // Every x86_64 part has these by definition of the baseline.
            F::Mmx | F::Sse | F::Sse2 => true,
            F::Sse3 => is_x86_feature_detected!("sse3"),
            F::Ssse3 => is_x86_feature_detected!("ssse3"),
            F::Sse41 => is_x86_feature_detected!("sse4.1"),
            F::Sse42 => is_x86_feature_detected!("sse4.2"),
            F::Popcnt => is_x86_feature_detected!("popcnt"),
            F::Fp16 => is_x86_feature_detected!("f16c"),
            F::Avx => is_x86_feature_detected!("avx"),
            F::Avx2 => is_x86_feature_detected!("avx2"),
            F::Fma3 => is_x86_feature_detected!("fma"),
            F::Avx512f => is_x86_feature_detected!("avx512f"),
            F::Avx512bw => is_x86_feature_detected!("avx512bw"),
            F::Avx512cd => is_x86_feature_detected!("avx512cd"),
            F::Avx512dq => is_x86_feature_detected!("avx512dq"),
            F::Avx512ifma => is_x86_feature_detected!("avx512ifma"),
            F::Avx512vbmi => is_x86_feature_detected!("avx512vbmi"),
            F::Avx512vl => is_x86_feature_detected!("avx512vl"),
            F::Avx512vbmi2 => is_x86_feature_detected!("avx512vbmi2"),
            F::Avx512vnni => is_x86_feature_detected!("avx512vnni"),
            F::Avx512bitalg => is_x86_feature_detected!("avx512bitalg"),
            F::Avx512vpopcntdq => is_x86_feature_detected!("avx512vpopcntdq"),
            // Feature groups: upstream sets these when every member is present.
            F::Avx512Common => {
                check_hardware_support(F::Avx512f)
                    && check_hardware_support(F::Avx512cd)
            }
            F::Avx512Skx => {
                check_hardware_support(F::Avx512Common)
                    && check_hardware_support(F::Avx512bw)
                    && check_hardware_support(F::Avx512dq)
                    && check_hardware_support(F::Avx512vl)
            }
            F::Avx512Cnl => {
                check_hardware_support(F::Avx512Skx)
                    && check_hardware_support(F::Avx512ifma)
                    && check_hardware_support(F::Avx512vbmi)
            }
            F::Avx512Clx => {
                check_hardware_support(F::Avx512Skx) && check_hardware_support(F::Avx512vnni)
            }
            F::Avx512Icl => {
                check_hardware_support(F::Avx512Skx)
                    && check_hardware_support(F::Avx512ifma)
                    && check_hardware_support(F::Avx512vbmi)
                    && check_hardware_support(F::Avx512vnni)
                    && check_hardware_support(F::Avx512vbmi2)
                    && check_hardware_support(F::Avx512bitalg)
                    && check_hardware_support(F::Avx512vpopcntdq)
            }
            // AVX-512ER/PF and the 4-operand pair only ever shipped on Xeon
            // Phi, which `std` has no detection macro for.
            _ => false,
        };
    }
    #[cfg(target_arch = "aarch64")]
    {
        use CpuFeature as F;
        match feature {
            // Advanced SIMD is mandatory in the AArch64 base architecture.
            F::Neon => true,
            F::NeonDotprod => std::arch::is_aarch64_feature_detected!("dotprod"),
            F::NeonFp16 => std::arch::is_aarch64_feature_detected!("fp16"),
            F::NeonBf16 => std::arch::is_aarch64_feature_detected!("bf16"),
            F::Sve => std::arch::is_aarch64_feature_detected!("sve"),
            _ => false,
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = feature;
        false
    }
}

/// `cv::getHardwareFeatureName(feature)`.
#[must_use]
pub fn get_hardware_feature_name(feature: CpuFeature) -> &'static str {
    feature.name()
}

/// `cv::getCPUFeaturesLine()` — the supported features as one space-separated line.
///
/// Upstream prefixes a feature the build was *not* compiled for with `*`. This
/// port compiles no SIMD kernels at all, so every name would carry one; the
/// prefix is therefore omitted rather than applied uniformly to no purpose.
#[must_use]
pub fn get_cpu_features_line() -> String {
    ALL_FEATURES
        .iter()
        .filter(|f| check_hardware_support(**f))
        .map(|f| f.name())
        .collect::<Vec<_>>()
        .join(" ")
}

/// `cv::utils::logging::LogLevel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum LogLevel {
    Silent = 0,
    Fatal = 1,
    Error = 2,
    Warning = 3,
    Info = 4,
    Debug = 5,
    Verbose = 6,
}

/// `cv::utils::logging::setLogLevel(level)`. Returns the previous level.
pub fn set_log_level(level: LogLevel) -> LogLevel {
    let prev = get_log_level();
    LOG_LEVEL.store(level as i32, Ordering::Relaxed);
    prev
}

/// `cv::utils::logging::getLogLevel()`.
#[must_use]
pub fn get_log_level() -> LogLevel {
    match LOG_LEVEL.load(Ordering::Relaxed) {
        0 => LogLevel::Silent,
        1 => LogLevel::Fatal,
        2 => LogLevel::Error,
        4 => LogLevel::Info,
        5 => LogLevel::Debug,
        6 => LogLevel::Verbose,
        _ => LogLevel::Warning,
    }
}

/// `cv::samples::findFile(relative_path, required, silentMode)`.
///
/// Searches the paths added by [`add_samples_data_search_path`], then
/// `OPENCV_SAMPLES_DATA_PATH`, then the working directory.
pub fn find_file(relative_path: &str, required: bool) -> crate::Result<Option<String>> {
    use std::path::Path;
    let mut roots: Vec<String> = search_paths();
    if let Ok(env) = std::env::var("OPENCV_SAMPLES_DATA_PATH") {
        roots.push(env);
    }
    roots.push(".".to_string());
    for r in &roots {
        let p = Path::new(r).join(relative_path);
        if p.exists() {
            return Ok(Some(p.to_string_lossy().into_owned()));
        }
    }
    if required {
        Err(crate::Error::new(
            crate::Code::StsObjectNotFound,
            format!("could not find {relative_path} in any of {roots:?}"),
        ))
    } else {
        Ok(None)
    }
}

/// `cv::samples::findFileOrKeep(relative_path, silentMode)` — returns the input
/// unchanged when nothing matches, so a caller can pass through absolute paths.
#[must_use]
pub fn find_file_or_keep(relative_path: &str) -> String {
    match find_file(relative_path, false) {
        Ok(Some(p)) => p,
        _ => relative_path.to_string(),
    }
}

/// The one search-path store. A `OnceLock` declared inside a function is a
/// distinct static per function, so this has to live at module scope for the
/// reader and the writers to share it.
static SEARCH_PATHS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

fn search_paths() -> Vec<String> {
    let m = SEARCH_PATHS.get_or_init(|| Mutex::new(Vec::new()));
    m.lock().map(|g| g.clone()).unwrap_or_default()
}

/// `cv::samples::addSamplesDataSearchPath(path)`.
pub fn add_samples_data_search_path(path: &str) {
    push_search_path(path.to_string());
}

/// `cv::samples::addSamplesDataSearchSubDirectory(subdir)`.
pub fn add_samples_data_search_sub_directory(subdir: &str) {
    push_search_path(subdir.to_string());
}

fn push_search_path(p: String) {
    let m = SEARCH_PATHS.get_or_init(|| Mutex::new(Vec::new()));
    if let Ok(mut g) = m.lock() {
        if !g.contains(&p) {
            g.push(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_feature_values_match_upstream() {
        assert_eq!(CpuFeature::Mmx as i32, 1);
        assert_eq!(CpuFeature::Avx2 as i32, 11);
        assert_eq!(CpuFeature::Avx512vpopcntdq as i32, 25);
        assert_eq!(CpuFeature::AvxVnni as i32, 28);
        assert_eq!(CpuFeature::Neon as i32, 100);
        assert_eq!(CpuFeature::Lasx as i32, 231);
        assert_eq!(CpuFeature::Avx512Icl as i32, 262);
        assert_eq!(CPU_MAX_FEATURE, 512);
    }

    #[test]
    fn hardware_support_agrees_with_the_baseline_of_this_target() {
        // Whatever host runs this, the architecture's mandatory SIMD is present
        // and a feature from another architecture is not.
        if cfg!(target_arch = "x86_64") {
            assert!(check_hardware_support(CpuFeature::Sse2));
            assert!(!check_hardware_support(CpuFeature::Neon));
        } else if cfg!(target_arch = "aarch64") {
            assert!(check_hardware_support(CpuFeature::Neon));
            assert!(!check_hardware_support(CpuFeature::Avx2));
        }
        // A feature group is never reported without its members.
        if check_hardware_support(CpuFeature::Avx512Skx) {
            assert!(check_hardware_support(CpuFeature::Avx512f));
            assert!(check_hardware_support(CpuFeature::Avx512bw));
        }
        // The line lists exactly the supported features, by upstream's names.
        let line = get_cpu_features_line();
        for f in ALL_FEATURES {
            assert_eq!(
                line.split(' ').any(|w| w == f.name()),
                check_hardware_support(*f),
                "{} disagrees with the feature line {line:?}",
                f.name()
            );
        }
    }

    #[test]
    fn tick_count_advances_and_converts_to_seconds() {
        let a = get_tick_count();
        let mut acc = 0u64;
        for i in 0..200_000u64 {
            acc = acc.wrapping_add(i * i);
        }
        assert_ne!(acc, u64::MAX);
        let b = get_tick_count();
        assert!(b >= a, "the tick count must be monotonic: {a} then {b}");
        assert_eq!(get_tick_frequency(), 1e9);
        // And the conversion is the one upstream documents.
        let secs = (b - a) as f64 / get_tick_frequency();
        assert!((0.0..60.0).contains(&secs), "implausible elapsed time {secs}");
    }

    #[test]
    fn tick_meter_accumulates_across_intervals() {
        let mut t = TickMeter::new();
        assert_eq!(t.get_counter(), 0);
        assert_eq!(t.get_avg_time_sec(), 0.0, "an empty meter averages zero, not NaN");
        assert_eq!(t.get_fps(), 0.0, "and reports no frame rate rather than infinity");

        for _ in 0..3 {
            t.start();
            let mut acc = 0u64;
            for i in 0..50_000u64 {
                acc = acc.wrapping_add(i);
            }
            assert_ne!(acc, u64::MAX);
            t.stop();
        }
        assert_eq!(t.get_counter(), 3);
        // Deliberately *not* asserting elapsed > 0. An earlier version did, and
        // it passed in debug and failed in release: the optimiser removes the
        // busy loop, so three intervals can legitimately measure 0 ns. What the
        // meter must guarantee is the bookkeeping, not that time passed.
        assert!(t.get_time_ticks() >= 0);
        // The unit conversions are consistent with each other.
        assert!((t.get_time_milli() - t.get_time_sec() * 1e3).abs() < 1e-6);
        assert!((t.get_time_micro() - t.get_time_sec() * 1e6).abs() < 1e-3);
        assert!((t.get_avg_time_milli() - t.get_time_milli() / 3.0).abs() < 1e-6);
        // The last interval is a part of the total, not the whole of it.
        assert!(t.get_last_time_ticks() <= t.get_time_ticks());
        assert!(t.get_fps() >= 0.0);

        t.reset();
        assert_eq!(t.get_counter(), 0);
        assert_eq!(t.get_time_ticks(), 0);
    }

    #[test]
    fn a_stop_without_a_start_records_nothing() {
        let mut t = TickMeter::new();
        t.stop();
        assert_eq!(t.get_counter(), 0, "an unmatched stop must not invent an interval");
        assert_eq!(t.get_time_ticks(), 0);
    }

    #[test]
    fn thread_count_defaults_to_the_cpu_count_and_follows_the_setting() {
        let cpus = get_number_of_cpus();
        assert!(cpus >= 1);
        set_num_threads(0);
        assert_eq!(get_num_threads(), cpus, "zero means the default");
        set_num_threads(3);
        assert_eq!(get_num_threads(), 3);
        set_num_threads(-1);
        assert_eq!(get_num_threads(), cpus, "negative also means the default");
        assert_eq!(get_thread_num(), 0);
    }

    #[test]
    fn the_version_matches_the_upstream_release_being_tracked() {
        assert_eq!((get_version_major(), get_version_minor()), (4, 15));
        assert_eq!(get_version_string(), "4.15.0-dev");
        let b = get_build_information();
        assert!(b.contains("pure Rust"), "the build string should say what this is: {b}");
        assert!(b.contains("4.15.0-dev"));
    }

    #[test]
    fn the_absent_acceleration_backends_all_report_false() {
        assert!(!use_ipp() && !use_ipp_not_exact());
        assert!(!have_openvx() && !use_openvx());
        assert!(get_openvx_context().is_none());
        // The setters are accepted and change nothing.
        set_use_ipp(true);
        set_use_ipp_not_exact(true);
        set_use_openvx(true);
        assert!(!use_ipp() && !use_openvx());
        assert_eq!(get_ipp_version(), "disabled");
        assert_eq!(get_default_algorithm_hint(), AlgorithmHint::Accurate);
    }

    #[test]
    fn feature_names_are_upstreams_spellings() {
        assert_eq!(get_hardware_feature_name(CpuFeature::Sse41), "SSE4.1");
        assert_eq!(get_hardware_feature_name(CpuFeature::Neon), "NEON");
        assert_eq!(get_hardware_feature_name(CpuFeature::AvxVnni), "AVX_VNNI");
        assert_eq!(get_hardware_feature_name(CpuFeature::Avx512Skx), "AVX512-SKX");
    }

    #[test]
    fn the_optimization_flag_round_trips() {
        let before = use_optimized();
        set_use_optimized(false);
        assert!(!use_optimized());
        set_use_optimized(true);
        assert!(use_optimized());
        set_use_optimized(before);
    }

    #[test]
    fn log_level_round_trips_and_returns_the_previous_value() {
        let before = get_log_level();
        let prev = set_log_level(LogLevel::Debug);
        assert_eq!(get_log_level(), LogLevel::Debug);
        assert_eq!(prev, before, "setLogLevel returns what was there before");
        set_log_level(LogLevel::Silent);
        assert_eq!(get_log_level(), LogLevel::Silent);
        set_log_level(before);
        // The levels order the way upstream's do.
        assert!(LogLevel::Error < LogLevel::Warning);
        assert!(LogLevel::Verbose > LogLevel::Debug);
    }

    #[test]
    fn find_file_locates_a_real_file_and_reports_a_missing_one() {
        let dir = std::env::temp_dir().join("cv_no_c_find_file_test");
        std::fs::create_dir_all(&dir).unwrap();
        let name = "utility_probe.txt";
        std::fs::write(dir.join(name), b"x").unwrap();
        add_samples_data_search_path(dir.to_str().unwrap());

        let found = find_file(name, true).unwrap();
        assert!(found.is_some(), "the file was just written to a registered search path");
        assert!(found.unwrap().ends_with(name));

        // Absent and not required is None, not an error.
        assert_eq!(find_file("definitely_not_here_9f2a.dat", false).unwrap(), None);
        // Absent and required names what it looked for.
        let e = find_file("definitely_not_here_9f2a.dat", true).unwrap_err();
        assert!(format!("{e}").contains("definitely_not_here_9f2a.dat"));
        // findFileOrKeep hands back the input rather than failing.
        assert_eq!(find_file_or_keep("definitely_not_here_9f2a.dat"), "definitely_not_here_9f2a.dat");

        std::fs::remove_dir_all(&dir).ok();
    }
}
