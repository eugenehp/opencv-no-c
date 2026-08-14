//! Backend selection across RLX's device family.
//!
//! # Why availability is probed, not asked
//!
//! RLX's `is_available(Device)` reports what was *compiled in*, which is not
//! the same as what *works*. Two of its backend features are empty markers
//! with no implementation behind them:
//!
//! ```text
//! ane     = []   # Device::Ane => cfg!(any(feature = "coreml", feature = "ane"))
//! directx = []
//! ```
//!
//! Enable bare `ane` without `coreml` and `is_available(Device::Ane)` returns
//! true, then running panics with "no backend registered for device ANE".
//! `directx` behaves the same way on macOS: available, then panic.
//!
//! So [`Backend::availability`] does not take RLX's word for it. The first time
//! a backend is queried it compiles and runs a tiny convolution inside
//! `catch_unwind`, and only a correct answer earns [`Availability::Ready`]. The
//! result is cached, so the cost is paid once per process per backend.

use core::fmt;

/// A compute device this crate can dispatch to.
///
/// Every variant exists in every build, so callers can name a backend
/// unconditionally and get [`Availability::NotCompiled`] instead of a compile
/// error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Backend {
    /// Pick the fastest device that is verified working on this machine.
    #[default]
    Auto,
    /// Host CPU (`Device::Cpu`).
    Cpu,
    /// Apple Metal (`Device::Metal`).
    Metal,
    /// Apple MLX (`Device::Mlx`).
    Mlx,
    /// Apple Neural Engine via CoreML (`Device::Ane`).
    ///
    /// Requires the `rlx-coreml` feature. RLX also defines a bare `ane`
    /// feature, which only flips an availability flag and has no backend — see
    /// the module docs.
    Ane,
    /// Cross-platform GPU via wgpu (`Device::Gpu`).
    Gpu,
    /// Native Vulkan (`Device::Vulkan`).
    Vulkan,
    /// OpenGL / WebGL compute (`Device::OpenGl`).
    OpenGl,
    /// WebGPU (`Device::WebGpu`) — wgpu under its browser-facing name.
    WebGpu,
    /// DirectX 12 compute (`Device::DirectX`), Windows only.
    DirectX,
    /// NVIDIA CUDA (`Device::Cuda`).
    Cuda,
    /// AMD ROCm (`Device::Rocm`).
    Rocm,
    /// AMD XDNA / Ryzen AI NPU (`Device::Xdna`).
    Xdna,
    /// Qualcomm Hexagon NPU (`Device::Hexagon`).
    ///
    /// Needs `QNN_SDK_ROOT`; excluded from `rlx-all` because its build script
    /// hard-fails without the Qualcomm SDK.
    Hexagon,
    /// Intel oneAPI Level Zero (`Device::OneApi`).
    OneApi,
    /// Google TPU via PJRT (`Device::Tpu`).
    Tpu,
}

/// Why a backend can or cannot be used right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Availability {
    /// Compiled in, and verified to compute correctly on this machine.
    Ready,
    /// Compiled in, but no working device — absent hardware, absent driver, or
    /// a marker feature with no implementation behind it.
    NoDevice,
    /// Not compiled into this build; enable the matching Cargo feature.
    NotCompiled,
}

impl Availability {
    /// Whether work can actually be dispatched to this backend.
    #[must_use]
    pub const fn is_ready(self) -> bool {
        matches!(self, Availability::Ready)
    }
}

impl Backend {
    /// Every named backend, excluding [`Backend::Auto`].
    ///
    /// Ordered so that [`default_backend`] prefers accelerators that are
    /// typically fastest first.
    pub const ALL: [Backend; 15] = [
        Backend::Metal,
        Backend::Mlx,
        Backend::Cuda,
        Backend::Rocm,
        Backend::Vulkan,
        Backend::Gpu,
        Backend::WebGpu,
        Backend::OpenGl,
        Backend::DirectX,
        Backend::Xdna,
        Backend::Hexagon,
        Backend::Ane,
        Backend::Tpu,
        Backend::OneApi,
        Backend::Cpu,
    ];

    /// The Cargo feature that compiles this backend in.
    #[must_use]
    pub const fn feature(self) -> &'static str {
        match self {
            Backend::Auto | Backend::Cpu => "rlx",
            Backend::Metal => "rlx-metal",
            Backend::Mlx => "rlx-mlx",
            Backend::Ane => "rlx-coreml",
            Backend::Gpu => "rlx-gpu",
            Backend::Vulkan => "rlx-vulkan",
            Backend::OpenGl => "rlx-opengl",
            Backend::WebGpu => "rlx-webgpu",
            Backend::DirectX => "rlx-directx",
            Backend::Cuda => "rlx-cuda",
            Backend::Rocm => "rlx-rocm",
            Backend::Xdna => "rlx-xdna",
            Backend::Hexagon => "rlx-hexagon",
            Backend::OneApi => "rlx-oneapi",
            Backend::Tpu => "rlx-tpu",
        }
    }

    /// Whether this backend's feature is compiled into the current build.
    #[must_use]
    pub const fn is_compiled(self) -> bool {
        match self {
            Backend::Auto | Backend::Cpu => cfg!(feature = "rlx"),
            Backend::Metal => cfg!(feature = "rlx-metal"),
            Backend::Mlx => cfg!(feature = "rlx-mlx"),
            Backend::Ane => cfg!(feature = "rlx-coreml"),
            Backend::Gpu => cfg!(feature = "rlx-gpu"),
            Backend::Vulkan => cfg!(feature = "rlx-vulkan"),
            Backend::OpenGl => cfg!(feature = "rlx-opengl"),
            Backend::WebGpu => cfg!(feature = "rlx-webgpu"),
            Backend::DirectX => cfg!(feature = "rlx-directx"),
            Backend::Cuda => cfg!(feature = "rlx-cuda"),
            Backend::Rocm => cfg!(feature = "rlx-rocm"),
            Backend::Xdna => cfg!(feature = "rlx-xdna"),
            Backend::Hexagon => cfg!(feature = "rlx-hexagon"),
            Backend::OneApi => cfg!(feature = "rlx-oneapi"),
            Backend::Tpu => cfg!(feature = "rlx-tpu"),
        }
    }

    /// Whether this backend runs on the host CPU rather than an accelerator.
    #[must_use]
    pub const fn is_cpu(self) -> bool {
        matches!(self, Backend::Cpu)
    }

    /// Whether this backend is a neural accelerator rather than a GPU or CPU.
    ///
    /// NPUs are tuned for large fused networks; on a single small convolution
    /// they are frequently *slower* than the CPU, so [`default_backend`] does
    /// not reach for them first.
    #[must_use]
    pub const fn is_npu(self) -> bool {
        matches!(self, Backend::Ane | Backend::Xdna | Backend::Hexagon | Backend::Tpu)
    }

    /// Map onto RLX's device enum.
    #[cfg(feature = "rlx")]
    pub(crate) fn to_rlx(self) -> Option<rlx::Device> {
        Some(match self {
            Backend::Auto | Backend::Cpu => rlx::Device::Cpu,
            Backend::Metal => rlx::Device::Metal,
            Backend::Mlx => rlx::Device::Mlx,
            Backend::Ane => rlx::Device::Ane,
            Backend::Gpu => rlx::Device::Gpu,
            Backend::Vulkan => rlx::Device::Vulkan,
            Backend::OpenGl => rlx::Device::OpenGl,
            Backend::WebGpu => rlx::Device::WebGpu,
            Backend::DirectX => rlx::Device::DirectX,
            Backend::Cuda => rlx::Device::Cuda,
            Backend::Rocm => rlx::Device::Rocm,
            Backend::Xdna => rlx::Device::Xdna,
            Backend::Hexagon => rlx::Device::Hexagon,
            Backend::OneApi => rlx::Device::OneApi,
            Backend::Tpu => rlx::Device::Tpu,
        })
    }

    /// Current availability, verified by actually running a kernel once.
    #[must_use]
    pub fn availability(self) -> Availability {
        if !self.is_compiled() {
            return Availability::NotCompiled;
        }
        #[cfg(not(feature = "rlx"))]
        {
            Availability::NotCompiled
        }
        #[cfg(feature = "rlx")]
        {
            if self == Backend::Auto {
                return if Backend::ALL.iter().any(|b| b.availability().is_ready()) {
                    Availability::Ready
                } else {
                    Availability::NoDevice
                };
            }
            if probe::works(self) {
                Availability::Ready
            } else {
                Availability::NoDevice
            }
        }
    }
}

/// One-shot verification that a backend computes correctly.
#[cfg(feature = "rlx")]
mod probe {
    use super::Backend;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    /// Cache of verified backends, keyed by discriminant order in `ALL`.
    fn cache() -> &'static Mutex<HashMap<u8, bool>> {
        static CACHE: OnceLock<Mutex<HashMap<u8, bool>>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Stable key for a backend.
    fn key(b: Backend) -> u8 {
        Backend::ALL.iter().position(|x| *x == b).unwrap_or(u8::MAX as usize) as u8
    }

    /// Run a 4x4 convolution and check the answer, catching any panic.
    ///
    /// A backend whose feature is a bare marker panics inside RLX's session
    /// with "no backend registered"; that must read as unavailable, not crash
    /// the caller.
    fn run_once(b: Backend) -> bool {
        use rlx::{DType, Graph, Session, Shape};
        let Some(device) = b.to_rlx() else { return false };
        if !rlx::is_available(device) {
            return false;
        }

        // Silence the panic message while probing: an expected failure here is
        // not something the user needs to see on stderr.
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = std::panic::catch_unwind(move || {
            let mut g = Graph::new("cv_accel_probe");
            let x = g.input("x", Shape::new(&[1, 1, 4, 4], DType::F32));
            let w = g.param("w", Shape::new(&[1, 1, 3, 3], DType::F32));
            let y = g.conv2d(x, w, [3, 3], [1, 1], [0, 0], [1, 1], 1);
            g.set_outputs(vec![y]);
            let mut c = Session::new(device).compile(g);
            c.set_param("w", &[1.0f32; 9]);
            let img: Vec<f32> = (0..16).map(|i| i as f32).collect();
            c.run(&[("x", &img)]).into_iter().next()
        });
        std::panic::set_hook(previous);

        // The 3x3 box sum of a 4x4 ramp, computed by hand.
        const WANT: [f32; 4] = [45.0, 54.0, 81.0, 90.0];
        match outcome {
            Ok(Some(v)) if v.len() == 4 => {
                v.iter().zip(WANT.iter()).all(|(g, w)| (g - w).abs() < 1e-3)
            }
            _ => false,
        }
    }

    /// Cached form of [`run_once`].
    pub(super) fn works(b: Backend) -> bool {
        let k = key(b);
        if let Some(hit) = cache().lock().ok().and_then(|c| c.get(&k).copied()) {
            return hit;
        }
        let result = run_once(b);
        if let Ok(mut c) = cache().lock() {
            c.insert(k, result);
        }
        result
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Backend::Auto => "auto",
            Backend::Cpu => "cpu",
            Backend::Metal => "metal",
            Backend::Mlx => "mlx",
            Backend::Ane => "ane",
            Backend::Gpu => "gpu",
            Backend::Vulkan => "vulkan",
            Backend::OpenGl => "opengl",
            Backend::WebGpu => "webgpu",
            Backend::DirectX => "directx",
            Backend::Cuda => "cuda",
            Backend::Rocm => "rocm",
            Backend::Xdna => "xdna",
            Backend::Hexagon => "hexagon",
            Backend::OneApi => "oneapi",
            Backend::Tpu => "tpu",
        };
        f.write_str(s)
    }
}

/// Every backend that is compiled in *and* verified working right now.
#[must_use]
pub fn available_backends() -> Vec<Backend> {
    Backend::ALL.into_iter().filter(|b| b.availability().is_ready()).collect()
}

/// The backend [`Backend::Auto`] resolves to, or `None` if nothing works.
///
/// Prefers a GPU, then anything else, then the CPU — NPUs are skipped in the
/// first pass because they lose to the CPU on single small kernels (measured:
/// the Neural Engine runs a 720p 5x5 correlation at 0.9x the CPU's speed).
#[must_use]
pub fn default_backend() -> Option<Backend> {
    let ready = available_backends();
    ready
        .iter()
        .copied()
        .find(|b| !b.is_cpu() && !b.is_npu())
        .or_else(|| ready.iter().copied().find(|b| !b.is_cpu()))
        .or_else(|| ready.first().copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_is_complete_and_unique() {
        let mut seen = std::collections::HashSet::new();
        for b in Backend::ALL {
            assert!(seen.insert(b), "duplicate entry for {b}");
            assert_ne!(b, Backend::Auto, "Auto must not appear in ALL");
        }
        assert_eq!(Backend::ALL.len(), 15, "RLX defines 15 devices");
    }

    #[test]
    fn every_backend_reports_a_distinct_feature_and_name() {
        let mut names = std::collections::HashSet::new();
        for b in Backend::ALL {
            assert!(b.feature().starts_with("rlx"), "{b} -> {}", b.feature());
            assert!(names.insert(b.to_string()), "duplicate name for {b:?}");
        }
    }

    #[test]
    fn uncompiled_backends_say_so() {
        for b in Backend::ALL {
            if !b.is_compiled() {
                assert_eq!(b.availability(), Availability::NotCompiled, "{b}");
            }
        }
    }

    #[test]
    fn ready_implies_compiled() {
        for b in Backend::ALL {
            if b.availability().is_ready() {
                assert!(b.is_compiled(), "{b} reported Ready without being compiled");
            }
        }
    }

    #[test]
    fn availability_is_cached_and_stable() {
        // Two calls must agree; a flapping probe would make dispatch
        // unpredictable.
        for b in Backend::ALL {
            assert_eq!(b.availability(), b.availability(), "{b} is unstable");
        }
    }

    #[test]
    fn default_backend_prefers_a_gpu_over_an_npu() {
        let ready = available_backends();
        if let Some(chosen) = default_backend() {
            assert!(chosen.availability().is_ready());
            let has_gpu = ready.iter().any(|b| !b.is_cpu() && !b.is_npu());
            if has_gpu {
                assert!(!chosen.is_npu(), "{chosen} chosen despite a GPU being ready");
                assert!(!chosen.is_cpu(), "{chosen} chosen despite a GPU being ready");
            }
        } else {
            assert!(ready.is_empty());
        }
    }

    #[test]
    fn npu_classification_is_right() {
        assert!(Backend::Ane.is_npu());
        assert!(Backend::Xdna.is_npu());
        assert!(Backend::Hexagon.is_npu());
        assert!(!Backend::Metal.is_npu());
        assert!(!Backend::Cpu.is_npu());
        assert!(Backend::Cpu.is_cpu());
    }

    #[test]
    fn marker_only_backends_never_report_ready_without_an_implementation() {
        // `directx` and `ane` are empty features upstream: RLX would claim they
        // are available. The probe must catch that. DirectX cannot work on a
        // non-Windows host, so if it is compiled here it must not be Ready.
        if Backend::DirectX.is_compiled() && !cfg!(target_os = "windows") {
            assert_eq!(
                Backend::DirectX.availability(),
                Availability::NoDevice,
                "DirectX claimed Ready on a non-Windows host"
            );
        }
    }

    #[test]
    fn without_the_rlx_feature_nothing_is_ready() {
        if cfg!(feature = "rlx") {
            return;
        }
        assert!(available_backends().is_empty());
        assert_eq!(default_backend(), None);
    }
}
