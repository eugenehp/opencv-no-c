//! Optional [RLX](https://github.com/MIT-RLX/rlx)-backed acceleration for
//! `opencv-no-c`.
//!
//! # Why this crate exists separately
//!
//! The rest of this workspace links no C. RLX's CPU backend **does**: on Apple
//! platforms `rlx-cpu`'s build script emits
//! `cargo:rustc-link-lib=framework=Accelerate` unconditionally (not behind a
//! Cargo feature), and a binary that actually calls into it also picks up
//! `libbz2` and `libiconv`.
//!
//! That is a reasonable trade — Accelerate routes GEMM through the AMX
//! coprocessor and is far faster than anything portable Rust will do — but it
//! is a trade, and it contradicts the guarantee the rest of the workspace
//! makes. So it lives here, behind a feature that is **off by default**:
//!
//! - `cargo build` — every crate is pure Rust, nothing but `libSystem`.
//! - `cargo build --features rlx` — CPU backend; the binary links Accelerate.
//! - `cargo build --features rlx-all` — every backend RLX offers.
//!
//! # Backends
//!
//! [`Backend`] names each device RLX can target. Every variant exists in every
//! build, so code can ask about CUDA on a Mac and get
//! [`Availability::NotCompiled`] or [`Availability::NoDevice`] rather than
//! failing to compile. [`available_backends`] reports what is usable *now*;
//! [`Backend::Auto`] resolves to the fastest of those, preferring an
//! accelerator over the CPU.
//!
//! With the feature off, every entry point here returns `Unavailable`, and
//! callers fall back to their portable implementation. Nothing silently
//! changes behaviour based on how the workspace was built.
//!
//! # Numerics
//!
//! RLX computes in `f32`. The portable kernels in `cv-core` and `cv-imgproc`
//! accumulate in `f64`. For `CV_32F` data that difference is immaterial — and
//! arguably closer to what OpenCV itself does — but the two paths are **not**
//! bit-identical on every input. Callers that need exactness should stay on the
//! portable path; `cv-imgproc`'s tests assert the two agree to a stated
//! tolerance rather than claiming equality.

#![doc(html_root_url = "https://docs.rs/cv-accel/0.1.0")]
#![warn(missing_docs)]

pub mod onnx;
pub use onnx::{load_onnx, load_onnx_on, OnnxModel};

pub mod backend;

pub use backend::{available_backends, default_backend, Availability, Backend};

use cv_core::{Code, Error, Result};

/// Whether this build has the RLX backend compiled in.
#[must_use]
pub const fn is_available() -> bool {
    cfg!(feature = "rlx")
}

/// Whether this build links any C library.
///
/// True exactly when RLX is enabled on a platform whose RLX build script links
/// a native BLAS. GPU backends add their own system frameworks on top.
#[must_use]
pub const fn links_c() -> bool {
    cfg!(all(feature = "rlx", target_vendor = "apple"))
}

/// Resolve a requested backend to a concrete, usable one.
#[cfg(feature = "rlx")]
fn resolve(requested: Backend) -> Result<rlx::Device> {
    let chosen = if requested == Backend::Auto {
        default_backend().ok_or_else(|| {
            Error::new(Code::StsError, "no RLX backend has usable hardware".to_string())
        })?
    } else {
        requested
    };
    match chosen.availability() {
        Availability::Ready => chosen.to_rlx().ok_or_else(|| {
            Error::new(Code::StsError, format!("{chosen} has no RLX device mapping"))
        }),
        Availability::NotCompiled => Err(Error::new(
            Code::StsNotImplemented,
            format!("backend `{chosen}` needs the `{}` feature", chosen.feature()),
        )),
        Availability::NoDevice => Err(Error::new(
            Code::StsError,
            format!("backend `{chosen}` is compiled in but no device is present"),
        )),
    }
}

/// The error returned by every entry point when built without the `rlx` feature.
#[must_use]
pub fn unavailable(what: &str) -> Error {
    Error::new(
        Code::StsNotImplemented,
        format!("{what} needs the `rlx` feature; this build is pure Rust"),
    )
}

/// Cache of compiled RLX graphs, keyed by shape.
///
/// Building a graph is cheap (sub-microsecond); **compiling** it is not, and it
/// dominates GPU dispatch. Measured on 3x726x1286 with a 7x7 kernel:
///
/// ```text
/// backend   compile     run
/// cpu        0.54ms   77.09ms     compile is 1% of a cold call
/// metal      5.33ms    1.99ms     compile is 73%
/// vulkan    32.21ms    3.07ms     compile is 91%
/// ```
///
/// Recompiling per call threw away most of the GPU's advantage. The graph
/// depends only on the shapes, not the pixel values or the kernel weights, so
/// one compile serves every later call with the same geometry.
#[cfg(feature = "rlx")]
mod cache {
    use super::Backend;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::mem::ManuallyDrop;

    /// Everything the compiled graph's structure depends on.
    #[derive(PartialEq, Eq, Hash, Clone, Copy)]
    pub(super) struct Key {
        pub backend: Backend,
        pub dims: [usize; 5],
    }

    thread_local! {
        // Thread-local rather than a global Mutex: `CompiledGraph` holds
        // backend state that is cheapest to keep on one thread, and this
        // sidesteps both lock contention and any Send requirement.
        //
        // `ManuallyDrop` is load-bearing, not laziness. `CompiledGraph::drop`
        // reaches into RLX's own thread-locals; when this map is destroyed at
        // thread exit those may already be gone, and the resulting
        // `AccessError` panics *inside a destructor*, which Rust escalates to
        // an abort. Verified: without this the test binary dies with
        // "panic in a destructor during cleanup ... aborting".
        //
        // The cache is process-lifetime by design, so never dropping it is the
        // intended lifetime rather than a leak in any meaningful sense. It is
        // bounded by the number of distinct shapes a program filters.
        static GRAPHS: RefCell<ManuallyDrop<HashMap<Key, rlx::CompiledGraph>>> =
            RefCell::new(ManuallyDrop::new(HashMap::new()));
    }

    /// Run `f` against the cached graph for `key`, compiling it on first use.
    pub(super) fn with<R>(
        key: Key,
        build: impl FnOnce() -> rlx::CompiledGraph,
        f: impl FnOnce(&mut rlx::CompiledGraph) -> R,
    ) -> R {
        GRAPHS.with(|g| {
            let mut map = g.borrow_mut();
            let entry = map.entry(key).or_insert_with(build);
            f(entry)
        })
    }

    /// Number of graphs currently cached on this thread.
    pub(super) fn len() -> usize {
        GRAPHS.with(|g| g.borrow().len())
    }
}

/// How many compiled graphs this thread currently holds.
///
/// Exposed for tests and diagnostics; the cache is keyed by shape, so a program
/// that filters a fixed-size stream settles at a handful of entries.
#[must_use]
pub fn cached_graph_count() -> usize {
    #[cfg(feature = "rlx")]
    {
        cache::len()
    }
    #[cfg(not(feature = "rlx"))]
    {
        0
    }
}

/// A batched 2-D cross-correlation, matching `cv::filter2D`'s convention.
///
/// `input` is `[batch, height, width]` in row-major order with no padding; the
/// kernel is applied to each batch plane independently. Output is
/// `[batch, height - kh + 1, width - kw + 1]` — callers pad beforehand if they
/// want a same-size result with a particular border mode.
///
/// RLX's `Op::Conv` is a correlation, not a flipped convolution, so the kernel
/// is passed through unchanged.
pub fn correlate2d_f32(
    input: &[f32],
    batch: usize,
    height: usize,
    width: usize,
    kernel: &[f32],
    kh: usize,
    kw: usize,
) -> Result<Vec<f32>> {
    correlate2d_f32_on(Backend::Auto, input, batch, height, width, kernel, kh, kw)
}

/// [`correlate2d_f32`] pinned to a specific [`Backend`].
///
/// Errors rather than silently falling back, so a caller asking for Metal
/// learns that Metal was unavailable instead of quietly getting the CPU.
#[allow(unused_variables, clippy::too_many_arguments)]
pub fn correlate2d_f32_on(
    backend: Backend,
    input: &[f32],
    batch: usize,
    height: usize,
    width: usize,
    kernel: &[f32],
    kh: usize,
    kw: usize,
) -> Result<Vec<f32>> {
    #[cfg(not(feature = "rlx"))]
    {
        Err(unavailable("correlate2d_f32"))
    }
    #[cfg(feature = "rlx")]
    {
        use rlx::{DType, Graph, Session, Shape};
        let device = resolve(backend)?;

        if input.len() != batch * height * width {
            return Err(Error::new(
                Code::StsUnmatchedSizes,
                format!("input is {} values, expected {}", input.len(), batch * height * width),
            ));
        }
        if kernel.len() != kh * kw {
            return Err(Error::new(
                Code::StsUnmatchedSizes,
                format!("kernel is {} values, expected {}", kernel.len(), kh * kw),
            ));
        }
        if height < kh || width < kw {
            return Err(Error::new(
                Code::StsBadSize,
                format!("input {height}x{width} is smaller than kernel {kh}x{kw}"),
            ));
        }

        // Each colour plane becomes a batch item with a single channel, so one
        // kernel applies to all of them without relying on grouped-conv
        // semantics.
        let key = cache::Key { backend, dims: [batch, height, width, kh, kw] };
        let out = cache::with(
            key,
            || {
                let mut g = Graph::new("cv_filter2d");
                let x = g.input("x", Shape::new(&[batch, 1, height, width], DType::F32));
                let w = g.param("w", Shape::new(&[1, 1, kh, kw], DType::F32));
                let y = g.conv2d(x, w, [kh, kw], [1, 1], [0, 0], [1, 1], 1);
                g.set_outputs(vec![y]);
                Session::new(device).compile(g)
            },
            |compiled| {
                // Weights are a parameter, so a cached graph serves any kernel
                // of the same size.
                compiled.set_param("w", kernel);
                compiled.run(&[("x", input)])
            },
        );
        out.into_iter().next().ok_or_else(|| {
            Error::new(Code::StsError, "RLX returned no output tensor".to_string())
        })
    }
}

/// Full NCHW convolution: `input` is `n x cin x h x w`, `weights` is
/// `cout x (cin/groups) x kh x kw`, `bias` is `cout` or empty.
///
/// This is the entry point [`cv-dnn`](https://docs.rs/cv-dnn)'s `Convolution`
/// layer uses. Unlike [`correlate2d_f32`] it carries channels, groups, stride,
/// padding and dilation, so it maps onto RLX's `conv2d` node directly instead
/// of being decomposed per plane.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    shape: ConvShape,
) -> Result<Vec<f32>> {
    conv2d_f32_on(Backend::Auto, input, weights, bias, shape)
}

/// Geometry of a [`conv2d_f32`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvShape {
    /// Batch size.
    pub batch: usize,
    /// Input channels.
    pub c_in: usize,
    /// Input height and width.
    pub hw: (usize, usize),
    /// Output channels.
    pub c_out: usize,
    /// Kernel height and width.
    pub k: (usize, usize),
    /// Stride, rows then columns.
    pub stride: (usize, usize),
    /// Zero padding, rows then columns.
    pub pad: (usize, usize),
    /// Dilation, rows then columns.
    pub dilation: (usize, usize),
    /// Group count; `c_in` and `c_out` must both divide by it.
    pub groups: usize,
}

impl ConvShape {
    /// Output spatial size after stride, padding and dilation.
    #[must_use]
    pub fn out_hw(&self) -> (usize, usize) {
        let f = |i: usize, k: usize, s: usize, p: usize, d: usize| {
            let eff = d * (k - 1) + 1;
            if i + 2 * p < eff {
                0
            } else {
                (i + 2 * p - eff) / s + 1
            }
        };
        (
            f(self.hw.0, self.k.0, self.stride.0, self.pad.0, self.dilation.0),
            f(self.hw.1, self.k.1, self.stride.1, self.pad.1, self.dilation.1),
        )
    }
}

/// [`conv2d_f32`] pinned to a specific [`Backend`].
#[allow(unused_variables, clippy::too_many_arguments)]
pub fn conv2d_f32_on(
    backend: Backend,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    s: ConvShape,
) -> Result<Vec<f32>> {
    if s.groups == 0 || s.c_in % s.groups != 0 || s.c_out % s.groups != 0 {
        return Err(Error::new(
            Code::StsBadArg,
            format!("groups={} does not divide c_in={} / c_out={}", s.groups, s.c_in, s.c_out),
        ));
    }
    if input.len() != s.batch * s.c_in * s.hw.0 * s.hw.1 {
        return Err(Error::new(
            Code::StsUnmatchedSizes,
            format!("input is {} values, expected {}", input.len(), s.batch * s.c_in * s.hw.0 * s.hw.1),
        ));
    }
    let wanted_w = s.c_out * (s.c_in / s.groups) * s.k.0 * s.k.1;
    if weights.len() != wanted_w {
        return Err(Error::new(
            Code::StsUnmatchedSizes,
            format!("weights is {} values, expected {wanted_w}", weights.len()),
        ));
    }
    if !bias.is_empty() && bias.len() != s.c_out {
        return Err(Error::new(
            Code::StsUnmatchedSizes,
            format!("bias is {} values, expected {} or 0", bias.len(), s.c_out),
        ));
    }

    #[cfg(not(feature = "rlx"))]
    {
        Err(unavailable("conv2d_f32"))
    }
    #[cfg(feature = "rlx")]
    {
        use rlx::{DType, Graph, Session, Shape};
        let device = resolve(backend)?;
        let key = cache::Key {
            backend,
            dims: [
                s.batch * 1_000_003 + s.c_in,
                s.hw.0 * 1_000_003 + s.hw.1,
                s.c_out * 1_000_003 + s.groups,
                s.k.0 * 1_000_003 + s.k.1,
                (s.stride.0 * 31 + s.stride.1) * 1_000_003 + s.pad.0 * 31 + s.pad.1 + s.dilation.0,
            ],
        };
        let out = cache::with(
            key,
            || {
                let mut g = Graph::new("cv_dnn_conv2d");
                let x = g.input("x", Shape::new(&[s.batch, s.c_in, s.hw.0, s.hw.1], DType::F32));
                let w = g.param(
                    "w",
                    Shape::new(&[s.c_out, s.c_in / s.groups, s.k.0, s.k.1], DType::F32),
                );
                let y = g.conv2d(
                    x,
                    w,
                    [s.k.0, s.k.1],
                    [s.stride.0, s.stride.1],
                    [s.pad.0, s.pad.1],
                    [s.dilation.0, s.dilation.1],
                    s.groups,
                );
                g.set_outputs(vec![y]);
                Session::new(device).compile(g)
            },
            |compiled| {
                compiled.set_param("w", weights);
                compiled.run(&[("x", input)])
            },
        );
        let mut v = out
            .into_iter()
            .next()
            .ok_or_else(|| Error::new(Code::StsError, "RLX returned no output tensor".to_string()))?;
        // RLX's conv2d node carries no bias term, so it is added here rather
        // than by growing the graph with a broadcast add for every layer.
        if !bias.is_empty() {
            let (oh, ow) = s.out_hw();
            let plane = oh * ow;
            for n in 0..s.batch {
                for (c, b) in bias.iter().enumerate().take(s.c_out) {
                    let base = (n * s.c_out + c) * plane;
                    for v in v[base..base + plane].iter_mut() {
                        *v += b;
                    }
                }
            }
        }
        Ok(v)
    }
}

/// Single-precision matrix product `a (m x k) * b (k x n)`, row-major.
pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    matmul_f32_on(Backend::Auto, a, b, m, k, n)
}

/// [`matmul_f32`] pinned to a specific [`Backend`].
#[allow(unused_variables)]
pub fn matmul_f32_on(
    backend: Backend,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    #[cfg(not(feature = "rlx"))]
    {
        Err(unavailable("matmul_f32"))
    }
    #[cfg(feature = "rlx")]
    {
        use rlx::{DType, Graph, Session, Shape};
        let device = resolve(backend)?;

        if a.len() != m * k || b.len() != k * n {
            return Err(Error::new(
                Code::StsUnmatchedSizes,
                format!(
                    "operands are {} and {} values, expected {} and {}",
                    a.len(),
                    b.len(),
                    m * k,
                    k * n
                ),
            ));
        }

        let mut g = Graph::new("cv_gemm");
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let w = g.param("w", Shape::new(&[k, n], DType::F32));
        let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);

        let mut compiled = Session::new(device).compile(g);
        compiled.set_param("w", b);
        let out = compiled.run(&[("x", a)]);
        out.into_iter().next().ok_or_else(|| {
            Error::new(Code::StsError, "RLX returned no output tensor".to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference correlation, written independently of both RLX and the
    /// `cv-imgproc` kernel so agreement is meaningful.
    fn reference(
        input: &[f32],
        batch: usize,
        h: usize,
        w: usize,
        kernel: &[f32],
        kh: usize,
        kw: usize,
    ) -> Vec<f32> {
        let (oh, ow) = (h - kh + 1, w - kw + 1);
        let mut out = vec![0f32; batch * oh * ow];
        for b in 0..batch {
            for y in 0..oh {
                for x in 0..ow {
                    let mut acc = 0f32;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            acc += input[b * h * w + (y + ky) * w + (x + kx)] * kernel[ky * kw + kx];
                        }
                    }
                    out[b * oh * ow + y * ow + x] = acc;
                }
            }
        }
        out
    }

    #[test]
    fn availability_flags_agree_with_the_build() {
        // These are the contract: a caller can ask, at runtime, whether this
        // build is still C-free.
        assert_eq!(is_available(), cfg!(feature = "rlx"));
        if !is_available() {
            assert!(!links_c(), "a build without RLX must not link C");
        }
    }

    #[test]
    fn entry_points_report_unavailable_without_the_feature() {
        if is_available() {
            return;
        }
        let e = correlate2d_f32(&[0.0; 9], 1, 3, 3, &[1.0], 1, 1).unwrap_err();
        assert_eq!(e.code, Code::StsNotImplemented);
        assert!(e.msg.contains("rlx"), "{}", e.msg);
        assert!(matmul_f32(&[0.0], &[0.0], 1, 1, 1).is_err());
    }

    #[test]
    fn correlation_matches_the_reference() {
        if !is_available() {
            return;
        }
        let (h, w, kh, kw) = (9usize, 11usize, 3usize, 3usize);
        let batch = 3usize;
        let input: Vec<f32> =
            (0..batch * h * w).map(|i| ((i * 37) % 251) as f32 / 251.0).collect();
        // An asymmetric kernel distinguishes correlation from convolution.
        let kernel: Vec<f32> = vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.25, 0.0, -0.35];

        let got = correlate2d_f32(&input, batch, h, w, &kernel, kh, kw).unwrap();
        let want = reference(&input, batch, h, w, &kernel, kh, kw);
        assert_eq!(got.len(), want.len());
        for (i, (g, r)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - r).abs() < 1e-5, "index {i}: {g} vs {r}");
        }
    }

    #[test]
    fn correlation_is_not_flipped() {
        if !is_available() {
            return;
        }
        // Kernel taps only the top-left neighbour. Correlation reads input[0],
        // a flipped convolution would read input[8].
        let input: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let mut kernel = vec![0f32; 9];
        kernel[0] = 1.0;
        let out = correlate2d_f32(&input, 1, 3, 3, &kernel, 3, 3).unwrap();
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn matmul_matches_a_hand_computed_product() {
        if !is_available() {
            return;
        }
        // [[1,2,3],[4,5,6]] * [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let out = matmul_f32(&a, &b, 2, 3, 2).unwrap();
        assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn every_available_backend_agrees_with_the_reference() {
        // The point of enabling many backends is that they compute the *same*
        // thing. Any backend that reports Ready must match the scalar
        // reference, or it is worse than not having it.
        let ready = available_backends();
        if ready.is_empty() {
            return;
        }
        let (h, w, kh, kw, batch) = (12usize, 16usize, 3usize, 3usize, 2usize);
        let input: Vec<f32> =
            (0..batch * h * w).map(|i| ((i * 53) % 97) as f32 / 97.0).collect();
        let kernel: Vec<f32> = vec![0.2, -0.1, 0.3, 0.05, 0.4, -0.25, 0.15, 0.0, -0.3];
        let want = reference(&input, batch, h, w, &kernel, kh, kw);

        for b in ready {
            let got = correlate2d_f32_on(b, &input, batch, h, w, &kernel, kh, kw)
                .unwrap_or_else(|e| panic!("{b} failed: {e}"));
            assert_eq!(got.len(), want.len(), "{b} returned the wrong length");
            for (i, (g, r)) in got.iter().zip(want.iter()).enumerate() {
                // GPU backends may use fused multiply-add or a different
                // summation order, so this is a tolerance, not equality.
                assert!((g - r).abs() < 1e-4, "{b} index {i}: {g} vs {r}");
            }
        }
    }

    #[test]
    fn every_available_backend_matmuls_correctly() {
        for b in available_backends() {
            let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let m = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
            let out = matmul_f32_on(b, &a, &m, 2, 3, 2)
                .unwrap_or_else(|e| panic!("{b} failed: {e}"));
            for (g, r) in out.iter().zip([58.0f32, 64.0, 139.0, 154.0].iter()) {
                assert!((g - r).abs() < 1e-3, "{b}: {g} vs {r}");
            }
        }
    }

    #[test]
    fn an_uncompiled_backend_errors_instead_of_falling_back() {
        // Silently substituting the CPU would make a benchmark meaningless.
        for b in Backend::ALL {
            if b.availability() != Availability::NotCompiled {
                continue;
            }
            let e = correlate2d_f32_on(b, &[0.0; 16], 1, 4, 4, &[1.0], 1, 1).unwrap_err();
            assert_eq!(e.code, Code::StsNotImplemented, "{b}: {}", e.msg);
            // Only the RLX-enabled build reaches `resolve`, which is what names
            // the specific missing feature; without RLX the message is the
            // blanket "needs the `rlx` feature".
            if is_available() {
                assert!(
                    e.msg.contains(b.feature()),
                    "{b}: error should name the missing feature, got {}",
                    e.msg
                );
            }
            return;
        }
    }

    #[test]
    fn a_backend_without_hardware_errors_clearly() {
        for b in Backend::ALL {
            if b.availability() == Availability::NoDevice {
                let e = correlate2d_f32_on(b, &[0.0; 16], 1, 4, 4, &[1.0], 1, 1).unwrap_err();
                assert!(e.msg.contains("no device"), "{b}: got {}", e.msg);
                return;
            }
        }
    }

    #[test]
    fn mismatched_shapes_are_rejected() {
        if !is_available() {
            return;
        }
        assert!(correlate2d_f32(&[0.0; 8], 1, 3, 3, &[1.0], 1, 1).is_err());
        assert!(correlate2d_f32(&[0.0; 9], 1, 3, 3, &[1.0, 2.0], 1, 1).is_err());
        // Kernel larger than the image.
        assert!(correlate2d_f32(&[0.0; 4], 1, 2, 2, &[0.0; 9], 3, 3).is_err());
        assert!(matmul_f32(&[0.0; 5], &[0.0; 6], 2, 3, 2).is_err());
    }
}
