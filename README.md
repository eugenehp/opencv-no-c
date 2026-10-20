# opencv-no-c

A pure-Rust port of OpenCV. No C, no C++, no `build.rs` linking step, no
`libopencv_*.dylib` on the system.

Tracks upstream **OpenCV 4.15.0-dev** @ `77dfa297d08fdecdc509fc01ad92a2e9ec776a57`.
Verified against a real **cv2 4.13.0** install.

```toml
[dependencies]
opencv-no-c = "0.1"
```

That is every module. Each OpenCV module is a feature of the same name
(`imgproc`, `imgcodecs`, `videoio`, `highgui`, `video`, `calib3d`,
`features2d`, `objdetect`, `dnn`, `ml`, `flann`, `photo`, `stitching`, `gapi`),
all on by default; `core` is always present. To build only what you use:

```toml
opencv-no-c = { version = "0.1", default-features = false, features = ["imgproc", "imgcodecs"] }
```

Rust 1.77+ (for `f64::round_ties_even`, which `cvRound` depends on).
Apache-2.0, matching upstream.

## Status

Two numbers, and they measure different things. Quoting the first without the
second is the main way to be misleading about this project.

### Symbol coverage: 4607 / 4607 (100%)

Generated, not asserted — `cargo xtask report` rebuilds it from the upstream
headers on every run.

```
module         ported    total   parity
core             1485     1485  100.00%
imgproc           659      659  100.00%
imgcodecs         170      170  100.00%
videoio           430      430  100.00%
highgui            69       69  100.00%
video             142      142  100.00%
calib3d           200      200  100.00%
features2d        169      169  100.00%
objdetect         178      178  100.00%
dnn               159      159  100.00%
ml                247      247  100.00%
flann              88       88  100.00%
photo              87       87  100.00%
stitching         151      151  100.00%
gapi              373      373  100.00%
------------------------------------------------
TOTAL            4607     4607  100.00%
```

This counts whether a symbol of that name exists here, claimed in a doc comment
and matched against the upstream header. **It does not measure whether the
thing computes the right answer.** During this port, `cv::DISOpticalFlow`
carried all twelve of upstream's parameters, its three presets, getters,
setters and passing unit tests — and ran Farneback. It counted as 100% the
whole time. A complete parameter set is the most convincing possible disguise.

### Numerical verification: 469 comparisons, 0 unexplained

`examples/parity_check.rs` diffs this port against arrays produced by an actual
`cv2` install:

```sh
pip install opencv-python
python3 tools/gen_reference.py /tmp/parity
cargo run --release -p opencv-no-c --example parity_check -- /tmp/parity
```

```
469 comparisons: 298 exact, 171 within tolerance, 0 MISMATCH
```

Every non-exact check carries a tolerance derived from a documented cause,
written at the call site, not fitted to the data. Those causes are catalogued
further down.

That covers roughly **169 distinct function families**. Against 4607 claimed
symbols, it is a small fraction, and the gap is where the risk lives.

### What has never been compared to anything

Whole areas are implemented and numerically unverified:

| area | state |
|---|---|
| `features2d` detectors | AGAST, BRISK, AKAZE, SIFT, KAZE, MSER, SimpleBlobDetector are all implemented; **none** has a reference comparison |
| `imgcodecs` round-trips | PNG decode is bit-exact; **JPEG decode is not** — grayscale within 1 level on 1.8% of pixels, colour within 2 on 39% |
| `videoio` | no numeric checks |
| `dnn` | no numeric checks |
| `highgui` | no numeric checks |
| stitching matchers, bundle adjusters | only reachable through the pipeline, not driven directly |
| `VoronoiSeamFinder` | cv2 segfaults on it, so no reference can be obtained |

Known specific gaps, found by trying to use the library rather than by reading:

- `cv::Canny(dx, dy, edges, …)` — the precomputed-gradient overload — is not
  implemented. Only the image-taking form is.
- `randn` uses Box-Muller. Upstream uses the Ziggurat method, so the
  *distribution* matches and the *sequence* does not. `randu` is sequence-exact
  on every depth.
- `warpPerspective` has one pixel of 19200 that is off by a level; see below.

Run `cargo xtask report` and the parity check for the live figures. 1730 tests,
114k lines of Rust, 17 crates.

## The reference library is not upstream's source

`cv2.getBuildInformation()` on the machine this was verified against reports:

```
Custom HAL: YES (carotene (ver 0.0.1) KleidiCV (ver 0.7.0))
```

For any function reached through `CALL_HAL`, the reference runs an ARM kernel
whose arithmetic is **not** what `imgwarp.cpp` says. Reading the source gives
the wrong answer, and only a differential check finds it. Established for
`remap` and `warpPerspective`:

- The HAL serves 8- and 16-bit **unsigned** images of **one or two** channels.
  Three or four channels of the same depth, and every signed or floating depth,
  fall through to OpenCV's portable path.
- The HAL interpolates from the **exact** fraction. The portable path snaps the
  sample coordinate to 1/32 of a pixel.
- The HAL rounds halves **away from zero**; `saturate_cast` rounds to even.
- `warpAffine` shows no such split — it quantises everywhere.

Both behaviours are implemented, gated on the depth-and-channel rule, because
parity is measured against the library as installed. A port targeting an x86
cv2 would want the portable branch in both cases. This is the single most
important thing to know before trusting a number in this README on another
machine.

## This is not a drop-in replacement

Symbol coverage is not substitutability. Three divergences are deliberate and
none of them is going to close by accident:

1. **Errors are returned, not thrown.** Every fallible call gives `Result<T>`.
2. **`Mat` owns its pixels.** `cv::Mat` is refcounted with *shallow* copy:
   `Mat b = a;` aliases, and writing through `b` mutates `a`. This port uses
   `Mat` plus `MatView`/`MatViewMut`. Code that relies on aliasing — very
   common, e.g. `mat(roi).setTo(0)` mutating the parent — will compile here and
   do something different. This is the deepest divergence and the hardest to
   close.
3. **Names are snake_case**, so no C++ or cv2 source compiles unchanged.

### Three features, one per divergence

```toml
opencv-no-c = { version = "0.1", features = ["compat", "dropin", "abi"] }
```

| feature | closes | what you get |
|---|---|---|
| `compat` | `Mat` aliasing | `SharedMat`: cloning shares, `deep_clone` is `Mat::clone()`, an ROI writes through |
| `dropin` | naming and error model | upstream's spelling, errors that unwind instead of returning |
| `abi` | linkage | a C-callable surface plus `libopencv_no_c.a` / `.dylib` |

**`compat`** — `cv::Mat`'s storage model as an additional type:

```rust
use opencv_no_c::core::compat::SharedMat;
use opencv_no_c::core::{Mat, MatLike, Rect, Scalar, types::CV_8UC1};

let a = SharedMat::new(Mat::zeros(10, 10, CV_8UC1)?);
let b = a.clone();                                    // shares, as `Mat b = a;` does
b.roi(Rect::new(2, 2, 4, 4))?.set_to(Scalar::all(7.0))?;
assert_eq!(a.borrow().at_f64(3, 3, 0)?, 7.0);         // written through
assert_eq!(a.borrow().at_f64(0, 0, 0)?, 0.0);         // untouched outside the ROI
# Ok::<(), opencv_no_c::Error>(())
```

`borrow()`/`borrow_mut()` yield guards implementing `MatLike`/`MatLikeMut`, so
every existing function takes one unchanged.

**`dropin`** — upstream's names, and failures that unwind:

```rust
# #[cfg(feature = "dropin")] {
use opencv_no_c::dropin::*;
use opencv_no_c::core::{types::CV_8UC3, Mat};
use opencv_no_c::imgproc::color::ColorConversion;

let src = Mat::zeros(64, 64, CV_8UC3).unwrap();
let gray  = cvtColor(&src, ColorConversion::Bgr2Gray);          // no `?`, no unwrap
let small = resize(&gray, Size::new(32, 32), 0.0, 0.0, Interpolation::Area);
let edges = Canny(&small, 50.0, 150.0, 3, false);
assert_eq!(edges.rows(), 32);
# }
```

The panic payload is the `Error` itself, so `catch(|| …)` recovers it as a
`Result` — and a panic that is *not* an `Error` is re-raised rather than
absorbed, so a genuine bug never comes back as a library failure.

**`abi`** — a C-callable surface, with `crates/opencv-no-c/include/opencv_no_c.h`:

```c
CvnoMat *src = cvno_mat_zeros(64, 48, 16);       /* CV_8UC3 */
CvnoMat *gray = NULL;
if (cvno_cvt_color(src, 6 /* COLOR_BGR2GRAY */, &gray) < 0)
    fprintf(stderr, "%s\n", cvno_last_error());
```

```sh
cargo build -p opencv-no-c --features abi --release
cc main.c target/release/libopencv_no_c.a
```

Fallible calls return `0` or `cv::Error::Code` unchanged — negative, and equal
to the constant OpenCV uses, so `StsBadArg` is `-5` in both. Handles are opaque
and caller-owned; nothing unwinds across the boundary, because unwinding into C
is undefined. Both a C and a C++ program are compiled and run against the static
library as part of verifying this.

**Exporting a C ABI links no C.** `extern "C"` is a calling convention, not a
dependency: these symbols come out of rustc and no C toolchain or C library is
involved. The "no C" premise is about not depending on one, and this does not.

### What none of them fix

`abi` is **not** OpenCV's C++ ABI, and cannot be. That depends on Itanium name
mangling, vtable layout and the memory layout of `std::vector`, `std::string`
and `cv::Mat` as a particular libstdc++ emitted them — none of which Rust can
reproduce. Code calling `cv::cvtColor` will not link against this crate. A
stable C ABI is the achievable thing, and symbols are prefixed `cvno_` so that
a program linking both this and a real `libopencv_core` gets a link error
rather than a silently-resolved wrong symbol.

`dropin` gives upstream's spelling, not source compatibility: argument order and
defaulted parameters differ, results are returned rather than filled through an
`OutputArray`, and Rust has no overloading. It removes the two most mechanical
differences; it is a translation aid, not a shim.

`compat`'s refcount is `Rc`-based and single-threaded where `cv::Mat`'s is
atomic, and conflicting borrows panic rather than racing.

**Why these are types and features rather than changes to `Mat`.** Cargo
features are additive and unify across the whole dependency graph: if any crate
anywhere turns one on, every crate gets it. A feature that changed what
`Mat::clone` means would silently change the behaviour of dependents that never
asked for it, at call sites the compiler would not flag. Adding a type is a
change no existing code can be surprised by.

If you want more than this, the remaining targets are a cv2-compatible Python
module (a PyO3 layer over a numeric core that is already here and verified —
binding work, not algorithm work) or a drop-in for the `opencv` Rust crate,
which `compat` is now most of.

## WebAssembly

`cv-samples` ports OpenCV's own sample programs and builds for
`wasm32-unknown-unknown`:

```sh
cargo build -p opencv-no-c-samples --target wasm32-unknown-unknown --release
```

The result is a 1.4 MB module with **zero host imports** and three exports
(`sample_count`, `sample_name`, `sample_run`). Nothing in that crate touches a
file system, clock, thread, environment variable or terminal; every input image
is embedded with `include_bytes!` and decoded in process.

Upstream samples end in `imshow` and a `waitKey`. A window is not a result, so
each port ends in named numeric arrays instead — which is what makes them
checkable as well as portable. Samples with a trackbar are run at the slider's
initial position, named in the module.

Of upstream's 97 C++ samples, 34 need `VideoCapture`/`VideoWriter`, 51 are
driven by `imshow`/trackbars/mouse, and 12 are pure computation. The camera ones
have no input source under WebAssembly and the interactive ones have no input
event; their *computations* port fine, but a literal port of all of them is not
achievable and this crate does not claim one.

## What works today

Every upstream module has a crate, and every symbol in the headers is claimed.
Rather than list 4607 names, here is what has been **verified against cv2**,
which is the part worth trusting:

- **core** — arithmetic with correct saturation, statistics, transcendentals
  (`fastAtan2`'s degree-7 polynomial and the 64-entry sine table, not `atan2`),
  linear algebra, layout ops, `FileStorage` for YAML and JSON in upstream's own
  layout, `RNG` sequence-exact with upstream on every integer and float depth
- **imgproc** — colour conversion, `resize` (all four paths, including the
  INTER_AREA scale rule below), filters, morphology, warps, pyramids,
  histograms and CLAHE, template matching, contours and shape descriptors,
  `watershed`, `distanceTransform`, `HoughLines`/`HoughCircles`, demosaicing
- **imgcodecs** — PNG (bit-exact), BMP, PNM, and a full RFC 1951 DEFLATE
  implementation written because PNG needs it and this port links no zlib
- **photo** — inpainting (Telea and Navier-Stokes), seamless cloning, HDR merge
  and tone mapping, stylisation, denoising
- **video** — Farneback, sparse Lucas-Kanade, ECC, CamShift, background
  subtraction, and dense inverse search with its variational refinement
- **calib3d**, **features2d**, **objdetect**, **ml**, **stitching**, **flann**,
  **dnn**, **gapi**, **videoio**, **highgui** — present; see the verification
  table above for how much of each is checked

## Numerical parity is the point

OpenCV's 8-bit kernels are fixed-point, not floating-point rounded at the end.
Reimplementing them "the obvious way" produces results that are off by one
level on a large fraction of pixels — close enough to look right, far enough to
fail any comparison test. This port reproduces the integer arithmetic directly:

| operation | upstream detail reproduced here |
|---|---|
| `cvRound` | round-half-to-**even** (`lrint`/`_mm_cvtss_si32`), not half-away-from-zero |
| RGB→Gray | 15-bit shift, `RY15=9798 GY15=19235 BY15=3735` summing to exactly `1<<15` |
| RGB↔YCrCb | 14-bit shift with a different coefficient set (`R2Y=4899 …`) |
| RGB→HSV 8U | 12-bit reciprocal lookup tables, integer sextant selection |
| `resize` INTER_LINEAR 8U | 11-bit weights, then `((b0*(S0>>4))>>16 + … + 2)>>2` |
| `resize` INTER_LINEAR @2× | silently promoted to INTER_AREA, as upstream does |
| `getGaussianKernel` | exact tabulated kernels for `ksize ≤ 9`, `σ = 0.15n + 0.35` default |
| `borderInterpolate` | the iterative reflection loop, correct far outside the image |
| `absdiff` on `CV_32S` | saturates instead of wrapping (upstream's tip-commit fix) |
| `getStructuringElement` | per-row half-width, not a pointwise ellipse test |
| `getSobelKernels` | the in-place difference recurrence, not repeated convolution |
| `equalizeHist` | scale divides by `total - hist[first]`, not `total` |
| `adaptiveThreshold` | `cvCeil(C)` for BINARY, `cvFloor(C)` for BINARY_INV |
| `warpAffine` / `remap` | sample coordinates quantised to 1/32 px (`INTER_BITS`) |
| `pyrUp` | x4 gain compensating the zero-stuffing exactly |
| `Canny` NMS | integer direction test (`\|dy\|<<15` vs `\|dx\|*13573`), asymmetric `>` / `>=` |
| `Canny` L2gradient | thresholds are **squared**, magnitude is not rooted |
| `Laplacian` ksize 1/3 | two tabulated kernels, not derived ones |
| BMP | 4-byte row padding; positive header height means bottom-up rows |
| PNM | file samples are RGB, `Mat` is BGR |
| PNG Paeth | tie-break order is normative: `a`, then `b`, then `c` |
| PNG | RGB→BGR, and *any* alpha (including grey+alpha) becomes 4-channel BGRA |
| `RNG` | multiply-with-carry step reproduced exactly, so a seed replays upstream's sequence |
| contour `moments` | Green's-theorem integration with sign-carrying divisors |
| `completeMomentState` | upstream's algebraic shortcuts for `mu21`/`mu12`, not the naive sums |

Each of these has a test that fails if the shortcut is taken instead.

Two entries in that table need a caveat, both discovered by measurement rather
than reading:

- **`warpAffine`/`remap` quantisation is conditional.** The 1/32-pixel rule is
  upstream's portable path, and it is what `warpAffine` does at every depth.
  `remap` and `warpPerspective` on 8- and 16-bit unsigned images of one or two
  channels are taken by the HAL instead and use the exact fraction — see
  "The reference library is not upstream's source" above.
- **`resize` INTER_AREA rounds two different ways.** Halving is the *only*
  integer scale that rounds halves up: `ResizeAreaFastVec` engages at exactly
  two in both axes with one, three or four channels and covers the row with
  `(a + b + c + d + 2) >> 2`. Every other integer scale, and every two-channel
  image, falls through to `saturate_cast(sum * 1.f/area)` and rounds halves to
  even. Getting this wrong put every level of the DIS pyramid a grey level out.

## Kernel performance

The portable kernels were originally written for clarity and were roughly 100x
slower than OpenCV. Three things were costing almost all of it:

1. **`border_interpolate` per kernel tap.** A 7×7 `filter2D` called it 98 times
   per pixel per channel. Now the source is extrapolated **once** into a padded
   copy and the inner loop has no border logic at all.
2. **A typed slice rebuilt per tap.** `at_f64` walks
   `pixel → row_as → check_depth` on every access. Kernels now widen the whole
   matrix once via `cv_core::to_f64` and index a flat buffer.
3. **A depth `match` per stored pixel.** Stores go through
   `cv_core::mat_from_f64`, which dispatches once for the whole image.

Arithmetic still happens in `f64` in the same order, so results are unchanged —
the 453 tests include exact-value assertions and all still pass.

1280×720, median of 5:

| kernel | before | after | speedup |
|---|---|---|---|
| `filter2D` 7×7 c3 | 662.7 ms | 48.8 ms | **13.6×** |
| `dilate` 3×3 c1 | 75.4 ms | 5.1 ms | **14.9×** |
| `filter2D` 3×3 c3 | 138.8 ms | 14.2 ms | 9.8× |
| `cvtColor` BGR2GRAY | 15.8 ms | 1.9 ms | 8.3× |
| `threshold` c1 | 6.4 ms | 0.83 ms | 7.7× |
| `blur` 5×5 c3 | 95.6 ms | 16.0 ms | 6.0× |
| `GaussianBlur` 5×5 c3 | 95.0 ms | 15.9 ms | 6.0× |
| `sepFilter2D` 5×5 c3 | 94.6 ms | 16.0 ms | 5.9× |
| `resize` area ½ c3 | 15.1 ms | 2.8 ms | 5.5× |
| `cvtColor` BGR2HSV | 20.8 ms | 4.0 ms | 5.2× |
| `medianBlur` 3 c1 | 43.2 ms | 9.0 ms | 4.8× |
| `resize` 1.5× linear c3 | 11.6 ms | 11.6 ms | 1.0× (already tight) |

Reproduce with
`cargo run --release -p opencv-no-c-imgproc --example bench_kernels`.

They are still single-threaded and unvectorised, so there is a lot left — but
the asymptotic waste is gone.

## Optional acceleration via RLX (off by default)

`cv-accel` bridges the hot kernels to [RLX](https://github.com/MIT-RLX/rlx) and
exposes every backend it supports. It is **opt-in**, because it costs the thing
this project is named for:

```sh
cargo build                                  # pure Rust; links libSystem only
cargo build --features opencv-no-c-imgproc/rlx        # CPU backend
cargo build --features opencv-no-c-imgproc/rlx-all    # every backend
```

As a dependency, `opencv-no-c = { version = "0.1", features = ["rlx"] }`.
RLX comes from crates.io, pinned to exactly **0.2.17**: the published 0.2.18
does not compile (its `rlx-runtime` and `rlx-ir` were cut from different
commits), and a caret requirement would select it.

Measured with `otool -L`: the default build links **1** library (`libSystem`);
`rlx-all` links **15** — Accelerate, CoreML, Metal, MetalPerformanceShaders(Graph),
QuartzCore, CoreGraphics, CoreFoundation, Foundation, IOKit, `libc++`, `libobjc`,
`libbz2`, `libiconv`. RLX's `rlx-cpu` build script emits
`cargo:rustc-link-lib=framework=Accelerate` unconditionally on Apple targets —
it is not behind a Cargo feature — so *any* build reaching RLX links C.

The dependency is confined to `cv-accel`, so there is one place to audit, and
`cv_accel::links_c()` answers at runtime whether the current build is C-free.

### Backends

All **15** devices RLX defines are wired up. Every [`Backend`] variant exists in
every build, so code can name CUDA on a Mac and get a status instead of a
compile error:

| status | meaning |
|---|---|
| `Ready` | compiled in, and **verified** to compute correctly here |
| `NoDevice` | compiled in, but no working device |
| `NotCompiled` | feature not enabled |

`cargo run --release -p opencv-no-c-accel --features rlx-all --example backends` reports
the current machine. On an Apple-silicon Mac:

| backend | feature | status |
|---|---|---|
| metal | `rlx-metal` | ready |
| mlx | `rlx-mlx` | ready |
| vulkan | `rlx-vulkan` | ready |
| gpu (wgpu) | `rlx-gpu` | ready |
| webgpu | `rlx-webgpu` | ready |
| opengl | `rlx-opengl` | ready |
| ane | `rlx-coreml` | ready |
| oneapi | `rlx-oneapi` | ready |
| cpu | `rlx` | ready |
| cuda | `rlx-cuda` | no device |
| rocm | `rlx-rocm` | no device |
| xdna | `rlx-xdna` | no device |
| tpu | `rlx-tpu` | no device |
| directx | `rlx-directx` | no device |
| hexagon | `rlx-hexagon` | **not built** |

`rlx-all` enables fourteen of them. **Hexagon is excluded on purpose**:
`rlx-qnn`'s build script compiles a C shim and hard-fails unless `QNN_SDK_ROOT`
points at the Qualcomm AI Engine Direct SDK, so including it would break the
build for anyone without that SDK.

RLX's facade does not forward every backend its runtime defines — `xdna`,
`opengl`, `directx`, `webgpu` and `qnn` are missing from it — so `cv-accel`
also depends on `rlx-runtime` directly and switches them on there. Cargo
unifies features across the shared instance, so the facade picks them up.

### Availability is probed, not asked

RLX's `is_available(Device)` reports what was **compiled**, which is not the
same as what **works**. Two of its backend features are empty markers:

```toml
ane     = []   # Device::Ane => cfg!(any(feature = "coreml", feature = "ane"))
directx = []
```

Enable bare `ane` without `coreml` and `is_available(Device::Ane)` returns true
— then running panics with *"no backend registered for device ANE"*. Verified
directly. `directx` does the same on macOS.

So `Backend::availability()` does not take RLX's word for it: the first time a
backend is queried it runs a small convolution inside `catch_unwind` and checks
the answer, caching the verdict. Only a correct result earns `Ready`. That is
why `directx` shows `no device` above rather than `ready`-then-crash.

**`coreml` vs `ane`**: `rlx-coreml` is the one that pulls in the actual backend
crate and drives the Neural Engine. `Backend::Ane` maps to it. The bare `ane`
feature is a detection flag only, and this crate never relies on it.

### Why the CPU backend was slow (and the fix, upstream in RLX)

RLX's CPU convolution was 77 ms where this crate's *scalar, single-threaded*
kernel took 49 ms — on a 14-core machine. That is not a slow kernel; it is the
wrong algorithm being selected.

`rlx-cpu` defaults to **im2col + BLAS sgemm**, with a comment noting it is the
"measured CPU optimum for TinyConv / MNIST CNN shapes". For single-channel
image filtering that choice is pathological:

```text
1280x720, 7x7, C_in = C_out = 1
  patch matrix = [C_in*kh*kw] x [H_out*W_out] = 49 x 921600
               = 181 MB, to filter a 3.7 MB image  (49x expansion)
  the GEMM is  [1 x 49] @ [49 x 921600]
```

`M = 1` — that is a matrix-*vector* product. No blocking, no register tiling,
AMX idle, and 181 MB of scratch traffic. im2col earns its keep through reuse in
the `M` dimension (each output channel re-reads the patch matrix); with one
output channel there is nothing to reuse.

RLX already ships a direct convolution that needs zero scratch, but it was
reachable only via an opt-in `RLX_DIRECT_CONV=1` environment variable. Measured
(3 planes, padded 720p):

| kernel | im2col | direct | gain |
|---|---|---|---|
| 3×3 | 11.60 ms | 1.17 ms | 9.9× |
| 7×7 | 61.75 ms | 4.49 ms | **13.8×** |

**The fix is upstream, not here.** `crates/backends/rlx-cpu/src/thunk/ops/conv.rs`
gained a shape-aware predicate, `im2col_gemm_is_degenerate`, so the direct
kernel is chosen automatically when the GEMM would be too thin to amortise the
patch matrix (`c_out/groups <= 4` and `kh*kw >= 9`). The environment variable
still forces it. Wide-channel ML convolutions are untouched and still take
im2col, which remains the optimum there — verified by checksum and timing on
64→64, 32→32 and 3→16 channel shapes. All 261 `rlx-cpu` tests pass, and the
image-shape outputs are bit-identical to the naive reference.

That single change took RLX's CPU path from **0.52× (slower than portable) to
3.9× faster**.

### Throughput

`filter2D`, 3-channel `CV_8U`, RLX **CPU** backend, after both the graph cache
and the upstream dispatch fix:

| image | kernel | portable | RLX cpu | speedup |
|---|---|---|---|---|
| 1280×720 | 7×7 | 47.5 ms | 12.0 ms | **4.0×** |
| 1280×720 | 5×5 | 25.4 ms | 10.6 ms | 2.4× |
| 1280×720 | 3×3 | 14.3 ms | 10.8 ms | 1.3× |
| 640×480 | 7×7 | 15.8 ms | 4.0 ms | 4.0× |
| 640×480 | 3×3 | 4.0 ms | — | below gate |

Dispatch is gated on **total work** (`pixels × kernel area >= 5M`), not on size
and kernel separately. The marshalling cost — pad, widen, planarise, run,
de-planarise, store — scales with pixels, while the convolution it replaces
scales with pixels × kernel area, so a single work threshold is what actually
predicts the crossover.

Raw backend throughput without imgproc marshalling — 3-channel 720p, 5×5,
median of 9 after 3 warm-ups:

| backend | median | vs cpu |
|---|---|---|
| vulkan | 3.2 ms | 12× |
| mlx | 5.3 ms | 7.3× |
| metal | 6.3 ms | 6.1× |
| gpu (wgpu) | 7.7 ms | 5.0× |
| cpu | 38.7 ms | 1.0× |
| oneapi | 40.5 ms | 1.0× |
| ane | 43.9 ms | 0.9× |
| opengl | 247.3 ms | 0.16× |

(That CPU column predates the dispatch fix; the `cv-accel` probe uses a
4-channel-out shape that does not trip the degenerate-GEMM predicate.)

Worth stating plainly rather than burying:

- **oneAPI accelerates nothing here.** Its build script says *"native SPIR-V
  kernels not built … compute runs via the CPU reference path"*.
- **The Neural Engine is slower than the CPU**, and **OpenGL is 6× slower**.
- Graph compilation, not kernel time, dominated GPU dispatch until it was
  cached: on Vulkan compile was 32.2 ms against a 3.1 ms run.

### A note on the graph cache

`CompiledGraph::drop` reaches into RLX's own thread-locals, so destroying the
cache at thread exit can hit an already-destroyed TLS slot — and a panic inside
a destructor is escalated by Rust to an **abort**. The cache therefore holds its
graphs in `ManuallyDrop` and never destroys them. That matches the intended
process lifetime and is bounded by the number of distinct shapes in use.

### Correctness across backends

Enabling many backends is only useful if they agree. Every backend reporting
`Ready` is tested against a scalar reference written independently of both RLX
and `cv-imgproc`, at a `1e-4` tolerance — GPUs reassociate sums and use fused
multiply-add, so this is a bound, not equality.

Asking for a backend that is unavailable **errors** rather than silently
falling back to the CPU, because a quiet fallback makes a benchmark a lie.

RLX accumulates in `f32` where the portable kernel uses `f64`.
`filter_2d_portable` is public so the pure path can be pinned, and the
`rlx_matches_portable_*` tests bound the disagreement at ≤1 level for `CV_8U`
and `1e-5` for `CV_32F`. On the benchmark images the observed difference was
**0** at every size.

RLX's `Op::Conv` is a correlation, not a flipped convolution — the same
convention `filter2D` uses — so kernels pass through unchanged. There is a test
asserting exactly that, since a silent flip is invisible on the symmetric
kernels most filters use.

## Cross-checked against other implementations

Round-trip tests only prove the encoder and decoder agree with *each other* —
they would both pass if, say, the Paeth predictor were wrong self-consistently.
So the codecs are also checked against outside implementations:

- The DEFLATE decoder is fed streams produced by CPython's `zlib` at level 9 —
  genuine fixed- and dynamic-Huffman blocks with overlapping back-references,
  none of which this crate's own (stored-block) encoder can generate.
- `tests/reference_png.rs` decodes PNG fixtures written by CPython `zlib` with
  the scanline filters cycled through all five types, comparing against pixels
  computed independently of this crate.
- PNGs written by this crate were verified to decode correctly with macOS
  `sips`, `file(1)`, and Python's `zlib` (which validates the DEFLATE stream and
  Adler-32 trailer).
- Both decoders are fuzzed with thousands of random and truncated buffers, since
  they run on untrusted file content.

## Extensions beyond OpenCV

A small number of things here have **no upstream counterpart**. They are kept
separate and are excluded from the parity figure by construction — the report
only counts symbols the headers actually define.

**Cosine metrics** (`cv_core::metrics`) — OpenCV has no cosine distance; its
norm-type enum stops at L1/L2/Hamming, and the only "cosine" in its API is the
DCT. Included because it is the standard way to compare descriptors and
embeddings, where magnitude carries no information:

```rust
use opencv_no_c::core::{cosine_similarity, cosine_distances, nearest_by_cosine};

let s = cosine_similarity(&a, &b)?;              // [-1, 1]
let d = cosine_distances(&query, &descriptors)?; // 1xN row, distances in [0, 2]
let (idx, dist) = nearest_by_cosine(&query, &descriptors)?;
# Ok::<(), opencv_no_c::Error>(())
```

Cross-checked against **scikit-learn** and **SciPy**: max absolute difference
`2.2e-16` (one ULP) across 40 descriptor rows, including parallel, opposite and
zero vectors (`cargo run --release -p opencv-no-c-core --example cosine_check -- <dir>`).

One deliberate divergence from SciPy: a zero-length vector yields similarity
`0` / distance `1` rather than `NaN`. SciPy returns `NaN`, which silently
poisons any downstream sort; scikit-learn returns `0`, and this follows
scikit-learn.

## The API divergences in detail

Summarised under "This is not a drop-in replacement" above; the mechanics:

**Errors are returned, not thrown.** Every fallible entry point returns
`Result<T, Error>`; `Error::code` carries the same numeric value
`cv::Exception::code` would.

**Matrices own their pixels.** `cv::Mat` is refcounted with shallow copy
semantics — `Mat b = a;` shares storage, and writing through `b` mutates `a`.
That needs interior mutability to express in safe Rust. Instead:

| upstream | here |
|---|---|
| `Mat b = a;` (shares) | `Mat::clone` (deep copy) |
| `Mat b = a(roi);` (shares) | `mat.roi(rect)` → `MatView` |
| write through a shared `Mat` | `mat.roi_mut(rect)` → `MatViewMut` |

Algorithms are written against the `MatLike`/`MatLikeMut` traits, so they accept
owned matrices and borrowed ROIs interchangeably.

## Usage

```rust
use opencv_no_c::prelude::*;
use opencv_no_c::imgproc::{cvt_color, resize, ColorConversion, Interpolation};

let img = Mat::new_with(64, 64, types::CV_8UC3, Scalar::new(20.0, 60.0, 200.0, 0.0))?;
let gray = cvt_color(&img, ColorConversion::Bgr2Gray)?;
let small = resize(&gray, Size::new(32, 32), 0.0, 0.0, Interpolation::Area)?;
# Ok::<(), opencv_no_c::Error>(())
```

With file I/O:

```rust
use opencv_no_c::imgcodecs::{imread, imwrite, ImreadMode};
use opencv_no_c::imgproc::canny;

let img = imread("input.png", ImreadMode::Grayscale)?;
let edges = canny(&img, 50.0, 150.0, 3, false)?;
imwrite("edges.png", &edges)?;
# Ok::<(), opencv_no_c::Error>(())
```

## Parity tooling

```sh
cargo xtask report                  # per-module parity table
cargo xtask extract                 # write parity/manifest.json
cargo xtask report --opencv /path   # point at a different checkout
```

`extract` scans the upstream headers for binding-exported symbols
(`CV_EXPORTS_W`, `CV_WRAP`, `CV_EXPORTS_AS`), enum constants and `CV_*` macros.
`report` diffs that against what this workspace claims.

**Claims live in doc comments.** Every ported item names its upstream
counterpart in backticks:

```rust
/// `cv::resize(src, dst, dsize, fx, fy, interpolation)`.
pub fn resize(...) { }
```

The report counts exactly what a reviewer reads, so the mapping cannot drift
from the code. Adding a function without its doc claim shows up as unported.

### Known limits of the scanner

It is a heuristic line scanner, not a C++ parser:

- Classes declared `CV_EXPORTS` (rather than `CV_EXPORTS_W`) are excluded, since
  it targets the binding-visible surface. `cv::Mat` itself is in this category.
- Declarations split across lines before the `(` are counted as unparsed;
  `extract` prints the count per module (~1% of symbols).
- A class counts as one symbol, so its method surface is understated.
- Only `///` item docs count as claims. Module-level `//!` prose does not,
  because prose legitimately names symbols a module does *not* implement — a
  header reading "OpenCV has no cosine metric, its norms are `NORM_HAMMING`…"
  was being scored as having ported `NORM_HAMMING`. Tightening this removed two
  false claims, including `LINE_AA`, which the same file explicitly documents
  as **not** ported.
- Constants generated by macro (`CV_8UC3` and the rest of the type table) are
  invisible to the scanner, since their docs are built with `concat!` rather
  than written as `///` lines. The figure is therefore an undercount, which is
  the safe direction for a parity claim.

The denominator is therefore approximate. It is stable run-to-run, which is what
matters for tracking progress.

It also counts *symbols*, not capability. Adding PNG moved the parity number by
zero, because `imread`/`imwrite`/`imdecode`/`imencode` were already claimed —
yet it is the difference between the crate being usable on real images and not.
Read the table as coverage of the API surface, not of the work.

## Layout

One crate per upstream module, so a caller pulls in only what it uses.
`opencv-no-c` re-exports all of them under upstream's module names.

```
crates/cv-core/       modules/core
crates/cv-imgproc/    modules/imgproc
crates/cv-imgcodecs/  modules/imgcodecs — PNG, BMP, PNM, JPEG, plus DEFLATE
crates/cv-calib3d/    modules/calib3d
crates/cv-features2d/ modules/features2d
crates/cv-objdetect/  modules/objdetect
crates/cv-photo/      modules/photo
crates/cv-video/      modules/video
crates/cv-videoio/    modules/videoio
crates/cv-highgui/    modules/highgui
crates/cv-ml/         modules/ml
crates/cv-flann/      modules/flann
crates/cv-dnn/        modules/dnn
crates/cv-gapi/       modules/gapi
crates/cv-stitching/  modules/stitching
crates/cv-accel/      optional RLX bridge — the only crate that can link C
crates/cv-samples/    OpenCV's sample programs, ported; builds for wasm32
crates/opencv-no-c/   facade re-exporting the above
xtask/                parity extraction and reporting
tools/                reference generators driven by a real cv2 install
parity/manifest.json  generated upstream API snapshot
```

## Adding to the port

1. Read the upstream implementation — not just the header. The fixed-point
   details above were all invisible from the declaration.
2. Write the Rust version against `MatLike`/`MatLikeMut` so it works on ROIs.
3. Name the upstream symbol in a backticked doc comment.
4. Add a test that would fail against a plausible-but-wrong implementation
   (a float shortcut, a half-away-from-zero round, a missing edge clamp).
5. `cargo xtask report` to confirm the number moved.
6. **Add a differential check.** A unit test written beside the code tests the
   code's own theory; only `tools/gen_reference.py` plus
   `examples/parity_check.rs` tests whether that theory is OpenCV's. Anything
   outside that harness should be read as unverified however many unit tests it
   has.

## License

Apache-2.0, matching upstream OpenCV. See [LICENSE](LICENSE).

The images under `crates/cv-samples/assets/` are OpenCV's own `samples/data`,
converted losslessly to PNG and redistributed under the same licence.
