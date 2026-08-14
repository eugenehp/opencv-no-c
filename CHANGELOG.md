# Changelog

Notable changes, newest first. Versions follow [semver](https://semver.org).

## 0.1.0 — unreleased

First release. A pure-Rust port of OpenCV 4.15.0-dev, verified against a real
cv2 4.13.0 install.

### Coverage

- Every symbol in upstream's binding-visible headers is claimed: **4607/4607**
  across all fifteen modules (`cargo xtask report`).
- **469 differential comparisons** against cv2: 298 bit-exact, 171 inside a
  tolerance whose cause is documented at the call site, 0 unexplained. That
  covers roughly 169 distinct function families.
- 1730 tests, clippy clean, `cargo doc` clean.

### Ported from upstream source rather than reconstructed

`cv::watershed`, `cv::DISOpticalFlow` and `cv::VariationalRefinement`,
`cv::TonemapMantiuk`, Navier-Stokes and Telea inpainting, VNG and edge-aware
demosaicing, the FAST corner score, the ORB sampling pattern, HOG, the
multi-band blender, and CamShift. Several of these previously existed as
plausible reimplementations that passed their own tests and did not match cv2.

### Behaviours reproduced that are not in upstream's source

The reference build runs KleidiCV ARM kernels for some functions. `remap` and
`warpPerspective` on 8- and 16-bit unsigned images of one or two channels
interpolate from the exact sub-pixel fraction and round halves away from zero,
where `imgwarp.cpp` quantises to 1/32 and rounds to even. Both paths are
implemented, selected by depth and channel count.

### Fixed

- `resize` with `INTER_AREA` rounded halves up at every integer scale. Only
  halving does; every other scale rounds to even.
- `randu` on `CV_32F`/`CV_64F` drew from `RNG::uniform`, which takes two
  32-bit steps and reads them unsigned. `RNG::fill` takes one and reads it
  signed, biased by the midpoint of the range — so every element was offset by
  half the range. Both depths are now sequence-exact.
- `watershed` used the sum of channel differences rather than the largest, did
  not overwrite the caller's border, and queued seeds once per adjacency
  instead of once per pixel.

### Known gaps

- JPEG decode is not bit-exact: grayscale within one level on 1.8% of pixels,
  colour within two on 39%. PNG is exact.
- `randn` uses Box-Muller; upstream uses the Ziggurat method, so the
  distribution matches and the sequence does not.
- `cv::Canny(dx, dy, …)`, the precomputed-gradient overload, is not implemented.
- One `warpPerspective` pixel in 19200 differs by a level, inside the HAL's
  reciprocal precision.
- The features2d detectors, `videoio`, `dnn`, `highgui` and imgcodecs
  round-trips have no differential coverage.

### Added

- `dropin` feature: upstream's names (`cvtColor`, `GaussianBlur`, `Canny`, …)
  over the same implementations, with errors unwinding as a panic carrying the
  `Error` rather than being returned. `catch(|| …)` recovers one as a `Result`;
  a panic that is not an `Error` is re-raised rather than absorbed.
- `abi` feature: a C-callable surface (`cvno_*`) plus
  `crates/opencv-no-c/include/opencv_no_c.h`, building `libopencv_no_c.a` and
  `.dylib`. Fallible calls return `cv::Error::Code` unchanged; handles are
  opaque and caller-owned; nothing unwinds across the boundary. Verified by
  compiling and running both a C and a C++ program against the static library.
  This is not OpenCV's C++ ABI, which Rust cannot reproduce, and symbols are
  prefixed `cvno_` so linking alongside a real `libopencv_core` gives a link
  error rather than a silently-resolved wrong symbol.
- `ColorConversion::from_i32` / `to_i32`, for call sites where the code arrives
  as an integer.
- `compat` feature, giving `cv_core::compat::SharedMat` — `cv::Mat`'s
  refcounted storage model, where cloning shares, `deep_clone` is
  `Mat::clone()`, and an ROI writes through into its parent. Its borrow guards
  implement `MatLike`/`MatLikeMut`, so existing functions take them unchanged.
  Additive by construction: a new type rather than a flag on `Mat`, because
  Cargo features unify across the dependency graph and changing `Mat::clone`
  under one would alter dependents that never asked for it.

### Not a drop-in replacement

Errors are returned rather than thrown, `Mat` owns its pixels rather than
sharing them on copy, and names are snake_case. See the README.
