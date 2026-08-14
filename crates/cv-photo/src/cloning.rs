//! Seamless cloning and its colour variants
//! (`modules/photo/src/seamless_cloning.cpp`).
//!
//! Poisson image editing (Pérez et al., 2003): rather than pasting pixels, the
//! source region's **gradient** is transplanted and the result reintegrated
//! under the destination's boundary values. The patch therefore takes on the
//! destination's colour and lighting while keeping the source's detail, which
//! is what makes the seam disappear.
//!
//! Concretely, for the mixed region Ω with boundary ∂Ω the routines solve
//!
//! ```text
//! ∇²f = ∇·v   inside Ω,     f = destination   on ∂Ω
//! ```
//!
//! where `v` is the guidance field: the source gradient for [`seamless_clone`],
//! a channel-scaled source gradient for [`color_change`], and a suppressed one
//! for [`texture_flattening`].
//!
//! # The solver
//!
//! Gauss–Seidel with successive over-relaxation, iterated to a residual
//! tolerance. Upstream uses a DCT-based direct solve, which is faster and
//! reaches the same fixed point; the iterative form is used here because it
//! needs no separate transform and its convergence is checkable. Results agree
//! to a few levels rather than exactly, which is why these are not in the
//! bit-exact set.

use cv_core::{cv_assert, Mat, MatLike, Result};

/// `cv::SeamlessCloneFlags` — which guidance field to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum CloneMethod {
    /// `NORMAL_CLONE` — transplant the source gradient unchanged.
    #[default]
    Normal = 1,
    /// `MIXED_CLONE` — at each pixel keep whichever of the source and
    /// destination gradients is stronger.
    ///
    /// The one to use when the source has holes or thin structure: the
    /// destination's own texture shows through instead of being flattened.
    Mixed = 2,
    /// `MONOCHROME_TRANSFER` — take the source's texture but not its colour.
    MonochromeTransfer = 3,
}

/// Solve `∇²f = div` inside the mask with `f = boundary` outside.
///
/// Conjugate gradients, not Gauss-Seidel. The distinction matters more than it
/// looks: a relaxation's per-sweep update goes small long before its
/// *low-frequency* error does, so a "largest update < tol" stopping rule
/// reports success on a solution that is still tens of levels out. That is
/// invisible for a `NORMAL_CLONE`, whose guidance field is an actual gradient
/// field and therefore nearly integrable, and glaring for `MIXED_CLONE`, whose
/// field is stitched together from two different ones and is not. The same
/// system solved exactly agrees with cv2 to a single level.
///
/// The operator is the 5-point Laplacian restricted to the masked region, with
/// the boundary values folded into the right-hand side, and it is negated so
/// the matrix is positive definite.
fn poisson_solve(
    boundary: &[f32],
    div: &[f32],
    mask: &[bool],
    h: usize,
    w: usize,
    tol: f32,
) -> Vec<f32> {
    let n = h * w;
    let mut f = boundary.to_vec();

    // `-A x`, where `A` is the Laplacian over the masked region. Values outside
    // the region are boundary data and belong in `b`, not in the operator.
    let apply = |x: &[f32], out: &mut [f32]| {
        for i in 0..n {
            if !mask[i] {
                out[i] = 0.0;
                continue;
            }
            let mut acc = 4.0 * x[i];
            for j in [i - 1, i + 1, i - w, i + w] {
                if mask[j] {
                    acc -= x[j];
                }
            }
            out[i] = acc;
        }
    };

    // b = -(div - sum of boundary neighbours)
    let mut b = vec![0.0f32; n];
    for i in 0..n {
        if !mask[i] {
            continue;
        }
        let mut acc = div[i];
        for j in [i - 1, i + 1, i - w, i + w] {
            if !mask[j] {
                acc -= boundary[j];
            }
        }
        b[i] = -acc;
    }

    let mut x = vec![0.0f32; n];
    for i in 0..n {
        if mask[i] {
            x[i] = boundary[i];
        }
    }
    let mut ax = vec![0.0f32; n];
    apply(&x, &mut ax);
    let mut r: Vec<f32> = (0..n).map(|i| if mask[i] { b[i] - ax[i] } else { 0.0 }).collect();
    let mut p = r.clone();
    let dot = |u: &[f32], v: &[f32]| -> f64 {
        (0..n).map(|i| f64::from(u[i]) * f64::from(v[i])).sum()
    };
    let mut rr = dot(&r, &r);
    let target = f64::from(tol) * f64::from(tol) * dot(&b, &b).max(1.0);
    let mut ap = vec![0.0f32; n];
    // The Laplacian's condition number grows with the region's diameter, and
    // conjugate gradients needs about its square root in iterations; a few
    // thousand covers any mask that fits in an image.
    for _ in 0..4000 {
        if rr <= target {
            break;
        }
        apply(&p, &mut ap);
        let pap = dot(&p, &ap);
        if pap.abs() <= f64::MIN_POSITIVE {
            break;
        }
        let alpha = (rr / pap) as f32;
        for i in 0..n {
            if mask[i] {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
        }
        let new_rr = dot(&r, &r);
        let beta = (new_rr / rr) as f32;
        for i in 0..n {
            if mask[i] {
                p[i] = r[i] + beta * p[i];
            }
        }
        rr = new_rr;
    }
    for i in 0..n {
        if mask[i] {
            f[i] = x[i];
        }
    }
    f
}

/// Read one channel of an image into a flat `f32` plane.
fn plane(img: &(impl MatLike + ?Sized), c: i32) -> Result<Vec<f32>> {
    let (h, w) = (img.rows() as usize, img.cols() as usize);
    let mut v = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            v[y * w + x] = img.at_f64(y as i32, x as i32, c)? as f32;
        }
    }
    Ok(v)
}

/// `BORDER_REFLECT_101` over `i64` indices, which is what the Poisson solver
/// carries. The shared `cv_core::reflect101` works in `i32`; the conversion is
/// lossless for any image this crate can hold.
fn reflect101(i: i64, n: usize) -> usize {
    cv_core::reflect101(i as i32, n as i32) as usize
}

/// Forward difference along x, as upstream's `computeGradientX` kernel
/// `[0, -1, 1]` through `filter2D`.
fn forward_x(p: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            g[y * w + x] = p[y * w + reflect101(x as i64 + 1, w)] - p[y * w + x];
        }
    }
    g
}

/// Forward difference along y.
fn forward_y(p: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            g[y * w + x] = p[reflect101(y as i64 + 1, h) * w + x] - p[y * w + x];
        }
    }
    g
}

/// Backward difference along x, upstream's `computeLaplacianX` kernel
/// `[-1, 1, 0]`. Applying it to a forward difference gives the 5-point
/// Laplacian the solver inverts, which is why the two must stay paired.
fn backward_x(p: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            g[y * w + x] = p[y * w + x] - p[y * w + reflect101(x as i64 - 1, w)];
        }
    }
    g
}

/// Backward difference along y.
fn backward_y(p: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            g[y * w + x] = p[y * w + x] - p[reflect101(y as i64 - 1, h) * w + x];
        }
    }
    g
}

/// Erode a 0/1 mask with a 3x3 rectangle, `iterations` times.
fn erode3(m: &[f32], h: usize, w: usize, iterations: usize) -> Vec<f32> {
    let mut cur = m.to_vec();
    for _ in 0..iterations {
        let mut next = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                let mut lo = f32::MAX;
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let yy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                        let xx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                        lo = lo.min(cur[yy * w + xx]);
                    }
                }
                next[y * w + x] = lo;
            }
        }
        cur = next;
    }
    cur
}

/// Shared body of the three cloning entry points.
///
/// # The domain is the mask's bounding box, not the mask
///
/// This is the part that is easy to get wrong and impossible to see in a
/// diagram. Upstream crops to the mask's bounding rectangle, builds a guidance
/// field over the *whole* rectangle — the destination's own gradient outside
/// the mask, the patch's inside — and integrates that with the rectangle's
/// border as the Dirichlet boundary. So the result differs from the
/// destination at pixels that are outside the mask but inside the box, and the
/// interior solution is not the same one you get from imposing the boundary at
/// the mask edge. Solving only inside the mask is a *different* and
/// better-posed problem, and it does not match cv2.
fn clone_impl<S, D, M>(
    src: &S,
    dst: &D,
    mask: &M,
    offset: (i32, i32),
    method: CloneMethod,
    scale: [f32; 3],
    crop_to_mask: bool,
) -> Result<Mat>
where
    S: MatLike + ?Sized,
    D: MatLike + ?Sized,
    M: MatLike + ?Sized,
{
    cv_assert!(dst.channels() == 3, "seamlessClone works on 3-channel images");
    cv_assert!(
        src.size() == mask.size(),
        "seamlessClone: the mask must match the source, {:?} vs {:?}",
        src.size(),
        mask.size()
    );
    let (dh, dw) = (dst.rows() as usize, dst.cols() as usize);
    let (sh, sw) = (src.rows() as usize, src.cols() as usize);
    let mut out = Mat::copy_from(dst)?;
    if sh < 3 || sw < 3 {
        return Ok(out);
    }

    // `seamlessClone` forces the mask's one-pixel frame to zero before taking
    // the bounding box, so the box always has a clean boundary to solve
    // against. `colorChange` and its siblings have no such step — they solve
    // over the whole image, whose own border is the boundary — so zeroing the
    // frame there would discard a ring of the region the caller asked to edit.
    let mut m = vec![0.0f32; sh * sw];
    let (lo, hi_y, hi_x) = if crop_to_mask { (1, sh - 1, sw - 1) } else { (0, sh, sw) };
    for y in lo..hi_y {
        for x in lo..hi_x {
            if mask.at_f64(y as i32, x as i32, 0)? > 0.0 {
                m[y * sw + x] = 1.0;
            }
        }
    }
    let (mut x0, mut y0, mut x1, mut y1) = (sw, sh, 0usize, 0usize);
    for y in 0..sh {
        for x in 0..sw {
            if m[y * sw + x] > 0.0 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if x0 > x1 || y0 > y1 {
        return Ok(out);
    }
    // `seamlessClone` crops to the mask's bounding box because the patch has
    // to be *moved*; `colorChange` and its siblings edit in place and solve
    // over the whole image, so their boundary is the image border.
    let (x0, y0, rw, rh) = if crop_to_mask {
        (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    } else {
        (0, 0, sw, sh)
    };

    // `offset` names the centre of the *bounding box* in the destination.
    let (lx, ty) = if crop_to_mask {
        (offset.0 - (rw / 2) as i32, offset.1 - (rh / 2) as i32)
    } else {
        (0, 0)
    };
    cv_assert!(
        lx >= 0 && ty >= 0 && lx as usize + rw <= dw && ty as usize + rh <= dh,
        "seamlessClone: the {rw}x{rh} patch centred at {offset:?} falls outside the {dw}x{dh} destination"
    );
    let (lx, ty) = (lx as usize, ty as usize);

    // Upstream erodes the mask three times with a 3x3 rectangle before using it
    // to select gradients, which pulls the seam a few pixels inside the mask.
    let bmask_full: Vec<f32> = (0..rh * rw)
        .map(|i| m[(y0 + i / rw) * sw + x0 + i % rw])
        .collect();
    // The erosion runs on the mask *in the context of the whole image*, not on
    // the cropped window. Upstream passes `mask(roi_s)` — a submatrix view —
    // to `erode`, and OpenCV's morphology reads a view's surroundings from its
    // parent rather than applying the border rule. Since the ROI is the mask's
    // own bounding box, everything just outside it is zero, so the erosion eats
    // three pixels in from the box edge as well as from the mask's boundary.
    // Eroding the crop on its own instead leaves the box edge untouched: for a
    // mask that fills its box, upstream is left with *nothing* and this was
    // left with everything, which is the whole difference between transferring
    // the patch and transferring nothing.
    let eroded_full = erode3(&m, sh, sw, 3);
    let bmask: Vec<f32> = (0..rh * rw)
        .map(|i| eroded_full[(y0 + i / rw) * sw + x0 + i % rw])
        .collect();

    // MONOCHROME_TRANSFER transfers the patch's *luminance* gradient, so the
    // region keeps the destination's hue. The grey must come from the same
    // fixed-point BGR2GRAY cv2 uses — a float `0.114b + 0.587g + 0.299r` is off
    // by a level often enough that the seam moves.
    let mono = if method == CloneMethod::MonochromeTransfer {
        let mut patch_bgr = Mat::zeros(rh as i32, rw as i32, dst.mat_type())?;
        for y in 0..rh {
            for x in 0..rw {
                if bmask_full[y * rw + x] > 0.0 {
                    for c in 0..3 {
                        cv_core::arithm::set_channel(
                            &mut patch_bgr,
                            y as i32,
                            x as i32,
                            c,
                            src.at_f64((y0 + y) as i32, (x0 + x) as i32, c)?,
                        )?;
                    }
                }
            }
        }
        let g = cv_imgproc::color::cvt_color(
            &patch_bgr,
            cv_imgproc::color::ColorConversion::Bgr2Gray,
        )?;
        Some(plane(&g, 0)?)
    } else {
        None
    };

    #[allow(clippy::needless_range_loop)] // `c` is the channel index, used for both planes and `scale`
    for c in 0..3 {
        let mut dest = vec![0.0f32; rh * rw];
        let mut patch = vec![0.0f32; rh * rw];
        for y in 0..rh {
            for x in 0..rw {
                dest[y * rw + x] = dst.at_f64((ty + y) as i32, (lx + x) as i32, c as i32)? as f32;
                // `src(roi_s).copyTo(sourceROI, mask(roi_s))` — the patch is
                // zero outside the mask, before erosion.
                if bmask_full[y * rw + x] > 0.0 {
                    patch[y * rw + x] = match &mono {
                        Some(g) => g[y * rw + x],
                        None => src.at_f64((y0 + y) as i32, (x0 + x) as i32, c as i32)? as f32,
                    };
                }
            }
        }

        let gxd = forward_x(&dest, rh, rw);
        let gyd = forward_y(&dest, rh, rw);
        let mut gxp = forward_x(&patch, rh, rw);
        let mut gyp = forward_y(&patch, rh, rw);

        for i in 0..rh * rw {
            let k = bmask[i];
            match method {
                CloneMethod::Normal | CloneMethod::MonochromeTransfer => {
                    gxp[i] *= k;
                    gyp[i] *= k;
                }
                // Upstream's rule really is on `|gx - gy|`, not on the gradient
                // magnitude: whichever field varies more *between* its two
                // components wins. Comparing magnitudes instead is the natural
                // reading and gives a different picture.
                CloneMethod::Mixed => {
                    if (gxp[i] - gyp[i]).abs() > (gxd[i] - gyd[i]).abs() {
                        gxp[i] *= k;
                        gyp[i] *= k;
                    } else {
                        gxp[i] = gxd[i] * k;
                        gyp[i] = gyd[i] * k;
                    }
                }
            }
            gxp[i] *= scale[c];
            gyp[i] *= scale[c];
        }

        // The destination's own gradient, kept only outside the mask.
        let fx: Vec<f32> = (0..rh * rw).map(|i| gxd[i] * (1.0 - bmask[i]) + gxp[i]).collect();
        let fy: Vec<f32> = (0..rh * rw).map(|i| gyd[i] * (1.0 - bmask[i]) + gyp[i]).collect();
        let lap_x = backward_x(&fx, rh, rw);
        let lap_y = backward_y(&fy, rh, rw);
        let div: Vec<f32> = (0..rh * rw).map(|i| lap_x[i] + lap_y[i]).collect();

        // Every interior pixel of the box is solved for, not just the masked
        // ones; the box border is the boundary condition.
        let interior: Vec<bool> = (0..rh * rw)
            .map(|i| {
                let (y, x) = (i / rw, i % rw);
                y > 0 && x > 0 && y + 1 < rh && x + 1 < rw
            })
            .collect();
        let solved = poisson_solve(&dest, &div, &interior, rh, rw, 1e-6);
        for y in 0..rh {
            for x in 0..rw {
                let v = f64::from(solved[y * rw + x]).clamp(0.0, 255.0);
                cv_core::arithm::set_channel(
                    &mut out,
                    (ty + y) as i32,
                    (lx + x) as i32,
                    c as i32,
                    v.round(),
                )?;
            }
        }
    }
    Ok(out)
}

/// `cv::seamlessClone(src, dst, mask, p, blend, flags)`.
///
/// `p` is the **centre** of where the source lands in the destination, not its
/// top-left corner — a detail worth stating, since passing a corner puts the
/// patch half an image away from where it was wanted.
pub fn seamless_clone<S, D, M>(
    src: &S,
    dst: &D,
    mask: &M,
    p: (i32, i32),
    method: CloneMethod,
) -> Result<Mat>
where
    S: MatLike + ?Sized,
    D: MatLike + ?Sized,
    M: MatLike + ?Sized,
{
    clone_impl(src, dst, mask, p, method, [1.0, 1.0, 1.0], true)
}

/// `cv::colorChange(src, mask, dst, red_mul, green_mul, blue_mul)`.
///
/// Scales each channel's gradient before reintegrating, which shifts the hue of
/// the masked region while keeping its texture and its boundary colour.
pub fn color_change<S, M>(
    src: &S,
    mask: &M,
    red_mul: f32,
    green_mul: f32,
    blue_mul: f32,
) -> Result<Mat>
where
    S: MatLike + ?Sized,
    M: MatLike + ?Sized,
{
    let centre = (src.cols() / 2, src.rows() / 2);
    // Channel order is BGR, so the blue multiplier comes first.
    clone_impl(src, src, mask, centre, CloneMethod::Normal, [blue_mul, green_mul, red_mul], false)
}

/// `cv::illuminationChange(src, mask, dst, alpha, beta)`.
///
/// Compresses the gradient by `alpha * |∇|^beta`, which flattens strong
/// specular highlights without touching the region's colour.
pub fn illumination_change<S, M>(src: &S, mask: &M, alpha: f32, beta: f32) -> Result<Mat>
where
    S: MatLike + ?Sized,
    M: MatLike + ?Sized,
{
    cv_assert!(src.channels() == 3, "illuminationChange works on 3-channel images");
    let (h, w) = (src.rows() as usize, src.cols() as usize);
    let mut region = vec![false; h * w];
    for y in 0..h {
        for x in 0..w {
            region[y * w + x] = mask.at_f64(y as i32, x as i32, 0)? > 0.0;
        }
    }

    let mut out = Mat::copy_from(src)?;
    for c in 0..3i32 {
        let sp = plane(src, c)?;
        // Rescale each gradient by alpha*|g|^beta before taking the divergence.
        let mut div = vec![0.0f32; h * w];
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                let mut total = 0.0;
                for &step in &[1usize, w] {
                    for &at in &[i, i + step] {
                        let g = sp[at] - sp[at - step];
                        let scaled = alpha * g.abs().max(1e-4).powf(beta) * g.signum();
                        total += if at == i { -scaled } else { scaled };
                    }
                }
                div[i] = total;
            }
        }
        let solved = poisson_solve(&sp, &div, &region, h, w, 1e-3);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if region[i] {
                    let v = f64::from(solved[i]).clamp(0.0, 255.0);
                    cv_core::arithm::set_channel(&mut out, y as i32, x as i32, c, v)?;
                }
            }
        }
    }
    Ok(out)
}

/// `cv::textureFlattening(src, mask, dst, low_threshold, high_threshold, kernel_size)`.
///
/// Keeps only the gradients that sit on a detected edge and zeroes the rest, so
/// the region loses its texture but keeps its shape — the washed-out,
/// illustration-like look.
pub fn texture_flattening<S, M>(
    src: &S,
    mask: &M,
    low_threshold: f64,
    high_threshold: f64,
    kernel_size: i32,
) -> Result<Mat>
where
    S: MatLike + ?Sized,
    M: MatLike + ?Sized,
{
    cv_assert!(src.channels() == 3, "textureFlattening works on 3-channel images");
    cv_assert!(
        kernel_size > 0 && kernel_size % 2 == 1,
        "kernel_size must be odd and positive, got {kernel_size}"
    );
    let (h, w) = (src.rows() as usize, src.cols() as usize);

    // Edges come from a gradient-magnitude threshold rather than full Canny:
    // only the binary "is this an edge" answer is used, and hysteresis makes
    // no difference to a mask that is immediately multiplied into a gradient.
    let mut gray = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            gray[y * w + x] = ((src.at_f64(y as i32, x as i32, 0)?
                + src.at_f64(y as i32, x as i32, 1)?
                + src.at_f64(y as i32, x as i32, 2)?)
                / 3.0) as f32;
        }
    }
    let mut edge = vec![false; h * w];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            let gx = gray[i + 1] - gray[i - 1];
            let gy = gray[i + w] - gray[i - w];
            let mag = f64::from((gx * gx + gy * gy).sqrt());
            edge[i] = mag >= low_threshold && mag <= high_threshold.max(low_threshold);
        }
    }

    let mut region = vec![false; h * w];
    for y in 0..h {
        for x in 0..w {
            region[y * w + x] = mask.at_f64(y as i32, x as i32, 0)? > 0.0;
        }
    }

    let mut out = Mat::copy_from(src)?;
    for c in 0..3i32 {
        let sp = plane(src, c)?;
        let mut div = vec![0.0f32; h * w];
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                let mut total = 0.0;
                for &step in &[1usize, w] {
                    for &at in &[i, i + step] {
                        // Gradients away from an edge are dropped entirely.
                        let g = if edge[at] { sp[at] - sp[at - step] } else { 0.0 };
                        total += if at == i { -g } else { g };
                    }
                }
                div[i] = total;
            }
        }
        let solved = poisson_solve(&sp, &div, &region, h, w, 1e-3);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if region[i] {
                    let v = f64::from(solved[i]).clamp(0.0, 255.0);
                    cv_core::arithm::set_channel(&mut out, y as i32, x as i32, c, v)?;
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::{CV_8UC1, CV_8UC3};

    fn solid(h: i32, w: i32, bgr: [u8; 3]) -> Mat {
        let mut v = Vec::with_capacity((h * w * 3) as usize);
        for _ in 0..h * w {
            v.extend_from_slice(&bgr);
        }
        Mat::from_vec(h, w, 3, &v).unwrap()
    }

    fn centre_mask(h: i32, w: i32, r: i32) -> Mat {
        let mut m = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in (h / 2 - r)..(h / 2 + r) {
            for x in (w / 2 - r)..(w / 2 + r) {
                cv_core::arithm::set_channel(&mut m, y, x, 0, 255.0).unwrap();
            }
        }
        m
    }

    /// The defining property: the patch takes the destination's colour.
    ///
    /// A flat source on a flat destination has zero guidance gradient, so the
    /// solution is the destination's own value throughout — the source colour
    /// vanishes entirely. If it did not, the seam would still be visible.
    #[test]
    fn a_flat_patch_takes_the_destinations_colour() {
        let src = solid(24, 24, [200, 40, 40]);
        let dst = solid(48, 48, [30, 180, 90]);
        let mask = centre_mask(24, 24, 8);
        let out = seamless_clone(&src, &dst, &mask, (24, 24), CloneMethod::Normal).unwrap();
        assert_eq!(out.size(), dst.size());
        // The centre of the pasted region must have become the destination's.
        for c in 0..3 {
            let got = out.at_f64(24, 24, c).unwrap();
            let want = dst.at_f64(24, 24, c).unwrap();
            assert!((got - want).abs() <= 1.0, "channel {c}: {got} vs {want}");
        }
    }

    #[test]
    fn pixels_outside_the_mask_are_untouched() {
        let src = solid(16, 16, [200, 40, 40]);
        let dst = solid(48, 48, [30, 180, 90]);
        let mask = centre_mask(16, 16, 4);
        let out = seamless_clone(&src, &dst, &mask, (24, 24), CloneMethod::Normal).unwrap();
        // A corner is far from the patch and must be bit-identical.
        for c in 0..3 {
            assert_eq!(out.at_f64(0, 0, c).unwrap(), dst.at_f64(0, 0, c).unwrap());
            assert_eq!(out.at_f64(47, 47, c).unwrap(), dst.at_f64(47, 47, c).unwrap());
        }
    }

    /// Detail survives even though the flat colour does not.
    #[test]
    fn source_texture_is_carried_into_the_result() {
        let (h, w) = (40i32, 40i32);
        let mut v = Vec::new();
        for _y in 0..h {
            for x in 0..w {
                // A hard stripe, so the transplanted gradient is unmistakable.
                let p = if (x / 4) % 2 == 0 { 40u8 } else { 210 };
                v.extend_from_slice(&[p, p, p]);
            }
        }
        let src = Mat::from_vec(h, w, 3, &v).unwrap();
        let dst = solid(64, 64, [120, 120, 120]);
        let mask = centre_mask(h, w, 12);
        let out = seamless_clone(&src, &dst, &mask, (32, 32), CloneMethod::Normal).unwrap();

        // Inside the patch the stripes must still vary; a flat result would
        // mean the gradient was lost.
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        for x in 26..38 {
            let p = out.at_f64(32, x, 0).unwrap();
            lo = lo.min(p);
            hi = hi.max(p);
        }
        assert!(hi - lo > 40.0, "the stripes should survive: range {lo}..{hi}");
    }

    #[test]
    fn every_clone_method_runs_and_stays_in_range() {
        let src = solid(20, 20, [200, 40, 40]);
        let dst = solid(40, 40, [30, 180, 90]);
        let mask = centre_mask(20, 20, 6);
        for m in [CloneMethod::Normal, CloneMethod::Mixed, CloneMethod::MonochromeTransfer] {
            let out = seamless_clone(&src, &dst, &mask, (20, 20), m).unwrap();
            for y in 0..40 {
                for x in 0..40 {
                    for c in 0..3 {
                        let v = out.at_f64(y, x, c).unwrap();
                        assert!((0.0..=255.0).contains(&v), "{m:?} produced {v}");
                    }
                }
            }
        }
        assert_eq!(CloneMethod::Normal as i32, 1);
        assert_eq!(CloneMethod::Mixed as i32, 2);
        assert_eq!(CloneMethod::MonochromeTransfer as i32, 3);
    }

    #[test]
    fn color_change_shifts_the_masked_region_only() {
        let (h, w) = (32i32, 32i32);
        let mut v = Vec::new();
        for _y in 0..h {
            for x in 0..w {
                let p = if (x / 3) % 2 == 0 { 60u8 } else { 190 };
                v.extend_from_slice(&[p, p, p]);
            }
        }
        let src = Mat::from_vec(h, w, 3, &v).unwrap();
        let mask = centre_mask(h, w, 8);
        let out = color_change(&src, &mask, 1.5, 0.5, 0.5).unwrap();
        assert_eq!(out.size(), src.size());
        // Outside the mask nothing moved.
        assert_eq!(out.at_f64(1, 1, 0).unwrap(), src.at_f64(1, 1, 0).unwrap());
        // Inside, the result stays a valid image.
        for c in 0..3 {
            let vv = out.at_f64(16, 16, c).unwrap();
            assert!((0.0..=255.0).contains(&vv));
        }
    }

    #[test]
    fn texture_flattening_reduces_variation_inside_the_mask() {
        let (h, w) = (32i32, 32i32);
        let mut v = Vec::new();
        for y in 0..h {
            for x in 0..w {
                // Fine texture: alternating every pixel, so nothing is an edge
                // at a high threshold and the whole lot should flatten.
                let p = if (x + y) % 2 == 0 { 90u8 } else { 160 };
                v.extend_from_slice(&[p, p, p]);
            }
        }
        let src = Mat::from_vec(h, w, 3, &v).unwrap();
        let mask = centre_mask(h, w, 8);
        let out = texture_flattening(&src, &mask, 200.0, 400.0, 3).unwrap();

        let spread = |m: &Mat| {
            let (mut lo, mut hi) = (f64::MAX, f64::MIN);
            for y in 12..20 {
                for x in 12..20 {
                    let p = m.at_f64(y, x, 0).unwrap();
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
            hi - lo
        };
        assert!(spread(&out) < spread(&src), "the texture should flatten");
        assert!(texture_flattening(&src, &mask, 10.0, 100.0, 2).is_err(), "even kernel");
    }

    #[test]
    fn illumination_change_keeps_the_image_valid() {
        let src = solid(24, 24, [120, 130, 140]);
        let mask = centre_mask(24, 24, 6);
        let out = illumination_change(&src, &mask, 0.2, 0.4).unwrap();
        assert_eq!(out.size(), src.size());
        for y in 0..24 {
            for x in 0..24 {
                for c in 0..3 {
                    let v = out.at_f64(y, x, c).unwrap();
                    assert!((0.0..=255.0).contains(&v) && v.is_finite());
                }
            }
        }
    }

    #[test]
    fn a_mismatched_mask_is_rejected() {
        let src = solid(16, 16, [1, 2, 3]);
        let dst = solid(32, 32, [4, 5, 6]);
        let wrong = Mat::zeros(8, 8, CV_8UC1).unwrap();
        assert!(seamless_clone(&src, &dst, &wrong, (16, 16), CloneMethod::Normal).is_err());
        // And a single-channel destination is not a colour image.
        let gray = Mat::zeros(32, 32, CV_8UC1).unwrap();
        let mask = centre_mask(16, 16, 4);
        assert!(seamless_clone(&src, &gray, &mask, (16, 16), CloneMethod::Normal).is_err());
        let _ = CV_8UC3;
    }
}
