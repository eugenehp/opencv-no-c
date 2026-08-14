//! Corner and feature detectors
//! (`modules/imgproc/src/corner.cpp`, `featureselect.cpp`, `cornersubpix.cpp`,
//! `deriv.cpp`).

use cv_core::types::{CV_32FC1, CV_64FC1};
use cv_core::{cv_assert, Mat, MatLike, MatLikeMut, Point, Result};

use crate::filter::sobel;
use cv_core::BorderType;


/// Box-sum the three structure-tensor planes over a `block_size` window.
///
/// Upstream uses `boxFilter` with `normalize = false`, so the entries are sums
/// rather than means. Getting that wrong scales every eigenvalue by the window
/// area and silently changes the meaning of a Harris threshold.
fn structure_tensor(
    gx: &[f64],
    gy: &[f64],
    h: i32,
    w: i32,
    block: i32,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = (h * w) as usize;
    let (mut a, mut b, mut c) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
    // `boxFilter(..., Size(block, block), Point(-1, -1), normalize = false)`:
    // the anchor is `block / 2`, so the window runs `-anchor ..= block-1-anchor`
    // and holds exactly `block` samples per axis. Using `-r ..= r` gives an odd
    // window whatever `block` is — 3x3 instead of 2x2 at the default — which
    // both sums the wrong pixels and sums too many of them.
    let anchor = block / 2;
    let (lo, hi) = (-anchor, block - 1 - anchor);
    for y in 0..h {
        for x in 0..w {
            // The box filter runs on a `CV_32F` plane, so the running sums
            // are `f32` too — accumulating in `f64` disagrees on a third of
            // pixels once the values reach 1e8.
            let (mut sa, mut sb, mut sc) = (0.0f32, 0.0f32, 0.0f32);
            for wy in lo..=hi {
                for wx in lo..=hi {
                    // BORDER_REFLECT_101, upstream's default for boxFilter.
                    let sy = cv_core::reflect101(y + wy, h);
                    let sx = cv_core::reflect101(x + wx, w);
                    // Upstream's `cov` plane is `CV_32FC3`: the products are
                    // formed and accumulated in `f32`, not `f64`.
                    let i = (sy * w + sx) as usize;
                    let (px, py) = (gx[i] as f32, gy[i] as f32);
                    sa += px * px;
                    sb += px * py;
                    sc += py * py;
                }
            }
            let i = (y * w + x) as usize;
            a[i] = f64::from(sa);
            b[i] = f64::from(sb);
            c[i] = f64::from(sc);
        }
    }
    (a, b, c)
}


/// `cv::cornerEigenValsAndVecs(src, dst, blockSize, ksize)`.
///
/// The result is six channels per pixel: `(l1, l2, x1, y1, x2, y2)`, the two
/// eigenvalues of the structure tensor in descending order followed by their
/// unit eigenvectors — upstream's layout exactly.
pub fn corner_eigen_vals_and_vecs<S: MatLike + ?Sized>(
    src: &S,
    block_size: i32,
    ksize: i32,
) -> Result<Mat> {
    cv_assert!(block_size > 0, "blockSize must be positive, got {block_size}");
    // `cornerHarris` scales the Sobel output before forming the tensor, and
    // the scale depends on both the aperture and the block size. Omitting it
    // leaves the response out by `scale^4` — a factor of ~1e12 at the defaults,
    // which is not a tolerance question but a different number entirely.
    cv_assert!(src.channels() == 1, "corner detectors need a single-channel image");
    let (h, w) = (src.rows(), src.cols());
    let mut scale =
        f64::from(1i32 << ((if ksize > 0 { ksize } else { 3 }) - 1)) * f64::from(block_size);
    if ksize < 0 {
        scale *= 2.0;
    }
    if src.depth() == cv_core::types::Depth::U8 {
        scale *= 255.0;
    }
    scale = 1.0 / scale;
    // The scale goes *into* `Sobel`, so it is applied inside the filter at
    // `CV_32F` rather than to an `f64` result afterwards.
    let dx = sobel(src, Some(cv_core::types::Depth::F32), 1, 0, ksize, scale, 0.0, BorderType::Reflect101)?;
    let dy = sobel(src, Some(cv_core::types::Depth::F32), 0, 1, ksize, scale, 0.0, BorderType::Reflect101)?;
    let mut gx = vec![0.0f64; (h * w) as usize];
    let mut gy = vec![0.0f64; (h * w) as usize];
    for y in 0..h {
        for x in 0..w {
            gx[(y * w + x) as usize] = dx.at_f64(y, x, 0)?;
            gy[(y * w + x) as usize] = dy.at_f64(y, x, 0)?;
        }
    }
    let (a, b, c) = structure_tensor(&gx, &gy, h, w, block_size);
    let mut out = Mat::zeros(h, w, cv_core::types::MatType::new(cv_core::types::Depth::F32, 6))?;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let (aa, bb, cc) = (a[i], b[i], c[i]);
            // `eigen2x2`, transcribed. The degenerate handling is upstream's
            // and is not a formality: when `|x| + |y|` falls below 1e-4 it
            // retries with the *other* row of `A - lambda I`, and only if that
            // is also degenerate does it rescale. The normaliser carries a
            // `+ DBL_EPSILON` so a zero eigenvector divides to zero rather than
            // NaN.
            let u = (aa + cc) * 0.5;
            let v = ((aa - cc) * (aa - cc) * 0.25 + bb * bb).sqrt();
            let (l1, l2) = (u + v, u - v);
            let ev = |l: f64| -> (f64, f64) {
                let (mut x, mut y) = (bb, l - aa);
                let mut e = x.abs();
                if e + y.abs() < 1e-4 {
                    y = bb;
                    x = l - cc;
                    e = x.abs();
                    if e + y.abs() < 1e-4 {
                        let e2 = 1.0 / (e + y.abs() + f64::from(f32::EPSILON));
                        x *= e2;
                        y *= e2;
                    }
                }
                let d = 1.0 / (x * x + y * y + f64::EPSILON).sqrt();
                (x * d, y * d)
            };
            let (x1, y1) = ev(l1);
            let (x2, y2) = ev(l2);
            let p = out.pixel_mut::<f32>(y, x)?;
            p[0] = l1 as f32;
            p[1] = l2 as f32;
            p[2] = x1 as f32;
            p[3] = y1 as f32;
            p[4] = x2 as f32;
            p[5] = y2 as f32;
        }
    }
    Ok(out)
}

/// `cv::cornerMinEigenVal(src, dst, blockSize, ksize)`.
pub fn corner_min_eigen_val<S: MatLike + ?Sized>(
    src: &S,
    block_size: i32,
    ksize: i32,
) -> Result<Mat> {
    // `cornerHarris` scales the Sobel output before forming the tensor, and
    // the scale depends on both the aperture and the block size. Omitting it
    // leaves the response out by `scale^4` — a factor of ~1e12 at the defaults,
    // which is not a tolerance question but a different number entirely.
    cv_assert!(src.channels() == 1, "corner detectors need a single-channel image");
    let (h, w) = (src.rows(), src.cols());
    let mut scale =
        f64::from(1i32 << ((if ksize > 0 { ksize } else { 3 }) - 1)) * f64::from(block_size);
    if ksize < 0 {
        scale *= 2.0;
    }
    if src.depth() == cv_core::types::Depth::U8 {
        scale *= 255.0;
    }
    scale = 1.0 / scale;
    // The scale goes *into* `Sobel`, so it is applied inside the filter at
    // `CV_32F` rather than to an `f64` result afterwards.
    let dx = sobel(src, Some(cv_core::types::Depth::F32), 1, 0, ksize, scale, 0.0, BorderType::Reflect101)?;
    let dy = sobel(src, Some(cv_core::types::Depth::F32), 0, 1, ksize, scale, 0.0, BorderType::Reflect101)?;
    let mut gx = vec![0.0f64; (h * w) as usize];
    let mut gy = vec![0.0f64; (h * w) as usize];
    for y in 0..h {
        for x in 0..w {
            gx[(y * w + x) as usize] = dx.at_f64(y, x, 0)?;
            gy[(y * w + x) as usize] = dy.at_f64(y, x, 0)?;
        }
    }
    let (a, b, c) = structure_tensor(&gx, &gy, h, w, block_size);
    let mut out = Mat::zeros(h, w, CV_32FC1)?;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let t = (a[i] - c[i]) * 0.5;
            let v = (a[i] + c[i]) * 0.5 - (t * t + b[i] * b[i]).sqrt();
            out.pixel_mut::<f32>(y, x)?[0] = v as f32;
        }
    }
    Ok(out)
}

/// `cv::cornerHarris(src, dst, blockSize, ksize, k)` — `det(M) - k*trace(M)^2`.
pub fn corner_harris<S: MatLike + ?Sized>(
    src: &S,
    block_size: i32,
    ksize: i32,
    k: f64,
) -> Result<Mat> {
    // `cornerHarris` scales the Sobel output before forming the tensor, and
    // the scale depends on both the aperture and the block size. Omitting it
    // leaves the response out by `scale^4` — a factor of ~1e12 at the defaults,
    // which is not a tolerance question but a different number entirely.
    cv_assert!(src.channels() == 1, "corner detectors need a single-channel image");
    let (h, w) = (src.rows(), src.cols());
    let mut scale =
        f64::from(1i32 << ((if ksize > 0 { ksize } else { 3 }) - 1)) * f64::from(block_size);
    if ksize < 0 {
        scale *= 2.0;
    }
    if src.depth() == cv_core::types::Depth::U8 {
        scale *= 255.0;
    }
    scale = 1.0 / scale;
    // The scale goes *into* `Sobel`, so it is applied inside the filter at
    // `CV_32F` rather than to an `f64` result afterwards.
    let dx = sobel(src, Some(cv_core::types::Depth::F32), 1, 0, ksize, scale, 0.0, BorderType::Reflect101)?;
    let dy = sobel(src, Some(cv_core::types::Depth::F32), 0, 1, ksize, scale, 0.0, BorderType::Reflect101)?;
    let mut gx = vec![0.0f64; (h * w) as usize];
    let mut gy = vec![0.0f64; (h * w) as usize];
    for y in 0..h {
        for x in 0..w {
            gx[(y * w + x) as usize] = dx.at_f64(y, x, 0)?;
            gy[(y * w + x) as usize] = dy.at_f64(y, x, 0)?;
        }
    }
    let (a, b, c) = structure_tensor(&gx, &gy, h, w, block_size);
    let mut out = Mat::zeros(h, w, CV_32FC1)?;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            // `(float)(a*c - b*b - k*(a + c)*(a + c))` with `a, b, c` in `f32`
            // and `k` a `double`, so only the `k` term widens.
            let (fa, fb, fc) = (a[i] as f32, b[i] as f32, c[i] as f32);
            let det = fa * fc - fb * fb;
            let tr = fa + fc;
            out.pixel_mut::<f32>(y, x)?[0] =
                (f64::from(det) - k * f64::from(tr) * f64::from(tr)) as f32;
        }
    }
    Ok(out)
}

/// `cv::preCornerDetect(src, dst, ksize)`.
///
/// `Dx^2*Dyy + Dy^2*Dxx - 2*Dx*Dy*Dxy`, upstream's feature map.
pub fn pre_corner_detect<S: MatLike + ?Sized>(src: &S, ksize: i32) -> Result<Mat> {
    use cv_core::types::Depth;
    cv_assert!(src.channels() == 1, "preCornerDetect needs a single-channel image");
    let (h, w) = (src.rows(), src.cols());
    let dx = sobel(src, Some(Depth::F64), 1, 0, ksize, 1.0, 0.0, BorderType::Reflect101)?;
    let dy = sobel(src, Some(Depth::F64), 0, 1, ksize, 1.0, 0.0, BorderType::Reflect101)?;
    let dxx = sobel(src, Some(Depth::F64), 2, 0, ksize, 1.0, 0.0, BorderType::Reflect101)?;
    let dyy = sobel(src, Some(Depth::F64), 0, 2, ksize, 1.0, 0.0, BorderType::Reflect101)?;
    let dxy = sobel(src, Some(Depth::F64), 1, 1, ksize, 1.0, 0.0, BorderType::Reflect101)?;
    // The expression is cubic in the derivatives, so upstream's aperture
    // normalisation enters as `factor^3`. Without it the output is out by ~4e8
    // at ksize 3 on `CV_32F` input.
    let mut factor = f64::from(1i32 << (ksize - 1));
    if src.depth() == Depth::U8 {
        factor *= 255.0;
    }
    let factor = 1.0 / (factor * factor * factor);

    let mut out = Mat::zeros(h, w, CV_32FC1)?;
    for y in 0..h {
        for x in 0..w {
            // Upstream's derivative planes are `CV_32F` and it combines them
            // in `f32`; keeping `f64` here disagrees on 3% of pixels.
            let (ax, ay) = (dx.at_f64(y, x, 0)? as f32, dy.at_f64(y, x, 0)? as f32);
            let v = ax * ax * dyy.at_f64(y, x, 0)? as f32
                + ay * ay * dxx.at_f64(y, x, 0)? as f32
                - 2.0 * ax * ay * dxy.at_f64(y, x, 0)? as f32;
            out.pixel_mut::<f32>(y, x)?[0] = v * factor as f32;
        }
    }
    Ok(out)
}

/// `cv::goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance, ...)`.
///
/// Shi-Tomasi by default; `use_harris` switches the response to [`corner_harris`].
/// Returns corners sorted by descending response, thinned so no two are closer
/// than `min_distance`.
pub fn good_features_to_track<S: MatLike + ?Sized>(
    image: &S,
    max_corners: i32,
    quality_level: f64,
    min_distance: f64,
    block_size: i32,
    use_harris: bool,
    k: f64,
) -> Result<Vec<Point>> {
    cv_assert!(quality_level > 0.0, "qualityLevel must be positive, got {quality_level}");
    cv_assert!(min_distance >= 0.0, "minDistance must be non-negative");
    let resp = if use_harris {
        corner_harris(image, block_size, 3, k)?
    } else {
        corner_min_eigen_val(image, block_size, 3)?
    };
    let (h, w) = (resp.rows(), resp.cols());

    let mut max_val = f64::NEG_INFINITY;
    for y in 0..h {
        for x in 0..w {
            max_val = max_val.max(resp.at_f64(y, x, 0)?);
        }
    }
    if max_val <= 0.0 {
        return Ok(Vec::new());
    }
    let threshold = max_val * quality_level;

    // Keep 3x3 local maxima above the threshold.
    let mut cand: Vec<(f64, Point)> = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let v = resp.at_f64(y, x, 0)?;
            if v < threshold {
                continue;
            }
            let mut peak = true;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if (dx != 0 || dy != 0) && resp.at_f64(y + dy, x + dx, 0)? > v {
                        peak = false;
                    }
                }
            }
            if peak {
                cand.push((v, Point::new(x, y)));
            }
        }
    }
    cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));

    let mut out: Vec<Point> = Vec::new();
    let min_d2 = min_distance * min_distance;
    for (_, p) in cand {
        if max_corners > 0 && out.len() as i32 >= max_corners {
            break;
        }
        if out.iter().any(|q| {
            let (dx, dy) = ((q.x - p.x) as f64, (q.y - p.y) as f64);
            dx * dx + dy * dy < min_d2
        }) {
            continue;
        }
        out.push(p);
    }
    Ok(out)
}

/// `cv::cornerSubPix(image, corners, winSize, zeroZone, criteria)`.
///
/// Refines each corner to sub-pixel accuracy by solving for the point where the
/// image gradient is orthogonal to the offset from the centre.
pub fn corner_sub_pix<S: MatLike + ?Sized>(
    image: &S,
    corners: &[Point],
    win: i32,
    max_iter: i32,
    epsilon: f64,
) -> Result<Vec<(f64, f64)>> {
    cv_assert!(image.channels() == 1, "cornerSubPix needs a single-channel image");
    cv_assert!(win > 0, "window radius must be positive, got {win}");
    let (h, w) = (image.rows(), image.cols());
    let (win_w, win_h) = (win * 2 + 1, win * 2 + 1);

    // The Gaussian mask upstream weights the window by: `exp(-x^2 - y^2)` with
    // x and y normalised by the half-window, so a pixel at the edge of the
    // window counts `exp(-1)` of one at the centre. Without it the fit is a
    // plain least squares over a square patch, which pulls the corner toward
    // whatever texture happens to sit in the window's corners.
    let mask: Vec<f64> = (0..win_h)
        .flat_map(|i2| {
            let y = f64::from(i2 - win) / f64::from(win);
            let vy = (-y * y).exp();
            (0..win_w).map(move |j| {
                let x = f64::from(j - win) / f64::from(win);
                vy * (-x * x).exp()
            })
        })
        .collect();

    // `getRectSubPix`: a bilinear patch centred on the *current* sub-pixel
    // estimate. Rounding the centre to an integer instead throws away the
    // refinement between iterations, which is the whole point.
    let sub_pix = |cx: f64, cy: f64, pw: i32, ph: i32| -> Result<Vec<f64>> {
        let (x0, y0) = (cx - f64::from(pw - 1) * 0.5, cy - f64::from(ph - 1) * 0.5);
        let (ix, iy) = (x0.floor() as i32, y0.floor() as i32);
        let (ax, ay) = (x0 - f64::from(ix), y0 - f64::from(iy));
        let at = |yy: i32, xx: i32| -> Result<f64> {
            image.at_f64(yy.clamp(0, h - 1), xx.clamp(0, w - 1), 0)
        };
        let mut buf = vec![0.0f64; (pw * ph) as usize];
        for r in 0..ph {
            for c in 0..pw {
                let (sy, sx) = (iy + r, ix + c);
                buf[(r * pw + c) as usize] = at(sy, sx)? * (1.0 - ax) * (1.0 - ay)
                    + at(sy, sx + 1)? * ax * (1.0 - ay)
                    + at(sy + 1, sx)? * (1.0 - ax) * ay
                    + at(sy + 1, sx + 1)? * ax * ay;
            }
        }
        Ok(buf)
    };

    // `eps` is compared against the *squared* step upstream.
    let eps2 = epsilon.max(0.0) * epsilon.max(0.0);
    let (pw, ph) = (win_w + 2, win_h + 2);
    let mut out = Vec::with_capacity(corners.len());
    for c in corners {
        let (mut cx, mut cy) = (f64::from(c.x), f64::from(c.y));
        let mut iter = 0;
        loop {
            let patch = sub_pix(cx, cy, pw, ph)?;
            let (mut a, mut b, mut cc) = (0.0f64, 0.0f64, 0.0f64);
            let (mut bb1, mut bb2) = (0.0f64, 0.0f64);
            for i2 in 0..win_h {
                let py = f64::from(i2 - win);
                for j in 0..win_w {
                    let m = mask[(i2 * win_w + j) as usize];
                    // Indices into the padded patch: the window starts at (1,1).
                    let at = |r: i32, cq: i32| patch[((r + 1) * pw + cq + 1) as usize];
                    let tgx = at(i2, j + 1) - at(i2, j - 1);
                    let tgy = at(i2 + 1, j) - at(i2 - 1, j);
                    let (gxx, gxy, gyy) = (tgx * tgx * m, tgx * tgy * m, tgy * tgy * m);
                    let px = f64::from(j - win);
                    a += gxx;
                    b += gxy;
                    cc += gyy;
                    bb1 += gxx * px + gxy * py;
                    bb2 += gxy * px + gyy * py;
                }
            }
            let det = a * cc - b * b;
            if det.abs() <= f64::EPSILON * f64::EPSILON {
                break;
            }
            let scale = 1.0 / det;
            let nx = cx + cc * scale * bb1 - b * scale * bb2;
            let ny = cy - b * scale * bb1 + a * scale * bb2;
            let err = (nx - cx) * (nx - cx) + (ny - cy) * (ny - cy);
            cx = nx;
            cy = ny;
            if cx < 0.0 || cx >= f64::from(w) || cy < 0.0 || cy >= f64::from(h) {
                break;
            }
            iter += 1;
            if iter >= max_iter.max(1) || err <= eps2 {
                break;
            }
        }
        // Poor convergence: if the refinement wandered further than the window
        // itself, upstream keeps the *seed*. Without this a corner that the fit
        // could not resolve is reported as having moved several pixels, which
        // is worse than not refining it — the caller has no way to tell.
        if (cx - f64::from(c.x)).abs() > f64::from(win)
            || (cy - f64::from(c.y)).abs() > f64::from(win)
        {
            cx = f64::from(c.x);
            cy = f64::from(c.y);
        }
        out.push((cx, cy));
    }
    Ok(out)
}

/// `cv::spatialGradient(src, dx, dy, ksize, borderType)`.
///
/// The two first-order Sobel derivatives in one pass, as `CV_16S`.
pub fn spatial_gradient<S: MatLike + ?Sized>(src: &S) -> Result<(Mat, Mat)> {
    use cv_core::types::Depth;
    Ok((sobel(src, Some(Depth::S16), 1, 0, 3, 1.0, 0.0, BorderType::Reflect101)?, sobel(src, Some(Depth::S16), 0, 1, 3, 1.0, 0.0, BorderType::Reflect101)?))
}

/// `cv::getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype)`.
pub fn get_gabor_kernel(
    width: i32,
    height: i32,
    sigma: f64,
    theta: f64,
    lambd: f64,
    gamma: f64,
    psi: f64,
) -> Result<Mat> {
    cv_assert!(width > 0 && height > 0, "kernel size must be positive");
    cv_assert!(lambd != 0.0, "lambda must be non-zero");
    let sigma_x = sigma;
    let sigma_y = sigma / gamma;
    let (c, s) = (theta.cos(), theta.sin());
    let (xmax, ymax) = (width / 2, height / 2);
    let scale = 1.0;
    let ex = -0.5 / (sigma_x * sigma_x);
    let ey = -0.5 / (sigma_y * sigma_y);
    let cscale = core::f64::consts::PI * 2.0 / lambd;

    let mut k = Mat::zeros(height, width, CV_64FC1)?;
    for y in -ymax..=ymax {
        for x in -xmax..=xmax {
            let (xr, yr) = (x as f64 * c + y as f64 * s, -(x as f64) * s + y as f64 * c);
            let v = scale * (ex * xr * xr + ey * yr * yr).exp() * (cscale * xr + psi).cos();
            let (ry, rx) = (ymax + y, xmax + x);
            if ry < height && rx < width {
                k.pixel_mut::<f64>(ry, rx)?[0] = v;
            }
        }
    }
    Ok(k)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bright square on a dark field: four strong corners.
    fn square() -> Mat {
        let mut v = vec![0u8; 40 * 40];
        for y in 10..30 {
            for x in 10..30 {
                v[y * 40 + x] = 255;
            }
        }
        Mat::from_vec(40, 40, 1, &v).unwrap()
    }

    #[test]
    fn harris_peaks_at_the_square_corners() {
        let img = square();
        let r = corner_harris(&img, 2, 3, 0.04).unwrap();
        // The response near a corner beats the response along an edge midpoint.
        let corner = r.at_f64(10, 10, 0).unwrap();
        let edge = r.at_f64(10, 20, 0).unwrap();
        let flat = r.at_f64(2, 2, 0).unwrap();
        assert!(corner > edge, "corner {corner} should beat edge {edge}");
        assert!(edge.abs() > flat.abs(), "edge {edge} should beat flat {flat}");
    }

    #[test]
    fn min_eigen_val_is_never_above_the_larger_eigenvalue() {
        let img = square();
        let mev = corner_min_eigen_val(&img, 3, 3).unwrap();
        let ev = corner_eigen_vals_and_vecs(&img, 3, 3).unwrap();
        for y in 0..40 {
            for x in 0..40 {
                let l2 = ev.at_f64(y, x, 1).unwrap();
                let m = mev.at_f64(y, x, 0).unwrap();
                assert!((l2 - m).abs() < 1e-3, "({x},{y}): {l2} vs {m}");
                assert!(ev.at_f64(y, x, 0).unwrap() >= l2 - 1e-6, "l1 must be >= l2");
            }
        }
    }

    #[test]
    fn eigenvectors_are_unit_where_there_is_structure() {
        // Upstream does **not** normalise unconditionally: `eigen2x2` divides
        // by `sqrt(x*x + y*y + DBL_EPSILON)`, so a flat region — where both
        // eigenvalues are zero and the eigenvector is (0, 0) — stays zero
        // rather than being forced to (1, 0). Verified against cv2, which
        // returns all six channels zero at a flat corner and leaves 1280 of
        // 1600 pixels non-unit on this very image.
        let ev = corner_eigen_vals_and_vecs(&square(), 3, 3).unwrap();
        let mut unit = 0;
        for y in 0..40 {
            for x in 0..40 {
                let l1 = ev.at_f64(y, x, 0).unwrap();
                for base in [2, 4] {
                    let a = ev.at_f64(y, x, base).unwrap();
                    let b = ev.at_f64(y, x, base + 1).unwrap();
                    let n = a.hypot(b);
                    if l1 == 0.0 {
                        assert_eq!(n, 0.0, "({x},{y}) flat but the vector is {a},{b}");
                    } else {
                        assert!(n <= 1.0 + 1e-6, "({x},{y}) over-long: {n}");
                        if (n - 1.0).abs() < 1e-3 {
                            unit += 1;
                        }
                    }
                }
            }
        }
        assert!(unit > 100, "only {unit} unit eigenvectors — the structure is gone");
    }

    #[test]
    fn good_features_finds_four_corners_and_honours_min_distance() {
        let img = square();
        let pts = good_features_to_track(&img, 10, 0.01, 5.0, 3, false, 0.04).unwrap();
        assert!(pts.len() >= 4, "expected at least the four corners, got {pts:?}");
        for i in 0..pts.len() {
            for j in i + 1..pts.len() {
                let (dx, dy) = ((pts[i].x - pts[j].x) as f64, (pts[i].y - pts[j].y) as f64);
                assert!(dx.hypot(dy) >= 5.0, "{:?} and {:?} are too close", pts[i], pts[j]);
            }
        }
    }

    #[test]
    fn good_features_respects_max_corners() {
        let pts = good_features_to_track(&square(), 2, 0.01, 1.0, 3, false, 0.04).unwrap();
        assert_eq!(pts.len(), 2);
    }

    #[test]
    fn good_features_on_a_flat_image_finds_nothing() {
        let flat = Mat::from_vec(20, 20, 1, &[128u8; 400]).unwrap();
        assert!(good_features_to_track(&flat, 10, 0.01, 1.0, 3, false, 0.04).unwrap().is_empty());
    }

    #[test]
    fn corner_sub_pix_stays_near_its_input() {
        let img = square();
        let seeds = vec![Point::new(10, 10), Point::new(29, 29)];
        let r = corner_sub_pix(&img, &seeds, 3, 20, 0.01).unwrap();
        assert_eq!(r.len(), 2);
        for (p, s) in r.iter().zip(seeds.iter()) {
            let d = (p.0 - s.x as f64).hypot(p.1 - s.y as f64);
            assert!(d < 3.0, "refinement ran away: {p:?} from {s:?}");
        }
    }

    #[test]
    fn gabor_kernel_has_the_requested_size_and_is_not_constant() {
        let k = get_gabor_kernel(9, 9, 2.0, 0.0, 4.0, 0.5, 0.0).unwrap();
        assert_eq!((k.rows(), k.cols()), (9, 9));
        let vals = k.to_vec::<f64>().unwrap();
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max > 0.0 && min < 0.0, "a Gabor kernel oscillates: {min}..{max}");
        // theta = 0 makes it separable along x: rows are identical up to the
        // Gaussian envelope, so column 4 (the centre) is the largest.
        assert!((vals[4 * 9 + 4] - max).abs() < 1e-12, "peak at the centre");
    }

    #[test]
    fn spatial_gradient_matches_sobel() {
        let img = square();
        let (gx, gy) = spatial_gradient(&img).unwrap();
        let sx = sobel(&img, Some(cv_core::types::Depth::S16), 1, 0, 3, 1.0, 0.0, BorderType::Reflect101).unwrap();
        assert_eq!(gx.to_vec::<i16>().unwrap(), sx.to_vec::<i16>().unwrap());
        assert_eq!(gy.rows(), img.rows());
    }
}
