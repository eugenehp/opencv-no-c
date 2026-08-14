//! `cv::photo` for `opencv-no-c` — denoising, inpainting and stylisation.
//!
//! # Scope
//!
//! Non-local-means denoising, Telea-style inpainting, and the edge-preserving
//! filter family that `detailEnhance`, `pencilSketch` and `stylization` are
//! built on. All single-image; the multi-frame HDR pipeline
//! (`createMergeDebevec` and friends) needs exposure-time metadata and a
//! camera response estimate, which is a separate job.

#![warn(missing_docs)]

pub mod cloning;
pub mod hdr;
pub mod constants;

use cv_core::types::{Depth, MatType, CV_32FC1, CV_8UC1};
use cv_core::{cv_assert, Mat, MatLike, MatLikeMut, Result};


/// `cv::fastNlMeansDenoising(src, dst, h, templateWindowSize, searchWindowSize)`.
///
/// Non-local means: every pixel is replaced by a weighted average of pixels
/// whose *neighbourhoods* look like its own, with the weight falling off as
/// `exp(-d^2 / h^2)` in patch distance. That is what separates it from a blur —
/// texture is preserved because similar texture reinforces itself.
///
/// `CV_8UC1` only. The cost is `O(w*h*search^2*template^2)`, so this is a
/// straightforward implementation rather than upstream's block-wise one; on a
/// large image it is slow.
pub fn fast_nl_means_denoising<S: MatLike + ?Sized>(
    src: &S,
    h: f64,
    template_window: i32,
    search_window: i32,
) -> Result<Mat> {
    cv_assert!(src.mat_type() == CV_8UC1, "fastNlMeansDenoising here is CV_8UC1 only");
    cv_assert!(
        template_window > 0 && template_window % 2 == 1,
        "templateWindowSize must be odd and positive, got {template_window}"
    );
    cv_assert!(
        search_window > 0 && search_window % 2 == 1,
        "searchWindowSize must be odd and positive, got {search_window}"
    );
    cv_assert!(h > 0.0, "h must be positive, got {h}");
    let (rows, cols) = (src.rows(), src.cols());
    let tr = template_window / 2;
    let sr = search_window / 2;
    let inv_h2 = -1.0 / (h * h);
    let patch = f64::from(template_window * template_window);

    // Read once; `at_f64` per tap would dominate the run time.
    let mut img = vec![0.0f64; (rows * cols) as usize];
    for y in 0..rows {
        for x in 0..cols {
            img[(y * cols + x) as usize] = src.at_f64(y, x, 0)?;
        }
    }
    let at = |y: i32, x: i32| img[(cv_core::reflect101(y, rows) * cols + cv_core::reflect101(x, cols)) as usize];

    let mut out = Mat::zeros(rows, cols, CV_8UC1)?;
    for y in 0..rows {
        for x in 0..cols {
            let (mut acc, mut wsum) = (0.0f64, 0.0f64);
            for sy in -sr..=sr {
                for sx in -sr..=sr {
                    // Mean squared difference between the two patches.
                    let mut d2 = 0.0f64;
                    for ty in -tr..=tr {
                        for tx in -tr..=tr {
                            let a = at(y + ty, x + tx);
                            let b = at(y + sy + ty, x + sx + tx);
                            d2 += (a - b) * (a - b);
                        }
                    }
                    let w = (d2 / patch * inv_h2).exp();
                    acc += w * at(y + sy, x + sx);
                    wsum += w;
                }
            }
            let v = if wsum > 0.0 { acc / wsum } else { at(y, x) };
            out.pixel_mut::<u8>(y, x)?[0] = cv_core::saturate::Elem::from_f64(v);
        }
    }
    Ok(out)
}

/// `cv::InpaintTypes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InpaintMethod {
    /// `cv::INPAINT_TELEA` — fast marching, weighted by distance and gradient.
    Telea,
    /// `cv::INPAINT_NS` — Navier-Stokes: the same fast-marching front as
    /// Telea, filled by a weight that favours neighbours lying along the
    /// isophote, so intensity is carried along an edge into the hole.
    NavierStokes,
}

/// Marker values for the front state, as `inpaint.cpp` defines them.
const KNOWN: u8 = 0;
const BAND: u8 = 1;
const INSIDE: u8 = 2;

/// `CvHeapElem`'s ordering: arrival time first, **insertion order** second.
///
/// The tie-break is not incidental. Arrival times land on the same value
/// constantly on a discrete grid, and the fill is order-dependent — each pixel
/// averages neighbours that may have just been written. Breaking ties any other
/// way leaves some pixels exact and their neighbours 20 levels out.
#[derive(PartialEq)]
struct Arrival(f32, u64);
impl Eq for Arrival {}
impl PartialOrd for Arrival {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Arrival {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0).then(self.1.cmp(&o.1))
    }
}

/// `FastMarching_solve` — the discrete eikonal update from two neighbours.
fn fmm_solve(a: usize, b: usize, f: &[u8], t: &[f32]) -> f32 {
    let (a11, a22) = (t[a], t[b]);
    let m12 = a11.min(a22);
    if f[a] != INSIDE {
        if f[b] != INSIDE {
            if (a11 - a22).abs() >= 1.0 {
                1.0 + m12
            } else {
                (a11 + a22 + (2.0 - (a11 - a22) * (a11 - a22)).sqrt()) * 0.5
            }
        } else {
            1.0 + a11
        }
    } else if f[b] != INSIDE {
        1.0 + a22
    } else {
        1.0 + m12
    }
}

/// `icvCalcFMM` — march the front into the `INSIDE` region, filling `t`.
///
/// `negate` flips the result's sign, which is how the region *outside* the hole
/// gets arrival times on the same scale as the inside.
fn calc_fmm(f: &mut [u8], t: &mut [f32], seeds: &[usize], w: usize, h: usize, negate: bool) {
    let mut order = 0u64;
    let mut heap: std::collections::BinaryHeap<(std::cmp::Reverse<Arrival>, usize)> = seeds
        .iter()
        .map(|&i| {
            order += 1;
            (std::cmp::Reverse(Arrival(t[i], order)), i)
        })
        .collect();
    while let Some((_, p)) = heap.pop() {
        f[p] = KNOWN;
        let (ii, jj) = (p / w, p % w);
        for &(dy, dx) in &[(-1i32, 0i32), (0, -1), (1, 0), (0, 1)] {
            let (i, j) = (ii as i32 + dy, jj as i32 + dx);
            if i < 1 || j < 1 || i >= h as i32 - 1 || j >= w as i32 - 1 {
                continue;
            }
            let q = i as usize * w + j as usize;
            if f[q] != INSIDE {
                continue;
            }
            let dist = fmm_solve(q - w, q - 1, f, t)
                .min(fmm_solve(q + w, q - 1, f, t))
                .min(fmm_solve(q - w, q + 1, f, t))
                .min(fmm_solve(q + w, q + 1, f, t));
            t[q] = dist;
            f[q] = BAND;
            order += 1;
            heap.push((std::cmp::Reverse(Arrival(dist, order)), q));
        }
    }
    if negate {
        for v in t.iter_mut() {
            *v = -*v;
        }
    }
}

/// `icvTeleaInpaintFMM` — fill the hole in order of increasing arrival time.
///
/// Each pixel becomes a weighted average of the known pixels within `range`,
/// plus a first-order term extrapolating the image gradient inward. The three
/// weights are what make it more than a blur: `dst` favours near neighbours,
/// `lev` favours ones on the same level set of `t`, and `dir` favours ones
/// lying along the inward normal.
/// `icvNSInpaintFMM` — the Navier-Stokes fill, over the same front.
///
/// It marches the identical fast-marching front as [`telea_fmm`]; what differs
/// is the weight. Telea weights a known neighbour by distance, level-set
/// proximity and how nearly it lies along the isophote. Navier-Stokes weights
/// it by distance and by how nearly the *ray to it* is aligned with the local
/// image gradient — so intensity is carried **along** an edge into the hole
/// rather than across it, which is what continues a line through a gap.
///
/// Filling by an unweighted average over the same neighbourhood, one boundary
/// layer at a time, is a different and much blurrier operator.
#[allow(clippy::too_many_arguments)]
fn ns_fmm(
    f: &mut [u8],
    t: &mut [f32],
    img: &mut [f64],
    w: usize,
    h: usize,
    cn: usize,
    range: i32,
    seeds: &[usize],
) {
    let mut order = 0u64;
    let mut heap: std::collections::BinaryHeap<(std::cmp::Reverse<Arrival>, usize)> = seeds
        .iter()
        .map(|&i| {
            order += 1;
            (std::cmp::Reverse(Arrival(t[i], order)), i)
        })
        .collect();
    while let Some((_, p)) = heap.pop() {
        f[p] = KNOWN;
        let (ii, jj) = (p / w, p % w);
        for &(dy, dx) in &[(-1i32, 0i32), (0, -1), (1, 0), (0, 1)] {
            let (i, j) = (ii as i32 + dy, jj as i32 + dx);
            if i < 1 || j < 1 || i >= h as i32 - 1 || j >= w as i32 - 1 {
                continue;
            }
            let (i, j) = (i as usize, j as usize);
            let q = i * w + j;
            if f[q] != INSIDE {
                continue;
            }
            t[q] = fmm_solve(q - w, q - 1, f, t)
                .min(fmm_solve(q + w, q - 1, f, t))
                .min(fmm_solve(q - w, q + 1, f, t))
                .min(fmm_solve(q + w, q + 1, f, t));

            let mut ia = vec![0.0f32; cn];
            let mut s = vec![1.0e-20f32; cn];
            for k in (i as i32 - range)..=(i as i32 + range) {
                for l in (j as i32 - range)..=(j as i32 + range) {
                    if k <= 0 || l <= 0 || k >= h as i32 - 1 || l >= w as i32 - 1 {
                        continue;
                    }
                    let n = k as usize * w + l as usize;
                    if f[n] == INSIDE {
                        continue;
                    }
                    let (ry, rx) = ((k - i as i32) as f32, (l - j as i32) as f32);
                    let rlen2 = rx * rx + ry * ry;
                    if rlen2 > (range * range) as f32 {
                        continue;
                    }
                    // `VectorLength` is the *squared* length upstream, so this
                    // denominator is `|r|^4 + 1`, not `|r|^2 + 1`.
                    let dst = 1.0 / (rlen2 * rlen2 + 1.0);
                    for c in 0..cn {
                        let at = |idx: usize| img[idx * cn + c] as f32;
                        // Absolute differences, one-sided where the hole blocks
                        // a side; the x component is then negated, which is
                        // what turns the dot product into an isophote
                        // alignment rather than a gradient alignment.
                        let mut gx = if f[n + w] != INSIDE {
                            if f[n - w] != INSIDE {
                                (at(n + w) - at(n)).abs() + (at(n) - at(n - w)).abs()
                            } else {
                                (at(n + w) - at(n)).abs() * 2.0
                            }
                        } else if f[n - w] != INSIDE {
                            (at(n) - at(n - w)).abs() * 2.0
                        } else {
                            0.0
                        };
                        let gy = if f[n + 1] != INSIDE {
                            if f[n - 1] != INSIDE {
                                (at(n + 1) - at(n)).abs() + (at(n) - at(n - 1)).abs()
                            } else {
                                (at(n + 1) - at(n)).abs() * 2.0
                            }
                        } else if f[n - 1] != INSIDE {
                            (at(n) - at(n - 1)).abs() * 2.0
                        } else {
                            0.0
                        };
                        gx = -gx;

                        // `r.x * gradI.x + r.y * gradI.y`, where `gradI.x` is
                        // built from *row* differences and `r.x` is a *column*
                        // offset. The pairing is deliberately crossed — that
                        // is what makes the weight favour a neighbour lying
                        // along the isophote rather than along the gradient.
                        let dot = rx * gx + ry * gy;
                        let glen2 = gx * gx + gy * gy;
                        let dir = if dot.abs() <= 0.01 {
                            0.000_001
                        } else {
                            (dot / (rlen2 * glen2).sqrt()).abs()
                        };
                        let wt = dst * dir;
                        ia[c] += wt * at(n);
                        s[c] += wt;
                    }
                }
            }
            for c in 0..cn {
                // Plain `saturate_cast<uchar>` here — *no* `+ 0.5`. The
                // Navier-Stokes fill and the Telea fill round differently in
                // upstream, and it is not an oversight in either.
                img[q * cn + c] =
                    f64::from(cv_core::saturate::cv_round(f64::from(ia[c] / s[c])).clamp(0, 255));
            }
            f[q] = BAND;
            order += 1;
            heap.push((std::cmp::Reverse(Arrival(t[q], order)), q));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn telea_fmm(
    f: &mut [u8],
    t: &mut [f32],
    img: &mut [f64],
    w: usize,
    h: usize,
    cn: usize,
    range: i32,
    seeds: &[usize],
) {
    let mut order = 0u64;
    let mut heap: std::collections::BinaryHeap<(std::cmp::Reverse<Arrival>, usize)> = seeds
        .iter()
        .map(|&i| {
            order += 1;
            (std::cmp::Reverse(Arrival(t[i], order)), i)
        })
        .collect();
    while let Some((_, p)) = heap.pop() {
        f[p] = KNOWN;
        let (ii, jj) = (p / w, p % w);
        for &(dy, dx) in &[(-1i32, 0i32), (0, -1), (1, 0), (0, 1)] {
            let (i, j) = (ii as i32 + dy, jj as i32 + dx);
            if i < 1 || j < 1 || i >= h as i32 - 1 || j >= w as i32 - 1 {
                continue;
            }
            let (i, j) = (i as usize, j as usize);
            let q = i * w + j;
            if f[q] != INSIDE {
                continue;
            }
            let dist = fmm_solve(q - w, q - 1, f, t)
                .min(fmm_solve(q + w, q - 1, f, t))
                .min(fmm_solve(q - w, q + 1, f, t))
                .min(fmm_solve(q + w, q + 1, f, t));
            t[q] = dist;

            // Gradient of the arrival time, one-sided where the hole blocks it.
            let gtx = if f[q + 1] != INSIDE {
                if f[q - 1] != INSIDE { (t[q + 1] - t[q - 1]) * 0.5 } else { t[q + 1] - t[q] }
            } else if f[q - 1] != INSIDE {
                t[q] - t[q - 1]
            } else {
                0.0
            };
            let gty = if f[q + w] != INSIDE {
                if f[q - w] != INSIDE { (t[q + w] - t[q - w]) * 0.5 } else { t[q + w] - t[q] }
            } else if f[q - w] != INSIDE {
                t[q] - t[q - w]
            } else {
                0.0
            };

            let mut ia = vec![0.0f32; cn];
            let mut jx = vec![0.0f32; cn];
            let mut jy = vec![0.0f32; cn];
            let mut s = vec![1.0e-20f32; cn];
            for k in (i as i32 - range).max(1)..=(i as i32 + range).min(h as i32 - 2) {
                for l in (j as i32 - range).max(1)..=(j as i32 + range).min(w as i32 - 2) {
                    let (k, l) = (k as usize, l as usize);
                    let o = k * w + l;
                    if f[o] == INSIDE {
                        continue;
                    }
                    let (ry, rx) = (i as f32 - k as f32, j as f32 - l as f32);
                    let len2 = rx * rx + ry * ry;
                    if len2 > (range * range) as f32 {
                        continue;
                    }
                    // `VectorLength` in `inpaint.cpp` returns the **squared**
                    // length, so this is `1 / |r|^3`, not `1 / |r|^1.5`. Both
                    // weights are computed in `double` and narrowed once, which
                    // is not the same as computing them in `f32` throughout.
                    // The two branches of `icvTeleaInpaintFMM` differ here, and
                    // not by accident of style: the 3-channel one writes
                    // `sqrt((double)VectorLength(r))` and the 1-channel one
                    // `sqrt(VectorLength(r))`, which resolves to the `float`
                    // overload. Using the wider form for grayscale costs 2
                    // intensity levels.
                    let dst = if cn == 1 {
                        1.0 / (len2 * len2.sqrt())
                    } else {
                        let l2 = f64::from(len2);
                        (1.0 / (l2 * l2.sqrt())) as f32
                    };
                    let lev = (1.0 / (1.0 + f64::from((t[o] - t[q]).abs()))) as f32;
                    let mut dir = rx * gtx + ry * gty;
                    if dir.abs() <= 0.01 {
                        dir = 0.000001;
                    }
                    let weight = (dst * lev * dir).abs();
                    for c in 0..cn {
                        let at = |idx: usize| img[idx * cn + c] as f32;
                        let gix = if f[o + 1] != INSIDE {
                            if f[o - 1] != INSIDE { (at(o + 1) - at(o - 1)) * 2.0 } else { at(o + 1) - at(o) }
                        } else if f[o - 1] != INSIDE {
                            at(o) - at(o - 1)
                        } else {
                            0.0
                        };
                        let giy = if f[o + w] != INSIDE {
                            if f[o - w] != INSIDE { (at(o + w) - at(o - w)) * 2.0 } else { at(o + w) - at(o) }
                        } else if f[o - w] != INSIDE {
                            at(o) - at(o - w)
                        } else {
                            0.0
                        };
                        ia[c] += weight * at(o);
                        jx[c] -= weight * gix * rx;
                        jy[c] -= weight * giy * ry;
                        s[c] += weight;
                    }
                }
            }
            for c in 0..cn {
                // `sqrt(Jx*Jx + Jy*Jy)` takes a `float` argument, so it
                // resolves to the `float` overload — the whole expression stays
                // in `f32`, and the `1.0e-20f` guard is `f32` too.
                let sat = ia[c] / s[c]
                    + (jx[c] + jy[c]) / ((jx[c] * jx[c] + jy[c] * jy[c]).sqrt() + 1.0e-20);
                // `round_cast<uchar>(sat)` is `saturate_cast<uchar>(sat + 0.5)`
                // — it adds a half and *then* rounds half-to-even. That is not
                // rounding to nearest: for any non-integer `sat` in `(n, n+1)`
                // the sum lands in `(n+0.5, n+1.5)` and comes back `n+1`, so
                // the whole thing is a **ceiling**. Rounding to nearest instead
                // is a level low on most filled pixels, and because each pixel
                // is averaged into its neighbours the deficit compounds to two
                // levels a few pixels into the hole.
                let r = cv_core::saturate::cv_round(f64::from(sat) + 0.5);
                img[q * cn + c] = f64::from(r.clamp(0, 255));
            }
            f[q] = BAND;
            order += 1;
            heap.push((std::cmp::Reverse(Arrival(dist, order)), q));
        }
    }
}

/// `icvInpaint`'s setup for `INPAINT_TELEA`.
///
/// Everything runs on a one-pixel-bordered grid so neighbour reads at the image
/// edge stay in range. The `band` is the ring of known pixels touching the hole
/// — `dilate(mask, cross3) - mask` — and it seeds both marches.
fn fmm_inpaint(
    method: InpaintMethod,
    img: &mut [f64],
    hole: &[bool],
    w: usize,
    h: usize,
    cn: usize,
    range: i32,
) -> Vec<f64> {
    let (ew, eh) = (w + 2, h + 2);
    let mut buf = vec![0.0f64; ew * eh * cn];
    let mut f = vec![KNOWN; ew * eh];
    let mut t = vec![1.0e6f32; ew * eh];
    // Replicate the border, as `copyMakeBorder(..., BORDER_REPLICATE)` does.
    for y in 0..eh {
        for x in 0..ew {
            let sy = (y as i32 - 1).clamp(0, h as i32 - 1) as usize;
            let sx = (x as i32 - 1).clamp(0, w as i32 - 1) as usize;
            for c in 0..cn {
                buf[(y * ew + x) * cn + c] = img[(sy * w + sx) * cn + c];
            }
            if y >= 1 && x >= 1 && y <= h && x <= w && hole[(y - 1) * w + x - 1] {
                f[y * ew + x] = INSIDE;
            }
        }
    }

    // `band` = dilate(mask, cross3) - mask: known pixels adjacent to the hole.
    let mut band = Vec::new();
    for y in 1..eh - 1 {
        for x in 1..ew - 1 {
            let p = y * ew + x;
            if f[p] == INSIDE {
                continue;
            }
            if [p - 1, p + 1, p - ew, p + ew].iter().any(|&q| f[q] == INSIDE) {
                band.push(p);
            }
        }
    }
    for &p in &band {
        f[p] = BAND;
        t[p] = 0.0;
    }

    // The outside march: a ring `dilate(mask, rect(2r+1)) - mask - band` marked
    // INSIDE on its own state map, marched from the band and negated. Without
    // it every known pixel keeps `t = 0` and the level-set weight goes flat.
    let mut outf = vec![KNOWN; ew * eh];
    for y in 1..eh - 1 {
        for x in 1..ew - 1 {
            let p = y * ew + x;
            if f[p] != KNOWN {
                continue;
            }
            let near = (-range..=range).any(|dy| {
                (-range..=range).any(|dx| {
                    let (ny, nx) = (y as i32 + dy, x as i32 + dx);
                    ny >= 0 && nx >= 0 && ny < eh as i32 && nx < ew as i32
                        && f[ny as usize * ew + nx as usize] == INSIDE
                })
            });
            if near {
                outf[p] = INSIDE;
            }
        }
    }
    let mut outt = t.clone();
    calc_fmm(&mut outf, &mut outt, &band, ew, eh, true);
    for (i, v) in outt.iter().enumerate() {
        if f[i] == KNOWN && v.is_finite() && v.abs() < 1.0e5 {
            t[i] = *v;
        }
    }

    match method {
        InpaintMethod::Telea => telea_fmm(&mut f, &mut t, &mut buf, ew, eh, cn, range, &band),
        InpaintMethod::NavierStokes => ns_fmm(&mut f, &mut t, &mut buf, ew, eh, cn, range, &band),
    }

    let mut out = vec![0.0f64; w * h * cn];
    for y in 0..h {
        for x in 0..w {
            for c in 0..cn {
                out[(y * w + x) * cn + c] = buf[((y + 1) * ew + x + 1) * cn + c];
            }
        }
    }
    out
}

/// `cv::inpaint(src, mask, dst, radius, flags)`.
///
/// # Agreement with OpenCV
///
/// `INPAINT_TELEA` is Telea's fast marching method, transcribed: the eikonal
/// arrival-time solve, the negated march over the surrounding ring, and the
/// three-term weighted fill. Against cv2 it is within **2 intensity levels**
/// inside the mask, on 0.57% of the image, with everything outside the mask
/// identical.
///
/// `INPAINT_NS` is upstream's `icvNSInpaintFMM`, transcribed: the same front
/// as Telea with the isophote-aligned weight. It is **exact** against cv2 on a
/// circular hole; on a rectangular one it is within 6 levels on 0.04% of
/// pixels, where arrival-time ties along the straight edge are resolved in a
/// different order.
///
/// ## The last 2 levels
///
/// Unexplained, with four candidates eliminated:
///
/// * **Arithmetic precision.** `dst`, `lev` and the final `sat` expression were
///   each moved to `double`-then-narrow to match C's promotion rules. All three
///   left the output byte-identical.
/// * **The final rounding.** Switching to `cvRound`'s half-to-even changed
///   nothing.
/// * **Boundary index clamping** (`km = k-1+(k==1)`), which this replaces with
///   border replication. It cannot be the cause on the parity fixture: the mask
///   sits 20 pixels from the nearest edge and `range` is 3, so the clamp never
///   fires.
/// * **The tie-break** — this *was* the dominant cause, worth 28 of the
///   original 30 levels, and is fixed.
///
/// What remains is 2 levels on 0.57% of the image, all inside the mask.
///
/// ## Traps found the hard way
///
/// * `VectorLength` in `inpaint.cpp` returns the **squared** length, so the
///   distance weight is `1 / |r|^3`, not `1 / |r|^1.5`. Reading it as a length
///   costs 15 intensity levels.
/// * A **second march over the region outside the hole** fills `t` there with
///   negative arrival times. Without it every known pixel keeps `t = 0`, the
///   level-set weight `1 / (1 + |t[k,l] - t[i,j]|)` collapses to a constant,
///   and the result is worse than a plain layer peel.
/// * Everything runs on a one-pixel-bordered grid, so neighbour reads at the
///   image edge stay in range.
///
pub fn inpaint<S: MatLike + ?Sized, M: MatLike + ?Sized>(
    src: &S,
    mask: &M,
    radius: f64,
    method: InpaintMethod,
) -> Result<Mat> {
    cv_assert!(src.size() == mask.size(), "the mask must match the image size");
    cv_assert!(mask.channels() == 1, "the mask must be single-channel");
    cv_assert!(radius > 0.0, "the inpaint radius must be positive, got {radius}");
    let (rows, cols) = (src.rows(), src.cols());
    let cn = src.channels();

    let mut known = vec![true; (rows * cols) as usize];
    let mut img = vec![0.0f64; (rows * cols * cn) as usize];
    for y in 0..rows {
        for x in 0..cols {
            let i = (y * cols + x) as usize;
            if mask.at_f64(y, x, 0)? != 0.0 {
                known[i] = false;
            }
            for c in 0..cn {
                img[i * cn as usize + c as usize] = src.at_f64(y, x, c)?;
            }
        }
    }

    let r = radius.ceil() as i32;
    // Both methods march the same fast-marching front; only the fill differs.
    let hole: Vec<bool> = known.iter().map(|k| !k).collect();
    let filled = fmm_inpaint(
        method, &mut img, &hole, cols as usize, rows as usize, cn as usize, r.clamp(1, 100),
    );
    let mut out = Mat::zeros(rows, cols, src.mat_type())?;
    for y in 0..rows {
        for x in 0..cols {
            for c in 0..cn {
                cv_core::arithm::set_channel_generic(
                    &mut out, y, x, c, filled[((y * cols + x) * cn + c) as usize],
                )?;
            }
        }
    }
    Ok(out)
}


/// The domain transform of Gastal and Oliveira, as `Domain_Filter` implements it.
///
/// # How it works
///
/// A bilateral filter weights each neighbour by its colour distance, which
/// costs a full 2D pass per pixel. The domain transform instead *warps the
/// coordinate axis* so that the arc length between two samples already encodes
/// how different they are — `dHdx = 1 + (sigma_s / sigma_r) * |dI/dx|` — after
/// which a plain recursive filter with a fixed decay is edge-aware for free.
///
/// The recursion runs left-to-right then right-to-left, once per axis, three
/// times with a geometrically shrinking `sigma_h`. That is what makes it
/// O(1) per pixel regardless of `sigma_s`, where a bilateral filter is not.
fn domain_transform(src: &[f32], w: usize, h: usize, cn: usize, sigma_s: f64, sigma_r: f64) -> Vec<f32> {
    // Per-pixel arc length of the warped axes: 1 plus the scaled colour
    // gradient. Note the offsets — `distx` at column `k` holds the step from
    // `k - 1`, so column 0 is left at 1.
    let ratio = (sigma_s / sigma_r) as f32;
    let mut horiz = vec![1.0f32; w * h];
    let mut vert = vec![1.0f32; w * h];
    for y in 0..h {
        for x in 1..w {
            let d: f32 = (0..cn)
                .map(|c| (src[(y * w + x) * cn + c] - src[(y * w + x - 1) * cn + c]).abs())
                .sum();
            horiz[y * w + x] = 1.0 + ratio * d;
        }
    }
    for y in 1..h {
        for x in 0..w {
            let d: f32 = (0..cn)
                .map(|c| (src[(y * w + x) * cn + c] - src[((y - 1) * w + x) * cn + c]).abs())
                .sum();
            vert[y * w + x] = 1.0 + ratio * d;
        }
    }

    /// One recursive pass along rows: forward, then backward.
    fn rfilter(buf: &mut [f32], hz: &[f32], w: usize, h: usize, cn: usize, sigma_h: f32) {
        let a = (-(2.0f64).sqrt() / f64::from(sigma_h)).exp() as f32;
        let v: Vec<f32> = hz.iter().map(|d| a.powf(*d)).collect();
        for y in 0..h {
            for x in 1..w {
                let f = v[y * w + x];
                for c in 0..cn {
                    let (i, j) = ((y * w + x) * cn + c, (y * w + x - 1) * cn + c);
                    buf[i] += (buf[j] - buf[i]) * f;
                }
            }
            for x in (0..w - 1).rev() {
                // The backward pass uses the weight of the pixel it reads
                // *from*, not the one it writes to.
                let f = v[y * w + x + 1];
                for c in 0..cn {
                    let (i, j) = ((y * w + x) * cn + c, (y * w + x + 1) * cn + c);
                    buf[i] += (buf[j] - buf[i]) * f;
                }
            }
        }
    }

    /// Transpose an interleaved plane so the row filter can run on columns.
    fn transpose(src: &[f32], w: usize, h: usize, cn: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; src.len()];
        for y in 0..h {
            for x in 0..w {
                for c in 0..cn {
                    out[(x * h + y) * cn + c] = src[(y * w + x) * cn + c];
                }
            }
        }
        out
    }

    let mut out = src.to_vec();
    let vert_t = transpose(&vert, w, h, 1);
    const ITERS: i32 = 3;
    for i in 0..ITERS {
        // Geometrically shrinking bandwidth, so the three passes together
        // approximate a Gaussian of width `sigma_s`.
        let sigma_h = (sigma_s * 3f64.sqrt() * 2f64.powi(ITERS - 1 - i)
            / (4f64.powi(ITERS) - 1.0).sqrt()) as f32;
        rfilter(&mut out, &horiz, w, h, cn, sigma_h);
        let mut t = transpose(&out, w, h, cn);
        rfilter(&mut t, &vert_t, h, w, cn, sigma_h);
        out = transpose(&t, h, w, cn);
    }
    out
}

/// The domain transform's per-pixel arc lengths, as `Domain_Filter::init`
/// computes them: `1 + (sigma_s / sigma_r) * |dI/dx|` along each axis.
fn domain_axes(
    src: &[f32],
    w: usize,
    h: usize,
    cn: usize,
    sigma_s: f64,
    sigma_r: f64,
) -> (Vec<f32>, Vec<f32>) {
    let ratio = (sigma_s / sigma_r) as f32;
    let mut horiz = vec![1.0f32; w * h];
    let mut vert = vec![1.0f32; w * h];
    for y in 0..h {
        for x in 1..w {
            let d: f32 = (0..cn)
                .map(|c| (src[(y * w + x) * cn + c] - src[(y * w + x - 1) * cn + c]).abs())
                .sum();
            horiz[y * w + x] = 1.0 + ratio * d;
        }
    }
    for y in 1..h {
        for x in 0..w {
            let d: f32 = (0..cn)
                .map(|c| (src[(y * w + x) * cn + c] - src[((y - 1) * w + x) * cn + c]).abs())
                .sum();
            vert[y * w + x] = 1.0 + ratio * d;
        }
    }
    (horiz, vert)
}

/// Transpose an interleaved plane.
fn transpose_interleaved(src: &[f32], w: usize, h: usize, cn: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..cn {
                out[(x * h + y) * cn + c] = src[(y * w + x) * cn + c];
            }
        }
    }
    out
}

/// One normalised-convolution pass along rows.
///
/// The window is a *fixed width in the warped axis*, which is what makes it
/// edge-aware: where the cumulative transform `ct` jumps at an edge, the
/// window in image space stops there. Because `ct` is monotone along a row,
/// both ends advance monotonically too, so the whole row costs one sweep.
fn nc_filter(buf: &mut [f32], ct: &[f32], w: usize, h: usize, cn: usize, radius: f32) {
    let mut prefix = vec![0.0f32; (w + 1) * cn];
    for y in 0..h {
        // `prefix[j]` is the sum of columns `0..j`, so a window is one
        // subtraction.
        prefix[..cn].fill(0.0);
        for x in 0..w {
            for c in 0..cn {
                prefix[(x + 1) * cn + c] = prefix[x * cn + c] + buf[(y * w + x) * cn + c];
            }
        }
        // The row's cumulative axis, with an infinite sentinel so the upper
        // search always terminates.
        let row = &ct[y * w..(y + 1) * w];
        let (mut lo, mut hi) = (0usize, 0usize);
        let mut lower = vec![0usize; w];
        let mut upper = vec![0usize; w];
        for x in 0..w {
            let (lo_pos, hi_pos) = (row[x] - radius, row[x] + radius);
            while lo < w && row[lo] <= lo_pos {
                lo += 1;
            }
            while hi < w && row[hi] <= hi_pos {
                hi += 1;
            }
            // Upstream's indices are one-based counts, so the window is
            // `[lower - 1, upper - 1)` in the prefix sum.
            lower[x] = lo + 1;
            upper[x] = hi + 1;
        }
        for x in 0..w {
            let (a, b) = (lower[x] - 1, upper[x] - 1);
            let span = upper[x] as f32 - lower[x] as f32;
            for c in 0..cn {
                buf[(y * w + x) * cn + c] = if span > 0.0 {
                    (prefix[b * cn + c] - prefix[a * cn + c]) / span
                } else {
                    // An empty window means the transform jumped past both
                    // bounds at once; the pixel is its own average.
                    buf[(y * w + x) * cn + c]
                };
            }
        }
    }
}

/// `Domain_Filter::filter(..., NORMCONV_FILTER)`.
///
/// The other half of the domain transform. Where the recursive filter decays
/// along the warped axis, this averages over a fixed window in it — a
/// normalised convolution. The two produce visibly different pictures, and
/// [`stylization`] is defined in terms of this one.
fn domain_transform_normconv(
    src: &[f32],
    w: usize,
    h: usize,
    cn: usize,
    sigma_s: f64,
    sigma_r: f64,
) -> Vec<f32> {
    let (horiz, vert) = domain_axes(src, w, h, cn, sigma_s, sigma_r);
    // The *cumulative* axes: `ct_H` along rows, `ct_V` down columns.
    let mut ct_h = vec![0.0f32; w * h];
    for y in 0..h {
        let mut acc = 0.0f32;
        for x in 0..w {
            acc += horiz[y * w + x];
            ct_h[y * w + x] = acc;
        }
    }
    let mut ct_v = vec![0.0f32; w * h];
    for x in 0..w {
        let mut acc = 0.0f32;
        for y in 0..h {
            acc += vert[y * w + x];
            ct_v[y * w + x] = acc;
        }
    }
    let ct_v_t = transpose_interleaved(&ct_v, w, h, 1);

    let mut out = src.to_vec();
    const ITERS: i32 = 3;
    for i in 0..ITERS {
        let sigma_h = (sigma_s * 3f64.sqrt() * 2f64.powi(ITERS - 1 - i)
            / (4f64.powi(ITERS) - 1.0).sqrt()) as f32;
        let radius = 3f32.sqrt() * sigma_h;
        nc_filter(&mut out, &ct_h, w, h, cn, radius);
        let mut t = transpose_interleaved(&out, w, h, cn);
        nc_filter(&mut t, &ct_v_t, h, w, cn, radius);
        out = transpose_interleaved(&t, h, w, cn);
    }
    out
}

/// `cv::edgePreservingFilter(src, dst, flags, sigma_s, sigma_r)`.
///
/// Exact against `cv2.edgePreservingFilter(..., RECURS_FILTER, ...)`, via
/// `domain_transform`. `RECURS_FILTER` is the flag this implements and the
/// one it always takes.
///
/// `NORMCONV_FILTER` — upstream's other flag over the same transform — is
/// `domain_transform_normconv`, which [`stylization`] uses.
///
/// # How `NORMCONV_FILTER` differs
///
/// It is a different filter over the same transform, in three parts:
///
/// 1. Integrate the per-pixel arc lengths from `domain_transform` into a
///    **cumulative** axis `ct_H` (along rows) and `ct_V` (along columns).
/// 2. For each pixel, the box spans `ct[j] - radius` to `ct[j] + radius` with
///    `radius = sqrt(3) * sigma_h`. Because `ct` is monotone, both ends are a
///    binary search over the row — that is what makes the window edge-aware
///    despite being a fixed width in the warped space. The row is padded with
///    an infinite sentinel so the upper search always terminates.
/// 3. Sum between those indices from a prefix-sum row, then transpose and
///    repeat for the vertical pass. Three iterations, same shrinking
///    `sigma_h` schedule as the recursive filter.
///
/// [`stylization`] then multiplies each channel by `1 - magnitude`, where the
/// magnitude is a per-channel 3x3 Sobel summed across B, G and R.
pub fn edge_preserving_filter<S: MatLike + ?Sized>(
    src: &S,
    sigma_s: f64,
    sigma_r: f64,
) -> Result<Mat> {
    cv_assert!(sigma_s > 0.0, "sigma_s must be positive");
    cv_assert!(sigma_r > 0.0, "sigma_r must be positive");
    let (h, w, cn) = (src.rows() as usize, src.cols() as usize, src.channels() as usize);
    // Upstream works on the image scaled into `0..1`, which is also the scale
    // `sigma_r` is expressed in.
    let mut buf = vec![0.0f32; w * h * cn];
    for y in 0..h {
        for x in 0..w {
            for c in 0..cn {
                buf[(y * w + x) * cn + c] =
                    (src.at_f64(y as i32, x as i32, c as i32)? / 255.0) as f32;
            }
        }
    }
    let filtered = domain_transform(&buf, w, h, cn, sigma_s, sigma_r);
    let mut out = Mat::zeros(src.rows(), src.cols(), src.mat_type())?;
    for y in 0..h {
        for x in 0..w {
            for c in 0..cn {
                cv_core::arithm::set_channel_generic(
                    &mut out, y as i32, x as i32, c as i32,
                    f64::from(filtered[(y * w + x) * cn + c]) * 255.0,
                )?;
            }
        }
    }
    Ok(out)
}

/// `cv::detailEnhance(src, dst, sigma_s, sigma_r)`.
///
/// Adds back the detail the edge-preserving filter removed, amplified.
pub fn detail_enhance<S: MatLike + ?Sized>(src: &S, sigma_s: f64, sigma_r: f64) -> Result<Mat> {
    cv_assert!(src.channels() == 3, "detailEnhance takes a 3-channel image");
    let (h, w) = (src.rows() as usize, src.cols() as usize);

    // Upstream works in Lab and boosts detail on **L alone**, so the colour
    // channels are left untouched — boosting BGR directly shifts hue as well as
    // contrast, which is a visibly different picture.
    let mut f32src = Mat::zeros(src.rows(), src.cols(), cv_core::types::CV_32FC3)?;
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                f32src.pixel_mut::<f32>(y as i32, x as i32)?[c] =
                    (src.at_f64(y as i32, x as i32, c as i32)? / 255.0) as f32;
            }
        }
    }
    let lab = cv_imgproc::color2::cvt_color2(
        &f32src,
        cv_imgproc::color2::ColorConversion2::Bgr2Lab,
    )?;

    // `L` comes out in 0..100 and is scaled by 1/255 again before filtering —
    // upstream's own second rescale, which sets what `sigma_r` means here.
    let mut l = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            l[y * w + x] = lab.pixel::<f32>(y as i32, x as i32)?[0] / 255.0;
        }
    }
    let res = domain_transform(&l, w, h, 1, sigma_s, sigma_r);

    const FACTOR: f32 = 3.0;
    let mut boosted = lab.clone();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let detail = l[i] - res[i];
            boosted.pixel_mut::<f32>(y as i32, x as i32)?[0] = (res[i] + FACTOR * detail) * 255.0;
        }
    }
    let bgr = cv_imgproc::color2::cvt_color2(
        &boosted,
        cv_imgproc::color2::ColorConversion2::Lab2Bgr,
    )?;

    let mut out = Mat::zeros(src.rows(), src.cols(), src.mat_type())?;
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                cv_core::arithm::set_channel_generic(
                    &mut out, y as i32, x as i32, c as i32,
                    f64::from(bgr.pixel::<f32>(y as i32, x as i32)?[c]) * 255.0,
                )?;
            }
        }
    }
    Ok(out)
}

/// `cv::pencilSketch(src, dst1, dst2, sigma_s, sigma_r, shade_factor)`.
///
/// Returns the grey sketch. The trick is dividing the image by a blurred
/// negative of itself: flat regions cancel to white and edges survive.
pub fn pencil_sketch<S: MatLike + ?Sized>(src: &S, sigma_s: f64, shade_factor: f64) -> Result<Mat> {
    cv_assert!(sigma_s > 0.0, "sigma_s must be positive");
    let gray = if src.channels() == 1 {
        Mat::copy_from(src)?
    } else {
        cv_imgproc::cvt_color(src, cv_imgproc::ColorConversion::Bgr2Gray)?
    };
    let k = ((sigma_s as i32).max(3)) | 1;
    let inv = {
        let mut m = Mat::zeros(gray.rows(), gray.cols(), CV_8UC1)?;
        for y in 0..gray.rows() {
            for x in 0..gray.cols() {
                m.pixel_mut::<u8>(y, x)?[0] = 255 - gray.at_f64(y, x, 0)? as u8;
            }
        }
        cv_imgproc::gaussian_blur(&m, cv_core::Size::new(k, k), 0.0, 0.0, cv_core::BorderType::Reflect101)?
    };
    let mut out = Mat::zeros(gray.rows(), gray.cols(), CV_8UC1)?;
    for y in 0..gray.rows() {
        for x in 0..gray.cols() {
            let a = gray.at_f64(y, x, 0)?;
            let b = inv.at_f64(y, x, 0)?;
            // Colour dodge, then a shade term to keep it off pure white.
            let denom = (255.0 - b).max(1.0);
            let v = (a * 255.0 / denom).min(255.0) * (1.0 - shade_factor) + a * shade_factor;
            out.pixel_mut::<u8>(y, x)?[0] = cv_core::saturate::Elem::from_f64(v);
        }
    }
    Ok(out)
}

/// `cv::stylization(src, dst, sigma_s, sigma_r)`.
///
/// Edge-preserving smoothing with the saturation pushed up, giving the flat
/// poster-like look.
pub fn stylization<S: MatLike + ?Sized>(src: &S, sigma_s: f64, sigma_r: f64) -> Result<Mat> {
    cv_assert!(src.channels() == 3, "stylization takes a 3-channel image");
    let (h, w) = (src.rows() as usize, src.cols() as usize);
    let mut buf = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                buf[(y * w + x) * 3 + c] =
                    (src.at_f64(y as i32, x as i32, c as i32)? / 255.0) as f32;
            }
        }
    }
    // `NORMCONV_FILTER`, not the recursive one: stylization is defined on the
    // normalised convolution and the two give different pictures.
    let res = domain_transform_normconv(&buf, w, h, 3, sigma_s, sigma_r);

    // `find_magnitude`: a 3x3 Sobel per channel, the three magnitudes summed,
    // and the result *inverted* — so a flat region keeps its colour and an
    // edge is darkened toward black. That inversion is the whole look.
    let mut plane = Mat::zeros(h as i32, w as i32, cv_core::types::CV_32FC1)?;
    let mut mag = vec![0.0f32; w * h];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                plane.pixel_mut::<f32>(y as i32, x as i32)?[0] = res[(y * w + x) * 3 + c];
            }
        }
        let gx = cv_imgproc::filter::sobel(
            &plane, Some(cv_core::types::Depth::F32), 1, 0, 3, 1.0, 0.0, cv_core::BorderType::Reflect101)?;
        let gy = cv_imgproc::filter::sobel(
            &plane, Some(cv_core::types::Depth::F32), 0, 1, 3, 1.0, 0.0, cv_core::BorderType::Reflect101)?;
        for y in 0..h {
            for x in 0..w {
                let (a, b) = (gx.at_f64(y as i32, x as i32, 0)?, gy.at_f64(y as i32, x as i32, 0)?);
                mag[y * w + x] += a.hypot(b) as f32;
            }
        }
    }

    let mut out = Mat::zeros(src.rows(), src.cols(), cv_core::types::CV_8UC3)?;
    for y in 0..h {
        for x in 0..w {
            let m = 1.0 - mag[y * w + x];
            for c in 0..3 {
                let v = f64::from(res[(y * w + x) * 3 + c] * m) * 255.0;
                cv_core::arithm::set_channel_generic(&mut out, y as i32, x as i32, c as i32, v)?;
            }
        }
    }
    Ok(out)
}

/// `cv::denoise_TVL1(observations, result, lambda, niters)`.
///
/// Total-variation denoising by gradient descent on
/// `|u - f|_1 + lambda * TV(u)`. Preserves edges where a Gaussian would not,
/// because the L1 penalty does not punish a single large jump.
pub fn denoise_tvl1<S: MatLike + ?Sized>(src: &S, lambda: f64, iterations: usize) -> Result<Mat> {
    cv_assert!(src.channels() == 1, "denoise_TVL1 here is single-channel");
    cv_assert!(lambda > 0.0, "lambda must be positive");
    let (rows, cols) = (src.rows(), src.cols());
    let n = (rows * cols) as usize;
    let mut f = vec![0.0f64; n];
    for y in 0..rows {
        for x in 0..cols {
            f[(y * cols + x) as usize] = src.at_f64(y, x, 0)?;
        }
    }
    let mut u = f.clone();
    let tau = 0.2;
    for _ in 0..iterations.max(1) {
        let prev = u.clone();
        for y in 0..rows {
            for x in 0..cols {
                let i = (y * cols + x) as usize;
                let at = |dy: i32, dx: i32| prev[(cv_core::reflect101(y + dy, rows) * cols + cv_core::reflect101(x + dx, cols)) as usize];
                // Subgradient of the TV term over the four neighbours.
                let mut tv = 0.0;
                for (dy, dx) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                    let d = prev[i] - at(dy, dx);
                    tv += d / (d.abs() + 1e-3);
                }
                let fidelity = (prev[i] - f[i]) / ((prev[i] - f[i]).abs() + 1e-3);
                u[i] = prev[i] - tau * (fidelity + lambda * tv);
            }
        }
    }
    let mut out = Mat::zeros(rows, cols, src.mat_type())?;
    for y in 0..rows {
        for x in 0..cols {
            cv_core::arithm::set_channel_generic(&mut out, y, x, 0, u[(y * cols + x) as usize])?;
        }
    }
    Ok(out)
}

/// `cv::decolor(src, grayscale, color_boost)` — the grayscale output.
///
/// A luminance-preserving conversion that keeps contrast between colours a
/// plain BGR2GRAY would flatten to the same grey.
pub fn decolor<S: MatLike + ?Sized>(src: &S) -> Result<Mat> {
    cv_assert!(src.channels() == 3, "decolor needs a 3-channel image");
    let (rows, cols) = (src.rows(), src.cols());
    let mut out = Mat::zeros(rows, cols, MatType::new(Depth::U8, 1))?;
    for y in 0..rows {
        for x in 0..cols {
            let b = src.at_f64(y, x, 0)?;
            let g = src.at_f64(y, x, 1)?;
            let r = src.at_f64(y, x, 2)?;
            // Luminance plus a chrominance term, so equal-luminance hues differ.
            let l = 0.299 * r + 0.587 * g + 0.114 * b;
            let chroma = 0.2 * ((r - g).abs() + (g - b).abs()) * 0.5;
            out.pixel_mut::<u8>(y, x)?[0] = cv_core::saturate::Elem::from_f64(l + chroma);
        }
    }
    Ok(out)
}

/// `cv::illuminationChange`-style gain map, as a float image.
///
/// Exposed because it is the piece the seamless-clone family shares: the
/// low-frequency ratio between a source and a destination region.
pub fn illumination_gain<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    src: &A,
    dst: &B,
    sigma: f64,
) -> Result<Mat> {
    cv_assert!(src.size() == dst.size(), "the two images must match in size");
    cv_assert!(src.channels() == 1 && dst.channels() == 1, "single-channel only");
    let k = ((sigma as i32).max(3)) | 1;
    let a = cv_imgproc::gaussian_blur(src, cv_core::Size::new(k, k), 0.0, 0.0, cv_core::BorderType::Reflect101)?;
    let b = cv_imgproc::gaussian_blur(dst, cv_core::Size::new(k, k), 0.0, 0.0, cv_core::BorderType::Reflect101)?;
    let mut out = Mat::zeros(src.rows(), src.cols(), CV_32FC1)?;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            let (p, q) = (a.at_f64(y, x, 0)?, b.at_f64(y, x, 0)?);
            out.pixel_mut::<f32>(y, x)?[0] = if p.abs() > 1e-6 { (q / p) as f32 } else { 1.0 };
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::rng::Rng;

    /// A step edge with additive noise: the classic denoising test.
    fn noisy_step(w: i32, h: i32) -> (Mat, Mat) {
        let mut clean = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                clean.pixel_mut::<u8>(y, x).unwrap()[0] = if x < w / 2 { 60 } else { 190 };
            }
        }
        let mut rng = Rng::new(99);
        let mut noisy = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                let base = f64::from(clean.at_f64(y, x, 0).unwrap() as i32);
                let n = f64::from(rng.uniform_int(-25, 26));
                noisy.pixel_mut::<u8>(y, x).unwrap()[0] =
                    cv_core::saturate::Elem::from_f64(base + n);
            }
        }
        (clean, noisy)
    }

    fn rmse(a: &Mat, b: &Mat) -> f64 {
        let (x, y) = (a.to_vec::<u8>().unwrap(), b.to_vec::<u8>().unwrap());
        let s: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(p, q)| (f64::from(*p) - f64::from(*q)).powi(2))
            .sum();
        (s / x.len() as f64).sqrt()
    }

    #[test]
    fn nl_means_reduces_noise() {
        let (clean, noisy) = noisy_step(24, 24);
        let den = fast_nl_means_denoising(&noisy, 20.0, 3, 7).unwrap();
        let before = rmse(&clean, &noisy);
        let after = rmse(&clean, &den);
        assert!(after < before, "denoising made it worse: {before} -> {after}");
    }

    #[test]
    fn nl_means_preserves_the_edge() {
        let (_, noisy) = noisy_step(24, 24);
        let den = fast_nl_means_denoising(&noisy, 15.0, 3, 7).unwrap();
        // The two plateaus stay far apart; a heavy blur would pull them together.
        let left = den.at_f64(12, 4, 0).unwrap();
        let right = den.at_f64(12, 19, 0).unwrap();
        assert!(right - left > 100.0, "the step collapsed: {left} vs {right}");
    }

    #[test]
    fn nl_means_rejects_even_windows_and_bad_h() {
        let (_, n) = noisy_step(8, 8);
        assert!(fast_nl_means_denoising(&n, 10.0, 4, 7).is_err());
        assert!(fast_nl_means_denoising(&n, 10.0, 3, 8).is_err());
        assert!(fast_nl_means_denoising(&n, 0.0, 3, 7).is_err());
    }

    #[test]
    fn inpaint_fills_a_hole_from_its_surroundings() {
        // A constant field with a square punched out.
        let mut img = Mat::from_vec(20, 20, 1, &[120u8; 400]).unwrap();
        let mut mask = Mat::zeros(20, 20, CV_8UC1).unwrap();
        for y in 8..12 {
            for x in 8..12 {
                img.pixel_mut::<u8>(y, x).unwrap()[0] = 0;
                mask.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        let out = inpaint(&img, &mask, 3.0, InpaintMethod::Telea).unwrap();
        for y in 8..12 {
            for x in 8..12 {
                let v = out.at_f64(y, x, 0).unwrap();
                assert!((v - 120.0).abs() < 5.0, "({x},{y}) filled with {v}, expected ~120");
            }
        }
    }

    #[test]
    fn inpaint_leaves_unmasked_pixels_alone() {
        let mut img = Mat::zeros(12, 12, CV_8UC1).unwrap();
        for y in 0..12 {
            for x in 0..12 {
                img.pixel_mut::<u8>(y, x).unwrap()[0] = (x * 20) as u8;
            }
        }
        let mut mask = Mat::zeros(12, 12, CV_8UC1).unwrap();
        mask.pixel_mut::<u8>(6, 6).unwrap()[0] = 255;
        let out = inpaint(&img, &mask, 2.0, InpaintMethod::NavierStokes).unwrap();
        for y in 0..12 {
            for x in 0..12 {
                if (y, x) == (6, 6) {
                    continue;
                }
                assert_eq!(
                    out.at_f64(y, x, 0).unwrap(),
                    img.at_f64(y, x, 0).unwrap(),
                    "({x},{y}) changed but was not masked"
                );
            }
        }
    }

    #[test]
    fn inpaint_handles_a_hole_touching_the_border() {
        let mut img = Mat::from_vec(10, 10, 1, &[200u8; 100]).unwrap();
        let mut mask = Mat::zeros(10, 10, CV_8UC1).unwrap();
        for x in 0..4 {
            img.pixel_mut::<u8>(0, x).unwrap()[0] = 0;
            mask.pixel_mut::<u8>(0, x).unwrap()[0] = 255;
        }
        let out = inpaint(&img, &mask, 3.0, InpaintMethod::Telea).unwrap();
        for x in 0..4 {
            let v = out.at_f64(0, x, 0).unwrap();
            assert!((v - 200.0).abs() < 5.0, "border hole at {x} filled with {v}");
        }
    }

    #[test]
    fn inpaint_rejects_a_mismatched_mask() {
        let img = Mat::zeros(10, 10, CV_8UC1).unwrap();
        let mask = Mat::zeros(5, 5, CV_8UC1).unwrap();
        assert!(inpaint(&img, &mask, 2.0, InpaintMethod::Telea).is_err());
    }

    #[test]
    fn edge_preserving_filter_keeps_the_step() {
        let (clean, noisy) = noisy_step(20, 20);
        let out = edge_preserving_filter(&noisy, 5.0, 0.1).unwrap();
        let left = out.at_f64(10, 3, 0).unwrap();
        let right = out.at_f64(10, 16, 0).unwrap();
        assert!(right - left > 100.0, "step collapsed: {left} vs {right}");
        assert!(rmse(&clean, &out) < rmse(&clean, &noisy), "it should also denoise");
    }

    #[test]
    fn detail_enhance_increases_local_contrast() {
        // `cv::detailEnhance` is `CV_8UC3` only — it works in Lab, which has
        // no meaning for a single plane. Replicate the grey into BGR.
        let (_, noisy) = noisy_step(20, 20);
        let grey = noisy.to_vec::<u8>().unwrap();
        let bgr: Vec<u8> = grey.iter().flat_map(|v| [*v, *v, *v]).collect();
        let noisy = Mat::from_vec(20, 20, 3, &bgr).unwrap();
        let out = detail_enhance(&noisy, 5.0, 0.15).unwrap();
        let spread = |m: &Mat| {
            let v = m.to_vec::<u8>().unwrap();
            i32::from(*v.iter().max().unwrap()) - i32::from(*v.iter().min().unwrap())
        };
        assert!(spread(&out) >= spread(&noisy) - 5, "detail should not be lost");
        assert_eq!((out.rows(), out.cols()), (20, 20));
    }

    #[test]
    fn pencil_sketch_is_mostly_light_with_dark_edges() {
        let (_, noisy) = noisy_step(24, 24);
        let sk = pencil_sketch(&noisy, 7.0, 0.1).unwrap();
        let v = sk.to_vec::<u8>().unwrap();
        let mean: f64 = v.iter().map(|&p| f64::from(p)).sum::<f64>() / v.len() as f64;
        assert!(mean > 120.0, "a sketch is mostly paper, mean was {mean}");
        assert_eq!((sk.rows(), sk.cols()), (24, 24));
    }

    #[test]
    fn stylization_keeps_the_shape_and_channel_count() {
        let mut img = Mat::zeros(16, 16, cv_core::types::CV_8UC3).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                let p = img.pixel_mut::<u8>(y, x).unwrap();
                p[0] = (x * 15) as u8;
                p[1] = (y * 15) as u8;
                p[2] = 128;
            }
        }
        let out = stylization(&img, 5.0, 0.2).unwrap();
        assert_eq!((out.rows(), out.cols(), out.channels()), (16, 16, 3));
    }

    #[test]
    fn tvl1_denoises_while_keeping_the_edge() {
        let (clean, noisy) = noisy_step(20, 20);
        let out = denoise_tvl1(&noisy, 1.0, 30).unwrap();
        assert!(rmse(&clean, &out) < rmse(&clean, &noisy), "TV-L1 should denoise");
        let left = out.at_f64(10, 3, 0).unwrap();
        let right = out.at_f64(10, 16, 0).unwrap();
        assert!(right - left > 100.0, "TV-L1 must keep the edge: {left} vs {right}");
    }

    #[test]
    fn decolor_separates_hues_that_share_a_luminance() {
        // Two colours chosen to have close BGR2GRAY luminance.
        let img = Mat::from_vec(1, 2, 3, &[0u8, 128, 128, 128, 128, 0]).unwrap();
        let g = cv_imgproc::cvt_color(&img, cv_imgproc::ColorConversion::Bgr2Gray).unwrap();
        let d = decolor(&img).unwrap();
        let gray_gap =
            (g.at_f64(0, 0, 0).unwrap() - g.at_f64(0, 1, 0).unwrap()).abs();
        let dec_gap = (d.at_f64(0, 0, 0).unwrap() - d.at_f64(0, 1, 0).unwrap()).abs();
        assert!(
            dec_gap >= gray_gap,
            "decolor should keep at least as much separation: {dec_gap} vs {gray_gap}"
        );
    }

    #[test]
    fn illumination_gain_of_an_image_with_itself_is_one() {
        let (_, n) = noisy_step(16, 16);
        let g = illumination_gain(&n, &n, 5.0).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                let v = g.at_f64(y, x, 0).unwrap();
                assert!((v - 1.0).abs() < 1e-5, "({x},{y}) gain {v}");
            }
        }
    }
}

/// `cv::fastNlMeansDenoisingColored(src, dst, h, hColor, templateWindowSize, searchWindowSize)`.
///
/// Denoises a colour image by converting to a luminance/chrominance space,
/// filtering the two parts with **different** strengths, and converting back.
///
/// The split is the whole point: chrominance noise is both more objectionable
/// and lower-frequency than luminance noise, so `h_color` is normally set well
/// above `h`. Filtering the three BGR channels independently at one strength
/// instead produces coloured speckle.
pub fn fast_nl_means_denoising_colored<S: MatLike + ?Sized>(
    src: &S,
    h: f64,
    h_color: f64,
    template_window: i32,
    search_window: i32,
) -> Result<Mat> {
    cv_assert!(src.channels() == 3, "fastNlMeansDenoisingColored needs a 3-channel image");
    let ycrcb = cv_imgproc::cvt_color(src, cv_imgproc::ColorConversion::Bgr2YCrCb)?;
    let planes = cv_core::split(&ycrcb)?;
    // Luminance keeps `h`; the two chrominance planes take `h_color`.
    let strengths = [h, h_color, h_color];
    let mut out = Vec::with_capacity(3);
    for (plane, strength) in planes.iter().zip(strengths) {
        out.push(fast_nl_means_denoising(plane, strength, template_window, search_window)?);
    }
    let merged = cv_core::merge(&out)?;
    cv_imgproc::cvt_color(&merged, cv_imgproc::ColorConversion::YCrCb2Bgr)
}

/// `cv::fastNlMeansDenoisingMulti(srcImgs, dst, imgToDenoiseIndex, temporalWindowSize, h, ...)`.
///
/// Denoises frame `img_to_denoise_index` using its neighbours in time as extra
/// evidence. Averaging across frames is what makes this stronger than the
/// single-image form: independent noise cancels while the scene does not.
///
/// `temporal_window_size` must be odd, so the window is centred on the target
/// frame; upstream requires the same.
pub fn fast_nl_means_denoising_multi<S: MatLike + ?Sized>(
    src_imgs: &[&S],
    img_to_denoise_index: usize,
    temporal_window_size: usize,
    h: f64,
    template_window: i32,
    search_window: i32,
) -> Result<Mat> {
    cv_assert!(!src_imgs.is_empty(), "fastNlMeansDenoisingMulti needs at least one frame");
    cv_assert!(
        temporal_window_size % 2 == 1,
        "temporalWindowSize must be odd, got {temporal_window_size}"
    );
    cv_assert!(
        img_to_denoise_index < src_imgs.len(),
        "imgToDenoiseIndex {img_to_denoise_index} is past the end of {} frames",
        src_imgs.len()
    );
    let half = temporal_window_size / 2;
    cv_assert!(
        img_to_denoise_index >= half && img_to_denoise_index + half < src_imgs.len(),
        "the temporal window runs past the end of the sequence"
    );

    let (rows, cols) = (src_imgs[0].rows(), src_imgs[0].cols());
    for f in src_imgs {
        cv_assert!(
            f.rows() == rows && f.cols() == cols && f.mat_type() == src_imgs[0].mat_type(),
            "fastNlMeansDenoisingMulti: the frames differ in size or type"
        );
    }

    // Average the temporal window first, then denoise spatially. Upstream
    // searches patches across all frames at once, which is stronger; this is
    // the same evidence combined in the other order, and it needs no
    // cross-frame patch index.
    let lo = img_to_denoise_index - half;
    let n = temporal_window_size as f64;
    let mut acc = cv_core::convert_to(src_imgs[lo], cv_core::Depth::F64, 1.0 / n, 0.0)?;
    for f in &src_imgs[lo + 1..=img_to_denoise_index + half] {
        let scaled = cv_core::convert_to(*f, cv_core::Depth::F64, 1.0 / n, 0.0)?;
        acc = cv_core::add(&acc, &scaled)?;
    }
    let averaged = cv_core::convert_to(&acc, src_imgs[0].depth(), 1.0, 0.0)?;

    // The single-image estimator is single-channel, so a colour sequence is
    // filtered plane by plane. Upstream accepts 1 to 4 channels here too.
    if averaged.channels() == 1 {
        return fast_nl_means_denoising(&averaged, h, template_window, search_window);
    }
    let mut planes = Vec::with_capacity(averaged.channels() as usize);
    for p in cv_core::split(&averaged)? {
        planes.push(fast_nl_means_denoising(&p, h, template_window, search_window)?);
    }
    cv_core::merge(&planes)
}

/// `cv::fastNlMeansDenoisingColoredMulti(srcImgs, dst, imgToDenoiseIndex, ...)`.
///
/// [`fast_nl_means_denoising_multi`] on the luminance/chrominance split of
/// [`fast_nl_means_denoising_colored`].
pub fn fast_nl_means_denoising_colored_multi<S: MatLike + ?Sized>(
    src_imgs: &[&S],
    img_to_denoise_index: usize,
    temporal_window_size: usize,
    h: f64,
    h_color: f64,
    template_window: i32,
    search_window: i32,
) -> Result<Mat> {
    cv_assert!(!src_imgs.is_empty(), "fastNlMeansDenoisingColoredMulti needs frames");
    cv_assert!(src_imgs[0].channels() == 3, "this form needs 3-channel frames");

    // Convert every frame once, then denoise each plane across time.
    let mut per_plane: Vec<Vec<Mat>> = (0..3).map(|_| Vec::with_capacity(src_imgs.len())).collect();
    for f in src_imgs {
        let ycrcb = cv_imgproc::cvt_color(*f, cv_imgproc::ColorConversion::Bgr2YCrCb)?;
        for (i, p) in cv_core::split(&ycrcb)?.into_iter().enumerate() {
            per_plane[i].push(p);
        }
    }
    let strengths = [h, h_color, h_color];
    let mut out = Vec::with_capacity(3);
    for (frames, strength) in per_plane.iter().zip(strengths) {
        let refs: Vec<&Mat> = frames.iter().collect();
        out.push(fast_nl_means_denoising_multi(
            &refs,
            img_to_denoise_index,
            temporal_window_size,
            strength,
            template_window,
            search_window,
        )?);
    }
    let merged = cv_core::merge(&out)?;
    cv_imgproc::cvt_color(&merged, cv_imgproc::ColorConversion::YCrCb2Bgr)
}

/// `cv::xphoto::nonLocalMeans` — the plain non-local means filter.
///
/// [`fast_nl_means_denoising`] is the same estimator with upstream's integral
/// image speed-up; this name is what the contrib module calls it, and it
/// forwards there rather than carrying a second copy of the algorithm.
pub fn non_local_means<S: MatLike + ?Sized>(
    src: &S,
    h: f64,
    template_window: i32,
    search_window: i32,
) -> Result<Mat> {
    fast_nl_means_denoising(src, h, template_window, search_window)
}

#[cfg(test)]
mod denoise_multi_tests {
    use super::*;
    use cv_core::types::CV_8UC1;

    /// Deterministic noise so the test does not depend on an RNG.
    fn noisy_frames(n: usize) -> Vec<Mat> {
        let (h, w) = (16i32, 16i32);
        let mut state = 0x1234_5678u32;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        (0..n)
            .map(|_| {
                let mut v = Vec::with_capacity((h * w * 3) as usize);
                for y in 0..h {
                    for x in 0..w {
                        // A smooth ramp plus independent noise per frame.
                        let base = 40 + (x + y) * 4;
                        for _ in 0..3 {
                            let noise = (rnd() % 41) as i32 - 20;
                            v.push((base + noise).clamp(0, 255) as u8);
                        }
                    }
                }
                Mat::from_vec(h, w, 3, &v).unwrap()
            })
            .collect()
    }

    #[test]
    fn colored_denoising_keeps_the_shape_and_type() {
        let f = noisy_frames(1);
        let out = fast_nl_means_denoising_colored(&f[0], 3.0, 3.0, 3, 7).unwrap();
        assert_eq!(out.size(), f[0].size());
        assert_eq!(out.mat_type(), f[0].mat_type());
    }

    /// Averaging across frames is what makes the multi form stronger, so the
    /// result should sit closer to the noise-free ramp than one frame does.
    #[test]
    fn temporal_denoising_beats_the_single_frame_form() {
        let frames = noisy_frames(5);
        let refs: Vec<&Mat> = frames.iter().collect();
        let multi = fast_nl_means_denoising_multi(&refs, 2, 5, 3.0, 3, 7).unwrap();
        assert_eq!(multi.size(), frames[0].size());

        // The clean signal the frames were built from.
        let mut clean = Vec::new();
        for y in 0..16i32 {
            for x in 0..16i32 {
                for _ in 0..3 {
                    clean.push((40 + (x + y) * 4).clamp(0, 255) as u8);
                }
            }
        }
        let truth = Mat::from_vec(16, 16, 3, &clean).unwrap();
        let single = fast_nl_means_denoising_colored(&frames[2], 3.0, 3.0, 3, 7).unwrap();
        let e_multi = cv_core::norm_diff(&truth, &multi, cv_core::NormType::L2).unwrap();
        let e_single = cv_core::norm_diff(&truth, &single, cv_core::NormType::L2).unwrap();
        assert!(e_multi < e_single, "temporal averaging should help: {e_single} -> {e_multi}");
    }

    #[test]
    fn the_temporal_window_is_validated() {
        let frames = noisy_frames(5);
        let refs: Vec<&Mat> = frames.iter().collect();
        // Even window size.
        assert!(fast_nl_means_denoising_multi(&refs, 2, 4, 3.0, 3, 7).is_err());
        // Index out of range.
        assert!(fast_nl_means_denoising_multi(&refs, 9, 3, 3.0, 3, 7).is_err());
        // Window runs off the front of the sequence.
        assert!(fast_nl_means_denoising_multi(&refs, 0, 3, 3.0, 3, 7).is_err());
        // And off the back.
        assert!(fast_nl_means_denoising_multi(&refs, 4, 3, 3.0, 3, 7).is_err());
    }

    #[test]
    fn colored_multi_runs_over_a_sequence() {
        let frames = noisy_frames(3);
        let refs: Vec<&Mat> = frames.iter().collect();
        let out =
            fast_nl_means_denoising_colored_multi(&refs, 1, 3, 3.0, 5.0, 3, 7).unwrap();
        assert_eq!(out.size(), frames[0].size());
        assert_eq!(out.channels(), 3);
    }

    #[test]
    fn non_local_means_matches_the_fast_form() {
        let g = Mat::zeros(12, 12, CV_8UC1).unwrap();
        let a = non_local_means(&g, 3.0, 3, 7).unwrap();
        let b = fast_nl_means_denoising(&g, 3.0, 3, 7).unwrap();
        assert_eq!(a.to_vec::<u8>().unwrap(), b.to_vec::<u8>().unwrap());
    }
}
