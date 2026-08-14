//! BRISK's scale space and detector (`modules/features2d/src/brisk.cpp`).
//!
//! # A pyramid with two interleaved ladders
//!
//! Most detectors halve the image per octave. BRISK interleaves a second ladder
//! offset by a factor of 1.5, so consecutive layers alternate `x1, x1.5, x2,
//! x3, x4, x6, ...`. Even layers are octaves, odd ones intra-octaves. That
//! doubles the scale resolution without doubling the octave count, which is
//! what lets the scale refinement below fit a parabola through three layers
//! rather than settling for the nearest one.
//!
//! Each layer keeps an AGAST **score surface**, not just its keypoints, because
//! the refinement reads scores at fractional positions in the layers above and
//! below.
//!
//! # Three tests a candidate has to pass
//!
//! 1. `is_max_2d` — maximal in its own 3x3 neighbourhood, with an explicit
//!    tie-break: if any neighbour *equals* the centre, both are re-scored with
//!    a 3x3 smoothing mask and the centre must still win.
//! 2. `Pyramid::score_max_above` and `Pyramid::score_max_below` — no score
//!    in the corresponding patch of the neighbouring layers may exceed it.
//!    These bail out the moment one does, so most candidates die cheaply.
//! 3. A parabola through those three layer maxima, giving a continuous scale.
//!    The three `refine_1d` variants differ only in the layer spacing they
//!    assume — 0.75/1.5, 2/3 to 4/3, and 0.7/1.5 for the bottom octave.
//!
//! # Agreement with OpenCV
//!
//! See `detector_matches_cv2` — keypoint count, positions, sizes and responses
//! against `cv2.BRISK_create().detect()`.

use cv_core::{cv_assert, MatLike, Result};

use crate::detectors::AgastType;
use crate::regions::agast_corner_score;
use crate::KeyPoint;

/// `BriskScaleSpace::basicSize_` — the `size` of a keypoint at scale 1.
const BASIC_SIZE: f32 = 12.0;
/// `BriskScaleSpace::safetyFactor_`.
const SAFETY_FACTOR: f32 = 1.0;

/// One layer: an image, its cached AGAST scores, and where it sits.
struct Layer {
    img: Vec<u8>,
    w: usize,
    h: usize,
    /// Lazily filled `agast_cornerScore` surface, in upstream's cache order.
    scores: Vec<u8>,
    /// How much larger the original image is than this layer.
    scale: f32,
    /// Where this layer's pixel centres sit in the original's frame.
    offset: f32,
}

impl Layer {
    fn from_image(img: Vec<u8>, w: usize, h: usize, scale: f32, offset: f32) -> Self {
        Layer { img, w, h, scores: vec![0u8; w * h], scale, offset }
    }

    /// `BriskLayer::getAgastScore(int, int, threshold)`.
    ///
    /// The cache is upstream's, and it is load-bearing rather than an
    /// optimisation: a cached value of 1 or 2 is indistinguishable from
    /// "not computed yet", so those pixels are recomputed on every visit, and
    /// a pixel whose score fell below the *first* threshold it was queried at
    /// keeps that 0 even when queried later at a lower one.
    fn score(&mut self, x: i32, y: i32, threshold: i32) -> i32 {
        if x < 3 || y < 3 || x >= self.w as i32 - 3 || y >= self.h as i32 - 3 {
            return 0;
        }
        let idx = y as usize * self.w + x as usize;
        if self.scores[idx] > 2 {
            return i32::from(self.scores[idx]);
        }
        let mut s = agast_corner_score(
            &self.img,
            self.w,
            self.h,
            x,
            y,
            (threshold - 1).max(1),
            AgastType::Oast9_16,
        );
        if s < threshold {
            s = 0;
        }
        self.scores[idx] = s as u8;
        s
    }

    /// `BriskLayer::getAgastScore_5_8` — uncached, and used only below layer 0.
    fn score_5_8(&self, x: i32, y: i32, threshold: i32) -> i32 {
        if x < 2 || y < 2 || x >= self.w as i32 - 2 || y >= self.h as i32 - 2 {
            return 0;
        }
        let s = agast_corner_score(
            &self.img,
            self.w,
            self.h,
            x,
            y,
            (threshold - 1).max(1),
            AgastType::Agast5_8,
        );
        if s < threshold {
            0
        } else {
            s
        }
    }

    /// `BriskLayer::getAgastScore(float, float, threshold)` at unit scale —
    /// bilinear over the four surrounding integer scores, truncated to a byte.
    fn score_at(&mut self, xf: f32, yf: f32, threshold: i32) -> i32 {
        let x = xf as i32;
        let y = yf as i32;
        let rx1 = xf - x as f32;
        let rx = 1.0 - rx1;
        let ry1 = yf - y as f32;
        let ry = 1.0 - ry1;
        let v = rx * ry * self.score(x, y, threshold) as f32
            + rx1 * ry * self.score(x + 1, y, threshold) as f32
            + rx * ry1 * self.score(x, y + 1, threshold) as f32
            + rx1 * ry1 * self.score(x + 1, y + 1, threshold) as f32;
        i32::from(v as u8)
    }
}

/// `BriskScaleSpace`.
struct Pyramid {
    layers: Vec<Layer>,
}

/// `subpixel2D` — fit a quadratic to a 3x3 of scores and return its peak.
///
/// Transcribed in integer arithmetic, including the degenerate paths: a
/// singular Hessian returns the centre, and a saddle or minimum falls back to
/// whichever *corner* of the 3x3 is highest rather than to the fitted point.
#[allow(clippy::too_many_arguments)]
fn subpixel_2d(
    s_0_0: i32,
    s_0_1: i32,
    s_0_2: i32,
    s_1_0: i32,
    s_1_1: i32,
    s_1_2: i32,
    s_2_0: i32,
    s_2_1: i32,
    s_2_2: i32,
) -> (f32, f32, f32) {
    let tmp1 = s_0_0 + s_0_2 - 2 * s_1_1 + s_2_0 + s_2_2;
    let coeff1 = 3 * (tmp1 + s_0_1 - ((s_1_0 + s_1_2) << 1) + s_2_1);
    let coeff2 = 3 * (tmp1 - ((s_0_1 + s_2_1) << 1) + s_1_0 + s_1_2);
    let tmp2 = s_0_2 - s_2_0;
    let tmp3 = s_0_0 + tmp2 - s_2_2;
    let tmp4 = tmp3 - 2 * tmp2;
    let coeff3 = -3 * (tmp3 + s_0_1 - s_2_1);
    let coeff4 = -3 * (tmp4 + s_1_0 - s_1_2);
    let coeff5 = (s_0_0 - s_0_2 - s_2_0 + s_2_2) * 4;
    let coeff6 =
        -(s_0_0 + s_0_2 - ((s_1_0 + s_0_1 + s_1_2 + s_2_1) * 2) - 5 * s_1_1 + s_2_0 + s_2_2) * 2;

    let h_det = 4 * coeff1 * coeff2 - coeff5 * coeff5;
    if h_det == 0 {
        return (0.0, 0.0, coeff6 as f32 / 18.0);
    }
    if !(h_det > 0 && coeff1 < 0) {
        // Not a maximum: take the best corner of the cell.
        let mut tmp_max = coeff3 + coeff4 + coeff5;
        let (mut dx, mut dy) = (1.0f32, 1.0f32);
        for (t, x, y) in [
            (-coeff3 + coeff4 - coeff5, -1.0, 1.0),
            (coeff3 - coeff4 - coeff5, 1.0, -1.0),
            (-coeff3 - coeff4 + coeff5, -1.0, -1.0),
        ] {
            if t > tmp_max {
                tmp_max = t;
                dx = x;
                dy = y;
            }
        }
        return (dx, dy, (tmp_max + coeff1 + coeff2 + coeff6) as f32 / 18.0);
    }

    let mut delta_x = (2 * coeff2 * coeff3 - coeff4 * coeff5) as f32 / -h_det as f32;
    let mut delta_y = (2 * coeff1 * coeff4 - coeff3 * coeff5) as f32 / -h_det as f32;
    let (tx, tx_) = (delta_x > 1.0, delta_x < -1.0);
    let (ty, ty_) = (delta_y > 1.0, delta_y < -1.0);

    if tx || tx_ || ty || ty_ {
        // The peak is outside the cell. Slide along each violated edge and keep
        // whichever constrained optimum is higher.
        let (mut dx1, mut dy1, mut dx2, mut dy2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        if tx {
            dx1 = 1.0;
            dy1 = (-((coeff4 + coeff5) as f32) / (2 * coeff2) as f32).clamp(-1.0, 1.0);
        } else if tx_ {
            dx1 = -1.0;
            dy1 = (-((coeff4 - coeff5) as f32) / (2 * coeff2) as f32).clamp(-1.0, 1.0);
        }
        if ty {
            dy2 = 1.0;
            dx2 = (-((coeff3 + coeff5) as f32) / (2 * coeff1) as f32).clamp(-1.0, 1.0);
        } else if ty_ {
            dy2 = -1.0;
            dx2 = (-((coeff3 - coeff5) as f32) / (2 * coeff1) as f32).clamp(-1.0, 1.0);
        }
        let eval = |dx: f32, dy: f32| {
            (coeff1 as f32 * dx * dx
                + coeff2 as f32 * dy * dy
                + coeff3 as f32 * dx
                + coeff4 as f32 * dy
                + coeff5 as f32 * dx * dy
                + coeff6 as f32)
                / 18.0
        };
        let (max1, max2) = (eval(dx1, dy1), eval(dx2, dy2));
        return if max1 > max2 { (dx1, dy1, max1) } else { (dx2, dy2, max2) };
    }

    let max = (coeff1 as f32 * delta_x * delta_x
        + coeff2 as f32 * delta_y * delta_y
        + coeff3 as f32 * delta_x
        + coeff4 as f32 * delta_y
        + coeff5 as f32 * delta_x * delta_y
        + coeff6 as f32)
        / 18.0;
    // Guard against -0.0 reaching the caller's arithmetic.
    if delta_x == 0.0 {
        delta_x = 0.0;
    }
    if delta_y == 0.0 {
        delta_y = 0.0;
    }
    (delta_x, delta_y, max)
}

/// Which of the three [`refine_1d`] spacings a layer needs.
#[derive(Clone, Copy)]
enum Spacing {
    /// Octave above the first: samples at 0.75, 1, 1.5.
    Octave,
    /// Intra-octave: samples at 2/3, 1, 4/3.
    Intra,
    /// The bottom octave, whose "below" is the 5-8 mask: 0.7, 1, 1.5.
    Bottom,
}

/// `refine1D`, `refine1D_1` and `refine1D_2` — a parabola through three layers.
///
/// The three differ only in the spacing of the samples, and each quantises its
/// inputs to 1/1024 first so the coefficients stay integers. If the parabola
/// opens upwards there is no interior maximum, so the best of the three samples
/// is returned instead.
fn refine_1d(s_05: f32, s0: f32, s05: f32, spacing: Spacing) -> (f32, f32) {
    let i_05 = (1024.0 * f64::from(s_05) + 0.5) as i32;
    let i0 = (1024.0 * f64::from(s0) + 0.5) as i32;
    let i05 = (1024.0 * f64::from(s05) + 0.5) as i32;

    let (a, b, c, lo, hi, div) = match spacing {
        Spacing::Octave => (
            16 * i_05 - 24 * i0 + 8 * i05,
            -40 * i_05 + 54 * i0 - 14 * i05,
            24 * i_05 - 27 * i0 + 6 * i05,
            0.75f32,
            1.5f32,
            3072.0f32,
        ),
        Spacing::Intra => (
            9 * i_05 - 18 * i0 + 9 * i05,
            -21 * i_05 + 36 * i0 - 15 * i05,
            12 * i_05 - 16 * i0 + 6 * i05,
            2.0 / 3.0,
            4.0 / 3.0,
            2048.0,
        ),
        Spacing::Bottom => (
            2 * i_05 - 4 * i0 + 2 * i05,
            -5 * i_05 + 8 * i0 - 3 * i05,
            3 * i_05 - 3 * i0 + i05,
            0.7,
            1.5,
            1024.0,
        ),
    };

    if a >= 0 {
        if s0 >= s_05 && s0 >= s05 {
            return (1.0, s0);
        }
        if s_05 >= s0 && s_05 >= s05 {
            return (lo, s_05);
        }
        if s05 >= s0 && s05 >= s_05 {
            return (hi, s05);
        }
    }
    let ret = (-(b as f32) / (2 * a) as f32).clamp(lo, hi);
    let max = (c as f32 + a as f32 * ret * ret + b as f32 * ret) / div;
    (ret, max)
}

impl Pyramid {
    /// `BriskScaleSpace::constructPyramid` — alternating 2/3 and 1/2 sampling.
    fn build<S: MatLike + ?Sized>(image: &S, octaves: i32) -> Result<Self> {
        let (h, w) = (image.rows() as usize, image.cols() as usize);
        let mut base = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                base[y * w + x] = image.at_f64(y as i32, x as i32, 0)? as u8;
            }
        }
        let n_layers = if octaves == 0 { 1 } else { 2 * octaves };
        let mut layers = vec![Layer::from_image(base, w, h, 1.0, 0.0)];
        if n_layers > 1 {
            layers.push(resample(&layers[0], 2, 3)?);
        }
        for i in (2..n_layers).step_by(2) {
            let a = resample(&layers[i as usize - 2], 1, 2)?;
            layers.push(a);
            let b = resample(&layers[i as usize - 1], 1, 2)?;
            layers.push(b);
        }
        Ok(Pyramid { layers })
    }

    /// `isMax2D` — maximal over 3x3, with upstream's smoothed tie-break.
    fn is_max_2d(&self, layer: usize, x: i32, y: i32) -> bool {
        let l = &self.layers[layer];
        if x < 1 || y < 1 || x + 1 >= l.w as i32 || y + 1 >= l.h as i32 {
            return false;
        }
        let at = |dy: i32, dx: i32| {
            i32::from(l.scores[(y + dy) as usize * l.w + (x + dx) as usize])
        };
        let center = at(0, 0);
        let n = [
            (0i32, -1i32),
            (0, 1),
            (-1, 0),
            (1, 0),
            (1, -1),
            (1, 1),
            (-1, 1),
            (-1, -1),
        ];
        for (dy, dx) in n {
            if center < at(dy, dx) {
                return false;
            }
        }
        // Ties are settled on a 3x3 smoothed score, not on the raw one, so a
        // plateau resolves to its centre of mass rather than to scan order.
        let ties: Vec<(i32, i32)> = n.iter().copied().filter(|&(dy, dx)| center == at(dy, dx)).collect();
        if ties.is_empty() {
            return true;
        }
        let smoothed = |cy: i32, cx: i32| -> i32 {
            let g = |dy: i32, dx: i32| {
                i32::from(l.scores[(cy + dy) as usize * l.w + (cx + dx) as usize])
            };
            4 * g(0, 0)
                + 2 * (g(0, -1) + g(0, 1) + g(-1, 0) + g(1, 0))
                + g(-1, -1)
                + g(-1, 1)
                + g(1, -1)
                + g(1, 1)
        };
        let sc = smoothed(y, x);
        for (dy, dx) in ties {
            let (cy, cx) = (y + dy, x + dx);
            if cy < 1 || cx < 1 || cy + 1 >= l.h as i32 || cx + 1 >= l.w as i32 {
                continue;
            }
            if smoothed(cy, cx) > sc {
                return false;
            }
        }
        true
    }

    /// A 3x3 of `getAgastScore` around an integer point.
    fn cell(&mut self, layer: usize, x: i32, y: i32) -> [i32; 9] {
        let l = &mut self.layers[layer];
        [
            l.score(x - 1, y - 1, 1),
            l.score(x - 1, y, 1),
            l.score(x - 1, y + 1, 1),
            l.score(x, y - 1, 1),
            l.score(x, y, 1),
            l.score(x, y + 1, 1),
            l.score(x + 1, y - 1, 1),
            l.score(x + 1, y, 1),
            l.score(x + 1, y + 1, 1),
        ]
    }

    /// `getScoreMaxAbove` / `getScoreMaxBelow`, which differ only in the patch
    /// they scan and the mapping back into this layer's frame.
    ///
    /// Returns `None` the moment any score in the patch exceeds `threshold` —
    /// that means the candidate is not maximal across scale, and upstream bails
    /// out rather than finishing the scan.
    fn score_max(
        &mut self,
        layer: usize,
        x_layer: i32,
        y_layer: i32,
        threshold: i32,
        above: bool,
    ) -> Option<(f32, f32, f32)> {
        let other = if above { layer + 1 } else { layer.checked_sub(1)? };
        if other >= self.layers.len() {
            return None;
        }
        let even = layer % 2 == 0;
        // The patch is the pre-image of this pixel in the neighbouring layer.
        let (num, half, den) = match (above, even) {
            (true, true) => (4.0f32, 2.0f32, 6.0f32),
            (true, false) => (6.0, 3.0, 8.0),
            (false, true) => (8.0, 4.0, 6.0),
            (false, false) => (6.0, 3.0, 4.0),
        };
        let sign = if above { -1.0f32 } else { 1.0 };
        let x_1 = (num * x_layer as f32 + sign - half) / den;
        let x1 = (num * x_layer as f32 + sign + half) / den;
        let y_1 = (num * y_layer as f32 + sign - half) / den;
        let y1 = (num * y_layer as f32 + sign + half) / den;

        let l = &mut self.layers[other];
        let mut max_x = x_1 as i32 + 1;
        let mut max_y = y_1 as i32 + 1;
        let mut maxval = l.score_at(x_1, y_1, 1) as f32;
        if maxval > threshold as f32 {
            return None;
        }
        // Leading edge and interior: any of these exceeding the threshold kills
        // the candidate. The trailing edge below does not get that test — an
        // asymmetry in upstream, preserved.
        for x in x_1 as i32 + 1..=x1 as i32 {
            let t = l.score_at(x as f32, y_1, 1) as f32;
            if t > threshold as f32 {
                return None;
            }
            if t > maxval {
                maxval = t;
                max_x = x;
            }
        }
        let t = l.score_at(x1, y_1, 1) as f32;
        if t > threshold as f32 {
            return None;
        }
        if t > maxval {
            maxval = t;
            max_x = x1 as i32;
        }
        for y in y_1 as i32 + 1..=y1 as i32 {
            let t = l.score_at(x_1, y as f32, 1) as f32;
            if t > threshold as f32 {
                return None;
            }
            if t > maxval {
                maxval = t;
                max_x = x_1 as i32 + 1;
                max_y = y;
            }
            for x in x_1 as i32 + 1..=x1 as i32 {
                let t = l.score_at(x as f32, y as f32, 1) as f32;
                if t > threshold as f32 {
                    return None;
                }
                if t > maxval {
                    maxval = t;
                    max_x = x;
                    max_y = y;
                }
            }
            let t = l.score_at(x1, y as f32, 1) as f32;
            if t > threshold as f32 {
                return None;
            }
            if t > maxval {
                maxval = t;
                max_x = x1 as i32;
                max_y = y;
            }
        }
        let t = l.score_at(x_1, y1, 1) as f32;
        if t > maxval {
            maxval = t;
            max_x = x_1 as i32 + 1;
            max_y = y1 as i32;
        }
        for x in x_1 as i32 + 1..=x1 as i32 {
            let t = l.score_at(x as f32, y1, 1) as f32;
            if t > maxval {
                maxval = t;
                max_x = x;
                max_y = y1 as i32;
            }
        }
        let t = l.score_at(x1, y1, 1) as f32;
        if t > maxval {
            maxval = t;
            max_x = x1 as i32;
            max_y = y1 as i32;
        }

        let c = self.cell(other, max_x, max_y);
        let (dx_1, dy_1, refined_max) =
            subpixel_2d(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]);
        let real_x = max_x as f32 + dx_1;
        let real_y = max_y as f32 + dy_1;

        // Map the refined position back into this layer's coordinates.
        let (mut dx, mut dy) = if above {
            let (m, d) = if even { (6.0f32, 4.0f32) } else { (8.0, 6.0) };
            ((real_x * m + 1.0) / d - x_layer as f32, (real_y * m + 1.0) / d - y_layer as f32)
        } else {
            let (m, d) = if even { (6.0f32, 8.0f32) } else { (4.0, 6.0) };
            ((real_x * m - 1.0) / d - x_layer as f32, (real_y * m - 1.0) / d - y_layer as f32)
        };
        let mut returnrefined = true;
        for v in [&mut dx, &mut dy] {
            if *v > 1.0 {
                *v = 1.0;
                returnrefined = false;
            } else if *v < -1.0 {
                *v = -1.0;
                returnrefined = false;
            }
        }
        Some((if returnrefined { refined_max.max(maxval) } else { maxval }, dx, dy))
    }

    /// `refine3D` — the full 2D-plus-scale refinement of one candidate.
    fn refine_3d(&mut self, layer: usize, x_layer: i32, y_layer: i32) -> Option<(f32, f32, f32, f32)> {
        let center = self.layers[layer].score(x_layer, y_layer, 1);
        let (max_above, dx_above, dy_above) =
            self.score_max(layer, x_layer, y_layer, center, true)?;

        let (scale, layer_scale, layer_offset);
        let (max, x, y);

        if layer % 2 == 0 {
            let (max_below, dx_below, dy_below) = if layer == 0 {
                // Below the bottom octave there is no layer, so upstream
                // substitutes the coarser 5-8 AGAST mask on the same image —
                // a smaller circle standing in for a smaller scale.
                let l = &self.layers[0];
                let c = [
                    l.score_5_8(x_layer - 1, y_layer - 1, 1),
                    l.score_5_8(x_layer - 1, y_layer, 1),
                    l.score_5_8(x_layer - 1, y_layer + 1, 1),
                    l.score_5_8(x_layer, y_layer - 1, 1),
                    l.score_5_8(x_layer, y_layer, 1),
                    l.score_5_8(x_layer, y_layer + 1, 1),
                    l.score_5_8(x_layer + 1, y_layer - 1, 1),
                    l.score_5_8(x_layer + 1, y_layer, 1),
                    l.score_5_8(x_layer + 1, y_layer + 1, 1),
                ];
                let mb = c.iter().copied().max().unwrap_or(0) as f32;
                let (dx, dy, _) = subpixel_2d(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]);
                (mb, dx, dy)
            } else {
                self.score_max(layer, x_layer, y_layer, center, false)?
            };

            let c = self.cell(layer, x_layer, y_layer);
            let (dx_layer, dy_layer, max_layer) =
                subpixel_2d(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]);
            let spacing = if layer == 0 { Spacing::Bottom } else { Spacing::Octave };
            let (s, m) = refine_1d(max_below, (center as f32).max(max_layer), max_above, spacing);
            scale = s;
            max = m;
            layer_scale = self.layers[layer].scale;
            layer_offset = self.layers[layer].offset;

            if scale > 1.0 {
                let r0 = (1.5 - scale) / 0.5;
                let r1 = 1.0 - r0;
                x = (r0 * dx_layer + r1 * dx_above + x_layer as f32) * layer_scale + layer_offset;
                y = (r0 * dy_layer + r1 * dy_above + y_layer as f32) * layer_scale + layer_offset;
            } else if layer == 0 {
                // Layer 0 is already in the original frame, so no rescaling.
                let r0 = (scale - 0.5) / 0.5;
                let r_1 = 1.0 - r0;
                x = r0 * dx_layer + r_1 * dx_below + x_layer as f32;
                y = r0 * dy_layer + r_1 * dy_below + y_layer as f32;
            } else {
                let r0 = (scale - 0.75) / 0.25;
                let r_1 = 1.0 - r0;
                x = (r0 * dx_layer + r_1 * dx_below + x_layer as f32) * layer_scale + layer_offset;
                y = (r0 * dy_layer + r_1 * dy_below + y_layer as f32) * layer_scale + layer_offset;
            }
        } else {
            let (max_below, dx_below, dy_below) =
                self.score_max(layer, x_layer, y_layer, center, false)?;
            let c = self.cell(layer, x_layer, y_layer);
            let (dx_layer, dy_layer, max_layer) =
                subpixel_2d(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]);
            let (s, m) =
                refine_1d(max_below, (center as f32).max(max_layer), max_above, Spacing::Intra);
            scale = s;
            max = m;
            layer_scale = self.layers[layer].scale;
            layer_offset = self.layers[layer].offset;
            if scale > 1.0 {
                let r0 = 4.0 - scale * 3.0;
                let r1 = 1.0 - r0;
                x = (r0 * dx_layer + r1 * dx_above + x_layer as f32) * layer_scale + layer_offset;
                y = (r0 * dy_layer + r1 * dy_above + y_layer as f32) * layer_scale + layer_offset;
            } else {
                let r0 = scale * 3.0 - 2.0;
                let r_1 = 1.0 - r0;
                x = (r0 * dx_layer + r_1 * dx_below + x_layer as f32) * layer_scale + layer_offset;
                y = (r0 * dy_layer + r_1 * dy_below + y_layer as f32) * layer_scale + layer_offset;
            }
        }
        Some((x, y, scale * layer_scale, max))
    }
}

/// `resize(..., INTER_AREA)` by `num/den` — 1/2 or 2/3.
fn resample(src: &Layer, num: usize, den: usize) -> Result<Layer> {
    let w = src.w / den * num;
    let h = src.h / den * num;
    let mut m = cv_core::Mat::zeros(src.h as i32, src.w as i32, cv_core::types::CV_8UC1)?;
    for y in 0..src.h {
        for x in 0..src.w {
            cv_core::arithm::set_channel_generic(
                &mut m,
                y as i32,
                x as i32,
                0,
                f64::from(src.img[y * src.w + x]),
            )?;
        }
    }
    let out = cv_imgproc::resize::resize(
        &m,
        cv_core::Size::new(w as i32, h as i32),
        0.0,
        0.0,
        cv_imgproc::resize::Interpolation::Area,
    )?;
    let mut img = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            img[y * w + x] = out.at_f64(y as i32, x as i32, 0)? as u8;
        }
    }
    let scale = src.scale * den as f32 / num as f32;
    Ok(Layer::from_image(img, w, h, scale, 0.5 * scale - 0.5))
}

/// `cv::BRISK::detect(image, keypoints)`.
///
/// `octaves` follows upstream: 0 means a single layer at the original
/// resolution, and `n > 0` means `2n` layers alternating octave and
/// intra-octave.
pub fn brisk_detect<S: MatLike + ?Sized>(
    image: &S,
    threshold: i32,
    octaves: i32,
    pattern_scale: f32,
) -> Result<Vec<KeyPoint>> {
    cv_assert!(image.channels() == 1, "BRISK requires a single-channel image");
    cv_assert!(threshold > 0, "the threshold must be positive");
    cv_assert!(octaves >= 0, "octaves must not be negative");

    let mut py = Pyramid::build(image, octaves)?;
    let safe_threshold = (threshold as f32 * SAFETY_FACTOR) as i32;
    let n_layers = py.layers.len();

    // Fill each layer's score surface, and collect the candidates from it. Both
    // come from the same pass so the cache is warm and consistent.
    let mut candidates: Vec<Vec<(i32, i32)>> = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let (w, h) = (py.layers[i].w as i32, py.layers[i].h as i32);
        let mut here = Vec::new();
        for y in 3..h - 3 {
            for x in 3..w - 3 {
                if py.layers[i].score(x, y, safe_threshold) > 0 {
                    here.push((x, y));
                }
            }
        }
        candidates.push(here);
    }

    let mut keypoints = Vec::new();
    if n_layers == 1 {
        for &(x, y) in &candidates[0] {
            if !py.is_max_2d(0, x, y) {
                continue;
            }
            let c = py.cell(0, x, y);
            let (dx, dy, max) = subpixel_2d(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]);
            keypoints.push(KeyPoint {
                pt: (x as f32 + dx, y as f32 + dy),
                size: BASIC_SIZE,
                angle: -1.0,
                response: max,
                octave: 0,
                class_id: -1,
            });
        }
        return Ok(prune(image, keypoints, pattern_scale));
    }

    for (i, pts) in candidates.into_iter().enumerate() {
        for (x, y) in pts {
            if !py.is_max_2d(i, x, y) {
                continue;
            }
            if i == n_layers - 1 {
                // The top layer has nothing above it, so it only has to beat
                // the layer below, and its scale is not refined.
                let center = py.layers[i].score(x, y, safe_threshold);
                if py.score_max(i, x, y, center, false).is_none() {
                    continue;
                }
                let c = py.cell(i, x, y);
                let (dx, dy, max) =
                    subpixel_2d(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]);
                let (s, o) = (py.layers[i].scale, py.layers[i].offset);
                keypoints.push(KeyPoint {
                    pt: ((x as f32 + dx) * s + o, (y as f32 + dy) * s + o),
                    size: BASIC_SIZE * s,
                    angle: -1.0,
                    response: max,
                    octave: i as i32,
                    class_id: -1,
                });
            } else if let Some((rx, ry, scale, score)) = py.refine_3d(i, x, y) {
                if score > threshold as f32 {
                    keypoints.push(KeyPoint {
                        pt: (rx, ry),
                        size: BASIC_SIZE * scale,
                        angle: -1.0,
                        response: score,
                        octave: i as i32,
                        class_id: -1,
                    });
                }
            }
        }
    }
    Ok(prune(image, keypoints, pattern_scale))
}

/// Drop keypoints whose sampling pattern would leave the image, as upstream
/// does before it looks at whether descriptors were requested.
fn prune<S: MatLike + ?Sized>(
    image: &S,
    keypoints: Vec<KeyPoint>,
    pattern_scale: f32,
) -> Vec<KeyPoint> {
    let pattern = crate::brisk_pattern::BriskPattern::new(pattern_scale);
    let (cols, rows) = (image.cols(), image.rows());
    keypoints.into_iter().filter(|k| pattern.fits(k, cols, rows)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::Mat;


    /// Pinned against `cv2.BRISK_create().detect()` on the reference texture,
    /// sorted by x then y — cv2 returns exactly 165 keypoints here.
    ///
    /// The tolerance is 0.01 px, which 156 of the 165 meet; the remaining nine
    /// agree to 0.6 px with identical `size` and `response`. Only the leading
    /// six are pinned individually, but the **count** is pinned exactly, and
    /// that is what catches a regression in any of the three filters.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn detector_matches_cv2() {
        const CV2: [(f32, f32, f32, f32); 6] = [
            (21.0, 67.0, 12.0, 38.11111068725586),
            (21.069469451904297, 27.100061416625977, 12.94285774230957, 34.43214416503906),
            (21.98082733154297, 20.931068420410156, 12.0, 32.0),
            (23.545339584350586, 75.55000305175781, 13.881261825561523, 40.13874816894531),
            (25.97915267944336, 26.60466766357422, 12.55886459350586, 31.45108413696289),
            (25.995193481445312, 109.49494934082031, 13.363636016845703, 36.85227584838867),
        ];
        let img = crate::akaze::tests::texture(160, 160);
        let mut kps = brisk_detect(&img, 30, 3, 1.0).unwrap();
        assert_eq!(kps.len(), 165, "cv2 finds 165 keypoints here");
        assert!(kps.iter().all(|k| (0..3).contains(&k.octave)), "3 octaves");

        kps.sort_by(|a, b| a.pt.0.total_cmp(&b.pt.0).then(a.pt.1.total_cmp(&b.pt.1)));
        for (k, &(x, y, size, response)) in kps.iter().zip(&CV2) {
            assert!(
                (k.pt.0 - x).abs() < 0.01 && (k.pt.1 - y).abs() < 0.01,
                "position ({}, {}) against cv2's ({x}, {y})",
                k.pt.0,
                k.pt.1
            );
            assert!((k.size - size).abs() < 1e-4, "size {} against {size}", k.size);
            assert!(
                (k.response - response).abs() < 1e-3,
                "response {} against {response}",
                k.response
            );
        }
    }

    #[test]
    fn the_pyramid_alternates_halves_and_two_thirds() {
        let img = crate::akaze::tests::texture(180, 180);
        let py = Pyramid::build(&img, 3).unwrap();
        assert_eq!(py.layers.len(), 6, "3 octaves means 6 layers");
        let scales: Vec<f32> = py.layers.iter().map(|l| l.scale).collect();
        assert_eq!(scales, vec![1.0, 1.5, 2.0, 3.0, 4.0, 6.0]);
        // Layer 0 is the original and needs no offset; every other layer's
        // pixel centres are displaced by half its own scale.
        assert_eq!(py.layers[0].offset, 0.0);
        for l in &py.layers[1..] {
            assert!((l.offset - (0.5 * l.scale - 0.5)).abs() < 1e-6);
        }
        assert_eq!((py.layers[1].w, py.layers[1].h), (120, 120));
        assert_eq!((py.layers[2].w, py.layers[2].h), (90, 90));
    }

    /// A parabola through three equal samples has no interior maximum, so the
    /// centre must be returned unchanged rather than a fitted point.
    #[test]
    fn refine_1d_returns_the_centre_for_a_flat_triple() {
        for spacing in [Spacing::Octave, Spacing::Intra, Spacing::Bottom] {
            let (scale, max) = refine_1d(10.0, 10.0, 10.0, spacing);
            assert_eq!((scale, max), (1.0, 10.0));
        }
        // And it leans toward the larger neighbour when there is one.
        let (scale, _) = refine_1d(5.0, 10.0, 20.0, Spacing::Octave);
        assert!(scale > 1.0, "a rising triple must refine upward, got {scale}");
        let (scale, _) = refine_1d(20.0, 10.0, 5.0, Spacing::Octave);
        assert!(scale < 1.0, "a falling triple must refine downward, got {scale}");
    }

    /// The returned score is the **fitted** peak, not the centre sample: it is
    /// a weighted combination of all nine values, so on a sharp peak it reads
    /// lower than the centre. That is upstream's behaviour and what the
    /// scale-space comparisons are calibrated against, so it is pinned here
    /// rather than "corrected".
    #[test]
    fn subpixel_2d_centres_a_symmetric_peak() {
        let cell = [1, 2, 1, 2, 9, 2, 1, 2, 1];
        let (dx, dy, max) = subpixel_2d(
            cell[0], cell[1], cell[2], cell[3], cell[4], cell[5], cell[6], cell[7], cell[8],
        );
        assert!(dx.abs() < 1e-6 && dy.abs() < 1e-6, "peak moved to ({dx}, {dy})");
        // coeff6 / 18 with the peak at the centre.
        assert!((max - 114.0 / 18.0).abs() < 1e-5, "fitted peak {max}");
        assert!(
            max > cell[1] as f32 && max < cell[4] as f32,
            "the fit should sit between the ring and the centre, got {max}"
        );

        // Shifting the mass one column right must move the peak that way.
        let (dx, _, _) = subpixel_2d(1, 2, 1, 2, 9, 2, 2, 5, 2);
        assert!(dx > 0.0, "peak should move toward the heavier side, got {dx}");
    }

    #[test]
    fn it_validates_its_input() {
        let colour = Mat::zeros(32, 32, cv_core::types::CV_8UC3).unwrap();
        assert!(brisk_detect(&colour, 30, 3, 1.0).is_err());
        let img = crate::akaze::tests::texture(64, 64);
        assert!(brisk_detect(&img, 0, 3, 1.0).is_err());
        assert!(brisk_detect(&img, 30, -1, 1.0).is_err());
    }

    #[test]
    fn a_flat_image_has_no_corners() {
        let flat = Mat::from_vec(96, 96, 1, &[128u8; 9216]).unwrap();
        assert!(brisk_detect(&flat, 30, 3, 1.0).unwrap().is_empty());
    }
}
