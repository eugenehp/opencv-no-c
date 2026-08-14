//! The two gradient-aware Bayer demosaicers
//! (`modules/imgproc/src/demosaicing.cpp`).
//!
//! [`crate::extra::demosaicing`] interpolates bilinearly, which is fast and
//! blurs every edge it crosses: averaging the two horizontal neighbours of a
//! photosite that sits on a vertical edge averages across the edge. Both
//! algorithms here fix that the same way — look at the local gradients first,
//! then interpolate only along the direction that varies least.
//!
//! - [`DemosaicMethod::Ea`], upstream's `COLOR_Bayer*2BGR_EA`, picks between
//!   the horizontal and vertical neighbour pairs by comparing two gradients.
//!   Cheap, and enough for most edges.
//! - [`DemosaicMethod::Vng`], upstream's `COLOR_Bayer*2BGR_VNG`, computes eight
//!   directional gradients over a 5x5 neighbourhood, keeps the ones below a
//!   threshold derived from the local minimum and maximum, and averages over
//!   just those. Slower, and it handles a corner — which has no single good
//!   direction — where the edge-aware method has to pick one.
//!
//! Both are noticeably better than bilinear on a colour edge and neither is
//! bit-exact against cv2: upstream's VNG in particular carries a hand-tuned
//! set of integer weights this reconstructs from the published algorithm.

use cv_core::saturate::cv_roundf;
use cv_core::types::{Depth, MatType};
use cv_core::{cv_assert, cv_error, Code, Mat, MatLike, MatLikeMut, Result};

use crate::extra::{BayerOutput, BayerPattern};

/// Which interpolation [`demosaicing_with`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DemosaicMethod {
    /// Bilinear — the plain `COLOR_Bayer*2BGR` codes.
    #[default]
    Bilinear,
    /// `COLOR_BayerBG2BGR_EA` / `COLOR_BayerBG2RGB_EA` and their
    /// `COLOR_BayerBGGR2BGR_EA` / `COLOR_BayerBGGR2RGB_EA` spellings, and the
    /// same for the other three phases: `COLOR_BayerGB2BGR_EA`,
    /// `COLOR_BayerGB2RGB_EA`, `COLOR_BayerGBRG2BGR_EA`,
    /// `COLOR_BayerGBRG2RGB_EA`, `COLOR_BayerRG2BGR_EA`, `COLOR_BayerRG2RGB_EA`,
    /// `COLOR_BayerRGGB2BGR_EA`, `COLOR_BayerRGGB2RGB_EA`,
    /// `COLOR_BayerGR2BGR_EA`, `COLOR_BayerGR2RGB_EA`,
    /// `COLOR_BayerGRBG2BGR_EA`, `COLOR_BayerGRBG2RGB_EA`.
    Ea,
    /// `COLOR_BayerBG2BGR_VNG` / `COLOR_BayerBG2RGB_VNG` and their
    /// `COLOR_BayerBGGR2BGR_VNG` / `COLOR_BayerBGGR2RGB_VNG` spellings, and the
    /// same for the other three phases: `COLOR_BayerGB2BGR_VNG`,
    /// `COLOR_BayerGB2RGB_VNG`, `COLOR_BayerGBRG2BGR_VNG`,
    /// `COLOR_BayerGBRG2RGB_VNG`, `COLOR_BayerRG2BGR_VNG`,
    /// `COLOR_BayerRG2RGB_VNG`, `COLOR_BayerRGGB2BGR_VNG`,
    /// `COLOR_BayerRGGB2RGB_VNG`, `COLOR_BayerGR2BGR_VNG`,
    /// `COLOR_BayerGR2RGB_VNG`, `COLOR_BayerGRBG2BGR_VNG`,
    /// `COLOR_BayerGRBG2RGB_VNG`.
    Vng,
}

/// `cv::demosaicing(src, dst, code, dcn)` for every method and output layout.
///
/// `CV_8UC1` in. The gradient-aware methods produce three or four channels
/// only; a grey output from them would be demosaic-then-grey, which is what
/// [`crate::extra::demosaicing_to`] already does more cheaply.
pub fn demosaicing_with<S: MatLike + ?Sized>(
    src: &S,
    pattern: BayerPattern,
    output: BayerOutput,
    method: DemosaicMethod,
) -> Result<Mat> {
    if method == DemosaicMethod::Bilinear {
        return crate::extra::demosaicing_to(src, pattern, output);
    }
    cv_assert!(src.channels() == 1, "demosaicing takes a single-channel mosaic");
    if src.depth() != Depth::U8 {
        return cv_error!(Code::StsUnsupportedFormat, "demosaicing requires CV_8U");
    }
    if output == BayerOutput::Gray {
        return cv_error!(
            Code::StsBadArg,
            "the gradient-aware codes have no grey form; demosaic then convert"
        );
    }
    let (h, w) = (src.rows(), src.cols());
    cv_assert!(h >= 5 && w >= 5, "the gradient-aware methods need at least a 5x5 mosaic");

    let raw: Vec<i32> = {
        let mut v = Vec::with_capacity((h * w) as usize);
        for y in 0..h {
            for x in 0..w {
                v.push(src.at_f64(y, x, 0)? as i32);
            }
        }
        v
    };

    match method {
        DemosaicMethod::Ea => edge_aware(&raw, h, w, pattern, output),
        _ => variable_gradients(&raw, h, w, pattern, output),
    }
}

/// How many channels `output` produces.
///
/// `BayerOutput::channels` is private to `extra`, and re-deriving it here keeps
/// this module from depending on that.
fn channels(output: BayerOutput) -> i32 {
    if matches!(output, BayerOutput::Bgra | BayerOutput::Rgba) {
        4
    } else {
        3
    }
}

/// Write one BGR triple into `dst` in the layout `output` asks for.
fn put(dst: &mut Mat, y: i32, x: i32, bgr: [i32; 3], output: BayerOutput) -> Result<()> {
    let red_first = matches!(output, BayerOutput::Rgb | BayerOutput::Rgba);
    let px = dst.pixel_mut::<u8>(y, x)?;
    let ordered = if red_first { [bgr[2], bgr[1], bgr[0]] } else { bgr };
    for (o, v) in px.iter_mut().zip(ordered.iter()) {
        *o = (*v).clamp(0, 255) as u8;
    }
    if channels(output) == 4 {
        px[3] = 255;
    }
    Ok(())
}

/// Copy the outer frame in from its neighbour.
///
/// Upstream interpolates rows `1..h-1` and columns `1..w-1` and then replicates,
/// columns first so a corner ends up holding the diagonal interior pixel.
fn replicate_frame(dst: &mut Mat, h: i32, w: i32, cn: usize) -> Result<()> {
    for y in 1..h - 1 {
        for (to, from) in [(0, 1), (w - 1, w - 2)] {
            for c in 0..cn {
                let v = dst.pixel::<u8>(y, from)?[c];
                dst.pixel_mut::<u8>(y, to)?[c] = v;
            }
        }
    }
    for (to, from) in [(0, 1), (h - 1, h - 2)] {
        let row = dst.row_bytes(from)?.to_vec();
        dst.row_bytes_mut(to)?.copy_from_slice(&row);
    }
    Ok(())
}

/// `cv::demosaicing` with a `COLOR_Bayer*2BGR_EA` code.
///
/// A transcription of upstream's `Bayer2RGB_EdgeAware_T_Invoker`, and it is
/// simpler than the name suggests: only the **green** channel at a red or blue
/// site is edge-aware, and only in the crudest way — compare the horizontal
/// and vertical first differences and average along whichever is smaller.
/// Every other channel is the same bilinear average [`crate::extra::demosaicing_to`]
/// uses. In particular there is no colour-difference term and no Laplacian
/// correction; adding either is a better demosaicer and a different one.
fn edge_aware(
    raw: &[i32],
    h: i32,
    w: i32,
    pattern: BayerPattern,
    output: BayerOutput,
) -> Result<Mat> {
    let s = |y: i32, x: i32| -> i32 { raw[(y * w + x) as usize] };
    let cn = channels(output);
    let mut dst = Mat::zeros(h, w, MatType::new(Depth::U8, cn))?;

    for y in 1..h - 1 {
        // Which of red and blue sits on this row's non-green sites, and
        // whether the row starts on a green one. Upstream carries these as two
        // flags toggled per row; deriving them from the pattern is the same
        // thing and cannot drift out of step with the bilinear path.
        let start_with_green = colour_at(pattern, y, 1) == 1;
        let non_green_x = if start_with_green { 2 } else { 1 };
        let blue = colour_at(pattern, y, non_green_x) == 0;
        let (own, other) = if blue { (0usize, 2usize) } else { (2, 0) };

        // Green at a red or blue site: interpolate along the flatter axis.
        let green = |y: i32, x: i32| -> i32 {
            let sum = if (s(y, x - 1) - s(y, x + 1)).abs() > (s(y + 1, x) - s(y - 1, x)).abs() {
                s(y + 1, x) + s(y - 1, x) + 1
            } else {
                s(y, x - 1) + s(y, x + 1) + 1
            };
            sum >> 1
        };
        let diagonals = |y: i32, x: i32| -> i32 {
            (s(y - 1, x - 1) + s(y - 1, x + 1) + s(y + 1, x - 1) + s(y + 1, x + 1) + 2) >> 2
        };

        let mut x = 1;
        if start_with_green {
            // A green site with the non-green neighbours above and below in
            // one colour and left and right in the other.
            let mut px = [0i32; 3];
            // The vertical neighbours of a green site carry the *other*
            // colour: the row's own non-green colour is the one to its left
            // and right.
            px[other] = (s(y - 1, x) + s(y + 1, x) + 1) >> 1;
            px[1] = s(y, x);
            px[own] = (s(y, x - 1) + s(y, x + 1) + 1) >> 1;
            put(&mut dst, y, x, px, output)?;
            x += 1;
        }

        while x < w - 2 {
            let mut px = [0i32; 3];
            px[own] = s(y, x);
            px[1] = green(y, x);
            px[other] = diagonals(y, x);
            put(&mut dst, y, x, px, output)?;

            // The green site next door: plain bilinear both ways.
            let mut px = [0i32; 3];
            px[own] = (s(y, x) + s(y, x + 2) + 1) >> 1;
            px[1] = s(y, x + 1);
            px[other] = (s(y - 1, x + 1) + s(y + 1, x + 1) + 1) >> 1;
            put(&mut dst, y, x + 1, px, output)?;
            x += 2;
        }

        if x <= w - 2 {
            let mut px = [0i32; 3];
            px[other] = diagonals(y, x);
            px[1] = green(y, x);
            px[own] = s(y, x);
            put(&mut dst, y, x, px, output)?;
        }
    }

    replicate_frame(&mut dst, h, w, cn as usize)?;
    Ok(dst)
}


/// Which colour sits at `(y, x)`: 0 blue, 1 green, 2 red.
///
/// The same table [`crate::extra::demosaicing_to`] uses, restated here because
/// upstream's two-letter names describe the *second* row's second and first
/// positions and deriving it afresh is how red and blue end up swapped.
fn colour_at(pattern: BayerPattern, y: i32, x: i32) -> usize {
    let table: [[usize; 2]; 2] = match pattern {
        BayerPattern::Bg => [[2, 1], [1, 0]],
        BayerPattern::Gb => [[1, 2], [0, 1]],
        BayerPattern::Rg => [[0, 1], [1, 2]],
        BayerPattern::Gr => [[1, 0], [2, 1]],
    };
    table[(y.rem_euclid(2)) as usize][(x.rem_euclid(2)) as usize]
}

/// `cv::demosaicing` with a `COLOR_Bayer*2BGR_VNG` code.
///
/// A transcription of upstream's `Bayer2RGB_VNG_8u`. Eight directional
/// gradients are computed from seven precomputed planes over the 5x5
/// neighbourhood; every direction whose gradient falls below
/// `min + max(max/2, 1)` contributes, and the estimate is the mean of just
/// those. Where an edge-aware method must commit to horizontal or vertical,
/// this can use three directions at a corner and one along a thin line — which
/// is the whole point of "variable number of gradients".
///
/// The two colour channels are recovered from the *sum differences* rather than
/// from the channels themselves, which is what keeps a grey edge grey.
fn variable_gradients(
    raw: &[i32],
    h: i32,
    w: i32,
    pattern: BayerPattern,
    output: BayerOutput,
) -> Result<Mat> {
    // Upstream pads by two with reflect-101 so the 5x5 neighbourhood is always
    // in range, and computes every output pixel — unlike the bilinear and
    // edge-aware paths there is no replicated frame.
    let (ph, pw) = (h + 4, w + 4);
    let reflect = |i: i32, n: i32| -> i32 {
        let mut i = i;
        loop {
            if i < 0 {
                i = -i;
            } else if i >= n {
                i = 2 * (n - 1) - i;
            } else {
                return i;
            }
        }
    };
    let mut pad = vec![0i32; (ph * pw) as usize];
    for y in 0..ph {
        for x in 0..pw {
            pad[(y * pw + x) as usize] =
                raw[(reflect(y - 2, h) * w + reflect(x - 2, w)) as usize];
        }
    }
    let s = |y: i32, x: i32| -> i32 { pad[(y * pw + x) as usize] };

    // The seven planes upstream keeps in a three-row ring buffer. Computing
    // them for the whole image instead trades memory for the modular indexing;
    // the values are identical.
    const PLANES: usize = 7;
    let mut g = vec![0i32; PLANES * (ph * pw) as usize];
    let idx = |k: usize, y: i32, x: i32| -> usize { k * (ph * pw) as usize + (y * pw + x) as usize };
    for y in 1..ph - 1 {
        for x in 1..pw - 1 {
            let vert = (s(y - 1, x - 1) - s(y + 1, x - 1)).abs()
                + (s(y - 1, x) - s(y + 1, x)).abs() * 2
                + (s(y - 1, x + 1) - s(y + 1, x + 1)).abs();
            let horiz = (s(y - 1, x - 1) - s(y - 1, x + 1)).abs()
                + (s(y, x - 1) - s(y, x + 1)).abs() * 2
                + (s(y + 1, x - 1) - s(y + 1, x + 1)).abs();
            let anti = (s(y - 1, x + 1) - s(y + 1, x - 1)).abs() * 2;
            let main = (s(y - 1, x - 1) - s(y + 1, x + 1)).abs() * 2;
            g[idx(0, y, x)] = vert;
            g[idx(1, y, x)] = horiz;
            g[idx(2, y, x)] = anti;
            g[idx(3, y, x)] = main;
            g[idx(4, y, x)] =
                anti + (s(y - 1, x) - s(y, x - 1)).abs() + (s(y + 1, x) - s(y, x + 1)).abs();
            g[idx(5, y, x)] =
                main + (s(y - 1, x) - s(y, x + 1)).abs() + (s(y + 1, x) - s(y, x - 1)).abs();
            g[idx(6, y, x)] = (s(y - 1, x) + s(y, x - 1) + s(y, x + 1) + s(y + 1, x)) >> 1;
        }
    }

    let cn = channels(output);
    let mut dst = Mat::zeros(h, w, MatType::new(Depth::U8, cn))?;
    for y in 2..ph - 2 {
        for x in 2..pw - 2 {
            let (oy, ox) = (y - 2, x - 2);
            let green_cell = colour_at(pattern, oy, ox) == 1;

            let grad_n = g[idx(0, y - 1, x)] + g[idx(0, y, x)];
            let grad_s = g[idx(0, y, x)] + g[idx(0, y + 1, x)];
            let grad_w = g[idx(1, y, x - 1)] + g[idx(1, y, x)];
            let grad_e = g[idx(1, y, x)] + g[idx(1, y, x + 1)];

            // The four diagonals are built from different planes depending on
            // the cell: a green site has twice as many same-colour neighbours
            // along a diagonal, so it can afford a four-term sum.
            let (grad_ne, grad_sw, grad_nw, grad_se) = if green_cell {
                (
                    g[idx(2, y - 1, x)] + g[idx(2, y - 1, x + 1)] + g[idx(2, y, x)] + g[idx(2, y, x + 1)],
                    g[idx(2, y, x)] + g[idx(2, y, x - 1)] + g[idx(2, y + 1, x)] + g[idx(2, y + 1, x - 1)],
                    g[idx(3, y - 1, x)] + g[idx(3, y - 1, x - 1)] + g[idx(3, y, x)] + g[idx(3, y, x - 1)],
                    g[idx(3, y, x)] + g[idx(3, y, x + 1)] + g[idx(3, y + 1, x)] + g[idx(3, y + 1, x + 1)],
                )
            } else {
                (
                    g[idx(4, y - 1, x + 1)] + g[idx(4, y, x)],
                    g[idx(4, y, x)] + g[idx(4, y + 1, x - 1)],
                    g[idx(5, y - 1, x - 1)] + g[idx(5, y, x)],
                    g[idx(5, y, x)] + g[idx(5, y + 1, x + 1)],
                )
            };

            let all = [grad_n, grad_s, grad_w, grad_e, grad_ne, grad_sw, grad_nw, grad_se];
            let min_grad = *all.iter().min().unwrap();
            let max_grad = *all.iter().max().unwrap();
            let thr = min_grad + (max_grad / 2).max(1);

            // `rs` is the centre's own colour at a non-green site and the
            // horizontal neighbour's at a green one; `bs` the other diagonal
            // colour. Upstream calls them R and B throughout and routes them to
            // the right channel at the end.
            let (mut rs, mut gs, mut bs, mut ng) = (0i32, 0i32, 0i32, 0usize);
            if green_cell {
                if grad_n < thr {
                    rs += s(y - 2, x - 1) + s(y - 2, x + 1);
                    gs += s(y - 2, x) + s(y, x);
                    bs += s(y - 1, x) * 2;
                    ng += 1;
                }
                if grad_s < thr {
                    rs += s(y + 2, x - 1) + s(y + 2, x + 1);
                    gs += s(y + 2, x) + s(y, x);
                    bs += s(y + 1, x) * 2;
                    ng += 1;
                }
                if grad_w < thr {
                    rs += s(y, x - 1) * 2;
                    gs += s(y, x - 2) + s(y, x);
                    bs += s(y - 1, x - 2) + s(y + 1, x - 2);
                    ng += 1;
                }
                if grad_e < thr {
                    rs += s(y, x + 1) * 2;
                    gs += s(y, x + 2) + s(y, x);
                    bs += s(y - 1, x + 2) + s(y + 1, x + 2);
                    ng += 1;
                }
                if grad_ne < thr {
                    rs += s(y - 2, x + 1) + s(y, x + 1);
                    gs += s(y - 1, x + 1) * 2;
                    bs += s(y - 1, x) + s(y - 1, x + 2);
                    ng += 1;
                }
                if grad_sw < thr {
                    rs += s(y + 2, x - 1) + s(y, x - 1);
                    gs += s(y + 1, x - 1) * 2;
                    bs += s(y + 1, x) + s(y + 1, x - 2);
                    ng += 1;
                }
                if grad_nw < thr {
                    rs += s(y - 2, x - 1) + s(y, x - 1);
                    gs += s(y - 1, x - 1) * 2;
                    bs += s(y - 1, x - 2) + s(y - 1, x);
                    ng += 1;
                }
                if grad_se < thr {
                    rs += s(y + 2, x + 1) + s(y, x + 1);
                    gs += s(y + 1, x + 1) * 2;
                    bs += s(y + 1, x + 2) + s(y + 1, x);
                    ng += 1;
                }
            } else {
                if grad_n < thr {
                    rs += s(y - 2, x) + s(y, x);
                    gs += s(y - 1, x) * 2;
                    bs += s(y - 1, x - 1) + s(y - 1, x + 1);
                    ng += 1;
                }
                if grad_s < thr {
                    rs += s(y + 2, x) + s(y, x);
                    gs += s(y + 1, x) * 2;
                    bs += s(y + 1, x - 1) + s(y + 1, x + 1);
                    ng += 1;
                }
                if grad_w < thr {
                    rs += s(y, x - 2) + s(y, x);
                    gs += s(y, x - 1) * 2;
                    bs += s(y - 1, x - 1) + s(y + 1, x - 1);
                    ng += 1;
                }
                if grad_e < thr {
                    rs += s(y, x + 2) + s(y, x);
                    gs += s(y, x + 1) * 2;
                    bs += s(y - 1, x + 1) + s(y + 1, x + 1);
                    ng += 1;
                }
                if grad_ne < thr {
                    rs += s(y - 2, x + 2) + s(y, x);
                    gs += g[idx(6, y - 1, x + 1)];
                    bs += s(y - 1, x + 1) * 2;
                    ng += 1;
                }
                if grad_sw < thr {
                    rs += s(y + 2, x - 2) + s(y, x);
                    gs += g[idx(6, y + 1, x - 1)];
                    bs += s(y + 1, x - 1) * 2;
                    ng += 1;
                }
                if grad_nw < thr {
                    rs += s(y - 2, x - 2) + s(y, x);
                    gs += g[idx(6, y - 1, x - 1)];
                    bs += s(y - 1, x - 1) * 2;
                    ng += 1;
                }
                if grad_se < thr {
                    rs += s(y + 2, x + 2) + s(y, x);
                    gs += g[idx(6, y + 1, x + 1)];
                    bs += s(y + 1, x + 1) * 2;
                    ng += 1;
                }
            }

            // `0.5 / ng`. Upstream's scalar path reads this from a literal
            // table of decimals — `scale[7]` is `0.0714286f`, seven digits,
            // not the correctly-rounded `0.07142857f` — while its SIMD path
            // divides. The two disagree by one level on about 0.4% of pixels,
            // so cv2's VNG output depends on how many columns its vector loop
            // could reach: the last nine columns of every row come out of the
            // table and the rest out of the division. The division is both the
            // intended value and what nearly every real build computes.
            let k = 0.5f32 / ng as f32;
            let mut px = [0i32; 3];
            if green_cell {
                let gv = s(y, x);
                px[1] = gv;
                // "R" is the colour of the horizontal neighbours, "B" that of
                // the vertical ones.
                px[colour_at(pattern, oy, ox - 1)] = gv + cv_roundf((rs - gs) as f32 * k);
                px[colour_at(pattern, oy - 1, ox)] = gv + cv_roundf((bs - gs) as f32 * k);
            } else {
                let c = colour_at(pattern, oy, ox);
                let rv = s(y, x);
                px[c] = rv;
                px[1] = rv + cv_roundf((gs - rs) as f32 * k);
                px[2 - c] = rv + cv_roundf((bs - rs) as f32 * k);
            }
            put(&mut dst, oy, ox, px, output)?;
        }
    }
    Ok(dst)
}



#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::CV_8UC1;

    /// Mosaic a full-colour image through a Bayer filter, so a demosaicer can
    /// be scored against the colours that actually went in.
    fn mosaic(rgb: &[(u8, u8, u8)], w: i32, h: i32, pattern: BayerPattern) -> Mat {
        let mut m = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                let (r, g, b) = rgb[(y * w + x) as usize];
                let v = match colour_at(pattern, y, x) {
                    0 => b,
                    1 => g,
                    _ => r,
                };
                m.pixel_mut::<u8>(y, x).unwrap()[0] = v;
            }
        }
        m
    }

    /// A flat colour field, which every demosaicer must reproduce exactly.
    fn flat(w: i32, h: i32, c: (u8, u8, u8)) -> Vec<(u8, u8, u8)> {
        vec![c; (w * h) as usize]
    }

    #[test]
    fn a_flat_colour_survives_every_method() {
        let (w, h) = (16, 16);
        for pattern in [BayerPattern::Bg, BayerPattern::Gb, BayerPattern::Rg, BayerPattern::Gr] {
            let truth = flat(w, h, (200, 100, 50));
            let m = mosaic(&truth, w, h, pattern);
            for method in [DemosaicMethod::Ea, DemosaicMethod::Vng] {
                let out = demosaicing_with(&m, pattern, BayerOutput::Bgr, method).unwrap();
                // Away from the border, where the clamp folds the mosaic back
                // on itself, the answer is exact.
                for y in 3..h - 3 {
                    for x in 3..w - 3 {
                        let (b, g, r) = (
                            out.at_f64(y, x, 0).unwrap(),
                            out.at_f64(y, x, 1).unwrap(),
                            out.at_f64(y, x, 2).unwrap(),
                        );
                        assert_eq!(
                            (b as u8, g as u8, r as u8),
                            (50, 100, 200),
                            "{pattern:?}/{method:?} at ({x}, {y})"
                        );
                    }
                }
            }
        }
    }

    /// A vertical colour edge: grey on the left, grey on the right, different
    /// levels. Bilinear smears it; the gradient-aware methods should not.
    fn vertical_edge(w: i32, h: i32) -> Vec<(u8, u8, u8)> {
        (0..(w * h))
            .map(|i| {
                let x = i % w;
                let v = if x < w / 2 { 40u8 } else { 210u8 };
                (v, v, v)
            })
            .collect()
    }

    #[test]
    fn the_gradient_aware_methods_beat_bilinear_across_an_edge() {
        let (w, h) = (24, 24);
        let truth = vertical_edge(w, h);
        let pattern = BayerPattern::Bg;
        let m = mosaic(&truth, w, h, pattern);

        let err = |out: &Mat| -> f64 {
            let mut e = 0.0;
            let mut n = 0.0;
            for y in 4..h - 4 {
                for x in 4..w - 4 {
                    let (r, g, b) = truth[(y * w + x) as usize];
                    for (c, want) in [(0, b), (1, g), (2, r)] {
                        e += (out.at_f64(y, x, c).unwrap() - f64::from(want)).abs();
                        n += 1.0;
                    }
                }
            }
            e / n
        };

        let bilinear = err(&crate::extra::demosaicing_to(&m, pattern, BayerOutput::Bgr).unwrap());
        let ea = err(&demosaicing_with(&m, pattern, BayerOutput::Bgr, DemosaicMethod::Ea).unwrap());
        let vng =
            err(&demosaicing_with(&m, pattern, BayerOutput::Bgr, DemosaicMethod::Vng).unwrap());
        assert!(ea < bilinear, "EA {ea} should beat bilinear {bilinear}");
        assert!(vng < bilinear, "VNG {vng} should beat bilinear {bilinear}");
    }

    #[test]
    fn the_output_layouts_agree_channel_for_channel() {
        let (w, h) = (12, 12);
        let truth: Vec<(u8, u8, u8)> = (0..(w * h))
            .map(|i| ((i * 7 % 256) as u8, (i * 13 % 256) as u8, (i * 29 % 256) as u8))
            .collect();
        let m = mosaic(&truth, w, h, BayerPattern::Rg);
        for method in [DemosaicMethod::Ea, DemosaicMethod::Vng] {
            let bgr = demosaicing_with(&m, BayerPattern::Rg, BayerOutput::Bgr, method).unwrap();
            let rgb = demosaicing_with(&m, BayerPattern::Rg, BayerOutput::Rgb, method).unwrap();
            let bgra = demosaicing_with(&m, BayerPattern::Rg, BayerOutput::Bgra, method).unwrap();
            assert_eq!(bgr.channels(), 3);
            assert_eq!(bgra.channels(), 4);
            for y in 0..h {
                for x in 0..w {
                    // RGB is BGR reversed, and BGRA is BGR with opaque alpha.
                    assert_eq!(bgr.at_f64(y, x, 0).unwrap(), rgb.at_f64(y, x, 2).unwrap());
                    assert_eq!(bgr.at_f64(y, x, 1).unwrap(), rgb.at_f64(y, x, 1).unwrap());
                    assert_eq!(bgr.at_f64(y, x, 2).unwrap(), rgb.at_f64(y, x, 0).unwrap());
                    assert_eq!(bgr.at_f64(y, x, 0).unwrap(), bgra.at_f64(y, x, 0).unwrap());
                    assert_eq!(bgra.at_f64(y, x, 3).unwrap(), 255.0);
                }
            }
        }
    }

    #[test]
    fn bilinear_still_routes_to_the_original_implementation() {
        let (w, h) = (10, 10);
        let truth = flat(w, h, (10, 20, 30));
        let m = mosaic(&truth, w, h, BayerPattern::Gb);
        let a = demosaicing_with(&m, BayerPattern::Gb, BayerOutput::Bgr, DemosaicMethod::Bilinear)
            .unwrap();
        let b = crate::extra::demosaicing_to(&m, BayerPattern::Gb, BayerOutput::Bgr).unwrap();
        assert_eq!(a.to_vec::<u8>().unwrap(), b.to_vec::<u8>().unwrap());
        // Bilinear keeps its grey form; the gradient-aware ones do not have one.
        assert!(demosaicing_with(&m, BayerPattern::Gb, BayerOutput::Gray, DemosaicMethod::Bilinear)
            .is_ok());
    }

    #[test]
    fn the_gradient_aware_methods_check_their_input() {
        let m = Mat::zeros(8, 8, CV_8UC1).unwrap();
        assert!(demosaicing_with(&m, BayerPattern::Bg, BayerOutput::Gray, DemosaicMethod::Vng)
            .is_err(),);
        assert!(demosaicing_with(&m, BayerPattern::Bg, BayerOutput::Gray, DemosaicMethod::Ea)
            .is_err());
        // Too small for a 5x5 neighbourhood.
        let tiny = Mat::zeros(4, 4, CV_8UC1).unwrap();
        assert!(demosaicing_with(&tiny, BayerPattern::Bg, BayerOutput::Bgr, DemosaicMethod::Vng)
            .is_err());
        // Not a mosaic.
        let colour = Mat::zeros(8, 8, MatType::new(Depth::U8, 3)).unwrap();
        assert!(demosaicing_with(&colour, BayerPattern::Bg, BayerOutput::Bgr, DemosaicMethod::Ea)
            .is_err());
        let deep = Mat::zeros(8, 8, MatType::new(Depth::U16, 1)).unwrap();
        assert!(demosaicing_with(&deep, BayerPattern::Bg, BayerOutput::Bgr, DemosaicMethod::Ea)
            .is_err());
        assert_eq!(DemosaicMethod::default(), DemosaicMethod::Bilinear);
    }
}
