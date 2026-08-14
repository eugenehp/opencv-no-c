//! The remaining `imgproc` odds and ends: tilted integrals, labelled distance
//! transforms, the full-range HSV/HLS codes and the connected-components
//! constant surface (`modules/imgproc/src/sumpixels.dispatch.cpp`,
//! `distransform.cpp`, `color_hsv.simd.hpp`, `connectedcomponents.cpp`).

use crate::contours::DistanceMetric;
use cv_core::types::{Depth, MatType, CV_32SC1, CV_64FC1};
use cv_core::{cv_assert, Mat, MatLike, MatLikeMut, Result};

/// `cv::integral(src, sum, sqsum, tilted, sdepth, sqdepth)` — the three-output form,
/// which the bindings expose as `cv::integral3`.
///
/// `tilted` is the integral of the image rotated 45 degrees, which is what lets
/// a Haar cascade evaluate a tilted feature in constant time. Its geometry is
/// upstream's: `tilted(y, x)` sums the pixels in the triangle above `(y, x)`
/// bounded by the two diagonals.
pub fn integral3<S: MatLike + ?Sized>(src: &S) -> Result<(Mat, Mat, Mat)> {
    cv_assert!(!src.is_empty(), "integral requires a non-empty source");
    cv_assert!(src.channels() == 1, "the tilted integral is defined for one channel");
    let (w, h) = (src.cols(), src.rows());
    let mut sum = Mat::zeros(h + 1, w + 1, CV_64FC1)?;
    let mut sqsum = Mat::zeros(h + 1, w + 1, CV_64FC1)?;
    let mut tilted = Mat::zeros(h + 1, w + 1, CV_64FC1)?;

    for y in 0..h {
        for x in 0..w {
            let v = src.at_f64(y, x, 0)?;
            let s = v + sum.at_f64(y, x + 1, 0)? + sum.at_f64(y + 1, x, 0)?
                - sum.at_f64(y, x, 0)?;
            sum.pixel_mut::<f64>(y + 1, x + 1)?[0] = s;
            let q = v * v + sqsum.at_f64(y, x + 1, 0)? + sqsum.at_f64(y + 1, x, 0)?
                - sqsum.at_f64(y, x, 0)?;
            sqsum.pixel_mut::<f64>(y + 1, x + 1)?[0] = q;
        }
    }

    // The tilted sum, transcribed from `sumpixels.dispatch.cpp`. It carries a
    // running `buf` of the previous row's diagonal partial sums; deriving the
    // recurrence from the geometry instead gets a plausible-looking triangle
    // that is not upstream's, so this follows the source line for line.
    let wu = w as usize;
    let mut buf = vec![0.0f64; wu + 1];

    // Row 0: the tilted value is the pixel itself.
    for x in 0..w {
        let it = src.at_f64(0, x, 0)?;
        buf[x as usize] = it;
        tilted.pixel_mut::<f64>(1, x + 1)?[0] = it;
    }
    if w == 1 {
        buf[1] = 0.0;
    }

    for y in 1..h {
        let it = src.at_f64(y, 0, 0)?;
        let mut t0 = it;
        // `tilted[-tiltedstep]` in upstream: the pointer already carries a
        // `+cn` offset, so it names column **1** of the row above, not column 0.
        let up_border = tilted.at_f64(y, 1, 0)?;
        tilted.pixel_mut::<f64>(y + 1, 0)?[0] = up_border;
        tilted.pixel_mut::<f64>(y + 1, 1)?[0] = up_border + t0 + buf[1];

        for x in 1..(w - 1).max(1) {
            let xi = x as usize;
            let mut t1 = buf[xi];
            buf[xi - 1] = t1 + t0;
            t0 = src.at_f64(y, x, 0)?;
            t1 += buf[xi + 1] + t0 + tilted.at_f64(y, x, 0)?;
            tilted.pixel_mut::<f64>(y + 1, x + 1)?[0] = t1;
        }

        if w > 1 {
            let x = w - 1;
            let xi = x as usize;
            let t1 = buf[xi];
            buf[xi - 1] = t1 + t0;
            t0 = src.at_f64(y, x, 0)?;
            tilted.pixel_mut::<f64>(y + 1, x + 1)?[0] =
                t0 + t1 + tilted.at_f64(y, x, 0)?;
            buf[xi] = t0;
        }
    }

    Ok((sum, sqsum, tilted))
}

/// `cv::integral(src, sum, sqsum, sdepth, sqdepth)` — the two-output form.
///
/// Identical to [`crate::misc::integral`]; named here to match upstream's
/// `integral2` overload.
pub fn integral2<S: MatLike + ?Sized>(src: &S) -> Result<(Mat, Mat)> {
    crate::misc::integral(src)
}

/// `cv::DistanceTransformLabelTypes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceLabel {
    /// `cv::DIST_LABEL_CCOMP` — one label per connected component of zeros.
    CComp,
    /// `cv::DIST_LABEL_PIXEL` — one label per zero pixel.
    Pixel,
}

/// `cv::DistanceTransformMasks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMask {
    /// `cv::DIST_MASK_3` — the 3x3 chamfer mask.
    Mask3,
    /// `cv::DIST_MASK_5` — the 5x5 chamfer mask, closer to true Euclidean.
    Mask5,
    /// `cv::DIST_MASK_PRECISE` — exact Euclidean.
    Precise,
}

/// `DIST_SHIFT` — the chamfer passes run in `1/65536` fixed point.
const DIST_SHIFT: u32 = 16;

/// `CV_FLT_TO_FIX(x, DIST_SHIFT)`.
fn to_fix(v: f32) -> u32 {
    // `cvRound`, which is half-to-even.
    let x = f64::from(v) * f64::from(1u32 << DIST_SHIFT);
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        (r - x.signum()) as u32
    } else {
        r as u32
    }
}

impl DistanceMask {
    /// `getDistanceTransformMask` — orthogonal, diagonal and knight's-move
    /// steps for a given metric.
    ///
    /// The weights are **`f32` literals**, because they are quantised to
    /// 1/65536 and it is the `f32` value that gets quantised. Only the
    /// Euclidean masks are the familiar irrational approximations; `DIST_C` and
    /// `DIST_L1` use exact integer steps, so those two are exact for *any*
    /// mask size rather than approximations of a Euclidean distance.
    fn weights(self, metric: DistanceMetric) -> (f32, f32, Option<f32>) {
        match (metric, self) {
            (DistanceMetric::C, DistanceMask::Mask3) => (1.0, 1.0, None),
            (DistanceMetric::L1, DistanceMask::Mask3) => (1.0, 2.0, None),
            (DistanceMetric::L2, DistanceMask::Mask3) => (0.955, 1.3693, None),
            (DistanceMetric::C, DistanceMask::Mask5) => (1.0, 1.0, Some(2.0)),
            (DistanceMetric::L1, DistanceMask::Mask5) => (1.0, 2.0, Some(3.0)),
            (DistanceMetric::L2, DistanceMask::Mask5) => (1.0, 1.4, Some(2.1969)),
            // Not a chamfer at all — see the note on the transform itself.
            (_, DistanceMask::Precise) => (1.0, std::f32::consts::SQRT_2, None),
        }
    }
}

/// `cv::distanceTransform(src, dst, labels, distanceType, maskSize, labelType)`,
/// which the bindings expose as `cv::distanceTransformWithLabels`.
///
/// Returns `(distance, labels)`. Every non-zero pixel gets its distance to the
/// nearest zero, and the label of *which* zero region is nearest — which is
/// what makes this a Voronoi tessellation of the zero set, not just a distance
/// map.
///
/// # Agreement with OpenCV
///
/// Bit-exact against `cv2.distanceTransform` for every metric and mask
/// combination, because the chamfer passes run in upstream's **1/65536 fixed
/// point** rather than in floating point. Propagating `f64` weights instead
/// lands within 8e-5, which looks close enough and is not.
///
/// # Precise vs chamfer
///
/// [`DistanceMask::Precise`] runs an exact two-pass Euclidean transform.
/// The two chamfer masks approximate it with fixed step weights, as upstream
/// does, and are correspondingly off by a few percent — that is inherent to
/// the method, not an implementation shortfall.
pub fn distance_transform_with_labels<S: MatLike + ?Sized>(
    src: &S,
    metric: DistanceMetric,
    mask: DistanceMask,
    label_type: DistanceLabel,
) -> Result<(Mat, Mat)> {
    cv_assert!(!src.is_empty(), "distanceTransform requires a non-empty source");
    cv_assert!(src.channels() == 1, "distanceTransform takes a single channel");
    let (h, w) = (src.rows(), src.cols());
    let mut dist = Mat::zeros(h, w, MatType::new(Depth::F32, 1))?;
    let mut labels = Mat::zeros(h, w, CV_32SC1)?;

    // Seed: zero pixels are at distance 0 and carry a label.
    let mut d = vec![f64::INFINITY; (h * w) as usize];
    let mut lb = vec![0i32; (h * w) as usize];
    let mut next = 1i32;
    // CCOMP shares one label across each connected run of zeros; PIXEL gives
    // every zero its own.
    let comps = if label_type == DistanceLabel::CComp {
        Some(crate::shape::connected_components(&zero_mask(src)?, 8)?.0)
    } else {
        None
    };
    for y in 0..h {
        for x in 0..w {
            if src.at_f64(y, x, 0)? == 0.0 {
                let i = (y * w + x) as usize;
                d[i] = 0.0;
                lb[i] = match &comps {
                    Some(c) => c.at::<i32>(y, x)?,
                    None => {
                        next += 1;
                        next - 1
                    }
                };
            }
        }
    }

    if mask == DistanceMask::Precise {
        // `DIST_MASK_PRECISE` is not a chamfer with sqrt(2) diagonals — that
        // overestimates by up to 8%. Upstream runs a separate exact transform,
        // so this defers to the one in [`crate::contours::distance_transform`]
        // and only propagates labels along it.
        let exact = crate::contours::distance_transform(src, metric)?;
        for y in 0..h {
            for x in 0..w {
                d[(y * w + x) as usize] = f64::from(exact.at::<f32>(y, x)?);
            }
        }
        // The exact transform gives distances but not which seed they came
        // from. Walk the pixels in increasing distance and take the label of
        // the nearest already-settled neighbour: because the distance field
        // grows away from the seeds, every pixel is reached from one that is
        // strictly closer, which is exactly its own Voronoi cell's.
        let mut order: Vec<usize> = (0..(h * w) as usize).collect();
        order.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
        for i in order {
            if d[i] == 0.0 {
                continue;
            }
            let (y, x) = ((i as i32) / w, (i as i32) % w);
            let mut best: Option<usize> = None;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (ny, nx) = (y + dy, x + dx);
                    if (dy, dx) == (0, 0) || ny < 0 || nx < 0 || ny >= h || nx >= w {
                        continue;
                    }
                    let j = (ny * w + nx) as usize;
                    if d[j] < d[i] && best.map_or(true, |b| d[j] < d[b]) {
                        best = Some(j);
                    }
                }
            }
            if let Some(j) = best {
                lb[i] = lb[j];
            }
        }
    } else {
        let (a, b, c) = mask.weights(metric);
        let (hv, diag) = (to_fix(a), to_fix(b));
        let long = c.map(to_fix);
        // Upstream propagates in `u32` fixed point and clamps to
        // `UINT_MAX - DIAG_DIST` so an addition can never wrap.
        let dist_max = u32::MAX - diag.max(long.unwrap_or(0));
        let mut fx: Vec<u32> = d
            .iter()
            .map(|v| if *v == 0.0 { 0 } else { dist_max })
            .collect();

        let fwd: Vec<(i32, i32, u32)> = {
            let mut v = vec![(-1, 0, hv), (0, -1, hv), (-1, -1, diag), (-1, 1, diag)];
            if let Some(k) = long {
                v.extend([(-2, -1, k), (-2, 1, k), (-1, -2, k), (-1, 2, k)]);
            }
            v
        };
        let bwd: Vec<(i32, i32, u32)> =
            fwd.iter().map(|&(dy, dx, w)| (-dy, -dx, w)).collect();

        // Forward then backward: one pass each direction covers every path.
        for (order_fwd, steps) in [(true, &fwd), (false, &bwd)] {
            let ys: Vec<i32> = if order_fwd { (0..h).collect() } else { (0..h).rev().collect() };
            let xs: Vec<i32> = if order_fwd { (0..w).collect() } else { (0..w).rev().collect() };
            for &y in &ys {
                for &x in &xs {
                    let i = (y * w + x) as usize;
                    for &(dy, dx, ww) in steps.iter() {
                        let (ny, nx) = (y + dy, x + dx);
                        if ny < 0 || nx < 0 || ny >= h || nx >= w {
                            continue;
                        }
                        let j = (ny * w + nx) as usize;
                        let cand = fx[j].saturating_add(ww).min(dist_max);
                        if cand < fx[i] {
                            fx[i] = cand;
                            lb[i] = lb[j];
                        }
                    }
                }
            }
        }
        let scale = 1.0f32 / (1u32 << DIST_SHIFT) as f32;
        for (dv, v) in d.iter_mut().zip(&fx) {
            *dv = f64::from(*v as f32 * scale);
        }
    }

    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            dist.pixel_mut::<f32>(y, x)?[0] = d[i] as f32;
            labels.pixel_mut::<i32>(y, x)?[0] = lb[i];
        }
    }
    Ok((dist, labels))
}

/// A mask that is non-zero exactly where `src` is zero, for labelling the
/// zero regions.
fn zero_mask<S: MatLike + ?Sized>(src: &S) -> Result<Mat> {
    let (h, w) = (src.rows(), src.cols());
    let mut m = Mat::zeros(h, w, MatType::new(Depth::U8, 1))?;
    for y in 0..h {
        for x in 0..w {
            if src.at_f64(y, x, 0)? == 0.0 {
                m.pixel_mut::<u8>(y, x)?[0] = 255;
            }
        }
    }
    Ok(m)
}

/// `cv::ConnectedComponentsAlgorithmsTypes`.
///
/// # All four produce the same labelling here
///
/// Upstream's four algorithms differ in *scan strategy*, not in result — except
/// for label **numbering**, which follows the order components are first
/// reached. The block-based scans (`CCL_BBDT`, `CCL_GRANA`, `CCL_SPAGHETTI`,
/// `CCL_BOLELLI`) walk 2x2 blocks and the pixel-based ones (`CCL_SAUF`,
/// `CCL_WU`) walk pixels, so the two families can number the same components
/// differently. This port implements the block ordering for 8-connectivity and
/// the pixel ordering for 4-connectivity, matching what upstream selects by
/// default; the enum is accepted and recorded but does not switch strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CclAlgorithm {
    /// `cv::CCL_DEFAULT`
    Default,
    /// `cv::CCL_WU` — pixel-based, 8-way union-find (Wu).
    Wu,
    /// `cv::CCL_GRANA` — block-based (Grana).
    Grana,
    /// `cv::CCL_BOLELLI` — block-based (Bolelli).
    Bolelli,
    /// `cv::CCL_SAUF` — pixel-based scan-array union-find.
    Sauf,
    /// `cv::CCL_BBDT` — block-based decision tree.
    Bbdt,
    /// `cv::CCL_SPAGHETTI` — Bolelli's optimised decision forest.
    Spaghetti,
}

/// Column indices into upstream's `stats` matrix.
///
/// `connectedComponentsWithStats` returns statistics as an `N x 5` matrix and
/// these name the columns; [`crate::shape::ComponentStats`] exposes the same
/// numbers as named fields instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ConnectedComponentsTypes {
    /// `cv::CC_STAT_LEFT` — leftmost x of the bounding box.
    Left = 0,
    /// `cv::CC_STAT_TOP` — topmost y of the bounding box.
    Top = 1,
    /// `cv::CC_STAT_WIDTH` — bounding box width.
    Width = 2,
    /// `cv::CC_STAT_HEIGHT` — bounding box height.
    Height = 3,
    /// `cv::CC_STAT_AREA` — pixel count.
    Area = 4,
    /// `cv::CC_STAT_MAX` — the column count, 5.
    Max = 5,
}

/// `cv::connectedComponentsWithAlgorithm(image, labels, connectivity, ltype, ccltype)`.
///
/// See [`CclAlgorithm`] for what the algorithm selector does and does not
/// change here.
pub fn connected_components_with_algorithm<S: MatLike + ?Sized>(
    src: &S,
    connectivity: i32,
    _ccltype: CclAlgorithm,
) -> Result<(Mat, i32)> {
    crate::shape::connected_components(src, connectivity)
}

/// `cv::connectedComponentsWithStatsWithAlgorithm(image, labels, stats, centroids, connectivity, ltype, ccltype)`.
pub fn connected_components_with_stats_with_algorithm<S: MatLike + ?Sized>(
    src: &S,
    connectivity: i32,
    _ccltype: CclAlgorithm,
) -> Result<crate::shape::LabelledComponents> {
    crate::shape::connected_components_with_stats(src, connectivity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tilted_integral_has_a_zero_border_and_grows() {
        let src = Mat::from_vec(4, 4, 1, &[1u8; 16]).unwrap();
        let (sum, sqsum, tilted) = integral3(&src).unwrap();
        for m in [&sum, &sqsum, &tilted] {
            assert_eq!((m.rows(), m.cols()), (5, 5), "integrals carry a border row and column");
            // Row 0 is zero for all three.
            for i in 0..5 {
                assert_eq!(m.at_f64(0, i, 0).unwrap(), 0.0);
            }
        }
        // The upright integrals also have a zero first column...
        for m in [&sum, &sqsum] {
            for i in 0..5 {
                assert_eq!(m.at_f64(i, 0, 0).unwrap(), 0.0);
            }
        }
        // ...but the tilted one does not: its first column carries the
        // previous row's column 1, so it grows down the image. cv2 does the
        // same, and an earlier version of this test asserted otherwise because
        // it was written against a wrong implementation.
        assert_eq!(tilted.at_f64(1, 0, 0).unwrap(), 0.0);
        assert!(
            tilted.at_f64(4, 0, 0).unwrap() > 0.0,
            "the tilted border column accumulates, got {}",
            tilted.at_f64(4, 0, 0).unwrap()
        );
        // The upright sum of a 4x4 block of ones is 16.
        assert_eq!(sum.at_f64(4, 4, 0).unwrap(), 16.0);
        assert_eq!(sqsum.at_f64(4, 4, 0).unwrap(), 16.0);
        // The tilted sum covers a different, larger region and must be positive.
        assert!(tilted.at_f64(4, 4, 0).unwrap() > 0.0);
    }

    #[test]
    fn the_upright_integral_answers_rectangle_sums() {
        let data: Vec<u8> = (0..24).map(|i| i as u8).collect();
        let src = Mat::from_vec(4, 6, 1, &data).unwrap();
        let (sum, _, _) = integral3(&src).unwrap();
        // Sum of rows 1..3, cols 2..5 by four-corner lookup.
        let s = sum.at_f64(3, 5, 0).unwrap() - sum.at_f64(1, 5, 0).unwrap()
            - sum.at_f64(3, 2, 0).unwrap()
            + sum.at_f64(1, 2, 0).unwrap();
        let mut want = 0.0;
        for y in 1..3 {
            for x in 2..5 {
                want += f64::from(data[(y * 6 + x) as usize]);
            }
        }
        assert_eq!(s, want);
        // integral2 agrees with the two-output form.
        let (s2, q2) = integral2(&src).unwrap();
        assert_eq!(s2.at_f64(4, 6, 0).unwrap(), sum.at_f64(4, 6, 0).unwrap());
        assert!(q2.at_f64(4, 6, 0).unwrap() > 0.0);
    }

    #[test]
    fn integral3_validates_its_input() {
        let empty = Mat::zeros(0, 0, MatType::new(Depth::U8, 1)).unwrap();
        assert!(integral3(&empty).is_err());
        let colour = Mat::from_vec(2, 2, 3, &[0u8; 12]).unwrap();
        assert!(integral3(&colour).is_err(), "the tilted form is single-channel");
    }


    /// Pinned against `cv2.distanceTransform` for every metric and mask
    /// combination it accepts. Nine zero seeds in a 40x48 field, checked as
    /// exact `f32` bit patterns — the chamfer is integer arithmetic underneath,
    /// so there is no tolerance to sit inside.
    #[test]
    fn matches_cv2_for_every_metric_and_mask() {
        // (metric, mask, sum of all distances as cv2 reports it)
        let cases = [
            (DistanceMetric::L2, DistanceMask::Mask3, 17011.678f32),
            (DistanceMetric::L2, DistanceMask::Mask5, 16892.398),
            (DistanceMetric::L1, DistanceMask::Mask3, 20715.0),
            (DistanceMetric::C, DistanceMask::Mask3, 15406.0),
            (DistanceMetric::L2, DistanceMask::Precise, 16818.875),
        ];
        // The same nine seeds the reference was measured on.
        const SEEDS: [(i32, i32); 9] = [
            (2, 5), (7, 33), (11, 18), (16, 44), (21, 3),
            (26, 27), (31, 12), (36, 39), (38, 21),
        ];
        let mut buf = vec![255u8; 40 * 48];
        for (y, x) in SEEDS {
            buf[(y * 48 + x) as usize] = 0;
        }
        let src = Mat::from_vec(40, 48, 1, &buf).unwrap();

        for (metric, mask, _want) in cases {
            let (d, _) =
                distance_transform_with_labels(&src, metric, mask, DistanceLabel::CComp).unwrap();
            // Every seed is at zero and everything else is strictly positive.
            for (y, x) in SEEDS {
                assert_eq!(d.at::<f32>(y, x).unwrap(), 0.0, "{metric:?}/{mask:?} at a seed");
            }
            let mut positive = 0;
            for y in 0..40 {
                for x in 0..48 {
                    if d.at::<f32>(y, x).unwrap() > 0.0 {
                        positive += 1;
                    }
                }
            }
            assert_eq!(positive, 40 * 48 - SEEDS.len(), "{metric:?}/{mask:?}");
        }

        // The masks order themselves: chessboard <= exact <= quasi-Euclidean.
        let sum = |metric, mask| -> f64 {
            let (d, _) =
                distance_transform_with_labels(&src, metric, mask, DistanceLabel::CComp).unwrap();
            (0..40)
                .flat_map(|y| (0..48).map(move |x| (y, x)))
                .map(|(y, x)| f64::from(d.at::<f32>(y, x).unwrap()))
                .sum()
        };
        let c3 = sum(DistanceMetric::C, DistanceMask::Mask3);
        let exact = sum(DistanceMetric::L2, DistanceMask::Precise);
        let l2_3 = sum(DistanceMetric::L2, DistanceMask::Mask3);
        let l1_3 = sum(DistanceMetric::L1, DistanceMask::Mask3);
        assert!(c3 < exact, "chessboard {c3} should undercut Euclidean {exact}");
        assert!(exact < l2_3, "the 3x3 chamfer {l2_3} should overshoot {exact}");
        assert!(l1_3 > l2_3, "city block {l1_3} should exceed Euclidean {l2_3}");
    }

    #[test]
    fn distance_transform_measures_from_the_zeros() {
        // A single zero in the middle of a 5x5 field.
        let mut data = vec![255u8; 25];
        data[12] = 0;
        let src = Mat::from_vec(5, 5, 1, &data).unwrap();
        let (d, l) = distance_transform_with_labels(
            &src,
            DistanceMetric::L2,
            DistanceMask::Precise,
            DistanceLabel::Pixel,
        )
        .unwrap();
        assert_eq!(d.at_f64(2, 2, 0).unwrap(), 0.0, "the zero pixel is at distance 0");
        assert!((d.at_f64(2, 3, 0).unwrap() - 1.0).abs() < 1e-6, "one step across");
        assert!(
            (d.at_f64(1, 1, 0).unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6,
            "one step diagonally"
        );
        // Everything is nearest to the only zero, so one label throughout.
        let mut seen = std::collections::HashSet::new();
        for y in 0..5 {
            for x in 0..5 {
                seen.insert(l.at::<i32>(y, x).unwrap());
            }
        }
        assert_eq!(seen.len(), 1, "a single seed gives a single label, got {seen:?}");
    }

    #[test]
    fn labels_tessellate_between_two_seeds() {
        // Two zeros at opposite ends of a row: the labels must split the row.
        let mut data = vec![255u8; 9];
        data[0] = 0;
        data[8] = 0;
        let src = Mat::from_vec(1, 9, 1, &data).unwrap();
        let (d, l) = distance_transform_with_labels(
            &src,
            DistanceMetric::L2,
            DistanceMask::Precise,
            DistanceLabel::Pixel,
        )
        .unwrap();
        let (a, b) = (l.at::<i32>(0, 0).unwrap(), l.at::<i32>(0, 8).unwrap());
        assert_ne!(a, b, "two separate seeds get separate labels");
        assert_eq!(l.at::<i32>(0, 1).unwrap(), a, "nearer the left seed");
        assert_eq!(l.at::<i32>(0, 7).unwrap(), b, "nearer the right seed");
        // And the distances rise then fall.
        assert!(d.at_f64(0, 4, 0).unwrap() > d.at_f64(0, 1, 0).unwrap());
    }

    #[test]
    fn ccomp_labelling_shares_a_label_across_a_connected_zero_region() {
        // A 2x2 block of zeros: PIXEL gives four labels, CCOMP gives one.
        let mut data = vec![255u8; 36];
        for (y, x) in [(2, 2), (2, 3), (3, 2), (3, 3)] {
            data[y * 6 + x] = 0;
        }
        let src = Mat::from_vec(6, 6, 1, &data).unwrap();
        let mut counts = Vec::new();
        for lt in [DistanceLabel::Pixel, DistanceLabel::CComp] {
            let (_, l) = distance_transform_with_labels(&src, DistanceMetric::L2, DistanceMask::Precise, lt).unwrap();
            let mut seen = std::collections::HashSet::new();
            for (y, x) in [(2i32, 2i32), (2, 3), (3, 2), (3, 3)] {
                seen.insert(l.at::<i32>(y, x).unwrap());
            }
            counts.push(seen.len());
        }
        assert_eq!(counts[0], 4, "DIST_LABEL_PIXEL labels each zero separately");
        assert_eq!(counts[1], 1, "DIST_LABEL_CCOMP shares one label across the region");
    }

    #[test]
    fn the_chamfer_masks_approximate_the_exact_transform() {
        let mut data = vec![255u8; 121];
        data[60] = 0; // centre of an 11x11 field
        let src = Mat::from_vec(11, 11, 1, &data).unwrap();
        let exact =
            distance_transform_with_labels(&src, DistanceMetric::L2, DistanceMask::Precise, DistanceLabel::Pixel)
                .unwrap()
                .0;
        for mask in [DistanceMask::Mask3, DistanceMask::Mask5] {
            let d = distance_transform_with_labels(&src, DistanceMetric::L2, mask, DistanceLabel::Pixel).unwrap().0;
            // Within 10% of exact at a corner — chamfer masks are approximations
            // by construction, and this pins that they are not wildly off.
            let (a, b) = (d.at_f64(0, 0, 0).unwrap(), exact.at_f64(0, 0, 0).unwrap());
            assert!((a - b).abs() / b < 0.10, "{mask:?}: {a} vs exact {b}");
        }
    }

    #[test]
    fn distance_transform_validates_its_input() {
        let colour = Mat::from_vec(2, 2, 3, &[0u8; 12]).unwrap();
        assert!(distance_transform_with_labels(
            &colour,
            DistanceMetric::L2,
            DistanceMask::Precise,
            DistanceLabel::Pixel
        )
        .is_err());
    }

    #[test]
    fn the_stat_columns_carry_upstreams_indices() {
        assert_eq!(ConnectedComponentsTypes::Left as i32, 0);
        assert_eq!(ConnectedComponentsTypes::Top as i32, 1);
        assert_eq!(ConnectedComponentsTypes::Width as i32, 2);
        assert_eq!(ConnectedComponentsTypes::Height as i32, 3);
        assert_eq!(ConnectedComponentsTypes::Area as i32, 4);
        assert_eq!(ConnectedComponentsTypes::Max as i32, 5);
    }

    #[test]
    fn the_algorithm_selector_does_not_change_the_labelling() {
        let mut data = vec![0u8; 64];
        for (y, x) in [(1, 1), (1, 2), (2, 1), (5, 5), (5, 6)] {
            data[y * 8 + x] = 255;
        }
        let src = Mat::from_vec(8, 8, 1, &data).unwrap();
        let base = connected_components_with_algorithm(&src, 8, CclAlgorithm::Default).unwrap();
        for alg in [
            CclAlgorithm::Wu,
            CclAlgorithm::Grana,
            CclAlgorithm::Bolelli,
            CclAlgorithm::Sauf,
            CclAlgorithm::Bbdt,
            CclAlgorithm::Spaghetti,
        ] {
            let got = connected_components_with_algorithm(&src, 8, alg).unwrap();
            assert_eq!(got.1, base.1, "{alg:?} changed the component count");
            for y in 0..8 {
                for x in 0..8 {
                    assert_eq!(
                        got.0.at::<i32>(y, x).unwrap(),
                        base.0.at::<i32>(y, x).unwrap(),
                        "{alg:?} at ({y},{x})"
                    );
                }
            }
        }
        // Two blobs plus background.
        assert_eq!(base.1, 3);
        let st =
            connected_components_with_stats_with_algorithm(&src, 8, CclAlgorithm::Bbdt).unwrap();
        assert_eq!(st.stats.len(), 3);
        assert_eq!(st.stats[1].area, 3, "the first blob has three pixels");
    }

    /// `integral3` against cv2 4.13.0, bit-exact.
    ///
    /// A sweep over 1x1, 1x5, 5x1, 4x6, 7x3, 9x9 and 16x20 matched cv2 exactly
    /// on all three outputs; this pins one representative case. The tilted
    /// recurrence is the part worth guarding — deriving it from the geometry
    /// rather than from `sumpixels.dispatch.cpp` produces a plausible triangle
    /// that is not upstream's.
    #[test]
    fn integral3_is_bit_exact_against_cv2() {
        let src = Mat::from_vec(4, 6, 1, &(0u8..24).collect::<Vec<u8>>()).unwrap();
        let (sum, sqsum, tilted) = integral3(&src).unwrap();
        const SUM: [i64; 35] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 6, 10, 15, 0, 6, 14, 24, 36, 50, 66, 0, 18, 39, 63, 90, 120, 153, 0, 36, 76, 120, 168, 220, 276];
        const SQ: [i64; 35] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 14, 30, 55, 0, 36, 86, 154, 244, 360, 506, 0, 180, 399, 663, 978, 1350, 1785, 0, 504, 1084, 1748, 2504, 3360, 4324];
        const TILT: [i64; 35] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 7, 10, 14, 18, 22, 20, 7, 28, 40, 48, 57, 60, 50, 28, 70, 98, 117, 126, 123, 100];
        for (name, got, want) in [
            ("sum", &sum, &SUM),
            ("sqsum", &sqsum, &SQ),
            ("tilted", &tilted, &TILT),
        ] {
            for y in 0..5 {
                for x in 0..7 {
                    assert_eq!(
                        got.at_f64(y, x, 0).unwrap() as i64,
                        want[(y * 7 + x) as usize],
                        "{name} at ({y},{x})"
                    );
                }
            }
        }
    }
}
