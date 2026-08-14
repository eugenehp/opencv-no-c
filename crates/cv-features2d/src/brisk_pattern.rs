//! BRISK's sampling pattern and descriptor (`modules/features2d/src/brisk.cpp`).
//!
//! # How the pattern does double duty
//!
//! BRISK samples a fixed set of points on concentric rings around the keypoint,
//! each smoothed by a Gaussian whose width grows with its ring's radius. Every
//! *pair* of those points is then sorted by separation:
//!
//! * pairs further apart than `dMin` are **long** pairs, and their intensity
//!   differences — weighted by the pair's own direction — vote on the
//!   keypoint's orientation;
//! * pairs closer than `dMax` are **short** pairs, and after the pattern is
//!   rotated by that orientation each contributes one descriptor bit.
//!
//! So orientation and description come from the same samples, which is what
//! distinguishes BRISK from ORB — ORB learns its 256 pairs offline and needs a
//! separate intensity-centroid step for orientation.
//!
//! # Everything here is integer
//!
//! `smoothed_intensity` is fixed-point throughout, and the descriptor bit is
//! `t1 > t2` on those integers. There is no tolerance to be within: a
//! transcription is either bit-exact against cv2 or it is wrong. See
//! `descriptors_match_cv2_on_its_own_keypoints`.

use cv_core::{cv_assert, MatLike, Result};

use crate::KeyPoint;

/// Descriptor length in bytes; upstream's BRISK is 512 bits.
pub const BRISK_BYTES: usize = 64;

/// `BRISK_Impl::scales_` — how finely the pattern is precomputed over size.
const SCALES: usize = 64;
/// `BRISK_Impl::scalerange_` — the span of sizes those scales cover.
const SCALERANGE: f32 = 30.0;
/// `BRISK_Impl::n_rot_` — the rotation look-up is quantised this finely.
const N_ROT: usize = 1024;
/// `BRISK_Impl::basicSize_` — the `size` a keypoint has at scale 1.
const BASIC_SIZE: f32 = 12.0;

/// Points per concentric ring (`nList`).
const RING_POINTS: [usize; 5] = [1, 10, 14, 15, 20];
/// Ring radii before scaling (`rList`, divided by the 0.85 factor).
const RING_RADIUS: [f64; 5] = [0.0, 2.9, 4.9, 7.4, 10.8];

/// One sample of the pattern at one scale and one rotation.
#[derive(Clone, Copy, Default)]
struct PatternPoint {
    x: f32,
    y: f32,
    sigma: f32,
}

/// A pair of pattern points contributing one descriptor bit.
#[derive(Clone, Copy)]
struct ShortPair {
    i: usize,
    j: usize,
}

/// A pair whose intensity difference votes on orientation, pre-weighted by the
/// pair's direction over its squared length.
#[derive(Clone, Copy)]
struct LongPair {
    i: usize,
    j: usize,
    weighted_dx: i32,
    weighted_dy: i32,
}

/// The precomputed pattern — `BRISK_Impl`'s `generateKernel` output.
pub struct BriskPattern {
    /// Indexed `[scale][rot][point]`, flattened.
    points: Vec<PatternPoint>,
    n_points: usize,
    /// Per scale: the border a keypoint must be inside for its pattern to fit.
    size_list: Vec<u32>,
    short_pairs: Vec<ShortPair>,
    long_pairs: Vec<LongPair>,
    /// Descriptor length in bytes, from the short-pair count.
    strings: usize,
}

impl BriskPattern {
    /// `BRISK_Impl::generateKernel` with upstream's default rings.
    ///
    /// # The 0.85
    ///
    /// Upstream scales every radius by `0.85 * patternScale` but leaves `dMax`
    /// and `dMin` at `5.85 * patternScale` and `8.2 * patternScale` — the
    /// thresholds are *not* scaled by 0.85. Applying it to both, or to neither,
    /// re-sorts the pairs and changes which comparisons become descriptor bits.
    #[must_use]
    pub fn new(pattern_scale: f32) -> Self {
        let f = 0.85 * f64::from(pattern_scale);
        // `rList` is a `vector<float>`: each radius is rounded to `f32` here,
        // and every later product inherits that rounding.
        let radius: Vec<f32> = RING_RADIUS.iter().map(|r| (f * r) as f32).collect();
        let d_max = 5.85 * pattern_scale;
        let d_min = 8.2 * pattern_scale;

        let n_points: usize = RING_POINTS.iter().sum();

        // The rotation table is built by repeated complex multiplication rather
        // than by calling sin/cos per entry, so its drift is part of the answer.
        let (mut cosval, mut sinval) = (1.0f64, 0.0f64);
        let dcos = (2.0 * std::f64::consts::PI / N_ROT as f64).cos();
        let dsin = (2.0 * std::f64::consts::PI / N_ROT as f64).sin();
        let mut cos_theta = vec![0.0f64; N_ROT];
        let mut sin_theta = vec![0.0f64; N_ROT];
        for rot in 0..N_ROT {
            sin_theta[rot] = sinval;
            cos_theta[rot] = cosval;
            let t = sinval * dcos + cosval * dsin;
            cosval = cosval * dcos - sinval * dsin;
            sinval = t;
        }

        // `lb_scale` and its step are `float` upstream, so the scale ladder is
        // built from an `f32` exponent.
        let lb_scale = SCALERANGE.ln() / std::f32::consts::LN_2;
        let lb_scale_step = lb_scale / SCALES as f32;
        const SIGMA_SCALE: f32 = 1.3;

        let mut points = vec![PatternPoint::default(); n_points * SCALES * N_ROT];
        let mut size_list = vec![0u32; SCALES];
        for (scale, size_slot) in size_list.iter_mut().enumerate() {
            // `scaleList_` is a `float[]`; `pow` runs in double and is then
            // narrowed, so the stored scale is an `f32`.
            let scale_val = 2f64.powf(f64::from(scale as f32 * lb_scale_step)) as f32;
            let mut outer = 0usize;
            for ring in 0..RING_RADIUS.len() {
                // `float * float`, widened afterwards — not a double product.
                let scale_radius = f64::from(scale_val * radius[ring]);
                let sigma: f32 = if ring == 0 {
                    SIGMA_SCALE * scale_val * 0.5
                } else {
                    (f64::from(SIGMA_SCALE * scale_val)
                        * f64::from(radius[ring])
                        * (std::f64::consts::PI / RING_POINTS[ring] as f64).sin())
                        as f32
                };
                let size = (scale_val * radius[ring] + sigma).ceil() as u32 + 1;
                *size_slot = (*size_slot).max(size);

                for num in 0..RING_POINTS[ring] {
                    let alpha = num as f64 * 2.0 * std::f64::consts::PI
                        / RING_POINTS[ring] as f64;
                    let (sa, ca) = (alpha.sin(), alpha.cos());
                    for rot in 0..N_ROT {
                        let (ct, st) = (cos_theta[rot], sin_theta[rot]);
                        let idx = scale * N_ROT * n_points + rot * n_points + outer;
                        points[idx] = PatternPoint {
                            x: (scale_radius * (ct * ca - st * sa)) as f32,
                            y: (scale_radius * (st * ca + ct * sa)) as f32,
                            sigma,
                        };
                    }
                    outer += 1;
                }
            }
        }

        // Pairs are sorted using scale 0, rotation 0 — the unrotated pattern at
        // its smallest — and the classification then holds for every scale.
        let (mut short_pairs, mut long_pairs) = (Vec::new(), Vec::new());
        let (d_min_sq, d_max_sq) = (d_min * d_min, d_max * d_max);
        for i in 1..n_points {
            for j in 0..i {
                let dx = points[j].x - points[i].x;
                let dy = points[j].y - points[i].y;
                let norm_sq = dx * dx + dy * dy;
                if norm_sq > d_min_sq {
                    long_pairs.push(LongPair {
                        i,
                        j,
                        weighted_dx: (f64::from(dx / norm_sq) * 2048.0 + 0.5) as i32,
                        weighted_dy: (f64::from(dy / norm_sq) * 2048.0 + 0.5) as i32,
                    });
                } else if norm_sq < d_max_sq {
                    short_pairs.push(ShortPair { i, j });
                }
            }
        }
        // `ceil(pairs / 128) * 4 * 4` — rounded up to whole 16-byte groups.
        let strings = (short_pairs.len() as f32 / 128.0).ceil() as usize * 16;

        BriskPattern { points, n_points, size_list, short_pairs, long_pairs, strings }
    }

    /// Would this keypoint's sampling pattern fit inside a `cols` x `rows`
    /// image?
    ///
    /// `cv::BRISK::detect` applies this too, not just `compute` — the border
    /// prune sits at the top of `computeDescriptorsAndOrDetect`, before the
    /// branch on whether descriptors were asked for. Skipping it here leaves
    /// roughly twice as many keypoints as cv2 returns, all of them near the
    /// edges where the pattern would hang off the image.
    #[must_use]
    pub fn fits(&self, kp: &KeyPoint, cols: i32, rows: i32) -> bool {
        let border = self.size_list[self.scale_for(kp.size)] as f32;
        kp.pt.0 >= border
            && kp.pt.0 < cols as f32 - border
            && kp.pt.1 >= border
            && kp.pt.1 < rows as f32 - border
    }

    /// The scale index upstream picks for a keypoint of this `size`.
    pub(crate) fn scale_for(&self, size: f32) -> usize {
        let lb_scalerange = SCALERANGE.ln() / std::f32::consts::LN_2;
        let basic_size06 = BASIC_SIZE * 0.6;
        let s = (SCALES as f32 / lb_scalerange * ((size / basic_size06).ln() / std::f32::consts::LN_2)
            + 0.5) as i32;
        (s.max(0) as usize).min(SCALES - 1)
    }
}

/// `BRISK_Impl::smoothedIntensity` — one pattern sample, in fixed point.
///
/// # Two regimes
///
/// Below half a pixel the "smoothing" is just bilinear interpolation at 1/1024.
/// Above it, the sample is the mean over a `2 * sigma` box with fractional
/// edges, computed as a weighted sum: corners get the product of both fractional
/// overlaps, edges one, the interior none.
///
/// Upstream computes the large case through an integral image; this sums the
/// box directly. The two are the same integer — the integral form only avoids
/// re-reading pixels — but the direct form is far easier to see is right.
fn smoothed_intensity(
    img: &[u8],
    cols: usize,
    rows: usize,
    key_x: f32,
    key_y: f32,
    p: PatternPoint,
) -> i32 {
    let xf = p.x + key_x;
    let yf = p.y + key_y;
    let x = xf as i32;
    let y = yf as i32;
    let sigma_half = p.sigma;
    let area = 4.0 * sigma_half * sigma_half;
    let at = |yy: i32, xx: i32| -> i32 {
        i32::from(img[(yy.clamp(0, rows as i32 - 1) as usize) * cols
            + xx.clamp(0, cols as i32 - 1) as usize])
    };

    if sigma_half < 0.5 {
        let r_x = ((xf - x as f32) * 1024.0) as i32;
        let r_y = ((yf - y as f32) * 1024.0) as i32;
        let (r_x_1, r_y_1) = (1024 - r_x, 1024 - r_y);
        // Note the corner order: upstream reads [0], [1], [step], [step+1] and
        // pairs them r_x*r_y with [step], not [step+1].
        let ret = r_x_1 * r_y_1 * at(y, x)
            + r_x * r_y_1 * at(y, x + 1)
            + r_x * r_y * at(y + 1, x)
            + r_x_1 * r_y * at(y + 1, x + 1);
        return (ret + 512) / 1024;
    }

    // `4194304.0` and `1024.0` are `double` literals, so both divisions widen.
    let scaling = (4_194_304.0f64 / f64::from(area)) as i32;
    let scaling2 = (f64::from(scaling as f32 * area) / 1024.0) as i32;
    if scaling2 == 0 {
        return 0;
    }

    let (x_1, x1) = (xf - sigma_half, xf + sigma_half);
    let (y_1, y1) = (yf - sigma_half, yf + sigma_half);
    let x_left = (x_1 + 0.5) as i32;
    let y_top = (y_1 + 0.5) as i32;
    let x_right = (x1 + 0.5) as i32;
    let y_bottom = (y1 + 0.5) as i32;
    let r_x_1 = x_left as f32 - x_1 + 0.5;
    let r_y_1 = y_top as f32 - y_1 + 0.5;
    let r_x1 = x1 - x_right as f32 + 0.5;
    let r_y1 = y1 - y_bottom as f32 + 0.5;
    let dx = x_right - x_left - 1;
    let dy = y_bottom - y_top - 1;

    let a = (r_x_1 * r_y_1 * scaling as f32) as i32;
    let b = (r_x1 * r_y_1 * scaling as f32) as i32;
    let c = (r_x1 * r_y1 * scaling as f32) as i32;
    let d = (r_x_1 * r_y1 * scaling as f32) as i32;
    let r_x_1_i = (r_x_1 * scaling as f32) as i32;
    let r_y_1_i = (r_y_1 * scaling as f32) as i32;
    let r_x1_i = (r_x1 * scaling as f32) as i32;
    let r_y1_i = (r_y1 * scaling as f32) as i32;

    // Corners, then the four edges, then the interior.
    let mut ret = a * at(y_top, x_left)
        + b * at(y_top, x_left + dx + 1)
        + c * at(y_top + dy + 1, x_left + dx + 1)
        + d * at(y_top + dy + 1, x_left);
    for i in 1..=dx {
        ret += r_y_1_i * at(y_top, x_left + i);
        ret += r_y1_i * at(y_top + dy + 1, x_left + i);
    }
    for j in 1..=dy {
        ret += r_x_1_i * at(y_top + j, x_left);
        ret += r_x1_i * at(y_top + j, x_left + dx + 1);
        for i in 1..=dx {
            ret += at(y_top + j, x_left + i) * scaling;
        }
    }
    (ret + scaling2 / 2) / scaling2
}

/// `cv::BRISK::compute(image, keypoints, descriptors)`.
///
/// Computes orientations and descriptors for keypoints that already exist —
/// upstream's `detectAndCompute` with `useProvidedKeypoints`. Keypoints whose
/// pattern would reach outside the image are **dropped**, so the returned
/// keypoint list can be shorter than the input; the descriptors correspond to
/// the keypoints that come back, not to the ones that went in.
pub fn brisk_compute<S: MatLike + ?Sized>(
    image: &S,
    pattern: &BriskPattern,
    keypoints: &[KeyPoint],
    upright: bool,
) -> Result<(Vec<KeyPoint>, Vec<Vec<u8>>)> {
    cv_assert!(image.channels() == 1, "BRISK requires a single-channel image");
    let (rows, cols) = (image.rows() as usize, image.cols() as usize);
    let mut buf = vec![0u8; rows * cols];
    for y in 0..rows {
        for x in 0..cols {
            buf[y * cols + x] = image.at_f64(y as i32, x as i32, 0)? as u8;
        }
    }

    let mut kept = Vec::new();
    let mut descs = Vec::new();
    let mut values = vec![0i32; pattern.n_points];

    for kp in keypoints {
        let scale = pattern.scale_for(kp.size);
        let border = pattern.size_list[scale] as f32;
        // `RoiPredicate`: half-open on the far edge.
        if kp.pt.0 < border
            || kp.pt.0 >= cols as f32 - border
            || kp.pt.1 < border
            || kp.pt.1 >= rows as f32 - border
        {
            continue;
        }
        let (x, y) = (kp.pt.0, kp.pt.1);
        let base = scale * N_ROT * pattern.n_points;

        // `atan2` in degrees, still signed — the sign matters below.
        let angle = if upright {
            -1.0f32
        } else {
            // Unrotated samples first, so the long pairs can vote.
            for (i, v) in values.iter_mut().enumerate() {
                *v = smoothed_intensity(&buf, cols, rows, x, y, pattern.points[base + i]);
            }
            let (mut dir0, mut dir1) = (0i32, 0i32);
            for lp in &pattern.long_pairs {
                let delta_t = values[lp.i] - values[lp.j];
                dir0 += delta_t * lp.weighted_dx / 1024;
                dir1 += delta_t * lp.weighted_dy / 1024;
            }
            f64::from(dir1).atan2(f64::from(dir0)) as f32 / std::f32::consts::PI * 180.0
        };

        // Quantised index into the rotation table.
        //
        // Computed from the **signed** angle, and the cast truncates toward
        // zero: at -90 degrees this is `(int)(-255.5)`, which is -255, then
        // wraps to 769. Normalising the angle into `0..360` first and then
        // quantising gives 768 — one slot out, and a different descriptor.
        let theta = if upright {
            0usize
        } else {
            let mut t = (N_ROT as f64 * (f64::from(angle) / 360.0) + 0.5) as i32;
            if t < 0 {
                t += N_ROT as i32;
            }
            if t >= N_ROT as i32 {
                t -= N_ROT as i32;
            }
            t as usize
        };

        let mut out = *kp;
        // Only *now* is the angle folded into `0..360`, and it is the raw
        // arctangent that is reported, not the quantised slot.
        out.angle = if angle < 0.0 { angle + 360.0 } else { angle };

        // Re-sample with the pattern rotated to the keypoint's orientation.
        let rbase = base + theta * pattern.n_points;
        for (i, v) in values.iter_mut().enumerate() {
            *v = smoothed_intensity(&buf, cols, rows, x, y, pattern.points[rbase + i]);
        }

        let mut desc = vec![0u8; pattern.strings];
        for (bit, sp) in pattern.short_pairs.iter().enumerate() {
            if values[sp.i] > values[sp.j] {
                desc[bit >> 3] |= 1 << (bit & 7);
            }
        }
        kept.push(out);
        descs.push(desc);
    }
    Ok((kept, descs))
}

#[cfg(test)]
mod tests {
    use super::*;


    /// cv2 4.13.0's BRISK keypoints on the deterministic texture, as
    /// `(x, y, size)`. Fed straight into [`brisk_compute`] so that only the
    /// descriptor is under test — the detector is a separate concern, and a
    /// difference there would otherwise mask a difference here.
    #[rustfmt::skip]
    #[allow(clippy::excessive_precision)]
    const CV2_KEYPOINTS: [(f32, f32, f32); 165] = [
        (21.0, 67.0, 12.0), (21.069469451904297, 27.100061416625977, 12.94285774230957), (21.98082733154297, 20.931068420410156, 12.0),
        (23.545339584350586, 75.55000305175781, 13.881261825561523), (25.97915267944336, 26.60466766357422, 12.55886459350586), (25.995193481445312, 109.49494934082031, 13.363636016845703),
        (26.03449249267578, 132.88636779785156, 12.771429061889648), (27.708703994750977, 71.77057647705078, 12.745124816894531), (27.93405532836914, 57.52968978881836, 12.488948822021484),
        (28.58013343811035, 65.45498657226562, 11.571428298950195), (28.635282516479492, 61.24909591674805, 13.216217041015625), (29.926286697387695, 117.2685775756836, 13.414310455322266),
        (30.019973754882812, 117.8864517211914, 12.479324340820312), (30.15110206604004, 44.07551574707031, 13.052716255187988), (30.333751678466797, 39.78464126586914, 14.978578567504883),
        (30.358898162841797, 39.584468841552734, 14.577354431152344), (30.91656494140625, 79.53667449951172, 11.554071426391602), (31.647714614868164, 101.06485748291016, 13.551724433898926),
        (32.05857467651367, 97.77082824707031, 12.535713195800781), (32.7170524597168, 62.04509735107422, 14.129467964172363), (32.79618453979492, 36.89248275756836, 12.466667175292969),
        (33.02613830566406, 86.96995544433594, 12.233293533325195), (33.03448486328125, 86.71083068847656, 13.941177368164062), (36.10165023803711, 47.95307922363281, 12.782608032226562),
        (36.670936584472656, 34.3604621887207, 13.007081985473633), (41.61457443237305, 93.65423583984375, 13.559999465942383), (42.006526947021484, 92.5207290649414, 23.378650665283203),
        (43.95769119262695, 73.59712982177734, 12.777777671813965), (44.66267395019531, 63.64030456542969, 17.325315475463867), (45.038909912109375, 59.61083221435547, 13.166150093078613),
        (45.427486419677734, 129.98631286621094, 14.362458229064941), (45.474117279052734, 99.16143798828125, 17.23763656616211), (47.071231842041016, 82.98397064208984, 12.571429252624512),
        (48.594661712646484, 49.95589828491211, 12.777777671813965), (50.32946014404297, 98.01114654541016, 13.120498657226562), (52.508384704589844, 121.14463806152344, 18.724864959716797),
        (52.587989807128906, 49.57114791870117, 14.930501937866211), (53.00434875488281, 50.610267639160156, 18.929729461669922), (53.06138610839844, 121.56536865234375, 19.57318687438965),
        (53.318721771240234, 135.77890014648438, 13.337362289428711), (55.00522232055664, 90.52383422851562, 14.175375938415527), (56.609344482421875, 22.95875358581543, 11.760000228881836),
        (57.60979080200195, 106.0575180053711, 13.151954650878906), (57.87212371826172, 93.84089660644531, 11.780488014221191), (61.05057144165039, 125.87577819824219, 12.076923370361328),
        (61.339073181152344, 38.36966323852539, 13.121838569641113), (61.717437744140625, 50.117156982421875, 12.0), (61.75484085083008, 57.74553680419922, 13.507621765136719),
        (62.05342483520508, 96.75391387939453, 14.726802825927734), (62.56483840942383, 25.125139236450195, 12.333333969116211), (62.74973678588867, 137.18809509277344, 13.408163070678711),
        (62.784000396728516, 82.89801025390625, 13.06894302368164), (62.83904266357422, 72.81458282470703, 12.230768203735352), (63.0, 137.4674835205078, 12.0),
        (63.29656982421875, 46.966087341308594, 11.862000465393066), (64.01982879638672, 34.571617126464844, 13.235294342041016), (64.13397979736328, 116.88140106201172, 13.838708877563477),
        (64.300537109375, 131.47113037109375, 16.010963439941406), (64.73375701904297, 132.5282440185547, 12.707073211669922), (64.76528930664062, 116.68547058105469, 20.794227600097656),
        (65.31499481201172, 97.46158599853516, 29.944316864013672), (65.52560424804688, 125.21041870117188, 16.935483932495117), (65.69757080078125, 126.68584442138672, 13.269231796264648),
        (66.11536407470703, 63.61783218383789, 13.449603080749512), (66.83189392089844, 77.77586364746094, 13.344827651977539), (67.95685577392578, 72.98704528808594, 12.8490571975708),
        (69.10675048828125, 39.436859130859375, 14.784143447875977), (70.97708129882812, 79.95834350585938, 15.399999618530273), (71.3981704711914, 80.44226837158203, 20.559795379638672),
        (72.22013854980469, 36.315818786621094, 18.0), (73.1844482421875, 67.15065002441406, 12.277996063232422), (75.54991149902344, 54.41276168823242, 14.620309829711914),
        (76.8684310913086, 27.281139373779297, 13.30769157409668), (82.07009887695312, 34.18627166748047, 11.217391967773438), (83.5579833984375, 125.61679077148438, 13.79658317565918),
        (83.63052368164062, 125.65815734863281, 12.302690505981445), (84.03526306152344, 85.18724060058594, 16.504310607910156), (84.7674789428711, 106.58369445800781, 12.766679763793945),
        (85.58405303955078, 54.800010681152344, 13.751869201660156), (86.08551788330078, 74.14474487304688, 14.886690139770508), (86.32728576660156, 63.098506927490234, 22.47159194946289),
        (86.44990539550781, 74.00208282470703, 19.65216064453125), (86.64157104492188, 22.896108627319336, 13.266666412353516), (87.76081085205078, 65.70022583007812, 12.409090042114258),
        (88.5954360961914, 129.47073364257812, 14.234781265258789), (90.48282623291016, 51.09770202636719, 23.956985473632812), (90.99423217773438, 91.81146240234375, 16.23928451538086),
        (92.0029296875, 72.35379791259766, 24.40540313720703), (92.54457092285156, 125.7420654296875, 17.02191925048828), (92.75714874267578, 127.52594757080078, 12.0455904006958),
        (93.34794616699219, 101.96441650390625, 14.236141204833984), (93.59878540039062, 101.7735366821289, 13.440224647521973), (94.62552642822266, 106.16339874267578, 12.222223281860352),
        (94.99807739257812, 29.90190887451172, 17.58854866027832), (95.8376693725586, 112.54551696777344, 18.15781021118164), (96.39939880371094, 35.30012130737305, 13.049999237060547),
        (96.58953094482422, 71.06269836425781, 11.681888580322266), (97.89189147949219, 114.35380554199219, 12.061168670654297), (97.97280883789062, 77.08588409423828, 13.410469055175781),
        (98.57659149169922, 65.23457336425781, 11.938776016235352), (99.60863494873047, 107.07938385009766, 13.48392105102539), (100.77252197265625, 60.73124694824219, 16.327756881713867),
        (100.92245483398438, 125.59584045410156, 11.800000190734863), (101.50469970703125, 93.48907470703125, 20.076923370361328), (101.94940185546875, 60.9404182434082, 22.862070083618164),
        (102.12068176269531, 83.81202697753906, 14.079560279846191), (102.19358825683594, 67.58468627929688, 13.076372146606445), (102.4943618774414, 95.64887237548828, 12.199999809265137),
        (102.62467956542969, 83.90250396728516, 12.968189239501953), (102.96448516845703, 29.0454044342041, 12.773059844970703), (103.42329406738281, 112.75361633300781, 14.143754959106445),
        (103.43216705322266, 21.5316219329834, 13.071428298950195), (104.00452423095703, 63.122257232666016, 11.625), (106.1505126953125, 109.2618179321289, 25.5),
        (106.20171356201172, 110.32395935058594, 13.181818008422852), (106.67558288574219, 94.01893615722656, 15.0), (106.7681655883789, 93.3888931274414, 25.96868896484375),
        (107.26969909667969, 76.95390319824219, 14.95473861694336), (107.99517822265625, 76.86651611328125, 13.2149076461792), (108.3165512084961, 59.73145294189453, 12.125951766967773),
        (110.33196258544922, 56.099395751953125, 14.481260299682617), (110.38978576660156, 75.125, 15.0), (110.71049499511719, 75.00956726074219, 15.920448303222656),
        (110.80410766601562, 37.76681137084961, 19.0), (111.22576141357422, 107.70771789550781, 12.319149017333984), (111.98579406738281, 93.8552017211914, 13.038759231567383),
        (113.85391998291016, 124.51310729980469, 13.08378791809082), (114.1391830444336, 73.6380844116211, 11.7619047164917), (114.61915588378906, 82.52894592285156, 12.833333015441895),
        (115.40641021728516, 131.66989135742188, 11.888888359069824), (115.85359191894531, 63.31818389892578, 12.257143020629883), (116.07015991210938, 36.963600158691406, 13.302160263061523),
        (116.9833984375, 55.11819076538086, 11.461538314819336), (117.20088958740234, 137.7445831298828, 12.322474479675293), (119.51046752929688, 68.77872467041016, 16.040363311767578),
        (119.60469818115234, 68.69186401367188, 13.391664505004883), (119.6257095336914, 133.1311798095703, 13.719082832336426), (120.62290954589844, 134.09426879882812, 12.461538314819336),
        (120.64881134033203, 46.0230598449707, 12.358184814453125), (122.0078125, 105.66947174072266, 19.200000762939453), (122.05409240722656, 24.616628646850586, 11.739130020141602),
        (123.9351806640625, 135.219482421875, 13.00957202911377), (124.42304229736328, 98.06883239746094, 17.25629997253418), (124.45561218261719, 98.26197052001953, 14.520000457763672),
        (124.68128967285156, 37.73309326171875, 11.91891860961914), (124.87173461914062, 126.13008880615234, 12.789473533630371), (125.20384216308594, 31.170793533325195, 12.63636302947998),
        (125.67185974121094, 59.27543258666992, 12.0), (126.80751037597656, 133.1904754638672, 12.349925994873047), (126.8117904663086, 53.53003692626953, 13.723403930664062),
        (126.89437103271484, 67.83927154541016, 18.494800567626953), (127.80323028564453, 37.05791473388672, 13.058822631835938), (128.19015502929688, 24.967660903930664, 13.941177368164062),
        (128.7857666015625, 120.74061584472656, 12.97012710571289), (128.9259033203125, 120.81288146972656, 12.585935592651367), (129.1795654296875, 106.19440460205078, 13.064516067504883),
        (129.83010864257812, 52.80174255371094, 15.828376770019531), (130.46595764160156, 53.42240905761719, 12.41084098815918), (131.56753540039062, 113.7129898071289, 13.161968231201172),
        (133.61509704589844, 98.03748321533203, 12.199748039245605), (133.77557373046875, 98.1019287109375, 12.991310119628906), (135.83883666992188, 130.484130859375, 13.5),
        (136.1625213623047, 126.7918930053711, 12.272727012634277), (137.60421752929688, 97.82062530517578, 12.18250846862793), (137.65805053710938, 33.31563186645508, 11.543179512023926),
    ];

    /// Descriptors and orientations are bit-exact against
    /// `cv2.BRISK_create().compute()`: all 165 angles identical, all 165
    /// descriptors identical, 0 differing bits of 84480.
    ///
    /// This is only true because the pattern is built at upstream's *float
    /// widths*. `rList` and `scaleList_` are `float` arrays, so their products
    /// round to `f32` before widening; computing the same expressions in `f64`
    /// throughout moves a handful of samples by one intensity level, which was
    /// enough to shift four orientations by up to 0.04 degrees and flip 9 bits.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn descriptors_match_cv2_on_its_own_keypoints() {
        /// cv2's first descriptor, in full.
        const CV2_DESC0: [u8; 16] = [
            79, 254, 255, 151, 17, 32, 193, 164, 0, 219, 19, 151, 208, 134, 27, 127,
        ];
        /// FNV-1a over every descriptor byte, in keypoint order.
        const CV2_DESC_HASH: u64 = 0x7c8c_3724_eef3_3c3d;
        const CV2_ANGLE0: f32 = 53.028423;

        let img = crate::akaze::tests::texture(160, 160);
        let kps: Vec<KeyPoint> = CV2_KEYPOINTS
            .iter()
            .map(|&(x, y, size)| KeyPoint {
                pt: (x, y),
                size,
                angle: -1.0,
                response: 0.0,
                octave: 0,
                class_id: -1,
            })
            .collect();

        let pattern = BriskPattern::new(1.0);
        let (out, descs) = brisk_compute(&img, &pattern, &kps, false).unwrap();
        assert_eq!(out.len(), CV2_KEYPOINTS.len(), "cv2 keeps all of these");
        assert!((out[0].angle - CV2_ANGLE0).abs() < 1e-4, "angle {}", out[0].angle);
        assert_eq!(descs[0][..16], CV2_DESC0, "first descriptor differs from cv2's");

        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for b in descs.iter().flatten() {
            hash = (hash ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        assert_eq!(hash, CV2_DESC_HASH, "descriptor bytes differ from cv2's somewhere");
    }

    #[test]
    fn the_pattern_has_upstreams_shape() {
        let p = BriskPattern::new(1.0);
        assert_eq!(p.n_points, 60, "1 + 10 + 14 + 15 + 20 sample points");
        // 512 bits is BRISK's advertised length, and it comes from the pair
        // count rounding up to whole 16-byte groups, not from a constant.
        assert_eq!(p.strings, BRISK_BYTES);
        assert!(
            p.short_pairs.len() > 128 && p.short_pairs.len() <= 512,
            "{} short pairs would not fill a 512-bit descriptor",
            p.short_pairs.len()
        );
        assert!(!p.long_pairs.is_empty(), "no long pairs means no orientation");
    }

    /// The thresholds are not scaled by the 0.85 that the radii are. If they
    /// were, the pair split would move — this pins that they do not.
    #[test]
    fn the_pair_split_uses_unscaled_thresholds() {
        let p = BriskPattern::new(1.0);
        assert_eq!((p.short_pairs.len(), p.long_pairs.len()), (512, 870));
    }

    #[test]
    fn the_scale_index_tracks_keypoint_size() {
        let p = BriskPattern::new(1.0);
        // basicSize_ * 0.6 = 7.2 is scale 0 by construction.
        assert_eq!(p.scale_for(7.2), 0);
        assert!(p.scale_for(1.0) == 0, "sizes below the range clamp to 0");
        assert_eq!(p.scale_for(1e6), SCALES - 1, "and above it to the last");
        assert!(p.scale_for(30.0) > p.scale_for(15.0), "must be monotonic");
    }
}
