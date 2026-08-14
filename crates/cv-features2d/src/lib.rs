//! `cv::features2d` for `opencv-no-c` — keypoints, descriptors and matching.
//!
//! # Scope
//!
//! FAST and Harris/Shi-Tomasi detection, an ORB-style oriented BRIEF
//! descriptor, and brute-force matching over Hamming or L2 distance. That is
//! the combination that actually gets used: a detector, a binary descriptor
//! and a matcher.
//!
//! SIFT and its relatives are not here. They are a substantially larger job —
//! scale-space pyramids, orientation histograms, tri-linear descriptor
//! interpolation — and pretending otherwise by shipping a half-version would
//! be worse than the gap.

#![warn(missing_docs)]

pub mod akaze;
pub mod brisk;
pub mod brisk_pattern;
pub mod brisk_scalespace;
pub mod detectors;
pub mod feature2d;
pub mod kaze;
pub mod matcher;
mod plane;
pub mod regions;
pub mod sift;

use cv_core::{cv_assert, Mat, MatLike, Point, Result, Scalar};

pub use cv_core::{DMatch, KeyPoint};

/// The 16 pixels of the Bresenham circle of radius 3, in upstream's order:
/// starting directly *below* the centre and going anticlockwise in image
/// coordinates. A different starting point traces the same ring and finds the
/// same arcs, but `cornerScore` indexes this order directly.
const CIRCLE: [(i32, i32); 16] = [
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 1),
    (3, 0),
    (3, -1),
    (2, -2),
    (1, -3),
    (0, -3),
    (-1, -3),
    (-2, -2),
    (-3, -1),
    (-3, 0),
    (-3, 1),
    (-2, 2),
    (-1, 3),
];

/// `cv::FAST(image, keypoints, threshold, nonmaxSuppression)`.
///
/// A corner is a pixel with at least `n` contiguous circle neighbours all
/// brighter than `centre + threshold` or all darker than `centre - threshold`.
/// `n` of 9 is upstream's `TYPE_9_16` default.
pub fn fast<S: MatLike + ?Sized>(
    image: &S,
    threshold: i32,
    nonmax_suppression: bool,
) -> Result<Vec<KeyPoint>> {
    fast_n(image, threshold, nonmax_suppression, 9)
}

/// [`fast`] with an explicit contiguous-arc length.
pub fn fast_n<S: MatLike + ?Sized>(
    image: &S,
    threshold: i32,
    nonmax_suppression: bool,
    n: usize,
) -> Result<Vec<KeyPoint>> {
    cv_assert!(image.channels() == 1, "FAST requires a single-channel image");
    cv_assert!((9..=12).contains(&n), "arc length must be 9..12, got {n}");
    let (h, w) = (image.rows(), image.cols());
    let threshold = threshold.clamp(0, 255);
    let k_arc = n - 1;

    let mut score = vec![0i32; (h * w) as usize];
    let mut corners: Vec<(i32, i32)> = Vec::new();

    for y in 3..h - 3 {
        for x in 3..w - 3 {
            let v = image.at_f64(y, x, 0)? as i32;
            let mut ring = [0i32; 16];
            for (k, &(dx, dy)) in CIRCLE.iter().enumerate() {
                ring[k] = image.at_f64(y + dy, x + dx, 0)? as i32;
            }
            // Upstream's rejection test, and it has to be this one. For a
            // 9-arc out of 16, of any two *diametrically opposite* ring points
            // at least one lies in the arc — so ANDing the eight opposite
            // pairs is an exact necessary condition. Requiring three of the
            // four compass points instead is not: a 9-arc covers either two or
            // three of them, so the two-point case was being thrown away, and
            // this found 1129 corners where cv2 found 1187.
            let tab = |p: i32| -> u8 {
                if p - v < -threshold {
                    1
                } else if p - v > threshold {
                    2
                } else {
                    0
                }
            };
            let mut d = tab(ring[0]) | tab(ring[8]);
            if d == 0 {
                continue;
            }
            d &= tab(ring[2]) | tab(ring[10]);
            d &= tab(ring[4]) | tab(ring[12]);
            d &= tab(ring[6]) | tab(ring[14]);
            if d == 0 {
                continue;
            }
            d &= tab(ring[1]) | tab(ring[9]);
            d &= tab(ring[3]) | tab(ring[11]);
            d &= tab(ring[5]) | tab(ring[13]);
            d &= tab(ring[7]) | tab(ring[15]);
            if d == 0 {
                continue;
            }

            // The ring is walked one and a half times round so a wrapping arc
            // is found without any modular bookkeeping.
            let arc = |brighter: bool| -> bool {
                let vt = if brighter { v + threshold } else { v - threshold };
                let mut count = 0usize;
                for k in 0..16 + k_arc + 1 {
                    let p = ring[k % 16];
                    if (brighter && p > vt) || (!brighter && p < vt) {
                        count += 1;
                        if count > k_arc {
                            return true;
                        }
                    } else {
                        count = 0;
                    }
                }
                false
            };
            if (d & 1 != 0 && arc(false)) || (d & 2 != 0 && arc(true)) {
                corners.push((x, y));
                if nonmax_suppression {
                    score[(y * w + x) as usize] = corner_score(&ring, v, threshold);
                }
            }
        }
    }

    let mut out = Vec::new();
    for &(x, y) in &corners {
        let s = score[(y * w + x) as usize];
        if nonmax_suppression {
            // Strictly greater than all eight neighbours; a non-corner scores
            // zero, so it never blocks one.
            let beaten = (-1..=1).any(|dy| {
                (-1..=1).any(|dx| {
                    (dx != 0 || dy != 0) && score[((y + dy) * w + x + dx) as usize] >= s
                })
            });
            if beaten {
                continue;
            }
        }
        out.push(KeyPoint {
            pt: (x as f32, y as f32),
            size: 7.0,
            angle: -1.0,
            // With suppression off upstream reports zero: the score is only
            // computed when something is going to compare it.
            response: s as f32,
            octave: 0,
            class_id: -1,
        });
    }
    Ok(out)
}

/// `cv::cornerScore<16>` — the largest threshold at which the point is still a
/// corner.
///
/// Not the sum of absolute ring differences, which is the natural stand-in and
/// is *not* monotone in this: the two disagree about which of two neighbouring
/// corners survives suppression, and the score is part of the API besides —
/// callers sort keypoints by it.
fn corner_score(ring: &[i32; 16], v: i32, threshold: i32) -> i32 {
    let d: Vec<i32> = (0..25).map(|k| v - ring[k % 16]).collect();
    let mut a0 = threshold;
    for k in (0..16).step_by(2) {
        let mut a = d[k + 1].min(d[k + 2]).min(d[k + 3]);
        if a <= a0 {
            continue;
        }
        a = a.min(d[k + 4]).min(d[k + 5]).min(d[k + 6]).min(d[k + 7]).min(d[k + 8]);
        a0 = a0.max(a.min(d[k])).max(a.min(d[k + 9]));
    }
    let mut b0 = -a0;
    for k in (0..16).step_by(2) {
        let mut b = d[k + 1].max(d[k + 2]).max(d[k + 3]).max(d[k + 4]).max(d[k + 5]);
        if b >= b0 {
            continue;
        }
        b = b.max(d[k + 6]).max(d[k + 7]).max(d[k + 8]);
        b0 = b0.min(b.max(d[k])).min(b.max(d[k + 9]));
    }
    -b0 - 1
}


/// The intensity centroid orientation of a patch — ORB's `IC_Angle`.
///
/// The angle of the vector from the patch centre to its intensity centroid.
/// This is what makes BRIEF rotation-invariant: the sampling pattern is
/// steered by it.
/// `u_max` — the half-width of each row of the circular patch, built so the
/// patch is symmetric about both diagonals.
///
/// A plain `dx*dx + dy*dy <= r*r` test is *not* the same shape: it is
/// symmetric about the axes but not about the diagonals, and the moment it
/// feeds is a direction estimate, so the asymmetry biases the angle.
fn u_max(half: i32) -> Vec<i32> {
    let mut umax = vec![0i32; (half + 2) as usize];
    let vmax = ((f64::from(half) * 2f64.sqrt() / 2.0) + 1.0).floor() as i32;
    let vmin = (f64::from(half) * 2f64.sqrt() / 2.0).ceil() as i32;
    for v in 0..=vmax {
        umax[v as usize] =
            cv_core::saturate::cv_round((f64::from(half * half) - f64::from(v * v)).sqrt());
    }
    let mut v0 = 0usize;
    let mut v = half;
    while v >= vmin {
        while umax[v0] == umax[v0 + 1] {
            v0 += 1;
        }
        umax[v as usize] = v0 as i32;
        v0 += 1;
        v -= 1;
    }
    umax
}

/// `cv::IC_Angle` — the intensity-centroid orientation, in degrees `[0, 360)`.
fn ic_angle<S: MatLike + ?Sized>(img: &S, x: i32, y: i32, r: i32) -> Result<f32> {
    let umax = u_max(r);
    let (mut m01, mut m10) = (0i64, 0i64);
    // The centre row on its own, then the rows in pairs.
    for u in -r..=r {
        m10 += i64::from(u) * img.at_f64(y, x + u, 0)? as i64;
    }
    for v in 1..=r {
        let mut v_sum = 0i64;
        let d = umax[v as usize];
        for u in -d..=d {
            let plus = img.at_f64(y + v, x + u, 0)? as i64;
            let minus = img.at_f64(y - v, x + u, 0)? as i64;
            v_sum += plus - minus;
            m10 += i64::from(u) * (plus + minus);
        }
        m01 += i64::from(v) * v_sum;
    }
    // `fastAtan2` returns `[0, 360)`, not `(-180, 180]`.
    let a = (m01 as f64).atan2(m10 as f64).to_degrees();
    Ok((if a < 0.0 { a + 360.0 } else { a }) as f32)
}

/// A 256-bit binary descriptor.
pub type Descriptor = [u8; 32];

/// `bit_pattern_31_` — upstream's BRIEF sampling pattern: 256 pairs of points
/// inside a 31x31 patch, stored as `(x1, y1, x2, y2)`.
///
/// This table is **learned**, not generated. The pairs were chosen for low
/// mutual correlation and high variance on a training set, and the source
/// records the measured mean and correlation beside each one. A pattern drawn
/// from a seeded RNG has the same shape and produces descriptors that match
/// nothing — not cv2's, and not another build of the same port with a
/// different seed.
#[rustfmt::skip]
const BIT_PATTERN_31: [i8; 1024] = [
    8, -3, 9, 5, 4, 2, 7, -12,
    -11, 9, -8, 2, 7, -12, 12, -13,
    2, -13, 2, 12, 1, -7, 1, 6,
    -2, -10, -2, -4, -13, -13, -11, -8,
    -13, -3, -12, -9, 10, 4, 11, 9,
    -13, -8, -8, -9, -11, 7, -9, 12,
    7, 7, 12, 6, -4, -5, -3, 0,
    -13, 2, -12, -3, -9, 0, -7, 5,
    12, -6, 12, -1, -3, 6, -2, 12,
    -6, -13, -4, -8, 11, -13, 12, -8,
    4, 7, 5, 1, 5, -3, 10, -3,
    3, -7, 6, 12, -8, -7, -6, -2,
    -2, 11, -1, -10, -13, 12, -8, 10,
    -7, 3, -5, -3, -4, 2, -3, 7,
    -10, -12, -6, 11, 5, -12, 6, -7,
    5, -6, 7, -1, 1, 0, 4, -5,
    9, 11, 11, -13, 4, 7, 4, 12,
    2, -1, 4, 4, -4, -12, -2, 7,
    -8, -5, -7, -10, 4, 11, 9, 12,
    0, -8, 1, -13, -13, -2, -8, 2,
    -3, -2, -2, 3, -6, 9, -4, -9,
    8, 12, 10, 7, 0, 9, 1, 3,
    7, -5, 11, -10, -13, -6, -11, 0,
    10, 7, 12, 1, -6, -3, -6, 12,
    10, -9, 12, -4, -13, 8, -8, -12,
    -13, 0, -8, -4, 3, 3, 7, 8,
    5, 7, 10, -7, -1, 7, 1, -12,
    3, -10, 5, 6, 2, -4, 3, -10,
    -13, 0, -13, 5, -13, -7, -12, 12,
    -13, 3, -11, 8, -7, 12, -4, 7,
    6, -10, 12, 8, -9, -1, -7, -6,
    -2, -5, 0, 12, -12, 5, -7, 5,
    3, -10, 8, -13, -7, -7, -4, 5,
    -3, -2, -1, -7, 2, 9, 5, -11,
    -11, -13, -5, -13, -1, 6, 0, -1,
    5, -3, 5, 2, -4, -13, -4, 12,
    -9, -6, -9, 6, -12, -10, -8, -4,
    10, 2, 12, -3, 7, 12, 12, 12,
    -7, -13, -6, 5, -4, 9, -3, 4,
    7, -1, 12, 2, -7, 6, -5, 1,
    -13, 11, -12, 5, -3, 7, -2, -6,
    7, -8, 12, -7, -13, -7, -11, -12,
    1, -3, 12, 12, 2, -6, 3, 0,
    -4, 3, -2, -13, -1, -13, 1, 9,
    7, 1, 8, -6, 1, -1, 3, 12,
    9, 1, 12, 6, -1, -9, -1, 3,
    -13, -13, -10, 5, 7, 7, 10, 12,
    12, -5, 12, 9, 6, 3, 7, 11,
    5, -13, 6, 10, 2, -12, 2, 3,
    3, 8, 4, -6, 2, 6, 12, -13,
    9, -12, 10, 3, -8, 4, -7, 9,
    -11, 12, -4, -6, 1, 12, 2, -8,
    6, -9, 7, -4, 2, 3, 3, -2,
    6, 3, 11, 0, 3, -3, 8, -8,
    7, 8, 9, 3, -11, -5, -6, -4,
    -10, 11, -5, 10, -5, -8, -3, 12,
    -10, 5, -9, 0, 8, -1, 12, -6,
    4, -6, 6, -11, -10, 12, -8, 7,
    4, -2, 6, 7, -2, 0, -2, 12,
    -5, -8, -5, 2, 7, -6, 10, 12,
    -9, -13, -8, -8, -5, -13, -5, -2,
    8, -8, 9, -13, -9, -11, -9, 0,
    1, -8, 1, -2, 7, -4, 9, 1,
    -2, 1, -1, -4, 11, -6, 12, -11,
    -12, -9, -6, 4, 3, 7, 7, 12,
    5, 5, 10, 8, 0, -4, 2, 8,
    -9, 12, -5, -13, 0, 7, 2, 12,
    -1, 2, 1, 7, 5, 11, 7, -9,
    3, 5, 6, -8, -13, -4, -8, 9,
    -5, 9, -3, -3, -4, -7, -3, -12,
    6, 5, 8, 0, -7, 6, -6, 12,
    -13, 6, -5, -2, 1, -10, 3, 10,
    4, 1, 8, -4, -2, -2, 2, -13,
    2, -12, 12, 12, -2, -13, 0, -6,
    4, 1, 9, 3, -6, -10, -3, -5,
    -3, -13, -1, 1, 7, 5, 12, -11,
    4, -2, 5, -7, -13, 9, -9, -5,
    7, 1, 8, 6, 7, -8, 7, 6,
    -7, -4, -7, 1, -8, 11, -7, -8,
    -13, 6, -12, -8, 2, 4, 3, 9,
    10, -5, 12, 3, -6, -5, -6, 7,
    8, -3, 9, -8, 2, -12, 2, 8,
    -11, -2, -10, 3, -12, -13, -7, -9,
    -11, 0, -10, -5, 5, -3, 11, 8,
    -2, -13, -1, 12, -1, -8, 0, 9,
    -13, -11, -12, -5, -10, -2, -10, 11,
    -3, 9, -2, -13, 2, -3, 3, 2,
    -9, -13, -4, 0, -4, 6, -3, -10,
    -4, 12, -2, -7, -6, -11, -4, 9,
    6, -3, 6, 11, -13, 11, -5, 5,
    11, 11, 12, 6, 7, -5, 12, -2,
    -1, 12, 0, 7, -4, -8, -3, -2,
    -7, 1, -6, 7, -13, -12, -8, -13,
    -7, -2, -6, -8, -8, 5, -6, -9,
    -5, -1, -4, 5, -13, 7, -8, 10,
    1, 5, 5, -13, 1, 0, 10, -13,
    9, 12, 10, -1, 5, -8, 10, -9,
    -1, 11, 1, -13, -9, -3, -6, 2,
    -1, -10, 1, 12, -13, 1, -8, -10,
    8, -11, 10, -6, 2, -13, 3, -6,
    7, -13, 12, -9, -10, -10, -5, -7,
    -10, -8, -8, -13, 4, -6, 8, 5,
    3, 12, 8, -13, -4, 2, -3, -3,
    5, -13, 10, -12, 4, -13, 5, -1,
    -9, 9, -4, 3, 0, 3, 3, -9,
    -12, 1, -6, 1, 3, 2, 4, -8,
    -10, -10, -10, 9, 8, -13, 12, 12,
    -8, -12, -6, -5, 2, 2, 3, 7,
    10, 6, 11, -8, 6, 8, 8, -12,
    -7, 10, -6, 5, -3, -9, -3, 9,
    -1, -13, -1, 5, -3, -7, -3, 4,
    -8, -2, -8, 3, 4, 2, 12, 12,
    2, -5, 3, 11, 6, -9, 11, -13,
    3, -1, 7, 12, 11, -1, 12, 4,
    -3, 0, -3, 6, 4, -11, 4, 12,
    2, -4, 2, 1, -10, -6, -8, 1,
    -13, 7, -11, 1, -13, 12, -11, -13,
    6, 0, 11, -13, 0, -1, 1, 4,
    -13, 3, -9, -2, -9, 8, -6, -3,
    -13, -6, -8, -2, 5, -9, 8, 10,
    2, 7, 3, -9, -1, -6, -1, -1,
    9, 5, 11, -2, 11, -3, 12, -8,
    3, 0, 3, 5, -1, 4, 0, 10,
    3, -6, 4, 5, -13, 0, -10, 5,
    5, 8, 12, 11, 8, 9, 9, -6,
    7, -4, 8, -12, -10, 4, -10, 9,
    7, 3, 12, 4, 9, -7, 10, -2,
    7, 0, 12, -2, -1, -6, 0, -11,
];

/// `cv::ORB::compute(image, keypoints, descriptors)`.
///
/// Oriented BRIEF: each keypoint's sampling pattern is rotated by its intensity
/// centroid angle, so the descriptor survives in-plane rotation. Keypoints too
/// close to the border to sample are dropped, and the returned keypoint list is
/// filtered to match.
pub fn compute_orb<S: MatLike + ?Sized>(
    image: &S,
    keypoints: &[KeyPoint],
) -> Result<(Vec<KeyPoint>, Vec<Descriptor>)> {
    cv_assert!(image.channels() == 1, "ORB descriptors need a single-channel image");
    let (h, w) = (image.rows(), image.cols());
    // Upstream's `border = max(edgeThreshold, max(patchSize/2,
    // HARRIS_BLOCK_SIZE/2)) + 1` = max(31, max(15, 4)) + 1 with the default
    // parameters. A keypoint nearer the edge than this is *dropped*, not
    // clamped — the 31x31 patch and the 31-pixel edge threshold both have to
    // fit, and a smaller margin silently describes points from truncated
    // neighbourhoods.
    const MARGIN: i32 = 32;
    // Upstream blurs every pyramid level with a 7x7 sigma-2 Gaussian before
    // sampling. BRIEF compares single pixels, so without the blur the
    // descriptor is dominated by sensor noise and is not repeatable between
    // two views of the same corner — which is the property it exists for.
    let blurred = cv_imgproc::filter::gaussian_blur(
        image,
        cv_core::Size::new(7, 7),
        2.0,
        2.0,
        cv_core::BorderType::Reflect101,
    )?;
    let image = &blurred;
    let mut kps = Vec::new();
    let mut descs = Vec::new();

    for kp in keypoints {
        let (x, y) = (kp.pt.0.round() as i32, kp.pt.1.round() as i32);
        if x < MARGIN || y < MARGIN || x >= w - MARGIN || y >= h - MARGIN {
            continue;
        }
        // The keypoint's *own* angle, as upstream uses. `cv::ORB::compute`
        // does not compute orientations — only `detectAndCompute` does, and
        // only for the keypoints it found itself. A caller who hands over
        // keypoints with the default angle of -1 gets descriptors sampled at
        // -1 degrees, and overwriting that with a fresh orientation gives a
        // different descriptor for the same input.
        let angle = kp.angle;
        // `angle *= (float)(CV_PI/180.f)` then `(float)cos(angle)`: the
        // multiplier is rounded to f32 first, the trig is evaluated in f64 and
        // the result rounded back.
        let a = f64::from(angle * (core::f32::consts::PI / 180.0));
        let (s, c) = (a.sin() as f32, a.cos() as f32);

        // `computeOrbDescriptor`: rotate each sampling point by the
        // keypoint's orientation and compare the pair. Everything is `f32`
        // upstream and the rounding is `cvRound`, so the sampled pixel can
        // differ by one position from an `f64` computation near a .5.
        let mut d = [0u8; 32];
        let sample = |px: i8, py: i8| -> Result<f64> {
            let (fx, fy) = (f32::from(px), f32::from(py));
            let ix = cv_core::saturate::cv_roundf(fx * c - fy * s);
            let iy = cv_core::saturate::cv_roundf(fx * s + fy * c);
            image.at_f64(y + iy, x + ix, 0)
        };
        for i in 0..256 {
            let q = i * 4;
            let pa = sample(BIT_PATTERN_31[q], BIT_PATTERN_31[q + 1])?;
            let pb = sample(BIT_PATTERN_31[q + 2], BIT_PATTERN_31[q + 3])?;
            if pa < pb {
                d[i / 8] |= 1 << (i % 8);
            }
        }
        let mut k = *kp;
        k.angle = angle;
        kps.push(k);
        descs.push(d);
    }
    Ok((kps, descs))
}

/// `cv::ORB::detectAndCompute(image, mask, keypoints, descriptors)`.
///
/// FAST detection with non-max suppression, keeping the `max_features`
/// strongest, then oriented BRIEF.
pub fn orb_detect_and_compute<S: MatLike + ?Sized>(
    image: &S,
    threshold: i32,
    max_features: usize,
) -> Result<(Vec<KeyPoint>, Vec<Descriptor>)> {
    let mut kps = fast(image, threshold, true)?;
    kps.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.pt.0.total_cmp(&b.pt.0))
    });
    if max_features > 0 && kps.len() > max_features {
        kps.truncate(max_features);
    }
    // Orientation belongs here, not in `compute_orb`: upstream computes it in
    // `detectAndCompute` for the keypoints it found and never in `compute`.
    let (h, w) = (image.rows(), image.cols());
    const MARGIN: i32 = 32;
    for k in &mut kps {
        let (x, y) = (k.pt.0.round() as i32, k.pt.1.round() as i32);
        if x >= MARGIN && y >= MARGIN && x < w - MARGIN && y < h - MARGIN {
            k.angle = ic_angle(image, x, y, 15)?;
        }
    }
    compute_orb(image, &kps)
}

/// Hamming distance between two binary descriptors.
#[must_use]
pub fn hamming(a: &Descriptor, b: &Descriptor) -> u32 {
    a.iter().zip(b).map(|(p, q)| (p ^ q).count_ones()).sum()
}

/// `cv::BFMatcher::match(queryDescriptors, trainDescriptors, matches)`.
///
/// One best match per query descriptor, by Hamming distance.
#[must_use]
pub fn bf_match(query: &[Descriptor], train: &[Descriptor]) -> Vec<DMatch> {
    let mut out = Vec::with_capacity(query.len());
    for (qi, q) in query.iter().enumerate() {
        let mut best = (u32::MAX, usize::MAX);
        for (ti, t) in train.iter().enumerate() {
            let d = hamming(q, t);
            if d < best.0 {
                best = (d, ti);
            }
        }
        if best.1 != usize::MAX {
            out.push(DMatch {
                query_idx: qi as i32,
                train_idx: best.1 as i32,
                img_idx: 0,
                distance: best.0 as f32,
            });
        }
    }
    out
}

/// `cv::BFMatcher::knnMatch(queryDescriptors, trainDescriptors, matches, k)`.
///
/// The `k` nearest train descriptors per query, ascending by distance. This is
/// what Lowe's ratio test needs.
#[must_use]
pub fn bf_knn_match(query: &[Descriptor], train: &[Descriptor], k: usize) -> Vec<Vec<DMatch>> {
    let mut out = Vec::with_capacity(query.len());
    for (qi, q) in query.iter().enumerate() {
        let mut all: Vec<(u32, usize)> =
            train.iter().enumerate().map(|(ti, t)| (hamming(q, t), ti)).collect();
        all.sort_by_key(|&(d, i)| (d, i));
        out.push(
            all.into_iter()
                .take(k)
                .map(|(d, ti)| DMatch {
                    query_idx: qi as i32,
                    train_idx: ti as i32,
                    img_idx: 0,
                    distance: d as f32,
                })
                .collect(),
        );
    }
    out
}

/// Lowe's ratio test over [`bf_knn_match`] output with `k = 2`.
///
/// Keeps a match only when the best candidate is clearly better than the second
/// — the standard filter for binary descriptors, where a raw nearest neighbour
/// is often noise.
#[must_use]
pub fn ratio_test(knn: &[Vec<DMatch>], ratio: f32) -> Vec<DMatch> {
    knn.iter()
        .filter_map(|m| {
            if m.len() < 2 {
                return m.first().copied();
            }
            if m[0].distance < ratio * m[1].distance {
                Some(m[0])
            } else {
                None
            }
        })
        .collect()
}

/// `cv::drawKeypoints(image, keypoints, outImage, color, flags)`.
pub fn draw_keypoints<S: MatLike + ?Sized>(
    image: &S,
    keypoints: &[KeyPoint],
    color: Scalar,
    rich: bool,
) -> Result<Mat> {
    use cv_imgproc::{circle, cvt_color, line, ColorConversion};
    let mut out = if image.channels() == 1 {
        cvt_color(image, ColorConversion::Gray2Bgr)?
    } else {
        Mat::copy_from(image)?
    };
    for kp in keypoints {
        let c = Point::new(kp.pt.0.round() as i32, kp.pt.1.round() as i32);
        let r = if rich { (kp.size / 2.0).round() as i32 } else { 3 };
        circle(&mut out, c, r.max(1), color, 1)?;
        // A radius line showing the orientation, when there is one.
        if rich && kp.angle >= 0.0 {
            let a = f64::from(kp.angle).to_radians();
            let e = Point::new(
                c.x + (f64::from(r) * a.cos()).round() as i32,
                c.y + (f64::from(r) * a.sin()).round() as i32,
            );
            line(&mut out, c, e, color, 1)?;
        }
    }
    Ok(out)
}

/// `cv::drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, ...)`.
///
/// The two images are placed side by side and every match is drawn as a line.
pub fn draw_matches<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    img1: &A,
    kp1: &[KeyPoint],
    img2: &B,
    kp2: &[KeyPoint],
    matches: &[DMatch],
    color: Scalar,
) -> Result<Mat> {
    use cv_core::types::{Depth, MatType};
    use cv_imgproc::{circle, line};
    let h = img1.rows().max(img2.rows());
    let w = img1.cols() + img2.cols();
    let mut out = Mat::zeros(h, w, MatType::new(Depth::U8, 3))?;

    let blit = |src: &dyn Fn(i32, i32, i32) -> Result<f64>,
                    rows: i32,
                    cols: i32,
                    cn: i32,
                    x0: i32,
                    dst: &mut Mat|
     -> Result<()> {
        for y in 0..rows {
            for x in 0..cols {
                for c in 0..3 {
                    let v = src(y, x, if cn == 1 { 0 } else { c })?;
                    cv_core::arithm::set_channel_generic(dst, y, x0 + x, c, v)?;
                }
            }
        }
        Ok(())
    };
    let (c1, c2) = (img1.channels(), img2.channels());
    blit(&|y, x, c| img1.at_f64(y, x, c), img1.rows(), img1.cols(), c1, 0, &mut out)?;
    blit(&|y, x, c| img2.at_f64(y, x, c), img2.rows(), img2.cols(), c2, img1.cols(), &mut out)?;

    let shift = img1.cols();
    for m in matches {
        let (qi, ti) = (m.query_idx as usize, m.train_idx as usize);
        if qi >= kp1.len() || ti >= kp2.len() {
            continue;
        }
        let a = Point::new(kp1[qi].pt.0.round() as i32, kp1[qi].pt.1.round() as i32);
        let b = Point::new(kp2[ti].pt.0.round() as i32 + shift, kp2[ti].pt.1.round() as i32);
        circle(&mut out, a, 3, color, 1)?;
        circle(&mut out, b, 3, color, 1)?;
        line(&mut out, a, b, color, 1)?;
    }
    Ok(out)
}

/// `cv::SimpleBlobDetector` — connected bright or dark regions passing filters.
#[derive(Debug, Clone, Copy)]
pub struct BlobParams {
    /// Grey level at which to binarise.
    pub threshold: f64,
    /// Detect dark blobs on a light background rather than the reverse.
    pub dark_blobs: bool,
    /// Smallest accepted area in pixels.
    pub min_area: f64,
    /// Largest accepted area in pixels.
    pub max_area: f64,
    /// Smallest accepted `4*pi*area / perimeter^2`.
    pub min_circularity: f64,
}

impl Default for BlobParams {
    fn default() -> Self {
        BlobParams {
            threshold: 128.0,
            dark_blobs: true,
            min_area: 25.0,
            max_area: 5000.0,
            min_circularity: 0.0,
        }
    }
}

/// `cv::SimpleBlobDetector::detect(image, keypoints)`.
pub fn detect_blobs<S: MatLike + ?Sized>(image: &S, p: BlobParams) -> Result<Vec<KeyPoint>> {
    use cv_imgproc::{connected_components_with_stats, threshold, ThresholdMethod, ThresholdType};
    cv_assert!(image.channels() == 1, "blob detection needs a single-channel image");
    let ttype = if p.dark_blobs { ThresholdType::BinaryInv } else { ThresholdType::Binary };
    let bin = threshold(image, p.threshold, 255.0, ttype, ThresholdMethod::Manual)?.dst;
    let lab = connected_components_with_stats(&bin, 8)?;

    let mut out = Vec::new();
    // Label 0 is the background.
    for (i, st) in lab.stats.iter().enumerate().skip(1) {
        let area = f64::from(st.area);
        if area < p.min_area || area > p.max_area {
            continue;
        }
        if p.min_circularity > 0.0 {
            // Perimeter estimated from the bounding box, which is enough to
            // separate a disc from a long thin streak.
            let per = 2.0 * f64::from(st.rect.width + st.rect.height);
            let circ = if per > 0.0 { 4.0 * core::f64::consts::PI * area / (per * per) } else { 0.0 };
            if circ < p.min_circularity {
                continue;
            }
        }
        let (cx, cy) = lab.centroids[i];
        out.push(KeyPoint {
            pt: (cx as f32, cy as f32),
            size: (2.0 * (area / core::f64::consts::PI).sqrt()) as f32,
            angle: -1.0,
            response: area as f32,
            octave: 0,
            class_id: -1,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::CV_8UC1;
    use cv_core::MatLikeMut;

    /// Isolated bright pixels on a dark field.
    ///
    /// This is the fixture rather than a filled square because **cv2's FAST
    /// finds nothing on a square** — a 90-degree corner does not produce a
    /// nine-pixel contiguous arc on the Bresenham circle, so the test would be
    /// asserting behaviour OpenCV does not have. cv2 4.13.0 returns exactly
    /// these five points at threshold 30.
    fn dots() -> Mat {
        let mut m = Mat::zeros(40, 40, CV_8UC1).unwrap();
        for (y, x) in [(10, 10), (10, 30), (30, 10), (30, 30), (20, 20)] {
            m.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
        }
        m
    }

    /// A field with enough structure to yield descriptors away from the border.
    fn textured() -> Mat {
        let mut m = Mat::zeros(80, 80, CV_8UC1).unwrap();
        for (y, x) in [(20, 20), (20, 55), (55, 20), (55, 55), (40, 40), (30, 60), (60, 30)] {
            m.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
        }
        m
    }

    #[test]
    fn fast_matches_cv2_on_isolated_dots() {
        let kps = fast(&dots(), 30, true).unwrap();
        let mut got: Vec<(i32, i32)> =
            kps.iter().map(|k| (k.pt.0 as i32, k.pt.1 as i32)).collect();
        got.sort();
        // cv2 4.13.0, FastFeatureDetector(30, true).
        assert_eq!(got, vec![(10, 10), (10, 30), (20, 20), (30, 10), (30, 30)]);
    }

    #[test]
    fn fast_finds_nothing_on_a_square_just_as_cv2_does() {
        // A filled square has no 9-arc anywhere on its corners. cv2 returns 0
        // here too; this pins that agreement so a future "fix" cannot quietly
        // diverge by making the arc test looser.
        let mut m = Mat::zeros(40, 40, CV_8UC1).unwrap();
        for y in 10..30 {
            for x in 10..30 {
                m.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        assert!(fast(&m, 30, true).unwrap().is_empty());
    }

    #[test]
    fn fast_finds_nothing_on_a_flat_image() {
        let flat = Mat::from_vec(30, 30, 1, &[128u8; 900]).unwrap();
        assert!(fast(&flat, 20, true).unwrap().is_empty());
    }

    #[test]
    fn a_higher_threshold_finds_no_more_corners() {
        let img = dots();
        let lo = fast(&img, 10, true).unwrap().len();
        let hi = fast(&img, 200, true).unwrap().len();
        assert!(hi <= lo, "raising the threshold cannot add corners: {lo} -> {hi}");
    }

    #[test]
    fn nonmax_suppression_never_adds_points() {
        let img = dots();
        let raw = fast(&img, 30, false).unwrap().len();
        let sup = fast(&img, 30, true).unwrap().len();
        assert!(sup <= raw, "suppression cannot add points: {raw} -> {sup}");
        assert_eq!(sup, 5);
    }

    #[test]
    fn fast_rejects_a_colour_image() {
        let m = Mat::zeros(20, 20, cv_core::types::CV_8UC3).unwrap();
        assert!(fast(&m, 20, true).is_err());
    }

    #[test]
    fn descriptors_are_stable_and_self_matching() {
        let img = textured();
        let (kps, descs) = orb_detect_and_compute(&img, 30, 50).unwrap();
        assert!(!descs.is_empty(), "no descriptors computed");
        assert_eq!(kps.len(), descs.len());
        let m = bf_match(&descs, &descs);
        assert_eq!(m.len(), descs.len());
        for (i, mm) in m.iter().enumerate() {
            assert_eq!(mm.distance, 0.0, "descriptor {i} did not match itself");
            // Not `train_idx == i`: a symmetric fixture can produce two
            // identical descriptors, and then the first one legitimately wins.
            assert_eq!(
                descs[mm.train_idx as usize], descs[i],
                "descriptor {i} matched a different bit pattern"
            );
        }
    }

    #[test]
    fn descriptors_are_deterministic_across_runs() {
        let img = textured();
        let (_, a) = orb_detect_and_compute(&img, 30, 20).unwrap();
        let (_, b) = orb_detect_and_compute(&img, 30, 20).unwrap();
        assert_eq!(a, b, "the BRIEF pattern must be fixed, not re-seeded per call");
    }

    #[test]
    fn descriptors_skip_keypoints_too_close_to_the_border() {
        // A dot in the corner cannot be described; it must be dropped from the
        // returned keypoint list too, so the two stay index-aligned. The image
        // has to be bigger than 2 * 32 for *any* keypoint to be describable:
        // upstream's border is `max(edgeThreshold, patchSize/2) + 1`.
        let mut m = Mat::zeros(100, 100, CV_8UC1).unwrap();
        m.pixel_mut::<u8>(5, 5).unwrap()[0] = 255;
        m.pixel_mut::<u8>(50, 50).unwrap()[0] = 255;
        let kps = fast(&m, 30, true).unwrap();
        assert_eq!(kps.len(), 2);
        let (out_kps, descs) = compute_orb(&m, &kps).unwrap();
        assert_eq!(out_kps.len(), descs.len());
        assert_eq!(out_kps.len(), 1, "the corner keypoint has no room for a patch");
        assert_eq!((out_kps[0].pt.0 as i32, out_kps[0].pt.1 as i32), (50, 50));
    }

    #[test]
    fn hamming_counts_differing_bits() {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        assert_eq!(hamming(&a, &b), 0);
        a[0] = 0b1111;
        assert_eq!(hamming(&a, &b), 4);
        b[31] = 0b1000_0000;
        assert_eq!(hamming(&a, &b), 5);
    }

    #[test]
    fn knn_match_is_sorted_and_the_right_length() {
        let q = vec![[0u8; 32]];
        let mut t = vec![[0u8; 32], [0xFFu8; 32], [0x0Fu8; 32]];
        t[0][0] = 1;
        let knn = bf_knn_match(&q, &t, 3);
        assert_eq!(knn.len(), 1);
        assert_eq!(knn[0].len(), 3);
        for w in knn[0].windows(2) {
            assert!(w[0].distance <= w[1].distance, "knn results must be ascending");
        }
        assert_eq!(knn[0][0].train_idx, 0, "the one-bit difference is nearest");
    }

    #[test]
    fn knn_caps_at_the_train_set_size() {
        let q = vec![[0u8; 32]];
        let t = vec![[0u8; 32]];
        assert_eq!(bf_knn_match(&q, &t, 5)[0].len(), 1);
    }

    #[test]
    fn ratio_test_keeps_the_unambiguous_match_and_drops_the_ambiguous_one() {
        let clear = vec![
            DMatch { query_idx: 0, train_idx: 0, img_idx: 0, distance: 10.0 },
            DMatch { query_idx: 0, train_idx: 1, img_idx: 0, distance: 90.0 },
        ];
        let murky = vec![
            DMatch { query_idx: 1, train_idx: 0, img_idx: 0, distance: 50.0 },
            DMatch { query_idx: 1, train_idx: 1, img_idx: 0, distance: 52.0 },
        ];
        let kept = ratio_test(&[clear, murky], 0.75);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].query_idx, 0);
    }

    #[test]
    fn draw_keypoints_produces_a_colour_image_with_marks() {
        let img = dots();
        let kps = fast(&img, 30, true).unwrap();
        let out = draw_keypoints(&img, &kps, Scalar::new(0.0, 0.0, 255.0, 0.0), true).unwrap();
        assert_eq!(out.channels(), 3);
        // Something red got drawn that was not in the greyscale source.
        let red = (0..40)
            .flat_map(|y| (0..40).map(move |x| (y, x)))
            .filter(|&(y, x)| {
                out.at_f64(y, x, 2).unwrap() > 200.0 && out.at_f64(y, x, 0).unwrap() < 50.0
            })
            .count();
        assert!(red > 0, "no keypoint markers drawn");
    }

    #[test]
    fn draw_matches_places_the_images_side_by_side() {
        let a = textured();
        let b = textured();
        let (ka, da) = orb_detect_and_compute(&a, 30, 10).unwrap();
        let (kb, db) = orb_detect_and_compute(&b, 30, 10).unwrap();
        let m = bf_match(&da, &db);
        let out = draw_matches(&a, &ka, &b, &kb, &m, Scalar::all(255.0)).unwrap();
        assert_eq!((out.rows(), out.cols()), (80, 160));
        assert_eq!(out.channels(), 3);
    }

    #[test]
    fn blob_detector_finds_dark_discs() {
        // Light background with two dark squares.
        let mut m = Mat::from_vec(60, 60, 1, &[255u8; 3600]).unwrap();
        for (cx, cy) in [(15, 15), (45, 45)] {
            for y in cy - 5..cy + 5 {
                for x in cx - 5..cx + 5 {
                    m.pixel_mut::<u8>(y, x).unwrap()[0] = 0;
                }
            }
        }
        let kps = detect_blobs(&m, BlobParams { min_area: 50.0, ..Default::default() }).unwrap();
        assert_eq!(kps.len(), 2, "expected two blobs, got {kps:?}");
        for k in &kps {
            let near = [(14.5f32, 14.5f32), (44.5, 44.5)]
                .iter()
                .any(|c| (k.pt.0 - c.0).abs() < 2.0 && (k.pt.1 - c.1).abs() < 2.0);
            assert!(near, "blob at {:?} is not where it was drawn", k.pt);
        }
    }

    #[test]
    fn blob_area_filters_apply() {
        let mut m = Mat::from_vec(40, 40, 1, &[255u8; 1600]).unwrap();
        for y in 10..14 {
            for x in 10..14 {
                m.pixel_mut::<u8>(y, x).unwrap()[0] = 0;
            }
        }
        // The 4x4 blob is 16 px: below a 50 px floor, above a 4 px one.
        assert!(detect_blobs(&m, BlobParams { min_area: 50.0, ..Default::default() }).unwrap().is_empty());
        assert_eq!(
            detect_blobs(&m, BlobParams { min_area: 4.0, ..Default::default() }).unwrap().len(),
            1
        );
    }

    #[test]
    fn keypoint_constructor_defaults_are_upstreams() {
        let k = KeyPoint::new(1.0, 2.0, 7.0);
        assert_eq!(k.pt, (1.0, 2.0));
        assert_eq!(k.angle, -1.0);
        assert_eq!(k.class_id, -1);
        assert_eq!(k.octave, 0);
    }
}
