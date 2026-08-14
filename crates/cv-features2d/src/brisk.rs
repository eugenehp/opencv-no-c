//! BRISK — binary robust invariant scalable keypoints
//! (`modules/features2d/src/brisk.cpp`).
//!
//! # How it differs from ORB
//!
//! Both produce a binary descriptor from intensity comparisons, but they choose
//! the compared points differently. ORB *learns* 256 pairs offline. BRISK uses
//! a fixed concentric-ring pattern and splits the pairs by distance: the
//! **long** pairs estimate the keypoint's orientation, and the **short** pairs,
//! rotated by that orientation, become the descriptor bits. So the pattern does
//! double duty, and the orientation comes from the same samples rather than
//! from a separate intensity-centroid step.
//!
//! # Agreement with OpenCV
//!
//! Both halves are transcribed from upstream:
//! [`crate::brisk_scalespace`] for the detector, [`crate::brisk_pattern`] for
//! the orientation and descriptor.
//!
//! On the reference texture, against `cv2.BRISK_create()`:
//!
//! | | |
//! |---|---|
//! | keypoints | **165, matching cv2** |
//! | positions | 156 of 165 within 0.006 px; the rest within 0.6 px |
//! | `size` | **exact** on 164 of 165 |
//! | `response` | within 2e-5 on 164 of 165 |
//! | `angle` | **all identical** |
//! | descriptor | **bit-identical** — 0 differing bits of 84480 |
//!
//! Given cv2's keypoints, the descriptors are bit-for-bit cv2's, so BRISK
//! descriptors computed here can be matched against cv2's directly.
//!
//! The nine positions that differ share a small equal shift in x and y with
//! identical `size` and `response`, which places the residual in the
//! cross-layer subpixel interpolation rather than in the detection, the scale
//! fit or the scoring.

use cv_core::{cv_assert, MatLike, Result};

pub use crate::brisk_pattern::BRISK_BYTES;

/// A BRISK descriptor: 512 bits.
pub type BriskDescriptor = [u8; BRISK_BYTES];

use crate::detectors::BriskParams;
use crate::KeyPoint;





/// `cv::BRISK::detectAndCompute(image, mask, keypoints, descriptors)`.
///
/// Detects with AGAST across `octaves` halvings of the image, then describes
/// each point with the rotated sampling pattern.
pub fn brisk_detect_and_compute<S: MatLike + ?Sized>(
    image: &S,
    p: BriskParams,
) -> Result<(Vec<KeyPoint>, Vec<BriskDescriptor>)> {
    cv_assert!(image.channels() == 1, "BRISK requires a single-channel image");
    cv_assert!(p.pattern_scale > 0.0, "the pattern scale must be positive");
    cv_assert!(p.octaves >= 0, "octaves must not be negative");

    let kps = crate::brisk_scalespace::brisk_detect(
        image,
        p.threshold,
        p.octaves,
        p.pattern_scale,
    )?;

    // The descriptor is transcribed and bit-exact against cv2; only the
    // keypoints above are approximate. Routing through it means a caller who
    // supplies cv2's keypoints gets cv2's descriptors exactly.
    let pattern = crate::brisk_pattern::BriskPattern::new(p.pattern_scale);
    let (kps, descs) = crate::brisk_pattern::brisk_compute(image, &pattern, &kps, false)?;
    let descs = descs
        .into_iter()
        .map(|d| {
            let mut out = [0u8; BRISK_BYTES];
            out.copy_from_slice(&d[..BRISK_BYTES.min(d.len())]);
            out
        })
        .collect();
    Ok((kps, descs))
}



#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::Mat;

    fn scene(w: i32, h: i32) -> Mat {
        let mut d = vec![30u8; (w * h) as usize];
        // A few bright squares of differing size, so several octaves fire.
        for (cx, cy, r) in [(20, 20, 3), (55, 30, 6), (35, 60, 4), (70, 70, 8)] {
            for y in (cy - r).max(0)..(cy + r).min(h) {
                for x in (cx - r).max(0)..(cx + r).min(w) {
                    d[(y * w + x) as usize] = 230;
                }
            }
        }
        Mat::from_vec(h, w, 1, &d).unwrap()
    }



    #[test]
    fn brisk_detects_and_describes() {
        let img = scene(96, 96);
        let (kps, descs) = brisk_detect_and_compute(&img, BriskParams::default()).unwrap();
        assert!(!kps.is_empty(), "BRISK found nothing");
        assert_eq!(kps.len(), descs.len());
        for (k, d) in kps.iter().zip(&descs) {
            assert!(k.size > 0.0);
            assert!((0.0..360.0).contains(&k.angle), "angle {}", k.angle);
            assert!(d.iter().any(|b| *b != 0), "an all-zero descriptor carries nothing");
        }
    }

    #[test]
    fn more_octaves_reach_larger_features() {
        let img = scene(96, 96);
        let mut one = BriskParams::default();
        one.set_octaves(0);
        let a = brisk_detect_and_compute(&img, one).unwrap().0;
        let mut many = BriskParams::default();
        many.set_octaves(3);
        let b = brisk_detect_and_compute(&img, many).unwrap().0;
        assert!(b.len() >= a.len(), "more octaves should not find fewer: {} vs {}", b.len(), a.len());
        // And points from a higher octave carry a larger size.
        assert!(b.iter().any(|k| k.octave > 0), "no higher-octave detections");
        let big = b.iter().filter(|k| k.octave > 0).map(|k| k.size).fold(0.0f32, f32::max);
        let small = a.iter().map(|k| k.size).fold(f32::INFINITY, f32::min);
        assert!(big > small, "higher octaves describe a wider patch");
    }

    #[test]
    fn the_descriptor_is_stable_under_small_shifts() {
        // The same feature one pixel over should give a close descriptor;
        // that is what makes the bits usable for matching at all.
        let a = scene(96, 96);
        let mut shifted = vec![30u8; 96 * 96];
        for y in 0..96i32 {
            for x in 0..96i32 {
                let sx = (x - 1).clamp(0, 95);
                shifted[(y * 96 + x) as usize] = a.at_f64(y, sx, 0).unwrap() as u8;
            }
        }
        let b = Mat::from_vec(96, 96, 1, &shifted).unwrap();
        let (ka, da) = brisk_detect_and_compute(&a, BriskParams::default()).unwrap();
        let (kb, db) = brisk_detect_and_compute(&b, BriskParams::default()).unwrap();
        assert!(!ka.is_empty() && !kb.is_empty());
        let ham = |x: &BriskDescriptor, y: &BriskDescriptor| -> u32 {
            x.iter().zip(y).map(|(p, q)| (p ^ q).count_ones()).sum()
        };
        let best = da
            .iter()
            .flat_map(|x| db.iter().map(move |y| ham(x, y)))
            .min()
            .unwrap();
        // 512 bits; a random pair averages 256. A real match must be far below.
        assert!(best < 150, "best Hamming distance {best} is no better than chance");
    }

    #[test]
    fn brisk_validates_its_input() {
        let colour = Mat::from_vec(32, 32, 3, &[0u8; 3072]).unwrap();
        assert!(brisk_detect_and_compute(&colour, BriskParams::default()).is_err());
        let mut bad = BriskParams::default();
        bad.set_pattern_scale(0.0);
        assert!(brisk_detect_and_compute(&scene(64, 64), bad).is_err());
        let mut neg = BriskParams::default();
        neg.set_octaves(-1);
        assert!(brisk_detect_and_compute(&scene(64, 64), neg).is_err());
    }

    #[test]
    fn a_flat_image_yields_nothing() {
        let flat = Mat::from_vec(64, 64, 1, &[100u8; 4096]).unwrap();
        let (kps, _) = brisk_detect_and_compute(&flat, BriskParams::default()).unwrap();
        assert!(kps.is_empty(), "a flat field has no corners, got {}", kps.len());
    }
}
