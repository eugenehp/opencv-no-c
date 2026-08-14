//! AGAST, GFTT and MSER (`modules/features2d/src/agast.cpp`,
//! `gftt.cpp`, `mser.cpp`).

use cv_core::{cv_assert, MatLike, Result};

use crate::detectors::{AgastParams, AgastType, GfttParams, MserParams};
use crate::KeyPoint;

/// The circle a given AGAST pattern tests, and the arc length that makes a corner.
///
/// The four patterns trade radius for cost: a radius-1 circle is eight pixels
/// and cheap but fires on fine texture, while the radius-3 circle is the same
/// one FAST uses and is the most selective.
fn pattern(t: AgastType) -> (&'static [(i32, i32)], usize) {
    // Radius 1: the 8-neighbourhood, clockwise from the top.
    const C8: [(i32, i32); 8] =
        [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)];
    // Radius 2: twelve pixels, clockwise from the top.
    const C12: [(i32, i32); 12] = [
        (0, -2),
        (1, -2),
        (2, -1),
        (2, 0),
        (2, 1),
        (1, 2),
        (0, 2),
        (-1, 2),
        (-2, 1),
        (-2, 0),
        (-2, -1),
        (-1, -2),
    ];
    // Radius 3: the sixteen-pixel Bresenham circle FAST uses.
    const C16: [(i32, i32); 16] = [
        (0, -3),
        (1, -3),
        (2, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 3),
        (-1, 3),
        (-2, 2),
        (-3, 1),
        (-3, 0),
        (-3, -1),
        (-2, -2),
        (-1, -3),
    ];
    match t {
        AgastType::Agast5_8 => (&C8, 5),
        // The two 12-pixel variants differ in how upstream's decision tree is
        // generated, not in the circle or the arc length; both are 7 of 12.
        AgastType::Agast7_12d | AgastType::Agast7_12s => (&C12, 7),
        AgastType::Oast9_16 => (&C16, 9),
    }
}

/// Longest run of contiguous `true` in a cyclic buffer.
fn longest_arc(flags: &[bool]) -> usize {
    let n = flags.len();
    if flags.iter().all(|f| *f) {
        return n;
    }
    let mut best = 0;
    let mut run = 0;
    // Two laps so a run wrapping the end is seen whole.
    for i in 0..n * 2 {
        if flags[i % n] {
            run += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    best.min(n)
}

/// `agast_cornerScore<...>` — the highest threshold at which `(x, y)` is still
/// a corner, or 0 if it is not one at `threshold`.
///
/// Exposed for [`crate::brisk_scalespace`], which needs the score surface
/// itself rather than the suppressed keypoints built from it.
pub(crate) fn agast_corner_score(
    img: &[u8],
    cols: usize,
    rows: usize,
    x: i32,
    y: i32,
    threshold: i32,
    t: AgastType,
) -> i32 {
    let (circle, need) = pattern(t);
    let radius = circle.iter().map(|(a, b)| a.abs().max(b.abs())).max().unwrap_or(3);
    if x < radius || y < radius || x >= cols as i32 - radius || y >= rows as i32 - radius {
        return 0;
    }
    let at = |yy: i32, xx: i32| i32::from(img[yy as usize * cols + xx as usize]);
    let is_corner = |thr: i32| -> bool {
        let c = at(y, x);
        let mut bright = vec![false; circle.len()];
        let mut dark = vec![false; circle.len()];
        for (i, (dx, dy)) in circle.iter().enumerate() {
            let v = at(y + dy, x + dx);
            bright[i] = v > c + thr;
            dark[i] = v < c - thr;
        }
        longest_arc(&bright) >= need || longest_arc(&dark) >= need
    };
    if !is_corner(threshold) {
        return 0;
    }
    let (mut lo, mut hi) = (threshold, 255);
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if is_corner(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// `cv::AGAST(image, keypoints, threshold, nonmaxSuppression, type)`.
///
/// A corner is a pixel with a long enough contiguous arc of circle neighbours
/// all brighter than `centre + threshold` or all darker than `centre -
/// threshold`. The score used for suppression is upstream's: the largest
/// threshold at which the pixel would still be a corner.
pub fn agast<S: MatLike + ?Sized>(
    image: &S,
    params: AgastParams,
    t: AgastType,
) -> Result<Vec<KeyPoint>> {
    cv_assert!(image.channels() == 1, "AGAST requires a single-channel image");
    cv_assert!(params.threshold > 0, "the threshold must be positive");
    let (circle, need) = pattern(t);
    let radius = circle.iter().map(|(a, b)| a.abs().max(b.abs())).max().unwrap_or(3);
    let (h, w) = (image.rows(), image.cols());
    if h <= 2 * radius || w <= 2 * radius {
        return Ok(Vec::new());
    }

    let is_corner = |y: i32, x: i32, thr: i32| -> Result<bool> {
        let c = image.at_f64(y, x, 0)? as i32;
        let mut bright = vec![false; circle.len()];
        let mut dark = vec![false; circle.len()];
        for (i, (dx, dy)) in circle.iter().enumerate() {
            let v = image.at_f64(y + dy, x + dx, 0)? as i32;
            bright[i] = v > c + thr;
            dark[i] = v < c - thr;
        }
        Ok(longest_arc(&bright) >= need || longest_arc(&dark) >= need)
    };

    let mut raw: Vec<(i32, i32, i32)> = Vec::new();
    for y in radius..h - radius {
        for x in radius..w - radius {
            if !is_corner(y, x, params.threshold)? {
                continue;
            }
            // Score: the highest threshold this pixel still passes. Binary
            // search rather than a linear scan, since the predicate is
            // monotone in the threshold.
            let (mut lo, mut hi) = (params.threshold, 255);
            while lo < hi {
                let mid = (lo + hi + 1) / 2;
                if is_corner(y, x, mid)? {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            raw.push((x, y, lo));
        }
    }

    if !params.nonmax_suppression {
        return Ok(raw
            .into_iter()
            .map(|(x, y, s)| KeyPoint {
                pt: (x as f32, y as f32),
                size: 7.0,
                angle: -1.0,
                response: s as f32,
                octave: 0,
                class_id: -1,
            })
            .collect());
    }

    // Keep a corner only if no 8-neighbour scores higher.
    //
    // Ties matter: a flat-topped ridge of equally-scoring pixels — which is
    // exactly what a short straight edge produces — would survive *whole*
    // under a strict `>` test, turning one feature into a run of a dozen. On
    // an equal score the earlier pixel in raster order wins.
    let mut score = vec![0i32; (h * w) as usize];
    for &(x, y, s) in &raw {
        score[(y * w + x) as usize] = s;
    }
    let mut out = Vec::new();
    for &(x, y, s) in &raw {
        let mut best = true;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if (dx, dy) == (0, 0) {
                    continue;
                }
                let n = score[((y + dy) * w + x + dx) as usize];
                let earlier = (dy, dx) < (0, 0);
                if n > s || (n == s && earlier) {
                    best = false;
                }
            }
        }
        if best {
            out.push(KeyPoint {
                pt: (x as f32, y as f32),
                size: 7.0,
                angle: -1.0,
                response: s as f32,
                octave: 0,
                class_id: -1,
            });
        }
    }
    Ok(out)
}

/// `cv::GFTTDetector::detect(image, keypoints)`.
///
/// Wraps `goodFeaturesToTrack`: corners ranked by the minimum eigenvalue of
/// the structure tensor, or by the Harris measure when `harris_detector` is
/// set, then thinned so no two are closer than `min_distance`.
pub fn gftt<S: MatLike + ?Sized>(image: &S, p: GfttParams) -> Result<Vec<KeyPoint>> {
    cv_assert!(image.channels() == 1, "GFTT requires a single-channel image");
    let pts = cv_imgproc::features::good_features_to_track(
        image,
        p.max_features,
        p.quality_level,
        p.min_distance,
        p.block_size,
        p.harris_detector,
        p.k,
    )?;
    // The response map, so each keypoint can carry its own strength.
    let resp = if p.harris_detector {
        cv_imgproc::features::corner_harris(image, p.block_size, 3, p.k)?
    } else {
        cv_imgproc::features::corner_min_eigen_val(image, p.block_size, 3)?
    };
    Ok(pts
        .into_iter()
        .map(|pt| {
            let r = resp.at_f64(pt.y, pt.x, 0).unwrap_or(0.0) as f32;
            KeyPoint {
                pt: (pt.x as f32, pt.y as f32),
                size: p.block_size as f32,
                angle: -1.0,
                response: r,
                octave: 0,
                class_id: -1,
            }
        })
        .collect())
}

/// One maximally stable extremal region.
#[derive(Debug, Clone)]
pub struct MserRegion {
    /// Pixels belonging to the region.
    pub points: Vec<(i32, i32)>,
    /// Bounding box as `(x, y, width, height)`.
    pub bbox: (i32, i32, i32, i32),
    /// Intensity threshold at which the region was most stable.
    pub level: i32,
}

/// `cv::MSER::detectRegions(image, msers, bboxes)`.
///
/// # What "maximally stable" means
///
/// Sweep a threshold from dark to light and watch the connected components of
/// the thresholded image. Most components grow steadily; a few barely change
/// over a wide range of thresholds, and those are the stable ones. Stability is
/// measured as the relative area change `|A(t+delta) - A(t-delta)| / A(t)`, and
/// a region is kept when that dips below `max_variation`.
///
/// This runs the sweep directly rather than through upstream's union-find
/// component tree. Same definition, and simpler to check; the cost is linear in
/// the number of thresholds rather than amortised across them.
pub fn mser_detect_regions<S: MatLike + ?Sized>(
    image: &S,
    p: MserParams,
) -> Result<Vec<MserRegion>> {
    cv_assert!(image.channels() == 1, "MSER requires a single-channel image");
    cv_assert!(p.delta > 0, "delta must be positive, got {}", p.delta);
    cv_assert!(p.min_area > 0 && p.max_area > p.min_area, "min_area must be below max_area");
    let (h, w) = (image.rows(), image.cols());
    if h == 0 || w == 0 {
        return Ok(Vec::new());
    }

    // Areas of every component at every level, keyed by the component's
    // smallest pixel index so the same region can be tracked across levels.
    let mut history: Vec<Areas> = Vec::new();
    let mut comps_at: Vec<Vec<Component>> = Vec::new();

    for level in 0..=255i32 {
        let (comps, areas) = components_below(image, level, h, w)?;
        history.push(areas);
        comps_at.push(comps);
    }

    let mut out: Vec<MserRegion> = Vec::new();
    let delta = p.delta as usize;
    for level in delta..=(255 - delta as i32) as usize {
        for (key, pts) in &comps_at[level] {
            let a = pts.len();
            if a < p.min_area as usize || a > p.max_area as usize {
                continue;
            }
            // The same component's area `delta` levels either side.
            let (Some(&lo), Some(&hi)) =
                (history[level - delta].get(key), history[level + delta].get(key))
            else {
                continue;
            };
            let variation = (hi as f64 - lo as f64).abs() / a as f64;
            if variation > p.max_variation {
                continue;
            }
            // Keep the most stable instance of each region rather than one per
            // level, which would return the same blob dozens of times.
            if out.iter().any(|r| r.points.len().abs_diff(a) * 4 < a) {
                continue;
            }
            let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            for &(x, y) in pts {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
            out.push(MserRegion {
                points: pts.clone(),
                bbox: (x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                level: level as i32,
            });
        }
    }
    Ok(out)
}

/// One connected component: its key and the pixels in it.
type Component = (usize, Vec<(i32, i32)>);
/// Component areas, keyed the same way.
type Areas = std::collections::HashMap<usize, usize>;

/// Connected components of the pixels at or below `level`, with their areas
/// keyed by the smallest linear pixel index in each.
fn components_below<S: MatLike + ?Sized>(
    image: &S,
    level: i32,
    h: i32,
    w: i32,
) -> Result<(Vec<Component>, Areas)> {
    let mut seen = vec![false; (h * w) as usize];
    let mut comps = Vec::new();
    let mut areas = std::collections::HashMap::new();
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if seen[i] || (image.at_f64(y, x, 0)? as i32) > level {
                continue;
            }
            // Flood fill from here.
            let mut stack = vec![(x, y)];
            let mut pts = Vec::new();
            seen[i] = true;
            let mut key = i;
            while let Some((cx, cy)) = stack.pop() {
                pts.push((cx, cy));
                key = key.min((cy * w + cx) as usize);
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (nx, ny) = (cx + dx, cy + dy);
                    if nx < 0 || ny < 0 || nx >= w || ny >= h {
                        continue;
                    }
                    let j = (ny * w + nx) as usize;
                    if !seen[j] && (image.at_f64(ny, nx, 0)? as i32) <= level {
                        seen[j] = true;
                        stack.push((nx, ny));
                    }
                }
            }
            areas.insert(key, pts.len());
            comps.push((key, pts));
        }
    }
    Ok((comps, areas))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::Mat;

    /// An image with isolated bright dots, which every corner detector fires on.
    fn dots(w: i32, h: i32, at: &[(i32, i32)]) -> Mat {
        let mut d = vec![0u8; (w * h) as usize];
        for &(x, y) in at {
            d[(y * w + x) as usize] = 255;
        }
        Mat::from_vec(h, w, 1, &d).unwrap()
    }

    #[test]
    fn the_longest_arc_wraps_around() {
        assert_eq!(longest_arc(&[true, true, false, true]), 3, "the run wraps the end");
        assert_eq!(longest_arc(&[true; 8]), 8, "a full circle is capped at its length");
        assert_eq!(longest_arc(&[false; 8]), 0);
        assert_eq!(longest_arc(&[true, false, true, false]), 1);
    }

    #[test]
    fn agast_finds_isolated_dots() {
        let img = dots(40, 40, &[(10, 10), (25, 18), (30, 31)]);
        for t in [
            AgastType::Agast5_8,
            AgastType::Agast7_12d,
            AgastType::Agast7_12s,
            AgastType::Oast9_16,
        ] {
            let kps = agast(&img, AgastParams::default(), t).unwrap();
            assert!(!kps.is_empty(), "{t:?} found nothing");
            // Every detection sits on or beside a dot.
            for k in &kps {
                let near = [(10, 10), (25, 18), (30, 31)].iter().any(|&(x, y)| {
                    (k.pt.0 - x as f32).abs() <= 3.0 && (k.pt.1 - y as f32).abs() <= 3.0
                });
                assert!(near, "{t:?} fired at {:?}, away from every dot", k.pt);
            }
            assert!(kps.iter().all(|k| k.response > 0.0), "a corner has a score");
        }
    }

    #[test]
    fn the_wider_patterns_are_more_selective() {
        // Fine texture: the radius-1 circle fires on it, radius-3 should not
        // fire more. This is the trade the pattern choice buys.
        let mut d = vec![0u8; 60 * 60];
        for y in 0..60 {
            for x in 0..60 {
                d[y * 60 + x] = if (x + y) % 2 == 0 { 0 } else { 255 };
            }
        }
        let img = Mat::from_vec(60, 60, 1, &d).unwrap();
        let n8 = agast(&img, AgastParams::default(), AgastType::Agast5_8).unwrap().len();
        let n16 = agast(&img, AgastParams::default(), AgastType::Oast9_16).unwrap().len();
        assert!(n16 <= n8, "the 16-pixel circle should not fire more: {n16} vs {n8}");
    }

    #[test]
    fn non_max_suppression_thins_the_result() {
        let img = dots(40, 40, &[(20, 20)]);
        let mut p = AgastParams::default();
        p.set_nonmax_suppression(false);
        let raw = agast(&img, p, AgastType::Oast9_16).unwrap();
        p.set_nonmax_suppression(true);
        let thin = agast(&img, p, AgastType::Oast9_16).unwrap();
        assert!(thin.len() <= raw.len(), "suppression must not add corners");
        assert!(!thin.is_empty(), "and must not remove them all");
    }

    #[test]
    fn agast_validates_its_input() {
        let colour = Mat::from_vec(20, 20, 3, &[0u8; 1200]).unwrap();
        assert!(agast(&colour, AgastParams::default(), AgastType::Oast9_16).is_err());
        let mut p = AgastParams::default();
        p.set_threshold(0);
        assert!(agast(&dots(20, 20, &[]), p, AgastType::Oast9_16).is_err());
        // Too small to hold the circle is empty, not an error.
        let tiny = Mat::from_vec(4, 4, 1, &[0u8; 16]).unwrap();
        assert!(agast(&tiny, AgastParams::default(), AgastType::Oast9_16).unwrap().is_empty());
    }

    #[test]
    fn gftt_finds_corners_and_respects_its_cap() {
        // A bright square has four corners.
        let mut d = vec![0u8; 60 * 60];
        for y in 20..40 {
            for x in 20..40 {
                d[y * 60 + x] = 255;
            }
        }
        let img = Mat::from_vec(60, 60, 1, &d).unwrap();
        let kps = gftt(&img, GfttParams::default()).unwrap();
        assert!(!kps.is_empty(), "a square has corners");
        assert!(kps.iter().all(|k| k.response > 0.0));

        let mut p = GfttParams::default();
        p.set_max_features(2);
        assert!(gftt(&img, p).unwrap().len() <= 2, "the cap must hold");

        // The Harris measure is a different ranking but still finds corners.
        let mut hp = GfttParams::default();
        hp.set_harris_detector(true);
        assert!(!gftt(&img, hp).unwrap().is_empty());
    }

    #[test]
    fn gftt_validates_its_input() {
        let colour = Mat::from_vec(20, 20, 3, &[0u8; 1200]).unwrap();
        assert!(gftt(&colour, GfttParams::default()).is_err());
    }

    #[test]
    fn mser_finds_a_stable_blob() {
        // A dark disc on a light field is stable across a wide threshold range.
        let (w, h) = (60, 60);
        let mut d = vec![200u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                if (x - 30) * (x - 30) + (y - 30) * (y - 30) <= 100 {
                    d[(y * w + x) as usize] = 20;
                }
            }
        }
        let img = Mat::from_vec(h, w, 1, &d).unwrap();
        let mut p = MserParams::default();
        p.set_min_area(50).set_max_area(2000);
        let regions = mser_detect_regions(&img, p).unwrap();
        assert!(!regions.is_empty(), "the disc is a maximally stable region");
        // The disc has area ~pi*100 = 314; the found region should be close.
        let best = regions.iter().min_by_key(|r| r.points.len().abs_diff(314)).unwrap();
        assert!(
            (best.points.len() as i64 - 314).abs() < 60,
            "region area {} is far from the disc's 314",
            best.points.len()
        );
        // And its bounding box straddles the centre.
        let (bx, by, bw, bh) = best.bbox;
        assert!(bx <= 30 && by <= 30 && bx + bw >= 30 && by + bh >= 30, "bbox {:?}", best.bbox);
    }

    #[test]
    fn mser_respects_the_area_bounds() {
        let (w, h) = (40, 40);
        let mut d = vec![200u8; (w * h) as usize];
        for y in 15..25 {
            for x in 15..25 {
                d[(y * w + x) as usize] = 20;
            }
        }
        let img = Mat::from_vec(h, w, 1, &d).unwrap();
        let mut p = MserParams::default();
        p.set_min_area(10).set_max_area(500);
        let all = mser_detect_regions(&img, p).unwrap();
        assert!(all.iter().all(|r| (10..=500).contains(&(r.points.len() as i32))));

        // A window that excludes the blob returns nothing.
        let mut tight = MserParams::default();
        tight.set_min_area(2000).set_max_area(3000);
        assert!(mser_detect_regions(&img, tight).unwrap().is_empty());
    }

    #[test]
    fn mser_validates_its_input() {
        let img = Mat::from_vec(10, 10, 1, &[0u8; 100]).unwrap();
        let mut p = MserParams::default();
        p.set_delta(0);
        assert!(mser_detect_regions(&img, p).is_err());
        let mut q = MserParams::default();
        q.set_min_area(100).set_max_area(50);
        assert!(mser_detect_regions(&img, q).is_err());
        let colour = Mat::from_vec(10, 10, 3, &[0u8; 300]).unwrap();
        assert!(mser_detect_regions(&colour, MserParams::default()).is_err());
    }

    #[test]
    fn a_flat_image_has_no_stable_regions_worth_reporting() {
        let flat = Mat::from_vec(30, 30, 1, &[128u8; 900]).unwrap();
        let mut p = MserParams::default();
        p.set_min_area(10).set_max_area(200);
        // The whole image is one component of 900 px, outside the area window.
        assert!(mser_detect_regions(&flat, p).unwrap().is_empty());
    }

    /// AGAST and MSER against cv2 4.13.0, exactly.
    #[test]
    fn agast_and_mser_match_cv2() {
        // Three isolated dots; cv2 returns exactly these, response 254, for
        // all four patterns.
        let img = dots(40, 40, &[(10, 10), (25, 18), (30, 31)]);
        for t in [
            AgastType::Agast5_8,
            AgastType::Agast7_12d,
            AgastType::Agast7_12s,
            AgastType::Oast9_16,
        ] {
            let mut p = AgastParams::default();
            p.set_threshold(10);
            let mut k = agast(&img, p, t).unwrap();
            k.sort_by_key(|a| (a.pt.0 as i32, a.pt.1 as i32));
            let got: Vec<(i32, i32, i32)> = k
                .iter()
                .map(|a| (a.pt.0 as i32, a.pt.1 as i32, a.response as i32))
                .collect();
            assert_eq!(
                got,
                vec![(10, 10, 254), (25, 18, 254), (30, 31, 254)],
                "{t:?} disagrees with cv2"
            );
        }

        // A dark disc of radius 10 on a light field. cv2's MSER returns one
        // region of area 317 with bounding box (20, 20, 21, 21).
        let (w, h) = (60i32, 60i32);
        let mut d = vec![200u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                if (x - 30) * (x - 30) + (y - 30) * (y - 30) <= 100 {
                    d[(y * w + x) as usize] = 20;
                }
            }
        }
        let disc = Mat::from_vec(h, w, 1, &d).unwrap();
        let mut mp = MserParams::default();
        mp.set_min_area(50).set_max_area(2000);
        let r = mser_detect_regions(&disc, mp).unwrap();
        assert_eq!(r.len(), 1, "cv2 finds exactly one region");
        assert_eq!(r[0].points.len(), 317, "cv2 area");
        assert_eq!(r[0].bbox, (20, 20, 21, 21), "cv2 bounding box");
    }
}
