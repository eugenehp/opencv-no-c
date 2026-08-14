//! Non-maximum suppression (`modules/dnn/src/nms.inl.hpp`, `nms.cpp`).

use cv_core::{cv_assert, Rect, Result};

/// Intersection-over-union of two upright boxes.
///
/// Upstream's `rectOverlap` uses `Rect` arithmetic, where a rectangle of width
/// `w` spans `w` pixels — so the intersection width is
/// `min(x+w) - max(x)`, with no `+1`. Adding one here would inflate every
/// score and change which boxes survive.
#[must_use]
pub fn box_iou(a: Rect, b: Rect) -> f64 {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.width).min(b.x + b.width);
    let y1 = (a.y + a.height).min(b.y + b.height);
    let iw = (x1 - x0).max(0) as f64;
    let ih = (y1 - y0).max(0) as f64;
    let inter = iw * ih;
    if inter <= 0.0 {
        return 0.0;
    }
    let ua = (a.width as f64 * a.height as f64) + (b.width as f64 * b.height as f64) - inter;
    if ua <= 0.0 {
        0.0
    } else {
        inter / ua
    }
}

/// `cv::dnn::NMSBoxes(bboxes, scores, score_threshold, nms_threshold, indices, eta, top_k)`.
///
/// Returns the surviving indices, ordered by descending score. `eta` shrinks
/// the IoU threshold as the loop proceeds, exactly as upstream's adaptive
/// variant does; pass `1.0` for the ordinary behaviour. `top_k` of 0 keeps all.
pub fn nms_boxes(
    boxes: &[Rect],
    scores: &[f32],
    score_threshold: f32,
    nms_threshold: f32,
    eta: f32,
    top_k: usize,
) -> Result<Vec<usize>> {
    cv_assert!(
        boxes.len() == scores.len(),
        "NMSBoxes got {} boxes but {} scores",
        boxes.len(),
        scores.len()
    );
    // Score-threshold, then sort by descending score. Upstream's sort is
    // stable on the original index, so equal scores keep input order.
    let mut order: Vec<usize> = (0..boxes.len()).filter(|&i| scores[i] > score_threshold).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    if top_k > 0 && order.len() > top_k {
        order.truncate(top_k);
    }

    let mut adaptive = nms_threshold;
    let mut keep: Vec<usize> = Vec::new();
    for &i in &order {
        let mut suppressed = false;
        for &j in &keep {
            if box_iou(boxes[i], boxes[j]) > adaptive as f64 {
                suppressed = true;
                break;
            }
        }
        if !suppressed {
            keep.push(i);
            if eta < 1.0 && adaptive > 0.5 {
                adaptive *= eta;
            }
        }
    }
    Ok(keep)
}

/// `cv::dnn::NMSBoxesBatched(bboxes, scores, class_ids, ...)`.
///
/// Suppression is applied within each class only. Upstream implements this by
/// offsetting every box by a per-class stride so classes cannot overlap; doing
/// it by partitioning is equivalent and does not risk integer overflow on a
/// large class id.
pub fn nms_boxes_batched(
    boxes: &[Rect],
    scores: &[f32],
    class_ids: &[i32],
    score_threshold: f32,
    nms_threshold: f32,
    eta: f32,
    top_k: usize,
) -> Result<Vec<usize>> {
    cv_assert!(
        boxes.len() == class_ids.len(),
        "NMSBoxesBatched got {} boxes but {} class ids",
        boxes.len(),
        class_ids.len()
    );
    let mut classes: Vec<i32> = class_ids.to_vec();
    classes.sort_unstable();
    classes.dedup();

    let mut keep: Vec<usize> = Vec::new();
    for cls in classes {
        let idx: Vec<usize> = (0..boxes.len()).filter(|&i| class_ids[i] == cls).collect();
        let sub_boxes: Vec<Rect> = idx.iter().map(|&i| boxes[i]).collect();
        let sub_scores: Vec<f32> = idx.iter().map(|&i| scores[i]).collect();
        let sub =
            nms_boxes(&sub_boxes, &sub_scores, score_threshold, nms_threshold, eta, top_k)?;
        keep.extend(sub.into_iter().map(|k| idx[k]));
    }
    // Restore global descending-score order across classes.
    keep.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    Ok(keep)
}

/// Score-decay rule for [`soft_nms_boxes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftNmsMethod {
    /// `cv::dnn::SoftNMSMethod::SOFTNMS_LINEAR`
    Linear,
    /// `cv::dnn::SoftNMSMethod::SOFTNMS_GAUSSIAN`
    Gaussian,
}

/// `cv::dnn::softNMSBoxes(bboxes, scores, updated_scores, score_threshold, nms_threshold, indices, top_k, sigma, method)`.
///
/// Instead of dropping an overlapping box, its score is decayed; boxes falling
/// below `score_threshold` afterwards are removed. Returns the kept indices
/// paired with their decayed scores.
pub fn soft_nms_boxes(
    boxes: &[Rect],
    scores: &[f32],
    score_threshold: f32,
    nms_threshold: f32,
    top_k: usize,
    sigma: f32,
    method: SoftNmsMethod,
) -> Result<Vec<(usize, f32)>> {
    cv_assert!(
        boxes.len() == scores.len(),
        "softNMSBoxes got {} boxes but {} scores",
        boxes.len(),
        scores.len()
    );
    let mut live: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    let mut out: Vec<(usize, f32)> = Vec::new();

    while !live.is_empty() {
        if top_k > 0 && out.len() >= top_k {
            break;
        }
        // Pick the current maximum.
        let (pos, &(bi, bs)) = live
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1 .1
                    .partial_cmp(&b.1 .1)
                    .unwrap_or(core::cmp::Ordering::Equal)
                    .then(b.1 .0.cmp(&a.1 .0))
            })
            .expect("live is non-empty");
        if bs <= score_threshold {
            break;
        }
        live.remove(pos);
        out.push((bi, bs));

        for entry in live.iter_mut() {
            let iou = box_iou(boxes[bi], boxes[entry.0]) as f32;
            let weight = match method {
                SoftNmsMethod::Linear => {
                    if iou > nms_threshold {
                        1.0 - iou
                    } else {
                        1.0
                    }
                }
                SoftNmsMethod::Gaussian => (-(iou * iou) / sigma).exp(),
            };
            entry.1 *= weight;
        }
        live.retain(|&(_, s)| s > score_threshold);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(x: i32, y: i32, w: i32, h: i32) -> Rect {
        Rect::new(x, y, w, h)
    }

    #[test]
    fn iou_of_identical_boxes_is_one() {
        assert!((box_iou(r(0, 0, 10, 10), r(0, 0, 10, 10)) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn iou_of_disjoint_boxes_is_zero() {
        assert_eq!(box_iou(r(0, 0, 10, 10), r(50, 50, 10, 10)), 0.0);
    }

    #[test]
    fn iou_uses_rect_arithmetic_without_an_off_by_one() {
        // Half-overlap: 10x10 boxes offset by 5 share a 5x10 strip.
        // inter = 50, union = 100 + 100 - 50 = 150.
        let got = box_iou(r(0, 0, 10, 10), r(5, 0, 10, 10));
        assert!((got - 50.0 / 150.0).abs() < 1e-12, "got {got}");
    }

    #[test]
    fn nms_matches_cv2_on_a_known_case() {
        // cv2.dnn.NMSBoxes([[0,0,10,10],[1,1,10,10],[20,20,10,10]],
        //                  [0.9,0.8,0.7], 0.5, 0.4) -> [0, 2]
        let boxes = [r(0, 0, 10, 10), r(1, 1, 10, 10), r(20, 20, 10, 10)];
        let scores = [0.9f32, 0.8, 0.7];
        let keep = nms_boxes(&boxes, &scores, 0.5, 0.4, 1.0, 0).unwrap();
        assert_eq!(keep, vec![0, 2]);
    }

    #[test]
    fn nms_drops_boxes_below_the_score_threshold() {
        let boxes = [r(0, 0, 10, 10), r(50, 50, 10, 10)];
        let scores = [0.9f32, 0.1];
        assert_eq!(nms_boxes(&boxes, &scores, 0.5, 0.4, 1.0, 0).unwrap(), vec![0]);
    }

    #[test]
    fn nms_top_k_limits_the_candidate_pool() {
        let boxes = [r(0, 0, 10, 10), r(50, 0, 10, 10), r(100, 0, 10, 10)];
        let scores = [0.9f32, 0.8, 0.7];
        // All three are disjoint, so without top_k all survive.
        assert_eq!(nms_boxes(&boxes, &scores, 0.0, 0.4, 1.0, 0).unwrap().len(), 3);
        assert_eq!(nms_boxes(&boxes, &scores, 0.0, 0.4, 1.0, 2).unwrap(), vec![0, 1]);
    }

    #[test]
    fn nms_returns_descending_score_order() {
        let boxes = [r(0, 0, 5, 5), r(50, 0, 5, 5), r(100, 0, 5, 5)];
        let scores = [0.1f32, 0.9, 0.5];
        assert_eq!(nms_boxes(&boxes, &scores, 0.0, 0.4, 1.0, 0).unwrap(), vec![1, 2, 0]);
    }

    #[test]
    fn batched_nms_does_not_suppress_across_classes() {
        // Two identical boxes with different class ids: both survive.
        let boxes = [r(0, 0, 10, 10), r(0, 0, 10, 10)];
        let scores = [0.9f32, 0.8];
        let same = nms_boxes(&boxes, &scores, 0.0, 0.4, 1.0, 0).unwrap();
        assert_eq!(same, vec![0], "same class: the weaker one is suppressed");
        let batched = nms_boxes_batched(&boxes, &scores, &[1, 2], 0.0, 0.4, 1.0, 0).unwrap();
        assert_eq!(batched, vec![0, 1], "different classes: both kept");
    }

    #[test]
    fn soft_nms_decays_rather_than_dropping() {
        let boxes = [r(0, 0, 10, 10), r(1, 1, 10, 10)];
        let scores = [0.9f32, 0.8];
        // Hard NMS removes the second box entirely.
        assert_eq!(nms_boxes(&boxes, &scores, 0.0, 0.4, 1.0, 0).unwrap(), vec![0]);
        // Soft NMS keeps it with a reduced score.
        let soft =
            soft_nms_boxes(&boxes, &scores, 0.01, 0.4, 0, 0.5, SoftNmsMethod::Linear).unwrap();
        assert_eq!(soft.len(), 2);
        assert_eq!(soft[0].0, 0);
        assert!(soft[1].1 < 0.8, "the overlapping box must be decayed, got {}", soft[1].1);
    }

    #[test]
    fn soft_nms_gaussian_never_reaches_zero_for_a_partial_overlap() {
        let boxes = [r(0, 0, 10, 10), r(5, 0, 10, 10)];
        let scores = [0.9f32, 0.8];
        let soft =
            soft_nms_boxes(&boxes, &scores, 0.0, 0.3, 0, 0.5, SoftNmsMethod::Gaussian).unwrap();
        assert_eq!(soft.len(), 2);
        assert!(soft[1].1 > 0.0 && soft[1].1 < 0.8);
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        assert!(nms_boxes(&[r(0, 0, 1, 1)], &[0.5, 0.5], 0.0, 0.5, 1.0, 0).is_err());
    }
}
