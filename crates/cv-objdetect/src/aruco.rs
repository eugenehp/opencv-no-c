//! ArUco marker detection (`modules/objdetect/src/aruco/aruco_detector.cpp`).
//!
//! # How a marker is found
//!
//! Adaptive threshold, then contours, then keep the ones that approximate to a
//! convex quadrilateral of plausible size. Each survivor is perspective-warped
//! to a square, its cells thresholded to bits, and the resulting grid looked up
//! in a dictionary — trying all four rotations, since a marker can be seen any
//! way up.
//!
//! The dictionary is what makes this robust: the codes are chosen so that any
//! two differ in many bits, so a few misread cells still identify the marker
//! uniquely, and a random quadrilateral almost never matches anything.
//!
//! # Ids are interchangeable with cv2's
//!
//! [`Dictionary::predefined`] carries upstream's own code tables (see
//! [`crate::aruco_data`]), so marker 3 here is the same pattern as cv2's marker
//! 3 — print with one and read with the other. The detection pipeline matches
//! too: on a marker pasted at (40, 30) in a 160x140 scene, cv2 reports corners
//! (40, 30) to (119, 109) and so does this.
//!
//! [`Dictionary::extend`] still *generates* well-separated codes, for families
//! upstream doesn't ship. Those ids are self-consistent but are not cv2's.

use crate::aruco_data;
use cv_core::{cv_assert, Mat, MatLike, Result};
use cv_imgproc::contours::{find_contours, ContourApproximation, RetrievalMode};
use cv_imgproc::threshold::{threshold, ThresholdMethod, ThresholdType};

/// `cv::aruco::Dictionary` — a set of marker codes and their bit layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Dictionary {
    /// Cells per side of the code, excluding the border.
    pub marker_size: usize,
    /// Bits that may differ before a code is rejected.
    pub max_correction_bits: usize,
    /// One entry per marker: `marker_size * marker_size` bits, row-major.
    pub codes: Vec<Vec<bool>>,
}

/// `cv::aruco::PredefinedDictionaryType` — the families upstream ships.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredefinedDictionary {
    /// `cv::aruco::DICT_4X4_50`
    Dict4x4_50,
    /// `cv::aruco::DICT_4X4_100`
    Dict4x4_100,
    /// `cv::aruco::DICT_5X5_50`
    Dict5x5_50,
    /// `cv::aruco::DICT_6X6_50`
    Dict6x6_50,
    /// `cv::aruco::DICT_7X7_50`
    Dict7x7_50,
}

impl PredefinedDictionary {
    /// Cells per side.
    #[must_use]
    pub fn marker_size(self) -> usize {
        match self {
            PredefinedDictionary::Dict4x4_50 | PredefinedDictionary::Dict4x4_100 => 4,
            PredefinedDictionary::Dict5x5_50 => 5,
            PredefinedDictionary::Dict6x6_50 => 6,
            PredefinedDictionary::Dict7x7_50 => 7,
        }
    }

    /// How many markers the family holds.
    #[must_use]
    pub fn count(self) -> usize {
        self.codes().len()
    }

    /// `cv::aruco::Dictionary::maxCorrectionBits` — upstream's own values.
    ///
    /// These are not `minDistance / 2`: upstream caps correction below what the
    /// separation would allow, trading recoverable bits for fewer false ids.
    #[must_use]
    pub fn max_correction_bits(self) -> usize {
        match self {
            PredefinedDictionary::Dict4x4_50 | PredefinedDictionary::Dict4x4_100 => 1,
            PredefinedDictionary::Dict5x5_50 => 3,
            PredefinedDictionary::Dict6x6_50 => 6,
            PredefinedDictionary::Dict7x7_50 => 9,
        }
    }

    /// The packed code table, one `u64` per marker.
    #[must_use]
    pub fn codes(self) -> &'static [u64] {
        match self {
            PredefinedDictionary::Dict4x4_50 => &aruco_data::DICT_4X4_50,
            PredefinedDictionary::Dict4x4_100 => &aruco_data::DICT_4X4_100,
            PredefinedDictionary::Dict5x5_50 => &aruco_data::DICT_5X5_50,
            PredefinedDictionary::Dict6x6_50 => &aruco_data::DICT_6X6_50,
            PredefinedDictionary::Dict7x7_50 => &aruco_data::DICT_7X7_50,
        }
    }
}

impl Dictionary {
    /// `cv::aruco::extendDictionary(nMarkers, markerSize, ...)`.
    ///
    /// Generates codes greedily, keeping each candidate only if it stays far
    /// enough in Hamming distance from every code already accepted **and from
    /// their rotations** — otherwise a marker would be confusable with a
    /// rotated view of another, which is exactly the case detection has to
    /// resolve.
    pub fn extend(n_markers: usize, marker_size: usize) -> Result<Self> {
        cv_assert!(marker_size >= 2, "a marker needs at least 2x2 cells");
        cv_assert!(n_markers > 0, "at least one marker is needed");
        let bits = marker_size * marker_size;
        let min_dist = (bits / 4).max(1);
        let mut codes: Vec<Vec<bool>> = Vec::new();
        // A cheap deterministic generator; the acceptance test does the work.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut tries = 0usize;
        while codes.len() < n_markers && tries < n_markers * 20_000 {
            tries += 1;
            let cand: Vec<bool> = (0..bits)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state & 1 == 1
                })
                .collect();
            let ok = codes.iter().all(|c| {
                (0..4).all(|r| hamming(&rotate(c, marker_size, r), &cand) >= min_dist)
            }) && (1..4).all(|r| hamming(&rotate(&cand, marker_size, r), &cand) >= min_dist);
            if ok {
                codes.push(cand);
            }
        }
        cv_assert!(
            codes.len() == n_markers,
            "could only generate {} of {n_markers} well-separated {marker_size}x{marker_size} codes",
            codes.len()
        );
        Ok(Dictionary { marker_size, max_correction_bits: min_dist / 2, codes })
    }

    /// `cv::aruco::getPredefinedDictionary(dict)`.
    ///
    /// Unpacks upstream's table, so ids match cv2's.
    #[must_use]
    pub fn predefined(d: PredefinedDictionary) -> Self {
        let marker_size = d.marker_size();
        let bits = marker_size * marker_size;
        let codes = d
            .codes()
            .iter()
            // Row-major, most significant of the `bits` used bits first.
            .map(|&w| (0..bits).map(|i| w >> (bits - 1 - i) & 1 == 1).collect())
            .collect();
        Dictionary { marker_size, max_correction_bits: d.max_correction_bits(), codes }
    }

    /// `cv::aruco::Dictionary::identify(onlyBits, idx, rotation, maxCorrectionRate)`.
    ///
    /// Returns the marker id and how many quarter-turns the observed grid is
    /// from the stored code.
    #[must_use]
    pub fn identify(&self, bits: &[bool]) -> Option<(usize, usize)> {
        if bits.len() != self.marker_size * self.marker_size {
            return None;
        }
        let mut best: Option<(usize, usize, usize)> = None;
        for (id, code) in self.codes.iter().enumerate() {
            for rot in 0..4 {
                let d = hamming(&rotate(code, self.marker_size, rot), bits);
                if d <= self.max_correction_bits
                    && best.map_or(true, |(bd, _, _)| d < bd)
                {
                    best = Some((d, id, rot));
                }
            }
        }
        best.map(|(_, id, rot)| (id, rot))
    }

    /// `cv::aruco::Dictionary::generateImageMarker(id, sidePixels, img, borderBits)`.
    pub fn generate_image_marker(
        &self,
        id: usize,
        side_pixels: i32,
        border_bits: usize,
    ) -> Result<Mat> {
        cv_assert!(id < self.codes.len(), "no marker {id} in a dictionary of {}", self.codes.len());
        cv_assert!(side_pixels > 0, "the side must be positive");
        let cells = self.marker_size + 2 * border_bits;
        let mut m = Mat::zeros(side_pixels, side_pixels, cv_core::types::MatType::new(cv_core::types::Depth::U8, 1))?;
        let code = &self.codes[id];
        for y in 0..side_pixels {
            for x in 0..side_pixels {
                let cy = (y as usize * cells) / side_pixels as usize;
                let cx = (x as usize * cells) / side_pixels as usize;
                // The border is always black; that is what lets the detector
                // find the marker's outline in the first place.
                let on = cy >= border_bits
                    && cx >= border_bits
                    && cy < cells - border_bits
                    && cx < cells - border_bits
                    && code[(cy - border_bits) * self.marker_size + (cx - border_bits)];
                cv_core::MatLikeMut::pixel_mut::<u8>(&mut m, y, x)?[0] = if on { 255 } else { 0 };
            }
        }
        Ok(m)
    }
}

fn hamming(a: &[bool], b: &[bool]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Rotate a square bit grid by `r` quarter turns, clockwise.
fn rotate(bits: &[bool], n: usize, r: usize) -> Vec<bool> {
    let mut cur = bits.to_vec();
    for _ in 0..(r % 4) {
        let mut next = vec![false; n * n];
        for y in 0..n {
            for x in 0..n {
                next[x * n + (n - 1 - y)] = cur[y * n + x];
            }
        }
        cur = next;
    }
    cur
}

/// `cv::aruco::DetectorParameters`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectorParameters {
    /// Smallest marker perimeter, as a fraction of the larger image side.
    pub min_marker_perimeter_rate: f64,
    /// Largest marker perimeter, likewise.
    pub max_marker_perimeter_rate: f64,
    /// Tolerance of `approxPolyDP`, relative to the perimeter.
    pub polygonal_approx_accuracy_rate: f64,
    /// Window for the adaptive threshold.
    pub adaptive_thresh_win_size: i32,
    /// Constant subtracted by the adaptive threshold.
    pub adaptive_thresh_constant: f64,
    /// Side of the square each candidate is warped to before reading bits.
    pub perspective_remove_pixel_per_cell: i32,
    /// Fraction of each cell ignored at its edge when deciding its bit.
    pub perspective_remove_ignored_margin_per_cell: f64,
}

impl Default for DetectorParameters {
    /// Upstream's defaults.
    fn default() -> Self {
        Self {
            min_marker_perimeter_rate: 0.03,
            max_marker_perimeter_rate: 4.0,
            polygonal_approx_accuracy_rate: 0.03,
            adaptive_thresh_win_size: 23,
            adaptive_thresh_constant: 7.0,
            perspective_remove_pixel_per_cell: 4,
            perspective_remove_ignored_margin_per_cell: 0.13,
        }
    }
}

/// The four corners of a candidate, in image coordinates.
pub type Quad = [(f64, f64); 4];

/// One detected marker.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectedMarker {
    /// Dictionary index.
    pub id: usize,
    /// The four corners, clockwise from the marker's own top-left.
    pub corners: Quad,
}

/// `cv::aruco::ArucoDetector::detectMarkers(image, corners, ids, rejectedImgPoints)`.
///
/// Returns the accepted markers and the quadrilaterals that looked like markers
/// but matched no code — upstream's `rejectedImgPoints`, which is the first
/// thing to look at when a marker is not being found.
pub fn detect_markers<S: MatLike + ?Sized>(
    image: &S,
    dict: &Dictionary,
    params: DetectorParameters,
) -> Result<(Vec<DetectedMarker>, Vec<Quad>)> {
    cv_assert!(image.channels() == 1, "detectMarkers takes a single-channel image");
    cv_assert!(!image.is_empty(), "detectMarkers needs a non-empty image");
    let (h, w) = (image.rows(), image.cols());

    let thr = cv_imgproc::misc::adaptive_threshold(
        image,
        255.0,
        cv_imgproc::misc::AdaptiveMethod::MeanC,
        ThresholdType::BinaryInv,
        params.adaptive_thresh_win_size | 1,
        params.adaptive_thresh_constant,
    )?;
    let (contours, _) = find_contours(&thr, RetrievalMode::List, ContourApproximation::Simple)?;

    let longer = f64::from(w.max(h));
    let (min_p, max_p) = (
        params.min_marker_perimeter_rate * longer,
        params.max_marker_perimeter_rate * longer,
    );

    let mut found = Vec::new();
    let mut rejected = Vec::new();
    for c in &contours {
        let per = cv_imgproc::shape::arc_length(c, true);
        if per < min_p || per > max_p {
            continue;
        }
        let approx =
            cv_imgproc::shape::approx_poly_dp(c, params.polygonal_approx_accuracy_rate * per, true)?;
        if approx.len() != 4 {
            continue;
        }
        // Order the corners consistently: counter-clockwise in image space,
        // which is clockwise once the y axis points down.
        let mut quad: [(f64, f64); 4] = [
            (f64::from(approx[0].x), f64::from(approx[0].y)),
            (f64::from(approx[1].x), f64::from(approx[1].y)),
            (f64::from(approx[2].x), f64::from(approx[2].y)),
            (f64::from(approx[3].x), f64::from(approx[3].y)),
        ];
        let cross = (quad[1].0 - quad[0].0) * (quad[2].1 - quad[0].1)
            - (quad[1].1 - quad[0].1) * (quad[2].0 - quad[0].0);
        if cross < 0.0 {
            quad.swap(1, 3);
        }

        match read_bits(image, &quad, dict.marker_size, params) {
            Ok(bits) => match dict.identify(&bits) {
                Some((id, rot)) => {
                    // Roll the corners so index 0 is the marker's own top-left.
                    let mut corners = quad;
                    corners.rotate_left((4 - rot) % 4);
                    found.push(DetectedMarker { id, corners });
                }
                None => rejected.push(quad),
            },
            Err(_) => rejected.push(quad),
        }
    }
    Ok((found, rejected))
}

/// Warp a candidate to a square and threshold its cells into bits.
fn read_bits<S: MatLike + ?Sized>(
    image: &S,
    quad: &Quad,
    marker_size: usize,
    params: DetectorParameters,
) -> Result<Vec<bool>> {
    // One border ring on each side, as every ArUco marker carries.
    let cells = marker_size + 2;
    let side = cells as i32 * params.perspective_remove_pixel_per_cell;
    let s = f64::from(side - 1);
    let dst = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)];
    let m = cv_imgproc::warp::get_perspective_transform(*quad, dst)?;
    let flat = cv_imgproc::warp::warp_perspective(
        image,
        m,
        cv_core::Size::new(side, side),
        cv_imgproc::resize::Interpolation::Linear,
        false,
        cv_core::border::BorderType::Replicate,
        cv_core::Scalar::default(),
    )?;

    // Otsu over the whole patch, so the split adapts to the local exposure.
    let bin = threshold(&flat, 0.0, 255.0, ThresholdType::Binary, ThresholdMethod::Otsu)?.dst;

    let per = params.perspective_remove_pixel_per_cell;
    let margin = (f64::from(per) * params.perspective_remove_ignored_margin_per_cell).round() as i32;
    let mut bits = Vec::with_capacity(marker_size * marker_size);
    for cy in 0..cells {
        for cx in 0..cells {
            let (mut on, mut total) = (0usize, 0usize);
            for y in (cy as i32 * per + margin)..((cy as i32 + 1) * per - margin).max(cy as i32 * per + margin + 1) {
                for x in (cx as i32 * per + margin)..((cx as i32 + 1) * per - margin).max(cx as i32 * per + margin + 1) {
                    if y >= side || x >= side {
                        continue;
                    }
                    total += 1;
                    if bin.at_f64(y, x, 0)? > 127.0 {
                        on += 1;
                    }
                }
            }
            let white = total > 0 && on * 2 > total;
            // The border ring must be dark; a light one means this is not a
            // marker, and reading its interior would be meaningless.
            let is_border = cy == 0 || cx == 0 || cy == cells - 1 || cx == cells - 1;
            if is_border {
                if white {
                    return Err(cv_core::Error::new(
                        cv_core::Code::StsBadArg,
                        "the candidate's border ring is not dark".to_string(),
                    ));
                }
            } else {
                bits.push(white);
            }
        }
    }
    Ok(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_generated_dictionary_separates_its_codes() {
        let d = Dictionary::extend(12, 5).unwrap();
        assert_eq!(d.codes.len(), 12);
        assert_eq!(d.marker_size, 5);
        // Every pair, in every rotation, differs in at least the minimum.
        let min = (5 * 5) / 4;
        for i in 0..d.codes.len() {
            for j in 0..d.codes.len() {
                for r in 0..4 {
                    if i == j && r == 0 {
                        continue;
                    }
                    let dist = hamming(&rotate(&d.codes[i], 5, r), &d.codes[j]);
                    assert!(dist >= min, "codes {i} and {j} at rotation {r} differ in only {dist}");
                }
            }
        }
    }

    #[test]
    fn rotation_is_a_quarter_turn_and_four_are_the_identity() {
        // 2x2 grid: [a b / c d] rotated clockwise is [c a / d b].
        let g = vec![true, false, false, false];
        let r1 = rotate(&g, 2, 1);
        assert_eq!(r1, vec![false, true, false, false]);
        assert_eq!(rotate(&g, 2, 4), g, "four turns is the identity");
        assert_eq!(rotate(&rotate(&g, 2, 1), 2, 3), g);
    }

    #[test]
    fn identify_recognises_a_code_in_every_rotation() {
        let d = Dictionary::extend(8, 4).unwrap();
        for (id, code) in d.codes.iter().enumerate() {
            for rot in 0..4 {
                let observed = rotate(code, 4, rot);
                let (got_id, got_rot) = d.identify(&observed).expect("its own code must match");
                assert_eq!(got_id, id, "rotation {rot} of marker {id}");
                assert_eq!(got_rot, rot, "reported rotation");
            }
        }
        // A grid of the wrong size is rejected rather than misread.
        assert!(d.identify(&[true; 9]).is_none());
    }

    #[test]
    fn a_generated_marker_has_a_dark_border_and_reads_back() {
        let d = Dictionary::extend(6, 4).unwrap();
        let img = d.generate_image_marker(2, 60, 1).unwrap();
        assert_eq!((img.rows(), img.cols()), (60, 60));
        // The outer ring is black, which is what the detector locks onto.
        for i in 0..60 {
            for (y, x) in [(0, i), (59, i), (i, 0), (i, 59)] {
                assert_eq!(img.at_f64(y, x, 0).unwrap(), 0.0, "border at ({y},{x})");
            }
        }
        assert!(d.generate_image_marker(99, 60, 1).is_err(), "no such marker");
        assert!(d.generate_image_marker(0, 0, 1).is_err());
    }

    /// End-to-end interchange. With the codes pinned to cv2's by
    /// [`the_predefined_codes_are_cv2s`], a marker this renders is byte-for-byte
    /// one cv2 renders — so detecting it here proves the round trip that
    /// [`Dictionary::extend`]'s generated codes could never make.
    ///
    /// Checked directly against cv2 4.13.0 when this was written: for ids 0, 3,
    /// 17 and 49 pasted at (40, 30) in a 160x140 scene, `cv2.aruco.detectMarkers`
    /// on cv2's own render and `detect_markers` here both report the same id.
    #[test]
    fn detect_markers_reads_a_predefined_marker_back() {
        let d = Dictionary::predefined(PredefinedDictionary::Dict4x4_50);
        for id in [0usize, 3, 17, 49] {
            let marker = d.generate_image_marker(id, 80, 1).unwrap();
            let (w, h) = (160i32, 140i32);
            let mut buf = vec![200u8; (w * h) as usize];
            for y in 0..80 {
                for x in 0..80 {
                    buf[((y + 30) * w + x + 40) as usize] =
                        marker.at_f64(y, x, 0).unwrap() as u8;
                }
            }
            let scene = Mat::from_vec(h, w, 1, &buf).unwrap();
            let (found, _) =
                detect_markers(&scene, &d, DetectorParameters::default()).unwrap();
            assert_eq!(found.len(), 1, "id {id}: got {} markers", found.len());
            assert_eq!(found[0].id, id, "id {id} read back wrong");
        }
    }

    #[test]
    fn detect_markers_finds_a_rendered_marker() {
        let d = Dictionary::extend(6, 4).unwrap();
        let marker = d.generate_image_marker(3, 80, 1).unwrap();
        // Paste it into a light background with a quiet margin.
        let (w, h) = (160i32, 140i32);
        let mut buf = vec![255u8; (w * h) as usize];
        let (ox, oy) = (40i32, 30i32);
        for y in 0..80 {
            for x in 0..80 {
                buf[((y + oy) * w + x + ox) as usize] = marker.at_f64(y, x, 0).unwrap() as u8;
            }
        }
        let scene = Mat::from_vec(h, w, 1, &buf).unwrap();

        let (found, _rejected) =
            detect_markers(&scene, &d, DetectorParameters::default()).unwrap();
        assert_eq!(found.len(), 1, "exactly one marker was drawn, got {}", found.len());
        assert_eq!(found[0].id, 3, "wrong id");
        // Its corners bound the region it was pasted into.
        let xs: Vec<f64> = found[0].corners.iter().map(|c| c.0).collect();
        let ys: Vec<f64> = found[0].corners.iter().map(|c| c.1).collect();
        let (x0, x1) = (xs.iter().cloned().fold(f64::MAX, f64::min), xs.iter().cloned().fold(f64::MIN, f64::max));
        let (y0, y1) = (ys.iter().cloned().fold(f64::MAX, f64::min), ys.iter().cloned().fold(f64::MIN, f64::max));
        assert!((x0 - f64::from(ox)).abs() < 6.0, "left edge {x0}");
        assert!((x1 - f64::from(ox + 79)).abs() < 6.0, "right edge {x1}");
        assert!((y0 - f64::from(oy)).abs() < 6.0, "top edge {y0}");
        assert!((y1 - f64::from(oy + 79)).abs() < 6.0, "bottom edge {y1}");
    }

    #[test]
    fn a_blank_scene_yields_nothing() {
        let d = Dictionary::extend(4, 4).unwrap();
        let blank = Mat::from_vec(80, 80, 1, &[200u8; 6400]).unwrap();
        let (found, _) = detect_markers(&blank, &d, DetectorParameters::default()).unwrap();
        assert!(found.is_empty(), "a blank scene has no markers, got {}", found.len());
    }

    #[test]
    fn the_predefined_families_have_upstreams_shapes() {
        assert_eq!(PredefinedDictionary::Dict4x4_50.marker_size(), 4);
        assert_eq!(PredefinedDictionary::Dict4x4_50.count(), 50);
        assert_eq!(PredefinedDictionary::Dict4x4_100.count(), 100);
        assert_eq!(PredefinedDictionary::Dict6x6_50.marker_size(), 6);
        assert_eq!(PredefinedDictionary::Dict7x7_50.marker_size(), 7);
        let d = Dictionary::predefined(PredefinedDictionary::Dict5x5_50);
        assert_eq!((d.marker_size, d.codes.len()), (5, 50));
    }

    /// The point of carrying upstream's tables: the *patterns* match, not just
    /// the shapes. Each string is cv2 4.13.0's `generateImageMarker` read cell
    /// by cell, so a marker printed there is this same id here.
    #[test]
    fn the_predefined_codes_are_cv2s() {
        for (fam, id, expect) in [
            (PredefinedDictionary::Dict4x4_50, 0, "1011010100110010"),
            (PredefinedDictionary::Dict4x4_50, 3, "1001100101000110"),
            (PredefinedDictionary::Dict4x4_50, 49, "0101000000010011"),
            (PredefinedDictionary::Dict5x5_50, 17, "0101111101010001001101111"),
            (
                PredefinedDictionary::Dict7x7_50,
                42,
                "0011010111101001001111111001111000110110011111000",
            ),
        ] {
            let d = Dictionary::predefined(fam);
            let got: String =
                d.codes[id].iter().map(|&b| if b { '1' } else { '0' }).collect();
            assert_eq!(got, expect, "{fam:?} id {id}");
        }
    }

    /// And the tables are usable both ways round: rendering a code and reading
    /// the cells back returns the id it was rendered from, at every rotation.
    #[test]
    fn a_predefined_marker_identifies_as_itself() {
        let d = Dictionary::predefined(PredefinedDictionary::Dict4x4_50);
        for id in [0usize, 7, 23, 49] {
            for rot in 0..4 {
                let bits = rotate(&d.codes[id], d.marker_size, rot);
                assert_eq!(
                    d.identify(&bits).map(|(i, _)| i),
                    Some(id),
                    "id {id} at rotation {rot}"
                );
            }
        }
    }

    /// Upstream's separation guarantee, checked rather than assumed: no code is
    /// within its correction radius of another code *or another's rotations*,
    /// so correction can never turn one marker into a different valid id.
    #[test]
    fn the_predefined_codes_stay_apart_under_rotation() {
        for fam in [
            PredefinedDictionary::Dict4x4_50,
            PredefinedDictionary::Dict4x4_100,
            PredefinedDictionary::Dict5x5_50,
            PredefinedDictionary::Dict6x6_50,
            PredefinedDictionary::Dict7x7_50,
        ] {
            let d = Dictionary::predefined(fam);
            let floor = 2 * d.max_correction_bits + 1;
            let mut worst = usize::MAX;
            for (i, a) in d.codes.iter().enumerate() {
                for b in &d.codes[i + 1..] {
                    for r in 0..4 {
                        worst = worst.min(hamming(&rotate(b, d.marker_size, r), a));
                    }
                }
            }
            assert!(
                worst >= floor,
                "{fam:?}: closest pair differs in {worst} bits, but correcting \
                 {} could reach it",
                d.max_correction_bits
            );
        }
    }

    #[test]
    fn detection_validates_its_input() {
        let d = Dictionary::extend(4, 4).unwrap();
        let colour = Mat::from_vec(20, 20, 3, &[0u8; 1200]).unwrap();
        assert!(detect_markers(&colour, &d, DetectorParameters::default()).is_err());
        assert!(Dictionary::extend(0, 4).is_err());
        assert!(Dictionary::extend(4, 1).is_err());
    }

    #[test]
    fn the_detector_defaults_match_upstream() {
        let p = DetectorParameters::default();
        assert_eq!(p.min_marker_perimeter_rate, 0.03);
        assert_eq!(p.max_marker_perimeter_rate, 4.0);
        assert_eq!(p.polygonal_approx_accuracy_rate, 0.03);
        assert_eq!(p.adaptive_thresh_win_size, 23);
        assert_eq!(p.adaptive_thresh_constant, 7.0);
        assert_eq!(p.perspective_remove_pixel_per_cell, 4);
    }
}
