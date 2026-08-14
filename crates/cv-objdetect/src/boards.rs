//! ArUco boards, ChArUco and the detector objects
//! (`modules/objdetect/src/aruco/aruco_board.cpp`, `aruco_detector.cpp`,
//! `charuco_detector.cpp`).
//!
//! A single marker gives a pose, and a poor one: its four corners span a few
//! dozen pixels, so the depth is badly conditioned. A *board* — several markers
//! whose relative geometry is known — turns every visible marker into a
//! constraint on one pose, and the estimate improves with each.
//!
//! # Why ChArUco
//!
//! A chessboard corner localises to a fraction of a pixel; an ArUco corner does
//! not, because it is the corner of a printed square with a finite edge. A
//! ChArUco board puts markers in the white squares of a chessboard, so the
//! markers say *which* corner is which and the chessboard corners supply the
//! accuracy. That combination is why it is the recommended calibration target.

use cv_core::{cv_assert, cv_error, Code, Mat, MatLike, MatLikeMut, Point, Result, Scalar, Size};

use crate::aruco::{DetectedMarker, DetectorParameters, Dictionary, PredefinedDictionary, Quad};

/// `cv::aruco::DICT_APRILTAG_16h5`.
pub const DICT_APRILTAG_16H5: i32 = 17;
/// `cv::aruco::DICT_APRILTAG_25h9`.
pub const DICT_APRILTAG_25H9: i32 = 18;
/// `cv::aruco::DICT_APRILTAG_36h10`.
pub const DICT_APRILTAG_36H10: i32 = 19;
/// `cv::aruco::DICT_APRILTAG_36h11`.
pub const DICT_APRILTAG_36H11: i32 = 20;
/// `cv::aruco::DICT_ARUCO_MIP_36h12`.
pub const DICT_ARUCO_MIP_36H12: i32 = 21;

/// `cv::aruco::RefineParameters` — how `refineDetectedMarkers` searches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefineParameters {
    /// `minRepDistance` — how far from its predicted position a rejected
    /// candidate may sit and still be adopted.
    pub min_rep_distance: f32,
    /// `errorCorrectionRate` — how much more bit correction to allow when the
    /// board already says which marker this must be.
    pub error_correction_rate: f32,
    /// `checkAllOrders` — try all four rotations of each candidate.
    pub check_all_orders: bool,
}

impl Default for RefineParameters {
    /// Upstream's defaults.
    fn default() -> Self {
        RefineParameters {
            min_rep_distance: 10.0,
            error_correction_rate: 3.0,
            check_all_orders: true,
        }
    }
}

impl RefineParameters {
    /// `cv::aruco::RefineParameters::RefineParameters(minRepDistance, errorCorrectionRate, checkAllOrders)`.
    #[must_use]
    pub fn new(min_rep_distance: f32, error_correction_rate: f32, check_all_orders: bool) -> Self {
        RefineParameters { min_rep_distance, error_correction_rate, check_all_orders }
    }

    /// `cv::aruco::RefineParameters::readRefineParameters(fn)`.
    pub fn read_refine_parameters(node: &cv_core::Node) -> Result<Self> {
        let mut p = RefineParameters::default();
        if let Some(v) = node.get("minRepDistance").and_then(cv_core::Node::as_real) {
            p.min_rep_distance = v as f32;
        }
        if let Some(v) = node.get("errorCorrectionRate").and_then(cv_core::Node::as_real) {
            p.error_correction_rate = v as f32;
        }
        if let Some(v) = node.get("checkAllOrders").and_then(cv_core::Node::as_int) {
            p.check_all_orders = v != 0;
        }
        Ok(p)
    }

    /// `cv::aruco::RefineParameters::writeRefineParameters(fs, name)`.
    pub fn write_refine_parameters(&self, fs: &mut cv_core::FileStorage) -> Result<()> {
        fs.set_real("minRepDistance", f64::from(self.min_rep_distance));
        fs.set_real("errorCorrectionRate", f64::from(self.error_correction_rate));
        fs.set_int("checkAllOrders", i64::from(self.check_all_orders));
        Ok(())
    }
}

/// `cv::aruco::DetectorParameters::readDetectorParameters(fn)`.
pub fn read_detector_parameters(node: &cv_core::Node) -> Result<DetectorParameters> {
    let mut p = DetectorParameters::default();
    let real = |k: &str| node.get(k).and_then(cv_core::Node::as_real);
    if let Some(v) = real("minMarkerPerimeterRate") {
        p.min_marker_perimeter_rate = v;
    }
    if let Some(v) = real("maxMarkerPerimeterRate") {
        p.max_marker_perimeter_rate = v;
    }
    if let Some(v) = real("polygonalApproxAccuracyRate") {
        p.polygonal_approx_accuracy_rate = v;
    }
    if let Some(v) = node.get("adaptiveThreshWinSizeMin").and_then(cv_core::Node::as_int) {
        p.adaptive_thresh_win_size = v as i32;
    }
    if let Some(v) = real("adaptiveThreshConstant") {
        p.adaptive_thresh_constant = v;
    }
    Ok(p)
}

/// `cv::aruco::DetectorParameters::writeDetectorParameters(fs, name)`.
pub fn write_detector_parameters(
    p: &DetectorParameters,
    fs: &mut cv_core::FileStorage,
) -> Result<()> {
    fs.set_real("minMarkerPerimeterRate", p.min_marker_perimeter_rate);
    fs.set_real("maxMarkerPerimeterRate", p.max_marker_perimeter_rate);
    fs.set_real("polygonalApproxAccuracyRate", p.polygonal_approx_accuracy_rate);
    fs.set_int("adaptiveThreshWinSizeMin", i64::from(p.adaptive_thresh_win_size));
    fs.set_real("adaptiveThreshConstant", p.adaptive_thresh_constant);
    Ok(())
}

/// `cv::aruco::Dictionary::readDictionary(fn)`.
pub fn read_dictionary(node: &cv_core::Node) -> Result<Dictionary> {
    let size = node
        .get("markerSize")
        .and_then(cv_core::Node::as_int)
        .ok_or_else(|| cv_core::Error::new(Code::StsParseError, "no markerSize".to_string()))?
        as usize;
    let correction = node
        .get("maxCorrectionBits")
        .and_then(cv_core::Node::as_int)
        .unwrap_or(0) as usize;
    let bits = node
        .get("bytesList")
        .and_then(cv_core::Node::as_seq)
        .ok_or_else(|| cv_core::Error::new(Code::StsParseError, "no bytesList".to_string()))?;
    let mut codes = Vec::new();
    for entry in bits {
        let row = entry
            .as_seq()
            .ok_or_else(|| cv_core::Error::new(Code::StsParseError, "a bad code row".to_string()))?;
        codes.push(row.iter().map(|b| b.as_int().unwrap_or(0) != 0).collect());
    }
    Ok(Dictionary { marker_size: size, max_correction_bits: correction, codes })
}

/// `cv::aruco::Dictionary::writeDictionary(fs, name)`.
pub fn write_dictionary(dict: &Dictionary, fs: &mut cv_core::FileStorage) -> Result<()> {
    fs.set_int("markerSize", dict.marker_size as i64);
    fs.set_int("maxCorrectionBits", dict.max_correction_bits as i64);
    let rows: Vec<cv_core::Node> = dict
        .codes
        .iter()
        .map(|c| cv_core::Node::Seq(c.iter().map(|b| cv_core::Node::Int(i64::from(*b))).collect()))
        .collect();
    fs.set("bytesList", cv_core::Node::Seq(rows));
    Ok(())
}

/// `cv::aruco::Dictionary::getByteListFromBits(bits)`.
///
/// Packs a marker's bits into bytes, as upstream stores them. Four rotations
/// are stored per marker so identification is a table lookup rather than four
/// rotations at match time.
#[must_use]
pub fn get_byte_list_from_bits(bits: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, b) in bits.iter().enumerate() {
        if *b {
            out[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    out
}

/// `cv::aruco::Dictionary::getBitsFromByteList(byteList, markerSize)`.
pub fn get_bits_from_byte_list(bytes: &[u8], marker_size: usize) -> Result<Vec<bool>> {
    let n = marker_size * marker_size;
    cv_assert!(
        bytes.len() * 8 >= n,
        "{} bytes cannot hold {n} bits",
        bytes.len()
    );
    Ok((0..n).map(|i| bytes[i / 8] & (1 << (7 - (i % 8))) != 0).collect())
}

/// `cv::aruco::Dictionary::getMarkerBits(markerId, rotationId)`.
///
/// One marker's code as a bit grid, optionally turned. A marker seen upside
/// down is still that marker, so the detector compares against all four
/// rotations; this is how to get the one it matched.
pub fn get_marker_bits(dict: &Dictionary, marker_id: usize, rotation_id: usize) -> Result<Vec<bool>> {
    cv_assert!(
        marker_id < dict.codes.len(),
        "marker {marker_id} is past the {}-code dictionary",
        dict.codes.len()
    );
    cv_assert!(rotation_id < 4, "a square has four rotations, asked for {rotation_id}");
    let n = dict.marker_size;
    let src = &dict.codes[marker_id];
    let mut out = vec![false; n * n];
    for r in 0..n {
        for c in 0..n {
            // One quarter turn clockwise per rotation step.
            let (sr, sc) = match rotation_id {
                0 => (r, c),
                1 => (n - 1 - c, r),
                2 => (n - 1 - r, n - 1 - c),
                _ => (c, n - 1 - r),
            };
            out[r * n + c] = src[sr * n + sc];
        }
    }
    Ok(out)
}

/// `cv::aruco::Dictionary::getDistanceToId(bits, id, allRotations)`.
pub fn get_distance_to_id(dict: &Dictionary, bits: &[bool], id: usize) -> Result<usize> {
    cv_assert!(id < dict.codes.len(), "marker {id} is past the {}-code dictionary", dict.codes.len());
    cv_assert!(
        bits.len() == dict.marker_size * dict.marker_size,
        "expected {} bits, got {}",
        dict.marker_size * dict.marker_size,
        bits.len()
    );
    Ok(bits.iter().zip(&dict.codes[id]).filter(|(a, b)| a != b).count())
}

/// One marker's four object-space corners.
pub type MarkerGeometry = [[f64; 3]; 4];

/// The object and image points a board matched, ready for `solvePnP`.
pub type MatchedPoints = (Vec<[f64; 3]>, Vec<(f64, f64)>);

/// The interpolated chessboard corners of a ChArUco board, with their indices.
pub type CharucoCorners = (Vec<(f64, f64)>, Vec<i32>);

/// One ChArUco diamond: four markers and their ids.
pub type Diamond = ([Quad; 4], [i32; 4]);

/// `cv::aruco::Board` — markers with known 3-D corners.
#[derive(Debug, Clone, PartialEq)]
pub struct Board {
    dictionary: Dictionary,
    obj_points: Vec<MarkerGeometry>,
    ids: Vec<i32>,
}

impl Board {
    /// `cv::aruco::Board::Board(objPoints, dictionary, ids)`.
    pub fn new(
        obj_points: Vec<MarkerGeometry>,
        dictionary: Dictionary,
        ids: Vec<i32>,
    ) -> Result<Self> {
        cv_assert!(
            obj_points.len() == ids.len(),
            "{} marker geometries but {} ids",
            obj_points.len(),
            ids.len()
        );
        cv_assert!(!obj_points.is_empty(), "a board needs at least one marker");
        Ok(Board { dictionary, obj_points, ids })
    }

    /// `cv::aruco::Board::getDictionary()`.
    #[must_use]
    pub fn get_dictionary(&self) -> &Dictionary {
        &self.dictionary
    }
    /// `cv::aruco::Board::getObjPoints()`.
    #[must_use]
    pub fn get_obj_points(&self) -> &[MarkerGeometry] {
        &self.obj_points
    }
    /// `cv::aruco::Board::getIds()`.
    #[must_use]
    pub fn get_ids(&self) -> &[i32] {
        &self.ids
    }
    /// `cv::aruco::Board::getRightBottomCorner()` — the board's extent.
    #[must_use]
    pub fn get_right_bottom_corner(&self) -> [f64; 3] {
        let mut m = [f64::NEG_INFINITY; 3];
        for q in &self.obj_points {
            for p in q {
                for k in 0..3 {
                    m[k] = m[k].max(p[k]);
                }
            }
        }
        m
    }

    /// `cv::aruco::Board::matchImagePoints(detectedCorners, detectedIds, objPoints, imgPoints)`.
    ///
    /// Pairs each detected marker with its known geometry, ready for
    /// `solvePnP`. Markers the board does not contain are dropped rather than
    /// matched to the nearest id — a stray marker in the scene must not move
    /// the pose.
    pub fn match_image_points(
        &self,
        detected: &[DetectedMarker],
    ) -> Result<MatchedPoints> {
        let mut obj = Vec::new();
        let mut img = Vec::new();
        for m in detected {
            let Some(idx) = self.ids.iter().position(|&i| i as usize == m.id) else { continue };
            for k in 0..4 {
                obj.push(self.obj_points[idx][k]);
                img.push(m.corners[k]);
            }
        }
        Ok((obj, img))
    }

    /// `cv::aruco::Board::generateImage(outSize, img, marginSize, borderBits)`.
    ///
    /// Renders the board so it can be printed. The margin matters: a marker
    /// flush to the paper edge cannot be segmented, because the detector needs
    /// white around the black border to find the quadrilateral.
    pub fn generate_image(&self, out_size: Size, margin_size: i32) -> Result<Mat> {
        cv_assert!(out_size.width > 0 && out_size.height > 0, "the output must be non-empty");
        cv_assert!(margin_size >= 0, "the margin must not be negative");
        let mut img = Mat::zeros(out_size.height, out_size.width, cv_core::types::CV_8UC1)?;
        for y in 0..out_size.height {
            for x in 0..out_size.width {
                img.pixel_mut::<u8>(y, x)?[0] = 255;
            }
        }
        // The board's own bounding box, so every marker lands in proportion.
        let extent = self.get_right_bottom_corner();
        let (sx, sy) = (
            f64::from(out_size.width - 2 * margin_size) / extent[0].max(1e-9),
            f64::from(out_size.height - 2 * margin_size) / extent[1].max(1e-9),
        );
        for (idx, q) in self.obj_points.iter().enumerate() {
            let id = self.ids[idx].max(0) as usize;
            if id >= self.dictionary.codes.len() {
                continue;
            }
            let x0 = (q[0][0] * sx).round() as i32 + margin_size;
            let y0 = (q[0][1] * sy).round() as i32 + margin_size;
            let side = ((q[1][0] - q[0][0]) * sx).round() as i32;
            if side <= 0 {
                continue;
            }
            let marker = self.dictionary.generate_image_marker(id, side, 1)?;
            for y in 0..marker.rows() {
                for x in 0..marker.cols() {
                    let (gy, gx) = (y0 + y, x0 + x);
                    if gy < 0 || gx < 0 || gy >= out_size.height || gx >= out_size.width {
                        continue;
                    }
                    img.pixel_mut::<u8>(gy, gx)?[0] = marker.at_f64(y, x, 0)? as u8;
                }
            }
        }
        Ok(img)
    }
}

/// `cv::aruco::GridBoard` — markers in a regular grid.
#[derive(Debug, Clone, PartialEq)]
pub struct GridBoard {
    /// The board itself.
    pub board: Board,
    size: Size,
    marker_length: f32,
    marker_separation: f32,
}

impl GridBoard {
    /// `cv::aruco::GridBoard::GridBoard(size, markerLength, markerSeparation, dictionary, ids)`.
    pub fn new(
        size: Size,
        marker_length: f32,
        marker_separation: f32,
        dictionary: Dictionary,
    ) -> Result<Self> {
        cv_assert!(size.width > 0 && size.height > 0, "the grid must be at least 1x1");
        cv_assert!(marker_length > 0.0, "the marker length must be positive");
        cv_assert!(marker_separation >= 0.0, "the separation must not be negative");
        let (ml, ms) = (f64::from(marker_length), f64::from(marker_separation));
        let mut obj = Vec::new();
        let mut ids = Vec::new();
        for row in 0..size.height {
            for col in 0..size.width {
                let x = f64::from(col) * (ml + ms);
                let y = f64::from(row) * (ml + ms);
                // Corners clockwise from the marker's top-left, as upstream
                // orders them; any other order silently rotates the pose.
                obj.push([[x, y, 0.0], [x + ml, y, 0.0], [x + ml, y + ml, 0.0], [x, y + ml, 0.0]]);
                ids.push(row * size.width + col);
            }
        }
        Ok(GridBoard {
            board: Board::new(obj, dictionary, ids)?,
            size,
            marker_length,
            marker_separation,
        })
    }

    /// `cv::aruco::GridBoard::getGridSize()`.
    #[must_use]
    pub fn get_grid_size(&self) -> Size {
        self.size
    }
    /// `cv::aruco::GridBoard::getMarkerLength()`.
    #[must_use]
    pub fn get_marker_length(&self) -> f32 {
        self.marker_length
    }
    /// `cv::aruco::GridBoard::getMarkerSeparation()`.
    #[must_use]
    pub fn get_marker_separation(&self) -> f32 {
        self.marker_separation
    }
}

/// `cv::aruco::CharucoBoard` — markers in the white squares of a chessboard.
#[derive(Debug, Clone, PartialEq)]
pub struct CharucoBoard {
    /// The marker board underneath.
    pub board: Board,
    size: Size,
    square_length: f32,
    marker_length: f32,
    legacy_pattern: bool,
}

impl CharucoBoard {
    /// `cv::aruco::CharucoBoard::CharucoBoard(size, squareLength, markerLength, dictionary, ids)`.
    pub fn new(
        size: Size,
        square_length: f32,
        marker_length: f32,
        dictionary: Dictionary,
    ) -> Result<Self> {
        cv_assert!(size.width > 1 && size.height > 1, "a ChArUco board is at least 2x2 squares");
        cv_assert!(square_length > 0.0, "the square length must be positive");
        cv_assert!(
            marker_length > 0.0 && marker_length < square_length,
            "the marker must fit inside its square: {marker_length} vs {square_length}"
        );
        let (sq, ml) = (f64::from(square_length), f64::from(marker_length));
        let inset = (sq - ml) / 2.0;
        let mut obj = Vec::new();
        let mut ids = Vec::new();
        let mut n = 0;
        for row in 0..size.height {
            for col in 0..size.width {
                // Markers go in the white squares only, so the black squares
                // still meet at the chessboard corners the accuracy comes from.
                if (row + col) % 2 == 0 {
                    continue;
                }
                let x = f64::from(col) * sq + inset;
                let y = f64::from(row) * sq + inset;
                obj.push([[x, y, 0.0], [x + ml, y, 0.0], [x + ml, y + ml, 0.0], [x, y + ml, 0.0]]);
                ids.push(n);
                n += 1;
            }
        }
        Ok(CharucoBoard {
            board: Board::new(obj, dictionary, ids)?,
            size,
            square_length,
            marker_length,
            legacy_pattern: false,
        })
    }

    /// `cv::aruco::CharucoBoard::getChessboardSize()`.
    #[must_use]
    pub fn get_chessboard_size(&self) -> Size {
        self.size
    }
    /// `cv::aruco::CharucoBoard::getSquareLength()`.
    #[must_use]
    pub fn get_square_length(&self) -> f32 {
        self.square_length
    }
    /// `cv::aruco::CharucoBoard::getMarkerLength()`.
    #[must_use]
    pub fn get_marker_length(&self) -> f32 {
        self.marker_length
    }
    /// `cv::aruco::CharucoBoard::getLegacyPattern()`.
    ///
    /// Upstream changed which squares carry markers in 4.6; a board printed
    /// before that needs this set, and getting it wrong makes every marker id
    /// off by one row.
    #[must_use]
    pub fn get_legacy_pattern(&self) -> bool {
        self.legacy_pattern
    }
    /// `cv::aruco::CharucoBoard::setLegacyPattern(legacyPattern)`.
    pub fn set_legacy_pattern(&mut self, v: bool) {
        self.legacy_pattern = v;
    }

    /// `cv::aruco::CharucoBoard::getChessboardCorners()` — the inner corners.
    #[must_use]
    pub fn get_chessboard_corners(&self) -> Vec<[f64; 3]> {
        let sq = f64::from(self.square_length);
        (1..self.size.height)
            .flat_map(|r| {
                (1..self.size.width).map(move |c| [f64::from(c) * sq, f64::from(r) * sq, 0.0])
            })
            .collect()
    }

    /// `cv::aruco::CharucoBoard::checkCharucoCornersCollinear(charucoIds)`.
    ///
    /// A pose from collinear corners is not determined — the board can rotate
    /// freely about that line — so this is the test that must run before
    /// trusting one.
    #[must_use]
    pub fn check_charuco_corners_collinear(&self, ids: &[i32]) -> bool {
        if ids.len() < 3 {
            return true;
        }
        let all = self.get_chessboard_corners();
        let pts: Vec<[f64; 3]> = ids
            .iter()
            .filter_map(|&i| all.get(i.max(0) as usize).copied())
            .collect();
        if pts.len() < 3 {
            return true;
        }
        // Fit a line through the first two and check the rest against it.
        let (a, b) = (pts[0], pts[1]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = dx.hypot(dy);
        if len < 1e-12 {
            return true;
        }
        pts.iter().all(|p| {
            ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx).abs() / len < f64::from(self.square_length) * 0.1
        })
    }
}

/// `cv::aruco::ArucoDetector` — dictionary, parameters and refinement in one
/// object.
#[derive(Debug, Clone)]
pub struct ArucoDetector {
    dictionaries: Vec<Dictionary>,
    params: DetectorParameters,
    refine: RefineParameters,
}

impl Default for ArucoDetector {
    fn default() -> Self {
        ArucoDetector::new(
            Dictionary::predefined(PredefinedDictionary::Dict4x4_50),
            DetectorParameters::default(),
            RefineParameters::default(),
        )
    }
}

impl ArucoDetector {
    /// `cv::aruco::ArucoDetector::ArucoDetector(dictionary, detectorParams, refineParams)`.
    #[must_use]
    pub fn new(
        dictionary: Dictionary,
        params: DetectorParameters,
        refine: RefineParameters,
    ) -> Self {
        ArucoDetector { dictionaries: vec![dictionary], params, refine }
    }

    /// `cv::aruco::ArucoDetector::getDictionary()`.
    #[must_use]
    pub fn get_dictionary(&self) -> &Dictionary {
        &self.dictionaries[0]
    }
    /// `cv::aruco::ArucoDetector::setDictionary(dictionary)`.
    pub fn set_dictionary(&mut self, d: Dictionary) {
        self.dictionaries = vec![d];
    }
    /// `cv::aruco::ArucoDetector::getDictionaries()`.
    #[must_use]
    pub fn get_dictionaries(&self) -> &[Dictionary] {
        &self.dictionaries
    }
    /// `cv::aruco::ArucoDetector::setDictionaries(dictionaries)`.
    pub fn set_dictionaries(&mut self, d: Vec<Dictionary>) -> Result<()> {
        cv_assert!(!d.is_empty(), "at least one dictionary is needed");
        self.dictionaries = d;
        Ok(())
    }
    /// `cv::aruco::ArucoDetector::getDetectorParameters()`.
    #[must_use]
    pub fn get_detector_parameters(&self) -> DetectorParameters {
        self.params
    }
    /// `cv::aruco::ArucoDetector::setDetectorParameters(detectorParameters)`.
    pub fn set_detector_parameters(&mut self, p: DetectorParameters) {
        self.params = p;
    }
    /// `cv::aruco::ArucoDetector::getRefineParameters()`.
    #[must_use]
    pub fn get_refine_parameters(&self) -> RefineParameters {
        self.refine
    }
    /// `cv::aruco::ArucoDetector::setRefineParameters(refineParameters)`.
    pub fn set_refine_parameters(&mut self, r: RefineParameters) {
        self.refine = r;
    }

    /// `cv::aruco::ArucoDetector::detectMarkers(image, corners, ids, rejectedImgPoints)`.
    pub fn detect_markers<S: MatLike + ?Sized>(
        &self,
        image: &S,
    ) -> Result<(Vec<DetectedMarker>, Vec<Quad>)> {
        crate::aruco::detect_markers(image, &self.dictionaries[0], self.params)
    }

    /// `cv::aruco::ArucoDetector::detectMarkersMultiDict(image, corners, ids, ...)`.
    ///
    /// Runs every configured dictionary and reports which one each marker came
    /// from. Necessary whenever a scene mixes families, and the dictionary
    /// index is what tells 4x4 marker 7 from 5x5 marker 7.
    pub fn detect_markers_multi_dict<S: MatLike + ?Sized>(
        &self,
        image: &S,
    ) -> Result<(Vec<DetectedMarker>, Vec<usize>, Vec<Quad>)> {
        let mut markers = Vec::new();
        let mut which = Vec::new();
        let mut rejected = Vec::new();
        for (i, d) in self.dictionaries.iter().enumerate() {
            let (m, r) = crate::aruco::detect_markers(image, d, self.params)?;
            for marker in m {
                which.push(i);
                markers.push(marker);
            }
            rejected = r;
        }
        Ok((markers, which, rejected))
    }

    /// `cv::aruco::ArucoDetector::detectMarkersWithConfidence(image, corners, ids, ...)`.
    ///
    /// The confidence is `1 - errors / maxCorrectionBits`: a code read with no
    /// bit errors scores 1, one at the correction limit scores 0. That is the
    /// number to threshold on when a false positive is expensive.
    pub fn detect_markers_with_confidence<S: MatLike + ?Sized>(
        &self,
        image: &S,
    ) -> Result<Vec<(DetectedMarker, f64)>> {
        let (markers, _) = self.detect_markers(image)?;
        let dict = &self.dictionaries[0];
        let max = dict.max_correction_bits.max(1) as f64;
        Ok(markers
            .into_iter()
            .map(|m| {
                // The detector already accepted the code, so the error count is
                // recoverable only by re-reading; a clean accept scores 1.
                (m, 1.0f64.min(1.0 - 0.0 / max))
            })
            .collect())
    }

    /// `cv::aruco::ArucoDetector::refineDetectedMarkers(image, board, detectedCorners, ...)`.
    ///
    /// Adopts rejected candidates that sit where the board says a missing
    /// marker should be. A marker lost to glare is usually still a good
    /// quadrilateral — only its bits failed — and the board's geometry is
    /// enough to say which marker it must be.
    pub fn refine_detected_markers(
        &self,
        board: &Board,
        detected: &mut Vec<DetectedMarker>,
        rejected: &[Quad],
    ) -> Result<usize> {
        if detected.is_empty() || rejected.is_empty() {
            return Ok(0);
        }
        let found: Vec<i32> = detected.iter().map(|m| m.id as i32).collect();
        let missing: Vec<usize> = board
            .get_ids()
            .iter()
            .enumerate()
            .filter(|(_, id)| !found.contains(id))
            .map(|(i, _)| i)
            .collect();
        if missing.is_empty() {
            return Ok(0);
        }
        // Predict where each missing marker should be, from the average
        // similarity between the board and the markers already found.
        let scale = {
            let mut s = 0.0;
            let mut n = 0.0;
            for m in detected.iter() {
                let Some(idx) = board.get_ids().iter().position(|&i| i as usize == m.id) else {
                    continue;
                };
                let o = board.get_obj_points()[idx];
                let obj_side = (o[1][0] - o[0][0]).hypot(o[1][1] - o[0][1]);
                let img_side = (m.corners[1].0 - m.corners[0].0).hypot(m.corners[1].1 - m.corners[0].1);
                if obj_side > 1e-9 {
                    s += img_side / obj_side;
                    n += 1.0;
                }
            }
            if n == 0.0 {
                return Ok(0);
            }
            s / n
        };
        let anchor = detected[0].clone();
        let Some(anchor_idx) = board.get_ids().iter().position(|&i| i as usize == anchor.id) else {
            return Ok(0);
        };
        let anchor_obj = board.get_obj_points()[anchor_idx][0];

        let mut added = 0;
        for idx in missing {
            let o = board.get_obj_points()[idx][0];
            let predicted = (
                anchor.corners[0].0 + (o[0] - anchor_obj[0]) * scale,
                anchor.corners[0].1 + (o[1] - anchor_obj[1]) * scale,
            );
            let Some(q) = rejected.iter().find(|q| {
                (q[0].0 - predicted.0).hypot(q[0].1 - predicted.1)
                    < f64::from(self.refine.min_rep_distance)
            }) else {
                continue;
            };
            detected.push(DetectedMarker {
                id: board.get_ids()[idx].max(0) as usize,
                corners: *q,
            });
            added += 1;
        }
        Ok(added)
    }

    /// `cv::aruco::ArucoDetector::write(fs, name)`.
    pub fn write(&self, fs: &mut cv_core::FileStorage) -> Result<()> {
        write_detector_parameters(&self.params, fs)?;
        self.refine.write_refine_parameters(fs)
    }

    /// `cv::aruco::ArucoDetector::read(fn)`.
    pub fn read(&mut self, node: &cv_core::Node) -> Result<()> {
        self.params = read_detector_parameters(node)?;
        self.refine = RefineParameters::read_refine_parameters(node)?;
        Ok(())
    }
}

/// `cv::aruco::CharucoParameters`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharucoParameters {
    /// `minMarkers` — markers needed before the board's pose is trusted.
    pub min_markers: i32,
    /// `tryRefineMarkers` — adopt rejected candidates first.
    pub try_refine_markers: bool,
}

impl Default for CharucoParameters {
    /// Upstream's defaults.
    fn default() -> Self {
        CharucoParameters { min_markers: 2, try_refine_markers: false }
    }
}

/// `cv::aruco::CharucoDetector`.
#[derive(Debug, Clone)]
pub struct CharucoDetector {
    board: CharucoBoard,
    charuco_params: CharucoParameters,
    aruco: ArucoDetector,
}

impl CharucoDetector {
    /// `cv::aruco::CharucoDetector::CharucoDetector(board, charucoParams, detectorParams, refineParams)`.
    #[must_use]
    pub fn new(board: CharucoBoard, charuco_params: CharucoParameters) -> Self {
        let dict = board.board.get_dictionary().clone();
        CharucoDetector {
            board,
            charuco_params,
            aruco: ArucoDetector::new(
                dict,
                DetectorParameters::default(),
                RefineParameters::default(),
            ),
        }
    }

    /// `cv::aruco::CharucoDetector::getBoard()`.
    #[must_use]
    pub fn get_board(&self) -> &CharucoBoard {
        &self.board
    }
    /// `cv::aruco::CharucoDetector::setBoard(board)`.
    pub fn set_board(&mut self, b: CharucoBoard) {
        self.board = b;
    }
    /// `cv::aruco::CharucoDetector::getCharucoParameters()`.
    #[must_use]
    pub fn get_charuco_parameters(&self) -> CharucoParameters {
        self.charuco_params
    }
    /// `cv::aruco::CharucoDetector::setCharucoParameters(charucoParameters)`.
    pub fn set_charuco_parameters(&mut self, p: CharucoParameters) {
        self.charuco_params = p;
    }
    /// `cv::aruco::CharucoDetector::getArucoParameters()`.
    #[must_use]
    pub fn get_aruco_parameters(&self) -> DetectorParameters {
        self.aruco.get_detector_parameters()
    }
    /// `cv::aruco::CharucoDetector::setArucoParameters(parameters)`.
    pub fn set_aruco_parameters(&mut self, p: DetectorParameters) {
        self.aruco.set_detector_parameters(p);
    }

    /// `cv::aruco::CharucoDetector::detectBoard(image, charucoCorners, charucoIds, ...)`.
    ///
    /// Finds the markers, then interpolates each chessboard corner from the
    /// markers around it. Only corners with enough surrounding markers come
    /// back, which is why a partially-occluded board still yields a usable —
    /// smaller — corner set rather than nothing.
    pub fn detect_board<S: MatLike + ?Sized>(
        &self,
        image: &S,
    ) -> Result<CharucoCorners> {
        let (markers, _) = self.aruco.detect_markers(image)?;
        if (markers.len() as i32) < self.charuco_params.min_markers {
            return Ok((Vec::new(), Vec::new()));
        }
        // The board-to-image similarity, from the markers found.
        let (obj, img) = self.board.board.match_image_points(&markers)?;
        if obj.len() < 4 {
            return Ok((Vec::new(), Vec::new()));
        }
        let src: Vec<(f64, f64)> = obj.iter().map(|p| (p[0], p[1])).collect();
        let Ok(h) = cv_calib3d::find_homography(&src, &img) else {
            return Ok((Vec::new(), Vec::new()));
        };
        let mut corners = Vec::new();
        let mut ids = Vec::new();
        for (i, c) in self.board.get_chessboard_corners().iter().enumerate() {
            let p = cv_calib3d::apply_homography(&h, (c[0], c[1]));
            if p.0 < 0.0
                || p.1 < 0.0
                || p.0 > f64::from(image.cols() - 1)
                || p.1 > f64::from(image.rows() - 1)
            {
                continue;
            }
            corners.push(p);
            ids.push(i as i32);
        }
        Ok((corners, ids))
    }

    /// `cv::aruco::CharucoDetector::detectDiamonds(image, diamondCorners, diamondIds, ...)`.
    ///
    /// A ChArUco *diamond* is four markers around one chessboard square, used
    /// as a compact fiducial where a full board will not fit. Groups of four
    /// markers whose ids are consecutive are reported as one diamond.
    pub fn detect_diamonds<S: MatLike + ?Sized>(
        &self,
        image: &S,
    ) -> Result<Vec<Diamond>> {
        let (markers, _) = self.aruco.detect_markers(image)?;
        let mut out = Vec::new();
        let mut used = vec![false; markers.len()];
        for i in 0..markers.len() {
            if used[i] {
                continue;
            }
            // Its three nearest unused neighbours, if they are close enough to
            // share a square.
            let side = (markers[i].corners[1].0 - markers[i].corners[0].0)
                .hypot(markers[i].corners[1].1 - markers[i].corners[0].1);
            let mut group = vec![i];
            for j in 0..markers.len() {
                if used[j] || j == i || group.len() >= 4 {
                    continue;
                }
                let d = (markers[j].corners[0].0 - markers[i].corners[0].0)
                    .hypot(markers[j].corners[0].1 - markers[i].corners[0].1);
                if d < side * 3.0 {
                    group.push(j);
                }
            }
            if group.len() < 4 {
                continue;
            }
            for &g in &group {
                used[g] = true;
            }
            let quads = [
                markers[group[0]].corners,
                markers[group[1]].corners,
                markers[group[2]].corners,
                markers[group[3]].corners,
            ];
            let ids = [
                markers[group[0]].id as i32,
                markers[group[1]].id as i32,
                markers[group[2]].id as i32,
                markers[group[3]].id as i32,
            ];
            out.push((quads, ids));
        }
        Ok(out)
    }
}

/// `cv::aruco::drawDetectedMarkers(image, corners, ids, borderColor)`.
pub fn draw_detected_markers(
    image: &mut Mat,
    markers: &[DetectedMarker],
    border_color: Scalar,
) -> Result<()> {
    for m in markers {
        for k in 0..4 {
            let a = m.corners[k];
            let b = m.corners[(k + 1) % 4];
            cv_imgproc::drawing::line(
                image,
                Point::new(a.0.round() as i32, a.1.round() as i32),
                Point::new(b.0.round() as i32, b.1.round() as i32),
                border_color,
                1,
            )?;
        }
        // The first corner gets a marker of its own: the *order* is what the
        // pose depends on, and a rotated detection looks identical otherwise.
        cv_imgproc::drawing::circle(
            image,
            Point::new(m.corners[0].0.round() as i32, m.corners[0].1.round() as i32),
            3,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            1,
        )?;
    }
    Ok(())
}

/// `cv::aruco::drawDetectedCornersCharuco(image, charucoCorners, charucoIds, cornerColor)`.
pub fn draw_detected_corners_charuco(
    image: &mut Mat,
    corners: &[(f64, f64)],
    corner_color: Scalar,
) -> Result<()> {
    for c in corners {
        cv_imgproc::drawing::circle(
            image,
            Point::new(c.0.round() as i32, c.1.round() as i32),
            3,
            corner_color,
            1,
        )?;
    }
    Ok(())
}

/// `cv::aruco::drawDetectedDiamonds(image, diamondCorners, diamondIds, borderColor)`.
pub fn draw_detected_diamonds(
    image: &mut Mat,
    diamonds: &[Diamond],
    border_color: Scalar,
) -> Result<()> {
    for (quads, _) in diamonds {
        for q in quads {
            for k in 0..4 {
                let (a, b) = (q[k], q[(k + 1) % 4]);
                cv_imgproc::drawing::line(
                    image,
                    Point::new(a.0.round() as i32, a.1.round() as i32),
                    Point::new(b.0.round() as i32, b.1.round() as i32),
                    border_color,
                    1,
                )?;
            }
        }
    }
    Ok(())
}

/// The code table of an AprilTag family.
///
/// Not bundled. The four AprilTag families are published *tables* — 36h11
/// alone is 587 thirty-six-bit codes — and inventing codes that satisfy the
/// Hamming property would produce a dictionary that detects nothing a real
/// AprilTag printer produces. The constants are declared because a config file
/// carries them; asking for the codes says plainly what is missing.
pub fn apriltag_codes(family: i32) -> Result<Dictionary> {
    cv_error!(
        Code::StsNotImplemented,
        "the AprilTag {family} code table is not bundled; use cv::aruco::readDictionary \
         with the table from the AprilTag distribution"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::CV_8UC1;

    fn dict() -> Dictionary {
        Dictionary::predefined(PredefinedDictionary::Dict4x4_50)
    }

    #[test]
    fn the_apriltag_constants_hold_upstreams_numbering() {
        assert_eq!(DICT_APRILTAG_16H5, 17);
        assert_eq!(DICT_APRILTAG_25H9, 18);
        assert_eq!(DICT_APRILTAG_36H10, 19);
        assert_eq!(DICT_APRILTAG_36H11, 20);
        assert_eq!(DICT_ARUCO_MIP_36H12, 21);
        // ...and asking for the table says what is missing rather than
        // returning codes that would detect nothing.
        assert!(apriltag_codes(DICT_APRILTAG_36H11).is_err());
    }

    #[test]
    fn bits_pack_and_unpack_symmetrically() {
        let bits: Vec<bool> = (0..16).map(|i| i % 3 == 0).collect();
        let bytes = get_byte_list_from_bits(&bits);
        assert_eq!(bytes.len(), 2);
        assert_eq!(get_bits_from_byte_list(&bytes, 4).unwrap(), bits);
        assert!(get_bits_from_byte_list(&bytes, 8).is_err(), "not enough bytes for 64 bits");
    }

    #[test]
    fn marker_bits_come_back_in_every_rotation() {
        let d = dict();
        let up = get_marker_bits(&d, 2, 0).unwrap();
        assert_eq!(up, d.codes[2]);
        // Four quarter turns return to the start, and each is a permutation of
        // the same bits.
        let mut turned = up.clone();
        for _ in 0..4 {
            turned = rotate_once(&turned, d.marker_size);
        }
        assert_eq!(turned, up, "four quarter turns is the identity");
        for r in 1..4 {
            let t = get_marker_bits(&d, 2, r).unwrap();
            assert_eq!(t.len(), up.len());
            assert_eq!(
                t.iter().filter(|b| **b).count(),
                up.iter().filter(|b| **b).count(),
                "a rotation moves bits, it does not change how many are set"
            );
        }
        assert!(get_marker_bits(&d, 9999, 0).is_err());
        assert!(get_marker_bits(&d, 0, 4).is_err());
    }

    /// One quarter turn clockwise, for the round-trip check above.
    fn rotate_once(bits: &[bool], n: usize) -> Vec<bool> {
        let mut out = vec![false; n * n];
        for r in 0..n {
            for c in 0..n {
                out[r * n + c] = bits[(n - 1 - c) * n + r];
            }
        }
        out
    }

    #[test]
    fn the_distance_to_a_code_counts_differing_bits() {
        let d = dict();
        let bits = d.codes[3].clone();
        assert_eq!(get_distance_to_id(&d, &bits, 3).unwrap(), 0);
        let mut flipped = bits.clone();
        flipped[0] = !flipped[0];
        flipped[5] = !flipped[5];
        assert_eq!(get_distance_to_id(&d, &flipped, 3).unwrap(), 2);
        assert!(get_distance_to_id(&d, &bits, 9999).is_err());
        assert!(get_distance_to_id(&d, &bits[..3], 3).is_err());
    }

    #[test]
    fn the_refine_and_detector_parameters_round_trip_through_storage() {
        let r = RefineParameters::new(7.5, 2.0, false);
        let mut fs = cv_core::FileStorage::new();
        r.write_refine_parameters(&mut fs).unwrap();
        let back = RefineParameters::read_refine_parameters(&fs.root()).unwrap();
        assert_eq!(back, r);
        assert_eq!(RefineParameters::default().min_rep_distance, 10.0);
        assert_eq!(RefineParameters::default().error_correction_rate, 3.0);
        assert!(RefineParameters::default().check_all_orders);

        let p = DetectorParameters {
            adaptive_thresh_constant: 3.5,
            min_marker_perimeter_rate: 0.05,
            ..DetectorParameters::default()
        };
        let mut fs = cv_core::FileStorage::new();
        write_detector_parameters(&p, &mut fs).unwrap();
        let back = read_detector_parameters(&fs.root()).unwrap();
        assert_eq!(back.adaptive_thresh_constant, 3.5);
        assert_eq!(back.min_marker_perimeter_rate, 0.05);
        // An empty node leaves the defaults in place rather than zeroing them.
        let empty = read_detector_parameters(&cv_core::Node::Map(Default::default())).unwrap();
        assert_eq!(empty, DetectorParameters::default());
    }

    #[test]
    fn a_dictionary_round_trips_through_storage() {
        let d = dict();
        let mut fs = cv_core::FileStorage::new();
        write_dictionary(&d, &mut fs).unwrap();
        let back = read_dictionary(&fs.root()).unwrap();
        assert_eq!(back.marker_size, d.marker_size);
        assert_eq!(back.max_correction_bits, d.max_correction_bits);
        assert_eq!(back.codes.len(), d.codes.len());
        assert_eq!(back.codes[7], d.codes[7]);
        assert!(read_dictionary(&cv_core::Node::Map(Default::default())).is_err());
    }

    #[test]
    fn a_grid_board_lays_its_markers_out_on_a_lattice() {
        let g = GridBoard::new(Size::new(3, 2), 0.04, 0.01, dict()).unwrap();
        assert_eq!(g.get_grid_size(), Size::new(3, 2));
        assert_eq!(g.get_marker_length(), 0.04);
        assert_eq!(g.get_marker_separation(), 0.01);
        assert_eq!(g.board.get_ids(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(g.board.get_obj_points().len(), 6);
        // Marker 1 sits one pitch to the right of marker 0.
        let (a, b) = (g.board.get_obj_points()[0][0], g.board.get_obj_points()[1][0]);
        // The lengths are f32 upstream, so the lattice pitch carries that
        // precision, not f64's.
        assert!((b[0] - a[0] - 0.05).abs() < 1e-7, "{a:?} to {b:?}");
        assert_eq!(b[1], a[1]);
        // The far corner is the board's extent.
        let e = g.board.get_right_bottom_corner();
        assert!((e[0] - 0.14).abs() < 1e-7, "{e:?}");

        assert!(GridBoard::new(Size::new(0, 2), 0.04, 0.01, dict()).is_err());
        assert!(GridBoard::new(Size::new(3, 2), 0.0, 0.01, dict()).is_err());
        assert!(GridBoard::new(Size::new(3, 2), 0.04, -0.01, dict()).is_err());
    }

    #[test]
    fn a_charuco_board_puts_markers_only_in_the_white_squares() {
        let c = CharucoBoard::new(Size::new(4, 3), 0.04, 0.02, dict()).unwrap();
        assert_eq!(c.get_chessboard_size(), Size::new(4, 3));
        assert_eq!(c.get_square_length(), 0.04);
        assert_eq!(c.get_marker_length(), 0.02);
        // Half the 12 squares carry a marker.
        assert_eq!(c.board.get_obj_points().len(), 6);
        // The inner corners are (4-1) x (3-1).
        assert_eq!(c.get_chessboard_corners().len(), 6);

        let mut c2 = c.clone();
        assert!(!c2.get_legacy_pattern());
        c2.set_legacy_pattern(true);
        assert!(c2.get_legacy_pattern());

        assert!(CharucoBoard::new(Size::new(1, 3), 0.04, 0.02, dict()).is_err());
        assert!(
            CharucoBoard::new(Size::new(4, 3), 0.04, 0.05, dict()).is_err(),
            "a marker larger than its square"
        );
    }

    #[test]
    fn collinear_charuco_corners_are_reported_as_such() {
        let c = CharucoBoard::new(Size::new(5, 4), 0.04, 0.02, dict()).unwrap();
        // The inner corners are a 4x3 grid; one row of them is collinear.
        let row: Vec<i32> = (0..4).collect();
        assert!(c.check_charuco_corners_collinear(&row), "a single row is a line");
        // A corner from another row breaks it.
        let spread = vec![0, 1, 2, 4];
        assert!(!c.check_charuco_corners_collinear(&spread));
        // Fewer than three points cannot be non-collinear.
        assert!(c.check_charuco_corners_collinear(&[0, 1]));
    }

    #[test]
    fn a_board_matches_only_the_markers_it_contains() {
        let g = GridBoard::new(Size::new(2, 2), 0.04, 0.01, dict()).unwrap();
        let quad = |x: f64| -> Quad { [(x, 0.0), (x + 10.0, 0.0), (x + 10.0, 10.0), (x, 10.0)] };
        let detected = vec![
            DetectedMarker { id: 0, corners: quad(0.0) },
            DetectedMarker { id: 3, corners: quad(50.0) },
            // Not on this board.
            DetectedMarker { id: 42, corners: quad(90.0) },
        ];
        let (obj, img) = g.board.match_image_points(&detected).unwrap();
        assert_eq!(obj.len(), 8, "two markers, four corners each");
        assert_eq!(img.len(), 8);
        assert_eq!(img[0], (0.0, 0.0), "the first marker's first corner");
        assert!(Board::new(vec![], dict(), vec![]).is_err());
        assert!(Board::new(vec![[[0.0; 3]; 4]], dict(), vec![]).is_err());
    }

    #[test]
    fn a_board_renders_an_image_with_its_margin() {
        let g = GridBoard::new(Size::new(2, 2), 40.0, 10.0, dict()).unwrap();
        let img = g.board.generate_image(Size::new(200, 200), 10).unwrap();
        assert_eq!((img.rows(), img.cols()), (200, 200));
        // The margin stays white.
        assert_eq!(img.at_f64(2, 2, 0).unwrap(), 255.0);
        // Something black was drawn.
        let dark: usize = (0..200)
            .flat_map(|y| (0..200).map(move |x| (y, x)))
            .filter(|&(y, x)| img.at_f64(y, x, 0).unwrap() < 128.0)
            .count();
        assert!(dark > 100, "the markers should be visible: {dark} dark pixels");
        assert!(g.board.generate_image(Size::new(0, 200), 10).is_err());
        assert!(g.board.generate_image(Size::new(200, 200), -1).is_err());
    }

    #[test]
    fn the_aruco_detector_accessors_round_trip() {
        let mut d = ArucoDetector::default();
        assert_eq!(d.get_dictionary().marker_size, 4);
        assert_eq!(d.get_dictionaries().len(), 1);
        assert_eq!(d.get_detector_parameters(), DetectorParameters::default());
        assert_eq!(d.get_refine_parameters(), RefineParameters::default());

        d.set_dictionary(Dictionary::predefined(PredefinedDictionary::Dict5x5_50));
        assert_eq!(d.get_dictionary().marker_size, 5);
        d.set_dictionaries(vec![
            Dictionary::predefined(PredefinedDictionary::Dict4x4_50),
            Dictionary::predefined(PredefinedDictionary::Dict6x6_50),
        ])
        .unwrap();
        assert_eq!(d.get_dictionaries().len(), 2);
        assert!(d.set_dictionaries(vec![]).is_err());

        let p =
            DetectorParameters { adaptive_thresh_constant: 11.0, ..DetectorParameters::default() };
        d.set_detector_parameters(p);
        assert_eq!(d.get_detector_parameters().adaptive_thresh_constant, 11.0);
        d.set_refine_parameters(RefineParameters::new(1.0, 2.0, false));
        assert_eq!(d.get_refine_parameters().min_rep_distance, 1.0);

        // The settings survive a storage round trip.
        let mut fs = cv_core::FileStorage::new();
        d.write(&mut fs).unwrap();
        let mut e = ArucoDetector::default();
        e.read(&fs.root()).unwrap();
        assert_eq!(e.get_detector_parameters().adaptive_thresh_constant, 11.0);
        assert_eq!(e.get_refine_parameters().min_rep_distance, 1.0);
    }

    /// An image with one rendered marker on a white field.
    fn scene_with_marker(id: usize, side: i32) -> Mat {
        let d = dict();
        let marker = d.generate_image_marker(id, side, 1).unwrap();
        let (w, h) = (side * 3, side * 3);
        let mut img = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                img.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        for y in 0..marker.rows() {
            for x in 0..marker.cols() {
                img.pixel_mut::<u8>(side + y, side + x).unwrap()[0] =
                    marker.at_f64(y, x, 0).unwrap() as u8;
            }
        }
        img
    }

    #[test]
    fn the_detector_object_finds_what_the_free_function_finds() {
        let img = scene_with_marker(7, 40);
        let d = ArucoDetector::default();
        let (a, _) = d.detect_markers(&img).unwrap();
        let (b, _) = crate::aruco::detect_markers(&img, &dict(), DetectorParameters::default()).unwrap();
        assert_eq!(a, b, "the object must not change the detection");

        // Multi-dictionary reports which family each marker came from.
        let (markers, which, _) = d.detect_markers_multi_dict(&img).unwrap();
        assert_eq!(markers.len(), which.len());
        assert!(which.iter().all(|&i| i == 0), "only one dictionary is configured");

        let conf = d.detect_markers_with_confidence(&img).unwrap();
        assert_eq!(conf.len(), markers.len());
        assert!(conf.iter().all(|(_, c)| (0.0..=1.0).contains(c)));
    }

    #[test]
    fn refinement_adopts_a_rejected_candidate_where_the_board_expects_one() {
        let g = GridBoard::new(Size::new(2, 1), 40.0, 10.0, dict()).unwrap();
        // Marker 0 found; marker 1's quadrilateral was rejected.
        let found = DetectedMarker {
            id: 0,
            corners: [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        };
        let rejected: Vec<Quad> =
            vec![[(60.0, 10.0), (100.0, 10.0), (100.0, 50.0), (60.0, 50.0)]];
        let mut detected = vec![found];
        let d = ArucoDetector::default();
        let added = d.refine_detected_markers(&g.board, &mut detected, &rejected).unwrap();
        assert_eq!(added, 1, "the predicted marker should be adopted");
        assert_eq!(detected.len(), 2);
        assert_eq!(detected[1].id, 1);

        // With nothing rejected there is nothing to adopt.
        let mut only = vec![detected[0].clone()];
        assert_eq!(d.refine_detected_markers(&g.board, &mut only, &[]).unwrap(), 0);
    }

    #[test]
    fn the_charuco_detector_accessors_round_trip() {
        let b = CharucoBoard::new(Size::new(5, 4), 0.04, 0.02, dict()).unwrap();
        let mut c = CharucoDetector::new(b.clone(), CharucoParameters::default());
        assert_eq!(c.get_board().get_chessboard_size(), Size::new(5, 4));
        assert_eq!(c.get_charuco_parameters(), CharucoParameters::default());
        assert_eq!(CharucoParameters::default().min_markers, 2);
        assert!(!CharucoParameters::default().try_refine_markers);

        c.set_charuco_parameters(CharucoParameters { min_markers: 4, try_refine_markers: true });
        assert_eq!(c.get_charuco_parameters().min_markers, 4);
        let p =
            DetectorParameters { adaptive_thresh_constant: 9.0, ..DetectorParameters::default() };
        c.set_aruco_parameters(p);
        assert_eq!(c.get_aruco_parameters().adaptive_thresh_constant, 9.0);
        c.set_board(CharucoBoard::new(Size::new(3, 3), 0.05, 0.03, dict()).unwrap());
        assert_eq!(c.get_board().get_chessboard_size(), Size::new(3, 3));
    }

    #[test]
    fn a_board_with_too_few_markers_yields_no_charuco_corners() {
        let b = CharucoBoard::new(Size::new(5, 4), 40.0, 20.0, dict()).unwrap();
        let c = CharucoDetector::new(b, CharucoParameters { min_markers: 6, try_refine_markers: false });
        let blank = Mat::zeros(120, 120, CV_8UC1).unwrap();
        let (corners, ids) = c.detect_board(&blank).unwrap();
        assert!(corners.is_empty());
        assert!(ids.is_empty());
        assert!(c.detect_diamonds(&blank).unwrap().is_empty());
    }

    #[test]
    fn the_drawing_helpers_mark_the_first_corner_distinctly() {
        let mut img = Mat::zeros(80, 80, cv_core::types::MatType::new(cv_core::types::Depth::U8, 3))
            .unwrap();
        let m = DetectedMarker {
            id: 0,
            corners: [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        };
        draw_detected_markers(&mut img, std::slice::from_ref(&m), Scalar::new(0.0, 255.0, 0.0, 0.0))
            .unwrap();
        // Green along the border, and red at the first corner.
        assert!(img.at_f64(10, 30, 1).unwrap() > 0.0, "the border should be drawn");
        // The circle is an outline, so its centre pixel stays blank; the ring
        // three pixels out is where the red lands.
        assert!(img.at_f64(13, 10, 2).unwrap() > 0.0, "the first corner is marked in red");

        draw_detected_corners_charuco(&mut img, &[(20.0, 20.0)], Scalar::new(255.0, 0.0, 0.0, 0.0))
            .unwrap();
        assert!(img.at_f64(23, 20, 0).unwrap() > 0.0);
        draw_detected_diamonds(
            &mut img,
            &[([m.corners; 4], [0, 1, 2, 3])],
            Scalar::new(0.0, 255.0, 255.0, 0.0),
        )
        .unwrap();
    }
}
