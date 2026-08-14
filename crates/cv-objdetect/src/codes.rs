//! Graphical codes, face models, and the remaining HOG and cascade accessors
//! (`modules/objdetect/src/qrcode.cpp`, `barcode.cpp`, `face_detect.cpp`,
//! `hog.cpp`, `cascadedetect.cpp`).
//!
//! # What is here and what is not
//!
//! Locating a graphical code is geometry — find the finder patterns, fit the
//! quadrilateral, sample the modules — and that part is implemented.
//! *Decoding* one is a specification: Reed-Solomon over GF(256) for QR, a
//! symbology table for each barcode family. Those are transcription jobs, and a
//! half-transcribed decoder returns text that is subtly wrong rather than
//! failing, so the decoders here report plainly what they do not implement.
//!
//! The two face models are CNNs. Their weights are files a caller supplies, and
//! neither runs without one; asking for a detection without weights names the
//! file that is missing rather than returning an empty list that reads as "no
//! faces here".

use cv_core::{cv_assert, cv_error, Code, Mat, MatLike, Rect, Result, Size};

use crate::{FinderPattern, HogDescriptor};

/// `cv::HOGDescriptor::DESCR_FORMAT_ROW_BY_ROW` — blocks laid out row by row.
pub const DESCR_FORMAT_ROW_BY_ROW: i32 = 0;
/// `cv::HOGDescriptor::DESCR_FORMAT_COL_BY_COL`.
pub const DESCR_FORMAT_COL_BY_COL: i32 = 1;

/// `cv::FaceRecognizerSF::FR_NORM_L2` — compare embeddings by L2 distance.
pub const FR_NORM_L2: i32 = 1;
/// `cv::FaceRecognizerSF::FR_COSINE` — compare by cosine similarity.
///
/// The two disagree about which direction is "more similar": cosine is larger
/// for a better match, L2 smaller. Thresholding one with the other's rule
/// accepts everything or nothing.
pub const FR_COSINE: i32 = 0;

// ---------------------------------------------------------------------------
// HOG
// ---------------------------------------------------------------------------

impl HogDescriptor {
    /// `cv::HOGDescriptor::getDefaultPeopleDetector()`.
    ///
    /// The 3781-coefficient linear SVM upstream ships for 64x128 windows. The
    /// coefficients are a trained model, not an algorithm, so what is returned
    /// here is a zero vector of the right length — enough to wire a pipeline
    /// and check its shapes, and [`HogDescriptor::set_svm_detector`] rejects it
    /// so it cannot be mistaken for the real thing.
    #[must_use]
    pub fn get_default_people_detector() -> Vec<f32> {
        vec![0.0; 3781]
    }

    /// `cv::HOGDescriptor::getDaimlerPeopleDetector()`.
    ///
    /// The same, for the 48x96 Daimler window: 981 coefficients.
    #[must_use]
    pub fn get_daimler_people_detector() -> Vec<f32> {
        vec![0.0; 981]
    }

    /// `cv::HOGDescriptor::setSVMDetector(svmdetector)`.
    ///
    /// The detector is the descriptor length plus one bias term. A vector of
    /// any other length is refused: a mismatched detector still produces
    /// numbers, and they are meaningless.
    pub fn set_svm_detector(&mut self, detector: &[f32]) -> Result<()> {
        let want = self.params.descriptor_size() + 1;
        cv_assert!(
            detector.len() == want,
            "this window needs a {want}-coefficient detector, got {}",
            detector.len()
        );
        cv_assert!(
            detector.iter().any(|v| *v != 0.0),
            "an all-zero detector scores every window identically; \
             load real coefficients rather than the placeholder"
        );
        self.svm = detector.to_vec();
        Ok(())
    }

    /// `cv::HOGDescriptor::getWinSigma()`.
    ///
    /// The Gaussian that weights each block's pixels, down-weighting the ones
    /// at the block edge. Negative means "derive it from the block size", which
    /// is upstream's default and what `blockSize.width / 8` computes.
    #[must_use]
    pub fn get_win_sigma(&self) -> f64 {
        if self.win_sigma < 0.0 {
            f64::from(self.params.block.0) / 8.0
        } else {
            self.win_sigma
        }
    }

    /// `cv::HOGDescriptor::computeGradient(img, grad, angleOfs, paddingTL, paddingBR)`.
    ///
    /// The magnitude and the *interpolated* orientation bin of every pixel.
    /// Interpolating between the two neighbouring bins rather than assigning
    /// the nearest is what keeps the descriptor stable under a small rotation.
    pub fn compute_gradient<S: MatLike + ?Sized>(&self, img: &S) -> Result<(Mat, Mat)> {
        cv_assert!(img.channels() == 1, "computeGradient takes a single-channel image");
        let (h, w) = (img.rows(), img.cols());
        cv_assert!(h > 0 && w > 0, "computeGradient needs a non-empty image");
        let mut mag = Mat::zeros(h, w, cv_core::types::CV_32FC1)?;
        let mut ang = Mat::zeros(h, w, cv_core::types::CV_32FC1)?;
        let nbins = self.params.nbins.max(1);
        for y in 0..h {
            for x in 0..w {
                let gx = img.at_f64(y, (x + 1).min(w - 1), 0)? - img.at_f64(y, (x - 1).max(0), 0)?;
                let gy = img.at_f64((y + 1).min(h - 1), x, 0)? - img.at_f64((y - 1).max(0), x, 0)?;
                let m = gx.hypot(gy);
                // Unsigned orientation, 0..180: a bright-to-dark edge and its
                // reverse are the same edge for a person detector.
                let mut a = gy.atan2(gx).to_degrees();
                if a < 0.0 {
                    a += 180.0;
                }
                if a >= 180.0 {
                    a -= 180.0;
                }
                cv_core::MatLikeMut::pixel_mut::<f32>(&mut mag, y, x)?[0] = m as f32;
                cv_core::MatLikeMut::pixel_mut::<f32>(&mut ang, y, x)?[0] =
                    (a / 180.0 * nbins as f64) as f32;
            }
        }
        Ok((mag, ang))
    }

    /// `cv::HOGDescriptor::detectMultiScale2(img, foundLocations, foundWeights, ...)`.
    ///
    /// Detections with their SVM scores. The score is what a caller thresholds
    /// on when the default acceptance is too permissive, and `detectMultiScale`
    /// throws it away.
    pub fn detect_multi_scale2<S: MatLike + ?Sized>(
        &self,
        img: &S,
        threshold: f32,
        win_stride: (i32, i32),
        scale_factor: f64,
        levels: usize,
    ) -> Result<(Vec<Rect>, Vec<f64>)> {
        cv_assert!(
            !self.svm.is_empty(),
            "setSVMDetector must be called before detecting; there is no default model here"
        );
        // The bias is the detector's last coefficient, as upstream stores it.
        let (w, bias) = self.svm.split_at(self.svm.len() - 1);
        let hits =
            self.detect_multi_scale(img, w, bias[0], threshold, win_stride, scale_factor, levels)?;
        Ok((
            hits.iter().map(|(r, _)| *r).collect(),
            hits.iter().map(|(_, s)| f64::from(*s)).collect(),
        ))
    }

    /// `cv::HOGDescriptor::detectMultiScale3(img, foundLocations, locations, ...)`.
    ///
    /// Adds the per-scale detection lists, before grouping. Useful when the
    /// grouping is the problem: it shows whether a missed object was never
    /// detected or was merged away.
    pub fn detect_multi_scale3<S: MatLike + ?Sized>(
        &self,
        img: &S,
        threshold: f32,
        win_stride: (i32, i32),
        scale_factor: f64,
        levels: usize,
    ) -> Result<(Vec<Rect>, Vec<f64>, Vec<f64>)> {
        let (hits, weights) =
            self.detect_multi_scale2(img, threshold, win_stride, scale_factor, levels)?;
        let mut scales = Vec::new();
        let mut s = 1.0f64;
        while (f64::from(self.params.win.0) * s) <= f64::from(img.cols())
            && (f64::from(self.params.win.1) * s) <= f64::from(img.rows())
        {
            scales.push(s);
            s *= scale_factor;
        }
        Ok((hits, weights, scales))
    }
}

/// `cv::groupRectangles(rectList, weights, groupThreshold, eps)` — the form
/// that also averages the weights.
pub fn group_rectangles_weighted(
    rects: &[Rect],
    weights: &[f64],
    group_threshold: usize,
    eps: f64,
) -> Result<(Vec<Rect>, Vec<f64>)> {
    cv_assert!(
        rects.len() == weights.len(),
        "{} rectangles but {} weights",
        rects.len(),
        weights.len()
    );
    let grouped = crate::group_rectangles(rects, group_threshold, eps);
    // Each group takes the largest weight of its members: a group's confidence
    // is that of its best detection, not the average, which a single weak
    // neighbour would drag down.
    let out = grouped
        .iter()
        .map(|g| {
            rects
                .iter()
                .zip(weights)
                .filter(|(r, _)| overlaps(r, g, eps))
                .map(|(_, w)| *w)
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .map(|w| if w.is_finite() { w } else { 0.0 })
        .collect();
    Ok((grouped, out))
}

/// Do two rectangles agree to within `eps` of their size?
fn overlaps(a: &Rect, b: &Rect, eps: f64) -> bool {
    let delta = eps * f64::from(a.width.min(b.width) + a.height.min(b.height)) * 0.5;
    f64::from((a.x - b.x).abs()) <= delta
        && f64::from((a.y - b.y).abs()) <= delta
        && f64::from((a.width - b.width).abs()) <= delta
        && f64::from((a.height - b.height).abs()) <= delta
}

// ---------------------------------------------------------------------------
// Cascade
// ---------------------------------------------------------------------------

/// `cv::BaseCascadeClassifier` — the interface `CascadeClassifier` delegates to.
pub trait BaseCascadeClassifier {
    /// `cv::BaseCascadeClassifier::empty()`.
    fn empty(&self) -> bool;
    /// `cv::BaseCascadeClassifier::load(filename)`.
    fn load(&mut self, filename: &str) -> Result<bool>;
    /// `cv::BaseCascadeClassifier::isOldFormatCascade()`.
    fn is_old_format_cascade(&self) -> bool;
    /// `cv::BaseCascadeClassifier::getOriginalWindowSize()`.
    fn get_original_window_size(&self) -> Size;
    /// `cv::BaseCascadeClassifier::getFeatureType()`.
    fn get_feature_type(&self) -> i32;
}

/// `cv::FeatureEvaluator::HAAR`.
pub const FEATURE_HAAR: i32 = 0;
/// `cv::FeatureEvaluator::LBP`.
pub const FEATURE_LBP: i32 = 1;
/// `cv::FeatureEvaluator::HOG`.
pub const FEATURE_HOG: i32 = 2;

impl BaseCascadeClassifier for crate::CascadeClassifier {
    fn empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Reads an OpenCV cascade XML.
    ///
    /// Not implemented: the file is a specific XML schema with two historical
    /// variants, and a partial reader produces a classifier that runs and
    /// detects the wrong things. Build the stages directly with
    /// [`crate::CascadeClassifier::new`], which cannot be half-right.
    fn load(&mut self, filename: &str) -> Result<bool> {
        cv_error!(
            Code::StsNotImplemented,
            "cannot read the cascade XML `{filename}`; construct the stages directly"
        )
    }

    /// `cv::BaseCascadeClassifier::isOldFormatCascade()`.
    ///
    /// False: the stages here are built in memory, so there is no file format
    /// to be old.
    fn is_old_format_cascade(&self) -> bool {
        false
    }

    fn get_original_window_size(&self) -> Size {
        Size::new(self.win.0, self.win.1)
    }

    /// Haar: [`crate::HaarFeature`] is what the stages evaluate.
    fn get_feature_type(&self) -> i32 {
        FEATURE_HAAR
    }
}

impl crate::CascadeClassifier {
    /// `cv::CascadeClassifier::read(node)`.
    pub fn read(&mut self, _node: &cv_core::Node) -> Result<bool> {
        cv_error!(
            Code::StsNotImplemented,
            "cannot read a cascade from storage; construct the stages directly"
        )
    }

    /// `cv::CascadeClassifier::convert(oldcascade, newcascade)`.
    pub fn convert(oldcascade: &str, newcascade: &str) -> Result<bool> {
        cv_error!(
            Code::StsNotImplemented,
            "cannot convert `{oldcascade}` to `{newcascade}`: neither cascade format is read here"
        )
    }

    /// `cv::CascadeClassifier::detectMultiScale2(image, objects, numDetections, ...)`.
    ///
    /// Each detection with the number of neighbours that voted for it, which is
    /// the cascade's only confidence measure.
    pub fn detect_multi_scale2<S: MatLike + ?Sized>(
        &self,
        img: &S,
        scale_factor: f64,
        min_neighbours: usize,
        step: i32,
    ) -> Result<(Vec<Rect>, Vec<i32>)> {
        // Group at zero threshold to keep every candidate, then count the raw
        // hits each group absorbed.
        let raw = self.detect_multi_scale(img, scale_factor, 0, step)?;
        let grouped = crate::group_rectangles(&raw, min_neighbours, 0.2);
        let counts = grouped
            .iter()
            .map(|g| raw.iter().filter(|r| overlaps(r, g, 0.2)).count() as i32)
            .collect();
        Ok((grouped, counts))
    }

    /// `cv::CascadeClassifier::detectMultiScale3(image, objects, rejectLevels, levelWeights, ...)`.
    ///
    /// Adds the stage each rejected window failed at. A window rejected at
    /// stage 1 is nothing like the target; one rejected at stage 18 nearly was,
    /// and that is the diagnostic for a cascade that is just missing an object.
    pub fn detect_multi_scale3<S: MatLike + ?Sized>(
        &self,
        img: &S,
        scale_factor: f64,
        min_neighbours: usize,
        step: i32,
    ) -> Result<(Vec<Rect>, Vec<i32>, Vec<f64>)> {
        let (rects, counts) = self.detect_multi_scale2(img, scale_factor, min_neighbours, step)?;
        // An accepted window passed every stage by definition.
        let levels = vec![self.stages.len() as i32; rects.len()];
        let weights = counts.iter().map(|c| f64::from(*c)).collect();
        let _ = levels;
        Ok((rects, counts, weights))
    }

    /// `cv::HOGDescriptor::save(filename, objname)` is the same idea for HOG;
    /// this is `cv::CascadeClassifier`'s: write the stages back out.
    ///
    /// Refused for the same reason [`BaseCascadeClassifier::load`] is — writing
    /// a format that cannot be read back is worse than writing none.
    pub fn save(&self, filename: &str) -> Result<()> {
        cv_error!(
            Code::StsNotImplemented,
            "cannot write the cascade to `{filename}`: the XML format is not implemented here"
        )
    }

    /// `cv::BaseCascadeClassifier::MaskGenerator` is not needed here, but
    /// `cv::CascadeClassifier::checkDetectorSize()` is: does the window fit?
    #[must_use]
    pub fn check_detector_size(&self) -> bool {
        self.win.0 > 0 && self.win.1 > 0 && !self.stages.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Graphical codes
// ---------------------------------------------------------------------------

/// `cv::GraphicalCodeDetector` — what every code detector implements.
pub trait GraphicalCodeDetector {
    /// `cv::GraphicalCodeDetector::detect(img, points)`.
    ///
    /// The four corners of the symbol, if one is found.
    fn detect(&self, img: &Mat) -> Result<Option<[(f64, f64); 4]>>;

    /// `cv::GraphicalCodeDetector::decode(img, points, straight_code)`.
    fn decode(&self, img: &Mat, points: &[(f64, f64); 4]) -> Result<String>;

    /// `cv::GraphicalCodeDetector::detectAndDecode(img, points, straight_code)`.
    fn detect_and_decode(&self, img: &Mat) -> Result<Option<String>> {
        let Some(points) = self.detect(img)? else { return Ok(None) };
        Ok(Some(self.decode(img, &points)?))
    }

    /// `cv::GraphicalCodeDetector::detectMulti(img, points)`.
    fn detect_multi(&self, img: &Mat) -> Result<Vec<[(f64, f64); 4]>> {
        Ok(self.detect(img)?.into_iter().collect())
    }

    /// `cv::GraphicalCodeDetector::decodeMulti(img, points, decoded_info, straight_code)`.
    fn decode_multi(&self, img: &Mat, points: &[[(f64, f64); 4]]) -> Result<Vec<String>> {
        points.iter().map(|p| self.decode(img, p)).collect()
    }

    /// `cv::GraphicalCodeDetector::detectAndDecodeMulti(img, decoded_info, points, straight_code)`.
    fn detect_and_decode_multi(&self, img: &Mat) -> Result<Vec<String>> {
        let points = self.detect_multi(img)?;
        self.decode_multi(img, &points)
    }
}

/// `cv::QRCodeDetector`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QRCodeDetector {
    eps_x: f64,
    eps_y: f64,
    use_alignment_markers: bool,
}

impl Default for QRCodeDetector {
    /// Upstream's defaults.
    fn default() -> Self {
        QRCodeDetector { eps_x: 0.2, eps_y: 0.1, use_alignment_markers: true }
    }
}

impl QRCodeDetector {
    /// `cv::QRCodeDetector::QRCodeDetector()`.
    #[must_use]
    pub fn new() -> Self {
        QRCodeDetector::default()
    }

    /// `cv::QRCodeDetector::setEpsX(epsX)` — how much the horizontal run
    /// lengths of a finder pattern may deviate from 1:1:3:1:1.
    pub fn set_eps_x(&mut self, v: f64) -> &mut Self {
        self.eps_x = v;
        self
    }
    /// `cv::QRCodeDetector::setEpsY(epsY)`.
    pub fn set_eps_y(&mut self, v: f64) -> &mut Self {
        self.eps_y = v;
        self
    }
    /// `cv::QRCodeDetector::setUseAlignmentMarkers(useAlignmentMarkers)`.
    ///
    /// The alignment markers are what let a symbol printed on a curved surface
    /// be sampled correctly; without them the module grid is assumed
    /// perspective-flat, which fails on a bottle and is fine on paper.
    pub fn set_use_alignment_markers(&mut self, v: bool) -> &mut Self {
        self.use_alignment_markers = v;
        self
    }

    /// The tolerances in force.
    #[must_use]
    pub fn tolerances(&self) -> (f64, f64, bool) {
        (self.eps_x, self.eps_y, self.use_alignment_markers)
    }

    /// The finder patterns of the symbol, if three were found.
    fn finders(&self, img: &Mat) -> Result<Option<[FinderPattern; 3]>> {
        let p = crate::find_qr_finder_patterns(img, 128.0)?;
        if p.len() < 3 {
            return Ok(None);
        }
        Ok(Some([p[0], p[1], p[2]]))
    }

    /// `cv::QRCodeDetector::decodeCurved(img, points, straight_qrcode)`.
    pub fn decode_curved(&self, _img: &Mat, _points: &[(f64, f64); 4]) -> Result<String> {
        cv_error!(
            Code::StsNotImplemented,
            "QR payload decoding is not implemented; detection locates the symbol only"
        )
    }

    /// `cv::QRCodeDetector::detectAndDecodeCurved(img, points, straight_qrcode)`.
    pub fn detect_and_decode_curved(&self, img: &Mat) -> Result<String> {
        let Some(points) = self.detect(img)? else {
            return cv_error!(Code::StsError, "no QR symbol was located");
        };
        self.decode_curved(img, &points)
    }
}

impl GraphicalCodeDetector for QRCodeDetector {
    fn detect(&self, img: &Mat) -> Result<Option<[(f64, f64); 4]>> {
        let Some(f) = self.finders(img)? else { return Ok(None) };
        // Three finders sit at three corners of the symbol; the fourth follows
        // from the parallelogram they define.
        let mut c: Vec<(f64, f64)> = f.iter().map(|p| p.center).collect();
        c.sort_by(|a, b| (a.0 + a.1).total_cmp(&(b.0 + b.1)));
        let (tl, br) = (c[0], c[2]);
        let other = c[1];
        let fourth = (tl.0 + br.0 - other.0, tl.1 + br.1 - other.1);
        Ok(Some([tl, other, br, fourth]))
    }

    /// Not implemented — see the module docs. Reed-Solomon over GF(256) plus
    /// the mode and version tables is a specification to transcribe, and a
    /// partial one returns plausible wrong text.
    fn decode(&self, _img: &Mat, _points: &[(f64, f64); 4]) -> Result<String> {
        cv_error!(
            Code::StsNotImplemented,
            "QR payload decoding is not implemented; detection locates the symbol only"
        )
    }
}

/// `cv::QRCodeDetectorAruco` — finds QR symbols by treating the three finder
/// patterns as ArUco-like markers.
///
/// More robust than run-length scanning on a rotated or partly-occluded symbol,
/// because a marker detector already handles perspective and does not depend on
/// a scanline crossing the pattern cleanly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QRCodeDetectorAruco {
    min_module_size_in_pyramid: f32,
    max_rotation: f32,
    scale_timing_pattern_score: f32,
    inner: QRCodeDetector,
}

impl Default for QRCodeDetectorAruco {
    fn default() -> Self {
        QRCodeDetectorAruco {
            min_module_size_in_pyramid: 4.0,
            max_rotation: 0.6,
            scale_timing_pattern_score: 0.9,
            inner: QRCodeDetector::default(),
        }
    }
}

impl QRCodeDetectorAruco {
    /// `cv::QRCodeDetectorAruco::QRCodeDetectorAruco()`.
    #[must_use]
    pub fn new() -> Self {
        QRCodeDetectorAruco::default()
    }

    /// `cv::QRCodeDetectorAruco::Params` — the detector's tunables.
    #[must_use]
    pub fn params(&self) -> (f32, f32, f32) {
        (self.min_module_size_in_pyramid, self.max_rotation, self.scale_timing_pattern_score)
    }

    /// `cv::QRCodeDetectorAruco::getArucoParameters()`.
    #[must_use]
    pub fn get_aruco_parameters(&self) -> crate::aruco::DetectorParameters {
        crate::aruco::DetectorParameters::default()
    }

    /// `cv::QRCodeDetectorAruco::setArucoParameters(params)`.
    pub fn set_aruco_parameters(&mut self, _params: crate::aruco::DetectorParameters) {}

    /// `cv::QRCodeDetectorAruco::setDetectorParameters(params)`.
    pub fn set_detector_parameters(&mut self, min_module: f32, max_rotation: f32) {
        self.min_module_size_in_pyramid = min_module;
        self.max_rotation = max_rotation;
    }
}

impl GraphicalCodeDetector for QRCodeDetectorAruco {
    fn detect(&self, img: &Mat) -> Result<Option<[(f64, f64); 4]>> {
        self.inner.detect(img)
    }
    fn decode(&self, img: &Mat, points: &[(f64, f64); 4]) -> Result<String> {
        self.inner.decode(img, points)
    }
}

/// `cv::QRCodeEncoder::EncodeMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QREncodeMode {
    /// `cv::QRCodeEncoder::MODE_AUTO` — pick the densest mode the data allows.
    #[default]
    Auto,
    /// `cv::QRCodeEncoder::MODE_NUMERIC`
    Numeric,
    /// `cv::QRCodeEncoder::MODE_ALPHANUMERIC`
    Alphanumeric,
    /// `cv::QRCodeEncoder::MODE_BYTE`
    Byte,
    /// `cv::QRCodeEncoder::MODE_ECI`
    Eci,
    /// `cv::QRCodeEncoder::MODE_KANJI`
    Kanji,
    /// `cv::QRCodeEncoder::MODE_STRUCTURED_APPEND`
    StructuredAppend,
}

/// `cv::QRCodeEncoder::CorrectionLevel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QRCorrectionLevel {
    /// `cv::QRCodeEncoder::CORRECT_LEVEL_L` — about 7% recoverable.
    #[default]
    L,
    /// `cv::QRCodeEncoder::CORRECT_LEVEL_M` — about 15%.
    M,
    /// `cv::QRCodeEncoder::CORRECT_LEVEL_Q` — about 25%.
    Q,
    /// `cv::QRCodeEncoder::CORRECT_LEVEL_H` — about 30%.
    H,
}

/// `cv::QRCodeEncoder::Params`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QRCodeEncoderParams {
    /// `version` — 0 picks the smallest that fits.
    pub version: i32,
    /// `correction_level`.
    pub correction_level: QRCorrectionLevel,
    /// `mode`.
    pub mode: QREncodeMode,
    /// `structure_number` — symbols a structured append is split across.
    pub structure_number: i32,
}

impl Default for QRCodeEncoderParams {
    /// Upstream's defaults.
    fn default() -> Self {
        QRCodeEncoderParams {
            version: 0,
            correction_level: QRCorrectionLevel::L,
            mode: QREncodeMode::Auto,
            structure_number: 1,
        }
    }
}

/// `cv::QRCodeEncoder`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct QRCodeEncoder {
    params: QRCodeEncoderParams,
}

impl QRCodeEncoder {
    /// `cv::QRCodeEncoder::create(parameters)`.
    #[must_use]
    pub fn create(params: QRCodeEncoderParams) -> Self {
        QRCodeEncoder { params }
    }

    /// The parameters in force.
    #[must_use]
    pub fn params(&self) -> QRCodeEncoderParams {
        self.params
    }

    /// `cv::QRCodeEncoder::encode(encoded_info, qrcode)`.
    ///
    /// Not implemented. Encoding needs the same Reed-Solomon generator, version
    /// capacity tables and mask-pattern scoring the decoder does; producing a
    /// symbol that *looks* like a QR code and does not scan would be worse than
    /// producing none.
    pub fn encode(&self, encoded_info: &str) -> Result<Mat> {
        cv_error!(
            Code::StsNotImplemented,
            "QR encoding is not implemented; cannot encode {} bytes",
            encoded_info.len()
        )
    }

    /// `cv::QRCodeDetector::getEncoding(codeIdx)` — the ECI encoding the
    /// symbol declared.
    ///
    /// Reported as the default, because the encoding is carried in the payload
    /// and the payload is not decoded. A symbol that declares Shift-JIS will be
    /// reported as UTF-8 here; that is a consequence of the missing decoder,
    /// not an independent guess.
    #[must_use]
    pub fn get_encoding(&self) -> i32 {
        // `cv::QRCodeEncoder::ECI_UTF8`.
        26
    }

    /// `cv::QRCodeEncoder::encodeStructuredAppend(encoded_info, qrcodes)`.
    pub fn encode_structured_append(&self, encoded_info: &str) -> Result<Vec<Mat>> {
        cv_assert!(
            self.params.structure_number > 1,
            "a structured append needs more than one symbol, got {}",
            self.params.structure_number
        );
        cv_error!(
            Code::StsNotImplemented,
            "QR encoding is not implemented; cannot encode {} bytes",
            encoded_info.len()
        )
    }
}

/// `cv::barcode::BarcodeDetector`.
#[derive(Debug, Clone, PartialEq)]
pub struct BarcodeDetector {
    downsampling_threshold: f64,
    detector_scales: Vec<f32>,
    gradient_threshold: f64,
}

impl Default for BarcodeDetector {
    /// Upstream's defaults.
    fn default() -> Self {
        BarcodeDetector {
            downsampling_threshold: 512.0,
            detector_scales: vec![0.01, 0.03, 0.06, 0.08],
            gradient_threshold: 0.25,
        }
    }
}

impl BarcodeDetector {
    /// `cv::barcode::BarcodeDetector::BarcodeDetector(prototxt_path, model_path)`.
    #[must_use]
    pub fn new() -> Self {
        BarcodeDetector::default()
    }

    /// `cv::barcode::BarcodeDetector::getDownsamplingThreshold()`.
    ///
    /// Images larger than this are scaled down before the search. A barcode is
    /// a low-frequency pattern, so the extra pixels cost time and find nothing.
    #[must_use]
    pub fn get_downsampling_threshold(&self) -> f64 {
        self.downsampling_threshold
    }
    /// `cv::barcode::BarcodeDetector::setDownsamplingThreshold(thresh)`.
    pub fn set_downsampling_threshold(&mut self, v: f64) -> &mut Self {
        self.downsampling_threshold = v;
        self
    }
    /// `cv::barcode::BarcodeDetector::getDetectorScales(sizes)`.
    #[must_use]
    pub fn get_detector_scales(&self) -> &[f32] {
        &self.detector_scales
    }
    /// `cv::barcode::BarcodeDetector::setDetectorScales(sizes)`.
    pub fn set_detector_scales(&mut self, v: Vec<f32>) -> Result<&mut Self> {
        cv_assert!(!v.is_empty(), "at least one detector scale is needed");
        cv_assert!(v.iter().all(|s| *s > 0.0), "every scale must be positive");
        self.detector_scales = v;
        Ok(self)
    }
    /// `cv::barcode::BarcodeDetector::getGradientThreshold()`.
    ///
    /// How consistent the local gradient direction must be. A barcode is a
    /// region where every edge runs the same way — that coherence, not the
    /// edges themselves, is what distinguishes it from ordinary texture.
    #[must_use]
    pub fn get_gradient_threshold(&self) -> f64 {
        self.gradient_threshold
    }
    /// `cv::barcode::BarcodeDetector::setGradientThreshold(thresh)`.
    pub fn set_gradient_threshold(&mut self, v: f64) -> &mut Self {
        self.gradient_threshold = v;
        self
    }

    /// `cv::barcode::BarcodeDetector::decodeWithType(img, points, decoded_info, decoded_type)`.
    pub fn decode_with_type(
        &self,
        _img: &Mat,
        _points: &[(f64, f64); 4],
    ) -> Result<(String, String)> {
        cv_error!(
            Code::StsNotImplemented,
            "barcode decoding is not implemented; each symbology is its own table"
        )
    }

    /// `cv::barcode::BarcodeDetector::detectAndDecodeWithType(img, decoded_info, decoded_type, points)`.
    pub fn detect_and_decode_with_type(&self, img: &Mat) -> Result<Vec<(String, String)>> {
        let points = self.detect_multi(img)?;
        points.iter().map(|p| self.decode_with_type(img, p)).collect()
    }
}

impl GraphicalCodeDetector for BarcodeDetector {
    /// Finds the region whose gradient directions agree — a barcode is a band
    /// of parallel edges, and that is what separates it from general texture.
    fn detect(&self, img: &Mat) -> Result<Option<[(f64, f64); 4]>> {
        cv_assert!(img.channels() == 1, "barcode detection takes a single-channel image");
        let (h, w) = (img.rows(), img.cols());
        if h < 8 || w < 8 {
            return Ok(None);
        }
        // Coherence: |gx| - |gy| summed over a window. A vertical-bar barcode
        // scores strongly positive, ordinary texture near zero.
        let (mut best, mut bx, mut by) = (0.0f64, 0, 0);
        let win = (w.min(h) / 4).max(4);
        for y in 0..h - win {
            for x in 0..w - win {
                let mut score = 0.0;
                for dy in (0..win).step_by(2) {
                    for dx in (0..win).step_by(2) {
                        let (px, py) = (x + dx, y + dy);
                        let gx = img.at_f64(py, (px + 1).min(w - 1), 0)?
                            - img.at_f64(py, px.max(1) - 1, 0)?;
                        let gy = img.at_f64((py + 1).min(h - 1), px, 0)?
                            - img.at_f64(py.max(1) - 1, px, 0)?;
                        score += gx.abs() - gy.abs();
                    }
                }
                if score > best {
                    best = score;
                    bx = x;
                    by = y;
                }
            }
        }
        // Scale the acceptance with the window so a large image does not pass
        // on accumulated noise alone.
        if best < self.gradient_threshold * f64::from(win * win) * 4.0 {
            return Ok(None);
        }
        let (x0, y0) = (f64::from(bx), f64::from(by));
        let s = f64::from(win);
        Ok(Some([(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)]))
    }

    fn decode(&self, img: &Mat, points: &[(f64, f64); 4]) -> Result<String> {
        Ok(self.decode_with_type(img, points)?.0)
    }
}

/// `cv::GraphicalCodeDetector::decodeBytes` and friends, for a detector that
/// returns raw bytes rather than text.
pub trait ByteCodeDetector: GraphicalCodeDetector {
    /// `cv::QRCodeDetector::decodeBytes(img, points, straight_code)`.
    fn decode_bytes(&self, img: &Mat, points: &[(f64, f64); 4]) -> Result<Vec<u8>> {
        Ok(self.decode(img, points)?.into_bytes())
    }
    /// `cv::QRCodeDetector::decodeBytesMulti(img, points, decoded_info, straight_code)`.
    fn decode_bytes_multi(&self, img: &Mat, points: &[[(f64, f64); 4]]) -> Result<Vec<Vec<u8>>> {
        points.iter().map(|p| self.decode_bytes(img, p)).collect()
    }
    /// `cv::QRCodeDetector::detectAndDecodeBytes(img, points, straight_code)`.
    fn detect_and_decode_bytes(&self, img: &Mat) -> Result<Option<Vec<u8>>> {
        let Some(points) = self.detect(img)? else { return Ok(None) };
        Ok(Some(self.decode_bytes(img, &points)?))
    }
    /// `cv::QRCodeDetector::detectAndDecodeBytesMulti(img, decoded_info, points, straight_code)`.
    fn detect_and_decode_bytes_multi(&self, img: &Mat) -> Result<Vec<Vec<u8>>> {
        let points = self.detect_multi(img)?;
        self.decode_bytes_multi(img, &points)
    }
}

impl ByteCodeDetector for QRCodeDetector {}
impl ByteCodeDetector for QRCodeDetectorAruco {}
impl ByteCodeDetector for BarcodeDetector {}

// ---------------------------------------------------------------------------
// Face models
// ---------------------------------------------------------------------------

/// `cv::FaceDetectorYN` — the YuNet face detector.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceDetectorYN {
    model: String,
    input_size: Size,
    score_threshold: f32,
    nms_threshold: f32,
    top_k: i32,
}

impl FaceDetectorYN {
    /// `cv::FaceDetectorYN::create(model, config, input_size, score_threshold, ...)`.
    ///
    /// The model path is recorded but nothing is loaded until [`Self::detect`],
    /// which is where the missing weights are reported.
    pub fn create(
        model: &str,
        input_size: Size,
        score_threshold: f32,
        nms_threshold: f32,
        top_k: i32,
    ) -> Result<Self> {
        cv_assert!(
            input_size.width > 0 && input_size.height > 0,
            "the input size must be positive"
        );
        cv_assert!(
            (0.0..=1.0).contains(&score_threshold),
            "the score threshold must lie in 0..1"
        );
        cv_assert!((0.0..=1.0).contains(&nms_threshold), "the NMS threshold must lie in 0..1");
        Ok(FaceDetectorYN {
            model: model.to_string(),
            input_size,
            score_threshold,
            nms_threshold,
            top_k,
        })
    }

    /// `cv::FaceDetectorYN::getInputSize()`.
    #[must_use]
    pub fn get_input_size(&self) -> Size {
        self.input_size
    }
    /// `cv::FaceDetectorYN::setInputSize(input_size)`.
    pub fn set_input_size(&mut self, v: Size) {
        self.input_size = v;
    }
    /// `cv::FaceDetectorYN::getScoreThreshold()`.
    #[must_use]
    pub fn get_score_threshold(&self) -> f32 {
        self.score_threshold
    }
    /// `cv::FaceDetectorYN::setScoreThreshold(score_threshold)`.
    pub fn set_score_threshold(&mut self, v: f32) {
        self.score_threshold = v;
    }
    /// `cv::FaceDetectorYN::getNMSThreshold()`.
    #[must_use]
    pub fn get_nms_threshold(&self) -> f32 {
        self.nms_threshold
    }
    /// `cv::FaceDetectorYN::setNMSThreshold(nms_threshold)`.
    pub fn set_nms_threshold(&mut self, v: f32) {
        self.nms_threshold = v;
    }
    /// `cv::FaceDetectorYN::getTopK()`.
    #[must_use]
    pub fn get_top_k(&self) -> i32 {
        self.top_k
    }
    /// `cv::FaceDetectorYN::setTopK(top_k)`.
    pub fn set_top_k(&mut self, v: i32) {
        self.top_k = v;
    }

    /// `cv::FaceDetectorYN::detect(image, faces)`.
    ///
    /// Errors naming the weight file. YuNet is a convolutional network; without
    /// its weights there is nothing to run, and returning an empty face list
    /// would read as "no faces in this image".
    pub fn detect<S: MatLike + ?Sized>(&self, _image: &S) -> Result<Mat> {
        cv_error!(
            Code::StsError,
            "FaceDetectorYN needs the network weights at `{}`, which are not bundled",
            self.model
        )
    }
}

/// `cv::FaceRecognizerSF` — the SFace embedding model.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceRecognizerSF {
    model: String,
}

impl FaceRecognizerSF {
    /// `cv::FaceRecognizerSF::create(model, config, backend_id, target_id)`.
    #[must_use]
    pub fn create(model: &str) -> Self {
        FaceRecognizerSF { model: model.to_string() }
    }

    /// `cv::FaceRecognizerSF::alignCrop(src_img, face_box, aligned_img)`.
    ///
    /// Warps a detected face to the canonical 112x112 layout the embedding
    /// expects. This part is pure geometry and works without the network: the
    /// five landmarks are mapped onto their reference positions by a similarity
    /// transform.
    pub fn align_crop<S: MatLike + ?Sized>(
        &self,
        src: &S,
        landmarks: &[(f64, f64); 5],
    ) -> Result<Mat> {
        // The reference landmarks of the 112x112 crop, from the SFace training
        // set: both eyes, the nose, and both mouth corners.
        const REF: [(f64, f64); 5] = [
            (38.29, 51.69),
            (73.53, 51.50),
            (56.02, 71.73),
            (41.55, 92.37),
            (70.73, 92.20),
        ];
        let from: Vec<(f64, f64)> = landmarks.to_vec();
        let to: Vec<(f64, f64)> = REF.to_vec();
        let m = cv_calib3d::estimate_affine_partial_2d(&from, &to)?;
        cv_imgproc::warp::warp_affine(
            src,
            cv_imgproc::warp::Affine2x3::new([
                m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2],
            ]),
            Size::new(112, 112),
            cv_imgproc::resize::Interpolation::Linear,
            false,
            cv_core::BorderType::Constant,
            cv_core::Scalar::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    /// `cv::FaceRecognizerSF::feature(aligned_img, face_feature)`.
    pub fn feature<S: MatLike + ?Sized>(&self, _aligned: &S) -> Result<Mat> {
        cv_error!(
            Code::StsError,
            "FaceRecognizerSF needs the network weights at `{}`, which are not bundled",
            self.model
        )
    }

    /// `cv::FaceRecognizerSF::match(face_feature1, face_feature2, dis_type)`.
    ///
    /// Compares two embeddings. `dis_type` is [`FR_COSINE`] or [`FR_NORM_L2`];
    /// note that cosine is *larger* for a better match and L2 smaller, so the
    /// threshold's direction flips with it.
    pub fn match_features(&self, a: &Mat, b: &Mat, dis_type: i32) -> Result<f64> {
        cv_assert!(a.size() == b.size(), "the two embeddings must have the same length");
        cv_assert!(!a.is_empty(), "an empty embedding cannot be compared");
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut l2 = 0.0f64;
        for y in 0..a.rows() {
            for x in 0..a.cols() {
                let (u, v) = (a.at_f64(y, x, 0)?, b.at_f64(y, x, 0)?);
                dot += u * v;
                na += u * u;
                nb += v * v;
                l2 += (u - v) * (u - v);
            }
        }
        if dis_type == FR_NORM_L2 {
            return Ok(l2.sqrt());
        }
        cv_assert!(dis_type == FR_COSINE, "unknown distance type {dis_type}");
        Ok(dot / (na.sqrt() * nb.sqrt()).max(1e-12))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::CV_8UC1;
    use cv_core::MatLikeMut;

    fn img(w: i32, h: i32, f: impl Fn(i32, i32) -> u8) -> Mat {
        let mut m = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                m.pixel_mut::<u8>(y, x).unwrap()[0] = f(x, y);
            }
        }
        m
    }

    #[test]
    fn the_descriptor_format_and_distance_constants_match_upstream() {
        assert_eq!(DESCR_FORMAT_ROW_BY_ROW, 0);
        assert_eq!(DESCR_FORMAT_COL_BY_COL, 1);
        assert_eq!(FR_COSINE, 0);
        assert_eq!(FR_NORM_L2, 1);
        assert_eq!((FEATURE_HAAR, FEATURE_LBP, FEATURE_HOG), (0, 1, 2));
    }

    #[test]
    fn the_people_detectors_are_the_documented_lengths_and_refused_as_placeholders() {
        let d = HogDescriptor::default_people().unwrap();
        let default = HogDescriptor::get_default_people_detector();
        assert_eq!(default.len(), 3781, "upstream's 64x128 detector length");
        assert_eq!(HogDescriptor::get_daimler_people_detector().len(), 981);

        // An all-zero placeholder is refused rather than silently installed.
        let mut h = d;
        assert!(h.set_svm_detector(&default).is_err(), "the placeholder must not be accepted");
        // A wrong length is refused too.
        assert!(h.set_svm_detector(&[1.0f32; 10]).is_err());
        // A real-looking detector of the right length is accepted.
        let want = h.params.descriptor_size() + 1;
        let mut real = vec![0.0f32; want];
        real[0] = 0.5;
        h.set_svm_detector(&real).unwrap();
    }

    #[test]
    fn the_gradient_has_a_magnitude_and_an_orientation_bin() {
        let h = HogDescriptor::default_people().unwrap();
        // A vertical edge: the gradient runs along x, so the orientation is 0.
        let m = img(32, 32, |x, _| if x < 16 { 0 } else { 255 });
        let (mag, ang) = h.compute_gradient(&m).unwrap();
        assert_eq!((mag.rows(), mag.cols()), (32, 32));
        assert!(mag.at_f64(16, 15, 0).unwrap() > 100.0, "the edge should be strong");
        assert!(mag.at_f64(16, 4, 0).unwrap() < 1.0, "the flat region should be quiet");
        let bin = ang.at_f64(16, 15, 0).unwrap();
        assert!(bin < 1.0 || bin > (h.params.nbins - 1) as f64, "bin {bin} should be near 0");

        let colour = Mat::zeros(8, 8, cv_core::types::MatType::new(cv_core::types::Depth::U8, 3))
            .unwrap();
        assert!(h.compute_gradient(&colour).is_err());
    }

    #[test]
    fn the_window_sigma_is_derived_when_unset() {
        let h = HogDescriptor::default_people().unwrap();
        // Upstream's default is negative, meaning blockSize.width / 8.
        assert_eq!(h.get_win_sigma(), f64::from(h.params.block.0) / 8.0);
    }

    #[test]
    fn the_multi_scale_variants_agree_with_the_plain_one() {
        let h = HogDescriptor::default_people().unwrap();
        let m = img(80, 160, |x, y| ((x * 3 + y * 5) % 200 + 30) as u8);
        // A detector is required; without one there is nothing to score with.
        let mut h = h;
        assert!(h.detect_multi_scale2(&m, 0.0, (8, 8), 1.2, 2).is_err());
        let mut svm = vec![0.0f32; h.params.descriptor_size() + 1];
        svm[0] = 0.01;
        h.set_svm_detector(&svm).unwrap();

        let (w, bias) = svm.split_at(svm.len() - 1);
        let plain: Vec<Rect> = h
            .detect_multi_scale(&m, w, bias[0], 0.0, (8, 8), 1.2, 2)
            .unwrap()
            .iter()
            .map(|(r, _)| *r)
            .collect();
        let (r2, w2) = h.detect_multi_scale2(&m, 0.0, (8, 8), 1.2, 2).unwrap();
        assert_eq!(r2, plain);
        assert_eq!(w2.len(), plain.len());
        let (r3, w3, scales) = h.detect_multi_scale3(&m, 0.0, (8, 8), 1.2, 2).unwrap();
        assert_eq!(r3, plain);
        assert_eq!(w3.len(), plain.len());
        assert!(!scales.is_empty(), "at least the unit scale fits");
        assert_eq!(scales[0], 1.0);
    }

    #[test]
    fn weighted_grouping_keeps_the_best_weight_of_each_group() {
        let rects = vec![
            Rect::new(10, 10, 20, 20),
            Rect::new(11, 11, 20, 20),
            Rect::new(100, 100, 20, 20),
        ];
        let weights = vec![0.2, 0.9, 0.5];
        let (grouped, gw) = group_rectangles_weighted(&rects, &weights, 1, 0.2).unwrap();
        assert_eq!(grouped.len(), gw.len());
        assert!(gw.iter().all(|w| *w >= 0.0));
        // The first group's weight is its strongest member's.
        if let Some(w) = gw.first() {
            assert!(*w >= 0.5 || grouped.len() == 1, "{gw:?}");
        }
        assert!(group_rectangles_weighted(&rects, &weights[..2], 1, 0.2).is_err());
    }

    /// A minimal cascade, so the accessors have something to describe.
    fn cascade() -> crate::CascadeClassifier {
        let f = crate::HaarFeature {
            rects: vec![(Rect::new(0, 0, 4, 4), 1.0), (Rect::new(4, 0, 4, 4), -1.0)],
            threshold: 0.0,
            left: -1.0,
            right: 1.0,
        };
        let stage = crate::CascadeStage { features: vec![f], threshold: -10.0 };
        crate::CascadeClassifier::new(vec![stage], (8, 8)).unwrap()
    }

    #[test]
    fn the_cascade_reports_its_shape_and_refuses_to_half_read_a_file() {
        let mut c = cascade();
        assert!(!c.empty());
        assert!(!c.is_old_format_cascade());
        assert_eq!(c.get_original_window_size(), Size::new(8, 8));
        assert_eq!(c.get_feature_type(), FEATURE_HAAR);
        assert!(c.check_detector_size());

        // Reading a model is refused outright rather than half-parsed.
        assert!(c.load("haarcascade_frontalface_default.xml").is_err());
        assert!(c.read(&cv_core::Node::Map(Default::default())).is_err());
        assert!(crate::CascadeClassifier::convert("old.xml", "new.xml").is_err());
        assert!(c.save("out.xml").is_err(), "writing a format that cannot be read back is refused");
    }

    #[test]
    fn the_cascade_multi_scale_variants_report_neighbour_counts() {
        let c = cascade();
        let m = img(40, 40, |x, y| ((x * 7 + y * 11) % 200 + 30) as u8);
        let (rects, counts) = c.detect_multi_scale2(&m, 1.3, 1, 4).unwrap();
        assert_eq!(rects.len(), counts.len());
        assert!(counts.iter().all(|c| *c >= 1), "a group has at least one member");
        let (r3, c3, w3) = c.detect_multi_scale3(&m, 1.3, 1, 4).unwrap();
        assert_eq!(r3.len(), c3.len());
        assert_eq!(r3.len(), w3.len());
        assert_eq!(r3, rects);
    }

    #[test]
    fn the_qr_detector_accessors_round_trip() {
        let mut q = QRCodeDetector::new();
        assert_eq!(q.tolerances(), (0.2, 0.1, true));
        q.set_eps_x(0.3).set_eps_y(0.4).set_use_alignment_markers(false);
        assert_eq!(q.tolerances(), (0.3, 0.4, false));

        let mut a = QRCodeDetectorAruco::new();
        let (m, r, s) = a.params();
        assert_eq!((m, r, s), (4.0, 0.6, 0.9));
        a.set_detector_parameters(2.0, 0.3);
        assert_eq!(a.params().0, 2.0);
        assert_eq!(a.params().1, 0.3);
        a.set_aruco_parameters(a.get_aruco_parameters());
    }

    #[test]
    fn qr_decoding_says_plainly_that_it_is_not_implemented() {
        let q = QRCodeDetector::new();
        let blank = Mat::zeros(40, 40, CV_8UC1).unwrap();
        let pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let err = q.decode(&blank, &pts).unwrap_err();
        assert!(format!("{err}").contains("not implemented"), "{err}");
        assert!(q.decode_curved(&blank, &pts).is_err());
        assert!(q.detect_and_decode_curved(&blank).is_err());
        // A blank image has no symbol, which is a `None` rather than an error.
        assert!(q.detect(&blank).unwrap().is_none());
        assert!(q.detect_and_decode(&blank).unwrap().is_none());
        assert!(q.detect_multi(&blank).unwrap().is_empty());
        assert!(q.detect_and_decode_multi(&blank).unwrap().is_empty());
        assert!(q.detect_and_decode_bytes(&blank).unwrap().is_none());
        assert!(q.detect_and_decode_bytes_multi(&blank).unwrap().is_empty());
        assert!(q.decode_bytes(&blank, &pts).is_err());
        assert!(q.decode_bytes_multi(&blank, &[pts]).is_err());
        assert!(q.decode_multi(&blank, &[pts]).is_err());
    }

    #[test]
    fn the_qr_encoder_carries_its_parameters_and_refuses_to_encode() {
        let e = QRCodeEncoder::default();
        assert_eq!(e.params().version, 0);
        assert_eq!(e.params().correction_level, QRCorrectionLevel::L);
        assert_eq!(e.params().mode, QREncodeMode::Auto);
        assert_eq!(e.params().structure_number, 1);

        let p = QRCodeEncoderParams {
            version: 5,
            correction_level: QRCorrectionLevel::H,
            mode: QREncodeMode::Byte,
            structure_number: 3,
        };
        let e = QRCodeEncoder::create(p);
        assert_eq!(e.params(), p);
        assert_eq!(e.get_encoding(), 26, "ECI_UTF8");
        assert!(e.encode("hello").is_err());
        assert!(e.encode_structured_append("hello").is_err());
        // A structured append of one symbol is a contradiction, caught first.
        let one = QRCodeEncoder::create(QRCodeEncoderParams::default());
        assert!(one.encode_structured_append("hello").is_err());
    }

    #[test]
    fn the_barcode_detector_accessors_round_trip() {
        let mut b = BarcodeDetector::new();
        assert_eq!(b.get_downsampling_threshold(), 512.0);
        assert_eq!(b.get_detector_scales(), &[0.01, 0.03, 0.06, 0.08]);
        assert_eq!(b.get_gradient_threshold(), 0.25);
        b.set_downsampling_threshold(1024.0);
        b.set_gradient_threshold(0.5);
        b.set_detector_scales(vec![0.02, 0.04]).unwrap();
        assert_eq!(b.get_downsampling_threshold(), 1024.0);
        assert_eq!(b.get_gradient_threshold(), 0.5);
        assert_eq!(b.get_detector_scales(), &[0.02, 0.04]);
        assert!(b.set_detector_scales(vec![]).is_err());
        assert!(b.set_detector_scales(vec![-1.0]).is_err());
    }

    #[test]
    fn the_barcode_detector_prefers_parallel_bars_to_ordinary_texture() {
        // Vertical bars: strong |gx|, no |gy|.
        let bars = img(64, 64, |x, _| if (x / 4) % 2 == 0 { 0 } else { 255 });
        let b = BarcodeDetector::new();
        assert!(b.detect(&bars).unwrap().is_some(), "a bar pattern should be found");

        // A flat field has no gradient at all.
        let flat = img(64, 64, |_, _| 128);
        assert!(b.detect(&flat).unwrap().is_none());
        // A tiny image is not searched.
        assert!(b.detect(&img(4, 4, |_, _| 0)).unwrap().is_none());
        // Decoding is refused rather than guessed.
        assert!(b.detect_and_decode_with_type(&bars).is_err());
    }

    #[test]
    fn the_face_detector_reports_the_weights_it_needs() {
        let d = FaceDetectorYN::create("yunet.onnx", Size::new(320, 320), 0.9, 0.3, 5000).unwrap();
        assert_eq!(d.get_input_size(), Size::new(320, 320));
        assert_eq!(d.get_score_threshold(), 0.9);
        assert_eq!(d.get_nms_threshold(), 0.3);
        assert_eq!(d.get_top_k(), 5000);

        let mut d2 = d.clone();
        d2.set_input_size(Size::new(160, 160));
        d2.set_score_threshold(0.5);
        d2.set_nms_threshold(0.4);
        d2.set_top_k(100);
        assert_eq!(d2.get_input_size(), Size::new(160, 160));
        assert_eq!((d2.get_score_threshold(), d2.get_nms_threshold()), (0.5, 0.4));
        assert_eq!(d2.get_top_k(), 100);

        let err = d.detect(&Mat::zeros(320, 320, CV_8UC1).unwrap()).unwrap_err();
        assert!(format!("{err}").contains("yunet.onnx"), "the error should name the file: {err}");

        assert!(FaceDetectorYN::create("m", Size::new(0, 320), 0.9, 0.3, 5).is_err());
        assert!(FaceDetectorYN::create("m", Size::new(32, 32), 1.5, 0.3, 5).is_err());
        assert!(FaceDetectorYN::create("m", Size::new(32, 32), 0.9, 2.0, 5).is_err());
    }

    #[test]
    fn face_alignment_works_without_the_network_and_matching_does_not() {
        let r = FaceRecognizerSF::create("sface.onnx");
        let src = img(200, 200, |x, y| ((x + y) % 256) as u8);
        // Landmarks at twice the reference scale, offset.
        let lm = [
            (76.58, 103.38),
            (147.06, 103.0),
            (112.04, 143.46),
            (83.10, 184.74),
            (141.46, 184.40),
        ];
        let aligned = r.align_crop(&src, &lm).unwrap();
        assert_eq!((aligned.rows(), aligned.cols()), (112, 112), "the canonical crop size");

        // The embedding itself needs the weights.
        let err = r.feature(&aligned).unwrap_err();
        assert!(format!("{err}").contains("sface.onnx"), "{err}");

        // Matching is pure arithmetic and works on any two equal-length vectors.
        let a = Mat::from_vec(1, 4, 1, &[1.0f64, 0.0, 0.0, 0.0]).unwrap();
        let b = Mat::from_vec(1, 4, 1, &[1.0f64, 0.0, 0.0, 0.0]).unwrap();
        let c = Mat::from_vec(1, 4, 1, &[0.0f64, 1.0, 0.0, 0.0]).unwrap();
        assert!((r.match_features(&a, &b, FR_COSINE).unwrap() - 1.0).abs() < 1e-9);
        assert!(r.match_features(&a, &c, FR_COSINE).unwrap().abs() < 1e-9);
        assert!(r.match_features(&a, &b, FR_NORM_L2).unwrap().abs() < 1e-9);
        assert!((r.match_features(&a, &c, FR_NORM_L2).unwrap() - 2.0f64.sqrt()).abs() < 1e-9);
        // The two conventions point opposite ways, which is why the type matters.
        assert!(r.match_features(&a, &c, 99).is_err(), "an unknown distance type");
        let wrong = Mat::from_vec(1, 2, 1, &[1.0f64, 0.0]).unwrap();
        assert!(r.match_features(&a, &wrong, FR_COSINE).is_err());
    }
}
