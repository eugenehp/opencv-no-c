//! The `Feature2D` and `DescriptorMatcher` class hierarchies, and bag-of-words
//! (`modules/features2d/include/opencv2/features2d.hpp`).
//!
//! Upstream reaches every detector through one abstract base: `Feature2D` has
//! `detect`, `compute`, `detectAndCompute`, and a few questions about the
//! descriptors it produces. That indirection is what lets a pipeline swap SIFT
//! for ORB without changing anything else, and it is what [`Feature2D`]
//! reproduces here.
//!
//! The algorithms themselves live in the sibling modules — [`crate::sift`],
//! [`crate::brisk`], [`crate::kaze`], [`crate::akaze`], [`crate::regions`] and
//! the free functions in the crate root. These types hold parameters and
//! dispatch; nothing is reimplemented, so a fix in `sift.rs` reaches [`Sift`]
//! too.
//!
//! # Descriptor width, type and norm
//!
//! [`Feature2D::descriptor_size`] and [`Feature2D::descriptor_type`] tell a
//! matcher what it is comparing, and [`Feature2D::default_norm`] tells it which
//! metric to compare with. That last one is the classic mistake: a binary
//! descriptor (ORB, BRISK, AKAZE) needs `NORM_HAMMING`, and comparing it under
//! L2 still runs and returns plausible-looking nonsense.
//!
//! Descriptors come back as a [`Mat`] of one row per keypoint, as upstream
//! does, so the width and element type travel with the data rather than being
//! remembered separately by the caller.

use cv_core::{cv_assert, cv_error, Code, DMatch, KeyPoint, Mat, MatLike, Node, Result};

use crate::detectors::{
    AgastParams, AgastType, AkazeDescriptor, AkazeParams, BriskParams, Diffusivity, FastParams,
    FastType, GfttParams, KazeParams, MserParams, OrbParams, OrbScore, SiftConfig,
};
use crate::matcher::NormType;

/// `cv::SimpleBlobDetector::Params` — the blob filter thresholds.
pub use crate::BlobParams as Params;

/// The rows of a descriptor matrix as bytes.
fn rows_u8(m: &Mat) -> Result<Vec<Vec<u8>>> {
    let mut out = Vec::with_capacity(m.rows().max(0) as usize);
    for y in 0..m.rows() {
        let mut row = Vec::with_capacity(m.cols().max(0) as usize);
        for x in 0..m.cols() {
            row.push(m.at_f64(y, x, 0)? as u8);
        }
        out.push(row);
    }
    Ok(out)
}

/// The rows of a descriptor matrix as floats.
fn rows_f32(m: &Mat) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(m.rows().max(0) as usize);
    for y in 0..m.rows() {
        let mut row = Vec::with_capacity(m.cols().max(0) as usize);
        for x in 0..m.cols() {
            row.push(m.at_f64(y, x, 0)? as f32);
        }
        out.push(row);
    }
    Ok(out)
}

/// Pack per-keypoint byte descriptors into a `rows x width` `CV_8U` matrix.
fn pack_u8(rows: &[Vec<u8>], width: usize) -> Result<Mat> {
    let flat: Vec<u8> = rows.iter().flatten().copied().collect();
    Mat::from_vec(rows.len() as i32, width as i32, 1, &flat)
}

/// Pack per-keypoint float descriptors into a `rows x width` `CV_32F` matrix.
fn pack_f32(rows: &[Vec<f32>], width: usize) -> Result<Mat> {
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    Mat::from_vec(rows.len() as i32, width as i32, 1, &flat)
}

/// `cv::Feature2D` — the interface every detector and descriptor implements.
///
/// The methods take `&Mat` rather than a generic `MatLike`, so the trait stays
/// object-safe: the whole point of the base class is to hold a detector behind
/// a `Box<dyn Feature2D>` and not care which one it is.
pub trait Feature2D {
    /// `cv::Feature2D::detect(image, keypoints, mask)`.
    fn detect(&self, image: &Mat) -> Result<Vec<KeyPoint>>;

    /// `cv::Feature2D::detectAndCompute(image, mask, keypoints, descriptors)`.
    ///
    /// A detector that produces no descriptors returns an empty matrix, which
    /// is what upstream does rather than failing.
    fn detect_and_compute(&self, image: &Mat) -> Result<(Vec<KeyPoint>, Mat)> {
        Ok((self.detect(image)?, Mat::empty(cv_core::types::CV_8UC1)))
    }

    /// `cv::Feature2D::compute(image, keypoints, descriptors)`.
    ///
    /// Describes keypoints someone else found. The default re-detects, since
    /// most detectors here fuse detection and description; the ones that can
    /// describe a supplied point cheaply override it.
    fn compute(&self, image: &Mat, _keypoints: &[KeyPoint]) -> Result<Mat> {
        Ok(self.detect_and_compute(image)?.1)
    }

    /// `cv::Feature2D::descriptorSize()` — elements per descriptor, 0 if none.
    fn descriptor_size(&self) -> i32 {
        0
    }

    /// `cv::Feature2D::descriptorType()` — the `CV_*` depth of one element.
    fn descriptor_type(&self) -> i32 {
        cv_core::types::CV_8U
    }

    /// `cv::Feature2D::defaultNorm()` — the metric its descriptors compare under.
    fn default_norm(&self) -> NormType {
        NormType::Hamming
    }

    /// `cv::Feature2D::empty()` — is this unusable as configured?
    fn empty(&self) -> bool {
        false
    }

    /// `cv::Feature2D::getDefaultName()`.
    fn get_default_name(&self) -> String;

    /// `cv::Feature2D::write(fs)` — store the parameters under `name`.
    fn write(&self, fs: &mut cv_core::FileStorage, name: &str) -> Result<()> {
        let _ = (fs, name);
        Ok(())
    }

    /// `cv::Feature2D::read(fn)` — restore them.
    fn read(&mut self, node: &Node) -> Result<()> {
        let _ = node;
        Ok(())
    }

    /// `cv::Feature2D::clear()` — reset to a freshly created state.
    fn clear(&mut self) {}
}

/// `cv::SIFT`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sift {
    /// The tunables; see [`SiftConfig`] for the upstream accessor names.
    pub params: SiftConfig,
}

impl Sift {
    /// `cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)`.
    #[must_use]
    pub fn create(params: SiftConfig) -> Self {
        Sift { params }
    }
}

impl Feature2D for Sift {
    fn detect(&self, image: &Mat) -> Result<Vec<KeyPoint>> {
        Ok(self.detect_and_compute(image)?.0)
    }

    fn detect_and_compute(&self, image: &Mat) -> Result<(Vec<KeyPoint>, Mat)> {
        let p = crate::sift::SiftParams {
            n_features: self.params.n_features.max(0) as usize,
            n_octave_layers: self.params.n_octave_layers.max(1) as usize,
            contrast_threshold: self.params.contrast_threshold,
            edge_threshold: self.params.edge_threshold,
            sigma: self.params.sigma,
            ..crate::sift::SiftParams::default()
        };
        let (kps, desc) = crate::sift::sift_detect_and_compute(image, p)?;
        let rows: Vec<Vec<f32>> = desc.iter().map(|d| d.to_vec()).collect();
        let m = pack_f32(&rows, 128)?;
        Ok((kps, m))
    }

    fn descriptor_size(&self) -> i32 {
        128
    }

    fn descriptor_type(&self) -> i32 {
        cv_core::types::CV_32F
    }

    /// SIFT is a float descriptor: L2, not Hamming.
    fn default_norm(&self) -> NormType {
        NormType::L2
    }

    fn get_default_name(&self) -> String {
        "Feature2D.SIFT".into()
    }
}

/// `cv::ORB`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Orb {
    /// The tunables; see [`OrbParams`].
    pub params: OrbParams,
}

impl Orb {
    /// `cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, ...)`.
    #[must_use]
    pub fn create(params: OrbParams) -> Self {
        Orb { params }
    }

    /// `cv::ORB::getScoreType()` — which corner measure ranks the keypoints.
    #[must_use]
    pub fn get_score_type(&self) -> OrbScore {
        OrbScore::Harris
    }

    /// `cv::ORB::setScoreType(scoreType)`.
    ///
    /// Accepted and ignored: the detector always ranks by the response that
    /// non-max suppression produced. Storing a setting nothing consults would
    /// be worse than saying so here.
    pub fn set_score_type(&mut self, _score: OrbScore) {}
}

impl Feature2D for Orb {
    fn detect(&self, image: &Mat) -> Result<Vec<KeyPoint>> {
        Ok(self.detect_and_compute(image)?.0)
    }

    fn detect_and_compute(&self, image: &Mat) -> Result<(Vec<KeyPoint>, Mat)> {
        let (kps, desc) = crate::orb_detect_and_compute(
            image,
            self.params.fast_threshold,
            self.params.n_features.max(0) as usize,
        )?;
        let rows: Vec<Vec<u8>> = desc.iter().map(|d| d.to_vec()).collect();
        let m = pack_u8(&rows, 32)?;
        Ok((kps, m))
    }

    /// `cv::ORB::compute` — oriented BRIEF on keypoints already found.
    fn compute(&self, image: &Mat, keypoints: &[KeyPoint]) -> Result<Mat> {
        let (_, desc) = crate::compute_orb(image, keypoints)?;
        let rows: Vec<Vec<u8>> = desc.iter().map(|d| d.to_vec()).collect();
        pack_u8(&rows, 32)
    }

    fn descriptor_size(&self) -> i32 {
        32
    }

    fn get_default_name(&self) -> String {
        "Feature2D.ORB".into()
    }
}

/// `cv::BRISK`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Brisk {
    /// The tunables; see [`BriskParams`].
    pub params: BriskParams,
}

impl Brisk {
    /// `cv::BRISK::create(thresh, octaves, patternScale)`.
    #[must_use]
    pub fn create(params: BriskParams) -> Self {
        Brisk { params }
    }
}

impl Feature2D for Brisk {
    fn detect(&self, image: &Mat) -> Result<Vec<KeyPoint>> {
        Ok(self.detect_and_compute(image)?.0)
    }

    fn detect_and_compute(&self, image: &Mat) -> Result<(Vec<KeyPoint>, Mat)> {
        let (kps, desc) = crate::brisk::brisk_detect_and_compute(image, self.params)?;
        let rows: Vec<Vec<u8>> = desc.iter().map(|d| d.to_vec()).collect();
        let m = pack_u8(&rows, crate::brisk::BRISK_BYTES)?;
        Ok((kps, m))
    }

    fn descriptor_size(&self) -> i32 {
        crate::brisk::BRISK_BYTES as i32
    }

    fn get_default_name(&self) -> String {
        "Feature2D.BRISK".into()
    }
}

/// `cv::KAZE`.
#[derive(Debug, Clone, Copy)]
pub struct Kaze {
    /// The tunables; see [`KazeParams`].
    pub params: KazeParams,
    /// The conductivity function driving the nonlinear diffusion.
    pub diffusivity: Diffusivity,
}

impl Default for Kaze {
    /// Upstream's `KAZE::create()` defaults, `DIFF_PM_G2` included.
    fn default() -> Self {
        Kaze { params: KazeParams::default(), diffusivity: Diffusivity::PmG2 }
    }
}

impl Kaze {
    /// `cv::KAZE::create(extended, upright, threshold, nOctaves, nOctaveLayers, diffusivity)`.
    #[must_use]
    pub fn create(params: KazeParams, diffusivity: Diffusivity) -> Self {
        Kaze { params, diffusivity }
    }

    /// `cv::KAZE::getDiffusivity()`.
    #[must_use]
    pub fn get_diffusivity(&self) -> Diffusivity {
        self.diffusivity
    }

    /// `cv::KAZE::setDiffusivity(diff)`.
    pub fn set_diffusivity(&mut self, diff: Diffusivity) {
        self.diffusivity = diff;
    }
}

impl Feature2D for Kaze {
    fn detect(&self, image: &Mat) -> Result<Vec<KeyPoint>> {
        Ok(self.detect_and_compute(image)?.0)
    }

    fn detect_and_compute(&self, image: &Mat) -> Result<(Vec<KeyPoint>, Mat)> {
        let (kps, desc) =
            crate::kaze::kaze_detect_and_compute(image, self.params, self.diffusivity)?;
        let width = self.descriptor_size() as usize;
        let m = pack_f32(&desc, width)?;
        Ok((kps, m))
    }

    fn descriptor_size(&self) -> i32 {
        if self.params.extended {
            128
        } else {
            64
        }
    }

    fn descriptor_type(&self) -> i32 {
        cv_core::types::CV_32F
    }

    /// M-SURF is a float descriptor.
    fn default_norm(&self) -> NormType {
        NormType::L2
    }

    fn get_default_name(&self) -> String {
        "Feature2D.KAZE".into()
    }
}

/// `cv::AKAZE`.
#[derive(Debug, Clone, Copy)]
pub struct Akaze {
    /// The tunables; see [`AkazeParams`].
    pub params: AkazeParams,
    /// Which descriptor to build.
    pub descriptor_type: AkazeDescriptor,
    /// The conductivity function.
    pub diffusivity: Diffusivity,
    /// `max_points` — keep at most this many, strongest first; negative is no limit.
    pub max_points: i32,
}

impl Default for Akaze {
    /// Upstream's `AKAZE::create()` defaults.
    fn default() -> Self {
        Akaze {
            params: AkazeParams::default(),
            descriptor_type: AkazeDescriptor::Mldb,
            diffusivity: Diffusivity::PmG2,
            max_points: -1,
        }
    }
}

impl Akaze {
    /// `cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, ...)`.
    #[must_use]
    pub fn create(
        params: AkazeParams,
        descriptor_type: AkazeDescriptor,
        diffusivity: Diffusivity,
        max_points: i32,
    ) -> Self {
        Akaze { params, descriptor_type, diffusivity, max_points }
    }

    /// `cv::AKAZE::getDescriptorType()`.
    #[must_use]
    pub fn get_descriptor_type(&self) -> AkazeDescriptor {
        self.descriptor_type
    }
    /// `cv::AKAZE::setDescriptorType(dtype)`.
    pub fn set_descriptor_type(&mut self, dtype: AkazeDescriptor) {
        self.descriptor_type = dtype;
    }
    /// `cv::AKAZE::getDiffusivity()`.
    #[must_use]
    pub fn get_diffusivity(&self) -> Diffusivity {
        self.diffusivity
    }
    /// `cv::AKAZE::setDiffusivity(diff)`.
    pub fn set_diffusivity(&mut self, diff: Diffusivity) {
        self.diffusivity = diff;
    }
    /// `cv::AKAZE::getMaxPoints()`.
    #[must_use]
    pub fn get_max_points(&self) -> i32 {
        self.max_points
    }
    /// `cv::AKAZE::setMaxPoints(max_points)`.
    pub fn set_max_points(&mut self, max_points: i32) {
        self.max_points = max_points;
    }

    /// The indices `max_points` keeps, strongest first then back into order.
    ///
    /// Returned as indices rather than applied in place because the keypoints
    /// and their descriptors have to be trimmed *together*: dropping one
    /// without the other silently misaligns every match downstream.
    fn keep(&self, kps: &[KeyPoint]) -> Option<Vec<usize>> {
        if self.max_points < 0 || kps.len() <= self.max_points as usize {
            return None;
        }
        let mut order: Vec<usize> = (0..kps.len()).collect();
        order.sort_by(|&a, &b| kps[b].response.total_cmp(&kps[a].response).then(a.cmp(&b)));
        order.truncate(self.max_points as usize);
        order.sort_unstable();
        Some(order)
    }
}

impl Feature2D for Akaze {
    fn detect(&self, image: &Mat) -> Result<Vec<KeyPoint>> {
        let kps = crate::akaze::akaze_detect(image, self.params, self.diffusivity)?;
        Ok(match self.keep(&kps) {
            Some(idx) => idx.iter().map(|&i| kps[i]).collect(),
            None => kps,
        })
    }

    fn detect_and_compute(&self, image: &Mat) -> Result<(Vec<KeyPoint>, Mat)> {
        let (mut kps, mut desc) =
            crate::akaze::akaze_detect_and_compute(image, self.params, self.diffusivity)?;
        if let Some(idx) = self.keep(&kps) {
            kps = idx.iter().map(|&i| kps[i]).collect();
            desc = idx.iter().map(|&i| desc[i]).collect();
        }
        let rows: Vec<Vec<u8>> = desc.iter().map(|d| d.to_vec()).collect();
        let width = rows.first().map_or(crate::kaze::AKAZE_BYTES, Vec::len);
        let m = pack_u8(&rows, width)?;
        Ok((kps, m))
    }

    fn descriptor_size(&self) -> i32 {
        if self.params.descriptor_size > 0 {
            self.params.descriptor_size / 8
        } else {
            crate::kaze::AKAZE_BYTES as i32
        }
    }

    fn get_default_name(&self) -> String {
        "Feature2D.AKAZE".into()
    }
}

/// `cv::MSER` — maximally stable extremal regions.
#[derive(Debug, Clone, Copy, Default)]
pub struct Mser {
    /// The tunables; see [`MserParams`].
    pub params: MserParams,
}

impl Mser {
    /// `cv::MSER::create(delta, min_area, max_area, max_variation, ...)`.
    #[must_use]
    pub fn create(params: MserParams) -> Self {
        Mser { params }
    }

    /// `cv::MSER::detectRegions(image, msers, bboxes)`.
    pub fn detect_regions(&self, image: &Mat) -> Result<Vec<crate::regions::MserRegion>> {
        crate::regions::mser_detect_regions(image, self.params)
    }

    /// `cv::MSER::getAreaThreshold()` — the growth ratio that ends a region.
    #[must_use]
    pub fn get_area_threshold(&self) -> f64 {
        1.01
    }
    /// `cv::MSER::setAreaThreshold(areaThreshold)`.
    pub fn set_area_threshold(&mut self, _area_threshold: f64) {}
    /// `cv::MSER::getMinDiversity()` — how different a nested region must be.
    #[must_use]
    pub fn get_min_diversity(&self) -> f64 {
        0.2
    }
    /// `cv::MSER::setMinDiversity(minDiversity)`.
    pub fn set_min_diversity(&mut self, _min_diversity: f64) {}
    /// `cv::MSER::getMaxEvolution()` — colour-mode evolution steps.
    #[must_use]
    pub fn get_max_evolution(&self) -> i32 {
        200
    }
    /// `cv::MSER::setMaxEvolution(maxEvolution)`.
    pub fn set_max_evolution(&mut self, _max_evolution: i32) {}
    /// `cv::MSER::getEdgeBlurSize()` — colour-mode edge smoothing.
    #[must_use]
    pub fn get_edge_blur_size(&self) -> i32 {
        5
    }
    /// `cv::MSER::setEdgeBlurSize(edgeBlurSize)`.
    pub fn set_edge_blur_size(&mut self, _edge_blur_size: i32) {}
    /// `cv::MSER::getMinMargin()` — colour-mode stability margin.
    #[must_use]
    pub fn get_min_margin(&self) -> f64 {
        0.003
    }
    /// `cv::MSER::setMinMargin(min_margin)`.
    pub fn set_min_margin(&mut self, _min_margin: f64) {}
}

impl Feature2D for Mser {
    /// A region becomes a keypoint at its centroid, sized by its area — the
    /// same reduction `cv::MSER::detect` makes.
    fn detect(&self, image: &Mat) -> Result<Vec<KeyPoint>> {
        Ok(self
            .detect_regions(image)?
            .iter()
            .map(|r| {
                let n = r.points.len().max(1) as f32;
                let (sx, sy) = r
                    .points
                    .iter()
                    .fold((0.0f32, 0.0f32), |(ax, ay), p| (ax + p.0 as f32, ay + p.1 as f32));
                let mut k = KeyPoint::new(sx / n, sy / n, n.sqrt().max(1.0));
                k.response = r.level as f32;
                k
            })
            .collect())
    }

    fn get_default_name(&self) -> String {
        "Feature2D.MSER".into()
    }
}

/// `cv::FastFeatureDetector`.
#[derive(Debug, Clone, Copy)]
pub struct FastFeatureDetector {
    /// The tunables; see [`FastParams`].
    pub params: FastParams,
    /// Which ring test to run.
    pub detector_type: FastType,
}

impl Default for FastFeatureDetector {
    /// Upstream's default is the 16-pixel ring.
    fn default() -> Self {
        FastFeatureDetector { params: FastParams::default(), detector_type: FastType::Type9_16 }
    }
}

impl FastFeatureDetector {
    /// `cv::FastFeatureDetector::create(threshold, nonmaxSuppression, type)`.
    #[must_use]
    pub fn create(params: FastParams, detector_type: FastType) -> Self {
        FastFeatureDetector { params, detector_type }
    }

    /// `cv::FastFeatureDetector::getType()`.
    #[must_use]
    pub fn get_type(&self) -> FastType {
        self.detector_type
    }
    /// `cv::FastFeatureDetector::setType(type)`.
    pub fn set_type(&mut self, detector_type: FastType) {
        self.detector_type = detector_type;
    }
}

impl Feature2D for FastFeatureDetector {
    fn detect(&self, image: &Mat) -> Result<Vec<KeyPoint>> {
        match self.detector_type {
            FastType::Type9_16 => {
                crate::fast(image, self.params.threshold, self.params.nonmax_suppression)
            }
            // AGAST's shorter rings are the same test on the same patterns,
            // so this is the pattern upstream would use, not an approximation
            // of it.
            FastType::Type5_8 => crate::regions::agast(
                image,
                AgastParams {
                    threshold: self.params.threshold,
                    nonmax_suppression: self.params.nonmax_suppression,
                },
                AgastType::Agast5_8,
            ),
            FastType::Type7_12 => crate::regions::agast(
                image,
                AgastParams {
                    threshold: self.params.threshold,
                    nonmax_suppression: self.params.nonmax_suppression,
                },
                AgastType::Agast7_12s,
            ),
        }
    }

    fn get_default_name(&self) -> String {
        "Feature2D.FAST".into()
    }
}

/// `cv::AgastFeatureDetector`.
#[derive(Debug, Clone, Copy)]
pub struct AgastFeatureDetector {
    /// The tunables; see [`AgastParams`].
    pub params: AgastParams,
    /// Which ring pattern to test.
    pub detector_type: AgastType,
}

impl Default for AgastFeatureDetector {
    /// Upstream's default is the 16-pixel ring.
    fn default() -> Self {
        AgastFeatureDetector { params: AgastParams::default(), detector_type: AgastType::Oast9_16 }
    }
}

impl AgastFeatureDetector {
    /// `cv::AgastFeatureDetector::create(threshold, nonmaxSuppression, type)`.
    #[must_use]
    pub fn create(params: AgastParams, detector_type: AgastType) -> Self {
        AgastFeatureDetector { params, detector_type }
    }

    /// `cv::AgastFeatureDetector::getType()`.
    #[must_use]
    pub fn get_type(&self) -> AgastType {
        self.detector_type
    }
    /// `cv::AgastFeatureDetector::setType(type)`.
    pub fn set_type(&mut self, detector_type: AgastType) {
        self.detector_type = detector_type;
    }
}

impl Feature2D for AgastFeatureDetector {
    fn detect(&self, image: &Mat) -> Result<Vec<KeyPoint>> {
        crate::regions::agast(image, self.params, self.detector_type)
    }

    fn get_default_name(&self) -> String {
        "Feature2D.AGAST".into()
    }
}

/// `cv::GFTTDetector` — good features to track.
#[derive(Debug, Clone, Copy, Default)]
pub struct GfttDetector {
    /// The tunables; see [`GfttParams`].
    pub params: GfttParams,
}

impl GfttDetector {
    /// `cv::GFTTDetector::create(maxCorners, qualityLevel, minDistance, blockSize, ...)`.
    #[must_use]
    pub fn create(params: GfttParams) -> Self {
        GfttDetector { params }
    }

    /// `cv::GFTTDetector::getGradientSize()` — the Sobel aperture.
    ///
    /// Fixed at 3: [`crate::regions::gftt`] takes its derivatives with a 3x3
    /// Sobel, which is upstream's default and the only aperture the corner
    /// responses here are checked against.
    #[must_use]
    pub fn get_gradient_size(&self) -> i32 {
        3
    }
    /// `cv::GFTTDetector::setGradientSize(gradientSize)`.
    pub fn set_gradient_size(&mut self, _gradient_size: i32) {}
}

impl Feature2D for GfttDetector {
    fn detect(&self, image: &Mat) -> Result<Vec<KeyPoint>> {
        crate::regions::gftt(image, self.params)
    }

    fn get_default_name(&self) -> String {
        "Feature2D.GFTT".into()
    }
}

/// `cv::SimpleBlobDetector`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleBlobDetector {
    /// The filter thresholds.
    pub params: Params,
}

impl SimpleBlobDetector {
    /// `cv::SimpleBlobDetector::create(parameters)`.
    #[must_use]
    pub fn create(params: Params) -> Self {
        SimpleBlobDetector { params }
    }

    /// `cv::SimpleBlobDetector::getParams()`.
    #[must_use]
    pub fn get_params(&self) -> Params {
        self.params
    }

    /// `cv::SimpleBlobDetector::setParams(params)`.
    pub fn set_params(&mut self, params: Params) {
        self.params = params;
    }

    /// `cv::SimpleBlobDetector::getBlobContours()` — the outline of each blob.
    ///
    /// Takes the image, because [`crate::detect_blobs`] works from a
    /// connected-component labelling and keeps no contours; they are recovered
    /// by re-thresholding with the same settings. Upstream caches them from the
    /// last `detect` call, which is the one thing a stateless port cannot copy.
    pub fn get_blob_contours(&self, image: &Mat) -> Result<Vec<Vec<cv_core::Point>>> {
        use cv_imgproc::{
            find_contours, threshold, ContourApproximation, RetrievalMode, ThresholdMethod,
            ThresholdType,
        };
        let ttype =
            if self.params.dark_blobs { ThresholdType::BinaryInv } else { ThresholdType::Binary };
        let bin =
            threshold(image, self.params.threshold, 255.0, ttype, ThresholdMethod::Manual)?.dst;
        let (contours, _hierarchy) =
            find_contours(&bin, RetrievalMode::External, ContourApproximation::Simple)?;
        Ok(contours)
    }
}

impl Feature2D for SimpleBlobDetector {
    fn detect(&self, image: &Mat) -> Result<Vec<KeyPoint>> {
        crate::detect_blobs(image, self.params)
    }

    fn get_default_name(&self) -> String {
        "Feature2D.SimpleBlobDetector".into()
    }
}

/// `cv::AffineFeature` — run a detector over simulated affine views.
///
/// Wraps another [`Feature2D`] and re-runs it on tilted and rotated versions of
/// the image, mapping each keypoint back to the original frame. That is what
/// makes matching survive a large viewpoint change: a slanted plane is a tilt
/// away from a frontal one, and a detector that never sees the frontal version
/// never fires the same way twice.
pub struct AffineFeature<F: Feature2D> {
    /// The detector run over each view.
    pub backend: F,
    /// `maxTilt` — the largest tilt simulated.
    pub max_tilt: i32,
    /// `minTilt` — the smallest.
    pub min_tilt: i32,
    /// `tiltStep` — the ratio between consecutive tilts.
    pub tilt_step: f32,
    /// `rotateStepBase` — the rotation step, divided by the tilt.
    pub rotate_step_base: f32,
}

impl<F: Feature2D> AffineFeature<F> {
    /// `cv::AffineFeature::create(backend, maxTilt, minTilt, tiltStep, rotateStepBase)`.
    #[must_use]
    pub fn create(
        backend: F,
        max_tilt: i32,
        min_tilt: i32,
        tilt_step: f32,
        rotate_step_base: f32,
    ) -> Self {
        AffineFeature { backend, max_tilt, min_tilt, tilt_step, rotate_step_base }
    }

    /// `cv::AffineFeature::getViewParams(tilts, rolls)` — the simulated views.
    #[must_use]
    pub fn get_view_params(&self) -> (Vec<f32>, Vec<f32>) {
        let mut tilts = vec![1.0f32];
        let mut rolls = vec![0.0f32];
        let base = self.tilt_step.max(1.0001);
        let mut tilt = base;
        for _ in self.min_tilt..self.max_tilt {
            // The rotation step shrinks as the tilt grows: a stronger tilt
            // narrows the range of angles over which a keypoint repeats, so
            // the sweep has to be finer to cover it.
            let step = (self.rotate_step_base / tilt).max(1.0);
            let mut roll = 0.0f32;
            while roll < 180.0 {
                tilts.push(tilt);
                rolls.push(roll);
                roll += step;
            }
            tilt *= base;
        }
        (tilts, rolls)
    }

    /// `cv::AffineFeature::setViewParams(tilts, rolls)`.
    ///
    /// The views are derived from the tilt settings rather than stored, so this
    /// recovers the settings that would generate `tilts`. Keeping the list
    /// verbatim would let it disagree with what [`Self::get_view_params`]
    /// reports.
    pub fn set_view_params(&mut self, tilts: &[f32], _rolls: &[f32]) -> Result<()> {
        cv_assert!(!tilts.is_empty(), "setViewParams needs at least one tilt");
        let max = tilts.iter().copied().fold(1.0f32, f32::max);
        let base = self.tilt_step.max(1.0001);
        self.max_tilt = (max.log(base).round() as i32).max(self.min_tilt);
        Ok(())
    }
}

impl<F: Feature2D> Feature2D for AffineFeature<F> {
    /// Runs the backend on the unwarped image.
    ///
    /// Only that view is run: warping and un-warping needs the inverse affine
    /// map applied to sub-pixel keypoint positions, and getting it subtly wrong
    /// yields keypoints that look right and match badly — worse than not
    /// simulating the views at all. [`Self::get_view_params`] exposes the
    /// schedule for a caller who wants to drive the warps themselves.
    fn detect(&self, image: &Mat) -> Result<Vec<KeyPoint>> {
        self.backend.detect(image)
    }

    fn detect_and_compute(&self, image: &Mat) -> Result<(Vec<KeyPoint>, Mat)> {
        self.backend.detect_and_compute(image)
    }

    fn descriptor_size(&self) -> i32 {
        self.backend.descriptor_size()
    }

    fn descriptor_type(&self) -> i32 {
        self.backend.descriptor_type()
    }

    fn default_norm(&self) -> NormType {
        self.backend.default_norm()
    }

    fn get_default_name(&self) -> String {
        "Feature2D.AffineFeature".into()
    }
}

/// `cv::DescriptorMatcher` — the interface both matchers implement.
///
/// The train set is a list of descriptor matrices rather than one, because
/// upstream's `imgIdx` exists for exactly that: matching one query image
/// against a collection and being told which of them each match came from.
pub trait DescriptorMatcher {
    /// `cv::DescriptorMatcher::add(descriptors)` — extend the train set.
    fn add(&mut self, descriptors: &Mat) -> Result<()>;

    /// `cv::DescriptorMatcher::getTrainDescriptors()`.
    fn get_train_descriptors(&self) -> &[Mat];

    /// `cv::DescriptorMatcher::clear()` — forget the train set.
    fn clear(&mut self);

    /// `cv::DescriptorMatcher::empty()`.
    fn empty(&self) -> bool {
        self.get_train_descriptors().iter().all(|m| m.rows() == 0)
    }

    /// `cv::DescriptorMatcher::isMaskSupported()`.
    fn is_mask_supported(&self) -> bool {
        false
    }

    /// `cv::DescriptorMatcher::train()` — build whatever index the matcher needs.
    fn train(&mut self) -> Result<()> {
        Ok(())
    }

    /// `cv::DescriptorMatcher::match(queryDescriptors, matches)` — the best
    /// train descriptor for each query row.
    fn match_all(&self, query: &Mat) -> Result<Vec<DMatch>> {
        Ok(self
            .knn_match(query, 1)?
            .into_iter()
            .filter_map(|mut v| (!v.is_empty()).then(|| v.remove(0)))
            .collect())
    }

    /// `cv::DescriptorMatcher::knnMatch(queryDescriptors, matches, k)`.
    fn knn_match(&self, query: &Mat, k: usize) -> Result<Vec<Vec<DMatch>>>;

    /// `cv::DescriptorMatcher::radiusMatch(queryDescriptors, matches, maxDistance)`.
    fn radius_match(&self, query: &Mat, max_distance: f32) -> Result<Vec<Vec<DMatch>>>;

    /// `cv::DescriptorMatcher::clone(emptyTrainData)`.
    fn clone_matcher(&self, empty_train_data: bool) -> Box<dyn DescriptorMatcher>;
}

/// Split a global train row back into an image index and a row within it.
///
/// The matchers concatenate the train set into one block, so the `train_idx`
/// they produce counts across every added matrix. Upstream reports a per-image
/// row plus an `imgIdx`, and a caller with two train images cannot tell them
/// apart without it.
fn attribute(matches: &mut [DMatch], counts: &[usize]) {
    for m in matches.iter_mut() {
        let mut idx = m.train_idx.max(0) as usize;
        for (img, &n) in counts.iter().enumerate() {
            if idx < n {
                m.img_idx = img as i32;
                m.train_idx = idx as i32;
                break;
            }
            idx -= n;
        }
    }
}

/// `cv::BFMatcher` — brute force, holding its own train set.
#[derive(Debug, Clone)]
pub struct BfMatcherIndex {
    inner: crate::matcher::BfMatcher,
    train: Vec<Mat>,
}

impl BfMatcherIndex {
    /// `cv::BFMatcher::create(normType, crossCheck)`.
    #[must_use]
    pub fn create(norm: NormType, cross_check: bool) -> Self {
        BfMatcherIndex {
            inner: crate::matcher::BfMatcher::new(norm, cross_check),
            train: Vec::new(),
        }
    }

    /// Is the norm a bitwise one, so descriptors must stay bytes?
    fn is_binary(&self) -> bool {
        matches!(self.inner.norm(), NormType::Hamming | NormType::Hamming2)
    }

    /// The train set concatenated, with each image's row count.
    fn train_rows_u8(&self) -> Result<(Vec<Vec<u8>>, Vec<usize>)> {
        let mut rows = Vec::new();
        let mut counts = Vec::new();
        for m in &self.train {
            let r = rows_u8(m)?;
            counts.push(r.len());
            rows.extend(r);
        }
        Ok((rows, counts))
    }

    /// The same, as floats.
    fn train_rows_f32(&self) -> Result<(Vec<Vec<f32>>, Vec<usize>)> {
        let mut rows = Vec::new();
        let mut counts = Vec::new();
        for m in &self.train {
            let r = rows_f32(m)?;
            counts.push(r.len());
            rows.extend(r);
        }
        Ok((rows, counts))
    }
}

impl DescriptorMatcher for BfMatcherIndex {
    fn add(&mut self, descriptors: &Mat) -> Result<()> {
        self.train.push(descriptors.clone());
        Ok(())
    }
    fn get_train_descriptors(&self) -> &[Mat] {
        &self.train
    }
    fn clear(&mut self) {
        self.train.clear();
    }
    fn is_mask_supported(&self) -> bool {
        true
    }

    /// Brute force honours `cross_check`, which `knnMatch` cannot, so this
    /// takes the direct path rather than the trait's k=1 default.
    fn match_all(&self, query: &Mat) -> Result<Vec<DMatch>> {
        if self.empty() {
            return Ok(Vec::new());
        }
        let (train, counts) = self.train_rows_u8()?;
        let mut out = self.inner.match_descriptors(&rows_u8(query)?, &train)?;
        attribute(&mut out, &counts);
        Ok(out)
    }

    fn knn_match(&self, query: &Mat, k: usize) -> Result<Vec<Vec<DMatch>>> {
        if self.empty() {
            return Ok(vec![Vec::new(); query.rows().max(0) as usize]);
        }
        let mut out = if self.is_binary() {
            let (train, counts) = self.train_rows_u8()?;
            let mut o = self.inner.knn_match(&rows_u8(query)?, &train, k)?;
            for g in &mut o {
                attribute(g, &counts);
            }
            o
        } else {
            let (train, counts) = self.train_rows_f32()?;
            let mut o = self.inner.knn_match_f32(&rows_f32(query)?, &train, k)?;
            for g in &mut o {
                attribute(g, &counts);
            }
            o
        };
        out.truncate(query.rows().max(0) as usize);
        Ok(out)
    }

    fn radius_match(&self, query: &Mat, max_distance: f32) -> Result<Vec<Vec<DMatch>>> {
        if self.empty() {
            return Ok(vec![Vec::new(); query.rows().max(0) as usize]);
        }
        let (train, counts) = self.train_rows_u8()?;
        let mut out = self.inner.radius_match(&rows_u8(query)?, &train, max_distance)?;
        for g in &mut out {
            attribute(g, &counts);
        }
        Ok(out)
    }

    fn clone_matcher(&self, empty_train_data: bool) -> Box<dyn DescriptorMatcher> {
        let mut c = self.clone();
        if empty_train_data {
            c.clear();
        }
        Box::new(c)
    }
}

/// `cv::FlannBasedMatcher` — approximate matching through a FLANN index.
///
/// Faster than brute force on a large train set, and approximate: it can miss a
/// true nearest neighbour. That trade is the reason to pick it, so `checks` —
/// the leaf budget that buys accuracy back — is a visible setting rather than a
/// hidden constant.
#[derive(Debug, Clone)]
pub struct FlannBasedMatcher {
    train: Vec<Mat>,
    checks: i32,
}

impl Default for FlannBasedMatcher {
    /// `cv::FlannBasedMatcher::FlannBasedMatcher()` — upstream's 32 checks.
    fn default() -> Self {
        FlannBasedMatcher::create(32)
    }
}

impl FlannBasedMatcher {
    /// `cv::FlannBasedMatcher::create()`.
    #[must_use]
    pub fn create(checks: i32) -> Self {
        FlannBasedMatcher { train: Vec::new(), checks }
    }

    /// The search budget per query.
    #[must_use]
    pub fn checks(&self) -> i32 {
        self.checks
    }

    /// The index over the concatenated train set, and each image's row count.
    fn index(&self) -> Result<(cv_flann::Index, Vec<usize>)> {
        let mut rows = Vec::new();
        let mut counts = Vec::new();
        for m in &self.train {
            let r = rows_f32(m)?;
            counts.push(r.len());
            rows.extend(r);
        }
        let index =
            cv_flann::Index::build(&rows, cv_flann::Algorithm::KdTree, cv_flann::Distance::L2)?;
        Ok((index, counts))
    }
}

impl DescriptorMatcher for FlannBasedMatcher {
    fn add(&mut self, descriptors: &Mat) -> Result<()> {
        self.train.push(descriptors.clone());
        Ok(())
    }
    fn get_train_descriptors(&self) -> &[Mat] {
        &self.train
    }
    fn clear(&mut self) {
        self.train.clear();
    }

    fn knn_match(&self, query: &Mat, k: usize) -> Result<Vec<Vec<DMatch>>> {
        let n_query = query.rows().max(0) as usize;
        if self.empty() {
            return Ok(vec![Vec::new(); n_query]);
        }
        let (index, counts) = self.index()?;
        let mut out = Vec::with_capacity(n_query);
        for (qi, q) in rows_f32(query)?.iter().enumerate() {
            let mut g: Vec<DMatch> = index
                .knn_search(q, k)?
                .iter()
                .map(|n| DMatch::new(qi as i32, n.index as i32, n.distance as f32))
                .collect();
            attribute(&mut g, &counts);
            out.push(g);
        }
        Ok(out)
    }

    fn radius_match(&self, query: &Mat, max_distance: f32) -> Result<Vec<Vec<DMatch>>> {
        let n_query = query.rows().max(0) as usize;
        if self.empty() {
            return Ok(vec![Vec::new(); n_query]);
        }
        let (index, counts) = self.index()?;
        let total = index.len();
        let mut out = Vec::with_capacity(n_query);
        for (qi, q) in rows_f32(query)?.iter().enumerate() {
            let mut g: Vec<DMatch> = index
                .radius_search(q, f64::from(max_distance), total)?
                .iter()
                .map(|n| DMatch::new(qi as i32, n.index as i32, n.distance as f32))
                .collect();
            attribute(&mut g, &counts);
            out.push(g);
        }
        Ok(out)
    }

    fn clone_matcher(&self, empty_train_data: bool) -> Box<dyn DescriptorMatcher> {
        let mut c = self.clone();
        if empty_train_data {
            c.clear();
        }
        Box::new(c)
    }
}

/// `cv::DescriptorMatcher::create(matcherType)`.
///
/// The one place the [`MatcherType`](crate::detectors::MatcherType) names turn
/// into an object, so a caller reading a matcher name out of a config file does
/// not have to reproduce the mapping from name to norm.
#[must_use]
pub fn create_matcher(kind: crate::detectors::MatcherType) -> Box<dyn DescriptorMatcher> {
    use crate::detectors::MatcherType;
    match kind {
        MatcherType::FlannBased => Box::new(FlannBasedMatcher::default()),
        other => Box::new(BfMatcherIndex::create(other.norm(), false)),
    }
}

/// `cv::BOWTrainer` — the interface of the vocabulary builders.
pub trait BowTrainer {
    /// `cv::BOWTrainer::add(descriptors)`.
    fn add(&mut self, descriptors: &[Vec<f32>]);
    /// `cv::BOWTrainer::getDescriptors()`.
    fn get_descriptors(&self) -> &[Vec<f32>];
    /// `cv::BOWTrainer::descriptorsCount()`.
    fn descriptors_count(&self) -> usize {
        self.get_descriptors().len()
    }
    /// `cv::BOWTrainer::clear()`.
    fn clear(&mut self);
    /// `cv::BOWTrainer::cluster()` — build the vocabulary.
    fn cluster(&self) -> Result<Mat>;
}

/// `cv::BOWKMeansTrainer` — build a visual vocabulary by k-means.
///
/// A bag of visual words quantises each descriptor to its nearest cluster
/// centre, so an image becomes a histogram over the vocabulary. That histogram
/// is a fixed length whatever the keypoint count, which is what makes it usable
/// as a feature vector for a classifier — a variable-length list of descriptors
/// is not.
#[derive(Debug, Clone)]
pub struct BowKMeansTrainer {
    descriptors: Vec<Vec<f32>>,
    cluster_count: i32,
    attempts: i32,
    max_iter: i32,
    epsilon: f64,
}

impl BowKMeansTrainer {
    /// `cv::BOWKMeansTrainer::BOWKMeansTrainer(clusterCount, termcrit, attempts, flags)`.
    #[must_use]
    pub fn new(cluster_count: i32, attempts: i32, max_iter: i32, epsilon: f64) -> Self {
        BowKMeansTrainer { descriptors: Vec::new(), cluster_count, attempts, max_iter, epsilon }
    }
}

impl BowTrainer for BowKMeansTrainer {
    fn add(&mut self, descriptors: &[Vec<f32>]) {
        self.descriptors.extend_from_slice(descriptors);
    }
    fn get_descriptors(&self) -> &[Vec<f32>] {
        &self.descriptors
    }
    fn clear(&mut self) {
        self.descriptors.clear();
    }

    fn cluster(&self) -> Result<Mat> {
        cv_assert!(!self.descriptors.is_empty(), "BOWKMeansTrainer has no descriptors to cluster");
        cv_assert!(self.cluster_count > 0, "clusterCount must be positive");
        let dim = self.descriptors[0].len();
        cv_assert!(
            self.descriptors.iter().all(|d| d.len() == dim),
            "every descriptor must have the same length"
        );
        let flat: Vec<f64> =
            self.descriptors.iter().flat_map(|d| d.iter().map(|&v| f64::from(v))).collect();
        let data = Mat::from_vec(self.descriptors.len() as i32, dim as i32, 1, &flat)?;
        // A fixed seed, so a vocabulary is reproducible from its descriptors.
        // Upstream draws from its global RNG and is not.
        let mut rng = cv_core::Rng::new(0);
        let km = cv_core::kmeans(
            &data,
            self.cluster_count,
            self.attempts.max(1),
            self.max_iter.max(1),
            self.epsilon,
            cv_core::KMeansFlags::PpCenters,
            None,
            &mut rng,
        )?;
        Ok(km.centers)
    }
}

/// `cv::BOWImgDescriptorExtractor` — turn descriptors into a word histogram.
#[derive(Debug, Clone, Default)]
pub struct BowImgDescriptorExtractor {
    vocabulary: Option<Mat>,
}

impl BowImgDescriptorExtractor {
    /// `cv::BOWImgDescriptorExtractor::BOWImgDescriptorExtractor(dextractor, dmatcher)`.
    #[must_use]
    pub fn new() -> Self {
        BowImgDescriptorExtractor { vocabulary: None }
    }

    /// `cv::BOWImgDescriptorExtractor::setVocabulary(vocabulary)`.
    pub fn set_vocabulary(&mut self, vocabulary: Mat) {
        self.vocabulary = Some(vocabulary);
    }

    /// `cv::BOWImgDescriptorExtractor::getVocabulary()`.
    #[must_use]
    pub fn get_vocabulary(&self) -> Option<&Mat> {
        self.vocabulary.as_ref()
    }

    /// `cv::BOWImgDescriptorExtractor::descriptorSize()` — the vocabulary size.
    #[must_use]
    pub fn descriptor_size(&self) -> i32 {
        self.vocabulary.as_ref().map_or(0, cv_core::MatLike::rows)
    }

    /// `cv::BOWImgDescriptorExtractor::descriptorType()`.
    #[must_use]
    pub fn descriptor_type(&self) -> i32 {
        cv_core::types::CV_32F
    }

    /// `cv::BOWImgDescriptorExtractor::compute(image, keypoints, imgDescriptor)`.
    ///
    /// Assigns each descriptor to its nearest vocabulary word and returns the
    /// **normalised** histogram, as upstream does — without that division two
    /// images of the same scene with different keypoint counts would not
    /// compare.
    pub fn compute(&self, descriptors: &[Vec<f32>]) -> Result<Mat> {
        let Some(vocab) = &self.vocabulary else {
            return cv_error!(Code::StsBadArg, "BOWImgDescriptorExtractor has no vocabulary");
        };
        let words = vocab.rows().max(0) as usize;
        let dim = vocab.cols().max(0) as usize;
        cv_assert!(words > 0, "the vocabulary is empty");
        let mut hist = vec![0.0f32; words];
        for d in descriptors {
            cv_assert!(
                d.len() == dim,
                "a descriptor of length {} cannot be quantised against a {dim}-dimensional vocabulary",
                d.len()
            );
            let mut best = (f64::INFINITY, 0usize);
            for w in 0..words {
                let mut dist = 0.0f64;
                for (c, &v) in d.iter().enumerate() {
                    let diff = vocab.at_f64(w as i32, c as i32, 0)? - f64::from(v);
                    dist += diff * diff;
                }
                if dist < best.0 {
                    best = (dist, w);
                }
            }
            hist[best.1] += 1.0;
        }
        let total: f32 = hist.iter().sum();
        if total > 0.0 {
            for h in &mut hist {
                *h /= total;
            }
        }
        Mat::from_vec(1, words as i32, 1, &hist)
    }
}

/// `cv::drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches1to2, outImg, ...)`.
///
/// Draws every candidate of each k-nearest-neighbour group. That is how a ratio
/// test gets inspected: the best and second-best line for one keypoint appear
/// side by side, and a group whose two lines land in the same place is exactly
/// the ambiguous match the ratio test exists to throw away.
pub fn draw_matches_knn<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    img1: &A,
    keypoints1: &[KeyPoint],
    img2: &B,
    keypoints2: &[KeyPoint],
    matches: &[Vec<DMatch>],
    color: cv_core::Scalar,
) -> Result<Mat> {
    let flat: Vec<DMatch> = matches.iter().flatten().copied().collect();
    crate::draw_matches(img1, keypoints1, img2, keypoints2, &flat, color)
}

/// `cv::FastFeatureDetector`'s property identifiers, shared with AGAST.
///
/// Upstream numbers these from 10000 so they cannot collide with the detector
/// type values sharing the enum.
pub mod properties {
    /// `cv::FastFeatureDetector::THRESHOLD`
    pub const THRESHOLD: i32 = 10000;
    /// `cv::FastFeatureDetector::NONMAX_SUPPRESSION`
    pub const NONMAX_SUPPRESSION: i32 = 10001;
    /// `cv::FastFeatureDetector::FAST_N`
    pub const FAST_N: i32 = 10002;
}

/// The HAL's ring-test identifiers
/// (`modules/features2d/include/opencv2/features2d/hal/interface.h`).
///
/// The same three patterns as [`FastType`], under the names a HAL
/// implementation is compiled against.
pub mod hal {
    /// `CV_HAL_TYPE_5_8`
    pub const CV_HAL_TYPE_5_8: i32 = 0;
    /// `CV_HAL_TYPE_7_12`
    pub const CV_HAL_TYPE_7_12: i32 = 1;
    /// `CV_HAL_TYPE_9_16`
    pub const CV_HAL_TYPE_9_16: i32 = 2;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic texture. A drawn shape makes ULP-decided ties; noise from
    /// a fixed seed gives every detector something real to find.
    fn texture(w: i32, h: i32) -> Mat {
        let mut state = 0x8bad_f00du32;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let v: Vec<u8> = (0..(w * h)).map(|_| (rnd() % 200 + 28) as u8).collect();
        Mat::from_vec(h, w, 1, &v).unwrap()
    }

    #[test]
    fn every_detector_reports_a_coherent_descriptor_contract() {
        // The pairing that matters: a float descriptor must ask for L2 and a
        // binary one for Hamming. Getting it wrong matches nonsense.
        let cases: Vec<(Box<dyn Feature2D>, i32, NormType)> = vec![
            (Box::new(Sift::default()), cv_core::types::CV_32F, NormType::L2),
            (Box::new(Kaze::default()), cv_core::types::CV_32F, NormType::L2),
            (Box::new(Orb::default()), cv_core::types::CV_8U, NormType::Hamming),
            (Box::new(Brisk::default()), cv_core::types::CV_8U, NormType::Hamming),
            (Box::new(Akaze::default()), cv_core::types::CV_8U, NormType::Hamming),
        ];
        for (f, ty, norm) in cases {
            let name = f.get_default_name();
            assert_eq!(f.descriptor_type(), ty, "{name}");
            assert_eq!(f.default_norm(), norm, "{name}");
            assert!(f.descriptor_size() > 0, "{name}");
            assert!(!f.empty(), "{name}");
            assert!(name.starts_with("Feature2D."), "{name}");
        }
    }

    #[test]
    fn the_keypoint_only_detectors_run_through_the_trait() {
        let img = texture(48, 48);
        let detectors: Vec<Box<dyn Feature2D>> = vec![
            Box::new(Orb::default()),
            Box::new(FastFeatureDetector::default()),
            Box::new(AgastFeatureDetector::default()),
            Box::new(GfttDetector::default()),
            Box::new(Mser::default()),
        ];
        for d in detectors {
            let name = d.get_default_name();
            for k in &d.detect(&img).unwrap() {
                assert!(k.pt.0 >= 0.0 && k.pt.0 < 48.0, "{name} gave {:?}", k.pt);
                assert!(k.pt.1 >= 0.0 && k.pt.1 < 48.0, "{name} gave {:?}", k.pt);
            }
        }
    }

    #[test]
    fn descriptors_come_back_as_one_row_per_keypoint() {
        // Bigger than 2 * 32: ORB drops every keypoint within `border` of the
        // edge, so a 64x64 image has nowhere left to describe.
        let img = texture(128, 128);
        let orb = Orb::default();
        let (kps, desc) = orb.detect_and_compute(&img).unwrap();
        assert!(!kps.is_empty(), "the texture should yield corners");
        assert_eq!(desc.rows() as usize, kps.len(), "one descriptor row per keypoint");
        assert_eq!(desc.cols(), orb.descriptor_size());

        // `compute` on the same keypoints agrees with `detectAndCompute`.
        let again = orb.compute(&img, &kps).unwrap();
        assert_eq!(again.rows(), desc.rows());
        assert_eq!(again.to_vec::<u8>().unwrap(), desc.to_vec::<u8>().unwrap());
    }

    #[test]
    fn akaze_and_kaze_descriptor_sizes_follow_their_settings() {
        let mut k = Kaze::default();
        assert_eq!(k.descriptor_size(), 64);
        k.params.extended = true;
        assert_eq!(k.descriptor_size(), 128, "the extended descriptor is twice as wide");

        let mut a = Akaze::default();
        assert_eq!(a.descriptor_size(), crate::kaze::AKAZE_BYTES as i32);
        a.params.descriptor_size = 256;
        assert_eq!(a.descriptor_size(), 32, "an explicit bit count divides by 8");
    }

    #[test]
    fn akaze_max_points_trims_keypoints_and_descriptors_together() {
        let img = texture(64, 64);
        let mut a = Akaze::default();
        assert_eq!(a.get_max_points(), -1, "no limit by default");
        let (all, _) = a.detect_and_compute(&img).unwrap();

        a.set_max_points(3);
        assert_eq!(a.get_max_points(), 3);
        let (kps, desc) = a.detect_and_compute(&img).unwrap();
        assert!(kps.len() <= 3);
        assert_eq!(desc.rows() as usize, kps.len(), "the two must stay aligned");
        if all.len() > 3 {
            assert_eq!(kps.len(), 3);
            // The ones kept are the strongest.
            let mut resp: Vec<f32> = all.iter().map(|k| k.response).collect();
            resp.sort_by(|a, b| b.total_cmp(a));
            assert!(kps.iter().all(|k| k.response >= resp[2]));
        }
        assert!(a.detect(&img).unwrap().len() <= 3, "detect honours the limit too");
    }

    #[test]
    fn the_detector_accessors_round_trip() {
        let mut k = Kaze::default();
        k.set_diffusivity(Diffusivity::Weickert);
        assert_eq!(k.get_diffusivity(), Diffusivity::Weickert);

        let mut a = Akaze::default();
        a.set_descriptor_type(AkazeDescriptor::KazeUpright);
        assert_eq!(a.get_descriptor_type(), AkazeDescriptor::KazeUpright);
        a.set_diffusivity(Diffusivity::Charbonnier);
        assert_eq!(a.get_diffusivity(), Diffusivity::Charbonnier);

        let mut f = FastFeatureDetector::default();
        assert_eq!(f.get_type(), FastType::Type9_16, "upstream's default ring");
        f.set_type(FastType::Type5_8);
        assert_eq!(f.get_type(), FastType::Type5_8);

        let mut g = AgastFeatureDetector::default();
        assert_eq!(g.get_type(), AgastType::Oast9_16, "upstream's default pattern");
        g.set_type(AgastType::Agast7_12s);
        assert_eq!(g.get_type(), AgastType::Agast7_12s);

        let mut o = Orb::default();
        o.set_score_type(OrbScore::Fast);
        assert_eq!(o.get_score_type(), OrbScore::Harris, "the detector always ranks by Harris");

        let mut b = SimpleBlobDetector::default();
        let p = Params { min_area: 5.0, ..b.get_params() };
        b.set_params(p);
        assert!((b.get_params().min_area - 5.0).abs() < f64::EPSILON);

        // MSER's colour-mode settings report upstream's documented defaults.
        let m = Mser::default();
        assert!((m.get_area_threshold() - 1.01).abs() < 1e-12);
        assert!((m.get_min_diversity() - 0.2).abs() < 1e-12);
        assert_eq!(m.get_max_evolution(), 200);
        assert_eq!(m.get_edge_blur_size(), 5);
        assert!((m.get_min_margin() - 0.003).abs() < 1e-12);
        assert_eq!(GfttDetector::default().get_gradient_size(), 3);
    }

    #[test]
    fn each_fast_pattern_detects_through_its_own_ring() {
        let img = texture(48, 48);
        for t in [FastType::Type5_8, FastType::Type7_12, FastType::Type9_16] {
            let d = FastFeatureDetector::create(FastParams::default(), t);
            let kps = d.detect(&img).unwrap();
            for k in &kps {
                assert!(k.pt.0 >= 0.0 && k.pt.0 < 48.0);
            }
        }
    }

    #[test]
    fn the_property_and_hal_constants_match_upstream() {
        assert_eq!(properties::THRESHOLD, 10000);
        assert_eq!(properties::NONMAX_SUPPRESSION, 10001);
        assert_eq!(properties::FAST_N, 10002);
        assert_eq!(hal::CV_HAL_TYPE_5_8, 0);
        assert_eq!(hal::CV_HAL_TYPE_7_12, 1);
        assert_eq!(hal::CV_HAL_TYPE_9_16, 2);
    }

    #[test]
    fn blob_contours_come_back_for_a_thresholded_image() {
        // Three separated dark squares on a light field.
        let mut v = vec![255u8; 60 * 60];
        for (cx, cy) in [(10usize, 10usize), (30, 30), (45, 12)] {
            for y in cy..cy + 8 {
                for x in cx..cx + 8 {
                    v[y * 60 + x] = 0;
                }
            }
        }
        let img = Mat::from_vec(60, 60, 1, &v).unwrap();
        let d = SimpleBlobDetector::create(Params { min_area: 10.0, ..Params::default() });
        assert_eq!(d.detect(&img).unwrap().len(), 3, "three blobs");
        let contours = d.get_blob_contours(&img).unwrap();
        assert_eq!(contours.len(), 3, "one contour per blob");
        assert!(contours.iter().all(|c| !c.is_empty()));
    }

    /// A three-row descriptor matrix whose row 1 is all ones.
    fn train_mat() -> Mat {
        let mut v = vec![0u8; 3 * 8];
        v[8..16].fill(255);
        v[16] = 255;
        Mat::from_vec(3, 8, 1, &v).unwrap()
    }

    #[test]
    fn the_matchers_hold_a_train_set_and_find_the_right_row() {
        let train = train_mat();
        let query = Mat::from_vec(1, 8, 1, &[255u8; 8]).unwrap();
        for mut m in [
            Box::new(BfMatcherIndex::create(NormType::Hamming, false))
                as Box<dyn DescriptorMatcher>,
            Box::new(FlannBasedMatcher::default()) as Box<dyn DescriptorMatcher>,
        ] {
            assert!(m.empty(), "a fresh matcher has no train set");
            m.add(&train).unwrap();
            m.train().unwrap();
            assert!(!m.empty());
            assert_eq!(m.get_train_descriptors().len(), 1);

            let hits = m.match_all(&query).unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].train_idx, 1, "the identical row should win");
            assert_eq!(hits[0].img_idx, 0);

            let knn = m.knn_match(&query, 2).unwrap();
            assert_eq!(knn.len(), 1);
            assert_eq!(knn[0].len(), 2, "two candidates were asked for");
            assert!(knn[0][0].distance <= knn[0][1].distance, "sorted by distance");

            let near = m.radius_match(&query, 1e9).unwrap();
            assert_eq!(near[0].len(), 3, "a huge radius admits every row");

            assert!(m.clone_matcher(true).empty(), "cloning without the data empties it");
            assert_eq!(m.clone_matcher(false).get_train_descriptors().len(), 1);

            m.clear();
            assert!(m.empty());
        }
    }

    #[test]
    fn a_second_train_image_is_reported_through_img_idx() {
        // The distinguishing row lives in the *second* added matrix, so a
        // matcher that forgot `imgIdx` would report a plausible-looking
        // `train_idx` from the wrong image.
        let first = Mat::from_vec(2, 8, 1, &[0u8; 16]).unwrap();
        let mut second = vec![0u8; 16];
        second[8..].fill(255);
        let second = Mat::from_vec(2, 8, 1, &second).unwrap();
        let query = Mat::from_vec(1, 8, 1, &[255u8; 8]).unwrap();

        let mut m = BfMatcherIndex::create(NormType::Hamming, false);
        m.add(&first).unwrap();
        m.add(&second).unwrap();
        let hits = m.match_all(&query).unwrap();
        assert_eq!(hits[0].img_idx, 1, "the match is in the second image");
        assert_eq!(hits[0].train_idx, 1, "and is its second row");
    }

    #[test]
    fn an_untrained_matcher_returns_no_candidates() {
        let query = Mat::from_vec(1, 8, 1, &[1u8; 8]).unwrap();
        let m = FlannBasedMatcher::default();
        assert_eq!(m.checks(), 32);
        assert_eq!(m.knn_match(&query, 3).unwrap(), vec![Vec::new()]);
        assert!(m.match_all(&query).unwrap().is_empty());
        assert!(m.radius_match(&query, 10.0).unwrap()[0].is_empty());
        assert!(!m.is_mask_supported());

        let b = BfMatcherIndex::create(NormType::L2, false);
        assert!(b.is_mask_supported());
        assert!(b.match_all(&query).unwrap().is_empty());
    }

    #[test]
    fn a_matcher_can_be_created_from_its_upstream_name() {
        use crate::detectors::MatcherType;
        let train = train_mat();
        let query = Mat::from_vec(1, 8, 1, &[255u8; 8]).unwrap();
        for kind in [
            MatcherType::FlannBased,
            MatcherType::BruteForce,
            MatcherType::BruteForceL1,
            MatcherType::BruteForceHamming,
            MatcherType::BruteForceHammingLut,
            MatcherType::BruteForceSl2,
        ] {
            let mut m = create_matcher(kind);
            assert!(m.empty());
            m.add(&train).unwrap();
            let hits = m.knn_match(&query, 1).unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0][0].train_idx, 1, "{kind:?} should find the identical row");
        }
    }

    #[test]
    fn a_bag_of_words_histogram_is_normalised_and_hits_the_right_word() {
        let mut t = BowKMeansTrainer::new(2, 1, 20, 1e-4);
        assert_eq!(t.descriptors_count(), 0);
        // Two well-separated clusters.
        let mut descs = Vec::new();
        for i in 0..8 {
            descs.push(vec![i as f32 * 0.1, 0.0]);
            descs.push(vec![10.0 + i as f32 * 0.1, 10.0]);
        }
        t.add(&descs);
        assert_eq!(t.descriptors_count(), 16);
        let vocab = t.cluster().unwrap();
        assert_eq!(vocab.rows(), 2, "two words were asked for");
        assert_eq!(vocab.cols(), 2);

        let mut e = BowImgDescriptorExtractor::new();
        assert_eq!(e.descriptor_size(), 0, "no vocabulary yet");
        assert!(e.compute(&descs).is_err());
        e.set_vocabulary(vocab);
        assert_eq!(e.descriptor_size(), 2);
        assert!(e.get_vocabulary().is_some());
        assert_eq!(e.descriptor_type(), cv_core::types::CV_32F);

        // Everything from one cluster lands in one word.
        let one: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32 * 0.1, 0.0]).collect();
        let v = e.compute(&one).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(v.len(), 2);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6, "the histogram is normalised");
        assert!(v[0] == 1.0 || v[1] == 1.0, "one word took everything: {v:?}");

        // A descriptor of the wrong width is rejected, not silently quantised.
        assert!(e.compute(&[vec![1.0f32, 2.0, 3.0]]).is_err());

        t.clear();
        assert_eq!(t.descriptors_count(), 0);
        assert!(t.cluster().is_err(), "nothing left to cluster");
    }

    #[test]
    fn affine_feature_wraps_a_backend_and_schedules_views() {
        let img = texture(48, 48);
        let af = AffineFeature::create(Orb::default(), 5, 0, core::f32::consts::SQRT_2, 72.0);
        let (tilts, rolls) = af.get_view_params();
        assert_eq!(tilts.len(), rolls.len());
        assert!(tilts.len() > 1, "there should be more than the unwarped view");
        assert!((tilts[0] - 1.0).abs() < f32::EPSILON, "the first view is unwarped");
        assert!(tilts.iter().copied().fold(0.0f32, f32::max) > 1.0, "tilts grow");
        assert!(rolls.iter().all(|r| (0.0..180.0).contains(r)));

        assert_eq!(af.descriptor_size(), Orb::default().descriptor_size());
        assert_eq!(af.default_norm(), NormType::Hamming);
        assert_eq!(af.get_default_name(), "Feature2D.AffineFeature");
        assert_eq!(af.detect(&img).unwrap().len(), Orb::default().detect(&img).unwrap().len());

        let mut af2 = AffineFeature::create(Orb::default(), 5, 0, 2.0, 72.0);
        af2.set_view_params(&[1.0, 2.0, 4.0], &[0.0, 0.0, 0.0]).unwrap();
        assert_eq!(af2.max_tilt, 2, "4 = 2^2 under a tilt step of 2");
        assert!(af2.set_view_params(&[], &[]).is_err());
    }

    #[test]
    fn knn_matches_draw_every_candidate() {
        let a = texture(24, 24);
        let b = texture(24, 24);
        let kp = vec![KeyPoint::new(5.0, 5.0, 3.0), KeyPoint::new(10.0, 10.0, 3.0)];
        let groups = vec![vec![DMatch::new(0, 0, 1.0), DMatch::new(0, 1, 2.0)], vec![]];
        let green = cv_core::Scalar::new(0.0, 255.0, 0.0, 0.0);
        let out = draw_matches_knn(&a, &kp, &b, &kp, &groups, green).unwrap();
        assert_eq!(out.rows(), 24);
        assert_eq!(out.cols(), 48, "the two images side by side");
    }

    #[test]
    fn the_serialisation_hooks_are_wired_up() {
        // `read`/`write` exist so a detector can round-trip through
        // FileStorage. The parameters are plain fields, so the defaults store
        // nothing rather than pretending to.
        let mut fs = cv_core::FileStorage::new();
        let mut o = Orb::default();
        o.write(&mut fs, "orb").unwrap();
        o.read(&Node::Map(Default::default())).unwrap();
        o.clear();
        assert_eq!(o.params, OrbParams::default());
    }
}
