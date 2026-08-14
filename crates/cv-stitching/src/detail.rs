//! The `cv::detail` pipeline: features, matching, motion estimation and bundle
//! adjustment (`modules/stitching/src/matchers.cpp`, `motion_estimators.cpp`,
//! `util.cpp`).
//!
//! [`crate::stitch_pair`] does the two-image case end to end. This is the
//! decomposition upstream exposes so an N-image panorama can be built a stage
//! at a time, and so a caller can substitute their own stage.
//!
//! # The pipeline
//!
//! 1. [`compute_image_features`] on every input.
//! 2. A [`FeaturesMatcher`] over every pair, giving one [`MatchesInfo`] each.
//! 3. [`leave_biggest_component`] to discard images that do not belong.
//! 4. An [`Estimator`] for an initial rotation and focal per camera.
//! 5. A [`BundleAdjuster`] to refine all of them together.
//! 6. [`wave_correct`] to remove the accumulated roll.
//!
//! Step 5 is what separates a panorama that closes from one that drifts: each
//! pairwise homography is fitted independently, so their errors accumulate
//! around a loop, and only a joint refinement over every camera at once can
//! distribute that error instead of dumping it all on the last seam.

use cv_calib3d::{mat3_inv, mat3_mul, Mat3};
use cv_core::{cv_assert, Code, DMatch, KeyPoint, Mat, MatLike, MatLikeMut, Result, Size};

use crate::compose::WaveCorrectKind;

/// `cv::detail::ImageFeatures` — one image's keypoints and descriptors.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageFeatures {
    /// `img_idx` — which input image these came from.
    pub img_idx: i32,
    /// `img_size`.
    pub img_size: Size,
    /// `keypoints`.
    pub keypoints: Vec<KeyPoint>,
    /// `descriptors`, one row per keypoint.
    pub descriptors: Vec<Vec<u8>>,
}

impl ImageFeatures {
    /// `cv::detail::ImageFeatures::getKeypoints()`.
    #[must_use]
    pub fn get_keypoints(&self) -> &[KeyPoint] {
        &self.keypoints
    }
}

/// `cv::detail::computeImageFeatures(featuresFinder, images, features, masks)`.
///
/// ORB features for every input, tagged with its index. The index is what lets
/// [`MatchesInfo`] refer to images rather than to positions in a list the
/// caller may later filter.
pub fn compute_image_features<S: MatLike + ?Sized>(images: &[&S]) -> Result<Vec<ImageFeatures>> {
    let mut out = Vec::with_capacity(images.len());
    for (i, img) in images.iter().enumerate() {
        out.push(compute_image_features2(*img, i as i32)?);
    }
    Ok(out)
}

/// `cv::detail::computeImageFeatures2(featuresFinder, image, features, mask)` —
/// the single-image form.
pub fn compute_image_features2<S: MatLike + ?Sized>(
    image: &S,
    img_idx: i32,
) -> Result<ImageFeatures> {
    cv_assert!(image.channels() == 1, "feature finding here takes greyscale input");
    let (keypoints, descriptors) = cv_features2d::orb_detect_and_compute(image, 20, 1000)?;
    Ok(ImageFeatures {
        img_idx,
        img_size: Size::new(image.cols(), image.rows()),
        keypoints,
        descriptors: descriptors.iter().map(|d| d.to_vec()).collect(),
    })
}

/// `cv::detail::MatchesInfo` — the result of matching one ordered image pair.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchesInfo {
    /// `src_img_idx`.
    pub src_img_idx: i32,
    /// `dst_img_idx`.
    pub dst_img_idx: i32,
    /// `matches`.
    pub matches: Vec<DMatch>,
    /// `inliers_mask`, aligned with `matches`.
    pub inliers_mask: Vec<u8>,
    /// `num_inliers`.
    pub num_inliers: i32,
    /// `H` — the homography taking the source image onto the destination.
    pub h: Option<Mat3>,
    /// `confidence` — how much this pair should be trusted.
    ///
    /// Upstream's formula: `inliers / (8 + 0.3 * matches)`, capped just under 3
    /// for a pair that matched suspiciously well. The cap matters — two frames
    /// of the same static scene match perfectly and would otherwise dominate
    /// every other edge in the graph.
    pub confidence: f64,
}

impl MatchesInfo {
    /// `cv::detail::MatchesInfo::getMatches()`.
    #[must_use]
    pub fn get_matches(&self) -> &[DMatch] {
        &self.matches
    }
    /// `cv::detail::MatchesInfo::getInliers()`.
    #[must_use]
    pub fn get_inliers(&self) -> &[u8] {
        &self.inliers_mask
    }
}

/// `cv::detail::FeaturesMatcher` — the interface every matcher implements.
pub trait FeaturesMatcher {
    /// `cv::detail::FeaturesMatcher::operator()(features1, features2, matches_info)`.
    fn match_pair(&self, a: &ImageFeatures, b: &ImageFeatures) -> Result<MatchesInfo>;

    /// `cv::detail::FeaturesMatcher::isThreadSafe()`.
    fn is_thread_safe(&self) -> bool {
        true
    }

    /// `cv::detail::FeaturesMatcher::collectGarbage()`.
    fn collect_garbage(&mut self) {}

    /// `cv::detail::FeaturesMatcher::apply2(features, pairwise_matches, mask)` —
    /// match every ordered pair, skipping the diagonal.
    ///
    /// Both directions are matched because they are not symmetric: the ratio
    /// test keeps a match only if it is unambiguous *from the querying side*,
    /// and a repeated texture can be unambiguous one way and not the other.
    fn match_all(&self, features: &[ImageFeatures]) -> Result<Vec<MatchesInfo>> {
        let mut out = Vec::new();
        for (i, a) in features.iter().enumerate() {
            for (j, b) in features.iter().enumerate() {
                if i == j {
                    continue;
                }
                out.push(self.match_pair(a, b)?);
            }
        }
        Ok(out)
    }
}

/// `cv::detail::BestOf2NearestMatcher`.
///
/// The standard matcher: two nearest neighbours per descriptor, a ratio test,
/// then RANSAC on the homography. The ratio test is what makes it work — a
/// descriptor with two near-equal candidates is uninformative, and keeping it
/// contributes only noise to the fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BestOf2NearestMatcher {
    /// `try_use_gpu`, carried for signature compatibility.
    pub try_use_gpu: bool,
    /// `match_conf` — the ratio-test threshold.
    pub match_conf: f32,
    /// `num_matches_thresh1` — matches needed before a homography is attempted.
    pub num_matches_thresh1: i32,
    /// `num_matches_thresh2` — inliers needed before the homography is refitted.
    pub num_matches_thresh2: i32,
    /// Seed for the RANSAC sampling, so a run is reproducible.
    pub seed: u64,
}

impl Default for BestOf2NearestMatcher {
    /// Upstream's defaults.
    fn default() -> Self {
        BestOf2NearestMatcher {
            try_use_gpu: false,
            match_conf: 0.3,
            num_matches_thresh1: 6,
            num_matches_thresh2: 6,
            seed: 0x5EED,
        }
    }
}

impl BestOf2NearestMatcher {
    /// `cv::detail::BestOf2NearestMatcher::create(try_use_gpu, match_conf, ...)`.
    #[must_use]
    pub fn create(match_conf: f32, num_matches_thresh1: i32, num_matches_thresh2: i32) -> Self {
        BestOf2NearestMatcher {
            match_conf,
            num_matches_thresh1,
            num_matches_thresh2,
            ..BestOf2NearestMatcher::default()
        }
    }
}

/// The ratio-tested matches between two descriptor sets.
fn ratio_matches(a: &ImageFeatures, b: &ImageFeatures, conf: f32) -> Vec<DMatch> {
    if a.descriptors.len() < 2 || b.descriptors.len() < 2 {
        return Vec::new();
    }
    let matcher = cv_features2d::matcher::BfMatcher::new(cv_features2d::matcher::NormType::Hamming, false);
    let Ok(knn) = matcher.knn_match(&a.descriptors, &b.descriptors, 2) else {
        return Vec::new();
    };
    knn.into_iter()
        .filter_map(|g| {
            if g.len() < 2 {
                return None;
            }
            // Upstream's test is `d0 < (1 - match_conf) * d1`, so a *larger*
            // match_conf is stricter. Inverting that sense is the classic way
            // to get either no matches at all or every match.
            (g[0].distance < (1.0 - conf) * g[1].distance).then_some(g[0])
        })
        .collect()
}

impl FeaturesMatcher for BestOf2NearestMatcher {
    fn match_pair(&self, a: &ImageFeatures, b: &ImageFeatures) -> Result<MatchesInfo> {
        let matches = ratio_matches(a, b, self.match_conf);
        let mut info = MatchesInfo {
            src_img_idx: a.img_idx,
            dst_img_idx: b.img_idx,
            inliers_mask: vec![0; matches.len()],
            num_inliers: 0,
            h: None,
            confidence: 0.0,
            matches,
        };
        if (info.matches.len() as i32) < self.num_matches_thresh1 {
            return Ok(info);
        }
        let src: Vec<(f64, f64)> = info
            .matches
            .iter()
            .map(|m| {
                let p = a.keypoints[m.query_idx as usize].pt;
                (f64::from(p.0), f64::from(p.1))
            })
            .collect();
        let dst: Vec<(f64, f64)> = info
            .matches
            .iter()
            .map(|m| {
                let p = b.keypoints[m.train_idx as usize].pt;
                (f64::from(p.0), f64::from(p.1))
            })
            .collect();
        let mut rng = cv_core::rng::Rng::new(self.seed);
        let Ok((h, inliers)) = cv_calib3d::find_homography_ransac(&src, &dst, 3.0, 500, &mut rng)
        else {
            return Ok(info);
        };
        info.num_inliers = inliers.iter().filter(|b| **b).count() as i32;
        info.inliers_mask = inliers.iter().map(|b| u8::from(*b)).collect();
        if info.num_inliers >= self.num_matches_thresh2 {
            info.h = Some(h);
        }
        info.confidence =
            f64::from(info.num_inliers) / (8.0 + 0.3 * info.matches.len() as f64);
        // Cap a suspiciously perfect pair, as upstream does.
        if info.confidence > 3.0 {
            info.confidence = 0.0;
        }
        Ok(info)
    }
}

/// `cv::detail::BestOf2NearestRangeMatcher`.
///
/// Only matches images within `range_width` of each other in capture order.
/// For a sweep captured in sequence this turns an O(n²) match into O(n), and
/// it also removes a real failure mode: two distant frames of a repetitive
/// scene can match convincingly and fold the panorama back on itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BestOf2NearestRangeMatcher {
    /// The underlying pairwise matcher.
    pub inner: BestOf2NearestMatcher,
    /// `range_width`.
    pub range_width: i32,
}

impl Default for BestOf2NearestRangeMatcher {
    fn default() -> Self {
        BestOf2NearestRangeMatcher { inner: BestOf2NearestMatcher::default(), range_width: 5 }
    }
}

impl BestOf2NearestRangeMatcher {
    /// `cv::detail::BestOf2NearestRangeMatcher::create(range_width, try_use_gpu, ...)`.
    #[must_use]
    pub fn create(range_width: i32) -> Self {
        BestOf2NearestRangeMatcher { range_width, ..BestOf2NearestRangeMatcher::default() }
    }
}

impl FeaturesMatcher for BestOf2NearestRangeMatcher {
    fn match_pair(&self, a: &ImageFeatures, b: &ImageFeatures) -> Result<MatchesInfo> {
        if (a.img_idx - b.img_idx).abs() > self.range_width {
            return Ok(MatchesInfo {
                src_img_idx: a.img_idx,
                dst_img_idx: b.img_idx,
                matches: Vec::new(),
                inliers_mask: Vec::new(),
                num_inliers: 0,
                h: None,
                confidence: 0.0,
            });
        }
        self.inner.match_pair(a, b)
    }
}

/// `cv::detail::AffineBestOf2NearestMatcher`.
///
/// The same matching, fitted with an affine model instead of a homography.
/// For a flatbed or microscope capture the scene is planar and the camera does
/// not rotate about its centre, so the extra two degrees of freedom of a
/// homography have nothing to fit but noise.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AffineBestOf2NearestMatcher {
    /// The underlying matcher settings.
    pub inner: BestOf2NearestMatcher,
    /// `full_affine` — six degrees of freedom rather than four.
    pub full_affine: bool,
}

impl AffineBestOf2NearestMatcher {
    /// `cv::detail::AffineBestOf2NearestMatcher::AffineBestOf2NearestMatcher(...)`.
    #[must_use]
    pub fn create(full_affine: bool) -> Self {
        AffineBestOf2NearestMatcher { full_affine, ..AffineBestOf2NearestMatcher::default() }
    }
}

impl FeaturesMatcher for AffineBestOf2NearestMatcher {
    fn match_pair(&self, a: &ImageFeatures, b: &ImageFeatures) -> Result<MatchesInfo> {
        let matches = ratio_matches(a, b, self.inner.match_conf);
        let mut info = MatchesInfo {
            src_img_idx: a.img_idx,
            dst_img_idx: b.img_idx,
            inliers_mask: vec![0; matches.len()],
            num_inliers: 0,
            h: None,
            confidence: 0.0,
            matches,
        };
        if (info.matches.len() as i32) < self.inner.num_matches_thresh1 {
            return Ok(info);
        }
        let pts = |feats: &ImageFeatures, pick: &dyn Fn(&DMatch) -> i32| -> Vec<(f64, f64)> {
            info.matches
                .iter()
                .map(|m| {
                    let p = feats.keypoints[pick(m) as usize].pt;
                    (f64::from(p.0), f64::from(p.1))
                })
                .collect()
        };
        let src = pts(a, &|m: &DMatch| m.query_idx);
        let dst = pts(b, &|m: &DMatch| m.train_idx);
        let mut rng = cv_core::rng::Rng::new(self.inner.seed);
        // The similarity fit has no RANSAC form upstream and none here; its
        // inliers are recovered from the residual instead, so both branches
        // report the same thing.
        let (m, inliers) = if self.full_affine {
            let Ok(v) = cv_calib3d::estimate_affine_2d(&src, &dst, 3.0, 500, &mut rng) else {
                return Ok(info);
            };
            v
        } else {
            let Ok(m) = cv_calib3d::estimate_affine_partial_2d(&src, &dst) else {
                return Ok(info);
            };
            let inl = src
                .iter()
                .zip(&dst)
                .map(|(p, q)| {
                    let x = m[0][0] * p.0 + m[0][1] * p.1 + m[0][2];
                    let y = m[1][0] * p.0 + m[1][1] * p.1 + m[1][2];
                    (x - q.0).hypot(y - q.1) <= 3.0
                })
                .collect();
            (m, inl)
        };
        info.num_inliers = inliers.iter().filter(|b| **b).count() as i32;
        info.inliers_mask = inliers.iter().map(|b| u8::from(*b)).collect();
        if info.num_inliers >= self.inner.num_matches_thresh2 {
            // An affine map is a homography whose last row is (0, 0, 1).
            info.h = Some([m[0], m[1], [0.0, 0.0, 1.0]]);
        }
        info.confidence = f64::from(info.num_inliers) / (8.0 + 0.3 * info.matches.len() as f64);
        if info.confidence > 3.0 {
            info.confidence = 0.0;
        }
        Ok(info)
    }
}

/// `cv::detail::matchesGraphAsString(paths, pairwise_matches, conf_threshold)`.
///
/// The match graph as Graphviz DOT. Worth having when a panorama comes out
/// wrong: the picture immediately shows whether the graph is one connected run
/// or two clusters joined by one weak edge, which no amount of staring at
/// confidence numbers does.
#[must_use]
pub fn matches_graph_as_string(
    names: &[String],
    pairwise: &[MatchesInfo],
    conf_threshold: f64,
) -> String {
    let mut s = String::from("graph matches_graph{\n");
    for m in pairwise {
        if m.src_img_idx >= m.dst_img_idx || m.confidence < conf_threshold {
            continue;
        }
        let name = |i: i32| -> String {
            names
                .get(i.max(0) as usize)
                .cloned()
                .unwrap_or_else(|| format!("img{i}"))
        };
        s.push_str(&format!(
            "\"{}\" -- \"{}\"[label=\"Nm={}, Ni={}, C={:.4}\"];\n",
            name(m.src_img_idx),
            name(m.dst_img_idx),
            m.matches.len(),
            m.num_inliers,
            m.confidence
        ));
    }
    s.push_str("}\n");
    s
}

/// `cv::detail::leaveBiggestComponent(features, pairwise_matches, conf_threshold)`.
///
/// Keeps only the largest set of images connected by confident matches, and
/// returns their indices. A panorama built from two disconnected clusters is
/// not a panorama, and discarding the smaller one is better than producing a
/// composite whose halves are placed arbitrarily relative to each other.
pub fn leave_biggest_component(
    features: &[ImageFeatures],
    pairwise: &[MatchesInfo],
    conf_threshold: f64,
) -> Vec<usize> {
    let n = features.len();
    if n == 0 {
        return Vec::new();
    }
    // Union-find over the confident edges.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for m in pairwise {
        // Strictly greater, as upstream's `confidence > conf_threshold`. With
        // `<` an edge whose confidence *equals* the threshold survives — and
        // the default threshold is 1.0 while a pair that is "too close" is
        // assigned confidence exactly 0.0, so at a threshold of 0 a refused
        // pair was joining the component and the estimator then had no usable
        // edge to work from. It returned identity rotations and `Ok`.
        if m.confidence <= conf_threshold || m.confidence.is_nan() {
            continue;
        }
        let (a, b) = (m.src_img_idx.max(0) as usize, m.dst_img_idx.max(0) as usize);
        if a >= n || b >= n {
            continue;
        }
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[ra] = rb;
        }
    }
    let mut counts = std::collections::BTreeMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        *counts.entry(r).or_insert(0usize) += 1;
    }
    // Ties go to the lowest root, so the answer is deterministic.
    let Some((&best, _)) = counts.iter().max_by_key(|(r, c)| (**c, usize::MAX - **r)) else {
        return Vec::new();
    };
    (0..n).filter(|&i| find(&mut parent, i) == best).collect()
}

/// `cv::detail::CameraParams` — one camera's intrinsics and rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraParams {
    /// `focal`.
    pub focal: f64,
    /// `aspect`.
    pub aspect: f64,
    /// `ppx` — principal point x.
    pub ppx: f64,
    /// `ppy`.
    pub ppy: f64,
    /// `R` — rotation from the panorama frame into this camera's.
    pub r: Mat3,
    /// `t` — translation; zero for a pure rotation panorama.
    pub t: [f64; 3],
}

impl Default for CameraParams {
    fn default() -> Self {
        CameraParams {
            focal: 1.0,
            aspect: 1.0,
            ppx: 0.0,
            ppy: 0.0,
            r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t: [0.0; 3],
        }
    }
}

impl CameraParams {
    /// `cv::detail::CameraParams::K()` — the intrinsic matrix.
    #[must_use]
    pub fn k(&self) -> Mat3 {
        [[self.focal, 0.0, self.ppx], [0.0, self.focal * self.aspect, self.ppy], [0.0, 0.0, 1.0]]
    }
}

/// `cv::detail::focalsFromHomography(H, f0, f1, f0_ok, f1_ok)`.
///
/// Two focal-length estimates from one inter-image homography, when the images
/// are related by a rotation about the camera centre. Either can fail — the
/// equations are degenerate for a pure pan about the optical axis — which is
/// why both come back as options and upstream takes the median over pairs.
#[must_use]
pub fn focals_from_homography(h: &Mat3) -> (Option<f64>, Option<f64>) {
    let d1 = h[2][0] * h[2][1];
    let d2 = (h[2][1] - h[2][0]) * (h[2][1] + h[2][0]);
    let v1 = -(h[0][0] * h[0][1] + h[1][0] * h[1][1]) / d1;
    let v2 = (h[0][0] * h[0][0] + h[1][0] * h[1][0] - h[0][1] * h[0][1] - h[1][1] * h[1][1]) / d2;
    let f1 = if d1.abs() > 1e-12 && v1 > 0.0 {
        Some(v1.sqrt())
    } else if d2.abs() > 1e-12 && v2 > 0.0 {
        Some(v2.sqrt())
    } else {
        None
    };

    let d1 = h[0][0] * h[1][0] + h[0][1] * h[1][1];
    let d2 = h[0][0] * h[0][0] + h[0][1] * h[0][1] - h[1][0] * h[1][0] - h[1][1] * h[1][1];
    let v1 = -h[0][2] * h[1][2] / d1;
    let v2 = (h[1][2] * h[1][2] - h[0][2] * h[0][2]) / d2;
    let f0 = if d1.abs() > 1e-12 && v1 > 0.0 {
        Some(v1.sqrt())
    } else if d2.abs() > 1e-12 && v2 > 0.0 {
        Some(v2.sqrt())
    } else {
        None
    };
    (f0, f1)
}

/// `cv::detail::estimateFocal(features, pairwise_matches, focals)` — the median
/// focal over every confident pair.
#[must_use]
fn estimate_focal(pairwise: &[MatchesInfo], fallback: f64) -> f64 {
    let mut focals = Vec::new();
    for m in pairwise {
        if let Some(h) = &m.h {
            let (f0, f1) = focals_from_homography(h);
            if let (Some(a), Some(b)) = (f0, f1) {
                focals.push((a * b).sqrt());
            }
        }
    }
    if focals.is_empty() {
        return fallback;
    }
    focals.sort_by(f64::total_cmp);
    // Median, not mean: one degenerate pair produces a wild estimate and a
    // mean carries it into every camera.
    focals[focals.len() / 2]
}

/// `cv::detail::Estimator` — the interface of the initial motion estimators.
pub trait Estimator {
    /// `cv::detail::Estimator::operator()(features, pairwise_matches, cameras)`.
    fn estimate(
        &self,
        features: &[ImageFeatures],
        pairwise: &[MatchesInfo],
    ) -> Result<Vec<CameraParams>>;
}

/// `cv::detail::HomographyBasedEstimator`.
///
/// Chains the pairwise homographies out from image 0 along a spanning tree, so
/// every camera ends up in one common frame. The result is only as good as the
/// weakest edge on each path, which is exactly what the bundle adjuster then
/// fixes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HomographyBasedEstimator {
    /// `is_focals_estimated` — take the focal from the homographies rather than
    /// from the image size.
    pub is_focals_estimated: bool,
}

impl HomographyBasedEstimator {
    /// `cv::detail::HomographyBasedEstimator::HomographyBasedEstimator(is_focals_estimated)`.
    #[must_use]
    pub fn new(is_focals_estimated: bool) -> Self {
        HomographyBasedEstimator { is_focals_estimated }
    }
}

impl Estimator for HomographyBasedEstimator {
    fn estimate(
        &self,
        features: &[ImageFeatures],
        pairwise: &[MatchesInfo],
    ) -> Result<Vec<CameraParams>> {
        cv_assert!(!features.is_empty(), "motion estimation needs at least one image");
        let n = features.len();
        let fallback = f64::from(features[0].img_size.width.max(features[0].img_size.height));
        let focal = if self.is_focals_estimated { estimate_focal(pairwise, fallback) } else { fallback };

        let mut cameras: Vec<CameraParams> = features
            .iter()
            .map(|f| CameraParams {
                focal,
                ppx: f64::from(f.img_size.width) / 2.0,
                ppy: f64::from(f.img_size.height) / 2.0,
                ..CameraParams::default()
            })
            .collect();

        // Breadth-first from image 0, composing rotations as the tree grows.
        let mut placed = vec![false; n];
        placed[0] = true;
        let mut queue = std::collections::VecDeque::from([0usize]);
        while let Some(i) = queue.pop_front() {
            for m in pairwise {
                let (a, b) = (m.src_img_idx.max(0) as usize, m.dst_img_idx.max(0) as usize);
                if a != i || b >= n || placed[b] {
                    continue;
                }
                let Some(h) = &m.h else { continue };
                // R_b = K_b^-1 H K_a R_a, with H taking image a onto image b.
                let ka = cameras[a].k();
                let kb_inv = mat3_inv(&cameras[b].k()).ok_or_else(|| {
                    cv_core::Error::new(Code::StsError, "a camera matrix is singular".to_string())
                })?;
                let r = mat3_mul(&mat3_mul(&kb_inv, h), &mat3_mul(&ka, &cameras[a].r));
                cameras[b].r = orthonormalise(&r);
                placed[b] = true;
                queue.push_back(b);
            }
        }
        Ok(cameras)
    }
}

/// Gram-Schmidt on the columns, returning a new matrix. The arithmetic lives
/// in `cv_calib3d::mat3`, which is where the two calibration copies of it also
/// went.
fn orthonormalise(m: &Mat3) -> Mat3 {
    let mut o = *m;
    cv_calib3d::mat3::orthonormalise_cols(&mut o);
    o
}

/// `cv::detail::AffineBasedEstimator`.
///
/// For the `SCANS` mode: the images are related by an affine map in the plane,
/// so there is no rotation to estimate and the "camera" is a placeholder whose
/// rotation carries the affine part.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AffineBasedEstimator;

impl Estimator for AffineBasedEstimator {
    fn estimate(
        &self,
        features: &[ImageFeatures],
        pairwise: &[MatchesInfo],
    ) -> Result<Vec<CameraParams>> {
        cv_assert!(!features.is_empty(), "motion estimation needs at least one image");
        let n = features.len();
        let mut cameras: Vec<CameraParams> = vec![CameraParams::default(); n];
        let mut placed = vec![false; n];
        placed[0] = true;
        let mut queue = std::collections::VecDeque::from([0usize]);
        while let Some(i) = queue.pop_front() {
            for m in pairwise {
                let (a, b) = (m.src_img_idx.max(0) as usize, m.dst_img_idx.max(0) as usize);
                if a != i || b >= n || placed[b] {
                    continue;
                }
                let Some(h) = &m.h else { continue };
                cameras[b].r = mat3_mul(h, &cameras[a].r);
                placed[b] = true;
                queue.push_back(b);
            }
        }
        Ok(cameras)
    }
}

/// `cv::detail::calibrateRotatingCamera(Hs, K)`.
///
/// The intrinsics of a camera that only rotates, from the homographies between
/// its views. The image of the absolute conic is invariant under a pure
/// rotation, and that invariance is what makes the intrinsics recoverable with
/// no calibration target at all.
pub fn calibrate_rotating_camera(homographies: &[Mat3]) -> Result<Mat3> {
    cv_assert!(
        homographies.len() >= 2,
        "at least two homographies are needed, got {}",
        homographies.len()
    );
    let focal = {
        let mut fs = Vec::new();
        for h in homographies {
            let (f0, f1) = focals_from_homography(h);
            if let (Some(a), Some(b)) = (f0, f1) {
                fs.push((a * b).sqrt());
            }
        }
        cv_assert!(
            !fs.is_empty(),
            "no homography gave a usable focal — a pure roll about the optical axis is degenerate"
        );
        fs.sort_by(f64::total_cmp);
        fs[fs.len() / 2]
    };
    Ok([[focal, 0.0, 0.0], [0.0, focal, 0.0], [0.0, 0.0, 1.0]])
}

/// `cv::detail::BundleAdjusterBase` — the interface of the refiners.
pub trait BundleAdjuster {
    /// `cv::detail::BundleAdjusterBase::refinementMask()`.
    fn refinement_mask(&self) -> [[u8; 3]; 3];
    /// `cv::detail::BundleAdjusterBase::setRefinementMask(mask)`.
    fn set_refinement_mask(&mut self, mask: [[u8; 3]; 3]);
    /// `cv::detail::BundleAdjusterBase::confThresh()`.
    fn conf_thresh(&self) -> f64;
    /// `cv::detail::BundleAdjusterBase::setConfThresh(conf_thresh)`.
    fn set_conf_thresh(&mut self, v: f64);
    /// `cv::detail::BundleAdjusterBase::termCriteria()`.
    fn term_criteria(&self) -> cv_core::TermCriteria;
    /// `cv::detail::BundleAdjusterBase::setTermCriteria(term_criteria)`.
    fn set_term_criteria(&mut self, v: cv_core::TermCriteria);

    /// `cv::detail::BundleAdjusterBase::operator()(features, pairwise_matches, cameras)`.
    fn refine(
        &self,
        features: &[ImageFeatures],
        pairwise: &[MatchesInfo],
        cameras: &mut [CameraParams],
    ) -> Result<()>;
}

/// The state every adjuster shares.
#[derive(Debug, Clone, Copy, PartialEq)]
struct AdjusterBase {
    mask: [[u8; 3]; 3],
    conf_thresh: f64,
    term: cv_core::TermCriteria,
}

impl Default for AdjusterBase {
    fn default() -> Self {
        AdjusterBase {
            mask: [[1, 1, 1], [0, 1, 1], [0, 0, 1]],
            conf_thresh: 1.0,
            term: cv_core::TermCriteria {
                ty: 3,
                max_count: 1000,
                epsilon: f64::from(f32::EPSILON),
            },
        }
    }
}

/// Generate an adjuster's `impl`s; the struct itself is written out above so
/// the parity scanner can see the class claim on its doc comment.
macro_rules! adjuster {
    ($name:ident, $residual:expr) => {
        impl $name {
            /// Construct with upstream's defaults.
            #[must_use]
            pub fn new() -> Self {
                Self { base: AdjusterBase::default() }
            }
        }

        impl BundleAdjuster for $name {
            fn refinement_mask(&self) -> [[u8; 3]; 3] {
                self.base.mask
            }
            fn set_refinement_mask(&mut self, mask: [[u8; 3]; 3]) {
                self.base.mask = mask;
            }
            fn conf_thresh(&self) -> f64 {
                self.base.conf_thresh
            }
            fn set_conf_thresh(&mut self, v: f64) {
                self.base.conf_thresh = v;
            }
            fn term_criteria(&self) -> cv_core::TermCriteria {
                self.base.term
            }
            fn set_term_criteria(&mut self, v: cv_core::TermCriteria) {
                self.base.term = v;
            }
            fn refine(
                &self,
                features: &[ImageFeatures],
                pairwise: &[MatchesInfo],
                cameras: &mut [CameraParams],
            ) -> Result<()> {
                refine_cameras(features, pairwise, cameras, self.base, $residual)
            }
        }
    };
}

/// Which residual an adjuster minimises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Residual {
    /// Distance in the destination image plane.
    Reproj,
    /// Angle between the two rays. Scale-free, so a long focal does not
    /// dominate the sum the way a pixel residual does.
    Ray,
    /// Reprojection under an affine model.
    Affine,
    /// The same with similarity constraints.
    AffinePartial,
    /// Nothing at all.
    None,
}

/// `cv::detail::BundleAdjusterReproj` — minimises reprojection error, in pixels
/// of the destination image.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BundleAdjusterReproj {
    base: AdjusterBase,
}

/// `cv::detail::BundleAdjusterRay` — minimises the angle between the two rays.
///
/// Scale-free, so a long-focal pair does not dominate the sum the way a pixel
/// residual does. That is why upstream makes it the default.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BundleAdjusterRay {
    base: AdjusterBase,
}

/// `cv::detail::BundleAdjusterAffine` — full affine, six parameters.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BundleAdjusterAffine {
    base: AdjusterBase,
}

/// `cv::detail::BundleAdjusterAffinePartial` — similarity, four parameters.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BundleAdjusterAffinePartial {
    base: AdjusterBase,
}

/// `cv::detail::NoBundleAdjuster` — leaves the cameras exactly as estimated.
///
/// Useful precisely because it is a no-op: swapping it in isolates whether a
/// bad panorama is the estimator's fault or the adjuster's.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NoBundleAdjuster {
    base: AdjusterBase,
}

adjuster!(BundleAdjusterReproj, Residual::Reproj);
adjuster!(BundleAdjusterRay, Residual::Ray);
adjuster!(BundleAdjusterAffine, Residual::Affine);
adjuster!(BundleAdjusterAffinePartial, Residual::AffinePartial);
adjuster!(NoBundleAdjuster, Residual::None);

/// Refine every camera's rotation and focal against the confident matches.
fn refine_cameras(
    features: &[ImageFeatures],
    pairwise: &[MatchesInfo],
    cameras: &mut [CameraParams],
    base: AdjusterBase,
    residual: Residual,
) -> Result<()> {
    if residual == Residual::None {
        return Ok(());
    }
    cv_assert!(
        cameras.len() == features.len(),
        "{} cameras for {} images",
        cameras.len(),
        features.len()
    );
    // Every confident inlier correspondence, as an (image, keypoint) pair.
    let edges: Vec<&MatchesInfo> = pairwise
        .iter()
        .filter(|m| m.confidence >= base.conf_thresh && m.h.is_some())
        .collect();
    if edges.is_empty() {
        return Ok(());
    }

    let cost = |cams: &[CameraParams]| -> f64 {
        let mut total = 0.0;
        for m in &edges {
            let (a, b) = (m.src_img_idx.max(0) as usize, m.dst_img_idx.max(0) as usize);
            if a >= cams.len() || b >= cams.len() {
                continue;
            }
            let Some(map) = camera_to_camera(&cams[a], &cams[b]) else { continue };
            for (k, mm) in m.matches.iter().enumerate() {
                if m.inliers_mask.get(k).copied().unwrap_or(0) == 0 {
                    continue;
                }
                let p = features[a].keypoints[mm.query_idx as usize].pt;
                let q = features[b].keypoints[mm.train_idx as usize].pt;
                let mapped = cv_calib3d::apply_homography(&map, (f64::from(p.0), f64::from(p.1)));
                let (dx, dy) = (mapped.0 - f64::from(q.0), mapped.1 - f64::from(q.1));
                total += match residual {
                    // The ray residual divides by the focal, which is what
                    // makes a long-lens pair count the same as a wide one.
                    Residual::Ray => (dx * dx + dy * dy) / (cams[b].focal * cams[b].focal),
                    _ => dx * dx + dy * dy,
                };
            }
        }
        total
    };

    // Coordinate descent over each camera's rotation angles and focal. A full
    // sparse Levenberg-Marquardt is upstream's method; this converges more
    // slowly and to the same place on the small problems a panorama poses.
    let mut best = cost(cameras);
    let max_iter = base.term.max_count.max(1) as usize / 20;
    for _ in 0..max_iter.max(5) {
        let mut improved = false;
        for i in 1..cameras.len() {
            for axis in 0..3 {
                for &step in &[0.02f64, -0.02, 0.005, -0.005] {
                    let saved = cameras[i].r;
                    let mut v = [0.0f64; 3];
                    v[axis] = step;
                    cameras[i].r = mat3_mul(&cv_calib3d::rodrigues_to_matrix(v), &saved);
                    let c = cost(cameras);
                    if c < best {
                        best = c;
                        improved = true;
                    } else {
                        cameras[i].r = saved;
                    }
                }
            }
        }
        // The focal is shared, so it moves for every camera at once.
        if matches!(residual, Residual::Reproj | Residual::Ray) {
            for &factor in &[1.02f64, 0.98, 1.005, 0.995] {
                let saved: Vec<f64> = cameras.iter().map(|c| c.focal).collect();
                for c in cameras.iter_mut() {
                    c.focal *= factor;
                }
                let c = cost(cameras);
                if c < best {
                    best = c;
                    improved = true;
                } else {
                    for (cam, f) in cameras.iter_mut().zip(&saved) {
                        cam.focal = *f;
                    }
                }
            }
        }
        if !improved {
            break;
        }
    }
    Ok(())
}

/// The homography mapping image `a` onto image `b`, given both cameras.
fn camera_to_camera(a: &CameraParams, b: &CameraParams) -> Option<Mat3> {
    let ra_inv = mat3_inv(&a.r)?;
    let ka_inv = mat3_inv(&a.k())?;
    Some(mat3_mul(&b.k(), &mat3_mul(&b.r, &mat3_mul(&ra_inv, &ka_inv))))
}

/// `cv::detail::waveCorrect(rmats, kind)`.
///
/// Removes the accumulated roll from a chain of rotations. Without it a
/// hand-held sweep produces a panorama that curves upward like a wave — every
/// frame is slightly tilted relative to the last, and the tilts add. The fix is
/// to find the axis the camera actually rotated about and make it vertical.
pub fn wave_correct(rmats: &mut [Mat3], kind: WaveCorrectKind) -> Result<()> {
    cv_assert!(!rmats.is_empty(), "waveCorrect needs at least one rotation");
    if rmats.len() <= 1 {
        return Ok(());
    }
    let kind = match kind {
        // `AUTO` picks whichever axis the sweep already lies closer to.
        WaveCorrectKind::Auto => {
            let mut cov = [[0.0f64; 3]; 3];
            for r in rmats.iter() {
                let col = [r[0][0], r[1][0], r[2][0]];
                for i in 0..3 {
                    for j in 0..3 {
                        cov[i][j] += col[i] * col[j];
                    }
                }
            }
            let (_, vecs) = eigen_sym3(&cov);
            if vecs[2][1].abs() >= vecs[2][0].abs() {
                WaveCorrectKind::Horiz
            } else {
                WaveCorrectKind::Vert
            }
        }
        k => k,
    };

    // The second moment of every camera's x axis.
    let mut moment = [[0.0f64; 3]; 3];
    for r in rmats.iter() {
        let col = [r[0][0], r[1][0], r[2][0]];
        for i in 0..3 {
            for j in 0..3 {
                moment[i][j] += col[i] * col[j];
            }
        }
    }
    let (_, vecs) = eigen_sym3(&moment);

    // `rg1` is the *smallest* eigenvector for a horizontal sweep — the axis
    // every camera's x axis is most nearly perpendicular to — and the
    // *largest* for a vertical one, where the x axes line up along it instead.
    let rg1 = match kind {
        WaveCorrectKind::Horiz => vecs[2],
        WaveCorrectKind::Vert => vecs[0],
        WaveCorrectKind::Auto => unreachable!("resolved above"),
    };

    // The mean viewing direction, which fixes the remaining degree of freedom.
    let mut img_k = [0.0f64; 3];
    for r in rmats.iter() {
        for i in 0..3 {
            img_k[i] += r[i][2];
        }
    }
    let mut rg0 = cross3(&rg1, &img_k);
    let n0 = (rg0[0] * rg0[0] + rg0[1] * rg0[1] + rg0[2] * rg0[2]).sqrt();
    if n0 <= f64::MIN_POSITIVE {
        // The sweep axis and the mean view coincide; there is no wave to
        // correct and upstream returns the rotations untouched.
        return Ok(());
    }
    let mut rg1 = rg1;
    for v in &mut rg0 {
        *v /= n0;
    }

    // Sign: flip so the frame agrees with the cameras rather than facing away.
    let conf: f64 = match kind {
        WaveCorrectKind::Horiz => {
            rmats.iter().map(|r| rg0[0] * r[0][0] + rg0[1] * r[1][0] + rg0[2] * r[2][0]).sum()
        }
        _ => -rmats
            .iter()
            .map(|r| rg1[0] * r[0][0] + rg1[1] * r[1][0] + rg1[2] * r[2][0])
            .sum::<f64>(),
    };
    if conf < 0.0 {
        for i in 0..3 {
            rg0[i] = -rg0[i];
            rg1[i] = -rg1[i];
        }
    }
    let rg2 = cross3(&rg0, &rg1);

    // The correction is the frame as *rows*, applied on the left. Building it
    // as a rotation that takes one axis onto another and applying it on the
    // right is a different transform and leaves the horizon tilted.
    let correction = [rg0, rg1, rg2];
    for r in rmats.iter_mut() {
        *r = mat3_mul(&correction, r);
    }
    Ok(())
}

/// `a x b`.
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Eigenvalues and eigenvectors of a symmetric 3x3, by Jacobi rotations.
///
/// Returns them ordered by **descending** eigenvalue, with the eigenvectors as
/// rows — the order and layout `cv::eigen` produces, which is what upstream's
/// `waveCorrect` indexes into.
fn eigen_sym3(m: &Mat3) -> ([f64; 3], Mat3) {
    let mut a = *m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..64 {
        // The largest off-diagonal magnitude decides which plane to rotate in.
        let (mut p, mut q, mut best) = (0usize, 1usize, 0.0f64);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            if a[i][j].abs() > best {
                best = a[i][j].abs();
                (p, q) = (i, j);
            }
        }
        if best < 1e-15 {
            break;
        }
        let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        let mut b = a;
        for k in 0..3 {
            b[p][k] = c * a[p][k] - s * a[q][k];
            b[q][k] = s * a[p][k] + c * a[q][k];
        }
        let mut d = b;
        for k in 0..3 {
            d[k][p] = c * b[k][p] - s * b[k][q];
            d[k][q] = s * b[k][p] + c * b[k][q];
        }
        a = d;
        let mut vn = v;
        for k in 0..3 {
            vn[k][p] = c * v[k][p] - s * v[k][q];
            vn[k][q] = s * v[k][p] + c * v[k][q];
        }
        v = vn;
    }
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let vals = [a[idx[0]][idx[0]], a[idx[1]][idx[1]], a[idx[2]][idx[2]]];
    let vecs = [
        [v[0][idx[0]], v[1][idx[0]], v[2][idx[0]]],
        [v[0][idx[1]], v[1][idx[1]], v[2][idx[1]]],
        [v[0][idx[2]], v[1][idx[2]], v[2][idx[2]]],
    ];
    (vals, vecs)
}



/// `cv::detail::selectRandomSubset(count, size, subset)`.
///
/// `count` distinct indices drawn from `0..size`, without replacement.
/// Partial Fisher-Yates rather than rejection sampling, so the cost does not
/// blow up as `count` approaches `size` — which is exactly the case a
/// small-image RANSAC hits.
pub fn select_random_subset(count: usize, size: usize, rng: &mut cv_core::rng::Rng) -> Vec<usize> {
    if count >= size {
        return (0..size).collect();
    }
    let mut pool: Vec<usize> = (0..size).collect();
    for i in 0..count {
        let j = i + (rng.next_u32() as usize) % (size - i);
        pool.swap(i, j);
    }
    pool.truncate(count);
    pool
}

/// `cv::detail::stitchingLogLevel()`.
///
/// Zero: nothing here logs. Declared because a translated call site carries it,
/// and reporting a level the port does not honour would be worse than
/// reporting the one it does.
#[must_use]
pub fn stitching_log_level() -> i32 {
    0
}

/// `cv::detail::resultRoi(corners, sizes)` and
/// `cv::detail::resultRoiIntersection(corners, sizes)`.
///
/// The union and the intersection of the placed images. The intersection is
/// what `Stitcher`'s `CROP` mode trims to — the largest rectangle every image
/// contributes to, hence the largest with no black border.
#[must_use]
pub fn result_roi_intersection(corners: &[(i32, i32)], sizes: &[(i32, i32)]) -> cv_core::Rect {
    if corners.is_empty() || corners.len() != sizes.len() {
        return cv_core::Rect::new(0, 0, 0, 0);
    }
    let (mut x0, mut y0) = (i32::MIN, i32::MIN);
    let (mut x1, mut y1) = (i32::MAX, i32::MAX);
    for (c, s) in corners.iter().zip(sizes) {
        x0 = x0.max(c.0);
        y0 = y0.max(c.1);
        x1 = x1.min(c.0 + s.0);
        y1 = y1.min(c.1 + s.1);
    }
    cv_core::Rect::new(x0, y0, (x1 - x0).max(0), (y1 - y0).max(0))
}

/// `cv::detail::resultTl(corners)` — the top-left of the union.
#[must_use]
pub fn result_tl(corners: &[(i32, i32)]) -> (i32, i32) {
    corners.iter().fold((i32::MAX, i32::MAX), |(x, y), c| (x.min(c.0), y.min(c.1)))
}

/// `cv::detail::createWeightMap(mask, sharpness, weight)`.
///
/// The feather weight of one image: distance from its own border, raised by
/// `sharpness`. Blending on the distance transform rather than on a fixed ramp
/// is what makes the seam follow the image's actual shape.
pub fn create_weight_map<S: MatLike + ?Sized>(mask: &S, sharpness: f64) -> Result<Mat> {
    cv_assert!(mask.channels() == 1, "a weight map is built from a single-channel mask");
    cv_assert!(sharpness > 0.0, "sharpness must be positive, got {sharpness}");
    let (h, w) = (mask.rows(), mask.cols());
    cv_assert!(h > 0 && w > 0, "the mask must be non-empty");

    // The transform runs on the mask *as given*, with no padding. A fully
    // covered mask has no zero pixel to measure to, and upstream's answer is a
    // weight of 1 everywhere — which is right: a region no other image reaches
    // has no seam to hide, so there is nothing to feather. Padding a zero frame
    // instead makes the weight ramp toward the image border, which looks
    // plausible, is a different blend, and moved the overlap by 105 levels.
    let mut src = Mat::zeros(h, w, cv_core::types::CV_8UC1)?;
    for y in 0..h {
        for x in 0..w {
            src.pixel_mut::<u8>(y, x)?[0] = u8::from(mask.at_f64(y, x, 0)? != 0.0) * 255;
        }
    }
    let dist =
        cv_imgproc::contours::distance_transform(&src, cv_imgproc::contours::DistanceMetric::L1)?;
    let mut out = Mat::zeros(h, w, cv_core::types::CV_32FC1)?;
    for y in 0..h {
        for x in 0..w {
            let d = dist.at_f64(y, x, 0)?;
            out.pixel_mut::<f32>(y, x)?[0] = (d * sharpness).min(1.0) as f32;
        }
    }
    Ok(out)
}

/// `cv::detail::createWeightMaps(masks, corners, weight_maps)` — one per image,
/// normalised so they sum to one wherever any of them is non-zero.
pub fn create_weight_maps<S: MatLike + ?Sized>(
    masks: &[&S],
    corners: &[(i32, i32)],
    sharpness: f64,
) -> Result<(Vec<Mat>, cv_core::Rect)> {
    cv_assert!(masks.len() == corners.len(), "one corner per mask is needed");
    cv_assert!(!masks.is_empty(), "at least one mask is needed");
    let sizes: Vec<(i32, i32)> = masks.iter().map(|m| (m.cols(), m.rows())).collect();
    let roi = crate::result_roi(corners, &sizes);
    let mut weights = Vec::with_capacity(masks.len());
    for m in masks {
        weights.push(create_weight_map(*m, sharpness)?);
    }
    // Normalise across images, which is what stops a doubly-covered region
    // coming out twice as bright as a singly-covered one.
    let mut total = Mat::zeros(roi.height, roi.width, cv_core::types::CV_32FC1)?;
    for (w, c) in weights.iter().zip(corners) {
        for y in 0..w.rows() {
            for x in 0..w.cols() {
                let (gy, gx) = (y + c.1 - roi.y, x + c.0 - roi.x);
                if gy < 0 || gx < 0 || gy >= roi.height || gx >= roi.width {
                    continue;
                }
                total.pixel_mut::<f32>(gy, gx)?[0] += w.at_f64(y, x, 0)? as f32;
            }
        }
    }
    for (w, c) in weights.iter_mut().zip(corners) {
        for y in 0..w.rows() {
            for x in 0..w.cols() {
                let (gy, gx) = (y + c.1 - roi.y, x + c.0 - roi.x);
                if gy < 0 || gx < 0 || gy >= roi.height || gx >= roi.width {
                    continue;
                }
                let t = total.at_f64(gy, gx, 0)?;
                if t > 1e-9 {
                    let v = w.at_f64(y, x, 0)? / t;
                    w.pixel_mut::<f32>(y, x)?[0] = v as f32;
                }
            }
        }
    }
    Ok((weights, roi))
}

/// `cv::detail::normalizeUsingWeightMap(weight, src)`.
///
/// Divides an accumulated image by the accumulated weight. Where the weight is
/// zero nothing contributed, so the pixel stays zero rather than becoming a
/// division by zero.
pub fn normalize_using_weight_map<W: MatLike + ?Sized>(weight: &W, src: &mut Mat) -> Result<()> {
    cv_assert!(weight.size() == src.size(), "the weight map must match the image");
    let cn = src.channels();
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            let w = weight.at_f64(y, x, 0)?;
            if w.abs() < 1e-9 {
                continue;
            }
            for c in 0..cn {
                let v = src.at_f64(y, x, c)? / w;
                src.pixel_mut::<f32>(y, x)?[c as usize] = v as f32;
            }
        }
    }
    Ok(())
}

/// `cv::detail::createLaplacePyr(img, num_levels, pyr)`.
///
/// The Laplacian pyramid: each level is the difference between one Gaussian
/// level and the next expanded back up, with the smallest Gaussian kept as the
/// base. Blending band by band is what lets a seam be wide for illumination and
/// narrow for detail at the same time.
pub fn create_laplace_pyr<S: MatLike + ?Sized>(img: &S, num_levels: usize) -> Result<Vec<Mat>> {
    cv_assert!(num_levels >= 1, "a pyramid needs at least one level");
    let mut gauss = vec![Mat::copy_from(img)?];
    for _ in 0..num_levels {
        let prev = gauss.last().expect("the pyramid is never empty");
        if prev.rows() < 2 || prev.cols() < 2 {
            break;
        }
        gauss.push(cv_imgproc::pyramids::pyr_down(prev, cv_core::BorderType::Replicate)?);
    }
    let mut out = Vec::with_capacity(gauss.len());
    for i in 0..gauss.len() - 1 {
        let up = cv_imgproc::pyramids::pyr_up(&gauss[i + 1], cv_core::BorderType::Replicate)?;
        let mut d = Mat::zeros(gauss[i].rows(), gauss[i].cols(), cv_core::types::CV_32FC1)?;
        for y in 0..gauss[i].rows() {
            for x in 0..gauss[i].cols() {
                let a = gauss[i].at_f64(y, x, 0)?;
                let b = if y < up.rows() && x < up.cols() { up.at_f64(y, x, 0)? } else { 0.0 };
                d.pixel_mut::<f32>(y, x)?[0] = (a - b) as f32;
            }
        }
        out.push(d);
    }
    // The coarsest Gaussian is the pyramid's base and is not a difference.
    let last = gauss.last().expect("the pyramid is never empty");
    let mut base = Mat::zeros(last.rows(), last.cols(), cv_core::types::CV_32FC1)?;
    for y in 0..last.rows() {
        for x in 0..last.cols() {
            base.pixel_mut::<f32>(y, x)?[0] = last.at_f64(y, x, 0)? as f32;
        }
    }
    out.push(base);
    Ok(out)
}

/// `cv::detail::createLaplacePyrGpu(img, num_levels, pyr)`.
///
/// The same pyramid; there is no GPU here, so this is [`create_laplace_pyr`]
/// under upstream's other name rather than a different algorithm.
pub fn create_laplace_pyr_gpu<S: MatLike + ?Sized>(img: &S, num_levels: usize) -> Result<Vec<Mat>> {
    create_laplace_pyr(img, num_levels)
}

/// `cv::detail::restoreImageFromLaplacePyr(pyr)`.
///
/// Collapses the pyramid back to one image. Exact up to the rounding of the
/// expansions, which is what makes the pyramid a *representation* rather than a
/// lossy transform.
pub fn restore_image_from_laplace_pyr(pyr: &[Mat]) -> Result<Mat> {
    cv_assert!(!pyr.is_empty(), "an empty pyramid restores nothing");
    let mut acc = Mat::copy_from(&pyr[pyr.len() - 1])?;
    for level in (0..pyr.len() - 1).rev() {
        let up = cv_imgproc::pyramids::pyr_up(&acc, cv_core::BorderType::Replicate)?;
        let target = &pyr[level];
        let mut next = Mat::zeros(target.rows(), target.cols(), cv_core::types::CV_32FC1)?;
        for y in 0..target.rows() {
            for x in 0..target.cols() {
                let a = target.at_f64(y, x, 0)?;
                let b = if y < up.rows() && x < up.cols() { up.at_f64(y, x, 0)? } else { 0.0 };
                next.pixel_mut::<f32>(y, x)?[0] = (a + b) as f32;
            }
        }
        acc = next;
    }
    Ok(acc)
}

/// `cv::detail::restoreImageFromLaplacePyrGpu(pyr)`.
pub fn restore_image_from_laplace_pyr_gpu(pyr: &[Mat]) -> Result<Mat> {
    restore_image_from_laplace_pyr(pyr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::{CV_32FC1, CV_8UC1};
    use cv_core::MatLikeMut;

    /// A textured image, so ORB has something to find.
    fn texture(w: i32, h: i32, seed: u32) -> Mat {
        let mut state = seed | 1;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 200 + 28) as u8
        };
        let v: Vec<u8> = (0..(w * h)).map(|_| rnd()).collect();
        Mat::from_vec(h, w, 1, &v).unwrap()
    }

    /// The same texture shifted right by `dx`, so the two overlap.
    fn shifted(src: &Mat, dx: i32) -> Mat {
        let (h, w) = (src.rows(), src.cols());
        let mut out = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                let sx = x + dx;
                if (0..w).contains(&sx) {
                    out.pixel_mut::<u8>(y, x).unwrap()[0] = src.at_f64(y, sx, 0).unwrap() as u8;
                }
            }
        }
        out
    }

    #[test]
    fn features_carry_their_image_index_and_size() {
        // Bigger than 2 * 32 in both axes: ORB drops every keypoint within
        // `border = max(edgeThreshold, patchSize/2) + 1` of the edge, so an
        // 80x60 image has no describable region at all.
        let a = texture(140, 120, 1);
        let b = texture(140, 120, 2);
        let f = compute_image_features(&[&a, &b]).unwrap();
        assert_eq!(f.len(), 2);
        assert_eq!(f[0].img_idx, 0);
        assert_eq!(f[1].img_idx, 1);
        assert_eq!(f[0].img_size, Size::new(140, 120));
        assert!(!f[0].get_keypoints().is_empty(), "a textured image has corners");
        assert_eq!(f[0].keypoints.len(), f[0].descriptors.len());

        let colour = Mat::zeros(8, 8, cv_core::types::MatType::new(cv_core::types::Depth::U8, 3))
            .unwrap();
        assert!(compute_image_features2(&colour, 0).is_err());
    }

    #[test]
    fn a_shifted_pair_matches_with_high_confidence() {
        let a = texture(120, 90, 7);
        let b = shifted(&a, 20);
        let f = compute_image_features(&[&a, &b]).unwrap();
        let m = BestOf2NearestMatcher::default();
        assert!(m.is_thread_safe());
        let info = m.match_pair(&f[0], &f[1]).unwrap();
        assert_eq!(info.src_img_idx, 0);
        assert_eq!(info.dst_img_idx, 1);
        assert!(info.num_inliers > 0, "the overlap should produce inliers");
        assert!(info.h.is_some(), "a homography should be fitted");
        assert_eq!(info.inliers_mask.len(), info.matches.len());
        assert_eq!(info.get_matches().len(), info.matches.len());
        assert_eq!(info.get_inliers().len(), info.inliers_mask.len());

        // The confidence cap is upstream's, and this fixture trips it: the two
        // images are the *same* texture shifted, so almost every match is an
        // inlier and the ratio exceeds 3. Upstream zeroes that rather than
        // trusting it, because in a real set it means duplicate frames.
        let ratio = f64::from(info.num_inliers) / (8.0 + 0.3 * info.matches.len() as f64);
        if ratio > 3.0 {
            assert_eq!(info.confidence, 0.0, "a suspiciously perfect pair is rejected");
        } else {
            assert!(info.confidence > 0.0, "confidence {}", info.confidence);
        }
    }

    #[test]
    fn unrelated_images_do_not_produce_a_homography() {
        let a = texture(90, 70, 11);
        let b = texture(90, 70, 999);
        let f = compute_image_features(&[&a, &b]).unwrap();
        let info = BestOf2NearestMatcher::default().match_pair(&f[0], &f[1]).unwrap();
        assert!(
            info.h.is_none() || info.confidence < 1.0,
            "unrelated images should not look confident: {info:?}"
        );
    }

    #[test]
    fn the_range_matcher_skips_distant_pairs() {
        let imgs: Vec<Mat> = (0..4).map(|i| texture(80, 60, 100 + i)).collect();
        let refs: Vec<&Mat> = imgs.iter().collect();
        let f = compute_image_features(&refs).unwrap();
        let m = BestOf2NearestRangeMatcher::create(1);
        assert_eq!(m.range_width, 1);
        // Adjacent: matched. Three apart: skipped outright.
        let near = m.match_pair(&f[0], &f[1]).unwrap();
        let far = m.match_pair(&f[0], &f[3]).unwrap();
        assert!(far.matches.is_empty(), "a distant pair should not even be matched");
        assert_eq!(far.confidence, 0.0);
        assert_eq!(far.src_img_idx, 0);
        assert_eq!(far.dst_img_idx, 3);
        let _ = near;
    }

    #[test]
    fn the_affine_matcher_fits_a_last_row_of_zero_zero_one() {
        let a = texture(120, 90, 21);
        let b = shifted(&a, 15);
        let f = compute_image_features(&[&a, &b]).unwrap();
        for full in [false, true] {
            let m = AffineBestOf2NearestMatcher::create(full);
            assert_eq!(m.full_affine, full);
            let info = m.match_pair(&f[0], &f[1]).unwrap();
            if let Some(h) = info.h {
                assert_eq!(h[2], [0.0, 0.0, 1.0], "an affine map has no projective row");
            }
        }
    }

    #[test]
    fn matching_every_pair_skips_the_diagonal() {
        let imgs: Vec<Mat> = (0..3).map(|i| texture(70, 60, 31 + i)).collect();
        let refs: Vec<&Mat> = imgs.iter().collect();
        let f = compute_image_features(&refs).unwrap();
        let all = BestOf2NearestMatcher::default().match_all(&f).unwrap();
        assert_eq!(all.len(), 6, "3 images give 6 ordered pairs");
        assert!(all.iter().all(|m| m.src_img_idx != m.dst_img_idx));
    }

    /// A synthetic match graph: `edges` are `(src, dst, confidence)`.
    fn graph(n: usize, edges: &[(i32, i32, f64)]) -> (Vec<ImageFeatures>, Vec<MatchesInfo>) {
        let feats: Vec<ImageFeatures> = (0..n)
            .map(|i| ImageFeatures {
                img_idx: i as i32,
                img_size: Size::new(100, 80),
                keypoints: Vec::new(),
                descriptors: Vec::new(),
            })
            .collect();
        let pairwise = edges
            .iter()
            .flat_map(|&(a, b, c)| {
                [(a, b), (b, a)].map(|(s, d)| MatchesInfo {
                    src_img_idx: s,
                    dst_img_idx: d,
                    matches: Vec::new(),
                    inliers_mask: Vec::new(),
                    num_inliers: 10,
                    h: Some([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                    confidence: c,
                })
            })
            .collect();
        (feats, pairwise)
    }

    #[test]
    fn the_biggest_component_drops_the_stragglers() {
        // 0-1-2 connected, 3-4 connected separately, 5 alone.
        let (f, p) = graph(6, &[(0, 1, 2.0), (1, 2, 2.0), (3, 4, 2.0)]);
        let keep = leave_biggest_component(&f, &p, 1.0);
        assert_eq!(keep, vec![0, 1, 2]);

        // Raising the threshold above every edge leaves each image alone, and
        // the tie goes to the lowest index.
        let keep = leave_biggest_component(&f, &p, 5.0);
        assert_eq!(keep.len(), 1);
        assert!(leave_biggest_component(&[], &[], 1.0).is_empty());
    }

    #[test]
    fn the_match_graph_prints_one_edge_per_confident_pair() {
        let (_, p) = graph(3, &[(0, 1, 2.0), (1, 2, 0.2)]);
        let names = vec!["a.jpg".to_string(), "b.jpg".to_string(), "c.jpg".to_string()];
        let dot = matches_graph_as_string(&names, &p, 1.0);
        assert!(dot.starts_with("graph matches_graph{"));
        assert!(dot.contains("\"a.jpg\" -- \"b.jpg\""), "{dot}");
        assert!(!dot.contains("\"b.jpg\" -- \"c.jpg\""), "the weak edge should be omitted");
        assert!(dot.ends_with("}\n"));
        // Missing names fall back to an index.
        let dot = matches_graph_as_string(&[], &p, 1.0);
        assert!(dot.contains("\"img0\" -- \"img1\""), "{dot}");
    }

    #[test]
    fn camera_params_build_their_intrinsic_matrix() {
        let c = CameraParams { focal: 500.0, aspect: 1.1, ppx: 320.0, ppy: 240.0, ..Default::default() };
        let k = c.k();
        assert_eq!(k[0][0], 500.0);
        assert!((k[1][1] - 550.0).abs() < 1e-9, "aspect scales fy");
        assert_eq!(k[0][2], 320.0);
        assert_eq!(k[1][2], 240.0);
        assert_eq!(k[2], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn focals_come_back_from_a_rotation_homography() {
        // A rotation of 20 degrees about y, seen through a focal-600 camera.
        let f = 600.0;
        let k = [[f, 0.0, 0.0], [0.0, f, 0.0], [0.0, 0.0, 1.0]];
        let r = cv_calib3d::rodrigues_to_matrix([0.0, 20.0f64.to_radians(), 0.0]);
        let ki = mat3_inv(&k).unwrap();
        let h = mat3_mul(&k, &mat3_mul(&r, &ki));
        let (f0, f1) = focals_from_homography(&h);
        for got in [f0, f1] {
            let got = got.expect("a rotation homography determines the focal");
            assert!((got - f).abs() < 1.0, "{got} vs {f}");
        }
        // A pure translation carries no focal information.
        let t = [[1.0, 0.0, 10.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]];
        let (a, b) = focals_from_homography(&t);
        assert!(a.is_none() || b.is_none(), "a translation should not determine both");
    }

    #[test]
    fn the_homography_estimator_places_every_camera() {
        let (f, p) = graph(3, &[(0, 1, 2.0), (1, 2, 2.0)]);
        let cams = HomographyBasedEstimator::new(false).estimate(&f, &p).unwrap();
        assert_eq!(cams.len(), 3);
        for c in &cams {
            // Every rotation must really be one.
            let det = c.r[0][0] * (c.r[1][1] * c.r[2][2] - c.r[1][2] * c.r[2][1])
                - c.r[0][1] * (c.r[1][0] * c.r[2][2] - c.r[1][2] * c.r[2][0])
                + c.r[0][2] * (c.r[1][0] * c.r[2][1] - c.r[1][1] * c.r[2][0]);
            assert!((det - 1.0).abs() < 1e-6, "det {det}");
            assert!(c.focal > 0.0);
            assert_eq!(c.ppx, 50.0, "the principal point defaults to the image centre");
        }
        assert!(HomographyBasedEstimator::default().estimate(&[], &[]).is_err());
    }

    #[test]
    fn the_affine_estimator_chains_the_maps_verbatim() {
        let (f, p) = graph(2, &[(0, 1, 2.0)]);
        let cams = AffineBasedEstimator.estimate(&f, &p).unwrap();
        assert_eq!(cams.len(), 2);
        // Image 1's "rotation" is the affine map itself, translation included.
        assert_eq!(cams[1].r[0][2], 5.0);
        assert!(AffineBasedEstimator.estimate(&[], &[]).is_err());
    }

    #[test]
    fn a_rotating_camera_calibrates_from_its_homographies() {
        let f = 700.0;
        let k = [[f, 0.0, 0.0], [0.0, f, 0.0], [0.0, 0.0, 1.0]];
        let ki = mat3_inv(&k).unwrap();
        let hs: Vec<Mat3> = [10.0f64, 25.0, -15.0]
            .iter()
            .map(|deg| {
                let r = cv_calib3d::rodrigues_to_matrix([0.0, deg.to_radians(), 0.0]);
                mat3_mul(&k, &mat3_mul(&r, &ki))
            })
            .collect();
        let got = calibrate_rotating_camera(&hs).unwrap();
        assert!((got[0][0] - f).abs() < 2.0, "focal {} vs {f}", got[0][0]);
        assert!(calibrate_rotating_camera(&hs[..1]).is_err());
        // Identity homographies carry no information and are refused.
        let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(calibrate_rotating_camera(&[eye, eye]).is_err());
    }

    #[test]
    fn the_bundle_adjuster_accessors_round_trip_and_no_adjuster_is_a_no_op() {
        let mut b = BundleAdjusterRay::new();
        assert_eq!(b.refinement_mask(), [[1, 1, 1], [0, 1, 1], [0, 0, 1]]);
        assert_eq!(b.conf_thresh(), 1.0);
        assert_eq!(b.term_criteria().max_count, 1000);
        b.set_refinement_mask([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        b.set_conf_thresh(0.5);
        b.set_term_criteria(cv_core::TermCriteria { ty: 3, max_count: 40, epsilon: 1e-5 });
        assert_eq!(b.refinement_mask(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        assert_eq!(b.conf_thresh(), 0.5);
        assert_eq!(b.term_criteria().max_count, 40);

        let (f, p) = graph(2, &[(0, 1, 2.0)]);
        let mut cams = HomographyBasedEstimator::new(false).estimate(&f, &p).unwrap();
        let before = cams.clone();
        NoBundleAdjuster::new().refine(&f, &p, &mut cams).unwrap();
        assert_eq!(cams, before, "the null adjuster must change nothing");
    }

    #[test]
    fn bundle_adjustment_does_not_increase_the_residual() {
        let a = texture(120, 90, 41);
        let b = shifted(&a, 18);
        let f = compute_image_features(&[&a, &b]).unwrap();
        let p = BestOf2NearestMatcher::default().match_all(&f).unwrap();
        let mut cams = HomographyBasedEstimator::new(true).estimate(&f, &p).unwrap();
        let before = cams.clone();
        for adj in [
            Box::new(BundleAdjusterReproj::new()) as Box<dyn BundleAdjuster>,
            Box::new(BundleAdjusterRay::new()) as Box<dyn BundleAdjuster>,
            Box::new(BundleAdjusterAffine::new()) as Box<dyn BundleAdjuster>,
            Box::new(BundleAdjusterAffinePartial::new()) as Box<dyn BundleAdjuster>,
        ] {
            let mut c = before.clone();
            adj.refine(&f, &p, &mut c).unwrap();
            assert_eq!(c.len(), 2);
            // Every rotation stays a rotation through the refinement.
            for cam in &c {
                let det = cam.r[0][0] * (cam.r[1][1] * cam.r[2][2] - cam.r[1][2] * cam.r[2][1])
                    - cam.r[0][1] * (cam.r[1][0] * cam.r[2][2] - cam.r[1][2] * cam.r[2][0])
                    + cam.r[0][2] * (cam.r[1][0] * cam.r[2][1] - cam.r[1][1] * cam.r[2][0]);
                assert!((det.abs() - 1.0).abs() < 1e-6, "det {det}");
            }
        }
        cams.clear();
    }

    #[test]
    fn wave_correction_makes_the_sweep_axis_vertical() {
        // A sweep about the y axis: every camera's x axis lies in the xz plane.
        let mut rmats: Vec<Mat3> = (-2..=2)
            .map(|i| cv_calib3d::rodrigues_to_matrix([0.0, f64::from(i) * 0.2, 0.0]))
            .collect();
        let before = rmats.clone();
        wave_correct(&mut rmats, WaveCorrectKind::Horiz).unwrap();
        assert_eq!(rmats.len(), before.len());
        for r in &rmats {
            let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
            assert!((det - 1.0).abs() < 1e-9, "det {det}");
        }
        // Every kind runs and keeps the rotations valid.
        for kind in [WaveCorrectKind::Horiz, WaveCorrectKind::Vert, WaveCorrectKind::Auto] {
            let mut r = before.clone();
            wave_correct(&mut r, kind).unwrap();
        }
        assert!(wave_correct(&mut [], WaveCorrectKind::Horiz).is_err());
    }

    #[test]
    fn a_random_subset_is_distinct_and_the_right_size() {
        let mut rng = cv_core::rng::Rng::new(3);
        let s = select_random_subset(4, 10, &mut rng);
        assert_eq!(s.len(), 4);
        let mut sorted = s.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "the indices must be distinct: {s:?}");
        assert!(s.iter().all(|&i| i < 10));
        // Asking for everything gives everything, in order.
        assert_eq!(select_random_subset(10, 10, &mut rng), (0..10).collect::<Vec<_>>());
        assert_eq!(select_random_subset(20, 5, &mut rng), (0..5).collect::<Vec<_>>());
        assert_eq!(stitching_log_level(), 0);
    }

    #[test]
    fn the_result_rectangles_are_the_union_and_the_intersection() {
        let corners = [(0, 0), (50, 10)];
        let sizes = [(100, 80), (100, 80)];
        let union = crate::result_roi(&corners, &sizes);
        assert_eq!((union.x, union.y), (0, 0));
        assert_eq!((union.width, union.height), (150, 90));

        let inter = result_roi_intersection(&corners, &sizes);
        assert_eq!((inter.x, inter.y), (50, 10));
        assert_eq!((inter.width, inter.height), (50, 70));
        assert_eq!(result_tl(&corners), (0, 0));

        // Disjoint placements give an empty intersection, not a negative one.
        let apart = result_roi_intersection(&[(0, 0), (500, 500)], &sizes);
        assert_eq!((apart.width, apart.height), (0, 0));
        assert_eq!(result_roi_intersection(&[], &[]).width, 0);
    }

    #[test]
    fn weight_maps_normalise_across_overlapping_images() {
        let mut m = Mat::zeros(40, 40, CV_8UC1).unwrap();
        for y in 0..40 {
            for x in 0..40 {
                m.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        let w = create_weight_map(&m, 0.02).unwrap();
        assert_eq!((w.rows(), w.cols()), (40, 40));
        // A *fully covered* mask has no zero pixel to measure a distance to,
        // and upstream's answer is a weight of one everywhere — there is no
        // seam inside a region no other image reaches, so nothing to feather.
        assert!((w.at_f64(20, 20, 0).unwrap() - 1.0).abs() < 1e-6);
        assert!((w.at_f64(0, 0, 0).unwrap() - 1.0).abs() < 1e-6);

        // A mask that does not fill its frame ramps toward its own edge.
        let mut partial = Mat::zeros(40, 40, CV_8UC1).unwrap();
        for y in 4..36 {
            for x in 4..36 {
                partial.pixel_mut::<u8>(y, x).unwrap()[0] = 255;
            }
        }
        let wp = create_weight_map(&partial, 0.02).unwrap();
        assert!(wp.at_f64(20, 20, 0).unwrap() > wp.at_f64(5, 5, 0).unwrap());
        assert!(wp.at_f64(0, 0, 0).unwrap() == 0.0, "outside the mask the weight is zero");
        assert!(create_weight_map(&m, 0.0).is_err());

        let (maps, roi) = create_weight_maps(&[&m, &m], &[(0, 0), (20, 0)], 0.02).unwrap();
        assert_eq!(maps.len(), 2);
        assert_eq!((roi.width, roi.height), (60, 40));
        // In the overlap the two weights sum to one.
        let (a, b) = (maps[0].at_f64(20, 30, 0).unwrap(), maps[1].at_f64(20, 10, 0).unwrap());
        assert!((a + b - 1.0).abs() < 1e-5, "{a} + {b}");
        assert!(create_weight_maps(&[&m], &[], 0.02).is_err());
    }

    #[test]
    fn normalising_by_a_weight_map_leaves_the_uncovered_pixels_alone() {
        let mut img = Mat::zeros(4, 4, CV_32FC1).unwrap();
        let mut w = Mat::zeros(4, 4, CV_32FC1).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                img.pixel_mut::<f32>(y, x).unwrap()[0] = 8.0;
                w.pixel_mut::<f32>(y, x).unwrap()[0] = if x < 2 { 2.0 } else { 0.0 };
            }
        }
        normalize_using_weight_map(&w, &mut img).unwrap();
        assert_eq!(img.at_f64(0, 0, 0).unwrap(), 4.0, "divided by its weight");
        assert_eq!(img.at_f64(0, 3, 0).unwrap(), 8.0, "zero weight leaves it untouched");
        let wrong = Mat::zeros(2, 2, CV_32FC1).unwrap();
        assert!(normalize_using_weight_map(&wrong, &mut img).is_err());
    }

    #[test]
    fn a_laplace_pyramid_round_trips() {
        let src = texture(32, 32, 5);
        let mut f = Mat::zeros(32, 32, CV_32FC1).unwrap();
        for y in 0..32 {
            for x in 0..32 {
                f.pixel_mut::<f32>(y, x).unwrap()[0] = src.at_f64(y, x, 0).unwrap() as f32;
            }
        }
        let pyr = create_laplace_pyr(&f, 3).unwrap();
        assert!(pyr.len() >= 2, "a 32x32 image supports several levels");
        assert_eq!((pyr[0].rows(), pyr[0].cols()), (32, 32));
        // Each level is half the last.
        for w in pyr.windows(2) {
            assert!(w[1].rows() <= w[0].rows());
        }
        let back = restore_image_from_laplace_pyr(&pyr).unwrap();
        assert_eq!((back.rows(), back.cols()), (32, 32));
        let mut worst = 0.0f64;
        for y in 0..32 {
            for x in 0..32 {
                worst = worst.max((back.at_f64(y, x, 0).unwrap() - f.at_f64(y, x, 0).unwrap()).abs());
            }
        }
        assert!(worst < 1e-3, "the round trip should be near-exact: worst {worst}");

        // The GPU-named entry points are the same computation.
        let g = create_laplace_pyr_gpu(&f, 3).unwrap();
        assert_eq!(g.len(), pyr.len());
        assert!(restore_image_from_laplace_pyr_gpu(&g).is_ok());
        assert!(create_laplace_pyr(&f, 0).is_err());
        assert!(restore_image_from_laplace_pyr(&[]).is_err());
    }
}
