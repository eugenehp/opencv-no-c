//! Dense stereo matching (`modules/calib3d/src/stereobm.cpp`, `stereosgbm.cpp`).
//!
//! Given a rectified pair, every scene point appears on the same row in both
//! images, displaced horizontally by an amount inversely proportional to its
//! depth. Finding that displacement for every pixel is the whole problem, and
//! the two matchers here take opposite approaches to it.
//!
//! [`StereoBm`] compares a square block around each pixel against the same
//! block at each candidate disparity and keeps the best match. It is fast and
//! decides each pixel on its own, so it fails exactly where a block carries no
//! information — a blank wall gives the same score at every disparity.
//!
//! [`StereoSgbm`] builds the same cost volume but then *aggregates along
//! paths*, charging a penalty for disparity changes between neighbours. A pixel
//! on a blank wall inherits its neighbours' answer instead of guessing. That is
//! why SGBM fills large untextured regions and BM leaves holes in them, and why
//! SGBM costs several times as much.
//!
//! # Disparity units
//!
//! Both return `CV_16S` with the disparity multiplied by [`DISP_SCALE`], as
//! upstream does — sub-pixel accuracy in an integer matrix. Dividing by 16 is
//! the caller's job, and forgetting to is the classic way to get a depth map
//! sixteen times too near.

use cv_core::types::{Depth, MatType, CV_16SC1, CV_32FC3};
use cv_core::{cv_assert, cv_error, Code, Mat, MatLike, MatLikeMut, Rect, Result};

/// `cv::StereoMatcher::DISP_SHIFT` — the fixed-point shift of the output.
pub const DISP_SHIFT: i32 = 4;
/// `cv::StereoMatcher::DISP_SCALE` — `1 << DISP_SHIFT`.
pub const DISP_SCALE: i32 = 1 << DISP_SHIFT;

/// `cv::StereoMatcher` — what both matchers implement.
pub trait StereoMatcher {
    /// `cv::StereoMatcher::compute(left, right, disparity)`.
    ///
    /// `CV_8UC1` in, `CV_16SC1` out, in [`DISP_SCALE`] units. Pixels with no
    /// confident match get [`Self::invalid`], which is what a caller must test
    /// for rather than trusting every value in the map.
    fn compute(&self, left: &Mat, right: &Mat) -> Result<Mat>;

    /// `cv::StereoMatcher::getMinDisparity()`.
    fn get_min_disparity(&self) -> i32;
    /// `cv::StereoMatcher::setMinDisparity(minDisparity)`.
    fn set_min_disparity(&mut self, v: i32);
    /// `cv::StereoMatcher::getNumDisparities()`.
    fn get_num_disparities(&self) -> i32;
    /// `cv::StereoMatcher::setNumDisparities(numDisparities)`.
    fn set_num_disparities(&mut self, v: i32);
    /// `cv::StereoMatcher::getBlockSize()`.
    fn get_block_size(&self) -> i32;
    /// `cv::StereoMatcher::setBlockSize(blockSize)`.
    fn set_block_size(&mut self, v: i32);
    /// `cv::StereoMatcher::getSpeckleWindowSize()`.
    fn get_speckle_window_size(&self) -> i32;
    /// `cv::StereoMatcher::setSpeckleWindowSize(speckleWindowSize)`.
    fn set_speckle_window_size(&mut self, v: i32);
    /// `cv::StereoMatcher::getSpeckleRange()`.
    fn get_speckle_range(&self) -> i32;
    /// `cv::StereoMatcher::setSpeckleRange(speckleRange)`.
    fn set_speckle_range(&mut self, v: i32);
    /// `cv::StereoMatcher::getDisp12MaxDiff()`.
    fn get_disp12_max_diff(&self) -> i32;
    /// `cv::StereoMatcher::setDisp12MaxDiff(disp12MaxDiff)`.
    fn set_disp12_max_diff(&mut self, v: i32);

    /// The value written where no match was accepted.
    ///
    /// Upstream uses `(minDisparity - 1) * DISP_SCALE`, which is below every
    /// legal disparity and therefore cannot be mistaken for one.
    fn invalid(&self) -> i16 {
        ((self.get_min_disparity() - 1) * DISP_SCALE) as i16
    }
}

/// Shape and parameter checks common to both matchers.
fn check_pair(left: &Mat, right: &Mat, num_disparities: i32, block_size: i32) -> Result<()> {
    cv_assert!(left.size() == right.size(), "the two views must match in size");
    cv_assert!(left.channels() == 1 && right.channels() == 1, "stereo matching is single-channel");
    if left.depth() != Depth::U8 || right.depth() != Depth::U8 {
        return cv_error!(Code::StsUnsupportedFormat, "stereo matching requires CV_8U");
    }
    cv_assert!(
        num_disparities > 0 && num_disparities % 16 == 0,
        "numDisparities must be positive and a multiple of 16, got {num_disparities}"
    );
    cv_assert!(
        block_size >= 3 && block_size % 2 == 1,
        "blockSize must be odd and at least 3, got {block_size}"
    );
    cv_assert!(
        left.cols() > num_disparities && left.rows() >= block_size,
        "a {}x{} image is too small for {num_disparities} disparities at block {block_size}",
        left.cols(),
        left.rows()
    );
    Ok(())
}

/// `cv::StereoBM::PreFilterType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreFilterType {
    /// `cv::StereoBM::PREFILTER_NORMALIZED_RESPONSE` — subtract the local mean.
    NormalizedResponse,
    /// `cv::StereoBM::PREFILTER_XSOBEL` — the horizontal Sobel, clipped.
    /// Upstream's default, and the better one: it removes a brightness
    /// difference between the two cameras, which the mean does not.
    #[default]
    XSobel,
}

/// `cv::StereoBM` — block matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StereoBm {
    min_disparity: i32,
    num_disparities: i32,
    block_size: i32,
    speckle_window_size: i32,
    speckle_range: i32,
    disp12_max_diff: i32,
    pre_filter_type: PreFilterType,
    pre_filter_size: i32,
    pre_filter_cap: i32,
    texture_threshold: i32,
    uniqueness_ratio: i32,
    smaller_block_size: i32,
    roi1: Rect,
    roi2: Rect,
}

impl Default for StereoBm {
    /// Upstream's `StereoBM::create()` defaults.
    fn default() -> Self {
        StereoBm {
            min_disparity: 0,
            num_disparities: 64,
            block_size: 21,
            speckle_window_size: 0,
            speckle_range: 0,
            disp12_max_diff: -1,
            pre_filter_type: PreFilterType::XSobel,
            pre_filter_size: 9,
            pre_filter_cap: 31,
            texture_threshold: 10,
            uniqueness_ratio: 15,
            smaller_block_size: 0,
            roi1: Rect::new(0, 0, 0, 0),
            roi2: Rect::new(0, 0, 0, 0),
        }
    }
}

impl StereoBm {
    /// `cv::StereoBM::create(numDisparities, blockSize)`.
    #[must_use]
    pub fn create(num_disparities: i32, block_size: i32) -> Self {
        StereoBm { num_disparities, block_size, ..StereoBm::default() }
    }

    /// `cv::StereoBM::getPreFilterType()`.
    #[must_use]
    pub fn get_pre_filter_type(&self) -> PreFilterType {
        self.pre_filter_type
    }
    /// `cv::StereoBM::setPreFilterType(preFilterType)`.
    pub fn set_pre_filter_type(&mut self, v: PreFilterType) {
        self.pre_filter_type = v;
    }
    /// `cv::StereoBM::getPreFilterSize()` — the normalised-response window.
    #[must_use]
    pub fn get_pre_filter_size(&self) -> i32 {
        self.pre_filter_size
    }
    /// `cv::StereoBM::setPreFilterSize(preFilterSize)`.
    pub fn set_pre_filter_size(&mut self, v: i32) {
        self.pre_filter_size = v;
    }
    /// `cv::StereoBM::getPreFilterCap()` — the clip applied to the prefiltered
    /// image, which is what stops one specular highlight dominating a block.
    #[must_use]
    pub fn get_pre_filter_cap(&self) -> i32 {
        self.pre_filter_cap
    }
    /// `cv::StereoBM::setPreFilterCap(preFilterCap)`.
    pub fn set_pre_filter_cap(&mut self, v: i32) {
        self.pre_filter_cap = v;
    }
    /// `cv::StereoBM::getTextureThreshold()` — reject a block whose prefiltered
    /// energy is too low to distinguish one disparity from another.
    #[must_use]
    pub fn get_texture_threshold(&self) -> i32 {
        self.texture_threshold
    }
    /// `cv::StereoBM::setTextureThreshold(textureThreshold)`.
    pub fn set_texture_threshold(&mut self, v: i32) {
        self.texture_threshold = v;
    }
    /// `cv::StereoBM::getUniquenessRatio()` — reject a match the runner-up
    /// comes within this percentage of.
    #[must_use]
    pub fn get_uniqueness_ratio(&self) -> i32 {
        self.uniqueness_ratio
    }
    /// `cv::StereoBM::setUniquenessRatio(uniquenessRatio)`.
    pub fn set_uniqueness_ratio(&mut self, v: i32) {
        self.uniqueness_ratio = v;
    }
    /// `cv::StereoBM::getSmallerBlockSize()`.
    #[must_use]
    pub fn get_smaller_block_size(&self) -> i32 {
        self.smaller_block_size
    }
    /// `cv::StereoBM::setSmallerBlockSize(blockSize)`.
    pub fn set_smaller_block_size(&mut self, v: i32) {
        self.smaller_block_size = v;
    }
    /// `cv::StereoBM::getROI1()`.
    #[must_use]
    pub fn get_roi1(&self) -> Rect {
        self.roi1
    }
    /// `cv::StereoBM::setROI1(roi1)`.
    pub fn set_roi1(&mut self, v: Rect) {
        self.roi1 = v;
    }
    /// `cv::StereoBM::getROI2()`.
    #[must_use]
    pub fn get_roi2(&self) -> Rect {
        self.roi2
    }
    /// `cv::StereoBM::setROI2(roi2)`.
    pub fn set_roi2(&mut self, v: Rect) {
        self.roi2 = v;
    }

    /// The prefiltered image as `i32`, in `-pre_filter_cap ..= pre_filter_cap`.
    fn prefilter(&self, m: &Mat) -> Result<Vec<i32>> {
        let (h, w) = (m.rows(), m.cols());
        let cap = self.pre_filter_cap.max(1);
        let mut out = vec![0i32; (h * w) as usize];
        match self.pre_filter_type {
            PreFilterType::XSobel => {
                for y in 0..h {
                    for x in 0..w {
                        let g = |dy: i32, dx: i32| -> Result<i32> {
                            Ok(m.at_f64((y + dy).clamp(0, h - 1), (x + dx).clamp(0, w - 1), 0)?
                                as i32)
                        };
                        // The 3x3 horizontal Sobel. Vertical structure carries
                        // no disparity information on a rectified pair, so only
                        // the x derivative is worth keeping.
                        let v = (g(-1, 1)? - g(-1, -1)?)
                            + 2 * (g(0, 1)? - g(0, -1)?)
                            + (g(1, 1)? - g(1, -1)?);
                        out[(y * w + x) as usize] = v.clamp(-cap, cap);
                    }
                }
            }
            PreFilterType::NormalizedResponse => {
                let r = (self.pre_filter_size.max(5) / 2).max(1);
                for y in 0..h {
                    for x in 0..w {
                        let mut sum = 0i32;
                        let mut n = 0i32;
                        for dy in -r..=r {
                            for dx in -r..=r {
                                sum += m.at_f64(
                                    (y + dy).clamp(0, h - 1),
                                    (x + dx).clamp(0, w - 1),
                                    0,
                                )? as i32;
                                n += 1;
                            }
                        }
                        let c = m.at_f64(y, x, 0)? as i32;
                        out[(y * w + x) as usize] = (c - sum / n.max(1)).clamp(-cap, cap);
                    }
                }
            }
        }
        Ok(out)
    }
}

/// `cv::StereoBM::create(numDisparities, blockSize)`.
#[must_use]
pub fn create_stereo_bm(num_disparities: i32, block_size: i32) -> StereoBm {
    StereoBm::create(num_disparities, block_size)
}

impl StereoMatcher for StereoBm {
    fn compute(&self, left: &Mat, right: &Mat) -> Result<Mat> {
        check_pair(left, right, self.num_disparities, self.block_size)?;
        let (h, w) = (left.rows(), left.cols());
        let (lp, rp) = (self.prefilter(left)?, self.prefilter(right)?);
        let r = self.block_size / 2;
        let invalid = self.invalid();
        let mut disp = vec![invalid; (h * w) as usize];

        // Upstream's `getValidDisparityROI`. Outside this band either the
        // disparity search runs off the edge of the right image or the
        // correlation window runs off the edge of the left one, so upstream
        // marks the pixel filtered rather than reporting the best of a
        // truncated search. It is wider than it looks: half a window *plus*
        // the widest disparity on the left.
        let min_d = self.get_min_disparity();
        let max_d = min_d + self.num_disparities - 1;
        for y in r..h - r {
            for x in (max_d + r).max(r)..(w - min_d - r).min(w - r) {
                // Texture: a block with no variation matches everywhere, so its
                // best disparity means nothing. Rejecting it here is what keeps
                // a blank wall out of the map instead of filling it with noise.
                let mut energy = 0i32;
                for dy in -r..=r {
                    for dx in -r..=r {
                        energy += lp[((y + dy) * w + x + dx) as usize].abs();
                    }
                }
                if energy < self.texture_threshold * self.block_size * self.block_size / 8 {
                    continue;
                }

                let mut best = (i32::MAX, 0i32);
                let mut second = i32::MAX;
                let mut costs = vec![i32::MAX; self.num_disparities as usize];
                for d in 0..self.num_disparities {
                    let dd = self.min_disparity + d;
                    if x - r - dd < 0 || x + r - dd >= w {
                        continue;
                    }
                    let mut sad = 0i32;
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let a = lp[((y + dy) * w + x + dx) as usize];
                            let b = rp[((y + dy) * w + x + dx - dd) as usize];
                            sad += (a - b).abs();
                        }
                    }
                    costs[d as usize] = sad;
                    if sad < best.0 {
                        second = best.0;
                        best = (sad, d);
                    } else if sad < second {
                        second = sad;
                    }
                }
                if best.0 == i32::MAX {
                    continue;
                }
                // Uniqueness: if the runner-up is nearly as good, the match is
                // ambiguous. Comparing against the *global* runner-up rather
                // than the neighbouring disparities is upstream's rule, and it
                // is what rejects a repeating texture.
                if self.uniqueness_ratio > 0 && second != i32::MAX {
                    let margin = best.0 + best.0 * self.uniqueness_ratio / 100;
                    let ambiguous = costs.iter().enumerate().any(|(i, &c)| {
                        c <= margin && (i as i32 - best.1).abs() > 1 && c != i32::MAX
                    });
                    if ambiguous {
                        continue;
                    }
                }

                // Sub-pixel by fitting a parabola through the winning cost and
                // its two neighbours — the reason the output is in 1/16ths.
                let d = best.1;
                let sub = if d > 0
                    && d + 1 < self.num_disparities
                    && costs[(d - 1) as usize] != i32::MAX
                    && costs[(d + 1) as usize] != i32::MAX
                {
                    let (c0, c1, c2) =
                        (costs[(d - 1) as usize], costs[d as usize], costs[(d + 1) as usize]);
                    let denom = c0 + c2 - 2 * c1;
                    if denom == 0 {
                        0.0
                    } else {
                        f64::from(c0 - c2) / (2.0 * f64::from(denom))
                    }
                } else {
                    0.0
                };
                let value = (f64::from(self.min_disparity + d) + sub) * f64::from(DISP_SCALE);
                disp[(y * w + x) as usize] = value.round() as i16;
            }
        }

        let mut out = Mat::zeros(h, w, CV_16SC1)?;
        for y in 0..h {
            for x in 0..w {
                out.pixel_mut::<i16>(y, x)?[0] = disp[(y * w + x) as usize];
            }
        }
        if self.speckle_window_size > 0 {
            filter_speckles(&mut out, f64::from(invalid), self.speckle_window_size, self.speckle_range)?;
        }
        Ok(out)
    }

    fn get_min_disparity(&self) -> i32 {
        self.min_disparity
    }
    fn set_min_disparity(&mut self, v: i32) {
        self.min_disparity = v;
    }
    fn get_num_disparities(&self) -> i32 {
        self.num_disparities
    }
    fn set_num_disparities(&mut self, v: i32) {
        self.num_disparities = v;
    }
    fn get_block_size(&self) -> i32 {
        self.block_size
    }
    fn set_block_size(&mut self, v: i32) {
        self.block_size = v;
    }
    fn get_speckle_window_size(&self) -> i32 {
        self.speckle_window_size
    }
    fn set_speckle_window_size(&mut self, v: i32) {
        self.speckle_window_size = v;
    }
    fn get_speckle_range(&self) -> i32 {
        self.speckle_range
    }
    fn set_speckle_range(&mut self, v: i32) {
        self.speckle_range = v;
    }
    fn get_disp12_max_diff(&self) -> i32 {
        self.disp12_max_diff
    }
    fn set_disp12_max_diff(&mut self, v: i32) {
        self.disp12_max_diff = v;
    }
}

/// `cv::StereoSGBM::Mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SgbmMode {
    /// `cv::StereoSGBM::MODE_SGBM` — five paths, upstream's default.
    #[default]
    Sgbm,
    /// `cv::StereoSGBM::MODE_HH` — all eight paths, over the whole image at
    /// once. Better and much hungrier for memory.
    Hh,
    /// `cv::StereoSGBM::MODE_SGBM_3WAY` — three paths; faster, coarser.
    Sgbm3Way,
    /// `cv::StereoSGBM::MODE_HH4` — the eight-path variant in four-row blocks.
    Hh4,
}

impl SgbmMode {
    /// The path directions this mode aggregates along, as `(dy, dx)`.
    ///
    /// A path is a 1-D dynamic program; running several and summing is what
    /// approximates the 2-D smoothness a full graph cut would enforce, at a
    /// fraction of the cost. More paths means fewer streaking artefacts.
    fn paths(self) -> &'static [(i32, i32)] {
        const ALL: [(i32, i32); 8] = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
        ];
        const FIVE: [(i32, i32); 5] = [(0, 1), (0, -1), (1, 0), (1, 1), (1, -1)];
        const THREE: [(i32, i32); 3] = [(0, 1), (1, 0), (1, 1)];
        match self {
            SgbmMode::Sgbm => &FIVE,
            SgbmMode::Sgbm3Way => &THREE,
            SgbmMode::Hh | SgbmMode::Hh4 => &ALL,
        }
    }
}

/// `cv::StereoSGBM` — semi-global block matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StereoSgbm {
    min_disparity: i32,
    num_disparities: i32,
    block_size: i32,
    p1: i32,
    p2: i32,
    disp12_max_diff: i32,
    pre_filter_cap: i32,
    uniqueness_ratio: i32,
    speckle_window_size: i32,
    speckle_range: i32,
    mode: SgbmMode,
}

impl Default for StereoSgbm {
    /// Upstream's `StereoSGBM::create()` defaults.
    fn default() -> Self {
        StereoSgbm {
            min_disparity: 0,
            num_disparities: 64,
            block_size: 3,
            p1: 0,
            p2: 0,
            disp12_max_diff: -1,
            pre_filter_cap: 0,
            uniqueness_ratio: 0,
            speckle_window_size: 0,
            speckle_range: 0,
            mode: SgbmMode::Sgbm,
        }
    }
}

impl StereoSgbm {
    /// `cv::StereoSGBM::create(minDisparity, numDisparities, blockSize, P1, P2, ...)`.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn create(
        min_disparity: i32,
        num_disparities: i32,
        block_size: i32,
        p1: i32,
        p2: i32,
        mode: SgbmMode,
    ) -> Self {
        StereoSgbm {
            min_disparity,
            num_disparities,
            block_size,
            p1,
            p2,
            mode,
            ..StereoSgbm::default()
        }
    }

    /// `cv::StereoSGBM::getP1()` — the penalty for a one-step disparity change.
    #[must_use]
    pub fn get_p1(&self) -> i32 {
        self.p1
    }
    /// `cv::StereoSGBM::setP1(P1)`.
    pub fn set_p1(&mut self, v: i32) {
        self.p1 = v;
    }
    /// `cv::StereoSGBM::getP2()` — the penalty for a larger one.
    ///
    /// `P2` must exceed `P1`: the small penalty lets a surface slant, the large
    /// one is what a real depth discontinuity has to pay. Setting them equal
    /// makes every edge as cheap as a gentle slope and the result is mush.
    #[must_use]
    pub fn get_p2(&self) -> i32 {
        self.p2
    }
    /// `cv::StereoSGBM::setP2(P2)`.
    pub fn set_p2(&mut self, v: i32) {
        self.p2 = v;
    }
    /// `cv::StereoSGBM::getPreFilterCap()`.
    #[must_use]
    pub fn get_pre_filter_cap(&self) -> i32 {
        self.pre_filter_cap
    }
    /// `cv::StereoSGBM::setPreFilterCap(preFilterCap)`.
    pub fn set_pre_filter_cap(&mut self, v: i32) {
        self.pre_filter_cap = v;
    }
    /// `cv::StereoSGBM::getUniquenessRatio()`.
    #[must_use]
    pub fn get_uniqueness_ratio(&self) -> i32 {
        self.uniqueness_ratio
    }
    /// `cv::StereoSGBM::setUniquenessRatio(uniquenessRatio)`.
    pub fn set_uniqueness_ratio(&mut self, v: i32) {
        self.uniqueness_ratio = v;
    }
    /// `cv::StereoSGBM::getMode()`.
    #[must_use]
    pub fn get_mode(&self) -> SgbmMode {
        self.mode
    }
    /// `cv::StereoSGBM::setMode(mode)`.
    pub fn set_mode(&mut self, v: SgbmMode) {
        self.mode = v;
    }

    /// The effective penalties, filling in upstream's defaults for zero.
    fn penalties(&self) -> (i32, i32) {
        let b = self.block_size.max(1);
        let p1 = if self.p1 > 0 { self.p1 } else { 8 * b * b };
        let p2 = if self.p2 > 0 { self.p2 } else { 32 * b * b };
        (p1, p2.max(p1 + 1))
    }
}

/// `cv::StereoSGBM::create(minDisparity, numDisparities, blockSize, ...)`.
#[must_use]
pub fn create_stereo_sgbm(
    min_disparity: i32,
    num_disparities: i32,
    block_size: i32,
) -> StereoSgbm {
    StereoSgbm { min_disparity, num_disparities, block_size, ..StereoSgbm::default() }
}

impl StereoMatcher for StereoSgbm {
    fn compute(&self, left: &Mat, right: &Mat) -> Result<Mat> {
        check_pair(left, right, self.num_disparities, self.block_size.max(3))?;
        let (h, w) = (left.rows(), left.cols());
        let nd = self.num_disparities as usize;
        let (p1, p2) = self.penalties();
        let cap = if self.pre_filter_cap > 0 { self.pre_filter_cap } else { 63 };

        // Birchfield-Tomasi style matching cost on the x-gradient. Using the
        // gradient rather than the intensity is what makes it tolerate a gain
        // difference between the two cameras.
        let grad = |m: &Mat| -> Result<Vec<i32>> {
            let mut g = vec![0i32; (h * w) as usize];
            for y in 0..h {
                for x in 0..w {
                    let a = m.at_f64(y, (x - 1).clamp(0, w - 1), 0)? as i32;
                    let b = m.at_f64(y, (x + 1).clamp(0, w - 1), 0)? as i32;
                    g[(y * w + x) as usize] = (b - a).clamp(-cap, cap);
                }
            }
            Ok(g)
        };
        let (lg, rg) = (grad(left)?, grad(right)?);

        // The raw cost volume, block-aggregated.
        let r = self.block_size.max(3) / 2;
        let mut cost = vec![0i32; (h * w) as usize * nd];
        for y in 0..h {
            for x in 0..w {
                for d in 0..nd {
                    let dd = self.min_disparity + d as i32;
                    let mut c = 0i32;
                    let mut n = 0i32;
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let (sy, sx) = ((y + dy).clamp(0, h - 1), (x + dx).clamp(0, w - 1));
                            let rx = sx - dd;
                            if rx < 0 || rx >= w {
                                continue;
                            }
                            c += (lg[(sy * w + sx) as usize] - rg[(sy * w + rx) as usize]).abs();
                            n += 1;
                        }
                    }
                    cost[((y * w + x) as usize) * nd + d] =
                        if n == 0 { cap * 4 } else { c / n };
                }
            }
        }

        // Path aggregation. Each path is a scan whose cost at a pixel is the
        // matching cost plus the cheapest way to arrive from the previous pixel
        // — same disparity free, one step costs P1, anything else P2.
        let mut agg = vec![0i32; (h * w) as usize * nd];
        for &(dy, dx) in self.mode.paths() {
            let mut prev = vec![0i32; nd];
            // Walk the image so the path's predecessor is always already done.
            let ys: Vec<i32> = if dy > 0 { (0..h).collect() } else { (0..h).rev().collect() };
            let xs: Vec<i32> = if dx > 0 { (0..w).collect() } else { (0..w).rev().collect() };
            for &y in &ys {
                for &x in &xs {
                    let (py, px) = (y - dy, x - dx);
                    let base = ((y * w + x) as usize) * nd;
                    let start = py < 0 || px < 0 || py >= h || px >= w;
                    if start {
                        for d in 0..nd {
                            let v = cost[base + d];
                            agg[base + d] += v;
                            prev[d] = v;
                        }
                        continue;
                    }
                    let min_prev = *prev.iter().min().unwrap_or(&0);
                    let mut cur = vec![0i32; nd];
                    for d in 0..nd {
                        let same = prev[d];
                        let step = prev[d.saturating_sub(1)].min(prev[(d + 1).min(nd - 1)]) + p1;
                        let jump = min_prev + p2;
                        // Subtracting the running minimum keeps the numbers
                        // bounded; without it the path cost grows without limit
                        // across a wide image and overflows.
                        cur[d] = cost[base + d] + same.min(step).min(jump) - min_prev;
                        agg[base + d] += cur[d];
                    }
                    prev = cur;
                }
            }
        }

        // Winner-take-all over the aggregated volume, with the same uniqueness
        // rule and parabola fit block matching uses.
        let invalid = self.invalid();
        let mut disp = vec![invalid; (h * w) as usize];
        // Upstream's `minX1`/`maxX1`. A pixel closer to the left edge than the
        // widest disparity has nowhere in the right image to be searched for,
        // so upstream leaves it invalid rather than reporting the best of a
        // truncated search. Skipping this returns a confident-looking number
        // for the first `numDisparities` columns of every row.
        let min_x1 = (self.min_disparity + self.num_disparities).max(0);
        let max_x1 = w + self.min_disparity.min(0);
        for y in 0..h {
            for x in min_x1..max_x1 {
                let base = ((y * w + x) as usize) * nd;
                let (mut best, mut bd) = (i32::MAX, 0usize);
                for d in 0..nd {
                    if agg[base + d] < best {
                        best = agg[base + d];
                        bd = d;
                    }
                }
                if self.uniqueness_ratio > 0 {
                    let margin = best + best * self.uniqueness_ratio / 100;
                    let ambiguous = (0..nd)
                        .any(|d| agg[base + d] <= margin && (d as i32 - bd as i32).abs() > 1);
                    if ambiguous {
                        continue;
                    }
                }
                let sub = if bd > 0 && bd + 1 < nd {
                    let (c0, c1, c2) = (agg[base + bd - 1], agg[base + bd], agg[base + bd + 1]);
                    let denom = c0 + c2 - 2 * c1;
                    if denom == 0 {
                        0.0
                    } else {
                        f64::from(c0 - c2) / (2.0 * f64::from(denom))
                    }
                } else {
                    0.0
                };
                let value =
                    (f64::from(self.min_disparity + bd as i32) + sub) * f64::from(DISP_SCALE);
                disp[(y * w + x) as usize] = value.round() as i16;
            }
        }

        let mut out = Mat::zeros(h, w, CV_16SC1)?;
        for y in 0..h {
            for x in 0..w {
                out.pixel_mut::<i16>(y, x)?[0] = disp[(y * w + x) as usize];
            }
        }
        if self.speckle_window_size > 0 {
            filter_speckles(&mut out, f64::from(invalid), self.speckle_window_size, self.speckle_range)?;
        }
        Ok(out)
    }

    fn get_min_disparity(&self) -> i32 {
        self.min_disparity
    }
    fn set_min_disparity(&mut self, v: i32) {
        self.min_disparity = v;
    }
    fn get_num_disparities(&self) -> i32 {
        self.num_disparities
    }
    fn set_num_disparities(&mut self, v: i32) {
        self.num_disparities = v;
    }
    fn get_block_size(&self) -> i32 {
        self.block_size
    }
    fn set_block_size(&mut self, v: i32) {
        self.block_size = v;
    }
    fn get_speckle_window_size(&self) -> i32 {
        self.speckle_window_size
    }
    fn set_speckle_window_size(&mut self, v: i32) {
        self.speckle_window_size = v;
    }
    fn get_speckle_range(&self) -> i32 {
        self.speckle_range
    }
    fn set_speckle_range(&mut self, v: i32) {
        self.speckle_range = v;
    }
    fn get_disp12_max_diff(&self) -> i32 {
        self.disp12_max_diff
    }
    fn set_disp12_max_diff(&mut self, v: i32) {
        self.disp12_max_diff = v;
    }
}

/// `cv::filterSpeckles(img, newVal, maxSpeckleSize, maxDiff, buf)`.
///
/// Floods each connected region whose neighbouring disparities differ by no
/// more than `max_diff`, and blanks the region if it is smaller than
/// `max_speckle_size`. A stereo matcher produces small islands at the wrong
/// depth wherever a match was a coincidence; they are the most visible defect
/// in a disparity map and the cheapest to remove.
pub fn filter_speckles<M: MatLikeMut + ?Sized>(
    img: &mut M,
    new_val: f64,
    max_speckle_size: i32,
    max_diff: i32,
) -> Result<i32> {
    cv_assert!(img.channels() == 1, "filterSpeckles takes a single-channel map");
    if img.depth() != Depth::S16 {
        return cv_error!(Code::StsUnsupportedFormat, "filterSpeckles requires CV_16S");
    }
    let (h, w) = (img.rows(), img.cols());
    let mut label = vec![0i32; (h * w) as usize];
    let mut removed = 0;
    let mut next = 0i32;
    let new_val = new_val as i16;

    for y0 in 0..h {
        for x0 in 0..w {
            if label[(y0 * w + x0) as usize] != 0 {
                continue;
            }
            let seed = img.at_f64(y0, x0, 0)? as i16;
            if seed == new_val {
                continue;
            }
            next += 1;
            let mut region = vec![(y0, x0)];
            label[(y0 * w + x0) as usize] = next;
            let mut i = 0;
            while i < region.len() {
                let (y, x) = region[i];
                i += 1;
                let v = img.at_f64(y, x, 0)? as i32;
                for (dy, dx) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                    let (ny, nx) = (y + dy, x + dx);
                    if ny < 0 || nx < 0 || ny >= h || nx >= w {
                        continue;
                    }
                    if label[(ny * w + nx) as usize] != 0 {
                        continue;
                    }
                    let nv = img.at_f64(ny, nx, 0)? as i32;
                    if nv as i16 == new_val || (nv - v).abs() > max_diff {
                        continue;
                    }
                    label[(ny * w + nx) as usize] = next;
                    region.push((ny, nx));
                }
            }
            if (region.len() as i32) <= max_speckle_size {
                removed += region.len() as i32;
                for (y, x) in region {
                    img.pixel_mut::<i16>(y, x)?[0] = new_val;
                }
            }
        }
    }
    Ok(removed)
}

/// `cv::validateDisparity(disparity, cost, minDisparity, numberOfDisparities, disp12MaxDiff)`.
///
/// A left-right consistency check: a pixel's disparity is trusted only if the
/// pixel it points at in the right image points back. That catches occlusions,
/// which no single-image test can — an occluded pixel has no true match, so
/// whatever it found is wrong however confident the cost looked.
pub fn validate_disparity<M: MatLikeMut + ?Sized>(
    disparity: &mut M,
    min_disparity: i32,
    number_of_disparities: i32,
    disp12_max_diff: i32,
) -> Result<()> {
    cv_assert!(disparity.channels() == 1, "validateDisparity takes a single-channel map");
    if disparity.depth() != Depth::S16 {
        return cv_error!(Code::StsUnsupportedFormat, "validateDisparity requires CV_16S");
    }
    if disp12_max_diff < 0 {
        return Ok(());
    }
    let (h, w) = (disparity.rows(), disparity.cols());
    let invalid = ((min_disparity - 1) * DISP_SCALE) as i16;
    let snapshot: Vec<i16> = {
        let mut v = Vec::with_capacity((h * w) as usize);
        for y in 0..h {
            for x in 0..w {
                v.push(disparity.at_f64(y, x, 0)? as i16);
            }
        }
        v
    };

    for y in 0..h {
        // The right-image disparity implied by this row's left-image matches.
        let mut rdisp = vec![i32::MIN; w as usize];
        for x in 0..w {
            let d = i32::from(snapshot[(y * w + x) as usize]);
            if d == i32::from(invalid) {
                continue;
            }
            let rx = x - (d + DISP_SCALE / 2) / DISP_SCALE;
            if rx < 0 || rx >= w {
                continue;
            }
            // The nearest surface wins the pixel; that is what occlusion means.
            if d > rdisp[rx as usize] {
                rdisp[rx as usize] = d;
            }
        }
        for x in 0..w {
            let d = i32::from(snapshot[(y * w + x) as usize]);
            if d == i32::from(invalid) {
                continue;
            }
            let rx = x - (d + DISP_SCALE / 2) / DISP_SCALE;
            let ok = (0..w).contains(&rx)
                && rdisp[rx as usize] != i32::MIN
                && (rdisp[rx as usize] - d).abs() <= disp12_max_diff * DISP_SCALE
                && d >= min_disparity * DISP_SCALE
                && d < (min_disparity + number_of_disparities) * DISP_SCALE;
            if !ok {
                disparity.pixel_mut::<i16>(y, x)?[0] = invalid;
            }
        }
    }
    Ok(())
}

/// `cv::getValidDisparityROI(roi1, roi2, minDisparity, numberOfDisparities, blockSize)`.
///
/// The rectangle where a disparity can actually be computed: a block needs its
/// full support on both sides, and the search shifts it by up to
/// `minDisparity + numberOfDisparities` columns. Outside this the map holds
/// whatever the border produced, not a measurement.
#[must_use]
pub fn get_valid_disparity_roi(
    roi1: Rect,
    roi2: Rect,
    min_disparity: i32,
    number_of_disparities: i32,
    block_size: i32,
) -> Rect {
    let b = (block_size - 1) / 2;
    let min_d = min_disparity.min(0);
    let max_d = (min_disparity + number_of_disparities - 1).max(0);
    let x = roi1.x.max(roi2.x + max_d) + b;
    let y = roi1.y.max(roi2.y) + b;
    let right = (roi1.x + roi1.width).min(roi2.x + roi2.width + min_d) - b;
    let bottom = (roi1.y + roi1.height).min(roi2.y + roi2.height) - b;
    Rect::new(x, y, (right - x).max(0), (bottom - y).max(0))
}

/// `cv::reprojectImageTo3D(disparity, _3dImage, Q, handleMissingValues, ddepth)`.
///
/// Turns a disparity map into points, through the `Q` matrix
/// [`crate::stereo::stereo_rectify`] produces. With `handle_missing_values`,
/// pixels at the minimum disparity — the ones the matcher gave up on — are set
/// to a very large `Z` rather than to a plausible-looking near point.
pub fn reproject_image_to_3d<M: MatLike + ?Sized>(
    disparity: &M,
    q: &[[f64; 4]; 4],
    handle_missing_values: bool,
) -> Result<Mat> {
    cv_assert!(disparity.channels() == 1, "reprojectImageTo3D takes a single-channel map");
    let (h, w) = (disparity.rows(), disparity.cols());
    let scale = if disparity.depth() == Depth::S16 { 1.0 / f64::from(DISP_SCALE) } else { 1.0 };
    // Upstream's marker for a point it could not place.
    const BIG_Z: f32 = 10_000.0;

    let mut min_d = f64::INFINITY;
    for y in 0..h {
        for x in 0..w {
            min_d = min_d.min(disparity.at_f64(y, x, 0)?);
        }
    }

    let mut out = Mat::zeros(h, w, CV_32FC3)?;
    for y in 0..h {
        for x in 0..w {
            let raw = disparity.at_f64(y, x, 0)?;
            let px = out.pixel_mut::<f32>(y, x)?;
            if handle_missing_values && raw <= min_d {
                px[0] = BIG_Z;
                px[1] = BIG_Z;
                px[2] = BIG_Z;
                continue;
            }
            let d = raw * scale;
            let (fx, fy) = (f64::from(x), f64::from(y));
            let hx = q[0][0] * fx + q[0][1] * fy + q[0][2] * d + q[0][3];
            let hy = q[1][0] * fx + q[1][1] * fy + q[1][2] * d + q[1][3];
            let hz = q[2][0] * fx + q[2][1] * fy + q[2][2] * d + q[2][3];
            let hw = q[3][0] * fx + q[3][1] * fy + q[3][2] * d + q[3][3];
            if hw.abs() < 1e-12 {
                px[0] = BIG_Z;
                px[1] = BIG_Z;
                px[2] = BIG_Z;
            } else {
                px[0] = (hx / hw) as f32;
                px[1] = (hy / hw) as f32;
                px[2] = (hz / hw) as f32;
            }
        }
    }
    let _ = MatType::new(Depth::F32, 3);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::CV_8UC1;

    /// A left/right pair with a known disparity of `shift`.
    ///
    /// Disparity is defined by `left[x] == right[x - d]`, so the *right* image
    /// is the one sampled further along the texture. Building it the other way
    /// round makes the true disparity negative and every matcher returns
    /// plausible nonsense — which is exactly what happened the first time.
    fn pair(w: i32, h: i32, shift: i32) -> (Mat, Mat) {
        let mut state = 0x2468_ace0u32;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 200 + 28) as u8
        };
        // One row of texture, repeated down the image so every block matches.
        let row: Vec<u8> = (0..w + shift).map(|_| rnd()).collect();
        let mut l = Mat::zeros(h, w, CV_8UC1).unwrap();
        let mut r = Mat::zeros(h, w, CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                l.pixel_mut::<u8>(y, x).unwrap()[0] = row[x as usize];
                r.pixel_mut::<u8>(y, x).unwrap()[0] = row[(x + shift) as usize];
            }
        }
        (l, r)
    }

    /// The most common disparity in the interior, in whole pixels.
    fn modal_disparity(d: &Mat, invalid: i16) -> Option<i32> {
        let mut counts = std::collections::BTreeMap::new();
        for y in d.rows() / 4..d.rows() * 3 / 4 {
            for x in d.cols() / 3..d.cols() * 3 / 4 {
                let v = d.at_f64(y, x, 0).unwrap() as i16;
                if v == invalid {
                    continue;
                }
                *counts.entry(i32::from(v) / DISP_SCALE).or_insert(0usize) += 1;
            }
        }
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(d, _)| d)
    }

    #[test]
    fn block_matching_recovers_a_known_shift() {
        let (l, r) = pair(120, 60, 12);
        let bm = StereoBm::create(32, 15);
        let d = bm.compute(&l, &r).unwrap();
        assert_eq!(d.mat_type(), CV_16SC1, "the output is CV_16S");
        assert_eq!((d.rows(), d.cols()), (60, 120));
        assert_eq!(modal_disparity(&d, bm.invalid()), Some(12));
    }

    #[test]
    fn semi_global_matching_recovers_the_same_shift() {
        let (l, r) = pair(120, 60, 12);
        let sgbm = create_stereo_sgbm(0, 32, 5);
        let d = sgbm.compute(&l, &r).unwrap();
        assert_eq!(modal_disparity(&d, sgbm.invalid()), Some(12));
    }

    #[test]
    fn every_sgbm_mode_finds_the_shift() {
        let (l, r) = pair(96, 48, 8);
        for mode in [SgbmMode::Sgbm, SgbmMode::Hh, SgbmMode::Sgbm3Way, SgbmMode::Hh4] {
            let mut s = create_stereo_sgbm(0, 16, 5);
            s.set_mode(mode);
            assert_eq!(s.get_mode(), mode);
            let d = s.compute(&l, &r).unwrap();
            assert_eq!(modal_disparity(&d, s.invalid()), Some(8), "{mode:?}");
        }
        assert_eq!(SgbmMode::default(), SgbmMode::Sgbm);
        // The path count is what distinguishes the modes.
        assert_eq!(SgbmMode::Sgbm.paths().len(), 5);
        assert_eq!(SgbmMode::Hh.paths().len(), 8);
        assert_eq!(SgbmMode::Sgbm3Way.paths().len(), 3);
    }

    #[test]
    fn an_untextured_region_is_rejected_rather_than_guessed() {
        // A flat grey pair: every disparity matches equally well, so a matcher
        // that answers at all is answering noise.
        let flat = |v: u8| {
            let mut m = Mat::zeros(48, 96, CV_8UC1).unwrap();
            for y in 0..48 {
                for x in 0..96 {
                    m.pixel_mut::<u8>(y, x).unwrap()[0] = v;
                }
            }
            m
        };
        let bm = StereoBm::create(16, 15);
        let d = bm.compute(&flat(128), &flat(128)).unwrap();
        let invalid = bm.invalid();
        let answered = (0..48)
            .flat_map(|y| (0..96).map(move |x| (y, x)))
            .filter(|&(y, x)| d.at_f64(y, x, 0).unwrap() as i16 != invalid)
            .count();
        assert_eq!(answered, 0, "the texture threshold should reject every pixel");
    }

    #[test]
    fn the_matchers_check_their_inputs() {
        let (l, r) = pair(96, 48, 8);
        let small = Mat::zeros(48, 40, CV_8UC1).unwrap();
        for m in [
            Box::new(StereoBm::create(16, 15)) as Box<dyn StereoMatcher>,
            Box::new(create_stereo_sgbm(0, 16, 5)) as Box<dyn StereoMatcher>,
        ] {
            assert!(m.compute(&l, &small).is_err(), "mismatched sizes");
            let f = Mat::zeros(48, 96, cv_core::types::CV_32FC1).unwrap();
            assert!(m.compute(&f, &f).is_err(), "not CV_8U");
        }
        // numDisparities must be a positive multiple of 16.
        assert!(StereoBm::create(15, 15).compute(&l, &r).is_err());
        assert!(StereoBm::create(0, 15).compute(&l, &r).is_err());
        // blockSize must be odd and at least 3.
        assert!(StereoBm::create(16, 14).compute(&l, &r).is_err());
        assert!(StereoBm::create(16, 1).compute(&l, &r).is_err());
        // ...and the image must be wider than the search range.
        let narrow = Mat::zeros(48, 16, CV_8UC1).unwrap();
        assert!(StereoBm::create(16, 5).compute(&narrow, &narrow).is_err());
    }

    #[test]
    fn the_shared_accessors_round_trip_on_both_matchers() {
        for m in [
            Box::new(StereoBm::default()) as Box<dyn StereoMatcher>,
            Box::new(StereoSgbm::default()) as Box<dyn StereoMatcher>,
        ] {
            let mut m = m;
            m.set_min_disparity(3);
            m.set_num_disparities(48);
            m.set_block_size(9);
            m.set_speckle_window_size(100);
            m.set_speckle_range(2);
            m.set_disp12_max_diff(5);
            assert_eq!(m.get_min_disparity(), 3);
            assert_eq!(m.get_num_disparities(), 48);
            assert_eq!(m.get_block_size(), 9);
            assert_eq!(m.get_speckle_window_size(), 100);
            assert_eq!(m.get_speckle_range(), 2);
            assert_eq!(m.get_disp12_max_diff(), 5);
            // The invalid marker follows minDisparity.
            assert_eq!(m.invalid(), (2 * DISP_SCALE) as i16);
        }
    }

    #[test]
    fn the_block_matcher_accessors_round_trip() {
        let mut b = StereoBm::default();
        // Upstream's defaults.
        assert_eq!(b.get_pre_filter_type(), PreFilterType::XSobel);
        assert_eq!(b.get_pre_filter_size(), 9);
        assert_eq!(b.get_pre_filter_cap(), 31);
        assert_eq!(b.get_texture_threshold(), 10);
        assert_eq!(b.get_uniqueness_ratio(), 15);
        assert_eq!(b.get_smaller_block_size(), 0);
        assert_eq!(b.get_block_size(), 21);
        assert_eq!(b.get_num_disparities(), 64);

        b.set_pre_filter_type(PreFilterType::NormalizedResponse);
        b.set_pre_filter_size(11);
        b.set_pre_filter_cap(20);
        b.set_texture_threshold(5);
        b.set_uniqueness_ratio(7);
        b.set_smaller_block_size(3);
        b.set_roi1(Rect::new(1, 2, 3, 4));
        b.set_roi2(Rect::new(5, 6, 7, 8));
        assert_eq!(b.get_pre_filter_type(), PreFilterType::NormalizedResponse);
        assert_eq!(b.get_pre_filter_size(), 11);
        assert_eq!(b.get_pre_filter_cap(), 20);
        assert_eq!(b.get_texture_threshold(), 5);
        assert_eq!(b.get_uniqueness_ratio(), 7);
        assert_eq!(b.get_smaller_block_size(), 3);
        assert_eq!(b.get_roi1(), Rect::new(1, 2, 3, 4));
        assert_eq!(b.get_roi2(), Rect::new(5, 6, 7, 8));
    }

    #[test]
    fn both_prefilters_still_find_the_shift() {
        let (l, r) = pair(120, 48, 10);
        for t in [PreFilterType::XSobel, PreFilterType::NormalizedResponse] {
            let mut b = StereoBm::create(32, 15);
            b.set_pre_filter_type(t);
            b.set_texture_threshold(3);
            let d = b.compute(&l, &r).unwrap();
            assert_eq!(modal_disparity(&d, b.invalid()), Some(10), "{t:?}");
        }
    }

    #[test]
    fn the_sgbm_accessors_round_trip_and_penalties_stay_ordered() {
        let mut s = StereoSgbm::default();
        assert_eq!((s.get_p1(), s.get_p2()), (0, 0), "zero means 'derive from blockSize'");
        // Derived, and P2 always exceeds P1 — the ordering is what makes an
        // edge cost more than a slope.
        let (p1, p2) = s.penalties();
        assert!(p2 > p1, "{p2} must exceed {p1}");

        s.set_p1(100);
        s.set_p2(50);
        let (p1, p2) = s.penalties();
        assert_eq!(p1, 100);
        assert!(p2 > p1, "an out-of-order P2 is corrected, not obeyed");

        s.set_pre_filter_cap(40);
        s.set_uniqueness_ratio(12);
        assert_eq!(s.get_pre_filter_cap(), 40);
        assert_eq!(s.get_uniqueness_ratio(), 12);
        assert_eq!(s.get_p1(), 100);
        assert_eq!(s.get_p2(), 50);
    }

    #[test]
    fn speckles_smaller_than_the_window_are_removed() {
        let mut m = Mat::zeros(20, 20, CV_16SC1).unwrap();
        for y in 0..20 {
            for x in 0..20 {
                m.pixel_mut::<i16>(y, x).unwrap()[0] = 100;
            }
        }
        // A three-pixel island at a very different disparity.
        for (y, x) in [(5, 5), (5, 6), (6, 5)] {
            m.pixel_mut::<i16>(y, x).unwrap()[0] = 900;
        }
        let removed = filter_speckles(&mut m, -16.0, 8, 4).unwrap();
        assert_eq!(removed, 3, "the island should go");
        assert_eq!(m.at_f64(5, 5, 0).unwrap(), -16.0);
        assert_eq!(m.at_f64(0, 0, 0).unwrap(), 100.0, "the large region survives");

        // A window smaller than the island leaves it alone.
        let mut m2 = Mat::zeros(20, 20, CV_16SC1).unwrap();
        for (y, x) in [(5, 5), (5, 6), (6, 5)] {
            m2.pixel_mut::<i16>(y, x).unwrap()[0] = 900;
        }
        assert_eq!(filter_speckles(&mut m2, -16.0, 2, 4).unwrap(), 0);
        let f = Mat::zeros(4, 4, cv_core::types::CV_32FC1).unwrap();
        let mut f = f;
        assert!(filter_speckles(&mut f, 0.0, 4, 1).is_err(), "CV_16S only");
    }

    #[test]
    fn left_right_validation_drops_inconsistent_pixels() {
        let (l, r) = pair(120, 48, 10);
        let bm = StereoBm::create(32, 15);
        let mut d = bm.compute(&l, &r).unwrap();
        let before = (0..48)
            .flat_map(|y| (0..120).map(move |x| (y, x)))
            .filter(|&(y, x)| d.at_f64(y, x, 0).unwrap() as i16 != bm.invalid())
            .count();
        validate_disparity(&mut d, 0, 32, 1).unwrap();
        let after = (0..48)
            .flat_map(|y| (0..120).map(move |x| (y, x)))
            .filter(|&(y, x)| d.at_f64(y, x, 0).unwrap() as i16 != bm.invalid())
            .count();
        assert!(after <= before, "validation can only remove: {after} vs {before}");
        assert!(after > before / 2, "it should not remove nearly everything: {after}/{before}");

        // A negative disp12MaxDiff disables the check, as upstream's does.
        let mut e = bm.compute(&l, &r).unwrap();
        let snapshot = e.to_vec::<i16>().unwrap();
        validate_disparity(&mut e, 0, 32, -1).unwrap();
        assert_eq!(e.to_vec::<i16>().unwrap(), snapshot);
    }

    #[test]
    fn the_valid_roi_shrinks_by_the_block_and_the_search_range() {
        let roi = get_valid_disparity_roi(
            Rect::new(0, 0, 100, 100),
            Rect::new(0, 0, 100, 100),
            0,
            16,
            9,
        );
        // 4 pixels of block radius on every side, plus 15 columns of search.
        assert_eq!(roi.x, 19);
        assert_eq!(roi.y, 4);
        assert_eq!(roi.x + roi.width, 96);
        assert_eq!(roi.y + roi.height, 96);
        // Disjoint inputs give an empty rectangle, not a negative one.
        let empty =
            get_valid_disparity_roi(Rect::new(0, 0, 10, 10), Rect::new(90, 90, 10, 10), 0, 16, 9);
        assert_eq!((empty.width, empty.height), (0, 0));
    }

    #[test]
    fn reprojection_maps_disparity_through_q() {
        // The Q a rectified pair with focal 100, baseline 0.1 and centre (50,
        // 25) produces: X = (x - cx)/W, Y = (y - cy)/W, Z = f/W, W = d/B.
        let q = [
            [1.0, 0.0, 0.0, -50.0],
            [0.0, 1.0, 0.0, -25.0],
            [0.0, 0.0, 0.0, 100.0],
            [0.0, 0.0, 1.0 / 0.1, 0.0],
        ];
        let mut d = Mat::zeros(50, 100, CV_16SC1).unwrap();
        for y in 0..50 {
            for x in 0..100 {
                // Two pixels of disparity everywhere, in 1/16ths.
                d.pixel_mut::<i16>(y, x).unwrap()[0] = (2 * DISP_SCALE) as i16;
            }
        }
        let pts = reproject_image_to_3d(&d, &q, false).unwrap();
        assert_eq!(pts.channels(), 3);
        // At the principal point X and Y vanish and Z is f*B/d = 100*0.1/2.
        assert!((f64::from(pts.at_f64(25, 50, 0).unwrap() as f32)).abs() < 1e-4);
        assert!((f64::from(pts.at_f64(25, 50, 1).unwrap() as f32)).abs() < 1e-4);
        assert!((pts.at_f64(25, 50, 2).unwrap() - 5.0).abs() < 1e-3);
        // Depth is inversely proportional to disparity: double it, halve Z.
        for y in 0..50 {
            for x in 0..100 {
                d.pixel_mut::<i16>(y, x).unwrap()[0] = (4 * DISP_SCALE) as i16;
            }
        }
        let near = reproject_image_to_3d(&d, &q, false).unwrap();
        assert!((near.at_f64(25, 50, 2).unwrap() - 2.5).abs() < 1e-3);
    }

    #[test]
    fn missing_disparities_are_pushed_to_infinity_when_asked() {
        let mut d = Mat::zeros(10, 10, CV_16SC1).unwrap();
        for y in 0..10 {
            for x in 0..10 {
                d.pixel_mut::<i16>(y, x).unwrap()[0] = (3 * DISP_SCALE) as i16;
            }
        }
        d.pixel_mut::<i16>(5, 5).unwrap()[0] = -DISP_SCALE as i16;
        let q = [
            [1.0, 0.0, 0.0, -5.0],
            [0.0, 1.0, 0.0, -5.0],
            [0.0, 0.0, 0.0, 50.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let handled = reproject_image_to_3d(&d, &q, true).unwrap();
        assert!(handled.at_f64(5, 5, 2).unwrap() > 9000.0, "the gap is pushed far away");
        assert!(handled.at_f64(0, 0, 2).unwrap() < 100.0, "a real point is not");
        // Without the flag the same pixel gets a plausible-looking near point.
        let raw = reproject_image_to_3d(&d, &q, false).unwrap();
        assert!(raw.at_f64(5, 5, 2).unwrap() < 9000.0);
    }
}
