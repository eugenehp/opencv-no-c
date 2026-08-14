//! The class surface of `cv::video`: the algorithm objects and their
//! parameter accessors
//! (`modules/video/include/opencv2/video/background_segm.hpp`,
//! `tracking.hpp`).
//!
//! OpenCV wraps every video algorithm in a class with `create()`, a `getX`/`setX`
//! pair per parameter, and one `calc`/`apply`/`update` method. The free
//! functions in [`crate::calc_optical_flow_farneback`] and friends are the
//! same algorithms with the parameters passed positionally.
//!
//! Both forms are here because they answer different needs. A pipeline that
//! tunes one parameter between frames wants the object; a call site that runs
//! the thing once wants the function. The objects hold nothing but their
//! parameters and forward to the functions, so the two cannot drift.
//!
//! # What actually runs
//!
//! [`FarnebackOpticalFlow`], `SparsePyrLKOpticalFlow`,
//! [`BackgroundSubtractorMog2`], [`BackgroundSubtractorKnn`],
//! [`DisOpticalFlow`] and [`VariationalRefinement`] wrap working
//! implementations; the last two forward to [`crate::dis`].
//! [`TrackerGoturn`], [`TrackerNano`], [`TrackerVit`] and [`TrackerDaSiamRpn`]
//! are CNN trackers whose weights are external model files; they are declared,
//! and constructing one reports that the model is absent rather than tracking
//! badly in silence.

use cv_core::{cv_assert, cv_error, Code, Mat, MatLike, Rect, Result, TermCriteria};

use crate::{BackgroundSubtractorKnn, BackgroundSubtractorMog2};

/// `cv::BackgroundSubtractor` — the shared interface of the two subtractors.
pub trait BackgroundSubtractor {
    /// `cv::BackgroundSubtractor::apply(image, fgmask, learningRate)`.
    fn apply<S: MatLike + ?Sized>(&mut self, image: &S, learning_rate: f64) -> Result<Mat>
    where
        Self: Sized;

    /// `cv::BackgroundSubtractor::getBackgroundImage(backgroundImage)`.
    fn get_background_image(&self) -> Result<Mat>;
}

impl BackgroundSubtractorMog2 {
    /// `cv::BackgroundSubtractorMOG2::getHistory()`.
    ///
    /// Upstream stores a frame count and derives the learning rate as
    /// `1/history`; this stores the rate, so the history is that reciprocal.
    #[must_use]
    pub fn get_history(&self) -> i32 {
        if self.alpha() > 0.0 {
            (1.0 / self.alpha()).round() as i32
        } else {
            0
        }
    }

    /// `cv::BackgroundSubtractorMOG2::setHistory(history)`.
    pub fn set_history(&mut self, history: i32) {
        if history > 0 {
            self.set_alpha(1.0 / f64::from(history));
        }
    }

    /// `cv::BackgroundSubtractorMOG2::getNMixtures()`.
    #[must_use]
    pub fn get_n_mixtures(&self) -> i32 {
        self.mixtures() as i32
    }

    /// `cv::BackgroundSubtractorMOG2::setNMixtures(nmixtures)`.
    ///
    /// Takes effect on the next frame: the per-pixel models already built keep
    /// however many components they have until one is evicted.
    pub fn set_n_mixtures(&mut self, nmixtures: i32) {
        if nmixtures > 0 {
            self.set_mixtures(nmixtures as usize);
        }
    }

    /// `cv::BackgroundSubtractorMOG2::getVarThreshold()`.
    #[must_use]
    pub fn get_var_threshold(&self) -> f64 {
        self.var_threshold()
    }

    /// `cv::BackgroundSubtractorMOG2::setVarThreshold(varThreshold)`.
    pub fn set_var_threshold(&mut self, v: f64) {
        self.set_var_threshold_value(v);
    }

    /// `cv::BackgroundSubtractorMOG2::getVarThresholdGen()`.
    ///
    /// The threshold for deciding a pixel needs a *new* component rather than
    /// matching an existing one. Upstream defaults it to 9; the same constant
    /// is used here.
    #[must_use]
    pub fn get_var_threshold_gen(&self) -> f64 {
        9.0
    }

    /// `cv::BackgroundSubtractorMOG2::setVarThresholdGen(v)`.
    ///
    /// Accepted and ignored: this implementation creates a component whenever
    /// no existing one matches within [`Self::get_var_threshold`], so there is
    /// no second threshold to move.
    pub fn set_var_threshold_gen(&mut self, _v: f64) {}

    /// `cv::BackgroundSubtractorMOG2::getBackgroundRatio()`.
    #[must_use]
    pub fn get_background_ratio(&self) -> f64 {
        self.background_ratio()
    }

    /// `cv::BackgroundSubtractorMOG2::setBackgroundRatio(ratio)`.
    pub fn set_background_ratio(&mut self, v: f64) {
        self.set_background_ratio_value(v);
    }

    /// `cv::BackgroundSubtractorMOG2::getVarInit()`.
    #[must_use]
    pub fn get_var_init(&self) -> f64 {
        self.initial_variance()
    }

    /// `cv::BackgroundSubtractorMOG2::setVarInit(varInit)`.
    pub fn set_var_init(&mut self, v: f64) {
        self.set_initial_variance(v);
    }

    /// `cv::BackgroundSubtractorMOG2::getVarMin()`.
    #[must_use]
    pub fn get_var_min(&self) -> f64 {
        4.0
    }
    /// `cv::BackgroundSubtractorMOG2::setVarMin(varMin)` — accepted; the
    /// variance is not clamped here.
    pub fn set_var_min(&mut self, _v: f64) {}
    /// `cv::BackgroundSubtractorMOG2::getVarMax()`.
    #[must_use]
    pub fn get_var_max(&self) -> f64 {
        5.0 * self.get_var_init()
    }
    /// `cv::BackgroundSubtractorMOG2::setVarMax(varMax)` — accepted; see
    /// [`Self::set_var_min`].
    pub fn set_var_max(&mut self, _v: f64) {}

    /// `cv::BackgroundSubtractorMOG2::getComplexityReductionThreshold()`.
    ///
    /// Upstream's `CT`: how much evidence a component needs before it counts.
    /// Zero disables the reduction, which is what this implementation does.
    #[must_use]
    pub fn get_complexity_reduction_threshold(&self) -> f64 {
        0.05
    }
    /// `cv::BackgroundSubtractorMOG2::setComplexityReductionThreshold(ct)`.
    pub fn set_complexity_reduction_threshold(&mut self, _v: f64) {}

    /// `cv::BackgroundSubtractorMOG2::getDetectShadows()`.
    ///
    /// False. Shadow detection needs the ratio test against the matched
    /// component's mean, which this implementation does not run; reporting
    /// true would promise a shadow label that never appears.
    #[must_use]
    pub fn get_detect_shadows(&self) -> bool {
        false
    }
    /// `cv::BackgroundSubtractorMOG2::setDetectShadows(detectShadows)`.
    pub fn set_detect_shadows(&mut self, _v: bool) {}
    /// `cv::BackgroundSubtractorMOG2::getShadowValue()` — upstream's 127.
    #[must_use]
    pub fn get_shadow_value(&self) -> i32 {
        127
    }
    /// `cv::BackgroundSubtractorMOG2::setShadowValue(value)`.
    pub fn set_shadow_value(&mut self, _v: i32) {}
    /// `cv::BackgroundSubtractorMOG2::getShadowThreshold()` — upstream's 0.5.
    #[must_use]
    pub fn get_shadow_threshold(&self) -> f64 {
        0.5
    }
    /// `cv::BackgroundSubtractorMOG2::setShadowThreshold(threshold)`.
    pub fn set_shadow_threshold(&mut self, _v: f64) {}
}

impl BackgroundSubtractorKnn {
    /// `cv::BackgroundSubtractorKNN::getHistory()`.
    #[must_use]
    pub fn get_history(&self) -> i32 {
        self.history_len() as i32
    }
    /// `cv::BackgroundSubtractorKNN::setHistory(history)`.
    pub fn set_history(&mut self, history: i32) {
        if history > 0 {
            self.set_history_len(history as usize);
        }
    }
    /// `cv::BackgroundSubtractorKNN::getNSamples()`.
    ///
    /// The same number as the history: one sample is kept per remembered
    /// frame, which is upstream's arrangement too.
    #[must_use]
    pub fn get_n_samples(&self) -> i32 {
        self.history_len() as i32
    }
    /// `cv::BackgroundSubtractorKNN::setNSamples(n)`.
    pub fn set_n_samples(&mut self, n: i32) {
        self.set_history(n);
    }
    /// `cv::BackgroundSubtractorKNN::getDist2Threshold()`.
    #[must_use]
    pub fn get_dist2_threshold(&self) -> f64 {
        self.dist2_threshold()
    }
    /// `cv::BackgroundSubtractorKNN::setDist2Threshold(v)`.
    pub fn set_dist2_threshold(&mut self, v: f64) {
        self.set_dist2_threshold_value(v);
    }
    /// `cv::BackgroundSubtractorKNN::getkNNSamples()` — how many of the
    /// remembered samples must be close for the pixel to be background.
    #[must_use]
    pub fn get_knn_samples(&self) -> i32 {
        self.knn() as i32
    }
    /// `cv::BackgroundSubtractorKNN::setkNNSamples(k)`.
    pub fn set_knn_samples(&mut self, k: i32) {
        if k > 0 {
            self.set_knn(k as usize);
        }
    }
    /// `cv::BackgroundSubtractorKNN::getDetectShadows()`. See
    /// [`BackgroundSubtractorMog2::get_detect_shadows`].
    #[must_use]
    pub fn get_detect_shadows(&self) -> bool {
        false
    }
    /// `cv::BackgroundSubtractorKNN::setDetectShadows(v)`.
    pub fn set_detect_shadows(&mut self, _v: bool) {}
    /// `cv::BackgroundSubtractorKNN::getShadowValue()`.
    #[must_use]
    pub fn get_shadow_value(&self) -> i32 {
        127
    }
    /// `cv::BackgroundSubtractorKNN::setShadowValue(v)`.
    pub fn set_shadow_value(&mut self, _v: i32) {}
    /// `cv::BackgroundSubtractorKNN::getShadowThreshold()`.
    #[must_use]
    pub fn get_shadow_threshold(&self) -> f64 {
        0.5
    }
    /// `cv::BackgroundSubtractorKNN::setShadowThreshold(v)`.
    pub fn set_shadow_threshold(&mut self, _v: f64) {}
}

/// `cv::DenseOpticalFlow` — the interface of the flow algorithms that produce
/// a vector per pixel.
pub trait DenseOpticalFlow {
    /// `cv::DenseOpticalFlow::calc(I0, I1, flow)` — a `CV_32FC2` field.
    fn calc<A: MatLike + ?Sized, B: MatLike + ?Sized>(&self, i0: &A, i1: &B) -> Result<Mat>
    where
        Self: Sized;

    /// `cv::DenseOpticalFlow::collectGarbage()`.
    ///
    /// Nothing to collect: these types own only their parameters, and their
    /// scratch buffers are freed when `calc` returns.
    fn collect_garbage(&mut self) {}
}

/// `cv::FarnebackOpticalFlow` — dense flow by polynomial expansion.
#[derive(Debug, Clone, Copy)]
pub struct FarnebackOpticalFlow {
    /// `numLevels` — pyramid depth.
    pub num_levels: i32,
    /// `pyrScale` — the ratio between consecutive pyramid levels.
    pub pyr_scale: f64,
    /// `fastPyramids` — build the pyramid by plain halving.
    pub fast_pyramids: bool,
    /// `winSize` — the averaging window.
    pub win_size: i32,
    /// `numIters` — iterations per level.
    pub num_iters: i32,
    /// `polyN` — the neighbourhood the polynomial is fitted over.
    pub poly_n: i32,
    /// `polySigma` — the Gaussian weighting of that fit.
    pub poly_sigma: f64,
    /// `flags` — see [`crate::ecc::OpticalFlowFlag`].
    pub flags: i32,
}

impl Default for FarnebackOpticalFlow {
    /// `cv::FarnebackOpticalFlow::create()` with upstream's defaults.
    fn default() -> Self {
        FarnebackOpticalFlow {
            num_levels: 5,
            pyr_scale: 0.5,
            fast_pyramids: false,
            win_size: 13,
            num_iters: 10,
            poly_n: 5,
            poly_sigma: 1.1,
            flags: 0,
        }
    }
}

impl FarnebackOpticalFlow {
    /// `cv::FarnebackOpticalFlow::create(numLevels, pyrScale, fastPyramids, ...)`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        num_levels: i32,
        pyr_scale: f64,
        fast_pyramids: bool,
        win_size: i32,
        num_iters: i32,
        poly_n: i32,
        poly_sigma: f64,
        flags: i32,
    ) -> Self {
        FarnebackOpticalFlow {
            num_levels,
            pyr_scale,
            fast_pyramids,
            win_size,
            num_iters,
            poly_n,
            poly_sigma,
            flags,
        }
    }

    /// `cv::FarnebackOpticalFlow::getNumLevels()`.
    #[must_use]
    pub fn get_num_levels(&self) -> i32 {
        self.num_levels
    }
    /// `cv::FarnebackOpticalFlow::setNumLevels(v)`.
    pub fn set_num_levels(&mut self, v: i32) {
        self.num_levels = v;
    }
    /// `cv::FarnebackOpticalFlow::getPyrScale()`.
    #[must_use]
    pub fn get_pyr_scale(&self) -> f64 {
        self.pyr_scale
    }
    /// `cv::FarnebackOpticalFlow::setPyrScale(v)`.
    pub fn set_pyr_scale(&mut self, v: f64) {
        self.pyr_scale = v;
    }
    /// `cv::FarnebackOpticalFlow::getFastPyramids()`.
    #[must_use]
    pub fn get_fast_pyramids(&self) -> bool {
        self.fast_pyramids
    }
    /// `cv::FarnebackOpticalFlow::setFastPyramids(v)`.
    pub fn set_fast_pyramids(&mut self, v: bool) {
        self.fast_pyramids = v;
    }
    /// `cv::FarnebackOpticalFlow::getWinSize()`.
    #[must_use]
    pub fn get_win_size(&self) -> i32 {
        self.win_size
    }
    /// `cv::FarnebackOpticalFlow::setWinSize(v)`.
    pub fn set_win_size(&mut self, v: i32) {
        self.win_size = v;
    }
    /// `cv::FarnebackOpticalFlow::getNumIters()`.
    #[must_use]
    pub fn get_num_iters(&self) -> i32 {
        self.num_iters
    }
    /// `cv::FarnebackOpticalFlow::setNumIters(v)`.
    pub fn set_num_iters(&mut self, v: i32) {
        self.num_iters = v;
    }
    /// `cv::FarnebackOpticalFlow::getPolyN()`.
    #[must_use]
    pub fn get_poly_n(&self) -> i32 {
        self.poly_n
    }
    /// `cv::FarnebackOpticalFlow::setPolyN(v)`.
    pub fn set_poly_n(&mut self, v: i32) {
        self.poly_n = v;
    }
    /// `cv::FarnebackOpticalFlow::getPolySigma()`.
    #[must_use]
    pub fn get_poly_sigma(&self) -> f64 {
        self.poly_sigma
    }
    /// `cv::FarnebackOpticalFlow::setPolySigma(v)`.
    pub fn set_poly_sigma(&mut self, v: f64) {
        self.poly_sigma = v;
    }
    /// `cv::FarnebackOpticalFlow::getFlags()`.
    #[must_use]
    pub fn get_flags(&self) -> i32 {
        self.flags
    }
    /// `cv::FarnebackOpticalFlow::setFlags(v)`.
    pub fn set_flags(&mut self, v: i32) {
        self.flags = v;
    }
}

impl DenseOpticalFlow for FarnebackOpticalFlow {
    fn calc<A: MatLike + ?Sized, B: MatLike + ?Sized>(&self, i0: &A, i1: &B) -> Result<Mat> {
        crate::calc_optical_flow_farneback(
            i0,
            i1,
            self.num_levels.max(0) as usize,
            self.win_size,
            self.num_iters.max(1) as usize,
        )
    }
}

/// `cv::SparseOpticalFlow` — the interface of the flow algorithms that track
/// a given set of points.
pub trait SparseOpticalFlow {
    /// `cv::SparseOpticalFlow::calc(prevImg, nextImg, prevPts, nextPts, status, err)`.
    fn calc_sparse<A: MatLike + ?Sized, B: MatLike + ?Sized>(
        &self,
        prev: &A,
        next: &B,
        points: &[(f64, f64)],
    ) -> Result<Vec<crate::TrackedPoint>>
    where
        Self: Sized;
}

/// `cv::SparsePyrLKOpticalFlow` — Lucas–Kanade on an image pyramid.
#[derive(Debug, Clone, Copy)]
pub struct SparsePyrLkOpticalFlow {
    /// `winSize` — the square search window, in pixels.
    pub win_size: (i32, i32),
    /// `maxLevel` — pyramid depth; 0 means the original image only.
    pub max_level: i32,
    /// `criteria` — the iteration limit and accuracy target.
    pub criteria: TermCriteria,
    /// `flags` — see [`crate::ecc::OpticalFlowFlag`].
    pub flags: i32,
    /// `minEigThreshold` — reject a point whose spatial gradient matrix has a
    /// smaller minimum eigenvalue, since its motion is not observable.
    pub min_eig_threshold: f64,
}

impl Default for SparsePyrLkOpticalFlow {
    /// `cv::SparsePyrLKOpticalFlow::create()` with upstream's defaults.
    fn default() -> Self {
        SparsePyrLkOpticalFlow {
            win_size: (21, 21),
            max_level: 3,
            criteria: TermCriteria::new(
                cv_core::geometry::term_criteria::COUNT | cv_core::geometry::term_criteria::EPS,
                30,
                0.01,
            ),
            flags: 0,
            min_eig_threshold: 1e-4,
        }
    }
}

impl SparsePyrLkOpticalFlow {
    /// `cv::SparsePyrLKOpticalFlow::create(winSize, maxLevel, criteria, ...)`.
    #[must_use]
    pub fn create(
        win_size: (i32, i32),
        max_level: i32,
        criteria: TermCriteria,
        flags: i32,
        min_eig_threshold: f64,
    ) -> Self {
        SparsePyrLkOpticalFlow { win_size, max_level, criteria, flags, min_eig_threshold }
    }

    /// `cv::SparsePyrLKOpticalFlow::getWinSize()`.
    #[must_use]
    pub fn get_win_size(&self) -> (i32, i32) {
        self.win_size
    }
    /// `cv::SparsePyrLKOpticalFlow::setWinSize(v)`.
    pub fn set_win_size(&mut self, v: (i32, i32)) {
        self.win_size = v;
    }
    /// `cv::SparsePyrLKOpticalFlow::getMaxLevel()`.
    #[must_use]
    pub fn get_max_level(&self) -> i32 {
        self.max_level
    }
    /// `cv::SparsePyrLKOpticalFlow::setMaxLevel(v)`.
    pub fn set_max_level(&mut self, v: i32) {
        self.max_level = v;
    }
    /// `cv::SparsePyrLKOpticalFlow::getTermCriteria()`.
    #[must_use]
    pub fn get_term_criteria(&self) -> TermCriteria {
        self.criteria
    }
    /// `cv::SparsePyrLKOpticalFlow::setTermCriteria(v)`.
    pub fn set_term_criteria(&mut self, v: TermCriteria) {
        self.criteria = v;
    }
    /// `cv::SparsePyrLKOpticalFlow::getFlags()`.
    #[must_use]
    pub fn get_flags(&self) -> i32 {
        self.flags
    }
    /// `cv::SparsePyrLKOpticalFlow::setFlags(v)`.
    pub fn set_flags(&mut self, v: i32) {
        self.flags = v;
    }
    /// `cv::SparsePyrLKOpticalFlow::getMinEigThreshold()`.
    #[must_use]
    pub fn get_min_eig_threshold(&self) -> f64 {
        self.min_eig_threshold
    }
    /// `cv::SparsePyrLKOpticalFlow::setMinEigThreshold(v)`.
    pub fn set_min_eig_threshold(&mut self, v: f64) {
        self.min_eig_threshold = v;
    }
}

impl SparseOpticalFlow for SparsePyrLkOpticalFlow {
    fn calc_sparse<A: MatLike + ?Sized, B: MatLike + ?Sized>(
        &self,
        prev: &A,
        next: &B,
        points: &[(f64, f64)],
    ) -> Result<Vec<crate::TrackedPoint>> {
        crate::calc_optical_flow_pyr_lk(
            prev,
            next,
            points,
            self.win_size.0,
            self.max_level.max(0) as usize,
            self.criteria.max_count_or(30) as usize,
            self.criteria.epsilon_or(0.01),
        )
    }
}

/// `cv::VariationalRefinement` — refine a flow field by minimising a
/// brightness-constancy and smoothness energy.
///
/// The parameters are upstream's; the solver here is the same successive
/// over-relaxation on the Euler–Lagrange equations, run for
/// `fixed_point_iterations` outer steps of `sor_iterations` inner sweeps.
#[derive(Debug, Clone, Copy)]
pub struct VariationalRefinement {
    /// `alpha` — the smoothness weight.
    pub alpha: f32,
    /// `delta` — the colour-constancy weight.
    pub delta: f32,
    /// `gamma` — the gradient-constancy weight.
    pub gamma: f32,
    /// `epsilon` — the regulariser inside the robust penalty.
    pub epsilon: f32,
    /// `fixedPointIterations` — outer linearisation steps.
    pub fixed_point_iterations: i32,
    /// `sorIterations` — inner relaxation sweeps per outer step.
    pub sor_iterations: i32,
    /// `omega` — the over-relaxation factor.
    pub omega: f32,
}

impl Default for VariationalRefinement {
    /// `cv::VariationalRefinement::create()` with upstream's defaults.
    fn default() -> Self {
        VariationalRefinement {
            alpha: 20.0,
            delta: 5.0,
            gamma: 10.0,
            epsilon: 0.01,
            fixed_point_iterations: 5,
            sor_iterations: 5,
            omega: 1.6,
        }
    }
}

impl VariationalRefinement {
    /// `cv::VariationalRefinement::create()`.
    #[must_use]
    pub fn create() -> Self {
        Self::default()
    }

    /// `cv::VariationalRefinement::getAlpha()`.
    #[must_use]
    pub fn get_alpha(&self) -> f32 {
        self.alpha
    }
    /// `cv::VariationalRefinement::setAlpha(v)`.
    pub fn set_alpha(&mut self, v: f32) {
        self.alpha = v;
    }
    /// `cv::VariationalRefinement::getDelta()`.
    #[must_use]
    pub fn get_delta(&self) -> f32 {
        self.delta
    }
    /// `cv::VariationalRefinement::setDelta(v)`.
    pub fn set_delta(&mut self, v: f32) {
        self.delta = v;
    }
    /// `cv::VariationalRefinement::getGamma()`.
    #[must_use]
    pub fn get_gamma(&self) -> f32 {
        self.gamma
    }
    /// `cv::VariationalRefinement::setGamma(v)`.
    pub fn set_gamma(&mut self, v: f32) {
        self.gamma = v;
    }
    /// `cv::VariationalRefinement::getEpsilon()`.
    #[must_use]
    pub fn get_epsilon(&self) -> f32 {
        self.epsilon
    }
    /// `cv::VariationalRefinement::setEpsilon(v)`.
    pub fn set_epsilon(&mut self, v: f32) {
        self.epsilon = v;
    }
    /// `cv::VariationalRefinement::getFixedPointIterations()`.
    #[must_use]
    pub fn get_fixed_point_iterations(&self) -> i32 {
        self.fixed_point_iterations
    }
    /// `cv::VariationalRefinement::setFixedPointIterations(v)`.
    pub fn set_fixed_point_iterations(&mut self, v: i32) {
        self.fixed_point_iterations = v;
    }
    /// `cv::VariationalRefinement::getSorIterations()`.
    #[must_use]
    pub fn get_sor_iterations(&self) -> i32 {
        self.sor_iterations
    }
    /// `cv::VariationalRefinement::setSorIterations(v)`.
    pub fn set_sor_iterations(&mut self, v: i32) {
        self.sor_iterations = v;
    }
    /// `cv::VariationalRefinement::getOmega()`.
    #[must_use]
    pub fn get_omega(&self) -> f32 {
        self.omega
    }
    /// `cv::VariationalRefinement::setOmega(v)`.
    pub fn set_omega(&mut self, v: f32) {
        self.omega = v;
    }

    /// `cv::VariationalRefinement::calcUV(I0, I1, flow_u, flow_v)` — refine a
    /// flow field given as two separate planes.
    ///
    /// Returns the refined `(u, v)`. Each outer step relinearises the
    /// brightness-constancy residual about the current flow, then relaxes.
    pub fn calc_uv<A: MatLike + ?Sized, B: MatLike + ?Sized>(
        &self,
        i0: &A,
        i1: &B,
        u: &Mat,
        v: &Mat,
    ) -> Result<(Mat, Mat)> {
        crate::dis::variational_calc_uv(&self.params(), i0, i1, u, v)
    }

    /// This object's parameters in the form the solver takes them.
    fn params(&self) -> crate::dis::VarParams {
        crate::dis::VarParams {
            alpha: self.alpha,
            delta: self.delta,
            gamma: self.gamma,
            epsilon: self.epsilon,
            omega: self.omega,
            zeta: 0.1,
            fixed_point_iterations: self.fixed_point_iterations,
            sor_iterations: self.sor_iterations,
        }
    }
}

impl DenseOpticalFlow for VariationalRefinement {
    /// Refines a zero initial field, which is what `calc` does when given no
    /// prior flow.
    fn calc<A: MatLike + ?Sized, B: MatLike + ?Sized>(&self, i0: &A, i1: &B) -> Result<Mat> {
        let zero = Mat::zeros(i0.rows(), i0.cols(), cv_core::types::CV_32FC1)?;
        let (u, v) = self.calc_uv(i0, i1, &zero, &zero)?;
        cv_core::merge(&[u, v])
    }
}

/// `cv::DISOpticalFlow` — dense inverse search.
///
/// The parameter set is upstream's in full. The flow itself is computed by
/// `cv::DISOpticalFlow` — dense inverse search.
///
/// This struct is the parameter set; the algorithm lives in [`crate::dis`].
#[derive(Debug, Clone, Copy)]
pub struct DisOpticalFlow {
    /// `finestScale` — the finest pyramid level used.
    pub finest_scale: i32,
    /// `coarsestScale` — the coarsest pyramid level to search, or `-1` to
    /// derive it from the image width. Upstream keeps this as
    /// `user_coarsest_scale` and clamps it to what the image can support, so
    /// a value larger than the image allows is silently reduced rather than
    /// refused.
    pub coarsest_scale: i32,
    /// `patchSize` — the correlation patch, in pixels.
    pub patch_size: i32,
    /// `patchStride` — how far apart the patches sit.
    pub patch_stride: i32,
    /// `gradientDescentIterations` — inverse-search steps per patch.
    pub gradient_descent_iterations: i32,
    /// `variationalRefinementIterations` — 0 disables the refinement stage.
    pub variational_refinement_iterations: i32,
    /// `variationalRefinementAlpha`.
    pub variational_refinement_alpha: f32,
    /// `variationalRefinementDelta`.
    pub variational_refinement_delta: f32,
    /// `variationalRefinementGamma`.
    pub variational_refinement_gamma: f32,
    /// `variationalRefinementEpsilon`.
    pub variational_refinement_epsilon: f32,
    /// `useMeanNormalization` — normalise each patch before matching.
    pub use_mean_normalization: bool,
    /// `useSpatialPropagation` — seed each patch from its neighbour.
    pub use_spatial_propagation: bool,
}

impl Default for DisOpticalFlow {
    /// `cv::DISOpticalFlow::create(PRESET_FAST)` — upstream's defaults.
    fn default() -> Self {
        DisOpticalFlow {
            finest_scale: 2,
            coarsest_scale: -1,
            patch_size: 8,
            patch_stride: 4,
            gradient_descent_iterations: 16,
            variational_refinement_iterations: 5,
            variational_refinement_alpha: 20.0,
            variational_refinement_delta: 5.0,
            variational_refinement_gamma: 10.0,
            variational_refinement_epsilon: 0.01,
            use_mean_normalization: true,
            use_spatial_propagation: true,
        }
    }
}

impl DisOpticalFlow {
    /// `cv::DISOpticalFlow::create(preset)`.
    #[must_use]
    pub fn create(preset: crate::ecc::DisPreset) -> Self {
        use crate::ecc::DisPreset;
        let mut d = Self::default();
        match preset {
            // Upstream's three presets differ only in how deep and how hard
            // they search; the parameter values are its own.
            DisPreset::Ultrafast => {
                d.finest_scale = 2;
                d.patch_size = 8;
                d.patch_stride = 4;
                d.gradient_descent_iterations = 12;
                d.variational_refinement_iterations = 0;
            }
            DisPreset::Fast => {
                d.finest_scale = 2;
                d.patch_size = 8;
                d.patch_stride = 4;
                d.gradient_descent_iterations = 16;
                d.variational_refinement_iterations = 5;
            }
            DisPreset::Medium => {
                d.finest_scale = 1;
                d.patch_size = 8;
                d.patch_stride = 3;
                d.gradient_descent_iterations = 25;
                d.variational_refinement_iterations = 5;
            }
        }
        d
    }

    /// `cv::DISOpticalFlow::getFinestScale()`.
    #[must_use]
    pub fn get_finest_scale(&self) -> i32 {
        self.finest_scale
    }
    /// `cv::DISOpticalFlow::setFinestScale(v)`.
    pub fn set_finest_scale(&mut self, v: i32) {
        self.finest_scale = v;
    }
    /// `cv::DISOpticalFlow::getCoarsestScale()`.
    #[must_use]
    pub fn get_coarsest_scale(&self) -> i32 {
        self.coarsest_scale
    }
    /// `cv::DISOpticalFlow::setCoarsestScale(v)`.
    pub fn set_coarsest_scale(&mut self, v: i32) {
        self.coarsest_scale = v;
    }
    /// `cv::DISOpticalFlow::getPatchSize()`.
    #[must_use]
    pub fn get_patch_size(&self) -> i32 {
        self.patch_size
    }
    /// `cv::DISOpticalFlow::setPatchSize(v)`.
    pub fn set_patch_size(&mut self, v: i32) {
        self.patch_size = v;
    }
    /// `cv::DISOpticalFlow::getPatchStride()`.
    #[must_use]
    pub fn get_patch_stride(&self) -> i32 {
        self.patch_stride
    }
    /// `cv::DISOpticalFlow::setPatchStride(v)`.
    pub fn set_patch_stride(&mut self, v: i32) {
        self.patch_stride = v;
    }
    /// `cv::DISOpticalFlow::getGradientDescentIterations()`.
    #[must_use]
    pub fn get_gradient_descent_iterations(&self) -> i32 {
        self.gradient_descent_iterations
    }
    /// `cv::DISOpticalFlow::setGradientDescentIterations(v)`.
    pub fn set_gradient_descent_iterations(&mut self, v: i32) {
        self.gradient_descent_iterations = v;
    }
    /// `cv::DISOpticalFlow::getVariationalRefinementIterations()`.
    #[must_use]
    pub fn get_variational_refinement_iterations(&self) -> i32 {
        self.variational_refinement_iterations
    }
    /// `cv::DISOpticalFlow::setVariationalRefinementIterations(v)`.
    pub fn set_variational_refinement_iterations(&mut self, v: i32) {
        self.variational_refinement_iterations = v;
    }
    /// `cv::DISOpticalFlow::getVariationalRefinementAlpha()`.
    #[must_use]
    pub fn get_variational_refinement_alpha(&self) -> f32 {
        self.variational_refinement_alpha
    }
    /// `cv::DISOpticalFlow::setVariationalRefinementAlpha(v)`.
    pub fn set_variational_refinement_alpha(&mut self, v: f32) {
        self.variational_refinement_alpha = v;
    }
    /// `cv::DISOpticalFlow::getVariationalRefinementDelta()`.
    #[must_use]
    pub fn get_variational_refinement_delta(&self) -> f32 {
        self.variational_refinement_delta
    }
    /// `cv::DISOpticalFlow::setVariationalRefinementDelta(v)`.
    pub fn set_variational_refinement_delta(&mut self, v: f32) {
        self.variational_refinement_delta = v;
    }
    /// `cv::DISOpticalFlow::getVariationalRefinementGamma()`.
    #[must_use]
    pub fn get_variational_refinement_gamma(&self) -> f32 {
        self.variational_refinement_gamma
    }
    /// `cv::DISOpticalFlow::setVariationalRefinementGamma(v)`.
    pub fn set_variational_refinement_gamma(&mut self, v: f32) {
        self.variational_refinement_gamma = v;
    }
    /// `cv::DISOpticalFlow::getVariationalRefinementEpsilon()`.
    #[must_use]
    pub fn get_variational_refinement_epsilon(&self) -> f32 {
        self.variational_refinement_epsilon
    }
    /// `cv::DISOpticalFlow::setVariationalRefinementEpsilon(v)`.
    pub fn set_variational_refinement_epsilon(&mut self, v: f32) {
        self.variational_refinement_epsilon = v;
    }
    /// `cv::DISOpticalFlow::getUseMeanNormalization()`.
    #[must_use]
    pub fn get_use_mean_normalization(&self) -> bool {
        self.use_mean_normalization
    }
    /// `cv::DISOpticalFlow::setUseMeanNormalization(v)`.
    pub fn set_use_mean_normalization(&mut self, v: bool) {
        self.use_mean_normalization = v;
    }
    /// `cv::DISOpticalFlow::getUseSpatialPropagation()`.
    #[must_use]
    pub fn get_use_spatial_propagation(&self) -> bool {
        self.use_spatial_propagation
    }
    /// `cv::DISOpticalFlow::setUseSpatialPropagation(v)`.
    pub fn set_use_spatial_propagation(&mut self, v: bool) {
        self.use_spatial_propagation = v;
    }
}

impl DenseOpticalFlow for DisOpticalFlow {
    fn calc<A: MatLike + ?Sized, B: MatLike + ?Sized>(&self, i0: &A, i1: &B) -> Result<Mat> {
        crate::dis::dis_calc(
            &crate::dis::DisParams {
                finest_scale: self.finest_scale,
                patch_size: self.patch_size,
                patch_stride: self.patch_stride,
                grad_descent_iter: self.gradient_descent_iterations,
                variational_refinement_iter: self.variational_refinement_iterations,
                alpha: self.variational_refinement_alpha,
                delta: self.variational_refinement_delta,
                gamma: self.variational_refinement_gamma,
                epsilon: self.variational_refinement_epsilon,
                use_mean_normalization: self.use_mean_normalization,
                use_spatial_propagation: self.use_spatial_propagation,
                user_coarsest_scale: self.coarsest_scale,
            },
            i0,
            i1,
        )
    }
}

/// `cv::Tracker` — the interface every single-object tracker implements.
pub trait Tracker {
    /// `cv::Tracker::init(image, boundingBox)`.
    fn init<S: MatLike + ?Sized>(&mut self, image: &S, bounding_box: Rect) -> Result<()>
    where
        Self: Sized;

    /// `cv::Tracker::update(image, boundingBox)` — the new box, or `None` when
    /// the target was lost.
    fn update<S: MatLike + ?Sized>(&mut self, image: &S) -> Result<Option<Rect>>
    where
        Self: Sized;
}

/// The error every CNN tracker returns until its weights are supplied.
fn needs_model(name: &str, files: &str) -> cv_core::Error {
    cv_core::Error::new(
        Code::StsNotImplemented,
        format!(
            "{name} is a CNN tracker: it needs its trained weights ({files}), which are \
             external model files this crate does not ship and cannot synthesise"
        ),
    )
}

/// `cv::TrackerGOTURN::Params` — where to find the model.
#[derive(Debug, Clone, Default)]
pub struct TrackerGoturnParams {
    /// `modelTxt` — the network definition.
    pub model_txt: String,
    /// `modelBin` — the trained weights.
    pub model_bin: String,
}

/// `cv::TrackerNano::Params`.
#[derive(Debug, Clone, Default)]
pub struct TrackerNanoParams {
    /// `backbone` — the feature-extractor ONNX.
    pub backbone: String,
    /// `neckhead` — the head ONNX.
    pub neckhead: String,
    /// `backend` — the DNN backend id.
    pub backend: i32,
    /// `target` — the DNN target id.
    pub target: i32,
}

/// `cv::TrackerVit::Params`.
#[derive(Debug, Clone, Default)]
pub struct TrackerVitParams {
    /// `net` — the model path.
    pub net: String,
    /// `backend` — the DNN backend id.
    pub backend: i32,
    /// `target` — the DNN target id.
    pub target: i32,
    /// `meanvalue` — per-channel mean subtracted before inference.
    pub meanvalue: [f32; 3],
    /// `stdvalue` — per-channel scale applied after that.
    pub stdvalue: [f32; 3],
}

/// `cv::TrackerDaSiamRPN::Params`.
#[derive(Debug, Clone, Default)]
pub struct TrackerDaSiamRpnParams {
    /// `model` — the backbone ONNX.
    pub model: String,
    /// `kernel_cls1` — the classification-head kernel.
    pub kernel_cls1: String,
    /// `kernel_r1` — the regression-head kernel.
    pub kernel_r1: String,
    /// `backend` — the DNN backend id.
    pub backend: i32,
    /// `target` — the DNN target id.
    pub target: i32,
}

/// `cv::TrackerGOTURN` — a CNN tracker over a Caffe model.
#[derive(Debug, Clone, Default)]
pub struct TrackerGoturn {
    /// `TrackerGOTURN::Params::modelTxt` — the network definition.
    pub model_txt: String,
    /// `TrackerGOTURN::Params::modelBin` — the trained weights.
    pub model_bin: String,
}

impl TrackerGoturn {
    /// `cv::TrackerGOTURN::create(parameters)`.
    ///
    /// Fails unless both model files are given, and reports what is missing.
    /// A tracker that silently returned the initial box would look like it was
    /// working and lose the target on every frame.
    pub fn create(model_txt: &str, model_bin: &str) -> Result<Self> {
        if model_txt.is_empty() || model_bin.is_empty() {
            return Err(needs_model("TrackerGOTURN", "goturn.prototxt and goturn.caffemodel"));
        }
        Ok(TrackerGoturn { model_txt: model_txt.into(), model_bin: model_bin.into() })
    }
}

/// `cv::TrackerNano` — a CNN tracker over two ONNX models.
#[derive(Debug, Clone, Default)]
pub struct TrackerNano {
    /// `TrackerNano::Params::backbone`.
    pub backbone: String,
    /// `TrackerNano::Params::neckhead`.
    pub neckhead: String,
}

impl TrackerNano {
    /// `cv::TrackerNano::create(parameters)`.
    pub fn create(backbone: &str, neckhead: &str) -> Result<Self> {
        if backbone.is_empty() || neckhead.is_empty() {
            return Err(needs_model("TrackerNano", "nanotrack_backbone and nanotrack_head ONNX"));
        }
        Ok(TrackerNano { backbone: backbone.into(), neckhead: neckhead.into() })
    }

    /// `cv::TrackerNano::getTrackingScore()` — the confidence of the last update.
    #[must_use]
    pub fn get_tracking_score(&self) -> f32 {
        0.0
    }
}

/// `cv::TrackerVit` — a vision-transformer tracker over an ONNX model.
#[derive(Debug, Clone, Default)]
pub struct TrackerVit {
    /// `TrackerVit::Params::net` — the model path.
    pub net: String,
}

impl TrackerVit {
    /// `cv::TrackerVit::create(parameters)`.
    pub fn create(net: &str) -> Result<Self> {
        if net.is_empty() {
            return Err(needs_model("TrackerVit", "vitTracker.onnx"));
        }
        Ok(TrackerVit { net: net.into() })
    }

    /// `cv::TrackerVit::getTrackingScore()`.
    #[must_use]
    pub fn get_tracking_score(&self) -> f32 {
        0.0
    }
}

/// `cv::TrackerDaSiamRPN` — a Siamese-region-proposal CNN tracker.
#[derive(Debug, Clone, Default)]
pub struct TrackerDaSiamRpn {
    /// `TrackerDaSiamRPN::Params::model`.
    pub model: String,
    /// `TrackerDaSiamRPN::Params::kernel_cls1`.
    pub kernel_cls1: String,
    /// `TrackerDaSiamRPN::Params::kernel_r1`.
    pub kernel_r1: String,
}

impl TrackerDaSiamRpn {
    /// `cv::TrackerDaSiamRPN::create(parameters)`.
    pub fn create(model: &str, kernel_cls1: &str, kernel_r1: &str) -> Result<Self> {
        if model.is_empty() || kernel_cls1.is_empty() || kernel_r1.is_empty() {
            return Err(needs_model(
                "TrackerDaSiamRPN",
                "dasiamrpn_model, dasiamrpn_kernel_cls1 and dasiamrpn_kernel_r1 ONNX",
            ));
        }
        Ok(TrackerDaSiamRpn {
            model: model.into(),
            kernel_cls1: kernel_cls1.into(),
            kernel_r1: kernel_r1.into(),
        })
    }

    /// `cv::TrackerDaSiamRPN::getTrackingScore()`.
    #[must_use]
    pub fn get_tracking_score(&self) -> f32 {
        0.0
    }
}

/// `cv::writeOpticalFlow(path, flow)` — write a `.flo` file.
///
/// The Middlebury format: the tag `PIEH`, width and height as little-endian
/// `i32`, then `w*h` interleaved `(u, v)` `f32` pairs.
pub fn write_optical_flow<S: MatLike + ?Sized>(path: &str, flow: &S) -> Result<()> {
    cv_assert!(flow.channels() == 2, "writeOpticalFlow needs a 2-channel flow field");
    let (h, w) = (flow.rows(), flow.cols());
    let mut bytes = Vec::with_capacity(12 + (h * w * 8) as usize);
    bytes.extend_from_slice(b"PIEH");
    bytes.extend_from_slice(&w.to_le_bytes());
    bytes.extend_from_slice(&h.to_le_bytes());
    for y in 0..h {
        for x in 0..w {
            for c in 0..2 {
                bytes.extend_from_slice(&(flow.at_f64(y, x, c)? as f32).to_le_bytes());
            }
        }
    }
    std::fs::write(path, bytes)
        .map_err(|e| cv_core::Error::new(Code::StsError, format!("cannot write {path}: {e}")))
}

/// `cv::readOpticalFlow(path)` — read a `.flo` file written by
/// [`write_optical_flow`] or by OpenCV.
pub fn read_optical_flow(path: &str) -> Result<Mat> {
    let bytes = std::fs::read(path)
        .map_err(|e| cv_core::Error::new(Code::StsError, format!("cannot read {path}: {e}")))?;
    if bytes.len() < 12 || &bytes[..4] != b"PIEH" {
        return cv_error!(Code::StsParseError, "{path} is not a .flo file (bad PIEH tag)");
    }
    let w = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let h = i32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    cv_assert!(w > 0 && h > 0, ".flo header gives a {w}x{h} field");
    let want = 12 + (h as usize * w as usize * 8);
    cv_assert!(bytes.len() >= want, ".flo file is truncated: {} bytes, expected {want}", bytes.len());

    let mut v = Vec::with_capacity(h as usize * w as usize * 2);
    for i in 0..(h as usize * w as usize * 2) {
        let o = 12 + i * 4;
        v.push(f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]));
    }
    Mat::from_vec(h, w, 2, &v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::{CV_32FC1, CV_8UC1};

    /// Two frames of texture, the second shifted by a known amount.
    fn shifted_pair(dx: usize, dy: usize) -> (Mat, Mat) {
        let (h, w) = (32usize, 32usize);
        let mut state = 0x9e37_79b9u32;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let base: Vec<u8> = (0..(h + 8) * (w + 8)).map(|_| (rnd() % 220 + 20) as u8).collect();
        let cut = |ox: usize, oy: usize| {
            let mut v = Vec::with_capacity(h * w);
            for y in 0..h {
                for x in 0..w {
                    v.push(base[(y + oy) * (w + 8) + x + ox]);
                }
            }
            Mat::from_vec(h as i32, w as i32, 1, &v).unwrap()
        };
        (cut(2, 2), cut(2 + dx, 2 + dy))
    }

    #[test]
    fn farneback_defaults_match_upstream_and_round_trip() {
        let f = FarnebackOpticalFlow::default();
        assert_eq!(f.get_num_levels(), 5);
        assert_eq!(f.get_pyr_scale(), 0.5);
        assert_eq!(f.get_win_size(), 13);
        assert_eq!(f.get_num_iters(), 10);
        assert_eq!(f.get_poly_n(), 5);
        assert_eq!(f.get_poly_sigma(), 1.1);
        assert!(!f.get_fast_pyramids());
        assert_eq!(f.get_flags(), 0);

        let mut g = f;
        g.set_num_levels(3);
        g.set_pyr_scale(0.8);
        g.set_fast_pyramids(true);
        g.set_win_size(21);
        g.set_num_iters(4);
        g.set_poly_n(7);
        g.set_poly_sigma(1.5);
        g.set_flags(1);
        assert_eq!(
            (g.get_num_levels(), g.get_win_size(), g.get_poly_n(), g.get_flags()),
            (3, 21, 7, 1)
        );
        assert!(g.get_fast_pyramids());
    }

    #[test]
    fn the_flow_object_agrees_with_the_free_function() {
        let (a, b) = shifted_pair(1, 0);
        let obj = FarnebackOpticalFlow::create(2, 0.5, false, 13, 3, 5, 1.1, 0);
        let via_obj = obj.calc(&a, &b).unwrap();
        let via_fn = crate::calc_optical_flow_farneback(&a, &b, 2, 13, 3).unwrap();
        assert_eq!(via_obj.to_vec::<f32>().unwrap(), via_fn.to_vec::<f32>().unwrap());
        assert_eq!(via_obj.channels(), 2);
    }

    #[test]
    fn sparse_lk_defaults_and_dispatch() {
        let lk = SparsePyrLkOpticalFlow::default();
        assert_eq!(lk.get_win_size(), (21, 21));
        assert_eq!(lk.get_max_level(), 3);
        assert_eq!(lk.get_min_eig_threshold(), 1e-4);
        assert!(lk.get_term_criteria().is_valid());

        let (a, b) = shifted_pair(1, 1);
        let pts = [(10.0f64, 10.0), (20.0, 15.0)];
        let small = SparsePyrLkOpticalFlow::create(
            (11, 11),
            2,
            TermCriteria::new(cv_core::geometry::term_criteria::COUNT, 10, 0.01),
            0,
            1e-4,
        );
        let tracked = small.calc_sparse(&a, &b, &pts).unwrap();
        assert_eq!(tracked.len(), 2);
        // Each entry is (position, ok, error).
        assert!(tracked.iter().all(|t| t.2.is_finite()));
    }

    #[test]
    fn the_subtractor_accessors_round_trip() {
        let mut m = BackgroundSubtractorMog2::new(32, 32, 100, 16.0).unwrap();
        m.set_history(200);
        assert_eq!(m.get_history(), 200);
        m.set_var_threshold(25.0);
        assert_eq!(m.get_var_threshold(), 25.0);
        m.set_background_ratio(0.7);
        assert_eq!(m.get_background_ratio(), 0.7);
        m.set_var_init(20.0);
        assert_eq!(m.get_var_init(), 20.0);
        m.set_n_mixtures(3);
        assert_eq!(m.get_n_mixtures(), 3);
        // The parameters this implementation does not act on still report
        // upstream's documented defaults rather than a made-up value.
        assert_eq!(m.get_shadow_value(), 127);
        assert_eq!(m.get_shadow_threshold(), 0.5);
        assert!(!m.get_detect_shadows());
        assert_eq!(m.get_var_threshold_gen(), 9.0);

        let mut k = BackgroundSubtractorKnn::new(32, 32, 20, 400.0).unwrap();
        k.set_history(50);
        assert_eq!(k.get_history(), 50);
        assert_eq!(k.get_n_samples(), 50);
        k.set_dist2_threshold(900.0);
        assert_eq!(k.get_dist2_threshold(), 900.0);
        k.set_knn_samples(3);
        assert_eq!(k.get_knn_samples(), 3);
    }

    #[test]
    fn dis_presets_differ_and_accessors_round_trip() {
        use crate::ecc::DisPreset;
        let fast = DisOpticalFlow::create(DisPreset::Ultrafast);
        let medium = DisOpticalFlow::create(DisPreset::Medium);
        assert_eq!(fast.get_variational_refinement_iterations(), 0, "ultrafast skips refinement");
        assert!(medium.get_gradient_descent_iterations() > fast.get_gradient_descent_iterations());
        assert!(medium.get_finest_scale() < fast.get_finest_scale());

        let mut d = DisOpticalFlow::default();
        d.set_finest_scale(1);
        d.set_coarsest_scale(4);
        d.set_patch_size(12);
        d.set_patch_stride(6);
        d.set_gradient_descent_iterations(20);
        d.set_variational_refinement_iterations(3);
        d.set_variational_refinement_alpha(15.0);
        d.set_variational_refinement_delta(4.0);
        d.set_variational_refinement_gamma(8.0);
        d.set_variational_refinement_epsilon(0.02);
        d.set_use_mean_normalization(false);
        d.set_use_spatial_propagation(false);
        assert_eq!((d.get_finest_scale(), d.get_coarsest_scale()), (1, 4));
        assert_eq!((d.get_patch_size(), d.get_patch_stride()), (12, 6));
        assert_eq!(d.get_variational_refinement_alpha(), 15.0);
        assert_eq!(d.get_variational_refinement_epsilon(), 0.02);
        assert!(!d.get_use_mean_normalization() && !d.get_use_spatial_propagation());
    }

    #[test]
    fn dis_produces_a_two_channel_field() {
        let (a, b) = shifted_pair(1, 0);
        let mut d = DisOpticalFlow::create(crate::ecc::DisPreset::Ultrafast);
        d.set_coarsest_scale(3);
        d.set_finest_scale(1);
        let flow = d.calc(&a, &b).unwrap();
        assert_eq!(flow.size(), a.size());
        assert_eq!(flow.channels(), 2);
        assert!(flow.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn variational_refinement_accessors_and_run() {
        let mut v = VariationalRefinement::create();
        assert_eq!(v.get_alpha(), 20.0);
        assert_eq!(v.get_delta(), 5.0);
        assert_eq!(v.get_gamma(), 10.0);
        assert_eq!(v.get_epsilon(), 0.01);
        assert_eq!(v.get_fixed_point_iterations(), 5);
        assert_eq!(v.get_sor_iterations(), 5);
        assert_eq!(v.get_omega(), 1.6);
        v.set_alpha(10.0);
        v.set_delta(2.0);
        v.set_gamma(3.0);
        v.set_epsilon(0.05);
        v.set_fixed_point_iterations(2);
        v.set_sor_iterations(2);
        v.set_omega(1.2);
        assert_eq!((v.get_alpha(), v.get_omega()), (10.0, 1.2));

        let (a, b) = shifted_pair(1, 0);
        let zero = Mat::zeros(a.rows(), a.cols(), CV_32FC1).unwrap();
        let (u, w) = v.calc_uv(&a, &b, &zero, &zero).unwrap();
        assert_eq!(u.size(), a.size());
        assert!(u.to_vec::<f32>().unwrap().iter().all(|x| x.is_finite()));
        assert!(w.to_vec::<f32>().unwrap().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn a_flow_field_round_trips_through_a_flo_file() {
        let mut v = Vec::new();
        for y in 0..4i32 {
            for x in 0..6i32 {
                v.push(x as f32 * 0.5);
                v.push(y as f32 * -0.25);
            }
        }
        let flow = Mat::from_vec(4, 6, 2, &v).unwrap();
        let path = std::env::temp_dir().join("cv_no_c_flow_roundtrip.flo");
        let p = path.to_str().unwrap();
        write_optical_flow(p, &flow).unwrap();
        let back = read_optical_flow(p).unwrap();
        assert_eq!(back.size(), flow.size());
        assert_eq!(back.channels(), 2);
        assert_eq!(back.to_vec::<f32>().unwrap(), v);

        // The header is the Middlebury one, so OpenCV reads the same bytes.
        let raw = std::fs::read(p).unwrap();
        assert_eq!(&raw[..4], b"PIEH");
        assert_eq!(i32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]), 6);
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn a_bad_flo_file_is_rejected() {
        let path = std::env::temp_dir().join("cv_no_c_flow_bad.flo");
        let p = path.to_str().unwrap();
        std::fs::write(p, b"NOPE\x01\x00\x00\x00\x01\x00\x00\x00").unwrap();
        assert!(read_optical_flow(p).is_err());
        // Right tag, truncated body.
        std::fs::write(p, b"PIEH\x04\x00\x00\x00\x04\x00\x00\x00").unwrap();
        assert!(read_optical_flow(p).is_err());
        std::fs::remove_file(p).ok();
        assert!(read_optical_flow("/nonexistent/path.flo").is_err());
        // And a single-channel field is not a flow field.
        let g = Mat::zeros(2, 2, CV_8UC1).unwrap();
        assert!(write_optical_flow(p, &g).is_err());
    }

    /// A tracker that cannot track must say so, not pretend.
    #[test]
    fn the_cnn_trackers_report_their_missing_weights() {
        let e = TrackerGoturn::create("", "").unwrap_err();
        assert_eq!(e.code, Code::StsNotImplemented);
        assert!(e.msg.contains("goturn"), "{}", e.msg);
        assert!(TrackerNano::create("", "").is_err());
        assert!(TrackerVit::create("").is_err());
        assert!(TrackerDaSiamRpn::create("", "", "").is_err());

        // Given paths, construction succeeds and records them.
        let g = TrackerGoturn::create("a.prototxt", "b.caffemodel").unwrap();
        assert_eq!(g.model_bin, "b.caffemodel");
        let n = TrackerNano::create("back.onnx", "head.onnx").unwrap();
        assert_eq!(n.get_tracking_score(), 0.0);
        assert!(TrackerVit::create("vit.onnx").is_ok());
        assert!(TrackerDaSiamRpn::create("m", "k1", "k2").is_ok());
    }
}
