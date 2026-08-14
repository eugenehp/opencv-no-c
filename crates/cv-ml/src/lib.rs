//! `cv::ml` for `opencv-no-c` — classical statistical learning.
//!
//! # Scope
//!
//! `KNearest`, `NormalBayesClassifier`, `LogisticRegression`, a linear `SVM`
//! trained by SMO, `DTrees` and `RTrees`, `ANN_MLP` with backpropagation, and
//! `EM` by expectation-maximisation. That covers the classifiers people
//! actually reach for.
//!
//! The SVM carries all of upstream's kernels — linear, polynomial, RBF,
//! sigmoid, chi-squared and histogram intersection — and handles more than two
//! classes by one-vs-one voting, as upstream does.
//!
//! Boosting is [`Boost`], a two-class Real AdaBoost over shallow weighted
//! trees. The regression and one-class SVM variants (`EPS_SVR`, `NU_SVR`,
//! `ONE_CLASS`) are not here.
//!
//! Everything trains on `f64` and is deterministic given the same seed, so a
//! test can assert an exact result instead of a tolerance band.

#![warn(missing_docs)]

pub mod params;
pub mod statmodel;

use cv_core::{cv_assert, Code, Error, Result};

/// `cv::ml::TrainData` — a training set: one sample per row, one label per row.
///
/// The two required parts are [`samples`](Self::samples) and
/// [`responses`](Self::responses); everything else describes them and is empty
/// by default, meaning "no restriction". A held-out test split lives here
/// rather than in the caller because upstream's `calcError` needs to know which
/// rows a model was *not* trained on — see [`Self::set_train_test_split`].
///
/// The accessor surface upstream exposes is in [`crate::statmodel`].
#[derive(Debug, Clone, PartialEq)]
pub struct TrainData {
    /// Feature rows.
    pub samples: Vec<Vec<f64>>,
    /// Class index or regression target per row.
    pub responses: Vec<f64>,
    /// `varIdx` — which variables take part; empty means all of them.
    pub var_idx: Vec<i32>,
    /// `varType` — one entry per variable, plus a trailing one for the
    /// response. Empty means everything is ordered.
    pub var_type: Vec<params::VariableTypes>,
    /// `sampleWeights` — empty means uniform.
    pub sample_weights: Vec<f64>,
    /// The missing-value mask, `samples`-shaped. Empty means nothing is missing.
    pub missing: Vec<Vec<bool>>,
    /// Variable names, in variable order. Empty means unnamed.
    pub names: Vec<String>,
    /// Whether a sample is a row or a column.
    pub layout: params::SampleTypes,
    /// Row indices used for training; empty means every row.
    pub train_sample_idx: Vec<i32>,
    /// Row indices held out for testing; empty means none.
    pub test_sample_idx: Vec<i32>,
}

impl TrainData {
    /// A set with only the two required parts filled in.
    ///
    /// Skips the checks [`TrainData::create`] makes, so it is crate-private:
    /// every caller here has already established the shape.
    pub(crate) fn raw(samples: Vec<Vec<f64>>, responses: Vec<f64>) -> Self {
        TrainData {
            samples,
            responses,
            var_idx: Vec::new(),
            var_type: Vec::new(),
            sample_weights: Vec::new(),
            missing: Vec::new(),
            names: Vec::new(),
            layout: params::SampleTypes::RowSample,
            train_sample_idx: Vec::new(),
            test_sample_idx: Vec::new(),
        }
    }

    /// `cv::ml::TrainData::create(samples, layout, responses)`.
    pub fn create(samples: Vec<Vec<f64>>, responses: Vec<f64>) -> Result<Self> {
        cv_assert!(!samples.is_empty(), "a training set needs at least one sample");
        cv_assert!(
            samples.len() == responses.len(),
            "{} samples but {} responses",
            samples.len(),
            responses.len()
        );
        let d = samples[0].len();
        cv_assert!(d > 0, "samples must have at least one feature");
        for (i, s) in samples.iter().enumerate() {
            cv_assert!(s.len() == d, "sample {i} has {} features, expected {d}", s.len());
        }
        Ok(TrainData::raw(samples, responses))
    }

    /// Feature count.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.samples[0].len()
    }

    /// Sample count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the set is empty — never true for a constructed [`TrainData`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Distinct response values, ascending: the class labels.
    #[must_use]
    pub fn classes(&self) -> Vec<f64> {
        let mut c = self.responses.clone();
        c.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        c.dedup();
        c
    }
}

/// `cv::ml::KNearest`.
#[derive(Debug, Clone)]
pub struct KNearest {
    data: TrainData,
}

impl KNearest {
    /// `cv::ml::KNearest::train(trainData, ...)`.
    #[must_use]
    pub fn train(data: TrainData) -> Self {
        KNearest { data }
    }

    /// `cv::ml::KNearest::findNearest(samples, k, results, neighborResponses, dist)`.
    ///
    /// Returns the majority label among the `k` nearest. Ties go to the
    /// smallest label, which makes the result deterministic — upstream's
    /// tie-break is unspecified.
    pub fn find_nearest(&self, sample: &[f64], k: usize) -> Result<f64> {
        cv_assert!(k > 0, "k must be positive");
        cv_assert!(
            sample.len() == self.data.dim(),
            "sample has {} features, the model has {}",
            sample.len(),
            self.data.dim()
        );
        let mut d: Vec<(f64, f64)> = self
            .data
            .samples
            .iter()
            .zip(&self.data.responses)
            .map(|(s, r)| {
                (s.iter().zip(sample).map(|(a, b)| (a - b) * (a - b)).sum::<f64>(), *r)
            })
            .collect();
        d.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))
        });
        let mut votes: Vec<(f64, usize)> = Vec::new();
        for (_, label) in d.iter().take(k) {
            match votes.iter_mut().find(|(l, _)| l == label) {
                Some(v) => v.1 += 1,
                None => votes.push((*label, 1)),
            }
        }
        votes.sort_by(|a, b| {
            b.1.cmp(&a.1).then(a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal))
        });
        Ok(votes[0].0)
    }
}

/// `cv::ml::NormalBayesClassifier` — Gaussian naive Bayes.
#[derive(Debug, Clone)]
pub struct NormalBayes {
    classes: Vec<f64>,
    /// Per class: mean and variance of each feature, plus the log prior.
    stats: Vec<(Vec<f64>, Vec<f64>, f64)>,
}

impl NormalBayes {
    /// `cv::ml::NormalBayesClassifier::train(trainData, flags)`.
    pub fn train(data: &TrainData) -> Result<Self> {
        let classes = data.classes();
        cv_assert!(classes.len() >= 2, "Bayes needs at least two classes");
        let d = data.dim();
        let mut stats = Vec::with_capacity(classes.len());
        for c in &classes {
            let rows: Vec<&Vec<f64>> = data
                .samples
                .iter()
                .zip(&data.responses)
                .filter(|(_, r)| *r == c)
                .map(|(s, _)| s)
                .collect();
            let n = rows.len() as f64;
            let mut mean = vec![0.0; d];
            for r in &rows {
                for j in 0..d {
                    mean[j] += r[j];
                }
            }
            for m in mean.iter_mut() {
                *m /= n;
            }
            let mut var = vec![0.0; d];
            for r in &rows {
                for j in 0..d {
                    let dv = r[j] - mean[j];
                    var[j] += dv * dv;
                }
            }
            // A floor on the variance: a feature that is constant within a
            // class would otherwise give an infinite likelihood and swamp
            // every other feature.
            for v in var.iter_mut() {
                *v = (*v / n).max(1e-9);
            }
            stats.push((mean, var, (n / data.len() as f64).ln()));
        }
        Ok(NormalBayes { classes, stats })
    }

    /// `cv::ml::StatModel::predict(samples, results, flags)`.
    pub fn predict(&self, sample: &[f64]) -> Result<f64> {
        cv_assert!(
            sample.len() == self.stats[0].0.len(),
            "sample has {} features, the model has {}",
            sample.len(),
            self.stats[0].0.len()
        );
        let mut best = (f64::NEG_INFINITY, self.classes[0]);
        for (i, (mean, var, prior)) in self.stats.iter().enumerate() {
            let mut ll = *prior;
            for j in 0..sample.len() {
                let d = sample[j] - mean[j];
                ll += -0.5 * (2.0 * core::f64::consts::PI * var[j]).ln() - d * d / (2.0 * var[j]);
            }
            if ll > best.0 {
                best = (ll, self.classes[i]);
            }
        }
        Ok(best.1)
    }
}

/// `cv::ml::LogisticRegression` — binary, trained by gradient descent.
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    /// Weights, with the bias last.
    pub weights: Vec<f64>,
}

impl LogisticRegression {
    /// `cv::ml::LogisticRegression::train`. Labels must be 0 or 1.
    pub fn train(data: &TrainData, learning_rate: f64, iterations: usize) -> Result<Self> {
        cv_assert!(learning_rate > 0.0, "the learning rate must be positive");
        for r in &data.responses {
            cv_assert!(*r == 0.0 || *r == 1.0, "logistic regression needs 0/1 labels, saw {r}");
        }
        let d = data.dim();
        let mut w = vec![0.0f64; d + 1];
        let n = data.len() as f64;
        for _ in 0..iterations.max(1) {
            let mut grad = vec![0.0f64; d + 1];
            for (s, y) in data.samples.iter().zip(&data.responses) {
                let z: f64 = s.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() + w[d];
                let p = 1.0 / (1.0 + (-z).exp());
                let e = p - y;
                for j in 0..d {
                    grad[j] += e * s[j];
                }
                grad[d] += e;
            }
            for j in 0..=d {
                w[j] -= learning_rate * grad[j] / n;
            }
        }
        Ok(LogisticRegression { weights: w })
    }

    /// Probability of class 1.
    #[must_use]
    pub fn probability(&self, sample: &[f64]) -> f64 {
        let d = self.weights.len() - 1;
        let z: f64 =
            sample.iter().zip(&self.weights[..d]).map(|(a, b)| a * b).sum::<f64>() + self.weights[d];
        1.0 / (1.0 + (-z).exp())
    }

    /// `cv::ml::StatModel::predict` — the thresholded class.
    #[must_use]
    pub fn predict(&self, sample: &[f64]) -> f64 {
        if self.probability(sample) >= 0.5 {
            1.0
        } else {
            0.0
        }
    }
}

/// `cv::ml::SVM::KernelTypes`.
///
/// The kernel is what lets an SVM separate classes no straight line can. It
/// enters only through `K(x, y)`, so the training loop below never sees the
/// feature space it implicitly maps into — that is the whole trick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SvmKernel {
    /// `cv::ml::SVM::LINEAR` — `x . y`.
    Linear,
    /// `cv::ml::SVM::POLY` — `(gamma * x.y + coef0)^degree`.
    Poly {
        /// Polynomial degree.
        degree: f64,
        /// Scale on the inner product.
        gamma: f64,
        /// Additive constant.
        coef0: f64,
    },
    /// `cv::ml::SVM::RBF` — `exp(-gamma * |x - y|^2)`. Upstream's default.
    Rbf {
        /// Width; larger means a tighter, more local kernel.
        gamma: f64,
    },
    /// `cv::ml::SVM::SIGMOID` — `tanh(gamma * x.y + coef0)`.
    ///
    /// Only positive semi-definite for some parameter choices, so it is not a
    /// Mercer kernel in general and needs its `gamma`/`coef0` tuned. That is a
    /// tuning burden, not a capability limit: at `gamma = 0.5, coef0 = -2` it
    /// separates XOR cleanly, and at `gamma = 1, coef0 = -1` it does not.
    Sigmoid {
        /// Scale on the inner product.
        gamma: f64,
        /// Additive constant.
        coef0: f64,
    },
    /// `cv::ml::SVM::CHI2` — `exp(-gamma * chi2(x, y))`, for histograms.
    Chi2 {
        /// Width.
        gamma: f64,
    },
    /// `cv::ml::SVM::INTER` — histogram intersection, `sum(min(x, y))`.
    Inter,
}

impl Default for SvmKernel {
    /// Upstream's default is RBF with `gamma = 1`.
    fn default() -> Self {
        SvmKernel::Rbf { gamma: 1.0 }
    }
}

impl SvmKernel {
    /// Evaluate `K(a, b)`.
    #[must_use]
    pub fn eval(&self, a: &[f64], b: &[f64]) -> f64 {
        let dot = || a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        match *self {
            SvmKernel::Linear => dot(),
            SvmKernel::Poly { degree, gamma, coef0 } => (gamma * dot() + coef0).powf(degree),
            SvmKernel::Rbf { gamma } => {
                let d2: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
                (-gamma * d2).exp()
            }
            SvmKernel::Sigmoid { gamma, coef0 } => (gamma * dot() + coef0).tanh(),
            SvmKernel::Chi2 { gamma } => {
                let c: f64 = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| {
                        let s = x + y;
                        if s.abs() < 1e-12 {
                            0.0
                        } else {
                            (x - y) * (x - y) / s
                        }
                    })
                    .sum();
                (-gamma * c).exp()
            }
            SvmKernel::Inter => a.iter().zip(b).map(|(x, y)| x.min(*y)).sum(),
        }
    }
}

/// Training parameters for [`Svm`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvmParams {
    /// Kernel.
    pub kernel: SvmKernel,
    /// `cv::ml::SVM::setC` — the soft-margin penalty. Larger fits harder.
    pub c: f64,
    /// KKT tolerance; upstream's default termination epsilon.
    pub tolerance: f64,
    /// Cap on outer SMO passes.
    pub max_iter: usize,
    /// `cv::ml::SVM::setNu` — the `nu` of `NU_SVC`, `ONE_CLASS` and `NU_SVR`.
    pub nu: f64,
    /// `cv::ml::SVM::setP` — the epsilon of `EPS_SVR`'s insensitive loss.
    pub p: f64,
    /// Which SVM formulation to fit.
    ///
    /// Only `SvmType::CSvc` is trained here; the others are carried so a
    /// configuration round-trips, and [`Svm::train_with`] rejects them rather
    /// than quietly fitting a `C_SVC` under another name.
    pub svm_type: params::SvmType,
}

impl Default for SvmParams {
    fn default() -> Self {
        SvmParams {
            kernel: SvmKernel::default(),
            c: 1.0,
            tolerance: 1e-3,
            max_iter: 1000,
            nu: 0.5,
            p: 0.0,
            svm_type: params::SvmType::CSvc,
        }
    }
}

/// The generator that shuffles cross-validation folds.
///
/// Fixed seed, so `train_auto` is reproducible across runs and machines.
fn rng_for_folds() -> cv_core::rng::Rng {
    cv_core::rng::Rng::new(0x5F0D_C0DE)
}

/// Which kernel *family* to tune, without its parameters fixed.
///
/// [`Svm::train_auto`] searches the parameters, so it needs the family alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvmKernelKind {
    /// [`SvmKernel::Linear`] — nothing to tune but `C`.
    Linear,
    /// [`SvmKernel::Poly`] — tunes `degree`, `gamma`, `coef0`.
    Poly,
    /// [`SvmKernel::Rbf`] — tunes `gamma`.
    Rbf,
    /// [`SvmKernel::Sigmoid`] — tunes `gamma`, `coef0`.
    Sigmoid,
    /// [`SvmKernel::Chi2`] — tunes `gamma`.
    Chi2,
    /// [`SvmKernel::Inter`] — nothing to tune but `C`.
    Inter,
}

impl SvmKernelKind {
    /// Build a concrete kernel from one point of the parameter grid.
    fn build(self, gamma: f64, coef0: f64, degree: f64) -> SvmKernel {
        match self {
            SvmKernelKind::Linear => SvmKernel::Linear,
            SvmKernelKind::Poly => SvmKernel::Poly { degree, gamma, coef0 },
            SvmKernelKind::Rbf => SvmKernel::Rbf { gamma },
            SvmKernelKind::Sigmoid => SvmKernel::Sigmoid { gamma, coef0 },
            SvmKernelKind::Chi2 => SvmKernel::Chi2 { gamma },
            SvmKernelKind::Inter => SvmKernel::Inter,
        }
    }

    /// Whether this family uses each tunable, so the search skips dead axes.
    fn uses(self) -> (bool, bool, bool) {
        match self {
            SvmKernelKind::Linear | SvmKernelKind::Inter => (false, false, false),
            SvmKernelKind::Rbf | SvmKernelKind::Chi2 => (true, false, false),
            SvmKernelKind::Sigmoid => (true, true, false),
            SvmKernelKind::Poly => (true, true, true),
        }
    }
}

/// `cv::ml::ParamGrid` — the values one parameter takes during auto-tuning.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamGrid {
    /// Upstream's form: `min`, `min*step`, `min*step^2`, ... up to `max`.
    Log {
        /// Smallest value; must be positive, since the sweep is multiplicative.
        min: f64,
        /// Upper bound, inclusive.
        max: f64,
        /// Multiplier between successive values; must exceed 1.
        step: f64,
    },
    /// An explicit list.
    ///
    /// Not something upstream has, and it is here for a specific reason: a
    /// multiplicative grid starts at a positive minimum and multiplies, so it
    /// **cannot express a negative or zero value**. `cv::ml::SVM::trainAuto`
    /// therefore cannot reach `coef0 = -2`, which is the setting that makes a
    /// sigmoid kernel separate XOR — its default grid runs 0.1 to 300 and it
    /// rejects a grid whose minimum is negative. This variant closes that hole
    /// without changing what [`ParamGrid::Log`] does.
    Values(Vec<f64>),
    /// A single value; the axis is not searched.
    Fixed(f64),
}

impl ParamGrid {
    /// The values this grid sweeps.
    pub fn values(&self) -> Result<Vec<f64>> {
        Ok(match self {
            ParamGrid::Log { min, max, step } => {
                cv_assert!(*min > 0.0, "a log grid needs a positive minimum, got {min}");
                cv_assert!(*step > 1.0, "a log grid needs a step above 1, got {step}");
                cv_assert!(max >= min, "the grid maximum must not be below its minimum");
                let mut v = Vec::new();
                let mut x = *min;
                while x <= *max * (1.0 + 1e-12) {
                    v.push(x);
                    x *= step;
                }
                v
            }
            ParamGrid::Values(v) => {
                cv_assert!(!v.is_empty(), "an explicit grid needs at least one value");
                v.clone()
            }
            ParamGrid::Fixed(v) => vec![*v],
        })
    }

    /// `cv::ml::SVM::getDefaultGrid(C)` — 0.1 to 500, step 5.
    #[must_use]
    pub fn default_c() -> Self {
        ParamGrid::Log { min: 0.1, max: 500.0, step: 5.0 }
    }
    /// `cv::ml::SVM::getDefaultGrid(GAMMA)` — 1e-5 to 0.6, step 15.
    #[must_use]
    pub fn default_gamma() -> Self {
        ParamGrid::Log { min: 1e-5, max: 0.6, step: 15.0 }
    }
    /// `cv::ml::SVM::getDefaultGrid(DEGREE)` — 0.01 to 4, step 7.
    #[must_use]
    pub fn default_degree() -> Self {
        ParamGrid::Log { min: 0.01, max: 4.0, step: 7.0 }
    }
    /// Upstream's `COEF` grid: 0.1 to 300, step 14 — positive values only.
    #[must_use]
    pub fn upstream_coef0() -> Self {
        ParamGrid::Log { min: 0.1, max: 300.0, step: 14.0 }
    }
    /// The `COEF` grid this port searches by default.
    ///
    /// Upstream's positive-only sweep with the negative half added. That is
    /// the difference between a sigmoid kernel that separates XOR and one that
    /// does not, and it costs nothing when the answer is positive.
    #[must_use]
    pub fn default_coef0() -> Self {
        ParamGrid::Values(vec![
            -300.0, -100.0, -30.0, -10.0, -3.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0,
            3.0, 10.0, 30.0, 100.0, 300.0,
        ])
    }
}

/// Search space for [`Svm::train_auto`].
#[derive(Debug, Clone, PartialEq)]
pub struct AutoParams {
    /// Kernel family to tune.
    pub kernel: SvmKernelKind,
    /// Cross-validation folds; upstream's default is 10.
    pub k_fold: usize,
    /// Grid over `C`.
    pub c: ParamGrid,
    /// Grid over `gamma`.
    pub gamma: ParamGrid,
    /// Grid over `coef0`.
    pub coef0: ParamGrid,
    /// Grid over the polynomial degree.
    pub degree: ParamGrid,
    /// KKT tolerance passed to each fit.
    pub tolerance: f64,
    /// Cap on SMO passes per fit.
    pub max_iter: usize,
}

impl AutoParams {
    /// Upstream's defaults for a kernel family, with this port's `coef0` grid.
    #[must_use]
    pub fn for_kernel(kernel: SvmKernelKind) -> Self {
        AutoParams {
            kernel,
            k_fold: 10,
            c: ParamGrid::default_c(),
            gamma: ParamGrid::default_gamma(),
            coef0: ParamGrid::default_coef0(),
            degree: ParamGrid::default_degree(),
            tolerance: 1e-3,
            max_iter: 1000,
        }
    }
}

/// One trained binary SVM.
#[derive(Debug, Clone)]
struct BinarySvm {
    alpha: Vec<f64>,
    sv: Vec<Vec<f64>>,
    y: Vec<f64>,
    b: f64,
    kernel: SvmKernel,
}

impl BinarySvm {
    /// Signed distance to the decision surface.
    fn decision(&self, x: &[f64]) -> f64 {
        self.alpha
            .iter()
            .zip(&self.y)
            .zip(&self.sv)
            .map(|((a, yy), s)| a * yy * self.kernel.eval(s, x))
            .sum::<f64>()
            + self.b
    }
}

/// Platt's SMO over labels already reduced to -1 and +1.
///
/// # Why the full algorithm rather than the simplified one
///
/// The textbook "simplified SMO" picks the second multiplier at random and
/// recomputes every error from scratch. That is `O(n)` per evaluation and
/// converges slowly enough that on a few hundred samples it silently stops
/// short of the optimum. This keeps an error cache and picks the second
/// multiplier to maximise `|E_i - E_j|`, which is what makes the step sizes
/// large and the iteration count bounded.
fn smo(samples: &[Vec<f64>], y: &[f64], p: &SvmParams) -> Result<BinarySvm> {
    let n = samples.len();
    let k = |i: usize, j: usize| p.kernel.eval(&samples[i], &samples[j]);
    let mut alpha = vec![0.0f64; n];
    let mut b = 0.0f64;
    // E[i] = f(x_i) - y_i. With alpha all zero and b zero, f is zero.
    let mut e: Vec<f64> = y.iter().map(|v| -v).collect();

    let take_step = |i: usize,
                     j: usize,
                     alpha: &mut Vec<f64>,
                     b: &mut f64,
                     e: &mut Vec<f64>|
     -> bool {
        if i == j {
            return false;
        }
        let (ai, aj) = (alpha[i], alpha[j]);
        let (yi, yj) = (y[i], y[j]);
        let (lo, hi) = if yi == yj {
            ((ai + aj - p.c).max(0.0), (ai + aj).min(p.c))
        } else {
            ((aj - ai).max(0.0), (p.c + aj - ai).min(p.c))
        };
        if hi - lo < 1e-12 {
            return false;
        }
        let (kii, kjj, kij) = (k(i, i), k(j, j), k(i, j));
        let eta = 2.0 * kij - kii - kjj;
        let mut nj = if eta < -1e-12 {
            (aj - yj * (e[i] - e[j]) / eta).clamp(lo, hi)
        } else {
            // A non-negative eta means the objective is not concave along this
            // direction; evaluate both endpoints and take the better one.
            let f_at = |a: f64| -> f64 {
                let d = a - aj;
                let di = -yi * yj * d;
                0.5 * kii * di * di
                    + 0.5 * kjj * d * d
                    + yi * yj * kij * di * d
                    + di * (e[i] + y[i])
                    + d * (e[j] + y[j])
            };
            if f_at(lo) < f_at(hi) - 1e-12 {
                lo
            } else if f_at(hi) < f_at(lo) - 1e-12 {
                hi
            } else {
                return false;
            }
        };
        if (nj - aj).abs() < 1e-12 * (nj + aj + 1e-12) {
            return false;
        }
        // Clamp the tiny numerical excursions past the box.
        if nj < 1e-10 {
            nj = 0.0;
        } else if nj > p.c - 1e-10 {
            nj = p.c;
        }
        let ni = ai + yi * yj * (aj - nj);

        let b1 = *b - e[i] - yi * (ni - ai) * kii - yj * (nj - aj) * kij;
        let b2 = *b - e[j] - yi * (ni - ai) * kij - yj * (nj - aj) * kjj;
        let nb = if ni > 0.0 && ni < p.c {
            b1
        } else if nj > 0.0 && nj < p.c {
            b2
        } else {
            (b1 + b2) * 0.5
        };
        let db = nb - *b;
        // Update every cached error by the change this step caused.
        for (t, et) in e.iter_mut().enumerate() {
            *et += yi * (ni - ai) * k(i, t) + yj * (nj - aj) * k(j, t) + db;
        }
        alpha[i] = ni;
        alpha[j] = nj;
        *b = nb;
        true
    };

    let violates = |i: usize, alpha: &[f64], e: &[f64]| -> bool {
        let r = e[i] * y[i];
        (r < -p.tolerance && alpha[i] < p.c) || (r > p.tolerance && alpha[i] > 0.0)
    };

    let mut examine_all = true;
    let mut changed = 0usize;
    let mut iters = 0usize;
    while (examine_all || changed > 0) && iters < p.max_iter {
        iters += 1;
        changed = 0;
        let pool: Vec<usize> = if examine_all {
            (0..n).collect()
        } else {
            (0..n).filter(|&i| alpha[i] > 0.0 && alpha[i] < p.c).collect()
        };
        for i in pool {
            if !violates(i, &alpha, &e) {
                continue;
            }
            // Second choice: the multiplier with the most distant error.
            let mut best = (0.0f64, usize::MAX);
            for j in 0..n {
                if j == i {
                    continue;
                }
                let d = (e[i] - e[j]).abs();
                if d > best.0 {
                    best = (d, j);
                }
            }
            let mut stepped = false;
            if best.1 != usize::MAX {
                stepped = take_step(i, best.1, &mut alpha, &mut b, &mut e);
            }
            // Fall back to a sweep if the heuristic pick did not move.
            if !stepped {
                for j in 0..n {
                    if take_step(i, j, &mut alpha, &mut b, &mut e) {
                        stepped = true;
                        break;
                    }
                }
            }
            if stepped {
                changed += 1;
            }
        }
        examine_all = if examine_all { false } else { changed == 0 };
    }

    let keep: Vec<usize> = (0..n).filter(|&i| alpha[i] > 1e-8).collect();
    if keep.is_empty() {
        return Err(Error::new(
            Code::StsError,
            "SVM training produced no support vectors; try a larger C or a different kernel"
                .to_string(),
        ));
    }
    Ok(BinarySvm {
        alpha: keep.iter().map(|&i| alpha[i]).collect(),
        sv: keep.iter().map(|&i| samples[i].clone()).collect(),
        y: keep.iter().map(|&i| y[i]).collect(),
        b,
        kernel: p.kernel,
    })
}

/// `cv::ml::SVM`.
///
/// Binary or multi-class, with any of upstream's kernels. Multi-class uses
/// one-vs-one voting, which is what upstream does — one-vs-all needs
/// calibrated scores across independently-trained models and gets it wrong on
/// unbalanced classes.
#[derive(Debug, Clone)]
pub struct Svm {
    /// Class labels in ascending order.
    labels: Vec<f64>,
    /// One classifier per label pair, with the pair it separates.
    machines: Vec<(usize, usize, BinarySvm)>,
    /// Per-class penalty multipliers; empty means uniform.
    pub(crate) class_weights: Vec<f64>,
}

impl Svm {
    /// `cv::ml::SVM::train(trainData, flags)` with explicit parameters.
    pub fn train_with(data: &TrainData, p: SvmParams) -> Result<Self> {
        cv_assert!(p.c > 0.0, "C must be positive, got {}", p.c);
        cv_assert!(p.tolerance > 0.0, "the tolerance must be positive");
        cv_assert!(
            p.svm_type == params::SvmType::CSvc,
            "only C_SVC is trained here; {:?} would need a different SMO objective",
            p.svm_type
        );
        let labels = data.classes();
        cv_assert!(labels.len() >= 2, "an SVM needs at least two classes, got {}", labels.len());

        let mut machines = Vec::new();
        for a in 0..labels.len() {
            for bidx in a + 1..labels.len() {
                // Only the two classes of this pair take part.
                let rows: Vec<(Vec<f64>, f64)> = data
                    .samples
                    .iter()
                    .zip(&data.responses)
                    .filter(|(_, r)| **r == labels[a] || **r == labels[bidx])
                    .map(|(s, r)| (s.clone(), if *r == labels[bidx] { 1.0 } else { -1.0 }))
                    .collect();
                let xs: Vec<Vec<f64>> = rows.iter().map(|(s, _)| s.clone()).collect();
                let ys: Vec<f64> = rows.iter().map(|(_, y)| *y).collect();
                machines.push((a, bidx, smo(&xs, &ys, &p)?));
            }
        }
        Ok(Svm { labels, machines, class_weights: Vec::new() })
    }

    /// `cv::ml::SVM::train` with a linear kernel, keeping the older signature.
    pub fn train(data: &TrainData, c: f64, max_iter: usize) -> Result<Self> {
        Self::train_with(
            data,
            SvmParams { kernel: SvmKernel::Linear, c, max_iter, ..Default::default() },
        )
    }

    /// Signed distance to the decision surface.
    ///
    /// Only meaningful for a binary model; for multi-class it is the first
    /// pair's margin, which is why [`Svm::predict`] votes instead.
    #[must_use]
    pub fn decision(&self, x: &[f64]) -> f64 {
        self.machines.first().map_or(0.0, |(_, _, m)| m.decision(x))
    }

    /// `cv::ml::StatModel::predict`.
    #[must_use]
    pub fn predict(&self, x: &[f64]) -> f64 {
        if self.machines.len() == 1 {
            let (a, b, m) = &self.machines[0];
            return if m.decision(x) >= 0.0 { self.labels[*b] } else { self.labels[*a] };
        }
        let mut votes = vec![0usize; self.labels.len()];
        for (a, b, m) in &self.machines {
            if m.decision(x) >= 0.0 {
                votes[*b] += 1;
            } else {
                votes[*a] += 1;
            }
        }
        // Ties go to the smallest label, so the result is deterministic.
        let best = votes
            .iter()
            .enumerate()
            .max_by(|p, q| p.1.cmp(q.1).then(q.0.cmp(&p.0)))
            .map_or(0, |(i, _)| i);
        self.labels[best]
    }

    /// `cv::ml::SVM::trainAuto(trainData, kFold, Cgrid, gammaGrid, ...)`.
    ///
    /// Grid search with k-fold cross-validation, then a refit on all the data
    /// at the winning point. Returns the model and the parameters chosen.
    ///
    /// # Fold assignment
    ///
    /// Folds come from a *seeded shuffle*, not from a stride. A stride
    /// (`i % k_fold`) is reproducible but aliases: on data whose class pattern
    /// repeats with a period that shares a factor with `k_fold`, one fold can
    /// take every instance of a class and the training split then never sees
    /// it. Four-fold cross-validation over a repeated XOR set does exactly
    /// that. Shuffling from a fixed seed keeps the run reproducible — upstream
    /// shuffles from its global RNG and is not — while removing the alias.
    pub fn train_auto(data: &TrainData, auto: &AutoParams) -> Result<(Self, SvmParams)> {
        cv_assert!(auto.k_fold >= 2, "cross-validation needs at least 2 folds");
        cv_assert!(
            data.len() >= auto.k_fold,
            "{} samples cannot make {} folds",
            data.len(),
            auto.k_fold
        );
        let (use_g, use_c0, use_deg) = auto.kernel.uses();
        let cs = auto.c.values()?;
        let gammas = if use_g { auto.gamma.values()? } else { vec![1.0] };
        let coef0s = if use_c0 { auto.coef0.values()? } else { vec![0.0] };
        let degrees = if use_deg { auto.degree.values()? } else { vec![1.0] };

        let mut order: Vec<usize> = (0..data.len()).collect();
        let mut rng = crate::rng_for_folds();
        for i in (1..order.len()).rev() {
            order.swap(i, (rng.next_u32() as usize) % (i + 1));
        }
        let mut folds = vec![0usize; data.len()];
        for (slot, &idx) in order.iter().enumerate() {
            folds[idx] = slot % auto.k_fold;
        }
        let mut best: Option<(f64, SvmParams)> = None;

        for &c in &cs {
            for &g in &gammas {
                for &c0 in &coef0s {
                    for &deg in &degrees {
                        let p = SvmParams {
                            kernel: auto.kernel.build(g, c0, deg),
                            c,
                            tolerance: auto.tolerance,
                            max_iter: auto.max_iter,
                            ..SvmParams::default()
                        };
                        let mut hits = 0usize;
                        let mut total = 0usize;
                        for f in 0..auto.k_fold {
                            let tr = TrainData::raw(
                                data.samples
                                    .iter()
                                    .zip(&folds)
                                    .filter(|(_, &ff)| ff != f)
                                    .map(|(s, _)| s.clone())
                                    .collect(),
                                data.responses
                                    .iter()
                                    .zip(&folds)
                                    .filter(|(_, &ff)| ff != f)
                                    .map(|(r, _)| *r)
                                    .collect(),
                            );
                            // A fold that lost a whole class cannot train; skip
                            // it rather than letting the error kill the search.
                            if tr.samples.is_empty() || tr.classes().len() < 2 {
                                continue;
                            }
                            let Ok(m) = Svm::train_with(&tr, p) else { continue };
                            for (s, r) in data.samples.iter().zip(&data.responses).zip(&folds).filter_map(
                                |((s, r), &ff)| if ff == f { Some((s, r)) } else { None },
                            ) {
                                total += 1;
                                if m.predict(s) == *r {
                                    hits += 1;
                                }
                            }
                        }
                        if total == 0 {
                            continue;
                        }
                        let score = hits as f64 / total as f64;
                        // Strictly greater, so the first point wins a tie and
                        // the search is order-deterministic.
                        if best.as_ref().map_or(true, |(bs, _)| score > *bs) {
                            best = Some((score, p));
                        }
                    }
                }
            }
        }
        let (_, p) = best.ok_or_else(|| {
            Error::new(Code::StsError, "trainAuto: no parameter point could be fitted".to_string())
        })?;
        Ok((Svm::train_with(data, p)?, p))
    }

    /// `cv::ml::SVM::getSupportVectors()` — the union across all pairs.
    #[must_use]
    pub fn support_vectors(&self) -> Vec<Vec<f64>> {
        let mut out: Vec<Vec<f64>> = Vec::new();
        for (_, _, m) in &self.machines {
            for s in &m.sv {
                if !out.contains(s) {
                    out.push(s.clone());
                }
            }
        }
        out
    }

    /// Number of one-vs-one classifiers; 1 for a binary problem.
    #[must_use]
    pub fn machine_count(&self) -> usize {
        self.machines.len()
    }

    /// The class labels this model was trained on.
    #[must_use]
    pub fn labels(&self) -> &[f64] {
        &self.labels
    }
}

/// A node of a [`DTree`].
#[derive(Debug, Clone)]
enum TreeNode {
    Leaf(f64),
    Split { feature: usize, at: f64, left: Box<TreeNode>, right: Box<TreeNode> },
}

/// `cv::ml::DTrees` — a CART classification tree on Gini impurity.
#[derive(Debug, Clone)]
pub struct DTree {
    root: TreeNode,
    /// Depth actually reached.
    pub depth: usize,
}

/// Gini impurity of a label multiset.
fn gini(labels: &[f64]) -> f64 {
    if labels.is_empty() {
        return 0.0;
    }
    let mut counts: Vec<(f64, usize)> = Vec::new();
    for l in labels {
        match counts.iter_mut().find(|(v, _)| v == l) {
            Some(c) => c.1 += 1,
            None => counts.push((*l, 1)),
        }
    }
    let n = labels.len() as f64;
    1.0 - counts.iter().map(|(_, c)| (*c as f64 / n).powi(2)).sum::<f64>()
}

/// Most frequent label, ties to the smallest.
fn majority(labels: &[f64]) -> f64 {
    let mut counts: Vec<(f64, usize)> = Vec::new();
    for l in labels {
        match counts.iter_mut().find(|(v, _)| v == l) {
            Some(c) => c.1 += 1,
            None => counts.push((*l, 1)),
        }
    }
    counts.sort_by(|a, b| {
        b.1.cmp(&a.1).then(a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal))
    });
    counts[0].0
}

impl DTree {
    /// `cv::ml::DTrees::train(trainData, flags)`.
    pub fn train(data: &TrainData, max_depth: usize, min_samples: usize) -> Result<Self> {
        cv_assert!(min_samples >= 1, "minSampleCount must be at least 1");
        let idx: Vec<usize> = (0..data.len()).collect();
        let mut reached = 0;
        let root = Self::build(data, &idx, 0, max_depth.max(1), min_samples, None, &mut reached);
        Ok(DTree { root, depth: reached })
    }

    /// Build one subtree. `features` restricts the candidate split columns,
    /// which is how [`RTrees`] injects randomness.
    fn build(
        data: &TrainData,
        idx: &[usize],
        depth: usize,
        max_depth: usize,
        min_samples: usize,
        features: Option<&[usize]>,
        reached: &mut usize,
    ) -> TreeNode {
        *reached = (*reached).max(depth);
        let labels: Vec<f64> = idx.iter().map(|&i| data.responses[i]).collect();
        let impurity = gini(&labels);
        if depth >= max_depth || idx.len() < min_samples * 2 || impurity <= 0.0 {
            return TreeNode::Leaf(majority(&labels));
        }

        let all: Vec<usize> = (0..data.dim()).collect();
        let cols = features.unwrap_or(&all);
        let mut best: Option<(f64, usize, f64)> = None;
        for &f in cols {
            let mut vals: Vec<f64> = idx.iter().map(|&i| data.samples[i][f]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            vals.dedup();
            for w in vals.windows(2) {
                let at = (w[0] + w[1]) * 0.5;
                let (l, r): (Vec<f64>, Vec<f64>) = idx
                    .iter()
                    .map(|&i| (data.samples[i][f], data.responses[i]))
                    .fold((Vec::new(), Vec::new()), |mut acc, (v, y)| {
                        if v < at {
                            acc.0.push(y);
                        } else {
                            acc.1.push(y);
                        }
                        acc
                    });
                if l.len() < min_samples || r.len() < min_samples {
                    continue;
                }
                let n = idx.len() as f64;
                let g = (l.len() as f64 / n) * gini(&l) + (r.len() as f64 / n) * gini(&r);
                if best.as_ref().map_or(true, |(bg, _, _)| g < *bg) {
                    best = Some((g, f, at));
                }
            }
        }
        let Some((g, f, at)) = best else {
            return TreeNode::Leaf(majority(&labels));
        };
        if g >= impurity {
            return TreeNode::Leaf(majority(&labels)); // no split helps
        }
        let (li, ri): (Vec<usize>, Vec<usize>) =
            idx.iter().partition(|&&i| data.samples[i][f] < at);
        TreeNode::Split {
            feature: f,
            at,
            left: Box::new(Self::build(data, &li, depth + 1, max_depth, min_samples, features, reached)),
            right: Box::new(Self::build(data, &ri, depth + 1, max_depth, min_samples, features, reached)),
        }
    }

    /// `cv::ml::StatModel::predict`.
    #[must_use]
    pub fn predict(&self, sample: &[f64]) -> f64 {
        let mut n = &self.root;
        loop {
            match n {
                TreeNode::Leaf(v) => return *v,
                TreeNode::Split { feature, at, left, right } => {
                    n = if sample[*feature] < *at { left } else { right };
                }
            }
        }
    }
}


/// One node of a weighted decision tree, as [`Boost`] grows them.
#[derive(Debug, Clone)]
enum WeakNode {
    /// The stage's contribution to the score for a sample landing here.
    Leaf(f64),
    /// `sample[feature] < at` goes left.
    Split { feature: usize, at: f64, left: Box<WeakNode>, right: Box<WeakNode> },
}

impl WeakNode {
    /// The score this tree assigns to `sample`.
    fn eval(&self, sample: &[f64]) -> f64 {
        let mut n = self;
        loop {
            match n {
                WeakNode::Leaf(v) => return *v,
                WeakNode::Split { feature, at, left, right } => {
                    n = if sample[*feature] < *at { left } else { right };
                }
            }
        }
    }
}

/// `cv::ml::Boost` — Real AdaBoost over shallow decision trees.
///
/// Two classes only, which is upstream's restriction too: boosting's weight
/// update is defined on a signed margin, and there is no signed margin for
/// three labels. Multi-class boosting exists but is a different algorithm.
///
/// Each round fits a depth-limited tree to the *weighted* sample, scores every
/// leaf with half the log-odds of the weight that reached it, and then
/// re-weights each sample by `exp(-y * f(x))` — so a sample the ensemble
/// already gets right stops mattering and the next tree spends its capacity on
/// the ones it does not. That is the whole idea, and it is why a boosted stump
/// beats a single deep tree on data with a bit of structure in it.
#[derive(Debug, Clone)]
pub struct Boost {
    stages: Vec<WeakNode>,
    /// The two class labels, in the order the signed margin uses.
    classes: [f64; 2],
}

impl Boost {
    /// `cv::ml::Boost::create()` plus `train`, with `BOOST_REAL`.
    ///
    /// `weak_count` rounds of trees no deeper than `max_depth`.
    pub fn train(data: &TrainData, weak_count: usize, max_depth: usize) -> Result<Self> {
        cv_assert!(weak_count > 0, "boosting needs at least one weak learner");
        cv_assert!(max_depth >= 1, "a weak tree needs at least one level");
        let classes = data.classes();
        cv_assert!(
            classes.len() == 2,
            "Boost handles two classes, got {}: use RTrees or SVM for more",
            classes.len()
        );
        let (neg, pos) = (classes[0], classes[1]);
        let n = data.len();
        // +1 for the second class, -1 for the first.
        let y: Vec<f64> =
            data.responses.iter().map(|&r| if r == pos { 1.0 } else { -1.0 }).collect();
        let mut w = vec![1.0 / n as f64; n];
        let idx: Vec<usize> = (0..n).collect();

        let mut stages = Vec::with_capacity(weak_count);
        for _ in 0..weak_count {
            let tree = Self::build(&data.samples, &y, &w, &idx, 0, max_depth);
            // Re-weight, then renormalise so the weights stay a distribution.
            let mut total = 0.0;
            for i in 0..n {
                w[i] *= (-y[i] * tree.eval(&data.samples[i])).exp();
                total += w[i];
            }
            if total <= 0.0 || !total.is_finite() {
                // Every sample is classified with certainty; more rounds add
                // nothing and the weights have underflowed.
                stages.push(tree);
                break;
            }
            for wi in &mut w {
                *wi /= total;
            }
            stages.push(tree);
        }
        Ok(Boost { stages, classes: [neg, pos] })
    }

    /// Grow one weighted tree by weighted Gini impurity.
    fn build(
        x: &[Vec<f64>],
        y: &[f64],
        w: &[f64],
        idx: &[usize],
        depth: usize,
        max_depth: usize,
    ) -> WeakNode {
        // Half the log-odds of the weight that reached this leaf. The clamp is
        // what stops a pure leaf becoming infinite and taking every later
        // weight to zero with it.
        let leaf = |idx: &[usize]| -> WeakNode {
            let (mut wp, mut wn) = (0.0f64, 0.0f64);
            for &i in idx {
                if y[i] > 0.0 {
                    wp += w[i];
                } else {
                    wn += w[i];
                }
            }
            const EPS: f64 = 1e-10;
            WeakNode::Leaf((0.5 * ((wp + EPS) / (wn + EPS)).ln()).clamp(-4.0, 4.0))
        };
        if depth >= max_depth || idx.len() < 2 {
            return leaf(idx);
        }

        let dim = x[0].len();
        let gini = |wp: f64, wn: f64| -> f64 {
            let t = wp + wn;
            if t <= 0.0 {
                0.0
            } else {
                t * (1.0 - (wp / t).powi(2) - (wn / t).powi(2))
            }
        };
        let mut best: Option<(f64, usize, f64)> = None;
        #[allow(clippy::needless_range_loop)] // `f` indexes every sample's row, not one slice
        for f in 0..dim {
            let mut order: Vec<usize> = idx.to_vec();
            order.sort_by(|&a, &b| x[a][f].total_cmp(&x[b][f]));
            let (mut lp, mut ln) = (0.0f64, 0.0f64);
            let (mut rp, mut rn) = (0.0f64, 0.0f64);
            for &i in &order {
                if y[i] > 0.0 {
                    rp += w[i];
                } else {
                    rn += w[i];
                }
            }
            for k in 0..order.len() - 1 {
                let i = order[k];
                if y[i] > 0.0 {
                    lp += w[i];
                    rp -= w[i];
                } else {
                    ln += w[i];
                    rn -= w[i];
                }
                let (a, b) = (x[i][f], x[order[k + 1]][f]);
                if a == b {
                    continue;
                }
                let score = gini(lp, ln) + gini(rp, rn);
                if best.is_none() || best.is_some_and(|(s, _, _)| score < s) {
                    best = Some((score, f, 0.5 * (a + b)));
                }
            }
        }
        let Some((_, f, at)) = best else { return leaf(idx) };
        let (l, r): (Vec<usize>, Vec<usize>) = idx.iter().partition(|&&i| x[i][f] < at);
        if l.is_empty() || r.is_empty() {
            return leaf(idx);
        }
        WeakNode::Split {
            feature: f,
            at,
            left: Box::new(Self::build(x, y, w, &l, depth + 1, max_depth)),
            right: Box::new(Self::build(x, y, w, &r, depth + 1, max_depth)),
        }
    }

    /// The summed margin over every round; positive picks the second class.
    #[must_use]
    pub fn decision_function(&self, sample: &[f64]) -> f64 {
        self.stages.iter().map(|s| s.eval(sample)).sum()
    }

    /// `cv::ml::StatModel::predict(samples, results, flags)`.
    #[must_use]
    pub fn predict(&self, sample: &[f64]) -> f64 {
        if self.decision_function(sample) > 0.0 {
            self.classes[1]
        } else {
            self.classes[0]
        }
    }

    /// `cv::ml::Boost::getWeakCount()` — how many rounds actually ran.
    #[must_use]
    pub fn weak_count(&self) -> usize {
        self.stages.len()
    }
}

/// `cv::ml::RTrees` — a bagged forest of [`DTree`]s with random feature subsets.
#[derive(Debug, Clone)]
pub struct RTrees {
    trees: Vec<DTree>,
}

impl RTrees {
    /// `cv::ml::RTrees::train`.
    ///
    /// `rng` drives both the bootstrap sampling and the per-node feature
    /// subset, so a forest is reproducible from its seed.
    pub fn train(
        data: &TrainData,
        n_trees: usize,
        max_depth: usize,
        rng: &mut cv_core::rng::Rng,
    ) -> Result<Self> {
        cv_assert!(n_trees > 0, "a forest needs at least one tree");
        let d = data.dim();
        // sqrt(d) features per tree is the usual default.
        let k = ((d as f64).sqrt().round() as usize).clamp(1, d);
        let mut trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            // Bootstrap sample.
            let boot: Vec<usize> =
                (0..data.len()).map(|_| (rng.next_u32() as usize) % data.len()).collect();
            let sub = TrainData::raw(
                boot.iter().map(|&i| data.samples[i].clone()).collect(),
                boot.iter().map(|&i| data.responses[i]).collect(),
            );
            let mut feats: Vec<usize> = (0..d).collect();
            // Partial Fisher-Yates for the feature subset.
            for i in 0..k {
                let j = i + (rng.next_u32() as usize) % (d - i);
                feats.swap(i, j);
            }
            let idx: Vec<usize> = (0..sub.len()).collect();
            let mut reached = 0;
            let root = DTree::build(&sub, &idx, 0, max_depth.max(1), 1, Some(&feats[..k]), &mut reached);
            trees.push(DTree { root, depth: reached });
        }
        Ok(RTrees { trees })
    }

    /// Majority vote across the forest.
    #[must_use]
    pub fn predict(&self, sample: &[f64]) -> f64 {
        let votes: Vec<f64> = self.trees.iter().map(|t| t.predict(sample)).collect();
        majority(&votes)
    }

    /// Number of trees.
    #[must_use]
    pub fn len(&self) -> usize {
        self.trees.len()
    }

    /// Whether the forest is empty — never true for a trained one.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }
}

/// `cv::ml::ANN_MLP` — one hidden layer, sigmoid activation, backpropagation.
#[derive(Debug, Clone)]
pub struct AnnMlp {
    /// Input-to-hidden weights, `hidden x (inputs + 1)`.
    w1: Vec<Vec<f64>>,
    /// Hidden-to-output weights, `outputs x (hidden + 1)`.
    w2: Vec<Vec<f64>>,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl AnnMlp {
    /// `cv::ml::ANN_MLP::create()` then `setLayerSizes`, with random weights.
    pub fn new(inputs: usize, hidden: usize, outputs: usize, rng: &mut cv_core::rng::Rng) -> Result<Self> {
        cv_assert!(inputs > 0 && hidden > 0 && outputs > 0, "every layer needs at least one unit");
        let mut w = |fan_in: usize| -> Vec<f64> {
            // Xavier-ish scaling; zero-initialised weights would leave every
            // hidden unit computing the same function forever.
            let s = (1.0 / fan_in as f64).sqrt();
            (0..fan_in + 1).map(|_| rng.uniform_f64(-s, s)).collect()
        };
        Ok(AnnMlp {
            w1: (0..hidden).map(|_| w(inputs)).collect(),
            w2: (0..outputs).map(|_| w(hidden)).collect(),
        })
    }

    /// Forward pass, returning the hidden and output activations.
    fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let h: Vec<f64> = self
            .w1
            .iter()
            .map(|w| sigmoid(x.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() + w[x.len()]))
            .collect();
        let o: Vec<f64> = self
            .w2
            .iter()
            .map(|w| sigmoid(h.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() + w[h.len()]))
            .collect();
        (h, o)
    }

    /// `cv::ml::ANN_MLP::train` — backpropagation over `targets`, one row per
    /// sample, each row the desired output vector.
    pub fn train(
        &mut self,
        samples: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
    ) -> Result<()> {
        cv_assert!(samples.len() == targets.len(), "samples and targets must match in count");
        cv_assert!(learning_rate > 0.0, "the learning rate must be positive");
        for _ in 0..epochs.max(1) {
            for (x, t) in samples.iter().zip(targets) {
                let (h, o) = self.forward(x);
                // Output deltas.
                let od: Vec<f64> =
                    o.iter().zip(t).map(|(oo, tt)| (oo - tt) * oo * (1.0 - oo)).collect();
                // Hidden deltas.
                let hd: Vec<f64> = h
                    .iter()
                    .enumerate()
                    .map(|(j, hh)| {
                        let s: f64 = od.iter().zip(&self.w2).map(|(d, w)| d * w[j]).sum();
                        s * hh * (1.0 - hh)
                    })
                    .collect();
                for (k, w) in self.w2.iter_mut().enumerate() {
                    for (j, hh) in h.iter().enumerate() {
                        w[j] -= learning_rate * od[k] * hh;
                    }
                    let n = h.len();
                    w[n] -= learning_rate * od[k];
                }
                for (j, w) in self.w1.iter_mut().enumerate() {
                    for (i, xx) in x.iter().enumerate() {
                        w[i] -= learning_rate * hd[j] * xx;
                    }
                    let n = x.len();
                    w[n] -= learning_rate * hd[j];
                }
            }
        }
        Ok(())
    }

    /// `cv::ml::StatModel::predict` — the raw output vector.
    #[must_use]
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        self.forward(x).1
    }

    /// Index of the largest output — the predicted class.
    #[must_use]
    pub fn predict_class(&self, x: &[f64]) -> usize {
        let o = self.predict(x);
        o.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i)
    }
}

/// `cv::ml::EM` — a diagonal-covariance Gaussian mixture, fitted by EM.
#[derive(Debug, Clone)]
pub struct Em {
    /// Mixture weights.
    pub weights: Vec<f64>,
    /// Component means.
    pub means: Vec<Vec<f64>>,
    /// Per-component, per-feature variances.
    pub variances: Vec<Vec<f64>>,
}

impl Em {
    /// `cv::ml::EM::trainEM(samples, logLikelihoods, labels, probs)`.
    ///
    /// Diagonal covariance only, which is upstream's `COV_MAT_DIAGONAL`. Full
    /// covariance needs a matrix inverse per component per iteration and is
    /// numerically fussy in a way that deserves its own attention.
    pub fn train(
        samples: &[Vec<f64>],
        k: usize,
        iterations: usize,
        rng: &mut cv_core::rng::Rng,
    ) -> Result<Self> {
        cv_assert!(k > 0, "EM needs at least one component");
        cv_assert!(samples.len() >= k, "{} samples cannot fit {k} components", samples.len());
        let d = samples[0].len();
        cv_assert!(d > 0, "samples must have at least one feature");

        // Seed the means by k-means++, as upstream's `trainEM` does (it runs
        // `kmeans` with `KMEANS_PP_CENTERS` before the first E step). Picking
        // random samples instead is the textbook initialisation and it is not
        // good enough: on three well-separated clouds it lands two seeds in one
        // cloud often enough that EM converges to a split that merges the other
        // two, and nothing in the iteration can recover from that.
        let dist2 = |a: &[f64], b: &[f64]| -> f64 {
            a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum()
        };
        let mut means: Vec<Vec<f64>> = Vec::with_capacity(k);
        means.push(samples[(rng.next_u32() as usize) % samples.len()].clone());
        while means.len() < k {
            // Each remaining point is a candidate with probability
            // proportional to its squared distance from the nearest seed.
            let d2: Vec<f64> = samples
                .iter()
                .map(|s| means.iter().map(|m| dist2(s, m)).fold(f64::MAX, f64::min))
                .collect();
            let total: f64 = d2.iter().sum();
            let next = if total <= 0.0 {
                // Every point coincides with a seed; any index will do.
                (rng.next_u32() as usize) % samples.len()
            } else {
                let target = f64::from(rng.next_u32()) / f64::from(u32::MAX) * total;
                let mut acc = 0.0;
                let mut pick = samples.len() - 1;
                for (i, v) in d2.iter().enumerate() {
                    acc += v;
                    if acc >= target {
                        pick = i;
                        break;
                    }
                }
                pick
            };
            means.push(samples[next].clone());
        }
        let mut weights = vec![1.0 / k as f64; k];
        let mut variances = vec![vec![1.0f64; d]; k];

        for _ in 0..iterations.max(1) {
            // E step: responsibilities.
            let mut resp = vec![vec![0.0f64; k]; samples.len()];
            for (n, s) in samples.iter().enumerate() {
                let mut total = 0.0;
                for c in 0..k {
                    let mut p = weights[c];
                    for j in 0..d {
                        let v = variances[c][j].max(1e-9);
                        let dv = s[j] - means[c][j];
                        p *= (-(dv * dv) / (2.0 * v)).exp()
                            / (2.0 * core::f64::consts::PI * v).sqrt();
                    }
                    resp[n][c] = p;
                    total += p;
                }
                if total > 0.0 {
                    for r in resp[n].iter_mut() {
                        *r /= total;
                    }
                } else {
                    // Every component vanished numerically; share it out.
                    for r in resp[n].iter_mut() {
                        *r = 1.0 / k as f64;
                    }
                }
            }
            // M step.
            for c in 0..k {
                let nc: f64 = resp.iter().map(|r| r[c]).sum();
                if nc < 1e-12 {
                    continue;
                }
                weights[c] = nc / samples.len() as f64;
                for j in 0..d {
                    means[c][j] =
                        samples.iter().zip(&resp).map(|(s, r)| r[c] * s[j]).sum::<f64>() / nc;
                }
                for j in 0..d {
                    variances[c][j] = samples
                        .iter()
                        .zip(&resp)
                        .map(|(s, r)| r[c] * (s[j] - means[c][j]).powi(2))
                        .sum::<f64>()
                        / nc;
                    variances[c][j] = variances[c][j].max(1e-9);
                }
            }
        }
        Ok(Em { weights, means, variances })
    }

    /// Index of the most likely component for a sample.
    #[must_use]
    pub fn predict(&self, s: &[f64]) -> usize {
        let mut best = (f64::NEG_INFINITY, 0usize);
        for c in 0..self.weights.len() {
            let mut ll = self.weights[c].max(1e-300).ln();
            for (j, sj) in s.iter().enumerate() {
                let v = self.variances[c][j].max(1e-9);
                let d = sj - self.means[c][j];
                ll += -0.5 * (2.0 * core::f64::consts::PI * v).ln() - d * d / (2.0 * v);
            }
            if ll > best.0 {
                best = (ll, c);
            }
        }
        best.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::rng::Rng;

    /// Two well-separated clusters, labelled 0 and 1.
    fn two_blobs() -> TrainData {
        let mut samples = Vec::new();
        let mut responses = Vec::new();
        let mut rng = Rng::new(11);
        for _ in 0..40 {
            samples.push(vec![rng.uniform_f64(-1.0, 1.0), rng.uniform_f64(-1.0, 1.0)]);
            responses.push(0.0);
        }
        for _ in 0..40 {
            samples.push(vec![rng.uniform_f64(9.0, 11.0), rng.uniform_f64(9.0, 11.0)]);
            responses.push(1.0);
        }
        TrainData::create(samples, responses).unwrap()
    }

    fn accuracy(data: &TrainData, f: impl Fn(&[f64]) -> f64) -> f64 {
        let hits = data
            .samples
            .iter()
            .zip(&data.responses)
            .filter(|(s, r)| f(s) == **r)
            .count();
        hits as f64 / data.len() as f64
    }

    #[test]
    fn train_data_validates_its_input() {
        assert!(TrainData::create(vec![], vec![]).is_err());
        assert!(TrainData::create(vec![vec![1.0]], vec![]).is_err());
        assert!(TrainData::create(vec![vec![1.0], vec![1.0, 2.0]], vec![0.0, 1.0]).is_err());
        let d = TrainData::create(vec![vec![1.0, 2.0]], vec![5.0]).unwrap();
        assert_eq!((d.len(), d.dim()), (1, 2));
        assert_eq!(d.classes(), vec![5.0]);
    }

    #[test]
    fn knearest_separates_two_blobs() {
        let d = two_blobs();
        let m = KNearest::train(d.clone());
        assert_eq!(m.find_nearest(&[0.0, 0.0], 5).unwrap(), 0.0);
        assert_eq!(m.find_nearest(&[10.0, 10.0], 5).unwrap(), 1.0);
        assert_eq!(accuracy(&d, |s| m.find_nearest(s, 3).unwrap()), 1.0);
    }

    #[test]
    fn knearest_rejects_bad_arguments() {
        let m = KNearest::train(two_blobs());
        assert!(m.find_nearest(&[0.0, 0.0], 0).is_err());
        assert!(m.find_nearest(&[0.0], 3).is_err());
    }

    #[test]
    fn normal_bayes_separates_two_blobs() {
        let d = two_blobs();
        let m = NormalBayes::train(&d).unwrap();
        assert_eq!(m.predict(&[0.0, 0.0]).unwrap(), 0.0);
        assert_eq!(m.predict(&[10.0, 10.0]).unwrap(), 1.0);
        assert_eq!(accuracy(&d, |s| m.predict(s).unwrap()), 1.0);
    }

    #[test]
    fn normal_bayes_needs_two_classes() {
        let d = TrainData::create(vec![vec![1.0], vec![2.0]], vec![0.0, 0.0]).unwrap();
        assert!(NormalBayes::train(&d).is_err());
    }

    #[test]
    fn normal_bayes_survives_a_constant_feature() {
        // Feature 1 is constant within each class: the variance floor is what
        // stops the likelihood going infinite.
        let d = TrainData::create(
            vec![vec![0.0, 5.0], vec![0.1, 5.0], vec![9.0, 5.0], vec![9.1, 5.0]],
            vec![0.0, 0.0, 1.0, 1.0],
        )
        .unwrap();
        let m = NormalBayes::train(&d).unwrap();
        assert_eq!(m.predict(&[0.05, 5.0]).unwrap(), 0.0);
        assert_eq!(m.predict(&[9.05, 5.0]).unwrap(), 1.0);
    }

    #[test]
    fn logistic_regression_learns_a_linear_boundary() {
        let d = two_blobs();
        let m = LogisticRegression::train(&d, 0.5, 2000).unwrap();
        assert_eq!(accuracy(&d, |s| m.predict(s)), 1.0);
        assert!(m.probability(&[10.0, 10.0]) > 0.9);
        assert!(m.probability(&[0.0, 0.0]) < 0.1);
    }

    #[test]
    fn logistic_regression_rejects_non_binary_labels() {
        let d = TrainData::create(vec![vec![1.0], vec![2.0]], vec![0.0, 2.0]).unwrap();
        assert!(LogisticRegression::train(&d, 0.1, 10).is_err());
        let ok = TrainData::create(vec![vec![1.0], vec![2.0]], vec![0.0, 1.0]).unwrap();
        assert!(LogisticRegression::train(&ok, 0.0, 10).is_err());
    }

    #[test]
    fn svm_separates_two_blobs_and_keeps_support_vectors() {
        let d = two_blobs();
        let m = Svm::train(&d, 1.0, 200).unwrap();
        assert_eq!(accuracy(&d, |s| m.predict(s)), 1.0);
        assert!(!m.support_vectors().is_empty());
        assert!(
            m.support_vectors().len() < d.len(),
            "not every sample should be a support vector"
        );
        assert!(m.decision(&[0.0, 0.0]) < 0.0);
        assert!(m.decision(&[10.0, 10.0]) > 0.0);
    }

    #[test]
    fn a_linear_svm_finds_the_exact_maximum_margin() {
        // Points at x = -1 and x = +1. The max-margin hyperplane is x = 0 with
        // unit-norm weights, so the support vectors sit at f = -1 and f = +1
        // exactly. This is the test that would catch an SMO that stops short.
        let d = TrainData::create(
            vec![vec![-1.0, 0.0], vec![-1.0, 5.0], vec![1.0, 0.0], vec![1.0, 5.0]],
            vec![0.0, 0.0, 1.0, 1.0],
        )
        .unwrap();
        let m = Svm::train(&d, 10.0, 2000).unwrap();
        assert!((m.decision(&[-1.0, 0.0]) + 1.0).abs() < 1e-3, "{}", m.decision(&[-1.0, 0.0]));
        assert!((m.decision(&[1.0, 0.0]) - 1.0).abs() < 1e-3, "{}", m.decision(&[1.0, 0.0]));
        assert!(m.decision(&[0.0, 0.0]).abs() < 1e-3, "the boundary must sit at x = 0");
    }

    #[test]
    fn only_a_nonlinear_kernel_solves_xor() {
        // The point of having kernels. cv2 4.13.0 on this exact problem also
        // gets 2/4 with a linear kernel and 4/4 with RBF and POLY.
        let xor = TrainData::create(
            vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]],
            vec![0.0, 1.0, 1.0, 0.0],
        )
        .unwrap();
        let hits = |k: SvmKernel| {
            let p = SvmParams { kernel: k, c: 100.0, ..Default::default() };
            Svm::train_with(&xor, p).map(|m| {
                xor.samples.iter().zip(&xor.responses).filter(|(s, r)| m.predict(s) == **r).count()
            })
        };
        assert_eq!(hits(SvmKernel::Linear).unwrap(), 2, "a line cannot split XOR");
        assert_eq!(hits(SvmKernel::Rbf { gamma: 1.0 }).unwrap(), 4, "RBF should");
        assert_eq!(
            hits(SvmKernel::Poly { degree: 2.0, gamma: 1.0, coef0: 1.0 }).unwrap(),
            4,
            "so should a quadratic"
        );
    }

    #[test]
    fn the_sigmoid_kernel_solves_xor_when_it_is_tuned() {
        // Sigmoid needs its parameters chosen; it is not a kernel that works
        // out of the box. At gamma = 1, coef0 = -1 it manages only 2 of 4 —
        // but that is a bad operating point, *not* a property of the kernel,
        // and it would be wrong to record it as one. cv2 4.13.0 swept over
        // C/gamma/coef0 lands on (100, 0.5, -2.0) for a clean 4/4, and so does
        // this port at the same point.
        let xor = TrainData::create(
            vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]],
            vec![0.0, 1.0, 1.0, 0.0],
        )
        .unwrap();
        let hits = |k: SvmKernel, c: f64| {
            let p = SvmParams { kernel: k, c, ..Default::default() };
            Svm::train_with(&xor, p)
                .map(|m| {
                    xor.samples
                        .iter()
                        .zip(&xor.responses)
                        .filter(|(s, r)| m.predict(s) == **r)
                        .count()
                })
                .unwrap_or(0)
        };
        assert_eq!(
            hits(SvmKernel::Sigmoid { gamma: 0.5, coef0: -2.0 }, 100.0),
            4,
            "sigmoid at cv2's operating point should separate XOR"
        );
        assert_eq!(
            hits(SvmKernel::Sigmoid { gamma: 1.0, coef0: -1.0 }, 100.0),
            2,
            "and fall over at a badly chosen one, which is what tuning is for"
        );
    }

    #[test]
    fn rbf_separates_concentric_rings_that_no_line_can() {
        let mut s = Vec::new();
        let mut r = Vec::new();
        for i in 0..60 {
            let a = f64::from(i) * 0.1047;
            s.push(vec![a.cos(), a.sin()]);
            r.push(0.0);
            s.push(vec![a.cos() * 4.0, a.sin() * 4.0]);
            r.push(1.0);
        }
        let d = TrainData::create(s, r).unwrap();
        let rbf = Svm::train_with(
            &d,
            SvmParams { kernel: SvmKernel::Rbf { gamma: 0.5 }, c: 10.0, ..Default::default() },
        )
        .unwrap();
        assert_eq!(accuracy(&d, |x| rbf.predict(x)), 1.0, "RBF should separate the rings");

        let lin = Svm::train_with(
            &d,
            SvmParams { kernel: SvmKernel::Linear, c: 10.0, ..Default::default() },
        )
        .unwrap();
        assert!(
            accuracy(&d, |x| lin.predict(x)) < 0.8,
            "a linear kernel cannot separate concentric rings"
        );
    }

    #[test]
    fn every_kernel_evaluates_sensibly() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        assert_eq!(SvmKernel::Linear.eval(&a, &b), 11.0);
        assert_eq!(SvmKernel::Poly { degree: 2.0, gamma: 1.0, coef0: 0.0 }.eval(&a, &b), 121.0);
        // RBF of a point with itself is 1, and falls off with distance.
        assert_eq!(SvmKernel::Rbf { gamma: 1.0 }.eval(&a, &a), 1.0);
        assert!(SvmKernel::Rbf { gamma: 1.0 }.eval(&a, &b) < 1e-3);
        assert_eq!(SvmKernel::Inter.eval(&a, &b), 3.0, "min(1,3) + min(2,4)");
        // Chi2 of a point with itself is 1, and the zero-sum guard holds.
        assert_eq!(SvmKernel::Chi2 { gamma: 1.0 }.eval(&a, &a), 1.0);
        assert_eq!(SvmKernel::Chi2 { gamma: 1.0 }.eval(&[0.0], &[0.0]), 1.0);
        assert!(SvmKernel::Sigmoid { gamma: 0.1, coef0: 0.0 }.eval(&a, &b).abs() <= 1.0);
    }

    #[test]
    fn train_auto_finds_the_sigmoid_setting_upstream_cannot_reach() {
        // The concrete payoff of the explicit grid. cv2's trainAuto sweeps
        // coef0 multiplicatively from 0.1 to 300 and rejects a grid with a
        // negative minimum, so it lands on coef0 = 0.1 and gets 2/4 on XOR.
        // Searching the negative half finds a point that separates it.
        let xor = TrainData::create(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        )
        .unwrap();
        let mut auto = AutoParams::for_kernel(SvmKernelKind::Sigmoid);
        auto.k_fold = 4;
        auto.gamma = ParamGrid::Values(vec![0.25, 0.5, 1.0, 2.0]);
        auto.c = ParamGrid::Values(vec![10.0, 100.0]);
        let (m, chosen) = Svm::train_auto(&xor, &auto).unwrap();
        let hits = xor
            .samples
            .iter()
            .zip(&xor.responses)
            .filter(|(s, r)| m.predict(s) == **r)
            .count();
        assert_eq!(hits, 8, "trainAuto should find a separating sigmoid, chose {chosen:?}");
        match chosen.kernel {
            SvmKernel::Sigmoid { coef0, .. } => {
                assert!(coef0 < 0.0, "the winning coef0 is negative, got {coef0}");
            }
            k => panic!("expected a sigmoid kernel, got {k:?}"),
        }

        // With upstream's positive-only grid the same search cannot get there.
        auto.coef0 = ParamGrid::upstream_coef0();
        let (m2, _) = Svm::train_auto(&xor, &auto).unwrap();
        let hits2 = xor
            .samples
            .iter()
            .zip(&xor.responses)
            .filter(|(s, r)| m2.predict(s) == **r)
            .count();
        assert!(hits2 < 8, "upstream's grid should fall short here, got {hits2}/8");
    }

    #[test]
    fn train_auto_tunes_an_rbf_kernel() {
        let d = two_blobs();
        let mut auto = AutoParams::for_kernel(SvmKernelKind::Rbf);
        auto.k_fold = 4;
        auto.c = ParamGrid::Values(vec![1.0, 10.0]);
        auto.gamma = ParamGrid::Values(vec![0.001, 0.01, 0.1]);
        let (m, chosen) = Svm::train_auto(&d, &auto).unwrap();
        assert_eq!(accuracy(&d, |s| m.predict(s)), 1.0);
        assert!(matches!(chosen.kernel, SvmKernel::Rbf { .. }));
    }

    #[test]
    fn param_grids_enumerate_correctly() {
        // Upstream's multiplicative form.
        let g = ParamGrid::Log { min: 1.0, max: 100.0, step: 10.0 };
        assert_eq!(g.values().unwrap(), vec![1.0, 10.0, 100.0]);
        assert_eq!(ParamGrid::Fixed(7.0).values().unwrap(), vec![7.0]);
        assert_eq!(ParamGrid::Values(vec![-1.0, 0.0]).values().unwrap(), vec![-1.0, 0.0]);

        // A log grid cannot start at or below zero — the reason `Values` exists.
        assert!(ParamGrid::Log { min: 0.0, max: 10.0, step: 2.0 }.values().is_err());
        assert!(ParamGrid::Log { min: -1.0, max: 10.0, step: 2.0 }.values().is_err());
        assert!(ParamGrid::Log { min: 1.0, max: 10.0, step: 1.0 }.values().is_err());
        assert!(ParamGrid::Values(vec![]).values().is_err());

        // Upstream's default C grid, for comparison.
        let c = ParamGrid::default_c().values().unwrap();
        assert_eq!(c.first(), Some(&0.1));
        assert!(c.iter().all(|&v| v <= 500.0));
    }

    #[test]
    fn train_auto_validates_its_folds() {
        let d = two_blobs();
        let mut auto = AutoParams::for_kernel(SvmKernelKind::Linear);
        auto.k_fold = 1;
        assert!(Svm::train_auto(&d, &auto).is_err(), "one fold is not cross-validation");
        auto.k_fold = 1000;
        assert!(Svm::train_auto(&d, &auto).is_err(), "more folds than samples");
    }

    #[test]
    fn multiclass_svm_uses_one_vs_one() {
        let mut rng = Rng::new(77);
        let mut s = Vec::new();
        let mut r = Vec::new();
        for (cx, cy, lab) in [(0.0, 0.0, 0.0), (10.0, 0.0, 1.0), (5.0, 9.0, 2.0)] {
            for _ in 0..30 {
                s.push(vec![cx + rng.uniform_f64(-1.5, 1.5), cy + rng.uniform_f64(-1.5, 1.5)]);
                r.push(lab);
            }
        }
        let d = TrainData::create(s, r).unwrap();
        let m = Svm::train(&d, 1.0, 500).unwrap();
        // Three classes give three pairs.
        assert_eq!(m.machine_count(), 3);
        assert_eq!(m.labels(), &[0.0, 1.0, 2.0]);
        assert_eq!(accuracy(&d, |x| m.predict(x)), 1.0);
        assert_eq!(m.predict(&[0.0, 0.0]), 0.0);
        assert_eq!(m.predict(&[10.0, 0.0]), 1.0);
        assert_eq!(m.predict(&[5.0, 9.0]), 2.0);
    }

    #[test]
    fn a_binary_problem_trains_one_machine() {
        let m = Svm::train(&two_blobs(), 1.0, 200).unwrap();
        assert_eq!(m.machine_count(), 1);
        assert_eq!(m.labels(), &[0.0, 1.0]);
    }

    #[test]
    fn svm_handles_overlapping_classes_without_failing() {
        // Not separable: the soft margin must absorb it rather than error.
        let mut rng = Rng::new(3);
        let mut s = Vec::new();
        let mut r = Vec::new();
        for _ in 0..60 {
            s.push(vec![rng.uniform_f64(-2.0, 2.0), rng.uniform_f64(-2.0, 2.0)]);
            r.push(0.0);
        }
        for _ in 0..60 {
            s.push(vec![rng.uniform_f64(-1.0, 3.0), rng.uniform_f64(-1.0, 3.0)]);
            r.push(1.0);
        }
        let d = TrainData::create(s, r).unwrap();
        let m = Svm::train(&d, 1.0, 300).unwrap();
        let a = accuracy(&d, |x| m.predict(x));
        assert!(a > 0.6, "should beat chance on overlapping blobs, got {a}");
    }

    #[test]
    fn svm_rejects_bad_parameters() {
        let d = TrainData::create(vec![vec![0.0], vec![1.0]], vec![0.0, 0.0]).unwrap();
        assert!(Svm::train(&d, 1.0, 10).is_err(), "one class is not a classification problem");
        let ok = two_blobs();
        assert!(Svm::train(&ok, 0.0, 10).is_err(), "C must be positive");
        assert!(Svm::train_with(&ok, SvmParams { tolerance: 0.0, ..Default::default() }).is_err());
    }

    #[test]
    fn decision_tree_learns_an_axis_aligned_rule() {
        // Class 1 exactly when x > 5; a tree should find that in one split.
        let samples: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, (i * 7 % 13) as f64]).collect();
        let responses: Vec<f64> = samples.iter().map(|s| if s[0] > 5.0 { 1.0 } else { 0.0 }).collect();
        let d = TrainData::create(samples, responses).unwrap();
        let t = DTree::train(&d, 4, 1).unwrap();
        assert_eq!(accuracy(&d, |s| t.predict(s)), 1.0);
        assert_eq!(t.predict(&[0.0, 0.0]), 0.0);
        assert_eq!(t.predict(&[19.0, 0.0]), 1.0);
    }

    #[test]
    fn decision_tree_depth_limit_is_honoured() {
        let d = two_blobs();
        let t = DTree::train(&d, 1, 1).unwrap();
        assert!(t.depth <= 1, "depth {} exceeded the limit", t.depth);
    }

    #[test]
    fn a_pure_node_becomes_a_leaf() {
        let d = TrainData::create(vec![vec![1.0], vec![2.0], vec![3.0]], vec![7.0, 7.0, 7.0])
            .unwrap();
        let t = DTree::train(&d, 5, 1).unwrap();
        assert_eq!(t.depth, 0, "a pure set needs no split");
        assert_eq!(t.predict(&[99.0]), 7.0);
    }

    #[test]
    fn random_forest_matches_or_beats_a_single_tree_on_noisy_data() {
        let mut rng = Rng::new(5);
        // An XOR-ish problem with noise, where one shallow tree struggles.
        let mut samples = Vec::new();
        let mut responses = Vec::new();
        for _ in 0..200 {
            let x = rng.uniform_f64(-1.0, 1.0);
            let y = rng.uniform_f64(-1.0, 1.0);
            samples.push(vec![x, y, rng.uniform_f64(-1.0, 1.0)]);
            responses.push(if (x > 0.0) == (y > 0.0) { 1.0 } else { 0.0 });
        }
        let d = TrainData::create(samples, responses).unwrap();
        let forest = RTrees::train(&d, 15, 6, &mut rng).unwrap();
        assert_eq!(forest.len(), 15);
        let acc = accuracy(&d, |s| forest.predict(s));
        assert!(acc > 0.85, "the forest should learn XOR, accuracy {acc}");
    }

    #[test]
    fn forest_is_reproducible_from_its_seed() {
        let d = two_blobs();
        let a = RTrees::train(&d, 5, 4, &mut Rng::new(1)).unwrap();
        let b = RTrees::train(&d, 5, 4, &mut Rng::new(1)).unwrap();
        for s in &d.samples {
            assert_eq!(a.predict(s), b.predict(s));
        }
    }

    #[test]
    fn mlp_learns_xor() {
        // The canonical test: XOR is not linearly separable, so a network that
        // learns it has a working hidden layer.
        let mut rng = Rng::new(2024);
        let samples = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
        let mut best = 0;
        // Backprop from a random start occasionally lands in a bad basin;
        // upstream restarts too. Three attempts is plenty.
        for seed in 0..3 {
            let mut net = AnnMlp::new(2, 6, 1, &mut Rng::new(2024 + seed)).unwrap();
            net.train(&samples, &targets, 3.0, 4000).unwrap();
            let hits = samples
                .iter()
                .zip(&targets)
                .filter(|(s, t)| (net.predict(s)[0] - t[0]).abs() < 0.3)
                .count();
            best = best.max(hits);
            if hits == 4 {
                break;
            }
        }
        let _ = &mut rng;
        assert_eq!(best, 4, "the MLP never learned XOR");
    }

    #[test]
    fn mlp_rejects_bad_shapes() {
        let mut rng = Rng::new(1);
        assert!(AnnMlp::new(0, 3, 1, &mut rng).is_err());
        let mut net = AnnMlp::new(2, 3, 2, &mut rng).unwrap();
        assert!(net.train(&[vec![0.0, 0.0]], &[], 0.1, 1).is_err());
        assert!(net.train(&[vec![0.0, 0.0]], &[vec![1.0, 0.0]], 0.0, 1).is_err());
    }

    #[test]
    fn mlp_predict_class_picks_the_largest_output() {
        let mut rng = Rng::new(9);
        let mut net = AnnMlp::new(2, 5, 3, &mut rng).unwrap();
        let samples = vec![vec![0.0, 0.0], vec![5.0, 5.0]];
        let targets = vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]];
        net.train(&samples, &targets, 1.0, 3000).unwrap();
        assert_eq!(net.predict_class(&[0.0, 0.0]), 0);
        assert_eq!(net.predict_class(&[5.0, 5.0]), 2);
    }

    #[test]
    fn em_recovers_two_gaussian_components() {
        let mut rng = Rng::new(31);
        let mut samples = Vec::new();
        for _ in 0..120 {
            samples.push(vec![rng.uniform_f64(-1.0, 1.0), rng.uniform_f64(-1.0, 1.0)]);
        }
        for _ in 0..120 {
            samples.push(vec![rng.uniform_f64(19.0, 21.0), rng.uniform_f64(19.0, 21.0)]);
        }
        let em = Em::train(&samples, 2, 40, &mut Rng::new(7)).unwrap();
        assert_eq!(em.weights.len(), 2);
        assert!((em.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9, "weights must normalise");
        // Every point in a cluster gets the same component, and the two differ.
        let a = em.predict(&samples[0]);
        let b = em.predict(&samples[200]);
        assert_ne!(a, b, "the two clusters must land in different components");
        assert!(samples[..120].iter().all(|s| em.predict(s) == a));
        assert!(samples[120..].iter().all(|s| em.predict(s) == b));
        // And the means sit on the clusters.
        let m = &em.means[a];
        assert!(m[0].abs() < 1.0, "component {a} mean {m:?} is not on the first cluster");
    }

    #[test]
    fn em_rejects_more_components_than_samples() {
        assert!(Em::train(&[vec![1.0]], 5, 10, &mut Rng::new(1)).is_err());
        assert!(Em::train(&[vec![1.0]], 0, 10, &mut Rng::new(1)).is_err());
    }
}
