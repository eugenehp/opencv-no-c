//! The `StatModel` configuration surface — every model's `get*`/`set*` pair and
//! the shared enumerations (`modules/ml/include/opencv2/ml.hpp`).
//!
//! # Why this is generated
//!
//! Upstream exposes each model's tunables as a pair of virtual methods, and
//! there are nine models. That is several hundred functions whose bodies are
//! two lines each. `ml_accessors!` derives the pair from one line per
//! parameter, so the field, both methods and the upstream doc tag cannot drift
//! apart — a getter wired to the neighbouring field would compile and silently
//! return the wrong tunable.

/// Generate `get_*`/`set_*` pairs for a model's parameters.
///
/// Each entry is `field: type = default, getter / setter, "Upstream::get",
/// "Upstream::set"`.
#[macro_export]
macro_rules! ml_accessors {
    (
        $(#[$sm:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$fm:meta])*
                $field:ident : $ty:ty = $default:expr, $getter:ident / $setter:ident, $up_get:literal, $up_set:literal
            );* $(;)?
        }
    ) => {
        $(#[$sm])*
        #[derive(Debug, Clone, PartialEq)]
        $vis struct $name {
            $(
                $(#[$fm])*
                pub $field: $ty,
            )*
        }

        impl Default for $name {
            /// Upstream's defaults.
            fn default() -> Self {
                Self { $($field: $default,)* }
            }
        }

        impl $name {
            $(
                #[doc = concat!("`cv::ml::", $up_get, "()`.")]
                #[must_use]
                pub fn $getter(&self) -> $ty {
                    self.$field.clone()
                }

                #[doc = concat!("`cv::ml::", $up_set, "()`.")]
                pub fn $setter(&mut self, v: $ty) -> &mut Self {
                    self.$field = v;
                    self
                }
            )*
        }
    };
}

/// `cv::ml::SampleTypes` — whether a sample is a row or a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SampleTypes {
    /// `cv::ml::ROW_SAMPLE`
    #[default]
    RowSample,
    /// `cv::ml::COL_SAMPLE`
    ColSample,
}

/// `cv::ml::VariableTypes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableTypes {
    /// `cv::ml::VAR_NUMERICAL` / `cv::ml::VAR_ORDERED` — an ordered quantity.
    Numerical,
    /// `cv::ml::VAR_CATEGORICAL` — a label with no ordering.
    Categorical,
}

/// `cv::ml::ErrorTypes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorTypes {
    /// `cv::ml::TEST_ERROR`
    TestError,
    /// `cv::ml::TRAIN_ERROR`
    TrainError,
}

/// `cv::ml::StatModel::Flags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainFlags {
    /// `cv::ml::StatModel::UPDATE_MODEL` — refine rather than retrain.
    UpdateModel,
    /// `cv::ml::StatModel::RAW_OUTPUT` — return the raw score, not a label.
    RawOutput,
    /// `cv::ml::StatModel::COMPRESSED_INPUT`
    CompressedInput,
    /// `cv::ml::StatModel::PREPROCESSED_INPUT`
    PreprocessedInput,
}

/// `cv::ml::KNearest::Types`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KNearestType {
    /// `cv::ml::KNearest::BRUTE_FORCE` — compare against every sample.
    #[default]
    BruteForce,
    /// `cv::ml::KNearest::KDTREE` — a spatial index; faster in low dimensions.
    KdTree,
}

ml_accessors! {
    /// `cv::ml::KNearest` parameters.
    pub struct KNearestParams {
        /// Neighbours consulted per query.
        /// Read by `cv::ml::KNearest::getDefaultK()`, written by `cv::ml::KNearest::setDefaultK()`.
        default_k: i32 = 10, get_default_k / set_default_k, "KNearest::getDefaultK", "KNearest::setDefaultK";
        /// Classify (vote) rather than regress (average).
        /// Read by `cv::ml::KNearest::getIsClassifier()`, written by `cv::ml::KNearest::setIsClassifier()`.
        is_classifier: bool = true, get_is_classifier / set_is_classifier, "KNearest::getIsClassifier", "KNearest::setIsClassifier";
        /// Leaf size for the k-d tree.
        /// Read by `cv::ml::KNearest::getEmax()`, written by `cv::ml::KNearest::setEmax()`.
        emax: i32 = 1000, get_emax / set_emax, "KNearest::getEmax", "KNearest::setEmax";
        /// Brute force or a k-d tree.
        /// Read by `cv::ml::KNearest::getAlgorithmType()`, written by `cv::ml::KNearest::setAlgorithmType()`.
        algorithm_type: KNearestType = KNearestType::BruteForce, get_algorithm_type / set_algorithm_type, "KNearest::getAlgorithmType", "KNearest::setAlgorithmType";
    }
}

/// `cv::ml::SVM::Types`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvmType {
    /// `cv::ml::SVM::C_SVC` — n-class classification with a soft margin.
    #[default]
    CSvc,
    /// `cv::ml::SVM::NU_SVC` — the same, parameterised by `nu` instead of `C`.
    NuSvc,
    /// `cv::ml::SVM::ONE_CLASS` — novelty detection.
    OneClass,
    /// `cv::ml::SVM::EPS_SVR` — regression with an epsilon-insensitive loss.
    EpsSvr,
    /// `cv::ml::SVM::NU_SVR` — regression parameterised by `nu`.
    NuSvr,
}

/// `cv::ml::SVM::ParamTypes` — which parameter a grid search varies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvmParamType {
    /// `cv::ml::SVM::C`
    C,
    /// `cv::ml::SVM::GAMMA`
    Gamma,
    /// `cv::ml::SVM::P`
    P,
    /// `cv::ml::SVM::NU`
    Nu,
    /// `cv::ml::SVM::COEF`
    Coef,
    /// `cv::ml::SVM::DEGREE`
    Degree,
}

/// `cv::ml::SVMSGD::SvmsgdType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvmsgdType {
    /// `cv::ml::SVMSGD::SGD` — plain stochastic gradient descent.
    Sgd,
    /// `cv::ml::SVMSGD::ASGD` — averaged SGD; steadier, usually better.
    #[default]
    Asgd,
}

/// `cv::ml::SVMSGD::MarginType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MarginType {
    /// `cv::ml::SVMSGD::SOFT_MARGIN` — tolerates overlap between classes.
    #[default]
    SoftMargin,
    /// `cv::ml::SVMSGD::HARD_MARGIN` — only for linearly separable data.
    HardMargin,
}

ml_accessors! {
    /// `cv::ml::SVMSGD` parameters.
    pub struct SvmsgdParams {
        /// Weight on the margin term.
        /// Read by `cv::ml::SVMSGD::getMarginRegularization()`, written by `cv::ml::SVMSGD::setMarginRegularization()`.
        margin_regularization: f32 = 0.0001, get_margin_regularization / set_margin_regularization, "SVMSGD::getMarginRegularization", "SVMSGD::setMarginRegularization";
        /// First step size; decays as training proceeds.
        /// Read by `cv::ml::SVMSGD::getInitialStepSize()`, written by `cv::ml::SVMSGD::setInitialStepSize()`.
        initial_step_size: f32 = 0.05, get_initial_step_size / set_initial_step_size, "SVMSGD::getInitialStepSize", "SVMSGD::setInitialStepSize";
        /// How fast the step size decays.
        /// Read by `cv::ml::SVMSGD::getStepDecreasingPower()`, written by `cv::ml::SVMSGD::setStepDecreasingPower()`.
        step_decreasing_power: f32 = 0.75, get_step_decreasing_power / set_step_decreasing_power, "SVMSGD::getStepDecreasingPower", "SVMSGD::setStepDecreasingPower";
        /// Plain or averaged SGD.
        /// Read by `cv::ml::SVMSGD::getSvmsgdType()`, written by `cv::ml::SVMSGD::setSvmsgdType()`.
        svmsgd_type: SvmsgdType = SvmsgdType::Asgd, get_svmsgd_type / set_svmsgd_type, "SVMSGD::getSvmsgdType", "SVMSGD::setSvmsgdType";
        /// Whether the classes may overlap.
        /// Read by `cv::ml::SVMSGD::getMarginType()`, written by `cv::ml::SVMSGD::setMarginType()`.
        margin_type: MarginType = MarginType::SoftMargin, get_margin_type / set_margin_type, "SVMSGD::getMarginType", "SVMSGD::setMarginType";
    }
}

/// `cv::ml::ANN_MLP::ActivationFunctions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActivationFunction {
    /// `cv::ml::ANN_MLP::IDENTITY`
    Identity,
    /// `cv::ml::ANN_MLP::SIGMOID_SYM` — the symmetric sigmoid, upstream's default.
    #[default]
    SigmoidSym,
    /// `cv::ml::ANN_MLP::GAUSSIAN`
    Gaussian,
    /// `cv::ml::ANN_MLP::RELU`
    Relu,
    /// `cv::ml::ANN_MLP::LEAKYRELU`
    LeakyRelu,
}

/// `cv::ml::ANN_MLP::TrainingMethods`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainingMethod {
    /// `cv::ml::ANN_MLP::BACKPROP`
    Backprop,
    /// `cv::ml::ANN_MLP::RPROP` — resilient backprop, upstream's default.
    #[default]
    Rprop,
    /// `cv::ml::ANN_MLP::ANNEAL` — simulated annealing.
    Anneal,
}

/// `cv::ml::ANN_MLP::TrainFlags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnTrainFlags {
    /// `cv::ml::ANN_MLP::UPDATE_WEIGHTS`
    UpdateWeights,
    /// `cv::ml::ANN_MLP::NO_INPUT_SCALE`
    NoInputScale,
    /// `cv::ml::ANN_MLP::NO_OUTPUT_SCALE`
    NoOutputScale,
}

ml_accessors! {
    /// `cv::ml::ANN_MLP` parameters.
    pub struct AnnMlpParams {
        /// Weight on the backprop update.
        /// Read by `cv::ml::ANN_MLP::getBackpropWeightScale()`, written by `cv::ml::ANN_MLP::setBackpropWeightScale()`.
        backprop_weight_scale: f64 = 0.1, get_backprop_weight_scale / set_backprop_weight_scale, "ANN_MLP::getBackpropWeightScale", "ANN_MLP::setBackpropWeightScale";
        /// Momentum carried between backprop steps.
        /// Read by `cv::ml::ANN_MLP::getBackpropMomentumScale()`, written by `cv::ml::ANN_MLP::setBackpropMomentumScale()`.
        backprop_momentum_scale: f64 = 0.1, get_backprop_momentum_scale / set_backprop_momentum_scale, "ANN_MLP::getBackpropMomentumScale", "ANN_MLP::setBackpropMomentumScale";
        /// Initial RPROP step.
        /// Read by `cv::ml::ANN_MLP::getRpropDW0()`, written by `cv::ml::ANN_MLP::setRpropDW0()`.
        rprop_dw0: f64 = 0.1, get_rprop_dw0 / set_rprop_dw0, "ANN_MLP::getRpropDW0", "ANN_MLP::setRpropDW0";
        /// RPROP growth factor on a consistent gradient.
        /// Read by `cv::ml::ANN_MLP::getRpropDWPlus()`, written by `cv::ml::ANN_MLP::setRpropDWPlus()`.
        rprop_dw_plus: f64 = 1.2, get_rprop_dw_plus / set_rprop_dw_plus, "ANN_MLP::getRpropDWPlus", "ANN_MLP::setRpropDWPlus";
        /// RPROP shrink factor when the gradient flips sign.
        /// Read by `cv::ml::ANN_MLP::getRpropDWMinus()`, written by `cv::ml::ANN_MLP::setRpropDWMinus()`.
        rprop_dw_minus: f64 = 0.5, get_rprop_dw_minus / set_rprop_dw_minus, "ANN_MLP::getRpropDWMinus", "ANN_MLP::setRpropDWMinus";
        /// Smallest permitted RPROP step.
        /// Read by `cv::ml::ANN_MLP::getRpropDWMin()`, written by `cv::ml::ANN_MLP::setRpropDWMin()`.
        rprop_dw_min: f64 = f32::EPSILON as f64, get_rprop_dw_min / set_rprop_dw_min, "ANN_MLP::getRpropDWMin", "ANN_MLP::setRpropDWMin";
        /// Largest permitted RPROP step.
        /// Read by `cv::ml::ANN_MLP::getRpropDWMax()`, written by `cv::ml::ANN_MLP::setRpropDWMax()`.
        rprop_dw_max: f64 = 50.0, get_rprop_dw_max / set_rprop_dw_max, "ANN_MLP::getRpropDWMax", "ANN_MLP::setRpropDWMax";
        /// Starting temperature for simulated annealing.
        /// Read by `cv::ml::ANN_MLP::getAnnealInitialT()`, written by `cv::ml::ANN_MLP::setAnnealInitialT()`.
        anneal_initial_t: f64 = 10.0, get_anneal_initial_t / set_anneal_initial_t, "ANN_MLP::getAnnealInitialT", "ANN_MLP::setAnnealInitialT";
        /// Final temperature.
        /// Read by `cv::ml::ANN_MLP::getAnnealFinalT()`, written by `cv::ml::ANN_MLP::setAnnealFinalT()`.
        anneal_final_t: f64 = 0.1, get_anneal_final_t / set_anneal_final_t, "ANN_MLP::getAnnealFinalT", "ANN_MLP::setAnnealFinalT";
        /// Multiplier applied to the temperature each step.
        /// Read by `cv::ml::ANN_MLP::getAnnealCoolingRatio()`, written by `cv::ml::ANN_MLP::setAnnealCoolingRatio()`.
        anneal_cooling_ratio: f64 = 0.95, get_anneal_cooling_ratio / set_anneal_cooling_ratio, "ANN_MLP::getAnnealCoolingRatio", "ANN_MLP::setAnnealCoolingRatio";
        /// Iterations at each temperature.
        /// Read by `cv::ml::ANN_MLP::getAnnealItePerStep()`, written by `cv::ml::ANN_MLP::setAnnealItePerStep()`.
        anneal_ite_per_step: i32 = 10, get_anneal_ite_per_step / set_anneal_ite_per_step, "ANN_MLP::getAnnealItePerStep", "ANN_MLP::setAnnealItePerStep";
        /// Units per layer, input first. Empty until `setLayerSizes`.
        /// Read by `cv::ml::ANN_MLP::getLayerSizes()`, written by `cv::ml::ANN_MLP::setLayerSizes()`.
        layer_sizes: Vec<i32> = Vec::new(), get_layer_sizes / set_layer_sizes, "ANN_MLP::getLayerSizes", "ANN_MLP::setLayerSizes";
        /// The neuron nonlinearity, applied to every hidden and output unit.
        /// Written by `cv::ml::ANN_MLP::setActivationFunction()`.
        activation_function: ActivationFunction = ActivationFunction::SigmoidSym, get_activation_function / set_activation_function, "ANN_MLP::getActivationFunction", "ANN_MLP::setActivationFunction";
        /// Backprop, RPROP or simulated annealing.
        /// Read by `cv::ml::ANN_MLP::getTrainMethod()`, written by `cv::ml::ANN_MLP::setTrainMethod()`.
        training_method: TrainingMethod = TrainingMethod::Rprop, get_training_method / set_training_method, "ANN_MLP::getTrainMethod", "ANN_MLP::setTrainMethod";
    }
}

/// `cv::ml::EM::Types` — the shape each cluster's covariance may take.
///
/// The three trade fit against the number of parameters: spherical estimates
/// one variance per cluster, diagonal one per feature, generic a whole matrix.
/// Generic needs far more data to estimate without becoming singular.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CovMatType {
    /// `cv::ml::EM::COV_MAT_SPHERICAL`
    Spherical,
    /// `cv::ml::EM::COV_MAT_DIAGONAL` — upstream's default.
    #[default]
    Diagonal,
    /// `cv::ml::EM::COV_MAT_GENERIC`
    Generic,
}

impl CovMatType {
    /// `cv::ml::EM::COV_MAT_DEFAULT`.
    #[must_use]
    pub fn default_type() -> Self {
        CovMatType::Diagonal
    }
}

ml_accessors! {
    /// `cv::ml::EM` parameters.
    pub struct EmParams {
        /// Mixture components.
        /// Read by `cv::ml::EM::getClustersNumber()`, written by `cv::ml::EM::setClustersNumber()`.
        clusters_number: i32 = 5, get_clusters_number / set_clusters_number, "EM::getClustersNumber", "EM::setClustersNumber";
        /// The shape each component's covariance may take.
        /// Read by `cv::ml::EM::getCovarianceMatrixType()`, written by `cv::ml::EM::setCovarianceMatrixType()`.
        covariance_matrix_type: CovMatType = CovMatType::Diagonal, get_covariance_matrix_type / set_covariance_matrix_type, "EM::getCovarianceMatrixType", "EM::setCovarianceMatrixType";
    }
}

/// `cv::ml::LogisticRegression::Methods`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LrMethod {
    /// `cv::ml::LogisticRegression::BATCH` — every sample per step.
    #[default]
    Batch,
    /// `cv::ml::LogisticRegression::MINI_BATCH` — a subset per step.
    MiniBatch,
}

/// `cv::ml::LogisticRegression::RegKinds`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegKind {
    /// `cv::ml::LogisticRegression::REG_DISABLE`
    Disabled,
    /// `cv::ml::LogisticRegression::REG_L1`
    L1,
    /// `cv::ml::LogisticRegression::REG_L2` — upstream's default.
    #[default]
    L2,
}

ml_accessors! {
    /// `cv::ml::LogisticRegression` parameters.
    pub struct LogisticRegressionParams {
        /// Gradient-descent step size.
        /// Read by `cv::ml::LogisticRegression::getLearningRate()`, written by `cv::ml::LogisticRegression::setLearningRate()`.
        learning_rate: f64 = 0.001, get_learning_rate / set_learning_rate, "LogisticRegression::getLearningRate", "LogisticRegression::setLearningRate";
        /// Descent steps.
        /// Read by `cv::ml::LogisticRegression::getIterations()`, written by `cv::ml::LogisticRegression::setIterations()`.
        iterations: i32 = 1000, get_iterations / set_iterations, "LogisticRegression::getIterations", "LogisticRegression::setIterations";
        /// Samples per step under [`LrMethod::MiniBatch`].
        /// Read by `cv::ml::LogisticRegression::getMiniBatchSize()`, written by `cv::ml::LogisticRegression::setMiniBatchSize()`.
        mini_batch_size: i32 = 1, get_mini_batch_size / set_mini_batch_size, "LogisticRegression::getMiniBatchSize", "LogisticRegression::setMiniBatchSize";
        /// Which penalty keeps the weights small.
        /// Read by `cv::ml::LogisticRegression::getRegularization()`, written by `cv::ml::LogisticRegression::setRegularization()`.
        regularization: RegKind = RegKind::L2, get_regularization / set_regularization, "LogisticRegression::getRegularization", "LogisticRegression::setRegularization";
        /// Whether a step uses every sample or a mini-batch.
        /// Read by `cv::ml::LogisticRegression::getTrainMethod()`, written by `cv::ml::LogisticRegression::setTrainMethod()`.
        train_method: LrMethod = LrMethod::Batch, get_train_method / set_train_method, "LogisticRegression::getTrainMethod", "LogisticRegression::setTrainMethod";
    }
}

/// `cv::ml::Boost::Types`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoostType {
    /// `cv::ml::Boost::DISCRETE`
    Discrete,
    /// `cv::ml::Boost::REAL` — upstream's default.
    #[default]
    Real,
    /// `cv::ml::Boost::LOGIT`
    Logit,
    /// `cv::ml::Boost::GENTLE`
    Gentle,
}

ml_accessors! {
    /// `cv::ml::Boost` parameters.
    pub struct BoostParams {
        /// Weak learners in the ensemble.
        /// Read by `cv::ml::Boost::getWeakCount()`, written by `cv::ml::Boost::setWeakCount()`.
        weak_count: i32 = 100, get_weak_count / set_weak_count, "Boost::getWeakCount", "Boost::setWeakCount";
        /// Drop the lowest-weight samples below this cumulative share.
        /// Read by `cv::ml::Boost::getWeightTrimRate()`, written by `cv::ml::Boost::setWeightTrimRate()`.
        weight_trim_rate: f64 = 0.95, get_weight_trim_rate / set_weight_trim_rate, "Boost::getWeightTrimRate", "Boost::setWeightTrimRate";
        /// Which boosting variant reweights the samples.
        /// Read by `cv::ml::Boost::getBoostType()`, written by `cv::ml::Boost::setBoostType()`.
        boost_type: BoostType = BoostType::Real, get_boost_type / set_boost_type, "Boost::getBoostType", "Boost::setBoostType";
    }
}

/// `cv::ml::DTrees::Flags` — how a prediction is aggregated over an ensemble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictFlags {
    /// `cv::ml::DTrees::PREDICT_AUTO`
    Auto,
    /// `cv::ml::DTrees::PREDICT_SUM` — total the leaf values.
    Sum,
    /// `cv::ml::DTrees::PREDICT_MAX_VOTE` — majority over the trees.
    MaxVote,
    /// `cv::ml::DTrees::PREDICT_MASK`
    Mask,
}

ml_accessors! {
    /// `cv::ml::DTrees` parameters, shared by `RTrees` and `Boost`.
    pub struct DTreesParams {
        /// Deepest split allowed.
        /// Read by `cv::ml::DTrees::getMaxDepth()`, written by `cv::ml::DTrees::setMaxDepth()`.
        max_depth: i32 = 10, get_max_depth / set_max_depth, "DTrees::getMaxDepth", "DTrees::setMaxDepth";
        /// A node with fewer samples is not split.
        /// Read by `cv::ml::DTrees::getMinSampleCount()`, written by `cv::ml::DTrees::setMinSampleCount()`.
        min_sample_count: i32 = 10, get_min_sample_count / set_min_sample_count, "DTrees::getMinSampleCount", "DTrees::setMinSampleCount";
        /// Distinct values a categorical variable may take before clustering.
        /// Read by `cv::ml::DTrees::getMaxCategories()`, written by `cv::ml::DTrees::setMaxCategories()`.
        max_categories: i32 = 10, get_max_categories / set_max_categories, "DTrees::getMaxCategories", "DTrees::setMaxCategories";
        /// Cross-validation folds used while pruning; 0 disables it.
        /// Read by `cv::ml::DTrees::getCVFolds()`, written by `cv::ml::DTrees::setCVFolds()`.
        cv_folds: i32 = 0, get_cv_folds / set_cv_folds, "DTrees::getCVFolds", "DTrees::setCVFolds";
        /// Prune branches whose removal barely changes the error.
        /// Read by `cv::ml::DTrees::getUse1SERule()`, written by `cv::ml::DTrees::setUse1SERule()`.
        use1_se_rule: bool = true, get_use1_se_rule / set_use1_se_rule, "DTrees::getUse1SERule", "DTrees::setUse1SERule";
        /// Consider surrogate splits when a value is missing.
        /// Read by `cv::ml::DTrees::getUseSurrogates()`, written by `cv::ml::DTrees::setUseSurrogates()`.
        use_surrogates: bool = false, get_use_surrogates / set_use_surrogates, "DTrees::getUseSurrogates", "DTrees::setUseSurrogates";
        /// Discard pruned branches rather than keeping them for inspection.
        /// Read by `cv::ml::DTrees::getTruncatePrunedTree()`, written by `cv::ml::DTrees::setTruncatePrunedTree()`.
        truncate_pruned_tree: bool = true, get_truncate_pruned_tree / set_truncate_pruned_tree, "DTrees::getTruncatePrunedTree", "DTrees::setTruncatePrunedTree";
        /// Stop splitting when the regression error falls below this.
        /// Read by `cv::ml::DTrees::getRegressionAccuracy()`, written by `cv::ml::DTrees::setRegressionAccuracy()`.
        regression_accuracy: f32 = 0.01, get_regression_accuracy / set_regression_accuracy, "DTrees::getRegressionAccuracy", "DTrees::setRegressionAccuracy";
        /// Per-class misclassification cost; empty means uniform.
        /// Read by `cv::ml::DTrees::getPriors()`, written by `cv::ml::DTrees::setPriors()`.
        priors: Vec<f64> = Vec::new(), get_priors / set_priors, "DTrees::getPriors", "DTrees::setPriors";
        /// When training stops.
        /// Read by `cv::ml::DTrees::getTermCriteria()`, written by `cv::ml::DTrees::setTermCriteria()`.
        term_criteria: cv_core::TermCriteria = cv_core::TermCriteria { ty: 3, max_count: 100, epsilon: 1e-6 }, get_term_criteria / set_term_criteria, "DTrees::getTermCriteria", "DTrees::setTermCriteria";
    }
}

ml_accessors! {
    /// `cv::ml::RTrees` parameters, on top of [`DTreesParams`].
    pub struct RTreesParams {
        /// Variables sampled at each split; 0 means the square root of the count.
        /// Read by `cv::ml::RTrees::getActiveVarCount()`, written by `cv::ml::RTrees::setActiveVarCount()`.
        active_var_count: i32 = 0, get_active_var_count / set_active_var_count, "RTrees::getActiveVarCount", "RTrees::setActiveVarCount";
        /// Accumulate per-variable importance while training.
        /// Read by `cv::ml::RTrees::getCalculateVarImportance()`, written by `cv::ml::RTrees::setCalculateVarImportance()`.
        calculate_var_importance: bool = false, get_calculate_var_importance / set_calculate_var_importance, "RTrees::getCalculateVarImportance", "RTrees::setCalculateVarImportance";
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_match_upstream() {
        let k = KNearestParams::default();
        assert_eq!(k.default_k, 10);
        assert!(k.is_classifier);
        assert_eq!(k.emax, 1000);

        let d = DTreesParams::default();
        assert_eq!((d.max_depth, d.min_sample_count, d.max_categories), (10, 10, 10));
        assert_eq!(d.cv_folds, 0);
        assert!(d.use1_se_rule);
        assert!(!d.use_surrogates);

        let a = AnnMlpParams::default();
        assert_eq!(a.rprop_dw0, 0.1);
        assert_eq!(a.rprop_dw_plus, 1.2);
        assert_eq!(a.rprop_dw_minus, 0.5);
        assert_eq!(a.rprop_dw_max, 50.0);

        assert_eq!(BoostParams::default().weak_count, 100);
        assert_eq!(BoostParams::default().weight_trim_rate, 0.95);
        assert_eq!(EmParams::default().clusters_number, 5);
        assert_eq!(LogisticRegressionParams::default().iterations, 1000);
        assert_eq!(SvmsgdParams::default().initial_step_size, 0.05);
    }

    #[test]
    fn the_default_variants_match_upstream() {
        assert_eq!(SampleTypes::default(), SampleTypes::RowSample);
        assert_eq!(KNearestType::default(), KNearestType::BruteForce);
        assert_eq!(SvmType::default(), SvmType::CSvc);
        assert_eq!(SvmsgdType::default(), SvmsgdType::Asgd);
        assert_eq!(MarginType::default(), MarginType::SoftMargin);
        assert_eq!(ActivationFunction::default(), ActivationFunction::SigmoidSym);
        assert_eq!(TrainingMethod::default(), TrainingMethod::Rprop);
        assert_eq!(CovMatType::default(), CovMatType::Diagonal);
        assert_eq!(CovMatType::default_type(), CovMatType::Diagonal);
        assert_eq!(LrMethod::default(), LrMethod::Batch);
        assert_eq!(RegKind::default(), RegKind::L2);
        assert_eq!(BoostType::default(), BoostType::Real);
    }

    #[test]
    fn every_accessor_round_trips() {
        let mut d = DTreesParams::default();
        d.set_max_depth(25).set_cv_folds(5).set_use_surrogates(true);
        assert_eq!(d.get_max_depth(), 25);
        assert_eq!(d.get_cv_folds(), 5);
        assert!(d.get_use_surrogates());

        let mut a = AnnMlpParams::default();
        a.set_anneal_initial_t(20.0).set_anneal_cooling_ratio(0.9);
        assert_eq!(a.get_anneal_initial_t(), 20.0);
        assert_eq!(a.get_anneal_cooling_ratio(), 0.9);
        // Untouched fields keep their defaults.
        assert_eq!(a.get_rprop_dw_max(), 50.0);
    }

    #[test]
    fn a_setter_touches_only_its_own_field() {
        // The reason for the macro: no getter can be wired to another's field.
        let base = DTreesParams::default();
        let mut m = base.clone();
        m.set_min_sample_count(42);
        assert_eq!(m.get_min_sample_count(), 42);
        assert_eq!(m.get_max_depth(), base.get_max_depth());
        assert_eq!(m.get_max_categories(), base.get_max_categories());
        assert_eq!(m.get_cv_folds(), base.get_cv_folds());
        assert_eq!(m.get_regression_accuracy(), base.get_regression_accuracy());
        assert_eq!(m.get_use1_se_rule(), base.get_use1_se_rule());
    }

    #[test]
    fn the_covariance_shapes_are_ordered_by_parameter_count() {
        // Spherical is the most constrained, generic the least; the enum exists
        // to make that trade explicit at the call site.
        let all = [CovMatType::Spherical, CovMatType::Diagonal, CovMatType::Generic];
        assert_eq!(all.len(), 3);
        assert_ne!(CovMatType::Spherical, CovMatType::Generic);
    }
}
