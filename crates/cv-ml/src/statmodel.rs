//! The `StatModel` interface, the `TrainData` accessor surface, and the model
//! accessors that read learned state (`modules/ml/include/opencv2/ml.hpp`).
//!
//! [`crate::params`] holds the tunables — the `get*`/`set*` pairs over plain
//! configuration. This module holds everything that needs the *data* or the
//! *fitted model*: how many samples there are, which rows are held out, what
//! the support vectors turned out to be, what a mixture's means came to.
//!
//! # Why `StatModel` is one trait
//!
//! Upstream's `StatModel` is what makes a model swappable: `train`, `predict`,
//! `calcError`, `save`, `load`, and a few questions about what kind of model it
//! is. Every model here implements [`StatModel`], so a caller can hold one
//! behind a generic and compare a tree against an SVM without knowing which it
//! has. Without that, "which classifier works best on this data" is a
//! copy-and-paste exercise.
//!
//! # Persistence
//!
//! [`StatModel::save`] and [`StatModel::load`] round-trip through
//! [`cv_core::FileStorage`] under the same top-level node name upstream uses
//! (`opencv_ml_svm`, `opencv_ml_rtrees`, …), so a file written here is
//! recognisable. The *contents* are this port's own layout — upstream's node
//! names describe upstream's internal representation, and reproducing those
//! for a differently-structured model would be a fiction.

use cv_core::{cv_assert, cv_error, Code, FileStorage, Node, Result, TermCriteria};

use crate::params::{SampleTypes, SvmParamType, SvmType, VariableTypes};
use crate::{
    AnnMlp, DTree, Em, KNearest, LogisticRegression, NormalBayes, ParamGrid, RTrees, Svm,
    SvmKernel, SvmParams, TrainData, TreeNode,
};

// ---------------------------------------------------------------------------
// Node helpers
// ---------------------------------------------------------------------------

/// A flat list of reals.
fn seq(v: &[f64]) -> Node {
    Node::Seq(v.iter().map(|&x| Node::Real(x)).collect())
}

/// A list of rows.
fn grid(rows: &[Vec<f64>]) -> Node {
    Node::Seq(rows.iter().map(|r| seq(r)).collect())
}

/// A named mapping.
fn map(pairs: Vec<(&str, Node)>) -> Node {
    Node::Map(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// The child called `key`, or an error naming what was expected.
fn field<'a>(node: &'a Node, key: &str) -> Result<&'a Node> {
    node.get(key)
        .ok_or_else(|| cv_core::Error::new(Code::StsParseError, format!("missing field `{key}`")))
}

/// A real child.
fn real_at(node: &Node, key: &str) -> Result<f64> {
    let n = field(node, key)?;
    n.as_real()
        .or_else(|| n.as_int().map(|i| i as f64))
        .ok_or_else(|| cv_core::Error::new(Code::StsParseError, format!("`{key}` is not a number")))
}

/// An integer child.
fn int_at(node: &Node, key: &str) -> Result<i64> {
    let n = field(node, key)?;
    n.as_int()
        .or_else(|| n.as_real().map(|r| r as i64))
        .ok_or_else(|| cv_core::Error::new(Code::StsParseError, format!("`{key}` is not an int")))
}

/// A string child.
fn str_at(node: &Node, key: &str) -> Result<String> {
    Ok(field(node, key)?
        .as_str()
        .ok_or_else(|| cv_core::Error::new(Code::StsParseError, format!("`{key}` is not a string")))?
        .to_string())
}

/// A flat list of reals, read back.
fn seq_at(node: &Node, key: &str) -> Result<Vec<f64>> {
    let items = field(node, key)?.as_seq().ok_or_else(|| {
        cv_core::Error::new(Code::StsParseError, format!("`{key}` is not a sequence"))
    })?;
    items
        .iter()
        .map(|n| {
            n.as_real().or_else(|| n.as_int().map(|i| i as f64)).ok_or_else(|| {
                cv_core::Error::new(Code::StsParseError, format!("`{key}` holds a non-number"))
            })
        })
        .collect()
}

/// A list of rows, read back.
fn grid_at(node: &Node, key: &str) -> Result<Vec<Vec<f64>>> {
    let rows = field(node, key)?.as_seq().ok_or_else(|| {
        cv_core::Error::new(Code::StsParseError, format!("`{key}` is not a sequence"))
    })?;
    rows.iter()
        .map(|r| {
            let items = r.as_seq().ok_or_else(|| {
                cv_core::Error::new(Code::StsParseError, format!("`{key}` holds a non-row"))
            })?;
            items
                .iter()
                .map(|n| {
                    n.as_real().or_else(|| n.as_int().map(|i| i as f64)).ok_or_else(|| {
                        cv_core::Error::new(
                            Code::StsParseError,
                            format!("`{key}` holds a non-number"),
                        )
                    })
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// StatModel
// ---------------------------------------------------------------------------

/// `cv::ml::StatModel` — the interface every model implements.
pub trait StatModel: Sized {
    /// The top-level node a saved model is stored under, matching upstream's
    /// naming (`opencv_ml_svm`, `opencv_ml_rtrees`, …).
    const STORAGE_KEY: &'static str;

    /// `cv::ml::StatModel::getVarCount()` — features the model expects.
    fn get_var_count(&self) -> i32;

    /// `cv::ml::StatModel::empty()`.
    fn empty(&self) -> bool;

    /// `cv::ml::StatModel::isTrained()`.
    fn is_trained(&self) -> bool {
        !self.empty()
    }

    /// `cv::ml::StatModel::isClassifier()`.
    fn is_classifier(&self) -> bool;

    /// `cv::ml::StatModel::predict(samples, results, flags)` for one sample.
    fn predict_one(&self, sample: &[f64]) -> Result<f64>;

    /// The model's learned state as a storage node.
    fn to_node(&self) -> Node;

    /// The inverse of [`Self::to_node`].
    fn from_node(node: &Node) -> Result<Self>;

    /// `cv::ml::StatModel::calcError(data, test, resp)`.
    ///
    /// The misclassification **percentage** for a classifier and the RMS error
    /// for a regressor — upstream's two conventions, kept because a caller
    /// comparing against cv2 would otherwise be off by a factor of a hundred.
    ///
    /// `test` selects the held-out rows; see
    /// [`TrainData::set_train_test_split`]. With no split set, both halves are
    /// the whole set, and a "test" error is really a training error — which is
    /// exactly what upstream does, and worth knowing before trusting the
    /// number.
    fn calc_error(&self, data: &TrainData, test: bool) -> Result<f64> {
        let idx = if test { data.get_test_sample_idx() } else { data.get_train_sample_idx() };
        cv_assert!(!idx.is_empty(), "no samples on the requested side of the split");
        let mut acc = 0.0f64;
        for &i in &idx {
            let i = i as usize;
            let got = self.predict_one(&data.samples[i])?;
            let want = data.responses[i];
            if self.is_classifier() {
                acc += f64::from(u8::from((got - want).abs() > 1e-9));
            } else {
                acc += (got - want) * (got - want);
            }
        }
        let n = idx.len() as f64;
        Ok(if self.is_classifier() { 100.0 * acc / n } else { (acc / n).sqrt() })
    }

    /// `cv::ml::StatModel::save(filename)`.
    fn save(&self, filename: &str) -> Result<()> {
        let mut fs = FileStorage::new();
        fs.set(Self::STORAGE_KEY, self.to_node());
        fs.write_file(filename)
    }

    /// `cv::ml::StatModel::load(filename, objname)`.
    ///
    /// Reads the node this model type writes. A file holding a *different*
    /// model is an error rather than a silently empty result: loading an SVM
    /// as a decision tree should not look like a tree with no splits.
    fn load(filename: &str) -> Result<Self> {
        let fs = FileStorage::read_file(filename)?;
        let node = fs.get(Self::STORAGE_KEY).ok_or_else(|| {
            cv_core::Error::new(
                Code::StsParseError,
                format!("`{filename}` holds no `{}` node", Self::STORAGE_KEY),
            )
        })?;
        Self::from_node(node)
    }
}

// ---------------------------------------------------------------------------
// TrainData
// ---------------------------------------------------------------------------

impl TrainData {
    /// `cv::ml::TrainData::getSamples()`.
    #[must_use]
    pub fn get_samples(&self) -> &[Vec<f64>] {
        &self.samples
    }

    /// `cv::ml::TrainData::getResponses()`.
    #[must_use]
    pub fn get_responses(&self) -> &[f64] {
        &self.responses
    }

    /// `cv::ml::TrainData::getNSamples()`.
    #[must_use]
    pub fn get_n_samples(&self) -> i32 {
        self.samples.len() as i32
    }

    /// `cv::ml::TrainData::getNAllVars()` — variables in the data.
    #[must_use]
    pub fn get_n_all_vars(&self) -> i32 {
        self.samples.first().map_or(0, |s| s.len() as i32)
    }

    /// `cv::ml::TrainData::getNVars()` — variables actually used.
    #[must_use]
    pub fn get_n_vars(&self) -> i32 {
        if self.var_idx.is_empty() {
            self.get_n_all_vars()
        } else {
            self.var_idx.len() as i32
        }
    }

    /// `cv::ml::TrainData::getLayout()`.
    #[must_use]
    pub fn get_layout(&self) -> SampleTypes {
        self.layout
    }

    /// `cv::ml::TrainData::getVarIdx()`.
    #[must_use]
    pub fn get_var_idx(&self) -> &[i32] {
        &self.var_idx
    }

    /// `cv::ml::TrainData::getVarType()`.
    ///
    /// One entry per variable plus a trailing one for the response. With
    /// nothing set, every variable is ordered and the response is categorical
    /// — upstream's assumption for a classification set.
    #[must_use]
    pub fn get_var_type(&self) -> Vec<VariableTypes> {
        if !self.var_type.is_empty() {
            return self.var_type.clone();
        }
        let mut v = vec![VariableTypes::Numerical; self.get_n_all_vars().max(0) as usize];
        v.push(VariableTypes::Categorical);
        v
    }

    /// `cv::ml::TrainData::getVarSymbolFlags()` — which variables were read
    /// from symbolic (string) columns.
    ///
    /// All zero. That flag records what a CSV *parser* saw, and nothing here
    /// parses CSV; the values arrive as numbers already.
    #[must_use]
    pub fn get_var_symbol_flags(&self) -> Vec<u8> {
        vec![0u8; self.get_n_all_vars().max(0) as usize]
    }

    /// `cv::ml::TrainData::getResponseType()`.
    #[must_use]
    pub fn get_response_type(&self) -> VariableTypes {
        *self.get_var_type().last().unwrap_or(&VariableTypes::Categorical)
    }

    /// `cv::ml::TrainData::getNames(names)` — the variable names.
    #[must_use]
    pub fn get_names(&self) -> &[String] {
        &self.names
    }

    /// `cv::ml::TrainData::getMissing()` — the missing-value mask.
    ///
    /// Every entry false when nothing was declared missing, rather than an
    /// empty matrix, so a caller can index it without a shape check.
    #[must_use]
    pub fn get_missing(&self) -> Vec<Vec<bool>> {
        if self.missing.is_empty() {
            vec![vec![false; self.get_n_all_vars().max(0) as usize]; self.samples.len()]
        } else {
            self.missing.clone()
        }
    }

    /// `cv::ml::TrainData::getSampleWeights()`.
    #[must_use]
    pub fn get_sample_weights(&self) -> Vec<f64> {
        if self.sample_weights.is_empty() {
            vec![1.0; self.samples.len()]
        } else {
            self.sample_weights.clone()
        }
    }

    /// `cv::ml::TrainData::getSample(varIdx, sidx, buf)` — one row, restricted
    /// to the variables in use.
    pub fn get_sample(&self, sidx: usize) -> Result<Vec<f64>> {
        cv_assert!(sidx < self.samples.len(), "sample {sidx} is past the end");
        let row = &self.samples[sidx];
        if self.var_idx.is_empty() {
            return Ok(row.clone());
        }
        self.var_idx
            .iter()
            .map(|&v| {
                let v = v as usize;
                row.get(v).copied().ok_or_else(|| {
                    cv_core::Error::new(Code::StsOutOfRange, format!("varIdx {v} is past the end"))
                })
            })
            .collect()
    }

    /// `cv::ml::TrainData::getValues(vi, sidx, values)` — one variable across
    /// the named samples.
    pub fn get_values(&self, vi: usize, sidx: &[i32]) -> Result<Vec<f64>> {
        sidx.iter()
            .map(|&i| {
                self.samples
                    .get(i.max(0) as usize)
                    .and_then(|r| r.get(vi))
                    .copied()
                    .ok_or_else(|| {
                        cv_core::Error::new(
                            Code::StsOutOfRange,
                            format!("no variable {vi} in sample {i}"),
                        )
                    })
            })
            .collect()
    }

    /// `cv::ml::TrainData::getClassLabels()` — distinct responses, ascending.
    #[must_use]
    pub fn get_class_labels(&self) -> Vec<f64> {
        self.classes()
    }

    /// `cv::ml::TrainData::getNormCatResponses()` — each response replaced by
    /// its index in [`Self::get_class_labels`].
    ///
    /// That indirection is what lets a model store per-class state in a plain
    /// array: the labels themselves may be any numbers at all.
    #[must_use]
    pub fn get_norm_cat_responses(&self) -> Vec<i32> {
        let labels = self.classes();
        self.responses
            .iter()
            .map(|r| labels.iter().position(|l| l == r).unwrap_or(0) as i32)
            .collect()
    }

    /// `cv::ml::TrainData::getCatCount(vi)` — distinct values of a variable.
    pub fn get_cat_count(&self, vi: usize) -> Result<i32> {
        cv_assert!(
            vi < self.get_n_all_vars().max(0) as usize,
            "variable {vi} is past the end"
        );
        let mut seen: Vec<f64> = self.samples.iter().map(|s| s[vi]).collect();
        seen.sort_by(|a, b| a.total_cmp(b));
        seen.dedup();
        Ok(seen.len() as i32)
    }

    /// `cv::ml::TrainData::getCatMap()` — every categorical variable's values,
    /// concatenated, sliced by [`Self::get_cat_ofs`].
    #[must_use]
    pub fn get_cat_map(&self) -> Vec<f64> {
        let types = self.get_var_type();
        let mut out = Vec::new();
        for vi in 0..self.get_n_all_vars().max(0) as usize {
            if types.get(vi) != Some(&VariableTypes::Categorical) {
                continue;
            }
            let mut vals: Vec<f64> = self.samples.iter().map(|s| s[vi]).collect();
            vals.sort_by(|a, b| a.total_cmp(b));
            vals.dedup();
            out.extend(vals);
        }
        out
    }

    /// `cv::ml::TrainData::getCatOfs()` — `(start, end)` into
    /// [`Self::get_cat_map`] per variable; `(0, 0)` for an ordered one.
    #[must_use]
    pub fn get_cat_ofs(&self) -> Vec<(i32, i32)> {
        let types = self.get_var_type();
        let mut out = Vec::new();
        let mut at = 0i32;
        for vi in 0..self.get_n_all_vars().max(0) as usize {
            if types.get(vi) != Some(&VariableTypes::Categorical) {
                out.push((0, 0));
                continue;
            }
            let n = self.get_cat_count(vi).unwrap_or(0);
            out.push((at, at + n));
            at += n;
        }
        out
    }

    /// `cv::ml::TrainData::getDefaultSubstValues()` — what to put in place of a
    /// missing value: the mean of the variable over the rows that have one.
    #[must_use]
    pub fn get_default_subst_values(&self) -> Vec<f64> {
        let missing = self.get_missing();
        (0..self.get_n_all_vars().max(0) as usize)
            .map(|vi| {
                let (sum, n) = self.samples.iter().zip(&missing).fold(
                    (0.0f64, 0usize),
                    |(s, n), (row, m)| {
                        if m.get(vi).copied().unwrap_or(false) {
                            (s, n)
                        } else {
                            (s + row[vi], n + 1)
                        }
                    },
                );
                if n == 0 {
                    0.0
                } else {
                    sum / n as f64
                }
            })
            .collect()
    }

    /// `cv::ml::TrainData::getTrainSampleIdx()`; every row when unsplit.
    #[must_use]
    pub fn get_train_sample_idx(&self) -> Vec<i32> {
        if self.train_sample_idx.is_empty() {
            (0..self.samples.len() as i32).collect()
        } else {
            self.train_sample_idx.clone()
        }
    }

    /// `cv::ml::TrainData::getTestSampleIdx()`.
    ///
    /// Every row when unsplit — which matches upstream, and means a "test"
    /// error measured without a split is a training error.
    #[must_use]
    pub fn get_test_sample_idx(&self) -> Vec<i32> {
        if self.test_sample_idx.is_empty() {
            (0..self.samples.len() as i32).collect()
        } else {
            self.test_sample_idx.clone()
        }
    }

    /// `cv::ml::TrainData::getNTrainSamples()`.
    #[must_use]
    pub fn get_n_train_samples(&self) -> i32 {
        self.get_train_sample_idx().len() as i32
    }

    /// `cv::ml::TrainData::getNTestSamples()`.
    #[must_use]
    pub fn get_n_test_samples(&self) -> i32 {
        self.get_test_sample_idx().len() as i32
    }

    /// `cv::ml::TrainData::getTrainSamples(layout, compressSamples, compressVars)`.
    #[must_use]
    pub fn get_train_samples(&self) -> Vec<Vec<f64>> {
        Self::get_sub_matrix(&self.samples, &self.get_train_sample_idx())
    }

    /// `cv::ml::TrainData::getTestSamples()`.
    #[must_use]
    pub fn get_test_samples(&self) -> Vec<Vec<f64>> {
        Self::get_sub_matrix(&self.samples, &self.get_test_sample_idx())
    }

    /// `cv::ml::TrainData::getTrainResponses()`.
    #[must_use]
    pub fn get_train_responses(&self) -> Vec<f64> {
        Self::get_sub_vector(&self.responses, &self.get_train_sample_idx())
    }

    /// `cv::ml::TrainData::getTestResponses()`.
    #[must_use]
    pub fn get_test_responses(&self) -> Vec<f64> {
        Self::get_sub_vector(&self.responses, &self.get_test_sample_idx())
    }

    /// `cv::ml::TrainData::getTrainNormCatResponses()`.
    #[must_use]
    pub fn get_train_norm_cat_responses(&self) -> Vec<i32> {
        let all = self.get_norm_cat_responses();
        self.get_train_sample_idx().iter().map(|&i| all[i as usize]).collect()
    }

    /// `cv::ml::TrainData::getTestNormCatResponses()`.
    #[must_use]
    pub fn get_test_norm_cat_responses(&self) -> Vec<i32> {
        let all = self.get_norm_cat_responses();
        self.get_test_sample_idx().iter().map(|&i| all[i as usize]).collect()
    }

    /// `cv::ml::TrainData::getTrainSampleWeights()`.
    #[must_use]
    pub fn get_train_sample_weights(&self) -> Vec<f64> {
        Self::get_sub_vector(&self.get_sample_weights(), &self.get_train_sample_idx())
    }

    /// `cv::ml::TrainData::getTestSampleWeights()`.
    #[must_use]
    pub fn get_test_sample_weights(&self) -> Vec<f64> {
        Self::get_sub_vector(&self.get_sample_weights(), &self.get_test_sample_idx())
    }

    /// `cv::ml::TrainData::getSubMatrix(matrix, idx, layout)` — the named rows.
    #[must_use]
    pub fn get_sub_matrix(matrix: &[Vec<f64>], idx: &[i32]) -> Vec<Vec<f64>> {
        idx.iter().filter_map(|&i| matrix.get(i.max(0) as usize).cloned()).collect()
    }

    /// `cv::ml::TrainData::getSubVector(vec, idx)` — the named entries.
    #[must_use]
    pub fn get_sub_vector(v: &[f64], idx: &[i32]) -> Vec<f64> {
        idx.iter().filter_map(|&i| v.get(i.max(0) as usize).copied()).collect()
    }

    /// `cv::ml::TrainData::setTrainTestSplit(count, shuffle)`.
    ///
    /// The first `count` rows train and the rest are held out. `rng` is
    /// explicit where upstream reaches for a global one, so a split is
    /// reproducible from its seed — an unreproducible split makes a reported
    /// test error impossible to check.
    pub fn set_train_test_split(
        &mut self,
        count: usize,
        rng: Option<&mut cv_core::rng::Rng>,
    ) -> Result<()> {
        cv_assert!(
            count <= self.samples.len(),
            "cannot train on {count} of {} samples",
            self.samples.len()
        );
        let mut idx: Vec<i32> = (0..self.samples.len() as i32).collect();
        if let Some(rng) = rng {
            Self::shuffle(&mut idx, rng);
        }
        self.test_sample_idx = idx.split_off(count);
        self.train_sample_idx = idx;
        Ok(())
    }

    /// `cv::ml::TrainData::setTrainTestSplitRatio(ratio, shuffle)`.
    pub fn set_train_test_split_ratio(
        &mut self,
        ratio: f64,
        rng: Option<&mut cv_core::rng::Rng>,
    ) -> Result<()> {
        cv_assert!((0.0..=1.0).contains(&ratio), "the split ratio must be in 0..1, got {ratio}");
        // Round rather than truncate, so a ratio of 0.5 on an odd count does
        // not silently drop a training row.
        let count = (ratio * self.samples.len() as f64).round() as usize;
        self.set_train_test_split(count.min(self.samples.len()), rng)
    }

    /// `cv::ml::TrainData::shuffleTrainTest()` — reshuffle which rows are held
    /// out, keeping the sizes.
    pub fn shuffle_train_test(&mut self, rng: &mut cv_core::rng::Rng) {
        let n_train = self.get_n_train_samples() as usize;
        let mut idx: Vec<i32> = (0..self.samples.len() as i32).collect();
        Self::shuffle(&mut idx, rng);
        self.test_sample_idx = idx.split_off(n_train.min(idx.len()));
        self.train_sample_idx = idx;
    }

    /// Fisher-Yates, so every permutation is equally likely.
    fn shuffle(idx: &mut [i32], rng: &mut cv_core::rng::Rng) {
        for i in (1..idx.len()).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            idx.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// SVM configuration and learned state
// ---------------------------------------------------------------------------

impl SvmParams {
    /// `cv::ml::SVM::getC()`.
    #[must_use]
    pub fn get_c(&self) -> f64 {
        self.c
    }

    /// `cv::ml::SVM::getNu()`.
    #[must_use]
    pub fn get_nu(&self) -> f64 {
        self.nu
    }
    /// `cv::ml::SVM::setNu(val)`.
    pub fn set_nu(&mut self, val: f64) -> &mut Self {
        self.nu = val;
        self
    }

    /// `cv::ml::SVM::getP()`.
    #[must_use]
    pub fn get_p(&self) -> f64 {
        self.p
    }
    /// `cv::ml::SVM::setP(val)`.
    pub fn set_p(&mut self, val: f64) -> &mut Self {
        self.p = val;
        self
    }

    /// `cv::ml::SVM::getType()`.
    #[must_use]
    pub fn get_type(&self) -> SvmType {
        self.svm_type
    }
    /// `cv::ml::SVM::setType(val)`.
    pub fn set_type(&mut self, val: SvmType) -> &mut Self {
        self.svm_type = val;
        self
    }

    /// `cv::ml::SVM::getKernelType()`.
    #[must_use]
    pub fn get_kernel_type(&self) -> crate::SvmKernelKind {
        use crate::SvmKernelKind as K;
        match self.kernel {
            SvmKernel::Linear => K::Linear,
            SvmKernel::Poly { .. } => K::Poly,
            SvmKernel::Rbf { .. } => K::Rbf,
            SvmKernel::Sigmoid { .. } => K::Sigmoid,
            SvmKernel::Chi2 { .. } => K::Chi2,
            SvmKernel::Inter => K::Inter,
        }
    }

    /// `cv::ml::SVM::setKernel(kernelType)`.
    ///
    /// Switching family keeps whatever `gamma`, `coef0` and `degree` were set,
    /// so `setKernel` after `setGamma` does not quietly discard the gamma.
    pub fn set_kernel(&mut self, kind: crate::SvmKernelKind) -> &mut Self {
        let (gamma, coef0, degree) = (self.get_gamma(), self.get_coef0(), self.get_degree());
        use crate::SvmKernelKind as K;
        self.kernel = match kind {
            K::Linear => SvmKernel::Linear,
            K::Poly => SvmKernel::Poly { degree, gamma, coef0 },
            K::Rbf => SvmKernel::Rbf { gamma },
            K::Sigmoid => SvmKernel::Sigmoid { gamma, coef0 },
            K::Chi2 => SvmKernel::Chi2 { gamma },
            K::Inter => SvmKernel::Inter,
        };
        self
    }

    /// `cv::ml::SVM::getGamma()`; 0 for a kernel that has none.
    #[must_use]
    pub fn get_gamma(&self) -> f64 {
        match self.kernel {
            SvmKernel::Poly { gamma, .. }
            | SvmKernel::Rbf { gamma }
            | SvmKernel::Sigmoid { gamma, .. }
            | SvmKernel::Chi2 { gamma } => gamma,
            SvmKernel::Linear | SvmKernel::Inter => 0.0,
        }
    }

    /// `cv::ml::SVM::setGamma(val)`; ignored by a kernel that has no gamma.
    pub fn set_gamma(&mut self, val: f64) -> &mut Self {
        match &mut self.kernel {
            SvmKernel::Poly { gamma, .. }
            | SvmKernel::Rbf { gamma }
            | SvmKernel::Sigmoid { gamma, .. }
            | SvmKernel::Chi2 { gamma } => *gamma = val,
            SvmKernel::Linear | SvmKernel::Inter => {}
        }
        self
    }

    /// `cv::ml::SVM::getCoef0()`.
    #[must_use]
    pub fn get_coef0(&self) -> f64 {
        match self.kernel {
            SvmKernel::Poly { coef0, .. } | SvmKernel::Sigmoid { coef0, .. } => coef0,
            _ => 0.0,
        }
    }

    /// `cv::ml::SVM::setCoef0(val)`.
    pub fn set_coef0(&mut self, val: f64) -> &mut Self {
        match &mut self.kernel {
            SvmKernel::Poly { coef0, .. } | SvmKernel::Sigmoid { coef0, .. } => *coef0 = val,
            _ => {}
        }
        self
    }

    /// `cv::ml::SVM::getDegree()`.
    #[must_use]
    pub fn get_degree(&self) -> f64 {
        match self.kernel {
            SvmKernel::Poly { degree, .. } => degree,
            _ => 0.0,
        }
    }

    /// `cv::ml::SVM::setDegree(val)`.
    pub fn set_degree(&mut self, val: f64) -> &mut Self {
        if let SvmKernel::Poly { degree, .. } = &mut self.kernel {
            *degree = val;
        }
        self
    }

    /// `cv::ml::SVM::getTermCriteria()`.
    #[must_use]
    pub fn get_term_criteria(&self) -> TermCriteria {
        TermCriteria {
            ty: cv_core::geometry::term_criteria::COUNT | cv_core::geometry::term_criteria::EPS,
            max_count: self.max_iter as i32,
            epsilon: self.tolerance,
        }
    }

    /// `cv::ml::SVM::setTermCriteria(val)`.
    pub fn set_term_criteria(&mut self, val: TermCriteria) -> &mut Self {
        if val.max_count > 0 {
            self.max_iter = val.max_count as usize;
        }
        if val.epsilon > 0.0 {
            self.tolerance = val.epsilon;
        }
        self
    }
}

impl ParamGrid {
    /// `cv::ml::SVM::getDefaultGridPtr(param_id)`.
    #[must_use]
    pub fn get_default_grid_ptr(param_id: SvmParamType) -> ParamGrid {
        match param_id {
            SvmParamType::C => ParamGrid::default_c(),
            SvmParamType::Gamma => ParamGrid::default_gamma(),
            SvmParamType::Degree => ParamGrid::default_degree(),
            SvmParamType::Coef => ParamGrid::default_coef0(),
            // Neither `p` nor `nu` is searched by the C_SVC trained here, so
            // upstream's grids are reported rather than invented.
            SvmParamType::P => ParamGrid::Log { min: 0.01, max: 100.0, step: 7.0 },
            SvmParamType::Nu => ParamGrid::Log { min: 0.01, max: 0.2, step: 3.0 },
        }
    }
}

impl Svm {
    /// `cv::ml::SVM::getClassWeights()`.
    #[must_use]
    pub fn get_class_weights(&self) -> &[f64] {
        &self.class_weights
    }

    /// `cv::ml::SVM::setClassWeights(val)`.
    ///
    /// One multiplier per label, in the order [`Svm::labels`] reports. Stored
    /// rather than applied: the machines are already fitted by the time this
    /// can be called, and silently leaving a set weight unused would be worse
    /// than requiring a retrain.
    pub fn set_class_weights(&mut self, weights: Vec<f64>) -> Result<()> {
        cv_assert!(
            weights.is_empty() || weights.len() == self.labels().len(),
            "{} weights for {} classes",
            weights.len(),
            self.labels().len()
        );
        self.class_weights = weights;
        Ok(())
    }

    /// `cv::ml::SVM::getUncompressedSupportVectors()`.
    ///
    /// Every pair's vectors with duplicates kept, where
    /// [`Svm::support_vectors`] merges them. For a linear kernel upstream
    /// compresses the whole model to one vector, which is why the two differ
    /// there and agree everywhere else.
    #[must_use]
    pub fn get_uncompressed_support_vectors(&self) -> Vec<Vec<f64>> {
        self.machines.iter().flat_map(|(_, _, m)| m.sv.iter().cloned()).collect()
    }

    /// `cv::ml::SVM::getDecisionFunction(i, alpha, svidx)`.
    ///
    /// Returns `(rho, alpha, svidx)` for the `i`th one-vs-one machine: the
    /// bias, the signed coefficients, and which rows of
    /// [`Svm::get_uncompressed_support_vectors`] they belong to.
    ///
    /// `rho` is **negated** relative to the `b` stored here, because that is
    /// upstream's sign convention — the decision is `sum(alpha*K) - rho`.
    pub fn get_decision_function(&self, i: usize) -> Result<(f64, Vec<f64>, Vec<i32>)> {
        cv_assert!(
            i < self.machines.len(),
            "decision function {i} of {}",
            self.machines.len()
        );
        let base: usize = self.machines[..i].iter().map(|(_, _, m)| m.sv.len()).sum();
        let (_, _, m) = &self.machines[i];
        let alpha: Vec<f64> = m.alpha.iter().zip(&m.y).map(|(a, y)| a * y).collect();
        let svidx: Vec<i32> = (0..m.sv.len()).map(|k| (base + k) as i32).collect();
        Ok((-m.b, alpha, svidx))
    }
}

// ---------------------------------------------------------------------------
// EM: learned state and the two half-steps
// ---------------------------------------------------------------------------

/// `cv::ml::EM`'s start-step and default constants.
pub mod em {
    /// `cv::ml::EM::START_E_STEP` — begin from the responsibilities.
    pub const START_E_STEP: i32 = 1;
    /// `cv::ml::EM::START_M_STEP` — begin from the parameters.
    pub const START_M_STEP: i32 = 2;
    /// `cv::ml::EM::START_AUTO_STEP` — seed internally, then run both.
    pub const START_AUTO_STEP: i32 = 0;
    /// `cv::ml::EM::DEFAULT_NCLUSTERS`.
    pub const DEFAULT_NCLUSTERS: i32 = 5;
    /// `cv::ml::EM::DEFAULT_MAX_ITERS`.
    pub const DEFAULT_MAX_ITERS: i32 = 100;
}

/// `cv::ml::SVM::CUSTOM` — the marker for a caller-supplied kernel.
///
/// Declared because a configuration may carry it; no kernel here is custom, so
/// [`SvmParams::get_kernel_type`] never returns it.
pub const CUSTOM: i32 = -1;

impl Em {
    /// `cv::ml::EM::getWeights()` — the mixing proportions.
    #[must_use]
    pub fn get_weights(&self) -> &[f64] {
        &self.weights
    }

    /// `cv::ml::EM::getMeans()`.
    #[must_use]
    pub fn get_means(&self) -> &[Vec<f64>] {
        &self.means
    }

    /// `cv::ml::EM::getCovs(covs)`.
    ///
    /// One matrix per component, diagonal — the fit is `COV_MAT_DIAGONAL`, so
    /// the off-diagonal entries are genuinely zero rather than unmodelled.
    #[must_use]
    pub fn get_covs(&self) -> Vec<Vec<Vec<f64>>> {
        self.variances
            .iter()
            .map(|v| {
                let d = v.len();
                (0..d)
                    .map(|i| (0..d).map(|j| if i == j { v[i] } else { 0.0 }).collect())
                    .collect()
            })
            .collect()
    }

    /// `cv::ml::EM::predict2(sample, probs)`.
    ///
    /// The log-likelihood of the sample and the component it most likely came
    /// from — upstream's `Vec2d`, in that order.
    #[must_use]
    pub fn predict2(&self, sample: &[f64]) -> (f64, i32) {
        let mut best = (f64::NEG_INFINITY, 0usize);
        let mut total = f64::NEG_INFINITY;
        for c in 0..self.weights.len() {
            let ll = self.component_log_likelihood(c, sample);
            if ll > best.0 {
                best = (ll, c);
            }
            total = log_sum_exp(total, ll);
        }
        (total, best.1 as i32)
    }

    /// `log(w_c) + log N(sample | mean_c, var_c)`.
    fn component_log_likelihood(&self, c: usize, sample: &[f64]) -> f64 {
        let mut ll = self.weights[c].max(1e-300).ln();
        for (j, sj) in sample.iter().enumerate() {
            let v = self.variances[c][j].max(1e-9);
            let d = sj - self.means[c][j];
            ll += -0.5 * (2.0 * core::f64::consts::PI * v).ln() - d * d / (2.0 * v);
        }
        ll
    }

    /// `cv::ml::EM::trainE(samples, means0, covs0, weights0, ...)`.
    ///
    /// Starts from supplied parameters and runs the E step first. Use this when
    /// the components are already roughly known — from a previous fit, or from
    /// domain knowledge — since a good start is what stops EM settling into a
    /// poor local optimum.
    pub fn train_e(
        samples: &[Vec<f64>],
        means0: &[Vec<f64>],
        covs0: &[Vec<f64>],
        weights0: &[f64],
        iterations: usize,
    ) -> Result<Self> {
        cv_assert!(!samples.is_empty(), "trainE needs at least one sample");
        cv_assert!(!means0.is_empty(), "trainE needs initial means");
        cv_assert!(
            means0.len() == covs0.len() && means0.len() == weights0.len(),
            "{} means, {} covariances and {} weights must agree",
            means0.len(),
            covs0.len(),
            weights0.len()
        );
        let d = samples[0].len();
        cv_assert!(
            means0.iter().all(|m| m.len() == d) && covs0.iter().all(|c| c.len() == d),
            "the initial parameters must have the sample dimension {d}"
        );
        let mut model = Em {
            weights: weights0.to_vec(),
            means: means0.to_vec(),
            variances: covs0.iter().map(|c| c.iter().map(|v| v.max(1e-9)).collect()).collect(),
        };
        for _ in 0..iterations {
            let resp = model.responsibilities(samples);
            model = Em::maximise(samples, &resp)?;
        }
        Ok(model)
    }

    /// `cv::ml::EM::trainM(samples, probs0, ...)`.
    ///
    /// Starts from supplied responsibilities and runs the M step first — the
    /// natural entry point when another method has already assigned samples to
    /// clusters and only the parameters are wanted.
    pub fn train_m(
        samples: &[Vec<f64>],
        probs0: &[Vec<f64>],
        iterations: usize,
    ) -> Result<Self> {
        cv_assert!(!samples.is_empty(), "trainM needs at least one sample");
        cv_assert!(
            probs0.len() == samples.len(),
            "{} responsibility rows for {} samples",
            probs0.len(),
            samples.len()
        );
        let mut model = Em::maximise(samples, probs0)?;
        for _ in 1..iterations.max(1) {
            let resp = model.responsibilities(samples);
            model = Em::maximise(samples, &resp)?;
        }
        Ok(model)
    }

    /// The E step: `P(component | sample)` for every pair.
    fn responsibilities(&self, samples: &[Vec<f64>]) -> Vec<Vec<f64>> {
        samples
            .iter()
            .map(|s| {
                let lls: Vec<f64> =
                    (0..self.weights.len()).map(|c| self.component_log_likelihood(c, s)).collect();
                let max = lls.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                // Subtract the maximum before exponentiating: without it a
                // log-likelihood around -800 underflows to zero and every
                // responsibility comes back NaN.
                let ex: Vec<f64> = lls.iter().map(|l| (l - max).exp()).collect();
                let sum: f64 = ex.iter().sum();
                ex.iter().map(|e| e / sum.max(1e-300)).collect()
            })
            .collect()
    }

    /// The M step: weights, means and variances from responsibilities.
    fn maximise(samples: &[Vec<f64>], resp: &[Vec<f64>]) -> Result<Self> {
        let k = resp.first().map_or(0, Vec::len);
        cv_assert!(k > 0, "the responsibilities name no components");
        let d = samples[0].len();
        let n = samples.len() as f64;
        let mut weights = vec![0.0; k];
        let mut means = vec![vec![0.0; d]; k];
        let mut variances = vec![vec![0.0; d]; k];
        for c in 0..k {
            let nc: f64 = resp.iter().map(|r| r[c]).sum::<f64>().max(1e-12);
            weights[c] = nc / n;
            for j in 0..d {
                means[c][j] = samples.iter().zip(resp).map(|(s, r)| r[c] * s[j]).sum::<f64>() / nc;
            }
            for j in 0..d {
                variances[c][j] = samples
                    .iter()
                    .zip(resp)
                    .map(|(s, r)| r[c] * (s[j] - means[c][j]).powi(2))
                    .sum::<f64>()
                    / nc;
                variances[c][j] = variances[c][j].max(1e-9);
            }
        }
        Ok(Em { weights, means, variances })
    }
}

/// `log(exp(a) + exp(b))` without overflowing.
fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

// ---------------------------------------------------------------------------
// The remaining per-model learned state
// ---------------------------------------------------------------------------

impl LogisticRegression {
    /// `cv::ml::LogisticRegression::get_learnt_thetas()`.
    ///
    /// One row per classifier — one here, since this is the binary form — with
    /// the bias **first**, which is upstream's layout.
    #[must_use]
    pub fn get_learnt_thetas(&self) -> Vec<Vec<f64>> {
        let mut row = vec![*self.weights.last().unwrap_or(&0.0)];
        row.extend(self.weights.iter().take(self.weights.len().saturating_sub(1)));
        vec![row]
    }
}

impl NormalBayes {
    /// `cv::ml::NormalBayesClassifier::predictProb(inputs, outputs, outputProbs, flags)`.
    ///
    /// The predicted label and the posterior over every class. The posteriors
    /// are normalised, so they sum to one — the raw log-likelihoods do not
    /// compare across samples, and a caller thresholding on "confidence" needs
    /// the normalised form.
    pub fn predict_prob(&self, sample: &[f64]) -> Result<(f64, Vec<f64>)> {
        cv_assert!(!self.classes.is_empty(), "the classifier is not trained");
        cv_assert!(
            sample.len() == self.stats[0].0.len(),
            "sample has {} features, the model has {}",
            sample.len(),
            self.stats[0].0.len()
        );
        let lls: Vec<f64> = self
            .stats
            .iter()
            .map(|(mean, var, logp)| {
                let mut ll = *logp;
                for (j, sj) in sample.iter().enumerate() {
                    let v = var[j].max(1e-12);
                    let d = sj - mean[j];
                    ll += -0.5 * (2.0 * core::f64::consts::PI * v).ln() - d * d / (2.0 * v);
                }
                ll
            })
            .collect();
        let max = lls.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ex: Vec<f64> = lls.iter().map(|l| (l - max).exp()).collect();
        let sum: f64 = ex.iter().sum::<f64>().max(1e-300);
        let probs: Vec<f64> = ex.iter().map(|e| e / sum).collect();
        let best = lls
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i);
        Ok((self.classes[best], probs))
    }
}

impl RTrees {
    /// `cv::ml::RTrees::getVotes(samples, results, flags)` — how many trees
    /// chose each class.
    #[must_use]
    pub fn get_votes(&self, sample: &[f64], labels: &[f64]) -> Vec<i32> {
        let mut votes = vec![0i32; labels.len()];
        for t in &self.trees {
            let p = t.predict(sample);
            if let Some(i) = labels.iter().position(|l| (l - p).abs() < 1e-9) {
                votes[i] += 1;
            }
        }
        votes
    }

    /// `cv::ml::RTrees::getVarImportance()`.
    ///
    /// How much each variable is used, as a share of all splits in the forest,
    /// weighted by how near the root the split is — a split at depth 0 divides
    /// every sample, one at depth 5 divides a thirty-second of them.
    ///
    /// This is a *usage* measure, not upstream's out-of-bag permutation
    /// importance, which needs the OOB rows retained per tree. It ranks
    /// variables the same way on separable data and is not the same number.
    #[must_use]
    pub fn get_var_importance(&self, n_vars: usize) -> Vec<f64> {
        let mut imp = vec![0.0f64; n_vars];
        for t in &self.trees {
            accumulate_importance(&t.root, 0, &mut imp);
        }
        let total: f64 = imp.iter().sum();
        if total > 0.0 {
            for v in &mut imp {
                *v /= total;
            }
        }
        imp
    }

    /// `cv::ml::RTrees::getOOBError()`.
    ///
    /// Requires the bootstrap masks, which [`RTrees::train`] does not keep, so
    /// this errors rather than returning a number that looks like an
    /// out-of-bag estimate and is not one. Use
    /// [`StatModel::calc_error`] against a held-out split instead.
    pub fn get_oob_error(&self) -> Result<f64> {
        cv_error!(
            Code::StsNotImplemented,
            "out-of-bag error needs the per-tree bootstrap masks, which training does not retain; \
             hold rows out with TrainData::set_train_test_split and use calcError"
        )
    }
}

/// Add each split's weight to its variable, halving with depth.
fn accumulate_importance(node: &TreeNode, depth: u32, imp: &mut [f64]) {
    if let TreeNode::Split { feature, left, right, .. } = node {
        if let Some(slot) = imp.get_mut(*feature) {
            *slot += 0.5f64.powi(depth as i32);
        }
        accumulate_importance(left, depth + 1, imp);
        accumulate_importance(right, depth + 1, imp);
    }
}

impl AnnMlp {
    /// `cv::ml::ANN_MLP::getWeights(layerIdx)` — layer 0 is input-to-hidden.
    pub fn get_weights(&self, layer_idx: usize) -> Result<Vec<Vec<f64>>> {
        match layer_idx {
            0 => Ok(self.w1.clone()),
            1 => Ok(self.w2.clone()),
            _ => cv_error!(
                Code::StsOutOfRange,
                "this network has two weight layers, asked for {layer_idx}"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// SVMSGD
// ---------------------------------------------------------------------------

/// `cv::ml::SVMSGD` — a linear SVM fitted by stochastic gradient descent.
///
/// Where [`Svm`] solves the dual with SMO and can use a kernel, this walks the
/// primal one sample at a time. That makes it linear-only, and makes it scale:
/// SMO holds a kernel matrix, SGD holds a weight vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Svmsgd {
    /// Weight vector, one entry per feature.
    weights: Vec<f64>,
    /// Bias.
    shift: f64,
    /// The two class labels, ascending; predictions come back as these.
    labels: Vec<f64>,
}

impl Svmsgd {
    /// `cv::ml::SVMSGD::train(trainData, flags)`.
    ///
    /// Two classes only, mapped to -1 and +1 internally and reported back as
    /// the original labels.
    pub fn train(
        data: &TrainData,
        p: &crate::params::SvmsgdParams,
        epochs: usize,
        rng: &mut cv_core::rng::Rng,
    ) -> Result<Self> {
        let labels = data.classes();
        cv_assert!(labels.len() == 2, "SVMSGD is binary; got {} classes", labels.len());
        cv_assert!(p.initial_step_size > 0.0, "the initial step size must be positive");
        cv_assert!(epochs > 0, "training needs at least one epoch");
        let d = data.dim();
        let mut w = vec![0.0f64; d];
        let mut b = 0.0f64;
        let lambda = f64::from(p.margin_regularization).max(1e-12);
        let mut t = 1u64;

        // Averaged SGD keeps the running mean of the iterates, which is what
        // makes ASGD steadier than plain SGD near the optimum.
        let averaged = p.svmsgd_type == crate::params::SvmsgdType::Asgd;
        let mut wbar = vec![0.0f64; d];
        let mut bbar = 0.0f64;
        let mut seen = 0u64;

        let hard = p.margin_type == crate::params::MarginType::HardMargin;
        for _ in 0..epochs {
            let mut order: Vec<usize> = (0..data.len()).collect();
            for i in (1..order.len()).rev() {
                let j = (rng.next_u32() as usize) % (i + 1);
                order.swap(i, j);
            }
            for &i in &order {
                let x = &data.samples[i];
                let y = if data.responses[i] == labels[1] { 1.0 } else { -1.0 };
                let eta = f64::from(p.initial_step_size)
                    / (t as f64).powf(f64::from(p.step_decreasing_power));
                let margin = y * (dot(&w, x) + b);
                for (wi, xi) in w.iter_mut().zip(x) {
                    // The regulariser applies every step; the hinge term only
                    // when the sample is inside the margin.
                    *wi -= eta * lambda * *wi;
                    if margin < 1.0 {
                        *wi += eta * y * xi;
                    }
                }
                // A hard margin has no bias term to fit: it assumes the classes
                // separate through the origin of the shifted space.
                if margin < 1.0 && !hard {
                    b += eta * y;
                }
                t += 1;
                if averaged {
                    seen += 1;
                    let k = seen as f64;
                    for (m, wi) in wbar.iter_mut().zip(&w) {
                        *m += (wi - *m) / k;
                    }
                    bbar += (b - bbar) / k;
                }
            }
        }
        Ok(if averaged {
            Svmsgd { weights: wbar, shift: bbar, labels }
        } else {
            Svmsgd { weights: w, shift: b, labels }
        })
    }

    /// `cv::ml::SVMSGD::getWeights()`.
    #[must_use]
    pub fn get_weights(&self) -> &[f64] {
        &self.weights
    }

    /// `cv::ml::SVMSGD::getShift()` — the bias.
    #[must_use]
    pub fn get_shift(&self) -> f64 {
        self.shift
    }

    /// `cv::ml::SVMSGD::setOptimalParameters(svmsgdType, marginType)`.
    ///
    /// Upstream's tuned starting points for the two combinations it documents.
    #[must_use]
    pub fn set_optimal_parameters(
        svmsgd_type: crate::params::SvmsgdType,
        margin_type: crate::params::MarginType,
    ) -> crate::params::SvmsgdParams {
        use crate::params::{MarginType, SvmsgdParams, SvmsgdType};
        let mut p = SvmsgdParams { svmsgd_type, margin_type, ..SvmsgdParams::default() };
        match (svmsgd_type, margin_type) {
            (SvmsgdType::Sgd, _) => {
                p.margin_regularization = 0.0001;
                p.initial_step_size = 0.05;
                p.step_decreasing_power = 1.0;
            }
            (SvmsgdType::Asgd, MarginType::SoftMargin) => {
                p.margin_regularization = 0.00001;
                p.initial_step_size = 0.05;
                p.step_decreasing_power = 0.75;
            }
            (SvmsgdType::Asgd, MarginType::HardMargin) => {
                p.margin_regularization = 0.01;
                p.initial_step_size = 0.05;
                p.step_decreasing_power = 0.75;
            }
        }
        p
    }
}

/// Inner product of two equal-length vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// StatModel implementations
// ---------------------------------------------------------------------------

impl StatModel for KNearest {
    const STORAGE_KEY: &'static str = "opencv_ml_knn";

    fn get_var_count(&self) -> i32 {
        self.data.dim() as i32
    }
    fn empty(&self) -> bool {
        self.data.samples.is_empty()
    }
    fn is_classifier(&self) -> bool {
        true
    }
    fn predict_one(&self, sample: &[f64]) -> Result<f64> {
        self.find_nearest(sample, 1)
    }
    fn to_node(&self) -> Node {
        map(vec![
            ("samples", grid(&self.data.samples)),
            ("responses", seq(&self.data.responses)),
        ])
    }
    fn from_node(node: &Node) -> Result<Self> {
        let data = TrainData::create(grid_at(node, "samples")?, seq_at(node, "responses")?)?;
        Ok(KNearest::train(data))
    }
}

impl StatModel for NormalBayes {
    const STORAGE_KEY: &'static str = "opencv_ml_nbayes";

    fn get_var_count(&self) -> i32 {
        self.stats.first().map_or(0, |(m, _, _)| m.len() as i32)
    }
    fn empty(&self) -> bool {
        self.classes.is_empty()
    }
    fn is_classifier(&self) -> bool {
        true
    }
    fn predict_one(&self, sample: &[f64]) -> Result<f64> {
        self.predict(sample)
    }
    fn to_node(&self) -> Node {
        map(vec![
            ("classes", seq(&self.classes)),
            ("means", grid(&self.stats.iter().map(|(m, _, _)| m.clone()).collect::<Vec<_>>())),
            ("vars", grid(&self.stats.iter().map(|(_, v, _)| v.clone()).collect::<Vec<_>>())),
            (
                "log_priors",
                seq(&self.stats.iter().map(|(_, _, p)| *p).collect::<Vec<_>>()),
            ),
        ])
    }
    fn from_node(node: &Node) -> Result<Self> {
        let classes = seq_at(node, "classes")?;
        let means = grid_at(node, "means")?;
        let vars = grid_at(node, "vars")?;
        let priors = seq_at(node, "log_priors")?;
        cv_assert!(
            means.len() == classes.len() && vars.len() == classes.len() && priors.len() == classes.len(),
            "a saved NormalBayes must have one mean, variance and prior per class"
        );
        let stats = means.into_iter().zip(vars).zip(priors).map(|((m, v), p)| (m, v, p)).collect();
        Ok(NormalBayes { classes, stats })
    }
}

impl StatModel for LogisticRegression {
    const STORAGE_KEY: &'static str = "opencv_ml_lr";

    fn get_var_count(&self) -> i32 {
        self.weights.len().saturating_sub(1) as i32
    }
    fn empty(&self) -> bool {
        self.weights.is_empty()
    }
    fn is_classifier(&self) -> bool {
        true
    }
    fn predict_one(&self, sample: &[f64]) -> Result<f64> {
        Ok(self.predict(sample))
    }
    fn to_node(&self) -> Node {
        map(vec![("weights", seq(&self.weights))])
    }
    fn from_node(node: &Node) -> Result<Self> {
        Ok(LogisticRegression { weights: seq_at(node, "weights")? })
    }
}

/// A kernel as a storage node.
fn kernel_node(k: &SvmKernel) -> Node {
    let (name, gamma, coef0, degree) = match *k {
        SvmKernel::Linear => ("linear", 0.0, 0.0, 0.0),
        SvmKernel::Poly { degree, gamma, coef0 } => ("poly", gamma, coef0, degree),
        SvmKernel::Rbf { gamma } => ("rbf", gamma, 0.0, 0.0),
        SvmKernel::Sigmoid { gamma, coef0 } => ("sigmoid", gamma, coef0, 0.0),
        SvmKernel::Chi2 { gamma } => ("chi2", gamma, 0.0, 0.0),
        SvmKernel::Inter => ("inter", 0.0, 0.0, 0.0),
    };
    map(vec![
        ("type", Node::Str(name.to_string())),
        ("gamma", Node::Real(gamma)),
        ("coef0", Node::Real(coef0)),
        ("degree", Node::Real(degree)),
    ])
}

/// The inverse of [`kernel_node`].
fn kernel_from(node: &Node) -> Result<SvmKernel> {
    let (gamma, coef0, degree) =
        (real_at(node, "gamma")?, real_at(node, "coef0")?, real_at(node, "degree")?);
    Ok(match str_at(node, "type")?.as_str() {
        "linear" => SvmKernel::Linear,
        "poly" => SvmKernel::Poly { degree, gamma, coef0 },
        "rbf" => SvmKernel::Rbf { gamma },
        "sigmoid" => SvmKernel::Sigmoid { gamma, coef0 },
        "chi2" => SvmKernel::Chi2 { gamma },
        "inter" => SvmKernel::Inter,
        other => {
            return cv_error!(Code::StsParseError, "unknown SVM kernel `{other}`");
        }
    })
}

impl StatModel for Svm {
    const STORAGE_KEY: &'static str = "opencv_ml_svm";

    fn get_var_count(&self) -> i32 {
        self.machines.first().and_then(|(_, _, m)| m.sv.first()).map_or(0, |v| v.len() as i32)
    }
    fn empty(&self) -> bool {
        self.machines.is_empty()
    }
    fn is_classifier(&self) -> bool {
        true
    }
    fn predict_one(&self, sample: &[f64]) -> Result<f64> {
        Ok(self.predict(sample))
    }
    fn to_node(&self) -> Node {
        let machines = self
            .machines
            .iter()
            .map(|(a, b, m)| {
                map(vec![
                    ("a", Node::Int(*a as i64)),
                    ("b", Node::Int(*b as i64)),
                    ("alpha", seq(&m.alpha)),
                    ("y", seq(&m.y)),
                    ("sv", grid(&m.sv)),
                    ("rho", Node::Real(m.b)),
                    ("kernel", kernel_node(&m.kernel)),
                ])
            })
            .collect();
        map(vec![
            ("labels", seq(&self.labels)),
            ("class_weights", seq(&self.class_weights)),
            ("machines", Node::Seq(machines)),
        ])
    }
    fn from_node(node: &Node) -> Result<Self> {
        let labels = seq_at(node, "labels")?;
        let class_weights = seq_at(node, "class_weights")?;
        let raw = field(node, "machines")?
            .as_seq()
            .ok_or_else(|| cv_core::Error::new(Code::StsParseError, "`machines` is not a list"))?;
        let mut machines = Vec::with_capacity(raw.len());
        for m in raw {
            let a = int_at(m, "a")? as usize;
            let b = int_at(m, "b")? as usize;
            cv_assert!(
                a < labels.len() && b < labels.len(),
                "a saved machine names class {a}/{b} of {}",
                labels.len()
            );
            machines.push((
                a,
                b,
                crate::BinarySvm {
                    alpha: seq_at(m, "alpha")?,
                    y: seq_at(m, "y")?,
                    sv: grid_at(m, "sv")?,
                    b: real_at(m, "rho")?,
                    kernel: kernel_from(field(m, "kernel")?)?,
                },
            ));
        }
        Ok(Svm { labels, machines, class_weights })
    }
}

/// A tree flattened pre-order: a leaf is one record, a split is a record then
/// its two subtrees.
fn tree_nodes(node: &TreeNode, out: &mut Vec<Node>) {
    match node {
        TreeNode::Leaf(v) => out.push(map(vec![
            ("kind", Node::Str("leaf".into())),
            ("value", Node::Real(*v)),
        ])),
        TreeNode::Split { feature, at, left, right } => {
            out.push(map(vec![
                ("kind", Node::Str("split".into())),
                ("feature", Node::Int(*feature as i64)),
                ("at", Node::Real(*at)),
            ]));
            tree_nodes(left, out);
            tree_nodes(right, out);
        }
    }
}

/// Rebuild a tree from the pre-order list, consuming as it goes.
fn tree_from(it: &mut core::slice::Iter<'_, Node>) -> Result<TreeNode> {
    let n = it
        .next()
        .ok_or_else(|| cv_core::Error::new(Code::StsParseError, "the tree ended early"))?;
    match str_at(n, "kind")?.as_str() {
        "leaf" => Ok(TreeNode::Leaf(real_at(n, "value")?)),
        "split" => {
            let feature = int_at(n, "feature")? as usize;
            let at = real_at(n, "at")?;
            let left = Box::new(tree_from(it)?);
            let right = Box::new(tree_from(it)?);
            Ok(TreeNode::Split { feature, at, left, right })
        }
        other => cv_error!(Code::StsParseError, "unknown tree node kind `{other}`"),
    }
}

impl StatModel for DTree {
    const STORAGE_KEY: &'static str = "opencv_ml_dtree";

    fn get_var_count(&self) -> i32 {
        max_feature(&self.root).map_or(0, |f| f as i32 + 1)
    }
    fn empty(&self) -> bool {
        matches!(self.root, TreeNode::Leaf(_)) && self.depth == 0
    }
    fn is_classifier(&self) -> bool {
        true
    }
    fn predict_one(&self, sample: &[f64]) -> Result<f64> {
        Ok(self.predict(sample))
    }
    fn to_node(&self) -> Node {
        let mut nodes = Vec::new();
        tree_nodes(&self.root, &mut nodes);
        map(vec![("depth", Node::Int(self.depth as i64)), ("nodes", Node::Seq(nodes))])
    }
    fn from_node(node: &Node) -> Result<Self> {
        let nodes = field(node, "nodes")?
            .as_seq()
            .ok_or_else(|| cv_core::Error::new(Code::StsParseError, "`nodes` is not a list"))?;
        let mut it = nodes.iter();
        let root = tree_from(&mut it)?;
        cv_assert!(it.next().is_none(), "the saved tree has trailing nodes");
        Ok(DTree { root, depth: int_at(node, "depth")?.max(0) as usize })
    }
}

/// The largest feature index any split tests.
fn max_feature(node: &TreeNode) -> Option<usize> {
    match node {
        TreeNode::Leaf(_) => None,
        TreeNode::Split { feature, left, right, .. } => Some(
            (*feature).max(max_feature(left).unwrap_or(0)).max(max_feature(right).unwrap_or(0)),
        ),
    }
}

impl StatModel for RTrees {
    const STORAGE_KEY: &'static str = "opencv_ml_rtrees";

    fn get_var_count(&self) -> i32 {
        self.trees.iter().map(StatModel::get_var_count).max().unwrap_or(0)
    }
    fn empty(&self) -> bool {
        self.trees.is_empty()
    }
    fn is_classifier(&self) -> bool {
        true
    }
    fn predict_one(&self, sample: &[f64]) -> Result<f64> {
        Ok(self.predict(sample))
    }
    fn to_node(&self) -> Node {
        map(vec![("trees", Node::Seq(self.trees.iter().map(StatModel::to_node).collect()))])
    }
    fn from_node(node: &Node) -> Result<Self> {
        let raw = field(node, "trees")?
            .as_seq()
            .ok_or_else(|| cv_core::Error::new(Code::StsParseError, "`trees` is not a list"))?;
        let trees = raw.iter().map(DTree::from_node).collect::<Result<Vec<_>>>()?;
        Ok(RTrees { trees })
    }
}

impl StatModel for AnnMlp {
    const STORAGE_KEY: &'static str = "opencv_ml_ann_mlp";

    fn get_var_count(&self) -> i32 {
        self.w1.first().map_or(0, |r| r.len().saturating_sub(1) as i32)
    }
    fn empty(&self) -> bool {
        self.w1.is_empty() || self.w2.is_empty()
    }
    /// A network with one output unit is used here as a regressor; with more,
    /// the argmax is a class.
    fn is_classifier(&self) -> bool {
        self.w2.len() > 1
    }
    fn predict_one(&self, sample: &[f64]) -> Result<f64> {
        let out = self.predict(sample);
        cv_assert!(!out.is_empty(), "the network produced no output");
        Ok(if self.is_classifier() { self.predict_class(sample) as f64 } else { out[0] })
    }
    fn to_node(&self) -> Node {
        map(vec![("w1", grid(&self.w1)), ("w2", grid(&self.w2))])
    }
    fn from_node(node: &Node) -> Result<Self> {
        Ok(AnnMlp { w1: grid_at(node, "w1")?, w2: grid_at(node, "w2")? })
    }
}

impl StatModel for Em {
    const STORAGE_KEY: &'static str = "opencv_ml_em";

    fn get_var_count(&self) -> i32 {
        self.means.first().map_or(0, |m| m.len() as i32)
    }
    fn empty(&self) -> bool {
        self.weights.is_empty()
    }
    /// EM assigns a component index, which is a label, not a quantity.
    fn is_classifier(&self) -> bool {
        true
    }
    fn predict_one(&self, sample: &[f64]) -> Result<f64> {
        Ok(self.predict(sample) as f64)
    }
    fn to_node(&self) -> Node {
        map(vec![
            ("weights", seq(&self.weights)),
            ("means", grid(&self.means)),
            ("variances", grid(&self.variances)),
        ])
    }
    fn from_node(node: &Node) -> Result<Self> {
        Ok(Em {
            weights: seq_at(node, "weights")?,
            means: grid_at(node, "means")?,
            variances: grid_at(node, "variances")?,
        })
    }
}

impl StatModel for Svmsgd {
    const STORAGE_KEY: &'static str = "opencv_ml_svmsgd";

    fn get_var_count(&self) -> i32 {
        self.weights.len() as i32
    }
    fn empty(&self) -> bool {
        self.weights.is_empty()
    }
    fn is_classifier(&self) -> bool {
        true
    }
    fn predict_one(&self, sample: &[f64]) -> Result<f64> {
        cv_assert!(
            sample.len() == self.weights.len(),
            "sample has {} features, the model has {}",
            sample.len(),
            self.weights.len()
        );
        Ok(if dot(&self.weights, sample) + self.shift >= 0.0 {
            self.labels[1]
        } else {
            self.labels[0]
        })
    }
    fn to_node(&self) -> Node {
        map(vec![
            ("weights", seq(&self.weights)),
            ("shift", Node::Real(self.shift)),
            ("labels", seq(&self.labels)),
        ])
    }
    fn from_node(node: &Node) -> Result<Self> {
        let labels = seq_at(node, "labels")?;
        cv_assert!(labels.len() == 2, "a saved SVMSGD must name two classes");
        Ok(Svmsgd { weights: seq_at(node, "weights")?, shift: real_at(node, "shift")?, labels })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{MarginType, SvmsgdParams, SvmsgdType};
    use cv_core::rng::Rng;

    /// Two linearly separable clusters, 20 rows each.
    fn blobs() -> TrainData {
        let mut samples = Vec::new();
        let mut responses = Vec::new();
        let mut rng = Rng::new(7);
        for _ in 0..20 {
            let mut j = || (rng.next_u32() % 100) as f64 / 100.0;
            samples.push(vec![j(), j()]);
            responses.push(0.0);
        }
        for _ in 0..20 {
            let mut j = || (rng.next_u32() % 100) as f64 / 100.0;
            samples.push(vec![5.0 + j(), 5.0 + j()]);
            responses.push(1.0);
        }
        TrainData::create(samples, responses).unwrap()
    }

    #[test]
    fn train_data_reports_its_shape() {
        let d = blobs();
        assert_eq!(d.get_n_samples(), 40);
        assert_eq!(d.get_n_all_vars(), 2);
        assert_eq!(d.get_n_vars(), 2, "no varIdx means every variable");
        assert_eq!(d.get_layout(), SampleTypes::RowSample);
        assert!(d.get_var_idx().is_empty());
        assert!(d.get_names().is_empty());
        assert_eq!(d.get_class_labels(), vec![0.0, 1.0]);
        assert_eq!(d.get_response_type(), VariableTypes::Categorical);
        assert_eq!(d.get_var_type().len(), 3, "one per variable plus the response");
        assert_eq!(d.get_var_symbol_flags(), vec![0, 0]);
        assert_eq!(d.get_sample_weights(), vec![1.0; 40], "uniform by default");
        assert_eq!(d.get_missing().len(), 40);
        assert!(d.get_missing().iter().all(|r| r.len() == 2 && !r[0] && !r[1]));
        assert_eq!(d.get_samples().len(), 40);
        assert_eq!(d.get_responses().len(), 40);

        // varIdx restricts what `getSample` hands back.
        let mut r = d.clone();
        r.var_idx = vec![1];
        assert_eq!(r.get_n_vars(), 1);
        assert_eq!(r.get_sample(0).unwrap(), vec![d.samples[0][1]]);
        assert_eq!(d.get_sample(0).unwrap(), d.samples[0]);
        assert!(d.get_sample(40).is_err());

        assert_eq!(d.get_values(0, &[0, 1]).unwrap(), vec![d.samples[0][0], d.samples[1][0]]);
        assert!(d.get_values(9, &[0]).is_err(), "no such variable");
    }

    #[test]
    fn normalised_responses_index_the_label_list() {
        let d = TrainData::create(
            vec![vec![0.0], vec![1.0], vec![2.0]],
            vec![10.0, 40.0, 10.0],
        )
        .unwrap();
        assert_eq!(d.get_class_labels(), vec![10.0, 40.0]);
        // The labels are 10 and 40; the normalised form is 0 and 1.
        assert_eq!(d.get_norm_cat_responses(), vec![0, 1, 0]);
    }

    #[test]
    fn categorical_variables_report_their_map_and_offsets() {
        let mut d = TrainData::create(
            vec![vec![1.0, 7.0], vec![2.0, 7.0], vec![1.0, 9.0]],
            vec![0.0, 1.0, 0.0],
        )
        .unwrap();
        // Variable 0 categorical, variable 1 ordered.
        d.var_type = vec![
            VariableTypes::Categorical,
            VariableTypes::Numerical,
            VariableTypes::Categorical,
        ];
        assert_eq!(d.get_cat_count(0).unwrap(), 2);
        assert_eq!(d.get_cat_count(1).unwrap(), 2);
        assert!(d.get_cat_count(5).is_err());
        assert_eq!(d.get_cat_map(), vec![1.0, 2.0], "only the categorical variable");
        assert_eq!(d.get_cat_ofs(), vec![(0, 2), (0, 0)]);
    }

    #[test]
    fn substitute_values_ignore_the_rows_marked_missing() {
        let mut d =
            TrainData::create(vec![vec![1.0], vec![3.0], vec![100.0]], vec![0.0, 1.0, 0.0])
                .unwrap();
        assert_eq!(d.get_default_subst_values(), vec![(1.0 + 3.0 + 100.0) / 3.0]);
        d.missing = vec![vec![false], vec![false], vec![true]];
        assert_eq!(d.get_default_subst_values(), vec![2.0], "the outlier is missing");
    }

    #[test]
    fn a_split_divides_the_rows_and_survives_a_reshuffle() {
        let mut d = blobs();
        // Unsplit, both halves are everything — upstream's behaviour, and the
        // reason a test error means nothing until a split is set.
        assert_eq!(d.get_n_train_samples(), 40);
        assert_eq!(d.get_n_test_samples(), 40);

        let mut rng = Rng::new(3);
        d.set_train_test_split(30, Some(&mut rng)).unwrap();
        assert_eq!(d.get_n_train_samples(), 30);
        assert_eq!(d.get_n_test_samples(), 10);
        assert_eq!(d.get_train_samples().len(), 30);
        assert_eq!(d.get_test_samples().len(), 10);
        assert_eq!(d.get_train_responses().len(), 30);
        assert_eq!(d.get_test_responses().len(), 10);
        assert_eq!(d.get_train_sample_weights().len(), 30);
        assert_eq!(d.get_test_sample_weights().len(), 10);
        assert_eq!(d.get_train_norm_cat_responses().len(), 30);
        assert_eq!(d.get_test_norm_cat_responses().len(), 10);

        // No row is on both sides, and every row is on one.
        let mut all = d.get_train_sample_idx();
        all.extend(d.get_test_sample_idx());
        all.sort_unstable();
        assert_eq!(all, (0..40).collect::<Vec<i32>>());

        let before = d.get_train_sample_idx();
        d.shuffle_train_test(&mut rng);
        assert_eq!(d.get_n_train_samples(), 30, "the sizes are kept");
        assert_ne!(d.get_train_sample_idx(), before, "and the membership changes");

        d.set_train_test_split_ratio(0.5, None).unwrap();
        assert_eq!(d.get_n_train_samples(), 20);
        assert_eq!(d.get_train_sample_idx(), (0..20).collect::<Vec<i32>>(), "unshuffled");
        assert!(d.set_train_test_split(41, None).is_err());
        assert!(d.set_train_test_split_ratio(1.5, None).is_err());
    }

    #[test]
    fn sub_matrix_and_sub_vector_take_the_named_rows() {
        let m = vec![vec![0.0], vec![1.0], vec![2.0]];
        assert_eq!(TrainData::get_sub_matrix(&m, &[2, 0]), vec![vec![2.0], vec![0.0]]);
        assert_eq!(TrainData::get_sub_vector(&[9.0, 8.0, 7.0], &[1, 2]), vec![8.0, 7.0]);
        // An index past the end is dropped, not a panic.
        assert!(TrainData::get_sub_matrix(&m, &[7]).is_empty());
    }

    #[test]
    fn the_svm_kernel_accessors_agree_with_the_kernel_they_describe() {
        let mut p = SvmParams::default();
        assert_eq!(p.get_c(), 1.0);
        assert_eq!(p.get_type(), SvmType::CSvc);
        assert_eq!(p.get_kernel_type(), crate::SvmKernelKind::Rbf, "upstream's default");
        assert_eq!(p.get_gamma(), 1.0);
        assert_eq!(p.get_coef0(), 0.0, "an RBF kernel has no coef0");
        assert_eq!(p.get_degree(), 0.0);

        p.set_gamma(0.25);
        assert_eq!(p.get_gamma(), 0.25);
        // Switching family carries gamma across rather than dropping it.
        p.set_kernel(crate::SvmKernelKind::Poly);
        assert_eq!(p.get_kernel_type(), crate::SvmKernelKind::Poly);
        assert_eq!(p.get_gamma(), 0.25);
        p.set_coef0(-2.0).set_degree(3.0);
        assert_eq!(p.get_coef0(), -2.0);
        assert_eq!(p.get_degree(), 3.0);
        assert_eq!(p.kernel, SvmKernel::Poly { degree: 3.0, gamma: 0.25, coef0: -2.0 });

        // A linear kernel has none of the three, and setting them is a no-op
        // rather than a silent change of family.
        p.set_kernel(crate::SvmKernelKind::Linear);
        p.set_gamma(9.0).set_coef0(9.0).set_degree(9.0);
        assert_eq!((p.get_gamma(), p.get_coef0(), p.get_degree()), (0.0, 0.0, 0.0));
        assert_eq!(p.kernel, SvmKernel::Linear);

        p.set_nu(0.3).set_p(0.05);
        assert_eq!((p.get_nu(), p.get_p()), (0.3, 0.05));
        p.set_type(SvmType::NuSvc);
        assert_eq!(p.get_type(), SvmType::NuSvc);
        // ...and a formulation the SMO here does not solve is refused.
        assert!(Svm::train_with(&blobs(), p).is_err());

        let mut q = SvmParams::default();
        q.set_term_criteria(TermCriteria { ty: 3, max_count: 42, epsilon: 1e-6 });
        assert_eq!(q.max_iter, 42);
        assert_eq!(q.tolerance, 1e-6);
        assert_eq!(q.get_term_criteria().max_count, 42);
    }

    #[test]
    fn the_default_grids_are_reachable_by_parameter_id() {
        for id in [
            SvmParamType::C,
            SvmParamType::Gamma,
            SvmParamType::Degree,
            SvmParamType::Coef,
            SvmParamType::P,
            SvmParamType::Nu,
        ] {
            let g = ParamGrid::get_default_grid_ptr(id);
            assert!(!g.values().unwrap().is_empty(), "{id:?} produced no values");
        }
        assert_eq!(ParamGrid::get_default_grid_ptr(SvmParamType::C), ParamGrid::default_c());
    }

    #[test]
    fn the_svm_exposes_its_decision_function_and_support_vectors() {
        let d = blobs();
        let m = Svm::train_with(&d, SvmParams { kernel: SvmKernel::Linear, ..Default::default() })
            .unwrap();
        assert_eq!(m.machine_count(), 1, "two classes make one machine");
        let (rho, alpha, svidx) = m.get_decision_function(0).unwrap();
        assert_eq!(alpha.len(), svidx.len());
        assert!(!alpha.is_empty(), "a fitted machine has support vectors");
        assert!(rho.is_finite());
        let un = m.get_uncompressed_support_vectors();
        assert_eq!(un.len(), alpha.len());
        // Every index the decision function names really is in that list.
        assert!(svidx.iter().all(|&i| (i as usize) < un.len()));
        assert!(m.get_decision_function(1).is_err());

        // The decision function reproduces the model's own margin.
        let x = &d.samples[0];
        let got: f64 =
            alpha.iter().zip(&un).map(|(a, sv)| a * SvmKernel::Linear.eval(sv, x)).sum::<f64>()
                - rho;
        assert!((got - m.decision(x)).abs() < 1e-9, "{got} vs {}", m.decision(x));

        let mut m2 = m.clone();
        assert!(m2.get_class_weights().is_empty());
        m2.set_class_weights(vec![1.0, 2.0]).unwrap();
        assert_eq!(m2.get_class_weights(), &[1.0, 2.0]);
        assert!(m2.set_class_weights(vec![1.0]).is_err(), "one weight for two classes");
    }

    #[test]
    fn em_reports_its_components_and_can_start_from_either_step() {
        let samples: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![f64::from(i) * 0.1])
            .chain((0..10).map(|i| vec![10.0 + f64::from(i) * 0.1]))
            .collect();

        // trainM: hard responsibilities, so the M step alone should recover
        // two well-separated components.
        let probs: Vec<Vec<f64>> = (0..20)
            .map(|i| if i < 10 { vec![1.0, 0.0] } else { vec![0.0, 1.0] })
            .collect();
        let m = Em::train_m(&samples, &probs, 1).unwrap();
        assert_eq!(m.get_weights().len(), 2);
        assert!((m.get_weights().iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let mut means: Vec<f64> = m.get_means().iter().map(|v| v[0]).collect();
        means.sort_by(f64::total_cmp);
        assert!(means[0] < 1.0 && means[1] > 9.0, "{means:?}");

        // The covariances are diagonal, and the diagonal is the variance.
        let covs = m.get_covs();
        assert_eq!(covs.len(), 2);
        assert_eq!(covs[0].len(), 1);
        assert!((covs[0][0][0] - m.variances[0][0]).abs() < 1e-12);

        // trainE: the same answer starting from those parameters.
        let e = Em::train_e(
            &samples,
            m.get_means(),
            &m.variances,
            m.get_weights(),
            3,
        )
        .unwrap();
        let mut emeans: Vec<f64> = e.get_means().iter().map(|v| v[0]).collect();
        emeans.sort_by(f64::total_cmp);
        assert!((emeans[0] - means[0]).abs() < 1e-6, "{emeans:?} vs {means:?}");

        // predict2 gives a log-likelihood and the winning component, and the
        // component agrees with `predict`.
        let (ll, comp) = m.predict2(&samples[0]);
        assert!(ll.is_finite() && ll < 0.0, "a density log-likelihood is negative here");
        assert_eq!(comp as usize, m.predict(&samples[0]));

        assert!(Em::train_e(&samples, &[], &[], &[], 1).is_err());
        assert!(Em::train_m(&samples, &probs[..3], 1).is_err());
    }

    #[test]
    fn logistic_regression_reports_its_thetas_bias_first() {
        let d = blobs();
        let m = LogisticRegression::train(&d, 0.05, 200).unwrap();
        let t = m.get_learnt_thetas();
        assert_eq!(t.len(), 1, "the binary form has one classifier");
        assert_eq!(t[0].len(), m.weights.len());
        assert_eq!(t[0][0], *m.weights.last().unwrap(), "the bias comes first");
        assert_eq!(t[0][1], m.weights[0]);
    }

    #[test]
    fn naive_bayes_posteriors_are_normalised_and_pick_the_predicted_class() {
        let d = blobs();
        let m = NormalBayes::train(&d).unwrap();
        for s in [&d.samples[0], &d.samples[39]] {
            let (label, probs) = m.predict_prob(s).unwrap();
            assert_eq!(probs.len(), 2);
            assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert_eq!(label, m.predict(s).unwrap(), "the label must match `predict`");
            let best = probs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            assert_eq!(label, d.classes()[best]);
        }
        assert!(m.predict_prob(&[1.0, 2.0, 3.0]).is_err(), "wrong feature count");
    }

    #[test]
    fn a_forest_reports_votes_and_variable_usage() {
        let d = blobs();
        let mut rng = Rng::new(5);
        let f = RTrees::train(&d, 12, 6, &mut rng).unwrap();
        let labels = d.classes();
        let votes = f.get_votes(&d.samples[0], &labels);
        assert_eq!(votes.len(), 2);
        assert_eq!(votes.iter().sum::<i32>(), f.len() as i32, "every tree votes once");
        assert!(votes[0] > votes[1], "a class-0 sample should mostly draw class-0 votes");

        let imp = f.get_var_importance(2);
        assert_eq!(imp.len(), 2);
        assert!((imp.iter().sum::<f64>() - 1.0).abs() < 1e-9, "importance is a share");

        // Out-of-bag error is refused rather than approximated.
        assert!(f.get_oob_error().is_err());
    }

    #[test]
    fn svmsgd_separates_two_blobs_and_exposes_its_hyperplane() {
        let d = blobs();
        let mut rng = Rng::new(1);
        let p = Svmsgd::set_optimal_parameters(SvmsgdType::Asgd, MarginType::SoftMargin);
        assert_eq!(p.svmsgd_type, SvmsgdType::Asgd);
        assert_eq!(p.margin_type, MarginType::SoftMargin);
        let m = Svmsgd::train(&d, &p, 30, &mut rng).unwrap();
        assert_eq!(m.get_weights().len(), 2);
        assert!(m.get_shift().is_finite());
        for (s, want) in d.samples.iter().zip(&d.responses) {
            assert_eq!(m.predict_one(s).unwrap(), *want, "misclassified {s:?}");
        }
        assert!(m.predict_one(&[1.0]).is_err(), "wrong feature count");

        // Both variants and both margin types produce usable settings.
        for t in [SvmsgdType::Sgd, SvmsgdType::Asgd] {
            for mt in [MarginType::SoftMargin, MarginType::HardMargin] {
                let p = Svmsgd::set_optimal_parameters(t, mt);
                assert!(p.initial_step_size > 0.0 && p.margin_regularization > 0.0);
            }
        }

        // A three-class set is refused rather than silently binarised.
        let three = TrainData::create(
            vec![vec![0.0], vec![1.0], vec![2.0]],
            vec![0.0, 1.0, 2.0],
        )
        .unwrap();
        assert!(Svmsgd::train(&three, &SvmsgdParams::default(), 1, &mut rng).is_err());
    }

    #[test]
    fn ann_mlp_hands_back_each_weight_layer() {
        let mut rng = Rng::new(2);
        let n = AnnMlp::new(2, 3, 2, &mut rng).unwrap();
        assert_eq!(n.get_weights(0).unwrap().len(), 3, "one row per hidden unit");
        assert_eq!(n.get_weights(0).unwrap()[0].len(), 3, "inputs plus a bias");
        assert_eq!(n.get_weights(1).unwrap().len(), 2, "one row per output");
        assert!(n.get_weights(2).is_err());
        assert_eq!(n.get_var_count(), 2);
        assert!(n.is_classifier(), "two outputs make it a classifier");
    }

    /// A path in the session scratch area, unique per test.
    fn tmp(name: &str) -> String {
        let dir = std::env::temp_dir();
        dir.join(format!("cv_ml_{name}.json")).to_string_lossy().into_owned()
    }

    #[test]
    fn every_model_round_trips_through_a_file() {
        let d = blobs();
        let mut rng = Rng::new(4);
        let probe = d.samples[0].clone();
        let probe2 = d.samples[39].clone();

        macro_rules! round_trip {
            ($name:literal, $ty:ty, $model:expr) => {{
                let m: $ty = $model;
                let path = tmp($name);
                m.save(&path).unwrap();
                let back = <$ty as StatModel>::load(&path).unwrap();
                assert_eq!(
                    m.predict_one(&probe).unwrap(),
                    back.predict_one(&probe).unwrap(),
                    concat!($name, " changed its answer across a save/load")
                );
                assert_eq!(m.predict_one(&probe2).unwrap(), back.predict_one(&probe2).unwrap());
                assert_eq!(m.get_var_count(), back.get_var_count());
                assert!(m.is_trained() && !m.empty());
                let _ = std::fs::remove_file(&path);
                m
            }};
        }

        let knn = round_trip!("knn", KNearest, KNearest::train(d.clone()));
        assert_eq!(knn.get_var_count(), 2);
        round_trip!("nbayes", NormalBayes, NormalBayes::train(&d).unwrap());
        round_trip!("lr", LogisticRegression, LogisticRegression::train(&d, 0.05, 200).unwrap());
        round_trip!(
            "svm",
            Svm,
            Svm::train_with(&d, SvmParams { kernel: SvmKernel::Linear, ..Default::default() })
                .unwrap()
        );
        round_trip!("dtree", DTree, DTree::train(&d, 5, 2).unwrap());
        round_trip!("rtrees", RTrees, RTrees::train(&d, 8, 5, &mut rng).unwrap());
        round_trip!("em", Em, Em::train(&d.samples, 2, 20, &mut rng).unwrap());
        round_trip!(
            "svmsgd",
            Svmsgd,
            Svmsgd::train(&d, &SvmsgdParams::default(), 20, &mut rng).unwrap()
        );
    }

    #[test]
    fn loading_the_wrong_kind_of_model_is_an_error() {
        let d = blobs();
        let path = tmp("mismatch");
        DTree::train(&d, 5, 2).unwrap().save(&path).unwrap();
        // The node name says `dtree`; asking for an SVM must fail rather than
        // produce an empty model.
        assert!(<Svm as StatModel>::load(&path).is_err());
        assert!(<DTree as StatModel>::load(&path).is_ok());
        let _ = std::fs::remove_file(&path);
        assert!(<DTree as StatModel>::load(&tmp("does-not-exist")).is_err());
    }

    #[test]
    fn calc_error_counts_misclassifications_as_a_percentage() {
        let mut d = blobs();
        let m = Svm::train_with(&d, SvmParams { kernel: SvmKernel::Linear, ..Default::default() })
            .unwrap();
        // The blobs are separable, so a linear SVM should get them all.
        assert!(m.calc_error(&d, false).unwrap() < 1e-9);

        // With a split, the held-out rows are scored on their own.
        let mut rng = Rng::new(8);
        d.set_train_test_split(30, Some(&mut rng)).unwrap();
        assert!(m.calc_error(&d, true).unwrap() < 1e-9);

        // A model that always answers 0 gets exactly the class-1 share wrong.
        let half = LogisticRegression { weights: vec![0.0, 0.0, -1.0] };
        let err = half.calc_error(&blobs(), false).unwrap();
        assert!((err - 50.0).abs() < 1e-9, "expected 50%, got {err}");
    }
}
