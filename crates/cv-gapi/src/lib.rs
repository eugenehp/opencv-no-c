//! `cv::gapi` for `opencv-no-c` — a deferred image-processing graph.
//!
//! # The idea
//!
//! Ordinary `imgproc` calls run immediately and allocate an intermediate for
//! every step. G-API instead *records* the operations, then compiles the
//! recording once and runs it on many frames. That is worth having for two
//! reasons: the graph can be inspected and optimised before it runs, and the
//! per-frame cost loses the repeated setup.
//!
//! ```no_run
//! use cv_gapi::{GComputation, GMat};
//! # fn main() -> cv_core::Result<()> {
//! let mut g = GComputation::new();
//! let inp = g.input();
//! let gray = g.cvt_color_bgr2gray(inp);
//! let blur = g.gaussian_blur(gray, 5, 1.5);
//! let out = g.canny(blur, 50.0, 150.0);
//! g.set_output(out);
//!
//! let compiled = g.compile()?;      // once
//! # let frame: cv_core::Mat = unimplemented!();
//! let result = compiled.apply(&frame)?;  // per frame
//! # Ok(()) }
//! ```
//!
//! # Relationship to RLX
//!
//! Upstream's G-API has pluggable *backends* (`fluid`, `ocl`, `ie`, `onnx`).
//! This port has two: a portable one built on `cv-imgproc`, and an RLX one for
//! the kernels that map onto tensor ops. [`GComputation::compile_on`] picks.
//! The graph itself is a plain Rust structure, so nothing about it depends on
//! RLX being present.
//!
//! # What is deliberately not modelled
//!
//! Upstream's `GKernelPackage`/`GBackend` machinery and its `cv::gapi::wip`
//! streaming sources. Those are extension points for third-party kernels; with
//! a closed operation set they would be ceremony with nothing behind it.

#![warn(missing_docs)]


pub mod backends;
pub mod constants;
pub mod graph;
pub mod kernels;
pub mod types;

use cv_core::types::Depth;
use cv_core::{cv_assert, Code, Error, Mat, MatLike, Result, Scalar};

/// A handle to a matrix *inside* a graph — `cv::GMat`.
///
/// It holds no data. Values only exist while [`Compiled::apply`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GMat(usize);

/// A handle to a scalar inside a graph — `cv::GScalar`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GScalar(usize);

/// Which implementation a compiled graph uses for its kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// `cv-imgproc` — always available.
    #[default]
    Portable,
    /// RLX, falling back to portable for any kernel it cannot take.
    Rlx,
}

/// One recorded operation.
#[derive(Debug, Clone, PartialEq)]
enum Op {
    Input,
    CvtColorBgr2Gray(GMat),
    CvtColorGray2Bgr(GMat),
    GaussianBlur(GMat, i32, f64),
    Blur(GMat, i32),
    MedianBlur(GMat, i32),
    Sobel(GMat, Depth, i32, i32, i32),
    Laplacian(GMat, Depth, i32),
    Canny(GMat, f64, f64),
    Threshold(GMat, f64, f64),
    Resize(GMat, i32, i32),
    Erode(GMat, i32),
    Dilate(GMat, i32),
    Add(GMat, GMat),
    Sub(GMat, GMat),
    Mul(GMat, GMat),
    AddC(GMat, Scalar),
    MulC(GMat, f64),
    AbsDiff(GMat, GMat),
    BitwiseAnd(GMat, GMat),
    BitwiseOr(GMat, GMat),
    BitwiseNot(GMat),
    Filter2D(GMat, Vec<f64>, i32, i32),
    Split(GMat, usize),
    Merge(Vec<GMat>),
    Flip(GMat, i32),
    EqualizeHist(GMat),
    /// A kernel recorded by name, from [`crate::kernels`].
    ///
    /// The graph holds the node — its id, its operands and its position — so
    /// the structure can be inspected and serialised. What it does *not* hold
    /// is a CPU binding, so a graph containing one is refused at compile time
    /// rather than evaluated as something else.
    Record {
        /// The upstream operation id.
        id: String,
        /// The nodes it consumes, in order.
        operands: Vec<GMat>,
    },
}

/// `cv::GComputation` — records operations, then compiles them.
#[derive(Debug, Default, Clone)]
pub struct GComputation {
    ops: Vec<Op>,
    inputs: Vec<GMat>,
    output: Option<GMat>,
}

impl GComputation {
    /// An empty graph.
    #[must_use]
    pub fn new() -> Self {
        GComputation::default()
    }

    fn push(&mut self, op: Op) -> GMat {
        self.ops.push(op);
        GMat(self.ops.len() - 1)
    }

    /// Declare a graph input — `cv::GMat()` in upstream's protocol.
    pub fn input(&mut self) -> GMat {
        let m = self.push(Op::Input);
        self.inputs.push(m);
        m
    }

    /// Mark a node as the graph's result.
    pub fn set_output(&mut self, m: GMat) {
        self.output = Some(m);
    }

    /// Number of recorded operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Record a kernel by name, for [`crate::kernels`].
    ///
    /// The node joins the graph like any other; only its evaluation is
    /// missing. That split is deliberate — the graph's *shape* is what most of
    /// G-API's surface operates on, and it is fully usable here.
    pub fn record(&mut self, id: &str, operands: &[GMat]) -> GMat {
        self.push(Op::Record { id: id.to_string(), operands: operands.to_vec() })
    }

    /// Record a kernel whose result is a scalar rather than a matrix.
    pub fn record_scalar(&mut self, id: &str, src: GMat) -> GScalar {
        let m = self.record(id, &[src]);
        GScalar(m.0)
    }

    /// Every recorded kernel id, in graph order.
    ///
    /// Only the named nodes appear; the built-in operations are identified by
    /// their variant rather than by a string.
    #[must_use]
    pub fn node_ids(&self) -> Vec<String> {
        self.ops
            .iter()
            .filter_map(|o| match o {
                Op::Record { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect()
    }

    /// Whether nothing has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// `cv::gapi::BGR2Gray`.
    pub fn cvt_color_bgr2gray(&mut self, src: GMat) -> GMat {
        self.push(Op::CvtColorBgr2Gray(src))
    }
    /// `cv::gapi::Gray2BGR`.
    pub fn cvt_color_gray2bgr(&mut self, src: GMat) -> GMat {
        self.push(Op::CvtColorGray2Bgr(src))
    }
    /// `cv::gapi::gaussianBlur`.
    pub fn gaussian_blur(&mut self, src: GMat, ksize: i32, sigma: f64) -> GMat {
        self.push(Op::GaussianBlur(src, ksize, sigma))
    }
    /// `cv::gapi::blur` — the normalised box filter.
    pub fn blur(&mut self, src: GMat, ksize: i32) -> GMat {
        self.push(Op::Blur(src, ksize))
    }
    /// `cv::gapi::medianBlur`.
    pub fn median_blur(&mut self, src: GMat, ksize: i32) -> GMat {
        self.push(Op::MedianBlur(src, ksize))
    }
    /// `cv::gapi::Sobel`.
    /// `ddepth` is upstream's, and it is not decoration: `CV_8U` saturates a
    /// negative gradient to zero, which is a different picture from the signed
    /// one. Hard-coding `CV_32F` here made every graph that asked for `CV_8U`
    /// disagree with cv2 on 40% of its pixels.
    pub fn sobel(&mut self, src: GMat, ddepth: Depth, dx: i32, dy: i32, ksize: i32) -> GMat {
        self.push(Op::Sobel(src, ddepth, dx, dy, ksize))
    }
    /// `cv::gapi::Laplacian`.
    pub fn laplacian(&mut self, src: GMat, ddepth: Depth, ksize: i32) -> GMat {
        self.push(Op::Laplacian(src, ddepth, ksize))
    }
    /// `cv::gapi::Canny`.
    pub fn canny(&mut self, src: GMat, low: f64, high: f64) -> GMat {
        self.push(Op::Canny(src, low, high))
    }
    /// `cv::gapi::threshold`, binary.
    pub fn threshold(&mut self, src: GMat, thresh: f64, maxval: f64) -> GMat {
        self.push(Op::Threshold(src, thresh, maxval))
    }
    /// `cv::gapi::resize`.
    pub fn resize(&mut self, src: GMat, w: i32, h: i32) -> GMat {
        self.push(Op::Resize(src, w, h))
    }
    /// `cv::gapi::erode`, with a rectangular element.
    pub fn erode(&mut self, src: GMat, ksize: i32) -> GMat {
        self.push(Op::Erode(src, ksize))
    }
    /// `cv::gapi::dilate`, with a rectangular element.
    pub fn dilate(&mut self, src: GMat, ksize: i32) -> GMat {
        self.push(Op::Dilate(src, ksize))
    }
    /// `cv::gapi::add`.
    pub fn add(&mut self, a: GMat, b: GMat) -> GMat {
        self.push(Op::Add(a, b))
    }
    /// `cv::gapi::sub`.
    pub fn sub(&mut self, a: GMat, b: GMat) -> GMat {
        self.push(Op::Sub(a, b))
    }
    /// `cv::gapi::mul`.
    pub fn mul(&mut self, a: GMat, b: GMat) -> GMat {
        self.push(Op::Mul(a, b))
    }
    /// `cv::gapi::addC`.
    pub fn add_c(&mut self, a: GMat, s: Scalar) -> GMat {
        self.push(Op::AddC(a, s))
    }
    /// `cv::gapi::mulC`.
    pub fn mul_c(&mut self, a: GMat, s: f64) -> GMat {
        self.push(Op::MulC(a, s))
    }
    /// `cv::gapi::absDiff`.
    pub fn abs_diff(&mut self, a: GMat, b: GMat) -> GMat {
        self.push(Op::AbsDiff(a, b))
    }
    /// `cv::gapi::bitwise_and`.
    pub fn bitwise_and(&mut self, a: GMat, b: GMat) -> GMat {
        self.push(Op::BitwiseAnd(a, b))
    }
    /// `cv::gapi::bitwise_or`.
    pub fn bitwise_or(&mut self, a: GMat, b: GMat) -> GMat {
        self.push(Op::BitwiseOr(a, b))
    }
    /// `cv::gapi::bitwise_not`.
    pub fn bitwise_not(&mut self, a: GMat) -> GMat {
        self.push(Op::BitwiseNot(a))
    }
    /// `cv::gapi::filter2D`.
    pub fn filter_2d(&mut self, src: GMat, kernel: &[f64], kw: i32, kh: i32) -> GMat {
        self.push(Op::Filter2D(src, kernel.to_vec(), kw, kh))
    }
    /// `cv::gapi::split3` / `split4` — one channel, selected by index.
    pub fn split(&mut self, src: GMat, channel: usize) -> GMat {
        self.push(Op::Split(src, channel))
    }
    /// `cv::gapi::merge3` / `merge4`.
    pub fn merge(&mut self, parts: &[GMat]) -> GMat {
        self.push(Op::Merge(parts.to_vec()))
    }
    /// `cv::gapi::flip`.
    pub fn flip(&mut self, src: GMat, code: i32) -> GMat {
        self.push(Op::Flip(src, code))
    }
    /// `cv::gapi::equalizeHist`.
    pub fn equalize_hist(&mut self, src: GMat) -> GMat {
        self.push(Op::EqualizeHist(src))
    }

    /// `cv::GComputation::compile()` on the portable backend.
    pub fn compile(&self) -> Result<Compiled> {
        self.compile_on(Backend::Portable)
    }

    /// `cv::GComputation::compile(args)` — choose the kernel backend.
    ///
    /// Compilation validates the graph up front: every operand must refer to an
    /// earlier node, and an output must be set. Catching a malformed graph here
    /// rather than mid-frame is the point of having a compile step at all.
    pub fn compile_on(&self, backend: Backend) -> Result<Compiled> {
        let out = self
            .output
            .ok_or_else(|| Error::new(Code::StsBadArg, "no output set on the graph".to_string()))?;
        cv_assert!(out.0 < self.ops.len(), "output node {} does not exist", out.0);
        cv_assert!(!self.inputs.is_empty(), "the graph has no input");

        // A node may only reference strictly earlier nodes, which is what makes
        // straight-line evaluation valid without a topological sort.
        for (i, op) in self.ops.iter().enumerate() {
            for dep in deps(op) {
                if dep.0 >= i {
                    return Err(Error::new(
                        Code::StsBadArg,
                        format!("node {i} references node {} which is not earlier", dep.0),
                    ));
                }
            }
        }
        // A recorded-by-name kernel has no CPU binding. Refusing here names
        // the kernel; evaluating it as a pass-through would produce an image
        // that is subtly not what the graph says.
        for op in &self.ops {
            if let Op::Record { id, .. } = op {
                return Err(Error::new(
                    Code::StsNotImplemented,
                    format!("the kernel `{id}` is recorded but has no CPU binding"),
                ));
            }
        }
        Ok(Compiled { ops: self.ops.clone(), inputs: self.inputs.clone(), output: out, backend })
    }
}

/// Operands of one op.
fn deps(op: &Op) -> Vec<GMat> {
    match op {
        Op::Input => vec![],
        Op::Record { operands, .. } => operands.clone(),
        Op::CvtColorBgr2Gray(a)
        | Op::CvtColorGray2Bgr(a)
        | Op::GaussianBlur(a, _, _)
        | Op::Blur(a, _)
        | Op::MedianBlur(a, _)
        | Op::Sobel(a, _, _, _, _)
        | Op::Laplacian(a, _, _)
        | Op::Canny(a, _, _)
        | Op::Threshold(a, _, _)
        | Op::Resize(a, _, _)
        | Op::Erode(a, _)
        | Op::Dilate(a, _)
        | Op::AddC(a, _)
        | Op::MulC(a, _)
        | Op::BitwiseNot(a)
        | Op::Filter2D(a, _, _, _)
        | Op::Split(a, _)
        | Op::Flip(a, _)
        | Op::EqualizeHist(a) => vec![*a],
        Op::Add(a, b)
        | Op::Sub(a, b)
        | Op::Mul(a, b)
        | Op::AbsDiff(a, b)
        | Op::BitwiseAnd(a, b)
        | Op::BitwiseOr(a, b) => vec![*a, *b],
        Op::Merge(v) => v.clone(),
    }
}

/// `cv::GCompiled` — a validated graph, ready to run on frame after frame.
#[derive(Debug, Clone)]
pub struct Compiled {
    ops: Vec<Op>,
    inputs: Vec<GMat>,
    output: GMat,
    backend: Backend,
}

impl Compiled {
    /// Which backend this graph was compiled for.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Number of graph inputs.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// `cv::GCompiled::operator()` for a single-input graph.
    pub fn apply<S: MatLike + ?Sized>(&self, src: &S) -> Result<Mat> {
        self.apply_many(&[src])
    }

    /// `cv::GCompiled::operator()` for a graph with several inputs.
    pub fn apply_many<S: MatLike + ?Sized>(&self, srcs: &[&S]) -> Result<Mat> {
        cv_assert!(
            srcs.len() == self.inputs.len(),
            "graph takes {} inputs, got {}",
            self.inputs.len(),
            srcs.len()
        );
        let mut vals: Vec<Option<Mat>> = vec![None; self.ops.len()];
        let mut next_input = 0usize;

        for i in 0..self.ops.len() {
            let v = self.eval(i, &vals, srcs, &mut next_input)?;
            vals[i] = Some(v);
        }
        vals[self.output.0]
            .take()
            .ok_or_else(|| Error::new(Code::StsError, "the output node produced nothing".to_string()))
    }

    /// Evaluate node `i`.
    fn eval<S: MatLike + ?Sized>(
        &self,
        i: usize,
        vals: &[Option<Mat>],
        srcs: &[&S],
        next_input: &mut usize,
    ) -> Result<Mat> {
        use cv_imgproc::{
            blur, canny, cvt_color, dilate, equalize_hist, erode, filter_2d, gaussian_blur,
            get_structuring_element, laplacian, median_blur, resize, sobel, threshold, ColorConversion,
            Interpolation, MorphShape, ThresholdMethod, ThresholdType,
        };
        let get = |m: GMat| -> Result<&Mat> {
            vals[m.0]
                .as_ref()
                .ok_or_else(|| Error::new(Code::StsError, format!("node {} was not evaluated", m.0)))
        };
        let border = cv_core::BorderType::Reflect101;

        Ok(match &self.ops[i] {
            // Compilation refuses these, so evaluation never reaches one.
            Op::Record { id, .. } => {
                return Err(Error::new(
                    Code::StsNotImplemented,
                    format!("the kernel `{id}` has no CPU binding"),
                ))
            }
            Op::Input => {
                let m = Mat::copy_from(srcs[*next_input])?;
                *next_input += 1;
                m
            }
            Op::CvtColorBgr2Gray(a) => cvt_color(get(*a)?, ColorConversion::Bgr2Gray)?,
            Op::CvtColorGray2Bgr(a) => cvt_color(get(*a)?, ColorConversion::Gray2Bgr)?,
            Op::GaussianBlur(a, k, s) => {
                gaussian_blur(get(*a)?, cv_core::Size::new(*k, *k), *s, *s, border)?
            }
            Op::Blur(a, k) => blur(get(*a)?, cv_core::Size::new(*k, *k), border)?,
            Op::MedianBlur(a, k) => median_blur(get(*a)?, *k)?,
            Op::Sobel(a, d, dx, dy, k) => {
                sobel(get(*a)?, Some(*d), *dx, *dy, *k, 1.0, 0.0, border)?
            }
            Op::Laplacian(a, d, k) => laplacian(get(*a)?, Some(*d), *k, 1.0, 0.0, border)?,
            Op::Canny(a, lo, hi) => canny(get(*a)?, *lo, *hi, 3, false)?,
            Op::Threshold(a, t, m) => {
                threshold(get(*a)?, *t, *m, ThresholdType::Binary, ThresholdMethod::Manual)?.dst
            }
            Op::Resize(a, w, h) => {
                resize(get(*a)?, cv_core::Size::new(*w, *h), 0.0, 0.0, Interpolation::Linear)?
            }
            Op::Erode(a, k) => {
                let el = get_structuring_element(MorphShape::Rect, cv_core::Size::new(*k, *k), cv_core::Point::new(-1, -1))?;
                erode(get(*a)?, &el, cv_core::Point::new(-1, -1), 1, border)?
            }
            Op::Dilate(a, k) => {
                let el = get_structuring_element(MorphShape::Rect, cv_core::Size::new(*k, *k), cv_core::Point::new(-1, -1))?;
                dilate(get(*a)?, &el, cv_core::Point::new(-1, -1), 1, border)?
            }
            Op::Add(a, b) => cv_core::add(get(*a)?, get(*b)?)?,
            Op::Sub(a, b) => cv_core::subtract(get(*a)?, get(*b)?)?,
            Op::Mul(a, b) => cv_core::multiply(get(*a)?, get(*b)?, 1.0)?,
            Op::AbsDiff(a, b) => cv_core::absdiff(get(*a)?, get(*b)?)?,
            Op::BitwiseAnd(a, b) => cv_core::bitwise_and(get(*a)?, get(*b)?)?,
            Op::BitwiseOr(a, b) => cv_core::bitwise_or(get(*a)?, get(*b)?)?,
            Op::BitwiseNot(a) => cv_core::bitwise_not(get(*a)?)?,
            Op::AddC(a, s) => {
                let src = get(*a)?;
                let mut k = Mat::zeros(src.rows(), src.cols(), src.mat_type())?;
                cv_core::arithm::set_to(&mut k, *s)?;
                cv_core::add(src, &k)?
            }
            Op::MulC(a, s) => {
                let src = get(*a)?;
                let mut k = Mat::zeros(src.rows(), src.cols(), src.mat_type())?;
                cv_core::arithm::set_to(&mut k, Scalar::all(1.0))?;
                cv_core::multiply(src, &k, *s)?
            }
            Op::Filter2D(a, kern, kw, kh) => filter_2d(
                get(*a)?,
                kern,
                cv_core::Size::new(*kw, *kh),
                None,
                cv_core::Point::new(-1, -1),
                0.0,
                border,
            )?,
            Op::Split(a, c) => {
                let parts = cv_core::split(get(*a)?)?;
                cv_assert!(*c < parts.len(), "channel {c} out of range for {} planes", parts.len());
                parts.into_iter().nth(*c).expect("checked above")
            }
            Op::Merge(parts) => {
                let mats: Result<Vec<Mat>> = parts.iter().map(|p| Ok(get(*p)?.clone())).collect();
                cv_core::merge(&mats?)?
            }
            Op::Flip(a, code) => cv_core::flip(get(*a)?, *code)?,
            Op::EqualizeHist(a) => equalize_hist(get(*a)?)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::MatLikeMut;

    fn gradient(w: i32, h: i32) -> Mat {
        let mut m = Mat::zeros(h, w, cv_core::types::CV_8UC1).unwrap();
        for y in 0..h {
            for x in 0..w {
                m.pixel_mut::<u8>(y, x).unwrap()[0] = ((x * 7 + y * 13) % 256) as u8;
            }
        }
        m
    }

    fn bgr(w: i32, h: i32) -> Mat {
        let mut m = Mat::zeros(h, w, cv_core::types::CV_8UC3).unwrap();
        for y in 0..h {
            for x in 0..w {
                let p = m.pixel_mut::<u8>(y, x).unwrap();
                p[0] = (x % 256) as u8;
                p[1] = (y % 256) as u8;
                p[2] = ((x + y) % 256) as u8;
            }
        }
        m
    }

    #[test]
    fn an_empty_graph_will_not_compile() {
        let g = GComputation::new();
        assert!(g.compile().is_err(), "no output was set");
    }

    #[test]
    fn a_graph_without_an_output_will_not_compile() {
        let mut g = GComputation::new();
        let _ = g.input();
        assert!(g.compile().is_err());
    }

    #[test]
    fn identity_graph_returns_its_input() {
        let mut g = GComputation::new();
        let i = g.input();
        g.set_output(i);
        let c = g.compile().unwrap();
        let src = gradient(8, 6);
        let out = c.apply(&src).unwrap();
        assert_eq!(out.to_vec::<u8>().unwrap(), src.to_vec::<u8>().unwrap());
    }

    #[test]
    fn a_pipeline_matches_running_the_steps_by_hand() {
        use cv_imgproc::{blur, canny, cvt_color, ColorConversion};
        let src = bgr(32, 24);

        let mut g = GComputation::new();
        let i = g.input();
        let gray = g.cvt_color_bgr2gray(i);
        let b = g.blur(gray, 3);
        let e = g.canny(b, 50.0, 150.0);
        g.set_output(e);
        let got = g.compile().unwrap().apply(&src).unwrap();

        let want = canny(
            &blur(
                &cvt_color(&src, ColorConversion::Bgr2Gray).unwrap(),
                cv_core::Size::new(3, 3),
                cv_core::BorderType::Reflect101,
            )
            .unwrap(),
            50.0,
            150.0,
            3,
            false,
        )
        .unwrap();
        assert_eq!(got.to_vec::<u8>().unwrap(), want.to_vec::<u8>().unwrap());
    }

    #[test]
    fn a_compiled_graph_is_reusable_across_frames() {
        let mut g = GComputation::new();
        let i = g.input();
        let t = g.threshold(i, 100.0, 255.0);
        g.set_output(t);
        let c = g.compile().unwrap();

        for seed in [8, 16, 32] {
            let f = gradient(seed, seed);
            let out = c.apply(&f).unwrap();
            assert_eq!((out.rows(), out.cols()), (seed, seed));
            // Every output pixel is 0 or 255.
            assert!(out.to_vec::<u8>().unwrap().iter().all(|&v| v == 0 || v == 255));
        }
    }

    #[test]
    fn two_input_graph_subtracts() {
        let mut g = GComputation::new();
        let a = g.input();
        let b = g.input();
        let d = g.abs_diff(a, b);
        g.set_output(d);
        let c = g.compile().unwrap();
        assert_eq!(c.input_count(), 2);

        let x = gradient(8, 8);
        let y = gradient(8, 8);
        let out = c.apply_many(&[&x, &y]).unwrap();
        assert!(out.to_vec::<u8>().unwrap().iter().all(|&v| v == 0), "a - a is zero");
    }

    #[test]
    fn wrong_input_count_is_rejected() {
        let mut g = GComputation::new();
        let a = g.input();
        let b = g.input();
        let s = g.add(a, b);
        g.set_output(s);
        let c = g.compile().unwrap();
        let x = gradient(4, 4);
        assert!(c.apply(&x).is_err(), "a two-input graph cannot take one frame");
    }

    #[test]
    fn split_and_merge_round_trip() {
        let src = bgr(8, 8);
        let mut g = GComputation::new();
        let i = g.input();
        let b = g.split(i, 0);
        let gg = g.split(i, 1);
        let r = g.split(i, 2);
        let m = g.merge(&[b, gg, r]);
        g.set_output(m);
        let out = g.compile().unwrap().apply(&src).unwrap();
        assert_eq!(out.channels(), 3);
        assert_eq!(out.to_vec::<u8>().unwrap(), src.to_vec::<u8>().unwrap());
    }

    #[test]
    fn a_diamond_graph_reuses_an_intermediate() {
        // gray -> {blur, sobel} -> absdiff. The gray node is evaluated once.
        let src = bgr(16, 16);
        let mut g = GComputation::new();
        let i = g.input();
        let gray = g.cvt_color_bgr2gray(i);
        let a = g.blur(gray, 3);
        let b = g.median_blur(gray, 3);
        let d = g.abs_diff(a, b);
        g.set_output(d);
        assert_eq!(g.len(), 5);
        let out = g.compile().unwrap().apply(&src).unwrap();
        assert_eq!((out.rows(), out.cols()), (16, 16));
    }

    #[test]
    fn mul_c_scales() {
        let mut g = GComputation::new();
        let i = g.input();
        let s = g.mul_c(i, 0.0);
        g.set_output(s);
        let out = g.compile().unwrap().apply(&gradient(4, 4)).unwrap();
        assert!(out.to_vec::<u8>().unwrap().iter().all(|&v| v == 0));
    }

    #[test]
    fn bitwise_not_inverts() {
        let mut g = GComputation::new();
        let i = g.input();
        let n = g.bitwise_not(i);
        g.set_output(n);
        let src = gradient(4, 4);
        let out = g.compile().unwrap().apply(&src).unwrap();
        for (a, b) in src.to_vec::<u8>().unwrap().iter().zip(out.to_vec::<u8>().unwrap().iter()) {
            assert_eq!(*b, !*a);
        }
    }

    #[test]
    fn resize_changes_the_output_size() {
        let mut g = GComputation::new();
        let i = g.input();
        let r = g.resize(i, 5, 7);
        g.set_output(r);
        let out = g.compile().unwrap().apply(&gradient(20, 20)).unwrap();
        assert_eq!((out.rows(), out.cols()), (7, 5));
    }

    #[test]
    fn backend_selection_is_recorded() {
        let mut g = GComputation::new();
        let i = g.input();
        g.set_output(i);
        assert_eq!(g.compile().unwrap().backend(), Backend::Portable);
        assert_eq!(g.compile_on(Backend::Rlx).unwrap().backend(), Backend::Rlx);
    }

    #[test]
    fn morphology_and_histogram_nodes_run() {
        let src = gradient(16, 16);
        let mut g = GComputation::new();
        let i = g.input();
        let e = g.erode(i, 3);
        let d = g.dilate(e, 3);
        let h = g.equalize_hist(d);
        g.set_output(h);
        let out = g.compile().unwrap().apply(&src).unwrap();
        assert_eq!((out.rows(), out.cols()), (16, 16));
    }

    #[test]
    fn filter_2d_node_applies_the_kernel() {
        let mut g = GComputation::new();
        let i = g.input();
        // A 1x1 identity kernel leaves the image alone.
        let f = g.filter_2d(i, &[1.0], 1, 1);
        g.set_output(f);
        let src = gradient(6, 6);
        let out = g.compile().unwrap().apply(&src).unwrap();
        assert_eq!(out.to_vec::<u8>().unwrap(), src.to_vec::<u8>().unwrap());
    }

    #[test]
    fn flip_node_mirrors() {
        let mut g = GComputation::new();
        let i = g.input();
        let f = g.flip(i, 1);
        g.set_output(f);
        let src = gradient(4, 2);
        let out = g.compile().unwrap().apply(&src).unwrap();
        for y in 0..2 {
            for x in 0..4 {
                assert_eq!(out.at_f64(y, x, 0).unwrap(), src.at_f64(y, 3 - x, 0).unwrap());
            }
        }
    }
}
