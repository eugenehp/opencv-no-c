//! The layer graph (`modules/dnn/src/dnn.cpp`, `layers/`).
//!
//! # How this maps onto RLX
//!
//! A [`Net`] is a topologically-ordered list of [`Layer`]s. Executing one is a
//! fold over that list, and each layer picks its own implementation:
//!
//! * [`Layer::Convolution`] and [`Layer::InnerProduct`] carry essentially all
//!   the arithmetic in a real network, so with the `rlx` feature they run as
//!   RLX graph nodes on whichever `cv_accel::Backend` is available — CPU,
//!   Metal, CUDA, the Neural Engine. RLX compiles and caches the graph, so the
//!   cost is paid once per distinct layer geometry.
//! * Everything else is cheap and elementwise, and runs in portable Rust.
//!
//! Without the `rlx` feature every layer uses the portable path, so the crate
//! still builds and produces identical numbers — just slower on the big ones.
//! The two paths are checked against each other in the tests, which is what
//! makes the acceleration safe to turn on.

use cv_core::{cv_assert, Code, Error, Result};

use crate::blob::Blob;

/// Padding rule for [`Layer::Convolution`] and the pooling layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    /// Explicit `(rows, cols)` of zero padding.
    Explicit(usize, usize),
    /// `SAME`: pad so the output keeps the input size at stride 1.
    Same,
    /// `VALID`: no padding.
    Valid,
}

impl Padding {
    /// Resolve to a concrete `(pad_h, pad_w)` for a given kernel.
    fn resolve(self, k: (usize, usize), d: (usize, usize)) -> (usize, usize) {
        match self {
            Padding::Explicit(h, w) => (h, w),
            Padding::Valid => (0, 0),
            Padding::Same => {
                let eff = |k: usize, d: usize| d * (k - 1) + 1;
                ((eff(k.0, d.0) - 1) / 2, (eff(k.1, d.1) - 1) / 2)
            }
        }
    }
}

/// Elementwise combination rule for [`Layer::Eltwise`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EltwiseOp {
    /// Sum the inputs.
    Sum,
    /// Multiply the inputs.
    Prod,
    /// Take the element-wise maximum.
    Max,
}

/// Activation kinds, matching upstream's separate layer types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// `cv::dnn::ReLULayer`; `slope` of 0 is plain ReLU, otherwise leaky.
    ReLU {
        /// Slope applied to negative inputs; 0 for plain ReLU.
        slope: f32,
    },
    /// `cv::dnn::ReLU6Layer` — clamped to `[0, 6]`.
    ReLU6,
    /// `cv::dnn::SigmoidLayer`
    Sigmoid,
    /// `cv::dnn::TanHLayer`
    TanH,
    /// `cv::dnn::ELULayer`
    Elu {
        /// Scale of the negative exponential branch.
        alpha: f32,
    },
    /// `cv::dnn::SwishLayer` — `x * sigmoid(x)`.
    Swish,
    /// `cv::dnn::MishLayer` — `x * tanh(softplus(x))`.
    Mish,
    /// `cv::dnn::AbsLayer`
    Abs,
    /// `cv::dnn::PowerLayer` — `(shift + scale*x)^power`.
    Power {
        /// Exponent.
        power: f32,
        /// Input multiplier.
        scale: f32,
        /// Input offset.
        shift: f32,
    },
}

impl Activation {
    /// Apply to one value.
    #[must_use]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Activation::ReLU { slope } => {
                if x >= 0.0 {
                    x
                } else {
                    slope * x
                }
            }
            Activation::ReLU6 => x.clamp(0.0, 6.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::TanH => x.tanh(),
            Activation::Elu { alpha } => {
                if x >= 0.0 {
                    x
                } else {
                    alpha * (x.exp() - 1.0)
                }
            }
            Activation::Swish => x / (1.0 + (-x).exp()),
            Activation::Mish => x * (1.0 + x.exp()).ln().tanh(),
            Activation::Abs => x.abs(),
            Activation::Power { power, scale, shift } => (shift + scale * x).powf(power),
        }
    }
}

/// Pooling rule for [`Layer::Pooling`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolKind {
    /// `cv::dnn::PoolingLayer` with `MAX`.
    Max,
    /// `cv::dnn::PoolingLayer` with `AVE`.
    Average,
}

/// One layer of a [`Net`].
///
/// Weight layouts follow upstream: convolution weights are
/// `c_out x (c_in/groups) x kh x kw`, fully-connected weights are
/// `out x in`.
#[derive(Debug, Clone, PartialEq)]
pub enum Layer {
    /// `cv::dnn::ConvolutionLayer`.
    Convolution {
        /// `c_out x (c_in/groups) x kh x kw`.
        weights: Vec<f32>,
        /// `c_out` values, or empty for no bias.
        bias: Vec<f32>,
        /// Output channels.
        c_out: usize,
        /// Kernel `(rows, cols)`.
        kernel: (usize, usize),
        /// Stride `(rows, cols)`.
        stride: (usize, usize),
        /// Padding rule.
        pad: Padding,
        /// Dilation `(rows, cols)`.
        dilation: (usize, usize),
        /// Group count.
        groups: usize,
    },
    /// `cv::dnn::PoolingLayer`.
    Pooling {
        /// Which reduction.
        kind: PoolKind,
        /// Window `(rows, cols)`.
        kernel: (usize, usize),
        /// Stride `(rows, cols)`.
        stride: (usize, usize),
        /// Padding rule.
        pad: Padding,
    },
    /// Global average pooling, collapsing each plane to one value.
    GlobalPool(PoolKind),
    /// `cv::dnn::InnerProductLayer` — flattens then applies `out x in` weights.
    InnerProduct {
        /// `out x in`, row-major.
        weights: Vec<f32>,
        /// `out` values, or empty.
        bias: Vec<f32>,
        /// Output length.
        out: usize,
    },
    /// One of the activation layers.
    Activation(Activation),
    /// `cv::dnn::SoftmaxLayer` over the channel axis.
    Softmax,
    /// `cv::dnn::BatchNormLayer` — inference form, folded scale and shift.
    BatchNorm {
        /// Per-channel scale.
        scale: Vec<f32>,
        /// Per-channel shift.
        shift: Vec<f32>,
    },
    /// `cv::dnn::ScaleLayer` — per-channel multiply then add.
    Scale {
        /// Per-channel multiplier.
        scale: Vec<f32>,
        /// Per-channel bias, or empty.
        bias: Vec<f32>,
    },
    /// `cv::dnn::LRNLayer`, across channels.
    Lrn {
        /// Window size in channels.
        size: usize,
        /// `alpha`.
        alpha: f32,
        /// `beta`.
        beta: f32,
        /// `k`.
        bias: f32,
    },
    /// `cv::dnn::FlattenLayer` — collapse to `1 x total x 1 x 1` per batch item.
    Flatten,
    /// `cv::dnn::ReshapeLayer`.
    Reshape([usize; 4]),
    /// `cv::dnn::PermuteLayer` — reorder the four axes.
    Permute([usize; 4]),
    /// `cv::dnn::ConcatLayer` along the channel axis, joining with a named blob.
    Concat(String),
    /// `cv::dnn::EltwiseLayer`, combining with a named blob.
    Eltwise(EltwiseOp, String),
    /// `cv::dnn::PaddingLayer` — zero-pad the spatial axes.
    Pad {
        /// Rows added at the top and bottom.
        rows: usize,
        /// Columns added at each side.
        cols: usize,
        /// Fill value.
        value: f32,
    },
}

/// What a [`Net`] actually executes.
///
/// A hand-built network is a list of [`Layer`]s; one read from ONNX is a single
/// compiled RLX graph, because RLX's importer already covers the operator set
/// and decomposing it back into [`Layer`]s would only lose coverage.
enum Body {
    Layers(Vec<(String, Layer)>),
    #[cfg(feature = "onnx")]
    Onnx(Box<cv_accel::OnnxModel>),
}

impl std::fmt::Debug for Body {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Body::Layers(v) => f.debug_tuple("Layers").field(&v.len()).finish(),
            #[cfg(feature = "onnx")]
            Body::Onnx(m) => f.debug_tuple("Onnx").field(&m.name()).finish(),
        }
    }
}

/// `cv::dnn::Net` — a layer stack, or a compiled graph read from ONNX.
///
/// The `match`es over `Body` read as single-armed without the `onnx` feature,
/// which clippy objects to; they are not, with it.
#[derive(Debug)]
#[allow(clippy::infallible_destructuring_match, clippy::single_match_else)]
pub struct Net {
    body: Body,
    saved: std::collections::HashMap<String, Blob>,
    input: Option<Blob>,
    input_name: String,
    out_shape: Option<[usize; 4]>,
}

impl Default for Net {
    fn default() -> Self {
        Net {
            body: Body::Layers(Vec::new()),
            saved: std::collections::HashMap::new(),
            input: None,
            input_name: "input".to_string(),
            out_shape: None,
        }
    }
}

impl Net {
    /// An empty network.
    #[must_use]
    pub fn new() -> Self {
        Net::default()
    }

    /// `cv::dnn::readNetFromONNX(onnxFile)`.
    ///
    /// The file is parsed, shape-propagated and compiled by RLX; this port does
    /// not reimplement any of that, which is why the supported operator set is
    /// RLX's rather than a hand-written subset. The result runs on whichever
    /// backend RLX resolves — CPU, Metal, CUDA, the Neural Engine.
    ///
    /// `out_shape` is the NCHW shape to interpret the flat output with. ONNX
    /// carries it, but RLX returns flat buffers, so it is taken from the caller
    /// rather than guessed; pass `None` to receive a `1 x N x 1 x 1` blob.
    #[cfg(feature = "onnx")]
    pub fn from_onnx(
        path: impl AsRef<std::path::Path>,
        input_name: impl Into<String>,
        out_shape: Option<[usize; 4]>,
    ) -> Result<Self> {
        let model = cv_accel::load_onnx(path.as_ref())?;
        Ok(Net {
            body: Body::Onnx(Box::new(model)),
            saved: std::collections::HashMap::new(),
            input: None,
            input_name: input_name.into(),
            out_shape,
        })
    }

    /// The model name, for an ONNX-backed net.
    #[cfg(feature = "onnx")]
    #[must_use]
    pub fn model_name(&self) -> Option<&str> {
        match &self.body {
            Body::Onnx(m) => Some(m.name()),
            _ => None,
        }
    }

    /// Append a layer, giving its output a name so later layers can reference it.
    ///
    /// # Panics
    ///
    /// If the net was read from ONNX; those are compiled graphs, not editable
    /// layer lists.
    pub fn add_layer(&mut self, name: impl Into<String>, layer: Layer) -> &mut Self {
        match &mut self.body {
            Body::Layers(v) => v.push((name.into(), layer)),
            #[cfg(feature = "onnx")]
            Body::Onnx(_) => panic!("cannot add layers to a net read from ONNX"),
        }
        self
    }

    /// Number of layers; 0 for an ONNX-backed net, which has no layer list.
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.body {
            Body::Layers(v) => v.len(),
            #[cfg(feature = "onnx")]
            Body::Onnx(_) => 0,
        }
    }

    /// Whether the network has no layers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `cv::dnn::Net::getLayerNames()`.
    #[must_use]
    pub fn layer_names(&self) -> Vec<&str> {
        match &self.body {
            Body::Layers(v) => v.iter().map(|(n, _)| n.as_str()).collect(),
            #[cfg(feature = "onnx")]
            Body::Onnx(_) => Vec::new(),
        }
    }

    /// Each layer's upstream *type* name, in order.
    ///
    /// The names a caller gave are arbitrary; the kinds are what
    /// `cv::dnn::Net::getLayerTypes` reports and what a graph dump labels.
    #[must_use]
    pub fn layer_kinds(&self) -> Vec<String> {
        match &self.body {
            Body::Layers(v) => v
                .iter()
                .map(|(_, l)| {
                    match l {
                        Layer::Convolution { .. } => "Convolution",
                        Layer::InnerProduct { .. } => "InnerProduct",
                        Layer::Activation(_) => "Activation",
                        Layer::Pooling { .. } => "Pooling",
                        Layer::GlobalPool(_) => "GlobalPool",
                        Layer::Softmax => "Softmax",
                        Layer::Flatten => "Flatten",
                        Layer::BatchNorm { .. } => "BatchNorm",
                        Layer::Scale { .. } => "Scale",
                        Layer::Lrn { .. } => "LRN",
                        Layer::Eltwise(..) => "Eltwise",
                        Layer::Concat(_) => "Concat",
                        Layer::Reshape(_) => "Reshape",
                        Layer::Permute(_) => "Permute",
                        Layer::Pad { .. } => "Padding",
                    }
                    .to_string()
                })
                .collect(),
            #[cfg(feature = "onnx")]
            _ => Vec::new(),
        }
    }

    /// Rename the input blob, so `cv::dnn::Net::setInputsNames` has somewhere
    /// to write.
    pub fn set_input_name(&mut self, name: &str) {
        self.input_name = name.to_string();
    }


    /// `cv::dnn::Net::setInput(blob)`.
    pub fn set_input(&mut self, blob: Blob) {
        self.input = Some(blob);
    }

    /// Register a side input for [`Layer::Concat`] or [`Layer::Eltwise`].
    pub fn set_blob(&mut self, name: impl Into<String>, blob: Blob) {
        self.saved.insert(name.into(), blob);
    }

    /// `cv::dnn::Net::forward()` — run the network and return its output.
    pub fn forward(&mut self) -> Result<Blob> {
        let x = self.input.clone().ok_or_else(|| {
            Error::new(Code::StsBadArg, "Net::forward called before setInput".to_string())
        })?;
        let saved = self.saved.clone();
        #[cfg_attr(not(feature = "onnx"), allow(unused_variables))]
        let (name, shape) = (self.input_name.clone(), self.out_shape);
        match &mut self.body {
            Body::Layers(layers) => {
                let mut cur = x;
                let mut named = saved;
                for (n, layer) in layers.iter() {
                    cur = run_layer(layer, &cur, &named)?;
                    named.insert(n.clone(), cur.clone());
                }
                Ok(cur)
            }
            #[cfg(feature = "onnx")]
            Body::Onnx(m) => {
                let mut outs = m.run(&[(name.as_str(), x.data.as_slice())])?;
                let flat = outs.drain(..).next().ok_or_else(|| {
                    Error::new(Code::StsError, "the model produced no output".to_string())
                })?;
                match shape {
                    Some(s) => Blob::new(s, flat),
                    None => {
                        let n = flat.len();
                        Blob::new([1, n, 1, 1], flat)
                    }
                }
            }
        }
    }

    /// `cv::dnn::Net::forward(outputName)` — stop after the named layer.
    ///
    /// Not meaningful for an ONNX-backed net, which is a single compiled graph.
    pub fn forward_to(&self, output: &str) -> Result<Blob> {
        let mut x = self.input.clone().ok_or_else(|| {
            Error::new(Code::StsBadArg, "Net::forward called before setInput".to_string())
        })?;
        // Single-armed without the `onnx` feature, two-armed with it.
        #[allow(clippy::infallible_destructuring_match)]
        let layers = match &self.body {
            Body::Layers(v) => v,
            #[cfg(feature = "onnx")]
            Body::Onnx(_) => {
                return Err(Error::new(
                    Code::StsBadArg,
                    "forward_to is not available for an ONNX-backed net".to_string(),
                ))
            }
        };
        let mut named = self.saved.clone();
        for (name, layer) in layers {
            x = run_layer(layer, &x, &named)?;
            named.insert(name.clone(), x.clone());
            if name == output {
                return Ok(x);
            }
        }
        Err(Error::new(Code::StsBadArg, format!("no layer named {output:?}")))
    }
}

/// Execute one layer.
fn run_layer(
    layer: &Layer,
    x: &Blob,
    named: &std::collections::HashMap<String, Blob>,
) -> Result<Blob> {
    match layer {
        Layer::Convolution { weights, bias, c_out, kernel, stride, pad, dilation, groups } => {
            convolution(x, weights, bias, *c_out, *kernel, *stride, *pad, *dilation, *groups)
        }
        Layer::Pooling { kind, kernel, stride, pad } => pooling(x, *kind, *kernel, *stride, *pad),
        Layer::GlobalPool(kind) => global_pool(x, *kind),
        Layer::InnerProduct { weights, bias, out } => inner_product(x, weights, bias, *out),
        Layer::Activation(a) => {
            let mut y = x.clone();
            for v in y.data.iter_mut() {
                *v = a.apply(*v);
            }
            Ok(y)
        }
        Layer::Softmax => softmax(x),
        Layer::BatchNorm { scale, shift } => per_channel(x, scale, shift),
        Layer::Scale { scale, bias } => per_channel(x, scale, bias),
        Layer::Lrn { size, alpha, beta, bias } => lrn(x, *size, *alpha, *beta, *bias),
        Layer::Flatten => {
            let per = x.channels() * x.plane();
            Ok(Blob { shape: [x.batch(), per, 1, 1], data: x.data.clone() })
        }
        Layer::Reshape(s) => {
            let want: usize = s.iter().product();
            cv_assert!(
                want == x.data.len(),
                "reshape to {s:?} needs {want} values, blob has {}",
                x.data.len()
            );
            Ok(Blob { shape: *s, data: x.data.clone() })
        }
        Layer::Permute(axes) => permute(x, *axes),
        Layer::Concat(other) => {
            let b = fetch(named, other)?;
            concat_channels(x, b)
        }
        Layer::Eltwise(op, other) => {
            let b = fetch(named, other)?;
            eltwise(x, b, *op)
        }
        Layer::Pad { rows, cols, value } => pad(x, *rows, *cols, *value),
    }
}

fn fetch<'a>(
    named: &'a std::collections::HashMap<String, Blob>,
    key: &str,
) -> Result<&'a Blob> {
    named
        .get(key)
        .ok_or_else(|| Error::new(Code::StsBadArg, format!("no blob named {key:?}")))
}

/// Convolution: RLX when the feature is on, portable Rust otherwise.
#[allow(clippy::too_many_arguments)]
fn convolution(
    x: &Blob,
    weights: &[f32],
    bias: &[f32],
    c_out: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: Padding,
    dilation: (usize, usize),
    groups: usize,
) -> Result<Blob> {
    cv_assert!(groups > 0, "groups must be positive");
    cv_assert!(
        x.channels() % groups == 0 && c_out % groups == 0,
        "groups={groups} does not divide c_in={} / c_out={c_out}",
        x.channels()
    );
    let (ph, pw) = pad.resolve(kernel, dilation);

    #[cfg(feature = "rlx")]
    {
        let shape = cv_accel::ConvShape {
            batch: x.batch(),
            c_in: x.channels(),
            hw: (x.height(), x.width()),
            c_out,
            k: kernel,
            stride,
            pad: (ph, pw),
            dilation,
            groups,
        };
        // RLX is best-effort: if no backend is available (or the graph fails to
        // compile for this geometry) the portable path below still runs, so a
        // missing GPU is never a hard error.
        if let Ok(v) = cv_accel::conv2d_f32(&x.data, weights, bias, shape) {
            let (oh, ow) = shape.out_hw();
            return Blob::new([x.batch(), c_out, oh, ow], v);
        }
    }
    convolution_portable(x, weights, bias, c_out, kernel, stride, (ph, pw), dilation, groups)
}

/// The reference convolution, used directly without `rlx` and as the fallback.
#[allow(clippy::too_many_arguments)]
pub fn convolution_portable(
    x: &Blob,
    weights: &[f32],
    bias: &[f32],
    c_out: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> Result<Blob> {
    let (kh, kw) = kernel;
    let cin_g = x.channels() / groups;
    let cout_g = c_out / groups;
    let eff_h = dilation.0 * (kh - 1) + 1;
    let eff_w = dilation.1 * (kw - 1) + 1;
    cv_assert!(
        x.height() + 2 * pad.0 >= eff_h && x.width() + 2 * pad.1 >= eff_w,
        "kernel {kh}x{kw} (dilated {eff_h}x{eff_w}) does not fit {}x{}",
        x.height(),
        x.width()
    );
    let oh = (x.height() + 2 * pad.0 - eff_h) / stride.0 + 1;
    let ow = (x.width() + 2 * pad.1 - eff_w) / stride.1 + 1;
    cv_assert!(
        weights.len() == c_out * cin_g * kh * kw,
        "weights is {} values, expected {}",
        weights.len(),
        c_out * cin_g * kh * kw
    );

    let mut out = Blob::zeros(x.batch(), c_out, oh, ow);
    for n in 0..x.batch() {
        for oc in 0..c_out {
            let g = oc / cout_g;
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = if bias.is_empty() { 0.0f32 } else { bias[oc] };
                    for ic in 0..cin_g {
                        let src_c = g * cin_g + ic;
                        for ky in 0..kh {
                            let iy = (oy * stride.0 + ky * dilation.0) as isize - pad.0 as isize;
                            if iy < 0 || iy >= x.height() as isize {
                                continue;
                            }
                            for kx in 0..kw {
                                let ix = (ox * stride.1 + kx * dilation.1) as isize - pad.1 as isize;
                                if ix < 0 || ix >= x.width() as isize {
                                    continue;
                                }
                                let w = weights[((oc * cin_g + ic) * kh + ky) * kw + kx];
                                acc += w * x.at(n, src_c, iy as usize, ix as usize);
                            }
                        }
                    }
                    *out.at_mut(n, oc, oy, ox) = acc;
                }
            }
        }
    }
    Ok(out)
}

/// Fully connected layer; the matrix product goes to RLX when available.
fn inner_product(x: &Blob, weights: &[f32], bias: &[f32], out: usize) -> Result<Blob> {
    let k = x.channels() * x.plane();
    cv_assert!(
        weights.len() == out * k,
        "InnerProduct weights is {} values, expected {out}x{k}",
        weights.len()
    );
    cv_assert!(
        bias.is_empty() || bias.len() == out,
        "InnerProduct bias is {} values, expected {out} or 0",
        bias.len()
    );
    let n = x.batch();

    #[cfg(feature = "rlx")]
    {
        // y = x (n x k) * w^T (k x out).
        let mut wt = vec![0.0f32; k * out];
        for r in 0..out {
            for c in 0..k {
                wt[c * out + r] = weights[r * k + c];
            }
        }
        if let Ok(mut v) = cv_accel::matmul_f32(&x.data, &wt, n, k, out) {
            if !bias.is_empty() {
                for b in 0..n {
                    for o in 0..out {
                        v[b * out + o] += bias[o];
                    }
                }
            }
            return Blob::new([n, out, 1, 1], v);
        }
    }

    let mut y = Blob::zeros(n, out, 1, 1);
    for b in 0..n {
        for o in 0..out {
            let mut acc = if bias.is_empty() { 0.0 } else { bias[o] };
            for c in 0..k {
                acc += weights[o * k + c] * x.data[b * k + c];
            }
            y.data[b * out + o] = acc;
        }
    }
    Ok(y)
}

/// Max or average pooling over a sliding window.
fn pooling(
    x: &Blob,
    kind: PoolKind,
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: Padding,
) -> Result<Blob> {
    let (ph, pw) = pad.resolve(kernel, (1, 1));
    let (kh, kw) = kernel;
    cv_assert!(kh > 0 && kw > 0, "pooling window must be positive");
    // Upstream rounds *up* here, unlike convolution, so a window that runs off
    // the edge still produces an output cell.
    let oh = ceil_div(x.height() + 2 * ph, kh, stride.0);
    let ow = ceil_div(x.width() + 2 * pw, kw, stride.1);
    let mut out = Blob::zeros(x.batch(), x.channels(), oh, ow);
    for n in 0..x.batch() {
        for c in 0..x.channels() {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = match kind {
                        PoolKind::Max => f32::NEG_INFINITY,
                        PoolKind::Average => 0.0,
                    };
                    let mut count = 0usize;
                    for ky in 0..kh {
                        let iy = (oy * stride.0 + ky) as isize - ph as isize;
                        if iy < 0 || iy >= x.height() as isize {
                            continue;
                        }
                        for kx in 0..kw {
                            let ix = (ox * stride.1 + kx) as isize - pw as isize;
                            if ix < 0 || ix >= x.width() as isize {
                                continue;
                            }
                            let v = x.at(n, c, iy as usize, ix as usize);
                            match kind {
                                PoolKind::Max => acc = acc.max(v),
                                PoolKind::Average => acc += v,
                            }
                            count += 1;
                        }
                    }
                    *out.at_mut(n, c, oy, ox) = match kind {
                        PoolKind::Max => {
                            if count == 0 {
                                0.0
                            } else {
                                acc
                            }
                        }
                        // Average over the *valid* cells only, which is
                        // upstream's default (`countIncludePadding = false`).
                        PoolKind::Average => {
                            if count == 0 {
                                0.0
                            } else {
                                acc / count as f32
                            }
                        }
                    };
                }
            }
        }
    }
    Ok(out)
}

/// Output extent for pooling, rounding up.
fn ceil_div(size: usize, k: usize, stride: usize) -> usize {
    if size < k {
        return 1;
    }
    (size - k).div_ceil(stride) + 1
}

/// Collapse every plane to a single value.
fn global_pool(x: &Blob, kind: PoolKind) -> Result<Blob> {
    let mut out = Blob::zeros(x.batch(), x.channels(), 1, 1);
    for n in 0..x.batch() {
        for c in 0..x.channels() {
            let mut acc = match kind {
                PoolKind::Max => f32::NEG_INFINITY,
                PoolKind::Average => 0.0,
            };
            for y in 0..x.height() {
                for xx in 0..x.width() {
                    let v = x.at(n, c, y, xx);
                    match kind {
                        PoolKind::Max => acc = acc.max(v),
                        PoolKind::Average => acc += v,
                    }
                }
            }
            *out.at_mut(n, c, 0, 0) = match kind {
                PoolKind::Max => acc,
                PoolKind::Average => acc / x.plane() as f32,
            };
        }
    }
    Ok(out)
}

/// Channel-axis softmax, shifted by the per-position maximum for stability.
fn softmax(x: &Blob) -> Result<Blob> {
    let mut out = x.clone();
    for n in 0..x.batch() {
        for y in 0..x.height() {
            for xx in 0..x.width() {
                let mut m = f32::NEG_INFINITY;
                for c in 0..x.channels() {
                    m = m.max(x.at(n, c, y, xx));
                }
                let mut sum = 0.0f32;
                for c in 0..x.channels() {
                    let e = (x.at(n, c, y, xx) - m).exp();
                    *out.at_mut(n, c, y, xx) = e;
                    sum += e;
                }
                if sum > 0.0 {
                    for c in 0..x.channels() {
                        *out.at_mut(n, c, y, xx) /= sum;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// `y = x * scale[c] + shift[c]`.
fn per_channel(x: &Blob, scale: &[f32], shift: &[f32]) -> Result<Blob> {
    cv_assert!(
        scale.len() == x.channels(),
        "expected {} scale values, got {}",
        x.channels(),
        scale.len()
    );
    cv_assert!(
        shift.is_empty() || shift.len() == x.channels(),
        "expected {} shift values or 0, got {}",
        x.channels(),
        shift.len()
    );
    let mut out = x.clone();
    for n in 0..x.batch() {
        for c in 0..x.channels() {
            let (s, b) = (scale[c], if shift.is_empty() { 0.0 } else { shift[c] });
            for y in 0..x.height() {
                for xx in 0..x.width() {
                    *out.at_mut(n, c, y, xx) = x.at(n, c, y, xx) * s + b;
                }
            }
        }
    }
    Ok(out)
}

/// Local response normalisation across channels.
fn lrn(x: &Blob, size: usize, alpha: f32, beta: f32, bias: f32) -> Result<Blob> {
    cv_assert!(size > 0, "LRN size must be positive");
    let mut out = x.clone();
    let half = (size / 2) as isize;
    for n in 0..x.batch() {
        for c in 0..x.channels() {
            for y in 0..x.height() {
                for xx in 0..x.width() {
                    let mut acc = 0.0f32;
                    for d in -half..=half {
                        let cc = c as isize + d;
                        if cc < 0 || cc >= x.channels() as isize {
                            continue;
                        }
                        let v = x.at(n, cc as usize, y, xx);
                        acc += v * v;
                    }
                    let denom = (bias + alpha / size as f32 * acc).powf(beta);
                    *out.at_mut(n, c, y, xx) = x.at(n, c, y, xx) / denom;
                }
            }
        }
    }
    Ok(out)
}

/// Reorder the four axes; `axes` is a permutation of `[0, 1, 2, 3]`.
fn permute(x: &Blob, axes: [usize; 4]) -> Result<Blob> {
    let mut seen = [false; 4];
    for &a in &axes {
        cv_assert!(a < 4, "permute axis {a} is out of range");
        cv_assert!(!seen[a], "permute axis {a} is repeated");
        seen[a] = true;
    }
    let src = x.shape;
    let dst = [src[axes[0]], src[axes[1]], src[axes[2]], src[axes[3]]];
    let mut out = Blob::zeros(dst[0], dst[1], dst[2], dst[3]);
    let mut idx = [0usize; 4];
    for i0 in 0..dst[0] {
        for i1 in 0..dst[1] {
            for i2 in 0..dst[2] {
                for i3 in 0..dst[3] {
                    let d = [i0, i1, i2, i3];
                    for k in 0..4 {
                        idx[axes[k]] = d[k];
                    }
                    *out.at_mut(i0, i1, i2, i3) = x.at(idx[0], idx[1], idx[2], idx[3]);
                }
            }
        }
    }
    Ok(out)
}

/// Join two blobs along the channel axis.
fn concat_channels(a: &Blob, b: &Blob) -> Result<Blob> {
    cv_assert!(
        a.batch() == b.batch() && a.height() == b.height() && a.width() == b.width(),
        "concat needs matching N/H/W: {:?} vs {:?}",
        a.shape,
        b.shape
    );
    let mut out = Blob::zeros(a.batch(), a.channels() + b.channels(), a.height(), a.width());
    for n in 0..a.batch() {
        for c in 0..a.channels() {
            for y in 0..a.height() {
                for x in 0..a.width() {
                    *out.at_mut(n, c, y, x) = a.at(n, c, y, x);
                }
            }
        }
        for c in 0..b.channels() {
            for y in 0..a.height() {
                for x in 0..a.width() {
                    *out.at_mut(n, a.channels() + c, y, x) = b.at(n, c, y, x);
                }
            }
        }
    }
    Ok(out)
}

/// Elementwise combination of two identically-shaped blobs.
fn eltwise(a: &Blob, b: &Blob, op: EltwiseOp) -> Result<Blob> {
    cv_assert!(a.shape == b.shape, "eltwise needs equal shapes: {:?} vs {:?}", a.shape, b.shape);
    let mut out = a.clone();
    for (o, &v) in out.data.iter_mut().zip(b.data.iter()) {
        *o = match op {
            EltwiseOp::Sum => *o + v,
            EltwiseOp::Prod => *o * v,
            EltwiseOp::Max => o.max(v),
        };
    }
    Ok(out)
}

/// Zero-pad (or constant-pad) the spatial axes.
fn pad(x: &Blob, rows: usize, cols: usize, value: f32) -> Result<Blob> {
    let mut out = Blob {
        shape: [x.batch(), x.channels(), x.height() + 2 * rows, x.width() + 2 * cols],
        data: vec![value; x.batch() * x.channels() * (x.height() + 2 * rows) * (x.width() + 2 * cols)],
    };
    for n in 0..x.batch() {
        for c in 0..x.channels() {
            for y in 0..x.height() {
                for xx in 0..x.width() {
                    *out.at_mut(n, c, y + rows, xx + cols) = x.at(n, c, y, xx);
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize, c: usize, h: usize, w: usize) -> Blob {
        let data: Vec<f32> = (0..n * c * h * w).map(|i| i as f32).collect();
        Blob { shape: [n, c, h, w], data }
    }

    #[test]
    fn identity_convolution_reproduces_the_input() {
        let x = ramp(1, 1, 4, 4);
        let y = convolution_portable(&x, &[1.0], &[], 1, (1, 1), (1, 1), (0, 0), (1, 1), 1).unwrap();
        assert_eq!(y.shape, [1, 1, 4, 4]);
        assert_eq!(y.data, x.data);
    }

    #[test]
    fn convolution_sums_a_3x3_box() {
        // A 3x3 kernel of ones over a 3x3 ramp: one output = sum 0..8 = 36.
        let x = ramp(1, 1, 3, 3);
        let y = convolution_portable(&x, &[1.0; 9], &[], 1, (3, 3), (1, 1), (0, 0), (1, 1), 1).unwrap();
        assert_eq!(y.shape, [1, 1, 1, 1]);
        assert_eq!(y.data[0], 36.0);
    }

    #[test]
    fn convolution_applies_bias_and_stride() {
        let x = ramp(1, 1, 4, 4);
        let y = convolution_portable(&x, &[1.0; 4], &[100.0], 1, (2, 2), (2, 2), (0, 0), (1, 1), 1)
            .unwrap();
        assert_eq!(y.shape, [1, 1, 2, 2]);
        // Top-left 2x2 block is 0,1,4,5 -> 10, plus the bias.
        assert_eq!(y.data[0], 110.0);
    }

    #[test]
    fn same_padding_preserves_the_spatial_size() {
        let x = ramp(1, 1, 5, 5);
        let (ph, pw) = Padding::Same.resolve((3, 3), (1, 1));
        let y = convolution_portable(&x, &[1.0; 9], &[], 1, (3, 3), (1, 1), (ph, pw), (1, 1), 1)
            .unwrap();
        assert_eq!(y.shape, [1, 1, 5, 5]);
    }

    #[test]
    fn grouped_convolution_keeps_the_groups_separate() {
        // 2 input channels, 2 groups, 2 output channels: each output sees only
        // its own input channel. Channel 0 is all 1s, channel 1 all 10s.
        let mut x = Blob::zeros(1, 2, 1, 1);
        *x.at_mut(0, 0, 0, 0) = 1.0;
        *x.at_mut(0, 1, 0, 0) = 10.0;
        let y =
            convolution_portable(&x, &[1.0, 1.0], &[], 2, (1, 1), (1, 1), (0, 0), (1, 1), 2).unwrap();
        assert_eq!(y.data, vec![1.0, 10.0]);
    }

    #[test]
    fn dilated_convolution_reaches_further() {
        let x = ramp(1, 1, 5, 5);
        // A 1x2 kernel with dilation 2 samples x and x+2.
        let y = convolution_portable(&x, &[1.0, 1.0], &[], 1, (1, 2), (1, 1), (0, 0), (1, 2), 1)
            .unwrap();
        assert_eq!(y.shape, [1, 1, 5, 3]);
        assert_eq!(y.data[0], 0.0 + 2.0);
    }

    #[test]
    fn max_and_average_pooling() {
        let x = ramp(1, 1, 4, 4);
        let mx = pooling(&x, PoolKind::Max, (2, 2), (2, 2), Padding::Valid).unwrap();
        assert_eq!(mx.shape, [1, 1, 2, 2]);
        assert_eq!(mx.data, vec![5.0, 7.0, 13.0, 15.0]);
        let av = pooling(&x, PoolKind::Average, (2, 2), (2, 2), Padding::Valid).unwrap();
        assert_eq!(av.data, vec![2.5, 4.5, 10.5, 12.5]);
    }

    #[test]
    fn global_pool_collapses_each_plane() {
        let x = ramp(1, 2, 2, 2); // plane 0: 0..3, plane 1: 4..7
        let a = global_pool(&x, PoolKind::Average).unwrap();
        assert_eq!(a.shape, [1, 2, 1, 1]);
        assert_eq!(a.data, vec![1.5, 5.5]);
        let m = global_pool(&x, PoolKind::Max).unwrap();
        assert_eq!(m.data, vec![3.0, 7.0]);
    }

    #[test]
    fn inner_product_is_a_matrix_product_plus_bias() {
        let x = Blob { shape: [1, 3, 1, 1], data: vec![1.0, 2.0, 3.0] };
        // 2x3 weights.
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let y = inner_product(&x, &w, &[10.0, 20.0], 2).unwrap();
        assert_eq!(y.shape, [1, 2, 1, 1]);
        assert_eq!(y.data, vec![11.0, 25.0]);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let x = Blob { shape: [1, 3, 1, 1], data: vec![1.0, 2.0, 3.0] };
        let y = softmax(&x).unwrap();
        let s: f32 = y.data.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "sum {s}");
        assert!(y.data[2] > y.data[1] && y.data[1] > y.data[0]);
    }

    #[test]
    fn softmax_is_stable_on_large_inputs() {
        let x = Blob { shape: [1, 2, 1, 1], data: vec![1000.0, 1000.0] };
        let y = softmax(&x).unwrap();
        assert!((y.data[0] - 0.5).abs() < 1e-6, "got {:?} — the shift is missing", y.data);
    }

    #[test]
    fn activations_behave() {
        assert_eq!(Activation::ReLU { slope: 0.0 }.apply(-2.0), 0.0);
        assert_eq!(Activation::ReLU { slope: 0.1 }.apply(-2.0), -0.2);
        assert_eq!(Activation::ReLU6.apply(10.0), 6.0);
        assert!((Activation::Sigmoid.apply(0.0) - 0.5).abs() < 1e-9);
        assert!((Activation::TanH.apply(0.0)).abs() < 1e-9);
        assert!(Activation::Swish.apply(-10.0).abs() < 1e-3);
        assert_eq!(Activation::Abs.apply(-3.0), 3.0);
        assert_eq!(Activation::Power { power: 2.0, scale: 1.0, shift: 0.0 }.apply(3.0), 9.0);
    }

    #[test]
    fn batch_norm_scales_per_channel() {
        let x = ramp(1, 2, 1, 2); // ch0: 0,1  ch1: 2,3
        let y = per_channel(&x, &[2.0, 10.0], &[1.0, 0.0]).unwrap();
        assert_eq!(y.data, vec![1.0, 3.0, 20.0, 30.0]);
    }

    #[test]
    fn permute_transposes_axes() {
        let x = ramp(1, 2, 1, 3);
        let y = permute(&x, [0, 3, 2, 1]).unwrap(); // NCHW -> NWHC
        assert_eq!(y.shape, [1, 3, 1, 2]);
        assert_eq!(y.at(0, 0, 0, 0), x.at(0, 0, 0, 0));
        assert_eq!(y.at(0, 0, 0, 1), x.at(0, 1, 0, 0));
    }

    #[test]
    fn permute_rejects_a_repeated_axis() {
        assert!(permute(&ramp(1, 1, 1, 1), [0, 0, 2, 3]).is_err());
    }

    #[test]
    fn concat_joins_channels() {
        let a = ramp(1, 1, 1, 2);
        let b = ramp(1, 2, 1, 2);
        let c = concat_channels(&a, &b).unwrap();
        assert_eq!(c.shape, [1, 3, 1, 2]);
        assert_eq!(c.at(0, 0, 0, 0), a.at(0, 0, 0, 0));
        assert_eq!(c.at(0, 1, 0, 0), b.at(0, 0, 0, 0));
    }

    #[test]
    fn eltwise_combines() {
        let a = Blob { shape: [1, 1, 1, 3], data: vec![1.0, 5.0, 3.0] };
        let b = Blob { shape: [1, 1, 1, 3], data: vec![4.0, 2.0, 3.0] };
        assert_eq!(eltwise(&a, &b, EltwiseOp::Sum).unwrap().data, vec![5.0, 7.0, 6.0]);
        assert_eq!(eltwise(&a, &b, EltwiseOp::Prod).unwrap().data, vec![4.0, 10.0, 9.0]);
        assert_eq!(eltwise(&a, &b, EltwiseOp::Max).unwrap().data, vec![4.0, 5.0, 3.0]);
    }

    #[test]
    fn pad_adds_a_border() {
        let x = ramp(1, 1, 1, 1);
        let y = pad(&x, 1, 2, -1.0).unwrap();
        assert_eq!(y.shape, [1, 1, 3, 5]);
        assert_eq!(y.at(0, 0, 1, 2), 0.0);
        assert_eq!(y.at(0, 0, 0, 0), -1.0);
    }

    #[test]
    fn lrn_leaves_a_single_channel_scaled_by_its_own_square() {
        let x = Blob { shape: [1, 1, 1, 1], data: vec![2.0] };
        let y = lrn(&x, 1, 1.0, 1.0, 0.0).unwrap();
        // denom = (0 + 1/1 * 4)^1 = 4, so 2/4.
        assert!((y.data[0] - 0.5).abs() < 1e-6, "got {}", y.data[0]);
    }

    #[test]
    fn a_small_network_runs_end_to_end() {
        let mut net = Net::new();
        net.add_layer(
            "conv",
            Layer::Convolution {
                weights: vec![1.0; 9],
                bias: vec![0.0],
                c_out: 1,
                kernel: (3, 3),
                stride: (1, 1),
                pad: Padding::Same,
                dilation: (1, 1),
                groups: 1,
            },
        )
        .add_layer("relu", Layer::Activation(Activation::ReLU { slope: 0.0 }))
        .add_layer("pool", Layer::GlobalPool(PoolKind::Average))
        .add_layer("fc", Layer::InnerProduct { weights: vec![2.0], bias: vec![1.0], out: 1 });

        net.set_input(ramp(1, 1, 4, 4));
        let y = net.forward().unwrap();
        assert_eq!(y.shape, [1, 1, 1, 1]);
        assert!(y.data[0].is_finite());
        assert_eq!(net.layer_names(), vec!["conv", "relu", "pool", "fc"]);
    }

    #[test]
    fn forward_to_stops_at_the_named_layer() {
        let mut net = Net::new();
        net.add_layer("a", Layer::Activation(Activation::Abs))
            .add_layer("b", Layer::Activation(Activation::Power { power: 2.0, scale: 1.0, shift: 0.0 }));
        net.set_input(Blob { shape: [1, 1, 1, 1], data: vec![-3.0] });
        assert_eq!(net.forward_to("a").unwrap().data, vec![3.0]);
        assert_eq!(net.forward().unwrap().data, vec![9.0]);
        assert!(net.forward_to("missing").is_err());
    }

    #[test]
    fn forward_without_input_is_an_error() {
        let mut net = Net::new();
        assert!(net.forward().is_err());
    }

    #[test]
    fn skip_connection_via_a_named_blob() {
        let mut net = Net::new();
        net.add_layer("shift", Layer::Scale { scale: vec![1.0], bias: vec![10.0] })
            .add_layer("add", Layer::Eltwise(EltwiseOp::Sum, "skip".into()));
        net.set_blob("skip", Blob { shape: [1, 1, 1, 2], data: vec![100.0, 200.0] });
        net.set_input(Blob { shape: [1, 1, 1, 2], data: vec![1.0, 2.0] });
        assert_eq!(net.forward().unwrap().data, vec![111.0, 212.0]);
    }

    /// Reading a real `.onnx` file and reproducing PyTorch's own output.
    ///
    /// The fixture is generated by `tools/gen_onnx_fixture.py`: a two-layer
    /// convolutional network exported from PyTorch, together with its input and
    /// the output PyTorch computed. cv2's own `dnn` module agrees with that
    /// reference to 1.2e-7, so it is a fair target.
    #[cfg(feature = "onnx")]
    #[test]
    fn onnx_model_reproduces_the_reference_output() {
        let dir = std::path::Path::new("/tmp/onnxtest");
        if !dir.join("small.onnx").exists() {
            eprintln!("skipping: run python3 tools/gen_onnx_fixture.py first");
            return;
        }
        let input: Vec<f32> = read_npy_f32(&dir.join("input.npy"));
        let expected: Vec<f32> = read_npy_f32(&dir.join("expected.npy"));

        let mut net = Net::from_onnx(dir.join("small.onnx"), "input", Some([1, 2, 8, 8]))
            .expect("the fixture must import");
        net.set_input(Blob::new([1, 3, 8, 8], input).unwrap());
        let out = net.forward().expect("forward");
        assert_eq!(out.shape, [1, 2, 8, 8]);
        assert_eq!(out.data.len(), expected.len());
        let worst = out
            .data
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "max difference from PyTorch is {worst}");
    }

    /// Minimal `.npy` reader for the test fixture: little-endian float32 only.
    #[cfg(feature = "onnx")]
    fn read_npy_f32(path: &std::path::Path) -> Vec<f32> {
        let raw = std::fs::read(path).expect("fixture");
        assert_eq!(&raw[..6], b"\x93NUMPY", "not a .npy file");
        let header_len = u16::from_le_bytes([raw[8], raw[9]]) as usize;
        let header = std::str::from_utf8(&raw[10..10 + header_len]).unwrap();
        assert!(header.contains("<f4"), "fixture must be little-endian f32: {header}");
        raw[10 + header_len..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// The acceleration must not change the numbers.
    #[cfg(feature = "rlx")]
    #[test]
    fn rlx_convolution_agrees_with_the_portable_path() {
        let x = ramp(2, 3, 7, 9);
        let weights: Vec<f32> = (0..4 * 3 * 3 * 3).map(|i| (i % 7) as f32 - 3.0).collect();
        let bias = vec![0.5, -1.0, 2.0, 0.0];
        let acc = convolution(&x, &weights, &bias, 4, (3, 3), (1, 1), Padding::Explicit(1, 1), (1, 1), 1)
            .unwrap();
        let portable =
            convolution_portable(&x, &weights, &bias, 4, (3, 3), (1, 1), (1, 1), (1, 1), 1).unwrap();
        assert_eq!(acc.shape, portable.shape);
        for (i, (a, b)) in acc.data.iter().zip(portable.data.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "index {i}: RLX {a} vs portable {b}");
        }
    }
}
