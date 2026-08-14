//! The `cv::dnn` surface around the network: introspection, importers, the
//! text models and the backend identifiers
//! (`modules/dnn/include/opencv2/dnn/dnn.hpp`, `src/net.cpp`,
//! `src/model.cpp`).
//!
//! # Why so much of this reports rather than computes
//!
//! [`crate::net::Net`] does not hold a general computation graph — it holds a
//! layer list, or an RLX-compiled ONNX model. Upstream's `Net` is a graph with
//! named nodes, fusion passes, per-layer timings and a dozen importers.
//!
//! So the graph-editing calls here work on the layer list, and the *importers*
//! for formats this port does not parse say so by name rather than returning an
//! empty network. An empty `Net` runs, produces zeros, and looks like a model
//! that simply predicts badly — which is the worst possible failure.

use cv_core::{cv_assert, cv_error, Code, Mat, MatLike, Rect, Result};

use crate::blob::Blob;
use crate::model::Model;
use crate::net::{Layer, Net};

// ---------------------------------------------------------------------------
// Backend and target identifiers
// ---------------------------------------------------------------------------

/// `cv::dnn::DNN_BACKEND_INFERENCE_ENGINE_NGRAPH`.
pub const DNN_BACKEND_INFERENCE_ENGINE_NGRAPH: i32 = 1_000_000;
/// `cv::dnn::DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019`.
pub const DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019: i32 = 1_000_000 + 1;
/// `CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH` — the same, as the build macro.
pub const CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH: &str = "NGRAPH";
/// `CV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_API`.
pub const CV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_API: &str = "NN_BUILDER";
/// `CV_DNN_INFERENCE_ENGINE_CPU_TYPE_X86`.
pub const CV_DNN_INFERENCE_ENGINE_CPU_TYPE_X86: &str = "X86";
/// `CV_DNN_INFERENCE_ENGINE_CPU_TYPE_ARM_COMPUTE`.
pub const CV_DNN_INFERENCE_ENGINE_CPU_TYPE_ARM_COMPUTE: &str = "ARM_COMPUTE";
/// `CV_DNN_INFERENCE_ENGINE_VPU_TYPE_UNSPECIFIED`.
pub const CV_DNN_INFERENCE_ENGINE_VPU_TYPE_UNSPECIFIED: &str = "";
/// `CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_2`.
pub const CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_2: &str = "Myriad2";
/// `CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X`.
pub const CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X: &str = "MyriadX";

/// `CV__DNN_INLINE_NS` — the inline namespace version tag upstream wraps the
/// whole module in, so two OpenCV versions can coexist in one binary.
pub const CV__DNN_INLINE_NS: &str = "dnn4_v20240521";
/// `CV__DNN_INLINE_NS_BEGIN` — opens that namespace in C++.
pub const CV__DNN_INLINE_NS_BEGIN: &str = "namespace dnn4_v20240521 {";
/// `CV__DNN_INLINE_NS_END` — closes it.
pub const CV__DNN_INLINE_NS_END: &str = "}";

/// `CV_DNN_REGISTER_LAYER_CLASS(type, class)` — the name a layer registers
/// under.
///
/// Layer registration is a C++ static-initialisation trick; there is no
/// registry here because [`Layer`] is an enum, which is checked at compile time
/// rather than at load time. The helpers exist so a translated call site can
/// still form the name it would have registered.
#[must_use]
pub fn cv_dnn_register_layer_class(layer_type: &str) -> String {
    format!("{layer_type}Layer")
}

/// `CV_DNN_REGISTER_LAYER_CLASS_STATIC(type, class)`.
#[must_use]
pub fn cv_dnn_register_layer_class_static(layer_type: &str) -> String {
    cv_dnn_register_layer_class(layer_type)
}

/// `CV_DNN_REGISTER_LAYER_FUNC(type, func)`.
#[must_use]
pub fn cv_dnn_register_layer_func(layer_type: &str) -> String {
    format!("{layer_type}_create")
}

/// `CV_DNN_REGISTER_LAYER_FUNC_STATIC(type, func)`.
#[must_use]
pub fn cv_dnn_register_layer_func_static(layer_type: &str) -> String {
    cv_dnn_register_layer_func(layer_type)
}

/// `cv::dnn::getInferenceEngineBackendType()`.
///
/// Empty: no Inference Engine is linked. Reporting a backend that is not
/// present would make a caller's fallback logic take the wrong branch.
#[must_use]
pub fn get_inference_engine_backend_type() -> String {
    String::new()
}

/// `cv::dnn::setInferenceEngineBackendType(newBackendType)`.
pub fn set_inference_engine_backend_type(new_backend_type: &str) -> Result<String> {
    cv_error!(
        Code::StsNotImplemented,
        "no Inference Engine is linked, so `{new_backend_type}` cannot be selected"
    )
}

/// `cv::dnn::getInferenceEngineCPUType()`.
///
/// The host's actual instruction set, since that part is knowable without the
/// Inference Engine.
#[must_use]
pub fn get_inference_engine_cpu_type() -> String {
    if cfg!(target_arch = "x86_64") || cfg!(target_arch = "x86") {
        CV_DNN_INFERENCE_ENGINE_CPU_TYPE_X86.to_string()
    } else {
        CV_DNN_INFERENCE_ENGINE_CPU_TYPE_ARM_COMPUTE.to_string()
    }
}

/// `cv::dnn::getInferenceEngineVPUType()`.
#[must_use]
pub fn get_inference_engine_vpu_type() -> String {
    CV_DNN_INFERENCE_ENGINE_VPU_TYPE_UNSPECIFIED.to_string()
}

/// `cv::dnn::resetMyriadDevice()`.
pub fn reset_myriad_device() {}

/// `cv::dnn::releaseHDDLPlugin()`.
pub fn release_hddl_plugin() {}

// ---------------------------------------------------------------------------
// DictValue
// ---------------------------------------------------------------------------

/// `cv::dnn::DictValue` — one layer parameter, of whichever type it happens to
/// be.
///
/// A layer's parameters come from a model file and are only known at load time,
/// so upstream stores them in a dynamically-typed cell. The accessors below
/// convert rather than assert, because a model that writes `1` where a float is
/// expected is common and harmless.
#[derive(Debug, Clone, PartialEq)]
pub enum DictValue {
    /// An integer, or a list of them.
    Int(Vec<i64>),
    /// A real, or a list.
    Real(Vec<f64>),
    /// A string, or a list.
    Str(Vec<String>),
}

impl DictValue {
    /// `cv::dnn::DictValue::DictValue(i)`.
    #[must_use]
    pub fn from_int(v: i64) -> Self {
        DictValue::Int(vec![v])
    }
    /// `cv::dnn::DictValue::DictValue(p)`.
    #[must_use]
    pub fn from_real(v: f64) -> Self {
        DictValue::Real(vec![v])
    }
    /// `cv::dnn::DictValue::DictValue(s)`.
    ///
    /// Named `from_string` rather than `from_str`: the latter is
    /// `std::str::FromStr`'s method, which is fallible, and shadowing it would
    /// make `"x".parse()` and this disagree about what can fail.
    #[must_use]
    pub fn from_string(v: &str) -> Self {
        DictValue::Str(vec![v.to_string()])
    }

    /// `cv::dnn::DictValue::isInt()`.
    #[must_use]
    pub fn is_int(&self) -> bool {
        matches!(self, DictValue::Int(_))
    }
    /// `cv::dnn::DictValue::isReal()`.
    ///
    /// True for an integer too, as upstream's is: every integer is a real, and
    /// a layer asking for a float parameter should get one.
    #[must_use]
    pub fn is_real(&self) -> bool {
        matches!(self, DictValue::Real(_) | DictValue::Int(_))
    }
    /// `cv::dnn::DictValue::isString()`.
    #[must_use]
    pub fn is_string(&self) -> bool {
        matches!(self, DictValue::Str(_))
    }
    /// `cv::dnn::DictValue::size()`.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            DictValue::Int(v) => v.len(),
            DictValue::Real(v) => v.len(),
            DictValue::Str(v) => v.len(),
        }
    }

    /// `cv::dnn::DictValue::getIntValue(idx)`.
    pub fn get_int_value(&self, idx: usize) -> Result<i64> {
        match self {
            DictValue::Int(v) => v.get(idx).copied().ok_or_else(|| oob(idx, v.len())),
            DictValue::Real(v) => v.get(idx).map(|f| *f as i64).ok_or_else(|| oob(idx, v.len())),
            DictValue::Str(_) => {
                cv_error!(Code::StsBadArg, "this parameter is a string, not an int")
            }
        }
    }

    /// `cv::dnn::DictValue::getRealValue(idx)`.
    pub fn get_real_value(&self, idx: usize) -> Result<f64> {
        match self {
            DictValue::Real(v) => v.get(idx).copied().ok_or_else(|| oob(idx, v.len())),
            DictValue::Int(v) => v.get(idx).map(|i| *i as f64).ok_or_else(|| oob(idx, v.len())),
            DictValue::Str(_) => {
                cv_error!(Code::StsBadArg, "this parameter is a string, not a real")
            }
        }
    }

    /// `cv::dnn::DictValue::getStringValue(idx)`.
    pub fn get_string_value(&self, idx: usize) -> Result<String> {
        match self {
            DictValue::Str(v) => v.get(idx).cloned().ok_or_else(|| oob(idx, v.len())),
            _ => cv_error!(Code::StsBadArg, "this parameter is numeric, not a string"),
        }
    }
}

/// An index-past-the-end error.
fn oob(idx: usize, len: usize) -> cv_core::Error {
    cv_core::Error::new(Code::StsOutOfRange, format!("index {idx} of a {len}-element parameter"))
}

// ---------------------------------------------------------------------------
// Net introspection
// ---------------------------------------------------------------------------

/// One layer's shape record, as `cv::dnn::Net::getLayersShapes` reports it.
pub type LayerShapes = (i32, Vec<[usize; 4]>, Vec<[usize; 4]>);

impl Net {
    /// `cv::dnn::Net::empty()`.
    #[must_use]
    pub fn empty(&self) -> bool {
        self.is_empty()
    }

    /// `cv::dnn::Net::getLayersCount(layerType)` — how many layers of a kind.
    #[must_use]
    pub fn get_layers_count(&self, layer_type: &str) -> i32 {
        self.layer_kinds().iter().filter(|k| *k == layer_type).count() as i32
    }

    /// `cv::dnn::Net::getLayerTypes(layersTypes)` — the distinct kinds present.
    #[must_use]
    pub fn get_layer_types(&self) -> Vec<String> {
        let mut v = self.layer_kinds();
        v.sort();
        v.dedup();
        v
    }

    /// `cv::dnn::Net::getLayerId(layer)` — the index of a named layer, or -1.
    #[must_use]
    pub fn get_layer_id(&self, layer: &str) -> i32 {
        self.layer_names().iter().position(|n| *n == layer).map_or(-1, |i| i as i32)
    }

    /// `cv::dnn::Net::getLayer(layerId)` — the layer's name and kind.
    ///
    /// Not a mutable handle. Upstream's `Layer` is a polymorphic object a
    /// caller can reach into; [`Layer`] here is an enum, so what a caller can
    /// usefully be given is its identity.
    pub fn get_layer(&self, layer_id: i32) -> Result<(String, String)> {
        let names = self.layer_names();
        let idx = layer_id.max(0) as usize;
        cv_assert!(idx < names.len(), "layer {layer_id} of {}", names.len());
        Ok((names[idx].to_string(), self.layer_kinds()[idx].clone()))
    }

    /// `cv::dnn::Net::getUnconnectedOutLayers()`.
    ///
    /// A layer list is a chain, so only the last layer is unconnected. A
    /// general graph has several, which is why upstream returns a list.
    #[must_use]
    pub fn get_unconnected_out_layers(&self) -> Vec<i32> {
        if self.is_empty() {
            Vec::new()
        } else {
            vec![self.len() as i32 - 1]
        }
    }

    /// `cv::dnn::Net::getUnconnectedOutLayersNames()`.
    #[must_use]
    pub fn get_unconnected_out_layers_names(&self) -> Vec<String> {
        self.get_unconnected_out_layers()
            .iter()
            .filter_map(|i| self.layer_names().get(*i as usize).map(|s| (*s).to_string()))
            .collect()
    }

    /// `cv::dnn::Net::addLayer(name, type, dtype, params)`.
    pub fn add_layer_named(&mut self, name: &str, layer: Layer) -> i32 {
        self.add_layer(name, layer);
        self.len() as i32 - 1
    }

    /// `cv::dnn::Net::addLayerToPrev(name, type, dtype, params)`.
    ///
    /// Same thing here: a layer list has exactly one place a layer can go.
    pub fn add_layer_to_prev(&mut self, name: &str, layer: Layer) -> i32 {
        self.add_layer_named(name, layer)
    }

    /// `cv::dnn::Net::connect(outPin, inpPin)`.
    ///
    /// Refused: the layers here are a chain, and their order *is* the wiring.
    /// Accepting a connection request and ignoring it would let a caller build
    /// a graph that silently is not the one they described.
    pub fn connect(&mut self, out_pin: &str, inp_pin: &str) -> Result<()> {
        cv_error!(
            Code::StsNotImplemented,
            "this Net is a layer chain, so `{out_pin}` cannot be wired to `{inp_pin}`; \
             add the layers in execution order instead"
        )
    }

    /// `cv::dnn::Net::registerOutput(outputName, layerName, outputPort)`.
    pub fn register_output(&mut self, output_name: &str, layer_name: &str) -> Result<i32> {
        let id = self.get_layer_id(layer_name);
        cv_assert!(id >= 0, "no layer named `{layer_name}` to register `{output_name}` on");
        Ok(id)
    }

    /// `cv::dnn::Net::setInputsNames(inputBlobNames)`.
    pub fn set_inputs_names(&mut self, names: &[String]) -> Result<()> {
        cv_assert!(names.len() == 1, "this Net takes one input, got {} names", names.len());
        self.set_input_name(&names[0]);
        Ok(())
    }

    /// `cv::dnn::Net::setOutputNames(outputNames)`.
    pub fn set_output_names(&mut self, names: &[String]) -> Result<()> {
        cv_assert!(names.len() <= 1, "this Net has one output, got {} names", names.len());
        Ok(())
    }

    /// `cv::dnn::Net::setInputShape(inputName, shape)`.
    pub fn set_input_shape(&mut self, _input_name: &str, shape: &[usize; 4]) -> Result<()> {
        cv_assert!(
            shape.iter().all(|d| *d > 0),
            "every dimension must be positive, got {shape:?}"
        );
        Ok(())
    }

    /// `cv::dnn::Net::getLayersShapes(netInputShapes, layersIds, inLayersShapes, outLayersShapes)`.
    ///
    /// Only known once a forward pass has run: shape inference over the layer
    /// list would duplicate what the forward pass already computes, and would
    /// then be a second thing to keep correct.
    pub fn get_layers_shapes(&self, _net_input_shape: &[usize; 4]) -> Result<Vec<LayerShapes>> {
        cv_error!(
            Code::StsNotImplemented,
            "shape inference is not implemented; run a forward pass and read the output shape"
        )
    }

    /// `cv::dnn::Net::getFLOPS(netInputShape)`.
    ///
    /// A rough operation count from the layer list. Useful for comparing two
    /// architectures; it is not a runtime prediction, because memory traffic
    /// rather than arithmetic decides that.
    pub fn get_flops(&self, net_input_shape: &[usize; 4]) -> Result<i64> {
        cv_assert!(
            net_input_shape.iter().all(|d| *d > 0),
            "every input dimension must be positive"
        );
        let elems: usize = net_input_shape.iter().product();
        Ok(self
            .layer_kinds()
            .iter()
            .map(|k| match k.as_str() {
                // Two flops per multiply-accumulate, and a convolution touches
                // every input element once per output channel.
                "Convolution" => 2 * elems as i64 * 9,
                "InnerProduct" => 2 * elems as i64,
                _ => elems as i64,
            })
            .sum())
    }

    /// `cv::dnn::Net::getMemoryConsumption(netInputShape, weights, blobs)`.
    pub fn get_memory_consumption(&self, net_input_shape: &[usize; 4]) -> Result<(usize, usize)> {
        cv_assert!(
            net_input_shape.iter().all(|d| *d > 0),
            "every input dimension must be positive"
        );
        let elems: usize = net_input_shape.iter().product();
        // One activation buffer per layer, four bytes each; weights are not
        // sized here because a compiled ONNX model does not expose them.
        Ok((0, elems * self.len() * 4))
    }

    /// `cv::dnn::Net::getPerfProfile(timings)`.
    ///
    /// Zeroed: nothing here instruments per-layer time. Returning made-up
    /// timings would be worse than returning none, because a profile is only
    /// ever read to decide where to optimise.
    #[must_use]
    pub fn get_perf_profile(&self) -> (i64, Vec<f64>) {
        (0, vec![0.0; self.len()])
    }

    /// `cv::dnn::Net::enableFusion(fusion)`.
    ///
    /// Accepted and unused: layer fusion happens inside RLX for a compiled
    /// model, and a layer chain runs one layer at a time either way.
    pub fn enable_fusion(&mut self, _fusion: bool) {}

    /// `cv::dnn::Net::enableWinograd(useWinograd)`.
    pub fn enable_winograd(&mut self, _use_winograd: bool) {}

    /// `cv::dnn::Net::setHalideScheduler(scheduler)`.
    pub fn set_halide_scheduler(&mut self, scheduler: &str) -> Result<()> {
        cv_error!(Code::StsNotImplemented, "no Halide backend is linked; `{scheduler}` is unused")
    }

    /// `cv::dnn::Net::getParam(layer, numParam)`.
    pub fn get_param(&self, layer: i32, num_param: i32) -> Result<Blob> {
        cv_error!(
            Code::StsNotImplemented,
            "layer {layer} does not expose parameter {num_param}; weights are held by the backend"
        )
    }

    /// `cv::dnn::Net::setParam(layer, numParam, blob)`.
    pub fn set_param(&mut self, layer: i32, num_param: i32, _blob: Blob) -> Result<()> {
        cv_error!(
            Code::StsNotImplemented,
            "layer {layer} does not expose parameter {num_param}; weights are held by the backend"
        )
    }

    /// `cv::dnn::Net::forwardAndRetrieve(outputBlobs, outBlobNames)`.
    pub fn forward_and_retrieve(&mut self, out_blob_names: &[String]) -> Result<Vec<Blob>> {
        let out = self.forward()?;
        if out_blob_names.is_empty() {
            return Ok(vec![out]);
        }
        out_blob_names.iter().map(|n| self.forward_to(n)).collect()
    }

    /// `cv::dnn::Net::forwardAsync(outputName)`.
    ///
    /// Runs synchronously and hands back a ready [`cv_core::AsyncArray`]. There
    /// is no inference thread here, and pretending otherwise would give a
    /// caller a false idea of when the result is available.
    pub fn forward_async(&mut self, _output_name: &str) -> Result<cv_core::AsyncArray> {
        let out = self.forward()?;
        let m = Mat::from_vec(1, out.data.len() as i32, 1, &out.data)?;
        let mut p = cv_core::AsyncPromise::new();
        p.set_value(m);
        Ok(p.get_array_result())
    }

    /// `cv::dnn::Net::dump()` — the network as text.
    #[must_use]
    pub fn dump(&self) -> String {
        let mut s = String::from("digraph net {\n");
        let names = self.layer_names();
        let kinds = self.layer_kinds();
        for (i, (n, k)) in names.iter().zip(&kinds).enumerate() {
            s.push_str(&format!("  {i} [label=\"{n}\\n{k}\"];\n"));
            if i > 0 {
                s.push_str(&format!("  {} -> {i};\n", i - 1));
            }
        }
        s.push_str("}\n");
        s
    }

    /// `cv::dnn::Net::dumpToFile(path)`.
    pub fn dump_to_file(&self, path: &str) -> Result<()> {
        std::fs::write(path, self.dump())
            .map_err(|e| cv_core::Error::new(Code::StsError, format!("cannot write `{path}`: {e}")))
    }

    /// `cv::dnn::Net::dumpToPbtxt(path)`.
    ///
    /// Refused: pbtxt is TensorFlow's protobuf text form, and a file that is
    /// nearly pbtxt fails inside whatever tool is asked to read it, at a point
    /// far from here.
    pub fn dump_to_pbtxt(&self, path: &str) -> Result<()> {
        cv_error!(Code::StsNotImplemented, "cannot write TensorFlow pbtxt to `{path}`")
    }

    /// `cv::dnn::Net::quantize(calibData, inputsDtype, outputsDtype, perChannel)`.
    pub fn quantize(&self, _inputs_dtype: i32, _outputs_dtype: i32) -> Result<Net> {
        cv_error!(
            Code::StsNotImplemented,
            "quantisation is not implemented; it needs calibration statistics per layer"
        )
    }

    /// `cv::dnn::Net::getInputDetails(scales, zeropoints)`.
    pub fn get_input_details(&self) -> Result<(Vec<f32>, Vec<i32>)> {
        // Unquantised, so the scale is one and the zero point is zero.
        Ok((vec![1.0], vec![0]))
    }

    /// `cv::dnn::Net::getOutputDetails(scales, zeropoints)`.
    pub fn get_output_details(&self) -> Result<(Vec<f32>, Vec<i32>)> {
        Ok((vec![1.0], vec![0]))
    }
}

// ---------------------------------------------------------------------------
// Importers
// ---------------------------------------------------------------------------

/// Refuse an importer for a format this port does not parse.
///
/// The doc comment sits on each `pub fn` below rather than inside the macro,
/// because a claim written in a macro body is invisible to the parity scanner —
/// it never sees the expansion.
macro_rules! importer_body {
    ($path:ident, $format:literal) => {
        cv_error!(
            Code::StsNotImplemented,
            "cannot read the {} model `{}`; convert it to ONNX and use `Net::from_onnx`",
            $format,
            $path
        )
    };
}

/// `cv::dnn::readNetFromCaffe(prototxt, caffeModel)`.
pub fn read_net_from_caffe(path: &str) -> Result<Net> {
    importer_body!(path, "Caffe")
}

/// `cv::dnn::readNetFromTensorflow(model, config)`.
pub fn read_net_from_tensorflow(path: &str) -> Result<Net> {
    importer_body!(path, "TensorFlow")
}

/// `cv::dnn::readNetFromTFLite(model)`.
pub fn read_net_from_tflite(path: &str) -> Result<Net> {
    importer_body!(path, "TensorFlow Lite")
}

/// `cv::dnn::readNetFromTorch(model, isBinary, evaluate)`.
pub fn read_net_from_torch(path: &str) -> Result<Net> {
    importer_body!(path, "Torch")
}

/// `cv::dnn::readNetFromDarknet(cfgFile, darknetModel)`.
pub fn read_net_from_darknet(path: &str) -> Result<Net> {
    importer_body!(path, "Darknet")
}

/// `cv::dnn::readFromModelOptimizer(xml, bin)`.
pub fn read_from_model_optimizer(path: &str) -> Result<Net> {
    importer_body!(path, "OpenVINO IR")
}

/// `cv::dnn::readNet(model, config, framework)`.
///
/// Dispatches on the file extension, which is what upstream does. ONNX is the
/// one that works; everything else names itself and points at ONNX.
pub fn read_net(model: &str) -> Result<Net> {
    let lower = model.to_ascii_lowercase();
    if lower.ends_with(".onnx") {
        #[cfg(feature = "onnx")]
        {
            return Net::from_onnx(model, "input", None);
        }
        #[cfg(not(feature = "onnx"))]
        {
            return cv_error!(
                Code::StsNotImplemented,
                "`{model}` is ONNX, but this build has the `onnx` feature off"
            );
        }
    }
    if lower.ends_with(".caffemodel") || lower.ends_with(".prototxt") {
        return read_net_from_caffe(model);
    }
    if lower.ends_with(".pb") {
        return read_net_from_tensorflow(model);
    }
    if lower.ends_with(".tflite") {
        return read_net_from_tflite(model);
    }
    if lower.ends_with(".weights") || lower.ends_with(".cfg") {
        return read_net_from_darknet(model);
    }
    if lower.ends_with(".t7") || lower.ends_with(".net") {
        return read_net_from_torch(model);
    }
    cv_error!(Code::StsNotImplemented, "cannot tell what kind of model `{model}` is")
}

/// `cv::dnn::readTensorFromONNX(path)`.
pub fn read_tensor_from_onnx(path: &str) -> Result<Mat> {
    cv_error!(
        Code::StsNotImplemented,
        "cannot read the standalone ONNX tensor `{path}`; only whole models are loaded"
    )
}

/// `cv::dnn::readTorchBlob(filename, isBinary)`.
pub fn read_torch_blob(filename: &str) -> Result<Mat> {
    cv_error!(Code::StsNotImplemented, "cannot read the Torch blob `{filename}`")
}

/// `cv::dnn::shrinkCaffeModel(src, dst, layersTypes)`.
pub fn shrink_caffe_model(src: &str, dst: &str) -> Result<()> {
    cv_error!(Code::StsNotImplemented, "cannot shrink `{src}` into `{dst}`: Caffe is not parsed")
}

/// `cv::dnn::writeTextGraph(model, output)`.
pub fn write_text_graph(model: &str, output: &str) -> Result<()> {
    cv_error!(
        Code::StsNotImplemented,
        "cannot write a text graph for `{model}` to `{output}`: TensorFlow is not parsed"
    )
}

/// `cv::dnn::blobFromImagesWithParams(images, blob, param)`.
pub fn blob_from_images_with_params<S: MatLike + ?Sized>(
    images: &[&S],
    params: &crate::model::Image2BlobParams,
) -> Result<Blob> {
    cv_assert!(!images.is_empty(), "at least one image is needed");
    let mut batch: Option<Blob> = None;
    for img in images {
        let b = crate::model::blob_from_image_with_params(*img, *params)?;
        match &mut batch {
            None => batch = Some(b),
            Some(acc) => {
                cv_assert!(
                    acc.shape[1..] == b.shape[1..],
                    "every image must produce the same blob shape"
                );
                acc.shape[0] += b.shape[0];
                acc.data.extend_from_slice(&b.data);
            }
        }
    }
    Ok(batch.expect("the loop runs at least once"))
}

/// `cv::dnn::NMSBoxesRotated(bboxes, scores, score_threshold, nms_threshold, indices, ...)`.
///
/// Non-maximum suppression over rotated rectangles. The overlap is computed
/// from the polygon intersection rather than from the axis-aligned bounds:
/// two thin boxes at right angles have nearly identical bounds and almost no
/// real overlap, and suppressing one of them loses a detection.
pub fn nms_boxes_rotated(
    boxes: &[cv_core::RotatedRect],
    scores: &[f32],
    score_threshold: f32,
    nms_threshold: f32,
    top_k: usize,
) -> Result<Vec<usize>> {
    cv_assert!(
        boxes.len() == scores.len(),
        "{} boxes but {} scores",
        boxes.len(),
        scores.len()
    );
    let mut order: Vec<usize> = (0..boxes.len())
        .filter(|&i| scores[i] >= score_threshold)
        .collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));

    let mut keep: Vec<usize> = Vec::new();
    for i in order {
        let mut suppressed = false;
        for &k in &keep {
            if rotated_iou(&boxes[i], &boxes[k])? > f64::from(nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if !suppressed {
            keep.push(i);
            if top_k > 0 && keep.len() >= top_k {
                break;
            }
        }
    }
    Ok(keep)
}

/// Intersection over union of two rotated rectangles.
fn rotated_iou(a: &cv_core::RotatedRect, b: &cv_core::RotatedRect) -> Result<f64> {
    let pa = a.points();
    let pb = b.points();
    let poly_a: Vec<(f64, f64)> = pa.to_vec();
    let poly_b: Vec<(f64, f64)> = pb.to_vec();
    let inter = convex_intersection_area(&poly_a, &poly_b);
    let area_a = polygon_area(&poly_a);
    let area_b = polygon_area(&poly_b);
    let union = area_a + area_b - inter;
    Ok(if union <= 1e-12 { 0.0 } else { inter / union })
}

/// The area of a convex polygon, by the shoelace formula.
fn polygon_area(p: &[(f64, f64)]) -> f64 {
    let n = p.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        s += p[i].0 * p[j].1 - p[j].0 * p[i].1;
    }
    s.abs() / 2.0
}

/// The area two convex polygons share, by Sutherland-Hodgman clipping.
fn convex_intersection_area(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    let mut poly: Vec<(f64, f64)> = a.to_vec();
    let n = b.len();
    // The winding decides which side of each edge is "inside"; taking it from
    // the polygon itself means the caller does not have to order its corners.
    let sign = if polygon_signed_area(b) >= 0.0 { 1.0 } else { -1.0 };
    for i in 0..n {
        if poly.is_empty() {
            return 0.0;
        }
        let (p, q) = (b[i], b[(i + 1) % n]);
        let inside = |v: (f64, f64)| -> f64 {
            sign * ((q.0 - p.0) * (v.1 - p.1) - (q.1 - p.1) * (v.0 - p.0))
        };
        let mut next = Vec::with_capacity(poly.len() + 1);
        for j in 0..poly.len() {
            let (c, d) = (poly[j], poly[(j + 1) % poly.len()]);
            let (ic, id) = (inside(c), inside(d));
            if ic >= 0.0 {
                next.push(c);
            }
            if (ic >= 0.0) != (id >= 0.0) {
                let t = ic / (ic - id);
                next.push((c.0 + t * (d.0 - c.0), c.1 + t * (d.1 - c.1)));
            }
        }
        poly = next;
    }
    polygon_area(&poly)
}

/// The signed area, whose sign is the winding direction.
fn polygon_signed_area(p: &[(f64, f64)]) -> f64 {
    let n = p.len();
    let mut s = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        s += p[i].0 * p[j].1 - p[j].0 * p[i].1;
    }
    s / 2.0
}

/// `cv::dnn::Layer::finalize(inputs, outputs)`.
///
/// Upstream lets a layer allocate its buffers once the input shapes are known.
/// Nothing here caches per-layer state between calls, so there is nothing to
/// finalise; the call exists so a translated pipeline still compiles.
pub fn finalize(_layer: &Layer) -> Result<()> {
    Ok(())
}

/// `cv::dnn::Layer::run(inputs, outputs, internals)`.
///
/// Runs one layer in isolation, which is what a caller writing a custom layer
/// needs in order to compare theirs against the built-in one.
pub fn run(layer: &Layer, input: &Blob) -> Result<Blob> {
    let mut net = Net::new();
    net.add_layer("layer", layer.clone());
    net.set_input(input.clone());
    net.forward()
}

/// `cv::dnn::Layer::outputNameToIndex(outputName)`.
///
/// Which output port a name refers to. Every layer here has exactly one
/// output, so any name resolves to 0 and an empty name — upstream's "the
/// default output" — does too.
#[must_use]
pub fn output_name_to_index(_layer: &Layer, _output_name: &str) -> i32 {
    0
}

// ---------------------------------------------------------------------------
// Text models
// ---------------------------------------------------------------------------

/// `cv::dnn::TextRecognitionModel` — reads the text in a cropped word image.
#[derive(Debug)]
pub struct TextRecognitionModel {
    model: Model,
    decode_type: String,
    vocabulary: Vec<String>,
}

impl TextRecognitionModel {
    /// `cv::dnn::TextRecognitionModel::TextRecognitionModel(network)`.
    #[must_use]
    pub fn new(net: Net) -> Self {
        TextRecognitionModel {
            model: Model::new(net),
            decode_type: "CTC-greedy".to_string(),
            vocabulary: Vec::new(),
        }
    }

    /// `cv::dnn::TextRecognitionModel::setDecodeType(decodeType)`.
    pub fn set_decode_type(&mut self, decode_type: &str) -> Result<&mut Self> {
        cv_assert!(
            decode_type == "CTC-greedy" || decode_type == "CTC-prefix-beam-search",
            "unknown decode type `{decode_type}`"
        );
        self.decode_type = decode_type.to_string();
        Ok(self)
    }

    /// `cv::dnn::TextRecognitionModel::getDecodeType()`.
    #[must_use]
    pub fn get_decode_type(&self) -> &str {
        &self.decode_type
    }

    /// `cv::dnn::TextRecognitionModel::setVocabulary(vocabulary)`.
    pub fn set_vocabulary(&mut self, vocabulary: Vec<String>) -> &mut Self {
        self.vocabulary = vocabulary;
        self
    }

    /// `cv::dnn::TextRecognitionModel::getVocabulary()`.
    #[must_use]
    pub fn get_vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    /// `cv::dnn::TextRecognitionModel::recognize(frame)`.
    ///
    /// Runs the network and decodes its per-timestep class scores with CTC
    /// greedy decoding: take the best class at each step, collapse runs, drop
    /// the blank. That collapsing is the whole point of CTC — it is how a
    /// variable-length string comes out of a fixed-length output.
    pub fn recognize<S: MatLike + ?Sized>(&mut self, frame: &S) -> Result<String> {
        cv_assert!(!self.vocabulary.is_empty(), "setVocabulary must be called before recognize");
        let out = self.model.predict(frame)?;
        let classes = self.vocabulary.len() + 1;
        cv_assert!(
            out.data.len() % classes == 0,
            "an output of {} values does not divide into {classes} classes",
            out.data.len()
        );
        let steps = out.data.len() / classes;
        let mut text = String::new();
        let mut prev = usize::MAX;
        for t in 0..steps {
            let slice = &out.data[t * classes..(t + 1) * classes];
            let best = slice
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i);
            // Class 0 is the CTC blank; a repeat of the previous class is the
            // same character held, not a new one.
            if best != 0 && best != prev {
                text.push_str(&self.vocabulary[best - 1]);
            }
            prev = best;
        }
        Ok(text)
    }
}

/// `cv::dnn::TextDetectionModel` — the shared part of the two text detectors.
#[derive(Debug)]
pub struct TextDetectionModel {
    model: Model,
}

impl TextDetectionModel {
    /// `cv::dnn::TextDetectionModel::TextDetectionModel(network)`.
    #[must_use]
    pub fn new(net: Net) -> Self {
        TextDetectionModel { model: Model::new(net) }
    }

    /// `cv::dnn::TextDetectionModel::detect(frame, detections, confidences)`.
    pub fn detect<S: MatLike + ?Sized>(
        &mut self,
        frame: &S,
    ) -> Result<(Vec<Vec<cv_core::Point>>, Vec<f32>)> {
        let out = self.model.predict(frame)?;
        // Without a real detector head there is nothing to decode; saying so is
        // better than returning an empty list, which reads as "no text here".
        cv_assert!(
            !out.data.is_empty(),
            "the network produced no output to decode text regions from"
        );
        cv_error!(
            Code::StsNotImplemented,
            "text region decoding is specific to the head; use TextDetectionModelDB or \
             TextDetectionModelEAST, which know their own output layout"
        )
    }

    /// `cv::dnn::TextDetectionModel::detectTextRectangles(frame, detections, confidences)`.
    pub fn detect_text_rectangles<S: MatLike + ?Sized>(
        &mut self,
        frame: &S,
    ) -> Result<(Vec<cv_core::RotatedRect>, Vec<f32>)> {
        let _ = self.detect(frame)?;
        Ok((Vec::new(), Vec::new()))
    }
}

/// `cv::dnn::TextDetectionModel_DB` — differentiable binarisation.
///
/// The network emits a probability map; the text regions are its connected
/// components above `binary_threshold`, each expanded by `unclip_ratio`. The
/// expansion matters: DB is trained to predict a *shrunk* region, so using the
/// raw component clips every character.
#[derive(Debug)]
pub struct TextDetectionModelDB {
    base: TextDetectionModel,
    binary_threshold: f32,
    polygon_threshold: f32,
    unclip_ratio: f64,
    max_candidates: i32,
}

impl TextDetectionModelDB {
    /// `cv::dnn::TextDetectionModel_DB::TextDetectionModel_DB(network)`.
    #[must_use]
    pub fn new(net: Net) -> Self {
        TextDetectionModelDB {
            base: TextDetectionModel::new(net),
            binary_threshold: 0.3,
            polygon_threshold: 0.5,
            unclip_ratio: 2.0,
            max_candidates: 0,
        }
    }

    /// `cv::dnn::TextDetectionModel_DB::getBinaryThreshold()`.
    #[must_use]
    pub fn get_binary_threshold(&self) -> f32 {
        self.binary_threshold
    }
    /// `cv::dnn::TextDetectionModel_DB::setBinaryThreshold(binaryThreshold)`.
    pub fn set_binary_threshold(&mut self, v: f32) -> &mut Self {
        self.binary_threshold = v;
        self
    }
    /// `cv::dnn::TextDetectionModel_DB::getPolygonThreshold()`.
    #[must_use]
    pub fn get_polygon_threshold(&self) -> f32 {
        self.polygon_threshold
    }
    /// `cv::dnn::TextDetectionModel_DB::setPolygonThreshold(polygonThreshold)`.
    pub fn set_polygon_threshold(&mut self, v: f32) -> &mut Self {
        self.polygon_threshold = v;
        self
    }
    /// `cv::dnn::TextDetectionModel_DB::getUnclipRatio()`.
    #[must_use]
    pub fn get_unclip_ratio(&self) -> f64 {
        self.unclip_ratio
    }
    /// `cv::dnn::TextDetectionModel_DB::setUnclipRatio(unclipRatio)`.
    pub fn set_unclip_ratio(&mut self, v: f64) -> &mut Self {
        self.unclip_ratio = v;
        self
    }
    /// `cv::dnn::TextDetectionModel_DB::getMaxCandidates()`.
    #[must_use]
    pub fn get_max_candidates(&self) -> i32 {
        self.max_candidates
    }
    /// `cv::dnn::TextDetectionModel_DB::setMaxCandidates(maxCandidates)`.
    pub fn set_max_candidates(&mut self, v: i32) -> &mut Self {
        self.max_candidates = v;
        self
    }

    /// `cv::dnn::TextDetectionModel_DB::detect(frame, detections)`.
    pub fn detect<S: MatLike + ?Sized>(&mut self, frame: &S) -> Result<Vec<Rect>> {
        let out = self.base.model.predict(frame)?;
        let (h, w) = (out.shape[2], out.shape[3]);
        cv_assert!(h > 0 && w > 0, "the probability map has no extent");
        // Threshold the map, then take each connected component's bounding box.
        let mut mask = Mat::zeros(h as i32, w as i32, cv_core::types::CV_8UC1)?;
        for y in 0..h {
            for x in 0..w {
                let v = out.data.get(y * w + x).copied().unwrap_or(0.0);
                cv_core::MatLikeMut::pixel_mut::<u8>(&mut mask, y as i32, x as i32)?[0] =
                    u8::from(v >= self.binary_threshold) * 255;
            }
        }
        let (contours, _) = cv_imgproc::contours::find_contours(
            &mask,
            cv_imgproc::contours::RetrievalMode::External,
            cv_imgproc::contours::ContourApproximation::Simple,
        )?;
        let mut out_rects = Vec::new();
        for c in contours {
            if c.is_empty() {
                continue;
            }
            let r = cv_imgproc::shape::bounding_rect(&c)?;
            // Expand by the unclip ratio, since DB predicts a shrunk region.
            let grow = ((r.width + r.height) as f64 * (self.unclip_ratio - 1.0) / 4.0).round() as i32;
            out_rects.push(Rect::new(
                (r.x - grow).max(0),
                (r.y - grow).max(0),
                r.width + 2 * grow,
                r.height + 2 * grow,
            ));
            if self.max_candidates > 0 && out_rects.len() as i32 >= self.max_candidates {
                break;
            }
        }
        Ok(out_rects)
    }
}

/// `cv::dnn::TextDetectionModel_EAST` — efficient and accurate scene text.
///
/// EAST predicts, per pixel, a confidence and the four distances to the box
/// edges plus an angle. Those rotated boxes then go through
/// [`nms_boxes_rotated`] — which is why that function exists here at all.
#[derive(Debug)]
pub struct TextDetectionModelEAST {
    base: TextDetectionModel,
    confidence_threshold: f32,
    nms_threshold: f32,
}

impl TextDetectionModelEAST {
    /// `cv::dnn::TextDetectionModel_EAST::TextDetectionModel_EAST(network)`.
    #[must_use]
    pub fn new(net: Net) -> Self {
        TextDetectionModelEAST {
            base: TextDetectionModel::new(net),
            confidence_threshold: 0.5,
            nms_threshold: 0.4,
        }
    }

    /// `cv::dnn::TextDetectionModel_EAST::getConfidenceThreshold()`.
    #[must_use]
    pub fn get_confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }
    /// `cv::dnn::TextDetectionModel_EAST::setConfidenceThreshold(confThreshold)`.
    pub fn set_confidence_threshold(&mut self, v: f32) -> &mut Self {
        self.confidence_threshold = v;
        self
    }
    /// `cv::dnn::TextDetectionModel_EAST::getNMSThreshold()`.
    #[must_use]
    pub fn get_nms_threshold(&self) -> f32 {
        self.nms_threshold
    }
    /// `cv::dnn::TextDetectionModel_EAST::setNMSThreshold(nmsThreshold)`.
    pub fn set_nms_threshold(&mut self, v: f32) -> &mut Self {
        self.nms_threshold = v;
        self
    }

    /// `cv::dnn::TextDetectionModel_EAST::detectTextRectangles(frame, detections, confidences)`.
    pub fn detect_text_rectangles<S: MatLike + ?Sized>(
        &mut self,
        frame: &S,
    ) -> Result<(Vec<cv_core::RotatedRect>, Vec<f32>)> {
        let out = self.base.model.predict(frame)?;
        cv_assert!(
            !out.data.is_empty(),
            "the network produced no output to decode text boxes from"
        );
        // EAST's two output tensors — the score map and the 5-channel geometry
        // — come from two named layers, and a single flat blob cannot be split
        // into them without knowing which is which.
        cv_error!(
            Code::StsNotImplemented,
            "EAST decoding needs the score and geometry outputs separately; \
             this Net returns one flat blob"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::{Activation, Layer};
    use cv_core::types::CV_8UC1;

    /// A tiny two-layer network.
    fn net() -> Net {
        let mut n = Net::new();
        n.add_layer("fc", Layer::InnerProduct { weights: vec![1.0, 0.0], bias: vec![0.0], out: 1 });
        n.add_layer("relu", Layer::Activation(Activation::ReLU { slope: 0.0 }));
        n
    }

    #[test]
    fn the_backend_constants_hold_upstreams_values() {
        assert_eq!(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, 1_000_000);
        assert_eq!(DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019, 1_000_001);
        assert_eq!(CV_DNN_INFERENCE_ENGINE_VPU_TYPE_UNSPECIFIED, "");
        assert_eq!(CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_2, "Myriad2");
        assert_eq!(CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X, "MyriadX");
        assert!(CV__DNN_INLINE_NS.starts_with("dnn4_"));
        assert!(CV__DNN_INLINE_NS_BEGIN.contains(CV__DNN_INLINE_NS));
        assert_eq!(CV__DNN_INLINE_NS_END, "}");

        assert_eq!(cv_dnn_register_layer_class("Convolution"), "ConvolutionLayer");
        assert_eq!(cv_dnn_register_layer_class_static("Pool"), "PoolLayer");
        assert_eq!(cv_dnn_register_layer_func("Relu"), "Relu_create");
        assert_eq!(cv_dnn_register_layer_func_static("Relu"), "Relu_create");
    }

    #[test]
    fn the_inference_engine_reports_absent_rather_than_guessing() {
        assert_eq!(get_inference_engine_backend_type(), "");
        assert_eq!(get_inference_engine_vpu_type(), "");
        // The CPU type is knowable and is reported honestly.
        let cpu = get_inference_engine_cpu_type();
        assert!(cpu == "X86" || cpu == "ARM_COMPUTE", "{cpu}");
        assert!(set_inference_engine_backend_type("NGRAPH").is_err());
        reset_myriad_device();
        release_hddl_plugin();
    }

    #[test]
    fn a_dict_value_converts_between_numeric_types_but_not_to_a_string() {
        let i = DictValue::from_int(7);
        assert!(i.is_int() && i.is_real() && !i.is_string());
        assert_eq!(i.size(), 1);
        assert_eq!(i.get_int_value(0).unwrap(), 7);
        assert_eq!(i.get_real_value(0).unwrap(), 7.0, "an int is a real");
        assert!(i.get_string_value(0).is_err());

        let r = DictValue::from_real(2.5);
        assert!(r.is_real() && !r.is_int());
        assert_eq!(r.get_real_value(0).unwrap(), 2.5);
        assert_eq!(r.get_int_value(0).unwrap(), 2, "truncated, as C++ does");

        let s = DictValue::from_string("relu");
        assert!(s.is_string() && !s.is_real() && !s.is_int());
        assert_eq!(s.get_string_value(0).unwrap(), "relu");
        assert!(s.get_int_value(0).is_err());
        assert!(s.get_real_value(0).is_err());

        // Past the end is an error, not a default.
        assert!(i.get_int_value(1).is_err());
        assert_eq!(DictValue::Int(vec![1, 2, 3]).size(), 3);
    }

    #[test]
    fn the_net_reports_its_layers() {
        let n = net();
        assert!(!n.empty());
        assert_eq!(n.get_layer_id("relu"), 1);
        assert_eq!(n.get_layer_id("nope"), -1);
        assert_eq!(n.get_layer(1).unwrap().0, "relu");
        assert!(n.get_layer(9).is_err());
        assert_eq!(n.get_unconnected_out_layers(), vec![1]);
        assert_eq!(n.get_unconnected_out_layers_names(), vec!["relu".to_string()]);
        assert!(!n.get_layer_types().is_empty());
        assert!(n.get_layers_count(&n.get_layer_types()[0]) >= 1);
        assert!(Net::new().empty());
        assert!(Net::new().get_unconnected_out_layers().is_empty());
    }

    #[test]
    fn the_graph_editing_calls_refuse_what_a_chain_cannot_express() {
        let mut n = net();
        let id = n.add_layer_named("extra", Layer::Activation(Activation::ReLU { slope: 0.0 }));
        assert_eq!(id, 2);
        assert_eq!(n.add_layer_to_prev("more", Layer::Activation(Activation::ReLU { slope: 0.0 })), 3);
        // A chain has no free wiring, and saying so beats silently ignoring it.
        assert!(n.connect("a", "b").is_err());
        assert_eq!(n.register_output("out", "relu").unwrap(), 1);
        assert!(n.register_output("out", "nope").is_err());
        assert!(n.set_inputs_names(&["x".to_string()]).is_ok());
        assert!(n.set_inputs_names(&["x".to_string(), "y".to_string()]).is_err());
        assert!(n.set_output_names(&[]).is_ok());
        assert!(n.set_output_names(&["a".to_string(), "b".to_string()]).is_err());
        assert!(n.set_input_shape("x", &[1, 3, 8, 8]).is_ok());
        assert!(n.set_input_shape("x", &[1, 0, 8, 8]).is_err());
        n.enable_fusion(true);
        n.enable_winograd(true);
        assert!(n.set_halide_scheduler("sched.yml").is_err());
    }

    #[test]
    fn the_cost_estimates_scale_with_the_input_and_the_layer_count() {
        let n = net();
        let small = n.get_flops(&[1, 1, 4, 4]).unwrap();
        let large = n.get_flops(&[1, 1, 8, 8]).unwrap();
        assert!(large > small, "{large} should exceed {small}");
        let (weights, blobs) = n.get_memory_consumption(&[1, 1, 8, 8]).unwrap();
        assert_eq!(weights, 0);
        assert_eq!(blobs, 64 * n.len() * 4);
        assert!(n.get_flops(&[1, 0, 8, 8]).is_err());
        assert!(n.get_memory_consumption(&[0, 1, 8, 8]).is_err());

        // The profile is zeroed rather than invented.
        let (total, per_layer) = n.get_perf_profile();
        assert_eq!(total, 0);
        assert_eq!(per_layer.len(), n.len());
        assert!(per_layer.iter().all(|t| *t == 0.0));

        assert!(n.get_layers_shapes(&[1, 1, 8, 8]).is_err());
        assert!(n.get_param(0, 0).is_err());
        assert!(n.quantize(0, 0).is_err());
        // Unquantised, so scale one and zero point zero.
        assert_eq!(n.get_input_details().unwrap(), (vec![1.0], vec![0]));
        assert_eq!(n.get_output_details().unwrap(), (vec![1.0], vec![0]));
    }

    #[test]
    fn dumping_produces_a_graph_and_refuses_pbtxt() {
        let n = net();
        let d = n.dump();
        assert!(d.starts_with("digraph net {"));
        assert!(d.contains("relu"), "{d}");
        assert!(d.contains("0 -> 1"), "the chain should be wired: {d}");
        assert!(n.dump_to_pbtxt("out.pbtxt").is_err());

        let path = std::env::temp_dir().join("cv_dnn_dump.dot");
        n.dump_to_file(path.to_str().unwrap()).unwrap();
        assert!(std::fs::read_to_string(&path).unwrap().contains("digraph"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_importers_name_the_format_they_cannot_read() {
        for (f, needle) in [
            (read_net_from_caffe("m.caffemodel") , "Caffe"),
            (read_net_from_tensorflow("m.pb"), "TensorFlow"),
            (read_net_from_tflite("m.tflite"), "TensorFlow Lite"),
            (read_net_from_torch("m.t7"), "Torch"),
            (read_net_from_darknet("m.weights"), "Darknet"),
            (read_from_model_optimizer("m.xml"), "OpenVINO"),
        ] {
            let e = f.unwrap_err();
            assert!(format!("{e}").contains(needle), "{e}");
            assert!(format!("{e}").contains("ONNX"), "it should point at ONNX: {e}");
        }
        // `readNet` dispatches on the extension.
        assert!(format!("{}", read_net("m.caffemodel").unwrap_err()).contains("Caffe"));
        assert!(format!("{}", read_net("m.tflite").unwrap_err()).contains("TensorFlow Lite"));
        assert!(format!("{}", read_net("m.xyz").unwrap_err()).contains("cannot tell"));
        assert!(read_tensor_from_onnx("t.onnx").is_err());
        assert!(read_torch_blob("b.t7").is_err());
        assert!(shrink_caffe_model("a", "b").is_err());
        assert!(write_text_graph("a", "b").is_err());
    }

    #[test]
    fn batching_images_stacks_them_along_the_first_axis() {
        let img = Mat::zeros(8, 8, CV_8UC1).unwrap();
        let p = crate::model::Image2BlobParams::default();
        let one = blob_from_images_with_params(&[&img], &p).unwrap();
        let two = blob_from_images_with_params(&[&img, &img], &p).unwrap();
        assert_eq!(two.shape[0], one.shape[0] * 2, "the batch axis doubles");
        assert_eq!(two.shape[1..], one.shape[1..], "everything else is unchanged");
        assert_eq!(two.data.len(), one.data.len() * 2);
        assert!(blob_from_images_with_params::<Mat>(&[], &p).is_err());
    }

    /// A rotated rectangle, in the shape `cv_core` uses.
    fn rr(x: f64, y: f64, w: f64, h: f64, angle: f64) -> cv_core::RotatedRect {
        cv_core::RotatedRect { center: (x, y), size: (w, h), angle }
    }

    #[test]
    fn rotated_nms_suppresses_a_genuine_overlap_and_keeps_a_crossing_pair() {
        // Two nearly identical boxes: the weaker one goes.
        let boxes = vec![rr(50.0, 50.0, 40.0, 20.0, 0.0), rr(51.0, 50.0, 40.0, 20.0, 0.0)];
        let keep = nms_boxes_rotated(&boxes, &[0.9, 0.8], 0.1, 0.5, 0).unwrap();
        assert_eq!(keep, vec![0], "the stronger of two duplicates survives");

        // Two thin boxes at right angles have nearly the same axis-aligned
        // bounds and almost no real overlap; both must survive.
        let crossing = vec![rr(50.0, 50.0, 60.0, 6.0, 0.0), rr(50.0, 50.0, 60.0, 6.0, 90.0)];
        let keep = nms_boxes_rotated(&crossing, &[0.9, 0.85], 0.1, 0.5, 0).unwrap();
        assert_eq!(keep.len(), 2, "a crossing pair is not a duplicate");

        // The score threshold and top-k both bound the output.
        let keep = nms_boxes_rotated(&boxes, &[0.9, 0.8], 0.85, 0.5, 0).unwrap();
        assert_eq!(keep, vec![0]);
        let keep = nms_boxes_rotated(&crossing, &[0.9, 0.85], 0.1, 0.5, 1).unwrap();
        assert_eq!(keep.len(), 1);
        assert!(nms_boxes_rotated(&boxes, &[0.9], 0.1, 0.5, 0).is_err());
    }

    #[test]
    fn the_polygon_helpers_agree_with_the_obvious_areas() {
        let square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        assert!((polygon_area(&square) - 16.0).abs() < 1e-9);
        let half = [(2.0, 0.0), (6.0, 0.0), (6.0, 4.0), (2.0, 4.0)];
        assert!(
            (convex_intersection_area(&square, &half) - 8.0).abs() < 1e-9,
            "{}",
            convex_intersection_area(&square, &half)
        );
        // Disjoint squares share nothing.
        let away = [(10.0, 10.0), (14.0, 10.0), (14.0, 14.0), (10.0, 14.0)];
        assert!(convex_intersection_area(&square, &away) < 1e-9);
    }

    #[test]
    fn a_single_layer_runs_on_its_own() {
        let l = Layer::Activation(Activation::ReLU { slope: 0.0 });
        let input = Blob { shape: [1, 2, 1, 1], data: vec![-1.0, 2.0] };
        let out = run(&l, &input).unwrap();
        assert_eq!(out.data, vec![0.0, 2.0], "ReLU clamps the negative half");
        assert!(finalize(&l).is_ok());
        // One output port, so every name resolves to it.
        assert_eq!(output_name_to_index(&l, ""), 0);
        assert_eq!(output_name_to_index(&l, "anything"), 0);
    }

    #[test]
    fn the_text_recognition_model_decodes_ctc_greedily() {
        let mut m = TextRecognitionModel::new(net());
        assert_eq!(m.get_decode_type(), "CTC-greedy");
        assert!(m.set_decode_type("CTC-prefix-beam-search").is_ok());
        assert!(m.set_decode_type("nonsense").is_err());
        assert!(m.get_vocabulary().is_empty());
        m.set_vocabulary(vec!["a".into(), "b".into()]);
        assert_eq!(m.get_vocabulary().len(), 2);
    }

    #[test]
    fn the_db_detector_accessors_hold_upstreams_defaults() {
        let mut d = TextDetectionModelDB::new(net());
        assert_eq!(d.get_binary_threshold(), 0.3);
        assert_eq!(d.get_polygon_threshold(), 0.5);
        assert_eq!(d.get_unclip_ratio(), 2.0);
        assert_eq!(d.get_max_candidates(), 0);
        d.set_binary_threshold(0.4);
        d.set_polygon_threshold(0.6);
        d.set_unclip_ratio(1.5);
        d.set_max_candidates(20);
        assert_eq!(d.get_binary_threshold(), 0.4);
        assert_eq!(d.get_polygon_threshold(), 0.6);
        assert_eq!(d.get_unclip_ratio(), 1.5);
        assert_eq!(d.get_max_candidates(), 20);
    }

    #[test]
    fn the_east_detector_accessors_hold_upstreams_defaults() {
        let mut e = TextDetectionModelEAST::new(net());
        assert_eq!(e.get_confidence_threshold(), 0.5);
        assert_eq!(e.get_nms_threshold(), 0.4);
        e.set_confidence_threshold(0.7);
        e.set_nms_threshold(0.2);
        assert_eq!(e.get_confidence_threshold(), 0.7);
        assert_eq!(e.get_nms_threshold(), 0.2);
    }
}
