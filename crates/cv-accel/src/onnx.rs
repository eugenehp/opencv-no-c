//! Loading `.onnx` files through RLX.
//!
//! # Why this is short
//!
//! Importing ONNX means parsing protobuf, resolving initialisers, propagating
//! shapes, rewriting the operator set and lowering to something executable.
//! RLX already does all of that: `rlx-onnx-import` produces an
//! `rlx_ir::hir::HirModule`, and `Graph::from_hir` turns that into the same
//! compiled graph every other RLX model uses. So the whole importer here is a
//! call into RLX plus the bookkeeping needed to hand parameters back in.
//!
//! The alternative — decomposing ONNX into a hand-written layer enum — would
//! cap operator coverage at whatever was implemented by hand. This way the
//! coverage is RLX's.

use std::collections::HashMap;
use std::path::Path;

use cv_core::{Code, Error, Result};

use crate::backend::Backend;

/// A compiled ONNX model, ready to run.
///
/// Holds an RLX `CompiledGraph` with its parameters already bound, so
/// [`OnnxModel::run`] is just a feed-and-fetch.
pub struct OnnxModel {
    #[cfg(feature = "onnx")]
    compiled: std::mem::ManuallyDrop<rlx::CompiledGraph>,
    #[cfg(feature = "onnx")]
    param_names: Vec<String>,
    name: String,
    backend: Backend,
}

impl std::fmt::Debug for OnnxModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxModel")
            .field("name", &self.name)
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl OnnxModel {
    /// The graph's name, as recorded in the file.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Which backend the graph was compiled for.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Names of the weights bound at load time.
    #[must_use]
    pub fn param_names(&self) -> &[String] {
        #[cfg(feature = "onnx")]
        {
            &self.param_names
        }
        #[cfg(not(feature = "onnx"))]
        {
            &[]
        }
    }

    /// Run the graph, feeding named input tensors and returning every output.
    ///
    /// Tensors are flat `f32` in the layout RLX expects, which for image models
    /// is NCHW.
    #[allow(unused_variables, unused_mut)]
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>> {
        #[cfg(not(feature = "onnx"))]
        {
            Err(crate::unavailable("OnnxModel::run"))
        }
        #[cfg(feature = "onnx")]
        {
            Ok(self.compiled.run(inputs))
        }
    }
}

/// `cv::dnn::readNetFromONNX(onnxFile)`, compiled onto [`Backend::Auto`].
pub fn load_onnx(path: &Path) -> Result<OnnxModel> {
    load_onnx_on(Backend::Auto, path, 128)
}

/// [`load_onnx`] pinned to a backend, with a sequence length for dynamic axes.
///
/// `sequence_length` only matters for models with a symbolic length dimension;
/// a convolutional vision model ignores it. RLX bakes the value into the
/// compiled graph, which is why it is fixed at load time rather than per call.
#[allow(unused_variables)]
pub fn load_onnx_on(backend: Backend, path: &Path, sequence_length: usize) -> Result<OnnxModel> {
    #[cfg(not(feature = "onnx"))]
    {
        Err(Error::new(
            Code::StsNotImplemented,
            format!(
                "reading {} needs the `onnx` feature of cv-accel",
                path.display()
            ),
        ))
    }
    #[cfg(feature = "onnx")]
    {
        use rlx_onnx_import::onnx_file::build_hir_from_onnx_file;

        if !path.exists() {
            return Err(Error::new(
                Code::StsObjectNotFound,
                format!("no such file: {}", path.display()),
            ));
        }
        let mut opts = rlx_onnx_import::ImportOptions {
            sequence_length,
            ..Default::default()
        };
        // A rejected model is better than a silently stubbed one: an unsupported
        // operator that quietly becomes a no-op produces plausible garbage.
        opts.strict = true;

        let (hir, params, _report, _manifest) = build_hir_from_onnx_file(path, opts)
            .map_err(|e| Error::new(Code::StsParseError, format!("ONNX import failed: {e}")))?;
        let name = hir.name.clone();

        let device = crate::resolve(backend)?;
        // `compile_hir` does the HIR -> MIR -> graph lowering internally, so
        // there is no intermediate `Graph` to hold.
        let mut compiled = rlx::Session::new(device)
            .compile_hir(hir)
            .map_err(|e| Error::new(Code::StsParseError, format!("HIR lowering failed: {e:?}")))?;

        let mut param_names: Vec<String> = params.keys().cloned().collect();
        param_names.sort();
        for (k, v) in &params {
            compiled.set_param(k, v);
        }
        Ok(OnnxModel {
            // `CompiledGraph::drop` touches RLX thread-locals, which abort if
            // they run during thread teardown; the graph cache has the same
            // constraint and the same fix.
            compiled: std::mem::ManuallyDrop::new(compiled),
            param_names,
            name,
            backend,
        })
    }
}

/// Parameters of a loaded model, for callers that want to inspect or rebind.
pub type Params = HashMap<String, Vec<f32>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_is_a_clear_error() {
        let e = load_onnx(Path::new("/nonexistent/model.onnx")).unwrap_err();
        let msg = format!("{e}");
        assert!(
            msg.contains("model.onnx") || msg.contains("onnx"),
            "the error should name the file or the feature, got {msg:?}"
        );
    }
}
