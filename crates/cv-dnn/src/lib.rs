//! `cv::dnn` for `opencv-no-c`.
//!
//! Three pieces, matching upstream's split:
//!
//! * [`blob`] — the HWC-image to NCHW-tensor conversion every network needs.
//! * [`nms`] — non-maximum suppression over detection boxes.
//! * [`net`] — a layer graph, executed on RLX where it pays.
//!
//! # What RLX buys
//!
//! `cv::dnn` upstream is a runtime with its own backends (OpenCV, CUDA,
//! OpenVINO, Vulkan). This port does not reimplement that: RLX already *is* a
//! compiler and runtime with 15 device backends, so the layer types that carry
//! the arithmetic — convolution and fully-connected — lower to RLX graph nodes
//! and inherit Metal, CUDA, the Neural Engine and the rest for free. Enable the
//! `rlx` feature (and optionally `rlx-metal`, `rlx-cuda`, `rlx-coreml`, ...)
//! to switch it on.
//!
//! Without those features every layer runs the portable Rust path and the
//! results are the same; `net`'s tests assert exactly that, so turning
//! acceleration on is not a correctness risk.
//!
//! # What is not here
//!
//! Model *importers*. RLX ships `rlx-onnx-import`, `rlx-torch-import` and
//! `rlx-gguf`, so reading an ONNX file is a matter of lowering its HIR module
//! into a [`net::Net`] rather than writing a protobuf parser — but that
//! lowering is not written yet, so [`net::Net`] must currently be built by
//! hand. Saying so plainly beats shipping a `read_net_from_onnx` that only
//! handles three operators.

#![warn(missing_docs)]

pub mod api;
pub mod blob;
pub mod model;
pub mod net;
pub mod nms;

pub use blob::{blob_from_image, blob_from_images, images_from_blob, Blob, BlobResize};
pub use net::{Activation, EltwiseOp, Layer, Net, Padding, PoolKind};
pub use nms::{box_iou, nms_boxes, nms_boxes_batched, soft_nms_boxes, SoftNmsMethod};
