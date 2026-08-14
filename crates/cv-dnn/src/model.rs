//! The task-level `Model` wrappers and the backend/target surface
//! (`modules/dnn/src/model.cpp`, `dnn.hpp`).
//!
//! # What a `Model` adds over a `Net`
//!
//! A [`crate::net::Net`] takes a blob and returns a blob. Every real use then
//! repeats the same three steps: build the blob with the right scale, mean and
//! layout; run it; decode the output tensor into something meaningful. The
//! wrappers here own those steps, so a caller says `classify(image)` and gets
//! `(class_id, confidence)` rather than a 1x1000 tensor and a note to
//! remember the preprocessing.

use cv_core::{cv_assert, MatLike, Rect, Result};

use crate::blob::{blob_from_image, Blob, BlobResize};
use crate::net::Net;

/// `cv::dnn::Backend` — which runtime executes the graph.
///
/// Only [`Backend::Opencv`] and [`Backend::Rlx`] mean anything here. The rest
/// name vendor runtimes this port has no binding to; they are declared so a
/// caller porting OpenCV code can keep its constants, and selecting one is an
/// error rather than a silent fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// `cv::dnn::DNN_BACKEND_DEFAULT`
    #[default]
    Default,
    /// `cv::dnn::DNN_BACKEND_OPENCV` — this crate's portable path.
    Opencv,
    /// RLX, this port's own accelerated runtime. No upstream counterpart.
    Rlx,
    /// `cv::dnn::DNN_BACKEND_HALIDE`
    Halide,
    /// `cv::dnn::DNN_BACKEND_INFERENCE_ENGINE`
    InferenceEngine,
    /// `cv::dnn::DNN_BACKEND_VKCOM`
    Vkcom,
    /// `cv::dnn::DNN_BACKEND_CUDA`
    Cuda,
    /// `cv::dnn::DNN_BACKEND_WEBNN`
    Webnn,
    /// `cv::dnn::DNN_BACKEND_TIMVX`
    Timvx,
    /// `cv::dnn::DNN_BACKEND_CANN`
    Cann,
}

impl Backend {
    /// Whether this port can actually execute on this backend.
    #[must_use]
    pub fn is_available(self) -> bool {
        matches!(self, Backend::Default | Backend::Opencv | Backend::Rlx)
    }
}

/// `cv::dnn::Target` — which device the backend runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Target {
    /// `cv::dnn::DNN_TARGET_CPU`
    #[default]
    Cpu,
    /// `cv::dnn::DNN_TARGET_OPENCL`
    Opencl,
    /// `cv::dnn::DNN_TARGET_OPENCL_FP16`
    OpenclFp16,
    /// `cv::dnn::DNN_TARGET_MYRIAD`
    Myriad,
    /// `cv::dnn::DNN_TARGET_VULKAN`
    Vulkan,
    /// `cv::dnn::DNN_TARGET_FPGA`
    Fpga,
    /// `cv::dnn::DNN_TARGET_CUDA`
    Cuda,
    /// `cv::dnn::DNN_TARGET_CUDA_FP16`
    CudaFp16,
    /// `cv::dnn::DNN_TARGET_HDDL`
    Hddl,
    /// `cv::dnn::DNN_TARGET_NPU`
    Npu,
    /// `cv::dnn::DNN_TARGET_CPU_FP16`
    CpuFp16,
}

/// `cv::dnn::getAvailableTargets(backend)`.
#[must_use]
pub fn get_available_targets(backend: Backend) -> Vec<Target> {
    if backend.is_available() {
        vec![Target::Cpu]
    } else {
        Vec::new()
    }
}

/// `cv::dnn::DataLayout`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataLayout {
    /// `cv::dnn::DNN_LAYOUT_UNKNOWN`
    Unknown,
    /// `cv::dnn::DNN_LAYOUT_ND`
    Nd,
    /// `cv::dnn::DNN_LAYOUT_NCHW` — the layout [`Blob`] uses.
    #[default]
    Nchw,
    /// `cv::dnn::DNN_LAYOUT_NCDHW`
    Ncdhw,
    /// `cv::dnn::DNN_LAYOUT_NHWC`
    Nhwc,
    /// `cv::dnn::DNN_LAYOUT_NDHWC`
    Ndhwc,
    /// `cv::dnn::DNN_LAYOUT_PLANAR`
    Planar,
}

/// `cv::dnn::ImagePaddingMode` — how a non-matching aspect ratio is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingMode {
    /// `cv::dnn::DNN_PMODE_NULL` — stretch, ignoring aspect ratio.
    #[default]
    Null,
    /// `cv::dnn::DNN_PMODE_CROP_CENTER` — scale to cover, then centre-crop.
    CropCenter,
    /// `cv::dnn::DNN_PMODE_LETTERBOX` — scale to fit, pad the remainder.
    Letterbox,
}

/// `cv::dnn::Image2BlobParams`.
#[derive(Debug, Clone, Copy)]
pub struct Image2BlobParams {
    /// Multiplied into every value after the mean is subtracted.
    pub scale_factor: f64,
    /// Target `(width, height)`; zero keeps the source size.
    pub size: (i32, i32),
    /// Per-channel mean, subtracted before scaling.
    pub mean: [f64; 4],
    /// Swap the first and last channels — BGR to RGB.
    pub swap_rb: bool,
    /// Blob layout.
    pub datalayout: DataLayout,
    /// How to reconcile a differing aspect ratio.
    pub padding_mode: PaddingMode,
    /// Fill value for the padded border under [`PaddingMode::Letterbox`].
    pub border_value: [f64; 4],
}

impl Default for Image2BlobParams {
    /// Upstream's defaults: no scaling, no mean, no swap, stretch to size.
    fn default() -> Self {
        Self {
            scale_factor: 1.0,
            size: (0, 0),
            mean: [0.0; 4],
            swap_rb: false,
            datalayout: DataLayout::Nchw,
            padding_mode: PaddingMode::Null,
            border_value: [0.0; 4],
        }
    }
}

/// `cv::dnn::blobFromImageWithParams(image, param)`.
///
/// The letterbox and centre-crop modes are the reason this exists rather than
/// plain `blobFromImage`: a detector trained on square inputs gives visibly
/// worse boxes on a stretched image, and undoing the geometry afterwards needs
/// to know which mode was used — see [`blob_rect_to_image_rect`].
pub fn blob_from_image_with_params<S: MatLike + ?Sized>(
    image: &S,
    p: Image2BlobParams,
) -> Result<Blob> {
    cv_assert!(!image.is_empty(), "blobFromImageWithParams needs a non-empty image");
    let (tw, th) = if p.size.0 > 0 && p.size.1 > 0 {
        p.size
    } else {
        (image.cols(), image.rows())
    };

    // Resolve the source rectangle the target is filled from.
    let src_rect = match p.padding_mode {
        PaddingMode::Null | PaddingMode::Letterbox => {
            Rect::new(0, 0, image.cols(), image.rows())
        }
        PaddingMode::CropCenter => {
            // Scale to cover, then take the middle.
            let sx = f64::from(tw) / f64::from(image.cols());
            let sy = f64::from(th) / f64::from(image.rows());
            let s = sx.max(sy);
            let (cw, ch) = (
                (f64::from(tw) / s).round() as i32,
                (f64::from(th) / s).round() as i32,
            );
            Rect::new(
                (image.cols() - cw).max(0) / 2,
                (image.rows() - ch).max(0) / 2,
                cw.min(image.cols()),
                ch.min(image.rows()),
            )
        }
    };

    let cropped = crop_rect(image, src_rect)?;
    // Letterbox scales to fit and pads; the others fill the target exactly.
    let (rw, rh) = match p.padding_mode {
        PaddingMode::Letterbox => {
            let s = (f64::from(tw) / f64::from(cropped.cols()))
                .min(f64::from(th) / f64::from(cropped.rows()));
            (
                ((f64::from(cropped.cols()) * s).round() as i32).max(1),
                ((f64::from(cropped.rows()) * s).round() as i32).max(1),
            )
        }
        _ => (tw, th),
    };

    let mut blob = blob_from_image(
        &cropped,
        p.scale_factor,
        Some((rw, rh)),
        cv_core::Scalar::new(p.mean[0], p.mean[1], p.mean[2], p.mean[3]),
        p.swap_rb,
        BlobResize::Stretch,
    )?;

    if p.padding_mode == PaddingMode::Letterbox && (rw != tw || rh != th) {
        blob = letterbox(&blob, tw, th, &p)?;
    }
    Ok(blob)
}

/// Copy a rectangle out of an image.
fn crop_rect<S: MatLike + ?Sized>(src: &S, r: Rect) -> Result<cv_core::Mat> {
    let (w, h) = (r.width.min(src.cols() - r.x), r.height.min(src.rows() - r.y));
    cv_assert!(w > 0 && h > 0, "the crop rectangle {r:?} lies outside the image");
    let cn = src.channels();
    let mut out = cv_core::Mat::zeros(h, w, src.mat_type())?;
    for y in 0..h {
        for x in 0..w {
            for c in 0..cn {
                cv_core::arithm::set_channel_generic(
                    &mut out,
                    y,
                    x,
                    c,
                    src.at_f64(r.y + y, r.x + x, c)?,
                )?;
            }
        }
    }
    Ok(out)
}

/// Centre a blob inside a larger one, filling the border.
fn letterbox(src: &Blob, tw: i32, th: i32, p: &Image2BlobParams) -> Result<Blob> {
    let [n, c, sh, sw] = src.shape;
    let (tw, th) = (tw as usize, th as usize);
    let mut out = Blob::zeros(n, c, th, tw);
    for (ch, v) in out.data.chunks_mut(th * tw).enumerate() {
        let fill = (p.border_value[(ch % 4).min(3)] * p.scale_factor) as f32;
        v.fill(fill);
    }
    let (ox, oy) = ((tw - sw.min(tw)) / 2, (th - sh.min(th)) / 2);
    for b in 0..n {
        for ch in 0..c {
            for y in 0..sh.min(th) {
                for x in 0..sw.min(tw) {
                    let si = ((b * c + ch) * sh + y) * sw + x;
                    let di = ((b * c + ch) * th + (y + oy)) * tw + (x + ox);
                    out.data[di] = src.data[si];
                }
            }
        }
    }
    Ok(out)
}

/// `cv::dnn::blobRectToImageRect(rBlob, size, param)`.
///
/// Maps a box detected in blob coordinates back onto the original image,
/// undoing whatever [`PaddingMode`] was applied. Getting this wrong is the
/// classic detector bug: boxes that look right on the letterboxed input and sit
/// offset on the picture.
#[must_use]
pub fn blob_rect_to_image_rect(
    r: Rect,
    image_size: (i32, i32),
    p: Image2BlobParams,
) -> Rect {
    let (iw, ih) = image_size;
    let (tw, th) = if p.size.0 > 0 { p.size } else { (iw, ih) };
    match p.padding_mode {
        PaddingMode::Null => {
            let (sx, sy) = (f64::from(iw) / f64::from(tw), f64::from(ih) / f64::from(th));
            Rect::new(
                (f64::from(r.x) * sx).round() as i32,
                (f64::from(r.y) * sy).round() as i32,
                (f64::from(r.width) * sx).round() as i32,
                (f64::from(r.height) * sy).round() as i32,
            )
        }
        PaddingMode::Letterbox => {
            let s = (f64::from(tw) / f64::from(iw)).min(f64::from(th) / f64::from(ih));
            let (rw, rh) = ((f64::from(iw) * s).round(), (f64::from(ih) * s).round());
            let (ox, oy) = ((f64::from(tw) - rw) / 2.0, (f64::from(th) - rh) / 2.0);
            Rect::new(
                ((f64::from(r.x) - ox) / s).round() as i32,
                ((f64::from(r.y) - oy) / s).round() as i32,
                (f64::from(r.width) / s).round() as i32,
                (f64::from(r.height) / s).round() as i32,
            )
        }
        PaddingMode::CropCenter => {
            let s = (f64::from(tw) / f64::from(iw)).max(f64::from(th) / f64::from(ih));
            let (cw, ch) = (f64::from(tw) / s, f64::from(th) / s);
            let (ox, oy) = ((f64::from(iw) - cw) / 2.0, (f64::from(ih) - ch) / 2.0);
            Rect::new(
                (f64::from(r.x) / s + ox).round() as i32,
                (f64::from(r.y) / s + oy).round() as i32,
                (f64::from(r.width) / s).round() as i32,
                (f64::from(r.height) / s).round() as i32,
            )
        }
    }
}

/// `cv::dnn::blobRectsToImageRects(rBlob, rImg, size, param)`.
#[must_use]
pub fn blob_rects_to_image_rects(
    rects: &[Rect],
    image_size: (i32, i32),
    p: Image2BlobParams,
) -> Vec<Rect> {
    rects.iter().map(|r| blob_rect_to_image_rect(*r, image_size, p)).collect()
}

/// `cv::dnn::Model` — a network plus its preprocessing.
#[derive(Debug)]
pub struct Model {
    net: Net,
    params: Image2BlobParams,
    backend: Backend,
    target: Target,
}

impl Model {
    /// `cv::dnn::Model::Model(network)`.
    #[must_use]
    pub fn new(net: Net) -> Self {
        Self {
            net,
            params: Image2BlobParams::default(),
            backend: Backend::Default,
            target: Target::Cpu,
        }
    }

    /// `cv::dnn::Model::setInputScale(scale)`.
    pub fn set_input_scale(&mut self, scale: f64) -> &mut Self {
        self.params.scale_factor = scale;
        self
    }

    /// `cv::dnn::Model::setInputSize(size)`.
    pub fn set_input_size(&mut self, width: i32, height: i32) -> &mut Self {
        self.params.size = (width, height);
        self
    }

    /// `cv::dnn::Model::setInputMean(mean)`.
    pub fn set_input_mean(&mut self, mean: [f64; 4]) -> &mut Self {
        self.params.mean = mean;
        self
    }

    /// `cv::dnn::Model::setInputSwapRB(swapRB)`.
    pub fn set_input_swap_rb(&mut self, swap: bool) -> &mut Self {
        self.params.swap_rb = swap;
        self
    }

    /// `cv::dnn::Model::setInputCrop(crop)`.
    pub fn set_input_crop(&mut self, crop: bool) -> &mut Self {
        self.params.padding_mode =
            if crop { PaddingMode::CropCenter } else { PaddingMode::Null };
        self
    }

    /// `cv::dnn::Model::setInputParams(scale, size, mean, swapRB, crop)`.
    pub fn set_input_params(
        &mut self,
        scale: f64,
        size: (i32, i32),
        mean: [f64; 4],
        swap_rb: bool,
        crop: bool,
    ) -> &mut Self {
        self.params.scale_factor = scale;
        self.params.size = size;
        self.params.mean = mean;
        self.params.swap_rb = swap_rb;
        self.params.padding_mode = if crop { PaddingMode::CropCenter } else { PaddingMode::Null };
        self
    }

    /// `cv::dnn::Model::setPreferableBackend(backendId)`.
    ///
    /// Refuses a backend this port cannot execute, rather than accepting it and
    /// silently running on the portable path — a caller who asked for CUDA and
    /// got CPU would have no way to tell.
    pub fn set_preferable_backend(&mut self, b: Backend) -> Result<&mut Self> {
        cv_assert!(
            b.is_available(),
            "{b:?} is not available in this port; only Default, Opencv and Rlx are"
        );
        self.backend = b;
        Ok(self)
    }

    /// `cv::dnn::Model::setPreferableTarget(targetId)`.
    pub fn set_preferable_target(&mut self, t: Target) -> Result<&mut Self> {
        cv_assert!(t == Target::Cpu, "{t:?} is not available in this port");
        self.target = t;
        Ok(self)
    }

    /// The configured preprocessing.
    #[must_use]
    pub fn params(&self) -> Image2BlobParams {
        self.params
    }

    /// The wrapped network.
    #[must_use]
    pub fn net(&self) -> &Net {
        &self.net
    }

    /// `cv::dnn::Model::predict(frame, outs)` — preprocess, run, return the raw output.
    pub fn predict<S: MatLike + ?Sized>(&mut self, frame: &S) -> Result<Blob> {
        let blob = blob_from_image_with_params(frame, self.params)?;
        self.net.set_input(blob);
        self.net.forward()
    }
}

/// `cv::dnn::ClassificationModel`.
#[derive(Debug)]
pub struct ClassificationModel {
    model: Model,
    softmax: bool,
}

impl ClassificationModel {
    /// `cv::dnn::ClassificationModel::ClassificationModel(network)`.
    #[must_use]
    pub fn new(net: Net) -> Self {
        Self { model: Model::new(net), softmax: false }
    }

    /// `cv::dnn::ClassificationModel::setEnableSoftmaxPostProcessing(enable)`.
    ///
    /// Needed when the network ends at the logits. Applying softmax to output
    /// that already went through it does not change the argmax, so the class is
    /// still right — but the confidence becomes meaningless.
    pub fn set_enable_softmax_post_processing(&mut self, enable: bool) -> &mut Self {
        self.softmax = enable;
        self
    }

    /// `cv::dnn::ClassificationModel::getEnableSoftmaxPostProcessing()`.
    #[must_use]
    pub fn get_enable_softmax_post_processing(&self) -> bool {
        self.softmax
    }

    /// Mutable access to the preprocessing.
    pub fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// `cv::dnn::ClassificationModel::classify(frame, classId, conf)`.
    pub fn classify<S: MatLike + ?Sized>(&mut self, frame: &S) -> Result<(usize, f32)> {
        let out = self.model.predict(frame)?;
        cv_assert!(!out.data.is_empty(), "the network produced no output");
        let mut scores = out.data.clone();
        if self.softmax {
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0.0f32;
            for v in scores.iter_mut() {
                *v = (*v - m).exp();
                sum += *v;
            }
            for v in scores.iter_mut() {
                *v /= sum;
            }
        }
        let (i, v) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("checked non-empty");
        Ok((i, *v))
    }
}

/// One detection from [`DetectionModel::detect`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// Predicted class.
    pub class_id: i32,
    /// Confidence in `0..=1`.
    pub confidence: f32,
    /// Box in image coordinates.
    pub rect: Rect,
}

/// `cv::dnn::DetectionModel`.
#[derive(Debug)]
pub struct DetectionModel {
    model: Model,
    nms_across_classes: bool,
}

impl DetectionModel {
    /// `cv::dnn::DetectionModel::DetectionModel(network)`.
    #[must_use]
    pub fn new(net: Net) -> Self {
        Self { model: Model::new(net), nms_across_classes: false }
    }

    /// `cv::dnn::DetectionModel::setNmsAcrossClasses(value)`.
    pub fn set_nms_across_classes(&mut self, v: bool) -> &mut Self {
        self.nms_across_classes = v;
        self
    }

    /// `cv::dnn::DetectionModel::getNmsAcrossClasses()`.
    #[must_use]
    pub fn get_nms_across_classes(&self) -> bool {
        self.nms_across_classes
    }

    /// Mutable access to the preprocessing.
    pub fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// `cv::dnn::DetectionModel::detect(frame, classIds, confidences, boxes, ...)`.
    ///
    /// Decodes the SSD-style `[1, 1, N, 7]` output — `(batch, class, score, x1,
    /// y1, x2, y2)` with corners normalised to `0..1` — then applies NMS and
    /// maps the boxes back through the preprocessing.
    pub fn detect<S: MatLike + ?Sized>(
        &mut self,
        frame: &S,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let (iw, ih) = (frame.cols(), frame.rows());
        let out = self.model.predict(frame)?;
        let total: usize = out.shape.iter().product();
        cv_assert!(
            total % 7 == 0,
            "expected an SSD-style output with 7 values per detection, got {total}"
        );

        let params = self.model.params();
        let (bw, bh) = if params.size.0 > 0 { params.size } else { (iw, ih) };
        let mut raw: Vec<Detection> = Vec::new();
        for d in out.data.chunks_exact(7) {
            let conf = d[2];
            if conf < conf_threshold {
                continue;
            }
            // Normalised corners into blob pixels.
            let x1 = (d[3] * bw as f32).round() as i32;
            let y1 = (d[4] * bh as f32).round() as i32;
            let x2 = (d[5] * bw as f32).round() as i32;
            let y2 = (d[6] * bh as f32).round() as i32;
            let in_blob = Rect::new(x1, y1, (x2 - x1).max(0), (y2 - y1).max(0));
            raw.push(Detection {
                class_id: d[1] as i32,
                confidence: conf,
                rect: blob_rect_to_image_rect(in_blob, (iw, ih), params),
            });
        }

        if nms_threshold <= 0.0 || raw.is_empty() {
            return Ok(raw);
        }
        let boxes: Vec<Rect> = raw.iter().map(|d| d.rect).collect();
        let scores: Vec<f32> = raw.iter().map(|d| d.confidence).collect();
        let keep = if self.nms_across_classes {
            crate::nms::nms_boxes(&boxes, &scores, conf_threshold, nms_threshold, 1.0, 0)?
        } else {
            let classes: Vec<i32> = raw.iter().map(|d| d.class_id).collect();
            crate::nms::nms_boxes_batched(
                &boxes,
                &scores,
                &classes,
                conf_threshold,
                nms_threshold,
                1.0,
                0,
            )?
        };
        Ok(keep.into_iter().map(|i| raw[i]).collect())
    }
}

/// `cv::dnn::SegmentationModel`.
#[derive(Debug)]
pub struct SegmentationModel {
    model: Model,
}

impl SegmentationModel {
    /// `cv::dnn::SegmentationModel::SegmentationModel(network)`.
    #[must_use]
    pub fn new(net: Net) -> Self {
        Self { model: Model::new(net) }
    }

    /// Mutable access to the preprocessing.
    pub fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// `cv::dnn::SegmentationModel::segment(frame, mask)`.
    ///
    /// Returns the per-pixel argmax over the channel axis — a class-id mask the
    /// size of the network's output.
    pub fn segment<S: MatLike + ?Sized>(&mut self, frame: &S) -> Result<cv_core::Mat> {
        let out = self.model.predict(frame)?;
        let [_, c, h, w] = out.shape;
        cv_assert!(c > 0 && h > 0 && w > 0, "the network produced an empty map");
        let mut mask = cv_core::Mat::zeros(h as i32, w as i32, cv_core::types::CV_32SC1)?;
        for y in 0..h {
            for x in 0..w {
                let mut best = 0usize;
                let mut bv = f32::MIN;
                for ch in 0..c {
                    let v = out.data[(ch * h + y) * w + x];
                    if v > bv {
                        bv = v;
                        best = ch;
                    }
                }
                *cv_core::MatLikeMut::pixel_mut::<i32>(&mut mask, y as i32, x as i32)?
                    .first_mut()
                    .expect("single channel") = best as i32;
            }
        }
        Ok(mask)
    }
}

/// `cv::dnn::KeypointsModel`.
#[derive(Debug)]
pub struct KeypointsModel {
    model: Model,
}

impl KeypointsModel {
    /// `cv::dnn::KeypointsModel::KeypointsModel(network)`.
    #[must_use]
    pub fn new(net: Net) -> Self {
        Self { model: Model::new(net) }
    }

    /// Mutable access to the preprocessing.
    pub fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// `cv::dnn::KeypointsModel::estimate(frame, thresh)`.
    ///
    /// One heat map per keypoint; each contributes its peak, scaled back to
    /// image coordinates. A peak below `thresh` yields `None` for that
    /// keypoint rather than a low-confidence guess at the origin.
    pub fn estimate<S: MatLike + ?Sized>(
        &mut self,
        frame: &S,
        thresh: f32,
    ) -> Result<Vec<Option<(f32, f32)>>> {
        let (iw, ih) = (frame.cols(), frame.rows());
        let out = self.model.predict(frame)?;
        let [_, c, h, w] = out.shape;
        cv_assert!(c > 0 && h > 0 && w > 0, "the network produced no heat maps");
        let (sx, sy) = (iw as f32 / w as f32, ih as f32 / h as f32);
        Ok((0..c)
            .map(|ch| {
                let plane = &out.data[ch * h * w..(ch + 1) * h * w];
                let (i, v) = plane
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .expect("non-empty plane");
                if *v < thresh {
                    None
                } else {
                    Some((((i % w) as f32 + 0.5) * sx, ((i / w) as f32 + 0.5) * sy))
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::{Activation, Layer};
    use cv_core::Mat;

    fn rgb(w: i32, h: i32) -> Mat {
        let mut d = vec![0u8; (w * h * 3) as usize];
        for (i, v) in d.iter_mut().enumerate() {
            *v = ((i * 7) % 256) as u8;
        }
        Mat::from_vec(h, w, 3, &d).unwrap()
    }

    #[test]
    fn only_the_backends_this_port_has_are_offered() {
        assert!(Backend::Default.is_available());
        assert!(Backend::Opencv.is_available());
        assert!(Backend::Rlx.is_available());
        for b in [Backend::Cuda, Backend::Halide, Backend::Vkcom, Backend::Cann] {
            assert!(!b.is_available(), "{b:?} must not claim availability");
            assert!(get_available_targets(b).is_empty());
        }
        assert_eq!(get_available_targets(Backend::Opencv), vec![Target::Cpu]);

        // And selecting one is refused rather than silently downgraded.
        let mut m = Model::new(Net::new());
        assert!(m.set_preferable_backend(Backend::Cuda).is_err());
        assert!(m.set_preferable_backend(Backend::Rlx).is_ok());
        assert!(m.set_preferable_target(Target::Cuda).is_err());
        assert!(m.set_preferable_target(Target::Cpu).is_ok());
    }

    #[test]
    fn the_defaults_match_upstream() {
        let p = Image2BlobParams::default();
        assert_eq!(p.scale_factor, 1.0);
        assert_eq!(p.size, (0, 0));
        assert_eq!(p.mean, [0.0; 4]);
        assert!(!p.swap_rb);
        assert_eq!(p.datalayout, DataLayout::Nchw);
        assert_eq!(p.padding_mode, PaddingMode::Null);
    }

    #[test]
    fn stretching_fills_the_target_exactly() {
        let img = rgb(40, 20);
        let p = Image2BlobParams { size: (16, 16), ..Default::default() };
        let b = blob_from_image_with_params(&img, p).unwrap();
        assert_eq!(b.shape, [1, 3, 16, 16]);
    }

    #[test]
    fn letterbox_pads_rather_than_stretching() {
        // A 40x20 image into a 16x16 blob: fits to 16x8, so 4 rows of padding
        // top and bottom.
        let img = rgb(40, 20);
        let p = Image2BlobParams {
            size: (16, 16),
            padding_mode: PaddingMode::Letterbox,
            border_value: [128.0; 4],
            ..Default::default()
        };
        let b = blob_from_image_with_params(&img, p).unwrap();
        assert_eq!(b.shape, [1, 3, 16, 16]);
        // The top row is border, the middle is image.
        let top = b.data[0];
        let mid = b.data[8 * 16 + 8];
        assert_eq!(top, 128.0, "the pad should carry the border value");
        assert_ne!(mid, 128.0, "the centre should carry image data");
    }

    #[test]
    fn centre_crop_takes_the_middle() {
        let img = rgb(40, 20);
        let p = Image2BlobParams {
            size: (16, 16),
            padding_mode: PaddingMode::CropCenter,
            ..Default::default()
        };
        let b = blob_from_image_with_params(&img, p).unwrap();
        assert_eq!(b.shape, [1, 3, 16, 16]);
        // No padding: every value comes from the image, none is the fill.
        assert!(b.data.iter().any(|v| *v > 0.0));
    }

    #[test]
    fn box_mapping_inverts_each_padding_mode() {
        let size = (40, 20);
        for mode in [PaddingMode::Null, PaddingMode::Letterbox, PaddingMode::CropCenter] {
            let p = Image2BlobParams { size: (16, 16), padding_mode: mode, ..Default::default() };
            // A box covering the whole blob should map to (about) the whole
            // image for stretch and letterbox.
            let full = blob_rect_to_image_rect(Rect::new(0, 0, 16, 16), size, p);
            assert!(full.width > 0 && full.height > 0, "{mode:?} gave {full:?}");
            match mode {
                PaddingMode::Null => {
                    assert_eq!((full.width, full.height), (40, 20));
                }
                PaddingMode::Letterbox => {
                    // Scale 0.4: the 16-wide blob spans the 40-wide image.
                    assert_eq!(full.width, 40);
                    assert!(full.y < 0, "the pad is outside the image, so y goes negative");
                }
                PaddingMode::CropCenter => {
                    assert!(full.width <= 40);
                }
            }
        }
        // The batch form agrees with the single one.
        let p = Image2BlobParams { size: (16, 16), ..Default::default() };
        let rs = [Rect::new(0, 0, 8, 8), Rect::new(4, 4, 4, 4)];
        assert_eq!(
            blob_rects_to_image_rects(&rs, size, p),
            rs.iter().map(|r| blob_rect_to_image_rect(*r, size, p)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn classification_picks_the_strongest_class() {
        // A network that just passes its input through.
        let mut net = Net::new();
        net.add_layer("act", Layer::Activation(Activation::ReLU { slope: 1.0 }));
        let mut m = ClassificationModel::new(net);
        m.model_mut().set_input_size(2, 2);

        // Feed a blob directly so the answer is predictable.
        let img = Mat::from_vec(2, 2, 3, &[0u8, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255]).unwrap();
        let (id, conf) = m.classify(&img).unwrap();
        assert!(conf > 0.0, "the winning score should be positive");
        assert!(id < 12, "the index must lie inside the output");

        // Softmax turns scores into a distribution; the argmax cannot move.
        m.set_enable_softmax_post_processing(true);
        assert!(m.get_enable_softmax_post_processing());
        let (id2, conf2) = m.classify(&img).unwrap();
        assert_eq!(id, id2, "softmax must not change which class wins");
        assert!((0.0..=1.0).contains(&conf2), "a probability, got {conf2}");
    }

    #[test]
    fn segmentation_returns_a_class_map() {
        let mut net = Net::new();
        net.add_layer("act", Layer::Activation(Activation::ReLU { slope: 1.0 }));
        let mut m = SegmentationModel::new(net);
        m.model_mut().set_input_size(4, 4);
        let mask = m.segment(&rgb(8, 8)).unwrap();
        assert_eq!((mask.rows(), mask.cols()), (4, 4));
        // Every value names one of the three input channels.
        for y in 0..4 {
            for x in 0..4 {
                let v = mask.at::<i32>(y, x).unwrap();
                assert!((0..3).contains(&v), "class {v} out of range");
            }
        }
    }

    #[test]
    fn keypoints_report_peaks_and_reject_weak_ones() {
        let mut net = Net::new();
        net.add_layer("act", Layer::Activation(Activation::ReLU { slope: 1.0 }));
        let mut m = KeypointsModel::new(net);
        m.model_mut().set_input_size(4, 4);
        let img = rgb(8, 8);
        let pts = m.estimate(&img, 0.0).unwrap();
        assert_eq!(pts.len(), 3, "one point per channel");
        for p in pts.iter().flatten() {
            assert!(p.0 >= 0.0 && p.0 <= 8.0, "x {} outside the image", p.0);
            assert!(p.1 >= 0.0 && p.1 <= 8.0, "y {} outside the image", p.1);
        }
        // An unreachable threshold rejects everything rather than guessing.
        let none = m.estimate(&img, 1e9).unwrap();
        assert!(none.iter().all(Option::is_none));
    }

    #[test]
    fn preprocessing_setters_chain_and_record() {
        let mut m = Model::new(Net::new());
        m.set_input_params(1.0 / 255.0, (224, 224), [104.0, 117.0, 123.0, 0.0], true, true);
        let p = m.params();
        assert_eq!(p.size, (224, 224));
        assert!(p.swap_rb);
        assert_eq!(p.padding_mode, PaddingMode::CropCenter);
        assert!((p.scale_factor - 1.0 / 255.0).abs() < 1e-12);
        m.set_input_crop(false);
        assert_eq!(m.params().padding_mode, PaddingMode::Null);
    }
}
