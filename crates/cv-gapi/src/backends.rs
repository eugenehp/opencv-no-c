//! Backends, streaming sources, inference configuration and the render
//! primitives (`modules/gapi/include/opencv2/gapi/cpu/gcpukernel.hpp`,
//! `ocl/goclkernel.hpp`, `fluid/gfluidkernel.hpp`, `streaming/source.hpp`,
//! `streaming/onevpl/*.hpp`, `infer/*.hpp`, `render/render_types.hpp`).
//!
//! # What a backend is
//!
//! A backend is the pairing of a *kernel implementation* with the *context* it
//! runs in. The context is the interesting half: a CPU kernel is handed plain
//! matrices, an OpenCL kernel is handed device buffers it must not touch from
//! the host, and a fluid kernel is handed a few *rows* at a time rather than a
//! whole image.
//!
//! That last one is the reason fluid exists. Processing a 4K frame through six
//! filters touches 200 MB of intermediates and misses cache at every step;
//! processing it in strips of a dozen rows keeps the working set in L2 and can
//! be several times faster with no change to the graph.
//!
//! # Sources, and why they are not just iterators
//!
//! A [`GSource`] hands frames to a streaming graph. It is a trait rather than
//! an iterator because a real source has a *pull* protocol with backpressure —
//! a camera that is not read from drops frames, and a file that is read too
//! fast must block — and because the frame it yields may live in device memory
//! that never touches the host.

use cv_core::{cv_assert, cv_error, Code, Mat, MatLike, Result};

use crate::graph::{GBackend, GKernelPackage, Rect, Scalar};

// ---------------------------------------------------------------------------
// Portability macros
// ---------------------------------------------------------------------------

/// `CV_ALWAYS_INLINE` — the attribute G-API's headers put on a hot helper.
///
/// Rust spells it `#[inline(always)]`. The constant carries the C++ spelling so
/// a translated header's intent is still visible; the attribute itself is what
/// a Rust caller writes.
pub const CV_ALWAYS_INLINE: &str = "#[inline(always)]";

/// `CV_OVERRIDE` — C++'s `override`, which G-API spells through a macro for
/// compilers that predate it.
///
/// Rust has no equivalent and needs none: a trait method that does not exist
/// on the trait is a compile error, which is exactly what `override` was added
/// to C++ to provide.
pub const CV_OVERRIDE: &str = "";

// ---------------------------------------------------------------------------
// Kernel contexts
// ---------------------------------------------------------------------------

/// `cv::GCPUContext` — what a CPU kernel is handed when it runs.
#[derive(Debug, Default)]
pub struct GCPUContext {
    inputs: Vec<Mat>,
    outputs: Vec<Mat>,
}

impl GCPUContext {
    /// A context over the given inputs.
    #[must_use]
    pub fn new(inputs: Vec<Mat>) -> Self {
        GCPUContext { inputs, outputs: Vec::new() }
    }

    /// `cv::GCPUContext::inMat(input)`.
    pub fn in_mat(&self, input: usize) -> Result<&Mat> {
        self.inputs.get(input).ok_or_else(|| {
            cv_core::Error::new(
                Code::StsOutOfRange,
                format!("input {input} of {}", self.inputs.len()),
            )
        })
    }

    /// `cv::GCPUContext::outMatR(output)`.
    pub fn out_mat_r(&mut self, output: usize) -> Result<&mut Mat> {
        cv_assert!(output < self.outputs.len(), "output {output} of {}", self.outputs.len());
        Ok(&mut self.outputs[output])
    }

    /// Declare the outputs a kernel will write.
    pub fn set_outputs(&mut self, outputs: Vec<Mat>) {
        self.outputs = outputs;
    }

    /// The outputs, once the kernel has run.
    #[must_use]
    pub fn outputs(&self) -> &[Mat] {
        &self.outputs
    }
}

/// The body of a CPU kernel: it reads its inputs from the context and writes
/// its outputs back.
type KernelBody = Box<dyn Fn(&mut GCPUContext) -> Result<()> + Send + Sync>;

/// `cv::GCPUKernel` — one operation's CPU implementation.
pub struct GCPUKernel {
    /// The operation id this implements.
    pub id: String,
    /// The body, called with the context.
    run: KernelBody,
}

impl std::fmt::Debug for GCPUKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GCPUKernel").field("id", &self.id).finish_non_exhaustive()
    }
}

impl GCPUKernel {
    /// `cv::GCPUKernel::GCPUKernel(f)`.
    pub fn new(
        id: &str,
        run: impl Fn(&mut GCPUContext) -> Result<()> + Send + Sync + 'static,
    ) -> Self {
        GCPUKernel { id: id.to_string(), run: Box::new(run) }
    }

    /// `cv::GCPUKernel::apply(ctx)`.
    pub fn apply(&self, ctx: &mut GCPUContext) -> Result<()> {
        (self.run)(ctx)
    }
}

/// `cv::GOCLContext` — what an OpenCL kernel is handed.
///
/// The matrices are `UMat`s upstream: device-resident buffers that a host
/// pointer must not be taken to. Reading one from the host is what turns an
/// OpenCL pipeline into a slower CPU pipeline with extra copies, so the
/// distinction is worth keeping in the type.
#[derive(Debug, Default)]
pub struct GOCLContext {
    inputs: Vec<cv_core::UMat>,
}

impl GOCLContext {
    /// A context over the given device buffers.
    #[must_use]
    pub fn new(inputs: Vec<cv_core::UMat>) -> Self {
        GOCLContext { inputs }
    }

    /// `cv::GOCLContext::inMat(input)`.
    pub fn in_mat(&self, input: usize) -> Result<&cv_core::UMat> {
        self.inputs.get(input).ok_or_else(|| {
            cv_core::Error::new(Code::StsOutOfRange, format!("input {input}"))
        })
    }
}

/// `cv::GOCLKernel` — one operation's OpenCL implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GOCLKernel {
    /// The operation id.
    pub id: String,
}

impl GOCLKernel {
    /// Declare an OpenCL implementation of an operation.
    #[must_use]
    pub fn new(id: &str) -> Self {
        GOCLKernel { id: id.to_string() }
    }
}

/// `cv::GFluidKernel` — a kernel that runs on a *window of rows*.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GFluidKernel {
    /// The operation id.
    pub id: String,
    /// `m_window` — how many input rows one output row needs.
    ///
    /// A 3x3 filter has a window of 3; a point operation has 1. The scheduler
    /// uses it to size the ring buffers between stages, which is what keeps the
    /// whole pipeline's working set in cache.
    pub window: i32,
}

impl GFluidKernel {
    /// Declare a fluid implementation with its row window.
    pub fn new(id: &str, window: i32) -> Result<Self> {
        cv_assert!(window > 0, "a fluid kernel's window must be at least one row");
        cv_assert!(window % 2 == 1, "a centred window must be odd, got {window}");
        Ok(GFluidKernel { id: id.to_string(), window })
    }

    /// How many rows of border each side of a strip needs.
    #[must_use]
    pub fn border_size(&self) -> i32 {
        self.window / 2
    }
}

/// `cv::GCompoundKernel` — a kernel defined as a subgraph of other kernels.
///
/// The compound is expanded before compilation, so its parts can each land on
/// whichever backend suits them. That is the payoff: a compound written once
/// picks up a new backend's kernels automatically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GCompoundKernel {
    /// The operation this defines.
    pub id: String,
    /// The kernels it expands into, in order.
    pub expansion: Vec<String>,
}

impl GCompoundKernel {
    /// Declare a compound and what it expands to.
    pub fn new(id: &str, expansion: Vec<String>) -> Result<Self> {
        cv_assert!(!expansion.is_empty(), "a compound kernel must expand to something");
        cv_assert!(
            !expansion.iter().any(|e| e == id),
            "`{id}` expands to itself, which would not terminate"
        );
        Ok(GCompoundKernel { id: id.to_string(), expansion })
    }
}

/// `cv::gapi::plaidml::GPlaidMLKernel` — a kernel expressed in PlaidML's
/// tensor language rather than as code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GPlaidMLKernel {
    /// The operation id.
    pub id: String,
}

impl GPlaidMLKernel {
    /// Declare a PlaidML implementation.
    #[must_use]
    pub fn new(id: &str) -> Self {
        GPlaidMLKernel { id: id.to_string() }
    }
}

/// `cv::gapi::python::GPythonKernel` — a kernel implemented in Python.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GPythonKernel {
    /// The operation id.
    pub id: String,
    /// The Python callable's qualified name.
    pub callable: String,
}

impl GPythonKernel {
    /// Declare a Python implementation.
    #[must_use]
    pub fn new(id: &str, callable: &str) -> Self {
        GPythonKernel { id: id.to_string(), callable: callable.to_string() }
    }
}

/// `cv::gapi::python::GPythonFunctor` — what registers a [`GPythonKernel`] into
/// a package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GPythonFunctor {
    /// The kernel it registers.
    pub kernel: GPythonKernel,
}

impl GPythonFunctor {
    /// `cv::gapi::python::GPythonFunctor::GPythonFunctor(id, meta, run)`.
    #[must_use]
    pub fn new(kernel: GPythonKernel) -> Self {
        GPythonFunctor { kernel }
    }

    /// `cv::gapi::python::GPythonFunctor::backend()`.
    #[must_use]
    pub fn backend(&self) -> GBackend {
        GBackend::Python
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// `cv::MediaFrame::View` — a borrowed look at a frame's planes.
///
/// A view exists so the frame's owner knows when the host is done with it. A
/// hardware frame may need mapping into host memory, and the mapping must be
/// released; holding a raw pointer instead is how a decoder's buffer pool runs
/// dry three seconds into a stream.
#[derive(Debug, Clone, PartialEq)]
pub struct View {
    /// One pointer's worth of data per plane.
    pub planes: Vec<Vec<u8>>,
    /// Bytes per row, per plane.
    pub strides: Vec<usize>,
}

impl View {
    /// Construct a view over already-mapped planes.
    pub fn new(planes: Vec<Vec<u8>>, strides: Vec<usize>) -> Result<Self> {
        cv_assert!(
            planes.len() == strides.len(),
            "{} planes but {} strides",
            planes.len(),
            strides.len()
        );
        cv_assert!(!planes.is_empty(), "a view needs at least one plane");
        Ok(View { planes, strides })
    }
}

/// `cv::MediaFrame` — a frame in whatever layout its producer chose.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaFrame {
    /// The frame's format and size.
    pub desc: crate::graph::GFrameDesc,
    /// The planes, once mapped.
    pub view: View,
}

impl MediaFrame {
    /// `cv::MediaFrame::Create<T>(...)`.
    #[must_use]
    pub fn new(desc: crate::graph::GFrameDesc, view: View) -> Self {
        MediaFrame { desc, view }
    }

    /// `cv::MediaFrame::desc()`.
    #[must_use]
    pub fn desc(&self) -> crate::graph::GFrameDesc {
        self.desc
    }

    /// `cv::MediaFrame::access(mode)` — map the planes for host access.
    #[must_use]
    pub fn access(&self) -> &View {
        &self.view
    }
}

/// `cv::RMat` — a matrix whose storage the backend owns.
///
/// An `RMat` may live on a device. Its `access` is what a host read costs, and
/// making that explicit is the point: a graph that never calls it can keep the
/// data on the accelerator from source to sink.
#[derive(Debug, Clone)]
pub struct RMat {
    desc: crate::graph::GMatDesc,
    data: Mat,
}

impl RMat {
    /// Wrap a host matrix as an `RMat`.
    pub fn new(data: Mat) -> Self {
        RMat { desc: crate::graph::descr_of(&data), data }
    }

    /// `cv::RMat::desc()`.
    #[must_use]
    pub fn desc(&self) -> crate::graph::GMatDesc {
        self.desc
    }

    /// `cv::RMat::access(mode)` — bring the data to the host.
    #[must_use]
    pub fn access(&self) -> &Mat {
        &self.data
    }
}

/// `cv::gapi::wip::Data` — one value flowing through a streaming graph.
#[derive(Debug, Clone)]
pub enum Data {
    /// A matrix.
    Mat(Mat),
    /// A backend-owned matrix.
    RMat(RMat),
    /// A media frame.
    Frame(MediaFrame),
    /// A scalar.
    Scalar(Scalar),
}

/// `cv::gapi::wip::IStreamSource`, which the graph refers to as `cv::GSource` —
/// where a streaming graph's frames come from.
pub trait GSource {
    /// `cv::gapi::wip::IStreamSource::pull(data)`.
    ///
    /// `Ok(false)` means the stream ended; an error means it broke. A caller
    /// must be able to tell those apart, which is why the end is not an error.
    fn pull(&mut self) -> Result<Option<Data>>;

    /// `cv::gapi::wip::IStreamSource::descr_of()`.
    fn descr_of(&self) -> crate::graph::GMatDesc;
}

/// A source over a fixed list of frames, for testing a graph without a camera.
#[derive(Debug, Clone)]
pub struct QueueSourceBase {
    frames: std::collections::VecDeque<Mat>,
    desc: crate::graph::GMatDesc,
}

impl QueueSourceBase {
    /// `cv::gapi::wip::QueueSourceBase::QueueSourceBase(desc)`.
    pub fn new(frames: Vec<Mat>) -> Result<Self> {
        cv_assert!(!frames.is_empty(), "a queue source needs at least one frame");
        let desc = crate::graph::descr_of(&frames[0]);
        cv_assert!(
            frames.iter().all(|f| crate::graph::descr_of(f) == desc),
            "every frame of a source must have the same descriptor"
        );
        Ok(QueueSourceBase { frames: frames.into(), desc })
    }

    /// `cv::gapi::wip::QueueSourceBase::push(data)`.
    pub fn push(&mut self, frame: Mat) -> Result<()> {
        cv_assert!(
            crate::graph::descr_of(&frame) == self.desc,
            "a pushed frame must match the source's descriptor"
        );
        self.frames.push_back(frame);
        Ok(())
    }

    /// How many frames are queued.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the queue has drained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

impl GSource for QueueSourceBase {
    fn pull(&mut self) -> Result<Option<Data>> {
        Ok(self.frames.pop_front().map(Data::Mat))
    }
    fn descr_of(&self) -> crate::graph::GMatDesc {
        self.desc
    }
}

/// `cv::gapi::wip::QueueInput` — the input end a caller pushes frames into.
#[derive(Debug, Clone)]
pub struct QueueInput {
    /// The queue being fed.
    pub source: QueueSourceBase,
}

impl QueueInput {
    /// Wrap a queue source as a graph input.
    #[must_use]
    pub fn new(source: QueueSourceBase) -> Self {
        QueueInput { source }
    }
}

/// `cv::gapi::wip::GStreamerSource` — frames from a GStreamer pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct GStreamerSource {
    /// The pipeline description, in `gst-launch` syntax.
    pub pipeline: String,
}

impl GStreamerSource {
    /// `cv::gapi::wip::GStreamerSource::GStreamerSource(pipeline)`.
    pub fn new(pipeline: &str) -> Result<Self> {
        cv_assert!(!pipeline.is_empty(), "a GStreamer pipeline description is required");
        Ok(GStreamerSource { pipeline: pipeline.to_string() })
    }
}

impl GSource for GStreamerSource {
    /// Errors: no GStreamer is linked.
    ///
    /// A source that returned "stream ended" instead would make a pipeline look
    /// like it had finished normally, which is the one failure mode worth
    /// avoiding here.
    fn pull(&mut self) -> Result<Option<Data>> {
        cv_error!(
            Code::StsError,
            "no GStreamer is linked, so the pipeline `{}` cannot be opened",
            self.pipeline
        )
    }
    fn descr_of(&self) -> crate::graph::GMatDesc {
        crate::graph::GMatDesc::default()
    }
}

/// `cv::gapi::wip::GStreamerPipeline` — several sources from one pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct GStreamerPipeline {
    /// The pipeline description.
    pub pipeline: String,
}

impl GStreamerPipeline {
    /// `cv::gapi::wip::GStreamerPipeline::GStreamerPipeline(pipeline)`.
    pub fn new(pipeline: &str) -> Result<Self> {
        cv_assert!(!pipeline.is_empty(), "a GStreamer pipeline description is required");
        Ok(GStreamerPipeline { pipeline: pipeline.to_string() })
    }

    /// `cv::gapi::wip::GStreamerPipeline::getStreamingSource(appsinkName)`.
    pub fn get_streaming_source(&self, appsink_name: &str) -> Result<GStreamerSource> {
        cv_assert!(!appsink_name.is_empty(), "an appsink name is required");
        GStreamerSource::new(&format!("{} ! {appsink_name}", self.pipeline))
    }
}

/// `cv::GStreamingCompiled` — a graph compiled for streaming.
///
/// Distinct from a compiled graph because it is *stateful*: it holds the
/// pipeline's in-flight frames between `pull` calls, which a one-shot `apply`
/// has no need for.
///
/// `Debug` is written out rather than derived because the source is a trait
/// object.
pub struct GStreamingCompiled {
    compiled: crate::Compiled,
    source: Option<Box<dyn GSource>>,
    running: bool,
}

impl std::fmt::Debug for GStreamingCompiled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GStreamingCompiled")
            .field("has_source", &self.source.is_some())
            .field("running", &self.running)
            .finish_non_exhaustive()
    }
}

impl GStreamingCompiled {
    /// `cv::GComputation::compileStreaming(args)`.
    #[must_use]
    pub fn new(compiled: crate::Compiled) -> Self {
        GStreamingCompiled { compiled, source: None, running: false }
    }

    /// `cv::GStreamingCompiled::setSource(ins)`.
    pub fn set_source(&mut self, source: Box<dyn GSource>) {
        self.source = Some(source);
    }

    /// `cv::GStreamingCompiled::start()`.
    pub fn start(&mut self) -> Result<()> {
        cv_assert!(self.source.is_some(), "setSource must be called before start");
        self.running = true;
        Ok(())
    }

    /// `cv::GStreamingCompiled::running()`.
    #[must_use]
    pub fn running(&self) -> bool {
        self.running
    }

    /// `cv::GStreamingCompiled::pull(outs)`.
    ///
    /// `Ok(None)` means the source is exhausted, which is how a caller's loop
    /// terminates.
    pub fn pull(&mut self) -> Result<Option<Mat>> {
        cv_assert!(self.running, "start() must be called before pull()");
        let Some(src) = self.source.as_mut() else {
            return cv_error!(Code::StsError, "no source is set");
        };
        let Some(Data::Mat(frame)) = src.pull()? else {
            self.running = false;
            return Ok(None);
        };
        Ok(Some(self.compiled.apply(&frame)?))
    }

    /// `cv::GStreamingCompiled::stop()`.
    pub fn stop(&mut self) {
        self.running = false;
    }
}

// ---------------------------------------------------------------------------
// oneVPL
// ---------------------------------------------------------------------------

/// `cv::gapi::wip::onevpl::CfgParam` — one decoder setting.
#[derive(Debug, Clone, PartialEq)]
pub struct CfgParam {
    /// The setting's name, e.g. `mfxImplDescription.mfxDecoderDescription.decoder.CodecID`.
    pub name: String,
    /// Its value, as text.
    pub value: String,
}

impl CfgParam {
    /// `cv::gapi::wip::onevpl::CfgParam::create(name, value, is_major)`.
    pub fn create(name: &str, value: &str) -> Result<Self> {
        cv_assert!(!name.is_empty(), "a configuration parameter needs a name");
        Ok(CfgParam { name: name.to_string(), value: value.to_string() })
    }
}

/// `cv::gapi::wip::onevpl::Device` — an accelerator the decoder may use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    /// A human-readable name.
    pub name: String,
    /// Which acceleration API it is reached through.
    pub accel: crate::types::AccelType,
}

impl Device {
    /// Construct a device record.
    #[must_use]
    pub fn new(name: &str, accel: crate::types::AccelType) -> Self {
        Device { name: name.to_string(), accel }
    }
}

/// `cv::gapi::wip::onevpl::Context` — the acceleration context a device runs in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Context {
    /// Which acceleration API.
    pub accel: crate::types::AccelType,
}

impl Context {
    /// Construct a context for an acceleration type.
    #[must_use]
    pub fn new(accel: crate::types::AccelType) -> Self {
        Context { accel }
    }
}

/// `cv::gapi::wip::onevpl::create_host_device()`.
#[must_use]
pub fn create_host_device() -> Device {
    Device::new("CPU", crate::types::AccelType::Host)
}

/// `cv::gapi::wip::onevpl::create_host_context()`.
#[must_use]
pub fn create_host_context() -> Context {
    Context::new(crate::types::AccelType::Host)
}

/// `cv::gapi::wip::onevpl::create_dx11_device(device_ptr, device_name)`.
///
/// Errors: DirectX 11 is a Windows API and nothing here links it. Returning a
/// device record that names no real device would let a caller select it and
/// fail much later, inside the decoder.
pub fn create_dx11_device(device_name: &str) -> Result<Device> {
    cv_error!(
        Code::StsNotImplemented,
        "no DirectX 11 is linked, so the device `{device_name}` cannot be created"
    )
}

/// `cv::gapi::wip::onevpl::create_dx11_context(device_context_ptr)`.
pub fn create_dx11_context() -> Result<Context> {
    cv_error!(Code::StsNotImplemented, "no DirectX 11 is linked")
}

/// `cv::gapi::wip::onevpl::create_vaapi_device(device_ptr, device_name)`.
pub fn create_vaapi_device(device_name: &str) -> Result<Device> {
    cv_error!(
        Code::StsNotImplemented,
        "no VAAPI is linked, so the device `{device_name}` cannot be created"
    )
}

/// `cv::gapi::wip::onevpl::create_vaapi_context(device_context_ptr)`.
pub fn create_vaapi_context() -> Result<Context> {
    cv_error!(Code::StsNotImplemented, "no VAAPI is linked")
}

/// `cv::gapi::wip::onevpl::IDeviceSelector` — chooses which device decodes.
pub trait IDeviceSelector {
    /// `cv::gapi::wip::onevpl::IDeviceSelector::select_devices()`.
    fn select_devices(&self) -> Vec<Device>;
}

/// The selector used when a caller names none: the host CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HostDeviceSelector;

impl IDeviceSelector for HostDeviceSelector {
    fn select_devices(&self) -> Vec<Device> {
        vec![create_host_device()]
    }
}

/// `cv::gapi::wip::onevpl::getDefaultDeviceSelector()`.
#[must_use]
pub fn get_default_device_selector() -> HostDeviceSelector {
    HostDeviceSelector
}

/// `cv::gapi::wip::onevpl::IDataProvider` — where the decoder reads bytes from.
pub trait IDataProvider {
    /// `cv::gapi::wip::onevpl::IDataProvider::fetch_bitstream_data(out_bitstream)`.
    fn fetch_bitstream_data(&mut self, out: &mut Vec<u8>) -> Result<bool>;
    /// `cv::gapi::wip::onevpl::IDataProvider::empty()`.
    fn empty(&self) -> bool;
}

/// A provider over an in-memory buffer.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MemoryDataProvider {
    bytes: Vec<u8>,
    at: usize,
}

impl MemoryDataProvider {
    /// Wrap a byte buffer as a data provider.
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        MemoryDataProvider { bytes, at: 0 }
    }
}

impl IDataProvider for MemoryDataProvider {
    fn fetch_bitstream_data(&mut self, out: &mut Vec<u8>) -> Result<bool> {
        if self.at >= self.bytes.len() {
            return Ok(false);
        }
        let end = (self.at + 4096).min(self.bytes.len());
        out.extend_from_slice(&self.bytes[self.at..end]);
        self.at = end;
        Ok(true)
    }
    fn empty(&self) -> bool {
        self.at >= self.bytes.len()
    }
}

/// `cv::gapi::wip::onevpl::DataProviderException` — the base of the provider
/// error family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataProviderException {
    /// What went wrong.
    pub message: String,
}

impl DataProviderException {
    /// Construct with a message.
    #[must_use]
    pub fn new(message: &str) -> Self {
        DataProviderException { message: message.to_string() }
    }

    /// `cv::gapi::wip::onevpl::DataProviderException::what()`.
    #[must_use]
    pub fn what(&self) -> &str {
        &self.message
    }
}

/// `cv::gapi::wip::onevpl::DataProviderSystemErrorException` — the underlying
/// I/O failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataProviderSystemErrorException {
    /// The base exception.
    pub base: DataProviderException,
    /// The `errno` the system reported.
    pub error_code: i32,
}

impl DataProviderSystemErrorException {
    /// Construct from an error code and a description.
    #[must_use]
    pub fn new(error_code: i32, message: &str) -> Self {
        DataProviderSystemErrorException {
            base: DataProviderException::new(message),
            error_code,
        }
    }
}

/// `cv::gapi::wip::onevpl::DataProviderUnsupportedException` — the stream is
/// well-formed but this build cannot decode it.
///
/// Separate from the system error because the two call for different responses:
/// an I/O failure may be worth retrying, an unsupported codec never is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataProviderUnsupportedException {
    /// The base exception.
    pub base: DataProviderException,
}

impl DataProviderUnsupportedException {
    /// Construct with a description.
    #[must_use]
    pub fn new(message: &str) -> Self {
        DataProviderUnsupportedException { base: DataProviderException::new(message) }
    }
}

/// `cv::gapi::wip::onevpl::DataProviderImplementationException` — the provider
/// itself is broken, as opposed to its input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataProviderImplementationException {
    /// The base exception.
    pub base: DataProviderException,
}

impl DataProviderImplementationException {
    /// Construct with a description.
    #[must_use]
    pub fn new(message: &str) -> Self {
        DataProviderImplementationException { base: DataProviderException::new(message) }
    }
}

/// `cv::gapi::wip::make_onevpl_src(filePath, cfg_params)`.
pub fn make_onevpl_src(file_path: &str, _cfg_params: &[CfgParam]) -> Result<GStreamerSource> {
    cv_error!(
        Code::StsNotImplemented,
        "no oneVPL runtime is linked, so `{file_path}` cannot be decoded"
    )
}

// ---------------------------------------------------------------------------
// Inference backends
// ---------------------------------------------------------------------------

/// Generate an inference backend's parameter block.
macro_rules! infer_params {
    ($name:ident, $tag:literal) => {
        impl $name {
            /// Name the model and the device it runs on.
            pub fn new(model: &str, device: &str) -> Result<Self> {
                cv_assert!(!model.is_empty(), "a model path is required");
                Ok($name {
                    model: model.to_string(),
                    device: device.to_string(),
                    params: Vec::new(),
                })
            }

            /// Add one backend-specific setting.
            pub fn set(&mut self, key: &str, value: &str) -> &mut Self {
                self.params.push((key.to_string(), value.to_string()));
                self
            }

            /// The backend's upstream tag.
            #[must_use]
            pub fn tag() -> &'static str {
                $tag
            }
        }
    };
}

/// `cv::gapi::ie::Params` and `cv::gapi::onnx::ep::OpenVINO` — OpenVINO
/// inference settings.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OpenVINO {
    /// Path to the model.
    pub model: String,
    /// Target device, e.g. `CPU` or `GPU`.
    pub device: String,
    /// Backend-specific settings.
    pub params: Vec<(String, String)>,
}

/// `cv::gapi::onnx::ep::CUDA` — the CUDA execution provider's settings.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CUDA {
    /// Path to the model.
    pub model: String,
    /// Device id, as text.
    pub device: String,
    /// Provider-specific settings.
    pub params: Vec<(String, String)>,
}

/// `cv::gapi::onnx::ep::CoreML` — the Core ML execution provider.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CoreML {
    /// Path to the model.
    pub model: String,
    /// Compute unit selection.
    pub device: String,
    /// Provider-specific settings.
    pub params: Vec<(String, String)>,
}

/// `cv::gapi::onnx::ep::DirectML` — the DirectML execution provider.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DirectML {
    /// Path to the model.
    pub model: String,
    /// Adapter name or index.
    pub device: String,
    /// Provider-specific settings.
    pub params: Vec<(String, String)>,
}

/// `cv::gapi::onnx::ep::TensorRT` — the TensorRT execution provider.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TensorRT {
    /// Path to the model.
    pub model: String,
    /// Device id, as text.
    pub device: String,
    /// Provider-specific settings.
    pub params: Vec<(String, String)>,
}

infer_params!(OpenVINO, "org.opencv.dnn.ie");
infer_params!(CUDA, "org.opencv.dnn.onnx.cuda");
infer_params!(CoreML, "org.opencv.dnn.onnx.coreml");
infer_params!(DirectML, "org.opencv.dnn.onnx.directml");
infer_params!(TensorRT, "org.opencv.dnn.onnx.tensorrt");

/// `cv::gapi::python::PyParams` — an inference network described from Python.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PyParams {
    /// The graph tag this network answers to.
    pub tag: String,
    /// Path to the model.
    pub model: String,
    /// The backend's name.
    pub backend: String,
}

impl PyParams {
    /// `cv::gapi::python::PyParams::PyParams(tag, model, device)`.
    pub fn new(tag: &str, model: &str, backend: &str) -> Result<Self> {
        cv_assert!(!tag.is_empty(), "a network needs a tag the graph can refer to");
        Ok(PyParams {
            tag: tag.to_string(),
            model: model.to_string(),
            backend: backend.to_string(),
        })
    }
}

/// `cv::gapi::wip::gst::GStreamerSource`'s sibling in the calibration API:
/// `cv::gapi::ColorCameraParams`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ColorCameraParams {
    /// Focal lengths.
    pub focal: (f64, f64),
    /// Principal point.
    pub principal: (f64, f64),
}

/// `cv::gapi::ColorCamera` — a colour camera in the streaming pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ColorCamera {
    /// Its intrinsics.
    pub params: ColorCameraParams,
}

/// `cv::gapi::Score` — a detection confidence, kept as its own type so it
/// cannot be mixed up with an index or a label.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Score(pub f32);

/// `cv::gapi::fluid::Buffer` — the ring buffer a fluid stage reads and writes.
///
/// It holds only as many rows as the deepest consumer's window needs, which is
/// what keeps a whole pipeline's working set small enough to stay in cache.
#[derive(Debug, Clone, PartialEq)]
pub struct Buffer {
    rows: Vec<Vec<u8>>,
    capacity: usize,
}

impl Buffer {
    /// A ring buffer holding `capacity` rows of `width` bytes.
    pub fn new(capacity: usize, width: usize) -> Result<Self> {
        cv_assert!(capacity > 0, "a fluid buffer needs at least one row");
        cv_assert!(width > 0, "a fluid buffer's rows need a width");
        Ok(Buffer { rows: Vec::with_capacity(capacity), capacity })
            .map(|mut b| {
                b.rows.push(vec![0u8; width]);
                b.rows.clear();
                b
            })
    }

    /// `cv::gapi::fluid::Buffer::length()` — rows currently held.
    #[must_use]
    pub fn length(&self) -> usize {
        self.rows.len()
    }

    /// Push a row, evicting the oldest once the ring is full.
    pub fn push(&mut self, row: Vec<u8>) {
        if self.rows.len() == self.capacity {
            self.rows.remove(0);
        }
        self.rows.push(row);
    }

    /// The `i`th held row, oldest first.
    #[must_use]
    pub fn row(&self, i: usize) -> Option<&[u8]> {
        self.rows.get(i).map(Vec::as_slice)
    }
}

// ---------------------------------------------------------------------------
// Render primitives
// ---------------------------------------------------------------------------

/// `cv::gapi::wip::draw::Rect` — a rectangle to draw.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DrawRect {
    /// Where.
    pub rect: Rect,
    /// Colour, in BGR.
    pub color: Scalar,
    /// Line thickness; negative fills.
    pub thick: i32,
}

/// `cv::gapi::wip::draw::Text` — a string to draw.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Text {
    /// What to draw.
    pub text: String,
    /// Bottom-left corner of the first character.
    pub org: (i32, i32),
    /// Font scale.
    pub ff: f64,
    /// Colour.
    pub color: Scalar,
    /// Whether the origin is the *top*-left instead.
    ///
    /// Upstream's flag, and worth having: every other primitive here is
    /// positioned from its top-left, and text alone is not.
    pub bottom_left_origin: bool,
}

/// `cv::gapi::wip::draw::Circle`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Circle {
    /// Centre.
    pub center: (i32, i32),
    /// Radius.
    pub radius: i32,
    /// Colour.
    pub color: Scalar,
    /// Thickness; negative fills.
    pub thick: i32,
}

/// `cv::gapi::wip::draw::Line`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Line {
    /// One end.
    pub pt1: (i32, i32),
    /// The other.
    pub pt2: (i32, i32),
    /// Colour.
    pub color: Scalar,
    /// Thickness.
    pub thick: i32,
}

/// `cv::gapi::wip::draw::Mosaic` — pixelate a region.
///
/// Used to anonymise a face or a plate. It is a primitive rather than a filter
/// because it must happen *inside* the render pass: doing it afterwards means
/// the un-pixelated frame existed, and on a recording pipeline that frame may
/// have been written.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Mosaic {
    /// The region to pixelate.
    pub mos: Rect,
    /// Cell size, in pixels.
    pub cell_sz: i32,
    /// How many times to decimate.
    pub decim: i32,
}

/// `cv::gapi::wip::draw::Image` — composite an image with an alpha mask.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Image {
    /// Top-left corner.
    pub org: (i32, i32),
    /// The image's size.
    pub size: (i32, i32),
    /// Per-pixel alpha, row-major.
    pub alpha: Vec<f32>,
}

/// `cv::gapi::wip::draw::Poly` — a filled or outlined polygon.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Poly {
    /// The vertices, in order.
    pub points: Vec<(i32, i32)>,
    /// Colour.
    pub color: Scalar,
    /// Thickness; negative fills.
    pub thick: i32,
}

/// One drawing primitive, as the render kernel consumes them.
#[derive(Debug, Clone, PartialEq)]
pub enum Prim {
    /// A rectangle.
    Rect(DrawRect),
    /// Some text.
    Text(Text),
    /// A circle.
    Circle(Circle),
    /// A line.
    Line(Line),
    /// A mosaic.
    Mosaic(Mosaic),
    /// An image.
    Image(Image),
    /// A polygon.
    Poly(Poly),
}

/// Draw a list of primitives onto a three-channel image.
///
/// The one render path that actually runs here; the graph nodes in
/// [`crate::kernels`] name it but are not yet bound to it.
pub fn render_prims(image: &mut Mat, prims: &[Prim]) -> Result<()> {
    cv_assert!(image.channels() == 3, "rendering targets a three-channel image");
    for p in prims {
        match p {
            Prim::Rect(r) => cv_imgproc::drawing::rectangle(
                image,
                cv_core::Rect::new(r.rect.x, r.rect.y, r.rect.width, r.rect.height),
                cv_core::Scalar::new(r.color.0[0], r.color.0[1], r.color.0[2], 0.0),
                r.thick,
            )?,
            Prim::Circle(c) => cv_imgproc::drawing::circle(
                image,
                cv_core::Point::new(c.center.0, c.center.1),
                c.radius,
                cv_core::Scalar::new(c.color.0[0], c.color.0[1], c.color.0[2], 0.0),
                c.thick,
            )?,
            Prim::Line(l) => cv_imgproc::drawing::line(
                image,
                cv_core::Point::new(l.pt1.0, l.pt1.1),
                cv_core::Point::new(l.pt2.0, l.pt2.1),
                cv_core::Scalar::new(l.color.0[0], l.color.0[1], l.color.0[2], 0.0),
                l.thick,
            )?,
            Prim::Poly(p) => {
                for i in 0..p.points.len() {
                    let a = p.points[i];
                    let b = p.points[(i + 1) % p.points.len()];
                    cv_imgproc::drawing::line(
                        image,
                        cv_core::Point::new(a.0, a.1),
                        cv_core::Point::new(b.0, b.1),
                        cv_core::Scalar::new(p.color.0[0], p.color.0[1], p.color.0[2], 0.0),
                        p.thick.max(1),
                    )?;
                }
            }
            Prim::Mosaic(m) => {
                // Average each cell and write it back, which is what makes the
                // region unrecoverable rather than merely blurred.
                let cell = m.cell_sz.max(1);
                for by in (m.mos.y..m.mos.y + m.mos.height).step_by(cell as usize) {
                    for bx in (m.mos.x..m.mos.x + m.mos.width).step_by(cell as usize) {
                        let mut acc = [0.0f64; 3];
                        let mut n = 0.0f64;
                        for y in by..(by + cell).min(image.rows()) {
                            for x in bx..(bx + cell).min(image.cols()) {
                                if y < 0 || x < 0 {
                                    continue;
                                }
                                for (c, a) in acc.iter_mut().enumerate() {
                                    *a += image.at_f64(y, x, c as i32)?;
                                }
                                n += 1.0;
                            }
                        }
                        if n == 0.0 {
                            continue;
                        }
                        for y in by..(by + cell).min(image.rows()) {
                            for x in bx..(bx + cell).min(image.cols()) {
                                if y < 0 || x < 0 {
                                    continue;
                                }
                                let px = cv_core::MatLikeMut::pixel_mut::<u8>(image, y, x)?;
                                for c in 0..3 {
                                    px[c] = (acc[c] / n).round().clamp(0.0, 255.0) as u8;
                                }
                            }
                        }
                    }
                }
            }
            Prim::Text(t) => cv_imgproc::text::put_text(
                image,
                &t.text,
                cv_core::Point::new(t.org.0, t.org.1),
                cv_imgproc::text::FontFace::Simplex,
                t.ff,
                cv_core::Scalar::new(t.color.0[0], t.color.0[1], t.color.0[2], 0.0),
                1,
                t.bottom_left_origin,
            )?,
            Prim::Image(_) => {
                return cv_error!(
                    Code::StsNotImplemented,
                    "compositing an image primitive needs its pixel data, which Image does not carry"
                )
            }
        }
    }
    Ok(())
}

/// The package of CPU kernel implementations this backend offers.
#[must_use]
pub fn cpu_kernels() -> GKernelPackage {
    GKernelPackage::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AccelType;

    fn mat(w: i32, h: i32) -> Mat {
        Mat::zeros(h, w, cv_core::types::CV_8UC3).unwrap()
    }

    #[test]
    fn the_portability_macros_carry_their_cxx_spellings() {
        assert_eq!(CV_ALWAYS_INLINE, "#[inline(always)]");
        assert_eq!(CV_OVERRIDE, "", "Rust needs no `override`");
    }

    #[test]
    fn a_cpu_kernel_reads_its_inputs_and_writes_its_outputs() {
        let k = GCPUKernel::new("test.copy", |ctx| {
            let src = ctx.in_mat(0)?.clone();
            ctx.set_outputs(vec![src]);
            Ok(())
        });
        assert_eq!(k.id, "test.copy");
        let mut ctx = GCPUContext::new(vec![mat(8, 8)]);
        assert!(ctx.in_mat(0).is_ok());
        assert!(ctx.in_mat(1).is_err(), "there is only one input");
        k.apply(&mut ctx).unwrap();
        assert_eq!(ctx.outputs().len(), 1);
        assert_eq!(ctx.out_mat_r(0).unwrap().cols(), 8);
        assert!(ctx.out_mat_r(5).is_err());
        assert!(format!("{k:?}").contains("test.copy"));
    }

    #[test]
    fn a_fluid_kernels_window_decides_its_border() {
        let k = GFluidKernel::new("blur3", 3).unwrap();
        assert_eq!(k.border_size(), 1);
        assert_eq!(GFluidKernel::new("blur5", 5).unwrap().border_size(), 2);
        assert_eq!(GFluidKernel::new("point", 1).unwrap().border_size(), 0);
        // An even window has no centre row, so it cannot be centred.
        assert!(GFluidKernel::new("bad", 4).is_err());
        assert!(GFluidKernel::new("bad", 0).is_err());
    }

    #[test]
    fn a_compound_kernel_cannot_expand_to_itself() {
        let c = GCompoundKernel::new("blur_then_sobel", vec!["blur".into(), "sobel".into()])
            .unwrap();
        assert_eq!(c.expansion.len(), 2);
        assert!(GCompoundKernel::new("x", vec![]).is_err());
        assert!(
            GCompoundKernel::new("x", vec!["y".into(), "x".into()]).is_err(),
            "a self-expansion would not terminate"
        );
    }

    #[test]
    fn the_other_backend_kernels_carry_their_ids() {
        assert_eq!(GOCLKernel::new("blur").id, "blur");
        assert_eq!(GPlaidMLKernel::new("blur").id, "blur");
        let py = GPythonKernel::new("blur", "mymod.blur");
        assert_eq!(py.callable, "mymod.blur");
        let f = GPythonFunctor::new(py);
        assert_eq!(f.backend(), GBackend::Python);
        assert!(cpu_kernels().empty());

        let ctx = GOCLContext::new(vec![]);
        assert!(ctx.in_mat(0).is_err(), "no device buffers were provided");
    }

    #[test]
    fn a_view_needs_one_stride_per_plane() {
        let v = View::new(vec![vec![0u8; 16], vec![0u8; 8]], vec![4, 4]).unwrap();
        assert_eq!(v.planes.len(), 2);
        assert!(View::new(vec![vec![0u8; 16]], vec![4, 4]).is_err());
        assert!(View::new(vec![], vec![]).is_err());

        let d = crate::graph::GFrameDesc { fmt: crate::types::MediaFormat::Nv12, size: (8, 8) };
        let f = MediaFrame::new(d, v);
        assert_eq!(f.desc().size, (8, 8));
        assert_eq!(f.access().planes.len(), 2);
    }

    #[test]
    fn an_rmat_reports_the_descriptor_of_what_it_wraps() {
        let r = RMat::new(mat(12, 10));
        assert_eq!(r.desc().size, (12, 10));
        assert_eq!(r.desc().chan, 3);
        assert_eq!(r.access().cols(), 12);
    }

    #[test]
    fn a_queue_source_drains_in_order_and_rejects_a_mismatched_frame() {
        let mut s = QueueSourceBase::new(vec![mat(8, 8), mat(8, 8)]).unwrap();
        assert_eq!(s.len(), 2);
        assert!(!s.is_empty());
        assert_eq!(s.descr_of().size, (8, 8));
        assert!(s.push(mat(8, 8)).is_ok());
        assert!(s.push(mat(4, 4)).is_err(), "a different shape breaks the contract");
        assert_eq!(s.len(), 3);

        for _ in 0..3 {
            assert!(matches!(s.pull().unwrap(), Some(Data::Mat(_))));
        }
        assert!(s.pull().unwrap().is_none(), "the end is a None, not an error");
        assert!(s.is_empty());

        assert!(QueueSourceBase::new(vec![]).is_err());
        assert!(QueueSourceBase::new(vec![mat(8, 8), mat(4, 4)]).is_err());
        let q = QueueInput::new(QueueSourceBase::new(vec![mat(8, 8)]).unwrap());
        assert_eq!(q.source.len(), 1);
    }

    #[test]
    fn the_gstreamer_source_says_it_is_not_linked_rather_than_ending() {
        let mut s = GStreamerSource::new("videotestsrc").unwrap();
        let e = s.pull().unwrap_err();
        assert!(format!("{e}").contains("videotestsrc"), "{e}");
        assert!(GStreamerSource::new("").is_err());

        let p = GStreamerPipeline::new("videotestsrc ! videoconvert").unwrap();
        let sub = p.get_streaming_source("appsink0").unwrap();
        assert!(sub.pipeline.contains("appsink0"));
        assert!(p.get_streaming_source("").is_err());
        assert!(GStreamerPipeline::new("").is_err());
    }

    #[test]
    fn a_streaming_graph_pulls_until_its_source_is_empty() {
        let mut g = crate::GComputation::new();
        let inp = g.input();
        let out = g.cvt_color_bgr2gray(inp);
        g.set_output(out);
        let mut s = GStreamingCompiled::new(g.compile().unwrap());
        assert!(!s.running());
        assert!(s.pull().is_err(), "pull before start");
        assert!(s.start().is_err(), "start before setSource");

        s.set_source(Box::new(QueueSourceBase::new(vec![mat(8, 8), mat(8, 8)]).unwrap()));
        s.start().unwrap();
        assert!(s.running());
        assert!(s.pull().unwrap().is_some());
        assert!(s.pull().unwrap().is_some());
        assert!(s.pull().unwrap().is_none(), "the source drained");
        assert!(!s.running(), "draining stops the pipeline");

        s.set_source(Box::new(QueueSourceBase::new(vec![mat(8, 8)]).unwrap()));
        s.start().unwrap();
        s.stop();
        assert!(!s.running());
    }

    #[test]
    fn the_onevpl_devices_report_what_is_actually_available() {
        let d = create_host_device();
        assert_eq!(d.accel, AccelType::Host);
        assert_eq!(create_host_context().accel, AccelType::Host);
        // The accelerated paths are absent and say so by name.
        assert!(format!("{}", create_dx11_device("adapter0").unwrap_err()).contains("adapter0"));
        assert!(create_dx11_context().is_err());
        assert!(create_vaapi_device("renderD128").is_err());
        assert!(create_vaapi_context().is_err());

        let sel = get_default_device_selector();
        let devices = sel.select_devices();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].accel, AccelType::Host);
        assert_eq!(Device::new("x", AccelType::Host).name, "x");
        assert_eq!(Context::new(AccelType::Host), Context::new(AccelType::Host));
    }

    #[test]
    fn a_data_provider_streams_its_buffer_then_reports_empty() {
        let mut p = MemoryDataProvider::new(vec![7u8; 5000]);
        assert!(!p.empty());
        let mut out = Vec::new();
        assert!(p.fetch_bitstream_data(&mut out).unwrap());
        assert_eq!(out.len(), 4096, "one chunk");
        assert!(p.fetch_bitstream_data(&mut out).unwrap());
        assert_eq!(out.len(), 5000, "the remainder");
        assert!(!p.fetch_bitstream_data(&mut out).unwrap(), "nothing left");
        assert!(p.empty());
        assert!(MemoryDataProvider::default().empty());

        assert!(make_onevpl_src("clip.h264", &[]).is_err());
        let c = CfgParam::create("mfx.CodecID", "H264").unwrap();
        assert_eq!(c.value, "H264");
        assert!(CfgParam::create("", "x").is_err());
    }

    #[test]
    fn the_provider_exceptions_are_distinguishable() {
        let base = DataProviderException::new("broken");
        assert_eq!(base.what(), "broken");
        let sys = DataProviderSystemErrorException::new(2, "no such file");
        assert_eq!(sys.error_code, 2);
        assert_eq!(sys.base.what(), "no such file");
        // Unsupported and implementation errors are distinct types precisely so
        // a caller can retry one and not the other.
        let unsup = DataProviderUnsupportedException::new("AV1");
        let impl_err = DataProviderImplementationException::new("internal");
        assert_ne!(unsup.base, impl_err.base);
        assert_eq!(unsup.base.what(), "AV1");
    }

    #[test]
    fn every_inference_backend_takes_a_model_and_a_device() {
        let mut ie = OpenVINO::new("face.xml", "CPU").unwrap();
        ie.set("PERFORMANCE_HINT", "LATENCY");
        assert_eq!(ie.params.len(), 1);
        assert_eq!(OpenVINO::tag(), "org.opencv.dnn.ie");
        assert!(OpenVINO::new("", "CPU").is_err(), "a model path is required");

        assert!(CUDA::new("m.onnx", "0").is_ok());
        assert!(CoreML::new("m.onnx", "ALL").is_ok());
        assert!(DirectML::new("m.onnx", "0").is_ok());
        assert!(TensorRT::new("m.onnx", "0").is_ok());
        assert_eq!(CUDA::tag(), "org.opencv.dnn.onnx.cuda");
        assert_ne!(CoreML::tag(), DirectML::tag());
        assert_ne!(TensorRT::tag(), CUDA::tag());

        let py = PyParams::new("face", "m.xml", "ie").unwrap();
        assert_eq!(py.tag, "face");
        assert!(PyParams::new("", "m.xml", "ie").is_err());
    }

    #[test]
    fn a_fluid_buffer_keeps_only_its_capacity() {
        let mut b = Buffer::new(3, 8).unwrap();
        assert_eq!(b.length(), 0);
        for i in 0..5u8 {
            b.push(vec![i; 8]);
        }
        assert_eq!(b.length(), 3, "the ring holds only its capacity");
        // The oldest rows were evicted, so row 0 is the third pushed.
        assert_eq!(b.row(0).unwrap()[0], 2);
        assert_eq!(b.row(2).unwrap()[0], 4);
        assert!(b.row(3).is_none());
        assert!(Buffer::new(0, 8).is_err());
        assert!(Buffer::new(3, 0).is_err());
    }

    #[test]
    fn the_render_primitives_draw_onto_a_three_channel_image() {
        let mut img = mat(64, 64);
        let white = Scalar::all(255.0);
        let prims = vec![
            Prim::Rect(DrawRect { rect: Rect::new(4, 4, 20, 20), color: white, thick: 1 }),
            Prim::Circle(Circle { center: (40, 40), radius: 8, color: white, thick: 1 }),
            Prim::Line(Line { pt1: (0, 0), pt2: (63, 63), color: white, thick: 1 }),
            Prim::Poly(Poly {
                points: vec![(10, 50), (20, 60), (5, 60)],
                color: white,
                thick: 1,
            }),
            Prim::Text(Text {
                text: "hi".into(),
                org: (2, 30),
                ff: 0.4,
                color: white,
                bottom_left_origin: false,
            }),
        ];
        render_prims(&mut img, &prims).unwrap();
        let lit: usize = (0..64)
            .flat_map(|y| (0..64).map(move |x| (y, x)))
            .filter(|&(y, x)| img.at_f64(y, x, 0).unwrap() > 0.0)
            .count();
        assert!(lit > 50, "the primitives should have drawn something: {lit}");

        // A single-channel target is refused.
        let mut grey = Mat::zeros(8, 8, cv_core::types::CV_8UC1).unwrap();
        assert!(render_prims(&mut grey, &prims).is_err());
        // An image primitive carries no pixels, and says so.
        assert!(render_prims(
            &mut img,
            &[Prim::Image(Image { org: (0, 0), size: (4, 4), alpha: vec![1.0; 16] })]
        )
        .is_err());
    }

    #[test]
    fn a_mosaic_flattens_the_region_it_covers() {
        // A gradient, so a mosaic has something to average away.
        let mut img = Mat::zeros(32, 32, cv_core::types::CV_8UC3).unwrap();
        for y in 0..32 {
            for x in 0..32 {
                let px = cv_core::MatLikeMut::pixel_mut::<u8>(&mut img, y, x).unwrap();
                px.fill((x * 8) as u8);
            }
        }
        let before = img.at_f64(8, 8, 0).unwrap() - img.at_f64(8, 12, 0).unwrap();
        assert!(before.abs() > 10.0, "the gradient should vary across the region");

        render_prims(
            &mut img,
            &[Prim::Mosaic(Mosaic { mos: Rect::new(4, 4, 16, 16), cell_sz: 8, decim: 0 })],
        )
        .unwrap();
        // Inside one cell every pixel is now the same.
        let a = img.at_f64(6, 6, 0).unwrap();
        let b = img.at_f64(9, 9, 0).unwrap();
        assert!((a - b).abs() < 1e-9, "one cell should be flat: {a} vs {b}");
        // Outside the region nothing changed.
        assert_eq!(img.at_f64(30, 30, 0).unwrap(), 240.0);
    }

    #[test]
    fn the_small_value_types_carry_what_they_say() {
        assert_eq!(Score(0.9).0, 0.9);
        assert!(Score(0.9) > Score(0.5));
        let c = ColorCamera {
            params: ColorCameraParams { focal: (500.0, 500.0), principal: (320.0, 240.0) },
        };
        assert_eq!(c.params.focal.0, 500.0);
        assert_eq!(ColorCamera::default().params.focal, (0.0, 0.0));
        assert!(Text::default().text.is_empty());
        assert_eq!(Mosaic::default().cell_sz, 0);
        assert!(Image::default().alpha.is_empty());
        assert!(Poly::default().points.is_empty());
        assert_eq!(DrawRect::default().thick, 0);
        assert_eq!(Line::default().thick, 0);
        assert_eq!(Circle::default().radius, 0);
    }
}
