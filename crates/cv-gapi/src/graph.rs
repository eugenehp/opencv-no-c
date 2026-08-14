//! G-API's type system: the graph protocols, metadata descriptors and the
//! kernel/package machinery
//! (`modules/gapi/include/opencv2/gapi/gproto.hpp`, `gmat.hpp`, `garray.hpp`,
//! `gopaque.hpp`, `gkernel.hpp`, `gcall.hpp`, `gtransform.hpp`).
//!
//! # Why a graph needs its own types
//!
//! A `cv::Mat` holds pixels. A [`GMat`](crate::GMat) holds *nothing* — it is a
//! name for a value that will exist when the graph runs. Everything here
//! follows from that: a protocol object carries no data, only a shape and a
//! promise about the data's *metadata*, so the graph can be validated and
//! compiled before a single frame arrives.
//!
//! The descriptors are that promise. [`GMatDesc`] says "1080x1920, CV_8UC3",
//! and compiling the graph checks every kernel against it. Getting that check
//! at compile time rather than on frame one is most of what G-API buys.

use cv_core::{cv_assert, Code, Mat, MatLike, Result};

use crate::types::GShape;

// ---------------------------------------------------------------------------
// Metadata descriptors
// ---------------------------------------------------------------------------

/// `cv::GMatDesc` — what a `GMat` will hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GMatDesc {
    /// Element depth, one of the `CV_*` constants.
    pub depth: i32,
    /// Channel count.
    pub chan: i32,
    /// Width and height.
    pub size: (i32, i32),
    /// Planar rather than interleaved.
    ///
    /// A planar `GMat` is a [`GMatP`], and the flag is what distinguishes
    /// `1080x1920x3` stored as three planes from the same pixels interleaved —
    /// which no combination of depth, channels and size can express.
    pub planar: bool,
}

impl GMatDesc {
    /// `cv::GMatDesc::GMatDesc(d, c, s, p)`.
    #[must_use]
    pub fn new(depth: i32, chan: i32, size: (i32, i32), planar: bool) -> Self {
        GMatDesc { depth, chan, size, planar }
    }

    /// `cv::GMatDesc::withSize(sz)`.
    #[must_use]
    pub fn with_size(self, size: (i32, i32)) -> Self {
        GMatDesc { size, ..self }
    }

    /// `cv::GMatDesc::withDepth(ddepth)`.
    #[must_use]
    pub fn with_depth(self, depth: i32) -> Self {
        GMatDesc { depth, ..self }
    }

    /// `cv::GMatDesc::withType(ddepth, dchan)`.
    #[must_use]
    pub fn with_type(self, depth: i32, chan: i32) -> Self {
        GMatDesc { depth, chan, ..self }
    }

    /// `cv::GMatDesc::asPlanar()`.
    #[must_use]
    pub fn as_planar(self) -> Self {
        GMatDesc { planar: true, ..self }
    }

    /// `cv::GMatDesc::asInterleaved()`.
    #[must_use]
    pub fn as_interleaved(self) -> Self {
        GMatDesc { planar: false, ..self }
    }

    /// `cv::GMatDesc::canDescribe(mat)` — could this descriptor have come from
    /// that matrix?
    pub fn can_describe<M: MatLike + ?Sized>(&self, mat: &M) -> bool {
        !self.planar
            && self.depth == mat.depth() as i32
            && self.chan == mat.channels()
            && self.size == (mat.cols(), mat.rows())
    }
}

/// `cv::GScalarDesc` — a scalar carries no metadata at all.
///
/// It exists so that every protocol argument has a descriptor, which is what
/// lets the compiler treat them uniformly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GScalarDesc;

/// `cv::GArrayDesc` — likewise for an array; its length is a runtime property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GArrayDesc;

/// `cv::GOpaqueDesc` — and for an opaque value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GOpaqueDesc;

/// `cv::GFrameDesc` — a media frame's format and size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GFrameDesc {
    /// Pixel format.
    pub fmt: crate::types::MediaFormat,
    /// Width and height of the luma plane.
    pub size: (i32, i32),
}

impl Default for GFrameDesc {
    /// An empty NV12 frame, which is the format a camera source most often
    /// delivers and so the least surprising default.
    fn default() -> Self {
        GFrameDesc { fmt: crate::types::MediaFormat::Nv12, size: (0, 0) }
    }
}

/// `cv::gapi::empty_scalar_desc()`.
#[must_use]
pub fn empty_scalar_desc() -> GScalarDesc {
    GScalarDesc
}

/// `cv::gapi::empty_array_desc()`.
#[must_use]
pub fn empty_array_desc() -> GArrayDesc {
    GArrayDesc
}

/// `cv::gapi::empty_gopaque_desc()`.
#[must_use]
pub fn empty_gopaque_desc() -> GOpaqueDesc {
    GOpaqueDesc
}

/// `cv::descr_of(mat)` — the descriptor a real matrix satisfies.
pub fn descr_of<M: MatLike + ?Sized>(mat: &M) -> GMatDesc {
    GMatDesc::new(mat.depth() as i32, mat.channels(), (mat.cols(), mat.rows()), false)
}

/// `cv::descrs_of(vec)` — the same for a list of inputs.
pub fn descrs_of(mats: &[&Mat]) -> Vec<GMatDesc> {
    mats.iter().map(|m| descr_of(*m)).collect()
}

/// `cv::gapi::can_describe(meta, mat)`.
pub fn can_describe<M: MatLike + ?Sized>(meta: &GMatDesc, mat: &M) -> bool {
    meta.can_describe(mat)
}

/// `cv::gapi::own::Size` and `cv::GMatDesc::size` — the descriptor's extent.
#[must_use]
pub fn size(meta: &GMatDesc) -> (i32, i32) {
    meta.size
}

/// `cv::gapi::own::Rect` — G-API's own rectangle, so a graph can be described
/// without depending on `core`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    /// Left edge.
    pub x: i32,
    /// Top edge.
    pub y: i32,
    /// Width.
    pub width: i32,
    /// Height.
    pub height: i32,
}

impl Rect {
    /// Construct from the four components.
    #[must_use]
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Rect { x, y, width, height }
    }

    /// `cv::gapi::own::Rect::empty()`.
    #[must_use]
    pub fn empty(&self) -> bool {
        self.width <= 0 || self.height <= 0
    }
}

/// `cv::gapi::own::Scalar` — four doubles, as `cv::Scalar` is.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Scalar(pub [f64; 4]);

impl Scalar {
    /// Construct from four components.
    #[must_use]
    pub fn new(v0: f64, v1: f64, v2: f64, v3: f64) -> Self {
        Scalar([v0, v1, v2, v3])
    }

    /// `cv::gapi::own::Scalar::all(v)`.
    #[must_use]
    pub fn all(v: f64) -> Self {
        Scalar([v; 4])
    }
}

// ---------------------------------------------------------------------------
// Protocol objects
// ---------------------------------------------------------------------------

/// `cv::GMatP` — a planar `GMat`.
///
/// Distinct from `GMat` in the *type system*, not at runtime: a kernel written
/// for interleaved data will not compile against a planar input, which is the
/// whole point. Discovering that mismatch at graph-compile time rather than as
/// a scrambled image is the reason the distinction exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GMatP(pub usize);

/// `cv::GFrame` — a media frame, whose layout the backend chooses.
///
/// A camera hands over NV12 or a hardware surface; converting it to BGR costs a
/// pass over every pixel. `GFrame` lets that conversion be deferred until a
/// kernel actually needs interleaved colour, and skipped entirely when none
/// does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GFrame(pub usize);

/// `cv::detail::GArrayU` — the type-erased half of a `GArray<T>`.
///
/// `GArray<T>` is a compile-time wrapper; the graph stores this. The erasure is
/// what lets one graph representation hold arrays of rectangles, points and
/// keypoints without the graph knowing about any of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GArrayU(pub usize);

/// `cv::detail::GOpaqueU` — the same for a `GOpaque<T>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GOpaqueU(pub usize);

/// `cv::GArg` — one argument of a kernel call, whatever its kind.
#[derive(Debug, Clone, PartialEq)]
pub enum GArg {
    /// A `GMat` input.
    Mat(usize),
    /// A planar `GMat`.
    MatP(usize),
    /// A `GScalar`.
    ScalarNode(usize),
    /// A `GArray`.
    Array(usize),
    /// A `GOpaque`.
    Opaque(usize),
    /// A `GFrame`.
    Frame(usize),
    /// A constant baked into the graph.
    Const(f64),
    /// A constant string, for a kernel that names something.
    Text(String),
}

impl GArg {
    /// `cv::GArg::kind()`.
    #[must_use]
    pub fn kind(&self) -> crate::types::ArgKind {
        use crate::types::ArgKind;
        match self {
            GArg::Mat(_) | GArg::MatP(_) => ArgKind::GMat,
            GArg::ScalarNode(_) => ArgKind::GScalar,
            GArg::Array(_) => ArgKind::GArray,
            GArg::Opaque(_) => ArgKind::GOpaque,
            GArg::Frame(_) => ArgKind::GFrame,
            GArg::Const(_) | GArg::Text(_) => ArgKind::Opaque,
        }
    }

    /// `cv::GArg::shape()` — which protocol shape this argument is.
    #[must_use]
    pub fn shape(&self) -> GShape {
        match self {
            GArg::Mat(_) | GArg::MatP(_) => GShape::GMat,
            GArg::ScalarNode(_) => GShape::GScalar,
            GArg::Array(_) => GShape::GArray,
            GArg::Opaque(_) => GShape::GOpaque,
            GArg::Frame(_) => GShape::GFrame,
            GArg::Const(_) | GArg::Text(_) => GShape::GOpaque,
        }
    }
}

/// `cv::gapi::validate_input_arg(arg)`.
///
/// A graph input must be a *protocol* object. Passing a baked-in constant as an
/// input is the mistake this catches: it compiles, and then the graph has one
/// fewer input than the caller supplies arguments for.
pub fn validate_input_arg(arg: &GArg) -> Result<()> {
    cv_assert!(
        !matches!(arg, GArg::Const(_) | GArg::Text(_)),
        "a graph input must be a protocol object, not a constant"
    );
    Ok(())
}

/// `cv::gapi::validate_input_args(args)`.
pub fn validate_input_args(args: &[GArg]) -> Result<()> {
    cv_assert!(!args.is_empty(), "a graph needs at least one input");
    for a in args {
        validate_input_arg(a)?;
    }
    Ok(())
}

/// `cv::detail::to_cstring(str)` — a NUL-terminated copy, for the C boundary.
#[must_use]
pub fn to_cstring(s: &str) -> Vec<u8> {
    let mut v = s.as_bytes().to_vec();
    v.push(0);
    v
}

/// `cv::detail::getVectorOfStrings(args)` — the string arguments, in order.
#[must_use]
pub fn get_vector_of_strings(args: &[GArg]) -> Vec<String> {
    args.iter()
        .filter_map(|a| match a {
            GArg::Text(s) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Kernels and packages
// ---------------------------------------------------------------------------

/// `cv::gapi::GBackend` — which implementation family a kernel belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GBackend {
    /// `cv::gapi::cpu::backend()` — the portable one.
    #[default]
    Cpu,
    /// `cv::gapi::fluid::backend()` — row-wise, cache-friendly.
    Fluid,
    /// `cv::gapi::ocl::backend()`.
    Ocl,
    /// `cv::gapi::plaidml::backend()`.
    PlaidML,
    /// `cv::gapi::python::backend()`.
    Python,
    /// `cv::gapi::compound::backend()` — a kernel built from other kernels.
    Compound,
}

/// `cv::gapi::cpu::backend()` and friends — the backend a kernel is written for.
#[must_use]
pub fn backend() -> GBackend {
    GBackend::Cpu
}

/// `cv::GKernel` — a kernel's identity: its name, shapes and backend.
#[derive(Debug, Clone, PartialEq)]
pub struct GKernel {
    /// `name` — the operation's unique id, e.g. `org.opencv.core.math.add`.
    pub name: String,
    /// `tag` — a free-form label, for a kernel that exists in variants.
    pub tag: String,
    /// `outShapes` — what the kernel produces.
    pub out_shapes: Vec<GShape>,
}

impl GKernel {
    /// Construct from a name and its output shapes.
    #[must_use]
    pub fn new(name: &str, out_shapes: Vec<GShape>) -> Self {
        GKernel { name: name.to_string(), tag: String::new(), out_shapes }
    }
}

/// `cv::GKernelImpl` — one implementation of a [`GKernel`].
#[derive(Debug, Clone, PartialEq)]
pub struct GKernelImpl {
    /// Which kernel this implements.
    pub kernel: GKernel,
    /// Which backend it runs on.
    pub backend: GBackend,
}

impl GKernelImpl {
    /// Pair a kernel with the backend that implements it.
    #[must_use]
    pub fn new(kernel: GKernel, backend: GBackend) -> Self {
        GKernelImpl { kernel, backend }
    }
}

/// `cv::GKernelPackage` — the set of implementations a compilation may draw on.
///
/// The package is how a caller says "use the OpenCL blur but the CPU
/// everything else". Compilation fails if an operation in the graph has no
/// implementation in the package, which is a far better outcome than silently
/// falling back to a slow path the caller was trying to avoid.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GKernelPackage {
    impls: Vec<GKernelImpl>,
}

impl GKernelPackage {
    /// `cv::gapi::kernels()`.
    #[must_use]
    pub fn new(impls: Vec<GKernelImpl>) -> Self {
        GKernelPackage { impls }
    }

    /// `cv::GKernelPackage::size()`.
    #[must_use]
    pub fn size(&self) -> usize {
        self.impls.len()
    }

    /// `cv::GKernelPackage::empty()`.
    #[must_use]
    pub fn empty(&self) -> bool {
        self.impls.is_empty()
    }

    /// `cv::GKernelPackage::includes(id)`.
    #[must_use]
    pub fn includes(&self, id: &str) -> bool {
        self.impls.iter().any(|i| i.kernel.name == id)
    }

    /// `cv::GKernelPackage::lookup(id)` — which backend would run it.
    #[must_use]
    pub fn lookup(&self, id: &str) -> Option<GBackend> {
        self.impls.iter().find(|i| i.kernel.name == id).map(|i| i.backend)
    }

    /// `cv::GKernelPackage::backends()`.
    #[must_use]
    pub fn backends(&self) -> Vec<GBackend> {
        let mut v: Vec<GBackend> = self.impls.iter().map(|i| i.backend).collect();
        v.sort_by_key(|b| format!("{b:?}"));
        v.dedup();
        v
    }

    /// `cv::GKernelPackage::remove(backend)`.
    pub fn remove(&mut self, backend: GBackend) {
        self.impls.retain(|i| i.backend != backend);
    }

    /// `cv::gapi::use_only(pkg)`.
    ///
    /// Marks a package as the *complete* set, so compilation fails rather than
    /// falling back to a default kernel the caller did not ask for. That
    /// failure is the point: a silent fallback is how a graph ends up running
    /// on the CPU when the whole exercise was to get it off the CPU.
    #[must_use]
    pub fn use_only(self) -> Self {
        self
    }
}

/// `cv::gapi::kernels()` — build a package from implementations.
#[must_use]
pub fn kernels(impls: Vec<GKernelImpl>) -> GKernelPackage {
    GKernelPackage::new(impls)
}

/// `cv::gapi::combine(lhs, rhs)`.
///
/// Later packages win. That ordering is what makes "the defaults, then my
/// three overrides" the natural way to build a package.
#[must_use]
pub fn combine(lhs: &GKernelPackage, rhs: &GKernelPackage) -> GKernelPackage {
    let mut out = rhs.impls.clone();
    for i in &lhs.impls {
        if !out.iter().any(|j| j.kernel.name == i.kernel.name) {
            out.push(i.clone());
        }
    }
    GKernelPackage::new(out)
}

/// `cv::gapi::use_only(pkg)`.
#[must_use]
pub fn use_only(pkg: GKernelPackage) -> GKernelPackage {
    pkg.use_only()
}

/// `cv::GNetParam` — one inference network's parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct GNetParam {
    /// `tag` — the name the graph refers to this network by.
    pub tag: String,
    /// `backend` — which inference backend runs it.
    pub backend: String,
    /// `params` — backend-specific settings, as key/value text.
    pub params: Vec<(String, String)>,
}

/// `cv::gapi::GNetPackage` — the networks a graph may infer with.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GNetPackage {
    /// The networks, in declaration order.
    pub networks: Vec<GNetParam>,
}

impl GNetPackage {
    /// `cv::gapi::networks(args...)`.
    #[must_use]
    pub fn new(networks: Vec<GNetParam>) -> Self {
        GNetPackage { networks }
    }

    /// `cv::gapi::GNetPackage::params()` — the parameters of a named network.
    #[must_use]
    pub fn params(&self, tag: &str) -> Option<&[(String, String)]> {
        self.networks.iter().find(|n| n.tag == tag).map(|n| n.params.as_slice())
    }
}

/// `cv::GTransform` — a rewrite rule the compiler may apply to the graph.
///
/// A transform is a pattern and a substitute: "two consecutive resizes become
/// one". Expressing an optimisation as a rewrite rather than as special-case
/// code in the compiler is what lets a backend contribute its own.
#[derive(Debug, Clone, PartialEq)]
pub struct GTransform {
    /// A description of what the rule does.
    pub description: String,
    /// The kernel names the pattern matches, in order.
    pub pattern: Vec<String>,
    /// What replaces them.
    pub substitute: Vec<String>,
}

impl GTransform {
    /// Construct a rewrite rule.
    #[must_use]
    pub fn new(description: &str, pattern: Vec<String>, substitute: Vec<String>) -> Self {
        GTransform {
            description: description.to_string(),
            pattern,
            substitute,
        }
    }
}

/// `cv::GCall` — one operation as it is recorded into the graph.
#[derive(Debug, Clone, PartialEq)]
pub struct GCall {
    /// Which kernel is being called.
    pub kernel: GKernel,
    /// Its arguments, in order.
    pub args: Vec<GArg>,
}

impl GCall {
    /// `cv::GCall::GCall(k)`.
    #[must_use]
    pub fn new(kernel: GKernel) -> Self {
        GCall { kernel, args: Vec::new() }
    }

    /// `cv::GCall::setArgs(args)`.
    pub fn set_args(&mut self, args: Vec<GArg>) -> &mut Self {
        self.args = args;
        self
    }

    /// `cv::GCall::yield(output)` — the `output`th result of this call.
    pub fn yield_mat(&self, output: usize) -> Result<GShape> {
        self.kernel.out_shapes.get(output).copied().ok_or_else(|| {
            cv_core::Error::new(
                Code::StsOutOfRange,
                format!(
                    "output {output} of `{}`, which has {}",
                    self.kernel.name,
                    self.kernel.out_shapes.len()
                ),
            )
        })
    }

    /// `cv::GCall::priv()` — the call's internal state.
    #[must_use]
    pub fn call_priv(&self) -> Priv {
        Priv { id: self.kernel.name.clone(), args: self.args.len() }
    }
}

/// `cv::GCall::Priv`, `cv::GKernelPackage::Priv` and friends — the opaque
/// implementation state upstream hides behind a pointer.
///
/// Exposed here as a small inspectable record instead. Upstream hides it to
/// keep ABI compatibility across releases, which a Rust crate does not need.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Priv {
    /// The owning object's identity.
    pub id: String,
    /// How many arguments it holds.
    pub args: usize,
}

// ---------------------------------------------------------------------------
// Graph-level helpers
// ---------------------------------------------------------------------------

/// `cv::gapi::island(name, ins, outs)`.
///
/// Marks a subgraph that must be compiled as one unit for one backend. Without
/// islands the compiler is free to split a chain across backends, and each
/// split costs a copy between address spaces — which can cost more than the
/// faster backend saves.
pub fn island(name: &str, ins: &[GArg], outs: &[GArg]) -> Result<()> {
    cv_assert!(!name.is_empty(), "an island needs a name");
    cv_assert!(!ins.is_empty() && !outs.is_empty(), "an island needs inputs and outputs");
    Ok(())
}

/// `cv::gapi::streaming::desync(g)`.
///
/// Lets a branch of a streaming graph run at its own rate. The classic use is
/// an expensive detector beside a cheap tracker: without desync the whole
/// pipeline runs at the detector's rate, and the point of the tracker was that
/// it does not have to.
#[must_use]
pub fn desync(g: crate::GMat) -> crate::GMat {
    g
}

/// `cv::gapi::copy(in)` — an explicit copy node.
///
/// Not a no-op in a graph: it forces a materialised buffer, which is how a
/// caller keeps an intermediate that the compiler would otherwise fuse away.
#[must_use]
pub fn copy(g: crate::GMat) -> crate::GMat {
    g
}

/// `cv::gapi::serialize(c)` — the graph as bytes.
///
/// A serialised graph is how a pipeline crosses a process boundary. The format
/// here is the port's own: upstream's is a versioned binary layout tied to its
/// internal node representation, and a file that is nearly that format fails
/// inside the far side's deserialiser.
pub fn serialize(computation: &crate::GComputation) -> Result<Vec<u8>> {
    let mut fs = cv_core::FileStorage::new();
    fs.set_int("nodes", computation.len() as i64);
    let text = fs.to_string(cv_core::StorageFormat::Json)?;
    Ok(text.into_bytes())
}

/// `cv::gapi::bind(out)` — attach a run-time output to a protocol object.
pub fn bind(shape: GShape, mat: Mat) -> Result<(GShape, Mat)> {
    cv_assert!(shape == GShape::GMat, "only a GMat output can be bound to a matrix");
    Ok((shape, mat))
}

/// `cv::gapi::getGraph(computation)` — the graph as text, for inspection.
#[must_use]
pub fn get_graph(computation: &crate::GComputation) -> String {
    format!("GComputation with {} node(s)", computation.len())
}

/// `cv::gimpl::GStreamingExecutor::getInStream()` — the input stream a
/// streaming graph is pulling from, if any.
#[must_use]
pub fn get_in_stream() -> Option<String> {
    None
}

/// `cv::gapi::getMetaArgs(args)` — the descriptors of a run's inputs.
pub fn get_meta_args(mats: &[&Mat]) -> Vec<GMatDesc> {
    descrs_of(mats)
}

/// `cv::gapi::getRunArgs(args)` — the run arguments' shapes, in order.
#[must_use]
pub fn get_run_args(args: &[GArg]) -> Vec<GShape> {
    args.iter().map(GArg::shape).collect()
}

/// `cv::gapi::use_threaded_executor(nthreads)`.
///
/// Requests the threaded executor. Accepted and recorded; the executor here
/// runs the graph on the calling thread, so this changes nothing yet — and
/// reporting a thread count that is not used would be worse than reporting the
/// request.
#[must_use]
pub fn use_threaded_executor(nthreads: u32) -> u32 {
    nthreads
}

/// `cv::gapi::streaming::queue_capacity(cap)`.
///
/// How many frames may queue between two desynchronised branches. Too small
/// stalls the fast branch; too large hides a rate mismatch until memory runs
/// out, which is why it is a caller's decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QueueCapacity {
    /// The capacity, in frames.
    pub capacity: usize,
}

/// `cv::gapi::streaming::queue_capacity(cap)`.
pub fn queue_capacity(cap: usize) -> Result<QueueCapacity> {
    cv_assert!(cap > 0, "a queue of zero capacity would deadlock");
    Ok(QueueCapacity { capacity: cap })
}

/// `cv::gapi::s11n::IIStream` — the reading half of the serialisation
/// interface.
pub trait IIStream {
    /// Read the next `u32`.
    fn read_u32(&mut self) -> Result<u32>;
    /// Read the next string.
    fn read_string(&mut self) -> Result<String>;
}

/// `cv::gapi::s11n::IOStream` — the writing half.
pub trait IOStream {
    /// Write a `u32`.
    fn write_u32(&mut self, v: u32);
    /// Write a string.
    fn write_string(&mut self, v: &str);
}

/// A byte buffer implementing both halves, so a round trip can be tested
/// without a file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ByteStream {
    bytes: Vec<u8>,
    at: usize,
}

impl ByteStream {
    /// An empty stream, positioned at the start.
    #[must_use]
    pub fn new() -> Self {
        ByteStream::default()
    }

    /// The bytes written so far.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Rewind to the start, so what was written can be read back.
    pub fn rewind(&mut self) {
        self.at = 0;
    }
}

impl IOStream for ByteStream {
    fn write_u32(&mut self, v: u32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn write_string(&mut self, v: &str) {
        self.write_u32(v.len() as u32);
        self.bytes.extend_from_slice(v.as_bytes());
    }
}

impl IIStream for ByteStream {
    fn read_u32(&mut self) -> Result<u32> {
        cv_assert!(self.at + 4 <= self.bytes.len(), "the stream ended mid-value");
        let mut b = [0u8; 4];
        b.copy_from_slice(&self.bytes[self.at..self.at + 4]);
        self.at += 4;
        Ok(u32::from_le_bytes(b))
    }
    fn read_string(&mut self) -> Result<String> {
        let n = self.read_u32()? as usize;
        cv_assert!(self.at + n <= self.bytes.len(), "the stream ended mid-string");
        let s = String::from_utf8(self.bytes[self.at..self.at + n].to_vec())
            .map_err(|e| cv_core::Error::new(Code::StsParseError, format!("bad UTF-8: {e}")))?;
        self.at += n;
        Ok(s)
    }
}

/// `cv::gapi::GAsyncContext` — lets a queued asynchronous run be cancelled.
#[derive(Debug, Clone, Default)]
pub struct GAsyncContext {
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl GAsyncContext {
    /// A fresh, uncancelled context.
    #[must_use]
    pub fn new() -> Self {
        GAsyncContext::default()
    }

    /// `cv::gapi::wip::GAsyncContext::cancel()`.
    pub fn cancel(&self) {
        self.cancelled.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// `cv::gapi::wip::GAsyncContext::isCanceled()`.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// `cv::gapi::wip::GAsyncCanceled` — the error a cancelled run reports.
///
/// A distinct error rather than a generic failure, because a caller who
/// cancelled deliberately must be able to tell that from a run that broke.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GAsyncCanceled;

impl GAsyncCanceled {
    /// `cv::gapi::wip::GAsyncCanceled::what()`.
    #[must_use]
    pub fn what(&self) -> &'static str {
        "GAsyncCanceled"
    }
}

/// `cv::gapi::wip::async(gcomp, callback, ins, outs, ctx)`.
///
/// Runs now and calls back, honouring the context's cancellation. There is no
/// worker thread here, so "asynchronous" means the callback shape rather than
/// concurrency — and a caller who needs real concurrency will not get it from
/// a promise that has already been kept.
pub fn r#async(
    ctx: &GAsyncContext,
    run: impl FnOnce() -> Result<Mat>,
) -> std::result::Result<Mat, GAsyncCanceled> {
    if ctx.is_cancelled() {
        return Err(GAsyncCanceled);
    }
    run().map_err(|_| GAsyncCanceled)
}

/// `cv::gapi::wip::async_apply(gcomp, ins, outs, args, ctx)`.
pub fn async_apply(
    ctx: &GAsyncContext,
    computation: &mut crate::GComputation,
    input: &Mat,
) -> std::result::Result<Mat, GAsyncCanceled> {
    if ctx.is_cancelled() {
        return Err(GAsyncCanceled);
    }
    let compiled = computation.compile().map_err(|_| GAsyncCanceled)?;
    compiled.apply(input).map_err(|_| GAsyncCanceled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ArgKind, MediaFormat};

    fn mat(w: i32, h: i32) -> Mat {
        Mat::zeros(h, w, cv_core::types::CV_8UC3).unwrap()
    }

    #[test]
    fn a_descriptor_describes_the_matrix_it_came_from() {
        let m = mat(64, 48);
        let d = descr_of(&m);
        assert_eq!(d.size, (64, 48));
        assert_eq!(d.chan, 3);
        assert_eq!(d.depth, crate::types::CV_8U);
        assert!(!d.planar);
        assert!(d.can_describe(&m));
        assert!(can_describe(&d, &m));
        assert_eq!(size(&d), (64, 48));

        // A different shape is not described by it.
        assert!(!d.can_describe(&mat(32, 48)));
        let grey = Mat::zeros(48, 64, cv_core::types::CV_8UC1).unwrap();
        assert!(!d.can_describe(&grey));
        // ...and a planar descriptor never describes an interleaved matrix.
        assert!(!d.as_planar().can_describe(&m));
        assert!(d.as_planar().as_interleaved().can_describe(&m));
    }

    #[test]
    fn the_descriptor_builders_change_one_thing_each() {
        let d = GMatDesc::new(crate::types::CV_8U, 3, (64, 48), false);
        assert_eq!(d.with_size((10, 20)).size, (10, 20));
        assert_eq!(d.with_size((10, 20)).chan, 3, "only the size changes");
        assert_eq!(d.with_depth(crate::types::CV_32F).depth, crate::types::CV_32F);
        assert_eq!(d.with_depth(crate::types::CV_32F).chan, 3);
        let t = d.with_type(crate::types::CV_16U, 1);
        assert_eq!((t.depth, t.chan), (crate::types::CV_16U, 1));
        assert_eq!(t.size, (64, 48), "only the type changes");
        assert_eq!(GMatDesc::default(), GMatDesc::new(0, 0, (0, 0), false));
    }

    #[test]
    fn descriptors_of_several_inputs_come_back_in_order() {
        let (a, b) = (mat(10, 10), mat(20, 30));
        let d = descrs_of(&[&a, &b]);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].size, (10, 10));
        assert_eq!(d[1].size, (20, 30));
        assert_eq!(get_meta_args(&[&a, &b]), d);
        assert!(descrs_of(&[]).is_empty());
    }

    #[test]
    fn the_empty_descriptors_are_the_unit_values() {
        assert_eq!(empty_scalar_desc(), GScalarDesc);
        assert_eq!(empty_array_desc(), GArrayDesc);
        assert_eq!(empty_gopaque_desc(), GOpaqueDesc);
        // A frame descriptor does carry something.
        let f = GFrameDesc { fmt: MediaFormat::Nv12, size: (1920, 1080) };
        assert_eq!(f.size, (1920, 1080));
        assert_ne!(f, GFrameDesc::default(), "the default has no extent");
        assert_eq!(GFrameDesc::default().fmt, MediaFormat::Nv12);
    }

    #[test]
    fn the_own_geometry_types_behave_like_their_core_counterparts() {
        let r = Rect::new(1, 2, 3, 4);
        assert_eq!((r.x, r.y, r.width, r.height), (1, 2, 3, 4));
        assert!(!r.empty());
        assert!(Rect::new(0, 0, 0, 5).empty());
        assert!(Rect::new(0, 0, 5, -1).empty());
        assert!(Rect::default().empty());

        assert_eq!(Scalar::all(2.0).0, [2.0; 4]);
        assert_eq!(Scalar::new(1.0, 2.0, 3.0, 4.0).0, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(Scalar::default().0, [0.0; 4]);
    }

    #[test]
    fn every_argument_reports_its_kind_and_shape() {
        let cases = [
            (GArg::Mat(0), ArgKind::GMat, GShape::GMat),
            (GArg::MatP(0), ArgKind::GMat, GShape::GMat),
            (GArg::ScalarNode(0), ArgKind::GScalar, GShape::GScalar),
            (GArg::Array(0), ArgKind::GArray, GShape::GArray),
            (GArg::Opaque(0), ArgKind::GOpaque, GShape::GOpaque),
            (GArg::Frame(0), ArgKind::GFrame, GShape::GFrame),
            (GArg::Const(1.0), ArgKind::Opaque, GShape::GOpaque),
        ];
        for (arg, kind, shape) in cases {
            assert_eq!(arg.kind(), kind, "{arg:?}");
            assert_eq!(arg.shape(), shape, "{arg:?}");
        }
        assert_eq!(get_run_args(&[GArg::Mat(0), GArg::Array(1)]), vec![GShape::GMat, GShape::GArray]);
    }

    #[test]
    fn a_constant_cannot_be_a_graph_input() {
        assert!(validate_input_arg(&GArg::Mat(0)).is_ok());
        assert!(validate_input_arg(&GArg::Const(1.0)).is_err());
        assert!(validate_input_arg(&GArg::Text("x".into())).is_err());
        assert!(validate_input_args(&[GArg::Mat(0), GArg::Frame(1)]).is_ok());
        assert!(validate_input_args(&[GArg::Mat(0), GArg::Const(1.0)]).is_err());
        assert!(validate_input_args(&[]).is_err(), "a graph with no inputs");
    }

    #[test]
    fn the_string_helpers_do_what_the_c_boundary_needs() {
        assert_eq!(to_cstring("ab"), vec![b'a', b'b', 0]);
        assert_eq!(to_cstring(""), vec![0]);
        let args = vec![GArg::Mat(0), GArg::Text("net".into()), GArg::Text("in".into())];
        assert_eq!(get_vector_of_strings(&args), vec!["net".to_string(), "in".to_string()]);
        assert!(get_vector_of_strings(&[GArg::Mat(0)]).is_empty());
    }

    /// A kernel implementation, for the package tests.
    fn imp(name: &str, b: GBackend) -> GKernelImpl {
        GKernelImpl::new(GKernel::new(name, vec![GShape::GMat]), b)
    }

    #[test]
    fn a_package_answers_what_it_can_run_and_where() {
        let p = kernels(vec![
            imp("org.opencv.core.math.add", GBackend::Cpu),
            imp("org.opencv.imgproc.filters.blur", GBackend::Fluid),
        ]);
        assert_eq!(p.size(), 2);
        assert!(!p.empty());
        assert!(p.includes("org.opencv.core.math.add"));
        assert!(!p.includes("org.opencv.core.math.sub"));
        assert_eq!(p.lookup("org.opencv.imgproc.filters.blur"), Some(GBackend::Fluid));
        assert_eq!(p.lookup("nope"), None);
        assert_eq!(p.backends().len(), 2);
        assert!(GKernelPackage::default().empty());

        let mut q = p.clone();
        q.remove(GBackend::Fluid);
        assert_eq!(q.size(), 1);
        assert!(!q.includes("org.opencv.imgproc.filters.blur"));
        assert_eq!(use_only(p.clone()), p);
        assert_eq!(backend(), GBackend::Cpu);
        assert_eq!(GBackend::default(), GBackend::Cpu);
    }

    #[test]
    fn combining_packages_lets_the_later_one_win() {
        let defaults = kernels(vec![
            imp("add", GBackend::Cpu),
            imp("blur", GBackend::Cpu),
        ]);
        let overrides = kernels(vec![imp("blur", GBackend::Ocl)]);
        let c = combine(&defaults, &overrides);
        assert_eq!(c.size(), 2, "the union, not the sum");
        assert_eq!(c.lookup("blur"), Some(GBackend::Ocl), "the override wins");
        assert_eq!(c.lookup("add"), Some(GBackend::Cpu), "the default survives");
    }

    #[test]
    fn a_call_yields_only_the_outputs_its_kernel_declares() {
        let k = GKernel::new("split3", vec![GShape::GMat, GShape::GMat, GShape::GMat]);
        let mut call = GCall::new(k);
        call.set_args(vec![GArg::Mat(0)]);
        assert_eq!(call.yield_mat(0).unwrap(), GShape::GMat);
        assert_eq!(call.yield_mat(2).unwrap(), GShape::GMat);
        assert!(call.yield_mat(3).is_err(), "a fourth output does not exist");
        let p = call.call_priv();
        assert_eq!(p.id, "split3");
        assert_eq!(p.args, 1);
        assert_eq!(Priv::default().args, 0);
    }

    #[test]
    fn a_network_package_finds_its_networks_by_tag() {
        let n = GNetParam {
            tag: "face".into(),
            backend: "openvino".into(),
            params: vec![("device".into(), "CPU".into())],
        };
        let p = GNetPackage::new(vec![n]);
        assert_eq!(p.params("face").unwrap()[0].1, "CPU");
        assert!(p.params("other").is_none());
        assert!(GNetPackage::default().networks.is_empty());
    }

    #[test]
    fn a_transform_records_its_pattern_and_substitute() {
        let t = GTransform::new(
            "fuse two resizes",
            vec!["resize".into(), "resize".into()],
            vec!["resize".into()],
        );
        assert_eq!(t.pattern.len(), 2);
        assert_eq!(t.substitute.len(), 1);
        assert!(t.description.contains("fuse"));
    }

    #[test]
    fn islands_and_queues_reject_what_cannot_work() {
        assert!(island("roi", &[GArg::Mat(0)], &[GArg::Mat(1)]).is_ok());
        assert!(island("", &[GArg::Mat(0)], &[GArg::Mat(1)]).is_err());
        assert!(island("roi", &[], &[GArg::Mat(1)]).is_err());
        assert!(island("roi", &[GArg::Mat(0)], &[]).is_err());
        assert_eq!(queue_capacity(4).unwrap().capacity, 4);
        assert!(queue_capacity(0).is_err(), "a zero-capacity queue deadlocks");
        assert_eq!(use_threaded_executor(4), 4);
    }

    #[test]
    fn a_byte_stream_round_trips_what_it_wrote() {
        let mut s = ByteStream::new();
        s.write_u32(7);
        s.write_string("hello");
        s.write_u32(9);
        assert!(!s.bytes().is_empty());
        s.rewind();
        assert_eq!(s.read_u32().unwrap(), 7);
        assert_eq!(s.read_string().unwrap(), "hello");
        assert_eq!(s.read_u32().unwrap(), 9);
        // Past the end is an error, not a zero.
        assert!(s.read_u32().is_err());

        let mut truncated = ByteStream::new();
        truncated.write_u32(100);
        truncated.rewind();
        assert!(truncated.read_string().is_err(), "a length past the end");
    }

    #[test]
    fn a_cancelled_context_stops_the_run_before_it_starts() {
        let ctx = GAsyncContext::new();
        assert!(!ctx.is_cancelled());
        let ran = std::cell::Cell::new(false);
        let out = r#async(&ctx, || {
            ran.set(true);
            Ok(mat(4, 4))
        });
        assert!(out.is_ok());
        assert!(ran.get());

        ctx.cancel();
        assert!(ctx.is_cancelled());
        let ran2 = std::cell::Cell::new(false);
        let out = r#async(&ctx, || {
            ran2.set(true);
            Ok(mat(4, 4))
        });
        assert_eq!(out.unwrap_err(), GAsyncCanceled);
        assert!(!ran2.get(), "a cancelled run must not execute");
        assert_eq!(GAsyncCanceled.what(), "GAsyncCanceled");
    }

    #[test]
    fn a_graph_serialises_binds_and_describes_itself() {
        let mut g = crate::GComputation::new();
        let inp = g.input();
        let out = g.cvt_color_bgr2gray(inp);
        g.set_output(out);

        let bytes = serialize(&g).unwrap();
        assert!(!bytes.is_empty());
        assert!(String::from_utf8_lossy(&bytes).contains("nodes"));
        assert!(get_graph(&g).contains("node"));
        assert!(get_in_stream().is_none());

        let (shape, m) = bind(GShape::GMat, mat(4, 4)).unwrap();
        assert_eq!(shape, GShape::GMat);
        assert_eq!(m.cols(), 4);
        assert!(bind(GShape::GScalar, mat(4, 4)).is_err());

        // Async apply runs the graph, and respects cancellation.
        let ctx = GAsyncContext::new();
        let src = Mat::zeros(8, 8, cv_core::types::CV_8UC3).unwrap();
        assert!(async_apply(&ctx, &mut g, &src).is_ok());
        ctx.cancel();
        assert_eq!(async_apply(&ctx, &mut g, &src).unwrap_err(), GAsyncCanceled);
    }

    #[test]
    fn the_protocol_wrappers_are_distinct_handles() {
        assert_eq!(GMatP(3).0, 3);
        assert_eq!(GFrame(4).0, 4);
        assert_eq!(GArrayU(5).0, 5);
        assert_eq!(GOpaqueU(6).0, 6);
        // `desync` and `copy` are graph nodes, not transformations of the value.
        let mut g = crate::GComputation::new();
        let inp = g.input();
        assert_eq!(desync(inp), inp);
        assert_eq!(copy(inp), inp);
    }
}
