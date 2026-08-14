//! `cv::cuda` — the CUDA API surface of a build with no CUDA devices
//! (`modules/core/include/opencv2/core/cuda.hpp`).
//!
//! # Why this exists
//!
//! OpenCV compiled without CUDA still **exposes** this whole namespace: cv2's
//! CPU-only wheel answers `cv2.cuda.getCudaEnabledDeviceCount()` with `0` and
//! lets you construct a `cv2.cuda_GpuMat()`. Code that branches on device
//! availability compiles and takes its CPU path. That is exactly what this
//! module reproduces, so a translation from C++ does not have to delete the
//! branch it was written with.
//!
//! # What it does
//!
//! [`get_cuda_enabled_device_count`] returns 0 and every query answers as it
//! does with no device: zeros, empty names, `false`. Anything that would need
//! to *touch* a device — [`GpuMat::upload`], [`set_device`] — returns
//! [`Code::GpuNotSupported`], the code upstream raises in the same situation.
//!
//! This is not a CUDA backend and will not become one: a pure-Rust port cannot
//! link the CUDA driver. What it is, is the honest shape of the API when no
//! device is present.

use crate::error::{Code, Error, Result};
use crate::geometry::Size;
use crate::mat::{Mat, MatLike};
use crate::types::MatType;

/// The error every operation that would need a real device returns.
fn no_device<T>(what: &str) -> Result<T> {
    Err(Error::new(
        Code::GpuNotSupported,
        format!("{what}: this build has no CUDA support"),
    ))
}

/// `cv::cuda::getCudaEnabledDeviceCount()` — always 0.
#[must_use]
pub fn get_cuda_enabled_device_count() -> i32 {
    0
}

/// `cv::cuda::setDevice(device)`.
pub fn set_device(device: i32) -> Result<()> {
    no_device(&format!("setDevice({device})"))
}

/// `cv::cuda::getDevice()`.
pub fn get_device() -> Result<i32> {
    no_device("getDevice")
}

/// `cv::cuda::resetDevice()`.
pub fn reset_device() -> Result<()> {
    no_device("resetDevice")
}

/// `cv::cuda::deviceSupports(feature_set)` — always false.
#[must_use]
pub fn device_supports(_feature_set: FeatureSet) -> bool {
    false
}

/// `cv::cuda::printCudaDeviceInfo(device)`.
pub fn print_cuda_device_info(_device: i32) {
    println!("*** CUDA Device Query ***");
    println!("Device count: 0");
}

/// `cv::cuda::printShortCudaDeviceInfo(device)`.
pub fn print_short_cuda_device_info(_device: i32) {
    println!("Device count: 0");
}

/// `cv::cuda::registerPageLocked(m)`.
pub fn register_page_locked(_m: &mut Mat) -> Result<()> {
    no_device("registerPageLocked")
}

/// `cv::cuda::unregisterPageLocked(m)`.
pub fn unregister_page_locked(_m: &mut Mat) -> Result<()> {
    no_device("unregisterPageLocked")
}

/// `cv::cuda::setBufferPoolUsage(on)` — accepted and ignored.
pub fn set_buffer_pool_usage(_on: bool) {}

/// `cv::cuda::setBufferPoolConfig(deviceId, stackSize, stackCount)` — ignored.
pub fn set_buffer_pool_config(_device_id: i32, _stack_size: usize, _stack_count: i32) {}

/// `cv::cuda::FeatureSet` — the compute capability a kernel needs.
///
/// The values are the capability number written without its dot, so
/// `FEATURE_SET_COMPUTE_35` is compute 3.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum FeatureSet {
    /// `FEATURE_SET_COMPUTE_10`
    Compute10 = 10,
    /// `FEATURE_SET_COMPUTE_11` — also `GLOBAL_ATOMICS`.
    Compute11 = 11,
    /// `FEATURE_SET_COMPUTE_12` — also `SHARED_ATOMICS`.
    Compute12 = 12,
    /// `FEATURE_SET_COMPUTE_13` — also `NATIVE_DOUBLE`.
    Compute13 = 13,
    /// `FEATURE_SET_COMPUTE_20`
    Compute20 = 20,
    /// `FEATURE_SET_COMPUTE_21`
    Compute21 = 21,
    /// `FEATURE_SET_COMPUTE_30` — also `WARP_SHUFFLE_FUNCTIONS`.
    Compute30 = 30,
    /// `FEATURE_SET_COMPUTE_32`
    Compute32 = 32,
    /// `FEATURE_SET_COMPUTE_35` — also `DYNAMIC_PARALLELISM`.
    Compute35 = 35,
    /// `FEATURE_SET_COMPUTE_50`
    Compute50 = 50,
}

/// `GLOBAL_ATOMICS` — upstream's alias for compute 1.1.
pub const GLOBAL_ATOMICS: FeatureSet = FeatureSet::Compute11;
/// `SHARED_ATOMICS` — upstream's alias for compute 1.2.
pub const SHARED_ATOMICS: FeatureSet = FeatureSet::Compute12;
/// `NATIVE_DOUBLE` — upstream's alias for compute 1.3.
pub const NATIVE_DOUBLE: FeatureSet = FeatureSet::Compute13;
/// `WARP_SHUFFLE_FUNCTIONS` — upstream's alias for compute 3.0.
pub const WARP_SHUFFLE_FUNCTIONS: FeatureSet = FeatureSet::Compute30;
/// `DYNAMIC_PARALLELISM` — upstream's alias for compute 3.5.
pub const DYNAMIC_PARALLELISM: FeatureSet = FeatureSet::Compute35;

/// Constants of the CUDA execution model, from `cuda/warp.hpp` and
/// `cuda/scan.hpp`.
///
/// Fixed properties of the hardware model rather than of any particular
/// device, so they are the real values even with no device present — a warp is
/// 32 lanes on every NVIDIA part to date.
pub mod warp {
    /// `LOG_WARP_SIZE` — `log2(WARP_SIZE)`.
    pub const LOG_WARP_SIZE: i32 = 5;
    /// `WARP_SIZE` — threads executing in lockstep.
    pub const WARP_SIZE: i32 = 1 << LOG_WARP_SIZE;
    /// `STRIDE` — the stride a warp-wide scan steps by; the warp size.
    pub const STRIDE: i32 = WARP_SIZE;
    /// `ScanKind::EXCLUSIVE` — each output excludes its own input.
    pub const EXCLUSIVE: i32 = 0;
    /// `ScanKind::INCLUSIVE` — each output includes its own input.
    pub const INCLUSIVE: i32 = 1;
}

/// `cv::cuda::TargetArchs` — which architectures the library was built for.
///
/// Every query is false: nothing was compiled for any device.
#[derive(Debug)]
pub struct TargetArchs;

impl TargetArchs {
    /// `TargetArchs::builtWith(feature_set)`.
    #[must_use]
    pub fn built_with(_feature_set: FeatureSet) -> bool {
        false
    }
    /// `TargetArchs::has(major, minor)`.
    #[must_use]
    pub fn has(_major: i32, _minor: i32) -> bool {
        false
    }
    /// `TargetArchs::hasPtx(major, minor)`.
    #[must_use]
    pub fn has_ptx(_major: i32, _minor: i32) -> bool {
        false
    }
    /// `TargetArchs::hasBin(major, minor)`.
    #[must_use]
    pub fn has_bin(_major: i32, _minor: i32) -> bool {
        false
    }
    /// `TargetArchs::hasEqualOrLessPtx(major, minor)`.
    #[must_use]
    pub fn has_equal_or_less_ptx(_major: i32, _minor: i32) -> bool {
        false
    }
    /// `TargetArchs::hasEqualOrGreater(major, minor)`.
    #[must_use]
    pub fn has_equal_or_greater(_major: i32, _minor: i32) -> bool {
        false
    }
    /// `TargetArchs::hasEqualOrGreaterPtx(major, minor)`.
    #[must_use]
    pub fn has_equal_or_greater_ptx(_major: i32, _minor: i32) -> bool {
        false
    }
    /// `TargetArchs::hasEqualOrGreaterBin(major, minor)`.
    #[must_use]
    pub fn has_equal_or_greater_bin(_major: i32, _minor: i32) -> bool {
        false
    }
}

/// `cv::cuda::DeviceInfo::ComputeMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum ComputeMode {
    /// `ComputeModeDefault` — several threads may share the device.
    #[default]
    Default = 0,
    /// `ComputeModeExclusive` — one thread at a time.
    Exclusive = 1,
    /// `ComputeModeProhibited` — no context may be created.
    Prohibited = 2,
    /// `ComputeModeExclusiveProcess` — one process at a time.
    ExclusiveProcess = 3,
}

/// `cv::cuda::DeviceInfo` — the properties of one CUDA device.
///
/// Constructible, as upstream's is, and every accessor answers for the absent
/// device: zero, empty, or false. [`DeviceInfo::is_compatible`] is the one to
/// branch on, and it is always false.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeviceInfo {
    id: i32,
}

impl DeviceInfo {
    /// `DeviceInfo::DeviceInfo()` — the current device, which does not exist.
    #[must_use]
    pub fn new() -> Self {
        DeviceInfo { id: 0 }
    }

    /// `DeviceInfo::DeviceInfo(device_id)`.
    #[must_use]
    pub fn with_id(device_id: i32) -> Self {
        DeviceInfo { id: device_id }
    }

    /// `DeviceInfo::deviceID()` — the index this was constructed with.
    #[must_use]
    pub fn device_id(&self) -> i32 {
        self.id
    }

    /// `DeviceInfo::name()` — empty, there being no device to name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        ""
    }

    /// `DeviceInfo::isCompatible()` — can this build run on the device?
    #[must_use]
    pub fn is_compatible(&self) -> bool {
        false
    }

    /// `DeviceInfo::supports(feature_set)`.
    #[must_use]
    pub fn supports(&self, _feature_set: FeatureSet) -> bool {
        false
    }

    /// `DeviceInfo::queryMemory(totalMemory, freeMemory)`.
    pub fn query_memory(&self) -> Result<(usize, usize)> {
        no_device("DeviceInfo::queryMemory")
    }

    /// `DeviceInfo::majorVersion()`.
    #[must_use]
    pub fn major_version(&self) -> i32 {
        0
    }
    /// `DeviceInfo::minorVersion()`.
    #[must_use]
    pub fn minor_version(&self) -> i32 {
        0
    }
    /// `DeviceInfo::multiProcessorCount()`.
    #[must_use]
    pub fn multi_processor_count(&self) -> i32 {
        0
    }
    /// `DeviceInfo::sharedMemPerBlock()`.
    #[must_use]
    pub fn shared_mem_per_block(&self) -> usize {
        0
    }
    /// `DeviceInfo::warpSize()`.
    #[must_use]
    pub fn warp_size(&self) -> i32 {
        0
    }
    /// `DeviceInfo::maxThreadsPerBlock()`.
    #[must_use]
    pub fn max_threads_per_block(&self) -> i32 {
        0
    }
    /// `DeviceInfo::maxThreadsDim()`.
    #[must_use]
    pub fn max_threads_dim(&self) -> [i32; 3] {
        [0; 3]
    }
    /// `DeviceInfo::maxGridSize()`.
    #[must_use]
    pub fn max_grid_size(&self) -> [i32; 3] {
        [0; 3]
    }
    /// `DeviceInfo::maxThreadsPerMultiProcessor()`.
    #[must_use]
    pub fn max_threads_per_multi_processor(&self) -> i32 {
        0
    }
    /// `DeviceInfo::regsPerBlock()`.
    #[must_use]
    pub fn regs_per_block(&self) -> i32 {
        0
    }
    /// `DeviceInfo::memPitch()`.
    #[must_use]
    pub fn mem_pitch(&self) -> usize {
        0
    }
    /// `DeviceInfo::totalConstMem()`.
    #[must_use]
    pub fn total_const_mem(&self) -> usize {
        0
    }
    /// `DeviceInfo::totalGlobalMem()`.
    #[must_use]
    pub fn total_global_mem(&self) -> usize {
        0
    }
    /// `DeviceInfo::totalMemory()`.
    #[must_use]
    pub fn total_memory(&self) -> usize {
        0
    }
    /// `DeviceInfo::freeMemory()`.
    #[must_use]
    pub fn free_memory(&self) -> usize {
        0
    }
    /// `DeviceInfo::clockRate()`.
    #[must_use]
    pub fn clock_rate(&self) -> i32 {
        0
    }
    /// `DeviceInfo::memoryClockRate()`.
    #[must_use]
    pub fn memory_clock_rate(&self) -> i32 {
        0
    }
    /// `DeviceInfo::memoryBusWidth()`.
    #[must_use]
    pub fn memory_bus_width(&self) -> i32 {
        0
    }
    /// `DeviceInfo::l2CacheSize()`.
    #[must_use]
    pub fn l2_cache_size(&self) -> i32 {
        0
    }
    /// `DeviceInfo::textureAlignment()`.
    #[must_use]
    pub fn texture_alignment(&self) -> usize {
        0
    }
    /// `DeviceInfo::texturePitchAlignment()`.
    #[must_use]
    pub fn texture_pitch_alignment(&self) -> usize {
        0
    }
    /// `DeviceInfo::surfaceAlignment()`.
    #[must_use]
    pub fn surface_alignment(&self) -> usize {
        0
    }
    /// `DeviceInfo::asyncEngineCount()`.
    #[must_use]
    pub fn async_engine_count(&self) -> i32 {
        0
    }
    /// `DeviceInfo::computeMode()`.
    #[must_use]
    pub fn compute_mode(&self) -> ComputeMode {
        ComputeMode::Default
    }
    /// `DeviceInfo::concurrentKernels()`.
    #[must_use]
    pub fn concurrent_kernels(&self) -> bool {
        false
    }
    /// `DeviceInfo::ECCEnabled()`.
    #[must_use]
    pub fn ecc_enabled(&self) -> bool {
        false
    }
    /// `DeviceInfo::canMapHostMemory()`.
    #[must_use]
    pub fn can_map_host_memory(&self) -> bool {
        false
    }
    /// `DeviceInfo::integrated()`.
    #[must_use]
    pub fn integrated(&self) -> bool {
        false
    }
    /// `DeviceInfo::kernelExecTimeoutEnabled()`.
    #[must_use]
    pub fn kernel_exec_timeout_enabled(&self) -> bool {
        false
    }
    /// `DeviceInfo::tccDriver()`.
    #[must_use]
    pub fn tcc_driver(&self) -> bool {
        false
    }
    /// `DeviceInfo::unifiedAddressing()`.
    #[must_use]
    pub fn unified_addressing(&self) -> bool {
        false
    }
    /// `DeviceInfo::pciBusID()`.
    #[must_use]
    pub fn pci_bus_id(&self) -> i32 {
        0
    }
    /// `DeviceInfo::pciDeviceID()`.
    #[must_use]
    pub fn pci_device_id(&self) -> i32 {
        0
    }
    /// `DeviceInfo::pciDomainID()`.
    #[must_use]
    pub fn pci_domain_id(&self) -> i32 {
        0
    }
    /// `DeviceInfo::maxTexture1D()`.
    #[must_use]
    pub fn max_texture1d(&self) -> i32 {
        0
    }
    /// `DeviceInfo::maxTexture1DMipmap()`.
    #[must_use]
    pub fn max_texture1d_mipmap(&self) -> i32 {
        0
    }
    /// `DeviceInfo::maxTexture1DLinear()`.
    #[must_use]
    pub fn max_texture1d_linear(&self) -> i32 {
        0
    }
    /// `DeviceInfo::maxTexture2D()`.
    #[must_use]
    pub fn max_texture2d(&self) -> [i32; 2] {
        [0; 2]
    }
    /// `DeviceInfo::maxTexture2DMipmap()`.
    #[must_use]
    pub fn max_texture2d_mipmap(&self) -> [i32; 2] {
        [0; 2]
    }
    /// `DeviceInfo::maxTexture2DLinear()`.
    #[must_use]
    pub fn max_texture2d_linear(&self) -> [i32; 3] {
        [0; 3]
    }
    /// `DeviceInfo::maxTexture2DGather()`.
    #[must_use]
    pub fn max_texture2d_gather(&self) -> [i32; 2] {
        [0; 2]
    }
    /// `DeviceInfo::maxTexture3D()`.
    #[must_use]
    pub fn max_texture3d(&self) -> [i32; 3] {
        [0; 3]
    }
    /// `DeviceInfo::maxTextureCubemap()`.
    #[must_use]
    pub fn max_texture_cubemap(&self) -> i32 {
        0
    }
    /// `DeviceInfo::maxTexture1DLayered()`.
    #[must_use]
    pub fn max_texture1d_layered(&self) -> [i32; 2] {
        [0; 2]
    }
    /// `DeviceInfo::maxTexture2DLayered()`.
    #[must_use]
    pub fn max_texture2d_layered(&self) -> [i32; 3] {
        [0; 3]
    }
    /// `DeviceInfo::maxTextureCubemapLayered()`.
    #[must_use]
    pub fn max_texture_cubemap_layered(&self) -> [i32; 2] {
        [0; 2]
    }
    /// `DeviceInfo::maxSurface1D()`.
    #[must_use]
    pub fn max_surface1d(&self) -> i32 {
        0
    }
    /// `DeviceInfo::maxSurface2D()`.
    #[must_use]
    pub fn max_surface2d(&self) -> [i32; 2] {
        [0; 2]
    }
    /// `DeviceInfo::maxSurface3D()`.
    #[must_use]
    pub fn max_surface3d(&self) -> [i32; 3] {
        [0; 3]
    }
    /// `DeviceInfo::maxSurface1DLayered()`.
    #[must_use]
    pub fn max_surface1d_layered(&self) -> [i32; 2] {
        [0; 2]
    }
    /// `DeviceInfo::maxSurface2DLayered()`.
    #[must_use]
    pub fn max_surface2d_layered(&self) -> [i32; 3] {
        [0; 3]
    }
    /// `DeviceInfo::maxSurfaceCubemap()`.
    #[must_use]
    pub fn max_surface_cubemap(&self) -> i32 {
        0
    }
    /// `DeviceInfo::maxSurfaceCubemapLayered()`.
    #[must_use]
    pub fn max_surface_cubemap_layered(&self) -> [i32; 2] {
        [0; 2]
    }
}

/// `cv::cuda::GpuMat::Allocator` — where device memory would come from.
///
/// A trait rather than an abstract class; nothing can implement it usefully
/// without a device, and [`default_allocator`] returns `None`.
pub trait Allocator: std::fmt::Debug {
    /// `Allocator::allocate(mat, rows, cols, elemSize)`.
    fn allocate(&self, rows: i32, cols: i32, elem_size: usize) -> Result<()>;
    /// `Allocator::free(mat)`.
    fn free(&self);
}

/// `cv::cuda::GpuMat::defaultAllocator()` — none, there being no device.
#[must_use]
pub fn default_allocator() -> Option<&'static dyn Allocator> {
    None
}

/// `cv::cuda::GpuMat::setDefaultAllocator(allocator)`.
pub fn set_default_allocator(_allocator: &'static dyn Allocator) -> Result<()> {
    no_device("GpuMat::setDefaultAllocator")
}

/// `cv::cuda::GpuData` — a reference-counted device allocation.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuData {
    /// `GpuData::size` — bytes the allocation would span.
    pub size: usize,
}

impl GpuData {
    /// `GpuData::GpuData(_size)` — records the size; allocates nothing.
    #[must_use]
    pub fn new(size: usize) -> Self {
        GpuData { size }
    }
}

/// `cv::cuda::GpuMat` — a matrix header for device memory.
///
/// Constructible and shape-queryable, as upstream's is with no device, but
/// every transfer fails. The header carries the size and type it was asked
/// for so that code computing an output shape still works.
#[derive(Debug, Clone, Copy)]
pub struct GpuMat {
    rows: i32,
    cols: i32,
    typ: MatType,
}

impl Default for GpuMat {
    fn default() -> Self {
        GpuMat { rows: 0, cols: 0, typ: crate::types::CV_8UC1 }
    }
}

impl GpuMat {
    /// `GpuMat::GpuMat()` — an empty header.
    #[must_use]
    pub fn new() -> Self {
        GpuMat::default()
    }

    /// `GpuMat::GpuMat(rows, cols, type)`.
    #[must_use]
    pub fn with_size(rows: i32, cols: i32, typ: MatType) -> Self {
        GpuMat { rows, cols, typ }
    }

    /// `GpuMat::rows`.
    #[must_use]
    pub fn rows(&self) -> i32 {
        self.rows
    }
    /// `GpuMat::cols`.
    #[must_use]
    pub fn cols(&self) -> i32 {
        self.cols
    }
    /// `GpuMat::size()`.
    #[must_use]
    pub fn size(&self) -> Size {
        Size::new(self.cols, self.rows)
    }
    /// `GpuMat::type()`.
    #[must_use]
    pub fn mat_type(&self) -> MatType {
        self.typ
    }
    /// `GpuMat::empty()`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows <= 0 || self.cols <= 0
    }
    /// `GpuMat::cudaPtr()` — null, there being no allocation.
    #[must_use]
    pub fn cuda_ptr(&self) -> *mut std::ffi::c_void {
        std::ptr::null_mut()
    }
    /// `GpuMat::getAllocator()`.
    #[must_use]
    pub fn get_allocator(&self) -> Option<&'static dyn Allocator> {
        None
    }

    /// `GpuMat::upload(arr)`.
    pub fn upload<M: MatLike + ?Sized>(&mut self, _src: &M) -> Result<()> {
        no_device("GpuMat::upload")
    }

    /// `GpuMat::download(dst)`.
    pub fn download(&self) -> Result<Mat> {
        no_device("GpuMat::download")
    }

    /// `GpuMat::createMatHeader()` — a CPU header over the device memory.
    pub fn create_mat_header(&self) -> Result<Mat> {
        no_device("GpuMat::createMatHeader")
    }
}

/// `cv::cuda::GpuMatND` — the N-dimensional form of [`GpuMat`].
///
/// Carries the shape it was asked for; every transfer fails as [`GpuMat`]'s do.
#[derive(Debug, Clone, Default)]
pub struct GpuMatND {
    shape: Vec<i32>,
    typ: Option<MatType>,
}

impl GpuMatND {
    /// `GpuMatND::GpuMatND()` — an empty header.
    #[must_use]
    pub fn new() -> Self {
        GpuMatND::default()
    }

    /// `GpuMatND::GpuMatND(size, type)`.
    #[must_use]
    pub fn with_shape(shape: &[i32], typ: MatType) -> Self {
        GpuMatND { shape: shape.to_vec(), typ: Some(typ) }
    }

    /// `GpuMatND::dims` — the number of dimensions.
    #[must_use]
    pub fn dims(&self) -> usize {
        self.shape.len()
    }

    /// `GpuMatND::size` — the extent of each dimension.
    #[must_use]
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }

    /// `GpuMatND::type()`.
    #[must_use]
    pub fn mat_type(&self) -> Option<MatType> {
        self.typ
    }

    /// `GpuMatND::empty()`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shape.is_empty() || self.shape.iter().any(|&d| d <= 0)
    }

    /// `GpuMatND::upload(src)`.
    pub fn upload<M: MatLike + ?Sized>(&mut self, _src: &M) -> Result<()> {
        no_device("GpuMatND::upload")
    }

    /// `GpuMatND::download(dst)`.
    pub fn download(&self) -> Result<Mat> {
        no_device("GpuMatND::download")
    }
}

/// `cv::cuda::createContinuous(rows, cols, type, arr)`.
pub fn create_continuous(rows: i32, cols: i32, typ: MatType) -> Result<GpuMat> {
    let _ = (rows, cols, typ);
    no_device("createContinuous")
}

/// `cv::cuda::ensureSizeIsEnough(rows, cols, type, arr)`.
pub fn ensure_size_is_enough(rows: i32, cols: i32, typ: MatType, _m: &mut GpuMat) -> Result<()> {
    let _ = (rows, cols, typ);
    no_device("ensureSizeIsEnough")
}

/// `cv::cuda::createGpuMatFromCudaMemory(rows, cols, type, cudaMemoryAddress, step)`.
pub fn create_gpu_mat_from_cuda_memory(
    rows: i32,
    cols: i32,
    typ: MatType,
    _address: usize,
    _step: usize,
) -> Result<GpuMat> {
    let _ = (rows, cols, typ);
    no_device("createGpuMatFromCudaMemory")
}

/// `cv::cuda::BufferPool` — a per-stream allocator for scratch device memory.
#[derive(Debug, Clone, Copy, Default)]
pub struct BufferPool;

impl BufferPool {
    /// `BufferPool::BufferPool(stream)`.
    #[must_use]
    pub fn new(_stream: &Stream) -> Self {
        BufferPool
    }
    /// `BufferPool::getBuffer(rows, cols, type)`.
    pub fn get_buffer(&self, rows: i32, cols: i32, typ: MatType) -> Result<GpuMat> {
        let _ = (rows, cols, typ);
        no_device("BufferPool::getBuffer")
    }
    /// `BufferPool::getAllocator()`.
    #[must_use]
    pub fn get_allocator(&self) -> Option<&'static dyn Allocator> {
        None
    }
}

/// `cv::cuda::HostMem::AllocType` — how page-locked host memory is mapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum AllocType {
    /// `HostMem::PAGE_LOCKED` — pinned, so transfers can be asynchronous.
    #[default]
    PageLocked = 1,
    /// `HostMem::SHARED` — also mapped into the device address space.
    Shared = 2,
    /// `HostMem::WRITE_COMBINED` — uncached on the host; fast to write, slow to read.
    WriteCombined = 4,
}

/// `cv::cuda::HostMem` — page-locked host memory.
///
/// Backed by an ordinary [`Mat`] here: without a driver there is nothing to pin,
/// and an ordinary allocation is what upstream falls back to as well when
/// pinning fails.
#[derive(Debug, Clone)]
pub struct HostMem {
    mat: Mat,
    alloc_type: AllocType,
}

impl HostMem {
    /// `HostMem::HostMem(alloc_type)`.
    #[must_use]
    pub fn new(alloc_type: AllocType) -> Self {
        HostMem { mat: Mat::empty(crate::types::CV_8UC1), alloc_type }
    }

    /// `HostMem::HostMem(rows, cols, type, alloc_type)`.
    pub fn with_size(rows: i32, cols: i32, typ: MatType, alloc_type: AllocType) -> Result<Self> {
        Ok(HostMem { mat: Mat::zeros(rows, cols, typ)?, alloc_type })
    }

    /// `HostMem::alloc_type`.
    #[must_use]
    pub fn alloc_type(&self) -> AllocType {
        self.alloc_type
    }

    /// `HostMem::createMatHeader()` — the host-side view, which does work.
    #[must_use]
    pub fn create_mat_header(&self) -> &Mat {
        &self.mat
    }

    /// `HostMem::createGpuMatHeader()`.
    pub fn create_gpu_mat_header(&self) -> Result<GpuMat> {
        no_device("HostMem::createGpuMatHeader")
    }

    /// `HostMem::getAllocator(alloc_type)`.
    #[must_use]
    pub fn get_allocator(_alloc_type: AllocType) -> Option<&'static dyn Allocator> {
        None
    }
}

/// `cv::cuda::Stream` — an ordered queue of device work.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stream;

impl Stream {
    /// `Stream::Stream()`.
    #[must_use]
    pub fn new() -> Self {
        Stream
    }

    /// `Stream::Null()` — the default stream.
    #[must_use]
    pub fn null() -> Self {
        Stream
    }

    /// `Stream::queryIfComplete()` — true, since nothing was ever queued.
    #[must_use]
    pub fn query_if_complete(&self) -> bool {
        true
    }

    /// `Stream::waitForCompletion()` — returns at once, the queue being empty.
    pub fn wait_for_completion(&self) {}

    /// `Stream::waitEvent(event)`.
    pub fn wait_event(&self, _event: &Event) {}

    /// `Stream::cudaPtr()` — null.
    #[must_use]
    pub fn cuda_ptr(&self) -> *mut std::ffi::c_void {
        std::ptr::null_mut()
    }

    /// `cv::cuda::StreamAccessor::wrapStream(stream)`.
    pub fn wrap_stream(_stream: usize) -> Result<Stream> {
        no_device("wrapStream")
    }
}

/// `cv::cuda::Event::CreateFlags`.
pub mod event_flags {
    /// `Event::DEFAULT`.
    pub const DEFAULT: i32 = 0x00;
    /// `Event::BLOCKING_SYNC` — wait by blocking rather than spinning.
    pub const BLOCKING_SYNC: i32 = 0x01;
    /// `Event::DISABLE_TIMING` — do not record a timestamp.
    pub const DISABLE_TIMING: i32 = 0x02;
    /// `Event::INTERPROCESS` — shareable between processes; needs `DISABLE_TIMING`.
    pub const INTERPROCESS: i32 = 0x04;
}

/// `cv::cuda::Event` — a marker recorded into a [`Stream`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Event;

impl Event {
    /// `Event::Event(flags)`.
    #[must_use]
    pub fn new(_flags: i32) -> Self {
        Event
    }

    /// `Event::record(stream)`.
    pub fn record(&mut self, _stream: &Stream) {}

    /// `Event::queryIfComplete()` — true, nothing having been queued.
    #[must_use]
    pub fn query_if_complete(&self) -> bool {
        true
    }

    /// `Event::waitForCompletion()`.
    pub fn wait_for_completion(&self) {}

    /// `Event::elapsedTime(start, end)` — milliseconds between two events.
    #[must_use]
    pub fn elapsed_time(_start: &Event, _end: &Event) -> f32 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CV_8UC3;

    #[test]
    fn the_device_count_is_zero_and_queries_answer() {
        assert_eq!(get_cuda_enabled_device_count(), 0);
        assert!(!device_supports(FeatureSet::Compute50));
        let d = DeviceInfo::new();
        assert!(!d.is_compatible());
        assert_eq!(d.name(), "");
        assert_eq!(d.major_version(), 0);
        assert_eq!(d.total_memory(), 0);
        assert_eq!(d.max_threads_dim(), [0, 0, 0]);
        assert_eq!(d.compute_mode(), ComputeMode::Default);
        assert_eq!(DeviceInfo::with_id(3).device_id(), 3);
    }

    #[test]
    fn anything_needing_a_device_reports_gpu_not_supported() {
        assert_eq!(set_device(0).unwrap_err().code, Code::GpuNotSupported);
        assert_eq!(get_device().unwrap_err().code, Code::GpuNotSupported);
        assert_eq!(reset_device().unwrap_err().code, Code::GpuNotSupported);
        assert_eq!(DeviceInfo::new().query_memory().unwrap_err().code, Code::GpuNotSupported);

        let mut g = GpuMat::with_size(4, 4, CV_8UC3);
        let src = Mat::zeros(4, 4, CV_8UC3).unwrap();
        assert_eq!(g.upload(&src).unwrap_err().code, Code::GpuNotSupported);
        assert_eq!(g.download().unwrap_err().code, Code::GpuNotSupported);
    }

    #[test]
    fn a_gpu_mat_header_still_reports_its_shape() {
        let g = GpuMat::with_size(3, 5, CV_8UC3);
        assert_eq!(g.size(), Size::new(5, 3));
        assert_eq!(g.rows(), 3);
        assert_eq!(g.cols(), 5);
        assert_eq!(g.mat_type(), CV_8UC3);
        assert!(!g.is_empty());
        assert!(GpuMat::new().is_empty());
        assert!(g.cuda_ptr().is_null());
        assert!(g.get_allocator().is_none());
    }

    #[test]
    fn a_gpu_mat_nd_carries_its_shape() {
        let g = GpuMatND::with_shape(&[2, 3, 4], CV_8UC3);
        assert_eq!(g.dims(), 3);
        assert_eq!(g.shape(), &[2, 3, 4]);
        assert_eq!(g.mat_type(), Some(CV_8UC3));
        assert!(!g.is_empty());
        assert!(GpuMatND::new().is_empty());
        // A zero extent makes it empty, as it does for GpuMat.
        assert!(GpuMatND::with_shape(&[2, 0, 4], CV_8UC3).is_empty());
        assert_eq!(g.download().unwrap_err().code, Code::GpuNotSupported);
    }

    #[test]
    fn feature_set_values_are_the_capability_numbers() {
        assert_eq!(FeatureSet::Compute10 as i32, 10);
        assert_eq!(FeatureSet::Compute35 as i32, 35);
        assert_eq!(FeatureSet::Compute50 as i32, 50);
        assert_eq!(GLOBAL_ATOMICS, FeatureSet::Compute11);
        assert_eq!(SHARED_ATOMICS, FeatureSet::Compute12);
        assert_eq!(NATIVE_DOUBLE, FeatureSet::Compute13);
        assert_eq!(WARP_SHUFFLE_FUNCTIONS, FeatureSet::Compute30);
        assert_eq!(DYNAMIC_PARALLELISM, FeatureSet::Compute35);
    }

    #[test]
    fn target_archs_reports_nothing_was_built() {
        assert!(!TargetArchs::has(5, 0));
        assert!(!TargetArchs::has_ptx(5, 0));
        assert!(!TargetArchs::has_bin(5, 0));
        assert!(!TargetArchs::has_equal_or_greater(1, 0));
        assert!(!TargetArchs::has_equal_or_greater_ptx(1, 0));
        assert!(!TargetArchs::has_equal_or_greater_bin(1, 0));
        assert!(!TargetArchs::has_equal_or_less_ptx(9, 9));
        assert!(!TargetArchs::built_with(FeatureSet::Compute10));
    }

    #[test]
    fn a_stream_is_always_idle_and_events_take_no_time() {
        let s = Stream::new();
        assert!(s.query_if_complete());
        s.wait_for_completion();
        assert!(Stream::null().query_if_complete());
        assert!(s.cuda_ptr().is_null());

        let (mut a, b) = (Event::new(event_flags::DEFAULT), Event::new(event_flags::DISABLE_TIMING));
        a.record(&s);
        assert!(a.query_if_complete());
        assert_eq!(Event::elapsed_time(&a, &b), 0.0);
        s.wait_event(&b);
        assert_eq!(Stream::wrap_stream(0).unwrap_err().code, Code::GpuNotSupported);
    }

    #[test]
    fn host_mem_falls_back_to_an_ordinary_allocation() {
        let h = HostMem::with_size(2, 3, CV_8UC3, AllocType::PageLocked).unwrap();
        assert_eq!(h.alloc_type(), AllocType::PageLocked);
        // The host side works; only the device view fails.
        assert_eq!(h.create_mat_header().size(), Size::new(3, 2));
        assert_eq!(h.create_gpu_mat_header().unwrap_err().code, Code::GpuNotSupported);
        assert!(HostMem::new(AllocType::Shared).create_mat_header().is_empty());
        assert!(HostMem::get_allocator(AllocType::WriteCombined).is_none());
        assert_eq!(AllocType::WriteCombined as i32, 4);
    }

    #[test]
    fn the_buffer_pool_has_nothing_to_hand_out() {
        let p = BufferPool::new(&Stream::null());
        assert_eq!(p.get_buffer(1, 1, CV_8UC3).unwrap_err().code, Code::GpuNotSupported);
        assert!(p.get_allocator().is_none());
        // The configuration setters are accepted and ignored.
        set_buffer_pool_usage(true);
        set_buffer_pool_config(0, 1024, 2);
        assert!(default_allocator().is_none());
    }

    #[test]
    fn the_allocation_helpers_report_the_missing_device() {
        assert_eq!(create_continuous(2, 2, CV_8UC3).unwrap_err().code, Code::GpuNotSupported);
        let mut g = GpuMat::new();
        assert_eq!(
            ensure_size_is_enough(2, 2, CV_8UC3, &mut g).unwrap_err().code,
            Code::GpuNotSupported
        );
        assert_eq!(
            create_gpu_mat_from_cuda_memory(2, 2, CV_8UC3, 0, 0).unwrap_err().code,
            Code::GpuNotSupported
        );
        assert_eq!(GpuData::new(64).size, 64);
        let mut m = Mat::zeros(1, 1, CV_8UC3).unwrap();
        assert!(register_page_locked(&mut m).is_err());
        assert!(unregister_page_locked(&mut m).is_err());
    }
}
