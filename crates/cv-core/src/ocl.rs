//! `cv::ocl` — the OpenCL API surface of a build without OpenCL
//! (`modules/core/include/opencv2/core/ocl.hpp`).
//!
//! The same reasoning as [`crate::cuda`]: OpenCV built with `WITH_OPENCL=OFF`
//! still exposes this namespace, `haveOpenCL()` returns false, and callers
//! branch on it. Reproducing that shape means C++ that queries the device
//! before choosing a path keeps working here.
//!
//! [`have_opencl`] is false and [`Device::default_device`] is the null device,
//! whose accessors all report zero, empty or false — which is what upstream's
//! `Device` does when its internal pointer is null, rather than crashing.
//!
//! No kernel here dispatches to OpenCL and none will: the runtime is a C
//! library this port does not link.

/// `cv::ocl::haveOpenCL()` — always false.
#[must_use]
pub fn have_opencl() -> bool {
    false
}

/// `cv::ocl::useOpenCL()` — always false.
#[must_use]
pub fn use_opencl() -> bool {
    false
}

/// `cv::ocl::setUseOpenCL(flag)` — accepted and ignored.
///
/// Upstream also ignores a `true` here when `haveOpenCL()` is false, so a
/// caller enabling it optimistically behaves the same either way.
pub fn set_use_opencl(_flag: bool) {}

/// `cv::ocl::haveAmdBlas()` — always false.
#[must_use]
pub fn have_amd_blas() -> bool {
    false
}

/// `cv::ocl::haveAmdFft()` — always false.
#[must_use]
pub fn have_amd_fft() -> bool {
    false
}

/// `cv::ocl::haveSVM()` — always false.
#[must_use]
pub fn have_svm() -> bool {
    false
}

/// `cv::ocl::finish()` — nothing is queued, so this returns at once.
pub fn finish() {}

/// `cv::ocl::Device::TYPE_*` — the device classes a query can ask for.
pub mod device_type {
    /// `Device::TYPE_DEFAULT`
    pub const TYPE_DEFAULT: i32 = 1 << 0;
    /// `Device::TYPE_CPU`
    pub const TYPE_CPU: i32 = 1 << 1;
    /// `Device::TYPE_GPU`
    pub const TYPE_GPU: i32 = 1 << 2;
    /// `Device::TYPE_ACCELERATOR`
    pub const TYPE_ACCELERATOR: i32 = 1 << 3;
    /// `Device::TYPE_DGPU` — a discrete GPU.
    pub const TYPE_DGPU: i32 = TYPE_GPU + (1 << 16);
    /// `Device::TYPE_IGPU` — an integrated GPU.
    pub const TYPE_IGPU: i32 = TYPE_GPU + (1 << 17);
    /// `Device::TYPE_ALL`
    pub const TYPE_ALL: u32 = 0xFFFF_FFFF;
}

/// `cv::ocl::Device`'s floating-point capability bits.
pub mod fp_config {
    /// `Device::FP_DENORM` — denormals are supported.
    pub const FP_DENORM: i32 = 1 << 0;
    /// `Device::FP_INF_NAN` — infinities and NaNs are supported.
    pub const FP_INF_NAN: i32 = 1 << 1;
    /// `Device::FP_ROUND_TO_NEAREST`
    pub const FP_ROUND_TO_NEAREST: i32 = 1 << 2;
    /// `Device::FP_ROUND_TO_ZERO`
    pub const FP_ROUND_TO_ZERO: i32 = 1 << 3;
    /// `Device::FP_ROUND_TO_INF`
    pub const FP_ROUND_TO_INF: i32 = 1 << 4;
    /// `Device::FP_FMA` — fused multiply-add is available.
    pub const FP_FMA: i32 = 1 << 5;
    /// `Device::FP_SOFT_FLOAT` — floats are emulated.
    pub const FP_SOFT_FLOAT: i32 = 1 << 6;
    /// `Device::FP_CORRECTLY_ROUNDED_DIVIDE_SQRT`
    pub const FP_CORRECTLY_ROUNDED_DIVIDE_SQRT: i32 = 1 << 7;
}

/// `cv::ocl::Device`'s execution, cache, memory and vendor enumerations.
pub mod device_caps {
    /// `Device::EXEC_KERNEL` — can run OpenCL C kernels.
    pub const EXEC_KERNEL: i32 = 1 << 0;
    /// `Device::EXEC_NATIVE_KERNEL` — can run native function kernels.
    pub const EXEC_NATIVE_KERNEL: i32 = 1 << 1;

    /// `Device::NO_CACHE`
    pub const NO_CACHE: i32 = 0;
    /// `Device::READ_ONLY_CACHE`
    pub const READ_ONLY_CACHE: i32 = 1;
    /// `Device::READ_WRITE_CACHE`
    pub const READ_WRITE_CACHE: i32 = 2;

    /// `Device::NO_LOCAL_MEM`
    pub const NO_LOCAL_MEM: i32 = 0;
    /// `Device::LOCAL_IS_LOCAL` — dedicated local memory.
    pub const LOCAL_IS_LOCAL: i32 = 1;
    /// `Device::LOCAL_IS_GLOBAL` — local memory carved out of global.
    pub const LOCAL_IS_GLOBAL: i32 = 2;

    /// `Device::UNKNOWN_VENDOR`
    pub const UNKNOWN_VENDOR: i32 = 0;
    /// `Device::VENDOR_AMD`
    pub const VENDOR_AMD: i32 = 1;
    /// `Device::VENDOR_INTEL`
    pub const VENDOR_INTEL: i32 = 2;
    /// `Device::VENDOR_NVIDIA`
    pub const VENDOR_NVIDIA: i32 = 3;
}

/// `cv::ocl::KernelArg`'s access flags.
pub mod kernel_arg {
    /// `KernelArg::LOCAL`
    pub const LOCAL: i32 = 1;
    /// `KernelArg::READ_ONLY`
    pub const READ_ONLY: i32 = 2;
    /// `KernelArg::WRITE_ONLY`
    pub const WRITE_ONLY: i32 = 4;
    /// `KernelArg::READ_WRITE`
    pub const READ_WRITE: i32 = 6;
    /// `KernelArg::CONSTANT`
    pub const CONSTANT: i32 = 8;
    /// `KernelArg::PTR_ONLY` — pass the pointer without the size metadata.
    pub const PTR_ONLY: i32 = 16;
    /// `KernelArg::NO_SIZE` — omit the size arguments.
    pub const NO_SIZE: i32 = 256;
}

/// `cv::ocl::svm::SVMCapabilities` — shared virtual memory support bits.
///
/// All absent, [`have_svm`] being false.
pub mod svm {
    /// `SVMCapabilities::SVM_COARSE_GRAIN_BUFFER` — sync at map/unmap.
    pub const SVM_COARSE_GRAIN_BUFFER: i32 = 1 << 0;
    /// `SVMCapabilities::SVM_FINE_GRAIN_BUFFER` — coherent within a buffer.
    pub const SVM_FINE_GRAIN_BUFFER: i32 = 1 << 1;
    /// `SVMCapabilities::SVM_FINE_GRAIN_SYSTEM` — the whole address space is shared.
    pub const SVM_FINE_GRAIN_SYSTEM: i32 = 1 << 2;
    /// `SVMCapabilities::SVM_ATOMICS` — atomics work across host and device.
    pub const SVM_ATOMICS: i32 = 1 << 3;
}

/// `cv::ocl::OclVectorStrategy` — how wide a vector a generated kernel uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum OclVectorStrategy {
    /// `OCL_VECTOR_OWN` — each array's own width; also `OCL_VECTOR_DEFAULT`.
    #[default]
    Own = 0,
    /// `OCL_VECTOR_MAX` — the widest across all arrays.
    Max = 1,
}

/// `OCL_VECTOR_DEFAULT` — upstream's alias for `OCL_VECTOR_OWN`.
pub const OCL_VECTOR_DEFAULT: OclVectorStrategy = OclVectorStrategy::Own;

/// `cv::ocl::Device` — an OpenCL device.
///
/// Always the null device. Every accessor answers as upstream's does when its
/// internal handle is null: zero, empty, or false. [`Device::available`] is the
/// query to branch on.
#[derive(Debug, Clone, Copy, Default)]
pub struct Device;

impl Device {
    /// `Device::Device()` — the null device.
    #[must_use]
    pub fn new() -> Self {
        Device
    }

    /// `Device::getDefault()`.
    #[must_use]
    pub fn default_device() -> Device {
        Device
    }

    /// `Device::ptr()` — null, there being no device.
    #[must_use]
    pub fn ptr(&self) -> *mut std::ffi::c_void {
        std::ptr::null_mut()
    }

    /// `Device::name()`.
    #[must_use]
    pub fn name(&self) -> &'static str {
        ""
    }
    /// `Device::extensions()`.
    #[must_use]
    pub fn extensions(&self) -> &'static str {
        ""
    }
    /// `Device::isExtensionSupported(name)`.
    #[must_use]
    pub fn is_extension_supported(&self, _name: &str) -> bool {
        false
    }
    /// `Device::version()`.
    #[must_use]
    pub fn version(&self) -> &'static str {
        ""
    }
    /// `Device::vendorName()`.
    #[must_use]
    pub fn vendor_name(&self) -> &'static str {
        ""
    }
    /// `Device::OpenCL_C_Version()`.
    #[must_use]
    pub fn opencl_c_version(&self) -> &'static str {
        ""
    }
    /// `Device::OpenCLVersion()`.
    #[must_use]
    pub fn opencl_version(&self) -> &'static str {
        ""
    }
    /// `Device::deviceVersionMajor()`.
    #[must_use]
    pub fn device_version_major(&self) -> i32 {
        0
    }
    /// `Device::deviceVersionMinor()`.
    #[must_use]
    pub fn device_version_minor(&self) -> i32 {
        0
    }
    /// `Device::driverVersion()`.
    #[must_use]
    pub fn driver_version(&self) -> &'static str {
        ""
    }
    /// `Device::type()`.
    #[must_use]
    pub fn device_type(&self) -> i32 {
        0
    }
    /// `Device::addressBits()`.
    #[must_use]
    pub fn address_bits(&self) -> i32 {
        0
    }
    /// `Device::available()` — false, there being no device.
    #[must_use]
    pub fn available(&self) -> bool {
        false
    }
    /// `Device::compilerAvailable()`.
    #[must_use]
    pub fn compiler_available(&self) -> bool {
        false
    }
    /// `Device::linkerAvailable()`.
    #[must_use]
    pub fn linker_available(&self) -> bool {
        false
    }
    /// `Device::doubleFPConfig()`.
    #[must_use]
    pub fn double_fp_config(&self) -> i32 {
        0
    }
    /// `Device::singleFPConfig()`.
    #[must_use]
    pub fn single_fp_config(&self) -> i32 {
        0
    }
    /// `Device::halfFPConfig()`.
    #[must_use]
    pub fn half_fp_config(&self) -> i32 {
        0
    }
    /// `Device::hasFP64()`.
    #[must_use]
    pub fn has_fp64(&self) -> bool {
        false
    }
    /// `Device::hasFP16()`.
    #[must_use]
    pub fn has_fp16(&self) -> bool {
        false
    }
    /// `Device::endianLittle()`.
    #[must_use]
    pub fn endian_little(&self) -> bool {
        false
    }
    /// `Device::errorCorrectionSupport()`.
    #[must_use]
    pub fn error_correction_support(&self) -> bool {
        false
    }
    /// `Device::executionCapabilities()`.
    #[must_use]
    pub fn execution_capabilities(&self) -> i32 {
        0
    }
    /// `Device::globalMemCacheSize()`.
    #[must_use]
    pub fn global_mem_cache_size(&self) -> usize {
        0
    }
    /// `Device::globalMemCacheType()`.
    #[must_use]
    pub fn global_mem_cache_type(&self) -> i32 {
        0
    }
    /// `Device::globalMemCacheLineSize()`.
    #[must_use]
    pub fn global_mem_cache_line_size(&self) -> i32 {
        0
    }
    /// `Device::globalMemSize()`.
    #[must_use]
    pub fn global_mem_size(&self) -> usize {
        0
    }
    /// `Device::localMemSize()`.
    #[must_use]
    pub fn local_mem_size(&self) -> usize {
        0
    }
    /// `Device::localMemType()`.
    #[must_use]
    pub fn local_mem_type(&self) -> i32 {
        0
    }
    /// `Device::hostUnifiedMemory()`.
    #[must_use]
    pub fn host_unified_memory(&self) -> bool {
        false
    }
    /// `Device::imageSupport()`.
    #[must_use]
    pub fn image_support(&self) -> bool {
        false
    }
    /// `Device::imageFromBufferSupport()`.
    #[must_use]
    pub fn image_from_buffer_support(&self) -> bool {
        false
    }
    /// `Device::imagePitchAlignment()`.
    #[must_use]
    pub fn image_pitch_alignment(&self) -> u32 {
        0
    }
    /// `Device::imageBaseAddressAlignment()`.
    #[must_use]
    pub fn image_base_address_alignment(&self) -> u32 {
        0
    }
    /// `Device::intelSubgroupsSupport()`.
    #[must_use]
    pub fn intel_subgroups_support(&self) -> bool {
        false
    }
    /// `Device::image2DMaxWidth()`.
    #[must_use]
    pub fn image2d_max_width(&self) -> usize {
        0
    }
    /// `Device::image2DMaxHeight()`.
    #[must_use]
    pub fn image2d_max_height(&self) -> usize {
        0
    }
    /// `Device::image3DMaxWidth()`.
    #[must_use]
    pub fn image3d_max_width(&self) -> usize {
        0
    }
    /// `Device::image3DMaxHeight()`.
    #[must_use]
    pub fn image3d_max_height(&self) -> usize {
        0
    }
    /// `Device::image3DMaxDepth()`.
    #[must_use]
    pub fn image3d_max_depth(&self) -> usize {
        0
    }
    /// `Device::imageMaxBufferSize()`.
    #[must_use]
    pub fn image_max_buffer_size(&self) -> usize {
        0
    }
    /// `Device::imageMaxArraySize()`.
    #[must_use]
    pub fn image_max_array_size(&self) -> usize {
        0
    }
    /// `Device::vendorID()`.
    #[must_use]
    pub fn vendor_id(&self) -> i32 {
        device_caps::UNKNOWN_VENDOR
    }
    /// `Device::isAMD()`.
    #[must_use]
    pub fn is_amd(&self) -> bool {
        self.vendor_id() == device_caps::VENDOR_AMD
    }
    /// `Device::isIntel()`.
    #[must_use]
    pub fn is_intel(&self) -> bool {
        self.vendor_id() == device_caps::VENDOR_INTEL
    }
    /// `Device::isNVidia()`.
    #[must_use]
    pub fn is_nvidia(&self) -> bool {
        self.vendor_id() == device_caps::VENDOR_NVIDIA
    }
    /// `Device::maxClockFrequency()`.
    #[must_use]
    pub fn max_clock_frequency(&self) -> i32 {
        0
    }
    /// `Device::maxComputeUnits()`.
    #[must_use]
    pub fn max_compute_units(&self) -> i32 {
        0
    }
    /// `Device::maxConstantArgs()`.
    #[must_use]
    pub fn max_constant_args(&self) -> i32 {
        0
    }
    /// `Device::maxConstantBufferSize()`.
    #[must_use]
    pub fn max_constant_buffer_size(&self) -> usize {
        0
    }
    /// `Device::maxMemAllocSize()`.
    #[must_use]
    pub fn max_mem_alloc_size(&self) -> usize {
        0
    }
    /// `Device::maxParameterSize()`.
    #[must_use]
    pub fn max_parameter_size(&self) -> usize {
        0
    }
    /// `Device::maxReadImageArgs()`.
    #[must_use]
    pub fn max_read_image_args(&self) -> i32 {
        0
    }
    /// `Device::maxWriteImageArgs()`.
    #[must_use]
    pub fn max_write_image_args(&self) -> i32 {
        0
    }
    /// `Device::maxSamplers()`.
    #[must_use]
    pub fn max_samplers(&self) -> i32 {
        0
    }
    /// `Device::maxWorkGroupSize()`.
    #[must_use]
    pub fn max_work_group_size(&self) -> usize {
        0
    }
    /// `Device::maxWorkItemDims()`.
    #[must_use]
    pub fn max_work_item_dims(&self) -> i32 {
        0
    }
    /// `Device::memBaseAddrAlign()`.
    #[must_use]
    pub fn mem_base_addr_align(&self) -> i32 {
        0
    }
    /// `Device::deviceID()`.
    #[must_use]
    pub fn device_id(&self) -> i32 {
        0
    }
    /// `Device::nativeVectorWidthChar()`.
    #[must_use]
    pub fn native_vector_width_char(&self) -> i32 {
        0
    }
    /// `Device::nativeVectorWidthShort()`.
    #[must_use]
    pub fn native_vector_width_short(&self) -> i32 {
        0
    }
    /// `Device::nativeVectorWidthInt()`.
    #[must_use]
    pub fn native_vector_width_int(&self) -> i32 {
        0
    }
    /// `Device::nativeVectorWidthLong()`.
    #[must_use]
    pub fn native_vector_width_long(&self) -> i32 {
        0
    }
    /// `Device::nativeVectorWidthFloat()`.
    #[must_use]
    pub fn native_vector_width_float(&self) -> i32 {
        0
    }
    /// `Device::nativeVectorWidthDouble()`.
    #[must_use]
    pub fn native_vector_width_double(&self) -> i32 {
        0
    }
    /// `Device::nativeVectorWidthHalf()`.
    #[must_use]
    pub fn native_vector_width_half(&self) -> i32 {
        0
    }
    /// `Device::preferredVectorWidthChar()`.
    #[must_use]
    pub fn preferred_vector_width_char(&self) -> i32 {
        0
    }
    /// `Device::preferredVectorWidthShort()`.
    #[must_use]
    pub fn preferred_vector_width_short(&self) -> i32 {
        0
    }
    /// `Device::preferredVectorWidthInt()`.
    #[must_use]
    pub fn preferred_vector_width_int(&self) -> i32 {
        0
    }
    /// `Device::preferredVectorWidthLong()`.
    #[must_use]
    pub fn preferred_vector_width_long(&self) -> i32 {
        0
    }
    /// `Device::preferredVectorWidthFloat()`.
    #[must_use]
    pub fn preferred_vector_width_float(&self) -> i32 {
        0
    }
    /// `Device::preferredVectorWidthDouble()`.
    #[must_use]
    pub fn preferred_vector_width_double(&self) -> i32 {
        0
    }
    /// `Device::preferredVectorWidthHalf()`.
    #[must_use]
    pub fn preferred_vector_width_half(&self) -> i32 {
        0
    }
    /// `Device::printfBufferSize()`.
    #[must_use]
    pub fn printf_buffer_size(&self) -> usize {
        0
    }
    /// `Device::profilingTimerResolution()`.
    #[must_use]
    pub fn profiling_timer_resolution(&self) -> usize {
        0
    }
}

/// `cv::ocl::OpenCLExecutionContext` — a device, context and queue together.
///
/// Always empty. [`OpenCLExecutionContext::empty`] is true, which is what
/// upstream's callers check before binding one to a thread.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenCLExecutionContext;

impl OpenCLExecutionContext {
    /// `OpenCLExecutionContext::getCurrent()`.
    #[must_use]
    pub fn current() -> OpenCLExecutionContext {
        OpenCLExecutionContext
    }
    /// `OpenCLExecutionContext::getCurrentRef()`.
    #[must_use]
    pub fn current_ref() -> OpenCLExecutionContext {
        OpenCLExecutionContext
    }
    /// `OpenCLExecutionContext::empty()` — always true.
    #[must_use]
    pub fn empty(&self) -> bool {
        true
    }
    /// `OpenCLExecutionContext::getDevice()`.
    #[must_use]
    pub fn get_device(&self) -> Device {
        Device
    }
    /// `OpenCLExecutionContext::bind()` — nothing to bind.
    pub fn bind(&self) {}
    /// `OpenCLExecutionContext::release()`.
    pub fn release(&self) {}
    /// `OpenCLExecutionContext::setUseOpenCL(flag)`.
    pub fn set_use_opencl(&self, _flag: bool) {}
    /// `OpenCLExecutionContext::useOpenCL()` — always false.
    #[must_use]
    pub fn use_opencl(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opencl_is_reported_absent() {
        assert!(!have_opencl());
        assert!(!use_opencl());
        assert!(!have_amd_blas());
        assert!(!have_amd_fft());
        assert!(!have_svm());
        // Enabling it optimistically is accepted and changes nothing, as
        // upstream also refuses to enable what it does not have.
        set_use_opencl(true);
        assert!(!use_opencl());
        finish();
    }

    #[test]
    fn the_null_device_answers_every_query() {
        let d = Device::default_device();
        assert!(!d.available());
        assert_eq!(d.name(), "");
        assert_eq!(d.vendor_id(), device_caps::UNKNOWN_VENDOR);
        assert!(!d.is_amd() && !d.is_intel() && !d.is_nvidia());
        assert_eq!(d.max_compute_units(), 0);
        assert_eq!(d.global_mem_size(), 0);
        assert!(!d.has_fp64() && !d.has_fp16());
        assert!(!d.is_extension_supported("cl_khr_fp64"));
        assert_eq!(d.preferred_vector_width_float(), 0);
        assert_eq!(d.native_vector_width_half(), 0);
        assert!(d.ptr().is_null());
        assert_eq!(Device::new().device_type(), 0);
    }

    #[test]
    fn device_type_constants_match_upstream() {
        use device_type::*;
        assert_eq!(TYPE_DEFAULT, 1);
        assert_eq!(TYPE_CPU, 2);
        assert_eq!(TYPE_GPU, 4);
        assert_eq!(TYPE_ACCELERATOR, 8);
        // The discrete and integrated flags are the GPU bit plus a marker.
        assert_eq!(TYPE_DGPU, TYPE_GPU + (1 << 16));
        assert_eq!(TYPE_IGPU, TYPE_GPU + (1 << 17));
        assert_eq!(TYPE_ALL, u32::MAX);
    }

    #[test]
    fn capability_constants_match_upstream() {
        assert_eq!(fp_config::FP_DENORM, 1);
        assert_eq!(fp_config::FP_CORRECTLY_ROUNDED_DIVIDE_SQRT, 128);
        assert_eq!(device_caps::EXEC_NATIVE_KERNEL, 2);
        assert_eq!(device_caps::READ_WRITE_CACHE, 2);
        assert_eq!(device_caps::LOCAL_IS_GLOBAL, 2);
        assert_eq!(device_caps::VENDOR_NVIDIA, 3);
        // READ_WRITE is the two access bits together.
        assert_eq!(kernel_arg::READ_WRITE, kernel_arg::READ_ONLY | kernel_arg::WRITE_ONLY);
        assert_eq!(kernel_arg::NO_SIZE, 256);
        assert_eq!(OCL_VECTOR_DEFAULT, OclVectorStrategy::Own);
        assert_eq!(OclVectorStrategy::Max as i32, 1);
    }

    #[test]
    fn the_execution_context_is_empty() {
        let c = OpenCLExecutionContext::current();
        assert!(c.empty());
        assert!(!c.use_opencl());
        assert!(!c.get_device().available());
        c.bind();
        c.set_use_opencl(true);
        assert!(!c.use_opencl());
        c.release();
        assert!(OpenCLExecutionContext::current_ref().empty());
    }
}
