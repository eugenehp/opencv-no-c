//! `cv::UMat` — the transparent-API matrix
//! (`modules/core/include/opencv2/core/mat.hpp`).
//!
//! `UMat` is `Mat` with the option of living in device memory. When OpenCV is
//! built without OpenCL — or has it disabled at runtime — every `UMat` is
//! backed by ordinary host memory and every operation runs the CPU kernel.
//! cv2's CPU-only wheel demonstrates this: `cv2.blur(cv2.UMat(img), (3,3))`
//! works and returns a `UMat`.
//!
//! That fallback is the whole implementation here. A [`UMat`] owns a [`Mat`],
//! [`UMat::get`] hands it back, and the usage flags are recorded so a round
//! trip preserves what the caller asked for. Nothing is lost relative to a
//! device-less OpenCV, because a device-less OpenCV does exactly this.
//!
//! The port's own functions take [`crate::MatLike`], which [`UMat`] implements,
//! so a `UMat` can be passed anywhere a `Mat` can without unwrapping.

use crate::error::Result;
use crate::mat::{Mat, MatLike, MatLikeMut};
use crate::types::MatType;

/// `cv::UMatUsageFlags` — a hint about where the buffer should live.
///
/// Recorded and reported; with no device there is only one place it can go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum UMatUsageFlags {
    /// `USAGE_DEFAULT` — let the allocator decide.
    #[default]
    Default = 0,
    /// `USAGE_ALLOCATE_HOST_MEMORY`
    AllocateHostMemory = 1 << 0,
    /// `USAGE_ALLOCATE_DEVICE_MEMORY`
    AllocateDeviceMemory = 1 << 1,
    /// `USAGE_ALLOCATE_SHARED_MEMORY`
    ///
    /// Not the same as host and device or-ed together, as upstream's comment
    /// is careful to say: it asks for one mapping visible to both.
    AllocateSharedMemory = 1 << 2,
}

/// `__UMAT_USAGE_FLAGS_32BIT` — upstream's hint that the enum is 32 bits wide.
pub const UMAT_USAGE_FLAGS_32BIT: i32 = 0x7fff_ffff;

/// `cv::UMatData::MemoryFlag` — the state of a buffer shared with a device.
///
/// None of these can be set here: with no device there is never a second copy
/// to fall out of date.
pub mod memory_flag {
    /// `UMatData::COPY_ON_MAP`
    pub const COPY_ON_MAP: i32 = 1;
    /// `UMatData::HOST_COPY_OBSOLETE` — the host copy needs refreshing.
    pub const HOST_COPY_OBSOLETE: i32 = 2;
    /// `UMatData::DEVICE_COPY_OBSOLETE` — the device copy needs refreshing.
    pub const DEVICE_COPY_OBSOLETE: i32 = 4;
    /// `UMatData::TEMP_UMAT` — a temporary wrapper around someone else's buffer.
    pub const TEMP_UMAT: i32 = 8;
    /// `UMatData::TEMP_COPIED_UMAT` — a temporary that owns a copy.
    pub const TEMP_COPIED_UMAT: i32 = 24;
    /// `UMatData::USER_ALLOCATED`
    pub const USER_ALLOCATED: i32 = 32;
    /// `UMatData::DEVICE_MEM_MAPPED`
    pub const DEVICE_MEM_MAPPED: i32 = 64;
    /// `UMatData::ASYNC_CLEANUP` — release the buffer off the calling thread.
    pub const ASYNC_CLEANUP: i32 = 128;
}

/// The `_InputArray::KindFlag` entries for the array types this port does not
/// hold natively, completing the set begun in [`crate::bindings_utils`].
pub mod kind {
    use crate::bindings_utils::KIND_SHIFT;

    /// `_InputArray::OPENGL_BUFFER`
    pub const OPENGL_BUFFER: i32 = 7 << KIND_SHIFT;
    /// `_InputArray::CUDA_HOST_MEM`
    pub const CUDA_HOST_MEM: i32 = 8 << KIND_SHIFT;
    /// `_InputArray::CUDA_GPU_MAT`
    pub const CUDA_GPU_MAT: i32 = 9 << KIND_SHIFT;
    /// `_InputArray::UMAT`
    pub const UMAT: i32 = 10 << KIND_SHIFT;
    /// `_InputArray::STD_VECTOR_UMAT`
    pub const STD_VECTOR_UMAT: i32 = 11 << KIND_SHIFT;
    /// `_InputArray::STD_VECTOR_CUDA_GPU_MAT`
    pub const STD_VECTOR_CUDA_GPU_MAT: i32 = 13 << KIND_SHIFT;
    /// `_InputArray::CUDA_GPU_MATND`
    pub const CUDA_GPU_MATND: i32 = 16 << KIND_SHIFT;
}

/// `cv::UMat`.
#[derive(Debug, Clone)]
pub struct UMat {
    mat: Mat,
    usage: UMatUsageFlags,
}

impl UMat {
    /// `UMat::UMat(usageFlags)` — an empty matrix.
    #[must_use]
    pub fn new(usage: UMatUsageFlags) -> Self {
        UMat { mat: Mat::empty(crate::types::CV_8UC1), usage }
    }

    /// `UMat::UMat(rows, cols, type, usageFlags)`.
    pub fn zeros(rows: i32, cols: i32, typ: MatType, usage: UMatUsageFlags) -> Result<Self> {
        Ok(UMat { mat: Mat::zeros(rows, cols, typ)?, usage })
    }

    /// `Mat::getUMat(accessFlags, usageFlags)` — wrap an existing matrix.
    #[must_use]
    pub fn from_mat(mat: Mat, usage: UMatUsageFlags) -> Self {
        UMat { mat, usage }
    }

    /// `UMat::getMat(accessFlags)` — borrow the host-side matrix.
    #[must_use]
    pub fn get(&self) -> &Mat {
        &self.mat
    }

    /// Mutable access to the host-side matrix.
    #[must_use]
    pub fn get_mut(&mut self) -> &mut Mat {
        &mut self.mat
    }

    /// Unwrap to the owned [`Mat`].
    #[must_use]
    pub fn into_mat(self) -> Mat {
        self.mat
    }

    /// `UMat::usageFlags` — what the caller asked for at construction.
    #[must_use]
    pub fn usage_flags(&self) -> UMatUsageFlags {
        self.usage
    }

    /// `UMat::release()` — drop the contents, leaving an empty matrix.
    pub fn release(&mut self) {
        self.mat = Mat::empty(self.mat.mat_type());
    }

    /// `UMat::clone()` — a deep copy, as it is for [`Mat`] here.
    pub fn try_clone(&self) -> Result<UMat> {
        Ok(UMat { mat: Mat::copy_from(&self.mat)?, usage: self.usage })
    }
}

// Delegating to the inner `Mat` is what makes a `UMat` usable anywhere the
// port takes a matrix, which is the point of the transparent API.
impl MatLike for UMat {
    fn rows(&self) -> i32 {
        self.mat.rows()
    }
    fn cols(&self) -> i32 {
        self.mat.cols()
    }
    fn mat_type(&self) -> MatType {
        self.mat.mat_type()
    }
    fn step(&self) -> usize {
        self.mat.step()
    }
    fn as_bytes(&self) -> &[u8] {
        self.mat.as_bytes()
    }
}

impl MatLikeMut for UMat {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.mat.as_bytes_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CV_8UC1, CV_8UC3};

    #[test]
    fn a_umat_round_trips_through_a_mat() {
        let m = Mat::from_vec(3, 4, 1, &(0u8..12).collect::<Vec<u8>>()).unwrap();
        let u = UMat::from_mat(m, UMatUsageFlags::Default);
        assert_eq!(u.get().to_vec::<u8>().unwrap(), (0u8..12).collect::<Vec<u8>>());
        assert_eq!(u.size(), crate::Size::new(4, 3));
        assert_eq!(u.usage_flags(), UMatUsageFlags::Default);
        assert_eq!(u.into_mat().rows(), 3);
    }

    /// The point of the transparent API: a `UMat` goes where a `Mat` goes.
    #[test]
    fn a_umat_is_accepted_by_the_ports_own_functions() {
        let m = Mat::from_vec(1, 4, 1, &[1u8, 2, 3, 4]).unwrap();
        let u = UMat::from_mat(m, UMatUsageFlags::AllocateHostMemory);
        // `norm` takes any `MatLike`.
        assert_eq!(crate::norm(&u, crate::NormType::L1).unwrap(), 10.0);
        // As does anything writing through `MatLikeMut`.
        let mut u2 = UMat::zeros(2, 2, CV_8UC1, UMatUsageFlags::Default).unwrap();
        u2.get_mut().set_to(crate::Scalar::from(5.0)).unwrap();
        assert_eq!(u2.get().to_vec::<u8>().unwrap(), vec![5, 5, 5, 5]);
    }

    #[test]
    fn release_empties_but_keeps_the_type() {
        let mut u = UMat::zeros(2, 3, CV_8UC3, UMatUsageFlags::Default).unwrap();
        assert!(!u.is_empty());
        u.release();
        assert!(u.is_empty());
        assert_eq!(u.mat_type(), CV_8UC3);
    }

    #[test]
    fn clone_is_a_deep_copy() {
        let mut a = UMat::zeros(1, 2, CV_8UC1, UMatUsageFlags::Default).unwrap();
        let b = a.try_clone().unwrap();
        a.get_mut().set_to(crate::Scalar::from(9.0)).unwrap();
        assert_eq!(b.get().to_vec::<u8>().unwrap(), vec![0, 0], "the copy is independent");
        assert_eq!(a.get().to_vec::<u8>().unwrap(), vec![9, 9]);
    }

    #[test]
    fn the_flag_values_match_upstream() {
        assert_eq!(UMatUsageFlags::Default as i32, 0);
        assert_eq!(UMatUsageFlags::AllocateHostMemory as i32, 1);
        assert_eq!(UMatUsageFlags::AllocateDeviceMemory as i32, 2);
        assert_eq!(UMatUsageFlags::AllocateSharedMemory as i32, 4);
        assert_eq!(UMAT_USAGE_FLAGS_32BIT, i32::MAX);
        assert_eq!(memory_flag::TEMP_COPIED_UMAT, 24);
        assert_eq!(memory_flag::ASYNC_CLEANUP, 128);
        // The kind flags continue the sequence the bindings module starts.
        assert_eq!(kind::UMAT, 10 << crate::bindings_utils::KIND_SHIFT);
        assert_eq!(kind::CUDA_GPU_MATND, 16 << crate::bindings_utils::KIND_SHIFT);
        assert_eq!(kind::OPENGL_BUFFER, 7 << crate::bindings_utils::KIND_SHIFT);
    }
}
