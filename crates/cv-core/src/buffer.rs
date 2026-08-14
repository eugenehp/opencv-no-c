//! 64-byte aligned byte buffer, the storage behind [`crate::Mat`].
//!
//! Upstream allocates through `cv::fastMalloc`, which over-allocates and hand-
//! aligns to `CV_MALLOC_ALIGN` (64). We take the alignment from the allocator
//! directly. The guarantee is load-bearing: `Mat` hands out `&[f64]` / `&[i32]`
//! views over these bytes, and a plain `Vec<u8>` is only 1-byte aligned, so
//! those casts would be unsound without it.

use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::ptr::NonNull;

/// `CV_MALLOC_ALIGN` — allocation alignment in bytes.
pub const CV_MALLOC_ALIGN: usize = 64;

/// An owned, zero-initialised, 64-byte aligned byte buffer.
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: `AlignedBuf` uniquely owns its allocation and hands out references
// only through `&self` / `&mut self`, so it carries no thread affinity.
unsafe impl Send for AlignedBuf {}
// SAFETY: see above; shared access is read-only.
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    /// Layout used for `len` bytes. Size is rounded up to a multiple of the
    /// alignment so `Layout` stays valid and `dealloc` can reconstruct it.
    fn layout(len: usize) -> Layout {
        let size = len.max(1).next_multiple_of(CV_MALLOC_ALIGN);
        Layout::from_size_align(size, CV_MALLOC_ALIGN).expect("Mat allocation layout overflow")
    }

    /// Allocate `len` zeroed bytes.
    #[must_use]
    pub fn zeroed(len: usize) -> Self {
        let layout = Self::layout(len);
        // SAFETY: `layout` has non-zero size (rounded up from `len.max(1)`).
        let raw = unsafe { alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(raw) else {
            handle_alloc_error(layout);
        };
        Self { ptr, len }
    }

    /// Length in bytes.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no bytes.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Immutable view of the whole buffer.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` points to `len` initialised (zeroed at minimum) bytes
        // owned by `self`, and the borrow ties the lifetime to `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutable view of the whole buffer.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, plus `&mut self` guarantees exclusivity.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `alloc_zeroed` with exactly this layout and
        // is freed once, here.
        unsafe { dealloc(self.ptr.as_ptr(), Self::layout(self.len)) }
    }
}

impl Clone for AlignedBuf {
    fn clone(&self) -> Self {
        let mut new = Self::zeroed(self.len);
        new.as_mut_slice().copy_from_slice(self.as_slice());
        new
    }
}

impl std::fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuf").field("len", &self.len).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_is_aligned_and_zeroed() {
        for len in [1usize, 7, 64, 65, 4096, 100_003] {
            let b = AlignedBuf::zeroed(len);
            assert_eq!(b.as_slice().as_ptr() as usize % CV_MALLOC_ALIGN, 0);
            assert_eq!(b.len(), len);
            assert!(b.as_slice().iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn clone_copies_contents() {
        let mut a = AlignedBuf::zeroed(32);
        a.as_mut_slice()[5] = 42;
        let b = a.clone();
        assert_eq!(b.as_slice()[5], 42);
        assert_eq!(b.as_slice().as_ptr() as usize % CV_MALLOC_ALIGN, 0);
    }

    #[test]
    fn zero_length_is_allowed() {
        let b = AlignedBuf::zeroed(0);
        assert!(b.is_empty());
        assert_eq!(b.as_slice().len(), 0);
    }
}
