//! `cv::Mat`'s sharing semantics, for code being ported from C++.
//!
//! Enabled by the `compat` feature. Nothing here changes the behaviour of
//! anything outside this module — it adds a type, it does not alter one.
//!
//! # Why this is a separate type and not a flag on [`Mat`]
//!
//! Cargo features are additive and unify across the whole dependency graph: if
//! any crate anywhere enables a feature, every crate gets it. A feature that
//! changed what `Mat::clone` means would therefore silently change the
//! behaviour of unrelated dependents that never asked for it, and the compiler
//! would not flag a single call site. So the sharing semantics live in a new
//! type, which is a change no existing code can be surprised by.
//!
//! # What `cv::Mat` actually does
//!
//! ```cpp
//! Mat a = Mat::zeros(10, 10, CV_8UC1);
//! Mat b = a;                   // shares storage — no pixels copied
//! b.at<uchar>(0, 0) = 7;       // a is now 7 at (0, 0) too
//! Mat c = a.clone();           // deep copy — independent
//! a(Rect(2, 2, 4, 4)).setTo(0) // writes through into a
//! ```
//!
//! That is what [`SharedMat`] reproduces:
//!
//! ```
//! use cv_core::compat::SharedMat;
//! use cv_core::{Mat, MatLike, Rect, Scalar, types::CV_8UC1};
//!
//! let a = SharedMat::new(Mat::zeros(10, 10, CV_8UC1)?);
//! let b = a.clone();                       // shares, as `Mat b = a;` does
//! b.roi(Rect::new(2, 2, 4, 4))?.set_to(Scalar::all(7.0))?;
//! assert_eq!(a.borrow().at_f64(3, 3, 0)?, 7.0);   // written through
//! assert_eq!(a.borrow().at_f64(0, 0, 0)?, 0.0);   // outside the ROI
//!
//! let c = a.deep_clone()?;                 // `Mat::clone()` — independent
//! a.set_to(Scalar::all(1.0))?;
//! assert_eq!(c.borrow().at_f64(0, 0, 0)?, 0.0);
//! # Ok::<(), cv_core::Error>(())
//! ```
//!
//! # What this does not fix
//!
//! Two of the three things that stop this crate being a drop-in are untouched,
//! and no feature flag can touch them:
//!
//! * **Errors are still returned, not thrown.** A C++ caller expects a throw;
//!   turning `Result` into a panic would be worse, not closer.
//! * **Names are still snake_case.** No C++ or `cv2` source compiles unchanged
//!   against this crate, whatever the aliasing model.
//!
//! # Threading
//!
//! `cv::Mat`'s reference count is atomic. This one is not: [`SharedMat`] is
//! `Rc`-based and single-threaded, and conflicting borrows panic at the point
//! of conflict rather than racing. That is a deliberate trade — a panic naming
//! the offending borrow is more useful than a lock that deadlocks, and code
//! sharing one matrix across threads was already relying on the C++ contract
//! that concurrent writes are the caller's problem.

use std::rc::Rc;
use core::cell::{Ref, RefCell, RefMut};

use crate::error::Result;
use crate::geometry::{Rect, Scalar, Size};
use crate::mat::{Mat, MatLike, MatLikeMut};
use crate::types::MatType;
use crate::{cv_assert, cv_error, Code};

/// A matrix handle with `cv::Mat`'s sharing semantics.
///
/// Cloning shares storage; [`SharedMat::deep_clone`] is what `cv::Mat::clone()`
/// does. A handle addresses a rectangle of the underlying buffer, so
/// [`SharedMat::roi`] yields another handle onto the same pixels.
#[derive(Debug, Clone)]
pub struct SharedMat {
    inner: Rc<RefCell<Mat>>,
    /// The region of `inner` this handle addresses, in absolute coordinates.
    rect: Rect,
}

impl SharedMat {
    /// Take ownership of `m` and hand back a shared handle onto it.
    #[must_use]
    pub fn new(m: Mat) -> Self {
        let rect = Rect::new(0, 0, m.cols(), m.rows());
        SharedMat { inner: Rc::new(RefCell::new(m)), rect }
    }

    /// `m(rect)` — another handle onto the same storage.
    ///
    /// `rect` is relative to this handle, as it is upstream, so ROIs compose.
    ///
    /// # Errors
    ///
    /// [`Code::StsBadArg`] if `rect` does not fit inside this handle.
    pub fn roi(&self, rect: Rect) -> Result<SharedMat> {
        if rect.x < 0
            || rect.y < 0
            || rect.width < 0
            || rect.height < 0
            || rect.x + rect.width > self.rect.width
            || rect.y + rect.height > self.rect.height
        {
            return cv_error!(
                Code::StsBadArg,
                "roi {rect:?} does not fit inside {}x{}",
                self.rect.width,
                self.rect.height
            );
        }
        Ok(SharedMat {
            inner: Rc::clone(&self.inner),
            rect: Rect::new(
                self.rect.x + rect.x,
                self.rect.y + rect.y,
                rect.width,
                rect.height,
            ),
        })
    }

    /// `cv::Mat::clone()` — an independent deep copy of the region this handle
    /// addresses, packed, with any row padding dropped.
    ///
    /// # Errors
    ///
    /// Propagates whatever copying the region fails with.
    pub fn deep_clone(&self) -> Result<SharedMat> {
        Ok(SharedMat::new(Mat::copy_from(&self.borrow())?))
    }

    /// How many handles share this storage — `Mat::u->refcount`.
    #[must_use]
    pub fn share_count(&self) -> usize {
        Rc::strong_count(&self.inner)
    }

    /// Whether two handles address the same allocation, regardless of region.
    #[must_use]
    pub fn shares_with(&self, other: &SharedMat) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }

    /// The region of the underlying buffer this handle addresses.
    #[must_use]
    pub fn rect(&self) -> Rect {
        self.rect
    }

    /// Rows in this handle's region.
    #[must_use]
    pub fn rows(&self) -> i32 {
        self.rect.height
    }

    /// Columns in this handle's region.
    #[must_use]
    pub fn cols(&self) -> i32 {
        self.rect.width
    }

    /// Extent as a [`Size`].
    #[must_use]
    pub fn size(&self) -> Size {
        Size::new(self.rect.width, self.rect.height)
    }

    /// Element type.
    #[must_use]
    pub fn mat_type(&self) -> MatType {
        self.inner.borrow().mat_type()
    }

    /// Borrow the region for reading.
    ///
    /// # Panics
    ///
    /// If another handle onto the same storage is currently borrowed mutably.
    #[must_use]
    pub fn borrow(&self) -> SharedRef<'_> {
        SharedRef { m: self.inner.borrow(), rect: self.rect }
    }

    /// Borrow the region for writing.
    ///
    /// # Panics
    ///
    /// If another handle onto the same storage is currently borrowed at all.
    #[must_use]
    pub fn borrow_mut(&self) -> SharedRefMut<'_> {
        SharedRefMut { m: self.inner.borrow_mut(), rect: self.rect }
    }

    /// `Mat::setTo(value)`, writing through into the shared storage.
    ///
    /// # Errors
    ///
    /// Propagates whatever the fill fails with.
    pub fn set_to(&self, value: Scalar) -> Result<()> {
        let mut g = self.borrow_mut();
        crate::arithm::set_to(&mut g, value)
    }

    /// Take the underlying matrix back, if this is the only handle left.
    ///
    /// # Errors
    ///
    /// [`Code::StsError`] if the storage is still shared, because unsharing it
    /// would have to copy and that is what [`SharedMat::deep_clone`] is for.
    pub fn into_mat(self) -> Result<Mat> {
        match Rc::try_unwrap(self.inner) {
            Ok(cell) => Ok(cell.into_inner()),
            Err(_) => cv_error!(
                Code::StsError,
                "the storage is still shared; use deep_clone to get an owned copy"
            ),
        }
    }
}

impl From<Mat> for SharedMat {
    fn from(m: Mat) -> Self {
        SharedMat::new(m)
    }
}

/// A read borrow of a [`SharedMat`]'s region.
///
/// Implements [`MatLike`], so every function in the crate takes one.
#[derive(Debug)]
pub struct SharedRef<'a> {
    m: Ref<'a, Mat>,
    rect: Rect,
}

/// A write borrow of a [`SharedMat`]'s region.
///
/// Implements [`MatLikeMut`], so every mutating function takes one.
#[derive(Debug)]
pub struct SharedRefMut<'a> {
    m: RefMut<'a, Mat>,
    rect: Rect,
}

/// Byte offset of a region's origin, and the span through its last row.
fn geometry(m: &Mat, rect: Rect) -> (usize, usize) {
    let offset = rect.y as usize * m.step() + rect.x as usize * m.elem_size();
    let span = if rect.height == 0 || rect.width == 0 {
        0
    } else {
        (rect.height as usize - 1) * m.step() + rect.width as usize * m.elem_size()
    };
    (offset, span)
}

impl MatLike for SharedRef<'_> {
    fn rows(&self) -> i32 {
        self.rect.height
    }
    fn cols(&self) -> i32 {
        self.rect.width
    }
    fn mat_type(&self) -> MatType {
        self.m.mat_type()
    }
    fn step(&self) -> usize {
        self.m.step()
    }
    fn as_bytes(&self) -> &[u8] {
        let (offset, span) = geometry(&self.m, self.rect);
        &self.m.as_bytes()[offset..offset + span]
    }
}

impl MatLike for SharedRefMut<'_> {
    fn rows(&self) -> i32 {
        self.rect.height
    }
    fn cols(&self) -> i32 {
        self.rect.width
    }
    fn mat_type(&self) -> MatType {
        self.m.mat_type()
    }
    fn step(&self) -> usize {
        self.m.step()
    }
    fn as_bytes(&self) -> &[u8] {
        let (offset, span) = geometry(&self.m, self.rect);
        &self.m.as_bytes()[offset..offset + span]
    }
}

impl MatLikeMut for SharedRefMut<'_> {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        let (offset, span) = geometry(&self.m, self.rect);
        &mut self.m.as_bytes_mut()[offset..offset + span]
    }
}

/// `cv::Mat::zeros(rows, cols, type)` as a shared handle.
///
/// # Errors
///
/// Propagates whatever allocation fails with.
pub fn zeros(rows: i32, cols: i32, ty: MatType) -> Result<SharedMat> {
    cv_assert!(rows >= 0 && cols >= 0, "size must not be negative");
    Ok(SharedMat::new(Mat::zeros(rows, cols, ty)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CV_8UC1, CV_8UC3};

    fn ramp() -> SharedMat {
        let mut m = Mat::zeros(6, 8, CV_8UC1).unwrap();
        for y in 0..6 {
            for x in 0..8 {
                crate::arithm::set_channel_generic(&mut m, y, x, 0, f64::from(y * 8 + x)).unwrap();
            }
        }
        SharedMat::new(m)
    }

    #[test]
    fn assignment_shares_storage_the_way_cv_mat_does() {
        let a = ramp();
        let b = a.clone();
        assert!(a.shares_with(&b));
        assert_eq!(a.share_count(), 2);
        b.set_to(Scalar::all(9.0)).unwrap();
        assert_eq!(a.borrow().at_f64(0, 0, 0).unwrap(), 9.0);
    }

    #[test]
    fn clone_is_deep_only_when_asked() {
        let a = ramp();
        let c = a.deep_clone().unwrap();
        assert!(!a.shares_with(&c));
        a.set_to(Scalar::all(1.0)).unwrap();
        // `c` kept the ramp; a deep copy is what `cv::Mat::clone()` gives.
        assert_eq!(c.borrow().at_f64(1, 1, 0).unwrap(), 9.0);
    }

    #[test]
    fn writing_through_an_roi_reaches_the_parent() {
        let a = ramp();
        let r = a.roi(Rect::new(2, 1, 3, 2)).unwrap();
        r.set_to(Scalar::all(0.0)).unwrap();
        // Inside the ROI.
        assert_eq!(a.borrow().at_f64(1, 2, 0).unwrap(), 0.0);
        assert_eq!(a.borrow().at_f64(2, 4, 0).unwrap(), 0.0);
        // Just outside it, on every side.
        assert_eq!(a.borrow().at_f64(0, 2, 0).unwrap(), 2.0);
        assert_eq!(a.borrow().at_f64(1, 1, 0).unwrap(), 9.0);
        assert_eq!(a.borrow().at_f64(1, 5, 0).unwrap(), 13.0);
        assert_eq!(a.borrow().at_f64(3, 2, 0).unwrap(), 26.0);
    }

    #[test]
    fn rois_compose_relative_to_their_parent() {
        let a = ramp();
        let outer = a.roi(Rect::new(2, 1, 5, 4)).unwrap();
        let inner = outer.roi(Rect::new(1, 1, 2, 2)).unwrap();
        // (2,1) then (1,1) lands at (3,2) of the original, not (1,1).
        assert_eq!(inner.rect(), Rect::new(3, 2, 2, 2));
        inner.set_to(Scalar::all(0.0)).unwrap();
        assert_eq!(a.borrow().at_f64(2, 3, 0).unwrap(), 0.0);
        assert_eq!(a.borrow().at_f64(2, 2, 0).unwrap(), 18.0);
    }

    #[test]
    fn an_roi_reports_its_own_size_not_the_parents() {
        let a = ramp();
        let r = a.roi(Rect::new(1, 1, 4, 3)).unwrap();
        assert_eq!((r.rows(), r.cols()), (3, 4));
        assert_eq!(r.borrow().rows(), 3);
        assert_eq!(r.borrow().cols(), 4);
        // The step stays the parent's, which is what makes it a view.
        assert_eq!(r.borrow().step(), a.borrow().step());
    }

    #[test]
    fn an_roi_outside_the_parent_is_refused() {
        let a = ramp();
        assert!(a.roi(Rect::new(0, 0, 9, 1)).is_err());
        assert!(a.roi(Rect::new(-1, 0, 2, 2)).is_err());
        let r = a.roi(Rect::new(4, 0, 4, 6)).unwrap();
        // Relative to the ROI, not the original: 4 columns are left, not 8.
        assert!(r.roi(Rect::new(0, 0, 5, 1)).is_err());
        assert!(r.roi(Rect::new(0, 0, 4, 1)).is_ok());
    }

    #[test]
    fn the_borrow_guards_satisfy_the_matrix_traits() {
        // The point of the guards: existing functions take them unchanged.
        let a = SharedMat::new(Mat::zeros(4, 4, CV_8UC3).unwrap());
        let b = SharedMat::new(Mat::zeros(4, 4, CV_8UC3).unwrap());
        b.set_to(Scalar::new(1.0, 2.0, 3.0, 0.0)).unwrap();
        let sum = crate::arithm::add(&a.borrow(), &b.borrow()).unwrap();
        assert_eq!(sum.at_f64(0, 0, 1).unwrap(), 2.0);
    }

    #[test]
    fn into_mat_succeeds_only_when_unshared() {
        let a = ramp();
        let b = a.clone();
        assert!(a.into_mat().is_err());
        assert!(b.into_mat().is_ok());
    }

    #[test]
    fn zeros_builds_a_handle_of_the_requested_shape() {
        let m = zeros(3, 5, CV_8UC1).unwrap();
        assert_eq!(m.size(), Size::new(5, 3));
        assert_eq!(m.mat_type(), CV_8UC1);
        assert_eq!(m.borrow().at_f64(2, 4, 0).unwrap(), 0.0);
    }
}
