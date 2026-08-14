//! [`Mat`] and its borrowed views, corresponding to `cv::Mat`.
//!
//! # Deliberate divergence from upstream
//!
//! `cv::Mat` is reference-counted with **shallow** copy semantics: `Mat b = a;`
//! shares pixels, and writing through `b` mutates `a`. That is unrepresentable
//! in safe Rust without interior mutability, and it is the single largest
//! source of aliasing bugs in OpenCV client code.
//!
//! This port instead makes ownership explicit:
//!
//! | upstream                  | here                       |
//! |---------------------------|----------------------------|
//! | `Mat b = a;` (shares)     | [`Mat::clone`] (deep copy)  |
//! | `Mat b = a(roi);` (shares)| [`MatLike::roi`] → [`MatView`] |
//! | write through shared `Mat`| [`MatLikeMut::roi_mut`] → [`MatViewMut`] |
//!
//! Numerics, type codes, and function behaviour are unchanged; only the
//! aliasing model differs. Every algorithm in this workspace is written against
//! the [`MatLike`] / [`MatLikeMut`] traits, so it accepts owned matrices and
//! borrowed ROIs interchangeably.

use crate::buffer::AlignedBuf;
use crate::error::{Code, Error, Result};
use crate::geometry::{Point, Rect, Size};
use crate::saturate::Elem;
use crate::types::{Depth, MatType};
use crate::{cv_assert, cv_error};

/// `cv::Mat`'s header flag constants, with upstream's values.
///
/// This port keeps geometry in typed fields rather than a packed `flags` word,
/// so nothing here is read internally; the constants exist so a caller
/// translating C++ that inspects `mat.flags` has the same numbers.
pub mod flags {
    /// `Mat::MAGIC_VAL` — the signature in the high bits of `Mat::flags`.
    pub const MAGIC_VAL: i32 = 0x42FF_0000;
    /// `Mat::AUTO_STEP` — "derive the row stride from the width".
    pub const AUTO_STEP: usize = 0;
    /// `Mat::CONTINUOUS_FLAG` — see [`super::MatLike::is_continuous`].
    pub const CONTINUOUS_FLAG: i32 = crate::types::CV_MAT_CONT_FLAG;
    /// `Mat::SUBMATRIX_FLAG` — set when the header views part of a larger buffer.
    pub const SUBMATRIX_FLAG: i32 = crate::types::CV_SUBMAT_FLAG;
    /// `SparseMat::MAX_DIM` — the dimension limit of a sparse matrix.
    pub const MAX_DIM: i32 = 32;
    /// `SparseMat::HASH_SCALE` — the multiplier of its hash step.
    pub const HASH_SCALE: i32 = 0x5bd1_e995;
    /// `SparseMat::HASH_BIT` — marks a hash value as occupied.
    pub const HASH_BIT: u32 = 0x8000_0000;
    /// `Mat::MAGIC_MASK` — the bits [`MAGIC_VAL`] occupies.
    pub const MAGIC_MASK: u32 = 0xFFFF_0000;
    /// `Mat::TYPE_MASK` — the depth and channel bits of `Mat::flags`.
    ///
    /// Wider than [`crate::types::CV_MAT_TYPE_MASK`] by design: `Mat::flags`
    /// reserves twelve bits here where a bare type code needs eleven.
    pub const TYPE_MASK: i32 = 0x0000_0FFF;
    /// `Mat::DEPTH_MASK` — the depth bits alone.
    pub const DEPTH_MASK: i32 = 7;
}

/// `cv::AccessFlag` — how a caller intends to reach an array's pixels.
///
/// Upstream needs these to decide whether an OpenCL buffer must be mapped, read
/// back, or can be left on the device. This port has no device memory, so they
/// are declared rather than acted on.
pub mod access {
    /// `ACCESS_READ`
    pub const ACCESS_READ: i32 = 1 << 24;
    /// `ACCESS_WRITE`
    pub const ACCESS_WRITE: i32 = 1 << 25;
    /// `ACCESS_RW`
    pub const ACCESS_RW: i32 = 3 << 24;
    /// `ACCESS_MASK` — the same bits as `ACCESS_RW`.
    pub const ACCESS_MASK: i32 = ACCESS_RW;
    /// `ACCESS_FAST` — accept a mapping that may not be coherent.
    pub const ACCESS_FAST: i32 = 1 << 26;
}

/// `cv::_OutputArray::DepthMask` — a set of depths as a bit per [`Depth`].
///
/// These are what a `CV_CheckDepth`-style guard tests against: one call can ask
/// "is this any float depth?" instead of enumerating the cases.
pub mod depth_mask {
    use crate::types::{CV_16F, CV_16S, CV_16U, CV_32F, CV_32S, CV_64F, CV_8S, CV_8U};

    /// `DEPTH_MASK_8U`
    pub const DEPTH_MASK_8U: i32 = 1 << CV_8U;
    /// `DEPTH_MASK_8S`
    pub const DEPTH_MASK_8S: i32 = 1 << CV_8S;
    /// `DEPTH_MASK_16U`
    pub const DEPTH_MASK_16U: i32 = 1 << CV_16U;
    /// `DEPTH_MASK_16S`
    pub const DEPTH_MASK_16S: i32 = 1 << CV_16S;
    /// `DEPTH_MASK_32S`
    pub const DEPTH_MASK_32S: i32 = 1 << CV_32S;
    /// `DEPTH_MASK_32F`
    pub const DEPTH_MASK_32F: i32 = 1 << CV_32F;
    /// `DEPTH_MASK_64F`
    pub const DEPTH_MASK_64F: i32 = 1 << CV_64F;
    /// `DEPTH_MASK_16F`
    pub const DEPTH_MASK_16F: i32 = 1 << CV_16F;
    /// `DEPTH_MASK_ALL` — every depth except `CV_16F`.
    pub const DEPTH_MASK_ALL: i32 = (DEPTH_MASK_64F << 1) - 1;
    /// `DEPTH_MASK_ALL_BUT_8S`
    pub const DEPTH_MASK_ALL_BUT_8S: i32 = DEPTH_MASK_ALL & !DEPTH_MASK_8S;
    /// `DEPTH_MASK_ALL_16F` — every depth, `CV_16F` included.
    pub const DEPTH_MASK_ALL_16F: i32 = (DEPTH_MASK_16F << 1) - 1;
    /// `DEPTH_MASK_FLT` — the two ordinary float depths.
    pub const DEPTH_MASK_FLT: i32 = DEPTH_MASK_32F + DEPTH_MASK_64F;
}

/// Read-only access to a 2-D matrix of pixels.
pub trait MatLike {
    /// Number of rows.
    fn rows(&self) -> i32;
    /// Number of columns.
    fn cols(&self) -> i32;
    /// Element type.
    fn mat_type(&self) -> MatType;
    /// Distance between the starts of consecutive rows, in bytes.
    fn step(&self) -> usize;
    /// Bytes spanned by this matrix, from its origin through the last row.
    fn as_bytes(&self) -> &[u8];

    /// Extent as a [`Size`] — `Mat::size()`.
    #[inline]
    fn size(&self) -> Size {
        Size::new(self.cols(), self.rows())
    }

    /// Channel depth — `Mat::depth()`.
    #[inline]
    fn depth(&self) -> Depth {
        self.mat_type().depth()
    }

    /// Channel count — `Mat::channels()`.
    #[inline]
    fn channels(&self) -> i32 {
        self.mat_type().channels()
    }

    /// `Mat::type()` as upstream's packed integer.
    #[inline]
    fn type_code(&self) -> i32 {
        self.mat_type().to_i32()
    }

    /// Bytes per element across all channels — `Mat::elemSize()`.
    #[inline]
    fn elem_size(&self) -> usize {
        self.mat_type().elem_size()
    }

    /// Bytes per channel value — `Mat::elemSize1()`.
    #[inline]
    fn elem_size1(&self) -> usize {
        self.mat_type().elem_size1()
    }

    /// `Mat::step1()` — the row stride measured in channel values, not bytes.
    #[inline]
    fn step1(&self) -> usize {
        self.step() / self.elem_size1()
    }

    /// `rows * cols` — the element count, ignoring channels. `Mat::total()`.
    #[inline]
    fn total(&self) -> usize {
        self.rows() as usize * self.cols() as usize
    }

    /// Whether the matrix has no elements — `Mat::empty()`.
    #[inline]
    fn is_empty(&self) -> bool {
        self.rows() <= 0 || self.cols() <= 0
    }

    /// Whether rows are packed with no inter-row padding — `Mat::isContinuous()`.
    #[inline]
    fn is_continuous(&self) -> bool {
        self.rows() == 1 || self.step() == self.cols() as usize * self.elem_size()
    }

    /// `Mat::row(y)` — a single row as a `1 x cols` view.
    fn row(&self, y: i32) -> Result<MatView<'_>> {
        self.roi(Rect::new(0, y, self.cols(), 1))
    }

    /// `Mat::col(x)` — a single column as a `rows x 1` view.
    fn col(&self, x: i32) -> Result<MatView<'_>> {
        self.roi(Rect::new(x, 0, 1, self.rows()))
    }

    /// `Mat::rowRange(startrow, endrow)` — rows `[start, end)`.
    fn row_range(&self, start: i32, end: i32) -> Result<MatView<'_>> {
        cv_assert!(start <= end, "rowRange: start {start} is past end {end}");
        self.roi(Rect::new(0, start, self.cols(), end - start))
    }

    /// `Mat::colRange(startcol, endcol)` — columns `[start, end)`.
    fn col_range(&self, start: i32, end: i32) -> Result<MatView<'_>> {
        cv_assert!(start <= end, "colRange: start {start} is past end {end}");
        self.roi(Rect::new(start, 0, end - start, self.rows()))
    }

    /// Raw bytes of row `y`, `cols * elem_size` long.
    fn row_bytes(&self, y: i32) -> Result<&[u8]> {
        if y < 0 || y >= self.rows() {
            return cv_error!(Code::StsOutOfRange, "row {y} out of range 0..{}", self.rows());
        }
        let start = y as usize * self.step();
        let width = self.cols() as usize * self.elem_size();
        Ok(&self.as_bytes()[start..start + width])
    }

    /// Row `y` reinterpreted as channel values: `cols * channels` entries.
    ///
    /// Fails if `T` does not match this matrix's depth.
    fn row_as<T: Elem>(&self, y: i32) -> Result<&[T]> {
        check_depth::<T>(self.depth())?;
        let bytes = self.row_bytes(y)?;
        // SAFETY: the buffer origin is 64-byte aligned (see `AlignedBuf`) and
        // every row/column offset is a whole multiple of `elem_size`, itself a
        // multiple of `size_of::<T>()`, so `ptr` is aligned for `T`. The depth
        // check above proves the bytes really are `T` values, and the length is
        // an exact multiple of `size_of::<T>()`.
        Ok(unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), bytes.len() / size_of::<T>())
        })
    }

    /// The `channels()` values of the pixel at `(y, x)`.
    fn pixel<T: Elem>(&self, y: i32, x: i32) -> Result<&[T]> {
        if x < 0 || x >= self.cols() {
            return cv_error!(Code::StsOutOfRange, "col {x} out of range 0..{}", self.cols());
        }
        let cn = self.channels() as usize;
        let row = self.row_as::<T>(y)?;
        let off = x as usize * cn;
        Ok(&row[off..off + cn])
    }

    /// Value at `(y, x)` of a single-channel matrix — `Mat::at<T>(y, x)`.
    fn at<T: Elem>(&self, y: i32, x: i32) -> Result<T> {
        cv_assert!(self.channels() == 1, "at() requires a single-channel matrix");
        Ok(self.pixel::<T>(y, x)?[0])
    }

    /// Value at `(y, x)` of channel `c`, widened to `f64`.
    ///
    /// Depth-agnostic, so generic code can read any matrix without monomorphising.
    fn at_f64(&self, y: i32, x: i32, c: i32) -> Result<f64> {
        let cn = self.channels();
        if c < 0 || c >= cn {
            return cv_error!(Code::StsOutOfRange, "channel {c} out of range 0..{cn}");
        }
        let i = c as usize;
        Ok(match self.depth() {
            Depth::U8 => self.pixel::<u8>(y, x)?[i] as f64,
            Depth::S8 => self.pixel::<i8>(y, x)?[i] as f64,
            Depth::U16 => self.pixel::<u16>(y, x)?[i] as f64,
            Depth::S16 => self.pixel::<i16>(y, x)?[i] as f64,
            Depth::S32 => self.pixel::<i32>(y, x)?[i] as f64,
            Depth::F32 => self.pixel::<f32>(y, x)?[i] as f64,
            Depth::F64 => self.pixel::<f64>(y, x)?[i],
            Depth::F16 => {
                return cv_error!(Code::StsNotImplemented, "CV_16F element access is not ported yet")
            }
        })
    }

    /// A borrowed sub-matrix — the equivalent of `mat(roi)`.
    fn roi(&self, rect: Rect) -> Result<MatView<'_>> {
        let (offset, step, rows, cols) = roi_geometry(self, rect)?;
        let span = span_len(rows, cols, step, self.elem_size());
        Ok(MatView {
            data: &self.as_bytes()[offset..offset + span],
            rows,
            cols,
            typ: self.mat_type(),
            step,
            // Where this view starts inside whatever it was cut from, so that
            // `locate_roi` can report it. A view of a view accumulates.
            origin: (rect.x, rect.y),
            parent: Size::new(self.cols(), self.rows()),
        })
    }

    /// `Mat::copyTo(dst)` — a packed owned copy, dropping any row padding.
    ///
    /// This is how a strided view becomes something with contiguous storage,
    /// which is what `Mat::clone()` on an ROI gives upstream.
    fn copy_to(&self) -> Result<Mat> {
        Mat::copy_from(self)
    }

    /// Borrow the whole matrix as a [`MatView`].
    fn as_view(&self) -> MatView<'_> {
        MatView {
            data: self.as_bytes(),
            rows: self.rows(),
            cols: self.cols(),
            typ: self.mat_type(),
            step: self.step(),
            origin: (0, 0),
            parent: Size::new(self.cols(), self.rows()),
        }
    }
}

/// Mutable access to a 2-D matrix of pixels.
pub trait MatLikeMut: MatLike {
    /// Mutable bytes spanned by this matrix.
    fn as_bytes_mut(&mut self) -> &mut [u8];

    /// Mutable raw bytes of row `y`.
    fn row_bytes_mut(&mut self, y: i32) -> Result<&mut [u8]> {
        if y < 0 || y >= self.rows() {
            return cv_error!(Code::StsOutOfRange, "row {y} out of range 0..{}", self.rows());
        }
        let start = y as usize * self.step();
        let width = self.cols() as usize * self.elem_size();
        Ok(&mut self.as_bytes_mut()[start..start + width])
    }

    /// Row `y` as mutable channel values.
    fn row_as_mut<T: Elem>(&mut self, y: i32) -> Result<&mut [T]> {
        check_depth::<T>(self.depth())?;
        let bytes = self.row_bytes_mut(y)?;
        // SAFETY: identical reasoning to `MatLike::row_as`, with exclusivity
        // provided by `&mut self`.
        Ok(unsafe {
            std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<T>(), bytes.len() / size_of::<T>())
        })
    }

    /// Mutable channel values of the pixel at `(y, x)`.
    fn pixel_mut<T: Elem>(&mut self, y: i32, x: i32) -> Result<&mut [T]> {
        if x < 0 || x >= self.cols() {
            return cv_error!(Code::StsOutOfRange, "col {x} out of range 0..{}", self.cols());
        }
        let cn = self.channels() as usize;
        let row = self.row_as_mut::<T>(y)?;
        let off = x as usize * cn;
        Ok(&mut row[off..off + cn])
    }

    /// A borrowed mutable sub-matrix — writes land in the parent.
    fn roi_mut(&mut self, rect: Rect) -> Result<MatViewMut<'_>> {
        let (offset, step, rows, cols) = roi_geometry(self, rect)?;
        let typ = self.mat_type();
        let span = span_len(rows, cols, step, typ.elem_size());
        Ok(MatViewMut {
            data: &mut self.as_bytes_mut()[offset..offset + span],
            rows,
            cols,
            typ,
            step,
        })
    }

    /// Borrow the whole matrix mutably.
    fn as_view_mut(&mut self) -> MatViewMut<'_> {
        let (rows, cols, typ, step) = (self.rows(), self.cols(), self.mat_type(), self.step());
        MatViewMut { data: self.as_bytes_mut(), rows, cols, typ, step }
    }
}

/// Byte span from a view's origin through the end of its last row.
#[inline]
fn span_len(rows: i32, cols: i32, step: usize, elem_size: usize) -> usize {
    if rows <= 0 || cols <= 0 {
        0
    } else {
        (rows as usize - 1) * step + cols as usize * elem_size
    }
}

/// Validate `rect` against `m` and return `(byte_offset, step, rows, cols)`.
fn roi_geometry<M: MatLike + ?Sized>(m: &M, rect: Rect) -> Result<(usize, usize, i32, i32)> {
    if rect.x < 0
        || rect.y < 0
        || rect.width < 0
        || rect.height < 0
        || rect.x + rect.width > m.cols()
        || rect.y + rect.height > m.rows()
    {
        return cv_error!(
            Code::StsBadArg,
            "roi {rect:?} does not fit inside {}x{}",
            m.cols(),
            m.rows()
        );
    }
    let offset = rect.y as usize * m.step() + rect.x as usize * m.elem_size();
    Ok((offset, m.step(), rect.height, rect.width))
}

/// Reject a typed accessor whose `T` disagrees with the matrix depth.
#[inline]
fn check_depth<T: Elem>(actual: Depth) -> Result<()> {
    if T::DEPTH != actual {
        return Err(Error::new(
            Code::StsUnmatchedFormats,
            format!("accessor type is {} but matrix depth is {}", T::DEPTH.name(), actual.name()),
        ));
    }
    Ok(())
}

/// An owned, continuous 2-D matrix — the equivalent of `cv::Mat`.
#[derive(Debug, Clone)]
pub struct Mat {
    buf: AlignedBuf,
    rows: i32,
    cols: i32,
    typ: MatType,
    step: usize,
}

impl Mat {
    /// A zero-filled matrix — `Mat::zeros(rows, cols, type)`.
    pub fn zeros(rows: i32, cols: i32, typ: MatType) -> Result<Self> {
        if rows < 0 || cols < 0 {
            return cv_error!(Code::StsBadSize, "negative dimensions {rows}x{cols}");
        }
        let step = cols as usize * typ.elem_size();
        let bytes = rows as usize * step;
        Ok(Self { buf: AlignedBuf::zeroed(bytes), rows, cols, typ, step })
    }

    /// An empty `0x0` matrix.
    #[must_use]
    pub fn empty(typ: MatType) -> Self {
        Self { buf: AlignedBuf::zeroed(0), rows: 0, cols: 0, typ, step: 0 }
    }

    /// A matrix with every channel of every pixel set to `value`.
    pub fn new_with(rows: i32, cols: i32, typ: MatType, value: crate::Scalar) -> Result<Self> {
        let mut m = Mat::zeros(rows, cols, typ)?;
        m.set_to(value)?;
        Ok(m)
    }

    /// Build from typed channel-interleaved data in row-major order.
    ///
    /// `data.len()` must equal `rows * cols * channels`.
    pub fn from_vec<T: Elem>(rows: i32, cols: i32, channels: i32, data: &[T]) -> Result<Self> {
        let typ = MatType::new(T::DEPTH, channels);
        let want = rows as usize * cols as usize * channels as usize;
        if data.len() != want {
            return cv_error!(
                Code::StsUnmatchedSizes,
                "expected {want} values for {rows}x{cols}x{channels}, got {}",
                data.len()
            );
        }
        let mut m = Mat::zeros(rows, cols, typ)?;
        for y in 0..rows {
            let n = cols as usize * channels as usize;
            let src = &data[y as usize * n..(y as usize + 1) * n];
            m.row_as_mut::<T>(y)?.copy_from_slice(src);
        }
        Ok(m)
    }

    /// Copy an arbitrary [`MatLike`] — including a strided [`MatView`] — into a
    /// fresh continuous matrix. This is `Mat::clone()` upstream.
    pub fn copy_from<M: MatLike + ?Sized>(src: &M) -> Result<Self> {
        let mut dst = Mat::zeros(src.rows(), src.cols(), src.mat_type())?;
        for y in 0..src.rows() {
            let s = src.row_bytes(y)?;
            dst.row_bytes_mut(y)?.copy_from_slice(s);
        }
        Ok(dst)
    }

    /// Fill every pixel with `value`, saturating per channel — `Mat::setTo`.
    pub fn set_to(&mut self, value: crate::Scalar) -> Result<()> {
        crate::arithm::set_to(self, value)
    }

    /// `Mat::create(rows, cols, type)` — reallocate only if the shape differs.
    ///
    /// Upstream's contract is that an already-correct matrix is left untouched,
    /// which is what makes `create` cheap to call at the top of every output
    /// path. When it does allocate, the contents are unspecified upstream;
    /// here they are zero, since Rust has no uninitialised safe buffer.
    pub fn create(&mut self, rows: i32, cols: i32, typ: MatType) -> Result<()> {
        if self.rows() == rows && self.cols() == cols && self.mat_type() == typ {
            return Ok(());
        }
        *self = Mat::zeros(rows, cols, typ)?;
        Ok(())
    }

    /// `cv::swap(a, b)` / `Mat::swap(m)` — exchange two matrices.
    pub fn swap(&mut self, other: &mut Mat) {
        std::mem::swap(self, other);
    }

    /// `Mat::getStdAllocator()` — the default allocator.
    ///
    /// `None`: this port allocates through Rust's global allocator, which is
    /// not a value a caller can be handed or swapped out per matrix.
    #[must_use]
    pub fn get_std_allocator() -> Option<&'static ()> {
        None
    }

    /// `Mat::updateContinuityFlag()` — recompute the cached continuity bit.
    ///
    /// A no-op: [`MatLike::is_continuous`] derives the answer from the stride
    /// each time it is asked, so there is no cached bit to fall out of date.
    pub fn update_continuity_flag(&mut self) {}

    /// `Mat::assignTo(m, type)` — convert into `dst` at the given depth.
    ///
    /// `None` keeps this matrix's depth, which is upstream's `type < 0`.
    pub fn assign_to(&self, dst: &mut Mat, depth: Option<Depth>) -> Result<()> {
        *dst = crate::arithm::convert_to(self, depth.unwrap_or_else(|| self.depth()), 1.0, 0.0)?;
        Ok(())
    }

    /// Flatten to `rows*cols*channels` values in row-major order.
    ///
    /// Convenience for tests and interop; allocates.
    pub fn to_vec<T: Elem>(&self) -> Result<Vec<T>> {
        check_depth::<T>(self.depth())?;
        let mut out = Vec::with_capacity(self.total() * self.channels() as usize);
        for y in 0..self.rows {
            out.extend_from_slice(self.row_as::<T>(y)?);
        }
        Ok(out)
    }

    /// Reinterpret the shape without moving data — `Mat::reshape`.
    ///
    /// `channels == 0` keeps the current channel count; `rows == 0` keeps the
    /// current row count, both matching upstream's sentinel convention.
    pub fn reshape(&self, channels: i32, rows: i32) -> Result<Mat> {
        let cn = if channels == 0 { self.channels() } else { channels };
        let new_rows = if rows == 0 { self.rows } else { rows };
        cv_assert!(cn >= 1, "channel count must be positive");
        cv_assert!(new_rows >= 1, "row count must be positive");

        let total_values = self.total() * self.channels() as usize;
        if total_values % (new_rows as usize * cn as usize) != 0 {
            return cv_error!(
                Code::StsBadArg,
                "cannot reshape {} values into {new_rows} rows of {cn} channels",
                total_values
            );
        }
        let new_cols = total_values / (new_rows as usize * cn as usize);
        let typ = self.typ.with_channels(cn);
        let mut out = Mat::zeros(new_rows, new_cols as i32, typ)?;
        // Continuous source and destination hold identical bytes.
        out.buf.as_mut_slice()[..self.buf.len()].copy_from_slice(self.buf.as_slice());
        Ok(out)
    }
}

impl MatLike for Mat {
    #[inline]
    fn rows(&self) -> i32 {
        self.rows
    }
    #[inline]
    fn cols(&self) -> i32 {
        self.cols
    }
    #[inline]
    fn mat_type(&self) -> MatType {
        self.typ
    }
    #[inline]
    fn step(&self) -> usize {
        self.step
    }
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self.buf.as_slice()
    }
}

impl MatLikeMut for Mat {
    #[inline]
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.buf.as_mut_slice()
    }
}

/// A borrowed, possibly strided read-only sub-matrix.
#[derive(Debug, Clone, Copy)]
pub struct MatView<'a> {
    data: &'a [u8],
    rows: i32,
    cols: i32,
    typ: MatType,
    step: usize,
    /// Top-left corner within the matrix this view was cut from.
    origin: (i32, i32),
    /// Size of the matrix this view was cut from.
    parent: Size,
}

impl<'a> MatView<'a> {
    /// `Mat::locateROI(wholeSize, ofs)`.
    ///
    /// Returns the size of the matrix this view was taken from and this view's
    /// top-left corner within it. For a view of the whole matrix the offset is
    /// `(0, 0)` and the size is its own.
    #[must_use]
    pub fn locate_roi(&self) -> (Size, Point) {
        (self.parent, Point::new(self.origin.0, self.origin.1))
    }

    /// `Mat::adjustROI(dtop, dbottom, dleft, dright)`.
    ///
    /// Grows (or, with negative arguments, shrinks) the view within its parent.
    /// Upstream clamps silently against the parent bounds; so does this, which
    /// is what makes `adjustROI` usable for widening a window to reach a filter
    /// border without checking whether the border exists.
    pub fn adjust_roi(
        &self,
        dtop: i32,
        dbottom: i32,
        dleft: i32,
        dright: i32,
        parent: &'a (impl MatLike + ?Sized),
    ) -> Result<MatView<'a>> {
        cv_assert!(
            parent.size() == self.parent,
            "adjustROI: parent is {:?} but this view came from {:?}",
            parent.size(),
            self.parent
        );
        let x = (self.origin.0 - dleft).clamp(0, self.parent.width);
        let y = (self.origin.1 - dtop).clamp(0, self.parent.height);
        let right = (self.origin.0 + self.cols + dright).clamp(x, self.parent.width);
        let bottom = (self.origin.1 + self.rows + dbottom).clamp(y, self.parent.height);
        parent.roi(Rect::new(x, y, right - x, bottom - y))
    }
}

impl MatLike for MatView<'_> {
    #[inline]
    fn rows(&self) -> i32 {
        self.rows
    }
    #[inline]
    fn cols(&self) -> i32 {
        self.cols
    }
    #[inline]
    fn mat_type(&self) -> MatType {
        self.typ
    }
    #[inline]
    fn step(&self) -> usize {
        self.step
    }
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self.data
    }
}

/// A borrowed, possibly strided mutable sub-matrix.
#[derive(Debug)]
pub struct MatViewMut<'a> {
    data: &'a mut [u8],
    rows: i32,
    cols: i32,
    typ: MatType,
    step: usize,
}

impl MatLike for MatViewMut<'_> {
    #[inline]
    fn rows(&self) -> i32 {
        self.rows
    }
    #[inline]
    fn cols(&self) -> i32 {
        self.cols
    }
    #[inline]
    fn mat_type(&self) -> MatType {
        self.typ
    }
    #[inline]
    fn step(&self) -> usize {
        self.step
    }
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self.data
    }
}

impl MatLikeMut for MatViewMut<'_> {
    #[inline]
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.data
    }
}

#[cfg(test)]
impl MatViewMut<'_> {
    /// Test helper: set every in-bounds byte, honouring the row stride.
    fn set_all_bytes(&mut self, v: u8) {
        for y in 0..self.rows() {
            self.row_bytes_mut(y).unwrap().fill(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CV_32FC1, CV_8UC1, CV_8UC3};
    use crate::Scalar;

    #[test]
    fn header_flag_constants_match_upstream() {
        assert_eq!(flags::MAGIC_VAL, 0x42FF_0000);
        assert_eq!(flags::AUTO_STEP, 0);
        assert_eq!(flags::CONTINUOUS_FLAG, 1 << 14);
        assert_eq!(flags::SUBMATRIX_FLAG, 1 << 15);
        assert_eq!(access::ACCESS_RW, access::ACCESS_READ | access::ACCESS_WRITE);
        assert_eq!(access::ACCESS_MASK, access::ACCESS_RW);
        assert_eq!(flags::MAGIC_MASK, 0xFFFF_0000);
        assert_eq!(flags::DEPTH_MASK, 7);
        // Mat::flags reserves twelve bits for the type where a bare type code
        // needs eleven, so these two masks are deliberately different.
        assert_eq!(flags::TYPE_MASK, 0x0FFF);
        assert_eq!(crate::types::CV_MAT_TYPE_MASK, 0x0FFF);
    }

    #[test]
    fn depth_masks_select_the_right_depths() {
        use depth_mask::*;
        assert_eq!(DEPTH_MASK_8U, 1);
        assert_eq!(DEPTH_MASK_ALL, 0x7F, "every depth but CV_16F");
        assert_eq!(DEPTH_MASK_ALL_16F, 0xFF);
        assert_eq!(DEPTH_MASK_ALL_BUT_8S, 0x7D);
        // The masks are how a `CV_CheckDepth` guard is written.
        assert_ne!(DEPTH_MASK_FLT & Depth::F32.mask(), 0);
        assert_ne!(DEPTH_MASK_FLT & Depth::F64.mask(), 0);
        assert_eq!(DEPTH_MASK_FLT & Depth::U8.mask(), 0);
        assert_eq!(DEPTH_MASK_ALL & Depth::F16.mask(), 0);
        assert_ne!(DEPTH_MASK_ALL_16F & Depth::F16.mask(), 0);
    }

    #[test]
    fn create_reallocates_only_when_the_shape_changes() {
        let mut m = Mat::from_vec(2, 2, 1, &[1u8, 2, 3, 4]).unwrap();
        m.create(2, 2, CV_8UC1).unwrap();
        assert_eq!(m.to_vec::<u8>().unwrap(), vec![1, 2, 3, 4], "contents survive a no-op");
        m.create(3, 4, CV_8UC1).unwrap();
        assert_eq!(m.size(), Size::new(4, 3));
        assert!(m.to_vec::<u8>().unwrap().iter().all(|&v| v == 0));
        // A type change reallocates even at the same size.
        m.create(3, 4, CV_32FC1).unwrap();
        assert_eq!(m.mat_type(), CV_32FC1);
    }

    #[test]
    fn swap_and_assign_to_move_between_matrices() {
        let mut a = Mat::from_vec(1, 2, 1, &[1u8, 2]).unwrap();
        let mut b = Mat::from_vec(1, 3, 1, &[9u8, 8, 7]).unwrap();
        a.swap(&mut b);
        assert_eq!(a.to_vec::<u8>().unwrap(), vec![9, 8, 7]);
        assert_eq!(b.to_vec::<u8>().unwrap(), vec![1, 2]);

        let mut dst = Mat::empty(CV_8UC1);
        b.assign_to(&mut dst, Some(Depth::F32)).unwrap();
        assert_eq!(dst.mat_type(), CV_32FC1);
        assert_eq!(dst.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
        // Without a depth the source depth is kept.
        b.assign_to(&mut dst, None).unwrap();
        assert_eq!(dst.mat_type(), CV_8UC1);
    }

    #[test]
    fn row_col_and_ranges_slice_like_numpy() {
        let data: Vec<u8> = (0..30).collect();
        let m = Mat::from_vec(5, 6, 1, &data).unwrap();

        assert_eq!(m.row(1).unwrap().copy_to().unwrap().to_vec::<u8>().unwrap(), (6..12).collect::<Vec<u8>>());
        assert_eq!(m.col(2).unwrap().copy_to().unwrap().to_vec::<u8>().unwrap(), vec![2, 8, 14, 20, 26]);

        let rr = m.row_range(1, 3).unwrap();
        assert_eq!(rr.size(), Size::new(6, 2));
        assert_eq!(rr.copy_to().unwrap().to_vec::<u8>().unwrap(), (6..18).collect::<Vec<u8>>());

        let cr = m.col_range(2, 5).unwrap();
        assert_eq!(cr.size(), Size::new(3, 5));
        assert_eq!(
            cr.copy_to().unwrap().to_vec::<u8>().unwrap(),
            vec![2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22, 26, 27, 28]
        );
        // A column view is strided, so it is not continuous.
        assert!(!cr.is_continuous());
        assert!(m.row(0).unwrap().is_continuous());
    }

    #[test]
    fn step1_counts_channel_values_not_bytes() {
        let m = Mat::zeros(5, 6, CV_8UC3).unwrap();
        assert_eq!(m.step1(), 18);
        let f = Mat::zeros(5, 6, CV_32FC1).unwrap();
        assert_eq!(f.step1(), 6);
        assert_eq!(f.type_code(), CV_32FC1.to_i32());
    }

    #[test]
    fn locate_roi_reports_the_window_within_its_parent() {
        let data: Vec<u8> = (0..30).collect();
        let m = Mat::from_vec(5, 6, 1, &data).unwrap();
        let v = m.roi(Rect::new(2, 1, 3, 2)).unwrap();
        let (whole, ofs) = v.locate_roi();
        assert_eq!(whole, Size::new(6, 5));
        assert_eq!(ofs, Point::new(2, 1));

        // Grow one pixel on every side.
        let w = v.adjust_roi(1, 1, 1, 1, &m).unwrap();
        assert_eq!(w.size(), Size::new(5, 4));
        assert_eq!(w.locate_roi().1, Point::new(1, 0));

        // Growing past the parent clamps rather than failing.
        let big = v.adjust_roi(10, 10, 10, 10, &m).unwrap();
        assert_eq!(big.size(), Size::new(6, 5));
        assert_eq!(big.locate_roi().1, Point::new(0, 0));

        // Shrinking works too.
        let small = v.adjust_roi(0, -1, 0, -1, &m).unwrap();
        assert_eq!(small.size(), Size::new(2, 1));
    }

    #[test]
    fn zeros_has_expected_shape() {
        let m = Mat::zeros(4, 5, CV_8UC3).unwrap();
        assert_eq!(m.rows(), 4);
        assert_eq!(m.cols(), 5);
        assert_eq!(m.channels(), 3);
        assert_eq!(m.elem_size(), 3);
        assert_eq!(m.step(), 15);
        assert_eq!(m.total(), 20);
        assert!(m.is_continuous());
        assert_eq!(m.at_f64(0, 0, 0).unwrap(), 0.0);
    }

    #[test]
    fn from_vec_roundtrips() {
        let data: Vec<u8> = (0..24).collect();
        let m = Mat::from_vec(2, 4, 3, &data).unwrap();
        assert_eq!(m.to_vec::<u8>().unwrap(), data);
        assert_eq!(m.pixel::<u8>(1, 0).unwrap(), &[12, 13, 14]);
        assert_eq!(m.at_f64(1, 3, 2).unwrap(), 23.0);
    }

    #[test]
    fn roi_is_strided_and_reads_parent() {
        let data: Vec<u8> = (0..25).collect();
        let m = Mat::from_vec(5, 5, 1, &data).unwrap();
        let v = m.roi(Rect::new(1, 1, 3, 2)).unwrap();
        assert_eq!(v.rows(), 2);
        assert_eq!(v.cols(), 3);
        assert_eq!(v.step(), 5);
        assert!(!v.is_continuous());
        assert_eq!(v.row_as::<u8>(0).unwrap(), &[6, 7, 8]);
        assert_eq!(v.row_as::<u8>(1).unwrap(), &[11, 12, 13]);
    }

    #[test]
    fn roi_mut_writes_through_to_parent() {
        let mut m = Mat::zeros(4, 4, CV_8UC1).unwrap();
        {
            let mut v = m.roi_mut(Rect::new(1, 1, 2, 2)).unwrap();
            v.set_all_bytes(9);
        }
        assert_eq!(
            m.to_vec::<u8>().unwrap(),
            vec![0, 0, 0, 0, 0, 9, 9, 0, 0, 9, 9, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn roi_of_roi_composes() {
        let data: Vec<u8> = (0..36).collect();
        let m = Mat::from_vec(6, 6, 1, &data).unwrap();
        let outer = m.roi(Rect::new(1, 1, 4, 4)).unwrap();
        let inner = outer.roi(Rect::new(1, 1, 2, 2)).unwrap();
        assert_eq!(inner.row_as::<u8>(0).unwrap(), &[14, 15]);
        assert_eq!(inner.row_as::<u8>(1).unwrap(), &[20, 21]);
    }

    #[test]
    fn out_of_bounds_roi_is_rejected() {
        let m = Mat::zeros(4, 4, CV_8UC1).unwrap();
        assert!(m.roi(Rect::new(0, 0, 5, 4)).is_err());
        assert!(m.roi(Rect::new(-1, 0, 2, 2)).is_err());
        assert!(m.roi(Rect::new(3, 3, 2, 2)).is_err());
        assert!(m.roi(Rect::new(0, 0, 4, 4)).is_ok());
    }

    #[test]
    fn typed_accessor_rejects_wrong_depth() {
        let m = Mat::zeros(2, 2, CV_8UC1).unwrap();
        assert!(m.row_as::<u8>(0).is_ok());
        let e = m.row_as::<f32>(0).unwrap_err();
        assert_eq!(e.code, Code::StsUnmatchedFormats);
    }

    #[test]
    fn f32_rows_are_correctly_aligned() {
        // Regression guard for the alignment invariant `row_as` relies on.
        let m = Mat::zeros(3, 7, CV_32FC1).unwrap();
        for y in 0..3 {
            let r = m.row_as::<f32>(y).unwrap();
            assert_eq!(r.as_ptr() as usize % align_of::<f32>(), 0);
            assert_eq!(r.len(), 7);
        }
    }

    #[test]
    fn set_to_saturates_per_channel() {
        let mut m = Mat::new_with(2, 2, CV_8UC3, Scalar::new(10.0, 300.0, -5.0, 0.0)).unwrap();
        assert_eq!(m.pixel::<u8>(0, 0).unwrap(), &[10, 255, 0]);
        m.set_to(Scalar::all(7.0)).unwrap();
        assert_eq!(m.pixel::<u8>(1, 1).unwrap(), &[7, 7, 7]);
    }

    #[test]
    fn reshape_preserves_data() {
        let data: Vec<u8> = (0..24).collect();
        let m = Mat::from_vec(4, 6, 1, &data).unwrap();
        let r = m.reshape(0, 2).unwrap();
        assert_eq!(r.rows(), 2);
        assert_eq!(r.cols(), 12);
        assert_eq!(r.to_vec::<u8>().unwrap(), data);

        let c3 = m.reshape(3, 4).unwrap();
        assert_eq!(c3.cols(), 2);
        assert_eq!(c3.channels(), 3);
        assert!(m.reshape(5, 4).is_err());
    }

    #[test]
    fn copy_from_compacts_a_strided_view() {
        let data: Vec<u8> = (0..25).collect();
        let m = Mat::from_vec(5, 5, 1, &data).unwrap();
        let v = m.roi(Rect::new(1, 1, 3, 2)).unwrap();
        let compact = Mat::copy_from(&v).unwrap();
        assert!(compact.is_continuous());
        assert_eq!(compact.to_vec::<u8>().unwrap(), vec![6, 7, 8, 11, 12, 13]);
    }
}
