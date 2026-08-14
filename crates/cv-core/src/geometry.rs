//! Geometric primitives from `core/include/opencv2/core/types.hpp`:
//! `Point_`, `Size_`, `Rect_`, `Range`, `Scalar_`.

use core::ops::{Add, Mul, Sub};

/// `cv::Point_<T>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Point_<T> {
    /// Horizontal coordinate.
    pub x: T,
    /// Vertical coordinate.
    pub y: T,
}

impl<T> Point_<T> {
    /// Construct a point.
    #[inline]
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T: Copy + Add<Output = T> + Mul<Output = T>> Point_<T> {
    /// Dot product, as `Point_::dot`.
    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y
    }
}

/// `cv::Point` — integer point.
pub type Point = Point_<i32>;
/// `cv::Point2f`.
pub type Point2f = Point_<f32>;
/// `cv::Point2d`.
pub type Point2d = Point_<f64>;

/// `cv::Size_<T>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Size_<T> {
    /// Width in pixels.
    pub width: T,
    /// Height in pixels.
    pub height: T,
}

impl<T> Size_<T> {
    /// Construct a size.
    #[inline]
    pub const fn new(width: T, height: T) -> Self {
        Self { width, height }
    }
}

impl Size_<i32> {
    /// `Size_::area`. Returns `i64` so large images cannot silently overflow —
    /// upstream returns a size_t-width value for the same reason.
    #[inline]
    #[must_use]
    pub const fn area(self) -> i64 {
        self.width as i64 * self.height as i64
    }

    /// `Size_::empty`.
    #[inline]
    #[must_use]
    pub const fn empty(self) -> bool {
        self.width <= 0 || self.height <= 0
    }
}

/// `cv::Size` — integer size.
pub type Size = Size_<i32>;
/// `cv::Size2f`.
pub type Size2f = Size_<f32>;
/// `cv::Size2d`.
pub type Size2d = Size_<f64>;

/// `cv::Rect_<T>` — origin plus extent, half-open on the far edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Rect_<T> {
    /// Left edge.
    pub x: T,
    /// Top edge.
    pub y: T,
    /// Extent along x.
    pub width: T,
    /// Extent along y.
    pub height: T,
}

impl<T> Rect_<T> {
    /// Construct a rectangle.
    #[inline]
    pub const fn new(x: T, y: T, width: T, height: T) -> Self {
        Self { x, y, width, height }
    }
}

impl<T: Copy + Add<Output = T>> Rect_<T> {
    /// Top-left corner — `Rect_::tl`.
    #[inline]
    pub fn tl(self) -> Point_<T> {
        Point_::new(self.x, self.y)
    }

    /// Bottom-right corner (exclusive) — `Rect_::br`.
    #[inline]
    pub fn br(self) -> Point_<T> {
        Point_::new(self.x + self.width, self.y + self.height)
    }

    /// Extent — `Rect_::size`.
    #[inline]
    pub fn size(self) -> Size_<T> {
        Size_::new(self.width, self.height)
    }
}

impl Rect_<i32> {
    /// `Rect_::area`.
    #[inline]
    #[must_use]
    pub const fn area(self) -> i64 {
        self.width as i64 * self.height as i64
    }

    /// `Rect_::empty`.
    #[inline]
    #[must_use]
    pub const fn empty(self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    /// `Rect_::contains` — half-open on the right and bottom edges.
    #[inline]
    #[must_use]
    pub const fn contains(self, pt: Point) -> bool {
        self.x <= pt.x && pt.x < self.x + self.width && self.y <= pt.y && pt.y < self.y + self.height
    }

    /// Intersection — `operator&`. An empty result is normalised to all-zero,
    /// exactly as upstream does.
    #[must_use]
    pub fn intersect(self, other: Self) -> Self {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);
        let r = Rect_::new(x1, y1, x2 - x1, y2 - y1);
        if r.empty() {
            Rect_::default()
        } else {
            r
        }
    }

    /// Bounding union — `operator|`.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        if self.empty() {
            return other;
        }
        if other.empty() {
            return self;
        }
        let x1 = self.x.min(other.x);
        let y1 = self.y.min(other.y);
        let x2 = (self.x + self.width).max(other.x + other.width);
        let y2 = (self.y + self.height).max(other.y + other.height);
        Rect_::new(x1, y1, x2 - x1, y2 - y1)
    }
}

/// `cv::Rect` — integer rectangle.
pub type Rect = Rect_<i32>;
/// `cv::Rect2f`.
pub type Rect2f = Rect_<f32>;
/// `cv::Rect2d`.
pub type Rect2d = Rect_<f64>;

/// `cv::Range` — half-open `[start, end)` index interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Range {
    /// Inclusive lower bound.
    pub start: i32,
    /// Exclusive upper bound.
    pub end: i32,
}

impl Range {
    /// Construct a range.
    #[inline]
    #[must_use]
    pub const fn new(start: i32, end: i32) -> Self {
        Self { start, end }
    }

    /// `Range::all()` — the sentinel spanning a whole dimension.
    #[inline]
    #[must_use]
    pub const fn all() -> Self {
        Self { start: i32::MIN, end: i32::MAX }
    }

    /// Whether this is the [`Range::all`] sentinel.
    #[inline]
    #[must_use]
    pub const fn is_all(self) -> bool {
        self.start == i32::MIN && self.end == i32::MAX
    }

    /// `Range::size`.
    #[inline]
    #[must_use]
    pub const fn size(self) -> i32 {
        self.end - self.start
    }

    /// `Range::empty`.
    #[inline]
    #[must_use]
    pub const fn empty(self) -> bool {
        self.start >= self.end
    }

    /// Resolve [`Range::all`] against a concrete dimension length.
    #[inline]
    #[must_use]
    pub const fn resolve(self, len: i32) -> Self {
        if self.is_all() {
            Range::new(0, len)
        } else {
            self
        }
    }
}

/// `cv::Scalar` — a 4-element `f64` tuple used for constants and per-channel values.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Scalar(pub [f64; 4]);

impl Scalar {
    /// Construct from four components.
    #[inline]
    #[must_use]
    pub const fn new(v0: f64, v1: f64, v2: f64, v3: f64) -> Self {
        Scalar([v0, v1, v2, v3])
    }

    /// `Scalar::all` — every channel set to the same value.
    #[inline]
    #[must_use]
    pub const fn all(v: f64) -> Self {
        Scalar([v; 4])
    }

    /// A scalar with only the first channel set; the rest are zero.
    #[inline]
    #[must_use]
    pub const fn single(v: f64) -> Self {
        Scalar([v, 0.0, 0.0, 0.0])
    }
}

impl core::ops::Index<usize> for Scalar {
    type Output = f64;
    #[inline]
    fn index(&self, i: usize) -> &f64 {
        &self.0[i]
    }
}

impl core::ops::IndexMut<usize> for Scalar {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.0[i]
    }
}

impl From<f64> for Scalar {
    #[inline]
    fn from(v: f64) -> Self {
        Scalar::single(v)
    }
}

impl From<[f64; 4]> for Scalar {
    #[inline]
    fn from(v: [f64; 4]) -> Self {
        Scalar(v)
    }
}

impl Add for Scalar {
    type Output = Scalar;
    #[inline]
    fn add(self, o: Scalar) -> Scalar {
        Scalar([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2], self.0[3] + o.0[3]])
    }
}

impl Sub for Scalar {
    type Output = Scalar;
    #[inline]
    fn sub(self, o: Scalar) -> Scalar {
        Scalar([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2], self.0[3] - o.0[3]])
    }
}

/// `cv::TermCriteria` — when an iterative algorithm should stop.
///
/// Both conditions may be active at once, in which case whichever fires first
/// ends the loop; that is upstream's rule, and it is why `type` is a bit set
/// rather than a choice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TermCriteria {
    /// Bit set of [`term_criteria::COUNT`] and [`term_criteria::EPS`].
    pub ty: i32,
    /// `maxCount` — iteration limit, read when `COUNT` is set.
    pub max_count: i32,
    /// `epsilon` — required accuracy, read when `EPS` is set.
    pub epsilon: f64,
}

/// `cv::TermCriteria::Type`.
pub mod term_criteria {
    /// `TermCriteria::COUNT` — stop after `maxCount` iterations.
    pub const COUNT: i32 = 1;
    /// `TermCriteria::MAX_ITER` — upstream's synonym for `COUNT`.
    pub const MAX_ITER: i32 = COUNT;
    /// `TermCriteria::EPS` — stop once the change falls below `epsilon`.
    pub const EPS: i32 = 2;
}

impl Default for TermCriteria {
    /// `TermCriteria()` — no condition set, so nothing ever stops the loop.
    fn default() -> Self {
        Self { ty: 0, max_count: 0, epsilon: 0.0 }
    }
}

impl TermCriteria {
    /// `cv::TermCriteria::TermCriteria(type, maxCount, epsilon)`.
    #[must_use]
    pub const fn new(ty: i32, max_count: i32, epsilon: f64) -> Self {
        Self { ty, max_count, epsilon }
    }

    /// `cv::TermCriteria::isValid()`.
    ///
    /// True when at least one enabled condition can actually fire: a `COUNT`
    /// with a positive limit, or an `EPS` with a non-negative tolerance.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let count_ok = (self.ty & term_criteria::COUNT) != 0 && self.max_count > 0;
        let eps_ok = (self.ty & term_criteria::EPS) != 0 && self.epsilon >= 0.0;
        count_ok || eps_ok
    }

    /// The iteration limit, or `fallback` when `COUNT` is not enabled.
    #[must_use]
    pub fn max_count_or(&self, fallback: i32) -> i32 {
        if (self.ty & term_criteria::COUNT) != 0 {
            self.max_count
        } else {
            fallback
        }
    }

    /// The accuracy target, or `fallback` when `EPS` is not enabled.
    #[must_use]
    pub fn epsilon_or(&self, fallback: f64) -> f64 {
        if (self.ty & term_criteria::EPS) != 0 {
            self.epsilon
        } else {
            fallback
        }
    }
}

/// `cv::RotatedRect` — a rectangle with a rotation about its centre.
///
/// Lives here rather than in `cv-imgproc` because upstream declares it in
/// `core/types.hpp`: `minAreaRect` and `fitEllipse` produce it, but `calib3d`
/// and `objdetect` consume it too.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotatedRect {
    /// Centre, in pixels.
    pub center: (f64, f64),
    /// `(width, height)` before rotation.
    pub size: (f64, f64),
    /// Rotation in **degrees**, clockwise, matching upstream.
    pub angle: f64,
}

impl RotatedRect {
    /// `cv::RotatedRect::RotatedRect(center, size, angle)`.
    #[must_use]
    pub const fn new(center: (f64, f64), size: (f64, f64), angle: f64) -> Self {
        Self { center, size, angle }
    }

    /// `cv::RotatedRect::points(pts)` / `cv::boxPoints`.
    ///
    /// Returns the four corners starting from the one upstream calls `pt[0]`
    /// (the lower-left in a y-down image) and proceeding clockwise.
    #[must_use]
    pub fn points(&self) -> [(f64, f64); 4] {
        let a = self.angle.to_radians();
        let (s, c) = (a.sin(), a.cos());
        let (hw, hh) = (self.size.0 * 0.5, self.size.1 * 0.5);
        let (cx, cy) = self.center;
        let corner = |dx: f64, dy: f64| (cx + dx * c - dy * s, cy + dx * s + dy * c);
        [corner(-hw, hh), corner(-hw, -hh), corner(hw, -hh), corner(hw, hh)]
    }

    /// Area of the rectangle.
    #[must_use]
    pub fn area(&self) -> f64 {
        self.size.0 * self.size.1
    }

    /// `cv::RotatedRect::boundingRect2f()` — the exact axis-aligned bounds.
    #[must_use]
    pub fn bounding_rect2f(&self) -> Rect2f {
        let p = self.points();
        let (mut x0, mut y0) = (f64::INFINITY, f64::INFINITY);
        let (mut x1, mut y1) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (x, y) in p {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        Rect2f::new(x0 as f32, y0 as f32, (x1 - x0) as f32, (y1 - y0) as f32)
    }

    /// `cv::RotatedRect::boundingRect()` — integer bounds that *contain* the
    /// rectangle.
    ///
    /// Upstream floors the top-left and ceils the bottom-right, then widens by
    /// one on each side; the extra pixel is deliberate, so that drawing the
    /// rectangle antialiased stays inside the returned box.
    #[must_use]
    pub fn bounding_rect(&self) -> Rect {
        let p = self.points();
        let x0 = p.iter().map(|q| q.0).fold(f64::INFINITY, f64::min).floor() as i32;
        let y0 = p.iter().map(|q| q.1).fold(f64::INFINITY, f64::min).floor() as i32;
        let x1 = p.iter().map(|q| q.0).fold(f64::NEG_INFINITY, f64::max).ceil() as i32;
        let y1 = p.iter().map(|q| q.1).fold(f64::NEG_INFINITY, f64::max).ceil() as i32;
        Rect::new(x0 - 1, y0 - 1, x1 - x0 + 2, y1 - y0 + 2)
    }
}

/// `cv::KeyPoint` — a salient point produced by a feature detector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyPoint {
    /// `pt` — subpixel location.
    pub pt: (f32, f32),
    /// `size` — diameter of the meaningful neighbourhood.
    pub size: f32,
    /// `angle` — orientation in degrees clockwise from the x axis, or `-1`.
    pub angle: f32,
    /// `response` — detector strength, used to rank keypoints.
    pub response: f32,
    /// `octave` — pyramid level the point was found at.
    pub octave: i32,
    /// `class_id` — caller-defined grouping, or `-1`.
    pub class_id: i32,
}

impl KeyPoint {
    /// `cv::KeyPoint::KeyPoint(x, y, size)`.
    ///
    /// `angle` defaults to `-1` (undefined) and `class_id` to `-1`, as upstream.
    #[must_use]
    pub const fn new(x: f32, y: f32, size: f32) -> Self {
        Self { pt: (x, y), size, angle: -1.0, response: 0.0, octave: 0, class_id: -1 }
    }

    /// `cv::KeyPoint::convert(keypoints, points2f, keypointIndexes)` —
    /// the point of each keypoint.
    ///
    /// With `indexes` empty every keypoint is converted, in order; otherwise
    /// only the ones named, in the order given. A negative index is an error,
    /// as it is upstream, rather than a placeholder point.
    pub fn convert(keypoints: &[KeyPoint], indexes: &[i32]) -> crate::Result<Vec<(f32, f32)>> {
        if indexes.is_empty() {
            return Ok(keypoints.iter().map(|k| k.pt).collect());
        }
        let mut out = Vec::with_capacity(indexes.len());
        for &idx in indexes {
            if idx < 0 {
                return crate::cv_error!(
                    crate::Code::StsBadArg,
                    "keypointIndexes has element < 0"
                );
            }
            let k = keypoints.get(idx as usize).ok_or_else(|| {
                crate::Error::new(
                    crate::Code::StsOutOfRange,
                    format!("keypoint index {idx} is past the end of a {}-element set", keypoints.len()),
                )
            })?;
            out.push(k.pt);
        }
        Ok(out)
    }

    /// `cv::KeyPoint::convert(points2f, keypoints, size, response, octave, class_id)`
    /// — the reverse, giving every point the same attributes.
    #[must_use]
    pub fn convert_from(
        points: &[(f32, f32)],
        size: f32,
        response: f32,
        octave: i32,
        class_id: i32,
    ) -> Vec<KeyPoint> {
        points
            .iter()
            .map(|&pt| KeyPoint { pt, size, angle: -1.0, response, octave, class_id })
            .collect()
    }

    /// `cv::KeyPoint::overlap(kp1, kp2)` — shared area over combined area.
    ///
    /// Each keypoint is the disc of diameter `size` centred on `pt`, and the
    /// result is the ratio of their intersection to their union: 0 when they
    /// do not touch, 1 when they coincide.
    ///
    /// Note the union upstream divides by is `π(a² + b²) − intersection`, not
    /// the usual `π(a² + b²) − 2·intersection`. That double-counts nothing but
    /// makes the ratio smaller than a strict Jaccard index would be; it is
    /// reproduced because matching benchmarks are calibrated against it.
    #[must_use]
    pub fn overlap(kp1: &KeyPoint, kp2: &KeyPoint) -> f32 {
        let (a, b) = (kp1.size * 0.5, kp2.size * 0.5);
        let (a_2, b_2) = (a * a, b * b);
        let (dx, dy) = (kp1.pt.0 - kp2.pt.0, kp1.pt.1 - kp2.pt.1);
        let c = (f64::from(dx) * f64::from(dx) + f64::from(dy) * f64::from(dy)).sqrt() as f32;

        // One disc entirely inside the other: the ratio is of their areas.
        if a.min(b) + c <= a.max(b) {
            return a_2.min(b_2) / a_2.max(b_2);
        }
        if c >= a + b {
            return 0.0;
        }

        let c_2 = c * c;
        let cos_alpha = (b_2 + c_2 - a_2) / (kp2.size * c);
        let cos_beta = (a_2 + c_2 - b_2) / (kp1.size * c);
        let (alpha, beta) = (cos_alpha.acos(), cos_beta.acos());
        let (sin_alpha, sin_beta) = (alpha.sin(), beta.sin());

        let segment_area = a_2 * beta + b_2 * alpha;
        let triangle_area = a_2 * sin_beta * cos_beta + b_2 * sin_alpha * cos_alpha;
        let intersection = segment_area - triangle_area;
        let union = (a_2 + b_2) * std::f32::consts::PI - intersection;
        intersection / union
    }

    /// `cv::KeyPoint::hash()`.
    #[must_use]
    pub fn hash(&self) -> usize {
        // Upstream multiplies the bit patterns of the fields together.
        let mut h = self.pt.0.to_bits() as usize;
        h = h.wrapping_mul(self.pt.1.to_bits() as usize);
        h = h.wrapping_mul(self.size.to_bits() as usize);
        h = h.wrapping_mul(self.angle.to_bits() as usize);
        h = h.wrapping_mul(self.response.to_bits() as usize);
        h = h.wrapping_mul(self.octave as usize);
        h.wrapping_mul(self.class_id as usize)
    }
}

/// `cv::DMatch` — one descriptor correspondence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DMatch {
    /// `queryIdx` — index into the query descriptor set.
    pub query_idx: i32,
    /// `trainIdx` — index into the train descriptor set.
    pub train_idx: i32,
    /// `imgIdx` — index of the train image, for multi-image matching.
    pub img_idx: i32,
    /// `distance` — descriptor distance; smaller is a better match.
    pub distance: f32,
}

impl DMatch {
    /// `cv::DMatch::DMatch(queryIdx, trainIdx, distance)`.
    #[must_use]
    pub const fn new(query_idx: i32, train_idx: i32, distance: f32) -> Self {
        Self { query_idx, train_idx, img_idx: -1, distance }
    }
}

/// Ordered by `distance`, which is what `cv::DMatch::operator<` compares.
impl PartialOrd for DMatch {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

/// `cv::Moments` — spatial, central and normalised central moments.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[allow(missing_docs)]
pub struct Moments {
    pub m00: f64,
    pub m10: f64,
    pub m01: f64,
    pub m20: f64,
    pub m11: f64,
    pub m02: f64,
    pub m30: f64,
    pub m21: f64,
    pub m12: f64,
    pub m03: f64,
    pub mu20: f64,
    pub mu11: f64,
    pub mu02: f64,
    pub mu30: f64,
    pub mu21: f64,
    pub mu12: f64,
    pub mu03: f64,
    pub nu20: f64,
    pub nu11: f64,
    pub nu02: f64,
    pub nu30: f64,
    pub nu21: f64,
    pub nu12: f64,
    pub nu03: f64,
}

impl Moments {
    /// `cv::Moments::Moments(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03)`.
    ///
    /// Takes the ten spatial moments and derives the central and normalised
    /// ones, which is what upstream's constructor does.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m00: f64,
        m10: f64,
        m01: f64,
        m20: f64,
        m11: f64,
        m02: f64,
        m30: f64,
        m21: f64,
        m12: f64,
        m03: f64,
    ) -> Self {
        let mut m =
            Moments { m00, m10, m01, m20, m11, m02, m30, m21, m12, m03, ..Moments::default() };
        m.complete();
        m
    }

    /// Centroid `(m10/m00, m01/m00)`, or `None` for a zero-mass shape.
    #[must_use]
    pub fn centroid(&self) -> Option<(f64, f64)> {
        if self.m00.abs() <= f64::EPSILON {
            return None;
        }
        Some((self.m10 / self.m00, self.m01 / self.m00))
    }

    /// Fill in the central and normalised moments from the spatial ones.
    ///
    /// A transcription of `completeMomentState`; the algebraic shortcuts (for
    /// example `mu21 = m21 - cx*(2*mu11 + cx*m01) - cy*mu20`) are upstream's.
    pub fn complete(&mut self) {
        let mut inv_m00 = 0.0;
        let (mut cx, mut cy) = (0.0, 0.0);
        if self.m00.abs() > f64::EPSILON {
            inv_m00 = 1.0 / self.m00;
            cx = self.m10 * inv_m00;
            cy = self.m01 * inv_m00;
        }

        let mu20 = self.m20 - self.m10 * cx;
        let mu11 = self.m11 - self.m10 * cy;
        let mu02 = self.m02 - self.m01 * cy;

        self.mu20 = mu20;
        self.mu11 = mu11;
        self.mu02 = mu02;
        self.mu30 = self.m30 - cx * (3.0 * mu20 + cx * self.m10);
        self.mu21 = self.m21 - cx * (2.0 * mu11 + cx * self.m01) - cy * mu20;
        self.mu12 = self.m12 - cy * (2.0 * mu11 + cy * self.m10) - cx * mu02;
        self.mu03 = self.m03 - cy * (3.0 * mu02 + cy * self.m01);

        // Normalisation divides mu_pq by m00^(1 + (p+q)/2).
        let inv_sqrt_m00 = inv_m00.abs().sqrt();
        let s2 = inv_m00 * inv_m00;
        let s3 = s2 * inv_sqrt_m00;
        self.nu20 = self.mu20 * s2;
        self.nu11 = self.mu11 * s2;
        self.nu02 = self.mu02 * s2;
        self.nu30 = self.mu30 * s3;
        self.nu21 = self.mu21 * s3;
        self.nu12 = self.mu12 * s3;
        self.nu03 = self.mu03 * s3;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn term_criteria_needs_a_condition_that_can_fire() {
        use term_criteria::{COUNT, EPS, MAX_ITER};
        assert_eq!(MAX_ITER, COUNT);
        assert!(!TermCriteria::default().is_valid());
        // Enabled but unsatisfiable: a zero iteration limit stops nothing.
        assert!(!TermCriteria::new(COUNT, 0, 0.0).is_valid());
        assert!(TermCriteria::new(COUNT, 30, 0.0).is_valid());
        assert!(TermCriteria::new(EPS, 0, 1e-6).is_valid());
        assert!(TermCriteria::new(COUNT | EPS, 30, 1e-6).is_valid());

        // A disabled half reads as the caller's fallback, not as zero.
        let c = TermCriteria::new(COUNT, 30, 0.0);
        assert_eq!(c.max_count_or(99), 30);
        assert_eq!(c.epsilon_or(1e-3), 1e-3);
    }

    #[test]
    fn rotated_rect_bounding_boxes_contain_every_corner() {
        let r = RotatedRect::new((10.0, 20.0), (8.0, 4.0), 30.0);
        let exact = r.bounding_rect2f();
        for (x, y) in r.points() {
            assert!(x as f32 >= exact.x - 1e-4 && x as f32 <= exact.x + exact.width + 1e-4);
            assert!(y as f32 >= exact.y - 1e-4 && y as f32 <= exact.y + exact.height + 1e-4);
        }
        // The integer box is the floor/ceil of that, widened by one each side.
        let b = r.bounding_rect();
        assert!(f64::from(b.x) <= f64::from(exact.x) - 1.0 + 1e-6);
        assert!(f64::from(b.x + b.width) >= f64::from(exact.x + exact.width) + 1.0 - 1e-6);
    }

    #[test]
    fn an_axis_aligned_rotated_rect_has_its_own_size() {
        let r = RotatedRect::new((5.0, 5.0), (4.0, 2.0), 0.0);
        let b = r.bounding_rect2f();
        assert_eq!((b.x, b.y, b.width, b.height), (3.0, 4.0, 4.0, 2.0));
        assert_eq!(r.area(), 8.0);
    }

    #[test]
    fn dmatch_orders_by_distance() {
        let mut v = [DMatch::new(0, 1, 3.0), DMatch::new(1, 2, 1.0), DMatch::new(2, 0, 2.0)];
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(v.map(|m| m.distance), [1.0, 2.0, 3.0]);
        assert_eq!(v[0].img_idx, -1);
    }

    /// Every value is `cv2.KeyPoint.overlap` on the same pair.
    #[test]
    fn keypoint_overlap_matches_cv2() {
        let kp = |x: f32, y: f32, s: f32| KeyPoint::new(x, y, s);
        let cases: &[(KeyPoint, KeyPoint, f32)] = &[
            (kp(0.0, 0.0, 10.0), kp(0.0, 0.0, 10.0), 1.0),
            (kp(0.0, 0.0, 10.0), kp(3.0, 0.0, 10.0), 0.453_316_8),
            // Disjoint discs share nothing.
            (kp(0.0, 0.0, 10.0), kp(20.0, 0.0, 10.0), 0.0),
            // Concentric: the ratio is of the two areas, 25/100.
            (kp(0.0, 0.0, 10.0), kp(0.0, 0.0, 20.0), 0.25),
            (kp(0.0, 0.0, 10.0), kp(2.0, 2.0, 14.0), 0.462_344_6),
            // Barely touching.
            (kp(0.0, 0.0, 10.0), kp(9.0, 0.0, 10.0), 0.019_049_13_f32),
        ];
        for (a, b, want) in cases {
            let got = KeyPoint::overlap(a, b);
            assert!((got - want).abs() < 1e-6, "{:?} vs {:?}: got {got}, cv2 {want}", a.pt, b.pt);
            // The measure is symmetric.
            assert!((KeyPoint::overlap(b, a) - got).abs() < 1e-6);
        }
    }

    #[test]
    fn keypoint_convert_round_trips() {
        let kps = [KeyPoint::new(1.0, 2.0, 3.0), KeyPoint::new(4.0, 5.0, 6.0),
                   KeyPoint::new(7.0, 8.0, 9.0)];
        // cv2.KeyPoint.convert(kps)
        assert_eq!(KeyPoint::convert(&kps, &[]).unwrap(), vec![(1.0, 2.0), (4.0, 5.0), (7.0, 8.0)]);
        // cv2.KeyPoint.convert(kps, [2, 0]) — the order given, not sorted.
        assert_eq!(KeyPoint::convert(&kps, &[2, 0]).unwrap(), vec![(7.0, 8.0), (1.0, 2.0)]);
        // A negative index is an error, as upstream makes it.
        assert!(KeyPoint::convert(&kps, &[-1]).is_err());
        assert!(KeyPoint::convert(&kps, &[9]).is_err());

        // cv2.KeyPoint.convert([(1,2),(3,4)], size=5, response=0.5, octave=2, class_id=7)
        let back = KeyPoint::convert_from(&[(1.0, 2.0), (3.0, 4.0)], 5.0, 0.5, 2, 7);
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].pt, (1.0, 2.0));
        assert_eq!((back[1].size, back[1].response, back[1].octave, back[1].class_id),
                   (5.0, 0.5, 2, 7));
    }

    #[test]
    fn keypoint_defaults_match_upstream() {
        let k = KeyPoint::new(1.0, 2.0, 7.0);
        assert_eq!(k.pt, (1.0, 2.0));
        assert_eq!(k.angle, -1.0);
        assert_eq!(k.class_id, -1);
        assert_eq!(k.octave, 0);
    }

    #[test]
    fn moments_of_a_unit_square_are_exact() {
        // A 1x1 mass at the origin: m00=1, m10=m01=0.
        let m = Moments::new(1.0, 0.0, 0.0, 1.0 / 12.0, 0.0, 1.0 / 12.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(m.centroid(), Some((0.0, 0.0)));
        assert!((m.mu20 - 1.0 / 12.0).abs() < 1e-15);
        assert!((m.nu20 - 1.0 / 12.0).abs() < 1e-15);
        assert_eq!(Moments::default().centroid(), None);
    }

    #[test]
    fn rect_intersection_normalises_empty() {
        let a = Rect::new(0, 0, 10, 10);
        let b = Rect::new(5, 5, 10, 10);
        assert_eq!(a.intersect(b), Rect::new(5, 5, 5, 5));

        let c = Rect::new(100, 100, 5, 5);
        assert_eq!(a.intersect(c), Rect::new(0, 0, 0, 0));
    }

    #[test]
    fn rect_union_spans_both() {
        let a = Rect::new(0, 0, 10, 10);
        let b = Rect::new(20, 5, 5, 5);
        assert_eq!(a.union(b), Rect::new(0, 0, 25, 10));
    }

    #[test]
    fn rect_contains_is_half_open() {
        let r = Rect::new(0, 0, 4, 4);
        assert!(r.contains(Point::new(0, 0)));
        assert!(r.contains(Point::new(3, 3)));
        assert!(!r.contains(Point::new(4, 3)));
        assert!(!r.contains(Point::new(-1, 0)));
    }

    #[test]
    fn size_area_does_not_overflow_i32() {
        let s = Size::new(100_000, 100_000);
        assert_eq!(s.area(), 10_000_000_000);
    }
}
