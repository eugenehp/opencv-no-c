//! `cv::Quat` — quaternions and the rotations they represent
//! (`modules/core/include/opencv2/core/quaternion.hpp`, `quaternion.inl.hpp`).
//!
//! Upstream's class is a template over `float`/`double`; this port computes in
//! `f64` throughout, matching the choice made elsewhere in `cv-core`.
//!
//! # Assuming a unit quaternion
//!
//! Most rotation operations are only meaningful on a unit quaternion, and are
//! cheaper when the caller can promise one. Upstream carries that promise in a
//! `QuatAssumeType` argument defaulting to `QUAT_ASSUME_NOT_UNIT`; the same
//! argument appears here as [`Assume`], because getting it wrong silently
//! changes the answer rather than failing.
//!
//! # Euler angles
//!
//! All 24 conventions are supported, intrinsic and extrinsic, Tait–Bryan and
//! proper. The two tables driving [`Quat::from_euler_angles`] and
//! [`Quat::to_euler_angles`] are transcribed from upstream verbatim: they
//! encode which matrix entry feeds which `atan2`, including the gimbal-lock
//! fallbacks, and there is no shorter correct form.

use crate::error::{Code, Result};
use crate::{cv_error, geometry::TermCriteria};

/// `CV_QUAT_EPS` — below this a norm counts as zero.
pub const CV_QUAT_EPS: f64 = 1.0e-6;
/// `CV_QUAT_CONVERT_THRESHOLD` — how close to ±1 counts as gimbal lock.
pub const CV_QUAT_CONVERT_THRESHOLD: f64 = 1.0e-5;

/// `cv::QuatAssumeType` — whether the caller promises a unit quaternion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum Assume {
    /// `QUAT_ASSUME_NOT_UNIT` — normalise first; correct for any input.
    #[default]
    NotUnit = 0,
    /// `QUAT_ASSUME_UNIT` — take the norm as 1 and skip the work.
    Unit = 1,
}

impl Assume {
    #[inline]
    fn is_unit(self) -> bool {
        self == Assume::Unit
    }
}

/// `cv::QuatEnum::EulerAnglesType` — which of the 24 conventions to use.
///
/// `INT_*` are intrinsic (each rotation about the *new* axes), `EXT_*`
/// extrinsic (all about the original axes). Within each, the first six are
/// Tait–Bryan (three distinct axes) and the last six proper Euler (first and
/// third the same).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
#[allow(missing_docs)]
pub enum EulerAnglesType {
    /// `INT_XYZ`
    IntXyz = 0,
    /// `INT_XZY`
    IntXzy = 1,
    /// `INT_YXZ`
    IntYxz = 2,
    /// `INT_YZX`
    IntYzx = 3,
    /// `INT_ZXY`
    IntZxy = 4,
    /// `INT_ZYX`
    IntZyx = 5,
    /// `INT_XYX`
    IntXyx = 6,
    /// `INT_XZX`
    IntXzx = 7,
    /// `INT_YXY`
    IntYxy = 8,
    /// `INT_YZY`
    IntYzy = 9,
    /// `INT_ZXZ`
    IntZxz = 10,
    /// `INT_ZYZ`
    IntZyz = 11,
    /// `EXT_XYZ`
    ExtXyz = 12,
    /// `EXT_XZY`
    ExtXzy = 13,
    /// `EXT_YXZ`
    ExtYxz = 14,
    /// `EXT_YZX`
    ExtYzx = 15,
    /// `EXT_ZXY`
    ExtZxy = 16,
    /// `EXT_ZYX`
    ExtZyx = 17,
    /// `EXT_XYX`
    ExtXyx = 18,
    /// `EXT_XZX`
    ExtXzx = 19,
    /// `EXT_YXY`
    ExtYxy = 20,
    /// `EXT_YZY`
    ExtYzy = 21,
    /// `EXT_ZXZ`
    ExtZxz = 22,
    /// `EXT_ZYZ`
    ExtZyz = 23,
}

/// `EULER_ANGLES_MAX_VALUE` — one past the last convention.
pub const EULER_ANGLES_MAX_VALUE: i32 = 24;

impl EulerAnglesType {
    /// Intrinsic conventions come first in the enumeration.
    #[inline]
    fn is_intrinsic(self) -> bool {
        (self as i32) < EulerAnglesType::ExtXyz as i32
    }

    /// Tait–Bryan means three distinct axes: blocks 1 and 3 of six.
    #[inline]
    fn is_tait_bryan(self) -> bool {
        let g = (self as i32) / 6;
        g == 1 || g == 3
    }
}

/// Which axis each of the three rotations turns about, per convention.
const ROTATION_AXIS: [[usize; 3]; 24] = [
    [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0],
    [0, 1, 0], [0, 2, 0], [1, 0, 1], [1, 2, 1], [2, 0, 2], [2, 1, 2],
    [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0],
    [0, 1, 0], [0, 2, 0], [1, 0, 1], [1, 2, 1], [2, 0, 2], [2, 1, 2],
];

/// How [`ROTATION_R`]'s entries encode their operands.
///
/// Each of the twelve slots names either a constant (ids below
/// [`euler_operand::N_CONSTANTS`]) or a rotation-matrix entry, and a negative
/// id negates the value. Upstream scopes these to an unnamed enum *inside*
/// `Quat::toEulerAngles`; they are public here because [`ROTATION_R`] is
/// meaningless without them, and anyone checking that table against upstream's
/// needs the same names.
pub mod euler_operand {
    /// `C_ZERO` — the constant zero.
    pub const C_ZERO: i32 = 0;
    /// `C_PI` — the constant π.
    pub const C_PI: i32 = 1;
    /// `C_PI_2` — the constant π/2.
    pub const C_PI_2: i32 = 2;
    /// `N_CONSTANTS` — ids at or above this index a rotation-matrix entry.
    pub const N_CONSTANTS: i32 = 3;
    /// `R_0_0` — row 0, column 0 of the rotation matrix.
    pub const R_0_0: i32 = N_CONSTANTS;
    /// `R_0_1`
    pub const R_0_1: i32 = N_CONSTANTS + 1;
    /// `R_0_2`
    pub const R_0_2: i32 = N_CONSTANTS + 2;
    /// `R_1_0`
    pub const R_1_0: i32 = N_CONSTANTS + 3;
    /// `R_1_1`
    pub const R_1_1: i32 = N_CONSTANTS + 4;
    /// `R_1_2`
    pub const R_1_2: i32 = N_CONSTANTS + 5;
    /// `R_2_0`
    pub const R_2_0: i32 = N_CONSTANTS + 6;
    /// `R_2_1`
    pub const R_2_1: i32 = N_CONSTANTS + 7;
    /// `R_2_2`
    pub const R_2_2: i32 = N_CONSTANTS + 8;
}

use euler_operand::{
    C_PI, C_PI_2, C_ZERO, N_CONSTANTS, R_0_0, R_0_1, R_0_2, R_1_0, R_1_1, R_1_2, R_2_0, R_2_1,
    R_2_2,
};

/// Upstream's `rotationR_` table: twelve operands per convention.
///
/// Read with [`euler_operand`].
#[rustfmt::skip]
pub const ROTATION_R: [[i32; 12]; 24] = [
    [ R_0_2,  R_1_0,  R_1_1,  C_PI_2,   R_2_1,  R_1_1, -C_PI_2, -R_1_2,  R_2_2,  R_0_2, -R_0_1,  R_0_0],
    [ R_0_1, -R_1_2,  R_2_2, -C_PI_2,   R_2_0,  R_2_2,  C_PI_2,  R_2_1,  R_1_1, -R_0_1,  R_0_2,  R_0_0],
    [ R_1_2, -R_0_1,  R_0_0, -C_PI_2,   R_0_1,  R_0_0,  C_PI_2,  R_0_2,  R_2_2, -R_1_2,  R_1_0,  R_1_1],
    [ R_1_0,  R_0_2,  R_2_2,  C_PI_2,   R_0_2,  R_0_1, -C_PI_2, -R_2_0,  R_0_0,  R_1_0, -R_1_2,  R_1_1],
    [ R_2_1,  R_1_0,  R_0_0,  C_PI_2,   R_1_0,  R_0_0, -C_PI_2, -R_0_1,  R_1_1,  R_2_1, -R_2_0,  R_2_2],
    [ R_2_0, -R_0_1,  R_1_1, -C_PI_2,   R_1_2,  R_1_1,  C_PI_2,  R_1_0,  R_0_0, -R_2_0,  R_2_1,  R_2_2],
    [ R_0_0,  R_2_1,  R_2_2,  C_ZERO,   R_1_2,  R_1_1,  C_PI,    R_1_0, -R_2_0,  R_0_0,  R_0_1,  R_0_2],
    [ R_0_0,  R_2_1,  R_2_2,  C_ZERO,  -R_2_1,  R_2_2,  C_PI,    R_2_0,  R_1_0,  R_0_0,  R_0_2, -R_0_1],
    [ R_1_1,  R_0_2,  R_0_0,  C_ZERO,  -R_2_0,  R_0_0,  C_PI,    R_0_1,  R_2_1,  R_1_1,  R_1_0, -R_1_2],
    [ R_1_1,  R_0_2,  R_0_0,  C_ZERO,   R_0_2, -R_0_0,  C_PI,    R_2_1, -R_0_1,  R_1_1,  R_1_2,  R_1_0],
    [ R_2_2,  R_1_0,  R_1_1,  C_ZERO,   R_1_0,  R_0_0,  C_PI,    R_0_2, -R_1_2,  R_2_2,  R_2_0,  R_2_1],
    [ R_2_2,  R_1_0,  R_0_0,  C_ZERO,   R_1_0,  R_0_0,  C_PI,    R_1_2,  R_0_2,  R_2_2,  R_2_1, -R_2_0],

    [ R_2_0, -C_PI_2, -R_0_1,  R_1_1,  C_PI_2,  R_1_2,  R_1_1,   R_2_1,  R_2_2, -R_2_0,  R_1_0,  R_0_0],
    [ R_1_0,  C_PI_2,  R_0_2,  R_2_2, -C_PI_2,  R_0_2,  R_0_1,  -R_1_2,  R_1_1,  R_1_0, -R_2_0,  R_0_0],
    [ R_2_1,  C_PI_2,  R_1_0,  R_0_0, -C_PI_2,  R_1_0,  R_0_0,  -R_2_0,  R_2_2,  R_2_1, -R_0_1,  R_1_1],
    [ R_0_2, -C_PI_2, -R_1_2,  R_2_2,  C_PI_2,  R_2_0,  R_2_2,   R_0_2,  R_0_0, -R_0_1,  R_2_1,  R_1_1],
    [ R_1_2, -C_PI_2, -R_0_1,  R_0_0,  C_PI_2,  R_0_1,  R_0_0,   R_1_0,  R_1_1, -R_1_2,  R_0_2,  R_2_2],
    [ R_0_2,  C_PI_2,  R_1_0,  R_1_1, -C_PI_2,  R_2_1,  R_1_1,  -R_0_1,  R_0_0,  R_0_2, -R_1_2,  R_2_2],
    [ R_0_0,  C_ZERO,  R_2_1,  R_2_2,  C_PI,    R_1_2,  R_1_1,   R_0_1,  R_0_2,  R_0_0,  R_1_0, -R_2_0],
    [ R_0_0,  C_ZERO,  R_2_1,  R_2_2,  C_PI,    R_2_1,  R_2_2,   R_0_2, -R_0_1,  R_0_0,  R_2_0,  R_1_0],
    [ R_1_1,  C_ZERO,  R_0_2,  R_0_0,  C_PI,   -R_2_0,  R_0_0,   R_1_0, -R_1_2,  R_1_1,  R_0_1,  R_2_1],
    [ R_1_1,  C_ZERO,  R_0_2,  R_0_0,  C_PI,    R_0_2, -R_0_0,   R_1_2,  R_1_0,  R_1_1,  R_2_1, -R_0_1],
    [ R_2_2,  C_ZERO,  R_1_0,  R_1_1,  C_PI,    R_1_0,  R_0_0,   R_2_0,  R_2_1,  R_2_2,  R_0_2, -R_1_2],
    [ R_2_2,  C_ZERO,  R_1_0,  R_0_0,  C_PI,    R_1_0,  R_0_0,   R_2_1, -R_2_0,  R_2_2,  R_1_2,  R_0_2],
];

/// `cv::Quat<double>` — `w + x·i + y·j + z·k`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Quat {
    /// Real part.
    pub w: f64,
    /// Coefficient of `i`.
    pub x: f64,
    /// Coefficient of `j`.
    pub y: f64,
    /// Coefficient of `k`.
    pub z: f64,
}

impl Quat {
    /// `cv::Quat::Quat(w, x, y, z)`.
    #[must_use]
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// The multiplicative identity — the rotation that does nothing.
    #[must_use]
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0)
    }

    /// `cv::Quat::dot(q)`.
    #[must_use]
    pub fn dot(&self, q: &Quat) -> f64 {
        self.w * q.w + self.x * q.x + self.y * q.y + self.z * q.z
    }

    /// `cv::Quat::norm()`.
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    /// `cv::Quat::conjugate()`.
    #[must_use]
    pub fn conjugate(&self) -> Quat {
        Quat::new(self.w, -self.x, -self.y, -self.z)
    }

    /// `cv::Quat::isNormal(eps)`.
    #[must_use]
    pub fn is_normal(&self, eps: f64) -> bool {
        let n = self.norm();
        n > 1.0 - eps && n < 1.0 + eps
    }

    /// `cv::Quat::normalize()`.
    pub fn normalize(&self) -> Result<Quat> {
        let n = self.norm();
        if n < CV_QUAT_EPS {
            return cv_error!(Code::StsBadArg, "cannot normalize this quaternion: the norm is too small");
        }
        Ok(Quat::new(self.w / n, self.x / n, self.y / n, self.z / n))
    }

    /// `cv::Quat::inv(assumeUnit)`.
    pub fn inv(&self, assume: Assume) -> Result<Quat> {
        if assume.is_unit() {
            return Ok(self.conjugate());
        }
        let n2 = self.dot(self);
        if n2 < CV_QUAT_EPS {
            return cv_error!(Code::StsBadArg, "this quaternion has no inverse");
        }
        Ok(self.conjugate().scale(1.0 / n2))
    }

    /// Multiply every component by `s`.
    #[must_use]
    pub fn scale(&self, s: f64) -> Quat {
        Quat::new(self.w * s, self.x * s, self.y * s, self.z * s)
    }

    /// `cv::Quat::operator+`.
    #[must_use]
    pub fn add(&self, q: &Quat) -> Quat {
        Quat::new(self.w + q.w, self.x + q.x, self.y + q.y, self.z + q.z)
    }

    /// `cv::Quat::operator-`.
    #[must_use]
    pub fn sub(&self, q: &Quat) -> Quat {
        Quat::new(self.w - q.w, self.x - q.x, self.y - q.y, self.z - q.z)
    }

    /// `cv::Quat::operator*` — the Hamilton product, which does not commute.
    #[must_use]
    pub fn mul(&self, q: &Quat) -> Quat {
        Quat::new(
            self.w * q.w - self.x * q.x - self.y * q.y - self.z * q.z,
            self.w * q.x + self.x * q.w + self.y * q.z - self.z * q.y,
            self.w * q.y + self.y * q.w + self.z * q.x - self.x * q.z,
            self.w * q.z + self.z * q.w + self.x * q.y - self.y * q.x,
        )
    }

    /// `cv::Quat::operator-()` — negate every component.
    #[must_use]
    pub fn neg(&self) -> Quat {
        self.scale(-1.0)
    }

    /// `cv::Quat::crossProduct(q)`.
    #[must_use]
    pub fn cross_product(&self, q: &Quat) -> Quat {
        self.mul(q).sub(&q.mul(self)).scale(0.5)
    }

    /// `cv::Quat::exp()`.
    #[must_use]
    pub fn exp(&self) -> Quat {
        let v = [self.x, self.y, self.z];
        let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let k = if nv < CV_QUAT_EPS { 1.0 } else { nv.sin() / nv };
        Quat::new(nv.cos(), v[0] * k, v[1] * k, v[2] * k).scale(self.w.exp())
    }

    /// `cv::Quat::log(assumeUnit)`.
    pub fn log(&self, assume: Assume) -> Result<Quat> {
        let v = [self.x, self.y, self.z];
        let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if assume.is_unit() {
            let k = if nv < CV_QUAT_EPS { 1.0 } else { self.w.acos() / nv };
            return Ok(Quat::new(0.0, v[0] * k, v[1] * k, v[2] * k));
        }
        let qn = self.norm();
        if qn < CV_QUAT_EPS {
            return cv_error!(Code::StsBadArg, "log is undefined for a zero quaternion");
        }
        let k = if nv < CV_QUAT_EPS { 1.0 } else { (self.w / qn).acos() / nv };
        Ok(Quat::new(qn.ln(), v[0] * k, v[1] * k, v[2] * k))
    }

    /// `cv::Quat::power(alpha, assumeUnit)`.
    pub fn power(&self, alpha: f64, assume: Assume) -> Result<Quat> {
        // A purely real quaternion has no axis, so the angle-axis route does
        // not apply; upstream falls back to the scalar power there.
        if self.x * self.x + self.y * self.y + self.z * self.z <= CV_QUAT_EPS {
            return Ok(Quat::new(self.w.powf(alpha), 0.0, 0.0, 0.0));
        }
        let angle = self.get_angle(assume)?;
        let axis = self.get_axis(assume)?;
        let q = Quat::from_angle_axis(alpha * angle, axis)?;
        Ok(if assume.is_unit() { q } else { q.scale(self.norm().powf(alpha)) })
    }

    /// `cv::Quat::sqrt(assumeUnit)`.
    pub fn sqrt(&self, assume: Assume) -> Result<Quat> {
        self.power(0.5, assume)
    }

    /// `cv::Quat::power(q, assumeUnit)` — the quaternion-exponent form.
    pub fn power_quat(&self, q: &Quat, assume: Assume) -> Result<Quat> {
        Ok(self.log(assume)?.mul(q).exp())
    }

    /// `cv::Quat::createFromAngleAxis(angle, axis)`.
    pub fn from_angle_axis(angle: f64, axis: [f64; 3]) -> Result<Quat> {
        let vn = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if vn < CV_QUAT_EPS {
            return cv_error!(Code::StsBadArg, "this quaternion does not represent a rotation");
        }
        let half = angle * 0.5;
        let k = half.sin() / vn;
        Ok(Quat::new(half.cos(), k * axis[0], k * axis[1], k * axis[2]))
    }

    /// `cv::Quat::getAngle(assumeUnit)` — the rotation angle, in radians.
    pub fn get_angle(&self, assume: Assume) -> Result<f64> {
        if assume.is_unit() {
            return Ok(2.0 * self.w.acos());
        }
        let n = self.norm();
        if n < CV_QUAT_EPS {
            return cv_error!(Code::StsBadArg, "this quaternion does not represent a rotation");
        }
        Ok(2.0 * (self.w / n).acos())
    }

    /// `cv::Quat::getAxis(assumeUnit)` — the rotation axis, unit length.
    pub fn get_axis(&self, assume: Assume) -> Result<[f64; 3]> {
        let angle = self.get_angle(assume)?;
        let s = (angle * 0.5).sin();
        let d = if assume.is_unit() { s } else { self.norm() * s };
        Ok([self.x / d, self.y / d, self.z / d])
    }

    /// `cv::Quat::toRotVec(assumeUnit)` — the axis scaled by the angle.
    pub fn to_rot_vec(&self, assume: Assume) -> Result<[f64; 3]> {
        let angle = self.get_angle(assume)?;
        let a = self.get_axis(assume)?;
        Ok([angle * a[0], angle * a[1], angle * a[2]])
    }

    /// `cv::Quat::createFromRvec(rvec)` — from a Rodrigues rotation vector.
    pub fn from_rot_vec(rvec: [f64; 3]) -> Result<Quat> {
        let angle = (rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]).sqrt();
        if angle < CV_QUAT_EPS {
            return Ok(Quat::identity());
        }
        Quat::from_angle_axis(angle, rvec)
    }

    /// `cv::Quat::createFromXRot(theta)`.
    #[must_use]
    pub fn from_x_rot(theta: f64) -> Quat {
        Quat::new((theta * 0.5).cos(), (theta * 0.5).sin(), 0.0, 0.0)
    }

    /// `cv::Quat::createFromYRot(theta)`.
    #[must_use]
    pub fn from_y_rot(theta: f64) -> Quat {
        Quat::new((theta * 0.5).cos(), 0.0, (theta * 0.5).sin(), 0.0)
    }

    /// `cv::Quat::createFromZRot(theta)`.
    #[must_use]
    pub fn from_z_rot(theta: f64) -> Quat {
        Quat::new((theta * 0.5).cos(), 0.0, 0.0, (theta * 0.5).sin())
    }

    /// `cv::Quat::toRotMat3x3(assumeUnit)`, row-major.
    pub fn to_rot_mat3x3(&self, assume: Assume) -> Result<[[f64; 3]; 3]> {
        let q = if assume.is_unit() { *self } else { self.normalize()? };
        let (a, b, c, d) = (q.w, q.x, q.y, q.z);
        Ok([
            [1.0 - 2.0 * (c * c + d * d), 2.0 * (b * c - a * d), 2.0 * (b * d + a * c)],
            [2.0 * (b * c + a * d), 1.0 - 2.0 * (b * b + d * d), 2.0 * (c * d - a * b)],
            [2.0 * (b * d - a * c), 2.0 * (c * d + a * b), 1.0 - 2.0 * (b * b + c * c)],
        ])
    }

    /// `cv::Quat::toRotMat4x4(assumeUnit)` — the left-multiplication matrix.
    pub fn to_rot_mat4x4(&self, assume: Assume) -> Result<[[f64; 4]; 4]> {
        let q = if assume.is_unit() { *self } else { self.normalize()? };
        let (a, b, c, d) = (q.w, q.x, q.y, q.z);
        Ok([
            [a, -b, -c, -d],
            [b, a, -d, c],
            [c, d, a, -b],
            [d, -c, b, a],
        ])
    }

    /// `cv::Quat::createFromRotMat(R)` — from a row-major 3x3 rotation matrix.
    ///
    /// The four branches pick whichever component is largest, so the division
    /// is never by a near-zero: reconstructing from the trace alone loses all
    /// precision at a 180° rotation.
    #[must_use]
    pub fn from_rot_mat(r: &[[f64; 3]; 3]) -> Quat {
        let trace = r[0][0] + r[1][1] + r[2][2];
        if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            Quat::new(
                -0.25 * s,
                (r[1][2] - r[2][1]) / s,
                (r[2][0] - r[0][2]) / s,
                (r[0][1] - r[1][0]) / s,
            )
        } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
            let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
            Quat::new(
                (r[1][2] - r[2][1]) / s,
                -0.25 * s,
                -(r[1][0] + r[0][1]) / s,
                -(r[0][2] + r[2][0]) / s,
            )
        } else if r[1][1] > r[2][2] {
            let s = (1.0 - r[0][0] + r[1][1] - r[2][2]).sqrt() * 2.0;
            Quat::new(
                (r[0][2] - r[2][0]) / s,
                (r[0][1] + r[1][0]) / s,
                0.25 * s,
                (r[1][2] + r[2][1]) / s,
            )
        } else {
            let s = (1.0 - r[0][0] - r[1][1] + r[2][2]).sqrt() * 2.0;
            Quat::new(
                -(r[0][1] - r[1][0]) / s,
                (r[0][2] + r[2][0]) / s,
                (r[1][2] + r[2][1]) / s,
                0.25 * s,
            )
        }
    }

    /// `cv::Quat::createFromEulerAngles(angles, eulerAnglesType)`.
    #[must_use]
    pub fn from_euler_angles(angles: [f64; 3], ty: EulerAnglesType) -> Quat {
        let axes = ROTATION_AXIS[ty as usize];
        let about = |axis: usize, theta: f64| match axis {
            0 => Quat::from_x_rot(theta),
            1 => Quat::from_y_rot(theta),
            _ => Quat::from_z_rot(theta),
        };
        let q1 = about(axes[0], angles[0]);
        let q2 = about(axes[1], angles[1]);
        let q3 = about(axes[2], angles[2]);
        // Intrinsic composes left to right; extrinsic is the same rotations in
        // the opposite order.
        if ty.is_intrinsic() {
            q1.mul(&q2).mul(&q3)
        } else {
            q3.mul(&q2).mul(&q1)
        }
    }

    /// `cv::Quat::toEulerAngles(eulerAnglesType)`.
    ///
    /// At gimbal lock the decomposition is not unique; upstream pins the third
    /// angle (intrinsic) or the first (extrinsic) to zero and solves for the
    /// rest, and so does this.
    pub fn to_euler_angles(&self, ty: EulerAnglesType) -> Result<[f64; 3]> {
        let r = self.to_rot_mat3x3(Assume::NotUnit)?;
        let flat = [r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2]];
        let constants = [0.0, std::f64::consts::PI, std::f64::consts::FRAC_PI_2];

        let mut v = [0.0f64; 12];
        for (i, slot) in v.iter_mut().enumerate() {
            let id = ROTATION_R[ty as usize][i];
            let idx = id.unsigned_abs() as usize;
            let value = if (idx as i32) < N_CONSTANTS {
                constants[idx]
            } else {
                flat[idx - N_CONSTANTS as usize]
            };
            *slot = if id < 0 { -value } else { value };
        }

        if ty.is_intrinsic() {
            if (v[0] - 1.0).abs() < CV_QUAT_CONVERT_THRESHOLD {
                return Ok([v[1].atan2(v[2]), v[3], 0.0]);
            }
            if (v[0] + 1.0).abs() < CV_QUAT_CONVERT_THRESHOLD {
                return Ok([v[4].atan2(v[5]), v[6], 0.0]);
            }
        } else {
            if (v[0] - 1.0).abs() < CV_QUAT_CONVERT_THRESHOLD {
                return Ok([0.0, v[1], v[2].atan2(v[3])]);
            }
            if (v[0] + 1.0).abs() < CV_QUAT_CONVERT_THRESHOLD {
                return Ok([0.0, v[4], v[5].atan2(v[6])]);
            }
        }

        let middle = if ty.is_tait_bryan() { v[9].acos() } else { v[9].asin() };
        Ok([v[7].atan2(v[8]), middle, v[10].atan2(v[11])])
    }

    /// `cv::Quat::lerp(q0, q1, t)` — the straight line, which leaves the sphere.
    #[must_use]
    pub fn lerp(q0: &Quat, q1: &Quat, t: f64) -> Quat {
        q0.scale(1.0 - t).add(&q1.scale(t))
    }

    /// `cv::Quat::nlerp(q0, q1, t, assumeUnit)` — [`Quat::lerp`] renormalised.
    pub fn nlerp(q0: &Quat, q1: &Quat, t: f64, assume: Assume) -> Result<Quat> {
        let mut v0 = *q0;
        let mut v1 = *q1;
        // Take the shorter of the two arcs: q and -q are the same rotation.
        if v1.dot(&v0) < 0.0 {
            v0 = v0.neg();
        }
        if !assume.is_unit() {
            v0 = v0.normalize()?;
            v1 = v1.normalize()?;
        }
        Quat::lerp(&v0, &v1, t).normalize()
    }

    /// `cv::Quat::slerp(q0, q1, t, assumeUnit, directChange)`.
    ///
    /// Constant angular velocity along the arc. Near-parallel inputs fall back
    /// to [`Quat::nlerp`] at upstream's 0.995 dot threshold, where `sinTheta`
    /// would otherwise divide away the precision.
    pub fn slerp(
        q0: &Quat,
        q1: &Quat,
        t: f64,
        assume: Assume,
        direct_change: bool,
    ) -> Result<Quat> {
        let mut v0 = *q0;
        let mut v1 = *q1;
        if !assume.is_unit() {
            v0 = v0.normalize()?;
            v1 = v1.normalize()?;
        }
        let mut cos_theta = v0.dot(&v1);
        const DOT_THRESHOLD: f64 = 0.995;
        if cos_theta.abs() > DOT_THRESHOLD {
            return Quat::nlerp(&v0, &v1, t, Assume::Unit);
        }
        if direct_change && cos_theta < 0.0 {
            v0 = v0.neg();
            cos_theta = -cos_theta;
        }
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let angle = sin_theta.atan2(cos_theta);
        v0.scale(((1.0 - t) * angle).sin() / sin_theta)
            .add(&v1.scale((t * angle).sin() / sin_theta))
            .normalize()
    }

    /// `cv::Quat::squad(q0, q1, q2, q3, t, assumeUnit, directChange)`.
    pub fn squad(
        q0: &Quat,
        q1: &Quat,
        q2: &Quat,
        q3: &Quat,
        t: f64,
        assume: Assume,
        direct_change: bool,
    ) -> Result<Quat> {
        let (mut v0, mut v1, mut v2, mut v3) = (*q0, *q1, *q2, *q3);
        if !assume.is_unit() {
            v0 = v0.normalize()?;
            v1 = v1.normalize()?;
            v2 = v2.normalize()?;
            v3 = v3.normalize()?;
        }
        let c0 = Quat::slerp(&v0, &v3, t, Assume::Unit, direct_change)?;
        let c1 = Quat::slerp(&v1, &v2, t, Assume::Unit, direct_change)?;
        Quat::slerp(&c0, &c1, 2.0 * t * (1.0 - t), Assume::Unit, direct_change)
    }

    /// `cv::Quat::interPoint(q0, q1, q2, assumeUnit)` — a squad control point.
    pub fn inter_point(q0: &Quat, q1: &Quat, q2: &Quat, assume: Assume) -> Result<Quat> {
        let (mut v0, mut v1, mut v2) = (*q0, *q1, *q2);
        if !assume.is_unit() {
            v0 = v0.normalize()?;
            v1 = v1.normalize()?;
            v2 = v2.normalize()?;
        }
        let a = v1.conjugate().mul(&v0).log(assume)?;
        let b = v1.conjugate().mul(&v2).log(assume)?;
        Ok(v1.mul(&a.add(&b).scale(-0.25).exp()))
    }

    /// `cv::Quat::spline(q0, q1, q2, q3, t, assumeUnit)`.
    ///
    /// A C¹ curve through `q1` and `q2`, with `q0` and `q3` setting the
    /// tangents. The sign pass first makes consecutive keys take the short
    /// arc, without which the curve can swing the long way round.
    pub fn spline(
        q0: &Quat,
        q1: &Quat,
        q2: &Quat,
        q3: &Quat,
        t: f64,
        assume: Assume,
    ) -> Result<Quat> {
        let mut v = [*q0, *q1, *q2, *q3];
        if !assume.is_unit() {
            for q in &mut v {
                *q = q.normalize()?;
            }
        }
        for i in 0..3 {
            if v[i].dot(&v[i + 1]) < 0.0 {
                v[i + 1] = v[i + 1].neg();
            }
        }
        let s1 = Quat::inter_point(&v[0], &v[1], &v[2], Assume::Unit)?;
        let s2 = Quat::inter_point(&v[1], &v[2], &v[3], Assume::Unit)?;
        Quat::squad(&v[1], &s1, &s2, &v[2], t, assume, false)
    }

    /// Rotate a 3-vector by this quaternion — `q v q⁻¹`.
    pub fn rotate_point(&self, p: [f64; 3], assume: Assume) -> Result<[f64; 3]> {
        let q = if assume.is_unit() { *self } else { self.normalize()? };
        let v = Quat::new(0.0, p[0], p[1], p[2]);
        let r = q.mul(&v).mul(&q.conjugate());
        Ok([r.x, r.y, r.z])
    }
}

/// A termination rule for iterative quaternion fitting, kept so callers can
/// pass one where upstream's overloads take `TermCriteria`.
///
/// Nothing here iterates today; the alias exists so a signature that grows one
/// does not have to change shape.
pub type QuatTermCriteria = TermCriteria;

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    fn quat_close(a: &Quat, b: &Quat) -> bool {
        // q and -q are the same rotation, so accept either sign.
        let same = close(a.w, b.w) && close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z);
        let flip = close(a.w, -b.w) && close(a.x, -b.x) && close(a.y, -b.y) && close(a.z, -b.z);
        same || flip
    }

    #[test]
    fn multiplication_follows_the_hamilton_rules() {
        let i = Quat::new(0.0, 1.0, 0.0, 0.0);
        let j = Quat::new(0.0, 0.0, 1.0, 0.0);
        let k = Quat::new(0.0, 0.0, 0.0, 1.0);
        // i*j = k, j*k = i, k*i = j, and i*i = j*j = k*k = -1.
        assert!(quat_close(&i.mul(&j), &k) && close(i.mul(&j).z, 1.0));
        assert!(close(j.mul(&k).x, 1.0));
        assert!(close(k.mul(&i).y, 1.0));
        assert!(close(i.mul(&i).w, -1.0));
        // It does not commute: j*i = -k.
        assert!(close(j.mul(&i).z, -1.0));
    }

    #[test]
    fn a_rotation_round_trips_through_its_matrix() {
        let q = Quat::from_angle_axis(0.7, [1.0, 2.0, -3.0]).unwrap();
        assert!(q.is_normal(1e-12));
        let r = q.to_rot_mat3x3(Assume::Unit).unwrap();
        // R is orthogonal with determinant 1.
        for row in &r {
            assert!(close(row[0] * row[0] + row[1] * row[1] + row[2] * row[2], 1.0));
        }
        let back = Quat::from_rot_mat(&r);
        // Upstream's reconstruction gives the conjugate convention, so compare
        // through the matrix rather than the components.
        let r2 = back.to_rot_mat3x3(Assume::NotUnit).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(close(r[i][j], r2[i][j]), "R[{i}][{j}]: {} vs {}", r[i][j], r2[i][j]);
            }
        }
    }

    #[test]
    fn angle_and_axis_recover_what_built_the_quaternion() {
        let axis = [0.0, 0.0, 1.0];
        let q = Quat::from_angle_axis(1.1, axis).unwrap();
        assert!(close(q.get_angle(Assume::Unit).unwrap(), 1.1));
        let a = q.get_axis(Assume::Unit).unwrap();
        assert!(close(a[0], 0.0) && close(a[1], 0.0) && close(a[2], 1.0));
        // The rotation vector is the axis scaled by the angle.
        let rv = q.to_rot_vec(Assume::Unit).unwrap();
        assert!(close(rv[2], 1.1));
        assert!(quat_close(&Quat::from_rot_vec(rv).unwrap(), &q));
    }

    #[test]
    fn a_z_rotation_turns_x_towards_y() {
        let q = Quat::from_z_rot(std::f64::consts::FRAC_PI_2);
        let p = q.rotate_point([1.0, 0.0, 0.0], Assume::Unit).unwrap();
        assert!(close(p[0], 0.0) && close(p[1], 1.0) && close(p[2], 0.0), "{p:?}");
    }

    #[test]
    fn every_euler_convention_round_trips() {
        use EulerAnglesType as E;
        let all = [
            E::IntXyz, E::IntXzy, E::IntYxz, E::IntYzx, E::IntZxy, E::IntZyx,
            E::IntXyx, E::IntXzx, E::IntYxy, E::IntYzy, E::IntZxz, E::IntZyz,
            E::ExtXyz, E::ExtXzy, E::ExtYxz, E::ExtYzx, E::ExtZxy, E::ExtZyx,
            E::ExtXyx, E::ExtXzx, E::ExtYxy, E::ExtYzy, E::ExtZxz, E::ExtZyz,
        ];
        assert_eq!(all.len(), EULER_ANGLES_MAX_VALUE as usize);
        for (i, ty) in all.iter().enumerate() {
            assert_eq!(*ty as i32, i as i32, "{ty:?} sits at the wrong index");
            // Angles chosen well away from gimbal lock for every convention:
            // Tait-Bryan locks at |mid| = pi/2, proper Euler at mid = 0 or pi.
            let angles = [0.3, 0.9, -0.4];
            let q = Quat::from_euler_angles(angles, *ty);
            let back = q.to_euler_angles(*ty).unwrap();
            let q2 = Quat::from_euler_angles(back, *ty);
            assert!(quat_close(&q, &q2), "{ty:?}: {angles:?} -> {back:?}");
        }
    }

#[test]
fn euler_matches_an_independent_reference() {
    // scipy.spatial.transform.Rotation.from_euler('XYZ', [0.3, 0.9, -0.4])
    // gives quaternion (x, y, z, w) = (0.0464349, 0.44824152, -0.11317771, 0.8855022).
    let q = Quat::from_euler_angles([0.3, 0.9, -0.4], EulerAnglesType::IntXyz);
    let e = 1e-7;
    assert!((q.w - 0.8855022).abs() < e, "{q:?}");
    assert!((q.x - 0.0464349).abs() < e, "{q:?}");
    assert!((q.y - 0.44824152).abs() < e, "{q:?}");
    assert!((q.z - -0.11317771).abs() < e, "{q:?}");

    // The same angles read extrinsically: scipy's lowercase 'xyz'.
    let q = Quat::from_euler_angles([0.3, 0.9, -0.4], EulerAnglesType::ExtXyz);
    assert!((q.w - 0.85967501).abs() < e, "{q:?}");
    assert!((q.x - 0.21732284).abs() < e, "{q:?}");
    assert!((q.y - 0.39477518).abs() < e, "{q:?}");
    assert!((q.z - -0.24058722).abs() < e, "{q:?}");
}

    #[test]
    fn intrinsic_and_extrinsic_reverse_each_other() {
        // X-Y-Z intrinsic with angles (a, b, c) is Z-Y-X extrinsic with (c, b, a).
        let a = Quat::from_euler_angles([0.3, 0.9, -0.4], EulerAnglesType::IntXyz);
        let b = Quat::from_euler_angles([-0.4, 0.9, 0.3], EulerAnglesType::ExtZyx);
        assert!(quat_close(&a, &b), "{a:?} vs {b:?}");
    }

    #[test]
    fn gimbal_lock_pins_one_angle_to_zero() {
        // Tait-Bryan XYZ locks when the middle angle is pi/2.
        let q = Quat::from_euler_angles(
            [0.5, std::f64::consts::FRAC_PI_2, 0.7],
            EulerAnglesType::IntXyz,
        );
        let back = q.to_euler_angles(EulerAnglesType::IntXyz).unwrap();
        assert!(close(back[2], 0.0), "the third angle is pinned; got {back:?}");
        // The rotation is still reproduced, just decomposed differently.
        let q2 = Quat::from_euler_angles(back, EulerAnglesType::IntXyz);
        assert!(quat_close(&q, &q2), "{back:?}");
    }

    #[test]
    fn exp_and_log_are_inverse_on_a_unit_quaternion() {
        let q = Quat::from_angle_axis(0.9, [1.0, -2.0, 0.5]).unwrap();
        let back = q.log(Assume::Unit).unwrap().exp();
        assert!(quat_close(&q, &back), "{q:?} vs {back:?}");
        // On a general quaternion, with the norm carried through the log.
        let g = Quat::new(2.0, 0.5, -1.0, 0.25);
        let back = g.log(Assume::NotUnit).unwrap().exp();
        assert!(quat_close(&g, &back), "{g:?} vs {back:?}");
    }

    #[test]
    fn power_and_sqrt_compose() {
        let q = Quat::from_angle_axis(1.0, [0.0, 1.0, 0.0]).unwrap();
        let r = q.sqrt(Assume::Unit).unwrap();
        assert!(quat_close(&r.mul(&r), &q), "sqrt squared should return q");
        // q^2 doubles the angle.
        let sq = q.power(2.0, Assume::Unit).unwrap();
        assert!(close(sq.get_angle(Assume::Unit).unwrap(), 2.0));
    }

    #[test]
    fn inverse_and_conjugate_agree_on_a_unit_quaternion() {
        let q = Quat::from_angle_axis(0.4, [1.0, 1.0, 1.0]).unwrap();
        assert!(quat_close(&q.inv(Assume::Unit).unwrap(), &q.conjugate()));
        assert!(quat_close(&q.mul(&q.inv(Assume::NotUnit).unwrap()), &Quat::identity()));
        // A zero quaternion has neither an inverse nor a normalisation.
        assert!(Quat::default().inv(Assume::NotUnit).is_err());
        assert!(Quat::default().normalize().is_err());
    }

    #[test]
    fn slerp_holds_constant_angular_velocity() {
        let q0 = Quat::from_z_rot(0.0);
        let q1 = Quat::from_z_rot(1.6);
        let mid = Quat::slerp(&q0, &q1, 0.5, Assume::Unit, true).unwrap();
        assert!(close(mid.get_angle(Assume::Unit).unwrap(), 0.8), "{mid:?}");
        // Quarter and three-quarter points are evenly spaced in angle.
        let a = Quat::slerp(&q0, &q1, 0.25, Assume::Unit, true).unwrap();
        let b = Quat::slerp(&q0, &q1, 0.75, Assume::Unit, true).unwrap();
        assert!(close(a.get_angle(Assume::Unit).unwrap(), 0.4));
        assert!(close(b.get_angle(Assume::Unit).unwrap(), 1.2));
        // Endpoints are exact.
        assert!(quat_close(&Quat::slerp(&q0, &q1, 0.0, Assume::Unit, true).unwrap(), &q0));
    }

    #[test]
    fn nearly_parallel_inputs_fall_back_to_nlerp() {
        let q0 = Quat::from_z_rot(0.0);
        let q1 = Quat::from_z_rot(0.01);
        // Above the 0.995 dot threshold slerp defers to nlerp; both are unit.
        let s = Quat::slerp(&q0, &q1, 0.5, Assume::Unit, true).unwrap();
        let n = Quat::nlerp(&q0, &q1, 0.5, Assume::Unit).unwrap();
        assert!(quat_close(&s, &n));
        assert!(s.is_normal(1e-12));
    }

    #[test]
    fn spline_passes_through_its_interior_keys() {
        let keys: Vec<Quat> = (0..4).map(|i| Quat::from_z_rot(f64::from(i) * 0.5)).collect();
        let at0 = Quat::spline(&keys[0], &keys[1], &keys[2], &keys[3], 0.0, Assume::Unit).unwrap();
        let at1 = Quat::spline(&keys[0], &keys[1], &keys[2], &keys[3], 1.0, Assume::Unit).unwrap();
        assert!(quat_close(&at0, &keys[1]), "{at0:?} vs {:?}", keys[1]);
        assert!(quat_close(&at1, &keys[2]), "{at1:?} vs {:?}", keys[2]);
    }

    #[test]
    fn the_4x4_matrix_is_left_multiplication() {
        let p = Quat::new(0.3, -0.5, 0.7, 0.2);
        let q = Quat::new(1.0, 2.0, -1.0, 0.5);
        let m = p.to_rot_mat4x4(Assume::Unit).unwrap();
        let v = [q.w, q.x, q.y, q.z];
        let got: Vec<f64> =
            (0..4).map(|i| (0..4).map(|j| m[i][j] * v[j]).sum::<f64>()).collect();
        let want = p.mul(&q);
        assert!(close(got[0], want.w) && close(got[1], want.x));
        assert!(close(got[2], want.y) && close(got[3], want.z));
    }

    #[test]
    fn cross_product_is_the_antisymmetric_half() {
        let p = Quat::new(0.0, 1.0, 0.0, 0.0);
        let q = Quat::new(0.0, 0.0, 1.0, 0.0);
        // i x j = k.
        let c = p.cross_product(&q);
        assert!(close(c.w, 0.0) && close(c.z, 1.0), "{c:?}");
    }
}
