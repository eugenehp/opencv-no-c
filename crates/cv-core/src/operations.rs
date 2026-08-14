//! Compound-assignment operators on [`Mat`]
//! (`modules/core/include/opencv2/core/operations.hpp`).
//!
//! Upstream generates `+=`, `-=`, `*=`, `/=`, `&=`, `|=` and `^=` for every
//! combination of `Mat`, `Mat_<T>`, `Scalar` and `double` through the
//! `CV_MAT_AUG_OPERATOR` macro family. Rust reaches the same surface through
//! the `std::ops::*Assign` traits, one impl per right-hand type.
//!
//! The four macros — `CV_MAT_AUG_OPERATOR`, `CV_MAT_AUG_OPERATOR1`,
//! `CV_MAT_AUG_OPERATOR_T` and `CV_MAT_AUG_OPERATOR_TN` — exist only to spell
//! out that cross product without repeating it; the trait impls below *are*
//! their expansion. `_T` is the `Mat_<T>` overload and `_TN` the `Matx`
//! one, neither of which this port has a separate type for: a typed matrix is
//! a [`Mat`] with a known [`crate::Depth`].
//!
//! # The panics
//!
//! Every operation these wrap can fail — mismatched sizes, an unsupported
//! depth — and the rest of this crate returns [`crate::Result`] for exactly
//! that reason. An operator cannot: `a += b` has nowhere to put an error.
//!
//! So these **panic** on failure, and the named functions
//! ([`crate::add`], [`crate::subtract`], …) remain the checked form. That is a
//! deliberate split, not an oversight: an operator is for the case where the
//! shapes are known to match, and silently producing a wrong-sized result
//! would be worse than a panic. Upstream throws here, which has the same
//! effect on a caller that was not going to handle it.
//!
//! `*=` follows upstream's asymmetry: against another matrix it is **matrix
//! multiplication**, not element-wise, while against a scalar it scales. That
//! surprises people, so it is worth stating rather than discovering.

use std::ops::{AddAssign, BitAndAssign, BitOrAssign, BitXorAssign, DivAssign, MulAssign, SubAssign};

use crate::geometry::Scalar;
use crate::mat::{Mat, MatLike};

/// Unwrap an operator's result, naming the operator in the panic.
fn expect<T>(what: &str, r: crate::Result<T>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("Mat {what}: {e}"),
    }
}

/// The expansion of `CV_MAT_AUG_OPERATOR(+=, cv::add(...), Mat, Mat)`.
impl AddAssign<&Mat> for Mat {
    /// `Mat::operator+=(Mat)` — element-wise, saturating.
    fn add_assign(&mut self, rhs: &Mat) {
        *self = expect("+=", crate::arithm::add(self, rhs));
    }
}

/// The expansion of `CV_MAT_AUG_OPERATOR1(+=, cv::add(...), Mat, Scalar)`.
///
/// `CV_MAT_AUG_OPERATOR` is `CV_MAT_AUG_OPERATOR1` applied twice, once for the
/// mutable receiver and once for the `const` one; Rust needs only the mutable
/// form, since `&mut self` is what `AddAssign` takes.
impl AddAssign<Scalar> for Mat {
    /// `Mat::operator+=(Scalar)`.
    fn add_assign(&mut self, rhs: Scalar) {
        *self = expect("+= Scalar", crate::arithm::add_scalar(self, rhs));
    }
}

impl SubAssign<&Mat> for Mat {
    /// `Mat::operator-=(Mat)` — element-wise, saturating.
    fn sub_assign(&mut self, rhs: &Mat) {
        *self = expect("-=", crate::arithm::subtract(self, rhs));
    }
}

impl SubAssign<Scalar> for Mat {
    /// `Mat::operator-=(Scalar)`.
    fn sub_assign(&mut self, rhs: Scalar) {
        *self = expect("-= Scalar", crate::arithm::subtract_scalar(self, rhs));
    }
}

/// The expansion of `CV_MAT_AUG_OPERATOR_T(*=, cv::gemm(...), Mat_<_Tp>, Mat)`.
///
/// `_T` is upstream's typed-matrix overload. A typed matrix here is a [`Mat`]
/// with a known [`crate::Depth`], so the one impl covers both.
impl MulAssign<&Mat> for Mat {
    /// `Mat::operator*=(Mat)` — **matrix** multiplication, as upstream's is.
    ///
    /// For the element-wise product use [`crate::multiply`], which is what
    /// `Mat::mul` does in C++.
    fn mul_assign(&mut self, rhs: &Mat) {
        *self = expect("*=", crate::matmul::matmul(self, rhs));
    }
}

impl MulAssign<f64> for Mat {
    /// `Mat::operator*=(double)` — scale every element.
    fn mul_assign(&mut self, rhs: f64) {
        *self = expect("*= double", crate::arithm::convert_to(self, self.depth(), rhs, 0.0));
    }
}

impl DivAssign<&Mat> for Mat {
    /// `Mat::operator/=(Mat)` — element-wise division.
    fn div_assign(&mut self, rhs: &Mat) {
        *self = expect("/=", crate::arithm::divide(self, rhs, 1.0));
    }
}

/// The expansion of `CV_MAT_AUG_OPERATOR_TN(/=, ..., Mat)`.
///
/// `_TN` is upstream's fixed-size `Matx` overload, which converts its operand
/// to a `Mat` first; this port has no separate fixed-size type, so the scalar
/// form is the one that remains.
impl DivAssign<f64> for Mat {
    /// `Mat::operator/=(double)`.
    ///
    /// Upstream scales by `1./b` rather than dividing each element, so the
    /// reciprocal is taken once; the same is done here to keep the rounding
    /// identical.
    fn div_assign(&mut self, rhs: f64) {
        *self = expect("/= double", crate::arithm::convert_to(self, self.depth(), 1.0 / rhs, 0.0));
    }
}

impl BitAndAssign<&Mat> for Mat {
    /// `Mat::operator&=(Mat)`.
    fn bitand_assign(&mut self, rhs: &Mat) {
        *self = expect("&=", crate::arithm::bitwise_and(self, rhs));
    }
}

impl BitOrAssign<&Mat> for Mat {
    /// `Mat::operator|=(Mat)`.
    fn bitor_assign(&mut self, rhs: &Mat) {
        *self = expect("|=", crate::arithm::bitwise_or(self, rhs));
    }
}

impl BitXorAssign<&Mat> for Mat {
    /// `Mat::operator^=(Mat)`.
    fn bitxor_assign(&mut self, rhs: &Mat) {
        *self = expect("^=", crate::arithm::bitwise_xor(self, rhs));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CV_8UC1;

    fn m(v: &[u8]) -> Mat {
        Mat::from_vec(1, v.len() as i32, 1, v).unwrap()
    }

    #[test]
    fn arithmetic_operators_saturate_like_the_named_functions() {
        let mut a = m(&[10, 200, 30]);
        a += &m(&[5, 100, 5]);
        assert_eq!(a.to_vec::<u8>().unwrap(), vec![15, 255, 35], "saturating add");

        let mut b = m(&[10, 5, 30]);
        b -= &m(&[5, 10, 5]);
        assert_eq!(b.to_vec::<u8>().unwrap(), vec![5, 0, 25], "saturating subtract");

        let mut c = m(&[10, 20, 30]);
        c *= 2.0;
        assert_eq!(c.to_vec::<u8>().unwrap(), vec![20, 40, 60]);
        c /= 2.0;
        assert_eq!(c.to_vec::<u8>().unwrap(), vec![10, 20, 30]);
    }

    #[test]
    fn scalar_operators_apply_per_channel() {
        let mut a = m(&[10, 20, 30]);
        a += Scalar::from(5.0);
        assert_eq!(a.to_vec::<u8>().unwrap(), vec![15, 25, 35]);
        a -= Scalar::from(15.0);
        assert_eq!(a.to_vec::<u8>().unwrap(), vec![0, 10, 20], "saturates at zero");

        // Three channels take three different offsets, which is the whole
        // point of a Scalar operand. cv2.add(b, (1,2,3,0)) on the same pixels.
        let mut c = Mat::from_vec(1, 2, 3, &[10u8, 20, 30, 40, 50, 60]).unwrap();
        c += Scalar([1.0, 2.0, 3.0, 0.0]);
        assert_eq!(c.to_vec::<u8>().unwrap(), vec![11, 22, 33, 41, 52, 63]);
        // cv2.subtract(b, (5,60,10,0)) — channel 1 saturates at zero.
        let mut d = Mat::from_vec(1, 2, 3, &[10u8, 20, 30, 40, 50, 60]).unwrap();
        d -= Scalar([5.0, 60.0, 10.0, 0.0]);
        assert_eq!(d.to_vec::<u8>().unwrap(), vec![5, 0, 20, 35, 0, 50]);
    }

    #[test]
    fn bitwise_operators_work_on_the_raw_bits() {
        let mut a = m(&[0b1100, 0b1010]);
        a &= &m(&[0b1010, 0b0110]);
        assert_eq!(a.to_vec::<u8>().unwrap(), vec![0b1000, 0b0010]);

        let mut b = m(&[0b1100, 0b1010]);
        b |= &m(&[0b0011, 0b0101]);
        assert_eq!(b.to_vec::<u8>().unwrap(), vec![0b1111, 0b1111]);

        let mut c = m(&[0b1100, 0b1010]);
        c ^= &m(&[0b1010, 0b1010]);
        assert_eq!(c.to_vec::<u8>().unwrap(), vec![0b0110, 0b0000]);
    }

    /// The asymmetry worth knowing: `*=` against a matrix is `gemm`, not `mul`.
    #[test]
    fn mul_assign_against_a_matrix_is_matrix_multiplication() {
        let mut a = Mat::from_vec(2, 2, 1, &[1.0f64, 2.0, 3.0, 4.0]).unwrap();
        let i = Mat::from_vec(2, 2, 1, &[0.0f64, 1.0, 1.0, 0.0]).unwrap();
        a *= &i;
        // Swapping the columns, which only matrix multiplication does.
        assert_eq!(a.to_vec::<f64>().unwrap(), vec![2.0, 1.0, 4.0, 3.0]);
    }

    #[test]
    fn division_by_a_scalar_matches_scaling_by_the_reciprocal() {
        let mut a = Mat::from_vec(1, 3, 1, &[1.0f64, 2.0, 3.0]).unwrap();
        a /= 4.0;
        let scaled =
            crate::arithm::convert_to(&Mat::from_vec(1, 3, 1, &[1.0f64, 2.0, 3.0]).unwrap(),
                crate::Depth::F64, 1.0 / 4.0, 0.0).unwrap();
        assert_eq!(a.to_vec::<f64>().unwrap(), scaled.to_vec::<f64>().unwrap());
    }

    #[test]
    #[should_panic(expected = "Mat +=")]
    fn a_size_mismatch_panics_rather_than_producing_a_wrong_result() {
        let mut a = Mat::zeros(1, 3, CV_8UC1).unwrap();
        a += &Mat::zeros(1, 5, CV_8UC1).unwrap();
    }
}
