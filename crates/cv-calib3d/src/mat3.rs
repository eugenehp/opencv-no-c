//! The 3x3 arithmetic the rest of this crate is built on.
//!
//! Rotations, homographies, essential and fundamental matrices are all
//! `[[f64; 3]; 3]`, and every module here had grown its own transpose and its
//! own matrix-vector product. These are the one copy of each. Nothing in this
//! module allocates or can fail, and every operation is written out in the
//! order upstream writes it, because the products feed comparisons whose
//! outcome a reassociation can change.

use crate::Mat3;

/// `m^T`.
#[must_use]
pub fn transpose3(m: &Mat3) -> Mat3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// `m * v`.
#[must_use]
pub fn mul3(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// `m^T * v`.
#[must_use]
pub fn mul3t(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// `a * b`.
#[must_use]
pub fn matmul3(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}

/// `det(m)`, expanded along the first row.
#[must_use]
pub fn det3(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Gram-Schmidt on the **rows**, completing the third with a cross product so
/// the result is a rotation and not merely orthogonal.
///
/// The `1e-300` floors are upstream's: a degenerate row leaves the matrix
/// finite rather than filling it with NaN.
pub fn orthonormalise_rows(r: &mut Mat3) {
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut a = r[0];
    let mut b = r[1];
    let na = norm(a).max(1e-300);
    for v in &mut a {
        *v /= na;
    }
    let d = dot(a, b);
    for i in 0..3 {
        b[i] -= d * a[i];
    }
    let nb = norm(b).max(1e-300);
    for v in &mut b {
        *v /= nb;
    }
    let c = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    *r = [a, b, c];
}

/// [`orthonormalise_rows`] on the **columns**.
///
/// Transposing either side of the row version is exact — it only moves values
/// — so this is the same arithmetic on the same numbers, not an approximation
/// of it.
pub fn orthonormalise_cols(m: &mut Mat3) {
    let mut t = transpose3(m);
    orthonormalise_rows(&mut t);
    *m = transpose3(&t);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Mat3 {
        [[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]]
    }

    #[test]
    fn transpose_is_an_involution() {
        let m = sample();
        assert_eq!(transpose3(&transpose3(&m)), m);
    }

    #[test]
    fn mul3t_agrees_with_transposing_first() {
        let (m, v) = (sample(), [1.5, -2.25, 0.75]);
        assert_eq!(mul3t(&m, v), mul3(&transpose3(&m), v));
    }

    #[test]
    fn matmul_by_the_transpose_of_a_rotation_is_the_identity() {
        let m = sample();
        let i = matmul3(&m, &transpose3(&m));
        for (r, row) in i.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                let want = f64::from(u8::from(r == c));
                assert!((v - want).abs() < 1e-12, "({r},{c}) was {v}");
            }
        }
    }

    #[test]
    fn det_of_a_rotation_is_one() {
        assert!((det3(&sample()) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_column_form_is_the_row_form_transposed() {
        // The property the consolidation rests on: two modules had written
        // these separately and they must stay the same arithmetic.
        let m = [[3.0, 0.1, -0.2], [0.05, 2.0, 0.3], [0.4, -0.15, 1.0]];
        let (mut rows, mut cols) = (transpose3(&m), m);
        orthonormalise_rows(&mut rows);
        orthonormalise_cols(&mut cols);
        assert_eq!(transpose3(&rows), cols);
    }

    #[test]
    fn a_degenerate_row_stays_finite() {
        let mut m = [[0.0; 3], [0.0; 3], [0.0; 3]];
        orthonormalise_rows(&mut m);
        assert!(m.iter().flatten().all(|v| v.is_finite()));
    }
}
