//! Discrete Fourier and cosine transforms (`modules/core/src/dxt.cpp`).
//!
//! # Layout
//!
//! Upstream packs a real-input DFT into a real-valued matrix (CCS format) by
//! default and only produces an explicit two-channel complex matrix when
//! `DFT_COMPLEX_OUTPUT` is set. This port always uses the explicit complex
//! layout — `CV_32FC2`/`CV_64FC2`, interleaved `(re, im)` — because CCS packing
//! is a storage optimisation that costs correctness bugs in callers and buys
//! nothing here. [`dft`] therefore behaves as if `DFT_COMPLEX_OUTPUT` were
//! always set, and [`idft`] accepts the same layout.
//!
//! # Algorithm
//!
//! A mixed-radix Bluestein-free split: power-of-two lengths use an iterative
//! radix-2 Cooley-Tukey, and every other length falls back to the naive
//! O(n^2) sum. That is slower than upstream's radix-3/5 factorisation for,
//! say, n=15, but it is exact in the same way and has no size restrictions.
//! Sizes are not padded, so unlike upstream there is no `getOptimalDFTSize`
//! requirement.

use crate::error::Result;
use crate::mat::{Mat, MatLike, MatLikeMut};
use crate::types::{MatType, CV_64FC1, CV_64FC2};
use crate::cv_assert;

/// Direction flag shared by [`dft`] and [`idft`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DftDirection {
    /// Forward transform, `exp(-2*pi*i*k*n/N)`.
    Forward,
    /// Inverse transform, `exp(+2*pi*i*k*n/N)`, scaled by `1/N`.
    Inverse,
}

/// `cv::DftFlags`, with upstream's values.
///
/// Upstream passes these or-ed together in one `flags` argument. This port
/// takes them as named parameters instead — [`dft`] and [`idft`] have a
/// `rows_only` for `DFT_ROWS`, the direction is a separate entry point, and the
/// complex-layout flags do not arise because the layout is always explicit (see
/// the module header). The constants are declared so a caller translating from
/// C++ can check a flags word it already has against them.
pub mod flags {
    /// `DFT_INVERSE` — perform the inverse transform; see [`super::idft`].
    pub const DFT_INVERSE: i32 = 1;
    /// `DFT_SCALE` — divide by the element count; [`super::idft`] always does.
    pub const DFT_SCALE: i32 = 2;
    /// `DFT_ROWS` — transform each row independently; the `rows_only` argument.
    pub const DFT_ROWS: i32 = 4;
    /// `DFT_COMPLEX_OUTPUT` — always in effect here, so never needs passing.
    pub const DFT_COMPLEX_OUTPUT: i32 = 16;
    /// `DFT_REAL_OUTPUT` — see [`super::idft_real`].
    pub const DFT_REAL_OUTPUT: i32 = 32;
    /// `DFT_COMPLEX_INPUT` — declare the input already complex; implied here.
    pub const DFT_COMPLEX_INPUT: i32 = 64;
    /// `DCT_INVERSE` — the `inverse` argument of [`super::dct`].
    pub const DCT_INVERSE: i32 = DFT_INVERSE;
    /// `DCT_ROWS` — the `rows_only` argument of [`super::dct`].
    pub const DCT_ROWS: i32 = DFT_ROWS;
}

/// In-place complex DFT of one line, chosen by length.
fn transform_line(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        radix2(re, im, inverse);
    } else {
        naive(re, im, inverse);
    }
}

/// Iterative radix-2 Cooley-Tukey, in place.
fn radix2(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let ang = sign * 2.0 * core::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let (xr, xi) = (re[i + k + len / 2], im[i + k + len / 2]);
                let (vr, vi) = (xr * cr - xi * ci, xr * ci + xi * cr);
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Direct O(n^2) evaluation, used for non-power-of-two lengths.
fn naive(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    let (sr, si) = (re.to_vec(), im.to_vec());
    for (k, (or, oi)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
        let (mut ar, mut ai) = (0.0f64, 0.0f64);
        for t in 0..n {
            let ang = sign * 2.0 * core::f64::consts::PI * (k * t % n) as f64 / n as f64;
            let (c, s) = (ang.cos(), ang.sin());
            ar += sr[t] * c - si[t] * s;
            ai += sr[t] * s + si[t] * c;
        }
        *or = ar;
        *oi = ai;
    }
}

/// Read any matrix as an interleaved complex plane pair.
fn to_complex<S: MatLike + ?Sized>(src: &S) -> Result<(Vec<f64>, Vec<f64>, i32, i32)> {
    let (h, w) = (src.rows(), src.cols());
    let cn = src.channels();
    cv_assert!(cn == 1 || cn == 2, "dft input must have 1 or 2 channels, got {cn}");
    let mut re = vec![0.0f64; (h * w) as usize];
    let mut im = vec![0.0f64; (h * w) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            re[i] = src.at_f64(y, x, 0)?;
            if cn == 2 {
                im[i] = src.at_f64(y, x, 1)?;
            }
        }
    }
    Ok((re, im, h, w))
}

/// Pack two planes back into a `CV_64FC2` matrix.
fn from_complex(re: &[f64], im: &[f64], h: i32, w: i32) -> Result<Mat> {
    let mut out = Mat::zeros(h, w, CV_64FC2)?;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let p = out.pixel_mut::<f64>(y, x)?;
            p[0] = re[i];
            p[1] = im[i];
        }
    }
    Ok(out)
}

/// Run the row transforms then the column transforms.
fn dft_2d(re: &mut [f64], im: &mut [f64], h: i32, w: i32, inverse: bool, rows_only: bool) {
    let (hu, wu) = (h as usize, w as usize);
    let mut br = vec![0.0f64; hu.max(wu)];
    let mut bi = vec![0.0f64; hu.max(wu)];

    for y in 0..hu {
        transform_line(&mut re[y * wu..(y + 1) * wu], &mut im[y * wu..(y + 1) * wu], inverse);
    }
    if rows_only || hu == 1 {
        return;
    }
    for x in 0..wu {
        for y in 0..hu {
            br[y] = re[y * wu + x];
            bi[y] = im[y * wu + x];
        }
        transform_line(&mut br[..hu], &mut bi[..hu], inverse);
        for y in 0..hu {
            re[y * wu + x] = br[y];
            im[y * wu + x] = bi[y];
        }
    }
}

/// `cv::dft(src, dst, flags)` — always with `DFT_COMPLEX_OUTPUT` semantics.
///
/// `rows_only` is upstream's `DFT_ROWS`. The result is `CV_64FC2`.
pub fn dft<S: MatLike + ?Sized>(
    src: &S,
    direction: DftDirection,
    rows_only: bool,
    scale: bool,
) -> Result<Mat> {
    cv_assert!(!src.is_empty(), "dft requires a non-empty input");
    let inverse = direction == DftDirection::Inverse;
    let (mut re, mut im, h, w) = to_complex(src)?;
    dft_2d(&mut re, &mut im, h, w, inverse, rows_only);
    if scale {
        let n = if rows_only { w as f64 } else { (h * w) as f64 };
        for (r, i) in re.iter_mut().zip(im.iter_mut()) {
            *r /= n;
            *i /= n;
        }
    }
    from_complex(&re, &im, h, w)
}

/// `cv::idft(src, dst, flags)` — the inverse transform, scaled by `1/N`.
///
/// Upstream leaves scaling to the caller unless `DFT_SCALE` is passed; here it
/// is applied, because an unscaled inverse is not an inverse.
///
/// The result is complex. For the real-valued inverse of a spectrum that is
/// known to be conjugate-symmetric — `DFT_REAL_OUTPUT` — use
/// [`idft_real`], which discards the (numerically negligible) imaginary part
/// rather than leaving the caller to do it.
pub fn idft<S: MatLike + ?Sized>(src: &S, rows_only: bool) -> Result<Mat> {
    dft(src, DftDirection::Inverse, rows_only, true)
}

/// `cv::idft(src, dst, DFT_SCALE | DFT_REAL_OUTPUT)` — a single-channel inverse.
///
/// Only meaningful when the spectrum came from a real input, in which case the
/// imaginary part is zero up to rounding. Upstream does not check that and
/// neither does this: it takes the real channel, which is what makes the result
/// a drop-in for the original image rather than a two-channel array the caller
/// has to split.
pub fn idft_real<S: MatLike + ?Sized>(src: &S, rows_only: bool) -> Result<Mat> {
    let complex = idft(src, rows_only)?;
    let (h, w) = (complex.rows(), complex.cols());
    let mut out = Mat::zeros(h, w, crate::types::MatType::new(complex.depth(), 1))?;
    for y in 0..h {
        for x in 0..w {
            crate::arithm::set_channel(&mut out, y, x, 0, complex.at_f64(y, x, 0)?)?;
        }
    }
    Ok(out)
}

/// `cv::mulSpectrums(a, b, dst, flags, conjB)`.
///
/// Both operands must be the two-channel complex layout [`dft`] produces.
pub fn mul_spectrums<A: MatLike + ?Sized, B: MatLike + ?Sized>(
    a: &A,
    b: &B,
    conj_b: bool,
) -> Result<Mat> {
    cv_assert!(a.size() == b.size(), "mulSpectrums operands must match in size");
    cv_assert!(
        a.channels() == 2 && b.channels() == 2,
        "mulSpectrums needs the 2-channel complex layout"
    );
    let (h, w) = (a.rows(), a.cols());
    let mut out = Mat::zeros(h, w, CV_64FC2)?;
    for y in 0..h {
        for x in 0..w {
            let (ar, ai) = (a.at_f64(y, x, 0)?, a.at_f64(y, x, 1)?);
            let (br, mut bi) = (b.at_f64(y, x, 0)?, b.at_f64(y, x, 1)?);
            if conj_b {
                bi = -bi;
            }
            let p = out.pixel_mut::<f64>(y, x)?;
            p[0] = ar * br - ai * bi;
            p[1] = ar * bi + ai * br;
        }
    }
    Ok(out)
}

/// One-dimensional DCT-II (or DCT-III when inverting), orthonormal.
///
/// This is upstream's normalisation: the k=0 basis vector carries
/// `sqrt(1/N)` and the rest `sqrt(2/N)`, so the transform is its own inverse
/// up to the direction flag.
fn dct_line(v: &[f64], inverse: bool) -> Vec<f64> {
    let n = v.len();
    let mut out = vec![0.0f64; n];
    let nf = n as f64;
    if !inverse {
        for (k, o) in out.iter_mut().enumerate() {
            let mut s = 0.0;
            for (t, &x) in v.iter().enumerate() {
                s += x * (core::f64::consts::PI * (t as f64 + 0.5) * k as f64 / nf).cos();
            }
            *o = s * if k == 0 { (1.0 / nf).sqrt() } else { (2.0 / nf).sqrt() };
        }
    } else {
        for (t, o) in out.iter_mut().enumerate() {
            let mut s = 0.0;
            for (k, &x) in v.iter().enumerate() {
                let w = if k == 0 { (1.0 / nf).sqrt() } else { (2.0 / nf).sqrt() };
                s += w * x * (core::f64::consts::PI * (t as f64 + 0.5) * k as f64 / nf).cos();
            }
            *o = s;
        }
    }
    out
}

/// `cv::dct(src, dst, flags)` — the real, separable cosine transform.
///
/// Single channel only, as upstream requires. Unlike upstream, the length need
/// not be even.
pub fn dct<S: MatLike + ?Sized>(src: &S, inverse: bool, rows_only: bool) -> Result<Mat> {
    cv_assert!(src.channels() == 1, "dct requires a single-channel matrix");
    cv_assert!(!src.is_empty(), "dct requires a non-empty input");
    let (h, w) = (src.rows(), src.cols());
    let mut buf = vec![0.0f64; (h * w) as usize];
    for y in 0..h {
        for x in 0..w {
            buf[(y * w + x) as usize] = src.at_f64(y, x, 0)?;
        }
    }
    let (hu, wu) = (h as usize, w as usize);
    for y in 0..hu {
        let row = dct_line(&buf[y * wu..(y + 1) * wu], inverse);
        buf[y * wu..(y + 1) * wu].copy_from_slice(&row);
    }
    if !rows_only && hu > 1 {
        let mut col = vec![0.0f64; hu];
        for x in 0..wu {
            for (y, c) in col.iter_mut().enumerate() {
                *c = buf[y * wu + x];
            }
            let t = dct_line(&col, inverse);
            for (y, &v) in t.iter().enumerate() {
                buf[y * wu + x] = v;
            }
        }
    }
    let mut out = Mat::zeros(h, w, CV_64FC1)?;
    for y in 0..h {
        for x in 0..w {
            out.pixel_mut::<f64>(y, x)?[0] = buf[(y * w + x) as usize];
        }
    }
    Ok(out)
}

/// `cv::idct(src, dst, flags)`.
pub fn idct<S: MatLike + ?Sized>(src: &S, rows_only: bool) -> Result<Mat> {
    dct(src, true, rows_only)
}

/// `cv::getOptimalDFTSize(vecsize)`.
///
/// Kept for API compatibility even though [`dft`] does not need it: this port
/// transforms any length directly. The value is still the smallest size at or
/// above `vecsize` that factors into 2, 3 and 5, so callers padding for speed
/// get the same answer as upstream.
#[must_use]
pub fn get_optimal_dft_size(vecsize: i32) -> i32 {
    if vecsize <= 2 {
        return vecsize.max(1);
    }
    let mut n = vecsize;
    loop {
        let mut m = n;
        for f in [2, 3, 5] {
            while m % f == 0 {
                m /= f;
            }
        }
        if m == 1 {
            return n;
        }
        n += 1;
    }
}

/// Convenience: the magnitude plane of a complex matrix, as `CV_64FC1`.
pub fn complex_magnitude<S: MatLike + ?Sized>(src: &S) -> Result<Mat> {
    cv_assert!(src.channels() == 2, "expected the 2-channel complex layout");
    let (h, w) = (src.rows(), src.cols());
    let mut out = Mat::zeros(h, w, CV_64FC1)?;
    for y in 0..h {
        for x in 0..w {
            let (re, im) = (src.at_f64(y, x, 0)?, src.at_f64(y, x, 1)?);
            out.pixel_mut::<f64>(y, x)?[0] = re.hypot(im);
        }
    }
    Ok(out)
}

/// The `MatType` a complex result uses, exposed so callers can pre-allocate.
#[must_use]
pub fn complex_type() -> MatType {
    CV_64FC2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real(v: &[f64], rows: i32, cols: i32) -> Mat {
        Mat::from_vec(rows, cols, 1, v).unwrap()
    }

    #[test]
    fn dft_of_a_constant_is_a_single_spike() {
        let m = real(&[1.0; 8], 1, 8);
        let f = dft(&m, DftDirection::Forward, false, false).unwrap();
        assert!((f.at_f64(0, 0, 0).unwrap() - 8.0).abs() < 1e-9);
        for x in 1..8 {
            assert!(f.at_f64(0, x, 0).unwrap().abs() < 1e-9, "bin {x} should be zero");
        }
    }

    #[test]
    fn dft_then_idft_is_the_identity() {
        // A non-power-of-two length exercises the naive path.
        for n in [8i32, 12, 15, 16] {
            let v: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin() * 3.0 + 1.0).collect();
            let m = real(&v, 1, n);
            let f = dft(&m, DftDirection::Forward, false, false).unwrap();
            let b = idft(&f, false).unwrap();
            for (i, &want) in v.iter().enumerate() {
                let got = b.at_f64(0, i as i32, 0).unwrap();
                assert!((got - want).abs() < 1e-9, "n={n} i={i}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn dft_2d_roundtrips() {
        let v: Vec<f64> = (0..12).map(|i| (i * i % 7) as f64).collect();
        let m = real(&v, 3, 4);
        let f = dft(&m, DftDirection::Forward, false, false).unwrap();
        let b = idft(&f, false).unwrap();
        for y in 0..3 {
            for x in 0..4 {
                let want = v[(y * 4 + x) as usize];
                assert!((b.at_f64(y, x, 0).unwrap() - want).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn mul_spectrums_conjugate_gives_a_real_autocorrelation_peak() {
        let v: Vec<f64> = (0..8).map(|i| (i % 3) as f64).collect();
        let m = real(&v, 1, 8);
        let f = dft(&m, DftDirection::Forward, false, false).unwrap();
        let p = mul_spectrums(&f, &f, true).unwrap();
        // |F|^2 is real and non-negative.
        for x in 0..8 {
            assert!(p.at_f64(0, x, 1).unwrap().abs() < 1e-9, "imag part at {x}");
            assert!(p.at_f64(0, x, 0).unwrap() >= -1e-9);
        }
    }

    #[test]
    fn dct_roundtrips_and_is_orthonormal() {
        let v: Vec<f64> = (0..6).map(|i| (i as f64 - 2.5) * 1.3).collect();
        let m = real(&v, 1, 6);
        let f = dct(&m, false, false).unwrap();
        let b = idct(&f, false).unwrap();
        for (i, &want) in v.iter().enumerate() {
            assert!((b.at_f64(0, i as i32, 0).unwrap() - want).abs() < 1e-9);
        }
        // Orthonormal: the transform preserves the L2 norm.
        let e0: f64 = v.iter().map(|x| x * x).sum();
        let e1: f64 = (0..6).map(|i| f.at_f64(0, i, 0).unwrap().powi(2)).sum();
        assert!((e0 - e1).abs() < 1e-9, "{e0} vs {e1}");
    }

    #[test]
    fn optimal_dft_size_matches_upstreams_factorisation() {
        // Values checked against cv2.getOptimalDFTSize.
        assert_eq!(get_optimal_dft_size(1), 1);
        assert_eq!(get_optimal_dft_size(7), 8);
        assert_eq!(get_optimal_dft_size(100), 100);
        assert_eq!(get_optimal_dft_size(101), 108);
        assert_eq!(get_optimal_dft_size(1000), 1000);
        assert_eq!(get_optimal_dft_size(1001), 1024);
    }
}
