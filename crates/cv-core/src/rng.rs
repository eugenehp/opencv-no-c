//! `cv::RNG` and the array fill helpers
//! (`core/include/opencv2/core/operations.hpp`, `core/src/rand.cpp`).
//!
//! The generator itself is bit-exact: the same seed produces the same sequence
//! as upstream, because `Rng::next` is a transcription of OpenCV's
//! multiply-with-carry step. That matters for reproducing results from OpenCV
//! code that seeds an `RNG` explicitly.

use crate::error::Result;
use crate::mat::{Mat, MatLikeMut};
use crate::types::Depth;
use crate::{cv_assert, geometry::Scalar};

/// `CV_RNG_COEFF` — the multiply-with-carry multiplier.
const CV_RNG_COEFF: u64 = 4_164_903_690;

/// `cv::RNG`'s distribution selector, as `RNG::fill` takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum Distribution {
    /// `RNG::UNIFORM` — uniform on `[a, b)`; see [`randu`].
    #[default]
    Uniform = 0,
    /// `RNG::NORMAL` — normal with mean `a` and standard deviation `b`; see [`randn`].
    Normal = 1,
}

/// `cv::RNG` — a multiply-with-carry generator.
#[derive(Debug, Clone, PartialEq)]
pub struct Rng {
    state: u64,
    /// The unused half of the last Box-Muller pair, if any.
    cached_gaussian: Option<f64>,
}

impl Default for Rng {
    /// `RNG()` — upstream's default seed.
    fn default() -> Self {
        Rng::new(2 * 3_141_592_653)
    }
}

impl Rng {
    /// `RNG(state)`. A zero seed is replaced by upstream's default, as
    /// `RNG::operator=` does, because zero is a fixed point of the recurrence.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Rng { state: if seed != 0 { seed } else { 0xFFFF_FFFF }, cached_gaussian: None }
    }

    /// `RNG::next()` — the raw 32-bit step.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.state = (self.state as u32 as u64) * CV_RNG_COEFF + (self.state >> 32);
        self.state as u32
    }

    /// `RNG::operator float()` — uniform in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 * 2.328_306_4e-10
    }

    /// `RNG::operator double()` — uniform in `[0, 1)`, consuming two words.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let t = self.next_u32() as u64;
        (((t << 32) | self.next_u32() as u64) as f64) * 5.421_010_862_427_522e-20
    }

    /// `RNG::uniform(int a, int b)` — uniform over `[a, b)`.
    #[inline]
    pub fn uniform_int(&mut self, a: i32, b: i32) -> i32 {
        if a == b {
            return a;
        }
        // Upstream's modulo of the raw 32-bit word, wrapping arithmetic and all.
        let span = (b as i64 - a as i64) as u64 as u32;
        (self.next_u32() % span).wrapping_add(a as u32) as i32
    }

    /// `RNG::uniform(double a, double b)`.
    #[inline]
    pub fn uniform_f64(&mut self, a: f64, b: f64) -> f64 {
        self.next_f64() * (b - a) + a
    }

    /// `RNG::gaussian(sigma)` — zero-mean normal deviate.
    ///
    /// Upstream uses its own `randn_0_1_32f` kernel; this is Box-Muller, which
    /// has the same distribution but not the same sequence. Every other method
    /// on this type is sequence-exact.
    pub fn gaussian(&mut self, sigma: f64) -> f64 {
        if let Some(g) = self.cached_gaussian.take() {
            return g * sigma;
        }
        // Reject u1 == 0 so the logarithm stays finite.
        let mut u1 = self.next_f64();
        while u1 <= f64::MIN_POSITIVE {
            u1 = self.next_f64();
        }
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * core::f64::consts::PI * u2;
        self.cached_gaussian = Some(r * theta.sin());
        r * theta.cos() * sigma
    }

    /// `cv::randu(dst, low, high)` — fill with uniform values per channel.
    ///
    /// **A float destination is not `uniform(a, b)` repeated.** `RNG::fill`
    /// takes one 32-bit step per element and maps the word to the range as
    /// `scale * X + bias`, where `X` is that word read **signed** and the bias
    /// is the midpoint `(a + b) / 2` — so the sign bit of the state is what
    /// puts a value in the lower or upper half of the interval.
    /// `RNG::operator double()` instead takes two steps and reads them
    /// unsigned. The two sequences share no values; using the scalar form here
    /// gave every element of a float `randu` an offset of half the range.
    pub fn fill_uniform<M: MatLikeMut + ?Sized>(
        &mut self,
        dst: &mut M,
        low: Scalar,
        high: Scalar,
    ) -> Result<()> {
        let cn = dst.channels();
        let depth = dst.depth();
        // `randf_32f` narrows both parameters to `f32` before using them;
        // `randf_64f` keeps them double. The narrowing is visible in the
        // result, so it is done here and not at the point of use.
        let narrow = depth == Depth::F32;
        let scale = |d: f64| {
            // 2^-64 for the rotated 64-bit word, 2^-32 for the 32-bit one.
            let s = if depth == Depth::F64 { 5.421_010_862_427_522e-20 } else { 2.328_306_436_538_696_3e-10 };
            if narrow {
                f64::from((d * s) as f32)
            } else {
                d * s
            }
        };
        for y in 0..dst.rows() {
            for x in 0..dst.cols() {
                for c in 0..cn {
                    let i = (c as usize).min(3);
                    let (lo, hi) = (low.0[i], high.0[i]);
                    let v = match depth {
                        Depth::F64 => {
                            self.state = (self.state as u32 as u64)
                                .wrapping_mul(CV_RNG_COEFF)
                                .wrapping_add(self.state >> 32);
                            // The whole state, rotated by 32 and read signed.
                            let w = self.state.rotate_right(32) as i64;
                            w as f64 * scale(hi - lo) + (hi + lo) * 0.5
                        }
                        Depth::F32 => {
                            let t = self.next_u32() as i32;
                            let bias = ((hi + lo) * 0.5) as f32;
                            f64::from(t as f32 * scale(hi - lo) as f32 + bias)
                        }
                        // Integer destinations draw from a half-open integer
                        // range, which `uniform_int` already implements.
                        _ => self.uniform_int(lo as i32, hi as i32) as f64,
                    };
                    set_one(dst, y, x, c, v)?;
                }
            }
        }
        Ok(())
    }

    /// `cv::randn(dst, mean, stddev)`.
    pub fn fill_normal<M: MatLikeMut + ?Sized>(
        &mut self,
        dst: &mut M,
        mean: Scalar,
        stddev: Scalar,
    ) -> Result<()> {
        let cn = dst.channels();
        for y in 0..dst.rows() {
            for x in 0..dst.cols() {
                for c in 0..cn {
                    let i = (c as usize).min(3);
                    let v = mean.0[i] + self.gaussian(stddev.0[i]);
                    set_one(dst, y, x, c, v)?;
                }
            }
        }
        Ok(())
    }

    /// `cv::randShuffle(dst, iterFactor)` — Fisher-Yates over the elements.
    pub fn shuffle<M: MatLikeMut + ?Sized>(&mut self, dst: &mut M) -> Result<()> {
        cv_assert!(dst.rows() == 1 || dst.cols() == 1, "randShuffle needs a single row or column");
        let n = dst.total();
        if n < 2 {
            return Ok(());
        }
        let cn = dst.channels();
        let is_row = dst.rows() == 1;
        let at = |i: usize| -> (i32, i32) {
            if is_row {
                (0, i as i32)
            } else {
                (i as i32, 0)
            }
        };
        for i in (1..n).rev() {
            let j = self.uniform_int(0, i as i32 + 1) as usize;
            if i == j {
                continue;
            }
            let (yi, xi) = at(i);
            let (yj, xj) = at(j);
            for c in 0..cn {
                let a = dst.at_f64(yi, xi, c)?;
                let b = dst.at_f64(yj, xj, c)?;
                set_one(dst, yi, xi, c, b)?;
                set_one(dst, yj, xj, c, a)?;
            }
        }
        Ok(())
    }
}

/// Depth-generic single-channel store for any [`MatLikeMut`].
fn set_one<M: MatLikeMut + ?Sized>(m: &mut M, y: i32, x: i32, c: i32, v: f64) -> Result<()> {
    use crate::saturate::Elem;
    use crate::types::Depth;
    let i = c as usize;
    match m.depth() {
        Depth::U8 => m.pixel_mut::<u8>(y, x)?[i] = u8::from_f64(v),
        Depth::S8 => m.pixel_mut::<i8>(y, x)?[i] = i8::from_f64(v),
        Depth::U16 => m.pixel_mut::<u16>(y, x)?[i] = u16::from_f64(v),
        Depth::S16 => m.pixel_mut::<i16>(y, x)?[i] = i16::from_f64(v),
        Depth::S32 => m.pixel_mut::<i32>(y, x)?[i] = i32::from_f64(v),
        Depth::F32 => m.pixel_mut::<f32>(y, x)?[i] = v as f32,
        Depth::F64 => m.pixel_mut::<f64>(y, x)?[i] = v,
        Depth::F16 => {
            return crate::cv_error!(
                crate::Code::StsNotImplemented,
                "CV_16F is not ported yet"
            )
        }
    }
    Ok(())
}

impl Rng {
    /// `cv::RNG::fill(mat, distType, a, b)`.
    ///
    /// Dispatches on `dist`: `a`/`b` are the bounds for
    /// [`Distribution::Uniform`] and the mean and standard deviation for
    /// [`Distribution::Normal`], which is upstream's overloading of the pair.
    pub fn fill<M: MatLikeMut + ?Sized>(
        &mut self,
        dst: &mut M,
        dist: Distribution,
        a: Scalar,
        b: Scalar,
    ) -> Result<()> {
        match dist {
            Distribution::Uniform => self.fill_uniform(dst, a, b),
            Distribution::Normal => self.fill_normal(dst, a, b),
        }
    }
}

/// `cv::randu(dst, low, high)` using a fresh default-seeded generator.
pub fn randu<M: MatLikeMut + ?Sized>(dst: &mut M, low: Scalar, high: Scalar) -> Result<()> {
    Rng::default().fill_uniform(dst, low, high)
}

/// `cv::randn(dst, mean, stddev)` using a fresh default-seeded generator.
pub fn randn<M: MatLikeMut + ?Sized>(dst: &mut M, mean: Scalar, stddev: Scalar) -> Result<()> {
    Rng::default().fill_normal(dst, mean, stddev)
}

/// Convenience: a new matrix filled with uniform noise.
pub fn random_mat(
    rows: i32,
    cols: i32,
    typ: crate::types::MatType,
    low: Scalar,
    high: Scalar,
    rng: &mut Rng,
) -> Result<Mat> {
    let mut m = Mat::zeros(rows, cols, typ)?;
    rng.fill_uniform(&mut m, low, high)?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CV_32FC1, CV_8UC1, CV_8UC3};
    use crate::MatLike;

    #[test]
    fn next_matches_the_upstream_recurrence() {
        // Reproduce the C++ expression directly for the first few steps.
        let mut state: u64 = 12345;
        let mut expected = Vec::new();
        for _ in 0..5 {
            state = (state as u32 as u64) * 4_164_903_690 + (state >> 32);
            expected.push(state as u32);
        }

        let mut rng = Rng::new(12345);
        let got: Vec<u32> = (0..5).map(|_| rng.next_u32()).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn the_same_seed_reproduces_the_same_sequence() {
        let mut a = Rng::new(999);
        let mut b = Rng::new(999);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
        // A different seed diverges.
        let mut c = Rng::new(1000);
        assert_ne!(a.next_u32(), c.next_u32());
    }

    #[test]
    fn a_zero_seed_does_not_stick() {
        // Zero is a fixed point of the recurrence, so it must be replaced.
        let mut rng = Rng::new(0);
        let first = rng.next_u32();
        let second = rng.next_u32();
        assert_ne!(first, 0);
        assert_ne!(first, second);
    }

    #[test]
    fn floats_stay_in_the_unit_interval() {
        let mut rng = Rng::new(7);
        for _ in 0..2000 {
            let f = rng.next_f32();
            assert!((0.0..1.0).contains(&f), "f32 {f} escaped [0,1)");
            let d = rng.next_f64();
            assert!((0.0..1.0).contains(&d), "f64 {d} escaped [0,1)");
        }
    }

    #[test]
    fn uniform_int_respects_its_bounds() {
        let mut rng = Rng::new(42);
        let mut seen_lo = false;
        let mut seen_hi = false;
        for _ in 0..5000 {
            let v = rng.uniform_int(10, 20);
            assert!((10..20).contains(&v), "{v} outside [10,20)");
            seen_lo |= v == 10;
            seen_hi |= v == 19;
        }
        assert!(seen_lo && seen_hi, "range endpoints never produced");
        // A degenerate range returns the bound itself.
        assert_eq!(rng.uniform_int(5, 5), 5);
    }

    #[test]
    fn uniform_int_handles_negative_ranges() {
        let mut rng = Rng::new(3);
        for _ in 0..2000 {
            let v = rng.uniform_int(-50, -10);
            assert!((-50..-10).contains(&v), "{v} outside [-50,-10)");
        }
    }

    #[test]
    fn uniform_f64_spans_the_requested_interval() {
        let mut rng = Rng::new(11);
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for _ in 0..5000 {
            let v = rng.uniform_f64(-2.5, 7.5);
            assert!((-2.5..7.5).contains(&v));
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(lo < -2.0 && hi > 7.0, "poor coverage: {lo}..{hi}");
    }

    #[test]
    fn gaussian_has_the_requested_moments() {
        let mut rng = Rng::new(2024);
        let n = 20000;
        let vals: Vec<f64> = (0..n).map(|_| rng.gaussian(3.0)).collect();
        let mean = vals.iter().sum::<f64>() / n as f64;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.1, "mean {mean} should be near 0");
        assert!((var.sqrt() - 3.0).abs() < 0.15, "sd {} should be near 3", var.sqrt());
    }

    #[test]
    fn gaussian_uses_both_halves_of_each_pair() {
        // Box-Muller caches one value; consuming an odd count must still work
        // and must not repeat the cached number.
        let mut rng = Rng::new(5);
        let a = rng.gaussian(1.0);
        let b = rng.gaussian(1.0);
        let c = rng.gaussian(1.0);
        assert_ne!(a, b);
        assert_ne!(b, c);
    }

    #[test]
    fn randu_fills_and_saturates_per_channel() {
        let mut m = Mat::zeros(8, 8, CV_8UC3).unwrap();
        let mut rng = Rng::new(1);
        rng.fill_uniform(&mut m, Scalar::all(10.0), Scalar::all(20.0)).unwrap();
        for v in m.to_vec::<u8>().unwrap() {
            assert!((10..20).contains(&v), "{v} outside [10,20)");
        }
    }

    #[test]
    fn randu_on_floats_uses_the_continuous_range() {
        let mut m = Mat::zeros(6, 6, CV_32FC1).unwrap();
        let mut rng = Rng::new(2);
        rng.fill_uniform(&mut m, Scalar::single(-1.0), Scalar::single(1.0)).unwrap();
        let v = m.to_vec::<f32>().unwrap();
        assert!(v.iter().all(|&x| (-1.0..1.0).contains(&x)));
        // Continuous values, not just integers.
        assert!(v.iter().any(|&x| x.fract() != 0.0));
    }

    #[test]
    fn randn_produces_the_requested_distribution() {
        let mut m = Mat::zeros(100, 100, CV_32FC1).unwrap();
        let mut rng = Rng::new(9);
        rng.fill_normal(&mut m, Scalar::single(0.5), Scalar::single(0.1)).unwrap();
        let (mu, sd) = crate::mean_std_dev::<Mat, Mat>(&m, None).unwrap();
        assert!((mu[0] - 0.5).abs() < 0.01, "mean {}", mu[0]);
        assert!((sd[0] - 0.1).abs() < 0.01, "sd {}", sd[0]);
    }

    #[test]
    fn shuffle_permutes_without_losing_elements() {
        let data: Vec<u8> = (0..50).collect();
        let mut m = Mat::from_vec(1, 50, 1, &data).unwrap();
        let mut rng = Rng::new(17);
        rng.shuffle(&mut m).unwrap();

        let mut got = m.to_vec::<u8>().unwrap();
        assert_ne!(got, data, "shuffle left the order untouched");
        got.sort_unstable();
        assert_eq!(got, data, "shuffle lost or duplicated elements");
    }

    #[test]
    fn shuffle_works_on_a_column() {
        let data: Vec<u8> = (0..20).collect();
        let mut m = Mat::from_vec(20, 1, 1, &data).unwrap();
        Rng::new(4).shuffle(&mut m).unwrap();
        let mut got = m.to_vec::<u8>().unwrap();
        got.sort_unstable();
        assert_eq!(got, data);
    }

    #[test]
    fn shuffle_rejects_a_two_dimensional_matrix() {
        let mut m = Mat::zeros(4, 4, CV_8UC1).unwrap();
        assert!(Rng::new(1).shuffle(&mut m).is_err());
    }

    #[test]
    fn free_functions_fill_a_matrix() {
        let mut m = Mat::zeros(4, 4, CV_8UC1).unwrap();
        randu(&mut m, Scalar::single(100.0), Scalar::single(101.0)).unwrap();
        assert!(m.to_vec::<u8>().unwrap().iter().all(|&v| v == 100));

        let mut f = Mat::zeros(4, 4, CV_32FC1).unwrap();
        randn(&mut f, Scalar::single(0.0), Scalar::single(1.0)).unwrap();
        assert!(f.to_vec::<f32>().unwrap().iter().any(|&v| v != 0.0));
    }

    #[test]
    fn random_mat_builds_a_filled_matrix() {
        let mut rng = Rng::new(8);
        let m = random_mat(5, 5, CV_8UC1, Scalar::single(0.0), Scalar::single(255.0), &mut rng)
            .unwrap();
        assert_eq!(m.size(), crate::Size::new(5, 5));
        let v = m.to_vec::<u8>().unwrap();
        assert!(v.iter().any(|&x| x != v[0]), "all values identical");
    }
}
