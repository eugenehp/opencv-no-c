//! Bit-exact `exp`, transcribed from `cv::softdouble`
//! (`modules/core/src/softfloat.cpp`).
//!
//! # Why this exists
//!
//! IEEE-754 requires `+`, `-`, `*` and `/` to be correctly rounded, so hardware
//! `f64` already agrees with OpenCV's software float on all four. **`exp` is
//! the exception** — the standard does not require it to be correctly rounded,
//! implementations differ, and OpenCV therefore ships its own so results are
//! reproducible across compilers and architectures.
//!
//! That one function is the whole gap. Measured against cv2 4.13.0, a
//! `getGaussianKernel` built on `f64::exp` differs by 1-2 ULP; built on this
//! one it is bit-identical. Since every Gaussian blur goes through that kernel,
//! this unblocks bit-exact parity for everything downstream of a blur —
//! `GaussianBlur`, the SIFT pyramid, the KAZE scale space.
//!
//! Only `exp` is transcribed. The rest of `softfloat` is unnecessary precisely
//! because hardware arithmetic already matches it.

/// `EXPTAB_SCALE` — the table holds 2^(i/64).
const EXPTAB_SCALE: i32 = 6;
/// `EXPTAB_MASK`
const EXPTAB_MASK: i32 = (1 << EXPTAB_SCALE) - 1;

/// `EXPPOLY_32F_A0`, as raw bits.
///
/// Kept as bits rather than a decimal literal: writing it out in decimal and
/// dropping a digit silently costs the last few ULP of every kernel built on
/// it, which is exactly the failure this module exists to prevent.
const EXPPOLY_A0_BITS: u64 = 0x3f83_ce0f_3e46_f431;

/// `expTab` — 2^(i/64) for i in 0..64, as raw `f64` bits.
static EXP_TAB: [u64; 64] = [
    0x3ff0000000000000, 0x3ff02c9a3e778061,
    0x3ff059b0d3158574, 0x3ff0874518759bc8,
    0x3ff0b5586cf9890f, 0x3ff0e3ec32d3d1a2,
    0x3ff11301d0125b51, 0x3ff1429aaea92de0,
    0x3ff172b83c7d517b, 0x3ff1a35beb6fcb75,
    0x3ff1d4873168b9aa, 0x3ff2063b88628cd6,
    0x3ff2387a6e756238, 0x3ff26b4565e27cdd,
    0x3ff29e9df51fdee1, 0x3ff2d285a6e4030b,
    0x3ff306fe0a31b715, 0x3ff33c08b26416ff,
    0x3ff371a7373aa9cb, 0x3ff3a7db34e59ff7,
    0x3ff3dea64c123422, 0x3ff4160a21f72e2a,
    0x3ff44e086061892d, 0x3ff486a2b5c13cd0,
    0x3ff4bfdad5362a27, 0x3ff4f9b2769d2ca7,
    0x3ff5342b569d4f82, 0x3ff56f4736b527da,
    0x3ff5ab07dd485429, 0x3ff5e76f15ad2148,
    0x3ff6247eb03a5585, 0x3ff6623882552225,
    0x3ff6a09e667f3bcd, 0x3ff6dfb23c651a2f,
    0x3ff71f75e8ec5f74, 0x3ff75feb564267c9,
    0x3ff7a11473eb0187, 0x3ff7e2f336cf4e62,
    0x3ff82589994cce13, 0x3ff868d99b4492ed,
    0x3ff8ace5422aa0db, 0x3ff8f1ae99157736,
    0x3ff93737b0cdc5e5, 0x3ff97d829fde4e50,
    0x3ff9c49182a3f090, 0x3ffa0c667b5de565,
    0x3ffa5503b23e255d, 0x3ffa9e6b5579fdbf,
    0x3ffae89f995ad3ad, 0x3ffb33a2b84f15fb,
    0x3ffb7f76f2fb5e47, 0x3ffbcc1e904bc1d2,
    0x3ffc199bdd85529c, 0x3ffc67f12e57d14b,
    0x3ffcb720dcef9069, 0x3ffd072d4a07897c,
    0x3ffd5818dcfba487, 0x3ffda9e603db3285,
    0x3ffdfc97337b9b5f, 0x3ffe502ee78b3ff6,
    0x3ffea4afa2a490da, 0x3ffefa1bee615a27,
    0x3fff50765b6e4540, 0x3fffa7c1819e90d8,
];

/// `cv::softdouble exp(const softdouble&)`.
///
/// Range-reduces by 1/ln(2) scaled by 64, looks the fractional part up in
/// `EXP_TAB`, and finishes with a degree-5 polynomial. The result is
/// bit-identical to OpenCV's.
#[must_use]
pub fn exp_bit_exact(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { f64::INFINITY } else { 0.0 };
    }

    // 1/ln(2) * (1 << EXPTAB_SCALE)
    let exp_prescale = f64::from_bits(0x3ff7_1547_652b_82fe) * f64::from(1 << EXPTAB_SCALE);
    let exp_postscale = 1.0 / f64::from(1 << EXPTAB_SCALE);
    // log10(DBL_MAX) < 3000
    let exp_max_val = f64::from(3000 * (1 << EXPTAB_SCALE));

    let a0c = f64::from_bits(EXPPOLY_A0_BITS);
    let a5 = 1.0 / a0c;
    let a4 = f64::from_bits(0x3fe6_2e42_fefa_39f1) / a0c;
    let a3 = f64::from_bits(0x3fce_bfbd_ff82_a45a) / a0c;
    let a2 = f64::from_bits(0x3fac_6b08_d81f_ec75) / a0c;
    let a1 = f64::from_bits(0x3f83_b2a7_2b4f_3cd3) / a0c;
    let a0 = f64::from_bits(0x3f55_e7aa_1566_c2a4) / a0c;

    // The unbiased exponent, matching upstream's `expF64UI(x.v) > 1023 + 10`.
    let biased_exp = ((x.to_bits() >> 52) & 0x7ff) as i32;
    let mut x0 = if biased_exp > 1023 + 10 {
        if x.is_sign_negative() {
            -exp_max_val
        } else {
            exp_max_val
        }
    } else {
        x * exp_prescale
    };

    // `cvRound` is round-half-to-even.
    let val0 = x0.round_ties_even() as i64;
    let t = ((val0 >> EXPTAB_SCALE) + 1023).clamp(0, 2047) as u64;
    // packToF64UI(0, t, 0) — the power of two 2^(t-1023).
    let buf = f64::from_bits(t << 52);

    x0 = (x0 - x0.round_ties_even()) * exp_postscale;

    let poly = ((((a0 * x0 + a1) * x0 + a2) * x0 + a3) * x0 + a4) * x0 + a5;
    let tab = f64::from_bits(EXP_TAB[(val0 as i32 & EXPTAB_MASK) as usize]);
    buf * a0c * tab * poly
}

/// `EXPPOLY_32F_A0` for the `f32` path — a *different* constant from the `f64`
/// one, and easy to conflate.
const EXPPOLY_A0_F32: f64 = 0.009_670_371_139_572_338;

/// `cv::hal::exp32f` — OpenCV's fast single-precision exponential.
///
/// Not the same routine as [`exp_bit_exact`]: it uses a degree-4 polynomial in
/// `f32` against a different `A0`. SIFT weights its orientation histogram with
/// this one, so reproducing SIFT's angles needs it rather than the `f64` form.
#[must_use]
pub fn exp32f(x: f32) -> f32 {
    // 1/ln(2), scaled by the table size. Spelled via the standard constant so
    // clippy does not read the literal as a typo for it.
    const EXP_PRESCALE: f32 = std::f32::consts::LOG2_E * 64.0;
    const POSTSCALE: f32 = 1.0 / 64.0;
    // Range clamp, as upstream does before scaling.
    let a4 = (1.000_000_000_000_002_4_f64 / EXPPOLY_A0_F32) as f32;
    let a3 = (0.693_147_180_552_144_8_f64 / EXPPOLY_A0_F32) as f32;
    let a2 = (0.240_226_501_095_130_15_f64 / EXPPOLY_A0_F32) as f32;
    let a1 = (0.055_503_393_667_531_25_f64 / EXPPOLY_A0_F32) as f32;

    let mut x0 = x.clamp(-88.0, 88.0) * EXP_PRESCALE;
    // `saturate_cast<int>` is round-half-to-even.
    let xi = x0.round_ties_even() as i32;
    x0 = (x0 - xi as f32) * POSTSCALE;

    let t = (xi >> EXPTAB_SCALE) + 127;
    let t = t.clamp(0, 255) as u32;
    let buf = f32::from_bits(t << 23);

    // Upstream's `expTab_f` is pre-scaled by `EXPPOLY_32F_A0`; [`EXP_TAB`] here
    // holds the unscaled powers of two shared with the f64 path, so the factor
    // is applied explicitly. Omitting it leaves the result 1/A0 — about 103x —
    // too large, which is what the first version of this did.
    let tab = f64::from_bits(EXP_TAB[(xi & EXPTAB_MASK) as usize]) as f32;
    buf * tab * (EXPPOLY_A0_F32 as f32) * ((((x0 + a1) * x0 + a2) * x0 + a3) * x0 + a4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_holds_powers_of_two_over_sixty_four() {
        assert_eq!(EXP_TAB.len(), 64);
        for (i, raw) in EXP_TAB.iter().enumerate() {
            let v = f64::from_bits(*raw);
            let want = 2f64.powf(i as f64 / 64.0);
            assert!((v - want).abs() < 1e-12, "entry {i}: {v} vs {want}");
        }
        assert_eq!(f64::from_bits(EXP_TAB[0]), 1.0);
    }

    #[test]
    fn it_agrees_with_the_hardware_exp_to_a_few_ulp() {
        // Not identical — that is the entire point — but it must be a correct
        // exponential, not merely a reproducible one.
        for x in [-5.0, -1.0, -0.25, 0.0, 0.5, 1.0, 3.0, 10.0] {
            let a = exp_bit_exact(x);
            let b = x.exp();
            assert!(
                (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                "exp({x}): {a} vs hardware {b}"
            );
        }
    }

    #[test]
    fn exp32f_is_a_correct_exponential() {
        // It is an approximation by design, but it must still be exp.
        for x in [-5.0f32, -1.0, 0.0, 0.5, 1.0, 3.0, 8.0] {
            let a = exp32f(x);
            let b = x.exp();
            assert!(
                (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                "exp32f({x}) = {a}, expected about {b}"
            );
        }
        assert!((exp32f(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_special_cases_match_upstream() {
        assert!(exp_bit_exact(f64::NAN).is_nan());
        assert_eq!(exp_bit_exact(f64::INFINITY), f64::INFINITY);
        assert_eq!(exp_bit_exact(f64::NEG_INFINITY), 0.0);
        assert_eq!(exp_bit_exact(0.0), 1.0);
    }
}
