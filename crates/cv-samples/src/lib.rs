//! The OpenCV sample programs, ported.
//!
//! Each module here is one file from `samples/` upstream, named after it. The
//! goal is not a demo but a **check**: every sample is reduced to the
//! computation it performs and run on exactly the input the upstream sample
//! uses, so that its result can be compared against what the reference library
//! produces for the same steps.
//!
//! The comparison itself is not wired up yet — the registry, the embedded
//! inputs and the first samples are. What exists today is that every
//! registered sample runs to completion and produces finite output, which the
//! tests below enforce.
//!
//! # What a sample becomes
//!
//! Upstream samples end in `imshow` and a `waitKey`. A window is not a result,
//! so each port ends instead in a [`SampleOutput`] — named numeric arrays and
//! named text lines. That is what gets compared, and it is also what makes the
//! set runnable anywhere.
//!
//! # WebAssembly
//!
//! This crate touches no file system, no clock, no threads, no environment and
//! no terminal. Every input is embedded with `include_bytes!` and decoded in
//! process, so the whole set builds and runs on `wasm32-unknown-unknown`:
//!
//! ```text
//! cargo build -p opencv-no-c-samples --target wasm32-unknown-unknown --release
//! ```
//!
//! [`sample_count`], [`sample_name`] and [`sample_run`] are the C-ABI entry
//! points a host calls; [`SAMPLES`] is the same registry for Rust callers.
//!
//! # Inputs are stored as PNG
//!
//! `assets/` holds the images the samples read, converted from
//! `samples/data/` to PNG. That is not cosmetic: this crate's PNG decoder is
//! bit-exact against the reference library, and its JPEG decoder is not — a
//! grayscale JPEG comes back within one level on 1.8% of pixels and a colour
//! one within two levels on 39%. Reading the original `.jpg` would put that
//! error into the input of every sample that uses one and make an exact
//! comparison impossible. The conversion is lossless: the PNG holds the
//! reference library's own decode of the original file.

#![warn(missing_docs)]
#![doc(html_root_url = "https://docs.rs/cv-samples/0.1.0")]

use cv_core::{Mat, MatLike, Result};

pub mod assets;
pub mod cpp;

/// What a ported sample produces instead of a window.
///
/// Arrays carry pixels, contours, matrices — anything the upstream sample
/// would have drawn or printed. Lines carry the text it wrote to `cout`.
#[derive(Debug, Default, Clone)]
pub struct SampleOutput {
    /// Named numeric results, in the order the sample produced them.
    pub arrays: Vec<(String, Vec<f64>)>,
    /// Named text results, for samples whose output really is text.
    pub lines: Vec<(String, String)>,
}

impl SampleOutput {
    /// An empty output.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a named array.
    pub fn push(&mut self, name: &str, values: Vec<f64>) {
        self.arrays.push((name.to_string(), values));
    }

    /// Record a whole image, flattened row-major with channels interleaved —
    /// the same layout `to_f64` gives and the same one the golden files use.
    ///
    /// # Errors
    ///
    /// Propagates whatever reading the matrix fails with.
    pub fn push_mat<M: MatLike + ?Sized>(&mut self, name: &str, m: &M) -> Result<()> {
        self.arrays.push((name.to_string(), cv_core::to_f64(m)?));
        Ok(())
    }

    /// Record a named line of text.
    pub fn push_line(&mut self, name: &str, text: impl Into<String>) {
        self.lines.push((name.to_string(), text.into()));
    }

    /// Look up an array by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&[f64]> {
        self.arrays.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_slice())
    }
}

/// One entry of the registry.
pub struct Sample {
    /// The name this sample is known by, matching the upstream file stem.
    pub name: &'static str,
    /// Where it came from, relative to the OpenCV tree.
    pub upstream: &'static str,
    /// One line on what the sample demonstrates.
    pub about: &'static str,
    /// Run it.
    pub run: fn() -> Result<SampleOutput>,
}

impl core::fmt::Debug for Sample {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Sample")
            .field("name", &self.name)
            .field("upstream", &self.upstream)
            .finish_non_exhaustive()
    }
}

/// Every ported sample.
pub static SAMPLES: &[Sample] = cpp::SAMPLES;

/// Find a sample by name.
#[must_use]
pub fn find(name: &str) -> Option<&'static Sample> {
    SAMPLES.iter().find(|s| s.name == name)
}

/// A `Mat` of `f64` values, for samples that build their input numerically.
///
/// # Errors
///
/// Fails if the dimensions do not match the data length.
pub fn mat_f64(rows: i32, cols: i32, cn: i32, data: &[f64]) -> Result<Mat> {
    Mat::from_vec(rows, cols, cn, data)
}

// --- C ABI, so a WebAssembly host can drive the set without a bindings crate.

/// How many samples are registered.
#[no_mangle]
pub extern "C" fn sample_count() -> u32 {
    SAMPLES.len() as u32
}

/// The name of sample `i`, as a pointer and length into static memory.
///
/// Returns a null pointer and zero length when `i` is out of range. The bytes
/// are UTF-8 and live for the life of the module, so the caller need not free
/// them.
///
/// # Safety
///
/// `len_out` must be null or point at a writable `u32`.
#[no_mangle]
pub unsafe extern "C" fn sample_name(i: u32, len_out: *mut u32) -> *const u8 {
    let Some(s) = SAMPLES.get(i as usize) else {
        if !len_out.is_null() {
            // SAFETY: the caller supplied a writable `u32`, checked non-null.
            unsafe { *len_out = 0 };
        }
        return core::ptr::null();
    };
    if !len_out.is_null() {
        // SAFETY: as above.
        unsafe { *len_out = s.name.len() as u32 };
    }
    s.name.as_ptr()
}

/// Run sample `i` and report only whether it succeeded.
///
/// The results themselves are not marshalled across the ABI — a host that
/// wants them links this crate as an `rlib` and calls [`SAMPLES`] directly.
/// This entry point exists so a WebAssembly host can execute the whole set and
/// see that every one of them completes.
#[no_mangle]
pub extern "C" fn sample_run(i: u32) -> i32 {
    match SAMPLES.get(i as usize) {
        Some(s) => match (s.run)() {
            Ok(_) => 0,
            Err(_) => -1,
        },
        None => -2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_sample_has_a_distinct_name_and_a_real_upstream_path() {
        let mut seen = std::collections::BTreeSet::new();
        for s in SAMPLES {
            assert!(seen.insert(s.name), "{} is registered twice", s.name);
            assert!(
                s.upstream.starts_with("samples/"),
                "{} points at {}, which is not a samples path",
                s.name,
                s.upstream
            );
            assert!(!s.about.is_empty(), "{} has no description", s.name);
        }
    }

    #[test]
    fn every_sample_runs_and_produces_something() {
        for s in SAMPLES {
            let out = (s.run)().unwrap_or_else(|e| panic!("{} failed: {e}", s.name));
            assert!(
                !out.arrays.is_empty() || !out.lines.is_empty(),
                "{} produced no output at all",
                s.name
            );
            for (k, v) in &out.arrays {
                assert!(!v.is_empty(), "{}: array {k} is empty", s.name);
                assert!(
                    v.iter().all(|x| x.is_finite()),
                    "{}: array {k} contains a non-finite value",
                    s.name
                );
            }
        }
    }

    #[test]
    fn the_c_abi_agrees_with_the_registry() {
        assert_eq!(sample_count() as usize, SAMPLES.len());
        let mut len = 0u32;
        // SAFETY: `len` is a live local `u32`.
        let p = unsafe { sample_name(0, &mut len) };
        assert!(!p.is_null());
        // SAFETY: `sample_name` returned a live pointer and its length.
        let name = unsafe { core::slice::from_raw_parts(p, len as usize) };
        assert_eq!(core::str::from_utf8(name).unwrap(), SAMPLES[0].name);
        assert_eq!(sample_run(0), 0);
        assert_eq!(sample_run(u32::MAX), -2);
    }
}
