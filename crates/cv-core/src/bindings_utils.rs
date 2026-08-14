//! `cv::utils::dump*` — one-line descriptions of an argument
//! (`modules/core/include/opencv2/core/bindings_utils.hpp`).
//!
//! These exist upstream to let the Python binding tests assert what a wrapper
//! actually received. They are useful in their own right when a shape or type
//! mismatch has to be pinned down, and the strings are reproduced exactly, so
//! output from here can be diffed against `cv2.utils.dumpInputArray(...)`.
//!
//! Upstream's `kind` and `flags` words describe the C++ `_InputArray`
//! type-erasure this port does not have; the values reproduced below are the
//! ones cv2 reports for a plain `Mat` or a `vector<Mat>` argument, which is
//! what a Rust caller is always passing.
//!
//! The small classes at the end — [`OriginalClassName`],
//! [`ClassWithKeywordProperties`], [`FunctionParams`] — exist upstream to
//! exercise the Python wrapper generator: renaming, keyword arguments, nested
//! types, chained setters. cv2 exposes them
//! (`cv2.utils.nested.OriginalClassName_originalName()` answers), so they are
//! ported with the same defaults and values rather than dropped.

use crate::geometry::Rect;
use crate::mat::MatLike;

/// `_InputArray::KIND_SHIFT` — where the kind bits start in `flags`.
pub const KIND_SHIFT: i32 = 16;
/// `_InputArray::KIND_MASK` — the bits holding the kind.
pub const KIND_MASK: i32 = 31 << KIND_SHIFT;
/// `_InputArray::MAT` — a single matrix argument.
pub const MAT: i32 = 1 << KIND_SHIFT;
/// `_InputArray::MATX` — a fixed-size matrix.
pub const MATX: i32 = 2 << KIND_SHIFT;
/// `_InputArray::STD_VECTOR` — a vector of scalars.
pub const STD_VECTOR: i32 = 3 << KIND_SHIFT;
/// `_InputArray::STD_VECTOR_VECTOR` — a vector of vectors.
pub const STD_VECTOR_VECTOR: i32 = 4 << KIND_SHIFT;
/// `_InputArray::STD_VECTOR_MAT` — a vector of matrices.
pub const STD_VECTOR_MAT: i32 = 5 << KIND_SHIFT;
/// `_InputArray::EXPR` — a lazily evaluated matrix expression.
pub const EXPR: i32 = 6 << KIND_SHIFT;
/// `_InputArray::NONE` — no argument.
pub const NONE: i32 = 0 << KIND_SHIFT;
/// `_InputArray::STD_ARRAY` — a `std::array` of scalars.
pub const STD_ARRAY: i32 = 9 << KIND_SHIFT;
/// `_InputArray::STD_ARRAY_MAT` — a `std::array` of matrices.
pub const STD_ARRAY_MAT: i32 = 10 << KIND_SHIFT;
/// `_InputArray::STD_BOOL_VECTOR` — `vector<bool>`, which is bit-packed in C++.
pub const STD_BOOL_VECTOR: i32 = 12 << KIND_SHIFT;
/// `_InputArray::FIXED_TYPE` — the destination type may not be changed.
pub const FIXED_TYPE: i32 = 0x8000 << KIND_SHIFT;
/// `_InputArray::FIXED_SIZE` — the destination size may not be changed.
pub const FIXED_SIZE: i32 = 0x4000 << KIND_SHIFT;

/// Shared body of the four array dumps.
fn dump_array(label: &str, flags: i32, m: &(impl MatLike + ?Sized)) -> String {
    format!(
        "{label}: empty()={} kind=0x{:08x} flags=0x{:08x} total(-1)={} dims(-1)=2 size(-1)={}x{} type(-1)={}",
        m.is_empty(),
        MAT,
        flags,
        m.total(),
        m.cols(),
        m.rows(),
        crate::check::type_to_string(m.type_code()),
    )
}

/// `cv::utils::dumpInputArray(argument)`.
pub fn dump_input_array(m: &(impl MatLike + ?Sized)) -> String {
    dump_array("InputArray", 0x0101_0000, m)
}

/// `cv::utils::dumpInputOutputArray(argument)`.
///
/// The only difference from [`dump_input_array`] is the access bits in `flags`,
/// which is exactly what the label is there to distinguish.
pub fn dump_input_output_array(m: &(impl MatLike + ?Sized)) -> String {
    dump_array("InputOutputArray", 0x0301_0000, m)
}

/// Shared body of the two vector-of-arrays dumps.
fn dump_array_of_arrays<M: MatLike>(label: &str, flags: i32, v: &[M]) -> String {
    let mut s = format!(
        "{label}: empty()={} kind=0x{:08x} flags=0x{:08x} total(-1)={} dims(-1)=1 size(-1)={}x1",
        v.is_empty(),
        STD_VECTOR_MAT,
        flags,
        v.len(),
        v.len(),
    );
    // Upstream reports the element type once, then each element's own shape.
    if let Some(first) = v.first() {
        s.push_str(&format!(" type(0)={}", crate::check::type_to_string(first.type_code())));
        for (i, m) in v.iter().enumerate() {
            s.push_str(&format!(" dims({i})=2 size({i})={}x{}", m.cols(), m.rows()));
        }
    }
    s
}

/// `cv::utils::dumpInputArrayOfArrays(argument)`.
pub fn dump_input_array_of_arrays<M: MatLike>(v: &[M]) -> String {
    dump_array_of_arrays("InputArrayOfArrays", 0x0105_0000, v)
}

/// `cv::utils::dumpInputOutputArrayOfArrays(argument)`.
pub fn dump_input_output_array_of_arrays<M: MatLike>(v: &[M]) -> String {
    dump_array_of_arrays("InputOutputArrayOfArrays", 0x0305_0000, v)
}

/// `cv::utils::dumpVectorOfInt(vec)`.
pub fn dump_vector_of_int(v: &[i32]) -> String {
    format!("[{}]", v.iter().map(i32::to_string).collect::<Vec<_>>().join(", "))
}

/// `cv::utils::dumpVectorOfDouble(vec)`.
///
/// Two decimals, which is upstream's `"%.2f"` — not a general-purpose format,
/// but the one the binding tests compare against.
pub fn dump_vector_of_double(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>().join(", "))
}

/// `cv::utils::dumpVectorOfRect(vec)`.
pub fn dump_vector_of_rect(v: &[Rect]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(|r| format!("[x={}, y={}, w={}, h={}]", r.x, r.y, r.width, r.height))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// `cv::utils::testEchoBooleanFunction(flag)` — returns its argument.
#[must_use]
pub fn test_echo_boolean_function(flag: bool) -> bool {
    flag
}

/// `cv::utils::fs::getCacheDirectoryForDownloads()`.
///
/// Honours `OPENCV_CACHE_DIR` if set, as upstream does, then falls back to the
/// platform cache directory. Nothing in this port downloads anything; the
/// function exists so code that asks where a model *would* go still works.
#[must_use]
pub fn get_cache_directory_for_downloads() -> String {
    cache_directory(
        std::env::var("OPENCV_CACHE_DIR").ok(),
        std::env::var("XDG_CACHE_HOME").ok(),
        std::env::var("HOME").ok(),
    )
}

/// The path choice, with the environment passed in.
///
/// Split out so it can be tested without mutating the process environment,
/// which every other thread shares.
fn cache_directory(
    cache_dir: Option<String>,
    xdg: Option<String>,
    home: Option<String>,
) -> String {
    if let Some(dir) = cache_dir.filter(|d| !d.is_empty()) {
        return dir;
    }
    let home = home.unwrap_or_else(|| ".".to_string());
    if cfg!(target_os = "macos") {
        format!("{home}/Library/Caches/opencv/downloads")
    } else {
        xdg.filter(|c| !c.is_empty())
            .map_or_else(|| format!("{home}/.cache/opencv/downloads"), |c| format!("{c}/opencv/downloads"))
    }
}

/// `cv::utils::ClassWithKeywordProperties` — a fixture whose field names are
/// reserved words in Python.
///
/// Ported because the same trap exists in Rust: `except` is fine, but a field
/// called `lambda` would not be, so the wrapper has to rename. Upstream's
/// defaults are 24 and 42.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassWithKeywordProperties {
    /// `lambda` — read-write upstream (`CV_PROP_RW`). Renamed here because
    /// `lambda` is a reserved word in Rust as well as Python.
    pub lambda_: i32,
    /// `except` — read-only upstream (`CV_PROP`). Also a reserved word.
    pub except_: i32,
}

impl Default for ClassWithKeywordProperties {
    fn default() -> Self {
        ClassWithKeywordProperties::new(24, 42)
    }
}

impl ClassWithKeywordProperties {
    /// `ClassWithKeywordProperties(lambda_arg = 24, except_arg = 42)`.
    #[must_use]
    pub const fn new(lambda_arg: i32, except_arg: i32) -> Self {
        ClassWithKeywordProperties { lambda_: lambda_arg, except_: except_arg }
    }
}

/// `cv::utils::FunctionParams` — a fixture for chained named-argument setters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FunctionParams {
    /// `lambda`, defaulting to -1.
    pub lambda_: i32,
    /// `sigma`, defaulting to 0.
    pub sigma: f32,
}

impl Default for FunctionParams {
    fn default() -> Self {
        FunctionParams { lambda_: -1, sigma: 0.0 }
    }
}

impl FunctionParams {
    /// `FunctionParams::setLambda(value)` — returns self, so calls chain.
    #[must_use]
    pub const fn set_lambda(mut self, value: i32) -> Self {
        self.lambda_ = value;
        self
    }

    /// `FunctionParams::setSigma(value)`.
    #[must_use]
    pub const fn set_sigma(mut self, value: f32) -> Self {
        self.sigma = value;
        self
    }
}

/// `cv::utils::nested::OriginalClassName::Params`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// `int_value`, defaulting to 123.
    pub int_value: i32,
    /// `float_value`, defaulting to 3.5.
    pub float_value: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params::new(123, 3.5)
    }
}

impl Params {
    /// `Params(int_param = 123, float_param = 3.5f)`.
    #[must_use]
    pub const fn new(int_param: i32, float_param: f32) -> Self {
        Params { int_value: int_param, float_value: float_param }
    }
}

/// `cv::utils::nested::OriginalClassName` — a fixture for the renaming path.
///
/// Upstream tags it `CV_WRAP_AS(ExportClassName)`, so Python sees
/// `ExportClassName` while [`OriginalClassName::original_name`] still reports
/// the C++ name. Rust needs no such indirection; both names are kept so the
/// behaviour a binding test asserts is reproducible.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OriginalClassName {
    params: Params,
}

/// `ExportClassName` — the name upstream's bindings expose for
/// [`OriginalClassName`].
pub type ExportClassName = OriginalClassName;

impl OriginalClassName {
    /// `OriginalClassName::create(params)`.
    #[must_use]
    pub const fn create(params: Params) -> Self {
        OriginalClassName { params }
    }

    /// `OriginalClassName::getIntParam()`.
    #[must_use]
    pub const fn get_int_param(&self) -> i32 {
        self.params.int_value
    }

    /// `OriginalClassName::getFloatParam()`.
    #[must_use]
    pub const fn get_float_param(&self) -> f32 {
        self.params.float_value
    }

    /// `OriginalClassName::originalName()` — the C++ class name.
    #[must_use]
    pub fn original_name() -> &'static str {
        "OriginalClassName"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::Mat;
    use crate::types::{CV_32FC3, CV_8UC1};

    /// Reference strings are `cv2.utils.dump*` on the same arguments.
    #[test]
    fn input_array_dumps_match_cv2() {
        let m = Mat::zeros(3, 4, CV_8UC1).unwrap();
        assert_eq!(
            dump_input_array(&m),
            "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=12 dims(-1)=2 size(-1)=4x3 type(-1)=CV_8UC1"
        );
        let f = Mat::zeros(3, 4, CV_32FC3).unwrap();
        assert_eq!(
            dump_input_array(&f),
            "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=12 dims(-1)=2 size(-1)=4x3 type(-1)=CV_32FC3"
        );
        let e = Mat::empty(CV_8UC1);
        assert_eq!(
            dump_input_array(&e),
            "InputArray: empty()=true kind=0x00010000 flags=0x01010000 total(-1)=0 dims(-1)=2 size(-1)=0x0 type(-1)=CV_8UC1"
        );
        let io = Mat::zeros(2, 2, CV_8UC1).unwrap();
        assert_eq!(
            dump_input_output_array(&io),
            "InputOutputArray: empty()=false kind=0x00010000 flags=0x03010000 total(-1)=4 dims(-1)=2 size(-1)=2x2 type(-1)=CV_8UC1"
        );
    }

    #[test]
    fn array_of_arrays_dumps_match_cv2() {
        let v = vec![Mat::zeros(3, 4, CV_8UC1).unwrap(), Mat::zeros(2, 2, CV_8UC1).unwrap()];
        assert_eq!(
            dump_input_array_of_arrays(&v),
            "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=2 dims(-1)=1 size(-1)=2x1 type(0)=CV_8UC1 dims(0)=2 size(0)=4x3 dims(1)=2 size(1)=2x2"
        );
        let one = vec![Mat::zeros(3, 4, CV_8UC1).unwrap()];
        assert_eq!(
            dump_input_output_array_of_arrays(&one),
            "InputOutputArrayOfArrays: empty()=false kind=0x00050000 flags=0x03050000 total(-1)=1 dims(-1)=1 size(-1)=1x1 type(0)=CV_8UC1 dims(0)=2 size(0)=4x3"
        );
    }

    #[test]
    fn vector_dumps_match_cv2() {
        assert_eq!(dump_vector_of_int(&[1, 2, 3]), "[1, 2, 3]");
        assert_eq!(dump_vector_of_int(&[]), "[]");
        assert_eq!(dump_vector_of_double(&[1.5, 2.0]), "[1.50, 2.00]");
        assert_eq!(dump_vector_of_rect(&[Rect::new(1, 2, 3, 4)]), "[[x=1, y=2, w=3, h=4]]");
        assert_eq!(dump_vector_of_rect(&[]), "[]");
    }


    /// Values checked against `cv2.utils.nested.*` and `cv2.utils.*`.
    #[test]
    fn the_binding_fixtures_match_cv2() {
        assert_eq!(OriginalClassName::original_name(), "OriginalClassName");
        // cv2: OriginalClassName_create().getIntParam() == 123, float == 3.5
        let o = OriginalClassName::create(Params::default());
        assert_eq!(o.get_int_param(), 123);
        assert_eq!(o.get_float_param(), 3.5);
        assert_eq!(OriginalClassName::default().get_int_param(), 123);
        // cv2.utils.ClassWithKeywordProperties(12, 3) -> lambda_=12, except_=3
        let k = ClassWithKeywordProperties::new(12, 3);
        assert_eq!((k.lambda_, k.except_), (12, 3));
        assert_eq!(ClassWithKeywordProperties::default(), ClassWithKeywordProperties::new(24, 42));
        // The chained setters return self, as upstream's do.
        let p = FunctionParams::default().set_lambda(7).set_sigma(1.5);
        assert_eq!((p.lambda_, p.sigma), (7, 1.5));
        assert_eq!(FunctionParams::default().lambda_, -1);
        // ExportClassName is the same type under the name the bindings show.
        let e: ExportClassName = OriginalClassName::create(Params::new(1, 2.0));
        assert_eq!(e.get_int_param(), 1);
    }

    #[test]
    fn echo_returns_its_argument() {
        assert!(test_echo_boolean_function(true));
        assert!(!test_echo_boolean_function(false));
    }

    #[test]
    fn the_cache_directory_honours_the_environment_variable() {
        let s = |x: &str| Some(x.to_string());
        // An explicit override wins outright.
        assert_eq!(cache_directory(s("/tmp/probe"), s("/xdg"), s("/home/u")), "/tmp/probe");
        // An empty override is treated as unset.
        let fallback = cache_directory(s(""), None, s("/home/u"));
        assert!(fallback.ends_with("opencv/downloads"), "{fallback}");
        assert!(fallback.starts_with("/home/u"), "{fallback}");
        // Missing HOME degrades to the working directory rather than failing.
        assert!(cache_directory(None, None, None).starts_with('.'));
        if !cfg!(target_os = "macos") {
            assert_eq!(
                cache_directory(None, s("/xdg"), s("/home/u")),
                "/xdg/opencv/downloads"
            );
        }
        // And the public entry point still answers.
        assert!(get_cache_directory_for_downloads().ends_with("opencv/downloads"));
    }
}
