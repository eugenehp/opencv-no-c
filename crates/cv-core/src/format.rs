//! `cv::format` and `cv::Formatter` — rendering a matrix as text
//! (`modules/core/src/out.cpp`).
//!
//! This is what `std::cout << mat` produces in C++, and it is worth having
//! byte-for-byte: the `FMT_PYTHON` and `FMT_NUMPY` outputs are meant to be
//! pasted straight into a Python session, and `FMT_CSV` into a spreadsheet.
//! Anything close-but-different would be worse than useless there.
//!
//! The renderer below is a direct transcription of upstream's `FormattedImpl`
//! state machine rather than a rewrite. The machine looks convoluted for what
//! it does, but the layout details it encodes — where the indent goes, when a
//! trailing comma appears, that `FMT_MATLAB` walks channels outermost — are
//! exactly the parts a rewrite would get subtly wrong.

use crate::error::Result;
use crate::mat::MatLike;
use crate::types::Depth;

/// `cv::Formatter::FormatType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum FormatType {
    /// `FMT_DEFAULT` — OpenCV's own `[a, b;\n c, d]` layout.
    #[default]
    Default = 0,
    /// `FMT_MATLAB` — channels printed as separate `(:, :, k)` planes.
    Matlab = 1,
    /// `FMT_CSV` — bare values, one row per line.
    Csv = 2,
    /// `FMT_PYTHON` — nested lists.
    Python = 3,
    /// `FMT_NUMPY` — an `array([...], dtype='...')` literal.
    Numpy = 4,
    /// `FMT_C` — a brace-enclosed C initialiser.
    C = 5,
}

/// Which brace goes where; the five slots of upstream's `braces[5]`.
#[derive(Clone, Copy)]
struct Braces {
    row_open: char,
    row_close: char,
    row_sep: char,
    cn_open: char,
    cn_close: char,
}

const NONE: char = '\0';

/// `cv::format(mtx, fmt)` — render `src` in the given style.
///
/// `multiline` is `Formatter::setMultiline`: with it off, rows are separated by
/// a space instead of a newline. A one-row matrix is always on one line, as
/// upstream forces.
pub fn format<S: MatLike + ?Sized>(src: &S, fmt: FormatType, multiline: bool) -> Result<String> {
    // Upstream's per-depth precision defaults, from `FormatterBase`.
    let precision = match src.depth() {
        Depth::F64 => 16,
        Depth::F16 => 4,
        _ => 8,
    };
    let single_line = src.rows() == 1 || !multiline;
    let cols_is_one = src.cols() == 1;

    let (prologue, epilogue, braces, align_order) = match fmt {
        FormatType::Default => (
            "[".to_string(),
            "]".to_string(),
            Braces { row_open: NONE, row_close: NONE, row_sep: ';', cn_open: NONE, cn_close: NONE },
            false,
        ),
        FormatType::Matlab => (
            String::new(),
            String::new(),
            Braces { row_open: NONE, row_close: NONE, row_sep: ';', cn_open: NONE, cn_close: NONE },
            true,
        ),
        FormatType::Csv => (
            String::new(),
            // Upstream ends a multi-row CSV with a newline and a single-row one
            // without, so a one-row matrix pastes inline.
            if src.rows() > 1 { "\n".to_string() } else { String::new() },
            Braces {
                row_open: NONE,
                row_close: NONE,
                row_sep: NONE,
                cn_open: NONE,
                cn_close: NONE,
            },
            false,
        ),
        FormatType::Python => (
            "[".to_string(),
            "]".to_string(),
            Braces {
                // A single column needs no inner brackets: each row is one value.
                row_open: if cols_is_one { NONE } else { '[' },
                row_close: if cols_is_one { NONE } else { ']' },
                row_sep: ',',
                cn_open: '[',
                cn_close: ']',
            },
            false,
        ),
        FormatType::Numpy => (
            "array([".to_string(),
            format!("], dtype='{}')", numpy_type(src.depth())),
            Braces {
                row_open: if cols_is_one { NONE } else { '[' },
                row_close: if cols_is_one { NONE } else { ']' },
                row_sep: ',',
                cn_open: '[',
                cn_close: ']',
            },
            false,
        ),
        FormatType::C => (
            "{".to_string(),
            "}".to_string(),
            Braces { row_open: NONE, row_close: NONE, row_sep: ',', cn_open: NONE, cn_close: NONE },
            false,
        ),
    };

    Renderer {
        src,
        prologue,
        epilogue,
        braces,
        single_line,
        align_order,
        precision,
        out: String::new(),
        row: 0,
        col: 0,
        cn: 0,
    }
    .run()
}

/// The dtype names upstream's `NumpyFormatter` uses, indexed by depth.
fn numpy_type(d: Depth) -> &'static str {
    match d {
        Depth::U8 => "uint8",
        Depth::S8 => "int8",
        Depth::U16 => "uint16",
        Depth::S16 => "int16",
        Depth::S32 => "int32",
        Depth::F32 => "float32",
        Depth::F64 => "float64",
        Depth::F16 => "float16",
    }
}

struct Renderer<'a, S: MatLike + ?Sized> {
    src: &'a S,
    prologue: String,
    epilogue: String,
    braces: Braces,
    single_line: bool,
    align_order: bool,
    precision: usize,
    out: String,
    row: i32,
    col: i32,
    cn: i32,
}

/// Upstream's `FormattedImpl` states.
#[derive(Clone, Copy, PartialEq)]
enum State {
    Prologue,
    Epilogue,
    Interlude,
    RowOpen,
    RowClose,
    CnOpen,
    CnClose,
    Value,
    Finished,
    LineSeparator,
    CnSeparator,
    ValueSeparator,
}

impl<S: MatLike + ?Sized> Renderer<'_, S> {
    fn run(mut self) -> Result<String> {
        let mut state = State::Prologue;
        // Each pass appends at most a few characters, so the bound is generous;
        // it exists only so a transcription slip cannot hang a caller.
        let limit = 64 * (self.src.total() * self.src.channels() as usize + 8);
        for _ in 0..limit {
            if state == State::Finished {
                return Ok(self.out);
            }
            state = self.step(state)?;
        }
        Ok(self.out)
    }

    #[allow(clippy::too_many_lines)]
    fn step(&mut self, state: State) -> Result<State> {
        let (rows, cols, mcn) = (self.src.rows(), self.src.cols(), self.src.channels());
        Ok(match state {
            State::Prologue => {
                self.row = 0;
                self.out.push_str(&self.prologue);
                if self.src.is_empty() {
                    State::Epilogue
                } else if self.align_order {
                    State::Interlude
                } else {
                    State::RowOpen
                }
            }
            State::Interlude => {
                if self.row >= rows {
                    self.cn += 1;
                    if self.cn >= mcn {
                        return Ok(State::Epilogue);
                    }
                    self.row = 0;
                    self.out.push_str(&format!("\n(:, :, {}) = \n", self.cn + 1));
                    return Ok(State::RowOpen);
                }
                self.out.push_str(&format!("(:, :, {}) = \n", self.cn + 1));
                State::RowOpen
            }
            State::Epilogue => {
                self.out.push_str(&self.epilogue);
                State::Finished
            }
            State::RowOpen => {
                self.col = 0;
                // Continuation rows line up under the prologue.
                if self.row > 0 {
                    for _ in 0..self.prologue.len() {
                        self.out.push(' ');
                    }
                }
                if self.braces.row_open != NONE {
                    self.out.push(self.braces.row_open);
                }
                State::CnOpen
            }
            State::RowClose => {
                self.row += 1;
                if self.braces.row_close != NONE {
                    self.out.push(self.braces.row_close);
                    if self.row < rows {
                        self.out.push(',');
                    }
                } else if self.braces.row_sep != NONE && self.row < rows {
                    self.out.push(self.braces.row_sep);
                }
                State::LineSeparator
            }
            State::CnOpen => {
                if !self.align_order {
                    self.cn = 0;
                }
                if mcn > 1 && self.braces.cn_open != NONE {
                    self.out.push(self.braces.cn_open);
                }
                State::Value
            }
            State::CnClose => {
                self.col += 1;
                let next =
                    if self.col >= cols { State::RowClose } else { State::CnSeparator };
                if mcn > 1 && self.braces.cn_close != NONE {
                    self.out.push(self.braces.cn_close);
                }
                next
            }
            State::Value => {
                let v = self.src.at_f64(self.row, self.col, self.cn)?;
                self.out.push_str(&self.value_to_str(v));
                if self.align_order {
                    return Ok(State::CnClose);
                }
                self.cn += 1;
                if self.cn < mcn {
                    State::ValueSeparator
                } else {
                    State::CnClose
                }
            }
            State::LineSeparator => {
                if self.row >= rows {
                    return Ok(if self.align_order { State::Interlude } else { State::Epilogue });
                }
                self.out.push(if self.single_line { ' ' } else { '\n' });
                State::RowOpen
            }
            State::CnSeparator => {
                self.out.push_str(", ");
                State::CnOpen
            }
            State::ValueSeparator => {
                self.out.push_str(", ");
                State::Value
            }
            State::Finished => State::Finished,
        })
    }

    /// One element, in the `printf` form upstream picks for this depth.
    fn value_to_str(&self, v: f64) -> String {
        match self.src.depth() {
            // `%3d`: right-aligned in three columns, so byte matrices line up.
            Depth::U8 | Depth::S8 => format!("{:3}", v as i64),
            Depth::U16 | Depth::S16 | Depth::S32 => format!("{}", v as i64),
            Depth::F32 | Depth::F64 | Depth::F16 => format_g(v, self.precision),
        }
    }
}

/// C's `%.Pg` conversion.
///
/// Rust has no `%g`, and the rule it encodes is not obvious: `%e` is used when
/// the decimal exponent falls outside `-4..P`, `%f` otherwise, and either way
/// trailing zeros — and a bare trailing point — are stripped. Getting this
/// wrong shows up as `1.0` where OpenCV writes `1`, on every float it prints.
fn format_g(v: f64, precision: usize) -> String {
    let p = precision.clamp(1, 20);
    if v.is_nan() {
        return if v.is_sign_negative() { "-nan".into() } else { "nan".into() };
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }

    // The exponent %e would use, taken from a rounded rendering so that a value
    // like 9.9999 at precision 3 counts as 1e1 rather than 1e0.
    let e_form = format!("{:.*e}", p - 1, v);
    let exp: i32 = e_form.rsplit('e').next().and_then(|s| s.parse().ok()).unwrap_or(0);

    if exp < -4 || exp >= p as i32 {
        let mantissa = e_form.split('e').next().unwrap_or("0");
        // C writes the exponent with a sign and at least two digits.
        format!("{}e{}{:02}", strip_zeros(mantissa), if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        strip_zeros(&format!("{v:.decimals$}"))
    }
}

/// Drop trailing zeros in a fractional part, and the point if nothing remains.
fn strip_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::Mat;

    fn u8m() -> Mat {
        Mat::from_vec(2, 3, 1, &[1u8, 2, 3, 4, 5, 6]).unwrap()
    }

    /// Reference strings are what `std::cout << mat` prints for each formatter.
    #[test]
    fn default_format_matches_the_cpp_stream_operator() {
        assert_eq!(format(&u8m(), FormatType::Default, true).unwrap(), "[  1,   2,   3;\n   4,   5,   6]");
        // Without multiline the rows join with a space — and the continuation
        // indent is still emitted after it, so the gap is one wider than above.
        assert_eq!(format(&u8m(), FormatType::Default, false).unwrap(), "[  1,   2,   3;    4,   5,   6]");
    }

    #[test]
    fn python_and_numpy_produce_pasteable_literals() {
        assert_eq!(
            format(&u8m(), FormatType::Python, true).unwrap(),
            "[[  1,   2,   3],\n [  4,   5,   6]]"
        );
        assert_eq!(
            format(&u8m(), FormatType::Numpy, true).unwrap(),
            "array([[  1,   2,   3],\n       [  4,   5,   6]], dtype='uint8')"
        );
    }

    #[test]
    fn csv_and_c_drop_the_row_brackets() {
        assert_eq!(format(&u8m(), FormatType::Csv, true).unwrap(), "  1,   2,   3\n  4,   5,   6\n");
        assert_eq!(format(&u8m(), FormatType::C, true).unwrap(), "{  1,   2,   3,\n   4,   5,   6}");
    }

    #[test]
    fn matlab_walks_channels_outermost() {
        let m = Mat::from_vec(1, 2, 2, &[1u8, 2, 3, 4]).unwrap();
        let s = format(&m, FormatType::Matlab, true).unwrap();
        // Each channel becomes its own plane, numbered from one.
        assert!(s.starts_with("(:, :, 1) = \n"), "got {s:?}");
        assert!(s.contains("(:, :, 2) = \n"), "got {s:?}");
        // Channel 1 holds 1 and 3; channel 2 holds 2 and 4.
        let planes: Vec<&str> = s.split("(:, :, ").collect();
        assert!(planes[1].contains("  1,   3"), "got {s:?}");
        assert!(planes[2].contains("  2,   4"), "got {s:?}");
    }

    #[test]
    fn a_column_vector_loses_its_inner_brackets() {
        let m = Mat::from_vec(3, 1, 1, &[1u8, 2, 3]).unwrap();
        assert_eq!(format(&m, FormatType::Python, true).unwrap(), "[  1,\n   2,\n   3]");
    }

    #[test]
    fn multichannel_values_group_per_pixel() {
        let m = Mat::from_vec(1, 2, 3, &[1u8, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(
            format(&m, FormatType::Default, true).unwrap(),
            "[  1,   2,   3,   4,   5,   6]"
        );
        // The Python formatter brackets each pixel.
        assert_eq!(
            format(&m, FormatType::Python, true).unwrap(),
            "[[[  1,   2,   3], [  4,   5,   6]]]"
        );
    }

    #[test]
    fn an_empty_matrix_is_just_the_delimiters() {
        let m = Mat::empty(crate::types::CV_8UC1);
        assert_eq!(format(&m, FormatType::Default, true).unwrap(), "[]");
        assert_eq!(format(&m, FormatType::Numpy, true).unwrap(), "array([], dtype='uint8')");
    }

    /// Values from Python's `'%.*g'`, which follows the same C rule.
    #[test]
    fn format_g_matches_c() {
        let cases: &[(f64, usize, &str)] = &[
            (1.0, 8, "1"),
            (0.5, 8, "0.5"),
            (-3.25, 8, "-3.25"),
            (0.0, 8, "0"),
            (100.0, 8, "100"),
            (1.0 / 3.0, 8, "0.33333333"),
            (1.0 / 3.0, 16, "0.3333333333333333"),
            (1e-5, 8, "1e-05"),
            (1e8, 8, "1e+08"),
            (12345678.0, 8, "12345678"),
            (123456789.0, 8, "1.2345679e+08"),
            (0.0001, 8, "0.0001"),
            (1.5e300, 16, "1.5e+300"),
            (9.9999, 3, "10"),
        ];
        for (v, p, want) in cases {
            assert_eq!(format_g(*v, *p), *want, "%.{p}g of {v}");
        }
        assert_eq!(format_g(f64::INFINITY, 8), "inf");
        assert_eq!(format_g(f64::NAN, 8), "nan");
    }

    #[test]
    fn floats_use_the_depth_appropriate_precision() {
        let f = Mat::from_vec(1, 2, 1, &[1.0f32 / 3.0, 2.5]).unwrap();
        assert_eq!(format(&f, FormatType::Default, true).unwrap(), "[0.33333334, 2.5]");
        let d = Mat::from_vec(1, 2, 1, &[1.0f64 / 3.0, 2.5]).unwrap();
        assert_eq!(format(&d, FormatType::Default, true).unwrap(), "[0.3333333333333333, 2.5]");
    }
}
