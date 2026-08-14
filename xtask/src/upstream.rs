//! Extracts the public API surface from the upstream OpenCV headers.
//!
//! This is deliberately a *heuristic* scanner, not a C++ parser. It keys off
//! the same export macros OpenCV's own Python/Java binding generators use
//! (`CV_EXPORTS_W`, `CV_WRAP`, `CV_EXPORTS_AS`), so the symbol set it produces
//! is close to what those bindings expose. Getting the denominator slightly
//! wrong is acceptable; silently dropping whole modules is not, so
//! [`scan_module`] reports how many declarations it could not classify.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// What kind of declaration a symbol came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// A free function or method exposed to the bindings.
    Function,
    /// A class or struct exposed to the bindings.
    Class,
    /// A named enum constant.
    EnumConst,
    /// A `CV_*` preprocessor macro, such as `CV_8U` or `CV_MAKETYPE`.
    Macro,
}

impl Kind {
    /// Lowercase tag used in the JSON manifest.
    pub fn tag(self) -> &'static str {
        match self {
            Kind::Function => "function",
            Kind::Class => "class",
            Kind::EnumConst => "enum",
            Kind::Macro => "macro",
        }
    }
}

/// One upstream symbol.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Symbol {
    /// Kind of declaration.
    pub kind: Kind,
    /// Bare symbol name, e.g. `resize` or `COLOR_BGR2GRAY`.
    pub name: String,
}

/// Every exported symbol of one OpenCV module.
#[derive(Debug)]
pub struct Module {
    /// Module directory name, e.g. `imgproc`.
    pub name: String,
    /// Sorted, de-duplicated symbols.
    pub symbols: Vec<Symbol>,
    /// Header lines carrying an export macro that could not be classified.
    pub unparsed: usize,
}

/// Scan `<opencv>/modules/<name>/include` for exported symbols.
pub fn scan_module(opencv_root: &Path, name: &str) -> std::io::Result<Module> {
    let include = opencv_root.join("modules").join(name).join("include");
    let mut set = BTreeSet::new();
    let mut unparsed = 0usize;

    let mut headers = Vec::new();
    collect_headers(&include, &mut headers)?;
    for h in headers {
        let text = match fs::read_to_string(&h) {
            Ok(t) => t,
            // Some headers are latin-1; skipping them loses a handful of
            // symbols, which is preferable to aborting the whole scan.
            Err(_) => continue,
        };
        scan_text(&text, &mut set, &mut unparsed);
    }

    Ok(Module { name: name.to_string(), symbols: set.into_iter().collect(), unparsed })
}

/// Recursively gather `.hpp` / `.h` files under `dir`.
fn collect_headers(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_headers(&path, out)?;
        } else {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            // `*_c.h` and `types_c.h` are the frozen C API, which this port
            // does not target; they would inflate the denominator with symbols
            // OpenCV itself considers deprecated.
            let stem = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if (ext == "hpp" || ext == "h") && !stem.ends_with("_c.h") {
                out.push(path);
            }
        }
    }
    Ok(())
}

/// Pull symbols out of one header's text.
fn scan_text(text: &str, out: &mut BTreeSet<Symbol>, unparsed: &mut usize) {
    let mut in_enum = false;
    let mut brace_depth = 0i32;
    let mut in_block_comment = false;

    for raw in text.lines() {
        // Doxygen blocks are prose, and prose is full of capitalised words that
        // look like enum constants. Without this, a sentence beginning "The
        // covariance matrix..." inside an enum's `/** */` contributes a symbol
        // named `The`. Four such phantoms were in the denominator before.
        let raw = strip_block_comment(raw, &mut in_block_comment);
        let line = strip_comment(&raw).trim().to_string();
        if line.is_empty() {
            continue;
        }

        // Track enum bodies so constant names can be harvested.
        if !in_enum && opens_enum(&line) {
            in_enum = true;
            brace_depth = 0;
        }
        if in_enum {
            let opens = line.matches('{').count() as i32;
            brace_depth += opens;
            for tok in enum_constants(&line) {
                out.insert(Symbol { kind: Kind::EnumConst, name: tok });
            }
            brace_depth -= line.matches('}').count() as i32;
            if brace_depth <= 0 && line.contains('}') {
                in_enum = false;
            }
            continue;
        }

        // Type codes and helpers like CV_8U / CV_MAKETYPE are `#define`s, not
        // enum members, but they are unquestionably part of the public surface.
        //
        // The directive is normalised first because a header may indent it
        // inside an `#if`: `#           define GAPI_EXPORTS __declspec(...)`
        // does not start with `#define `, so it fell through to the function
        // parser, which read `__declspec` as an exported symbol.
        let directive = line
            .strip_prefix('#')
            .map(|r| format!("#{}", r.trim_start()))
            .unwrap_or_else(|| line.clone());
        if directive.starts_with('#') {
            if let Some(rest) = directive.strip_prefix("#define ") {
                let name: String =
                    rest.trim().chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                if name.starts_with("CV_") && is_ident(&name) && !name.ends_with('_') {
                    out.insert(Symbol { kind: Kind::Macro, name });
                }
            }
            continue;
        }
        // G-API declares its kernels with its own macro. Missing it understated
        // both sides of the gapi score: none of the ~200 kernel functions were
        // counted in the denominator, and a port claiming `cv::gapi::blur`
        // matched nothing.
        let has_export = line.contains("CV_EXPORTS_W")
            || line.contains("CV_EXPORTS_AS")
            || line.contains("GAPI_EXPORTS_W")
            || line.contains("GAPI_EXPORTS")
            || line.starts_with("CV_WRAP")
            || line.contains(" CV_WRAP ");
        if !has_export {
            continue;
        }

        // `CV_EXPORTS_AS(newName)` and `CV_WRAP_AS(newName)` rename the
        // binding; that name wins. Without this the *macro* is read as the
        // function name — `CV_WRAP_AS(compute) void compute2(...)` scored a
        // symbol called `CV_WRAP_AS`, which nothing could ever port.
        for macro_name in ["CV_EXPORTS_AS(", "CV_WRAP_AS("] {
            if let Some(alias) = between(&line, macro_name, ")") {
                let alias = alias.trim();
                if is_ident(alias) {
                    out.insert(Symbol { kind: Kind::Function, name: alias.to_string() });
                    break;
                }
            }
        }
        if line.contains("CV_EXPORTS_AS(") || line.contains("CV_WRAP_AS(") {
            continue;
        }

        // A class/struct declaration has no argument list on its own line.
        if line.starts_with("class ") || line.starts_with("struct ") {
            if let Some(n) = class_name(&line) {
                out.insert(Symbol { kind: Kind::Class, name: n });
            } else {
                *unparsed += 1;
            }
            continue;
        }

        match function_name(&line) {
            Some(n) => {
                out.insert(Symbol { kind: Kind::Function, name: n });
            }
            None => *unparsed += 1,
        }
    }
}

/// Does this line begin an enum **definition** whose body follows?
///
/// A line can mention `enum` without opening one — `initWideAngleProjMap` takes
/// an `enum UndistortTypes projType = ...` parameter. Latching onto that leaves
/// the scanner inside an enum body for the rest of the header, and every
/// capitalised token afterwards becomes a phantom constant: `InputArray`,
/// `Size`, `imageSize` and `destImageWidth` were all in the calib3d
/// denominator, unportable by construction.
///
/// A real definition either opens its body on the same line, or is a bare
/// `enum Name` / `enum class Name : type` with the brace on the next — neither
/// of which carries a `;`, a `(` or an `=`.
fn opens_enum(line: &str) -> bool {
    if !(line.starts_with("enum ") || line == "enum" || line.starts_with("enum{")) {
        return false;
    }
    line.contains('{') || !(line.contains(';') || line.contains('(') || line.contains('='))
}

/// Remove a trailing `//` comment, leaving string literals alone (headers in
/// this codebase never put `//` inside a literal on a declaration line).
fn strip_comment(line: &str) -> &str {
    match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    }
}

/// Remove any `/* ... */` span from one line, carrying the open state across
/// lines in `inside`.
///
/// Returns only the code outside the comment. A line that both opens and
/// closes a comment keeps the text on either side, so
/// `int x; /* note */ int y;` still yields both declarations.
fn strip_block_comment(line: &str, inside: &mut bool) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    loop {
        if *inside {
            match rest.find("*/") {
                Some(i) => {
                    rest = &rest[i + 2..];
                    *inside = false;
                }
                None => return out,
            }
        } else {
            match rest.find("/*") {
                Some(i) => {
                    out.push_str(&rest[..i]);
                    rest = &rest[i + 2..];
                    *inside = true;
                }
                None => {
                    out.push_str(rest);
                    return out;
                }
            }
        }
    }
}

/// Extract `NAME` from `NAME = value,` inside an enum body.
fn enum_constants(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    for part in line.split(',') {
        let part = part.trim().trim_start_matches('{').trim();
        // Take the leading identifier, whether or not `= value` follows.
        let name: String = part.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        if name.len() > 2 && is_ident(&name) && name.chars().any(|c| c.is_ascii_uppercase()) {
            out.push(name);
        }
    }
    out
}

/// Name of a class from `class CV_EXPORTS_W Foo : public Bar`.
fn class_name(line: &str) -> Option<String> {
    let mut toks = line.split_whitespace();
    // Skip `class`/`struct`, then any export macro, and take the next token.
    toks.next()?;
    for t in toks {
        if t.starts_with("CV_EXPORTS") || t.starts_with("GAPI_EXPORTS") || t.starts_with("CV_WRAP") {
            continue;
        }
        let name: String = t.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        if is_ident(&name) {
            return Some(name);
        }
        return None;
    }
    None
}

/// Name of a function: the identifier immediately preceding the first `(`.
fn function_name(line: &str) -> Option<String> {
    let open = line.find('(')?;
    let head = &line[..open];
    let name: String =
        head.chars().rev().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
    let name: String = name.chars().rev().collect();
    if is_ident(&name) && !is_cxx_keyword(&name) {
        Some(name)
    } else {
        None
    }
}

/// Text between two delimiters, if both are present.
fn between<'a>(s: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let i = s.find(start)? + start.len();
    let j = s[i..].find(end)? + i;
    Some(&s[i..j])
}

/// Whether `s` is a plausible C++ identifier.
fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && !s.chars().next().unwrap().is_ascii_digit()
        && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Keywords that can sit before `(` and would otherwise look like a name.
fn is_cxx_keyword(s: &str) -> bool {
    matches!(
        s,
        "if" | "for" | "while" | "switch" | "return" | "sizeof" | "operator" | "explicit"
            | "static" | "inline" | "virtual" | "const" | "typedef" | "template" | "class"
            | "struct" | "enum" | "namespace" | "using" | "public" | "private" | "protected"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(text: &str) -> Vec<Symbol> {
        let mut set = BTreeSet::new();
        let mut n = 0;
        scan_text(text, &mut set, &mut n);
        set.into_iter().collect()
    }

    #[test]
    fn extracts_a_free_function() {
        let s = scan("CV_EXPORTS_W void resize( InputArray src, OutputArray dst, Size dsize );");
        assert_eq!(s, vec![Symbol { kind: Kind::Function, name: "resize".into() }]);
    }

    #[test]
    fn extracts_a_class() {
        let s = scan("class CV_EXPORTS_W CLAHE : public Algorithm");
        assert_eq!(s, vec![Symbol { kind: Kind::Class, name: "CLAHE".into() }]);
    }

    #[test]
    fn honours_the_binding_alias() {
        let s = scan("CV_EXPORTS_AS(integral3) void integral( InputArray src );");
        assert_eq!(s, vec![Symbol { kind: Kind::Function, name: "integral3".into() }]);
    }

    #[test]
    fn extracts_enum_constants() {
        let s = scan("enum InterpolationFlags {\n INTER_NEAREST = 0,\n INTER_LINEAR = 1,\n};");
        let names: Vec<_> = s.iter().map(|x| x.name.as_str()).collect();
        assert!(names.contains(&"INTER_NEAREST"));
        assert!(names.contains(&"INTER_LINEAR"));
        assert!(s.iter().all(|x| x.kind == Kind::EnumConst));
    }

    #[test]
    fn ignores_unexported_declarations() {
        assert!(scan("void internalHelper(int x);").is_empty());
    }

    #[test]
    fn strips_trailing_comments() {
        let s = scan("CV_EXPORTS_W void blur(InputArray src); // CV_EXPORTS_W not_a_symbol(");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].name, "blur");
    }

    #[test]
    fn does_not_mistake_keywords_for_names() {
        assert!(function_name("CV_EXPORTS_W void f() { if (x) {").is_some());
        assert_eq!(function_name("    if (x) {"), None);
    }

    #[test]
    fn extracts_cv_wrap_methods() {
        let s = scan("    CV_WRAP virtual void apply(InputArray src, OutputArray dst) = 0;");
        assert_eq!(s, vec![Symbol { kind: Kind::Function, name: "apply".into() }]);
    }

    #[test]
    fn gapi_kernels_are_extracted() {
        // G-API uses its own export macro. Before this was recognised, none of
        // the ~200 kernel functions counted, so the gapi denominator was 144
        // instead of 372 and every `cv::gapi::` claim matched nothing.
        let s = scan("GAPI_EXPORTS_W GMat blur(const GMat& src, const Size& ksize);");
        assert!(s.iter().any(|x| x.name == "blur"), "got {s:?}");
        let t = scan("GAPI_EXPORTS GMat BGR2Gray(const GMat& src);");
        assert!(t.iter().any(|x| x.name == "BGR2Gray"), "got {t:?}");
    }

    #[test]
    fn doxygen_prose_is_not_harvested_as_enum_constants() {
        // Real headers put a `/** @brief ... */` block inside an enum body.
        // Every capitalised word in that prose used to become a symbol.
        let src = "enum Foo {\n\
                   /** @brief The covariance matrix will be nsamples x nsamples.\n\
                    Similar to the scrambled form. */\n\
                   REAL_ONE = 1,\n\
                   REAL_TWO = 2\n\
                   };";
        let s = scan(src);
        let names: Vec<&str> = s.iter().map(|x| x.name.as_str()).collect();
        assert!(names.contains(&"REAL_ONE") && names.contains(&"REAL_TWO"), "got {names:?}");
        assert!(!names.contains(&"The"), "prose leaked in: {names:?}");
        assert!(!names.contains(&"Similar"), "prose leaked in: {names:?}");
    }

    #[test]
    fn a_block_comment_does_not_swallow_the_code_around_it() {
        let mut inside = false;
        assert_eq!(strip_block_comment("int x; /* note */ int y;", &mut inside), "int x;  int y;");
        assert!(!inside);
        // An unterminated block keeps the state for the next line.
        assert_eq!(strip_block_comment("code /* open", &mut inside), "code ");
        assert!(inside);
        assert_eq!(strip_block_comment("still inside", &mut inside), "");
        assert_eq!(strip_block_comment("done */ tail", &mut inside), " tail");
        assert!(!inside);
    }

    #[test]
    fn an_enum_parameter_does_not_open_an_enum_body() {
        // A real definition, either brace-on-line or brace-on-next.
        assert!(opens_enum("enum ScoreMethod {SCORE_METHOD_RANSAC=0};"));
        assert!(opens_enum("enum class Foo : int"));
        assert!(opens_enum("enum"));
        // A parameter of enum type is not a definition.
        assert!(!opens_enum(
            "enum UndistortTypes projType = PROJ_SPHERICAL_EQRECT, double alpha = 0);"
        ));
        assert!(!opens_enum("enum Foo bar(int x);"));
        assert!(!opens_enum("enumerate the options"));
    }
}
