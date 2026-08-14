//! Collects the upstream symbols this workspace claims to have ported.
//!
//! The claim lives in the doc comments: every ported item names its upstream
//! counterpart in backticks, e.g.
//!
//! ```text
//! /// `cv::resize(src, dst, dsize, fx, fy, interpolation)`.
//! /// `COLOR_BGR2GRAY`.
//! /// `Mat::setTo`
//! ```
//!
//! Keeping the mapping in the doc comment rather than a side table means it
//! cannot drift: a reviewer reading the function sees the same claim the parity
//! report counts.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// Claims grouped by the crate that makes them.
///
/// Grouping matters: `resize`, `merge` and `split` all exist in several
/// upstream modules, so a flat symbol set would credit `gapi` for work done in
/// `imgproc`. Each crate is scored only against the module it ports.
pub type Claims = BTreeMap<String, BTreeSet<String>>;

/// Scan every `.rs` file under `crates/`, grouping claims by crate directory.
pub fn scan(workspace_root: &Path) -> std::io::Result<Claims> {
    let crates_dir = workspace_root.join("crates");
    let mut out: Claims = BTreeMap::new();
    if !crates_dir.is_dir() {
        return Ok(out);
    }

    for entry in fs::read_dir(&crates_dir)? {
        let crate_dir = entry?.path();
        if !crate_dir.is_dir() {
            continue;
        }
        let Some(crate_name) = crate_dir.file_name().and_then(|s| s.to_str()) else { continue };

        let mut files = Vec::new();
        collect_rs(&crate_dir, &mut files)?;
        let set = out.entry(crate_name.to_string()).or_default();
        for f in files {
            let text = fs::read_to_string(&f)?;
            let lines: Vec<&str> = text.lines().map(str::trim).collect();
            // Track whether we are inside the body of a public enum, struct or
            // trait, whose variants, fields and methods are all exported
            // without carrying their own `pub`.
            let mut pub_body_depth: i32 = 0;
            let mut depth: i32 = 0;
            for (i, line) in lines.iter().enumerate() {
                let opens = line.matches('{').count() as i32;
                let closes = line.matches('}').count() as i32;
                if opens > 0
                    && pub_body_depth == 0
                    && (line.starts_with("pub enum ")
                        || line.starts_with("pub struct ")
                        || line.starts_with("pub trait "))
                {
                    pub_body_depth = depth + 1;
                }
                let was_in_pub_body = pub_body_depth > 0 && depth >= pub_body_depth;
                depth += opens - closes;
                if pub_body_depth > 0 && depth < pub_body_depth {
                    pub_body_depth = 0;
                }
                // Only item docs count as claims. `//!` module headers are
                // prose, and prose legitimately names symbols the module does
                // *not* implement — a header explaining "OpenCV has no cosine
                // metric, its norms are `NORM_HAMMING` ..." would otherwise be
                // scored as having ported `NORM_HAMMING`.
                //
                // `#[doc = "..."]` is accepted alongside `///` because it is
                // the same thing spelled differently, and it is the only
                // spelling a `macro_rules!` can emit. Without it, a generated
                // accessor is invisible however well documented it is — which
                // is a property of the generator, not of the port.
                let is_doc_attr = line.starts_with("#[doc = \"") || line.starts_with("#[doc=\"");
                if !line.starts_with("///") && !is_doc_attr {
                    continue;
                }
                // ...and the doc block has to be *attached to something
                // public*. Without this the metric counts documentation rather
                // than implementation: any `///` line mentioning `cv::foo`
                // scores `foo` as ported, including one that exists only to say
                // the function is absent, or that sits on a private helper.
                if !documents_public_item(&lines, i, was_in_pub_body) {
                    continue;
                }
                for name in claims_in(line) {
                    set.insert(name);
                }
            }
        }
    }
    Ok(out)
}

/// Does the doc block containing line `i` sit on a public item?
///
/// Walks forward past the rest of the doc comment and any attributes to the
/// first real line, and asks whether it declares something exported. A claim on
/// a private helper, a test, or a bare comment does not count as ported.
fn documents_public_item(lines: &[&str], i: usize, in_pub_body: bool) -> bool {
    let mut exported_macro = false;
    for line in lines.iter().skip(i) {
        // A plain `//` note between the doc comment and the item is still part
        // of the item's preamble — it does not make the item private. Missing
        // this cost real claims: `Sobel`, `Scharr` and `kmeans` all have an
        // explanatory `//` line under their `///`, and all three read as
        // unported.
        if line.starts_with("///")
            || line.starts_with("//")
            || line.starts_with("#[")
            || line.is_empty()
        {
            // A `macro_rules!` carries no `pub`; `#[macro_export]` is what
            // makes it reachable, so the attribute has to be remembered.
            exported_macro |= line.starts_with("#[macro_export]");
            continue;
        }
        // `pub(crate)` is deliberately excluded: crate-private means the symbol
        // is not reachable by a caller, whatever the docs say.
        if line.starts_with("pub(crate)") {
            return false;
        }
        if line.starts_with("macro_rules!") {
            return exported_macro;
        }
        if line.starts_with("pub ") || line.starts_with("impl ") {
            return true;
        }
        // A variant, field or trait method inside a public enum, struct or
        // trait is exported by its container even though the line itself
        // carries no `pub`.
        return in_pub_body;
    }
    false
}

/// Doc lines that begin a `cv::` claim and never close it, as
/// `(file, line number, text)`.
///
/// A claim has to fit on one line: [`claims_in`] stops at the first unclosed
/// backtick, so wrapping a long signature onto a second `///` line silently
/// scores nothing. That mistake has cost real coverage several times and is
/// invisible in review — the doc comment reads perfectly either way.
///
/// The test is deliberately narrow. An odd backtick count alone is not a bug:
/// wrapped prose does it, and so does every ```` ``` ```` fence. Only a
/// dangling span that *starts with* `cv::` is unambiguously a truncated claim.
#[cfg_attr(not(test), allow(dead_code))]
pub fn unbalanced_doc_lines(workspace_root: &Path) -> std::io::Result<Vec<(String, usize, String)>> {
    let mut out = Vec::new();
    let mut files = Vec::new();
    collect_rs(&workspace_root.join("crates"), &mut files)?;
    for f in files {
        let text = fs::read_to_string(&f)?;
        for (i, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if !line.starts_with("///") || line.matches('`').count() % 2 == 0 {
                continue;
            }
            let Some(at) = line.rfind('`') else { continue };
            if line[at + 1..].trim_start().starts_with("cv::") {
                out.push((f.display().to_string(), i + 1, line.to_string()));
            }
        }
    }
    Ok(out)
}

/// Recursively gather `.rs` files.
fn collect_rs(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_rs(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// Extract upstream symbol names from the backticked spans of one doc line.
pub fn claims_in(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find('`') {
        let before = rest[..start].chars().next_back();
        let after = &rest[start + 1..];
        let Some(end) = after.find('`') else { break };
        let span = &after[..end];
        let tail = &after[end + 1..];
        rest = tail;

        // Skip Rust intra-doc links like [`Mat`] — those reference this port's
        // own items, not upstream symbols.
        if before == Some('[') && tail.starts_with(']') {
            continue;
        }
        if let Some(name) = normalise(span) {
            out.push(name);
        }
    }
    out
}

/// Reduce a backticked span to a bare upstream symbol name, or `None` if it
/// does not look like one.
fn normalise(span: &str) -> Option<String> {
    let span = span.trim();
    // Drop an argument list: `resize(src, dst, ...)` -> `resize`.
    let head = span.split('(').next()?.trim();
    // Drop a qualifier: `cv::resize` / `Mat::setTo` -> `resize` / `setTo`.
    let bare = head.rsplit("::").next()?.trim();

    if bare.is_empty() || !bare.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    if bare.chars().next()?.is_ascii_digit() {
        return None;
    }
    // Short names are noise — `dx`, `fy`, `k` appear in prose constantly. A
    // *qualified* short name is not: nobody writes `cv::FileNode::at` or
    // `cv::detail::CameraParams::K` unless they mean the symbol. Upstream has
    // several such members, and dropping the qualified case to keep the
    // unqualified rule would make every one of them unclaimable however well
    // it were ported.
    let qualified = head.contains("::");
    if bare.len() < 3 && !qualified {
        return None;
    }
    Some(bare.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_a_function_claim() {
        let c = claims_in("/// `cv::resize(src, dst, dsize, fx, fy, interpolation)`.");
        assert_eq!(c, vec!["resize"]);
    }

    #[test]
    fn extracts_a_method_claim() {
        assert_eq!(claims_in("/// `Mat::setTo`"), vec!["setTo"]);
    }

    #[test]
    fn extracts_a_constant_claim() {
        assert_eq!(claims_in("    /// `COLOR_BGR2GRAY`."), vec!["COLOR_BGR2GRAY"]);
    }

    #[test]
    fn extracts_several_from_one_line() {
        let c = claims_in("/// `COLOR_BGR2BGRA` / `COLOR_RGB2RGBA` — add alpha.");
        assert_eq!(c, vec!["COLOR_BGR2BGRA", "COLOR_RGB2RGBA"]);
    }

    #[test]
    fn rejects_prose_and_punctuation() {
        assert!(claims_in("/// `1 << 15`").is_empty());
        assert!(claims_in("/// `a` and `b`").is_empty());
        assert!(claims_in("/// no backticks here").is_empty());
        // An unqualified short name is prose: `dx` names an argument, not a
        // symbol.
        assert!(claims_in("/// scaled by `dx` and `dy`").is_empty());
    }

    #[test]
    fn a_qualified_short_name_is_a_claim() {
        // The qualifier is what distinguishes a symbol from prose.
        assert_eq!(claims_in("/// `cv::FileNode::at(i)`."), vec!["at"]);
        assert_eq!(claims_in("/// `cv::detail::CameraParams::K()`."), vec!["K"]);
        assert!(claims_in("/// `at`").is_empty(), "bare, it is a preposition");
        assert!(claims_in("/// scaled by `k`").is_empty());
    }

    #[test]
    fn handles_an_unclosed_backtick() {
        assert!(claims_in("/// `dangling").is_empty());
    }

    #[test]
    fn module_prose_is_not_a_claim() {
        // The `//!` filter lives in `scan`; verify the rule it applies.
        let prose = "//! OpenCV also defines `NORM_HAMMING`.";
        let item = "/// `cv::resize(src, dst)`.";
        assert!(!prose.trim_start().starts_with("///"), "module prose must be skipped");
        assert!(item.trim_start().starts_with("///"), "item docs must be kept");
        assert_eq!(claims_in(item), vec!["resize"]);
    }

    #[test]
    fn skips_rust_intra_doc_links() {
        assert!(claims_in("/// See [`MatLike`] for details.").is_empty());
        // A real claim on the same line still registers.
        assert_eq!(
            claims_in("/// [`Mat`] implements `cv::Mat::setTo`."),
            vec!["setTo"]
        );
    }

    #[test]
    fn a_plain_comment_does_not_detach_a_doc_claim() {
        // Real shape: `/// claim`, then a `//` note explaining a decision,
        // then an attribute, then the item.
        let lines = vec![
            "/// `cv::Sobel(src, dst, ddepth)`.",
            "// The argument list mirrors upstream's one-for-one.",
            "#[allow(clippy::too_many_arguments)]",
            "pub fn sobel<S>(src: &S) -> Result<Mat> {",
        ];
        assert!(documents_public_item(&lines, 0, false), "the `//` note detached the claim");
    }

    #[test]
    fn an_exported_macro_counts_but_a_private_one_does_not() {
        let exported = vec!["/// `CV_Error`", "#[macro_export]", "macro_rules! cv_error {"];
        assert!(documents_public_item(&exported, 0, false));
        let private = vec!["/// `CV_Error`", "macro_rules! helper {"];
        assert!(!documents_public_item(&private, 0, false));
    }

    #[test]
    fn a_claim_on_a_private_item_still_does_not_count() {
        let lines = vec!["/// `cv::resize(src, dst)`.", "fn helper() {}"];
        assert!(!documents_public_item(&lines, 0, false));
        let crate_private = vec!["/// `cv::resize(src, dst)`.", "pub(crate) fn helper() {}"];
        assert!(!documents_public_item(&crate_private, 0, false));
    }

    #[test]
    fn a_doc_attribute_is_read_like_a_doc_comment() {
        // `#[doc = "..."]` is what a `macro_rules!` emits; it means exactly
        // what `///` means, so a generated accessor must count.
        let lines = vec![
            "#[doc = \"`cv::ORB::getMaxFeatures()`.\"]",
            "pub fn get_max_features(&self) -> i32 {",
        ];
        assert!(documents_public_item(&lines, 0, false));
        assert_eq!(claims_in(lines[0]), vec!["getMaxFeatures"]);
    }

    #[test]
    fn no_doc_line_in_the_workspace_has_an_unclosed_backtick() {
        // The workspace root is two levels up from `xtask/src`.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let bad = unbalanced_doc_lines(&root).expect("the crates tree should be readable");
        assert!(
            bad.is_empty(),
            "these doc lines wrap a backticked span, so the claim on them is not counted:\n{}",
            bad.iter()
                .map(|(f, n, l)| format!("  {f}:{n}: {l}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}
