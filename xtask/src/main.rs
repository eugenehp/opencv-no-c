//! Parity tooling for the pure-Rust OpenCV port.
//!
//! ```text
//! cargo xtask extract [--opencv <path>]   # write parity/manifest.json
//! cargo xtask report  [--opencv <path>]   # print per-module parity
//! ```
//!
//! The point of this tool is to make "100% parity" a number that moves rather
//! than a claim. `extract` snapshots the upstream API surface; `report` diffs
//! it against the symbols this workspace's doc comments claim to implement.

mod ported;
mod upstream;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use upstream::Kind;

/// Which workspace crate ports which upstream module.
///
/// A crate is scored only against its own module. Without this, symbols whose
/// names appear in several modules (`resize`, `split`, `merge`, …) would credit
/// modules where nothing has been written.
fn crate_for(module: &str) -> String {
    format!("cv-{module}")
}

/// Modules of the upstream tree, in dependency order.
const MODULES: &[&str] = &[
    "core",
    "imgproc",
    "imgcodecs",
    "videoio",
    "highgui",
    "video",
    "calib3d",
    "features2d",
    "objdetect",
    "dnn",
    "ml",
    "flann",
    "photo",
    "stitching",
    "gapi",
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("report");
    let opencv = flag(&args, "--opencv")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("..").join("opencv"));

    if !opencv.join("modules").is_dir() {
        eprintln!(
            "error: {} does not look like an OpenCV checkout (no modules/ directory)",
            opencv.display()
        );
        return ExitCode::FAILURE;
    }

    let result = match cmd {
        "extract" => extract(&opencv),
        "report" => report(&opencv),
        "missing" => {
            let module = args.get(1).filter(|a| !a.starts_with("--")).cloned();
            missing(&opencv, module.as_deref())
        }
        other => {
            eprintln!("error: unknown command {other:?}; expected `extract`, `report` or `missing`");
            return ExitCode::FAILURE;
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Value of `--name <value>` from the argument list.
fn flag(args: &[String], name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).cloned()
}

/// Directory containing this workspace's `Cargo.toml`.
fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR points at xtask/; the workspace is its parent.
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

/// Write `parity/manifest.json` describing the upstream API surface.
fn extract(opencv: &Path) -> std::io::Result<()> {
    let root = workspace_root();
    let out_dir = root.join("parity");
    std::fs::create_dir_all(&out_dir)?;

    let mut json = String::from("{\n  \"modules\": {\n");
    let mut first = true;
    let mut grand_total = 0usize;

    for m in MODULES {
        let module = upstream::scan_module(opencv, m)?;
        grand_total += module.symbols.len();
        if !first {
            json.push_str(",\n");
        }
        first = false;
        json.push_str(&format!("    \"{}\": [\n", module.name));
        for (i, s) in module.symbols.iter().enumerate() {
            if i > 0 {
                json.push_str(",\n");
            }
            json.push_str(&format!(
                "      {{\"kind\": \"{}\", \"name\": \"{}\"}}",
                s.kind.tag(),
                s.name
            ));
        }
        json.push_str("\n    ]");
        eprintln!("{:<12} {:>6} symbols ({} unparsed)", module.name, module.symbols.len(), module.unparsed);
    }
    json.push_str("\n  }\n}\n");

    let path = out_dir.join("manifest.json");
    std::fs::write(&path, json)?;
    eprintln!("\nwrote {} ({grand_total} symbols)", path.display());
    Ok(())
}

/// Print a per-module parity table.
fn report(opencv: &Path) -> std::io::Result<()> {
    let root = workspace_root();
    let claimed = ported::scan(&root)?;

    println!("Parity vs OpenCV {} @ {}", cv_version(opencv), &cv_commit(opencv)[..12.min(cv_commit(opencv).len())]);
    println!();
    println!("{:<12} {:>8} {:>8} {:>8}   bar", "module", "ported", "total", "parity");
    println!("{}", "-".repeat(60));

    let mut total_done = 0usize;
    let mut total_all = 0usize;

    let empty = BTreeSet::new();
    for m in MODULES {
        let module = upstream::scan_module(opencv, m)?;
        let names: BTreeSet<&str> = module.symbols.iter().map(|s| s.name.as_str()).collect();
        let mine = claimed.get(&crate_for(m)).unwrap_or(&empty);
        let done = names.iter().filter(|n| mine.contains(**n)).count();
        let all = names.len();
        total_done += done;
        total_all += all;

        let pct = if all == 0 { 0.0 } else { done as f64 * 100.0 / all as f64 };
        println!(
            "{:<12} {:>8} {:>8} {:>7.2}%   {}",
            module.name,
            done,
            all,
            pct,
            bar(pct)
        );
    }

    println!("{}", "-".repeat(60));
    let pct = if total_all == 0 { 0.0 } else { total_done as f64 * 100.0 / total_all as f64 };
    println!("{:<12} {:>8} {:>8} {:>7.2}%   {}", "TOTAL", total_done, total_all, pct, bar(pct));

    // A breakdown of what is claimed but not matched usually means a doc
    // comment names something the scanner did not find upstream — worth seeing.
    let upstream_all: BTreeSet<String> = MODULES
        .iter()
        .filter_map(|m| upstream::scan_module(opencv, m).ok())
        .flat_map(|m| m.symbols.into_iter().map(|s| s.name))
        .collect();
    let unmatched: Vec<&String> = claimed
        .values()
        .flatten()
        .filter(|c| !upstream_all.contains(*c))
        .collect();
    if !unmatched.is_empty() {
        println!("\n{} claimed symbol(s) not found upstream:", unmatched.len());
        for u in unmatched.iter().take(20) {
            println!("  {u}");
        }
        if unmatched.len() > 20 {
            println!("  ... and {} more", unmatched.len() - 20);
        }
    }

    let classes = MODULES
        .iter()
        .filter_map(|m| upstream::scan_module(opencv, m).ok())
        .flat_map(|m| m.symbols)
        .filter(|s| s.kind == Kind::Class)
        .count();
    println!("\n(of {total_all} symbols, {classes} are classes — each implies its own method surface)");
    Ok(())
}

/// List the upstream symbols a module has not claimed yet, grouped by kind.
///
/// `report` gives the number; this gives the work. Grouping by kind matters
/// because the kinds are not equally expensive: a missing `EnumConst` is one
/// line, a missing `Class` implies a whole method surface.
fn missing(opencv: &Path, module: Option<&str>) -> std::io::Result<()> {
    let root = workspace_root();
    let claimed = ported::scan(&root)?;
    let empty = BTreeSet::new();

    let wanted: Vec<&str> = match module {
        Some(m) => MODULES.iter().copied().filter(|x| *x == m).collect(),
        None => MODULES.to_vec(),
    };
    if wanted.is_empty() {
        eprintln!("error: unknown module {:?}; expected one of {}", module.unwrap(), MODULES.join(", "));
        return Ok(());
    }

    for m in wanted {
        let scanned = upstream::scan_module(opencv, m)?;
        let mine = claimed.get(&crate_for(m)).unwrap_or(&empty);
        let mut gaps: Vec<&upstream::Symbol> =
            scanned.symbols.iter().filter(|s| !mine.contains(s.name.as_str())).collect();
        gaps.sort_by(|a, b| a.kind.cmp(&b.kind).then_with(|| a.name.cmp(&b.name)));

        println!("== {m}: {} missing of {}", gaps.len(), scanned.symbols.len());
        for kind in [Kind::Class, Kind::Function, Kind::EnumConst, Kind::Macro] {
            let of_kind: Vec<&str> =
                gaps.iter().filter(|s| s.kind == kind).map(|s| s.name.as_str()).collect();
            if of_kind.is_empty() {
                continue;
            }
            println!("\n  {} ({}):", kind.tag(), of_kind.len());
            for name in of_kind {
                println!("    {name}");
            }
        }
        println!();
    }
    Ok(())
}

/// A 20-cell progress bar.
fn bar(pct: f64) -> String {
    let filled = ((pct / 100.0) * 20.0).round() as usize;
    format!("[{}{}]", "#".repeat(filled), ".".repeat(20 - filled))
}

/// Read `CV_VERSION` out of the upstream tree.
fn cv_version(opencv: &Path) -> String {
    let p = opencv.join("modules/core/include/opencv2/core/version.hpp");
    let Ok(text) = std::fs::read_to_string(p) else { return "unknown".into() };
    let field = |key: &str| -> String {
        text.lines()
            .find_map(|l| l.strip_prefix(&format!("#define {key}")))
            .map(|v| v.trim().to_string())
            .unwrap_or_else(|| "?".into())
    };
    format!(
        "{}.{}.{}",
        field("CV_VERSION_MAJOR"),
        field("CV_VERSION_MINOR"),
        field("CV_VERSION_REVISION")
    )
}

/// Read the upstream git HEAD, if the checkout has one.
fn cv_commit(opencv: &Path) -> String {
    let head = opencv.join(".git/HEAD");
    let Ok(text) = std::fs::read_to_string(&head) else { return "unknown".into() };
    let text = text.trim();
    if let Some(rf) = text.strip_prefix("ref: ") {
        std::fs::read_to_string(opencv.join(".git").join(rf))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "unknown".into())
    } else {
        text.to_string()
    }
}
