//! `cv::FileStorage` — reading and writing YAML and JSON
//! (`modules/core/src/persistence*.cpp`).
//!
//! # Scope
//!
//! YAML and JSON, both directions. XML is not here: OpenCV's XML dialect
//! carries type information in attributes that the other two put in the
//! structure, so it is a third parser rather than a third serialiser, and the
//! two formats below cover what people actually write.
//!
//! # Layout
//!
//! A matrix serialises the way upstream does it — a mapping with `rows`,
//! `cols`, `dt` and a flat `data` sequence — so a file written here opens in
//! OpenCV and vice versa. `dt` is upstream's type letter: `u` for `CV_8U`,
//! `f` for `CV_32F`, and so on, prefixed by the channel count when above one.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::mat::{Mat, MatLike};
use crate::types::{Depth, MatType};
use crate::{cv_assert, Code, Error, Result};

/// A value in a storage tree — `cv::FileNode`.
///
/// Not `PartialEq`: [`Mat`] deliberately is not, since comparing two images
/// for exact equality is almost never the question a caller means to ask.
#[derive(Debug, Clone)]
pub enum Node {
    /// An integer.
    Int(i64),
    /// A real number.
    Real(f64),
    /// A string.
    Str(String),
    /// An ordered list.
    Seq(Vec<Node>),
    /// A named mapping. Ordered, so a round trip preserves key order.
    Map(BTreeMap<String, Node>),
    /// A matrix, stored in upstream's `rows`/`cols`/`dt`/`data` form.
    Mat(Mat),
}

impl Node {
    /// Read as an integer, if it is one.
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Node::Int(v) => Some(*v),
            Node::Real(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Read as a real, accepting an integer too.
    #[must_use]
    pub fn as_real(&self) -> Option<f64> {
        match self {
            Node::Real(v) => Some(*v),
            Node::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Read as a string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Node::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Read as a sequence.
    #[must_use]
    pub fn as_seq(&self) -> Option<&[Node]> {
        match self {
            Node::Seq(v) => Some(v),
            _ => None,
        }
    }

    /// Read as a matrix.
    #[must_use]
    pub fn as_mat(&self) -> Option<&Mat> {
        match self {
            Node::Mat(m) => Some(m),
            _ => None,
        }
    }

    /// Look a key up in a mapping — `cv::FileNode::operator[]`, which the
    /// bindings expose as `cv::FileNode::getNode(nodename)`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Node> {
        match self {
            Node::Map(m) => m.get(key),
            _ => None,
        }
    }

    /// Index into a sequence — `cv::FileNode::operator[](int)`, exposed to the
    /// bindings as `cv::FileNode::at(i)`.
    ///
    /// `None` for an out-of-range index or a node that is not a sequence.
    /// Upstream returns an empty `FileNode` for both, which reads the same but
    /// cannot be told apart from a node that is genuinely empty.
    #[must_use]
    pub fn at(&self, i: usize) -> Option<&Node> {
        match self {
            Node::Seq(v) => v.get(i),
            _ => None,
        }
    }

    /// `cv::FileNode::type()` — one of the [`node_type`] constants.
    ///
    /// A `Mat` reports `MAP`, because that is what it is on disk: upstream
    /// serialises it as a mapping of `rows`, `cols`, `dt` and `data`.
    #[must_use]
    pub fn node_type(&self) -> i32 {
        match self {
            Node::Int(_) => node_type::INT,
            Node::Real(_) => node_type::REAL,
            Node::Str(_) => node_type::STR,
            Node::Seq(_) => node_type::SEQ,
            Node::Map(_) | Node::Mat(_) => node_type::MAP,
        }
    }

    /// `cv::FileNode::isNone()`.
    ///
    /// Always false: an absent key is `None` from [`Node::get`] rather than an
    /// empty node, so there is no value here that stands for "missing".
    #[must_use]
    pub fn is_none(&self) -> bool {
        false
    }

    /// `cv::FileNode::isInt()`.
    #[must_use]
    pub fn is_int(&self) -> bool {
        matches!(self, Node::Int(_))
    }

    /// `cv::FileNode::isReal()`.
    #[must_use]
    pub fn is_real(&self) -> bool {
        matches!(self, Node::Real(_))
    }

    /// `cv::FileNode::isString()`.
    #[must_use]
    pub fn is_string(&self) -> bool {
        matches!(self, Node::Str(_))
    }

    /// `cv::FileNode::isSeq()`.
    #[must_use]
    pub fn is_seq(&self) -> bool {
        matches!(self, Node::Seq(_))
    }

    /// `cv::FileNode::isMap()`.
    #[must_use]
    pub fn is_map(&self) -> bool {
        matches!(self, Node::Map(_) | Node::Mat(_))
    }

    /// `cv::FileNode::isNamed()`.
    ///
    /// Always false: a name belongs to the mapping that holds the node, and
    /// this representation does not copy it into the child.
    #[must_use]
    pub fn is_named(&self) -> bool {
        false
    }

    /// `cv::FileNode::empty()` — an empty collection, or an empty string.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Node::Seq(v) => v.is_empty(),
            Node::Map(m) => m.is_empty(),
            Node::Str(s) => s.is_empty(),
            _ => false,
        }
    }

    /// `cv::FileNode::size()` — element count of a collection, else 1.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Node::Seq(v) => v.len(),
            Node::Map(m) => m.len(),
            _ => 1,
        }
    }

    /// `cv::FileNode::rawSize()` — bytes a `readRaw` would consume.
    ///
    /// Only a matrix has a raw byte form here; everything else reports zero,
    /// since this port never writes the packed binary layout upstream's
    /// `rawSize` measures.
    #[must_use]
    pub fn raw_size(&self) -> usize {
        match self {
            Node::Mat(m) => m.total() * m.elem_size(),
            _ => 0,
        }
    }

    /// `cv::FileNode::keys()` — the names in a mapping, in order.
    #[must_use]
    pub fn keys(&self) -> Vec<&str> {
        match self {
            Node::Map(m) => m.keys().map(String::as_str).collect(),
            _ => Vec::new(),
        }
    }

    /// `cv::FileNode::real()` — the value as a double, or 0 if it is not a number.
    ///
    /// Upstream's conversion operators return a default rather than failing, so
    /// `(double)fs["missing"]` yields 0; [`Node::as_real`] is the checked form.
    #[must_use]
    pub fn real(&self) -> f64 {
        self.as_real().unwrap_or(0.0)
    }

    /// `cv::FileNode::string()` — the value as a string, or empty.
    #[must_use]
    pub fn string(&self) -> &str {
        self.as_str().unwrap_or("")
    }

    /// `cv::FileNode::mat()` — the value as a matrix, or an empty one.
    #[must_use]
    pub fn mat(&self) -> Mat {
        match self {
            Node::Mat(m) => m.clone(),
            _ => Mat::empty(crate::types::CV_8UC1),
        }
    }

    /// `cv::FileNode::name()`.
    ///
    /// Always empty: a name belongs to the mapping entry, not the value, and
    /// this representation does not duplicate it into the child. Use
    /// [`Node::keys`] on the parent to recover it.
    #[must_use]
    pub fn name(&self) -> &str {
        ""
    }
}

/// `cv::FileNode`'s type constants, with upstream's values.
pub mod node_type {
    /// `FileNode::NONE` — an empty node.
    pub const NONE: i32 = 0;
    /// `FileNode::INT`.
    pub const INT: i32 = 1;
    /// `FileNode::REAL`.
    pub const REAL: i32 = 2;
    /// `FileNode::FLOAT` — upstream's synonym for `REAL`.
    pub const FLOAT: i32 = REAL;
    /// `FileNode::STR` — a UTF-8 string.
    pub const STR: i32 = 3;
    /// `FileNode::STRING` — upstream's synonym for `STR`.
    pub const STRING: i32 = STR;
    /// `FileNode::SEQ`.
    pub const SEQ: i32 = 4;
    /// `FileNode::MAP`.
    pub const MAP: i32 = 5;
    /// `FileNode::TYPE_MASK` — the bits holding the type.
    pub const TYPE_MASK: i32 = 7;
    /// `FileNode::FLOW` — write a collection in YAML's compact form.
    pub const FLOW: i32 = 8;
    /// `FileNode::UNIFORM` — every element has the same numeric type.
    ///
    /// Shares its bit with [`FLOW`]: upstream only sets `FLOW` when writing and
    /// only reads `UNIFORM` when parsing, so the two never collide.
    pub const UNIFORM: i32 = 8;
    /// `FileNode::EMPTY` — an empty sequence or mapping.
    pub const EMPTY: i32 = 16;
    /// `FileNode::NAMED` — the node is an element of a mapping.
    pub const NAMED: i32 = 32;
}

/// `cv::FileStorage::Mode`, with upstream's values.
///
/// This port opens storages through [`FileStorage::read_file`],
/// [`FileStorage::write_file`] and [`FileStorage::from_str`] rather than
/// through one flags word, so these are declared for translating C++ code.
pub mod mode {
    /// `FileStorage::READ`.
    pub const READ: i32 = 0;
    /// `FileStorage::WRITE`.
    pub const WRITE: i32 = 1;
    /// `FileStorage::APPEND`.
    pub const APPEND: i32 = 2;
    /// `FileStorage::MEMORY` — read from or write to a string in memory.
    pub const MEMORY: i32 = 4;
    /// `FileStorage::FORMAT_MASK`.
    pub const FORMAT_MASK: i32 = 7 << 3;
    /// `FileStorage::FORMAT_AUTO` — pick the format from the file extension.
    pub const FORMAT_AUTO: i32 = 0;
    /// `FileStorage::FORMAT_XML`. Not supported here; see the module header.
    pub const FORMAT_XML: i32 = 1 << 3;
    /// `FileStorage::FORMAT_YAML`.
    pub const FORMAT_YAML: i32 = 2 << 3;
    /// `FileStorage::FORMAT_JSON`.
    pub const FORMAT_JSON: i32 = 3 << 3;
    /// `FileStorage::BASE64` — write raw data Base64-encoded.
    pub const BASE64: i32 = 64;
    /// `FileStorage::WRITE_BASE64`.
    pub const WRITE_BASE64: i32 = BASE64 | WRITE;
}

/// `cv::FileStorage::State` — where a sequential writer is in the structure.
///
/// This port builds the whole tree before serialising, so no writer state is
/// ever observable; the constants exist for translation.
pub mod state {
    /// `FileStorage::UNDEFINED`.
    pub const UNDEFINED: i32 = 0;
    /// `FileStorage::VALUE_EXPECTED`.
    pub const VALUE_EXPECTED: i32 = 1;
    /// `FileStorage::NAME_EXPECTED`.
    pub const NAME_EXPECTED: i32 = 2;
    /// `FileStorage::INSIDE_MAP`.
    pub const INSIDE_MAP: i32 = 4;
}

/// Which syntax a [`FileStorage`] reads or writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// `cv::FileStorage::FORMAT_YAML`
    Yaml,
    /// `cv::FileStorage::FORMAT_JSON`
    Json,
}

impl Format {
    /// Guess from a file extension, as `cv::FileStorage` does.
    #[must_use]
    pub fn from_path(path: &str) -> Format {
        let p = path.to_ascii_lowercase();
        if p.ends_with(".json") {
            Format::Json
        } else {
            Format::Yaml
        }
    }
}

/// `CV__LEGACY_PERSISTENCE` — read files written by OpenCV 2.x.
///
/// False. The 2.x writer emitted a different matrix element order; this port
/// reads and writes only the 3.x/4.x form, which is what every current OpenCV
/// produces.
pub const CV_LEGACY_PERSISTENCE: bool = false;

/// `cv::FileStorage`.
#[derive(Debug, Clone, Default)]
pub struct FileStorage {
    root: BTreeMap<String, Node>,
}

impl FileStorage {
    /// An empty storage.
    #[must_use]
    pub fn new() -> Self {
        FileStorage::default()
    }

    /// Store a node under `key` — `cv::FileStorage::operator<<`.
    pub fn set(&mut self, key: &str, node: Node) -> &mut Self {
        self.root.insert(key.to_string(), node);
        self
    }

    /// Store an integer.
    pub fn set_int(&mut self, key: &str, v: i64) -> &mut Self {
        self.set(key, Node::Int(v))
    }

    /// Store a real.
    pub fn set_real(&mut self, key: &str, v: f64) -> &mut Self {
        self.set(key, Node::Real(v))
    }

    /// Store a string.
    pub fn set_str(&mut self, key: &str, v: &str) -> &mut Self {
        self.set(key, Node::Str(v.to_string()))
    }

    /// Store a matrix.
    pub fn set_mat<M: MatLike + ?Sized>(&mut self, key: &str, m: &M) -> Result<&mut Self> {
        Ok(self.set(key, Node::Mat(Mat::copy_from(m)?)))
    }

    /// `cv::FileStorage::startWriteStruct(name, flags, typeName)`.
    ///
    /// Upstream's writer is sequential, so it needs a matched
    /// `endWriteStruct`; here the whole collection is handed over at once and
    /// there is nothing left open afterwards. Pass [`Node::Seq`] or
    /// [`Node::Map`] according to which of `FileNode::SEQ` / `FileNode::MAP`
    /// the C++ call would have used.
    pub fn start_write_struct(&mut self, name: &str, contents: Node) -> &mut Self {
        self.set(name, contents)
    }

    /// `cv::FileStorage::endWriteStruct()`.
    ///
    /// A no-op, returned for symmetry with [`FileStorage::start_write_struct`]:
    /// nothing is left half-written for it to close.
    pub fn end_write_struct(&mut self) -> &mut Self {
        self
    }

    /// `cv::FileStorage::writeComment(comment, append)`.
    ///
    /// Not stored: comments do not survive a round trip through this tree, and
    /// silently dropping one on write is better than emitting it at a position
    /// the reader would then treat as a key.
    pub fn write_comment(&mut self, _comment: &str, _append: bool) -> &mut Self {
        self
    }

    /// Read a node back — `cv::FileStorage::operator[]`, which the bindings
    /// expose as `cv::FileStorage::getNode(nodename)`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Node> {
        self.root.get(key)
    }

    /// `cv::FileStorage::root(streamIdx)` — the whole tree as one mapping node.
    #[must_use]
    pub fn root(&self) -> Node {
        Node::Map(self.root.clone())
    }

    /// `cv::FileStorage::getFirstTopLevelNode()`.
    #[must_use]
    pub fn first_top_level_node(&self) -> Option<&Node> {
        self.root.values().next()
    }

    /// `cv::FileStorage::isOpened()`.
    ///
    /// Always true: this type holds a parsed tree rather than a file handle, so
    /// a failed open is an `Err` from [`FileStorage::read_file`] and never
    /// yields a storage at all.
    #[must_use]
    pub fn is_opened(&self) -> bool {
        true
    }

    /// `cv::FileStorage::getFormat()`.
    ///
    /// Reports the format a [`FileStorage::write_file`] to `path` would choose.
    /// Upstream remembers the format the storage was opened with; this type
    /// does not carry one, since the same tree can be written either way.
    #[must_use]
    pub fn format_for_path(path: &str) -> Format {
        Format::from_path(path)
    }

    /// `cv::FileStorage::releaseAndGetString()` — serialise, then empty.
    pub fn release_and_get_string(&mut self, format: Format) -> Result<String> {
        let text = self.to_string(format)?;
        self.root.clear();
        Ok(text)
    }

    /// `cv::FileStorage::release()` — drop everything stored.
    pub fn release(&mut self) {
        self.root.clear();
    }

    /// Keys present, in order.
    #[must_use]
    pub fn keys(&self) -> Vec<&str> {
        self.root.keys().map(String::as_str).collect()
    }

    /// Whether anything has been stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.root.is_empty()
    }

    /// Number of top-level keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.root.len()
    }

    /// Serialise to a string.
    pub fn to_string(&self, format: Format) -> Result<String> {
        let mut out = String::new();
        match format {
            Format::Yaml => {
                out.push_str("%YAML:1.0\n---\n");
                for (k, v) in &self.root {
                    write_yaml(&mut out, k, v, 0)?;
                }
            }
            Format::Json => {
                out.push_str("{\n");
                let n = self.root.len();
                for (i, (k, v)) in self.root.iter().enumerate() {
                    write_json(&mut out, Some(k), v, 1)?;
                    out.push_str(if i + 1 < n { ",\n" } else { "\n" });
                }
                out.push('}');
            }
        }
        Ok(out)
    }

    /// `cv::FileStorage::open(filename, WRITE)` then write everything.
    pub fn write_file(&self, path: &str) -> Result<()> {
        let text = self.to_string(Format::from_path(path))?;
        std::fs::write(path, text)
            .map_err(|e| Error::new(Code::StsError, format!("cannot write {path}: {e}")))
    }

    /// Parse from a string.
    pub fn from_str(text: &str, format: Format) -> Result<Self> {
        match format {
            Format::Yaml => parse_yaml(text),
            Format::Json => parse_json(text),
        }
    }

    /// `cv::FileStorage::open(filename, READ)`.
    pub fn read_file(path: &str) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::new(Code::StsObjectNotFound, format!("cannot read {path}: {e}")))?;
        Self::from_str(&text, Format::from_path(path))
    }
}

/// Upstream's `dt` letter for a depth.
fn dt_letter(d: Depth) -> Result<char> {
    Ok(match d {
        Depth::U8 => 'u',
        Depth::S8 => 'c',
        Depth::U16 => 'w',
        Depth::S16 => 's',
        Depth::S32 => 'i',
        Depth::F32 => 'f',
        Depth::F64 => 'd',
        Depth::F16 => {
            return Err(Error::new(
                Code::StsUnsupportedFormat,
                "CV_16F has no FileStorage type letter here".to_string(),
            ))
        }
    })
}

/// The depth a `dt` letter names.
fn letter_depth(c: char) -> Result<Depth> {
    Ok(match c {
        'u' => Depth::U8,
        'c' => Depth::S8,
        'w' => Depth::U16,
        's' => Depth::S16,
        'i' => Depth::S32,
        'f' => Depth::F32,
        'd' => Depth::F64,
        other => {
            return Err(Error::new(
                Code::StsParseError,
                format!("unknown FileStorage type letter {other:?}"),
            ))
        }
    })
}

/// Every channel value of a matrix, row-major.
fn mat_values<M: MatLike + ?Sized>(m: &M) -> Result<Vec<f64>> {
    let mut v = Vec::with_capacity((m.rows() * m.cols() * m.channels()) as usize);
    for y in 0..m.rows() {
        for x in 0..m.cols() {
            for c in 0..m.channels() {
                v.push(m.at_f64(y, x, c)?);
            }
        }
    }
    Ok(v)
}

/// Render a number the way both formats want it.
///
/// Integers print without a decimal point; reals keep enough digits to survive
/// the round trip, which `{}` alone does not guarantee for `f64`.
fn num(v: f64, integral: bool) -> String {
    if integral {
        format!("{}", v as i64)
    } else if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{v:.1}")
    } else {
        format!("{v:?}")
    }
}

fn write_yaml(out: &mut String, key: &str, node: &Node, indent: usize) -> Result<()> {
    let pad = "   ".repeat(indent);
    match node {
        Node::Int(v) => writeln!(out, "{pad}{key}: {v}").ok(),
        Node::Real(v) => writeln!(out, "{pad}{key}: {}", num(*v, false)).ok(),
        Node::Str(s) => writeln!(out, "{pad}{key}: \"{}\"", escape(s)).ok(),
        Node::Seq(items) => {
            writeln!(out, "{pad}{key}:").ok();
            for it in items {
                write_yaml_item(out, it, indent + 1)?;
            }
            Some(())
        }
        Node::Map(m) => {
            writeln!(out, "{pad}{key}:").ok();
            for (k, v) in m {
                write_yaml(out, k, v, indent + 1)?;
            }
            Some(())
        }
        Node::Mat(m) => {
            let d = m.depth();
            let integral = !d.is_float();
            let letter = dt_letter(d)?;
            let dt = if m.channels() > 1 {
                format!("{}{letter}", m.channels())
            } else {
                letter.to_string()
            };
            writeln!(out, "{pad}{key}: !!opencv-matrix").ok();
            writeln!(out, "{pad}   rows: {}", m.rows()).ok();
            writeln!(out, "{pad}   cols: {}", m.cols()).ok();
            writeln!(out, "{pad}   dt: {dt}").ok();
            let vals: Vec<String> =
                mat_values(m)?.into_iter().map(|v| num(v, integral)).collect();
            writeln!(out, "{pad}   data: [ {} ]", vals.join(", ")).ok()
        }
    };
    Ok(())
}

fn write_yaml_item(out: &mut String, node: &Node, indent: usize) -> Result<()> {
    let pad = "   ".repeat(indent);
    match node {
        Node::Int(v) => {
            writeln!(out, "{pad}- {v}").ok();
        }
        Node::Real(v) => {
            writeln!(out, "{pad}- {}", num(*v, false)).ok();
        }
        Node::Str(s) => {
            writeln!(out, "{pad}- \"{}\"", escape(s)).ok();
        }
        other => {
            // A nested structure inside a sequence: emit it as a one-key map.
            writeln!(out, "{pad}-").ok();
            write_yaml(out, "value", other, indent + 1)?;
        }
    }
    Ok(())
}

fn write_json(out: &mut String, key: Option<&str>, node: &Node, indent: usize) -> Result<()> {
    let pad = "   ".repeat(indent);
    let head = match key {
        Some(k) => format!("{pad}\"{}\": ", escape(k)),
        None => pad.clone(),
    };
    match node {
        Node::Int(v) => {
            write!(out, "{head}{v}").ok();
        }
        Node::Real(v) => {
            write!(out, "{head}{}", num(*v, false)).ok();
        }
        Node::Str(s) => {
            write!(out, "{head}\"{}\"", escape(s)).ok();
        }
        Node::Seq(items) => {
            writeln!(out, "{head}[").ok();
            for (i, it) in items.iter().enumerate() {
                write_json(out, None, it, indent + 1)?;
                out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
            }
            write!(out, "{pad}]").ok();
        }
        Node::Map(m) => {
            writeln!(out, "{head}{{").ok();
            for (i, (k, v)) in m.iter().enumerate() {
                write_json(out, Some(k), v, indent + 1)?;
                out.push_str(if i + 1 < m.len() { ",\n" } else { "\n" });
            }
            write!(out, "{pad}}}").ok();
        }
        Node::Mat(m) => {
            let d = m.depth();
            let integral = !d.is_float();
            let letter = dt_letter(d)?;
            let dt = if m.channels() > 1 {
                format!("{}{letter}", m.channels())
            } else {
                letter.to_string()
            };
            let vals: Vec<String> =
                mat_values(m)?.into_iter().map(|v| num(v, integral)).collect();
            writeln!(out, "{head}{{").ok();
            writeln!(out, "{pad}   \"rows\": {},", m.rows()).ok();
            writeln!(out, "{pad}   \"cols\": {},", m.cols()).ok();
            writeln!(out, "{pad}   \"dt\": \"{dt}\",").ok();
            writeln!(out, "{pad}   \"data\": [ {} ]", vals.join(", ")).ok();
            write!(out, "{pad}}}").ok();
        }
    }
    Ok(())
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(other) => out.push(other),
                None => break,
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Build a matrix from the pieces a storage file carries.
fn assemble_mat(rows: i32, cols: i32, dt: &str, data: &[f64]) -> Result<Mat> {
    let mut chars = dt.chars().peekable();
    let mut cn = 0i32;
    while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
        cn = cn * 10 + chars.next().unwrap().to_digit(10).unwrap() as i32;
    }
    let cn = if cn == 0 { 1 } else { cn };
    let letter = chars.next().ok_or_else(|| {
        Error::new(Code::StsParseError, format!("empty dt field in {dt:?}"))
    })?;
    let depth = letter_depth(letter)?;
    cv_assert!(rows > 0 && cols > 0, "a stored matrix must be at least 1x1, got {cols}x{rows}");
    let want = (rows * cols * cn) as usize;
    cv_assert!(
        data.len() == want,
        "the data list holds {} values but {rows}x{cols}x{cn} needs {want}",
        data.len()
    );
    let mut m = Mat::zeros(rows, cols, MatType::new(depth, cn))?;
    let mut i = 0;
    for y in 0..rows {
        for x in 0..cols {
            for c in 0..cn {
                crate::arithm::set_channel_generic(&mut m, y, x, c, data[i])?;
                i += 1;
            }
        }
    }
    Ok(m)
}

/// Split a bracketed list into its comma-separated numbers.
fn parse_numbers(s: &str) -> Result<Vec<f64>> {
    let inner = s.trim().trim_start_matches('[').trim_end_matches(']');
    let mut out = Vec::new();
    for tok in inner.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        out.push(t.parse::<f64>().map_err(|_| {
            Error::new(Code::StsParseError, format!("{t:?} is not a number"))
        })?);
    }
    Ok(out)
}

/// Indentation-driven YAML parser for the subset this module writes.
fn parse_yaml(text: &str) -> Result<FileStorage> {
    let mut fs = FileStorage::new();
    // Flatten continued `data: [ ... ]` lists onto one line first: upstream
    // wraps them at 80 columns, so a reader that assumes one line per value
    // fails on any real calibration file.
    let joined = join_wrapped_lists(text);
    let lines: Vec<&str> = joined
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#') && !t.starts_with("%YAML") && t != "---"
        })
        .collect();

    let mut i = 0usize;
    while i < lines.len() {
        let (key, node, used) = parse_yaml_entry(&lines, i, indent_of(lines[i]))?;
        if used == 0 {
            i += 1;
            continue;
        }
        fs.set(&key, node);
        i += used;
    }
    Ok(fs)
}

fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Join a `[` list that spans several lines into one.
fn join_wrapped_lists(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth = 0i32;
    for line in text.lines() {
        let opens = line.matches('[').count() as i32;
        let closes = line.matches(']').count() as i32;
        if depth > 0 {
            out.push(' ');
            out.push_str(line.trim());
        } else {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(line);
        }
        depth += opens - closes;
        if depth < 0 {
            depth = 0;
        }
    }
    out.push('\n');
    out
}

/// Parse one `key: value` entry and everything nested under it.
fn parse_yaml_entry(lines: &[&str], at: usize, indent: usize) -> Result<(String, Node, usize)> {
    let line = lines[at];
    let trimmed = line.trim();
    let (key, rest) = trimmed.split_once(':').ok_or_else(|| {
        Error::new(Code::StsParseError, format!("expected `key: value`, got {trimmed:?}"))
    })?;
    let key = key.trim().to_string();
    let rest = rest.trim();

    // A matrix: the tag, then rows/cols/dt/data beneath it.
    if rest.starts_with("!!opencv-matrix") {
        let mut fields: BTreeMap<String, String> = BTreeMap::new();
        let mut used = 1;
        while at + used < lines.len() && indent_of(lines[at + used]) > indent {
            let t = lines[at + used].trim();
            if let Some((k, v)) = t.split_once(':') {
                fields.insert(k.trim().to_string(), v.trim().to_string());
            }
            used += 1;
        }
        let need = |k: &str| -> Result<&String> {
            fields.get(k).ok_or_else(|| {
                Error::new(Code::StsParseError, format!("matrix {key:?} has no {k} field"))
            })
        };
        let rows: i32 = need("rows")?.parse().unwrap_or(0);
        let cols: i32 = need("cols")?.parse().unwrap_or(0);
        let dt = need("dt")?.clone();
        let data = parse_numbers(need("data")?)?;
        return Ok((key, Node::Mat(assemble_mat(rows, cols, &dt, &data)?), used));
    }

    if !rest.is_empty() {
        return Ok((key, scalar_node(rest), 1));
    }

    // Nested block: a sequence if the first child starts with `-`.
    let mut used = 1;
    let mut items: Vec<Node> = Vec::new();
    let mut map: BTreeMap<String, Node> = BTreeMap::new();
    let mut is_seq = false;
    while at + used < lines.len() && indent_of(lines[at + used]) > indent {
        let child_indent = indent_of(lines[at + used]);
        let t = lines[at + used].trim();
        if let Some(v) = t.strip_prefix("- ") {
            is_seq = true;
            items.push(scalar_node(v.trim()));
            used += 1;
        } else if t == "-" {
            is_seq = true;
            used += 1;
        } else {
            let (k, n, u) = parse_yaml_entry(lines, at + used, child_indent)?;
            map.insert(k, n);
            used += u.max(1);
        }
    }
    Ok((key, if is_seq { Node::Seq(items) } else { Node::Map(map) }, used))
}

/// Classify a bare YAML/JSON scalar.
fn scalar_node(s: &str) -> Node {
    let t = s.trim().trim_end_matches(',');
    if t.starts_with('"') {
        return Node::Str(unescape(t.trim_matches('"')));
    }
    if t.starts_with('[') {
        return match parse_numbers(t) {
            Ok(v) => Node::Seq(v.into_iter().map(Node::Real).collect()),
            Err(_) => Node::Str(t.to_string()),
        };
    }
    if let Ok(i) = t.parse::<i64>() {
        return Node::Int(i);
    }
    if let Ok(f) = t.parse::<f64>() {
        return Node::Real(f);
    }
    Node::Str(t.to_string())
}

/// Minimal JSON parser for what this module writes.
fn parse_json(text: &str) -> Result<FileStorage> {
    let bytes: Vec<char> = text.chars().collect();
    let mut pos = 0usize;
    let node = json_value(&bytes, &mut pos)?;
    let mut fs = FileStorage::new();
    match node {
        Node::Map(m) => {
            for (k, v) in m {
                fs.set(&k, v);
            }
            Ok(fs)
        }
        _ => Err(Error::new(Code::StsParseError, "JSON root must be an object".to_string())),
    }
}

fn json_skip(c: &[char], p: &mut usize) {
    while *p < c.len() && c[*p].is_whitespace() {
        *p += 1;
    }
}

fn json_value(c: &[char], p: &mut usize) -> Result<Node> {
    json_skip(c, p);
    let bad = |m: &str| Error::new(Code::StsParseError, format!("JSON: {m}"));
    match c.get(*p) {
        Some('{') => {
            *p += 1;
            let mut map = BTreeMap::new();
            loop {
                json_skip(c, p);
                if c.get(*p) == Some(&'}') {
                    *p += 1;
                    break;
                }
                let k = match json_value(c, p)? {
                    Node::Str(s) => s,
                    _ => return Err(bad("object keys must be strings")),
                };
                json_skip(c, p);
                if c.get(*p) != Some(&':') {
                    return Err(bad("expected ':' after a key"));
                }
                *p += 1;
                map.insert(k, json_value(c, p)?);
                json_skip(c, p);
                if c.get(*p) == Some(&',') {
                    *p += 1;
                }
            }
            // A mapping carrying rows/cols/dt/data is a matrix, not a map.
            if let (Some(r), Some(cc), Some(d), Some(data)) =
                (map.get("rows"), map.get("cols"), map.get("dt"), map.get("data"))
            {
                if let (Some(r), Some(cc), Some(d), Some(items)) =
                    (r.as_int(), cc.as_int(), d.as_str(), data.as_seq())
                {
                    let vals: Vec<f64> = items.iter().filter_map(Node::as_real).collect();
                    return Ok(Node::Mat(assemble_mat(r as i32, cc as i32, d, &vals)?));
                }
            }
            Ok(Node::Map(map))
        }
        Some('[') => {
            *p += 1;
            let mut items = Vec::new();
            loop {
                json_skip(c, p);
                if c.get(*p) == Some(&']') {
                    *p += 1;
                    break;
                }
                items.push(json_value(c, p)?);
                json_skip(c, p);
                if c.get(*p) == Some(&',') {
                    *p += 1;
                }
            }
            Ok(Node::Seq(items))
        }
        Some('"') => {
            *p += 1;
            let mut s = String::new();
            while let Some(&ch) = c.get(*p) {
                *p += 1;
                if ch == '"' {
                    return Ok(Node::Str(unescape(&s)));
                }
                if ch == '\\' {
                    if let Some(&esc) = c.get(*p) {
                        s.push('\\');
                        s.push(esc);
                        *p += 1;
                    }
                } else {
                    s.push(ch);
                }
            }
            Err(bad("unterminated string"))
        }
        Some(_) => {
            let start = *p;
            while *p < c.len() && !matches!(c[*p], ',' | '}' | ']' | ' ' | '\n' | '\r' | '\t') {
                *p += 1;
            }
            let t: String = c[start..*p].iter().collect();
            Ok(scalar_node(&t))
        }
        None => Err(bad("unexpected end of input")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CV_32FC1, CV_8UC1};

    fn sample_mat() -> Mat {
        Mat::from_vec(2, 3, 1, &[1.0f32, 2.5, -3.0, 4.0, 5.5, 6.0]).unwrap()
    }

    #[test]
    fn node_predicates_agree_with_the_type_code() {
        let cases: [(Node, i32); 5] = [
            (Node::Int(1), node_type::INT),
            (Node::Real(1.0), node_type::REAL),
            (Node::Str("x".into()), node_type::STR),
            (Node::Seq(vec![Node::Int(1)]), node_type::SEQ),
            (Node::Map(BTreeMap::new()), node_type::MAP),
        ];
        for (n, want) in &cases {
            assert_eq!(n.node_type(), *want, "{n:?}");
        }
        assert!(Node::Int(1).is_int() && !Node::Int(1).is_real());
        assert!(Node::Real(1.0).is_real());
        assert!(Node::Str(String::new()).is_string() && Node::Str(String::new()).is_empty());
        assert!(Node::Seq(vec![]).is_seq() && Node::Seq(vec![]).is_empty());
        assert_eq!(Node::Seq(vec![Node::Int(1), Node::Int(2)]).size(), 2);

        // The two `operator[]` overloads: by name into a map, by index into a
        // sequence. Each returns nothing when applied to the other kind.
        let seq = Node::Seq(vec![Node::Int(7), Node::Str("x".into())]);
        assert_eq!(seq.at(0).and_then(Node::as_int), Some(7));
        assert_eq!(seq.at(1).and_then(Node::as_str), Some("x"));
        assert!(seq.at(2).is_none(), "past the end");
        assert!(seq.get("anything").is_none(), "a sequence has no names");
        let mut map = std::collections::BTreeMap::new();
        map.insert("k".to_string(), Node::Int(3));
        let map = Node::Map(map);
        assert_eq!(map.get("k").and_then(Node::as_int), Some(3));
        assert!(map.at(0).is_none(), "a mapping is not indexable");

        // A matrix is a mapping on disk, so it reports MAP.
        let m = Node::Mat(sample_mat());
        assert!(m.is_map());
        assert_eq!(m.raw_size(), 6 * 4);
    }

    #[test]
    fn release_and_get_string_empties_the_storage() {
        let mut fs = FileStorage::new();
        fs.set_int("n", 7);
        assert!(fs.is_opened());
        assert_eq!(fs.first_top_level_node().and_then(Node::as_int), Some(7));
        let text = fs.release_and_get_string(Format::Json).unwrap();
        assert!(text.contains("\"n\""), "got {text}");
        assert!(fs.is_empty());
    }

    #[test]
    fn a_struct_written_whole_reads_back_as_a_collection() {
        let mut fs = FileStorage::new();
        fs.start_write_struct("pts", Node::Seq(vec![Node::Int(1), Node::Int(2)]))
            .end_write_struct()
            .write_comment("dropped on purpose", false);
        let text = fs.to_string(Format::Yaml).unwrap();
        let back = FileStorage::from_str(&text, Format::Yaml).unwrap();
        assert_eq!(back.get("pts").map(Node::size), Some(2));
        assert_eq!(back.root().keys(), vec!["pts"]);
    }

    #[test]
    fn format_follows_the_extension() {
        assert_eq!(FileStorage::format_for_path("a.json"), Format::Json);
        assert_eq!(FileStorage::format_for_path("a.yml"), Format::Yaml);
    }

    #[test]
    fn a_matrix_round_trips_through_yaml() {
        let m = sample_mat();
        let mut fs = FileStorage::new();
        fs.set_mat("K", &m).unwrap();
        let text = fs.to_string(Format::Yaml).unwrap();
        assert!(text.starts_with("%YAML:1.0"), "the header is what cv2 looks for");
        assert!(text.contains("!!opencv-matrix"), "got:\n{text}");
        assert!(text.contains("dt: f"));

        let back = FileStorage::from_str(&text, Format::Yaml).unwrap();
        let got = back.get("K").and_then(Node::as_mat).expect("K should be a matrix");
        assert_eq!((got.rows(), got.cols(), got.channels()), (2, 3, 1));
        assert_eq!(got.mat_type(), CV_32FC1);
        assert_eq!(got.to_vec::<f32>().unwrap(), m.to_vec::<f32>().unwrap());
    }

    #[test]
    fn a_matrix_round_trips_through_json() {
        let m = sample_mat();
        let mut fs = FileStorage::new();
        fs.set_mat("K", &m).unwrap();
        let text = fs.to_string(Format::Json).unwrap();
        let back = FileStorage::from_str(&text, Format::Json).unwrap();
        let got = back.get("K").and_then(Node::as_mat).unwrap();
        assert_eq!(got.to_vec::<f32>().unwrap(), m.to_vec::<f32>().unwrap());
    }

    #[test]
    fn every_depth_and_channel_count_survives() {
        for (name, m) in [
            ("u8", Mat::from_vec(1, 4, 1, &[0u8, 7, 128, 255]).unwrap()),
            ("i32", Mat::from_vec(2, 2, 1, &[-5i32, 0, 9, 1_000_000]).unwrap()),
            ("f64", Mat::from_vec(1, 3, 1, &[0.125f64, -2.5, 1e10]).unwrap()),
            ("3ch", Mat::from_vec(1, 2, 3, &[1u8, 2, 3, 4, 5, 6]).unwrap()),
        ] {
            let mut fs = FileStorage::new();
            fs.set_mat("m", &m).unwrap();
            for fmt in [Format::Yaml, Format::Json] {
                let text = fs.to_string(fmt).unwrap();
                let back = FileStorage::from_str(&text, fmt).unwrap();
                let got = back.get("m").and_then(Node::as_mat).unwrap();
                assert_eq!(got.mat_type(), m.mat_type(), "{name} via {fmt:?}: type changed");
                assert_eq!(
                    (got.rows(), got.cols()),
                    (m.rows(), m.cols()),
                    "{name} via {fmt:?}: shape changed"
                );
                for y in 0..m.rows() {
                    for x in 0..m.cols() {
                        for c in 0..m.channels() {
                            assert_eq!(
                                got.at_f64(y, x, c).unwrap(),
                                m.at_f64(y, x, c).unwrap(),
                                "{name} via {fmt:?} at ({x},{y},{c})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scalars_and_strings_round_trip() {
        let mut fs = FileStorage::new();
        fs.set_int("count", 42)
            .set_real("scale", 0.125)
            .set_str("name", "a \"quoted\" value");
        for fmt in [Format::Yaml, Format::Json] {
            let back = FileStorage::from_str(&fs.to_string(fmt).unwrap(), fmt).unwrap();
            assert_eq!(back.get("count").and_then(Node::as_int), Some(42), "{fmt:?}");
            assert_eq!(back.get("scale").and_then(Node::as_real), Some(0.125), "{fmt:?}");
            assert_eq!(
                back.get("name").and_then(Node::as_str),
                Some("a \"quoted\" value"),
                "{fmt:?}"
            );
        }
    }

    #[test]
    fn an_integer_stays_an_integer_and_a_real_keeps_its_point() {
        let mut fs = FileStorage::new();
        fs.set_int("i", 7).set_real("r", 7.0);
        let y = fs.to_string(Format::Yaml).unwrap();
        assert!(y.contains("i: 7\n"), "got:\n{y}");
        // Without the trailing `.0` a whole-valued real reads back as an int.
        assert!(y.contains("r: 7.0"), "got:\n{y}");
        let back = FileStorage::from_str(&y, Format::Yaml).unwrap();
        assert!(matches!(back.get("i"), Some(Node::Int(7))));
        assert!(matches!(back.get("r"), Some(Node::Real(_))));
    }

    /// Files cv2 actually writes, verbatim.
    ///
    /// Two details here are not what a from-scratch writer would produce and
    /// would break a parser that only ever read its own output: cv2 emits
    /// `800.` — a trailing point with no digit after it — and leaves strings
    /// **unquoted**. Both appear below on purpose.
    #[test]
    fn files_written_by_cv2_parse() {
        let yaml = "%YAML:1.0\n---\ncamera_matrix: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n   data: [ 800., 0., 320., 0., 800., 240., 0., 0., 1. ]\nimage_width: 640\nnote: written by cv2\n";
        let fs = FileStorage::from_str(yaml, Format::Yaml).unwrap();
        let k = fs.get("camera_matrix").and_then(Node::as_mat).expect("camera_matrix");
        assert_eq!(
            k.to_vec::<f64>().unwrap(),
            vec![800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(fs.get("image_width").and_then(Node::as_int), Some(640));
        assert_eq!(fs.get("note").and_then(Node::as_str), Some("written by cv2"));

        let json = "{\n    \"m\": {\n        \"type_id\": \"opencv-matrix\",\n        \"rows\": 1,\n        \"cols\": 3,\n        \"dt\": \"d\",\n        \"data\": [ -2.0000000000000001e-01, 5.0e-02, 0. ]\n    },\n    \"n\": 7\n}";
        let fj = FileStorage::from_str(json, Format::Json).unwrap();
        let m = fj.get("m").and_then(Node::as_mat).expect("m");
        let v = m.to_vec::<f64>().unwrap();
        assert!((v[0] + 0.2).abs() < 1e-12 && (v[1] - 0.05).abs() < 1e-12 && v[2] == 0.0);
        assert_eq!(fj.get("n").and_then(Node::as_int), Some(7));
    }

    #[test]
    fn a_wrapped_data_list_is_rejoined() {
        // Upstream wraps `data` at about 80 columns. A reader that assumes one
        // line per list would see a truncated matrix and not notice.
        let text = "%YAML:1.0\n---\nM: !!opencv-matrix\n   rows: 2\n   cols: 3\n   dt: i\n   data: [ 1, 2, 3,\n       4, 5, 6 ]\n";
        let fs = FileStorage::from_str(text, Format::Yaml).unwrap();
        let m = fs.get("M").and_then(Node::as_mat).expect("M should parse");
        assert_eq!(m.to_vec::<i32>().unwrap(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn a_short_or_long_data_list_is_an_error() {
        let short = "%YAML:1.0\n---\nM: !!opencv-matrix\n   rows: 2\n   cols: 3\n   dt: i\n   data: [ 1, 2, 3 ]\n";
        let e = FileStorage::from_str(short, Format::Yaml).unwrap_err();
        assert!(e.msg.contains("needs 6"), "the error should say what was expected: {}", e.msg);
    }

    #[test]
    fn an_unknown_type_letter_is_named() {
        let text = "%YAML:1.0\n---\nM: !!opencv-matrix\n   rows: 1\n   cols: 1\n   dt: z\n   data: [ 1 ]\n";
        let e = FileStorage::from_str(text, Format::Yaml).unwrap_err();
        assert!(e.msg.contains('z'), "got: {}", e.msg);
    }

    #[test]
    fn nested_maps_and_sequences_survive_yaml() {
        let mut inner = BTreeMap::new();
        inner.insert("a".to_string(), Node::Int(1));
        inner.insert("b".to_string(), Node::Real(2.5));
        let mut fs = FileStorage::new();
        fs.set("group", Node::Map(inner));
        fs.set("list", Node::Seq(vec![Node::Int(1), Node::Int(2), Node::Int(3)]));
        let text = fs.to_string(Format::Yaml).unwrap();
        let back = FileStorage::from_str(&text, Format::Yaml).unwrap();
        assert_eq!(back.get("group").and_then(|g| g.get("a")).and_then(Node::as_int), Some(1));
        assert_eq!(back.get("group").and_then(|g| g.get("b")).and_then(Node::as_real), Some(2.5));
        let seq = back.get("list").and_then(Node::as_seq).expect("a sequence");
        assert_eq!(seq.len(), 3);
        assert_eq!(seq[2].as_int(), Some(3));
    }

    #[test]
    fn nested_structures_survive_json() {
        let mut inner = BTreeMap::new();
        inner.insert("k".to_string(), Node::Str("v".into()));
        let mut fs = FileStorage::new();
        fs.set("group", Node::Map(inner));
        fs.set("list", Node::Seq(vec![Node::Real(1.5), Node::Str("two".into())]));
        let back =
            FileStorage::from_str(&fs.to_string(Format::Json).unwrap(), Format::Json).unwrap();
        assert_eq!(back.get("group").and_then(|g| g.get("k")).and_then(Node::as_str), Some("v"));
        let seq = back.get("list").and_then(Node::as_seq).unwrap();
        assert_eq!(seq[0].as_real(), Some(1.5));
        assert_eq!(seq[1].as_str(), Some("two"));
    }

    #[test]
    fn the_format_is_chosen_by_extension() {
        assert_eq!(Format::from_path("a/b.json"), Format::Json);
        assert_eq!(Format::from_path("a/b.YML"), Format::Yaml);
        assert_eq!(Format::from_path("a/b.yaml"), Format::Yaml);
        assert_eq!(Format::from_path("no_extension"), Format::Yaml);
    }

    #[test]
    fn files_round_trip_on_disk() {
        let dir = std::env::temp_dir().join("cv_fs_rt");
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["out.yml", "out.json"] {
            let path = dir.join(name).to_string_lossy().to_string();
            let mut fs = FileStorage::new();
            fs.set_mat("camera", &sample_mat()).unwrap();
            fs.set_int("frames", 120);
            fs.write_file(&path).unwrap();

            let back = FileStorage::read_file(&path).unwrap();
            assert_eq!(back.get("frames").and_then(Node::as_int), Some(120), "{name}");
            assert_eq!(
                back.get("camera").and_then(Node::as_mat).unwrap().to_vec::<f32>().unwrap(),
                sample_mat().to_vec::<f32>().unwrap(),
                "{name}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_file_is_a_clear_error() {
        let e = FileStorage::read_file("/nonexistent/config.yml").unwrap_err();
        assert_eq!(e.code, Code::StsObjectNotFound);
    }

    #[test]
    fn keys_and_emptiness_report_correctly() {
        let mut fs = FileStorage::new();
        assert!(fs.is_empty());
        fs.set_int("b", 1).set_int("a", 2);
        assert_eq!(fs.keys(), vec!["a", "b"], "keys come back sorted, so files are stable");
        assert_eq!(fs.len(), 2);
        assert!(fs.get("missing").is_none());
    }

    #[test]
    fn node_accessors_reject_the_wrong_type() {
        let n = Node::Str("x".into());
        assert!(n.as_int().is_none() && n.as_real().is_none() && n.as_mat().is_none());
        assert_eq!(n.as_str(), Some("x"));
        // An int reads as a real, which is what callers expect.
        assert_eq!(Node::Int(3).as_real(), Some(3.0));
        assert_eq!(Node::Real(3.7).as_int(), Some(3));
    }

    #[test]
    fn an_8u_matrix_keeps_its_depth_rather_than_widening() {
        let m = Mat::from_vec(1, 2, 1, &[200u8, 1]).unwrap();
        let mut fs = FileStorage::new();
        fs.set_mat("m", &m).unwrap();
        let text = fs.to_string(Format::Yaml).unwrap();
        assert!(text.contains("dt: u"), "got:\n{text}");
        // And the values print without a decimal point.
        assert!(text.contains("data: [ 200, 1 ]"), "got:\n{text}");
        let back = FileStorage::from_str(&text, Format::Yaml).unwrap();
        assert_eq!(back.get("m").unwrap().as_mat().unwrap().mat_type(), CV_8UC1);
    }
}
