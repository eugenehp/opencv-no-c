//! `cv::Algorithm` — the base every OpenCV algorithm class derives from
//! (`modules/core/include/opencv2/core.hpp`, `modules/core/src/algorithm.cpp`).
//!
//! Upstream this is an abstract class with virtual `read`/`write`/`clear`;
//! here it is a trait, so a type opts in rather than inheriting. Everything
//! upstream gives a default body has one here too, which means implementing
//! [`Algorithm`] for a type that only wants `save` costs a single empty impl.

use crate::error::Result;
use crate::persistence::{FileStorage, Format, Node};

/// `cv::Param` — the runtime type tag of a named algorithm parameter.
///
/// Upstream's `Algorithm` can be driven by name (`algo->set("threshold", 0.5)`)
/// and needs a tag to dispatch on. This port has no such dynamic path — a
/// parameter is a field, and [`Algorithm::read`]/[`Algorithm::write`] carry it
/// through a [`FileStorage`] — so the enum is declared for translating C++ that
/// inspects `ParamType<T>::type` rather than for dispatching on here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Param {
    /// `Param::INT`
    Int = 0,
    /// `Param::BOOLEAN`
    Boolean = 1,
    /// `Param::REAL`
    Real = 2,
    /// `Param::STRING`
    String = 3,
    /// `Param::MAT`
    Mat = 4,
    /// `Param::MAT_VECTOR`
    MatVector = 5,
    /// `Param::ALGORITHM`
    Algorithm = 6,
    /// `Param::FLOAT`
    Float = 7,
    /// `Param::UNSIGNED_INT`
    UnsignedInt = 8,
    /// `Param::UINT64`
    Uint64 = 9,
    /// `Param::UCHAR` — note that 10 is unused upstream.
    Uchar = 11,
    /// `Param::SCALAR`
    Scalar = 12,
}

/// `cv::Algorithm`.
///
/// The name is the one hook worth overriding for most types: `getDefaultName`
/// is what [`Algorithm::save`] writes as the top-level key, and upstream's
/// readers look the object up under it.
pub trait Algorithm {
    /// `cv::Algorithm::getDefaultName()`.
    ///
    /// Upstream returns the C++ class name; a Rust implementor supplies its
    /// own, since there is no portable way to recover it from the type.
    fn default_name(&self) -> String;

    /// `cv::Algorithm::clear()` — reset to a freshly constructed state.
    ///
    /// Does nothing by default, as upstream's does.
    fn clear(&mut self) {}

    /// `cv::Algorithm::empty()` — is the object unusable as it stands?
    ///
    /// False by default, as upstream's is.
    fn empty(&self) -> bool {
        false
    }

    /// `cv::Algorithm::write(fs)` — store the parameters into `fs`.
    ///
    /// Does nothing by default; an implementor that stores nothing round-trips
    /// as an empty mapping rather than failing.
    fn write(&self, fs: &mut FileStorage) -> Result<()> {
        let _ = fs;
        Ok(())
    }

    /// `cv::Algorithm::write(fs, name)` — store under an explicit key.
    fn write_named(&self, fs: &mut FileStorage, name: &str) -> Result<()> {
        let mut inner = FileStorage::new();
        self.write(&mut inner)?;
        fs.set(name, inner.root());
        Ok(())
    }

    /// `cv::Algorithm::read(fn)` — restore parameters from a node.
    ///
    /// Does nothing by default, matching upstream.
    fn read(&mut self, node: &Node) -> Result<()> {
        let _ = node;
        Ok(())
    }

    /// `cv::Algorithm::save(filename)`.
    ///
    /// Writes the object under [`Algorithm::default_name`], which is the key
    /// upstream's `Algorithm::load` then looks for.
    fn save(&self, filename: &str) -> Result<()> {
        let mut fs = FileStorage::new();
        self.write_named(&mut fs, &self.default_name())?;
        fs.write_file(filename)
    }

    /// `cv::Algorithm::write(fs)` rendered to a string, for tests and logging.
    fn to_storage_string(&self, format: Format) -> Result<String> {
        let mut fs = FileStorage::new();
        self.write_named(&mut fs, &self.default_name())?;
        fs.to_string(format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal implementor: two parameters and a name.
    #[derive(Default)]
    struct Threshold {
        level: i64,
        label: String,
    }

    impl Algorithm for Threshold {
        fn default_name(&self) -> String {
            "opencv_threshold".to_string()
        }
        fn clear(&mut self) {
            *self = Threshold::default();
        }
        fn empty(&self) -> bool {
            self.label.is_empty()
        }
        fn write(&self, fs: &mut FileStorage) -> Result<()> {
            fs.set_int("level", self.level).set_str("label", &self.label);
            Ok(())
        }
        fn read(&mut self, node: &Node) -> Result<()> {
            self.level = node.get("level").and_then(Node::as_int).unwrap_or(0);
            self.label = node.get("label").and_then(Node::as_str).unwrap_or("").to_string();
            Ok(())
        }
    }

    #[test]
    fn an_algorithm_round_trips_through_a_storage() {
        let a = Threshold { level: 42, label: "otsu".into() };
        let text = a.to_storage_string(Format::Yaml).unwrap();
        assert!(text.contains("opencv_threshold"), "got:\n{text}");

        let fs = FileStorage::from_str(&text, Format::Yaml).unwrap();
        let node = fs.get("opencv_threshold").expect("written under the default name");
        let mut b = Threshold::default();
        b.read(node).unwrap();
        assert_eq!((b.level, b.label.as_str()), (42, "otsu"));
    }

    #[test]
    fn clear_and_empty_follow_the_implementor() {
        let mut a = Threshold { level: 7, label: "x".into() };
        assert!(!a.empty());
        a.clear();
        assert!(a.empty());
        assert_eq!(a.level, 0);
    }

    #[test]
    fn save_writes_a_file_under_the_default_name() {
        let path = std::env::temp_dir().join("cv_no_c_algorithm_save.yml");
        let a = Threshold { level: 3, label: "t".into() };
        a.save(path.to_str().unwrap()).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("opencv_threshold"), "got:\n{text}");
        std::fs::remove_file(&path).ok();
    }

    /// A type that overrides nothing still compiles and behaves as upstream's
    /// base class does.
    #[test]
    fn the_defaults_are_upstreams_defaults() {
        struct Bare;
        impl Algorithm for Bare {
            fn default_name(&self) -> String {
                "Bare".into()
            }
        }
        let mut b = Bare;
        assert!(!b.empty());
        b.clear();
        assert!(b.to_storage_string(Format::Json).unwrap().contains("Bare"));
    }
}
