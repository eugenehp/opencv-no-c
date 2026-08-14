//! The images the samples read, embedded.
//!
//! Every byte here is compiled into the binary, so nothing in this crate ever
//! touches a file system and the whole set runs unchanged under WebAssembly.
//!
//! The files are PNG, converted from `samples/data/` with the reference
//! library. That conversion is lossless — the PNG holds the exact pixels the
//! reference library decodes the original into — and it is deliberate: this
//! crate's PNG decoder is bit-exact against the reference and its JPEG decoder
//! is not, so reading a `.jpg` would inject a decode error of one to two
//! levels into the input of every sample that uses one.

use cv_core::{Code, Error, Mat, Result};
use cv_imgcodecs::{imdecode, ImreadMode};

/// One embedded image.
pub struct Asset {
    /// The name upstream uses, without an extension.
    pub name: &'static str,
    /// The original file in `samples/data/`.
    pub upstream: &'static str,
    /// The PNG bytes.
    pub bytes: &'static [u8],
}

impl core::fmt::Debug for Asset {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Asset")
            .field("name", &self.name)
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

/// Every embedded image.
pub static ASSETS: &[Asset] = &[
    Asset {
        name: "fruits",
        upstream: "samples/data/fruits.jpg",
        bytes: include_bytes!("../assets/fruits.png"),
    },
    Asset {
        name: "baboon",
        upstream: "samples/data/baboon.jpg",
        bytes: include_bytes!("../assets/baboon.png"),
    },
    Asset {
        name: "stuff",
        upstream: "samples/data/stuff.jpg",
        bytes: include_bytes!("../assets/stuff.png"),
    },
    Asset { name: "pic1", upstream: "samples/data/pic1.png", bytes: include_bytes!("../assets/pic1.png") },
    Asset { name: "pic2", upstream: "samples/data/pic2.png", bytes: include_bytes!("../assets/pic2.png") },
    Asset { name: "pic3", upstream: "samples/data/pic3.png", bytes: include_bytes!("../assets/pic3.png") },
    Asset { name: "pic4", upstream: "samples/data/pic4.png", bytes: include_bytes!("../assets/pic4.png") },
    Asset { name: "pic5", upstream: "samples/data/pic5.png", bytes: include_bytes!("../assets/pic5.png") },
    Asset { name: "pic6", upstream: "samples/data/pic6.png", bytes: include_bytes!("../assets/pic6.png") },
];

/// The raw bytes of an embedded image.
///
/// # Errors
///
/// Fails if no asset is registered under that name.
pub fn bytes(name: &str) -> Result<&'static [u8]> {
    ASSETS
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.bytes)
        .ok_or_else(|| Error::new(Code::StsObjectNotFound, format!("no embedded asset named {name}")))
}

/// Decode an embedded image in the given mode — `imread` without a file.
///
/// # Errors
///
/// Fails if the name is unknown or the image cannot be decoded.
pub fn load(name: &str, mode: ImreadMode) -> Result<Mat> {
    imdecode(bytes(name)?, mode)
}

/// `imread(name, IMREAD_COLOR)`.
///
/// # Errors
///
/// As [`load`].
pub fn color(name: &str) -> Result<Mat> {
    load(name, ImreadMode::Color)
}

/// `imread(name, IMREAD_GRAYSCALE)`.
///
/// # Errors
///
/// As [`load`].
pub fn gray(name: &str) -> Result<Mat> {
    load(name, ImreadMode::Grayscale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::MatLike;

    #[test]
    fn every_asset_decodes_and_is_named_once() {
        let mut seen = std::collections::BTreeSet::new();
        for a in ASSETS {
            assert!(seen.insert(a.name), "{} is registered twice", a.name);
            let m = load(a.name, ImreadMode::Unchanged).unwrap_or_else(|e| {
                panic!("{} ({}) did not decode: {e}", a.name, a.upstream)
            });
            assert!(m.rows() > 0 && m.cols() > 0, "{} decoded to nothing", a.name);
        }
    }

    #[test]
    fn the_three_read_modes_agree_on_size() {
        let (c, g) = (color("pic1").unwrap(), gray("pic1").unwrap());
        assert_eq!(c.size(), g.size());
        assert_eq!(c.channels(), 3);
        assert_eq!(g.channels(), 1);
    }

    #[test]
    fn an_unknown_name_is_an_error_not_a_panic() {
        assert!(bytes("no-such-asset").is_err());
    }
}
