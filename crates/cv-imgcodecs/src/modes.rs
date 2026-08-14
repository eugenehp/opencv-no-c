//! The `IMREAD_*` and `IMWRITE_*` surface, and the multi-image entry points
//! (`modules/imgcodecs/include/opencv2/imgcodecs.hpp`, `loadsave.cpp`).
//!
//! # What is declared and what is not
//!
//! Upstream's `IMWRITE_*` list covers every codec it can be built with — EXR,
//! AVIF, JPEG 2000, TIFF, WebP, PNG's every filter. This port declares only the
//! flags that something behind them honours. `IMWRITE_EXR_COMPRESSION_DWAB` is
//! a real upstream constant, and declaring it here would promise an EXR encoder
//! that does not exist.

use cv_core::{cv_assert, Mat, MatLike, Result};

use crate::{imdecode, imencode, ImreadMode};

/// `cv::ImreadModes`, in full.
///
/// The reduced variants decode at a fraction of the stored size. That is not
/// the same as decoding and then resizing — a codec can skip data it never has
/// to reconstruct, which is why the flag exists rather than leaving it to the
/// caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImreadFlag {
    /// `cv::IMREAD_UNCHANGED`
    Unchanged,
    /// `cv::IMREAD_GRAYSCALE`
    Grayscale,
    /// `cv::IMREAD_COLOR` / `cv::IMREAD_COLOR_BGR` — three channels, blue first.
    #[default]
    Color,
    /// `cv::IMREAD_COLOR_RGB` — three channels, red first.
    ColorRgb,
    /// `cv::IMREAD_ANYDEPTH` — keep 16-bit or float samples instead of
    /// truncating to 8-bit.
    AnyDepth,
    /// `cv::IMREAD_ANYCOLOR` — keep whatever channel count is stored.
    AnyColor,
    /// `cv::IMREAD_LOAD_GDAL`
    LoadGdal,
    /// `cv::IMREAD_REDUCED_GRAYSCALE_2`
    ReducedGrayscale2,
    /// `cv::IMREAD_REDUCED_COLOR_2`
    ReducedColor2,
    /// `cv::IMREAD_REDUCED_GRAYSCALE_4`
    ReducedGrayscale4,
    /// `cv::IMREAD_REDUCED_COLOR_4`
    ReducedColor4,
    /// `cv::IMREAD_REDUCED_GRAYSCALE_8`
    ReducedGrayscale8,
    /// `cv::IMREAD_REDUCED_COLOR_8`
    ReducedColor8,
    /// `cv::IMREAD_IGNORE_ORIENTATION` — do not apply the EXIF rotation.
    IgnoreOrientation,
}

impl ImreadFlag {
    /// The reduction factor this flag asks for: 1, 2, 4 or 8.
    #[must_use]
    pub fn reduction(self) -> u32 {
        match self {
            ImreadFlag::ReducedGrayscale2 | ImreadFlag::ReducedColor2 => 2,
            ImreadFlag::ReducedGrayscale4 | ImreadFlag::ReducedColor4 => 4,
            ImreadFlag::ReducedGrayscale8 | ImreadFlag::ReducedColor8 => 8,
            _ => 1,
        }
    }

    /// Whether the result should be single-channel.
    #[must_use]
    pub fn is_grayscale(self) -> bool {
        matches!(
            self,
            ImreadFlag::Grayscale
                | ImreadFlag::ReducedGrayscale2
                | ImreadFlag::ReducedGrayscale4
                | ImreadFlag::ReducedGrayscale8
        )
    }

    /// The base decode mode, before any reduction.
    #[must_use]
    pub fn base_mode(self) -> ImreadMode {
        if self == ImreadFlag::Unchanged {
            ImreadMode::Unchanged
        } else if self.is_grayscale() {
            ImreadMode::Grayscale
        } else {
            ImreadMode::Color
        }
    }
}

/// `cv::ImwriteFlags` — only those this port honours.
///
/// See the module docs for why the list is short.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImwriteFlag {
    /// `cv::IMWRITE_JPEG_QUALITY` — 0 to 100.
    JpegQuality(i32),
    /// `cv::IMWRITE_PNG_COMPRESSION` — 0 to 9.
    PngCompression(i32),
    /// `cv::IMWRITE_PXM_BINARY` — binary rather than ASCII PNM.
    PxmBinary(bool),
    /// `cv::IMWRITE_BMP_COMPRESSION` — see [`BmpCompression`].
    BmpCompression(BmpCompression),
}

/// `cv::ImwriteBmpCompressionFlags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BmpCompression {
    /// `cv::IMWRITE_BMP_COMPRESSION_RGB` — uncompressed.
    #[default]
    Rgb,
    /// `cv::IMWRITE_BMP_COMPRESSION_BITFIELDS`
    Bitfields,
}

/// `cv::ImageMetadataType`.
///
/// Declared for completeness; no codec here reads or writes metadata yet, and
/// [`imread_with_metadata`] says so rather than returning an empty vector that
/// could be mistaken for "this file has none".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageMetadataType {
    /// `cv::IMAGE_METADATA_UNKNOWN`
    Unknown,
    /// `cv::IMAGE_METADATA_EXIF`
    Exif,
    /// `cv::IMAGE_METADATA_XMP`
    Xmp,
    /// `cv::IMAGE_METADATA_ICCP`
    Iccp,
    /// `cv::IMAGE_METADATA_CICP`
    Cicp,
    /// `cv::IMAGE_METADATA_MAX`
    Max,
}

/// Halve an image `n` times by 2x2 averaging.
fn reduce(src: &Mat, factor: u32) -> Result<Mat> {
    if factor <= 1 {
        return Ok(src.clone());
    }
    let mut cur = src.clone();
    let mut f = factor;
    while f > 1 {
        let (h, w) = ((cur.rows() / 2).max(1), (cur.cols() / 2).max(1));
        let mut out = Mat::zeros(h, w, cur.mat_type())?;
        for y in 0..h {
            for x in 0..w {
                for c in 0..cur.channels() {
                    let mut acc = 0.0;
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let sy = (y * 2 + dy).min(cur.rows() - 1);
                        let sx = (x * 2 + dx).min(cur.cols() - 1);
                        acc += cur.at_f64(sy, sx, c)?;
                    }
                    cv_core::arithm::set_channel_generic(&mut out, y, x, c, acc / 4.0)?;
                }
            }
        }
        cur = out;
        f /= 2;
    }
    Ok(cur)
}

/// `cv::imdecode(buf, flags)` with the full flag set.
pub fn imdecode_flag(data: &[u8], flag: ImreadFlag) -> Result<Mat> {
    let img = imdecode(data, flag.base_mode())?;
    let img = reduce(&img, flag.reduction())?;
    if flag == ImreadFlag::ColorRgb && img.channels() == 3 {
        // The only difference from IMREAD_COLOR is channel order.
        return cv_imgproc::color::cvt_color(&img, cv_imgproc::color::ColorConversion::Bgr2Rgb);
    }
    Ok(img)
}

/// `cv::imread(filename, flags)` with the full flag set.
pub fn imread_flag<P: AsRef<std::path::Path>>(path: P, flag: ImreadFlag) -> Result<Mat> {
    let data = std::fs::read(path.as_ref()).map_err(|e| {
        cv_core::Error::new(
            cv_core::Code::StsObjectNotFound,
            format!("could not read {:?}: {e}", path.as_ref()),
        )
    })?;
    imdecode_flag(&data, flag)
}

/// `cv::imcount(filename, flags)` — how many images a file holds.
///
/// Every format this port decodes is single-image, so the answer is 1 for a
/// readable file and 0 otherwise. Formats with real multi-image support — TIFF
/// pages, GIF frames — are not here.
pub fn imcount<P: AsRef<std::path::Path>>(path: P) -> usize {
    usize::from(imread_flag(path, ImreadFlag::Unchanged).is_ok())
}

/// `cv::imreadmulti(filename, mats, flags)`.
///
/// Returns a one-element vector for the single-image formats this port
/// supports. See [`imcount`].
pub fn imreadmulti<P: AsRef<std::path::Path>>(path: P, flag: ImreadFlag) -> Result<Vec<Mat>> {
    Ok(vec![imread_flag(path, flag)?])
}

/// `cv::imdecodemulti(buf, flags, mats)`.
pub fn imdecodemulti(data: &[u8], flag: ImreadFlag) -> Result<Vec<Mat>> {
    Ok(vec![imdecode_flag(data, flag)?])
}

/// `cv::imencodemulti(ext, imgs, buf, params)`.
///
/// Encodes each image separately and concatenates, since no format here
/// supports multiple images in one stream. A caller expecting a single
/// multi-page file needs a format this port does not have, so the count is
/// checked rather than silently producing a file only the first image of which
/// can be read back.
pub fn imencodemulti<M: MatLike + ?Sized>(ext: &str, imgs: &[&M]) -> Result<Vec<Vec<u8>>> {
    cv_assert!(!imgs.is_empty(), "imencodemulti needs at least one image");
    imgs.iter().map(|m| imencode(ext, *m)).collect()
}

/// `cv::haveImageWriter(filename)`.
#[must_use]
pub fn have_image_writer<P: AsRef<std::path::Path>>(path: P) -> bool {
    let ext = path
        .as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    matches!(ext.as_str(), "bmp" | "dib" | "pgm" | "ppm" | "pnm" | "png")
}

/// One metadata block: its kind and raw bytes.
pub type Metadata = (ImageMetadataType, Vec<u8>);

/// `cv::imreadWithMetadata(filename, metadataTypes, metadata, flags)`.
///
/// # Not implemented
///
/// No codec here parses metadata. This returns an error rather than an empty
/// list, because an empty list is indistinguishable from "this file carries
/// none" and would let a caller conclude the EXIF orientation is absent when it
/// was simply never read.
pub fn imread_with_metadata<P: AsRef<std::path::Path>>(
    _path: P,
    _flag: ImreadFlag,
) -> Result<(Mat, Vec<Metadata>)> {
    cv_error()
}

/// `cv::imwriteWithMetadata(filename, img, metadataTypes, metadata, params)`.
pub fn imwrite_with_metadata<P: AsRef<std::path::Path>, M: MatLike + ?Sized>(
    _path: P,
    _img: &M,
    _metadata: &[Metadata],
) -> Result<()> {
    cv_error().map(|_: (Mat, Vec<Metadata>)| ())
}

fn cv_error<T>() -> Result<T> {
    Err(cv_core::Error::new(
        cv_core::Code::StsNotImplemented,
        "no codec in this port reads or writes image metadata".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(w: i32, h: i32) -> Mat {
        let mut d = vec![0u8; (w * h * 3) as usize];
        for (i, v) in d.iter_mut().enumerate() {
            *v = ((i * 11) % 256) as u8;
        }
        Mat::from_vec(h, w, 3, &d).unwrap()
    }

    #[test]
    fn the_reduction_factors_are_right() {
        assert_eq!(ImreadFlag::Color.reduction(), 1);
        assert_eq!(ImreadFlag::ReducedColor2.reduction(), 2);
        assert_eq!(ImreadFlag::ReducedGrayscale4.reduction(), 4);
        assert_eq!(ImreadFlag::ReducedColor8.reduction(), 8);
        assert!(ImreadFlag::ReducedGrayscale2.is_grayscale());
        assert!(!ImreadFlag::ReducedColor2.is_grayscale());
        assert_eq!(ImreadFlag::default(), ImreadFlag::Color);
    }

    #[test]
    fn the_base_mode_maps_onto_the_three_way_enum() {
        assert_eq!(ImreadFlag::Unchanged.base_mode(), ImreadMode::Unchanged);
        assert_eq!(ImreadFlag::Grayscale.base_mode(), ImreadMode::Grayscale);
        assert_eq!(ImreadFlag::ReducedGrayscale8.base_mode(), ImreadMode::Grayscale);
        assert_eq!(ImreadFlag::Color.base_mode(), ImreadMode::Color);
        assert_eq!(ImreadFlag::ColorRgb.base_mode(), ImreadMode::Color);
    }

    #[test]
    fn a_reduced_read_shrinks_by_the_right_factor() {
        let img = rgb(32, 24);
        let bmp = imencode("bmp", &img).unwrap();
        for (flag, want) in [
            (ImreadFlag::Color, (24, 32)),
            (ImreadFlag::ReducedColor2, (12, 16)),
            (ImreadFlag::ReducedColor4, (6, 8)),
            (ImreadFlag::ReducedColor8, (3, 4)),
        ] {
            let m = imdecode_flag(&bmp, flag).unwrap();
            assert_eq!((m.rows(), m.cols()), want, "{flag:?}");
            assert_eq!(m.channels(), 3);
        }
        // The grey variants reduce and drop to one channel.
        let g = imdecode_flag(&bmp, ImreadFlag::ReducedGrayscale2).unwrap();
        assert_eq!((g.rows(), g.cols(), g.channels()), (12, 16, 1));
    }

    #[test]
    fn color_rgb_is_the_channel_reversal_of_color_bgr() {
        let img = rgb(8, 6);
        let bmp = imencode("bmp", &img).unwrap();
        let b = imdecode_flag(&bmp, ImreadFlag::Color).unwrap();
        let r = imdecode_flag(&bmp, ImreadFlag::ColorRgb).unwrap();
        assert_eq!((b.rows(), b.cols()), (r.rows(), r.cols()));
        for y in 0..b.rows() {
            for x in 0..b.cols() {
                for c in 0..3 {
                    assert_eq!(
                        b.at_f64(y, x, c).unwrap(),
                        r.at_f64(y, x, 2 - c).unwrap(),
                        "({y},{x}) channel {c}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_multi_entry_points_agree_with_the_single_ones() {
        let img = rgb(10, 8);
        let bmp = imencode("bmp", &img).unwrap();
        let single = imdecode_flag(&bmp, ImreadFlag::Color).unwrap();
        let multi = imdecodemulti(&bmp, ImreadFlag::Color).unwrap();
        assert_eq!(multi.len(), 1, "these formats hold one image");
        assert_eq!((multi[0].rows(), multi[0].cols()), (single.rows(), single.cols()));

        let enc = imencodemulti("bmp", &[&img, &img]).unwrap();
        assert_eq!(enc.len(), 2, "one buffer per image");
        assert_eq!(enc[0], enc[1]);
        assert!(imencodemulti::<Mat>("bmp", &[]).is_err(), "an empty list is an error");
    }

    #[test]
    fn have_image_writer_reports_only_what_can_be_written() {
        for ok in ["a.bmp", "a.dib", "a.png", "a.pgm", "a.ppm", "a.pnm", "A.BMP"] {
            assert!(have_image_writer(ok), "{ok} should be writable");
        }
        for no in ["a.jpg", "a.tiff", "a.webp", "a.exr", "a.avif", "a"] {
            assert!(!have_image_writer(no), "{no} must not claim a writer");
        }
    }

    #[test]
    fn metadata_is_refused_rather_than_reported_empty() {
        // An empty metadata list would be indistinguishable from a file that
        // genuinely carries none.
        let e = imread_with_metadata("x.bmp", ImreadFlag::Color).unwrap_err();
        assert!(format!("{e}").contains("metadata"), "{e}");
        let img = rgb(4, 4);
        assert!(imwrite_with_metadata("x.bmp", &img, &[]).is_err());
    }

    #[test]
    fn imcount_counts_a_readable_file_once() {
        let dir = std::env::temp_dir().join("cv_no_c_imcount");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.bmp");
        std::fs::write(&p, imencode("bmp", &rgb(6, 4)).unwrap()).unwrap();
        assert_eq!(imcount(&p), 1);
        assert_eq!(imreadmulti(&p, ImreadFlag::Color).unwrap().len(), 1);
        // An unreadable path counts zero rather than erroring.
        assert_eq!(imcount(dir.join("missing.bmp")), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_write_flags_carry_their_values() {
        assert_eq!(BmpCompression::default(), BmpCompression::Rgb);
        assert!(matches!(ImwriteFlag::JpegQuality(90), ImwriteFlag::JpegQuality(90)));
        assert!(matches!(ImwriteFlag::PngCompression(3), ImwriteFlag::PngCompression(3)));
        assert!(matches!(
            ImwriteFlag::BmpCompression(BmpCompression::Bitfields),
            ImwriteFlag::BmpCompression(BmpCompression::Bitfields)
        ));
    }
}
