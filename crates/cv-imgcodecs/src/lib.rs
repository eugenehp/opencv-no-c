//! Pure-Rust port of the OpenCV `imgcodecs` module.
//!
//! Targets upstream **OpenCV 4.15.0-dev** (`77dfa297d08fdecdc509fc01ad92a2e9ec776a57`).
//!
//! # Format coverage
//!
//! Only the formats that need no compression codec are implemented: **BMP**
//! (uncompressed 8/24/32-bit) and **PNM** (`P2`/`P3`/`P5`/`P6`). PNG, JPEG,
//! TIFF and WebP all require a deflate, DCT or VP8 implementation, which is a
//! separate body of work; [`imread`] returns `StsUnsupportedFormat` for them
//! rather than silently producing something wrong.
//!
//! # Channel order
//!
//! PNM stores samples as RGB, while OpenCV's `Mat` convention is BGR. The
//! reader and writer swap accordingly, so a PPM round-trip through this crate
//! preserves colours as OpenCV would.

#![doc(html_root_url = "https://docs.rs/cv-imgcodecs/0.1.0")]
#![warn(missing_docs)]

pub mod animation;
pub mod constants;
pub mod modes;

pub mod jpeg;
pub mod bmp;
pub mod deflate;
pub mod png;
pub mod pnm;

use std::path::Path;

use cv_core::{cv_error, Code, Depth, Mat, MatLike, Result};

/// `cv::ImreadModes` — the subset ported so far, with upstream's values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ImreadMode {
    /// `IMREAD_UNCHANGED` — return the image as stored.
    Unchanged = -1,
    /// `IMREAD_GRAYSCALE` — always convert to a single channel.
    Grayscale = 0,
    /// `IMREAD_COLOR` — always convert to 3-channel BGR. This is the default.
    Color = 1,
}

/// Which container a byte buffer holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Bmp,
    Pnm,
    Png,
    /// Baseline JPEG.
    Jpeg,
}

/// Identify a format from its magic bytes.
fn sniff(data: &[u8]) -> Result<Format> {
    if data.len() >= 2 && &data[..2] == b"BM" {
        return Ok(Format::Bmp);
    }
    if data.len() >= 2 && data[0] == b'P' && (b'1'..=b'6').contains(&data[1]) {
        return Ok(Format::Pnm);
    }
    if png::is_png(data) {
        return Ok(Format::Png);
    }
    if jpeg::is_jpeg(data) {
        return Ok(Format::Jpeg);
    }
    // Name the format we recognised but cannot decode, so the error is useful.
    let known: &[(&[u8], &str)] = &[
        (b"GIF8", "GIF"),
        (b"II*\x00", "TIFF"),
        (b"MM\x00*", "TIFF"),
        (b"RIFF", "WebP"),
    ];
    for (magic, name) in known {
        if data.starts_with(magic) {
            return cv_error!(
                Code::StsUnsupportedFormat,
                "{name} decoding is not ported yet (needs a compression codec)"
            );
        }
    }
    cv_error!(Code::StsError, "unrecognised image format")
}

/// `cv::imdecode(buf, flags)`.
pub fn imdecode(data: &[u8], mode: ImreadMode) -> Result<Mat> {
    let img = match sniff(data)? {
        Format::Bmp => bmp::decode(data)?,
        Format::Pnm => pnm::decode(data)?,
        Format::Png => png::decode(data)?,
        Format::Jpeg => jpeg::decode(data)?,
    };
    apply_mode(img, mode)
}

/// `cv::imread(filename, flags)`.
pub fn imread<P: AsRef<Path>>(path: P, mode: ImreadMode) -> Result<Mat> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(|e| {
        cv_core::Error::new(Code::StsError, format!("could not read {}: {e}", path.display()))
    })?;
    imdecode(&data, mode)
}

/// Convert a freshly decoded image to the channel count `mode` requires.
fn apply_mode(img: Mat, mode: ImreadMode) -> Result<Mat> {
    use cv_imgproc::{cvt_color, ColorConversion};
    let cn = img.channels();
    Ok(match (mode, cn) {
        (ImreadMode::Unchanged, _) => img,
        (ImreadMode::Grayscale, 1) => img,
        (ImreadMode::Grayscale, 3) => cvt_color(&img, ColorConversion::Bgr2Gray)?,
        (ImreadMode::Grayscale, 4) => cvt_color(&img, ColorConversion::Bgra2Gray)?,
        (ImreadMode::Color, 1) => cvt_color(&img, ColorConversion::Gray2Bgr)?,
        (ImreadMode::Color, 3) => img,
        (ImreadMode::Color, 4) => cvt_color(&img, ColorConversion::Bgra2Bgr)?,
        (_, n) => {
            return cv_error!(Code::StsUnsupportedFormat, "cannot convert a {n}-channel image")
        }
    })
}

/// `cv::imencode(ext, img, buf)`.
///
/// `ext` is an extension such as `".bmp"`, `".pgm"` or `".ppm"`, with or
/// without the leading dot.
pub fn imencode<M: MatLike + ?Sized>(ext: &str, img: &M) -> Result<Vec<u8>> {
    if img.depth() != Depth::U8 {
        return cv_error!(
            Code::StsUnsupportedFormat,
            "only CV_8U images can be encoded, got {}",
            img.depth().name()
        );
    }
    if img.is_empty() {
        return cv_error!(Code::StsBadSize, "cannot encode an empty image");
    }
    let e = ext.trim_start_matches('.').to_ascii_lowercase();
    match e.as_str() {
        "bmp" | "dib" => bmp::encode(img),
        "pgm" | "ppm" | "pnm" => pnm::encode(img, true),
        "png" => png::encode(img),
        "pbm" => cv_error!(Code::StsUnsupportedFormat, "PBM (1-bit) encoding is not ported"),
        "jpg" | "jpeg" | "tif" | "tiff" | "webp" => cv_error!(
            Code::StsUnsupportedFormat,
            "{e} encoding is not ported yet (needs a compression codec)"
        ),
        _ => cv_error!(Code::StsUnsupportedFormat, "unknown extension {ext:?}"),
    }
}

/// `cv::imwrite(filename, img)` — the format is chosen from the extension.
pub fn imwrite<P: AsRef<Path>, M: MatLike + ?Sized>(path: P, img: &M) -> Result<()> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| cv_core::Error::new(Code::StsBadArg, "path has no extension".to_string()))?;
    let data = imencode(ext, img)?;
    std::fs::write(path, data).map_err(|e| {
        cv_core::Error::new(Code::StsError, format!("could not write {}: {e}", path.display()))
    })
}

/// `cv::haveImageReader(filename)` — whether this build can decode the file.
pub fn have_image_reader<P: AsRef<Path>>(path: P) -> bool {
    std::fs::read(path.as_ref()).map(|d| sniff(&d).is_ok()).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{types::CV_8UC1, types::CV_8UC3, MatLike, MatLikeMut, Scalar};

    fn gradient_bgr() -> Mat {
        let mut m = Mat::zeros(5, 7, CV_8UC3).unwrap();
        for y in 0..5 {
            for x in 0..7 {
                let p = m.pixel_mut::<u8>(y, x).unwrap();
                p[0] = (x * 30) as u8;
                p[1] = (y * 50) as u8;
                p[2] = 200;
            }
        }
        m
    }

    #[test]
    fn mode_values_match_opencv() {
        assert_eq!(ImreadMode::Unchanged as i32, -1);
        assert_eq!(ImreadMode::Grayscale as i32, 0);
        assert_eq!(ImreadMode::Color as i32, 1);
    }

    #[test]
    fn bmp_roundtrip_preserves_pixels() {
        let m = gradient_bgr();
        let buf = imencode(".bmp", &m).unwrap();
        let back = imdecode(&buf, ImreadMode::Unchanged).unwrap();
        assert_eq!(back.size(), m.size());
        assert_eq!(back.channels(), 3);
        assert_eq!(back.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn ppm_roundtrip_preserves_bgr_order() {
        // The PPM file itself is RGB; a round-trip must restore BGR.
        let m = gradient_bgr();
        let buf = imencode(".ppm", &m).unwrap();
        let back = imdecode(&buf, ImreadMode::Unchanged).unwrap();
        assert_eq!(back.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
        assert_eq!(back.pixel::<u8>(1, 2).unwrap(), m.pixel::<u8>(1, 2).unwrap());
    }

    #[test]
    fn pgm_roundtrip_preserves_grayscale() {
        let data: Vec<u8> = (0..35).map(|i| (i * 7) as u8).collect();
        let m = Mat::from_vec(5, 7, 1, &data).unwrap();
        let buf = imencode(".pgm", &m).unwrap();
        let back = imdecode(&buf, ImreadMode::Unchanged).unwrap();
        assert_eq!(back.channels(), 1);
        assert_eq!(back.to_vec::<u8>().unwrap(), data);
    }

    #[test]
    fn bmp_handles_widths_needing_row_padding() {
        // 24-bit BMP rows pad to a 4-byte boundary; widths 1..4 exercise every
        // padding amount.
        for w in 1..=8 {
            let m = Mat::new_with(3, w, CV_8UC3, Scalar::new(1.0, 2.0, 3.0, 0.0)).unwrap();
            let buf = imencode(".bmp", &m).unwrap();
            let back = imdecode(&buf, ImreadMode::Unchanged).unwrap();
            assert_eq!(back.size(), m.size(), "width {w}");
            assert_eq!(back.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap(), "width {w}");
        }
    }

    #[test]
    fn grayscale_bmp_roundtrips_through_a_palette() {
        let data: Vec<u8> = (0..12).map(|i| (i * 20) as u8).collect();
        let m = Mat::from_vec(3, 4, 1, &data).unwrap();
        let buf = imencode(".bmp", &m).unwrap();
        let back = imdecode(&buf, ImreadMode::Unchanged).unwrap();
        assert_eq!(back.channels(), 1);
        assert_eq!(back.to_vec::<u8>().unwrap(), data);
    }

    #[test]
    fn imread_dispatches_to_the_jpeg_decoder() {
        let path = std::path::Path::new("/tmp/jpegtest/color.jpg");
        if !path.exists() {
            eprintln!("skipping: run python3 tools/gen_jpeg_fixture.py first");
            return;
        }
        let m = imread(path, ImreadMode::Color).unwrap();
        assert_eq!((m.rows(), m.cols(), m.channels()), (32, 32, 3));
        // And greyscale conversion still applies on top of it.
        let g = imread(path, ImreadMode::Grayscale).unwrap();
        assert_eq!(g.channels(), 1);
    }

    #[test]
    fn imread_mode_converts_channels() {
        let m = gradient_bgr();
        let buf = imencode(".bmp", &m).unwrap();

        let gray = imdecode(&buf, ImreadMode::Grayscale).unwrap();
        assert_eq!(gray.channels(), 1);
        let color = imdecode(&buf, ImreadMode::Color).unwrap();
        assert_eq!(color.channels(), 3);

        // A grayscale source read as colour becomes 3 identical channels.
        let g = Mat::from_vec(2, 2, 1, &[10u8, 20, 30, 40]).unwrap();
        let gbuf = imencode(".pgm", &g).unwrap();
        let as_color = imdecode(&gbuf, ImreadMode::Color).unwrap();
        assert_eq!(as_color.channels(), 3);
        assert_eq!(as_color.pixel::<u8>(0, 0).unwrap(), &[10, 10, 10]);
    }

    #[test]
    fn unsupported_formats_are_named_not_guessed() {
        // Formats we recognise but cannot decode must say which one they are,
        // rather than falling through to a generic parse failure.
        //
        // JPEG is no longer in that set — it decodes — so a malformed one is a
        // *parse* error naming JPEG, not an "unsupported format".
        let truncated_jpeg = b"\xFF\xD8\xFF\xE0 rest";
        let e = imdecode(truncated_jpeg, ImreadMode::Color).unwrap_err();
        assert_eq!(e.code, Code::StsParseError);
        assert!(e.msg.contains("JPEG"), "{}", e.msg);

        let tiff = b"II*\x00 rest";
        assert!(imdecode(tiff, ImreadMode::Color).unwrap_err().msg.contains("TIFF"));

        assert_eq!(imencode(".jpg", &gradient_bgr()).unwrap_err().code, Code::StsUnsupportedFormat);
    }

    #[test]
    fn png_roundtrips_through_the_dispatcher() {
        let m = gradient_bgr();
        let buf = imencode(".png", &m).unwrap();
        assert_eq!(&buf[1..4], b"PNG");
        let back = imdecode(&buf, ImreadMode::Unchanged).unwrap();
        assert_eq!(back.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());

        // And through the mode conversions.
        assert_eq!(imdecode(&buf, ImreadMode::Grayscale).unwrap().channels(), 1);
        assert_eq!(imdecode(&buf, ImreadMode::Color).unwrap().channels(), 3);
    }

    #[test]
    fn a_truncated_png_errors_rather_than_panics() {
        let buf = imencode(".png", &gradient_bgr()).unwrap();
        for cut in [8, 20, 33, buf.len() / 2, buf.len() - 4] {
            assert!(imdecode(&buf[..cut], ImreadMode::Unchanged).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn garbage_input_is_rejected() {
        assert_eq!(imdecode(b"not an image", ImreadMode::Color).unwrap_err().code, Code::StsError);
        assert!(imdecode(b"", ImreadMode::Color).is_err());
    }

    #[test]
    fn truncated_files_error_rather_than_panic() {
        let m = gradient_bgr();
        let buf = imencode(".bmp", &m).unwrap();
        for cut in [2, 10, 20, 40, buf.len() / 2] {
            assert!(imdecode(&buf[..cut], ImreadMode::Unchanged).is_err(), "cut at {cut}");
        }
        let pbuf = imencode(".ppm", &m).unwrap();
        for cut in [3, 8, pbuf.len() / 2] {
            assert!(imdecode(&pbuf[..cut], ImreadMode::Unchanged).is_err(), "ppm cut at {cut}");
        }
    }

    #[test]
    fn non_8u_images_are_rejected() {
        let f = Mat::zeros(2, 2, cv_core::types::CV_32FC1).unwrap();
        assert_eq!(imencode(".bmp", &f).unwrap_err().code, Code::StsUnsupportedFormat);
    }

    #[test]
    fn empty_images_are_rejected() {
        let e = Mat::zeros(0, 0, CV_8UC1).unwrap();
        assert_eq!(imencode(".bmp", &e).unwrap_err().code, Code::StsBadSize);
    }

    #[test]
    fn file_roundtrip_through_the_filesystem() {
        let mut path = std::env::temp_dir();
        path.push("opencv_no_c_imgcodecs_test.bmp");
        let m = gradient_bgr();
        imwrite(&path, &m).unwrap();
        assert!(have_image_reader(&path));
        let back = imread(&path, ImreadMode::Unchanged).unwrap();
        assert_eq!(back.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_reports_an_error() {
        let e = imread("/nonexistent/path/to/image.bmp", ImreadMode::Color).unwrap_err();
        assert_eq!(e.code, Code::StsError);
    }

    #[test]
    fn encoding_accepts_a_strided_view() {
        let m = gradient_bgr();
        let v = m.roi(cv_core::Rect::new(1, 1, 3, 3)).unwrap();
        let buf = imencode(".bmp", &v).unwrap();
        let back = imdecode(&buf, ImreadMode::Unchanged).unwrap();
        assert_eq!(back.size(), cv_core::Size::new(3, 3));
        assert_eq!(back.pixel::<u8>(0, 0).unwrap(), m.pixel::<u8>(1, 1).unwrap());
    }
}
