//! Netpbm (PNM) encoder and decoder (`modules/imgcodecs/src/grfmt_pxm.cpp`).
//!
//! Supports `P2`/`P5` (grayscale) and `P3`/`P6` (colour), in both ASCII and
//! binary form, with `maxval` up to 65535. `P1`/`P4` (1-bit bitmaps) are not
//! ported.
//!
//! PNM samples are **RGB**; `Mat` is **BGR**. The swap happens here so callers
//! never see the difference.

use cv_core::{cv_error, Code, Depth, Mat, MatLike, MatType, Result};

/// A byte cursor that skips PNM whitespace and `#` comments.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Advance past whitespace and comment lines.
    fn skip_filler(&mut self) {
        while self.pos < self.data.len() {
            let c = self.data[self.pos];
            if c.is_ascii_whitespace() {
                self.pos += 1;
            } else if c == b'#' {
                // Comments run to end of line and may appear between any tokens.
                while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    /// Read the next whitespace-delimited token.
    fn token(&mut self) -> Result<&'a [u8]> {
        self.skip_filler();
        let start = self.pos;
        while self.pos < self.data.len() && !self.data[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        if start == self.pos {
            return cv_error!(Code::StsError, "unexpected end of PNM header");
        }
        Ok(&self.data[start..self.pos])
    }

    /// Read the next token as a non-negative integer.
    fn int(&mut self) -> Result<i64> {
        let t = self.token()?;
        core::str::from_utf8(t)
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .filter(|v| *v >= 0)
            .ok_or_else(|| {
                cv_core::Error::new(Code::StsError, "malformed integer in PNM header".to_string())
            })
    }
}

/// Decode a PNM image into `CV_8UC1`/`CV_8UC3` (or the 16-bit equivalents).
pub fn decode(data: &[u8]) -> Result<Mat> {
    let mut r = Reader::new(data);
    let magic = r.token()?;
    if magic.len() != 2 || magic[0] != b'P' {
        return cv_error!(Code::StsError, "not a PNM file");
    }
    let (channels, binary) = match magic[1] {
        b'2' => (1, false),
        b'3' => (3, false),
        b'5' => (1, true),
        b'6' => (3, true),
        b'1' | b'4' => {
            return cv_error!(Code::StsUnsupportedFormat, "PBM (1-bit) decoding is not ported")
        }
        c => return cv_error!(Code::StsError, "unknown PNM magic P{}", c as char),
    };

    let width = r.int()?;
    let height = r.int()?;
    let maxval = r.int()?;
    if width <= 0 || height <= 0 {
        return cv_error!(Code::StsBadSize, "invalid PNM dimensions {width}x{height}");
    }
    if maxval == 0 || maxval > 65535 {
        return cv_error!(Code::StsError, "PNM maxval {maxval} out of range 1..=65535");
    }
    let wide = maxval > 255;
    let depth = if wide { Depth::U16 } else { Depth::U8 };
    let (w, h) = (width as i32, height as i32);
    let mut img = Mat::zeros(h, w, MatType::new(depth, channels))?;

    let n = (w as usize) * (h as usize) * (channels as usize);
    let mut samples: Vec<u32> = Vec::with_capacity(n);

    if binary {
        // Exactly one whitespace byte separates the header from the raster.
        let start = r.pos + 1;
        let bytes_per = if wide { 2 } else { 1 };
        let need = n * bytes_per;
        let raster = data.get(start..start + need).ok_or_else(|| {
            cv_core::Error::new(
                Code::StsError,
                format!("truncated PNM raster: need {need} bytes"),
            )
        })?;
        if wide {
            // Binary PGM/PPM store 16-bit samples big-endian.
            samples.extend(raster.chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]]) as u32));
        } else {
            samples.extend(raster.iter().map(|&b| b as u32));
        }
    } else {
        for _ in 0..n {
            samples.push(r.int()? as u32);
        }
    }

    for y in 0..h {
        let base = y as usize * w as usize * channels as usize;
        for x in 0..w as usize {
            let o = base + x * channels as usize;
            for c in 0..channels as usize {
                // PNM is RGB; reverse into Mat's BGR order.
                let src_c = if channels == 3 { 2 - c } else { c };
                let v = samples[o + src_c] as f64;
                cv_core::arithm::set_channel(&mut img, y, x as i32, c as i32, v)?;
            }
        }
    }
    Ok(img)
}

/// Encode a `CV_8U` matrix as PGM (1 channel) or PPM (3 channels).
///
/// `binary` selects `P5`/`P6` over the ASCII `P2`/`P3`.
pub fn encode<M: MatLike + ?Sized>(img: &M, binary: bool) -> Result<Vec<u8>> {
    let cn = img.channels();
    if !(cn == 1 || cn == 3) {
        return cv_error!(Code::StsUnsupportedFormat, "PNM needs 1 or 3 channels, got {cn}");
    }
    let magic = match (cn, binary) {
        (1, true) => "P5",
        (3, true) => "P6",
        (1, false) => "P2",
        _ => "P3",
    };

    let (w, h) = (img.cols(), img.rows());
    let mut out = Vec::new();
    out.extend_from_slice(format!("{magic}\n{w} {h}\n255\n").as_bytes());

    if binary {
        for y in 0..h {
            let row = img.row_as::<u8>(y)?;
            for x in 0..w as usize {
                let o = x * cn as usize;
                if cn == 1 {
                    out.push(row[o]);
                } else {
                    // Mat is BGR; PNM wants RGB.
                    out.extend_from_slice(&[row[o + 2], row[o + 1], row[o]]);
                }
            }
        }
    } else {
        for y in 0..h {
            let row = img.row_as::<u8>(y)?;
            let mut line = String::new();
            for x in 0..w as usize {
                let o = x * cn as usize;
                if cn == 1 {
                    line.push_str(&format!("{} ", row[o]));
                } else {
                    line.push_str(&format!("{} {} {} ", row[o + 2], row[o + 1], row[o]));
                }
            }
            line.push('\n');
            out.extend_from_slice(line.as_bytes());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{types::CV_8UC3, Scalar, Size};

    fn rgbish() -> Mat {
        // BGR values chosen so a channel swap is immediately visible.
        Mat::from_vec(1, 2, 3, &[1u8, 2, 3, 4, 5, 6]).unwrap()
    }

    #[test]
    fn binary_ppm_stores_rgb_order() {
        let m = rgbish();
        let b = encode(&m, true).unwrap();
        let header_end = b.iter().position(|&c| c == b'\n').unwrap();
        assert_eq!(&b[..header_end], b"P6");
        // Raster must be RGB: (3,2,1) then (6,5,4).
        let raster = &b[b.len() - 6..];
        assert_eq!(raster, &[3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn ppm_roundtrip_restores_bgr() {
        let m = rgbish();
        let back = decode(&encode(&m, true).unwrap()).unwrap();
        assert_eq!(back.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn ascii_and_binary_agree() {
        let m = Mat::new_with(3, 4, CV_8UC3, Scalar::new(9.0, 80.0, 200.0, 0.0)).unwrap();
        let a = decode(&encode(&m, false).unwrap()).unwrap();
        let b = decode(&encode(&m, true).unwrap()).unwrap();
        assert_eq!(a.to_vec::<u8>().unwrap(), b.to_vec::<u8>().unwrap());
        assert_eq!(a.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn pgm_roundtrip() {
        let data: Vec<u8> = (0..12).map(|i| (i * 21) as u8).collect();
        let m = Mat::from_vec(3, 4, 1, &data).unwrap();
        let back = decode(&encode(&m, true).unwrap()).unwrap();
        assert_eq!(back.channels(), 1);
        assert_eq!(back.to_vec::<u8>().unwrap(), data);
    }

    #[test]
    fn comments_and_extra_whitespace_are_skipped() {
        let src = b"P2\n# a comment\n  3 2  \n# another\n255\n0 1 2\n3 4 5\n";
        let m = decode(src).unwrap();
        assert_eq!(m.size(), Size::new(3, 2));
        assert_eq!(m.to_vec::<u8>().unwrap(), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn sixteen_bit_maxval_yields_u16() {
        // Binary 16-bit samples are big-endian.
        let mut src = b"P5\n2 1\n65535\n".to_vec();
        src.extend_from_slice(&[0x12, 0x34, 0xAB, 0xCD]);
        let m = decode(&src).unwrap();
        assert_eq!(m.depth(), Depth::U16);
        assert_eq!(m.to_vec::<u16>().unwrap(), vec![0x1234, 0xABCD]);
    }

    #[test]
    fn truncated_raster_is_rejected() {
        let src = b"P5\n4 4\n255\nonly-a-few";
        assert!(decode(src).is_err());
    }

    #[test]
    fn malformed_headers_are_rejected() {
        assert!(decode(b"P9\n1 1\n255\n\x00").is_err());
        assert!(decode(b"P5\n0 1\n255\n").is_err());
        assert!(decode(b"P5\n1 1\n0\n\x00").is_err());
        assert!(decode(b"P5\n1 1\n99999\n\x00").is_err());
        assert!(decode(b"P5\nabc 1\n255\n\x00").is_err());
        assert!(decode(b"P5").is_err());
    }

    #[test]
    fn one_bit_formats_are_named_as_unported() {
        let e = decode(b"P4\n8 1\n\xFF").unwrap_err();
        assert_eq!(e.code, Code::StsUnsupportedFormat);
        assert!(e.msg.contains("PBM"), "{}", e.msg);
    }

    #[test]
    fn four_channel_input_is_rejected() {
        let m = Mat::zeros(2, 2, cv_core::types::CV_8UC4).unwrap();
        assert_eq!(encode(&m, true).unwrap_err().code, Code::StsUnsupportedFormat);
    }
}
