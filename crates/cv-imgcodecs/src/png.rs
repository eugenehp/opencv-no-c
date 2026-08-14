//! PNG encoder and decoder (`modules/imgcodecs/src/grfmt_png.cpp`).
//!
//! Upstream delegates to libpng; this is a direct implementation over the
//! [`crate::deflate`] codec. Non-interlaced images of every colour type and bit
//! depth are supported; Adam7 interlacing is rejected with a clear error rather
//! than mis-decoded.
//!
//! Channel mapping follows OpenCV's loader, not PNG's own layout: RGB becomes
//! BGR, and any image with an alpha channel — including greyscale+alpha —
//! becomes 4-channel BGRA.

use cv_core::{cv_error, Code, Depth, Mat, MatLike, MatType, Result};

use crate::deflate::{crc32, zlib_compress, zlib_decompress};

/// The 8-byte PNG signature.
const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// Whether `data` starts with the PNG signature.
#[must_use]
pub fn is_png(data: &[u8]) -> bool {
    data.len() >= 8 && data[..8] == SIGNATURE
}

/// Header fields from the `IHDR` chunk.
#[derive(Debug, Clone, Copy)]
struct Header {
    width: u32,
    height: u32,
    bit_depth: u8,
    colour_type: u8,
    interlace: u8,
}

impl Header {
    /// Samples stored per pixel, before any BGR/alpha remapping.
    fn samples(&self) -> usize {
        match self.colour_type {
            0 | 3 => 1,
            2 => 3,
            4 => 2,
            _ => 4,
        }
    }

    /// Bits per pixel in the raw scanline.
    fn bits_per_pixel(&self) -> usize {
        self.samples() * self.bit_depth as usize
    }

    /// Filter stride: bytes per pixel, at least one.
    fn filter_stride(&self) -> usize {
        self.bits_per_pixel().div_ceil(8).max(1)
    }

    /// Bytes per unfiltered scanline, excluding the leading filter byte.
    fn row_bytes(&self) -> usize {
        (self.width as usize * self.bits_per_pixel()).div_ceil(8)
    }
}

/// Read a big-endian `u32`.
fn be32(d: &[u8], at: usize) -> Result<u32> {
    d.get(at..at + 4)
        .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| cv_core::Error::new(Code::StsError, "truncated PNG chunk".to_string()))
}

/// Decode a PNG image.
pub fn decode(data: &[u8]) -> Result<Mat> {
    if !is_png(data) {
        return cv_error!(Code::StsError, "not a PNG file");
    }

    let mut pos = 8usize;
    let mut header: Option<Header> = None;
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut idat: Vec<u8> = Vec::new();
    let mut saw_end = false;

    while pos + 8 <= data.len() {
        let len = be32(data, pos)? as usize;
        let ctype = data
            .get(pos + 4..pos + 8)
            .ok_or_else(|| cv_core::Error::new(Code::StsError, "truncated PNG".to_string()))?;
        let ctype = [ctype[0], ctype[1], ctype[2], ctype[3]];
        let body_start = pos + 8;
        let body = data.get(body_start..body_start + len).ok_or_else(|| {
            cv_core::Error::new(Code::StsError, "PNG chunk extends past end of file".to_string())
        })?;

        // The CRC covers the type and the body, not the length.
        let stored_crc = be32(data, body_start + len)?;
        let actual = crc32(&data[pos + 4..body_start + len]);
        if stored_crc != actual {
            return cv_error!(
                Code::StsError,
                "CRC mismatch in {} chunk",
                String::from_utf8_lossy(&ctype)
            );
        }

        match &ctype {
            b"IHDR" => {
                if len != 13 {
                    return cv_error!(Code::StsError, "IHDR must be 13 bytes, got {len}");
                }
                let h = Header {
                    width: be32(body, 0)?,
                    height: be32(body, 4)?,
                    bit_depth: body[8],
                    colour_type: body[9],
                    interlace: body[12],
                };
                validate(&h, body[10], body[11])?;
                header = Some(h);
            }
            b"PLTE" => {
                if len % 3 != 0 {
                    return cv_error!(Code::StsError, "PLTE length {len} is not a multiple of 3");
                }
                palette = body.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            }
            b"IDAT" => idat.extend_from_slice(body),
            b"IEND" => {
                saw_end = true;
                break;
            }
            // Ancillary chunks (tEXt, gAMA, pHYs, …) carry no pixels.
            _ => {}
        }
        pos = body_start + len + 4;
    }

    let h = header.ok_or_else(|| {
        cv_core::Error::new(Code::StsError, "PNG has no IHDR chunk".to_string())
    })?;
    if !saw_end {
        return cv_error!(Code::StsError, "PNG has no IEND chunk");
    }
    if idat.is_empty() {
        return cv_error!(Code::StsError, "PNG has no IDAT data");
    }
    if h.colour_type == 3 && palette.is_empty() {
        return cv_error!(Code::StsError, "indexed PNG has no PLTE chunk");
    }

    let raw = zlib_decompress(&idat)?;
    let scanlines = unfilter(&raw, &h)?;
    to_mat(&scanlines, &h, &palette)
}

/// Reject header combinations PNG does not define, or that we do not support.
fn validate(h: &Header, compression: u8, filter: u8) -> Result<()> {
    if h.width == 0 || h.height == 0 {
        return cv_error!(Code::StsBadSize, "PNG dimensions must be non-zero");
    }
    if compression != 0 {
        return cv_error!(Code::StsError, "unknown PNG compression method {compression}");
    }
    if filter != 0 {
        return cv_error!(Code::StsError, "unknown PNG filter method {filter}");
    }
    if h.interlace == 1 {
        return cv_error!(
            Code::StsUnsupportedFormat,
            "Adam7-interlaced PNG is not ported yet"
        );
    }
    if h.interlace != 0 {
        return cv_error!(Code::StsError, "unknown PNG interlace method {}", h.interlace);
    }
    // The legal depth set differs per colour type; see PNG spec table 11.1.
    let ok = match h.colour_type {
        0 => matches!(h.bit_depth, 1 | 2 | 4 | 8 | 16),
        3 => matches!(h.bit_depth, 1 | 2 | 4 | 8),
        2 | 4 | 6 => matches!(h.bit_depth, 8 | 16),
        _ => false,
    };
    if !ok {
        return cv_error!(
            Code::StsError,
            "colour type {} does not allow bit depth {}",
            h.colour_type,
            h.bit_depth
        );
    }
    Ok(())
}

/// `PaethPredictor` from the PNG specification.
#[inline]
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (ai, bi, ci) = (a as i32, b as i32, c as i32);
    let p = ai + bi - ci;
    let pa = (p - ai).abs();
    let pb = (p - bi).abs();
    let pc = (p - ci).abs();
    // Ties resolve toward `a`, then `b` — the order is normative.
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Reverse the per-scanline filters, returning packed rows without filter bytes.
fn unfilter(raw: &[u8], h: &Header) -> Result<Vec<u8>> {
    let row_bytes = h.row_bytes();
    let stride = h.filter_stride();
    let height = h.height as usize;
    let expected = (row_bytes + 1) * height;
    if raw.len() < expected {
        return cv_error!(
            Code::StsError,
            "PNG pixel data is {} bytes, expected {expected}",
            raw.len()
        );
    }

    let mut out = vec![0u8; row_bytes * height];
    for y in 0..height {
        let ftype = raw[y * (row_bytes + 1)];
        let src = &raw[y * (row_bytes + 1) + 1..y * (row_bytes + 1) + 1 + row_bytes];

        for x in 0..row_bytes {
            let a = if x >= stride { out[y * row_bytes + x - stride] } else { 0 };
            let b = if y > 0 { out[(y - 1) * row_bytes + x] } else { 0 };
            let c = if y > 0 && x >= stride { out[(y - 1) * row_bytes + x - stride] } else { 0 };

            let v = match ftype {
                0 => src[x],
                1 => src[x].wrapping_add(a),
                2 => src[x].wrapping_add(b),
                // Average uses the *floor* of the mean of the two neighbours,
                // computed in a wider type so it cannot overflow.
                3 => src[x].wrapping_add(((a as u16 + b as u16) / 2) as u8),
                4 => src[x].wrapping_add(paeth(a, b, c)),
                n => return cv_error!(Code::StsError, "unknown PNG filter type {n} on row {y}"),
            };
            out[y * row_bytes + x] = v;
        }
    }
    Ok(out)
}

/// Extract the `i`-th sample of a packed scanline for sub-byte depths.
#[inline]
fn packed_sample(row: &[u8], index: usize, bit_depth: u8) -> u8 {
    let per_byte = 8 / bit_depth as usize;
    let byte = row[index / per_byte];
    let shift = 8 - bit_depth as usize * (index % per_byte + 1);
    let mask = (1u16 << bit_depth) - 1;
    ((byte as u16 >> shift) & mask) as u8
}

/// Convert unfiltered scanlines into a `Mat` in OpenCV's channel order.
fn to_mat(rows: &[u8], h: &Header, palette: &[[u8; 3]]) -> Result<Mat> {
    let (w, ht) = (h.width as usize, h.height as usize);
    let row_bytes = h.row_bytes();
    let wide = h.bit_depth == 16;
    let depth = if wide { Depth::U16 } else { Depth::U8 };

    // Alpha in any form becomes 4-channel BGRA, matching OpenCV's loader.
    let out_cn: i32 = match h.colour_type {
        0 => 1,
        2 | 3 => 3,
        _ => 4,
    };
    let mut img = Mat::zeros(ht as i32, w as i32, MatType::new(depth, out_cn))?;
    // Scale factor lifting a sub-byte greyscale sample to the full 0..255 range.
    let sub_byte_scale = if h.bit_depth < 8 {
        255.0 / ((1u16 << h.bit_depth) - 1) as f64
    } else {
        1.0
    };

    for y in 0..ht {
        let row = &rows[y * row_bytes..(y + 1) * row_bytes];
        for x in 0..w {
            // Pull the raw samples for this pixel.
            let mut s = [0f64; 4];
            let n = h.samples();
            for (k, slot) in s.iter_mut().enumerate().take(n) {
                *slot = if wide {
                    let o = (x * n + k) * 2;
                    u16::from_be_bytes([row[o], row[o + 1]]) as f64
                } else if h.bit_depth == 8 {
                    row[x * n + k] as f64
                } else {
                    packed_sample(row, x * n + k, h.bit_depth) as f64
                };
            }

            match h.colour_type {
                0 => {
                    let v = if h.bit_depth < 8 { s[0] * sub_byte_scale } else { s[0] };
                    cv_core::arithm::set_channel(&mut img, y as i32, x as i32, 0, v)?;
                }
                2 => {
                    // PNG stores RGB; Mat wants BGR.
                    for c in 0..3 {
                        cv_core::arithm::set_channel(
                            &mut img,
                            y as i32,
                            x as i32,
                            c,
                            s[2 - c as usize],
                        )?;
                    }
                }
                3 => {
                    let idx = s[0] as usize;
                    let e = palette.get(idx).copied().unwrap_or([0, 0, 0]);
                    // Palette entries are RGB.
                    for c in 0..3 {
                        cv_core::arithm::set_channel(
                            &mut img,
                            y as i32,
                            x as i32,
                            c,
                            e[2 - c as usize] as f64,
                        )?;
                    }
                }
                4 => {
                    // Greyscale + alpha: replicate luma into BGR, keep alpha.
                    for c in 0..3 {
                        cv_core::arithm::set_channel(&mut img, y as i32, x as i32, c, s[0])?;
                    }
                    cv_core::arithm::set_channel(&mut img, y as i32, x as i32, 3, s[1])?;
                }
                _ => {
                    for c in 0..3 {
                        cv_core::arithm::set_channel(
                            &mut img,
                            y as i32,
                            x as i32,
                            c,
                            s[2 - c as usize],
                        )?;
                    }
                    cv_core::arithm::set_channel(&mut img, y as i32, x as i32, 3, s[3])?;
                }
            }
        }
    }
    Ok(img)
}

/// Append a length-type-data-CRC chunk.
fn push_chunk(out: &mut Vec<u8>, ctype: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(ctype);
    out.extend_from_slice(body);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// Encode a `CV_8U` or `CV_16U` matrix as a non-interlaced PNG.
///
/// Scanlines are written with filter type 0 (None). Filtering exists to make
/// the subsequent DEFLATE pass more effective, and this crate's encoder emits
/// stored blocks anyway, so a smarter filter would not shrink the output.
pub fn encode<M: MatLike + ?Sized>(img: &M) -> Result<Vec<u8>> {
    let cn = img.channels();
    let colour_type: u8 = match cn {
        1 => 0,
        3 => 2,
        4 => 6,
        n => return cv_error!(Code::StsUnsupportedFormat, "PNG needs 1, 3 or 4 channels, got {n}"),
    };
    let bit_depth: u8 = match img.depth() {
        Depth::U8 => 8,
        Depth::U16 => 16,
        d => {
            return cv_error!(
                Code::StsUnsupportedFormat,
                "PNG needs CV_8U or CV_16U, got {}",
                d.name()
            )
        }
    };
    let (w, h) = (img.cols(), img.rows());

    let mut out = Vec::new();
    out.extend_from_slice(&SIGNATURE);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.push(bit_depth);
    ihdr.push(colour_type);
    ihdr.push(0); // compression: deflate
    ihdr.push(0); // filter method: adaptive
    ihdr.push(0); // interlace: none
    push_chunk(&mut out, b"IHDR", &ihdr);

    // Build the filtered raster: one filter byte then the row's samples.
    let sample_bytes = if bit_depth == 16 { 2 } else { 1 };
    let mut raster = Vec::with_capacity(h as usize * (1 + w as usize * cn as usize * sample_bytes));
    for y in 0..h {
        raster.push(0); // filter: None
        for x in 0..w {
            for c in 0..cn {
                // Mat is BGR(A); PNG wants RGB(A). Alpha stays in place.
                let src_c = if cn >= 3 && c < 3 { 2 - c } else { c };
                let v = img.at_f64(y, x, src_c)?;
                if bit_depth == 16 {
                    raster.extend_from_slice(&(v as u16).to_be_bytes());
                } else {
                    raster.push(v as u8);
                }
            }
        }
    }
    push_chunk(&mut out, b"IDAT", &zlib_compress(&raster));
    push_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{types::CV_8UC1, types::CV_8UC3, types::CV_8UC4, MatLikeMut, Size};

    fn colour_image() -> Mat {
        let mut m = Mat::zeros(6, 8, CV_8UC3).unwrap();
        for y in 0..6 {
            for x in 0..8 {
                let p = m.pixel_mut::<u8>(y, x).unwrap();
                p[0] = (x * 30) as u8;
                p[1] = (y * 40) as u8;
                p[2] = ((x + y) * 15) as u8;
            }
        }
        m
    }

    #[test]
    fn signature_is_recognised() {
        assert!(is_png(&SIGNATURE));
        assert!(!is_png(b"BM123456"));
        assert!(!is_png(b"\x89PNG"));
    }

    #[test]
    fn rgb_roundtrip_preserves_pixels() {
        let m = colour_image();
        let png = encode(&m).unwrap();
        assert!(is_png(&png));
        let back = decode(&png).unwrap();
        assert_eq!(back.size(), m.size());
        assert_eq!(back.channels(), 3);
        assert_eq!(back.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn grayscale_roundtrip() {
        let data: Vec<u8> = (0..48).map(|i| (i * 5) as u8).collect();
        let m = Mat::from_vec(6, 8, 1, &data).unwrap();
        let back = decode(&encode(&m).unwrap()).unwrap();
        assert_eq!(back.channels(), 1);
        assert_eq!(back.to_vec::<u8>().unwrap(), data);
    }

    #[test]
    fn rgba_roundtrip_preserves_alpha() {
        let mut m = Mat::zeros(3, 3, CV_8UC4).unwrap();
        for y in 0..3 {
            for x in 0..3 {
                let p = m.pixel_mut::<u8>(y, x).unwrap();
                p[0] = 10;
                p[1] = 20;
                p[2] = 30;
                p[3] = (x * 100 + y * 10) as u8;
            }
        }
        let back = decode(&encode(&m).unwrap()).unwrap();
        assert_eq!(back.channels(), 4);
        assert_eq!(back.to_vec::<u8>().unwrap(), m.to_vec::<u8>().unwrap());
    }

    #[test]
    fn sixteen_bit_roundtrip() {
        let data: Vec<u16> = vec![0, 1000, 40000, 65535, 12345, 500];
        let m = Mat::from_vec(2, 3, 1, &data).unwrap();
        let png = encode(&m).unwrap();
        let back = decode(&png).unwrap();
        assert_eq!(back.depth(), Depth::U16);
        assert_eq!(back.to_vec::<u16>().unwrap(), data);
    }

    #[test]
    fn chunk_layout_is_well_formed() {
        let png = encode(&colour_image()).unwrap();
        assert_eq!(&png[..8], &SIGNATURE);
        // First chunk must be IHDR of length 13.
        assert_eq!(be32(&png, 8).unwrap(), 13);
        assert_eq!(&png[12..16], b"IHDR");
        // Last chunk must be a zero-length IEND with its known CRC.
        let n = png.len();
        assert_eq!(&png[n - 8..n - 4], b"IEND");
        assert_eq!(be32(&png, n - 4).unwrap(), 0xAE42_6082);
    }

    #[test]
    fn corrupt_crc_is_detected() {
        let mut png = encode(&colour_image()).unwrap();
        // Flip a byte inside the IHDR body; the CRC check must catch it.
        png[20] ^= 0xFF;
        let e = decode(&png).unwrap_err();
        assert!(e.msg.contains("CRC"), "{}", e.msg);
    }

    #[test]
    fn interlaced_images_are_rejected_by_name() {
        let mut png = encode(&colour_image()).unwrap();
        // IHDR body starts at 16; interlace is its 13th byte.
        png[16 + 12] = 1;
        // Repair the CRC so the interlace check is what fires.
        let crc = crc32(&png[12..16 + 13]);
        png[16 + 13..16 + 17].copy_from_slice(&crc.to_be_bytes());
        let e = decode(&png).unwrap_err();
        assert_eq!(e.code, Code::StsUnsupportedFormat);
        assert!(e.msg.contains("nterlac"), "{}", e.msg);
    }

    #[test]
    fn paeth_predictor_matches_the_specification() {
        // Worked examples from the PNG spec's filter section.
        assert_eq!(paeth(0, 0, 0), 0);
        assert_eq!(paeth(10, 20, 15), 15); // p=15, ties resolve to c here
        assert_eq!(paeth(200, 100, 150), 150);
        // p = 1+2-3 = 0, so pa=1 is the smallest and `a` wins.
        assert_eq!(paeth(1, 2, 3), 1);
        // When a is the closest prediction it wins outright.
        assert_eq!(paeth(50, 0, 0), 50);
    }

    #[test]
    fn all_five_filters_reconstruct_a_known_image() {
        // Build an image, then re-encode its raster under each filter type and
        // confirm the decoder recovers the original bytes.
        let m = colour_image();
        let base = encode(&m).unwrap();
        let expected = decode(&base).unwrap().to_vec::<u8>().unwrap();

        let h = Header { width: 8, height: 6, bit_depth: 8, colour_type: 2, interlace: 0 };
        let row_bytes = h.row_bytes();
        let stride = h.filter_stride();

        // Unfiltered raster in PNG (RGB) order.
        let mut plain = vec![0u8; row_bytes * 6];
        for y in 0..6usize {
            for x in 0..8usize {
                for c in 0..3usize {
                    plain[y * row_bytes + x * 3 + c] =
                        m.at_f64(y as i32, x as i32, 2 - c as i32).unwrap() as u8;
                }
            }
        }

        for ftype in 0u8..=4 {
            let mut raster = Vec::new();
            for y in 0..6usize {
                raster.push(ftype);
                for x in 0..row_bytes {
                    let cur = plain[y * row_bytes + x];
                    let a = if x >= stride { plain[y * row_bytes + x - stride] } else { 0 };
                    let b = if y > 0 { plain[(y - 1) * row_bytes + x] } else { 0 };
                    let c =
                        if y > 0 && x >= stride { plain[(y - 1) * row_bytes + x - stride] } else { 0 };
                    let pred = match ftype {
                        0 => 0,
                        1 => a,
                        2 => b,
                        3 => ((a as u16 + b as u16) / 2) as u8,
                        _ => paeth(a, b, c),
                    };
                    raster.push(cur.wrapping_sub(pred));
                }
            }

            let mut png = Vec::new();
            png.extend_from_slice(&SIGNATURE);
            let mut ihdr = Vec::new();
            ihdr.extend_from_slice(&8u32.to_be_bytes());
            ihdr.extend_from_slice(&6u32.to_be_bytes());
            ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
            push_chunk(&mut png, b"IHDR", &ihdr);
            push_chunk(&mut png, b"IDAT", &zlib_compress(&raster));
            push_chunk(&mut png, b"IEND", &[]);

            let back = decode(&png).unwrap();
            assert_eq!(back.to_vec::<u8>().unwrap(), expected, "filter type {ftype}");
        }
    }

    #[test]
    fn missing_chunks_are_reported() {
        let png = encode(&colour_image()).unwrap();
        // Truncate before IEND.
        let n = png.len();
        assert!(decode(&png[..n - 12]).is_err());
        // Signature only.
        assert!(decode(&SIGNATURE).is_err());
    }

    #[test]
    fn illegal_depth_for_colour_type_is_rejected() {
        let h = Header { width: 1, height: 1, bit_depth: 1, colour_type: 2, interlace: 0 };
        assert!(validate(&h, 0, 0).is_err());
        let h = Header { width: 1, height: 1, bit_depth: 16, colour_type: 3, interlace: 0 };
        assert!(validate(&h, 0, 0).is_err());
        let h = Header { width: 1, height: 1, bit_depth: 4, colour_type: 0, interlace: 0 };
        assert!(validate(&h, 0, 0).is_ok());
    }

    #[test]
    fn packed_samples_unpack_correctly() {
        // One byte holding eight 1-bit samples, MSB first.
        let row = [0b1010_0110u8];
        let bits: Vec<u8> = (0..8).map(|i| packed_sample(&row, i, 1)).collect();
        assert_eq!(bits, vec![1, 0, 1, 0, 0, 1, 1, 0]);

        // Two 4-bit samples per byte.
        let row = [0xAB];
        assert_eq!(packed_sample(&row, 0, 4), 0xA);
        assert_eq!(packed_sample(&row, 1, 4), 0xB);

        // Four 2-bit samples per byte.
        let row = [0b11_01_00_10];
        let vals: Vec<u8> = (0..4).map(|i| packed_sample(&row, i, 2)).collect();
        assert_eq!(vals, vec![3, 1, 0, 2]);
    }

    #[test]
    fn one_bit_grayscale_scales_to_full_range() {
        // A 1-bit greyscale PNG must decode to 0 and 255, not 0 and 1.
        let mut png = Vec::new();
        png.extend_from_slice(&SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&8u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[1, 0, 0, 0, 0]);
        push_chunk(&mut png, b"IHDR", &ihdr);
        // One scanline: filter 0, then the bit pattern 1010 0110.
        push_chunk(&mut png, b"IDAT", &zlib_compress(&[0u8, 0b1010_0110]));
        push_chunk(&mut png, b"IEND", &[]);

        let m = decode(&png).unwrap();
        assert_eq!(m.size(), Size::new(8, 1));
        assert_eq!(m.to_vec::<u8>().unwrap(), vec![255, 0, 255, 0, 0, 255, 255, 0]);
    }

    #[test]
    fn indexed_palette_decodes_to_bgr() {
        let mut png = Vec::new();
        png.extend_from_slice(&SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&2u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 3, 0, 0, 0]);
        push_chunk(&mut png, b"IHDR", &ihdr);
        // Palette holds RGB; the decoder must emit BGR.
        push_chunk(&mut png, b"PLTE", &[255, 0, 0, 0, 0, 255]);
        push_chunk(&mut png, b"IDAT", &zlib_compress(&[0u8, 0, 1]));
        push_chunk(&mut png, b"IEND", &[]);

        let m = decode(&png).unwrap();
        assert_eq!(m.channels(), 3);
        assert_eq!(m.pixel::<u8>(0, 0).unwrap(), &[0, 0, 255]);
        assert_eq!(m.pixel::<u8>(0, 1).unwrap(), &[255, 0, 0]);
    }

    #[test]
    fn grayscale_alpha_becomes_bgra() {
        let mut png = Vec::new();
        png.extend_from_slice(&SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&2u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 4, 0, 0, 0]);
        push_chunk(&mut png, b"IHDR", &ihdr);
        push_chunk(&mut png, b"IDAT", &zlib_compress(&[0u8, 100, 50, 200, 255]));
        push_chunk(&mut png, b"IEND", &[]);

        let m = decode(&png).unwrap();
        assert_eq!(m.channels(), 4);
        assert_eq!(m.pixel::<u8>(0, 0).unwrap(), &[100, 100, 100, 50]);
        assert_eq!(m.pixel::<u8>(0, 1).unwrap(), &[200, 200, 200, 255]);
    }

    #[test]
    fn arbitrary_bytes_never_panic() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for len in [8usize, 16, 40, 100, 300] {
            for _ in 0..100 {
                let mut buf: Vec<u8> = SIGNATURE.to_vec();
                buf.extend((0..len).map(|_| (next() & 0xFF) as u8));
                let _ = decode(&buf);
            }
        }
    }

    #[test]
    fn unsupported_inputs_are_rejected() {
        let f = Mat::zeros(2, 2, cv_core::types::CV_32FC1).unwrap();
        assert_eq!(encode(&f).unwrap_err().code, Code::StsUnsupportedFormat);
        let c2 = Mat::zeros(2, 2, MatType::new(Depth::U8, 2)).unwrap();
        assert_eq!(encode(&c2).unwrap_err().code, Code::StsUnsupportedFormat);
        let _ = CV_8UC1;
    }
}
