//! Windows BMP encoder and decoder (`modules/imgcodecs/src/grfmt_bmp.cpp`).
//!
//! Uncompressed `BI_RGB` only: 8-bit paletted, 24-bit BGR and 32-bit BGRA.
//! RLE4/RLE8 and the `BI_BITFIELDS` masks are not ported.
//!
//! Two details bite anyone writing a BMP by hand: rows are padded to a 4-byte
//! boundary, and a positive header height means the rows are stored
//! **bottom-up**. Both are handled here.

use cv_core::{cv_error, Code, Mat, MatLike, MatLikeMut, Result};

/// Bytes in the `BITMAPFILEHEADER`.
const FILE_HEADER: usize = 14;
/// Bytes in the `BITMAPINFOHEADER`.
const INFO_HEADER: usize = 40;

/// Read a little-endian `u32`.
fn u32le(d: &[u8], at: usize) -> Result<u32> {
    d.get(at..at + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| cv_core::Error::new(Code::StsError, "truncated BMP header".to_string()))
}

/// Read a little-endian `i32`.
fn i32le(d: &[u8], at: usize) -> Result<i32> {
    u32le(d, at).map(|v| v as i32)
}

/// Read a little-endian `u16`.
fn u16le(d: &[u8], at: usize) -> Result<u16> {
    d.get(at..at + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .ok_or_else(|| cv_core::Error::new(Code::StsError, "truncated BMP header".to_string()))
}

/// Bytes per padded row for `width` pixels at `bpp` bits each.
#[inline]
fn stride(width: i32, bpp: u16) -> usize {
    // Each row is padded up to a multiple of four bytes.
    (width as usize * bpp as usize).div_ceil(32) * 4
}

/// Decode an uncompressed BMP into a `CV_8UC1` or `CV_8UC3` matrix.
pub fn decode(data: &[u8]) -> Result<Mat> {
    if data.len() < FILE_HEADER + INFO_HEADER || &data[..2] != b"BM" {
        return cv_error!(Code::StsError, "not a BMP file");
    }
    let data_offset = u32le(data, 10)? as usize;
    let header_size = u32le(data, 14)? as usize;
    if header_size < INFO_HEADER {
        return cv_error!(Code::StsUnsupportedFormat, "BITMAPCOREHEADER is not ported");
    }

    let width = i32le(data, 18)?;
    let raw_height = i32le(data, 22)?;
    let bpp = u16le(data, 28)?;
    let compression = u32le(data, 30)?;

    if compression != 0 {
        return cv_error!(
            Code::StsUnsupportedFormat,
            "only uncompressed BI_RGB BMP is ported, got compression {compression}"
        );
    }
    if width <= 0 || raw_height == 0 {
        return cv_error!(Code::StsBadSize, "invalid BMP dimensions {width}x{raw_height}");
    }
    // A negative height means the rows are stored top-down.
    let bottom_up = raw_height > 0;
    let height = raw_height.abs();

    let (channels, use_palette) = match bpp {
        8 => (1, true),
        24 => (3, false),
        32 => (4, false),
        n => return cv_error!(Code::StsUnsupportedFormat, "unsupported BMP depth {n} bpp"),
    };

    // The palette sits between the info header and the pixel data.
    let palette: Vec<[u8; 3]> = if use_palette {
        let colors_used = u32le(data, 46)? as usize;
        let n = if colors_used == 0 { 256 } else { colors_used };
        let start = FILE_HEADER + header_size;
        let end = start + n * 4;
        let bytes = data
            .get(start..end)
            .ok_or_else(|| cv_core::Error::new(Code::StsError, "truncated BMP palette".to_string()))?;
        bytes.chunks_exact(4).map(|c| [c[0], c[1], c[2]]).collect()
    } else {
        Vec::new()
    };

    let row_bytes = stride(width, bpp);
    let needed = data_offset + row_bytes * height as usize;
    if data.len() < needed {
        return cv_error!(
            Code::StsError,
            "truncated BMP pixel data: need {needed} bytes, have {}",
            data.len()
        );
    }

    // An 8-bit BMP whose palette is not pure grey still carries colour, so it
    // must decode to three channels.
    let grey_palette = use_palette
        && palette.iter().enumerate().all(|(i, c)| {
            let v = i as u8;
            c[0] == v && c[1] == v && c[2] == v
        });
    let out_cn = if use_palette && !grey_palette { 3 } else { channels.min(3) };
    let mut img = Mat::zeros(height, width, cv_core::MatType::new(cv_core::Depth::U8, out_cn))?;

    for y in 0..height {
        let src_row = if bottom_up { height - 1 - y } else { y };
        let base = data_offset + src_row as usize * row_bytes;
        let row = &data[base..base + row_bytes];
        let dst = img.row_as_mut::<u8>(y)?;

        for x in 0..width as usize {
            let o = x * out_cn as usize;
            match bpp {
                8 => {
                    let idx = row[x] as usize;
                    if grey_palette {
                        dst[o] = palette.get(idx).map_or(row[x], |c| c[0]);
                    } else {
                        // Palette entries are stored BGRA, which is already the
                        // Mat channel order.
                        let c = palette.get(idx).copied().unwrap_or([0, 0, 0]);
                        dst[o] = c[0];
                        dst[o + 1] = c[1];
                        dst[o + 2] = c[2];
                    }
                }
                24 => {
                    dst[o] = row[x * 3];
                    dst[o + 1] = row[x * 3 + 1];
                    dst[o + 2] = row[x * 3 + 2];
                }
                _ => {
                    // 32-bit: drop the alpha byte, keeping BGR.
                    dst[o] = row[x * 4];
                    dst[o + 1] = row[x * 4 + 1];
                    dst[o + 2] = row[x * 4 + 2];
                }
            }
        }
    }
    Ok(img)
}

/// Encode a `CV_8UC1` or `CV_8UC3` matrix as an uncompressed BMP.
///
/// Single-channel images are written as 8-bit with a 256-entry greyscale
/// palette; three- and four-channel images are written as 24-bit BGR.
pub fn encode<M: MatLike + ?Sized>(img: &M) -> Result<Vec<u8>> {
    let cn = img.channels();
    if !(cn == 1 || cn == 3 || cn == 4) {
        return cv_error!(Code::StsUnsupportedFormat, "BMP needs 1, 3 or 4 channels, got {cn}");
    }
    let (width, height) = (img.cols(), img.rows());
    let paletted = cn == 1;
    let bpp: u16 = if paletted { 8 } else { 24 };

    let palette_bytes = if paletted { 256 * 4 } else { 0 };
    let row_bytes = stride(width, bpp);
    let pixel_bytes = row_bytes * height as usize;
    let data_offset = FILE_HEADER + INFO_HEADER + palette_bytes;
    let file_size = data_offset + pixel_bytes;

    let mut out = Vec::with_capacity(file_size);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved
    out.extend_from_slice(&(data_offset as u32).to_le_bytes());

    out.extend_from_slice(&(INFO_HEADER as u32).to_le_bytes());
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes()); // positive: bottom-up
    out.extend_from_slice(&1u16.to_le_bytes()); // planes
    out.extend_from_slice(&bpp.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
    out.extend_from_slice(&(pixel_bytes as u32).to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes()); // x pixels/metre
    out.extend_from_slice(&0i32.to_le_bytes()); // y pixels/metre
    out.extend_from_slice(&(if paletted { 256u32 } else { 0 }).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // important colours

    if paletted {
        for i in 0..256u32 {
            let v = i as u8;
            out.extend_from_slice(&[v, v, v, 0]);
        }
    }

    // Rows are written bottom-up to match the positive height above.
    for y in (0..height).rev() {
        let row = img.row_as::<u8>(y)?;
        let before = out.len();
        for x in 0..width as usize {
            let o = x * cn as usize;
            if paletted {
                out.push(row[o]);
            } else {
                out.extend_from_slice(&[row[o], row[o + 1], row[o + 2]]);
            }
        }
        // Pad to the 4-byte row boundary.
        out.resize(before + row_bytes, 0);
    }
    debug_assert_eq!(out.len(), file_size);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{types::CV_8UC3, Scalar};

    #[test]
    fn stride_pads_to_four_bytes() {
        assert_eq!(stride(1, 24), 4);
        assert_eq!(stride(2, 24), 8);
        assert_eq!(stride(4, 24), 12);
        assert_eq!(stride(1, 8), 4);
        assert_eq!(stride(5, 8), 8);
        assert_eq!(stride(4, 32), 16);
    }

    #[test]
    fn header_fields_are_written_correctly() {
        let m = Mat::new_with(3, 5, CV_8UC3, Scalar::new(1.0, 2.0, 3.0, 0.0)).unwrap();
        let b = encode(&m).unwrap();
        assert_eq!(&b[..2], b"BM");
        assert_eq!(u32le(&b, 2).unwrap() as usize, b.len());
        assert_eq!(u32le(&b, 14).unwrap(), 40);
        assert_eq!(i32le(&b, 18).unwrap(), 5);
        assert_eq!(i32le(&b, 22).unwrap(), 3);
        assert_eq!(u16le(&b, 28).unwrap(), 24);
        assert_eq!(u32le(&b, 30).unwrap(), 0);
    }

    #[test]
    fn rows_are_stored_bottom_up() {
        // Top row is 10, bottom row is 20; the file must hold 20 first.
        let m = Mat::from_vec(2, 1, 3, &[10u8, 10, 10, 20, 20, 20]).unwrap();
        let b = encode(&m).unwrap();
        let off = u32le(&b, 10).unwrap() as usize;
        assert_eq!(b[off], 20, "first stored row should be the image's last");
        let back = decode(&b).unwrap();
        assert_eq!(back.pixel::<u8>(0, 0).unwrap(), &[10, 10, 10]);
        assert_eq!(back.pixel::<u8>(1, 0).unwrap(), &[20, 20, 20]);
    }

    #[test]
    fn top_down_bitmaps_decode_correctly() {
        // Flip the sign of the height and reverse the stored rows: the decoded
        // image must be identical.
        let m = Mat::from_vec(2, 1, 3, &[10u8, 10, 10, 20, 20, 20]).unwrap();
        let mut b = encode(&m).unwrap();
        let off = u32le(&b, 10).unwrap() as usize;
        let row_bytes = stride(1, 24);
        b[22..26].copy_from_slice(&(-2i32).to_le_bytes());
        let (a, c) = (off, off + row_bytes);
        for i in 0..row_bytes {
            b.swap(a + i, c + i);
        }
        let back = decode(&b).unwrap();
        assert_eq!(back.pixel::<u8>(0, 0).unwrap(), &[10, 10, 10]);
        assert_eq!(back.pixel::<u8>(1, 0).unwrap(), &[20, 20, 20]);
    }

    #[test]
    fn padding_bytes_are_zero() {
        // 1 pixel of 24bpp is 3 bytes, padded to 4.
        let m = Mat::new_with(1, 1, CV_8UC3, Scalar::new(1.0, 2.0, 3.0, 0.0)).unwrap();
        let b = encode(&m).unwrap();
        let off = u32le(&b, 10).unwrap() as usize;
        assert_eq!(&b[off..off + 3], &[1, 2, 3]);
        assert_eq!(b[off + 3], 0);
    }

    #[test]
    fn colour_palette_decodes_to_three_channels() {
        // Build an 8-bit BMP whose palette is not greyscale.
        let m = Mat::from_vec(1, 2, 1, &[0u8, 1]).unwrap();
        let mut b = encode(&m).unwrap();
        // Rewrite palette entries 0 and 1 as distinct colours (BGRA order).
        let pal = FILE_HEADER + INFO_HEADER;
        b[pal..pal + 8].copy_from_slice(&[255, 0, 0, 0, 0, 255, 0, 0]);
        let back = decode(&b).unwrap();
        assert_eq!(back.channels(), 3);
        assert_eq!(back.pixel::<u8>(0, 0).unwrap(), &[255, 0, 0]);
        assert_eq!(back.pixel::<u8>(0, 1).unwrap(), &[0, 255, 0]);
    }

    #[test]
    fn compressed_bitmaps_are_rejected() {
        let m = Mat::new_with(2, 2, CV_8UC3, Scalar::all(1.0)).unwrap();
        let mut b = encode(&m).unwrap();
        b[30..34].copy_from_slice(&1u32.to_le_bytes()); // BI_RLE8
        let e = decode(&b).unwrap_err();
        assert_eq!(e.code, Code::StsUnsupportedFormat);
    }

    #[test]
    fn unsupported_bit_depths_are_rejected() {
        let m = Mat::new_with(2, 2, CV_8UC3, Scalar::all(1.0)).unwrap();
        let mut b = encode(&m).unwrap();
        b[28..30].copy_from_slice(&4u16.to_le_bytes());
        assert_eq!(decode(&b).unwrap_err().code, Code::StsUnsupportedFormat);
    }

    #[test]
    fn zero_dimensions_are_rejected() {
        let m = Mat::new_with(2, 2, CV_8UC3, Scalar::all(1.0)).unwrap();
        let mut b = encode(&m).unwrap();
        b[18..22].copy_from_slice(&0i32.to_le_bytes());
        assert_eq!(decode(&b).unwrap_err().code, Code::StsBadSize);
    }
}
