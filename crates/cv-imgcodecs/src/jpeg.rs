//! Baseline JPEG decoding (`modules/imgcodecs/src/grfmt_jpeg.cpp`).
//!
//! # Scope
//!
//! Baseline sequential DCT, Huffman-coded, 8-bit — which is what essentially
//! every camera and every `imwrite` produces. Progressive JPEG, arithmetic
//! coding and 12-bit samples are rejected with a clear error rather than
//! decoded wrongly.
//!
//! # Why write one at all
//!
//! JPEG is the format the rest of the workspace most obviously lacked, and it
//! is entirely self-contained: entropy decoding, dequantisation, an inverse
//! DCT and a colour transform. No part of it needs anything outside this file.

use cv_core::types::{Depth, MatType};
use cv_core::{Code, Error, Mat, MatLikeMut, Result};

/// Zig-zag order: the sequence the 64 DCT coefficients are stored in.
#[rustfmt::skip]
const ZIGZAG: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// A decoded Huffman table, as a canonical code list.
#[derive(Default, Clone)]
struct HuffTable {
    /// `mincode[l]`, `maxcode[l]`, `valptr[l]` for code length `l+1`.
    mincode: [i32; 17],
    maxcode: [i32; 17],
    valptr: [i32; 17],
    values: Vec<u8>,
}

impl HuffTable {
    /// Build from the JPEG `DHT` form: counts per length, then the values.
    fn build(counts: &[u8; 16], values: Vec<u8>) -> Self {
        let mut t = HuffTable { values, ..Default::default() };
        let mut code = 0i32;
        let mut k = 0i32;
        for l in 1..=16usize {
            t.valptr[l] = k;
            t.mincode[l] = code;
            code += i32::from(counts[l - 1]);
            k += i32::from(counts[l - 1]);
            t.maxcode[l] = code - 1;
            if counts[l - 1] == 0 {
                // No codes of this length; mark it unmatchable.
                t.maxcode[l] = -1;
            }
            code <<= 1;
        }
        t
    }
}

/// Bit reader over the entropy-coded segment, handling byte stuffing.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bits: u32,
    count: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0, bits: 0, count: 0 }
    }

    /// Next bit, MSB first.
    ///
    /// A `0xFF` byte inside entropy-coded data is followed by a stuffed `0x00`,
    /// because `0xFF` otherwise starts a marker. Missing that turns a marker
    /// into image data and the picture falls apart halfway down.
    fn bit(&mut self) -> Result<u32> {
        if self.count == 0 {
            let Some(&b) = self.data.get(self.pos) else {
                return Err(Error::new(Code::StsParseError, "JPEG data ended mid-scan".to_string()));
            };
            self.pos += 1;
            if b == 0xFF {
                match self.data.get(self.pos) {
                    Some(0x00) => self.pos += 1,
                    // A restart marker: skip it and carry on.
                    Some(m) if (0xD0..=0xD7).contains(m) => {
                        self.pos += 1;
                        self.bits = 0;
                        self.count = 0;
                        return self.bit();
                    }
                    _ => {
                        return Err(Error::new(
                            Code::StsParseError,
                            "unexpected JPEG marker inside the scan".to_string(),
                        ))
                    }
                }
            }
            self.bits = u32::from(b);
            self.count = 8;
        }
        self.count -= 1;
        Ok((self.bits >> self.count) & 1)
    }

    /// Decode one Huffman symbol.
    fn huff(&mut self, t: &HuffTable) -> Result<u8> {
        let mut code = self.bit()? as i32;
        for l in 1..=16usize {
            if t.maxcode[l] >= 0 && code <= t.maxcode[l] {
                let idx = (t.valptr[l] + code - t.mincode[l]) as usize;
                return t.values.get(idx).copied().ok_or_else(|| {
                    Error::new(Code::StsParseError, "JPEG Huffman index out of range".to_string())
                });
            }
            code = (code << 1) | self.bit()? as i32;
        }
        Err(Error::new(Code::StsParseError, "invalid JPEG Huffman code".to_string()))
    }

    /// Read `n` bits as a signed JPEG coefficient.
    fn receive_extend(&mut self, n: u8) -> Result<i32> {
        if n == 0 {
            return Ok(0);
        }
        let mut v = 0i32;
        for _ in 0..n {
            v = (v << 1) | self.bit()? as i32;
        }
        // Values in the lower half of the range are negative; this is JPEG's
        // sign convention and it is not two's complement.
        Ok(if v < (1 << (n - 1)) { v - (1 << n) + 1 } else { v })
    }

    /// Discard any partial byte, at a restart boundary.
    fn align(&mut self) {
        self.count = 0;
    }
}

/// Separable inverse DCT over an 8x8 block.
///
/// The straightforward O(n^3) form. AAN and Loeffler are faster; this is
/// clearer and the cost is not what dominates a decode.
fn idct8x8(inp: &[f64; 64], out: &mut [f64; 64]) {
    let mut tmp = [0.0f64; 64];
    let c = |u: usize| if u == 0 { std::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
    // Rows.
    for y in 0..8 {
        for x in 0..8 {
            let mut s = 0.0;
            for u in 0..8 {
                s += c(u)
                    * inp[y * 8 + u]
                    * ((2.0 * x as f64 + 1.0) * u as f64 * std::f64::consts::PI / 16.0).cos();
            }
            tmp[y * 8 + x] = s * 0.5;
        }
    }
    // Columns.
    for x in 0..8 {
        for y in 0..8 {
            let mut s = 0.0;
            for v in 0..8 {
                s += c(v)
                    * tmp[v * 8 + x]
                    * ((2.0 * y as f64 + 1.0) * v as f64 * std::f64::consts::PI / 16.0).cos();
            }
            out[y * 8 + x] = s * 0.5;
        }
    }
}

/// One component of the frame.
#[derive(Clone, Copy, Default)]
struct Component {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    td: usize,
    ta: usize,
}

/// `cv::imread` for a baseline JPEG.
///
/// Returns `CV_8UC3` in **BGR** order for a colour image and `CV_8UC1` for
/// greyscale, matching the rest of the workspace and OpenCV.
pub fn decode(data: &[u8]) -> Result<Mat> {
    let bad = |m: &str| Error::new(Code::StsParseError, format!("JPEG: {m}"));
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return Err(bad("missing the SOI marker"));
    }
    let mut qt = [[0u16; 64]; 4];
    let mut dc: Vec<HuffTable> = vec![HuffTable::default(); 4];
    let mut ac: Vec<HuffTable> = vec![HuffTable::default(); 4];
    let mut comps: Vec<Component> = Vec::new();
    let (mut width, mut height) = (0usize, 0usize);
    let mut restart_interval = 0usize;
    let mut i = 2usize;

    while i + 1 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = data[i + 1];
        i += 2;
        match marker {
            0xD8 | 0x01 | 0xD0..=0xD7 => continue,
            0xD9 => break,
            _ => {}
        }
        if i + 1 >= data.len() {
            return Err(bad("truncated segment header"));
        }
        let len = usize::from(data[i]) << 8 | usize::from(data[i + 1]);
        if len < 2 || i + len > data.len() {
            return Err(bad("segment length runs past the end of the file"));
        }
        let seg = &data[i + 2..i + len];

        match marker {
            // Quantisation tables.
            0xDB => {
                let mut p = 0;
                while p < seg.len() {
                    let pq = seg[p] >> 4;
                    let tq = usize::from(seg[p] & 15);
                    p += 1;
                    if tq >= 4 {
                        return Err(bad("quantisation table index out of range"));
                    }
                    for (k, slot) in qt[tq].iter_mut().enumerate() {
                        *slot = if pq == 0 {
                            u16::from(*seg.get(p + k).ok_or_else(|| bad("short DQT"))?)
                        } else {
                            let a = *seg.get(p + 2 * k).ok_or_else(|| bad("short DQT"))?;
                            let b = *seg.get(p + 2 * k + 1).ok_or_else(|| bad("short DQT"))?;
                            u16::from(a) << 8 | u16::from(b)
                        };
                    }
                    p += if pq == 0 { 64 } else { 128 };
                }
            }
            // Huffman tables.
            0xC4 => {
                let mut p = 0;
                while p + 17 <= seg.len() {
                    let class = seg[p] >> 4;
                    let id = usize::from(seg[p] & 15);
                    if id >= 4 {
                        return Err(bad("Huffman table index out of range"));
                    }
                    let mut counts = [0u8; 16];
                    counts.copy_from_slice(&seg[p + 1..p + 17]);
                    let total: usize = counts.iter().map(|&c| usize::from(c)).sum();
                    if p + 17 + total > seg.len() {
                        return Err(bad("short DHT"));
                    }
                    let values = seg[p + 17..p + 17 + total].to_vec();
                    let t = HuffTable::build(&counts, values);
                    if class == 0 {
                        dc[id] = t;
                    } else {
                        ac[id] = t;
                    }
                    p += 17 + total;
                }
            }
            0xDD => {
                if seg.len() >= 2 {
                    restart_interval = usize::from(seg[0]) << 8 | usize::from(seg[1]);
                }
            }
            // Baseline frame header.
            0xC0 | 0xC1 => {
                if seg.len() < 6 {
                    return Err(bad("short SOF"));
                }
                if seg[0] != 8 {
                    return Err(bad("only 8-bit samples are supported"));
                }
                height = usize::from(seg[1]) << 8 | usize::from(seg[2]);
                width = usize::from(seg[3]) << 8 | usize::from(seg[4]);
                let n = usize::from(seg[5]);
                if seg.len() < 6 + n * 3 {
                    return Err(bad("short component list"));
                }
                comps = (0..n)
                    .map(|c| Component {
                        id: seg[6 + c * 3],
                        h: usize::from(seg[7 + c * 3] >> 4).max(1),
                        v: usize::from(seg[7 + c * 3] & 15).max(1),
                        tq: usize::from(seg[8 + c * 3]).min(3),
                        td: 0,
                        ta: 0,
                    })
                    .collect();
            }
            // The variants this decoder deliberately refuses.
            0xC2 => return Err(bad("progressive JPEG is not supported")),
            0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                return Err(bad("only baseline sequential DCT is supported"))
            }
            // Start of scan: everything after this is entropy-coded.
            0xDA => {
                if width == 0 || height == 0 || comps.is_empty() {
                    return Err(bad("SOS before a valid SOF"));
                }
                let ns = usize::from(*seg.first().ok_or_else(|| bad("short SOS"))?);
                for s in 0..ns {
                    let cid = *seg.get(1 + s * 2).ok_or_else(|| bad("short SOS"))?;
                    let t = *seg.get(2 + s * 2).ok_or_else(|| bad("short SOS"))?;
                    if let Some(c) = comps.iter_mut().find(|c| c.id == cid) {
                        c.td = usize::from(t >> 4).min(3);
                        c.ta = usize::from(t & 15).min(3);
                    }
                }
                let scan = &data[i + len..];
                return decode_scan(scan, &comps, &qt, &dc, &ac, width, height, restart_interval);
            }
            _ => {}
        }
        i += len;
    }
    Err(bad("no start-of-scan marker found"))
}

/// Entropy-decode the scan and reconstruct the image.
#[allow(clippy::too_many_arguments)]
fn decode_scan(
    scan: &[u8],
    comps: &[Component],
    qt: &[[u16; 64]; 4],
    dc: &[HuffTable],
    ac: &[HuffTable],
    width: usize,
    height: usize,
    restart_interval: usize,
) -> Result<Mat> {
    let bad = |m: &str| Error::new(Code::StsParseError, format!("JPEG: {m}"));
    let hmax = comps.iter().map(|c| c.h).max().unwrap_or(1);
    let vmax = comps.iter().map(|c| c.v).max().unwrap_or(1);
    let mcu_w = 8 * hmax;
    let mcu_h = 8 * vmax;
    let mcus_x = width.div_ceil(mcu_w);
    let mcus_y = height.div_ceil(mcu_h);

    // Each component is stored at its *own* resolution and upsampled after the
    // scan. Scattering straight into a full-resolution plane would mean
    // replicating chroma samples, and replication is visibly worse than the
    // triangle filter libjpeg uses — on a 4:2:0 image it costs about 4x the
    // mean error against cv2.
    let cw: Vec<usize> = comps.iter().map(|c| mcus_x * c.h * 8).collect();
    let ch: Vec<usize> = comps.iter().map(|c| mcus_y * c.v * 8).collect();
    let mut planes: Vec<Vec<f64>> =
        comps.iter().enumerate().map(|(i, _)| vec![0.0; cw[i] * ch[i]]).collect();
    let mut pred = vec![0i32; comps.len()];
    let mut r = BitReader::new(scan);
    let mut since_restart = 0usize;

    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            if restart_interval > 0 && since_restart == restart_interval {
                // A restart resets the DC predictors and the bit alignment.
                r.align();
                pred.iter_mut().for_each(|p| *p = 0);
                since_restart = 0;
            }
            since_restart += 1;

            for (ci, c) in comps.iter().enumerate() {
                for by in 0..c.v {
                    for bx in 0..c.h {
                        // One 8x8 block.
                        let mut coef = [0.0f64; 64];
                        let q = &qt[c.tq];
                        let sym = r.huff(&dc[c.td])?;
                        let diff = r.receive_extend(sym)?;
                        pred[ci] += diff;
                        coef[0] = f64::from(pred[ci]) * f64::from(q[0]);

                        let mut k = 1usize;
                        while k < 64 {
                            let rs = r.huff(&ac[c.ta])?;
                            let (run, size) = (usize::from(rs >> 4), rs & 15);
                            if size == 0 {
                                if run == 15 {
                                    k += 16; // ZRL: sixteen zeroes
                                    continue;
                                }
                                break; // EOB
                            }
                            k += run;
                            if k >= 64 {
                                return Err(bad("AC coefficient index out of range"));
                            }
                            let v = r.receive_extend(size)?;
                            coef[ZIGZAG[k]] = f64::from(v) * f64::from(q[k]);
                            k += 1;
                        }

                        let mut px = [0.0f64; 64];
                        idct8x8(&coef, &mut px);

                        let ox = (mx * c.h + bx) * 8;
                        let oy = (my * c.v + by) * 8;
                        for yy in 0..8 {
                            for xx in 0..8 {
                                let (tx, ty) = (ox + xx, oy + yy);
                                if tx < cw[ci] && ty < ch[ci] {
                                    planes[ci][ty * cw[ci] + tx] = px[yy * 8 + xx] + 128.0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Upsample every component to full resolution with the triangle filter.
    let full: Vec<Vec<f64>> = comps
        .iter()
        .enumerate()
        .map(|(ci, c)| upsample(&planes[ci], cw[ci], ch[ci], hmax / c.h, vmax / c.v, width, height))
        .collect();
    let planes = full;

    // Colour transform. JPEG stores YCbCr; OpenCV hands back BGR.
    let cn: i32 = if comps.len() >= 3 { 3 } else { 1 };
    let mut out = Mat::zeros(
        height as i32,
        width as i32,
        MatType::new(Depth::U8, cn),
    )?;
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let p = out.pixel_mut::<u8>(y as i32, x as i32)?;
            if cn == 1 {
                p[0] = cv_core::saturate::Elem::from_f64(planes[0][i]);
            } else {
                let (yy, cb, cr) = (planes[0][i], planes[1][i] - 128.0, planes[2][i] - 128.0);
                let rr = yy + 1.402 * cr;
                let gg = yy - 0.344_136 * cb - 0.714_136 * cr;
                let bb = yy + 1.772 * cb;
                p[0] = cv_core::saturate::Elem::from_f64(bb);
                p[1] = cv_core::saturate::Elem::from_f64(gg);
                p[2] = cv_core::saturate::Elem::from_f64(rr);
            }
        }
    }
    Ok(out)
}

/// Triangle-filter (bilinear) upsampling of one component plane.
///
/// libjpeg calls this "fancy upsampling" and enables it by default; the
/// alternative is replicating each chroma sample over its whole block, which
/// leaves visible 2x2 steps along every colour edge. Sample centres sit at
/// `(i + 0.5) * factor - 0.5` in the output, which is what makes the
/// interpolation land symmetrically rather than shifted half a chroma pixel.
fn upsample(
    src: &[f64],
    sw: usize,
    sh: usize,
    fx: usize,
    fy: usize,
    dw: usize,
    dh: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; dw * dh];
    let at = |y: isize, x: isize| -> f64 {
        let yy = y.clamp(0, sh as isize - 1) as usize;
        let xx = x.clamp(0, sw as isize - 1) as usize;
        src[yy * sw + xx]
    };
    for y in 0..dh {
        // Position in source coordinates, centre-aligned.
        let sy = (y as f64 + 0.5) / fy as f64 - 0.5;
        let iy = sy.floor() as isize;
        let ay = sy - iy as f64;
        for x in 0..dw {
            let sx = (x as f64 + 0.5) / fx as f64 - 0.5;
            let ix = sx.floor() as isize;
            let ax = sx - ix as f64;
            out[y * dw + x] = at(iy, ix) * (1.0 - ax) * (1.0 - ay)
                + at(iy, ix + 1) * ax * (1.0 - ay)
                + at(iy + 1, ix) * (1.0 - ax) * ay
                + at(iy + 1, ix + 1) * ax * ay;
        }
    }
    out
}

/// Whether `data` starts with the JPEG signature.
#[must_use]
pub fn is_jpeg(data: &[u8]) -> bool {
    data.len() >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::MatLike;

    /// A tiny baseline JPEG produced by `tools/gen_jpeg_fixture.py`, if present.
    fn fixture(name: &str) -> Option<Vec<u8>> {
        std::fs::read(std::path::Path::new("/tmp/jpegtest").join(name)).ok()
    }

    #[test]
    fn signature_detection() {
        assert!(is_jpeg(&[0xFF, 0xD8, 0xFF, 0xE0]));
        assert!(!is_jpeg(&[0x89, b'P', b'N', b'G']));
        assert!(!is_jpeg(&[0xFF, 0xD8]));
    }

    #[test]
    fn a_non_jpeg_is_rejected_clearly() {
        let e = decode(&[0u8; 32]).unwrap_err();
        assert!(format!("{e}").contains("SOI"), "got {e}");
    }

    #[test]
    fn truncated_input_is_rejected() {
        assert!(decode(&[0xFF, 0xD8]).is_err());
        assert!(decode(&[0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0xFF]).is_err());
    }

    #[test]
    fn progressive_jpeg_is_refused_by_name() {
        // SOI, then an SOF2 (progressive) header.
        let d = [0xFF, 0xD8, 0xFF, 0xC2, 0x00, 0x0B, 8, 0, 8, 0, 8, 1, 1, 0x11, 0];
        let e = decode(&d).unwrap_err();
        assert!(
            format!("{e}").contains("progressive"),
            "an unsupported variant must say which one, got {e}"
        );
    }

    #[test]
    fn huffman_table_builds_canonical_codes() {
        // Two codes of length 2, one of length 3.
        let mut counts = [0u8; 16];
        counts[1] = 2;
        counts[2] = 1;
        let t = HuffTable::build(&counts, vec![10, 20, 30]);
        assert_eq!(t.mincode[2], 0);
        assert_eq!(t.maxcode[2], 1);
        assert_eq!(t.valptr[2], 0);
        assert_eq!(t.mincode[3], 4);
        assert_eq!(t.maxcode[3], 4);
        assert_eq!(t.maxcode[1], -1, "a length with no codes must be unmatchable");
    }

    #[test]
    fn receive_extend_uses_jpegs_sign_convention() {
        // 2 bits: 0b00 -> -3, 0b01 -> -2, 0b10 -> 2, 0b11 -> 3.
        let cases: [(u8, i32); 4] = [(0b00, -3), (0b01, -2), (0b10, 2), (0b11, 3)];
        for (bits, want) in cases {
            let buf = [bits << 6];
            let mut r = BitReader::new(&buf);
            assert_eq!(r.receive_extend(2).unwrap(), want, "bits {bits:02b}");
        }
    }

    #[test]
    fn the_bit_reader_unstuffs_ff00() {
        // 0xFF 0x00 is a literal 0xFF byte, not a marker.
        let mut r = BitReader::new(&[0xFF, 0x00, 0b1010_0000]);
        for _ in 0..8 {
            assert_eq!(r.bit().unwrap(), 1, "the stuffed byte should read as 0xFF");
        }
        assert_eq!(r.bit().unwrap(), 1);
        assert_eq!(r.bit().unwrap(), 0);
    }

    #[test]
    fn a_real_marker_in_the_scan_is_an_error() {
        let mut r = BitReader::new(&[0xFF, 0xC0]);
        assert!(r.bit().is_err(), "0xFF followed by a marker byte is not image data");
    }

    #[test]
    fn the_inverse_dct_of_a_dc_only_block_is_flat() {
        let mut inp = [0.0f64; 64];
        inp[0] = 8.0 * 128.0 / 8.0; // a DC term
        let mut out = [0.0f64; 64];
        idct8x8(&inp, &mut out);
        let first = out[0];
        for v in out {
            assert!((v - first).abs() < 1e-9, "a DC-only block must be constant");
        }
    }

    #[test]
    fn the_inverse_dct_inverts_the_forward_one() {
        // Forward DCT of a ramp, then back.
        let src: [f64; 64] = core::array::from_fn(|i| ((i * 7) % 61) as f64 - 30.0);
        let c = |u: usize| if u == 0 { std::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
        let mut fwd = [0.0f64; 64];
        for v in 0..8 {
            for u in 0..8 {
                let mut s = 0.0;
                for y in 0..8 {
                    for x in 0..8 {
                        s += src[y * 8 + x]
                            * ((2.0 * x as f64 + 1.0) * u as f64 * std::f64::consts::PI / 16.0).cos()
                            * ((2.0 * y as f64 + 1.0) * v as f64 * std::f64::consts::PI / 16.0)
                                .cos();
                    }
                }
                fwd[v * 8 + u] = 0.25 * c(u) * c(v) * s;
            }
        }
        let mut back = [0.0f64; 64];
        idct8x8(&fwd, &mut back);
        for i in 0..64 {
            assert!((back[i] - src[i]).abs() < 1e-9, "coefficient {i}: {} vs {}", back[i], src[i]);
        }
    }

    #[test]
    fn upsampling_interpolates_rather_than_replicating() {
        // Two chroma samples, 0 and 100, doubled to four outputs. Replication
        // would give [0, 0, 100, 100]; the triangle filter ramps between them.
        let out = upsample(&[0.0, 100.0], 2, 1, 2, 1, 4, 1);
        assert_eq!(out.len(), 4);
        assert!(out[0] < 1.0, "the first sample keeps its value: {out:?}");
        assert!(out[3] > 99.0, "so does the last: {out:?}");
        assert!(
            out[1] > 1.0 && out[1] < out[2] && out[2] < 99.0,
            "the middle must ramp, not step: {out:?}"
        );
    }

    #[test]
    fn upsampling_by_one_is_the_identity() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(upsample(&src, 2, 2, 1, 1, 2, 2), src);
    }

    #[test]
    fn zigzag_is_a_permutation() {
        let mut seen = [false; 64];
        for &z in &ZIGZAG {
            assert!(!seen[z], "{z} appears twice");
            seen[z] = true;
        }
        assert_eq!(ZIGZAG[0], 0, "the DC term comes first");
        assert_eq!(ZIGZAG[63], 63);
    }

    #[test]
    fn decodes_a_real_greyscale_jpeg() {
        let Some(d) = fixture("gray.jpg") else {
            eprintln!("skipping: run python3 tools/gen_jpeg_fixture.py first");
            return;
        };
        let m = decode(&d).expect("the fixture must decode");
        assert_eq!(m.channels(), 1);
        assert_eq!((m.rows(), m.cols()), (32, 32));
        // Compare against the raw pixels the fixture was made from.
        let raw = fixture("gray.raw").expect("fixture");
        let got = m.to_vec::<u8>().unwrap();
        let worst = got
            .iter()
            .zip(raw.iter())
            .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
            .max()
            .unwrap_or(0);
        // Measured worst-case against cv2's own decode is 1; the bound is
        // deliberately tight so a regression in the IDCT cannot hide.
        assert!(worst <= 2, "worst greyscale error {worst}");
    }

    #[test]
    fn decodes_a_real_colour_jpeg_in_bgr_order() {
        let Some(d) = fixture("color.jpg") else {
            eprintln!("skipping: run python3 tools/gen_jpeg_fixture.py first");
            return;
        };
        let m = decode(&d).expect("the fixture must decode");
        assert_eq!(m.channels(), 3);
        assert_eq!((m.rows(), m.cols()), (32, 32));
        let raw = fixture("color.raw").expect("fixture"); // BGR, as cv2 stores it
        let got = m.to_vec::<u8>().unwrap();
        assert_eq!(got.len(), raw.len());
        let worst = got
            .iter()
            .zip(raw.iter())
            .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
            .max()
            .unwrap_or(0);
        // Measured worst is 2 with triangle-filter chroma upsampling. It was
        // 16 with sample replication, which is what this bound guards.
        assert!(worst <= 4, "worst colour error {worst}");
        // Channel order: the fixture's top-left is a known blue-dominant pixel.
        let b = m.at_f64(0, 0, 0).unwrap();
        let r = m.at_f64(0, 0, 2).unwrap();
        assert!(
            (b - f64::from(raw[0])).abs() <= 20.0 && (r - f64::from(raw[2])).abs() <= 20.0,
            "channels look swapped: got B={b} R={r}, expected B={} R={}",
            raw[0],
            raw[2]
        );
    }

    #[test]
    fn decodes_a_subsampled_jpeg() {
        let Some(d) = fixture("subsampled.jpg") else {
            eprintln!("skipping: run python3 tools/gen_jpeg_fixture.py first");
            return;
        };
        let m = decode(&d).expect("4:2:0 should decode");
        assert_eq!((m.rows(), m.cols(), m.channels()), (32, 32, 3));
        let raw = fixture("color.raw").expect("fixture");
        let got = m.to_vec::<u8>().unwrap();
        // Chroma subsampling costs more than quantisation does.
        let mean = got
            .iter()
            .zip(raw.iter())
            .map(|(a, b)| f64::from((i32::from(*a) - i32::from(*b)).abs()))
            .sum::<f64>()
            / got.len() as f64;
        // Measured 1.49; replication gave 3.78.
        assert!(mean < 2.5, "mean 4:2:0 error {mean}");
    }
}
