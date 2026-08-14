//! YUV4MPEG2 reading and writing.
//!
//! A plain-text header, then `FRAME\n` followed by raw planes. It is the format
//! video tools hand each other when nobody wants a codec in the way, and it is
//! about a hundred lines to support properly.

use cv_core::types::{Depth, MatType};
use cv_core::{cv_assert, Code, Error, Mat, MatLike, MatLikeMut, Result};

/// Chroma subsampling, from the header's `C` tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Chroma {
    /// `C420` — half resolution in both axes. The default when unstated.
    #[default]
    C420,
    /// `C422` — half horizontal.
    C422,
    /// `C444` — full resolution.
    C444,
    /// `Cmono` — luma only.
    Mono,
}

impl Chroma {
    /// Bytes one frame occupies at the given size.
    fn frame_bytes(self, w: usize, h: usize) -> usize {
        match self {
            Chroma::C420 => w * h + 2 * (w.div_ceil(2) * h.div_ceil(2)),
            Chroma::C422 => w * h + 2 * (w.div_ceil(2) * h),
            Chroma::C444 => 3 * w * h,
            Chroma::Mono => w * h,
        }
    }

    /// Chroma plane dimensions.
    fn chroma_size(self, w: usize, h: usize) -> (usize, usize) {
        match self {
            Chroma::C420 => (w.div_ceil(2), h.div_ceil(2)),
            Chroma::C422 => (w.div_ceil(2), h),
            Chroma::C444 => (w, h),
            Chroma::Mono => (0, 0),
        }
    }
}

/// A parsed Y4M stream.
#[derive(Debug)]
pub struct Y4mFile {
    data: Vec<u8>,
    /// Frame width.
    pub width: i32,
    /// Frame height.
    pub height: i32,
    /// Frames per second, from the `F` tag.
    pub fps: f64,
    /// Chroma layout.
    pub chroma: Chroma,
    /// Offset of each frame's pixel data.
    offsets: Vec<usize>,
}

impl Y4mFile {
    /// Frame count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Whether the stream holds no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Decode frame `i` to BGR.
    pub fn frame(&self, i: usize) -> Result<Mat> {
        let off = *self.offsets.get(i).ok_or_else(|| {
            Error::new(Code::StsOutOfRange, format!("Y4M frame {i} is past the end"))
        })?;
        let (w, h) = (self.width as usize, self.height as usize);
        let (cw, chh) = self.chroma.chroma_size(w, h);
        let y_plane = &self.data[off..off + w * h];
        let mut m = Mat::zeros(self.height, self.width, MatType::new(Depth::U8, 3))?;

        for y in 0..h {
            for x in 0..w {
                let luma = f64::from(y_plane[y * w + x]);
                let (cb, cr) = if self.chroma == Chroma::Mono {
                    (128.0, 128.0)
                } else {
                    // Nearest chroma sample; Y4M carries no interpolation hint.
                    let (sx, sy) = ((x * cw / w).min(cw - 1), (y * chh / h).min(chh - 1));
                    let ub = off + w * h + sy * cw + sx;
                    let vb = off + w * h + cw * chh + sy * cw + sx;
                    (f64::from(self.data[ub]), f64::from(self.data[vb]))
                };
                let (cb, cr) = (cb - 128.0, cr - 128.0);
                let p = m.pixel_mut::<u8>(y as i32, x as i32)?;
                p[0] = cv_core::saturate::Elem::from_f64(luma + 1.772 * cb);
                p[1] = cv_core::saturate::Elem::from_f64(luma - 0.344_136 * cb - 0.714_136 * cr);
                p[2] = cv_core::saturate::Elem::from_f64(luma + 1.402 * cr);
            }
        }
        Ok(m)
    }
}

/// Parse a Y4M stream.
pub fn parse(data: Vec<u8>) -> Result<Y4mFile> {
    let bad = |m: &str| Error::new(Code::StsParseError, format!("Y4M: {m}"));
    let nl = data.iter().position(|&b| b == b'\n').ok_or_else(|| bad("no header terminator"))?;
    let header = std::str::from_utf8(&data[..nl]).map_err(|_| bad("header is not UTF-8"))?;
    if !header.starts_with("YUV4MPEG2") {
        return Err(bad("missing the YUV4MPEG2 signature"));
    }

    let (mut w, mut h) = (0i32, 0i32);
    let mut fps = 25.0f64;
    let mut chroma = Chroma::default();
    for tag in header.split_whitespace().skip(1) {
        let (k, v) = tag.split_at(1);
        match k {
            "W" => w = v.parse().map_err(|_| bad("bad width"))?,
            "H" => h = v.parse().map_err(|_| bad("bad height"))?,
            "F" => {
                // "num:den" frames per second.
                let (a, b) = v.split_once(':').ok_or_else(|| bad("bad frame rate"))?;
                let (a, b): (f64, f64) =
                    (a.parse().map_err(|_| bad("bad frame rate"))?, b.parse().map_err(|_| bad("bad frame rate"))?);
                fps = if b != 0.0 { a / b } else { 0.0 };
            }
            "C" => {
                chroma = match v {
                    v if v.starts_with("420") => Chroma::C420,
                    v if v.starts_with("422") => Chroma::C422,
                    v if v.starts_with("444") => Chroma::C444,
                    "mono" => Chroma::Mono,
                    other => return Err(bad(&format!("unsupported chroma {other:?}"))),
                }
            }
            _ => {}
        }
    }
    cv_assert!(w > 0 && h > 0, "Y4M reports a {w}x{h} frame");

    let fb = chroma.frame_bytes(w as usize, h as usize);
    let mut offsets = Vec::new();
    let mut p = nl + 1;
    while p + 6 <= data.len() {
        if &data[p..p + 5] != b"FRAME" {
            break;
        }
        // The frame header runs to the next newline, and may carry parameters.
        let Some(rel) = data[p..].iter().position(|&b| b == b'\n') else { break };
        let start = p + rel + 1;
        if start + fb > data.len() {
            break; // truncated trailing frame
        }
        offsets.push(start);
        p = start + fb;
    }
    Ok(Y4mFile { data, width: w, height: h, fps, chroma, offsets })
}

/// Encode one BGR frame as a 4:2:0 Y4M payload.
pub fn encode_frame<S: MatLike + ?Sized>(frame: &S) -> Result<Vec<u8>> {
    let (w, h) = (frame.cols() as usize, frame.rows() as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = vec![0u8; w * h + 2 * cw * ch];
    // Luma at full resolution.
    for y in 0..h {
        for x in 0..w {
            let (b, g, r) = (
                frame.at_f64(y as i32, x as i32, 0)?,
                frame.at_f64(y as i32, x as i32, 1.min(frame.channels() - 1))?,
                frame.at_f64(y as i32, x as i32, 2.min(frame.channels() - 1))?,
            );
            out[y * w + x] =
                cv_core::saturate::Elem::from_f64(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
    // Chroma averaged over each 2x2 block, which is what subsampling means —
    // taking one corner instead would alias every colour edge.
    for cy in 0..ch {
        for cx in 0..cw {
            let (mut sb, mut sg, mut sr, mut n) = (0.0, 0.0, 0.0, 0.0);
            for dy in 0..2 {
                for dx in 0..2 {
                    let (y, x) = (cy * 2 + dy, cx * 2 + dx);
                    if y >= h || x >= w {
                        continue;
                    }
                    sb += frame.at_f64(y as i32, x as i32, 0)?;
                    sg += frame.at_f64(y as i32, x as i32, 1.min(frame.channels() - 1))?;
                    sr += frame.at_f64(y as i32, x as i32, 2.min(frame.channels() - 1))?;
                    n += 1.0;
                }
            }
            let (b, g, r) = (sb / n, sg / n, sr / n);
            let u = -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0;
            let v = 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0;
            out[w * h + cy * cw + cx] = cv_core::saturate::Elem::from_f64(u);
            out[w * h + cw * ch + cy * cw + cx] = cv_core::saturate::Elem::from_f64(v);
        }
    }
    Ok(out)
}

/// Assemble a Y4M file from encoded frames.
pub fn write(frames: &[Vec<u8>], width: i32, height: i32, fps: f64) -> Result<Vec<u8>> {
    cv_assert!(width > 0 && height > 0, "the frame size must be positive");
    cv_assert!(fps > 0.0, "fps must be positive");
    // Express the rate as a ratio, which is what the format wants.
    let num = (fps * 1000.0).round() as u64;
    let mut out = format!("YUV4MPEG2 W{width} H{height} F{num}:1000 Ip A1:1 C420\n").into_bytes();
    for f in frames {
        out.extend_from_slice(b"FRAME\n");
        out.extend_from_slice(f);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bgr(w: i32, h: i32) -> Mat {
        let mut m = Mat::zeros(h, w, MatType::new(Depth::U8, 3)).unwrap();
        for y in 0..h {
            for x in 0..w {
                let p = m.pixel_mut::<u8>(y, x).unwrap();
                p[0] = ((x * 9) % 256) as u8;
                p[1] = ((y * 13) % 256) as u8;
                p[2] = ((x + y) * 5 % 256) as u8;
            }
        }
        m
    }

    #[test]
    fn header_parsing_reads_every_tag() {
        let mut d = b"YUV4MPEG2 W4 H2 F30000:1001 Ip A1:1 C420\n".to_vec();
        d.extend_from_slice(b"FRAME\n");
        d.extend_from_slice(&[128u8; 4 * 2 + 2 * 2]);
        let f = parse(d).unwrap();
        assert_eq!((f.width, f.height), (4, 2));
        assert!((f.fps - 29.97).abs() < 0.01, "fps {}", f.fps);
        assert_eq!(f.chroma, Chroma::C420);
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn a_bad_header_is_rejected() {
        assert!(parse(b"not y4m\n".to_vec()).is_err());
        assert!(parse(b"YUV4MPEG2 W0 H0\n".to_vec()).is_err());
        assert!(parse(b"YUV4MPEG2 W4 H4 Cbogus\n".to_vec()).is_err());
    }

    #[test]
    fn frame_sizes_follow_the_chroma_layout() {
        assert_eq!(Chroma::C444.frame_bytes(4, 4), 48);
        assert_eq!(Chroma::C422.frame_bytes(4, 4), 16 + 16);
        assert_eq!(Chroma::C420.frame_bytes(4, 4), 16 + 8);
        assert_eq!(Chroma::Mono.frame_bytes(4, 4), 16);
        // Odd sizes round up, so a 5-wide 4:2:0 frame has 3 chroma columns.
        assert_eq!(Chroma::C420.chroma_size(5, 5), (3, 3));
    }

    #[test]
    fn a_grey_frame_round_trips_exactly() {
        // Grey has no chroma to lose, so the round trip should be exact.
        let mut m = Mat::zeros(4, 6, MatType::new(Depth::U8, 3)).unwrap();
        for y in 0..4 {
            for x in 0..6 {
                let v = ((x * 20 + y * 7) % 200 + 20) as u8;
                let p = m.pixel_mut::<u8>(y, x).unwrap();
                p[0] = v;
                p[1] = v;
                p[2] = v;
            }
        }
        let enc = encode_frame(&m).unwrap();
        let file = write(&[enc], 6, 4, 25.0).unwrap();
        let f = parse(file).unwrap();
        assert_eq!(f.len(), 1);
        let back = f.frame(0).unwrap();
        for y in 0..4 {
            for x in 0..6 {
                for c in 0..3 {
                    let (a, b) = (m.at_f64(y, x, c).unwrap(), back.at_f64(y, x, c).unwrap());
                    assert!((a - b).abs() <= 2.0, "({x},{y},{c}): {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn a_colour_frame_survives_subsampling_approximately() {
        let m = bgr(8, 6);
        let enc = encode_frame(&m).unwrap();
        let f = parse(write(&[enc], 8, 6, 30.0).unwrap()).unwrap();
        assert!((f.fps - 30.0).abs() < 0.01);
        let back = f.frame(0).unwrap();
        assert_eq!((back.rows(), back.cols(), back.channels()), (6, 8, 3));
        // 4:2:0 loses chroma detail; luma should stay close.
        let mut worst_luma = 0.0f64;
        for y in 0..6 {
            for x in 0..8 {
                let lum = |m: &Mat| {
                    0.114 * m.at_f64(y, x, 0).unwrap()
                        + 0.587 * m.at_f64(y, x, 1).unwrap()
                        + 0.299 * m.at_f64(y, x, 2).unwrap()
                };
                worst_luma = worst_luma.max((lum(&m) - lum(&back)).abs());
            }
        }
        assert!(worst_luma < 12.0, "luma drifted by {worst_luma}");
    }

    #[test]
    fn multiple_frames_are_indexed_independently() {
        let a = bgr(4, 4);
        let mut b = Mat::zeros(4, 4, MatType::new(Depth::U8, 3)).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                b.pixel_mut::<u8>(y, x).unwrap()[1] = 200;
            }
        }
        let file = write(&[encode_frame(&a).unwrap(), encode_frame(&b).unwrap()], 4, 4, 10.0)
            .unwrap();
        let f = parse(file).unwrap();
        assert_eq!(f.len(), 2);
        let (f0, f1) = (f.frame(0).unwrap(), f.frame(1).unwrap());
        assert_ne!(f0.to_vec::<u8>().unwrap(), f1.to_vec::<u8>().unwrap());
        assert!(f.frame(2).is_err(), "an out-of-range index must be an error");
    }

    #[test]
    fn a_truncated_final_frame_is_dropped_not_decoded() {
        let mut d = write(&[encode_frame(&bgr(4, 4)).unwrap()], 4, 4, 10.0).unwrap();
        d.extend_from_slice(b"FRAME\n");
        d.extend_from_slice(&[0u8; 3]); // far short of a full frame
        let f = parse(d).unwrap();
        assert_eq!(f.len(), 1, "the incomplete frame must not be indexed");
    }
}
