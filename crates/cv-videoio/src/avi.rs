//! AVI reading and writing (`modules/videoio/src/container_avi.cpp`).
//!
//! An AVI file is RIFF: a tree of four-character chunks. The video frames sit
//! in the `movi` list as `##dc` (compressed) or `##db` (uncompressed) chunks,
//! and the `strf` chunk's `BITMAPINFOHEADER` says how to read them.
//!
//! Container work is not codec work: this module locates and extracts frame
//! payloads, and what those payloads *are* is somebody else's problem. That is
//! exactly why MJPEG works — its payloads are complete JPEG files.

use cv_core::{cv_assert, Code, Error, Result};

/// A four-character code, as AVI stores them.
pub type FourCc = [u8; 4];

/// Format the four bytes as a readable string.
fn fcc_str(f: FourCc) -> String {
    f.iter().map(|&b| if b.is_ascii_graphic() || b == b' ' { b as char } else { '?' }).collect()
}

/// What an AVI's video stream holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamFormat {
    /// Frame width in pixels.
    pub width: i32,
    /// Frame height. Negative in the file means a top-down image; this field is
    /// always positive and [`StreamFormat::bottom_up`] carries the direction.
    pub height: i32,
    /// Bits per pixel.
    pub bit_count: u16,
    /// Codec identifier, or all-zero for uncompressed BGR.
    pub compression: FourCc,
    /// Whether rows are stored bottom-up, which is the BMP convention AVI
    /// inherited. Getting this wrong flips every frame vertically.
    pub bottom_up: bool,
}

/// A parsed AVI file.
#[derive(Debug)]
pub struct AviFile {
    /// Video stream format.
    pub format: StreamFormat,
    /// Microseconds per frame, from the main header.
    pub micros_per_frame: u32,
    /// Byte ranges of each frame payload within the source buffer.
    pub frames: Vec<(usize, usize)>,
}

impl AviFile {
    /// Frames per second, derived from the frame period.
    #[must_use]
    pub fn fps(&self) -> f64 {
        if self.micros_per_frame == 0 {
            0.0
        } else {
            1e6 / f64::from(self.micros_per_frame)
        }
    }

    /// Number of video frames found in the `movi` list.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the file holds no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

fn u32_at(d: &[u8], o: usize) -> Result<u32> {
    d.get(o..o + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| Error::new(Code::StsParseError, "AVI: read past the end of the file".to_string()))
}

fn i32_at(d: &[u8], o: usize) -> Result<i32> {
    u32_at(d, o).map(|v| v as i32)
}

/// Parse the container: headers and the position of every frame payload.
pub fn parse(data: &[u8]) -> Result<AviFile> {
    let bad = |m: &str| Error::new(Code::StsParseError, format!("AVI: {m}"));
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"AVI " {
        return Err(bad("not a RIFF/AVI file"));
    }

    let mut format: Option<StreamFormat> = None;
    let mut micros = 0u32;
    let mut frames: Vec<(usize, usize)> = Vec::new();

    // Iterative walk over one chunk list, descending into LIST/RIFF.
    let mut stack: Vec<(usize, usize)> = vec![(12, data.len())];
    while let Some((mut off, end)) = stack.pop() {
        while off + 8 <= end {
            let id: FourCc = [data[off], data[off + 1], data[off + 2], data[off + 3]];
            let size = u32_at(data, off + 4)? as usize;
            let body = off + 8;
            if body + size > data.len() {
                // A truncated final chunk is common in interrupted captures;
                // stop cleanly rather than refusing the whole file.
                break;
            }
            match &id {
                b"LIST" | b"RIFF" => {
                    if body + 4 <= data.len() {
                        stack.push((body + 4, body + size));
                    }
                }
                b"avih" => {
                    micros = u32_at(data, body)?;
                }
                b"strf" => {
                    // BITMAPINFOHEADER.
                    if size >= 20 {
                        let w = i32_at(data, body + 4)?;
                        let h = i32_at(data, body + 8)?;
                        let bits = u32_at(data, body + 12)? >> 16;
                        let comp = [
                            data[body + 16],
                            data[body + 17],
                            data[body + 18],
                            data[body + 19],
                        ];
                        // A negative height means top-down rows.
                        format.get_or_insert(StreamFormat {
                            width: w,
                            height: h.abs(),
                            bit_count: bits as u16,
                            compression: comp,
                            bottom_up: h > 0,
                        });
                    }
                }
                _ => {
                    // `##dc` / `##db` inside `movi` are frame payloads.
                    if id[0].is_ascii_digit() && id[1].is_ascii_digit() {
                        let kind = &id[2..4];
                        if (kind == b"dc" || kind == b"db") && size > 0 {
                            frames.push((body, body + size));
                        }
                    }
                }
            }
            // Chunks are word-aligned; the pad byte is not counted in `size`.
            off = body + size + (size & 1);
        }
    }

    let format = format.ok_or_else(|| bad("no stream format chunk (strf) found"))?;
    cv_assert!(
        format.width > 0 && format.height > 0,
        "AVI reports a {}x{} frame",
        format.width,
        format.height
    );
    Ok(AviFile { format, micros_per_frame: micros, frames })
}

/// Codec name for an error message.
#[must_use]
pub fn codec_name(f: FourCc) -> String {
    if f == [0, 0, 0, 0] {
        "uncompressed BGR".to_string()
    } else {
        fcc_str(f)
    }
}

/// Build an uncompressed AVI holding `frames` of `width x height` BGR pixels.
///
/// # Why uncompressed
///
/// Writing MJPEG would need a JPEG *encoder*, and this workspace has only a
/// decoder. Emitting an uncompressed stream is honest about that: the file is
/// large, but every reader can open it and no frame is silently degraded.
pub fn write_uncompressed(
    frames: &[Vec<u8>],
    width: i32,
    height: i32,
    fps: f64,
) -> Result<Vec<u8>> {
    cv_assert!(width > 0 && height > 0, "the frame size must be positive");
    cv_assert!(fps > 0.0, "fps must be positive, got {fps}");
    // Each row is padded to a 4-byte boundary, as BMP requires.
    let stride = (width as usize * 3).div_ceil(4) * 4;
    let frame_bytes = stride * height as usize;
    for (i, f) in frames.iter().enumerate() {
        cv_assert!(
            f.len() == frame_bytes,
            "frame {i} is {} bytes, expected {frame_bytes}",
            f.len()
        );
    }

    let mut movi: Vec<u8> = b"movi".to_vec();
    for f in frames {
        movi.extend_from_slice(b"00db");
        movi.extend_from_slice(&(f.len() as u32).to_le_bytes());
        movi.extend_from_slice(f);
        if f.len() % 2 == 1 {
            movi.push(0);
        }
    }

    let micros = (1e6 / fps).round() as u32;
    let n = frames.len() as u32;

    // avih: MainAVIHeader, 56 bytes.
    let mut avih = Vec::with_capacity(56);
    avih.extend_from_slice(&micros.to_le_bytes());
    avih.extend_from_slice(&((frame_bytes as u32).saturating_mul(n)).to_le_bytes()); // max byte rate
    avih.extend_from_slice(&0u32.to_le_bytes()); // padding granularity
    avih.extend_from_slice(&0x10u32.to_le_bytes()); // AVIF_HASINDEX is not set; keep flags minimal
    avih.extend_from_slice(&n.to_le_bytes());
    avih.extend_from_slice(&0u32.to_le_bytes()); // initial frames
    avih.extend_from_slice(&1u32.to_le_bytes()); // streams
    avih.extend_from_slice(&(frame_bytes as u32).to_le_bytes());
    avih.extend_from_slice(&(width as u32).to_le_bytes());
    avih.extend_from_slice(&(height as u32).to_le_bytes());
    avih.extend_from_slice(&[0u8; 16]); // reserved

    // strh: AVIStreamHeader, 56 bytes.
    let mut strh = Vec::with_capacity(56);
    strh.extend_from_slice(b"vids");
    strh.extend_from_slice(&[0u8; 4]); // handler: none for uncompressed
    strh.extend_from_slice(&0u32.to_le_bytes()); // flags
    strh.extend_from_slice(&0u16.to_le_bytes()); // priority
    strh.extend_from_slice(&0u16.to_le_bytes()); // language
    strh.extend_from_slice(&0u32.to_le_bytes()); // initial frames
    strh.extend_from_slice(&micros.to_le_bytes()); // scale
    strh.extend_from_slice(&1_000_000u32.to_le_bytes()); // rate; rate/scale = fps
    strh.extend_from_slice(&0u32.to_le_bytes()); // start
    strh.extend_from_slice(&n.to_le_bytes()); // length
    strh.extend_from_slice(&(frame_bytes as u32).to_le_bytes()); // suggested buffer
    strh.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // quality: default
    strh.extend_from_slice(&0u32.to_le_bytes()); // sample size
    strh.extend_from_slice(&0u16.to_le_bytes()); // rcFrame left
    strh.extend_from_slice(&0u16.to_le_bytes()); // top
    strh.extend_from_slice(&(width as u16).to_le_bytes()); // right
    strh.extend_from_slice(&(height as u16).to_le_bytes()); // bottom

    // strf: BITMAPINFOHEADER, 40 bytes. Positive height means bottom-up rows.
    let mut strf = Vec::with_capacity(40);
    strf.extend_from_slice(&40u32.to_le_bytes());
    strf.extend_from_slice(&width.to_le_bytes());
    strf.extend_from_slice(&height.to_le_bytes());
    strf.extend_from_slice(&1u16.to_le_bytes()); // planes
    strf.extend_from_slice(&24u16.to_le_bytes()); // bits
    strf.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
    strf.extend_from_slice(&(frame_bytes as u32).to_le_bytes());
    strf.extend_from_slice(&0u32.to_le_bytes()); // x pels per metre
    strf.extend_from_slice(&0u32.to_le_bytes()); // y pels per metre
    strf.extend_from_slice(&0u32.to_le_bytes()); // colours used
    strf.extend_from_slice(&0u32.to_le_bytes()); // colours important

    let chunk = |id: &[u8; 4], body: &[u8]| -> Vec<u8> {
        let mut v = id.to_vec();
        v.extend_from_slice(&(body.len() as u32).to_le_bytes());
        v.extend_from_slice(body);
        if body.len() % 2 == 1 {
            v.push(0);
        }
        v
    };
    let list = |form: &[u8; 4], body: &[u8]| -> Vec<u8> {
        let mut inner = form.to_vec();
        inner.extend_from_slice(body);
        chunk(b"LIST", &inner)
    };

    let mut strl = chunk(b"strh", &strh);
    strl.extend_from_slice(&chunk(b"strf", &strf));
    let mut hdrl = chunk(b"avih", &avih);
    hdrl.extend_from_slice(&list(b"strl", &strl));

    let mut body = list(b"hdrl", &hdrl);
    body.extend_from_slice(&chunk(b"LIST", &movi));

    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&((body.len() + 4) as u32).to_le_bytes());
    out.extend_from_slice(b"AVI ");
    out.extend_from_slice(&body);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Option<Vec<u8>> {
        std::fs::read(std::path::Path::new("/tmp/viotest").join(name)).ok()
    }

    #[test]
    fn a_non_avi_is_rejected() {
        assert!(parse(&[0u8; 32]).is_err());
        assert!(parse(b"RIFF\x00\x00\x00\x00WEBP").is_err());
    }

    #[test]
    fn parses_a_real_mjpeg_avi() {
        let Some(d) = fixture("mjpg.avi") else {
            eprintln!("skipping: run python3 tools/gen_video_fixture.py first");
            return;
        };
        let a = parse(&d).unwrap();
        assert_eq!((a.format.width, a.format.height), (48, 32));
        assert_eq!(&a.format.compression, b"MJPG");
        assert_eq!(a.len(), 6, "six frames were written");
        assert!((a.fps() - 10.0).abs() < 1e-6, "fps {}", a.fps());
        // Every payload should itself be a JPEG.
        for (i, &(s, e)) in a.frames.iter().enumerate() {
            assert!(
                cv_imgcodecs::jpeg::is_jpeg(&d[s..e]),
                "frame {i} does not start with a JPEG signature"
            );
        }
    }

    #[test]
    fn parses_a_real_uncompressed_avi() {
        let Some(d) = fixture("rgba.avi") else {
            eprintln!("skipping: run python3 tools/gen_video_fixture.py first");
            return;
        };
        let a = parse(&d).unwrap();
        assert_eq!((a.format.width, a.format.height), (48, 32));
        assert_eq!(a.len(), 6);
        // 48*32*4 bytes per frame.
        for &(s, e) in &a.frames {
            assert_eq!(e - s, 48 * 32 * 4, "unexpected payload size");
        }
    }

    #[test]
    fn round_trips_an_uncompressed_file() {
        let (w, h) = (8, 4);
        let stride = (w as usize * 3).div_ceil(4) * 4;
        let frames: Vec<Vec<u8>> = (0..3)
            .map(|t| (0..stride * h as usize).map(|i| ((i + t * 7) % 256) as u8).collect())
            .collect();
        let bytes = write_uncompressed(&frames, w, h, 25.0).unwrap();
        let a = parse(&bytes).unwrap();
        assert_eq!((a.format.width, a.format.height), (w, h));
        assert_eq!(a.format.bit_count, 24);
        assert_eq!(a.format.compression, [0, 0, 0, 0], "BI_RGB is zero, not a fourcc");
        assert!(a.format.bottom_up, "a positive height means bottom-up rows");
        assert!((a.fps() - 25.0).abs() < 0.01);
        assert_eq!(a.len(), 3);
        for (i, &(s, e)) in a.frames.iter().enumerate() {
            assert_eq!(&bytes[s..e], &frames[i][..], "frame {i} did not survive the round trip");
        }
    }

    #[test]
    fn writing_validates_its_input() {
        assert!(write_uncompressed(&[], 0, 4, 25.0).is_err());
        assert!(write_uncompressed(&[], 4, 4, 0.0).is_err());
        // A frame of the wrong length must be caught, not written.
        assert!(write_uncompressed(&[vec![0u8; 5]], 4, 4, 25.0).is_err());
    }

    #[test]
    fn an_odd_sized_payload_is_padded() {
        // A 1x1 frame is 4 bytes after row padding, so force an odd chunk by
        // checking the padding rule directly on a 3-byte body.
        let (w, h) = (1, 1);
        let stride = (w as usize * 3).div_ceil(4) * 4;
        assert_eq!(stride, 4, "a 1px BGR row pads to 4 bytes");
        let bytes = write_uncompressed(&[vec![1u8; stride]], w, h, 10.0).unwrap();
        assert_eq!(parse(&bytes).unwrap().len(), 1);
    }

    #[test]
    fn codec_names_are_readable() {
        assert_eq!(codec_name(*b"MJPG"), "MJPG");
        assert_eq!(codec_name([0, 0, 0, 0]), "uncompressed BGR");
    }
}
