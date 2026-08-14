//! `cv::videoio` for `opencv-no-c` — capture and writing, without a codec.
//!
//! # What is here, and why that is the line
//!
//! `videoio` is usually the module a pure-Rust port writes off, because
//! H.264, HEVC and VP9 are enormous. But a lot of what `cv::VideoCapture`
//! actually does is *container* work, and containers are tractable:
//!
//! * **MJPEG AVI** — every frame is a complete JPEG, so the container walk plus
//!   [`cv_imgcodecs::jpeg`] is a real video decoder. This is the common output
//!   of `cv::VideoWriter` and of many industrial cameras.
//! * **Uncompressed AVI** — raw BGR or BGRA frames.
//! * **Y4M** — raw YUV4MPEG, the format video tooling passes around uncompressed.
//! * **Image sequences** — `cv::CAP_IMAGES`, driven by a `printf` pattern.
//!   Upstream treats these as a first-class capture backend and so does this.
//!
//! # What is not here
//!
//! Any *inter-frame* codec: H.264, HEVC, VP8/9, AV1, MPEG-4. Those need
//! motion compensation, in-loop filtering and entropy coders that are each
//! larger than this whole workspace, and a partial one would decode the first
//! keyframe and then drift. [`VideoCapture::open`] names the codec it found
//! rather than returning noise.
//!
//! Live camera capture is also absent: it needs AVFoundation, V4L2 or
//! DirectShow, none of which is reachable without linking C.

#![warn(missing_docs)]

pub mod constants;
pub mod backend;

pub mod avi;
pub mod y4m;

use std::path::{Path, PathBuf};

use cv_core::types::{Depth, MatType};
use cv_core::{cv_assert, Code, Error, Mat, MatLike, MatLikeMut, Result};

/// `cv::VideoWriter::fourcc(c1, c2, c3, c4)`.
#[must_use]
pub fn fourcc(a: char, b: char, c: char, d: char) -> u32 {
    u32::from(a as u8) | u32::from(b as u8) << 8 | u32::from(c as u8) << 16 | u32::from(d as u8) << 24
}

/// `cv::VideoCaptureProperties` — the subset that means something here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapProp {
    /// `cv::CAP_PROP_POS_FRAMES` — index of the next frame to be read.
    PosFrames,
    /// `cv::CAP_PROP_FRAME_WIDTH`
    FrameWidth,
    /// `cv::CAP_PROP_FRAME_HEIGHT`
    FrameHeight,
    /// `cv::CAP_PROP_FPS`
    Fps,
    /// `cv::CAP_PROP_FRAME_COUNT`
    FrameCount,
    /// `cv::CAP_PROP_FOURCC`
    FourCc,
}

/// Where a [`VideoCapture`] is reading from.
enum Source {
    /// An AVI held in memory, with the parsed container.
    Avi { data: Vec<u8>, file: avi::AviFile },
    /// A Y4M stream.
    Y4m(y4m::Y4mFile),
    /// A `printf`-style image sequence, with the paths already resolved.
    Images(Vec<PathBuf>),
}

/// `cv::VideoCapture`.
pub struct VideoCapture {
    source: Source,
    pos: usize,
}

impl std::fmt::Debug for VideoCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.source {
            Source::Avi { file, .. } => format!("AVI {}", avi::codec_name(file.format.compression)),
            Source::Y4m(_) => "Y4M".to_string(),
            Source::Images(v) => format!("images ({} frames)", v.len()),
        };
        f.debug_struct("VideoCapture")
            .field("source", &kind)
            .field("pos", &self.pos)
            .field("frames", &self.frame_count())
            .finish()
    }
}

impl VideoCapture {
    /// `cv::VideoCapture::open(filename)`.
    ///
    /// A `printf` pattern such as `frames/img_%04d.png` opens an image
    /// sequence, matching upstream's `CAP_IMAGES` backend.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let p = path.as_ref();
        let s = p.to_string_lossy();
        if s.contains('%') {
            return Self::open_sequence(&s);
        }
        let data = std::fs::read(p).map_err(|e| {
            Error::new(Code::StsObjectNotFound, format!("cannot open {}: {e}", p.display()))
        })?;
        Self::open_bytes(data)
    }

    /// Open from an in-memory file.
    pub fn open_bytes(data: Vec<u8>) -> Result<Self> {
        if data.len() >= 9 && &data[..9] == b"YUV4MPEG2" {
            return Ok(VideoCapture { source: Source::Y4m(y4m::parse(data)?), pos: 0 });
        }
        let file = avi::parse(&data)?;
        // Refuse an inter-frame codec by name rather than returning garbage.
        let c = file.format.compression;
        let supported = matches!(&c, b"MJPG" | b"mjpg" | b"jpeg" | b"JPEG")
            || c == [0, 0, 0, 0]
            || matches!(&c, b"DIB " | b"RGBA" | b"RGB2" | b"raw ");
        if !supported {
            return Err(Error::new(
                Code::StsNotImplemented,
                format!(
                    "AVI codec {:?} is not supported; this port decodes MJPEG and uncompressed \
                     streams only (inter-frame codecs need motion compensation)",
                    avi::codec_name(c)
                ),
            ));
        }
        Ok(VideoCapture { source: Source::Avi { data, file }, pos: 0 })
    }

    /// Open a `printf`-pattern image sequence, e.g. `shot_%03d.png`.
    ///
    /// Indices start at 0 and stop at the first gap, which is upstream's rule.
    pub fn open_sequence(pattern: &str) -> Result<Self> {
        let mut paths = Vec::new();
        for i in 0..100_000 {
            let p = PathBuf::from(expand(pattern, i));
            if !p.exists() {
                // Allow a sequence to start at 1 instead of 0.
                if i == 0 {
                    continue;
                }
                break;
            }
            paths.push(p);
        }
        if paths.is_empty() {
            return Err(Error::new(
                Code::StsObjectNotFound,
                format!("no files matched the pattern {pattern:?}"),
            ));
        }
        Ok(VideoCapture { source: Source::Images(paths), pos: 0 })
    }

    /// `cv::VideoCapture::isOpened()` — always true for a constructed capture.
    #[must_use]
    pub fn is_opened(&self) -> bool {
        true
    }

    /// `cv::VideoCapture::get(CAP_PROP_FRAME_COUNT)`.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        match &self.source {
            Source::Avi { file, .. } => file.len(),
            Source::Y4m(f) => f.len(),
            Source::Images(v) => v.len(),
        }
    }

    /// `cv::VideoCapture::get(propId)`.
    #[must_use]
    pub fn get(&self, prop: CapProp) -> f64 {
        match prop {
            CapProp::PosFrames => self.pos as f64,
            CapProp::FrameCount => self.frame_count() as f64,
            CapProp::FrameWidth => match &self.source {
                Source::Avi { file, .. } => f64::from(file.format.width),
                Source::Y4m(f) => f64::from(f.width),
                Source::Images(_) => 0.0,
            },
            CapProp::FrameHeight => match &self.source {
                Source::Avi { file, .. } => f64::from(file.format.height),
                Source::Y4m(f) => f64::from(f.height),
                Source::Images(_) => 0.0,
            },
            CapProp::Fps => match &self.source {
                Source::Avi { file, .. } => file.fps(),
                Source::Y4m(f) => f.fps,
                Source::Images(_) => 0.0,
            },
            CapProp::FourCc => match &self.source {
                Source::Avi { file, .. } => {
                    let c = file.format.compression;
                    u32::from_le_bytes(c) as f64
                }
                _ => 0.0,
            },
        }
    }

    /// `cv::VideoCapture::set(CAP_PROP_POS_FRAMES, index)`.
    pub fn set_pos(&mut self, index: usize) -> Result<()> {
        cv_assert!(
            index <= self.frame_count(),
            "frame {index} is past the end of a {}-frame stream",
            self.frame_count()
        );
        self.pos = index;
        Ok(())
    }

    /// `cv::VideoCapture::read(image)`.
    ///
    /// Returns `None` at the end of the stream, as upstream's `false` return
    /// signals.
    pub fn read(&mut self) -> Result<Option<Mat>> {
        if self.pos >= self.frame_count() {
            return Ok(None);
        }
        let i = self.pos;
        self.pos += 1;
        let m = match &self.source {
            Source::Avi { data, file } => {
                let (s, e) = file.frames[i];
                decode_avi_frame(&data[s..e], &file.format)?
            }
            Source::Y4m(f) => f.frame(i)?,
            Source::Images(v) => cv_imgcodecs::imread(&v[i], cv_imgcodecs::ImreadMode::Color)?,
        };
        Ok(Some(m))
    }

    /// `cv::VideoCapture::grab()` then `retrieve()`, collapsed: read everything.
    pub fn read_all(&mut self) -> Result<Vec<Mat>> {
        let mut out = Vec::with_capacity(self.frame_count().saturating_sub(self.pos));
        while let Some(m) = self.read()? {
            out.push(m);
        }
        Ok(out)
    }
}

/// Substitute a frame index into a `printf`-style pattern.
///
/// Handles `%d` and `%0Nd`, which is what upstream's image backend accepts.
fn expand(pattern: &str, i: usize) -> String {
    let Some(pct) = pattern.find('%') else {
        return pattern.to_string();
    };
    let rest = &pattern[pct + 1..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    let after = &rest[digits.len()..];
    if !after.starts_with('d') {
        return pattern.to_string();
    }
    let width: usize = digits.trim_start_matches('0').parse().unwrap_or(0);
    let num = if digits.starts_with('0') && width > 0 {
        format!("{i:0width$}")
    } else {
        i.to_string()
    };
    format!("{}{}{}", &pattern[..pct], num, &after[1..])
}

/// Turn one AVI frame payload into a `Mat`.
fn decode_avi_frame(payload: &[u8], f: &avi::StreamFormat) -> Result<Mat> {
    // MJPEG: the payload is a whole JPEG file.
    if cv_imgcodecs::jpeg::is_jpeg(payload) {
        return cv_imgcodecs::jpeg::decode(payload);
    }
    let (w, h) = (f.width, f.height);
    let bytes_per_px = usize::from(f.bit_count.max(24)) / 8;
    cv_assert!(
        bytes_per_px == 3 || bytes_per_px == 4,
        "uncompressed AVI with {} bits per pixel is not supported",
        f.bit_count
    );
    // Rows are padded to a 4-byte boundary.
    let stride = (w as usize * bytes_per_px).div_ceil(4) * 4;
    cv_assert!(
        payload.len() >= stride * h as usize,
        "frame payload is {} bytes, expected at least {}",
        payload.len(),
        stride * h as usize
    );
    let mut m = Mat::zeros(h, w, MatType::new(Depth::U8, 3))?;
    for y in 0..h {
        // Bottom-up storage: the first row in the file is the last on screen.
        let src_row = if f.bottom_up { h - 1 - y } else { y };
        let base = src_row as usize * stride;
        for x in 0..w {
            let o = base + x as usize * bytes_per_px;
            let p = m.pixel_mut::<u8>(y, x)?;
            p[0] = payload[o];
            p[1] = payload[o + 1];
            p[2] = payload[o + 2];
        }
    }
    Ok(m)
}

/// What a [`VideoWriter`] produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriterFormat {
    /// Uncompressed BGR in an AVI container.
    UncompressedAvi,
    /// One image file per frame, from a `printf` pattern.
    ImageSequence,
    /// Raw YUV4MPEG.
    Y4m,
}

/// `cv::VideoWriter`.
pub struct VideoWriter {
    format: WriterFormat,
    path: PathBuf,
    width: i32,
    height: i32,
    fps: f64,
    frames: Vec<Vec<u8>>,
    written: usize,
}

impl std::fmt::Debug for VideoWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoWriter")
            .field("format", &self.format)
            .field("size", &(self.width, self.height))
            .field("fps", &self.fps)
            .field("written", &self.written)
            .finish()
    }
}

impl VideoWriter {
    /// `cv::VideoWriter::open(filename, fourcc, fps, frameSize)`.
    ///
    /// A path containing `%` writes an image sequence; a `.y4m` extension
    /// writes YUV4MPEG; anything else writes an uncompressed AVI.
    pub fn open<P: AsRef<Path>>(path: P, fps: f64, width: i32, height: i32) -> Result<Self> {
        cv_assert!(width > 0 && height > 0, "the frame size must be positive");
        cv_assert!(fps > 0.0, "fps must be positive, got {fps}");
        let p = path.as_ref().to_path_buf();
        let s = p.to_string_lossy().to_string();
        let format = if s.contains('%') {
            WriterFormat::ImageSequence
        } else if s.to_ascii_lowercase().ends_with(".y4m") {
            WriterFormat::Y4m
        } else {
            WriterFormat::UncompressedAvi
        };
        Ok(VideoWriter { format, path: p, width, height, fps, frames: Vec::new(), written: 0 })
    }

    /// `cv::VideoWriter::isOpened()`.
    #[must_use]
    pub fn is_opened(&self) -> bool {
        true
    }

    /// Which container this writer produces.
    #[must_use]
    pub fn format(&self) -> WriterFormat {
        self.format
    }

    /// `cv::VideoWriter::write(image)`.
    ///
    /// For an image sequence the file is written immediately; for a container
    /// the frames are buffered until [`VideoWriter::release`].
    pub fn write<S: MatLike + ?Sized>(&mut self, frame: &S) -> Result<()> {
        cv_assert!(
            frame.cols() == self.width && frame.rows() == self.height,
            "frame is {}x{}, the writer was opened for {}x{}",
            frame.cols(),
            frame.rows(),
            self.width,
            self.height
        );
        match self.format {
            WriterFormat::ImageSequence => {
                let p = expand(&self.path.to_string_lossy(), self.written);
                cv_imgcodecs::imwrite(&p, frame)?;
            }
            WriterFormat::UncompressedAvi => {
                let bpp = 3usize;
                let stride = (self.width as usize * bpp).div_ceil(4) * 4;
                let mut buf = vec![0u8; stride * self.height as usize];
                for y in 0..self.height {
                    // Bottom-up, matching the header this writer emits.
                    let dst = (self.height - 1 - y) as usize * stride;
                    for x in 0..self.width {
                        let o = dst + x as usize * bpp;
                        for c in 0..3 {
                            let v = if frame.channels() > c { frame.at_f64(y, x, c)? } else { frame.at_f64(y, x, 0)? };
                            buf[o + c as usize] = cv_core::saturate::Elem::from_f64(v);
                        }
                    }
                }
                self.frames.push(buf);
            }
            WriterFormat::Y4m => {
                self.frames.push(y4m::encode_frame(frame)?);
            }
        }
        self.written += 1;
        Ok(())
    }

    /// `cv::VideoWriter::release()` — flush the container to disk.
    pub fn release(&mut self) -> Result<()> {
        match self.format {
            WriterFormat::ImageSequence => Ok(()),
            WriterFormat::UncompressedAvi => {
                let bytes =
                    avi::write_uncompressed(&self.frames, self.width, self.height, self.fps)?;
                std::fs::write(&self.path, bytes).map_err(|e| {
                    Error::new(Code::StsError, format!("cannot write {}: {e}", self.path.display()))
                })
            }
            WriterFormat::Y4m => {
                let bytes = y4m::write(&self.frames, self.width, self.height, self.fps)?;
                std::fs::write(&self.path, bytes).map_err(|e| {
                    Error::new(Code::StsError, format!("cannot write {}: {e}", self.path.display()))
                })
            }
        }
    }

    /// Frames accepted so far.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.written
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Option<PathBuf> {
        let p = PathBuf::from("/tmp/viotest").join(name);
        p.exists().then_some(p)
    }

    #[test]
    fn fourcc_packs_little_endian() {
        // cv2.VideoWriter_fourcc(*'MJPG') == 1196444237.
        assert_eq!(fourcc('M', 'J', 'P', 'G'), 1_196_444_237);
    }

    #[test]
    fn pattern_expansion_handles_width_and_padding() {
        assert_eq!(expand("img_%03d.png", 7), "img_007.png");
        assert_eq!(expand("img_%d.png", 7), "img_7.png");
        assert_eq!(expand("a/%04d/b.png", 12), "a/0012/b.png");
        assert_eq!(expand("no_pattern.png", 3), "no_pattern.png");
    }

    #[test]
    fn reads_a_real_mjpeg_avi_frame_by_frame() {
        let Some(p) = fixture("mjpg.avi") else {
            eprintln!("skipping: run python3 tools/gen_video_fixture.py first");
            return;
        };
        let mut c = VideoCapture::open(&p).unwrap();
        assert!(c.is_opened());
        assert_eq!(c.frame_count(), 6);
        assert_eq!(c.get(CapProp::FrameWidth), 48.0);
        assert_eq!(c.get(CapProp::FrameHeight), 32.0);
        assert!((c.get(CapProp::Fps) - 10.0).abs() < 1e-6);

        let mut n = 0;
        while let Some(m) = c.read().unwrap() {
            assert_eq!((m.rows(), m.cols(), m.channels()), (32, 48, 3));
            n += 1;
        }
        assert_eq!(n, 6);
        assert!(c.read().unwrap().is_none(), "reading past the end yields None");
        assert_eq!(c.get(CapProp::PosFrames), 6.0);
    }

    /// MJPEG frames decode to what libjpeg gets from the same bytes.
    ///
    /// The reference is `cv2.imdecode` on each extracted payload, **not**
    /// `cv2.VideoCapture`. Those two disagree: VideoCapture goes through
    /// FFmpeg's MJPEG decoder, imdecode through libjpeg, and on this fixture
    /// they differ by up to **29 levels on identical bytes**. Measuring
    /// against FFmpeg would be measuring the gap between two of OpenCV's own
    /// decoders, not this one's accuracy.
    #[test]
    fn mjpeg_frames_match_libjpegs_decode() {
        let (Some(p), Ok(raw)) =
            (fixture("mjpg.avi"), std::fs::read("/tmp/viotest/frames_libjpeg.raw"))
        else {
            eprintln!("skipping: run python3 tools/gen_video_fixture.py first");
            return;
        };
        let mut c = VideoCapture::open(&p).unwrap();
        let frames = c.read_all().unwrap();
        assert_eq!(frames.len(), 6);
        let per = 32 * 48 * 3;
        for (i, m) in frames.iter().enumerate() {
            let got = m.to_vec::<u8>().unwrap();
            let want = &raw[i * per..(i + 1) * per];
            let worst = got
                .iter()
                .zip(want)
                .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
                .max()
                .unwrap_or(0);
            // Measured worst across the six frames is 8, mean about 0.1.
            assert!(worst <= 10, "frame {i} worst error {worst}");
        }
    }

    #[test]
    fn reads_an_uncompressed_avi() {
        let Some(p) = fixture("rgba.avi") else {
            eprintln!("skipping: run python3 tools/gen_video_fixture.py first");
            return;
        };
        let mut c = VideoCapture::open(&p).unwrap();
        let frames = c.read_all().unwrap();
        assert_eq!(frames.len(), 6);
        for m in &frames {
            assert_eq!((m.rows(), m.cols(), m.channels()), (32, 48, 3));
        }
    }

    #[test]
    fn seeking_replays_the_same_frame() {
        let Some(p) = fixture("mjpg.avi") else { return };
        let mut c = VideoCapture::open(&p).unwrap();
        c.set_pos(3).unwrap();
        let a = c.read().unwrap().unwrap();
        c.set_pos(3).unwrap();
        let b = c.read().unwrap().unwrap();
        assert_eq!(a.to_vec::<u8>().unwrap(), b.to_vec::<u8>().unwrap());
        assert!(c.set_pos(999).is_err(), "seeking past the end must fail");
    }

    #[test]
    fn an_unsupported_codec_is_named_not_guessed() {
        // A minimal AVI header claiming H.264.
        let frames: Vec<Vec<u8>> = vec![vec![0u8; 16]];
        let mut bytes = avi::write_uncompressed(&frames, 2, 2, 10.0).unwrap();
        // Overwrite BI_RGB with 'H264' in the strf compression field.
        let pos = bytes.windows(4).position(|w| w == b"strf").unwrap();
        let comp = pos + 8 + 16;
        bytes[comp..comp + 4].copy_from_slice(b"H264");
        let e = VideoCapture::open_bytes(bytes).unwrap_err();
        assert_eq!(e.code, Code::StsNotImplemented);
        assert!(e.msg.contains("H264"), "the error must name the codec: {}", e.msg);
    }

    #[test]
    fn a_missing_file_is_a_clear_error() {
        let e = VideoCapture::open("/nonexistent/movie.avi").unwrap_err();
        assert_eq!(e.code, Code::StsObjectNotFound);
    }

    #[test]
    fn reads_an_image_sequence() {
        if fixture("seq_000.png").is_none() {
            eprintln!("skipping: run python3 tools/gen_video_fixture.py first");
            return;
        }
        let mut c = VideoCapture::open("/tmp/viotest/seq_%03d.png").unwrap();
        assert_eq!(c.frame_count(), 6);
        let frames = c.read_all().unwrap();
        assert_eq!(frames.len(), 6);
        assert_eq!((frames[0].rows(), frames[0].cols()), (32, 48));
    }

    #[test]
    fn an_empty_pattern_is_an_error() {
        assert!(VideoCapture::open("/tmp/viotest/nothing_%03d.png").is_err());
    }

    #[test]
    fn writes_and_reads_back_an_uncompressed_avi() {
        let dir = std::env::temp_dir().join("cv_videoio_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.avi");
        let (w, h) = (16, 12);

        let mut originals = Vec::new();
        let mut wr = VideoWriter::open(&path, 20.0, w, h).unwrap();
        assert_eq!(wr.format(), WriterFormat::UncompressedAvi);
        for t in 0..4 {
            let mut m = Mat::zeros(h, w, MatType::new(Depth::U8, 3)).unwrap();
            for y in 0..h {
                for x in 0..w {
                    let p = m.pixel_mut::<u8>(y, x).unwrap();
                    p[0] = ((x * 7 + t * 30) % 256) as u8;
                    p[1] = ((y * 11) % 256) as u8;
                    p[2] = ((x + y + t * 5) % 256) as u8;
                }
            }
            wr.write(&m).unwrap();
            originals.push(m);
        }
        assert_eq!(wr.frame_count(), 4);
        wr.release().unwrap();

        let mut c = VideoCapture::open(&path).unwrap();
        assert_eq!(c.frame_count(), 4);
        assert!((c.get(CapProp::Fps) - 20.0).abs() < 0.01);
        let back = c.read_all().unwrap();
        assert_eq!(back.len(), 4);
        for (i, (a, b)) in originals.iter().zip(back.iter()).enumerate() {
            assert_eq!(
                a.to_vec::<u8>().unwrap(),
                b.to_vec::<u8>().unwrap(),
                "frame {i} did not survive an *uncompressed* round trip"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_wrong_sized_frame_is_rejected() {
        let dir = std::env::temp_dir().join("cv_videoio_bad");
        std::fs::create_dir_all(&dir).unwrap();
        let mut wr = VideoWriter::open(dir.join("x.avi"), 10.0, 8, 8).unwrap();
        let wrong = Mat::zeros(4, 4, MatType::new(Depth::U8, 3)).unwrap();
        assert!(wr.write(&wrong).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writer_validates_its_geometry() {
        assert!(VideoWriter::open("/tmp/x.avi", 10.0, 0, 8).is_err());
        assert!(VideoWriter::open("/tmp/x.avi", 0.0, 8, 8).is_err());
    }

    #[test]
    fn writes_an_image_sequence() {
        let dir = std::env::temp_dir().join("cv_videoio_seq");
        std::fs::create_dir_all(&dir).unwrap();
        let pattern = dir.join("f_%02d.png").to_string_lossy().to_string();
        let mut wr = VideoWriter::open(&pattern, 10.0, 8, 6).unwrap();
        assert_eq!(wr.format(), WriterFormat::ImageSequence);
        for _ in 0..3 {
            wr.write(&Mat::zeros(6, 8, MatType::new(Depth::U8, 3)).unwrap()).unwrap();
        }
        wr.release().unwrap();
        assert!(dir.join("f_00.png").exists());
        assert!(dir.join("f_02.png").exists());
        // And it reads back as a sequence.
        let mut c = VideoCapture::open(&pattern).unwrap();
        assert_eq!(c.frame_count(), 3);
        assert_eq!(c.read_all().unwrap().len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
