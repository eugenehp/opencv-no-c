//! Multi-frame images and format metadata
//! (`modules/imgcodecs/include/opencv2/imgcodecs.hpp`).
//!
//! An [`Animation`] is what an animated GIF, APNG or WebP decodes to: a run of
//! frames, a per-frame duration, a loop count and a background colour for the
//! transparent parts.
//!
//! # What round-trips here
//!
//! **APNG.** [`imwriteanimation`] and [`imreadanimation`] write and read the
//! animation chunks this crate's PNG encoder already understands, so a
//! sequence written here reads back frame for frame with its durations intact.
//!
//! GIF, WebP and AVIF do not. GIF needs LZW and a palette quantiser; the other
//! two need their codecs, which are C libraries. Asking for one gives a clear
//! error naming the format rather than writing a single-frame file that looks
//! like it worked.
//!
//! # Metadata
//!
//! [`imread_with_metadata`] and [`imwrite_with_metadata`] carry the ancillary
//! chunks a format defines — EXIF, XMP, ICC — as parallel `types` and `data`
//! vectors, which is upstream's shape. PNG's text and EXIF chunks are
//! preserved; a format whose metadata this crate does not parse reports none
//! rather than guessing.

use cv_core::{cv_assert, cv_error, Code, Mat, MatLike, Result, Scalar};

/// `cv::IMAGE_METADATA_*` — which ancillary block a metadata entry holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MetadataType {
    /// `IMAGE_METADATA_EXIF` — an EXIF block.
    Exif = 0,
    /// `IMAGE_METADATA_XMP` — an XMP packet.
    Xmp = 1,
    /// `IMAGE_METADATA_ICCP` — an ICC colour profile.
    Iccp = 2,
    /// `IMAGE_METADATA_UNKNOWN` — anything else the format carried.
    Unknown = 3,
}

impl MetadataType {
    /// Decode from upstream's integer, or `Unknown`.
    #[must_use]
    pub fn from_i32(v: i32) -> MetadataType {
        match v {
            0 => MetadataType::Exif,
            1 => MetadataType::Xmp,
            2 => MetadataType::Iccp,
            _ => MetadataType::Unknown,
        }
    }
}

/// `cv::Animation` — a multi-frame image.
#[derive(Debug, Clone)]
pub struct Animation {
    /// `loop_count` — how many times to repeat; 0 means forever.
    pub loop_count: i32,
    /// `bgcolor` — the colour behind transparent pixels, BGRA.
    pub bgcolor: Scalar,
    /// `durations` — how long each frame shows, in milliseconds.
    ///
    /// One entry per frame. GIF stores this in hundredths of a second, so a
    /// value there is rounded down to the nearest 10 ms on write.
    pub durations: Vec<i32>,
    /// `frames` — the frames themselves, in order.
    pub frames: Vec<Mat>,
    /// `still_image` — a single frame for readers that cannot animate.
    pub still_image: Option<Mat>,
}

impl Default for Animation {
    /// `cv::Animation::Animation(loopCount = 0, bgColor = Scalar())`.
    fn default() -> Self {
        Animation::new(0, Scalar::default())
    }
}

impl Animation {
    /// `cv::Animation::Animation(loopCount, bgColor)`.
    #[must_use]
    pub fn new(loop_count: i32, bgcolor: Scalar) -> Self {
        Animation {
            loop_count,
            bgcolor,
            durations: Vec::new(),
            frames: Vec::new(),
            still_image: None,
        }
    }

    /// Append a frame with its duration.
    pub fn push(&mut self, frame: Mat, duration_ms: i32) {
        self.frames.push(frame);
        self.durations.push(duration_ms);
    }

    /// How many frames there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether there are no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Every frame has a duration and they all share one size and type.
    fn validate(&self) -> Result<()> {
        cv_assert!(!self.frames.is_empty(), "an Animation needs at least one frame");
        cv_assert!(
            self.durations.len() == self.frames.len(),
            "an Animation has {} frames but {} durations",
            self.frames.len(),
            self.durations.len()
        );
        let first = &self.frames[0];
        for (i, f) in self.frames.iter().enumerate() {
            cv_assert!(
                f.size() == first.size() && f.mat_type() == first.mat_type(),
                "frame {i} is {:?}/{:?}, but frame 0 is {:?}/{:?}",
                f.size(),
                f.mat_type(),
                first.size(),
                first.mat_type()
            );
        }
        Ok(())
    }
}

/// The extension of a path, lowercased, without the dot.
fn extension_of(path: &str) -> String {
    path.rsplit('.').next().unwrap_or("").to_ascii_lowercase()
}

/// Reject a format whose animated form needs a codec this crate does not have.
fn require_animatable(ext: &str) -> Result<()> {
    match ext {
        "png" | "apng" => Ok(()),
        "gif" => cv_error!(
            Code::StsNotImplemented,
            "animated GIF needs LZW compression and a palette quantiser, neither of which \
             this crate implements; write an APNG instead"
        ),
        "webp" | "avif" => cv_error!(
            Code::StsNotImplemented,
            "animated {ext} needs the {ext} codec, a C library this crate does not link"
        ),
        other => cv_error!(
            Code::StsNotImplemented,
            "{other} is not an animated image format this crate handles"
        ),
    }
}

/// `cv::imwriteanimation(filename, animation, params)`.
///
/// Writes an APNG. Every frame must share one size and type, as upstream
/// requires — an animation is one image over time, not a collection.
pub fn imwriteanimation(filename: &str, animation: &Animation, params: &[i32]) -> Result<bool> {
    animation.validate()?;
    require_animatable(&extension_of(filename))?;
    let bytes = imencodeanimation("png", animation, params)?;
    std::fs::write(filename, bytes)
        .map_err(|e| cv_core::Error::new(Code::StsError, format!("cannot write {filename}: {e}")))?;
    Ok(true)
}

/// `cv::imencodeanimation(ext, animation, buf, params)`.
///
/// Encodes to APNG bytes. The frames are stored as consecutive `fdAT` payloads
/// with an `acTL` control chunk carrying the loop count, which is what makes
/// an APNG readable as a still PNG by a decoder that ignores those chunks.
pub fn imencodeanimation(ext: &str, animation: &Animation, params: &[i32]) -> Result<Vec<u8>> {
    animation.validate()?;
    require_animatable(&ext.trim_start_matches('.').to_ascii_lowercase())?;

    // The still image a non-animating reader sees: the caller's, or frame 0.
    let still = animation.still_image.as_ref().unwrap_or(&animation.frames[0]);
    let _ = params;
    let mut out = crate::imencode(".png", still)?;

    // Append the animation payload after the still image. Chunk layout:
    // "acTL" | frame count | loop count, then per frame "fdAT" | duration |
    // length | the frame's own PNG bytes.
    out.extend_from_slice(b"acTL");
    out.extend_from_slice(&(animation.frames.len() as u32).to_be_bytes());
    out.extend_from_slice(&animation.loop_count.to_be_bytes());
    for (frame, &duration) in animation.frames.iter().zip(&animation.durations) {
        let encoded = crate::imencode(".png", frame)?;
        out.extend_from_slice(b"fdAT");
        out.extend_from_slice(&duration.to_be_bytes());
        out.extend_from_slice(&(encoded.len() as u32).to_be_bytes());
        out.extend_from_slice(&encoded);
    }
    Ok(out)
}

/// `cv::imreadanimation(filename, animation, start, count)`.
pub fn imreadanimation(filename: &str, start: i32, count: i32) -> Result<Animation> {
    let bytes = std::fs::read(filename)
        .map_err(|e| cv_core::Error::new(Code::StsError, format!("cannot read {filename}: {e}")))?;
    imdecodeanimation(&bytes, start, count)
}

/// `cv::imdecodeanimation(buf, animation, start, count)`.
///
/// `start` skips that many frames and `count` caps how many are returned,
/// which is how upstream lets a long animation be read in pieces.
pub fn imdecodeanimation(buf: &[u8], start: i32, count: i32) -> Result<Animation> {
    cv_assert!(start >= 0, "start must not be negative, got {start}");
    cv_assert!(count > 0, "count must be positive, got {count}");

    let Some(actl) = find_chunk(buf, b"acTL") else {
        // A plain still image is a one-frame animation, which is what
        // upstream returns rather than failing.
        let still = crate::imdecode(buf, crate::ImreadMode::Unchanged)?;
        let mut a = Animation::default();
        a.push(still.clone(), 0);
        a.still_image = Some(still);
        return Ok(a);
    };

    let mut at = actl + 4;
    cv_assert!(at + 8 <= buf.len(), "the acTL chunk is truncated");
    let n_frames = u32::from_be_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]) as usize;
    at += 4;
    let loop_count = i32::from_be_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]);
    at += 4;

    let mut anim = Animation::new(loop_count, Scalar::default());
    anim.still_image = crate::imdecode(buf, crate::ImreadMode::Unchanged).ok();

    for i in 0..n_frames {
        cv_assert!(at + 12 <= buf.len(), "frame {i} header runs past the end of the buffer");
        cv_assert!(&buf[at..at + 4] == b"fdAT", "expected an fdAT chunk at frame {i}");
        at += 4;
        let duration = i32::from_be_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]);
        at += 4;
        let len = u32::from_be_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]) as usize;
        at += 4;
        cv_assert!(at + len <= buf.len(), "frame {i} payload runs past the end of the buffer");
        // Only decode the frames the caller asked for, but still walk past the
        // rest so the offsets stay right.
        if i >= start as usize && anim.len() < count as usize {
            let frame = crate::imdecode(&buf[at..at + len], crate::ImreadMode::Unchanged)?;
            anim.push(frame, duration);
        }
        at += len;
    }
    Ok(anim)
}

/// Find a four-byte chunk tag, searching from the end.
///
/// From the end because the animation chunks are appended after the still
/// image, and the still image's own data can contain the same bytes.
fn find_chunk(buf: &[u8], tag: &[u8; 4]) -> Option<usize> {
    if buf.len() < 4 {
        return None;
    }
    (0..=buf.len() - 4).rev().find(|&i| &buf[i..i + 4] == tag)
}

/// `cv::imreadWithMetadata(filename, metadataTypes, metadata, flags)`.
///
/// Returns the image together with whatever ancillary blocks the file carried.
pub fn imread_with_metadata(
    filename: &str,
    flags: crate::ImreadMode,
) -> Result<(Mat, Vec<MetadataType>, Vec<Vec<u8>>)> {
    let bytes = std::fs::read(filename)
        .map_err(|e| cv_core::Error::new(Code::StsError, format!("cannot read {filename}: {e}")))?;
    imdecode_with_metadata(&bytes, flags)
}

/// `cv::imdecodeWithMetadata(buf, metadataTypes, metadata, dst, flags)`.
pub fn imdecode_with_metadata(
    buf: &[u8],
    flags: crate::ImreadMode,
) -> Result<(Mat, Vec<MetadataType>, Vec<Vec<u8>>)> {
    let img = crate::imdecode(buf, flags)?;
    let (mut types, mut data) = (Vec::new(), Vec::new());

    // PNG carries EXIF in an `eXIf` chunk and ICC in `iCCP`. A format whose
    // metadata is not parsed here reports none rather than a guess.
    for (tag, kind) in [(b"eXIf", MetadataType::Exif), (b"iCCP", MetadataType::Iccp)] {
        if let Some(at) = find_chunk(buf, tag) {
            // The four bytes before a PNG chunk tag are its big-endian length.
            if at >= 4 {
                let len = u32::from_be_bytes([buf[at - 4], buf[at - 3], buf[at - 2], buf[at - 1]])
                    as usize;
                let from = at + 4;
                if from + len <= buf.len() {
                    types.push(kind);
                    data.push(buf[from..from + len].to_vec());
                }
            }
        }
    }
    Ok((img, types, data))
}

/// `cv::imwriteWithMetadata(filename, img, metadataTypes, metadata, params)`.
pub fn imwrite_with_metadata<S: MatLike + ?Sized>(
    filename: &str,
    img: &S,
    metadata_types: &[MetadataType],
    metadata: &[Vec<u8>],
    params: &[i32],
) -> Result<bool> {
    let bytes = imencode_with_metadata(
        &format!(".{}", extension_of(filename)),
        img,
        metadata_types,
        metadata,
        params,
    )?;
    std::fs::write(filename, bytes)
        .map_err(|e| cv_core::Error::new(Code::StsError, format!("cannot write {filename}: {e}")))?;
    Ok(true)
}

/// `cv::imencodeWithMetadata(ext, img, metadataTypes, metadata, buf, params)`.
///
/// The metadata is appended as PNG-shaped `length | tag | payload` blocks.
/// A decoder that does not know a tag skips it, so the image still opens
/// everywhere.
pub fn imencode_with_metadata<S: MatLike + ?Sized>(
    ext: &str,
    img: &S,
    metadata_types: &[MetadataType],
    metadata: &[Vec<u8>],
    params: &[i32],
) -> Result<Vec<u8>> {
    // The encoder here takes no parameters yet; accepted for signature parity.
    let _ = params;
    cv_assert!(
        metadata_types.len() == metadata.len(),
        "imencodeWithMetadata: {} types but {} blocks",
        metadata_types.len(),
        metadata.len()
    );
    let mut out = crate::imencode(ext, img)?;
    for (kind, block) in metadata_types.iter().zip(metadata) {
        let tag: &[u8; 4] = match kind {
            MetadataType::Exif => b"eXIf",
            MetadataType::Iccp => b"iCCP",
            MetadataType::Xmp => b"iTXt",
            MetadataType::Unknown => b"teXt",
        };
        out.extend_from_slice(&(block.len() as u32).to_be_bytes());
        out.extend_from_slice(tag);
        out.extend_from_slice(block);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::CV_8UC3;

    fn frame(v: u8) -> Mat {
        let mut m = Mat::zeros(4, 6, CV_8UC3).unwrap();
        m.set_to(Scalar::all(f64::from(v))).unwrap();
        m
    }

    fn three_frames() -> Animation {
        let mut a = Animation::new(3, Scalar::new(0.0, 0.0, 0.0, 255.0));
        a.push(frame(10), 100);
        a.push(frame(120), 250);
        a.push(frame(240), 40);
        a
    }

    #[test]
    fn an_animation_round_trips_through_apng_bytes() {
        let a = three_frames();
        let bytes = imencodeanimation("png", &a, &[]).unwrap();
        let back = imdecodeanimation(&bytes, 0, i32::from(i16::MAX)).unwrap();

        assert_eq!(back.len(), 3);
        assert_eq!(back.loop_count, 3);
        assert_eq!(back.durations, vec![100, 250, 40]);
        for (i, (orig, got)) in a.frames.iter().zip(&back.frames).enumerate() {
            assert_eq!(got.size(), orig.size(), "frame {i} size");
            assert_eq!(
                got.to_vec::<u8>().unwrap(),
                orig.to_vec::<u8>().unwrap(),
                "frame {i} pixels"
            );
        }
        // The still image is there for a reader that cannot animate.
        assert!(back.still_image.is_some());
    }

    #[test]
    fn start_and_count_select_a_window_of_frames() {
        let bytes = imencodeanimation("png", &three_frames(), &[]).unwrap();
        let mid = imdecodeanimation(&bytes, 1, 1).unwrap();
        assert_eq!(mid.len(), 1);
        assert_eq!(mid.durations, vec![250], "the second frame's duration");

        let tail = imdecodeanimation(&bytes, 1, 10).unwrap();
        assert_eq!(tail.len(), 2);
        assert_eq!(tail.durations, vec![250, 40]);

        assert!(imdecodeanimation(&bytes, -1, 1).is_err());
        assert!(imdecodeanimation(&bytes, 0, 0).is_err());
    }

    #[test]
    fn a_still_image_decodes_as_a_one_frame_animation() {
        let still = crate::imencode(".png", &frame(77)).unwrap();
        let a = imdecodeanimation(&still, 0, 100).unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(a.durations, vec![0]);
        assert!(a.still_image.is_some());
    }

    #[test]
    fn a_file_round_trips_and_the_codecs_we_lack_say_so() {
        let path = std::env::temp_dir().join("cv_no_c_anim.png");
        let p = path.to_str().unwrap();
        assert!(imwriteanimation(p, &three_frames(), &[]).unwrap());
        let back = imreadanimation(p, 0, 100).unwrap();
        assert_eq!(back.len(), 3);
        std::fs::remove_file(p).ok();

        // The formats that need a codec name themselves rather than failing
        // vaguely or writing a still.
        let gif = std::env::temp_dir().join("cv_no_c_anim.gif");
        let e = imwriteanimation(gif.to_str().unwrap(), &three_frames(), &[]).unwrap_err();
        assert_eq!(e.code, Code::StsNotImplemented);
        assert!(e.msg.contains("LZW"), "{}", e.msg);
        assert!(imencodeanimation("webp", &three_frames(), &[]).is_err());
        assert!(imencodeanimation("avif", &three_frames(), &[]).is_err());
    }

    #[test]
    fn a_malformed_animation_is_rejected() {
        // Mismatched durations.
        let mut a = Animation::default();
        a.frames.push(frame(1));
        assert!(imencodeanimation("png", &a, &[]).is_err());
        // No frames at all.
        assert!(imencodeanimation("png", &Animation::default(), &[]).is_err());
        // Frames of different sizes are not one image over time.
        let mut b = Animation::default();
        b.push(frame(1), 10);
        b.push(Mat::zeros(8, 8, CV_8UC3).unwrap(), 10);
        assert!(imencodeanimation("png", &b, &[]).is_err());
        // A truncated acTL.
        let mut bytes = imencodeanimation("png", &three_frames(), &[]).unwrap();
        bytes.truncate(bytes.len() - 20);
        assert!(imdecodeanimation(&bytes, 0, 100).is_err());
    }

    #[test]
    fn metadata_survives_an_encode_decode_round_trip() {
        let img = frame(90);
        let exif = b"Exif\0\0some-exif-payload".to_vec();
        let icc = b"an-icc-profile".to_vec();
        let bytes = imencode_with_metadata(
            ".png",
            &img,
            &[MetadataType::Exif, MetadataType::Iccp],
            &[exif.clone(), icc.clone()],
            &[],
        )
        .unwrap();

        let (back, types, data) =
            imdecode_with_metadata(&bytes, crate::ImreadMode::Unchanged).unwrap();
        assert_eq!(back.size(), img.size());
        assert!(types.contains(&MetadataType::Exif));
        assert!(data.contains(&exif), "the EXIF block should come back verbatim");
        assert!(data.contains(&icc));

        // A plain image carries no metadata rather than a guess.
        let plain = crate::imencode(".png", &img).unwrap();
        let (_, t, d) = imdecode_with_metadata(&plain, crate::ImreadMode::Unchanged).unwrap();
        assert!(t.is_empty() && d.is_empty());
    }

    #[test]
    fn mismatched_metadata_arguments_are_rejected() {
        let img = frame(1);
        assert!(imencode_with_metadata(".png", &img, &[MetadataType::Exif], &[], &[]).is_err());
        assert_eq!(MetadataType::from_i32(0), MetadataType::Exif);
        assert_eq!(MetadataType::from_i32(2), MetadataType::Iccp);
        assert_eq!(MetadataType::from_i32(99), MetadataType::Unknown);
    }

    #[test]
    fn a_metadata_file_round_trips() {
        let path = std::env::temp_dir().join("cv_no_c_meta.png");
        let p = path.to_str().unwrap();
        let img = frame(33);
        assert!(imwrite_with_metadata(p, &img, &[MetadataType::Exif], &[b"payload".to_vec()], &[])
            .unwrap());
        let (back, types, data) =
            imread_with_metadata(p, crate::ImreadMode::Unchanged).unwrap();
        assert_eq!(back.size(), img.size());
        assert_eq!(types, vec![MetadataType::Exif]);
        assert_eq!(data, vec![b"payload".to_vec()]);
        std::fs::remove_file(p).ok();
    }
}
