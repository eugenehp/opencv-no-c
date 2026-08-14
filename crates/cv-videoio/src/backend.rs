//! Capture backends and the remaining reachable `CAP_PROP_*` surface
//! (`modules/videoio/include/opencv2/videoio.hpp`, `src/backend_plugin.cpp`).
//!
//! # Most of `videoio`'s constants are out of reach
//!
//! Upstream declares roughly 360 `CAP_PROP_*` codes for XIMEA, GigE Vision,
//! OpenNI, PvAPI, Aravis and IntelPerc cameras — properties of hardware SDKs
//! this port has no binding to. They are deliberately absent: declaring
//! `CAP_PROP_XI_TS_RST_MODE` would promise a XIMEA camera behind it.
//!
//! What is here is the surface a *file* reader can honestly answer for, plus
//! the backend enumeration, which reports what this port can actually open.

use cv_core::{cv_assert, Result};

/// `cv::VideoCaptureAPIs` — which implementation opens a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaptureApi {
    /// `cv::CAP_ANY` — let the library choose.
    #[default]
    Any,
    /// `cv::CAP_IMAGES` — a `printf`-style numbered image sequence.
    Images,
    /// `cv::CAP_OPENCV_MJPEG` — OpenCV's own MJPEG-in-AVI reader. The closest
    /// name for what this port implements.
    OpencvMjpeg,
    /// `cv::CAP_FFMPEG`
    Ffmpeg,
    /// `cv::CAP_GSTREAMER`
    Gstreamer,
    /// `cv::CAP_V4L` / `cv::CAP_V4L2`
    V4l,
    /// `cv::CAP_DSHOW`
    DShow,
    /// `cv::CAP_MSMF`
    Msmf,
    /// `cv::CAP_AVFOUNDATION`
    AvFoundation,
    /// `cv::CAP_INTEL_MFX`
    IntelMfx,
    /// `cv::CAP_FIREWIRE` / `cv::CAP_IEEE1394` / `cv::CAP_CMU1394` /
    /// `cv::CAP_FIREWARE`
    Firewire,
}

impl CaptureApi {
    /// `cv::videoio_registry::hasBackend(api)`.
    ///
    /// True only for what this port can actually open. A caller who asks for
    /// FFMPEG and silently gets the built-in reader would draw wrong
    /// conclusions about which formats will work.
    #[must_use]
    pub fn has_backend(self) -> bool {
        matches!(self, CaptureApi::Any | CaptureApi::Images | CaptureApi::OpencvMjpeg)
    }

    /// `cv::videoio_registry::isBackendBuiltIn(api)`.
    ///
    /// Everything available here is compiled in; there is no plugin loader.
    #[must_use]
    pub fn is_backend_built_in(self) -> bool {
        self.has_backend()
    }

    /// `cv::videoio_registry::getBackendName(api)`.
    #[must_use]
    pub fn backend_name(self) -> &'static str {
        match self {
            CaptureApi::Any => "ANY",
            CaptureApi::Images => "CV_IMAGES",
            CaptureApi::OpencvMjpeg => "CV_MJPEG",
            CaptureApi::Ffmpeg => "FFMPEG",
            CaptureApi::Gstreamer => "GSTREAMER",
            CaptureApi::V4l => "V4L2",
            CaptureApi::DShow => "DSHOW",
            CaptureApi::Msmf => "MSMF",
            CaptureApi::AvFoundation => "AVFOUNDATION",
            CaptureApi::IntelMfx => "INTEL_MFX",
            CaptureApi::Firewire => "FIREWIRE",
        }
    }
}

/// `cv::videoio_registry::getBackends()`.
#[must_use]
pub fn get_backends() -> Vec<CaptureApi> {
    vec![CaptureApi::Any, CaptureApi::Images, CaptureApi::OpencvMjpeg]
}

/// `cv::videoio_registry::getStreamBackends()`.
#[must_use]
pub fn get_stream_backends() -> Vec<CaptureApi> {
    get_backends()
}

/// `cv::videoio_registry::getWriterBackends()`.
#[must_use]
pub fn get_writer_backends() -> Vec<CaptureApi> {
    vec![CaptureApi::Any, CaptureApi::Images, CaptureApi::OpencvMjpeg]
}

/// `cv::videoio_registry::getCameraBackends()`.
///
/// Empty: this port reads files and image sequences, not devices.
#[must_use]
pub fn get_camera_backends() -> Vec<CaptureApi> {
    Vec::new()
}

/// `cv::videoio_registry::getBackendPluginVersion(api, version_ABI, version_API)`.
///
/// Every backend here is built in, so there is no plugin version to report.
pub fn get_backend_plugin_version(api: CaptureApi) -> Result<(i32, i32)> {
    cv_assert!(
        !api.is_backend_built_in(),
        "{} is built in, not a plugin",
        api.backend_name()
    );
    Err(cv_core::Error::new(
        cv_core::Code::StsObjectNotFound,
        format!("{} is not available in this port", api.backend_name()),
    ))
}

/// The remaining `CAP_PROP_*` codes a file reader can answer for.
///
/// Separate from [`crate::CapProp`], which is the subset already wired into
/// [`crate::VideoCapture`]; these are declared so a caller porting OpenCV code
/// can name them, and [`CapPropExt::is_supported`] says which do anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapPropExt {
    /// `cv::CAP_PROP_POS_MSEC` — timestamp of the next frame.
    PosMsec,
    /// `cv::CAP_PROP_POS_AVI_RATIO` — position as a fraction of the whole.
    PosAviRatio,
    /// `cv::CAP_PROP_FORMAT` — the `Mat` type `retrieve` returns.
    Format,
    /// `cv::CAP_PROP_MODE` — backend-specific capture mode.
    Mode,
    /// `cv::CAP_PROP_CONVERT_RGB` — convert to BGR on read.
    ConvertRgb,
    /// `cv::CAP_PROP_CHANNEL` — which stream of a multi-stream file.
    Channel,
    /// `cv::CAP_PROP_BUFFERSIZE`
    BufferSize,
    /// `cv::CAP_PROP_ORIENTATION_META` — rotation recorded in the container.
    OrientationMeta,
    /// `cv::CAP_PROP_ORIENTATION_AUTO` — apply that rotation automatically.
    OrientationAuto,
    /// `cv::CAP_PROP_N_THREADS`
    NThreads,
    /// `cv::CAP_PROP_BITRATE`
    Bitrate,
    /// `cv::CAP_PROP_SAR_NUM` — sample aspect ratio numerator.
    SarNum,
    /// `cv::CAP_PROP_SAR_DEN`
    SarDen,
    /// `cv::CAP_PROP_VIDEO_TOTAL_CHANNELS`
    VideoTotalChannels,
    /// `cv::CAP_PROP_VIDEO_STREAM`
    VideoStream,
    /// `cv::CAP_PROP_AUDIO_STREAM`
    AudioStream,
    /// `cv::CAP_PROP_BRIGHTNESS` — camera control; no meaning for a file.
    Brightness,
    /// `cv::CAP_PROP_CONTRAST`
    Contrast,
    /// `cv::CAP_PROP_SATURATION`
    Saturation,
    /// `cv::CAP_PROP_HUE`
    Hue,
    /// `cv::CAP_PROP_GAIN`
    Gain,
    /// `cv::CAP_PROP_EXPOSURE`
    Exposure,
    /// `cv::CAP_PROP_AUTO_EXPOSURE`
    AutoExposure,
    /// `cv::CAP_PROP_AUTOFOCUS`
    Autofocus,
    /// `cv::CAP_PROP_MONOCHROME`
    Monochrome,
    /// `cv::CAP_PROP_RECTIFICATION`
    Rectification,
    /// `cv::CAP_PROP_CODEC_PIXEL_FORMAT`
    CodecPixelFormat,
    /// `cv::CAP_PROP_HW_ACCELERATION`
    HwAcceleration,
    /// `cv::CAP_PROP_HW_DEVICE`
    HwDevice,
}

impl CapPropExt {
    /// Whether a file-backed capture in this port can answer this property.
    ///
    /// The camera controls — brightness, exposure, focus — are declared so
    /// ported code compiles, and report `false` here because a file has no
    /// such setting. Returning a plausible number instead would be worse than
    /// saying so.
    #[must_use]
    pub fn is_supported(self) -> bool {
        matches!(
            self,
            CapPropExt::PosMsec
                | CapPropExt::PosAviRatio
                | CapPropExt::Format
                | CapPropExt::ConvertRgb
                | CapPropExt::VideoTotalChannels
        )
    }

    /// Whether this names a camera control rather than a stream property.
    #[must_use]
    pub fn is_camera_control(self) -> bool {
        matches!(
            self,
            CapPropExt::Brightness
                | CapPropExt::Contrast
                | CapPropExt::Saturation
                | CapPropExt::Hue
                | CapPropExt::Gain
                | CapPropExt::Exposure
                | CapPropExt::AutoExposure
                | CapPropExt::Autofocus
                | CapPropExt::Monochrome
                | CapPropExt::Rectification
        )
    }
}

/// `cv::VideoWriter::fourcc(c1, c2, c3, c4)` for the codecs this port writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriterCodec {
    /// Motion JPEG in AVI — what [`crate::VideoWriter`] produces.
    Mjpg,
    /// Uncompressed BGR in AVI.
    Raw,
}

impl WriterCodec {
    /// The four-character code as a `u32`, in `fourcc` order.
    #[must_use]
    pub fn fourcc(self) -> u32 {
        match self {
            WriterCodec::Mjpg => crate::fourcc('M', 'J', 'P', 'G'),
            WriterCodec::Raw => crate::fourcc('D', 'I', 'B', ' '),
        }
    }
}

/// `cv::VideoCapture::setExceptionMode(enable)` / `getExceptionMode()`.
///
/// This port returns `Result` throughout, so a failed read is already visible
/// to the caller. The flag is tracked for API compatibility and does not change
/// behaviour — there is no silent-failure mode to switch out of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExceptionMode(pub bool);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_backends_this_port_has_report_available() {
        for a in [CaptureApi::Any, CaptureApi::Images, CaptureApi::OpencvMjpeg] {
            assert!(a.has_backend(), "{} should be available", a.backend_name());
            assert!(a.is_backend_built_in());
        }
        for a in [
            CaptureApi::Ffmpeg,
            CaptureApi::Gstreamer,
            CaptureApi::V4l,
            CaptureApi::DShow,
            CaptureApi::Msmf,
            CaptureApi::AvFoundation,
            CaptureApi::IntelMfx,
            CaptureApi::Firewire,
        ] {
            assert!(!a.has_backend(), "{} must not claim availability", a.backend_name());
        }
        assert_eq!(CaptureApi::default(), CaptureApi::Any);
    }

    #[test]
    fn the_backend_lists_agree_with_has_backend() {
        for a in get_backends() {
            assert!(a.has_backend(), "{} listed but unavailable", a.backend_name());
        }
        for a in get_writer_backends() {
            assert!(a.has_backend());
        }
        assert_eq!(get_stream_backends(), get_backends());
        // No device capture in a file-only port.
        assert!(get_camera_backends().is_empty());
    }

    #[test]
    fn the_backend_names_match_upstreams() {
        assert_eq!(CaptureApi::Any.backend_name(), "ANY");
        assert_eq!(CaptureApi::Ffmpeg.backend_name(), "FFMPEG");
        assert_eq!(CaptureApi::V4l.backend_name(), "V4L2");
        assert_eq!(CaptureApi::OpencvMjpeg.backend_name(), "CV_MJPEG");
        assert_eq!(CaptureApi::Images.backend_name(), "CV_IMAGES");
    }

    #[test]
    fn plugin_versions_are_refused_for_built_in_and_absent_backends() {
        // Built in: there is no plugin to version.
        let e = get_backend_plugin_version(CaptureApi::OpencvMjpeg).unwrap_err();
        assert!(format!("{e}").contains("built in"), "{e}");
        // Absent: nothing to report either.
        let e2 = get_backend_plugin_version(CaptureApi::Ffmpeg).unwrap_err();
        assert!(format!("{e2}").contains("not available"), "{e2}");
    }

    #[test]
    fn camera_controls_are_declared_but_report_unsupported() {
        // A file has no exposure. Saying so beats returning a plausible number.
        for p in [
            CapPropExt::Brightness,
            CapPropExt::Exposure,
            CapPropExt::Autofocus,
            CapPropExt::Gain,
            CapPropExt::Hue,
        ] {
            assert!(p.is_camera_control(), "{p:?} is a camera control");
            assert!(!p.is_supported(), "{p:?} cannot be answered for a file");
        }
        // Stream properties are not camera controls.
        for p in [CapPropExt::PosMsec, CapPropExt::PosAviRatio, CapPropExt::Format] {
            assert!(!p.is_camera_control());
            assert!(p.is_supported(), "{p:?} should be answerable");
        }
        // And some stream properties are still out of reach.
        assert!(!CapPropExt::Bitrate.is_supported());
        assert!(!CapPropExt::HwAcceleration.is_supported());
    }

    #[test]
    fn the_writer_codecs_produce_the_right_fourcc() {
        // 'MJPG' little-endian packed.
        assert_eq!(WriterCodec::Mjpg.fourcc(), crate::fourcc('M', 'J', 'P', 'G'));
        assert_eq!(WriterCodec::Raw.fourcc(), crate::fourcc('D', 'I', 'B', ' '));
        assert_ne!(WriterCodec::Mjpg.fourcc(), WriterCodec::Raw.fourcc());
    }

    #[test]
    fn exception_mode_round_trips() {
        assert!(!ExceptionMode::default().0);
        assert!(ExceptionMode(true).0);
    }
}

/// `cv::IStreamReader` — a source of encoded bytes that is not a file.
///
/// A `VideoCapture` normally opens a path. This is the interface for handing
/// it a stream instead — a network socket, an in-memory buffer, an archive
/// entry — so that decoding does not require the data to exist on disk first.
///
/// [`std::io::Read`] plus [`std::io::Seek`] is the same contract, so a caller
/// with one of those gets an implementation for free via the blanket impl
/// below.
pub trait IStreamReader {
    /// `cv::IStreamReader::read(buffer, size)` — bytes actually read, 0 at end.
    fn read(&mut self, buffer: &mut [u8]) -> cv_core::Result<usize>;

    /// `cv::IStreamReader::seek(offset, origin)`.
    ///
    /// `origin` is `SEEK_SET` (0), `SEEK_CUR` (1) or `SEEK_END` (2), matching
    /// C's `fseek`, which is what upstream's callers pass.
    fn seek(&mut self, offset: i64, origin: i32) -> cv_core::Result<i64>;
}

impl<T: std::io::Read + std::io::Seek> IStreamReader for T {
    fn read(&mut self, buffer: &mut [u8]) -> cv_core::Result<usize> {
        std::io::Read::read(self, buffer).map_err(|e| {
            cv_core::Error::new(cv_core::Code::StsError, format!("stream read failed: {e}"))
        })
    }

    fn seek(&mut self, offset: i64, origin: i32) -> cv_core::Result<i64> {
        let from = match origin {
            0 => std::io::SeekFrom::Start(offset.max(0) as u64),
            1 => std::io::SeekFrom::Current(offset),
            2 => std::io::SeekFrom::End(offset),
            other => {
                return Err(cv_core::Error::new(
                    cv_core::Code::StsBadArg,
                    format!("seek origin must be 0, 1 or 2, got {other}"),
                ))
            }
        };
        std::io::Seek::seek(self, from).map(|p| p as i64).map_err(|e| {
            cv_core::Error::new(cv_core::Code::StsError, format!("stream seek failed: {e}"))
        })
    }
}

/// `cv::videoio_registry::getStreamBufferedBackends()`.
///
/// The backends that can read from an [`IStreamReader`] rather than a path.
/// Empty: the two container readers here take a byte slice directly, which
/// covers the same need without a buffering layer.
#[must_use]
pub fn get_stream_buffered_backends() -> Vec<CaptureApi> {
    Vec::new()
}

/// The version a plugin reports, as `(ABI, API)`.
///
/// Both zero for every query below: a plugin version identifies a shared
/// library loaded at runtime, and this crate loads none — its backends are
/// compiled in. Zero is what upstream reports for a built-in backend too.
type PluginVersion = (i32, i32);

/// `cv::videoio_registry::getCameraBackendPluginVersion(api, version_ABI, version_API)`.
#[must_use]
pub fn get_camera_backend_plugin_version(_api: CaptureApi) -> PluginVersion {
    (0, 0)
}

/// `cv::videoio_registry::getStreamBackendPluginVersion(api, version_ABI, version_API)`.
#[must_use]
pub fn get_stream_backend_plugin_version(_api: CaptureApi) -> PluginVersion {
    (0, 0)
}

/// `cv::videoio_registry::getStreamBufferedBackendPluginVersion(api, version_ABI, version_API)`.
#[must_use]
pub fn get_stream_buffered_backend_plugin_version(_api: CaptureApi) -> PluginVersion {
    (0, 0)
}

/// `cv::videoio_registry::getWriterBackendPluginVersion(api, version_ABI, version_API)`.
#[must_use]
pub fn get_writer_backend_plugin_version(_api: CaptureApi) -> PluginVersion {
    (0, 0)
}

/// Constants of the WinRT and AVI container backends.
///
/// Neither backend is reachable here — the WinRT one needs a Windows Runtime
/// this crate does not link, and the AVI colour-space codes describe the
/// private container reader's internal formats. The names are declared because
/// they appear in code translated from C++.
pub mod backend_constants {
    /// `OPEN_CAMERA` — the WinRT message that opens the capture device.
    pub const OPEN_CAMERA: i32 = 300;
    /// `CLOSE_CAMERA` — and the one that closes it.
    pub const CLOSE_CAMERA: i32 = 301;
    /// `UPDATE_IMAGE_ELEMENT` — refresh the displayed frame.
    pub const UPDATE_IMAGE_ELEMENT: i32 = 302;
    /// `SHOW_TRACKBAR` — reveal the seek control.
    pub const SHOW_TRACKBAR: i32 = 303;

    /// `COLORSPACE_GRAY` — single-channel luma.
    pub const COLORSPACE_GRAY: i32 = 0;
    /// `COLORSPACE_RGBA` — four-channel, alpha last.
    pub const COLORSPACE_RGBA: i32 = 1;
    /// `COLORSPACE_BGR` — three-channel, blue first.
    pub const COLORSPACE_BGR: i32 = 2;
    /// `COLORSPACE_YUV444P` — planar YUV with no chroma subsampling.
    pub const COLORSPACE_YUV444P: i32 = 3;
}

#[cfg(test)]
mod stream_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn a_cursor_satisfies_the_stream_reader_contract() {
        let mut c = Cursor::new(b"0123456789".to_vec());
        let mut buf = [0u8; 4];
        assert_eq!(IStreamReader::read(&mut c, &mut buf).unwrap(), 4);
        assert_eq!(&buf, b"0123");

        // SEEK_SET, SEEK_CUR and SEEK_END, as C numbers them.
        assert_eq!(IStreamReader::seek(&mut c, 8, 0).unwrap(), 8);
        assert_eq!(IStreamReader::read(&mut c, &mut buf).unwrap(), 2, "two bytes left");
        assert_eq!(IStreamReader::seek(&mut c, -3, 2).unwrap(), 7);
        assert_eq!(IStreamReader::seek(&mut c, 1, 1).unwrap(), 8);
        // At the end, a read returns zero rather than failing.
        IStreamReader::seek(&mut c, 0, 2).unwrap();
        assert_eq!(IStreamReader::read(&mut c, &mut buf).unwrap(), 0);
        // An unknown origin is rejected.
        assert!(IStreamReader::seek(&mut c, 0, 9).is_err());
    }

    #[test]
    fn the_plugin_versions_report_built_in() {
        assert_eq!(get_camera_backend_plugin_version(CaptureApi::Any), (0, 0));
        assert_eq!(get_stream_backend_plugin_version(CaptureApi::Any), (0, 0));
        assert_eq!(get_stream_buffered_backend_plugin_version(CaptureApi::Any), (0, 0));
        assert_eq!(get_writer_backend_plugin_version(CaptureApi::Any), (0, 0));
        assert!(get_stream_buffered_backends().is_empty());
    }

    #[test]
    fn the_backend_constants_match_upstream() {
        use backend_constants::*;
        assert_eq!(OPEN_CAMERA, 300);
        assert_eq!(SHOW_TRACKBAR, 303);
        assert_eq!(COLORSPACE_GRAY, 0);
        assert_eq!(COLORSPACE_BGR, 2);
        assert_eq!(COLORSPACE_YUV444P, 3);
    }
}
