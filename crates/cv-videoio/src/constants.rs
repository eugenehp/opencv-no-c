//! `cv::videoio`'s property, backend and format constants
//! (`modules/videoio/include/opencv2/videoio.hpp`).
//!
//! Four hundred-odd integers naming what a `VideoCapture` or `VideoWriter`
//! property means, which backend to open with, and what each vendor SDK calls
//! its own settings. They are generated from the upstream header so a value
//! cannot be mistranscribed, and every one is the number `cv2` reports.
//!
//! # These are names, not capabilities
//!
//! Declaring `CAP_FFMPEG` does not make an FFMPEG backend exist. This port
//! reads and writes the container formats in [`crate::avi`] and
//! [`crate::y4m`]; everything else — V4L2, GStreamer, XIMEA, OpenNI — needs a
//! C library it deliberately does not link. `crate::backend::available`
//! answers which backends actually work.
//!
//! The constants earn their place anyway: code translated from C++ or Python
//! passes them around, compares them, and stores them in config, and it should
//! not have to hard-code `1900` for FFMPEG. Passing one this port cannot serve
//! gives a clear error rather than a silent fallback.
//!
//! # Wrapping
//!
//! A few generator flags are `1 << 31` or `1 << 30` in a C++ `int` enum, so
//! they arrive negative: `CAP_OPENNI_DEPTH_GENERATOR` is `-2147483648`. That
//! is what `cv2` reports and what a bitmask comparison against a property
//! needs, so the wrapped value is used rather than a widened one.

/// `CAP_ANY` — Auto detect == 0.
pub const CAP_ANY: i32 = 0;
/// `CAP_VFW` — Video For Windows (obsolete, removed).
pub const CAP_VFW: i32 = 200;
/// `CAP_V4L` — V4L/V4L2 capturing support.
pub const CAP_V4L: i32 = 200;
/// `CAP_V4L2` — Same as CAP_V4L.
pub const CAP_V4L2: i32 = 200;
/// `CAP_FIREWIRE` — IEEE 1394 drivers.
pub const CAP_FIREWIRE: i32 = 300;
/// `CAP_FIREWARE` — Same value as CAP_FIREWIRE.
pub const CAP_FIREWARE: i32 = 300;
/// `CAP_IEEE1394` — Same value as CAP_FIREWIRE.
pub const CAP_IEEE1394: i32 = 300;
/// `CAP_DC1394` — Same value as CAP_FIREWIRE.
pub const CAP_DC1394: i32 = 300;
/// `CAP_CMU1394` — Same value as CAP_FIREWIRE.
pub const CAP_CMU1394: i32 = 300;
/// `CAP_QT` — QuickTime (obsolete, removed).
pub const CAP_QT: i32 = 500;
/// `CAP_UNICAP` — Unicap drivers (obsolete, removed).
pub const CAP_UNICAP: i32 = 600;
/// `CAP_DSHOW` — DirectShow (via videoInput).
pub const CAP_DSHOW: i32 = 700;
/// `CAP_PVAPI` — PvAPI, Prosilica GigE SDK.
pub const CAP_PVAPI: i32 = 800;
/// `CAP_OPENNI` — OpenNI (for Kinect).
pub const CAP_OPENNI: i32 = 900;
/// `CAP_OPENNI_ASUS` — OpenNI (for Asus Xtion).
pub const CAP_OPENNI_ASUS: i32 = 910;
/// `CAP_ANDROID` — MediaNDK (API Level 21+) and NDK Camera (API level 24+)
///   for Android.
pub const CAP_ANDROID: i32 = 1000;
/// `CAP_XIAPI` — XIMEA Camera API.
pub const CAP_XIAPI: i32 = 1100;
/// `CAP_AVFOUNDATION` — AVFoundation framework for iOS (OS X Lion will have
///   the same API).
pub const CAP_AVFOUNDATION: i32 = 1200;
/// `CAP_GIGANETIX` — Smartek Giganetix GigEVisionSDK.
pub const CAP_GIGANETIX: i32 = 1300;
/// `CAP_MSMF` — Microsoft Media Foundation (via videoInput). See platform
///   specific notes above.
pub const CAP_MSMF: i32 = 1400;
/// `CAP_WINRT` — Microsoft Windows Runtime using Media Foundation.
pub const CAP_WINRT: i32 = 1410;
/// `CAP_INTELPERC` — RealSense (former Intel Perceptual Computing SDK).
pub const CAP_INTELPERC: i32 = 1500;
/// `CAP_REALSENSE` — Synonym for CAP_INTELPERC.
pub const CAP_REALSENSE: i32 = 1500;
/// `CAP_OPENNI2` — OpenNI2 (for Kinect).
pub const CAP_OPENNI2: i32 = 1600;
/// `CAP_OPENNI2_ASUS` — OpenNI2 (for Asus Xtion and Occipital Structure
///   sensors).
pub const CAP_OPENNI2_ASUS: i32 = 1610;
/// `CAP_OPENNI2_ASTRA` — OpenNI2 (for Orbbec Astra).
pub const CAP_OPENNI2_ASTRA: i32 = 1620;
/// `CAP_GPHOTO2` — gPhoto2 connection.
pub const CAP_GPHOTO2: i32 = 1700;
/// `CAP_GSTREAMER` — GStreamer.
pub const CAP_GSTREAMER: i32 = 1800;
/// `CAP_FFMPEG` — Open and record video file or stream using the FFMPEG
///   library.
pub const CAP_FFMPEG: i32 = 1900;
/// `CAP_IMAGES` — OpenCV Image Sequence (e.g. img_%02d.jpg).
pub const CAP_IMAGES: i32 = 2000;
/// `CAP_ARAVIS` — Aravis SDK.
pub const CAP_ARAVIS: i32 = 2100;
/// `CAP_OPENCV_MJPEG` — Built-in OpenCV MotionJPEG codec.
pub const CAP_OPENCV_MJPEG: i32 = 2200;
/// `CAP_INTEL_MFX` — Intel MediaSDK.
pub const CAP_INTEL_MFX: i32 = 2300;
/// `CAP_XINE` — XINE engine (Linux).
pub const CAP_XINE: i32 = 2400;
/// `CAP_UEYE` — uEye Camera API.
pub const CAP_UEYE: i32 = 2500;
/// `CAP_OBSENSOR` — For Orbbec 3D-Sensor device/module (Astra+, Femto,
///   Astra2, Gemini2, Gemini2L, Gemini2XL, Gemini330, Femto Mega) attention:
///   Astra2 cameras currently only support Windows and Linux kernel versions no
///   higher than 4.15, and higher versions of Linux kernel may have exceptions.
pub const CAP_OBSENSOR: i32 = 2600;
/// `CAP_PROP_POS_MSEC` — Current position of the video file in milliseconds.
pub const CAP_PROP_POS_MSEC: i32 = 0;
/// `CAP_PROP_POS_FRAMES` — 0-based index of the frame to be decoded/captured
///   next. When the index i is set in RAW mode (CAP_PROP_FORMAT == -1) this
///   will seek to the key frame k, where k <= i.
pub const CAP_PROP_POS_FRAMES: i32 = 1;
/// `CAP_PROP_POS_AVI_RATIO` — Relative position of the video file: 0=start of
///   the film, 1=end of the film.
pub const CAP_PROP_POS_AVI_RATIO: i32 = 2;
/// `CAP_PROP_FRAME_WIDTH` — Width of the frames in the video stream.
pub const CAP_PROP_FRAME_WIDTH: i32 = 3;
/// `CAP_PROP_FRAME_HEIGHT` — Height of the frames in the video stream.
pub const CAP_PROP_FRAME_HEIGHT: i32 = 4;
/// `CAP_PROP_FPS` — Frame rate.
pub const CAP_PROP_FPS: i32 = 5;
/// `CAP_PROP_FOURCC` — 4-character code of codec. see VideoWriter::fourcc .
pub const CAP_PROP_FOURCC: i32 = 6;
/// `CAP_PROP_FRAME_COUNT` — Number of frames in the video file.
pub const CAP_PROP_FRAME_COUNT: i32 = 7;
/// `CAP_PROP_FORMAT` — Format of the %Mat objects (see Mat::type()) returned
///   by VideoCapture::retrieve().
pub const CAP_PROP_FORMAT: i32 = 8;
/// `CAP_PROP_MODE` — Backend-specific value indicating the current capture
///   mode.
pub const CAP_PROP_MODE: i32 = 9;
/// `CAP_PROP_BRIGHTNESS` — Brightness of the image (only for those cameras
///   that support).
pub const CAP_PROP_BRIGHTNESS: i32 = 10;
/// `CAP_PROP_CONTRAST` — Contrast of the image (only for cameras).
pub const CAP_PROP_CONTRAST: i32 = 11;
/// `CAP_PROP_SATURATION` — Saturation of the image (only for cameras).
pub const CAP_PROP_SATURATION: i32 = 12;
/// `CAP_PROP_HUE` — Hue of the image (only for cameras).
pub const CAP_PROP_HUE: i32 = 13;
/// `CAP_PROP_GAIN` — Gain of the image (only for those cameras that support).
pub const CAP_PROP_GAIN: i32 = 14;
/// `CAP_PROP_EXPOSURE` — Exposure (only for those cameras that support).
pub const CAP_PROP_EXPOSURE: i32 = 15;
/// `CAP_PROP_CONVERT_RGB` — Boolean flags indicating whether images should be
///   converted to BGR. <br/>.
pub const CAP_PROP_CONVERT_RGB: i32 = 16;
/// `CAP_PROP_WHITE_BALANCE_BLUE_U` — Currently unsupported.
pub const CAP_PROP_WHITE_BALANCE_BLUE_U: i32 = 17;
/// `CAP_PROP_RECTIFICATION` — Rectification flag for stereo cameras (note:
///   only supported by DC1394 v 2.x backend currently).
pub const CAP_PROP_RECTIFICATION: i32 = 18;
/// `CAP_PROP_MONOCHROME`.
pub const CAP_PROP_MONOCHROME: i32 = 19;
/// `CAP_PROP_SHARPNESS`.
pub const CAP_PROP_SHARPNESS: i32 = 20;
/// `CAP_PROP_AUTO_EXPOSURE` — DC1394: exposure control done by camera, user
///   can adjust reference level using this feature.
pub const CAP_PROP_AUTO_EXPOSURE: i32 = 21;
/// `CAP_PROP_GAMMA`.
pub const CAP_PROP_GAMMA: i32 = 22;
/// `CAP_PROP_TEMPERATURE`.
pub const CAP_PROP_TEMPERATURE: i32 = 23;
/// `CAP_PROP_TRIGGER`.
pub const CAP_PROP_TRIGGER: i32 = 24;
/// `CAP_PROP_TRIGGER_DELAY`.
pub const CAP_PROP_TRIGGER_DELAY: i32 = 25;
/// `CAP_PROP_WHITE_BALANCE_RED_V`.
pub const CAP_PROP_WHITE_BALANCE_RED_V: i32 = 26;
/// `CAP_PROP_ZOOM` — Android: May switch physical cameras/lenses. Factor and
///   range are hardware-dependent.
pub const CAP_PROP_ZOOM: i32 = 27;
/// `CAP_PROP_FOCUS`.
pub const CAP_PROP_FOCUS: i32 = 28;
/// `CAP_PROP_GUID`.
pub const CAP_PROP_GUID: i32 = 29;
/// `CAP_PROP_ISO_SPEED`.
pub const CAP_PROP_ISO_SPEED: i32 = 30;
/// `CAP_PROP_BACKLIGHT`.
pub const CAP_PROP_BACKLIGHT: i32 = 32;
/// `CAP_PROP_PAN`.
pub const CAP_PROP_PAN: i32 = 33;
/// `CAP_PROP_TILT`.
pub const CAP_PROP_TILT: i32 = 34;
/// `CAP_PROP_ROLL`.
pub const CAP_PROP_ROLL: i32 = 35;
/// `CAP_PROP_IRIS`.
pub const CAP_PROP_IRIS: i32 = 36;
/// `CAP_PROP_SETTINGS` — Pop up video/camera filter dialog (note: only
///   supported by DSHOW backend currently. The property value is ignored).
pub const CAP_PROP_SETTINGS: i32 = 37;
/// `CAP_PROP_BUFFERSIZE`.
pub const CAP_PROP_BUFFERSIZE: i32 = 38;
/// `CAP_PROP_AUTOFOCUS`.
pub const CAP_PROP_AUTOFOCUS: i32 = 39;
/// `CAP_PROP_SAR_NUM` — Sample aspect ratio: num/den (num).
pub const CAP_PROP_SAR_NUM: i32 = 40;
/// `CAP_PROP_SAR_DEN` — Sample aspect ratio: num/den (den).
pub const CAP_PROP_SAR_DEN: i32 = 41;
/// `CAP_PROP_BACKEND` — Current backend (enum VideoCaptureAPIs). Read-only
///   property.
pub const CAP_PROP_BACKEND: i32 = 42;
/// `CAP_PROP_CHANNEL` — Video input or Channel Number (only for those cameras
///   that support).
pub const CAP_PROP_CHANNEL: i32 = 43;
/// `CAP_PROP_AUTO_WB` — enable/ disable auto white-balance.
pub const CAP_PROP_AUTO_WB: i32 = 44;
/// `CAP_PROP_WB_TEMPERATURE` — white-balance color temperature.
pub const CAP_PROP_WB_TEMPERATURE: i32 = 45;
/// `CAP_PROP_CODEC_PIXEL_FORMAT` — (read-only) codec's pixel format.
///   4-character code - see VideoWriter::fourcc . Subset of the `AV_PIX_FMT_*` list (http
///   s://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/raw.c) or -1 if
///   unknown.
pub const CAP_PROP_CODEC_PIXEL_FORMAT: i32 = 46;
/// `CAP_PROP_BITRATE` — (read-only) Video bitrate in kbits/s.
pub const CAP_PROP_BITRATE: i32 = 47;
/// `CAP_PROP_ORIENTATION_META` — (read-only) Frame rotation defined by stream
///   meta (applicable for FFmpeg and AVFoundation back-ends only).
pub const CAP_PROP_ORIENTATION_META: i32 = 48;
/// `CAP_PROP_ORIENTATION_AUTO` — if true - rotates output frames of CvCapture
///   considering video file's metadata (applicable for FFmpeg and AVFoundation
///   back-ends only) (<https://github.com/opencv/opencv/issues/15499>).
pub const CAP_PROP_ORIENTATION_AUTO: i32 = 49;
/// `CAP_PROP_HW_ACCELERATION` — ((open-only)) Hardware acceleration type (see
///   VideoAccelerationType). Setting supported only via 'params' parameter in
///   cv::VideoCapture constructor / .open() method. Default value is backend-
///   specific.
pub const CAP_PROP_HW_ACCELERATION: i32 = 50;
/// `CAP_PROP_HW_DEVICE` — ((open-only)) Hardware device index (select GPU if
///   multiple available). Device enumeration is acceleration type specific.
pub const CAP_PROP_HW_DEVICE: i32 = 51;
/// `CAP_PROP_HW_ACCELERATION_USE_OPENCL` — ((open-only)) If non-zero, create
///   new OpenCL context and bind it to current thread. The OpenCL context
///   created with Video Acceleration context attached it (if not attached yet)
///   for optimized GPU data copy between HW accelerated decoder and cv::UMat.
pub const CAP_PROP_HW_ACCELERATION_USE_OPENCL: i32 = 52;
/// `CAP_PROP_OPEN_TIMEOUT_MSEC` — ((open-only)) timeout in milliseconds for
///   opening a video capture (applicable for FFmpeg and GStreamer back-ends
///   only).
pub const CAP_PROP_OPEN_TIMEOUT_MSEC: i32 = 53;
/// `CAP_PROP_READ_TIMEOUT_MSEC` — ((open-only)) timeout in milliseconds for
///   reading from a video capture (applicable for FFmpeg and GStreamer back-
///   ends only).
pub const CAP_PROP_READ_TIMEOUT_MSEC: i32 = 54;
/// `CAP_PROP_STREAM_OPEN_TIME_USEC` — (read-only) time in microseconds since
///   Jan 1 1970 when stream was opened. Applicable for FFmpeg backend only.
///   Useful for RTSP and other live streams.
pub const CAP_PROP_STREAM_OPEN_TIME_USEC: i32 = 55;
/// `CAP_PROP_VIDEO_TOTAL_CHANNELS` — (read-only) Number of video channels.
pub const CAP_PROP_VIDEO_TOTAL_CHANNELS: i32 = 56;
/// `CAP_PROP_VIDEO_STREAM` — ((open-only)) Specify video stream, 0-based
///   index. Use -1 to disable video stream from file or IP cameras. Default
///   value is 0.
pub const CAP_PROP_VIDEO_STREAM: i32 = 57;
/// `CAP_PROP_AUDIO_STREAM` — ((open-only)) Specify stream in multi-language
///   media files, -1 - disable audio processing or microphone. Default value is
///   -1.
pub const CAP_PROP_AUDIO_STREAM: i32 = 58;
/// `CAP_PROP_AUDIO_POS` — (read-only) Audio position is measured in samples.
///   Accurate audio sample timestamp of previous grabbed fragment. See
///   CAP_PROP_AUDIO_SAMPLES_PER_SECOND and CAP_PROP_AUDIO_SHIFT_NSEC.
pub const CAP_PROP_AUDIO_POS: i32 = 59;
/// `CAP_PROP_AUDIO_SHIFT_NSEC` — (read only) Contains the time difference
///   between the start of the audio stream and the video stream in nanoseconds.
///   Positive value means that audio is started after the first video frame.
///   Negative value means that audio is started before the first video frame.
pub const CAP_PROP_AUDIO_SHIFT_NSEC: i32 = 60;
/// `CAP_PROP_AUDIO_DATA_DEPTH` — (open, read) Alternative definition to bits-
///   per-sample, but with clear handling of 32F / 32S.
pub const CAP_PROP_AUDIO_DATA_DEPTH: i32 = 61;
/// `CAP_PROP_AUDIO_SAMPLES_PER_SECOND` — (open, read) determined from
///   file/codec input. If not specified, then selected audio sample rate is
///   44100.
pub const CAP_PROP_AUDIO_SAMPLES_PER_SECOND: i32 = 62;
/// `CAP_PROP_AUDIO_BASE_INDEX` — (read-only) Index of the first audio channel
///   for .retrieve() calls. That audio channel number continues enumeration
///   after video channels.
pub const CAP_PROP_AUDIO_BASE_INDEX: i32 = 63;
/// `CAP_PROP_AUDIO_TOTAL_CHANNELS` — (read-only) Number of audio channels in
///   the selected audio stream (mono, stereo, etc).
pub const CAP_PROP_AUDIO_TOTAL_CHANNELS: i32 = 64;
/// `CAP_PROP_AUDIO_TOTAL_STREAMS` — (read-only) Number of audio streams.
pub const CAP_PROP_AUDIO_TOTAL_STREAMS: i32 = 65;
/// `CAP_PROP_AUDIO_SYNCHRONIZE` — (open, read) Enables audio synchronization.
pub const CAP_PROP_AUDIO_SYNCHRONIZE: i32 = 66;
/// `CAP_PROP_LRF_HAS_KEY_FRAME` — FFmpeg back-end only - Indicates whether
///   the Last Raw Frame (LRF), output from VideoCapture::read() when
///   VideoCapture is initialized with VideoCapture::open(CAP_FFMPEG,
///   {CAP_PROP_FORMAT, -1}) or VideoCapture::set(CAP_PROP_FORMAT,-1) is called
///   before the first call to VideoCapture::read(), contains encoded data for a
///   key frame.
pub const CAP_PROP_LRF_HAS_KEY_FRAME: i32 = 67;
/// `CAP_PROP_CODEC_EXTRADATA_INDEX` — Positive index indicates that returning
///   extra data is supported by the video back end. This can be retrieved as
///   `cap.retrieve(data, <returned index>)`. E.g. When reading from a h264
///   encoded RTSP stream, the FFmpeg backend could return the SPS and/or PPS if
///   available (if sent in reply to a DESCRIBE request), from calls to
///   `cap.retrieve(data, <returned index>)`.
pub const CAP_PROP_CODEC_EXTRADATA_INDEX: i32 = 68;
/// `CAP_PROP_FRAME_TYPE` — (read-only) FFmpeg back-end only - Frame type
///   ascii code (73 = 'I', 80 = 'P', 66 = 'B' or 63 = '?' if unknown) of the
///   most recently read frame.
pub const CAP_PROP_FRAME_TYPE: i32 = 69;
/// `CAP_PROP_N_THREADS` — ((open-only)) Set the maximum number of threads to
///   use. Use 0 to use as many threads as CPU cores (applicable for FFmpeg
///   back-end only).
pub const CAP_PROP_N_THREADS: i32 = 70;
/// `CAP_PROP_PTS` — (read-only) FFmpeg back-end only - presentation timestamp
///   of the most recently read frame using the FPS time base. e.g. fps = 25,
///   VideoCapture::get(\ref CAP_PROP_PTS) = 3, presentation time = 3/25
///   seconds.
pub const CAP_PROP_PTS: i32 = 71;
/// `CAP_PROP_DTS_DELAY` — (read-only) FFmpeg back-end only - maximum
///   difference between presentation (pts) and decompression timestamps (dts)
///   using FPS time base. e.g. delay is maximum when frame_num = 0, if true,
///   VideoCapture::get(\ref CAP_PROP_PTS) = 0 and VideoCapture::get(\ref
///   CAP_PROP_DTS_DELAY) = 2, dts = -2. Non zero values usually imply the
///   stream is encoded using B-frames which are not decoded in presentation
///   order.
pub const CAP_PROP_DTS_DELAY: i32 = 72;
/// `CAP_PROP_IMAGE_SEQ_START` — ((open-only)) Start number for image
///   sequences opened with a printf-style pattern (e.g. 'frame_%05d.dpx'). Sets
///   the initial frame number and disables automatic first-frame detection.
///   Applicable to \ref CAP_FFMPEG (passed as the image2 demuxer
///   'start_number') and \ref CAP_IMAGES backends. Default: not set (automatic
///   detection).
pub const CAP_PROP_IMAGE_SEQ_START: i32 = 73;
/// `CV__CAP_PROP_LATEST`.
pub const CV__CAP_PROP_LATEST: i32 = 74;
/// `VIDEOWRITER_PROP_QUALITY` — Current quality (0..100%) of the encoded
///   videostream. Can be adjusted dynamically in some codecs.
pub const VIDEOWRITER_PROP_QUALITY: i32 = 1;
/// `VIDEOWRITER_PROP_FRAMEBYTES` — (Read-only): Size of just encoded video
///   frame. Note that the encoding order may be different from representation
///   order.
pub const VIDEOWRITER_PROP_FRAMEBYTES: i32 = 2;
/// `VIDEOWRITER_PROP_NSTRIPES` — Number of stripes for parallel encoding. -1
///   for auto detection.
pub const VIDEOWRITER_PROP_NSTRIPES: i32 = 3;
/// `VIDEOWRITER_PROP_IS_COLOR` — If it is not zero, the encoder will expect
///   and encode color frames, otherwise it.
pub const VIDEOWRITER_PROP_IS_COLOR: i32 = 4;
/// `VIDEOWRITER_PROP_DEPTH` — Defaults to \ref CV_8U.
pub const VIDEOWRITER_PROP_DEPTH: i32 = 5;
/// `VIDEOWRITER_PROP_HW_ACCELERATION` — ((open-only)) Hardware acceleration
///   type (see VideoAccelerationType). Setting supported only via 'params'
///   parameter in VideoWriter constructor / .open() method. Default value is
///   backend-specific.
pub const VIDEOWRITER_PROP_HW_ACCELERATION: i32 = 6;
/// `VIDEOWRITER_PROP_HW_DEVICE` — ((open-only)) Hardware device index (select
///   GPU if multiple available). Device enumeration is acceleration type
///   specific.
pub const VIDEOWRITER_PROP_HW_DEVICE: i32 = 7;
/// `VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL` — ((open-only)) If non-zero,
///   create new OpenCL context and bind it to current thread. The OpenCL
///   context created with Video Acceleration context attached it (if not
///   attached yet) for optimized GPU data copy between cv::UMat and HW
///   accelerated encoder.
pub const VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL: i32 = 8;
/// `VIDEOWRITER_PROP_RAW_VIDEO` — ((open-only)) Set to non-zero to enable
///   encapsulation of an encoded raw video stream. Each raw encoded video frame
///   should be passed to VideoWriter::write() as single row or column of a \ref
///   CV_8UC1 Mat. \note If the key frame interval is not 1 then it must be
///   manually specified by the user. This can either be performed during
///   initialization passing \ref VIDEOWRITER_PROP_KEY_INTERVAL as one of the
///   extra encoder params to \ref VideoWriter::VideoWriter(const String &, int,
///   double, const Size &, const std::vector< int > &params) or afterwards by
///   setting the \ref VIDEOWRITER_PROP_KEY_FLAG with \ref VideoWriter::set()
///   before writing each frame. FFMpeg backend only.
pub const VIDEOWRITER_PROP_RAW_VIDEO: i32 = 9;
/// `VIDEOWRITER_PROP_KEY_INTERVAL` — ((open-only)) Set the key frame interval
///   using raw video encapsulation (\ref VIDEOWRITER_PROP_RAW_VIDEO != 0).
///   Defaults to 1 when not set. FFmpeg back-end only.
pub const VIDEOWRITER_PROP_KEY_INTERVAL: i32 = 10;
/// `VIDEOWRITER_PROP_KEY_FLAG` — Set to non-zero to signal that the following
///   frames are key frames or zero if not, when encapsulating raw video (\ref
///   VIDEOWRITER_PROP_RAW_VIDEO != 0). FFmpeg back-end only.
pub const VIDEOWRITER_PROP_KEY_FLAG: i32 = 11;
/// `VIDEOWRITER_PROP_PTS` — Specifies the frame presentation timestamp for
///   each frame using the FPS time base. This property is only necessary when
///   encapsulating externally encoded video where the decoding order differs
///   from the presentation order, such as in GOP patterns with bi-directional
///   B-frames. The value should be provided by your external encoder and for
///   video sources with fixed frame rates it is equivalent to dividing the
///   current frame's presentation time (\ref CAP_PROP_POS_MSEC) by the frame
///   duration (1000.0 / VideoCapture::get(\ref CAP_PROP_FPS)). It can be
///   queried from the resulting encapsulated video file using
///   VideoCapture::get(\ref CAP_PROP_PTS). FFmpeg back-end only.
pub const VIDEOWRITER_PROP_PTS: i32 = 12;
/// `VIDEOWRITER_PROP_DTS_DELAY` — Specifies the maximum difference between
///   presentation (pts) and decompression timestamps (dts) using the FPS time
///   base. This property is necessary only when encapsulating externally
///   encoded video where the decoding order differs from the presentation
///   order, such as in GOP patterns with bi-directional B-frames. The value
///   should be calculated based on the specific GOP pattern used during
///   encoding. For example, in a GOP with presentation order IBP and decoding
///   order IPB, this value would be 1, as the B-frame is the second frame
///   presented but the third to be decoded. It can be queried from the
///   resulting encapsulated video file using VideoCapture::get(\ref
///   CAP_PROP_DTS_DELAY). Non-zero values usually imply the stream is encoded
///   using B-frames. FFmpeg back-end only.
pub const VIDEOWRITER_PROP_DTS_DELAY: i32 = 13;
/// `VIDEOWRITER_PROP_COLOR_SPACE` — ((open-only)) GStreamer backend only.
///   Pixel format for the encoding profile. Default is "I420". Other values:
///   "NV12", "BGRx". See GStreamer raw video formats for more options.
pub const VIDEOWRITER_PROP_COLOR_SPACE: i32 = 14;
/// `VIDEOWRITER_PROP_ENABLE_ALPHA` — ((open-only)) FFmpeg backend only.
///   Defines that input frames contain alpha channel.
pub const VIDEOWRITER_PROP_ENABLE_ALPHA: i32 = 15;
/// `CV__VIDEOWRITER_PROP_LATEST`.
pub const CV__VIDEOWRITER_PROP_LATEST: i32 = 16;
/// `VIDEO_ACCELERATION_NONE` — Do not require any specific H/W acceleration,
///   prefer software processing.
pub const VIDEO_ACCELERATION_NONE: i32 = 0;
/// `VIDEO_ACCELERATION_ANY` — Prefer to use H/W acceleration. If no one
///   supported, then fallback to software processing.
pub const VIDEO_ACCELERATION_ANY: i32 = 1;
/// `VIDEO_ACCELERATION_D3D11` — DirectX 11.
pub const VIDEO_ACCELERATION_D3D11: i32 = 2;
/// `VIDEO_ACCELERATION_VAAPI` — VAAPI.
pub const VIDEO_ACCELERATION_VAAPI: i32 = 3;
/// `VIDEO_ACCELERATION_MFX` — libmfx (Intel MediaSDK/oneVPL).
pub const VIDEO_ACCELERATION_MFX: i32 = 4;
/// `VIDEO_ACCELERATION_DRM` — Raspberry Pi V4.
pub const VIDEO_ACCELERATION_DRM: i32 = 5;
/// `CAP_PROP_DC1394_OFF` — turn the feature off (not controlled manually nor
///   automatically).
pub const CAP_PROP_DC1394_OFF: i32 = -4;
/// `CAP_PROP_DC1394_MODE_MANUAL` — set automatically when a value of the
///   feature is set by the user.
pub const CAP_PROP_DC1394_MODE_MANUAL: i32 = -3;
/// `CAP_PROP_DC1394_MODE_AUTO`.
pub const CAP_PROP_DC1394_MODE_AUTO: i32 = -2;
/// `CAP_PROP_DC1394_MODE_ONE_PUSH_AUTO`.
pub const CAP_PROP_DC1394_MODE_ONE_PUSH_AUTO: i32 = -1;
/// `CAP_PROP_DC1394_MAX`.
pub const CAP_PROP_DC1394_MAX: i32 = 31;
/// `CAP_OPENNI_DEPTH_GENERATOR`.
pub const CAP_OPENNI_DEPTH_GENERATOR: i32 = -2147483648;
/// `CAP_OPENNI_IMAGE_GENERATOR`.
pub const CAP_OPENNI_IMAGE_GENERATOR: i32 = 1073741824;
/// `CAP_OPENNI_IR_GENERATOR`.
pub const CAP_OPENNI_IR_GENERATOR: i32 = 536870912;
/// `CAP_OPENNI_GENERATORS_MASK`.
pub const CAP_OPENNI_GENERATORS_MASK: i32 = -536870912;
/// `CAP_PROP_OPENNI_OUTPUT_MODE`.
pub const CAP_PROP_OPENNI_OUTPUT_MODE: i32 = 100;
/// `CAP_PROP_OPENNI_FRAME_MAX_DEPTH` — In mm.
pub const CAP_PROP_OPENNI_FRAME_MAX_DEPTH: i32 = 101;
/// `CAP_PROP_OPENNI_BASELINE` — In mm.
pub const CAP_PROP_OPENNI_BASELINE: i32 = 102;
/// `CAP_PROP_OPENNI_FOCAL_LENGTH` — In pixels.
pub const CAP_PROP_OPENNI_FOCAL_LENGTH: i32 = 103;
/// `CAP_PROP_OPENNI_REGISTRATION` — Flag that synchronizes the remapping
///   depth map to image map.
pub const CAP_PROP_OPENNI_REGISTRATION: i32 = 104;
/// `CAP_PROP_OPENNI_REGISTRATION_ON`.
pub const CAP_PROP_OPENNI_REGISTRATION_ON: i32 = 104;
/// `CAP_PROP_OPENNI_APPROX_FRAME_SYNC`.
pub const CAP_PROP_OPENNI_APPROX_FRAME_SYNC: i32 = 105;
/// `CAP_PROP_OPENNI_MAX_BUFFER_SIZE`.
pub const CAP_PROP_OPENNI_MAX_BUFFER_SIZE: i32 = 106;
/// `CAP_PROP_OPENNI_CIRCLE_BUFFER`.
pub const CAP_PROP_OPENNI_CIRCLE_BUFFER: i32 = 107;
/// `CAP_PROP_OPENNI_MAX_TIME_DURATION`.
pub const CAP_PROP_OPENNI_MAX_TIME_DURATION: i32 = 108;
/// `CAP_PROP_OPENNI_GENERATOR_PRESENT`.
pub const CAP_PROP_OPENNI_GENERATOR_PRESENT: i32 = 109;
/// `CAP_PROP_OPENNI2_SYNC`.
pub const CAP_PROP_OPENNI2_SYNC: i32 = 110;
/// `CAP_PROP_OPENNI2_MIRROR`.
pub const CAP_PROP_OPENNI2_MIRROR: i32 = 111;
/// `CAP_OPENNI_IMAGE_GENERATOR_PRESENT`.
pub const CAP_OPENNI_IMAGE_GENERATOR_PRESENT: i32 = 1073741933;
/// `CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE`.
pub const CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE: i32 = 1073741924;
/// `CAP_OPENNI_DEPTH_GENERATOR_PRESENT`.
pub const CAP_OPENNI_DEPTH_GENERATOR_PRESENT: i32 = -2147483539;
/// `CAP_OPENNI_DEPTH_GENERATOR_BASELINE`.
pub const CAP_OPENNI_DEPTH_GENERATOR_BASELINE: i32 = -2147483546;
/// `CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH`.
pub const CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH: i32 = -2147483545;
/// `CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION`.
pub const CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION: i32 = -2147483544;
/// `CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION_ON`.
pub const CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION_ON: i32 = -2147483544;
/// `CAP_OPENNI_IR_GENERATOR_PRESENT`.
pub const CAP_OPENNI_IR_GENERATOR_PRESENT: i32 = 536871021;
/// `CAP_OPENNI_DEPTH_MAP` — Depth values in mm (CV_16UC1).
pub const CAP_OPENNI_DEPTH_MAP: i32 = 0;
/// `CAP_OPENNI_POINT_CLOUD_MAP` — XYZ in meters (CV_32FC3).
pub const CAP_OPENNI_POINT_CLOUD_MAP: i32 = 1;
/// `CAP_OPENNI_DISPARITY_MAP` — Disparity in pixels (CV_8UC1).
pub const CAP_OPENNI_DISPARITY_MAP: i32 = 2;
/// `CAP_OPENNI_DISPARITY_MAP_32F` — Disparity in pixels (CV_32FC1).
pub const CAP_OPENNI_DISPARITY_MAP_32F: i32 = 3;
/// `CAP_OPENNI_VALID_DEPTH_MASK` — CV_8UC1.
pub const CAP_OPENNI_VALID_DEPTH_MASK: i32 = 4;
/// `CAP_OPENNI_BGR_IMAGE` — Data given from RGB image generator.
pub const CAP_OPENNI_BGR_IMAGE: i32 = 5;
/// `CAP_OPENNI_GRAY_IMAGE` — Data given from RGB image generator.
pub const CAP_OPENNI_GRAY_IMAGE: i32 = 6;
/// `CAP_OPENNI_IR_IMAGE` — Data given from IR image generator.
pub const CAP_OPENNI_IR_IMAGE: i32 = 7;
/// `CAP_OPENNI_VGA_30HZ`.
pub const CAP_OPENNI_VGA_30HZ: i32 = 0;
/// `CAP_OPENNI_SXGA_15HZ`.
pub const CAP_OPENNI_SXGA_15HZ: i32 = 1;
/// `CAP_OPENNI_SXGA_30HZ`.
pub const CAP_OPENNI_SXGA_30HZ: i32 = 2;
/// `CAP_OPENNI_QVGA_30HZ`.
pub const CAP_OPENNI_QVGA_30HZ: i32 = 3;
/// `CAP_OPENNI_QVGA_60HZ`.
pub const CAP_OPENNI_QVGA_60HZ: i32 = 4;
/// `CAP_PROP_GSTREAMER_QUEUE_LENGTH` — Default is 1.
pub const CAP_PROP_GSTREAMER_QUEUE_LENGTH: i32 = 200;
/// `CAP_PROP_PVAPI_MULTICASTIP` — IP for enable multicast master mode. 0 for
///   disable multicast.
pub const CAP_PROP_PVAPI_MULTICASTIP: i32 = 300;
/// `CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE` — FrameStartTriggerMode: Determines
///   how a frame is initiated.
pub const CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE: i32 = 301;
/// `CAP_PROP_PVAPI_DECIMATIONHORIZONTAL` — Horizontal sub-sampling of the
///   image.
pub const CAP_PROP_PVAPI_DECIMATIONHORIZONTAL: i32 = 302;
/// `CAP_PROP_PVAPI_DECIMATIONVERTICAL` — Vertical sub-sampling of the image.
pub const CAP_PROP_PVAPI_DECIMATIONVERTICAL: i32 = 303;
/// `CAP_PROP_PVAPI_BINNINGX` — Horizontal binning factor.
pub const CAP_PROP_PVAPI_BINNINGX: i32 = 304;
/// `CAP_PROP_PVAPI_BINNINGY` — Vertical binning factor.
pub const CAP_PROP_PVAPI_BINNINGY: i32 = 305;
/// `CAP_PROP_PVAPI_PIXELFORMAT` — Pixel format.
pub const CAP_PROP_PVAPI_PIXELFORMAT: i32 = 306;
/// `CAP_PVAPI_FSTRIGMODE_FREERUN` — Freerun.
pub const CAP_PVAPI_FSTRIGMODE_FREERUN: i32 = 0;
/// `CAP_PVAPI_FSTRIGMODE_SYNCIN1` — SyncIn1.
pub const CAP_PVAPI_FSTRIGMODE_SYNCIN1: i32 = 1;
/// `CAP_PVAPI_FSTRIGMODE_SYNCIN2` — SyncIn2.
pub const CAP_PVAPI_FSTRIGMODE_SYNCIN2: i32 = 2;
/// `CAP_PVAPI_FSTRIGMODE_FIXEDRATE` — FixedRate.
pub const CAP_PVAPI_FSTRIGMODE_FIXEDRATE: i32 = 3;
/// `CAP_PVAPI_FSTRIGMODE_SOFTWARE` — Software.
pub const CAP_PVAPI_FSTRIGMODE_SOFTWARE: i32 = 4;
/// `CAP_PVAPI_DECIMATION_OFF` — Off.
pub const CAP_PVAPI_DECIMATION_OFF: i32 = 1;
/// `CAP_PVAPI_DECIMATION_2OUTOF4` — 2 out of 4 decimation.
pub const CAP_PVAPI_DECIMATION_2OUTOF4: i32 = 2;
/// `CAP_PVAPI_DECIMATION_2OUTOF8` — 2 out of 8 decimation.
pub const CAP_PVAPI_DECIMATION_2OUTOF8: i32 = 4;
/// `CAP_PVAPI_DECIMATION_2OUTOF16` — 2 out of 16 decimation.
pub const CAP_PVAPI_DECIMATION_2OUTOF16: i32 = 8;
/// `CAP_PVAPI_PIXELFORMAT_MONO8` — Mono8.
pub const CAP_PVAPI_PIXELFORMAT_MONO8: i32 = 1;
/// `CAP_PVAPI_PIXELFORMAT_MONO16` — Mono16.
pub const CAP_PVAPI_PIXELFORMAT_MONO16: i32 = 2;
/// `CAP_PVAPI_PIXELFORMAT_BAYER8` — Bayer8.
pub const CAP_PVAPI_PIXELFORMAT_BAYER8: i32 = 3;
/// `CAP_PVAPI_PIXELFORMAT_BAYER16` — Bayer16.
pub const CAP_PVAPI_PIXELFORMAT_BAYER16: i32 = 4;
/// `CAP_PVAPI_PIXELFORMAT_RGB24` — Rgb24.
pub const CAP_PVAPI_PIXELFORMAT_RGB24: i32 = 5;
/// `CAP_PVAPI_PIXELFORMAT_BGR24` — Bgr24.
pub const CAP_PVAPI_PIXELFORMAT_BGR24: i32 = 6;
/// `CAP_PVAPI_PIXELFORMAT_RGBA32` — Rgba32.
pub const CAP_PVAPI_PIXELFORMAT_RGBA32: i32 = 7;
/// `CAP_PVAPI_PIXELFORMAT_BGRA32` — Bgra32.
pub const CAP_PVAPI_PIXELFORMAT_BGRA32: i32 = 8;
/// `CAP_PROP_XI_DOWNSAMPLING` — Change image resolution by binning or
///   skipping.
pub const CAP_PROP_XI_DOWNSAMPLING: i32 = 400;
/// `CAP_PROP_XI_DATA_FORMAT` — Output data format.
pub const CAP_PROP_XI_DATA_FORMAT: i32 = 401;
/// `CAP_PROP_XI_OFFSET_X` — Horizontal offset from the origin to the area of
///   interest (in pixels).
pub const CAP_PROP_XI_OFFSET_X: i32 = 402;
/// `CAP_PROP_XI_OFFSET_Y` — Vertical offset from the origin to the area of
///   interest (in pixels).
pub const CAP_PROP_XI_OFFSET_Y: i32 = 403;
/// `CAP_PROP_XI_TRG_SOURCE` — Defines source of trigger.
pub const CAP_PROP_XI_TRG_SOURCE: i32 = 404;
/// `CAP_PROP_XI_TRG_SOFTWARE` — Generates an internal trigger. PRM_TRG_SOURCE
///   must be set to TRG_SOFTWARE.
pub const CAP_PROP_XI_TRG_SOFTWARE: i32 = 405;
/// `CAP_PROP_XI_GPI_SELECTOR` — Selects general purpose input.
pub const CAP_PROP_XI_GPI_SELECTOR: i32 = 406;
/// `CAP_PROP_XI_GPI_MODE` — Set general purpose input mode.
pub const CAP_PROP_XI_GPI_MODE: i32 = 407;
/// `CAP_PROP_XI_GPI_LEVEL` — Get general purpose level.
pub const CAP_PROP_XI_GPI_LEVEL: i32 = 408;
/// `CAP_PROP_XI_GPO_SELECTOR` — Selects general purpose output.
pub const CAP_PROP_XI_GPO_SELECTOR: i32 = 409;
/// `CAP_PROP_XI_GPO_MODE` — Set general purpose output mode.
pub const CAP_PROP_XI_GPO_MODE: i32 = 410;
/// `CAP_PROP_XI_LED_SELECTOR` — Selects camera signalling LED.
pub const CAP_PROP_XI_LED_SELECTOR: i32 = 411;
/// `CAP_PROP_XI_LED_MODE` — Define camera signalling LED functionality.
pub const CAP_PROP_XI_LED_MODE: i32 = 412;
/// `CAP_PROP_XI_MANUAL_WB` — Calculates White Balance(must be called during
///   acquisition).
pub const CAP_PROP_XI_MANUAL_WB: i32 = 413;
/// `CAP_PROP_XI_AUTO_WB` — Automatic white balance.
pub const CAP_PROP_XI_AUTO_WB: i32 = 414;
/// `CAP_PROP_XI_AEAG` — Automatic exposure/gain.
pub const CAP_PROP_XI_AEAG: i32 = 415;
/// `CAP_PROP_XI_EXP_PRIORITY` — Exposure priority (0.5 - exposure 50%, gain
///   50%).
pub const CAP_PROP_XI_EXP_PRIORITY: i32 = 416;
/// `CAP_PROP_XI_AE_MAX_LIMIT` — Maximum limit of exposure in AEAG procedure.
pub const CAP_PROP_XI_AE_MAX_LIMIT: i32 = 417;
/// `CAP_PROP_XI_AG_MAX_LIMIT` — Maximum limit of gain in AEAG procedure.
pub const CAP_PROP_XI_AG_MAX_LIMIT: i32 = 418;
/// `CAP_PROP_XI_AEAG_LEVEL` — Average intensity of output signal AEAG should
///   achieve(in %).
pub const CAP_PROP_XI_AEAG_LEVEL: i32 = 419;
/// `CAP_PROP_XI_TIMEOUT` — Image capture timeout in milliseconds.
pub const CAP_PROP_XI_TIMEOUT: i32 = 420;
/// `CAP_PROP_XI_EXPOSURE` — Exposure time in microseconds.
pub const CAP_PROP_XI_EXPOSURE: i32 = 421;
/// `CAP_PROP_XI_EXPOSURE_BURST_COUNT` — Sets the number of times of exposure
///   in one frame.
pub const CAP_PROP_XI_EXPOSURE_BURST_COUNT: i32 = 422;
/// `CAP_PROP_XI_GAIN_SELECTOR` — Gain selector for parameter Gain allows to
///   select different type of gains.
pub const CAP_PROP_XI_GAIN_SELECTOR: i32 = 423;
/// `CAP_PROP_XI_GAIN` — Gain in dB.
pub const CAP_PROP_XI_GAIN: i32 = 424;
/// `CAP_PROP_XI_DOWNSAMPLING_TYPE` — Change image downsampling type.
pub const CAP_PROP_XI_DOWNSAMPLING_TYPE: i32 = 426;
/// `CAP_PROP_XI_BINNING_SELECTOR` — Binning engine selector.
pub const CAP_PROP_XI_BINNING_SELECTOR: i32 = 427;
/// `CAP_PROP_XI_BINNING_VERTICAL` — Vertical Binning - number of vertical
///   photo-sensitive cells to combine together.
pub const CAP_PROP_XI_BINNING_VERTICAL: i32 = 428;
/// `CAP_PROP_XI_BINNING_HORIZONTAL` — Horizontal Binning - number of
///   horizontal photo-sensitive cells to combine together.
pub const CAP_PROP_XI_BINNING_HORIZONTAL: i32 = 429;
/// `CAP_PROP_XI_BINNING_PATTERN` — Binning pattern type.
pub const CAP_PROP_XI_BINNING_PATTERN: i32 = 430;
/// `CAP_PROP_XI_DECIMATION_SELECTOR` — Decimation engine selector.
pub const CAP_PROP_XI_DECIMATION_SELECTOR: i32 = 431;
/// `CAP_PROP_XI_DECIMATION_VERTICAL` — Vertical Decimation - vertical sub-
///   sampling of the image - reduces the vertical resolution of the image by
///   the specified vertical decimation factor.
pub const CAP_PROP_XI_DECIMATION_VERTICAL: i32 = 432;
/// `CAP_PROP_XI_DECIMATION_HORIZONTAL` — Horizontal Decimation - horizontal
///   sub-sampling of the image - reduces the horizontal resolution of the image
///   by the specified vertical decimation factor.
pub const CAP_PROP_XI_DECIMATION_HORIZONTAL: i32 = 433;
/// `CAP_PROP_XI_DECIMATION_PATTERN` — Decimation pattern type.
pub const CAP_PROP_XI_DECIMATION_PATTERN: i32 = 434;
/// `CAP_PROP_XI_TEST_PATTERN_GENERATOR_SELECTOR` — Selects which test pattern
///   generator is controlled by the TestPattern feature.
pub const CAP_PROP_XI_TEST_PATTERN_GENERATOR_SELECTOR: i32 = 587;
/// `CAP_PROP_XI_TEST_PATTERN` — Selects which test pattern type is generated
///   by the selected generator.
pub const CAP_PROP_XI_TEST_PATTERN: i32 = 588;
/// `CAP_PROP_XI_IMAGE_DATA_FORMAT` — Output data format.
pub const CAP_PROP_XI_IMAGE_DATA_FORMAT: i32 = 435;
/// `CAP_PROP_XI_SHUTTER_TYPE` — Change sensor shutter type(CMOS sensor).
pub const CAP_PROP_XI_SHUTTER_TYPE: i32 = 436;
/// `CAP_PROP_XI_SENSOR_TAPS` — Number of taps.
pub const CAP_PROP_XI_SENSOR_TAPS: i32 = 437;
/// `CAP_PROP_XI_AEAG_ROI_OFFSET_X` — Automatic exposure/gain ROI offset X.
pub const CAP_PROP_XI_AEAG_ROI_OFFSET_X: i32 = 439;
/// `CAP_PROP_XI_AEAG_ROI_OFFSET_Y` — Automatic exposure/gain ROI offset Y.
pub const CAP_PROP_XI_AEAG_ROI_OFFSET_Y: i32 = 440;
/// `CAP_PROP_XI_AEAG_ROI_WIDTH` — Automatic exposure/gain ROI Width.
pub const CAP_PROP_XI_AEAG_ROI_WIDTH: i32 = 441;
/// `CAP_PROP_XI_AEAG_ROI_HEIGHT` — Automatic exposure/gain ROI Height.
pub const CAP_PROP_XI_AEAG_ROI_HEIGHT: i32 = 442;
/// `CAP_PROP_XI_BPC` — Correction of bad pixels.
pub const CAP_PROP_XI_BPC: i32 = 445;
/// `CAP_PROP_XI_WB_KR` — White balance red coefficient.
pub const CAP_PROP_XI_WB_KR: i32 = 448;
/// `CAP_PROP_XI_WB_KG` — White balance green coefficient.
pub const CAP_PROP_XI_WB_KG: i32 = 449;
/// `CAP_PROP_XI_WB_KB` — White balance blue coefficient.
pub const CAP_PROP_XI_WB_KB: i32 = 450;
/// `CAP_PROP_XI_WIDTH` — Width of the Image provided by the device (in
///   pixels).
pub const CAP_PROP_XI_WIDTH: i32 = 451;
/// `CAP_PROP_XI_HEIGHT` — Height of the Image provided by the device (in
///   pixels).
pub const CAP_PROP_XI_HEIGHT: i32 = 452;
/// `CAP_PROP_XI_REGION_SELECTOR` — Selects Region in Multiple ROI which
///   parameters are set by width, height, ... ,region mode.
pub const CAP_PROP_XI_REGION_SELECTOR: i32 = 589;
/// `CAP_PROP_XI_REGION_MODE` — Activates/deactivates Region selected by
///   Region Selector.
pub const CAP_PROP_XI_REGION_MODE: i32 = 595;
/// `CAP_PROP_XI_LIMIT_BANDWIDTH` — Set/get bandwidth(datarate)(in Megabits).
pub const CAP_PROP_XI_LIMIT_BANDWIDTH: i32 = 459;
/// `CAP_PROP_XI_SENSOR_DATA_BIT_DEPTH` — Sensor output data bit depth.
pub const CAP_PROP_XI_SENSOR_DATA_BIT_DEPTH: i32 = 460;
/// `CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH` — Device output data bit depth.
pub const CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH: i32 = 461;
/// `CAP_PROP_XI_IMAGE_DATA_BIT_DEPTH` — bitdepth of data returned by function
///   xiGetImage.
pub const CAP_PROP_XI_IMAGE_DATA_BIT_DEPTH: i32 = 462;
/// `CAP_PROP_XI_OUTPUT_DATA_PACKING` — Device output data packing (or
///   grouping) enabled. Packing could be enabled if output_data_bit_depth > 8
///   and packing capability is available.
pub const CAP_PROP_XI_OUTPUT_DATA_PACKING: i32 = 463;
/// `CAP_PROP_XI_OUTPUT_DATA_PACKING_TYPE` — Data packing type. Some cameras
///   supports only specific packing type.
pub const CAP_PROP_XI_OUTPUT_DATA_PACKING_TYPE: i32 = 464;
/// `CAP_PROP_XI_IS_COOLED` — Returns 1 for cameras that support cooling.
pub const CAP_PROP_XI_IS_COOLED: i32 = 465;
/// `CAP_PROP_XI_COOLING` — Start camera cooling.
pub const CAP_PROP_XI_COOLING: i32 = 466;
/// `CAP_PROP_XI_TARGET_TEMP` — Set sensor target temperature for cooling.
pub const CAP_PROP_XI_TARGET_TEMP: i32 = 467;
/// `CAP_PROP_XI_CHIP_TEMP` — Camera sensor temperature.
pub const CAP_PROP_XI_CHIP_TEMP: i32 = 468;
/// `CAP_PROP_XI_HOUS_TEMP` — Camera housing temperature.
pub const CAP_PROP_XI_HOUS_TEMP: i32 = 469;
/// `CAP_PROP_XI_HOUS_BACK_SIDE_TEMP` — Camera housing back side temperature.
pub const CAP_PROP_XI_HOUS_BACK_SIDE_TEMP: i32 = 590;
/// `CAP_PROP_XI_SENSOR_BOARD_TEMP` — Camera sensor board temperature.
pub const CAP_PROP_XI_SENSOR_BOARD_TEMP: i32 = 596;
/// `CAP_PROP_XI_CMS` — Mode of color management system.
pub const CAP_PROP_XI_CMS: i32 = 470;
/// `CAP_PROP_XI_APPLY_CMS` — Enable applying of CMS profiles to xiGetImage
///   (see XI_PRM_INPUT_CMS_PROFILE, XI_PRM_OUTPUT_CMS_PROFILE).
pub const CAP_PROP_XI_APPLY_CMS: i32 = 471;
/// `CAP_PROP_XI_IMAGE_IS_COLOR` — Returns 1 for color cameras.
pub const CAP_PROP_XI_IMAGE_IS_COLOR: i32 = 474;
/// `CAP_PROP_XI_COLOR_FILTER_ARRAY` — Returns color filter array type of RAW
///   data.
pub const CAP_PROP_XI_COLOR_FILTER_ARRAY: i32 = 475;
/// `CAP_PROP_XI_GAMMAY` — Luminosity gamma.
pub const CAP_PROP_XI_GAMMAY: i32 = 476;
/// `CAP_PROP_XI_GAMMAC` — Chromaticity gamma.
pub const CAP_PROP_XI_GAMMAC: i32 = 477;
/// `CAP_PROP_XI_SHARPNESS` — Sharpness Strength.
pub const CAP_PROP_XI_SHARPNESS: i32 = 478;
/// `CAP_PROP_XI_CC_MATRIX_00` — Color Correction Matrix element (0, 0).
pub const CAP_PROP_XI_CC_MATRIX_00: i32 = 479;
/// `CAP_PROP_XI_CC_MATRIX_01` — Color Correction Matrix element (0, 1).
pub const CAP_PROP_XI_CC_MATRIX_01: i32 = 480;
/// `CAP_PROP_XI_CC_MATRIX_02` — Color Correction Matrix element (0, 2).
pub const CAP_PROP_XI_CC_MATRIX_02: i32 = 481;
/// `CAP_PROP_XI_CC_MATRIX_03` — Color Correction Matrix element (0, 3).
pub const CAP_PROP_XI_CC_MATRIX_03: i32 = 482;
/// `CAP_PROP_XI_CC_MATRIX_10` — Color Correction Matrix element (1, 0).
pub const CAP_PROP_XI_CC_MATRIX_10: i32 = 483;
/// `CAP_PROP_XI_CC_MATRIX_11` — Color Correction Matrix element (1, 1).
pub const CAP_PROP_XI_CC_MATRIX_11: i32 = 484;
/// `CAP_PROP_XI_CC_MATRIX_12` — Color Correction Matrix element (1, 2).
pub const CAP_PROP_XI_CC_MATRIX_12: i32 = 485;
/// `CAP_PROP_XI_CC_MATRIX_13` — Color Correction Matrix element (1, 3).
pub const CAP_PROP_XI_CC_MATRIX_13: i32 = 486;
/// `CAP_PROP_XI_CC_MATRIX_20` — Color Correction Matrix element (2, 0).
pub const CAP_PROP_XI_CC_MATRIX_20: i32 = 487;
/// `CAP_PROP_XI_CC_MATRIX_21` — Color Correction Matrix element (2, 1).
pub const CAP_PROP_XI_CC_MATRIX_21: i32 = 488;
/// `CAP_PROP_XI_CC_MATRIX_22` — Color Correction Matrix element (2, 2).
pub const CAP_PROP_XI_CC_MATRIX_22: i32 = 489;
/// `CAP_PROP_XI_CC_MATRIX_23` — Color Correction Matrix element (2, 3).
pub const CAP_PROP_XI_CC_MATRIX_23: i32 = 490;
/// `CAP_PROP_XI_CC_MATRIX_30` — Color Correction Matrix element (3, 0).
pub const CAP_PROP_XI_CC_MATRIX_30: i32 = 491;
/// `CAP_PROP_XI_CC_MATRIX_31` — Color Correction Matrix element (3, 1).
pub const CAP_PROP_XI_CC_MATRIX_31: i32 = 492;
/// `CAP_PROP_XI_CC_MATRIX_32` — Color Correction Matrix element (3, 2).
pub const CAP_PROP_XI_CC_MATRIX_32: i32 = 493;
/// `CAP_PROP_XI_CC_MATRIX_33` — Color Correction Matrix element (3, 3).
pub const CAP_PROP_XI_CC_MATRIX_33: i32 = 494;
/// `CAP_PROP_XI_DEFAULT_CC_MATRIX` — Set default Color Correction Matrix.
pub const CAP_PROP_XI_DEFAULT_CC_MATRIX: i32 = 495;
/// `CAP_PROP_XI_TRG_SELECTOR` — Selects the type of trigger.
pub const CAP_PROP_XI_TRG_SELECTOR: i32 = 498;
/// `CAP_PROP_XI_ACQ_FRAME_BURST_COUNT` — Sets number of frames acquired by
///   burst. This burst is used only if trigger is set to FrameBurstStart.
pub const CAP_PROP_XI_ACQ_FRAME_BURST_COUNT: i32 = 499;
/// `CAP_PROP_XI_DEBOUNCE_EN` — Enable/Disable debounce to selected GPI.
pub const CAP_PROP_XI_DEBOUNCE_EN: i32 = 507;
/// `CAP_PROP_XI_DEBOUNCE_T0` — Debounce time (x * 10us).
pub const CAP_PROP_XI_DEBOUNCE_T0: i32 = 508;
/// `CAP_PROP_XI_DEBOUNCE_T1` — Debounce time (x * 10us).
pub const CAP_PROP_XI_DEBOUNCE_T1: i32 = 509;
/// `CAP_PROP_XI_DEBOUNCE_POL` — Debounce polarity (pol = 1 t0 - falling edge,
///   t1 - rising edge).
pub const CAP_PROP_XI_DEBOUNCE_POL: i32 = 510;
/// `CAP_PROP_XI_LENS_MODE` — Status of lens control interface. This shall be
///   set to XI_ON before any Lens operations.
pub const CAP_PROP_XI_LENS_MODE: i32 = 511;
/// `CAP_PROP_XI_LENS_APERTURE_VALUE` — Current lens aperture value in stops.
///   Examples: 2.8, 4, 5.6, 8, 11.
pub const CAP_PROP_XI_LENS_APERTURE_VALUE: i32 = 512;
/// `CAP_PROP_XI_LENS_FOCUS_MOVEMENT_VALUE` — Lens current focus movement
///   value to be used by XI_PRM_LENS_FOCUS_MOVE in motor steps.
pub const CAP_PROP_XI_LENS_FOCUS_MOVEMENT_VALUE: i32 = 513;
/// `CAP_PROP_XI_LENS_FOCUS_MOVE` — Moves lens focus motor by steps set in
///   XI_PRM_LENS_FOCUS_MOVEMENT_VALUE.
pub const CAP_PROP_XI_LENS_FOCUS_MOVE: i32 = 514;
/// `CAP_PROP_XI_LENS_FOCUS_DISTANCE` — Lens focus distance in cm.
pub const CAP_PROP_XI_LENS_FOCUS_DISTANCE: i32 = 515;
/// `CAP_PROP_XI_LENS_FOCAL_LENGTH` — Lens focal distance in mm.
pub const CAP_PROP_XI_LENS_FOCAL_LENGTH: i32 = 516;
/// `CAP_PROP_XI_LENS_FEATURE_SELECTOR` — Selects the current feature which is
///   accessible by XI_PRM_LENS_FEATURE.
pub const CAP_PROP_XI_LENS_FEATURE_SELECTOR: i32 = 517;
/// `CAP_PROP_XI_LENS_FEATURE` — Allows access to lens feature value currently
///   selected by XI_PRM_LENS_FEATURE_SELECTOR.
pub const CAP_PROP_XI_LENS_FEATURE: i32 = 518;
/// `CAP_PROP_XI_DEVICE_MODEL_ID` — Returns device model id.
pub const CAP_PROP_XI_DEVICE_MODEL_ID: i32 = 521;
/// `CAP_PROP_XI_DEVICE_SN` — Returns device serial number.
pub const CAP_PROP_XI_DEVICE_SN: i32 = 522;
/// `CAP_PROP_XI_IMAGE_DATA_FORMAT_RGB32_ALPHA` — The alpha channel of RGB32
///   output image format.
pub const CAP_PROP_XI_IMAGE_DATA_FORMAT_RGB32_ALPHA: i32 = 529;
/// `CAP_PROP_XI_IMAGE_PAYLOAD_SIZE` — Buffer size in bytes sufficient for
///   output image returned by xiGetImage.
pub const CAP_PROP_XI_IMAGE_PAYLOAD_SIZE: i32 = 530;
/// `CAP_PROP_XI_TRANSPORT_PIXEL_FORMAT` — Current format of pixels on
///   transport layer.
pub const CAP_PROP_XI_TRANSPORT_PIXEL_FORMAT: i32 = 531;
/// `CAP_PROP_XI_SENSOR_CLOCK_FREQ_HZ` — Sensor clock frequency in Hz.
pub const CAP_PROP_XI_SENSOR_CLOCK_FREQ_HZ: i32 = 532;
/// `CAP_PROP_XI_SENSOR_CLOCK_FREQ_INDEX` — Sensor clock frequency index.
///   Sensor with selected frequencies have possibility to set the frequency
///   only by this index.
pub const CAP_PROP_XI_SENSOR_CLOCK_FREQ_INDEX: i32 = 533;
/// `CAP_PROP_XI_SENSOR_OUTPUT_CHANNEL_COUNT` — Number of output channels from
///   sensor used for data transfer.
pub const CAP_PROP_XI_SENSOR_OUTPUT_CHANNEL_COUNT: i32 = 534;
/// `CAP_PROP_XI_FRAMERATE` — Define framerate in Hz.
pub const CAP_PROP_XI_FRAMERATE: i32 = 535;
/// `CAP_PROP_XI_COUNTER_SELECTOR` — Select counter.
pub const CAP_PROP_XI_COUNTER_SELECTOR: i32 = 536;
/// `CAP_PROP_XI_COUNTER_VALUE` — Counter status.
pub const CAP_PROP_XI_COUNTER_VALUE: i32 = 537;
/// `CAP_PROP_XI_ACQ_TIMING_MODE` — Type of sensor frames timing.
pub const CAP_PROP_XI_ACQ_TIMING_MODE: i32 = 538;
/// `CAP_PROP_XI_AVAILABLE_BANDWIDTH` — Calculate and returns available
///   interface bandwidth(int Megabits).
pub const CAP_PROP_XI_AVAILABLE_BANDWIDTH: i32 = 539;
/// `CAP_PROP_XI_BUFFER_POLICY` — Data move policy.
pub const CAP_PROP_XI_BUFFER_POLICY: i32 = 540;
/// `CAP_PROP_XI_LUT_EN` — Activates LUT.
pub const CAP_PROP_XI_LUT_EN: i32 = 541;
/// `CAP_PROP_XI_LUT_INDEX` — Control the index (offset) of the coefficient to
///   access in the LUT.
pub const CAP_PROP_XI_LUT_INDEX: i32 = 542;
/// `CAP_PROP_XI_LUT_VALUE` — Value at entry LUTIndex of the LUT.
pub const CAP_PROP_XI_LUT_VALUE: i32 = 543;
/// `CAP_PROP_XI_TRG_DELAY` — Specifies the delay in microseconds (us) to
///   apply after the trigger reception before activating it.
pub const CAP_PROP_XI_TRG_DELAY: i32 = 544;
/// `CAP_PROP_XI_TS_RST_MODE` — Defines how time stamp reset engine will be
///   armed.
pub const CAP_PROP_XI_TS_RST_MODE: i32 = 545;
/// `CAP_PROP_XI_TS_RST_SOURCE` — Defines which source will be used for
///   timestamp reset. Writing this parameter will trigger settings of engine
///   (arming).
pub const CAP_PROP_XI_TS_RST_SOURCE: i32 = 546;
/// `CAP_PROP_XI_IS_DEVICE_EXIST` — Returns 1 if camera connected and works
///   properly.
pub const CAP_PROP_XI_IS_DEVICE_EXIST: i32 = 547;
/// `CAP_PROP_XI_ACQ_BUFFER_SIZE` — Acquisition buffer size in
///   buffer_size_unit. Default bytes.
pub const CAP_PROP_XI_ACQ_BUFFER_SIZE: i32 = 548;
/// `CAP_PROP_XI_ACQ_BUFFER_SIZE_UNIT` — Acquisition buffer size unit in
///   bytes. Default 1. E.g. Value 1024 means that buffer_size is in KiBytes.
pub const CAP_PROP_XI_ACQ_BUFFER_SIZE_UNIT: i32 = 549;
/// `CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_SIZE` — Acquisition transport buffer
///   size in bytes.
pub const CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_SIZE: i32 = 550;
/// `CAP_PROP_XI_BUFFERS_QUEUE_SIZE` — Queue of field/frame buffers.
pub const CAP_PROP_XI_BUFFERS_QUEUE_SIZE: i32 = 551;
/// `CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_COMMIT` — Number of buffers to commit to
///   low level.
pub const CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_COMMIT: i32 = 552;
/// `CAP_PROP_XI_RECENT_FRAME` — GetImage returns most recent frame.
pub const CAP_PROP_XI_RECENT_FRAME: i32 = 553;
/// `CAP_PROP_XI_DEVICE_RESET` — Resets the camera to default state.
pub const CAP_PROP_XI_DEVICE_RESET: i32 = 554;
/// `CAP_PROP_XI_COLUMN_FPN_CORRECTION` — Correction of column FPN.
pub const CAP_PROP_XI_COLUMN_FPN_CORRECTION: i32 = 555;
/// `CAP_PROP_XI_ROW_FPN_CORRECTION` — Correction of row FPN.
pub const CAP_PROP_XI_ROW_FPN_CORRECTION: i32 = 591;
/// `CAP_PROP_XI_SENSOR_MODE` — Current sensor mode. Allows to select sensor
///   mode by one integer. Setting of this parameter affects: image dimensions
///   and downsampling.
pub const CAP_PROP_XI_SENSOR_MODE: i32 = 558;
/// `CAP_PROP_XI_HDR` — Enable High Dynamic Range feature.
pub const CAP_PROP_XI_HDR: i32 = 559;
/// `CAP_PROP_XI_HDR_KNEEPOINT_COUNT` — The number of kneepoints in the PWLR.
pub const CAP_PROP_XI_HDR_KNEEPOINT_COUNT: i32 = 560;
/// `CAP_PROP_XI_HDR_T1` — Position of first kneepoint(in % of
///   XI_PRM_EXPOSURE).
pub const CAP_PROP_XI_HDR_T1: i32 = 561;
/// `CAP_PROP_XI_HDR_T2` — Position of second kneepoint (in % of
///   XI_PRM_EXPOSURE).
pub const CAP_PROP_XI_HDR_T2: i32 = 562;
/// `CAP_PROP_XI_KNEEPOINT1` — Value of first kneepoint (% of sensor
///   saturation).
pub const CAP_PROP_XI_KNEEPOINT1: i32 = 563;
/// `CAP_PROP_XI_KNEEPOINT2` — Value of second kneepoint (% of sensor
///   saturation).
pub const CAP_PROP_XI_KNEEPOINT2: i32 = 564;
/// `CAP_PROP_XI_IMAGE_BLACK_LEVEL` — Last image black level counts. Can be
///   used for Offline processing to recall it.
pub const CAP_PROP_XI_IMAGE_BLACK_LEVEL: i32 = 565;
/// `CAP_PROP_XI_HW_REVISION` — Returns hardware revision number.
pub const CAP_PROP_XI_HW_REVISION: i32 = 571;
/// `CAP_PROP_XI_DEBUG_LEVEL` — Set debug level.
pub const CAP_PROP_XI_DEBUG_LEVEL: i32 = 572;
/// `CAP_PROP_XI_AUTO_BANDWIDTH_CALCULATION` — Automatic bandwidth
///   calculation.
pub const CAP_PROP_XI_AUTO_BANDWIDTH_CALCULATION: i32 = 573;
/// `CAP_PROP_XI_FFS_FILE_ID` — File number.
pub const CAP_PROP_XI_FFS_FILE_ID: i32 = 594;
/// `CAP_PROP_XI_FFS_FILE_SIZE` — Size of file.
pub const CAP_PROP_XI_FFS_FILE_SIZE: i32 = 580;
/// `CAP_PROP_XI_FREE_FFS_SIZE` — Size of free camera FFS.
pub const CAP_PROP_XI_FREE_FFS_SIZE: i32 = 581;
/// `CAP_PROP_XI_USED_FFS_SIZE` — Size of used camera FFS.
pub const CAP_PROP_XI_USED_FFS_SIZE: i32 = 582;
/// `CAP_PROP_XI_FFS_ACCESS_KEY` — Setting of key enables file operations on
///   some cameras.
pub const CAP_PROP_XI_FFS_ACCESS_KEY: i32 = 583;
/// `CAP_PROP_XI_SENSOR_FEATURE_SELECTOR` — Selects the current feature which
///   is accessible by XI_PRM_SENSOR_FEATURE_VALUE.
pub const CAP_PROP_XI_SENSOR_FEATURE_SELECTOR: i32 = 585;
/// `CAP_PROP_XI_SENSOR_FEATURE_VALUE` — Allows access to sensor feature value
///   currently selected by XI_PRM_SENSOR_FEATURE_SELECTOR.
pub const CAP_PROP_XI_SENSOR_FEATURE_VALUE: i32 = 586;
/// `CAP_PROP_ARAVIS_AUTOTRIGGER` — Automatically trigger frame capture if
///   camera is configured with software trigger.
pub const CAP_PROP_ARAVIS_AUTOTRIGGER: i32 = 600;
/// `CAP_PROP_ANDROID_DEVICE_TORCH`.
pub const CAP_PROP_ANDROID_DEVICE_TORCH: i32 = 8001;
/// `CAP_PROP_IOS_DEVICE_FOCUS`.
pub const CAP_PROP_IOS_DEVICE_FOCUS: i32 = 9001;
/// `CAP_PROP_IOS_DEVICE_EXPOSURE`.
pub const CAP_PROP_IOS_DEVICE_EXPOSURE: i32 = 9002;
/// `CAP_PROP_IOS_DEVICE_FLASH`.
pub const CAP_PROP_IOS_DEVICE_FLASH: i32 = 9003;
/// `CAP_PROP_IOS_DEVICE_WHITEBALANCE`.
pub const CAP_PROP_IOS_DEVICE_WHITEBALANCE: i32 = 9004;
/// `CAP_PROP_IOS_DEVICE_TORCH`.
pub const CAP_PROP_IOS_DEVICE_TORCH: i32 = 9005;
/// `CAP_PROP_GIGA_FRAME_OFFSET_X`.
pub const CAP_PROP_GIGA_FRAME_OFFSET_X: i32 = 10001;
/// `CAP_PROP_GIGA_FRAME_OFFSET_Y`.
pub const CAP_PROP_GIGA_FRAME_OFFSET_Y: i32 = 10002;
/// `CAP_PROP_GIGA_FRAME_WIDTH_MAX`.
pub const CAP_PROP_GIGA_FRAME_WIDTH_MAX: i32 = 10003;
/// `CAP_PROP_GIGA_FRAME_HEIGHT_MAX`.
pub const CAP_PROP_GIGA_FRAME_HEIGHT_MAX: i32 = 10004;
/// `CAP_PROP_GIGA_FRAME_HEIGH_MAX`.
pub const CAP_PROP_GIGA_FRAME_HEIGH_MAX: i32 = 10004;
/// `CAP_PROP_GIGA_FRAME_SENS_WIDTH`.
pub const CAP_PROP_GIGA_FRAME_SENS_WIDTH: i32 = 10005;
/// `CAP_PROP_GIGA_FRAME_SENS_HEIGHT`.
pub const CAP_PROP_GIGA_FRAME_SENS_HEIGHT: i32 = 10006;
/// `CAP_PROP_GIGA_FRAME_SENS_HEIGH`.
pub const CAP_PROP_GIGA_FRAME_SENS_HEIGH: i32 = 10006;
/// `CAP_PROP_INTELPERC_PROFILE_COUNT`.
pub const CAP_PROP_INTELPERC_PROFILE_COUNT: i32 = 11001;
/// `CAP_PROP_INTELPERC_PROFILE_IDX`.
pub const CAP_PROP_INTELPERC_PROFILE_IDX: i32 = 11002;
/// `CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE`.
pub const CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE: i32 = 11003;
/// `CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE`.
pub const CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE: i32 = 11004;
/// `CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD`.
pub const CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD: i32 = 11005;
/// `CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ`.
pub const CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ: i32 = 11006;
/// `CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT`.
pub const CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT: i32 = 11007;
/// `CAP_INTELPERC_DEPTH_GENERATOR`.
pub const CAP_INTELPERC_DEPTH_GENERATOR: i32 = 536870912;
/// `CAP_INTELPERC_IMAGE_GENERATOR`.
pub const CAP_INTELPERC_IMAGE_GENERATOR: i32 = 268435456;
/// `CAP_INTELPERC_IR_GENERATOR`.
pub const CAP_INTELPERC_IR_GENERATOR: i32 = 134217728;
/// `CAP_INTELPERC_GENERATORS_MASK`.
pub const CAP_INTELPERC_GENERATORS_MASK: i32 = 939524096;
/// `CAP_INTELPERC_DEPTH_MAP` — Each pixel is a 16-bit integer. The value
///   indicates the distance from an object to the camera's XY plane or the
///   Cartesian depth.
pub const CAP_INTELPERC_DEPTH_MAP: i32 = 0;
/// `CAP_INTELPERC_UVDEPTH_MAP` — Each pixel contains two 32-bit floating
///   point values in the range of 0-1, representing the mapping of depth
///   coordinates to the color coordinates.
pub const CAP_INTELPERC_UVDEPTH_MAP: i32 = 1;
/// `CAP_INTELPERC_IR_MAP` — Each pixel is a 16-bit integer. The value
///   indicates the intensity of the reflected laser beam.
pub const CAP_INTELPERC_IR_MAP: i32 = 2;
/// `CAP_INTELPERC_IMAGE`.
pub const CAP_INTELPERC_IMAGE: i32 = 3;
/// `CAP_PROP_GPHOTO2_PREVIEW` — Capture only preview from liveview mode.
pub const CAP_PROP_GPHOTO2_PREVIEW: i32 = 17001;
/// `CAP_PROP_GPHOTO2_WIDGET_ENUMERATE` — Readonly, returns (const char *).
pub const CAP_PROP_GPHOTO2_WIDGET_ENUMERATE: i32 = 17002;
/// `CAP_PROP_GPHOTO2_RELOAD_CONFIG` — Trigger, only by set. Reload camera
///   settings.
pub const CAP_PROP_GPHOTO2_RELOAD_CONFIG: i32 = 17003;
/// `CAP_PROP_GPHOTO2_RELOAD_ON_CHANGE` — Reload all settings on set.
pub const CAP_PROP_GPHOTO2_RELOAD_ON_CHANGE: i32 = 17004;
/// `CAP_PROP_GPHOTO2_COLLECT_MSGS` — Collect messages with details.
pub const CAP_PROP_GPHOTO2_COLLECT_MSGS: i32 = 17005;
/// `CAP_PROP_GPHOTO2_FLUSH_MSGS` — Readonly, returns (const char *).
pub const CAP_PROP_GPHOTO2_FLUSH_MSGS: i32 = 17006;
/// `CAP_PROP_SPEED` — Exposure speed. Can be readonly, depends on camera
///   program.
pub const CAP_PROP_SPEED: i32 = 17007;
/// `CAP_PROP_APERTURE` — Aperture. Can be readonly, depends on camera
///   program.
pub const CAP_PROP_APERTURE: i32 = 17008;
/// `CAP_PROP_EXPOSUREPROGRAM` — Camera exposure program.
pub const CAP_PROP_EXPOSUREPROGRAM: i32 = 17009;
/// `CAP_PROP_VIEWFINDER` — Enter liveview mode.
pub const CAP_PROP_VIEWFINDER: i32 = 17010;
/// `CAP_PROP_IMAGES_BASE`.
pub const CAP_PROP_IMAGES_BASE: i32 = 18000;
/// `CAP_PROP_IMAGES_LAST` — excluding.
pub const CAP_PROP_IMAGES_LAST: i32 = 19000;
/// `CAP_OBSENSOR_DEPTH_MAP` — Depth values in mm (CV_16UC1).
pub const CAP_OBSENSOR_DEPTH_MAP: i32 = 0;
/// `CAP_OBSENSOR_BGR_IMAGE` — Data given from BGR stream generator.
pub const CAP_OBSENSOR_BGR_IMAGE: i32 = 1;
/// `CAP_OBSENSOR_IR_IMAGE` — Data given from IR stream generator(CV_16UC1).
pub const CAP_OBSENSOR_IR_IMAGE: i32 = 2;
/// `CAP_OBSENSOR_DEPTH_GENERATOR`.
pub const CAP_OBSENSOR_DEPTH_GENERATOR: i32 = 536870912;
/// `CAP_OBSENSOR_IMAGE_GENERATOR`.
pub const CAP_OBSENSOR_IMAGE_GENERATOR: i32 = 268435456;
/// `CAP_OBSENSOR_IR_GENERATOR`.
pub const CAP_OBSENSOR_IR_GENERATOR: i32 = 134217728;
/// `CAP_OBSENSOR_GENERATORS_MASK`.
pub const CAP_OBSENSOR_GENERATORS_MASK: i32 = 939524096;
/// `CAP_PROP_OBSENSOR_INTRINSIC_FX`.
pub const CAP_PROP_OBSENSOR_INTRINSIC_FX: i32 = 26001;
/// `CAP_PROP_OBSENSOR_INTRINSIC_FY`.
pub const CAP_PROP_OBSENSOR_INTRINSIC_FY: i32 = 26002;
/// `CAP_PROP_OBSENSOR_INTRINSIC_CX`.
pub const CAP_PROP_OBSENSOR_INTRINSIC_CX: i32 = 26003;
/// `CAP_PROP_OBSENSOR_INTRINSIC_CY`.
pub const CAP_PROP_OBSENSOR_INTRINSIC_CY: i32 = 26004;
/// `CAP_PROP_OBSENSOR_RGB_POS_MSEC`.
pub const CAP_PROP_OBSENSOR_RGB_POS_MSEC: i32 = 26005;
/// `CAP_PROP_OBSENSOR_DEPTH_POS_MSEC`.
pub const CAP_PROP_OBSENSOR_DEPTH_POS_MSEC: i32 = 26006;
/// `CAP_PROP_OBSENSOR_DEPTH_WIDTH`.
pub const CAP_PROP_OBSENSOR_DEPTH_WIDTH: i32 = 26007;
/// `CAP_PROP_OBSENSOR_DEPTH_HEIGHT`.
pub const CAP_PROP_OBSENSOR_DEPTH_HEIGHT: i32 = 26008;
/// `CAP_PROP_OBSENSOR_DEPTH_FPS`.
pub const CAP_PROP_OBSENSOR_DEPTH_FPS: i32 = 26009;
/// `CAP_PROP_OBSENSOR_COLOR_DISTORTION_K1`.
pub const CAP_PROP_OBSENSOR_COLOR_DISTORTION_K1: i32 = 26010;
/// `CAP_PROP_OBSENSOR_COLOR_DISTORTION_K2`.
pub const CAP_PROP_OBSENSOR_COLOR_DISTORTION_K2: i32 = 26011;
/// `CAP_PROP_OBSENSOR_COLOR_DISTORTION_K3`.
pub const CAP_PROP_OBSENSOR_COLOR_DISTORTION_K3: i32 = 26012;
/// `CAP_PROP_OBSENSOR_COLOR_DISTORTION_K4`.
pub const CAP_PROP_OBSENSOR_COLOR_DISTORTION_K4: i32 = 26013;
/// `CAP_PROP_OBSENSOR_COLOR_DISTORTION_K5`.
pub const CAP_PROP_OBSENSOR_COLOR_DISTORTION_K5: i32 = 26014;
/// `CAP_PROP_OBSENSOR_COLOR_DISTORTION_K6`.
pub const CAP_PROP_OBSENSOR_COLOR_DISTORTION_K6: i32 = 26015;
/// `CAP_PROP_OBSENSOR_COLOR_DISTORTION_P1`.
pub const CAP_PROP_OBSENSOR_COLOR_DISTORTION_P1: i32 = 26016;
/// `CAP_PROP_OBSENSOR_COLOR_DISTORTION_P2`.
pub const CAP_PROP_OBSENSOR_COLOR_DISTORTION_P2: i32 = 26017;

#[cfg(test)]
mod tests {
    use super::*;

    /// Spot checks against `cv2`, spread across the value ranges: the base
    /// properties, a backend id, a vendor block, and the wrapped flags.
    #[test]
    fn the_values_are_the_ones_cv2_reports() {
        assert_eq!(CAP_PROP_POS_MSEC, 0);
        assert_eq!(CAP_PROP_FRAME_WIDTH, 3);
        assert_eq!(CAP_PROP_FRAME_HEIGHT, 4);
        assert_eq!(CAP_PROP_FPS, 5);
        assert_eq!(CAP_PROP_FOURCC, 6);
        assert_eq!(CAP_PROP_FRAME_COUNT, 7);
        assert_eq!(CAP_ANY, 0);
        assert_eq!(CAP_FFMPEG, 1900);
        assert_eq!(CAP_IMAGES, 2000);
        assert_eq!(VIDEOWRITER_PROP_QUALITY, 1);
        assert_eq!(CAP_PROP_XI_TIMEOUT, 420);
        assert_eq!(CAP_PROP_HW_ACCELERATION, 50);
    }

    #[test]
    fn the_generator_flags_wrap_the_way_cv2_reports_them() {
        // `1 << 31` in a C++ int enum; cv2 gives the negative value, and a
        // mask test against a property depends on getting the same one.
        assert_eq!(CAP_OPENNI_DEPTH_GENERATOR, i32::MIN);
        assert_eq!(CAP_OPENNI_IMAGE_GENERATOR, 1 << 30);
        assert_eq!(CAP_INTELPERC_DEPTH_GENERATOR, 1 << 29);
        // And the derived "present" flags keep their base plus offset.
        assert_eq!(CAP_OPENNI_IMAGE_GENERATOR_PRESENT, 1_073_741_933);
    }

    #[test]
    fn aliases_share_their_targets_value() {
        // Upstream keeps several spellings of the same backend.
        assert_eq!(CAP_V4L, CAP_V4L2);
        assert_eq!(CAP_FIREWIRE, CAP_IEEE1394);
        assert_eq!(CAP_FIREWIRE, CAP_DC1394);
        assert_eq!(CAP_FIREWIRE, CAP_CMU1394);
    }
}
