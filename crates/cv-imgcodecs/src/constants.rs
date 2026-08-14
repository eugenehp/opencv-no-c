//! `cv::imgcodecs` write-parameter and read-mode constants
//! (`modules/imgcodecs/include/opencv2/imgcodecs.hpp`).
//!
//! The `IMWRITE_*` values name the knobs [`crate::imencode`] and
//! [`crate::imwrite`] accept as `(key, value)` pairs, and the `IMREAD_*` values
//! the decode modes. Generated from the upstream header, so every number is
//! the one `cv2` reports.
//!
//! # Which of these do something here
//!
//! The read modes and the parameters of the formats this crate encodes —
//! PNG's compression level and strategy, JPEG's quality, PNM's binary flag —
//! are honoured. The rest name knobs of codecs that need a C library
//! (WebP, JPEG 2000, EXR, AVIF, TIFF, GIF, HDR, JPEG XL) and are declared so
//! that a parameter list translated from C++ keeps its meaning; passing one
//! gives a clear error rather than being silently dropped.
//!
//! [`crate::modes`] holds the read modes as a typed enum, which is what new
//! code should reach for; these are the integers for interoperating.

/// `IMWRITE_JPEG_PROGRESSIVE` — Enable JPEG features, 0 or 1, default is
///   False.
pub const IMWRITE_JPEG_PROGRESSIVE: i32 = 2;
/// `IMWRITE_JPEG_OPTIMIZE` — Enable JPEG features, 0 or 1, default is False.
pub const IMWRITE_JPEG_OPTIMIZE: i32 = 3;
/// `IMWRITE_JPEG_RST_INTERVAL` — JPEG restart interval, 0 - 65535, default is
///   0 - no restart.
pub const IMWRITE_JPEG_RST_INTERVAL: i32 = 4;
/// `IMWRITE_JPEG_LUMA_QUALITY` — Separate luma quality level, 0 - 100,
///   default is -1 - don't use. If JPEG_LIB_VERSION < 70, Not supported.
pub const IMWRITE_JPEG_LUMA_QUALITY: i32 = 5;
/// `IMWRITE_JPEG_CHROMA_QUALITY` — Separate chroma quality level, 0 - 100,
///   default is -1 - don't use. If JPEG_LIB_VERSION < 70, Not supported.
pub const IMWRITE_JPEG_CHROMA_QUALITY: i32 = 6;
/// `IMWRITE_JPEG_SAMPLING_FACTOR` — For JPEG, set sampling factor. See
///   cv::ImwriteJPEGSamplingFactorParams.
pub const IMWRITE_JPEG_SAMPLING_FACTOR: i32 = 7;
/// `IMWRITE_PNG_STRATEGY` — For PNG, One of cv::ImwritePNGFlags, default is
///   IMWRITE_PNG_STRATEGY_RLE.
pub const IMWRITE_PNG_STRATEGY: i32 = 17;
/// `IMWRITE_PNG_BILEVEL` — For PNG, Binary level PNG, 0 or 1, default is 0.
///   For APNG, it is not supported.
pub const IMWRITE_PNG_BILEVEL: i32 = 18;
/// `IMWRITE_PNG_FILTER` — For PNG, One of cv::ImwritePNGFilterFlags, default
///   is IMWRITE_PNG_FILTER_SUB. For APNG, it is not supported.
pub const IMWRITE_PNG_FILTER: i32 = 19;
/// `IMWRITE_PNG_ZLIBBUFFER_SIZE` — For PNG with libpng, sets the size of the
///   internal zlib compression buffer in bytes, from 6 to 1048576(1024 KiB).
///   Default is 8192(8 KiB). For normal use, 131072(128 KiB) or 262144(256 KiB)
///   may be sufficient. If WITH_SPNG=ON, it is not supported. For APNG, it is
///   not supported.
pub const IMWRITE_PNG_ZLIBBUFFER_SIZE: i32 = 20;
/// `IMWRITE_EXR_TYPE` — override EXR storage type (FLOAT (FP32) is default).
pub const IMWRITE_EXR_TYPE: i32 = 48;
/// `IMWRITE_EXR_COMPRESSION` — override EXR compression type (ZIP_COMPRESSION
///   = 3 is default).
pub const IMWRITE_EXR_COMPRESSION: i32 = 49;
/// `IMWRITE_EXR_DWA_COMPRESSION_LEVEL` — override EXR DWA compression level
///   (45 is default).
pub const IMWRITE_EXR_DWA_COMPRESSION_LEVEL: i32 = 50;
/// `IMWRITE_WEBP_QUALITY` — For WEBP, it can be a lossy quality from 1 to 100
///   (the higher is the better) for IMWRITE_WEBP_LOSSLESS_OFF. By default
///   (without this parameter) or if quality > 100, IMWRITE_WEBP_LOSSLESS_ON is
///   used instead.
pub const IMWRITE_WEBP_QUALITY: i32 = 64;
/// `IMWRITE_WEBP_LOSSLESS_MODE` — For WEBP, it can be a lossless compression
///   strategy. See cv::ImwriteWEBPLosslessMode. Default is
///   IMWRITE_WEBP_LOSSLESS_OFF. For Animated WEBP, it is not supported.
pub const IMWRITE_WEBP_LOSSLESS_MODE: i32 = 65;
/// `IMWRITE_HDR_COMPRESSION` — specify HDR compression.
pub const IMWRITE_HDR_COMPRESSION: i32 = 80;
/// `IMWRITE_PAM_TUPLETYPE` — For PAM, sets the TUPLETYPE field to the
///   corresponding string value that is defined for the format.
pub const IMWRITE_PAM_TUPLETYPE: i32 = 128;
/// `IMWRITE_TIFF_RESUNIT` — For TIFF, use to specify which DPI resolution
///   unit to set. See ImwriteTiffResolutionUnitFlags. Default is
///   IMWRITE_TIFF_RESOLUTION_UNIT_INCH.
pub const IMWRITE_TIFF_RESUNIT: i32 = 256;
/// `IMWRITE_TIFF_XDPI` — For TIFF, use to specify the X direction DPI.
pub const IMWRITE_TIFF_XDPI: i32 = 257;
/// `IMWRITE_TIFF_YDPI` — For TIFF, use to specify the Y direction DPI.
pub const IMWRITE_TIFF_YDPI: i32 = 258;
/// `IMWRITE_TIFF_COMPRESSION` — For TIFF, use to specify the image
///   compression scheme. See cv::ImwriteTiffCompressionFlags. The compression
///   scheme can be specified by this flag; the default is LZW compression,
///   except for 32F depth where it is NONE.
pub const IMWRITE_TIFF_COMPRESSION: i32 = 259;
/// `IMWRITE_TIFF_ROWSPERSTRIP` — For TIFF, use to specify the number of rows
///   per strip.
pub const IMWRITE_TIFF_ROWSPERSTRIP: i32 = 278;
/// `IMWRITE_TIFF_PREDICTOR` — For TIFF, use to specify predictor. See
///   cv::ImwriteTiffPredictorFlags. Default is
///   IMWRITE_TIFF_PREDICTOR_HORIZONTAL .
pub const IMWRITE_TIFF_PREDICTOR: i32 = 317;
/// `IMWRITE_JPEG2000_COMPRESSION_X1000` — For JPEG2000, use to specify the
///   target compression rate (multiplied by 1000). The value can be from 0 to
///   1000. Default is 1000.
pub const IMWRITE_JPEG2000_COMPRESSION_X1000: i32 = 272;
/// `IMWRITE_AVIF_QUALITY` — For AVIF, it can be a quality between 0 and 100
///   (the higher the better). Default is 95.
pub const IMWRITE_AVIF_QUALITY: i32 = 512;
/// `IMWRITE_AVIF_DEPTH` — For AVIF, it can be 8, 10 or 12. If >8, it is
///   stored/read as CV_16U. Default is 8.
pub const IMWRITE_AVIF_DEPTH: i32 = 513;
/// `IMWRITE_AVIF_SPEED` — For AVIF, it is between 0 (slowest) and
///   10(fastest). Default is 9.
pub const IMWRITE_AVIF_SPEED: i32 = 514;
/// `IMWRITE_JPEGXL_QUALITY` — For JPEG XL, it can be a quality from 0 to 100
///   (the higher is the better). Default value is 95. If set, distance
///   parameter is re-calicurated from quality level automatically. This
///   parameter request libjxl v0.10 or later.
pub const IMWRITE_JPEGXL_QUALITY: i32 = 640;
/// `IMWRITE_JPEGXL_EFFORT` — For JPEG XL, encoder effort/speed level without
///   affecting decoding speed; it is between 1 (fastest) and 10 (slowest).
///   Default is 7.
pub const IMWRITE_JPEGXL_EFFORT: i32 = 641;
/// `IMWRITE_JPEGXL_DISTANCE` — For JPEG XL, distance level for lossy
///   compression: target max butteraugli distance, lower = higher quality, 0 =
///   lossless; range: 0 .. 25. Default is 1.
pub const IMWRITE_JPEGXL_DISTANCE: i32 = 642;
/// `IMWRITE_JPEGXL_DECODING_SPEED` — For JPEG XL, decoding speed tier for the
///   provided options; minimum is 0 (slowest to decode, best quality/density),
///   and maximum is 4 (fastest to decode, at the cost of some quality/density).
///   Default is 0.
pub const IMWRITE_JPEGXL_DECODING_SPEED: i32 = 643;
/// `IMWRITE_GIF_LOOP` — Not functional since 4.12.0. Replaced by
///   cv::Animation::loop_count.
pub const IMWRITE_GIF_LOOP: i32 = 1024;
/// `IMWRITE_GIF_SPEED` — Not functional since 4.12.0. Replaced by
///   cv::Animation::durations.
pub const IMWRITE_GIF_SPEED: i32 = 1025;
/// `IMWRITE_GIF_QUALITY` — For GIF, it can be a quality from 1 to 8. Default
///   is 2. See cv::ImwriteGifCompressionFlags.
pub const IMWRITE_GIF_QUALITY: i32 = 1026;
/// `IMWRITE_GIF_DITHER` — For GIF, it can be a quality from -1(most dither)
///   to 3(no dither). Default is 0.
pub const IMWRITE_GIF_DITHER: i32 = 1027;
/// `IMWRITE_GIF_TRANSPARENCY` — For GIF, the alpha channel lower than this
///   will be set to transparent. Default is 1.
pub const IMWRITE_GIF_TRANSPARENCY: i32 = 1028;
/// `IMWRITE_GIF_COLORTABLE` — For GIF, 0 means global color table is used, 1
///   means local color table is used. Default is 0.
pub const IMWRITE_GIF_COLORTABLE: i32 = 1029;
/// `IMWRITE_JPEG_SAMPLING_FACTOR_411` — 4x1,1x1,1x1.
pub const IMWRITE_JPEG_SAMPLING_FACTOR_411: i32 = 4264209;
/// `IMWRITE_JPEG_SAMPLING_FACTOR_420` — 2x2,1x1,1x1(Default).
pub const IMWRITE_JPEG_SAMPLING_FACTOR_420: i32 = 2232593;
/// `IMWRITE_JPEG_SAMPLING_FACTOR_422` — 2x1,1x1,1x1.
pub const IMWRITE_JPEG_SAMPLING_FACTOR_422: i32 = 2167057;
/// `IMWRITE_JPEG_SAMPLING_FACTOR_440` — 1x2,1x1,1x1.
pub const IMWRITE_JPEG_SAMPLING_FACTOR_440: i32 = 1184017;
/// `IMWRITE_JPEG_SAMPLING_FACTOR_444` — 1x1,1x1,1x1(No subsampling).
pub const IMWRITE_JPEG_SAMPLING_FACTOR_444: i32 = 1118481;
/// `IMWRITE_TIFF_COMPRESSION_NONE` — dump mode.
pub const IMWRITE_TIFF_COMPRESSION_NONE: i32 = 1;
/// `IMWRITE_TIFF_COMPRESSION_CCITTRLE` — CCITT modified Huffman RLE.
pub const IMWRITE_TIFF_COMPRESSION_CCITTRLE: i32 = 2;
/// `IMWRITE_TIFF_COMPRESSION_CCITTFAX3` — CCITT Group 3 fax encoding.
pub const IMWRITE_TIFF_COMPRESSION_CCITTFAX3: i32 = 3;
/// `IMWRITE_TIFF_COMPRESSION_CCITT_T4` — CCITT T.4 (TIFF 6 name).
pub const IMWRITE_TIFF_COMPRESSION_CCITT_T4: i32 = 3;
/// `IMWRITE_TIFF_COMPRESSION_CCITTFAX4` — CCITT Group 4 fax encoding.
pub const IMWRITE_TIFF_COMPRESSION_CCITTFAX4: i32 = 4;
/// `IMWRITE_TIFF_COMPRESSION_CCITT_T6` — CCITT T.6 (TIFF 6 name).
pub const IMWRITE_TIFF_COMPRESSION_CCITT_T6: i32 = 4;
/// `IMWRITE_TIFF_COMPRESSION_LZW` — Lempel-Ziv & Welch.
pub const IMWRITE_TIFF_COMPRESSION_LZW: i32 = 5;
/// `IMWRITE_TIFF_COMPRESSION_OJPEG` — !6.0 JPEG.
pub const IMWRITE_TIFF_COMPRESSION_OJPEG: i32 = 6;
/// `IMWRITE_TIFF_COMPRESSION_JPEG` — %JPEG DCT compression.
pub const IMWRITE_TIFF_COMPRESSION_JPEG: i32 = 7;
/// `IMWRITE_TIFF_COMPRESSION_T85` — !TIFF/FX T.85 JBIG compression.
pub const IMWRITE_TIFF_COMPRESSION_T85: i32 = 9;
/// `IMWRITE_TIFF_COMPRESSION_T43` — !TIFF/FX T.43 colour by layered JBIG
///   compression.
pub const IMWRITE_TIFF_COMPRESSION_T43: i32 = 10;
/// `IMWRITE_TIFF_COMPRESSION_NEXT` — NeXT 2-bit RLE.
pub const IMWRITE_TIFF_COMPRESSION_NEXT: i32 = 32766;
/// `IMWRITE_TIFF_COMPRESSION_CCITTRLEW` — 1 w/ word alignment.
pub const IMWRITE_TIFF_COMPRESSION_CCITTRLEW: i32 = 32771;
/// `IMWRITE_TIFF_COMPRESSION_PACKBITS` — Macintosh RLE.
pub const IMWRITE_TIFF_COMPRESSION_PACKBITS: i32 = 32773;
/// `IMWRITE_TIFF_COMPRESSION_THUNDERSCAN` — ThunderScan RLE.
pub const IMWRITE_TIFF_COMPRESSION_THUNDERSCAN: i32 = 32809;
/// `IMWRITE_TIFF_COMPRESSION_IT8CTPAD` — IT8 CT w/padding.
pub const IMWRITE_TIFF_COMPRESSION_IT8CTPAD: i32 = 32895;
/// `IMWRITE_TIFF_COMPRESSION_IT8LW` — IT8 Linework RLE.
pub const IMWRITE_TIFF_COMPRESSION_IT8LW: i32 = 32896;
/// `IMWRITE_TIFF_COMPRESSION_IT8MP` — IT8 Monochrome picture.
pub const IMWRITE_TIFF_COMPRESSION_IT8MP: i32 = 32897;
/// `IMWRITE_TIFF_COMPRESSION_IT8BL` — IT8 Binary line art.
pub const IMWRITE_TIFF_COMPRESSION_IT8BL: i32 = 32898;
/// `IMWRITE_TIFF_COMPRESSION_PIXARFILM` — Pixar companded 10bit LZW.
pub const IMWRITE_TIFF_COMPRESSION_PIXARFILM: i32 = 32908;
/// `IMWRITE_TIFF_COMPRESSION_PIXARLOG` — Pixar companded 11bit ZIP.
pub const IMWRITE_TIFF_COMPRESSION_PIXARLOG: i32 = 32909;
/// `IMWRITE_TIFF_COMPRESSION_DEFLATE` — Deflate compression, legacy tag.
pub const IMWRITE_TIFF_COMPRESSION_DEFLATE: i32 = 32946;
/// `IMWRITE_TIFF_COMPRESSION_ADOBE_DEFLATE` — Deflate compression, as
///   recognized by Adobe.
pub const IMWRITE_TIFF_COMPRESSION_ADOBE_DEFLATE: i32 = 8;
/// `IMWRITE_TIFF_COMPRESSION_DCS` — Kodak DCS encoding.
pub const IMWRITE_TIFF_COMPRESSION_DCS: i32 = 32947;
/// `IMWRITE_TIFF_COMPRESSION_JBIG` — ISO JBIG.
pub const IMWRITE_TIFF_COMPRESSION_JBIG: i32 = 34661;
/// `IMWRITE_TIFF_COMPRESSION_SGILOG` — SGI Log Luminance RLE.
pub const IMWRITE_TIFF_COMPRESSION_SGILOG: i32 = 34676;
/// `IMWRITE_TIFF_COMPRESSION_SGILOG24` — SGI Log 24-bit packed.
pub const IMWRITE_TIFF_COMPRESSION_SGILOG24: i32 = 34677;
/// `IMWRITE_TIFF_COMPRESSION_JP2000` — Leadtools JPEG2000.
pub const IMWRITE_TIFF_COMPRESSION_JP2000: i32 = 34712;
/// `IMWRITE_TIFF_COMPRESSION_LERC` — ESRI Lerc codec:
///   <https://github.com/Esri/lerc>.
pub const IMWRITE_TIFF_COMPRESSION_LERC: i32 = 34887;
/// `IMWRITE_TIFF_COMPRESSION_LZMA` — LZMA2.
pub const IMWRITE_TIFF_COMPRESSION_LZMA: i32 = 34925;
/// `IMWRITE_TIFF_COMPRESSION_ZSTD` — ZSTD: WARNING not registered in Adobe-
///   maintained registry.
pub const IMWRITE_TIFF_COMPRESSION_ZSTD: i32 = 50000;
/// `IMWRITE_TIFF_COMPRESSION_WEBP` — WEBP: WARNING not registered in Adobe-
///   maintained registry.
pub const IMWRITE_TIFF_COMPRESSION_WEBP: i32 = 50001;
/// `IMWRITE_TIFF_COMPRESSION_JXL` — JPEGXL: WARNING not registered in Adobe-
///   maintained registry.
pub const IMWRITE_TIFF_COMPRESSION_JXL: i32 = 50002;
/// `IMWRITE_TIFF_PREDICTOR_NONE` — no prediction scheme used.
pub const IMWRITE_TIFF_PREDICTOR_NONE: i32 = 1;
/// `IMWRITE_TIFF_PREDICTOR_HORIZONTAL` — horizontal differencing.
pub const IMWRITE_TIFF_PREDICTOR_HORIZONTAL: i32 = 2;
/// `IMWRITE_TIFF_PREDICTOR_FLOATINGPOINT` — floating point predictor.
pub const IMWRITE_TIFF_PREDICTOR_FLOATINGPOINT: i32 = 3;
/// `IMWRITE_TIFF_RESOLUTION_UNIT_NONE` — no absolute unit of measurement.
pub const IMWRITE_TIFF_RESOLUTION_UNIT_NONE: i32 = 1;
/// `IMWRITE_TIFF_RESOLUTION_UNIT_INCH` — inch.
pub const IMWRITE_TIFF_RESOLUTION_UNIT_INCH: i32 = 2;
/// `IMWRITE_TIFF_RESOLUTION_UNIT_CENTIMETER` — centimeter.
pub const IMWRITE_TIFF_RESOLUTION_UNIT_CENTIMETER: i32 = 3;
/// `IMWRITE_EXR_TYPE_HALF` — store as HALF (FP16).
pub const IMWRITE_EXR_TYPE_HALF: i32 = 1;
/// `IMWRITE_EXR_TYPE_FLOAT` — store as FP32 (default).
pub const IMWRITE_EXR_TYPE_FLOAT: i32 = 2;
/// `IMWRITE_EXR_COMPRESSION_NO` — no compression.
pub const IMWRITE_EXR_COMPRESSION_NO: i32 = 0;
/// `IMWRITE_EXR_COMPRESSION_RLE` — run length encoding.
pub const IMWRITE_EXR_COMPRESSION_RLE: i32 = 1;
/// `IMWRITE_EXR_COMPRESSION_ZIPS` — zlib compression, one scan line at a
///   time.
pub const IMWRITE_EXR_COMPRESSION_ZIPS: i32 = 2;
/// `IMWRITE_EXR_COMPRESSION_ZIP` — zlib compression, in blocks of 16 scan
///   lines.
pub const IMWRITE_EXR_COMPRESSION_ZIP: i32 = 3;
/// `IMWRITE_EXR_COMPRESSION_PIZ` — piz-based wavelet compression.
pub const IMWRITE_EXR_COMPRESSION_PIZ: i32 = 4;
/// `IMWRITE_EXR_COMPRESSION_PXR24` — lossy 24-bit float compression.
pub const IMWRITE_EXR_COMPRESSION_PXR24: i32 = 5;
/// `IMWRITE_EXR_COMPRESSION_B44` — lossy 4-by-4 pixel block compression,
///   fixed compression rate.
pub const IMWRITE_EXR_COMPRESSION_B44: i32 = 6;
/// `IMWRITE_EXR_COMPRESSION_B44A` — lossy 4-by-4 pixel block compression,
///   flat fields are compressed more.
pub const IMWRITE_EXR_COMPRESSION_B44A: i32 = 7;
/// `IMWRITE_EXR_COMPRESSION_DWAA` — lossy DCT based compression, in blocks of
///   32 scanlines. More efficient for partial buffer access. Supported since
///   OpenEXR 2.2.0.
pub const IMWRITE_EXR_COMPRESSION_DWAA: i32 = 8;
/// `IMWRITE_EXR_COMPRESSION_DWAB` — lossy DCT based compression, in blocks of
///   256 scanlines. More efficient space wise and faster to decode full frames
///   than DWAA_COMPRESSION. Supported since OpenEXR 2.2.0.
pub const IMWRITE_EXR_COMPRESSION_DWAB: i32 = 9;
/// `IMWRITE_PNG_STRATEGY_DEFAULT` — Use this value for normal data.
pub const IMWRITE_PNG_STRATEGY_DEFAULT: i32 = 0;
/// `IMWRITE_PNG_STRATEGY_FILTERED` — Use this value for data produced by a
///   filter (or predictor).Filtered data consists mostly of small values with a
///   somewhat random distribution. In this case, the compression algorithm is
///   tuned to compress them better.
pub const IMWRITE_PNG_STRATEGY_FILTERED: i32 = 1;
/// `IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY` — Use this value to force Huffman
///   encoding only (no string match).
pub const IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY: i32 = 2;
/// `IMWRITE_PNG_STRATEGY_RLE` — Use this value to limit match distances to
///   one (run-length encoding).
pub const IMWRITE_PNG_STRATEGY_RLE: i32 = 3;
/// `IMWRITE_PNG_STRATEGY_FIXED` — Using this value prevents the use of
///   dynamic Huffman codes, allowing for a simpler decoder for special
///   applications.
pub const IMWRITE_PNG_STRATEGY_FIXED: i32 = 4;
/// `IMWRITE_PNG_FILTER_NONE` — Applies no filter to the PNG image (useful
///   when you want to save the raw pixel data without any compression filter).
pub const IMWRITE_PNG_FILTER_NONE: i32 = 8;
/// `IMWRITE_PNG_FILTER_SUB` — Applies the "sub" filter, which calculates the
///   difference between the current byte and the previous byte in the row.
pub const IMWRITE_PNG_FILTER_SUB: i32 = 16;
/// `IMWRITE_PNG_FILTER_UP` — applies the "up" filter, which calculates the
///   difference between the current byte and the corresponding byte directly
///   above it.
pub const IMWRITE_PNG_FILTER_UP: i32 = 32;
/// `IMWRITE_PNG_FILTER_AVG` — applies the "average" filter, which calculates
///   the average of the byte to the left and the byte above.
pub const IMWRITE_PNG_FILTER_AVG: i32 = 64;
/// `IMWRITE_PNG_FILTER_PAETH` — applies the "Paeth" filter, a more complex
///   filter that predicts the next pixel value based on neighboring pixels.
pub const IMWRITE_PNG_FILTER_PAETH: i32 = 128;
/// `IMWRITE_PNG_FAST_FILTERS` — This is a combination of
///   IMWRITE_PNG_FILTER_NONE, IMWRITE_PNG_FILTER_SUB, and
///   IMWRITE_PNG_FILTER_UP, typically used for faster compression.
pub const IMWRITE_PNG_FAST_FILTERS: i32 = 56;
/// `IMWRITE_PNG_ALL_FILTERS` — This combines all available filters (NONE,
///   SUB, UP, AVG, and PAETH), which will attempt to apply all of them for the
///   best possible compression.
pub const IMWRITE_PNG_ALL_FILTERS: i32 = 248;
/// `IMWRITE_PAM_FORMAT_NULL`.
pub const IMWRITE_PAM_FORMAT_NULL: i32 = 0;
/// `IMWRITE_PAM_FORMAT_BLACKANDWHITE`.
pub const IMWRITE_PAM_FORMAT_BLACKANDWHITE: i32 = 1;
/// `IMWRITE_PAM_FORMAT_GRAYSCALE`.
pub const IMWRITE_PAM_FORMAT_GRAYSCALE: i32 = 2;
/// `IMWRITE_PAM_FORMAT_GRAYSCALE_ALPHA`.
pub const IMWRITE_PAM_FORMAT_GRAYSCALE_ALPHA: i32 = 3;
/// `IMWRITE_PAM_FORMAT_RGB`.
pub const IMWRITE_PAM_FORMAT_RGB: i32 = 4;
/// `IMWRITE_PAM_FORMAT_RGB_ALPHA`.
pub const IMWRITE_PAM_FORMAT_RGB_ALPHA: i32 = 5;
/// `IMWRITE_WEBP_LOSSLESS_OFF` — Lossy compression mode. Uses
///   IMWRITE_WEBP_QUALITY to control compression. (Default).
pub const IMWRITE_WEBP_LOSSLESS_OFF: i32 = 0;
/// `IMWRITE_WEBP_LOSSLESS_ON` — Standard lossless compression. May modify or
///   discard RGB values of fully transparent pixels to improve compression
///   ratio.
pub const IMWRITE_WEBP_LOSSLESS_ON: i32 = 1;
/// `IMWRITE_WEBP_LOSSLESS_PRESERVE_COLOR` — Exact lossless compression.
///   Preserves all RGB data even for pixels with 0 alpha (equivalent to WebP's
///   exact flag).
pub const IMWRITE_WEBP_LOSSLESS_PRESERVE_COLOR: i32 = 2;
/// `IMWRITE_HDR_COMPRESSION_NONE`.
pub const IMWRITE_HDR_COMPRESSION_NONE: i32 = 0;
/// `IMWRITE_HDR_COMPRESSION_RLE`.
pub const IMWRITE_HDR_COMPRESSION_RLE: i32 = 1;
/// `IMWRITE_GIF_FAST_NO_DITHER`.
pub const IMWRITE_GIF_FAST_NO_DITHER: i32 = 1;
/// `IMWRITE_GIF_FAST_FLOYD_DITHER`.
pub const IMWRITE_GIF_FAST_FLOYD_DITHER: i32 = 2;
/// `IMWRITE_GIF_COLORTABLE_SIZE_8`.
pub const IMWRITE_GIF_COLORTABLE_SIZE_8: i32 = 3;
/// `IMWRITE_GIF_COLORTABLE_SIZE_16`.
pub const IMWRITE_GIF_COLORTABLE_SIZE_16: i32 = 4;
/// `IMWRITE_GIF_COLORTABLE_SIZE_32`.
pub const IMWRITE_GIF_COLORTABLE_SIZE_32: i32 = 5;
/// `IMWRITE_GIF_COLORTABLE_SIZE_64`.
pub const IMWRITE_GIF_COLORTABLE_SIZE_64: i32 = 6;
/// `IMWRITE_GIF_COLORTABLE_SIZE_128`.
pub const IMWRITE_GIF_COLORTABLE_SIZE_128: i32 = 7;
/// `IMWRITE_GIF_COLORTABLE_SIZE_256`.
pub const IMWRITE_GIF_COLORTABLE_SIZE_256: i32 = 8;
