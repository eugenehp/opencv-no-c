/* opencv-no-c — C ABI.
 *
 * Enabled by the `abi` Cargo feature, which produces `libopencv_no_c.a` and
 * `libopencv_no_c.dylib`/`.so`. Link either.
 *
 *     cargo build -p opencv-no-c --features abi --release
 *     cc main.c target/release/libopencv_no_c.a
 *
 * This is NOT OpenCV's C++ ABI. That depends on name mangling, vtable layout
 * and the memory layout of std::vector and cv::Mat as a particular libstdc++
 * emitted them, none of which Rust can reproduce. This is a stable C ABI in
 * the shape OpenCV's own C API had: opaque handles, status codes, and an
 * out-parameter for the result.
 *
 * Conventions:
 *   - Fallible calls return 0 on success, or cv::Error::Code unchanged. Those
 *     are negative, so `rc < 0` means failure and the value compares equal to
 *     the same constant OpenCV uses (StsBadArg is -5 in both).
 *   - cvno_last_error() gives the message for the last failure on this thread.
 *   - Handles are owned by the caller; release with the matching _release.
 *     Passing NULL to a _release is accepted, as free() is.
 *   - Nothing unwinds across this boundary; a panic becomes a status code.
 */
#ifndef OPENCV_NO_C_H
#define OPENCV_NO_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CvnoMat CvnoMat;
typedef struct CvnoBuffer CvnoBuffer;

/* Diagnostics. */
const char* cvno_version(void);
const char* cvno_last_error(void);

/* Matrix lifetime and shape. `type` is upstream's packed CV_8UC3 encoding. */
CvnoMat* cvno_mat_zeros(int rows, int cols, int type);
void     cvno_mat_release(CvnoMat* m);
int      cvno_mat_rows(const CvnoMat* m);
int      cvno_mat_cols(const CvnoMat* m);
int      cvno_mat_type(const CvnoMat* m);
size_t   cvno_mat_step(const CvnoMat* m);
/* rows * step bytes, valid until the handle is released. */
const unsigned char* cvno_mat_data(const CvnoMat* m);

/* Encoded byte buffers. */
const unsigned char* cvno_buffer_data(const CvnoBuffer* b);
size_t cvno_buffer_len(const CvnoBuffer* b);
void   cvno_buffer_release(CvnoBuffer* b);

/* imgcodecs. `flags` is IMREAD_UNCHANGED (-1), GRAYSCALE (0), COLOR (1). */
int cvno_imdecode(const unsigned char* data, size_t len, int flags, CvnoMat** out);
int cvno_imencode(const char* ext, const CvnoMat* src, CvnoBuffer** out);

/* imgproc. Enum arguments take upstream's numeric values. */
int cvno_cvt_color(const CvnoMat* src, int code, CvnoMat** out);
int cvno_resize(const CvnoMat* src, int width, int height,
                double fx, double fy, int interpolation, CvnoMat** out);
int cvno_gaussian_blur(const CvnoMat* src, int kwidth, int kheight,
                       double sigma_x, double sigma_y, CvnoMat** out);
int cvno_canny(const CvnoMat* src, double threshold1, double threshold2,
               int aperture_size, int l2_gradient, CvnoMat** out);
/* `type` may carry THRESH_OTSU (8) or THRESH_TRIANGLE (16), as upstream's
 * does; `applied` receives the threshold actually used and may be NULL. */
int cvno_threshold(const CvnoMat* src, double thresh, double maxval, int type,
                   CvnoMat** out, double* applied);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENCV_NO_C_H */
