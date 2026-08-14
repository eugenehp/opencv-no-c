#!/usr/bin/env python3
"""Generate OpenCV reference arrays for the parity check.

    pip install opencv-python
    python3 tools/gen_reference.py /tmp/parity
    cargo run --release -p opencv-no-c --example parity_check -- /tmp/parity

Writes one raw `.bin` per array plus a `manifest.txt` of
`name rows cols channels dtype`. Deliberately dependency-light so the Rust side
needs no image or array parser.
"""
import os
import sys

import cv2
import numpy as np


def main() -> None:
    d = sys.argv[1] if len(sys.argv) > 1 else "parity_ref"
    os.makedirs(d, exist_ok=True)
    np.random.seed(7)

    # Structure plus noise, so smooth regions and edges are both represented.
    h, w = 120, 160
    yy, xx = np.mgrid[0:h, 0:w]
    base = ((xx * 7 + yy * 13) % 256).astype(np.uint8)
    noise = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    gray = ((base.astype(np.int32) + noise.astype(np.int32)) // 2).astype(np.uint8)
    bgr = np.dstack([gray, np.roll(gray, 5, axis=1), np.roll(gray, 9, axis=0)]).astype(np.uint8)

    man = []

    def dump(name, a):
        a = np.ascontiguousarray(a)
        cn = 1 if a.ndim == 2 else a.shape[2]
        dt = {
            np.dtype("uint8"): "u8",
            np.dtype("uint16"): "u16",
            np.dtype("int32"): "i32",
            np.dtype("float32"): "f32",
            np.dtype("float64"): "f64",
        }[a.dtype]
        a.tofile(os.path.join(d, name + ".bin"))
        man.append(f"{name} {a.shape[0]} {a.shape[1]} {cn} {dt}")

    dump("in_gray", gray)
    dump("in_bgr", bgr)
    dump("in_b", np.roll(gray, 17))

    r101 = cv2.BORDER_REFLECT_101
    dump("cvt_bgr2gray", cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    dump("cvt_bgr2hsv", cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    dump("cvt_bgr2ycrcb", cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))
    dump("cvt_gray2bgr", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    dump("resize_nn_half", cv2.resize(bgr, (80, 60), interpolation=cv2.INTER_NEAREST))
    dump("resize_lin_half", cv2.resize(bgr, (80, 60), interpolation=cv2.INTER_LINEAR))
    dump("resize_area_half", cv2.resize(bgr, (80, 60), interpolation=cv2.INTER_AREA))
    # Halving is the only integer AREA scale that rounds half **up**; every
    # other one goes through `saturate_cast` and rounds half to even. Three is
    # odd so no tie can arise, which is why it agrees with both rules.
    dump("resize_area_4x", cv2.resize(gray, (40, 30), interpolation=cv2.INTER_AREA))
    dump("resize_area_3x", cv2.resize(gray, (53, 40), interpolation=cv2.INTER_AREA))
    dump("resize_area_5x", cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA))
    dump("resize_area_4x_bgr", cv2.resize(bgr, (40, 30), interpolation=cv2.INTER_AREA))
    dump("resize_area_half_bgr2", cv2.resize(bgr[:, :, :2].copy(), (80, 60), interpolation=cv2.INTER_AREA))
    dump("resize_lin_3x", cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_LINEAR))
    ex = cv2.INTER_LINEAR_EXACT
    dump("resize_linexact_3x", cv2.resize(gray, (w * 3, h * 3), interpolation=ex))
    dump("resize_linexact_half", cv2.resize(bgr, (80, 60), interpolation=ex))
    dump("resize_linexact_125", cv2.resize(bgr, (200, 150), interpolation=ex))
    dump("resize_linexact_3rd", cv2.resize(gray, (w // 3, h // 3), interpolation=ex))
    dump("resize_area_3rd", cv2.resize(gray, (w // 3, h // 3), interpolation=cv2.INTER_AREA))

    k3 = np.ones((3, 3), np.float64) / 9.0
    dump("filter2d_box3", cv2.filter2D(bgr, -1, k3, borderType=r101))
    ksharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float64)
    dump("filter2d_sharp", cv2.filter2D(gray, -1, ksharp, borderType=r101))
    dump("blur5", cv2.blur(bgr, (5, 5), borderType=r101))
    dump("gauss5", cv2.GaussianBlur(bgr, (5, 5), 1.4, borderType=r101))
    dump("gauss5_s0", cv2.GaussianBlur(gray, (5, 5), 0, borderType=r101))
    dump("median3", cv2.medianBlur(gray, 3))
    dump("median5", cv2.medianBlur(gray, 5))

    dump("sobel_x_f32", cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE))
    dump("lap1_f32", cv2.Laplacian(gray, cv2.CV_32F, ksize=1, borderType=r101))
    dump("lap3_f32", cv2.Laplacian(gray, cv2.CV_32F, ksize=3, borderType=r101))

    _, t = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    dump("thresh_bin", t)
    tv, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dump("thresh_otsu", t)
    with open(os.path.join(d, "otsu_value.txt"), "w") as f:
        f.write(str(tv))

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dump("erode3", cv2.erode(gray, se, borderType=cv2.BORDER_REPLICATE))
    dump("dilate3", cv2.dilate(gray, se, borderType=cv2.BORDER_REPLICATE))
    dump("se_ellipse5", cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    dump("se_ellipse7", cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    dump("canny", cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=False))
    dump("border_r101", cv2.copyMakeBorder(gray, 4, 4, 4, 4, r101))
    dump("border_wrap", cv2.copyMakeBorder(gray, 4, 4, 4, 4, cv2.BORDER_WRAP))
    dump("equalize", cv2.equalizeHist(gray))
    dump("integral_f64", cv2.integral(gray, sdepth=cv2.CV_64F))

    a, b = gray, np.roll(gray, 17)
    dump("add", cv2.add(a, b))
    dump("sub", cv2.subtract(a, b))
    dump("absdiff", cv2.absdiff(a, b))

    for n, sg in [(3, 0), (5, 0), (7, 0), (5, 1.4), (11, 0), (9, 2.5)]:
        dump(f"gk_{n}_{str(sg).replace('.', 'p')}", cv2.getGaussianKernel(n, sg).astype(np.float64))

    extended(d, gray, bgr, dump)
    unverified(d, gray, bgr, dump)
    dense_flow(d, gray, bgr, dump)
    calib(d, dump)
    photo_and_shapes(d, gray, bgr, dump)
    models_and_boards(d, gray, bgr, dump)
    boards_stitching_gapi(d, gray, bgr, dump)
    last_mile(d, gray, bgr, dump)
    core_and_video(d, gray, bgr, dump)
    features(d, gray, bgr, dump)
    flann_and_hog(d, gray, bgr, dump)
    remaining(d, gray, bgr, dump)

    with open(os.path.join(d, "manifest.txt"), "w") as f:
        f.write("\n".join(man) + "\n")
    print(f"wrote {len(man)} arrays from cv2 {cv2.__version__} to {d}")




def extended(d, gray, bgr, dump):
    """Reference arrays for the rest of the ported surface.

    Kept separate from `main`'s core set so the two can be read independently.
    """
    import numpy as np
    import cv2

    h, w = gray.shape
    b2 = np.roll(gray, 17)
    f32 = (gray.astype(np.float32) / 255.0) * 4.0 - 2.0
    dump("in_f32", f32)

    # ---- scalar-path arithmetic ----
    # OpenCV vectorises `divide` and `addWeighted` in blocks of 8, and the
    # vector path disagrees with its own scalar path on exact .5 ties. A
    # 7-element buffer stays below that block, so these two check the port
    # against upstream's *scalar reference* implementation, where OpenCV is
    # self-consistent. See README "What still diverges".
    sa = np.array([[89, 129, 173, 193, 175, 153, 133]], np.uint8)
    sb = np.array([[103, 58, 116, 121, 70, 102, 38]], np.uint8)
    dump("in_sa", sa)
    dump("in_sb", sb)
    dump("ar_div_scalar", cv2.divide(sa, sb))
    dump("ar_aw_scalar", cv2.addWeighted(sa, 0.7, sb, 0.4, 12.0))

    # ---- core arithmetic ----
    dump("ar_mul", cv2.multiply(gray, b2))
    dump("ar_mul_s", cv2.multiply(gray, b2, scale=0.01))
    dump("ar_div", cv2.divide(gray, b2))
    dump("ar_addw", cv2.addWeighted(gray, 0.7, b2, 0.4, 12.0))
    dump("ar_min", cv2.min(gray, b2))
    dump("ar_max", cv2.max(gray, b2))
    dump("ar_and", cv2.bitwise_and(gray, b2))
    dump("ar_or", cv2.bitwise_or(gray, b2))
    dump("ar_xor", cv2.bitwise_xor(gray, b2))
    dump("ar_not", cv2.bitwise_not(gray))
    dump("ar_cmp_gt", cv2.compare(gray, b2, cv2.CMP_GT))
    dump("ar_cmp_le", cv2.compare(gray, b2, cv2.CMP_LE))
    dump("ar_inrange", cv2.inRange(bgr, (40, 40, 40), (200, 200, 200)))
    dump("ar_conv_f32", gray.astype(np.float32) * 0.5 + 3.0)
    dump("ar_conv_u8", cv2.convertScaleAbs(gray, alpha=0.37, beta=5.0))
    dump("ar_transpose", cv2.transpose(gray))
    for c in (0, 1, -1):
        dump(f"ar_flip{c+1}", cv2.flip(bgr, c))
    ch = cv2.split(bgr)
    for i, p in enumerate(ch):
        dump(f"ar_split{i}", p)
    dump("ar_merge", cv2.merge(ch))
    mask = (gray > 128).astype(np.uint8) * 255
    dump("in_mask", mask)
    dump("ar_add_mask", cv2.add(gray, b2, mask=mask))

    # ---- statistics (scalars go through a 1xN f64 array) ----
    def scal(name, vals):
        dump(name, np.array(vals, np.float64).reshape(1, -1))

    mn, mx, mnl, mxl = cv2.minMaxLoc(gray)
    scal("st_minmax", [mn, mx, mnl[0], mnl[1], mxl[0], mxl[1]])
    scal("st_sum", list(cv2.sumElems(bgr)))
    scal("st_nonzero", [cv2.countNonZero(gray)])
    scal("st_mean", list(cv2.mean(bgr)))
    mu, sd = cv2.meanStdDev(bgr)
    scal("st_meanstd", list(mu.ravel()) + list(sd.ravel()))
    scal("st_norms", [cv2.norm(f32, cv2.NORM_INF), cv2.norm(f32, cv2.NORM_L1),
                      cv2.norm(f32, cv2.NORM_L2), cv2.norm(f32, cv2.NORM_L2SQR)])
    scal("st_normdiff", [cv2.norm(gray, b2, cv2.NORM_INF), cv2.norm(gray, b2, cv2.NORM_L1),
                         cv2.norm(gray, b2, cv2.NORM_L2)])
    dump("st_normalize", cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX))

    # ---- math functions ----
    pos = np.abs(f32) + 0.25
    dump("in_pos", pos)
    dump("mf_sqrt", cv2.sqrt(pos))
    dump("mf_exp", cv2.exp(f32 * 0.25))
    dump("mf_log", cv2.log(pos))
    dump("mf_pow3", cv2.pow(f32, 3.0))
    dump("mf_pow_half", cv2.pow(f32, 0.5))
    yv = np.roll(f32, 5).astype(np.float32)
    dump("in_f32b", yv)
    dump("mf_mag", cv2.magnitude(f32, yv))
    dump("mf_phase_deg", cv2.phase(f32, yv, angleInDegrees=True))
    dump("mf_phase_rad", cv2.phase(f32, yv, angleInDegrees=False))
    mag, ang = cv2.cartToPolar(f32, yv)
    dump("mf_c2p_mag", mag)
    dump("mf_c2p_ang", ang)
    px, py = cv2.polarToCart(mag, ang)
    dump("mf_p2c_x", px)
    dump("mf_p2c_y", py)
    lut = np.array([(i * 3) % 256 for i in range(256)], np.uint8)
    dump("in_lut", lut.reshape(1, 256))
    dump("mf_lut", cv2.LUT(gray, lut))

    # ---- linear algebra ----
    rng = np.random.RandomState(3)
    # Names avoid `in_A`/`in_B`: a case-insensitive filesystem (the macOS
    # default) would fold those onto the existing `in_b` array.
    A = rng.randn(6, 6).astype(np.float64)
    B = rng.randn(6, 4).astype(np.float64)
    spd = A @ A.T + 6 * np.eye(6)
    dump("la_mat_a", A)
    dump("la_mat_b", B)
    dump("la_mat_spd", spd)
    dump("la_gemm", cv2.gemm(A, B, 1.0, None, 0.0))
    dump("la_gemm_at", cv2.gemm(A, B, 2.0, None, 0.0, flags=cv2.GEMM_1_T))
    scal("la_det", [cv2.determinant(A)])
    scal("la_trace", list(cv2.trace(A))[:1])
    dump("la_inv", cv2.invert(A, flags=cv2.DECOMP_LU)[1])
    dump("la_inv_chol", cv2.invert(spd, flags=cv2.DECOMP_CHOLESKY)[1])
    dump("la_solve", cv2.solve(A, B, flags=cv2.DECOMP_LU)[1])
    dump("la_mulT_ata", cv2.mulTransposed(A, True))
    dump("la_mulT_aat", cv2.mulTransposed(A, False))

    # ---- reshaping / channels ----
    dump("mp_reduce_sum0", cv2.reduce(gray, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F))
    dump("mp_reduce_avg1", cv2.reduce(gray, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F))
    # REDUCE_MAX/MIN keep the source depth, so run them on the u8 image directly.
    dump("mp_reduce_max0", cv2.reduce(gray, 0, cv2.REDUCE_MAX))
    dump("mp_reduce_min1", cv2.reduce(gray, 1, cv2.REDUCE_MIN))
    dump("mp_repeat", cv2.repeat(gray[:8, :8], 2, 3))
    dump("mp_hconcat", cv2.hconcat([gray[:, :20], gray[:, 20:50]]))
    dump("mp_vconcat", cv2.vconcat([gray[:20], gray[20:50]]))
    dump("mp_sort_row", cv2.sort(gray, cv2.SORT_EVERY_ROW | cv2.SORT_ASCENDING))
    dump("mp_sort_coldesc", cv2.sort(gray, cv2.SORT_EVERY_COLUMN | cv2.SORT_DESCENDING))
    dump("mp_sortidx", cv2.sortIdx(gray, cv2.SORT_EVERY_ROW | cv2.SORT_ASCENDING))
    dump("mp_extract1", cv2.extractChannel(bgr, 1))

    # ---- more colour conversions ----
    for name, code in [("bgra", cv2.COLOR_BGR2BGRA), ("rgb", cv2.COLOR_BGR2RGB)]:
        dump(f"cvt_{name}", cv2.cvtColor(bgr, code))
    dump("cvt_rgb2gray", cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY))
    dump("cvt_ycrcb2bgr", cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb), cv2.COLOR_YCrCb2BGR))
    dump("cvt_hsv2bgr", cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2BGR))

    # ---- borders ----
    for nm, bt in [("rep", cv2.BORDER_REPLICATE), ("ref", cv2.BORDER_REFLECT),
                   ("con", cv2.BORDER_CONSTANT)]:
        dump(f"bd_{nm}", cv2.copyMakeBorder(gray, 3, 5, 7, 2, bt, value=0))

    # ---- more filters ----
    dump("fl_box_un", cv2.boxFilter(gray, -1, (3, 3), normalize=False,
                                    borderType=cv2.BORDER_REFLECT_101))
    dump("fl_sep", cv2.sepFilter2D(gray, cv2.CV_32F, np.array([1., 2., 1.]),
                                   np.array([-1., 0., 1.]), borderType=cv2.BORDER_REFLECT_101))
    dump("fl_sobel_y", cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5,
                                 borderType=cv2.BORDER_REPLICATE))
    dump("fl_scharr", cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1,
                                borderType=cv2.BORDER_REPLICATE))
    dump("fl_gauss7", cv2.GaussianBlur(bgr, (7, 7), 2.1, borderType=cv2.BORDER_REPLICATE))
    dump("fl_median7", cv2.medianBlur(gray, 7))

    # ---- thresholds ----
    for nm, tt in [("trunc", cv2.THRESH_TRUNC), ("tozero", cv2.THRESH_TOZERO),
                   ("tozeroinv", cv2.THRESH_TOZERO_INV), ("bininv", cv2.THRESH_BINARY_INV)]:
        dump(f"th_{nm}", cv2.threshold(gray, 100, 240, tt)[1])
    tv, tr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    dump("th_triangle", tr)
    with open(os.path.join(d, "triangle_value.txt"), "w") as f:
        f.write(str(tv))
    dump("th_adapt_mean", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 7, 4))

    # ---- morphology ----
    for nm, sh in [("rect", cv2.MORPH_RECT), ("cross", cv2.MORPH_CROSS)]:
        dump(f"se_{nm}5", cv2.getStructuringElement(sh, (5, 5)))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for nm, op in [("open", cv2.MORPH_OPEN), ("close", cv2.MORPH_CLOSE),
                   ("grad", cv2.MORPH_GRADIENT), ("tophat", cv2.MORPH_TOPHAT),
                   ("blackhat", cv2.MORPH_BLACKHAT)]:
        dump(f"mo_{nm}", cv2.morphologyEx(gray, op, se, borderType=cv2.BORDER_REPLICATE))
    dump("mo_erode2", cv2.erode(gray, se, iterations=2, borderType=cv2.BORDER_REPLICATE))

    # ---- pyramids ----
    dump("py_down", cv2.pyrDown(bgr))
    dump("py_up", cv2.pyrUp(bgr))
    dump("py_down_odd", cv2.pyrDown(gray[:117, :157]))

    # ---- histogram ----
    dump("hi_hist", cv2.calcHist([gray], [0], None, [256], [0, 256]))
    dump("hi_hist16", cv2.calcHist([gray], [0], None, [16], [0, 256]))

    # ---- template matching ----
    templ = gray[30:46, 40:60].copy()
    dump("in_templ", templ)
    for nm, m in [("sqdiff", cv2.TM_SQDIFF), ("ccorr", cv2.TM_CCORR),
                  ("ccoeff", cv2.TM_CCOEFF), ("ccoeffn", cv2.TM_CCOEFF_NORMED),
                  ("ccorrn", cv2.TM_CCORR_NORMED), ("sqdiffn", cv2.TM_SQDIFF_NORMED)]:
        dump(f"tm_{nm}", cv2.matchTemplate(gray, templ, m))

    # ---- shape ----
    binimg = (gray > 140).astype(np.uint8)
    dump("in_bin", binimg * 255)
    m = cv2.moments(binimg, binaryImage=True)
    scal("sh_moments", [m[k] for k in ("m00", "m10", "m01", "m20", "m11", "m02",
                                       "m30", "m21", "m12", "m03",
                                       "mu20", "mu11", "mu02", "mu30", "mu21", "mu12", "mu03",
                                       "nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03")])
    poly = np.array([[10, 10], [90, 14], [120, 70], [60, 110], [15, 80]], np.int32)
    dump("in_poly", poly.reshape(-1, 2).astype(np.float64))
    scal("sh_polyarea", [cv2.contourArea(poly), cv2.contourArea(poly, True),
                         cv2.arcLength(poly, True), cv2.arcLength(poly, False)])
    x, y, ww, hh = cv2.boundingRect(poly)
    scal("sh_bbox", [x, y, ww, hh])
    hull = cv2.convexHull(poly, clockwise=True)
    dump("sh_hull", hull.reshape(-1, 2).astype(np.float64))
    # A second, general-position set. The pentagon above is degenerate for
    # upstream's Sklansky sweep -- its lowest point is also its leftmost, so one
    # of the four quarter-sweeps collapses to a single point and the emitted
    # cycle starts one vertex later than the sweep alone predicts. In this set
    # neither extreme-y point sits at an end of the x-sorted order.
    poly2 = np.array([[160, 57], [29, 63], [84, 156], [91, 26], [70, 118],
                      [156, 141], [188, 43], [168, 19], [110, 59]], np.int32)
    dump("in_poly2", poly2.reshape(-1, 2).astype(np.float64))
    dump("sh_hull2", cv2.convexHull(poly2, clockwise=True).reshape(-1, 2).astype(np.float64))
    dump("sh_hull2_ccw", cv2.convexHull(poly2, clockwise=False).reshape(-1, 2).astype(np.float64))
    ap = cv2.approxPolyDP(poly, 5.0, True)
    dump("sh_approx", ap.reshape(-1, 2).astype(np.float64))
    scal("sh_ppt", [cv2.pointPolygonTest(poly, (50.0, 50.0), False),
                    cv2.pointPolygonTest(poly, (50.0, 50.0), True),
                    cv2.pointPolygonTest(poly, (200.0, 50.0), False),
                    cv2.pointPolygonTest(poly, (200.0, 50.0), True)])
    n_lbl, lbl = cv2.connectedComponents(binimg, connectivity=8)
    dump("sh_cc_labels", lbl.astype(np.int32))
    scal("sh_cc_n", [n_lbl])
    n2, lbl2, stats, cents = cv2.connectedComponentsWithStats(binimg, connectivity=4)
    dump("sh_ccs_stats", stats.astype(np.float64))
    dump("sh_ccs_cent", cents.astype(np.float64))

    # ---- warping ----
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 23.0, 1.3)
    dump("wp_rotmat", M)
    src3 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst3 = np.float32([[5, 9], [w - 12, 3], [11, h - 7]])
    dump("wp_affmat", cv2.getAffineTransform(src3, dst3))
    src4 = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst4 = np.float32([[7, 4], [w - 9, 11], [w - 3, h - 6], [2, h - 14]])
    permat = cv2.getPerspectiveTransform(src4, dst4)
    dump("wp_permat", permat)
    # Pins `Perspective3x3::invert`, which nothing else covers.
    dump("ip_perspinv", cv2.invert(permat, flags=cv2.DECOMP_LU)[1])

    # --- distanceTransform: every metric and mask cv2 accepts. The chamfer
    # passes run in 1/65536 fixed point upstream, so these are bit-exact
    # comparisons, not tolerance ones.
    seeds = np.full((40, 48), 255, np.uint8)
    for sy, sx in [(2, 5), (7, 33), (11, 18), (16, 44), (21, 3),
                   (26, 27), (31, 12), (36, 39), (38, 21)]:
        seeds[sy, sx] = 0
    dump("in_dt_seeds", seeds)
    dump("dt_l2_3", cv2.distanceTransform(seeds, cv2.DIST_L2, 3))
    dump("dt_l2_5", cv2.distanceTransform(seeds, cv2.DIST_L2, 5))
    dump("dt_l1_3", cv2.distanceTransform(seeds, cv2.DIST_L1, 3))
    dump("dt_c_3", cv2.distanceTransform(seeds, cv2.DIST_C, 3))
    dump("dt_precise", cv2.distanceTransform(seeds, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))

    # --- corner and gradient operators not previously covered.
    dump("ip_harris", cv2.cornerHarris(gray.astype(np.float32), 2, 3, 0.04))
    dump("ip_precorner", cv2.preCornerDetect(gray.astype(np.float32), 3))
    # `spatialGradient` returns CV_16S; widened to i32 for the fixture format.
    gx, gy = cv2.spatialGradient(gray)
    dump("ip_spatgrad_x", gx.astype(np.int32))
    dump("ip_spatgrad_y", gy.astype(np.int32))

    # --- geometric transforms.
    rotm = cv2.getRotationMatrix2D((gray.shape[1] / 2.0, gray.shape[0] / 2.0), 27.0, 1.15)
    dump("ip_warpaffine", cv2.warpAffine(gray, rotm, (gray.shape[1], gray.shape[0])))
    dump("ip_warpperspective",
         cv2.warpPerspective(gray, cv2.getPerspectiveTransform(src4, dst4),
                             (gray.shape[1], gray.shape[0])))

    dump("ip_mineigen", cv2.cornerMinEigenVal(gray.astype(np.float32), 2, 3))
    dump("ip_bilateral", cv2.bilateralFilter(gray, 5, 40.0, 8.0))
    # 3-channel: upstream's colour distance is L1, not squared L2.
    dump("ip_bilateral_bgr", cv2.bilateralFilter(bgr, 5, 40.0, 8.0))

    # --- remap with an explicit swirl, so the maps are not degenerate.
    hh, ww = gray.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    mapx = (xx + 3.0 * np.sin(yy / 9.0)).astype(np.float32)
    mapy = (yy + 3.0 * np.cos(xx / 11.0)).astype(np.float32)
    dump("in_mapx", mapx)
    dump("in_mapy", mapy)
    dump("ip_remap_nn", cv2.remap(gray, mapx, mapy, cv2.INTER_NEAREST))
    dump("ip_remap_lin", cv2.remap(gray, mapx, mapy, cv2.INTER_LINEAR))

    # --- eigen/structure operators and accumulators.
    dump("ip_eigenvv", cv2.cornerEigenValsAndVecs(gray.astype(np.float32), 3, 3))
    dump("ip_sqrbox", cv2.sqrBoxFilter(gray.astype(np.float32), -1, (3, 3), normalize=True))
    acc = np.zeros(gray.shape, np.float32)
    cv2.accumulate(gray.astype(np.float32), acc)
    cv2.accumulate(np.roll(gray, 17).astype(np.float32), acc)
    dump("ip_accumulate", acc)
    accw = gray.astype(np.float32).copy()
    cv2.accumulateWeighted(np.roll(gray, 17).astype(np.float32), accw, 0.3)
    dump("ip_accweighted", accw)
    accs = np.zeros(gray.shape, np.float32)
    cv2.accumulateSquare(gray.astype(np.float32), accs)
    dump("ip_accsquare", accs)

    # --- connected components on a thresholded field.
    _, binimg = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    dump("in_binary", binimg)
    n, lab = cv2.connectedComponents(binimg, connectivity=8)
    dump("ip_cc8", lab)
    n4, lab4 = cv2.connectedComponents(binimg, connectivity=4)
    dump("ip_cc4", lab4)

    # --- morphologyEx, all five compound operations.
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for nm, op in [("open", cv2.MORPH_OPEN), ("close", cv2.MORPH_CLOSE),
                   ("grad", cv2.MORPH_GRADIENT), ("tophat", cv2.MORPH_TOPHAT),
                   ("blackhat", cv2.MORPH_BLACKHAT)]:
        dump("ip_morph_" + nm, cv2.morphologyEx(gray, op, se))

    # --- floodFill from a seed inside the bright region.
    ff = gray.copy()
    mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), np.uint8)
    cnt, _, _, rect = cv2.floodFill(ff, mask, (40, 30), 200, 40, 40, 4)
    dump("ip_floodfill", ff)
    with open(os.path.join(d, "ip_floodfill_meta.txt"), "w") as fh:
        fh.write(f"{cnt} {rect[0]} {rect[1]} {rect[2]} {rect[3]}\n")

    # --- goodFeaturesToTrack, Shi-Tomasi and Harris.
    for nm, hk in [("shi", False), ("harris", True)]:
        pts = cv2.goodFeaturesToTrack(gray, 20, 0.01, 5.0, blockSize=3,
                                      useHarrisDetector=hk, k=0.04)
        arr = np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2)
        dump("ip_gftt_" + nm, arr.astype(np.float32))

    # ---- core linear algebra and misc, previously untested ----
    sym = np.array([[6.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 4.0]], np.float64)
    dump("in_sym3", sym)
    dump("co_eigen_vals", cv2.eigen(sym)[1])
    dump("co_eigen_vecs", cv2.eigen(sym)[2])
    w_, u_, vt_ = cv2.SVDecomp(A)
    dump("co_svd_w", w_)
    dump("co_svd_u", u_)
    dump("co_svd_vt", vt_)
    dump("co_trace", np.array([[float(cv2.trace(A)[0])]]))
    dump("co_scaleabs", cv2.convertScaleAbs(gray, alpha=1.7, beta=-30.0))
    dump("co_rotate90", cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE))
    dump("co_rotate180", cv2.rotate(gray, cv2.ROTATE_180))
    dump("co_rotate270", cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE))
    pts3 = np.array([[[1.0, 2.0], [30.0, 8.0], [55.0, 41.0], [7.0, 63.0]]], np.float64)
    dump("in_ptransform_src", pts3.reshape(-1, 2))
    dump("co_ptransform", cv2.perspectiveTransform(pts3, permat).reshape(-1, 2))
    cov, mean_ = cv2.calcCovarMatrix(A, None,
                                     cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
    dump("co_covar", cov)
    dump("co_covar_mean", mean_)

    # --- more core: minMaxLoc, split, DFT round trip, spectra, scaleAdd.
    mn, mx, mnl, mxl = cv2.minMaxLoc(gray)
    dump("co_minmaxloc", np.array([[float(mn), float(mx),
                                    float(mnl[0]), float(mnl[1]),
                                    float(mxl[0]), float(mxl[1])]]))
    for i, ch in enumerate(cv2.split(bgr)):
        dump(f"co_split{i}", ch)
    f32in = gray.astype(np.float32)
    sp = cv2.dft(f32in, flags=cv2.DFT_COMPLEX_OUTPUT)
    dump("co_dft", sp)
    dump("co_idft", cv2.idft(sp, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT))
    sp2 = cv2.dft(np.roll(f32in, 17), flags=cv2.DFT_COMPLEX_OUTPUT)
    dump("co_mulspectrums", cv2.mulSpectrums(sp, sp2, 0))
    dump("co_scaleadd", cv2.scaleAdd(A, 2.5, np.roll(A, 3, axis=0)))
    ident = np.zeros((5, 5), np.float64)
    cv2.setIdentity(ident, 3.0)
    dump("co_setidentity", ident)

    # --- Mahalanobis, PCA, sortIdx, copyMakeBorder variants.
    icov = np.linalg.inv(np.cov(A.T) + 1e-6 * np.eye(6))
    v1 = A[0:1, :].copy()
    v2 = A[1:2, :].copy()
    dump("in_maha_icov", icov)
    dump("co_mahalanobis", np.array([[cv2.Mahalanobis(v1, v2, icov)]]))
    mean_pca, eig_pca = cv2.PCACompute(A, mean=None, maxComponents=3)
    dump("co_pca_mean", mean_pca)
    dump("co_pca_vecs", eig_pca)
    for nm, bt in [("const", cv2.BORDER_CONSTANT), ("replicate", cv2.BORDER_REPLICATE),
                   ("reflect", cv2.BORDER_REFLECT)]:
        dump("co_border_" + nm, cv2.copyMakeBorder(gray, 3, 4, 5, 6, bt, value=7))

    # ---- calib3d, previously untested ----
    rvec = np.array([0.31, -0.72, 0.14], np.float64)
    Rm, _ = cv2.Rodrigues(rvec)
    dump("cb_rodrigues_fwd", Rm)
    dump("cb_rodrigues_inv", cv2.Rodrigues(Rm)[0].reshape(1, 3))
    K = np.array([[520.0, 0.0, 319.5], [0.0, 515.0, 239.5], [0.0, 0.0, 1.0]], np.float64)
    fovx, fovy, flen, pp, ar = cv2.calibrationMatrixValues(K, (640, 480), 4.8, 3.6)
    dump("cb_calibvalues", np.array([[fovx, fovy, flen, pp[0], pp[1], ar]]))
    rq = cv2.RQDecomp3x3(np.array([[3.0, 1.0, 2.0], [0.0, 4.0, 1.5], [0.0, 0.0, 2.0]], np.float64))
    dump("cb_rq_r", rq[1])
    dump("cb_rq_q", rq[2])
    hsrc = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 80.0], [0.0, 80.0],
                     [50.0, 40.0], [20.0, 70.0]], np.float64)
    hdst = np.array([[5.0, 3.0], [98.0, 9.0], [104.0, 77.0], [1.0, 71.0],
                     [52.0, 41.0], [21.0, 66.0]], np.float64)
    dump("in_homog_src", hsrc)
    dump("in_homog_dst", hdst)
    dump("cb_homography", cv2.findHomography(hsrc, hdst, 0)[0])

    # ---- video, previously untested ----
    # A translated copy of the texture, so the flow has a known answer.
    shift = np.roll(np.roll(gray, 2, axis=0), 3, axis=1)
    dump("in_flow_a", gray)
    dump("in_flow_b", shift)
    dump("vi_ecc", np.array([[cv2.computeECC(gray.astype(np.float32),
                                             shift.astype(np.float32))]]))
    pts = np.array([[[30., 25.]], [[70., 55.]], [[110., 40.]], [[50., 90.]]], np.float32)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        gray, shift, pts, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    dump("in_flow_pts", pts.reshape(-1, 2).astype(np.float64))
    dump("vi_pyrlk", nxt.reshape(-1, 2).astype(np.float64))
    dump("vi_pyrlk_status", st.reshape(-1, 1).astype(np.int32))

    # ---- photo, previously untested ----
    dump("ph_nlmeans", cv2.fastNlMeansDenoising(gray, None, 10.0, 7, 21))
    ipmask = np.zeros(gray.shape, np.uint8)
    ipmask[20:26, 30:40] = 255
    ipmask[60:64, 90:110] = 255
    dump("in_inpaint_mask", ipmask)
    dump("ph_inpaint_telea", cv2.inpaint(gray, ipmask, 3.0, cv2.INPAINT_TELEA))
    dump("ph_inpaint_ns", cv2.inpaint(gray, ipmask, 3.0, cv2.INPAINT_NS))
    dump("ph_edgepreserve", cv2.edgePreservingFilter(bgr, None, cv2.RECURS_FILTER, 60.0, 0.4))
    dump("ph_detailenhance", cv2.detailEnhance(bgr, None, 10.0, 0.15))
    dump("ph_stylization", cv2.stylization(bgr, None, 60.0, 0.45))

    # ---- objdetect + imgcodecs, previously untested ----
    rects = np.array([[10, 10, 20, 20], [12, 11, 20, 20], [11, 12, 21, 21],
                      [80, 60, 30, 30], [82, 61, 30, 30], [200, 5, 15, 15]])
    dump("in_grouprects", rects.astype(np.float64))
    grouped, _ = cv2.groupRectangles(rects.tolist(), 1, 0.2)
    g = np.zeros((0, 4)) if len(grouped) == 0 else np.array(grouped, np.float64)
    dump("od_grouprects", g)

    ok, png = cv2.imencode(".png", bgr)
    dump("ic_png_roundtrip", cv2.imdecode(png, cv2.IMREAD_COLOR))
    ok, pgm = cv2.imencode(".pgm", gray)
    dump("ic_pgm_roundtrip", cv2.imdecode(pgm, cv2.IMREAD_GRAYSCALE))
    dump("ic_decode_gray_of_color", cv2.imdecode(png, cv2.IMREAD_GRAYSCALE))

    # ---- ml, previously untested ----
    tr = rng.randn(60, 3).astype(np.float32)
    lab = (tr[:, 0] + tr[:, 1] > 0).astype(np.int32)
    dump("in_ml_train", tr.astype(np.float64))
    dump("in_ml_labels", lab.reshape(-1, 1).astype(np.int32))
    knn = cv2.ml.KNearest_create()
    knn.train(tr, cv2.ml.ROW_SAMPLE, lab)
    _, knn_res, _, _ = knn.findNearest(tr[:10], 3)
    dump("ml_knn", knn_res.reshape(-1, 1).astype(np.int32))
    nb = cv2.ml.NormalBayesClassifier_create()
    nb.train(tr, cv2.ml.ROW_SAMPLE, lab)
    _, nb_res = nb.predict(tr[:10])
    dump("ml_bayes", nb_res.reshape(-1, 1).astype(np.int32))

    # ---- shape fitting, previously untested ----
    poly = np.array([[10, 12], [48, 8], [70, 40], [55, 78], [20, 70], [6, 44],
                     [30, 30], [40, 55]], np.int32)
    dump("in_shape_pts", poly.astype(np.float64))
    mar = cv2.minAreaRect(poly)
    dump("sh_minarearect", np.array([[mar[0][0], mar[0][1], mar[1][0], mar[1][1], mar[2]]]))
    ctr, rad = cv2.minEnclosingCircle(poly)
    dump("sh_minenclosing", np.array([[ctr[0], ctr[1], rad]]))
    dump("sh_fitline_l2", cv2.fitLine(poly, cv2.DIST_L2, 0, 0.01, 0.01).reshape(1, 4))
    mom = cv2.moments(poly)
    dump("sh_humoments", cv2.HuMoments(mom).reshape(1, 7))

    # ---- core linear algebra and statistics, previously untested ----
    # A well-conditioned symmetric system, so every decomposition is defined.
    A = np.array([[4., 1., 2.], [1., 5., 3.], [2., 3., 6.]])
    bcol = np.array([[1.], [2.], [3.]])
    dump("in_la_a", A)
    dump("in_la_b", bcol)
    for name, flag in [("lu", cv2.DECOMP_LU), ("chol", cv2.DECOMP_CHOLESKY),
                       ("svd", cv2.DECOMP_SVD), ("eig", cv2.DECOMP_EIG),
                       ("qr", cv2.DECOMP_QR)]:
        _, x = cv2.solve(A, bcol, flags=flag)
        dump(f"la_solve_{name}", x)
    for name, flag in [("svd", cv2.DECOMP_SVD), ("eig", cv2.DECOMP_EIG),
                       ("lu", cv2.DECOMP_LU)]:
        _, inv = cv2.invert(A, flags=flag)
        dump(f"la_invert_{name}", inv)

    # Over-determined: least squares through SVD, QR and the normal equations.
    M = np.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.]])
    yv = np.array([[6.], [5.], [7.], [10.]])
    dump("in_la_over_a", M)
    dump("in_la_over_b", yv)
    for name, flag in [("svd", cv2.DECOMP_SVD), ("qr", cv2.DECOMP_QR),
                       ("normal", cv2.DECOMP_NORMAL | cv2.DECOMP_LU)]:
        _, x = cv2.solve(M, yv, flags=flag)
        dump(f"la_over_{name}", x)
    _, pinv = cv2.invert(M, flags=cv2.DECOMP_SVD)
    dump("la_pinv", pinv)

    # PCA on the same sample set the covariance test uses.
    samples = rng.randn(40, 3).astype(np.float64) @ np.array(
        [[3., 0., 0.], [0., 1., 0.], [0., 0., 0.2]])
    dump("in_pca_samples", samples)
    mean_pca, evec = cv2.PCACompute(samples, mean=None)
    dump("la_pca_mean", mean_pca)
    # Sign is arbitrary per axis; pin it so the comparison is meaningful.
    evec = evec * np.sign(evec[np.arange(evec.shape[0]),
                               np.argmax(np.abs(evec), axis=1)])[:, None]
    dump("la_pca_evec", evec)

    # Linear programming: Cormen's worked example.
    lp_c = np.array([[3., 1., 2.]])
    lp_a = np.array([[1., 1., 3., 30.], [2., 2., 5., 24.], [4., 1., 2., 36.]])
    dump("in_lp_c", lp_c)
    dump("in_lp_a", lp_a)
    lp_z = np.zeros((0, 0), np.float64)
    lp_ret = cv2.solveLP(lp_c, lp_a, lp_z)
    dump("lp_result", np.array([[float(lp_ret[0])]]))
    dump("lp_z", lp_ret[1])

    # Norms: the bit-counting family and the masked and relative forms.
    ha = np.array([[0b10110100, 0b00001111, 0b11111111, 0b00000000]], np.uint8)
    hb = np.array([[0b10110000, 0b10101010, 0b00000000, 0b11000011]], np.uint8)
    dump("in_hamming_a", ha)
    dump("in_hamming_b", hb)
    dump("st_norm_hamming", np.array([[
        cv2.norm(ha, cv2.NORM_HAMMING),
        cv2.norm(ha, cv2.NORM_HAMMING2),
        cv2.norm(ha, hb, cv2.NORM_HAMMING),
        cv2.norm(ha, hb, cv2.NORM_HAMMING2)]]))
    gmask = (gray > 127).astype(np.uint8)
    dump("in_norm_mask", gmask)
    gf = gray.astype(np.float64)
    dump("st_norm_masked", np.array([[
        cv2.norm(gf, cv2.NORM_L1, gmask),
        cv2.norm(gf, cv2.NORM_L2, gmask),
        cv2.norm(gf, cv2.NORM_INF, gmask)]]))
    bf = np.roll(gray, 17).astype(np.float64)
    dump("st_norm_relative", np.array([[
        cv2.norm(gf, bf, cv2.NORM_L2 | cv2.NORM_RELATIVE),
        cv2.norm(gf, bf, cv2.NORM_L1 | cv2.NORM_RELATIVE)]]))

    # REDUCE_SUM2, added alongside the other reductions.
    dump("mp_reduce_sum2_0", cv2.reduce(gray, 0, cv2.REDUCE_SUM2, dtype=cv2.CV_64F))
    dump("mp_reduce_sum2_1", cv2.reduce(gray, 1, cv2.REDUCE_SUM2, dtype=cv2.CV_64F))


def remaining(d, gray, bgr, dump):
    """The imgproc, core and video calls no check had reached."""
    import numpy as np
    import cv2

    h, w = gray.shape

    # ---- imgproc ----
    dump("rm_clahe", cv2.createCLAHE(2.0, (8, 8)).apply(gray))
    dump("rm_colormap_jet", cv2.applyColorMap(gray, cv2.COLORMAP_JET))
    dump("rm_colormap_hot", cv2.applyColorMap(gray, cv2.COLORMAP_HOT))

    # Polar warps: pure resampling geometry.
    centre = (w / 2.0, h / 2.0)
    maxr = min(w, h) / 2.0
    # `WARP_FILL_OUTLIERS` on every call. Without it `remap` is given
    # `BORDER_TRANSPARENT`, which *skips* destination pixels whose source falls
    # outside the image — so they keep whatever the freshly allocated `dst`
    # happened to contain. That is reproducible within one process and is not
    # defined behaviour, and it is worth 75 levels at the outermost radius.
    fill = cv2.WARP_FILL_OUTLIERS
    dump("rm_linearpolar", cv2.linearPolar(gray, centre, maxr, cv2.INTER_LINEAR + fill))
    dump("rm_warppolar", cv2.warpPolar(gray, (80, 120), centre, maxr,
                                       cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR + fill))
    dump("rm_warppolar_log", cv2.warpPolar(gray, (80, 120), centre, maxr,
                                           cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG + fill))

    # Corner refinement from integer seeds.
    seeds = np.array([[40.0, 30.0], [80.0, 60.0], [120.0, 45.0]], np.float32).reshape(-1, 1, 2)
    dump("in_csp_seeds", seeds.reshape(-1, 2).astype(np.float64))
    ref = cv2.cornerSubPix(gray, seeds.copy(), (5, 5), (-1, -1),
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    dump("rm_cornersubpix", ref.reshape(-1, 2).astype(np.float64))

    # Shape matching on two contours of the same object at different scales.
    a = np.zeros((80, 80), np.uint8)
    cv2.ellipse(a, (40, 40), (25, 15), 20, 0, 360, 255, -1)
    b = np.zeros((80, 80), np.uint8)
    cv2.ellipse(b, (40, 40), (18, 11), 20, 0, 360, 255, -1)
    ca = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    cb = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    dump("in_ms_a", ca.reshape(-1, 2).astype(np.float64))
    dump("in_ms_b", cb.reshape(-1, 2).astype(np.float64))
    dump("rm_matchshapes", np.array([[cv2.matchShapes(ca, cb, m, 0.0)
                                      for m in (cv2.CONTOURS_MATCH_I1,
                                                cv2.CONTOURS_MATCH_I2,
                                                cv2.CONTOURS_MATCH_I3)]]))

    # Watershed from a seeded marker image.
    mk = np.zeros((h, w), np.int32)
    mk[10:14, 10:14] = 1
    mk[100:104, 140:144] = 2
    mk[58:62, 78:82] = 3
    dump("in_ws_markers", mk)
    ws = mk.copy()
    cv2.watershed(bgr, ws)
    dump("rm_watershed", ws)

    dump("rm_blendlinear", cv2.blendLinear(
        gray.astype(np.float32), np.roll(gray, 17).astype(np.float32),
        np.linspace(0, 1, w, dtype=np.float32)[None, :].repeat(h, 0),
        np.linspace(1, 0, w, dtype=np.float32)[None, :].repeat(h, 0)))

    # ---- core ----
    mm = cv2.minMaxLoc(gray)
    dump("rm_minmaxidx", np.array([[mm[0], mm[1]]]))
    mixed = np.zeros((h, w, 3), np.uint8)
    cv2.mixChannels([bgr], [mixed], [0, 2, 1, 1, 2, 0])
    dump("rm_mixchannels", mixed)
    ins = bgr.copy()
    cv2.insertChannel(np.full((h, w), 77, np.uint8), ins, 1)
    dump("rm_insertchannel", ins)

    # ---- video: background subtraction ----
    # A static scene with a moving square: after a few frames the model has
    # learnt the background and the square is the only foreground.
    frames = []
    for i in range(8):
        f = gray.copy()
        cv2.rectangle(f, (20 + 8 * i, 40), (40 + 8 * i, 60), 255, -1)
        frames.append(f)
    for i, f in enumerate(frames):
        dump(f"in_bg{i}", f)
    for name, ctor in [("mog2", lambda: cv2.createBackgroundSubtractorMOG2(500, 16.0, False)),
                       ("knn", lambda: cv2.createBackgroundSubtractorKNN(500, 400.0, False))]:
        sub = ctor()
        for f in frames[:-1]:
            sub.apply(f, learningRate=-1)
        dump(f"rm_bg_{name}", sub.apply(frames[-1], learningRate=-1))


def flann_and_hog(d, gray, bgr, dump):
    """flann nearest-neighbour search and the HOG descriptor."""
    import numpy as np
    import cv2

    # ---- flann ----
    # A linear index answers exactly, so the neighbours and distances are
    # determined; a kd-tree on the same data must agree with it.
    rs = np.random.RandomState(53)
    pts = rs.randn(200, 8).astype(np.float32)
    qs = rs.randn(12, 8).astype(np.float32)
    dump("in_fln_pts", pts.astype(np.float64))
    dump("in_fln_q", qs.astype(np.float64))
    # `FLANN_INDEX_LINEAR` is not exported by the python binding; its
    # enum value is 0. A linear index answers exactly.
    idx = cv2.flann_Index(pts, dict(algorithm=0))
    nb, dist = idx.knnSearch(qs, 3)
    dump("fln_linear_idx", nb.astype(np.float64))
    dump("fln_linear_dist", dist.astype(np.float64))

    # ---- HOG ----
    # One window's descriptor: fixed geometry, fixed normalisation, no
    # detector involved.
    win = cv2.resize(gray, (64, 128), interpolation=cv2.INTER_LINEAR)
    dump("in_hog_win", win)
    hog = cv2.HOGDescriptor()
    dump("od_hog", hog.compute(win).reshape(-1, 1).astype(np.float64))


def features(d, gray, bgr, dump):
    """features2d — nothing in this module had a differential check."""
    import numpy as np
    import cv2

    # ---- FAST ----
    # A fixed decision rule on pixel values: exactly comparable.
    for thr, nms in [(30, True), (30, False), (10, True)]:
        det = cv2.FastFeatureDetector_create(thr, nms)
        kp = det.detect(gray, None)
        arr = np.array(sorted([(k.pt[0], k.pt[1], k.response) for k in kp]), np.float64)
        dump(f"f2_fast_{thr}_{int(nms)}", arr if len(arr) else np.zeros((0, 3)))

    # ---- brute-force Hamming matching ----
    # On descriptors handed to it, so no detector is involved and the answer is
    # determined: for each query row, the train row at least Hamming distance.
    rs = np.random.RandomState(41)
    qd = rs.randint(0, 256, (20, 32)).astype(np.uint8)
    td = rs.randint(0, 256, (35, 32)).astype(np.uint8)
    dump("in_bf_q", qd)
    dump("in_bf_t", td)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    ms = bf.match(qd, td)
    ms = sorted(ms, key=lambda m: m.queryIdx)
    dump("f2_bf_match", np.array([[m.queryIdx, m.trainIdx, m.distance] for m in ms], np.float64))
    knn = bf.knnMatch(qd, td, k=2)
    dump("f2_bf_knn", np.array([[m.trainIdx, m.distance] for row in knn for m in row], np.float64))

    # ---- ORB descriptors on supplied keypoints ----
    # `compute` takes the keypoints, so this isolates the descriptor from the
    # detector: it is the BRIEF sampling pattern and the orientation that are
    # being compared, not the corner finder.
    pts = [(40.0, 30.0), (80.0, 60.0), (120.0, 45.0), (60.0, 90.0), (100.0, 75.0)]
    dump("in_orb_pts", np.array(pts, np.float64))
    orb = cv2.ORB_create()
    # Two cases. With the default angle of -1 `compute` samples the pattern
    # essentially unrotated — it does *not* compute an orientation, that only
    # happens inside `detectAndCompute`. With explicit angles the rotation path
    # is exercised.
    for tag, angles in [("plain", [-1.0] * 5), ("rot", [0.0, 45.0, 90.0, 180.0, 275.0])]:
        kps = [cv2.KeyPoint(x, y, 31.0, a) for (x, y), a in zip(pts, angles)]
        kps2, desc = orb.compute(gray, kps)
        dump(f"f2_orb_{tag}_angles", np.array([[k.angle] for k in kps2], np.float64))
        dump(f"f2_orb_{tag}_desc", desc if desc is not None else np.zeros((0, 32), np.uint8))


def core_and_video(d, gray, bgr, dump):
    """core functions and the rest of `video`, neither of which was covered."""
    import numpy as np
    import cv2

    h, w = gray.shape
    f32 = np.fromfile(f"{d}/in_f32.bin", np.float32).reshape(h, w)

    # ---- core ----
    dump("co_dct", cv2.dct(f32.astype(np.float64)))
    dump("co_idct", cv2.idct(cv2.dct(f32.astype(np.float64))))
    dump("co_dct_rows", cv2.dct(f32.astype(np.float64), flags=cv2.DCT_ROWS))
    dump("co_psnr", np.array([[cv2.PSNR(gray, np.roll(gray, 17))]]))
    nz = cv2.findNonZero((gray > 200).astype(np.uint8))
    dump("co_findnonzero", (nz.reshape(-1, 2) if nz is not None
                            else np.zeros((0, 2), np.int32)).astype(np.float64))

    # `batchDistance` on two small float sets, L2 without the square root.
    qa = f32[:8, :4].copy()
    qb = np.roll(f32, 5, 1)[:6, :4].copy()
    dump("in_bd_a", qa)
    dump("in_bd_b", qb)
    dist, idx = cv2.batchDistance(qa, qb, cv2.CV_32F, K=1, normType=cv2.NORM_L2)
    dump("co_bd_dist", dist.reshape(-1, 1).astype(np.float64))
    dump("co_bd_idx", idx.reshape(-1, 1).astype(np.float64))

    # Rotated-rectangle intersection: pure geometry.
    r1 = ((60.0, 50.0), (40.0, 30.0), 0.0)
    r2 = ((70.0, 55.0), (30.0, 40.0), 25.0)
    ret, pts = cv2.rotatedRectangleIntersection(r1, r2)
    dump("co_rri_type", np.array([[float(ret)]]))
    dump("co_rri_pts", (pts.reshape(-1, 2) if pts is not None
                        else np.zeros((0, 2), np.float32)).astype(np.float64))

    poly1 = np.array([[10, 10], [70, 20], [60, 80], [15, 60]], np.float32)
    poly2 = np.array([[40, 5], [90, 40], [50, 90], [20, 45]], np.float32)
    dump("in_icc_p1", poly1.astype(np.float64))
    dump("in_icc_p2", poly2.astype(np.float64))
    area, inter = cv2.intersectConvexConvex(poly1, poly2)
    dump("co_icc_area", np.array([[float(area)]]))
    dump("co_icc_pts", np.array(inter, np.float64).reshape(-1, 2))

    # A cubic with known roots.
    coeffs = np.array([-6.0, 11.0, -6.0, 1.0])   # (x-1)(x-2)(x-3)
    _, roots = cv2.solvePoly(coeffs)
    dump("co_solvepoly", np.array(roots, np.float64).reshape(-1, 2))

    # k-means with the labels supplied, so nothing depends on an RNG.
    pts3 = np.vstack([
        np.random.RandomState(21).randn(30, 2) * 0.4 + c
        for c in ([-2.0, -2.0], [2.0, -2.0], [0.0, 2.5])]).astype(np.float32)
    lab0 = np.repeat(np.arange(3), 30).astype(np.int32).reshape(-1, 1)
    dump("in_km_pts", pts3.astype(np.float64))
    dump("in_km_labels", lab0)
    crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-8)
    compactness, labels, centers = cv2.kmeans(
        pts3, 3, lab0.copy(), crit, 1, cv2.KMEANS_USE_INITIAL_LABELS)
    dump("co_km_centers", centers.astype(np.float64))
    dump("co_km_labels", labels.reshape(-1, 1).astype(np.float64))
    dump("co_km_compactness", np.array([[float(compactness)]]))

    # ---- video ----
    # Two frames related by a known translation, so the flow has a truth.
    tex = cv2.GaussianBlur(np.random.RandomState(33).randint(0, 256, (h + 16, w + 16))
                           .astype(np.uint8), (5, 5), 0)
    fa = np.ascontiguousarray(tex[4:4 + h, 4:4 + w])
    fb = np.ascontiguousarray(tex[4 + 2:4 + 2 + h, 4 + 3:4 + 3 + w])
    dump("in_of_a", fa)
    dump("in_of_b", fb)
    flow = cv2.calcOpticalFlowFarneback(fa, fb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dump("vi_farneback", flow)

    # meanShift / CamShift on a back-projection with one clear mode.
    prob = np.zeros((h, w), np.uint8)
    cv2.circle(prob, (95, 70), 18, 255, -1)
    dump("in_ms_prob", prob)
    crit2 = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1.0)
    _, win = cv2.meanShift(prob, (60, 40, 40, 40), crit2)
    dump("vi_meanshift", np.array([[float(v) for v in win]]))
    box, win2 = cv2.CamShift(prob, (60, 40, 40, 40), crit2)
    dump("vi_camshift_win", np.array([[float(v) for v in win2]]))
    dump("vi_camshift_box", np.array([[box[0][0], box[0][1], box[1][0], box[1][1], box[2]]]))

    # A Kalman filter run: pure linear algebra, so it must match exactly.
    kf = cv2.KalmanFilter(4, 2, 0)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                    [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[0], [0], [0], [0]], np.float32)
    track = []
    for i in range(12):
        pred = kf.predict()
        z = np.array([[3.0 * i + 1.0], [2.0 * i - 0.5]], np.float32)
        corr = kf.correct(z)
        track.append([float(pred[0, 0]), float(pred[1, 0]),
                      float(corr[0, 0]), float(corr[1, 0])])
    dump("vi_kalman", np.array(track, np.float64))


def last_mile(d, gray, bgr, dump):
    """Response calibration, Mantiuk, the seam finders and the detail helpers."""
    import numpy as np
    import cv2

    h, w = gray.shape
    bracket = [np.fromfile(f"{d}/in_bracket{i}.bin", np.uint8).reshape(h, w, 3) for i in range(3)]
    times = np.array([1 / 30.0, 1 / 8.0, 1 / 2.0], np.float32)

    # ---- response calibration ----
    # Debevec's fit is seeded from randomly chosen sample sites unless the
    # random flag is off, so pin it: a response curve that moves between runs
    # cannot be a reference.
    cd = cv2.createCalibrateDebevec(70, 10.0, False)
    dump("cal_debevec", cd.process(bracket, times))
    cr = cv2.createCalibrateRobertson(30, 0.01)
    dump("cal_robertson", cr.process(bracket, times))

    hdrimg = np.fromfile(f"{d}/in_hdr.bin", np.float32).reshape(h, w, 3)
    dump("tm_mantiuk", cv2.createTonemapMantiuk(1.0, 0.7, 1.0).process(hdrimg))

    # ---- seam finders ----
    # Two overlapping views of the same texture, shifted, so there is a genuine
    # best cut rather than a tie everywhere.
    left = np.dstack([gray[:60, :80]] * 3).copy()
    right = np.dstack([np.roll(gray[:60, :80], 5, 1)] * 3).copy()
    masks = [np.full((60, 80), 255, np.uint8) for _ in range(2)]
    corners = [(0, 0), (40, 0)]
    dump("in_seam_l", left)
    dump("in_seam_r", right)
    for name, finder in [("dp", cv2.detail_DpSeamFinder("COLOR")),
                         ("gc", cv2.detail_GraphCutSeamFinder("COST_COLOR"))]:
        out = finder.find([left.astype(np.float32), right.astype(np.float32)],
                          corners, [m.copy() for m in masks])
        for i, m in enumerate(out):
            arr = np.array(m.get() if hasattr(m, "get") else m)
            dump(f"sm_{name}{i}", arr)
    # `detail_VoronoiSeamFinder.find` segfaults in the 4.13.0 wheel, so it has
    # no fixture. Its unit tests are the only check it can have.

    # ---- end-to-end registration ----
    # Two views of one texture related by a known homography. The feature
    # detectors differ between the two libraries, so the *matches* will not be
    # the same set — but the geometry they recover from those matches has to
    # be, and that is what a caller depends on.
    rs = np.random.RandomState(31)
    wide = rs.randint(28, 228, (120, 200)).astype(np.uint8)
    reg_a = np.ascontiguousarray(wide[:, :160])
    Hreg = np.array([[1.0, 0.06, -30.0], [0.03, 1.02, -4.0], [0.0002, 0.0001, 1.0]])
    reg_b = cv2.warpPerspective(wide, np.linalg.inv(Hreg), (160, 120), flags=cv2.INTER_NEAREST)
    dump("in_reg_a", reg_a)
    dump("in_reg_b", reg_b)
    st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    st.setPanoConfidenceThresh(0.3)
    status = st.estimateTransform([cv2.cvtColor(reg_a, cv2.COLOR_GRAY2BGR),
                                   cv2.cvtColor(reg_b, cv2.COLOR_GRAY2BGR)])
    dump("reg_status", np.array([[float(status)]]))
    if status == 0:
        cams = st.cameras()
        dump("reg_focals", np.array([[float(c.focal) for c in cams]]))
        rel = np.array(cams[0].R).T @ np.array(cams[1].R)
        angle = float(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
        dump("reg_angle", np.array([[angle]]))

    # ---- detail geometry helpers ----
    rmats = [cv2.Rodrigues(np.array([0.02 * i, 0.3 * i, 0.01 * i]))[0].astype(np.float32)
             for i in range(4)]
    dump("in_wave_R", np.array(rmats, np.float64).reshape(-1, 3))
    for kind, nm in [(cv2.detail.WAVE_CORRECT_HORIZ, "horiz"),
                     (cv2.detail.WAVE_CORRECT_VERT, "vert")]:
        out = cv2.detail.waveCorrect([r.copy() for r in rmats], kind)
        dump(f"wc_{nm}", np.array(out, np.float64).reshape(-1, 3))

    sizes = [(80, 60), (80, 60), (70, 50)]
    cs = [(0, 0), (40, 0), (10, 30)]
    dump("dt_result_roi", np.array([[float(v) for v in cv2.detail.resultRoi(
        cs, [np.zeros((s[1], s[0], 3), np.uint8) for s in sizes])]]))
    dump("dt_result_tl", np.array([[float(v) for v in cv2.detail.resultTl(cs)]]))
    dump("dt_result_roi_int", np.array([[float(v) for v in cv2.detail.resultRoiIntersection(
        cs, [(s[0], s[1]) for s in sizes])]]))

    # `focalsFromHomography` writes through output reference parameters that
    # the python binding drops, so it returns None and cannot be a fixture.
    # A homography from a known rotating camera gives the port a ground truth
    # anyway: for K with f = 420, the recovered focal must be 420.
    # The principal point must be at the origin: upstream's closed form assumes
    # `K = diag(f, f, 1)`, and a homography carrying a `cx`/`cy` does not
    # satisfy the relations it inverts.
    K = np.array([[420.0, 0.0, 0.0], [0.0, 420.0, 0.0], [0.0, 0.0, 1.0]])
    R = cv2.Rodrigues(np.array([0.03, 0.25, 0.02]))[0]
    dump("in_focal_H", K @ R @ np.linalg.inv(K))


def boards_stitching_gapi(d, gray, bgr, dump):
    """ChArUco, the stitching stages above the warpers, and the rest of G-API."""
    import numpy as np
    import cv2

    h, w = gray.shape
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # ---- ChArUco and grid boards ----
    # The geometry is a fixed table: the inner chessboard corners and the
    # marker object points are exactly determined by the board's dimensions.
    cb = cv2.aruco.CharucoBoard((5, 4), 0.04, 0.02, adict)
    dump("cu_ids", cb.getIds().astype(np.float64).reshape(-1, 1))
    dump("cu_chess", cb.getChessboardCorners().astype(np.float64))
    dump("cu_obj", np.array(cb.getObjPoints(), np.float64).reshape(-1, 3))
    dump("cu_image", cb.generateImage((400, 320), marginSize=10))

    gb = cv2.aruco.GridBoard((3, 2), 0.04, 0.01, adict)
    dump("gb_ids", gb.getIds().astype(np.float64).reshape(-1, 1))
    dump("gb_obj", np.array(gb.getObjPoints(), np.float64).reshape(-1, 3))
    dump("gb_image", gb.generateImage((300, 220), marginSize=8))

    # ---- stitching: exposure, seams, blending ----
    # Two overlapping tiles at different exposures. A compensator that works
    # brings their gains to a ratio matching the brightness ratio; one that
    # does not returns 1 and 1.
    tile = (gray[:60, :80].astype(np.float32))
    imgs = [np.clip(np.dstack([tile] * 3) * s, 0, 255).astype(np.uint8) for s in (0.7, 1.15)]
    masks = [np.full((60, 80), 255, np.uint8) for _ in range(2)]
    corners = [(0, 0), (40, 0)]
    for i, im in enumerate(imgs):
        dump(f"in_st_img{i}", im)
    gc = cv2.detail_GainCompensator()
    gc.feed(corners, imgs, masks)
    dump("sti_gains", np.array([float(g[0][0]) for g in gc.getMatGains([])]).reshape(1, -1))

    bl = cv2.detail_FeatherBlender()
    bl.prepare((0, 0, 120, 60))
    for im, mk, c in zip(imgs, masks, corners):
        bl.feed(im.astype(np.int16), mk, c)
    dst, dstm = bl.blend(None, None)
    dump("sti_feather", np.clip(dst, 0, 255).astype(np.uint8))
    dump("sti_feather_mask", dstm)

    mb = cv2.detail_MultiBandBlender()
    mb.setNumBands(3)
    mb.prepare((0, 0, 120, 60))
    for im, mk, c in zip(imgs, masks, corners):
        mb.feed(im.astype(np.int16), mk, c)
    dst2, _ = mb.blend(None, None)
    dump("sti_multiband", np.clip(dst2, 0, 255).astype(np.uint8))

    # `detail_VoronoiSeamFinder.find` segfaults in the cv2 4.13.0 wheel, so
    # there is no reference to compare against. The seam finders are checked by
    # their own unit tests instead; see the README.

    # ---- the rest of G-API ----
    def graph1(build, arg):
        gi = cv2.GMat()
        return cv2.GComputation(cv2.GIn(gi), cv2.GOut(build(gi))).apply(cv2.gin(arg))

    dump("gp_median", graph1(lambda g: cv2.gapi.medianBlur(g, 5), gray))
    dump("gp_canny", graph1(lambda g: cv2.gapi.Canny(g, 50.0, 150.0), gray))
    dump("gp_erode", graph1(lambda g: cv2.gapi.erode3x3(g), gray))
    dump("gp_dilate", graph1(lambda g: cv2.gapi.dilate3x3(g), gray))
    dump("gp_not", graph1(lambda g: cv2.gapi.bitwise_not(g), gray))
    gi = cv2.GMat()
    b, g_, rr = cv2.gapi.split3(gi)
    dump("gp_split_g", cv2.GComputation(cv2.GIn(gi), cv2.GOut(g_)).apply(cv2.gin(bgr)))
    dump("gp_merge", cv2.GComputation(cv2.GIn(gi), cv2.GOut(
        cv2.gapi.merge3(*cv2.gapi.split3(gi)))).apply(cv2.gin(bgr)))


def models_and_boards(d, gray, bgr, dump):
    """The classifiers, the marker boards, the warpers and the remaining photo.

    Two classifiers trained by different procedures will not have the same
    parameters, so the check here is the one a caller actually depends on: on
    data whose answer is unambiguous, do they *predict* the same labels?
    """
    import numpy as np
    import cv2

    h, w = gray.shape

    # ---- classifiers ----
    # Three well-separated clouds. Anything that has learned the problem at all
    # agrees here; anything that disagrees is broken, not merely different.
    rs = np.random.RandomState(17)
    centres = [(-3.0, -3.0), (3.0, -2.5), (0.0, 3.5)]
    X = np.vstack([rs.randn(50, 2) * 0.5 + np.array(c) for c in centres]).astype(np.float32)
    y = np.hstack([np.full(50, i) for i in range(3)]).astype(np.int32)
    q = np.array([[-2.8, -3.2], [3.1, -2.2], [0.2, 3.6], [-3.4, -2.6],
                  [2.6, -2.9], [-0.3, 3.1], [3.4, -2.6], [-2.5, -3.5]], np.float32)
    dump("in_cls_x", X.astype(np.float64))
    dump("in_cls_y", y.reshape(-1, 1))
    dump("in_cls_q", q.astype(np.float64))

    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(10.0)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
    svm.train(X, cv2.ml.ROW_SAMPLE, y)
    dump("cls_svm", svm.predict(q)[1].astype(np.float64))

    dt = cv2.ml.DTrees_create()
    dt.setMaxDepth(6)
    dt.setCVFolds(0)
    dt.setMinSampleCount(2)
    dt.train(X, cv2.ml.ROW_SAMPLE, y)
    dump("cls_dtree", dt.predict(q)[1].astype(np.float64))

    rt = cv2.ml.RTrees_create()
    rt.setMaxDepth(6)
    rt.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 50, 0))
    rt.train(X, cv2.ml.ROW_SAMPLE, y)
    dump("cls_rtrees", rt.predict(q)[1].astype(np.float64))

    # Boost and logistic regression are two-class only.
    X2 = X[:100]
    y2 = y[:100].astype(np.int32)
    q2 = q[[0, 1, 3, 4, 6]]
    dump("in_cls_x2", X2.astype(np.float64))
    dump("in_cls_y2", y2.reshape(-1, 1))
    dump("in_cls_q2", q2.astype(np.float64))
    bo = cv2.ml.Boost_create()
    bo.setBoostType(cv2.ml.BOOST_REAL)
    bo.setWeakCount(20)
    bo.setMaxDepth(2)
    bo.train(X2, cv2.ml.ROW_SAMPLE, y2)
    dump("cls_boost", bo.predict(q2)[1].astype(np.float64))

    lr = cv2.ml.LogisticRegression_create()
    lr.setLearningRate(0.1)
    lr.setIterations(1000)
    lr.setTrainMethod(cv2.ml.LogisticRegression_BATCH)
    lr.train(X2, cv2.ml.ROW_SAMPLE, y2.astype(np.float32))
    dump("cls_logreg", lr.predict(q2)[1].astype(np.float64))

    em = cv2.ml.EM_create()
    em.setClustersNumber(3)
    em.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_DIAGONAL)
    em.trainEM(X)
    # Component *numbering* is arbitrary, so record which query points share a
    # component rather than which number they got.
    # `predict2` returns `(Vec2d(log-likelihood, index), posteriors)`;
    # `[1]` is the posterior vector, not the label.
    lab = np.array([int(em.predict2(qq.reshape(1, -1))[0][1]) for qq in q])
    same = (lab[:, None] == lab[None, :]).astype(np.float64)
    dump("cls_em_same", same)

    ann = cv2.ml.ANN_MLP_create()
    ann.setLayerSizes(np.array([2, 8, 3], np.int32))
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1, 0.1)
    ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 2000, 1e-6))
    onehot = np.zeros((len(y), 3), np.float32)
    onehot[np.arange(len(y)), y] = 1.0
    ann.train(X, cv2.ml.ROW_SAMPLE, onehot)
    dump("cls_ann", np.array([[float(np.argmax(ann.predict(qq.reshape(1, -1))[1]))] for qq in q]))

    # ---- circles grid ----
    # Rendered from a known geometry, like the chessboard.
    # A 5x4 grid: cv2's clustering finder rejects a 4x3 outright, whatever the
    # spacing — it needs enough blobs to fit a grid to.
    pat = (5, 4)
    sp = 60
    gridimg = np.full((100 + pat[1] * sp, 100 + pat[0] * sp), 255, np.uint8)
    truth = []
    for r in range(pat[1]):
        for c in range(pat[0]):
            cx, cy = 60 + c * sp, 60 + r * sp
            cv2.circle(gridimg, (cx, cy), 12, 0, -1)
            truth.append((cx, cy))
    dump("in_circles", gridimg)
    ok, centers = cv2.findCirclesGrid(gridimg, pat, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
    dump("cg_found", np.array([[float(bool(ok))]]))
    dump("cg_centers", (centers.reshape(-1, 2) if ok else np.zeros((0, 2), np.float32)))
    dump("cg_truth", np.array(truth, np.float64))

    # ---- ArUco ----
    # The dictionary bytes are a fixed table, so they must match exactly.
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for mid in (0, 1, 7, 23, 49):
        dump(f"ar_marker{mid}", cv2.aruco.generateImageMarker(adict, mid, 60))
    scene = np.full((240, 320), 255, np.uint8)
    m = cv2.aruco.generateImageMarker(adict, 23, 80)
    scene[70:150, 110:190] = m
    dump("in_aruco", scene)
    corners, ids, _ = cv2.aruco.ArucoDetector(adict).detectMarkers(scene)
    dump("ar_ids", (np.array(ids, np.float64).reshape(-1, 1) if ids is not None
                    else np.zeros((0, 1), np.float64)))
    dump("ar_corners", (np.array(corners, np.float64).reshape(-1, 2) if ids is not None
                        else np.zeros((0, 2), np.float64)))

    # ---- stitching warpers ----
    # Pure geometry with no estimation in it, so these are exactly comparable.
    K = np.array([[300.0, 0.0, 79.5], [0.0, 300.0, 59.5], [0.0, 0.0, 1.0]], np.float32)
    R = cv2.Rodrigues(np.array([0.05, 0.30, -0.02]))[0].astype(np.float32)
    dump("in_warp_K", K.astype(np.float64))
    dump("in_warp_R", R.astype(np.float64))
    for name, scale in [("spherical", 300.0), ("cylindrical", 300.0), ("plane", 300.0)]:
        wp = {"spherical": cv2.PyRotationWarper("spherical", scale),
              "cylindrical": cv2.PyRotationWarper("cylindrical", scale),
              "plane": cv2.PyRotationWarper("plane", scale)}[name]
        pts = np.array([[0.0, 0.0], [159.0, 0.0], [0.0, 119.0], [159.0, 119.0],
                        [80.0, 60.0]], np.float32)
        out = np.array([wp.warpPoint(tuple(p), K, R) for p in pts], np.float64)
        dump(f"wa_{name}_pts", out)
        roi = wp.warpRoi((160, 120), K, R)
        dump(f"wa_{name}_roi", np.array([[float(v) for v in roi]]))

    # ---- G-API ----
    # A graph should produce exactly what the equivalent call chain produces.
    # G-API exists to fuse and schedule, not to compute differently, so any
    # divergence here is a bug in the graph and not a design choice.
    # The python binding wants GIn/GOut even for a single input.
    def graph1(build, arg):
        gi = cv2.GMat()
        return cv2.GComputation(cv2.GIn(gi), cv2.GOut(build(gi))).apply(cv2.gin(arg))

    dump("gp_gray_blur", graph1(lambda g: cv2.gapi.blur(cv2.gapi.BGR2Gray(g), (5, 5)), bgr))
    # `gapi.threshold` takes GScalars, not doubles — a graph parameter can
    # itself be a node, which is the point of the type.
    dump("gp_gray_thresh",
         graph1(lambda g: cv2.gapi.threshold(cv2.gapi.BGR2Gray(g),
                                             cv2.GScalar(128.0),
                                             cv2.GScalar(255.0),
                                             cv2.THRESH_BINARY), bgr))
    dump("gp_chain",
         graph1(lambda g: cv2.gapi.Sobel(cv2.gapi.gaussianBlur(g, (5, 5), 1.4),
                                         cv2.CV_8U, 1, 0), gray))
    ga, gb = cv2.GMat(), cv2.GMat()
    dump("gp_add", cv2.GComputation(cv2.GIn(ga, gb), cv2.GOut(cv2.gapi.add(ga, gb)))
         .apply(cv2.gin(gray, np.roll(gray, 17))))

    # ---- the rest of photo ----
    dmg = gray.copy()
    hole = np.zeros((h, w), np.uint8)
    cv2.rectangle(hole, (60, 40), (80, 55), 255, -1)
    dmg[hole > 0] = 0
    dump("in_ns_img", dmg)
    dump("in_ns_mask", hole)
    dump("ph_inpaint_ns2", cv2.inpaint(dmg, hole, 3, cv2.INPAINT_NS))

    # `calculateShift` is deliberately not a fixture. On this image cv2 returns
    # (-25, -24) for a true shift of (-2, 3): the pattern is periodic, so the
    # median bitmaps correlate about equally well at many offsets and the answer
    # is whichever the search happened to reach first. A reference the reference
    # library gets wrong tests nothing. `computeBitmaps` *is* well defined --
    # threshold at the median, exclude a band around it -- and it is the piece
    # the alignment is built on.
    mtb = cv2.createAlignMTB()
    tb = np.zeros((h, w), np.uint8)
    eb = np.zeros((h, w), np.uint8)
    mtb.computeBitmaps(gray, tb, eb)
    dump("ph_mtb_tb", tb)
    dump("ph_mtb_eb", eb)


def photo_and_shapes(d, gray, bgr, dump):
    """photo (cloning, tonemapping, decolor) and the shape detectors."""
    import numpy as np
    import cv2

    h, w = gray.shape

    # ---- seamless cloning ----
    # A patch pasted into a background with a different mean: the whole point of
    # Poisson cloning is that the *gradients* transfer and the level does not,
    # so a fixture where the two match would pass with the solve stubbed out.
    src = np.dstack([np.full((h, w), 60, np.uint8),
                     np.full((h, w), 160, np.uint8),
                     np.full((h, w), 220, np.uint8)])
    src[40:80, 50:110] = bgr[40:80, 50:110]
    dstb = np.dstack([(gray // 2 + 90).astype(np.uint8)] * 3)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (80, 60), 28, 255, -1)
    dump("in_sc_src", src)
    dump("in_sc_dst", dstb)
    dump("in_sc_mask", mask)
    # **Every call gets its own copy of the mask.** `cv2.seamlessClone` erodes
    # the caller's mask *in place* — `Cloning::computeDerivatives` calls
    # `erode(binaryMask, binaryMask, ...)` on a submatrix view of it — so a
    # second call on the same array sees a mask three pixels smaller, and a
    # third sees it smaller again. Sharing one mask across these three calls
    # made every fixture after the first a reference for a different problem.
    for code, nm in [(cv2.NORMAL_CLONE, "normal"), (cv2.MIXED_CLONE, "mixed"),
                     (cv2.MONOCHROME_TRANSFER, "mono")]:
        dump(f"sc_{nm}", cv2.seamlessClone(src, dstb, mask.copy(), (80, 60), code))

    # Keyword arguments, not positional: the third parameter is `dst`, so
    # `colorChange(src, mask, 1.5, 0.5, 1.0)` silently passes 1.5 as the output
    # array and shifts every multiplier by one.
    dump("ph_color_change",
         cv2.colorChange(src, mask.copy(), red_mul=1.5, green_mul=0.5, blue_mul=1.0))

    # A small clone whose mask *fills its bounding box*. This is the case that
    # discriminates how the mask erosion is applied: upstream erodes a
    # submatrix view, so OpenCV reads the surrounding zeros from the parent and
    # three iterations wipe a 6x6 mask entirely, leaving the destination
    # untouched. Eroding the cropped window on its own leaves it fully set and
    # transfers the whole patch — the opposite result.
    rs2 = np.random.RandomState(2)
    tiny_dst = np.dstack([np.tile(np.linspace(40, 200, 14), (14, 1))] * 3).astype(np.uint8)
    tiny_src = np.dstack([rs2.randint(0, 256, (14, 14))] * 3).astype(np.uint8)
    tiny_mask = np.zeros((14, 14), np.uint8)
    tiny_mask[4:10, 4:10] = 255
    dump("in_tiny_src", tiny_src)
    dump("in_tiny_dst", tiny_dst)
    dump("in_tiny_mask", tiny_mask)
    for code, nm in [(cv2.NORMAL_CLONE, "normal"), (cv2.MIXED_CLONE, "mixed")]:
        dump(f"tc_{nm}", cv2.seamlessClone(tiny_src, tiny_dst, tiny_mask.copy(), (7, 7), code))

    # And one where the erosion leaves a real mask behind, so the MIXED
    # selection actually runs: a 20x20 square erodes to 14x14.
    rs3 = np.random.RandomState(5)
    med_dst = np.dstack([np.tile(np.linspace(40, 200, 40), (40, 1))] * 3).astype(np.uint8)
    med_src = np.dstack([rs3.randint(0, 256, (40, 40))] * 3).astype(np.uint8)
    med_mask = np.zeros((40, 40), np.uint8)
    med_mask[10:30, 10:30] = 255
    dump("in_med_src", med_src)
    dump("in_med_dst", med_dst)
    dump("in_med_mask", med_mask)
    for code, nm in [(cv2.NORMAL_CLONE, "normal"), (cv2.MIXED_CLONE, "mixed")]:
        dump(f"mc_{nm}", cv2.seamlessClone(med_src, med_dst, med_mask.copy(), (20, 20), code))
    grey_out, colour_boost = cv2.decolor(bgr)
    dump("ph_decolor", grey_out)
    dump("ph_decolor_boost", colour_boost)

    # ---- tonemapping ----
    # A radiance map with a real dynamic range, not a scaled 8-bit image: a
    # tonemapper on a low-range input is indistinguishable from a gamma curve.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = (0.02 + 8.0 * np.exp(-(((xx - 100) / 40) ** 2 + ((yy - 50) / 30) ** 2))
           + 0.3 * (gray.astype(np.float32) / 255.0))
    hdrimg = np.dstack([rad, rad * 1.15, rad * 0.85]).astype(np.float32)
    dump("in_hdr", hdrimg)
    dump("tm_gamma", cv2.createTonemap(2.2).process(hdrimg))
    dump("tm_drago", cv2.createTonemapDrago(1.0, 1.0, 0.85).process(hdrimg))
    dump("tm_reinhard", cv2.createTonemapReinhard(1.0, 0.0, 0.0, 0.0).process(hdrimg))

    # Exposure fusion over a bracket, which needs no response curve.
    bracket = [np.clip(hdrimg * s * 255.0, 0, 255).astype(np.uint8) for s in (0.05, 0.2, 0.8)]
    for i, b in enumerate(bracket):
        dump(f"in_bracket{i}", b)
    # Note the defaults: contrast 1, saturation 1, exposure *0*.
    dump("hdr_mertens", cv2.createMergeMertens().process(bracket))
    times = np.array([1 / 30.0, 1 / 8.0, 1 / 2.0], np.float32)
    dump("hdr_debevec", cv2.createMergeDebevec().process(bracket, times))
    dump("hdr_robertson", cv2.createMergeRobertson().process(bracket, times))

    # ---- pyramids ----
    # On float data and an odd-sized level, which is where a Laplacian pyramid
    # actually exercises pyrUp's dstsize argument.
    pf = (gray.astype(np.float32) / 255.0)
    pf3 = np.dstack([pf, np.roll(pf, 3, 1), np.roll(pf, 5, 0)])
    dump("in_pyr", pf3)
    d1 = cv2.pyrDown(pf3)
    d2 = cv2.pyrDown(d1)
    dump("pyr_down1", d1)
    dump("pyr_down2", d2)
    dump("pyr_up1", cv2.pyrUp(d1))
    dump("pyr_up_sized", cv2.pyrUp(d2, dstsize=(d1.shape[1], d1.shape[0])))
    odd = pf3[:61, :83]
    dump("in_pyr_odd", odd)
    do = cv2.pyrDown(odd)
    dump("pyr_down_odd", do)
    dump("pyr_up_odd", cv2.pyrUp(do, dstsize=(odd.shape[1], odd.shape[0])))

    # ---- LSD ----
    seg = np.zeros((h, w), np.uint8)
    cv2.line(seg, (20, 20), (140, 45), 255, 2)
    cv2.line(seg, (30, 100), (120, 30), 255, 2)
    cv2.line(seg, (10, 70), (150, 78), 255, 2)
    seg = cv2.GaussianBlur(seg, (3, 3), 0)
    dump("in_lsd", seg)
    lines = cv2.createLineSegmentDetector().detect(seg)[0]
    lines = np.zeros((0, 4), np.float32) if lines is None else lines.reshape(-1, 4)
    dump("lsd_lines", lines)

    # ---- generalised Hough ----
    templ = np.zeros((60, 60), np.uint8)
    cv2.rectangle(templ, (18, 12), (42, 46), 255, -1)
    scene = np.zeros((h, w), np.uint8)
    scene[45:79, 70:95] = 255
    dump("in_gh_templ", templ)
    dump("in_gh_scene", scene)
    gh = cv2.createGeneralizedHoughBallard()
    # The default vote threshold (100) rejects a 25x34 rectangle: only its
    # boundary votes, and a small shape simply has fewer edge pixels than that.
    gh.setVotesThreshold(30)
    gh.setTemplate(templ)
    pos = gh.detect(scene)[0]
    pos = np.zeros((0, 4), np.float32) if pos is None else pos.reshape(-1, 4)
    dump("gh_ballard", pos)

    # ---- intelligent scissors ----
    scis = cv2.segmentation.IntelligentScissorsMB()
    scis.applyImage(bgr)
    scis.buildMap((20, 20))
    contour = scis.getContour((120, 90)).reshape(-1, 2)
    dump("is_contour", contour.astype(np.float64))


def calib(d, dump):
    """Reference arrays for the rest of calib3d.

    Every board here is *rendered* by projecting a known geometry through a
    known pose, so there is a ground truth independent of both libraries and a
    disagreement can be attributed rather than argued about.
    """
    import numpy as np
    import cv2

    K = np.array([[520.0, 0.0, 319.5], [0.0, 520.0, 239.5], [0.0, 0.0, 1.0]])
    dist = np.zeros(5)
    pat = (7, 6)
    objp = np.zeros((pat[0] * pat[1], 3), np.float64)
    objp[:, :2] = np.mgrid[0:pat[0], 0:pat[1]].T.reshape(-1, 2) * 0.025

    # Several views of the same board: one view cannot separate focal length
    # from distance, so a single-view "calibration" is only ever a pose.
    views = [
        (np.array([[0.05], [-0.10], [0.02]]), np.array([[-0.08], [-0.06], [0.50]])),
        (np.array([[-0.25], [0.18], [-0.05]]), np.array([[-0.07], [-0.07], [0.55]])),
        (np.array([[0.30], [0.22], [0.08]]), np.array([[-0.09], [-0.05], [0.60]])),
        (np.array([[-0.12], [-0.30], [0.11]]), np.array([[-0.06], [-0.08], [0.52]])),
        (np.array([[0.20], [-0.05], [-0.14]]), np.array([[-0.08], [-0.06], [0.48]])),
    ]
    imgpts = [cv2.projectPoints(objp, rv, tv, K, dist)[0].reshape(-1, 2) for rv, tv in views]
    dump("in_cc_obj", objp)
    for i, ip in enumerate(imgpts):
        dump(f"in_cc_img{i}", ip)

    rms, Kc, dc, rvs, tvs = cv2.calibrateCamera(
        [objp.astype(np.float32)] * len(views),
        [ip.astype(np.float32) for ip in imgpts],
        (640, 480), None, None)
    dump("cc_rms", np.array([[float(rms)]]))
    dump("cc_K", Kc)
    dump("cc_dist", np.array(dc).reshape(1, -1))
    dump("cc_poses", np.hstack([np.array(rvs).reshape(len(views), 3),
                                np.array(tvs).reshape(len(views), 3)]))

    # P3P: exactly three points, so up to four poses satisfy the geometry and
    # the solver must return them all rather than pick one.
    o3 = objp[[0, 6, 35]]
    i3 = cv2.projectPoints(o3, views[0][0], views[0][1], K, dist)[0].reshape(-1, 2)
    dump("in_p3p_obj", o3)
    dump("in_p3p_img", i3)
    n, rv3, tv3 = cv2.solveP3P(o3, i3, K, dist, flags=cv2.SOLVEPNP_P3P)
    dump("p3p_count", np.array([[float(n)]]))
    # The truth the solutions are searched for; ordering is not defined.
    dump("p3p_truth", np.hstack([views[0][0].ravel(), views[0][1].ravel()]).reshape(1, -1))

    # Hand-eye: a fixed camera-to-gripper transform recovered from motion.
    def rt(rvec, tvec):
        return cv2.Rodrigues(np.array(rvec, np.float64))[0], np.array(tvec, np.float64)

    X_R, X_t = rt([0.3, -0.2, 0.15], [0.05, -0.03, 0.10])   # cam2gripper, the answer
    g2b, t2c = [], []
    base = [([0.1, 0.2, 0.3], [0.4, 0.1, 0.5]), ([-0.4, 0.15, 0.25], [0.3, -0.2, 0.45]),
            ([0.25, -0.35, 0.1], [0.2, 0.3, 0.6]), ([0.05, 0.45, -0.2], [0.5, 0.05, 0.35]),
            ([-0.3, -0.1, 0.4], [0.15, 0.25, 0.55])]
    # Target is fixed in the base frame; the camera rides the gripper.
    T_R, T_t = rt([0.2, 0.1, -0.3], [0.6, 0.2, 0.9])         # target2base
    for rv, tv in base:
        gR, gt = rt(rv, tv)                                   # gripper2base
        # target2cam = (gripper2cam) . (base2gripper) . (target2base)
        cR, ct = X_R.T, -X_R.T @ X_t                          # gripper2cam
        bR, bt = gR.T, -gR.T @ gt                             # base2gripper
        R1, t1 = bR @ T_R, bR @ T_t + bt
        g2b.append((gR, gt))
        t2c.append((cR @ R1, cR @ t1 + ct))
    dump("in_he_g2b", np.vstack([np.hstack([R.reshape(1, 9), t.reshape(1, 3)]) for R, t in g2b]))
    dump("in_he_t2c", np.vstack([np.hstack([R.reshape(1, 9), t.reshape(1, 3)]) for R, t in t2c]))
    heR, het = cv2.calibrateHandEye([R for R, _ in g2b], [t for _, t in g2b],
                                    [R for R, _ in t2c], [t for _, t in t2c],
                                    method=cv2.CALIB_HAND_EYE_TSAI)
    dump("he_R", heR)
    dump("he_t", np.array(het).reshape(1, 3))
    dump("he_truth", np.hstack([X_R.reshape(1, 9), X_t.reshape(1, 3)]))

    # A rendered chessboard, so corner detection has a ground truth.
    board = np.full((480, 640), 255, np.uint8)
    sq = 0.025
    # A (7, 6) inner-corner pattern is 8 squares wide by 7 tall; the offset
    # puts the first inner corner at the object-space origin.
    for r in range(7):
        for c in range(8):
            if (r + c) % 2:
                continue
            quad = np.array([[c * sq, r * sq, 0], [(c + 1) * sq, r * sq, 0],
                             [(c + 1) * sq, (r + 1) * sq, 0], [c * sq, (r + 1) * sq, 0]])
            pts = cv2.projectPoints(quad - np.array([sq, sq, 0]), views[0][0], views[0][1],
                                    K, dist)[0].reshape(-1, 2)
            cv2.fillConvexPoly(board, pts.round().astype(np.int32), 0)
    dump("in_board", board)
    found, corners = cv2.findChessboardCorners(board, pat)
    dump("cb_found", np.array([[float(bool(found))]]))
    dump("cb_corners", (corners.reshape(-1, 2) if found else np.zeros((0, 2), np.float32)))


UNVERIFIED_MARKER = None


def dense_flow(d, gray, bgr, dump):
    """DIS and variational refinement, which had never been checked at all."""
    a = np.fromfile(os.path.join(d, "in_flow_a.bin"), np.uint8).reshape(120, 160)
    b = np.fromfile(os.path.join(d, "in_flow_b.bin"), np.uint8).reshape(120, 160)

    # Variational refinement on its own, from a zero field. `calcUV` takes and
    # returns the two flow planes separately.
    for name, fpi, sor in [("base", 5, 5), ("min", 1, 1), ("mid", 2, 3)]:
        vr = cv2.VariationalRefinement_create()
        vr.setFixedPointIterations(fpi)
        vr.setSorIterations(sor)
        u = np.zeros((120, 160), np.float32)
        v = np.zeros((120, 160), np.float32)
        u, v = vr.calcUV(a, b, u, v)
        dump(f"vr_{name}_u", u)
        dump(f"vr_{name}_v", v)

    # Refinement started from a non-zero field, which is how DIS uses it and
    # the only way the warp actually moves.
    vr = cv2.VariationalRefinement_create()
    u0 = np.full((120, 160), 2.0, np.float32)
    v0 = np.full((120, 160), -1.0, np.float32)
    u1, v1 = vr.calcUV(a, b, u0.copy(), v0.copy())
    dump("vr_warm_u", u1)
    dump("vr_warm_v", v1)

    # DIS at each preset, and with the refinement stage switched off so the
    # inverse search and densification are checked on their own.
    for nm, preset in [("ultrafast", cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST),
                       ("fast", cv2.DISOPTICAL_FLOW_PRESET_FAST),
                       ("medium", cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)]:
        dis = cv2.DISOpticalFlow_create(preset)
        dump(f"dis_{nm}", dis.calc(a, b, None))
        dis2 = cv2.DISOpticalFlow_create(preset)
        dis2.setVariationalRefinementIterations(0)
        dump(f"dis_{nm}_norefine", dis2.calc(a, b, None))

    # No spatial propagation: the stripe decomposition stops mattering.
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(False)
    dis.setVariationalRefinementIterations(0)
    dump("dis_noprop", dis.calc(a, b, None))

    # No mean normalisation, which swaps both patch kernels.
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseMeanNormalization(False)
    dis.setVariationalRefinementIterations(0)
    dump("dis_nomeannorm", dis.calc(a, b, None))


def unverified(d, gray, bgr, dump):
    """Reference arrays for the surface that had no differential check.

    Everything here was written from upstream's source and tested against
    hand-computed values, which proves self-consistency and nothing about cv2.
    These are the checks that make the claim real.
    """
    import numpy as np
    import cv2

    h, w = gray.shape

    # ---- Hough ----
    # Three lines at angles that are not multiples of the theta step, so no
    # accumulator tie decides the answer. See the note on synthetic fixtures:
    # a symmetric figure makes the comparison about float ordering, not maths.
    lines_img = np.zeros((h, w), np.uint8)
    cv2.line(lines_img, (10, 5), (150, 100), 255, 1)
    cv2.line(lines_img, (5, 90), (155, 30), 255, 1)
    cv2.line(lines_img, (60, 0), (75, 119), 255, 1)
    dump("in_hough_lines", lines_img)

    hl = cv2.HoughLines(lines_img, 1, np.pi / 180, 60)
    dump("hg_lines", (hl.reshape(-1, 2) if hl is not None else np.zeros((0, 2), np.float32)))

    hlp = cv2.HoughLinesP(lines_img, 1, np.pi / 180, 40, minLineLength=30, maxLineGap=3)
    dump("hg_linesp_count", np.array([[0 if hlp is None else len(hlp)]], np.int32))

    # HoughCircles wants a blurred input; on a hard binary edge the gradient
    # direction is quantised by the rasterisation staircase and the peak walks.
    circ = np.zeros((h, w), np.uint8)
    cv2.circle(circ, (52, 61), 20, 255, -1)
    cv2.circle(circ, (115, 45), 14, 255, -1)
    circ = cv2.GaussianBlur(circ, (5, 5), 1.5)
    dump("in_hough_circles", circ)
    hc = cv2.HoughCircles(circ, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=20,
                          minRadius=8, maxRadius=30)
    dump("hg_circles", (hc.reshape(-1, 3) if hc is not None else np.zeros((0, 3), np.float32)))

    # ---- demosaicing ----
    # A real mosaic, not a flat field: VNG and EA are gradient-directed, so a
    # constant patch would pass with the interpolation entirely wrong.
    mosaic = np.zeros((h, w), np.uint8)
    mosaic[0::2, 0::2] = bgr[0::2, 0::2, 0]   # B
    mosaic[0::2, 1::2] = bgr[0::2, 1::2, 1]   # G
    mosaic[1::2, 0::2] = bgr[1::2, 0::2, 1]   # G
    mosaic[1::2, 1::2] = bgr[1::2, 1::2, 2]   # R
    dump("in_mosaic", mosaic)
    dump("dm_bilinear", cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2BGR))
    dump("dm_vng", cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2BGR_VNG))
    dump("dm_ea", cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2BGR_EA))

    # ---- EMD ----
    # Two 1-D signatures with an obvious optimal plan, plus an unbalanced pair
    # where the surplus has to stay put.
    s1 = np.array([[0.4, 0.0], [0.3, 3.0], [0.3, 7.0]], np.float32)
    s2 = np.array([[0.5, 1.0], [0.5, 6.0]], np.float32)
    dump("in_emd_s1", s1.astype(np.float64))
    dump("in_emd_s2", s2.astype(np.float64))
    for code, nm in [(cv2.DIST_L1, "l1"), (cv2.DIST_L2, "l2"), (cv2.DIST_C, "c")]:
        val, _, flow = cv2.EMD(s1, s2, code)
        dump(f"emd_{nm}", np.array([[float(val)]], np.float64))
        dump(f"emd_{nm}_flow", flow.astype(np.float64))

    # ---- stereo ----
    # Disparity is defined by left[x] == right[x - d], so the right image
    # samples *further along* the texture. Getting this backwards makes correct
    # code look broken, which it did once already.
    tex = np.random.RandomState(11).randint(0, 256, (60, 120)).astype(np.uint8)
    tex = cv2.GaussianBlur(tex, (3, 3), 0)
    disp_true = 8
    left = tex[:, :96]
    right = tex[:, disp_true:disp_true + 96]
    dump("in_stereo_l", np.ascontiguousarray(left))
    dump("in_stereo_r", np.ascontiguousarray(right))
    bm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    dump("sm_bm", bm.compute(left, right).astype(np.int32))
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=5)
    dump("sm_sgbm", sgbm.compute(left, right).astype(np.int32))

    # ---- calibration ----
    # A chessboard rendered by projecting a known board with a known pose,
    # so the corner positions have a ground truth independent of both ports.
    K = np.array([[520.0, 0.0, 79.5], [0.0, 520.0, 59.5], [0.0, 0.0, 1.0]])
    dist = np.zeros(5)
    pat = (6, 5)
    objp = np.zeros((pat[0] * pat[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pat[0], 0:pat[1]].T.reshape(-1, 2) * 0.025
    rvec = np.array([[0.08], [-0.12], [0.03]])
    tvec = np.array([[-0.06], [-0.05], [0.55]])
    dump("in_cal_obj", objp.astype(np.float64))
    dump("in_cal_K", K)
    dump("in_cal_rvec", rvec)
    dump("in_cal_tvec", tvec)
    proj, jac = cv2.projectPoints(objp.astype(np.float64), rvec, tvec, K, dist)
    dump("cal_project", proj.reshape(-1, 2).astype(np.float64))

    ok, rv, tv = cv2.solvePnP(objp.astype(np.float64), proj, K, dist,
                              flags=cv2.SOLVEPNP_ITERATIVE)
    dump("cal_pnp", np.hstack([rv.ravel(), tv.ravel()]).reshape(1, -1))

    # Rodrigues both ways: the matrix and the vector recovered from it.
    R, _ = cv2.Rodrigues(rvec)
    dump("cal_rodrigues_R", R)
    back, _ = cv2.Rodrigues(R)
    dump("cal_rodrigues_v", back.reshape(1, -1))

    # ---- ml ----
    # Two well-separated Gaussian clouds: every classifier here should agree
    # with cv2 exactly, because the decision is nowhere near a boundary.
    rs = np.random.RandomState(3)
    a = rs.randn(40, 2) * 0.4 + np.array([-2.0, -2.0])
    b = rs.randn(40, 2) * 0.4 + np.array([2.0, 2.0])
    X = np.vstack([a, b]).astype(np.float32)
    y = np.hstack([np.zeros(40), np.ones(40)]).astype(np.int32)
    q = np.array([[-1.5, -1.8], [1.7, 2.2], [-2.4, -1.1], [2.9, 1.4], [0.1, -0.2]], np.float32)
    dump("in_ml_x", X.astype(np.float64))
    dump("in_ml_y", y.reshape(-1, 1))
    dump("in_ml_q", q.astype(np.float64))

    knn = cv2.ml.KNearest_create()
    knn.train(X, cv2.ml.ROW_SAMPLE, y)
    _, res, nbr, dist_k = knn.findNearest(q, 3)
    dump("mlx_knn", res.astype(np.float64))
    dump("mlx_knn_dist", dist_k.astype(np.float64))

    nb = cv2.ml.NormalBayesClassifier_create()
    nb.train(X, cv2.ml.ROW_SAMPLE, y)
    _, nbres = nb.predict(q)
    dump("mlx_bayes", nbres.astype(np.float64).reshape(-1, 1))


if __name__ == "__main__":
    main()
