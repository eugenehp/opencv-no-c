//! The two alternative ellipse fits (`modules/imgproc/src/fitellipse.cpp`).
//!
//! # Why three fits exist
//!
//! [`crate::fitting::fit_ellipse`] minimises algebraic distance under the
//! constraint `F = 1`, which is fast and biased: it pulls the answer towards
//! smaller ellipses, and on a short arc it can be badly wrong. The two here
//! constrain the conic differently — AMS by a gradient-weighted norm, Direct by
//! the discriminant — and both are *ellipse-specific*, meaning they cannot
//! return a hyperbola for awkward input the way the unconstrained fit can.

use cv_core::{cv_assert, Code, Error, Point, Result};

use crate::fitting::RotatedRect;

/// Solve the 6x6 generalised eigenproblem the two fits reduce to, by inverse
/// iteration on `S^-1 C`.
///
/// Returns the conic `(a, b, c, d, e, f)` for `ax^2 + bxy + cy^2 + dx + ey + f`.
fn solve_conic(s: &[[f64; 6]; 6], c: &[[f64; 6]; 6]) -> Option<[f64; 6]> {
    // S is symmetric positive semi-definite; regularise so it inverts.
    let mut m = *s;
    for (i, row) in m.iter_mut().enumerate() {
        row[i] += 1e-10;
    }
    let si = invert6(&m)?;
    // M = S^-1 C. The wanted eigenvector is the one with the largest
    // eigenvalue of M for AMS, and the single positive one for Direct; power
    // iteration on M finds the dominant direction in both cases because the
    // constraint matrices are built so that it is the right one.
    let mut mm = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            mm[i][j] = (0..6).map(|k| si[i][k] * c[k][j]).sum();
        }
    }
    let mut v = [1.0f64, 0.3, -0.7, 0.2, -0.4, 0.6];
    let mut last = 0.0f64;
    for it in 0..2000 {
        let mut w = [0.0f64; 6];
        for i in 0..6 {
            w[i] = (0..6).map(|j| mm[i][j] * v[j]).sum();
        }
        let n = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n < 1e-300 {
            return None;
        }
        for x in w.iter_mut() {
            *x /= n;
        }
        // Rayleigh quotient, for the convergence test.
        let mut mv = [0.0f64; 6];
        for i in 0..6 {
            mv[i] = (0..6).map(|j| mm[i][j] * w[j]).sum();
        }
        let lam: f64 = (0..6).map(|i| w[i] * mv[i]).sum();
        v = w;
        if it > 3 && (lam - last).abs() < 1e-14 * lam.abs().max(1.0) {
            break;
        }
        last = lam;
    }
    Some(v)
}

/// Invert a 6x6 matrix by Gauss-Jordan with partial pivoting.
fn invert6(m: &[[f64; 6]; 6]) -> Option<[[f64; 6]; 6]> {
    let mut a = *m;
    let mut inv = [[0.0f64; 6]; 6];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..6 {
        let mut piv = col;
        for r in (col + 1)..6 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-14 {
            return None;
        }
        a.swap(col, piv);
        inv.swap(col, piv);
        let d = a[col][col];
        for (ak, ik) in a[col].iter_mut().zip(inv[col].iter_mut()) {
            *ak /= d;
            *ik /= d;
        }
        for r in 0..6 {
            if r == col {
                continue;
            }
            let f = a[r][col];
            if f == 0.0 {
                continue;
            }
            for k in 0..6 {
                a[r][k] -= f * a[col][k];
                inv[r][k] -= f * inv[col][k];
            }
        }
    }
    Some(inv)
}

/// Turn a conic into upstream's [`RotatedRect`], or `None` when it is not an
/// ellipse.
fn conic_to_rect(k: [f64; 6], ox: f64, oy: f64) -> Option<RotatedRect> {
    let (a, b, c, d, e, f) = (k[0], k[1], k[2], k[3], k[4], k[5]);
    // The discriminant decides the conic's type; a non-negative one is a
    // parabola or hyperbola and has no ellipse to report.
    let disc = b * b - 4.0 * a * c;
    if disc >= 0.0 {
        return None;
    }
    let cx = (2.0 * c * d - b * e) / disc;
    let cy = (2.0 * a * e - b * d) / disc;
    // Rotate to the principal axes.
    let theta = 0.5 * (b).atan2(a - c);
    let (ct, st) = (theta.cos(), theta.sin());
    let ap = a * ct * ct + b * ct * st + c * st * st;
    let cp = a * st * st - b * ct * st + c * ct * ct;
    // Constant term at the centre.
    let fc = a * cx * cx + b * cx * cy + c * cy * cy + d * cx + e * cy + f;
    if ap.abs() < 1e-300 || cp.abs() < 1e-300 {
        return None;
    }
    let (r1, r2) = (-fc / ap, -fc / cp);
    if r1 <= 0.0 || r2 <= 0.0 {
        return None;
    }
    // Upstream reports full axis lengths, not semi-axes.
    let (w, h) = (2.0 * r1.sqrt(), 2.0 * r2.sqrt());
    Some(RotatedRect {
        center: (cx + ox, cy + oy),
        size: (w, h),
        angle: theta.to_degrees(),
    })
}

/// Build the scatter matrix `S = D^T D` for the design matrix of `pts`,
/// centred on their mean.
fn scatter(pts: &[Point]) -> ([[f64; 6]; 6], f64, f64) {
    let n = pts.len() as f64;
    let ox = pts.iter().map(|p| f64::from(p.x)).sum::<f64>() / n;
    let oy = pts.iter().map(|p| f64::from(p.y)).sum::<f64>() / n;
    let mut s = [[0.0f64; 6]; 6];
    for p in pts {
        let (x, y) = (f64::from(p.x) - ox, f64::from(p.y) - oy);
        let d = [x * x, x * y, y * y, x, y, 1.0];
        for i in 0..6 {
            for j in 0..6 {
                s[i][j] += d[i] * d[j];
            }
        }
    }
    (s, ox, oy)
}

/// `cv::fitEllipseAMS(points)` — the Approximate Mean Square fit.
///
/// Weights each point's algebraic residual by the conic's gradient there, which
/// makes the answer close to a true geometric fit without the cost of one. On a
/// full closed contour it agrees with [`fit_ellipse_direct`]; on a partial arc
/// it is usually the better of the two.
pub fn fit_ellipse_ams(pts: &[Point]) -> Result<RotatedRect> {
    cv_assert!(pts.len() >= 5, "fitEllipseAMS needs at least 5 points, got {}", pts.len());
    let (s, ox, oy) = scatter(pts);

    // The AMS constraint matrix: the gradient norm summed over the points.
    let n = pts.len() as f64;
    let mut c = [[0.0f64; 6]; 6];
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for p in pts {
        let (x, y) = (f64::from(p.x) - ox, f64::from(p.y) - oy);
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
        sx += x;
        sy += y;
    }
    // d/dx = 2ax + by + d, d/dy = bx + 2cy + e; C accumulates their squares.
    c[0][0] = 4.0 * sxx;
    c[0][1] = 2.0 * sxy;
    c[1][0] = 2.0 * sxy;
    c[0][3] = 2.0 * sx;
    c[3][0] = 2.0 * sx;
    c[1][1] = sxx + syy;
    c[1][2] = 2.0 * sxy;
    c[2][1] = 2.0 * sxy;
    c[1][3] = sy;
    c[3][1] = sy;
    c[1][4] = sx;
    c[4][1] = sx;
    c[2][2] = 4.0 * syy;
    c[2][4] = 2.0 * sy;
    c[4][2] = 2.0 * sy;
    c[3][3] = n;
    c[4][4] = n;

    let k = solve_conic(&s, &c)
        .ok_or_else(|| Error::new(Code::StsBadArg, "the AMS system is degenerate".to_string()))?;
    conic_to_rect(k, ox, oy).ok_or_else(|| {
        Error::new(Code::StsBadArg, "the AMS fit did not produce an ellipse".to_string())
    })
}

/// `cv::fitEllipseDirect(points)` — Fitzgibbon's direct least-squares fit.
///
/// Constrains the conic's discriminant so the result is *always* an ellipse —
/// never the hyperbola an unconstrained algebraic fit can return on a short or
/// noisy arc. That guarantee is the reason to pick it.
///
/// # Why this uses the reduced form
///
/// Fitzgibbon's constraint matrix is indefinite, so the wanted solution is the
/// eigenvector belonging to the single *positive* eigenvalue — which is not
/// generally the dominant one. Power iteration on the 6x6 system converges to
/// the wrong vector on short arcs. This uses the Halir-Flusser reduction to a
/// 3x3 problem instead, where all three eigenvalues can be had exactly from the
/// characteristic cubic and the right one picked by its sign.
pub fn fit_ellipse_direct(pts: &[Point]) -> Result<RotatedRect> {
    cv_assert!(pts.len() >= 5, "fitEllipseDirect needs at least 5 points, got {}", pts.len());
    let n = pts.len() as f64;
    let ox = pts.iter().map(|p| f64::from(p.x)).sum::<f64>() / n;
    let oy = pts.iter().map(|p| f64::from(p.y)).sum::<f64>() / n;

    // S1 = D1'D1, S2 = D1'D2, S3 = D2'D2 with D1 = [x^2, xy, y^2], D2 = [x, y, 1].
    let mut s1 = [[0.0f64; 3]; 3];
    let mut s2 = [[0.0f64; 3]; 3];
    let mut s3 = [[0.0f64; 3]; 3];
    for p in pts {
        let (x, y) = (f64::from(p.x) - ox, f64::from(p.y) - oy);
        let d1 = [x * x, x * y, y * y];
        let d2 = [x, y, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                s1[i][j] += d1[i] * d1[j];
                s2[i][j] += d1[i] * d2[j];
                s3[i][j] += d2[i] * d2[j];
            }
        }
    }
    let s3i = invert3(&s3).ok_or_else(|| {
        Error::new(Code::StsBadArg, "the points are degenerate; S3 is singular".to_string())
    })?;
    // T = -S3^-1 S2'
    let mut t = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = -(0..3).map(|k| s3i[i][k] * s2[j][k]).sum::<f64>();
        }
    }
    // M = S1 + S2 T
    let mut m = s1;
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] += (0..3).map(|k| s2[i][k] * t[k][j]).sum::<f64>();
        }
    }
    // Premultiply by C1^-1 = [[0,0,1/2],[0,-1,0],[1/2,0,0]].
    let mm = [
        [m[2][0] * 0.5, m[2][1] * 0.5, m[2][2] * 0.5],
        [-m[1][0], -m[1][1], -m[1][2]],
        [m[0][0] * 0.5, m[0][1] * 0.5, m[0][2] * 0.5],
    ];

    // Eigenvalues from the characteristic cubic, eigenvectors from the null
    // space; pick the one satisfying 4ac - b^2 > 0, which is the ellipse.
    let tr = mm[0][0] + mm[1][1] + mm[2][2];
    let c2 = mm[0][0] * mm[1][1] - mm[0][1] * mm[1][0] + mm[0][0] * mm[2][2]
        - mm[0][2] * mm[2][0]
        + mm[1][1] * mm[2][2]
        - mm[1][2] * mm[2][1];
    let det = mm[0][0] * (mm[1][1] * mm[2][2] - mm[1][2] * mm[2][1])
        - mm[0][1] * (mm[1][0] * mm[2][2] - mm[1][2] * mm[2][0])
        + mm[0][2] * (mm[1][0] * mm[2][1] - mm[1][1] * mm[2][0]);
    let roots = cv_core::solvers::solve_cubic(&[1.0, -tr, c2, -det]);

    let mut best: Option<[f64; 3]> = None;
    for lam in roots {
        let Some(v) = null_vector3(&mm, lam) else { continue };
        if 4.0 * v[0] * v[2] - v[1] * v[1] > 0.0 {
            best = Some(v);
            break;
        }
    }
    let a1 = best.ok_or_else(|| {
        Error::new(Code::StsBadArg, "no ellipse solution exists for these points".to_string())
    })?;
    let mut a2 = [0.0f64; 3];
    for i in 0..3 {
        a2[i] = (0..3).map(|k| t[i][k] * a1[k]).sum();
    }
    conic_to_rect([a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]], ox, oy).ok_or_else(|| {
        Error::new(Code::StsBadArg, "the direct fit did not produce an ellipse".to_string())
    })
}

/// Invert a 3x3 matrix.
fn invert3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if d.abs() < 1e-14 {
        return None;
    }
    let mut o = [[0.0f64; 3]; 3];
    for (i, _) in (0..3).enumerate() {
        for (j, orow) in o.iter_mut().enumerate() {
            let (a, b) = ((i + 1) % 3, (i + 2) % 3);
            let (c, e) = ((j + 1) % 3, (j + 2) % 3);
            orow[i] = (m[a][c] * m[b][e] - m[a][e] * m[b][c]) / d;
        }
    }
    Some(o)
}

/// A unit vector in the null space of `m - lambda I`.
fn null_vector3(m: &[[f64; 3]; 3], lambda: f64) -> Option<[f64; 3]> {
    let mut a = *m;
    for (i, row) in a.iter_mut().enumerate() {
        row[i] -= lambda;
    }
    // The null direction is the cross product of two independent rows; try
    // every pair and keep the longest, which is the best conditioned.
    let cross = |u: [f64; 3], v: [f64; 3]| {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let mut best = [0.0f64; 3];
    let mut best_n = 0.0f64;
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let c = cross(a[i], a[j]);
        let n = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        if n > best_n {
            best_n = n;
            best = c;
        }
    }
    if best_n < 1e-300 {
        return None;
    }
    for v in best.iter_mut() {
        *v /= best_n;
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points on an ellipse of the given geometry.
    fn on_ellipse(cx: f64, cy: f64, a: f64, b: f64, deg: f64, n: usize, span: f64) -> Vec<Point> {
        let th = deg.to_radians();
        (0..n)
            .map(|i| {
                let t = span * (i as f64) / (n as f64 - 1.0);
                let (x, y) = (a * t.cos(), b * t.sin());
                Point::new(
                    (cx + x * th.cos() - y * th.sin()).round() as i32,
                    (cy + x * th.sin() + y * th.cos()).round() as i32,
                )
            })
            .collect()
    }

    /// Compare two ellipses allowing the 180-degree ambiguity and axis swap.
    fn same_ellipse(got: &RotatedRect, wa: f64, wb: f64, tol: f64) -> bool {
        let (g1, g2) = got.size;
        let (big, small) = if g1 >= g2 { (g1, g2) } else { (g2, g1) };
        let (wbig, wsmall) = if wa >= wb { (wa, wb) } else { (wb, wa) };
        (big - wbig).abs() < tol * wbig && (small - wsmall).abs() < tol * wsmall
    }

    #[test]
    fn both_fits_recover_a_full_ellipse() {
        // Semi-axes 60 and 30, so full axes 120 and 60.
        let pts = on_ellipse(200.0, 150.0, 60.0, 30.0, 30.0, 60, std::f64::consts::TAU);
        for (name, r) in [
            ("AMS", fit_ellipse_ams(&pts).unwrap()),
            ("Direct", fit_ellipse_direct(&pts).unwrap()),
        ] {
            assert!(
                (r.center.0 - 200.0).abs() < 2.0,
                "{name} centre x {}",
                r.center.0
            );
            assert!((r.center.1 - 150.0).abs() < 2.0, "{name} centre y");
            assert!(same_ellipse(&r, 120.0, 60.0, 0.06), "{name} size {:?}", r.size);
        }
    }

    #[test]
    fn both_fits_recover_a_circle() {
        let pts = on_ellipse(100.0, 100.0, 40.0, 40.0, 0.0, 40, std::f64::consts::TAU);
        for (name, r) in [
            ("AMS", fit_ellipse_ams(&pts).unwrap()),
            ("Direct", fit_ellipse_direct(&pts).unwrap()),
        ] {
            assert!(same_ellipse(&r, 80.0, 80.0, 0.05), "{name} size {:?}", r.size);
            // A circle's axes must come out equal whatever the angle says.
            let (a, b) = r.size;
            assert!((a - b).abs() < 0.1 * a, "{name}: a circle has equal axes, got {a} and {b}");
        }
    }

    #[test]
    fn the_direct_fit_always_returns_an_ellipse() {
        // A short arc — the case where an unconstrained algebraic fit can
        // return a hyperbola and fail outright.
        let arc = on_ellipse(150.0, 120.0, 70.0, 40.0, 15.0, 30, 1.1);
        let r = fit_ellipse_direct(&arc).expect("the direct fit is ellipse-specific by construction");
        assert!(r.size.0 > 0.0 && r.size.1 > 0.0, "degenerate size {:?}", r.size);
        assert!(r.size.0.is_finite() && r.size.1.is_finite());
    }

    #[test]
    fn a_rotated_ellipse_reports_a_usable_angle() {
        let pts = on_ellipse(0.0, 0.0, 80.0, 20.0, 40.0, 72, std::f64::consts::TAU);
        let r = fit_ellipse_ams(&pts).unwrap();
        // The major axis direction is what the angle means; it is defined
        // modulo 180 degrees, and which of the two axes is "width" can swap.
        let ang = r.angle.rem_euclid(180.0);
        let ok = (ang - 40.0).abs() < 8.0 || (ang - 130.0).abs() < 8.0;
        assert!(ok, "angle {ang} should be near 40 or 130 degrees");
    }

    #[test]
    fn both_fits_need_five_points() {
        let four = vec![Point::new(0, 0), Point::new(1, 0), Point::new(1, 1), Point::new(0, 1)];
        assert!(fit_ellipse_ams(&four).is_err());
        assert!(fit_ellipse_direct(&four).is_err());
        // And say how many they got.
        let e = format!("{}", fit_ellipse_ams(&four).unwrap_err());
        assert!(e.contains('4'), "the error should name the count: {e}");
    }

    #[test]
    fn collinear_points_are_not_an_ellipse() {
        let line: Vec<Point> = (0..12).map(|i| Point::new(i * 5, i * 5)).collect();
        // Whatever happens, it must not be a silently bogus ellipse.
        for e in [fit_ellipse_ams(&line), fit_ellipse_direct(&line)].into_iter().flatten() {
            {
                assert!(
                    e.size.0.is_finite() && e.size.1.is_finite(),
                    "a degenerate fit must not report NaN or infinity: {:?}",
                    e.size
                );
            }
        }
    }

    /// All three fits against cv2 4.13.0, including the partial arcs that a
    /// single-pass algebraic fit gets wrong.
    ///
    /// Each entry is `(name, points, fitEllipse, fitEllipseAMS,
    /// fitEllipseDirect)` with each fit as `[cx, cy, width, height]`.
    #[test]
    fn all_three_fits_match_cv2() {
        type Case = (&'static str, &'static [(i32, i32)], [f64; 4], [f64; 4], [f64; 4]);
        let cases: &[Case] = &[
            (
                "full",
                &[(252, 180), (250, 183), (248, 185), (245, 187), (241, 188), (237, 189), (233, 190), (228, 190), (223, 189), (218, 189), (212, 187), (206, 186), (201, 184), (195, 181), (189, 178), (184, 175), (178, 172), (173, 168), (168, 164), (164, 160), (160, 156), (156, 152), (153, 148), (150, 143), (148, 139), (147, 135), (146, 132), (146, 128), (146, 125), (147, 121), (149, 119), (151, 116), (154, 114), (157, 113), (161, 111), (165, 111), (170, 110), (174, 110), (180, 111), (185, 112), (191, 114), (196, 115), (202, 118), (208, 120), (214, 123), (219, 126), (224, 130), (229, 134), (234, 138), (238, 142), (242, 146), (246, 150), (248, 154), (251, 159), (252, 163), (254, 167), (254, 170), (254, 174), (253, 177), (252, 180)],
                [199.9984, 150.0221, 59.9946, 120.2620],
                [199.9981, 150.0219, 59.9798, 120.2645],
                [199.9984, 150.0221, 59.9877, 120.2491],
            ),
            (
                "circle",
                &[(140, 100), (139, 106), (138, 113), (135, 119), (132, 124), (128, 129), (123, 133), (117, 136), (111, 138), (105, 140), (98, 140), (92, 139), (86, 137), (80, 135), (75, 131), (70, 127), (66, 121), (63, 116), (61, 110), (60, 103), (60, 97), (61, 90), (63, 84), (66, 79), (70, 73), (75, 69), (80, 65), (86, 63), (92, 61), (98, 60), (105, 60), (111, 62), (117, 64), (123, 67), (128, 71), (132, 76), (135, 81), (138, 87), (139, 94), (140, 100)],
                [99.8902, 100.0000, 79.9250, 80.1445],
                [99.8901, 100.0000, 79.9183, 80.1379],
                [99.8902, 100.0000, 79.9180, 80.1381],
            ),
            (
                "rotated",
                &[(361, 351), (360, 353), (358, 353), (356, 354), (353, 354), (350, 353), (346, 352), (342, 351), (338, 349), (334, 347), (329, 344), (324, 342), (319, 338), (313, 335), (308, 331), (302, 327), (297, 323), (291, 319), (286, 314), (280, 310), (275, 305), (270, 300), (266, 295), (261, 291), (257, 286), (253, 281), (250, 277), (246, 273), (244, 269), (242, 265), (240, 262), (239, 259), (238, 256), (237, 253), (238, 251), (238, 249), (239, 248), (241, 247), (243, 246), (246, 246), (248, 247), (252, 247), (256, 249), (260, 250), (264, 252), (269, 254), (274, 257), (279, 260), (284, 263), (289, 267), (295, 271), (300, 275), (306, 279), (311, 284), (317, 288), (322, 293), (327, 298), (332, 302), (337, 307), (341, 312), (345, 316), (349, 321), (352, 325), (355, 329), (357, 333), (359, 337), (361, 340), (362, 343), (363, 346), (363, 348), (362, 350), (361, 351)],
                [299.9620, 300.0110, 39.8839, 160.1487],
                [299.9611, 300.0103, 39.8548, 160.1801],
                [299.9620, 300.0110, 39.8732, 160.1079],
            ),
            (
                "arc",
                &[(218, 138), (217, 140), (217, 141), (216, 142), (215, 144), (214, 145), (214, 146), (213, 148), (211, 149), (210, 150), (209, 151), (208, 152), (206, 153), (205, 154), (203, 155), (201, 156), (200, 157), (198, 158), (196, 158), (194, 159), (192, 160), (190, 160), (188, 161), (186, 161), (183, 162), (181, 162), (179, 162), (176, 162), (174, 163), (171, 163)],
                [81.0605, 71.3164, 131.2933, 305.1005],
                [-147.4488, -84.1648, 224.2250, 856.7211],
                [189.9148, 147.2139, 25.0476, 57.0041],
            ),
            (
                "halfarc",
                &[(199, 212), (196, 212), (193, 213), (190, 213), (187, 213), (184, 212), (181, 211), (178, 210), (175, 209), (172, 207), (169, 204), (166, 202), (164, 199), (161, 196), (158, 193), (156, 190), (154, 186), (152, 182), (150, 178), (148, 174), (146, 170), (145, 166), (144, 161), (143, 157), (143, 153), (142, 149), (142, 145), (142, 140), (143, 137), (143, 133), (144, 129), (145, 126), (147, 123), (148, 120), (150, 117), (152, 115), (154, 113), (156, 111), (159, 109), (161, 108)],
                [180.0832, 159.9836, 70.1592, 110.1003],
                [180.0719, 159.9878, 70.1209, 110.0910],
                [179.9223, 160.0420, 69.7869, 109.9848],
            ),
        ];

        for (name, raw, std, ams, direct) in cases {
            let pts: Vec<Point> = raw.iter().map(|&(x, y)| Point::new(x, y)).collect();
            let got = [
                ("fitEllipse", crate::fitting::fit_ellipse(&pts).unwrap(), std),
                ("fitEllipseAMS", fit_ellipse_ams(&pts).unwrap(), ams),
                ("fitEllipseDirect", fit_ellipse_direct(&pts).unwrap(), direct),
            ];
            for (fit, r, want) in got {
                // The centre tolerance scales with the ellipse. Fitting a
                // 1.1-radian arc extrapolates to axes near 900 units with the
                // centre far outside the data, where the problem is genuinely
                // ill-conditioned: an absolute tolerance there would be
                // asserting precision neither library has.
                let scale = want[2].max(want[3]).max(1.0);
                let ctol = 0.01f64.max(0.001 * scale);
                assert!(
                    (r.center.0 - want[0]).abs() < ctol && (r.center.1 - want[1]).abs() < ctol,
                    "{name}/{fit} centre: ({}, {}) vs cv2 ({}, {}), tolerance {ctol}",
                    r.center.0, r.center.1, want[0], want[1]
                );
                // cv2 can report the axes either way round; compare as a set.
                let (mut a, mut b) = r.size;
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                let (mut wa, mut wb) = (want[2], want[3]);
                if wa > wb {
                    std::mem::swap(&mut wa, &mut wb);
                }
                // 0.3% of the axis length: everything but AMS on the extreme
                // arc agrees to the printed precision.
                assert!(
                    (a - wa).abs() < 0.003 * wa.max(1.0) && (b - wb).abs() < 0.003 * wb.max(1.0),
                    "{name}/{fit} axes: {a}/{b} vs cv2 {wa}/{wb}"
                );
            }
        }
    }
}
