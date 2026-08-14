//! Hershey vector text (`modules/imgproc/src/drawing.cpp`, `hershey_fonts.cpp`).
//!
//! # Glyph encoding
//!
//! Upstream stores each glyph as an ASCII string. The first two bytes are the
//! left and right bearing; every pair after that is a vertex, and a space
//! starts a new stroke. All coordinates are the byte value minus `'R'`, so the
//! origin sits at `'R'` and the units are the font's own em grid — the same
//! encoding Dr Hershey's original 1967 tables used.
//!
//! `FONT_HERSHEY_SIMPLEX` is the only face transcribed here. The strings below
//! are lifted verbatim from `g_HersheyGlyphs` via the `HersheySimplex` index
//! table, so the letterforms are upstream's, not an approximation.

use cv_core::{cv_assert, MatLikeMut, Point, Result, Scalar};

use crate::drawing::line;

/// `cv::HersheyFonts` — the faces this port carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FontFace {
    /// `cv::FONT_HERSHEY_SIMPLEX`
    #[default]
    Simplex,
}

/// Glyph strokes for `FONT_HERSHEY_SIMPLEX`, indexed by `ch as usize - 32`.
///
/// Covers the 95 printable ASCII characters, `' '` through `'~'`.
#[rustfmt::skip]
const SIMPLEX: [&str; 95] = [
    "JZ",
    "MWRFRT RYQZR[SZRY",
    "JZNFNM VFVM",
    "G]OFOb UFUb JQZQ JWZW",
    "H\\PBP_ TBT_ YIWGTFPFMGKIKKLMMNOOUQWRXSYUYXWZT[P[MZKX",
    "F^[FYGVHSHPGNFLFJGIIIKKMMMOLPJPHNF [FI[ YTWTUUTWTYV[X[ZZ[X[VYT",
    "E_\\O\\N[MZMYNXPVUTXRZP[L[JZIYHWHUISJRQNRMSKSIRGPFNGMIMKNNPQUXWZY[[[\\Z\\Y",
    "NVRFRM",
    "KYVBTDRGPKOPOTPYR]T`Vb",
    "KYNBPDRGTKUPUTTYR]P`Nb",
    "JZRLRX MOWU WOMU",
    "E_RIR[ IR[R",
    "MWSZR[QZRYSZS\\R^Q_",
    "E_IR[R",
    "MWRYQZR[SZRY",
    "G][BIb",
    "H\\QFNGLJKOKRLWNZQ[S[VZXWYRYOXJVGSFQF",
    "H\\NJPISFS[",
    "H\\LKLJMHNGPFTFVGWHXJXLWNUQK[Y[",
    "H\\MFXFRNUNWOXPYSYUXXVZS[P[MZLYKW",
    "H\\UFKTZT UFU[",
    "H\\WFMFLOMNPMSMVNXPYSYUXXVZS[P[MZLYKW",
    "H\\XIWGTFRFOGMJLOLTMXOZR[S[VZXXYUYTXQVOSNRNOOMQLT",
    "H\\YFO[ KFYF",
    "H\\PFMGLILKMMONSOVPXRYTYWXYWZT[P[MZLYKWKTLRNPQOUNWMXKXIWGTFPF",
    "H\\XMWPURRSQSNRLPKMKLLINGQFRFUGWIXMXRWWUZR[P[MZLX",
    "MWRMQNROSNRM RYQZR[SZRY",
    "MWRMQNROSNRM SZR[QZRYSZS\\R^Q_",
    "F^ZIJRZ[",
    "E_IO[O IU[U",
    "F^JIZRJ[",
    "I[LKLJMHNGPFTFVGWHXJXLWNVORQRT RYQZR[SZRY",
    "DaWNVLTKQKOLNMMOMRNTOUQVTVVUWS WKWSXUYV[V\\U]S]O\\L[JYHWGTFQFNGLHJJILHOHRIUJWLYNZQ[T[WZYY",
    "I[RFJ[ RFZ[ MTWT",
    "G\\KFK[ KFTFWGXHYJYLXNWOTP KPTPWQXRYTYWXYWZT[K[",
    "H]ZKYIWGUFQFOGMILKKNKSLVMXOZQ[U[WZYXZV",
    "G\\KFK[ KFRFUGWIXKYNYSXVWXUZR[K[",
    "H[LFL[ LFYF LPTP L[Y[",
    "HZLFL[ LFYF LPTP",
    "H]ZKYIWGUFQFOGMILKKNKSLVMXOZQ[U[WZYXZVZS USZS",
    "G]KFK[ YFY[ KPYP",
    "NVRFR[",
    "JZVFVVUYTZR[P[NZMYLVLT",
    "G\\KFK[ YFKT POY[",
    "HYLFL[ L[X[",
    "F^JFJ[ JFR[ ZFR[ ZFZ[",
    "G]KFK[ KFY[ YFY[",
    "G]PFNGLIKKJNJSKVLXNZP[T[VZXXYVZSZNYKXIVGTFPF",
    "G\\KFK[ KFTFWGXHYJYMXOWPTQKQ",
    "G]PFNGLIKKJNJSKVLXNZP[T[VZXXYVZSZNYKXIVGTFPF SWY]",
    "G\\KFK[ KFTFWGXHYJYLXNWOTPKP RPY[",
    "H\\YIWGTFPFMGKIKKLMMNOOUQWRXSYUYXWZT[P[MZKX",
    "JZRFR[ KFYF",
    "G]KFKULXNZQ[S[VZXXYUYF",
    "I[JFR[ ZFR[",
    "F^HFM[ RFM[ RFW[ \\FW[",
    "H\\KFY[ YFK[",
    "I[JFRPR[ ZFRP",
    "H\\YFK[ KFYF K[Y[",
    "KYOBOb OBVB ObVb",
    "G]IL[b",
    "KYUBUb NBUB NbUb",
    "G]JTROZT JTRPZT",
    "I[J[Z[",
    "LXPFUL PFOGUL",
    "I\\XMX[ XPVNTMQMONMPLSLUMXOZQ[T[VZXX",
    "H[LFL[ LPNNPMSMUNWPXSXUWXUZS[P[NZLX",
    "I[XPVNTMQMONMPLSLUMXOZQ[T[VZXX",
    "I\\XFX[ XPVNTMQMONMPLSLUMXOZQ[T[VZXX",
    "I[LSXSXQWOVNTMQMONMPLSLUMXOZQ[T[VZXX",
    "MYWFUFSGRJR[ OMVM",
    "I\\XMX]W`VaTbQbOa XPVNTMQMONMPLSLUMXOZQ[T[VZXX",
    "I\\MFM[ MQPNRMUMWNXQX[",
    "NVQFRGSFREQF RMR[",
    "MWRFSGTFSERF SMS^RaPbNb",
    "IZMFM[ WMMW QSX[",
    "NVRFR[",
    "CaGMG[ GQJNLMOMQNRQR[ RQUNWMZM\\N]Q][",
    "I\\MMM[ MQPNRMUMWNXQX[",
    "I\\QMONMPLSLUMXOZQ[T[VZXXYUYSXPVNTMQM",
    "H[LMLb LPNNPMSMUNWPXSXUWXUZS[P[NZLX",
    "I\\XMXb XPVNTMQMONMPLSLUMXOZQ[T[VZXX",
    "KXOMO[ OSPPRNTMWM",
    "J[XPWNTMQMNNMPNRPSUTWUXWXXWZT[Q[NZMX",
    "MYRFRWSZU[W[ OMVM",
    "I\\MMMWNZP[S[UZXW XMX[",
    "JZLMR[ XMR[",
    "G]JMN[ RMN[ RMV[ ZMV[",
    "J[MMX[ XMM[",
    "JZLMR[ XMR[P_NaLbKb",
    "J[XMM[ MMXM M[X[",
    "KYTBQEPHPJQMSOSPORSTSUQWPZP\\Q_Tb",
    "NVRBRb",
    "KYPBSETHTJSMQOQPURQTQUSWTZT\\S_Pb",
    "F^IUISJPLONOPPTSVTXTZS[Q ISJQLPNPPQTTVUXUZT[Q[O",
];

/// Baseline drop of the simplex face, in glyph units.
///
/// Upstream computes it as `-(ascii[0] & 15)`; the simplex header is
/// `(9 + 12*16) + FONT_HAVE_GREEK`, whose low nibble is 9.
const BASE_LINE: i32 = -9;

/// Cap height of the simplex face, in glyph units.
///
/// Upstream reads it as `(ascii[0] >> 4) & 15` from the same header nibble pair
/// that gives [`BASE_LINE`]: the simplex header is `(9 + 12*16)`, so the low
/// nibble is 9 and the next is 12.
const CAP_LINE: i32 = 12;

/// Decoded strokes of one glyph, in glyph units, plus its advance.
struct Glyph {
    /// Polylines; each is a run of connected vertices.
    strokes: Vec<Vec<(i32, i32)>>,
    /// Left bearing.
    left: i32,
    /// Right bearing — the pen advance.
    right: i32,
}

/// Parse one glyph string.
fn decode(s: &str) -> Glyph {
    let b = s.as_bytes();
    if b.len() < 2 {
        return Glyph { strokes: Vec::new(), left: 0, right: 0 };
    }
    let left = b[0] as i32 - b'R' as i32;
    let right = b[1] as i32 - b'R' as i32;
    let mut strokes = Vec::new();
    let mut cur: Vec<(i32, i32)> = Vec::new();
    let mut i = 2;
    while i < b.len() {
        if b[i] == b' ' {
            // A space ends the current stroke; upstream skips one byte.
            if cur.len() > 1 {
                strokes.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
            i += 1;
            continue;
        }
        if i + 1 >= b.len() {
            break;
        }
        cur.push((b[i] as i32 - b'R' as i32, b[i + 1] as i32 - b'R' as i32));
        i += 2;
    }
    if cur.len() > 1 {
        strokes.push(cur);
    }
    Glyph { strokes, left, right }
}

/// Look up a character, falling back to the space glyph for anything outside
/// printable ASCII rather than erroring — upstream substitutes too.
fn glyph_for(c: char) -> &'static str {
    let i = c as u32;
    if (32..127).contains(&i) {
        SIMPLEX[(i - 32) as usize]
    } else {
        SIMPLEX[0]
    }
}

/// `cv::getTextSize(text, fontFace, fontScale, thickness, baseLine)`.
///
/// Returns `((width, height), baseline)`, where `baseline` is the distance from
/// the text's bottom to the baseline — the same triple upstream fills in.
#[must_use]
pub fn get_text_size(text: &str, _face: FontFace, font_scale: f64, thickness: i32) -> ((i32, i32), i32) {
    let base_line = -BASE_LINE;
    // Upstream's three formulas, transcribed rather than approximated. The
    // `thickness` term in the width and the *integer* division in the height
    // are both load-bearing: without them a 16-character string at scale 1 and
    // thickness 2 comes out 250x26 instead of cv2's 252x22.
    let mut view_x = 0.0f64;
    for c in text.chars() {
        let g = decode(glyph_for(c));
        view_x += (g.right - g.left) as f64 * font_scale;
    }
    let width = cv_core::saturate::cv_round(view_x + f64::from(thickness));
    let height = cv_core::saturate::cv_round(
        f64::from(CAP_LINE + base_line) * font_scale + f64::from((thickness + 1) / 2),
    );
    let base = cv_core::saturate::cv_round(f64::from(base_line) * font_scale + f64::from(thickness) * 0.5);
    ((width, height), base)
}

/// `cv::putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)`.
///
/// `org` is the bottom-left corner of the first character, as upstream defines
/// it — not the top-left, which is the usual first guess and puts the text a
/// line too high.
#[allow(clippy::too_many_arguments)]
pub fn put_text<M: MatLikeMut + ?Sized>(
    img: &mut M,
    text: &str,
    org: Point,
    face: FontFace,
    font_scale: f64,
    color: Scalar,
    thickness: i32,
    bottom_left_origin: bool,
) -> Result<()> {
    let _ = face;
    cv_assert!(font_scale > 0.0, "fontScale must be positive, got {font_scale}");
    if text.is_empty() {
        return Ok(());
    }
    let hscale = font_scale;
    let vscale = if bottom_left_origin { -font_scale } else { font_scale };
    let thickness = thickness.max(1);

    let mut pen_x = org.x as f64;
    let pen_y = org.y as f64 + BASE_LINE as f64 * vscale;
    for c in text.chars() {
        let g = decode(glyph_for(c));
        pen_x -= g.left as f64 * hscale;
        for stroke in &g.strokes {
            let pts: Vec<Point> = stroke
                .iter()
                .map(|&(x, y)| {
                    Point::new(
                        (pen_x + x as f64 * hscale).round() as i32,
                        (pen_y + y as f64 * vscale).round() as i32,
                    )
                })
                .collect();
            for w in pts.windows(2) {
                line(img, w[0], w[1], color, thickness)?;
            }
        }
        pen_x += g.right as f64 * hscale;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::types::CV_8UC1;
    use cv_core::{Mat, MatLike};

    fn canvas(w: i32, h: i32) -> Mat {
        Mat::zeros(h, w, CV_8UC1).unwrap()
    }
    fn lit(m: &Mat) -> usize {
        m.to_vec::<u8>().unwrap().iter().filter(|&&v| v > 0).count()
    }

    #[test]
    fn every_printable_ascii_has_a_glyph() {
        assert_eq!(SIMPLEX.len(), 95);
        for c in ' '..='~' {
            let g = decode(glyph_for(c));
            assert!(g.right >= g.left, "{c:?} has a negative advance");
            if c != ' ' {
                assert!(!g.strokes.is_empty(), "{c:?} has no strokes");
            }
        }
    }

    #[test]
    fn space_is_blank_but_advances() {
        let g = decode(glyph_for(' '));
        assert!(g.strokes.is_empty(), "space must draw nothing");
        assert!(g.right > g.left, "space must still move the pen");
    }

    #[test]
    fn the_capital_a_decodes_to_upstreams_strokes() {
        // "I[RFJ[ RFZ[ MTWT": bearings I,[ then three strokes.
        let g = decode(SIMPLEX[(b'A' - 32) as usize]);
        assert_eq!(g.left, b'I' as i32 - b'R' as i32);
        assert_eq!(g.right, b'[' as i32 - b'R' as i32);
        assert_eq!(g.strokes.len(), 3, "A is two diagonals and a crossbar");
        assert_eq!(g.strokes[2].len(), 2, "the crossbar is one segment");
    }

    #[test]
    fn text_draws_something_and_scales() {
        let mut small = canvas(200, 60);
        put_text(&mut small, "Hello", Point::new(5, 40), FontFace::Simplex, 1.0, Scalar::all(255.0), 1, false)
            .unwrap();
        let mut big = canvas(200, 60);
        put_text(&mut big, "Hello", Point::new(5, 40), FontFace::Simplex, 2.0, Scalar::all(255.0), 1, false)
            .unwrap();
        assert!(lit(&small) > 50, "text must draw pixels, got {}", lit(&small));
        assert!(lit(&big) > lit(&small), "doubling the scale must draw more");
    }

    #[test]
    fn empty_text_is_a_no_op() {
        let mut m = canvas(20, 20);
        put_text(&mut m, "", Point::new(0, 10), FontFace::Simplex, 1.0, Scalar::all(255.0), 1, false)
            .unwrap();
        assert_eq!(lit(&m), 0);
    }

    #[test]
    fn org_is_the_baseline_not_the_top() {
        // Drawn at y = 40, the glyphs sit above it; nothing lands near the top.
        let mut m = canvas(200, 60);
        put_text(&mut m, "Hg", Point::new(5, 40), FontFace::Simplex, 1.0, Scalar::all(255.0), 1, false)
            .unwrap();
        let top_rows: usize = (0..10)
            .map(|y| (0..200).filter(|&x| m.at_f64(y, x, 0).unwrap() > 0.0).count())
            .sum();
        assert_eq!(top_rows, 0, "text must not reach the top of the image");
        let mid: usize = (25..40)
            .map(|y| (0..200).filter(|&x| m.at_f64(y, x, 0).unwrap() > 0.0).count())
            .sum();
        assert!(mid > 0, "text should sit just above the baseline");
    }

    #[test]
    fn bottom_left_origin_flips_vertically() {
        let mut a = canvas(120, 60);
        put_text(&mut a, "A", Point::new(10, 30), FontFace::Simplex, 1.0, Scalar::all(255.0), 1, false)
            .unwrap();
        let mut b = canvas(120, 60);
        put_text(&mut b, "A", Point::new(10, 30), FontFace::Simplex, 1.0, Scalar::all(255.0), 1, true)
            .unwrap();
        assert_ne!(a.to_vec::<u8>().unwrap(), b.to_vec::<u8>().unwrap());
        // The flipped one extends below the baseline instead of above.
        let below = |m: &Mat| (31..60).map(|y| (0..120).filter(|&x| m.at_f64(y, x, 0).unwrap() > 0.0).count()).sum::<usize>();
        assert!(below(&b) > below(&a));
    }

    #[test]
    fn text_size_matches_cv2() {
        // cv2 4.13.0: getTextSize("Hello, World! 42", FONT_HERSHEY_SIMPLEX,
        // 1.0, 2) -> ((252, 22), 10).
        let ((w, h), b) = get_text_size("Hello, World! 42", FontFace::Simplex, 1.0, 2);
        assert_eq!((w, h, b), (252, 22, 10));
    }

    #[test]
    fn text_size_grows_with_the_string_and_the_scale() {
        let ((w1, h1), b1) = get_text_size("A", FontFace::Simplex, 1.0, 1);
        let ((w2, _), _) = get_text_size("AAAA", FontFace::Simplex, 1.0, 1);
        assert!(w2 > w1 * 3, "four A's should be about four times as wide");
        let ((w3, h3), b3) = get_text_size("A", FontFace::Simplex, 2.0, 1);
        assert!(w3 > w1 && h3 > h1 && b3 >= b1);
        assert!(b1 > 0, "the baseline drop must be positive");
    }

    #[test]
    fn text_size_of_an_empty_string_is_the_thickness() {
        // Not zero: upstream's width is `cvRound(view_x + thickness)`, so an
        // empty string still reports the pen width. cv2 4.13.0 returns
        // ((1, 22), 10), ((2, 22), 10) and ((3, 23), 10) for thickness 1, 2, 3.
        assert_eq!(get_text_size("", FontFace::Simplex, 1.0, 1), ((1, 22), 10));
        assert_eq!(get_text_size("", FontFace::Simplex, 1.0, 2), ((2, 22), 10));
        assert_eq!(get_text_size("", FontFace::Simplex, 1.0, 3), ((3, 23), 10));
    }

    #[test]
    fn drawn_width_is_close_to_the_reported_width() {
        let text = "Wide";
        let ((w, _), _) = get_text_size(text, FontFace::Simplex, 1.0, 1);
        let mut m = canvas(400, 80);
        put_text(&mut m, text, Point::new(10, 50), FontFace::Simplex, 1.0, Scalar::all(255.0), 1, false)
            .unwrap();
        let cols: Vec<i32> = (0..400)
            .filter(|&x| (0..80).any(|y| m.at_f64(y, x, 0).unwrap() > 0.0))
            .collect();
        let drawn = cols.last().unwrap() - cols.first().unwrap();
        assert!(
            (drawn - w).abs() < w / 4,
            "getTextSize says {w}, the drawing spans {drawn}"
        );
    }

    #[test]
    fn a_non_ascii_character_falls_back_instead_of_panicking() {
        let mut m = canvas(60, 40);
        put_text(&mut m, "☺", Point::new(5, 30), FontFace::Simplex, 1.0, Scalar::all(255.0), 1, false)
            .unwrap();
        assert_eq!(lit(&m), 0, "the fallback is the blank space glyph");
    }

    #[test]
    fn zero_scale_is_rejected() {
        let mut m = canvas(20, 20);
        assert!(put_text(&mut m, "x", Point::new(0, 10), FontFace::Simplex, 0.0, Scalar::all(255.0), 1, false).is_err());
    }
}
