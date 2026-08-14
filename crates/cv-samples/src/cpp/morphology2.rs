//! `samples/cpp/morphology2.cpp` — erosion, dilation, opening and closing.
//!
//! Upstream has two windows, each with a trackbar starting at `max_iters`
//! (10), which its callbacks read as offset zero — a 1x1 element. A 1x1
//! element is the identity, so running only the initial position would show
//! nothing. Every position the sample can reach is produced instead, for each
//! of the three element shapes it cycles through with the `e` key.

use cv_core::{Point, Result, Size};
use cv_imgproc::morph::{get_structuring_element, morphology_ex, MorphShape, MorphType};

use crate::{assets, SampleOutput};

/// Upstream's `max_iters`.
const MAX_ITERS: i32 = 10;

/// Run the sample.
///
/// # Errors
///
/// Propagates any failure from building the element or applying it.
pub fn run() -> Result<SampleOutput> {
    let src = assets::color("baboon")?;
    let mut out = SampleOutput::new();

    for (shape, sname) in [
        (MorphShape::Rect, "rect"),
        (MorphShape::Cross, "cross"),
        (MorphShape::Ellipse, "ellipse"),
    ] {
        // The trackbar spans `0..=2*max_iters+1` and the callback maps it to a
        // signed offset: below the midpoint it opens, above it closes.
        for pos in [0, MAX_ITERS / 2, MAX_ITERS, MAX_ITERS + 5, 2 * MAX_ITERS] {
            let n = pos - MAX_ITERS;
            let an = n.abs();
            let element = get_structuring_element(
                shape,
                Size::new(an * 2 + 1, an * 2 + 1),
                Point::new(an, an),
            )?;
            let (op, oname) = if n < 0 {
                (MorphType::Open, "open")
            } else {
                (MorphType::Close, "close")
            };
            let m = morphology_ex(
                &src,
                op,
                &element,
                Point::new(an, an),
                1,
                cv_core::BorderType::Reflect101,
            )?;
            out.push_mat(&format!("{sname}_{oname}_{an}"), &m)?;

            let (op, oname) =
                if n < 0 { (MorphType::Erode, "erode") } else { (MorphType::Dilate, "dilate") };
            let m = morphology_ex(
                &src,
                op,
                &element,
                Point::new(an, an),
                1,
                cv_core::BorderType::Reflect101,
            )?;
            out.push_mat(&format!("{sname}_{oname}_{an}"), &m)?;
        }
    }
    Ok(out)
}
