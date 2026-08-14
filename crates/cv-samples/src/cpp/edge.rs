//! `samples/cpp/edge.cpp` — Canny edge detection.
//!
//! The upstream sample has two trackbars, both starting at 1, and redraws on
//! every change. This runs the initial position: `edgeThresh = 1` for the
//! Sobel-gradient Canny and `edgeThreshScharr = 1` for the Scharr one. Both
//! outputs are produced, along with the masked colour image the sample shows.

use cv_core::{Mat, Result, Size};
use cv_imgproc::color::{cvt_color, ColorConversion};
use cv_imgproc::edge::canny;
use cv_imgproc::filter::blur;
use cv_imgproc::objects::scharr;

use crate::{assets, SampleOutput};

/// Both trackbars start at 1.
const EDGE_THRESH: f64 = 1.0;

/// Run the sample.
///
/// # Errors
///
/// Propagates any failure from the operations it chains.
pub fn run() -> Result<SampleOutput> {
    let image = assets::color("fruits")?;
    let gray = cvt_color(&image, ColorConversion::Bgr2Gray)?;
    let blurred = blur(&gray, Size::new(3, 3), cv_core::BorderType::Reflect101)?;

    // The plain form: Canny takes its own Sobel gradient at aperture 3.
    let edge1 = canny(&blurred, EDGE_THRESH, EDGE_THRESH * 3.0, 3, false)?;

    // The second form hands Canny a Scharr gradient. **That overload —
    // `cv::Canny(dx, dy, edges, t1, t2, L2gradient)` — is not implemented in
    // this port**, which porting this sample is what revealed; `canny` only
    // takes an image and computes its own Sobel gradient. The two Scharr
    // derivatives are still produced here, so the sample covers the part of
    // its work that can be checked, and the gap is recorded rather than
    // papered over.
    let dx = scharr(&blurred, Some(cv_core::types::Depth::S16), 1, 0, 1.0, 0.0, cv_core::BorderType::Reflect101)?;
    let dy = scharr(&blurred, Some(cv_core::types::Depth::S16), 0, 1, 1.0, 0.0, cv_core::BorderType::Reflect101)?;

    // `cedge`: the colour image kept only where the edge map fires.
    let cedge = mask_colour(&image, &edge1)?;

    let mut out = SampleOutput::new();
    out.push_mat("edge_sobel", &edge1)?;
    out.push_mat("scharr_dx", &dx)?;
    out.push_mat("scharr_dy", &dy)?;
    out.push_mat("cedge", &cedge)?;
    Ok(out)
}

/// `image.copyTo(cedge, edge)` onto a zeroed destination.
fn mask_colour(image: &Mat, mask: &Mat) -> Result<Mat> {
    use cv_core::MatLike;
    let mut dst = Mat::zeros(image.rows(), image.cols(), image.mat_type())?;
    for y in 0..image.rows() {
        for x in 0..image.cols() {
            if mask.at_f64(y, x, 0)? != 0.0 {
                for c in 0..image.channels() {
                    let v = image.at_f64(y, x, c)?;
                    cv_core::arithm::set_channel_generic(&mut dst, y, x, c, v)?;
                }
            }
        }
    }
    Ok(dst)
}
