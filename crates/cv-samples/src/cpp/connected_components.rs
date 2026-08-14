//! `samples/cpp/connected_components.cpp` — labelling thresholded blobs.
//!
//! Upstream's trackbar sets the threshold and starts at 100; the image is
//! thresholded to binary and the 8-connected components are labelled. The
//! label image is the result, along with the component count.

use cv_imgproc::shape::connected_components;
use cv_imgproc::threshold::{threshold, ThresholdMethod, ThresholdType};

use crate::{assets, SampleOutput};

/// The trackbar's initial position.
const THRESHVAL: f64 = 100.0;

/// Run the sample.
///
/// # Errors
///
/// Propagates any failure from thresholding or labelling.
pub fn run() -> Result<SampleOutput, cv_core::Error> {
    let img = assets::gray("stuff")?;
    // Upstream picks the comparison direction from the trackbar's sign; at the
    // initial positive value it keeps the pixels **below** the threshold.
    let bw = threshold(&img, THRESHVAL, 255.0, ThresholdType::BinaryInv, ThresholdMethod::Manual)?.dst;
    let (labels, n) = connected_components(&bw, 8)?;

    let mut out = SampleOutput::new();
    out.push_mat("binary", &bw)?;
    out.push_mat("labels", &labels)?;
    out.push("n_labels", vec![f64::from(n)]);
    Ok(out)
}
