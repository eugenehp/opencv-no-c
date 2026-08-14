//! `samples/cpp/distrans.cpp` — the distance transform.
//!
//! Upstream thresholds the image at a trackbar value starting at 100, then
//! runs `distanceTransform` under whichever metric the keyboard has selected.
//! All three metrics are produced here at that initial threshold.

use cv_imgproc::contours::{distance_transform, DistanceMetric};
use cv_imgproc::threshold::{threshold, ThresholdMethod, ThresholdType};

use crate::{assets, SampleOutput};

/// The trackbar's initial position.
const EDGE_THRESH: f64 = 100.0;

/// Run the sample.
///
/// # Errors
///
/// Propagates any failure from thresholding or the transform.
pub fn run() -> Result<SampleOutput, cv_core::Error> {
    let gray = assets::gray("stuff")?;
    let edge = threshold(&gray, EDGE_THRESH, 255.0, ThresholdType::Binary, ThresholdMethod::Manual)?.dst;
    let mut out = SampleOutput::new();
    out.push_mat("thresholded", &edge)?;
    for (metric, name) in [
        (DistanceMetric::L1, "L1"),
        (DistanceMetric::L2, "L2"),
        (DistanceMetric::C, "C"),
    ] {
        out.push_mat(&format!("dist_{name}"), &distance_transform(&edge, metric)?)?;
    }
    Ok(out)
}
