//! One module per file in `samples/cpp/`.
//!
//! Each is named after the upstream file and performs the same computation on
//! the same input. Where the upstream sample ends in `imshow`, the port ends
//! in a [`SampleOutput`](crate::SampleOutput) carrying what would have been
//! drawn; where it ends in `cout`, the port carries the text.
//!
//! Trackbars are gone. A sample whose upstream form has a slider is run at the
//! slider's initial position — the value the program shows before anyone
//! touches it — and that position is named in the module.

use crate::Sample;

pub mod connected_components;
pub mod contours2;
pub mod distrans;
pub mod edge;
pub mod morphology2;

/// The registry, in alphabetical order.
pub static SAMPLES: &[Sample] = &[
    Sample {
        name: "connected_components",
        upstream: "samples/cpp/connected_components.cpp",
        about: "Threshold a grayscale image and label its 8-connected components.",
        run: connected_components::run,
    },
    Sample {
        name: "contours2",
        upstream: "samples/cpp/contours2.cpp",
        about: "findContours and drawContours over a synthetic set of nested shapes.",
        run: contours2::run,
    },
    Sample {
        name: "distrans",
        upstream: "samples/cpp/distrans.cpp",
        about: "The distance transform under each metric, with and without labels.",
        run: distrans::run,
    },
    Sample {
        name: "edge",
        upstream: "samples/cpp/edge.cpp",
        about: "Canny, both from its own Sobel gradient and from a Scharr one.",
        run: edge::run,
    },
    Sample {
        name: "morphology2",
        upstream: "samples/cpp/morphology2.cpp",
        about: "Open, close, erode and dilate with rectangular, elliptical and cross kernels.",
        run: morphology2::run,
    },
];
