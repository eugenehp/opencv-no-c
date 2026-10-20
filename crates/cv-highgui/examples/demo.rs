//! A live Canny demo: two windows, a trackbar driving the threshold.
//!
//!     cargo run -p opencv-no-c-highgui --features gui --example demo
//!
//! Move the slider and the edges update; press any key or close the window to
//! quit. This is the `cv::namedWindow` + `cv::createTrackbar` + `cv::imshow`
//! loop, reshaped around eframe owning the event loop.
use cv_core::{Mat, MatLikeMut, Result};
use cv_highgui::{Session, WindowFlag};

fn main() -> Result<()> {
    // A synthetic scene with clear edges.
    let (w, h) = (320, 240);
    let mut src = Mat::zeros(h, w, cv_core::types::CV_8UC1)?;
    for y in 0..h {
        for x in 0..w {
            let v = if (x / 40 + y / 40) % 2 == 0 { 40 } else { 200 };
            src.pixel_mut::<u8>(y, x)?[0] = v;
        }
    }
    for y in 80..160 {
        for x in 100..220 {
            src.pixel_mut::<u8>(y, x)?[0] = 255;
        }
    }

    let mut s = Session::new();
    s.named_window("source", WindowFlag::AutoSize);
    s.named_window("edges", WindowFlag::Normal);
    s.create_trackbar("edges", "low", 50, 255)?;
    s.create_trackbar("edges", "high", 150, 255)?;
    s.imshow("source", &src)?;

    let mut last = (-1, -1);
    s.on_frame(move |s| {
        let lo = s.trackbar_pos("edges", "low").unwrap_or(50);
        let hi = s.trackbar_pos("edges", "high").unwrap_or(150);
        // Only recompute when a slider actually moved.
        if (lo, hi) != last {
            last = (lo, hi);
            if let Ok(e) = cv_imgproc::canny(&src, f64::from(lo), f64::from(hi), 3, false) {
                let _ = s.imshow("edges", &e);
            }
        }
        s.last_key().is_none()
    });
    s.run()
}
