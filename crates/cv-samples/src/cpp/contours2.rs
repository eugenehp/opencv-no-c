//! `samples/cpp/contours2.cpp` — findContours over synthetic nested shapes.
//!
//! The input is drawn, not loaded, so this sample needs no asset at all. The
//! upstream trackbar selects a contour level from -3 to 3 and starts at 3,
//! which `drawContours` reads as "every level"; each level is produced here.

use cv_core::{Mat, Point, Result, Scalar};
use cv_imgproc::contours::{find_contours, ContourApproximation, RetrievalMode};
use cv_imgproc::draw2::ellipse;
use cv_imgproc::drawing::line;

use crate::SampleOutput;

/// Upstream's canvas is 500 square.
const W: i32 = 500;

/// Run the sample.
///
/// # Errors
///
/// Propagates any failure from drawing or contour following.
pub fn run() -> Result<SampleOutput> {
    let img = draw_faces()?;
    let (contours, hierarchy) =
        find_contours(&img, RetrievalMode::Tree, ContourApproximation::Simple)?;

    let mut out = SampleOutput::new();
    out.push_mat("drawn", &img)?;
    // The contours themselves, flattened: count, then each contour's length
    // followed by its points. This is the result the sample draws.
    let mut flat = vec![contours.len() as f64];
    for c in &contours {
        flat.push(c.len() as f64);
        for p in c {
            flat.push(f64::from(p.x));
            flat.push(f64::from(p.y));
        }
    }
    out.push("contours", flat);
    out.push(
        "hierarchy",
        hierarchy
            .iter()
            .flat_map(|h| {
                [f64::from(h.next), f64::from(h.prev), f64::from(h.first_child), f64::from(h.parent)]
            })
            .collect(),
    );
    Ok(out)
}

/// The six concentric "faces" upstream draws before following contours.
fn draw_faces() -> Result<Mat> {
    let mut img = Mat::zeros(W, W, cv_core::types::CV_8UC1)?;
    let white = Scalar::all(255.0);
    let black = Scalar::all(0.0);
    for i in 0..6 {
        let dx = (i % 2) * 250 - 30;
        let dy = (i / 2) * 150;
        if i == 0 {
            for j in 0..=10 {
                let angle = (f64::from(j) + 5.0) * core::f64::consts::PI / 21.0;
                line(
                    &mut img,
                    Point::new(
                        cv_core::cv_round(f64::from(dx + 100 + j * 10) - 80.0 * angle.cos()),
                        cv_core::cv_round(f64::from(dy + 100) - 90.0 * angle.sin()),
                    ),
                    Point::new(
                        cv_core::cv_round(f64::from(dx + 100 + j * 10) + 80.0 * angle.cos()),
                        cv_core::cv_round(f64::from(dy + 100) + 90.0 * angle.sin()),
                    ),
                    white,
                    1,
                )?;
            }
        }
        ellipse(&mut img, Point::new(dx + 150, dy + 100), (100, 70), 0.0, 0.0, 360.0, white, -1)?;
        ellipse(&mut img, Point::new(dx + 115, dy + 70), (30, 20), 0.0, 0.0, 360.0, black, -1)?;
        ellipse(&mut img, Point::new(dx + 185, dy + 70), (30, 20), 0.0, 0.0, 360.0, black, -1)?;
        ellipse(&mut img, Point::new(dx + 115, dy + 70), (15, 15), 0.0, 0.0, 360.0, white, -1)?;
        ellipse(&mut img, Point::new(dx + 185, dy + 70), (15, 15), 0.0, 0.0, 360.0, white, -1)?;
        ellipse(&mut img, Point::new(dx + 115, dy + 70), (5, 5), 0.0, 0.0, 360.0, black, -1)?;
        ellipse(&mut img, Point::new(dx + 185, dy + 70), (5, 5), 0.0, 0.0, 360.0, black, -1)?;
        ellipse(&mut img, Point::new(dx + 150, dy + 100), (10, 5), 0.0, 0.0, 360.0, black, -1)?;
        ellipse(&mut img, Point::new(dx + 150, dy + 150), (40, 10), 0.0, 0.0, 360.0, black, -1)?;
        ellipse(&mut img, Point::new(dx + 27, dy + 100), (20, 35), 0.0, 0.0, 360.0, white, -1)?;
        ellipse(&mut img, Point::new(dx + 273, dy + 100), (20, 35), 0.0, 0.0, 360.0, white, -1)?;
    }
    Ok(img)
}
