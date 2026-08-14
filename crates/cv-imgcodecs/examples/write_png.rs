//! Writes sample PNG files so they can be validated against an external decoder.
use cv_core::{types::CV_8UC1, types::CV_8UC3, types::CV_8UC4, Mat, MatLikeMut};

fn main() -> cv_core::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());

    let mut rgb = Mat::zeros(32, 48, CV_8UC3)?;
    for y in 0..32 {
        for x in 0..48 {
            let p = rgb.pixel_mut::<u8>(y, x)?;
            p[0] = (x * 5) as u8;
            p[1] = (y * 8) as u8;
            p[2] = ((x + y) * 3) as u8;
        }
    }
    cv_imgcodecs::imwrite(format!("{dir}/rgb.png"), &rgb)?;

    let mut gray = Mat::zeros(16, 16, CV_8UC1)?;
    for y in 0..16 {
        for x in 0..16 {
            gray.pixel_mut::<u8>(y, x)?[0] = (y * 16 + x) as u8;
        }
    }
    cv_imgcodecs::imwrite(format!("{dir}/gray.png"), &gray)?;

    let mut rgba = Mat::zeros(8, 8, CV_8UC4)?;
    for y in 0..8 {
        for x in 0..8 {
            let p = rgba.pixel_mut::<u8>(y, x)?;
            p[0] = 200; p[1] = 100; p[2] = 50; p[3] = (x * 32) as u8;
        }
    }
    cv_imgcodecs::imwrite(format!("{dir}/rgba.png"), &rgba)?;
    println!("wrote rgb.png gray.png rgba.png to {dir}");
    Ok(())
}
