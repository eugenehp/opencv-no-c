//! Times `filter_2d` against `filter_2d_portable` so the value of the `rlx`
//! feature can be measured rather than assumed.
//!
//! `cargo run --release -p cv-imgproc --example bench_filter`
//! `cargo run --release -p cv-imgproc --features rlx --example bench_filter`
use cv_core::{types::CV_8UC3, BorderType, Mat, MatLikeMut, Point, Size};
use cv_imgproc::filter::{filter_2d, filter_2d_portable};
use std::time::Instant;

fn image(w: i32, h: i32) -> Mat {
    let mut m = Mat::zeros(h, w, CV_8UC3).unwrap();
    for y in 0..h {
        for x in 0..w {
            let p = m.pixel_mut::<u8>(y, x).unwrap();
            for (c, s) in p.iter_mut().enumerate() {
                *s = ((x * 7 + y * 13 + c as i32 * 31) % 256) as u8;
            }
        }
    }
    m
}

fn main() {
    println!("rlx feature: {}", cfg!(feature = "rlx"));
    for (w, h) in [(256, 256), (640, 480), (1280, 720)] {
        let img = image(w, h);
        for k in [3usize, 5, 7] {
            let kernel = vec![1.0 / (k * k) as f64; k * k];
            let ks = Size::new(k as i32, k as i32);

            let t0 = Instant::now();
            let a = filter_2d(&img, &kernel, ks, None, Point::new(-1, -1), 0.0, BorderType::Reflect101).unwrap();
            let dispatched = t0.elapsed();

            let t1 = Instant::now();
            let b = filter_2d_portable(&img, &kernel, ks, None, Point::new(-1, -1), 0.0, BorderType::Reflect101).unwrap();
            let portable = t1.elapsed();

            let worst = a.to_vec::<u8>().unwrap().iter()
                .zip(b.to_vec::<u8>().unwrap().iter())
                .map(|(p, q)| (*p as i32 - *q as i32).abs()).max().unwrap_or(0);

            println!(
                "  {w:>4}x{h:<4} k={k}  dispatched {:>9.2?}  portable {:>9.2?}  speedup {:>5.2}x  worst diff {worst}",
                dispatched, portable,
                portable.as_secs_f64() / dispatched.as_secs_f64()
            );
        }
    }
}
