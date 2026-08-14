//! Broad timing harness for the portable kernels.
//!
//! `cargo run --release -p cv-imgproc --example bench_kernels`
//!
//! Reports the median of several runs; single runs are too noisy to compare
//! against. Throughput is quoted in megapixels per second so numbers stay
//! comparable across image sizes.

use cv_core::{types::CV_8UC1, types::CV_8UC3, BorderType, Mat, MatLikeMut, Point, Size};
use cv_imgproc::{
    blur, cvt_color, dilate, filter_2d, gaussian_blur, get_structuring_element, median_blur,
    resize, sep_filter_2d, threshold, ColorConversion, Interpolation, MorphShape, ThresholdMethod,
    ThresholdType,
};
use std::time::{Duration, Instant};

/// Median wall time over `n` runs after one warm-up.
fn bench(n: usize, mut f: impl FnMut()) -> Duration {
    f();
    let mut s: Vec<Duration> = (0..n)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .collect();
    s.sort_unstable();
    s[n / 2]
}

fn image(w: i32, h: i32, cn: i32) -> Mat {
    let typ = if cn == 1 { CV_8UC1 } else { CV_8UC3 };
    let mut m = Mat::zeros(h, w, typ).unwrap();
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
    let (w, h) = (1280, 720);
    let mpix = (w * h) as f64 / 1e6;
    let c3 = image(w, h, 3);
    let c1 = image(w, h, 1);
    let runs = 5;

    println!("{w}x{h}, median of {runs}\n");
    println!("{:<28} {:>10} {:>12}", "kernel", "time", "Mpix/s");
    println!("{}", "-".repeat(52));

    let row = |name: &str, d: Duration| {
        println!("{:<28} {:>10.2?} {:>12.1}", name, d, mpix / d.as_secs_f64());
    };

    let k3 = vec![1.0 / 9.0; 9];
    row(
        "filter_2d 3x3 c3",
        bench(runs, || {
            filter_2d(&c3, &k3, Size::new(3, 3), None, Point::new(-1, -1), 0.0, BorderType::Reflect101).unwrap();
        }),
    );
    let k7 = vec![1.0 / 49.0; 49];
    row(
        "filter_2d 7x7 c3",
        bench(runs, || {
            filter_2d(&c3, &k7, Size::new(7, 7), None, Point::new(-1, -1), 0.0, BorderType::Reflect101).unwrap();
        }),
    );
    let s5 = vec![0.2; 5];
    row(
        "sep_filter_2d 5x5 c3",
        bench(runs, || {
            sep_filter_2d(&c3, &s5, &s5, None, Point::new(-1, -1), 0.0, BorderType::Reflect101).unwrap();
        }),
    );
    row(
        "blur 5x5 c3",
        bench(runs, || {
            blur(&c3, Size::new(5, 5), BorderType::Reflect101).unwrap();
        }),
    );
    row(
        "gaussian_blur 5x5 c3",
        bench(runs, || {
            gaussian_blur(&c3, Size::new(5, 5), 1.4, 0.0, BorderType::Reflect101).unwrap();
        }),
    );
    row(
        "cvt_color BGR2GRAY",
        bench(runs, || {
            cvt_color(&c3, ColorConversion::Bgr2Gray).unwrap();
        }),
    );
    row(
        "cvt_color BGR2HSV",
        bench(runs, || {
            cvt_color(&c3, ColorConversion::Bgr2Hsv).unwrap();
        }),
    );
    row(
        "resize half area c3",
        bench(runs, || {
            resize(&c3, Size::new(w / 2, h / 2), 0.0, 0.0, Interpolation::Area).unwrap();
        }),
    );
    row(
        "resize 1.5x linear c3",
        bench(runs, || {
            resize(&c3, Size::new(w * 3 / 2, h * 3 / 2), 0.0, 0.0, Interpolation::Linear).unwrap();
        }),
    );
    row(
        "threshold binary c1",
        bench(runs, || {
            threshold(&c1, 128.0, 255.0, ThresholdType::Binary, ThresholdMethod::Manual).unwrap();
        }),
    );
    let se = get_structuring_element(MorphShape::Rect, Size::new(3, 3), Point::new(-1, -1)).unwrap();
    row(
        "dilate 3x3 c1",
        bench(runs, || {
            dilate(&c1, &se, Point::new(-1, -1), 1, BorderType::Replicate).unwrap();
        }),
    );
    row(
        "median_blur 3 c1",
        bench(3, || {
            median_blur(&c1, 3).unwrap();
        }),
    );
}
