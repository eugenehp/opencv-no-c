//! Reports which RLX backends this build can reach, and times each on a
//! realistic `filter2D`-shaped correlation.
//!
//! `cargo run --release -p opencv-no-c-accel --features rlx-all --example backends`
use cv_accel::{available_backends, correlate2d_f32_on, Availability, Backend};
use std::time::Instant;

fn main() {
    println!("links C: {}", cv_accel::links_c());
    println!();
    println!("{:<8} {:<12} feature", "backend", "status");
    println!("{}", "-".repeat(40));
    for b in Backend::ALL {
        let s = match b.availability() {
            Availability::Ready => "ready",
            Availability::NoDevice => "no device",
            Availability::NotCompiled => "not built",
        };
        println!("{:<8} {:<12} {}", b.to_string(), s, b.feature());
    }

    let ready = available_backends();
    if ready.is_empty() {
        println!("\nno backend available; build with --features rlx-all");
        return;
    }

    // 3-channel 720p with a 5x5 kernel, padded as filter2D would.
    let (batch, h, w, k) = (3usize, 724usize, 1284usize, 5usize);
    let input: Vec<f32> = (0..batch * h * w).map(|i| (i % 251) as f32 / 251.0).collect();
    let kernel = vec![1.0f32 / (k * k) as f32; k * k];

    println!("\n{batch}x{h}x{w}, {k}x{k} kernel ({WARMUP} warmup + {ITERS} timed)");
    println!("{:<8} {:>12} {:>12} {:>10}", "backend", "median", "best", "vs cpu");

    // GPU timings on a shared device are extremely noisy — a single run varied
    // by 10x across invocations during development. Report the median of
    // several, and the best, so an outlier cannot masquerade as a result.
    const WARMUP: usize = 3;
    const ITERS: usize = 9;
    let time_one = |b: &Backend| -> Result<(std::time::Duration, std::time::Duration), cv_core::Error> {
        for _ in 0..WARMUP {
            correlate2d_f32_on(*b, &input, batch, h, w, &kernel, k, k)?;
        }
        let mut samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t = Instant::now();
            correlate2d_f32_on(*b, &input, batch, h, w, &kernel, k, k)?;
            samples.push(t.elapsed());
        }
        samples.sort_unstable();
        Ok((samples[ITERS / 2], samples[0]))
    };

    // Time the CPU first so every row has a baseline to compare against,
    // regardless of the order backends appear in.
    let mut rows: Vec<(Backend, std::time::Duration, std::time::Duration)> = Vec::new();
    for b in &ready {
        match time_one(b) {
            Ok((med, best)) => rows.push((*b, med, best)),
            Err(e) => println!("{:<8} {:>12} {e}", b.to_string(), "error"),
        }
    }
    let cpu_med = rows.iter().find(|(b, _, _)| b.is_cpu()).map(|(_, m, _)| *m);
    rows.sort_by_key(|(_, m, _)| *m);
    for (b, med, best) in rows {
        let rel = cpu_med
            .map(|c| format!("{:.2}x", c.as_secs_f64() / med.as_secs_f64()))
            .unwrap_or_else(|| "-".into());
        println!("{:<8} {:>12.2?} {:>12.2?} {:>10}", b.to_string(), med, best, rel);
    }
}
