//! Spot-check the newly ported core operations against recorded cv2 values.
use opencv_no_c::core::*;

fn m(v: &[f64], r: i32, c: i32) -> Mat {
    Mat::from_vec(r, c, 1, v).unwrap()
}
fn close(a: f64, b: f64, t: f64, what: &str) {
    assert!((a - b).abs() < t, "{what}: {a} vs cv2 {b}");
    println!("  ok  {what:<28} {a:>12.6}  (cv2 {b:.6})");
}

fn main() -> Result<()> {
    println!("eigen:");
    let (w, v) = eigen(&m(&[4.0, 1.0, 1.0, 3.0], 2, 2))?;
    close(w.at_f64(0, 0, 0)?, 4.618034, 1e-6, "lambda0");
    close(w.at_f64(1, 0, 0)?, 2.381966, 1e-6, "lambda1");
    close(v.at_f64(0, 0, 0)?.abs(), 0.8506508, 1e-6, "v0.x");
    close(v.at_f64(0, 1, 0)?.abs(), 0.5257311, 1e-6, "v0.y");

    println!("SVD:");
    let s = svd(&m(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2))?;
    close(s.w.at_f64(0, 0, 0)?, 9.525518, 1e-6, "sigma0");
    close(s.w.at_f64(1, 0, 0)?, 0.514301, 1e-6, "sigma1");

    println!("dct (1x8):");
    let x = m(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1, 8);
    let d = dct(&x, false, false)?;
    for (i, want) in [12.727922, -6.442323, 0.0, -0.673455, 0.0, -0.200903, 0.0, -0.050702]
        .iter()
        .enumerate()
    {
        close(d.at_f64(0, i as i32, 0)?, *want, 1e-5, &format!("dct[{i}]"));
    }

    println!("dft (1x8, complex output):");
    let f = dft(&x, DftDirection::Forward, false, false)?;
    let re = [36.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0];
    let im = [0.0, 9.656854, 4.0, 1.656854, 0.0, -1.656854, -4.0, -9.656854];
    for i in 0..8 {
        close(f.at_f64(0, i, 0)?, re[i as usize], 1e-5, &format!("re[{i}]"));
        close(f.at_f64(0, i, 1)?, im[i as usize], 1e-5, &format!("im[{i}]"));
    }

    println!("PCA:");
    let pts: Vec<f64> = (0..10).flat_map(|i| [i as f64, 2.0 * i as f64]).collect();
    let p = pca_compute(&m(&pts, 10, 2), DataLayout::Rows, 1)?;
    close(p.mean.at_f64(0, 0, 0)?, 4.5, 1e-9, "mean.x");
    close(p.mean.at_f64(0, 1, 0)?, 9.0, 1e-9, "mean.y");
    close(p.eigenvectors.at_f64(0, 0, 0)?.abs(), 0.447214, 1e-6, "axis.x");
    close(p.eigenvectors.at_f64(0, 1, 0)?.abs(), 0.894427, 1e-6, "axis.y");
    close(p.eigenvalues.at_f64(0, 0, 0)?, 41.25, 1e-6, "variance");

    println!("getOptimalDFTSize:");
    assert_eq!(get_optimal_dft_size(101), 108);
    assert_eq!(get_optimal_dft_size(1001), 1024);
    println!("  ok  matches cv2 on 101 -> 108 and 1001 -> 1024");

    println!("\nall new core ops agree with cv2 4.13.0");
    Ok(())
}
