//! Cross-check the cosine metrics against scikit-learn and SciPy.
//!
//! Round-trip tests only prove self-consistency. These reference values come
//! from `sklearn.metrics.pairwise.cosine_similarity` and
//! `scipy.spatial.distance.cosine`, computed independently of this crate.
use cv_core::{cosine_distance, cosine_similarities, cosine_similarity, Mat, MatLike, Rect};

fn load(p: &str) -> Vec<f64> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() -> cv_core::Result<()> {
    let d = std::env::args().nth(1).expect("usage: cosine_check <dir>");
    let q = load(&format!("{d}/q.bin"));
    let dd = load(&format!("{d}/D.bin"));
    let sk = load(&format!("{d}/sk_sims.bin"));
    let sp = load(&format!("{d}/scipy_dist.bin"));
    let (k, n) = (q.len(), sk.len());

    let qm = Mat::from_vec(1, k as i32, 1, &q)?;
    let dm = Mat::from_vec(n as i32, k as i32, 1, &dd)?;

    let sims = cosine_similarities(&qm, &dm)?;
    let mut worst_sk = 0f64;
    let mut worst_sp = 0f64;
    for i in 0..n {
        let got = sims.at::<f64>(0, i as i32)?;
        worst_sk = worst_sk.max((got - sk[i]).abs());
        if sp[i].is_finite() {
            let row = dm.roi(Rect::new(0, i as i32, k as i32, 1))?;
            worst_sp = worst_sp.max((cosine_distance(&qm, &row)? - sp[i]).abs());
        }
    }
    println!("vs sklearn cosine_similarity : max abs diff {worst_sk:.3e} over {n} rows");
    println!("vs scipy   cosine distance   : max abs diff {worst_sp:.3e}");

    // The degenerate row: scipy yields NaN, this crate yields a usable 0/1.
    let zero = dm.roi(Rect::new(0, 5, k as i32, 1))?;
    println!(
        "zero row -> similarity {:.1}, distance {:.1} (scipy gives {})",
        cosine_similarity(&qm, &zero)?,
        cosine_distance(&qm, &zero)?,
        sp[5]
    );
    assert!(worst_sk < 1e-12 && worst_sp < 1e-12, "cross-check failed");
    println!("OK");
    Ok(())
}
