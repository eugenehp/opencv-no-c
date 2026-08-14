//! Load the ONNX fixture and print the difference from PyTorch's own output.
fn main() -> cv_core::Result<()> {
    let dir = std::path::Path::new("/tmp/onnxtest");
    let rd = |p: &str| -> Vec<f32> {
        let raw = std::fs::read(dir.join(p)).expect("fixture");
        let hl = u16::from_le_bytes([raw[8], raw[9]]) as usize;
        raw[10 + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    let input = rd("input.npy");
    let expected = rd("expected.npy");

    let mut net = cv_dnn::Net::from_onnx(dir.join("small.onnx"), "input", Some([1, 2, 8, 8]))?;
    println!("model name: {:?}", net.model_name());
    net.set_input(cv_dnn::Blob::new([1, 3, 8, 8], input)?);
    let out = net.forward()?;
    let worst = out.data.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("output shape {:?}, {} values", out.shape, out.data.len());
    println!("first 4 got     : {:?}", &out.data[..4]);
    println!("first 4 pytorch : {:?}", &expected[..4]);
    println!("max |difference|: {worst:e}");
    Ok(())
}
