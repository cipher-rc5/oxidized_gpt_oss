// file: src/ffn.rs
// description: SwiGLU helper for feed-forward and expert blocks.
// author: cipher-rc5

pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| {
            let sig = 1.0 / (1.0 + (-g).exp());
            (g * sig) * u
        })
        .collect()
}
