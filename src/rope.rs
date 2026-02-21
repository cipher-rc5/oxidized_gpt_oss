// file: src/rope.rs
// description: Rotary positional embedding helpers for gpt-oss style RoPE application.
// author: cipher-rc5
// created: 2026-02-21
// modified: 2026-02-21

pub struct RopeCache {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub max_seq_len: usize,
    pub head_dim: usize,
}

impl RopeCache {
    pub fn new(max_seq_len: usize, head_dim: usize, rope_base: f32) -> Self {
        let half = head_dim / 2;
        let mut cos = vec![0.0; max_seq_len * half];
        let mut sin = vec![0.0; max_seq_len * half];
        for pos in 0..max_seq_len {
            for i in 0..half {
                let theta = 1.0f32 / rope_base.powf((2.0 * i as f32) / head_dim as f32);
                let angle = pos as f32 * theta;
                cos[pos * half + i] = angle.cos();
                sin[pos * half + i] = angle.sin();
            }
        }
        Self {
            cos,
            sin,
            max_seq_len,
            head_dim,
        }
    }
}
