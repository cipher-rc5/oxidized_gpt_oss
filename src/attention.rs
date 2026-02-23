// file: src/attention.rs
// description: Attention-related data structures for grouped-query attention and KV cache.
// author: cipher-rc5

use candle_core::Tensor;

pub struct KVCache {
    pub key: Option<Tensor>,
    pub value: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            key: None,
            value: None,
        }
    }
}

pub fn is_sliding_layer(layer_idx: usize) -> bool {
    layer_idx % 2 == 0
}
