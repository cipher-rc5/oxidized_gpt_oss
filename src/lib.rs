// file: src/lib.rs
// description: Crate module wiring and public exports for runtime, model, and inference entrypoints.
// author: cipher-rc5

pub mod attention;
pub mod backend;
pub mod benchmark;
pub mod config;
pub mod convert;
pub mod dtype;
pub mod ffn;
pub mod inference;
pub mod loader;
pub mod memory;
pub mod model;
pub mod moe;
pub mod mxfp4;
pub mod rope;
pub mod sampler;
pub mod tokenizer;
pub mod utils;

pub use backend::metal::{MetalBuffer, MetalCompute, MetalDevice};
pub use config::{MoEConfig, ModelConfig, RoutingStrategy};
pub use dtype::{F6E2M3, F6E3M2, F8E8M0, MxBlock, MxFp4};
pub use inference::{GenerationConfig, InferenceEngine};
pub use loader::ModelCheckpoint;
pub use model::GPTModel;
pub use mxfp4::dequantize_mxfp4;
pub use tokenizer::{
    ReasoningEffort, apply_harmony_chat_template, load_tokenizer, strip_harmony_reasoning,
};

use anyhow::Result;
use std::sync::Arc;

pub struct GPTRuntime {
    pub device: Arc<MetalDevice>,
    pub model: GPTModel,
    pub config: ModelConfig,
}

impl GPTRuntime {
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let config = ModelConfig::load_from_path(model_path)?;
        let device = MetalDevice::new()?;
        let model = GPTModel::load_from_safetensors(model_path, &config, Arc::clone(&device))?;

        Ok(Self {
            device,
            model,
            config,
        })
    }

    pub fn generate(&self) -> Result<String> {
        Ok(String::new())
    }
}
