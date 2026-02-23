// file: src/inference.rs
// description: Runs prefill/decode generation with Harmony formatting, stop tokens, and sampling.
// author: cipher-rc5

use anyhow::Result;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::backend::metal::MetalDevice;
use crate::config::ModelConfig;
use crate::memory::MemoryManager;
use crate::model::GPTModel;
use crate::sampler::sample_next_token;
use crate::tokenizer::{
    ReasoningEffort, apply_harmony_chat_template, decode, encode, harmony_stop_token_ids,
    strip_harmony_reasoning,
};

pub struct InferenceEngine {
    model: GPTModel,
    device: Arc<MetalDevice>,
    config: ModelConfig,
    memory_manager: MemoryManager,
}

pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub reasoning_effort: ReasoningEffort,
    pub show_thinking: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 1.0,
            top_p: 1.0,
            reasoning_effort: ReasoningEffort::Low,
            show_thinking: false,
        }
    }
}

impl InferenceEngine {
    pub fn new(
        model_path: &std::path::Path,
        config: &ModelConfig,
        device: Arc<MetalDevice>,
    ) -> Result<Self> {
        let model = GPTModel::load_from_safetensors(model_path, config, Arc::clone(&device))?;
        let memory_manager = MemoryManager::new(false, &device)?;
        Ok(Self {
            model,
            device,
            config: config.clone(),
            memory_manager,
        })
    }

    pub fn generate_chat(
        &self,
        tokenizer: &Tokenizer,
        system: Option<&str>,
        user_prompt: &str,
        gen_config: &GenerationConfig,
    ) -> Result<String> {
        let rendered =
            apply_harmony_chat_template(system, user_prompt, gen_config.reasoning_effort);
        let mut input_ids = encode(tokenizer, &rendered)?;
        let stop_ids = harmony_stop_token_ids(tokenizer)?;
        let mut generated = Vec::new();

        self.model.reset_kv_cache();

        // Prefill on the full prompt once.
        let mut logits_buffer = self.model.forward(&input_ids, None)?;

        for _ in 0..gen_config.max_tokens {
            let logits = buffer_to_f32_logits(&logits_buffer, self.config.vocab_size)?;
            let next = sample_next_token(&logits, gen_config.temperature, gen_config.top_p)?;
            generated.push(next);
            input_ids.push(next);

            if stop_ids.contains(&next) || input_ids.len() >= self.config.max_sequence_length {
                break;
            }

            // Decode step on latest token only (KV-cache placeholder path for now).
            logits_buffer = self.model.forward(&[next], None)?;
        }

        let decoded = decode(tokenizer, &generated)?;
        self.memory_manager.cleanup_intermediate_tensors()?;
        Ok(strip_harmony_reasoning(&decoded, gen_config.show_thinking))
    }

    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()
    }
}

fn buffer_to_f32_logits(
    buffer: &crate::backend::metal::MetalBuffer,
    vocab_size: usize,
) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; vocab_size * 2];
    buffer.read_data(&mut bytes)?;
    let mut out = vec![0.0f32; vocab_size];
    for (i, c) in bytes.chunks_exact(2).enumerate() {
        out[i] = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
    }
    Ok(out)
}
