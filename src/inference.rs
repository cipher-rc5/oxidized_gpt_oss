use anyhow::{anyhow, Result};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::backend::metal::{MetalBuffer, MetalDevice};
use crate::config::ModelConfig;
use crate::memory::MemoryManager;
use crate::model::GPTModel;

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
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(50),
            repetition_penalty: 1.0,
        }
    }
}

impl InferenceEngine {
    pub fn new(
        model_path: &std::path::Path,
        config: &ModelConfig,
        device: Arc<MetalDevice>,
    ) -> Result<Self> {
        info!("Initializing inference engine");

        let model = GPTModel::load_from_safetensors(model_path, config, Arc::clone(&device))?;
        let memory_manager = MemoryManager::new(false, &device)?;

        Ok(Self {
            model,
            device,
            config: config.clone(),
            memory_manager,
        })
    }

    pub fn generate(
        &self,
        prompt: &str,
        tokenizer: &Tokenizer,
        gen_config: &GenerationConfig,
    ) -> Result<String> {
        info!("Starting generation for prompt: {}", prompt);

        let encoding = tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let mut input_ids = encoding.get_ids().to_vec();

        let mut generated_tokens = Vec::new();

        for step in 0..gen_config.max_tokens {
            debug!("Generation step {}/{}", step + 1, gen_config.max_tokens);

            let logits = self.model.forward(&input_ids, None)?;

            let next_token = self.sample_token(&logits, gen_config)?;

            if next_token == tokenizer.token_to_id("<|endoftext|>").unwrap_or(50256) {
                break;
            }

            generated_tokens.push(next_token);
            input_ids.push(next_token);

            if input_ids.len() > self.config.max_sequence_length {
                input_ids.drain(0..1);
            }
        }

        let decoded = tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow!("Decoding failed: {}", e))?;

        info!(
            "Generation complete, {} tokens generated",
            generated_tokens.len()
        );

        self.memory_manager.cleanup_intermediate_tensors()?;

        Ok(decoded)
    }

    fn sample_token(&self, logits: &MetalBuffer, gen_config: &GenerationConfig) -> Result<u32> {
        let mut logits_vec = vec![0.0f32; self.config.vocab_size];

        let mut logits_bytes = vec![0u8; self.config.vocab_size * 2]; // 2 bytes per f16
        logits.read_data(&mut logits_bytes)?;

        for (i, chunk) in logits_bytes.chunks_exact(2).enumerate() {
            if i < self.config.vocab_size {
                let f16_bytes = [chunk[0], chunk[1]];
                let f16_val = half::f16::from_le_bytes(f16_bytes);
                logits_vec[i] = f16_val.to_f32();
            }
        }

        if gen_config.temperature > 0.0 {
            for logit in &mut logits_vec {
                *logit /= gen_config.temperature;
            }
        }

        let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut probs = Vec::with_capacity(logits_vec.len());

        for &logit in &logits_vec {
            let prob = (logit - max_logit).exp();
            probs.push(prob);
            sum += prob;
        }

        for prob in &mut probs {
            *prob /= sum;
        }

        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(k) = gen_config.top_k {
            indices.truncate(k);
        }

        let mut cumsum = 0.0f32;
        let mut top_p_indices = Vec::new();
        for &idx in &indices {
            cumsum += probs[idx];
            top_p_indices.push(idx);
            if cumsum >= gen_config.top_p {
                break;
            }
        }

        let rand_val = rand::random_f32();
        let mut cumsum = 0.0f32;

        for &idx in &top_p_indices {
            cumsum += probs[idx];
            if rand_val < cumsum {
                return Ok(idx as u32);
            }
        }

        Ok(top_p_indices.last().copied().unwrap_or(0) as u32)
    }

    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()
    }
}

mod rand {
    use std::cell::RefCell;

    thread_local! {
        static RNG_STATE: RefCell<u64> = RefCell::new(0x123456789abcdef0);
    }

    pub fn random_f32() -> f32 {
        RNG_STATE.with(|state| {
            let mut s = state.borrow_mut();
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            ((*s as f64) / (u64::MAX as f64)) as f32
        })
    }
}

// use anyhow::{anyhow, Result};
// use candle_core::{DType, Device, Tensor, WithDType};
// use candle_transformers::generation::LogitsProcessor;
// use std::collections::HashMap;
// use std::path::Path;
// use tokenizers::Tokenizer;
// use tracing::{debug, info};

// use crate::config::{MoEConfig, ModelConfig};
// use crate::memory::MemoryManager;
// use crate::model::GPTModel;

// pub struct InferenceEngine {
//     model: GPTModel,
//     device: Device,
//     memory_manager: MemoryManager,
//     logits_processor: LogitsProcessor,
//     moe_enabled: bool,
//     moe_metrics: Option<MoEMetrics>,
// }

// #[derive(Debug, Default)]
// pub struct MoEMetrics {
//     pub expert_usage: HashMap<usize, usize>,
//     pub routing_entropy: f32,
//     pub load_balance_loss: f32,
//     pub total_tokens_processed: usize,
//     pub layer_timings: HashMap<usize, f64>, // layer_idx -> avg_time_ms
//     pub expert_efficiency: HashMap<usize, f32>, // expert_idx -> utilization_ratio
//     pub routing_decisions: Vec<RoutingDecision>,
//     pub step_timings: Vec<f64>, // per-step generation times
// }

// #[derive(Debug, Clone)]
// pub struct RoutingDecision {
//     pub step: usize,                  // TODO: Use for tracking decision timeline
//     pub layer_idx: usize,             // TODO: Use for layer-specific analysis
//     pub selected_experts: Vec<usize>, // TODO: Use for expert selection tracking
//     pub expert_weights: Vec<f32>,     // TODO: Use for weight analysis
//     pub routing_confidence: f32,      // TODO: Use for confidence analysis
// }

// impl InferenceEngine {
//     pub async fn new(
//         model_path: &Path,
//         config: &ModelConfig,
//         device: Device,
//         use_mmap: bool,
//         precision: &str,
//     ) -> Result<Self> {
//         Self::new_with_moe(model_path, config, device, use_mmap, precision, false, None).await
//     }

//     pub async fn new_with_moe(
//         model_path: &Path,
//         config: &ModelConfig,
//         device: Device,
//         use_mmap: bool,
//         precision: &str,
//         enable_moe: bool,
//         custom_moe_config: Option<MoEConfig>,
//     ) -> Result<Self> {
//         info!(
//             "Loading model from: {} (MoE: {})",
//             model_path.display(),
//             enable_moe
//         );

//         let dtype = match precision {
//             "f16" => DType::F16,
//             "bf16" => DType::BF16,
//             "f32" => DType::F32,
//             _ => return Err(anyhow!("Unsupported precision: {}", precision)),
//         };

//         let mut memory_manager = MemoryManager::new(use_mmap, &device)?;

//         // Determine MoE configuration
//         let moe_config = if enable_moe {
//             let moe_cfg = custom_moe_config
//                 .or_else(|| config.get_moe_config())
//                 .ok_or_else(|| anyhow!("MoE enabled but no configuration available"))?;

//             // Check memory feasibility for MoE
//             Self::validate_moe_memory_requirements(&memory_manager, config, &moe_cfg, &dtype)?;

//             // Optimize memory manager for MoE
//             memory_manager.optimize_for_moe()?;

//             Some(moe_cfg)
//         } else {
//             None
//         };

//         // Load model with optional MoE support
//         let model = if let Some(ref moe_cfg) = moe_config {
//             GPTModel::load_from_path_with_moe(
//                 model_path,
//                 config,
//                 &device,
//                 dtype,
//                 &memory_manager,
//                 Some(moe_cfg),
//             )
//             .await?
//         } else {
//             GPTModel::load_from_path(model_path, config, &device, dtype, &memory_manager).await?
//         };

//         let logits_processor = LogitsProcessor::new(42, None, None);

//         // Initialize MoE metrics if enabled
//         let moe_metrics = if enable_moe {
//             Some(MoEMetrics::new(config))
//         } else {
//             None
//         };

//         info!(
//             "Model loaded successfully (MoE enabled: {})",
//             moe_config.is_some()
//         );
//         if let Some(ref moe_cfg) = moe_config {
//             info!(
//                 "MoE Configuration: {} experts, {} per token, strategy: {:?}",
//                 moe_cfg.num_experts, moe_cfg.experts_per_token, moe_cfg.routing_strategy
//             );
//         }

//         Ok(Self {
//             model,
//             device,
//             memory_manager,
//             logits_processor,
//             moe_enabled: enable_moe,
//             moe_metrics,
//         })
//     }

//     fn validate_moe_memory_requirements(
//         memory_manager: &MemoryManager,
//         config: &ModelConfig,
//         moe_config: &MoEConfig,
//         dtype: &DType,
//     ) -> Result<()> {
//         let dtype_size = match dtype {
//             DType::F32 => 4,
//             DType::F16 | DType::BF16 => 2,
//             _ => 4, // default
//         };

//         let intermediate_size = config.intermediate_size.unwrap_or(4 * config.hidden_size);

//         // Estimate total memory requirement
//         let moe_layers = config.get_num_moe_layers();
//         let total_memory_per_layer = memory_manager.estimate_moe_memory_requirements(
//             moe_config.num_experts,
//             config.hidden_size,
//             intermediate_size,
//             dtype_size,
//         );

//         let total_moe_memory = total_memory_per_layer * moe_layers;

//         // Get available system memory (simplified - you might want to use sysinfo)
//         let available_memory = 32 * 1024 * 1024 * 1024; // Assume 32GB available

//         if !memory_manager.check_moe_memory_feasibility(
//             moe_config.num_experts,
//             config.hidden_size,
//             intermediate_size,
//             dtype_size,
//             available_memory,
//         ) {
//             return Err(anyhow!(
//                 "Insufficient memory for MoE model. Required: ~{:.2} GB, Available: ~{:.2} GB. \
//                 Consider reducing num_experts or using lower precision.",
//                 total_moe_memory as f64 / 1e9,
//                 available_memory as f64 / 1e9
//             ));
//         }

//         info!(
//             "MoE memory validation passed. Estimated usage: {:.2} GB",
//             total_moe_memory as f64 / 1e9
//         );

//         Ok(())
//     }

//     pub async fn generate(
//         &mut self,
//         prompt: &str,
//         tokenizer: &Tokenizer,
//         max_tokens: usize,
//         temperature: f64,
//         top_p: f64,
//     ) -> Result<String> {
//         let encoding = tokenizer.encode(prompt, true).map_err(|e| anyhow!(e))?;
//         let input_ids = encoding.get_ids();

//         debug!("Input tokens: {} tokens", input_ids.len());

//         let input_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;

//         let generated_tokens = self
//             .generate_tokens(input_tensor, max_tokens, temperature, top_p)
//             .await?;

//         tokenizer
//             .decode(&generated_tokens, true)
//             .map_err(|e| anyhow!(e))
//     }

//     pub async fn generate_with_metrics(
//         &mut self,
//         prompt: &str,
//         tokenizer: &Tokenizer,
//         max_tokens: usize,
//         temperature: f64,
//         top_p: f64,
//     ) -> Result<(String, Option<MoEMetrics>)> {
//         // Reset metrics if MoE is enabled
//         if self.moe_enabled
//             && let Some(model_config) = self.get_model_config()
//         {
//             self.moe_metrics = Some(MoEMetrics::new(&model_config));
//         }

//         let response = self
//             .generate(prompt, tokenizer, max_tokens, temperature, top_p)
//             .await?;

//         // Finalize metrics
//         if let Some(ref mut metrics) = self.moe_metrics {
//             metrics.finalize();
//         }

//         Ok((response, self.moe_metrics.take()))
//     }

//     pub async fn generate_with_moe_metrics(
//         &mut self,
//         prompt: &str,
//         tokenizer: &Tokenizer,
//         max_tokens: usize,
//         temperature: f64,
//         top_p: f64,
//     ) -> Result<(String, MoEMetrics)> {
//         let metrics = MoEMetrics::new(&ModelConfig {
//             num_layers: 0,
//             hidden_size: 0,
//             num_attention_heads: 0,
//             intermediate_size: None,
//             vocab_size: 0,
//             tie_word_embeddings: false,
//             max_sequence_length: 0,
//             num_experts: None,
//             experts_per_token: None,
//             expert_capacity_factor: None,
//             moe_layers: None,
//             use_swiglu: false,
//             routing_strategy: None,
//         }); // TODO: Pass actual config

//         // ... existing generation code ...
//         // During generation, collect MoE metrics

//         let generated_text = self
//             .generate(prompt, tokenizer, max_tokens, temperature, top_p)
//             .await?;

//         Ok((generated_text, metrics))
//     }

//     async fn generate_tokens(
//         &mut self,
//         input_ids: Tensor,
//         max_tokens: usize,
//         temperature: f64,
//         top_p: f64,
//     ) -> Result<Vec<u32>> {
//         let mut generated_tokens = Vec::new();
//         let mut cache = self.model.create_cache()?;

//         self.logits_processor = LogitsProcessor::new(42, Some(temperature), Some(top_p));

//         for step in 0..max_tokens {
//             let step_start = std::time::Instant::now();

//             let logits = if step == 0 {
//                 self.model.forward(&input_ids, &mut cache)?
//             } else {
//                 let last_token =
//                     Tensor::new(&[generated_tokens[step - 1]], &self.device)?.unsqueeze(0)?;
//                 self.model.forward(&last_token, &mut cache)?
//             };

//             let next_token = self.logits_processor.sample(&logits)?;
//             let next_token_id = match next_token.to_scalar() {
//                 candle_core::scalar::Scalar::U32(v) => v,
//                 candle_core::scalar::Scalar::U8(v) => v as u32,
//                 candle_core::scalar::Scalar::I64(v) => v as u32,
//                 s => s.to_f64() as u32,
//             };

//             generated_tokens.push(next_token_id);

//             // Update MoE metrics if enabled
//             if let Some(ref mut metrics) = self.moe_metrics {
//                 let step_time = step_start.elapsed().as_secs_f64() * 1000.0; // ms
//                 metrics.update_step_timing(step, step_time);
//                 metrics.total_tokens_processed += 1;
//             }

//             // TODO: Make EOS token configurable
//             if next_token_id == 2 {
//                 break;
//             }

//             // Periodic memory cleanup
//             if step > 0 && step % 50 == 0 {
//                 self.memory_manager.cleanup_intermediate_tensors()?;

//                 if self.moe_enabled && step % 100 == 0 {
//                     debug!("MoE memory cleanup at step {}", step);
//                     if let Ok(stats) = self.memory_manager.get_detailed_stats() {
//                         debug!(
//                             "Current memory usage: {:.2} MB",
//                             stats.current_usage as f64 / 1e6
//                         );
//                     }
//                 }
//             }
//         }

//         Ok(generated_tokens)
//     }

//     pub fn get_memory_usage(&self) -> Result<(usize, usize)> {
//         self.memory_manager.get_usage_stats()
//     }

//     pub fn get_detailed_memory_stats(&self) -> Result<crate::memory::MemoryStats> {
//         // TODO: Use this method for detailed memory reporting
//         self.memory_manager.get_detailed_stats()
//     }

//     pub fn print_memory_summary(&self) {
//         self.memory_manager.print_memory_summary();
//     }

//     pub fn is_moe_enabled(&self) -> bool {
//         self.moe_enabled
//     }

//     pub fn get_moe_metrics(&self) -> Option<&MoEMetrics> {
//         self.moe_metrics.as_ref()
//     }

//     fn get_model_config(&self) -> Option<ModelConfig> {
//         // In a real implementation, you'd store a reference to the config
//         // For now, we'll create a default one
//         None
//     }

//     // Advanced MoE-specific methods
//     pub fn reset_moe_metrics(&mut self) {
//         if self.moe_enabled
//             && let Some(model_config) = self.get_model_config()
//         {
//             self.moe_metrics = Some(MoEMetrics::new(&model_config));
//         }
//     }

//     pub async fn benchmark_moe_performance(
//         &mut self,
//         tokenizer: &Tokenizer,
//         test_prompts: &[&str],
//         max_tokens: usize,
//     ) -> Result<MoEBenchmarkResults> {
//         if !self.moe_enabled {
//             return Err(anyhow!("MoE is not enabled for benchmarking"));
//         }

//         let mut results = MoEBenchmarkResults::new();

//         for (i, prompt) in test_prompts.iter().enumerate() {
//             info!(
//                 "Benchmarking prompt {}/{}: {}",
//                 i + 1,
//                 test_prompts.len(),
//                 prompt.chars().take(50).collect::<String>()
//             );

//             self.reset_moe_metrics();

//             let start_time = std::time::Instant::now();
//             let (response, metrics) = self
//                 .generate_with_metrics(prompt, tokenizer, max_tokens, 0.7, 0.9)
//                 .await?;
//             let total_time = start_time.elapsed();

//             if let Some(metrics) = metrics {
//                 results.add_run(BenchmarkRun {
//                     prompt: prompt.to_string(),
//                     response,
//                     total_time,
//                     metrics,
//                 });
//             }
//         }

//         results.compute_summary();
//         Ok(results)
//     }

//     // Streaming generation for real-time applications
//     pub async fn generate_stream<F>(
//         &mut self,
//         prompt: &str,
//         tokenizer: &Tokenizer,
//         max_tokens: usize,
//         temperature: f64,
//         top_p: f64,
//         mut callback: F,
//     ) -> Result<String>
//     where
//         F: FnMut(&str) -> Result<bool>, // Returns false to stop generation
//     {
//         // TODO: Use this method for streaming generation
//         // Implementation would be similar to generate_tokens but with callback
//         let encoding = tokenizer.encode(prompt, true).map_err(|e| anyhow!(e))?;
//         let input_ids = encoding.get_ids();

//         let input_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
//         let mut generated_tokens = Vec::new();
//         let mut cache = self.model.create_cache()?;
//         let mut full_response = String::new();

//         self.logits_processor = LogitsProcessor::new(42, Some(temperature), Some(top_p));

//         for step in 0..max_tokens {
//             let logits = if step == 0 {
//                 self.model.forward(&input_tensor, &mut cache)?
//             } else {
//                 let last_token =
//                     Tensor::new(&[generated_tokens[step - 1]], &self.device)?.unsqueeze(0)?;
//                 self.model.forward(&last_token, &mut cache)?
//             };

//             let next_token = self.logits_processor.sample(&logits)?;
//             let next_token_id = match next_token.to_scalar() {
//                 candle_core::scalar::Scalar::U32(v) => v,
//                 candle_core::scalar::Scalar::U8(v) => v as u32,
//                 candle_core::scalar::Scalar::I64(v) => v as u32,
//                 s => s.to_f64() as u32,
//             };

//             generated_tokens.push(next_token_id);

//             // Decode the new token and call the callback
//             if let Ok(token_text) = tokenizer.decode(&[next_token_id], false) {
//                 full_response.push_str(&token_text);

//                 // Call the callback with the new token
//                 if !callback(&token_text)? {
//                     break; // Stop generation if callback returns false
//                 }
//             }

//             // EOS check
//             if next_token_id == 2 {
//                 break;
//             }

//             // Memory cleanup
//             if step > 0 && step % 50 == 0 {
//                 self.memory_manager.cleanup_intermediate_tensors()?;
//             }
//         }

//         Ok(full_response)
//     }

//     // Expert analysis for debugging and optimization
//     pub async fn analyze_expert_usage(
//         &mut self,
//         test_prompts: &[&str],
//         tokenizer: &Tokenizer,
//     ) -> Result<ExpertAnalysis> {
//         // TODO: Use this method for expert usage analysis
//         if !self.moe_enabled {
//             return Err(anyhow!("Expert analysis requires MoE to be enabled"));
//         }

//         let mut analysis = ExpertAnalysis::new();

//         for prompt in test_prompts {
//             self.reset_moe_metrics();

//             let _ = self
//                 .generate_with_metrics(prompt, tokenizer, 100, 0.7, 0.9)
//                 .await?;

//             if let Some(metrics) = &self.moe_metrics {
//                 analysis.add_prompt_analysis(prompt, metrics);
//             }
//         }

//         analysis.compute_global_patterns();
//         Ok(analysis)
//     }

//     // Model introspection
//     pub fn get_model_info(&self) -> ModelInfo {
//         // TODO: Use this method for model introspection
//         ModelInfo {
//             moe_enabled: self.moe_enabled,
//             device: format!("{:?}", self.device),
//             memory_usage: self.get_memory_usage().unwrap_or((0, 0)),
//         }
//     }
// }

// impl MoEMetrics {
//     pub fn new(_config: &ModelConfig) -> Self {
//         Self {
//             expert_usage: HashMap::new(),
//             routing_entropy: 0.0,
//             load_balance_loss: 0.0,
//             total_tokens_processed: 0,
//             layer_timings: HashMap::new(),
//             expert_efficiency: HashMap::new(),
//             routing_decisions: Vec::new(),
//             step_timings: Vec::new(),
//         }
//     }

//     pub fn update_expert_usage(&mut self, expert_id: usize) {
//         *self.expert_usage.entry(expert_id).or_insert(0) += 1;
//     }

//     pub fn update_expert_usage_tensor(&mut self, _expert_indices: &Tensor) -> Result<()> {
//         // Update expert usage statistics from tensor
//         // This would track which experts are being used most frequently
//         Ok(())
//     }

//     pub fn update_step_timing(&mut self, step: usize, time_ms: f64) {
//         self.step_timings.push(time_ms);
//         // For layer timings, we'd need more detailed instrumentation
//         self.layer_timings.insert(step, time_ms);
//     }

//     pub fn add_routing_decision(&mut self, decision: RoutingDecision) {
//         // TODO: Use this method for tracking routing decisions
//         // Update expert usage based on routing decision
//         for &expert_id in &decision.selected_experts {
//             self.update_expert_usage(expert_id);
//         }
//         self.routing_decisions.push(decision);
//     }

//     pub fn compute_routing_entropy(&self) -> f32 {
//         let total_usage: usize = self.expert_usage.values().sum();
//         if total_usage == 0 {
//             return 0.0;
//         }

//         let mut entropy = 0.0;
//         for &usage in self.expert_usage.values() {
//             if usage > 0 {
//                 let prob = usage as f32 / total_usage as f32;
//                 entropy -= prob * prob.log2();
//             }
//         }
//         entropy
//     }

//     pub fn compute_routing_entropy_from_tensor(&self) -> f32 {
//         // Compute entropy of expert usage to measure load distribution
//         let total_usage: usize = self.expert_usage.values().sum();
//         if total_usage == 0 {
//             return 0.0;
//         }

//         let mut entropy = 0.0;
//         for &usage in self.expert_usage.values() {
//             if usage > 0 {
//                 let prob = usage as f32 / total_usage as f32;
//                 entropy -= prob * prob.log2();
//             }
//         }
//         entropy
//     }

//     pub fn compute_expert_efficiency(&mut self) {
//         let total_usage: usize = self.expert_usage.values().sum();
//         if total_usage == 0 {
//             return;
//         }

//         let num_experts = self.expert_usage.len();
//         let ideal_usage = total_usage as f32 / num_experts as f32;

//         for (&expert_id, &usage) in &self.expert_usage {
//             let efficiency = if ideal_usage > 0.0 {
//                 (usage as f32 / ideal_usage).min(1.0)
//             } else {
//                 0.0
//             };
//             self.expert_efficiency.insert(expert_id, efficiency);
//         }
//     }

//     pub fn compute_routing_confidence_stats(&self) -> (f32, f32, f32) {
//         if self.routing_decisions.is_empty() {
//             return (0.0, 0.0, 0.0);
//         }

//         let confidences: Vec<f32> = self
//             .routing_decisions
//             .iter()
//             .map(|d| d.routing_confidence)
//             .collect();

//         let mean = confidences.iter().sum::<f32>() / confidences.len() as f32;
//         let min = confidences.iter().fold(f32::INFINITY, |a, &b| a.min(b));
//         let max = confidences.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

//         (mean, min, max)
//     }

//     pub fn finalize(&mut self) {
//         self.routing_entropy = self.compute_routing_entropy();
//         self.compute_expert_efficiency();

//         // Compute load balance loss (simplified)
//         let avg_efficiency: f32 = self.expert_efficiency.values().sum::<f32>()
//             / self.expert_efficiency.len().max(1) as f32;
//         let variance: f32 = self
//             .expert_efficiency
//             .values()
//             .map(|&eff| (eff - avg_efficiency).powi(2))
//             .sum::<f32>()
//             / self.expert_efficiency.len().max(1) as f32;
//         self.load_balance_loss = variance.sqrt();
//     }

//     pub fn print_stats(&self) {
//         println!("\n=== MoE Performance Metrics ===");
//         println!("Total Tokens Processed: {}", self.total_tokens_processed);
//         println!("Routing Entropy: {:.4}", self.routing_entropy);
//         println!("Load Balance Loss: {:.4}", self.load_balance_loss);

//         // Routing confidence stats
//         let (mean_conf, min_conf, max_conf) = self.compute_routing_confidence_stats();
//         println!(
//             "Routing Confidence: {:.3} (avg), {:.3}-{:.3} (range)",
//             mean_conf, min_conf, max_conf
//         );

//         println!("\nExpert Usage:");
//         let mut usage_vec: Vec<_> = self.expert_usage.iter().collect();
//         usage_vec.sort_by_key(|&(id, _)| id);
//         for (&expert_id, &usage) in usage_vec {
//             let efficiency = self.expert_efficiency.get(&expert_id).unwrap_or(&0.0);
//             println!(
//                 "  Expert {}: {} tokens ({:.1}% efficiency)",
//                 expert_id,
//                 usage,
//                 efficiency * 100.0
//             );
//         }

//         if !self.step_timings.is_empty() {
//             let avg_time: f64 =
//                 self.step_timings.iter().sum::<f64>() / self.step_timings.len() as f64;
//             let min_time = self
//                 .step_timings
//                 .iter()
//                 .fold(f64::INFINITY, |a, &b| a.min(b));
//             let max_time = self
//                 .step_timings
//                 .iter()
//                 .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
//             println!(
//                 "\nStep Timing: {:.2} ms (avg), {:.2}-{:.2} ms (range)",
//                 avg_time, min_time, max_time
//             );
//         }

//         println!("===================================\n");
//     }

//     pub fn get_summary(&self) -> String {
//         format!(
//             "MoE Summary: {} tokens, {:.3} entropy, {:.3} load_loss, {:.1}% avg_efficiency",
//             self.total_tokens_processed,
//             self.routing_entropy,
//             self.load_balance_loss,
//             self.expert_efficiency.values().sum::<f32>()
//                 / self.expert_efficiency.len().max(1) as f32
//                 * 100.0
//         )
//     }

//     pub fn export_detailed_stats(&self) -> DetailedMoEStats {
//         // TODO: Use this method for exporting detailed statistics
//         DetailedMoEStats {
//             expert_usage: self.expert_usage.clone(),
//             routing_decisions: self.routing_decisions.clone(),
//             step_timings: self.step_timings.clone(),
//             efficiency_scores: self.expert_efficiency.clone(),
//             routing_entropy: self.routing_entropy,
//             load_balance_loss: self.load_balance_loss,
//         }
//     }
// }

// #[derive(Debug)]
// pub struct MoEBenchmarkResults {
//     pub runs: Vec<BenchmarkRun>,
//     pub summary: Option<BenchmarkSummary>,
// }

// #[derive(Debug)]
// pub struct BenchmarkRun {
//     pub prompt: String,
//     pub response: String, // TODO: Use for response quality analysis
//     pub total_time: std::time::Duration,
//     pub metrics: MoEMetrics,
// }

// #[derive(Debug)]
// pub struct BenchmarkSummary {
//     pub avg_tokens_per_sec: f64,
//     pub avg_routing_entropy: f32,
//     pub avg_load_balance_loss: f32,
//     pub total_runs: usize,
//     pub expert_usage_distribution: HashMap<usize, f32>, // expert_id -> avg_usage_percentage
//     pub performance_variance: f64,
// }

// #[derive(Debug)]
// pub struct ExpertAnalysis {
//     // TODO: Use this struct for expert specialization analysis
//     pub prompt_patterns: HashMap<String, Vec<usize>>, // prompt_type -> preferred_experts
//     pub expert_specializations: HashMap<usize, Vec<String>>, // expert_id -> domain_keywords
//     pub global_usage_patterns: HashMap<usize, f32>,   // expert_id -> overall_usage_rate
// }

// #[derive(Debug)]
// pub struct ModelInfo {
//     // TODO: Use this struct for model introspection
//     pub moe_enabled: bool,
//     pub device: String,
//     pub memory_usage: (usize, usize), // (peak, current)
// }

// #[derive(Debug, Clone)]
// pub struct DetailedMoEStats {
//     // TODO: Use this struct for detailed MoE statistics export
//     pub expert_usage: HashMap<usize, usize>,
//     pub routing_decisions: Vec<RoutingDecision>,
//     pub step_timings: Vec<f64>,
//     pub efficiency_scores: HashMap<usize, f32>,
//     pub routing_entropy: f32,
//     pub load_balance_loss: f32,
// }

// impl MoEBenchmarkResults {
//     pub fn new() -> Self {
//         Self {
//             runs: Vec::new(),
//             summary: None,
//         }
//     }

//     pub fn add_run(&mut self, run: BenchmarkRun) {
//         self.runs.push(run);
//     }

//     pub fn compute_summary(&mut self) {
//         if self.runs.is_empty() {
//             return;
//         }

//         let total_runs = self.runs.len();
//         let total_tokens: usize = self
//             .runs
//             .iter()
//             .map(|r| r.metrics.total_tokens_processed)
//             .sum();
//         let total_time: f64 = self.runs.iter().map(|r| r.total_time.as_secs_f64()).sum();

//         let avg_tokens_per_sec = if total_time > 0.0 {
//             total_tokens as f64 / total_time
//         } else {
//             0.0
//         };

//         let avg_routing_entropy = self
//             .runs
//             .iter()
//             .map(|r| r.metrics.routing_entropy)
//             .sum::<f32>()
//             / total_runs as f32;

//         let avg_load_balance_loss = self
//             .runs
//             .iter()
//             .map(|r| r.metrics.load_balance_loss)
//             .sum::<f32>()
//             / total_runs as f32;

//         // Compute performance variance
//         let tokens_per_sec_values: Vec<f64> = self
//             .runs
//             .iter()
//             .map(|r| {
//                 if r.total_time.as_secs_f64() > 0.0 {
//                     r.metrics.total_tokens_processed as f64 / r.total_time.as_secs_f64()
//                 } else {
//                     0.0
//                 }
//             })
//             .collect();

//         let mean_tps =
//             tokens_per_sec_values.iter().sum::<f64>() / tokens_per_sec_values.len() as f64;
//         let variance = tokens_per_sec_values
//             .iter()
//             .map(|&tps| (tps - mean_tps).powi(2))
//             .sum::<f64>()
//             / tokens_per_sec_values.len() as f64;
//         let performance_variance = variance.sqrt();

//         // Compute expert usage distribution
//         let mut expert_usage_totals: HashMap<usize, usize> = HashMap::new();
//         let mut total_expert_calls = 0;

//         for run in &self.runs {
//             for (&expert_id, &usage) in &run.metrics.expert_usage {
//                 *expert_usage_totals.entry(expert_id).or_insert(0) += usage;
//                 total_expert_calls += usage;
//             }
//         }

//         let expert_usage_distribution = expert_usage_totals
//             .into_iter()
//             .map(|(expert_id, total_usage)| {
//                 let percentage = if total_expert_calls > 0 {
//                     (total_usage as f32 / total_expert_calls as f32) * 100.0
//                 } else {
//                     0.0
//                 };
//                 (expert_id, percentage)
//             })
//             .collect();

//         self.summary = Some(BenchmarkSummary {
//             avg_tokens_per_sec,
//             avg_routing_entropy,
//             avg_load_balance_loss,
//             total_runs,
//             expert_usage_distribution,
//             performance_variance,
//         });
//     }

//     pub fn print_summary(&self) {
//         if let Some(ref summary) = self.summary {
//             println!("\n=== MoE Benchmark Summary ===");
//             println!("Total Runs: {}", summary.total_runs);
//             println!(
//                 "Average Tokens/sec: {:.2} (±{:.2})",
//                 summary.avg_tokens_per_sec, summary.performance_variance
//             );
//             println!(
//                 "Average Routing Entropy: {:.4}",
//                 summary.avg_routing_entropy
//             );
//             println!(
//                 "Average Load Balance Loss: {:.4}",
//                 summary.avg_load_balance_loss
//             );

//             println!("\nExpert Usage Distribution:");
//             let mut usage_vec: Vec<_> = summary.expert_usage_distribution.iter().collect();
//             usage_vec.sort_by_key(|&(id, _)| id);
//             for (&expert_id, &percentage) in usage_vec {
//                 println!("  Expert {}: {:.1}%", expert_id, percentage);
//             }
//             println!("==============================\n");
//         } else {
//             println!("No benchmark summary available. Run compute_summary() first.");
//         }
//     }

//     pub fn export_to_json(&self) -> Result<String> {
//         // Manual serialization since we don't have Serialize implementation
//         let mut result = String::new();
//         result.push_str("{\n  \"runs\": [");

//         for (i, run) in self.runs.iter().enumerate() {
//             if i > 0 {
//                 result.push(',');
//             }
//             result.push_str(&format!(
//                 "\n    {{ \"prompt\": \"{}\", \"tokens\": {}, \"time_ms\": {} }}",
//                 run.prompt
//                     .replace("\"", "\\\"")
//                     .chars()
//                     .take(30)
//                     .collect::<String>(),
//                 run.metrics.total_tokens_processed,
//                 run.total_time.as_millis()
//             ));
//         }

//         result.push_str("\n  ]");

//         if let Some(ref summary) = self.summary {
//             result.push_str(&format!(",\n  \"summary\": {{\n    \"avg_tokens_per_sec\": {},\n    \"avg_routing_entropy\": {},\n    \"avg_load_balance_loss\": {},\n    \"total_runs\": {}\n  }}",
//                     summary.avg_tokens_per_sec,
//                     summary.avg_routing_entropy,
//                     summary.avg_load_balance_loss,
//                     summary.total_runs));
//         }

//         result.push_str("\n}");
//         Ok(result)
//     }
// }

// impl ExpertAnalysis {
//     pub fn new() -> Self {
//         // TODO: Use this method for creating expert analysis instances
//         Self {
//             prompt_patterns: HashMap::new(),
//             expert_specializations: HashMap::new(),
//             global_usage_patterns: HashMap::new(),
//         }
//     }

//     pub fn add_prompt_analysis(&mut self, prompt: &str, metrics: &MoEMetrics) {
//         // TODO: Use this method for adding prompt analysis
//         // Simple keyword-based categorization
//         let prompt_type = self.categorize_prompt(prompt);

//         // Find most used experts for this prompt
//         let mut expert_usage: Vec<_> = metrics.expert_usage.iter().collect();
//         expert_usage.sort_by(|a, b| b.1.cmp(a.1));

//         let top_experts: Vec<usize> = expert_usage
//             .iter()
//             .take(3)
//             .map(|&(&expert_id, _)| expert_id)
//             .collect();

//         self.prompt_patterns
//             .insert(prompt_type.clone(), top_experts.clone());

//         // Update expert specializations
//         for expert_id in top_experts {
//             self.expert_specializations
//                 .entry(expert_id)
//                 .or_default()
//                 .push(prompt_type.clone());
//         }
//     }

//     fn categorize_prompt(&self, prompt: &str) -> String {
//         // TODO: Use this method for prompt categorization
//         let prompt_lower = prompt.to_lowercase();

//         if prompt_lower.contains("math")
//             || prompt_lower.contains("calculate")
//             || prompt_lower.contains("equation")
//         {
//             "mathematics".to_string()
//         } else if prompt_lower.contains("code")
//             || prompt_lower.contains("program")
//             || prompt_lower.contains("function")
//         {
//             "programming".to_string()
//         } else if prompt_lower.contains("story")
//             || prompt_lower.contains("write")
//             || prompt_lower.contains("creative")
//         {
//             "creative_writing".to_string()
//         } else if prompt_lower.contains("explain")
//             || prompt_lower.contains("what is")
//             || prompt_lower.contains("how does")
//         {
//             "explanation".to_string()
//         } else if prompt_lower.contains("analyze")
//             || prompt_lower.contains("compare")
//             || prompt_lower.contains("evaluate")
//         {
//             "analysis".to_string()
//         } else {
//             "general".to_string()
//         }
//     }

//     pub fn compute_global_patterns(&mut self) {
//         // TODO: Use this method for computing global usage patterns
//         // Compute overall usage patterns across all prompts
//         let mut total_usage: HashMap<usize, usize> = HashMap::new();

//         for experts in self.prompt_patterns.values() {
//             for &expert_id in experts {
//                 *total_usage.entry(expert_id).or_insert(0) += 1;
//             }
//         }

//         let total_calls: usize = total_usage.values().sum();

//         for (expert_id, usage) in total_usage {
//             let rate = usage as f32 / total_calls as f32;
//             self.global_usage_patterns.insert(expert_id, rate);
//         }
//     }

//     pub fn print_analysis(&self) {
//         // TODO: Use this method for printing analysis results
//         println!("\n=== Expert Specialization Analysis ===");

//         println!("Prompt Type -> Preferred Experts:");
//         for (prompt_type, experts) in &self.prompt_patterns {
//             println!("  {}: {:?}", prompt_type, experts);
//         }

//         println!("\nExpert Specializations:");
//         for (expert_id, domains) in &self.expert_specializations {
//             println!("  Expert {}: {:?}", expert_id, domains);
//         }

//         println!("\nGlobal Usage Patterns:");
//         let mut usage_vec: Vec<_> = self.global_usage_patterns.iter().collect();
//         usage_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
//         for (&expert_id, &rate) in usage_vec {
//             println!("  Expert {}: {:.1}%", expert_id, rate * 100.0);
//         }

//         println!("=====================================\n");
//     }
// }
