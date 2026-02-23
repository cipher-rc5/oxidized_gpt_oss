// file: src/model.rs
// description: GPT model scaffolding, layer loading, and forward-pass execution over backend buffers.
// author: cipher-rc5

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use half::{bf16, f16};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use tracing::{debug, info};

use crate::backend::metal::{MetalBuffer, MetalCompute, MetalDevice, StorageMode};
use crate::config::{AttentionLayerType, ModelConfig};
use crate::loader::ModelCheckpoint;
use crate::moe::MoELayer;

pub struct GPTModel {
    config: ModelConfig,
    device: Arc<MetalDevice>,
    compute: MetalCompute,
    embeddings: MetalBuffer,
    layers: Vec<TransformerLayer>,
    ln_f: LayerNorm,
    lm_head: Option<MetalBuffer>,
}

pub enum MlpOrMoE {
    Mlp(MLP),
    MoE(MoELayer),
}

pub struct TransformerLayer {
    attention: Attention,
    mlp_or_moe: MlpOrMoE,
    ln1: LayerNorm,
    ln2: LayerNorm,
}

pub struct Attention {
    q_proj: MetalBuffer,
    k_proj: MetalBuffer,
    v_proj: MetalBuffer,
    o_proj: MetalBuffer,
    q_biases: Option<MetalBuffer>,
    k_biases: Option<MetalBuffer>,
    v_biases: Option<MetalBuffer>,
    o_biases: Option<MetalBuffer>,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    sliding_window: Option<usize>,
    cache: Mutex<AttentionCache>,
}

#[derive(Default)]
struct AttentionCache {
    k: Vec<f32>,
    v: Vec<f32>,
    len: usize,
    kv_dim: usize,
}

pub struct MLP {
    pub gate_proj: MetalBuffer,
    pub down_proj: MetalBuffer,
    pub up_proj: MetalBuffer,
    pub gate_bias: Option<MetalBuffer>,
    pub down_bias: Option<MetalBuffer>,
    pub up_bias: Option<MetalBuffer>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

pub struct LayerNorm {
    gamma: MetalBuffer,
    beta: MetalBuffer,
    eps: f32,
}

#[derive(Deserialize)]
struct SafetensorsIndexFile {
    weight_map: HashMap<String, String>,
}

fn weight_map_cache() -> &'static Mutex<HashMap<PathBuf, Arc<HashMap<String, String>>>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<HashMap<String, String>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn checkpoint_cache() -> &'static Mutex<HashMap<PathBuf, Arc<ModelCheckpoint>>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<ModelCheckpoint>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

impl GPTModel {
    pub fn load_from_safetensors(
        path: &Path,
        config: &ModelConfig,
        device: Arc<MetalDevice>,
    ) -> Result<Self> {
        info!("Loading model from {:?}", path);

        let _ = Self::load_checkpoint(path);

        let compute = MetalCompute::new(Arc::clone(&device))?;

        let embeddings = Self::load_embedding_weights(path, config, &device)?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            info!("Loading layer {}/{}", layer_idx + 1, config.num_layers);
            layers.push(Self::load_transformer_layer(
                path, config, &device, layer_idx,
            )?);
        }

        let ln_f = Self::load_layer_norm(path, &device, "model.norm", config.hidden_size)?;

        let lm_head = if !config.tie_word_embeddings {
            Some(Self::load_tensor(path, &device, "lm_head.weight")?)
        } else {
            None
        };

        info!("Model loaded successfully");

        Ok(Self {
            config: config.clone(),
            device,
            compute,
            embeddings,
            layers,
            ln_f,
            lm_head,
        })
    }

    fn load_embedding_weights(
        path: &Path,
        _config: &ModelConfig,
        device: &Arc<MetalDevice>,
    ) -> Result<MetalBuffer> {
        Self::load_tensor(path, device, "model.embed_tokens.weight")
    }

    fn load_transformer_layer(
        path: &Path,
        config: &ModelConfig,
        device: &Arc<MetalDevice>,
        layer_idx: usize,
    ) -> Result<TransformerLayer> {
        let prefix = format!("model.layers.{}", layer_idx);

        let q_proj =
            Self::load_tensor(path, device, &format!("{}.self_attn.q_proj.weight", prefix))?;
        let k_proj =
            Self::load_tensor(path, device, &format!("{}.self_attn.k_proj.weight", prefix))?;
        let v_proj =
            Self::load_tensor(path, device, &format!("{}.self_attn.v_proj.weight", prefix))?;
        let o_proj =
            Self::load_tensor(path, device, &format!("{}.self_attn.o_proj.weight", prefix))?;

        // let q_scales =
        //     Self::load_tensor(path, device, &format!("{}.self_attn.q_proj.scales", prefix))?;
        // let k_scales =
        //     Self::load_tensor(path, device, &format!("{}.self_attn.k_proj.scales", prefix))?;
        // let v_scales =
        //     Self::load_tensor(path, device, &format!("{}.self_attn.v_proj.scales", prefix))?;
        // let o_scales =
        //     Self::load_tensor(path, device, &format!("{}.self_attn.o_proj.scales", prefix))?;

        anyhow::ensure!(
            config.num_attention_heads > 0,
            "num_attention_heads must be greater than zero"
        );
        let num_kv_heads = config
            .num_key_value_heads
            .unwrap_or(config.num_attention_heads);
        anyhow::ensure!(
            num_kv_heads > 0,
            "num_key_value_heads must be greater than zero"
        );
        anyhow::ensure!(
            config.hidden_size % config.num_attention_heads == 0,
            "hidden_size ({}) must be divisible by num_attention_heads ({})",
            config.hidden_size,
            config.num_attention_heads
        );
        anyhow::ensure!(
            config.hidden_size % num_kv_heads == 0,
            "hidden_size ({}) must be divisible by num_key_value_heads ({})",
            config.hidden_size,
            num_kv_heads
        );

        let configured_intermediate = config.intermediate_size.unwrap_or(config.hidden_size * 4);

        let q_biases = None;
        let k_biases = None;
        let v_biases = None;
        let o_biases = None;

        let hidden_size = config.hidden_size;
        let q_output_dim = (q_proj.size() / 2) / hidden_size;
        let k_output_dim = (k_proj.size() / 2) / hidden_size;
        let inferred_head_dim = (k_output_dim / num_kv_heads).max(1);
        let inferred_num_heads = (q_output_dim / inferred_head_dim).max(1);

        let attention = Attention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            // q_scales,
            // k_scales,
            // v_scales,
            // o_scales,
            q_biases,
            k_biases,
            v_biases,
            o_biases,
            // q_group_size: 64,
            // k_group_size: 64,
            // v_group_size: 64,
            // o_group_size: 64,
            num_heads: inferred_num_heads,
            head_dim: inferred_head_dim,
            hidden_size: config.hidden_size,
            sliding_window: match config.layer_types.as_ref().and_then(|v| v.get(layer_idx)) {
                Some(AttentionLayerType::SlidingAttention) => config.sliding_window,
                _ => None,
            },
            cache: Mutex::new(AttentionCache::default()),
        };

        let mlp_or_moe = if config.is_moe_layer(layer_idx) {
            let router = Self::load_tensor(path, device, &format!("{}.mlp.router.weight", prefix))?;
            let num_experts = config
                .num_experts
                .expect("MoE layer requested but num_experts missing in config");

            let gate_proj_weight = Self::load_tensor(
                path,
                device,
                &format!("{}.mlp.experts.gate_proj.weight", prefix),
            )?;
            let down_proj_weight = Self::load_tensor(
                path,
                device,
                &format!("{}.mlp.experts.down_proj.weight", prefix),
            )?;
            let up_proj_weight = Self::load_tensor(
                path,
                device,
                &format!("{}.mlp.experts.up_proj.weight", prefix),
            )?;

            let gate_proj_data = buffer_to_f32_vec(&gate_proj_weight)?;
            let down_proj_data = buffer_to_f32_vec(&down_proj_weight)?;
            let up_proj_data = buffer_to_f32_vec(&up_proj_weight)?;

            let gate_bias_weight = Self::load_tensor_optional(
                path,
                device,
                &format!("{}.mlp.experts.gate_proj.bias", prefix),
            )?;
            let down_bias_weight = Self::load_tensor_optional(
                path,
                device,
                &format!("{}.mlp.experts.down_proj.bias", prefix),
            )?;
            let up_bias_weight = Self::load_tensor_optional(
                path,
                device,
                &format!("{}.mlp.experts.up_proj.bias", prefix),
            )?;

            let gate_bias_data = gate_bias_weight
                .map(|b| buffer_to_f32_vec(&b))
                .transpose()?;
            let down_bias_data = down_bias_weight
                .map(|b| buffer_to_f32_vec(&b))
                .transpose()?;
            let up_bias_data = up_bias_weight.map(|b| buffer_to_f32_vec(&b)).transpose()?;

            let hidden_size = config.hidden_size;

            let gate_proj_data_size = gate_proj_data.len();
            let calculated_intermediate_size = gate_proj_data_size / (num_experts * hidden_size);

            if configured_intermediate != calculated_intermediate_size {
                info!(
                    "Warning: intermediate_size in config.json ({}) does not match calculated intermediate_size ({}). Using calculated size.",
                    configured_intermediate, calculated_intermediate_size
                );
            }

            let intermediate_size = calculated_intermediate_size;

            let gate_proj_expert_size = intermediate_size * hidden_size;
            let down_proj_expert_size = hidden_size * intermediate_size;
            let up_proj_expert_size = intermediate_size * hidden_size;

            let mut experts = Vec::with_capacity(num_experts);
            for i in 0..num_experts {
                let gate_proj = buffer_from_f32(
                    device,
                    &gate_proj_data[i * gate_proj_expert_size..(i + 1) * gate_proj_expert_size],
                )?;
                let down_proj = buffer_from_f32(
                    device,
                    &down_proj_data[i * down_proj_expert_size..(i + 1) * down_proj_expert_size],
                )?;
                let up_proj = buffer_from_f32(
                    device,
                    &up_proj_data[i * up_proj_expert_size..(i + 1) * up_proj_expert_size],
                )?;

                let gate_bias = if let Some(data) = &gate_bias_data {
                    Some(buffer_from_f32(
                        device,
                        &data[i * intermediate_size..(i + 1) * intermediate_size],
                    )?)
                } else {
                    None
                };

                let down_bias = if let Some(data) = &down_bias_data {
                    Some(buffer_from_f32(
                        device,
                        &data[i * hidden_size..(i + 1) * hidden_size],
                    )?)
                } else {
                    None
                };

                let up_bias = if let Some(data) = &up_bias_data {
                    Some(buffer_from_f32(
                        device,
                        &data[i * intermediate_size..(i + 1) * intermediate_size],
                    )?)
                } else {
                    None
                };

                experts.push(MLP {
                    gate_proj,
                    down_proj,
                    up_proj,
                    gate_bias,
                    down_bias,
                    up_bias,
                    hidden_size: config.hidden_size,
                    intermediate_size,
                });
            }
            MlpOrMoE::MoE(MoELayer { experts, router })
        } else {
            let gate_proj =
                Self::load_tensor(path, device, &format!("{}.mlp.gate_proj.weight", prefix))?;
            let down_proj =
                Self::load_tensor(path, device, &format!("{}.mlp.down_proj.weight", prefix))?;
            let up_proj =
                Self::load_tensor(path, device, &format!("{}.mlp.up_proj.weight", prefix))?;

            let gate_bias = Self::load_tensor_optional(
                path,
                device,
                &format!("{}.mlp.gate_proj.bias", prefix),
            )?;
            let down_bias = Self::load_tensor_optional(
                path,
                device,
                &format!("{}.mlp.down_proj.bias", prefix),
            )?;
            let up_bias =
                Self::load_tensor_optional(path, device, &format!("{}.mlp.up_proj.bias", prefix))?;

            MlpOrMoE::Mlp(MLP {
                gate_proj,
                down_proj,
                up_proj,
                gate_bias,
                down_bias,
                up_bias,
                hidden_size: config.hidden_size,
                intermediate_size: configured_intermediate,
            })
        };

        let ln1 = Self::load_layer_norm(
            path,
            device,
            &format!("{}.input_layernorm", prefix),
            config.hidden_size,
        )?;
        let ln2 = Self::load_layer_norm(
            path,
            device,
            &format!("{}.post_attention_layernorm", prefix),
            config.hidden_size,
        )?;

        Ok(TransformerLayer {
            attention,
            mlp_or_moe,
            ln1,
            ln2,
        })
    }

    fn load_layer_norm(
        path: &Path,
        device: &Arc<MetalDevice>,
        prefix: &str,
        hidden_size: usize,
    ) -> Result<LayerNorm> {
        let gamma = Self::load_tensor(path, device, &format!("{}.weight", prefix))?;
        let beta = Self::load_tensor_optional(path, device, &format!("{}.bias", prefix))?;

        let beta = if let Some(beta) = beta {
            beta
        } else {
            let beta_data = vec![0.0f32; hidden_size];
            buffer_from_f32(device, &beta_data)?
        };

        Ok(LayerNorm {
            gamma,
            beta,
            eps: 1e-5,
        })
    }

    fn load_tensor(path: &Path, device: &Arc<MetalDevice>, name: &str) -> Result<MetalBuffer> {
        if let Ok(checkpoint) = Self::load_checkpoint(path)
            && let Some(buffer) = Self::maybe_load_q2_packed_tensor(device, &checkpoint, name)?
        {
            return Ok(buffer);
        }

        if let Ok(checkpoint) = Self::load_checkpoint(path)
            && let Ok(tensor) = checkpoint.get(name)
        {
            return Self::tensor_to_buffer(device, tensor);
        }

        let safetensors_path = path.join("model.safetensors");
        if safetensors_path.exists() {
            Self::load_tensor_from_file(&safetensors_path, device, name)
        } else {
            Self::load_tensor_from_shards(path, device, name)
        }
    }

    fn load_tensor_from_shards(
        path: &Path,
        device: &Arc<MetalDevice>,
        name: &str,
    ) -> Result<MetalBuffer> {
        let weight_map = Self::load_weight_map(path)?;
        let shard_name = weight_map
            .get(name)
            .with_context(|| format!("Tensor {} not found in weight map", name))?;
        let shard_path = path.join(shard_name);
        Self::load_tensor_from_file(&shard_path, device, name)
    }

    fn load_weight_map(path: &Path) -> Result<Arc<HashMap<String, String>>> {
        use std::fs::File;

        let cache_key = path.to_path_buf();
        {
            let cache = weight_map_cache();
            if let Some(map) = cache.lock().unwrap().get(&cache_key) {
                return Ok(map.clone());
            }
        }

        let index_path = path.join("model.safetensors.index.json");
        let file =
            File::open(&index_path).with_context(|| format!("Failed to open {:?}", index_path))?;
        let index: SafetensorsIndexFile = serde_json::from_reader(file)
            .with_context(|| format!("Failed to parse {:?}", index_path))?;
        let map = Arc::new(index.weight_map);

        let mut cache = weight_map_cache().lock().unwrap();
        cache.insert(cache_key, map.clone());
        Ok(map)
    }

    fn load_tensor_from_file(
        file_path: &Path,
        device: &Arc<MetalDevice>,
        name: &str,
    ) -> Result<MetalBuffer> {
        use memmap2::Mmap;
        use std::fs::File;

        let file =
            File::open(file_path).with_context(|| format!("Failed to open {:?}", file_path))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
        let tensor = tensors
            .tensor(name)
            .with_context(|| format!("Tensor {} not found", name))?;

        let data = tensor.data();
        let buffer = device.allocate_buffer(data.len(), StorageMode::Shared)?;
        buffer.write_data(data)?;
        Ok(buffer)
    }

    fn load_tensor_optional(
        path: &Path,
        device: &Arc<MetalDevice>,
        name: &str,
    ) -> Result<Option<MetalBuffer>> {
        match Self::load_tensor(path, device, name) {
            Ok(buffer) => Ok(Some(buffer)),
            Err(_) => Ok(None),
        }
    }

    fn load_checkpoint(path: &Path) -> Result<Arc<ModelCheckpoint>> {
        let key = path.to_path_buf();
        {
            let cache = checkpoint_cache();
            if let Some(cp) = cache.lock().unwrap().get(&key) {
                return Ok(cp.clone());
            }
        }

        let cp = Arc::new(ModelCheckpoint::load(path, &Device::Cpu)?);
        let mut cache = checkpoint_cache().lock().unwrap();
        cache.insert(key, cp.clone());
        Ok(cp)
    }

    fn tensor_to_buffer(
        device: &Arc<MetalDevice>,
        tensor: &candle_core::Tensor,
    ) -> Result<MetalBuffer> {
        let flat = tensor.flatten_all()?;
        let vals: Vec<f32> = match flat.dtype() {
            DType::BF16 => flat
                .to_vec1::<bf16>()?
                .into_iter()
                .map(|v| v.to_f32())
                .collect(),
            DType::F16 => flat
                .to_vec1::<f16>()?
                .into_iter()
                .map(|v| v.to_f32())
                .collect(),
            DType::F32 => flat.to_vec1::<f32>()?,
            DType::U8 => flat
                .to_vec1::<u8>()?
                .into_iter()
                .map(|v| v as f32)
                .collect(),
            DType::U32 => flat
                .to_vec1::<u32>()?
                .into_iter()
                .map(|v| v as f32)
                .collect(),
            DType::I32 => flat
                .to_vec1::<i32>()?
                .into_iter()
                .map(|v| v as f32)
                .collect(),
            other => anyhow::bail!("unsupported tensor dtype for model load: {other:?}"),
        };
        buffer_from_f32(device, &vals)
    }

    fn maybe_load_q2_packed_tensor(
        device: &Arc<MetalDevice>,
        checkpoint: &ModelCheckpoint,
        name: &str,
    ) -> Result<Option<MetalBuffer>> {
        if !name.ends_with(".weight") {
            return Ok(None);
        }

        let scales_name = format!("{}.scales", name.trim_end_matches(".weight"));
        let biases_name = format!("{}.biases", name.trim_end_matches(".weight"));

        let Ok(weight) = checkpoint.get(name) else {
            return Ok(None);
        };
        if weight.dtype() != DType::U8 {
            return Ok(None);
        }
        let Ok(scales) = checkpoint.get(&scales_name) else {
            return Ok(None);
        };
        let Ok(biases) = checkpoint.get(&biases_name) else {
            return Ok(None);
        };

        let w_shape = weight.shape().dims();
        let s_shape = scales.shape().dims();
        let b_shape = biases.shape().dims();
        if w_shape.len() < 2 || s_shape.len() < 2 || b_shape.len() < 2 {
            return Ok(None);
        }

        let w_rows = w_shape[..w_shape.len() - 1].iter().product::<usize>();
        let s_rows = s_shape[..s_shape.len() - 1].iter().product::<usize>();
        let b_rows = b_shape[..b_shape.len() - 1].iter().product::<usize>();
        let w_cols = *w_shape.last().unwrap();
        let s_cols = *s_shape.last().unwrap();
        let b_cols = *b_shape.last().unwrap();

        if w_rows != s_rows || w_rows != b_rows || s_cols != b_cols {
            return Ok(None);
        }
        if s_cols == 0 || w_cols == 0 || w_cols % s_cols != 0 {
            return Ok(None);
        }

        let w_vals = Self::tensor_to_u8_vec(weight)?;
        let s_vals = Self::tensor_to_f32_vec(scales)?;
        let b_vals = Self::tensor_to_f32_vec(biases)?;

        let packed_2bit_mode = (w_cols * 4) % 64 == 0 && (w_cols * 4) / 64 == s_cols;

        let out = if packed_2bit_mode {
            let cols = w_cols * 4;
            let mut out = vec![0.0f32; w_rows * cols];
            for r in 0..w_rows {
                for c_pack in 0..w_cols {
                    let src_idx = r * w_cols + c_pack;
                    let byte = w_vals[src_idx];
                    let q = [
                        byte & 0x03,
                        (byte >> 2) & 0x03,
                        (byte >> 4) & 0x03,
                        (byte >> 6) & 0x03,
                    ];
                    for i in 0..4 {
                        let c = c_pack * 4 + i;
                        let sb_idx = r * s_cols + (c / 64);
                        out[r * cols + c] = q[i] as f32 * s_vals[sb_idx] + b_vals[sb_idx];
                    }
                }
            }
            out
        } else {
            // Quantization groups partition the last dimension directly.
            let group_size = w_cols / s_cols;
            if group_size == 0 {
                return Ok(None);
            }
            let mut out = vec![0.0f32; w_rows * w_cols];
            for r in 0..w_rows {
                for c in 0..w_cols {
                    let src_idx = r * w_cols + c;
                    let sb_idx = r * s_cols + (c / group_size);
                    out[src_idx] = w_vals[src_idx] as f32 * s_vals[sb_idx] + b_vals[sb_idx];
                }
            }
            out
        };

        Ok(Some(buffer_from_f32(device, &out)?))
    }

    fn tensor_to_u8_vec(tensor: &candle_core::Tensor) -> Result<Vec<u8>> {
        let flat = tensor.flatten_all()?;
        match flat.dtype() {
            DType::U8 => Ok(flat.to_vec1::<u8>()?),
            other => anyhow::bail!("expected U8 tensor for quantized weight, got {other:?}"),
        }
    }

    fn tensor_to_f32_vec(tensor: &candle_core::Tensor) -> Result<Vec<f32>> {
        let flat = tensor.flatten_all()?;
        match flat.dtype() {
            DType::BF16 => Ok(flat
                .to_vec1::<bf16>()?
                .into_iter()
                .map(|v| v.to_f32())
                .collect()),
            DType::F16 => Ok(flat
                .to_vec1::<f16>()?
                .into_iter()
                .map(|v| v.to_f32())
                .collect()),
            DType::F32 => Ok(flat.to_vec1::<f32>()?),
            DType::U8 => Ok(flat
                .to_vec1::<u8>()?
                .into_iter()
                .map(|v| v as f32)
                .collect()),
            DType::U32 => Ok(flat
                .to_vec1::<u32>()?
                .into_iter()
                .map(|v| v as f32)
                .collect()),
            DType::I32 => Ok(flat
                .to_vec1::<i32>()?
                .into_iter()
                .map(|v| v as f32)
                .collect()),
            other => anyhow::bail!("unsupported tensor dtype in conversion: {other:?}"),
        }
    }

    pub fn forward(&self, input_ids: &[u32], _position_ids: Option<&[u32]>) -> Result<MetalBuffer> {
        let mut hidden_states = self.embed_tokens(input_ids)?;

        let emb_vec = buffer_to_f32_vec(&hidden_states)?;
        tracing::info!(
            "After embedding: buffer_size={}, seq_len={}, dim_per_token={}",
            emb_vec.len(),
            input_ids.len(),
            emb_vec.len() / input_ids.len()
        );

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            debug!("Processing layer {}", layer_idx);
            hidden_states = layer.forward(&hidden_states, &self.compute)?;
        }

        // Infer hidden dimension from buffer size
        let hidden_vec = buffer_to_f32_vec(&hidden_states)?;
        let seq_len = input_ids.len();
        let hidden_dim = hidden_vec.len() / seq_len;

        hidden_states = self
            .ln_f
            .forward(&hidden_states, &self.compute, hidden_dim)?;

        let logits = self.compute_lm_logits(&hidden_states)?;

        Ok(logits)
    }

    pub fn reset_kv_cache(&self) {
        for layer in &self.layers {
            layer.attention.clear_cache();
        }
    }

    fn embed_tokens(&self, input_ids: &[u32]) -> Result<MetalBuffer> {
        let seq_len = input_ids.len();
        let embeddings = buffer_to_f32_vec(&self.embeddings)?;

        // Calculate actual embedding dimension from the embedding matrix
        let actual_embedding_dim = embeddings.len() / self.config.vocab_size;

        // Get the expected dimension from the first layer's layer norm
        let first_layer_ln_gamma = buffer_to_f32_vec(&self.layers[0].ln1.gamma)?;
        let expected_dim = first_layer_ln_gamma.len();

        tracing::debug!(
            "Embeddings: total_len={}, vocab_size={}, calculated_embedding_dim={}, expected_layer_dim={}, config_hidden_size={}",
            embeddings.len(),
            self.config.vocab_size,
            actual_embedding_dim,
            expected_dim,
            self.config.hidden_size
        );

        // Use the expected dimension (pad or truncate as needed)
        let output_dim = expected_dim;
        let copy_dim = actual_embedding_dim.min(expected_dim);

        tracing::info!(
            "Embedding dimension adjustment: actual={}, expected={}, copy={}, output={}",
            actual_embedding_dim,
            expected_dim,
            copy_dim,
            output_dim
        );

        let mut output = vec![0.0f32; seq_len * output_dim];

        for (t_idx, &token_id) in input_ids.iter().enumerate() {
            let token_id = token_id as usize;
            anyhow::ensure!(
                token_id < self.config.vocab_size,
                "Token id {} out of bounds",
                token_id
            );
            let src_start = token_id * actual_embedding_dim;
            let dst_start = t_idx * output_dim;
            // Copy the embedding (up to copy_dim elements), rest will be zero-padded
            output[dst_start..dst_start + copy_dim]
                .copy_from_slice(&embeddings[src_start..src_start + copy_dim]);
        }

        buffer_from_f32(&self.device, &output)
    }

    fn compute_lm_logits(&self, hidden_states: &MetalBuffer) -> Result<MetalBuffer> {
        let weight = if let Some(ref lm_head) = self.lm_head {
            lm_head
        } else {
            &self.embeddings
        };

        let hidden = buffer_to_f32_vec(hidden_states)?;
        let weight_data = buffer_to_f32_vec(weight)?;
        let vocab_size = self.config.vocab_size;

        // Infer the actual hidden dimension from the weight matrix
        let actual_hidden_dim = weight_data.len() / vocab_size;

        tracing::debug!(
            "LM head: weight_len={}, vocab_size={}, calculated_hidden_dim={}, config_hidden_size={}",
            weight_data.len(),
            vocab_size,
            actual_hidden_dim,
            self.config.hidden_size
        );

        let seq_len = hidden.len() / actual_hidden_dim;
        anyhow::ensure!(seq_len > 0, "No hidden states available for logits");
        anyhow::ensure!(
            hidden.len() == seq_len * actual_hidden_dim,
            "Hidden states size {} doesn't match expected {} (seq_len={}, hidden_dim={})",
            hidden.len(),
            seq_len * actual_hidden_dim,
            seq_len,
            actual_hidden_dim
        );

        let last_token = &hidden[(seq_len - 1) * actual_hidden_dim..seq_len * actual_hidden_dim];

        let mut logits = vec![0.0f32; vocab_size];

        for vocab_idx in 0..vocab_size {
            let weight_start = vocab_idx * actual_hidden_dim;
            let weight_row = &weight_data[weight_start..weight_start + actual_hidden_dim];
            logits[vocab_idx] = dot(last_token, weight_row);
        }

        buffer_from_f32(&self.device, &logits)
    }
}

impl TransformerLayer {
    pub fn forward(
        &self,
        hidden_states: &MetalBuffer,
        compute: &MetalCompute,
    ) -> Result<MetalBuffer> {
        let residual = hidden_states;

        // Infer actual hidden dimension by reading the layer norm gamma size
        let ln1_gamma_vec = buffer_to_f32_vec(&self.ln1.gamma)?;
        let hidden_dim = ln1_gamma_vec.len();

        tracing::debug!(
            "TransformerLayer: ln1_gamma_size={}, using_hidden_dim={}",
            ln1_gamma_vec.len(),
            hidden_dim
        );

        let normed = self.ln1.forward(hidden_states, compute, hidden_dim)?;

        let attn_output = self.attention.forward(&normed, compute)?;

        let mut hidden = add_tensors(residual, &attn_output, compute)?;

        let residual2 = &hidden;

        // Use ln2 gamma size for the second norm (should match attention output dim)
        let ln2_gamma_vec = buffer_to_f32_vec(&self.ln2.gamma)?;
        let hidden_dim2 = ln2_gamma_vec.len();

        let normed2 = self.ln2.forward(&hidden, compute, hidden_dim2)?;

        let mlp_output = match &self.mlp_or_moe {
            MlpOrMoE::Mlp(mlp) => mlp.forward(&normed2, compute)?,
            MlpOrMoE::MoE(moe) => moe.forward(&normed2, compute)?,
        };

        hidden = add_tensors(residual2, &mlp_output, compute)?;

        Ok(hidden)
    }
}

impl Attention {
    fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.k.clear();
        cache.v.clear();
        cache.len = 0;
        cache.kv_dim = 0;
    }

    pub fn forward(
        &self,
        hidden_states: &MetalBuffer,
        compute: &MetalCompute,
    ) -> Result<MetalBuffer> {
        let hidden = buffer_to_f32_vec(hidden_states)?;
        let seq_len = hidden.len() / self.hidden_size;

        tracing::debug!(
            "Attention forward: seq_len={}, hidden_size={}, num_heads={}, head_dim={}",
            seq_len,
            self.hidden_size,
            self.num_heads,
            self.head_dim
        );

        // Q projection
        let q_weight = buffer_to_f32_vec(&self.q_proj)?;

        // Calculate actual output dimension from weight matrix
        let q_output_dim = q_weight.len() / self.hidden_size;
        tracing::debug!(
            "Q projection: weight_len={}, hidden_size={}, calculated_output_dim={}",
            q_weight.len(),
            self.hidden_size,
            q_output_dim
        );

        let mut q = matmul(&hidden, &q_weight, seq_len, self.hidden_size, q_output_dim);

        if let Some(ref bias) = self.q_biases {
            let bias_vec = buffer_to_f32_vec(bias)?;
            tracing::debug!(
                "Q bias: length={}, expected={}",
                bias_vec.len(),
                q_output_dim
            );
            apply_bias_safe(&mut q, seq_len, q_output_dim, &bias_vec, "Q")?;
        }

        // K projection
        let k_weight = buffer_to_f32_vec(&self.k_proj)?;
        let k_output_dim = k_weight.len() / self.hidden_size;

        tracing::debug!(
            "K projection: weight_len={}, hidden_size={}, calculated_output_dim={}",
            k_weight.len(),
            self.hidden_size,
            k_output_dim
        );

        let mut k = matmul(&hidden, &k_weight, seq_len, self.hidden_size, k_output_dim);

        if let Some(ref bias) = self.k_biases {
            let bias_vec = buffer_to_f32_vec(bias)?;
            tracing::debug!(
                "K bias: length={}, expected={}",
                bias_vec.len(),
                k_output_dim
            );
            apply_bias_safe(&mut k, seq_len, k_output_dim, &bias_vec, "K")?;
        }

        // V projection
        let v_weight = buffer_to_f32_vec(&self.v_proj)?;
        let v_output_dim = v_weight.len() / self.hidden_size;

        tracing::debug!(
            "V projection: weight_len={}, hidden_size={}, calculated_output_dim={}",
            v_weight.len(),
            self.hidden_size,
            v_output_dim
        );

        let mut v = matmul(&hidden, &v_weight, seq_len, self.hidden_size, v_output_dim);

        if let Some(ref bias) = self.v_biases {
            let bias_vec = buffer_to_f32_vec(bias)?;
            tracing::debug!(
                "V bias: length={}, expected={}",
                bias_vec.len(),
                v_output_dim
            );
            apply_bias_safe(&mut v, seq_len, v_output_dim, &bias_vec, "V")?;
        }

        // Calculate actual number of heads from projection output dimensions
        let q_num_heads = q_output_dim / self.head_dim;
        let kv_num_heads = k_output_dim / self.head_dim;

        tracing::debug!(
            "Attention heads: q_num_heads={}, kv_num_heads={}, head_dim={}",
            q_num_heads,
            kv_num_heads,
            self.head_dim
        );

        // Maintain a lightweight KV cache for incremental decoding.
        let num_kv_groups = q_num_heads / kv_num_heads;
        let mut cache = self.cache.lock().unwrap();
        if seq_len == 1 {
            if cache.len == 0 || cache.kv_dim != k_output_dim {
                cache.k = k.clone();
                cache.v = v.clone();
                cache.len = 1;
                cache.kv_dim = k_output_dim;
            } else {
                cache.k.extend_from_slice(&k);
                cache.v.extend_from_slice(&v);
                cache.len += 1;
            }
        } else {
            cache.k = k.clone();
            cache.v = v.clone();
            cache.len = seq_len;
            cache.kv_dim = k_output_dim;
        }

        let total_len = cache.len;
        let k_all = &cache.k;
        let v_all = &cache.v;

        let mut context = vec![0.0f32; seq_len * q_output_dim];
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();

        for head in 0..q_num_heads {
            let kv_head = head / num_kv_groups;
            for target in 0..seq_len {
                let q_slice = head_slice(&q, target, head, q_num_heads, self.head_dim);
                let absolute_target = if seq_len == 1 { total_len - 1 } else { target };
                let source_start = if let Some(window) = self.sliding_window {
                    absolute_target.saturating_add(1).saturating_sub(window)
                } else {
                    0
                };
                let mut scores = Vec::with_capacity(absolute_target - source_start + 1);
                for source in source_start..=absolute_target {
                    let k_slice = head_slice(k_all, source, kv_head, kv_num_heads, self.head_dim);
                    let mut dot = 0.0f32;
                    for i in 0..self.head_dim {
                        dot += q_slice[i] * k_slice[i];
                    }
                    scores.push(dot * scale);
                }
                softmax_inplace(&mut scores);

                let out_start = target * q_output_dim + head * self.head_dim;
                let out_slice = &mut context[out_start..out_start + self.head_dim];
                for val in out_slice.iter_mut() {
                    *val = 0.0;
                }

                for (source, &weight) in (source_start..=absolute_target).zip(scores.iter()) {
                    let v_slice = head_slice(v_all, source, kv_head, kv_num_heads, self.head_dim);
                    for i in 0..self.head_dim {
                        out_slice[i] += weight * v_slice[i];
                    }
                }
            }
        }

        // O projection
        let o_weight = buffer_to_f32_vec(&self.o_proj)?;
        let o_output_dim = o_weight.len() / q_output_dim;

        tracing::debug!(
            "O projection: weight_len={}, context_dim={}, calculated_output_dim={}",
            o_weight.len(),
            q_output_dim,
            o_output_dim
        );

        let mut projected = matmul(&context, &o_weight, seq_len, q_output_dim, o_output_dim);

        if let Some(ref bias) = self.o_biases {
            let bias_vec = buffer_to_f32_vec(bias)?;
            tracing::debug!(
                "O bias: length={}, expected={}",
                bias_vec.len(),
                o_output_dim
            );
            apply_bias_safe(&mut projected, seq_len, o_output_dim, &bias_vec, "O")?;
        }

        buffer_from_f32(&compute.device, &projected)
    }
}

impl MLP {
    pub fn forward(
        &self,
        hidden_states: &MetalBuffer,
        compute: &MetalCompute,
    ) -> Result<MetalBuffer> {
        let hidden = buffer_to_f32_vec(hidden_states)?;
        let seq_len = hidden.len() / self.hidden_size;

        tracing::debug!(
            "MLP forward: seq_len={}, hidden_size={}, intermediate_size={}",
            seq_len,
            self.hidden_size,
            self.intermediate_size
        );

        // Gate projection
        let gate_proj = buffer_to_f32_vec(&self.gate_proj)?;
        let mut gate_output = matmul(
            &hidden,
            &gate_proj,
            seq_len,
            self.hidden_size,
            self.intermediate_size,
        );

        if let Some(ref bias) = self.gate_bias {
            let bias_vec = buffer_to_f32_vec(bias)?;
            tracing::debug!(
                "Gate bias: length={}, expected={}",
                bias_vec.len(),
                self.intermediate_size
            );
            apply_bias_safe(
                &mut gate_output,
                seq_len,
                self.intermediate_size,
                &bias_vec,
                "Gate",
            )?;
        }

        gate_output.iter_mut().for_each(|v| *v = silu(*v));

        // Up projection
        let up_proj = buffer_to_f32_vec(&self.up_proj)?;
        let mut up_output = matmul(
            &hidden,
            &up_proj,
            seq_len,
            self.hidden_size,
            self.intermediate_size,
        );

        if let Some(ref bias) = self.up_bias {
            let bias_vec = buffer_to_f32_vec(bias)?;
            tracing::debug!(
                "Up bias: length={}, expected={}",
                bias_vec.len(),
                self.intermediate_size
            );
            apply_bias_safe(
                &mut up_output,
                seq_len,
                self.intermediate_size,
                &bias_vec,
                "Up",
            )?;
        }

        // Element-wise multiplication
        let mut hidden_mlp = vec![0.0f32; gate_output.len()];
        for i in 0..gate_output.len() {
            hidden_mlp[i] = gate_output[i] * up_output[i];
        }

        // Down projection
        let down_proj = buffer_to_f32_vec(&self.down_proj)?;
        let mut output = matmul(
            &hidden_mlp,
            &down_proj,
            seq_len,
            self.intermediate_size,
            self.hidden_size,
        );

        if let Some(ref bias) = self.down_bias {
            let bias_vec = buffer_to_f32_vec(bias)?;
            tracing::debug!(
                "Down bias: length={}, expected={}",
                bias_vec.len(),
                self.hidden_size
            );
            apply_bias_safe(&mut output, seq_len, self.hidden_size, &bias_vec, "Down")?;
        }

        buffer_from_f32(&compute.device, &output)
    }
}

impl LayerNorm {
    pub fn forward(
        &self,
        hidden_states: &MetalBuffer,
        compute: &MetalCompute,
        hidden_size: usize,
    ) -> Result<MetalBuffer> {
        let output = compute
            .device
            .allocate_buffer(hidden_states.size(), StorageMode::Shared)?;

        let total_elements = hidden_states.size() / 2;
        let batch_size = (total_elements / hidden_size).max(1);

        compute.layernorm(
            hidden_states,
            &output,
            &self.gamma,
            &self.beta,
            batch_size,
            hidden_size,
            self.eps,
        )?;

        Ok(output)
    }
}

use crate::utils::*;
