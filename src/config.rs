// file: src/config.rs
// description: Model configuration parsing and validation helpers for gpt-oss checkpoints.
// author: cipher-rc5

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AttentionLayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[serde(alias = "n_layer", alias = "num_hidden_layers")]
    pub num_layers: usize,
    #[serde(alias = "n_embd")]
    pub hidden_size: usize,
    #[serde(alias = "n_head")]
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub vocab_size: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(
        alias = "n_positions",
        alias = "max_position_embeddings",
        default = "default_max_sequence_length"
    )]
    pub max_sequence_length: usize, // TODO: Use this field for sequence length validation

    // MoE-specific fields
    #[serde(alias = "num_local_experts")]
    pub num_experts: Option<usize>,
    pub experts_per_token: Option<usize>,
    pub expert_capacity_factor: Option<f32>,
    pub moe_layers: Option<Vec<usize>>, // Which layers should be MoE
    #[serde(default)]
    pub use_swiglu: bool,
    pub routing_strategy: Option<String>,

    #[serde(default)]
    pub layer_types: Option<Vec<AttentionLayerType>>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rms_norm_eps: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct MoEConfig {
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub expert_capacity_factor: f32, // TODO: Use for expert capacity management
    pub use_swiglu: bool,            // TODO: Use for activation function selection
    pub routing_strategy: RoutingStrategy,
}

#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    TopK,
    Switch,
    Expert1,
}

impl std::str::FromStr for RoutingStrategy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "topk" | "top_k" => Ok(RoutingStrategy::TopK),
            "switch" => Ok(RoutingStrategy::Switch),
            "expert1" => Ok(RoutingStrategy::Expert1),
            _ => Err(anyhow::anyhow!("Unknown routing strategy: {}", s)),
        }
    }
}

fn default_max_sequence_length() -> usize {
    2048
}

impl ModelConfig {
    pub fn gpt_oss_20b() -> Self {
        let layer_types = (0..24)
            .map(|i| {
                if i % 2 == 0 {
                    AttentionLayerType::SlidingAttention
                } else {
                    AttentionLayerType::FullAttention
                }
            })
            .collect();

        Self {
            num_layers: 24,
            hidden_size: 2880,
            num_attention_heads: 64,
            num_key_value_heads: Some(8),
            intermediate_size: Some(2880),
            vocab_size: 201_088,
            tie_word_embeddings: false,
            max_sequence_length: 131_072,
            num_experts: Some(32),
            experts_per_token: Some(4),
            expert_capacity_factor: None,
            moe_layers: None,
            use_swiglu: true,
            routing_strategy: Some("topk".to_string()),
            layer_types: Some(layer_types),
            head_dim: Some(64),
            sliding_window: Some(128),
            rope_theta: Some(150_000.0),
            rms_norm_eps: Some(1e-5),
        }
    }

    pub fn load_from_path(path: &Path) -> Result<Self> {
        let config_path = path.join("config.json");
        let file = File::open(&config_path)
            .with_context(|| format!("Failed to open config.json at {:?}", config_path))?;
        let config: ModelConfig = serde_json::from_reader(file)
            .with_context(|| format!("Failed to parse config.json at {:?}", config_path))?;
        Ok(config)
    }

    pub fn validate_against_official_20b(&self) -> Result<()> {
        anyhow::ensure!(
            self.num_layers == 24,
            "expected num_layers=24 for gpt-oss-20b, got {}",
            self.num_layers
        );
        anyhow::ensure!(
            self.hidden_size == 2880,
            "expected hidden_size=2880 for gpt-oss-20b, got {}",
            self.hidden_size
        );
        anyhow::ensure!(
            self.num_attention_heads == 64,
            "expected num_attention_heads=64 for gpt-oss-20b, got {}",
            self.num_attention_heads
        );
        anyhow::ensure!(
            self.num_key_value_heads.unwrap_or(0) == 8,
            "expected num_key_value_heads=8 for gpt-oss-20b, got {:?}",
            self.num_key_value_heads
        );
        anyhow::ensure!(
            self.num_experts.unwrap_or(0) == 32,
            "expected num_experts=32 for gpt-oss-20b, got {:?}",
            self.num_experts
        );
        anyhow::ensure!(
            self.experts_per_token.unwrap_or(0) == 4,
            "expected experts_per_token=4 for gpt-oss-20b, got {:?}",
            self.experts_per_token
        );
        anyhow::ensure!(
            self.vocab_size == 201_088,
            "expected vocab_size=201088 for gpt-oss-20b, got {}",
            self.vocab_size
        );
        anyhow::ensure!(
            self.max_sequence_length == 131_072,
            "expected max_sequence_length=131072 for gpt-oss-20b, got {}",
            self.max_sequence_length
        );
        anyhow::ensure!(
            (self.rope_theta.unwrap_or_default() - 150_000.0).abs() < f32::EPSILON,
            "expected rope_theta=150000.0 for gpt-oss-20b, got {:?}",
            self.rope_theta
        );
        Ok(())
    }

    pub fn total_params(&self) -> u64 {
        let mut total_params = 0;

        // Embedding layer
        total_params += (self.vocab_size * self.hidden_size) as u64;

        // Transformer layers
        let intermediate_size = self.intermediate_size.unwrap_or(4 * self.hidden_size);

        for layer_idx in 0..self.num_layers {
            if self.is_moe_layer(layer_idx) {
                // MoE layer parameters
                let num_experts = self.num_experts.unwrap_or(8);
                let layer_params =
                    // Attention qkv
                    (self.hidden_size * 3 * self.hidden_size) as u64 +
                    // Attention proj
                    (self.hidden_size * self.hidden_size) as u64 +
                    // MoE gate
                    (self.hidden_size * num_experts) as u64 +
                    // MoE experts (w1, w2, w3 for each expert)
                    (num_experts * (
                        (self.hidden_size * intermediate_size) + // w1
                        (intermediate_size * self.hidden_size) + // w2
                        (self.hidden_size * intermediate_size)   // w3
                    )) as u64 +
                    // LayerNorms
                    (2 * self.hidden_size) as u64;
                total_params += layer_params;
            } else {
                // Regular Mlp layer parameters
                let layer_params =
                    // Attention qkv
                    (self.hidden_size * 3 * self.hidden_size) as u64 +
                    // Attention proj
                    (self.hidden_size * self.hidden_size) as u64 +
                    // Mlp fc1
                    (self.hidden_size * intermediate_size) as u64 +
                    // Mlp fc2
                    (intermediate_size * self.hidden_size) as u64 +
                    // LayerNorms
                    (2 * self.hidden_size) as u64;
                total_params += layer_params;
            }
        }

        // Final LayerNorm
        total_params += (2 * self.hidden_size) as u64;

        // LM Head
        if !self.tie_word_embeddings {
            total_params += (self.hidden_size * self.vocab_size) as u64;
        }

        total_params
    }

    pub fn supports_moe(&self) -> bool {
        self.num_experts.is_some() && self.num_experts.unwrap() > 0
    }

    pub fn get_moe_config(&self) -> Option<MoEConfig> {
        if let Some(num_experts) = self.num_experts {
            let routing_strategy = self
                .routing_strategy
                .as_ref()
                .and_then(|s| s.parse().ok())
                .unwrap_or(RoutingStrategy::TopK);

            Some(MoEConfig {
                num_experts,
                experts_per_token: self.experts_per_token.unwrap_or(2),
                expert_capacity_factor: self.expert_capacity_factor.unwrap_or(1.0),
                use_swiglu: self.use_swiglu,
                routing_strategy,
            })
        } else {
            None
        }
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        if let Some(moe_layers) = &self.moe_layers {
            moe_layers.contains(&layer_idx)
        } else {
            self.supports_moe()
        }
    }

    pub fn get_num_moe_layers(&self) -> usize {
        if let Some(moe_layers) = &self.moe_layers {
            moe_layers.len()
        } else if self.supports_moe() {
            // Count layers that would be MoE with default pattern
            (0..self.num_layers)
                .filter(|&i| self.is_moe_layer(i))
                .count()
        } else {
            0
        }
    }

    pub fn get_total_experts(&self) -> usize {
        if self.supports_moe() {
            self.num_experts.unwrap() * self.get_num_moe_layers()
        } else {
            0
        }
    }
}
