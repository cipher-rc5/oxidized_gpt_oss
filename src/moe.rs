use crate::backend::metal::{MetalBuffer, MetalCompute};
use crate::model::MLP;
use crate::utils::{buffer_from_f32, buffer_to_f32_vec, matmul, softmax_inplace};
use anyhow::Result;

pub struct MoELayer {
    pub experts: Vec<MLP>,
    pub router: MetalBuffer,
}

impl MoELayer {
    pub fn forward(
        &self,
        hidden_states: &MetalBuffer,
        compute: &MetalCompute,
    ) -> Result<MetalBuffer> {
        // 1-route tokens to experts
        let router_weights = buffer_to_f32_vec(&self.router)?;
        let hidden = buffer_to_f32_vec(hidden_states)?;
        let (seq_len, hidden_size) = (
            hidden.len() / self.experts[0].hidden_size,
            self.experts[0].hidden_size,
        );

        // Calculate actual router output dimension from weight shape
        let router_output_dim = router_weights.len() / hidden_size;

        tracing::debug!(
            "MoE router: weight_len={}, hidden_size={}, calculated_output_dim={}, num_experts={}",
            router_weights.len(),
            hidden_size,
            router_output_dim,
            self.experts.len()
        );

        let router_logits = matmul(
            &hidden,
            &router_weights,
            seq_len,
            hidden_size,
            router_output_dim,
        );

        // 2-select top-k experts
        let mut expert_indices = Vec::with_capacity(seq_len);
        let mut expert_weights = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let logits = &router_logits[i * router_output_dim..(i + 1) * router_output_dim];
            let mut sorted_logits = logits.iter().enumerate().collect::<Vec<_>>();
            sorted_logits.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top_k_indices = sorted_logits
                .iter()
                .take(2)
                .map(|(i, _)| *i)
                .collect::<Vec<_>>();
            let top_k_logits = sorted_logits
                .iter()
                .take(2)
                .map(|(_, l)| **l)
                .collect::<Vec<_>>();

            let mut softmax_weights = top_k_logits.clone();
            softmax_inplace(&mut softmax_weights);

            expert_indices.push(top_k_indices);
            expert_weights.push(softmax_weights);
        }

        // 3-process with experts
        let mut final_hidden_states = vec![0.0f32; hidden.len()];
        for i in 0..seq_len {
            let token_hidden_state = &hidden[i * hidden_size..(i + 1) * hidden_size];
            let mut expert_output = vec![0.0f32; hidden_size];

            for j in 0..2 {
                let expert_idx = expert_indices[i][j];
                let expert = &self.experts[expert_idx];
                let weight = expert_weights[i][j];

                let token_buffer = buffer_from_f32(&compute.device, token_hidden_state)?;
                let output_buffer = expert.forward(&token_buffer, compute)?;
                let output_vec = buffer_to_f32_vec(&output_buffer)?;

                for k in 0..hidden_size {
                    expert_output[k] += weight * output_vec[k];
                }
            }

            final_hidden_states[i * hidden_size..(i + 1) * hidden_size]
                .copy_from_slice(&expert_output);
        }

        buffer_from_f32(&compute.device, &final_hidden_states)
    }
}
