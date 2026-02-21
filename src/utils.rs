// file: src/utils.rs
// description: Utility helpers for buffer conversion, tensor math, and shape-safe operations.
// author: cipher-rc5
// created: 2026-02-21
// modified: 2026-02-21

use crate::backend::metal::{MetalBuffer, MetalDevice, StorageMode};
use anyhow::Result;
use half::f16;

pub fn buffer_to_f32_vec(buffer: &MetalBuffer) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; buffer.size()];
    buffer.read_data(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() % std::mem::size_of::<f16>() == 0,
        "Buffer size is not aligned to f16 elements"
    );
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(f16::from_bits(bits).to_f32());
    }
    Ok(out)
}

pub fn buffer_from_f32(device: &std::sync::Arc<MetalDevice>, data: &[f32]) -> Result<MetalBuffer> {
    let mut bytes = Vec::with_capacity(data.len() * std::mem::size_of::<f16>());
    for &value in data {
        bytes.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
    }
    let buffer = device.allocate_buffer(bytes.len(), StorageMode::Shared)?;
    buffer.write_data(&bytes)?;
    Ok(buffer)
}

pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // Computes A(m x k) * B^T, where B is stored as (n x k).
    // This matches common checkpoint layout for linear weights: (out_features, in_features).
    // Validate dimensions before computation
    let expected_a_len = m * k;
    let expected_b_len = k * n;

    if a.len() != expected_a_len {
        tracing::error!(
            "matmul dimension mismatch: matrix A has length {} but expected {} (m={}, k={})",
            a.len(),
            expected_a_len,
            m,
            k
        );
        panic!(
            "matmul: matrix A dimension mismatch: got {}, expected {} ({}x{})",
            a.len(),
            expected_a_len,
            m,
            k
        );
    }

    if b.len() != expected_b_len {
        tracing::error!(
            "matmul dimension mismatch: matrix B has length {} but expected {} (k={}, n={})",
            b.len(),
            expected_b_len,
            k,
            n
        );
        panic!(
            "matmul: matrix B dimension mismatch: got {}, expected {} ({}x{})",
            b.len(),
            expected_b_len,
            k,
            n
        );
    }

    let mut output = vec![0.0f32; m * n];

    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for inner in 0..k {
                let a_idx = row * k + inner;
                let b_idx = col * k + inner;
                sum += a[a_idx] * b[b_idx];
            }
            output[row * n + col] = sum;
        }
    }
    output
}

pub fn softmax_inplace(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in values.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum == 0.0 {
        let uniform = 1.0 / values.len() as f32;
        for v in values.iter_mut() {
            *v = uniform;
        }
    } else {
        for v in values.iter_mut() {
            *v /= sum;
        }
    }
}

pub fn add_tensors(
    a: &MetalBuffer,
    b: &MetalBuffer,
    compute: &crate::backend::metal::MetalCompute,
) -> Result<MetalBuffer> {
    anyhow::ensure!(a.size() == b.size(), "Mismatched tensor sizes for addition");
    let a_vals = buffer_to_f32_vec(a)?;
    let b_vals = buffer_to_f32_vec(b)?;

    let mut result = vec![0.0f32; a_vals.len()];
    for i in 0..a_vals.len() {
        result[i] = a_vals[i] + b_vals[i];
    }

    buffer_from_f32(&compute.device, &result)
}

pub fn apply_bias(matrix: &mut [f32], rows: usize, cols: usize, bias: &[f32]) -> Result<()> {
    anyhow::ensure!(
        bias.len() == cols,
        "Bias length {} does not match columns {}",
        bias.len(),
        cols
    );

    let expected_matrix_len = rows * cols;
    anyhow::ensure!(
        matrix.len() == expected_matrix_len,
        "Matrix length {} does not match expected {} (rows={}, cols={})",
        matrix.len(),
        expected_matrix_len,
        rows,
        cols
    );

    for row in 0..rows {
        for col in 0..cols {
            matrix[row * cols + col] += bias[col];
        }
    }
    Ok(())
}

pub fn apply_bias_safe(
    matrix: &mut [f32],
    rows: usize,
    cols: usize,
    bias: &[f32],
    bias_name: &str,
) -> Result<()> {
    if bias.len() == cols {
        apply_bias(matrix, rows, cols, bias)
    } else if bias.len() > cols && bias.len() % cols == 0 {
        tracing::warn!(
            "{} bias length {} does not match expected columns {}. Using first {} elements.",
            bias_name,
            bias.len(),
            cols,
            cols
        );
        apply_bias(matrix, rows, cols, &bias[..cols])
    } else {
        anyhow::bail!(
            "{} bias length {} cannot be reconciled with columns {}. Not a clean multiple.",
            bias_name,
            bias.len(),
            cols
        )
    }
}

pub fn head_slice<'a>(
    data: &'a [f32],
    token_idx: usize,
    head_idx: usize,
    num_heads: usize,
    head_dim: usize,
) -> &'a [f32] {
    let hidden_size = num_heads * head_dim;
    let start = token_idx * hidden_size + head_idx * head_dim;
    let end = start + head_dim;

    if end > data.len() {
        panic!(
            "head_slice: index out of bounds. Trying to access [{}..{}] but data.len()={} (token_idx={}, head_idx={}, num_heads={}, head_dim={})",
            start,
            end,
            data.len(),
            token_idx,
            head_idx,
            num_heads,
            head_dim
        );
    }

    &data[start..end]
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845608f32;
    let coeff = 0.044715f32;
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + coeff * x * x * x)).tanh())
}

pub fn silu(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-x).exp()))
}

pub fn buffer_to_u8_vec(buffer: &crate::backend::metal::MetalBuffer) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; buffer.size()];
    buffer.read_data(&mut bytes)?;
    Ok(bytes)
}

pub fn dequantize_buffer(
    data_buffer: &crate::backend::metal::MetalBuffer,
    scales_buffer: &crate::backend::metal::MetalBuffer,
    biases_buffer: &Option<crate::backend::metal::MetalBuffer>,
    group_size: usize,
) -> Result<Vec<f32>> {
    let data = buffer_to_u8_vec(data_buffer)?;
    let scales = buffer_to_f32_vec(scales_buffer)?;
    let biases = if let Some(biases_buffer) = biases_buffer {
        Some(buffer_to_f32_vec(biases_buffer)?)
    } else {
        None
    };

    let mut dequantized = Vec::with_capacity(data.len());
    for (i, &val) in data.iter().enumerate() {
        let group_idx = i / group_size;
        let scale = scales[group_idx];
        let bias = if let Some(biases) = &biases {
            biases[group_idx]
        } else {
            0.0
        };
        dequantized.push((val as f32 - bias) * scale);
    }
    Ok(dequantized)
}
