#include <metal_stdlib>
using namespace metal;

struct MxFp4Block {
    half scale;
    packed_uchar4 data[4];
};

inline float mxfp4_to_float(uchar value, half scale) {
    int sign = (value & 0x08) ? -1 : 1;
    int exp_bits = (value >> 1) & 0x03;
    int mantissa_bit = value & 0x01;

    if (exp_bits == 0 && mantissa_bit == 0) {
        return 0.0f;
    }

    int exp = exp_bits - 2;
    float mantissa = 1.0f + float(mantissa_bit) * 0.5f;

    return sign * mantissa * exp2(float(exp)) * float(scale);
}

inline uchar float_to_mxfp4(float value, half scale) {
    float scaled = value / float(scale);
    float abs_val = abs(scaled);
    uchar sign = (value < 0.0f) ? 0x08 : 0x00;

    if (abs_val == 0.0f) {
        return sign;
    }

    int exp = int(floor(log2(abs_val)));
    exp = clamp(exp, -2, 1);
    uchar exp_bits = uchar(exp + 2) & 0x03;

    float mantissa = (abs_val / exp2(float(exp))) - 1.0f;
    uchar mantissa_bit = (mantissa >= 0.5f) ? 0x01 : 0x00;

    return sign | (exp_bits << 1) | mantissa_bit;
}

kernel void matmul_mxfp4(
    device const MxFp4Block* a [[buffer(0)]],
    device const MxFp4Block* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    constant uint* params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint M = params[0];
    const uint N = params[1];
    const uint K = params[2];

    const uint row = gid.y;
    const uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    const uint blocks_per_row_a = (K + 31) / 32;
    const uint blocks_per_row_b = (N + 31) / 32;

    for (uint k_block = 0; k_block < blocks_per_row_a; ++k_block) {
        uint a_block_idx = row * blocks_per_row_a + k_block;
        MxFp4Block a_block = a[a_block_idx];

        for (uint k_elem = 0; k_elem < 32 && (k_block * 32 + k_elem) < K; ++k_elem) {
            uint byte_idx = k_elem / 2;
            uint nibble = k_elem % 2;
            uchar a_packed = a_block.data[byte_idx / 4][byte_idx % 4];
            uchar a_val = (nibble == 0) ? (a_packed & 0x0F) : ((a_packed >> 4) & 0x0F);

            float a_float = mxfp4_to_float(a_val, a_block.scale);

            uint k_global = k_block * 32 + k_elem;
            uint b_block_idx = k_global * blocks_per_row_b + (col / 32);
            uint b_elem_in_block = col % 32;

            if (b_block_idx < (K * blocks_per_row_b)) {
                MxFp4Block b_block = b[b_block_idx];
                uint b_byte_idx = b_elem_in_block / 2;
                uint b_nibble = b_elem_in_block % 2;
                uchar b_packed = b_block.data[b_byte_idx / 4][b_byte_idx % 4];
                uchar b_val = (b_nibble == 0) ? (b_packed & 0x0F) : ((b_packed >> 4) & 0x0F);

                float b_float = mxfp4_to_float(b_val, b_block.scale);
                sum += a_float * b_float;
            }
        }
    }

    c[row * N + col] = half(sum);
}

kernel void matmul_fp16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    constant uint* params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint M = params[0];
    const uint N = params[1];
    const uint K = params[2];

    const uint row = gid.y;
    const uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    for (uint k = 0; k < K; ++k) {
        sum += float(a[row * K + k]) * float(b[k * N + col]);
    }

    c[row * N + col] = half(sum);
}

kernel void mxfp4_unpack(
    device const MxFp4Block* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint* num_blocks [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint block_idx = gid / 32;
    uint elem_idx = gid % 32;

    if (block_idx >= *num_blocks) return;

    MxFp4Block block = input[block_idx];

    uint byte_idx = elem_idx / 2;
    uint nibble = elem_idx % 2;
    uchar packed = block.data[byte_idx / 4][byte_idx % 4];
    uchar value = (nibble == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

    output[gid] = half(mxfp4_to_float(value, block.scale));
}

kernel void mxfp4_pack(
    device const half* input [[buffer(0)]],
    device MxFp4Block* output [[buffer(1)]],
    constant uint* num_blocks [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint block_idx = gid;

    if (block_idx >= *num_blocks) return;

    float max_abs = 0.0f;
    for (uint i = 0; i < 32; ++i) {
        float val = abs(float(input[block_idx * 32 + i]));
        max_abs = max(max_abs, val);
    }

    half scale = half(max_abs / 7.5f);
    if (scale == 0.0h) scale = 1.0h;

    output[block_idx].scale = scale;

    for (uint i = 0; i < 16; ++i) {
        uint idx0 = block_idx * 32 + i * 2;
        uint idx1 = idx0 + 1;

        uchar val0 = float_to_mxfp4(float(input[idx0]), scale);
        uchar val1 = float_to_mxfp4(float(input[idx1]), scale);

        uchar packed = (val0 & 0x0F) | ((val1 & 0x0F) << 4);

        uint byte_idx = i;
        output[block_idx].data[byte_idx / 4][byte_idx % 4] = packed;
    }
}

kernel void softmax(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint* params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const uint batch_idx = gid.y;
    const uint seq_len = params[0];
    const uint offset = batch_idx * seq_len;

    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];

    float local_max = -INFINITY;
    for (uint i = tid.x; i < seq_len; i += 256) {
        local_max = max(local_max, float(input[offset + i]));
    }
    shared_max[tid.x] = local_max;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid.x == 0) {
        float global_max = shared_max[0];
        for (uint i = 1; i < 256; ++i) {
            global_max = max(global_max, shared_max[i]);
        }
        shared_max[0] = global_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = shared_max[0];

    float local_sum = 0.0f;
    for (uint i = tid.x; i < seq_len; i += 256) {
        float exp_val = exp(float(input[offset + i]) - global_max);
        local_sum += exp_val;
        output[offset + i] = half(exp_val);
    }
    shared_sum[tid.x] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid.x == 0) {
        float global_sum = 0.0f;
        for (uint i = 0; i < 256; ++i) {
            global_sum += shared_sum[i];
        }
        shared_sum[0] = global_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = shared_sum[0];

    for (uint i = tid.x; i < seq_len; i += 256) {
        output[offset + i] = half(float(output[offset + i]) / global_sum);
    }
}

kernel void layernorm(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const half* gamma [[buffer(2)]],
    device const half* beta [[buffer(3)]],
    constant float* params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const uint hidden_size = uint(params[0]);
    const float eps = params[1];
    const uint batch_idx = gid;
    const uint offset = batch_idx * hidden_size;

    threadgroup float shared_sum[256];
    threadgroup float shared_sq_sum[256];

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for (uint i = tid; i < hidden_size; i += 256) {
        float val = float(input[offset + i]);
        local_sum += val;
        local_sq_sum += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sq_sum[tid] = local_sq_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float mean = 0.0f;
        float sq_mean = 0.0f;

        for (uint i = 0; i < 256; ++i) {
            mean += shared_sum[i];
            sq_mean += shared_sq_sum[i];
        }

        mean /= float(hidden_size);
        sq_mean /= float(hidden_size);

        float variance = sq_mean - mean * mean;
        float inv_std = rsqrt(variance + eps);

        shared_sum[0] = mean;
        shared_sum[1] = inv_std;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = shared_sum[0];
    float inv_std = shared_sum[1];

    for (uint i = tid; i < hidden_size; i += 256) {
        float normalized = (float(input[offset + i]) - mean) * inv_std;
        output[offset + i] = half(normalized * float(gamma[i]) + float(beta[i]));
    }
}

kernel void add(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    c[gid] = a[gid] + b[gid];
}

kernel void mul(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    c[gid] = a[gid] * b[gid];
}

kernel void gelu(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;

    float x = float(input[gid]);
    float x_cubed = x * x * x;
    float tanh_arg = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
    float tanh_val = tanh(tanh_arg);

    output[gid] = half(0.5f * x * (1.0f + tanh_val));
}

kernel void silu(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    output[gid] = half(x / (1.0f + exp(-x)));
}

kernel void rope(
    device half* qk [[buffer(0)]],
    device const float* freqs [[buffer(1)]],
    constant uint* params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint batch_idx = gid.z;
    const uint head_idx = gid.y;
    const uint pos = gid.x;

    const uint num_heads = params[0];
    const uint head_dim = params[1];
    const uint seq_len = params[2];

    if (pos >= seq_len || head_idx >= num_heads) return;

    const uint base_idx = (batch_idx * num_heads * seq_len * head_dim) +
                          (head_idx * seq_len * head_dim) +
                          (pos * head_dim);

    for (uint i = 0; i < head_dim / 2; ++i) {
        float freq = freqs[pos * (head_dim / 2) + i];
        float cos_val = cos(freq);
        float sin_val = sin(freq);

        float q0 = float(qk[base_idx + i * 2]);
        float q1 = float(qk[base_idx + i * 2 + 1]);

        qk[base_idx + i * 2] = half(q0 * cos_val - q1 * sin_val);
        qk[base_idx + i * 2 + 1] = half(q0 * sin_val + q1 * cos_val);
    }
}
