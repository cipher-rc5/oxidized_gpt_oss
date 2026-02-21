# Directory Structure

```
src/
  backend/
    metal/
      kernels/
        kernels.metal
        mod.rs
      metal_impl.rs
      mod.rs
    mod.rs
  benchmark.rs
  config.rs
  convert.rs
  dtype.rs
  inference.rs
  lib.rs
  main.rs
  memory.rs
  model.rs
  moe.rs
  utils.rs
Cargo.toml
```

# Files

## File: src/backend/metal/kernels/kernels.metal

```
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
```

## File: src/backend/metal/kernels/mod.rs

```rust
use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

/// Metal shader source that gets embedded into the binary at compile time.
pub const SOURCE: &str = include_str!("kernels.metal");

/// Enumeration of every compute kernel that exists inside `kernels.metal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetalKernel {
    MatmulMxFp4,
    MatmulFp16,
    MxFp4Unpack,
    MxFp4Pack,
    Softmax,
    LayerNorm,
    Add,
    Mul,
    Gelu,
    Silu,
    Rope,
}

impl MetalKernel {
    pub fn name(&self) -> &'static str {
        match self {
            MetalKernel::MatmulMxFp4 => "matmul_mxfp4",
            MetalKernel::MatmulFp16 => "matmul_fp16",
            MetalKernel::MxFp4Unpack => "mxfp4_unpack",
            MetalKernel::MxFp4Pack => "mxfp4_pack",
            MetalKernel::Softmax => "softmax",
            MetalKernel::LayerNorm => "layernorm",
            MetalKernel::Add => "add",
            MetalKernel::Mul => "mul",
            MetalKernel::Gelu => "gelu",
            MetalKernel::Silu => "silu",
            MetalKernel::Rope => "rope",
        }
    }
}

/// Lazily compiled cache of compute pipeline states for every Metal kernel.
pub struct KernelManager {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: Mutex<HashMap<MetalKernel, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
}

impl KernelManager {
    /// Compile the shader library once and prepare for pipeline creation.
    pub fn new(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> Result<Self> {
        let source = NSString::from_str(SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|err: Retained<NSError>| {
                anyhow!(
                    "Failed to compile Metal kernels: {}",
                    err.localizedDescription().to_string()
                )
            })?;

        Ok(Self {
            device: device.clone(),
            library,
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    /// Retrieve (and cache) the pipeline for the requested kernel.
    pub fn pipeline(
        &self,
        kernel: MetalKernel,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        if let Some(pipeline) = self.pipelines.lock().unwrap().get(&kernel) {
            return Ok(pipeline.clone());
        }

        let compiled = self.compile_pipeline(kernel)?;
        self.pipelines
            .lock()
            .unwrap()
            .insert(kernel, compiled.clone());
        Ok(compiled)
    }

    fn compile_pipeline(
        &self,
        kernel: MetalKernel,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        let function_name = NSString::from_str(kernel.name());
        let function = self
            .library
            .newFunctionWithName(&function_name)
            .ok_or_else(|| anyhow!("Kernel not found in library: {}", kernel.name()))?;

        self.device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err: Retained<NSError>| {
                anyhow!(
                    "Failed to create pipeline for {}: {}",
                    kernel.name(),
                    err.localizedDescription().to_string()
                )
            })
    }
}
```

## File: src/backend/metal/metal_impl.rs

```rust
use anyhow::{Result, anyhow};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize, MTLStorageMode,
};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;
use tracing::info;

use super::kernels::{KernelManager, MetalKernel};

pub struct MetalDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

pub struct MetalBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    size: usize,
    device: Arc<MetalDevice>,
}

pub struct MetalCompute {
    pub device: Arc<MetalDevice>,
    kernels: KernelManager,
}

pub struct MetalBackend {
    compute: MetalCompute,
}

fn resource_options(storage_mode: MTLStorageMode) -> MTLResourceOptions {
    match storage_mode {
        MTLStorageMode::Shared => MTLResourceOptions::StorageModeShared,
        MTLStorageMode::Managed => MTLResourceOptions::StorageModeManaged,
        MTLStorageMode::Private => MTLResourceOptions::StorageModePrivate,
        MTLStorageMode::Memoryless => MTLResourceOptions::StorageModeMemoryless,
        _ => MTLResourceOptions::StorageModeShared,
    }
}

impl MetalDevice {
    pub fn new() -> Result<Arc<Self>> {
        let device = objc2_metal::MTLCreateSystemDefaultDevice()
            .ok_or_else(|| anyhow!("Failed to create Metal device"))?;

        info!("Metal device: {}", device.name());

        let command_queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("Failed to create command queue"))?;

        Ok(Arc::new(Self {
            device,
            command_queue,
        }))
    }

    pub fn allocate_buffer(
        self: &Arc<Self>,
        size: usize,
        storage_mode: MTLStorageMode,
    ) -> Result<MetalBuffer> {
        MetalBuffer::new(Arc::clone(self), size, storage_mode)
    }

    pub fn synchronize(&self) -> Result<()> {
        // All work is submitted synchronously for now, so there's nothing to do.
        Ok(())
    }
}

impl MetalBuffer {
    pub fn new(
        device: Arc<MetalDevice>,
        size: usize,
        storage_mode: MTLStorageMode,
    ) -> Result<Self> {
        let buffer = device
            .device
            .newBufferWithLength_options(size, resource_options(storage_mode))
            .ok_or_else(|| anyhow!("Failed to allocate buffer of size {}", size))?;

        Ok(Self {
            buffer,
            size,
            device,
        })
    }

    pub fn write_data(&self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(anyhow!(
                "Data size {} exceeds buffer size {}",
                data.len(),
                self.size
            ));
        }

        unsafe {
            let contents = self.buffer.contents();
            std::ptr::copy_nonoverlapping(data.as_ptr(), contents.as_ptr().cast(), data.len());
        }

        Ok(())
    }

    pub fn read_data(&self, data: &mut [u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(anyhow!(
                "Data size {} exceeds buffer size {}",
                data.len(),
                self.size
            ));
        }

        unsafe {
            let contents = self.buffer.contents();
            std::ptr::copy_nonoverlapping(contents.as_ptr().cast(), data.as_mut_ptr(), data.len());
        }

        Ok(())
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl MetalCompute {
    pub fn new(device: Arc<MetalDevice>) -> Result<Self> {
        let kernels = KernelManager::new(&device.device)?;
        Ok(Self { device, kernels })
    }

    fn pipeline(
        &self,
        kernel: MetalKernel,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.kernels.pipeline(kernel)
    }

    pub fn matmul_fp16(
        &self,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let pipeline = self.pipeline(MetalKernel::MatmulFp16)?;

        let command_buffer = self
            .device
            .command_queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("Failed to create command buffer"))?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("Failed to create compute encoder"))?;

        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&a.buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&b.buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&c.buffer), 0, 2);
        }

        let params = vec![m as u32, n as u32, k as u32];
        let params_ptr = NonNull::new(params.as_ptr() as *mut c_void)
            .ok_or_else(|| anyhow!("Params pointer was null"))?;
        let params_buffer = unsafe {
            self.device.device.newBufferWithBytes_length_options(
                params_ptr,
                params.len() * std::mem::size_of::<u32>(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| anyhow!("Failed to create params buffer"))?;

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
        }

        let grid_size = MTLSize {
            width: n,
            height: m,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };

        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        Ok(())
    }

    pub fn layernorm(
        &self,
        input: &MetalBuffer,
        output: &MetalBuffer,
        gamma: &MetalBuffer,
        beta: &MetalBuffer,
        batch_size: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<()> {
        let pipeline = self.pipeline(MetalKernel::LayerNorm)?;

        let command_buffer = self
            .device
            .command_queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("Failed to create command buffer"))?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("Failed to create compute encoder"))?;

        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&input.buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&output.buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&gamma.buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&beta.buffer), 0, 3);
        }

        let params = vec![hidden_size as f32, eps];
        let params_ptr = NonNull::new(params.as_ptr() as *mut c_void)
            .ok_or_else(|| anyhow!("Params pointer was null"))?;
        let params_buffer = unsafe {
            self.device.device.newBufferWithBytes_length_options(
                params_ptr,
                params.len() * std::mem::size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| anyhow!("Failed to create params buffer"))?;

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 4);
        }

        let grid_size = MTLSize {
            width: batch_size,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        Ok(())
    }
}

impl MetalBackend {
    pub fn new() -> Result<Self> {
        let device = MetalDevice::new()?;
        let compute = MetalCompute::new(device)?;

        Ok(Self { compute })
    }

    pub fn get_compute(&self) -> &MetalCompute {
        &self.compute
    }
}

impl Clone for MetalBuffer {
    fn clone(&self) -> Self {
        let cloned_buffer = self
            .device
            .allocate_buffer(self.size, MTLStorageMode::Shared)
            .expect("Failed to clone buffer");

        let mut data = vec![0u8; self.size];
        self.read_data(&mut data).expect("Failed to read data");
        cloned_buffer
            .write_data(&data)
            .expect("Failed to write data");

        cloned_buffer
    }
}
```

## File: src/backend/metal/mod.rs

```rust
pub mod kernels;
pub mod metal_impl;

pub use metal_impl::{MetalBackend, MetalBuffer, MetalCompute, MetalDevice};
```

## File: src/backend/mod.rs

```rust
pub mod metal;
```

## File: src/benchmark.rs

```rust
use std::time::{Duration, Instant};
use tracing::info;

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation: String,
    pub duration: Duration,
    pub throughput: Option<f64>,
}

pub struct Benchmark {
    start: Instant,
    operation: String,
}

impl Benchmark {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            operation: operation.into(),
        }
    }

    pub fn finish(self) -> BenchmarkResult {
        let duration = self.start.elapsed();
        info!("{} took {:?}", self.operation, duration);

        BenchmarkResult {
            operation: self.operation,
            duration,
            throughput: None,
        }
    }

    pub fn finish_with_tokens(self, num_tokens: usize) -> BenchmarkResult {
        let duration = self.start.elapsed();
        let tokens_per_sec = num_tokens as f64 / duration.as_secs_f64();

        info!(
            "{} took {:?} ({:.2} tokens/sec)",
            self.operation, duration, tokens_per_sec
        );

        BenchmarkResult {
            operation: self.operation,
            duration,
            throughput: Some(tokens_per_sec),
        }
    }
}

pub struct PerformanceMetrics {
    pub tokens_generated: usize,
    pub total_time: Duration,
    pub time_to_first_token: Duration,
    pub tokens_per_second: f64,
    pub ms_per_token: f64,
}

impl PerformanceMetrics {
    pub fn new(
        tokens_generated: usize,
        total_time: Duration,
        time_to_first_token: Duration,
    ) -> Self {
        let tokens_per_second = tokens_generated as f64 / total_time.as_secs_f64();
        let ms_per_token = total_time.as_millis() as f64 / tokens_generated as f64;

        Self {
            tokens_generated,
            total_time,
            time_to_first_token,
            tokens_per_second,
            ms_per_token,
        }
    }

    pub fn print(&self) {
        info!("Performance Metrics:");
        info!("  Tokens generated: {}", self.tokens_generated);
        info!("  Total time: {:?}", self.total_time);
        info!("  Time to first token: {:?}", self.time_to_first_token);
        info!("  Throughput: {:.2} tokens/sec", self.tokens_per_second);
        info!("  Latency: {:.2} ms/token", self.ms_per_token);
    }
}
```

## File: src/config.rs

```rust
use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs::File;
use std::path::Path;

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
    pub fn load_from_path(path: &Path) -> Result<Self> {
        let config_path = path.join("config.json");
        let file = File::open(&config_path)
            .with_context(|| format!("Failed to open config.json at {:?}", config_path))?;
        let config: ModelConfig = serde_json::from_reader(file)
            .with_context(|| format!("Failed to parse config.json at {:?}", config_path))?;
        Ok(config)
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
```

## File: src/convert.rs

```rust
use crate::dtype::MxBlock;
use anyhow::{Context, Result};
use half::f16;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub struct ModelConverter;

impl ModelConverter {
    pub fn convert_f16_to_mxfp4(input_path: &Path, output_path: &Path) -> Result<()> {
        println!("Converting F16 weights to MXFP4 format...");

        let mut input = File::open(input_path).context("Failed to open input file")?;

        let mut buffer = Vec::new();
        input.read_to_end(&mut buffer)?;

        // Convert buffer to f16 values safely
        if buffer.len() % 2 != 0 {
            return Err(anyhow::anyhow!(
                "Buffer length must be even for f16 conversion"
            ));
        }

        let mut f16_values = Vec::with_capacity(buffer.len() / 2);
        for chunk in buffer.chunks_exact(2) {
            let bytes = [chunk[0], chunk[1]];
            f16_values.push(f16::from_le_bytes(bytes));
        }

        let num_blocks = (f16_values.len() + 31) / 32;
        let mut output_blocks = Vec::with_capacity(num_blocks);

        for chunk in f16_values.chunks(32) {
            let mut values = [0.0f32; 32];
            for (i, &v) in chunk.iter().enumerate() {
                values[i] = v.to_f32();
            }

            if chunk.len() < 32 {
                for i in chunk.len()..32 {
                    values[i] = 0.0;
                }
            }

            let block = MxBlock::from_f32_slice(&values);
            output_blocks.push(block);
        }

        let mut output = File::create(output_path).context("Failed to create output file")?;

        for block in &output_blocks {
            let packed = block.pack();
            output.write_all(&packed)?;
        }

        println!("Conversion complete!");
        println!("Original size: {} bytes", buffer.len());
        println!("Compressed size: {} bytes", output_blocks.len() * 18);
        println!(
            "Compression ratio: {:.2}x",
            buffer.len() as f32 / (output_blocks.len() * 18) as f32
        );

        Ok(())
    }

    pub fn validate_mxfp4_accuracy(original_path: &Path, converted_path: &Path) -> Result<()> {
        println!("Validating MXFP4 conversion accuracy...");

        let mut original = File::open(original_path)?;
        let mut original_buf = Vec::new();
        original.read_to_end(&mut original_buf)?;

        // Convert buffer to f16 values safely
        if original_buf.len() % 2 != 0 {
            return Err(anyhow::anyhow!(
                "Buffer length must be even for f16 conversion"
            ));
        }

        let mut f16_values = Vec::with_capacity(original_buf.len() / 2);
        for chunk in original_buf.chunks_exact(2) {
            let bytes = [chunk[0], chunk[1]];
            f16_values.push(f16::from_le_bytes(bytes));
        }

        let mut converted = File::open(converted_path)?;
        let mut converted_buf = Vec::new();
        converted.read_to_end(&mut converted_buf)?;

        let num_blocks = converted_buf.len() / 18;
        let mut reconstructed = Vec::with_capacity(f16_values.len());

        for i in 0..num_blocks {
            let block_data = &converted_buf[i * 18..(i + 1) * 18];
            let block = MxBlock::unpack(block_data).context("Failed to unpack block")?;
            let values = block.to_f32_vec();
            reconstructed.extend_from_slice(&values);
        }

        reconstructed.truncate(f16_values.len());

        let mut max_error = 0.0f32;
        let mut avg_error = 0.0f32;
        let mut relative_errors = Vec::new();

        for (&orig, &recon) in f16_values.iter().zip(reconstructed.iter()) {
            let orig_f32 = orig.to_f32();
            let abs_error = (orig_f32 - recon).abs();
            let rel_error = if orig_f32.abs() > 1e-6 {
                abs_error / orig_f32.abs()
            } else {
                abs_error
            };

            max_error = max_error.max(abs_error);
            avg_error += abs_error;
            relative_errors.push(rel_error);
        }

        avg_error /= f16_values.len() as f32;

        relative_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_rel_error = relative_errors[relative_errors.len() / 2];
        let p99_rel_error = relative_errors[(relative_errors.len() as f32 * 0.99) as usize];

        println!("\nAccuracy Analysis:");
        println!("  Max absolute error: {:.6}", max_error);
        println!("  Avg absolute error: {:.6}", avg_error);
        println!("  Median relative error: {:.4}%", median_rel_error * 100.0);
        println!(
            "  99th percentile relative error: {:.4}%",
            p99_rel_error * 100.0
        );

        if p99_rel_error > 0.20 {
            println!("\nWarning: High error rate detected. Model accuracy may be impacted.");
        } else {
            println!("\nValidation passed! Quantization quality is good.");
        }

        Ok(())
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use tempfile::NamedTempFile;

//     #[test]
//     fn test_conversion_roundtrip() {
//         let original_values: Vec<f16> = (0..1024)
//             .map(|i| f16::from_f32((i as f32) * 0.01))
//             .collect();

//         let original_bytes: &[u8] = bytemuck::cast_slice(&original_values);

//         let mut original_file = NamedTempFile::new().unwrap();
//         original_file.write_all(original_bytes).unwrap();

//         let converted_file = NamedTempFile::new().unwrap();

//         ModelConverter::convert_f16_to_mxfp4(original_file.path(), converted_file.path()).unwrap();

//         ModelConverter::validate_mxfp4_accuracy(original_file.path(), converted_file.path())
//             .unwrap();
//     }
// }
```

## File: src/dtype.rs

```rust
use bytemuck::{Pod, Zeroable};
use half::f16;
use std::fmt;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct MxFp4(u8);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct F6E2M3(u8);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct F6E3M2(u8);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct F8E8M0(u8);

impl MxFp4 {
    // const SCALE_BITS: u8 = 8;

    pub fn new(value: u8) -> Self {
        Self(value & 0x0F)
    }

    pub fn from_f32(value: f32) -> Self {
        let abs_val = value.abs();
        let sign = if value < 0.0 { 0x08 } else { 0x00 };

        if abs_val == 0.0 {
            return Self(sign);
        }

        let exp = abs_val.log2().floor() as i32;
        let exp_clamped = exp.clamp(-2, 1);
        let exp_bits = ((exp_clamped + 2) as u8) & 0x03;

        let mantissa = (abs_val / 2.0_f32.powi(exp_clamped)) - 1.0;
        let mantissa_bit = if mantissa >= 0.5 { 0x01 } else { 0x00 };

        Self(sign | (exp_bits << 1) | mantissa_bit)
    }

    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = if (bits & 0x08) != 0 { -1.0 } else { 1.0 };
        let exp_bits = (bits >> 1) & 0x03;
        let mantissa_bit = bits & 0x01;

        if exp_bits == 0 && mantissa_bit == 0 {
            return 0.0;
        }

        let exp = (exp_bits as i32) - 2;
        let mantissa = 1.0 + (mantissa_bit as f32) * 0.5;

        sign * mantissa * 2.0_f32.powi(exp)
    }

    pub fn to_f16(self) -> f16 {
        f16::from_f32(self.to_f32())
    }

    pub fn pack_pair(a: MxFp4, b: MxFp4) -> u8 {
        (a.0 & 0x0F) | ((b.0 & 0x0F) << 4)
    }

    pub fn unpack_pair(packed: u8) -> (MxFp4, MxFp4) {
        (MxFp4(packed & 0x0F), MxFp4((packed >> 4) & 0x0F))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MxBlock {
    pub scale: f16,
    pub values: [MxFp4; 32],
}

impl MxBlock {
    pub fn new(scale: f16) -> Self {
        Self {
            scale,
            values: [MxFp4(0); 32],
        }
    }

    pub fn from_f32_slice(values: &[f32]) -> Self {
        assert!(values.len() == 32, "MxBlock requires exactly 32 values");

        let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        let scale = if max_abs > 0.0 {
            f16::from_f32(max_abs / 7.5)
        } else {
            f16::from_f32(1.0)
        };

        let scale_f32 = scale.to_f32();
        let mut block = Self::new(scale);

        for (i, &value) in values.iter().enumerate() {
            let scaled = value / scale_f32;
            block.values[i] = MxFp4::from_f32(scaled);
        }

        block
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        let scale = self.scale.to_f32();
        self.values.iter().map(|v| v.to_f32() * scale).collect()
    }

    pub fn pack(&self) -> Vec<u8> {
        let mut packed = Vec::with_capacity(18);

        packed.extend_from_slice(&self.scale.to_bits().to_le_bytes());

        for chunk in self.values.chunks(2) {
            let byte = if chunk.len() == 2 {
                MxFp4::pack_pair(chunk[0], chunk[1])
            } else {
                chunk[0].0
            };
            packed.push(byte);
        }

        packed
    }

    pub fn unpack(data: &[u8]) -> Option<Self> {
        if data.len() < 18 {
            return None;
        }

        let scale_bits = u16::from_le_bytes([data[0], data[1]]);
        let scale = f16::from_bits(scale_bits);

        let mut values = [MxFp4(0); 32];
        for (i, &byte) in data[2..18].iter().enumerate() {
            let (a, b) = MxFp4::unpack_pair(byte);
            values[i * 2] = a;
            if i * 2 + 1 < 32 {
                values[i * 2 + 1] = b;
            }
        }

        Some(Self { scale, values })
    }
}

impl F6E2M3 {
    pub fn from_f32(value: f32) -> Self {
        let abs_val = value.abs();
        let sign = if value < 0.0 { 0x20 } else { 0x00 };

        if abs_val == 0.0 {
            return Self(sign);
        }

        let exp = abs_val.log2().floor() as i32;
        let exp_clamped = exp.clamp(-1, 2);
        let exp_bits = ((exp_clamped + 1) as u8) & 0x03;

        let mantissa = (abs_val / 2.0_f32.powi(exp_clamped)) - 1.0;
        let mantissa_scaled = (mantissa * 7.0).round() as u8;
        let mantissa_bits = mantissa_scaled.min(7);

        Self(sign | (exp_bits << 3) | mantissa_bits)
    }

    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = if (bits & 0x20) != 0 { -1.0 } else { 1.0 };
        let exp_bits = (bits >> 3) & 0x03;
        let mantissa_bits = bits & 0x07;

        if exp_bits == 0 && mantissa_bits == 0 {
            return 0.0;
        }

        let exp = (exp_bits as i32) - 1;
        let mantissa = 1.0 + (mantissa_bits as f32) / 7.0;

        sign * mantissa * 2.0_f32.powi(exp)
    }
}

impl F6E3M2 {
    pub fn from_f32(value: f32) -> Self {
        let abs_val = value.abs();
        let sign = if value < 0.0 { 0x20 } else { 0x00 };

        if abs_val == 0.0 {
            return Self(sign);
        }

        let exp = abs_val.log2().floor() as i32;
        let exp_clamped = exp.clamp(-3, 4);
        let exp_bits = ((exp_clamped + 3) as u8) & 0x07;

        let mantissa = (abs_val / 2.0_f32.powi(exp_clamped)) - 1.0;
        let mantissa_scaled = (mantissa * 3.0).round() as u8;
        let mantissa_bits = mantissa_scaled.min(3);

        Self(sign | (exp_bits << 2) | mantissa_bits)
    }

    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = if (bits & 0x20) != 0 { -1.0 } else { 1.0 };
        let exp_bits = (bits >> 2) & 0x07;
        let mantissa_bits = bits & 0x03;

        if exp_bits == 0 && mantissa_bits == 0 {
            return 0.0;
        }

        let exp = (exp_bits as i32) - 3;
        let mantissa = 1.0 + (mantissa_bits as f32) / 3.0;

        sign * mantissa * 2.0_f32.powi(exp)
    }
}

impl F8E8M0 {
    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 {
            return Self(0);
        }

        let exp = value.log2().floor() as i32;
        let exp_clamped = exp.clamp(-127, 127);
        let exp_biased = (exp_clamped + 127) as u8;

        Self(exp_biased)
    }

    pub fn to_f32(self) -> f32 {
        if self.0 == 0 {
            return 0.0;
        }

        let exp = (self.0 as i32) - 127;
        2.0_f32.powi(exp)
    }
}

impl fmt::Display for MxFp4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl fmt::Display for F6E2M3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl fmt::Display for F6E3M2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl fmt::Display for F8E8M0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mxfp4_conversion() {
        let values = vec![0.0, 1.0, -1.0, 2.5, -3.5];
        for &val in &values {
            let fp4 = MxFp4::from_f32(val);
            let reconstructed = fp4.to_f32();
            assert!(
                (reconstructed - val).abs() < 1.0,
                "Value {} reconstructed as {}",
                val,
                reconstructed
            );
        }
    }

    #[test]
    fn test_mx_block() {
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let block = MxBlock::from_f32_slice(&values);
        let reconstructed = block.to_f32_vec();

        for (original, &recon) in values.iter().zip(reconstructed.iter()) {
            let error = (recon - original).abs() / original.max(1.0);
            assert!(error < 0.2, "Reconstruction error too high: {}", error);
        }
    }
}
```

## File: src/inference.rs

```rust
use anyhow::{Result, anyhow};
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
```

## File: src/lib.rs

```rust
pub mod backend;
pub mod benchmark;
pub mod config;
pub mod convert;
pub mod dtype;
pub mod inference;
pub mod memory;
pub mod model;
pub mod moe;
pub mod utils;

pub use backend::metal::{MetalBuffer, MetalCompute, MetalDevice};
pub use config::{MoEConfig, ModelConfig, RoutingStrategy};
pub use dtype::{F6E2M3, F6E3M2, F8E8M0, MxBlock, MxFp4};
pub use inference::{GenerationConfig, InferenceEngine};
pub use model::GPTModel;

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
```

## File: src/main.rs

```rust
use anyhow::Result;
use clap::Parser;
use oxidized_gpt_oss::{GenerationConfig, InferenceEngine, MetalDevice, ModelConfig};
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tracing::{Level, info};
use tracing_subscriber;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "Path to model directory")]
    model_path: PathBuf,

    #[arg(short, long, help = "Path to tokenizer file")]
    tokenizer_path: Option<PathBuf>,

    #[arg(
        short,
        long,
        default_value = "What is the meaning of life?",
        help = "Input prompt"
    )]
    prompt: String,

    #[arg(long, default_value = "512", help = "Maximum tokens to generate")]
    max_tokens: usize,

    #[arg(long, default_value = "0.7", help = "Sampling temperature")]
    temperature: f32,

    #[arg(long, default_value = "0.9", help = "Top-p sampling")]
    top_p: f32,

    #[arg(long, help = "Top-k sampling")]
    top_k: Option<usize>,

    #[arg(short, long, help = "Enable verbose logging")]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let log_level = if args.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };

    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .init();

    info!("GPT-OSS-20B MXFP4 Inference Engine");
    info!("Model path: {:?}", args.model_path);

    info!("Initializing Metal device...");
    let device = MetalDevice::new()?;

    info!("Loading model configuration...");
    let config = ModelConfig::load_from_path(&args.model_path)?;

    info!("Model configuration:");
    info!("  Layers: {}", config.num_layers);
    info!("  Hidden size: {}", config.hidden_size);
    info!("  Attention heads: {}", config.num_attention_heads);
    info!("  Vocab size: {}", config.vocab_size);
    info!("  Max sequence length: {}", config.max_sequence_length);
    if config.supports_moe() {
        info!("  MoE enabled:");
        if let Some(moe_config) = config.get_moe_config() {
            info!("    Experts: {}", moe_config.num_experts);
            info!("    Experts per token: {}", moe_config.experts_per_token);
        }
    }

    info!("Loading tokenizer...");
    let tokenizer_path = args
        .tokenizer_path
        .unwrap_or_else(|| args.model_path.join("tokenizer.json"));
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    info!("Initializing inference engine...");
    let engine = InferenceEngine::new(&args.model_path, &config, device)?;

    let gen_config = GenerationConfig {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: 1.0,
    };

    info!("Prompt: {}", args.prompt);
    info!("Generating...");

    let start = std::time::Instant::now();
    let output = engine.generate(&args.prompt, &tokenizer, &gen_config)?;
    let elapsed = start.elapsed();

    println!("\nGenerated text:");
    println!("{}", output);
    println!("\nGeneration time: {:.2}s", elapsed.as_secs_f32());

    let tokens_generated = output.split_whitespace().count();
    if elapsed.as_secs_f32() > 0.0 {
        println!(
            "Tokens/second: {:.2}",
            tokens_generated as f32 / elapsed.as_secs_f32()
        );
    }

    Ok(())
}

// use anyhow::Result;
// use candle_core::Device;
// use clap::Parser;
// use std::path::PathBuf;
// use std::time::Instant;
// use tokenizers::Tokenizer;
// use tracing::{error, info, warn};

// mod config;
// mod inference;
// mod memory;
// mod model;

// use crate::config::{MoEConfig, ModelConfig, RoutingStrategy};
// use crate::inference::InferenceEngine;

// #[derive(Parser)]
// #[command(name = "gpt-oss-runner")]
// #[command(about = "High-performance GPT-OSS inference on Mac M3 Ultra with MoE support")]
// struct Args {
//     /// Path to model directory
//     #[arg(short, long)]
//     model_path: PathBuf,

//     /// Path to tokenizer
//     #[arg(short, long)]
//     tokenizer_path: Option<PathBuf>,

//     /// Interactive chat mode
//     #[arg(short, long)]
//     interactive: bool,

//     /// Input prompt (non-interactive mode)
//     #[arg(short, long)]
//     prompt: Option<String>,

//     /// Maximum tokens to generate
//     #[arg(long, default_value = "512")]
//     max_tokens: usize,

//     /// Temperature for sampling
//     #[arg(long, default_value = "0.7")]
//     temperature: f64,

//     /// Top-p for nucleus sampling
//     #[arg(long, default_value = "0.9")]
//     top_p: f64,

//     /// Batch size for processing
//     #[arg(long, default_value = "1")]
//     batch_size: usize,

//     /// Use Metal GPU acceleration
//     #[arg(long, default_value = "true")]
//     use_metal: bool,

//     /// Memory mapping for large models
//     #[arg(long, default_value = "true")]
//     use_mmap: bool,

//     /// Model precision (f16, bf16, f32)
//     #[arg(long, default_value = "f16")]
//     precision: String,

//     // MoE-specific arguments
//     /// Enable Mixture of Experts
//     #[arg(long)]
//     enable_moe: bool,

//     /// Number of experts (for MoE models)
//     #[arg(long, default_value = "8")]
//     num_experts: usize,

//     /// Experts per token (for MoE models)
//     #[arg(long, default_value = "2")]
//     experts_per_token: usize,

//     /// MoE routing strategy (topk, switch, expert1)
//     #[arg(long, default_value = "topk")]
//     routing_strategy: String,

//     /// Expert capacity factor (for Switch routing)
//     #[arg(long, default_value = "1.0")]
//     expert_capacity_factor: f32,

//     /// Show detailed MoE metrics
//     #[arg(long)]
//     show_moe_metrics: bool,

//     /// Run MoE benchmark mode
//     #[arg(long)]
//     benchmark_moe: bool,

//     /// Number of benchmark runs
//     #[arg(long, default_value = "3")]
//     benchmark_runs: usize,

//     /// Print memory statistics
//     #[arg(long)]
//     memory_stats: bool,
// }

// async fn main_async() -> Result<()> {
//     // Initialize logging
//     tracing_subscriber::fmt()
//         .with_env_filter("info")
//         .with_target(false)
//         .init();

//     let args = Args::parse();

//     // Check system capabilities
//     check_system_requirements(&args)?;

//     // Initialize device (Metal GPU or CPU fallback)
//     let device = initialize_device(args.use_metal)?;
//     info!("Using device: {:?}", device);

//     let config = ModelConfig::load_from_path(&args.model_path)?;
//     info!("Model config loaded");
//     info!("Total parameters: {}", format_number(config.total_params()));

//     // Print MoE information if enabled
//     if args.enable_moe || config.supports_moe() {
//         print_moe_info(&config, &args);
//     }

//     // Load tokenizer
//     let tokenizer_path = args
//         .tokenizer_path
//         .clone()
//         .unwrap_or_else(|| args.model_path.join("tokenizer.json"));
//     let tokenizer = Tokenizer::from_file(&tokenizer_path)
//         .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

//     // Create MoE configuration if enabled
//     let custom_moe_config = if args.enable_moe {
//         Some(create_moe_config_from_args(&args)?)
//     } else {
//         None
//     };

//     // Initialize inference engine
//     let mut engine = InferenceEngine::new_with_moe(
//         &args.model_path,
//         &config,
//         device,
//         args.use_mmap,
//         &args.precision,
//         args.enable_moe || config.supports_moe(),
//         custom_moe_config,
//     )
//     .await?;

//     // Print memory statistics if requested
//     if args.memory_stats {
//         engine.print_memory_summary();
//     }

//     // Choose execution mode
//     if args.benchmark_moe {
//         run_moe_benchmark(&mut engine, &tokenizer, &args).await?;
//     } else if args.interactive {
//         run_interactive_mode(&mut engine, &tokenizer, &args).await?;
//     } else if let Some(prompt) = args.prompt.as_ref() {
//         run_single_inference(&mut engine, &tokenizer, prompt, &args).await?;
//     } else {
//         eprintln!("Please provide either --interactive, --prompt, or --benchmark-moe");
//         std::process::exit(1);
//     }

//     Ok(())
// }

// fn main() -> Result<()> {
//     use tokio::runtime::Builder;

//     Builder::new_multi_thread()
//         .enable_all()
//         .build()
//         .unwrap()
//         .block_on(main_async())
// }

// fn create_moe_config_from_args(args: &Args) -> Result<MoEConfig> {
//     let routing_strategy = args.routing_strategy.parse::<RoutingStrategy>()?;

//     Ok(MoEConfig {
//         num_experts: args.num_experts,
//         experts_per_token: args.experts_per_token,
//         expert_capacity_factor: args.expert_capacity_factor,
//         use_swiglu: true, // Default to SwiGLU for MoE
//         routing_strategy,
//     })
// }

// fn print_moe_info(config: &ModelConfig, args: &Args) {
//     println!("\n=== MoE Configuration ===");

//     if config.supports_moe() {
//         println!("Model has native MoE support");
//         if let Some(moe_config) = config.get_moe_config() {
//             println!("  Native experts: {}", moe_config.num_experts);
//             println!("  Experts per token: {}", moe_config.experts_per_token);
//             println!("  MoE layers: {}", config.get_num_moe_layers());
//             println!("  Total experts: {}", config.get_total_experts());
//         }
//     }

//     if args.enable_moe {
//         println!("Runtime MoE override enabled");
//         println!("  Override experts: {}", args.num_experts);
//         println!("  Override experts per token: {}", args.experts_per_token);
//         println!("  Routing strategy: {}", args.routing_strategy);
//     }

//     println!("=========================\n");
// }

// fn check_system_requirements(args: &Args) -> Result<()> {
//     use sysinfo::System;

//     let mut sys = System::new_all();
//     sys.refresh_all();

//     let total_memory = sys.total_memory() / 1024 / 1024 / 1024; // GB
//     info!("Total system memory: {} GB", total_memory);

//     // Enhanced memory estimation for MoE
//     let base_memory_need = if args.precision == "f32" { 40 } else { 20 };
//     let moe_memory_multiplier = if args.enable_moe {
//         1.0 + (args.num_experts as f32 * 0.1) // Rough estimate
//     } else {
//         1.0
//     };

//     let estimated_memory_need = (base_memory_need as f32 * moe_memory_multiplier) as u64;

//     if total_memory < estimated_memory_need {
//         warn!(
//             "System has {}GB RAM, but model may need {}GB (including MoE overhead). \
//             Consider using model sharding, quantization, or reducing num_experts.",
//             total_memory, estimated_memory_need
//         );
//     }

//     // Check for Metal availability on macOS
//     if args.use_metal && !cfg!(target_os = "macos") {
//         warn!("Metal GPU acceleration is only available on macOS. Falling back to CPU.");
//     }

//     Ok(())
// }

// fn initialize_device(use_metal: bool) -> Result<Device> {
//     if use_metal && cfg!(target_os = "macos") {
//         match Device::new_metal(0) {
//             Ok(device) => {
//                 info!("Successfully initialized Metal device");
//                 Ok(device)
//             }
//             Err(e) => {
//                 warn!(
//                     "Failed to initialize Metal device: {}. Falling back to CPU.",
//                     e
//                 );
//                 Ok(Device::Cpu)
//             }
//         }
//     } else {
//         info!("Using CPU device");
//         Ok(Device::Cpu)
//     }
// }

// async fn run_single_inference(
//     engine: &mut InferenceEngine,
//     tokenizer: &Tokenizer,
//     prompt: &str,
//     args: &Args,
// ) -> Result<()> {
//     let start_time = Instant::now();

//     info!("Generating response for prompt: {}", prompt);

//     let (response, moe_metrics) = if engine.is_moe_enabled() && args.show_moe_metrics {
//         engine
//             .generate_with_metrics(
//                 prompt,
//                 tokenizer,
//                 args.max_tokens,
//                 args.temperature,
//                 args.top_p,
//             )
//             .await?
//     } else {
//         let response = engine
//             .generate(
//                 prompt,
//                 tokenizer,
//                 args.max_tokens,
//                 args.temperature,
//                 args.top_p,
//             )
//             .await?;
//         (response, None)
//     };

//     let elapsed = start_time.elapsed();
//     let tokens_per_second = if elapsed.as_secs_f64() > 0.0 {
//         args.max_tokens as f64 / elapsed.as_secs_f64()
//     } else {
//         0.0
//     };

//     println!("\n{}", response);
//     println!("\n--- Performance Stats ---");
//     println!("Time: {:.2}s", elapsed.as_secs_f64());
//     println!("Tokens/sec: {:.2}", tokens_per_second);

//     if let Ok((peak_mem, total_mem)) = engine.get_memory_usage() {
//         println!(
//             "Memory usage: {:.2} MB (peak) / {:.2} MB (total)",
//             peak_mem as f64 / 1e6,
//             total_mem as f64 / 1e6
//         );
//     }

//     // Print MoE metrics if available
//     if let Some(metrics) = moe_metrics {
//         metrics.print_stats();
//     }

//     if args.memory_stats {
//         engine.print_memory_summary();
//     }

//     Ok(())
// }

// async fn run_interactive_mode(
//     engine: &mut InferenceEngine,
//     tokenizer: &Tokenizer,
//     args: &Args,
// ) -> Result<()> {
//     use std::io::{self, Write};

//     println!("GPT-OSS Interactive Mode");
//     if engine.is_moe_enabled() {
//         println!("MoE is enabled - type 'moe-stats' to see expert usage");
//     }
//     println!("Commands: 'quit/exit' to quit, 'memory' for memory stats, 'clear' to reset");
//     println!();

//     loop {
//         print!("> ");
//         io::stdout().flush()?;

//         let mut input = String::new();
//         io::stdin().read_line(&mut input)?;
//         let input = input.trim();

//         match input {
//             "quit" | "exit" => break,
//             "memory" => {
//                 engine.print_memory_summary();
//                 continue;
//             }
//             "moe-stats" => {
//                 if let Some(metrics) = engine.get_moe_metrics() {
//                     metrics.print_stats();
//                 } else {
//                     println!("MoE is not enabled or no metrics available.");
//                 }
//                 continue;
//             }
//             "clear" => {
//                 if engine.is_moe_enabled() {
//                     engine.reset_moe_metrics();
//                     println!("MoE metrics reset.");
//                 }
//                 continue;
//             }
//             "" => continue,
//             _ => {}
//         }

//         let start_time = Instant::now();

//         match engine
//             .generate(
//                 input,
//                 tokenizer,
//                 args.max_tokens,
//                 args.temperature,
//                 args.top_p,
//             )
//             .await
//         {
//             Ok(response) => {
//                 let elapsed = start_time.elapsed();
//                 println!("{}", response);
//                 println!("({:.2}s)\n", elapsed.as_secs_f64());
//             }
//             Err(e) => {
//                 error!("Generation failed: {}", e);
//             }
//         }
//     }

//     println!("Goodbye!");
//     Ok(())
// }

// async fn run_moe_benchmark(
//     engine: &mut InferenceEngine,
//     tokenizer: &Tokenizer,
//     args: &Args,
// ) -> Result<()> {
//     if !engine.is_moe_enabled() {
//         return Err(anyhow::anyhow!("MoE benchmark requires --enable-moe"));
//     }

//     println!("Starting MoE Benchmark...");

//     let test_prompts = vec![
//         "Explain the theory of relativity in simple terms.",
//         "Write a short story about a robot learning to paint.",
//         "Describe the process of photosynthesis.",
//         "What are the benefits and drawbacks of renewable energy?",
//         "How does machine learning work?",
//     ];

//     let prompts_to_use = if args.benchmark_runs < test_prompts.len() {
//         &test_prompts[..args.benchmark_runs]
//     } else {
//         &test_prompts
//     };

//     let results = engine
//         .benchmark_moe_performance(tokenizer, prompts_to_use, args.max_tokens)
//         .await?;

//     results.print_summary();

//     // Print detailed results if requested
//     if args.show_moe_metrics {
//         println!("\n=== Detailed Run Results ===");
//         for (i, run) in results.runs.iter().enumerate() {
//             println!(
//                 "\nRun {}: {}",
//                 i + 1,
//                 run.prompt.chars().take(50).collect::<String>()
//             );
//             println!("Time: {:.2}s", run.total_time.as_secs_f64());
//             println!("Metrics: {}", run.metrics.get_summary());
//         }
//     }

//     Ok(())
// }

// fn format_number(n: u64) -> String {
//     if n >= 1_000_000_000 {
//         format!("{:.1}B", n as f64 / 1_000_000_000.0)
//     } else if n >= 1_000_000 {
//         format!("{:.1}M", n as f64 / 1_000_000.0)
//     } else if n >= 1_000 {
//         format!("{:.1}K", n as f64 / 1_000.0)
//     } else {
//         n.to_string()
//     }
// }
```

## File: src/memory.rs

```rust
//! Memory tracking utilities tailored for the Metal backend.
//!
//! The original Candle-based implementation tracked tensors directly; here we
//! keep lightweight accounting so we can surface peak usage and detect MoE
//! hotspots without touching the performance-critical kernels.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{debug, warn};

use crate::backend::metal::MetalDevice;

#[derive(Debug, Default, Clone)]
struct Usage {
    current: usize,
    peak: usize,
}

impl Usage {
    fn add(&mut self, bytes: usize) {
        self.current += bytes;
        if self.current > self.peak {
            self.peak = self.current;
        }
    }

    fn sub(&mut self, bytes: usize) {
        self.current = self.current.saturating_sub(bytes);
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub gpu_current: usize,
    pub gpu_peak: usize,
    pub host_current: usize,
    pub host_peak: usize,
    pub expert_usage: HashMap<usize, usize>,
    pub pooled_tensors: usize,
}

struct PooledTensor {
    len: usize,
    data: Vec<f32>,
}

pub struct MemoryManager {
    use_mmap: bool,
    gpu_usage: Arc<Mutex<Usage>>,
    host_usage: Arc<Mutex<Usage>>,
    expert_usage: Arc<Mutex<HashMap<usize, usize>>>,
    tensor_pool: Arc<Mutex<Vec<PooledTensor>>>,
}

impl MemoryManager {
    pub fn new(use_mmap: bool, device: &Arc<MetalDevice>) -> Result<Self> {
        let _ = device;
        Ok(Self {
            use_mmap,
            gpu_usage: Arc::new(Mutex::new(Usage::default())),
            host_usage: Arc::new(Mutex::new(Usage::default())),
            expert_usage: Arc::new(Mutex::new(HashMap::new())),
            tensor_pool: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn use_mmap(&self) -> bool {
        self.use_mmap
    }

    pub fn record_gpu_allocation(&self, bytes: usize) {
        self.gpu_usage.lock().unwrap().add(bytes);
    }

    pub fn record_gpu_deallocation(&self, bytes: usize) {
        self.gpu_usage.lock().unwrap().sub(bytes);
    }

    pub fn record_host_allocation(&self, bytes: usize) {
        self.host_usage.lock().unwrap().add(bytes);
    }

    pub fn record_host_deallocation(&self, bytes: usize) {
        self.host_usage.lock().unwrap().sub(bytes);
    }

    pub fn track_expert_memory(&self, expert_id: usize, bytes: usize) {
        let mut expert_usage = self.expert_usage.lock().unwrap();
        let usage = expert_usage.entry(expert_id).or_insert(0);
        *usage += bytes;
        self.host_usage.lock().unwrap().add(bytes);
    }

    pub fn release_expert_memory(&self, expert_id: usize, bytes: usize) {
        let mut expert_usage = self.expert_usage.lock().unwrap();
        if let Some(usage) = expert_usage.get_mut(&expert_id) {
            *usage = usage.saturating_sub(bytes);
        }
        self.host_usage.lock().unwrap().sub(bytes);
    }

    pub fn add_to_pool(&self, tensor: Vec<f32>) {
        let mut pool = self.tensor_pool.lock().unwrap();
        const MAX_POOL_ITEMS: usize = 32;
        if pool.len() >= MAX_POOL_ITEMS {
            pool.remove(0);
        }
        pool.push(PooledTensor {
            len: tensor.len(),
            data: tensor,
        });
    }

    pub fn get_from_pool(&self, len: usize) -> Option<Vec<f32>> {
        let mut pool = self.tensor_pool.lock().unwrap();
        if let Some(idx) = pool.iter().position(|t| t.len == len) {
            let tensor = pool.remove(idx);
            return Some(tensor.data);
        }
        None
    }

    pub fn cleanup_intermediate_tensors(&self) -> Result<()> {
        let mut pool = self.tensor_pool.lock().unwrap();
        const MAX_POOL_BYTES: usize = 128 * 1024 * 1024; // 128MB
        let mut accumulated: usize = 0;
        pool.retain(|tensor| {
            accumulated += tensor.len * std::mem::size_of::<f32>();
            accumulated <= MAX_POOL_BYTES
        });

        let host_current = self.host_usage.lock().unwrap().current;
        const HOST_THRESHOLD: usize = 8 * 1024 * 1024 * 1024; // 8GB
        if host_current > HOST_THRESHOLD {
            warn!(
                "Host allocations above {} GB (currently {:.2} GB)",
                HOST_THRESHOLD as f64 / 1e9,
                host_current as f64 / 1e9
            );
        }

        Ok(())
    }

    pub fn force_cleanup(&self) -> Result<()> {
        self.tensor_pool.lock().unwrap().clear();
        self.expert_usage.lock().unwrap().clear();
        self.host_usage.lock().unwrap().current = 0;
        self.gpu_usage.lock().unwrap().current = 0;
        debug!("Forced memory cleanup on device");
        Ok(())
    }

    pub fn stats(&self) -> MemoryStats {
        let gpu_usage = self.gpu_usage.lock().unwrap().clone();
        let host_usage = self.host_usage.lock().unwrap().clone();
        let expert_usage = self.expert_usage.lock().unwrap().clone();
        let pooled_tensors = self.tensor_pool.lock().unwrap().len();

        MemoryStats {
            gpu_current: gpu_usage.current,
            gpu_peak: gpu_usage.peak,
            host_current: host_usage.current,
            host_peak: host_usage.peak,
            expert_usage,
            pooled_tensors,
        }
    }

    pub fn print_memory_summary(&self) {
        let stats = self.stats();
        println!("Memory usage summary:");
        println!("  GPU current: {:.2} MB", stats.gpu_current as f64 / 1e6);
        println!("  GPU peak: {:.2} MB", stats.gpu_peak as f64 / 1e6);
        println!("  Host current: {:.2} MB", stats.host_current as f64 / 1e6);
        println!("  Host peak: {:.2} MB", stats.host_peak as f64 / 1e6);
        println!("  Tensor pool size: {}", stats.pooled_tensors);

        if !stats.expert_usage.is_empty() {
            println!("  Expert allocations:");
            for (expert, bytes) in stats.expert_usage.iter() {
                println!("    Expert {}: {:.2} MB", expert, *bytes as f64 / 1e6);
            }
        }
    }
}
```

## File: src/model.rs

```rust
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use tracing::{debug, info};

use crate::backend::metal::{MetalBuffer, MetalCompute, MetalDevice};
use crate::config::ModelConfig;
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

impl GPTModel {
    pub fn load_from_safetensors(
        path: &Path,
        config: &ModelConfig,
        device: Arc<MetalDevice>,
    ) -> Result<Self> {
        info!("Loading model from {:?}", path);

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

        let head_dim = config.hidden_size / config.num_attention_heads;
        let configured_intermediate = config.intermediate_size.unwrap_or(config.hidden_size * 4);

        let q_biases = Self::load_tensor_optional(
            path,
            device,
            &format!("{}.self_attn.q_proj.biases", prefix),
        )?;
        let k_biases = Self::load_tensor_optional(
            path,
            device,
            &format!("{}.self_attn.k_proj.biases", prefix),
        )?;
        let v_biases = Self::load_tensor_optional(
            path,
            device,
            &format!("{}.self_attn.v_proj.biases", prefix),
        )?;
        let o_biases = Self::load_tensor_optional(
            path,
            device,
            &format!("{}.self_attn.o_proj.biases", prefix),
        )?;

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
            num_heads: config.num_attention_heads,
            head_dim,
            hidden_size: config.hidden_size,
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
        let buffer = device.allocate_buffer(data.len(), objc2_metal::MTLStorageMode::Shared)?;
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

        // Grouped query attention computation
        let num_kv_groups = q_num_heads / kv_num_heads;

        let mut context = vec![0.0f32; seq_len * q_output_dim];
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();

        for head in 0..q_num_heads {
            let kv_head = head / num_kv_groups;
            for target in 0..seq_len {
                let q_slice = head_slice(&q, target, head, q_num_heads, self.head_dim);
                let mut scores = Vec::with_capacity(target + 1);
                for source in 0..=target {
                    let k_slice = head_slice(&k, source, kv_head, kv_num_heads, self.head_dim);
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

                for (source, &weight) in (0..=target).zip(scores.iter()) {
                    let v_slice = head_slice(&v, source, kv_head, kv_num_heads, self.head_dim);
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
            .allocate_buffer(hidden_states.size(), objc2_metal::MTLStorageMode::Shared)?;

        compute.layernorm(
            hidden_states,
            &output,
            &self.gamma,
            &self.beta,
            1,
            hidden_size,
            self.eps,
        )?;

        Ok(output)
    }
}

use crate::utils::*;
```

## File: src/moe.rs

```rust
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
```

## File: src/utils.rs

```rust
// file: src/utils.rs
// description: Utility functions for tensor operations with bounds checking and dimension validation
// reference: Panic at src/utils.rs:41 - matmul index out of bounds, original bias error at src/utils.rs:85

use crate::backend::metal::{MetalBuffer, MetalDevice};
use anyhow::Result;
use half::f16;
use objc2_metal::MTLStorageMode;

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
    let buffer = device.allocate_buffer(bytes.len(), MTLStorageMode::Shared)?;
    buffer.write_data(&bytes)?;
    Ok(buffer)
}

pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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
                let b_idx = inner * n + col;
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
```

## File: Cargo.toml

```toml
[package]
name = "oxidized_gpt_oss"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = "1.0.99"
bytemuck = { version = "1.24.0", features = ["derive"] }
clap = { version = "4.5.47", features = ["derive"] }
half = "2.7.1"
memmap2 = "0.9.9"
objc2 = "0.6.3"
objc2-foundation = "0.3.2"
objc2-metal = "0.3.2"
rand = "0.9.2"
rayon = "1.11.0"
safetensors = "0.6.2"
serde = { version = "1.0.223", features = ["derive"] }
serde_json = "1.0.145"
tokenizers = "0.22.0"
tokio = "1.47.1"
toml = "0.9.5"
tracing = "0.1.41"
tracing-subscriber = { version = "0.3.20", features = ["env-filter"] }

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"

[profile.dev]
opt-level = 1
```
