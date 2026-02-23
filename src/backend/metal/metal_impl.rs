// file: src/backend/metal/metal_impl.rs
// description: Native Metal backend with device management, buffers, and kernel dispatch wrappers.
// author: cipher-rc5

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
