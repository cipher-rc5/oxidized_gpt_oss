// file: src/backend/metal/kernels/mod.rs
// description: Embedded Metal shader registry and compute pipeline cache.
// author: cipher-rc5

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
