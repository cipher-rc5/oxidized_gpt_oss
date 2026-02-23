// file: src/backend/metal_stub.rs
// description: Compatibility backend used when native Metal support is unavailable.
// author: cipher-rc5

use anyhow::{Result, anyhow};
use half::f16;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug)]
pub enum StorageMode {
    Shared,
    Managed,
    Private,
    Memoryless,
}

pub struct MetalDevice;

pub struct MetalBuffer {
    data: Arc<Mutex<Vec<u8>>>,
}

pub struct MetalCompute {
    pub device: Arc<MetalDevice>,
}

pub struct MetalBackend {
    compute: MetalCompute,
}

impl MetalDevice {
    pub fn new() -> Result<Arc<Self>> {
        Ok(Arc::new(Self))
    }

    pub fn allocate_buffer(
        self: &Arc<Self>,
        size: usize,
        _storage_mode: StorageMode,
    ) -> Result<MetalBuffer> {
        Ok(MetalBuffer {
            data: Arc::new(Mutex::new(vec![0u8; size])),
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

impl MetalBuffer {
    pub fn write_data(&self, data: &[u8]) -> Result<()> {
        let mut guard = self.data.lock().unwrap();
        if data.len() > guard.len() {
            return Err(anyhow!(
                "Data size {} exceeds buffer size {}",
                data.len(),
                guard.len()
            ));
        }
        guard[..data.len()].copy_from_slice(data);
        Ok(())
    }

    pub fn read_data(&self, out: &mut [u8]) -> Result<()> {
        let guard = self.data.lock().unwrap();
        if out.len() > guard.len() {
            return Err(anyhow!(
                "Data size {} exceeds buffer size {}",
                out.len(),
                guard.len()
            ));
        }
        out.copy_from_slice(&guard[..out.len()]);
        Ok(())
    }

    pub fn size(&self) -> usize {
        self.data.lock().unwrap().len()
    }
}

impl Clone for MetalBuffer {
    fn clone(&self) -> Self {
        let data = self.data.lock().unwrap().clone();
        Self {
            data: Arc::new(Mutex::new(data)),
        }
    }
}

impl MetalCompute {
    pub fn new(device: Arc<MetalDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    pub fn matmul_fp16(
        &self,
        _a: &MetalBuffer,
        _b: &MetalBuffer,
        _c: &MetalBuffer,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<()> {
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
        let mut in_bytes = vec![0u8; input.size()];
        let mut g_bytes = vec![0u8; gamma.size()];
        let mut b_bytes = vec![0u8; beta.size()];
        input.read_data(&mut in_bytes)?;
        gamma.read_data(&mut g_bytes)?;
        beta.read_data(&mut b_bytes)?;

        let to_f32 = |bytes: &[u8]| -> Result<Vec<f32>> {
            if bytes.len() % 2 != 0 {
                return Err(anyhow!("Buffer size {} not aligned to f16", bytes.len()));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect())
        };

        let input_vals = to_f32(&in_bytes)?;
        let gamma_vals = to_f32(&g_bytes)?;
        let beta_vals = to_f32(&b_bytes)?;
        let mut out_vals = vec![0.0f32; input_vals.len()];

        for b in 0..batch_size {
            let offset = b * hidden_size;
            let row = &input_vals[offset..offset + hidden_size];
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / hidden_size as f32;
            let inv = 1.0f32 / (mean_sq + eps).sqrt();
            for i in 0..hidden_size {
                out_vals[offset + i] = row[i] * inv * gamma_vals[i] + beta_vals[i];
            }
        }

        let mut out_bytes = Vec::with_capacity(out_vals.len() * 2);
        for v in out_vals {
            out_bytes.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
        }
        output.write_data(&out_bytes)?;
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
