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
