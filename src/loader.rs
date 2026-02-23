// file: src/loader.rs
// description: Safetensors checkpoint loader with MXFP4 dequantization and BF16 tensor materialization.
// author: cipher-rc5

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use bytemuck::try_cast_slice;
use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::mxfp4::dequantize_mxfp4;

#[derive(Deserialize)]
struct SafetensorsIndexFile {
    weight_map: HashMap<String, String>,
}

pub struct ModelCheckpoint {
    tensors: HashMap<String, Tensor>,
}

impl ModelCheckpoint {
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self> {
        let weight_map = load_weight_map(model_dir)?;
        let mut shard_cache: HashMap<PathBuf, Mmap> = HashMap::new();
        let mut tensors = HashMap::new();

        for name in weight_map.keys() {
            if name.ends_with(".scales") {
                let blocks_name = format!("{}.blocks", name.trim_end_matches(".scales"));
                if weight_map.contains_key(&blocks_name) {
                    continue;
                }
            }

            if let Some(base_name) = name.strip_suffix(".blocks") {
                let scales_name = format!("{base_name}.scales");
                let blocks = load_raw(name, &weight_map, model_dir, &mut shard_cache)?;
                let scales = load_raw(&scales_name, &weight_map, model_dir, &mut shard_cache)
                    .with_context(|| format!("missing scales tensor for blocks tensor {name}"))?;

                let blocks_shape = blocks.shape.clone();
                anyhow::ensure!(
                    !blocks_shape.is_empty(),
                    "invalid empty shape for tensor {name}"
                );

                let rows = blocks_shape[..blocks_shape.len() - 1]
                    .iter()
                    .copied()
                    .product::<usize>()
                    .max(1);
                let cols = blocks_shape[blocks_shape.len() - 1] * 2;

                let blocks_bytes = blocks.data.as_slice();
                let scales_u16: &[u16] = try_cast_slice(scales.data.as_slice()).map_err(|e| {
                    anyhow::anyhow!("failed to cast scales data for {scales_name} to u16: {e}")
                })?;

                let dequant_bits = dequantize_mxfp4(blocks_bytes, scales_u16, rows, cols);
                let values: Vec<bf16> = dequant_bits.into_iter().map(bf16::from_bits).collect();

                let mut out_shape = blocks_shape[..blocks_shape.len() - 1].to_vec();
                out_shape.push(cols);
                let tensor = Tensor::from_vec(values, out_shape, device)?.to_dtype(DType::BF16)?;
                tensors.insert(base_name.to_string(), tensor);
                continue;
            }

            let raw = load_raw(name, &weight_map, model_dir, &mut shard_cache)?;
            let tensor = tensor_from_raw(raw, device)
                .with_context(|| format!("failed to load tensor {name}"))?;
            tensors.insert(name.clone(), tensor);
        }

        Ok(Self { tensors })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing tensor '{name}' in checkpoint"))
    }

    pub fn tensors(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }
}

fn load_weight_map(model_dir: &Path) -> Result<HashMap<String, String>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let file = File::open(&index_path)
            .with_context(|| format!("failed to open index file {}", index_path.display()))?;
        let index: SafetensorsIndexFile = serde_json::from_reader(file)
            .with_context(|| format!("failed to parse index file {}", index_path.display()))?;
        return Ok(index.weight_map);
    }

    let single_file = model_dir.join("model.safetensors");
    let shard = open_shard(&single_file)?;
    let tensors = SafeTensors::deserialize(&shard).with_context(|| {
        format!(
            "failed to parse safetensors shard {}",
            single_file.display()
        )
    })?;
    let mut map = HashMap::new();
    for name in tensors.names() {
        map.insert(name.to_string(), "model.safetensors".to_string());
    }
    Ok(map)
}

fn tensor_from_raw(raw: RawTensor, device: &Device) -> Result<Tensor> {
    let shape = raw.shape;
    match raw.dtype {
        Dtype::BF16 => {
            let bits: &[u16] = try_cast_slice(raw.data.as_slice())
                .map_err(|e| anyhow::anyhow!("BF16 tensor cast to u16 failed: {e}"))?;
            let out: Vec<bf16> = bits.iter().copied().map(bf16::from_bits).collect();
            Ok(Tensor::from_vec(out, shape, device)?.to_dtype(DType::BF16)?)
        }
        Dtype::F16 => {
            let bits: &[u16] = try_cast_slice(raw.data.as_slice())
                .map_err(|e| anyhow::anyhow!("F16 tensor cast to u16 failed: {e}"))?;
            let out: Vec<bf16> = bits
                .iter()
                .copied()
                .map(|bits| f16::from_bits(bits))
                .map(|v| bf16::from_f32(v.to_f32()))
                .collect();
            Ok(Tensor::from_vec(out, shape, device)?.to_dtype(DType::BF16)?)
        }
        Dtype::F32 => {
            let vals: &[f32] = try_cast_slice(raw.data.as_slice())
                .map_err(|e| anyhow::anyhow!("F32 tensor cast failed: {e}"))?;
            let out: Vec<bf16> = vals.iter().copied().map(bf16::from_f32).collect();
            Ok(Tensor::from_vec(out, shape, device)?.to_dtype(DType::BF16)?)
        }
        Dtype::U8 => Ok(Tensor::from_vec(raw.data, shape, device)?),
        Dtype::U16 => {
            let vals: &[u16] = try_cast_slice(raw.data.as_slice())
                .map_err(|e| anyhow::anyhow!("U16 tensor cast failed: {e}"))?;
            let out: Vec<u32> = vals.iter().copied().map(|v| v as u32).collect();
            Ok(Tensor::from_vec(out, shape, device)?)
        }
        Dtype::U32 => {
            let vals: &[u32] = try_cast_slice(raw.data.as_slice())
                .map_err(|e| anyhow::anyhow!("U32 tensor cast failed: {e}"))?;
            Ok(Tensor::from_vec(vals.to_vec(), shape, device)?)
        }
        Dtype::I32 => {
            let vals: &[i32] = try_cast_slice(raw.data.as_slice())
                .map_err(|e| anyhow::anyhow!("I32 tensor cast failed: {e}"))?;
            Ok(Tensor::from_vec(vals.to_vec(), shape, device)?)
        }
        other => {
            anyhow::bail!("unsupported tensor dtype {other:?} for tensor load")
        }
    }
}

struct RawTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn open_shard(path: &Path) -> Result<Mmap> {
    let file =
        File::open(path).with_context(|| format!("failed to open shard {}", path.display()))?;
    // SAFETY: the underlying file is immutable for checkpoint reading, and the mmap
    // is kept alive by the cache while tensor views are materialized.
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to mmap shard {}", path.display()))?;
    Ok(mmap)
}

fn load_raw(
    name: &str,
    weight_map: &HashMap<String, String>,
    model_dir: &Path,
    cache: &mut HashMap<PathBuf, Mmap>,
) -> Result<RawTensor> {
    let shard_rel = weight_map
        .get(name)
        .with_context(|| format!("tensor {name} missing from index weight_map"))?;
    let shard_path = model_dir.join(shard_rel);
    if !cache.contains_key(&shard_path) {
        cache.insert(shard_path.clone(), open_shard(&shard_path)?);
    }

    let shard = cache.get(&shard_path).unwrap();
    let tensors = SafeTensors::deserialize(shard)
        .with_context(|| format!("failed to parse safetensors shard {}", shard_path.display()))?;
    let view = tensors
        .tensor(name)
        .with_context(|| format!("tensor {name} missing from shard {}", shard_path.display()))?;

    Ok(RawTensor {
        dtype: view.dtype(),
        shape: view.shape().to_vec(),
        data: view.data().to_vec(),
    })
}

pub fn checkpoint_tensor_names(model_dir: &Path) -> Result<Vec<String>> {
    let weight_map = load_weight_map(model_dir)?;
    let mut out = BTreeSet::new();
    for name in weight_map.keys() {
        if let Some(base) = name.strip_suffix(".blocks") {
            out.insert(base.to_string());
        } else if name.ends_with(".scales") {
            continue;
        } else {
            out.insert(name.clone());
        }
    }
    Ok(out.into_iter().collect())
}
