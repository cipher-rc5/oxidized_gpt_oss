// file: src/backend/metal/mod.rs
// description: Native Metal backend module exports.
// author: cipher-rc5

pub mod kernels;
pub mod metal_impl;

pub use metal_impl::{MetalBackend, MetalBuffer, MetalCompute, MetalDevice};
pub use objc2_metal::MTLStorageMode as StorageMode;
