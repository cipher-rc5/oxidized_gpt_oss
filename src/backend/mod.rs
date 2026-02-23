// file: src/backend/mod.rs
// description: Backend module selector for native Metal or compatibility fallback implementations.
// author: cipher-rc5

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

#[cfg(not(all(feature = "metal", target_os = "macos")))]
#[path = "metal_stub.rs"]
pub mod metal;
