use std::env;
use std::path::PathBuf;

use anyhow::Result;
use oxidized_gpt_oss::{
    GenerationConfig, InferenceEngine, MetalDevice, ModelConfig, ReasoningEffort,
    apply_harmony_chat_template, dequantize_mxfp4, load_tokenizer,
};

fn model_path_from_env() -> Option<PathBuf> {
    env::var("OXIDIZED_GPT_OSS_MODEL_PATH")
        .ok()
        .map(PathBuf::from)
}

#[test]
fn tokenizer_roundtrip() -> Result<()> {
    let Some(path) = model_path_from_env() else {
        println!("skipping: OXIDIZED_GPT_OSS_MODEL_PATH not set");
        return Ok(());
    };
    let tok = load_tokenizer(&path)?;
    let text = "Hello, world!";
    let ids = oxidized_gpt_oss::tokenizer::encode(&tok, text)?;
    let out = oxidized_gpt_oss::tokenizer::decode(&tok, &ids)?;
    assert_eq!(text, out);
    Ok(())
}

#[test]
fn mxfp4_dequant_sanity() -> Result<()> {
    // Lightweight synthetic sanity check when no weights are available.
    let rows = 2;
    let cols = 32;
    let blocks = vec![0x21u8; rows * cols / 2];
    let scales = vec![half::bf16::from_f32(0.1).to_bits(); rows * (cols / 32)];
    let out = dequantize_mxfp4(&blocks, &scales, rows, cols);
    let vals: Vec<f32> = out
        .into_iter()
        .map(|b| half::bf16::from_bits(b).to_f32())
        .collect();
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / vals.len() as f32;
    let std = var.sqrt();
    assert!(mean.abs() < 1.0);
    assert!(std > 0.0);
    Ok(())
}

#[test]
fn harmony_format_prefix() -> Result<()> {
    let rendered =
        apply_harmony_chat_template(Some("You are helpful."), "hi", ReasoningEffort::Low);
    assert!(rendered.starts_with("<|start|>system<|message|>"));
    Ok(())
}

#[test]
fn greedy_generation_determinism() -> Result<()> {
    let Some(path) = model_path_from_env() else {
        println!("skipping: OXIDIZED_GPT_OSS_MODEL_PATH not set");
        return Ok(());
    };

    let config = ModelConfig::load_from_path(&path)?;
    let tok = load_tokenizer(&path)?;
    let device = MetalDevice::new()?;
    let engine = InferenceEngine::new(&path, &config, device)?;
    let cfg = GenerationConfig {
        max_tokens: 16,
        temperature: 0.0,
        top_p: 1.0,
        reasoning_effort: ReasoningEffort::Low,
        show_thinking: true,
    };
    let a = engine.generate_chat(&tok, None, "The sky is", &cfg)?;
    let b = engine.generate_chat(&tok, None, "The sky is", &cfg)?;
    assert_eq!(a, b);
    Ok(())
}
