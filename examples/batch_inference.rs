// use anyhow::Result;
// use gpt_metal_mxfp4::inference::GenerationConfig;
// use gpt_metal_mxfp4::{InferenceEngine, MetalDevice, ModelConfig};
// use std::path::Path;
// use std::sync::Arc;
// use tokenizers::Tokenizer;

// fn main() -> Result<()> {
//     tracing_subscriber::fmt()
//         .with_max_level(tracing::Level::INFO)
//         .init();

//     let model_path = Path::new("path/to/gpt-oss-20b-mxfp4");

//     let device = Arc::new(MetalDevice::new()?);
//     println!("Initialized Metal device");

//     let config = ModelConfig::load_from_path(model_path)?;
//     println!("Loaded model config");

//     let engine = InferenceEngine::new(model_path, &config, device)?;
//     println!("Loaded inference engine");

//     let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))?;
//     println!("Loaded tokenizer");

//     let prompts = vec![
//         "Explain quantum computing in simple terms:",
//         "Write a haiku about artificial intelligence:",
//         "What are the benefits of renewable energy?",
//     ];

//     let gen_config = GenerationConfig {
//         max_tokens: 256,
//         temperature: 0.8,
//         top_p: 0.95,
//         top_k: Some(40),
//         repetition_penalty: 1.1,
//     };

//     for (i, prompt) in prompts.iter().enumerate() {
//         println!("\n{'=':.^60}", format!(" Example {} ", i + 1));
//         println!("Prompt: {}", prompt);
//         println!("{:-^60}", "");

//         let start = std::time::Instant::now();
//         let output = engine.generate(prompt, &tokenizer, &gen_config)?;
//         let elapsed = start.elapsed();

//         println!("{}", output);
//         println!("\n{:-^60}", "");
//         println!("Generated in {:.2}s", elapsed.as_secs_f64());
//     }

//     Ok(())
// }
fn main() {
    println!("example disabled: batch_inference scaffold");
}
