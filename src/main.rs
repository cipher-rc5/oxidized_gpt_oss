use anyhow::Result;
use clap::Parser;
use oxidized_gpt_oss::{GenerationConfig, InferenceEngine, MetalDevice, ModelConfig};
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tracing::{info, Level};
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
