// file: src/main.rs
// description: CLI entrypoint for gpt-oss-20b loading, tensor listing, single prompt generation, and interactive mode.
// author: cipher-rc5
// created: 2026-02-21
// modified: 2026-02-21

use anyhow::Result;
use candle_core::Device;
use clap::{ArgAction, Parser};
use oxidized_gpt_oss::{
    GenerationConfig, InferenceEngine, ModelCheckpoint, ModelConfig, ReasoningEffort,
};
use std::io::{self, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tracing::Level;

#[derive(Parser, Debug)]
#[command(author, version, about = "gpt-oss-20b Candle runner")]
struct Args {
    #[arg(long, help = "Path to gpt-oss-20b model directory")]
    model_path: PathBuf,

    #[arg(long, help = "List checkpoint tensors and exit")]
    list_tensors: bool,

    #[arg(long, value_enum, default_value = "low")]
    reasoning_effort: EffortArg,

    #[arg(long, default_value_t = 512)]
    max_tokens: usize,

    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    #[arg(long, default_value_t = 1.0)]
    top_p: f32,

    #[arg(long)]
    interactive: bool,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long, action = ArgAction::SetTrue)]
    show_thinking: bool,

    #[arg(long, action = ArgAction::SetTrue)]
    verbose: bool,

    #[arg(long, action = ArgAction::SetTrue, default_value_t = true)]
    use_metal: bool,

    #[arg(long, action = ArgAction::SetTrue)]
    no_metal: bool,

    #[arg(long, action = ArgAction::SetTrue)]
    bench: bool,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum EffortArg {
    Low,
    Medium,
    High,
}

impl From<EffortArg> for ReasoningEffort {
    fn from(v: EffortArg) -> Self {
        match v {
            EffortArg::Low => ReasoningEffort::Low,
            EffortArg::Medium => ReasoningEffort::Medium,
            EffortArg::High => ReasoningEffort::High,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_max_level(if args.verbose {
            Level::DEBUG
        } else {
            Level::INFO
        })
        .with_target(false)
        .init();

    let model_path = resolve_model_path(&args.model_path);

    if args.list_tensors {
        let checkpoint = ModelCheckpoint::load(&model_path, &Device::Cpu)?;
        let mut entries: Vec<_> = checkpoint.tensors().iter().collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));
        for (name, tensor) in entries {
            println!("{}\t{:?}", name, tensor.shape().dims());
        }
        return Ok(());
    }

    let config = ModelConfig::load_from_path(&model_path)?;
    config.validate_against_official_20b()?;

    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    let allow_metal = args.use_metal && !args.no_metal;
    if !allow_metal {
        eprintln!(
            "warning: --no-metal requested; using compatibility backend if built without metal feature"
        );
    }

    let device = oxidized_gpt_oss::MetalDevice::new()?;
    let engine = InferenceEngine::new(&model_path, &config, device)?;

    let gen_cfg = GenerationConfig {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        reasoning_effort: args.reasoning_effort.into(),
        show_thinking: args.show_thinking,
    };

    if args.interactive {
        run_interactive(&engine, &tokenizer, &gen_cfg, args.bench)?;
        return Ok(());
    }

    let prompt = args.prompt.as_deref().unwrap_or("The capital of France is");
    run_once(&engine, &tokenizer, &gen_cfg, prompt, args.bench)
}

fn resolve_model_path(input: &std::path::Path) -> PathBuf {
    if input.exists() {
        return input.to_path_buf();
    }
    if let Some(name) = input.file_name() {
        let candidate = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(name);
        if candidate.exists() {
            eprintln!(
                "warning: model path '{}' not found, using '{}'",
                input.display(),
                candidate.display()
            );
            return candidate;
        }
    }
    input.to_path_buf()
}

fn run_once(
    engine: &InferenceEngine,
    tokenizer: &Tokenizer,
    cfg: &GenerationConfig,
    prompt: &str,
    bench: bool,
) -> Result<()> {
    let start = std::time::Instant::now();
    let out = engine.generate_chat(tokenizer, None, prompt, cfg)?;
    println!("{}", out);
    if bench {
        eprintln!("elapsed_ms={}", start.elapsed().as_millis());
    }
    Ok(())
}

fn run_interactive(
    engine: &InferenceEngine,
    tokenizer: &Tokenizer,
    cfg: &GenerationConfig,
    bench: bool,
) -> Result<()> {
    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "exit" || line == "quit" {
            break;
        }
        run_once(engine, tokenizer, cfg, line, bench)?;
    }
    Ok(())
}
