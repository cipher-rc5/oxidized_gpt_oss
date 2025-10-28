# Oxidized GPT-OSS

A high-performance Rust implementation of GPT-OSS inference optimized for Apple M3 Ultra with Metal GPU acceleration and advanced Mixture of Experts (MoE) support.

## Features

- **High-Performance Inference**: Optimized for Apple M3 Ultra with Metal GPU acceleration
- **Mixture of Experts (MoE) Support**: Advanced routing and load balancing for efficient expert computation
- **Memory Management**: Sophisticated memory pooling and optimization for large models
- **Flexible Configuration**: Support for multiple precision formats (f16, bf16, f32)
- **Comprehensive Metrics**: Detailed MoE performance tracking and analysis
- **Interactive Mode**: Real-time chat interface with MoE statistics
- **Benchmarking**: Built-in MoE performance benchmarking tools

## Installation

### Prerequisites

- Rust 2024 edition or later
- macOS with Apple M3 Ultra (recommended)
- Metal GPU acceleration support

### Building

```bash
# Clone the repository
git clone <repository-url>
cd oxidized_gpt_oss

# Build in release mode with optimizations
./build.sh

# Or build manually
cargo build --release
```

## Usage

### Basic Inference

```bash
# Run with a model path
./target/release/oxidized_gpt_oss --model-path /path/to/model --prompt "Hello, world!"

# Interactive mode
./target/release/oxidized_gpt_oss --model-path /path/to/model --interactive

# With MoE enabled
./target/release/oxidized_gpt_oss --model-path /path/to/model --enable-moe --show-moe-metrics --interactive
```

### Command Line Options

- `-m, --model-path <PATH>`: Path to model directory (required)
- `-t, --tokenizer-path <PATH>`: Path to tokenizer (optional)
- `-i, --interactive`: Interactive chat mode
- `-p, --prompt <PROMPT>`: Input prompt for single inference
- `--max-tokens <N>`: Maximum tokens to generate (default: 512)
- `--temperature <T>`: Sampling temperature (default: 0.7)
- `--top-p <P>`: Top-p nucleus sampling (default: 0.9)
- `--use-metal`: Enable Metal GPU acceleration (default: true)
- `--use-mmap`: Enable memory mapping for large models (default: true)
- `--precision <TYPE>`: Model precision (f16, bf16, f32) (default: f16)

#### MoE-Specific Options

- `--enable-moe`: Enable Mixture of Experts
- `--num-experts <N>`: Number of experts (default: 8)
- `--experts-per-token <N>`: Experts per token (default: 2)
- `--routing-strategy <STRATEGY>`: Routing strategy (topk, switch, expert1) (default: topk)
- `--expert-capacity-factor <F>`: Expert capacity factor (default: 1.0)
- `--show-moe-metrics`: Show detailed MoE metrics
- `--benchmark-moe`: Run MoE benchmark mode
- `--benchmark-runs <N>`: Number of benchmark runs (default: 3)
- `--memory-stats`: Print memory statistics

### MoE Configuration

The implementation supports multiple routing strategies:

- **TopK**: Select top-k experts per token (recommended for most use cases)
- **Switch**: Switch routing for load balancing
- **Expert1**: Single expert per token (fastest, least flexible)

### Memory Management

The system includes advanced memory management features:

- **Memory Pooling**: Efficient tensor reuse
- **MoE Memory Tracking**: Per-expert memory usage monitoring
- **Automatic Cleanup**: Periodic memory cleanup during generation
- **Memory Mapping**: Support for large model memory mapping

## Architecture

### Core Components

- **Model Layer** (`src/model.rs`): GPT model implementation with MoE support
- **Inference Engine** (`src/inference.rs`): Main inference orchestration with MoE metrics
- **Memory Manager** (`src/memory.rs`): Advanced memory management and optimization
- **Configuration** (`src/config.rs`): Model and MoE configuration handling

### MoE Implementation

The MoE implementation includes:

- **Expert Management**: Dynamic expert loading and memory tracking
- **Routing Strategies**: Multiple routing algorithms for expert selection
- **Load Balancing**: Advanced load balancing to prevent expert overload
- **Performance Metrics**: Comprehensive tracking of MoE performance
- **Batched Computation**: Efficient batched expert computation

## Performance

### Optimizations

- **Metal GPU Acceleration**: Optimized for Apple M3 Ultra
- **f16 Precision**: Default half-precision for memory efficiency
- **Memory Mapping**: Support for large model memory mapping
- **Batched Processing**: Efficient batch processing for multiple experts

### Benchmarking

Use the built-in benchmarking tools to evaluate MoE performance:

```bash
./target/release/oxidized_gpt_oss --model-path /path/to/model --enable-moe --benchmark-moe --show-moe-metrics
```

## Development

### Building for Development

```bash
cargo build
```

### Running Tests

```bash
cargo test
```

### Code Style

- **Formatting**: Use `dprint fmt` or `rustfmt --edition 2021`
- **Linting**: Run `cargo clippy` to check for issues
- **Line Length**: 120 characters
- **Indentation**: 2 spaces

### Adding New Features

1. Follow the existing code style and patterns
2. Add appropriate error handling with `anyhow::Result`
3. Include comprehensive documentation
4. Add tests for new functionality
5. Update this README if adding user-facing features

## Troubleshooting

### Common Issues

1. **Runtime Panic**: Ensure you're not creating nested tokio runtimes
2. **Memory Issues**: Use `--use-mmap` for large models
3. **GPU Issues**: Verify Metal support and try `--use-metal=false` if needed

### Performance Tips

- Use f16 precision for best performance on M3 Ultra
- Enable memory mapping for models > 16GB
- Use appropriate expert counts based on available memory
- Monitor MoE metrics to optimize routing strategies

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Projects

- [OpenAI GPT-OSS](https://github.com/openai/gpt-oss): Original GPT-OSS implementation
- [Unsloth GPT-OSS](https://github.com/unslothai/unsloth): Optimized versions and tools
- [Candle Framework](https://github.com/huggingface/candle): ML framework used for inference