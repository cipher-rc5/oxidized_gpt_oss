#!/bin/bash

# GPT-OSS Rust Build Script for Mac M3 Ultra
# Optimized for maximum performance with Metal acceleration

set -e

echo " Building GPT-OSS Runner for Mac M3 Ultra..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "WARNING:  This build script is optimized for macOS. Continuing anyway..."
fi

# Set optimization flags for M3 Ultra
export RUSTFLAGS="\
    -C target-cpu=native \
    -C opt-level=3 \
    -C lto=fat \
    -C codegen-units=1 \
    -C panic=abort \
    -C link-arg=-fuse-ld=lld \
"

# Enable Metal and Accelerate framework support
export CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS="\
    $RUSTFLAGS \
    -C link-arg=-framework \
    -C link-arg=Metal \
    -C link-arg=-framework \
    -C link-arg=MetalKit \
    -C link-arg=-framework \
    -C link-arg=Accelerate \
"

# Create optimized build
echo " Building with maximum optimizations..."
cargo build --release --target aarch64-apple-darwin

# Optional: Build with specific Metal features
echo " Building with Metal features..."
cargo build --release --features "metal,accelerate" --target aarch64-apple-darwin

# Create run script
echo " Creating run script..."
cat > run_gpt.sh << 'EOF'
#!/bin/bash

# GPT-OSS Runner Script
# Usage: ./run_gpt.sh [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/target/aarch64-apple-darwin/release/gpt-oss-runner"

# Default paths - modify these for your setup
MODEL_PATH="${MODEL_PATH:-./models/gpt-oss-20b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./models/gpt-oss-20b/tokenizer.json}"

# Performance settings for M3 Ultra
export RUST_LOG="${RUST_LOG:-info}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"  # M3 Ultra has 24 cores
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-24}"

# Metal GPU settings
export MTL_DEVICE_MEMORY_COALESCED=1
export MTL_DEBUG_LAYER=0  # Set to 1 for debugging

# Memory settings for large model
ulimit -n 65536  # Increase file descriptor limit
# ulimit -m unlimited  # Uncomment if you need unlimited memory

echo " Starting GPT-OSS Runner..."
echo " Model path: $MODEL_PATH"
echo " Tokenizer path: $TOKENIZER_PATH"
echo "  Using $(sysctl -n hw.ncpu) CPU cores"
echo " Available memory: $(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}')"

# Check if model exists
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "❌ Model directory not found: $MODEL_PATH"
    echo "Please set MODEL_PATH environment variable or place model in ./models/gpt-oss-20b"
    exit 1
fi

# Run with provided arguments or interactive mode
if [[ $# -eq 0 ]]; then
    echo "  Starting in interactive mode..."
    exec "$BINARY" \
        --model-path "$MODEL_PATH" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --interactive \
        --use-metal true \
        --precision f16
else
    exec "$BINARY" \
        --model-path "$MODEL_PATH" \
        --tokenizer-path "$TOKENIZER_PATH" \
        "$@"
fi
EOF

chmod +x run_gpt.sh

# Create benchmark script
echo " Creating benchmark script..."
cat > benchmark.sh << 'EOF'
#!/bin/bash

# GPT-OSS Benchmark Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-./models/gpt-oss-20b}"

echo " Running GPT-OSS Benchmarks..."

# Test prompts for benchmarking
PROMPTS=(
    "The future of artificial intelligence is"
    "In a world where technology has advanced beyond our wildest dreams"
    "Write a short story about a robot who discovers emotions"
)

for precision in f16 f32; do
    echo " Testing with $precision precision..."

    for prompt in "${PROMPTS[@]}"; do
        echo " Testing prompt: ${prompt:0:50}..."

        time ./run_gpt.sh \
            --prompt "$prompt" \
            --max-tokens 100 \
            --precision "$precision" \
            --temperature 0.7

        echo "---"
    done
done

echo " Benchmark complete!"
EOF

chmod +x benchmark.sh

# Create memory monitoring script
echo " Creating memory monitoring script..."
cat > monitor_memory.sh << 'EOF'
#!/bin/bash

# Monitor memory usage during inference

echo " Starting memory monitoring..."
echo "Time,RSS_MB,Virtual_MB,GPU_MB" > memory_log.csv

while true; do
    # Get process info for gpt-oss-runner
    PID=$(pgrep -f gpt-oss-runner | head -1)

    if [[ -n "$PID" ]]; then
        # Get memory usage
        RSS=$(ps -o rss= -p $PID 2>/dev/null | awk '{print $1/1024}')
        VIRTUAL=$(ps -o vsz= -p $PID 2>/dev/null | awk '{print $1/1024}')

        # Get GPU memory (Metal doesn't have easy monitoring, so estimate)
        GPU_MEM="N/A"

        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

        if [[ -n "$RSS" ]]; then
            echo "$TIMESTAMP,$RSS,$VIRTUAL,$GPU_MEM" >> memory_log.csv
            echo "[$TIMESTAMP] RSS: ${RSS}MB, Virtual: ${VIRTUAL}MB"
        fi
    fi

    sleep 2
done
EOF

chmod +x monitor_memory.sh

echo " Build complete!"
echo ""
echo " Usage:"
echo "  ./run_gpt.sh --interactive                    # Interactive mode"
echo "  ./run_gpt.sh --prompt \"Hello world\"          # Single generation"
echo "  ./benchmark.sh                               # Run benchmarks"
echo "  ./monitor_memory.sh                          # Monitor memory usage"
echo ""
echo " Configuration:"
echo "  Set MODEL_PATH environment variable to your model directory"
echo "  Set TOKENIZER_PATH for custom tokenizer location"
echo ""
echo " Performance tips:"
echo "  - Use f16 precision for better memory usage"
echo "  - Enable --use-mmap for very large models"
echo "  - Monitor memory with Activity Monitor"
echo "  - Consider model quantization for even better performance"
