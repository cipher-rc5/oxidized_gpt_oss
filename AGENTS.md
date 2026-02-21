# AGENTS.md - Coding Agent Guidelines

## Build/Test Commands

- Build: `cargo build --release`
- Optimized build: `./build.sh` (includes Metal/M3 Ultra optimizations)
- Format: `dprint fmt` or `rustfmt --edition 2021 src/**/*.rs`
- Lint: `cargo clippy`
- Run: `cargo run -- --help` or `./run_gpt.sh` (after build)
- Test: `cargo test` (no tests currently implemented)
- Benchmark: `./benchmark.sh`

## Code Style

- **Language**: Rust 2024 edition
- **Imports**: Group std, external crates, then local modules with blank lines between
- **Formatting**: Use dprint (configured in dprint.json) - 120 char lines, 2-space indent
- **Types**: Explicit types for public APIs, use type aliases for complex types
- **Naming**: snake_case for variables/functions, PascalCase for types, SCREAMING_SNAKE for constants
- **Error handling**: Use `anyhow::Result<T>` for fallible functions, `.with_context()` for error context
- **Async**: Use tokio runtime, prefer async/await over futures combinators
- **Memory**: Use `candle_core::Device` abstraction, prefer Metal GPU on macOS

## Architecture

- Main binary in `src/main.rs` with CLI using clap
- Config loading in `src/config.rs` with serde deserialization
- Core model logic in `src/model.rs` using candle framework
- Memory management in `src/memory.rs`
- Inference engine in `src/inference.rs`

## Performance Notes

- Target Apple M3 Ultra with Metal acceleration
- Use f16 precision by default for memory efficiency
- Enable memory mapping for large models
- Profile with Activity Monitor for memory usage

## Notes

- No Cursor rules (.cursor/rules/ or .cursorrules) found
- No Copilot instructions (.github/copilot-instructions.md) found
