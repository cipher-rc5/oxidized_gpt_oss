# Oxidized GPT-OSS

Rust/Candle inference runner for `openai/gpt-oss-20b` with MXFP4 checkpoint loading and Harmony-formatted prompting.   
THIS UNIT IS NOT PRODUCTION GRADE AND SHOULD NOT BE USED IN ANY CAPACITY, ITS STRICTLY FOR LEARNING THE INTERNALS OF GPT-OSS

## Download Weights

Use the official OpenAI repository layout:

```bash
hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
```

Point `--model-path` at the downloaded `original/` directory.

## Build

```bash
cargo build --release
```

Optional optimized script:

```bash
./build.sh
```

## Usage

List tensors (smoke test for checkpoint loading):

```bash
./target/release/oxidized_gpt_oss --model-path /path/to/gpt-oss-20b/original --list-tensors
```

Single prompt:

```bash
./target/release/oxidized_gpt_oss \
  --model-path /path/to/gpt-oss-20b/original \
  --prompt "The capital of France is" \
  --temperature 1.0 --top-p 1.0 --max-tokens 128
```

Interactive:

```bash
./target/release/oxidized_gpt_oss \
  --model-path /path/to/gpt-oss-20b/original \
  --interactive
```

## Important Notes

- The model expects Harmony formatting; prompts are wrapped by the runtime before tokenization.
- Recommended sampling for gpt-oss: `--temperature 1.0 --top-p 1.0`.
- `--reasoning-effort` supports: `low`, `medium`, `high`.
- Use `--show-thinking` to keep reasoning/control markers in decoded output.

## CLI Flags

- `--model-path <PATH>` (required)
- `--reasoning-effort <low|medium|high>`
- `--max-tokens <N>`
- `--temperature <T>`
- `--top-p <P>`
- `--interactive`
- `--prompt <TEXT>`
- `--show-thinking`
- `--verbose`
- `--use-metal`
- `--no-metal`
- `--bench`
- `--list-tensors`

## Tests

```bash
cargo test
```

Integration tests that need model files look for:

```bash
export OXIDIZED_GPT_OSS_MODEL_PATH=/path/to/gpt-oss-20b/original
```
