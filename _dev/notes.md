# notes

## commands
```bash
repomix --style markdown -o _v07-llm.md --verbose --parsable-style --no-file-summary --include src,Cargo.toml
```

MoE logic with Metal is complex, involving custom `MoELayer`/`Expert` structs, router logic with softmax and top-k, and efficient gather/scatter operation

```md
oxidized_gpt_oss/gpt-oss-20b-MXFP4-Q8 on  master [?]
❯ tree
.
├── config.json
├── generation_config.json
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json

1 directory, 9 files
```


cd /Users/excalibur/Desktop/dev/oxidized_gpt_oss && cargo run -- --model-path /Users/excalibur/Desktop/dev/oxidized_gpt_oss/gpt-oss-20b-MXFP4-Q8 --max-tokens 20 2>&1 | tail -50


cd /Users/excalibur/Desktop/dev/oxidized_gpt_oss && cargo run -- --model-path /Users/excalibur/Desktop/dev/oxidized_gpt_oss/gpt-oss-20b-MXFP4-Q8 --max-tokens 1 2>&1 | tail -5 &
sleep 45
pkill -f "cargo run"
wait
