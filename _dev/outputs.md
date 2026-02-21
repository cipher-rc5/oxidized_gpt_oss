# outputs

## iteration_1

```md
oxidized_gpt_oss on  main [?] is 📦 v0.1.0 via 🦀 took 23s
❯ cargo run -- --model-path /Users/excalibur/Desktop/oxidized_gpt_oss/gpt-oss-20b-MXFP4-Q8 --max-tokens 20
Finished `dev` profile [optimized + debuginfo] target(s) in 0.05s
Running `target/debug/oxidized_gpt_oss --model-path /Users/excalibur/Desktop/oxidized_gpt_oss/gpt-oss-20b-MXFP4-Q8 --max-tokens 20`
2025-10-28T15:51:55.316581Z INFO GPT-OSS-20B MXFP4 Inference Engine
2025-10-28T15:51:55.316602Z INFO Model path: "/Users/excalibur/Desktop/oxidized_gpt_oss/gpt-oss-20b-MXFP4-Q8"
2025-10-28T15:51:55.316608Z INFO Initializing Metal device...
2025-10-28T15:51:55.358710Z INFO Metal device: Apple M3 Ultra
2025-10-28T15:51:55.359619Z INFO Loading model configuration...
2025-10-28T15:51:55.375298Z INFO Model configuration:
2025-10-28T15:51:55.375304Z INFO Layers: 24
2025-10-28T15:51:55.375306Z INFO Hidden size: 2880
2025-10-28T15:51:55.375308Z INFO Attention heads: 64
2025-10-28T15:51:55.375309Z INFO Vocab size: 201088
2025-10-28T15:51:55.375310Z INFO Max sequence length: 131072
2025-10-28T15:51:55.375312Z INFO MoE enabled:
2025-10-28T15:51:55.375313Z INFO Experts: 32
2025-10-28T15:51:55.375314Z INFO Experts per token: 4
2025-10-28T15:51:55.375316Z INFO Loading tokenizer...
2025-10-28T15:51:55.672209Z INFO Initializing inference engine...
2025-10-28T15:51:55.672223Z INFO Initializing inference engine
2025-10-28T15:51:55.672238Z INFO Loading model from "/Users/excalibur/Desktop/oxidized_gpt_oss/gpt-oss-20b-MXFP4-Q8"
2025-10-28T15:51:56.568997Z INFO Loading layer 1/24
2025-10-28T15:51:57.352593Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:51:57.899888Z INFO Loading layer 2/24
2025-10-28T15:51:58.669570Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:51:59.201232Z INFO Loading layer 3/24
2025-10-28T15:51:59.944265Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:00.475660Z INFO Loading layer 4/24
2025-10-28T15:52:01.236807Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:01.764496Z INFO Loading layer 5/24
2025-10-28T15:52:02.516041Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:03.042715Z INFO Loading layer 6/24
2025-10-28T15:52:03.796832Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:04.324922Z INFO Loading layer 7/24
2025-10-28T15:52:05.064691Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:05.595375Z INFO Loading layer 8/24
2025-10-28T15:52:06.335316Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:06.866391Z INFO Loading layer 9/24
2025-10-28T15:52:07.642604Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:08.175084Z INFO Loading layer 10/24
2025-10-28T15:52:08.929842Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:09.459184Z INFO Loading layer 11/24
2025-10-28T15:52:10.228994Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:10.759480Z INFO Loading layer 12/24
2025-10-28T15:52:11.518017Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:12.048278Z INFO Loading layer 13/24
2025-10-28T15:52:12.797844Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:13.328424Z INFO Loading layer 14/24
2025-10-28T15:52:14.110653Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:14.639611Z INFO Loading layer 15/24
2025-10-28T15:52:15.396008Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:15.921166Z INFO Loading layer 16/24
2025-10-28T15:52:16.688878Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:17.216024Z INFO Loading layer 17/24
2025-10-28T15:52:17.975485Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:18.506559Z INFO Loading layer 18/24
2025-10-28T15:52:19.253693Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:19.786424Z INFO Loading layer 19/24
2025-10-28T15:52:20.547238Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:21.077828Z INFO Loading layer 20/24
2025-10-28T15:52:21.802015Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:22.329728Z INFO Loading layer 21/24
2025-10-28T15:52:23.093144Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:23.618545Z INFO Loading layer 22/24
2025-10-28T15:52:24.375231Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:24.899179Z INFO Loading layer 23/24
2025-10-28T15:52:25.688396Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:26.212910Z INFO Loading layer 24/24
2025-10-28T15:52:27.002228Z INFO Warning: intermediate_size in config.json (2880) does not match calculated intermediate_size (720). Using calculated size.
2025-10-28T15:52:28.269761Z INFO Model loaded successfully
2025-10-28T15:52:28.269779Z INFO Prompt: What is the meaning of life?
2025-10-28T15:52:28.269782Z INFO Generating...
2025-10-28T15:52:28.269783Z INFO Starting generation for prompt: What is the meaning of life?
2025-10-28T15:52:28.678123Z INFO Embedding dimension adjustment: actual=1440, expected=2880, copy=1440, output=2880
2025-10-28T15:52:28.678299Z INFO After embedding: buffer_size=20160, seq_len=7, dim_per_token=2880
2025-10-28T15:52:28.811794Z WARN Q bias length 184320 does not match expected columns 2048. Using first 2048 elements.
2025-10-28T15:52:28.816632Z WARN K bias length 23040 does not match expected columns 256. Using first 256 elements.
2025-10-28T15:52:28.821427Z WARN V bias length 23040 does not match expected columns 256. Using first 256 elements.
2025-10-28T15:52:28.868529Z WARN O bias length 184320 does not match expected columns 2880. Using first 2880 elements.
2025-10-28T15:52:37.752078Z INFO Embedding dimension adjustment: actual=1440, expected=2880, copy=1440, output=2880
2025-10-28T15:52:37.752263Z INFO After embedding: buffer_size=23040, seq_len=8, dim_per_token=2880
```
