# MiniPamayo Qwen3.5

Minimal Qwen3.5-native Stage 1 experiments live here.

Current workflow:

1. Convert an episode `telemetry.mcap` into `jsonl + images/`
2. Run a short profiling probe for VRAM and step time
3. Run a longer Stage 1 training loop with validation and checkpoints

The first usable scripts are:

- `minipamayo_qwen35.data.mcap_stage1_extractor`
- `minipamayo_qwen35.profile_stage1`
- `minipamayo_qwen35.train_stage1`
