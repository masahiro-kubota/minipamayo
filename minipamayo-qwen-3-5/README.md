# MiniPamayo Qwen3.5

Minimal Qwen3.5-native Stage 1 experiments live here.

Current workflow:

1. Convert episodes under `datasets/raw/<collection_name>/` into `jsonl + images/` with a data config
2. Run a short profiling probe for VRAM and step time
3. Run a longer Stage 1 training loop with a JSON config, validation, and checkpoints
4. Evaluate Stage 1 with token metrics and trajectory metrics

Recommended repo layout:

- Raw episode logs: `datasets/raw/<collection_name>/<episode_id>/`
- Extracted Stage 1 datasets: `datasets/processed/stage1/<dataset_name>/`
- Split files: `datasets/splits/stage1/`
- Data configs: `configs/data/stage1/*.json`
- Train configs: `configs/train/stage1/*.json`
- Eval configs: `configs/eval/stage1/*.json`

See `datasets/README.md` for the expected directory structure.

Canonical entrypoints are:

- `minipamayo_qwen35.data.extract_stage1`
- `minipamayo_qwen35.profile_stage1`
- `minipamayo_qwen35.train.stage1`
- `minipamayo_qwen35.eval.stage1`

Recorded-entrypoint examples:

- `uv run python -m minipamayo_qwen35.data.extract_stage1 --config-json configs/data/stage1/ignore_rule_data.json`
- `uv run python -m minipamayo_qwen35.train.stage1 --config-json configs/train/stage1/ignore_rule_data_12gb.json`

Both entrypoints intentionally reject CLI overrides so the JSON files remain the full run record.
