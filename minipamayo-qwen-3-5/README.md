# MiniPamayo Qwen3.5

Minimal Qwen3.5-native Stage 1 experiments live here.

Current workflow:

1. Convert episodes under `datasets/raw/<collection_name>/` into `jsonl + images/` with a data config
2. Run a short profiling probe for VRAM and step time
3. Run a longer Stage 1 training loop with a JSON config, explicit train/val inputs, and checkpoints
4. Evaluate Stage 1 on a test split with token metrics and trajectory metrics

Recommended repo layout:

- Raw episode logs: `datasets/raw/<collection_name>/<episode_id>/`
- Extracted Stage 1 datasets: `datasets/processed/stage1/<dataset_name>/`
- Split files: `datasets/splits/stage1/`
- Data configs: `configs/data/stage1/*.json`
- Train configs: `configs/train/stage1/*.json`
- Eval configs: `configs/eval/stage1/*.json`

See `datasets/README.md` for the expected directory structure.

Canonical entrypoints are:

- `minipamayo_qwen35.data.stage1`
- `minipamayo_qwen35.train.profile_stage1`
- `minipamayo_qwen35.train.stage1`
- `minipamayo_qwen35.eval.stage1`

Recorded-entrypoint examples:

- `uv run python -m minipamayo_qwen35.data.stage1 --config-json configs/data/stage1/ignore_rule_data.json`
- `uv run python -m minipamayo_qwen35.train.stage1 --config-json configs/train/stage1/ignore_rule_data_12gb.json`
- `uv run python -m minipamayo_qwen35.eval.stage1 --config-json configs/eval/stage1/ignore_rule_data_12gb.json`

The data, train, and eval entrypoints intentionally reject CLI overrides so the JSON files remain the full run record.

Stage 1 train configs now support:

- `train_jsonl` as one or more required training split JSONLs and `val_jsonl` as zero or more optional validation split JSONLs
  Each value may be either a single path string or a JSON array of path strings.
- `resume_from_checkpoint` for epoch-level resume from `last.pt`
- `early_stopping_patience` and `early_stopping_min_delta`
- `image_min_pixels` and `image_max_pixels` to control the Qwen processor image-token budget
  `0` keeps the processor default, and smaller `image_max_pixels` reduces image tokens for tighter VRAM budgets

Stage 1 eval configs use `test_jsonl` explicitly so the evaluation split is named separately from train/val.

Stage 1 train outputs record:

- `run_config.json` with resolved args plus git commit, dataset fingerprint, GPU info, and processor settings
- `summary.json` with the same run metadata plus final metrics
- `last.pt`, `best.pt`, and `final.pt` for resume and comparison
