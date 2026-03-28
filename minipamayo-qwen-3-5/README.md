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
- Data configs: `configs/stage1/data/*.json`
- Train configs: `configs/stage1/train/canonical/*.json`
- Eval configs: `configs/stage1/eval/canonical/*.json`
- Experiment configs: `configs/stage1/train/experiments/**` and `configs/stage1/eval/experiments/**`

See `datasets/README.md` for the expected directory structure.

Canonical entrypoints are:

- `minipamayo_qwen35.stage1.data`
- `minipamayo_qwen35.stage1.train.profile`
- `minipamayo_qwen35.stage1.train`
- `minipamayo_qwen35.stage1.eval`

Stage 1 experiments live under the same stage package:

- `minipamayo_qwen35.stage1.train.experiments.steer_only`
- `minipamayo_qwen35.stage1.eval.experiments.steer_only`

Recorded-entrypoint examples:

- `uv run python -m minipamayo_qwen35.stage1.data --config-json configs/stage1/data/ignore_rule_data.json`
- `uv run python -m minipamayo_qwen35.stage1.train --config-json configs/stage1/train/canonical/ignore_rule_data_12gb.json`
- `uv run python -m minipamayo_qwen35.stage1.eval --config-json configs/stage1/eval/canonical/ignore_rule_data_12gb.json`
- `uv run python -m minipamayo_qwen35.stage1.train.experiments.steer_only --config-json configs/stage1/train/experiments/steer_only/ignore_rule_data_12gb.json`

The data, train, and eval entrypoints intentionally reject CLI overrides so the JSON files remain the full run record.

Stage 1 train configs now support:

- `train_jsonl` as one or more required training split JSONLs and `val_jsonl` as zero or more optional validation split JSONLs
  Each value may be either a single path string or a JSON array of path strings.
- `resume_from_checkpoint` for epoch-level resume from `last.pt`
- `early_stopping_patience` and `early_stopping_min_delta`
- fixed Alpamayo-style image budget: `image_min_pixels=163840`, `image_max_pixels=196608`
  Canonical paths reject any other values so unlimited image tokens never slip in by accident.

Stage 1 eval configs use `test_jsonl` explicitly so the evaluation split is named separately from train/val.

Stage 1 train outputs record:

- `run_config.json` with resolved args plus git commit, dataset fingerprint, GPU info, and processor settings
- `summary.json` with the same run metadata plus final metrics
- `last.pt`, `best.pt`, and `final.pt` for resume and comparison

Canonical Stage 1 predicts interleaved acceleration plus curvature tokens and is the only Stage 1 path that connects to the current Stage 2/3/4 pipeline. The `steer_only` entrypoints are experiment-only and record a train-corpus-derived `kappa_range` in `run_config.json`, `summary.json`, and checkpoint metadata.
