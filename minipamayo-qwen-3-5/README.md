# MiniPamayo Qwen3.5

Paper-aligned Qwen3.5 stage experiments live here.

Current workflow:

1. Convert episodes under `datasets/raw/<collection_name>/` into `jsonl + images/` with a data config
2. Run a short profiling probe for VRAM and step time
3. Run canonical `Stage1A` VLM CE training
4. Run canonical `Stage1B` expert CFM training
5. Evaluate the stage that was trained

Recommended repo layout:

- Raw episode logs: `datasets/raw/<collection_name>/<episode_id>/`
- Extracted Stage 1 datasets: `datasets/processed/stage1/<dataset_name>/`
- Split files: `datasets/splits/stage1/`
- Data configs: `configs/stage1/data/*.json`
- Stage1A train configs: `configs/stage1/vlm_ce/train/**`
- Stage1A eval configs: `configs/stage1/vlm_ce/eval/**`
- Stage1A inference configs: `configs/stage1/vlm_ce/inference/**`
- Stage1B train/eval/inference configs: `configs/stage1/expert_cfm/**`
- Stage2 configs: `configs/stage2/reasoning_sft/**`
- Stage3 configs: `configs/stage3/post_training/**`

See `datasets/README.md` for the expected directory structure.

Canonical entrypoints are:

- `minipamayo_qwen35.stage1.data`
- `minipamayo_qwen35.stage1.vlm_ce.train.profile`
- `minipamayo_qwen35.stage1.vlm_ce.train`
- `minipamayo_qwen35.stage1.vlm_ce.eval`
- `minipamayo_qwen35.stage1.vlm_ce.inference`
- `minipamayo_qwen35.stage1.expert_cfm.train`
- `minipamayo_qwen35.stage1.expert_cfm.eval`
- `minipamayo_qwen35.stage1.expert_cfm.inference`
- `minipamayo_qwen35.stage2.reasoning_sft.train`
- `minipamayo_qwen35.stage3.post_training.train`

Stage 1 experiments live under the same stage package:

- `minipamayo_qwen35.stage1.vlm_ce.train.experiments.steer_only`
- `minipamayo_qwen35.stage1.vlm_ce.eval.experiments.steer_only`

Recorded-entrypoint examples:

- `uv run python -m minipamayo_qwen35.stage1.data --config-json configs/stage1/data/ignore_rule_data.json`
- `uv run python -m minipamayo_qwen35.stage1.vlm_ce.train.profile --config-json configs/stage1/vlm_ce/profile/canonical/ignore_rule_data_k64_dt01_smoke_12gb_forward_only.json`
- `uv run python -m minipamayo_qwen35.stage1.vlm_ce.train --config-json configs/stage1/vlm_ce/train/canonical/ignore_rule_data_k64_dt01_12gb.json`
- `uv run python -m minipamayo_qwen35.stage1.vlm_ce.eval --config-json configs/stage1/vlm_ce/eval/canonical/ignore_rule_data_k64_dt01_12gb.json`
- `uv run python -m minipamayo_qwen35.stage1.vlm_ce.inference --config-json configs/stage1/vlm_ce/inference/canonical/ignore_rule_data_k64_dt01_sample.json`
- `uv run python -m minipamayo_qwen35.stage1.expert_cfm.train --config-json configs/stage1/expert_cfm/train/canonical/ignore_rule_data_k64_dt01_smoke_12gb.json`
- `uv run python -m minipamayo_qwen35.stage1.vlm_ce.train.experiments.steer_only --config-json configs/stage1/vlm_ce/train/experiments/steer_only/ignore_rule_data_k64_dt01_smoke_12gb.json`

The data, train, eval, profile, and inference entrypoints intentionally reject CLI overrides so the JSON files remain the full run record.

Stage 1 train configs now support:

- `train_jsonl` as one or more required training split JSONLs and `val_jsonl` as zero or more optional validation split JSONLs
  Each value may be either a single path string or a JSON array of path strings.
- `resume_from_checkpoint` for epoch-level resume from `last.pt`
- `early_stopping_patience` and `early_stopping_min_delta`
- fixed Alpamayo-style image budget: `image_min_pixels=163840`, `image_max_pixels=196608`
  Canonical paths reject any other values so unlimited image tokens never slip in by accident.

Stage 1 eval configs use `test_jsonl` explicitly so the evaluation split is named separately from train/val.

Canonical `Stage2 / reasoning_sft` expects the same observation contract as `Stage1A` plus `reasoning_text`.

- required additions over Stage1: `reasoning_text`
- required observation fields carried over: `ego_history_xyz`, `ego_history_rot`
- contract note: [stage2-reasoning-sft-dataset-contract.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/stage2-reasoning-sft-dataset-contract.md)

Stage 1 train outputs record:

- `run_config.json` with resolved args plus git commit, dataset fingerprint, GPU info, and processor settings
- `summary.json` with the same run metadata plus final metrics
- `last.pt`, `best.pt`, and `final.pt` for resume and comparison

Canonical `Stage1A` predicts interleaved acceleration plus curvature tokens with `k=64`, `dt=0.1`, Alpamayo-style image budget, and history-conditioned prompts. Canonical `Stage1B` consumes the frozen `Stage1A` prompt KV-cache and trains a continuous action decoder with CFM. The `steer_only` entrypoints remain experiment-only and record a train-corpus-derived `kappa_range` in `run_config.json`, `summary.json`, and checkpoint metadata.
