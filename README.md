# MiniPamayo

MiniPamayo is a reimplementation of [Alpamayo](https://github.com/NVlabs/alpamayo) built on the Qwen3.5-0.8B VLM.  
This repository collects its training experiments, data specifications, and shared checkpoints.  
The main implementation lives under [`minipamayo-qwen-3-5/`](./minipamayo-qwen-3-5/), which contains the Qwen3.5-0.8B-based multi-stage training pipeline.

## Overview

This repository includes:

- MiniPamayo training code built around Qwen3.5
- Configurations and experiment entrypoints for Stage 1 through Stage 3
- Dataset layout conventions and telemetry specifications
- Shared checkpoints and a lightweight inspection GUI

If you are new to this repo, the fastest starting point is [`minipamayo-qwen-3-5/README.md`](./minipamayo-qwen-3-5/README.md).

## Repository Layout

```text
.
├── minipamayo-qwen-3-5/   # Main implementation and training configs
├── shared_checkpoints/    # Shared checkpoints and lightweight GUI
├── scripts/               # Root-level helper scripts
├── mcap_spec.md           # MCAP telemetry specification
├── ruff.toml              # Ruff configuration
└── .pre-commit-config.yaml
```

### Main Project

The actual project code lives in `minipamayo-qwen-3-5/`.

| Path | Role |
| --- | --- |
| [`minipamayo-qwen-3-5/src/minipamayo_qwen35/`](./minipamayo-qwen-3-5/src/minipamayo_qwen35/) | Stage-specific Python implementation |
| [`minipamayo-qwen-3-5/configs/`](./minipamayo-qwen-3-5/configs/) | JSON configs for preprocessing, training, evaluation, and inference |
| [`minipamayo-qwen-3-5/datasets/`](./minipamayo-qwen-3-5/datasets/) | Dataset layout rules and expected local structure |
| [`minipamayo-qwen-3-5/docs/`](./minipamayo-qwen-3-5/docs/) | Design docs, experiment records, and notes |
| [`minipamayo-qwen-3-5/tests/`](./minipamayo-qwen-3-5/tests/) | Smoke tests, contract tests, and artifact tests |
| [`minipamayo-qwen-3-5/env/`](./minipamayo-qwen-3-5/env/) | Repo-local environment setup such as CUDA helpers |

## Training Stages

The implementation is organized by stage:

- `stage1.preprocess`
  Convert raw episodes into training-ready `jsonl + images`
- `stage1.vlm_ce`
  Train the VLM side with discrete trajectory tokens
- `stage1.expert_cfm`
  Train the continuous action decoder
- `stage2.reasoning_sft`
  Supervised fine-tuning with reasoning annotations
- `stage3.post_training`
  RL-based post-training and alignment

This repo is designed around JSON configs as the primary run record rather than heavy CLI overrides.  
As a result, many entrypoints accept `--config-json`.

## Quick Start

### Requirements

- Python 3.12+
- `uv`
- CUDA toolkit 12.8

Canonical runs assume CUDA 12.8.

### Setup

```bash
cd /home/masa/minipamayo/minipamayo-qwen-3-5
. ./env/cuda-12.8.sh
uv sync --locked
```

### Dataset Layout

Raw data is expected to follow a layout like this:

```text
minipamayo-qwen-3-5/datasets/
  raw/<collection_name>/<episode_id>/
  processed/stage1/<dataset_name>/
  splits/stage1/
```

See [`minipamayo-qwen-3-5/datasets/README.md`](./minipamayo-qwen-3-5/datasets/README.md) for details.

### Example Workflow

```bash
uv run python -m minipamayo_qwen35.stage1.preprocess \
  --config-json configs/stage1/data/ignore_rule_data.json

uv run python -m minipamayo_qwen35.stage1.vlm_ce.train.profile \
  --config-json configs/stage1/vlm_ce/profile/canonical/ignore_rule_data_k64_dt01_smoke_12gb_forward_only.json

uv run python -m minipamayo_qwen35.stage1.vlm_ce.train \
  --config-json configs/stage1/vlm_ce/train/canonical/ignore_rule_data_k64_dt01_12gb.json
```

From there, continue with `eval`, `inference`, `stage2`, and `stage3` as needed.


