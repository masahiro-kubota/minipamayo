# Stage1 Shared Boundary Plan

状態:
- 2026-03-30 時点で、この方針はコード反映済み。
- 以下は「何を `stage1/` 配下の shared source に置き、何を `vlm_ce/` / `expert_cfm/` の front-end に残すか」の現状整理である。

このメモは、`src/minipamayo_qwen35/stage1/vlm_ce` と
`src/minipamayo_qwen35/stage1/expert_cfm` のあいだで共通化できるものを、
`src/minipamayo_qwen35/stage1/` 直下に寄せるための整理である。

今回のスコープ:
- 対象は `vlm_ce` と `expert_cfm` の共有境界だけ。
- `stage2/` や `stage3/` まで広げた理想形ではなく、
  まず `Stage1A front-end` と `Stage1B front-end` のあいだの依存を正すことが目的。
- Alpamayo mirror file は触らない。

## 解消した依存

この整理で、以下の依存は解消した。

- `stage1/expert_cfm/train.py`
  - 以前は `vlm_ce.train` から train helper を import していた
  - いまは `stage1_train_runtime.py` と `stage1_train_data.py` を使う
- `stage1/stage1a_conditioning.py`
  - 以前は `vlm_ce.prompting` / `vlm_ce.runtime` に依存していた
  - いまは `stage1a_prompting.py` / `stage1a_runtime.py` に依存する
- `stage2/` / `stage3/`
  - 以前は `stage1.vlm_ce.components/prompting/runtime` を shared source として見ていた
  - いまは `stage1a_components.py` / `stage1a_prompting.py` / `stage1_train_runtime.py` を参照する

つまり現在の境界は、
- `stage1/`
  - Stage1 shared source
- `stage1/vlm_ce/`
  - Stage1A front-end
- `stage1/expert_cfm/`
  - Stage1B front-end

に揃っている。

## すでに `stage1/` 直下で正しいもの

以下はすでに shared boundary として自然で、そのままでよい。

- `stage1/dataset.py`
  - `Stage1JsonlDataset`
  - `stage1_collate`
- `stage1/checkpoint_completion.py`
  - completed checkpoint guard
- `stage1/stage1a_conditioning.py`
  - Stage1A conditioning bridge
- `stage1/stage1b_action_expert.py`
  - Stage1B detached expert
- `stage1/stage1b_diffusion_adapter.py`
  - Stage1B detached diffusion adapter

## `stage1/` 直下に置くもの

### 1. `stage1_train_runtime.py`

`stage1/` へ出す対象:
- `set_seed`
- `release_cuda_memory`
- `format_gib`
- `log_gpu_preflight`
- `write_run_config`
- `maybe_wandb_log`
- `maybe_wandb_finish`
- `metric_improved`
- `best_metric_from_history`

理由:
- 少なくとも `vlm_ce.train` と `expert_cfm.train` の両方が使う
- これらは `Stage1A` 専用でも `Stage1B` 専用でもない
- 今のように `expert_cfm -> vlm_ce.train` で引くのは境界が悪い

### 2. `stage1_train_data.py`

`stage1/` へ出す対象:
- `Stage1JsonlDataset + stage1_collate` を前提にした
  train/val split
  `DataLoader` 構築
  `max_samples`
  `val_fraction`
  `num_workers`
  `pin_memory`
  `persistent_workers`
  の policy

理由:
- 2 系統の train path がほぼ同じ split policy を持っている
- ここに差分が残ると、Stage1A / Stage1B の学習条件ズレになる

### 3. `stage1_json_cli.py`

`stage1/` へ出す対象:
- generic な
  - `load_stage1_config_args(...)`
  - `parse_stage1_json_only_args(...)`

理由:
- JSON config parse の policy は package ではなく Stage1 共通
- いまの重複は引数 schema の差ではなく bootstrap の差

残すもの:
- `expert_cfm/cli.py` の `add_stage1b_common_args(...)`
- `vlm_ce` 側の parser definition

つまり、
- parser の shape は package-local
- JSON bootstrap は `stage1/` shared

に分ける

### 4. `stage1a_components.py`

`stage1/` へ出す対象:
- `resolve_dtype`
- `resolve_checkpoint_kind`
- `resolve_checkpoint_args`
- `resolve_processor_path`
- `resolve_task_spec_from_checkpoint`
- `build_processor_kwargs`
- `build_model_load_kwargs`
- `load_checkpoint`
- `build_training_token_contract`
- `build_stage1_metadata`
- `load_components`

理由:
- これらは `vlm_ce` front-end helper ではなく、Stage1A checkpoint/bootstrap contract
- `stage1a_conditioning.py` が依存している時点で、`vlm_ce` package-local に置くべきではない

### 5. `stage1a_prompting.py`

`stage1/` へ出す対象:
- `move_inputs_to_device`
- `model_forward_inputs`
- `inject_history_token_ids`
- `inject_history_inputs_embeds`
- `append_token_to_model_inputs`
- `prepare_prompt_inputs_with_history`
- `prepare_alpamayo_prompt_inputs_with_history`
- `build_full_inputs_from_prompt_inputs`
- `prepare_batch`

理由:
- これは `vlm_ce` の UI ではなく Stage1A prompt/input contract そのもの
- `stage1a_conditioning.py` が依存しているので、shared source として明示した方がよい

### 6. `stage1a_runtime.py`

`stage1/` へ出す対象:
- `Stage1ARuntime`
- `load_stage1a_runtime`
- `prepare_stage1a_prompt_inputs`
- `prepare_stage1a_training_batch`
- `run_stage1a_teacher_forced_batch`
- `decode_stage1a_generated_batch`
- `run_stage1a_rollout_batch`
- `extract_prompt_cache`
- `compute_token_accuracy`
- `greedy_generate_action_tokens`

理由:
- これは `vlm_ce` runner の loop ではなく、Stage1A runtime contract
- `stage1a_conditioning.py` がここに依存している以上、
  `stage1/` shared source にする方が境界が自然
- token accuracy と discrete token generation も、`train.py` / `eval.py` / `inference.py` を跨いで使う Stage1A runtime の一部なので、最終的に `stage1a_runtime.py` に寄せた

## package-local のまま残すもの

### `vlm_ce` に残す

- `vlm_ce/train.py`
  - Stage1A training loop
- `vlm_ce/eval.py`
  - Stage1A eval loop
- `vlm_ce/inference.py`
  - single-sample payload
- `vlm_ce/profile.py`
  - smoke/profile
- `vlm_ce/metrics.py`
  - `require_record_field`
  - `infer_vision_tokens`
  - つまり eval/inference payload 向け helper だけ
- `vlm_ce/eval_artifacts.py`
  - MCAP / artifact 出力
- `vlm_ce/train_steer_only.py`
- `vlm_ce/eval_steer_only.py`

### `expert_cfm` に残す

- `expert_cfm/train.py`
  - Stage1B training loop
- `expert_cfm/eval.py`
  - Stage1B eval loop
- `expert_cfm/inference.py`
  - Stage1B single-sample front-end
- `expert_cfm/runtime.py`
  - Stage1B inference orchestration
- `expert_cfm/metadata.py`
- `expert_cfm/metrics.py`
- `expert_cfm/pid.py`
- `expert_cfm/cli.py`

## 現在の配置

```text
stage1/
  checkpoint_completion.py
  dataset.py
  stage1_train_runtime.py
  stage1_train_data.py
  stage1_json_cli.py
  stage1a_components.py
  stage1a_prompting.py
  stage1a_runtime.py
  stage1a_conditioning.py
  stage1b_action_expert.py
  stage1b_diffusion_adapter.py
  validate.py
  vlm_ce/
    __init__.py
    cli.py
    eval.py
    eval_artifacts.py
    eval_steer_only.py
    inference.py
    metrics.py
    profile.py
    train.py
    train_steer_only.py
  expert_cfm/
    cli.py
    eval.py
    inference.py
    metadata.py
    metrics.py
    pid.py
    runtime.py
    train.py
```

## 残してよい差分

この整理後も、`vlm_ce` と `expert_cfm` の front-end は完全一致にはしない。

- `vlm_ce`
  - token CE 学習
  - rollout/eval artifact
  - single-sample payload
- `expert_cfm`
  - prompt cache conditioning
  - diffusion expert loss/sample
  - PID override

つまり、
- `stage1/` 直下
  - package 間で再利用する contract / runtime / CLI / train-data policy
- `vlm_ce/` と `expert_cfm/`
  - stage 固有の loop / payload / artifact

という境界を維持する。
    train.py
    eval.py
    eval_artifacts.py
    inference.py
    profile.py
    metrics.py
    generation.py
    train_steer_only.py
    eval_steer_only.py
  expert_cfm/
    __init__.py
    cli.py
    train.py
    eval.py
    inference.py
    runtime.py
    metadata.py
    metrics.py
    pid.py
```

## まずやる順番

1. `expert_cfm/train.py -> vlm_ce.train` の依存を消す
   - `stage1_train_runtime.py` を作る
2. `vlm_ce/train.py` と `expert_cfm/train.py` の dataloader policy を `stage1_train_data.py` に寄せる
3. `stage1a_conditioning.py -> vlm_ce.prompting/runtime` の依存を消す
   - `stage1a_components.py`
   - `stage1a_prompting.py`
   - `stage1a_runtime.py`
   を作る
4. `vlm_ce/cli.py` と `expert_cfm/cli.py` の JSON bootstrap を `stage1_json_cli.py` に寄せる

## 完了条件

- `expert_cfm` が `vlm_ce` package に依存しない
- `stage1a_conditioning.py` が `vlm_ce.*` に依存しない
- `vlm_ce` と `expert_cfm` の共通 source が `stage1/` 直下に見える形で置かれている
- `vlm_ce/` と `expert_cfm/` は front-end と package-local glue に近づく
