# Stage1 Shared Boundary Plan

状態:
- 2026-03-30 時点で、この plan はコード反映済み。

このメモは、`src/minipamayo_qwen35/stage1/vlm_ce` と
`src/minipamayo_qwen35/stage1/expert_cfm` のあいだで共通化できるものを、
`src/minipamayo_qwen35/stage1/` 直下に寄せるための整理である。

今回のスコープ:
- 対象は `vlm_ce` と `expert_cfm` の共有境界だけ。
- `stage2/` や `stage3/` まで広げた理想形ではなく、
  まず `Stage1A front-end` と `Stage1B front-end` のあいだの依存を正すことが目的。
- Alpamayo mirror file は触らない。

## いま問題になっている依存

現状、`expert_cfm` はまだ `vlm_ce` package を shared library として見ている箇所がある。

もっとも明確なのはこれ:
- `stage1/expert_cfm/train.py`
  - `from ..vlm_ce.train import format_gib, log_gpu_preflight, maybe_wandb_finish, maybe_wandb_log, metric_improved, release_cuda_memory, set_seed, write_run_config`

さらに、
- `stage1/stage1a_conditioning.py`
  - `stage1.vlm_ce.prompting`
  - `stage1.vlm_ce.runtime`
  に依存している

つまり今は、
- `vlm_ce`
  - entrypoint package
- `expert_cfm`
  - 別 entrypoint package

であるはずなのに、`vlm_ce` の中身が `expert_cfm` から shared source として見えている。

これをやめて、
- `stage1/` 直下
  - Stage1 shared source
- `vlm_ce/`
  - Stage1A front-end
- `expert_cfm/`
  - Stage1B front-end

という構造に揃える。

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

## `stage1/` 直下へ出すべきもの

### 1. `stage1_train_runtime.py`

現在の問題:
- `expert_cfm/train.py` が `vlm_ce.train` から train helper を import している

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

補足:
- 将来的には `utils/` に上げてもよいが、まずは `stage1/` 直下に切り出して
  `vlm_ce` package 依存を消すのが先

### 2. `stage1_train_data.py`

現在の問題:
- `vlm_ce/train.py` と `expert_cfm/train.py` がそれぞれ `build_dataloaders(...)` を持っている

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

現在の問題:
- `vlm_ce/cli.py`
- `expert_cfm/cli.py`

がどちらも
- `load_json_payload`
- `resolve_path_base`
- `normalize_arg_config`
- `--config-json only`

の流れを別実装で持っている

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

現在の置き場:
- `stage1/vlm_ce/components.py`

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

現在の置き場:
- `stage1/vlm_ce/prompting.py`

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

現在の置き場:
- `stage1/vlm_ce/runtime.py`

`stage1/` へ出す対象:
- `Stage1ARuntime`
- `load_stage1a_runtime`
- `prepare_stage1a_prompt_inputs`
- `prepare_stage1a_training_batch`
- `run_stage1a_teacher_forced_batch`
- `decode_stage1a_generated_batch`
- `run_stage1a_rollout_batch`
- `extract_prompt_cache`

理由:
- これは `vlm_ce` runner の loop ではなく、Stage1A runtime contract
- `stage1a_conditioning.py` がここに依存している以上、
  `stage1/` shared source にする方が境界が自然

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
- `vlm_ce/generation.py`
  - Stage1A discrete action-token generation
- `vlm_ce/metrics.py`
  - token accuracy / inference payload 向け helper
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

## リファクタ後の目標像

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
