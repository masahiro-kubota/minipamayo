## 背景

`stage3/post_training` は package 構成としては整理されているが、まだ `stage1` / `stage2` の内部 helper に直接依存している。

現状の主な直接依存は以下。

- [bundle.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/rollout/bundle.py)
  - `stage1.vlm_ce.train.load_checkpoint`
  - `stage2.reasoning_sft.wrapper.load_stage2_inference_bundle`
- [sampler.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/rollout/sampler.py)
  - `stage1.vlm_ce.train.append_token_to_model_inputs`
  - `stage1.vlm_ce.train.model_forward_inputs`
  - `stage1.vlm_ce.train.prepare_prompt_inputs_with_history`
- [train/runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/train/runner.py)
  - `stage1.vlm_ce.train.format_gib`
  - `stage1.vlm_ce.train.log_gpu_preflight`
  - `stage1.vlm_ce.train.maybe_wandb_finish`
  - `stage1.vlm_ce.train.maybe_wandb_log`
  - `stage1.vlm_ce.train.set_seed`
  - `stage1.vlm_ce.train.write_run_config`
- [eval/runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/eval/runner.py)
  - `stage1.vlm_ce.train.format_gib`
  - `stage1.vlm_ce.train.load_checkpoint`
  - `stage1.vlm_ce.train.set_seed`
- [dataset.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/dataset.py)
  - `stage1.dataset.read_jsonl`
  - `stage2.reasoning_sft.dataset.ReasoningSftJsonlDataset`
  - `stage2.reasoning_sft.dataset.reasoning_sft_collate`
- [build_manifest.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/curation/build_manifest.py)
  - `stage1.dataset.read_jsonl`

`stage3` は今すぐ使う予定がないため、ここではコード変更は行わず、後でまとめて直すための方針だけを固定する。

後方互換は持たない。

## 目的

- `stage3/post_training` から `stage1` / `stage2` の内部 module への直接 import を減らす
- shared な契約は `minipamayo_qwen35` top-level 配下に寄せる
- `stage3` が `stage1` / `stage2` の entrypoint helper にぶら下がらない構成にする

## 修正予定

### 1. JSONL IO を shared utility に出す

新設:

- `src/minipamayo_qwen35/utils/jsonl.py`

入れるもの:

- `normalize_jsonl_paths(...)`
- `read_jsonl(...)`

置き換え先:

- [stage1/dataset.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/dataset.py)
- [stage2/reasoning_sft/dataset.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage2/reasoning_sft/dataset.py)
- [stage3/post_training/dataset.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/dataset.py)
- [stage3/post_training/curation/build_manifest.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/curation/build_manifest.py)
- [reasoning/synthetic_dataset.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/reasoning/synthetic_dataset.py)

狙い:

- `stage3` の `stage1.dataset.read_jsonl` 依存を消す
- JSONL path 正規化を stage 固有コードから切り離す

### 2. Reasoning SFT dataset を shared 層へ移す

新設:

- `src/minipamayo_qwen35/reasoning/dataset.py`

移すもの:

- `ReasoningSftJsonlDataset`
- `reasoning_sft_collate`

現行ソース:

- [stage2/reasoning_sft/dataset.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage2/reasoning_sft/dataset.py)

置き換え先:

- `stage2/reasoning_sft/train/eval/inference/common`
- `stage3/post_training/dataset.py`

狙い:

- `stage3` の `stage2.reasoning_sft.dataset` 依存を消す
- Stage2/Stage3 が共有している dataset 契約を top-level `reasoning/` に寄せる

### 3. checkpoint loader を shared utility に出す

新設:

- `src/minipamayo_qwen35/utils/checkpoint.py`

入れるもの:

- `load_checkpoint(...)`

現行ソース:

- [stage1/vlm_ce/train/runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/vlm_ce/train/runner.py)

置き換え先:

- [stage1/vlm_ce/train/runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/vlm_ce/train/runner.py)
- [stage2/reasoning_sft/wrapper.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage2/reasoning_sft/wrapper.py)
- [stage3/post_training/rollout/bundle.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/rollout/bundle.py)
- [stage3/post_training/eval/runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/eval/runner.py)

狙い:

- `stage3` の `stage1.vlm_ce.train.load_checkpoint` 依存を消す

### 4. training/runtime helper を shared utility に出す

新設:

- `src/minipamayo_qwen35/utils/train_runtime.py`

入れるもの:

- `set_seed(...)`
- `format_gib(...)`
- `log_gpu_preflight(...)`
- `write_run_config(...)`
- `maybe_wandb_log(...)`
- `maybe_wandb_finish(...)`

現行ソース:

- [stage1/vlm_ce/train/runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/vlm_ce/train/runner.py)

置き換え先:

- [stage2/reasoning_sft/train/runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage2/reasoning_sft/train/runner.py)
- [stage3/post_training/train/runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/train/runner.py)
- [stage3/post_training/eval/runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/eval/runner.py)

狙い:

- `stage3` が `stage1` trainer の helper 集合に依存しないようにする
- runtime helper の正本を shared utility にする

### 5. prompt input helper を shared module に出す

新設:

- `src/minipamayo_qwen35/models/prompt_inputs.py`

入れるもの:

- `move_inputs_to_device(...)`
- `model_forward_inputs(...)`
- `inject_history_token_ids(...)`
- `inject_history_inputs_embeds(...)`
- `append_token_to_model_inputs(...)`
- `prepare_prompt_inputs_with_history(...)`

現行ソース:

- [stage1/vlm_ce/train/runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/vlm_ce/train/runner.py)

置き換え先:

- [stage1/vlm_ce/train/runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/vlm_ce/train/runner.py)
- [stage1/vlm_ce/eval/runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/vlm_ce/eval/runner.py)
- [stage3/post_training/rollout/sampler.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/rollout/sampler.py)

狙い:

- `stage3` の generation/log-prob 周りが `stage1` trainer helper に依存しないようにする

### 6. wrapper bundle builder を top-level へ出す

新設候補:

- `src/minipamayo_qwen35/models/alpamayo_bundle.py`

移すもの:

- `load_stage2_inference_bundle(...)`
- それに付随する private helper

現行ソース:

- [stage2/reasoning_sft/wrapper.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage2/reasoning_sft/wrapper.py)

置き換え先:

- [stage2/reasoning_sft/inference/runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage2/reasoning_sft/inference/runner.py)
- [stage3/post_training/rollout/bundle.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/rollout/bundle.py)
- [test_inference.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/test_inference.py)

狙い:

- `stage3` の `stage2.reasoning_sft.wrapper` 依存を消す
- wrapper assembly を `stage2` 専用ではなく shared top-level model helper にする

補足:

- この top-level builder 自体は内部で `stage1.vlm_ce.eval.load_components(...)` を呼んでもよい
- ただし `stage3` から見える import 先は top-level に固定する
- その後必要なら `load_components(...)` も別途 shared 化する

### 7. Stage3 motion decode の shared core 利用は別タスクとして扱う

現状:

- [stage3/post_training/rollout/sampler.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage3/post_training/rollout/sampler.py)
  は `bundle.wrapper.traj_tokenizer.decode(...)` を使っている

問題:

- Stage3 が `stage1/stage1b_expert.py` / `stage1/stage1b_diffusion.py` を十分活用していない

扱い:

- これは「shared dependency cleanup」とは別のタスク
- まずは import 境界の整理を終えてから着手する

## 実装順

1. `utils/jsonl.py`
2. `reasoning/dataset.py`
3. `utils/checkpoint.py`
4. `utils/train_runtime.py`
5. `models/prompt_inputs.py`
6. `models/alpamayo_bundle.py`
7. Stage3 import 張り替え
8. Stage2 / Stage1 import 張り替え
9. 不要になった stage-specific re-export / wrapper を削除

## 確認項目

- `rg -n "stage1|stage2" src/minipamayo_qwen35/stage3/post_training`
  - ここで残る stage-specific import を最小化する
- `uv run python -m py_compile $(find src/minipamayo_qwen35 -name '*.py' | sort)`
- `uv run python -m minipamayo_qwen35.stage3.post_training.train --help`
- `uv run python -m minipamayo_qwen35.stage3.post_training.eval --help`
- `uv run python -m minipamayo_qwen35.stage2.reasoning_sft.inference --help`

## 期待する最終状態

- `stage3/post_training` は top-level shared layer を主に使う
- `stage3` から `stage1` / `stage2` の runner/helper への直接 import は無いか、bundle assembly の奥に隠れる
- `stage1`, `stage2`, `stage3` の共有契約は
  - `action_space/`
  - `contract/`
  - `diffusion/`
  - `models/`
  - `reasoning/`
  - `utils/`
  に集約される
