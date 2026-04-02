# Stage2 Structure Cleanup

このメモは、`src/minipamayo_qwen35/stage2` の構成整理で確定した方針と、実際に反映した内容をまとめたものです。

前提:
- 後方互換は維持しない。
- `contract/`, `action_space/`, `models/`, `diffusion/`, `helper.py`, `config.py` は top-level shared core として維持する。
- `stage2` 配下には Stage 2 固有の train / eval / inference / preprocess と、その stage 固有 glue だけを残す。

## 反映後のレイアウト

```text
stage2/
  README.md
  reasoning_sft/
    __init__.py
    common.py
    dataset.py
    wrapper.py
    preprocess/
      __init__.py
      __main__.py
      build_jsonl.py
    train/
      __init__.py
      __main__.py
      canonical.py
      runner.py
    eval/
      __init__.py
      __main__.py
      canonical.py
      runner.py
    inference/
      __init__.py
      __main__.py
      canonical.py
      runner.py
```

## 反映した整理

### 1. `data/` を解体した

旧:
- `reasoning_sft/data/build_jsonl.py`
- `reasoning_sft/data/dataset.py`
- `reasoning_sft/data/__main__.py`

新:
- `reasoning_sft/preprocess/build_jsonl.py`
- `reasoning_sft/preprocess/__main__.py`
- `reasoning_sft/dataset.py`

理由:
- preprocess と runtime dataset を分けたかったため
- `stage1` と同じ流儀に揃えるため

### 2. `prompt.py` を削除した

旧:
- `reasoning_sft/prompt.py`
  - `contract.prompt.build_reasoning_prompt_text(...)` の thin wrapper

新:
- call site が `contract.prompt.build_reasoning_prompt_text(...)` を直接使う

理由:
- Stage 2 固有の prompt 契約を持っていなかったため
- contract の正本が `contract.prompt` なので二重化を避けたかったため

### 3. shared logic を `common.py` に寄せた

`reasoning_sft/common.py` に寄せたもの:
- `prepare_stage2_batch(...)`
- `compute_weighted_loss(...)`
- `compute_token_metrics(...)`
- `evaluate(...)`
- `generate_reasoning_handoff(...)`
- `evaluate_handoff_probe(...)`
- `build_handoff_probe_dataset(...)`

結果:
- `train/runner.py -> inference/runner.py` の依存を削除
- `eval/runner.py -> train/runner.py` の依存を削除

### 4. wrapper assembly を `wrapper.py` に寄せた

`reasoning_sft/wrapper.py` に寄せたもの:
- `AlpamayoR1Config` builder
- Stage1A checkpoint + Stage1B checkpoint から wrapper を組む helper
- token contract patching
- Stage 2 inference bundle loader

結果:
- `inference/runner.py` は sample load / input assembly / output payload に集中
- wrapper build は runner 固有ではなくなった

### 5. synthetic reasoning experiment を `stage2` から削除した

削除したもの:
- `train/experiments/synthetic_reasoning*`
- `eval/experiments/synthetic_reasoning*`
- `configs/stage2/reasoning_sft/experiments/synthetic_reasoning/*`

理由:
- canonical Stage 2 reasoning SFT の package を薄く保つため
- 中身が Stage 3 / shared reasoning experiment 寄りで、Stage 2 本流ではなかったため

## 確定した方針

### 1. runtime dataset は `dataset.py` に置く

`ReasoningSftJsonlDataset` と `reasoning_sft_collate` は Stage 2 canonical path の runtime contract なので、`reasoning_sft/dataset.py` に置く。

### 2. preprocess は `preprocess/` にまとめる

`samples_reasoning_sft.jsonl` 生成は学習ループとは責務が違うので、`reasoning_sft/preprocess/` にまとめる。

### 3. runner 間 import は増やさない

shared logic は `common.py` または `wrapper.py` に置き、`train`, `eval`, `inference` の runner 同士は直接 import しない。

### 4. wrapper build は Stage 2 shared helper に残す

`AlpamayoR1` の組み立ては Stage 2 固有の glue なので、top-level に出さず `stage2/reasoning_sft/wrapper.py` に残す。

### 5. synthetic reasoning experiment は canonical Stage 2 に戻さない

synthetic reasoning は shared `reasoning/` か `stage3` 側で扱い、canonical Stage 2 package には戻さない。

## 確認

通したもの:
- `python -m py_compile`
- `uv run python -m minipamayo_qwen35.stage2.reasoning_sft.preprocess --help`
- `uv run python -m minipamayo_qwen35.stage2.reasoning_sft.train --help`
- `uv run python -m minipamayo_qwen35.stage2.reasoning_sft.eval --help`
- `uv run python -m minipamayo_qwen35.stage2.reasoning_sft.inference --help`
- `uv run python -m minipamayo_qwen35.stage3.post_training.train --help`

確認したこと:
- `stage2.reasoning_sft.data` 参照が消えている
- `reasoning_sft.prompt` 参照が消えている
- `..train.runner.evaluate` と `..inference.runner.generate_reasoning_handoff` のような runner 間 import が消えている

## 完了条件

- `reasoning_sft/data/` という mixed-responsibility package が無い
- `prompt.py` が無い
- train / eval / inference の shared logic が `common.py` に寄っている
- wrapper build が `wrapper.py` にまとまっている
- synthetic reasoning experiment が `stage2` canonical path から外れている
