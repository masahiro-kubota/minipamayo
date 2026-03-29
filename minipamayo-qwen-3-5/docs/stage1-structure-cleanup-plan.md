# Stage1 Structure Cleanup Plan

状態:
- 2026-03-30 時点で、この cleanup は反映済み。

このメモは、`src/minipamayo_qwen35/stage1` のディレクトリ構成を整理するための方針をまとめたものです。

前提:
- `contract/`, `action_space/`, `diffusion/`, `models/`, `geometry/` は top-level shared core として維持する。
- `stage1` 配下には、Stage 1 の train / eval / inference と、その stage 固有 glue だけを残す。
- Alpamayo との pure core 差分は増やさない。

## 現状

`stage1` の大枠は以下の 3 つに分かれている。

- `data/`
- `vlm_ce/`
- `expert_cfm/`

このうち、`vlm_ce/` は比較的整理されている。

- `train/`
- `eval/`
- `inference/`
- `experiments/`

一方で、以下の 2 箇所はまだ責務が混ざっている。

### 1. `stage1/data/` -> `stage1/preprocess`, `stage1/dataset.py`, `stage1/validate.py`

いまの `data/` には以下が同居している。

- `extract.py`
  - 前処理 / JSONL 抽出
- `dataset.py`
  - train / eval 用 dataset reader
- `validate.py`
  - saved action と canonical action の検証

問題:
- `data` という名前では責務が広すぎる
- `extract` と `dataset` と `validate` は性質が違う
- `__init__.py` が `main` まで export していて package surface が雑

### 2. `stage1/expert_cfm/core/` -> `stage1/expert_cfm/common.py`

いまの `core/` は実質 compatibility wrapper である。

- `model.py`
  - top-level `models/action_expert.py` の再 export
- `diffusion.py`
  - top-level `diffusion/action_expert.py` の再 export
- `common.py`
  - Stage1B 固有の prompt-cache / metadata helper

問題:
- `model.py` と `diffusion.py` は pure core ではない
- 実際の runner はすでに top-level `models` / `diffusion` を直接 import している
- `core/` という名前に対して中身が薄すぎる

## 確定方針

### 1. `stage1/data` は責務ごとに分割する

目標レイアウト:

```text
stage1/
  preprocess/
    extract.py
  dataset.py
  validate.py
  vlm_ce/
  expert_cfm/
```

反映:
- `extract.py` は `stage1/preprocess/` に移す
- `dataset.py` は `stage1/dataset.py` に寄せる
- `validate.py` は `stage1/validate.py` に寄せる
- `stage1/data/` package 自体を消す

理由:
- 前処理と学習時 reader を分けた方が責務が明確
- `data` package の曖昧さを減らせる
- CLI entrypoint も `preprocess` に寄せた方が自然

### 2. `stage1/expert_cfm/core/` は薄くする

反映:
- `stage1/expert_cfm/core/model.py` は削除した
- `stage1/expert_cfm/core/diffusion.py` は削除した
- 必要な import は top-level `models.action_expert` / `diffusion.action_expert` に直した
- `stage1/expert_cfm/core/common.py` は `stage1/expert_cfm/common.py` に移した

理由:
- `common.py` は Stage1B 固有 helper として意味がある
- `model.py` / `diffusion.py` は compatibility layer に過ぎず、構造を濁す
- pure core はすでに top-level に出ている

目標レイアウト:

```text
stage1/
  expert_cfm/
    common.py
    train/
    eval/
    inference/
```

### 3. `vlm_ce/` は現状維持でよい

方針:
- `train/`, `eval/`, `inference/` は維持
- `experiments/steer_only.py` は experimental entrypoint として残してよい
- `profile.py` も train 補助として現状維持でよい

理由:
- ここはすでに entrypoint 単位で整理されている
- 先に触るべきなのは `data` と `expert_cfm/core`

## 実装順

1. `stage1/data` を分割する
2. import をすべて張り替える
3. `stage1/expert_cfm/core/model.py` / `diffusion.py` を削除する
4. `stage1/expert_cfm/common.py` に runner import を揃える
5. `py_compile` と `--help` で確認する

## 確認項目

- `python -m py_compile ...`
- `uv run python -m minipamayo_qwen35.stage1.preprocess --help`
- `uv run python -m minipamayo_qwen35.stage1.validate --help`
- `uv run python -m minipamayo_qwen35.stage1.vlm_ce.train --help`
- `uv run python -m minipamayo_qwen35.stage1.expert_cfm.train --help`
- `uv run python -m minipamayo_qwen35.stage1.expert_cfm.eval --help`
- `uv run python -m minipamayo_qwen35.stage1.expert_cfm.inference --help`

## 完了条件

- `stage1/data/` という曖昧な package が無い
- `stage1/expert_cfm/core/model.py` / `diffusion.py` が無い
- `stage1` 配下には Stage1 固有の glue だけが残る
- top-level shared core との責務境界が明確になる
