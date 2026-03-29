# Alpamayo Core Alignment Plan

このメモは、`minipamayo-qwen-3-5` を `/home/masa/minipamayo/related_repos/alpamayo` に段階的に寄せるための実装順を整理したものです。

前提:
- 目的は「今の repo 都合の abstraction を守ること」ではなく、「Alpamayo 公開実装との差分を減らすこと」。
- `config.py` / `models/alpamayo_r1.py` はすでに持ち込んでおり、`stage2` inference は wrapper 経由に寄せた。
- `record_adapter.py` のような repo 固有 glue は残してよいが、pure core に混ぜない。

## 目標

最終的に top-level に以下の shared core を揃える。

- `action_space/`
- `diffusion/`
- `models/`
- `geometry/`

各 stage はそれらを import するだけに寄せる。

## 現状の diff 状況

2026-03-30 時点で、以下の `diff -ru` を使って Alpamayo 側と比較した。

```bash
diff -ru \
  /home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/action_space \
  /home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/action_space

diff -ru \
  /home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/diffusion \
  /home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/diffusion

diff -ru \
  /home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/models \
  /home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/models

diff -ru \
  /home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/geometry \
  /home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/geometry
```

### 1. 差分なし

以下は Alpamayo 側と一致している。

- `action_space/action_space.py`
- `diffusion/base.py`
- `geometry/rotation.py`
- `models/action_in_proj.py`
- `models/base_model.py`
- `models/delta_tokenizer.py`
- `models/token_utils.py`

### 2. import path 差だけ

以下は実質 import path 差だけで、ロジック差は残っていない。

- `action_space/discrete_action_space.py`
- `action_space/unicycle_accel_curvature.py`
- `action_space/utils.py`
- `config.py`
- `diffusion/flow_matching.py`
- `models/alpamayo_r1.py`

### 3. repo 固有として残している差分

以下は Alpamayo には無いが、`minipamayo-qwen-3-5` 側の stage 分割や JSONL record 契約のために残している。

- `action_space/record_adapter.py`
  - record から tensor を読む
  - shape adapter
  - canonical action / waypoint payload の橋渡し
- `diffusion/action_expert.py`
  - `Stage1B` の expert decoding に必要な diffusion adapter
- `models/action_expert.py`
  - `Stage1B` 用 shared action expert 本体

### 4. まだ未移植の大物

大物の未移植は解消済み。

- `config.py`
- `models/alpamayo_r1.py`

はどちらも repo に入り、`stage2/reasoning_sft/inference/runner.py` は manual handoff ではなく wrapper を組み立てて呼ぶ形に変わった。

## 残りの実装方針

### 1. repo 固有 core の位置づけを固定する

対象:
- `action_space/record_adapter.py`
- `diffusion/action_expert.py`
- `models/action_expert.py`

やること:
- これらを「Alpamayo に無い repo 固有 core」と明示的に扱う
- pure Alpamayo mirror と混ぜない

注意:
- `record_adapter.py` は Alpamayo loader 不在を埋める層
- `action_expert.py` は end-to-end wrapper 未導入の間の shared 実装

### 2. wrapper builder をどこに残すか決める

対象:
- `stage2/reasoning_sft/inference/runner.py` の wrapper 組み立て部分

注意:
- いまは Stage1A / Stage1B checkpoint 契約を読んで `AlpamayoR1` を構成する repo 固有 builder が runner に残っている
- 再利用が必要になった時点で shared helper へ出す

## 優先順位

残り:
- repo 固有 core (`record_adapter.py`, `diffusion/action_expert.py`, `models/action_expert.py`) の最終位置づけ
- wrapper builder の最終配置
- GPU が空いた時の wrapper inference smoke

## 各段階の確認

各段階で最低限確認する。

- `py_compile`
- `stage1.vlm_ce.train --help`
- `stage1.expert_cfm.train --help`
- `stage2.reasoning_sft.inference --help`
- 必要なら single-sample inference smoke

## 完了条件

- pure shared core は top-level `action_space/`, `diffusion/`, `models/`, `geometry/` にある
- stage 配下には train/eval/inference と stage 固有 glue だけが残る
- Alpamayo の同名 file と diff を見たとき、差分は import path と repo 固有 glue に限定される
- `stage2` inference の handoff は wrapper 経由で実行される
