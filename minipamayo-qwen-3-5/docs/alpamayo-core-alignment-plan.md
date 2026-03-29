# Alpamayo Core Alignment Plan

このメモは、`minipamayo-qwen-3-5` を `/home/masa/minipamayo/related_repos/alpamayo` に段階的に寄せるための実装順を整理したものです。

前提:
- 目的は「今の repo 都合の abstraction を守ること」ではなく、「Alpamayo 公開実装との差分を減らすこと」。
- ただし一気に end-to-end wrapper を持ち込むのではなく、破壊半径の小さい pure core から寄せる。
- `record_adapter.py` のような repo 固有 glue は残してよいが、pure core に混ぜない。

## 目標

最終的に top-level に以下の shared core を揃える。

- `action_space/`
- `diffusion/`
- `models/`
- `geometry/`

各 stage はそれらを import するだけに寄せる。

## 現状の diff 状況

2026-03-29 時点で、以下の `diff -ru` を使って Alpamayo 側と比較した。

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
- `diffusion/flow_matching.py`

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

### 4. package file の差分

以下は Alpamayo 側では package export を持っていないか、export 内容が違う。

- `action_space/__init__.py`
- `diffusion/__init__.py`
- `models/__init__.py`
- `geometry/__init__.py`

この差分は pure core の numerics 差ではなく、`minipamayo-qwen-3-5` 側の import 整理のための差分。

### 5. まだ未移植の大物

残っている本質的な未移植はこれ。

- Alpamayo `models/alpamayo_r1.py` 相当

これは end-to-end wrapper なので、最後に触る。

## 残りの実装方針

### 1. package file の扱いを決める

対象:
- `action_space/__init__.py`
- `diffusion/__init__.py`
- `models/__init__.py`
- `geometry/__init__.py`

やること:
- Alpamayo mirror を優先するなら、export を減らす
- repo 内 import 利便性を優先するなら、この差分は許容する

注意:
- ここはロジック差ではなく package 境界の差なので、優先度は低い

### 2. repo 固有 core の位置づけを固定する

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

### 3. 最後に end-to-end wrapper を考える

対象:
- Alpamayo `models/alpamayo_r1.py`

ここでの狙い:
- Alpamayo の end-to-end 構造に近づける

注意:
- 一番破壊半径が大きい
- `stage1`, `stage2`, `stage3` をまたぐので最後

## 優先順位

残り:
- `models/alpamayo_r1.py` 相当
- package file 差分を詰めるかどうかの判断
- repo 固有 core (`record_adapter.py`, `diffusion/action_expert.py`, `models/action_expert.py`) の最終位置づけ

## 各段階の確認

各段階で最低限確認する。

- `py_compile`
- `stage1.vlm_ce.train --help`
- `stage1.expert_cfm.train --help`
- `stage2.reasoning_sft.train --help`
- 必要なら 1 epoch smoke

## 完了条件

- pure shared core は top-level `action_space/`, `diffusion/`, `models/`, `geometry/` にある
- stage 配下には train/eval/inference と stage 固有 glue だけが残る
- Alpamayo の同名 file と diff を見たとき、差分は import path と repo 固有 glue に限定される
