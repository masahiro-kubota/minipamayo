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

## 未完の実装順

### 1. `stage1/expert_cfm/core` を薄くする

対象:
- `src/minipamayo_qwen35/stage1/expert_cfm/core/model.py`
- `src/minipamayo_qwen35/stage1/expert_cfm/core/common.py`

やること:
- `diffusion/` と `action_space/` を top-level shared core として参照する
- `stage1/expert_cfm/core` には stage1B 固有 wiring だけを残す

ここでの狙い:
- pure core を top-level に集約する
- stage 固有の glue と shared numerics を分ける

### 2. `base_model.py` を使う側の整理

現状:
- `models/base_model.py` 自体は移植済み
- ただし `stage1/stage2` はまだこの shared scaffold を使っていない

ここでの狙い:
- VLM + trajectory token fusion の shared contract をまとめる

注意:
- 影響範囲が大きい
- `stage1/stage2` の prompt / token / cache 契約に触れるので後回し

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
- `stage1/expert_cfm/core` の薄型化
- `base_model.py` を使う側の整理
- `alpamayo_r1.py` 相当

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
- Alpamayo の同名 file と diff を見たとき、差分は import path と repo 固有 glue にほぼ限定される
