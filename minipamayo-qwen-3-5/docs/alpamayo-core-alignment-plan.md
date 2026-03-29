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

## 実装順

### 1. `hydra` 化を最小範囲で入れる

対象:
- `src/minipamayo_qwen35/action_space/discrete_action_space.py`
- `src/minipamayo_qwen35/contract/task_spec.py`

やること:
- `DiscreteTrajectoryTokenizer` を `action_space_cfg` 受け取りに戻す
- tokenizer 内で `hydra.instantiate(...)` を使う
- `CanonicalStage1Spec` は instantiated object ではなく cfg を tokenizer に渡す

ここでの狙い:
- Alpamayo の config 契約に最小限で合わせる
- `action_space` 周りから hydra の流儀を戻す

注意:
- checkpoint metadata の `action_space_cfg` はそのまま使える形を維持する
- `record_adapter.py` は hydra 化しない

### 2. top-level `diffusion/` を作る

対象:
- いまの `stage1/expert_cfm/core/diffusion.py`
- Alpamayo の `diffusion/base.py`
- Alpamayo の `diffusion/flow_matching.py`

やること:
- pure diffusion 部分を `src/minipamayo_qwen35/diffusion/` に出す
- file 名も Alpamayo に寄せる

想定:
- `diffusion/base.py`
- `diffusion/flow_matching.py`

ここでの狙い:
- `Stage1B` 専用 core と pure diffusion core を分ける
- Alpamayo との diff を見やすくする

注意:
- train/eval/inference glue はまだ `stage1/expert_cfm/` に残す

### 3. `stage1/expert_cfm/core` を薄くする

対象:
- `src/minipamayo_qwen35/stage1/expert_cfm/core/model.py`
- `src/minipamayo_qwen35/stage1/expert_cfm/core/common.py`

やること:
- `diffusion/` と `action_space/` を top-level shared core として参照する
- `stage1/expert_cfm/core` には stage1B 固有 wiring だけを残す

ここでの狙い:
- pure core を top-level に集約する
- stage 固有の glue と shared numerics を分ける

### 4. top-level `models/` を作る

最初に持ってくる候補:
- Alpamayo `models/action_in_proj.py`
- Alpamayo `models/token_utils.py`
- 必要なら Alpamayo `models/delta_tokenizer.py`

ここでの狙い:
- shared model utilities を stage 配下から切り離す
- Alpamayo の `models/` 構造に寄せる

注意:
- まずは軽い shared file から
- `alpamayo_r1.py` 相当はまだ持ち込まない

### 5. 必要なら `base_model.py` 相当を作る

対象:
- Alpamayo `models/base_model.py`

ここでの狙い:
- VLM + trajectory token fusion の shared contract をまとめる

注意:
- 影響範囲が大きい
- `stage1/stage2` の prompt / token / cache 契約に触れるので後回し

### 6. 最後に end-to-end wrapper を考える

対象:
- Alpamayo `models/alpamayo_r1.py`

ここでの狙い:
- Alpamayo の end-to-end 構造に近づける

注意:
- 一番破壊半径が大きい
- `stage1`, `stage2`, `stage3` をまたぐので最後

## 優先順位

先にやる:
- `hydra` 化
- `diffusion/` 切り出し
- `models/` の shared file 移植

後でやる:
- `base_model.py`
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
