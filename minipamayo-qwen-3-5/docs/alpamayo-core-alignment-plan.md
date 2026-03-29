# Alpamayo Core Alignment Plan

このメモは、`minipamayo-qwen-3-5` を `/home/masa/minipamayo/related_repos/alpamayo` に段階的に寄せるための実装順を整理したものです。

前提:
- 目的は「今の repo 都合の abstraction を守ること」ではなく、「Alpamayo 公開実装との差分を減らすこと」。
- `config.py` / `models/alpamayo_r1.py` はすでに持ち込んでおり、`stage2` inference は wrapper 経由に寄せた。
- `helper.py` / `test_inference.py` も top-level に追加済みで、Alpamayo-style の推論入口は一通り揃った。
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

diff -u \
  /home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/helper.py \
  /home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/helper.py

diff -u \
  /home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/test_inference.py \
  /home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/test_inference.py
```

### 1. 差分なし

以下は Alpamayo 側と一致している。

- `action_space/action_space.py`
- `action_space/__init__.py`
- `diffusion/base.py`
- `diffusion/__init__.py`
- `geometry/rotation.py`
- `helper.py`
- `models/action_in_proj.py`
- `models/delta_tokenizer.py`
- `models/token_utils.py`

### 2. import path 差だけ

以下は実質 import path 差だけで、ロジック差は残っていない。

- `action_space/discrete_action_space.py`
- `action_space/unicycle_accel_curvature.py`
- `action_space/utils.py`
- `config.py`
- `diffusion/flow_matching.py`

### 3. runtime shim / entrypoint 差分

以下は Alpamayo 実装をほぼ持ち込んでいるが、`minipamayo-qwen-3-5` の runtime 差や入力契約差を吸収するための差分が残っている。

- `models/base_model.py`
  - `tie_weights(self, *args, **kwargs)` の互換 shim がある
  - これは `transformers==5.4.0` 側の追加引数を受けるため
- `models/alpamayo_r1.py`
  - pure import path 差に加えて以下が残っている
  - Alpamayo の inline 4D additive mask ではなく、inline の 2D padding mask を使う
  - これは `transformers==5.4.0 + Qwen3.5-0.8B + flash_attention_2` の expert path が 2D padding mask を期待するため
  - `transformers==4.57.1` の `Qwen3-VL` には 4D mask を padding mask に落とす互換分岐があるが、`Qwen3.5` の expert path には同等の分岐が無い
  - Alpamayo の in-place `prompt_cache.crop(...)` ではなく、step ごとに cache clone を使う
    - これは `Qwen3_5DynamicCache` が `crop()` を持たないため
- `test_inference.py`
  - Alpamayo は `load_physical_aiavdataset.py` から直接 tensor を作る
  - こちらは `samples_reasoning_sft.jsonl` と `record_adapter` 系の契約を使う
  - wrapper 呼び出し自体は揃えたが、入力 loader は別物
  - これは runtime shim ではなく、demo entrypoint の入力契約差分

### 4. repo 固有として残している差分

以下は Alpamayo には無いが、`minipamayo-qwen-3-5` 側の stage 分割や JSONL record 契約のために残している。

- `action_space/record_adapter.py`
  - record から tensor を読む
  - shape adapter
  - canonical action / waypoint payload の橋渡し
- `diffusion/action_expert.py`
  - `Stage1B` の expert decoding に必要な diffusion adapter
- `models/action_expert.py`
  - `Stage1B` 用 shared action expert 本体

## 確定した方針

### 1. repo 固有 core は top-level shared に残す

対象:
- `action_space/record_adapter.py`
- `diffusion/action_expert.py`
- `models/action_expert.py`

方針:
- これらは「Alpamayo に無い repo 固有 core」として top-level shared に残す
- stage 配下へ戻さず、pure Alpamayo mirror と混ぜない

注意:
- `record_adapter.py` は Alpamayo loader 不在を埋める層
- `action_expert.py` は Stage1B train/eval/inference と wrapper の両方が使う shared 実装

### 2. wrapper builder は当面 runner に残す

対象:
- `stage2/reasoning_sft/inference/runner.py` の wrapper 組み立て部分

方針:
- Stage1A / Stage1B checkpoint 契約を読んで `AlpamayoR1` を構成する builder は、当面 `runner.py` に残す
- 2つ目の明確な caller が出た時点で shared helper へ切り出す

### 3. runtime shim は Qwen3.5 runtime 差として維持する

対象:
- `models/base_model.py`
- `models/alpamayo_r1.py`

方針:
- `Qwen3.5-0.8B` を使う前提を維持する限り、Alpamayo 純正との差分は runtime shim として受容する
- 差分ゼロは目指さず、「なぜ必要かが説明できる shim だけ残す」を採用する

補足:
- `stage2` wrapper inference の expert hard-coded `sdpa` は削除済み
- `stage1B` eval / inference / train は同じ 2D expert mask helper に揃った

## 優先順位

優先度の高い残件は、このメモの範囲では基本的に解消済み。
以後は以下を必要に応じて進める。

- `load_physical_aiavdataset.py` 相当を追加して、Alpamayo と同じ demo 入力経路も持つかどうか
- wrapper builder を shared helper に切り出すだけの再利用需要が出るかどうか
- `Qwen3.5` runtime shim が将来の `transformers` / model stack 更新で不要になるかどうか

## 各段階の確認

各段階で最低限確認する。

- `py_compile`
- `stage1.vlm_ce.train --help`
- `stage1.expert_cfm.train --help`
- `stage2.reasoning_sft.inference --help`
- `test_inference.py --help`
- 必要なら single-sample inference smoke

## 完了条件

- pure shared core は top-level `action_space/`, `diffusion/`, `models/`, `geometry/` にある
- stage 配下には train/eval/inference と stage 固有 glue だけが残る
- Alpamayo の同名 file と diff を見たとき、差分は
  - import path
  - repo 固有 glue
  - Qwen3.5 runtime shim
  に限定される
- `stage2` inference の handoff は wrapper 経由で実行される
