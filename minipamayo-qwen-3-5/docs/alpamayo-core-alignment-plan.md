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

### 3. runtime shim を含む差分

以下は Alpamayo 実装をほぼ持ち込んでいるが、`minipamayo-qwen-3-5` の runtime 差を吸収するための shim が残っている。

- `models/base_model.py`
  - `tie_weights(self, *args, **kwargs)` の互換 shim がある
  - これは `transformers==5.4.0` 側の追加引数を受けるため
- `models/alpamayo_r1.py`
  - pure import path 差に加えて以下が残っている
  - Alpamayo の inline 4D additive mask ではなく、inline の 2D padding mask を使う
  - これは `transformers==5.4.0 + Qwen3.5-0.8B + flash_attention_2` の expert path が 2D padding mask を期待するため
  - `prompt_cache.crop(...)` を `hasattr` で guard する
  - いずれも `Qwen3.5-0.8B + transformers 5.4` の cache / flash-attn runtime 差のための shim
- `helper.py`
  - Alpamayo は `BASE_PROCESSOR_NAME = "Qwen/Qwen3-VL-2B-Instruct"` を使う
  - こちらは Stage1A checkpoint 横の saved processor を読む
  - system/user text も current canonical contract に合わせている
- `test_inference.py`
  - Alpamayo は `load_physical_aiavdataset.py` から直接 tensor を作る
  - こちらは `samples_reasoning_sft.jsonl` と `record_adapter` 系の契約を使う
  - wrapper 呼び出し自体は揃えたが、入力 loader は別物

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

### 5. まだ未移植の大物

大物の未移植は解消済み。

- `config.py`
- `models/alpamayo_r1.py`

はどちらも repo に入り、`stage2/reasoning_sft/inference/runner.py` は manual handoff ではなく wrapper を組み立てて呼ぶ形に変わった。

一方で、Alpamayo 側にある以下はまだそのままの形では持ってきていない。

- `load_physical_aiavdataset.py`
  - こちらは JSONL + `record_adapter.py` 経由なので未移植

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

### 3. runtime shim をどこまで残すか決める

対象:
- `models/base_model.py`
- `models/alpamayo_r1.py`
- `helper.py`

注意:
- `Qwen3.5-0.8B` を使う前提を維持する限り、Alpamayo 純正のままでは通らない可能性が高い
- 差分ゼロを目指すのではなく、「なぜ必要かが説明できる shim だけ残す」が現実的

補足:
- `stage2` wrapper inference の expert hard-coded `sdpa` は削除済み
- `stage1B` eval / inference / train は同じ 2D expert mask helper に揃った

## 優先順位

残り:
- repo 固有 core (`record_adapter.py`, `diffusion/action_expert.py`, `models/action_expert.py`) の最終位置づけ
- wrapper builder の最終配置
- runtime shim (`base_model.py`, `alpamayo_r1.py`, `helper.py`) の整理

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
