# Stage 2 Reorganization Plan

`stage2` は `stage1a` / `stage1b` と比べると、共通責務が `train` / `eval` / `inference` に散っています。

現状の主な問題は次です。

- `config-json` の解決ロジックが `train/runner.py`, `eval/runner.py`, `inference/runner.py` に重複している
- `stage1a_checkpoint` を起点に `load_components(...)` してモデルを復元する処理が複数箇所に散っている
- `stage2 checkpoint` の検証、`stage2_metadata` / `run_metadata` / summary payload の扱いが entrypoint ごとにばらついている
- `ReasoningSftJsonlDataset` 用の dataloader 構築が `stage1_train_data.py` 相当の共通レイヤを持っていない
- `common.py` が「batch 準備」「loss」「eval」「generation/handoff probe」まで抱えており、責務が広い
- inference だけが `wrapper.py` を通じて別系統の bundle を持ち、`eval` / `train` と復元経路が揃っていない

## 基本方針

`stage1a` / `stage1b` を参考に、`stage2` も次の原則で整理する。

- runner は薄くする
- config 解決は 1 箇所に寄せる
- checkpoint / runtime の復元は 1 箇所に寄せる
- dataset / dataloader 構築は 1 箇所に寄せる
- metadata / checkpoint schema は canonical に固定する
- 学習・評価・推論の差分だけを各 entrypoint に残す
- 新しい shared module を作る前に、まず `stage1` 側の既存 shared module を再利用する

重要なのは、`stage2` を細かい utility 群に分割しすぎないことです。`stage1a` / `stage1b` と同じく、まずは「責務単位の中くらいのモジュール」に寄せるのが良いです。

## 目標構成

第一段階の目標は、`stage2/reasoning_sft` を次の責務に整理することです。

- `cli.py`
  - `stage1/stage1_json_cli.py` 相当
  - 実装はできるだけ `stage1/stage1_json_cli.py` の薄い wrapper にする
  - `config-json` の canonical 解決
  - `parse_*_json_only_args(...)`
  - 共通 path key / list key の定義
  - CUDA 前提や runtime 前提の共通検証
- `dataset.py`
  - `ReasoningSftJsonlDataset`, collate, train/val dataloader, handoff probe dataset
- `runtime.py`
  - `stage1/stage1a_runtime.py`, `stage1/expert_cfm/runtime.py` 相当
  - 学習周辺の足回りは `stage1/stage1_train_runtime.py` をそのまま使う
  - Stage 2 用 batch 準備
  - weighted loss
  - token metrics
  - teacher-forced eval
  - reasoning rollout + handoff 用 helper
  - runtime dataclass の定義
- `bundle.py`
  - `stage2 inference/eval/train` が共有する復元経路
  - Stage 1A 復元は `stage1a_components` / `stage1a_conditioning` に委譲する
  - Stage 1A checkpoint から model / processor / token contract を再構築
  - Stage 2 checkpoint の読み込みと state_dict 適用
  - 必要に応じて Stage 1B checkpoint と wrapper の組み立て
- `wrapper.py`
  - Alpamayo wrapper の組み立てに専念
  - config 解析や sample I/O は持たない

必要になったら第二段階で次を追加抽出する。

- `train_data.py`
  - dataset/dataloader 周辺がさらに肥大化したときだけ切り出す
- `metadata.py`
  - checkpoint payload / summary payload builder が複数 caller に広がったときだけ切り出す

## 想定ディレクトリ構成

整理後の `stage2` は、少なくとも次の構成を目標にするのがよいです。

```text
stage2/
├── README.md
├── REORGANIZATION.md
├── __init__.py
└── reasoning_sft/
    ├── __init__.py
    ├── cli.py
    ├── dataset.py
    ├── runtime.py
    ├── bundle.py
    ├── wrapper.py
    ├── train.py
    ├── eval.py
    ├── inference.py
    ├── preprocess/
    │   ├── __init__.py
    │   ├── __main__.py
    │   └── build_jsonl.py
```

この構成での役割分担は次です。

- `cli.py`
  - `train/eval/inference` が共有する config-json 解決と引数 validation
- `dataset.py`
  - `ReasoningSftJsonlDataset`, collate, train/val split, dataloader, handoff probe dataset
- `runtime.py`
  - teacher-forced batch 準備、loss、metrics、eval、rollout helper
- `bundle.py`
  - Stage 1A / Stage 2 / Stage 1B をまたぐ復元経路の共通化
- `wrapper.py`
  - Alpamayo wrapper の assembly
- `train.py`, `eval.py`, `inference.py`
  - entrypoint ごとの薄い orchestration

補足です。

- `common.py` はこの構成では不要にするか、残すとしてもごく小さい pure helper のみを置く
- `preprocess/` は当面は独立のままでよい
- `train/`, `eval/`, `inference/` のネストは `stage1` と揃わないので、最終形としては採らない
- もし既存の module path 互換が必要なら、`train/`, `eval/`, `inference/` は一時的な shim としてだけ残す

## リファクタ開始前に固定しておく前提

着手前に、次の前提を文書上で固定しておく必要があります。

### 1. 最初の PR では挙動を変えない

最初の PR の目的は構造整理であって、学習ロジックや推論ロジックの改善ではありません。

そのため、最初の PR では次を変えない前提で進めます。

- config JSON の schema
- CLI 引数名
- checkpoint schema
- summary / output JSON の key
- seed, autocast, generation parameter の既定動作
- Stage 1A / Stage 1B checkpoint の解釈方法

### 2. 公開 entrypoint は壊さない

少なくとも次の実行経路は、リファクタ途中でも維持する前提にします。

- `python -m minipamayo_qwen35.stage2.reasoning_sft.preprocess`
- `python -m minipamayo_qwen35.stage2.reasoning_sft.train`
- `python -m minipamayo_qwen35.stage2.reasoning_sft.eval`
- `python -m minipamayo_qwen35.stage2.reasoning_sft.inference`

### 3. 既存 import path も確認対象に含める

現状は docs や補助コードが次のような path を参照しています。

- `minipamayo_qwen35.stage2.reasoning_sft.train`
- `minipamayo_qwen35.stage2.reasoning_sft.eval`
- `minipamayo_qwen35.stage2.reasoning_sft.inference`
- `minipamayo_qwen35.stage2.reasoning_sft.inference.runner`

特に `src/minipamayo_qwen35/test_inference.py` は `inference.runner` を直接 import しています。

このため、`train.py` / `eval.py` / `inference.py` への flatten は最終形としては自然でも、最初の PR では次のどちらかが必要です。

- 旧 path を shim で残す
- 参照箇所を同じ PR で一括更新する

結論として、flatten は第一段階の必須条件ではなく、互換性整理とセットで扱うべきです。

## 最初の PR のスコープ

最初の PR は、移動よりも「共通責務の抽出」を優先します。

含めるもの:

- `cli.py` の追加
- `dataset.py` への dataloader / handoff probe 構築の寄せ
- `runtime.py` の追加
- `bundle.py` の追加
- `wrapper.py` の責務縮小
- 既存 runner から shared module を呼ぶ形への書き換え

含めないもの:

- いきなりの大規模ファイル移動
- public module path の削除
- checkpoint schema の変更
- output JSON schema の変更

この文書の想定ディレクトリ構成は最終形の候補であって、第一段階の PR で必ずそこまで到達する必要はありません。

## 旧実装から新責務への対応表

着手時に迷わないよう、まずは次の対応表で移すのがよいです。

| 現在の場所 | 新しい置き場 | 備考 |
| --- | --- | --- |
| `train/runner.py:_load_config_args`, `eval/runner.py:_load_config_args`, `inference/runner.py:_load_config_args` | `cli.py` | `stage1_json_cli.py` の wrapper に寄せる |
| `train/runner.py:build_dataloaders` | `dataset.py` | handoff probe dataset 構築も同じ場所に寄せる |
| `common.py:prepare_stage2_batch` | `runtime.py` | teacher-forced path の中心 |
| `common.py:compute_weighted_loss` | `runtime.py` | pure function のままでよい |
| `common.py:compute_token_metrics` | `runtime.py` | pure function のままでよい |
| `common.py:evaluate` | `runtime.py` | `evaluate_stage2(...)` に寄せる |
| `common.py:generate_reasoning_handoff` | `runtime.py` | rollout helper |
| `common.py:evaluate_handoff_probe` | `runtime.py` | probe 専用 helper |
| `wrapper.py:load_stage2_inference_bundle` | `bundle.py` | `wrapper.py` 自体は assembly 専用に縮小する |
| `train/runner.py` の `stage2_metadata` / checkpoint payload 生成 | 当面は `train/runner.py` に残す | schema を固定した後、必要なら `metadata.py` へ出す |

## どの時点で flatten するか

`stage1` と揃えるなら、最終的には `reasoning_sft/train.py`, `eval.py`, `inference.py` が自然です。

ただし、現状は次の制約があります。

- `python -m minipamayo_qwen35.stage2.reasoning_sft.train` などの entrypoint が存在する
- docs が `stage2/reasoning_sft/train` という現在構成を前提に書かれている
- 一部コードが `inference.runner` を import している

そのため、flatten の推奨順は次です。

1. まず shared module を追加し、既存 `train/runner.py`, `eval/runner.py`, `inference/runner.py` を薄くする
2. 互換性 shim を作るか、参照箇所を一括更新できる状態にする
3. そのあとで `train.py`, `eval.py`, `inference.py` へ flatten する

つまり、flatten 自体は「整理の仕上げ」であって、「最初の分割作業」ではありません。

## リファクタ中の互換性ルール

途中段階では、次を壊したら失敗とみなします。

- `uv run python -m minipamayo_qwen35.stage2.reasoning_sft.train --help`
- `uv run python -m minipamayo_qwen35.stage2.reasoning_sft.eval --help`
- `uv run python -m minipamayo_qwen35.stage2.reasoning_sft.inference --help`
- 既存 config JSON での引数解決
- 既存 checkpoint の load
- `test_inference.py` が依存している import path

## 最低限の検証項目

リファクタ各段階で、少なくとも次は確認します。

### import / entrypoint

- `python -m minipamayo_qwen35.stage2.reasoning_sft.train --help`
- `python -m minipamayo_qwen35.stage2.reasoning_sft.eval --help`
- `python -m minipamayo_qwen35.stage2.reasoning_sft.inference --help`
- `python -m minipamayo_qwen35.stage2.reasoning_sft.preprocess --help`

### config / runtime

- 既存 config JSON を 1 つ使って `parse_args()` が通る
- train/eval/inference で path resolution の結果が変わっていない
- Stage 2 checkpoint から `stage1a_checkpoint` を引く経路が維持されている

### payload / schema

- train checkpoint の top-level key が変わっていない
- eval summary の top-level key が変わっていない
- inference output JSON の top-level key が変わっていない

### behavior smoke

- teacher-forced eval が 1 batch 動く
- inference が 1 sample 動く
- handoff probe が有効時に code path を通る

## Open Questions

着手前に埋めるべき open question は次です。

- flatten を第一段階でやるか、第二段階に送るか
  - 現時点では第二段階を推奨
- `stage2_metadata` builder を初手から作るか
  - 現時点では不要
- `dataset.py` に train loader helper を寄せるか、すぐ `train_data.py` を作るか
  - 現時点では `dataset.py` で十分
- `inference.runner` import を public API とみなすか
  - 少なくとも移行期間中は壊さない前提で扱う

## Stage 1 と共通化できるところ

`stage2` の整理では、`stage2` 専用モジュールを増やす前に、まず `stage1` の既存共通部品を使い回すべきです。

### まず直接再利用するもの

- `stage1/stage1_json_cli.py`
  - `train/eval/inference` で重複している `_load_config_args(...)` と `config-json only` の解決は、これを直接使う
  - `stage2/reasoning_sft/cli.py` は薄い wrapper にする
- `stage1/stage1_train_runtime.py`
  - `set_seed`, `release_cuda_memory`, `log_gpu_preflight`, `write_run_config`, `metric_improved`, W&B helper は新設しない
  - すでに `train` が使っているので、`eval` / `inference` 側の runtime helper もできるだけここに寄せる
- `stage1/stage1a_components.py`
  - Stage 1A checkpoint から `model`, `processor`, `registry`, `history_registry`, `history_quantizer`, `quantizer` を戻す処理は Stage 2 で再実装しない
  - `stage2` 側では `bundle.py` がこれを呼ぶだけにする
- `stage1/stage1a_conditioning.py`
  - prompt text 推論、prompt cache 抽出、Stage 1A conditioning の再利用はここを起点にする
  - Stage 2 が独自に prompt-cache 再構成コードを増やさない
- `stage1/expert_cfm/runtime.py` と `stage1/expert_cfm/cli.py`
  - Stage 2 inference は Stage 1B handoff を含むので、`flow_steps`, CUDA requirement, Stage 1B runtime validation の考え方はここに揃える
  - 命名が揃うなら、引数追加 helper も部分的に再利用する

### 将来的に抽出を検討できるもの

- train/val dataloader 構築
  - `stage1/stage1_train_data.py` と Stage 2 の `train_data.py` は構造が似る
  - ただし dataset class と collate, handoff probe 追加要件が違うので、最初は無理に 1 ファイルへ統合しない
  - もし Stage 3 でも同じ split/loader 骨格が増えるなら、その時点で top-level shared に上げる
- checkpoint / summary schema builder
  - `args`, `initial_eval`, `metrics_history`, `run_metadata` などは `stage1a` / `stage2` で似た shape を取る
  - ただし payload の中身は stage 固有情報が多いので、まずは schema の命名規則だけを合わせる
- runtime dataclass の設計
  - `Stage1ARuntime`, `Stage1BRuntime` のように、Stage 2 も `Stage2TrainingRuntime`, `Stage2InferenceRuntime` を持つ
  - dataclass 自体を共通化する必要はないが、設計パターンは合わせる

### 共通化しすぎない方がよいもの

- `ReasoningSftJsonlDataset` 自体
  - Stage 1 と supervision の shape が違うので、dataset 実装は分けた方が読みやすい
- Stage 2 の weighted loss / handoff probe
  - `<|cot_end|>`, `<|traj_future_start|>` を前提にしたロジックは Stage 2 固有
- Alpamayo wrapper assembly
  - これは Stage 2 inference 固有の都合が強いので、Stage 1 shared に押し戻さない

ひとことで言うと、`stage2` が `stage1` と共通化すべきなのは「足回りと復元経路」であって、「reasoning 特有のロジック」ではありません。

## 推奨する責務分離

### 1. CLI を `stage1_json_cli` 型に揃える

最優先でやるべきです。今の `train`, `eval`, `inference` はそれぞれ `_load_config_args(...)` と `parse_args()` を持っていますが、ほぼ同じ処理です。

整理後は次の形に寄せます。

- `load_stage2_config_args(...)`
- `parse_stage2_json_only_args(...)`
- `add_stage2_train_args(parser)`
- `add_stage2_eval_args(parser)`
- `add_stage2_inference_args(parser)`

これにより、各 runner では

- parser 定義
- stage 固有の追加 validation
- main orchestration

だけを持てば十分になります。

### 2. Stage 2 用 runtime/bundle を一本化する

今は次の 3 系統があります。

- train: `stage1a_checkpoint` から `load_components(...)` して学習用 model を構築
- eval: checkpoint から `stage1a_checkpoint` を引いて再度 `load_components(...)`
- inference: `wrapper.py` 経由で別の bundle を復元

この構成だと、token contract や image budget や state_dict 適用ルールが drift しやすいです。

目標は次の 2 本に揃えることです。

- `load_stage2_training_runtime(...)`
  - Stage 1A ベースモデルを読み込み、Stage 2 teacher-forced 学習に必要な部品を束ねる
- `load_stage2_inference_runtime(...)`
  - Stage 2 checkpoint と Stage 1B checkpoint を使い、reasoning rollout と handoff を実行できる bundle を返す

`eval` は前者に近いので、可能なら `load_stage2_eval_runtime(...)` を別に作るより、`load_stage2_training_runtime(..., checkpoint=...)` のような形で吸収した方が良いです。

## 3. `common.py` を runtime 中心に再編する

`common.py` 自体は悪くないですが、今は責務が広すぎます。

第一段階では分割しすぎず、次のように整理するのが現実的です。

- `runtime.py`
  - `prepare_stage2_batch`
  - `compute_weighted_loss`
  - `compute_token_metrics`
  - `run_stage2_teacher_forced_batch`
  - `evaluate_stage2`
  - `generate_reasoning_handoff`
  - `evaluate_handoff_probe`
- `common.py`
  - なくすか、定数や小さい pure helper のみ残す

特に `run_stage2_teacher_forced_batch(...)` を導入すると、`train` と `eval` が `stage1a` の `run_stage1a_teacher_forced_batch(...)` と同じ読み味になります。

## 4. Data loader はまず `dataset.py` に寄せる

今の `build_dataloaders(args)` は `train/runner.py` に閉じていますが、これは少なくとも runner からは外に出すべきです。

ただし `stage2` はまだ単一タスクなので、初手から `train_data.py` を増やす必要はありません。まずは `dataset.py` に寄せるのが妥当です。

最低限ほしい API は次です。

- `build_stage2_train_val_dataloaders(...)`
- `build_stage2_handoff_probe_dataset(...)`

これにより、train runner から dataset 分割と loader 設定を外せます。validation split の条件や `max_samples` の扱いも 1 箇所に固定できます。

もし将来 `dataset.py` が肥大化したら、その時点で `stage1/stage1_train_data.py` 相当として `train_data.py` を追加すれば十分です。

## 5. metadata / checkpoint schema は最初は runner に残してよい

`stage2_metadata` の組み立て、checkpoint payload、summary payload は、まず schema と key 名を安定させることを優先します。

ただし、これも初手から `metadata.py` を増やす必要はありません。まずは key 名と schema を安定させ、複数 caller で同じ payload builder が欲しくなった時点で切り出す方が自然です。

候補 API:

- `build_stage2_metadata(dataset, args, *, processor_settings)`
- `build_stage2_checkpoint_payload(...)`
- `build_stage2_eval_summary(...)`
- `build_stage2_inference_summary(...)`

狙いは次です。

- canonical key を固定する
- train/eval/inference で metadata 名が微妙にズレるのを防ぐ
- downstream が読む checkpoint schema を安定させる

## 6. wrapper は inference 専用の assembly に閉じ込める

`wrapper.py` は方向性としては良いです。問題は「Stage 2 復元の一部」を抱え込んでいることです。

整理方針は次です。

- `bundle.py` が checkpoint / stage1a / stage1b の読み込み順序を管理する
- `wrapper.py` は「必要な部品が揃ったら Alpamayo wrapper を組む」だけにする
- `inference.py` は sample 読み込みと output payload 生成だけに寄せる

つまり依存方向を次に固定します。

- runner -> bundle -> wrapper

であって、

- runner -> wrapper -> 追加で checkpoint 解釈

にはしない方が保守しやすいです。

## 7. train / eval / inference runner に残す責務

整理後の runner は次だけを持つのが望ましいです。

### train.py

- train 用 parser
- `main()`
- optimizer / scheduler / early stopping の orchestration
- log 出力

### eval.py

- eval 用 parser
- `main()`
- runtime を呼んで summary を書く

### inference.py

- inference 用 parser
- sample 1 件の I/O
- runtime を呼んで output payload を書く

runner から消すべきものは次です。

- JSON config base dir 解決
- Stage 1A checkpoint 由来の model 復元
- Stage 2 checkpoint schema の個別検証
- dataloader 構築の詳細
- metadata payload の細部

## 段階的な実施順

一気にやると壊れやすいので、次の順番を推奨します。

1. `cli.py` を追加し、`train/eval/inference` の config 解決を共通化する
2. `dataset.py` に train 側の dataset / split / handoff probe 構築を寄せる
3. `runtime.py` を追加し、`prepare_stage2_batch`, `evaluate`, `generate_reasoning_handoff` を移す
4. `bundle.py` を追加し、`eval` / `inference` の復元経路を共通化する
5. `wrapper.py` を assembly 専用に縮小する
6. 最後に `common.py` を縮小し、runner を薄くする
7. 必要になったら `train_data.py` / `metadata.py` を追加抽出する

この順番なら、各段階で runner の public behavior をほぼ変えずに内部整理できます。

## 非目標

今回の整理では、次は無理にやらない方がよいです。

- `stage2` 全体をさらに細かい micro utility に分解すること
- `preprocess/build_jsonl.py` まで同じ CLI に完全統一すること
- Alpamayo wrapper のアルゴリズム自体を変更すること
- `stage1a` / `stage1b` と完全に同じファイル名構成にそろえること

まずは「runner を薄くし、shared layer を作る」ことが先です。

## 完了条件

第一段階の完了条件は次です。

- `train/eval/inference` に `_load_config_args(...)` が残っていない
- `train/eval/inference` に `load_components(...)` の直呼びが残っていない
- Stage 2 の teacher-forced 評価 API が 1 箇所に定義されている
- inference の wrapper 構築が bundle 経由に統一されている
- `python -m minipamayo_qwen35.stage2.reasoning_sft.train/eval/inference` の entrypoint が壊れていない
- README と code structure が一致している

第二段階まで含めて整理完了と言うなら、さらに次を満たす状態です。

- 必要なら `train.py`, `eval.py`, `inference.py` への flatten が完了している
- 旧 import path が shim か一括更新で整理されている
- `stage2_metadata` / checkpoint payload / summary payload builder を shared 化する必要が実際に生じており、その抽出が完了している

## ひとことで言うと

`stage2` は今、機能自体よりも「entrypoint ごとの再実装」が負債になっています。

`stage1a` / `stage1b` を参考にするなら、第一段階の整理軸は

- `json_cli`
- `dataset`
- `runtime`
- `bundle`
- `wrapper`

の 5 層です。

まずこの 5 層に寄せれば、`stage2` はかなり読みやすくなります。
