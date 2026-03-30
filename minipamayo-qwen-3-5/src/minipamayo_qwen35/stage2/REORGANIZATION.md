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
- `train_data.py`
  - `stage1/stage1_train_data.py` 相当
  - `ReasoningSftJsonlDataset` 用の train/val dataloader 構築
  - `val_fraction` 分割ロジック
  - handoff probe dataset 構築
- `runtime.py`
  - `stage1/stage1a_runtime.py`, `stage1/expert_cfm/runtime.py` 相当
  - 学習周辺の足回りは `stage1/stage1_train_runtime.py` をそのまま使う
  - Stage 2 用 batch 準備
  - weighted loss
  - token metrics
  - teacher-forced eval
  - reasoning rollout + handoff 用 helper
  - runtime dataclass の定義
- `metadata.py`
  - `stage1/expert_cfm/metadata.py` 相当
  - `stage2_metadata` の生成
  - checkpoint payload / summary payload の canonical builder
- `bundle.py`
  - `stage2 inference/eval/train` が共有する復元経路
  - Stage 1A 復元は `stage1a_components` / `stage1a_conditioning` に委譲する
  - Stage 1A checkpoint から model / processor / token contract を再構築
  - Stage 2 checkpoint の読み込みと state_dict 適用
  - 必要に応じて Stage 1B checkpoint と wrapper の組み立て
- `wrapper.py`
  - Alpamayo wrapper の組み立てに専念
  - config 解析や sample I/O は持たない

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
    ├── train_data.py
    ├── runtime.py
    ├── metadata.py
    ├── bundle.py
    ├── wrapper.py
    ├── preprocess/
    │   ├── __init__.py
    │   ├── __main__.py
    │   └── build_jsonl.py
    ├── train/
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── canonical.py
    │   └── runner.py
    ├── eval/
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── canonical.py
    │   └── runner.py
    └── inference/
        ├── __init__.py
        ├── __main__.py
        ├── canonical.py
        └── runner.py
```

この構成での役割分担は次です。

- `cli.py`
  - `train/eval/inference` が共有する config-json 解決と引数 validation
- `dataset.py`
  - `ReasoningSftJsonlDataset` と collate
- `train_data.py`
  - train/val split, dataloader, handoff probe dataset
- `runtime.py`
  - teacher-forced batch 準備、loss、metrics、eval、rollout helper
- `metadata.py`
  - `stage2_metadata`, checkpoint payload, summary payload
- `bundle.py`
  - Stage 1A / Stage 2 / Stage 1B をまたぐ復元経路の共通化
- `wrapper.py`
  - Alpamayo wrapper の assembly
- `train/eval/inference/runner.py`
  - entrypoint ごとの薄い orchestration

補足です。

- `common.py` はこの構成では不要にするか、残すとしてもごく小さい pure helper のみを置く
- `preprocess/` は当面は独立のままでよい
- `canonical.py` と `__main__.py` は公開 entrypoint 名を壊さないための薄いラッパとして残す

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

## 4. Data loader を `stage1_train_data.py` 型に外出しする

今の `build_dataloaders(args)` は `train/runner.py` に閉じていますが、これは `stage1/stage1_train_data.py` と同じく shared に出してよい責務です。

最低限ほしい API は次です。

- `build_stage2_train_val_dataloaders(...)`
- `build_stage2_handoff_probe_dataset(...)`

これにより、train runner から dataset 分割と loader 設定を外せます。validation split の条件や `max_samples` の扱いも 1 箇所に固定できます。

## 5. metadata / checkpoint schema を builder 化する

`stage2_metadata` の組み立て、checkpoint payload、summary payload は runner の中に直接書かず、builder にまとめるべきです。

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
- `inference/runner.py` は sample 読み込みと output payload 生成だけに寄せる

つまり依存方向を次に固定します。

- runner -> bundle -> wrapper

であって、

- runner -> wrapper -> 追加で checkpoint 解釈

にはしない方が保守しやすいです。

## 7. train / eval / inference runner に残す責務

整理後の runner は次だけを持つのが望ましいです。

### train/runner.py

- train 用 parser
- `main()`
- optimizer / scheduler / early stopping の orchestration
- log 出力

### eval/runner.py

- eval 用 parser
- `main()`
- runtime を呼んで summary を書く

### inference/runner.py

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
2. `train_data.py` を追加し、train 側の dataset / split / handoff probe 構築を外出しする
3. `runtime.py` を追加し、`prepare_stage2_batch`, `evaluate`, `generate_reasoning_handoff` を移す
4. `metadata.py` を追加し、`stage2_metadata` と checkpoint payload を builder 化する
5. `bundle.py` を追加し、`eval` / `inference` の復元経路を共通化する
6. 最後に `common.py` を縮小し、runner を薄くする

この順番なら、各段階で runner の public behavior をほぼ変えずに内部整理できます。

## 非目標

今回の整理では、次は無理にやらない方がよいです。

- `stage2` 全体をさらに細かい micro utility に分解すること
- `preprocess/build_jsonl.py` まで同じ CLI に完全統一すること
- Alpamayo wrapper のアルゴリズム自体を変更すること
- `stage1a` / `stage1b` と完全に同じファイル名構成にそろえること

まずは「runner を薄くし、shared layer を作る」ことが先です。

## 完了条件

整理が完了したと言える状態は次です。

- `train/eval/inference` に `_load_config_args(...)` が残っていない
- `train/eval/inference` に `load_components(...)` の直呼びが残っていない
- `stage2_metadata` と checkpoint payload が builder 経由でしか生成されない
- Stage 2 の teacher-forced 評価 API が 1 箇所に定義されている
- inference の wrapper 構築が bundle 経由に統一されている
- README と code structure が一致している

## ひとことで言うと

`stage2` は今、機能自体よりも「entrypoint ごとの再実装」が負債になっています。

`stage1a` / `stage1b` を参考にするなら、整理の軸は

- `json_cli`
- `train_data`
- `runtime`
- `metadata`
- `bundle`

の 5 層です。

まずこの 5 層に寄せれば、`stage2` はかなり読みやすくなります。
