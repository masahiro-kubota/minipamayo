# Stage1 VLM_CE Shared Runtime Plan

状態:
- 2026-03-30 時点で、この plan は未反映。

このメモは、`src/minipamayo_qwen35/stage1/vlm_ce` を `stage1/expert_cfm` と同じ発想で shared helper ベースに整理し、train / eval / inference のズレを減らすための方針をまとめたものです。

前提:
- `contract/`, `action_space/`, `diffusion/`, `models/`, `geometry/`, `helper.py` は top-level shared core として維持する。
- `stage1` 配下には Stage 1 固有の runtime glue を置いてよい。
- `vlm_ce` の entrypoint は thin wrapper に寄せる。
- Alpamayo mirror 扱いの file はこの refactor では変更しない。変更対象は `stage1/*` shared glue と `stage1/vlm_ce/*` entrypoint 層に限定する。
- phase 1 では挙動を変えず、まず code path を共通化する。
- とくに以下は分けて扱う。
  - teacher-forced 学習/評価の canonical prompt path
  - autoregressive 推論/評価の Alpamayo-style prompt path

## リファクタ後の目標ディレクトリ構成

`expert_cfm` と同じ考え方で、`vlm_ce` も最終的には flat な entrypoint package にする。

目標:

```text
stage1/
  stage1a_conditioning.py
  vlm_ce/
    __init__.py
    cli.py
    components.py
    prompting.py
    generation.py
    metrics.py
    runtime.py
    train.py
    eval.py
    inference.py
    profile.py
```

この形にすると、`expert_cfm` と同様に

- `stage1/stage1a_conditioning.py`
  - downstream stage 向けの最小 bridge
- `stage1/vlm_ce/*`
  - shared runtime / bootstrap / prompting / metrics
  - thin entrypoint / loop / output formatting

という責務分離になる。

`task_spec` variant は、`expert_cfm` のように package 構造で増やすのではなく、

- `train.py`
- `eval.py`
- `inference.py`

が `Stage1TaskSpec` を明示引数または thin wrapper で受ける構造にする。

つまり、最終形では今の

- `train/runner.py`
- `train/canonical.py`
- `train/experiments/steer_only.py`
- `eval/runner.py`
- `eval/canonical.py`
- `eval/experiments/steer_only.py`
- `inference/alpamayo_style.py`
- `inference/canonical.py`

のような mode ごとの入れ子は解消し、entrypoint は `expert_cfm` と同じ粒度に揃える。

## 現状

`vlm_ce` は見た目上は `train/`, `eval/`, `inference/` に分かれているが、shared helper の置き場が runner 側に寄っている。

いま shared source になっているもの:

- `train/runner.py`
  - `prepare_batch(...)`
  - `prepare_prompt_inputs_with_history(...)`
  - `build_full_inputs_from_prompt_inputs(...)`
  - `compute_token_accuracy(...)`
  - `build_model_load_kwargs(...)`
  - `build_processor_kwargs(...)`
  - `load_checkpoint(...)`
  - `inject_history_inputs_embeds(...)`
  - `move_inputs_to_device(...)`
  - `model_forward_inputs(...)`
  - `format_gib(...)`
- `eval/runner.py`
  - `load_components(...)`
  - `greedy_generate_action_tokens(...)`
  - `resolve_checkpoint_args(...)`
  - `resolve_processor_path(...)`
  - `resolve_dtype(...)`
  - `prepare_alpamayo_prompt_inputs_with_history(...)`
- `inference/alpamayo_style.py`
  - `resolve_task_spec(...)`
  - `resolve_checkpoint_kind(...)`
  - single-sample の processor / registry / history-quantizer / quantizer bootstrap
  - single-sample の Alpamayo-style prompt build / decode / metric path
- `eval/__init__.py`
  - `load_components`
  - `main`
- `inference/__init__.py`
  - `main`
- `train/__init__.py`
  - runner helper の再 export surface
- `vlm_ce/__init__.py`
  - package root だが、将来形の public surface は未整理
- `train/profile.py`
  - `prepare_batch(...)`
  - `build_model_load_kwargs(...)`
  - `model_forward_inputs(...)`
  - `format_gib(...)`
  を `train/runner.py` から直接 import している

そのため依存方向がねじれている。

- `stage1a_conditioning.py` は `vlm_ce.eval` の `load_components(...)` と `vlm_ce.train` の `prepare_prompt_inputs_with_history(...)` / `model_forward_inputs(...)` に依存している
- `stage1a_conditioning.py` は `CanonicalStage1Spec()` を hardcode しており、shared checkpoint/task-spec bootstrap を使っていない
- `inference/alpamayo_style.py` は `eval/runner.py` の helper を import している
- `train/__init__.py` は runner を library surface として再 export している
- `train/profile.py` は `train/runner.py` の helper を import して smoke/profile を組んでいる

つまり今は、`vlm_ce` の runner が entrypoint であると同時に library source になっている。

## 問題

### 1. dependency direction が逆転している

本来は

- shared helper
- runner

の順で依存すべきだが、現状は

- `stage1a_conditioning.py` -> `vlm_ce.eval` / `vlm_ce.train`
- `inference/alpamayo_style.py` -> `eval/runner.py`

となっている。

これだと `vlm_ce` runner の中身を直すたびに、Stage1B 条件抽出や Stage1A inference が巻き込まれる。

### 2. prompt / generate / metrics の責務が runner 内に残っている

現状の分散は大きく 3 つある。

- prompt 構築
  - canonical processor prompt は `train/runner.py`
  - Alpamayo-style message prompt は `eval/runner.py` と `inference/alpamayo_style.py`
- generate / decode
  - greedy generate は `eval/runner.py`
  - inference はそこを import している
- metric 計算
  - token accuracy は `train/runner.py`
  - trajectory / action metrics は `eval/runner.py` と `inference/alpamayo_style.py`

### 3. entrypoint package が厚い

現状の `vlm_ce` は runner 以外にも

- `vlm_ce/__init__.py`
- `train/__init__.py` の re-export surface
- `train/canonical.py`
- `eval/canonical.py`
- `inference/canonical.py`
- `__main__.py`
- `eval/__init__.py` の re-export
- `inference/__init__.py` の re-export

といった薄い wrapper を抱えている。

これ自体は動くが、どこが public API でどこが entrypoint なのかが曖昧になる。

### 4. config / preflight / completion-guard が entrypoint ごとに散っている

現状の `vlm_ce` では、JSON-only config parse と runtime preflight が複数箇所に分散している。

- `train/runner.py`
  - `_load_config_args(...)`
  - `parse_args(...)`
- `eval/runner.py`
  - `_load_config_args(...)`
  - `parse_args(...)`
  - `require_completed_training_run(...)`
  - `enforce_runtime_prerequisites(...)`
- `inference/alpamayo_style.py`
  - `_load_config_args(...)`
  - `parse_args(...)`
  - `enforce_runtime_prerequisites(...)`
  - checkpoint completion guard は未適用
- `train/profile.py`
  - 独自 `parse_args(...)`
  - `enforce_runtime_prerequisites(...)`

このため、引数契約や preflight policy を変えるたびに複数 file を触る必要がある。

### 5. processor / image-budget policy の出所が分散している

現状の pixel budget は 2 系統ある。

- `utils/image_budget.py`
  - canonical fixed budget
- `helper.py`
  - `MIN_PIXELS`, `MAX_PIXELS`

数値は一致しているが、`eval/train` は前者、`inference/alpamayo_style.py` は後者を見ている。

つまり policy は実質同じでも、source-of-truth が分かれている。

### 6. eval 専用 helper まで shared 化すると境界が濁る

`eval/runner.py` には shared に見えるが、実際には eval 専用のものもある。

- `require_extract_summary(...)`
- `infer_episode_id(...)`
- `record_time_ns(...)`
- `elapsed_seconds(...)`
- MCAP schema / writer helper

これらは Stage1A shared runtime ではなく、eval layer に残す方がよい。

一方で、eval 由来でも shared に出すべき helper もある。

- `require_record_field(...)`
  - いまは `inference/alpamayo_style.py` からも import されているため、eval layer に置いたままだと依存方向がねじれる
- checkpoint 完了 guard
  - eval だけでなく inference / downstream conditioning でも同じ policy を共有すべきである

### 7. task-spec variant と decode path も shared 契約に含める必要がある

`vlm_ce` には canonical だけでなく `steer_only` variant がある。

- `train/canonical.py`
- `eval/canonical.py`
- `train/experiments/steer_only.py`
- `eval/experiments/steer_only.py`

現状は `main(task_spec=...)` 注入で回っているので、この構造自体は薄い wrapper として妥当である。

ただし、shared runtime 側が `CanonicalStage1Spec` 前提になってしまうと、thin-wrapper 化した後に variant ごとにまた分岐実装が生まれる。

さらに eval / inference には token から最終出力への decode path がある。

- token ids -> bins
- token ids -> target tensor
- target tensor -> full action tensor
- full action tensor -> rollout waypoints

この path を eval / inference に別々に残すと、Stage1A の本質的な挙動差がまた生まれる。

また、現状は inference 側に `steer_only` entrypoint が無い。

- `inference/canonical.py`
  - `alpamayo_style.py` の thin wrapper
- `inference/experiments/steer_only.py`
  - 未実装

shared runtime 化したあとも canonical-only 前提のままだと、後で `steer_only` inference を足すときにまた専用 script を増やしやすい。

したがって、inference も最初から `task_spec` を切り替えられる shared surface にしておくべきである。

### 8. metadata / checkpoint bootstrap も一部重複している

`vlm_ce` では、prompt/generate だけでなく metadata bootstrap も分散している。

- `train/runner.py`
  - `build_stage1_metadata(...)`
  - `checkpoint_payload(...)`
  - `full_checkpoint_payload(...)`
  - `model_only_checkpoint_payload(...)`
- `train/profile.py`
  - `build_stage1_metadata(...)`

特に `build_stage1_metadata(...)` は train と profile の両方に存在しており、registry / history / quantizer / task-spec metadata の契約が二重管理になっている。

checkpoint save 自体は train 固有でよいが、

- `stage1_metadata`
- registry/history/quantizer metadata
- task-spec metadata の組み立て

は shared helper に寄せる方がよい。

## 確定方針

### 1. Stage1A shared runtime を `stage1/` 直下へ引き上げる

`vlm_ce` runner の中で library 的に使われている helper は、`vlm_ce` の外に出す。

目標:

```text
stage1/
  stage1a_conditioning.py
  vlm_ce/
    __init__.py
    cli.py
    components.py
    prompting.py
    generation.py
    metrics.py
    runtime.py
    train.py
    eval.py
    inference.py
    profile.py
```

ここでの責務は次のとおり。

- `vlm_ce/cli.py`
  - JSON-only config parse
  - 共通 path key / list key 解決
  - CUDA / preflight / completion guard の共通 policy
  - canonical image-budget validation
  - `vlm_ce` 各 entrypoint の thin CLI helper
- `vlm_ce/components.py`
  - checkpoint / processor / tokenizer / registry / quantizer / dtype の解決
  - `load_components(...)` 相当
  - `resolve_checkpoint_args(...)`
  - `resolve_checkpoint_kind(...)`
  - `resolve_processor_path(...)`
  - `resolve_dtype(...)`
  - `resolve_task_spec_from_checkpoint(...)`
  - `build_model_load_kwargs(...)`
  - `build_processor_kwargs(...)`
  - `load_checkpoint(...)`
  - tokenizer special-token 追加
  - registry / history_registry / quantizer bootstrap
  - `task_spec` を受けた variant-aware bootstrap
  - `stage1_metadata` 構築 helper
  - registry / history / quantizer metadata の組み立て helper
  - processor settings / image-budget source-of-truth の統一
  - completed-run guard の共通入口
- `vlm_ce/prompting.py`
  - `build_prompt_text(...)`
  - `prepare_prompt_inputs_with_history(...)`
  - `prepare_alpamayo_prompt_inputs_with_history(...)`
  - `inject_history_inputs_embeds(...)`
  - `build_full_inputs_from_prompt_inputs(...)`
  - `prepare_batch(...)`
  - `move_inputs_to_device(...)`
- `vlm_ce/generation.py`
  - `ActionTokenLogitsProcessor`
  - `greedy_generate_action_tokens(...)`
  - 必要なら token append / decode helper もここに寄せる
- `vlm_ce/metrics.py`
  - `compute_token_accuracy(...)`
  - target / action / waypoint metric helper
  - `require_record_field(...)`
- `vlm_ce/runtime.py`
  - `Stage1ARuntime` dataclass
  - `load_stage1a_runtime(...)`
  - `run_stage1a_teacher_forced_batch(...)`
  - `run_stage1a_rollout_batch(...)`
  - train / eval / inference / profile が共通で使う batch-level API
  - token -> target -> action -> waypoint の decode 済み batch result
  - single-sample inference と batch eval の rollout/decode path を同じ code path に乗せる
  - `model_forward_inputs(...)`
- `stage1/stage1a_conditioning.py`
  - `expert_cfm` など downstream stage が必要とする最小 bridge
  - `vlm_ce/components.py` / `vlm_ce/prompting.py` / `vlm_ce/runtime.py` を参照するだけにする

`format_gib(...)` のような cross-entrypoint の小物 utility も、`train/runner.py` に置いたまま re-export しない。必要なら `vlm_ce/runtime.py` か小さい shared util file に寄せる。

### 2. `stage1a_conditioning.py` は shared runtime を参照する薄い bridge に戻す

いまの `stage1a_conditioning.py` は `vlm_ce.eval` / `vlm_ce.train` を参照しているが、これは逆向き依存なのでやめる。

目標:

- `expert_cfm` が必要とする Stage1A prompt-cache extraction は
  - `vlm_ce/components.py`
  - `vlm_ce/prompting.py`
  - `vlm_ce/runtime.py`

だけを参照する

これで

- `expert_cfm` -> `stage1a_conditioning.py` -> `vlm_ce/* shared`
- `vlm_ce` -> `vlm_ce/* shared`

の一方向依存に揃う。

さらに、`stage1a_conditioning.py` 自体も checkpoint から `task_spec` / registry / quantizer を shared bootstrap で解決する形にする。`CanonicalStage1Spec()` hardcode は残さない。

### 3. prompt style の差は code duplication ではなく strategy として表現する

いまの Stage1A には、意図的に 2 つの prompt 経路がある。

- teacher-forced path
  - `processor(text=..., images=...)`
- autoregressive Alpamayo-style path
  - `processor.apply_chat_template(...)`

これを無理に 1 個に潰すのではなく、shared module の中で明示的に 2 strategy として持つ。

例えば:

- `prompt_mode="canonical_teacher_forced"`
- `prompt_mode="alpamayo_message_rollout"`

のようにしておくと、

- code path は共通化できる
- 挙動差は意図されたものとして残せる
- 将来完全統一したいときの差分も見やすい

### 4. `task_spec` は shared runtime の明示引数にする

canonical と `steer_only` の差は、別 runner 実装ではなく `Stage1TaskSpec` の差として扱う。

つまり、

- shared component bootstrap
- batch encode
- rollout decode
- metric helper

の全部が `task_spec` を明示引数に受ける形にする。

これで `canonical.py` / `experiments/steer_only.py` は、最終的にも「どの `task_spec` を渡すか」だけの thin wrapper にできる。

inference 側も同様で、将来 `steer_only` inference を足すとしても、別 script を増やすのではなく `task_spec` wrapper だけで済む構造にする。

### 5. eval / inference は shared runtime の thin wrapper にする

目標は `expert_cfm` と同じで、

- `eval.py`
  - dataset loop
  - extract summary / MCAP writer
  - summary 集計
- `inference.py`
  - single-sample dataset access
  - payload 出力

だけを持ち、

- model load
- prompt build
- greedy rollout
- teacher-forced forward
- decode
- metric helper

は shared runtime を呼ぶだけにする。

ただし、以下は eval layer に残す。

- `extract_summary.json` 読み出し
- MCAP writer / schema
- episode/time metadata 整形
- summary payload の最終集計

逆に、single-sample inference でも使っているものは eval layer に残さない。

- `require_record_field(...)`
- checkpoint/task-spec/kind 解決 helper

### 6. `train.py` も runner 内 helper を減らす

train loop 自体は残るが、以下は shared module へ寄せる。

- batch 構築
- teacher-forced forward helper
- token accuracy
- checkpoint/model/processor load helper

train が持つべきものは主にこれだけにする。

- optimizer / scheduler
- resume / early stopping
- checkpoint save
- wandb / summary

`profile.py` も同じ shared helper を使うようにする。

つまり profile は独立実装を持たず、

- component load
- prompt build
- teacher-forced batch forward

を shared runtime から呼ぶだけにする。

### 7. `vlm_ce` package 自体も最終的には flat に寄せる

最終目標:

```text
stage1/vlm_ce/
  __init__.py
  cli.py
  components.py
  prompting.py
  generation.py
  metrics.py
  runtime.py
  train.py
  eval.py
  inference.py
  profile.py
```

理由:

- `train/runner.py` のような runner-as-library をやめたい
- `train/canonical.py` / `eval/canonical.py` / `inference/canonical.py` の薄い wrapper は最終的に不要
- public surface を entrypoint ごとに明確化したい
- `vlm_ce/__init__.py` / `inference/__init__.py` / 各 `__main__.py` は runner helper を export しない最小 surface にしたい

`task_spec` variant は directory を増やして表現しない。

- default entrypoint は `train.py` / `eval.py` / `inference.py`
- variant 切り替えは `Stage1TaskSpec` 引数か、ごく薄い top-level wrapper に限定する
- `expert_cfm` と同じく、package 構造自体は flat に保つ

ただしこれは phase 2 でよい。phase 1 は shared module 抽出を優先する。

## 実装順

### Phase 1: shared helper の抽出

1. `vlm_ce/cli.py` を作る
2. `vlm_ce/components.py` を作る
3. `vlm_ce/prompting.py` を作る
4. `vlm_ce/generation.py` を作る
5. `vlm_ce/metrics.py` を作る
6. `vlm_ce/runtime.py` を作る
7. `stage1a_conditioning.py` の依存を `vlm_ce` runner から切る

この phase の完了条件:

- `stage1a_conditioning.py` が `vlm_ce.eval` / `vlm_ce.train` を import しない
- `inference/alpamayo_style.py` が `eval/runner.py` を import しない
- `train/__init__.py` に大量 re-export が不要になる
- `train/profile.py` が `train/runner.py` helper を import しない
- pixel budget の source-of-truth が 1 箇所になる
- `task_spec` variant が shared bootstrap/runtime の引数として通る
- inference にも completion guard が入り、eval と同じ checkpoint policy になる
- `require_record_field(...)` が eval layer から外れ、inference が eval helper を import しない

### Phase 2: runner の thin-wrapper 化

1. `eval/runner.py` から
   - config parse / preflight
   - component load
   - generate helper
   - prompt helper
   - metric helper
   を外す
2. `inference/alpamayo_style.py` を shared runtime 呼び出しに置き換える
3. `train/runner.py` から config parse / bootstrap / batch/prompt/metric helper を外す
4. `profile.py` も shared component/prompt helper を使うようにする

この phase の完了条件:

- `eval` と `inference` は runtime 呼び出しと output 整形だけになる
- `train` は optimizer loop と checkpoint orchestration が主になる
- `profile` は forward/backward profiling orchestration だけになる
- token -> action -> waypoint decode path が shared runtime に揃う

### Phase 3: package flatten

1. `vlm_ce/train.py`, `eval.py`, `inference.py`, `profile.py` に畳む
2. `canonical.py` wrapper を消す
3. `vlm_ce/__init__.py`, `inference/__init__.py`, 各 `__main__.py` を最小 surface に整理する
4. `train/__init__.py` と `eval/__init__.py` の runner re-export surface を消す

## 非目標

この plan では以下は同時にやらない。

- teacher-forced prompt と Alpamayo-style prompt の意味論的統一
- Stage1A の学習 recipe 自体の変更
- Stage1A checkpoint schema の変更
- MCAP schema / output payload schema の変更
- Alpamayo mirror file の変更

つまり phase 1 と phase 2 の目的は、**挙動変更ではなく code path の一本化**である。

## 確認項目

- `python -m py_compile ...`
- `uv run python -m minipamayo_qwen35.stage1.vlm_ce.train --help`
- `uv run python -m minipamayo_qwen35.stage1.vlm_ce.eval --help`
- `uv run python -m minipamayo_qwen35.stage1.vlm_ce.inference --help`
- Stage1A inference smoke
- Stage1A eval smoke
- Stage1B runtime smoke
  - `stage1a_conditioning.py` 経由の prompt-cache 抽出が壊れていないこと

## 完了条件

- `vlm_ce` runner が shared helper の source ではなくなる
- `stage1a_conditioning.py` が `vlm_ce` runner を参照しない
- `eval` と `inference` の generate/prompt/metric path が shared runtime に揃う
- `train` / `eval` / `inference` / `profile` で model/processor/checkpoint load 契約が一致する
- config parse / preflight / image-budget policy の source-of-truth が 1 箇所になる
- `canonical` と `steer_only` が `task_spec` 差だけで動く
- Stage1A の挙動差が intentional な strategy 差だけになる
