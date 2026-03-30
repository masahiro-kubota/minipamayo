# Alpamayo Action-Space Alignment Plan

## 目的

- `/home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/action_space` を、`minipamayo-qwen-3-5` 側の action-space 実装の実質的な正本として扱う。
- `stage1` / `stage2` の action 契約を、repo 独自都合ではなく Alpamayo の実装都合に寄せる。
- `diff -u` で見たとき、pure action-space 本体の差分を極小化する。

## 前提

- 最終目的は論文実装への整合である。
- 公式 Alpamayo 推論コードは、その論文実装に整合していると仮定する。
- そのため、既存 checkpoint / config 互換よりも Alpamayo 契約との一致を優先する。
- ただし、repo 固有の record / JSONL / payload 整形まで Alpamayo 側に存在するわけではないので、それらは別 adapter 層として残す。

## 原則

1. `action_space/` の pure numerics は Alpamayo にできるだけ verbatim で合わせる。
2. repo 固有の shape 整形や record 読み出しは `action_space/` 本体に混ぜず、adapter 層に分離する。
3. `stage1` / `stage2` は Alpamayo 契約に合わせて call site を直す。
4. `diff` を小さくする対象と、repo 固有でよい対象を明確に分ける。

## 対象と非対象

### Alpamayo に揃える対象

- `action_space/action_space.py`
- `action_space/utils.py`
- `action_space/unicycle_accel_curvature.py`
- `action_space/discrete_action_space.py`
- `geometry/rotation.py`

### repo 固有でよい対象

- `contract/record_adapter.py`
- `stage1` / `stage2` の runner 側の payload 整形
- JSONL record から tensor を組み立てる helper
- `kappa_only` のような実験用 `task_spec`

## 理想レイアウト

```text
src/minipamayo_qwen35/
  action_space/
    __init__.py
    action_space.py
    discrete_action_space.py
    unicycle_accel_curvature.py
    utils.py

  geometry/
    rotation.py

  contract/
    record_adapter.py
    ...

  stage1/
    ...

  stage2/
    ...
```

## ファイルごとの方針

### `action_space.py`

- Alpamayo 側の抽象基底と public method 名に揃える。
- docstring, import 順, license header まで可能な限り合わせる。
- repo 固有 helper は追加しない。

### `utils.py`

- Alpamayo 側の数値 helper に寄せる。
- 関数名、並び順、引数順、docstring を合わせる。
- ここに record / dataset 依存を入れない。

### `unicycle_accel_curvature.py`

- Alpamayo 側にほぼ verbatim で合わせる。
- `k` alias は持たない。
- single-traj-group wrapper は持たない。
- `(batch, T, ...)` を前提とした pure numerics のみを持つ。

### `discrete_action_space.py`

- できるだけ Alpamayo 側の `DiscreteTrajectoryTokenizer` に寄せる。
- ただし repo 側の token registry / tokenizer 契約との接着が必要なら、その差分は最小限に限定する。
- このファイルだけは `stage1A` の token 契約に直結するため、変更後に追加検証を必須とする。

### `contract/record_adapter.py`

- Alpamayo には存在しない repo 固有層として扱う。
- 責務は以下に限定する。
  - record から history/future tensor を読む
  - single-traj-group の入出力整形
  - canonical action と waypoint payload の橋渡し
- pure numerics はここに書かない。

## 実装順

### Phase 1

- `stage1` / `stage2` の runner 側から group 軸前提を外す。
- `pred_xyz[:, 0, ...]`, `pred_xyz[0, 0, ...]` のような前提を除去する。
- `contract/record_adapter.py` へ shape adapter を集約する。

### Phase 2

- `discrete_action_space.py` の constructor / call site を Alpamayo 契約へ寄せる。
- `UnicycleAccelCurvatureActionSpace(k=...)` のような repo 固有 alias を除去する。
- `n_waypoints=...` を正本にする。

### Phase 3

- `unicycle_accel_curvature.py` を Alpamayo ほぼ verbatim に置き換える。
- 目標は `diff -u` 上で
  - import path
  - license header
  以外の差分をほぼ消すこと。

### Phase 4

- `action_space.py` と `utils.py` を Alpamayo に合わせる。
- 行順、docstring、関数名、クラス名まで合わせる。

### Phase 5

- `discrete_action_space.py` を Alpamayo に寄せる。
- 必要であれば repo 固有 adapter を別 helper に逃がす。
- この段階で tokenization 契約への影響を集中チェックする。

## 検証

### 低リスク検証

- `py_compile`
- entrypoint の `--help`
- `diff -u` で Alpamayo 対応ファイルとの差分確認

### 必須検証

- `DiscreteTrajectoryTokenizer` の encode/decode roundtrip
- 既存 `samples.jsonl` に対する token 列の不意な変化確認
- `stage1A` smoke train/eval
- `stage1B` inference shape
- `stage2` handoff inference

## 完了条件

- `action_space.py`, `utils.py`, `unicycle_accel_curvature.py`, `geometry/rotation.py` は Alpamayo とほぼ import path 差だけになる。
- `discrete_action_space.py` の差分は、repo 固有 token registry / tokenizer 接着の最小限に限定される。
- repo 固有の shape / record adapter は `contract/record_adapter.py` に閉じる。
- `stage1` / `stage2` が Alpamayo 契約前提で動く。

## 直近の注意点

- 一番壊しやすいのは `discrete_action_space.py`。
- ここを先に verbatim 化すると、`stage1A` の supervision と token id が静かに変わる可能性がある。
- そのため、pure numerics の一致より先に、adapter と call site の責務分離を済ませる。
