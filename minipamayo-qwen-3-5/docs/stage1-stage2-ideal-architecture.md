# Stage1 / Stage2 Ideal Architecture

## 目的

- `stage1` と `stage2` の token / prompt / sequence contract を 1 箇所に集約する
- `train`, `eval`, `inference` の entrypoint は保ちつつ、意味定義の重複をなくす
- Alpamayo の推論コードと比較するときに、どこが contract の正本かを明確にする
- `stage1A`, `stage1B`, `stage2` のあいだで token の意味がずれないようにする

## いまの問題

- `prompt.py`, `registry.py`, `quantizer.py`, `history.py`, `task_spec.py` が別々にあり、contract が散っている
- `stage1` と `stage2` がどの shared contract を再利用しているかが見えにくい
- `common/` という名前が広すぎて、contract の正本なのか utility なのか判別しづらい
- Alpamayo との整合を見るときに、token 意味論の参照先が複数ある

## 設計原則

1. token の意味を定義するコードは 1 箇所に置く
2. `stage1` と `stage2` は top-level の `contract/` を import して使うだけにする
3. `train`, `eval`, `inference` は execution layer として残す
4. `expert_cfm` の core は `stage1B` 専用の model layer として分離する
5. prompt 文面、special token、history token、future token、target layout を contract 層に寄せる

## 理想レイアウト

```text
src/minipamayo_qwen35/
  contract/
    prompt.py
    history_tokens.py
    trajectory_tokens.py
    sequence_layout.py
    task_spec.py
    __init__.py

  stage1/
    data/
      extract.py
      dataset.py
      unicycle_accel_curvature.py
      __init__.py

    vlm_ce/
      train/
        canonical.py
        runner.py
        profile.py
        experiments/
      eval/
        canonical.py
        runner.py
        experiments/
      inference/
        canonical.py
        helper.py
        alpamayo_style.py

    expert_cfm/
      core/
        action_space.py
        utils.py
        diffusion.py
        model.py
        common.py
        rotation.py
        __init__.py
      train/
        canonical.py
        runner.py
      eval/
        canonical.py
        runner.py
      inference/
        canonical.py
        runner.py

  stage2/
    reasoning_sft/
      data/
        dataset.py
        __init__.py
      train/
        canonical.py
        runner.py
        experiments/
      eval/
        canonical.py
        runner.py
        experiments/
      inference/
        canonical.py
        runner.py
      __init__.py
```

## 各ディレクトリの責務

### `contract`

ここを `stage1A`, `stage1B`, `stage2`, 推論 runner が共有する contract の正本にする。

- `prompt.py`
  - system prompt
  - user prompt
  - Alpamayo-like reasoning prompt
  - `<|cot_start|>`, `<|cot_end|>`, `<|traj_future_start|>`, `<|traj_future_end|>`
- `history_tokens.py`
  - `<|traj_history_start|>`, `<|traj_history|>`, `<|traj_history_end|>`
  - history quantization
  - placeholder replacement
- `trajectory_tokens.py`
  - `<i0> ...` など future trajectory token 語彙
  - discrete bin と token id の相互変換
  - `(a, kappa)` quantization
- `sequence_layout.py`
  - `stage1A` の target layout
  - `stage2` の target layout
  - 将来 `Reason + traj tokens` まで戻すときの列定義
- `task_spec.py`
  - canonical
  - steer_only
  - 将来の派生 experiment

### `stage1/data`

- raw dataset から canonical schema を作る
- history / future trajectory / action / reasoning source を保存する
- token の意味は持たない
  - token 化は `contract/` 側でやる

### `stage1/vlm_ce`

- `stage1A`
- VLM に discrete token 契約を注入する execution layer
- token 意味論は `contract/` から import するだけにする

### `stage1/expert_cfm/core`

- `stage1B`
- continuous trajectory decoder
- Alpamayo action-expert 寄せの model / diffusion / action space を置く
- prompt や token layout は持たない

### `stage2/reasoning_sft`

- `stage2`
- `contract/` の prompt / token / layout を使う
- `stage2` 自身は reasoning supervision と handoff 実行だけを持つ
- future token の意味を再定義しない

## contract の正本

理想的には、以下はすべて top-level の `contract/` に寄せる。

- prompt special tokens
- history special tokens
- future trajectory special tokens
- token id の語彙範囲
- `(a, kappa)` の bin 意味
- history quantization 方式
- `Reason -> trajectory` の sequence order
- handoff 境界 token

逆に、以下には contract を置かない。

- `stage1/vlm_ce/train`
- `stage1/vlm_ce/eval`
- `stage1/vlm_ce/inference`
- `stage2/reasoning_sft/train`
- `stage2/reasoning_sft/eval`
- `stage2/reasoning_sft/inference`

これらは execution layer だけにする。

## Stage1A / Stage2 の関係

理想的には `stage1A` と `stage2` は同じ `contract/` の token 契約を共有する。

つまり、

- `contract/` が future trajectory token の意味を定義する
- `stage1A` はその contract に従って離散 trajectory token を学習する
- `stage2` はその same token contract を target sequence の後半に使う
- `stage1B` は `stage2` の KV-cache / hidden state handoff を受けて連続 trajectory を出す

この関係にすると、

- `stage1A` で定義した token 契約
- `stage2` での reasoning-action consistency
- `stage1B` での continuous decoding

の役割分担が見えやすい。

## config の理想配置

```text
configs/
  stage1/
    data/
    vlm_ce/
      train/
      eval/
      inference/
      profile/
    expert_cfm/
      train/
      eval/
      inference/

  stage2/
    reasoning_sft/
      data/
      train/
      eval/
      inference/
```

## 実装時の注意

- directory refactor で token の意味を変えない
- `stage2` 側に prompt/token 定義を複製しない
- Alpamayo 比較で重要なものほど `contract/` に近づける
- `common/` という曖昧な名前は減らす

## 最終的な狙い

この構成にすると、質問に対して

- prompt 契約はどこか
- history token 契約はどこか
- trajectory token 契約はどこか
- `stage2` は何を再利用しているか

を 1 箇所ずつ示せるようになる。
