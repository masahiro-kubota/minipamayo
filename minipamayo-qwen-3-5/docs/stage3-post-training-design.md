# Stage3 Post-Training Design

このメモは、Alpamayo-R1 論文の Sec. 5.3 を正本として、`minipamayo-qwen-3-5` に新しく実装する canonical `stage3` の設計方針をまとめたものです。

前提:
- 既存の `src/minipamayo_qwen35/stage3/post_training` は捨ててよい。
- `stage1` と `stage2` はすでに存在し、top-level shared core として `contract/`, `action_space/`, `models/`, `diffusion/`, `helper.py`, `config.py` がある。
- 目標は「今ある Stage3 を少し直す」ことではなく、「論文の RL-based post-training を実装できる構成を新しく切り直す」こと。

## 論文上の Stage3 要件

論文上の Stage3 は Sec. 5.3 `RL-based Post-Training` で、要点は次のとおりです。

- Stage2 SFT の目的は `log πθ(Reason, a | o)` であり、`Reason` だけではなく discrete action / trajectory token `a` まで含む。Sec. 5.2, Eq. (9)
- Stage3 は GRPO で rollout 群 `{τ_i}` を最適化する。Sec. 5.3.1, Eq. (10)
- 報酬は 3 系統ある。Sec. 5.3.2
  - reasoning quality reward
  - reasoning-action consistency reward
  - low-level trajectory quality reward
- RL 用データは、モデルの implicit preference と reward-induced distribution の disagreement が大きい sample を優先して curate する。Sec. 5.3.3

このため canonical Stage3 の policy contract は、論文どおり

```text
o -> Reason tokens + discrete action tokens -> continuous trajectory for reward/eval
```

であるべきです。

## いまの repo の前提とギャップ

### 1. current Stage2 はまだ論文どおりではない

現行 canonical `stage2` は reasoning text の生成と handoff 境界 token の生成を中心にしており、論文が前提にしている `Reason + discrete action tokens` を full sequence として SFT していません。

つまり、論文準拠の canonical Stage3 を実装する前提としては、

- Stage2 policy が `Reason + a` を autoregressive に出せること

が必要です。

この条件を満たしていない状態で RL を始めると、論文 Sec. 5.3 の `reasoning-action consistency` は token-level action policy ではなく、現在の handoff 実装への最適化に化けます。これは canonical Stage3 とはみなしません。

### 2. current dataset だけでは safety reward が足りない

現行の processed JSONL には

- image path
- ego history
- ego future
- canonical action
- reasoning text

はありますが、論文 Eq. (11) の

- `collision(x_pred)` を評価する surrounding obstacle state
- close encounter 相当の scene-level safety state

をそのまま計算できるだけの情報はありません。

そのため Stage3 は段階的に実装します。

- v0:
  - reasoning reward
  - consistency reward
  - trajectory imitation + jerk
- v1:
  - collision / close encounter / traffic-rule を追加
  - richer scene state か simulator hookup を導入

## canonical Stage3 の設計原則

1. Stage3 の trainable policy は Stage2 SFT policy そのものにする。
2. Stage1B continuous expert は RL の主 policy にしない。
3. Stage1B expert / diffusion / action-space は frozen rollout-decode-and-reward component として使う。
4. GRPO の reference policy は frozen Stage2 SFT checkpoint にする。
5. Stage3 独自の prompt / token 契約は作らない。
6. Stage3 が使う rollout contract は `stage2` と shared core から import する。
7. 既存 `stage3` の local pseudo-reward 実装は再利用しない。

## 採用する policy 分解

### trainable policy

- 役割:
  - `o -> Reason + a` を autoregressive に生成する
- 中身:
  - Stage2 SFT checkpoint から初期化した VLM policy
- 学習対象:
  - Stage2 SFT で trainable だった parameter set をそのまま引き継ぐ
  - Stage3 独自に freeze policy を変えない

### frozen reference policy

- 役割:
  - GRPO の KL regularization で使う `π_ref`
- 中身:
  - trainable policy を初期化したのと同じ Stage2 SFT checkpoint の frozen copy

### frozen continuous expert stack

- 役割:
  - discrete action rollout または handoff state から continuous trajectory を出す
  - trajectory reward と motion eval の基盤にする
- 中身:
  - `models/alpamayo_r1.py`
  - `models/action_expert.py`
  - `diffusion/action_expert.py`
  - `action_space/`
- 学習対象ではない

## rollout contract

Stage3 の 1 rollout は、論文準拠では次の 3 層で扱います。

### token rollout

```text
τ_i(token) = [Reason_i, a_i]
```

- `Reason_i`
  - chain-of-causality reasoning tokens
- `a_i`
  - Stage1/Stage2 と共有する discrete action tokens

### decoded motion rollout

```text
τ_i(motion) = x_i
```

- `x_i`
  - frozen expert stack から得る continuous trajectory

### reward rollout

```text
τ_i = {Reason_i, a_i, x_i, logprob_i, reward_i}
```

Stage3 の train loop は token rollout を最適化対象にし、motion rollout は reward evaluation 用に使います。

## GRPO objective

Stage3 の損失は論文 Sec. 5.3.1 に合わせて group-relative で扱います。

- 1 sample ごとに `K` 個 rollout する
- 各 rollout に scalar reward `r_i` を与える
- `A_i = r_i - mean(r)`
- relative advantage に応じて重み付けした policy gradient を計算する
- frozen reference policy との KL を入れる

実装上は次を分けます。

- rollout sampler
- reward aggregation
- GRPO loss
- optimization

これらを train runner に直書きしないことが重要です。

## reward 設計

### 1. reasoning quality reward

論文どおり、large reasoning model critic を前提にします。

- 入力:
  - visual observation
  - GT reasoning
  - predicted reasoning
- 出力:
  - scalar score

設計上は reward model を stage3 本体に埋め込まず、adapter interface にします。

```text
rewards/reasoning.py
```

想定 API:

```python
score_reasoning(sample, predicted_reasoning) -> float
```

### 2. reasoning-action consistency reward

論文どおり、reasoning から intent を parse し、trajectory から meta-action を作って longitudinal / lateral の両軸で一致判定します。

必要な処理:

- predicted reasoning -> closed decision set
- predicted trajectory -> meta-action sequence
- decision vs meta-action rule-based match

これは external LRM に依存しないので、canonical Stage3 の v0 から入れます。

```text
rewards/consistency.py
```

想定 API:

```python
score_consistency(sample, predicted_reasoning, predicted_traj) -> float
```

### 3. trajectory quality reward

論文 Eq. (11) では

- imitation L2
- collision penalty
- jerk penalty

を使います。

ただし current dataset の制約上、collision を full paper fidelity では計算できません。よって canonical Stage3 は二段階にします。

#### v0

- `L2(x_pred, x_expert)`
- `jerk(x_pred)`

#### v1

- `collision(x_pred)`
- close encounter / semantic safety

```text
rewards/trajectory.py
```

想定 API:

```python
score_trajectory(sample, predicted_traj) -> dict[str, float]
```

`dict` にして、reward aggregate 側で重み付けする方が後から safety term を追加しやすいです。

### 4. reward aggregation

論文では総 reward は 3 成分の和です。実装でも aggregate は独立 module にします。

```text
rewards/aggregate.py
```

役割:
- component reward の重み付け
- reward logging
- group relative advantage 用の scalar reward 生成

## data curation

論文 Sec. 5.3.3 に合わせて、Stage3 には dataset curation を別レイヤとして持たせます。

必要な処理:

1. current policy で rollout する
2. rollout logits から implicit distribution を作る
3. reward から Boltzmann distribution を作る
4. disagreement が大きい sample を優先する
5. random sample を混ぜて manifest を作る

ここは train loop と分けて、

```text
curation/disagreement.py
curation/build_manifest.py
```

のような preprocess path に置きます。

Stage3 train 本体は curated manifest を読むだけにします。

## 新しいディレクトリ構成

canonical Stage3 は次のレイアウトで作り直します。

```text
src/minipamayo_qwen35/stage3/
  post_training/
    __init__.py
    dataset.py
    common.py
    rollout/
      __init__.py
      bundle.py
      sampler.py
      parser.py
    rewards/
      __init__.py
      reasoning.py
      consistency.py
      trajectory.py
      aggregate.py
    curation/
      __init__.py
      disagreement.py
      build_manifest.py
    train/
      __init__.py
      __main__.py
      canonical.py
      runner.py
    eval/
      __init__.py
      __main__.py
      canonical.py
      runner.py
```

## 各ディレクトリの責務

### `dataset.py`

- curated manifest を読む
- base sample は Stage2 reasoning dataset contract を使う
- Stage3 固有の dataset split / filter だけを持つ

### `rollout/bundle.py`

- trainable policy
- frozen reference policy
- frozen expert stack

を 1 つの bundle にまとめる。

ここで現在 `stage2/reasoning_sft/wrapper.py` にある wrapper assembly を shared 化する。

理由:
- Stage3 が wrapper assembly の 2nd caller になるため
- `stage2` local helper のままでは cross-stage dependency が増えるため

### `rollout/sampler.py`

- grouped rollout generation
- logprob collection
- reference logprob collection
- decoded trajectory generation

### `rollout/parser.py`

- generated tokens から `Reason` と `a` を分ける
- invalid rollout の扱いを canonical にする

注意:
- 既存 `core/rollout_parser.py` は古い sequence contract 前提なので再利用しない

### `rewards/*`

- reasoning reward
- consistency reward
- trajectory reward
- aggregate

の 4 分割にする。

### `curation/*`

- disagreement-based data mining
- RL training manifest 生成

### `train/runner.py`

責務は以下に限定する。

- dataloader
- rollout sampler 呼び出し
- reward aggregation 呼び出し
- GRPO loss
- optimizer / checkpoint / logging

reward の中身や rollout parsing を train runner に埋め込まない。

### `eval/runner.py`

最低限これを出す。

- reasoning grading
- reasoning-action consistency
- ADE / FDE / minADE / minFDE
- close encounter rate

ただし close encounter は v1 で実装する。

## 既存 Stage3 で捨てるもの

以下は再利用しません。

- `stage3/post_training/train/runner.py`
  - local pseudo reward
  - manual rollout loop
  - old GRPO-like objective
- `stage3/post_training/eval/runner.py`
  - placeholder 実装
- `stage3/post_training/core/trajectory_decoder.py`
  - old hidden-state decoder
- `stage3/post_training/core/rollout_parser.py`
  - old sequence contract parser
- `stage3/post_training/*/experiments/synthetic_reasoning*`
  - canonical Stage3 ではない

理由:
- 論文の reward 設計と一致しない
- current shared Alpamayo wrapper path と一致しない
- current Stage2 / Stage1B contract と clean につながらない

## 実装前の必須前提

canonical Stage3 の実装を始める前に、次の 2 点を満たす必要があります。

### 1. Stage2 を `Reason + discrete action tokens` に戻す

これは必須です。現行の reasoning-only handoff contract のまま canonical Stage3 を実装しないこと。

方針:
- Stage2 canonical を paper-aligned contract に戻す
- もし一時的に bridge variant が必要なら、canonical とは別 path に切る

### 2. Stage3 用 reward data contract を決める

最低限 v0 では次を使えるようにする。

- GT reasoning
- GT action or GT trajectory
- predicted trajectory から meta-action を作るための ego state

v1 では safety reward 用に次を追加する。

- surrounding obstacle state
- collision / close encounter evaluation contract

## 実装順

### Phase 0: contract fix

- Stage2 canonical sequence を `Reason + a` に戻す
- Stage3 bundle が読む shared wrapper builder を shared 化する

### Phase 1: rollout-only skeleton

- `dataset.py`
- `rollout/bundle.py`
- `rollout/sampler.py`
- `rollout/parser.py`

を実装し、reward なしで grouped rollout ができることを確認する。

### Phase 2: reward v0

- `rewards/consistency.py`
- `rewards/trajectory.py` の `L2 + jerk`
- `rewards/aggregate.py`

を入れる。

### Phase 3: reasoning critic

- `rewards/reasoning.py`
- external LRM adapter

を入れる。

### Phase 4: train / eval

- GRPO train runner
- eval runner
- checkpoint / manifest / logging

### Phase 5: data curation

- disagreement mining
- random mix
- curated manifest pipeline

### Phase 6: safety reward

- collision / close encounter / traffic-rule

を追加する。

## 完了条件

canonical Stage3 が完成したとみなす条件は次です。

- 既存 `stage3/post_training` の local pseudo-reward 実装を参照していない
- trainable policy と frozen reference policy が分かれている
- frozen Stage1B expert stack を reward / decode 用に使っている
- grouped rollout + GRPO + KL が実装されている
- reasoning / consistency / trajectory reward が module 分離されている
- disagreement-based curation path がある
- Stage3 が `Reason + a` contract の上で動いている

## 非目標

この設計では、次は最初からやりません。

- current simplified Stage2 handoff contract のまま canonical Stage3 を作ること
- 既存 `stage3` の runner を延命すること
- Stage3 package の中で prompt / token contract を再定義すること
- fake collision reward で paper safety reward を名乗ること

## 参照

- Alpamayo-R1 paper, Sec. 5.2 `Eliciting Reasoning`, Eq. (9)
- Alpamayo-R1 paper, Sec. 5.3 `RL-based Post-Training`
- Alpamayo-R1 paper, Sec. 5.3.1 `Post-Training Algorithm`, Eq. (10)
- Alpamayo-R1 paper, Sec. 5.3.2 `Reward Model`, Eq. (11)
- Alpamayo-R1 paper, Sec. 5.3.3 `Post-Training Data Curation for Cost-Effective Training`
