# Alpamayo 論文アラインメント修正方針

この文書は、`minipamayo_qwen35` を Alpamayo 論文の段階構成と入力契約にできるだけ揃えるための修正方針です。

目的は、今の簡略実装を少しずつ継ぎ足すことではなく、

- 何を canonical にするか
- どこを experiment に閉じ込めるか
- 論文との差分をどの順番で潰すか

を先に固定することです。

## 前提

- Alpamayo 論文の `Stage1` は `Sec. 5.1 Action Modality Injection`
- 論文上の `Stage1` には少なくとも次が含まれる
  - 離散 trajectory token の `cross-entropy`
  - separate action-expert の `flow matching`
- 論文上の入力は `o_image + o_egomotion`
- 論文上の future action representation は `64 waypoint × 2 = 128 token`
- `Reason` を Stage1 でどう supervision しているかは不明確

参照:

- [alpamayo-stage1-training-notes.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/alpamayo-stage1-training-notes.md)

## 目標

最終的に本流を次の構成にする。

1. 論文の `Stage1`
   - `Stage1A`: VLM の discrete action token CE
   - `Stage1B`: action-expert の CFM
2. 論文の `Stage2`
   - reasoning SFT
3. 論文の `Stage3`
   - RL / post-training

つまり、今の repo の

- `stage1`
- `stage2`
- `stage3`
- `stage4`

という naming には引きずられず、論文準拠の依存関係を優先する。

## 非目標

- `steer_only`
- smoke dataset
- `k=128` の VRAM 確認
- history を prompt text に埋めるだけの暫定策

これらは検証用としては有効だが、canonical pipeline にはしない。

## 現在の主なズレ

### 入力契約

- [ ] `stage1` が `o_image` しか実質使っていない
- [ ] `o_egomotion` / history tensor が train/eval/inference で未使用
- [ ] history placeholder token と実 tensor 注入の両立がない

### Stage1 の中身

- [ ] 現状の `stage1` は discrete token CE のみ
- [ ] 現状の flow matching decoder は `stage2` に分かれている
- [ ] その decoder は論文のような KV-cache conditioning ではなく final hidden states conditioning

### Reasoning の段階

- [ ] 現状の `stage2` は synthetic reasoning text を条件に使っている
- [ ] 論文の `Stage2` に相当する CoC reasoning SFT と naming / 契約が揃っていない

### 問題設定

- [ ] canonical path が `64 waypoint × 2 = 128 token` をまだ既定としていない
- [ ] `steer_only` や `kappa_only` が本流に近い位置に見えてしまう

## 採用する大方針

### 1. 論文本流と実験系を明確に分離する

- [ ] `canonical` は Alpamayo 論文寄りの本流だけに使う
- [ ] `steer_only`, `short_horizon`, `smoke` は `experiments/` に隔離する
- [ ] 本流に実験用 shortcut を混ぜない

### 2. Stage の意味を論文基準に合わせる

- [ ] 今の `stage2 decoder` 相当は、論文準拠では `Stage1B` 扱いに再編する
- [ ] reasoning SFT は論文準拠で `Stage2`
- [ ] RL / post-training は論文準拠で `Stage3`

### 3. history は text 埋め込みではなく構造化入力を本命にする

- [ ] canonical path では `ego_history_xyz / ego_history_rot` を使う
- [ ] prompt text だけに history を埋める方法は experiment 扱いにする

### 4. 画像 token budget は Alpamayo 寄りを canonical default にする

- [x] canonical default は `min_pixels=163840`, `max_pixels=196608`
- [x] 無制限画像 token は experiment 扱いにする

## 推奨する最終レイアウト

```text
src/minipamayo_qwen35/
  stage1/
    data/
    tokenization/
    vlm_ce/
      train/
      eval/
    expert_cfm/
      train/
      eval/
    inference/
  stage2/
    reasoning_sft/
      train/
      eval/
  stage3/
    post_training/
      train/
      eval/
```

config も mirror する。

```text
configs/
  stage1/
    data/
    vlm_ce/
      canonical/
      experiments/
    expert_cfm/
      canonical/
      experiments/
    inference/
      canonical/
      experiments/
  stage2/
    reasoning_sft/
      canonical/
      experiments/
  stage3/
    post_training/
      canonical/
      experiments/
```

## Canonical Stage1A: VLM CE

### 目的

- `o_image + o_egomotion (+ Reason slot)` を条件に
- `128` 個の離散 action token を CE で学習する

### 必須要件

- [ ] 入力に history を含める
- [ ] canonical target は `accel + kappa`
- [ ] canonical horizon は `64 waypoint`
- [ ] canonical timestep は `0.1s`
- [ ] canonical image budget は Alpamayo 相当

### 追加で必要な修正

- [ ] extractor に history 出力を追加する
  - `ego_history_xyz`
  - `ego_history_rot`
  - 必要なら local / ego-frame 版も保存する
- [ ] Stage1 dataset に history field を追加する
- [ ] processor / model 入力に history placeholder token を入れる
- [ ] placeholder を実 history token/tensor に fuse する経路を追加する
- [ ] prompt を Alpamayo 寄りの最小形に整理する

## Canonical Stage1B: action-expert CFM

### 目的

- Stage1A で得た VLM 表現を条件に
- separate action-expert を CFM で学習する

### 必須要件

- [ ] condition は `KV-cache` 寄りに揃える
- [ ] expert の loss は VLM に逆伝播させない
- [ ] target は continuous `(64, 2)` action tensor

### 現状との差

- [ ] 今の decoder は final hidden states conditioning
- [ ] 今の decoder は論文の expert と同型ではない

### 修正方針

- [ ] 今の `TrajectoryDecoder` をそのまま canonical にしない
- [ ] `stage1/expert_cfm` として separate action-expert を切り出す
- [ ] 可能なら Alpamayo 方式に近い
  - action in projection
  - expert transformer
  - action out projection
  - diffusion sampler
  へ寄せる
- [ ] まずは `detach(KV-cache)` 条件で expert を学習できる最小実装を作る

## Canonical Stage2: reasoning SFT

### 目的

- Stage1 で action generation capability を持ったモデルに対して
- CoC reasoning を SFT する

### 方針

- [ ] synthetic reasoning text を canonical から外す
- [ ] CoC dataset 契約を別に定義する
- [ ] Stage1 と同じ観測契約の上に reasoning supervision を足す

### 現状との差

- [ ] 今の `stage2` は論文の Stage2 ではなく、むしろ Stage1B 寄り
- [ ] 今の `stage3` が reasoning SFT に近いので、再配置が必要

## Canonical Stage3: RL / post-training

### 目的

- reasoning quality
- reasoning-action consistency
- trajectory quality

を RL 的に改善する

### 方針

- [ ] 現行 `stage4` を土台に再設計する
- [ ] reward contract を論文の整理に合わせる
- [ ] Stage2 完了モデルを起点にする

## 未確定事項

次は論文だけでは断定できないので、文書として明示しておく。

- [ ] Stage1A と Stage1B を完全に分けるか
- [ ] Stage1 の後半で CE と CFM を同時に持つ hybrid training にするか
- [ ] `Reason` を Stage1A でどう扱うか
  - 空スロット
  - 既存 backbone reasoning
  - 何らかの簡易 trace

現時点の採用方針は次の通り。

- [ ] まずは `Stage1A -> Stage1B` の段階分離で実装する
- [ ] joint / hybrid 化は後で明示的に比較実験する

理由:

- Alpamayo 論文は CE と CFM の両方を Stage1 に含めることは明示している
- ただし同一 minibatch joint training だったかまでは断定できない
- まずは保守的で実装可能な形を canonical にしたい

## 実装順序

### Phase 1: データ契約を直す

- [ ] extractor に history を追加する
- [ ] Stage1 dataset を history 対応にする
- [ ] history を含む canonical sample schema を固定する
- [ ] `k=64`, `dt=0.1` の canonical dataset config を作る

### Phase 2: Stage1A を作り直す

- [ ] current `stage1` を `stage1/vlm_ce` に整理する
- [ ] history token / tensor 注入経路を入れる
- [ ] canonical prompt を整理する
- [ ] canonical `128 token` path を確立する
- [ ] `steer_only` を完全に experiment へ押し込む

### Phase 3: Stage1B を作り直す

- [ ] current `stage2` を `stage1/expert_cfm` の観点で分解する
- [ ] hidden-state conditioning を KV-cache conditioning に寄せる
- [ ] action-expert の最小版を切る
- [ ] CFM train/eval/inference を canonical 化する

### Phase 4: Stage2 reasoning SFT を再配置する

- [ ] current `stage3` の役割を見直す
- [ ] CoC dataset 契約を定義する
- [ ] `stage2/reasoning_sft` を新設する

### Phase 5: Stage3 post-training を再配置する

- [ ] current `stage4` を論文基準の Stage3 に寄せる
- [ ] reward / rollout 契約を整理する

### Phase 6: config / docs / naming を揃える

- [ ] config path を新 layout に合わせる
- [ ] README のコマンド例を更新する
- [ ] 実験系 config を `experiments/` に寄せる
- [ ] 旧 naming を削除する

## 検証方針

### Stage1A

- [ ] `py_compile`
- [ ] config parse
- [ ] 1 batch の token 数確認
- [ ] Alpamayo 相当 image budget で VRAM 測定
- [ ] canonical 128 token で train/eval smoke

### Stage1B

- [ ] condition source が期待通りか確認
- [ ] VLM 側へ expert 勾配が戻っていないことを確認
- [ ] CFM loss が下がることを確認
- [ ] continuous trajectory rollout が成立することを確認

### Stage2 / Stage3

- [ ] reasoning token / action token の系列契約確認
- [ ] checkpoint metadata 契約確認
- [ ] inference path が train 契約と一致していることを確認

## コミット粒度

- [ ] `refactor: add canonical stage1 history schema`
- [ ] `refactor: move qwen35 canonical stage1 ce under vlm_ce`
- [ ] `feat: add qwen35 canonical stage1 expert cfm path`
- [ ] `refactor: realign qwen35 reasoning stage to alpamayo stage2`
- [ ] `refactor: realign qwen35 post-training stage to alpamayo stage3`
- [ ] `docs: add alpamayo alignment notes and commands`

## 要約

この修正方針の核心は次の 4 点です。

- [ ] canonical は Alpamayo 論文準拠に寄せる
- [ ] `o_egomotion` / history を本流に入れる
- [ ] 今の `stage2 decoder` 相当は論文準拠では Stage1B とみなす
- [ ] `steer_only` や smoke は experiment に閉じ込める

この方針に従うことで、

- 今の repo の stage naming のズレ
- 入力契約のズレ
- Stage1 の役割のズレ

をまとめて解消できる。
