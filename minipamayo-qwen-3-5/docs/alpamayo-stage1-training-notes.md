# Alpamayo 論文における Stage1 学習メモ

このメモは、Alpamayo 論文の `Sec. 5.1 Action Modality Injection` を中心に、

- 論文に明示されていること
- 論文だけでは断定できないこと
- 公開コードから補えること
- その結果、Stage1 をどう解釈するのが一番堅いか

を整理したものです。

ここでの目的は、現在の `minipamayo_qwen35` 実装を説明することではありません。  
あくまで「論文上の Stage1 は本来どういう学習段階だと読むのが自然か」を整理することが目的です。

## 対象範囲

- `Sec. 5.1 Action Modality Injection`
- 離散 trajectory token の `cross-entropy`
- action-expert の `flow matching`
- `o_egomotion` / history 入力の扱い

## 論文に明示されていること

### 1. Stage1 では VLM を離散 trajectory token で cross-entropy 学習する

論文は、Stage1 で action modality を離散 token として VLM に注入し、`Eq.(1)` の training token sequence に対して `cross-entropy loss` で学習すると明記しています。

参照:

- [alpamayo-paper.txt](/home/masa/minipamayo/minipamayo/paper/alpamayo/alpamayo-paper.txt#L1170)
- [alpamayo-paper.txt](/home/masa/minipamayo/minipamayo/paper/alpamayo/alpamayo-paper.txt#L291)

`Eq.(1)` の系列は次です。

```text
[o_image, o_egomotion, Reason, τ]
```

つまり論文上の Stage1 は、画像だけではなく `o_egomotion` も入力に含む前提です。

### 2. 未来 trajectory は 128 個の離散 action token で表される

論文では future trajectory が

- 64 waypoint
- 各 waypoint に 2 つの量子化値
  - acceleration `a_i`
  - curvature `kappa_i`

で表されると書かれています。

したがって未来 trajectory は

- `64 × 2 = 128 token`

です。

参照:

- [alpamayo-paper.txt](/home/masa/minipamayo/minipamayo/paper/alpamayo/alpamayo-paper.txt#L1173)
- [alpamayo-paper.txt](/home/masa/minipamayo/minipamayo/paper/alpamayo/alpamayo-paper.txt#L299)

### 3. Stage1 では別モジュールの action-expert も flow matching で学習する

論文は同じ `Sec. 5.1` の中で、離散 token 学習に加えて、別の action-expert を `flow matching` で学習すると明記しています。

参照:

- [alpamayo-paper.txt](/home/masa/minipamayo/minipamayo/paper/alpamayo/alpamayo-paper.txt#L1187)

論文の記述では、

- action-expert は `[o_image, o_egomotion, Reason]` から得た VLM の `KV-cache`
- noisy action `a_t`

を受け取り、

- ベクトル場 `v_Theta(a_t, o, Reason)`

を予測します。

そして loss は vanilla conditional flow matching loss です。

参照:

- [alpamayo-paper.txt](/home/masa/minipamayo/minipamayo/paper/alpamayo/alpamayo-paper.txt#L1191)
- [alpamayo-paper.txt](/home/masa/minipamayo/minipamayo/paper/alpamayo/alpamayo-paper.txt#L1195)

### 4. action-expert の勾配は VLM に戻さない

論文は、expert 学習時には VLM から得た `KV-cache` に `stop-gradient` をかけると明記しています。

つまり、

- expert は VLM の状態に条件付けされる
- ただし expert の loss で VLM を更新しない

という構造です。

参照:

- [alpamayo-paper.txt](/home/masa/minipamayo/minipamayo/paper/alpamayo/alpamayo-paper.txt#L1205)

### 5. Stage2 に入る時点で、すでに action generation capability を持っている前提

`Sec. 5.2` は

- `Having established a VLA with action generation capabilities in Sec. 5.1`

で始まります。

つまり論文は、Stage1 の終了時点で少なくとも

- 離散 action token の出力能力
- continuous action / trajectory を出す expert 側の能力

を備えた状態になっている前提です。

参照:

- [alpamayo-paper.txt](/home/masa/minipamayo/minipamayo/paper/alpamayo/alpamayo-paper.txt#L1219)

## 論文だけでは明示されていないこと

論文だけでは、次の点は断定できません。

- `L = L_ce + lambda * L_cfm` のような joint loss で同時最適化したのか
- Stage1 をさらに 2 つの sub-stage に分けたのか
- CE と CFM を同一 minibatch loop で回したのか、別 loop なのか
- Stage1 で `Reason` に対して具体的にどのような supervision を与えたのか

このあたりは paper text には書き切られていません。

## 公開コードから補えること

公開コードを見ると、history と action-expert は補助的な要素ではなく、かなり中心的な要素として実装されています。

### 1. history は構造化された model input として入っている

推論コードでは、モデルに次が渡されています。

- `ego_history_xyz`
- `ego_history_rot`

参照:

- [test_inference.py](/home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/test_inference.py#L48)

さらに model 側には history trajectory を token 化する処理があります。

参照:

- [base_model.py](/home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/models/base_model.py#L83)

また、chat template の user prompt にも history placeholder token が入ります。

参照:

- [helper.py](/home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/helper.py#L36)

### 2. action-expert は VLM とは別モジュール

公開モデルは

- VLM
- separate `expert`
- `action_space`
- `diffusion`

を個別に持っています。

参照:

- [alpamayo_r1.py](/home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/models/alpamayo_r1.py#L69)
- [alpamayo_r1.py](/home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/models/alpamayo_r1.py#L90)

### 3. 推論時は VLM rollout のあとに expert を動かす

公開コードの推論パスは概ねこうです。

1. VLM を autoregressive rollout
2. VLM の `KV-cache` / prompt cache を得る
3. その cache を条件として expert を動かす
4. diffusion を action space 上で回す
5. continuous action を trajectory に戻す

参照:

- [alpamayo_r1.py](/home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/models/alpamayo_r1.py#L150)
- [alpamayo_r1.py](/home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/models/alpamayo_r1.py#L252)

### 4. continuous expert 側は `(64, 2)` の action tensor を扱う

公開コードの action space は、128 個の scalar token を直接 autoregressive に出すのではなく、

- 64 time step
- 2 channel (`accel`, `curvature`)

の continuous action tensor を扱っています。

参照:

- [unicycle_accel_curvature.py](/home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/action_space/unicycle_accel_curvature.py#L91)

これは論文の説明とも整合します。

- VLM 側では 128 個の離散 token で supervision
- expert 側では `64 × 2` の continuous action を生成

## 一番堅い Stage1 解釈

論文と公開コードを合わせて読むと、Stage1 の一番堅い解釈は次です。

### Stage1A: VLM の action-modality injection

VLM に対して、次のような系列を `cross-entropy` で学習する。

- image observations
- egomotion history
- reasoning slot
- 128 個の discrete future action token

この段階の目的は、

- VLM が discrete action token を autoregressive に扱えるようにすること

です。

### Stage1B: action-expert の flow matching 学習

別の action-expert に対して、conditional flow matching を学習する。

- 条件: VLM の `[o_image, o_egomotion, Reason]` 由来の `KV-cache`
- 入力: noisy action `a_t`
- 目標: vector field `u(a_t | a)`

このとき `KV-cache` には `stop-gradient` をかけます。

この段階の目的は、

- VLM の文脈条件付きで
- physically feasible な continuous action / trajectory を出せる expert を得ること

です。

## なぜ「Stage1 を 2 sub-stage と読む」のが安全か

論文は

- `VLM は CE で学習する`
- `expert は CFM で学習する`

ことは明記していますが、

- `L_stage1 = L_ce + lambda * L_cfm`

のような combined loss は書いていません。

したがって、論文に忠実に寄せるなら、

- Stage1A: CE
- Stage1B: CFM

の 2 段階として理解する方が安全です。

これは「NVIDIA が必ずそう実装している」と証明するものではありません。  
ただし、paper text に書いてある範囲から無理なく引ける解釈としてはこれが最も保守的です。

## `Reason` を Stage1 でどう扱っているかはまだ曖昧

ここは依然として曖昧です。

事実としては、

- `Eq.(1)` には `Reason` が入っている
- しかし `Stage2` で reasoning capability を強化すると明言している

参照:

- [alpamayo-paper.txt](/home/masa/minipamayo/minipamayo/paper/alpamayo/alpamayo-paper.txt#L291)
- [alpamayo-paper.txt](/home/masa/minipamayo/minipamayo/paper/alpamayo/alpamayo-paper.txt#L1219)

なので、一番自然な読みは次です。

- Stage1 の主目的は reasoning 学習そのものではない
- Stage1 は action grounding が中心
- Stage2 で初めて reasoning を構造化・高品質化する

つまり、`Reason` が系列上存在していても、Stage1 を「reasoning 学習の本番」とみなすのはたぶん違う、ということです。

## 実務的な結論

論文に寄せるなら Stage1 は次のように扱うべきです。

- 画像だけの予備実験ではない
- discrete token だけの簡易段階ではない
- 後で捨てる smoke stage ではない

むしろ Stage1 は、

- image + egomotion history を入力に使い
- VLM に discrete action token を学習させ
- さらに continuous decoder として action-expert を学習し
- Stage2 に入る時点で `action generation capability` を成立させる

段階だと読むのが自然です。

## 短い要約

- Stage1 には `discrete token CE` と `action-expert CFM` の両方が入っている
- Stage1 の入力には `o_egomotion` も含まれる
- Stage1 が終わった時点で `action generation capability` を持っている前提
- 論文は CE と CFM を joint loss でやったのか、2 sub-stage なのかを明記していない
- 一番安全な解釈は:
  - Stage1A: VLM の CE
  - Stage1B: expert の CFM
