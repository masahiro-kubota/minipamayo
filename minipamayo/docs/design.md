# MiniPamayo 設計書 v0.3

## 1. 目的

- Alpamayo-R1（VLA：Vision-Language-Action）の**技術的理解**を目的に、同種の構成要素と学習段取りを RTX 4090 単体で再現する
- 目標は「SOTA 性能」ではなく、**学習が回り、Alpamayo の主要コンセプトを段階的に体験する**こと
- 対象コンセプト: 制御ベースアクション表現、Dual Representation（離散トークン + Flow）、構造化推論（CoC）、RL ポストトレーニング

### 1.1 コンポーネントサイジングの検討（RTX 4090 VRAM 制約）

Alpamayo 0.5B 構成（DINOv2 + Qwen2.5-0.5B）を RTX 4090 単体で全パイプライン学習できるかを検討した結果、**可能**と判断した。以下にその根拠を示す。

#### 全パイプラインの最大 VRAM 見積もり

bf16 学習時の固定コストは **N × 12 bytes**（パラメータ 2B + AdamW 1st moment 4B + 2nd moment 4B + 勾配 2B）で支配される。

**構成: DINOv2 ViT-B/14 (86M) + Adapter (~2M) + Qwen2.5-0.5B (494M) = 582M**

| Stage | Trainable | 固定コスト (N×12) | Activation + overhead | 合計 |
|---|---|---|---|---|
| VLM Stage 1 (Adapter のみ) | 2M | 0.02 GB | ~3 GB | **~3 GB** |
| VLM Stage 2 / ドメイン SFT (全解凍) | 582M | 6.98 GB | ~3 GB | **~10 GB** |
| **Cosmos RL (全解凍 + ref policy)** | 582M + ref 582M×2 | 6.98 + 1.16 GB | ~3 GB | **~11 GB** |
| MiniPamayo Stage 0-1 (全解凍) | 582M + head <1M | 6.98 GB | ~3 GB | **~10 GB** |
| MiniPamayo Stage 2 (Flow 学習) | Traj. Decoder ~150M | 1.80 GB + VLM推論 1.16 GB | ~3 GB | **~6 GB** |
| MiniPamayo Stage 3 (CoC SFT, 全解凍) | 582M | 6.98 GB | ~4 GB | **~11 GB** |
| **MiniPamayo Stage 4 RL (LLM + ref)** | LLM 494M + ref 582M×2 | 5.93 + 1.16 GB | ~3 GB | **~10 GB** |

**最大 VRAM: ~11 GB（Cosmos RL / MiniPamayo Stage 3 時）。RTX 4090 (24 GB) に対して ~13 GB の余裕。**

#### SmolLM2-360M からの変更理由

当初 SmolLM2-360M (362M) を採用していたが、VRAM 計算の結果 Qwen2.5-0.5B (494M) でも全パイプラインが収まることが判明。Qwen2.5-0.5B に変更する利点:

1. **Alpamayo 0.5B と同一の LLM**: アーキテクチャ再現の忠実度が向上
2. **GQA (2 KV heads)**: SmolLM2 の MHA (15 heads) より KV cache が効率的。RL の rollout 生成に有利
3. **より強力な事前学習**: 言語理解力が高く、VLM 学習や CoC 推論に有利

同様に DINOv2 も ViT-S/14 (21M, hidden=384) から ViT-B/14 (86M, hidden=768) に変更。hidden dim が LLM の 896 に近く（768→896 = 1.2倍）、Adapter の射影ギャップが小さくなる。

---

## 2. 制約と前提

| 項目 | 値 |
|---|---|
| GPU | RTX 4090（24 GB VRAM） |
| 凍結 | **Stage ごとに制御**（詳細は §3.7 参照） |
| カメラ | **1 台** |
| 解像度 | 224×224（必要に応じて縮小可） |
| 視覚トークン | **16〜32 tokens / image** |

---

## 3. アーキテクチャ

```
┌──────────┐     ┌──────────────────┐     ┌──────────┐     ┌──────────────┐
│  Camera   │────▶│  DINOv2 ViT-B/14 │────▶│  Adapter  │────▶│ Qwen2.5-0.5B │
│ (224×224) │     │  (Vision Enc.)   │     │ (768→896) │     │    (LLM)     │
└──────────┘     └──────────────────┘     └──────────┘     └──────┬───────┘
                                                                  │
                                                     ┌────────────┴────────────┐
                                                     │                         │
                                              ┌──────▼──────┐          ┌──────▼──────┐
                                              │ Action Head  │          │ Traj Decoder │
                                              │ (MLP回帰)    │          │ (Flow Match) │
                                              │ [Stage 0]    │          │ [Stage 2]    │
                                              └─────────────┘          └─────────────┘
```

### 3.1 Vision Encoder — DINOv2 ViT-B/14

- モデル: `facebook/dinov2-base`（ViT-B/14、86M params）
- 入力: RGB 224×224
- 出力: パッチ特徴 (16×16)=256 patches × 768 dim
- hidden dim 768 は Qwen2.5-0.5B の 896 に近く（1.2倍）、Adapter の射影ギャップが小さい
- Stage 0: trainable / Stage 2: frozen（§3.7 参照）

### 3.2 Vision → LLM Adapter（視覚トークン圧縮）

- 入力: DINOv2 パッチ特徴 (256 × 768)
- 出力: **N_vis tokens (16〜32) × d_llm (896)**
- 方式（実装容易性で選択、後で置換可）:

| 優先度 | 方式 | 概要 |
|---|---|---|
| 1 | Cross-Attention Pooling | learnable query 16/32 個で DINOパッチに attend |
| 2 | MLP + Attention Pool | 簡易版 |
| 3 | 平均 Pool + Linear | 最小実装（fail-fast 用） |

**初期実装**: まず方式 3（平均 Pool + Linear）で全パイプラインを通し、後で方式 1 に置き換える。

### 3.3 Language Model — Qwen2.5-0.5B

- モデル: `Qwen/Qwen2.5-0.5B`（494M params）
- アーキテクチャ: decoder-only Transformer
  - hidden_dim: 896
  - num_layers: 24
  - num_attention_heads: 14
  - num_kv_heads: 2（GQA — KV cache が効率的、RL rollout に有利）
  - vocab_size: 151,936
- Alpamayo 0.5B と**同一の LLM** — アーキテクチャ再現の忠実度が高い
- Stage 0: trainable / Stage 2: frozen（§3.7 参照）
- **必須要件**: KV-cache を取り出せること（Stage 2 Flow の条件付けに使用）

### 3.4 VLM 構築 + ドメイン SFT（前段パイプライン）

MiniPamayo の行動予測（Stage 0〜4）の前に、DINOv2 + Adapter + Qwen2.5-0.5B を VLM として完成させ、運転ドメインの知識を注入する。この前段パイプラインは 2 つのサブプロジェクトに分離している:

#### Qwen2.5-VL Mini（汎用 VLM 構築）

Qwen2.5-VL / LLaVA の学習パイプラインに倣い、DINOv2 ViT-B/14 + Qwen2.5-0.5B から**汎用 VLM**を構築する。

1. Feature Alignment: Adapter のみ学習（画像キャプションペアで視覚-言語アライメント）
2. Visual Instruction Tuning: Adapter + LLM 学習（視覚 QA で VLM 能力を獲得）

詳細は [Qwen2.5-VL Mini 設計書](qwen-vl-mini/design.md) を参照。

#### Cosmos Reason Mini（運転ドメイン特化）

Cosmos-Reason の学習パイプラインに倣い、Qwen2.5-VL Mini の上に**運転ドメインの Physical AI 知識**を注入する。

1. Physical AI SFT: 運転シーン QA で SFT
2. Physical AI RL: MCQ ベースの GRPO で推論品質向上

詳細は [Cosmos Reason Mini 設計書](cosmos-reason-mini/design.md) を参照。

#### 全体の流れ

```
Qwen2.5-VL Mini → Cosmos Reason Mini → MiniPamayo Stage 0
（汎用 VLM）      （運転ドメイン特化）  （行動予測）
```

学習済み重み（Vision Encoder + Adapter + LLM）が MiniPamayo Stage 0 の初期値となる。

### 3.5 アクション表現

Alpamayo 論文に倣い、生の位置 (x, y) ではなく**制御ベース表現**を採用する。

#### 制御ベース表現（ユニサイクルダイナミクス）

Alpamayo では raw position waypoint はセンサノイズの影響を受けやすく、学習の収束が悪化するとの知見がある（論文 §3.2.2）。代わりに、ユニサイクルダイナミクスに基づく制御入力を予測する:

- 制御入力: **a = {(aᵢ, κᵢ)}** — 各タイムステップの加速度 `a` と曲率 `κ`
- ダイナミクス: Euler 積分で (x, y, θ, v) の軌道に変換
- **MiniPamayo での設定**:
  - 予測ホライズン: **6.4秒**（64 waypoints @ 10Hz）— Alpamayo と同一
  - 制御入力: 64 × 2 = **128 値**
  - GT 制御列: ego pose の軌道から最小二乗法（Tikhonov 正則化）で逆算

```
x_{i+1} = x_i + Δt/2 * (v_i cos θ_i + v_{i+1} cos θ_{i+1})
y_{i+1} = y_i + Δt/2 * (v_i sin θ_i + v_{i+1} sin θ_{i+1})
θ_{i+1} = θ_i + Δt * κ_i * v_i + Δt²/2 * κ_i * a_i
v_{i+1} = v_i + Δt * a_i
```

**段階的導入**: 初期実装（fail-fast）では `[steer, throttle]`（2D）で動作確認し、パイプラインが安定した後に制御ベース表現へ移行してもよい。

### 3.6 Action Head（3段階）

#### Stage A — MLP 回帰ヘッド（Stage 0 で使用）

- 入力: LLM 最終層 hidden state
- 出力: 制御入力列 `{(aᵢ, κᵢ)}` — (K, 2)（初期は `[steer, throttle]` (2,) でも可）
- Loss: Huber / L2

#### Stage B — 離散トークン化 + MLP 回帰の Dual Representation（Stage 1 で使用）

Alpamayo の核心的設計の一つ。**学習時は離散トークン、推論時は Flow Matching** を使う Dual Representation 戦略:

- **学習時の離散表現**:
  - 制御入力 (aᵢ, κᵢ) を所定の範囲で均一量子化し、離散トークン化
  - 64 waypoints × 2 values = **128 離散トークン** を LLM の語彙に特殊トークンとして追加
  - LLM が推論トークン（将来の CoC）とアクショントークンを**同じ自己回帰フレームワーク**で生成
  - Loss: cross-entropy（LLM の標準的な next-token prediction）
- **Dual Representation のメリット**（Alpamayo 論文 §5.1）:
  1. 推論（Reasoning）と軌道が共通のトークン空間を共有し、密な結合が可能
  2. RL ポストトレーニング時に離散トークンを通じて直接勾配を流せる
  3. 離散表現が車両ダイナミクスの強い教師信号となる
  4. Flow Matching による推論時デコードは 128 トークンの自己回帰生成より高速

#### Stage C — Trajectory Decoder / Flow Matching ヘッド（Stage 2 で使用）

Alpamayo の「Trajectory Decoder」に相当。VLM の出力を条件として、Flow Matching で連続軌道を生成する。

- **サイジング**: ~150M params
  - Alpamayo 10B 版は LLM 7B に対して Trajectory Decoder 2B（約30%）。同比率で LLM 494M の ~30% = **~150M**
  - Stage 2 では VLM は frozen（推論のみ、~1.16 GB）のため、150M の Decoder 学習には ~1.8 GB（150M×12）+ activation ~3 GB = **~6 GB**。VRAM に十分な余裕あり
- 条件付け: **LLM KV-cache（hidden state シーケンス）→ cross-attention**（Alpamayo §5.1 準拠）
  - Decoder の各ブロックで、action token を query、VLM hidden state シーケンスを key/value として cross-attention
  - AdaLN でタイムステップ条件付けを self-attention、cross-attention、MLP に適用
- Flow network: 小さな Transformer（LLM と同じ attention head 数・次元、ただし hidden/MLP は小さい）
- 入力: noisy action aₜ + VLM hidden state sequence（stop-gradient）
- 出力: velocity field vΘ(aₜ, o) の予測 → Euler 積分で連続軌道を生成
- Loss: **CFM loss** — Gaussian OT path `aₜ = t·a + (1-t)·ε`, target `u = a - ε`, **t ~ Beta(α, β)**（shifted beta distribution、Alpamayo §5.2）
- Flow steps: 推論時 10（δt = 0.1）
- **勾配制御**: Trajectory Decoder 学習時、VLM（Vision Encoder + LLM）からの KV-cache / hidden states には **stop-gradient** を適用し、Decoder の勾配が VLM 側に逆伝播しないようにする（Alpamayo 論文 §5.1 と同様）

### 3.7 Stage ごとの勾配制御方針

Alpamayo 論文に倣い、学習 Stage ごとにモジュールの trainable / frozen を切り替える。

| Stage | Vision Encoder | Adapter | LLM | Action Head (MLP) | Flow Head |
|---|---|---|---|---|---|
| **ドメイン SFT** | trainable | trainable | trainable | — | — |
| **Stage 0（回帰）** | trainable | trainable | trainable | trainable | — |
| **Stage 1（離散トークン）** | trainable | trainable | trainable | — | — |
| **Stage 2（Flow）** | frozen（stop-grad） | frozen（stop-grad） | frozen（stop-grad） | — | trainable |
| **Stage 3（推論 SFT）** | trainable | trainable | trainable | — | frozen |
| **Stage 4（RL）** | frozen | frozen | trainable | — | frozen |

- **ドメイン SFT**: Vision Encoder + Adapter + LLM すべて trainable。画像→テキスト記述の SFT で、Adapter のマッピングと LLM の視覚理解を同時に獲得する（§3.4）。
- **Stage 0**: ドメイン SFT の重みを初期値として使用。全モジュール trainable。勾配が DINO → Adapter → LLM → Action Head の全経路に流れることを確認する。
- **Stage 1**: LLM の語彙に離散アクショントークンを追加し、cross-entropy loss で学習。全モジュール trainable。
- **Stage 2**: Stage 0/1 で学習済みの Vision Encoder + Adapter + LLM の重みを初期値として使用し、これらには **stop-gradient を適用**。Flow Head のみを学習する。これにより前段で獲得した表現を壊さずに Flow の学習を安定させる。
  - Alpamayo 論文では Action Expert 学習時に VLM の KV-cache に stop-gradient を適用している（§5.1: "we apply a stop-gradient to the KV-cache produced by the VLM to prevent gradients from the expert back-propagating into the VLM weights"）。
  - MiniPamayo では LLM hidden state sequence（`.detach()`）を Trajectory Decoder の cross-attention に渡す方式で同等の勾配制御を実現。
- **Stage 3**: CoC 推論データで SFT。推論トークン + 離散軌道トークンの joint 学習。
- **Stage 4**: RL ポストトレーニング。LLM のみ trainable（他は frozen）。KL 正則化で SFT モデルからの逸脱を防ぐ。
- **（発展）**: 各 Stage の fine-tune 後、全体を小さい学習率で end-to-end fine-tune することも検討可。

### 3.8 Reasoning — Chain of Causation (CoC)

Alpamayo-R1 の核心的貢献。構造化された推論を LLM に生成させ、推論と行動の因果整合性を確保する。

#### CoC の構造

Alpamayo では、自由形式の CoT（Chain of Thought）ではなく、以下の3要素で構造化された推論を生成する:

1. **Driving Decision（運転意思決定）**: 閉じた集合から選択
   - 縦方向: set speed tracking, lead obstacle following, speed adaptation, gap-searching, acceleration for passing, yield, stop for static constraints
   - 横方向: lane keeping, merge/split, out-of-lane nudge, in-lane nudge, lane change, pull-over, turn, lateral maneuver abort
2. **Critical Components（重要な構成要素）**: 意思決定に直接影響する因果要因
   - 例: 信号状態、先行車、歩行者、車線構成、ルート指示等
3. **Composed CoC Trace（推論トレース）**: 上記を結ぶ自然言語の因果推論

#### MiniPamayo での簡略化

- **閉じた意思決定集合**: Alpamayo の完全なリストから、使用するデータセットに関連するサブセットのみ定義（2軸で分類）
  - 縦方向: `{go_straight, follow_lead, stop, yield}`
  - 横方向: `{lane_keeping, turn_left, turn_right, lane_change_left, lane_change_right}`
- **推論トレースの生成**: LLM が `[visual_tokens] → [CoC reasoning tokens] → [action tokens]` のシーケンスを自己回帰的に生成
- **推論データの作成**: 教師 VLM（例: GPT-4o）を使って、学習データに CoC アノテーションを付与する auto-labeling パイプライン
- **因果混乱の防止**: Alpamayo に倣い、過去の観測（history window）のみから因果要因を特定する

#### 学習シーケンス

```
入力:  [visual_tokens, egomotion_tokens]
出力:  [CoC_reasoning_tokens, meta_action_tokens, trajectory_tokens]
       ├── 自然言語推論 ──┤├── 意思決定 ──┤├── 制御入力列 ──┤
```

### 3.9 RL ポストトレーニング

SFT（§3.8）だけでは以下の問題が残る（Alpamayo 論文 §5.2）:
- データバイアスによる不完全な因果推論の学習
- 推論と行動の不一致（「止まる」と言いながら止まらない等）
- 視覚的根拠のない推論のハルシネーション

これを改善するため、RL ポストトレーニングを行う。

#### アルゴリズム: GRPO（Alpamayo §5.3.2 準拠）

Alpamayo に倣い **GRPO（Group Relative Policy Optimization）** を採用:
- モデルから K 個のロールアウトをサンプリング
- Advantage: `A_i = r_i - r̄`（グループ平均、std 正規化なし）
- **Softmax-weighted policy gradient**: `L = -Σ softmax(β·A_i) × (log π_θ(τ_i) - λ_KL · KL)`
- KL 正則化で SFT モデル（reference policy）からの逸脱を防止
- PPO-style の clipping や multi-step 更新は使用しない

#### 報酬信号（3要素）

1. **推論品質 (r_reason)**: LRM（大規模推論モデル）による 0-5 スケール採点
   - 行動一貫性: 推論が正しい運転行動を記述しているか
   - 因果推論品質: 因果要因が正しく特定されているか
   - MiniPamayo では外部 LLM API（例: Claude API）で代替可
2. **CoC-Action 一貫性 (r_consistency)**: バイナリ報酬
   - 予測軌道から meta-action を抽出し、推論トレースの意図と照合
   - 一致 → 1、不一致 → 0
3. **低レベル軌道品質 (r_traj)**（ペナルティ形式、Alpamayo §5.3.2）:
   - `r_traj = -(λ_L2·||x_pred-x_expert||²_2 + λ_coll·I[collision] + λ_jerk·J(x_pred))`
   - L2 距離ペナルティ、バイナリ衝突指示関数、ジャーク抑制

#### MiniPamayo での実装方針

- VRAM 制約上、ロールアウト数は小さく（K=4〜8）
- 推論品質報酬は外部 API 呼び出しとしてオフライン計算
- 軌道品質報酬はオンラインで計算可能（L2 + ジャーク）
- 衝突判定は、データセットが提供する周囲物体情報に依存（nuScenes なら利用可）

### 3.10 効率的ビジョンエンコーディング

MiniPamayo は単一カメラのため Alpamayo のマルチカメラ効率化は直接不要だが、Alpamayo の Vision Encoding 戦略を理解・記録しておく。

#### Alpamayo のアプローチ

| 方式 | 入力 | トークン数 / image | 圧縮率 | 備考 |
|---|---|---|---|---|
| Single-Image | 1枚ずつ | 160（448×280） | 1× | デフォルト |
| **Triplane** | マルチカメラ同時 | ~41 | 3.9× | 3D 誘導バイアス、カメラ数と解像度に非依存 |
| **Flex** | マルチカメラ・マルチタイムステップ | 可変 | 最大 20× | self-attention + 固定 query で圧縮 |

#### MiniPamayo への示唆

- 現在の DINOv2 ViT-B/14（256 patches）→ Adapter（16~32 tokens）の圧縮は、Alpamayo の Flex に近い思想
- 将来マルチカメラに拡張する場合は Triplane が候補
- Single-Image Tokenization の 2× bilinear downsampling は、MiniPamayo の Adapter 前段に導入可能（256 → 64 patches → Adapter → 16 tokens）

---

## 4. 入出力仕様

### 4.1 入力

| 入力 | 形状 | 備考 |
|---|---|---|
| 画像 | RGB 224×224 | カメラ 1 台 |
| egomotion 履歴 | (T, D) — speed, yaw rate 等 | 制御ベース表現の初期状態に必要 |
| テキスト | なし or CoC 推論プロンプト | Stage 3 以降で使用 |
| (任意) ルート情報 | ナビゲーション指示 | Alpamayo では性能向上に寄与 |

### 4.2 出力

| Stage | 出力 | 形状 | 備考 |
|---|---|---|---|
| Stage 0（回帰 fail-fast） | steer, throttle | (2,) | 最小検証用 |
| Stage 0（回帰） | 制御入力列 (a, κ) | (64, 2) | 制御ベース表現、6.4秒 @ 10Hz |
| Stage 1（離散トークン） | 離散アクショントークン | (128,) tokens | 64 waypoints × 2 values |
| Stage 2（Flow） | 制御入力列 (a, κ) | (K, 2) — Flow で生成 | 連続・マルチモーダル |
| Stage 3（推論 SFT） | CoC 推論 + 離散トークン | テキスト + (2K,) tokens | 自己回帰生成 |
| Stage 4（RL） | CoC 推論 + 離散トークン | テキスト + (2K,) tokens | RL で品質向上 |

---

## 5. VRAM 見積もり（概算）

### 5.1 コンポーネント別パラメータ

| コンポーネント | パラメータ数 | bf16 サイズ | 備考 |
|---|---|---|---|
| DINOv2 ViT-B/14 | 86M | ~172 MB | hidden=768, 12 layers |
| Adapter | ~2M | ~4 MB | 方式による |
| Qwen2.5-0.5B | 494M | ~988 MB | hidden=896, 24 layers, GQA |
| Action Head (MLP) | <1M | ~2 MB | |
| Trajectory Decoder | ~150M | ~300 MB | §3.6 参照。LLM の ~30% |
| **VLM 合計** | **~582M** | **~1.16 GB** | |
| **全体合計** | **~730M** | **~1.46 GB** | |

### 5.2 学習時 VRAM 計算式

bf16 学習時の固定コスト: **N × 12 bytes**（パラメータ 2B + AdamW 1st moment 4B + 2nd moment 4B + 勾配 2B）

### 5.3 Stage 別 VRAM 見積もり

§1.1 の VRAM テーブルを再掲（詳細な導出は §1.1 を参照）:

| Stage | Trainable | 固定コスト (N×12) | Activation + overhead | 合計 |
|---|---|---|---|---|
| VLM Stage 1 (Adapter のみ) | 2M | 0.02 GB | ~3 GB | **~3 GB** |
| VLM Stage 2 / ドメイン SFT (全解凍) | 582M | 6.98 GB | ~3 GB | **~10 GB** |
| **Cosmos RL (全解凍 + ref policy)** | 582M + ref 582M×2 | 6.98 + 1.16 GB | ~3 GB | **~11 GB** |
| MiniPamayo Stage 0-1 (全解凍) | 582M + head <1M | 6.98 GB | ~3 GB | **~10 GB** |
| MiniPamayo Stage 2 (Flow 学習) | Traj. Decoder ~150M | 1.80 GB + VLM推論 1.16 GB | ~3 GB | **~6 GB** |
| MiniPamayo Stage 3 (CoC SFT, 全解凍) | 582M | 6.98 GB | ~4 GB | **~11 GB** |
| **MiniPamayo Stage 4 RL (LLM + ref)** | LLM 494M + ref 582M×2 | 5.93 + 1.16 GB | ~3 GB | **~10 GB** |

**最大 VRAM: ~11 GB（Cosmos RL / MiniPamayo Stage 3 時）。RTX 4090 (24 GB) に対して ~13 GB の余裕。**

### 5.4 Trajectory Decoder の VRAM 余裕

Stage 2 では VLM は frozen（推論のみ）のため、Trajectory Decoder に使える VRAM は大きい:

- 利用可能: 24 - 1.16 (VLM推論) - 2 (activation) - 1.5 (overhead) = **~19 GB**
- 最大 Decoder サイズ: 19 GB / 12 bytes ≈ **1.6B**（Alpamayo 10B の 2B には届かないが十分大きい）
- 採用サイズ: **~150M**（LLM 494M の ~30%、Alpamayo 10B の LLM:Decoder 比率と同等）

**結論**: 全パイプラインを通じて最大 ~11 GB。RTX 4090 (24 GB) で十分実行可能。

---

## 6. データセット

Alpamayo 0.5B は **80,000 時間**の内部データで学習している。MiniPamayo は公開データセットのみを使用するため、データ量に大きな差がある。この差を可能な限り縮小するため、複数の公開データセットを組み合わせて段階的にスケールアップする。

詳細な調査結果は [datasets.md](datasets.md) を参照。

### 6.1 段階的データ戦略

| Phase | データ | 合計時間 | Alpamayo 比 | 目的 |
|---|---|---|---|---|
| **A（検証）** | nuScenes Full + comma2k19 | ~40h | 1/2,000 | パイプライン検証 |
| **B（推奨）** | + commaCarSegments | ~500-2,500h | 1/30-160 | 汎化性能改善 |
| **C（理想）** | + nuPlan + Waymo E2E + CARLA | ~3,000+h | 1/20 | 実用的な性能 |

### 6.2 主要データソース

| データセット | データ量 | 映像 | 制御信号 | ライセンス |
|---|---|---|---|---|
| nuScenes Full | 5.5h (1,000 シーン) | 6 カメラ | ego pose + CAN bus | CC BY-NC-SA 4.0 |
| comma2k19 | 33h | フロント 1 台 | steering_angle, wheel_speeds | MIT |
| commaCarSegments | ~2,500h (145K セグメント) | フロント 1 台 | CAN bus | 公開 (HuggingFace) |
| nuPlan (navtrain) | ~1,300h | カメラ | ego 軌道 | 学術無料 |
| Waymo E2E | ~12h (5K セグメント) | 8 カメラ | ego 状態 + waypoint | 非商用研究 |
| CARLA シミュレータ | 無制限 | 全センサー | 完全な制御信号 | MIT |

### 6.3 スケールギャップ対策

データ量の差を埋めるため、データ以外のアプローチも活用する:

- **事前学習の活用**: DINOv2（ImageNet）+ Cosmos Reason Mini（運転ドメイン VLM）の重みが初期値。ゼロからの学習ではない
- **データ拡張**: 左右反転（実質 2 倍）、時間オフセット、カラーjitter
- **知識蒸留**: Cosmos Reason Mini の VLM 出力をシーン記述として追加入力に活用
- **シミュレータ生成**: CARLA でロングテール・コーナーケースを無制限に生成

---

## 7. Alpamayo 0.5B との対応表

比較対象はアーキテクチャ構成が最も近い **Alpamayo 0.5B（DINOv2 + Qwen2.5-0.5B）**。詳細は [差分分析](alpamayo-vs-minipamayo.md) 参照。

| 要素 | Alpamayo 0.5B | MiniPamayo | 状態 |
|---|---|---|---|
| Vision Encoder | DINOv2 | DINOv2 ViT-B/14 (86M) | **同系列**（Alpamayo 0.5B の DINOv2 サイズは論文に明記なし） |
| LLM | Qwen2.5-0.5B | **Qwen2.5-0.5B** | **同一モデル** |
| Adapter | DINOv2 → Qwen Projector | DINOv2 → Qwen Adapter | **同じ課題** |
| Trajectory Decoder | Flow Matching（サイズ不明） | Flow Matching (~150M) | 同思想（LLM の ~30%） |
| 予測ホライズン | 6.4秒（64 waypoints @ 10Hz） | **6.4秒**（64 waypoints @ 10Hz） | **同一** |
| カメラ | マルチカメラ（7台）＋時系列 | **1 台** | 差分 |
| アクション表現 | 制御ベース (a, κ) | 制御ベース (a, κ) | **同一** |
| Dual Representation | 離散トークン（学習）+ Flow（推論） | 同一（同じ LLM vocab に離散トークン追加） | **同一** |
| 条件付け | KV-cache（stop-gradient） | Hidden state sequence + cross-attention（stop-gradient） | **同思想** |
| Reasoning | CoC（構造化推論） | CoC（簡略版サブセット） | 同思想 |
| 学習戦略 | ドメインSFT → Action Injection → SFT → RL | ドメインSFT → 回帰 → 離散 → Flow → SFT → RL | 同思想 |
| RL ポストトレーニング | GRPO（softmax-weighted） + マルチ報酬 | GRPO（softmax-weighted） + マルチ報酬 | **同一** |
| 勾配制御 | Stage ごとに stop-grad / KL | 同一 | **同一** |
| ドメイン SFT | Cosmos-Reason パイプラインで実施 | auto-labeling + SFT | 同思想 |
| データ | 内部データ 80K 時間 | 公開データセット | **差分** |
| 総パラメータ | ~0.5B + α | ~730M | 同オーダー |

---

## 8. 実装上の推奨初期設定

```yaml
# 4090 で通すためのデフォルト
vision_encoder: facebook/dinov2-base  # ViT-B/14, 86M
llm: Qwen/Qwen2.5-0.5B               # 494M
image_size: 224
n_visual_tokens: 16
text_input: null  # or fixed short prompt
flow_steps: 10    # Stage 2 開始時
trajectory_decoder_params: 150M       # LLM の ~30%
micro_batch_size: 1
grad_accumulation_steps: 16
precision: bf16
gradient_checkpointing: true  # DINO / LLM / Trajectory Decoder すべて
optimizer: AdamW
learning_rate: 1.0e-4
weight_decay: 0.01
scheduler: cosine_with_warmup
```
