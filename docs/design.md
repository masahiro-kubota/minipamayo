# MiniPamayo 設計書 v0.2

## 1. 目的

- Alpamayo-R1（VLA：Vision-Language-Action）の**技術的理解**を目的に、同種の構成要素と学習段取りを RTX 4090 単体で再現する
- 目標は「SOTA 性能」ではなく、**学習が回り、Alpamayo の主要コンセプトを段階的に体験する**こと
- 対象コンセプト: 制御ベースアクション表現、Dual Representation（離散トークン + Flow）、構造化推論（CoC）、RL ポストトレーニング

---

## 2. 制約と前提

| 項目 | 値 |
|---|---|
| GPU | RTX 4090（24 GB VRAM） |
| 凍結 | **Stage ごとに制御**（詳細は §3.5 参照） |
| カメラ | **1 台** |
| 解像度 | 224×224（必要に応じて縮小可） |
| 視覚トークン | **16〜32 tokens / image** |

---

## 3. アーキテクチャ

```
┌──────────┐     ┌──────────────────┐     ┌──────────┐     ┌─────────────┐
│  Camera   │────▶│  DINOv2 ViT-S/14 │────▶│  Adapter  │────▶│  SmolLM2    │
│ (224×224) │     │  (Vision Enc.)   │     │ (Pooling) │     │  360M (LLM) │
└──────────┘     └──────────────────┘     └──────────┘     └──────┬──────┘
                                                                  │
                                                     ┌────────────┴────────────┐
                                                     │                         │
                                              ┌──────▼──────┐          ┌──────▼──────┐
                                              │ Action Head  │          │  Flow Head   │
                                              │ (MLP回帰)    │          │ (Flow Match) │
                                              │ [Stage 0]    │          │ [Stage 2]    │
                                              └─────────────┘          └─────────────┘
```

### 3.1 Vision Encoder — DINOv2 ViT-S/14

- モデル: `facebook/dinov2-small`（ViT-S/14、21M params）
- 入力: RGB 224×224
- 出力: パッチ特徴 (16×16)=256 patches × 384 dim
- Stage 0: trainable / Stage 2: frozen（§3.7 参照）

### 3.2 Vision → LLM Adapter（視覚トークン圧縮）

- 入力: DINOv2 パッチ特徴 (256 × 384)
- 出力: **N_vis tokens (16〜32) × d_llm (960)**
- 方式（実装容易性で選択、後で置換可）:

| 優先度 | 方式 | 概要 |
|---|---|---|
| 1 | Cross-Attention Pooling | learnable query 16/32 個で DINOパッチに attend |
| 2 | MLP + Attention Pool | 簡易版 |
| 3 | 平均 Pool + Linear | 最小実装（fail-fast 用） |

**初期実装**: まず方式 3（平均 Pool + Linear）で全パイプラインを通し、後で方式 1 に置き換える。

### 3.3 Language Model — SmolLM2-360M

- モデル: `HuggingFaceTB/SmolLM2-360M`
- アーキテクチャ: decoder-only Transformer
  - hidden_dim: 960
  - num_layers: 32
  - num_heads: 15
  - vocab_size: 49,152
- Stage 0: trainable / Stage 2: frozen（§3.7 参照）
- **必須要件**: KV-cache を取り出せること（Stage 2 Flow の条件付けに使用）

### 3.4 VLM 構築 + ドメイン SFT（前段パイプライン）

MiniPamayo の行動予測（Stage 0〜4）の前に、DINOv2 + Adapter + SmolLM2 を VLM として完成させ、運転ドメインの知識を注入する。この前段パイプラインは 2 つのサブプロジェクトに分離している:

#### Qwen2.5-VL Mini（汎用 VLM 構築）

Qwen2.5-VL / LLaVA の学習パイプラインに倣い、DINOv2 + SmolLM2 から**汎用 VLM**を構築する。

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
  - 予測ホライズン: **3.2秒**（32 waypoints @ 10Hz）— Alpamayo の 6.4秒から縮小
  - 制御入力: 32 × 2 = **64 値**
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
  - 32 waypoints × 2 values = **64 離散トークン** を LLM の語彙に特殊トークンとして追加
  - LLM が推論トークン（将来の CoC）とアクショントークンを**同じ自己回帰フレームワーク**で生成
  - Loss: cross-entropy（LLM の標準的な next-token prediction）
- **Dual Representation のメリット**（Alpamayo 論文 §5.1）:
  1. 推論（Reasoning）と軌道が共通のトークン空間を共有し、密な結合が可能
  2. RL ポストトレーニング時に離散トークンを通じて直接勾配を流せる
  3. 離散表現が車両ダイナミクスの強い教師信号となる
  4. Flow Matching による推論時デコードは 64 トークンの自己回帰生成より高速

#### Stage C — Flow Matching ヘッド（Stage 2 で使用）

- 条件付け:
  - **Option A（軽量）**: LLM 最終層 hidden states
  - **Option B（Alpamayo 寄り）**: LLM KV-cache
- Flow network: 小さな Transformer（LLM と同じ attention head 数・次元、ただし hidden/MLP は小さい）
- 入力: KV-cache + ノイズ付き制御入力 aₜ（拡散タイムステップ t を embedding して加算）
- 出力: velocity field vΘ(aₜ, o) の予測 → Euler 積分で連続軌道を生成
- Loss: **Conditional Flow Matching (CFM) loss** — Gaussian OT path で `aₜ = t·a + (1-t)·ε`, target field `u = a - ε`
- Flow steps: 推論時 10（δt = 0.1）
- **勾配制御**: Flow Head 学習時、VLM（Vision Encoder + LLM）からの KV-cache / hidden states には **stop-gradient** を適用し、Flow Head の勾配が VLM 側に逆伝播しないようにする（Alpamayo 論文 §5.1 と同様）

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

- **閉じた意思決定集合**: Alpamayo の完全なリストから、使用するデータセットに関連するサブセットのみ定義
  - 例（nuScenes 向け）: `{go_straight, turn_left, turn_right, stop, follow_lead, lane_change_left, lane_change_right, yield}`
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

#### アルゴリズム: GRPO

Alpamayo に倣い **GRPO（Group Relative Policy Optimization）** を採用:
- モデルから K 個のロールアウトをサンプリング
- グループ内の相対的な advantage で重み付け
- KL 正則化で SFT モデル（reference policy）からの逸脱を防止

#### 報酬信号（3要素）

1. **推論品質 (r_reason)**: LRM（大規模推論モデル）による 0-5 スケール採点
   - 行動一貫性: 推論が正しい運転行動を記述しているか
   - 因果推論品質: 因果要因が正しく特定されているか
   - MiniPamayo では外部 LLM API（例: Claude API）で代替可
2. **CoC-Action 一貫性 (r_consistency)**: バイナリ報酬
   - 予測軌道から meta-action を抽出し、推論トレースの意図と照合
   - 一致 → 1、不一致 → 0
3. **低レベル軌道品質 (r_traj)**:
   - L2 距離: 予測軌道 vs エキスパート軌道
   - 衝突ペナルティ: 周囲障害物との衝突判定
   - ジャーク抑制: 急激な制御変化へのペナルティ

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

- 現在の DINOv2 ViT-S/14（256 patches）→ Adapter（16~32 tokens）の圧縮は、Alpamayo の Flex に近い思想
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
| Stage 0（回帰） | 制御入力列 (a, κ) | (K, 2) | 制御ベース表現 |
| Stage 1（離散トークン） | 離散アクショントークン | (2K,) tokens | LLM 語彙に追加 |
| Stage 2（Flow） | 制御入力列 (a, κ) | (K, 2) — Flow で生成 | 連続・マルチモーダル |
| Stage 3（推論 SFT） | CoC 推論 + 離散トークン | テキスト + (2K,) tokens | 自己回帰生成 |
| Stage 4（RL） | CoC 推論 + 離散トークン | テキスト + (2K,) tokens | RL で品質向上 |

---

## 5. VRAM 見積もり（概算）

| コンポーネント | パラメータ数 | bf16 サイズ | 備考 |
|---|---|---|---|
| DINOv2 ViT-S/14 | 21M | ~42 MB | |
| SmolLM2-360M | 362M | ~724 MB | |
| Adapter | ~1M | ~2 MB | 方式による |
| Action Head (MLP) | <1M | ~2 MB | |
| **パラメータ合計** | **~385M** | **~770 MB** | |
| オプティマイザ状態 (AdamW) | — | ~3.1 GB | params×8 bytes (fp32 momentum + variance) |
| 勾配 | — | ~770 MB | params と同サイズ (bf16) |
| Activation（checkpointing ON） | — | ~2-4 GB | バッチサイズ依存 |
| **合計推定** | — | **~7-9 GB** | micro-batch=1 時 |

**結論**: 24 GB に対して十分余裕あり。micro-batch=2〜4 も試行可能。

---

## 6. Alpamayo 0.5B との対応表

比較対象はアーキテクチャ構成が最も近い **Alpamayo 0.5B（DINOv2 + Qwen2.5-0.5B）**。詳細は [差分分析](alpamayo-vs-minipamayo.md) 参照。

| 要素 | Alpamayo 0.5B | MiniPamayo | 状態 |
|---|---|---|---|
| Vision Encoder | DINOv2 | DINOv2 ViT-S/14 (21M) | **同系列**（サイズは異なる可能性あり） |
| LLM | Qwen2.5-0.5B | SmolLM2-360M | 同規模の decoder-only LLM |
| Adapter | DINOv2 → Qwen Projector | DINOv2 → SmolLM2 Adapter | **同じ課題** |
| カメラ | マルチカメラ（7台）＋時系列 | **1 台** | 差分 |
| アクション表現 | 制御ベース (a, κ) | 同思想 (a, κ) | §3.5 一致 |
| Dual Representation | 離散トークン（学習）+ Flow（推論） | 同思想 | §3.6 一致 |
| Flow Head | Flow Matching | Flow Matching（小規模） | 一致 |
| 条件付け | KV-cache（stop-gradient） | 同思想 | 一致 |
| Reasoning | CoC（構造化推論） | 同思想（簡略版） | §3.8 一致 |
| 学習戦略 | ドメインSFT → Action Injection → SFT → RL | ドメインSFT → 回帰 → 離散 → Flow → SFT → RL | §3.7 一致 |
| RL ポストトレーニング | GRPO + マルチ報酬 | 同思想（簡略版） | §3.9 一致 |
| 勾配制御 | Stage ごとに stop-grad / KL | 同思想 | §3.7 一致 |
| ドメイン SFT | Cosmos-Reason パイプラインで実施 | auto-labeling + SFT | §3.4 一致 |
| データ | 内部データ 80K 時間 | 公開データセット | 差分 |

---

## 7. 実装上の推奨初期設定

```yaml
# 4090 で通すためのデフォルト
image_size: 224
n_visual_tokens: 16
text_input: null  # or fixed short prompt
flow_steps: 10    # Stage 2 開始時
micro_batch_size: 1
grad_accumulation_steps: 16
precision: bf16
gradient_checkpointing: true  # DINO / LLM / Flow すべて
optimizer: AdamW
learning_rate: 1.0e-4
weight_decay: 0.01
scheduler: cosine_with_warmup
```
