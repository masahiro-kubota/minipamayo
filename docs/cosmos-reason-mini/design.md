# Cosmos Reason Mini 設計書 v0.2

## 1. 目的

Alpamayo-R1 の VLM バックボーンである **Cosmos-Reason** の技術的理解を目的に、同等の学習パイプライン（Physical AI SFT + Physical AI RL）を DINOv2 + SmolLM2 の小規模構成で再現する。

**前提**: [Qwen2.5-VL Mini](../qwen-vl-mini/design.md) で構築した汎用 VLM（DINOv2 + Adapter + SmolLM2、画像→テキストの基礎能力を獲得済み）を入力とする。Cosmos Reason Mini は、この VLM に**運転ドメインの Physical AI 知識**を注入する段階。

```
Qwen2.5-VL Mini（汎用 VLM）→ Cosmos Reason Mini（本設計書）→ MiniPamayo（行動予測）
```

### Cosmos-Reason との対応関係

| 観点 | Cosmos-Reason-7B | Cosmos Reason Mini | 備考 |
|---|---|---|---|
| 前段の VLM | Qwen2.5-VL（既製品） | Qwen2.5-VL Mini（自前構築） | 同じ役割 |
| 学習: SFT | Physical AI SFT（~4M サンプル） | 同思想（小規模、数千〜1万） | 同じ 2 段階学習の第 1 段階 |
| 学習: RL | Physical AI RL（GRPO + MCQ 報酬） | 同思想（小規模） | 同じ 2 段階学習の第 2 段階 |
| 対象ドメイン | 汎用 Physical AI（ロボット、AV、人間） | **自動運転に特化** | MiniPamayo の目的に合わせて絞る |

---

## 2. 制約と前提

| 項目 | 値 |
|---|---|
| GPU | RTX 4090（24 GB VRAM） |
| 初期重み | **Qwen2.5-VL Mini の学習済み重み**（汎用 VLM 構築済み） |
| 対象ドメイン | 自動運転（フロントカメラ 1 台） |
| 入力 | 画像（224×224）— 動画入力は将来拡張 |

## 3. アーキテクチャ

Qwen2.5-VL Mini と同一（DINOv2 ViT-S/14 + Adapter + SmolLM2-360M）。アーキテクチャの詳細は [Qwen2.5-VL Mini 設計書 §3](../qwen-vl-mini/design.md) を参照。

Cosmos Reason Mini では**アーキテクチャの変更は行わず**、Qwen2.5-VL Mini の重みを初期値として、運転ドメインに特化した SFT + RL を行う。

---

## 4. 学習パイプライン

**前提**: Qwen2.5-VL Mini で Feature Alignment + Visual Instruction Tuning 済みの重みを使用。

Cosmos-Reason の 2 段階学習に倣う:

```
[Qwen2.5-VL Mini 完了] → Stage 1: Physical AI SFT（教師あり微調整）→ Stage 2: Physical AI RL（強化学習）
```

### 4.1 Physical AI SFT

Cosmos-Reason の SFT は 3 カテゴリのデータで構成されている:

1. **物理的常識 SFT**（Physical Common Sense）— 空間・時間・基本物理学の理解
2. **具現化推論 SFT**（Embodied Reasoning）— 行動予測・タスク完了確認・アフォーダンス
3. **直観的物理学 SFT**（Intuitive Physics）— 空間連続性・時間の矢・物体の永続性

Cosmos Reason Mini では**自動運転に特化**し、以下のサブセットを実施:

#### 4.1.1 運転シーン理解 SFT（物理的常識に対応）

教師 VLM（GPT-4o 等）を使い、運転データセットの画像から QA ペアを生成:

**理解タスク（Understanding）**:
- シーン記述: 道路状況、天候、時刻、道路タイプ
- 空間関係: 先行車・歩行者・信号・標識の位置関係
- 物体属性: 車両の色・種類・大きさ、信号の状態

**推論タスク（Reasoning）**:
- 因果推論: 「なぜ先行車は減速しているのか？」
- 時間推論: 「次に何が起こりそうか？」
- 空間推論: 「この車線変更は安全か？」

```
入力:  [visual_tokens] + [質問テキスト]
出力:  [思考トレース（CoT）] + [回答テキスト]
Loss:  cross-entropy（標準 SFT）
```

#### 4.1.2 運転行動推論 SFT（具現化推論に対応）

Cosmos-Reason の Embodied Reasoning SFT を自動運転に特化:

- **次の行動予測**: 「ego vehicle は次にどう行動すべきか？」
- **タスク完了確認**: 「車線変更は完了したか？」
- **アフォーダンス**: 「この状況で右折は可能か？」

Cosmos-Reason では BridgeData V2、RoboVQA 等のロボット操作データを使用しているが、Cosmos Reason Mini では**運転データセット（nuScenes, comma2k19）のみ**を使用する。

#### 4.1.3 直観的物理学 SFT（任意・発展）

Cosmos-Reason の自己教師あり学習タスクの簡易版:

- **時間の矢（Arrow of Time）**: 動画の順再生 / 逆再生を判定
- **空間パズル（Spatial Puzzles）**: シャッフルされた画像パッチの元位置を推定

これらは追加的な物理理解の強化に有用だが、MiniPamayo の目的（技術理解）に対しては優先度が低い。

#### 4.1.4 データ作成パイプライン

Cosmos-Reason の 5 段階キュレーションパイプラインを簡略化:

```
Step 1: 画像選定
    運転データセットから代表的なフレームを選定
    ↓
Step 2: 教師 VLM によるキャプション生成
    GPT-4o 等でシーンの詳細記述を生成
    ↓
Step 3: QA ペア生成
    LLM を使い、キャプションに基づいて「理解」と「推論」の QA を生成
    ↓
Step 4: 推論トレース抽出（任意）
    推論 QA に対して教師 LLM で CoT 推論トレースを生成
    ↓
Step 5: クリーニング
    ビジュアルコンテキストへの不要な参照を除去
```

Cosmos-Reason では Step 4 に DeepSeek-R1 を使用。Cosmos Reason Mini では Claude API / GPT-4o 等で代替可。

#### 4.1.5 SFT データ規模の目安

| カテゴリ | Cosmos-Reason-7B | Cosmos Reason Mini | 備考 |
|---|---|---|---|
| 理解 QA | ~2.0M | ~3,000〜5,000 | 運転シーンに特化 |
| 推論 QA | ~1.85M | ~2,000〜5,000 | CoT トレース付き |
| 直観的物理学 | ~63K | 0（任意） | 優先度低 |
| **合計** | **~3.85M** | **~5,000〜10,000** | 桁違いの差だが目的は技術理解 |

#### 4.1.6 SFT 学習設定

| 項目 | Cosmos-Reason-7B | Cosmos Reason Mini |
|---|---|---|
| 初期重み | Qwen2.5-VL（既製 VLM） | **Qwen2.5-VL Mini（自前 VLM）** |
| イテレーション | 12,500 | ~1,000〜3,000（データ規模に応じて） |
| 学習率 | 1e-5 → 1e-6（cosine） | 2e-5 → 2e-6（cosine） |
| バッチサイズ | 256（グローバル） | micro-batch=1, grad_accum=16 |
| オプティマイザ | Fused Adam (β1=0.9, β2=0.95) | AdamW (β1=0.9, β2=0.95) |
| 重み減衰 | 0.1 | 0.01 |
| 精度 | bf16 | bf16 |
| gradient checkpointing | — | ON（DINO + LLM） |

**学習率の変更**: Qwen2.5-VL Mini で既に視覚-言語アライメントが完了しているため、以前の設計（1e-4）より小さい学習率（2e-5）で微調整する。Cosmos-Reason1 が Qwen2.5-VL の上に 1e-5 で SFT するのと同じ考え方。

### 4.2 Physical AI RL

Cosmos-Reason の RL ポストトレーニングを小規模に再現する。

#### 4.2.1 アルゴリズム: GRPO

Cosmos-Reason と同じ GRPO を採用:

- 各質問に対して K 個の応答をサンプリング
- グループ内で報酬を正規化し advantage を計算:
  ```
  A_i = (R(o_i) - mean(G)) / std(G)
  ```
- KL 正則化で SFT モデルからの逸脱を防止

#### 4.2.2 報酬設計: MCQ ベース

Cosmos-Reason の核心的アイデア: **MCQ（多肢選択問題）に変換することで、ルールベース・検証可能な報酬を実現**。

```
質問: この状況で ego vehicle が次にとるべき行動は？
A) 加速して追い越す
B) 減速して車間距離を確保する  ← 正解
C) 車線変更する
D) 停車する

報酬: 正解選択 → 1、不正解 → 0
```

- SFT データの QA を MCQ 形式に変換
- 回答は `<answer>B</answer>` のようなタグ形式で検証
- 正規表現パターンマッチングで自動採点

#### 4.2.3 RL データ

| カテゴリ | Cosmos-Reason-7B | Cosmos Reason Mini | 備考 |
|---|---|---|---|
| 物理的常識 MCQ | ~5,100 | ~500〜1,000 | 運転シーンに特化 |
| 具現化推論 MCQ | ~1,200 | ~500〜1,000 | 行動予測・判断 |
| 直観的物理学 MCQ | ~24,000 | 0（任意） | 優先度低 |
| **合計** | **~30,300** | **~1,000〜2,000** | |

#### 4.2.4 RL 学習設定

| 項目 | Cosmos-Reason-7B | Cosmos Reason Mini |
|---|---|---|
| イテレーション | 500 | ~100〜300 |
| ロールアウト数 / 質問 | 9 | 4〜8（VRAM 制約） |
| 最大トークン長 | 6,144 | 2,048（推論トレースが短いため） |
| 学習率 | 4e-6 | 4e-6 |
| KL 係数 | 0.005 | 0.005 |
| バッチサイズ | 128 質問 | 4〜8 質問 |

---

## 5. 入出力仕様

### 5.1 入力

| 入力 | 形状 | 備考 |
|---|---|---|
| 画像 | RGB 224×224 | カメラ 1 台 |
| 質問テキスト | 自然言語 | SFT / RL の質問 |

### 5.2 出力

| Stage | 出力 | 備考 |
|---|---|---|
| SFT（理解） | シーン記述テキスト | 道路状況、物体の位置・状態 |
| SFT（推論） | CoT 推論トレース + 回答 | 因果推論、行動推論 |
| RL | MCQ 回答（タグ形式） | `<answer>B</answer>` |

---

## 6. 評価

### 6.1 評価指標

Cosmos-Reason に倣い、MCQ の正解率で評価:

- **運転シーン理解**: 空間関係、物体認識、シーン記述の正確さ
- **運転行動推論**: 次の行動予測、因果推論の正確さ
- **SFT → RL の改善幅**: RL 後に MCQ 正解率がどの程度改善するか

### 6.2 評価データ

- SFT / RL の学習データとは別に、評価用の MCQ セットを用意（~100〜200 問）
- 5 回のランダムシード平均で報告（Cosmos-Reason と同様）

---

## 7. VRAM 見積もり（概算）

### SFT 時

| コンポーネント | パラメータ数 | bf16 サイズ |
|---|---|---|
| DINOv2 ViT-S/14 | 21M | ~42 MB |
| SmolLM2-360M | 362M | ~724 MB |
| Adapter | ~1M | ~2 MB |
| **パラメータ合計** | ~384M | ~768 MB |
| オプティマイザ状態 (AdamW) | — | ~3.1 GB |
| 勾配 | — | ~768 MB |
| Activation（checkpointing ON） | — | ~2-4 GB |
| **合計推定** | — | **~7-9 GB** |

### RL 時（追加コスト）

| コンポーネント | 追加メモリ |
|---|---|
| Reference policy（frozen SFT モデル） | ~768 MB |
| K 個のロールアウトバッファ | ~数百 MB |
| **RL 合計推定** | **~9-12 GB** |

**結論**: RTX 4090（24 GB）で SFT・RL ともに十分実行可能。

---

## 8. 全体パイプラインにおける位置付け

```
Qwen2.5-VL Mini             Cosmos Reason Mini（本設計書）   MiniPamayo
┌─────────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│ Feature Alignment   │     │ Physical AI SFT     │     │ Stage 0: 回帰        │
│ Visual Instruction  │────▶│  運転シーン理解 QA   │────▶│ Stage 1: 離散化       │
│ Tuning              │     │ Physical AI RL      │     │ Stage 2: Flow        │
│                     │     │  MCQ + GRPO         │     │ Stage 3: CoC SFT     │
└─────────────────────┘     └─────────────────────┘     │ Stage 4: RL          │
汎用 VLM 構築               運転ドメイン特化             └──────────────────────┘
                                                        行動予測
```

- Qwen2.5-VL Mini で「画像→テキスト」の基礎 VLM 能力を獲得
- Cosmos Reason Mini で運転ドメインの理解・推論能力を追加
- その重み（Vision Encoder + Adapter + LLM）が MiniPamayo Stage 0 の初期値となる
- Cosmos Reason Mini の推論能力（CoT）は、MiniPamayo Stage 3（CoC SFT）の基盤となる
- Cosmos Reason Mini の RL 学習経験は、MiniPamayo Stage 4（RL）の設計に活用できる
