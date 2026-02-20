# Cosmos Reason Mini 設計書 v0.3

## 1. 目的

Alpamayo-R1 の VLM バックボーンである **Cosmos-Reason** の技術的理解を目的に、同等の学習パイプライン（Physical AI SFT + Physical AI RL）を DINOv2 ViT-B/14 + Qwen2.5-0.5B の小規模構成で再現する。

**前提**: [Qwen2.5-VL Mini](../qwen-vl-mini/design.md) で構築した汎用 VLM（DINOv2 ViT-B/14 + Adapter + Qwen2.5-0.5B、画像→テキストの基礎能力を獲得済み）を入力とする。Cosmos Reason Mini は、この VLM に**運転ドメインの Physical AI 知識**を注入する段階。

```
Qwen2.5-VL Mini（汎用 VLM）→ Cosmos Reason Mini（本設計書）→ MiniPamayo（行動予測）
```

### Cosmos-Reason との対応関係

| 観点 | Cosmos-Reason-7B | Cosmos Reason Mini | 備考 |
|---|---|---|---|
| 前段の VLM | Qwen2.5-VL（既製品） | Qwen2.5-VL Mini（自前構築） | 同じ役割 |
| 学習: SFT | Physical AI SFT（~3.85M サンプル） | 同思想（~2.3M、**~1.7 倍差**） | 同じ 2 段階学習の第 1 段階 |
| 学習: RL | Physical AI RL（GRPO + MCQ 報酬） | 同思想（小規模） | 同じ 2 段階学習の第 2 段階 |
| 対象ドメイン | 汎用 Physical AI（ロボット、AV、人間） | **マルチドメイン（運転メイン + ロボット・人間活動・一般物理）** | 物理的常識の転移を活用 |

---

## 2. 制約と前提

| 項目 | 値 |
|---|---|
| GPU | RTX 4090（24 GB VRAM） |
| 初期重み | **Qwen2.5-VL Mini の学習済み重み**（汎用 VLM 構築済み） |
| 対象ドメイン | 自動運転（フロントカメラ 1 台） |
| 入力 | 画像（224×224）— 動画入力は将来拡張 |

## 3. アーキテクチャ

Qwen2.5-VL Mini と同一（DINOv2 ViT-B/14 + Adapter + Qwen2.5-0.5B）。アーキテクチャの詳細は [Qwen2.5-VL Mini 設計書 §3](../qwen-vl-mini/design.md) を参照。

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

Cosmos-Reason1 と同様に**マルチドメインの Physical AI データ**で学習する。物理的常識（重力・慣性・物体の永続性等）はドメイン横断で転移するため、運転データだけに限定せず、ロボット操作・人間活動・一般的な物理推論のデータを含める。

> **重要**: Cosmos-Reason1 の SFT データには MCQ（多肢選択問題）も含まれている（~1.2M 理解 MCQ + ~605K 推論 MCQ）。これにより、モデルは SFT の段階で `<think>...</think><answer>X</answer>` 形式を学習する。Cosmos Reason Mini でも SFT データに MCQ + CoT を混合する（`convert_mcq_to_sft.py` で変換）。

```
入力:  [visual_tokens] + [質問テキスト]
出力:  [思考トレース（CoT）] + [回答テキスト]
Loss:  cross-entropy（標準 SFT）
```

#### 4.1.1 物理的常識 SFT（Physical Common Sense）

空間・時間・基本物理学の理解。Cosmos-Reason1 の 5 つのエンボディメントドメインに対応:

| ドメイン | Cosmos-Reason1 のソース | Cosmos Reason Mini のソース | タスク例 |
|---|---|---|---|
| 自動運転 | nuScenes 等 | DriveLM, NuScenes-QA, Reason2Drive | 空間関係、物体認識、シーン記述 |
| ロボット操作 | BridgeData V2, RoboVQA | BridgeData V2（フレーム抽出 + QA 生成） | 把持、配置、物体の支持関係 |
| 人間活動 | Ego4D, HoloAssist | Ego4D（フレーム抽出 + QA 生成） | 日常動作、物体操作、因果関係 |
| 動物 | キュレーション動画 | 0（スキップ） | — |
| 一般物理 | 自己教師あり | PhysBench, CLEVR（既存 QA 活用） | 重力、衝突、安定性 |

**理解タスク（Understanding）**:
- 空間関係: 物体間の位置関係、距離、大きさ
- 物体属性: 形状、素材、状態（開/閉、満/空）
- シーン記述: 環境の状況、物体の配置

**推論タスク（Reasoning）**:
- 因果推論: 「なぜこの物体は倒れたのか？」
- 時間推論: 「次に何が起こりそうか？」
- 空間推論: 「この配置は安定しているか？」

#### 4.1.2 具現化推論 SFT（Embodied Reasoning）

行動予測・タスク完了確認・アフォーダンスの理解:

- **次の行動予測**: 「ego vehicle / ロボットは次にどう行動すべきか？」
- **タスク完了確認**: 「車線変更 / 物体の把持は完了したか？」
- **アフォーダンス**: 「この状況で右折 / この物体を掴むことは可能か？」

運転ドメインが最大比率だが、ロボット操作や人間活動のデータも含めることで、行動推論の汎化性を高める。

#### 4.1.3 直観的物理学 SFT（Intuitive Physics）

Cosmos-Reason の自己教師あり学習タスクの簡易版:

- **物体の永続性**: 遮蔽された物体の存在を推定
- **支持関係**: 物体が安定しているか（重力・摩擦の理解）
- **衝突予測**: 物体の軌道と衝突の可能性

PhysBench 等の既存ベンチマークデータを活用。Cosmos-Reason1 では ~63K だが、Mini では小規模で実施。

#### 4.1.4 データ取得戦略

Cosmos-Reason1 の SFT データ合計 ~3.85M に対し、**桁違いの差を解消**するため、以下の 3 段階で既存公開データを最大限活用する。GPT-4o による QA 生成は最小限に留める。

**核心戦略**: Cosmos-Reason1 の SFT データセットが HuggingFace で公開されている（`nvidia/Cosmos-Reason1-SFT-Dataset`、CC-BY-4.0）。このアノテーション（1.7M+ QA ペア）を直接活用し、動画→画像フレーム抽出で対応する。

**Tier 1: 画像ベースの既存 QA（変換のみ、API コスト 0）**

| データセット | QA 数 | 使用数 | ディスク容量 | ライセンス | 備考 |
|---|---|---|---|---|---|
| NuScenes-QA | ~460K | ~460K | ~100 MB (annotation) + nuScenes 画像 | CC BY-NC-SA 4.0 | テンプレートベース空間推論 |
| DriveLM-nuScenes | ~300K | ~300K | ~200 MB (annotation) + nuScenes 画像 | CC BY-NC-SA 4.0 | Graph VQA (Perception/Prediction/Planning) |
| Robo2VLM-1 | ~684K | ~200K | ~107 GB (画像+QA) | CC-BY-4.0 | ロボット操作 MCQ。画像同梱 |
| CLEVR | ~865K | ~200K | ~18 GB (画像+QA) | CC-BY-4.0 | 合成画像の空間推論 |
| PhysBench | ~10K | ~10K | ~3.5 GB (画像) | 公開 | 物理推論ベンチマーク |
| CODA-LM | ~42K | ~42K | ~99 MB (annotation) + CODA 画像 | Apache 2.0 | コーナーケース特化 |
| **Tier 1 小計** | | **~1.2M** | | | |

**Tier 2: 動画→フレーム抽出 + 既存アノテーション活用**

| データセット | QA 数 | 使用数 | ディスク容量 | ライセンス | 備考 |
|---|---|---|---|---|---|
| Cosmos-Reason1 SFT (BridgeData V2 分) | ~258K | ~200K | ~84 GB (全体) | CC-BY-4.0 | フレーム抽出必要 |
| Cosmos-Reason1 SFT (RoboVQA 分) | ~1.14M | ~300K | (動画同梱) | CC-BY-4.0 | フレーム抽出必要 |
| Cosmos-Reason1 SFT (HoloAssist 分) | ~273K | ~200K | (動画別途 DL) | CDLAv2 | フレーム抽出必要 |
| Cosmos-Reason1 SFT (AgiBot 分) | ~38.9K | ~38K | (動画同梱) | CC-BY-4.0 | ヒューマノイド操作 |
| Cosmos-Reason1 SFT (AV 分) | ~12.4K | ~12K | (動画同梱) | CC-BY-4.0 | 自動運転 |
| SUTD-TrafficQA | ~62.5K | ~62K | ~20.5 GB (動画+QA) | GitHub 公開 | 交通シーン MCQ |
| Reason2Drive | ~633K | ~300K | 大 (動画) | GitHub 公開 | チェーン推論 QA |
| **Tier 2 小計** | | **~1.1M** | | | |

**Tier 3: GPT-4o による QA 生成（補完）**

| データセット | ソース画像/動画 | 生成目標 | ディスク容量 | 備考 |
|---|---|---|---|---|
| BDD-X | ~7K 動画 (BDD100K) | ~26K QA | ~717 KB (annotation) | 行動+理由説明 |
| Something-Something V2 | ~220K clips | ~50K QA | ~19.4 GB (動画) | 物体操作の因果関係 |
| nuScenes 追加 QA | 既存画像 | ~50K QA | 0 (画像は既存) | 空間推論 QA |
| **Tier 3 小計** | | **~126K** | | API コスト要 |

#### 4.1.5 ディスク容量見積もり

| データ | 容量 | 備考 |
|---|---|---|
| **共通画像: nuScenes** (keyframes) | **~35 GB** | NuScenes-QA, DriveLM, Cosmos-Reason1 AV で共有 |
| Cosmos-Reason1 SFT Dataset | ~84 GB | HuggingFace からDL。動画+アノテーション |
| Robo2VLM-1 (サブセット) | ~30 GB | 200K/684K ≈ 30% |
| CLEVR (サブセット) | ~5 GB | 200K/865K のサブセット |
| PhysBench (画像) | ~3.5 GB | 画像のみ。動画は不要 |
| SUTD-TrafficQA | ~20.5 GB | 動画+アノテーション |
| Something-Something V2 | ~19.4 GB | 全動画 |
| CODA-LM (annotations) | ~99 MB | CODA 画像は別途 |
| DriveLM / NuScenes-QA annotations | ~300 MB | nuScenes 画像は上記で共有 |
| Reason2Drive (サブセット) | ~50 GB (推定) | 動画からフレーム抽出 |
| **合計見積もり** | **~250 GB** | |

#### 4.1.6 SFT データ規模の目標

| カテゴリ | Cosmos-Reason-7B | Cosmos Reason Mini | 差 |
|---|---|---|---|
| 自動運転 QA | ~12K + 内部データ | ~1.1M (NuScenes-QA + DriveLM + Reason2Drive + CODA-LM + SUTD-TrafficQA) | — |
| ロボット操作 QA | ~1.40M | ~738K (Robo2VLM-1 200K + Cosmos-Reason1 BridgeData 200K + RoboVQA 300K + AgiBot 38K) | **~1.9 倍差** |
| 人間活動 QA | ~273K | ~250K (Cosmos-Reason1 HoloAssist 200K + SSv2 50K) | **ほぼ同等** |
| 一般物理推論 QA | ~63K | ~210K (CLEVR 200K + PhysBench 10K) | **超過** |
| **合計** | **~3.85M** | **~2.3M** | **~1.7 倍差** |

> **従来の ~13-26 倍差から ~1.7 倍差に縮小**。Cosmos-Reason1 SFT Dataset の公開データ活用と、既存 QA データセットの積極的な採用により、桁違いの差を解消。

**ドメイン比率**:

| ドメイン | QA 数 | 比率 | 根拠 |
|---|---|---|---|
| 自動運転 | ~1.1M | ~48% | MiniPamayo の最終目標、NuScenes-QA+DriveLM+Reason2Drive が豊富 |
| ロボット操作 | ~738K | ~32% | Cosmos-Reason1 SFT アノテーション + Robo2VLM-1 |
| 人間活動 | ~250K | ~11% | Cosmos-Reason1 HoloAssist + SSv2 |
| 一般物理推論 | ~210K | ~9% | CLEVR サブセット + PhysBench |

> **注意**: 0.5B モデルでは過大なデータが有害になる可能性がある（TinyLLaVA 知見）。Tier 1 (~1.2M) から始めて過学習を監視し、Tier 2 を段階的に追加する。全量 (~2.3M) で過学習する場合はサブサンプリングで調整。

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

- 各質問に対して K 個の応答をサンプリング（ロールアウト）
- グループ内で報酬を正規化し advantage を計算:
  ```
  A_i = (R(o_i) - mean(G)) / std(G)
  ```
- **Multi-step 更新 (μ > 1)**: 同じロールアウトデータに対して μ 回の最適化ステップを実行。PPO clipping の ratio は old policy（ロールアウト時）に対して計算するため、2 ステップ目以降で ratio ≠ 1.0 となり clipping が有効に機能する
- KL 正則化で SFT モデル（reference policy）からの逸脱を防止

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

| カテゴリ | Cosmos-Reason-7B | Cosmos Reason Mini | データソース |
|---|---|---|---|
| 物理的常識 MCQ | ~5,100 | ~3,000〜5,000 | SUTD-TrafficQA + 自前 MCQ 変換 |
| 具現化推論 MCQ | ~1,200 | ~2,000〜5,000 | SFT 推論 QA → MCQ 変換 |
| 直観的物理学 MCQ | ~24,000 | 0（任意） | — |
| **合計** | **~30,300** | **~5,000〜10,000** | |

#### 4.2.4 RL 学習設定

| 項目 | Cosmos-Reason-7B | Cosmos Reason Mini |
|---|---|---|
| イテレーション | 500 | ~100〜300 |
| ロールアウト数 / 質問 | 9 | 4〜8（VRAM 制約） |
| 最大トークン長 | 6,144 | 2,048（推論トレースが短いため） |
| 学習率 | 4e-6 | 4e-6 |
| KL 係数 | 0.005 | 0.005 |
| バッチサイズ | 128 質問 | 4〜8 質問 |
| μ（最適化ステップ数） | 記載なし（multi-step） | 4 |

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

bf16 学習時の固定コスト: **N × 12 bytes**（パラメータ 2B + AdamW 1st moment 4B + 2nd moment 4B + 勾配 2B）

### SFT 時（全解凍）

| コンポーネント | パラメータ数 | bf16 サイズ |
|---|---|---|
| DINOv2 ViT-B/14 | 86M | ~172 MB |
| Qwen2.5-0.5B | 494M | ~988 MB |
| Adapter | ~2M | ~4 MB |
| **パラメータ合計** | **~582M** | **~1.16 GB** |
| 学習コスト (582M × 12 bytes) | — | ~6.98 GB |
| Activation（checkpointing ON） | — | ~3 GB |
| **合計推定** | — | **~10 GB** |

### RL 時（追加コスト）

| コンポーネント | 追加メモリ |
|---|---|
| Reference policy（frozen VLM、推論のみ） | ~1.16 GB |
| K 個のロールアウトバッファ | ~数百 MB |
| **RL 合計推定** | **~11 GB** |

**結論**: RTX 4090（24 GB）で SFT・RL ともに十分実行可能。~13 GB の余裕。

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

---

## 9. 小規模 VLM 論文からの知見（設計への反映）

詳細は [小規模 VLM 構築の知見まとめ](../small-vlm-research.md) を参照。以下は Cosmos Reason Mini の設計に直接影響する知見を抜粋。

### 9.1 SFT における CoT データの制限

**SmolVLM の発見**: 小型モデル（~500M）では CoT データを全体の **0.02〜0.05%** に制限すべき。過剰な CoT は容量を圧迫し、視覚表現を損なう。

**設計への反映**: §4.1.5 の推論 QA（CoT トレース付き）の比率に注意。推論 QA を全体の 30〜40% 程度に抑え、CoT トレース自体も短く簡潔にする。0.5B モデルの容量制約を考慮し、長い推論チェーンは避ける。

### 9.2 データ品質 > データ量

**TinyLLaVA の発見**: 1.1B LLM ではデータ量を増やすと POPE 等で逆に性能低下。Sub-1B モデルではデータの品質が量に勝る。

**設計への反映**: §4.1.5 の目標数（5,000〜10,000）は上限として扱う。少量（~1,000）から始めて品質を確認し、効果があれば段階的にスケールする。低品質な auto-labeled データを大量に投入しない。

### 9.3 LLM-SFT データの再利用は有害

**SmolVLM の発見**: LLM の命令追従データをマルチモーダル学習で再利用すると、画像で最大 -6.5%、動画で -3.7% の性能低下。

**設計への反映**: Cosmos Reason Mini の SFT データは全て画像付きの運転ドメイン QA とし、テキストのみの汎用 SFT データの混入を避ける。

### 9.4 エポック数と過学習

**Imp の発見**: 小型 VLM では 2 エポックが最適。3 エポック以上で過学習。

**設計への反映**: SFT は 1〜2 エポック（§4.1.6 のイテレーション数に反映済み）。チェックポイントを頻繁に保存し（25 ステップごと）、複数指標で最適なものを選択。

### 9.5 DINOv2 の特性を活かした SFT

**COMM 論文の発見**: DINOv2 は細粒度の空間情報に優れる（CLIP より grounding で +2.8pt）。一方でグローバルな意味推論は弱い。

**設計への反映**: 運転シーン理解 SFT（§4.1.1）では、DINOv2 の強みである**空間関係の QA**（先行車との距離、歩行者の位置等）を重視する。抽象的な推論よりも視覚的に具体的な QA を優先。
