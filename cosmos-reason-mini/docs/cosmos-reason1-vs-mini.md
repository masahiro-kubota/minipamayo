# Cosmos-Reason1 vs Cosmos Reason Mini：差分分析

本ドキュメントは、[Cosmos-Reason1 論文](../cosmos-reason1/cosmos-reason1-paper.md)の **7B 構成**と [Cosmos Reason Mini 設計書](design.md) を比較し、差分を整理する。

> **比較対象の選定理由**: Cosmos-Reason1 には 7B / 56B の構成がある。56B は Mamba-MLP-Transformer ハイブリッドという特殊なアーキテクチャを使用しており、Cosmos Reason Mini の Dense Transformer 構成とは異なるため、**7B 構成**（Dense Transformer ベース）を比較対象とする。

---

## 1. 全体アーキテクチャ

両者はいずれも **Vision Encoder → Projector → LLM** の decoder-only マルチモーダルアーキテクチャを共有しており、構成がよく似ている。

| 観点 | Cosmos-Reason1-7B | Cosmos Reason Mini | 差分 |
|---|---|---|---|
| Vision Encoder | ViT-676M | DINOv2 ViT-B/14 (86M) | **規模差 ~8倍** |
| Projector | 2層 MLP + PixelShuffle | Cross-Attention Pooling or MLP | 方式が異なる |
| LLM | Qwen2.5-7B (Dense Transformer) | Qwen2.5-0.5B (494M) | **同じ Qwen ファミリー、規模差 ~14倍** |
| 学習 Stage 1 | Physical AI SFT | 同思想（小規模） | **一致** |
| 学習 Stage 2 | Physical AI RL (GRPO) | 同思想（小規模） | **一致** |
| 対象ドメイン | 汎用 Physical AI（5 ドメイン） | マルチドメイン（4 ドメイン、運転メイン） | 動物のみスキップ |
| 入力 | 動画（最大32フレーム） | 画像（1フレーム） | **差分** |
| 総パラメータ | ~8B | ~582M | **規模差 ~14倍** |

---

## 2. Vision Encoder

| 観点 | Cosmos-Reason1-7B | Cosmos Reason Mini | 備考 |
|---|---|---|---|
| モデル | ViT-676M | DINOv2 ViT-B/14 | 事前学習方法が異なる（後述） |
| パラメータ数 | 676M | 86M | ~8倍差 |
| パッチサイズ | 14×14 | 14×14 | **一致** |
| レイヤー数 | 32 | 12 | |
| 隠れ次元 | 1,280 | 768 | |
| FFN 隠れ次元 | 3,456 | 3,072 | |
| 入力解像度 | 動的（448×448 タイル） | 224×224（固定） | |
| 動画対応 | 最大32フレーム、2fps | 1フレーム（画像のみ） | **差分** |
| パッチ数 / フレーム | 1,024 → PixelShuffle で 256 | 256 | 最終的なトークン数は同じ |

### 事前学習方法の違い

- **Cosmos-Reason1-7B の ViT**: Qwen2.5-VL の Vision Encoder をそのまま使用。大規模なマルチモーダルデータで事前学習済み
- **Cosmos Reason Mini の DINOv2**: 自己教師あり学習（DINO v2）で画像特徴の汎用表現を学習。テキストとの対応付けは Adapter が担当

### 影響

- DINOv2 は汎用的な視覚特徴を持つが、テキストとの整合性は事前に学習されていない。Adapter の学習がより重要になる
- 解像度が低い（224×224 vs 448×448）ため、遠方の物体認識に不利
- 動画入力非対応のため、時間的な推論（動きの予測等）は画像 1 枚の情報のみに依存

---

## 3. Projector / Adapter

| 観点 | Cosmos-Reason1-7B | Cosmos Reason Mini | 備考 |
|---|---|---|---|
| 方式 | 2層 MLP + 2×2×2 PixelShuffle | Cross-Attention Pooling or MLP | |
| 入力次元 | 1,280 | 768 | Vision Encoder の出力次元 |
| 隠れ次元 | 5,120 | — | |
| 出力次元 | 3,584（= LLM の hidden_dim） | 896（= Qwen2.5-0.5B の hidden_dim） | |
| ダウンサンプリング | 2×2×2（H×W×T） | 256→16（Cross-Attention の場合） | |
| 出力トークン数 / フレーム | 256（1,024 → PixelShuffle 1/4） | 16〜256（方式による） | |

### 設計思想の違い

- **Cosmos-Reason1**: PixelShuffle（隣接パッチの結合）による空間ダウンサンプリング。空間構造を保持しつつトークン数を削減。動画の場合は時間方向にも 2× 圧縮
- **Cosmos Reason Mini**: Cross-Attention Pooling は learnable query でパッチ特徴を圧縮。空間構造は暗黙的に query が学習。256 → 16 トークンと圧縮率が高い

---

## 4. Language Model

| 観点 | Cosmos-Reason1-7B | Cosmos Reason Mini | 備考 |
|---|---|---|---|
| ベースモデル | Qwen2.5-7B | Qwen2.5-0.5B | **同じ Qwen ファミリー** |
| アーキテクチャ | Dense Transformer | Dense Transformer | **一致** |
| パラメータ数 | ~7B | 494M | ~14倍差 |
| 隠れ次元 | 3,584 | 896 | |
| レイヤー数 | 28 | 24 | |
| アテンションヘッド数（Q / KV） | 28 / 4（GQA） | 14 / 2（GQA） | 両方 GQA |
| FFN 隠れ次元 | 18,944 | 4,864 | |
| 語彙数 | ~150K（Qwen） | 151,646 | **同一語彙** |

### 影響

- パラメータ数の差は推論品質に直結する。Qwen2.5-0.5B では複雑な因果推論（長い CoT）の表現力に限界がある
- ただし Cosmos-Reason の論文でも「モデルサイズが小さいほうが SFT の改善幅（相対値）は大きい」と報告しており（7B: +6.9% vs 56B: +2.0%）、小規模モデルでも SFT の効果は期待できる

---

## 5. SFT データ

### データ規模比較

| ドメイン | Cosmos-Reason1-7B | Cosmos Reason Mini | 差 | Mini のデータソース |
|---|---|---|---|---|
| 自動運転 | ~12K (AV) + 内部 | ~1.1M | — | NuScenes-QA 460K + DriveLM 300K + Reason2Drive 300K + CODA-LM 42K + SUTD-TrafficQA 62K |
| ロボット操作 | ~1.40M | ~738K | **~1.9 倍差** | Robo2VLM-1 200K + Cosmos-Reason1 SFT (BridgeData 200K + RoboVQA 300K + AgiBot 38K) |
| 人間活動 | ~273K | ~250K | **ほぼ同等** | Cosmos-Reason1 SFT (HoloAssist 200K) + SSv2 50K |
| 一般物理推論 | ~63K | ~210K | **超過** | CLEVR 200K + PhysBench 10K |
| **合計** | **~3.85M** | **~2.3M** | **~1.7 倍差** | |

> **従来の ~13-26 倍差から ~1.7 倍差に縮小**。Cosmos-Reason1 SFT Dataset（HuggingFace 公開、CC-BY-4.0）の直接活用と、既存 QA データセットの積極的な採用により桁違いの差を解消。

### データソースの容量一覧

| データセット | QA 数 | ディスク容量 | ライセンス | 取得方法 |
|---|---|---|---|---|
| nuScenes (keyframes, 共通画像) | — | ~35 GB | CC BY-NC-SA 4.0 | 公式 DL |
| NuScenes-QA | ~460K | ~100 MB (annotation) | CC BY-NC-SA 4.0 | GitHub |
| DriveLM-nuScenes | ~300K | ~200 MB (annotation) | CC BY-NC-SA 4.0 | HuggingFace |
| Robo2VLM-1 (サブセット) | ~684K (200K 使用) | ~30 GB (200K 分) | CC-BY-4.0 | HuggingFace |
| CLEVR (サブセット) | ~865K (200K 使用) | ~5 GB | CC-BY-4.0 | Stanford |
| PhysBench | ~10K | ~3.5 GB | 公開 | HuggingFace |
| CODA-LM | ~42K | ~99 MB (annotation) | Apache 2.0 | HuggingFace |
| Cosmos-Reason1 SFT Dataset | ~1.7M | ~84 GB | CC-BY-4.0 | HuggingFace |
| SUTD-TrafficQA | ~62.5K | ~20.5 GB | GitHub 公開 | Zenodo |
| Reason2Drive (サブセット) | ~633K (300K 使用) | ~50 GB (推定) | GitHub 公開 | GitHub |
| Something-Something V2 | ~220K clips | ~19.4 GB | 研究用 | 公式 |
| **合計** | | **~250 GB** | | |

### データ作成パイプラインの違い

| ステップ | Cosmos-Reason1-7B | Cosmos Reason Mini | 備考 |
|---|---|---|---|
| 1. ソース | 人間キュレーション動画 | **Cosmos-Reason1 SFT 公開データ + 既存 QA データセット** | 公開データ活用で大幅コスト削減 |
| 2. キャプション | 人間アノテーター or VLM | 既存アノテーション活用（一部のみ GPT-4o） | |
| 3. QA 生成 | LLM でキャプションから生成 | 既存 QA 変換がメイン + 一部 GPT-4o | |
| 4. 推論トレース | DeepSeek-R1 | Claude API / GPT-4o（Tier 3 のみ） | |
| 5. クリーニング | ルールベース + リライト | 同思想 | **一致** |

### SFT 対象ドメインの比較

| ドメイン | Cosmos-Reason1 | Cosmos Reason Mini | 備考 |
|---|---|---|---|
| 自律走行車 | ✅ | ✅（~48%、~1.1M） | NuScenes-QA, DriveLM, Reason2Drive, CODA-LM, SUTD-TrafficQA |
| ロボット操作 | ✅ BridgeData V2, RoboVQA | ✅（~32%、~738K） | Robo2VLM-1 + Cosmos-Reason1 SFT (BridgeData + RoboVQA + AgiBot) |
| 人間活動 | ✅ Ego4D, HoloAssist | ✅（~11%、~250K） | Cosmos-Reason1 SFT (HoloAssist) + SSv2 |
| 動物 | ✅ | ❌ スキップ | データ取得が困難 |
| 一般物理推論 | ✅ 自己教師あり | ✅（~9%、~210K） | CLEVR + PhysBench |

### 影響

- **桁違いの差を解消**: Cosmos-Reason1 SFT Dataset の公開データ活用により ~3.85M vs ~2.3M（~1.7 倍差）
- マルチドメインで物理的常識の転移学習効果が期待できる
- 動画→画像フレーム抽出が主要な追加作業（Cosmos-Reason1 SFT, SUTD-TrafficQA, Reason2Drive）
- ディスク容量 ~250 GB が必要

---

## 6. SFT 学習設定

| 観点 | Cosmos-Reason1-7B | Cosmos Reason Mini | 備考 |
|---|---|---|---|
| イテレーション | 12,500 | ~1,000〜3,000 | データ規模に比例 |
| 学習率スケジュール | cosine: 1e-5 → 1e-6 | cosine: 2e-5 → 2e-6 | Mini の方が若干大きい LR |
| グローバルバッチサイズ | 256 | 16（micro=1 × accum=16） | |
| オプティマイザ | Fused Adam (β1=0.9, β2=0.95) | AdamW (β1=0.9, β2=0.95) | β は **一致** |
| 重み減衰 | 0.1 | 0.01 | |
| 精度 | bf16 | bf16 | **一致** |
| gradient checkpointing | 不明 | ON | |

### 学習率が異なる理由

- Cosmos-Reason1 は大規模な事前学習済み VLM を微調整するため、小さい LR（1e-5）が適切
- Cosmos Reason Mini は Qwen2.5-VL Mini で視覚-言語アライメント済みだが、規模が小さいため若干大きめの LR（2e-5）で微調整

---

## 7. RL（強化学習）

| 観点 | Cosmos-Reason1-7B | Cosmos Reason Mini | 備考 |
|---|---|---|---|
| アルゴリズム | GRPO | GRPO | **一致** |
| 報酬タイプ | MCQ 正解率（ルールベース） | MCQ 正解率（ルールベース） | **一致** |
| 報酬検証 | 正規表現マッチング | 正規表現マッチング | **一致** |
| RL データ | ~30,300 MCQ | ~5,000〜10,000 MCQ | ~3〜6倍差 |
| ロールアウト数 / 質問 | 9 | 4〜8 | VRAM 制約で縮小 |
| バッチサイズ | 128 質問 | 4〜8 質問 | |
| 最大トークン長 | 6,144 | 2,048 | Mini は推論が短い |
| 学習率 | 4e-6 | 4e-6 | **一致** |
| KL 係数 | 0.005 | 0.005 | **一致** |
| イテレーション | 500 | ~100〜300 | |

### RL データの違い

| データカテゴリ | Cosmos-Reason1-7B | Cosmos Reason Mini | 備考 |
|---|---|---|---|
| 物理的常識 MCQ | ~5,100（人間アノテーション） | ~3,000〜5,000（SUTD-TrafficQA + 自前 MCQ） | |
| 具現化推論 MCQ | ~1,200（5ドメイン） | ~2,000〜5,000（SFT QA → MCQ 変換） | |
| 直観的物理学 MCQ | ~24,000（自己教師あり生成） | 0 | |

Cosmos-Reason1 の RL データの大部分（~24K / ~30K）は直観的物理学の自己教師ありデータ。Cosmos Reason Mini はこれをスキップするため、RL データは大幅に少ない。

### RL インフラの違い

- **Cosmos-Reason1**: 完全非同期RL訓練フレームワーク。ポリシー訓練とアクター展開を異種デプロイ。5D 並列処理（DP、PP、CP、FSDP、TP）。ノード障害時の自動再構成
- **Cosmos Reason Mini**: RTX 4090 単体。同期的なロールアウト → 報酬計算 → 更新のシンプルなループ

---

## 8. 評価

| 観点 | Cosmos-Reason1-7B | Cosmos Reason Mini | 備考 |
|---|---|---|---|
| 物理的常識ベンチマーク | 604 問（426 動画） | ~100〜200 問（運転特化） | |
| 具現化推論ベンチマーク | 610 問（6 ベンチマーク） | 上記に含む | |
| 直観的物理学ベンチマーク | 300 問（3 タスク） | 0 | |
| 評価形式 | MCQ + 二値質問 | MCQ | |
| 報告方法 | 5 シード平均 | 5 シード平均 | **一致** |

### Cosmos-Reason1 の報告精度（参考）

| ベンチマーク | SFT 前 | SFT 後 | RL 後 |
|---|---|---|---|
| 物理的常識（7B） | 47.4% | 54.3% (+6.9%) | 56.2% (+1.9%) |
| 具現化推論（7B） | 50.8% | 61.8% (+11.0%) | — |
| 直観的物理学（7B） | 42.1% | 74.5% (+32.4%) | 81.5% (+7.0%) |

Cosmos Reason Mini では同水準の絶対精度は期待できないが、**SFT / RL による相対的な改善傾向**は再現できる可能性がある。

---

## 9. Cosmos-Reason1-56B との追加的な差分（参考）

56B 構成は 7B とも大きく異なる。参考として記載:

- **Vision Encoder**: InternViT-300M-V2.5（7B の ViT-676M とは異なるモデル）
- **LLM バックボーン**: Nemotron-H（Mamba-MLP-Transformer ハイブリッド、118 層）— Dense Transformer ではない
- **入力**: 448×448 固定解像度、最大 32 フレーム
- **性能**: 物理的常識 60.2%（OpenAI o1 の 59.9% を上回る）

---

## 10. まとめ：実質的な差分

### 設計思想が一致している点

| 項目 | 状態 |
|---|---|
| アーキテクチャ（Vision Encoder → Projector → LLM） | **一致** |
| 2 段階学習（Physical AI SFT → Physical AI RL） | **一致** |
| SFT データパイプライン（キャプション → QA → 推論トレース → クリーニング） | **一致** |
| RL アルゴリズム（GRPO） | **一致** |
| RL 報酬設計（MCQ ベース、ルールベース検証） | **一致** |
| KL 正則化による SFT モデルからの逸脱防止 | **一致** |

### スケールの差（意図的な簡略化）

| 差分 | Cosmos-Reason1-7B | Cosmos Reason Mini | 影響 |
|---|---|---|---|
| Vision Encoder 規模 | 676M | 86M | 視覚特徴の表現力 |
| LLM 規模 | ~7B | 494M | 推論品質・CoT の深さ |
| SFT データ規模 | ~3.85M | ~2.3M | 汎化性能（**~1.7 倍差**） |
| RL データ規模 | ~30K | ~5,000〜10,000 | RL の効果量 |
| バッチサイズ | 256 | 16 | 学習安定性 |

### 構造的な差（アーキテクチャの違い）

| 差分 | 影響 | 対応方針 |
|---|---|---|
| **Vision Encoder の種類**（Qwen ViT vs DINOv2） | DINOv2 はテキストとの対応付けが未学習 | Adapter + SFT で橋渡しを学習 |
| **動画入力なし** | 時間的推論が弱い | 画像のみで可能な推論に限定 |
| **対象ドメイン**（5 ドメイン → 4 ドメイン） | 動物ドメインのみスキップ | 他 4 ドメインはカバー |
| **直観的物理学 SFT の規模差** | ~63K | ~5K-10K（PhysBench + CLEVR） | 小規模だが実施 |
| **推論トレースの教師モデル**（DeepSeek-R1 → Claude/GPT-4o） | 推論トレースの品質差 | 実験で品質を確認 |
