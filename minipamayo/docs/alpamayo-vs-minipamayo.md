# Alpamayo 0.5B vs MiniPamayo 設計書：差分分析

本ドキュメントは、[Alpamayo-R1 論文](alpamayo/alpamayo-paper.md)の **0.5B 構成**（DINOv2 + Qwen2.5-0.5B）と [MiniPamayo 設計書 v0.3](design.md) を比較し、差分を整理する。

> **比較対象の選定理由**: Alpamayo には 0.5B / 3B / 7B（10B）の構成がある。MiniPamayo は RTX 4090 単体での技術理解が目的であり、アーキテクチャ構成が最も近い **0.5B 構成**（DINOv2 + Qwen2.5-0.5B）を比較対象とする。10B 構成（Cosmos-Reason-7B + 内蔵 ViT + 2B Flow デコーダ）は規模が異なりすぎるため、特記事項として言及するにとどめる。

---

## 1. 全体アーキテクチャ

両者はいずれも **DINOv2 → Adapter/Projector → LLM → Action Head** という基本パイプラインを共有しており、構成がよく似ている。

| 観点 | Alpamayo 0.5B | MiniPamayo | 差分 |
|---|---|---|---|
| Vision Encoder | DINOv2 | DINOv2 ViT-B/14 (86M) | **同系列**（サイズは後述） |
| LLM | Qwen2.5-0.5B | **Qwen2.5-0.5B** | **同一モデル** |
| カメラ | マルチカメラ（7台）＋時系列 | **1台**（フロント） | **差分** |
| Trajectory Decoder | Flow Matching Expert (~2B) | Flow Matching Expert (~95M) | **規模差 ~21倍**（後述 §6.1） |
| 学習戦略 | Action Injection → SFT → RL | 回帰 → 離散 → Flow → SFT → RL | 同思想（MiniPamayo は fail-fast で段階的） |
| 総パラメータ（VLM + Decoder） | ~0.5B + α | ~730M | 同オーダー |

---

## 2. Vision Encoder

両者とも DINOv2 を採用している。

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| モデル | DINOv2（サイズは論文に明記なし） | DINOv2 ViT-B/14 (86M) | hidden=768、LLM の 896 に近い |
| 入力解像度 | 448×280（論文デフォルト） | 224×224 | MiniPamayo は VRAM 節約のため縮小 |
| マルチカメラ | 7台入力 | 1台 | 0.5B でもマルチカメラを処理 |
| マルチタイムステップ | 2秒の履歴（複数フレーム） | 単一フレーム | MiniPamayo は時系列未対応 |

### 影響
- **解像度の差**: 224×224 は DINOv2 の標準入力サイズなので問題なし。ただし低解像度は遠方物体の認識に不利
- **カメラ数**: MiniPamayo の最大の簡略化。360度認識ができないため、側方・後方の物体に対する推論は不可能
- **時系列**: Alpamayo は過去2秒の履歴を使って時間的変化（接近する車両等）を推論できる。MiniPamayo は egomotion 入力で部分的に補完

---

## 3. Adapter / Projector

両者とも「DINOv2 の出力を LLM の embedding 空間にマッピングする」モジュールが必要であり、役割は同じ。

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| 方式 | Projector（詳細は論文に明記なし） | Cross-Attention Pooling（16~32 query） | |
| DINOv2 → LLM の次元変換 | 必要 | 必要（768 → 896） | 射影ギャップ 1.2倍で小さい |
| トークン数圧縮 | 不明（マルチカメラなので効率化が必要） | 256 → 16~32（積極的に圧縮） | |

### 備考
10B 構成では Cosmos-Reason 内蔵 ViT + 2層 MLP Projector を使うが、0.5B 構成は DINOv2 を外部 Vision Encoder として使うため、MiniPamayo と同様に DINOv2 → LLM を橋渡しする Adapter/Projector が必要になる。この点で **MiniPamayo の Adapter 設計は 0.5B Alpamayo と同じ課題を解いている**。

---

## 4. Language Model

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| モデル | Qwen2.5-0.5B | **Qwen2.5-0.5B** | **同一モデル** |
| パラメータ数 | 494M | 494M | **同一** |
| アーキテクチャ | decoder-only Transformer, GQA | 同一 | hidden=896, 24層, 2 KV heads |
| 視覚入力の事前学習 | なし（テキスト LLM） | なし（テキスト LLM） | **同条件** |
| ドメイン SFT | Cosmos-Reason パイプラインで運転データ SFT 済み | Cosmos Reason Mini で小規模 SFT | 同思想（スケール差） |

### ドメイン SFT
Alpamayo では 0.5B であっても、Cosmos-Reason のパイプラインで物理 AI 向けの SFT（運転シーン記述、推論トレース等）が行われている。MiniPamayo でも設計書 §3.4 で同様のドメイン SFT を Stage 0 の前段として実施する方針。教師 VLM による auto-labeling で運転シーンの QA データを作成し、Qwen2.5-0.5B に運転ドメインの視覚理解を獲得させる。

ただし、Alpamayo の Cosmos-Reason パイプラインは大規模データ（Physical AI 全般を含む）で行われているのに対し、MiniPamayo は公開データセットの範囲内での小規模 SFT となる。

---

## 5. アクション表現

設計書 v0.3 では Alpamayo と同一の制御ベース表現・予測ホライズンを採用しており、**完全に一致**している。

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| 表現 | (加速度 a, 曲率 κ) | (a, κ) | **同一** |
| ダイナミクス | ユニサイクル + Euler 積分 | 同じ | **同一** |
| 予測ホライズン | 6.4秒（64 waypoints @ 10Hz） | 6.4秒（64 waypoints @ 10Hz） | **同一** |
| 離散トークン数 | 128 tokens (64×2) | 128 tokens (64×2) | **同一** |
| GT 逆算 | 最小二乗法 + Tikhonov 正則化 | 同じ | **同一** |

---

## 6. Dual Representation（離散トークン + Flow）

Alpamayo も MiniPamayo も、離散トークンと Flow Matching の **2つの行動表現を併用** する。これは Alpamayo が論文 §5.1-5.2 で提案した設計であり、MiniPamayo も同じ設計を採用している。

### なぜ離散トークンが必要なのか（Flow Matching だけではダメな理由）

Flow Matching デコーダは hidden states に stop-gradient を適用するため、**VLM に勾配が流れない**：

```
画像 → VLM → hidden states → [stop-gradient] → デコーダ → 軌道
                                    ↑
                            勾配がここで切れる → VLM は学習できない
```

VLM が「この画像を見たらどう動くべきか」を学習するには、VLM 自身に勾配が流れる経路が必要。離散トークンがその役割を担う：

```
画像 → VLM → lm_head → 離散トークン予測 → CE loss → VLM を更新
                ↑
        勾配が VLM まで到達する
```

つまり **2つの表現は異なる役割** を持つ：

| | 離散トークン (CE loss) | Flow Matching (CFM loss) |
|---|---|---|
| **学習対象** | VLM | デコーダのみ |
| **目的** | VLM に行動モダリティを獲得させる | 連続で滑らかな軌道を生成する |
| **勾配** | VLM に流れる | VLM には流れない (stop-gradient) |
| **推論時に使うか** | 使わない（教師信号のみ） | 使う（最終出力） |

離散トークンは推論時には使わない。**VLM を教育するための教師信号** として存在し、推論時は Flow Matching が実際の軌道を出力する。この設計は Alpamayo §5.1-5.2 で明確に述べられており、MiniPamayo も同一の設計思想に従っている。

### 比較表

同じ LLM（Qwen2.5-0.5B）の語彙に離散トークンを追加し、同じ GQA 構造の KV-cache で条件付けるため、**実質的に同一の設計**。

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| 学習時 | 離散トークン（cross-entropy） | 同じ | **同一**（同一 LLM vocab） |
| 推論時 | Flow Matching デコーダ | 同じ | **同一** |
| Trajectory Decoder 規模 | ~2B（10B 版、ソースコード確認済み） | ~95M（24L, 512h, 8heads） | **規模差 ~21倍**（§6.1 参照） |
| Flow の条件付け | KV-cache（past_key_values） | KV-cache（past_key_values） | **同一**（Alpamayo 準拠に移行済み） |
| 学習方式 | **同時学習**（CE + CFM を同一ループで） | **順次学習**（Stage 1 → Stage 2） | **意図的な差分** |

### 6.1 Expert（Flow Matching デコーダ）アーキテクチャの実態

**ソースコード調査**（`alpamayo/src/alpamayo_r1/models/alpamayo_r1.py`）により、Alpamayo の Expert は **VLM の text_config をコピーした巨大な Transformer** であることが判明した。

```python
# alpamayo_r1.py, line 88-92
expert_config = copy.deepcopy(self.vlm.config.text_config)
if config.expert_cfg is not None:
    for key, value in config.expert_cfg.items():
        setattr(expert_config, key, value)
self.expert = AutoModel.from_config(expert_config)
```

HuggingFace `nvidia/Alpamayo-R1-10B` の `config.json` から確認した具体的なパラメータ：

| パラメータ | Qwen3-VL-8B text_config（元） | Expert（expert_cfg でオーバーライド後） |
|---|---|---|
| **hidden_size** | 4096 | **2048** |
| **num_hidden_layers** | 36 | **36**（変更なし） |
| **num_attention_heads** | 32 | **16** |
| **intermediate_size** | 12288 | **8256** |
| **head_dim** | 128 | 128 |
| **推定パラメータ数** | ~8B | **~2B** |

#### MiniPamayo との比較

| | Alpamayo Expert (10B版) | MiniPamayo Expert (デフォルト) |
|---|---|---|
| **アーキテクチャ** | Qwen3 Transformer（VLM text_config コピー） | Qwen2 Transformer（同パターン） |
| **層数** | **36** | **24**（VLM と一致） |
| **hidden_dim** | **2048** | **512** |
| **attention heads** | **16** | **8** |
| **num_kv_heads** | 8 | **2**（VLM と一致） |
| **intermediate_size** | **8256** | **2048** |
| **推定パラメータ** | **~2B** | **~95M** |
| **倍率** | — | **~21倍小さい** |

#### 条件付け方式（Alpamayo 準拠に移行済み）

| | Alpamayo Expert | MiniPamayo Expert |
|---|---|---|
| **方式** | `past_key_values`（VLM の KV-cache をそのまま渡す） | `past_key_values`（同一方式） |
| **attention 種別** | **non-causal**（`is_causal=False`） | **non-causal**（同一） |
| **入力エンコーディング** | Fourier Feature V2 + MLP (4層, 1024dim) | Fourier Feature V2 + MLP (4層, 1024dim)（同一） |
| **KV-cache crop** | 各ステップ後に Expert 追加分を crop | 同一パターン |

MiniPamayo の Expert は Alpamayo と同じく Qwen2 Transformer アーキテクチャを採用し、VLM の `past_key_values` をネイティブに消費する。`num_kv_heads=2`, `head_dim=64` を VLM と一致させることで KV-cache の直接受け渡しが可能。

#### 残る規模差と改善方針

~21倍の規模差は、**カーブシーンで Flow Matching が曲線軌跡を生成しにくい** 問題の一因と考えられる。改善策：

1. **Classifier-Free Guidance (CFG)**: 訓練時に conditioning を確率的にドロップし、推論時にガイダンスで増幅
2. **Expert スケールアップ**: 512h → 896h（~280M、フル構成）
3. **カーブシーンのオーバーサンプリング**: `WeightedRandomSampler` で実装済み（3× 重み付け）

### 意図的な差分: 順次学習（Stage 分割）

Alpamayo では Stage 1 (Action Modality Injection) で離散トークン (CE loss) と Flow Matching Expert (CFM loss) を**同一ループ内で同時に学習**する。MiniPamayo ではこれを **Stage 1（離散トークン）と Stage 2（Flow Matching デコーダ）に分割**している。

#### Alpamayo の同時学習の実態

「同時学習」の内部構造は以下の通り：

```
1つの training step 内で：
  VLM forward pass → hidden states
    ├── CE loss（離散トークン） → VLM パラメータを更新
    └── hidden states.detach() → CFM loss → デコーダパラメータのみ更新
```

stop-gradient により CFM loss は VLM に逆伝播しない。つまり **VLM にとって CFM は存在しないのと同じ** であり、「同時学習」と言いつつ VLM とデコーダの学習は実質的に独立している。

#### Alpamayo が同時学習を選んだ理由（推定）

技術的なメリットはほぼない。考えられる理由：

1. **実装の簡潔さ** — 1つの training loop で済み、チェックポイント管理が不要
2. **GPU 利用効率** — VLM の forward pass を CE と CFM で使い回せる（順次では同じ forward を2回実行する）
3. **論文構成** — 「統一的な学習フレームワーク」の方が論文として見栄えが良い

#### MiniPamayo が順次学習を選んだ理由

1. **Fail-fast** — Stage ごとに独立検証でき、問題を早期発見できる
2. **デコーダの学習安定性** — 同時学習ではデコーダが学習初期の不安定な VLM hidden states を入力として学習する。順次学習では Stage 1 で収束済みの安定した hidden states で学習できるため、デコーダの収束が速く安定する
3. **デバッグ容易性** — Stage 1 で VLM の離散トークン品質を確認し、Stage 2 でデコーダの連続軌道品質を個別に確認できる

#### 結果への影響

**VLM の学習結果は同一**: stop-gradient により CE loss のみが VLM を更新するため、同時学習でも順次学習でも VLM のパラメータ収束先は同じ。

**デコーダへの影響は順次学習が有利**: 同時学習ではデコーダの入力分布が VLM の学習中に変動し続ける。順次学習では固定された分布で学習するため、むしろ安定した学習が期待できる。

---

## 7. Reasoning — Chain of Causation (CoC)

設計書 v0.3 でカバー済み。

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| CoC 構造 | Driving Decision + Critical Components + CoC Trace | 同思想（簡略版） | |
| 意思決定集合 | 縦方向 8種 + 横方向 8種 | サブセットのみ | MiniPamayo はデータセットに応じて絞る |
| ラベリング | 人間 + VLM（GPT-5, Qwen3-VL） | VLM auto-labeling のみ | 人間ラベリングは MiniPamayo の範囲外 |
| データ規模 | 大規模（80K 時間のデータから選定） | 小規模（公開データセット） | |

### 差分: CoC データの品質とスケール
Alpamayo では専門のアノテーターによる人間ラベリング + VLM auto-labeling のハイブリッドで、厳格な QA プロセスを経ている。MiniPamayo は auto-labeling のみに依存するため、CoC データの品質が劣る可能性がある。ただし論文 §4.3.3 で「不完全な auto-label でも SFT には十分有効」と述べられており、RL ポストトレーニングで品質を補完できる。

---

## 8. RL ポストトレーニング

同じ LLM に対する GRPO のため、**アルゴリズムは同一**。差分は計算リソースとスケールのみ。

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| アルゴリズム | GRPO | 同じ | **一致** |
| 報酬: 推論品質 | LRM（DeepSeek-R1, Cosmos-Reason） | 外部 LLM API | MiniPamayo は API 呼び出しで代替 |
| 報酬: CoC-Action 一貫性 | ルールベース（バイナリ） | 同じ | **一致** |
| 報酬: 軌道品質 | L2 + 衝突 + ジャーク | 同思想 | |
| ロールアウト数 | 不明 | K=4~8（VRAM 制約） | |
| データキュレーション | 高情報利得サンプル優先 | 未定 | MiniPamayo はデータ量が小さいため全データ使用の可能性 |

### 差分: 計算コスト
RL は計算コストが高い。Alpamayo は複数 GPU ノードで分散処理しているが、MiniPamayo は RTX 4090 単体のため、ロールアウト数を絞り、推論品質報酬をオフラインで計算する等の工夫が必要。

---

## 9. 効率的ビジョンエンコーディング

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| Single-Image | 対応 | DINOv2 + Adapter | 基本は同等 |
| Triplane | マルチカメラ効率化 | 非対応（1カメラ） | MiniPamayo には不要 |
| Flex | マルチカメラ・マルチタイムステップ効率化 | 非対応 | |

MiniPamayo は 1 カメラのため、マルチカメラ効率化は不要。ただし Adapter による 256→16 トークン圧縮は、Flex の思想に近い（固定 query による情報圧縮）。

---

## 10. データセット

| 観点 | Alpamayo 0.5B | MiniPamayo（Phase A） | MiniPamayo（Phase B） | MiniPamayo（Phase C） |
|---|---|---|---|---|
| データソース | NVIDIA 内部データ | nuScenes + comma2k19 | + commaCarSegments | + nuPlan + CARLA |
| **データ規模** | **80,000 時間** | **~40 時間** | **~500-2,500 時間** | **~3,000+ 時間** |
| **Alpamayo 比** | **1×** | **1/2,000** | **1/30-160** | **1/20** |
| カメラ | 7 台サラウンド | 1 台フロント | 同左 | 同左 |
| CoC アノテーション | 人間 + VLM auto-labeling | VLM auto-labeling のみ | 同左 | 同左 |
| 地理的多様性 | 25 か国 2,500 都市 | 限定的 | 多様（223 車種） | さらに多様 |

**データ量のギャップが最大の課題**。Phase A（~40 時間）ではパイプライン検証のみ。汎化性能を得るには Phase B 以降が必要。commaCarSegments（~2,500 時間、HuggingFace で公開）が最も入手容易なスケールアップ手段。

詳細は [datasets.md](datasets.md) を参照。

---

## 11. 評価手法

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| Open-loop | minADE6@3s / minADE6@6.4s | ADE / FDE | 同系統の指標 |
| Closed-loop | AlpaSim シミュレーション | なし | |
| 推論品質 | LRM 0-5 スケール | 外部 LLM API で同様の評価は可能 | |
| CoC-Action 一貫性 | ルールベース照合 | 同思想で実装可能 | |
| 実車テスト | RTX 6000 Pro Blackwell | なし | |

---

## 12. まとめ: 実質的な差分

設計書 v0.3 では Alpamayo 0.5B と**コンポーネント選定がほぼ一致**（同一 LLM、同系列 Vision Encoder、同一予測ホライズン）。残る差分は主に**運用・スケールの制約**に起因する。

### v0.3 で解消された差分
| 旧差分（v0.2） | v0.3 での対応 |
|---|---|
| LLM が異なる（SmolLM2-360M vs Qwen2.5-0.5B） | **同一モデル**（Qwen2.5-0.5B）を採用 |
| DINOv2 サイズ（ViT-S vs ViT-B/L？） | **ViT-B/14** に変更。射影ギャップ 1.2倍 |
| 予測ホライズン（3.2s vs 6.4s） | **6.4秒**に統一。128 離散トークン |
| Trajectory Decoder の方式不明 | Alpamayo 準拠 KV-cache Expert（~95M デフォルト） |

### 残る構造的な差
| 差分 | 影響 | 対応方針 |
|---|---|---|
| **カメラ数（1台 vs 7台）** | 360度認識不可、側方/後方の推論不可 | MiniPamayo の制約として受け入れ |
| **時系列入力なし** | 動的な状況変化の推論が弱い | egomotion 入力で部分補完 |

### 残るスケールの差
| 差分 | 影響 | 対応方針 |
|---|---|---|
| **行動予測データ規模** | 汎化性能の限界（80,000h vs ~40h） | commaCarSegments（~2,500h）+ nuPlan（~1,300h）で ~3,000h+ に拡大 → [datasets.md](datasets.md) |
| **ドメイン SFT のスケール差** | 大規模 Physical AI データ vs 小規模公開データ | auto-labeling + RL で補完 |
| **CoC ラベリング** | 人間ラベリングなし | VLM auto-labeling のみ。論文で「十分有効」と報告 |

### 10B 構成との追加的な差分（参考）
10B 構成は 0.5B とも大きく異なる。参考として記載:
- **VLM バックボーン**: Cosmos-Reason-7B（物理 AI 事前学習済み VLM）— DINOv2 ではなく内蔵 ViT を使用
- **Flow Expert**: **~2B params**（ソースコード確認済み）。VLM の `text_config` をコピーし、`expert_cfg` で `hidden_size=2048`, `num_attention_heads=16`, `intermediate_size=8256` にオーバーライド。**36層**は VLM と同じ
- **Expert 条件付け**: VLM の `past_key_values`（KV-cache）を `past_key_values` 引数でそのまま渡す。Expert は VLM と同じ Transformer アーキテクチャのため KV-cache をネイティブに消費可能
- **入力エンコーディング**: Fourier Feature V2（logspace 周波数 × sin/cos）+ MLP (4層, 1024dim) で noisy action を hidden_size=2048 に射影
- **効率的 Vision Encoding**: Triplane / Flex によるマルチカメラ圧縮
- **推論能力**: 7B LLM の方が複雑な因果推論が可能（論文 §6.5 でスケーリング効果を確認）
