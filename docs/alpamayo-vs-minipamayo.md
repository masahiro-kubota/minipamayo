# Alpamayo 0.5B vs MiniPamayo 設計書：差分分析

本ドキュメントは、[Alpamayo-R1 論文](alpamayo/alpamayo-paper.md)の **0.5B 構成**（DINOv2 + Qwen2.5-0.5B）と [MiniPamayo 設計書 v0.2](design.md) を比較し、差分を整理する。

> **比較対象の選定理由**: Alpamayo には 0.5B / 3B / 7B（10B）の構成がある。MiniPamayo は RTX 4090 単体での技術理解が目的であり、アーキテクチャ構成が最も近い **0.5B 構成**（DINOv2 + Qwen2.5-0.5B）を比較対象とする。10B 構成（Cosmos-Reason-7B + 内蔵 ViT + 2B Flow デコーダ）は規模が異なりすぎるため、特記事項として言及するにとどめる。

---

## 1. 全体アーキテクチャ

両者はいずれも **DINOv2 → Adapter/Projector → LLM → Action Head** という基本パイプラインを共有しており、構成がよく似ている。

| 観点 | Alpamayo 0.5B | MiniPamayo | 差分 |
|---|---|---|---|
| Vision Encoder | DINOv2 | DINOv2 ViT-S/14 | **同系列**（サイズは後述） |
| LLM | Qwen2.5-0.5B | SmolLM2-360M | 同規模の decoder-only LLM |
| カメラ | マルチカメラ（7台）＋時系列 | **1台**（フロント） | **大きな差分** |
| Action Head | Flow Matching | Flow Matching（小規模） | 同思想 |
| 学習戦略 | Action Injection → SFT → RL | 回帰 → 離散 → Flow → SFT → RL | 同思想（MiniPamayo は fail-fast で段階的） |
| 総パラメータ（LLM + Vision） | ~0.5B + α | ~385M | 同オーダー |

---

## 2. Vision Encoder

両者とも DINOv2 を採用しているが、モデルサイズが異なる可能性がある。

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| モデル | DINOv2（ViT-B/L の可能性） | DINOv2 ViT-S/14 (21M) | 0.5B でどのサイズを使うかは論文に明記なし |
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
| DINOv2 → LLM の次元変換 | 必要 | 必要（384 → 960） | |
| トークン数圧縮 | 不明（マルチカメラなので効率化が必要） | 256 → 16~32（積極的に圧縮） | |

### 備考
10B 構成では Cosmos-Reason 内蔵 ViT + 2層 MLP Projector を使うが、0.5B 構成は DINOv2 を外部 Vision Encoder として使うため、MiniPamayo と同様に DINOv2 → LLM を橋渡しする Adapter/Projector が必要になる。この点で **MiniPamayo の Adapter 設計は 0.5B Alpamayo と同じ課題を解いている**。

---

## 4. Language Model

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| モデル | Qwen2.5-0.5B | SmolLM2-360M | |
| パラメータ数 | ~500M | 362M | 同オーダー |
| アーキテクチャ | decoder-only Transformer | decoder-only Transformer | 同じ |
| 視覚入力の事前学習 | なし（テキスト LLM） | なし（テキスト LLM） | **同条件** |
| ドメイン SFT | Cosmos-Reason パイプラインで運転データ SFT 済み | なし | **差分** |

### ドメイン SFT
Alpamayo では 0.5B であっても、Cosmos-Reason のパイプラインで物理 AI 向けの SFT（運転シーン記述、推論トレース等）が行われている。MiniPamayo でも設計書 §3.4 で同様のドメイン SFT を Stage 0 の前段として実施する方針。教師 VLM による auto-labeling で運転シーンの QA データを作成し、SmolLM2 に運転ドメインの視覚理解を獲得させる。

ただし、Alpamayo の Cosmos-Reason パイプラインは大規模データ（Physical AI 全般を含む）で行われているのに対し、MiniPamayo は公開データセットの範囲内での小規模 SFT となる。

---

## 5. アクション表現

設計書 v0.2 では Alpamayo に倣って制御ベース表現を採用しており、**思想は一致**している。

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| 表現 | (加速度 a, 曲率 κ) | 同思想 (a, κ) | **一致** |
| ダイナミクス | ユニサイクル + Euler 積分 | 同じ | **一致** |
| 予測ホライズン | 6.4秒（64 waypoints @ 10Hz） | 3.2秒（32 waypoints @ 10Hz） | MiniPamayo は短縮 |
| 離散トークン数 | 128 tokens (64×2) | 64 tokens (32×2) | ホライズンに比例 |
| GT 逆算 | 最小二乗法 + Tikhonov 正則化 | 同じ | **一致** |

### 差分: 予測ホライズン
MiniPamayo は 3.2秒に短縮している。短い方が学習は容易だが、長距離の計画能力は制限される。Alpamayo の ablation では予測ホライズンの影響は明示的に議論されていないが、6.4秒は交差点通過や車線変更をカバーするのに十分な長さとして設定されている。

---

## 6. Dual Representation（離散トークン + Flow）

設計書 v0.2 でカバー済み。思想は一致。

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| 学習時 | 離散トークン（cross-entropy） | 同じ | **一致** |
| 推論時 | Flow Matching デコーダ | 同じ | **一致** |
| Flow デコーダ規模 | 不明（10B 版は 2B） | 小さな Transformer | 0.5B 版のデコーダサイズは論文に明記なし |
| Flow の条件付け | KV-cache（stop-gradient） | 同思想 | **一致** |

---

## 7. Reasoning — Chain of Causation (CoC)

設計書 v0.2 でカバー済み。

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

設計書 v0.2 でカバー済み。

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

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| データソース | NVIDIA 内部データ（非公開） | 公開データセット | |
| データ規模 | 80,000 時間 | 数千〜数十万フレーム | 桁違いの差 |
| カメラ | 7 台サラウンドビュー | 1 台フロント | |
| CoC アノテーション | 人間 + VLM auto-labeling | VLM auto-labeling のみ | |
| 地理的多様性 | 25 か国 2,500 都市以上 | データセットに依存 | |

これは MiniPamayo の制約上避けられない差分。公開データセット（nuScenes, comma2k19）で学習が回ることを確認するのが目的。

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

設計書 v0.2 では Alpamayo の主要コンセプト（制御ベース表現、Dual Representation、CoC、RL）をすべてカバーしており、**設計思想は 0.5B Alpamayo とほぼ一致している**。残る実質的な差分は以下の通り:

### スケールの差（意図的な簡略化）
| 差分 | 影響 | 対応方針 |
|---|---|---|
| DINOv2 サイズ（ViT-S vs ViT-B/L？） | 視覚特徴の表現力 | RTX 4090 で余裕があれば ViT-B に変更可 |
| LLM サイズ（360M vs 500M） | 言語理解力・推論力 | 同オーダーなので大きな差にはならない |
| 予測ホライズン（3.2s vs 6.4s） | 長距離計画能力 | 学習が安定すれば延長可能 |
| Flow デコーダ規模 | 軌道生成の品質 | 小規模から始めて拡大可能 |

### 構造的な差（アーキテクチャの違い）
| 差分 | 影響 | 対応方針 |
|---|---|---|
| **カメラ数（1台 vs 7台）** | 360度認識不可、側方/後方の推論不可 | MiniPamayo の制約として受け入れ |
| **時系列入力なし** | 動的な状況変化の推論が弱い | egomotion 入力で部分補完 |
| **ドメイン SFT のスケール差** | 大規模 Physical AI データで SFT | 公開データで小規模 SFT（§3.4） |
| **データ規模・品質** | 汎化性能の限界 | 技術理解が目的なので許容 |

### 10B 構成との追加的な差分（参考）
10B 構成は 0.5B とも大きく異なる。参考として記載:
- **VLM バックボーン**: Cosmos-Reason-7B（物理 AI 事前学習済み VLM）— DINOv2 ではなく内蔵 ViT を使用
- **Flow デコーダ**: 2B params の大規模 Transformer
- **効率的 Vision Encoding**: Triplane / Flex によるマルチカメラ圧縮
- **推論能力**: 7B LLM の方が複雑な因果推論が可能（論文 §6.5 でスケーリング効果を確認）
