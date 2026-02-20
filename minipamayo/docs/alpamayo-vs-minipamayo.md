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
| Trajectory Decoder | Flow Matching | Flow Matching (~150M) | 同思想（LLM の ~30%） |
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

同じ LLM（Qwen2.5-0.5B）の語彙に離散トークンを追加し、同じ GQA 構造の KV-cache で条件付けるため、**実質的に同一の設計**。

| 観点 | Alpamayo 0.5B | MiniPamayo | 備考 |
|---|---|---|---|
| 学習時 | 離散トークン（cross-entropy） | 同じ | **同一**（同一 LLM vocab） |
| 推論時 | Flow Matching デコーダ | 同じ | **同一** |
| Trajectory Decoder 規模 | 不明（10B 版は 2B） | ~150M（LLM の ~30%） | 0.5B 版のサイズは論文に明記なし |
| Flow の条件付け | KV-cache（stop-gradient） | 同じ GQA の KV-cache | **同一** |

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
| Trajectory Decoder の比率不明 | **~150M**（LLM の ~30%、Alpamayo 10B と同比率） |

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
- **Flow デコーダ**: 2B params の大規模 Transformer
- **効率的 Vision Encoding**: Triplane / Flex によるマルチカメラ圧縮
- **推論能力**: 7B LLM の方が複雑な因果推論が可能（論文 §6.5 でスケーリング効果を確認）
