# 小規模 VLM 構築の知見まとめ

論文・ブログのサーベイに基づく、~500M〜1B 規模の VLM 構築における実践的知見。
特に DINOv2 ViT-B/14 + Qwen2.5-0.5B（~582M）の構成に関連する内容を整理。

---

## 1. アーキテクチャ設計

### 1.1 Vision Encoder と LLM のバランス

**知見**: 小さい LLM に大きすぎる Vision Encoder を付けても逆効果。バランスが重要。

| 出典 | 内容 |
|---|---|
| SmolVLM | 428M エンコーダ + 135M LM で性能低下。93M SigLIP-B/16 + 360M SmolLM2 が最適バランス |
| Idefics2 | LLM の品質改善（+5.1pt）がビジョンエンコーダ改善（+3.3pt）より VLM 性能に寄与 |

**Mini への示唆**: DINOv2 ViT-B/14（86M）+ Qwen2.5-0.5B（494M）は SmolVLM-500M と同等の規模感。エンコーダ比率 ~15% は適切なバランス。

### 1.2 コネクタ（Adapter）の選択

**知見**: 小規模 VLM では MLP コネクタが Resampler を上回る傾向。

| 出典 | 内容 |
|---|---|
| TinyLLaVA | MLP が Resampler を全ベンチマークで上回る。小規模モデルでは MLP の情報損失が少ない |
| Idefics2 | Perceiver Resampler で 729→64 トークンに削減しても +8.5pt 改善（ただし 8B モデル） |
| COMM | DINOv2 には **2 層以上の MLP が必須**。1 層 Linear では大幅に性能低下 |

**Mini への示唆**: 現在の 2 層 MLP（768→896→896）は妥当。トークン圧縮を行う場合は Pixel Shuffle + MLP が有力。

### 1.3 トークン圧縮

**知見**: 小型 VLM ほど積極的なトークン圧縮が有効。

| 方式 | 圧縮率 | 採用例 | 備考 |
|---|---|---|---|
| Pixel Shuffle r=2 | 4 倍（256→64） | Idefics3, InternVL | 大型モデル向け |
| Pixel Shuffle r=4 | 16 倍（256→16） | SmolVLM-256M/500M | **小型モデルに最適** |
| 隣接 4 パッチグループ化 | 4 倍（256→64） | Qwen2.5-VL | 空間情報を明示的に保持 |
| Cross-Attention Pooling | 可変（256→任意） | Perceiver Resampler | 学習可能、柔軟 |

**Mini への示唆**: DINOv2 の 256 トークンに対して **Pixel Shuffle r=4（16 トークン）** または **r=2（64 トークン）** を検討。SmolVLM の知見では r=4 が ~500M モデルに有効。

---

## 2. DINOv2 固有の注意点

### 2.1 CLIP vs DINOv2

| 側面 | CLIP/SigLIP | DINOv2 |
|---|---|---|
| テキストアライメント | 事前学習済み（即利用可能） | **なし（別途学習必要）** |
| 空間的局所情報 | 弱い（グローバル） | **強い（細粒度パッチ特徴）** |
| OCR・テキスト認識 | 優位 | 弱い |
| 密予測（セグメンテーション等） | 粗い | **優秀** |

出典: COMM (arXiv:2310.08825), Cambrian-1 (NeurIPS 2024)

### 2.2 DINOv2 を VLM で使うためのアライメント手法

| 手法 | 概要 | コスト |
|---|---|---|
| MLP プロジェクター（LLaVA 方式） | 2 層 MLP で LLM 空間に射影 | 低い。**現在の Mini 設計** |
| チャネル連結（OpenVLA 方式） | DINOv2 + SigLIP を連結→MLP | 中。エンコーダ 2 つ必要 |
| SAIL（CVPR 2025） | 23M ペアで DINOv2 にテキストアライメントを学習 | 低い。A100 1 台、~5 時間 |
| dino.txt（CVPR 2025） | フローズン DINOv2 + 2 Transformer ブロック | 低い |

**Mini への示唆**: DINOv2 単独使用なら 2 層 MLP + 十分な Alignment データで対応可能。CLIP との融合は性能向上が見込めるが、実装複雑度が増す。

### 2.3 DINOv2 のファインチューニング

| 戦略 | 詳細 |
|---|---|
| **フローズン推奨** | DINOv2 の汎用特徴は高品質。アンフリーズの改善は 2-5% 程度 |
| Stage 1: Adapter のみ学習 | エンコーダ + LLM フローズン、プロジェクターのみ学習 |
| Stage 2: LLM 解凍 | プロジェクター + LLM（or LoRA）を学習。**全解凍時は訓練発散リスクあり** |
| **TinyLLaVA の知見** | 小規模モデルでは ViT の一部 unfreeze が有効（前半 12 層フローズン、後半解凍） |

---

## 3. 学習戦略

### 3.1 ハイパーパラメータの目安

| 段階 | 学習率 | バッチサイズ | エポック | 凍結部分 |
|---|---|---|---|---|
| Stage 1（Alignment） | 1e-3 | 256 | 1 | ViT + LLM（Adapter のみ学習） |
| Stage 2（Instruction Tuning） | 2e-5 | 128 | 1〜2 | ViT（Adapter + LLM 学習） |

出典: TinyLLaVA, LLaVA-Phi, LLaVA-1.5

### 3.2 エポック数の注意

**知見**: 2 エポックが最適。3 エポック以上で過学習リスク。

| 出典 | 内容 |
|---|---|
| Imp | 1 エポック: -0.5pt、2 エポック: 最適、3 エポック: -0.4pt（過学習） |
| SmolVLM | チェックポイント選択が重要。長く学習すると DocVQA が低下 |

### 3.3 LoRA vs 完全微調整

| 出典 | 内容 |
|---|---|
| Imp | **LoRA rank=256 が完全微調整を上回る**（sub-3B モデル） |
| Idefics2 | バックボーン全解凍は訓練発散。**LoRA/DoRA で安定化** |
| LLaVA-Phi | 2.7B では LoRA なし完全微調整が十分機能 |

**Mini への示唆**: Qwen2.5-0.5B（494M）では完全微調整が可能だが、訓練が不安定な場合は LoRA を検討。

---

## 4. データ戦略

### 4.1 データ品質 > データ量

**知見**: Sub-1B モデルではデータ量を増やしても飽和・劣化する場合がある。

| 出典 | 内容 |
|---|---|
| TinyLLaVA | 1.1B LLM では ShareGPT4V（1246K）で POPE が低下。**容量不足だとデータ増加が逆効果** |
| SmolVLM | LLM-SFT データの再利用で画像 -6.5%、動画 -3.7%。**専用データが必要** |
| Imp | テストセットとのデータ重複を排除することが重要 |

### 4.2 CoT データの制限

**知見**: 小型モデルでは CoT（Chain-of-Thought）データを **0.02〜0.05% に制限**すべき。

出典: SmolVLM — 過剰な CoT は小規模モデルの容量を圧迫し、視覚表現を損なう。

**Cosmos Reason Mini への示唆**: SFT で CoT 推論トレースを付与する際、推論 QA の比率を全体の数パーセント以下に抑える。

### 4.3 位置トークンの設計

**知見**: 文字列ベースの位置トークン（`<row_1_col_2>` 等）は「OCR loss plague」を引き起こす。**学習可能な位置埋め込みを使用すべき**。

出典: SmolVLM — 文字列位置トークンは OCR 損失の不安定化を引き起こし、学習済み位置埋め込みに置き換えることで改善。

---

## 5. 落とし穴・注意点まとめ

| # | 落とし穴 | 出典 | 対策 |
|---|---|---|---|
| 1 | ViT と LLM のサイズ不均衡 | SmolVLM | バランスの取れた配分（~15:85） |
| 2 | バックボーン全解凍時の訓練発散 | Idefics2 | LoRA/DoRA で安定化 |
| 3 | Sub-1B でデータ量を増やすと劣化 | TinyLLaVA | データ品質を重視 |
| 4 | 3 エポック以上で過学習 | Imp | 2 エポックが最適 |
| 5 | 文字列位置トークンの不安定性 | SmolVLM | 学習可能トークンを使用 |
| 6 | LLM-SFT データの流用 | SmolVLM | マルチモーダル専用データを用意 |
| 7 | 過剰な CoT データ | SmolVLM | 全体の 0.02〜0.05% に制限 |
| 8 | DINOv2 単独のテキスト理解の弱さ | COMM | MLP 2 層以上で補う。CLIP 融合も検討 |
| 9 | チェックポイント選択ミス | SmolVLM | 頻繁に保存し、複数指標で最適を選択 |

---

## 6. Mini（DINOv2 ViT-B/14 + Qwen2.5-0.5B）への具体的推奨

### アーキテクチャ
- **Adapter**: 2 層 MLP（768→896→896）を維持。Pixel Shuffle r=4 でトークン圧縮（256→16）を強く推奨
- **DINOv2**: Stage 1 ではフローズン。Stage 2 では後半レイヤーのみ解凍を検討（TinyLLaVA Share recipe）
- **コンテキスト長**: トークン圧縮により 256→16 にすれば、テキストトークンに余裕ができる

### 学習
- Stage 1: lr=1e-3、Adapter のみ学習、1 エポック
- Stage 2: lr=2e-5、全パラメータ or LLM + Adapter 学習、1〜2 エポック
- 訓練不安定時は LoRA（rank=256）を検討
- チェックポイントを 25 ステップごとに保存

### データ
- データ品質を最優先。量で補おうとしない
- CoT 推論トレースは全体の 0.05% 以下に
- LLM の SFT データを流用しない

---

## 参考文献

- [SmolVLM: Redefining small and efficient multimodal models](https://arxiv.org/abs/2504.05299) — HuggingFace, 2025
- [TinyLLaVA: A Framework of Small-scale Large Multimodal Models](https://arxiv.org/abs/2402.14289) — 2024
- [What matters when building vision-language models? (Idefics2)](https://arxiv.org/abs/2405.02246) — NeurIPS 2024
- [Imp: Highly Capable Large Multimodal Models for Mobile Devices](https://imp-vl.github.io/) — 2024
- [LLaVA-Phi: Efficient Multi-Modal Assistant with Small Language Model](https://arxiv.org/abs/2401.02330) — 2024
- [MoE-LLaVA: Mixture of Experts for Large Vision-Language Models](https://arxiv.org/abs/2401.15947) — TMM 2025
- [From CLIP to DINO (COMM)](https://arxiv.org/abs/2310.08825) — 2023
- [SAIL: Assessing and Learning Alignment of Unimodal Vision and Language Models](https://arxiv.org/abs/2412.04616) — CVPR 2025
- [DINOv2 Meets Text (dino.txt)](https://arxiv.org/abs/2412.16334) — CVPR 2025
- [Talking to DINO (Talk2DINO)](https://arxiv.org/abs/2411.19331) — ICCV 2025
- [Cambrian-1](https://arxiv.org/abs/2406.16860) — NeurIPS 2024
- [OpenVLA](https://arxiv.org/abs/2406.09246) — 2024
- [Eagle: Mixture of Encoders](https://arxiv.org/abs/2408.15998) — NVIDIA, 2024
