# VLM 学習段階の比較: LLaVA vs Qwen2.5-VL

## LLaVA-1.5 の学習段階（2段階）

| | Stage 1: Feature Alignment | Stage 2: Visual Instruction Tuning |
|---|---|---|
| **学習対象** | Projector のみ | 全パラメータ |
| **凍結** | ViT (CLIP) + LLM | なし |
| **データ** | CC3M-595K キャプション | LLaVA-Instruct-mix665K |
| **学習率** | 1e-3 | 2e-5 |
| **バッチサイズ** | 256 | 128 |
| **エポック** | 1 | 1 |
| **目的** | 視覚特徴→LLM空間のマッピング | 画像QA・会話能力の獲得 |

## Qwen2.5-VL の学習段階（5段階）

| | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|---|---|---|---|---|---|
| **名称** | Visual Pre-Training | Multimodal Pre-Training | Long-Context Pre-Training | SFT | DPO |
| **学習対象** | ViT + Merger | 全パラメータ | 全パラメータ | LLM + Merger | LLM + Merger |
| **凍結** | LLM | なし | なし | ViT | ViT |
| **データ量** | 1.5T tokens | 2T tokens | 0.6T tokens | ~200万エントリ | 選好データ |
| **seq長** | 8,192 | 8,192 | 32,768 | — | — |

## 凍結・解凍パターンの比較

| 段階 | LLaVA | Qwen2.5-VL |
|---|---|---|
| 初期学習 | ViT frozen, LLM frozen, **Projector のみ** | **ViT trainable**, LLM frozen |
| 本学習 | **全解凍** | **全解凍** |
| 後段階 | — | **ViT frozen**, LLM + Merger のみ |

**注目点**: LLaVA は初期段階で Projector のみ学習するが、Qwen2.5-VL は Phase 1 で ViT 自体を学習する。方向性が逆。

## Qwen2.5-VL Mini の位置づけ

**LLaVA 方式を踏襲**。用語も LLaVA に合わせて "Stage" を使用する。
Qwen2.5-VL の "Phase" は論文参照時のみ使用。

| Mini | LLaVA との対応 | 学習対象 | 凍結 | データ |
|---|---|---|---|---|
| Stage 1 | LLaVA Stage 1 と同方式 | Adapter のみ | DINOv2 + LLM | 595K キャプション |
| Stage 2 | LLaVA Stage 2 と同方式 | 全パラメータ | なし | 150K 指示データ |

LLaVA との差異:
- ViT が CLIP ではなく DINOv2（テキスト対応なし）
- Stage 2 で VE の学習率を分離（LLM 2e-5, VE 1e-5）
- Stage 2 が 2 エポック（Imp 論文で最適と判明）
