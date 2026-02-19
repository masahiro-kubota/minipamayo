# Stage 2 学習レポート: Visual Instruction Tuning

## 概要

| 項目 | 値 |
|------|-----|
| タスク | Visual Instruction Tuning (全パラメータ fine-tune) |
| データ | LLaVA-Instruct-150K (157,712 samples) |
| 画像 | COCO 2014 train (82,783 images) |
| Epochs | 2 |
| 総ステップ数 | 2,464 |
| Global Batch Size | 128 (batch=1 x grad_accum=128) |
| 学習率 | VE: 1e-5, LLM+Adapter: 2e-5 (cosine schedule) |
| Warmup | 73 steps (3%) |
| Max Sequence Length | 2,048 tokens |
| Weight Decay | 0.1 |
| Peak VRAM | 8,009 MB |
| 学習時間 | 約 5.5 時間 |
| GPU | NVIDIA GeForce RTX 4090 (24GB) |
| wandb | [stage2-instruct](https://wandb.ai/norikenpi-individual/qwen-vl-mini/runs/zttkiiql) |

## モデル構成

- **VisionEncoder**: DINOv2 ViT-B/14 (86M params) - **解凍**
- **Adapter**: 2-layer MLP 768→3072→896 (5M params)
- **LLM**: Qwen2.5-0.5B (494M params)
- **合計**: 585,729,024 params (100% trainable)

## Loss 統計

| 指標 | 値 |
|------|-----|
| 初期 loss (Step 200) | 1.7731 |
| 最小 loss | 0.6101 |
| 最大 loss | 2.4889 |
| 平均 loss | 1.4966 |
| 最終付近 loss (Step 2460) | 1.6459 |

### Loss 推移の特徴

- Stage 1 adapter 重みから開始し、初期 loss ~1.77
- Step 200-500 で急激に低下、その後プラトーに入る
- loss は 0.6-2.5 の範囲で大きく振動（Instruction tuning の性質上正常）
  - バッチごとのタスク難易度差が大きい（短い QA vs 長い説明文）
  - 最後の1 microbatch の loss のみ記録のためノイジー
- grad_norm は 3.5→2.7 へ徐々に低下（収束方向）

## T2 診断チェック結果

| チェック | 結果 |
|----------|------|
| NaN/Inf 検出 | ✓ 発生なし |
| Learning check (Step 250) | ✓ 1.77 → 1.59 |
| DINOv2 重み更新 | ✓ 更新確認 |
| High gradient norm | 7回検出（最大 51.59 at Step 654）、クリッピング済み |

### Gradient Norm スパイクについて

Step 654 で grad_norm=51.59 のスパイクが発生したが、max_grad_norm=1.0 でクリッピングされており、
loss の発散は見られなかった。これは Instruction tuning データの一部に特異なサンプル（非常に長い応答
or 珍しい画像）が含まれていたことが原因と推測される。

## チェックポイント

| ファイル | サイズ |
|----------|--------|
| checkpoint-2464.pt (最終) | 3.8 GB |
| checkpoint-99 ~ checkpoint-2399 (中間) | 各 3.8 GB x 24 |

**注意**: 合計 95GB。ディスク節約のため、最終チェックポイント + 数個の中間チェックポイントのみ残す推奨。

## 学習中の対処

1. **Python 出力バッファリング問題**: バックグラウンド実行で stdout がバッファリングされ、ログが見えない問題。`PYTHONUNBUFFERED=1` + `flush=True` の `log()` ヘルパーで解決。

2. **--resume 機能追加**: バッファリング修正のための再起動に伴い、checkpoint-199 から再開する機能を train_stage2.py に追加。

## 次のステップ

1. 推論テスト（COCO 画像 + 質問で応答品質を確認）
2. 不要チェックポイントの整理（ディスク節約）
3. Cosmos Reason Mini の設計・実装へ
