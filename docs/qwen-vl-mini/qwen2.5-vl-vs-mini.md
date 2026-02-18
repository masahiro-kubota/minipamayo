# Qwen2.5-VL vs Qwen2.5-VL Mini 比較

## 概要

| | Qwen2.5-VL-7B | Qwen2.5-VL Mini |
|---|---|---|
| 目的 | 汎用 Vision-Language Model | 汎用 VLM の小規模再現（技術理解） |
| 総パラメータ数 | ~8.3B（ViT 675M + LLM 7.6B） | ~384M（DINOv2 21M + Adapter ~1M + SmolLM2 362M） |
| GPU | マルチ GPU | RTX 4090 1 枚（24 GB） |

---

## 1. アーキテクチャ

### 1.1 Vision Encoder

| | Qwen2.5-VL | Mini |
|---|---|---|
| モデル | 独自設計 ViT | DINOv2 ViT-S/14 |
| パラメータ数 | ~675M | 21M |
| Hidden size | 1280 | 384 |
| レイヤー数 | 32 | 12 |
| ヘッド数 | 16 | 6 |
| パッチサイズ | 14 | 14 |
| 事前学習 | **CLIP**（テキスト対応あり） | **DINOv2**（自己教師あり、テキスト対応なし） |
| Attention | Window Attention（28層）+ Full Attention（4層） | 標準の Full Self-Attention |
| 位置エンコーディング | 2D-RoPE | DINOv2 のデフォルト（学習済み位置埋め込み） |
| 入力解像度 | **動的**（28 の倍数、任意サイズ） | **固定** 224×224 |
| 動画対応 | あり（3D パッチ、2 フレームグループ化） | なし（画像のみ） |

**最大の違い**: CLIP 事前学習 vs DINOv2 自己教師あり学習。CLIP は視覚特徴がテキスト空間と部分的に対応しているが、DINOv2 はテキストとの対応が一切ない。このギャップは Adapter と Stage 2 の全パラメータ解凍で補う。

### 1.2 Adapter / Merger

| | Qwen2.5-VL | Mini（初期） | Mini（改善） |
|---|---|---|---|
| 方式 | 2層 MLP + 隣接 4 パッチグループ化 | 2層 MLP（圧縮なし） | トークン圧縮付き MLP or Cross-Attention |
| 入力 → 出力トークン数 | N → N/4（4 倍圧縮） | 256 → 256（圧縮なし） | 256 → 64 or 16 |
| 入力 → 出力次元 | 1280×4 → 3584（7B の場合） | 384 → 960 | 384 → 960 |

**Qwen2.5-VL の Merger の仕組み**: 空間的に隣接する 2×2 = 4 パッチの特徴を結合（concat）し、2 層 MLP で LLM の hidden dimension に射影。これにより画像トークン数が 4 分の 1 に削減される。

### 1.3 LLM

| | Qwen2.5-VL | Mini |
|---|---|---|
| モデル | Qwen2.5-7B | SmolLM2-360M |
| パラメータ数 | 7.6B | 362M |
| Hidden size | 3584 | 960 |
| レイヤー数 | 28 | 32 |
| ヘッド数（Q / KV） | 28 / 4（GQA） | 15 / 15（MHA） |
| Vocab size | 151,646 | 49,152 |
| 位置エンコーディング | **MRoPE**（temporal, height, width の 3 成分） | **標準 1D RoPE** |
| コンテキスト長 | 32K（推論時 128K まで拡張可） | 2K〜8K |

**MRoPE**: Qwen2.5-VL は LLM の RoPE を 3 成分に分解し、画像パッチの空間位置を LLM に伝える。SmolLM2 は標準 1D RoPE のみなので、空間位置は DINOv2 の出力特徴に暗黙的に含まれる情報に依存する。

---

## 2. 学習パイプライン

### 2.1 全体比較

| Phase | Qwen2.5-VL | Mini | 備考 |
|---|---|---|---|
| ViT 初期化 | DataComp + 社内データで **CLIP 事前学習** | HuggingFace の **DINOv2 事前学習済み重み** | ViT の出発点が異なる |
| LLM 初期化 | **Qwen2.5-7B** 事前学習済み重み | **SmolLM2-360M** 事前学習済み重み | — |
| Phase 1 | Visual Pre-Training | Feature Alignment | 同思想、対象が異なる |
| Phase 2 | Multimodal Pre-Training | Visual Instruction Tuning | **同方式**（全解凍） |
| Phase 3 | Long-Context Pre-Training | — | スキップ |
| Phase 4 | SFT | — | Cosmos Reason Mini で実施 |
| Phase 5 | DPO | — | Cosmos Reason Mini で実施 |

### 2.2 Phase 1 の比較

| | Qwen2.5-VL Phase 1 | Mini Stage 1 |
|---|---|---|
| 名称 | Visual Pre-Training | Feature Alignment |
| trainable | **ViT のみ** | **Adapter のみ** |
| frozen | LLM | DINOv2 + SmolLM2 |
| データ | Image Caption, Knowledge, OCR | LLaVA-CC3M-Pretrain-595K |
| トークン / サンプル数 | **1.5T トークン** | **595K サンプル** |
| シーケンス長 | 8,192 | — |
| 学習率 | 非公開 | 1e-3 |
| 目的 | ViT を CLIP 初期化から VLM 用に適応 | Adapter に視覚-言語マッピングを学習させる |

**差異の理由**: Qwen2.5-VL の ViT は CLIP 初期化だが VLM 向けに fine-tune が必要。Mini の DINOv2 は frozen のまま、ランダム初期化の Adapter のみ学習して視覚特徴を LLM 空間にマッピングする。

### 2.3 Phase 2 の比較

| | Qwen2.5-VL Phase 2 | Mini Stage 2 |
|---|---|---|
| 名称 | Multimodal Pre-Training | Visual Instruction Tuning |
| trainable | **全パラメータ**（ViT + Merger + LLM） | **全パラメータ**（DINOv2 + Adapter + LLM） |
| データ | Pure text, Interleaved, VQA, Video, Grounding, Agent | LLaVA-Instruct-150K |
| トークン / サンプル数 | **2T トークン** | **150K サンプル** |
| シーケンス長 | 8,192 | — |
| 学習率 | 非公開 | 2e-5 |
| 目的 | 視覚と言語の深い接続を構築 | VQA・会話能力の獲得 |

**同じ点**: 全パラメータを解凍して end-to-end 学習。

### 2.4 Phase 3〜5（Mini ではスキップ / 別プロジェクト）

| Phase | Qwen2.5-VL | Mini での対応 |
|---|---|---|
| Phase 3: Long-Context | seq 長 32,768、0.6T トークン | スキップ（運転タスクに不要） |
| Phase 4: SFT | ~200 万エントリ、ViT frozen | Cosmos Reason Mini で実施 |
| Phase 5: DPO | 選好データ | Cosmos Reason Mini で GRPO として実施 |

---

## 3. データ規模

| | Qwen2.5-VL | Mini | 倍率 |
|---|---|---|---|
| Phase 1 | 1.5T トークン | 595K サンプル | ~数千倍 |
| Phase 2 | 2T トークン | 150K サンプル | ~数万倍 |
| Phase 3 | 0.6T トークン | — | — |
| SFT | ~200 万エントリ | — | — |
| **合計** | **~4.1T トークン** | **~750K サンプル** | 桁違い |

Mini の目的は SOTA 性能ではなく技術理解なので、この規模差は許容。

---

## 4. 意図的な簡略化（Mini で見送った機能）

| 機能 | Qwen2.5-VL | Mini | 理由 |
|---|---|---|---|
| 動的解像度 | 任意サイズ（28 の倍数） | 固定 224×224 | 実装簡略化 |
| 動画入力 | 動的 FPS サンプリング | なし | 将来拡張 |
| MRoPE | 3 成分位置エンコーディング | 標準 1D RoPE | SmolLM2 のアーキテクチャ変更が必要 |
| Window Attention（ViT） | 28/32 層で使用 | なし | DINOv2 は標準 ViT |
| Long-Context 学習 | Phase 3（32K seq） | なし | 運転タスクに不要 |
| 絶対座標グラウンディング | 実ピクセル座標 | なし | 運転タスクでは不要 |
| QwenVL HTML フォーマット | 文書要素を HTML に統一 | なし | OCR/文書タスクは対象外 |

---

## 5. 構造的な差異（性能に影響しうる）

| 差異 | 影響 | 対策 |
|---|---|---|
| CLIP vs DINOv2（テキスト対応の有無） | Adapter の学習負担が大きい | Stage 1 で十分な学習 + Stage 2 で DINOv2 解凍 |
| ViT サイズ（675M vs 21M） | 視覚特徴の表現力が低い | DINOv2 は self-supervised で効率的。21M でも高品質な汎用特徴 |
| LLM サイズ（7.6B vs 362M） | 言語生成能力の上限が低い | 短い回答に限定。技術理解が目的なので許容 |
| トークン圧縮（4 倍 vs なし） | LLM コンテキストの圧迫 | 隣接パッチグループ化 or Cross-Attention Pooling で対応 |
| MRoPE なし | 画像パッチの空間位置が LLM に明示的に伝わらない | DINOv2 の出力に暗黙的な空間情報が含まれるため致命的ではない |
| GQA vs MHA | KV キャッシュ効率の差 | SmolLM2 が小さいので影響は軽微 |

---

## 6. 設計思想の一致点

上記の差異にもかかわらず、以下の設計思想は Qwen2.5-VL と一致している:

1. **2 段階学習**: Phase 1（アライメント）→ Phase 2（全パラメータ解凍）の段階的学習
2. **Phase 1 で LLM を frozen**: テキスト生成能力を保護しつつ視覚-言語接続を構築
3. **Phase 2 で全解凍**: end-to-end 学習で視覚と言語の深い接続を構築
4. **MLP ベースの Adapter**: 2 層 MLP で視覚特徴を LLM 空間に射影
5. **事前学習済みコンポーネントの組み合わせ**: ViT と LLM を別々に事前学習し、Adapter で接続
6. **Cross-entropy loss**: 全ステージで next-token prediction の標準損失を使用
