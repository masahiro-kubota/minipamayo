# 小型 VLM のデータ量・性能比較

## 目的

Qwen2.5-VL Mini（DINOv2 + Qwen2.5-0.5B, ~585M params）の Stage 2 データ量が適切かを判断するため、
同規模の VLM が使用している学習データ量と POPE 性能を比較する。

---

## Stage 2 (Instruction Tuning) データ量比較

| モデル | 総パラメータ | Vision Encoder | LLM | Stage 2 データ | POPE Acc |
|--------|-------------|----------------|-----|---------------|----------|
| **TinyLLaVA-Qwen2-0.5B** | ~500M | SigLIP-SO | Qwen2-0.5B | **665K** (ShareGPT4V-mix) | **86.6%** |
| **TinyLLaVA-Phi-2** | ~3.1B | SigLIP-SO | Phi-2 (2.7B) | **665K** (ShareGPT4V-mix) | 86.4% |
| **Imp-v1** | ~3B | SigLIP-SO | Phi-2 (2.7B) | **665K** (LLaVA-mix) | 86.1% |
| **Imp-v1.5** | ~3B | SigLIP-SO | Phi-2 (2.7B) | **1M** (665K + OCR/Chart 32K + GPT4V 330K) | 88.0% |
| **SmolVLM-256M** | ~256M | SigLIP-SO (shape-optimized) | SmolLM2-135M | **Cauldron** (~1.8M) | ~75% |
| **MobileVLM v2 (1.7B)** | ~1.7B | CLIP ViT-L | MobileLLaMA 1.4B | **665K** (LLaVA-mix) | 84.3% |
| **Bunny-v1.0-3B** | ~3B | SigLIP-SO | Phi-2 (2.7B) | **695K** (SVIT-mix) | 86.8% |
| **LLaVA-1.5 (7B)** | ~7B | CLIP ViT-L | Vicuna-7B | **665K** (LLaVA-mix) | 85.9% |
| **Qwen-VL Mini (Stage 2)** | ~585M | DINOv2 ViT-B/14 | Qwen2.5-0.5B | **158K** | **59.8%** |
| **Qwen-VL Mini (Stage 2.1)** | ~585M | DINOv2 ViT-B/14 | Qwen2.5-0.5B | **695K** (Bunny SVIT-mix) | ? |

### POPE スコアの直接比較可能性

上記の POPE Acc は全モデルで**同一の評価プロトコル**に基づいており、直接比較可能:

- **評価画像**: COCO val2014 の 500 枚（全モデル共通）
- **質問セット**: POPE が提供する固定の yes/no 質問ファイル（random/popular/adversarial 各 3,000 問）
- **報告スコア**: 通常は 3 バリアントの平均 Accuracy
- **val2014 除外**: 学習データから val2014 を除外するのは標準的慣行（data leak 防止）。LLaVA-1.5 の `prepare_mix665k` 相当の処理は全モデルが実施

---

## 分析

### 665K が事実上の標準

上記のほぼ全てのモデルが **665K 以上**のデータで Stage 2 を実施している。
LLaVA-mix-665K または ShareGPT4V-mix-665K が小型 VLM の事実上の標準データセット。

### 最も近い比較対象: TinyLLaVA-Qwen2-0.5B

- **同じ Qwen2 系 0.5B LLM** を使用
- Vision Encoder は SigLIP（我々は DINOv2）
- **665K で POPE 86.6%** を達成
- 我々の 158K → 59.8% との差は、データ量の差（665K vs 158K）が主要因の一つと考えられる

### データ量と POPE の関係

| データ量 | 代表モデル | POPE 範囲 |
|----------|-----------|-----------|
| 158K | Qwen-VL Mini (Stage 2) | 59.8% |
| 665K | LLaVA-1.5, TinyLLaVA, Imp-v1, MobileVLM v2 | 84-87% |
| ~1M | Imp-v1.5 | 88% |
| ~1.8M | SmolVLM-256M (Cauldron) | ~75%* |

*SmolVLM-256M は 135M LLM のため、データ量が多くてもモデル容量の制約で性能が制限される。

### Vision Encoder の影響

DINOv2 は CLIP/SigLIP と比べて text-aware な特徴が弱い（OCR が苦手）。
しかし POPE は物体存在判定であり、DINOv2 の物体認識能力は高いため、
データ量とデータ多様性を揃えれば POPE での大幅な改善が期待できる。

---

## 結論

**382K (COCO のみ) では不十分**である可能性が高い。根拠:

1. 同じ LLM を使う TinyLLaVA が 665K で POPE 86.6% を達成している
2. 665K 未満で Stage 2 を実施している小型 VLM は見当たらない
3. 382K にしている理由は「非 COCO 画像未ダウンロード」という消極的理由のみ

**推奨**: 非 COCO データ（GQA, TextVQA, OCR-VQA, VG）の画像をダウンロードし、
665K 全量で Stage 2.1 を実施すべき。追加コストは:
- ダウンロード: ~20-40 GB、数十分〜数時間
- 学習時間: ~7h → ~12h（665K × 1 epoch）
- コード変更: prepare_mix665k.py のフィルタを全画像ソースに拡張するだけ

---

## 出典

- TinyLLaVA: Zhou et al., "TinyLLaVA: A Framework of Small-scale Large Multimodal Models" (2024)
- Imp: Shao et al., "Imp: Highly Capable Large Multimodal Models for Mobile Devices" (2024)
- SmolVLM: HuggingFace (2024)
- MobileVLM v2: Chu et al., "MobileVLM V2" (2024)
- Bunny: He et al., "Efficient Multimodal Learning from Data-centric Perspective" (2024)
- LLaVA-1.5: Liu et al., "Improved Baselines with Visual Instruction Tuning" (2023)
