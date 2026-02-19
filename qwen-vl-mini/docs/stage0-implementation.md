# Stage 0: 基盤構築 — 具体的実装プラン

## 方針

**最もシンプルな構成で forward + generate が動くことを確認する**。性能最適化は Stage 1 以降で行う。

### 初期構成（最小構成）

| 項目 | 選択 | 値 |
|---|---|---|
| Vision Encoder | `facebook/dinov2-base` | 86M params |
| 入力解像度 | 224×224 | DINOv2 デフォルト |
| VE 出力 | 最終層パッチトークン | (B, 256, 768) |
| Adapter | 2 層 MLP, GELU, Ratio 4 | Linear(768,3072)→GELU→Linear(3072,896) |
| トークン圧縮 | なし | 256 トークンそのまま |
| LLM | `Qwen/Qwen2.5-0.5B` | 494M params |
| 精度 | bf16 | fp16 禁止 |

---

## プロジェクト構成

```
qwen-vl-mini/
├── docs/                          # 設計書・計画書（既存）
├── paper/                         # 参考論文（既存）
└── src/
    └── qwen_vl_mini/
        ├── __init__.py
        ├── model.py               # Step 1-4: VisionEncoder, Adapter, QwenVLMini
        └── test_forward.py        # Step 5: 動作確認スクリプト
```

---

## Step 1: VisionEncoder ラッパー

DINOv2 ViT-B/14 をロードし、パッチトークンを返すラッパー。

```python
class VisionEncoder(nn.Module):
    """DINOv2 ViT-B/14 wrapper."""
    def __init__(self):
        # facebook/dinov2-base をロード
        # requires_grad = False（Stage 1 では frozen）

    def forward(self, pixel_values: Tensor) -> Tensor:
        # pixel_values: (B, 3, 224, 224) — ImageNet 正規化済み
        # → DINOv2 forward
        # → パッチトークン: (B, 256, 768)  ※ CLS トークンは除外
        # 注意: dinov2-base は register token なし版なので除外処理不要
```

**入力前処理**:
- ImageNet 正規化: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Resize(224, 224) + ToTensor() + Normalize()

**実装ポイント**:
- `facebook/dinov2-base` の forward は `last_hidden_state` に CLS + 256 パッチトークンを返す
- **CLS トークン（index 0）を除外してパッチトークンのみ返す**: `output[:, 1:, :]`
- 初期は frozen なので `requires_grad_(False)` で勾配計算を無効化

---

## Step 2: Adapter（2 層 MLP）

DINOv2 の 768 次元を Qwen2.5-0.5B の 896 次元に射影する。

```python
class Adapter(nn.Module):
    """2-layer MLP projector (COMM Table 6: Ratio 4)."""
    def __init__(self, vision_dim=768, llm_dim=896, ratio=4):
        # Linear(768, 3072) → GELU → Linear(3072, 896)

    def forward(self, vision_features: Tensor) -> Tensor:
        # vision_features: (B, 256, 768)
        # → (B, 256, 896)
```

**実装ポイント**:
- 隠れ層次元 = vision_dim × ratio = 768 × 4 = 3072
- GELU 活性化（PyTorch の `nn.GELU()`）
- bias あり（デフォルト）
- Stage 1 では Adapter のみ trainable

---

## Step 3: LLM ラッパー（Qwen2.5-0.5B）

visual tokens をテキスト embedding に注入し、テキスト生成する。

```python
class QwenVLMini(nn.Module):
    """DINOv2 + Adapter + Qwen2.5-0.5B unified model."""
    def __init__(self):
        self.vision_encoder = VisionEncoder()
        self.adapter = Adapter()
        self.llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # 1. 画像 → DINOv2 → パッチトークン (B, 256, 768)
        # 2. パッチトークン → Adapter → visual embeddings (B, 256, 896)
        # 3. input_ids → LLM の token embedding → text embeddings (B, T, 896)
        # 4. [visual_embeddings, text_embeddings] を concat → (B, 256+T, 896)
        # 5. concat した embeddings を LLM に入力（embedding layer をバイパス）
        # 6. labels が渡されたら loss 計算（visual token 部分は ignore_index=-100）

    def generate(self, pixel_values, input_ids, **kwargs):
        # 自己回帰テキスト生成
```

### visual tokens の注入方法

Qwen2.5-0.5B に visual tokens を渡す際の具体的な手順:

```
[visual_tokens (256)] [<|im_start|>user\nDescribe...<|im_end|>\n<|im_start|>assistant\n]
 ↑ DINOv2 → Adapter     ↑ テキスト embedding
 ↑ loss: ignore          ↑ loss: ignore           → [生成テキスト] ← loss: 計算対象
```

1. `pixel_values` を `VisionEncoder` + `Adapter` に通して `visual_embeds` を得る
2. `input_ids` を `llm.model.embed_tokens()` で `text_embeds` に変換
3. `torch.cat([visual_embeds, text_embeds], dim=1)` で結合
4. 結合した `inputs_embeds` を `llm.model()` に渡す（`input_ids` ではなく `inputs_embeds` を使用）
5. loss 計算時は visual tokens 部分のラベルを `-100` に設定

### Qwen2.5 のチャットテンプレート

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{質問}<|im_end|>
<|im_start|>assistant
{回答}<|im_end|>
```

特殊トークン:
- `<|im_start|>`: 会話ターンの開始
- `<|im_end|>`: 会話ターンの終了
- `<|endoftext|>`: テキスト終端

---

## Step 4: attention_mask の構築

visual tokens と text tokens を結合するため、attention_mask も手動で構築する必要がある。

```python
# visual tokens 部分の attention_mask: すべて 1
visual_mask = torch.ones(B, 256, dtype=torch.long, device=device)

# text tokens 部分の attention_mask: tokenizer から取得
text_mask = attention_mask  # (B, T)

# 結合
combined_mask = torch.cat([visual_mask, text_mask], dim=1)  # (B, 256+T)
```

### labels の構築

```python
# visual tokens 部分: ignore (-100)
visual_labels = torch.full((B, 256), -100, dtype=torch.long, device=device)

# text tokens 部分: 回答部分のみ loss 計算
# Qwen2.5 テンプレートで assistant の回答部分のみ有効
# それ以外（system, user, special tokens）は -100
text_labels = ...  # assistant 回答部分のみ input_ids、それ以外は -100

# 結合
combined_labels = torch.cat([visual_labels, text_labels], dim=1)
```

---

## テスト計画

ファイル: `src/qwen_vl_mini/test_forward.py`（実装済み・通過済み）

### T1: モデル構造テスト

| テスト | 合格基準 | 状態 |
|---|---|---|
| forward が loss を返す（labels 有効時） | loss > 0, NaN でない | **通過** (loss=7.46) |
| forward が logits を返す | shape = (B, 256+T, vocab_size) | **通過** |
| generate がトークンを生成する | 出力トークン数 > 入力トークン数 | **通過** (30 tokens) |
| VRAM が 24GB 以内 | peak < 24,000 MB | **通過** (2,522 MB) |
| bf16 で動作 | NaN なし | **通過** |
| パラメータ数が想定通り | total ~585M, trainable ~5M (Stage 1) | **通過** |

### T2: Stage 切り替えテスト

```python
def test_set_stage1():
    """Stage 1: Adapter のみ trainable。"""
    model.set_stage1()
    for p in model.vision_encoder.parameters():
        assert not p.requires_grad
    for p in model.llm.parameters():
        assert not p.requires_grad
    for p in model.adapter.parameters():
        assert p.requires_grad

def test_set_stage2():
    """Stage 2: 全パラメータ trainable。"""
    model.set_stage2()
    for p in model.vision_encoder.parameters():
        assert p.requires_grad
    for p in model.adapter.parameters():
        assert p.requires_grad
    for p in model.llm.parameters():
        assert p.requires_grad
```

---

## Exit 条件

- [x] `model.forward()` が loss を返す（NaN でないこと）
- [x] `model.generate()` がテキストを生成する（空でないこと）
- [x] VRAM 使用量が RTX 4090 (24 GB) 以内
- [x] bf16 で動作すること

---

## 実装順序

```
Step 1: VisionEncoder  — DINOv2 ロード + forward          (30 min)
Step 2: Adapter        — 2 層 MLP                          (15 min)
Step 3: QwenVLMini     — 統合モデル + forward + generate    (2-3 hours)
Step 4: Attention/Label — mask 構築                         (↑ に含む)
Step 5: テスト          — forward + generate + VRAM          (30 min)
```

**最大の作業量は Step 3**（LLM への visual tokens 注入と loss mask の実装）。

---

## 次のステップ（Stage 0 完了後）

Stage 0 が通ったら Stage 1（Feature Alignment）に進む:
- LLaVA-CC3M-Pretrain-595K のダウンロード
- DataLoader 実装
- Adapter のみ学習（VE + LLM frozen）
- 学習ループ + wandb ロギング
