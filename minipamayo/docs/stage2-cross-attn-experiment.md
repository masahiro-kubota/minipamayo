# Stage 2: Cross-Attention Decoder 実験結果

## 背景

KV-cache Expert (Qwen2 Transformer, 146M params) が val CFM ≈ 2.1 でプラトーした。
原因を Qwen2.5-0.5B の KV head 数 (=2) によるボトルネックと仮定し、
VLM hidden states (896 dim) を直接 cross-attention で参照する decoder を実装・検証した。

## 実装した Cross-Attention Decoder

- `CrossAttentionDecoder`: hidden=256, layers=4, heads=4, ~6.7M params
- `FlowTransformerBlock`: AdaLN + self-attention + cross-attention + FFN
- `SinusoidalTimeEmbedding`: sin/cos + MLP
- per-waypoint Fourier V2 encoding (ActionInProj 再利用)
- action 正規化、Beta(2,5) time distribution
- VLM hidden states (B, L, 896) を cross-attention の K/V として使用

## 結果

### Cross-Attention Decoder (2 回の学習)

| Run | Epoch | Train CFM | Val CFM | Notes |
|-----|-------|-----------|---------|-------|
| 1   | E1    | 2.460     | **2.257** | Best val |
| 1   | E2    | 2.399     | 2.308   | |
| 1   | E3    | 2.380     | 2.287   | |
| 1   | E4    | 2.382     | 2.281   | |
| 1   | E5    | (killed)  | -       | プロセス終了 |
| 2   | E1    | 2.481     | 2.308   | warmup 中 |
| 2   | E2    | 2.413     | **2.240** | Best val |
| 2   | E3    | 2.410     | 2.255   | |
| 2   | E4    | 2.392     | 2.306   | 悪化 → 停止 |

### 比較

| Decoder | Params | Best Val CFM | 状態 |
|---------|--------|-------------|------|
| KV-cache Expert | 146M | **2.109** | E2 best → plateau |
| Cross-Attention | 6.7M | **2.240** | E2 best → 悪化 |

## 結論

- Cross-Attention decoder は KV-cache Expert (2.109) を超えられなかった (2.240)
- 「KV head ボトルネック」仮説は棄却
- ただし、パラメータ数の差 (22倍) も一因の可能性あり

## 旧実装との比較で判明した問題

commit 54346b5 以前の旧実装（曲がれていた）と、現行実装の乖離が大きい:

| 項目 | 旧実装 (動いていた) | KV-cache Expert | Cross-Attn (新) |
|------|---------------------|----------------|-----------------|
| VLM conditioning | mean-pool → (B, 896) | KV-cache | full seq (B, L, 896) |
| Action encoding | Linear(action_dim, hidden) | Fourier V2 per-waypoint | Fourier V2 per-waypoint |
| Transformer | self-attn only, 2-token | Qwen2 (KV-cache) | self-attn + cross-attn |
| Time sampling | Uniform | Beta(2,5) | Beta(2,5) |
| Action normalization | なし | あり | あり |
| K | 64 | 6 | 6 |
| Default params | 512d, 12L, 8H | 640d, 24L, 10H | 256d, 4L, 4H |

→ 旧実装に戻して ablation study を行い、どの変更が性能劣化の原因か特定する。
