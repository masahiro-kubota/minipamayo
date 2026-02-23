# Stage 2 Loss Plateau Investigation

## 問題

Stage 2 Flow Matching Expert の val CFM loss が **~2.0 でプラトー**し、設計書の目標 (<1.0) に到達できない。
カーブシーンで軌道が曲がらず、kappa 予測が常にほぼ 0。

## 学習ログ

| Epoch | Train CFM | Val CFM | 備考 |
|-------|-----------|---------|------|
| 1 | 2.457 | 2.247 | |
| 2 | 2.414 | 2.185 | |
| 3 | 2.260 | 2.065 | |
| 4 | 2.219 | 2.049 | |
| 5 | 2.212 | 2.087 | **ここからプラトー** |
| 6 | 2.217 | 2.019 | |
| 7 | 2.202 | 2.026 | |
| 8 | 2.197 | 2.026 | 設計書: 0.986 |
| 9 | 2.198 | 2.018 | |
| 10 | (killed) | — | 改善の兆候なし |

設計書 (`stage2-flow-matching.md`) では epoch 8 で val CFM = 0.986 と記載。
ただし設計書の数値は**旧 cross-attention アーキテクチャ**の結果であり、現行の KV-cache Expert では再現されていない。

---

## KV-cache 診断結果

`scripts/diagnose_kv_cache.py` で検証（50 サンプル）。

### Test 1: CFM loss 比較

| 条件 | Mean CFM Loss | 差分 |
|------|---------------|------|
| Correct KV (正しい画像の KV-cache) | 2.110 | baseline |
| Shuffled KV (別画像の KV-cache) | 2.157 | **+2.2%** |
| Zero KV (全要素ゼロ) | 2.041 | **-3.2%** |

**結論**: Expert は VLM の KV-cache を使っていない。
ゼロ KV のほうが loss が低い = KV-cache が**性能を悪化**させている。

### Test 2: 予測軌道比較

| メトリクス | 値 |
|---|---|
| Mean L2 diff (correct vs shuffled) | 2.14 |

一見大きいが、kappa 値を見ると両方とも ≈0 (0.01-0.05 範囲)。
差は主に acceleration 側。**kappa（曲率）は KV-cache に無関係に常にほぼ 0**。

### Test 3: KV-cache 統計

| Layer | Key Norm | Value Norm | Key Std | Value Std |
|-------|----------|------------|---------|-----------|
| 0 | **1467.0** | 1.5 | **32.15** | 0.034 |
| 12 | 72.5 | 11.5 | 1.60 | 0.253 |
| 23 | 78.2 | 33.4 | 1.73 | 0.734 |

**Layer 0 の Key norm が異常に大きい** (Value の ~1000 倍)。
RoPE は norm を保存する回転なので、元の K projection が大きな値を出力している。

KV-cache の形状: `(1, 2, 16, 64)` = batch=1, KV_heads=2, seq=16, head_dim=64

---

## アーキテクチャ分析

### Expert (TrajectoryDecoder)

```
hidden_size:        640
num_attention_heads: 10  (Q heads)
num_key_value_heads:  2  (KV heads, GQA)
head_dim:            64
num_hidden_layers:   24
rope_theta:     1000000.0
```

### VLM (Qwen2.5-0.5B)

```
hidden_size:        896
num_attention_heads: 14  (Q heads)
num_key_value_heads:  2  (KV heads, GQA)
head_dim:            64
num_hidden_layers:   24
rope_theta:     1000000.0
```

**互換性**: KV heads=2, head_dim=64 で一致 → KV-cache の shape は互換。

### Attention の流れ

```
Expert Q (from hidden=640) · VLM K (from hidden=896)^T → attention scores
```

Layer 0 の VLM Key std=32.15 → attention score ≈ Q·K^T/√64 が非常に大きくなり、
softmax が飽和 → 勾配消失 → Expert は KV-cache を無視するように学習。

---

## Vanilla Qwen vs 自作 VLM の KV-cache 比較

自作 VLM の問題か？ → **違う。Qwen2.5-0.5B 固有の特性。**

| | Vanilla Qwen (テキスト入力) | 自作 VLM (画像入力) |
|---|---|---|
| **入力 std** | 0.0145 | 0.998 (69倍) |
| **Layer 0 K std** | **32.18** | **32.15** |
| **Layer 0 V std** | 0.026 | 0.034 |
| **Layer 12 K std** | 1.96 | 1.60 |
| **Layer 23 K std** | 2.00 | 1.73 |

入力スケールが 69 倍異なるのに、KV-cache はほぼ同一。
LLM 内部の RMSNorm が入力スケールの差を吸収している。

**Layer 0 の K std ≈ 32 は Qwen2.5-0.5B の K projection の固有特性であり、
自作 VLM のアダプタや学習が原因ではない。**

---

## 根本原因

### Qwen2.5-0.5B の Key スケール問題 + 学習ダイナミクスの失敗

1. Qwen2.5-0.5B の Layer 0 は K std ≈ 32 を出力する（テキスト/画像問わず）
2. Expert の Q projection はランダム初期化（std ≈ 0.01-0.1）
3. Q·K^T/√64 の attention score が初期から巨大 → softmax 飽和 → 勾配消失
4. Expert は「KV-cache を無視する」局所解に落ちる
5. KV-cache なしの周辺分布で CFM loss ≈ 2.0 に収束（条件付き分布に移行できない）

---

## Alpamayo 実装との比較

`related_repos/alpamayo/` のコードおよび HuggingFace の `nvidia/Alpamayo-R1-10B` config.json を分析。

### Alpamayo のアーキテクチャ

| | **Alpamayo-R1-10B** | **MiniPamayo** |
|---|---|---|
| VLM | Qwen3-VL-8B (hidden=4096) | Qwen2.5-0.5B (hidden=896) |
| Expert hidden | 2048 | 640 |
| Expert Q heads | 16 | 10 |
| **KV heads** | **8 (VLM から継承)** | **2 (VLM から継承)** |
| **head_dim** | **128** | **64** |
| Expert layers | 32 (VLM から継承) | 24 (VLM から継承) |
| Expert パラメータ | ~2B | 145M |

### Expert config の生成方法（alpamayo_r1.py:87-92）

```python
expert_config = copy.deepcopy(self.vlm.config.text_config)  # VLM 設定のコピー
if config.expert_cfg is not None:
    for key, value in config.expert_cfg.items():
        setattr(expert_config, key, value)  # hidden_size, num_heads 等を上書き
self.expert = AutoModel.from_config(expert_config)
```

`expert_cfg` で上書きされるのは `hidden_size`, `num_attention_heads`, `head_dim`, `intermediate_size` のみ。
**`num_key_value_heads` と `num_hidden_layers` は VLM から継承**（KV-cache 互換性のため）。

### KV-cache の扱い

**Alpamayo は KV-cache を一切加工していない。** VLM → Expert にそのまま渡している。

```python
# alpamayo_r1.py:269-278
expert_out_base = self.expert(
    inputs_embeds=future_token_embeds,
    position_ids=position_ids,
    past_key_values=prompt_cache,   # VLM の KV-cache をそのまま
    attention_mask=attention_mask,
    use_cache=True,
    **forward_kwargs,
)
prompt_cache.crop(prefill_seq_len)  # crop パターンも同じ
```

正規化、射影、スケーリングは一切なし。

### KV-cache の中身の違い（最重要）

**Alpamayo** の KV-cache:
- 画像トークン（数百〜数千）
- Egomotion トークン
- **CoC 推論テキスト**（VLM が自己回帰生成したもの）

論文 §5.1 より:
> エキスパートはVLMからの系列 [o_image, o_egomotion, Reason] のKVキャッシュと、
> ノイズ付き制御の埋め込みの両方を入力として受け取る

**MiniPamayo** の KV-cache:
- **16 個の visual token のみ**（DINOv2 → adapter 出力）

Alpamayo の Expert は「前方にカーブがある」「左に車線変更する」等の推論テキストを
KV-cache 経由で受け取るため、軌道予測に必要な情報が豊富。
我々の 16 トークン visual-only KV-cache では情報量が圧倒的に不足している可能性がある。

### 論文のスケール問題への言及

なし。§5.1 には以下の記述のみ：
> 学習時には、エキスパートからの勾配がVLMの重みに逆伝播するのを防ぐため、
> VLMが生成するKVキャッシュにストップグラディエントを適用する。

8B スケールではスケール問題が発生しないか、問題にならないレベルと推測。

---

## 根本原因のまとめ

1. **KV-cache の情報量不足**: 16 visual token では Expert が条件付き分布を学習するのに不十分
2. **Key スケール問題**: Layer 0 K std ≈ 32 → softmax 飽和 → 勾配消失
3. **学習ダイナミクスの失敗**: 上記 1+2 により Expert は KV-cache を無視する局所解に落ちる
4. **結果**: 周辺分布（画像無関係の平均軌道）で CFM loss ≈ 2.0 に収束

---

## 検証済み・除外した原因

| 候補 | 結果 | 詳細 |
|------|------|------|
| 自作 VLM の KV-cache 品質 | ✗ 関係なし | Vanilla Qwen テキスト入力と同じ K std |
| RoPE theta 不一致 | ✗ 関係なし | 1M に統一済み、修正前後で差なし |
| 学習エポック不足 | ✗ 主因ではない | 30 epoch でも 2.0 プラトー |
| 正規化の不均衡 | ✗ 関係なし | accel/kappa は normalize 後 std≈1 |
| Curve oversampling 不足 | ✗ 関係なし | 3× oversampling 実装済み |

---

## 対策案

### 案 A: KV-cache にスケール正規化を追加

VLM の KV-cache を Expert に渡す前に LayerNorm を適用。

```python
# trajectory_decoder.py の forward() 内
for layer_idx in range(len(kv_cache.layers)):
    kv_cache.layers[layer_idx].keys = layer_norm(kv_cache.layers[layer_idx].keys)
    kv_cache.layers[layer_idx].values = layer_norm(kv_cache.layers[layer_idx].values)
```

- メリット: スケール問題を直接解決、実装が簡単
- リスク: LayerNorm のパラメータも学習が必要
- Alpamayo との乖離: Alpamayo は KV-cache 無加工だが、0.5B スケールでは必要な適応

### 案 B: KV-cache に学習可能な Projection を追加

VLM の KV-cache (2 heads × 64 dim) を Expert の KV 空間に射影する小さな Linear 層を追加。

```python
self.kv_key_proj = nn.ModuleList([nn.Linear(64, 64) for _ in range(24)])
self.kv_val_proj = nn.ModuleList([nn.Linear(64, 64) for _ in range(24)])
```

- メリット: 表現空間の不一致を学習で解決
- パラメータ増加: 24 layers × 2 × (64×64 + 64) ≈ 200K（全体 145M の 0.1%）
- Alpamayo との乖離: 小。KV-cache の前処理を追加するだけ

### 案 C: Cross-attention に戻す（旧アーキテクチャ復活）

Stage 2 設計書の数値（val CFM 0.986）は旧 cross-attention アーキテクチャの結果。
KV-cache 方式を諦めて cross-attention に戻せば確実に動く。

- メリット: 実績がある（val CFM 0.986 達成済み）
- デメリット: Alpamayo 論文の設計思想から乖離
- 備考: 0.5B スケールでは KV-cache 方式が機能しないなら、これが現実的な選択

### 案 D: VLM にテキスト推論を生成させてから KV-cache を作る

Alpamayo と同様に、VLM に CoC テキスト推論を生成させ、
画像 + テキストの完全な KV-cache を Expert に渡す。

- メリット: Alpamayo に最も忠実。KV-cache の情報量を大幅に増加
- デメリット: Stage 3 (SFT) が先に必要。VLM がまだ CoC 推論を生成できない
- 備考: Stage 3→2 の順序変更が必要。推論テキストの品質が Expert 性能に直結

### 案 E: VLM の K/V projection を unfreeze して共同学習

VLM の KV projection 層だけ unfreeze して、Expert と一緒に学習。

```python
# train_stage2.py
for layer in vlm.llm.model.layers:
    layer.self_attn.k_proj.requires_grad_(True)
    layer.self_attn.v_proj.requires_grad_(True)
```

- メリット: VLM の KV 表現を Expert に適応させる
- リスク: VLM の他の能力が劣化する可能性
- 備考: Alpamayo は stop gradient で VLM を freeze しているので、論文とは乖離

---

## 推奨アクション

**状況に応じて 2 段階で対応する。**

### Phase 1: 案 A (LayerNorm) を最速で試す

理由:
1. 実装が最も簡単（数行の変更）
2. スケール問題が主因ならこれだけで改善する
3. 10 分で検証可能

### Phase 2: 案 A で改善しない場合

- **案 C (cross-attention 戻し)** が最も確実なフォールバック
  - 0.5B スケールでは KV-cache 方式が根本的に困難
  - 旧アーキテクチャで val CFM 0.986 の実績あり
- **案 D (テキスト推論込み KV-cache)** は将来的に検討
  - Stage 3 (SFT) 完了後に再挑戦する価値あり
