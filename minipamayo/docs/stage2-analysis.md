# Stage 2 Flow Matching Expert: 現状分析と方針

## 1. そもそも何をやっているのか

Alpamayo の学習パイプラインは 4 段階：

```
Stage 1: VLM が画像→離散トークンを学習
Stage 2: Expert が「画像+推論テキスト → 連続軌道」を学習 ← ★ここで詰まっている
Stage 3: VLM が推論テキストを生成できるように SFT
Stage 4: RL で全体を最適化
```

**Stage 2 の目標**: Flow Matching Expert（Transformer）が、VLM の出力を条件として、車の連続軌道（加速度 a + 曲率 kappa × 6 ステップ）を生成できるようにする。

---

## 2. Flow Matching とは

通常の回帰（Huber loss で直接予測）と違い、Flow Matching はノイズから段階的に軌道を生成する手法：

```
ノイズ ε (ランダム) → [Expert が 10 ステップで少しずつ修正] → 軌道 a₁ (予測)
```

**学習時**: ランダムな時刻 t で「ノイズと正解の中間状態 aₜ」を作り、Expert に「正解方向への速度場 v」を予測させる。

**メリット**: 多峰分布（「右にも左にも曲がれる」）を表現できる。直接回帰だと平均に潰れる。

**CFM Loss**: Expert の予測速度と正解速度の MSE。値が小さいほど良い。

---

## 3. 2 つのアーキテクチャの違い

### A. 旧版: Cross-Attention Decoder（コミット 54346b5 以前）

```
VLM
├── 画像 → Vision Encoder → Adapter → hidden states (16 tokens, 896 dim)
└── hidden states を projection → (16 tokens, 256 dim)
                                       ↓
Expert (小さい Transformer, ~3M params)
├── Action + Timestep → 入力トークン
├── Self-Attention（トークン間）
├── Cross-Attention（→ VLM hidden states を直接参照）★
├── AdaLN（timestep で変調）
└── → 速度場 v を出力
```

**ポイント**: Expert が VLM の hidden states を**明示的に cross-attention で参照**。シンプルで直接的。

### B. 現行版: KV-cache Expert（Alpamayo 式、コミット a2353f0 以降）

```
VLM (Qwen2.5-0.5B, 凍結)
├── [system][user][画像][質問][assistant][推論テキスト] → VLM forward
└── → KV-cache (24 layers × 2 KV heads × 171 tokens × 64 dim)
                                       ↓
Expert (Qwen2 Transformer, 145M params)
├── Action + Timestep → Fourier encoding → 入力トークン (6 個)
├── past_key_values = VLM の KV-cache を受け取る
├── GQA: 10 Q heads / 2 KV heads（VLM と KV head 数を合わせる制約）
└── → 速度場 v を出力
```

**ポイント**: Alpamayo 論文 §5.1 に忠実。Expert が VLM の KV-cache を `past_key_values` として受け取り、**暗黙的に attention で参照**。

---

## 4. なぜ KV-cache Expert がうまくいかないのか

### 4.1 スケールの差

|  | Alpamayo | MiniPamayo |
|--|----------|------------|
| VLM | Cosmos-Reason 7B | Qwen2.5-0.5B |
| Expert | ~4B | 145M |
| KV heads | 多い（推定 8+） | **2** |
| KV-cache 情報量 | 豊か | **ボトルネック** |

**KV head が 2 しかない**ことが致命的。171 トークン分の情報（画像 16 + 推論テキスト ~155）が、たった 2 つの KV head (各 64 dim) に圧縮される。Expert の 10 Q heads がこの 2 KV heads から情報を取り出す必要があるが、情報が失われている。

### 4.2 旧版が機能していた理由

旧版の cross-attention は `nn.MultiheadAttention` を使い、hidden states を**直接 896 dim で参照**できた。KV head 数の制約がなく、情報のボトルネックがなかった。

### 4.3 学習結果の比較

**KV-cache Expert v1（LR=1e-4, dropout なし）:**

| Epoch | Train CFM | Val CFM | 備考 |
|-------|-----------|---------|------|
| Init  | -         | 2.618   | |
| 1     | 2.458     | 2.260   | |
| 2     | 2.251     | **2.109** | ← best |
| 3     | 2.171     | 2.142   | ↑ overfitting 開始 |
| 4     | 2.073     | 2.167   | |
| 5     | 1.986     | 2.139   | |
| 6     | 1.920     | 2.274   | ← E1 レベルまで悪化 |

**KV-cache Expert v2（LR=3e-5, dropout=0.1）:**

| Epoch | Train CFM | Val CFM |
|-------|-----------|---------|
| 1     | 2.494     | 2.258   |
| 2     | 2.363     | 2.136   |
| 3     | (OOM kill で中断、train ~2.16) | |

→ **v1 も v2 も val 2.1 付近で頭打ち**。dropout + 低 LR でも改善せず。

---

## 5. 選択肢

### A. Cross-Attention Decoder に戻す + GT 推論 teacher-forcing

旧版の cross-attention decoder を復活させつつ、今回の改善（GT 推論テキストを VLM に通して hidden states を作る）を組み合わせる。

```
VLM (凍結)
├── [system][user][画像][質問][assistant][推論テキスト] → VLM forward
└── → hidden states (171 tokens, 896 dim) ★ full hidden states を使う
                                       ↓
                              projection (896 → 256)
                                       ↓
Expert (Cross-Attention Transformer, ~3-10M params)
├── Cross-Attention で hidden states を直接参照
└── → 速度場 v を出力
```

**メリット**:
- KV head ボトルネックがない（hidden states を直接参照）
- 小さいモデルで十分（3-10M params）
- 旧版で動作実績がある
- GT 推論テキストの情報も含まれる

**デメリット**:
- Alpamayo の論文とは異なるアーキテクチャ

### B. 現在の best checkpoint で Stage 3 に進む

KV-cache Expert の best (val 2.109) をそのまま使い、Stage 3 (SFT) → Stage 4 (RL) に進む。

**メリット**: 時間をかけずに先に進める
**デメリット**: Expert の性能が限定的なまま

### C. データを増やして KV-cache Expert を再学習

nuScenes の他のデータや augmentation で学習データを増やす。

**メリット**: Alpamayo に忠実なアーキテクチャを維持
**デメリット**: VLM が 0.5B / KV head が 2 という根本問題は解決しない。データだけ増やしても KV-cache のボトルネックは変わらない可能性が高い。

---

## 6. 推奨

**A（Cross-Attention + GT 推論）が最も現実的**。理由：

1. 旧版で曲がれていた実績がある
2. GT 推論テキストの hidden states を条件に使えるので、「推論→軌道」の対応が学べる
3. 小さいモデルで overfitting しにくい
4. 実装変更は比較的少ない（旧版のコードが git に残っている）
5. Stage 3/4 との接続も hidden states ベースなので自然
