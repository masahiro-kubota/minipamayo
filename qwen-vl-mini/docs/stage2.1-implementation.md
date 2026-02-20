# Stage 2.1: データ・学習改善 — 具体的実装プラン

## 目的

Stage 2 の POPE 評価で判明したハルシネーション問題（Yes Ratio 86-90%, Specificity 23.7%）を改善する。
plan.md の「任意」項目（NEFTune、データ多様化）を適用し、**Bunny-695K (SVIT-mix)** で再学習・再評価する。

### Bunny-695K を使用する根拠

同規模 VLM の調査（[small-vlm-comparison.md](small-vlm-comparison.md) 参照）により、
**665K 級データが小型 VLM の Stage 2 における事実上の標準データ量**であることが判明した。

Bunny-695K は LLaVA-mix 665K をベースに 2 点改良されたデータセット:

1. **LLaVA-Instruct ~158K → SVIT データに置換**: GPT-4 で生成された高品質な会話・複雑推論・参照 QA データ。Visual Genome + COCO ベース
2. **ShareGPT text-only ~41K → WizardLM-evol-instruct ~70K に置換**: より多様で高品質な instruction following データ（+30K 増量）

VQA 系データ（VQAv2, GQA, TextVQA, OCR-VQA, VG）は LLaVA-mix 665K と同一のため、**追加画像のダウンロードは不要**。

| データセット | Bunny-695K での扱い | 比較データ |
|---|---|---|
| LLaVA-mix 665K (標準) | Imp-v1: POPE **86.1%** | LLaVA-1.5: POPE **85.9%** |
| ShareGPT4V-mix 665K | TinyLLaVA-Qwen2-0.5B: POPE **86.6%** | 追加画像 DL 必要（SAM, WebData, DIV2K） |
| **Bunny-695K (SVIT-mix)** | **Bunny-3B: POPE 86.8%** | **追加画像 DL 不要、Apache 2.0** |

出典: [BoyaWu10/Bunny-v1_0-data](https://huggingface.co/datasets/BoyaWu10/Bunny-v1_0-data)（Apache 2.0 ライセンス）

### Bunny-695K データと MiniPamayo の目的の整合性

MiniPamayo は汎用 VLM ではなく、Cosmos Reason Mini（空間推論）への基盤として DINOv2 を採用している（[plan.md §なぜ DINOv2 を選んだのか](plan.md#なぜ-vision-encoder-に-dinov2-を選んだのか) 参照）。

| データ | 量 | LLaVA-mix からの変更 | MiniPamayo に有用か | 理由 |
|--------|-----|---------------------|-------------------|------|
| SVIT (会話・推論) | ~158K | LLaVA-Instruct を置換 | **非常に有用** | GPT-4 生成の高品質会話。VG ベースで空間推論にも寄与 |
| VQAv2 (COCO) | ~184K | 変更なし | **有用** | Yes/No QA を多く含み、POPE のハルシネーション改善に直結 |
| GQA | ~72K | 変更なし | **非常に有用** | Visual Genome の scene graph ベースの QA。物体の位置・関係性の推論 |
| VG (Visual Genome) | ~86K | 変更なし | **非常に有用** | 物体検出・関係推論。空間推論の基盤データ |
| WizardLM text-only | ~70K | ShareGPT 41K から置換・増量 | **有用** | instruction following 改善（+30K 増量） |
| TextVQA | ~22K | 変更なし | △ | DINOv2 は OCR が苦手なため効果限定的 |
| OCR-VQA | ~80K | 変更なし | △ | 同上 |

Bunny-695K を使う目的:
1. **POPE の Yes bias を改善する**（VQAv2 の Yes/No QA + NEFTune）
2. **空間的な理解力を上げる**（GQA + VG + SVIT の空間推論データ）
3. **データ量を標準レベルに揃える**（158K → ~672K）
4. **会話・指示追従の質を上げる**（SVIT + WizardLM は LLaVA-Instruct + ShareGPT より高品質）

---

## Stage 2 → Stage 2.1 変更点サマリ

| 項目 | Stage 2 (baseline) | Stage 2.1 (改善) | 変更理由 |
|------|-------------------|------------------|----------|
| **初期重み** | Stage 1 Adapter のみ | Stage 1 Adapter のみ (同一) | Clean start で Yes bias を排除 |
| **訓練データ** | LLaVA-Instruct-150K (実数 157,712) | Bunny-695K (val2014 除外後 ~672K) | 高品質データ + 全画像ソースを使用 |
| **データ内訳** | COCO 会話 157K のみ | SVIT 158K + COCO VQA 184K + GQA 72K + TextVQA 22K + OCR-VQA 80K + VG 86K + WizardLM 70K | LLaVA-mix 665K ベース、会話+text-only を高品質化 |
| **text-only 混合** | なし | WizardLM-evol-instruct ~70K (約 10%) | instruction following 能力向上（ShareGPT 41K から増量） |
| **NEFTune** | なし | alpha=5 | ハルシネーション抑制 |
| **Epochs** | 2 | 1 | データ量 ~4.3 倍のため（総サンプル数は同程度: Stage 2 315K vs Stage 2.1 ~672K） |
| **Global Batch Size** | 128 | 128 (変更なし) | — |
| **LLM + Adapter LR** | 2e-5 | 2e-5 (変更なし) | — |
| **VE LR** | 1e-5 | 1e-5 (変更なし) | — |
| **Weight Decay** | 0.1 | 0.1 (変更なし) | — |
| **Max Seq Length** | 2,048 | 2,048 (変更なし) | — |
| **出力先** | `checkpoints/stage2/` | `checkpoints/stage2.1/` | 比較のため別ディレクトリ |
| **推定学習時間** | 5.5h | ~13h | データ量増加分 |

### 変更しない項目の根拠

- **LR・Weight Decay・バッチサイズ**: Stage 2 で学習は安定していた（発散なし）。ハイパーパラメータは問題ではない
- **モデルアーキテクチャ**: DINOv2 + Adapter + Qwen2.5-0.5B は変更なし。問題はデータとハルシネーション抑制
- **初期重み**: Stage 2 チェックポイントではなく Stage 1 から再開。Stage 2 の Yes bias が重みに残るリスクを回避

---

## プロジェクト構成（Stage 2.1 で追加・変更するファイル）

```
qwen-vl-mini/src/qwen_vl_mini/
├── model.py                     # NEFTune 対応を追加
├── data/
│   ├── instruct_dataset.py      # text-only サンプル対応、パス解決拡張
│   ├── prepare_mix665k.py       # LLaVA-mix 665K 用前処理（参考用に残す）
│   └── prepare_bunny695k.py     # 新規: Bunny-695K → val2014 除外・パス正規化
├── train_stage2.py              # --neftune_alpha 引数追加、デフォルト変更
├── eval_benchmark.py            # --pope_prompt 引数追加
└── test_pipeline_stage2.py      # text-only テスト追加
```

---

## Step 0: POPE プロンプト変更テスト（ゼロコスト）

既存 Stage 2 チェックポイントで POPE プロンプトを変更し、instruction following の影響を切り分ける。
再学習不要。

### 変更箇所: eval_benchmark.py

`--pope_prompt` CLI 引数を追加し、プロンプトを切り替え可能にする。

```python
# generate_pope_answer に pope_prompt パラメータを追加:
def generate_pope_answer(model, image_path, question, device, pope_prompt="..."):
    prompt_text = question + "\n" + pope_prompt

# evaluate_pope にも pope_prompt を伝播

# main() に --pope_prompt 引数を追加:
parser.add_argument("--pope-prompt", type=str,
    default="Answer the question using a single word or phrase.",
    help="Prompt suffix for POPE questions")
```

### 実行

```bash
# 既存プロンプト（ベースライン確認用）
cd qwen-vl-mini && uv run python -m qwen_vl_mini.eval_benchmark \
    --checkpoint checkpoints/stage2/checkpoint-2464.pt \
    --stage 2 --pope-dir data/pope --image-dir data/coco/val2014

# Yes/No プロンプト（テスト）
cd qwen-vl-mini && uv run python -m qwen_vl_mini.eval_benchmark \
    --checkpoint checkpoints/stage2/checkpoint-2464.pt \
    --stage 2 --pope-dir data/pope --image-dir data/coco/val2014 \
    --pope-prompt "Answer with Yes or No."
```

### 期待する結果

- Yes Ratio が 86-90% → 50-70% に減少すれば、instruction following が問題の主因
- Yes Ratio が変わらなければ、モデルの視覚的判断能力自体が問題

いずれの場合もデータ改善は実施する（プロンプト依存の改善は本質的ではない）。

---

## Step 1: 画像ダウンロードとデータ前処理

### 1a: 非 COCO 画像のダウンロード

Bunny-695K は LLaVA-mix 665K と同じ画像ソースを使用。**全てダウンロード済み。**

| データセット | 画像ソース | ダウンロード先 | 枚数 | 状態 |
|-------------|-----------|--------------|------|------|
| COCO train2014 | MS-COCO | `data/coco/train2014/` | 82,783 | **済** |
| GQA | Visual Genome images | `data/gqa/images/` | 148,854 | **済** |
| TextVQA | TextVQA train images | `data/textvqa/train_images/` | 25,119 | **済** |
| OCR-VQA | 書籍カバー画像 | `data/ocr_vqa/images/` | 79,999 | **済** (1枚欠損) |
| VG | Visual Genome VG_100K, VG_100K_2 | `data/vg/VG_100K/`, `data/vg/VG_100K_2/` | 108,249 | **済** |

### 1b: Bunny-695K JSON のダウンロードと前処理

#### 新規ファイル: data/prepare_bunny695k.py

Bunny-695K JSON から val2014 エントリを除外し、画像パスを正規化するスクリプト。
`prepare_mix665k.py` と同じロジックで、入力が `bunny_695k.json` に変わる。

#### Bunny-695K の構成

| ソース | Bunny-695K | LLaVA-mix 665K との差分 | 採否 |
|--------|-----------|----------------------|------|
| SVIT 会話・推論 (COCO/VG) | ~158K | **LLaVA-Instruct 158K を GPT-4 生成データに置換** | **採用** |
| VQAv2 短答 QA (COCO) | ~184K | 変更なし | **採用** |
| GQA | ~72K | 変更なし | **採用** |
| TextVQA | ~22K | 変更なし | **採用** |
| OCR-VQA | ~80K | 変更なし | **採用** |
| VG (Visual Genome) | ~86K | 変更なし | **採用** |
| WizardLM text-only | ~70K | **ShareGPT 41K から置換・増量** | **採用** |
| COCO val2014 由来 | ~23K | — | **除外** (POPE data leak 防止) |

#### データ分離の確認

```
画像ソース:
  COCO train2014:   82,783 枚 → 学習用
  COCO val2014:     40,504 枚 → 評価用（POPE: 500 枚使用）
  GQA/VG:           ~108K 枚 → 学習用
  TextVQA:          ~28K 枚 → 学習用
  OCR-VQA:          ~80K 枚 → 学習用

Bunny-695K のうち:
  COCO train2014 エントリ: ~341K → 採用
  COCO val2014 エントリ:   ~23K  → 除外（POPE data leak 防止）
  GQA:                     ~72K  → 採用
  TextVQA:                 ~22K  → 採用
  OCR-VQA:                ~80K  → 採用
  VG:                      ~86K  → 採用
  WizardLM text-only:      ~70K  → 採用
  合計採用:               ~672K
```

#### val2014 除外の理由

COCO train2017 = train2014 + val2014 の統合。Bunny-695K の COCO エントリには val2014 由来の画像が含まれる。
POPE 評価は val2014 の 500 枚を使うため、学習データに val2014 画像が入ると data leak になる。

#### 画像パスの設計

出力 JSON の `image` フィールドは `data/` ディレクトリからの相対パスとする。
**注意**: Bunny-695K の画像パス形式は LLaVA-mix 665K と異なる可能性がある。JSON ダウンロード後にパス形式を確認し、`prepare_bunny695k.py` で適切に正規化する。

| 期待される出力パス | 実ファイル |
|---------|-----------|
| `coco/train2014/000000033471.jpg` | `data/coco/train2014/COCO_train2014_000000033471.jpg` |
| `gqa/images/2375429.jpg` | `data/gqa/images/2375429.jpg` |
| `textvqa/train_images/XXX.jpg` | `data/textvqa/train_images/XXX.jpg` |
| `ocr_vqa/images/XXX.jpg` | `data/ocr_vqa/images/XXX.jpg` |
| `vg/VG_100K/XXX.jpg` | `data/vg/VG_100K/XXX.jpg` |
| (text-only) | なし |

#### 実行

```bash
# Bunny-695K JSON のダウンロード (831 MB)
cd qwen-vl-mini && huggingface-cli download BoyaWu10/Bunny-v1_0-data \
    finetune/bunny_695k.json --local-dir data/bunny

# 前処理（val2014 除外・パス正規化）
cd qwen-vl-mini && uv run python -m qwen_vl_mini.data.prepare_bunny695k \
    --input data/bunny/finetune/bunny_695k.json \
    --output data/llava-instruct/bunny695k_full.json \
    --val2014_dir data/coco/val2014
```

---

## Step 2: InstructDataset 拡張

### 変更ファイル: data/instruct_dataset.py

#### 2a: image_root パラメータの追加

665K は複数の画像ソース（COCO, GQA, TextVQA, OCR-VQA, VG）を含むため、`image_dir`（単一ディレクトリ）ではなく `image_root`（`data/` ディレクトリ）を基準としてパスを解決する。

```python
class InstructDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_dir: str,       # 後方互換（COCO 専用パス）
        image_root: str = "", # 665K 用: data/ ディレクトリのルート
        ...
    ):
        self.image_dir = Path(image_dir)
        self.image_root = Path(image_root) if image_root else None
```

#### 2b: text-only サンプル対応

`image` キーがないエントリ（text-only）に対応。黒画像（ゼロテンソル）をダミーとして渡す。

```python
def __getitem__(self, idx: int) -> dict:
    sample = self.data[idx]

    # --- Image ---
    image_name = sample.get("image", "")
    if image_name:
        image_path = self._resolve_image_path(image_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, Exception):
            return self[random.randint(0, len(self) - 1)]
        pixel_values = self.transform(image)
    else:
        # Text-only: ダミー黒画像
        pixel_values = torch.zeros(3, 224, 224)
    ...
```

**黒画像ダミーの妥当性**: visual tokens の labels は常に -100 でマスクされるため、黒画像の visual features が loss に直接影響することはない。forward pass で黒画像の特徴がテキスト生成のコンテキストに入る点は懸念だが、LLaVA-1.5 自体が 665K の text-only データを同じ方法（ダミー画像）で処理しており、先行手法として実績がある。また text-only は全体の約 6% であり、影響は限定的。

#### 2c: _resolve_image_path の拡張

multi-source 対応。`image_root` が設定されている場合はそちらを優先的に使用する。

```python
def _resolve_image_path(self, image_name: str) -> Path:
    # 1. image_root が設定されていれば、相対パスとして解決
    if self.image_root:
        # COCO: coco/train2014/000000033471.jpg → COCO_train2014_000000033471.jpg
        if image_name.startswith("coco/"):
            bare_id = image_name.split("/")[-1]  # 000000033471.jpg
            coco_path = self.image_root / "coco" / "train2014" / f"COCO_train2014_{bare_id}"
            if coco_path.exists():
                return coco_path
        # 非 COCO: そのまま解決 (gqa/images/XXX.jpg 等)
        path = self.image_root / image_name
        if path.exists():
            return path

    # 2. Fallback: image_dir を使用（後方互換）
    if "/" in image_name:
        image_name = image_name.split("/")[-1]

    path = self.image_dir / image_name
    if path.exists():
        return path
    coco_name = f"COCO_train2014_{image_name}"
    coco_path = self.image_dir / coco_name
    if coco_path.exists():
        return coco_path
    return path
```

---

## Step 3: NEFTune 実装

### 変更ファイル: model.py

NEFTune (Noisy Embedding Fine-Tuning) は入力埋め込みに uniform noise を注入してハルシネーションを抑制する手法。
Idefics2 §4.2 で採用。

```python
class QwenVLMini(nn.Module):
    IGNORE_INDEX = -100

    def __init__(
        self,
        vision_model_name: str = "facebook/dinov2-base",
        llm_model_name: str = "Qwen/Qwen2.5-0.5B",
        neftune_alpha: float = 0.0,
    ):
        super().__init__()
        self.neftune_alpha = neftune_alpha
        ...

    def _build_inputs(self, pixel_values, input_ids, attention_mask, labels=None):
        ...
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # NEFTune: add uniform noise during training
        if self.training and self.neftune_alpha > 0:
            dims = torch.tensor(
                inputs_embeds.shape[1] * inputs_embeds.shape[2],
                dtype=inputs_embeds.dtype,
            )
            mag = self.neftune_alpha / torch.sqrt(dims)
            inputs_embeds = inputs_embeds + torch.zeros_like(inputs_embeds).uniform_(-mag, mag)
        ...
```

### NEFTune の動作

- `alpha=0`: 無効（Stage 2 と同じ動作）
- `alpha=5`: Idefics2 で使用された値。推奨開始値
- training 時のみノイズ注入（eval/generate 時は無効）

---

## Step 4: train_stage2.py 改修

### 変更ファイル: train_stage2.py

#### 変更内容

1. `--neftune_alpha` 引数を DEFAULTS に追加
2. `--image_root` 引数を追加（multi-source 画像パス解決用）
3. モデル初期化時に `neftune_alpha` を渡す
4. InstructDataset に `image_root` を渡す
5. デフォルト値の変更:
   - `json_path`: `bunny695k_full.json`
   - `epochs`: 1（データ量増加に伴い）
   - `output_dir`: `checkpoints/stage2.1`
6. wandb run name を `stage2.1-bunny695k-neftune` に変更

```python
DEFAULTS = {
    "json_path": "data/llava-instruct/bunny695k_full.json",
    "output_dir": "checkpoints/stage2.1",
    "epochs": 1,
    "neftune_alpha": 5.0,
    # ... 他は変更なし
}
```

---

## Step 5: パイプラインテスト

### 変更ファイル: test_pipeline_stage2.py

新データファイルと text-only サンプルの動作確認を追加。

---

## Step 6: 学習実行

```bash
cd qwen-vl-mini && PYTHONUNBUFFERED=1 uv run python -m qwen_vl_mini.train_stage2 \
    --json_path data/llava-instruct/bunny695k_full.json \
    --image_dir data/coco/train2014 \
    --image_root data \
    --output_dir checkpoints/stage2.1 \
    --stage1_checkpoint checkpoints/stage1/checkpoint-2325.pt \
    --neftune_alpha 5 \
    --epochs 1 \
    --save_steps 100 \
    --logging_steps 5
```

### 推定

- データ: ~672K × 1 epoch → ~5,250 optimizer steps（grad_accum=128）
- 学習時間: ~13 時間（Stage 2 の 5.5h × 672K/315K ≈ 11.7h + マージン）
- VRAM: ~8 GB（変更なし）

---

## Step 7: 評価

### 7a: POPE 評価

```bash
cd qwen-vl-mini && uv run python -m qwen_vl_mini.eval_benchmark \
    --checkpoint checkpoints/stage2.1/checkpoint-XXXX.pt \
    --stage 2 --pope-dir data/pope --image-dir data/coco/val2014
```

### 7b: 定性的評価

```bash
cd qwen-vl-mini && uv run python -m qwen_vl_mini.eval_qualitative \
    --checkpoint checkpoints/stage2.1/checkpoint-XXXX.pt \
    --stage 2 --image-dir data/coco/val2014 --seed 42
```

### データ分離の確認

Stage 2.1 の学習データと POPE 評価データは完全に分離されている:

- **学習データ**: COCO train2014 (82,783 枚) + GQA/VG/TextVQA/OCR-VQA 画像 + text-only (画像なし) — val2014 由来エントリは明示的に除外済み
- **評価データ**: COCO val2014 (40,504 枚) — POPE 評価はこのうち 500 枚を使用
- **非 COCO 画像**: GQA/VG/TextVQA/OCR-VQA の画像は COCO val2014 とは完全に別のデータセットであり、重複なし

Bunny-695K の COCO エントリには val2014 由来の画像が含まれるが、
`prepare_bunny695k.py` で val2014 画像 ID を照合し除外するため、data leak は発生しない。

### 7c: 比較指標

| 指標 | Stage 2 (baseline) | Stage 2.1 (目標) |
|------|-------------------|-----------------|
| POPE Random Accuracy | 59.8% | >60% |
| POPE Random Specificity | 23.7% | >40% |
| Yes Ratio | 86.1% | <70% |
| 定性的: 会話能力 | 良好 | 劣化なし |

---

## Step 8: レポート更新

- `stage2-report.md` に「Stage 2.1 改善結果」セクションを追記
- `plan.md` §4.4-4.5 のチェックボックスを更新
- Stage 2 baseline との比較表を記載

---

## Stage 2.1 が目標未達の場合の選択肢

Stage 2.1 で POPE 目標（Accuracy >60%, Specificity >40%）を達成できなかった場合、以下の選択肢がある。
コストの低い順に並べており、上から順に検討する。

### A. ハイパーパラメータ調整（追加学習コスト: ~12h/回）

| 調整項目 | 試行案 | 根拠 |
|----------|--------|------|
| NEFTune alpha | 10, 15 | alpha=5 で不十分ならより強いノイズ |
| Epochs | 2 | 1 epoch では学習不足の可能性 |
| VE frozen | `--ve_frozen` | VE の fine-tune がハルシネーションを悪化させている可能性 |
| LR 低減 | LLM 1e-5, VE 5e-6 | 過学習抑制 |

### B. plan.md の未実施「任意」項目

plan.md §3.1 に記載されている残りの任意項目：

| 項目 | 概要 | 出典 |
|------|------|------|
| ShareGPT4V キャプション置換 | SFT データの 3-5% を高品質キャプションに置換 | ShareGPT4V Figure 2 |
| GPT4V-annotated データ追加 | ShareGPT-4V 20K + LAION-GPT-V 10K + ALLaVA 300K | Imp Table 1 §3.2 |
| OCR/Chart データ追加 | DVQA + ChartQA + DocVQA + AI2D + InfographicVQA = 32K | Imp §3.1, Table 2 |

これらは追加データのダウンロードと前処理が必要。

### C. 現状で Cosmos Reason Mini に進む

定量的目標が未達でも、定性的に VLM 能力（会話、物体記述）が機能していれば、Cosmos Reason Mini の開発に進む選択肢もある。Cosmos 側での追加学習で補える可能性があり、VLM 単体の POPE スコアに固執する必要はないかもしれない。

### 判断基準

| Stage 2.1 結果 | 次のアクション |
|----------------|---------------|
| 目標達成（POPE >60%, Specificity >40%） | Stage 2.1 完了 → Cosmos に進む |
| 改善あり（Yes Ratio 減少）だが目標未達 | A のハイパーパラメータ調整を 1-2 回試す |
| 改善なし（Yes Ratio 変化なし） | B の追加データを検討 |
| 悪化（定性的にも劣化） | Stage 2 チェックポイントに戻り、C で Cosmos に進む |

---

## 完了状況

| Step | 状態 | 備考 |
|------|------|------|
| Step 0: POPE プロンプトテスト | **完了** | Yes/No プロンプトで Yes Ratio 改善を確認 |
| Step 1: データ前処理 | **完了** | 画像 DL 完了、Bunny-695K JSON DL・前処理完了 (671,646 entries) |
| Step 2: InstructDataset 拡張 | **完了** | text-only + image_root + multi-source 対応済み |
| Step 3: NEFTune 実装 | **完了** | model.py に alpha パラメータ追加済み |
| Step 4: train_stage2.py 改修 | **完了** | json_path=bunny695k_full.json, wandb=stage2.1-bunny695k-neftune |
| Step 5: パイプラインテスト | **完了** | 全テスト通過 |
| Step 6: 学習実行 | **完了** | 5,247 steps, ~13h, Peak VRAM 10,066 MB, NaN skip 12回 |
| Step 7: 評価 | **完了** | POPE Random 85.9% (+26.1pt), Popular 84.6%, Adversarial 80.2% |
| Step 8: レポート更新 | **完了** | stage2-report.md, plan.md 更新済み |
