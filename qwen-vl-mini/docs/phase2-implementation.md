# Phase 2: Stage 1 Feature Alignment — 具体的実装プラン

## 目的

Adapter が DINOv2 の視覚特徴を Qwen2.5-0.5B の入力空間にマッピングすることを学ぶ。
VE + LLM は frozen、**Adapter のみ学習**。

---

## プロジェクト構成（Phase 2 で追加するファイル）

```
qwen-vl-mini/src/qwen_vl_mini/
├── model.py               # Phase 1 で作成済み
├── data/
│   ├── __init__.py
│   └── pretrain_dataset.py # Step 2: Stage 1 用 Dataset
├── train_stage1.py         # Step 3: 学習スクリプト
└── eval_qualitative.py     # Step 4: 定性的評価
```

---

## Step 1: データダウンロード

### LLaVA-CC3M-Pretrain-595K

```bash
# HuggingFace からダウンロード（~7 GB: images.zip 6.46GB + chat.json 211MB）
huggingface-cli download liuhaotian/LLaVA-CC3M-Pretrain-595K --local-dir data/llava-pretrain
```

**データ構造**:
```
data/llava-pretrain/
├── chat.json           # 595K サンプルの会話データ
├── metadata.json       # メタデータ
└── images/             # images.zip を展開
    ├── GCC_train_000000000.jpg
    ├── GCC_train_000000001.jpg
    └── ...
```

**chat.json のフォーマット**:
```json
{
  "id": "GCC_train_002582585",
  "image": "GCC_train_002582585.jpg",
  "conversations": [
    {"from": "human", "value": "Provide a brief description of the given image.\n<image>"},
    {"from": "gpt", "value": "olive oil is a healthy ingredient used liberally."}
  ]
}
```

---

## Step 2: Dataset 実装

### PretrainDataset

```python
class PretrainDataset(Dataset):
    """LLaVA-CC3M-Pretrain-595K for Stage 1 Feature Alignment."""

    def __init__(self, json_path, image_dir, tokenizer, transform):
        # chat.json をロード
        # 各サンプル: {"id", "image", "conversations"}

    def __getitem__(self, idx):
        # 1. 画像ロード + transform (224x224, ImageNet正規化)
        # 2. human の value から質問テキスト抽出（"<image>" は除去）
        # 3. gpt の value からキャプション抽出
        # 4. Qwen2.5 チャットテンプレートでフォーマット
        # 5. tokenize して input_ids, attention_mask, labels を返す
        #
        # returns: {
        #   "pixel_values": (3, 224, 224),
        #   "input_ids": (T,),
        #   "attention_mask": (T,),
        #   "labels": (T,),  ← キャプション部分のみ、残りは -100
        # }
```

### テキストフォーマットと Loss マスクの詳細

Stage 1 では画像キャプションの生成を学習する。入力テンプレート:

```
<|im_start|>system
You are a visual assistant.<|im_end|>
<|im_start|>user
{質問プロンプト（11種からランダム選択）}<|im_end|>
<|im_start|>assistant
{キャプション}<|im_end|>
```

**Loss マスク**:
- system + user 部分 → `-100`（loss 計算しない）
- **assistant の回答部分のみ → 実際の token ID**（loss 計算対象）
- `<|im_start|>assistant\n` の直後からキャプション末尾の `<|im_end|>` まで

**実装方法**:
```python
# 方法: 全体を tokenize → assistant 回答部分の位置を特定 → labels を構築

# 1. prompt 部分（system + user + "<|im_start|>assistant\n"）を tokenize
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

# 2. 全体（prompt + caption + <|im_end|>）を tokenize
full_text = prompt_text + caption + "<|im_end|>"
full_ids = tokenizer.encode(full_text, add_special_tokens=False)

# 3. labels 構築
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```

### 11 種類の質問プロンプト（LLaVA Table 11）

```python
PRETRAIN_PROMPTS = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
]
```

各サンプルで **ランダムに 1 つ選択**して使用。過学習を防ぐ。

### Collator

```python
class PretrainCollator:
    """Pad sequences to same length within a batch."""
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        # pixel_values: stack → (B, 3, 224, 224)
        # input_ids: pad → (B, T)
        # attention_mask: pad → (B, T)
        # labels: pad with -100 → (B, T)
        # max_length で truncate（Stage 1 は短いキャプションなので 512 で十分）
```

---

## Step 3: 学習スクリプト

### train_stage1.py

```python
def main():
    # --- Config ---
    config = {
        "lr": 1e-3,
        "batch_size": 4,              # micro-batch (VRAM に合わせて調整)
        "grad_accum": 64,             # → global batch = 256
        "epochs": 1,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,          # Stage 1 は Adapter のみなので 0 で OK
        "max_grad_norm": 1.0,
        "save_steps": 500,
        "logging_steps": 10,
        "output_dir": "checkpoints/stage1",
    }

    # --- Model ---
    model = QwenVLMini()
    model.set_stage1()  # VE + LLM frozen, Adapter only
    model.to(device)

    # --- Data ---
    dataset = PretrainDataset(json_path, image_dir, model.tokenizer, IMAGE_TRANSFORM)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"],
                            shuffle=True, num_workers=4,
                            collate_fn=PretrainCollator(model.tokenizer))

    # --- Optimizer ---
    # Adapter のみ: parameters with requires_grad
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        betas=(0.9, 0.95),
        weight_decay=config["weight_decay"],
    )

    # --- Scheduler ---
    num_training_steps = len(dataloader) // config["grad_accum"] * config["epochs"]
    num_warmup_steps = int(num_training_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # --- Training Loop ---
    model.train()
    for epoch in range(config["epochs"]):
        for step, batch in enumerate(dataloader):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    pixel_values=batch["pixel_values"].to(device),
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = output.loss / config["grad_accum"]

            loss.backward()

            if (step + 1) % config["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # logging
                wandb.log({"loss": loss.item() * config["grad_accum"], "lr": scheduler.get_last_lr()[0]})

            # checkpoint
            if (step + 1) % config["save_steps"] == 0:
                save_checkpoint(model, optimizer, step, config["output_dir"])
```

### チェックポイント保存

Stage 1 では **Adapter の重みのみ保存**すれば十分（VE + LLM は変更されていない）:

```python
def save_checkpoint(model, optimizer, step, output_dir):
    torch.save({
        "adapter_state_dict": model.adapter.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }, f"{output_dir}/checkpoint-{step}.pt")
```

### 学習時間の見積もり

- データ: 595K サンプル
- Global batch size: 256
- Steps/epoch: ~2,324
- RTX 4090 1 枚: Adapter のみなので forward は高速
- **見積もり: 3〜6 時間**（micro-batch=4 の場合。I/O がボトルネックになる可能性あり）

---

## Step 4: 評価

### 定性的評価 (eval_qualitative.py)

Stage 1 前後で同じ画像に対する出力を比較する。

```python
def evaluate_qualitative(model, image_paths):
    """複数の画像に対してキャプション生成し、表示する。"""
    model.eval()
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        pixel_values = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)
        prompt = model.prepare_prompt("Describe this image.")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                pixel_values,
                prompt["input_ids"].to(device),
                prompt["attention_mask"].to(device),
                max_new_tokens=100,
                do_sample=False,  # greedy decoding
            )

        text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Image: {path}")
        print(f"Output: {text}\n")
```

**評価用画像**: COCO val2014 から 5〜10 枚を手動選択（犬、風景、料理、人物、テキスト付き等）

### Exit 条件チェック

| 条件 | 確認方法 |
|---|---|
| Loss が安定して下がる | wandb の loss curve を確認 |
| 画像に関連したテキストが生成される | eval_qualitative.py で目視確認 |
| 画像の内容に応じて出力が変化する | 異なる画像で異なるキャプションが出るか |

---

## テスト計画

### T1: データパイプライン単体テスト（実装直後に実行）

Dataset と Collator が正しく動作するかを、学習開始前に検証する。
ファイル: `tests/test_pretrain_dataset.py`

```python
def test_dataset_returns_correct_shapes():
    """各サンプルの出力形状が正しいこと。"""
    sample = dataset[0]
    assert sample["pixel_values"].shape == (3, 224, 224)
    assert sample["input_ids"].ndim == 1       # (T,)
    assert sample["attention_mask"].ndim == 1  # (T,)
    assert sample["labels"].ndim == 1          # (T,)
    assert len(sample["input_ids"]) == len(sample["labels"])

def test_image_normalization():
    """ImageNet 正規化が適用されていること。"""
    pv = sample["pixel_values"]
    # 正規化後は [-3, 3] 程度の範囲（[0,1] ではない）
    assert pv.min() < 0, "Should be normalized (not [0,1])"
    assert pv.max() < 4, "Should be normalized"

def test_loss_mask_correct():
    """labels が assistant の回答部分のみ有効であること。"""
    sample = dataset[0]
    labels = sample["labels"]
    input_ids = sample["input_ids"]

    # labels のうち -100 でない部分（= loss 計算対象）を取得
    valid_mask = labels != -100
    valid_count = valid_mask.sum().item()

    # 有効な labels は 1 トークン以上あること（キャプション）
    assert valid_count > 0, "Should have at least 1 valid label token"

    # 有効な labels はシーケンスの後半にあること
    #（前半は system + user + <|im_start|>assistant）
    first_valid = valid_mask.nonzero()[0].item()
    assert first_valid > 5, "Valid labels should not start at the beginning"

    # 有効な labels の最後に <|im_end|> が含まれること（EOS 学習）
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    last_valid_idx = valid_mask.nonzero()[-1].item()
    assert input_ids[last_valid_idx] == im_end_id, \
        "Last valid label should be <|im_end|>"

def test_prompt_variants_used():
    """11 種の質問プロンプトが使われていること。"""
    prompts_seen = set()
    for i in range(100):
        sample = dataset[i]
        text = tokenizer.decode(sample["input_ids"])
        for p in PRETRAIN_PROMPTS:
            if p in text:
                prompts_seen.add(p)
    assert len(prompts_seen) >= 3, \
        f"Should use multiple prompt variants, saw {len(prompts_seen)}"

def test_collator_batch():
    """Collator がバッチを正しくパディングすること。"""
    samples = [dataset[i] for i in range(4)]
    batch = collator(samples)
    B = 4
    assert batch["pixel_values"].shape == (B, 3, 224, 224)
    assert batch["input_ids"].shape[0] == B
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == batch["input_ids"].shape
    # パディング部分の labels は -100
    pad_mask = batch["attention_mask"] == 0
    assert (batch["labels"][pad_mask] == -100).all()

def test_model_forward_with_real_data():
    """実データでモデル forward が通ること。"""
    batch = collator([dataset[0]])
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            pixel_values=batch["pixel_values"].to(device),
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
    assert not torch.isnan(output.loss), "Loss should not be NaN"
    assert output.loss.item() > 0, "Loss should be positive"
```

### T2: 学習中の自動チェック（学習スクリプト内に組み込み）

学習ループ内で異常を自動検知する。

```python
# train_stage1.py の学習ループ内に組み込む

# === Check 1: Loss NaN 検知 ===
if torch.isnan(output.loss):
    raise RuntimeError(f"Loss is NaN at step {step}. Aborting.")

# === Check 2: 初期 loss の記録 + 学習進行の確認 ===
if global_step == 0:
    initial_loss = loss_value
if global_step == 100:
    assert loss_value < initial_loss, \
        f"Loss not decreasing: initial={initial_loss:.4f}, step100={loss_value:.4f}"
    print(f"✓ Learning check passed: {initial_loss:.4f} → {loss_value:.4f}")

# === Check 3: Adapter の重みが更新されていること ===
if global_step == 1:
    adapter_hash_before = hash(tuple(
        p.data.flatten()[:10].tolist() for p in model.adapter.parameters()
    ))
if global_step == 2:
    adapter_hash_after = hash(tuple(
        p.data.flatten()[:10].tolist() for p in model.adapter.parameters()
    ))
    assert adapter_hash_before != adapter_hash_after, \
        "Adapter weights should have changed"
```

### T3: 学習後の評価テスト（学習完了後に実行）

ファイル: `tests/test_stage1_trained.py`

```python
def test_different_images_different_outputs():
    """異なる画像で異なるキャプションが生成されること（縮退していない）。"""
    outputs = []
    for img_path in eval_images[:5]:
        text = generate_caption(model, img_path)
        outputs.append(text)
    # 全部同じテキストではないこと
    unique = set(outputs)
    assert len(unique) >= 2, \
        f"Model outputs are degenerate: all outputs are '{outputs[0]}'"

def test_loss_significantly_lower():
    """学習後の loss が初期値より十分低いこと。"""
    avg_loss = compute_avg_loss(model, eval_dataloader, num_batches=50)
    # 未学習時の loss は ~7-10（vocab size の log）
    # 学習後は ~3-5 を期待
    assert avg_loss < 6.0, \
        f"Loss too high after training: {avg_loss:.4f}"

def test_output_contains_image_words():
    """画像に関連する単語が生成されること（ランダムではない）。"""
    # 犬の画像
    text = generate_caption(model, "test_images/dog.jpg")
    # 最低限なにかの名詞が含まれている（完全ランダムでない証拠）
    common_nouns = ["dog", "cat", "man", "woman", "car", "tree", "building",
                    "water", "sky", "food", "table", "people", "child"]
    has_noun = any(noun in text.lower() for noun in common_nouns)
    # Stage 1 なので完璧を求めない。ただし完全なゴミではないことを確認
    print(f"Generated: {text}")
    print(f"Contains common noun: {has_noun}")
    # assert ではなく print で確認（Stage 1 は品質が低い可能性があるため）
```

### テスト実行タイミング

| テスト | タイミング | 実行方法 |
|---|---|---|
| **T1** (データパイプライン) | Dataset/Collator 実装直後 | `pytest tests/test_pretrain_dataset.py` |
| **T2** (学習中チェック) | 学習中（自動） | train_stage1.py 内に組み込み |
| **T3** (学習後評価) | 学習完了後 | `python tests/test_stage1_trained.py` |

---

## Step 5: 壊れた画像の対処

CC3M は元 URL からのクロールで収集されたため、一部の画像が壊れている可能性がある。

```python
# Dataset.__getitem__ で画像ロード失敗時はスキップ
try:
    image = Image.open(image_path).convert("RGB")
except (OSError, PIL.UnidentifiedImageError):
    # ランダムに別のサンプルを返す
    return self.__getitem__(random.randint(0, len(self) - 1))
```

---

## 実装順序

```
Step 1: データダウンロード (llava-pretrain)          (30 min〜数時間、回線速度による)
Step 2: PretrainDataset + Collator                   (1-2 hours)
Step 3: train_stage1.py (学習ループ)                  (1-2 hours)
Step 4: eval_qualitative.py (定性的評価)              (30 min)
学習実行                                              (3-6 hours)
```

---

## 次のステップ（Phase 2 完了後）

Phase 2 の Exit 条件をクリアしたら Phase 3（Stage 2: Visual Instruction Tuning）に進む。
Adapter の学習済み重みを Phase 3 の初期値として引き継ぐ。
