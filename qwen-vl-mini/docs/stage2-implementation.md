# Stage 2: Visual Instruction Tuning — 具体的実装プラン

## 目的

画像について質問に答えたり詳細に説明したりする VLM 能力を獲得する。
**DINOv2 + Adapter + LLM を全解凍**して end-to-end で学習。

---

## プロジェクト構成（Stage 2 で追加するファイル）

```
qwen-vl-mini/src/qwen_vl_mini/
├── model.py                 # Stage 0 で作成済み
├── data/
│   ├── __init__.py
│   ├── pretrain_dataset.py  # Stage 1 で作成済み
│   └── instruct_dataset.py  # Step 2: Stage 2 用 Dataset
├── train_stage1.py          # Stage 1 で作成済み
├── train_stage2.py          # Step 3: Stage 2 学習スクリプト
├── eval_qualitative.py      # Stage 1 で作成済み（拡張）
└── eval_benchmark.py        # Step 5: 定量的評価
```

---

## Step 1: データダウンロード

### メインデータ: LLaVA-Instruct-150K

```bash
# 会話データ (~229 MB)
huggingface-cli download liuhaotian/LLaVA-Instruct-150K --local-dir data/llava-instruct

# 画像: COCO 2014 train (~13 GB)
# LLaVA-Instruct-150K の画像は全て COCO 2014 train から
wget http://images.cocodataset.org/zips/train2014.zip -P data/
unzip data/train2014.zip -d data/coco/
```

**データ構造**:
```
data/
├── llava-instruct/
│   └── llava_instruct_150k.json   # 150K サンプルの会話データ
└── coco/
    └── train2014/
        ├── COCO_train2014_000000000009.jpg
        └── ...
```

**llava_instruct_150k.json のフォーマット**:
```json
{
  "id": "000000033471",
  "image": "COCO_train2014_000000033471.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat are the colors of the bus in the image?"},
    {"from": "gpt", "value": "The bus in the image is white and red."},
    {"from": "human", "value": "What feature can be seen on the front of the bus?"},
    {"from": "gpt", "value": "The front of the bus features a large window..."}
  ]
}
```

- **マルチターン会話**: 1 つの画像に対して複数の QA ペア
- `<image>` トークンは最初の human ターンに含まれる

---

## Step 2: Dataset 実装

### InstructDataset

```python
class InstructDataset(Dataset):
    """LLaVA-Instruct-150K for Stage 2 Visual Instruction Tuning."""

    def __init__(self, json_path, image_dir, tokenizer, transform, max_length=2048):
        # llava_instruct_150k.json をロード

    def __getitem__(self, idx):
        # 1. 画像ロード + transform
        # 2. マルチターン会話を Qwen2.5 テンプレートに変換
        # 3. tokenize + labels 構築（assistant の回答部分のみ loss 計算）
        # 4. max_length で truncate
        #
        # returns: {
        #   "pixel_values": (3, 224, 224),
        #   "input_ids": (T,),
        #   "attention_mask": (T,),
        #   "labels": (T,),
        # }
```

### マルチターン会話の変換

LLaVA の会話フォーマット → Qwen2.5 チャットテンプレートへの変換:

```
元データ:
  human: "<image>\nWhat are the colors of the bus?"
  gpt:   "The bus is white and red."
  human: "What feature can be seen on the front?"
  gpt:   "The front features a large window..."

↓ 変換後 (Qwen2.5 テンプレート):

<|im_start|>system
You are a visual assistant.<|im_end|>
<|im_start|>user
What are the colors of the bus?<|im_end|>
<|im_start|>assistant
The bus is white and red.<|im_end|>
<|im_start|>user
What feature can be seen on the front?<|im_end|>
<|im_start|>assistant
The front features a large window...<|im_end|>
```

**注意**: `<image>` トークンは human の value から除去する（visual tokens は model 側で別途注入）。

### Labels 構築（マルチターン対応）

```python
def build_labels(tokenizer, messages, caption_texts):
    """Build labels where only assistant responses have loss.

    全体を tokenize した後、assistant の応答部分だけ有効な label にする。
    system, user, special tokens はすべて -100。
    """
    # 1. 全体テキストを構築
    full_text = ""
    for msg in messages:
        full_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # 2. assistant 応答部分の位置を特定
    #    "<|im_start|>assistant\n" の直後 〜 "<|im_end|>" の直前
    labels = [-100] * len(full_ids)
    # ... assistant 部分を特定して labels にコピー ...

    return full_ids, labels
```

**Loss マスクの注意点**:
- `<|im_start|>assistant\n` → `-100`（プロンプト部分）
- assistant の回答テキスト → **有効な label**
- `<|im_end|>` → **有効な label**（EOS の生成を学習する必要がある）

### Collator

```python
class InstructCollator:
    """Pad sequences within a batch. Truncate to max_length."""
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # pixel_values: stack → (B, 3, 224, 224)
        # input_ids: pad → (B, T), 右 padding
        # attention_mask: pad → (B, T)
        # labels: pad with -100 → (B, T)
        # truncate to max_length (2048)
```

---

## Step 3: 学習スクリプト

### train_stage2.py

```python
def main():
    # --- Config ---
    config = {
        "lr": 2e-5,                   # LLM + Adapter
        "ve_lr": 1e-5,                # DINOv2 (メイン LR の半分)
        "batch_size": 1,              # micro-batch (全パラメータ解凍なので小さく)
        "grad_accum": 128,            # → global batch = 128
        "epochs": 2,                  # Imp: 2ep が最適
        "warmup_ratio": 0.03,
        "weight_decay": 0.1,          # LLaVA-Phi: 小型モデルでは正則化重要
        "max_grad_norm": 1.0,
        "save_steps": 25,             # SmolVLM: 最適点は訓練終了時とは限らない
        "logging_steps": 5,
        "output_dir": "checkpoints/stage2",
        "stage1_checkpoint": "checkpoints/stage1/best.pt",
    }

    # --- Model ---
    model = QwenVLMini()

    # Stage 1 の Adapter 重みをロード
    ckpt = torch.load(config["stage1_checkpoint"])
    model.adapter.load_state_dict(ckpt["adapter_state_dict"])
    print("Loaded Stage 1 adapter weights.")

    model.set_stage2()  # 全パラメータ解凍
    model.to(device)

    # gradient checkpointing で VRAM 節約
    model.llm.gradient_checkpointing_enable()

    # --- Data ---
    dataset = InstructDataset(json_path, image_dir, model.tokenizer, IMAGE_TRANSFORM)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"],
                            shuffle=True, num_workers=4,
                            collate_fn=InstructCollator(model.tokenizer))

    # --- Optimizer (パラメータグループ分離) ---
    ve_params = list(model.vision_encoder.parameters())
    other_params = list(model.adapter.parameters()) + list(model.llm.parameters())

    optimizer = AdamW([
        {"params": ve_params, "lr": config["ve_lr"]},        # DINOv2: 1e-5
        {"params": other_params, "lr": config["lr"]},         # LLM + Adapter: 2e-5
    ], betas=(0.9, 0.95), weight_decay=config["weight_decay"])

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

                wandb.log({
                    "loss": loss.item() * config["grad_accum"],
                    "lr_ve": optimizer.param_groups[0]["lr"],
                    "lr_llm": optimizer.param_groups[1]["lr"],
                })

            if (step + 1) % config["save_steps"] == 0:
                save_full_checkpoint(model, optimizer, step, config["output_dir"])

    # 最終チェックポイント
    save_full_checkpoint(model, optimizer, "final", config["output_dir"])
```

### チェックポイント保存（全パラメータ）

Stage 2 では全パラメータが変更されるため、全モデルを保存:

```python
def save_full_checkpoint(model, optimizer, step, output_dir):
    torch.save({
        "vision_encoder_state_dict": model.vision_encoder.state_dict(),
        "adapter_state_dict": model.adapter.state_dict(),
        "llm_state_dict": model.llm.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }, f"{output_dir}/checkpoint-{step}.pt")
```

### VRAM 見積もり

| コンポーネント | サイズ |
|---|---|
| 全パラメータ (582M × 12 bytes) | ~7.0 GB |
| Activation (checkpointing ON) | ~2-3 GB |
| **合計** | **~10 GB** |

RTX 4090 (24 GB) で余裕あり。micro-batch=2 も試せる可能性あり。

### 学習時間の見積もり

- データ: 150K サンプル × 2 エポック = 300K steps
- Global batch: 128 → ~2,344 optimizer steps
- 全パラメータ解凍 + gradient checkpointing
- **見積もり: 12〜24 時間**

---

## Step 4: 訓練不安定時の対策

plan.md §3.2 の段階的対策を実装する。
**まず全解凍で試行し、発散したら順に対策を適用する。**

### 対策 1: DINOv2 後半 6 層のみ解凍（Share Recipe）

```python
def set_stage2_share_recipe(model):
    """TinyLLaVA Share recipe: freeze first 6 layers of DINOv2."""
    model.vision_encoder.dinov2.requires_grad_(False)
    # ViT-B/14 (12層): 後半 6 層 (6-11) のみ解凍
    for layer in model.vision_encoder.dinov2.encoder.layer[6:]:
        layer.requires_grad_(True)
    model.adapter.requires_grad_(True)
    model.llm.requires_grad_(True)
```

### 対策 2: LoRA (rank=256)

```python
from peft import LoraConfig, get_peft_model

def apply_lora(model, rank=256):
    """Apply LoRA to LLM (Imp: rank=256 が最適)."""
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
    )
    model.llm = get_peft_model(model.llm, lora_config)
```

LoRA 使用時は `peft` パッケージが必要: `uv add peft`

### 対策 3: DINOv2 frozen のまま Adapter + LLM のみ

```python
def set_stage2_ve_frozen(model):
    """COMM方式: DINOv2 frozen, Adapter + LLM のみ学習。"""
    model.vision_encoder.freeze()
    model.adapter.requires_grad_(True)
    model.llm.requires_grad_(True)
```

### 推奨ワークフロー

```
1. まず VE frozen + Adapter + LLM で設計を固める（データ、ハイパラ検証）
2. 設計が固まったら全解凍 (set_stage2) で最終訓練
3. 発散したら → Share Recipe → LoRA → VE frozen の順に試す
```

---

## テスト計画

### T1: データパイプライン単体テスト（実装直後に実行）

マルチターン会話の処理と loss mask が正しいかを検証する。
ファイル: `tests/test_instruct_dataset.py`

```python
def test_dataset_returns_correct_shapes():
    """出力形状が正しいこと。"""
    sample = dataset[0]
    assert sample["pixel_values"].shape == (3, 224, 224)
    assert sample["input_ids"].ndim == 1
    assert len(sample["input_ids"]) == len(sample["labels"])
    assert len(sample["input_ids"]) <= 2048, "Should be truncated"

def test_image_tag_removed():
    """<image> トークンがテキストから除去されていること。"""
    sample = dataset[0]
    text = tokenizer.decode(sample["input_ids"])
    assert "<image>" not in text, "<image> should be removed from text"

def test_loss_mask_only_assistant():
    """labels が assistant の回答部分のみ有効であること。"""
    sample = dataset[0]
    labels = sample["labels"]
    input_ids = sample["input_ids"]

    valid_mask = labels != -100
    valid_count = valid_mask.sum().item()
    assert valid_count > 0, "Should have valid labels"

    # 有効な labels のテキストを復元
    valid_ids = input_ids[valid_mask]
    valid_text = tokenizer.decode(valid_ids)

    # system/user のテキストが含まれていないこと
    assert "You are a visual assistant" not in valid_text, \
        "System prompt should not be in labels"

    # im_end が含まれていること（EOS 学習）
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert im_end_id in input_ids[valid_mask].tolist(), \
        "Labels should include <|im_end|>"

def test_multiturn_labels():
    """マルチターンで全ての assistant ターンが labels に含まれること。"""
    # マルチターン（4ターン以上）のサンプルを探す
    for i in range(len(dataset)):
        sample = dataset[i]
        labels = sample["labels"]
        input_ids = sample["input_ids"]

        valid_mask = labels != -100
        if valid_mask.sum() < 10:
            continue

        # 有効な label の位置が不連続（= 複数の assistant ターン）であること
        valid_indices = valid_mask.nonzero().squeeze()
        if valid_indices.ndim == 0:
            continue
        gaps = (valid_indices[1:] - valid_indices[:-1]) > 1
        if gaps.any():
            print(f"✓ Multi-turn labels verified at sample {i}: "
                  f"{gaps.sum().item()+1} separate assistant segments")
            return
    print("⚠ No multi-turn sample found (may need more data)")

def test_truncation():
    """2048 トークン以上の会話が正しく truncate されること。"""
    # 長い会話サンプルで確認
    for i in range(len(dataset)):
        sample = dataset[i]
        assert len(sample["input_ids"]) <= 2048

def test_collator_batch():
    """Collator がバッチを正しく構築すること。"""
    samples = [dataset[i] for i in range(4)]
    batch = collator(samples)
    assert batch["pixel_values"].shape == (4, 3, 224, 224)
    assert batch["input_ids"].shape[0] == 4
    # パディング部分の labels は -100
    pad_mask = batch["attention_mask"] == 0
    assert (batch["labels"][pad_mask] == -100).all()
```

### T2: 学習中の自動チェック（学習スクリプト内に組み込み）

```python
# train_stage2.py の学習ループ内に組み込む

# === Check 1: Loss NaN / Inf 検知 ===
if torch.isnan(output.loss) or torch.isinf(output.loss):
    raise RuntimeError(
        f"Loss is {'NaN' if torch.isnan(output.loss) else 'Inf'} at step {step}. "
        f"Consider: lower lr, gradient clipping, or Share Recipe."
    )

# === Check 2: Gradient norm モニタリング ===
if (step + 1) % config["grad_accum"] == 0:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
    wandb.log({"grad_norm": grad_norm.item()})
    if grad_norm > 10.0:
        print(f"⚠ High gradient norm: {grad_norm:.2f} at step {step}")

# === Check 3: 学習進行の確認 ===
if global_step == 0:
    initial_loss = loss_value
if global_step == 50:
    if loss_value >= initial_loss:
        print(f"⚠ Loss not decreasing after 50 steps: "
              f"{initial_loss:.4f} → {loss_value:.4f}")
    else:
        print(f"✓ Learning check: {initial_loss:.4f} → {loss_value:.4f}")

# === Check 4: DINOv2 の重みが更新されていること ===
if global_step == 0:
    ve_snapshot = model.vision_encoder.dinov2.encoder.layer[-1] \
        .attention.attention.query.weight.data[:3, :3].clone()
if global_step == 10:
    ve_current = model.vision_encoder.dinov2.encoder.layer[-1] \
        .attention.attention.query.weight.data[:3, :3]
    if torch.equal(ve_snapshot, ve_current):
        print("⚠ DINOv2 weights not changing — check requires_grad")
    else:
        print("✓ DINOv2 weights are being updated")
```

### T3: 学習後の評価テスト（学習完了後に実行）

ファイル: `tests/test_stage2_trained.py`

```python
def test_different_questions_different_answers():
    """同じ画像に異なる質問をすると異なる回答が返ること。"""
    questions = [
        "Describe this image.",
        "How many people are in this image?",
        "What colors are dominant?",
    ]
    answers = []
    for q in questions:
        text = generate_answer(model, "test_images/street.jpg", q)
        answers.append(text)
    unique = set(answers)
    assert len(unique) >= 2, \
        f"Same answer for different questions: {answers}"

def test_yes_no_capability():
    """Yes/No 形式の質問に Yes か No で答えられること（POPE の前提）。"""
    yes_no_questions = [
        ("test_images/dog.jpg", "Is there a dog in this image?"),
        ("test_images/dog.jpg", "Is there an elephant in this image?"),
    ]
    for img_path, question in yes_no_questions:
        answer = generate_answer(model, img_path, question)
        first_word = answer.strip().split()[0].lower().rstrip(".,!")
        is_yes_no = first_word in ("yes", "no")
        print(f"Q: {question} → A: {answer[:50]} (yes/no: {is_yes_no})")
        # assert ではなく確認（Stage 2 初期は不安定な場合がある）

def test_pope_above_random():
    """POPE accuracy がランダム (50%) を上回ること。"""
    accuracy = evaluate_pope(model, pope_data_path)
    print(f"POPE accuracy: {accuracy:.1%}")
    assert accuracy > 0.50, \
        f"POPE accuracy ({accuracy:.1%}) should be above random (50%)"

def test_stage2_better_than_stage1():
    """Stage 2 が Stage 1 より改善していること。"""
    loss_stage1 = compute_avg_loss(model_stage1, eval_dataloader, 50)
    loss_stage2 = compute_avg_loss(model_stage2, eval_dataloader, 50)
    print(f"Avg loss: Stage 1={loss_stage1:.4f}, Stage 2={loss_stage2:.4f}")
    assert loss_stage2 < loss_stage1, \
        "Stage 2 should have lower loss than Stage 1"
```

### テスト実行タイミング

| テスト | タイミング | 実行方法 | 合格基準 |
|---|---|---|---|
| **T1** (データパイプライン) | Dataset 実装直後 | `pytest tests/test_instruct_dataset.py` | 全 assert 通過 |
| **T2** (学習中チェック) | 学習中（自動） | train_stage2.py 内 | NaN なし, grad_norm < 10, loss 減少 |
| **T3** (学習後評価) | 学習完了後 | `python tests/test_stage2_trained.py` | POPE > 50% |

---

## Step 5: 評価

### 定性的評価（eval_qualitative.py を拡張）

Stage 2 ではより多様な質問で評価:

```python
EVAL_QUESTIONS = [
    "Describe this image in detail.",
    "What objects can you see in this image?",
    "What is the weather like in this image?",
    "How many people are in this image?",
    "What colors are dominant in this image?",
]
```

### 定量的評価（eval_benchmark.py）

lmms-eval を使用したベンチマーク評価:

```bash
uv add lmms-eval
```

```python
# カスタムモデルを lmms-eval に登録する方法は
# lmms-eval のドキュメントを参照
# もしくはシンプルに自前で POPE を実装する:

def evaluate_pope(model, pope_data_path):
    """POPE (Polling-based Object Probing Evaluation).

    Yes/No の物体存在判定。Accuracy を計測。
    ランダム回答は 50%。Stage 2 後は 60%+ を目標。
    """
    correct = 0
    total = 0
    for sample in pope_data:
        # image + "Is there a {object} in the image?"
        # → model.generate → "Yes" or "No"
        # → ground truth と比較
        ...
    return correct / total
```

### 評価プロトコル（plan.md §10.19）

- **greedy decoding**: `do_sample=False, temperature=None`
- **VQA 系ストップワード**: `\n` で生成停止
- **評価用プロンプト**:
  - POPE → "Answer the question using a single word or phrase."
  - ScienceQA → "Answer with the option's letter from the given choices directly."

### 評価ベンチマーク

| 優先度 | ベンチマーク | 目標 | 備考 |
|---|---|---|---|
| **Tier 1** | POPE | >60% | ランダム 50% を大幅に上回る |
| **Tier 1** | ScienceQA-IMG | >55% | ランダム 25% を上回る |
| Tier 2 | VQAv2 | 定性的に改善 | スコアは参考値 |

### Exit 条件

| 条件 | 確認方法 |
|---|---|
| Loss が安定して下がる | wandb の loss curve |
| 画像への質問に妥当な回答 | eval_qualitative.py |
| Stage 1 より明確に改善 | 同じ画像で Stage 1/2 の出力比較 |
| POPE > 50% (ランダム超え) | eval_benchmark.py |

---

## Step 6: Cosmos Reason Mini への引き継ぎ

Stage 2 完了後、学習済み重みを保存して Cosmos Reason Mini に引き継ぐ。

```python
def export_for_cosmos_reason(model, output_path):
    """Export trained weights for Cosmos Reason Mini."""
    torch.save({
        "vision_encoder": model.vision_encoder.state_dict(),
        "adapter": model.adapter.state_dict(),
        "llm": model.llm.state_dict(),
        "config": {
            "vision_model": "facebook/dinov2-base",
            "llm_model": "Qwen/Qwen2.5-0.5B",
            "adapter_type": "mlp",
            "adapter_ratio": 4,
            "image_size": 224,
            "num_visual_tokens": 256,
        },
    }, output_path)
```

---

## 実装順序

```
Step 1: データダウンロード (COCO 2014 + LLaVA-Instruct)   (1-2 hours)
Step 2: InstructDataset + Collator                          (2-3 hours)
Step 3: train_stage2.py (学習ループ + パラメータグループ)    (2-3 hours)
Step 4: 不安定時の対策コード (必要に応じて)                 (1 hour)
Step 5: eval_benchmark.py (POPE 評価)                       (1-2 hours)
学習実行                                                    (12-24 hours)
Step 6: Cosmos Reason Mini 引き継ぎ用 export                (30 min)
```

---

## 全体フロー図

```
Stage 0 (完了)
  model.py: VisionEncoder + Adapter + QwenVLMini
  ↓
Stage 1 (Feature Alignment)
  data: LLaVA-CC3M-Pretrain-595K
  学習: Adapter のみ (lr=1e-3, 1 epoch, ~3-6h)
  評価: 画像→キャプション生成の定性確認
  ↓ Adapter 重みを引き継ぎ
Stage 2 (Visual Instruction Tuning)
  data: LLaVA-Instruct-150K (+ COCO images)
  学習: 全パラメータ (lr=2e-5/1e-5, 2 epochs, ~12-24h)
  評価: POPE >60%, 定性的に VQA が動く
  ↓ 全重みを export
Cosmos Reason Mini に引き継ぎ
```
