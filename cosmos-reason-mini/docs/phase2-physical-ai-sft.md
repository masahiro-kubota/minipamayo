# Phase 2: Physical AI SFT — 具体的実装プラン

## 目的

Cosmos-Reason1 の Physical AI SFT（§5.1 + §7.1）を小規模再現する。
Phase 1 で作成した運転ドメインの QA データを使い、Qwen2.5-VL Mini の重みを初期値として **運転シーンの理解・推論能力** を SFT で獲得する。

### Cosmos-Reason1 論文との対応

| Cosmos-Reason1 §7.1 | Cosmos Reason Mini Phase 2 | 差分 |
|---|---|---|
| 初期重み: Qwen2.5-VL（既製 VLM） | Qwen2.5-VL Mini Stage 2.1 checkpoint | 同思想（VLM を SFT で特化） |
| データ: ~4M サンプル、5 ドメイン | ~7,500 サンプル、運転のみ | ~400 倍差 |
| LR: 1e-5 → 1e-6（cosine） | LR: 2e-5 → 2e-6（cosine） | Mini の方が若干大きめ（規模差を補う） |
| 12,500 iterations | ~300〜470 iterations | データ規模に比例 |
| バッチ 256（multi-node） | バッチ 16（single GPU） | VRAM 制約 |
| 結果: 物理的常識 +6.9%, 具現化推論 +11.0% | 定性的改善 + MCQ ベースライン確立 | 評価方法が異なる |

---

## 前提条件

- Phase 1 完了: `data/sft/qa_train.json` と `data/sft/qa_eval.json` が存在すること
- Qwen2.5-VL Mini Stage 2.1 チェックポイント: `../qwen-vl-mini/checkpoints/stage2.1/checkpoint-5247.pt`
- nuScenes 画像: `data/nuscenes/samples/CAM_FRONT/` にアクセス可能

---

## 学習設定サマリ

| 項目 | 値 | 根拠 |
|---|---|---|
| **初期重み** | Stage 2.1 checkpoint-5247.pt | 汎用 VLM 能力を初期値として活用 |
| **訓練データ** | qa_train.json (~7,500 QA) | Phase 1 で生成 |
| **Trainable** | VE + Adapter + LLM（全パラメータ） | design.md §4.1.6 に準拠 |
| **LR (LLM + Adapter)** | 2e-5 → 2e-6 (cosine) | Cosmos-Reason1 (1e-5) より若干大きめ |
| **LR (VE)** | 1e-5 → 1e-6 (cosine) | LLM の半分 |
| **Micro Batch** | 1 | VRAM 制約 |
| **Grad Accumulation** | 16 | Global Batch Size = 16 |
| **Epochs** | 1〜2 | Imp 知見: 小型 VLM では 2 epoch が最適 |
| **Weight Decay** | 0.01 | Cosmos-Reason1 の 0.1 より低め（データ量が少ないため） |
| **Precision** | bf16 | — |
| **Gradient Checkpointing** | ON | VRAM 節約 |
| **Optimizer** | AdamW (β1=0.9, β2=0.95) | Cosmos-Reason1 と同じ β 値 |
| **Max Seq Length** | 2,048 | — |
| **Save Steps** | 25 | SmolVLM 知見: 最適点は途中にある |
| **NEFTune** | alpha=5 | Stage 2.1 で効果確認済み |
| **出力先** | `checkpoints/sft/` | — |

### 推定学習パラメータ

| 項目 | 1 epoch | 2 epochs |
|---|---|---|
| データ | ~7,500 samples | ~15,000 samples |
| Optimizer Steps | ~469 | ~938 |
| 推定時間 | ~0.5h | ~1h |
| VRAM | ~10 GB | ~10 GB |

**データ量が少ないため、学習時間は非常に短い。** 1 回の実験が 30 分〜1 時間で完了するため、ハイパーパラメータ探索が容易。

### 学習率の設計根拠

- **Cosmos-Reason1 (7B)**: 1e-5 → 1e-6。大規模事前学習済み VLM を微調整
- **Qwen2.5-VL Mini Stage 2.1 (0.5B)**: 2e-5。Stage 2 と同じ LR。規模が小さいため少し大きめが安定

Stage 2.1 で LR=2e-5 が安定していた実績があるため、同じ値を初期設定とする。

---

## プロジェクト構成（Phase 2 で追加・変更するファイル）

```
cosmos-reason-mini/src/cosmos_reason_mini/
├── model_loader.py              # 新規: QwenVLMini のロード・初期化ユーティリティ
├── data/
│   └── driving_dataset.py       # 新規: 運転 QA 用 Dataset クラス
├── train_sft.py                 # 新規: Physical AI SFT 学習スクリプト
├── eval_qualitative.py          # 新規: 定性的評価（テキスト生成）
└── eval_mcq.py                  # 新規: MCQ 評価（Phase 3 で使用、ベースライン記録）
```

---

## Step 0: モデルローダー実装

### 目的

qwen-vl-mini の `QwenVLMini` モデルを Cosmos Reason Mini から利用するためのユーティリティ。
Stage 2.1 のチェックポイントをロードし、SFT 用に再構成する。

### 新規ファイル: model_loader.py

```python
"""QwenVLMini モデルのロードユーティリティ。"""
import torch
from qwen_vl_mini.model import QwenVLMini


def load_vlm_from_checkpoint(
    checkpoint_path: str,
    neftune_alpha: float = 0.0,
    device: str = "cuda",
) -> QwenVLMini:
    """Stage 2.1 チェックポイントから VLM をロードする。

    チェックポイント形式（train_stage2.py の save_checkpoint）:
        {
            "vision_encoder_state_dict": ...,
            "adapter_state_dict": ...,
            "llm_state_dict": ...,
            "optimizer_state_dict": ...,
            "scheduler_state_dict": ...,
            "global_step": int,
            "epoch": int,
        }
    """
    model = QwenVLMini(neftune_alpha=neftune_alpha)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # 3 つの state_dict を個別にロード（train_stage2.py と同じ方式）
    model.vision_encoder.load_state_dict(ckpt["vision_encoder_state_dict"])
    model.adapter.load_state_dict(ckpt["adapter_state_dict"])
    model.llm.load_state_dict(ckpt["llm_state_dict"])

    model = model.to(device)
    return model
```

**注意**: `qwen_vl_mini` パッケージがインポート可能であること。`pyproject.toml` の `[tool.uv.sources]` で依存関係を設定済み。

---

## Step 1: DrivingQADataset 実装

### 目的

Phase 1 で生成した LLaVA 形式 JSON を読み込み、QwenVLMini の入力形式に変換する Dataset クラス。
qwen-vl-mini の `InstructDataset` をベースにするが、画像パスの解決が異なるため新規作成する。

### 新規ファイル: data/driving_dataset.py

```python
"""運転ドメイン QA データセット。

InstructDataset (qwen-vl-mini) と同じトークン化方式を使用する。
差分は画像パス解決のみ（nuScenes 単一ソース）。
"""
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

IGNORE_INDEX = -100


class DrivingQADataset(Dataset):
    """Phase 1 で生成した LLaVA 形式の運転 QA データを読み込む。

    JSON フォーマット:
    [
        {
            "id": "driving_understanding_000001",
            "image": "samples/CAM_FRONT/xxx.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\\nQuestion?"},
                {"from": "gpt", "value": "Answer."}
            ]
        },
        ...
    ]
    """

    def __init__(
        self,
        json_path: str,
        image_root: str,             # data/nuscenes/ ディレクトリ
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
        max_length: int = 2048,
        transform=None,
    ):
        with open(json_path) as f:
            self.data = json.load(f)
        self.image_root = Path(image_root)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        self.max_length = max_length
        self.transform = transform or self._default_transform()

    def _default_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _build_messages(self, conversations: list[dict]) -> list[dict]:
        """LLaVA conversations → Qwen2.5 chat messages に変換。
        InstructDataset._build_messages() と同一ロジック。
        """
        messages = [{"role": "system", "content": "You are a visual assistant."}]
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            content = (
                turn["value"]
                .replace("<image>\n", "")
                .replace("\n<image>", "")
                .replace("<image>", "")
            ).strip()
            if content:
                messages.append({"role": role, "content": content})
        return messages

    def _tokenize_with_labels(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        """メッセージをトークン化し、ラベルマスクを生成。
        InstructDataset._tokenize_with_labels() と同一ロジック。
        assistant の body + footer のみ loss 対象。
        """
        all_ids = []
        all_labels = []
        for msg in messages:
            header = f"<|im_start|>{msg['role']}\n"
            body = msg["content"]
            footer = "<|im_end|>\n"

            header_ids = self.tokenizer.encode(header, add_special_tokens=False)
            body_ids = self.tokenizer.encode(body, add_special_tokens=False)
            footer_ids = self.tokenizer.encode(footer, add_special_tokens=False)
            turn_ids = header_ids + body_ids + footer_ids

            if msg["role"] == "assistant":
                turn_labels = [IGNORE_INDEX] * len(header_ids) + body_ids + footer_ids
            else:
                turn_labels = [IGNORE_INDEX] * len(turn_ids)

            all_ids.extend(turn_ids)
            all_labels.extend(turn_labels)
        return all_ids, all_labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        # --- Image ---
        image_path = self.image_root / sample["image"]
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, Exception):
            return self[random.randint(0, len(self) - 1)]
        pixel_values = self.transform(image)

        # --- Text (InstructDataset と同一方式) ---
        messages = self._build_messages(sample["conversations"])
        input_ids, labels = self._tokenize_with_labels(messages)

        # Truncate
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
```

### InstructDataset との差分

| 項目 | InstructDataset (qwen-vl-mini) | DrivingQADataset |
|---|---|---|
| 画像パス解決 | COCO/GQA/VG 等の multi-source | nuScenes 単一ソース |
| text-only 対応 | あり（WizardLM） | なし（全て画像付き） |
| 画像解像度 | 多様（COCO: 640×480 等） | 統一（1600×900 → 224×224） |
| データ形式 | LLaVA JSON | LLaVA JSON（同一） |

**DrivingQADataset を新規作成する理由**: InstructDataset は COCO 固有のファイル名変換（bare ID → COCO_train2014_XXX.jpg）等の互換ロジックが多い。nuScenes 用にシンプルなクラスを作る方がメンテナブル。

---

## Step 2: SFT 学習スクリプト

### 目的

qwen-vl-mini の `train_stage2.py` をベースに、Cosmos Reason Mini の SFT 用に調整した学習スクリプト。

### 新規ファイル: train_sft.py

```python
"""Physical AI SFT 学習スクリプト。

Usage:
    cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.train_sft \
        --json_path data/sft/qa_train.json \
        --image_root data/nuscenes \
        --vlm_checkpoint ../qwen-vl-mini/checkpoints/stage2.1/checkpoint-5247.pt \
        --output_dir checkpoints/sft
"""
import argparse
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import wandb

# 主要コンポーネント
DEFAULTS = {
    "json_path": "data/sft/qa_train.json",
    "image_root": "data/nuscenes",
    "vlm_checkpoint": "../qwen-vl-mini/checkpoints/stage2.1/checkpoint-5247.pt",
    "output_dir": "checkpoints/sft",
    "epochs": 1,
    "lr_llm": 2e-5,
    "lr_ve": 1e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "grad_accum_steps": 16,
    "max_length": 2048,
    "neftune_alpha": 5.0,
    "save_steps": 25,
    "logging_steps": 5,
}


def create_optimizer(model, lr_llm, lr_ve, weight_decay):
    """VE と LLM/Adapter で異なる LR を設定。"""
    ve_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("vision_encoder."):
            ve_params.append(param)
        else:
            other_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": lr_llm},
            {"params": ve_params, "lr": lr_ve},
        ],
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )


def create_scheduler(optimizer, num_training_steps, warmup_ratio):
    """Cosine annealing with warmup (min_lr = lr/10)."""
    warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(num_training_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def collate_fn(batch):
    """可変長シーケンスのパディング。"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i, :seq_len] = item["labels"]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()
    for key, default in DEFAULTS.items():
        arg_type = type(default) if not isinstance(default, bool) else None
        parser.add_argument(f"--{key}", type=arg_type, default=default)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model ---
    from cosmos_reason_mini.model_loader import load_vlm_from_checkpoint
    model = load_vlm_from_checkpoint(
        args.vlm_checkpoint,
        neftune_alpha=args.neftune_alpha,
        device=device,
    )
    model.train()

    # Gradient checkpointing
    if hasattr(model.vision_encoder, "gradient_checkpointing_enable"):
        model.vision_encoder.gradient_checkpointing_enable()
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()

    # --- Dataset ---
    from cosmos_reason_mini.data.driving_dataset import DrivingQADataset
    dataset = DrivingQADataset(
        json_path=args.json_path,
        image_root=args.image_root,
        max_length=args.max_length,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=collate_fn,
        num_workers=4, pin_memory=True,
    )

    # --- Optimizer & Scheduler ---
    num_training_steps = len(dataloader) * args.epochs // args.grad_accum_steps
    optimizer = create_optimizer(model, args.lr_llm, args.lr_ve, args.weight_decay)
    scheduler = create_scheduler(optimizer, num_training_steps, args.warmup_ratio)

    # --- wandb ---
    wandb.init(project="cosmos-reason-mini", name="phase2-sft", config=vars(args))

    # --- Training Loop ---
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # autocast 必須: DINOv2 (fp32) + LLM (bf16) の dtype 不一致を解決
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    wandb.log({
                        "loss": outputs["loss"].item(),
                        "lr": scheduler.get_last_lr()[0],
                        "step": global_step,
                        "epoch": epoch,
                    })

                if global_step % args.save_steps == 0:
                    save_path = Path(args.output_dir) / f"checkpoint-{global_step}.pt"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    # train_stage2.py と同一のチェックポイント形式
                    torch.save({
                        "vision_encoder_state_dict": model.vision_encoder.state_dict(),
                        "adapter_state_dict": model.adapter.state_dict(),
                        "llm_state_dict": model.llm.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                    }, save_path)
                    print(f"Saved checkpoint: {save_path}")

    # Final save
    save_path = Path(args.output_dir) / f"checkpoint-{global_step}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "vision_encoder_state_dict": model.vision_encoder.state_dict(),
        "adapter_state_dict": model.adapter.state_dict(),
        "llm_state_dict": model.llm.state_dict(),
        "global_step": global_step,
    }, save_path)
    print(f"Training complete. Final checkpoint: {save_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
```

### train_stage2.py との主な差分

| 項目 | train_stage2.py (qwen-vl-mini) | train_sft.py (cosmos-reason-mini) |
|---|---|---|
| Dataset | InstructDataset | DrivingQADataset |
| 初期重み | Stage 1 checkpoint | Stage 2.1 checkpoint |
| LR | llm=2e-5, ve=1e-5 | 同一 |
| Weight Decay | 0.1 | **0.01**（少データでの過学習抑制） |
| Save Steps | 100 | **25**（SmolVLM 知見） |
| Grad Accum | 128 | **16**（データ量が少ないため） |
| Epochs | 1 | 1〜2 |
| wandb run name | stage2.1-bunny695k-neftune | **phase2-sft** |

---

## Step 3: パイプラインテスト

### 3a: データセットテスト

```bash
cd cosmos-reason-mini && uv run python -c "
from cosmos_reason_mini.data.driving_dataset import DrivingQADataset

ds = DrivingQADataset(
    json_path='data/sft/qa_mini.json',
    image_root='data/nuscenes',
)
print(f'Dataset size: {len(ds)}')
sample = ds[0]
print(f'pixel_values: {sample[\"pixel_values\"].shape}')
print(f'input_ids: {sample[\"input_ids\"].shape}')
print(f'labels non-ignored: {(sample[\"labels\"] != -100).sum().item()}')
print('Dataset test: OK')
"
```

### 3b: モデルロード + Forward テスト

```bash
cd cosmos-reason-mini && uv run python -c "
import torch
from cosmos_reason_mini.model_loader import load_vlm_from_checkpoint
from cosmos_reason_mini.data.driving_dataset import DrivingQADataset

model = load_vlm_from_checkpoint(
    '../qwen-vl-mini/checkpoints/stage2.1/checkpoint-5247.pt',
    neftune_alpha=5.0,
)
model.train()

ds = DrivingQADataset('data/sft/qa_mini.json', 'data/nuscenes')
sample = ds[0]
batch = {k: v.unsqueeze(0).cuda() for k, v in sample.items()}
outputs = model(**batch)
print(f'Loss: {outputs[\"loss\"].item():.4f}')
print('Forward test: OK')
"
```

### 3c: 期待する結果

- Dataset が正しくサンプルを返すこと
- Forward pass が通り、loss が有限値であること
- 初期 loss は ~3〜5 程度を想定（Stage 2.1 の最終 loss が ~3.3 だが、ドメインが変わるため上がる）

---

## Step 4: 学習実行

```bash
cd cosmos-reason-mini && PYTHONUNBUFFERED=1 uv run python -m cosmos_reason_mini.train_sft \
    --json_path data/sft/qa_train.json \
    --image_root data/nuscenes \
    --vlm_checkpoint ../qwen-vl-mini/checkpoints/stage2.1/checkpoint-5247.pt \
    --output_dir checkpoints/sft \
    --epochs 1 \
    --neftune_alpha 5 \
    --save_steps 25 \
    --logging_steps 5
```

### 推定

| 項目 | 値 |
|---|---|
| データ | ~7,500 × 1 epoch |
| Optimizer Steps | ~469 (7500 / grad_accum=16) |
| 推定学習時間 | ~30 分 |
| VRAM | ~10 GB |
| チェックポイント数 | ~19 (469/25) |

### 監視項目

1. **Loss カーブ**: 単調に減少すること。発散したら LR を半分に
2. **VRAM**: `nvidia-smi` で 10-12 GB 程度であること
3. **NaN**: loss が NaN になったらスキップ or LR 低減

### 訓練不安定時の対応

design.md §9.7 に基づき:
1. **全パラメータ解凍で発散** → LoRA (rank=256) を検討
2. **VE fine-tune で不安定** → `--freeze_ve` で VE を凍結

---

## Step 5: 評価

### 5a: 定性的評価

学習データとは**別の画像**（eval split）で推論を実行し、生成テキストの品質を確認。

```bash
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.eval_qualitative \
    --checkpoint checkpoints/sft/checkpoint-XXX.pt \
    --image_root data/nuscenes \
    --eval_json data/sft/qa_eval.json \
    --num_samples 20 \
    --seed 42
```

#### 新規ファイル: eval_qualitative.py（概要）

```python
"""定性的評価: eval 画像に対してテキスト生成し、GT と比較。"""
# 1. チェックポイントからモデルをロード
# 2. eval JSON からランダムに N サンプルを選択
# 3. 各サンプルに対して:
#    a. 画像を入力
#    b. 質問をプロンプトとして入力
#    c. model.generate() でテキスト生成
#    d. GT answer と生成テキストを並べて表示
# 4. 結果を JSON/テキストで保存
```

#### 評価基準

| 基準 | 内容 | Stage 2.1 baseline との比較 |
|---|---|---|
| シーン記述の正確さ | 道路タイプ、天候、物体の正しい記述 | 汎用記述 → 運転特化記述に改善 |
| 空間関係の正確さ | 先行車・歩行者の位置が正しいか | 新規能力 |
| 行動推論の妥当性 | 「減速すべき」等の判断が妥当か | 新規能力 |
| ハルシネーション | 存在しない物体を述べていないか | 劣化なし |
| テキスト品質 | 文法的に正しく、自然な日本語/英語か | 劣化なし |

### 5b: MCQ ベースライン記録

Phase 3（RL）の比較用に、SFT モデルの MCQ 正解率をベースラインとして記録する。
MCQ データは Phase 3 Step 0 で作成するが、ベースライン記録はここで予告。

### 5c: SFT 前後の比較

| 評価 | SFT 前 (Stage 2.1) | SFT 後 (Phase 2) | 期待 |
|---|---|---|---|
| 汎用 VLM 能力 | 良好（POPE 85.9%） | 劣化なし | ドメイン特化で汎用能力を大きく損なわない |
| 運転シーン記述 | 一般的な画像記述 | **運転に特化した記述** | 改善 |
| 空間推論（運転） | 基礎的 | **向上** | DINOv2 の強みを活かす |
| 行動推論（運転） | なし | **新規獲得** | SFT の主目的 |

---

## Step 6: チェックポイント選定

### 方針

SmolVLM の知見に基づき、**訓練終了時が必ずしも最適ではない**。
25 ステップごとに保存した ~19 チェックポイントから、最適なものを選定する。

### 選定基準

1. **定性的評価が良好** (Step 5a)
2. **Loss が十分低い**（ただし最小 loss ≠ 最良）
3. **ハルシネーションが少ない**

### 実行

```bash
# 主要なチェックポイント（25, 100, 200, 300, 最終）で定性評価を実行
for step in 25 100 200 300 469; do
    cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.eval_qualitative \
        --checkpoint checkpoints/sft/checkpoint-${step}.pt \
        --eval_json data/sft/qa_eval.json \
        --num_samples 20 --seed 42 \
        --output results/sft/qualitative_step${step}.json
done
```

---

## SFT が期待通りに進まない場合

| 問題 | 対策 |
|---|---|
| Loss が下がらない | LR を 2 倍に（4e-5）、epochs を 2 に |
| Loss が NaN / 発散 | LR を半分に、VE を frozen に |
| 運転固有の知識が獲得されない | データ量を増やす（nuScenes Full の全フレームを使用）|
| 汎用能力が大幅に劣化 | LR を下げる（1e-5）、epochs を 1 に抑える |
| VRAM 不足 | Grad accum を 32 に、batch size を確認 |

---

## 完了状況

| Step | 状態 | 備考 |
|------|------|------|
| Step 0: モデルローダー実装 | ✅ 完了 | model_loader.py |
| Step 1: DrivingQADataset 実装 | ✅ 完了 | driving_dataset.py |
| Step 2: SFT 学習スクリプト | ✅ 完了 | train_sft.py (autocast 必須) |
| Step 3: パイプラインテスト | ✅ 完了 (Mini) | 201 samples, forward OK |
| Step 4: 学習実行 | ✅ 完了 (Mini) | 24 steps, loss 1.32 → 0.76 |
| Step 5: 評価 | ✅ 完了 (Mini) | 定性的改善確認、MCQ 64.4% |
| Step 6: チェックポイント選定 | ✅ 完了 (Mini) | checkpoint-24.pt |

### Mini パイプライン検証結果

| 項目 | 値 |
|------|------|
| データ | 201 QA (Mini) |
| Epochs | 2 |
| Optimizer steps | 24 |
| Loss | 1.32 → 0.76 |
| 定性的評価 | SFT 後に GT 一致する回答あり |
| MCQ ベースライン | 64.4% (101 MCQ)
