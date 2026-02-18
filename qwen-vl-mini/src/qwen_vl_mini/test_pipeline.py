"""Fail-fast pipeline validation for Stage 1 training.

Tests the full pipeline (Dataset → DataLoader → forward → backward → optimizer step)
with a tiny subset of data before committing to full training.
"""

import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from qwen_vl_mini.data.pretrain_dataset import PretrainCollator, PretrainDataset
from qwen_vl_mini.model import IMAGE_TRANSFORM, QwenVLMini

DATA_DIR = Path("data/llava-pretrain")
JSON_PATH = DATA_DIR / "chat.json"
# Images are directly in llava-pretrain/, not in images/ subdirectory
IMAGE_DIR = DATA_DIR


def create_mini_json(n_samples: int = 8) -> Path:
    """Create a tiny JSON with only n_samples entries (using real images)."""
    with open(JSON_PATH) as f:
        full_data = json.load(f)

    # Filter to samples whose images actually exist
    mini_data = []
    for sample in full_data:
        img_path = IMAGE_DIR / sample["image"]
        if img_path.exists():
            mini_data.append(sample)
            if len(mini_data) >= n_samples:
                break

    mini_path = DATA_DIR / "_mini_test.json"
    with open(mini_path, "w") as f:
        json.dump(mini_data, f)
    print(f"Created mini dataset: {len(mini_data)} samples → {mini_path}")
    return mini_path


def test_dataset_loading(model, mini_json: Path):
    """T1-1: Dataset loads correctly and returns valid tensors."""
    print("\n=== T1-1: Dataset Loading ===")
    dataset = PretrainDataset(
        json_path=str(mini_json),
        image_dir=str(IMAGE_DIR),
        tokenizer=model.tokenizer,
        transform=IMAGE_TRANSFORM,
        max_length=512,
    )
    print(f"  Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"  pixel_values shape: {sample['pixel_values'].shape}")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")

    # Validate shapes
    assert sample["pixel_values"].shape == (3, 224, 224), "pixel_values shape mismatch"
    assert sample["input_ids"].dim() == 1, "input_ids should be 1D"
    assert sample["labels"].shape == sample["input_ids"].shape, "labels/input_ids shape mismatch"

    # Validate label mask: some should be -100 (prompt), some should be valid (caption)
    n_ignored = (sample["labels"] == -100).sum().item()
    n_valid = (sample["labels"] != -100).sum().item()
    print(f"  Labels: {n_ignored} ignored (prompt), {n_valid} valid (caption)")
    assert n_valid > 0, "No valid labels found — caption not being trained on!"
    assert n_ignored > 0, "No ignored labels — prompt should be masked!"

    print("  ✓ Dataset loading passed")
    return dataset


def test_collator(model, dataset):
    """T1-2: Collator pads correctly and DataLoader works."""
    print("\n=== T1-2: Collator + DataLoader ===")
    collator = PretrainCollator(model.tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collator)

    batch = next(iter(dataloader))
    print(f"  pixel_values: {batch['pixel_values'].shape}")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")

    B, T = batch["input_ids"].shape
    assert batch["pixel_values"].shape[0] == B, "batch size mismatch"
    assert batch["attention_mask"].shape == (B, T), "attention_mask shape mismatch"
    assert batch["labels"].shape == (B, T), "labels shape mismatch"

    # Padding check: attention_mask should have 0s only at padded positions
    for i in range(B):
        pad_positions = (batch["attention_mask"][i] == 0).sum().item()
        print(f"  Sample {i}: {T - pad_positions} tokens + {pad_positions} padding")

    print("  ✓ Collator + DataLoader passed")
    return dataloader


def test_forward_backward(model, dataloader, device):
    """T1-3: Full forward + backward pass with real data."""
    print("\n=== T1-3: Forward + Backward ===")
    model.set_stage1()
    model = model.to(device)
    model.train()

    batch = next(iter(dataloader))

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            pixel_values=batch["pixel_values"].to(device),
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

    loss = output.loss
    print(f"  Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss should be positive!"

    loss.backward()
    print("  ✓ Backward pass succeeded")

    # Check gradients exist on adapter
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.adapter.parameters()
    )
    assert has_grad, "Adapter has no gradients!"
    print("  ✓ Adapter gradients present")

    # Check no gradients on frozen modules
    ve_grads = any(p.grad is not None for p in model.vision_encoder.parameters())
    llm_grads = any(p.grad is not None for p in model.llm.parameters())
    assert not ve_grads, "VisionEncoder should have no gradients (frozen)!"
    assert not llm_grads, "LLM should have no gradients (frozen)!"
    print("  ✓ Frozen modules have no gradients")

    return loss.item()


def test_optimizer_step(model, dataloader, device):
    """T1-4: Optimizer step updates adapter weights."""
    print("\n=== T1-4: Optimizer Step ===")
    model.set_stage1()
    model = model.to(device)
    model.train()

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        betas=(0.9, 0.95),
    )

    # Snapshot adapter weights before step
    before = {k: v.clone() for k, v in model.adapter.state_dict().items()}

    batch = next(iter(dataloader))
    optimizer.zero_grad()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            pixel_values=batch["pixel_values"].to(device),
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

    output.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Check adapter weights changed
    after = model.adapter.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed, "Adapter weights didn't change after optimizer step!"
    print("  ✓ Adapter weights updated")

    return output.loss.item()


def test_overfit_mini(model, dataloader, device, n_steps: int = 20):
    """T1-5: Can overfit on tiny data (loss should decrease significantly)."""
    print(f"\n=== T1-5: Overfit Test ({n_steps} steps) ===")
    model.set_stage1()
    model = model.to(device)
    model.train()

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        betas=(0.9, 0.95),
    )

    losses = []
    for step in range(n_steps):
        batch = next(iter(dataloader))
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )

        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = output.loss.item()
        losses.append(loss_val)
        if step % 5 == 0 or step == n_steps - 1:
            print(f"  Step {step:3d}: loss={loss_val:.4f}")

    first_loss = losses[0]
    last_loss = losses[-1]
    ratio = last_loss / first_loss

    print(f"\n  Loss: {first_loss:.4f} → {last_loss:.4f} (ratio: {ratio:.3f})")
    if last_loss < first_loss:
        print("  ✓ Loss decreased — model is learning!")
    else:
        print("  ⚠ Loss did NOT decrease — something may be wrong")

    assert not any(torch.isnan(torch.tensor(v)) for v in losses), "NaN loss during overfit test!"


def main():
    print("=" * 60)
    print("  Stage 1 Pipeline Validation (Fail-Fast)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create mini dataset
    mini_json = create_mini_json(n_samples=8)

    # Load model
    print("\nLoading model...")
    model = QwenVLMini()

    # Run tests sequentially
    dataset = test_dataset_loading(model, mini_json)
    dataloader = test_collator(model, dataset)
    test_forward_backward(model, dataloader, device)
    test_optimizer_step(model, dataloader, device)
    test_overfit_mini(model, dataloader, device, n_steps=20)

    # Cleanup
    mini_json.unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("  ALL PIPELINE TESTS PASSED")
    print("=" * 60)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak VRAM: {peak:.0f} MB")


if __name__ == "__main__":
    main()
