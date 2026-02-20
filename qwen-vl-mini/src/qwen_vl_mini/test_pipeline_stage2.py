"""Fail-fast pipeline validation for Stage 2.1 training.

Tests the full pipeline (InstructDataset → DataLoader → forward → backward → optimizer step)
with a tiny subset of data before committing to full training.

Stage 2.1 additions over Stage 2:
- 665K multi-source dataset (COCO + GQA + TextVQA + OCR-VQA + VG + text-only)
- text-only samples (dummy zero image)
- NEFTune noise injection
- image_root for multi-source path resolution
"""

import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from qwen_vl_mini.data.instruct_dataset import InstructCollator, InstructDataset
from qwen_vl_mini.model import IMAGE_TRANSFORM, QwenVLMini

JSON_PATH = Path("data/llava-instruct/mix665k_full.json")
JSON_PATH_150K = Path("data/llava-instruct/llava_instruct_150k.json")
IMAGE_DIR = Path("data/coco/train2014")
IMAGE_ROOT = Path("data")
STAGE1_CKPT = Path("checkpoints/stage1/checkpoint-2325.pt")


def create_mini_json(n_samples: int = 8) -> Path:
    """Create a tiny JSON with only n_samples entries (using real images).

    Includes COCO samples + text-only samples if available in 665K data.
    Falls back to 150K if 665K not yet prepared.
    """
    json_path = JSON_PATH if JSON_PATH.exists() else JSON_PATH_150K
    print(f"Using dataset: {json_path}")

    with open(json_path) as f:
        full_data = json.load(f)

    mini_data = []
    has_text_only = False

    for sample in full_data:
        image_name = sample.get("image", "")

        if not image_name:
            # Text-only sample
            if not has_text_only:
                mini_data.append(sample)
                has_text_only = True
                if len(mini_data) >= n_samples:
                    break
            continue

        # Try to resolve image path
        if IMAGE_ROOT.exists() and "/" in image_name:
            if image_name.startswith("coco/"):
                bare_id = image_name.split("/")[-1]
                path = IMAGE_ROOT / "coco" / "train2014" / f"COCO_train2014_{bare_id}"
            else:
                path = IMAGE_ROOT / image_name
        else:
            bare = image_name.split("/")[-1] if "/" in image_name else image_name
            path = IMAGE_DIR / bare
            if not path.exists():
                path = IMAGE_DIR / f"COCO_train2014_{bare}"

        if path.exists():
            mini_data.append(sample)
            if len(mini_data) >= n_samples:
                break

    mini_path = Path("data/llava-instruct/_mini_test.json")
    with open(mini_path, "w") as f:
        json.dump(mini_data, f)
    text_only_count = sum(1 for s in mini_data if not s.get("image", ""))
    print(
        f"Created mini dataset: {len(mini_data)} samples ({text_only_count} text-only) → {mini_path}"
    )
    return mini_path


def test_dataset_loading(model, mini_json: Path):
    """T2-1: InstructDataset loads correctly and returns valid tensors."""
    print("\n=== T2-1: InstructDataset Loading ===")
    dataset = InstructDataset(
        json_path=str(mini_json),
        image_dir=str(IMAGE_DIR),
        tokenizer=model.tokenizer,
        transform=IMAGE_TRANSFORM,
        max_length=512,
        image_root=str(IMAGE_ROOT) if IMAGE_ROOT.exists() else "",
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

    # Validate label mask: system+user masked, assistant has valid labels
    n_ignored = (sample["labels"] == -100).sum().item()
    n_valid = (sample["labels"] != -100).sum().item()
    print(f"  Labels: {n_ignored} ignored (system+user), {n_valid} valid (assistant)")
    assert n_valid > 0, "No valid labels — assistant response not being trained on!"
    assert n_ignored > 0, "No ignored labels — system/user should be masked!"

    # Decode a small portion to sanity-check
    valid_mask = sample["labels"] != -100
    valid_ids = sample["input_ids"][valid_mask][:20]
    decoded = model.tokenizer.decode(valid_ids)
    print(f"  Sample valid tokens (first 20): {decoded[:80]}...")

    # Test text-only sample if present
    for i in range(len(dataset)):
        s = dataset.data[i]
        if not s.get("image", ""):
            print(f"\n  Testing text-only sample (idx={i})...")
            text_only = dataset[i]
            assert text_only["pixel_values"].shape == (3, 224, 224), "text-only pixel_values shape"
            assert text_only["pixel_values"].abs().sum() == 0, "text-only should be zero tensor"
            n_valid_to = (text_only["labels"] != -100).sum().item()
            assert n_valid_to > 0, "text-only should have valid labels"
            print(f"  ✓ Text-only sample OK (zero image, {n_valid_to} valid labels)")
            break

    print("  ✓ InstructDataset loading passed")
    return dataset


def test_collator(model, dataset):
    """T2-2: InstructCollator pads correctly and DataLoader works."""
    print("\n=== T2-2: InstructCollator + DataLoader ===")
    collator = InstructCollator(model.tokenizer, max_length=512)
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

    for i in range(B):
        pad_positions = (batch["attention_mask"][i] == 0).sum().item()
        valid_labels = (batch["labels"][i] != -100).sum().item()
        print(
            f"  Sample {i}: {T - pad_positions} tokens + {pad_positions} padding, {valid_labels} valid labels"
        )

    print("  ✓ InstructCollator + DataLoader passed")
    return dataloader


def test_stage1_checkpoint_loading(model):
    """T2-3: Stage 1 adapter checkpoint loads correctly."""
    print("\n=== T2-3: Stage 1 Checkpoint Loading ===")
    ckpt = torch.load(STAGE1_CKPT, map_location="cpu", weights_only=True)
    print(f"  Checkpoint keys: {list(ckpt.keys())}")

    # Load adapter weights
    model.adapter.load_state_dict(ckpt["adapter_state_dict"])
    print(f"  ✓ Adapter weights loaded from {STAGE1_CKPT}")

    # Verify adapter weights are different from random init
    # (if they loaded correctly, they shouldn't be all zeros)
    total_norm = sum(p.data.norm().item() for p in model.adapter.parameters())
    print(f"  Adapter total weight norm: {total_norm:.4f}")
    assert total_norm > 0, "Adapter weights are all zeros after loading!"
    print("  ✓ Stage 1 checkpoint loading passed")


def test_forward_backward_stage2(model, dataloader, device):
    """T2-4: Full forward + backward with Stage 2 mode (all params trainable)."""
    print("\n=== T2-4: Forward + Backward (Stage 2 full fine-tune) ===")
    model.set_stage2()
    model = model.to(device)
    model.llm.gradient_checkpointing_enable()
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable / total:.1%})")

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

    # Check gradients on all modules
    adapter_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.adapter.parameters()
    )
    assert adapter_grad, "Adapter has no gradients!"
    print("  ✓ Adapter gradients present")

    llm_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.llm.parameters())
    assert llm_grad, "LLM has no gradients!"
    print("  ✓ LLM gradients present")

    ve_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.vision_encoder.parameters()
    )
    assert ve_grad, "VisionEncoder has no gradients (should be unfrozen in Stage 2)!"
    print("  ✓ VisionEncoder gradients present")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Current peak VRAM: {peak:.0f} MB")

    return loss.item()


def test_param_groups_optimizer(model, dataloader, device):
    """T2-5: Parameter groups with separate lr + optimizer step updates all weights."""
    print("\n=== T2-5: Parameter Groups + Optimizer Step ===")
    model.set_stage2()
    model = model.to(device)
    model.llm.gradient_checkpointing_enable()
    model.train()

    ve_params = [p for p in model.vision_encoder.parameters() if p.requires_grad]
    other_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("vision_encoder.")
    ]

    print(f"  VE params: {sum(p.numel() for p in ve_params):,}")
    print(f"  Other params (Adapter+LLM): {sum(p.numel() for p in other_params):,}")

    optimizer = AdamW(
        [
            {"params": ve_params, "lr": 1e-5},
            {"params": other_params, "lr": 2e-5},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Snapshot weights before step
    ve_before = (
        model.vision_encoder.dinov2.encoder.layer[-1]
        .attention.attention.query.weight.data[:3, :3]
        .clone()
    )
    adapter_before = {k: v.clone() for k, v in model.adapter.state_dict().items()}

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
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"  Grad norm: {grad_norm:.4f}")
    optimizer.step()

    # Check VE weights changed
    ve_after = model.vision_encoder.dinov2.encoder.layer[-1].attention.attention.query.weight.data[
        :3, :3
    ]
    ve_changed = not torch.equal(ve_before, ve_after)
    assert ve_changed, "VE weights didn't change after optimizer step!"
    print("  ✓ VisionEncoder weights updated")

    # Check adapter weights changed
    adapter_after = model.adapter.state_dict()
    adapter_changed = any(
        not torch.equal(adapter_before[k], adapter_after[k]) for k in adapter_before
    )
    assert adapter_changed, "Adapter weights didn't change after optimizer step!"
    print("  ✓ Adapter weights updated")

    print("  ✓ Parameter groups + optimizer step passed")
    return output.loss.item()


def test_neftune(model, dataloader, device):
    """T2-N: NEFTune noise injection works correctly."""
    print("\n=== T2-N: NEFTune Noise Injection ===")
    model.set_stage2()
    model = model.to(device)
    model.train()

    batch = next(iter(dataloader))
    pv = batch["pixel_values"].to(device)
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)

    # Run twice — with noise the embeddings should differ
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out1 = model._build_inputs(pv, ids, mask)
        out2 = model._build_inputs(pv, ids, mask)

    diff = (out1["inputs_embeds"] - out2["inputs_embeds"]).abs().sum().item()
    print(f"  Embedding diff between two forward passes: {diff:.4f}")
    assert diff > 0, "NEFTune noise should produce different embeddings each call!"

    # Verify noise is disabled in eval mode
    model.eval()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out3 = model._build_inputs(pv, ids, mask)
        out4 = model._build_inputs(pv, ids, mask)

    diff_eval = (out3["inputs_embeds"] - out4["inputs_embeds"]).abs().sum().item()
    print(f"  Embedding diff in eval mode: {diff_eval:.4f}")
    assert diff_eval == 0, "NEFTune should NOT add noise in eval mode!"

    model.train()
    print("  ✓ NEFTune test passed")


def test_overfit_mini(model, dataloader, device, n_steps: int = 10):
    """T2-6: Can overfit on tiny data (loss should decrease)."""
    print(f"\n=== T2-6: Overfit Test ({n_steps} steps) ===")
    model.set_stage2()
    model = model.to(device)
    model.llm.gradient_checkpointing_enable()
    model.train()

    ve_params = [p for p in model.vision_encoder.parameters() if p.requires_grad]
    other_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("vision_encoder.")
    ]

    optimizer = AdamW(
        [
            {"params": ve_params, "lr": 1e-5},
            {"params": other_params, "lr": 2e-5},
        ],
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
        if step % 2 == 0 or step == n_steps - 1:
            print(f"  Step {step:3d}: loss={loss_val:.4f}")

    first_loss = losses[0]
    last_loss = losses[-1]
    print(f"\n  Loss: {first_loss:.4f} → {last_loss:.4f}")
    if last_loss < first_loss:
        print("  ✓ Loss decreased — model is learning!")
    else:
        print("  ⚠ Loss did NOT decrease — check configuration")

    assert not any(torch.isnan(torch.tensor(v)) for v in losses), "NaN loss during overfit test!"


def main():
    print("=" * 60)
    print("  Stage 2 Pipeline Validation (Fail-Fast)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Pre-checks
    json_path = JSON_PATH if JSON_PATH.exists() else JSON_PATH_150K
    assert json_path.exists(), f"Dataset JSON not found: {JSON_PATH} or {JSON_PATH_150K}"
    assert IMAGE_DIR.exists(), f"COCO train2014 dir not found: {IMAGE_DIR}"
    assert STAGE1_CKPT.exists(), f"Stage 1 checkpoint not found: {STAGE1_CKPT}"

    # Create mini dataset
    mini_json = create_mini_json(n_samples=8)

    # Load model with NEFTune
    print("\nLoading model (NEFTune alpha=5.0)...")
    model = QwenVLMini(neftune_alpha=5.0)

    # Run tests sequentially
    test_stage1_checkpoint_loading(model)
    dataset = test_dataset_loading(model, mini_json)
    dataloader = test_collator(model, dataset)
    test_neftune(model, dataloader, device)
    test_forward_backward_stage2(model, dataloader, device)
    test_param_groups_optimizer(model, dataloader, device)
    test_overfit_mini(model, dataloader, device, n_steps=10)

    # Cleanup
    mini_json.unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("  ALL STAGE 2.1 PIPELINE TESTS PASSED")
    print("=" * 60)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak VRAM: {peak:.0f} MB")


if __name__ == "__main__":
    main()
