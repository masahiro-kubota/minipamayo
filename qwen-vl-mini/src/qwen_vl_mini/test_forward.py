"""Phase 1 動作確認: forward + generate + VRAM."""

import torch

from qwen_vl_mini.model import QwenVLMini


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # TF32 有効化 (design.md §10.15)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.cuda.reset_peak_memory_stats()

    print("Loading model...")
    model = QwenVLMini()
    model = model.to(device)
    model.set_stage1()  # VE + LLM frozen, Adapter only
    print("Model loaded.")

    # Trainable params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params (Stage 1): {trainable:,}")

    # --- Test 1: forward pass ---
    print("\n=== Test 1: forward pass ===")
    pixel_values = torch.randn(1, 3, 224, 224, device=device)
    prompt = model.prepare_prompt("Describe this image.")
    input_ids = prompt["input_ids"].to(device)
    attention_mask = prompt["attention_mask"].to(device)

    # Build labels: all -100 (no loss) for prompt, just testing forward
    labels = torch.full_like(input_ids, QwenVLMini.IGNORE_INDEX)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(pixel_values, input_ids, attention_mask, labels=labels)

    print(f"Loss: {output.loss}")
    print(f"Logits shape: {output.logits.shape}")
    assert output.logits is not None, "Logits should not be None"
    print("forward pass OK")

    # --- Test 2: forward with actual labels ---
    print("\n=== Test 2: forward with loss ===")
    # Simulate: last 5 tokens of input are the "answer"
    labels2 = torch.full_like(input_ids, QwenVLMini.IGNORE_INDEX)
    labels2[:, -5:] = input_ids[:, -5:]  # Use actual tokens as labels for last 5

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output2 = model(pixel_values, input_ids, attention_mask, labels=labels2)

    print(f"Loss: {output2.loss.item():.4f}")
    assert not torch.isnan(output2.loss), "Loss should not be NaN"
    assert output2.loss.item() > 0, "Loss should be positive"
    print("forward with loss OK")

    # --- Test 3: generate ---
    print("\n=== Test 3: generate ===")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        generated_ids = model.generate(
            pixel_values,
            input_ids,
            attention_mask,
            max_new_tokens=30,
            do_sample=False,
        )

    generated_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated ({generated_ids.shape[1]} tokens): {generated_text[:200]}")
    assert generated_ids.shape[1] > input_ids.shape[1], "Should generate new tokens"
    print("generate OK")

    # --- Test 4: VRAM ---
    print("\n=== Test 4: VRAM ===")
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak VRAM: {peak_mb:.0f} MB ({peak_mb / 1024:.2f} GB)")
    assert peak_mb < 24_000, "Should fit in RTX 4090 (24 GB)"
    print("VRAM OK")

    print("\n=== All Phase 1 tests passed ===")


if __name__ == "__main__":
    main()
