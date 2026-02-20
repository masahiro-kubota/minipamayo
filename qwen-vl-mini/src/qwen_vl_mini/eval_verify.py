"""Verify evaluation scripts by running on a known HuggingFace VLM.

Supports:
  - SmolVLM-500M (default): ScienceQA-IMG published=80.0%
  - SmolVLM-256M: ScienceQA-IMG published=73.8%
  - LLaVA-1.5-7B: POPE Random=85.9%, ScienceQA-IMG=66.8%

Usage:
    # ScienceQA with SmolVLM-500M (default)
    uv run python -m qwen_vl_mini.eval_verify --scienceqa

    # POPE with SmolVLM-500M
    uv run python -m qwen_vl_mini.eval_verify --pope

    # Both benchmarks
    uv run python -m qwen_vl_mini.eval_verify --pope --scienceqa

    # LLaVA-1.5-7B
    uv run python -m qwen_vl_mini.eval_verify --model llava-hf/llava-1.5-7b-hf --pope
"""

import argparse
import string
from pathlib import Path

import torch
from PIL import Image

from qwen_vl_mini.eval_benchmark import (
    load_pope,
    parse_choice_letter,
    parse_yes_no,
)

# Published reference scores for verification
EXPECTED_SCORES = {
    "HuggingFaceTB/SmolVLM-500M-Instruct": {"scienceqa": 80.0},
    "HuggingFaceTB/SmolVLM-256M-Instruct": {"scienceqa": 73.8},
    "llava-hf/llava-1.5-7b-hf": {"pope_random": 85.9, "scienceqa": 66.8},
}


def load_model_hf(model_id: str, device: str):
    """Load a HuggingFace VLM. Auto-detects model type."""
    from transformers import AutoProcessor

    print(f"Loading model: {model_id}")

    from transformers import AutoModelForImageTextToText

    dtype = torch.bfloat16 if "smolvlm" in model_id.lower() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    print(f"Model loaded on {device}")
    return model, processor


def generate_answer(
    model, processor, image: Image.Image, prompt_text: str, device: str, max_new_tokens: int = 20
) -> str:
    """Generate answer using a HuggingFace VLM."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Decode only the generated tokens (skip input)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return processor.decode(generated_ids, skip_special_tokens=True)


def verify_pope(
    model,
    processor,
    pope_dir: str,
    image_dir: str,
    device: str,
    variants: list[str],
    pope_prompt: str,
) -> dict:
    """Run POPE evaluation on HuggingFace model."""
    results = {}
    for variant in variants:
        pope_path = str(Path(pope_dir) / f"coco_pope_{variant}.json")
        print(f"\n=== POPE {variant} ({pope_path}) ===")

        samples = load_pope(pope_path)
        tp = fp = tn = fn = 0
        yes_count = 0
        total = len(samples)

        for i, sample in enumerate(samples):
            image_path = str(Path(image_dir) / sample["image"])
            image = Image.open(image_path).convert("RGB")

            prompt_text = sample["text"] + "\n" + pope_prompt
            raw_answer = generate_answer(
                model, processor, image, prompt_text, device, max_new_tokens=20
            )
            pred = parse_yes_no(raw_answer)
            label = sample["label"].lower()

            if pred == "yes":
                yes_count += 1
            if pred == "yes" and label == "yes":
                tp += 1
            elif pred == "yes" and label == "no":
                fp += 1
            elif pred == "no" and label == "no":
                tn += 1
            elif pred == "no" and label == "yes":
                fn += 1

            if (i + 1) % 500 == 0:
                acc_so_far = (tp + tn) / (i + 1)
                print(f"  [{i + 1}/{total}] acc={acc_so_far:.1%}")

        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        yes_ratio = yes_count / total if total > 0 else 0

        result = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "yes_ratio": yes_ratio,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        results[variant] = result

        print(f"  Accuracy:  {accuracy:.1%}")
        print(f"  Precision: {precision:.1%}")
        print(f"  Recall:    {recall:.1%}")
        print(f"  F1:        {f1:.1%}")
        print(f"  Yes Ratio: {yes_ratio:.1%}")
        print(f"  (TP={tp} FP={fp} TN={tn} FN={fn})")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Variant':<15} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Yes%':>7}")
    print("-" * 60)
    for variant, result in results.items():
        print(
            f"{variant:<15} "
            f"{result['accuracy']:>6.1%} "
            f"{result['precision']:>6.1%} "
            f"{result['recall']:>6.1%} "
            f"{result['f1']:>6.1%} "
            f"{result['yes_ratio']:>6.1%}"
        )
    print("=" * 60)
    return results


def format_scienceqa_prompt_vlmevalkit(question: str, choices: list[str], hint: str | None) -> str:
    """Format ScienceQA prompt following VLMEvalKit convention (SmolVLM official)."""
    parts = []
    if hint:
        parts.append(f"Hint: {hint}")
    parts.append(f"Question: {question}")
    parts.append("Choices:")
    for i, choice in enumerate(choices):
        letter = string.ascii_uppercase[i]
        parts.append(f"{letter}. {choice}")
    parts.append("Answer with the letter.")
    return "\n".join(parts)


def verify_scienceqa(model, processor, device: str) -> dict:
    """Run ScienceQA-IMG evaluation on HuggingFace model."""
    from datasets import load_dataset

    dataset_name = "lmms-lab/ScienceQA-IMG"
    print("\n=== ScienceQA-IMG ===")
    print(f"Loading dataset: {dataset_name} (split=test)")
    ds = load_dataset(dataset_name, split="test")
    total = len(ds)
    print(f"  Total samples: {total}")

    correct = 0
    unparsed = 0

    for i, sample in enumerate(ds):
        image = sample["image"].convert("RGB")
        hint = sample.get("hint", "") or ""
        prompt_text = format_scienceqa_prompt_vlmevalkit(
            sample["question"], sample["choices"], hint if hint.strip() else None
        )

        raw_answer = generate_answer(
            model, processor, image, prompt_text, device, max_new_tokens=10
        )
        pred_letter = parse_choice_letter(raw_answer, len(sample["choices"]))
        gt_letter = string.ascii_uppercase[sample["answer"]]

        if pred_letter is None:
            unparsed += 1
        elif pred_letter == gt_letter:
            correct += 1

        if (i + 1) % 200 == 0:
            acc_so_far = correct / (i + 1)
            print(f"  [{i + 1}/{total}] acc={acc_so_far:.1%} (unparsed={unparsed})")

    accuracy = correct / total if total > 0 else 0
    result = {"accuracy": accuracy, "correct": correct, "total": total, "unparsed": unparsed}
    print(f"\n  Accuracy:  {accuracy:.1%}")
    print(f"  Correct:   {correct}/{total}")
    print(f"  Unparsed:  {unparsed}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Verify eval scripts with known HuggingFace VLM")
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument("--pope", action="store_true", help="Run POPE evaluation")
    parser.add_argument("--scienceqa", action="store_true", help="Run ScienceQA-IMG evaluation")
    parser.add_argument(
        "--pope-dir",
        type=str,
        default="data/pope",
        help="Directory containing POPE JSONL files",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/coco/val2014",
        help="Directory with COCO val2014 images",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["random"],
        help="POPE variants to evaluate (default: random only for speed)",
    )
    parser.add_argument(
        "--pope-prompt",
        type=str,
        default="Answer the question using a single word or phrase.",
        help="Prompt suffix for POPE questions",
    )
    args = parser.parse_args()

    # Default: run ScienceQA if neither specified
    if not args.pope and not args.scienceqa:
        args.scienceqa = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, processor = load_model_hf(args.model, device)

    # Show expected scores
    expected = EXPECTED_SCORES.get(args.model, {})
    if expected:
        print(f"\nExpected scores for {args.model}:")
        for key, val in expected.items():
            print(f"  {key}: ~{val}%")
    print()

    if args.pope:
        verify_pope(
            model, processor, args.pope_dir, args.image_dir, device, args.variants, args.pope_prompt
        )

    if args.scienceqa:
        verify_scienceqa(model, processor, device)


if __name__ == "__main__":
    main()
