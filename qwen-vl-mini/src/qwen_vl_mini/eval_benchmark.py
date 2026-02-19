"""Quantitative benchmark evaluation (POPE)."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from qwen_vl_mini.eval_qualitative import load_model
from qwen_vl_mini.model import IMAGE_TRANSFORM


def load_pope(pope_path: str) -> list[dict]:
    """Load POPE JSONL file."""
    samples = []
    with open(pope_path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def parse_yes_no(text: str) -> str:
    """Parse Yes/No from model output (POPE official protocol).

    Following the official POPE evaluate.py:
    1. Take text before first period
    2. Remove commas, split by space
    3. If 'No', 'not', or 'no' in words -> 'no', else -> 'yes'
    """
    if text.find(".") != -1:
        text = text.split(".")[0]
    text = text.replace(",", "")
    words = text.split(" ")
    if "No" in words or "not" in words or "no" in words:
        return "no"
    return "yes"


def generate_pope_answer(
    model,
    image_path: str,
    question: str,
    device: str,
) -> str:
    """Generate answer for a POPE question."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)

    prompt_text = question + "\nAnswer the question using a single word or phrase."
    prompt = model.prepare_prompt(prompt_text)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            pixel_values,
            prompt["input_ids"].to(device),
            prompt["attention_mask"].to(device),
            max_new_tokens=20,
            do_sample=False,
        )

    return model.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def evaluate_pope(
    model,
    pope_path: str,
    image_dir: str,
    device: str,
) -> dict:
    """Evaluate POPE benchmark.

    Returns dict with accuracy, precision, recall, f1, yes_ratio.
    """
    samples = load_pope(pope_path)
    tp = fp = tn = fn = 0
    yes_count = 0
    total = len(samples)

    for i, sample in enumerate(samples):
        image_path = str(Path(image_dir) / sample["image"])
        raw_answer = generate_pope_answer(model, image_path, sample["text"], device)
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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="POPE benchmark evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pt file (omit for random init baseline)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=2,
        choices=[1, 2],
        help="Stage number (1: adapter only, 2: full model)",
    )
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
        default=["random", "popular", "adversarial"],
        help="POPE variants to evaluate",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.checkpoint is not None:
        print(f"Loading Stage {args.stage} checkpoint: {args.checkpoint}")
    else:
        print("Loading model with random init (no checkpoint)")
    model = load_model(args.checkpoint, args.stage, device)
    print(f"Model loaded. Device: {device}\n")

    results = {}
    for variant in args.variants:
        pope_path = str(Path(args.pope_dir) / f"coco_pope_{variant}.json")
        print(f"=== POPE {variant} ({pope_path}) ===")

        result = evaluate_pope(model, pope_path, args.image_dir, device)
        results[variant] = result

        print(f"  Accuracy:  {result['accuracy']:.1%}")
        print(f"  Precision: {result['precision']:.1%}")
        print(f"  Recall:    {result['recall']:.1%}")
        print(f"  F1:        {result['f1']:.1%}")
        print(f"  Yes Ratio: {result['yes_ratio']:.1%}")
        print(f"  (TP={result['tp']} FP={result['fp']} TN={result['tn']} FN={result['fn']})")
        print()

    # Summary table
    print("=" * 60)
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


if __name__ == "__main__":
    main()
