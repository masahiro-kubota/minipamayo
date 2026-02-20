"""Quantitative benchmark evaluation (POPE, ScienceQA-IMG)."""

import argparse
import json
import re
import string
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

    1. Take first line only (model may generate continuation)
    2. Take text before first period
    3. Remove commas, split by space
    4. If 'No', 'not', or 'no' in words -> 'no', else -> 'yes'
    """
    text = text.strip().split("\n")[0]
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
    pope_prompt: str = "Answer the question using a single word or phrase.",
) -> str:
    """Generate answer for a POPE question."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)

    prompt_text = question + "\n" + pope_prompt
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
    pope_prompt: str = "Answer the question using a single word or phrase.",
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
        raw_answer = generate_pope_answer(model, image_path, sample["text"], device, pope_prompt)
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


def parse_choice_letter(text: str, num_choices: int) -> str | None:
    """Parse choice letter (A/B/C/D/E) from model output.

    Returns the letter if found, None if unparseable.
    """
    text = text.strip().split("\n")[0].strip()
    valid_letters = list(string.ascii_uppercase[:num_choices])

    # Direct single letter
    if text in valid_letters:
        return text

    # Starts with letter followed by punctuation or space: "B.", "B)", "B "
    m = re.match(r"^([A-Z])[\.\)\s:,]", text)
    if m and m.group(1) in valid_letters:
        return m.group(1)

    # "(B)" pattern
    m = re.search(r"\(([A-Z])\)", text)
    if m and m.group(1) in valid_letters:
        return m.group(1)

    # First occurrence of a valid letter as a standalone word
    for word in text.split():
        cleaned = word.strip(".,;:!?()\"'")
        if cleaned in valid_letters:
            return cleaned

    return None


def format_scienceqa_prompt(question: str, choices: list[str], hint: str | None) -> str:
    """Format ScienceQA prompt following LLaVA-1.5 convention."""
    parts = []
    if hint:
        parts.append(f"Context: {hint}")
    parts.append(question)
    for i, choice in enumerate(choices):
        letter = string.ascii_uppercase[i]
        parts.append(f"({letter}) {choice}")
    parts.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(parts)


def evaluate_scienceqa(
    model,
    device: str,
    dataset_name: str = "lmms-lab/ScienceQA-IMG",
    split: str = "test",
) -> dict:
    """Evaluate ScienceQA-IMG benchmark.

    Returns dict with accuracy and per-category stats.
    """
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_name} (split={split})")
    ds = load_dataset(dataset_name, split=split)
    total = len(ds)
    print(f"  Total samples: {total}")

    correct = 0
    unparsed = 0

    for i, sample in enumerate(ds):
        image = sample["image"].convert("RGB")
        pixel_values = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)

        hint = sample.get("hint", "") or ""
        prompt_text = format_scienceqa_prompt(
            sample["question"], sample["choices"], hint if hint.strip() else None
        )
        prompt = model.prepare_prompt(prompt_text)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                pixel_values,
                prompt["input_ids"].to(device),
                prompt["attention_mask"].to(device),
                max_new_tokens=10,
                do_sample=False,
            )

        raw_answer = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
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
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "unparsed": unparsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation (POPE, ScienceQA-IMG)")
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
    parser.add_argument(
        "--pope-prompt",
        type=str,
        default="Answer the question using a single word or phrase.",
        help="Prompt suffix for POPE questions",
    )
    parser.add_argument(
        "--scienceqa",
        action="store_true",
        help="Run ScienceQA-IMG evaluation instead of POPE",
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

    if args.scienceqa:
        # ScienceQA-IMG evaluation
        print("=== ScienceQA-IMG ===")
        result = evaluate_scienceqa(model, device)
        print(f"\n  Accuracy:  {result['accuracy']:.1%}")
        print(f"  Correct:   {result['correct']}/{result['total']}")
        print(f"  Unparsed:  {result['unparsed']}")
    else:
        # POPE evaluation
        print(f'POPE prompt: "{args.pope_prompt}"\n')

        results = {}
        for variant in args.variants:
            pope_path = str(Path(args.pope_dir) / f"coco_pope_{variant}.json")
            print(f"=== POPE {variant} ({pope_path}) ===")

            result = evaluate_pope(model, pope_path, args.image_dir, device, args.pope_prompt)
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
