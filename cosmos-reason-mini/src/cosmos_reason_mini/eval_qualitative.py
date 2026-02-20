"""定性的評価: eval 画像に対してテキスト生成し、GT と比較。

Usage:
    cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.eval_qualitative \
        --checkpoint checkpoints/sft/checkpoint-XXX.pt \
        --eval_json data/sft/qa_mini.json \
        --image_root data/nuscenes \
        --num_samples 10
"""

import argparse
import json
import os
import random

import torch
from PIL import Image
from torchvision import transforms

from cosmos_reason_mini.model_loader import load_vlm_from_checkpoint


def default_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def extract_question(conversations):
    """LLaVA conversations から質問テキストを抽出。"""
    for turn in conversations:
        if turn["from"] == "human":
            return (
                turn["value"]
                .replace("<image>\n", "")
                .replace("\n<image>", "")
                .replace("<image>", "")
            ).strip()
    return ""


def extract_answer(conversations):
    """LLaVA conversations から GT 回答を抽出。"""
    for turn in conversations:
        if turn["from"] == "gpt":
            return turn["value"]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--image_root", default="data/nuscenes")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output", default=None, help="Save results to JSON")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_vlm_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    # Load eval data
    with open(args.eval_json) as f:
        data = json.load(f)
    samples = random.sample(data, min(args.num_samples, len(data)))

    transform = default_transform()
    results = []

    for i, sample in enumerate(samples):
        question = extract_question(sample["conversations"])
        gt_answer = extract_answer(sample["conversations"])
        image_path = os.path.join(args.image_root, sample["image"])

        # Prepare inputs
        image = Image.open(image_path).convert("RGB")
        pixel_values = transform(image).unsqueeze(0).to(device)
        prompt = model.prepare_prompt(question)
        input_ids = prompt["input_ids"].to(device)
        attention_mask = prompt["attention_mask"].to(device)

        # Generate
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        # When generate() receives inputs_embeds, output contains only generated tokens
        generated = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        result = {
            "id": sample.get("id", f"sample_{i}"),
            "image": sample["image"],
            "question": question,
            "gt_answer": gt_answer,
            "generated": generated,
        }
        results.append(result)

        print(f"\n--- [{i + 1}/{len(samples)}] {result['id']} ---")
        print(f"Q: {question}")
        print(f"GT: {gt_answer}")
        print(f"Gen: {generated}")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
