"""MCQ 評価スクリプト。

SFT モデルまたは RL モデルの MCQ 正解率を計測する。
Cosmos-Reason1 §6 のベンチマーク評価に対応。

Usage:
    cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.eval_mcq \
        --checkpoint checkpoints/sft-mini/checkpoint-24.pt \
        --mcq_json data/rl/mcq_mini.json \
        --image_root data/nuscenes
"""

import argparse
import json
import os
import random

import torch
from tqdm import tqdm

from cosmos_reason_mini.data.mcq_dataset import MCQDataset
from cosmos_reason_mini.grpo import extract_answer
from cosmos_reason_mini.model_loader import load_vlm_from_checkpoint


def evaluate_mcq(model, dataset, device, max_new_tokens=256, temperature=0.1, num_seeds=1) -> dict:
    """MCQ 正解率を計測。

    Args:
        model: QwenVLMini (eval mode)
        dataset: MCQDataset
        device: torch device
        max_new_tokens: 生成最大トークン数
        temperature: サンプリング温度 (0 でグリーディ)
        num_seeds: 評価回数 (選択肢シャッフルの多様性)

    Returns:
        dict with avg_accuracy and per_seed details
    """
    all_results = []

    for seed in range(num_seeds):
        random.seed(seed)
        torch.manual_seed(seed)
        correct = 0
        total = 0
        details = []

        for idx in tqdm(range(len(dataset)), desc=f"Seed {seed}"):
            sample = dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            prompt = sample["prompt"]
            correct_answer = sample["correct"]

            # Tokenize with chat template
            prompt_dict = model.prepare_prompt(prompt)
            input_ids = prompt_dict["input_ids"].to(device)
            attention_mask = prompt_dict["attention_mask"].to(device)

            # Generate
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )

            generated = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            extracted = extract_answer(generated)
            is_correct = extracted == correct_answer

            if is_correct:
                correct += 1
            total += 1

            details.append(
                {
                    "id": sample["id"],
                    "correct_answer": correct_answer,
                    "extracted": extracted,
                    "is_correct": is_correct,
                    "generated": generated[:200],
                }
            )

        accuracy = correct / total if total > 0 else 0
        all_results.append(
            {
                "seed": seed,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "details": details,
            }
        )
        print(f"  Seed {seed}: {correct}/{total} = {accuracy * 100:.1f}%")

    avg_accuracy = sum(r["accuracy"] for r in all_results) / len(all_results)
    return {
        "avg_accuracy": avg_accuracy,
        "per_seed": all_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mcq_json", default="data/rl/mcq_mini.json")
    parser.add_argument("--image_root", default="data/nuscenes")
    parser.add_argument("--output", default="results/mcq_eval.json")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Loading model from {args.checkpoint}...")
    model = load_vlm_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    dataset = MCQDataset(
        args.mcq_json,
        args.image_root,
        shuffle_options=args.num_seeds > 1,
    )
    print(f"MCQ dataset: {len(dataset)} samples, {args.num_seeds} seed(s)")

    results = evaluate_mcq(
        model,
        dataset,
        device,
        max_new_tokens=args.max_new_tokens,
        num_seeds=args.num_seeds,
    )
    print(f"\nMCQ Accuracy ({args.num_seeds}-seed avg): {results['avg_accuracy'] * 100:.1f}%")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
