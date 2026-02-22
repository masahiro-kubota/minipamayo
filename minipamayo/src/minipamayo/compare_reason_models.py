"""Compare gpt-4o vs gpt-4o-mini for reasoning quality scoring (r_reason).

Generates PRED CoC from Stage 3 model, then scores with both models.
Reports correlation, mean difference, and per-sample comparison.

Usage:
    cd minipamayo && uv run python -m minipamayo.compare_reason_models \
        --checkpoint checkpoints/stage3/best.pt \
        --coc_data data/coc_annotations_trainval.jsonl \
        --n_samples 50
"""

import argparse
import json
import random
import time

import torch
from transformers import AutoTokenizer

from .data.coc_dataset import CoCDataset, build_chat_token_ids
from .models.discrete_head import DiscreteActionTokenizer
from .models.minipamayo import MiniPamayo
from .rewards import ReasonReward


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage3/best.pt")
    parser.add_argument("--coc_data", type=str, default="data/coc_annotations_trainval.jsonl")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def greedy_generate(model, prompt_embeds, max_tokens, device):
    """Greedy autoregressive generation, returns token IDs."""
    input_embeds = prompt_embeds.clone()
    token_ids = []

    for _ in range(max_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.llm(inputs_embeds=input_embeds)
        logits = outputs.logits[:, -1, :]
        token_id = logits.argmax(dim=-1)
        token_ids.append(token_id.item())

        if token_id.item() in (151643, 151645):  # EOS or <|im_end|>
            break

        next_embed = model.llm.get_input_embeddings()(token_id.unsqueeze(0))
        input_embeds = torch.cat([input_embeds, next_embed.to(input_embeds.dtype)], dim=1)

    return token_ids


def extract_reasoning_text(token_ids, text_tokenizer, vocab_offset):
    """Extract reasoning text (non-action tokens) from generated sequence."""
    text_tokens = [t for t in token_ids if t < vocab_offset]
    return text_tokenizer.decode(text_tokens, skip_special_tokens=True)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)

    # Tokenizers
    action_tokenizer = DiscreteActionTokenizer(n_bins=args.n_bins)
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    vocab_offset = action_tokenizer.vocab_offset
    new_vocab = vocab_offset + args.n_bins

    # Load GT annotations
    gt_annotations = []
    with open(args.coc_data) as f:
        for line in f:
            gt_annotations.append(json.loads(line))

    # Dataset
    dataset = CoCDataset(
        annotations_path=args.coc_data,
        tokenizer=text_tokenizer,
        action_tokenizer=action_tokenizer,
        K=args.K,
    )
    print(f"Total samples: {len(dataset)}")

    # Sample indices
    n = min(args.n_samples, len(dataset))
    indices = sorted(random.sample(range(len(dataset)), n))
    print(f"Evaluating {n} samples")

    # Load model
    print("Loading Stage 3 model...")
    model = MiniPamayo(adapter_type="cross_attention", action_dim=args.K * 2)
    model.llm.resize_token_embeddings(new_vocab)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    # Step 1: Generate PRED CoC for all samples
    print(f"\n{'=' * 60}")
    print("Step 1: Generating PRED CoC traces...")
    print(f"{'=' * 60}")

    pred_data = []  # list of (image_path, gt_reasoning, pred_reasoning)

    with torch.no_grad():
        for count, i in enumerate(indices):
            sample = dataset[i]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            v0 = sample["v0"].item()

            # Build prompt
            chat_ids = build_chat_token_ids(text_tokenizer, v0)
            embed_layer = model.llm.get_input_embeddings()

            system_ids = torch.tensor(chat_ids["system_ids"], device=device)
            user_prefix_ids = torch.tensor(chat_ids["user_prefix_ids"], device=device)
            ego_question_ids = torch.tensor(chat_ids["ego_question_ids"], device=device)
            asst_prefix_ids = torch.tensor(chat_ids["asst_prefix_ids"], device=device)

            system_embeds = embed_layer(system_ids.unsqueeze(0))
            user_prefix_embeds = embed_layer(user_prefix_ids.unsqueeze(0))
            ego_question_embeds = embed_layer(ego_question_ids.unsqueeze(0))
            asst_prefix_embeds = embed_layer(asst_prefix_ids.unsqueeze(0))

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                patch_features = model.vision_encoder(pixel_values)
                visual_embeds = model.adapter(patch_features)

            target_dtype = system_embeds.dtype
            prompt_embeds = torch.cat(
                [
                    system_embeds,
                    user_prefix_embeds,
                    visual_embeds.to(target_dtype),
                    ego_question_embeds,
                    asst_prefix_embeds,
                ],
                dim=1,
            )

            token_ids = greedy_generate(model, prompt_embeds, 300, device)
            pred_text = extract_reasoning_text(token_ids, text_tokenizer, vocab_offset)

            # GT reasoning text
            from .data.coc_dataset import format_coc_text

            gt_text = format_coc_text(gt_annotations[i]["coc"])
            image_path = gt_annotations[i]["image_path"]

            pred_data.append((image_path, gt_text, pred_text))

            if count < 3:
                print(f"\n  --- Sample {i} ---")
                print(f"  GT:   {gt_text[:150]}...")
                print(f"  PRED: {pred_text[:150]}...")

            if (count + 1) % 10 == 0:
                print(f"  Generated {count + 1}/{n}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    print(f"\nGenerated {len(pred_data)} PRED traces. GPU memory freed.")

    # Step 2: Score with both models
    print(f"\n{'=' * 60}")
    print("Step 2: Scoring with gpt-4o and gpt-4o-mini...")
    print(f"{'=' * 60}")

    scorer_4o = ReasonReward(cache_dir="data/reason_cache_4o", model="gpt-4o")
    scorer_mini = ReasonReward(cache_dir="data/reason_cache_mini", model="gpt-4o-mini")

    scores_4o = []
    scores_mini = []

    for count, (image_path, gt_text, pred_text) in enumerate(pred_data):
        score_4o = scorer_4o.compute(image_path, gt_text, pred_text)
        score_mini = scorer_mini.compute(image_path, gt_text, pred_text)

        scores_4o.append(score_4o)
        scores_mini.append(score_mini)

        print(
            f"  [{count + 1:3d}/{n}] 4o={score_4o:.0f}  mini={score_mini:.0f}  diff={score_4o - score_mini:+.0f}"
        )

        # Rate limit
        time.sleep(0.1)

    # Step 3: Analysis
    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")

    import math

    mean_4o = sum(scores_4o) / n
    mean_mini = sum(scores_mini) / n
    mean_diff = sum(a - b for a, b in zip(scores_4o, scores_mini, strict=False)) / n
    abs_diff = sum(abs(a - b) for a, b in zip(scores_4o, scores_mini, strict=False)) / n

    # Exact agreement
    exact_match = sum(1 for a, b in zip(scores_4o, scores_mini, strict=False) if a == b) / n
    # Within ±1 agreement
    within_1 = sum(1 for a, b in zip(scores_4o, scores_mini, strict=False) if abs(a - b) <= 1) / n

    # Pearson correlation
    std_4o = math.sqrt(sum((s - mean_4o) ** 2 for s in scores_4o) / n)
    std_mini = math.sqrt(sum((s - mean_mini) ** 2 for s in scores_mini) / n)
    if std_4o > 0 and std_mini > 0:
        cov = (
            sum(
                (a - mean_4o) * (b - mean_mini)
                for a, b in zip(scores_4o, scores_mini, strict=False)
            )
            / n
        )
        pearson = cov / (std_4o * std_mini)
    else:
        pearson = float("nan")

    # Spearman rank correlation
    def rank(vals):
        indexed = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    ranks_4o = rank(scores_4o)
    ranks_mini = rank(scores_mini)
    n_r = len(ranks_4o)
    mean_r4o = sum(ranks_4o) / n_r
    mean_rmini = sum(ranks_mini) / n_r
    std_r4o = math.sqrt(sum((r - mean_r4o) ** 2 for r in ranks_4o) / n_r)
    std_rmini = math.sqrt(sum((r - mean_rmini) ** 2 for r in ranks_mini) / n_r)
    if std_r4o > 0 and std_rmini > 0:
        cov_r = (
            sum(
                (a - mean_r4o) * (b - mean_rmini)
                for a, b in zip(ranks_4o, ranks_mini, strict=False)
            )
            / n_r
        )
        spearman = cov_r / (std_r4o * std_rmini)
    else:
        spearman = float("nan")

    print("\n  Model Means:")
    print(f"    gpt-4o:      {mean_4o:.2f}")
    print(f"    gpt-4o-mini: {mean_mini:.2f}")

    print("\n  Agreement:")
    print(f"    Mean diff (4o - mini):   {mean_diff:+.2f}")
    print(f"    Mean |diff|:             {abs_diff:.2f}")
    print(f"    Exact match rate:        {exact_match:.1%}")
    print(f"    Within ±1 match rate:    {within_1:.1%}")

    print("\n  Correlation:")
    print(f"    Pearson:   {pearson:.3f}")
    print(f"    Spearman:  {spearman:.3f}")

    # Score distribution
    print("\n  Score Distribution:")
    for score in range(6):
        c4o = scores_4o.count(score)
        cmini = scores_mini.count(score)
        print(f"    Score {score}: 4o={c4o:3d} ({c4o / n:.0%})  mini={cmini:3d} ({cmini / n:.0%})")

    # Confusion-style: where do they disagree most?
    print("\n  Disagreement cases (|diff| >= 2):")
    for i, (s4o, smini) in enumerate(zip(scores_4o, scores_mini, strict=False)):
        if abs(s4o - smini) >= 2:
            idx = indices[i]
            print(f"    Sample {idx}: 4o={s4o:.0f} mini={smini:.0f} (diff={s4o - smini:+.0f})")
            print(f"      PRED: {pred_data[i][2][:100]}...")

    # Conclusion
    print(f"\n{'=' * 60}")
    print("Conclusion")
    print(f"{'=' * 60}")
    if pearson >= 0.8:
        print(f"  Pearson={pearson:.3f} >= 0.8: gpt-4o-mini is a viable replacement.")
        print("  Estimated cost savings: ~90% ($110 -> ~$11)")
    elif pearson >= 0.6:
        print(f"  Pearson={pearson:.3f}: moderate correlation. Consider with caution.")
    else:
        print(f"  Pearson={pearson:.3f} < 0.6: low correlation. Recommend using gpt-4o.")

    print("\nDone.")


if __name__ == "__main__":
    main()
