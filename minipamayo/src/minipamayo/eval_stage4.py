"""Stage 4 evaluation: GRPO-trained model.

Compares Stage 3 (SFT) vs Stage 4 (RL) on:
  1. Composite reward (r_reason + r_consistency + r_traj including r_collision)
  2. Driving decision accuracy
  3. Trajectory metrics (ADE/FDE)
  4. Collision rate

Uses Qwen chat template matching Stage 3/4 training.

Usage:
    cd minipamayo && uv run python -m minipamayo.eval_stage4 \
        --checkpoint checkpoints/stage4/best.pt \
        --ref_checkpoint checkpoints/stage3/best.pt \
        --coc_data data/coc_annotations.jsonl
"""

import argparse
import json

import torch
from transformers import AutoTokenizer

from .data.coc_dataset import CoCDataset, build_chat_token_ids
from .models.discrete_head import DiscreteActionTokenizer
from .models.dynamics import forward_dynamics_batch
from .models.minipamayo import MiniPamayo
from .rewards import (
    collision_reward,
    composite_reward,
    consistency_reward,
    trajectory_reward,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo Stage 4 evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ref_checkpoint", type=str, default="checkpoints/stage3/best.pt")
    parser.add_argument("--coc_data", type=str, default="data/coc_annotations.jsonl")
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--max_text_len", type=int, default=2048)
    parser.add_argument("--show_samples", type=int, default=5)
    return parser.parse_args()


def build_eval_prompt_embeds(model, pixel_values, v0, text_tokenizer, device):
    """Build prompt embeddings using Qwen chat template for evaluation."""
    chat_ids = build_chat_token_ids(text_tokenizer, v0)
    embed_layer = model.llm.get_input_embeddings()

    system_ids = torch.tensor(chat_ids["system_ids"], dtype=torch.long, device=device)
    user_prefix_ids = torch.tensor(chat_ids["user_prefix_ids"], dtype=torch.long, device=device)
    ego_question_ids = torch.tensor(chat_ids["ego_question_ids"], dtype=torch.long, device=device)
    asst_prefix_ids = torch.tensor(chat_ids["asst_prefix_ids"], dtype=torch.long, device=device)

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

    return prompt_embeds


def greedy_generate(model, prompt_embeds, max_tokens, device, vocab_offset, n_bins):
    """Greedy autoregressive generation."""
    input_embeds = prompt_embeds.clone()
    token_ids = []

    for _ in range(max_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.llm(inputs_embeds=input_embeds)
        logits = outputs.logits[:, -1, :]
        token_id = logits.argmax(dim=-1)
        token_ids.append(token_id.item())

        if token_id.item() == 151643:  # EOS
            break
        if token_id.item() == 151645:  # <|im_end|>
            break

        next_embed = model.llm.get_input_embeddings()(token_id.unsqueeze(0))
        input_embeds = torch.cat([input_embeds, next_embed.to(input_embeds.dtype)], dim=1)

    return token_ids


def parse_from_tokens(token_ids, text_tokenizer, action_tokenizer, vocab_offset, n_bins, K):
    """Parse decision and action from generated tokens."""
    # Action
    action_ids = [t for t in token_ids if vocab_offset <= t < vocab_offset + n_bins]
    if len(action_ids) >= K * 2:
        action_ids = action_ids[: K * 2]
    else:
        action_ids = action_ids + [vocab_offset] * (K * 2 - len(action_ids))
    action = action_tokenizer.decode(action_ids)

    # Decision
    text_tokens = [t for t in token_ids if t < vocab_offset]
    text = text_tokenizer.decode(text_tokens, skip_special_tokens=True)

    decision = {}
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("longitudinal:"):
            decision["longitudinal"] = line.split(":", 1)[1].strip()
        elif line.startswith("lateral:"):
            decision["lateral"] = line.split(":", 1)[1].strip()

    if "longitudinal" not in decision:
        decision["longitudinal"] = "go_straight"
    if "lateral" not in decision:
        decision["lateral"] = "lane_keeping"

    return action, decision, text


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action_tokenizer = DiscreteActionTokenizer(n_bins=args.n_bins)
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    vocab_offset = action_tokenizer.vocab_offset
    new_vocab = vocab_offset + args.n_bins

    # GT annotations
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
        max_text_len=args.max_text_len,
    )
    print(f"Total: {len(dataset)}")

    # RL model
    print("Loading RL model...")
    rl_model = MiniPamayo(adapter_type="cross_attention", action_dim=args.K * 2)
    rl_model.llm.resize_token_embeddings(new_vocab)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    rl_model.load_state_dict(ckpt["model_state_dict"])
    rl_model = rl_model.to(device)
    rl_model.eval()
    print(f"Loaded: {args.checkpoint}")

    # Reference SFT model
    ref_model = None
    ref_path = args.ref_checkpoint
    if ref_path and torch.load(ref_path, map_location="cpu", weights_only=True):
        print("Loading reference SFT model...")
        ref_model = MiniPamayo(adapter_type="cross_attention", action_dim=args.K * 2)
        ref_model.llm.resize_token_embeddings(new_vocab)
        ref_ckpt = torch.load(ref_path, map_location="cpu", weights_only=True)
        ref_model.load_state_dict(ref_ckpt["model_state_dict"])
        ref_model = ref_model.to(device)
        ref_model.eval()

    # Evaluate both models
    results = {"rl": {}, "sft": {}}

    for model_name, model in [("rl", rl_model), ("sft", ref_model)]:
        if model is None:
            continue

        print(f"\n{'=' * 70}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'=' * 70}")

        all_rewards = []
        all_r_consistency = []
        all_r_traj = []
        all_r_collision = []
        all_actions = []
        all_v0 = []
        all_gt_wp = []
        decision_correct = {"longitudinal": 0, "lateral": 0}
        decision_total = 0

        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
                v0 = sample["v0"]
                gt_waypoints = sample["gt_waypoints"]
                gt_decision = gt_annotations[i]["coc"]["driving_decision"]
                obstacles = gt_annotations[i].get("obstacles", [])

                # Build prompt with chat template + egomotion
                prompt_embeds = build_eval_prompt_embeds(
                    model, pixel_values, v0.item(), text_tokenizer, device
                )

                token_ids = greedy_generate(
                    model, prompt_embeds, 300, device, vocab_offset, args.n_bins
                )

                action, decision, _text = parse_from_tokens(
                    token_ids, text_tokenizer, action_tokenizer, vocab_offset, args.n_bins, args.K
                )

                action_t = torch.tensor(action, dtype=torch.float32)
                pred_kv = action_t.reshape(args.K, 2)
                pred_a = pred_kv[:, 0]
                pred_kappa = pred_kv[:, 1]

                # Rewards (with obstacles)
                r_c = consistency_reward(pred_a, pred_kappa, v0, gt_decision)
                r_t = trajectory_reward(pred_a, pred_kappa, gt_waypoints, v0, obstacles=obstacles)
                r_total = composite_reward(
                    pred_a,
                    pred_kappa,
                    gt_waypoints,
                    v0,
                    gt_decision,
                    obstacles=obstacles,
                )

                # Collision rate (separate metric)
                pred_wp = forward_dynamics_batch(
                    pred_a.unsqueeze(0), pred_kappa.unsqueeze(0), v0.unsqueeze(0)
                ).squeeze(0)
                r_col = collision_reward(pred_wp, obstacles)

                all_rewards.append(r_total)
                all_r_consistency.append(r_c)
                all_r_traj.append(r_t)
                all_r_collision.append(r_col)
                all_actions.append(action_t)
                all_v0.append(v0)
                all_gt_wp.append(gt_waypoints)

                decision_total += 1
                if decision.get("longitudinal") == gt_decision["longitudinal"]:
                    decision_correct["longitudinal"] += 1
                if decision.get("lateral") == gt_decision["lateral"]:
                    decision_correct["lateral"] += 1

                if i < args.show_samples:
                    print(
                        f"  [{i}] R={r_total:.3f} (c={r_c:.1f}, t={r_t:.3f}, "
                        f"col={r_col:.2f}) dec={decision}"
                    )

        # Metrics
        mean_reward = sum(all_rewards) / len(all_rewards)
        mean_r_c = sum(all_r_consistency) / len(all_r_consistency)
        mean_r_t = sum(all_r_traj) / len(all_r_traj)
        mean_r_col = sum(all_r_collision) / len(all_r_collision)
        collision_rate = 1.0 - mean_r_col  # fraction of waypoints with collisions

        actions = torch.stack(all_actions)
        v0s = torch.stack(all_v0)
        gt_wp = torch.stack(all_gt_wp)

        ar_kv = actions.reshape(-1, args.K, 2)

        pred_wp = forward_dynamics_batch(ar_kv[:, :, 0], ar_kv[:, :, 1], v0s, dt=0.1)
        disp_errors = torch.norm(pred_wp - gt_wp, dim=2)
        ade = disp_errors.mean().item()
        fde = disp_errors[:, -1].mean().item()

        long_acc = decision_correct["longitudinal"] / max(decision_total, 1)
        lat_acc = decision_correct["lateral"] / max(decision_total, 1)

        results[model_name] = {
            "reward": mean_reward,
            "r_consistency": mean_r_c,
            "r_traj": mean_r_t,
            "r_collision": mean_r_col,
            "collision_rate": collision_rate,
            "ade": ade,
            "fde": fde,
            "long_acc": long_acc,
            "lat_acc": lat_acc,
        }

        print(f"\n  Composite Reward: {mean_reward:.4f}")
        print(f"  r_consistency:    {mean_r_c:.4f}")
        print(f"  r_traj:           {mean_r_t:.4f}")
        print(f"  r_collision:      {mean_r_col:.4f}")
        print(f"  Collision Rate:   {collision_rate:.4f}")
        print(f"  ADE:              {ade:.4f} m")
        print(f"  FDE:              {fde:.4f} m")
        print(f"  Longitudinal Acc: {long_acc:.4f}")
        print(f"  Lateral Acc:      {lat_acc:.4f}")

    # Comparison
    if "rl" in results and "sft" in results:
        print(f"\n{'=' * 70}")
        print("SFT vs RL Comparison")
        print(f"{'=' * 70}")
        lower_is_better = {"ade", "fde", "collision_rate"}
        for metric in [
            "reward",
            "r_consistency",
            "r_traj",
            "r_collision",
            "collision_rate",
            "ade",
            "fde",
            "long_acc",
            "lat_acc",
        ]:
            sft_val = results["sft"][metric]
            rl_val = results["rl"][metric]
            diff = rl_val - sft_val
            better = (
                "+"
                if (diff > 0 and metric not in lower_is_better)
                or (diff < 0 and metric in lower_is_better)
                else "-"
            )
            print(f"  {metric:20s}: SFT={sft_val:.4f} -> RL={rl_val:.4f} ({better}{abs(diff):.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
