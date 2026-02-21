"""Stage 4: GRPO (Group Relative Policy Optimization).

3-element reward: r_reason + r_consistency + r_traj (including r_collision).
LLM only trainable, everything else frozen.

Uses Qwen chat template matching Stage 3:
  <|im_start|>system\n{system_msg}<|im_end|>\n
  <|im_start|>user\n[visual_tokens] Speed: {v0} m/s. ...?<|im_end|>\n
  <|im_start|>assistant\n{generated tokens}<|im_end|>

Usage:
    cd minipamayo && uv run python -m minipamayo.train_stage4 \
        --stage3_checkpoint checkpoints/stage3/best.pt \
        --coc_data data/coc_annotations.jsonl
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .data.coc_dataset import CoCDataset, build_chat_token_ids
from .models.discrete_head import DiscreteActionTokenizer
from .models.minipamayo import MiniPamayo
from .rewards import ReasonReward, composite_reward


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo Stage 4 GRPO")
    parser.add_argument("--coc_data", type=str, default="data/coc_annotations.jsonl")
    parser.add_argument(
        "--stage3_checkpoint",
        type=str,
        default="checkpoints/stage3/best.pt",
    )
    parser.add_argument("--K_traj", type=int, default=64, help="Trajectory waypoints")
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--n_rollouts", type=int, default=4, help="Rollouts per sample")
    parser.add_argument("--mu", type=int, default=4, help="Multi-step updates per rollout")
    parser.add_argument("--eps", type=float, default=0.2, help="GRPO clipping coefficient")
    parser.add_argument("--beta", type=float, default=0.05, help="KL coefficient")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--max_gen_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="checkpoints/stage4")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit dataset size (0=all)")
    parser.add_argument("--max_text_len", type=int, default=2048)
    parser.add_argument(
        "--use_reason_reward",
        action="store_true",
        help="Enable r_reason via external LLM API (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--reason_model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for r_reason scoring",
    )
    return parser.parse_args()


def build_prompt_embeds(model, pixel_values, v0, text_tokenizer, device):
    """Build prompt embeddings using Qwen chat template.

    Returns the concatenated embeddings for:
      [system] [user_prefix] [visual_tokens] [ego_question]
      [asst_prefix]

    This forms the prompt prefix from which the model generates.
    """
    # Get chat template token IDs
    chat_ids = build_chat_token_ids(text_tokenizer, v0)
    embed_layer = model.llm.get_input_embeddings()

    # Encode each segment
    system_ids = torch.tensor(chat_ids["system_ids"], dtype=torch.long, device=device)
    user_prefix_ids = torch.tensor(chat_ids["user_prefix_ids"], dtype=torch.long, device=device)
    ego_question_ids = torch.tensor(chat_ids["ego_question_ids"], dtype=torch.long, device=device)
    asst_prefix_ids = torch.tensor(chat_ids["asst_prefix_ids"], dtype=torch.long, device=device)

    system_embeds = embed_layer(system_ids.unsqueeze(0))
    user_prefix_embeds = embed_layer(user_prefix_ids.unsqueeze(0))
    ego_question_embeds = embed_layer(ego_question_ids.unsqueeze(0))
    asst_prefix_embeds = embed_layer(asst_prefix_ids.unsqueeze(0))

    # Visual embeddings
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
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

    return prompt_embeds.detach()


@torch.no_grad()
def generate_rollout(model, prompt_embeds, max_tokens, temperature, device):
    """Generate a single rollout with temperature sampling.

    Returns:
        token_ids: list of generated token IDs
        log_probs: tensor of log probabilities for each token
    """
    input_embeds = prompt_embeds.clone()
    token_ids = []
    log_probs = []

    for _ in range(max_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.llm(inputs_embeds=input_embeds)
        logits = outputs.logits[:, -1, :] / temperature
        probs = torch.softmax(logits.float(), dim=-1)
        token_id = torch.multinomial(probs, 1).squeeze(-1)  # (1,)

        lp = torch.log(probs[0, token_id.item()] + 1e-10)
        token_ids.append(token_id.item())
        log_probs.append(lp.item())

        if token_id.item() == 151643:  # Qwen EOS
            break
        if token_id.item() == 151645:  # <|im_end|> — end of assistant turn
            break

        next_embed = model.llm.get_input_embeddings()(token_id.unsqueeze(0))
        input_embeds = torch.cat([input_embeds, next_embed.to(input_embeds.dtype)], dim=1)

    return token_ids, torch.tensor(log_probs, dtype=torch.float32, device=device)


def compute_sequence_log_prob(model, prompt_embeds, token_ids, device):
    """Compute log probability of a token sequence under current policy.

    Teacher-forced forward pass to get log prob of each token.
    """
    embed_layer = model.llm.get_input_embeddings()
    token_ids_t = torch.tensor(token_ids, dtype=torch.long, device=device)
    token_embeds = embed_layer(token_ids_t.unsqueeze(0))  # (1, T, 896)

    input_embeds = torch.cat(
        [prompt_embeds, token_embeds],
        dim=1,
    )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.llm(inputs_embeds=input_embeds)

    # Log probs at each generated position
    n_prefix = prompt_embeds.shape[1]
    logits = outputs.logits[0, n_prefix - 1 : -1, :]  # shifted
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    # Gather log probs for actual tokens
    sequence_log_probs = log_probs[torch.arange(len(token_ids), device=device), token_ids_t]

    return sequence_log_probs  # (T,)


def parse_action_from_rollout(token_ids, vocab_offset, n_bins, K, action_tokenizer):
    """Extract action tokens from generated sequence."""
    action_ids = [t for t in token_ids if vocab_offset <= t < vocab_offset + n_bins]
    if len(action_ids) >= K * 2:
        action_ids = action_ids[: K * 2]
    else:
        action_ids = action_ids + [vocab_offset] * (K * 2 - len(action_ids))
    return action_tokenizer.decode(action_ids)


def parse_decision_from_rollout(token_ids, text_tokenizer, vocab_offset):
    """Parse driving decision from generated tokens."""
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

    return decision


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"K_traj={args.K_traj}, n_rollouts={args.n_rollouts}, mu={args.mu}")

    # Tokenizers
    action_tokenizer = DiscreteActionTokenizer(n_bins=args.n_bins)
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    vocab_offset = action_tokenizer.vocab_offset

    # Load GT annotations
    gt_annotations = []
    with open(args.coc_data) as f:
        for line in f:
            gt_annotations.append(json.loads(line))
    print(f"Loaded {len(gt_annotations)} annotations")

    # Dataset (for visual inputs)
    dataset = CoCDataset(
        annotations_path=args.coc_data,
        tokenizer=text_tokenizer,
        action_tokenizer=action_tokenizer,
        K=args.K_traj,
        max_text_len=args.max_text_len,
    )

    # Policy model (Stage 3 checkpoint)
    print("Building policy model...")
    policy = MiniPamayo(adapter_type="cross_attention", action_dim=args.K_traj * 2)
    new_vocab = vocab_offset + args.n_bins
    policy.llm.resize_token_embeddings(new_vocab)

    stage3_path = Path(args.stage3_checkpoint)
    if stage3_path.exists():
        ckpt = torch.load(stage3_path, map_location="cpu", weights_only=True)
        policy.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded Stage 3: {stage3_path}")
    else:
        print(f"WARNING: Stage 3 checkpoint not found at {stage3_path}")

    # Reference policy (frozen copy of SFT model)
    print("Creating reference policy...")
    ref_policy = copy.deepcopy(policy)
    ref_policy.requires_grad_(False)
    ref_policy.eval()
    ref_policy = ref_policy.to(device)

    # Stage 4 gradient control: only LLM trainable
    policy.set_stage4()
    policy.llm.gradient_checkpointing_enable()
    policy = policy.to(device)

    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable:,} (LLM only)")

    optimizer = torch.optim.AdamW(
        [p for p in policy.llm.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # ReasonReward (optional, requires OPENAI_API_KEY)
    reason_reward_fn = None
    if args.use_reason_reward:
        reason_reward_fn = ReasonReward(
            cache_dir="data/reason_reward_cache",
            model=args.reason_model,
        )
        print(f"r_reason enabled (model={args.reason_model})")

    # Training
    print("\n=== Starting GRPO Training ===")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_reward = -float("inf")
    all_rewards_log = []

    for epoch in range(args.max_epochs):
        policy.train()
        epoch_rewards = []
        epoch_policy_loss = []
        epoch_kl = []

        n_samples = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)
        for i in range(n_samples):
            sample = dataset[i]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            v0 = sample["v0"]
            gt_waypoints = sample["gt_waypoints"]
            gt_decision = gt_annotations[i]["coc"]["driving_decision"]
            obstacles = gt_annotations[i].get("obstacles", [])

            # Build prompt embeddings with chat template + egomotion
            prompt_embeds = build_prompt_embeds(
                policy, pixel_values, v0.item(), text_tokenizer, device
            )

            # 1. Generate K rollouts
            rollouts = []
            for _ in range(args.n_rollouts):
                token_ids, old_lp = generate_rollout(
                    policy,
                    prompt_embeds,
                    args.max_gen_tokens,
                    args.temperature,
                    device,
                )
                rollouts.append({"token_ids": token_ids, "old_log_probs": old_lp})

            # 2. Compute rewards
            rewards = []
            for ro in rollouts:
                action = parse_action_from_rollout(
                    ro["token_ids"], vocab_offset, args.n_bins, args.K_traj, action_tokenizer
                )
                action_t = torch.tensor(action, dtype=torch.float32)
                pred_kv = action_t.reshape(args.K_traj, 2)
                pred_a = pred_kv[:, 0]
                pred_kappa = pred_kv[:, 1]

                # r_reason (optional)
                r_reason_score = None
                if reason_reward_fn is not None:
                    decision_text = parse_decision_from_rollout(
                        ro["token_ids"], text_tokenizer, vocab_offset
                    )
                    reasoning_text = text_tokenizer.decode(
                        [t for t in ro["token_ids"] if t < vocab_offset],
                        skip_special_tokens=True,
                    )
                    action_text = (
                        f"longitudinal: {decision_text['longitudinal']}, "
                        f"lateral: {decision_text['lateral']}"
                    )
                    r_reason_score = reason_reward_fn.compute(reasoning_text, action_text)

                r = composite_reward(
                    pred_a,
                    pred_kappa,
                    gt_waypoints,
                    v0,
                    gt_decision,
                    obstacles=obstacles,
                    r_reason=r_reason_score,
                )
                rewards.append(r)

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            epoch_rewards.append(rewards_t.mean().item())

            # 3. Advantage (group relative)
            if rewards_t.std() > 1e-8:
                advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
            else:
                advantages = torch.zeros_like(rewards_t)

            # 4. Multi-step policy update
            for _mu_step in range(args.mu):
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)

                for k, ro in enumerate(rollouts):
                    if len(ro["token_ids"]) == 0:
                        continue

                    # New log probs under current policy
                    new_lp = compute_sequence_log_prob(
                        policy, prompt_embeds, ro["token_ids"], device
                    )

                    # Old log probs (from rollout generation)
                    old_lp = ro["old_log_probs"][: len(new_lp)]

                    # Policy ratio
                    ratio = (new_lp - old_lp.detach()).exp()
                    seq_ratio = ratio.mean()

                    # Clipped surrogate
                    adv = advantages[k]
                    surr1 = seq_ratio * adv
                    surr2 = torch.clamp(seq_ratio, 1 - args.eps, 1 + args.eps) * adv
                    policy_loss = -torch.min(surr1, surr2)

                    # KL with reference policy
                    with torch.no_grad():
                        # Build ref prompt embeds
                        ref_prompt = build_prompt_embeds(
                            ref_policy, pixel_values, v0.item(), text_tokenizer, device
                        )
                        ref_lp = compute_sequence_log_prob(
                            ref_policy, ref_prompt, ro["token_ids"], device
                        )
                    kl = (new_lp - ref_lp).mean()

                    total_loss = total_loss + policy_loss + args.beta * kl

                total_loss = total_loss / args.n_rollouts
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.llm.parameters(), 1.0)
                optimizer.step()

                epoch_policy_loss.append(total_loss.item())
                epoch_kl.append(kl.item())

            global_step += 1

            if global_step % args.log_every == 0:
                avg_r = sum(epoch_rewards[-args.log_every :]) / args.log_every
                avg_pl = sum(epoch_policy_loss[-args.log_every * args.mu :]) / max(
                    len(epoch_policy_loss[-args.log_every * args.mu :]), 1
                )
                avg_kl = sum(epoch_kl[-args.log_every * args.mu :]) / max(
                    len(epoch_kl[-args.log_every * args.mu :]), 1
                )
                print(
                    f"[E{epoch + 1}] Step {global_step:3d} | "
                    f"Reward: {avg_r:.4f} | Policy Loss: {avg_pl:.4f} | KL: {avg_kl:.4f}"
                )

        # Epoch summary
        mean_reward = sum(epoch_rewards) / len(epoch_rewards)
        mean_pl = sum(epoch_policy_loss) / len(epoch_policy_loss)
        mean_kl = sum(epoch_kl) / len(epoch_kl)
        all_rewards_log.append(mean_reward)

        print(
            f"\n=== Epoch {epoch + 1}/{args.max_epochs} | "
            f"Mean Reward: {mean_reward:.4f} | "
            f"Policy Loss: {mean_pl:.4f} | "
            f"KL: {mean_kl:.4f} ===\n"
        )

        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": policy.state_dict(),
                    "mean_reward": mean_reward,
                    "K": args.K_traj,
                    "n_bins": args.n_bins,
                },
                save_dir / "best.pt",
            )

    torch.save(
        {
            "epoch": args.max_epochs,
            "global_step": global_step,
            "model_state_dict": policy.state_dict(),
            "mean_reward": mean_reward,
            "K": args.K_traj,
            "n_bins": args.n_bins,
            "reward_history": all_rewards_log,
        },
        save_dir / "final.pt",
    )

    # Summary
    print("\n=== Summary ===")
    print(f"  Reward: {all_rewards_log[0]:.4f} -> {all_rewards_log[-1]:.4f}")
    print(f"  Best reward: {best_reward:.4f}")

    if torch.cuda.is_available():
        print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\nDone.")


if __name__ == "__main__":
    main()
