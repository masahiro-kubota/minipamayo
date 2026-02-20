"""Physical AI RL (GRPO) 学習スクリプト。

Multi-step GRPO: 各 iteration で rollout を先に全バッチ分実行し、
その後 μ 回の最適化ステップを行う。μ > 1 により PPO clipping が
有効に機能する (ratio != 1.0 になるため)。

Usage:
    cd cosmos-reason-mini && PYTHONUNBUFFERED=1 uv run python -m cosmos_reason_mini.train_rl \
        --sft_checkpoint checkpoints/sft-mini/checkpoint-24.pt \
        --mcq_train data/rl/mcq_mini.json \
        --image_root data/nuscenes \
        --output_dir checkpoints/rl-mini \
        --mu 4 \
        --no_wandb
"""

import argparse
import os
import random
from pathlib import Path

import torch

from cosmos_reason_mini.data.mcq_dataset import MCQDataset
from cosmos_reason_mini.grpo import (
    GRPOConfig,
    GRPOTrainer,
    compute_advantages,
    compute_format_reward,
    compute_reward,
)
from cosmos_reason_mini.model_loader import load_vlm_from_checkpoint

DEFAULTS = {
    "sft_checkpoint": "checkpoints/sft-mini/checkpoint-24.pt",
    "mcq_train": "data/rl/mcq_train.json",
    "mcq_eval": "data/rl/mcq_eval.json",
    "image_root": "data/nuscenes",
    "output_dir": "checkpoints/rl",
    "num_rollouts": 4,
    "lr": 4e-6,
    "kl_coeff": 0.005,
    "clip_epsilon": 0.2,
    "temperature": 0.7,
    "max_new_tokens": 256,
    "mu": 4,
    "iterations": 200,
    "eval_every": 50,
    "save_every": 50,
    "batch_size": 4,
    "logging_steps": 5,
    "no_wandb": False,
}


def save_checkpoint(model, iteration, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vision_encoder_state_dict": model.vision_encoder.state_dict(),
            "adapter_state_dict": model.adapter.state_dict(),
            "llm_state_dict": model.llm.state_dict(),
            "iteration": iteration,
        },
        save_path,
    )
    print(f"Saved checkpoint: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    for key, default in DEFAULTS.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{key}", action="store_true", default=default)
        else:
            parser.add_argument(f"--{key}", type=type(default), default=default)
    args = parser.parse_args()

    device = torch.device("cuda")
    use_wandb = not args.no_wandb

    # --- Policy Model (trainable) ---
    print(f"Loading policy model from {args.sft_checkpoint}...")
    policy = load_vlm_from_checkpoint(args.sft_checkpoint, device=device)

    # RL では LLM のみ trainable (VE + Adapter は frozen)
    for name, param in policy.named_parameters():
        if not name.startswith("llm."):
            param.requires_grad = False
    policy.train()

    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Trainable: {trainable_params / 1e6:.1f}M / {total_params / 1e6:.1f}M")

    # --- Reference Model (frozen) ---
    print("Loading reference model...")
    ref = load_vlm_from_checkpoint(args.sft_checkpoint, device=device)
    ref.eval()

    # --- GRPO ---
    config = GRPOConfig(
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        clip_epsilon=args.clip_epsilon,
        kl_coeff=args.kl_coeff,
        lr=args.lr,
        mu=args.mu,
    )
    trainer = GRPOTrainer(policy, ref, policy.tokenizer, config)
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=config.lr,
        betas=(0.9, 0.95),
    )

    # --- Dataset ---
    train_dataset = MCQDataset(args.mcq_train, args.image_root, shuffle_options=True)
    print(f"Train dataset: {len(train_dataset)} MCQ")

    # --- wandb ---
    if use_wandb:
        import wandb

        wandb.init(project="cosmos-reason-mini", name="phase3-rl-grpo", config=vars(args))

    # --- Training Loop ---
    print(f"\nStarting GRPO training: {args.iterations} iterations")
    print(f"  batch_size={args.batch_size}, K={args.num_rollouts}, mu={args.mu}")
    print(f"  lr={args.lr}, kl_coeff={args.kl_coeff}, clip_epsilon={args.clip_epsilon}")

    for iteration in range(args.iterations):
        # ランダムに batch_size 個の質問を選択
        indices = random.sample(range(len(train_dataset)), min(args.batch_size, len(train_dataset)))

        # === Phase 1: Rollout (全バッチ分を先に実行) ===
        batch_data = []
        iter_reward = 0.0
        iter_correct = 0
        iter_total = 0

        for idx in indices:
            sample = train_dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            correct_answer = sample["correct"]
            prompt = sample["prompt"]

            prompt_dict = policy.prepare_prompt(prompt)
            input_ids = prompt_dict["input_ids"].to(device)
            attention_mask = prompt_dict["attention_mask"].to(device)

            # Rollout: K 個の応答を生成 + old_log_probs を記録
            generated_texts, generated_ids_list, old_log_probs = trainer.rollout(
                pixel_values, input_ids, attention_mask
            )

            # 報酬計算
            rewards = []
            for text in generated_texts:
                r = compute_reward(text, correct_answer)
                r += compute_format_reward(text)
                rewards.append(r)
            rewards_tensor = torch.tensor(rewards, device=device)

            # 統計
            iter_reward += sum(rewards)
            iter_correct += sum(1 for r in rewards if r >= 1.0)
            iter_total += len(rewards)

            # Advantage 計算
            advantages = compute_advantages(rewards_tensor)

            # advantage が非ゼロの質問のみ保存
            if advantages.abs().sum() > 0:
                batch_data.append(
                    {
                        "pixel_values": pixel_values,
                        "input_ids": input_ids,
                        "generated_ids_list": generated_ids_list,
                        "advantages": advantages,
                        "old_log_probs": old_log_probs,
                    }
                )

        # === Phase 2: Multi-step 最適化 (μ 回) ===
        iter_kl = 0.0
        iter_loss = 0.0
        iter_ratio = 0.0
        mu_steps_done = 0

        if batch_data:
            for _mu_step in range(config.mu):
                optimizer.zero_grad()

                step_loss = 0.0
                step_kl = 0.0
                step_ratio = 0.0
                step_count = 0

                for data in batch_data:
                    loss, stats = trainer.compute_loss(
                        data["pixel_values"],
                        data["input_ids"],
                        data["generated_ids_list"],
                        data["advantages"],
                        data["old_log_probs"],
                    )
                    loss.backward()
                    step_loss += loss.item()
                    step_kl += stats.get("kl", 0)
                    step_ratio += stats.get("ratio", 1.0)
                    step_count += 1

                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()

                if step_count > 0:
                    iter_loss += step_loss / step_count
                    iter_kl += step_kl / step_count
                    iter_ratio += step_ratio / step_count
                    mu_steps_done += 1

        # Logging
        avg_reward = iter_reward / max(iter_total, 1)
        accuracy = iter_correct / max(iter_total, 1)
        avg_kl = iter_kl / max(mu_steps_done, 1)
        avg_loss = iter_loss / max(mu_steps_done, 1)
        avg_ratio = iter_ratio / max(mu_steps_done, 1)

        log_dict = {
            "iteration": iteration,
            "avg_reward": avg_reward,
            "accuracy": accuracy,
            "kl": avg_kl,
            "loss": avg_loss,
            "ratio": avg_ratio,
            "valid_questions": len(batch_data),
        }

        if use_wandb:
            import wandb

            wandb.log(log_dict)

        if iteration % args.logging_steps == 0:
            print(
                f"Iter {iteration}/{args.iterations}: "
                f"reward={avg_reward:.3f}, acc={accuracy * 100:.1f}%, "
                f"kl={avg_kl:.4f}, ratio={avg_ratio:.3f}, loss={avg_loss:.4f}, "
                f"valid_q={len(batch_data)}/{len(indices)}"
            )

        # Save
        if (iteration + 1) % args.save_every == 0:
            save_checkpoint(
                policy,
                iteration + 1,
                Path(args.output_dir) / f"checkpoint-{iteration + 1}.pt",
            )

        # Eval
        if (iteration + 1) % args.eval_every == 0 and os.path.exists(args.mcq_eval):
            from cosmos_reason_mini.eval_mcq import evaluate_mcq

            eval_dataset = MCQDataset(args.mcq_eval, args.image_root, shuffle_options=False)
            policy.eval()
            eval_results = evaluate_mcq(policy, eval_dataset, device, num_seeds=1)
            eval_acc = eval_results["avg_accuracy"]
            print(f"  Eval MCQ Accuracy: {eval_acc * 100:.1f}%")
            if use_wandb:
                wandb.log({"eval_mcq_accuracy": eval_acc})
            policy.train()

    # Final save
    save_checkpoint(
        policy,
        args.iterations,
        Path(args.output_dir) / "checkpoint-final.pt",
    )
    print(f"\nRL training complete. {args.iterations} iterations, mu={config.mu}.")

    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
