"""Canonical Stage 3 evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ...utils.artifact_paths import resolve_bundle_dir
from ...utils.checkpoint import load_checkpoint
from ...utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ...utils.train_runtime import format_gib, set_seed
from .cli import artifact_scope_for_config, parse_stage3_json_only_args, resolve_stage3_device
from .common import CANONICAL_STAGE3_POLICY_OUTPUT_CONTRACT
from .dataset import Stage3PostTrainingDataset, build_stage3_dataloader
from .bundle import load_stage3_rollout_bundle
from .rewards import RewardWeights, build_reasoning_reward_scorer
from .runtime import sample_view_from_batch, score_stage3_rollout, write_json
from .sampler import generate_grouped_rollouts

CONFIG_PATH_KEYS = {
    "checkpoint",
    "stage2_checkpoint",
    "stage1b_checkpoint",
    "eval_jsonl",
    "manifest_jsonl",
    "save_dir",
}
CONFIG_LIST_KEYS = {"eval_jsonl"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate canonical Stage 3 post-training.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--stage2-checkpoint", type=str, default="")
    parser.add_argument("--stage1b-checkpoint", type=str, default="")
    parser.add_argument("--eval-jsonl", type=str, default="")
    parser.add_argument("--manifest-jsonl", type=str, default="")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-gen-tokens", type=int, default=256)
    parser.add_argument("--reasoning-reward-mode", type=str, default="disabled")
    parser.add_argument("--reward-weight-reason", type=float, default=0.0)
    parser.add_argument("--reward-weight-consistency", type=float, default=0.5)
    parser.add_argument("--reward-weight-traj", type=float, default=0.5)
    parser.add_argument("--traj-l2-weight", type=float, default=1.0)
    parser.add_argument("--traj-jerk-weight", type=float, default=0.1)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--image-min-pixels", type=int, default=CANONICAL_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=CANONICAL_IMAGE_MAX_PIXELS)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_stage3_json_only_args(
        parser=parser,
        path_keys=CONFIG_PATH_KEYS,
        list_keys=CONFIG_LIST_KEYS,
        json_only_error="Stage 3 evaluation accepts only --config-json. Put all settings in the JSON file.",
    )
    if not args.stage2_checkpoint:
        raise RuntimeError("`stage2_checkpoint` must be defined in the config JSON.")
    if not args.stage1b_checkpoint:
        raise RuntimeError("`stage1b_checkpoint` must be defined in the config JSON.")
    if not args.eval_jsonl:
        raise RuntimeError("`eval_jsonl` must be defined in the config JSON.")
    args.save_dir = str(
        resolve_bundle_dir(
            args.save_dir,
            scope=artifact_scope_for_config(args.config_json, kind="eval"),
            run_name=Path(args.config_json).resolve().stem,
        )
    )
    if args.num_rollouts <= 0:
        raise RuntimeError("`num_rollouts` must be > 0.")
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_stage3_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This Stage 3 evaluator is intended to run on CUDA.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    dataset = Stage3PostTrainingDataset(
        args.eval_jsonl,
        manifest_jsonl=args.manifest_jsonl or None,
        max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError("Stage 3 eval dataset is empty.")
    dataloader = build_stage3_dataloader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )

    bundle = load_stage3_rollout_bundle(
        stage2_checkpoint_path=args.stage2_checkpoint,
        stage1b_checkpoint_path=args.stage1b_checkpoint,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        flow_steps=args.flow_steps,
        device=device,
    )
    if args.checkpoint:
        checkpoint = load_checkpoint(Path(args.checkpoint))
        if "model_state_dict" not in checkpoint:
            raise RuntimeError("Stage 3 eval checkpoint is missing canonical `model_state_dict`.")
        bundle.policy_model.load_state_dict(checkpoint["model_state_dict"])
    bundle.policy_model.eval()

    reasoning_scorer = build_reasoning_reward_scorer(args.reasoning_reward_mode)
    reward_weights = RewardWeights(
        reasoning=args.reward_weight_reason,
        consistency=args.reward_weight_consistency,
        trajectory=args.reward_weight_traj,
    )

    sample_rows: list[dict] = []
    total_reward = 0.0
    total_reason = 0.0
    total_consistency = 0.0
    total_traj = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_minade = 0.0
    total_minfde = 0.0
    total_valid = 0.0

    for batch in dataloader:
        sample = sample_view_from_batch(batch)
        rollouts = generate_grouped_rollouts(
            bundle=bundle,
            batch=batch,
            num_rollouts=args.num_rollouts,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_gen_tokens,
            requires_policy_grad=False,
        )
        reward_rows = [
            score_stage3_rollout(
                sample=sample,
                rollout=rollout,
                reasoning_scorer=reasoning_scorer,
                reward_weights=reward_weights,
                traj_l2_weight=args.traj_l2_weight,
                traj_jerk_weight=args.traj_jerk_weight,
            )
            for rollout in rollouts
        ]
        best_index = max(range(len(reward_rows)), key=lambda idx: reward_rows[idx].total_reward)
        best_rollout = rollouts[best_index]
        best_reward = reward_rows[best_index]
        minade = min(row.ade for row in reward_rows)
        minfde = min(row.fde for row in reward_rows)
        valid_rate = sum(1 for rollout in rollouts if rollout.parsed.valid) / len(rollouts)

        total_reward += best_reward.total_reward
        total_reason += best_reward.reasoning_reward
        total_consistency += best_reward.consistency_reward
        total_traj += best_reward.trajectory_reward
        total_ade += best_reward.ade
        total_fde += best_reward.fde
        total_minade += minade
        total_minfde += minfde
        total_valid += valid_rate

        sample_rows.append(
            {
                "sample_id": sample["sample_id"],
                "reward": best_reward.total_reward,
                "reason_reward": best_reward.reasoning_reward,
                "consistency_reward": best_reward.consistency_reward,
                "traj_reward": best_reward.trajectory_reward,
                "ade": best_reward.ade,
                "fde": best_reward.fde,
                "minade": minade,
                "minfde": minfde,
                "valid_rollout_rate": valid_rate,
                "reasoning_text": best_rollout.parsed.reasoning_text,
                "generated_action_tokens": best_rollout.parsed.generated_action_count,
                "rollout_valid": best_rollout.parsed.valid,
                "rollout_issues": list(best_rollout.parsed.issues),
            }
        )

    count = len(sample_rows)
    summary = {
        "num_samples": count,
        "reward": total_reward / max(count, 1),
        "reason_reward": total_reason / max(count, 1),
        "consistency_reward": total_consistency / max(count, 1),
        "traj_reward": total_traj / max(count, 1),
        "ade": total_ade / max(count, 1),
        "fde": total_fde / max(count, 1),
        "minade": total_minade / max(count, 1),
        "minfde": total_minfde / max(count, 1),
        "valid_rollout_rate": total_valid / max(count, 1),
        "policy_output_contract": CANONICAL_STAGE3_POLICY_OUTPUT_CONTRACT,
        "peak_reserved_gib": (
            format_gib(torch.cuda.max_memory_reserved(device)) if torch.cuda.is_available() else None
        ),
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
    }
    write_json(save_dir / "summary.json", summary)
    write_json(save_dir / "samples.json", sample_rows)
    print(json.dumps(summary, ensure_ascii=False))


__all__ = ["build_parser", "parse_args", "main"]


if __name__ == "__main__":
    main()
