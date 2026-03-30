"""Canonical Stage 3 evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ....utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ....utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from ....stage1.vlm_ce.components import load_checkpoint
from ....stage1.vlm_ce.runtime import format_gib, set_seed
from ..common import CANONICAL_STAGE3_POLICY_OUTPUT_CONTRACT
from ..dataset import Stage3PostTrainingDataset, stage3_post_training_collate
from ..rewards import RewardWeights, aggregate_rewards, build_reasoning_reward_scorer
from ..rewards.consistency import score_consistency
from ..rewards.trajectory import score_trajectory
from ..rollout import generate_grouped_rollouts, load_stage3_rollout_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH_KEYS = {
    "checkpoint",
    "stage2_checkpoint",
    "stage1b_checkpoint",
    "eval_jsonl",
    "manifest_jsonl",
    "save_dir",
}


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
        default="minipamayo-qwen-3-5/artifacts/eval/stage3/post_training",
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


def _load_config_args(config_json: str, parser: argparse.ArgumentParser) -> tuple[str, dict, dict]:
    config_path, payload = load_json_payload(config_json)
    raw_config = payload.get("args") if isinstance(payload, dict) and "args" in payload else payload
    if not isinstance(raw_config, dict):
        raise RuntimeError("Config JSON must be an object or an object with an `args` object.")
    base_dir = resolve_path_base(
        config_path,
        payload,
        default_base="project_root",
        base_dirs={
            "project_root": PROJECT_ROOT,
            "config_dir": config_path.parent,
        },
    )
    config_args = normalize_arg_config(
        raw_config,
        parser,
        exclude_dests={"help", "config_json"},
        path_keys=CONFIG_PATH_KEYS,
        list_keys={"eval_jsonl"},
        base_dir=base_dir,
    )
    return str(config_path), payload, config_args


def parse_args() -> argparse.Namespace:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return build_parser().parse_args()

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-json", type=str, required=True)
    pre_args, remaining = pre_parser.parse_known_args()
    if remaining:
        raise RuntimeError(
            "Stage 3 evaluation accepts only --config-json. Put all settings in the JSON file."
        )

    parser = build_parser()
    config_path, config_payload, config_args = _load_config_args(pre_args.config_json, parser)
    parser.set_defaults(**config_args, config_json=config_path)
    args = parser.parse_args()
    args.config_json = config_path
    args.config_payload = config_payload
    args.config_args = config_args

    if not args.stage2_checkpoint:
        raise RuntimeError("`stage2_checkpoint` must be defined in the config JSON.")
    if not args.stage1b_checkpoint:
        raise RuntimeError("`stage1b_checkpoint` must be defined in the config JSON.")
    if not args.eval_jsonl:
        raise RuntimeError("`eval_jsonl` must be defined in the config JSON.")
    if args.num_rollouts <= 0:
        raise RuntimeError("`num_rollouts` must be > 0.")
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    return args


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sample_view(batch: dict) -> dict:
    sample = {
        "sample_id": batch["sample_id"][0],
        "image_path": batch["image_path"][0],
        "action": batch["action"][0].detach().cpu(),
        "v0": batch["v0"][0].detach().cpu(),
        "gt_waypoints": batch["gt_waypoints"][0].detach().cpu(),
        "dt": float(batch["dt"][0]),
        "ego_history_xyz": batch["ego_history_xyz"][0].detach().cpu(),
        "ego_history_rot": batch["ego_history_rot"][0].detach().cpu(),
        "ego_future_xyz": batch["ego_future_xyz"][0].detach().cpu(),
        "ego_future_rot": batch["ego_future_rot"][0].detach().cpu(),
        "reasoning_text": batch["reasoning_text"][0],
    }
    for key in ("command", "planner_state", "decision_longitudinal", "decision_lateral"):
        if key in batch:
            sample[key] = batch[key][0]
    return sample


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
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
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=stage3_post_training_collate,
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
        sample = _sample_view(batch)
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
        reward_rows = []
        for rollout in rollouts:
            reasoning_result = reasoning_scorer.score(sample, rollout.parsed.reasoning_text)
            consistency_result = score_consistency(
                reasoning_text=rollout.parsed.reasoning_text,
                pred_future_xyz=rollout.pred_future_xyz,
                v0=float(sample["v0"].item()),
                dt=float(sample["dt"]),
            )
            trajectory_result = score_trajectory(
                sample=sample,
                pred_future_xyz=rollout.pred_future_xyz,
                l2_weight=args.traj_l2_weight,
                jerk_weight=args.traj_jerk_weight,
            )
            reward_rows.append(
                aggregate_rewards(
                    reasoning=reasoning_result,
                    consistency=consistency_result,
                    trajectory=trajectory_result,
                    weights=reward_weights,
                )
            )

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
    _write_json(save_dir / "summary.json", summary)
    _write_json(save_dir / "samples.json", sample_rows)
    print(json.dumps(summary, ensure_ascii=False))
