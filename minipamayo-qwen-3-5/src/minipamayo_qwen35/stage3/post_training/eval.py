"""Canonical Stage 3 evaluation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ...inspector.manifests import upsert_manifest
from ...utils.checkpoint import load_checkpoint
from ...utils.eval_reporting import (
    EvalReporter,
    add_eval_reporting_args,
    apply_eval_reporting_artifact_policy,
    reporting_path_keys,
    validate_eval_reporting_args,
)
from ...utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ...utils.run_metadata import collect_dataset_view_fingerprint, collect_processor_settings
from ...utils.train_runtime import format_gib, set_seed
from .cli import artifact_scope_for_config, parse_stage3_json_only_args, require_stage3_cuda_device
from .common import CANONICAL_STAGE3_POLICY_OUTPUT_CONTRACT
from .dataset import Stage3PostTrainingDataset, build_stage3_dataloader
from .bundle import load_stage3_rollout_bundle
from .rewards import RewardWeights, build_reasoning_reward_scorer
from .runtime import sample_view_from_batch, score_stage3_rollout
from .sampler import generate_grouped_rollouts

CONFIG_PATH_KEYS = {
    "checkpoint",
    "stage2_checkpoint",
    "stage1b_checkpoint",
    "eval_jsonl",
    "manifest_jsonl",
    "output_json",
} | reporting_path_keys(include_per_sample_jsonl=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate canonical Stage 3 post-training.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--stage2-checkpoint", type=str, default="")
    parser.add_argument("--stage1b-checkpoint", type=str, default="")
    parser.add_argument("--eval-jsonl", type=str, default="")
    parser.add_argument("--manifest-jsonl", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
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
    add_eval_reporting_args(parser, include_per_sample_jsonl=True)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_stage3_json_only_args(
        parser=parser,
        path_keys=CONFIG_PATH_KEYS,
        json_only_error="Stage 3 evaluation accepts only --config-json. Put all settings in the JSON file.",
    )
    if not args.stage2_checkpoint:
        raise RuntimeError("`stage2_checkpoint` must be defined in the config JSON.")
    if not args.eval_jsonl:
        raise RuntimeError("`eval_jsonl` must be defined in the config JSON.")
    if not args.stage1b_checkpoint:
        raise RuntimeError("`stage1b_checkpoint` must be defined in the config JSON.")
    apply_eval_reporting_artifact_policy(
        args,
        scope=artifact_scope_for_config(args.config_json, kind="eval"),
        include_per_sample_jsonl=True,
    )
    if args.num_rollouts <= 0:
        raise RuntimeError("`num_rollouts` must be > 0.")
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    validate_eval_reporting_args(args, require_per_sample_jsonl=True)
    return args


def _tensor_xy_rows(tensor) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in tensor[..., :2].tolist()]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = require_stage3_cuda_device(
        device_name=args.device,
        git_cwd=Path(__file__).resolve().parent,
        error_message="This Stage 3 evaluator is intended to run on CUDA.",
    )

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
    effective_checkpoint = args.checkpoint or args.stage2_checkpoint

    reasoning_scorer = build_reasoning_reward_scorer(args.reasoning_reward_mode)
    reward_weights = RewardWeights(
        reasoning=args.reward_weight_reason,
        consistency=args.reward_weight_consistency,
        trajectory=args.reward_weight_traj,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    reporter = EvalReporter.from_args(
        args=args,
        stage="stage3_eval",
        total_samples=len(dataset),
        checkpoint=effective_checkpoint,
        dataset_path=args.eval_jsonl,
        extra_wandb_config={
            "entrypoint": "stage3.post_training.eval",
            "num_rollouts": int(args.num_rollouts),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "max_gen_tokens": int(args.max_gen_tokens),
            "max_samples": int(args.max_samples),
        },
    )
    reporter.emit_setup(
        "stage3_eval_setup",
        {
            "checkpoint": effective_checkpoint,
            "base_stage2_checkpoint": args.stage2_checkpoint,
            "base_stage1b_checkpoint": args.stage1b_checkpoint,
            "eval_jsonl": args.eval_jsonl,
            "eval_size": len(dataset),
            "num_rollouts": args.num_rollouts,
            "image_min_pixels": args.image_min_pixels,
            "image_max_pixels": args.image_max_pixels,
        },
    )
    wandb_run_url = str(getattr(reporter.wandb_run, "url", ""))

    total_reward = 0.0
    total_reason = 0.0
    total_consistency = 0.0
    total_traj = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_minade = 0.0
    total_minfde = 0.0
    total_valid = 0.0

    try:
        for sample_index, batch in enumerate(dataloader):
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

            reporter.emit_sample(
                {
                    "event": "sample",
                    "sample_index": sample_index,
                    "sample_id": str(sample["sample_id"]),
                    "image_path": str(sample["image_path"]),
                    "command": str(sample.get("command", "")),
                    "planner_state": str(sample.get("planner_state", "")),
                    "decision_longitudinal": str(sample.get("decision_longitudinal", "")),
                    "decision_lateral": str(sample.get("decision_lateral", "")),
                    "gt_waypoints": _tensor_xy_rows(sample["gt_waypoints"]),
                    "pred_waypoints": _tensor_xy_rows(best_rollout.pred_future_xyz),
                    "reasoning_text": str(sample["reasoning_text"]),
                    "reasoning_text_pred": str(best_rollout.parsed.reasoning_text),
                    "ade_m": float(best_reward.ade),
                    "fde_m": float(best_reward.fde),
                    "minade_m": float(minade),
                    "minfde_m": float(minfde),
                    "valid_rollout_rate": float(valid_rate),
                    "generated_action_tokens": int(best_rollout.parsed.generated_action_count),
                    "rollout_valid": bool(best_rollout.parsed.valid),
                    "rollout_issues": list(best_rollout.parsed.issues),
                    "decision_pred": dict(best_rollout.parsed.decision or {}),
                    "metrics": {
                        "reward": float(best_reward.total_reward),
                        "reason_reward": float(best_reward.reasoning_reward),
                        "consistency_reward": float(best_reward.consistency_reward),
                        "traj_reward": float(best_reward.trajectory_reward),
                        "ade_m": float(best_reward.ade),
                        "fde_m": float(best_reward.fde),
                        "minade_m": float(minade),
                        "minfde_m": float(minfde),
                        "valid_rollout_rate": float(valid_rate),
                    },
                },
                print_to_stdout=sample_index < 5,
            )
            processed_samples = sample_index + 1
            reporter.emit_progress(
                processed_samples=processed_samples,
                running_metrics={
                    "reward": total_reward / processed_samples,
                    "reason_reward": total_reason / processed_samples,
                    "consistency_reward": total_consistency / processed_samples,
                    "traj_reward": total_traj / processed_samples,
                    "ade_m": total_ade / processed_samples,
                    "fde_m": total_fde / processed_samples,
                    "minade_m": total_minade / processed_samples,
                    "minfde_m": total_minfde / processed_samples,
                    "valid_rollout_rate": total_valid / processed_samples,
                },
            )

        count = len(dataset)
        summary = {
            "config_json": args.config_json,
            "config_payload": args.config_payload,
            "config_args": args.config_args,
            "run_args": vars(args),
            "checkpoint": effective_checkpoint,
            "base_stage2_checkpoint": args.stage2_checkpoint,
            "base_stage1b_checkpoint": args.stage1b_checkpoint,
            "stage3_checkpoint": args.checkpoint or None,
            "eval_jsonl": args.eval_jsonl,
            "num_samples": count,
            "eval_size": count,
            "metrics": {
                "reward": total_reward / max(count, 1),
                "reason_reward": total_reason / max(count, 1),
                "consistency_reward": total_consistency / max(count, 1),
                "traj_reward": total_traj / max(count, 1),
                "ade_m": total_ade / max(count, 1),
                "fde_m": total_fde / max(count, 1),
                "minade_m": total_minade / max(count, 1),
                "minfde_m": total_minfde / max(count, 1),
                "valid_rollout_rate": total_valid / max(count, 1),
                "peak_reserved_gib": (
                    format_gib(torch.cuda.max_memory_reserved(device))
                    if torch.cuda.is_available()
                    else None
                ),
                "elapsed_seconds": round(time.perf_counter() - start_time, 3),
            },
            "dataset_fingerprint": collect_dataset_view_fingerprint(dataset),
            "processor_settings": collect_processor_settings(
                bundle.processor,
                requested_min_pixels=args.image_min_pixels or None,
                requested_max_pixels=args.image_max_pixels or None,
            ),
            "stage2_metadata": bundle.stage2_metadata,
            "policy_output_contract": CANONICAL_STAGE3_POLICY_OUTPUT_CONTRACT,
        }
        reporter.emit_summary("stage3_eval_summary", summary)
        upsert_manifest(
            artifact_kind="eval",
            stage="stage3_eval",
            run_name=Path(args.output_json).resolve().stem,
            summary_json=args.output_json,
            checkpoint=effective_checkpoint,
            dataset_path=str(args.eval_jsonl),
            progress_json=str(args.progress_json),
            per_sample_jsonl=str(args.per_sample_jsonl),
            wandb_run_url=wandb_run_url,
        )
    except Exception as exc:
        reporter.emit_failure("stage3_eval_failure", exc)
        raise
    finally:
        reporter.close()


__all__ = ["build_parser", "parse_args", "main"]


if __name__ == "__main__":
    main()
