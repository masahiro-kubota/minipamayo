"""Canonical Stage 1B expert-CFM evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ...inspector.manifests import upsert_manifest
from ...utils.eval_reporting import (
    EvalReporter,
    add_eval_reporting_args,
    reporting_path_keys,
    validate_eval_reporting_args,
)
from ..checkpoint_completion import require_completed_training_run
from ..dataset import Stage1JsonlDataset, stage1_collate
from .cli import (
    COMMON_CONFIG_PATH_KEYS,
    add_stage1b_common_args,
    parse_stage1b_json_only_args,
    require_stage1b_cuda_device,
    validate_stage1b_runtime_args,
)
from ...utils.json_config import normalize_required_string_list
from .metrics import compute_stage1b_action_mae_sums, compute_stage1b_waypoint_metrics
from .runtime import (
    load_stage1b_runtime,
    run_stage1b_inference_batch,
)

CONFIG_PATH_KEYS = COMMON_CONFIG_PATH_KEYS | {"eval_jsonl"} | reporting_path_keys(
    include_per_sample_jsonl=True
)
MULTI_VALUE_CONFIG_KEYS = {"eval_jsonl"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the canonical Stage 1B expert CFM path.")
    add_stage1b_common_args(parser)
    parser.add_argument("--eval-jsonl", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    add_eval_reporting_args(parser, include_per_sample_jsonl=True)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_stage1b_json_only_args(
        parser=parser,
        path_keys=CONFIG_PATH_KEYS,
        list_keys=MULTI_VALUE_CONFIG_KEYS,
        json_only_error="Stage 1B evaluation accepts only --config-json. Put all settings in the JSON file.",
    )
    args.eval_jsonl = normalize_required_string_list(args.eval_jsonl, key_name="eval_jsonl")
    validate_stage1b_runtime_args(args)
    validate_eval_reporting_args(args, require_per_sample_jsonl=True)
    return args


def main() -> None:
    args = parse_args()
    require_completed_training_run(
        args.stage1_checkpoint,
        checkpoint_label="Stage 1A checkpoint",
        required_summary_keys=["completed_epochs", "best_epoch", "stop_reason"],
        allowed_stop_reasons={"max_epochs", "early_stopping"},
    )
    require_completed_training_run(
        args.checkpoint,
        checkpoint_label="Stage 1B checkpoint",
        required_summary_keys=["best_epoch", "history_length"],
    )
    device = require_stage1b_cuda_device()

    dataset = Stage1JsonlDataset(args.eval_jsonl)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=stage1_collate,
        persistent_workers=args.num_workers > 0,
    )

    runtime = load_stage1b_runtime(
        stage1_checkpoint=args.stage1_checkpoint,
        checkpoint=args.checkpoint,
        flow_steps=args.flow_steps,
        device=device,
    )
    stage1b_metadata = runtime.stage1b_metadata
    reporter = EvalReporter.from_args(
        args=args,
        stage="stage1b_eval",
        total_samples=len(dataset),
        checkpoint=args.checkpoint,
        dataset_path=",".join(args.eval_jsonl),
        extra_wandb_config={
            "entrypoint": "stage1.expert_cfm.eval",
            "flow_steps": int(args.flow_steps),
            "include_pid_override": bool(args.include_pid_override),
            "batch_size": int(args.batch_size),
        },
    )
    reporter.emit_setup(
        "stage1b_eval_setup",
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
            "eval_jsonl": args.eval_jsonl,
            "num_samples": len(dataset),
            "batch_size": args.batch_size,
            "flow_steps": args.flow_steps,
            "condition_source": stage1b_metadata["condition_source"],
            "pid_override_enabled": bool(args.include_pid_override),
        },
    )
    wandb_run_url = str(getattr(reporter.wandb_run, "url", ""))

    total_loss = 0.0
    total_batches = 0
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0
    total_action_steps = 0
    canonical_total_mean_max_lateral = 0.0
    canonical_global_max_lateral = 0.0
    pid_total_ade = 0.0
    pid_total_fde = 0.0
    pid_total_mean_max_lateral = 0.0
    pid_global_max_lateral = 0.0
    pid_total_action_mae_accel = 0.0
    pid_total_action_mae_kappa = 0.0
    canonical_total_action_mae_accel = 0.0
    canonical_total_action_mae_kappa = 0.0

    try:
        with torch.no_grad():
            for batch in dataloader:
                outputs = run_stage1b_inference_batch(
                    runtime=runtime,
                    batch=batch,
                    include_pid_override=args.include_pid_override,
                    pid_target_speed_kmh=args.pid_target_speed_kmh,
                    pid_kp=args.pid_kp,
                    pid_ki=args.pid_ki,
                    pid_kd=args.pid_kd,
                    compute_loss=True,
                )
                if outputs.loss is None or outputs.gt_action is None or outputs.gt_action_seq is None:
                    raise RuntimeError("Stage 1B eval expected shared runtime outputs with loss and labels.")
                loss = outputs.loss
                gt_action = outputs.gt_action
                gt_action_seq = outputs.gt_action_seq
                pred_action = outputs.pred_action
                if outputs.gt_waypoints is None:
                    raise RuntimeError("Stage 1B eval expected ground-truth waypoints in the batch.")
                gt_waypoints = outputs.gt_waypoints
                pred_waypoints = outputs.pred_waypoints
                canonical_metrics = compute_stage1b_waypoint_metrics(
                    pred_waypoints=pred_waypoints,
                    gt_waypoints=gt_waypoints,
                )
                if outputs.pid_action is not None and outputs.pid_waypoints is not None:
                    pid_action = outputs.pid_action
                    pid_waypoints = outputs.pid_waypoints
                    pid_metrics = compute_stage1b_waypoint_metrics(
                        pred_waypoints=pid_waypoints,
                        gt_waypoints=gt_waypoints,
                    )
                else:
                    pid_action = None
                    pid_waypoints = None

                batch_size = gt_action.shape[0]
                action_horizon = int(gt_action_seq.shape[1])
                batch_loss_value = float(loss.detach().cpu())
                for row_idx in range(batch_size):
                    sample_index = total_samples + row_idx
                    sample_gt_action = gt_action_seq[row_idx].detach().cpu()
                    sample_pred_action = pred_action[row_idx].detach().cpu()
                    sample_gt_waypoints = gt_waypoints[row_idx].detach().cpu()
                    sample_pred_waypoints = pred_waypoints[row_idx].detach().cpu()
                    sample_action_mae_accel = float(
                        (sample_pred_action[:, 0] - sample_gt_action[:, 0]).abs().mean().item()
                    )
                    sample_action_mae_kappa = float(
                        (sample_pred_action[:, 1] - sample_gt_action[:, 1]).abs().mean().item()
                    )
                    sample_ade = float(canonical_metrics.ade_per_sample[row_idx].item())
                    sample_fde = float(canonical_metrics.fde_per_sample[row_idx].item())
                    sample_max_lateral = float(canonical_metrics.max_lateral_per_sample[row_idx].item())
                    sample_payload = {
                        "event": "sample",
                        "sample_index": sample_index,
                        "sample_id": batch["sample_id"][row_idx],
                        "image_path": str(batch["image_path"][row_idx]),
                        "command": batch["command"][row_idx],
                        "flow_steps": int(args.flow_steps),
                        "cfm_loss": batch_loss_value,
                        "gt_action": [[float(x) for x in step] for step in sample_gt_action.tolist()],
                        "pred_action": [[float(x) for x in step] for step in sample_pred_action.tolist()],
                        "gt_waypoints": [
                            [float(point[0]), float(point[1])]
                            for point in sample_gt_waypoints.tolist()
                        ],
                        "pred_waypoints": [
                            [float(point[0]), float(point[1])]
                            for point in sample_pred_waypoints.tolist()
                        ],
                        "ade_m": sample_ade,
                        "fde_m": sample_fde,
                        "max_lateral_error_m": sample_max_lateral,
                        "metrics": {
                            "cfm_loss": batch_loss_value,
                            "action_mae_accel": sample_action_mae_accel,
                            "action_mae_kappa": sample_action_mae_kappa,
                            "ade_m": sample_ade,
                            "fde_m": sample_fde,
                            "max_lateral_error_m": sample_max_lateral,
                            "action_horizon": action_horizon,
                        },
                    }
                    if pid_action is not None and pid_waypoints is not None:
                        sample_pid_action = pid_action[row_idx].detach().cpu()
                        sample_pid_waypoints = pid_waypoints[row_idx].detach().cpu()
                        sample_pid_action_mae_accel = float(
                            (sample_pid_action[:, 0] - sample_gt_action[:, 0]).abs().mean().item()
                        )
                        sample_pid_action_mae_kappa = float(
                            (sample_pid_action[:, 1] - sample_gt_action[:, 1]).abs().mean().item()
                        )
                        sample_payload["pid_override"] = {
                            "target_speed_kmh": float(args.pid_target_speed_kmh),
                            "pid_gains": {
                                "kp": float(args.pid_kp),
                                "ki": float(args.pid_ki),
                                "kd": float(args.pid_kd),
                            },
                            "pred_action": [
                                [float(x) for x in step] for step in sample_pid_action.tolist()
                            ],
                            "pred_waypoints": [
                                [float(point[0]), float(point[1])]
                                for point in sample_pid_waypoints.tolist()
                            ],
                            "ade_m": float(pid_metrics.ade_per_sample[row_idx].item()),
                            "fde_m": float(pid_metrics.fde_per_sample[row_idx].item()),
                            "max_lateral_error_m": float(
                                pid_metrics.max_lateral_per_sample[row_idx].item()
                            ),
                            "metrics": {
                                "action_mae_accel": sample_pid_action_mae_accel,
                                "action_mae_kappa": sample_pid_action_mae_kappa,
                                "ade_m": float(pid_metrics.ade_per_sample[row_idx].item()),
                                "fde_m": float(pid_metrics.fde_per_sample[row_idx].item()),
                                "max_lateral_error_m": float(
                                    pid_metrics.max_lateral_per_sample[row_idx].item()
                                ),
                                "action_horizon": action_horizon,
                            },
                        }
                    reporter.emit_sample(sample_payload, print_to_stdout=sample_index < 5)

                total_loss += float(loss.detach().cpu())
                total_batches += 1
                total_ade += float(canonical_metrics.ade_per_sample.sum().item())
                total_fde += float(canonical_metrics.fde_per_sample.sum().item())
                total_samples += batch_size
                total_action_steps += batch_size * int(gt_action_seq.shape[1])
                canonical_total_mean_max_lateral += float(
                    canonical_metrics.max_lateral_per_sample.sum().item()
                )
                canonical_global_max_lateral = max(
                    canonical_global_max_lateral, float(canonical_metrics.lateral_error.max().item())
                )
                accel_mae, kappa_mae = compute_stage1b_action_mae_sums(
                    pred_action=pred_action,
                    gt_action_seq=gt_action_seq,
                )
                canonical_total_action_mae_accel += accel_mae
                canonical_total_action_mae_kappa += kappa_mae
                if pid_action is not None:
                    pid_total_ade += float(pid_metrics.ade_per_sample.sum().item())
                    pid_total_fde += float(pid_metrics.fde_per_sample.sum().item())
                    pid_total_mean_max_lateral += float(pid_metrics.max_lateral_per_sample.sum().item())
                    pid_global_max_lateral = max(
                        pid_global_max_lateral, float(pid_metrics.lateral_error.max().item())
                    )
                    pid_accel_mae, pid_kappa_mae = compute_stage1b_action_mae_sums(
                        pred_action=pid_action,
                        gt_action_seq=gt_action_seq,
                    )
                    pid_total_action_mae_accel += pid_accel_mae
                    pid_total_action_mae_kappa += pid_kappa_mae

                running_metrics = {
                    "cfm_loss": total_loss / max(1, total_batches),
                    "ade_m": total_ade / max(1, total_samples),
                    "fde_m": total_fde / max(1, total_samples),
                    "mean_max_lateral_error_m": canonical_total_mean_max_lateral / max(1, total_samples),
                    "global_max_lateral_error_m": canonical_global_max_lateral,
                    "action_mae_accel": canonical_total_action_mae_accel / max(1, total_action_steps),
                    "action_mae_kappa": canonical_total_action_mae_kappa / max(1, total_action_steps),
                }
                if args.include_pid_override:
                    running_metrics["pid_override"] = {
                        "ade_m": pid_total_ade / max(1, total_samples),
                        "fde_m": pid_total_fde / max(1, total_samples),
                        "mean_max_lateral_error_m": pid_total_mean_max_lateral / max(1, total_samples),
                        "global_max_lateral_error_m": pid_global_max_lateral,
                        "action_mae_accel": pid_total_action_mae_accel / max(1, total_action_steps),
                        "action_mae_kappa": pid_total_action_mae_kappa / max(1, total_action_steps),
                    }
                reporter.emit_progress(
                    processed_samples=total_samples,
                    running_metrics=running_metrics,
                )

        summary = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
            "num_samples": total_samples,
            "cfm_loss": total_loss / max(1, total_batches),
            "ade_m": total_ade / max(1, total_samples),
            "fde_m": total_fde / max(1, total_samples),
            "mean_max_lateral_error_m": canonical_total_mean_max_lateral / max(1, total_samples),
            "global_max_lateral_error_m": canonical_global_max_lateral,
            "flow_steps": args.flow_steps,
            "condition_source": stage1b_metadata["condition_source"],
            "pid_override_enabled": bool(args.include_pid_override),
            "action_mae_accel": canonical_total_action_mae_accel / max(1, total_action_steps),
            "action_mae_kappa": canonical_total_action_mae_kappa / max(1, total_action_steps),
        }
        if args.include_pid_override:
            summary["pid_override"] = {
                "target_speed_kmh": float(args.pid_target_speed_kmh),
                "pid_gains": {
                    "kp": float(args.pid_kp),
                    "ki": float(args.pid_ki),
                    "kd": float(args.pid_kd),
                },
                "ade_m": pid_total_ade / max(1, total_samples),
                "fde_m": pid_total_fde / max(1, total_samples),
                "mean_max_lateral_error_m": pid_total_mean_max_lateral / max(1, total_samples),
                "global_max_lateral_error_m": pid_global_max_lateral,
                "action_mae_accel": pid_total_action_mae_accel / max(1, total_action_steps),
                "action_mae_kappa": pid_total_action_mae_kappa / max(1, total_action_steps),
            }
        reporter.emit_summary("stage1b_eval_summary", summary)
        upsert_manifest(
            artifact_kind="eval",
            stage="stage1b_eval",
            run_name=Path(args.output_json).resolve().stem,
            summary_json=args.output_json,
            checkpoint=args.checkpoint,
            dataset_path=",".join(args.eval_jsonl),
            progress_json=str(args.progress_json),
            per_sample_jsonl=str(args.per_sample_jsonl),
            wandb_run_url=wandb_run_url,
        )
    except Exception as exc:
        reporter.emit_failure("stage1b_eval_failure", exc)
        raise
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
