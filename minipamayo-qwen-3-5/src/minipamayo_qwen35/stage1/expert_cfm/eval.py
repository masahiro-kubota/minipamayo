"""Canonical Stage 1B expert-CFM evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
from .runtime import (
    load_stage1b_runtime,
    run_stage1b_inference_batch,
)

CONFIG_PATH_KEYS = COMMON_CONFIG_PATH_KEYS | {"eval_jsonl"}
MULTI_VALUE_CONFIG_KEYS = {"eval_jsonl"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the canonical Stage 1B expert CFM path.")
    add_stage1b_common_args(parser)
    parser.add_argument("--eval-jsonl", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
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
            displacement = torch.norm(pred_waypoints - gt_waypoints, dim=2)
            lateral_error = (pred_waypoints[:, :, 1] - gt_waypoints[:, :, 1]).abs()
            if outputs.pid_action is not None and outputs.pid_waypoints is not None:
                pid_action = outputs.pid_action
                pid_waypoints = outputs.pid_waypoints
                pid_displacement = torch.norm(pid_waypoints - gt_waypoints, dim=2)
                pid_lateral_error = (pid_waypoints[:, :, 1] - gt_waypoints[:, :, 1]).abs()
            else:
                pid_action = None

            batch_size = gt_action.shape[0]
            total_loss += float(loss.detach().cpu())
            total_batches += 1
            total_ade += float(displacement.mean(dim=1).sum().item())
            total_fde += float(displacement[:, -1].sum().item())
            total_samples += batch_size
            total_action_steps += batch_size * int(gt_action_seq.shape[1])
            canonical_total_mean_max_lateral += float(lateral_error.max(dim=1).values.sum().item())
            canonical_global_max_lateral = max(
                canonical_global_max_lateral, float(lateral_error.max().item())
            )
            canonical_total_action_mae_accel += float(
                (pred_action[:, :, 0] - gt_action_seq[:, :, 0]).abs().sum().item()
            )
            canonical_total_action_mae_kappa += float(
                (pred_action[:, :, 1] - gt_action_seq[:, :, 1]).abs().sum().item()
            )
            if pid_action is not None:
                pid_total_ade += float(pid_displacement.mean(dim=1).sum().item())
                pid_total_fde += float(pid_displacement[:, -1].sum().item())
                pid_total_mean_max_lateral += float(pid_lateral_error.max(dim=1).values.sum().item())
                pid_global_max_lateral = max(pid_global_max_lateral, float(pid_lateral_error.max().item()))
                pid_total_action_mae_accel += float(
                    (pid_action[:, :, 0] - gt_action_seq[:, :, 0]).abs().sum().item()
                )
                pid_total_action_mae_kappa += float(
                    (pid_action[:, :, 1] - gt_action_seq[:, :, 1]).abs().sum().item()
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
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
