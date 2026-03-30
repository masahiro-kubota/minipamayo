"""Canonical Stage 1B single-sample inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..dataset import Stage1JsonlDataset, stage1_collate
from .cli import (
    COMMON_CONFIG_PATH_KEYS,
    add_stage1b_common_args,
    parse_stage1b_json_only_args,
    require_stage1b_cuda_device,
    validate_stage1b_runtime_args,
)
from .runtime import (
    load_stage1b_runtime,
    run_stage1b_inference_batch,
)

CONFIG_PATH_KEYS = COMMON_CONFIG_PATH_KEYS | {"sample_jsonl"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run canonical Stage 1B inference on one sample.")
    add_stage1b_common_args(parser)
    parser.add_argument("--sample-jsonl", type=str, default="")
    parser.add_argument("--sample-index", type=int, default=0)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_stage1b_json_only_args(
        parser=parser,
        path_keys=CONFIG_PATH_KEYS,
        json_only_error="Stage 1B inference accepts only --config-json. Put all settings in the JSON file.",
    )
    validate_stage1b_runtime_args(args)
    if not args.checkpoint or not args.stage1_checkpoint or not args.sample_jsonl:
        raise RuntimeError("`checkpoint`, `stage1_checkpoint`, and `sample_jsonl` must be defined.")
    if args.sample_index < 0:
        raise RuntimeError("`sample_index` must be >= 0.")
    return args


def main() -> None:
    args = parse_args()
    device = require_stage1b_cuda_device()
    dataset = Stage1JsonlDataset(args.sample_jsonl)
    if args.sample_index >= len(dataset):
        raise RuntimeError(
            f"`sample_index` {args.sample_index} is out of range for dataset size {len(dataset)}."
        )
    sample = dataset[args.sample_index]
    batch = stage1_collate([sample])

    runtime = load_stage1b_runtime(
        stage1_checkpoint=args.stage1_checkpoint,
        checkpoint=args.checkpoint,
        flow_steps=args.flow_steps,
        device=device,
    )
    outputs = run_stage1b_inference_batch(
        runtime=runtime,
        batch=batch,
        include_pid_override=args.include_pid_override,
        pid_target_speed_kmh=args.pid_target_speed_kmh,
        pid_kp=args.pid_kp,
        pid_ki=args.pid_ki,
        pid_kd=args.pid_kd,
    )
    pred_action = outputs.pred_action
    pred_waypoints = outputs.pred_waypoints[0].detach().cpu()
    gt_waypoints = outputs.gt_waypoints[0].detach().cpu()
    errors = torch.norm(pred_waypoints.cpu() - gt_waypoints, dim=1)
    lateral_error = (pred_waypoints[:, 1] - gt_waypoints[:, 1]).abs()
    payload = {
        "sample_id": sample["sample_id"],
        "image_path": sample["image_path"],
        "flow_steps": args.flow_steps,
        "dt": runtime.dt,
        "pred_action": pred_action[0].detach().cpu().tolist(),
        "pred_waypoints": pred_waypoints.detach().cpu().tolist(),
        "gt_waypoints": gt_waypoints.tolist(),
        "ade_m": float(errors.mean().item()),
        "fde_m": float(errors[-1].item()),
        "max_lateral_error_m": float(lateral_error.max().item()),
    }
    if outputs.pid_action is not None and outputs.pid_waypoints is not None:
        pid_action = outputs.pid_action
        pid_waypoints = outputs.pid_waypoints[0].detach().cpu()
        pid_errors = torch.norm(pid_waypoints - gt_waypoints, dim=1)
        pid_lateral_error = (pid_waypoints[:, 1] - gt_waypoints[:, 1]).abs()
        payload["pid_override"] = {
            "target_speed_kmh": float(args.pid_target_speed_kmh),
            "pid_gains": {
                "kp": float(args.pid_kp),
                "ki": float(args.pid_ki),
                "kd": float(args.pid_kd),
            },
            "pred_action": pid_action[0].detach().cpu().tolist(),
            "pred_waypoints": pid_waypoints.tolist(),
            "ade_m": float(pid_errors.mean().item()),
            "fde_m": float(pid_errors[-1].item()),
            "max_lateral_error_m": float(pid_lateral_error.max().item()),
        }
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
