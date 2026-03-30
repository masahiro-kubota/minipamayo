"""Metadata helpers for canonical Stage 1B training checkpoints."""

from __future__ import annotations

import torch


def compute_action_stats(dataset) -> dict[str, float]:
    accel_rows = []
    kappa_rows = []
    for index in range(len(dataset)):
        action = dataset[index]["action"].cpu().numpy()
        accel_rows.append(action[0::2])
        kappa_rows.append(action[1::2])
    accel = torch.tensor(accel_rows, dtype=torch.float32).flatten()
    kappa = torch.tensor(kappa_rows, dtype=torch.float32).flatten()
    accel_std = float(torch.std(accel, unbiased=False).item())
    kappa_std = float(torch.std(kappa, unbiased=False).item())
    if accel_std <= 0.0 or kappa_std <= 0.0:
        raise RuntimeError("Stage 1B action normalization requires non-zero accel and kappa std.")
    return {
        "accel_mean": float(torch.mean(accel).item()),
        "accel_std": accel_std,
        "kappa_mean": float(torch.mean(kappa).item()),
        "kappa_std": kappa_std,
    }


def build_stage1b_metadata(
    dataset,
    args,
    expert_config: dict,
    action_stats: dict,
    diffusion_cfg: dict,
) -> dict:
    record = dataset[0]
    action = record["action"]
    gt_waypoints = record["gt_waypoints"]
    dt_value = float(record["dt"].item()) if hasattr(record["dt"], "item") else float(record["dt"])
    return {
        "stage1_checkpoint": args.stage1_checkpoint,
        "train_jsonl": list(args.train_jsonl),
        "val_jsonl": list(args.val_jsonl) if args.val_jsonl is not None else None,
        "sample_format": "jsonl+images",
        "condition_source": "prompt_past_key_values",
        "conditioning_contract": "detached_prompt_cache_from_stage1a_prompt",
        "k": len(gt_waypoints),
        "action_dim": int(action.shape[0]),
        "dt": dt_value,
        "expert_architecture": "alpamayo_style_action_expert",
        "diffusion_architecture": "flow_matching",
        "diffusion_cfg": dict(diffusion_cfg),
        "action_space_contract": "alpamayo_unicycle_accel_curvature_single_traj_group",
        "action_space_cfg": {
            "_target_": "minipamayo_qwen35.action_space.unicycle_accel_curvature.UnicycleAccelCurvatureActionSpace",
            "dt": dt_value,
            "n_waypoints": len(gt_waypoints),
            "accel_mean": float(action_stats["accel_mean"]),
            "accel_std": float(action_stats["accel_std"]),
            "curvature_mean": float(action_stats["kappa_mean"]),
            "curvature_std": float(action_stats["kappa_std"]),
            "theta_lambda": 1e-6,
            "theta_ridge": 1e-8,
            "v_lambda": 1e-6,
            "v_ridge": 1e-4,
            "a_lambda": 1e-4,
            "a_ridge": 1e-4,
            "kappa_lambda": 1e-4,
            "kappa_ridge": 1e-4,
        },
        "expert_config": expert_config,
    }
