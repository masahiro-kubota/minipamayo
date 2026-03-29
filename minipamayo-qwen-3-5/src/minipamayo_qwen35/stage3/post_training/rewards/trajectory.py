"""Trajectory-quality reward terms for canonical Stage 3 v0."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _compute_jerk_penalty(pred_future_xyz: torch.Tensor, dt: float) -> float:
    xy = pred_future_xyz[..., :2].to(torch.float32)
    if xy.shape[0] < 4:
        return 0.0
    velocity = (xy[1:] - xy[:-1]) / dt
    acceleration = (velocity[1:] - velocity[:-1]) / dt
    jerk = (acceleration[1:] - acceleration[:-1]) / dt
    return float(torch.linalg.norm(jerk, dim=-1).mean().item())


@dataclass(frozen=True)
class TrajectoryRewardResult:
    reward: float
    ade: float
    fde: float
    l2_penalty: float
    jerk_penalty: float


def score_trajectory(
    *,
    sample: dict,
    pred_future_xyz: torch.Tensor,
    l2_weight: float,
    jerk_weight: float,
) -> TrajectoryRewardResult:
    pred_xy = pred_future_xyz[..., :2].to(torch.float32)
    gt_xy = sample["gt_waypoints"].to(torch.float32).cpu()
    if pred_xy.shape != gt_xy.shape:
        raise RuntimeError(
            "Predicted trajectory and ground-truth waypoints have different shapes.\n"
            f"pred={tuple(pred_xy.shape)}\n"
            f"gt={tuple(gt_xy.shape)}"
        )
    l2_penalty = float(torch.mean((pred_xy - gt_xy) ** 2).item())
    displacement = torch.linalg.norm(pred_xy - gt_xy, dim=-1)
    ade = float(displacement.mean().item())
    fde = float(displacement[-1].item())
    jerk_penalty = _compute_jerk_penalty(pred_future_xyz, float(sample["dt"]))
    reward = -(float(l2_weight) * l2_penalty + float(jerk_weight) * jerk_penalty)
    return TrajectoryRewardResult(
        reward=reward,
        ade=ade,
        fde=fde,
        l2_penalty=l2_penalty,
        jerk_penalty=jerk_penalty,
    )
