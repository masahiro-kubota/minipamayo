"""Shared metric helpers for canonical Stage 1B evaluation and inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Stage1BWaypointMetrics:
    """Canonical waypoint error tensors derived from predicted and ground-truth futures."""

    displacement: torch.Tensor
    lateral_error: torch.Tensor
    ade_per_sample: torch.Tensor
    fde_per_sample: torch.Tensor
    max_lateral_per_sample: torch.Tensor


def compute_stage1b_waypoint_metrics(
    *,
    pred_waypoints: torch.Tensor,
    gt_waypoints: torch.Tensor,
) -> Stage1BWaypointMetrics:
    """Compute canonical ADE/FDE/lateral-error tensors for one batch."""

    if pred_waypoints.shape != gt_waypoints.shape:
        raise RuntimeError(
            "Predicted and ground-truth waypoints must have the same shape.\n"
            f"pred={tuple(pred_waypoints.shape)!r}\n"
            f"gt={tuple(gt_waypoints.shape)!r}"
        )
    displacement = torch.norm(pred_waypoints - gt_waypoints, dim=-1)
    lateral_error = (pred_waypoints[..., 1] - gt_waypoints[..., 1]).abs()
    return Stage1BWaypointMetrics(
        displacement=displacement,
        lateral_error=lateral_error,
        ade_per_sample=displacement.mean(dim=-1),
        fde_per_sample=displacement[..., -1],
        max_lateral_per_sample=lateral_error.max(dim=-1).values,
    )


def compute_stage1b_action_mae_sums(
    *,
    pred_action: torch.Tensor,
    gt_action_seq: torch.Tensor,
) -> tuple[float, float]:
    """Compute summed action-channel MAE for canonical `(accel, kappa)` actions."""

    if pred_action.shape != gt_action_seq.shape:
        raise RuntimeError(
            "Predicted and ground-truth actions must have the same shape.\n"
            f"pred={tuple(pred_action.shape)!r}\n"
            f"gt={tuple(gt_action_seq.shape)!r}"
        )
    accel_mae = float((pred_action[..., 0] - gt_action_seq[..., 0]).abs().sum().item())
    kappa_mae = float((pred_action[..., 1] - gt_action_seq[..., 1]).abs().sum().item())
    return accel_mae, kappa_mae
