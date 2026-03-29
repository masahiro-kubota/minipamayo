"""Reasoning-action consistency reward for canonical Stage 3."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..rollout.parser import parse_decision_from_text


def _trajectory_heading_delta(pred_future_xyz: torch.Tensor) -> float:
    xy = pred_future_xyz[..., :2]
    if xy.shape[0] < 2:
        return 0.0
    deltas = xy[1:] - xy[:-1]
    if deltas.shape[0] == 0:
        return 0.0
    start_heading = torch.atan2(deltas[0, 1], deltas[0, 0])
    end_heading = torch.atan2(deltas[-1, 1], deltas[-1, 0])
    return float((end_heading - start_heading).item())


def infer_decision_from_trajectory(
    *,
    pred_future_xyz: torch.Tensor,
    v0: float,
    dt: float,
) -> dict[str, str]:
    xy = pred_future_xyz[..., :2]
    if xy.shape[0] < 2:
        final_speed = float(v0)
    else:
        velocities = torch.linalg.norm(xy[1:] - xy[:-1], dim=-1) / max(float(dt), 1e-6)
        final_speed = float(velocities[-1].item())

    final_y = float(xy[-1, 1].item()) if xy.numel() > 0 else 0.0
    heading_delta = _trajectory_heading_delta(pred_future_xyz)
    mean_accel = (final_speed - float(v0)) / max(float(dt) * max(xy.shape[0] - 1, 1), 1e-6)

    if final_speed <= 0.5:
        longitudinal = "stop"
    elif mean_accel < -0.5:
        longitudinal = "yield"
    else:
        longitudinal = "go_straight"

    if heading_delta > 0.4:
        lateral = "turn_left"
    elif heading_delta < -0.4:
        lateral = "turn_right"
    elif final_y > 1.0:
        lateral = "lane_change_left"
    elif final_y < -1.0:
        lateral = "lane_change_right"
    else:
        lateral = "lane_keeping"
    return {"longitudinal": longitudinal, "lateral": lateral}


@dataclass(frozen=True)
class ConsistencyRewardResult:
    reward: float
    reasoning_decision: dict[str, str] | None
    trajectory_decision: dict[str, str]


def score_consistency(
    *,
    reasoning_text: str,
    pred_future_xyz: torch.Tensor,
    v0: float,
    dt: float,
) -> ConsistencyRewardResult:
    reasoning_decision = parse_decision_from_text(reasoning_text)
    trajectory_decision = infer_decision_from_trajectory(
        pred_future_xyz=pred_future_xyz,
        v0=v0,
        dt=dt,
    )
    if reasoning_decision is None:
        return ConsistencyRewardResult(
            reward=0.0,
            reasoning_decision=None,
            trajectory_decision=trajectory_decision,
        )

    longitudinal_match = (
        reasoning_decision["longitudinal"] == trajectory_decision["longitudinal"]
    )
    lateral_match = reasoning_decision["lateral"] == trajectory_decision["lateral"]
    return ConsistencyRewardResult(
        reward=1.0 if longitudinal_match and lateral_match else 0.0,
        reasoning_decision=reasoning_decision,
        trajectory_decision=trajectory_decision,
    )
