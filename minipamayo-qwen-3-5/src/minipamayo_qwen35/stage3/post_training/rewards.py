"""Reward helpers for canonical Stage 3."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .parser import parse_decision_from_text


@dataclass(frozen=True)
class ReasoningRewardResult:
    reward: float
    mode: str
    matched: bool | None


class DisabledReasoningRewardScorer:
    mode = "disabled"

    def score(self, sample: dict, reasoning_text: str) -> ReasoningRewardResult:
        del sample, reasoning_text
        return ReasoningRewardResult(reward=0.0, mode=self.mode, matched=None)


class ExactMatchReasoningRewardScorer:
    mode = "exact_match"

    def score(self, sample: dict, reasoning_text: str) -> ReasoningRewardResult:
        target = str(sample["reasoning_text"]).strip()
        matched = reasoning_text.strip() == target
        return ReasoningRewardResult(
            reward=1.0 if matched else 0.0,
            mode=self.mode,
            matched=matched,
        )


def build_reasoning_reward_scorer(mode: str):
    normalized = mode.strip().lower()
    if normalized == "disabled":
        return DisabledReasoningRewardScorer()
    if normalized == "exact_match":
        return ExactMatchReasoningRewardScorer()
    raise RuntimeError(
        "Unsupported Stage 3 reasoning_reward_mode. Supported modes are "
        "'disabled' and 'exact_match'."
    )


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


@dataclass(frozen=True)
class RewardWeights:
    reasoning: float
    consistency: float
    trajectory: float


@dataclass(frozen=True)
class AggregatedReward:
    total_reward: float
    reasoning_reward: float
    consistency_reward: float
    trajectory_reward: float
    ade: float
    fde: float
    l2_penalty: float
    jerk_penalty: float


def aggregate_rewards(
    *,
    reasoning: ReasoningRewardResult,
    consistency: ConsistencyRewardResult,
    trajectory: TrajectoryRewardResult,
    weights: RewardWeights,
) -> AggregatedReward:
    total_reward = (
        float(weights.reasoning) * reasoning.reward
        + float(weights.consistency) * consistency.reward
        + float(weights.trajectory) * trajectory.reward
    )
    return AggregatedReward(
        total_reward=total_reward,
        reasoning_reward=reasoning.reward,
        consistency_reward=consistency.reward,
        trajectory_reward=trajectory.reward,
        ade=trajectory.ade,
        fde=trajectory.fde,
        l2_penalty=trajectory.l2_penalty,
        jerk_penalty=trajectory.jerk_penalty,
    )


__all__ = [
    "AggregatedReward",
    "ConsistencyRewardResult",
    "ReasoningRewardResult",
    "RewardWeights",
    "TrajectoryRewardResult",
    "aggregate_rewards",
    "build_reasoning_reward_scorer",
    "infer_decision_from_trajectory",
    "score_consistency",
    "score_trajectory",
]
