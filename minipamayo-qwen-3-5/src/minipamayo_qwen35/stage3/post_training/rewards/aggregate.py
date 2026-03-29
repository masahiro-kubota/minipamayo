"""Reward aggregation for canonical Stage 3."""

from __future__ import annotations

from dataclasses import dataclass

from .consistency import ConsistencyRewardResult
from .reasoning import ReasoningRewardResult
from .trajectory import TrajectoryRewardResult


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
