"""Reward components for canonical Stage 3."""

from .aggregate import AggregatedReward, RewardWeights, aggregate_rewards
from .consistency import ConsistencyRewardResult, score_consistency
from .reasoning import ReasoningRewardResult, build_reasoning_reward_scorer
from .trajectory import TrajectoryRewardResult, score_trajectory

__all__ = [
    "AggregatedReward",
    "ConsistencyRewardResult",
    "ReasoningRewardResult",
    "RewardWeights",
    "TrajectoryRewardResult",
    "aggregate_rewards",
    "build_reasoning_reward_scorer",
    "score_consistency",
    "score_trajectory",
]
