"""Reasoning-reward adapters for canonical Stage 3."""

from __future__ import annotations

from dataclasses import dataclass


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
