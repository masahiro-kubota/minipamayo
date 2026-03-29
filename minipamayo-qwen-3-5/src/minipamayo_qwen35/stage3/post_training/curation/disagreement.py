"""Disagreement-based curation helpers for Stage 3."""

from __future__ import annotations

import torch


def _normalize_distribution(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1:
        raise RuntimeError(f"Expected a 1D tensor, got shape {tuple(values.shape)}.")
    return torch.softmax(values, dim=0)


def policy_distribution_from_logprobs(policy_logprob_sums: torch.Tensor) -> torch.Tensor:
    return _normalize_distribution(policy_logprob_sums.to(torch.float32))


def reward_boltzmann_distribution(rewards: torch.Tensor, beta: float) -> torch.Tensor:
    return _normalize_distribution(float(beta) * rewards.to(torch.float32))


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = 0.5 * (p + q)
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = m.clamp_min(eps)
    return 0.5 * torch.sum(p * (p.log() - m.log())) + 0.5 * torch.sum(q * (q.log() - m.log()))


def compute_disagreement_score(
    *,
    policy_logprob_sums: torch.Tensor,
    rewards: torch.Tensor,
    reward_beta: float,
) -> float:
    policy_dist = policy_distribution_from_logprobs(policy_logprob_sums)
    reward_dist = reward_boltzmann_distribution(rewards, reward_beta)
    return float(js_divergence(policy_dist, reward_dist).item())
