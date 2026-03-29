"""Shared helpers for canonical Stage 3."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

CANONICAL_STAGE3_POLICY_OUTPUT_CONTRACT = "reason_and_action_tokens"
STAGE3_REWARD_CONTRACT_V0 = "reasoning+consistency+trajectory_l2_jerk"


def validate_stage2_policy_contract(checkpoint: dict) -> dict:
    stage2_metadata = checkpoint.get("stage2_metadata")
    if not isinstance(stage2_metadata, dict):
        raise RuntimeError("Stage 2 checkpoint is missing canonical `stage2_metadata`.")

    policy_output_contract = stage2_metadata.get("policy_output_contract")
    if policy_output_contract != CANONICAL_STAGE3_POLICY_OUTPUT_CONTRACT:
        raise RuntimeError(
            "Canonical Stage 3 requires a Stage 2 checkpoint trained on the "
            "`Reason + discrete action tokens` contract.\n"
            f"required_policy_output_contract={CANONICAL_STAGE3_POLICY_OUTPUT_CONTRACT!r}\n"
            f"checkpoint_policy_output_contract={policy_output_contract!r}\n"
            f"checkpoint_target_layout={stage2_metadata.get('target_layout')!r}\n"
            "Current reasoning-handoff Stage 2 checkpoints are intentionally rejected."
        )
    return stage2_metadata


def freeze_model(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def configure_trainable_policy(model) -> int:
    trainable_params = 0
    for name, parameter in model.named_parameters():
        lower = name.lower()
        if any(token in lower for token in ("vision", "visual", "patch_embed", "merger")):
            parameter.requires_grad_(False)
        else:
            parameter.requires_grad_(True)
            trainable_params += int(parameter.numel())
    return trainable_params


def maybe_autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def clone_model_inputs(model_inputs: dict) -> dict:
    cloned: dict[str, Any] = {}
    for key, value in model_inputs.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def group_relative_advantages(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.ndim != 1:
        raise RuntimeError(f"Stage 3 expects 1D reward tensors, got {tuple(rewards.shape)}.")
    return rewards - rewards.mean()


def compute_grpo_loss(
    *,
    policy_logprob_sums: torch.Tensor,
    ref_logprob_sums: torch.Tensor,
    rewards: torch.Tensor,
    kl_weight: float,
    sample_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if policy_logprob_sums.shape != ref_logprob_sums.shape:
        raise RuntimeError(
            "Policy/ref rollout logprob shapes do not match.\n"
            f"policy={tuple(policy_logprob_sums.shape)}\n"
            f"ref={tuple(ref_logprob_sums.shape)}"
        )
    advantages = group_relative_advantages(rewards)
    policy_loss = -(advantages.detach() * policy_logprob_sums).mean()
    approx_kl = (policy_logprob_sums - ref_logprob_sums).mean()
    total_loss = (policy_loss + kl_weight * approx_kl) * float(sample_weight)
    return total_loss, {
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()) if rewards.numel() > 1 else 0.0,
        "advantage_abs_mean": float(advantages.abs().mean().item()),
        "policy_logprob_mean": float(policy_logprob_sums.mean().detach().item()),
        "reference_logprob_mean": float(ref_logprob_sums.mean().detach().item()),
        "approx_kl": float(approx_kl.detach().item()),
    }


def stage3_checkpoint_payload(
    *,
    model,
    optimizer,
    args: dict,
    stage3_metadata: dict,
    metrics_history: list[dict],
    run_metadata: dict,
    epoch: int,
    global_step: int,
) -> dict:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "args": dict(args),
        "stage3_metadata": dict(stage3_metadata),
        "metrics_history": list(metrics_history),
        "run_metadata": dict(run_metadata),
    }
