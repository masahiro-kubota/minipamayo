"""Stage 1B-specific diffusion adapters layered on top of shared Flow Matching."""

from __future__ import annotations

import copy

import hydra.utils as hyu
import torch
import torch.nn as nn

from ..models.action_expert import (
    flow_matching_loss,
    flow_matching_sample,
)

DEFAULT_STAGE1B_BETA_ALPHA = 2.0
DEFAULT_STAGE1B_BETA_BETA = 5.0
DEFAULT_STAGE1B_FLOW_STEPS = 10


class BaseDiffusion(nn.Module):
    """Minimal interface compatible with Alpamayo-style expert decoding."""

    def loss(
        self,
        *,
        expert,
        gt_action: torch.Tensor,
        prompt_cache,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        *,
        expert,
        prompt_cache,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class FlowMatchingDiffusion(BaseDiffusion):
    """Vanilla conditional flow matching with Euler sampling."""

    def __init__(
        self,
        *,
        beta_alpha: float = 2.0,
        beta_beta: float = 5.0,
        n_steps: int = 10,
    ):
        super().__init__()
        if beta_alpha <= 0.0 or beta_beta <= 0.0:
            raise RuntimeError("FlowMatchingDiffusion beta parameters must be > 0.")
        if n_steps <= 0:
            raise RuntimeError("FlowMatchingDiffusion `n_steps` must be > 0.")
        self.beta_alpha = float(beta_alpha)
        self.beta_beta = float(beta_beta)
        self.n_steps = int(n_steps)

    def loss(
        self,
        *,
        expert,
        gt_action: torch.Tensor,
        prompt_cache,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        conditioning = expert.prepare_conditioning(
            prompt_cache=prompt_cache,
            prompt_attention_mask=prompt_attention_mask,
        )
        return flow_matching_loss(
            expert=expert,
            gt_action=gt_action,
            conditioning=conditioning,
            beta_alpha=self.beta_alpha,
            beta_beta=self.beta_beta,
        )

    @torch.no_grad()
    def sample(
        self,
        *,
        expert,
        prompt_cache,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        conditioning = expert.prepare_conditioning(
            prompt_cache=prompt_cache,
            prompt_attention_mask=prompt_attention_mask,
        )
        return flow_matching_sample(
            expert=expert,
            conditioning=conditioning,
            n_steps=self.n_steps,
        )


def build_stage1b_diffusion_cfg(
    *,
    beta_alpha: float = DEFAULT_STAGE1B_BETA_ALPHA,
    beta_beta: float = DEFAULT_STAGE1B_BETA_BETA,
    n_steps: int = DEFAULT_STAGE1B_FLOW_STEPS,
) -> dict:
    return {
        "_target_": "minipamayo_qwen35.stage1.stage1b_diffusion.FlowMatchingDiffusion",
        "beta_alpha": float(beta_alpha),
        "beta_beta": float(beta_beta),
        "n_steps": int(n_steps),
    }


def instantiate_stage1b_diffusion(
    *,
    diffusion_cfg: dict | None = None,
    n_steps: int | None = None,
) -> BaseDiffusion:
    resolved_cfg = (
        build_stage1b_diffusion_cfg()
        if diffusion_cfg is None
        else copy.deepcopy(diffusion_cfg)
    )
    resolved_cfg.pop("x_dims", None)
    if n_steps is not None:
        resolved_cfg["n_steps"] = int(n_steps)
    return hyu.instantiate(resolved_cfg)
