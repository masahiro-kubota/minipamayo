"""Stage 1B-specific diffusion adapters layered on top of shared Flow Matching."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..diffusion.flow_matching import FlowMatching
from ..models.action_expert import reshape_action_for_expert, reshape_flow_matching_timesteps


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
        gt_action = reshape_action_for_expert(
            gt_action,
            action_dims=expert.action_dims,
            action_dim=expert.action_dim,
        )
        normalized_gt_action = expert.normalize(gt_action)
        noise = torch.randn_like(normalized_gt_action)
        beta_dist = torch.distributions.Beta(self.beta_alpha, self.beta_beta)
        t = beta_dist.sample((gt_action.shape[0],)).to(
            device=gt_action.device,
            dtype=torch.float32,
        )
        t_column, t_expert = reshape_flow_matching_timesteps(t, batch_size=gt_action.shape[0])
        t_mixed = t_column
        while t_mixed.dim() < normalized_gt_action.dim():
            t_mixed = t_mixed.unsqueeze(-1)
        mixed = t_mixed * normalized_gt_action + (1.0 - t_mixed) * noise
        target = normalized_gt_action - noise
        pred = expert(
            noisy_action=mixed,
            t=t_expert,
            prompt_cache=prompt_cache,
            prompt_attention_mask=prompt_attention_mask,
        )
        return torch.mean((pred - target) ** 2)

    @torch.no_grad()
    def sample(
        self,
        *,
        expert,
        prompt_cache,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = prompt_attention_mask.shape[0]
        sampler = FlowMatching(
            x_dims=expert.action_dims,
            num_inference_steps=self.n_steps,
        )

        def step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            _, t_expert = reshape_flow_matching_timesteps(t, batch_size=x.shape[0])
            return expert(
                noisy_action=x,
                t=t_expert,
                prompt_cache=prompt_cache,
                prompt_attention_mask=prompt_attention_mask,
            )

        current = sampler.sample(
            batch_size=batch_size,
            step_fn=step_fn,
            device=prompt_attention_mask.device,
        )
        return expert.denormalize(current)
