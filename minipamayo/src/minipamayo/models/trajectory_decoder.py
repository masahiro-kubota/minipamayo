"""Trajectory Decoder with Conditional Flow Matching (CFM).

Generates continuous (a, kappa) x K trajectories conditioned on LLM hidden states.
Uses Gaussian Optimal Transport path for flow matching.

Fail-fast config: 4 layers, 256 dim (~3M params).
Production config: 12 layers, 512 dim (~150M params).
"""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding + MLP projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) timestep in [0, 1]
        Returns:
            emb: (B, dim)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.mlp(emb)


class FlowTransformerBlock(nn.Module):
    """Transformer block with AdaLN + cross-attention for KV-cache conditioning."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        # AdaLN modulation: time -> (scale1, shift1, scale_cross, shift_cross, scale2, shift2)
        self.adaLN = nn.Linear(dim, dim * 6)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, dim)
            t_emb: (B, dim)
            kv_cache: (B, L_cond, dim) — projected VLM hidden state sequence
        """
        # AdaLN parameters
        params = self.adaLN(t_emb).unsqueeze(1)  # (B, 1, dim*6)
        s1, b1, sc, bc, s2, b2 = params.chunk(6, dim=-1)

        # Self-attention with AdaLN
        h = self.norm1(x) * (1 + s1) + b1
        h, _ = self.attn(h, h, h)
        x = x + h

        # Cross-attention with KV-cache (Alpamayo §5.2)
        h = self.norm_cross(x) * (1 + sc) + bc
        h, _ = self.cross_attn(h, kv_cache, kv_cache)
        x = x + h

        # MLP with AdaLN
        h = self.norm2(x) * (1 + s2) + b2
        h = self.mlp(h)
        x = x + h

        return x


class TrajectoryDecoder(nn.Module):
    """Flow Matching trajectory decoder.

    Predicts velocity field v_theta(a_t, t, c) for CFM.
    Input: noisy action sequence a_t + condition from LLM.
    Output: predicted velocity field.
    """

    def __init__(
        self,
        action_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        condition_dim: int = 896,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Project action to hidden dim
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Project condition to hidden dim
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)

        # Time embedding
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [FlowTransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )

        # Output projection
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        a_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field.

        Args:
            a_t: (B, action_dim) noisy action at time t
            t: (B,) timestep in [0, 1]
            condition: (B, L_cond, condition_dim) VLM hidden state sequence

        Returns:
            v_theta: (B, action_dim) predicted velocity
        """
        # Project inputs
        h_action = self.action_proj(a_t).unsqueeze(1)  # (B, 1, hidden)
        kv_cache = self.cond_proj(condition)  # (B, L_cond, hidden)
        t_emb = self.time_emb(t)  # (B, hidden)

        # Action tokens as query sequence
        x = h_action  # (B, 1, hidden)

        # Transformer blocks with time conditioning + cross-attention to KV-cache
        for block in self.blocks:
            x = block(x, t_emb, kv_cache)

        # Output projection
        x = self.norm_out(x[:, 0, :])  # (B, hidden)
        return self.out_proj(x)  # (B, action_dim)


def cfm_loss(
    model: TrajectoryDecoder,
    action_gt: torch.Tensor,
    condition: torch.Tensor,
    beta_a: float = 2.0,
    beta_b: float = 5.0,
) -> torch.Tensor:
    """Conditional Flow Matching loss with shifted beta schedule (Alpamayo §5.2).

    Gaussian OT path: a_t = t * a + (1 - t) * epsilon
    Target field: u = a - epsilon
    Loss: ||v_theta(a_t, t, c) - u||^2

    Args:
        model: TrajectoryDecoder
        action_gt: (B, action_dim) ground truth action
        condition: (B, L_cond, condition_dim) VLM hidden state sequence
        beta_a: Beta distribution alpha parameter
        beta_b: Beta distribution beta parameter

    Returns:
        loss: scalar
    """
    B = action_gt.shape[0]
    device = action_gt.device

    # Shifted beta distribution for time sampling (Alpamayo §5.2)
    t = torch.distributions.Beta(beta_a, beta_b).sample((B,)).to(device)
    epsilon = torch.randn_like(action_gt)

    # Gaussian OT path
    a_t = t.unsqueeze(1) * action_gt + (1 - t.unsqueeze(1)) * epsilon

    # Target velocity
    u_target = action_gt - epsilon

    # Predicted velocity
    v_pred = model(a_t, t, condition)

    # MSE loss
    return (v_pred - u_target).pow(2).mean()


@torch.no_grad()
def cfm_sample(
    model: TrajectoryDecoder,
    condition: torch.Tensor,
    action_dim: int,
    n_steps: int = 10,
) -> torch.Tensor:
    """Sample trajectory using Euler integration of the learned flow.

    Args:
        model: TrajectoryDecoder
        condition: (B, L_cond, condition_dim) VLM hidden state sequence
        action_dim: dimension of action vector
        n_steps: number of Euler integration steps

    Returns:
        a_1: (B, action_dim) sampled action (at t=1)
    """
    B = condition.shape[0]
    device = condition.device
    dt = 1.0 / n_steps

    # Start from noise
    a_t = torch.randn(B, action_dim, device=device)

    for i in range(n_steps):
        t = torch.full((B,), i / n_steps, device=device)
        v = model(a_t, t, condition)
        a_t = a_t + dt * v

    return a_t
