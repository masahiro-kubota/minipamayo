"""Alpamayo-style Flow Matching Expert for trajectory generation.

Uses a Qwen2 Transformer as the denoising network, conditioned on VLM KV-cache
via past_key_values (Alpamayo §5.1-5.2). Action inputs are encoded with
Fourier Feature V2 + MLP (Alpamayo action_in_proj).

KV-cache compatibility constraint:
  Expert must have same num_kv_heads (2) and head_dim (64) as Qwen2.5-0.5B
  so VLM's past_key_values can be consumed directly by Expert.

Default config: 24 layers, 640 hidden, 10 heads (~146M params).
  Matches Alpamayo's Expert/VLM ratio (~25%): 146M / 494M ≈ 30%.
Full config: 24 layers, 896 hidden, 14 heads (~280M params).
"""

import math

import torch
import torch.nn as nn
from transformers import AutoModel, Qwen2Config
from transformers.cache_utils import DynamicCache

# ============================================================
# Fourier Feature V2 (from Alpamayo action_in_proj.py)
# ============================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class FourierEncoderV2(nn.Module):
    """Fourier feature encoder with log-spaced frequencies (Alpamayo §5.2)."""

    def __init__(self, dim: int = 20, max_freq: float = 100.0):
        super().__init__()
        half = dim // 2
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        self.out_dim = dim
        self.register_buffer("freqs", freqs[None, :])  # (1, half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (...,) → (..., dim)"""
        arg = x[..., None] * self.freqs * 2 * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)


class MLPEncoder(nn.Module):
    """MLP with RMSNorm + SiLU (from Alpamayo action_in_proj.py)."""

    def __init__(self, num_input_feats: int, num_layers: int, hidden_size: int, out_dim: int):
        super().__init__()
        layers = [nn.Linear(num_input_feats, hidden_size), nn.SiLU()]
        for i in range(num_layers):
            if i < num_layers - 1:
                layers.extend(
                    [RMSNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.SiLU()]
                )
            else:
                layers.extend([RMSNorm(hidden_size), nn.Linear(hidden_size, out_dim)])
        self.trunk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class ActionInProj(nn.Module):
    """Per-waypoint Fourier + MLP action input projection (Alpamayo §5.2).

    Encodes each action dimension (a, kappa) separately with Fourier features,
    concatenates with timestep Fourier features, and projects to expert hidden dim.
    """

    def __init__(
        self,
        K: int,
        out_dim: int,
        num_fourier_feats: int = 20,
        max_freq: float = 100.0,
        mlp_hidden_size: int = 1024,
        mlp_num_layers: int = 4,
    ):
        super().__init__()
        self.K = K
        # Separate Fourier encoder for each action dim (a and kappa)
        self.action_fourier = nn.ModuleList(
            [FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq) for _ in range(2)]
        )
        self.timestep_fourier = FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq)

        num_input_feats = num_fourier_feats * 2 + num_fourier_feats  # 2 action dims + timestep
        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats,
            num_layers=mlp_num_layers,
            hidden_size=mlp_hidden_size,
            out_dim=out_dim,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K, 2) noisy action [a, kappa] per waypoint
            t: (B, 1) or (B,) timestep in [0, 1]
        Returns:
            (B, K, out_dim) projected action features
        """
        B, K, _ = x.shape
        # Fourier-encode each action dim separately
        action_feats = torch.cat(
            [enc(x[:, :, i]) for i, enc in enumerate(self.action_fourier)], dim=-1
        )  # (B, K, num_fourier * 2)

        # Fourier-encode timestep and broadcast to all waypoints
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_feats = self.timestep_fourier(t[:, -1])  # (B, num_fourier)
        t_feats = t_feats.unsqueeze(1).expand(-1, K, -1)  # (B, K, num_fourier)

        # Concat and project
        feats = torch.cat([action_feats, t_feats], dim=-1)  # (B, K, num_fourier * 3)
        projected = self.encoder(feats.flatten(0, 1))  # (B*K, out_dim)
        return self.norm(projected.reshape(B, K, -1))  # (B, K, out_dim)


# ============================================================
# KV-cache utilities
# ============================================================


def clone_kv_cache(cache: DynamicCache) -> DynamicCache:
    """Deep clone a DynamicCache for reuse across multiple samples."""
    new_cache = DynamicCache()
    for layer_idx in range(len(cache.layers)):
        new_cache.update(
            cache.layers[layer_idx].keys.clone(),
            cache.layers[layer_idx].values.clone(),
            layer_idx,
        )
    return new_cache


# ============================================================
# Trajectory Decoder (Alpamayo-style Expert)
# ============================================================


class TrajectoryDecoder(nn.Module):
    """Alpamayo-style Flow Matching Expert.

    A Qwen2 Transformer that consumes VLM KV-cache via past_key_values
    and predicts velocity fields for trajectory generation.
    """

    # Qwen2.5-0.5B fixed values for KV-cache compatibility
    VLM_NUM_KV_HEADS = 2
    VLM_HEAD_DIM = 64

    def __init__(
        self,
        K: int = 6,
        hidden_size: int = 640,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 10,
        intermediate_size: int | None = None,
        num_fourier_feats: int = 20,
        fourier_max_freq: float = 100.0,
        mlp_hidden_size: int = 1024,
        mlp_num_layers: int = 4,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        kappa_mean: float = 0.0,
        kappa_std: float = 1.0,
    ):
        """
        Args:
            K: Number of waypoints (e.g. 6 for 3s @ 0.5s intervals)
            hidden_size: Expert hidden dim. Must = num_attention_heads * 64 (head_dim).
            num_hidden_layers: Must match VLM (24 for Qwen2.5-0.5B) for KV-cache compat.
            num_attention_heads: Expert Q heads. hidden_size / num_attention_heads must = 64.
            intermediate_size: FFN intermediate dim. Default: hidden_size * 4.
            accel_mean/std: Action normalization stats for acceleration.
            kappa_mean/std: Action normalization stats for curvature.
        """
        super().__init__()
        self.K = K
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # Action normalization (Alpamayo §5.1)
        self.register_buffer("accel_mean", torch.tensor(accel_mean))
        self.register_buffer("accel_std", torch.tensor(accel_std))
        self.register_buffer("kappa_mean", torch.tensor(kappa_mean))
        self.register_buffer("kappa_std", torch.tensor(kappa_std))

        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        # Validate KV-cache compatibility
        head_dim = hidden_size // num_attention_heads
        assert head_dim == self.VLM_HEAD_DIM, (
            f"head_dim must be {self.VLM_HEAD_DIM} for KV-cache compat, "
            f"got {head_dim} (hidden={hidden_size}, heads={num_attention_heads})"
        )

        # Build Expert Transformer (Qwen2 architecture, no embed_tokens)
        expert_config = Qwen2Config(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=self.VLM_NUM_KV_HEADS,
            intermediate_size=intermediate_size,
            max_position_embeddings=8192,
            use_cache=True,
            rms_norm_eps=1e-6,
        )
        self.expert = AutoModel.from_config(expert_config)
        del self.expert.embed_tokens  # we provide our own embeddings

        # Action input projection (Fourier V2 + MLP)
        self.action_in_proj = ActionInProj(
            K=K,
            out_dim=hidden_size,
            num_fourier_feats=num_fourier_feats,
            max_freq=fourier_max_freq,
            mlp_hidden_size=mlp_hidden_size,
            mlp_num_layers=mlp_num_layers,
        )

        # Action output projection
        self.action_out_proj = nn.Linear(hidden_size, 2)

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize raw action (interleaved a, kappa) to zero-mean unit-variance.

        Args:
            action: (B, K*2) interleaved [a_0, kappa_0, a_1, kappa_1, ...]
        Returns:
            (B, K*2) normalized action
        """
        out = action.clone()
        out[:, 0::2] = (out[:, 0::2] - self.accel_mean) / self.accel_std
        out[:, 1::2] = (out[:, 1::2] - self.kappa_mean) / self.kappa_std
        return out

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action back to physical units.

        Args:
            action: (B, K*2) normalized interleaved action
        Returns:
            (B, K*2) raw action in physical units
        """
        out = action.clone()
        out[:, 0::2] = out[:, 0::2] * self.accel_std + self.accel_mean
        out[:, 1::2] = out[:, 1::2] * self.kappa_std + self.kappa_mean
        return out

    def forward(
        self,
        a_t: torch.Tensor,
        t: torch.Tensor,
        kv_cache: DynamicCache,
        prefill_seq_len: int,
    ) -> torch.Tensor:
        """Predict velocity field (one denoising step).

        Args:
            a_t: (B, K*2) flat noisy action at time t
            t: (B,) timestep in [0, 1]
            kv_cache: DynamicCache from VLM (will be cropped back after use)
            prefill_seq_len: original VLM KV-cache sequence length

        Returns:
            v_theta: (B, K*2) predicted velocity field
        """
        B = a_t.shape[0]
        device = a_t.device

        # Reshape flat action to (B, K, 2) for Fourier encoding
        x = a_t.view(B, self.K, 2)
        inputs_embeds = self.action_in_proj(x, t)  # (B, K, hidden_size)

        # Position IDs: continue from VLM sequence
        position_ids = torch.arange(self.K, device=device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1) + prefill_seq_len

        # Attention mask: attend to all VLM tokens + expert tokens, non-causal
        total_len = prefill_seq_len + self.K
        attention_mask = torch.zeros(
            (B, 1, self.K, total_len),
            dtype=inputs_embeds.dtype,
            device=device,
        )

        # Expert forward with VLM KV-cache (non-causal, Alpamayo §5.2)
        expert_out = self.expert(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=kv_cache,
            attention_mask=attention_mask,
            use_cache=True,
            is_causal=False,
        )

        # Crop KV-cache to remove expert-added tokens (Alpamayo pattern)
        kv_cache.crop(prefill_seq_len)

        # Output projection: (B, K, hidden_size) → (B, K, 2)
        last_hidden = expert_out.last_hidden_state[:, -self.K :]
        pred = self.action_out_proj(last_hidden)  # (B, K, 2)

        return pred.reshape(B, self.K * 2)  # (B, K*2) flat


# ============================================================
# CFM Loss and Sampling
# ============================================================


def cfm_loss(
    model: TrajectoryDecoder,
    action_gt: torch.Tensor,
    kv_cache: DynamicCache,
    prefill_seq_len: int,
    beta_a: float = 2.0,
    beta_b: float = 5.0,
) -> torch.Tensor:
    """Conditional Flow Matching loss with shifted beta schedule (Alpamayo §5.2).

    Gaussian OT path: a_t = t * a + (1 - t) * epsilon
    Target field: u = a - epsilon
    Loss: ||v_theta(a_t, t, c) - u||^2

    Args:
        model: TrajectoryDecoder (Expert)
        action_gt: (B, K*2) ground truth action (interleaved a, kappa)
        kv_cache: DynamicCache from VLM (will be cloned internally)
        prefill_seq_len: VLM KV-cache sequence length
        beta_a, beta_b: Beta distribution parameters for time sampling

    Returns:
        loss: scalar
    """
    B = action_gt.shape[0]
    device = action_gt.device

    # Normalize GT action to zero-mean unit-variance (Alpamayo §5.1)
    action_norm = model.normalize(action_gt)

    # Clone KV-cache since forward modifies it (crop pattern)
    cache = clone_kv_cache(kv_cache)

    t = torch.distributions.Beta(beta_a, beta_b).sample((B,)).to(device)
    epsilon = torch.randn_like(action_norm)

    # Gaussian OT path (in normalized space)
    a_t = t.unsqueeze(1) * action_norm + (1 - t.unsqueeze(1)) * epsilon

    # Target velocity (in normalized space)
    u_target = action_norm - epsilon

    # Predicted velocity (single step, cache is cloned so original is preserved)
    v_pred = model(a_t, t, cache, prefill_seq_len)

    return (v_pred - u_target).pow(2).mean()


def load_decoder_from_checkpoint(ckpt_path, device):
    """Load Expert decoder from checkpoint.

    Args:
        ckpt_path: path to Stage 2 checkpoint (.pt)
        device: torch device

    Returns:
        decoder: TrajectoryDecoder on device, eval mode
        K: number of waypoints
        ckpt: full checkpoint dict
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    K = ckpt.get("K", 6)
    arch = ckpt.get("architecture", "unknown")

    if arch != "expert_kv_cache":
        raise ValueError(
            f"Unknown architecture '{arch}' in checkpoint. "
            "Old cross-attention checkpoints are not compatible."
        )

    decoder = TrajectoryDecoder(
        K=K,
        hidden_size=ckpt["hidden_size"],
        num_hidden_layers=ckpt["num_hidden_layers"],
        num_attention_heads=ckpt["num_attention_heads"],
        intermediate_size=ckpt.get("intermediate_size"),
        num_fourier_feats=ckpt.get("num_fourier_feats", 20),
        fourier_max_freq=ckpt.get("fourier_max_freq", 100.0),
        mlp_hidden_size=ckpt.get("mlp_hidden_size", 1024),
        mlp_num_layers=ckpt.get("mlp_num_layers", 4),
    )
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder = decoder.to(device).eval()
    decoder.requires_grad_(False)
    return decoder, K, ckpt


@torch.no_grad()
def cfm_sample(
    model: TrajectoryDecoder,
    kv_cache: DynamicCache,
    prefill_seq_len: int,
    n_steps: int = 10,
) -> torch.Tensor:
    """Sample trajectory using Euler integration of the learned flow.

    Args:
        model: TrajectoryDecoder (Expert)
        kv_cache: DynamicCache from VLM (will be cloned internally)
        prefill_seq_len: VLM KV-cache sequence length
        n_steps: number of Euler integration steps

    Returns:
        a_1: (B, K*2) sampled action (at t=1)
    """
    B = kv_cache.layers[0].keys.shape[0]
    device = kv_cache.layers[0].keys.device
    action_dim = model.K * 2

    # Clone KV-cache for this sample (crop pattern resets each step)
    cache = clone_kv_cache(kv_cache)

    dt = 1.0 / n_steps
    a_t = torch.randn(B, action_dim, device=device)

    for i in range(n_steps):
        t = torch.full((B,), i / n_steps, device=device)
        v = model(a_t, t, cache, prefill_seq_len)
        a_t = a_t + dt * v

    # Denormalize back to physical units
    return model.denormalize(a_t)
