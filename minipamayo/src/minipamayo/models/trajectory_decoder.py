"""Flow Matching Expert for trajectory generation.

Two decoder architectures:

1. TrajectoryDecoder (KV-cache Expert, Alpamayo §5.1):
   Qwen2 Transformer conditioned on VLM KV-cache via past_key_values.
   Requires KV-cache compatibility: same num_kv_heads (2), head_dim (64).
   Default: 24 layers, 640 hidden, 10 heads (~146M params).

2. CrossAttentionDecoder (cross-attention to VLM hidden states):
   Custom Transformer with AdaLN + cross-attention to VLM hidden states.
   No KV-cache constraint — directly attends to full 896-dim hidden states.
   Default: 4 layers, 256 hidden, 4 heads (~6M params).
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
        attention_dropout: float = 0.0,
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
            rope_theta=1000000.0,  # Must match VLM (Qwen2.5-0.5B) for KV-cache compat
            attention_dropout=attention_dropout,
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

    if arch == "expert_kv_cache":
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
            attention_dropout=ckpt.get("attention_dropout", 0.0),
        )
    elif arch == "simple":
        decoder = SimpleDecoder(
            action_dim=ckpt.get("action_dim", K * 2),
            hidden_dim=ckpt.get("hidden_dim", 256),
            num_layers=ckpt.get("num_layers", 4),
            num_heads=ckpt.get("num_heads", 4),
            condition_dim=ckpt.get("condition_dim", 896),
            accel_mean=ckpt.get("accel_mean", 0.0),
            accel_std=ckpt.get("accel_std", 1.0),
            kappa_mean=ckpt.get("kappa_mean", 0.0),
            kappa_std=ckpt.get("kappa_std", 1.0),
            use_action_norm=ckpt.get("use_action_norm", False),
        )
    elif arch == "cross_attention":
        decoder = CrossAttentionDecoder(
            K=K,
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            num_heads=ckpt["num_heads"],
            mlp_ratio=ckpt.get("mlp_ratio", 4),
            condition_dim=ckpt.get("condition_dim", 896),
            num_fourier_feats=ckpt.get("num_fourier_feats", 20),
            fourier_max_freq=ckpt.get("fourier_max_freq", 100.0),
            action_mlp_hidden=ckpt.get("action_mlp_hidden", 256),
            action_mlp_layers=ckpt.get("action_mlp_layers", 2),
            dropout=ckpt.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown architecture '{arch}' in checkpoint.")

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


# ============================================================
# Simple Decoder (旧実装: mean-pool + self-attn only)
# ============================================================


class SelfAttnFlowBlock(nn.Module):
    """Self-attention-only Transformer block with AdaLN (旧実装)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        # AdaLN: 4 params (scale1, shift1, scale2, shift2)
        self.adaLN = nn.Linear(dim, dim * 4)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        params = self.adaLN(t_emb).unsqueeze(1)  # (B, 1, dim*4)
        s1, b1, s2, b2 = params.chunk(4, dim=-1)

        h = self.norm1(x) * (1 + s1) + b1
        h, _ = self.attn(h, h, h)
        x = x + h

        h = self.norm2(x) * (1 + s2) + b2
        h = self.mlp(h)
        x = x + h
        return x


class SimpleDecoder(nn.Module):
    """旧実装の Flow Matching decoder (pre-54346b5).

    - Mean-pooled VLM hidden states → 1 condition vector
    - 2-token sequence [condition, action] で self-attention only
    - action は Linear projection (Fourier なし)
    - action 正規化なし
    """

    def __init__(
        self,
        action_dim: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        condition_dim: int = 896,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        kappa_mean: float = 0.0,
        kappa_std: float = 1.0,
        use_action_norm: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.K = action_dim // 2
        self.use_action_norm = use_action_norm

        self.register_buffer("accel_mean", torch.tensor(accel_mean))
        self.register_buffer("accel_std", torch.tensor(accel_std))
        self.register_buffer("kappa_mean", torch.tensor(kappa_mean))
        self.register_buffer("kappa_std", torch.tensor(kappa_std))

        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)

        self.blocks = nn.ModuleList(
            [SelfAttnFlowBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )

        self.norm_out = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, action_dim)

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        out = action.clone()
        out[:, 0::2] = (out[:, 0::2] - self.accel_mean) / self.accel_std
        out[:, 1::2] = (out[:, 1::2] - self.kappa_mean) / self.kappa_std
        return out

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        out = action.clone()
        out[:, 0::2] = out[:, 0::2] * self.accel_std + self.accel_mean
        out[:, 1::2] = out[:, 1::2] * self.kappa_std + self.kappa_mean
        return out

    def forward(
        self,
        a_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            a_t: (B, action_dim) noisy action
            t: (B,) timestep
            condition: (B, condition_dim) mean-pooled VLM hidden states
        Returns:
            v_theta: (B, action_dim) predicted velocity
        """
        h_action = self.action_proj(a_t).unsqueeze(1)  # (B, 1, hidden)
        h_cond = self.cond_proj(condition).unsqueeze(1)  # (B, 1, hidden)
        t_emb = self.time_emb(t)  # (B, hidden)

        x = torch.cat([h_cond, h_action], dim=1)  # (B, 2, hidden)

        for block in self.blocks:
            x = block(x, t_emb)

        x = self.norm_out(x[:, 1, :])  # action position
        return self.out_proj(x)  # (B, action_dim)


def cfm_loss_simple(
    model: SimpleDecoder,
    action_gt: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    """SimpleDecoder CFM loss. 正規化は use_action_norm フラグで制御."""
    B = action_gt.shape[0]
    device = action_gt.device

    if model.use_action_norm:
        action_gt = model.normalize(action_gt)

    t = torch.rand(B, device=device)
    epsilon = torch.randn_like(action_gt)

    a_t = t.unsqueeze(1) * action_gt + (1 - t.unsqueeze(1)) * epsilon
    u_target = action_gt - epsilon

    v_pred = model(a_t, t, condition)
    return (v_pred - u_target).pow(2).mean()


@torch.no_grad()
def cfm_sample_simple(
    model: SimpleDecoder,
    condition: torch.Tensor,
    n_steps: int = 10,
) -> torch.Tensor:
    """SimpleDecoder CFM sampling. 正規化ありの場合は denormalize して返す."""
    B = condition.shape[0]
    device = condition.device

    dt = 1.0 / n_steps
    a_t = torch.randn(B, model.action_dim, device=device)

    for i in range(n_steps):
        t = torch.full((B,), i / n_steps, device=device)
        v = model(a_t, t, condition)
        a_t = a_t + dt * v

    if model.use_action_norm:
        return model.denormalize(a_t)
    return a_t


# ============================================================
# Cross-Attention Decoder
# ============================================================


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding with MLP projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) → (B, dim)"""
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


class FlowTransformerBlock(nn.Module):
    """Transformer block with AdaLN + self-attention + cross-attention + FFN."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        # AdaLN: time_emb → 6 scale/shift params (2 per sub-layer)
        self.adaln = nn.Linear(dim, dim * 6)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, K, dim) action tokens
            t_emb: (B, dim) timestep embedding
            condition: (B, L, dim) projected VLM hidden states
        """
        params = self.adaln(t_emb).unsqueeze(1)  # (B, 1, dim*6)
        s1, b1, sc, bc, s2, b2 = params.chunk(6, dim=-1)

        # Self-attention with AdaLN
        h = self.norm1(x) * (1 + s1) + b1
        h, _ = self.attn(h, h, h)
        x = x + h

        # Cross-attention: Q=action tokens, K/V=condition (VLM hidden states)
        h = self.norm_cross(x) * (1 + sc) + bc
        h, _ = self.cross_attn(h, condition, condition)
        x = x + h

        # FFN with AdaLN
        h = self.norm2(x) * (1 + s2) + b2
        h = self.mlp(h)
        x = x + h

        return x


class CrossAttentionDecoder(nn.Module):
    """Cross-Attention Flow Matching decoder.

    Uses direct cross-attention to VLM hidden states (896 dim) instead of
    KV-cache, avoiding the KV-head bottleneck of the Qwen2.5-0.5B-based Expert.
    Default config: ~6M params (vs 146M for KV-cache Expert).
    """

    def __init__(
        self,
        K: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        condition_dim: int = 896,
        num_fourier_feats: int = 20,
        fourier_max_freq: float = 100.0,
        action_mlp_hidden: int = 256,
        action_mlp_layers: int = 2,
        dropout: float = 0.0,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        kappa_mean: float = 0.0,
        kappa_std: float = 1.0,
    ):
        super().__init__()
        self.K = K
        self.hidden_dim = hidden_dim

        # Action normalization (same as TrajectoryDecoder)
        self.register_buffer("accel_mean", torch.tensor(accel_mean))
        self.register_buffer("accel_std", torch.tensor(accel_std))
        self.register_buffer("kappa_mean", torch.tensor(kappa_mean))
        self.register_buffer("kappa_std", torch.tensor(kappa_std))

        # Condition projection: VLM hidden states → decoder dim
        self.cond_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Action input projection (reuse Fourier V2 + MLP)
        self.action_in_proj = ActionInProj(
            K=K,
            out_dim=hidden_dim,
            num_fourier_feats=num_fourier_feats,
            max_freq=fourier_max_freq,
            mlp_hidden_size=action_mlp_hidden,
            mlp_num_layers=action_mlp_layers,
        )

        # Timestep embedding
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                FlowTransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 2)  # per-waypoint (a, kappa)

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize raw action to zero-mean unit-variance."""
        out = action.clone()
        out[:, 0::2] = (out[:, 0::2] - self.accel_mean) / self.accel_std
        out[:, 1::2] = (out[:, 1::2] - self.kappa_mean) / self.kappa_std
        return out

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action back to physical units."""
        out = action.clone()
        out[:, 0::2] = out[:, 0::2] * self.accel_std + self.accel_mean
        out[:, 1::2] = out[:, 1::2] * self.kappa_std + self.kappa_mean
        return out

    def forward(
        self,
        a_t: torch.Tensor,
        t: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field.

        Args:
            a_t: (B, K*2) flat noisy action at time t
            t: (B,) timestep in [0, 1]
            hidden_states: (B, L, condition_dim) VLM hidden states

        Returns:
            v_theta: (B, K*2) predicted velocity field
        """
        B = a_t.shape[0]

        # Project condition
        condition = self.cond_proj(hidden_states)  # (B, L, hidden_dim)

        # Action embedding: (B, K*2) → (B, K, 2) → (B, K, hidden_dim)
        x = self.action_in_proj(a_t.view(B, self.K, 2), t)

        # Timestep embedding
        t_emb = self.time_emb(t)  # (B, hidden_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, condition)

        # Output: (B, K, hidden_dim) → (B, K, 2) → (B, K*2)
        x = self.norm_out(x)
        pred = self.out_proj(x)
        return pred.reshape(B, self.K * 2)


# ============================================================
# Cross-Attention CFM Loss and Sampling
# ============================================================


def cfm_loss_cross_attn(
    model: CrossAttentionDecoder,
    action_gt: torch.Tensor,
    hidden_states: torch.Tensor,
    beta_a: float = 2.0,
    beta_b: float = 5.0,
) -> torch.Tensor:
    """CFM loss for CrossAttentionDecoder.

    Same Gaussian OT path as cfm_loss, but conditioned on VLM hidden states
    instead of KV-cache.
    """
    B = action_gt.shape[0]
    device = action_gt.device

    action_norm = model.normalize(action_gt)
    t = torch.distributions.Beta(beta_a, beta_b).sample((B,)).to(device)
    epsilon = torch.randn_like(action_norm)

    a_t = t.unsqueeze(1) * action_norm + (1 - t.unsqueeze(1)) * epsilon
    u_target = action_norm - epsilon

    v_pred = model(a_t, t, hidden_states)
    return (v_pred - u_target).pow(2).mean()


@torch.no_grad()
def cfm_sample_cross_attn(
    model: CrossAttentionDecoder,
    hidden_states: torch.Tensor,
    n_steps: int = 10,
) -> torch.Tensor:
    """Sample trajectory for CrossAttentionDecoder via Euler integration."""
    B = hidden_states.shape[0]
    device = hidden_states.device
    action_dim = model.K * 2

    dt = 1.0 / n_steps
    a_t = torch.randn(B, action_dim, device=device)

    for i in range(n_steps):
        t = torch.full((B,), i / n_steps, device=device)
        v = model(a_t, t, hidden_states)
        a_t = a_t + dt * v

    return model.denormalize(a_t)
