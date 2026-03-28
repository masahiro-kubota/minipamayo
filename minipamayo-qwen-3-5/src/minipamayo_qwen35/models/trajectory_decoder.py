"""Hidden-state-conditioned Flow Matching decoder for the Qwen3.5 path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Simple RMSNorm used by the action projection MLP."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class FourierEncoderV2(nn.Module):
    """Log-spaced Fourier features for scalar inputs."""

    def __init__(self, dim: int = 20, max_freq: float = 100.0):
        super().__init__()
        half = dim // 2
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        self.register_buffer("freqs", freqs[None, :])
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arg = x[..., None] * self.freqs * 2 * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1) * math.sqrt(2.0)


class MLPEncoder(nn.Module):
    """Small MLP encoder used for per-waypoint action projection."""

    def __init__(self, num_input_feats: int, num_layers: int, hidden_size: int, out_dim: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(num_input_feats, hidden_size), nn.SiLU()]
        for layer_idx in range(num_layers):
            if layer_idx < num_layers - 1:
                layers.extend(
                    [
                        RMSNorm(hidden_size),
                        nn.Linear(hidden_size, hidden_size),
                        nn.SiLU(),
                    ]
                )
            else:
                layers.extend(
                    [
                        RMSNorm(hidden_size),
                        nn.Linear(hidden_size, out_dim),
                    ]
                )
        self.trunk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class ActionInProj(nn.Module):
    """Per-waypoint action plus timestep embedding."""

    def __init__(
        self,
        out_dim: int,
        num_fourier_feats: int = 20,
        max_freq: float = 100.0,
        mlp_hidden_size: int = 1024,
        mlp_num_layers: int = 4,
    ):
        super().__init__()
        self.action_fourier = nn.ModuleList(
            [FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq) for _ in range(2)]
        )
        self.timestep_fourier = FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq)
        input_dim = num_fourier_feats * 3
        self.encoder = MLPEncoder(
            num_input_feats=input_dim,
            num_layers=mlp_num_layers,
            hidden_size=mlp_hidden_size,
            out_dim=out_dim,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size, k_steps, _ = x.shape
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        action_feats = torch.cat(
            [encoder(x[:, :, dim_idx]) for dim_idx, encoder in enumerate(self.action_fourier)],
            dim=-1,
        )
        timestep_feats = self.timestep_fourier(t[:, -1]).unsqueeze(1).expand(-1, k_steps, -1)
        feats = torch.cat([action_feats, timestep_feats], dim=-1)
        projected = self.encoder(feats.flatten(0, 1))
        return self.norm(projected.reshape(batch_size, k_steps, -1))


@dataclass(frozen=True)
class DecoderConfig:
    k: int
    condition_dim: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    attention_dropout: float
    num_fourier_feats: int
    fourier_max_freq: float
    mlp_hidden_size: int
    mlp_num_layers: int


class TrajectoryDecoder(nn.Module):
    """Cross-attends from action tokens into Qwen3.5 final hidden states."""

    def __init__(
        self,
        *,
        k: int,
        condition_dim: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        attention_dropout: float = 0.0,
        num_fourier_feats: int = 20,
        fourier_max_freq: float = 100.0,
        mlp_hidden_size: int = 1024,
        mlp_num_layers: int = 4,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        kappa_mean: float = 0.0,
        kappa_std: float = 1.0,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise RuntimeError(
                "`hidden_size` must be divisible by `num_attention_heads` for the trajectory decoder."
            )
        if accel_std <= 0.0 or kappa_std <= 0.0:
            raise RuntimeError("Action normalization std must be strictly positive.")

        self.k = k
        self.condition_dim = condition_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.num_fourier_feats = num_fourier_feats
        self.fourier_max_freq = fourier_max_freq
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_num_layers = mlp_num_layers

        self.register_buffer("accel_mean", torch.tensor(float(accel_mean), dtype=torch.float32))
        self.register_buffer("accel_std", torch.tensor(float(accel_std), dtype=torch.float32))
        self.register_buffer("kappa_mean", torch.tensor(float(kappa_mean), dtype=torch.float32))
        self.register_buffer("kappa_std", torch.tensor(float(kappa_std), dtype=torch.float32))

        self.action_in_proj = ActionInProj(
            out_dim=hidden_size,
            num_fourier_feats=num_fourier_feats,
            max_freq=fourier_max_freq,
            mlp_hidden_size=mlp_hidden_size,
            mlp_num_layers=mlp_num_layers,
        )
        self.condition_proj = nn.Linear(condition_dim, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.action_out_proj = nn.Linear(hidden_size, 2)

    @property
    def action_dim(self) -> int:
        return self.k * 2

    def export_config(self) -> DecoderConfig:
        return DecoderConfig(
            k=self.k,
            condition_dim=self.condition_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            attention_dropout=self.attention_dropout,
            num_fourier_feats=self.num_fourier_feats,
            fourier_max_freq=self.fourier_max_freq,
            mlp_hidden_size=self.mlp_hidden_size,
            mlp_num_layers=self.mlp_num_layers,
        )

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        out = action.to(torch.float32).clone()
        out[:, 0::2] = (out[:, 0::2] - self.accel_mean) / self.accel_std
        out[:, 1::2] = (out[:, 1::2] - self.kappa_mean) / self.kappa_std
        return out

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        out = action.to(torch.float32).clone()
        out[:, 0::2] = out[:, 0::2] * self.accel_std + self.accel_mean
        out[:, 1::2] = out[:, 1::2] * self.kappa_std + self.kappa_mean
        return out

    def forward(
        self,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
        condition_hidden_states: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_action.shape[-1] != self.action_dim:
            raise RuntimeError(
                f"Expected action dim {self.action_dim}, got {noisy_action.shape[-1]}."
            )
        if condition_hidden_states.shape[-1] != self.condition_dim:
            raise RuntimeError(
                "Condition hidden states do not match decoder condition_dim: "
                f"{condition_hidden_states.shape[-1]} vs {self.condition_dim}."
            )

        batch_size = noisy_action.shape[0]
        noisy_action = noisy_action.reshape(batch_size, self.k, 2)
        action_tokens = self.action_in_proj(noisy_action, t)
        memory = self.condition_proj(condition_hidden_states.to(action_tokens.dtype))

        memory_key_padding_mask = None
        if condition_mask is not None:
            memory_key_padding_mask = ~condition_mask.to(dtype=torch.bool)

        hidden = self.decoder(
            tgt=action_tokens,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        output = self.action_out_proj(self.output_norm(hidden))
        return output.reshape(batch_size, self.action_dim)


def cfm_loss(
    decoder: TrajectoryDecoder,
    gt_action: torch.Tensor,
    condition_hidden_states: torch.Tensor,
    condition_mask: torch.Tensor | None = None,
    *,
    beta_alpha: float = 2.0,
    beta_beta: float = 5.0,
) -> torch.Tensor:
    """Conditional Flow Matching loss in normalized action space."""

    normalized_gt_action = decoder.normalize(gt_action)
    noise = torch.randn_like(normalized_gt_action)
    beta_dist = torch.distributions.Beta(beta_alpha, beta_beta)
    t = beta_dist.sample((gt_action.shape[0],)).to(device=gt_action.device, dtype=torch.float32)
    mixed = t.unsqueeze(-1) * normalized_gt_action + (1.0 - t.unsqueeze(-1)) * noise
    target = normalized_gt_action - noise
    pred = decoder(
        noisy_action=mixed,
        t=t,
        condition_hidden_states=condition_hidden_states,
        condition_mask=condition_mask,
    )
    return torch.mean((pred - target) ** 2)


@torch.no_grad()
def cfm_sample(
    decoder: TrajectoryDecoder,
    condition_hidden_states: torch.Tensor,
    condition_mask: torch.Tensor | None = None,
    *,
    n_steps: int = 10,
) -> torch.Tensor:
    """Euler integration in normalized action space."""

    if n_steps <= 0:
        raise RuntimeError("`n_steps` must be > 0 for Flow Matching sampling.")

    batch_size = condition_hidden_states.shape[0]
    current = torch.randn(
        batch_size,
        decoder.action_dim,
        device=condition_hidden_states.device,
        dtype=condition_hidden_states.dtype,
    )
    dt = 1.0 / float(n_steps)

    for step_idx in range(n_steps):
        t = torch.full(
            (batch_size,),
            fill_value=float(step_idx) / float(n_steps),
            device=current.device,
            dtype=torch.float32,
        )
        velocity = decoder(
            noisy_action=current,
            t=t,
            condition_hidden_states=condition_hidden_states,
            condition_mask=condition_mask,
        )
        current = current + dt * velocity

    return decoder.denormalize(current)


def load_decoder_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[TrajectoryDecoder, dict]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    if "decoder_config" not in checkpoint:
        raise RuntimeError("Stage 2 checkpoint is missing canonical `decoder_config` metadata.")
    if "action_stats" not in checkpoint:
        raise RuntimeError("Stage 2 checkpoint is missing canonical `action_stats` metadata.")
    decoder_config = checkpoint["decoder_config"]
    action_stats = checkpoint["action_stats"]
    required_cfg_keys = [
        "k",
        "condition_dim",
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "intermediate_size",
        "attention_dropout",
        "num_fourier_feats",
        "fourier_max_freq",
        "mlp_hidden_size",
        "mlp_num_layers",
    ]
    missing_cfg_keys = [key for key in required_cfg_keys if key not in decoder_config]
    if missing_cfg_keys:
        raise RuntimeError(
            "Stage 2 checkpoint is missing canonical decoder config fields:\n"
            + "\n".join(missing_cfg_keys)
        )
    required_stat_keys = ["accel_mean", "accel_std", "kappa_mean", "kappa_std"]
    missing_stat_keys = [key for key in required_stat_keys if key not in action_stats]
    if missing_stat_keys:
        raise RuntimeError(
            "Stage 2 checkpoint is missing canonical action stats fields:\n"
            + "\n".join(missing_stat_keys)
        )

    decoder = TrajectoryDecoder(
        k=int(decoder_config["k"]),
        condition_dim=int(decoder_config["condition_dim"]),
        hidden_size=int(decoder_config["hidden_size"]),
        num_layers=int(decoder_config["num_layers"]),
        num_attention_heads=int(decoder_config["num_attention_heads"]),
        intermediate_size=int(decoder_config["intermediate_size"]),
        attention_dropout=float(decoder_config["attention_dropout"]),
        num_fourier_feats=int(decoder_config["num_fourier_feats"]),
        fourier_max_freq=float(decoder_config["fourier_max_freq"]),
        mlp_hidden_size=int(decoder_config["mlp_hidden_size"]),
        mlp_num_layers=int(decoder_config["mlp_num_layers"]),
        accel_mean=float(action_stats["accel_mean"]),
        accel_std=float(action_stats["accel_std"]),
        kappa_mean=float(action_stats["kappa_mean"]),
        kappa_std=float(action_stats["kappa_std"]),
    )
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    decoder.to(device)
    decoder.eval()
    return decoder, checkpoint
