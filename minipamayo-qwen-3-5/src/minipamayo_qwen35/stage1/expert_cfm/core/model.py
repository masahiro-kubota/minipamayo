"""Alpamayo-style canonical Stage 1B action expert."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from ....models.action_in_proj import PerWaypointActionInProjV2


@dataclass(frozen=True)
class Stage1ActionExpertConfig:
    k: int
    action_dims: tuple[int, int]
    expert_text_config: dict
    expert_cfg: dict
    action_in_proj_cfg: dict
    action_out_proj_cfg: dict
    keep_same_dtype: bool
    expert_non_causal_attention: bool
    num_fourier_feats: int
    fourier_max_freq: float
    mlp_hidden_size: int
    mlp_num_layers: int


def build_text_config(config_dict: dict):
    payload = dict(config_dict)
    if "model_type" not in payload:
        raise RuntimeError("Expert text config is missing `model_type`.")
    model_type = payload.pop("model_type")
    return AutoConfig.for_model(model_type, **payload)


def prompt_cache_seq_length(prompt_cache, fallback_seq_len: int) -> int:
    if hasattr(prompt_cache, "get_seq_length"):
        seq_len = int(prompt_cache.get_seq_length())
        if seq_len > 0:
            return seq_len
    return int(fallback_seq_len)


def clone_prompt_cache_for_expert(prompt_cache, num_layers: int):
    if hasattr(prompt_cache, "key_cache") and hasattr(prompt_cache, "value_cache"):
        if len(prompt_cache.key_cache) < num_layers or len(prompt_cache.value_cache) < num_layers:
            raise RuntimeError(
                "Prompt cache does not have enough layers for the configured expert.\n"
                f"prompt_cache_layers={len(prompt_cache.key_cache)}\n"
                f"expert_layers={num_layers}"
            )
        cloned = copy.deepcopy(prompt_cache)
        cloned.key_cache = list(cloned.key_cache[:num_layers])
        cloned.value_cache = list(cloned.value_cache[:num_layers])
        return cloned
    if isinstance(prompt_cache, tuple):
        if len(prompt_cache) < num_layers:
            raise RuntimeError(
                "Prompt cache tuple does not have enough layers for the configured expert.\n"
                f"prompt_cache_layers={len(prompt_cache)}\n"
                f"expert_layers={num_layers}"
            )
        return tuple(prompt_cache[:num_layers])
    if isinstance(prompt_cache, list):
        if len(prompt_cache) < num_layers:
            raise RuntimeError(
                "Prompt cache list does not have enough layers for the configured expert.\n"
                f"prompt_cache_layers={len(prompt_cache)}\n"
                f"expert_layers={num_layers}"
            )
        return list(prompt_cache[:num_layers])
    raise RuntimeError(f"Unsupported prompt cache type for Stage 1B expert: {type(prompt_cache)!r}")


def build_expert_position_ids(
    prompt_attention_mask: torch.Tensor,
    future_token_count: int,
) -> torch.Tensor:
    batch_size = prompt_attention_mask.shape[0]
    prompt_lengths = prompt_attention_mask.to(dtype=torch.long).sum(dim=1)
    position_ids = torch.arange(
        future_token_count,
        device=prompt_attention_mask.device,
        dtype=torch.long,
    )
    position_ids = position_ids.view(1, 1, future_token_count).expand(
        3, batch_size, future_token_count
    )
    return position_ids + prompt_lengths.view(1, batch_size, 1)


def build_expert_attention_mask(
    prompt_attention_mask: torch.Tensor,
    prefill_seq_len: int,
    future_token_count: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size = prompt_attention_mask.shape[0]
    mask = torch.zeros(
        (batch_size, 1, future_token_count, prefill_seq_len + future_token_count),
        device=prompt_attention_mask.device,
        dtype=dtype,
    )
    prompt_lengths = prompt_attention_mask.to(dtype=torch.long).sum(dim=1)
    neg_inf = torch.finfo(mask.dtype).min
    for row_idx, prompt_len in enumerate(prompt_lengths.tolist()):
        if prompt_len < prefill_seq_len:
            mask[row_idx, :, :, prompt_len:prefill_seq_len] = neg_inf
    return mask


class Stage1ActionExpert(nn.Module):
    """Prompt-cache-conditioned action expert aligned with public Alpamayo structure."""

    def __init__(
        self,
        *,
        k: int,
        expert_text_config: dict,
        action_dims: tuple[int, int] = (64, 2),
        expert_non_causal_attention: bool = True,
        num_fourier_feats: int = 20,
        fourier_max_freq: float = 100.0,
        mlp_hidden_size: int = 1024,
        mlp_num_layers: int = 4,
        keep_same_dtype: bool = True,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        kappa_mean: float = 0.0,
        kappa_std: float = 1.0,
    ):
        super().__init__()
        if action_dims[-1] != 2:
            raise RuntimeError(f"Canonical Stage 1B expects 2 action channels, got {action_dims}.")
        if action_dims[0] != k:
            raise RuntimeError(
                "Canonical Stage 1B action token count must match k.\n"
                f"k={k}\n"
                f"action_dims={action_dims}"
            )
        if accel_std <= 0.0 or kappa_std <= 0.0:
            raise RuntimeError("Action normalization std must be strictly positive.")

        self.k = int(k)
        self.action_dims = tuple(int(x) for x in action_dims)
        self.expert_non_causal_attention = bool(expert_non_causal_attention)
        self.num_fourier_feats = int(num_fourier_feats)
        self.fourier_max_freq = float(fourier_max_freq)
        self.mlp_hidden_size = int(mlp_hidden_size)
        self.mlp_num_layers = int(mlp_num_layers)
        self.keep_same_dtype = bool(keep_same_dtype)
        self.expert_text_config = dict(expert_text_config)

        text_config = build_text_config(expert_text_config)
        self.expert = AutoModel.from_config(text_config)
        if hasattr(self.expert, "embed_tokens"):
            del self.expert.embed_tokens

        hidden_size = int(text_config.hidden_size)
        self.expert_num_layers = int(text_config.num_hidden_layers)
        self.action_in_proj = PerWaypointActionInProjV2(
            in_dims=list(self.action_dims),
            out_dim=hidden_size,
            num_enc_layers=self.mlp_num_layers,
            hidden_size=self.mlp_hidden_size,
            max_freq=self.fourier_max_freq,
            num_fourier_feats=self.num_fourier_feats,
        )
        self.action_out_proj = nn.Linear(hidden_size, self.action_dims[-1])

        if self.keep_same_dtype:
            expert_dtype = self.expert.dtype
            self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
            self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)

        self.register_buffer("accel_mean", torch.tensor(float(accel_mean), dtype=torch.float32))
        self.register_buffer("accel_std", torch.tensor(float(accel_std), dtype=torch.float32))
        self.register_buffer("kappa_mean", torch.tensor(float(kappa_mean), dtype=torch.float32))
        self.register_buffer("kappa_std", torch.tensor(float(kappa_std), dtype=torch.float32))

    @property
    def action_dim(self) -> int:
        return self.k * self.action_dims[-1]

    def export_config(self) -> Stage1ActionExpertConfig:
        return Stage1ActionExpertConfig(
            k=self.k,
            action_dims=self.action_dims,
            expert_text_config=self.expert_text_config,
            expert_cfg=dict(self.expert_text_config),
            action_in_proj_cfg={
                "_target_": "minipamayo_qwen35.models.action_in_proj.PerWaypointActionInProjV2",
                "num_enc_layers": self.mlp_num_layers,
                "hidden_size": self.mlp_hidden_size,
                "max_freq": self.fourier_max_freq,
                "num_fourier_feats": self.num_fourier_feats,
            },
            action_out_proj_cfg={
                "_target_": "torch.nn.Linear",
            },
            keep_same_dtype=self.keep_same_dtype,
            expert_non_causal_attention=self.expert_non_causal_attention,
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
        *,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
        prompt_cache,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_action.shape[-1] != self.action_dim:
            raise RuntimeError(
                f"Expected action dim {self.action_dim}, got {noisy_action.shape[-1]}."
            )
        if prompt_attention_mask.dim() != 2:
            raise RuntimeError(
                "Stage 1B expert expects a 2D prompt attention mask.\n"
                f"prompt_attention_mask.shape={tuple(prompt_attention_mask.shape)!r}"
            )

        batch_size = noisy_action.shape[0]
        noisy_action = noisy_action.reshape(batch_size, self.k, self.action_dims[-1])
        future_token_embeds = self.action_in_proj(noisy_action, t)
        expert_dtype = next(self.expert.parameters()).dtype
        future_token_embeds = future_token_embeds.to(
            device=prompt_attention_mask.device,
            dtype=expert_dtype,
        )

        expert_prompt_cache = clone_prompt_cache_for_expert(prompt_cache, self.expert_num_layers)
        prefill_seq_len = prompt_cache_seq_length(
            expert_prompt_cache, prompt_attention_mask.shape[1]
        )
        position_ids = build_expert_position_ids(prompt_attention_mask, self.k)
        attention_mask = build_expert_attention_mask(
            prompt_attention_mask,
            prefill_seq_len,
            self.k,
            dtype=expert_dtype,
        )
        forward_kwargs = {}
        if self.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        expert_out = self.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=expert_prompt_cache,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            **forward_kwargs,
        )
        if hasattr(expert_prompt_cache, "crop"):
            expert_prompt_cache.crop(prefill_seq_len)
        last_hidden = expert_out.last_hidden_state[:, -self.k :]
        last_hidden = last_hidden.to(dtype=self.action_out_proj.weight.dtype)
        pred = self.action_out_proj(last_hidden).reshape(batch_size, self.action_dim)
        return pred


def cfm_loss(
    expert: Stage1ActionExpert,
    gt_action: torch.Tensor,
    prompt_cache,
    prompt_attention_mask: torch.Tensor,
    *,
    beta_alpha: float = 2.0,
    beta_beta: float = 5.0,
) -> torch.Tensor:
    normalized_gt_action = expert.normalize(gt_action)
    noise = torch.randn_like(normalized_gt_action)
    beta_dist = torch.distributions.Beta(beta_alpha, beta_beta)
    t = beta_dist.sample((gt_action.shape[0],)).to(device=gt_action.device, dtype=torch.float32)
    mixed = t.unsqueeze(-1) * normalized_gt_action + (1.0 - t.unsqueeze(-1)) * noise
    target = normalized_gt_action - noise
    pred = expert(
        noisy_action=mixed,
        t=t,
        prompt_cache=prompt_cache,
        prompt_attention_mask=prompt_attention_mask,
    )
    return torch.mean((pred - target) ** 2)


@torch.no_grad()
def cfm_sample(
    expert: Stage1ActionExpert,
    prompt_cache,
    prompt_attention_mask: torch.Tensor,
    *,
    n_steps: int = 10,
) -> torch.Tensor:
    if n_steps <= 0:
        raise RuntimeError("`n_steps` must be > 0 for Flow Matching sampling.")
    batch_size = prompt_attention_mask.shape[0]
    current = torch.randn(
        batch_size,
        expert.action_dim,
        device=prompt_attention_mask.device,
        dtype=expert.action_out_proj.weight.dtype,
    )
    dt = 1.0 / float(n_steps)
    for step_idx in range(n_steps):
        t = torch.full(
            (batch_size,),
            fill_value=float(step_idx) / float(n_steps),
            device=current.device,
            dtype=torch.float32,
        )
        velocity = expert(
            noisy_action=current,
            t=t,
            prompt_cache=prompt_cache,
            prompt_attention_mask=prompt_attention_mask,
        )
        current = current + dt * velocity
    return expert.denormalize(current)


def load_action_expert_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[Stage1ActionExpert, dict]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if "expert_config" not in checkpoint:
        if "decoder_config" in checkpoint:
            raise RuntimeError(
                "This Stage 1B checkpoint uses the old cross-attention decoder format. "
                "Train a new canonical Stage 1B expert checkpoint."
            )
        raise RuntimeError("Stage 1B checkpoint is missing canonical `expert_config` metadata.")
    if "action_stats" not in checkpoint:
        raise RuntimeError("Stage 1B checkpoint is missing canonical `action_stats` metadata.")
    if "expert_state_dict" not in checkpoint:
        raise RuntimeError("Stage 1B checkpoint is missing canonical `expert_state_dict`.")
    expert_config = checkpoint["expert_config"]
    action_stats = checkpoint["action_stats"]
    required_cfg_keys = [
        "k",
        "action_dims",
        "expert_text_config",
        "expert_non_causal_attention",
        "num_fourier_feats",
        "fourier_max_freq",
        "mlp_hidden_size",
        "mlp_num_layers",
    ]
    missing_cfg_keys = [key for key in required_cfg_keys if key not in expert_config]
    if missing_cfg_keys:
        raise RuntimeError(
            "Stage 1B checkpoint expert_config is missing canonical fields:\n"
            + "\n".join(missing_cfg_keys)
        )
    required_stats_keys = ["accel_mean", "accel_std", "kappa_mean", "kappa_std"]
    missing_stats_keys = [key for key in required_stats_keys if key not in action_stats]
    if missing_stats_keys:
        raise RuntimeError(
            "Stage 1B checkpoint action_stats is missing canonical fields:\n"
            + "\n".join(missing_stats_keys)
        )
    expert = Stage1ActionExpert(
        k=int(expert_config["k"]),
        action_dims=tuple(int(x) for x in expert_config["action_dims"]),
        expert_text_config=dict(expert_config["expert_text_config"]),
        expert_non_causal_attention=bool(expert_config["expert_non_causal_attention"]),
        num_fourier_feats=int(expert_config["num_fourier_feats"]),
        fourier_max_freq=float(expert_config["fourier_max_freq"]),
        mlp_hidden_size=int(expert_config["mlp_hidden_size"]),
        mlp_num_layers=int(expert_config["mlp_num_layers"]),
        keep_same_dtype=bool(expert_config.get("keep_same_dtype", True)),
        accel_mean=float(action_stats["accel_mean"]),
        accel_std=float(action_stats["accel_std"]),
        kappa_mean=float(action_stats["kappa_mean"]),
        kappa_std=float(action_stats["kappa_std"]),
    ).to(device)
    expert.load_state_dict(checkpoint["expert_state_dict"])
    expert.eval()
    return expert, checkpoint
