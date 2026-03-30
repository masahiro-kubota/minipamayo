"""Alpamayo-style canonical Stage 1B action expert."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .action_in_proj import PerWaypointActionInProjV2
from ..diffusion.flow_matching import FlowMatching


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


@dataclass(frozen=True)
class Stage1ExpertConditioning:
    """Precomputed expert conditioning shared across detached diffusion steps."""

    prompt_cache: object
    position_ids: torch.Tensor
    attention_mask: torch.Tensor


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


def _validate_prompt_cache_layers(prompt_cache, num_layers: int) -> None:
    if hasattr(prompt_cache, "key_cache") and hasattr(prompt_cache, "value_cache"):
        if len(prompt_cache.key_cache) < num_layers or len(prompt_cache.value_cache) < num_layers:
            raise RuntimeError(
                "Prompt cache does not have enough layers for the configured expert.\n"
                f"prompt_cache_layers={len(prompt_cache.key_cache)}\n"
                f"expert_layers={num_layers}"
            )
        return
    if isinstance(prompt_cache, tuple):
        if len(prompt_cache) < num_layers:
            raise RuntimeError(
                "Prompt cache tuple does not have enough layers for the configured expert.\n"
                f"prompt_cache_layers={len(prompt_cache)}\n"
                f"expert_layers={num_layers}"
            )
        return
    if isinstance(prompt_cache, list):
        if len(prompt_cache) < num_layers:
            raise RuntimeError(
                "Prompt cache list does not have enough layers for the configured expert.\n"
                f"prompt_cache_layers={len(prompt_cache)}\n"
                f"expert_layers={num_layers}"
            )
        return
    raise RuntimeError(f"Unsupported prompt cache type for Stage 1B expert: {type(prompt_cache)!r}")


def clone_prompt_cache_for_expert(prompt_cache, num_layers: int):
    _validate_prompt_cache_layers(prompt_cache, num_layers)
    if hasattr(prompt_cache, "key_cache") and hasattr(prompt_cache, "value_cache"):
        cloned = copy.deepcopy(prompt_cache)
        cloned.key_cache = list(cloned.key_cache[:num_layers])
        cloned.value_cache = list(cloned.value_cache[:num_layers])
        return cloned
    if isinstance(prompt_cache, tuple):
        return tuple(prompt_cache[:num_layers])
    if isinstance(prompt_cache, list):
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
) -> torch.Tensor:
    prompt_mask = prompt_attention_mask.to(dtype=torch.long)
    batch_size = prompt_mask.shape[0]
    if prompt_mask.shape[1] < prefill_seq_len:
        prompt_mask = torch.cat(
            [
                prompt_mask,
                torch.zeros(
                    (batch_size, prefill_seq_len - prompt_mask.shape[1]),
                    device=prompt_mask.device,
                    dtype=prompt_mask.dtype,
                ),
            ],
            dim=1,
        )
    elif prompt_mask.shape[1] > prefill_seq_len:
        prompt_mask = prompt_mask[:, :prefill_seq_len]

    future_mask = torch.ones(
        (batch_size, future_token_count),
        device=prompt_mask.device,
        dtype=prompt_mask.dtype,
    )
    return torch.cat([prompt_mask, future_mask], dim=1)


def build_expert_attention_mask_from_offsets(
    offsets: torch.Tensor,
    prefill_seq_len: int,
    future_token_count: int,
) -> torch.Tensor:
    batch_size = offsets.shape[0]
    prompt_mask = torch.zeros(
        (batch_size, prefill_seq_len),
        device=offsets.device,
        dtype=torch.long,
    )
    for row_idx, offset in enumerate(offsets.tolist()):
        prompt_mask[row_idx, : max(0, min(int(offset), prefill_seq_len))] = 1
    return build_expert_attention_mask(prompt_mask, prefill_seq_len, future_token_count)


def reshape_action_for_expert(
    action: torch.Tensor,
    *,
    action_dims: tuple[int, int],
    action_dim: int,
) -> torch.Tensor:
    """Canonicalize flat or structured Stage 1B actions to action-space shape."""

    if action.dim() >= 2 and tuple(action.shape[-2:]) == action_dims:
        return action
    if action.shape[-1] == action_dim:
        return action.reshape(*action.shape[:-1], *action_dims)
    raise RuntimeError(
        "Stage 1B action tensor does not match the canonical action-space contract.\n"
        f"expected_flat_last_dim={action_dim}\n"
        f"expected_structured_suffix={action_dims!r}\n"
        f"found={tuple(action.shape)!r}"
    )


def restore_action_shape(
    structured_action: torch.Tensor,
    *,
    reference: torch.Tensor,
    action_dims: tuple[int, int],
    action_dim: int,
) -> torch.Tensor:
    """Restore an action-space tensor to the caller's original flat/structured shape."""

    if reference.dim() >= 2 and tuple(reference.shape[-2:]) == action_dims:
        return structured_action
    if reference.shape[-1] == action_dim:
        return structured_action.reshape(*reference.shape[:-1], action_dim)
    raise RuntimeError(
        "Stage 1B action tensor does not match the canonical action-space contract.\n"
        f"expected_flat_last_dim={action_dim}\n"
        f"expected_structured_suffix={action_dims!r}\n"
        f"found={tuple(reference.shape)!r}"
    )


def prepare_expert_conditioning(
    *,
    prompt_cache,
    prompt_attention_mask: torch.Tensor,
    future_token_count: int,
    num_layers: int,
) -> Stage1ExpertConditioning:
    """Precompute the fixed expert-side conditioning tensors for detached decoding."""

    if prompt_attention_mask.dim() != 2:
        raise RuntimeError(
            "Stage 1B expert expects a 2D prompt attention mask.\n"
            f"prompt_attention_mask.shape={tuple(prompt_attention_mask.shape)!r}"
        )
    _validate_prompt_cache_layers(prompt_cache, num_layers)
    prefill_seq_len = prompt_cache_seq_length(prompt_cache, prompt_attention_mask.shape[1])
    return Stage1ExpertConditioning(
        prompt_cache=prompt_cache,
        position_ids=build_expert_position_ids(prompt_attention_mask, future_token_count),
        attention_mask=build_expert_attention_mask(
            prompt_attention_mask,
            prefill_seq_len,
            future_token_count,
        ),
    )


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
        structured_action = reshape_action_for_expert(
            action,
            action_dims=self.action_dims,
            action_dim=self.action_dim,
        )
        out = structured_action.to(torch.float32).clone()
        out[..., 0] = (out[..., 0] - self.accel_mean) / self.accel_std
        out[..., 1] = (out[..., 1] - self.kappa_mean) / self.kappa_std
        return restore_action_shape(
            out,
            reference=action,
            action_dims=self.action_dims,
            action_dim=self.action_dim,
        )

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        structured_action = reshape_action_for_expert(
            action,
            action_dims=self.action_dims,
            action_dim=self.action_dim,
        )
        out = structured_action.to(torch.float32).clone()
        out[..., 0] = out[..., 0] * self.accel_std + self.accel_mean
        out[..., 1] = out[..., 1] * self.kappa_std + self.kappa_mean
        return restore_action_shape(
            out,
            reference=action,
            action_dims=self.action_dims,
            action_dim=self.action_dim,
        )

    def prepare_conditioning(
        self,
        *,
        prompt_cache,
        prompt_attention_mask: torch.Tensor,
    ) -> Stage1ExpertConditioning:
        return prepare_expert_conditioning(
            prompt_cache=prompt_cache,
            prompt_attention_mask=prompt_attention_mask,
            future_token_count=self.k,
            num_layers=self.expert_num_layers,
        )

    def forward_with_conditioning(
        self,
        *,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
        conditioning: Stage1ExpertConditioning,
    ) -> torch.Tensor:
        if conditioning.attention_mask.dim() != 2:
            raise RuntimeError(
                "Stage 1B expert expects a 2D expert attention mask.\n"
                f"conditioning.attention_mask.shape={tuple(conditioning.attention_mask.shape)!r}"
            )
        if conditioning.position_ids.dim() != 3:
            raise RuntimeError(
                "Stage 1B expert expects 3D position ids.\n"
                f"conditioning.position_ids.shape={tuple(conditioning.position_ids.shape)!r}"
            )

        structured_noisy_action = reshape_action_for_expert(
            noisy_action,
            action_dims=self.action_dims,
            action_dim=self.action_dim,
        )
        batch_size = structured_noisy_action.shape[0]
        if conditioning.attention_mask.shape[0] != batch_size:
            raise RuntimeError(
                "Stage 1B expert batch size does not match the expert conditioning.\n"
                f"batch_size={batch_size}\n"
                f"conditioning.attention_mask.shape={tuple(conditioning.attention_mask.shape)!r}"
            )
        future_token_embeds = self.action_in_proj(structured_noisy_action, t)

        # Qwen3_5DynamicCache does not implement crop(), so keep the prefill
        # cache immutable and hand each expert call its own clone.
        expert_prompt_cache = clone_prompt_cache_for_expert(
            conditioning.prompt_cache, self.expert_num_layers
        )
        forward_kwargs = {}
        if self.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        expert_out = self.expert(
            inputs_embeds=future_token_embeds,
            position_ids=conditioning.position_ids,
            past_key_values=expert_prompt_cache,
            attention_mask=conditioning.attention_mask,
            use_cache=True,
            return_dict=True,
            **forward_kwargs,
        )
        last_hidden = expert_out.last_hidden_state[:, -self.k :]
        pred = self.action_out_proj(last_hidden)
        return restore_action_shape(
            pred,
            reference=noisy_action,
            action_dims=self.action_dims,
            action_dim=self.action_dim,
        )

    def forward(
        self,
        *,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
        prompt_cache,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        conditioning = self.prepare_conditioning(
            prompt_cache=prompt_cache,
            prompt_attention_mask=prompt_attention_mask,
        )
        return self.forward_with_conditioning(
            noisy_action=noisy_action,
            t=t,
            conditioning=conditioning,
        )


def cfm_loss(
    expert: Stage1ActionExpert,
    gt_action: torch.Tensor,
    prompt_cache,
    prompt_attention_mask: torch.Tensor,
    *,
    beta_alpha: float = 2.0,
    beta_beta: float = 5.0,
) -> torch.Tensor:
    conditioning = expert.prepare_conditioning(
        prompt_cache=prompt_cache,
        prompt_attention_mask=prompt_attention_mask,
    )
    return flow_matching_loss(
        expert=expert,
        gt_action=gt_action,
        conditioning=conditioning,
        beta_alpha=beta_alpha,
        beta_beta=beta_beta,
    )


def flow_matching_loss(
    *,
    expert: Stage1ActionExpert,
    gt_action: torch.Tensor,
    conditioning: Stage1ExpertConditioning,
    beta_alpha: float = 2.0,
    beta_beta: float = 5.0,
) -> torch.Tensor:
    gt_action = reshape_action_for_expert(
        gt_action,
        action_dims=expert.action_dims,
        action_dim=expert.action_dim,
    )
    normalized_gt_action = expert.normalize(gt_action)
    noise = torch.randn_like(normalized_gt_action)
    beta_dist = torch.distributions.Beta(beta_alpha, beta_beta)
    t = beta_dist.sample((gt_action.shape[0],)).to(device=gt_action.device, dtype=torch.float32)
    t_column, t_expert = reshape_flow_matching_timesteps(t, batch_size=gt_action.shape[0])
    t_mixed = t_column
    while t_mixed.dim() < normalized_gt_action.dim():
        t_mixed = t_mixed.unsqueeze(-1)
    mixed = t_mixed * normalized_gt_action + (1.0 - t_mixed) * noise
    target = normalized_gt_action - noise
    pred = expert.forward_with_conditioning(
        noisy_action=mixed,
        t=t_expert,
        conditioning=conditioning,
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
    conditioning = expert.prepare_conditioning(
        prompt_cache=prompt_cache,
        prompt_attention_mask=prompt_attention_mask,
    )
    return flow_matching_sample(
        expert=expert,
        conditioning=conditioning,
        n_steps=n_steps,
    )


@torch.no_grad()
def flow_matching_sample(
    *,
    expert: Stage1ActionExpert,
    conditioning: Stage1ExpertConditioning,
    n_steps: int = 10,
) -> torch.Tensor:
    if n_steps <= 0:
        raise RuntimeError("`n_steps` must be > 0 for Flow Matching sampling.")
    sampler = FlowMatching(
        x_dims=expert.action_dims,
        num_inference_steps=n_steps,
    )

    def step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _, t_expert = reshape_flow_matching_timesteps(t, batch_size=x.shape[0])
        return expert.forward_with_conditioning(
            noisy_action=x,
            t=t_expert,
            conditioning=conditioning,
        )

    current = sampler.sample(
        batch_size=conditioning.attention_mask.shape[0],
        step_fn=step_fn,
        device=conditioning.attention_mask.device,
    )
    return expert.denormalize(current)


def reshape_flow_matching_timesteps(
    t: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize Flow Matching timestep tensors for both mixing and expert conditioning.

    The Stage 1B expert path expects a single diffusion timestep per batch item, but
    call sites may carry it as `[B]`, `[B, 1]`, or `[B, 1, 1]` depending on whether
    the source is the loss path or the Euler sampler.
    """

    t = t.to(dtype=torch.float32)
    if t.shape[0] != batch_size:
        raise RuntimeError(
            "Flow Matching timestep batch size does not match the expert batch.\n"
            f"batch_size={batch_size}\n"
            f"t.shape={tuple(t.shape)!r}"
        )
    if t.dim() == 1:
        return t.unsqueeze(-1), t.view(batch_size, 1, 1)
    if t.dim() == 2:
        if t.shape[1] != 1:
            raise RuntimeError(
                "Flow Matching timestep tensor must carry exactly one scalar per sample.\n"
                f"t.shape={tuple(t.shape)!r}"
            )
        return t, t.unsqueeze(-1)
    if t.dim() == 3:
        if tuple(t.shape[1:]) != (1, 1):
            raise RuntimeError(
                "Flow Matching timestep tensor must carry exactly one scalar per sample.\n"
                f"t.shape={tuple(t.shape)!r}"
            )
        return t.view(batch_size, 1), t
    raise RuntimeError(
        "Unsupported Flow Matching timestep rank for Stage 1B expert.\n"
        f"t.shape={tuple(t.shape)!r}"
    )


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
