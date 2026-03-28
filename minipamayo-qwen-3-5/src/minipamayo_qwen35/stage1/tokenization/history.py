"""History trajectory tokenization helpers for canonical Stage 1."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

HISTORY_START_TOKEN = "<|traj_history_start|>"
HISTORY_PLACEHOLDER_TOKEN = "<|traj_history|>"
HISTORY_END_TOKEN = "<|traj_history_end|>"
HISTORY_SPECIAL_TOKENS = [
    HISTORY_START_TOKEN,
    HISTORY_PLACEHOLDER_TOKEN,
    HISTORY_END_TOKEN,
]


def canonicalize_history_sample_numpy(
    history_xyz: np.ndarray,
    history_rot: np.ndarray,
    *,
    history_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(history_xyz, dtype=np.float32)
    rot = np.asarray(history_rot, dtype=np.float32)
    if xyz.ndim == 3:
        if xyz.shape[0] != 1:
            raise RuntimeError(
                "Canonical history sample currently supports exactly one trajectory group.\n"
                f"ego_history_xyz.shape={xyz.shape!r}"
            )
        xyz = xyz[0]
    if rot.ndim == 4:
        if rot.shape[0] != 1:
            raise RuntimeError(
                "Canonical history sample currently supports exactly one trajectory group.\n"
                f"ego_history_rot.shape={rot.shape!r}"
            )
        rot = rot[0]
    if xyz.shape != (history_steps, 3):
        raise RuntimeError(
            "Expected canonical `ego_history_xyz` shape "
            f"(1, {history_steps}, 3) or ({history_steps}, 3), got {history_xyz.shape!r}."
        )
    if rot.shape != (history_steps, 3, 3):
        raise RuntimeError(
            "Expected canonical `ego_history_rot` shape "
            f"(1, {history_steps}, 3, 3) or ({history_steps}, 3, 3), got {history_rot.shape!r}."
        )
    return xyz, rot


def canonicalize_history_sample_tensors(
    history_xyz: torch.Tensor,
    history_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if history_xyz.dim() == 2:
        history_xyz = history_xyz.unsqueeze(0)
    if history_rot.dim() == 3:
        history_rot = history_rot.unsqueeze(0)
    if history_xyz.dim() != 3 or history_xyz.shape[0] != 1 or history_xyz.shape[-1] != 3:
        raise RuntimeError(
            "Expected canonical `ego_history_xyz` sample tensor shape "
            f"(1, T, 3), got {tuple(history_xyz.shape)!r}."
        )
    if history_rot.dim() != 4 or history_rot.shape[0] != 1 or history_rot.shape[-2:] != (3, 3):
        raise RuntimeError(
            "Expected canonical `ego_history_rot` sample tensor shape "
            f"(1, T, 3, 3), got {tuple(history_rot.shape)!r}."
        )
    return history_xyz, history_rot


def canonicalize_history_batch_tensors(
    history_xyz: torch.Tensor,
    history_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if history_xyz.dim() == 3:
        history_xyz = history_xyz.unsqueeze(1)
    if history_rot.dim() == 4:
        history_rot = history_rot.unsqueeze(1)
    if history_xyz.dim() != 4 or history_xyz.shape[1] != 1 or history_xyz.shape[-1] != 3:
        raise RuntimeError(
            "Expected canonical `ego_history_xyz` batch tensor shape "
            f"(batch, 1, T, 3), got {tuple(history_xyz.shape)!r}."
        )
    if history_rot.dim() != 5 or history_rot.shape[1] != 1 or history_rot.shape[-2:] != (3, 3):
        raise RuntimeError(
            "Expected canonical `ego_history_rot` batch tensor shape "
            f"(batch, 1, T, 3, 3), got {tuple(history_rot.shape)!r}."
        )
    return history_xyz, history_rot


def _quantize_scalar(value: float, value_range: tuple[float, float], n_bins: int) -> int:
    lower, upper = float(value_range[0]), float(value_range[1])
    if upper <= lower:
        raise RuntimeError(f"Invalid quantization range: {value_range!r}")
    clipped = min(max(float(value), lower), upper)
    normalized = (clipped - lower) / (upper - lower)
    return int(round(normalized * float(n_bins - 1)))


def _yaw_from_rot_mats(rot_mats: np.ndarray) -> np.ndarray:
    return np.arctan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0]).astype(np.float32)


def yaw_from_rot_mats_torch(rot_mats: torch.Tensor) -> torch.Tensor:
    return torch.atan2(rot_mats[..., 1, 0], rot_mats[..., 0, 0])


@dataclass(frozen=True)
class HistoryTrajectoryQuantizer:
    """Uniform scalar quantizer for ego-frame history `(x, y, yaw)`."""

    history_steps: int = 16
    n_bins: int = 256
    x_range: tuple[float, float] = (-20.0, 20.0)
    y_range: tuple[float, float] = (-20.0, 20.0)
    yaw_range: tuple[float, float] = (-math.pi, math.pi)

    @property
    def token_count(self) -> int:
        return self.history_steps * 3

    def encode_bin_ids(self, history_xyz: np.ndarray, history_rot: np.ndarray) -> list[int]:
        xyz, rot = canonicalize_history_sample_numpy(
            history_xyz,
            history_rot,
            history_steps=self.history_steps,
        )
        yaw = _yaw_from_rot_mats(rot)
        bin_ids: list[int] = []
        for step_idx in range(self.history_steps):
            bin_ids.append(
                _quantize_scalar(float(xyz[step_idx, 0]), self.x_range, self.n_bins)
            )
            bin_ids.append(
                _quantize_scalar(float(xyz[step_idx, 1]), self.y_range, self.n_bins)
            )
            bin_ids.append(_quantize_scalar(float(yaw[step_idx]), self.yaw_range, self.n_bins))
        return bin_ids

    def metadata(self) -> dict:
        return {
            "history_steps": self.history_steps,
            "n_bins": self.n_bins,
            "x_range": list(self.x_range),
            "y_range": list(self.y_range),
            "yaw_range": list(self.yaw_range),
            "token_count": self.token_count,
            "token_layout": "per_step_xyz_yaw_scalar_bins_single_traj_group",
        }

    def canonical_scalar_ranges(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        return (self.x_range, self.y_range, self.yaw_range)


@dataclass
class HistoryTokenRegistry:
    """Tokenizer-visible registry for Stage 1 history trajectory tokens."""

    n_bins: int = 256
    token_prefix: str = "hist"
    token_strings: list[str] = field(init=False)
    token_ids: list[int] = field(default_factory=list)
    id_to_bin: dict[int, int] = field(default_factory=dict)
    start_token_id: int | None = None
    placeholder_token_id: int | None = None
    end_token_id: int | None = None

    def __post_init__(self) -> None:
        self.token_strings = [f"<{self.token_prefix}_{i:03d}>" for i in range(self.n_bins)]

    def add_to_tokenizer(self, tokenizer) -> int:
        existing_vocab = tokenizer.get_vocab()
        missing = [tok for tok in self.token_strings if tok not in existing_vocab]
        added = tokenizer.add_tokens(missing)
        tokenizer.add_tokens(HISTORY_SPECIAL_TOKENS, special_tokens=True)
        self.token_ids = tokenizer.convert_tokens_to_ids(self.token_strings)
        self.id_to_bin = {token_id: i for i, token_id in enumerate(self.token_ids)}
        self.start_token_id = int(tokenizer.convert_tokens_to_ids(HISTORY_START_TOKEN))
        self.placeholder_token_id = int(tokenizer.convert_tokens_to_ids(HISTORY_PLACEHOLDER_TOKEN))
        self.end_token_id = int(tokenizer.convert_tokens_to_ids(HISTORY_END_TOKEN))
        return added

    def encode_bin_token_ids(self, bin_ids: list[int]) -> list[int]:
        if not self.token_ids:
            raise RuntimeError("History token registry is not attached to a tokenizer yet.")
        return [self.token_ids[int(bin_idx)] for bin_idx in bin_ids]

    def encode_history_token_ids(
        self,
        history_xyz: np.ndarray,
        history_rot: np.ndarray,
        quantizer: HistoryTrajectoryQuantizer,
    ) -> list[int]:
        return self.encode_bin_token_ids(quantizer.encode_bin_ids(history_xyz, history_rot))

    def replace_placeholder_ids(
        self,
        input_ids: torch.Tensor,
        history_token_id_rows: list[list[int]],
    ) -> torch.Tensor:
        if self.placeholder_token_id is None:
            raise RuntimeError("History token registry is not attached to a tokenizer yet.")
        fused = input_ids.clone()
        for row_idx, row_token_ids in enumerate(history_token_id_rows):
            placeholder_positions = torch.nonzero(
                fused[row_idx] == self.placeholder_token_id,
                as_tuple=False,
            ).flatten()
            if placeholder_positions.numel() != len(row_token_ids):
                raise RuntimeError(
                    "History placeholder count does not match encoded history token count.\n"
                    f"expected={len(row_token_ids)}\n"
                    f"found={int(placeholder_positions.numel())}"
                )
            fused[row_idx, placeholder_positions] = torch.tensor(
                row_token_ids,
                dtype=fused.dtype,
                device=fused.device,
            )
        return fused

    def metadata(self) -> dict:
        return {
            "n_bins": self.n_bins,
            "token_prefix": self.token_prefix,
            "token_strings": self.token_strings,
            "special_tokens": list(HISTORY_SPECIAL_TOKENS),
        }


def encode_history_token_id_rows(
    *,
    history_xyz: torch.Tensor,
    history_rot: torch.Tensor,
    history_registry: HistoryTokenRegistry,
    history_quantizer: HistoryTrajectoryQuantizer,
) -> list[list[int]]:
    batch_xyz, batch_rot = canonicalize_history_batch_tensors(history_xyz, history_rot)
    history_token_id_rows: list[list[int]] = []
    for row_idx in range(batch_xyz.shape[0]):
        xyz_np = batch_xyz[row_idx].detach().cpu().numpy()
        rot_np = batch_rot[row_idx].detach().cpu().numpy()
        history_token_id_rows.append(
            history_registry.encode_history_token_ids(
                xyz_np,
                rot_np,
                history_quantizer,
            )
        )
    return history_token_id_rows


def history_xyz_rot_to_scalars_torch(
    history_xyz: torch.Tensor,
    history_rot: torch.Tensor,
) -> torch.Tensor:
    batch_xyz, batch_rot = canonicalize_history_batch_tensors(history_xyz, history_rot)
    squeezed_xyz = batch_xyz[:, 0]
    squeezed_rot = batch_rot[:, 0]
    yaw = yaw_from_rot_mats_torch(squeezed_rot).unsqueeze(-1)
    return torch.stack(
        [squeezed_xyz[..., 0], squeezed_xyz[..., 1], yaw[..., 0]],
        dim=-1,
    )


def normalize_history_scalars_torch(
    history_scalars: torch.Tensor,
    quantizer: HistoryTrajectoryQuantizer,
) -> torch.Tensor:
    if history_scalars.dim() != 3 or history_scalars.shape[-1] != 3:
        raise RuntimeError(
            "Expected canonical history scalars to have shape (batch, history_steps, 3), "
            f"got {tuple(history_scalars.shape)!r}."
        )
    ranges = torch.tensor(
        [
            [float(quantizer.x_range[0]), float(quantizer.x_range[1])],
            [float(quantizer.y_range[0]), float(quantizer.y_range[1])],
            [float(quantizer.yaw_range[0]), float(quantizer.yaw_range[1])],
        ],
        dtype=history_scalars.dtype,
        device=history_scalars.device,
    )
    lower = ranges[:, 0].view(1, 1, 3)
    upper = ranges[:, 1].view(1, 1, 3)
    clipped = torch.minimum(torch.maximum(history_scalars, lower), upper)
    normalized = (clipped - lower) / (upper - lower)
    return normalized.mul(2.0).sub(1.0)


def interpolate_history_token_embeddings(
    *,
    embedding_weight: torch.Tensor,
    history_xyz: torch.Tensor,
    history_rot: torch.Tensor,
    history_registry: HistoryTokenRegistry,
    history_quantizer: HistoryTrajectoryQuantizer,
) -> torch.Tensor:
    if not history_registry.token_ids:
        raise RuntimeError(
            "History token registry must be attached to a tokenizer before interpolation."
        )
    batch_xyz, batch_rot = canonicalize_history_batch_tensors(history_xyz, history_rot)
    if batch_xyz.shape[2] != history_quantizer.history_steps:
        raise RuntimeError(
            "History tensor step count does not match the canonical quantizer.\n"
            f"history_xyz.shape={tuple(batch_xyz.shape)!r}\n"
            f"history_steps={history_quantizer.history_steps}"
        )
    history_scalars = history_xyz_rot_to_scalars_torch(batch_xyz, batch_rot)
    normalized = normalize_history_scalars_torch(history_scalars, history_quantizer)
    scaled = normalized.add(1.0).mul(0.5 * float(history_quantizer.n_bins - 1))
    lower_idx = torch.floor(scaled).to(torch.long)
    upper_idx = torch.ceil(scaled).to(torch.long)
    frac = (scaled - lower_idx.to(scaled.dtype)).unsqueeze(-1)

    history_token_ids = torch.tensor(
        history_registry.token_ids,
        dtype=torch.long,
        device=history_xyz.device,
    )
    flat_lower = lower_idx.reshape(-1)
    flat_upper = upper_idx.reshape(-1)
    lower_token_ids = history_token_ids.index_select(dim=0, index=flat_lower)
    upper_token_ids = history_token_ids.index_select(dim=0, index=flat_upper)
    lower_embeds = embedding_weight.index_select(dim=0, index=lower_token_ids)
    upper_embeds = embedding_weight.index_select(dim=0, index=upper_token_ids)
    flat_interp = lower_embeds * (1.0 - frac.reshape(-1, 1)) + upper_embeds * frac.reshape(-1, 1)
    return flat_interp.reshape(batch_xyz.shape[0], history_quantizer.token_count, -1)
