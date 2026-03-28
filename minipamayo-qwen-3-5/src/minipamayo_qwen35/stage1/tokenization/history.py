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


def _quantize_scalar(value: float, value_range: tuple[float, float], n_bins: int) -> int:
    lower, upper = float(value_range[0]), float(value_range[1])
    if upper <= lower:
        raise RuntimeError(f"Invalid quantization range: {value_range!r}")
    clipped = min(max(float(value), lower), upper)
    normalized = (clipped - lower) / (upper - lower)
    return int(round(normalized * float(n_bins - 1)))


def _yaw_from_rot_mats(rot_mats: np.ndarray) -> np.ndarray:
    return np.arctan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0]).astype(np.float32)


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
        if history_xyz.shape != (self.history_steps, 3):
            raise RuntimeError(
                "Expected canonical `ego_history_xyz` shape "
                f"({self.history_steps}, 3), got {history_xyz.shape!r}."
            )
        if history_rot.shape != (self.history_steps, 3, 3):
            raise RuntimeError(
                "Expected canonical `ego_history_rot` shape "
                f"({self.history_steps}, 3, 3), got {history_rot.shape!r}."
            )
        yaw = _yaw_from_rot_mats(history_rot)
        bin_ids: list[int] = []
        for step_idx in range(self.history_steps):
            bin_ids.append(_quantize_scalar(float(history_xyz[step_idx, 0]), self.x_range, self.n_bins))
            bin_ids.append(_quantize_scalar(float(history_xyz[step_idx, 1]), self.y_range, self.n_bins))
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
            "token_layout": "per_step_xyz_yaw_scalar_bins",
        }


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
