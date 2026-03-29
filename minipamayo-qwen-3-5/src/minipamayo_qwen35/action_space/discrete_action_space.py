"""Discrete action-space tokenization aligned with Alpamayo naming."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace


def _canonicalize_future_xyz(value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 3:
        return value
    if value.dim() == 4 and value.shape[1] == 1:
        return value[:, 0]
    raise RuntimeError(
        "Expected `fut_xyz` shape (batch, T, 3) or (batch, 1, T, 3), "
        f"got {tuple(value.shape)!r}."
    )


def _canonicalize_future_rot(value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 4:
        return value
    if value.dim() == 5 and value.shape[1] == 1:
        return value[:, 0]
    raise RuntimeError(
        "Expected `fut_rot` shape (batch, T, 3, 3) or (batch, 1, T, 3, 3), "
        f"got {tuple(value.shape)!r}."
    )


@dataclass(frozen=True)
class DiscreteTrajectoryTokenizer:
    """Alpamayo-like discrete tokenizer over interleaved (a, kappa) controls."""

    n_bins: int = 256
    a_range: tuple[float, float] = (-6.0, 6.0)
    kappa_range: tuple[float, float] = (-0.2, 0.2)

    @property
    def vocab_size(self) -> int:
        return self.n_bins

    @property
    def dims_min(self) -> list[float]:
        return [float(self.a_range[0]), float(self.kappa_range[0])]

    @property
    def dims_max(self) -> list[float]:
        return [float(self.a_range[1]), float(self.kappa_range[1])]

    def _build_action_space(self, *, k: int) -> UnicycleAccelCurvatureActionSpace:
        return UnicycleAccelCurvatureActionSpace(k=int(k))

    def quantize_bin(self, value: float, v_min: float, v_max: float) -> int:
        clamped = max(v_min, min(v_max - 1e-8, float(value)))
        bin_idx = int((clamped - v_min) / (v_max - v_min) * self.n_bins)
        return min(max(bin_idx, 0), self.n_bins - 1)

    def dequantize_bin(self, bin_idx: int, v_min: float, v_max: float) -> float:
        return v_min + (bin_idx + 0.5) * (v_max - v_min) / self.n_bins

    def encode_bin_ids(self, action: np.ndarray) -> list[int]:
        bin_ids: list[int] = []
        for i, value in enumerate(action):
            if i % 2 == 0:
                v_min, v_max = self.a_range
            else:
                v_min, v_max = self.kappa_range
            bin_ids.append(self.quantize_bin(float(value), v_min, v_max))
        return bin_ids

    def decode_bin_ids(self, bin_ids: list[int]) -> np.ndarray:
        values: list[float] = []
        for i, bin_idx in enumerate(bin_ids):
            if i % 2 == 0:
                v_min, v_max = self.a_range
            else:
                v_min, v_max = self.kappa_range
            values.append(self.dequantize_bin(int(bin_idx), v_min, v_max))
        return np.asarray(values, dtype=np.float32)

    def encode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        fut_xyz: torch.Tensor,
        fut_rot: torch.Tensor,
        hist_tstamp: torch.Tensor | None = None,
        fut_tstamp: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        del hist_tstamp, fut_tstamp
        future_xyz = _canonicalize_future_xyz(fut_xyz)
        future_rot = _canonicalize_future_rot(fut_rot)
        action_space = self._build_action_space(k=int(future_xyz.shape[1]))
        action = action_space.traj_to_action(
            traj_history_xyz=hist_xyz,
            traj_history_rot=hist_rot,
            traj_future_xyz=future_xyz,
            traj_future_rot=future_rot,
        )
        dims_min = torch.tensor(self.dims_min, device=action.device, dtype=action.dtype)
        dims_max = torch.tensor(self.dims_max, device=action.device, dtype=action.dtype)
        action = (action - dims_min) / (dims_max - dims_min)
        action = (action * float(self.n_bins - 1)).round().long()
        action = action.clamp(0, self.n_bins - 1)
        return action.reshape(action.shape[0], -1)

    def decode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        tokens: torch.LongTensor,
        hist_tstamp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        del hist_tstamp
        if tokens.dim() != 2 or tokens.shape[1] % 2 != 0:
            raise RuntimeError(
                "ActionQuantizer.decode expects tokens shaped (batch, 2*k).\n"
                f"found={tuple(tokens.shape)!r}"
            )
        k = tokens.shape[1] // 2
        action = tokens.reshape(-1, k, 2).to(hist_xyz.dtype)
        dims_min = torch.tensor(self.dims_min, device=action.device, dtype=action.dtype)
        dims_max = torch.tensor(self.dims_max, device=action.device, dtype=action.dtype)
        action = action / float(self.n_bins - 1)
        action = action * (dims_max - dims_min) + dims_min
        action_space = self._build_action_space(k=k)
        fut_xyz, fut_rot = action_space.action_to_traj(action, hist_xyz, hist_rot)
        return fut_xyz, fut_rot, None


ActionQuantizer = DiscreteTrajectoryTokenizer
