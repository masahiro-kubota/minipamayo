"""Discrete action quantization for Stage 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionQuantizer:
    """Shared-bin quantizer for interleaved (a, kappa) controls."""

    n_bins: int = 256
    a_range: tuple[float, float] = (-6.0, 6.0)
    kappa_range: tuple[float, float] = (-0.2, 0.2)

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
