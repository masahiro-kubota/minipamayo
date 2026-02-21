"""Discrete action tokenizer for Stage 1.

Quantizes continuous (a, kappa) control inputs to discrete token IDs.
Shared bins: a and kappa use the same 256 token IDs, position determines meaning.

Token format (interleaved): [a_0, kappa_0, a_1, kappa_1, ..., a_{K-1}, kappa_{K-1}]
"""

import numpy as np
import torch


class DiscreteActionTokenizer:
    """Quantize/dequantize (a, kappa) to discrete token IDs.

    Uses shared bins (256 tokens) for both a and kappa.
    Position in the sequence determines whether a bin represents a or kappa.
    """

    def __init__(
        self,
        n_bins: int = 256,
        a_range: tuple[float, float] = (-6.0, 6.0),
        kappa_range: tuple[float, float] = (-0.1, 0.1),
        vocab_offset: int = 151936,
    ):
        self.n_bins = n_bins
        self.a_min, self.a_max = a_range
        self.kappa_min, self.kappa_max = kappa_range
        self.vocab_offset = vocab_offset

    def _quantize_value(self, value: float, v_min: float, v_max: float) -> int:
        """Quantize a single value to bin index [0, n_bins-1]."""
        clamped = max(v_min, min(v_max - 1e-8, value))
        bin_idx = int((clamped - v_min) / (v_max - v_min) * self.n_bins)
        return min(bin_idx, self.n_bins - 1)

    def _dequantize_value(self, bin_idx: int, v_min: float, v_max: float) -> float:
        """Dequantize bin index to value (bin center)."""
        return v_min + (bin_idx + 0.5) * (v_max - v_min) / self.n_bins

    def encode(self, action: np.ndarray) -> list[int]:
        """Encode interleaved (a, kappa) x K to token IDs.

        Args:
            action: (K*2,) interleaved [a_0, k_0, a_1, k_1, ...]

        Returns:
            token_ids: list of K*2 token IDs
        """
        token_ids = []
        for i, val in enumerate(action):
            if i % 2 == 0:  # acceleration
                bin_idx = self._quantize_value(val, self.a_min, self.a_max)
            else:  # curvature
                bin_idx = self._quantize_value(val, self.kappa_min, self.kappa_max)
            token_ids.append(self.vocab_offset + bin_idx)
        return token_ids

    def encode_batch(self, actions: torch.Tensor) -> torch.Tensor:
        """Batch encode: (B, K*2) float actions -> (B, K*2) token IDs."""
        B, K2 = actions.shape
        token_ids = torch.zeros(B, K2, dtype=torch.long, device=actions.device)

        for i in range(K2):
            if i % 2 == 0:  # acceleration
                v_min, v_max = self.a_min, self.a_max
            else:  # curvature
                v_min, v_max = self.kappa_min, self.kappa_max

            clamped = actions[:, i].clamp(v_min, v_max - 1e-8)
            bin_idx = ((clamped - v_min) / (v_max - v_min) * self.n_bins).long()
            bin_idx = bin_idx.clamp(0, self.n_bins - 1)
            token_ids[:, i] = self.vocab_offset + bin_idx

        return token_ids

    def decode(self, token_ids: list[int]) -> np.ndarray:
        """Decode token IDs to continuous (a, kappa) values."""
        values = []
        for i, tid in enumerate(token_ids):
            bin_idx = tid - self.vocab_offset
            if i % 2 == 0:  # acceleration
                values.append(self._dequantize_value(bin_idx, self.a_min, self.a_max))
            else:  # curvature
                values.append(self._dequantize_value(bin_idx, self.kappa_min, self.kappa_max))
        return np.array(values, dtype=np.float32)

    def decode_batch(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Batch decode: (B, K*2) token IDs -> (B, K*2) float actions."""
        B, K2 = token_ids.shape
        values = torch.zeros(B, K2, dtype=torch.float32, device=token_ids.device)

        for i in range(K2):
            bin_idx = (token_ids[:, i] - self.vocab_offset).float()
            if i % 2 == 0:  # acceleration
                v_min, v_max = self.a_min, self.a_max
            else:  # curvature
                v_min, v_max = self.kappa_min, self.kappa_max
            values[:, i] = v_min + (bin_idx + 0.5) * (v_max - v_min) / self.n_bins

        return values
