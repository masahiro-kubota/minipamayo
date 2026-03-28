"""Tokenizer-visible Stage 1 target token registry for the Qwen3.5 path."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def format_stage1_token(prefix: str, index: int) -> str:
    if prefix == "i":
        return f"<i{index}>"
    return f"<{prefix}_{index:03d}>"

@dataclass
class Stage1TokenRegistry:
    """Owns the extra Stage 1 target tokens added to the tokenizer."""

    n_bins: int = 256
    token_prefix: str = "i"
    start_index: int = 0
    token_strings: list[str] = field(init=False)
    token_ids: list[int] = field(default_factory=list)
    id_to_bin: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.token_strings = [
            format_stage1_token(self.token_prefix, self.start_index + i)
            for i in range(self.n_bins)
        ]

    def add_to_tokenizer(self, tokenizer) -> int:
        existing_vocab = tokenizer.get_vocab()
        missing = [tok for tok in self.token_strings if tok not in existing_vocab]
        added = tokenizer.add_tokens(missing)
        self.token_ids = tokenizer.convert_tokens_to_ids(self.token_strings)
        self.id_to_bin = {token_id: i for i, token_id in enumerate(self.token_ids)}
        return added

    def encode_bin_token_ids(self, bin_ids: list[int]) -> list[int]:
        if not self.token_ids:
            raise RuntimeError("Token registry is not attached to a tokenizer yet.")
        return [self.token_ids[int(bin_idx)] for bin_idx in bin_ids]

    def decode_token_ids_to_bin_ids(self, token_ids: list[int]) -> list[int]:
        return [self.id_to_bin[int(token_id)] for token_id in token_ids]

    def encode_target_token_ids(self, target: np.ndarray, quantizer) -> list[int]:
        return self.encode_bin_token_ids(quantizer.encode_bin_ids(target))

    def decode_target_token_ids(self, token_ids: list[int], quantizer) -> np.ndarray:
        return quantizer.decode_bin_ids(self.decode_token_ids_to_bin_ids(token_ids))

    def encode_action_token_ids(self, action: np.ndarray, quantizer) -> list[int]:
        return self.encode_target_token_ids(action, quantizer)

    def encode_action_text(
        self,
        action: np.ndarray,
        quantizer,
    ) -> str:
        token_ids = self.encode_action_token_ids(action, quantizer)
        tokens = [self.token_strings[self.id_to_bin[token_id]] for token_id in token_ids]
        return " ".join(tokens)

    def decode_action_token_ids(self, token_ids: list[int], quantizer) -> np.ndarray:
        return self.decode_target_token_ids(token_ids, quantizer)

    @property
    def token_id_min(self) -> int:
        if not self.token_ids:
            raise RuntimeError("Token registry is not attached to a tokenizer yet.")
        return min(self.token_ids)

    @property
    def token_id_max(self) -> int:
        if not self.token_ids:
            raise RuntimeError("Token registry is not attached to a tokenizer yet.")
        return max(self.token_ids)

    def mask_logits(self, scores: torch.Tensor) -> torch.Tensor:
        if not self.token_ids:
            raise RuntimeError("Token registry is not attached to a tokenizer yet.")
        masked = scores.clone()
        masked[:, self.token_ids] = torch.finfo(masked.dtype).min
        return masked
