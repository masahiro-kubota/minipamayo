"""Tokenizer-visible action token registry for the Qwen3.5 Stage 1 path."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .action_quantizer import ActionQuantizer


@dataclass
class Stage1TokenRegistry:
    """Owns the extra Stage 1 action tokens added to the tokenizer."""

    n_bins: int = 256
    token_prefix: str = "act"
    token_strings: list[str] = field(init=False)
    token_ids: list[int] = field(default_factory=list)
    id_to_bin: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.token_strings = [f"<{self.token_prefix}_{i:03d}>" for i in range(self.n_bins)]

    def add_to_tokenizer(self, tokenizer) -> int:
        existing_vocab = tokenizer.get_vocab()
        missing = [tok for tok in self.token_strings if tok not in existing_vocab]
        added = tokenizer.add_tokens(missing)
        self.token_ids = tokenizer.convert_tokens_to_ids(self.token_strings)
        self.id_to_bin = {token_id: i for i, token_id in enumerate(self.token_ids)}
        return added

    def encode_action_token_ids(
        self,
        action: np.ndarray,
        quantizer: ActionQuantizer,
    ) -> list[int]:
        if not self.token_ids:
            raise RuntimeError("Token registry is not attached to a tokenizer yet.")
        return [self.token_ids[bin_idx] for bin_idx in quantizer.encode_bin_ids(action)]

    def encode_action_text(
        self,
        action: np.ndarray,
        quantizer: ActionQuantizer,
    ) -> str:
        token_ids = self.encode_action_token_ids(action, quantizer)
        tokens = [self.token_strings[self.id_to_bin[token_id]] for token_id in token_ids]
        return " ".join(tokens)

    def decode_action_token_ids(
        self,
        token_ids: list[int],
        quantizer: ActionQuantizer,
    ) -> np.ndarray:
        bin_ids = [self.id_to_bin[token_id] for token_id in token_ids]
        return quantizer.decode_bin_ids(bin_ids)
