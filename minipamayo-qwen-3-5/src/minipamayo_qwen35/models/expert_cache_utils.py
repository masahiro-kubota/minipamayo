"""Shared prompt-cache helpers used by Alpamayo expert paths."""

from __future__ import annotations

import copy


def validate_prompt_cache_layers(prompt_cache, num_layers: int) -> None:
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
    raise RuntimeError(f"Unsupported prompt cache type for expert path: {type(prompt_cache)!r}")


def clone_prompt_cache_for_expert(prompt_cache, num_layers: int):
    validate_prompt_cache_layers(prompt_cache, num_layers)
    if hasattr(prompt_cache, "key_cache") and hasattr(prompt_cache, "value_cache"):
        cloned = copy.deepcopy(prompt_cache)
        cloned.key_cache = list(cloned.key_cache[:num_layers])
        cloned.value_cache = list(cloned.value_cache[:num_layers])
        return cloned
    if isinstance(prompt_cache, tuple):
        return tuple(prompt_cache[:num_layers])
    if isinstance(prompt_cache, list):
        return list(prompt_cache[:num_layers])
    raise RuntimeError(f"Unsupported prompt cache type for expert path: {type(prompt_cache)!r}")
