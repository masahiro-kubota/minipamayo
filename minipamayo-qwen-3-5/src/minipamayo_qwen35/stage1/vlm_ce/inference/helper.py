"""Alpamayo-style helpers for Stage 1 inference."""

from __future__ import annotations

import collections.abc
from typing import Any

import torch
from transformers import AutoProcessor

from ....utils.image_budget import CANONICAL_IMAGE_MAX_PIXELS, CANONICAL_IMAGE_MIN_PIXELS

MIN_PIXELS = CANONICAL_IMAGE_MIN_PIXELS
MAX_PIXELS = CANONICAL_IMAGE_MAX_PIXELS
SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."


def create_message(frames: list[Any], user_text: str) -> list[dict[str, Any]]:
    """Construct the message using images and a short Stage 1 instruction."""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": frame} for frame in frames]
            + [
                {
                    "type": "text",
                    "text": user_text,
                }
            ],
        },
    ]


def get_processor(processor_path: str) -> AutoProcessor:
    """Get the processor with Alpamayo-style fixed image token budget."""
    return AutoProcessor.from_pretrained(
        processor_path,
        trust_remote_code=True,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )


def to_device(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Recursively cast data into the specified device, dtype."""
    if isinstance(data, torch.Tensor):
        data = data.to(
            device=device,
            dtype=dtype,
        )
        return data
    elif isinstance(data, collections.abc.Mapping):
        return {key: to_device(data[key], device=device, dtype=dtype) for key in data}
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return [to_device(elem, device=device, dtype=dtype) for elem in data]
    else:
        return data
