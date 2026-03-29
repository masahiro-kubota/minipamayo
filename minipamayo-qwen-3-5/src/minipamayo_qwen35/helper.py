"""Alpamayo-style helper utilities shared by inference entrypoints."""

from __future__ import annotations

import collections.abc
from typing import Any

import torch
from transformers import AutoProcessor

from .contract.history_tokens import HistoryTrajectoryQuantizer
from .contract.prompt import (
    ALPAMAYO_REASONING_USER_TEXT,
    COT_START_TOKEN,
    DEFAULT_SYSTEM_PROMPT,
    build_history_placeholder,
)
from .utils.image_budget import CANONICAL_IMAGE_MAX_PIXELS, CANONICAL_IMAGE_MIN_PIXELS

MIN_PIXELS = CANONICAL_IMAGE_MIN_PIXELS
MAX_PIXELS = CANONICAL_IMAGE_MAX_PIXELS


def create_message(
    frames: torch.Tensor,
    *,
    num_traj_token: int | None = None,
    user_text: str | None = None,
):
    """Construct the Alpamayo-style message using images and CoT prefill."""
    if frames.ndim != 4:
        raise ValueError(f"{frames.ndim=}, expected 4 (N, C, H, W)")

    if num_traj_token is None:
        num_traj_token = HistoryTrajectoryQuantizer().token_count
    if user_text is None:
        user_text = f"{build_history_placeholder(num_traj_token)}{ALPAMAYO_REASONING_USER_TEXT}"

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": DEFAULT_SYSTEM_PROMPT,
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
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": COT_START_TOKEN,
                }
            ],
        },
    ]


def get_processor(processor_path: str, tokenizer: Any | None = None) -> AutoProcessor:
    """Load the canonical processor with Alpamayo-style pixel settings."""
    processor = AutoProcessor.from_pretrained(
        processor_path,
        trust_remote_code=True,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    if tokenizer is not None:
        processor.tokenizer = tokenizer
    return processor


def to_device(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Recursively cast data into the specified device, dtype."""
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    if isinstance(data, collections.abc.Mapping):
        return {key: to_device(data[key], device=device, dtype=dtype) for key in data}
    if isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return [to_device(elem, device=device, dtype=dtype) for elem in data]
    return data
