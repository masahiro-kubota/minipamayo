"""Alpamayo-style helpers for Stage 1 inference."""

from __future__ import annotations

import collections.abc
from typing import Any

import torch
from transformers import AutoProcessor

from ....utils.image_budget import CANONICAL_IMAGE_MAX_PIXELS, CANONICAL_IMAGE_MIN_PIXELS
from ...prompt import ALPAMAYO_REASONING_USER_TEXT, COT_START_TOKEN, build_history_placeholder

MIN_PIXELS = CANONICAL_IMAGE_MIN_PIXELS
MAX_PIXELS = CANONICAL_IMAGE_MAX_PIXELS
BASE_PROCESSOR_NAME = "Qwen/Qwen3-VL-2B-Instruct"
SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."


def _normalize_frames(frames: torch.Tensor | list[Any]) -> list[Any]:
    if isinstance(frames, torch.Tensor):
        if frames.ndim != 4:
            raise ValueError(f"{frames.ndim=}, expected 4 (N, C, H, W)")
        return [frame for frame in frames]
    return list(frames)


def create_message(
    frames: torch.Tensor | list[Any],
    user_text: str | None = None,
    *,
    history_token_count: int = 48,
    include_assistant_prefill: bool = True,
) -> list[dict[str, Any]]:
    """Construct the message following Alpamayo helper conventions."""
    normalized_frames = _normalize_frames(frames)
    text = user_text
    if text is None:
        text = (
            f"{build_history_placeholder(history_token_count)}"
            f"{ALPAMAYO_REASONING_USER_TEXT}"
        )
    messages = [
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
            "content": [{"type": "image", "image": frame} for frame in normalized_frames]
            + [
                {
                    "type": "text",
                    "text": text,
                }
            ],
        },
    ]
    if include_assistant_prefill:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": COT_START_TOKEN,
                    }
                ],
            }
        )
    return messages


def get_processor(tokenizer, processor_name: str = BASE_PROCESSOR_NAME) -> AutoProcessor:
    """Get the processor with Alpamayo-style fixed image token budget."""
    processor = AutoProcessor.from_pretrained(
        processor_name,
        trust_remote_code=True,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    processor.tokenizer = tokenizer
    return processor


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
