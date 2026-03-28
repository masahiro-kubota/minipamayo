"""Prompt builders for canonical Stage 1."""

from __future__ import annotations

from .tokenization.history import (
    HISTORY_END_TOKEN,
    HISTORY_PLACEHOLDER_TOKEN,
    HISTORY_START_TOKEN,
)

DEFAULT_SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."
DEFAULT_QUESTION = "Output the future trajectory as action tokens in order. Do not provide explanations."


def build_history_placeholder(history_token_count: int) -> str:
    if history_token_count <= 0:
        return ""
    return (
        f"{HISTORY_START_TOKEN}"
        f"{HISTORY_PLACEHOLDER_TOKEN * history_token_count}"
        f"{HISTORY_END_TOKEN}"
    )


def build_prompt_text(processor, question: str, history_token_count: int = 0) -> str:
    history_prefix = build_history_placeholder(history_token_count)
    user_text = f"{history_prefix}\n{question}" if history_prefix else question
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": DEFAULT_SYSTEM_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
