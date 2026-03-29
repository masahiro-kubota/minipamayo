"""Shared prompt builders for the Stage 1 / Stage 2 contract."""

from __future__ import annotations

from .history_tokens import (
    HISTORY_END_TOKEN,
    HISTORY_PLACEHOLDER_TOKEN,
    HISTORY_START_TOKEN,
)

DEFAULT_SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."
DEFAULT_QUESTION = "Output the future trajectory as action tokens in order. Do not provide explanations."
ALPAMAYO_REASONING_USER_TEXT = (
    "output the chain-of-thought reasoning of the driving process, "
    "then output the future trajectory."
)
COT_START_TOKEN = "<|cot_start|>"
COT_END_TOKEN = "<|cot_end|>"
TRAJ_FUTURE_START_TOKEN = "<|traj_future_start|>"
TRAJ_FUTURE_END_TOKEN = "<|traj_future_end|>"
PROMPT_SPECIAL_TOKENS = [
    COT_START_TOKEN,
    COT_END_TOKEN,
    TRAJ_FUTURE_START_TOKEN,
    TRAJ_FUTURE_END_TOKEN,
]


def add_prompt_special_tokens(tokenizer) -> int:
    return int(tokenizer.add_tokens(PROMPT_SPECIAL_TOKENS, special_tokens=True))


def build_history_placeholder(history_token_count: int) -> str:
    if history_token_count <= 0:
        return ""
    return (
        f"{HISTORY_START_TOKEN}"
        f"{HISTORY_PLACEHOLDER_TOKEN * history_token_count}"
        f"{HISTORY_END_TOKEN}"
    )


def build_messages(
    *,
    user_text: str,
    assistant_prefill: str | None = None,
    image_count: int = 1,
) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": DEFAULT_SYSTEM_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(image_count)]
            + [{"type": "text", "text": user_text}],
        },
    ]
    if assistant_prefill is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_prefill},
                ],
            }
        )
    return messages


def build_chat_prompt_text(
    processor,
    *,
    user_text: str,
    assistant_prefill: str | None = None,
    image_count: int = 1,
) -> str:
    messages = build_messages(
        user_text=user_text,
        assistant_prefill=assistant_prefill,
        image_count=image_count,
    )
    continue_final_message = assistant_prefill is not None
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not continue_final_message,
        continue_final_message=continue_final_message,
    )


def build_prompt_text(
    processor,
    question: str,
    history_token_count: int = 0,
    *,
    assistant_prefill: str | None = None,
) -> str:
    history_prefix = build_history_placeholder(history_token_count)
    user_text = f"{history_prefix}{question}" if history_prefix else question
    return build_chat_prompt_text(
        processor,
        user_text=user_text,
        assistant_prefill=assistant_prefill,
    )


def build_reasoning_prompt_text(processor, history_token_count: int = 0) -> str:
    history_prefix = build_history_placeholder(history_token_count)
    user_text = f"{history_prefix}{ALPAMAYO_REASONING_USER_TEXT}"
    return build_chat_prompt_text(
        processor,
        user_text=user_text,
        assistant_prefill=COT_START_TOKEN,
    )
