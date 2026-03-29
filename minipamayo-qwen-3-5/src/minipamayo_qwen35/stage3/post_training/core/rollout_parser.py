"""Parsing helpers for Stage 3/4 generated rollouts."""

from __future__ import annotations

import numpy as np

from ....action_space.discrete_action_space import DiscreteTrajectoryTokenizer
from ....contract.trajectory_tokens import Stage1TokenRegistry
from ....reasoning.synthetic import normalize_label


def parse_decision_from_text(text: str) -> dict[str, str] | None:
    decision: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("longitudinal:"):
            decision["longitudinal"] = normalize_label(line.split(":", 1)[1])
        elif lower.startswith("lateral:"):
            decision["lateral"] = normalize_label(line.split(":", 1)[1])
    if "longitudinal" in decision and "lateral" in decision:
        return decision
    return None


def split_generated_token_ids(
    token_ids: list[int],
    registry: Stage1TokenRegistry,
) -> tuple[list[int], list[int]]:
    action_token_id_set = set(registry.token_ids)
    text_token_ids: list[int] = []
    action_token_ids: list[int] = []
    for token_id in token_ids:
        if token_id in action_token_id_set:
            action_token_ids.append(token_id)
        else:
            text_token_ids.append(token_id)
    return text_token_ids, action_token_ids


def _neutral_action_token_ids(
    registry: Stage1TokenRegistry,
    quantizer: DiscreteTrajectoryTokenizer,
    action_len: int,
) -> list[int]:
    zeros = np.zeros(action_len, dtype=np.float32)
    return registry.encode_action_token_ids(zeros, quantizer)


def decode_action_token_ids(
    token_ids: list[int],
    registry: Stage1TokenRegistry,
    quantizer: DiscreteTrajectoryTokenizer,
    action_len: int,
) -> tuple[np.ndarray, int]:
    token_rows = list(token_ids[:action_len])
    generated_count = len(token_rows)
    if generated_count < action_len:
        neutral = _neutral_action_token_ids(registry, quantizer, action_len)
        token_rows.extend(neutral[generated_count:action_len])
    return registry.decode_action_token_ids(token_rows, quantizer), generated_count


def parse_generated_sequence(
    token_ids: list[int],
    tokenizer,
    registry: Stage1TokenRegistry,
    quantizer: DiscreteTrajectoryTokenizer,
    action_len: int,
) -> dict:
    text_token_ids, action_token_ids = split_generated_token_ids(token_ids, registry)
    reasoning_text = tokenizer.decode(text_token_ids, skip_special_tokens=True).strip()
    action, generated_action_count = decode_action_token_ids(
        token_ids=action_token_ids,
        registry=registry,
        quantizer=quantizer,
        action_len=action_len,
    )
    decision = parse_decision_from_text(reasoning_text)
    return {
        "reasoning_text": reasoning_text,
        "action": action,
        "decision": decision,
        "action_token_ids": action_token_ids[:action_len],
        "generated_action_count": generated_action_count,
    }
