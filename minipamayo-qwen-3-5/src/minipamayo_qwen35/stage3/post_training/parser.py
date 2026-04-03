"""Canonical parsing of Stage 3 generated sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...reasoning.synthetic import normalize_label


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


def _neutral_action_token_ids(registry, quantizer, expected_action_token_count: int) -> list[int]:
    zeros = np.zeros(expected_action_token_count, dtype=np.float32)
    return registry.encode_action_token_ids(zeros, quantizer)


def decode_action_token_ids(
    *,
    token_ids: list[int],
    registry,
    quantizer,
    expected_action_token_count: int,
) -> tuple[np.ndarray, list[int], int]:
    token_rows = list(token_ids[:expected_action_token_count])
    generated_count = len(token_rows)
    if generated_count < expected_action_token_count:
        neutral = _neutral_action_token_ids(registry, quantizer, expected_action_token_count)
        token_rows.extend(neutral[generated_count:expected_action_token_count])
    action = registry.decode_action_token_ids(token_rows, quantizer)
    return action, token_rows, generated_count


@dataclass(frozen=True)
class ParsedStage3Sequence:
    generated_token_ids: list[int]
    reasoning_token_ids: list[int]
    action_token_ids: list[int]
    padded_action_token_ids: list[int]
    reasoning_text: str
    action: np.ndarray
    decision: dict[str, str] | None
    generated_action_count: int
    valid: bool
    issues: tuple[str, ...]


def parse_generated_sequence(
    *,
    token_ids: list[int],
    tokenizer,
    registry,
    quantizer,
    expected_action_token_count: int,
) -> ParsedStage3Sequence:
    action_token_id_set = set(registry.token_ids)
    reasoning_token_ids: list[int] = []
    action_token_ids: list[int] = []
    issues: list[str] = []
    in_action_segment = False

    for token_id in token_ids:
        if tokenizer.pad_token_id is not None and token_id == int(tokenizer.pad_token_id):
            break
        if tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id):
            break
        if token_id in action_token_id_set:
            in_action_segment = True
            action_token_ids.append(int(token_id))
            continue
        if in_action_segment:
            issues.append("non_action_token_after_action_segment")
            continue
        reasoning_token_ids.append(int(token_id))

    if not action_token_ids:
        issues.append("missing_action_segment")
    if len(action_token_ids) < expected_action_token_count:
        issues.append("truncated_action_segment")

    action, padded_action_token_ids, generated_action_count = decode_action_token_ids(
        token_ids=action_token_ids,
        registry=registry,
        quantizer=quantizer,
        expected_action_token_count=expected_action_token_count,
    )
    reasoning_text = tokenizer.decode(reasoning_token_ids, skip_special_tokens=True).strip()
    if not reasoning_text:
        issues.append("empty_reasoning_text")
    decision = parse_decision_from_text(reasoning_text)
    if decision is None:
        issues.append("missing_structured_decision")

    return ParsedStage3Sequence(
        generated_token_ids=[int(token_id) for token_id in token_ids],
        reasoning_token_ids=reasoning_token_ids,
        action_token_ids=action_token_ids[:expected_action_token_count],
        padded_action_token_ids=padded_action_token_ids,
        reasoning_text=reasoning_text,
        action=action,
        decision=decision,
        generated_action_count=generated_action_count,
        valid=not issues,
        issues=tuple(issues),
    )


__all__ = [
    "ParsedStage3Sequence",
    "decode_action_token_ids",
    "parse_decision_from_text",
    "parse_generated_sequence",
]
