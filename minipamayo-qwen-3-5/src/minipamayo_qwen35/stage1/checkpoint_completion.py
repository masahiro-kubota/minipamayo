"""Helpers for refusing evaluation on incomplete training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def require_completed_training_run(
    checkpoint_path: str | Path,
    *,
    checkpoint_label: str,
    required_summary_keys: Iterable[str],
    allowed_stop_reasons: Iterable[str] | None = None,
) -> dict:
    resolved_checkpoint = Path(checkpoint_path).resolve()
    summary_path = resolved_checkpoint.parent / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"Refusing to evaluate {checkpoint_label} before the training run completed.\n"
            f"checkpoint={resolved_checkpoint}\n"
            f"expected_completion_marker={summary_path}\n"
            "Wait for the train runner to finish and then evaluate the completed run's `best.pt`."
        )
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Training completion marker for {checkpoint_label} is not valid JSON: {summary_path}"
        ) from exc
    if not isinstance(summary, dict):
        raise RuntimeError(
            f"Training completion marker for {checkpoint_label} must be a JSON object: {summary_path}"
        )
    missing_keys = [key for key in required_summary_keys if key not in summary]
    if missing_keys:
        raise RuntimeError(
            f"Training completion marker for {checkpoint_label} is missing required fields.\n"
            f"summary_json={summary_path}\n"
            + "\n".join(missing_keys)
        )
    if allowed_stop_reasons is not None:
        stop_reason = summary.get("stop_reason")
        allowed = {str(reason) for reason in allowed_stop_reasons}
        if stop_reason not in allowed:
            raise RuntimeError(
                f"Training completion marker for {checkpoint_label} has unexpected stop_reason.\n"
                f"summary_json={summary_path}\n"
                f"found={stop_reason!r}\n"
                f"expected_one_of={sorted(allowed)!r}"
            )
    return summary
