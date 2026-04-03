"""Helpers for deriving contiguous block groupings from normalized samples."""

from __future__ import annotations

from dataclasses import replace
from statistics import mean
from typing import Any

from .models import NormalizedGroup, NormalizedSample


def _metric_values(samples: list[NormalizedSample], attr_name: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = getattr(sample, attr_name)
        if value is not None:
            values.append(float(value))
    return values


def summarize_group_samples(samples: list[NormalizedSample]) -> dict[str, Any]:
    commands = sorted({sample.command for sample in samples if sample.command})
    ade_values = _metric_values(samples, "ade_m")
    fde_values = _metric_values(samples, "fde_m")
    kappa_values = _metric_values(samples, "action_mae_kappa")
    token_values = _metric_values(samples, "token_accuracy")
    summary: dict[str, Any] = {
        "length": len(samples),
        "commands": commands,
    }
    if ade_values:
        summary["mean_ade_m"] = mean(ade_values)
        summary["max_ade_m"] = max(ade_values)
    if fde_values:
        summary["mean_fde_m"] = mean(fde_values)
        summary["max_fde_m"] = max(fde_values)
        worst_sample = max(
            (sample for sample in samples if sample.fde_m is not None),
            key=lambda sample: float(sample.fde_m or 0.0),
        )
        summary["worst_sample_id"] = worst_sample.sample_id
    if kappa_values:
        summary["mean_action_mae_kappa"] = mean(kappa_values)
    if token_values:
        summary["mean_token_accuracy"] = mean(token_values)
    return summary


def derive_groups(samples: list[NormalizedSample]) -> tuple[list[NormalizedSample], list[NormalizedGroup]]:
    if not samples:
        return [], []

    ordered_samples = sorted(samples, key=lambda sample: (int(sample.sample_index), str(sample.sample_id)))
    raw_groups: list[list[NormalizedSample]] = []
    current_group: list[NormalizedSample] = []
    prev_sample: NormalizedSample | None = None
    for sample in ordered_samples:
        if prev_sample is not None and int(sample.sample_index) != int(prev_sample.sample_index) + 1:
            raw_groups.append(current_group)
            current_group = []
        current_group.append(sample)
        prev_sample = sample
    if current_group:
        raw_groups.append(current_group)

    grouped_samples: list[NormalizedSample] = []
    groups: list[NormalizedGroup] = []
    for raw_group in raw_groups:
        start_sample_index = int(raw_group[0].sample_index)
        end_sample_index = int(raw_group[-1].sample_index)
        group_id = f"derived:{start_sample_index}-{end_sample_index}"
        group_length = len(raw_group)
        resolved_samples = [
            replace(
                sample,
                group_id=group_id,
                group_frame_index=frame_index,
                group_length=group_length,
            )
            for frame_index, sample in enumerate(raw_group)
        ]
        grouped_samples.extend(resolved_samples)
        groups.append(
            NormalizedGroup(
                group_id=group_id,
                run_name=resolved_samples[0].run_name,
                stage=resolved_samples[0].stage,
                samples=tuple(resolved_samples),
                start_sample_index=start_sample_index,
                end_sample_index=end_sample_index,
                metrics_summary=summarize_group_samples(resolved_samples),
            )
        )
    return grouped_samples, groups
