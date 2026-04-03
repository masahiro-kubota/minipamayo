"""Overview page for the Streamlit eval inspector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ..models import NormalizedRun

SUMMARY_PRIORITY = {
    "stage1a_eval": [
        "teacher_forced_loss",
        "teacher_forced_token_accuracy",
        "autoregressive_token_accuracy",
        "action_mae_kappa",
        "ade_m",
        "fde_m",
    ],
    "stage1b_eval": [
        "cfm_loss",
        "ade_m",
        "fde_m",
        "mean_max_lateral_error_m",
        "global_max_lateral_error_m",
        "action_mae_kappa",
        "pid_override/fde_m",
    ],
    "stage2_eval": [
        "metrics/loss",
        "metrics/token_accuracy",
    ],
    "stage2_inference": [
        "metrics/ade_m",
        "metrics/fde_m",
    ],
}


def _flatten_summary(payload: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        flat_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_summary(value, prefix=flat_key))
        else:
            flattened[flat_key] = value
    return flattened


def _summary_items(run: NormalizedRun) -> list[tuple[str, Any]]:
    flattened = _flatten_summary(run.summary)
    preferred = SUMMARY_PRIORITY.get(run.stage, [])
    items: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for key in preferred:
        if key in flattened:
            items.append((key, flattened[key]))
            seen.add(key)
    for key, value in sorted(flattened.items()):
        if key in seen:
            continue
        if isinstance(value, bool | int | float):
            items.append((key, value))
    return items[:12]


def _render_summary_cards(run: NormalizedRun) -> None:
    summary_items = _summary_items(run)
    if not summary_items:
        st.caption("No numeric summary metrics available.")
        return
    columns = st.columns(min(4, len(summary_items)))
    for idx, (key, value) in enumerate(summary_items):
        column = columns[idx % len(columns)]
        if isinstance(value, float):
            column.metric(key, f"{value:.4f}")
        else:
            column.metric(key, str(value))


def _render_plot_gallery(run: NormalizedRun) -> None:
    if not run.manifest.plots:
        st.caption("No plot artifacts attached to this run.")
        return
    st.subheader("Plots")
    image_items = sorted(
        (key, path)
        for key, path in run.manifest.plots.items()
        if Path(path).suffix.lower() == ".png"
    )
    json_items = sorted(
        (key, path)
        for key, path in run.manifest.plots.items()
        if Path(path).suffix.lower() == ".json"
    )
    if image_items:
        columns = st.columns(2)
        for idx, (key, path) in enumerate(image_items):
            with columns[idx % len(columns)]:
                st.caption(key)
                st.image(str(path), use_container_width=True)
    if json_items:
        for key, path in json_items:
            with st.expander(f"JSON Artifact: {key}"):
                st.code(str(path))
                st.json(json.loads(Path(path).read_text(encoding="utf-8")))


def _render_block_table(run: NormalizedRun) -> None:
    st.subheader("Blocks")
    if run.browser_unavailable_reason:
        st.info(run.browser_unavailable_reason)
        return
    if not run.groups:
        st.info("No blocks available for this artifact.")
        return
    block_rows = [group.to_row() for group in run.groups]
    st.dataframe(pd.DataFrame(block_rows), use_container_width=True, hide_index=True)


def render_overview(run: NormalizedRun, filtered_rows: pd.DataFrame) -> None:
    st.subheader("Overview")
    st.write(f"`{run.run_name}`")
    if run.invalid_reason:
        st.warning(run.invalid_reason)
    if run.manifest.wandb_run_url:
        st.link_button("Open W&B Run", run.manifest.wandb_run_url)
    _render_summary_cards(run)
    _render_plot_gallery(run)
    _render_block_table(run)
    st.subheader("Filtered Frames")
    if filtered_rows.empty:
        st.info("No samples matched the current filters.")
        return
    st.dataframe(filtered_rows, use_container_width=True, hide_index=True)
