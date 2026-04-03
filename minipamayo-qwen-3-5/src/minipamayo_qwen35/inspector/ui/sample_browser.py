"""Sample browser page for the Streamlit eval inspector."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..models import NormalizedRun, NormalizedSample
from ..plotting import build_trajectory_overlay_figure
from ..registry import sample_lookup


def _sample_label(sample: NormalizedSample) -> str:
    if sample.fde_m is not None:
        return f"{sample.sample_id} | FDE={sample.fde_m:.2f}"
    if sample.token_accuracy is not None:
        return f"{sample.sample_id} | token_acc={sample.token_accuracy:.3f}"
    return sample.sample_id


def _initial_index(samples: list[NormalizedSample], sample_id_hint: str) -> int:
    if sample_id_hint:
        for idx, sample in enumerate(samples):
            if sample.sample_id == sample_id_hint:
                return idx
    return 0


def _select_sample(
    *,
    samples: list[NormalizedSample],
    state_key: str,
    sample_id_hint: str,
) -> NormalizedSample | None:
    if not samples:
        return None
    if state_key not in st.session_state:
        st.session_state[state_key] = _initial_index(samples, sample_id_hint)
    state_index = int(st.session_state[state_key])
    prev_col, main_col, next_col = st.columns([1, 5, 1])
    if prev_col.button("Prev", key=f"{state_key}_prev", disabled=state_index <= 0):
        state_index -= 1
    if next_col.button("Next", key=f"{state_key}_next", disabled=state_index >= len(samples) - 1):
        state_index += 1
    selected_index = main_col.selectbox(
        "Sample",
        options=list(range(len(samples))),
        index=state_index,
        format_func=lambda idx: _sample_label(samples[idx]),
        key=f"{state_key}_select",
    )
    st.session_state[state_key] = int(selected_index)
    return samples[int(selected_index)]


def _render_sample_panel(sample: NormalizedSample, *, title: str) -> None:
    st.markdown(f"**{title}**")
    top_left, top_right = st.columns([1, 1])
    if sample.image_path and Path(sample.image_path).exists():
        top_left.image(sample.image_path, caption=sample.sample_id, use_container_width=True)
    else:
        top_left.caption("Image missing.")

    if sample.gt_waypoints or sample.pred_waypoints or sample.pid_pred_waypoints:
        figure = build_trajectory_overlay_figure(sample)
        try:
            top_right.pyplot(figure, clear_figure=True, use_container_width=True)
        finally:
            figure.clear()
    else:
        top_right.caption("No trajectory payload for this artifact.")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Sample ID", sample.sample_id)
    metric_cols[1].metric("Command", sample.command or "-")
    metric_cols[2].metric("ADE", f"{sample.ade_m:.3f}" if sample.ade_m is not None else "-")
    metric_cols[3].metric("FDE", f"{sample.fde_m:.3f}" if sample.fde_m is not None else "-")
    metric_cols[4].metric(
        "Token Acc",
        f"{sample.token_accuracy:.3f}" if sample.token_accuracy is not None else "-",
    )
    extra_cols = st.columns(2)
    extra_cols[0].metric(
        "Kappa MAE",
        f"{sample.action_mae_kappa:.4f}" if sample.action_mae_kappa is not None else "-",
    )
    extra_cols[1].metric(
        "Max Lateral",
        f"{sample.max_lateral_error_m:.4f}" if sample.max_lateral_error_m is not None else "-",
    )
    if sample.reasoning_text_gt:
        st.markdown("**Ground Truth Reasoning**")
        st.code(sample.reasoning_text_gt)
    if sample.reasoning_text_pred:
        st.markdown("**Predicted Reasoning**")
        st.code(sample.reasoning_text_pred)


def render_sample_browser(
    *,
    active_run: NormalizedRun,
    filtered_samples: list[NormalizedSample],
    sample_id_hint: str = "",
    counterpart_run: NormalizedRun | None = None,
) -> None:
    st.subheader("Sample Browser")
    if not filtered_samples:
        st.info("No samples matched the current filters.")
        return
    selected_sample = _select_sample(
        samples=filtered_samples,
        state_key=f"sample_browser_{active_run.stage}_{active_run.run_name}",
        sample_id_hint=sample_id_hint,
    )
    if selected_sample is None:
        st.info("No sample selected.")
        return

    _render_sample_panel(selected_sample, title=f"Active Artifact: {active_run.run_name}")

    if counterpart_run is not None:
        matched = sample_lookup(counterpart_run).get(selected_sample.sample_id)
        if matched is not None:
            st.divider()
            _render_sample_panel(matched, title=f"Matched Artifact: {counterpart_run.run_name}")
