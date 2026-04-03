"""Cross-stage comparison page for the Streamlit eval inspector."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ..models import NormalizedRun, NormalizedSample
from ..plotting import build_trajectory_overlay_figure
from ..registry import compare_runs_by_sample_id


def _render_stage_sample(column, *, title: str, sample: NormalizedSample | None) -> None:
    with column:
        st.markdown(f"**{title}**")
        if sample is None:
            st.caption("No sample for this stage.")
            return
        if sample.image_path and Path(sample.image_path).exists():
            st.image(sample.image_path, caption=sample.sample_id, use_container_width=True)
        if sample.gt_waypoints or sample.pred_waypoints or sample.pid_pred_waypoints:
            figure = build_trajectory_overlay_figure(sample)
            try:
                st.pyplot(figure, clear_figure=True, use_container_width=False)
            finally:
                figure.clear()
        metrics = {
            "ADE": f"{sample.ade_m:.3f}" if sample.ade_m is not None else "-",
            "FDE": f"{sample.fde_m:.3f}" if sample.fde_m is not None else "-",
            "Token Acc": f"{sample.token_accuracy:.3f}" if sample.token_accuracy is not None else "-",
            "Kappa MAE": (
                f"{sample.action_mae_kappa:.4f}" if sample.action_mae_kappa is not None else "-"
            ),
        }
        st.json(metrics)
        if sample.reasoning_text_pred:
            st.markdown("**Predicted Reasoning**")
            st.code(sample.reasoning_text_pred)
        if sample.reasoning_text_gt:
            st.markdown("**Ground Truth Reasoning**")
            st.code(sample.reasoning_text_gt)


def render_compare(
    *,
    stage1a_run: NormalizedRun | None,
    stage1b_run: NormalizedRun | None,
    stage2_eval_run: NormalizedRun | None,
    stage2_inference_run: NormalizedRun | None,
    sample_id_hint: str = "",
) -> None:
    st.subheader("Cross-Stage Compare")
    invalid_messages: list[str] = []
    if stage1a_run is not None and stage1a_run.invalid_reason:
        invalid_messages.append(f"Stage1A excluded: {stage1a_run.invalid_reason}")
        stage1a_run = None
    if stage1b_run is not None and stage1b_run.invalid_reason:
        invalid_messages.append(f"Stage1B excluded: {stage1b_run.invalid_reason}")
        stage1b_run = None
    if stage2_eval_run is not None and stage2_eval_run.invalid_reason:
        invalid_messages.append(f"Stage2 Eval excluded: {stage2_eval_run.invalid_reason}")
        stage2_eval_run = None
    if stage2_inference_run is not None and stage2_inference_run.invalid_reason:
        invalid_messages.append(f"Stage2 Inference excluded: {stage2_inference_run.invalid_reason}")
        stage2_inference_run = None
    for message in invalid_messages:
        st.warning(message)
    rows = compare_runs_by_sample_id(
        stage1a_run=stage1a_run,
        stage1b_run=stage1b_run,
        stage2_eval_run=stage2_eval_run,
        stage2_inference_run=stage2_inference_run,
    )
    if not rows:
        st.info("No selected runs are available for comparison.")
        return
    overlap_df = pd.DataFrame(
        [
            {
                "rows": len(rows),
                "stage1a_present": sum(row["stage1a"] is not None for row in rows),
                "stage1b_present": sum(row["stage1b"] is not None for row in rows),
                "stage2_eval_present": sum(row["stage2_eval"] is not None for row in rows),
                "stage2_inference_present": sum(
                    row["stage2_inference"] is not None for row in rows
                ),
            }
        ]
    )
    st.dataframe(overlap_df, use_container_width=True, hide_index=True)
    sample_ids = [row["sample_id"] for row in rows]
    state_key = "compare_sample_id"
    hint_key = "compare_sample_id_hint"
    default_index = sample_ids.index(sample_id_hint) if sample_id_hint in sample_ids else 0
    if state_key not in st.session_state:
        st.session_state[state_key] = sample_ids[default_index]
        st.session_state[hint_key] = sample_id_hint
    elif sample_id_hint and st.session_state.get(hint_key) != sample_id_hint and sample_id_hint in sample_ids:
        st.session_state[state_key] = sample_id_hint
        st.session_state[hint_key] = sample_id_hint
    selected_sample_id = st.selectbox(
        "Compare Sample ID",
        options=sample_ids,
        index=sample_ids.index(st.session_state[state_key]) if st.session_state[state_key] in sample_ids else default_index,
        key=state_key,
    )
    selected_row = next(row for row in rows if row["sample_id"] == selected_sample_id)
    stage1a_col, stage1b_col, stage2_col = st.columns(3)
    _render_stage_sample(stage1a_col, title="Stage1A", sample=selected_row["stage1a"])
    _render_stage_sample(stage1b_col, title="Stage1B", sample=selected_row["stage1b"])
    with stage2_col:
        st.markdown("**Stage2**")
        if selected_row["stage2_eval"] is None and selected_row["stage2_inference"] is None:
            st.caption("No Stage2 sample for this sample_id.")
        else:
            _render_stage_sample(st.container(), title="Teacher-Forced Eval", sample=selected_row["stage2_eval"])
            st.divider()
            _render_stage_sample(st.container(), title="Generated Inference", sample=selected_row["stage2_inference"])
