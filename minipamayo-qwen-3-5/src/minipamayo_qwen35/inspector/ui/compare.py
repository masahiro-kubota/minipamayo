"""Comparison page for the Streamlit eval inspector."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ..models import NormalizedRun, NormalizedSample
from ..plotting import build_trajectory_overlay_figure
from ..registry import compare_runs_by_sample_id

STAGE_LABELS = {
    "stage1a_eval": "Stage1A Eval",
    "stage1b_eval": "Stage1B Eval",
    "stage2_eval": "Stage2 Eval",
    "stage2_inference": "Stage2 Inference",
    "stage3_eval": "Stage3 Eval",
}


def _render_stage_sample(
    column,
    *,
    title: str,
    run: NormalizedRun | None,
    sample: NormalizedSample | None,
) -> None:
    with column:
        st.markdown(f"**{title}**")
        if run is not None:
            st.caption(run.run_name)
        if sample is None:
            st.caption("No sample for this artifact.")
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


def _selected_run_rows(compare_runs: dict[str, NormalizedRun | None]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for stage in ["stage1a_eval", "stage1b_eval", "stage2_eval", "stage2_inference", "stage3_eval"]:
        run = compare_runs.get(stage)
        if run is None:
            continue
        rows.append(
            {
                "stage": STAGE_LABELS.get(stage, stage),
                "run_name": run.run_name,
                "summary_json": Path(run.manifest.summary_json).name,
                "samples": len(run.samples),
                "payload": "per-sample" if run.manifest.per_sample_jsonl else "summary-only",
            }
        )
    return rows


def render_compare(
    *,
    compare_runs: dict[str, NormalizedRun | None],
    sample_id_hint: str = "",
) -> None:
    st.subheader("Compare")

    selected_rows = _selected_run_rows(compare_runs)
    if not selected_rows:
        st.info("Select at least one artifact in Compare mode.")
        return

    st.markdown("**Compared Artifacts**")
    st.dataframe(pd.DataFrame(selected_rows), use_container_width=True, hide_index=True)

    stage1a_run = compare_runs.get("stage1a_eval")
    stage1b_run = compare_runs.get("stage1b_eval")
    stage2_eval_run = compare_runs.get("stage2_eval")
    stage2_inference_run = compare_runs.get("stage2_inference")
    stage3_eval_run = compare_runs.get("stage3_eval")

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
    if stage3_eval_run is not None and stage3_eval_run.invalid_reason:
        invalid_messages.append(f"Stage3 Eval excluded: {stage3_eval_run.invalid_reason}")
        stage3_eval_run = None
    for message in invalid_messages:
        st.warning(message)

    rows = compare_runs_by_sample_id(
        stage1a_run=stage1a_run,
        stage1b_run=stage1b_run,
        stage2_eval_run=stage2_eval_run,
        stage2_inference_run=stage2_inference_run,
        stage3_eval_run=stage3_eval_run,
    )
    if not rows:
        st.info("No selected runs are available for sample-level comparison.")
        return

    overlap_df = pd.DataFrame(
        [
            {
                "rows": len(rows),
                "stage1a_present": sum(row["stage1a"] is not None for row in rows),
                "stage1b_present": sum(row["stage1b"] is not None for row in rows),
                "stage2_eval_present": sum(row["stage2_eval"] is not None for row in rows),
                "stage2_inference_present": sum(row["stage2_inference"] is not None for row in rows),
                "stage3_present": sum(row["stage3"] is not None for row in rows),
            }
        ]
    )
    st.markdown("**Overlap Summary**")
    st.dataframe(overlap_df, use_container_width=True, hide_index=True)

    sample_ids = [row["sample_id"] for row in rows]
    state_key = "compare_mode_sample_id"
    hint_key = "compare_mode_sample_id_hint"
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
        index=(
            sample_ids.index(st.session_state[state_key])
            if st.session_state[state_key] in sample_ids
            else default_index
        ),
        key=state_key,
    )
    selected_row = next(row for row in rows if row["sample_id"] == selected_sample_id)
    stage1a_col, stage1b_col, stage2_col, stage3_col = st.columns(4)
    _render_stage_sample(
        stage1a_col,
        title="Stage1A Eval",
        run=stage1a_run,
        sample=selected_row["stage1a"],
    )
    _render_stage_sample(
        stage1b_col,
        title="Stage1B Eval",
        run=stage1b_run,
        sample=selected_row["stage1b"],
    )
    with stage2_col:
        st.markdown("**Stage2**")
        if stage2_eval_run is not None:
            st.caption(f"Eval: {stage2_eval_run.run_name}")
        if stage2_inference_run is not None:
            st.caption(f"Inference: {stage2_inference_run.run_name}")
        if selected_row["stage2_eval"] is None and selected_row["stage2_inference"] is None:
            st.caption("No Stage2 sample for this sample_id.")
        else:
            _render_stage_sample(
                st.container(),
                title="Teacher-Forced Eval",
                run=stage2_eval_run,
                sample=selected_row["stage2_eval"],
            )
            st.divider()
            _render_stage_sample(
                st.container(),
                title="Generated Inference",
                run=stage2_inference_run,
                sample=selected_row["stage2_inference"],
            )
    _render_stage_sample(
        stage3_col,
        title="Stage3 Eval",
        run=stage3_eval_run,
        sample=selected_row["stage3"],
    )
