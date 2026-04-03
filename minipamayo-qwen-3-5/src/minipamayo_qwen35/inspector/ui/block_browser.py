"""Block-first browser for grouped curve inspection."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..grouping import summarize_group_samples
from ..models import NormalizedGroup, NormalizedRun, NormalizedSample
from ..plotting import build_group_metric_timeline_figure, build_trajectory_overlay_figure


def _group_metric_value(group: NormalizedGroup, metric_name: str | None, *, mode: str) -> float:
    if metric_name == "token_accuracy":
        values = [
            float(sample.token_accuracy)
            for sample in group.samples
            if sample.token_accuracy is not None
        ]
        if not values:
            return float("inf")
        return min(values) if mode == "worst_metric" else sum(values) / len(values)
    attr_name = metric_name or "fde_m"
    values = [
        float(getattr(sample, attr_name))
        for sample in group.samples
        if getattr(sample, attr_name) is not None
    ]
    if not values:
        return float("-inf")
    return max(values) if mode == "worst_metric" else sum(values) / len(values)


def _filtered_groups(
    active_run: NormalizedRun,
    filtered_samples: list[NormalizedSample],
) -> list[NormalizedGroup]:
    if not filtered_samples:
        return []
    run_groups = {group.group_id: group for group in active_run.groups}
    samples_by_group: dict[str, list[NormalizedSample]] = {}
    for sample in filtered_samples:
        samples_by_group.setdefault(sample.group_id, []).append(sample)
    groups: list[NormalizedGroup] = []
    for group_id, samples in samples_by_group.items():
        ordered_samples = sorted(samples, key=lambda sample: int(sample.group_frame_index))
        source_group = run_groups.get(group_id)
        if source_group is None:
            start_sample_index = int(ordered_samples[0].sample_index)
            end_sample_index = int(ordered_samples[-1].sample_index)
        else:
            start_sample_index = int(source_group.start_sample_index)
            end_sample_index = int(source_group.end_sample_index)
        groups.append(
            NormalizedGroup(
                group_id=group_id,
                run_name=active_run.run_name,
                stage=active_run.stage,
                samples=tuple(ordered_samples),
                start_sample_index=start_sample_index,
                end_sample_index=end_sample_index,
                metrics_summary=summarize_group_samples(ordered_samples),
            )
        )
    return groups


def _sort_groups(
    groups: list[NormalizedGroup],
    *,
    block_sort: str,
    metric_name: str | None,
) -> list[NormalizedGroup]:
    if block_sort == "length_desc":
        return sorted(groups, key=lambda group: (-group.length, group.start_sample_index))
    if block_sort in {"worst_metric", "mean_metric"}:
        return sorted(
            groups,
            key=lambda group: (
                -_group_metric_value(group, metric_name, mode=block_sort),
                group.start_sample_index,
            ),
        )
    return sorted(groups, key=lambda group: group.start_sample_index)


def _group_label(group: NormalizedGroup) -> str:
    worst_fde = group.metrics_summary.get("max_fde_m")
    worst_text = f"{float(worst_fde):.2f}" if worst_fde is not None else "-"
    return (
        f"{group.group_id} | len={group.length} | "
        f"start={group.start_sample_index} | end={group.end_sample_index} | worst_fde={worst_text}"
    )


def _sync_group_selection(
    groups: list[NormalizedGroup],
    *,
    state_key: str,
) -> int:
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    selected_index = int(st.session_state[state_key])
    selected_index = max(0, min(selected_index, len(groups) - 1))
    st.session_state[state_key] = selected_index
    return selected_index


def _sync_frame_selection(
    group: NormalizedGroup,
    *,
    state_key: str,
    sample_id_hint: str,
) -> int:
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    if sample_id_hint:
        for idx, sample in enumerate(group.samples):
            if sample.sample_id == sample_id_hint:
                st.session_state[state_key] = idx
                break
    selected_index = int(st.session_state[state_key])
    selected_index = max(0, min(selected_index, len(group.samples) - 1))
    st.session_state[state_key] = selected_index
    return selected_index


def _render_sample_metrics(sample: NormalizedSample) -> None:
    metric_cols = st.columns(5)
    metric_cols[0].metric("Sample ID", sample.sample_id)
    metric_cols[1].metric("Command", sample.command or "-")
    metric_cols[2].metric("ADE", f"{sample.ade_m:.3f}" if sample.ade_m is not None else "-")
    metric_cols[3].metric("FDE", f"{sample.fde_m:.3f}" if sample.fde_m is not None else "-")
    metric_cols[4].metric(
        "Token Acc",
        f"{sample.token_accuracy:.3f}" if sample.token_accuracy is not None else "-",
    )
    extra_cols = st.columns(3)
    extra_cols[0].metric(
        "Kappa MAE",
        f"{sample.action_mae_kappa:.4f}" if sample.action_mae_kappa is not None else "-",
    )
    extra_cols[1].metric("Source Frame", sample.source_frame_id or "-")
    extra_cols[2].metric(
        "Block Frame",
        f"{sample.group_frame_index + 1} / {sample.group_length}",
    )
    if sample.reasoning_text_gt:
        st.markdown("**Ground Truth Reasoning**")
        st.code(sample.reasoning_text_gt)
    if sample.reasoning_text_pred:
        st.markdown("**Predicted Reasoning**")
        st.code(sample.reasoning_text_pred)


def render_block_browser(
    *,
    active_run: NormalizedRun,
    filtered_samples: list[NormalizedSample],
    filters: dict,
) -> NormalizedSample | None:
    st.subheader("Curve Block Browser")
    if active_run.browser_unavailable_reason:
        st.info(active_run.browser_unavailable_reason)
        return None
    if not filtered_samples:
        st.info("No samples matched the current filters.")
        return None

    groups = _sort_groups(
        _filtered_groups(active_run, filtered_samples),
        block_sort=str(filters.get("block_sort", "dataset_order")),
        metric_name=filters.get("metric_name"),
    )
    if not groups:
        st.info("No blocks matched the current filters.")
        return None

    group_state_key = f"block_browser_group_index_{active_run.stage}_{active_run.run_name}"
    frame_state_key = f"block_browser_frame_index_{active_run.stage}_{active_run.run_name}"
    sample_state_key = f"block_browser_sample_id_{active_run.stage}_{active_run.run_name}"
    group_widget_key = f"{group_state_key}_select"
    frame_widget_key = f"{frame_state_key}_slider"
    active_group_key = f"{group_state_key}_active_group_id"

    if group_widget_key not in st.session_state:
        st.session_state[group_widget_key] = 0
    selected_group_index = _sync_group_selection(groups, state_key=group_widget_key)
    block_prev_col, block_select_col, block_next_col = st.columns([1, 6, 1])
    if block_prev_col.button("Prev Block", key=f"{group_state_key}_prev", disabled=selected_group_index <= 0):
        st.session_state[group_widget_key] = max(0, int(st.session_state[group_widget_key]) - 1)
    if block_next_col.button(
        "Next Block",
        key=f"{group_state_key}_next",
        disabled=selected_group_index >= len(groups) - 1,
    ):
        st.session_state[group_widget_key] = min(
            len(groups) - 1,
            int(st.session_state[group_widget_key]) + 1,
        )
    selected_group_index = block_select_col.selectbox(
        "Block",
        options=list(range(len(groups))),
        format_func=lambda idx: _group_label(groups[idx]),
        key=group_widget_key,
    )
    selected_group = groups[int(selected_group_index)]
    if st.session_state.get(active_group_key) != selected_group.group_id:
        st.session_state[active_group_key] = selected_group.group_id
        st.session_state[frame_widget_key] = 0

    if frame_widget_key not in st.session_state:
        st.session_state[frame_widget_key] = 0
    selected_frame_index = _sync_frame_selection(
        selected_group,
        state_key=frame_widget_key,
        sample_id_hint=str(filters.get("sample_id_jump", "")),
    )
    frame_prev_col, frame_slider_col, frame_next_col = st.columns([1, 6, 1])
    if frame_prev_col.button("Prev Frame", key=f"{frame_state_key}_prev", disabled=selected_frame_index <= 0):
        st.session_state[frame_widget_key] = max(0, int(st.session_state[frame_widget_key]) - 1)
    if frame_next_col.button(
        "Next Frame",
        key=f"{frame_state_key}_next",
        disabled=selected_frame_index >= len(selected_group.samples) - 1,
    ):
        st.session_state[frame_widget_key] = min(
            len(selected_group.samples) - 1,
            int(st.session_state[frame_widget_key]) + 1,
        )
    selected_frame_index = frame_slider_col.slider(
        "Frame In Block",
        min_value=0,
        max_value=len(selected_group.samples) - 1,
        value=int(st.session_state[frame_widget_key]),
        key=frame_widget_key,
    )
    selected_sample = selected_group.samples[int(selected_frame_index)]
    st.session_state[sample_state_key] = selected_sample.sample_id

    st.caption(
        f"Selected block: {selected_group.group_id} | "
        f"{selected_group.start_sample_index}-{selected_group.end_sample_index}"
    )
    image_col, trajectory_col = st.columns([1.3, 0.7])
    if selected_sample.image_path and Path(selected_sample.image_path).exists():
        image_col.image(selected_sample.image_path, caption=selected_sample.sample_id, use_container_width=True)
    else:
        image_col.caption("Image missing.")
    if selected_sample.gt_waypoints or selected_sample.pred_waypoints or selected_sample.pid_pred_waypoints:
        trajectory_figure = build_trajectory_overlay_figure(selected_sample)
        try:
            trajectory_col.pyplot(trajectory_figure, clear_figure=True, use_container_width=False)
        finally:
            trajectory_figure.clear()
    else:
        trajectory_col.caption("No trajectory payload for this artifact.")

    _render_sample_metrics(selected_sample)

    st.markdown("**Block Metric Timeline**")
    timeline_figure = build_group_metric_timeline_figure(
        selected_group,
        current_frame_index=int(selected_frame_index),
    )
    try:
        st.pyplot(timeline_figure, clear_figure=True, use_container_width=True)
    finally:
        timeline_figure.clear()
    return selected_sample
