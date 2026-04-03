"""Sidebar controls for the Streamlit eval inspector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from ..models import ArtifactManifest, NormalizedRun
from ..registry import ManifestRegistry

STAGE_GROUP_LABELS = {
    "stage1a": "Stage1A",
    "stage1b": "Stage1B",
    "stage2": "Stage2",
}

FOCUS_STAGE_LABELS = {
    "stage1a_eval": "Stage1A Eval",
    "stage1b_eval": "Stage1B Eval",
    "stage2_eval": "Stage2 Eval",
    "stage2_inference": "Stage2 Inference",
}

BLOCK_SORT_LABELS = {
    "dataset_order": "Dataset Order",
    "worst_metric": "Worst Metric",
    "mean_metric": "Mean Metric",
    "length_desc": "Length",
}

FILTERABLE_METRICS = [
    ("fde_m", "FDE"),
    ("ade_m", "ADE"),
    ("max_lateral_error_m", "Max Lateral Error"),
    ("action_mae_kappa", "Kappa MAE"),
    ("token_accuracy", "Token Accuracy"),
]


def _manifest_label(manifest: ArtifactManifest) -> str:
    summary_name = Path(manifest.summary_json).name
    suffix = "summary-only" if manifest.per_sample_jsonl is None else "per-sample"
    return f"{manifest.run_name} [{summary_name} | {suffix}]"


def _select_manifest(
    *,
    label: str,
    manifests: tuple[ArtifactManifest, ...],
    key: str,
) -> ArtifactManifest | None:
    if not manifests:
        st.caption(f"{label}: no artifacts")
        return None
    labels = [_manifest_label(manifest) for manifest in manifests]
    label_to_manifest = {display: manifest for display, manifest in zip(labels, manifests)}
    selected = st.selectbox(label, labels, key=key)
    return label_to_manifest[selected]


def render_registry_controls(default_artifact_root: str) -> tuple[str, bool, list[str]]:
    with st.sidebar:
        st.header("Inspector")
        artifact_root = st.text_input("Artifact Root", value=default_artifact_root)
        request_backfill = st.button("Backfill Manifests")
        enabled_stage_groups = st.multiselect(
            "Stage Toggle",
            options=list(STAGE_GROUP_LABELS),
            default=list(STAGE_GROUP_LABELS),
            format_func=lambda key: STAGE_GROUP_LABELS[key],
        )
    return artifact_root, request_backfill, enabled_stage_groups


def render_run_selectors(
    registry: ManifestRegistry,
    *,
    enabled_stage_groups: list[str],
) -> dict[str, ArtifactManifest | str | None]:
    with st.sidebar:
        st.header("Runs")
        selected_stage1a = (
            _select_manifest(
                label="Stage1A Run",
                manifests=registry.stage1a_manifests,
                key="inspector_stage1a_manifest",
            )
            if "stage1a" in enabled_stage_groups
            else None
        )
        selected_stage1b = (
            _select_manifest(
                label="Stage1B Run",
                manifests=registry.stage1b_manifests,
                key="inspector_stage1b_manifest",
            )
            if "stage1b" in enabled_stage_groups
            else None
        )
        selected_stage2_eval = (
            _select_manifest(
                label="Stage2 Eval Run",
                manifests=registry.stage2_eval_manifests,
                key="inspector_stage2_eval_manifest",
            )
            if "stage2" in enabled_stage_groups
            else None
        )
        selected_stage2_inference = (
            _select_manifest(
                label="Stage2 Inference Run",
                manifests=registry.stage2_inference_manifests,
                key="inspector_stage2_inference_manifest",
            )
            if "stage2" in enabled_stage_groups
            else None
        )

        focus_stage_options: list[str] = []
        if selected_stage1a is not None:
            focus_stage_options.append("stage1a_eval")
        if selected_stage1b is not None:
            focus_stage_options.append("stage1b_eval")
        if selected_stage2_eval is not None:
            focus_stage_options.append("stage2_eval")
        if selected_stage2_inference is not None:
            focus_stage_options.append("stage2_inference")

        focus_stage = None
        if focus_stage_options:
            focus_stage = st.selectbox(
                "Focus Artifact",
                options=focus_stage_options,
                format_func=lambda key: FOCUS_STAGE_LABELS[key],
                key="inspector_focus_stage",
            )
    return {
        "stage1a_eval": selected_stage1a,
        "stage1b_eval": selected_stage1b,
        "stage2_eval": selected_stage2_eval,
        "stage2_inference": selected_stage2_inference,
        "focus_stage": focus_stage,
    }


def render_filter_controls(active_run: NormalizedRun | None) -> dict[str, Any]:
    with st.sidebar:
        st.header("Filters")
        sample_id_jump = st.text_input("Sample ID Jump", value="")
        if active_run is None:
            st.caption("No active run.")
            return {
                "sample_id_jump": sample_id_jump,
                "block_sort": "dataset_order",
                "command_filter": [],
                "metric_name": None,
                "metric_range": None,
            }
        if active_run.browser_unavailable_reason:
            st.caption(active_run.browser_unavailable_reason)
            return {
                "sample_id_jump": sample_id_jump,
                "block_sort": "dataset_order",
                "command_filter": [],
                "metric_name": None,
                "metric_range": None,
            }

        rows = [sample.to_row() for sample in active_run.samples]
        command_values = sorted({str(row["command"]) for row in rows if str(row["command"])})
        command_filter = st.multiselect(
            "Command Filter",
            options=command_values,
            default=command_values,
        )

        metric_options: list[str] = []
        metric_labels = {key: label for key, label in FILTERABLE_METRICS}
        metric_value_map: dict[str, list[float]] = {}
        for metric_name, _ in FILTERABLE_METRICS:
            values = [
                float(row[metric_name])
                for row in rows
                if row.get(metric_name) is not None
            ]
            if values:
                metric_options.append(metric_name)
                metric_value_map[metric_name] = values

        if not metric_options:
            return {
                "sample_id_jump": sample_id_jump,
                "block_sort": "dataset_order",
                "command_filter": command_filter,
                "metric_name": None,
                "metric_range": None,
            }

        metric_name = st.selectbox(
            "Metric Range",
            options=metric_options,
            format_func=lambda key: metric_labels[key],
        )
        min_value = float(min(metric_value_map[metric_name]))
        max_value = float(max(metric_value_map[metric_name]))
        if min_value == max_value:
            st.caption(f"{metric_labels[metric_name]} is constant at {min_value:.4f}.")
            metric_range = (min_value, max_value)
        else:
            metric_range = st.slider(
                f"{metric_labels[metric_name]} Range",
                min_value=min_value,
                max_value=max_value,
                value=(min_value, max_value),
            )
        block_sort = st.selectbox(
            "Block Sort",
            options=list(BLOCK_SORT_LABELS),
            format_func=lambda key: BLOCK_SORT_LABELS[key],
            index=0,
        )
    return {
        "sample_id_jump": sample_id_jump,
        "block_sort": block_sort,
        "command_filter": command_filter,
        "metric_name": metric_name,
        "metric_range": metric_range,
    }
