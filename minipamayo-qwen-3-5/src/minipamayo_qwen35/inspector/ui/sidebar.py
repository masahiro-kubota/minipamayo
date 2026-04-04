"""Sidebar controls for the Streamlit eval inspector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from ..models import ArtifactManifest, NormalizedRun
from ..registry import ManifestRegistry

ARTIFACT_STAGE_LABELS = {
    "stage1a_eval": "Stage1A Eval",
    "stage1b_eval": "Stage1B Eval",
    "stage2_eval": "Stage2 Eval",
    "stage2_inference": "Stage2 Inference",
    "stage3_eval": "Stage3 Eval",
}

VIEW_MODE_LABELS = {
    "inspect": "Inspect",
    "compare": "Compare",
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

MANIFEST_ATTRS_BY_STAGE = {
    "stage1a_eval": "stage1a_manifests",
    "stage1b_eval": "stage1b_manifests",
    "stage2_eval": "stage2_eval_manifests",
    "stage2_inference": "stage2_inference_manifests",
    "stage3_eval": "stage3_eval_manifests",
}


def _manifest_label(manifest: ArtifactManifest, *, include_stage: bool) -> str:
    stage_label = ARTIFACT_STAGE_LABELS.get(manifest.stage, manifest.stage)
    summary_name = Path(manifest.summary_json).name
    suffix = "per-sample" if manifest.per_sample_jsonl else "summary-only"
    prefix = f"{stage_label} | " if include_stage else ""
    return f"{prefix}{manifest.run_name} [{summary_name} | {suffix}]"


def _select_manifest(
    *,
    label: str,
    manifests: tuple[ArtifactManifest, ...],
    key: str,
    include_stage: bool,
) -> ArtifactManifest | None:
    if not manifests:
        st.caption(f"{label}: no artifacts")
        return None
    labels = [_manifest_label(manifest, include_stage=include_stage) for manifest in manifests]
    label_to_manifest = {display: manifest for display, manifest in zip(labels, manifests)}
    selected = st.selectbox(label, labels, key=key)
    return label_to_manifest[selected]


def _manifests_for_stage(registry: ManifestRegistry, stage: str) -> tuple[ArtifactManifest, ...]:
    attr_name = MANIFEST_ATTRS_BY_STAGE[stage]
    return getattr(registry, attr_name)


def _available_stage_keys(registry: ManifestRegistry) -> list[str]:
    return [
        stage
        for stage in MANIFEST_ATTRS_BY_STAGE
        if _manifests_for_stage(registry, stage)
    ]


def render_registry_controls(default_artifact_root: str) -> tuple[str, bool, str]:
    with st.sidebar:
        st.header("Inspector")
        artifact_root = st.text_input("Artifact Root", value=default_artifact_root)
        request_backfill = st.button("Backfill Manifests")
        view_mode = st.radio(
            "View Mode",
            options=["inspect", "compare"],
            index=0,
            format_func=lambda key: VIEW_MODE_LABELS[key],
            key="inspector_view_mode",
        )
    return artifact_root, request_backfill, str(view_mode)


def render_inspect_selectors(registry: ManifestRegistry) -> dict[str, ArtifactManifest | None]:
    with st.sidebar:
        st.header("Inspect")
        available_stage_keys = _available_stage_keys(registry)
        if not available_stage_keys:
            st.caption("No artifacts are available.")
            return {"active_manifest": None}
        active_stage = st.selectbox(
            "Active Stage",
            options=available_stage_keys,
            format_func=lambda key: ARTIFACT_STAGE_LABELS[key],
            key="inspector_active_stage",
        )
        manifests = _manifests_for_stage(registry, str(active_stage))
        active_manifest = _select_manifest(
            label="Active Artifact",
            manifests=manifests,
            key=f"inspector_active_manifest_{active_stage}",
            include_stage=False,
        )
    return {"active_manifest": active_manifest}


def render_compare_selectors(registry: ManifestRegistry) -> dict[str, ArtifactManifest | None]:
    with st.sidebar:
        st.header("Compare")
        selected: dict[str, ArtifactManifest | None] = {}
        available_stage_keys = _available_stage_keys(registry)
        for stage in available_stage_keys:
            selected[stage] = _select_manifest(
                label=f"{ARTIFACT_STAGE_LABELS[stage]} Artifact",
                manifests=_manifests_for_stage(registry, stage),
                key=f"inspector_compare_manifest_{stage}",
                include_stage=False,
            )
    return selected


def render_filter_controls(active_run: NormalizedRun | None) -> dict[str, Any]:
    with st.sidebar:
        st.header("Filters")
        sample_id_jump = st.text_input("Sample ID Jump", value="")
        if active_run is None:
            st.caption("No active artifact.")
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
