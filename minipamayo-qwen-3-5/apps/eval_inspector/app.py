"""Streamlit app for browsing local eval and inference artifacts."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from minipamayo_qwen35.inspector.backfill import backfill_artifact_manifests
from minipamayo_qwen35.inspector.manifests import load_manifest
from minipamayo_qwen35.inspector.models import ArtifactManifest, NormalizedRun, NormalizedSample
from minipamayo_qwen35.inspector.registry import default_artifact_root, load_manifest_registry, load_normalized_run
from minipamayo_qwen35.inspector.ui import (
    render_active_artifact_header,
    render_block_browser,
    render_compare,
    render_compare_selectors,
    render_filter_controls,
    render_inspect_selectors,
    render_raw_json,
    render_registry_controls,
    render_sample_browser,
    render_summary,
)


st.set_page_config(page_title="Eval Inspector", layout="wide")

COMPARE_STAGE_KEYS = [
    "stage1a_eval",
    "stage1b_eval",
    "stage2_eval",
    "stage2_inference",
    "stage3_eval",
]


@st.cache_data(show_spinner=False)
def cached_registry(artifact_root: str):
    return load_manifest_registry(artifact_root)


@st.cache_data(show_spinner=False)
def cached_run(manifest_path: str) -> NormalizedRun:
    manifest = load_manifest(manifest_path)
    return load_normalized_run(manifest)


def _load_selected_run(manifest: ArtifactManifest | None) -> NormalizedRun | None:
    if manifest is None:
        return None
    try:
        return cached_run(str(manifest.manifest_path))
    except Exception as exc:
        st.error(f"Failed to load `{manifest.run_name}`: {exc}")
        return None


def _sample_metric_value(sample: NormalizedSample, metric_name: str | None) -> float | None:
    if metric_name is None:
        return None
    value = getattr(sample, metric_name, None)
    if value is None:
        return None
    return float(value)


def _sort_samples(
    samples: list[NormalizedSample],
    *,
    metric_name: str | None,
) -> list[NormalizedSample]:
    if metric_name is None:
        return sorted(samples, key=lambda sample: int(sample.sample_index))
    return sorted(samples, key=lambda sample: int(sample.sample_index))


def _filter_samples(
    active_run: NormalizedRun | None,
    filters: dict,
) -> list[NormalizedSample]:
    if active_run is None:
        return []
    filtered: list[NormalizedSample] = []
    command_filter = set(filters.get("command_filter", []))
    metric_name = filters.get("metric_name")
    metric_range = filters.get("metric_range")
    for sample in active_run.samples:
        if command_filter and sample.command and sample.command not in command_filter:
            continue
        metric_value = _sample_metric_value(sample, metric_name)
        if metric_name is not None:
            if metric_value is None:
                continue
            if metric_range is not None:
                low, high = metric_range
                if metric_value < float(low) or metric_value > float(high):
                    continue
        filtered.append(sample)
    return _sort_samples(
        filtered,
        metric_name=metric_name,
    )


def _filtered_rows(samples: list[NormalizedSample]) -> pd.DataFrame:
    return pd.DataFrame([sample.to_row() for sample in samples])


def main() -> None:
    st.title("Eval Inspector")
    default_root = str(default_artifact_root())
    artifact_root, request_backfill, view_mode = render_registry_controls(default_root)
    if request_backfill:
        written_paths = backfill_artifact_manifests(artifact_root)
        cached_registry.clear()
        cached_run.clear()
        st.sidebar.success(f"Backfilled {len(written_paths)} manifests.")

    registry = cached_registry(artifact_root)
    if not registry.manifests:
        st.warning("No inspector manifests found. Run backfill or generate new eval artifacts first.")
        return

    if view_mode == "compare":
        selected_manifests = render_compare_selectors(registry)
        compare_runs = {
            stage: _load_selected_run(selected_manifests.get(stage))
            for stage in COMPARE_STAGE_KEYS
        }
        render_compare(compare_runs=compare_runs)
        return

    selected = render_inspect_selectors(registry)
    active_manifest = selected.get("active_manifest")
    if active_manifest is None:
        st.info("Select an active artifact from the sidebar.")
        return

    active_run = _load_selected_run(active_manifest)
    if active_run is None:
        st.info("The selected artifact could not be loaded.")
        return

    render_active_artifact_header(active_run)
    filters = render_filter_controls(active_run)
    filtered_samples = _filter_samples(active_run, filters)
    filtered_rows = _filtered_rows(filtered_samples)

    summary_tab, block_tab, samples_tab, raw_tab = st.tabs(
        ["Summary", "Curve Blocks", "Samples", "Raw"]
    )
    selected_block_sample_id = str(filters.get("sample_id_jump", ""))
    with summary_tab:
        render_summary(active_run)
    with block_tab:
        selected_block_sample = render_block_browser(
            active_run=active_run,
            filtered_samples=filtered_samples,
            filters=filters,
        )
        if selected_block_sample is not None:
            selected_block_sample_id = selected_block_sample.sample_id
    with samples_tab:
        render_sample_browser(
            active_run=active_run,
            filtered_samples=filtered_samples,
            filtered_rows=filtered_rows,
            sample_id_hint=selected_block_sample_id,
        )
    with raw_tab:
        render_raw_json(
            active_run=active_run,
            filtered_samples=filtered_samples,
            sample_id_hint=selected_block_sample_id,
        )


if __name__ == "__main__":
    main()
