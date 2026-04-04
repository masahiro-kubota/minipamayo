"""Raw JSON page for the Streamlit eval inspector."""

from __future__ import annotations

import streamlit as st

from ..models import NormalizedRun, NormalizedSample


def _pick_sample(samples: list[NormalizedSample], sample_id_hint: str) -> NormalizedSample | None:
    if not samples:
        return None
    if sample_id_hint:
        for sample in samples:
            if sample.sample_id == sample_id_hint:
                return sample
    return samples[0]


def render_raw_json(
    *,
    active_run: NormalizedRun,
    filtered_samples: list[NormalizedSample],
    sample_id_hint: str = "",
) -> None:
    st.subheader("Raw")
    st.caption(f"Active Artifact: {active_run.run_name}")
    selected_sample = _pick_sample(filtered_samples, sample_id_hint)
    manifest_col, summary_col = st.columns(2)
    manifest_col.markdown("**Manifest**")
    manifest_col.json(active_run.manifest.to_dict())
    summary_col.markdown("**Summary**")
    summary_col.json(active_run.summary)
    if selected_sample is not None:
        st.markdown("**Selected Sample**")
        st.json(selected_sample.raw)
