"""Raw JSON page for the Streamlit eval inspector."""

from __future__ import annotations

import streamlit as st

from ..models import NormalizedRun, NormalizedSample
from ..registry import sample_lookup


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
    counterpart_run: NormalizedRun | None = None,
) -> None:
    st.subheader("Raw JSON")
    selected_sample = _pick_sample(filtered_samples, sample_id_hint)
    manifest_col, summary_col = st.columns(2)
    manifest_col.markdown("**Manifest**")
    manifest_col.json(active_run.manifest.to_dict())
    summary_col.markdown("**Summary**")
    summary_col.json(active_run.summary)
    if selected_sample is not None:
        st.markdown("**Selected Sample**")
        st.json(selected_sample.raw)
    if counterpart_run is not None and selected_sample is not None:
        matched = sample_lookup(counterpart_run).get(selected_sample.sample_id)
        if matched is not None:
            st.markdown("**Matched Counterpart Sample**")
            st.json(matched.raw)
