"""
Streamlit preview UI.

Loads a JSONL manifest (output of the ingest+score pipeline), shows a grid of
images sorted by aesthetic score, lets the operator approve or reject each.
Reviews are persisted to the manifest via ManifestStore — closing the tab and
re-opening preserves state.

Run:
    streamlit run auracast/app/streamlit_app.py -- --manifest manifests/latest.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None  # noqa: N816

from auracast.persistence import ManifestStore
from auracast.schema.models import ProcessingStatus, ReviewStatus


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/latest.jsonl"),
        help="Path to a Manifest JSONL written by the pipeline.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover — Streamlit entry point
    if st is None:
        raise RuntimeError("Streamlit is not installed. `pip install streamlit`.")
    args = _parse_args()

    st.set_page_config(page_title="AuraCast", layout="wide")
    st.title("AuraCast — Curation Preview")

    if not args.manifest.exists():
        st.warning(f"No manifest at {args.manifest}. Run the pipeline first.")
        return

    # ManifestStore is the canonical read/write path. Streamlit's reruns make
    # this a hot path; we re-read on each render so that a separate pipeline
    # run (or another tab) is reflected without restart.
    store = ManifestStore(args.manifest)

    # ---- Sidebar: filters + counters -----------------------------------
    st.sidebar.metric("Total images", len(store))
    approved = sum(1 for x in store.all() if x.review_status == ReviewStatus.APPROVED)
    rejected = sum(1 for x in store.all() if x.review_status == ReviewStatus.REJECTED)
    pending = sum(1 for x in store.all() if x.review_status == ReviewStatus.PENDING)
    st.sidebar.metric("Approved", approved)
    st.sidebar.metric("Rejected", rejected)
    st.sidebar.metric("Pending", pending)
    st.sidebar.divider()

    status_filter = st.sidebar.multiselect(
        "Review status",
        options=[s.value for s in ReviewStatus],
        default=[ReviewStatus.PENDING.value, ReviewStatus.APPROVED.value],
    )
    min_score = st.sidebar.slider("Minimum aesthetic score", 0.0, 1.0, 0.0, 0.01)
    hide_failed = st.sidebar.checkbox("Hide failed", value=True)

    # ---- Filter + sort -------------------------------------------------
    items = store.all()
    items = [x for x in items if x.review_status.value in status_filter]
    if hide_failed:
        items = [x for x in items if x.processing_status != ProcessingStatus.FAILED]
    items = [x for x in items if (x.top_score() or 0.0) >= min_score]
    items.sort(key=lambda x: (x.top_score() or 0.0), reverse=True)

    st.write(f"Showing **{len(items)}** image(s).")

    cols_per_row = 4
    for row_start in range(0, len(items), cols_per_row):
        row = items[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, row):
            with col:
                rec = item.record
                if rec.file_path and rec.file_path.exists():
                    st.image(str(rec.file_path), use_container_width=True)
                else:
                    st.write("(image bytes unavailable)")
                score = item.top_score()
                st.caption(f"score = {score:.3f}" if score is not None else "unscored")
                badge = {
                    ReviewStatus.PENDING: "🟡 pending",
                    ReviewStatus.APPROVED: "✅ approved",
                    ReviewStatus.REJECTED: "❌ rejected",
                }[item.review_status]
                st.caption(f"`{rec.image_id.hex[:8]}` · {rec.source.value} · {badge}")

                if item.captions:
                    latest = item.captions[-1]
                    st.write(latest.caption)
                    if latest.attributes:
                        attr_str = " · ".join(f"**{k}:** {v}" for k, v in latest.attributes.items())
                        st.caption(attr_str)

                btn_cols = st.columns(2)
                if btn_cols[0].button("Approve", key=f"a-{rec.image_id}"):
                    store.update_review(rec.image_id, ReviewStatus.APPROVED)
                    st.rerun()
                if btn_cols[1].button("Reject", key=f"r-{rec.image_id}"):
                    store.update_review(rec.image_id, ReviewStatus.REJECTED)
                    st.rerun()


if __name__ == "__main__":  # pragma: no cover
    main()
