"""
Streamlit preview UI.

Loads a JSONL manifest (output of the ingest+score pipeline), shows a grid of
images sorted by aesthetic score, lets the operator approve or reject each.

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

from auracast.schema.models import Manifest, ReviewStatus


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

    manifest = Manifest.from_jsonl(args.manifest.read_text())
    sorted_items = sorted(
        manifest.items,
        key=lambda x: (x.top_score() or 0.0),
        reverse=True,
    )

    st.sidebar.metric("Total images", len(manifest.items))
    st.sidebar.metric(
        "Approved",
        sum(1 for x in manifest.items if x.review_status == ReviewStatus.APPROVED),
    )

    cols_per_row = 4
    for row_start in range(0, len(sorted_items), cols_per_row):
        row = sorted_items[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, row):
            with col:
                rec = item.record
                if rec.file_path and rec.file_path.exists():
                    st.image(str(rec.file_path), use_column_width=True)
                else:
                    st.write("(image bytes unavailable)")
                score = item.top_score()
                st.caption(f"score = {score:.3f}" if score is not None else "unscored")
                st.caption(f"`{rec.image_id.hex[:8]}` · {rec.source.value}")
                btn_cols = st.columns(2)
                if btn_cols[0].button("Approve", key=f"a-{rec.image_id}"):
                    item.review_status = ReviewStatus.APPROVED
                if btn_cols[1].button("Reject", key=f"r-{rec.image_id}"):
                    item.review_status = ReviewStatus.REJECTED


if __name__ == "__main__":  # pragma: no cover
    main()
