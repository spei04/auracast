"""
Streamlit preview UI.

Loads a JSONL manifest (output of the ingest+score pipeline), shows a grid of
images sorted by aesthetic score, lets the operator approve or reject each.
Reviews are persisted to the manifest via ManifestStore — closing the tab and
re-opening preserves state.

Includes a "Sync from Drive" sidebar control: enter a Drive folder ID, click
Sync, and the app pulls + scores any new images in-process. Dedupe means
already-known images aren't re-fetched or re-scored.

Run:
    streamlit run auracast/app/streamlit_app.py -- --manifest manifests/latest.jsonl
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None  # noqa: N816

from auracast.persistence import ManifestStore
from auracast.schema.models import ProcessingStatus, ReviewStatus

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/latest.jsonl"),
        help="Path to a Manifest JSONL written by the pipeline.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/gdrive_cache"),
        help="Cache directory used by the in-app Drive sync.",
    )
    return parser.parse_args()


def _sync_from_drive(
    store: ManifestStore,
    folder_id: str,
    max_items: int,
    download_dir: Path,
) -> tuple[int, int]:
    """Pull from Drive, dedupe by content_hash, score new images.

    Returns (new_count, skipped_count).
    """
    from auracast.auth.google_oauth import load_credentials
    from auracast.ingest.google_drive import GoogleDriveIngest
    from auracast.scripts.pipeline import _score_pass

    creds = load_credentials(interactive=False)
    ingest = GoogleDriveIngest(
        credentials=creds,
        download_dir=download_dir,
        folder_id=folder_id,
        max_items=max_items,
    )
    raw_records = ingest.collect()

    # Dedupe against the existing manifest by content_hash.
    new_records = []
    skipped = 0
    for rec in raw_records:
        if rec.content_hash and store.find_by_hash(rec.content_hash) is not None:
            skipped += 1
            continue
        new_records.append(rec)

    if new_records:
        _score_pass(store, new_records, batch_size=16)
        store.bulk_add([])  # flush

    return len(new_records), skipped


def _render_sync_sidebar(store: ManifestStore, download_dir: Path) -> None:
    """Render the Drive sync controls. Re-runs Streamlit when work happens."""
    st.sidebar.subheader("Sync from Drive")

    # Saved folder ID persists across reruns via session_state; seed from
    # env var so the user can default it without typing each time.
    default_folder = st.session_state.get(
        "drive_folder_id",
        os.environ.get("AURACAST_DRIVE_FOLDER", ""),
    )
    folder_id = st.sidebar.text_input(
        "Drive folder ID",
        value=default_folder,
        help="The part after /folders/ in your Drive URL.",
        key="drive_folder_input",
    )
    max_items = st.sidebar.number_input(
        "Max items per sync", min_value=1, max_value=500, value=50, step=10,
    )

    if st.sidebar.button("🔄 Sync now", type="primary", use_container_width=True):
        if not folder_id.strip():
            st.sidebar.error("Enter a folder ID first.")
        else:
            st.session_state["drive_folder_id"] = folder_id.strip()
            with st.spinner("Pulling + scoring from Drive..."):
                try:
                    new_count, skipped = _sync_from_drive(
                        store, folder_id.strip(), int(max_items), download_dir,
                    )
                except Exception as e:  # noqa: BLE001
                    st.sidebar.error(f"Sync failed: {e}")
                else:
                    st.sidebar.success(
                        f"Synced {new_count + skipped} item(s) "
                        f"({new_count} new, {skipped} already known)."
                    )
                    st.rerun()


def main() -> None:  # pragma: no cover — Streamlit entry point
    if st is None:
        raise RuntimeError("Streamlit is not installed. `pip install streamlit`.")
    args = _parse_args()

    st.set_page_config(page_title="AuraCast", layout="wide")
    st.title("AuraCast — Curation Preview")

    # ManifestStore handles a missing file by starting empty — important so the
    # Sync button works on a fresh install with no manifest yet.
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    store = ManifestStore(args.manifest)

    # ---- Sidebar: sync + filters + counters ----------------------------
    _render_sync_sidebar(store, args.download_dir)
    st.sidebar.divider()

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

    if len(store) == 0:
        st.info("No images yet. Use the sidebar Sync from Drive to get started.")
        return

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
