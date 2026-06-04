"""
Streamlit preview UI.

Multi-project curation. Each project = one Drive folder + one manifest.
Switch between projects via the sidebar. Each project tracks its own
approve/reject state. A "Finalize" button per project moves rejected
images to Drive Trash so the folder ends up as exactly the approved set.

Run:
    streamlit run auracast/app/streamlit_app.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None  # noqa: N816

from auracast.engine.registry import (
    SCORER_DESCRIPTIONS,
    SCORER_LABELS,
    rescore_store,
)
from auracast.persistence import ManifestStore
from auracast.projects import ProjectsStore, parse_folder_id, slugify
from auracast.schema.models import (
    DriveProject,
    ProcessingStatus,
    ReviewStatus,
    ScorerModel,
    scorer_takes_text,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--projects-config",
        type=Path,
        default=Path("manifests/projects.json"),
        help="Path to the projects config JSON.",
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("manifests"),
        help="Directory where per-project manifests are stored.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/gdrive_cache"),
        help="Cache directory used by Drive sync.",
    )
    return parser.parse_args()


# -------- Sidebar sections ---------------------------------------------


def _render_project_picker(projects: ProjectsStore, manifests_dir: Path) -> DriveProject | None:
    """Top-of-sidebar project switcher + new-project form. Returns the active project."""
    st.sidebar.header("📁 Projects")
    all_projects = projects.all()

    if not all_projects:
        st.sidebar.info("No projects yet. Create one below to get started.")
    else:
        names = [p.name for p in all_projects]
        active = projects.active()
        idx = names.index(active.name) if active else 0
        selected = st.sidebar.selectbox(
            "Active project",
            options=names,
            index=idx,
            key="active_project_select",
        )
        if active is None or selected != active.name:
            projects.set_active(selected)

    with st.sidebar.expander("➕ New project", expanded=not all_projects):
        _render_new_project_form(projects, manifests_dir)

    return projects.active()


def _render_new_project_form(projects: ProjectsStore, manifests_dir: Path) -> None:
    """New-project form: pick from Drive OR paste URL/ID. No nested expanders.

    Streamlit forbids st.expander inside st.expander, so the picker is laid
    out inline here. Loaded folders are cached in session_state so re-runs
    don't re-fetch.
    """
    name = st.text_input("Project name", placeholder="e.g. Spring Aesthetic", key="np_name")

    # ---- Folder picker (primary path) ----------------------------------
    st.markdown("**Pick a folder from your Drive:**")
    cols = st.columns([3, 1])
    with cols[1]:
        if st.button("🔄 Refresh", key="np_load_folders", help="Fetch your Drive folder list"):
            try:
                from auracast.auth.google_oauth import load_credentials
                from auracast.ingest.google_drive import list_my_folders
                creds = load_credentials(interactive=False)
                folders = list_my_folders(creds, max_items=500)
                st.session_state["available_folders"] = folders
                if not folders:
                    st.warning("No folders found in your Drive.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't list folders: {e}")

    folders = st.session_state.get("available_folders", [])
    picked_folder_id = ""
    with cols[0]:
        if not folders:
            st.caption("Click **Refresh** to load your Drive folders.")
        else:
            options = ["— pick a folder —"] + [f["name"] for f in folders]
            choice = st.selectbox(
                f"Your folders ({len(folders)})",
                options=options,
                key="np_picker",
                label_visibility="collapsed",
            )
            if choice != "— pick a folder —":
                idx = options.index(choice) - 1
                picked_folder_id = folders[idx]["id"]
                st.caption(f"Selected ID: `{picked_folder_id[:24]}…`")

    # ---- Manual entry (fallback) ---------------------------------------
    st.markdown("**Or paste a Drive URL / ID:**")
    folder_input = st.text_input(
        "URL or ID",
        placeholder="https://drive.google.com/drive/folders/...",
        key="np_folder",
        label_visibility="collapsed",
    )

    # The picker takes priority over the text box.
    folder_id = picked_folder_id or parse_folder_id(folder_input)

    if st.button("Create project", type="primary", key="np_create"):
        if not name.strip():
            st.error("Project name is required.")
            return
        if not folder_id:
            st.error("Pick a folder or paste a URL/ID.")
            return
        if projects.get(name.strip()):
            st.error(f"A project named '{name.strip()}' already exists.")
            return
        manifest_path = manifests_dir / f"{slugify(name)}.jsonl"
        project = DriveProject(
            name=name.strip(),
            folder_id=folder_id,
            manifest_path=manifest_path,
        )
        projects.add(project)
        projects.set_active(project.name)
        st.success(f"Created '{project.name}'. Use Sync to populate it.")
        st.rerun()


def _render_sync_sidebar(project: DriveProject, store: ManifestStore, download_dir: Path) -> None:
    st.sidebar.subheader("⚙️ Sync from Drive")
    max_items = st.sidebar.number_input(
        "Max items per sync", min_value=1, max_value=500, value=50, step=10,
    )
    if st.sidebar.button("🔄 Sync now", type="primary", width="stretch"):
        with st.spinner("Pulling + scoring from Drive..."):
            try:
                new_count, skipped = _sync_from_drive(
                    store, project.folder_id, int(max_items), download_dir,
                )
            except Exception as e:  # noqa: BLE001
                st.sidebar.error(f"Sync failed: {e}")
            else:
                st.sidebar.success(
                    f"Synced {new_count + skipped} item(s) "
                    f"({new_count} new, {skipped} already known)."
                )
                st.rerun()


def _render_aesthetic_section(
    project: DriveProject, store: ManifestStore, projects: ProjectsStore,
) -> None:
    """Top-of-page scorer-model picker + (conditional) prompt + score button.

    Model dropdown swaps the scoring backend; prompt inputs are shown only
    for backends that consume text.
    """
    with st.expander("🎯 Aesthetic goal — what should we optimize for?", expanded=len(store) == 0):
        # ---- Model selection -----------------------------------------
        model_options = list(SCORER_LABELS.keys())
        try:
            default_idx = model_options.index(project.scorer_model)
        except ValueError:
            default_idx = 0

        selected_model: ScorerModel = st.selectbox(
            "Scoring model",
            options=model_options,
            index=default_idx,
            format_func=lambda m: SCORER_LABELS[m],
            key=f"scorer_model_{project.name}",
        )
        st.caption(SCORER_DESCRIPTIONS[selected_model])

        takes_text = scorer_takes_text(selected_model)
        pos = project.positive_prompt
        neg = project.negative_prompt

        if takes_text:
            pos = st.text_area(
                "Positive prompt (what you're looking for)",
                value=project.positive_prompt,
                height=80,
                key=f"pos_{project.name}",
                help="E.g. 'warm authentic smile, direct eye contact, golden-hour lighting'",
            )
            with st.expander("Advanced: negative prompt"):
                neg = st.text_area(
                    "Negative prompt (what to penalize)",
                    value=project.negative_prompt,
                    height=60,
                    key=f"neg_{project.name}",
                )
        else:
            st.info(
                "This model is a trained predictor — it doesn't take a text prompt. "
                "It returns its own learned opinion of aesthetic quality."
            )

        n_to_score = len(store)
        button_label = (
            f"🎯 Score all {n_to_score} image(s)"
            if n_to_score else "🎯 Save scorer choice (no images yet)"
        )
        if st.button(button_label, type="primary", key=f"score_{project.name}"):
            # Persist the picked model + prompts so this project remembers.
            projects.update_scorer_model(project.name, selected_model)
            projects.update_prompts(
                project.name,
                pos.strip() if takes_text else project.positive_prompt,
                neg.strip() if takes_text else project.negative_prompt,
            )
            if n_to_score == 0:
                st.success("Saved. Sync from Drive to start scoring.")
                st.rerun()
                return
            with st.spinner(f"Scoring {n_to_score} image(s) with {SCORER_LABELS[selected_model]}..."):
                try:
                    n_scored = rescore_store(
                        store,
                        model=selected_model,
                        positive_prompt=pos.strip() if takes_text else "",
                        negative_prompt=neg.strip() if takes_text else "",
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Scoring failed: {e}")
                else:
                    st.success(f"Re-scored {n_scored} image(s) with {SCORER_LABELS[selected_model]}.")
                    st.rerun()


def _render_finalize_section(project: DriveProject, store: ManifestStore) -> None:
    """Bottom-of-page button: trash all REJECTED items on Drive."""
    rejected = [x for x in store.all() if x.review_status == ReviewStatus.REJECTED]
    st.divider()
    st.subheader("🗑 Finalize project")
    st.write(
        f"**{len(rejected)}** rejected image(s) will be moved to Drive Trash. "
        f"They stay recoverable in Drive's Trash for ~30 days."
    )
    if not rejected:
        return
    confirm = st.checkbox(
        f"I understand — move {len(rejected)} file(s) to Drive Trash",
        key=f"finalize_confirm_{project.name}",
    )
    if st.button(
        "Finalize: trash rejected on Drive",
        disabled=not confirm,
        type="primary",
        key=f"finalize_btn_{project.name}",
    ):
        with st.spinner("Trashing rejected files on Drive..."):
            try:
                ok, failed = _trash_rejected(store, rejected)
            except Exception as e:  # noqa: BLE001
                st.error(f"Finalize failed: {e}")
            else:
                if failed:
                    st.warning(f"Trashed {ok}; {len(failed)} failed: {failed}")
                else:
                    st.success(f"Trashed {ok} file(s) on Drive.")
                st.rerun()


# -------- Backend helpers ----------------------------------------------


def _sync_from_drive(
    store: ManifestStore,
    folder_id: str,
    max_items: int,
    download_dir: Path,
) -> tuple[int, int]:
    """Pull from Drive, dedupe by content_hash, score new images.
    Returns (new_count, skipped_count)."""
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

    new_records = []
    skipped = 0
    for rec in raw_records:
        if rec.content_hash and store.find_by_hash(rec.content_hash) is not None:
            skipped += 1
            continue
        new_records.append(rec)

    if new_records:
        _score_pass(store, new_records, batch_size=16)
        store.bulk_add([])

    return len(new_records), skipped


def _trash_rejected(store: ManifestStore, rejected_items) -> tuple[int, dict[str, str]]:
    """Move rejected images' Drive originals to Trash. Returns (ok, failed_map)."""
    from auracast.auth.google_oauth import load_credentials
    from auracast.ingest.google_drive import trash_files

    file_ids = [it.record.source_ref for it in rejected_items if it.record.source_ref]
    if not file_ids:
        return 0, {}

    creds = load_credentials(interactive=False)
    results = trash_files(creds, file_ids)
    ok = sum(1 for v in results.values() if v is None)
    failed = {k: v for k, v in results.items() if v is not None}
    return ok, failed


# -------- Main page ----------------------------------------------------


def main() -> None:  # pragma: no cover — Streamlit entry point
    if st is None:
        raise RuntimeError("Streamlit is not installed. `pip install streamlit`.")
    args = _parse_args()

    st.set_page_config(page_title="AuraCast", layout="wide")
    st.title("AuraCast — Curation Preview")

    args.projects_config.parent.mkdir(parents=True, exist_ok=True)
    args.manifests_dir.mkdir(parents=True, exist_ok=True)
    projects = ProjectsStore(args.projects_config)
    project = _render_project_picker(projects, args.manifests_dir)

    if project is None:
        st.info("Create your first project from the sidebar.")
        return

    st.subheader(f"Project: {project.name}")
    st.caption(
        f"Drive folder `{project.folder_id}` · manifest `{project.manifest_path}`"
    )

    project.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    store = ManifestStore(project.manifest_path)

    # Top-of-page aesthetic prompt editor + Score button.
    _render_aesthetic_section(project, store, projects)

    # Sidebar sync, filters, stats
    st.sidebar.divider()
    _render_sync_sidebar(project, store, args.download_dir)
    st.sidebar.divider()

    st.sidebar.metric("Total images", len(store))
    approved = sum(1 for x in store.all() if x.review_status == ReviewStatus.APPROVED)
    rejected_count = sum(1 for x in store.all() if x.review_status == ReviewStatus.REJECTED)
    pending = sum(1 for x in store.all() if x.review_status == ReviewStatus.PENDING)
    st.sidebar.metric("Approved", approved)
    st.sidebar.metric("Rejected", rejected_count)
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
        st.info("No images yet. Use **Sync from Drive** in the sidebar to populate this project.")
        return

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
                    st.image(str(rec.file_path), width="stretch")
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

                btn_cols = st.columns(3)
                if btn_cols[0].button("Approve", key=f"a-{rec.image_id}"):
                    store.update_review(rec.image_id, ReviewStatus.APPROVED)
                    st.rerun()
                if btn_cols[1].button("Reject", key=f"r-{rec.image_id}"):
                    store.update_review(rec.image_id, ReviewStatus.REJECTED)
                    st.rerun()
                # Undo: back to PENDING. Only enabled if the item isn't already pending.
                is_pending = item.review_status == ReviewStatus.PENDING
                if btn_cols[2].button(
                    "↺", key=f"u-{rec.image_id}",
                    help="Reset to pending",
                    disabled=is_pending,
                ):
                    store.update_review(rec.image_id, ReviewStatus.PENDING)
                    st.rerun()

    _render_finalize_section(project, store)


if __name__ == "__main__":  # pragma: no cover
    main()
