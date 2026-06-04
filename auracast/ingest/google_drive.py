"""
Google Drive ingest + lightweight folder/file management helpers.

Same IngestSource shape as LocalDirectoryIngest / GooglePhotosIngest — the
pipeline doesn't know or care which one it's holding. Lists image files in
Drive (optionally restricted to a folder), downloads bytes via the Drive
v3 `files.get_media` endpoint, writes a content_hash, yields ImageRecord
rows with source=GOOGLE_DRIVE.

Also exposes module-level helpers used by the Streamlit UI:
  - list_my_folders(credentials, ...) — for the folder picker.
  - trash_files(credentials, file_ids) — for the Finalize button.
Both require the full `drive` scope (see auracast.auth.google_oauth).

To use as ingest:
  - Put images in any Google Drive folder.
  - Get its folder ID from the URL (the part after /folders/), or use the
    UI folder picker.
"""

from __future__ import annotations

import io
import logging
import mimetypes
from pathlib import Path
from typing import AsyncIterator, Iterable

from PIL import Image

from auracast.auth.google_oauth import build_drive_service
from auracast.ingest.base import IngestSource
from auracast.persistence import sha256_file
from auracast.schema.models import ImageRecord, IngestSourceKind

logger = logging.getLogger(__name__)


# Drive v3 caps pageSize at 1000, but realistic numbers are smaller.
MAX_PAGE_SIZE = 100

# Files.list query: image MIME types only. Folder filter is appended at runtime.
_BASE_QUERY = "mimeType contains 'image/' and trashed = false"


class GoogleDriveIngest(IngestSource):
    """Lists image files from Google Drive and yields ImageRecord per item.

    Args:
        credentials: a `google.oauth2.credentials.Credentials`.
        download_dir: local cache for downloaded image bytes.
        folder_id: restrict to this folder (its direct children). None = walk
            all images the user owns / has access to.
        page_size: items per API call (clamped to MAX_PAGE_SIZE).
        max_items: hard cap; None = no limit.
        recursive: when folder_id is given, also walk subfolders. Default False
            because Drive's recursive walk needs separate queries per subfolder
            and the latency adds up fast — set True if you've organized photos
            into nested folders.
    """

    def __init__(
        self,
        credentials,
        *,
        download_dir: Path | str,
        folder_id: str | None = None,
        page_size: int = 50,
        max_items: int | None = None,
        recursive: bool = False,
    ):
        self.credentials = credentials
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.folder_id = folder_id
        self.page_size = min(page_size, MAX_PAGE_SIZE)
        self.max_items = max_items
        self.recursive = recursive
        self._service = None

    def _ensure_service(self):
        if self._service is None:
            self._service = build_drive_service(self.credentials)

    def _query_for_folder(self, folder_id: str | None) -> str:
        if folder_id is None:
            return _BASE_QUERY
        return f"'{folder_id}' in parents and {_BASE_QUERY}"

    def _list_subfolders(self, parent_id: str) -> Iterable[str]:
        self._ensure_service()
        page_token = None
        q = f"'{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        while True:
            kwargs = {
                "q": q,
                "pageSize": self.page_size,
                "fields": "nextPageToken, files(id)",
            }
            if page_token:
                kwargs["pageToken"] = page_token
            resp = self._service.files().list(**kwargs).execute()
            for f in resp.get("files", []):
                yield f["id"]
            page_token = resp.get("nextPageToken")
            if not page_token:
                return

    def _list_files_in_folder(self, folder_id: str | None) -> Iterable[dict]:
        """Yield image-file metadata dicts for one folder (no recursion)."""
        self._ensure_service()
        page_token = None
        q = self._query_for_folder(folder_id)
        while True:
            kwargs = {
                "q": q,
                "pageSize": self.page_size,
                "fields": "nextPageToken, files(id, name, mimeType, size)",
            }
            if page_token:
                kwargs["pageToken"] = page_token
            resp = self._service.files().list(**kwargs).execute()
            for f in resp.get("files", []) or []:
                yield f
            page_token = resp.get("nextPageToken")
            if not page_token:
                return

    def _list_files(self) -> Iterable[dict]:
        """Top-level walk: one folder if non-recursive, BFS otherwise."""
        emitted = 0
        queue: list[str | None] = [self.folder_id]
        visited: set[str | None] = set()
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            for f in self._list_files_in_folder(cur):
                yield f
                emitted += 1
                if self.max_items is not None and emitted >= self.max_items:
                    return
            if self.recursive and cur is not None:
                for sub in self._list_subfolders(cur):
                    queue.append(sub)

    def _download_file(self, drive_file: dict) -> Path | None:
        """Download an image file to download_dir. Returns local path or None."""
        from googleapiclient.http import MediaIoBaseDownload

        file_id = drive_file["id"]
        name = drive_file.get("name", file_id)
        mime = drive_file.get("mimeType", "image/jpeg")
        # Sanitize the name a bit so it's filesystem-safe; prefix with id for uniqueness.
        safe_name = name.replace("/", "_").replace("\\", "_")
        suffix = Path(safe_name).suffix or (mimetypes.guess_extension(mime) or ".jpg")
        stem = Path(safe_name).stem or "image"
        dest = self.download_dir / f"{file_id}_{stem}{suffix}"
        if dest.exists():
            return dest

        try:
            self._ensure_service()
            request = self._service.files().get_media(fileId=file_id)
            with io.FileIO(dest, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
        except Exception as e:  # noqa: BLE001
            logger.warning("Drive download failed for %s (%s): %s", file_id, name, e)
            if dest.exists():
                dest.unlink()
            return None
        return dest

    def _build_record(self, drive_file: dict, local_path: Path) -> ImageRecord | None:
        try:
            with Image.open(local_path) as im:
                width, height = im.size
        except Exception as e:  # noqa: BLE001
            logger.warning("downloaded file %s not decodable: %s", local_path, e)
            return None

        return ImageRecord(
            source=IngestSourceKind.GOOGLE_DRIVE,
            file_path=local_path,
            source_ref=drive_file.get("id"),
            content_hash=sha256_file(local_path),
            width=width,
            height=height,
            file_size_bytes=local_path.stat().st_size,
            mime_type=drive_file.get("mimeType"),
        )

    # ---- Public iteration -----------------------------------------------

    async def __aiter__(self) -> AsyncIterator[ImageRecord]:
        for f in self._list_files():
            dest = self._download_file(f)
            if dest is None:
                continue
            rec = self._build_record(f, dest)
            if rec is not None:
                yield rec

    def collect(self) -> list[ImageRecord]:
        out: list[ImageRecord] = []
        for f in self._list_files():
            dest = self._download_file(f)
            if dest is None:
                continue
            rec = self._build_record(f, dest)
            if rec is not None:
                out.append(rec)
        return out


# -------- Module-level helpers (for the Streamlit UI) -------------------


def list_my_folders(
    credentials, *, page_size: int = 100, max_items: int = 200,
) -> list[dict]:
    """Return [{'id', 'name', 'parents'}] for folders the user *owns*.

    Filters out folders shared with the user from elsewhere via the
    `'me' in owners` predicate. Capped at `max_items` to keep the picker
    UI responsive. Use for the folder dropdown in Streamlit.
    """
    service = build_drive_service(credentials)
    q = (
        "mimeType = 'application/vnd.google-apps.folder' "
        "and trashed = false "
        "and 'me' in owners"
    )
    fields = "nextPageToken, files(id, name, parents)"
    out: list[dict] = []
    page_token = None
    while True:
        kwargs = {"q": q, "pageSize": min(page_size, MAX_PAGE_SIZE), "fields": fields,
                  "orderBy": "name"}
        if page_token:
            kwargs["pageToken"] = page_token
        resp = service.files().list(**kwargs).execute()
        for f in resp.get("files", []) or []:
            out.append(f)
            if len(out) >= max_items:
                return out
        page_token = resp.get("nextPageToken")
        if not page_token:
            return out


def trash_files(credentials, file_ids: list[str]) -> dict[str, str | None]:
    """Move the given Drive file IDs to Trash. Soft delete — recoverable in
    Drive UI for ~30 days.

    Returns {file_id: error_message_or_None}. None means success.
    """
    service = build_drive_service(credentials)
    results: dict[str, str | None] = {}
    for fid in file_ids:
        try:
            service.files().update(fileId=fid, body={"trashed": True}).execute()
            results[fid] = None
        except Exception as e:  # noqa: BLE001
            logger.warning("trash failed for %s: %s", fid, e)
            results[fid] = f"{type(e).__name__}: {e}"
    return results
