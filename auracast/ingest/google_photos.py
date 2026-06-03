"""
Google Photos Library API ingest.

Paginates through mediaItems, downloads bytes via the per-item baseUrl,
writes them to a local cache directory, and yields ImageRecord rows with
source=GOOGLE_PHOTOS — same downstream interface as LocalDirectoryIngest.

Authentication is delegated to `auracast.auth.google_oauth.load_credentials`,
which gives us a `Credentials` object that the googleapiclient honors.

The Photos Library API is rate-limited and as of 2024 has restricted access
for newly-registered third-party apps. If your GCP project doesn't have
Photos API access, prefer `auracast.ingest.google_drive` (forthcoming).
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import AsyncIterator, Iterable

import httpx
from PIL import Image

from auracast.auth.google_oauth import build_photos_service
from auracast.ingest.base import IngestSource
from auracast.persistence import sha256_file
from auracast.schema.models import ImageRecord, IngestSourceKind

logger = logging.getLogger(__name__)


# Photos Library API caps page_size at 100 for mediaItems.list and 25 for searches.
MAX_PAGE_SIZE = 100


class GooglePhotosIngest(IngestSource):
    """Walks Google Photos and yields ImageRecord per item.

    Args:
        credentials: a `google.oauth2.credentials.Credentials` (use
            `auracast.auth.google_oauth.load_credentials(...)` to obtain).
        download_dir: local cache directory for downloaded bytes.
        page_size: items per API call (clamped to MAX_PAGE_SIZE).
        max_items: hard cap; None = no limit.
        album_id: if set, list only items from this album (uses
            `mediaItems.search` instead of `mediaItems.list`).
    """

    def __init__(
        self,
        credentials,
        *,
        download_dir: Path | str,
        page_size: int = 50,
        max_items: int | None = None,
        album_id: str | None = None,
    ):
        self.credentials = credentials
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.page_size = min(page_size, MAX_PAGE_SIZE)
        self.max_items = max_items
        self.album_id = album_id
        self._service = None  # built on first use

    def _ensure_service(self):
        if self._service is None:
            self._service = build_photos_service(self.credentials)

    def _list_media_items(self) -> Iterable[dict]:
        """Paginate through mediaItems, optionally filtered by album_id."""
        self._ensure_service()
        page_token = None
        emitted = 0
        while True:
            if self.album_id:
                body = {"pageSize": self.page_size, "albumId": self.album_id}
                if page_token:
                    body["pageToken"] = page_token
                resp = self._service.mediaItems().search(body=body).execute()
            else:
                kwargs = {"pageSize": self.page_size}
                if page_token:
                    kwargs["pageToken"] = page_token
                resp = self._service.mediaItems().list(**kwargs).execute()

            items = resp.get("mediaItems", []) or []
            for it in items:
                # Filter to images only — the API returns videos too.
                mime = it.get("mimeType", "")
                if not mime.startswith("image/"):
                    continue
                yield it
                emitted += 1
                if self.max_items is not None and emitted >= self.max_items:
                    return

            page_token = resp.get("nextPageToken")
            if not page_token:
                return

    def _download_bytes(self, item: dict) -> Path | None:
        """Download an item's bytes to `download_dir`. Returns the local path
        or None if the download failed."""
        base_url = item.get("baseUrl")
        if not base_url:
            logger.warning("item %s missing baseUrl; skipping", item.get("id"))
            return None
        # The Photos API requires appending '=d' to baseUrl to get the full
        # original-resolution download URL.
        url = f"{base_url}=d"

        media_id = item["id"]
        suffix = mimetypes.guess_extension(item.get("mimeType", "image/jpeg")) or ".jpg"
        dest = self.download_dir / f"{media_id}{suffix}"
        if dest.exists():
            return dest

        try:
            with httpx.stream("GET", url, timeout=30.0, follow_redirects=True) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
        except Exception as e:  # noqa: BLE001 — network errors are diverse
            logger.warning("download failed for %s: %s", media_id, e)
            if dest.exists():
                dest.unlink()
            return None
        return dest

    def _build_record(self, item: dict, local_path: Path) -> ImageRecord | None:
        """Combine the API metadata + local bytes into an ImageRecord."""
        try:
            with Image.open(local_path) as im:
                width, height = im.size
        except Exception as e:  # noqa: BLE001
            logger.warning("downloaded file %s not decodable: %s", local_path, e)
            return None

        return ImageRecord(
            source=IngestSourceKind.GOOGLE_PHOTOS,
            file_path=local_path,
            source_ref=item.get("id"),
            content_hash=sha256_file(local_path),
            width=width,
            height=height,
            file_size_bytes=local_path.stat().st_size,
            mime_type=item.get("mimeType"),
        )

    # ---- Public iteration -----------------------------------------------

    async def __aiter__(self) -> AsyncIterator[ImageRecord]:
        for item in self._list_media_items():
            dest = self._download_bytes(item)
            if dest is None:
                continue
            rec = self._build_record(item, dest)
            if rec is not None:
                yield rec

    def collect(self) -> list[ImageRecord]:
        """Sync convenience for non-async callers (pipeline, tests)."""
        out: list[ImageRecord] = []
        for item in self._list_media_items():
            dest = self._download_bytes(item)
            if dest is None:
                continue
            rec = self._build_record(item, dest)
            if rec is not None:
                out.append(rec)
        return out
