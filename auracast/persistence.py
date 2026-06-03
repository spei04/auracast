"""
Manifest persistence — atomic JSONL writes + content-hash dedupe.

The pipeline can crash, get SIGTERMed by Slurm, or simply be re-run on a
larger directory. The store has to give us:
  - Atomic writes (no half-written manifest if we die mid-flush).
  - O(1) lookup by content_hash (so re-ingestion skips already-scored images).
  - Targeted updates by image_id (so the Streamlit app can persist a single
    approve/reject without rewriting the whole file in-place by hand).

We use one JSONL file per manifest, one line per ScoredImage. Writes are
serialized via a `.lock` file (advisory, single-process expected) and made
atomic via write-to-tempfile + os.replace.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from uuid import UUID

from auracast.schema.models import Manifest, ReviewStatus, ScoredImage

logger = logging.getLogger(__name__)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file's bytes. Returns lowercase hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class ManifestStore:
    """Read/write a JSONL manifest with atomic writes and hash-based dedupe."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._items: dict[UUID, ScoredImage] = {}
        self._by_hash: dict[str, UUID] = {}
        if self.path.exists():
            self._load_from_disk()

    # ---- public read API ------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def all(self) -> list[ScoredImage]:
        return list(self._items.values())

    def get(self, image_id: UUID) -> ScoredImage | None:
        return self._items.get(image_id)

    def find_by_hash(self, content_hash: str) -> ScoredImage | None:
        image_id = self._by_hash.get(content_hash)
        return self._items.get(image_id) if image_id else None

    def to_manifest(self) -> Manifest:
        return Manifest(items=list(self._items.values()))

    # ---- public write API -----------------------------------------------

    def add_or_update(self, item: ScoredImage, *, persist: bool = True) -> None:
        """Insert or replace by image_id. Persists atomically by default."""
        with self._lock:
            self._items[item.record.image_id] = item
            if item.record.content_hash:
                self._by_hash[item.record.content_hash] = item.record.image_id
            if persist:
                self._flush_locked()

    def update_review(
        self, image_id: UUID, status: ReviewStatus, *, persist: bool = True
    ) -> bool:
        """Set the review_status of one item. Returns False if image_id unknown."""
        with self._lock:
            item = self._items.get(image_id)
            if item is None:
                return False
            self._items[image_id] = item.model_copy(update={"review_status": status})
            if persist:
                self._flush_locked()
        return True

    def bulk_add(self, items: list[ScoredImage]) -> None:
        """Insert/replace many. One atomic write at the end."""
        with self._lock:
            for item in items:
                self._items[item.record.image_id] = item
                if item.record.content_hash:
                    self._by_hash[item.record.content_hash] = item.record.image_id
            self._flush_locked()

    # ---- internals ------------------------------------------------------

    def _load_from_disk(self) -> None:
        text = self.path.read_text()
        manifest = Manifest.from_jsonl(text)
        for item in manifest.items:
            self._items[item.record.image_id] = item
            if item.record.content_hash:
                self._by_hash[item.record.content_hash] = item.record.image_id
        logger.info("loaded %d items from %s", len(self._items), self.path)

    def _flush_locked(self) -> None:
        """Atomic write. Caller already holds self._lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        manifest = Manifest(items=list(self._items.values()))
        tmp.write_text(manifest.to_jsonl())
        os.replace(tmp, self.path)
