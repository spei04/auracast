"""
Local-directory ingest. Walks a directory, yields one ImageRecord per image.

Used during development to exercise the full pipeline (ingest -> score ->
embed -> manifest -> Streamlit) without needing Google API credentials.

In production this is replaced by GoogleAPIIngest (forthcoming) which yields
the same ImageRecord type — no downstream changes required.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import AsyncIterator, Iterable

import aiofiles  # noqa: F401  — pulled in so the dep is loaded; used in later async work
from PIL import Image

from auracast.ingest.base import IngestSource
from auracast.persistence import sha256_file
from auracast.schema.models import ImageRecord, IngestSourceKind

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")


class LocalDirectoryIngest(IngestSource):
    """Walks `root` and yields ImageRecord per image file."""

    def __init__(
        self,
        root: Path | str,
        suffixes: Iterable[str] = DEFAULT_IMAGE_SUFFIXES,
        recursive: bool = True,
        compute_hash: bool = True,
    ):
        self.root = Path(root)
        self.suffixes = tuple(s.lower() for s in suffixes)
        self.recursive = recursive
        self.compute_hash = compute_hash
        if not self.root.exists():
            raise FileNotFoundError(f"ingest root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"ingest root is not a directory: {self.root}")

    def iter_paths(self) -> Iterable[Path]:
        glob = "**/*" if self.recursive else "*"
        for p in sorted(self.root.glob(glob)):
            if p.is_file() and p.suffix.lower() in self.suffixes:
                yield p

    def _build_record(self, path: Path) -> ImageRecord | None:
        try:
            with Image.open(path) as im:
                width, height = im.size
        except Exception as e:  # noqa: BLE001 — third-party formats raise various things
            logger.warning("skipping %s: cannot decode (%s)", path, e)
            return None
        mime_type, _ = mimetypes.guess_type(path.name)
        content_hash = sha256_file(path) if self.compute_hash else None
        return ImageRecord(
            source=IngestSourceKind.LOCAL_DIRECTORY,
            file_path=path,
            content_hash=content_hash,
            width=width,
            height=height,
            file_size_bytes=path.stat().st_size,
            mime_type=mime_type,
        )

    async def __aiter__(self) -> AsyncIterator[ImageRecord]:
        for p in self.iter_paths():
            rec = self._build_record(p)
            if rec is not None:
                yield rec

    # Sync convenience for non-async callers (e.g. scripts, tests).
    def collect(self) -> list[ImageRecord]:
        return [r for r in (self._build_record(p) for p in self.iter_paths()) if r is not None]
