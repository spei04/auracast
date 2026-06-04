"""
Projects persistence + Drive folder parsing.

A "project" is one Drive folder ↔ one curation manifest. The Streamlit app
keeps a sidebar list of projects and switches between them.

This module:
  - parse_folder_id: extract a folder ID from a pasted URL or accept a bare ID.
  - ProjectsStore: atomic load/save of `manifests/projects.json` (mirrors
    the design of ManifestStore — single-process, write-tempfile + replace).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path

from auracast.schema.models import DriveProject, ProjectsConfig

logger = logging.getLogger(__name__)


_FOLDER_URL_RE = re.compile(r"/folders/([^/?#]+)")


def parse_folder_id(s: str) -> str:
    """Extract a Drive folder ID from a URL, or pass through a bare ID.

    Examples:
        parse_folder_id("https://drive.google.com/drive/folders/ABC123") -> "ABC123"
        parse_folder_id("ABC123") -> "ABC123"
        parse_folder_id("  ABC123  ") -> "ABC123"
    """
    s = (s or "").strip()
    m = _FOLDER_URL_RE.search(s)
    return m.group(1) if m else s


def slugify(name: str) -> str:
    """Filesystem-safe slug from a project name. Used to derive manifest paths."""
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_").lower()
    return s or "project"


class ProjectsStore:
    """Read/write the projects config JSON with atomic writes."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cfg = ProjectsConfig()
        if self.path.exists():
            try:
                self._cfg = ProjectsConfig.model_validate_json(self.path.read_text())
            except Exception as e:  # noqa: BLE001
                logger.warning("projects config %s unreadable (%s); starting empty",
                               self.path, e)

    @property
    def config(self) -> ProjectsConfig:
        return self._cfg

    def all(self) -> list[DriveProject]:
        return list(self._cfg.projects)

    def get(self, name: str) -> DriveProject | None:
        return self._cfg.get(name)

    def active(self) -> DriveProject | None:
        name = self._cfg.active_project_name
        return self._cfg.get(name) if name else None

    def set_active(self, name: str) -> bool:
        with self._lock:
            if self._cfg.get(name) is None:
                return False
            self._cfg.active_project_name = name
            self._flush_locked()
        return True

    def add(self, project: DriveProject) -> None:
        with self._lock:
            self._cfg.add(project)
            self._flush_locked()

    def remove(self, name: str) -> bool:
        with self._lock:
            removed = self._cfg.remove(name)
            if removed:
                self._flush_locked()
            return removed

    def _flush_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(self._cfg.model_dump_json(indent=2))
        os.replace(tmp, self.path)
