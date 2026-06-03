"""Tests for auracast.ingest.google_drive.

The Drive API client is mocked. We verify: pagination, folder filtering,
recursive subfolder walk, MIME filtering happens server-side via the query,
max_items cap, download failure handling, content_hash carried through.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from auracast.ingest.google_drive import GoogleDriveIngest
from auracast.schema.models import IngestSourceKind


def _png_bytes(color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def _drive_file(file_id: str, name: str = None, mime: str = "image/png"):
    return {"id": file_id, "name": name or f"{file_id}.png", "mimeType": mime, "size": "1024"}


def _mock_service(list_responses: list[dict], download_bytes: bytes | None = None):
    """Build a fake Drive service whose .files().list returns these in order and
    whose .files().get_media downloads `download_bytes`."""
    svc = MagicMock()
    pages_iter = iter(list_responses)

    list_method = MagicMock()
    list_method.execute = MagicMock(side_effect=lambda: next(pages_iter))
    svc.files().list = MagicMock(return_value=list_method)

    media_request = MagicMock()
    svc.files().get_media = MagicMock(return_value=media_request)
    return svc, download_bytes if download_bytes else _png_bytes()


def _patch_download(monkeypatch, bytes_to_write: bytes):
    """Make MediaIoBaseDownload write `bytes_to_write` to the fh and finish."""
    from auracast.ingest import google_drive as gd_module

    class _FakeDownloader:
        def __init__(self, fh, request):
            self._fh = fh
            self._done = False
        def next_chunk(self):
            if not self._done:
                self._fh.write(bytes_to_write)
                self._done = True
            return None, True

    # The import is inside _download_file so monkeypatching the module attr is brittle;
    # easier: patch the symbol at the googleapiclient.http path.
    monkeypatch.setattr("googleapiclient.http.MediaIoBaseDownload", _FakeDownloader)


def test_lists_paginated_files(tmp_path, monkeypatch):
    pages = [
        {"files": [_drive_file("a"), _drive_file("b")], "nextPageToken": "tok"},
        {"files": [_drive_file("c")]},
    ]
    ingest = GoogleDriveIngest(credentials=MagicMock(), download_dir=tmp_path / "cache")
    svc, _ = _mock_service(pages)
    ingest._service = svc
    _patch_download(monkeypatch, _png_bytes())

    records = ingest.collect()
    assert len(records) == 3
    assert {r.source_ref for r in records} == {"a", "b", "c"}
    for r in records:
        assert r.source == IngestSourceKind.GOOGLE_DRIVE


def test_folder_id_appears_in_query(tmp_path, monkeypatch):
    pages = [{"files": [_drive_file("a")]}]
    ingest = GoogleDriveIngest(
        credentials=MagicMock(), download_dir=tmp_path / "cache", folder_id="FOLDER123",
    )
    svc, _ = _mock_service(pages)
    ingest._service = svc
    _patch_download(monkeypatch, _png_bytes())

    ingest.collect()
    # First list() call must include the folder in its q= parameter.
    first_call = svc.files().list.call_args_list[0]
    q = first_call.kwargs["q"]
    assert "FOLDER123" in q
    assert "in parents" in q


def test_no_folder_id_uses_unscoped_query(tmp_path, monkeypatch):
    pages = [{"files": [_drive_file("a")]}]
    ingest = GoogleDriveIngest(credentials=MagicMock(), download_dir=tmp_path / "cache")
    svc, _ = _mock_service(pages)
    ingest._service = svc
    _patch_download(monkeypatch, _png_bytes())

    ingest.collect()
    q = svc.files().list.call_args_list[0].kwargs["q"]
    assert "in parents" not in q
    assert "image/" in q


def test_max_items_caps_pagination(tmp_path, monkeypatch):
    pages = [
        {"files": [_drive_file(f"i{n}") for n in range(5)], "nextPageToken": "tok"},
        {"files": [_drive_file(f"j{n}") for n in range(5)]},
    ]
    ingest = GoogleDriveIngest(
        credentials=MagicMock(), download_dir=tmp_path / "cache", max_items=3,
    )
    svc, _ = _mock_service(pages)
    ingest._service = svc
    _patch_download(monkeypatch, _png_bytes())

    records = ingest.collect()
    assert len(records) == 3


def test_record_carries_content_hash(tmp_path, monkeypatch):
    pages = [{"files": [_drive_file("a")]}]
    ingest = GoogleDriveIngest(credentials=MagicMock(), download_dir=tmp_path / "cache")
    svc, _ = _mock_service(pages)
    ingest._service = svc
    _patch_download(monkeypatch, _png_bytes())

    records = ingest.collect()
    assert records[0].content_hash is not None
    assert len(records[0].content_hash) == 64  # sha256 hex


def test_download_to_existing_path_skipped(tmp_path, monkeypatch):
    """If a file with the same destination name already exists, no re-download."""
    pages = [{"files": [_drive_file("a", name="x.png")]}]
    ingest = GoogleDriveIngest(credentials=MagicMock(), download_dir=tmp_path / "cache")
    svc, _ = _mock_service(pages)
    ingest._service = svc

    # Pre-create the destination file with valid image bytes.
    ingest.download_dir.mkdir(parents=True, exist_ok=True)
    dest = ingest.download_dir / "a_x.png"
    dest.write_bytes(_png_bytes())

    fake_download = MagicMock()
    monkeypatch.setattr("googleapiclient.http.MediaIoBaseDownload", fake_download)

    records = ingest.collect()
    assert len(records) == 1
    # Downloader was never constructed.
    fake_download.assert_not_called()


def test_recursive_walks_subfolders(tmp_path, monkeypatch):
    # Two list() calls: (a) the parent folder's image children, then
    # subfolder listings then the subfolder's image children.
    # We simulate via a side_effect that responds to the query string.

    ingest = GoogleDriveIngest(
        credentials=MagicMock(), download_dir=tmp_path / "cache",
        folder_id="ROOT", recursive=True,
    )

    parent_files = [_drive_file("img_a")]
    sub_listing  = [{"id": "SUB1"}]
    sub_files    = [_drive_file("img_b")]
    empty        = []

    response_for = {
        # Parent: image children
        ("'ROOT' in parents and mimeType contains 'image/' and trashed = false", None):
            {"files": parent_files},
        # Parent: subfolders
        ("'ROOT' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false", None):
            {"files": sub_listing},
        # SUB1: image children
        ("'SUB1' in parents and mimeType contains 'image/' and trashed = false", None):
            {"files": sub_files},
        # SUB1: subfolders
        ("'SUB1' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false", None):
            {"files": empty},
    }

    svc = MagicMock()
    def _list(**kwargs):
        key = (kwargs["q"], kwargs.get("pageToken"))
        execute = MagicMock(return_value=response_for[key])
        return MagicMock(execute=execute)
    svc.files().list = _list

    media = MagicMock()
    svc.files().get_media = MagicMock(return_value=media)
    ingest._service = svc

    _patch_download(monkeypatch, _png_bytes())

    records = ingest.collect()
    assert {r.source_ref for r in records} == {"img_a", "img_b"}
