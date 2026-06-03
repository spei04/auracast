"""Tests for the local-directory ingest pipeline."""

from __future__ import annotations

import pytest

from auracast.ingest.local_pipeline import LocalDirectoryIngest
from auracast.schema.models import IngestSourceKind


def test_ingest_walks_directory(mock_image_dir):
    ingest = LocalDirectoryIngest(mock_image_dir)
    records = ingest.collect()
    # 5 images written by the fixture; the .txt file is filtered out.
    assert len(records) == 5
    for r in records:
        assert r.source == IngestSourceKind.LOCAL_DIRECTORY
        assert r.file_path is not None
        assert r.file_path.exists()
        assert r.width == 64 and r.height == 64
        assert r.file_size_bytes is not None and r.file_size_bytes > 0


def test_ingest_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalDirectoryIngest(tmp_path / "does-not-exist")


def test_ingest_rejects_non_directory(tmp_path):
    f = tmp_path / "afile.txt"
    f.write_text("hi")
    with pytest.raises(NotADirectoryError):
        LocalDirectoryIngest(f)


def test_ingest_suffix_filter(mock_image_dir):
    # Only .jpg suffix — none of the fixture images match.
    ingest = LocalDirectoryIngest(mock_image_dir, suffixes=(".jpg",))
    records = ingest.collect()
    assert records == []


async def test_ingest_async_iter(mock_image_dir):
    ingest = LocalDirectoryIngest(mock_image_dir)
    collected = []
    async for rec in ingest:
        collected.append(rec)
    assert len(collected) == 5
