"""Tests for auracast.persistence — atomic writes, dedupe, review updates."""

from __future__ import annotations

from pathlib import Path

import pytest

from auracast.persistence import ManifestStore, sha256_file
from auracast.schema.models import (
    AestheticScore,
    ImageRecord,
    IngestSourceKind,
    ProcessingStatus,
    ReviewStatus,
    ScoredImage,
)


def _make_image(tmp_path: Path, color: tuple[int, int, int], name: str) -> Path:
    from PIL import Image
    p = tmp_path / name
    Image.new("RGB", (32, 32), color).save(p)
    return p


def _record_with_hash(path: Path) -> ImageRecord:
    return ImageRecord(
        source=IngestSourceKind.LOCAL_DIRECTORY,
        file_path=path,
        content_hash=sha256_file(path),
    )


def test_sha256_of_identical_files_matches(tmp_path):
    p1 = _make_image(tmp_path, (10, 20, 30), "a.png")
    p2 = _make_image(tmp_path, (10, 20, 30), "b.png")
    assert sha256_file(p1) == sha256_file(p2)


def test_sha256_of_different_files_differs(tmp_path):
    p1 = _make_image(tmp_path, (10, 20, 30), "a.png")
    p2 = _make_image(tmp_path, (200, 100, 50), "c.png")
    assert sha256_file(p1) != sha256_file(p2)


def test_store_persists_atomically(tmp_path):
    manifest_path = tmp_path / "m.jsonl"
    store = ManifestStore(manifest_path)
    rec = ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY)
    store.add_or_update(ScoredImage(record=rec))
    assert manifest_path.exists()
    # No leftover .tmp file
    assert not manifest_path.with_suffix(".jsonl.tmp").exists()


def test_store_round_trip(tmp_path):
    manifest_path = tmp_path / "m.jsonl"
    store = ManifestStore(manifest_path)
    rec = ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY, content_hash="abc123")
    store.add_or_update(ScoredImage(record=rec))

    store2 = ManifestStore(manifest_path)
    assert len(store2) == 1
    assert store2.find_by_hash("abc123") is not None
    assert store2.find_by_hash("abc123").record.image_id == rec.image_id


def test_store_dedupe_by_hash(tmp_path):
    manifest_path = tmp_path / "m.jsonl"
    store = ManifestStore(manifest_path)
    rec1 = ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY, content_hash="dup")
    store.add_or_update(ScoredImage(record=rec1))

    # Same hash, different image_id: lookup should return the latest one
    rec2 = ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY, content_hash="dup")
    store.add_or_update(ScoredImage(record=rec2))

    found = store.find_by_hash("dup")
    assert found is not None
    # Two records (different UUIDs); we don't enforce hash uniqueness here.
    assert len(store) == 2


def test_update_review_persists(tmp_path):
    manifest_path = tmp_path / "m.jsonl"
    store = ManifestStore(manifest_path)
    rec = ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY)
    store.add_or_update(ScoredImage(record=rec))
    assert store.update_review(rec.image_id, ReviewStatus.APPROVED) is True

    store2 = ManifestStore(manifest_path)
    assert store2.get(rec.image_id).review_status == ReviewStatus.APPROVED


def test_update_review_unknown_id_returns_false(tmp_path):
    from uuid import uuid4
    store = ManifestStore(tmp_path / "m.jsonl")
    assert store.update_review(uuid4(), ReviewStatus.APPROVED) is False


def test_derive_processing_status():
    rec = ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY)
    si = ScoredImage(record=rec)
    assert si.derive_processing_status() == ProcessingStatus.PENDING

    si_with_error = ScoredImage(record=rec, error="boom")
    assert si_with_error.derive_processing_status() == ProcessingStatus.FAILED


def test_bulk_add_writes_once(tmp_path, monkeypatch):
    manifest_path = tmp_path / "m.jsonl"
    store = ManifestStore(manifest_path)

    # Spy on _flush_locked by counting calls
    n_flushes = [0]
    original = store._flush_locked

    def counting_flush():
        n_flushes[0] += 1
        original()

    store._flush_locked = counting_flush
    items = [
        ScoredImage(record=ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY))
        for _ in range(5)
    ]
    store.bulk_add(items)
    assert n_flushes[0] == 1
    assert len(store) == 5
