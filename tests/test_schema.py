"""Tests for auracast.schema.models."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from auracast.schema.models import (
    AestheticScore,
    Caption,
    Embedding,
    ImageRecord,
    IngestSourceKind,
    Manifest,
    ReviewStatus,
    ScoredImage,
)


def test_image_record_minimal():
    rec = ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY)
    assert rec.image_id is not None
    assert rec.source == IngestSourceKind.LOCAL_DIRECTORY
    assert rec.ingested_at is not None


def test_image_record_file_path_coerced_to_path(tmp_path):
    rec = ImageRecord(
        source=IngestSourceKind.LOCAL_DIRECTORY,
        file_path=str(tmp_path / "x.png"),
    )
    assert isinstance(rec.file_path, Path)


def test_image_record_rejects_extra_fields():
    with pytest.raises(Exception):
        ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY, bogus="nope")


def test_aesthetic_score_in_range():
    s = AestheticScore(image_id=uuid4(), scorer="clip", score=0.42)
    assert 0.0 <= s.score <= 1.0


def test_aesthetic_score_out_of_range_rejected():
    with pytest.raises(Exception):
        AestheticScore(image_id=uuid4(), scorer="clip", score=1.5)


def test_embedding_dim_must_match_vector_length():
    with pytest.raises(Exception):
        Embedding(image_id=uuid4(), model="clip", dim=4, vector=[0.1, 0.2])


def test_embedding_round_trip():
    e = Embedding(image_id=uuid4(), model="clip", dim=3, vector=[0.1, 0.2, 0.3])
    json_s = e.model_dump_json()
    e2 = Embedding.model_validate_json(json_s)
    assert e2.vector == e.vector
    assert e2.dim == e.dim


def test_caption_attributes_default_empty():
    c = Caption(image_id=uuid4(), model="qwen2vl", caption="a cat")
    assert c.attributes == {}


def test_scored_image_top_score_none_when_no_scores():
    rec = ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY)
    si = ScoredImage(record=rec)
    assert si.top_score() is None


def test_scored_image_top_score_returns_max():
    rec = ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY)
    rec_id = rec.image_id
    si = ScoredImage(record=rec, scores=[
        AestheticScore(image_id=rec_id, scorer="a", score=0.3),
        AestheticScore(image_id=rec_id, scorer="b", score=0.8),
        AestheticScore(image_id=rec_id, scorer="c", score=0.5),
    ])
    assert si.top_score() == pytest.approx(0.8)


def test_manifest_jsonl_round_trip():
    rec = ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY, width=64, height=64)
    si = ScoredImage(
        record=rec,
        scores=[AestheticScore(image_id=rec.image_id, scorer="clip", score=0.7)],
        embeddings=[Embedding(image_id=rec.image_id, model="clip", dim=2, vector=[0.5, 0.5])],
        review_status=ReviewStatus.APPROVED,
    )
    m = Manifest(items=[si, si])
    text = m.to_jsonl()
    assert text.count("\n") == 1  # two items -> one separator
    m2 = Manifest.from_jsonl(text)
    assert len(m2.items) == 2
    assert m2.items[0].review_status == ReviewStatus.APPROVED
