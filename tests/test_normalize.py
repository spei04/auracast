"""Tests for the min-max score normalization helpers in streamlit_app."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Streamlit isn't a hard test dep; the helpers we want to test live in the
# Streamlit module, which imports `streamlit` at top-level. Skip the whole
# file gracefully if streamlit can't be imported (CI without streamlit).
try:
    from auracast.app.streamlit_app import _normalize, _score_range
except ModuleNotFoundError:
    pytest.skip("streamlit not installed; skipping streamlit_app helper tests",
                allow_module_level=True)

from auracast.persistence import ManifestStore
from auracast.schema.models import (
    AestheticScore,
    ImageRecord,
    IngestSourceKind,
    ScoredImage,
)


def test_normalize_simple_range():
    assert _normalize(5.0, 0.0, 10.0) == pytest.approx(0.5)
    assert _normalize(0.0, 0.0, 10.0) == pytest.approx(0.0)
    assert _normalize(10.0, 0.0, 10.0) == pytest.approx(1.0)


def test_normalize_within_bounds():
    assert _normalize(0.7, 0.3, 0.9) == pytest.approx((0.7 - 0.3) / (0.9 - 0.3))


def test_normalize_degenerate_returns_half():
    """When all scores are tied, normalization is undefined — return 0.5."""
    assert _normalize(0.5, 0.5, 0.5) == 0.5
    assert _normalize(0.7, 0.7, 0.7) == 0.5


def test_normalize_handles_inverted_range_safely():
    """If somehow lo > hi, don't crash — fall back to 0.5."""
    assert _normalize(0.5, 0.9, 0.1) == 0.5


def test_score_range_empty_store_defaults_to_unit_interval(tmp_path):
    store = ManifestStore(tmp_path / "m.jsonl")
    lo, hi = _score_range(store)
    assert (lo, hi) == (0.0, 1.0)


def _scored_item(score: float | None) -> ScoredImage:
    rec = ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY)
    scores = []
    if score is not None:
        scores.append(AestheticScore(image_id=rec.image_id, scorer="t", score=score))
    return ScoredImage(record=rec, scores=scores)


def test_score_range_with_scored_items(tmp_path):
    store = ManifestStore(tmp_path / "m.jsonl")
    store.bulk_add([_scored_item(0.2), _scored_item(0.7), _scored_item(0.5)])
    lo, hi = _score_range(store)
    assert lo == pytest.approx(0.2)
    assert hi == pytest.approx(0.7)


def test_score_range_skips_unscored(tmp_path):
    store = ManifestStore(tmp_path / "m.jsonl")
    store.bulk_add([_scored_item(None), _scored_item(0.6), _scored_item(0.6)])
    lo, hi = _score_range(store)
    assert lo == pytest.approx(0.6)
    assert hi == pytest.approx(0.6)
