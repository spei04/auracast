"""Tests for the Qwen2-VL describer.

Real model loading is impossible on a CPU-only dev box (15 GB download +
GPU required), so we test the *contract*: prompt formatting, JSON parsing,
graceful fallback when JSON is malformed, and the stub interface.

The actual model is replaced with a fake that records what it was called
with and returns canned strings.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from auracast.engine.qwen2vl_describer import (
    DEFAULT_PROMPT,
    Qwen2VLDescriber,
    _extract_json,
    _parse_response,
)
from auracast.schema.models import ImageRecord, IngestSourceKind


# -------- _extract_json -------------------------------------------------


def test_extract_json_clean():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_surrounding_text():
    raw = 'Here is the JSON:\n{"caption": "a cat", "mood": "warm"}\nThat is all.'
    parsed = _extract_json(raw)
    assert parsed == {"caption": "a cat", "mood": "warm"}


def test_extract_json_in_code_fence():
    raw = '```json\n{"caption": "fenced"}\n```'
    assert _extract_json(raw) == {"caption": "fenced"}


def test_extract_json_returns_none_on_garbage():
    assert _extract_json("not json at all just words") is None


# -------- _parse_response ----------------------------------------------


def test_parse_response_full_json():
    raw = (
        '{"caption": "Golden hour on the beach.", '
        '"mood": "warm", "subject": "seascape", '
        '"hashtags": ["#sunset", "#beach", "#goldenhour"]}'
    )
    caption, attrs = _parse_response(raw)
    assert caption == "Golden hour on the beach."
    assert attrs["mood"] == "warm"
    assert attrs["subject"] == "seascape"
    assert "#sunset" in attrs["hashtags"]


def test_parse_response_partial_json():
    """Missing keys should not crash — they just don't appear in attributes."""
    caption, attrs = _parse_response('{"caption": "only a caption"}')
    assert caption == "only a caption"
    assert attrs == {}


def test_parse_response_fallback_to_raw_text():
    """When JSON parsing fails entirely, raw text becomes the caption."""
    caption, attrs = _parse_response("Just a freeform description, no braces.")
    assert caption == "Just a freeform description, no braces."
    assert attrs == {}


def test_parse_response_strips_whitespace():
    caption, _ = _parse_response('   {"caption": "trimmed"}   ')
    assert caption == "trimmed"


# -------- stub interface (no model load) -------------------------------


def test_describe_stub_produces_one_per_record():
    describer = Qwen2VLDescriber()
    recs = [
        ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY, width=128, height=128),
        ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY, width=256, height=256),
    ]
    captions = describer.describe_stub(recs)
    assert len(captions) == 2
    assert all(c.model.endswith("#stub") for c in captions)
    assert all(c.image_id in {r.image_id for r in recs} for c in captions)


# -------- real describe() path with mocked model ----------------------


def _record_for_image(tmp_path: Path) -> ImageRecord:
    p = tmp_path / "x.png"
    Image.new("RGB", (32, 32), (100, 150, 200)).save(p)
    return ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY, file_path=p, width=32, height=32)


def test_describe_skips_records_without_local_bytes(tmp_path, monkeypatch):
    describer = Qwen2VLDescriber()
    # Pretend the model is loaded so _ensure_model is a no-op.
    describer._model = MagicMock()
    describer._processor = MagicMock()
    describer._generate_one = MagicMock(return_value='{"caption": "ignored"}')

    rec_no_bytes = ImageRecord(source=IngestSourceKind.LOCAL_DIRECTORY)  # no file_path
    captions = describer.describe([rec_no_bytes])
    assert captions == []
    describer._generate_one.assert_not_called()


def test_describe_returns_parsed_caption(tmp_path):
    describer = Qwen2VLDescriber()
    describer._model = MagicMock()
    describer._processor = MagicMock()
    describer._generate_one = MagicMock(return_value=(
        '{"caption": "A vivid abstract.", '
        '"mood": "vibrant", "subject": "abstract", "hashtags": ["#art"]}'
    ))

    rec = _record_for_image(tmp_path)
    captions = describer.describe([rec])
    assert len(captions) == 1
    c = captions[0]
    assert c.image_id == rec.image_id
    assert c.caption == "A vivid abstract."
    assert c.attributes["mood"] == "vibrant"
    assert "#art" in c.attributes["hashtags"]


def test_describe_continues_on_per_image_error(tmp_path):
    """One bad image shouldn't kill the whole batch."""
    describer = Qwen2VLDescriber()
    describer._model = MagicMock()
    describer._processor = MagicMock()

    rec_good = _record_for_image(tmp_path)
    # Patch _generate_one to raise once, then succeed.
    calls = {"n": 0}

    def flaky_generate(image):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated OOM")
        return '{"caption": "recovered"}'

    describer._generate_one = flaky_generate
    captions = describer.describe([rec_good, rec_good])
    # First fails, second succeeds.
    assert len(captions) == 1
    assert captions[0].caption == "recovered"


def test_default_prompt_asks_for_json():
    """Sanity-check the prompt — it must demand JSON output, otherwise parsing breaks."""
    assert "JSON" in DEFAULT_PROMPT
    assert "caption" in DEFAULT_PROMPT
    assert "mood" in DEFAULT_PROMPT
