"""Tests for auracast.ingest.google_photos.

We don't hit the real Photos API. We mock the service returned by
`build_photos_service` and the httpx download, then verify pagination,
album filtering, image-MIME filtering, and download failure handling.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from auracast.ingest.google_photos import GooglePhotosIngest
from auracast.schema.models import IngestSourceKind


def _make_media_item(item_id: str, mime: str = "image/jpeg") -> dict:
    return {
        "id": item_id,
        "mimeType": mime,
        "baseUrl": f"https://photos.example.com/{item_id}",
    }


def _png_bytes(color=(50, 100, 150)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def _mock_service(pages: list[dict]):
    """Build a fake Photos service whose .mediaItems().list/search returns these pages in order."""
    svc = MagicMock()
    pages_iter = iter(pages)

    list_method = MagicMock()
    list_method.execute = MagicMock(side_effect=lambda: next(pages_iter))
    svc.mediaItems().list = MagicMock(return_value=list_method)

    # search() takes a body kwarg
    search_method = MagicMock()
    search_method.execute = MagicMock(side_effect=lambda: next(pages_iter))
    svc.mediaItems().search = MagicMock(return_value=search_method)
    return svc


def test_lists_all_items_paginated(tmp_path):
    pages = [
        {"mediaItems": [_make_media_item("a"), _make_media_item("b")], "nextPageToken": "tok"},
        {"mediaItems": [_make_media_item("c")]},
    ]

    creds = MagicMock()
    ingest = GooglePhotosIngest(credentials=creds, download_dir=tmp_path / "cache")
    ingest._service = _mock_service(pages)

    with patch("httpx.stream") as mock_stream:
        ctx = MagicMock()
        ctx.iter_bytes.return_value = [_png_bytes()]
        ctx.raise_for_status = MagicMock()
        mock_stream.return_value.__enter__.return_value = ctx
        records = ingest.collect()

    assert len(records) == 3
    assert {r.source_ref for r in records} == {"a", "b", "c"}
    assert all(r.source == IngestSourceKind.GOOGLE_PHOTOS for r in records)


def test_filters_non_image_mime_types(tmp_path):
    pages = [{
        "mediaItems": [
            _make_media_item("img", mime="image/jpeg"),
            _make_media_item("vid", mime="video/mp4"),
        ],
    }]

    ingest = GooglePhotosIngest(credentials=MagicMock(), download_dir=tmp_path / "cache")
    ingest._service = _mock_service(pages)

    with patch("httpx.stream") as mock_stream:
        ctx = MagicMock()
        ctx.iter_bytes.return_value = [_png_bytes()]
        ctx.raise_for_status = MagicMock()
        mock_stream.return_value.__enter__.return_value = ctx
        records = ingest.collect()

    assert len(records) == 1
    assert records[0].source_ref == "img"


def test_max_items_caps_pagination(tmp_path):
    pages = [
        {"mediaItems": [_make_media_item(f"i{n}") for n in range(5)], "nextPageToken": "tok"},
        {"mediaItems": [_make_media_item(f"j{n}") for n in range(5)]},
    ]

    ingest = GooglePhotosIngest(
        credentials=MagicMock(), download_dir=tmp_path / "cache", max_items=3,
    )
    ingest._service = _mock_service(pages)

    with patch("httpx.stream") as mock_stream:
        ctx = MagicMock()
        ctx.iter_bytes.return_value = [_png_bytes()]
        ctx.raise_for_status = MagicMock()
        mock_stream.return_value.__enter__.return_value = ctx
        records = ingest.collect()

    assert len(records) == 3


def test_album_id_uses_search_endpoint(tmp_path):
    pages = [{"mediaItems": [_make_media_item("a")]}]

    ingest = GooglePhotosIngest(
        credentials=MagicMock(), download_dir=tmp_path / "cache", album_id="album-123",
    )
    svc = _mock_service(pages)
    ingest._service = svc

    with patch("httpx.stream") as mock_stream:
        ctx = MagicMock()
        ctx.iter_bytes.return_value = [_png_bytes()]
        ctx.raise_for_status = MagicMock()
        mock_stream.return_value.__enter__.return_value = ctx
        records = ingest.collect()

    assert len(records) == 1
    svc.mediaItems().search.assert_called_once()
    svc.mediaItems().list.assert_not_called()
    # search() body must include the album id
    body = svc.mediaItems().search.call_args.kwargs["body"]
    assert body["albumId"] == "album-123"


def test_download_failure_skips_item(tmp_path):
    pages = [{"mediaItems": [_make_media_item("good"), _make_media_item("bad")]}]
    ingest = GooglePhotosIngest(credentials=MagicMock(), download_dir=tmp_path / "cache")
    ingest._service = _mock_service(pages)

    call_count = {"n": 0}

    def _stream(method, url, **kwargs):
        call_count["n"] += 1
        cm = MagicMock()
        ctx = MagicMock()
        if "bad" in url:
            ctx.raise_for_status.side_effect = Exception("HTTP 500")
        else:
            ctx.iter_bytes.return_value = [_png_bytes()]
            ctx.raise_for_status = MagicMock()
        cm.__enter__.return_value = ctx
        cm.__exit__.return_value = False
        return cm

    with patch("httpx.stream", side_effect=_stream):
        records = ingest.collect()

    assert len(records) == 1
    assert records[0].source_ref == "good"


def test_missing_base_url_skipped(tmp_path):
    pages = [{"mediaItems": [
        {"id": "no-url", "mimeType": "image/jpeg"},  # no baseUrl
        _make_media_item("ok"),
    ]}]
    ingest = GooglePhotosIngest(credentials=MagicMock(), download_dir=tmp_path / "cache")
    ingest._service = _mock_service(pages)

    with patch("httpx.stream") as mock_stream:
        ctx = MagicMock()
        ctx.iter_bytes.return_value = [_png_bytes()]
        ctx.raise_for_status = MagicMock()
        mock_stream.return_value.__enter__.return_value = ctx
        records = ingest.collect()

    assert len(records) == 1
    assert records[0].source_ref == "ok"


def test_record_carries_content_hash(tmp_path):
    pages = [{"mediaItems": [_make_media_item("a")]}]
    ingest = GooglePhotosIngest(credentials=MagicMock(), download_dir=tmp_path / "cache")
    ingest._service = _mock_service(pages)

    with patch("httpx.stream") as mock_stream:
        ctx = MagicMock()
        ctx.iter_bytes.return_value = [_png_bytes()]
        ctx.raise_for_status = MagicMock()
        mock_stream.return_value.__enter__.return_value = ctx
        records = ingest.collect()

    assert records[0].content_hash is not None
    assert len(records[0].content_hash) == 64  # sha256 hex
