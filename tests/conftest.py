"""Shared pytest fixtures."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def mock_image_dir(tmp_path: Path) -> Path:
    """Drop a handful of small random-color PNGs into a tmp dir."""
    n = 5
    for i in range(n):
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        Image.new("RGB", (64, 64), color).save(tmp_path / f"img_{i:02d}.png")
    # Add a non-image to exercise the suffix filter.
    (tmp_path / "ignore.txt").write_text("not an image")
    return tmp_path
