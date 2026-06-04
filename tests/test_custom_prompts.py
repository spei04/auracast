"""Tests for per-project aesthetic prompts."""

from __future__ import annotations

from pathlib import Path

from auracast.projects import ProjectsStore
from auracast.schema.models import DriveProject


def test_drive_project_has_default_prompts():
    p = DriveProject(name="A", folder_id="f1", manifest_path=Path("a.jsonl"))
    assert "high-quality" in p.positive_prompt
    assert "blurry" in p.negative_prompt


def test_drive_project_custom_prompts():
    p = DriveProject(
        name="Moody",
        folder_id="f1",
        manifest_path=Path("a.jsonl"),
        positive_prompt="dark cinematic shadow play",
        negative_prompt="overexposed cheerful daylight",
    )
    assert p.positive_prompt == "dark cinematic shadow play"


def test_store_update_prompts(tmp_path):
    store = ProjectsStore(tmp_path / "p.json")
    store.add(DriveProject(name="A", folder_id="f1", manifest_path=tmp_path / "a.jsonl"))

    assert store.update_prompts("A", "moody", "bright") is True
    fresh = ProjectsStore(tmp_path / "p.json").get("A")
    assert fresh.positive_prompt == "moody"
    assert fresh.negative_prompt == "bright"


def test_store_update_prompts_unknown_returns_false(tmp_path):
    store = ProjectsStore(tmp_path / "p.json")
    assert store.update_prompts("missing", "x", "y") is False


def test_prompts_round_trip_through_disk(tmp_path):
    config = tmp_path / "p.json"
    store = ProjectsStore(config)
    store.add(DriveProject(
        name="A", folder_id="f1", manifest_path=tmp_path / "a.jsonl",
        positive_prompt="serene minimalist nature photography",
        negative_prompt="cluttered busy urban snapshot",
    ))
    # Re-open from disk
    store2 = ProjectsStore(config)
    p = store2.get("A")
    assert p.positive_prompt == "serene minimalist nature photography"
    assert p.negative_prompt == "cluttered busy urban snapshot"
