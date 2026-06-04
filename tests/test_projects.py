"""Tests for auracast.projects + the DriveProject/ProjectsConfig schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from auracast.projects import ProjectsStore, parse_folder_id, slugify
from auracast.schema.models import DriveProject, ProjectsConfig


# -------- parse_folder_id ------------------------------------------------


def test_parse_folder_id_from_url():
    url = "https://drive.google.com/drive/folders/18B4e8D-akVEyU0BhbzvhvgZMS8rvSqm9"
    assert parse_folder_id(url) == "18B4e8D-akVEyU0BhbzvhvgZMS8rvSqm9"


def test_parse_folder_id_with_query_string():
    url = "https://drive.google.com/drive/folders/ABC123?usp=sharing"
    assert parse_folder_id(url) == "ABC123"


def test_parse_folder_id_bare_id():
    assert parse_folder_id("ABC123") == "ABC123"


def test_parse_folder_id_strips_whitespace():
    assert parse_folder_id("   ABC123\n") == "ABC123"


def test_parse_folder_id_empty():
    assert parse_folder_id("") == ""
    assert parse_folder_id(None) == ""  # type: ignore[arg-type]


# -------- slugify --------------------------------------------------------


def test_slugify_basic():
    assert slugify("Spring Aesthetic") == "spring_aesthetic"


def test_slugify_strips_special_chars():
    assert slugify("Travel: 2026!!") == "travel_2026"


def test_slugify_empty_falls_back():
    assert slugify("") == "project"
    assert slugify("???") == "project"


# -------- ProjectsConfig --------------------------------------------------


def test_config_add_first_sets_active():
    cfg = ProjectsConfig()
    p = DriveProject(name="A", folder_id="f1", manifest_path=Path("a.jsonl"))
    cfg.add(p)
    assert cfg.active_project_name == "A"


def test_config_add_duplicate_raises():
    cfg = ProjectsConfig()
    cfg.add(DriveProject(name="A", folder_id="f1", manifest_path=Path("a.jsonl")))
    with pytest.raises(ValueError, match="already exists"):
        cfg.add(DriveProject(name="A", folder_id="f2", manifest_path=Path("b.jsonl")))


def test_config_remove_updates_active():
    cfg = ProjectsConfig()
    cfg.add(DriveProject(name="A", folder_id="f1", manifest_path=Path("a.jsonl")))
    cfg.add(DriveProject(name="B", folder_id="f2", manifest_path=Path("b.jsonl")))
    assert cfg.active_project_name == "A"
    cfg.remove("A")
    assert cfg.active_project_name == "B"


def test_config_remove_last_clears_active():
    cfg = ProjectsConfig()
    cfg.add(DriveProject(name="A", folder_id="f1", manifest_path=Path("a.jsonl")))
    cfg.remove("A")
    assert cfg.active_project_name is None


def test_config_get():
    cfg = ProjectsConfig()
    p = DriveProject(name="A", folder_id="f1", manifest_path=Path("a.jsonl"))
    cfg.add(p)
    assert cfg.get("A") is not None
    assert cfg.get("missing") is None


# -------- ProjectsStore --------------------------------------------------


def test_store_round_trip(tmp_path):
    config_path = tmp_path / "projects.json"
    store = ProjectsStore(config_path)
    store.add(DriveProject(name="A", folder_id="f1", manifest_path=tmp_path / "a.jsonl"))
    store.add(DriveProject(name="B", folder_id="f2", manifest_path=tmp_path / "b.jsonl"))

    # Reload from disk
    store2 = ProjectsStore(config_path)
    assert {p.name for p in store2.all()} == {"A", "B"}
    assert store2.active().name == "A"  # first added is active


def test_store_set_active(tmp_path):
    config_path = tmp_path / "projects.json"
    store = ProjectsStore(config_path)
    store.add(DriveProject(name="A", folder_id="f1", manifest_path=tmp_path / "a.jsonl"))
    store.add(DriveProject(name="B", folder_id="f2", manifest_path=tmp_path / "b.jsonl"))

    assert store.set_active("B") is True
    # Reload — active persists
    store2 = ProjectsStore(config_path)
    assert store2.active().name == "B"


def test_store_set_active_unknown_returns_false(tmp_path):
    store = ProjectsStore(tmp_path / "p.json")
    assert store.set_active("missing") is False


def test_store_remove(tmp_path):
    store = ProjectsStore(tmp_path / "p.json")
    store.add(DriveProject(name="A", folder_id="f1", manifest_path=tmp_path / "a.jsonl"))
    assert store.remove("A") is True
    assert len(store.all()) == 0
    assert store.active() is None


def test_store_corrupt_file_starts_empty(tmp_path):
    config_path = tmp_path / "p.json"
    config_path.write_text("not valid json")
    store = ProjectsStore(config_path)
    assert store.all() == []
