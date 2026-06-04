"""Tests for the scorer registry + ScorerModel enum + DriveProject integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from auracast.engine.registry import (
    SCORER_DESCRIPTIONS,
    SCORER_LABELS,
)
from auracast.projects import ProjectsStore
from auracast.schema.models import (
    DriveProject,
    ScorerModel,
    SCORER_MODELS_TAKING_TEXT,
    scorer_takes_text,
)


# -------- Enum + helper sanity -----------------------------------------


def test_all_scorer_models_have_labels_and_descriptions():
    for m in ScorerModel:
        assert m in SCORER_LABELS, f"{m} missing from SCORER_LABELS"
        assert m in SCORER_DESCRIPTIONS, f"{m} missing from SCORER_DESCRIPTIONS"
        assert SCORER_LABELS[m]  # non-empty
        assert SCORER_DESCRIPTIONS[m]


def test_scorer_takes_text_predictor_does_not():
    assert scorer_takes_text(ScorerModel.LAION_AESTHETIC) is False


def test_scorer_takes_text_prompt_models_do():
    assert scorer_takes_text(ScorerModel.CLIP_PROMPT) is True
    assert scorer_takes_text(ScorerModel.QWEN_VL_PROMPT) is True


def test_takes_text_set_membership():
    assert ScorerModel.CLIP_PROMPT in SCORER_MODELS_TAKING_TEXT
    assert ScorerModel.QWEN_VL_PROMPT in SCORER_MODELS_TAKING_TEXT
    assert ScorerModel.LAION_AESTHETIC not in SCORER_MODELS_TAKING_TEXT


# -------- DriveProject + ProjectsStore integration --------------------


def test_drive_project_default_scorer_is_clip_prompt():
    p = DriveProject(name="A", folder_id="f1", manifest_path=Path("a.jsonl"))
    assert p.scorer_model == ScorerModel.CLIP_PROMPT


def test_drive_project_custom_scorer():
    p = DriveProject(
        name="A", folder_id="f1", manifest_path=Path("a.jsonl"),
        scorer_model=ScorerModel.LAION_AESTHETIC,
    )
    assert p.scorer_model == ScorerModel.LAION_AESTHETIC


def test_store_update_scorer_model(tmp_path):
    store = ProjectsStore(tmp_path / "p.json")
    store.add(DriveProject(name="A", folder_id="f1", manifest_path=tmp_path / "a.jsonl"))
    assert store.update_scorer_model("A", ScorerModel.QWEN_VL_PROMPT) is True

    # Round-trip via disk
    store2 = ProjectsStore(tmp_path / "p.json")
    assert store2.get("A").scorer_model == ScorerModel.QWEN_VL_PROMPT


def test_store_update_scorer_unknown_name_returns_false(tmp_path):
    store = ProjectsStore(tmp_path / "p.json")
    assert store.update_scorer_model("missing", ScorerModel.LAION_AESTHETIC) is False


def test_scorer_model_round_trips_through_json(tmp_path):
    config = tmp_path / "p.json"
    store = ProjectsStore(config)
    store.add(DriveProject(
        name="Portraits",
        folder_id="f1",
        manifest_path=tmp_path / "a.jsonl",
        scorer_model=ScorerModel.QWEN_VL_PROMPT,
    ))
    reloaded = ProjectsStore(config).get("Portraits")
    assert reloaded.scorer_model == ScorerModel.QWEN_VL_PROMPT
