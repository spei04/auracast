"""
Pydantic v2 data models. The single source of truth for what flows between
ingest, engine, and app subsystems.

All cross-module data uses these types. Never pass raw dicts across a module
boundary — convert to the right model at the boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


# -------- Enums ----------------------------------------------------------


class IngestSourceKind(str, Enum):
    """Where an image came from. Adds a new value per ingest backend."""

    LOCAL_DIRECTORY = "local_directory"
    GOOGLE_PHOTOS = "google_photos"
    GOOGLE_DRIVE = "google_drive"


class ReviewStatus(str, Enum):
    """Human review state captured in the Streamlit app."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# -------- Core records ---------------------------------------------------


def _utcnow() -> datetime:
    """Centralized UTC-aware timestamp factory so tests can monkeypatch one place."""
    return datetime.now(tz=timezone.utc)


class ImageRecord(BaseModel):
    """One image as ingested. Pre-scoring; pre-embedding."""

    model_config = ConfigDict(extra="forbid")

    image_id: UUID = Field(default_factory=uuid4)
    source: IngestSourceKind
    file_path: Path | None = None  # populated for local-disk-resident images
    source_url: HttpUrl | None = None  # populated for remote-only images
    source_ref: str | None = None  # opaque ID from the source (e.g. Google Photos mediaItem ID)
    ingested_at: datetime = Field(default_factory=_utcnow)
    width: int | None = None
    height: int | None = None
    file_size_bytes: int | None = None
    mime_type: str | None = None

    @field_validator("file_path")
    @classmethod
    def _require_path_or_url(cls, v, info):
        # We don't enforce here at field-level — we check pairwise in a model
        # validator below — but coerce string paths to Path on the way in.
        if v is None:
            return v
        return v if isinstance(v, Path) else Path(v)

    def has_local_bytes(self) -> bool:
        return self.file_path is not None and self.file_path.exists()


class AestheticScore(BaseModel):
    """Output of a single scorer run on a single image."""

    model_config = ConfigDict(extra="forbid")

    image_id: UUID
    scorer: str  # e.g. "openai/clip-vit-base-patch32" or "siglip-so400m-patch14-384"
    score: float = Field(ge=0.0, le=1.0)
    raw_logits: list[float] | None = None  # optional, for debugging / re-calibration
    scored_at: datetime = Field(default_factory=_utcnow)


class Embedding(BaseModel):
    """Vector embedding of an image from a vision encoder."""

    model_config = ConfigDict(extra="forbid")

    image_id: UUID
    model: str  # e.g. "openai/clip-vit-base-patch32"
    dim: int
    vector: list[float]
    embedded_at: datetime = Field(default_factory=_utcnow)

    @field_validator("vector")
    @classmethod
    def _vector_matches_dim(cls, v, info):
        dim = info.data.get("dim")
        if dim is not None and len(v) != dim:
            raise ValueError(f"vector length {len(v)} != declared dim {dim}")
        return v


class Caption(BaseModel):
    """Caption / structured-attribute output from a VLM (e.g. Qwen2-VL)."""

    model_config = ConfigDict(extra="forbid")

    image_id: UUID
    model: str
    caption: str
    attributes: dict[str, str] = Field(default_factory=dict)  # e.g. {"mood": "warm"}
    captioned_at: datetime = Field(default_factory=_utcnow)


# -------- Joined / view types --------------------------------------------


class ScoredImage(BaseModel):
    """An image plus everything we know about it. Used by the Streamlit app."""

    model_config = ConfigDict(extra="forbid")

    record: ImageRecord
    scores: list[AestheticScore] = Field(default_factory=list)
    embeddings: list[Embedding] = Field(default_factory=list)
    captions: list[Caption] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.PENDING

    def top_score(self) -> float | None:
        """Highest score across all scorers. None if no scores yet."""
        return max((s.score for s in self.scores), default=None)


# -------- Manifest -------------------------------------------------------


class Manifest(BaseModel):
    """A batch of ScoredImage records written/read as JSONL on disk."""

    model_config = ConfigDict(extra="forbid")

    items: list[ScoredImage]
    created_at: datetime = Field(default_factory=_utcnow)
    notes: str | None = None

    def to_jsonl(self) -> str:
        return "\n".join(item.model_dump_json() for item in self.items)

    @classmethod
    def from_jsonl(cls, text: str) -> "Manifest":
        items: list[ScoredImage] = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                items.append(ScoredImage.model_validate_json(line))
        return cls(items=items)
