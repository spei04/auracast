"""
Abstract ingest interface. Every ingest backend (local dir, Google Photos,
Google Drive, ...) implements `IngestSource` so downstream consumers don't
need to know where the bytes came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from auracast.schema.models import ImageRecord


class IngestSource(ABC):
    """Abstract base. Yields ImageRecord; does NOT score or embed."""

    @abstractmethod
    async def __aiter__(self) -> AsyncIterator[ImageRecord]:  # pragma: no cover
        """Yield ImageRecord one at a time. May fetch lazily."""
        raise NotImplementedError
        # The `yield` here is to satisfy the AsyncIterator type at runtime;
        # subclasses override the whole method.
        yield  # type: ignore[unreachable]
