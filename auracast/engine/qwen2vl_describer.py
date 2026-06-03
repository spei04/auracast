"""
Qwen2-VL describer (skeleton).

In production this generates rich captions and extracts structured attributes
(mood, dominant subject, recommended hashtags, ...) per image. For Phase 0 we
ship the interface + a stub that returns deterministic placeholder captions;
real model loading lands in Phase 1.

Heavy model loading lives behind a property so import-time cost stays zero
on the Mac dev box.
"""

from __future__ import annotations

import logging
from typing import Sequence

from PIL import Image

from auracast.engine.device import DeviceSpec, pick_device_and_dtype
from auracast.schema.models import Caption, ImageRecord

logger = logging.getLogger(__name__)


DEFAULT_QWEN_MODEL = "Qwen/Qwen2-VL-7B-Instruct"


class Qwen2VLDescriber:
    """Generates Caption records from ImageRecords. Skeleton — see Phase 1."""

    def __init__(self, model_id: str = DEFAULT_QWEN_MODEL, device_spec: DeviceSpec | None = None):
        self.model_id = model_id
        self.spec = device_spec or pick_device_and_dtype()
        self._model = None  # lazy-loaded

    def _ensure_model(self):
        if self._model is not None:
            return
        # Phase 1: actual load via:
        #   from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        #   self._processor = AutoProcessor.from_pretrained(self.model_id)
        #   self._model = Qwen2VLForConditionalGeneration.from_pretrained(
        #       self.model_id, torch_dtype=self.spec.dtype, device_map="auto"
        #   )
        raise NotImplementedError(
            "Qwen2-VL real loading lands in Phase 1; until then use describe_stub()."
        )

    def describe_stub(self, records: Sequence[ImageRecord]) -> list[Caption]:
        """Deterministic placeholder. Returns one Caption per record."""
        out: list[Caption] = []
        for rec in records:
            caption = (
                f"[stub] image {rec.image_id.hex[:8]} from {rec.source.value}"
                + (f" ({rec.width}x{rec.height})" if rec.width and rec.height else "")
            )
            out.append(Caption(
                image_id=rec.image_id,
                model=f"{self.model_id}#stub",
                caption=caption,
                attributes={"mood": "unknown", "subject": "unknown"},
            ))
        return out

    def describe(self, records: Sequence[ImageRecord]) -> list[Caption]:
        """Real captioning. Lazy-loads the model on first call."""
        self._ensure_model()
        raise NotImplementedError("Phase 1")
