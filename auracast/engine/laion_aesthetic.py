"""
LAION Aesthetic Predictor — a CLIP-L/14 backbone + a tiny linear head trained
on ~600k human aesthetic ratings (LAION-Aesthetics dataset).

No prompt needed; the head returns a single scalar per image (roughly in
the 0-10 range) which we map into AuraCast's 0-1 score convention.

Reference: https://github.com/christophschuhmann/improved-aesthetic-predictor
We use the `sac+logos+ava1-l14-linearMSE.pth` variant, which is a single
nn.Linear(768, 1) head. Weights are ~3 MB and downloaded on first use.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from auracast.engine.device import DeviceSpec, pick_device_and_dtype
from auracast.schema.models import AestheticScore, Embedding, ImageRecord

logger = logging.getLogger(__name__)


CLIP_BACKBONE_ID = "openai/clip-vit-large-patch14"
WEIGHTS_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
    "raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)


def _default_weights_path() -> Path:
    return Path.home() / ".cache" / "auracast" / "laion_aesthetic" / "linearMSE.pth"


def _download_weights(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    logger.info("downloading LAION aesthetic head -> %s", dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    tmp.replace(dest)
    return dest


@dataclass
class ScorerOutput:
    score: AestheticScore
    embedding: Embedding


class LaionAestheticScorer:
    """Single-pass aesthetic scorer. No prompt input.

    Workflow:
      image -> CLIP-L/14 image features (768-dim, L2-normalized)
            -> linear head -> raw score (~0..10)
            -> /10 clipped to [0, 1] -> AestheticScore.
    """

    SCORER_ID = "laion-aesthetic-linearMSE-l14"

    def __init__(
        self,
        device_spec: DeviceSpec | None = None,
        weights_path: Path | None = None,
    ):
        self.spec = device_spec or pick_device_and_dtype()
        self.weights_path = weights_path or _default_weights_path()
        self._clip: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None
        self._head: nn.Module | None = None

    def _ensure_loaded(self):
        if self._head is not None:
            return
        logger.info("loading CLIP-L/14 backbone on %s (%s)", self.spec.device, self.spec.dtype)
        self._processor = CLIPProcessor.from_pretrained(CLIP_BACKBONE_ID)
        self._clip = (
            CLIPModel.from_pretrained(CLIP_BACKBONE_ID, torch_dtype=self.spec.dtype)
            .to(self.spec.device)
            .eval()
        )
        weights = _download_weights(self.weights_path)
        head = nn.Linear(768, 1)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        head.load_state_dict(state)
        self._head = head.to(self.spec.device).eval()

    @torch.no_grad()
    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self._processor(images=list(images), return_tensors="pt").to(self.spec.device)
        pixel_values = inputs["pixel_values"].to(self.spec.dtype)
        vision_outputs = self._clip.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output  # [B, 1024]  (CLIP-L hidden)
        features = self._clip.visual_projection(pooled)  # [B, 768]
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def score_batch(self, records: Sequence[ImageRecord]) -> list[ScorerOutput]:
        self._ensure_loaded()
        pil_images: list[Image.Image] = []
        keep: list[int] = []
        for i, rec in enumerate(records):
            if not rec.has_local_bytes():
                logger.warning("skipping %s: no local bytes", rec.image_id)
                continue
            try:
                pil_images.append(Image.open(rec.file_path).convert("RGB"))
                keep.append(i)
            except Exception as e:  # noqa: BLE001
                logger.warning("skipping %s: decode failed (%s)", rec.image_id, e)

        if not pil_images:
            return []

        embeds = self._encode_images(pil_images)  # [B, 768]
        raw_scores = self._head(embeds).squeeze(-1)  # [B]
        normalized = (raw_scores / 10.0).clamp(0.0, 1.0).cpu().tolist()
        vectors = embeds.cpu().tolist()
        dim = embeds.shape[-1]

        out: list[ScorerOutput] = []
        for li, oi in enumerate(keep):
            rec = records[oi]
            out.append(ScorerOutput(
                score=AestheticScore(
                    image_id=rec.image_id,
                    scorer=self.SCORER_ID,
                    score=float(normalized[li]),
                    raw_logits=[float(raw_scores[li])],
                ),
                embedding=Embedding(
                    image_id=rec.image_id,
                    model=CLIP_BACKBONE_ID,
                    dim=int(dim),
                    vector=[float(x) for x in vectors[li]],
                ),
            ))
        return out
