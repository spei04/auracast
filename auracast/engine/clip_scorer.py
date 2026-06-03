"""
CLIP / SigLIP scoring + embedding.

One class, two outputs per image:
  - AestheticScore in [0, 1] (from cosine similarity to a curated aesthetic prompt).
  - Embedding (the image-tower vector — same one used for similarity search later).

Default backbone: `openai/clip-vit-base-patch32` — small, ~150 MB, fast on A100.
Swap via constructor arg without changing the call sites in the pipeline.

All `torch` imports are gated to this module (per CLAUDE.md convention).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from auracast.engine.device import DeviceSpec, pick_device_and_dtype
from auracast.schema.models import AestheticScore, Embedding, ImageRecord

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "openai/clip-vit-base-patch32"

# Anchor prompts: the cosine similarity of an image's embedding to the
# *positive* prompt minus its similarity to the *negative* prompt — squashed
# to [0, 1] via sigmoid — is our aesthetic score. This is a lightweight
# stand-in for a fine-tuned aesthetic head; revisit when we have human-labeled
# preference data.
DEFAULT_POSITIVE_PROMPT = "a beautiful, well-composed, high-quality photograph"
DEFAULT_NEGATIVE_PROMPT = "a blurry, low-quality, poorly composed snapshot"


@dataclass
class ScorerOutput:
    score: AestheticScore
    embedding: Embedding


class CLIPScorer:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        positive_prompt: str = DEFAULT_POSITIVE_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        device_spec: DeviceSpec | None = None,
    ):
        self.model_id = model_id
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.spec = device_spec or pick_device_and_dtype()

        logger.info("loading CLIP model %s on %s (%s)", model_id, self.spec.device, self.spec.dtype)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=self.spec.dtype).to(
            self.spec.device
        )
        self.model.eval()
        self._prompt_embeds = self._encode_prompt_embeddings()

    @torch.no_grad()
    def _encode_prompt_embeddings(self) -> torch.Tensor:
        """Cache the [2, D] (positive, negative) prompt embeddings.

        We bypass `get_text_features` and call `text_model` + `text_projection`
        directly — `get_text_features`'s return type varies across transformers
        versions (sometimes a Tensor, sometimes a ModelOutput wrapper).
        """
        inputs = self.processor(
            text=[self.positive_prompt, self.negative_prompt],
            return_tensors="pt",
            padding=True,
        ).to(self.spec.device)
        text_outputs = self.model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        pooled = text_outputs.pooler_output  # [2, hidden_dim]
        text_features = self.model.text_projection(pooled)  # [2, proj_dim]
        return F.normalize(text_features.float(), dim=-1)

    @torch.no_grad()
    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.spec.device)
        pixel_values = inputs["pixel_values"].to(self.spec.dtype)
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output       # [B, hidden_dim]
        image_features = self.model.visual_projection(pooled)  # [B, proj_dim]
        return F.normalize(image_features.float(), dim=-1)

    @torch.no_grad()
    def score_batch(self, records: Sequence[ImageRecord]) -> list[ScorerOutput]:
        """Load images, compute scores + embeddings, return one ScorerOutput per input."""
        pil_images: list[Image.Image] = []
        keep_indices: list[int] = []
        for i, rec in enumerate(records):
            if not rec.has_local_bytes():
                logger.warning("skipping record %s: no local bytes", rec.image_id)
                continue
            try:
                pil_images.append(Image.open(rec.file_path).convert("RGB"))
                keep_indices.append(i)
            except Exception as e:  # noqa: BLE001
                logger.warning("skipping record %s: decode failed (%s)", rec.image_id, e)

        if not pil_images:
            return []

        image_features = self._encode_images(pil_images)  # [B, D]
        # Aesthetic score: pos_sim - neg_sim, scaled by CLIP's logit_scale, then sigmoid.
        sims = image_features @ self._prompt_embeds.T  # [B, 2]
        margin = sims[:, 0] - sims[:, 1]                # [B]
        # Scale by half of CLIP's logit_scale for numerical stability; sigmoid -> [0,1].
        logit_scale = float(self.model.logit_scale.exp().detach().cpu()) * 0.5
        scores = torch.sigmoid(margin * logit_scale).cpu().tolist()

        dim = image_features.shape[-1]
        vectors = image_features.cpu().tolist()

        outputs: list[ScorerOutput] = []
        for local_i, original_i in enumerate(keep_indices):
            rec = records[original_i]
            outputs.append(
                ScorerOutput(
                    score=AestheticScore(
                        image_id=rec.image_id,
                        scorer=self.model_id,
                        score=float(scores[local_i]),
                        raw_logits=[float(sims[local_i, 0]), float(sims[local_i, 1])],
                    ),
                    embedding=Embedding(
                        image_id=rec.image_id,
                        model=self.model_id,
                        dim=int(dim),
                        vector=[float(x) for x in vectors[local_i]],
                    ),
                )
            )
        return outputs
