"""
Qwen2-VL describer.

Generates one Caption per image: a one-sentence description plus structured
attributes (mood, subject, hashtags) parsed from a JSON response. The model
is asked to emit JSON; we extract it robustly and fall back to free-form
text if parsing fails.

The 7B-Instruct variant is the default — best quality/cost trade-off on a
single A100-80GB at FP16 (~14 GB VRAM). 2B is available for fast iteration;
72B requires multi-GPU and is intentionally out of scope.

Model loading is lazy. Instantiating the class is cheap; the first
`describe()` call pays the ~30s load cost.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Sequence

from PIL import Image

from auracast.engine.device import DeviceSpec, pick_device_and_dtype
from auracast.schema.models import AestheticScore, Caption, Embedding, ImageRecord

logger = logging.getLogger(__name__)


DEFAULT_QWEN_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

DEFAULT_PROMPT = (
    "You are analyzing an image for a curated Instagram feed. "
    "Respond with ONLY a JSON object — no markdown, no commentary — "
    "matching exactly this schema:\n"
    "{\n"
    '  "caption": "one engaging sentence describing the image, suitable for an Instagram post",\n'
    '  "mood": "one of: warm, cool, neutral, moody, vibrant, serene, dramatic",\n'
    '  "subject": "the dominant subject in 1-3 words (e.g. \'urban architecture\', \'mountain landscape\')",\n'
    '  "hashtags": ["3-5 relevant hashtags, each starting with #"]\n'
    "}"
)


def _extract_json(text: str) -> dict | None:
    """Find a JSON object inside `text`. Returns None if nothing parses.

    Tried in order: whole-string parse, fenced ```json``` block, first {...} span.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _parse_response(raw_text: str) -> tuple[str, dict[str, str]]:
    """Turn the model's raw output into (caption, attributes).

    `attributes` always has string values to match the Caption schema; lists
    (hashtags) are joined with spaces. If JSON parsing fails entirely, the
    raw text becomes the caption and attributes is empty.
    """
    parsed = _extract_json(raw_text)
    if parsed is None:
        # Free-form fallback. Caption gets the trimmed raw text.
        return raw_text.strip(), {}

    caption = str(parsed.get("caption", "")).strip()
    attributes: dict[str, str] = {}
    for key in ("mood", "subject"):
        if key in parsed and parsed[key] is not None:
            attributes[key] = str(parsed[key]).strip()
    if "hashtags" in parsed and isinstance(parsed["hashtags"], list):
        attributes["hashtags"] = " ".join(str(h) for h in parsed["hashtags"])
    return caption, attributes


class Qwen2VLDescriber:
    """Generates Caption records from ImageRecords using Qwen2-VL."""

    def __init__(
        self,
        model_id: str = DEFAULT_QWEN_MODEL,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 256,
        device_spec: DeviceSpec | None = None,
    ):
        self.model_id = model_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.spec = device_spec or pick_device_and_dtype()
        self._model = None
        self._processor = None

    # ---- model loading -------------------------------------------------

    def _ensure_model(self):
        """Lazy-load Qwen2-VL. Idempotent."""
        if self._model is not None:
            return
        # Imports deferred so importing this module never costs us the
        # ~3-5s transformers init time. Critical on a Mac dev box where
        # we never actually load the model.
        import torch  # noqa: F401
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        logger.info(
            "loading Qwen2-VL %s on %s (%s) — this is heavy",
            self.model_id, self.spec.device, self.spec.dtype,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        # device_map="auto" lets transformers shard across multi-GPU when
        # available; on a single A100 it pins everything to GPU 0.
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.spec.dtype,
            device_map="auto",
        )
        self._model.eval()

    # ---- single-image generation --------------------------------------

    def _generate_one(self, image: Image.Image) -> str:
        """Run the model on a single image. Returns the raw decoded string."""
        import torch  # local import keeps top-level cheap

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": self.prompt},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.spec.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        # Strip the prompt tokens — keep only newly-generated tokens.
        prompt_len = inputs.input_ids.shape[1]
        new_tokens = generated_ids[:, prompt_len:]
        decoded = self._processor.batch_decode(new_tokens, skip_special_tokens=True)
        return decoded[0]

    # ---- public API ----------------------------------------------------

    def describe(self, records: Sequence[ImageRecord]) -> list[Caption]:
        """Real captioning. Skips records without local bytes.

        Currently single-image-at-a-time (Qwen2-VL's variable-resolution
        processor makes batching nontrivial — revisit in Phase 2 if it
        becomes a throughput bottleneck).
        """
        self._ensure_model()
        out: list[Caption] = []
        for rec in records:
            if not rec.has_local_bytes():
                logger.warning("skipping %s: no local bytes", rec.image_id)
                continue
            try:
                with Image.open(rec.file_path) as im:
                    im = im.convert("RGB")
                    raw = self._generate_one(im)
            except Exception as e:  # noqa: BLE001 — model failures are diverse
                logger.warning("describe failed for %s: %s", rec.image_id, e)
                continue
            caption_text, attributes = _parse_response(raw)
            out.append(Caption(
                image_id=rec.image_id,
                model=self.model_id,
                caption=caption_text,
                attributes=attributes,
            ))
        return out

    # ---- Scoring via VLM judgment --------------------------------------

    SCORING_PROMPT_TEMPLATE = (
        "You are scoring a candidate Instagram photo against this brief:\n"
        "\"{brief}\"\n\n"
        "Rate the image from 0.00 to 1.00 on how well it matches the brief. "
        "Pay attention to: facial expression and emotional authenticity when "
        "people are present, eye contact, lighting, composition (rule of "
        "thirds, headroom), background, and overall mood fit.\n\n"
        "Respond with ONLY this JSON, no commentary:\n"
        '{{"score": <0.00-1.00>, "reasons": "one sentence"}}'
    )

    def score(self, records: Sequence[ImageRecord], brief: str):
        """Use Qwen2-VL to rate each image's fit to a natural-language brief.

        Returns a list of (AestheticScore, raw_response) pairs; the score is
        wrapped to match the same shape CLIP/LAION scorers return so the
        Streamlit code can store it identically.
        """
        from auracast.engine.qwen2vl_describer import _extract_json
        self._ensure_model()
        prompt = self.SCORING_PROMPT_TEMPLATE.format(brief=brief.strip() or "a beautiful Instagram photograph")

        outputs: list[tuple[AestheticScore, str]] = []
        for rec in records:
            if not rec.has_local_bytes():
                logger.warning("skipping %s: no local bytes", rec.image_id)
                continue
            try:
                with Image.open(rec.file_path) as im:
                    im = im.convert("RGB")
                    raw = self._generate_one_with_prompt(im, prompt)
            except Exception as e:  # noqa: BLE001
                logger.warning("VLM score failed for %s: %s", rec.image_id, e)
                continue
            parsed = _extract_json(raw)
            try:
                score = float(parsed["score"]) if parsed else 0.0
            except (KeyError, TypeError, ValueError):
                score = 0.0
            # Clamp to [0, 1] for schema compliance.
            score = max(0.0, min(1.0, score))
            outputs.append((
                AestheticScore(
                    image_id=rec.image_id,
                    scorer=f"qwen2vl|{self.model_id}",
                    score=score,
                    raw_logits=None,
                ),
                raw,
            ))
        return outputs

    def _generate_one_with_prompt(self, image: Image.Image, prompt: str) -> str:
        """Like _generate_one but with a caller-supplied prompt."""
        import torch
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.spec.device)
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        prompt_len = inputs.input_ids.shape[1]
        new_tokens = generated_ids[:, prompt_len:]
        return self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def describe_stub(self, records: Sequence[ImageRecord]) -> list[Caption]:
        """Deterministic placeholder. Same shape as describe(), no model load."""
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
