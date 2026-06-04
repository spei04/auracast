"""
Scorer dispatch — pick the right backend by `ScorerModel` and run it.

Centralizes "which class to instantiate for which dropdown choice" so the
Streamlit app and CLI script don't have to duplicate the logic.
"""

from __future__ import annotations

import logging
from typing import Sequence

from auracast.persistence import ManifestStore
from auracast.schema.models import (
    AestheticScore,
    Embedding,
    ImageRecord,
    ScorerModel,
)

logger = logging.getLogger(__name__)


SCORER_LABELS: dict[ScorerModel, str] = {
    ScorerModel.CLIP_PROMPT: "CLIP — custom prompt (fast, prompt-driven)",
    ScorerModel.LAION_AESTHETIC: "LAION Aesthetic — general beauty (no prompt)",
    ScorerModel.QWEN_VL_PROMPT: "Qwen2-VL — smart custom prompt (slow, best for nuance)",
}


SCORER_DESCRIPTIONS: dict[ScorerModel, str] = {
    ScorerModel.CLIP_PROMPT: (
        "CLIP ViT-B/32 with positive and negative text prompts. Fast (~0.1s/image), "
        "decent at any criterion you can describe in a sentence. Best general-purpose pick."
    ),
    ScorerModel.LAION_AESTHETIC: (
        "CLIP-L/14 + LAION aesthetic head trained on ~600k human aesthetic ratings. "
        "Ignores prompts — returns its trained opinion of 'is this a beautiful photo'. "
        "Strong at landscapes and general aesthetic quality."
    ),
    ScorerModel.QWEN_VL_PROMPT: (
        "Qwen2-VL judges each image against your text brief. Best at portraits, "
        "facial expressions, eye contact, mood — anything nuanced CLIP can't grasp. "
        "Slow (~1-5s/image) but most flexible. Loads ~14 GB of weights on first use."
    ),
}


def score_records(
    model: ScorerModel,
    records: Sequence[ImageRecord],
    positive_prompt: str = "",
    negative_prompt: str = "",
) -> list[tuple[AestheticScore, Embedding | None]]:
    """Score `records` using the chosen `model`. Returns (score, embedding-or-None) pairs.

    Embedding is None for VLM-based scorers (they don't expose a vector head).
    """
    if model == ScorerModel.CLIP_PROMPT:
        from auracast.engine.clip_scorer import CLIPScorer
        scorer = CLIPScorer(
            positive_prompt=positive_prompt or "a beautiful, well-composed, high-quality photograph",
            negative_prompt=negative_prompt or "a blurry, low-quality, poorly composed snapshot",
        )
        return [(o.score, o.embedding) for o in scorer.score_batch(records)]

    if model == ScorerModel.LAION_AESTHETIC:
        from auracast.engine.laion_aesthetic import LaionAestheticScorer
        scorer = LaionAestheticScorer()
        return [(o.score, o.embedding) for o in scorer.score_batch(records)]

    if model == ScorerModel.QWEN_VL_PROMPT:
        from auracast.engine.qwen2vl_describer import Qwen2VLDescriber
        describer = Qwen2VLDescriber()
        outputs = describer.score(records, brief=positive_prompt)
        # VLM has no canonical image embedding for our store.
        return [(score, None) for (score, _raw) in outputs]

    raise ValueError(f"unknown scorer model: {model}")


def rescore_store(
    store: ManifestStore,
    model: ScorerModel,
    positive_prompt: str = "",
    negative_prompt: str = "",
) -> int:
    """Re-score every item with local bytes in `store` using `model`.

    Replaces each item's .scores list with one new AestheticScore so the UI
    sorts by the active scorer's output (no stale prior-prompt entries).
    Embeddings are updated when the scorer produces them; for VLM-based
    scorers the existing embeddings are kept.

    Returns the number of items actually re-scored.
    """
    records = [item.record for item in store.all() if item.record.has_local_bytes()]
    if not records:
        return 0

    results = score_records(model, records, positive_prompt, negative_prompt)
    by_id = {score.image_id: (score, embed) for (score, embed) in results}

    n_updated = 0
    for item in list(store.all()):
        pair = by_id.get(item.record.image_id)
        if pair is None:
            continue
        score, embed = pair
        update_fields = {"scores": [score]}
        if embed is not None:
            update_fields["embeddings"] = [embed]
        updated = item.model_copy(update=update_fields)
        updated.processing_status = updated.derive_processing_status()
        store.add_or_update(updated, persist=False)
        n_updated += 1
    store.bulk_add([])
    return n_updated
