"""
End-to-end production pipeline runner. Resumable, dedupe-aware.

    # Score only (CLIP)
    python -m auracast.scripts.pipeline --source-dir data/mock_images

    # Score + caption (CLIP + Qwen2-VL)
    python -m auracast.scripts.pipeline --source-dir data/mock_images --caption

Subsequent invocations with the same `--manifest` will *skip* images whose
content_hash already appears, so iterating on a directory is cheap. Pass
`--rescore` to force re-scoring even when a hash matches (useful when
swapping the scoring model). Pass `--recaption` to re-run captioning on
items that already have a caption.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from auracast.engine.clip_scorer import CLIPScorer
from auracast.ingest.local_pipeline import LocalDirectoryIngest
from auracast.persistence import ManifestStore
from auracast.schema.models import ImageRecord, ProcessingStatus, ScoredImage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True, help="Directory of images to ingest.")
    p.add_argument(
        "--manifest", type=Path, default=Path("manifests/latest.jsonl"),
        help="JSONL manifest to read/write. Created if missing.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--rescore", action="store_true",
        help="Re-score images even if their hash already appears in the manifest.",
    )
    p.add_argument(
        "--retry-failed", action="store_true",
        help="Retry items whose processing_status is FAILED.",
    )
    p.add_argument(
        "--caption", action="store_true",
        help="Also run Qwen2-VL captioning. ~30s additional model load + per-image inference.",
    )
    p.add_argument(
        "--recaption", action="store_true",
        help="Re-run captioning on items that already have a caption (implies --caption).",
    )
    p.add_argument(
        "--caption-model", default=None,
        help="Override the Qwen2-VL model id (e.g. Qwen/Qwen2-VL-2B-Instruct for fast iteration).",
    )
    return p.parse_args()


def _score_pass(store: ManifestStore, todo: list[ImageRecord], batch_size: int) -> None:
    """Run CLIP scoring over `todo` and merge into the store. Failures recorded."""
    scorer = CLIPScorer()
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        try:
            outputs = scorer.score_batch(batch)
        except Exception as e:  # noqa: BLE001
            logger.exception("score batch starting at %d failed", i)
            for rec in batch:
                store.add_or_update(ScoredImage(
                    record=rec,
                    processing_status=ProcessingStatus.FAILED,
                    error=f"{type(e).__name__}: {e}",
                ), persist=False)
            continue
        by_id = {o.score.image_id: o for o in outputs}
        for rec in batch:
            o = by_id.get(rec.image_id)
            if o is None:
                si = ScoredImage(
                    record=rec,
                    processing_status=ProcessingStatus.FAILED,
                    error="scorer dropped this image (e.g. decode failure)",
                )
            else:
                si = ScoredImage(record=rec, scores=[o.score], embeddings=[o.embedding])
                si.processing_status = si.derive_processing_status()
            store.add_or_update(si, persist=False)
        logger.info("scored %d / %d", min(i + batch_size, len(todo)), len(todo))


def _caption_pass(store: ManifestStore, items: list[ScoredImage], model_id: str | None) -> None:
    """Run Qwen2-VL captioning over `items` and merge captions into the store."""
    from auracast.engine.qwen2vl_describer import DEFAULT_QWEN_MODEL, Qwen2VLDescriber

    describer = Qwen2VLDescriber(model_id=model_id or DEFAULT_QWEN_MODEL)
    records = [si.record for si in items]
    captions = describer.describe(records)
    by_id = {c.image_id: c for c in captions}
    for si in items:
        c = by_id.get(si.record.image_id)
        if c is None:
            # Captioning failed for this one — leave any previous state alone, log it.
            logger.warning("no caption produced for %s", si.record.image_id)
            continue
        # Append; don't overwrite — historical captions may be useful for A/B.
        new_captions = list(si.captions) + [c]
        updated = si.model_copy(update={"captions": new_captions})
        updated.processing_status = updated.derive_processing_status()
        store.add_or_update(updated, persist=False)
    logger.info("captioned %d / %d items", sum(1 for si in items if by_id.get(si.record.image_id)),
                len(items))


def main() -> None:
    args = _parse()
    do_caption = args.caption or args.recaption

    store = ManifestStore(args.manifest)
    logger.info("manifest: %s (%d existing items)", args.manifest, len(store))

    ingest = LocalDirectoryIngest(args.source_dir, compute_hash=True)
    raw_records = ingest.collect()
    logger.info("found %d images in %s", len(raw_records), args.source_dir)

    # Partition for the scoring pass.
    todo_score: list[ImageRecord] = []
    skipped_score = 0
    for rec in raw_records:
        existing = store.find_by_hash(rec.content_hash) if rec.content_hash else None
        if existing is None:
            todo_score.append(rec)
            continue
        if args.rescore:
            todo_score.append(rec)
            continue
        if args.retry_failed and existing.processing_status == ProcessingStatus.FAILED:
            todo_score.append(rec)
            continue
        skipped_score += 1
    logger.info("score pass: skipping %d already-processed; scoring %d",
                skipped_score, len(todo_score))

    if todo_score:
        _score_pass(store, todo_score, args.batch_size)
        store.bulk_add([])  # flush

    # Captioning pass — re-read state from the store after scoring.
    if do_caption:
        # Find which items in the store correspond to this directory (by hash)
        # and need a caption.
        items_in_dir: list[ScoredImage] = []
        for rec in raw_records:
            si = store.find_by_hash(rec.content_hash) if rec.content_hash else None
            if si is None:
                continue
            if si.processing_status == ProcessingStatus.FAILED:
                continue  # don't caption things we couldn't even score
            if si.captions and not args.recaption:
                continue
            items_in_dir.append(si)
        logger.info("caption pass: %d item(s) to caption", len(items_in_dir))
        if items_in_dir:
            _caption_pass(store, items_in_dir, args.caption_model)
            store.bulk_add([])  # flush

    logger.info("done. manifest now has %d items.", len(store))


if __name__ == "__main__":
    main()
