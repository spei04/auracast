"""
End-to-end production pipeline runner. Resumable, dedupe-aware.

    python -m auracast.scripts.pipeline \\
        --source-dir data/mock_images \\
        --manifest manifests/latest.jsonl \\
        --batch-size 32

Subsequent invocations with the same `--manifest` will *skip* images whose
content_hash already appears, so iterating on a directory is cheap. Pass
`--rescore` to force re-scoring even when a hash matches (useful when
swapping the scoring model).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from auracast.engine.clip_scorer import CLIPScorer
from auracast.ingest.local_pipeline import LocalDirectoryIngest
from auracast.persistence import ManifestStore
from auracast.schema.models import ProcessingStatus, ScoredImage

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
    return p.parse_args()


def main() -> None:
    args = _parse()
    store = ManifestStore(args.manifest)
    logger.info("manifest: %s (%d existing items)", args.manifest, len(store))

    # 1. Ingest source directory. Hashing lets us dedupe against the existing manifest.
    ingest = LocalDirectoryIngest(args.source_dir, compute_hash=True)
    raw_records = ingest.collect()
    logger.info("found %d images in %s", len(raw_records), args.source_dir)

    # 2. Partition into (need to score) vs (already done).
    todo: list = []
    skipped = 0
    for rec in raw_records:
        existing = store.find_by_hash(rec.content_hash) if rec.content_hash else None
        if existing is None:
            todo.append(rec)
            continue
        if args.rescore:
            todo.append(rec)
            continue
        if args.retry_failed and existing.processing_status == ProcessingStatus.FAILED:
            todo.append(rec)
            continue
        skipped += 1
    logger.info("skipping %d already-processed; scoring %d", skipped, len(todo))

    if not todo:
        logger.info("nothing to do.")
        return

    # 3. Score in batches. Failures don't kill the run — mark FAILED on the record.
    scorer = CLIPScorer()
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        try:
            outputs = scorer.score_batch(batch)
        except Exception as e:  # noqa: BLE001 — engine errors are diverse, log and continue
            logger.exception("batch starting at %d failed", i)
            for rec in batch:
                si = ScoredImage(
                    record=rec,
                    processing_status=ProcessingStatus.FAILED,
                    error=f"{type(e).__name__}: {e}",
                )
                store.add_or_update(si, persist=False)
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
                si = ScoredImage(
                    record=rec,
                    scores=[o.score],
                    embeddings=[o.embedding],
                )
                si.processing_status = si.derive_processing_status()
            store.add_or_update(si, persist=False)
        logger.info("processed %d / %d", min(i + args.batch_size, len(todo)), len(todo))

    # 4. One atomic flush at the end (we deferred per-batch writes for speed).
    store.bulk_add([])  # noop, but forces a flush
    logger.info("wrote %s (%d items total)", args.manifest, len(store))


if __name__ == "__main__":
    main()
