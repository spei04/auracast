"""
End-to-end mock pipeline demo. Local dir of images -> ImageRecords ->
CLIP scoring + embedding -> Manifest written as JSONL.

    python -m auracast.scripts.mock_pipeline_demo data/mock_images
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from auracast.engine.clip_scorer import CLIPScorer
from auracast.ingest.local_pipeline import LocalDirectoryIngest
from auracast.schema.models import Manifest, ScoredImage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mock_pipeline_demo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Directory of images to ingest.")
    parser.add_argument(
        "--manifest-out", type=Path, default=Path("manifests/latest.jsonl"),
        help="Where to write the JSONL Manifest."
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    ingest = LocalDirectoryIngest(args.root)
    records = ingest.collect()
    logger.info("ingested %d image records from %s", len(records), args.root)
    if not records:
        logger.warning("no images found — nothing to do.")
        return

    scorer = CLIPScorer()
    scored: list[ScoredImage] = []
    for i in range(0, len(records), args.batch_size):
        batch = records[i:i + args.batch_size]
        outputs = scorer.score_batch(batch)
        # Index outputs by image_id so we keep one ScoredImage per record.
        by_id = {o.score.image_id: o for o in outputs}
        for rec in batch:
            o = by_id.get(rec.image_id)
            scored.append(ScoredImage(
                record=rec,
                scores=[o.score] if o else [],
                embeddings=[o.embedding] if o else [],
            ))
        logger.info("scored %d / %d", min(i + args.batch_size, len(records)), len(records))

    manifest = Manifest(items=scored, notes=f"mock pipeline demo over {args.root}")
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(manifest.to_jsonl())
    logger.info("wrote %s (%d items)", args.manifest_out, len(scored))


if __name__ == "__main__":
    main()
