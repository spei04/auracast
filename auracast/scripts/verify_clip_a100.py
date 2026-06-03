"""
A100 / GPU verification: load a lightweight CLIP model under FP16, run a
small batch of synthetic images through it, and assert the output is sane.

Use it both as a smoke test (does the env actually work on this node?) and
as a quick microbenchmark (per-image latency, peak VRAM).

Run locally:
    python -m auracast.scripts.verify_clip_a100

Run on the Beery cluster:
    sbatch scripts/slurm/verify_a100.sh
"""

from __future__ import annotations

import logging
import time

import torch
from PIL import Image

from auracast.engine.clip_scorer import DEFAULT_MODEL, CLIPScorer
from auracast.engine.device import pick_device_and_dtype
from auracast.schema.models import ImageRecord, IngestSourceKind

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("verify_clip_a100")


def _make_synthetic_records(n: int, tmp_dir) -> list[ImageRecord]:
    """Write `n` solid-color PNGs to tmp_dir and return ImageRecords pointing at them."""
    import random

    records: list[ImageRecord] = []
    for i in range(n):
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        path = tmp_dir / f"synthetic_{i:03d}.png"
        Image.new("RGB", (224, 224), color).save(path)
        records.append(ImageRecord(
            source=IngestSourceKind.LOCAL_DIRECTORY,
            file_path=path,
            width=224,
            height=224,
        ))
    return records


def main(n_images: int = 16) -> None:
    spec = pick_device_and_dtype()
    logger.info("=" * 60)
    logger.info("[verify] device=%s  dtype=%s  name=%s  is_a100=%s",
                spec.device, spec.dtype, spec.name, spec.is_a100)
    logger.info("[verify] torch=%s  cuda_available=%s",
                torch.__version__, torch.cuda.is_available())

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        records = _make_synthetic_records(n_images, tmp)
        logger.info("[verify] generated %d synthetic 224x224 images at %s", n_images, tmp)

        t0 = time.time()
        scorer = CLIPScorer(model_id=DEFAULT_MODEL, device_spec=spec)
        t_load = time.time() - t0
        logger.info("[verify] model load took %.2fs", t_load)

        t0 = time.time()
        outputs = scorer.score_batch(records)
        t_score = time.time() - t0

    # --- Assertions on shape, dtype, value ranges ---
    assert len(outputs) == n_images, f"got {len(outputs)} outputs, expected {n_images}"
    for out in outputs:
        assert 0.0 <= out.score.score <= 1.0, f"score out of [0,1]: {out.score.score}"
        assert out.embedding.dim == len(out.embedding.vector), \
            f"embedding dim mismatch: {out.embedding.dim} vs {len(out.embedding.vector)}"
        # Embedding should be L2-normalized; norm ~= 1.0
        norm = sum(v * v for v in out.embedding.vector) ** 0.5
        assert abs(norm - 1.0) < 1e-2, f"embedding not L2-normalized: |v| = {norm:.4f}"

    embedding_dim = outputs[0].embedding.dim
    mean_score = sum(o.score.score for o in outputs) / len(outputs)

    logger.info("[verify] scored %d images in %.2fs (%.1f img/s)",
                n_images, t_score, n_images / max(t_score, 1e-9))
    logger.info("[verify] embedding dim = %d", embedding_dim)
    logger.info("[verify] mean aesthetic score = %.4f", mean_score)

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        logger.info("[verify] peak VRAM = %.1f MB", peak_mb)

    logger.info("[verify] OK")


if __name__ == "__main__":
    main()
