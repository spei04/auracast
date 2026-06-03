"""
Qwen2-VL A100 verification.

Loads the real model under FP16/BF16, captions a small batch of synthetic
images, asserts the output shape + that we got valid JSON (or at least a
non-empty fallback caption). Logs latency + peak VRAM.

Run locally (will fail on CPU/Mac — too big):
    python -m auracast.scripts.verify_qwen2vl_a100

On the cluster:
    sbatch scripts/slurm/verify_qwen2vl.sh
"""

from __future__ import annotations

import argparse
import logging
import time

import torch
from PIL import Image

from auracast.engine.device import pick_device_and_dtype
from auracast.engine.qwen2vl_describer import DEFAULT_QWEN_MODEL, Qwen2VLDescriber
from auracast.schema.models import ImageRecord, IngestSourceKind

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("verify_qwen2vl_a100")


def _make_synthetic_records(n: int, tmp_dir) -> list[ImageRecord]:
    import random
    records: list[ImageRecord] = []
    for i in range(n):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        path = tmp_dir / f"synthetic_{i:03d}.png"
        # 448x448 is comfortable for Qwen2-VL's variable resolution.
        Image.new("RGB", (448, 448), color).save(path)
        records.append(ImageRecord(
            source=IngestSourceKind.LOCAL_DIRECTORY,
            file_path=path,
            width=448,
            height=448,
        ))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL,
                        help="HF model id (e.g. Qwen/Qwen2-VL-2B-Instruct for fast test).")
    parser.add_argument("--n-images", type=int, default=2)
    args = parser.parse_args()

    spec = pick_device_and_dtype()
    logger.info("=" * 60)
    logger.info("[verify] device=%s dtype=%s name=%s is_a100=%s",
                spec.device, spec.dtype, spec.name, spec.is_a100)
    logger.info("[verify] model=%s", args.model)

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen2-VL verification requires a CUDA GPU. Submit via sbatch.")

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        records = _make_synthetic_records(args.n_images, tmp)

        t0 = time.time()
        describer = Qwen2VLDescriber(model_id=args.model, device_spec=spec)
        describer._ensure_model()
        t_load = time.time() - t0
        logger.info("[verify] model load took %.2fs", t_load)

        t0 = time.time()
        captions = describer.describe(records)
        t_caption = time.time() - t0

    assert len(captions) == args.n_images, f"got {len(captions)} captions, expected {args.n_images}"
    for c in captions:
        assert c.caption, f"empty caption for {c.image_id}"
        logger.info("[verify]   %s  caption=%r  attrs=%s",
                    c.image_id.hex[:8], c.caption[:80], c.attributes)

    logger.info("[verify] captioned %d images in %.2fs (%.2f s/img)",
                args.n_images, t_caption, t_caption / max(args.n_images, 1))

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        logger.info("[verify] peak VRAM = %.1f MB", peak_mb)

    logger.info("[verify] OK")


if __name__ == "__main__":
    main()
