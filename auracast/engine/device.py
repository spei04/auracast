"""
Device + dtype selection.

A100s get FP16 mixed-precision. H100s/Hopper get BF16 by default (better
numerics, no autoscaler needed). Anything else (CPU / MPS dev box) gets FP32
and a warning that throughput will be terrible.

All engine modules go through `pick_device_and_dtype()` so the policy lives
in one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceSpec:
    device: torch.device
    dtype: torch.dtype
    name: str          # human-readable GPU name or "cpu" / "mps"
    is_a100: bool


def _is_a100(name: str) -> bool:
    return "A100" in name.upper()


def _is_hopper(name: str) -> bool:
    return "H100" in name.upper() or "H200" in name.upper()


def pick_device_and_dtype() -> DeviceSpec:
    """Choose the right (device, dtype) for the host."""
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        is_a100 = _is_a100(name)
        if _is_hopper(name):
            dtype = torch.bfloat16
        elif is_a100:
            dtype = torch.float16
        else:
            # Older / consumer GPUs (3090, 4090, T4, L4, ...). FP16 is usually safe.
            dtype = torch.float16
        spec = DeviceSpec(device=torch.device("cuda", idx), dtype=dtype, name=name, is_a100=is_a100)
        logger.info("CUDA detected: %s — using %s", name, dtype)
        return spec

    if torch.backends.mps.is_available():
        logger.warning("MPS detected — FP32, slow. For real runs use CUDA.")
        return DeviceSpec(
            device=torch.device("mps"), dtype=torch.float32, name="mps", is_a100=False
        )

    logger.warning("No GPU detected — running on CPU. Acceptable only for unit tests.")
    return DeviceSpec(device=torch.device("cpu"), dtype=torch.float32, name="cpu", is_a100=False)
