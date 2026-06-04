"""
Device + dtype selection.

Picks a (device, dtype) pair for the host:
  - CUDA available  -> FP16 on `cuda:0`.
  - Apple MPS        -> FP32 on `mps` (slow but works for dev).
  - Otherwise        -> FP32 on CPU (slow; for unit tests).

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
    name: str  # human-readable device name or "cpu" / "mps"


def pick_device_and_dtype() -> DeviceSpec:
    """Choose the right (device, dtype) for the host."""
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        spec = DeviceSpec(device=torch.device("cuda", idx), dtype=torch.float16, name=name)
        logger.info("CUDA detected: %s — using %s", name, spec.dtype)
        return spec

    if torch.backends.mps.is_available():
        logger.warning("MPS detected — FP32, slow. For real runs use CUDA.")
        return DeviceSpec(device=torch.device("mps"), dtype=torch.float32, name="mps")

    logger.warning("No GPU detected — running on CPU. Acceptable only for unit tests.")
    return DeviceSpec(device=torch.device("cpu"), dtype=torch.float32, name="cpu")
