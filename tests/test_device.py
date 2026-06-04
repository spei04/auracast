"""Tests for engine.device — CPU-only, never asserts on GPU presence."""

from __future__ import annotations

import torch

from auracast.engine.device import DeviceSpec, pick_device_and_dtype


def test_pick_returns_device_spec():
    spec = pick_device_and_dtype()
    assert isinstance(spec, DeviceSpec)
    assert isinstance(spec.device, torch.device)
    assert isinstance(spec.dtype, torch.dtype)
    assert spec.name


def test_dtype_is_one_of_known():
    spec = pick_device_and_dtype()
    assert spec.dtype in (torch.float16, torch.float32)


def test_name_is_nonempty():
    spec = pick_device_and_dtype()
    assert isinstance(spec.name, str) and spec.name
