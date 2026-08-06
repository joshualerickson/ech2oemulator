"""Device selection and tensor transfer helpers.

``auto`` makes the command portable: it uses the first CUDA GPU when PyTorch
can see one and otherwise preserves the established CPU behavior.
"""
from __future__ import annotations

import torch


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or a CUDA device and validate availability."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but this PyTorch installation cannot see a CUDA GPU")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"--device {requested} was requested, but only {torch.cuda.device_count()} CUDA GPU(s) are visible")
    return device


def to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move one tensor, using asynchronous copies where pinned input permits it."""
    return value.to(device, non_blocking=device.type == "cuda")
