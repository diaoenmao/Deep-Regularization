"""
Shared optimizer helpers for pruning methods.
"""
from __future__ import annotations

import torch
from torch import Tensor


def soft_thresholding(b: Tensor, u: Tensor) -> Tensor:
    """Element-wise soft-thresholding.

    Args:
        b: Input tensor.
        u: Non-negative threshold (can be broadcast).
    Returns:
        Tensor after applying sign(b) * max(|b|-u, 0).
    """
    return torch.sign(b) * torch.clamp(torch.abs(b) - u, min=0.0)


def safe_norm(x: Tensor, p: float = 2.0, dim=None, keepdim: bool = False, eps: float = 1e-8) -> Tensor:
    """Numerically stable p-norm used in ADMM updates."""
    return torch.norm(x, p=p, dim=dim, keepdim=keepdim) + eps
