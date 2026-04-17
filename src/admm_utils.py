# -*- coding: utf-8 -*-
"""
ADMM utility functions for feature selection.

This module contains mathematical utilities used by the ADMM
feature-selection wrappers:
  - soft_thresholding: proximal operator for L1 norm
  - safe_norm: numerical stable L2 norm
  - safe_cbrt: numerical stable cube root
  - solve_cubic_ratio_norm: Cardano's formula for Ratio Norm proximal
  - solve_cubic_paper: wrapper for paper's cubic equation
"""

import torch


def soft_thresholding(v: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Soft thresholding operator: sign(v) * max(|v| - alpha, 0).

    This is the proximal operator for the L1 norm:
        prox_{alpha * ||.||_1}(v) = sign(v) * max(|v| - alpha, 0)

    Args:
        v: Input tensor
        alpha: Threshold (same shape as v or broadcastable)

    Returns:
        Soft-thresholded tensor
    """
    return torch.sign(v) * torch.clamp(torch.abs(v) - alpha, min=0.0)


def safe_norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safe L2 norm with epsilon for numerical stability.

    Args:
        v: Input tensor
        eps: Small constant for numerical stability

    Returns:
        L2 norm of v plus epsilon
    """
    return torch.norm(v, p=2) + eps


def safe_cbrt(x: torch.Tensor) -> torch.Tensor:
    """Safe cube root that handles negative numbers.

    For x < 0: cbrt(x) = -cbrt(|x|)
    For x >= 0: cbrt(x) = x^(1/3)

    Args:
        x: Input tensor

    Returns:
        Element-wise cube root
    """
    return torch.sign(x) * torch.abs(x) ** (1.0 / 3.0)


def solve_cubic_ratio_norm(D: torch.Tensor) -> torch.Tensor:
    """Solve tau^3 - tau - D = 0 for tau >= 0 using Cardano's formula.

    For the equation tau^3 - tau - D = 0:
    - Discriminant: Delta = D^2/4 - 1/27
    - If Delta >= 0: one real root tau = cbrt(D/2 + sqrt(Delta)) + cbrt(D/2 - sqrt(Delta))
    - If Delta < 0: three real roots, use trigonometric method

    This is the proximal operator for the Ratio Norm R(z) = ||z||_1 / ||z||_2.

    Args:
        D: Input tensor (should be non-negative for Ratio Norm)

    Returns:
        tau: Positive real root of the cubic equation

    References:
        Cardano's formula for cubic equations. See Boyd & Vandenberghe
        "Convex Optimization" for proximal operators.
    """
    tau = torch.zeros_like(D)

    # Discriminant for tau^3 - tau - D = 0: Delta = D^2/4 - 1/27
    discriminant = (D / 2.0) ** 2 - 1.0 / 27.0

    # Case 1: discriminant >= 0 (one real root)
    case1 = discriminant >= 0
    if torch.any(case1):
        sqrt_disc = torch.sqrt(discriminant[case1])
        # tau = cbrt(D/2 + sqrt(Delta)) + cbrt(D/2 - sqrt(Delta))
        # FIX: Use safe_cbrt to handle negative values
        term1 = D[case1] / 2.0 + sqrt_disc
        term2 = D[case1] / 2.0 - sqrt_disc  # FIX: was -D/2 + sqrt_disc (wrong sign)
        tau[case1] = safe_cbrt(term1) + safe_cbrt(term2)

    # Case 2: discriminant < 0 (three real roots, use trigonometric method)
    case2 = ~case1
    if torch.any(case2):
        # For tau^3 - tau - D = 0, the trigonometric solution is:
        # tau = 2/sqrt(3) * cos(theta/3) where theta = arccos(3*sqrt(3)*D / 2)
        sqrt_3 = torch.sqrt(torch.tensor(3.0, device=D.device))
        arg = torch.clamp(3.0 * sqrt_3 * D[case2] / 2.0, min=-1.0, max=1.0)
        theta = torch.acos(arg)
        tau[case2] = 2.0 * torch.cos(theta / 3.0) / sqrt_3

    return tau


def solve_cubic_paper(D: torch.Tensor) -> torch.Tensor:
    """Solve cubic equation from the paper formulation.

    This uses the same Cardano's formula as solve_cubic_ratio_norm,
    adapted for the paper's specific cubic equation format.

    Args:
        D: Input tensor

    Returns:
        Positive real root of the cubic equation
    """
    return solve_cubic_ratio_norm(D)
