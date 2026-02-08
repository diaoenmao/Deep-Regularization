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


def safe_cbrt(x: Tensor) -> Tensor:
    """Numerically stable cube root that handles negative values.

    Standard pow(x, 1/3) returns NaN for negative x in PyTorch.
    This function correctly computes sign(x) * |x|^(1/3).

    Args:
        x: Input tensor (can contain negative values).
    Returns:
        Cube root of x with proper sign handling.
    """
    return torch.sign(x) * torch.pow(torch.abs(x) + 1e-30, 1.0 / 3.0)


def solve_cubic_depressed(gamma: Tensor, eps: float = 1e-10) -> Tensor:
    """Solve the depressed cubic equation: t³ - t - γ = 0.

    This arises from the proximal operator of the Ratio Norm (L1/L2).
    Uses Cardano's formula with proper handling of all discriminant cases.

    The depressed cubic t³ + pt + q = 0 has p = -1, q = -γ.
    Discriminant Δ = -4p³ - 27q² = 4 - 27γ²

    Cases:
    - Δ > 0: Three distinct real roots (use trigonometric method)
    - Δ = 0: Multiple root (one or two distinct real roots)
    - Δ < 0: One real root (use Cardano's formula)

    For the Ratio Norm proximal, we want the largest positive real root.

    Args:
        gamma: The γ parameter tensor (can be any shape).
        eps: Small epsilon for numerical stability.

    Returns:
        The largest positive real root τ for each element.
    """
    # Discriminant: Δ = 4 - 27γ²
    gamma_sq = gamma * gamma
    discriminant = 4.0 - 27.0 * gamma_sq

    # Initialize output
    tau = torch.zeros_like(gamma)

    # Case 1: Δ < 0 (one real root) - Use Cardano's formula
    # This is the most common case in practice
    mask_one_root = discriminant < -eps

    if mask_one_root.any():
        g = gamma[mask_one_root]
        # For t³ - t - γ = 0, Cardano gives:
        # t = cbrt(γ/2 + sqrt(γ²/4 - 1/27)) + cbrt(γ/2 - sqrt(γ²/4 - 1/27))
        sqrt_term = torch.sqrt(torch.clamp(g * g / 4.0 - 1.0 / 27.0, min=0.0))
        u = safe_cbrt(g / 2.0 + sqrt_term)
        v = safe_cbrt(g / 2.0 - sqrt_term)
        tau[mask_one_root] = u + v

    # Case 2: Δ > 0 (three real roots) - Use trigonometric method
    # t_k = 2/√3 * cos(θ/3 - 2πk/3) where cos(3θ) = 3√3 γ / 2
    mask_three_roots = discriminant > eps

    if mask_three_roots.any():
        g = gamma[mask_three_roots]
        # cos(3θ) = 3√3 γ / 2, clamped to [-1, 1]
        cos_3theta = torch.clamp(3.0 * 1.7320508075688772 * g / 2.0, min=-1.0, max=1.0)
        theta = torch.acos(cos_3theta) / 3.0

        # Three roots: we want the largest positive one
        # t_0 = 2/√3 * cos(θ)
        # t_1 = 2/√3 * cos(θ - 2π/3)
        # t_2 = 2/√3 * cos(θ - 4π/3)
        scale = 2.0 / 1.7320508075688772  # 2/√3
        t0 = scale * torch.cos(theta)
        t1 = scale * torch.cos(theta - 2.0943951023931953)  # 2π/3
        t2 = scale * torch.cos(theta - 4.1887902047863905)  # 4π/3

        # Take the maximum (largest positive root)
        tau[mask_three_roots] = torch.max(torch.max(t0, t1), t2)

    # Case 3: Δ ≈ 0 (multiple root)
    # When γ ≈ ±2/(3√3), we have a double root
    mask_double = ~mask_one_root & ~mask_three_roots

    if mask_double.any():
        g = gamma[mask_double]
        # For small discriminant, use the single root formula as approximation
        sqrt_term = torch.sqrt(torch.clamp(g * g / 4.0 - 1.0 / 27.0 + eps, min=0.0))
        u = safe_cbrt(g / 2.0 + sqrt_term)
        v = safe_cbrt(g / 2.0 - sqrt_term)
        tau[mask_double] = u + v

    # Ensure τ ≥ 1 (from the Ratio Norm proximal constraint)
    # The proximal operator requires λ ≥ 1 for proper scaling
    tau = torch.clamp(tau, min=1.0)

    return tau


def solve_cubic_ratio_norm(D_k: Tensor, eps: float = 1e-10) -> Tensor:
    """Solve the cubic equation for the Ratio Norm y-update.

    The y-update in ADMM requires solving for τ (tao_k) where:
        τ³ - τ - D_k = 0

    This is equivalent to the depressed cubic with γ = D_k.

    The current codebase uses the formula:
        C_K = ((27*D_k + 2 + sqrt((27*D_k + 2)² - 4)) / 2)^(1/3)
        τ = 1/3 + (1/3)*(C_K + 1/C_K)

    This is a simplified form that works when D_k > 0 and the discriminant
    is negative (one real root case). However, it fails when:
    - D_k is very small (numerical instability in 1/C_K)
    - D_k is negative (can happen with certain score configurations)
    - The discriminant is positive (three real roots)

    This function provides a complete solution handling all cases.

    Args:
        D_k: The D_k parameter from the ADMM update (can be any shape).
        eps: Small epsilon for numerical stability.

    Returns:
        The scaling factor τ for the y-update.
    """
    return solve_cubic_depressed(D_k, eps=eps)


def compute_y_update_scale(
    score: Tensor,
    dk: Tensor,
    ck: Tensor,
    p_scale: float,
    C: float,
    N: int,
    eps: float = 1e-10,
) -> Tensor:
    """Compute the scaling factor τ for the y-update in ADMM.

    This encapsulates the full computation:
        η = ||score * dk||_2
        μ = C * ck / N
        D_k = (μ * score²) / (p * η³)
        τ = solve_cubic(D_k)

    Args:
        score: Wanda/importance scores (same shape as weights).
        dk: The d_k = q_k + v_k/p intermediate variable.
        ck: The c_k = ||score * z_k||_1 sparsity measure.
        p_scale: Penalty parameter (1/lr).
        C: Sparsity control hyperparameter.
        N: Dataset size for normalization.
        eps: Numerical stability epsilon.

    Returns:
        Scaling factor τ with same shape as score (broadcast-ready).
    """
    # Compute η = ||score * dk||_2
    yita = safe_norm(score * dk, eps=eps)

    # Compute μ = C * ck / N
    miu = C * ck / N

    # Compute D_k = (μ * score²) / (p * η³)
    yita_cubed = torch.clamp(yita ** 3, min=eps)
    D_k = (miu * score * score) / (p_scale * yita_cubed)

    # Solve cubic and return τ
    return solve_cubic_ratio_norm(D_k, eps=eps)
