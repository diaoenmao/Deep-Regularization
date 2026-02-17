"""Input-group ADMM pruning optimizer for feature selection.

This optimizer applies ADMM with **column-wise (input-feature) group
sparsity** on the first Linear layer of an MLP.  Instead of driving
individual weights to zero (element-wise soft-thresholding), it drives
entire *columns* of W₁ to zero via group soft-thresholding:

    z_{:,j} = v_{:,j} · max(0, 1 − λ / ‖v_{:,j}‖₂)

This means feature j is either fully kept or fully removed — exactly
the inductive bias needed for feature selection on nonlinear problems
like XOR, where element-wise sparsity scatters zeros uniformly and
fails to identify informative features.

Usage:
    This optimizer is meant to be applied **only to the first Linear
    layer** of the MLP.  Hidden layers should be trained with a
    standard optimizer (e.g. Adam with mild L2) so that nonlinear
    representation capacity is preserved.
"""

import torch
import numpy as np
from torch.optim import Optimizer

from .utils import (
    safe_norm,
    soft_thresholding,
    group_soft_thresholding_column,
    solve_cubic_ratio_norm,
    safe_cbrt,
)


class ADMM_Input_Group(Optimizer):
    """ADMM optimizer that induces column-wise group sparsity.

    Designed for the first Linear layer of an MLP used for feature
    selection.  The ADMM q/y updates follow the standard Ratio-Norm
    formulation; the key difference is the **z-step**, which uses
    ``group_soft_thresholding_column`` instead of element-wise
    ``soft_thresholding``.

    For 1-D parameters (bias), element-wise soft-thresholding is kept
    as a fallback.

    Parameters
    ----------
    params : iterable
        Parameters of the *first Linear layer only* (weight + bias).
    lr, N, C : float
        Learning rate, dataset size, sparsity coefficient.
    vk, wk, yk, zk : list[Tensor]
        ADMM dual / auxiliary buffers (same shapes as params).
    score : list[Tensor]
        WANDA importance scores aligned with params.
    """

    def __init__(self, params, lr, N, C, vk, wk, yk, zk, score):
        self.lr = lr
        self.N = N
        self.C = C
        self.vk = vk
        self.wk = wk
        self.yk = yk
        self.zk = zk
        self.score = score
        super().__init__(params, {})

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not hasattr(self, "rho"):
            self.rho = 1.0 / self.lr
        p_scale = self.rho

        r_norms: list[float] = []
        s_norms: list[float] = []

        for group in self.param_groups:
            for w, vk, yk, zk, wk, score in zip(
                group["params"], self.vk, self.yk, self.zk, self.wk, self.score
            ):
                grad = w.grad
                if grad is None:
                    continue

                zk_old = zk.clone()

                if w.dim() == 2:
                    # --- 2-D weight: column-wise group ADMM ---
                    w.data, vk.data, yk.data, zk.data, wk.data = (
                        self._linear_column_group_update(
                            w, vk, yk, zk, wk, grad, score, p_scale
                        )
                    )
                elif w.dim() == 1:
                    # --- 1-D bias: light element-wise ADMM ---
                    w.data, vk.data, yk.data, zk.data, wk.data = (
                        self._vector_update(
                            w, vk, yk, zk, wk, grad, score, p_scale
                        )
                    )

                r_norms.append(torch.norm(w.data - zk_old).item())
                s_norms.append(p_scale * torch.norm(zk.data - zk_old).item())

        self.r_norm = float(np.mean(r_norms)) if r_norms else 0.0
        self.s_norm = float(np.mean(s_norms)) if s_norms else 0.0
        return loss

    # ------------------------------------------------------------------
    # Core: column-group z-step
    # ------------------------------------------------------------------
    def _linear_column_group_update(self, w, vk, yk, zk, wk, grad, score, p_scale):
        """ADMM update for a 2-D weight with *column*-group sparsity.

        The q-step and y-step mirror ADMM_neuron (per-output-neuron
        Ratio Norm), but the z-step uses ``group_soft_thresholding_column``
        so that entire *input features* (columns) are zeroed together.
        """
        out_features = w.shape[0]
        score_safe = score + 1e-8

        # ---- q-update (same as neuron) ----
        qk = 0.5 * (yk + zk - vk / p_scale - wk / p_scale) / score_safe
        qk -= grad / (score_safe * p_scale * 2.0)

        # ---- y-update (per-output-neuron Ratio Norm, same as neuron) ----
        ck = torch.norm(score_safe * zk, p=1, dim=1, keepdim=True).expand_as(w)
        dk = score_safe * qk + vk / p_scale
        yita = torch.norm(dk, p=2, dim=1, keepdim=True) + 1e-8
        yita = yita.expand_as(w)

        miu = self.C * ck / self.N
        D_k = (miu * score_safe * score_safe) / (p_scale * torch.clamp(yita ** 3, min=1e-10))
        tao_k = solve_cubic_ratio_norm(D_k)

        if torch.allclose(dk, torch.zeros_like(dk)):
            fangsuo = safe_cbrt(ck / p_scale)
            rand = torch.randn_like(yk)
            norm_n = torch.norm(rand, p=2, dim=1, keepdim=True) + 1e-8
            yk = rand * (fangsuo / norm_n.expand_as(w))
        else:
            yk = tao_k * dk

        # ---- z-update: COLUMN-GROUP soft-thresholding ----
        update_val = score_safe * qk + wk / p_scale
        # The group threshold controls how aggressively columns shrink.
        # Using C directly (not C/(N*rho)) so that the per-step shrinkage
        # factor  max(0, 1 - thresh/||col||)  is meaningful relative to
        # the column norms (~1.0 for a warmed-up MLP).
        group_thresh = self.C
        zk = group_soft_thresholding_column(update_val, group_thresh)

        # ---- dual updates ----
        vk = vk + p_scale * (score_safe * qk - yk)
        wk = wk + p_scale * (score_safe * qk - zk)

        # weights ← z (ADMM consensus)
        w = zk
        return w, vk, yk, zk, wk

    # ------------------------------------------------------------------
    def _vector_update(self, w, vk, yk, zk, wk, grad, score, p_scale):
        """Element-wise ADMM for 1-D params (bias)."""
        score_safe = score + 1e-8

        qk = 0.5 * (yk + zk - vk / p_scale - wk / p_scale) / score_safe
        qk -= grad / (score_safe * p_scale * 2.0)

        ck = torch.norm(score_safe * zk, p=1)
        dk = score_safe * qk + vk / p_scale
        yita = safe_norm(dk)

        miu = self.C * ck / self.N
        D_k = (miu * score_safe * score_safe) / (p_scale * torch.clamp(yita ** 3, min=1e-10))
        tao_k = solve_cubic_ratio_norm(D_k)

        if torch.allclose(dk, torch.zeros_like(dk)):
            fangsuo = safe_cbrt(ck / p_scale)
            rand = torch.randn_like(yk)
            yk = rand * (fangsuo / safe_norm(rand))
        else:
            yk = tao_k * dk

        # Bias gets a very mild element-wise threshold
        base_thresh = self.lr * self.C * 0.0001
        thresh = base_thresh / score_safe
        thresh = torch.clamp(thresh, min=1e-6, max=0.1)
        update_val = score_safe * qk + wk / p_scale
        zk = soft_thresholding(update_val, thresh)

        vk = vk + p_scale * (score_safe * qk - yk)
        wk = wk + p_scale * (score_safe * qk - zk)
        w = zk
        return w, vk, yk, zk, wk
