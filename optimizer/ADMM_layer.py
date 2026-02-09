import torch
import numpy as np
from torch.optim import Optimizer

from .utils import safe_norm, soft_thresholding, solve_cubic_ratio_norm, safe_cbrt


class ADMM_Adam_layer(Optimizer):
    """Layer-wise ADMM pruning with Wanda scores.

    Each layer is treated independently for the Ratio Norm regularization.
    This provides finer-grained control than global pruning while being
    more efficient than neuron-wise pruning.

    The update follows the same ADMM formulation as global, but applied per-layer:

        q_k = 0.5 * (y_k + z_k - v_k/p - w_k/p)/score - grad/(score * p * 2)
        y_k <- τ * (score * q_k + v_k/p)    where τ solves τ³ - τ - D_k = 0
        z_k <- soft_threshold(score * q_k + w_k/p, threshold)
        v_k <- v_k + p * (score*q_k - y_k)
        w_k <- w_k + p * (score*q_k - z_k)

    where p = 1/lr (penalty parameter).
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
        super(ADMM_Adam_layer, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Use mutable rho if set externally (adaptive rho), else default 1/lr
        if not hasattr(self, 'rho'):
            self.rho = 1.0 / self.lr
        p_scale = self.rho

        r_norms = []
        s_norms = []
        for group in self.param_groups:
            for w, vk_temp, yk_temp, zk_temp, wk_temp, score_temp in zip(
                group["params"], self.vk, self.yk, self.zk, self.wk, self.score
            ):
                grad = w.grad
                if grad is None:
                    continue
                score_safe = score_temp + 1e-8

                # Save z_old for dual residual
                zk_old = zk_temp.clone()

                # q_k update: combines primal variables and gradient
                qk = 0.5 * (yk_temp + zk_temp - vk_temp / p_scale - wk_temp / p_scale) / score_safe
                qk -= grad / (score_safe * p_scale * 2.0)

                # y_k update: solve cubic equation for Ratio Norm proximal
                dk = score_safe * qk + vk_temp / p_scale
                ck = torch.norm(score_safe * zk_temp, p=1)
                yita = safe_norm(dk)
                miu = self.C * ck / self.N

                # D_k for cubic solver: τ³ - τ - D_k = 0
                D_k = (miu * score_safe * score_safe) / (p_scale * torch.clamp(yita ** 3, min=1e-10))
                tao_k = solve_cubic_ratio_norm(D_k)

                # Handle edge case when dk ≈ 0
                if torch.allclose(dk, torch.zeros_like(dk)):
                    fangsuo = safe_cbrt(ck / p_scale)
                    random_tensor = torch.randn_like(yk_temp)
                    yk_temp.copy_(random_tensor * (fangsuo / safe_norm(random_tensor)))
                else:
                    yk_temp.copy_(tao_k * dk)

                # z_k update: soft-thresholding for sparsity
                # Heuristic threshold: lr * C * 0.001 / score  
                # Note: mathematically C/(N*||y||_2*rho) but current C values
                # are tuned for this heuristic.
                base_thresh = self.lr * self.C * 0.001
                thresh = base_thresh / score_safe
                thresh = torch.clamp(thresh, min=1e-6, max=0.1)

                update_val = score_safe * qk + wk_temp / p_scale
                zk_temp.copy_(soft_thresholding(update_val, thresh))

                # Dual variable updates
                vk_temp.add_(p_scale * (score_safe * qk - yk_temp))
                wk_temp.add_(p_scale * (score_safe * qk - zk_temp))

                # Update weights to pruned values
                w.copy_(zk_temp)

                # Track per-layer residuals
                r_norms.append(torch.norm(score_safe * qk - zk_temp).item())
                s_norms.append(p_scale * torch.norm(zk_temp - zk_old).item())

        # Store averaged residuals for adaptive rho
        self.r_norm = float(np.mean(r_norms)) if r_norms else 0.0
        self.s_norm = float(np.mean(s_norms)) if s_norms else 0.0

        return loss