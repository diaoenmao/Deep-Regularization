
import torch
from torch.optim import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .utils import safe_norm, soft_thresholding, solve_cubic_ratio_norm, safe_cbrt


class ADMM_Adam_global(Optimizer):
    """Global ADMM pruning step with Wanda scaling.

    The update follows a scaled ADMM formulation:

        q_k = 0.5 * (y_k + z_k - v_k/p - w_k/p)/score - grad/(score * p * 2)
        y_k <- τ * (score * q_k + v_k/p)    where τ solves τ³ - τ - D_k = 0
        z_k <- soft_threshold(score * q_k + w_k/p, threshold)
        v_k <- v_k + p * (score*q_k - y_k)
        w_k <- w_k + p * (score*q_k - z_k)

    where p = 1/lr (penalty parameter).

    The cubic equation arises from the proximal operator of the Ratio Norm (L1/L2).
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
        super(ADMM_Adam_global, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        score_vec = parameters_to_vector(self.score) + 1e-8
        params_vec = parameters_to_vector(self.param_groups[0]["params"])
        vk_vec = parameters_to_vector(self.vk)
        wk_vec = parameters_to_vector(self.wk)
        yk_vec = parameters_to_vector(self.yk)
        zk_vec = parameters_to_vector(self.zk)

        # Save z_old for dual residual computation
        zk_old = zk_vec.clone()

        grad_vec = parameters_to_vector([p.grad for p in self.param_groups[0]["params"]])
        # Use mutable rho if set externally (adaptive rho), else default 1/lr
        if not hasattr(self, 'rho'):
            self.rho = 1.0 / self.lr
        p_scale = self.rho

        # q_k update: combines primal variables and gradient
        qk = 0.5 * (yk_vec + zk_vec - vk_vec / p_scale - wk_vec / p_scale) / score_vec
        qk -= grad_vec / (score_vec * p_scale * 2.0)

        # y_k update: solve cubic equation for Ratio Norm proximal
        dk = score_vec * qk + vk_vec / p_scale
        ck = torch.norm(score_vec * zk_vec, p=1)
        yita = safe_norm(dk)
        miu = self.C * ck / self.N

        # D_k for cubic solver: τ³ - τ - D_k = 0
        D_k = (miu * score_vec * score_vec) / (p_scale * torch.clamp(yita ** 3, min=1e-10))
        tao_k = solve_cubic_ratio_norm(D_k)

        # Handle edge case when dk ≈ 0
        if torch.allclose(dk, torch.zeros_like(dk)):
            fangsuo = safe_cbrt(ck / p_scale)
            random_tensor = torch.randn_like(yk_vec)
            yk_vec.copy_(random_tensor * (fangsuo / safe_norm(random_tensor)))
        else:
            yk_vec.copy_(tao_k * dk)

        # z_k update: soft-thresholding for sparsity
        # Heuristic threshold: lr * C * 0.01 / score
        # Note: mathematically C/(N*||y||_2*rho) but current C values are tuned
        # for this heuristic. /score makes it score-adaptive (prune low-importance).
        base_thresh = self.lr * self.C * 0.01
        thresh = base_thresh / score_vec
        thresh = torch.clamp(thresh, min=1e-6, max=0.1)

        update_vec = score_vec * qk + wk_vec / p_scale
        zk_vec.copy_(soft_thresholding(update_vec, thresh))

        # Dual variable updates
        vk_vec.add_(p_scale * (score_vec * qk - yk_vec))
        wk_vec.add_(p_scale * (score_vec * qk - zk_vec))

        # Write back to parameter buffers
        vector_to_parameters(zk_vec, self.param_groups[0]["params"])
        vector_to_parameters(vk_vec, self.vk)
        vector_to_parameters(wk_vec, self.wk)
        vector_to_parameters(yk_vec, self.yk)
        vector_to_parameters(zk_vec, self.zk)

        # Track primal/dual residuals for adaptive rho (Boyd §3.4.1)
        self.r_norm = torch.norm(score_vec * qk - zk_vec).item()
        self.s_norm = p_scale * torch.norm(zk_vec - zk_old).item()

        return None
