
import torch
from torch.optim import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .utils import safe_norm, soft_thresholding


class ADMM_Adam_global(Optimizer):
    """Global ADMM pruning step with Wanda scaling.

    The update follows a scaled ADMM formulation:

        q_k = 0.5 * (y_k + z_k - v_k/p - w_k/p)/score - grad/(score * p * 2)
        y_k <- lambda * (score * q_k + v_k/p)
        z_k <- soft_threshold(score * q_k + w_k/p, (C/N)/(p * ||y_k||_2))
        v_k <- v_k + p * (score*q_k - y_k)
        w_k <- w_k + p * (score*q_k - z_k)

    where p = 1/lr.
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

        grad_vec = parameters_to_vector([p.grad for p in self.param_groups[0]["params"]])
        p_scale = 1.0 / self.lr

        qk = 0.5 * (yk_vec + zk_vec - vk_vec / p_scale - wk_vec / p_scale) / score_vec
        qk -= grad_vec / (score_vec * p_scale * 2.0)

        u = score_vec * qk + vk_vec / p_scale
        r = safe_norm(u)
        gamma = (self.C / self.N * torch.norm(zk_vec, p=1)) / (p_scale * torch.clamp(r**3, min=1e-10))
        delta = torch.sqrt(torch.clamp(((gamma + 2.0 / 27.0) ** 2) / 4.0 - 1.0 / 729.0, min=0.0))
        lam = 1.0 / 3.0 + torch.pow((gamma + 2.0 / 27.0) / 2.0 + delta, 1.0 / 3.0)
        lam += torch.pow((gamma + 2.0 / 27.0) / 2.0 - delta, 1.0 / 3.0)

        if torch.allclose(u, torch.zeros_like(u)):
            ck = torch.norm(zk_vec, p=1)
            fangsuo = (ck / p_scale) ** (1.0 / 3.0)
            random_tensor = torch.randn_like(yk_vec)
            yk_vec.copy_(random_tensor * (fangsuo / torch.norm(random_tensor, p=2)))
        else:
            yk_vec.copy_(lam * (score_vec * qk + vk_vec / p_scale))

        # Use weight-magnitude-based threshold for stability
        weight_scale = safe_norm(params_vec) + 1e-8
        # Scale threshold by C (sparsity control) and inversely by weight magnitude
        base_thresh = self.C * 0.01  # C controls pruning strength
        thresh = torch.clamp(base_thresh / weight_scale, min=1e-6, max=0.5)
        
        update_vec = score_vec * qk + wk_vec / p_scale
        zk_vec.copy_(soft_thresholding(update_vec, thresh))

        vk_vec.add_(p_scale * (score_vec * qk - yk_vec))
        wk_vec.add_(p_scale * (score_vec * qk - zk_vec))

        vector_to_parameters(zk_vec, self.param_groups[0]["params"])
        vector_to_parameters(vk_vec, self.vk)
        vector_to_parameters(wk_vec, self.wk)
        vector_to_parameters(yk_vec, self.yk)
        vector_to_parameters(zk_vec, self.zk)

        return None
