import torch
from torch.optim import Optimizer

from .utils import safe_norm, soft_thresholding


class ADMM_Adam_Layer(Optimizer):
    """Layer-wise ADMM pruning with Wanda scores."""

    def __init__(self, params, lr, N, C, vk, wk, yk, zk, score):
        self.lr = lr
        self.N = N
        self.C = C
        self.vk = vk
        self.wk = wk
        self.yk = yk
        self.zk = zk
        self.score = score
        super(ADMM_Adam_Layer, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        p_scale = 1.0 / self.lr
        for group in self.param_groups:
            for w, vk_temp, yk_temp, zk_temp, wk_temp, score_temp in zip(
                group["params"], self.vk, self.yk, self.zk, self.wk, self.score
            ):
                grad = w.grad
                if grad is None:
                    continue
                score_safe = score_temp + 1e-8

                qk = 0.5 * (yk_temp + zk_temp - vk_temp / p_scale - wk_temp / p_scale) / score_safe
                qk -= grad / (score_safe * p_scale * 2.0)

                ck = torch.norm(score_safe * zk_temp, p=1)
                dk = score_safe * qk + vk_temp / p_scale
                yita = safe_norm(dk)
                miu = self.C * ck / self.N
                D_k = (miu * torch.mul(score_safe, score_safe)) / (
                    p_scale * torch.clamp(yita**3, min=1e-10)
                )
                C_K = ((27 * D_k + 2 + torch.sqrt(torch.clamp((27 * D_k + 2) ** 2 - 4, min=0.0))) / 2) ** (
                    1 / 3
                )
                tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

                if torch.all(dk == 0):
                    fangsuo = (ck / p_scale) ** (1 / 3)
                    random_tensor = torch.randn_like(yk_temp)
                    yk_temp.copy_(random_tensor * (fangsuo / safe_norm(random_tensor)))
                else:
                    yk_temp.copy_(tao_k * dk)

                # Use weight-magnitude-based threshold for stability
                weight_scale = safe_norm(w) + 1e-8
                # Scale threshold by C (sparsity control) and inversely by weight magnitude
                base_thresh = self.C * 0.001  # C controls pruning strength (smaller for per-layer)
                thresh = torch.clamp(base_thresh / weight_scale, min=1e-6, max=0.1)
                
                update_val = score_safe * qk + wk_temp / p_scale
                zk_temp.copy_(soft_thresholding(update_val, thresh))

                vk_temp.add_(p_scale * (score_safe * qk - yk_temp))
                wk_temp.add_(p_scale * (score_safe * qk - zk_temp))
                w.copy_(zk_temp)

        return loss