"""Neuron-wise Lasso optimizer (fixed version).

Includes the stabilized implementation directly so no companion file is
required. Thresholds are clamped and scaled by scores to avoid over-pruning.
"""

import torch
from torch.optim import Optimizer


def soft_thresholding(b, u):
    """Elementwise soft-thresholding."""
    return torch.sign(b) * torch.max(torch.zeros_like(b), torch.abs(b) - u)


class Lasso_neuron(Optimizer):
    def __init__(self, params, lr, N, C, vk, zk, score):
        self.lr = lr
        self.N = N
        self.C = C
        self.vk = vk
        self.zk = zk
        self.score = score
        super().__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for w, vk_temp, zk_temp, wanda_score_1 in zip(group["params"], self.vk, self.zk, self.score):
                w_len = len(w.shape)

                if w_len == 4:
                    wa, za = self.cnn_neuronwise_pruning(w, zk_temp, self.lr, self.N, self.C, wanda_score_1)
                elif w_len == 2:
                    wa, za = self.fullycont(w, zk_temp, self.lr, self.N, self.C, wanda_score_1)
                elif w_len == 1:
                    wa, za = self.batchnorm_and_bias_pruning(w, zk_temp, self.lr, self.N, self.C, wanda_score_1)
                else:
                    continue

                w.copy_(wa)
                zk_temp.copy_(za)

        return loss

    def cnn_neuronwise_pruning(self, w, zk_temp, lr, N, C, wanda_score_1):
        out_channels = w.shape[0]
        v = w - lr * w.grad

        # Element-wise soft threshold (L1)
        base_threshold = torch.clamp(torch.tensor(lr * C / N, device=w.device), min=1e-6, max=0.001)
        score_scale = torch.clamp(torch.abs(wanda_score_1), min=1e-3, max=10.0)
        # Lower scores -> higher threshold (prune more), higher scores -> lower threshold (keep)
        score_adjusted_threshold = base_threshold / score_scale

        tilde_w = soft_thresholding(v, score_adjusted_threshold)
        
        # Group-wise shrinkage (L2) - much gentler
        weighted_tilde_w = tilde_w.view(out_channels, -1)
        norm_group = torch.norm(weighted_tilde_w, p=2, dim=1)

        # Use a much smaller group threshold
        group_threshold = base_threshold * 0.1
        norm_group_exp = norm_group.view(out_channels, 1, 1, 1).expand_as(w)
        scale = torch.where(
            norm_group_exp > group_threshold,
            1 - torch.clamp(group_threshold / norm_group_exp, max=0.5),
            torch.ones_like(norm_group_exp) * 0.5,  # Don't zero out completely
        )

        new_w = scale * tilde_w
        zk_temp.copy_(new_w)
        w.copy_(zk_temp)
        return w, zk_temp

    def fullycont(self, w, zk_temp, lr, N, C, wanda_score_1):
        v = w - lr * w.grad

        # Element-wise soft threshold
        base_threshold = torch.clamp(torch.tensor(lr * C / N, device=w.device), min=1e-6, max=0.001)
        score_scale = torch.clamp(torch.abs(wanda_score_1), min=1e-3, max=10.0)
        score_adjusted_threshold = base_threshold / score_scale

        tilde_w = soft_thresholding(v, score_adjusted_threshold)
        
        # Group-wise shrinkage - gentler
        norm_group = torch.norm(tilde_w, p=2, dim=1)
        norm_group_exp = norm_group.unsqueeze(1).expand_as(w)

        group_threshold = base_threshold * 0.1
        scale = torch.where(
            norm_group_exp > group_threshold,
            1 - torch.clamp(group_threshold / norm_group_exp, max=0.5),
            torch.ones_like(norm_group_exp) * 0.5,
        )

        new_w = scale * tilde_w
        zk_temp.copy_(new_w)
        w.copy_(zk_temp)
        return w, zk_temp

    def batchnorm_and_bias_pruning(self, w, zk_temp, lr, N, C, wanda_score_1):
        v = w - lr * w.grad

        # Element-wise soft threshold
        base_threshold = torch.clamp(torch.tensor(lr * C / N, device=w.device), min=1e-6, max=0.001)
        score_scale = torch.clamp(torch.abs(wanda_score_1), min=1e-3, max=10.0)
        score_adjusted_threshold = base_threshold / score_scale

        tilde_w = soft_thresholding(v, score_adjusted_threshold)
        
        # Group-wise shrinkage - very gentle for bias/BN
        norm_group = torch.norm(tilde_w, p=2)
        group_threshold = base_threshold * 0.05

        if norm_group <= group_threshold:
            new_w = tilde_w * 0.5  # Don't zero completely
        else:
            scale = 1 - torch.clamp(group_threshold / norm_group, max=0.3)
            new_w = scale * tilde_w

        zk_temp.copy_(new_w)
        w.copy_(zk_temp)
        return w, zk_temp


__all__ = ["Lasso_neuron", "soft_thresholding"]