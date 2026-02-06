"""Neuron-wise ADMM pruning optimizer.

This optimizer applies ADMM pruning at the neuron (output channel) level,
providing the finest granularity of control. Each neuron's weights are
treated as a group for the Ratio Norm regularization.
"""
import torch
from torch.optim import Optimizer

from .utils import safe_norm, soft_thresholding, solve_cubic_ratio_norm, safe_cbrt


class ADMM_Adam_neuron(Optimizer):
    """Neuron-wise ADMM pruning with Wanda scores.

    Each output neuron (channel) is treated independently for pruning.
    This provides the finest-grained control but is more computationally
    expensive than layer-wise or global pruning.

    The update follows the same ADMM formulation, but applied per-neuron:

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
        super(ADMM_Adam_neuron, self).__init__(params, {})

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

                w_dim = len(w.shape)

                if w_dim == 4:
                    # Conv2d layer: [out_channels, in_channels, kH, kW]
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = \
                        self._conv_neuronwise_update(
                            w, vk_temp, yk_temp, zk_temp, wk_temp, grad, score_temp, p_scale
                        )
                elif w_dim == 2:
                    # Linear layer: [out_features, in_features]
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = \
                        self._linear_neuronwise_update(
                            w, vk_temp, yk_temp, zk_temp, wk_temp, grad, score_temp, p_scale
                        )
                elif w_dim == 1:
                    # Bias or BatchNorm parameters: [num_features]
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = \
                        self._vector_update(
                            w, vk_temp, yk_temp, zk_temp, wk_temp, grad, score_temp, p_scale
                        )

        return loss

    def _conv_neuronwise_update(self, w, vk, yk, zk, wk, grad, score, p_scale):
        """ADMM update for Conv2d layers, per output channel (neuron)."""
        out_channels = w.shape[0]
        score_safe = score + 1e-8

        # q_k update (consistent with global/layer formulation)
        qk = 0.5 * (yk + zk - vk / p_scale - wk / p_scale) / score_safe
        qk -= grad / (score_safe * p_scale * 2.0)

        # Compute per-neuron norms for y_k update
        # ck = ||score * zk||_1 per neuron
        ck = torch.norm((score_safe * zk).view(out_channels, -1), p=1, dim=1)
        ck = ck.view(out_channels, 1, 1, 1).expand_as(w)

        # dk = score * qk + vk / p
        dk = score_safe * qk + vk / p_scale

        # yita = ||score * dk||_2 per neuron
        yita = torch.norm((score_safe * dk).view(out_channels, -1), p=2, dim=1) + 1e-8
        yita = yita.view(out_channels, 1, 1, 1).expand_as(w)

        # D_k for cubic solver
        miu = self.C * ck / self.N
        D_k = (miu * score_safe * score_safe) / (p_scale * torch.clamp(yita ** 3, min=1e-10))
        tao_k = solve_cubic_ratio_norm(D_k)

        # y_k update
        if torch.allclose(dk, torch.zeros_like(dk)):
            fangsuo = safe_cbrt(ck / p_scale)
            random_tensor = torch.randn_like(yk)
            norm_per_neuron = torch.norm(random_tensor.view(out_channels, -1), p=2, dim=1) + 1e-8
            norm_per_neuron = norm_per_neuron.view(out_channels, 1, 1, 1).expand_as(w)
            yk = random_tensor * (fangsuo / norm_per_neuron)
        else:
            yk = tao_k * dk

        # z_k update: soft-thresholding
        # Threshold scales with lr*C (matching Lasso) and inversely with score
        base_thresh = self.lr * self.C * 0.0001
        thresh = base_thresh / score_safe
        thresh = torch.clamp(thresh, min=1e-6, max=0.1)

        update_val = score_safe * qk + wk / p_scale
        zk = soft_thresholding(update_val, thresh)

        # Dual variable updates
        vk = vk + p_scale * (score_safe * qk - yk)
        wk = wk + p_scale * (score_safe * qk - zk)

        # Update weights to pruned values
        w = zk

        return w, vk, yk, zk, wk

    def _linear_neuronwise_update(self, w, vk, yk, zk, wk, grad, score, p_scale):
        """ADMM update for Linear layers, per output neuron."""
        out_features = w.shape[0]
        score_safe = score + 1e-8

        # q_k update (consistent with global/layer formulation)
        qk = 0.5 * (yk + zk - vk / p_scale - wk / p_scale) / score_safe
        qk -= grad / (score_safe * p_scale * 2.0)

        # Compute per-neuron norms for y_k update
        ck = torch.norm(score_safe * zk, p=1, dim=1, keepdim=True).expand_as(w)
        dk = score_safe * qk + vk / p_scale
        yita = torch.norm(score_safe * dk, p=2, dim=1, keepdim=True) + 1e-8
        yita = yita.expand_as(w)

        # D_k for cubic solver
        miu = self.C * ck / self.N
        D_k = (miu * score_safe * score_safe) / (p_scale * torch.clamp(yita ** 3, min=1e-10))
        tao_k = solve_cubic_ratio_norm(D_k)

        # y_k update
        if torch.allclose(dk, torch.zeros_like(dk)):
            fangsuo = safe_cbrt(ck / p_scale)
            random_tensor = torch.randn_like(yk)
            norm_per_neuron = torch.norm(random_tensor, p=2, dim=1, keepdim=True) + 1e-8
            norm_per_neuron = norm_per_neuron.expand_as(w)
            yk = random_tensor * (fangsuo / norm_per_neuron)
        else:
            yk = tao_k * dk

        # z_k update: soft-thresholding
        # Threshold scales with lr*C (matching Lasso) and inversely with score
        base_thresh = self.lr * self.C * 0.0001
        thresh = base_thresh / score_safe
        thresh = torch.clamp(thresh, min=1e-6, max=0.1)

        update_val = score_safe * qk + wk / p_scale
        zk = soft_thresholding(update_val, thresh)

        # Dual variable updates
        vk = vk + p_scale * (score_safe * qk - yk)
        wk = wk + p_scale * (score_safe * qk - zk)

        w = zk

        return w, vk, yk, zk, wk

    def _vector_update(self, w, vk, yk, zk, wk, grad, score, p_scale):
        """ADMM update for 1D parameters (bias, BatchNorm)."""
        score_safe = score + 1e-8

        # q_k update (consistent with global/layer formulation)
        qk = 0.5 * (yk + zk - vk / p_scale - wk / p_scale) / score_safe
        qk -= grad / (score_safe * p_scale * 2.0)

        # Compute norms for y_k update
        ck = torch.norm(score_safe * zk, p=1)
        dk = score_safe * qk + vk / p_scale
        yita = safe_norm(score_safe * dk)

        # D_k for cubic solver
        miu = self.C * ck / self.N
        D_k = (miu * score_safe * score_safe) / (p_scale * torch.clamp(yita ** 3, min=1e-10))
        tao_k = solve_cubic_ratio_norm(D_k)

        # y_k update
        if torch.allclose(dk, torch.zeros_like(dk)):
            fangsuo = safe_cbrt(ck / p_scale)
            random_tensor = torch.randn_like(yk)
            yk = random_tensor * (fangsuo / safe_norm(random_tensor))
        else:
            yk = tao_k * dk

        # z_k update: soft-thresholding
        # Threshold scales with lr*C (matching Lasso) and inversely with score
        base_thresh = self.lr * self.C * 0.0001
        thresh = base_thresh / score_safe
        thresh = torch.clamp(thresh, min=1e-6, max=0.1)

        update_val = score_safe * qk + wk / p_scale
        zk = soft_thresholding(update_val, thresh)

        # Dual variable updates
        vk = vk + p_scale * (score_safe * qk - yk)
        wk = wk + p_scale * (score_safe * qk - zk)

        w = zk

        return w, vk, yk, zk, wk
