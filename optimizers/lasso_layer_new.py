from torch.optim import Optimizer
import torch
from .utils import soft_thresholding
import math


class LASSO_Layer(Optimizer):
    def __init__(self, params, lr, N, C, score, model, wk, zk, vk, beta, beta2, v0, v1, k, adam):
        defaults = dict(lr=lr, N=N, C=C, beta=beta, beta2=beta2)
        super(LASSO_Layer, self).__init__(params, defaults)
        self.score = score
        self.model = model
        self.wk = wk
        self.zk = zk
        self.vk = vk
        self.v0 = v0
        self.v1 = v1
        self.k = k
        self.adam = adam

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for w, score_temp, wk_temp, vk_temp, zk_temp, v0_temp, v1_temp in zip(group['params'], self.score, self.wk,
                                                                                  self.vk, self.zk, self.v0, self.v1):
                if w.grad is None:
                    continue

                epi = 1e-8
                grad = w.grad

                lr = self.defaults['lr']

                if self.adam:
                    v0_temp = self.defaults['beta'] * v0_temp + (1 - self.defaults['beta']) * grad
                    bias_1 = 1 - self.defaults['beta'] ** (self.k + 1)
                    v0_corrected = v0_temp / bias_1
                    v1_temp = self.defaults['beta2'] * v1_temp + (1 - self.defaults['beta2']) * grad.pow(2)
                    bias_2 = 1 - self.defaults['beta2'] ** (self.k + 1)
                    v1_corrected = v1_temp / bias_2
                    lr = lr * math.sqrt(bias_2) / bias_1
                    lr = lr / (torch.sqrt(v1_corrected) + epi)
                    grad = v0_corrected

                w.data = w.data - lr * grad
                w_copy = w.data
                mask1 = torch.where(w.data > 0, 1, 0.0)
                mask2 = torch.where(w.data < 0, 1, 0.0)
                w.data = mask1 * (w.data - (self.defaults['C'] / self.defaults['N']) * lr) + mask2 * (
                            w.data + (self.defaults['C'] / self.defaults['N']) * lr)
                w.data = (torch.where(abs(w.data - w_copy) < abs(w_copy), 1, 0)) * w.data


        self.k += 1
        return loss

    def update_base_learning_rate(self, new_lr):
        self.defaults['lr'] = new_lr