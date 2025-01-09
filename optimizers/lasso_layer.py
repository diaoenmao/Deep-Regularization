from torch.optim import Optimizer
import torch
from .utils import soft_thresholding
import math

class LASSO_Layer(Optimizer):
    def __init__(self, params, lr, N, C, score, model, wk, zk, vk, beta, beta2, v0, v1, k, adam):
        self.lr = lr
        self.N = N  # NUMBER OF SAMPLES
        self.C = C  # REGULARIZATION CONSTANT
        self.score = score 
        self.model = model
        self.wk = wk
        self.zk = zk
        self.vk = vk
        self.beta = beta
        self.beta2 = beta2
        self.v0 = v0
        self.v1 = v1
        self.k = k
        self.adam = adam
        super(LASSO_Layer, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for w, score_temp, wk_temp, vk_temp, zk_temp, v0_temp, v1_temp in zip(group['params'], self.score, self.wk, self.vk, self.zk, self.v0, self.v1):
                if w.grad is None:
                    continue

                epi = 1e-8
                p = 1 / self.lr
                
                grad = w.grad

                v0_temp = self.beta * v0_temp + (1 - self.beta) * grad
                bias_1 = 1 - self.beta ** (self.k + 1)
                v0_corrected = v0_temp / bias_1

                if self.adam:
                    v1_temp = self.beta2 * v1_temp + (1 - self.beta2) * grad.pow(2)
                    bias_2 = 1 - self.beta2 ** (self.k + 1)
                    v1_corrected = v1_temp / bias_2

                grad = v0_corrected

                lr = self.lr * math.sqrt(bias_2) / bias_1
                lr = lr / (torch.sqrt(v1_corrected) + epi)

                wk_new = wk_temp - vk_temp / p - grad / p
                b = wk_new + vk_temp / p
                u = self.C / self.N / p * torch.abs(score_temp)
                zk_new = soft_thresholding(b, u)
                vk_new = vk_temp + (wk_new - zk_new) * p

                zk_temp.copy_(zk_new)
                wk_temp.copy_(wk_new)
                vk_temp.copy_(vk_new)
                w.copy_(zk_temp)

        self.k += 1
        return loss


    def update_base_learning_rate(self, new_lr):
        self.lr = new_lr