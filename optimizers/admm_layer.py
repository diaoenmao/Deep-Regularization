from torch.optim import Optimizer
import torch
from .utils import soft_thresholding


class ADMM_Layer(Optimizer):

    def __init__(self, params, lr, N, C, vk, wk, yk, zk, beta, beta2, v0, v1, k, score, model, adam=True):
        defaults = dict(lr=lr, N=N, C=C, beta=beta, beta2=beta2)
        self.v0 = v0
        self.v1 = v1
        self.k = k
        self.vk = vk
        self.wk = wk
        self.yk = yk
        self.zk = zk
        self.score = score
        self.model = model
        self.adam = adam  # New parameter to toggle Adam
        super(ADMM_Layer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        epi = 1e-8
        for group in self.param_groups:
            for w, vk_temp, yk_temp, zk_temp, wk_temp, v0_temp, v1_temp, score_temp in zip(group['params'], self.vk, self.yk,
                                                                        self.zk, self.wk, self.v0, self.v1, self.score):
                if w.grad is None:
                    continue

                grad = w.grad
                
                # Adam-style momentum updates
                v0_temp = self.defaults['beta'] * v0_temp + (1 - self.defaults['beta']) * grad
                
                if self.adam:
                    # Second moment estimate
                    v1_temp = self.defaults['beta2'] * v1_temp + (1 - self.defaults['beta2']) * torch.mul(grad, grad)
                    v1_new = v1_temp / (1 - self.defaults['beta2'] ** (self.k + 1))
                    lr = self.defaults['lr'] / (torch.sqrt(v1_new) + epi) / (1 - self.defaults['beta'] ** (self.k + 1))
                else:
                    lr = self.defaults['lr']

                grad = v0_temp  # Use momentum-corrected gradient

                p = 1 / lr
                qk = 0.5 * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - grad / p)

                ck = torch.norm(score_temp * zk_temp, p=1)
                dk = qk + vk_temp / p
                yita = torch.norm(score_temp * dk, p=2) + epi
                miu = self.defaults['C'] * ck / self.defaults['N']
                D_k = (miu * torch.mul(score_temp, score_temp)) / (p * (yita) ** 3)
                C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** (1 / 2)) / 2) ** (1 / 3)
                tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

                if torch.all(dk == 0):
                    fangsuo = (ck / p) ** (1 / 3)
                    random_tensor = torch.randn_like(yk_temp)
                    yk_temp.copy_(random_tensor * (fangsuo / torch.norm(random_tensor, p=2)))
                else:
                    yk_temp.copy_(tao_k * dk)

                b = qk + wk_temp / p
                u = (self.defaults['C'] / self.defaults['N']) / (p * torch.norm(yk_temp, p=2))

                zk_temp.copy_(soft_thresholding(b, u))

                vk_temp.add_(p * (qk - yk_temp))
                wk_temp.add_(p * (qk - zk_temp))
                w.copy_(zk_temp)

        self.k += 1
        return loss
    
    def update_base_learning_rate(self, new_lr):
        self.defaults['lr'] = new_lr