from torch.optim import Optimizer
import torch
from .utils import soft_thresholding


class ADMM_Layer(Optimizer):

    def __init__(self, params, lr, N, C, vk, wk, yk, zk, beta, beta2 ,v0, v1, k, score):
        self.lr = lr
        self.N = N #NUMBER OF SAMPLE
        self.C = C #CONSTANT
        self.vk = vk
        self.wk = wk
        self.yk = yk
        self.zk = zk
        self.beta = beta
        self.beta2 = beta2
        self.v0 = v0
        self.v1 = v1
        self.k = k
        self.score = score
        super(ADMM_Layer, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        epi = 1e-6

        for group in self.param_groups:
            for w, vk_temp, yk_temp, zk_temp, wk_temp, score_temp in zip(group['params'], self.vk, self.yk,
                                                                            self.zk, self.wk, self.score):

                grad = w.grad

                p = 1 / self.lr
                qk = 0.5 * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - grad / p)

                ck = torch.norm(score_temp * zk_temp, p=1) 
                dk = qk + vk_temp / p
                yita = torch.norm(score_temp * dk, p=2)
                miu = self.C * ck / self.N
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
                u = (self.C / self.N) / (p * torch.norm(yk_temp, p=2))

                zk_temp.copy_(soft_thresholding(b, u))

                vk_temp.add_(p * (qk - yk_temp))
                wk_temp.add_(p * (qk - zk_temp))
                w.copy_(zk_temp)

        return loss
    
    def update_base_learning_rate(self, new_lr):
        self.lr = new_lr