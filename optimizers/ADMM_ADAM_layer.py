from torch.optim import Optimizer
import torch
import torch.nn as nn
import sys


def soft_thresholding(b, u):
    z = torch.sign(b) * torch.max(torch.zeros_like(b), torch.abs(b)-u)
    return z


class ADMM_Adam_Layer(Optimizer):

    def __init__(self, params, lr, N, C, vk, wk, yk, zk, beta, beta2 ,v0, v1, k):
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
        super(ADMM_Adam_Layer, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        epi = 1e-8

        for group in self.param_groups:
            for w, vk_temp, yk_temp, zk_temp, wk_temp, v0_temp, v1_temp in zip(group['params'], self.vk, self.yk, self.zk, self.wk, self.v0, self.v1):

                # if w.isnan().any():
                #     print("Warning: NaN values detected in w")
                #     print("w:", w)
                #     print("w.isnan().any():", w.isnan().any())
                #     print("w.isinf().any():", w.isinf().any())
                #     print("k:", self.k)
                #     sys.exit()

                grad = w.grad
                print("Shape of grad:", grad.shape)
                # if torch.isnan(grad).any():
                #     print("Warning: NaN values detected in grad")
                #     print("grad:", grad)
                #     print("w:", w)
                #     print("w.grad:", w.grad)
                #     print("w.grad.isnan().any():", w.grad.isnan().any())
                #     print("w.grad.isinf().any():", w.grad.isinf().any())
                #     print("k:", self.k)
                #     sys.exit()

                v0_temp.mul_(self.beta).add_(grad, alpha=1 - self.beta)

                # if torch.isnan(v0_temp).any():
                #     print("Warning: NaN values detected in v0_temp")
                #     print("v0_temp:", v0_temp)
                #     print("self.beta:", self.beta)
                #     print("self.k:", self.k)
                #     print("grad:", grad)
                #     print("grad.isnan().any():", grad.isnan().any())
                #     print("grad.isinf().any():", grad.isinf().any())
                #     sys.exit()
                
                # Use element-wise multiplication instead of matrix multiplication
                v1_temp.mul_(self.beta2).add_(grad * grad, alpha=1 - self.beta2)
                
                if torch.any(v1_temp < 0):
                    print("Warning: Negative values detected in v1_temp")
                    neg_mask = v1_temp < 0
                    print("v1_temp:", v1_temp[neg_mask])
                    print("self.beta2:", self.beta2)
                    print("grad:", grad[neg_mask])
                    print("k:", self.k)
                    sys.exit()

                # if torch.isnan(v1_temp).any():
                #     print("Warning: NaN values detected in v1_temp")
                #     print("v1_temp:", v1_temp)
                #     print("self.beta2:", self.beta2)
                #     print("self.k:", self.k)
                #     sys.exit()

                v1_new = v1_temp / (1 - self.beta2 ** (self.k + 1))

                # # Check if v1_new contains NaN values
                # if torch.isnan(v1_new).any():
                #     print("Warning: NaN values detected in v1_new")
                #     print("v1_temp:", v1_temp)
                #     print("self.beta2:", self.beta2)
                #     print("self.k:", self.k)
                #     sys.exit()

                lr = self.lr / (torch.sqrt(v1_new) + epi) / (1 - self.beta ** (self.k + 1))

                # if torch.isnan(lr).any():
                #     print("Warning: NaN values detected in lr")
                #     print("lr:", lr)
                #     print("v1_new:", v1_new)
                #     print("self.lr:", self.lr)
                #     print("self.beta:", self.beta)
                #     print("self.k:", self.k)
                #     sys.exit()

                grad = v0_temp

                p = 1 / lr
                qk = 0.5 * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - v0_temp / p)

                ck = torch.norm(zk_temp, p=1)
                dk = qk + vk_temp / p
                yita = torch.norm(dk, p=2)
                D_k = ((self.C / self.N) * ck) / (p * (yita ** 3))
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

                self.k += 1

        return loss
    
    def update_base_learning_rate(self, new_lr):
        self.lr = new_lr