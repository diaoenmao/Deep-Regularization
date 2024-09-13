import torch
from torch.optim import Optimizer

def soft_thresholding(b, u):
    return torch.sign(b) * torch.max(torch.zeros_like(b), torch.abs(b) - u)

class ADMM_Adam_neuron(Optimizer):
    def __init__(self, params, lr, N, C, vk, wk, yk, zk):
        self.lr = lr
        self.N = N
        self.C = C
        self.vk = vk
        self.wk = wk
        self.yk = yk
        self.zk = zk

        super(ADMM_Adam_neuron, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            for w, vk_temp, yk_temp, zk_temp, wk_temp in zip(group['params'], self.vk, self.yk, self.zk, self.wk):

                w_len = len(w.shape)

                if w_len == 4:
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = self.cnn_neuronwise_pruning(
                        w, vk_temp, yk_temp, zk_temp, wk_temp, self.lr, self.N, self.C
                    )
                elif w_len == 2:
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = self.fullycont(
                        w, vk_temp, yk_temp, zk_temp, wk_temp, self.lr, self.N, self.C
                    )
                elif w_len == 1:
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = self.batchnorm_and_bias_pruning(
                        w, vk_temp, yk_temp, zk_temp, wk_temp, self.lr, self.N, self.C
                    )

        return loss

    def cnn_neuronwise_pruning(self, w, vk_temp, yk_temp, zk_temp, wk_temp, lr, N, C):
        shape0, _, _, _ = w.shape
        p = 1 / lr
        grad = w.grad

        qk = 0.5 * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - grad / p)

        ck = torch.norm(zk_temp.view(shape0, -1), p=1, dim=1).view(shape0, 1, 1, 1).expand_as(w)

        dk = qk + vk_temp / p
        yita = torch.norm(dk.view(shape0, -1), p=2, dim=1).view(shape0, 1, 1, 1).expand_as(w) + 1e-8
        D_k = torch.div(C / N * ck, p * (yita ** 3))
        C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** 0.5) / 2) ** (1 / 3)
        tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

        if torch.all(dk == 0):
            fangsuo = (ck / p) ** (1 / 3)
            random_tensor = torch.randn_like(yk_temp)
            yk_temp = random_tensor * (fangsuo / torch.norm(random_tensor.view(shape0, -1), p=2, dim=1).view(shape0, 1, 1, 1).expand_as(w))
        else:
            yk_temp = torch.mul(tao_k, dk)

        b = qk + wk_temp / p
        yk_temp_norm = torch.norm(yk_temp.view(shape0, -1), p=2, dim=1).view(shape0, 1, 1, 1).expand_as(w) + 1e-8
        u = (C / N) / (p * yk_temp_norm)
        zk_temp = soft_thresholding(b, u)

        vk_temp = vk_temp + p * (qk - yk_temp)
        wk_temp = wk_temp + p * (qk - zk_temp)
        w = zk_temp

        return w, vk_temp, yk_temp, zk_temp, wk_temp

    def fullycont(self, w, vk_temp, yk_temp, zk_temp, wk_temp, lr, N, C):
        shape0, shape1 = w.shape
        p = 1 / lr
        grad = w.grad

        qk = 0.5 * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - grad / p)

        ck = torch.norm(zk_temp, p=1, dim=1).unsqueeze(1).expand_as(w)
        dk = qk + vk_temp / p
        yita = torch.norm(dk, p=2, dim=1).unsqueeze(1).expand_as(w)
        D_k = torch.div(C / N * ck, p * (yita ** 3))
        C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** 0.5) / 2) ** (1 / 3)
        tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

        if torch.all(dk == 0):
            fangsuo = (ck / p) ** (1 / 3)
            random_tensor = torch.randn_like(yk_temp)
            yk_temp = random_tensor * (fangsuo / torch.norm(random_tensor, p=2, dim=1).unsqueeze(1).expand_as(w))
        else:
            yk_temp = torch.mul(tao_k, dk)

        b = qk + wk_temp / p
        yk_temp_norm = torch.norm(yk_temp, p=2, dim=1).unsqueeze(1).expand_as(w)
        u = (C / N) / (p * yk_temp_norm)
        zk_temp = soft_thresholding(b, u)

        vk_temp = vk_temp + p * (qk - yk_temp)
        wk_temp = wk_temp + p * (qk - zk_temp)
        w = zk_temp

        return w, vk_temp, yk_temp, zk_temp, wk_temp

    def batchnorm_and_bias_pruning(self, w, vk_temp, yk_temp, zk_temp, wk_temp, lr, N, C):
        p = 1 / lr
        grad = w.grad

        qk = 0.5 * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - grad / p)

        ck = torch.norm(zk_temp, p=1)
        dk = qk + vk_temp / p
        yita = torch.norm(dk, p=2)
        D_k = (C / N * ck) / (p * (yita ** 3))
        C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** 0.5) / 2) ** (1 / 3)
        tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

        if torch.all(dk == 0):
            fangsuo = (ck / p) ** (1 / 3)
            random_tensor = torch.randn_like(yk_temp)
            yk_temp = random_tensor * (fangsuo / torch.norm(random_tensor, p=2))
        else:
            yk_temp = tao_k * dk

        b = qk + wk_temp / p
        u = (C / N) / (p * torch.norm(yk_temp, p=2))
        zk_temp = soft_thresholding(b, u)

        vk_temp = vk_temp + p * (qk - yk_temp)
        wk_temp = wk_temp + p * (qk - zk_temp)
        w = zk_temp

        return w, vk_temp, yk_temp, zk_temp, wk_temp

    def update_base_learning_rate(self, new_lr):
        self.lr = new_lr
