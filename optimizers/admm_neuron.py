import torch
from torch.optim import Optimizer
from .utils import soft_thresholding

class ADMM_Neuron(Optimizer):
    def __init__(self, params, lr, N, C, vk, wk, yk, zk, score, model, beta, beta2, v0, v1, k):
        self.lr = lr
        self.N = N
        self.C = C
        self.vk = vk
        self.wk = wk
        self.yk = yk
        self.zk = zk
        self.score = score
        self.model = model
        self.beta = beta
        self.beta2 = beta2
        self.v0 = v0
        self.v1 = v1
        self.k = k
        super(ADMM_Neuron, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            for w, vk_temp, yk_temp, zk_temp, wk_temp, score_temp in zip(group['params'], self.vk, self.yk, self.zk, self.wk,self.score):

                w_len = len(w.shape)

                if w_len == 4:
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = self.cnn_neuronwise_pruning(
                        w, vk_temp, yk_temp, zk_temp, wk_temp, self.lr, self.N, self.C, score_temp
                    )
                elif w_len == 2:
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = self.fullycont(
                        w, vk_temp, yk_temp, zk_temp, wk_temp, self.lr, self.N, self.C, score_temp
                    )
                elif w_len == 1:
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = self.batchnorm_and_bias_pruning(
                        w, vk_temp, yk_temp, zk_temp, wk_temp, self.lr, self.N, self.C, score_temp
                    )

        return loss

    def cnn_neuronwise_pruning(self, w, vk_temp, yk_temp, zk_temp, wk_temp, lr, N, C,wanda_score_1):
        shape0, _, _, _ = w.shape
        p = 1 / lr
        grad = w.grad

        qk = 0.5 * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - grad / p)

        ck = torch.norm((wanda_score_1 * zk_temp).view(shape0, -1), p=1, dim=1).view(shape0, 1, 1, 1).expand_as(w) #此处有修改，在于将zk_temp修改为了wanda_score_1 * zk_temp

        dk = qk + vk_temp / p
        yita = torch.norm((wanda_score_1 * dk).view(shape0, -1), p=2, dim=1).view(shape0, 1, 1, 1).expand_as(w) + 1e-8 #同上改动
        miu = self.C * ck / self.N
        D_k = (miu * torch.mul(wanda_score_1, wanda_score_1)) / (p * (yita) ** 3) #在scorebased更新中，D_K的表述有所修改
        C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** (1 / 2)) / 2) ** (1 / 3)
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

        zk_temp = soft_thresholding(b=b,
                                    u=u)
        vk_temp = vk_temp + p * (qk - yk_temp)
        wk_temp = wk_temp + p * (qk - zk_temp)
        w = zk_temp

        return w, vk_temp, yk_temp, zk_temp, wk_temp

    def fullycont(self, w, vk_temp, yk_temp, zk_temp, wk_temp, lr, N, C, wanda_score_1): #此处同cnn_neuronwise_pruning的改动
        shape0, shape1 = w.shape
        p = 1 / lr
        grad = w.grad

        qk = 0.5 * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - grad / p)

        ck = torch.norm(wanda_score_1 * zk_temp, p=1, dim=1).unsqueeze(1).expand_as(w)
        dk = qk + vk_temp / p
        yita = torch.norm(wanda_score_1 * dk, p=2, dim=1).unsqueeze(1).expand_as(w)
        miu = self.C * ck / self.N
        D_k = (miu * torch.mul(wanda_score_1, wanda_score_1)) / (p * (yita) ** 3)
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

    def batchnorm_and_bias_pruning(self, w, vk_temp, yk_temp, zk_temp, wk_temp, lr, N, C, wanda_score_1): #此处同cnn_neuronwise_pruning的改动
        p = 1 / lr
        grad = w.grad

        qk = 0.5 * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - grad / p)

        ck = torch.norm(wanda_score_1 * zk_temp, p=1)
        dk = qk + vk_temp / p
        yita = torch.norm(wanda_score_1 * dk, p=2)
        miu = self.C * ck / self.N
        D_k = (miu * torch.mul(wanda_score_1, wanda_score_1)) / (p * (yita) ** 3)
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
        self.defaults['lr'] = new_lr