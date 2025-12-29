
import torch
from torch.optim import Optimizer

def soft_thresholding(b, u):
    return torch.sign(b) * torch.max(torch.zeros_like(b), torch.abs(b) - u)

class ADMM_Adam_neuron(Optimizer):
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

        for group in self.param_groups:
            for w, vk_temp, yk_temp, zk_temp, wk_temp, wanda_score_1 in zip(
                group["params"], self.vk, self.yk, self.zk, self.wk, self.score
            ):

                w_len = len(w.shape)

                if w_len == 4: #判断该prune哪一层，该层是CNN层，如果加了别的模型，如果多了一些别的层，这个地方可能需要重新调整，不够robust
                     #进入该层后使用cnn_neuronwise_pruning函数执行ADMM更新
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = self.cnn_neuronwise_pruning(
                        w, vk_temp, yk_temp, zk_temp, wk_temp, self.lr, self.N, self.C, wanda_score_1
                    )
                elif w_len == 2: #同上，此是全连接层
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = self.fullycont(
                        w, vk_temp, yk_temp, zk_temp, wk_temp, self.lr, self.N, self.C, wanda_score_1
                    )
                elif w_len == 1:#同上，此是所有bias参数以及batchnorm的参数
                    w.data, vk_temp.data, yk_temp.data, zk_temp.data, wk_temp.data = self.batchnorm_and_bias_pruning(
                        w, vk_temp, yk_temp, zk_temp, wk_temp, self.lr, self.N, self.C, wanda_score_1
                    )

        return loss

    def cnn_neuronwise_pruning(self, w, vk_temp, yk_temp, zk_temp, wk_temp, lr, N, C,wanda_score_1):
        shape0, _, _, _ = w.shape
        p = 1 / lr
        grad = w.grad

        qk = 0.5 * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - grad / p)

        ck = torch.norm((wanda_score_1 * zk_temp).view(shape0, -1), p=1, dim=1).view(shape0, 1, 1, 1).expand_as(w)

        dk = qk + vk_temp / p
        yita = torch.norm((wanda_score_1 * dk).view(shape0, -1), p=2, dim=1).view(shape0, 1, 1, 1).expand_as(w) + 1e-8
        miu = self.C * ck / self.N
        # ✅ P5修复：添加epsilon避免除法产生Inf
        D_k = (miu * torch.mul(wanda_score_1, wanda_score_1)) / (p * torch.clamp((yita) ** 3, min=1e-10))
        C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** (1 / 2)) / 2) ** (1 / 3)
        tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

        if torch.all(dk == 0):
            fangsuo = (ck / p) ** (1 / 3)
            random_tensor = torch.randn_like(yk_temp)
            yk_temp = random_tensor * (fangsuo / torch.norm(random_tensor.view(shape0, -1), p=2, dim=1).view(shape0, 1, 1, 1).expand_as(w))
        else:
            yk_temp = torch.mul(tao_k, dk)

        b = qk + wk_temp / p
        # Use weight-magnitude-based threshold for stability
        weight_scale = torch.norm(w.view(shape0, -1), p=2, dim=1).view(shape0, 1, 1, 1).expand_as(w) + 1e-8
        base_thresh = C * 0.0001  # C controls pruning strength (smaller for per-neuron)
        u = torch.clamp(base_thresh / weight_scale, min=1e-6, max=0.1)

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
        yita = torch.norm(wanda_score_1 * dk, p=2, dim=1).unsqueeze(1).expand_as(w) + 1e-8
        miu = self.C * ck / self.N
        # ✅ P5修复：添加epsilon避免除法产生Inf
        D_k = (miu * torch.mul(wanda_score_1, wanda_score_1)) / (p * torch.clamp((yita) ** 3, min=1e-10))
        C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** 0.5) / 2) ** (1 / 3)
        tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

        if torch.all(dk == 0):
            fangsuo = (ck / p) ** (1 / 3)
            random_tensor = torch.randn_like(yk_temp)
            yk_temp = random_tensor * (fangsuo / torch.norm(random_tensor, p=2, dim=1).unsqueeze(1).expand_as(w))
        else:
            yk_temp = torch.mul(tao_k, dk)

        b = qk + wk_temp / p
        # Use weight-magnitude-based threshold for stability
        weight_scale = torch.norm(w, p=2, dim=1).unsqueeze(1).expand_as(w) + 1e-8
        base_thresh = C * 0.0001  # C controls pruning strength (smaller for per-neuron)
        u = torch.clamp(base_thresh / weight_scale, min=1e-6, max=0.1)
        zk_temp = soft_thresholding(b, u)

        vk_temp = vk_temp + p * (qk - yk_temp)
        wk_temp = wk_temp + p * (qk - zk_temp)
        w = zk_temp

        return w, vk_temp, yk_temp, zk_temp, wk_temp

    def batchnorm_and_bias_pruning(self, w, vk_temp, yk_temp, zk_temp, wk_temp, lr, N, C, wanda_score_1):
        p = 1 / lr
        grad = w.grad

        qk = 0.5 * (yk_temp + zk_temp - vk_temp / p - wk_temp / p - grad / p)

        ck = torch.norm(wanda_score_1 * zk_temp, p=1)
        dk = qk + vk_temp / p
        yita = torch.norm(wanda_score_1 * dk, p=2) + 1e-8
        miu = self.C * ck / self.N
        D_k = (miu * torch.mul(wanda_score_1, wanda_score_1)) / (p * torch.clamp((yita) ** 3, min=1e-10))
        C_K = ((27 * D_k + 2 + torch.sqrt(torch.clamp((27 * D_k + 2) ** 2 - 4, min=0.0))) / 2) ** (1 / 3)
        tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

        if torch.all(dk == 0):
            fangsuo = (ck / p) ** (1 / 3)
            random_tensor = torch.randn_like(yk_temp)
            yk_temp = random_tensor * (fangsuo / (torch.norm(random_tensor, p=2) + 1e-8))
        else:
            yk_temp = torch.mul(tao_k, dk)

        b = qk + wk_temp / p
        # Use weight-magnitude-based threshold for stability
        weight_scale = torch.norm(w, p=2) + 1e-8
        base_thresh = C * 0.0001  # C controls pruning strength (smaller for per-neuron)
        u = torch.clamp(base_thresh / weight_scale, min=1e-6, max=0.1)
        zk_temp = soft_thresholding(b, u)

        vk_temp = vk_temp + p * (qk - yk_temp)
        wk_temp = wk_temp + p * (qk - zk_temp)
        w = zk_temp

        return w, vk_temp, yk_temp, zk_temp, wk_temp
