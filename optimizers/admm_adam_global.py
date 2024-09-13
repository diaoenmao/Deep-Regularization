import torch
from torch.optim import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def soft_thresholding(b, u):
    z = torch.sign(b) * torch.max(torch.zeros_like(b), torch.abs(b)-u)

    return z

class ADMM_Adam_global(Optimizer):

    def __init__(self, params, model, lr, N, C, vk, wk, yk, zk, beta, beta2 ,v0, v1, k):
        self.model = model
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
        super(ADMM_Adam_global, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        epi = 1e-8
        w = parameters_to_vector(self.model.parameters())
        grad = parameters_to_vector([param.grad for param in self.model.parameters()])

        self.v0 = self.beta * self.v0 + (1 - self.beta) * grad
        self.v1 = self.beta2 * self.v1 + (1 - self.beta) * torch.mul(grad, grad)

        v1_new = self.v1 / (1 - self.beta2 ** (self.k + 1))

        lr = self.lr / (torch.sqrt(v1_new) + epi) / (1 - self.beta ** (self.k + 1))

        grad = self.v0

        p = 1 / lr
        qk = 0.5 * (self.yk + self.zk - self.vk / p - self.wk / p - grad / p)

        ck = torch.norm(self.zk, p=1)
        dk = qk + self.vk / p
        yita = torch.norm(dk, p=2) + 1e-8
        D_k = ((self.C / self.N) * ck) / (p * (yita ** 3))
        C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** (1 / 2)) / 2) ** (1 / 3)
        tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

        if torch.all(dk == 0):
            fangsuo = (ck / p) ** (1 / 3)
            random_tensor = torch.randn_like(self.yk)
            self.yk.copy_(random_tensor * (fangsuo / torch.norm(random_tensor, p=2)))
        else:
            self.yk.copy_(tao_k * dk)

        self.zk = soft_thresholding(b=qk + self.wk / p,
                                         u=(self.C / self.N) / (p * torch.norm(self.yk, p=2)))

        self.vk.add_(p * (qk - self.yk))
        self.wk.add_(p * (qk - self.zk))

        vector_to_parameters(self.zk, self.model.parameters())

        self.k += 1

        return loss

    def update_base_learning_rate(self, new_lr):
        self.lr = new_lr