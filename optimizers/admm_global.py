import torch
from torch.optim import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .utils import soft_thresholding

class ADMM_Global(Optimizer):

    def __init__(self, params, model, lr, N, C, vk, wk, yk, zk, beta, beta2 ,v0, v1, k, score, adam):
        
        defaults = dict(lr=lr, N=N, C=C, beta=beta, beta2=beta2)
        super(ADMM_Global, self).__init__(params, defaults)

        # Store these as state rather than attributes
        self.state['vk'] = parameters_to_vector(vk)
        self.state['wk'] = parameters_to_vector(wk)
        self.state['yk'] = parameters_to_vector(yk)
        self.state['zk'] = parameters_to_vector(zk)
        self.state['score'] = parameters_to_vector(score)
        self.state['v0'] = v0
        self.state['v1'] = v1
        self.state['k'] = k
        
        self.model = model  # This can stay as attribute
        self.adam = adam
        
    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        epi = 1e-8
        grad = parameters_to_vector([param.grad for param in self.model.parameters()])

        self.state['v0'] = self.defaults['beta'] * self.state['v0'] + (1 - self.defaults['beta']) * grad
        
        if self.adam:
            self.state['v1'] = self.defaults['beta2'] * self.state['v1'] + (1 - self.defaults['beta2']) * torch.mul(grad, grad)
            v1_new = self.state['v1'] / (1 - self.defaults['beta2'] ** (self.state['k'] + 1))
            lr = self.defaults['lr'] / (torch.sqrt(v1_new) + epi) / (1 - self.defaults['beta'] ** (self.state['k'] + 1))

        grad = self.state['v0']

        p = 1 / lr
        qk = 0.5 * (self.state['yk'] + self.state['zk'] - self.state['vk'] / p - self.state['wk'] / p - grad / p)

        ck = torch.norm(torch.mul(self.state['score'], self.state['zk']), p=1)
        dk = qk + self.state['vk'] / p
        yita = torch.norm(torch.mul(self.state['score'], dk), p=2) + 1e-8
        miu = self.defaults['C'] * ck / self.defaults['N']
        D_k = (miu * torch.mul(self.state['score'], self.state['score'])) / (p * (yita ** 3))
        C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** (1 / 2)) / 2) ** (1 / 3)
        tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

        if torch.all(dk == 0):
            fangsuo = (ck / p) ** (1 / 3)
            random_tensor = torch.randn_like(self.state['yk'])
            self.state['yk'].copy_(random_tensor * (fangsuo / torch.norm(random_tensor, p=2)))
        else:
            self.state['yk'].copy_(tao_k * dk)

        self.state['zk'] = soft_thresholding(b=qk + self.state['wk'] / p,
                                         u=(self.defaults['C'] / self.defaults['N']) / (p * torch.norm(self.state['yk'], p=2)))

        self.state['vk'].add_(p * (qk - self.state['yk']))
        self.state['wk'].add_(p * (qk - self.state['zk']))

        vector_to_parameters(self.state['zk'], self.model.parameters())

        self.state['k'] += 1

        return loss

    def update_base_learning_rate(self, new_lr):
        self.defaults['lr'] = new_lr