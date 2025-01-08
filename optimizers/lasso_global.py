from torch.optim import Optimizer
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .utils import soft_thresholding
import math


class LASSO_Global(Optimizer):

    def __init__(self, params, lr, N, C, score, model, wk, zk, vk, v0, v1, k, beta=0.9, beta2=0.999):
        defaults = dict(lr=lr, N=N, C=C, beta=beta, beta2=beta2)
        super(LASSO_Global, self).__init__(params, defaults)
        
        # Initialize state
        self.state['wk'] = wk
        self.state['zk'] = zk
        self.state['vk'] = vk
        self.state['score'] = score
        self.state['v0'] = v0
        self.state['v1'] = v1
        self.state['k'] = k
        self.model = model

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        epi = 1e-8
        grad = parameters_to_vector([param.grad for param in self.model.parameters()])
        
        # Clip gradients to prevent explosion
        grad = torch.clamp(grad, min=-1.0, max=1.0)
        
        # Adam-style updates with stability checks
        self.state['v0'] = self.defaults['beta'] * self.state['v0'] + (1 - self.defaults['beta']) * grad
        self.state['v1'] = self.defaults['beta2'] * self.state['v1'] + (1 - self.defaults['beta2']) * grad.pow(2)

        # Bias correction
        bias_correction1 = 1 - self.defaults['beta'] ** (self.state['k'] + 1)
        bias_correction2 = 1 - self.defaults['beta2'] ** (self.state['k'] + 1)
        
        v0_corrected = self.state['v0'] / bias_correction1
        v1_corrected = self.state['v1'] / bias_correction2

        # Compute adaptive learning rate
        lr = self.defaults['lr'] * math.sqrt(bias_correction2) / bias_correction1
        lr = lr / (torch.sqrt(v1_corrected) + epi)
        
        # Clip learning rate
        lr = torch.clamp(lr, min=1e-8, max=1.0)
        p = 1/lr
        
        # LASSO updates
        wk_new = self.state['wk'] - self.state['vk'] / p - v0_corrected / p
        b = wk_new + self.state['vk'] / p
        u = self.defaults['C'] / self.defaults['N'] / p * torch.abs(self.state['score'])
        zk_new = soft_thresholding(b, u)
        vk_new = self.state['vk'] + (wk_new - zk_new) * p

        # Update states
        self.state['zk'].copy_(zk_new)
        self.state['wk'].copy_(wk_new)
        self.state['vk'].copy_(vk_new)
        
        # Update model parameters
        vector_to_parameters(self.state['zk'], self.model.parameters())
        
        self.state['k'] += 1
        return loss

    def update_base_learning_rate(self, new_lr):
        self.defaults['lr'] = new_lr