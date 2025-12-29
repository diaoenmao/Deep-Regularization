import torch
from torch.optim import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def soft_thresholding(b, u):
    return torch.sign(b) * torch.max(torch.zeros_like(b), torch.abs(b) - u)


class Lasso_global(Optimizer):

    def __init__(self, params, lr, N, C, vk, zk, score):
        self.lr = lr
        self.N = N  # NUMBER OF SAMPLE
        self.C = C  # CONSTANT
        self.vk = vk
        self.zk = zk
        self.score = score
        super(Lasso_global, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):

        w = parameters_to_vector(self.param_groups[0]['params'])
        score_temp = parameters_to_vector(self.score) + 1e-8
        grad = parameters_to_vector([p.grad for p in self.param_groups[0]['params']])
        lr = self.lr

        # Gradient descent step
        v = w - lr * grad
        
        # Soft thresholding with score-adjusted threshold
        # Higher score = more important = lower threshold (keep)
        # Lower score = less important = higher threshold (prune)
        base_thresh = lr * self.C * 0.01  # C controls sparsity strength
        thresh = base_thresh / score_temp
        thresh = torch.clamp(thresh, min=1e-6, max=0.1)
        
        new_w = soft_thresholding(v, thresh)
        
        vector_to_parameters(new_w, self.param_groups[0]['params'])

        return None


    __all__ = ["Lasso_global", "soft_thresholding"]