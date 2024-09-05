from torch.optim import Optimizer
import torch
import torch.nn as nn
'''

该代码为lasso优化器的baseline代码，用于与我们的算法进行对比，如果你需要，可以使用，注意，这个代码也是在layer-wise部署的

'''
def Soft_Thresholding(b, u):
    z = torch.sign(b) * torch.max(torch.zeros_like(b), torch.abs(b)-u)
    return z

class ADMM_LASSO(Optimizer):

    def __init__(self, params, lr, N, C):
        self.lr = lr
        self.N = N #NUMBER OF SAMPLE
        self.C = C #CONSTANT
        super(ADMM_LASSO, self).__init__(params, {})

    def step(self, closure=None):

        loss = None
        for group in self.param_groups:
            for w in group['params']:
                w.data = w.data - self.lr * w.grad

                w_copy = w.data
                mask1 = torch.where(w.data > 0, 1, 0.0)
                mask2 = torch.where(w.data < 0, 1, 0.0)
                w.data = mask1 * (w.data - (self.C / self.N) * self.lr) + mask2 * (w.data + (self.C / self.N) * self.lr)
                w.data = (torch.where(abs(w.data - w_copy) < abs(w_copy), 1, 0)) * w.data

        return None