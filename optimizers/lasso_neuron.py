from torch.optim import Optimizer
import torch
import torch.nn as nn
from .utils import soft_thresholding
import math

class LASSO_Neuron(Optimizer):
    def __init__(self, params, model, lr, N, C, score, wk, zk, vk, beta, beta2, v0, v1, k, adam):
        defaults = dict(lr=lr, N=N, C=C, beta=beta, beta2=beta2)
        super(LASSO_Neuron, self).__init__(params, defaults)
        self.model = model
        self.score = score  # Score per neuron
        self.wk = wk
        self.zk = zk
        self.vk = vk
        self.v0 = v0
        self.v1 = v1
        self.k = k
        self.adam = adam

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for w, score_temp, wk_temp, vk_temp, zk_temp, v0_temp, v1_temp in zip(group['params'], self.score, self.wk, self.vk, self.zk, self.v0, self.v1):
                if w.grad is None:
                    continue

                epi = 1e-8
                grad = w.grad
                v0_temp = self.defaults['beta'] * v0_temp + (1 - self.defaults['beta']) * grad
                bias_1 = 1 - self.defaults['beta'] ** (self.k + 1)
                v0_corrected = v0_temp / bias_1
                lr = self.defaults['lr']

                if self.adam:
                    v1_temp = self.defaults['beta2'] * v1_temp + (1 - self.defaults['beta2']) * grad.pow(2)
                    bias_2 = 1 - self.defaults['beta2'] ** (self.k + 1)
                    v1_corrected = v1_temp / bias_2
                    lr = lr * math.sqrt(bias_2) / bias_1
                    lr = lr / (torch.sqrt(v1_corrected) + epi)

                grad = v0_corrected

                p = 1 / lr

                w_len = len(w.shape)
                if w_len == 4:  # Conv layer
                    self._handle_conv_layer(w, score_temp, wk_temp, vk_temp, zk_temp, grad, p)
                elif w_len == 2:  # Linear layer
                    self._handle_linear_layer(w, score_temp, wk_temp, vk_temp, zk_temp, grad, p)
                elif w_len == 1:  # Bias or BatchNorm
                    self._handle_bias_layer(w, score_temp, wk_temp, vk_temp, zk_temp, grad, p)

        self.k += 1
        return loss

    def _handle_conv_layer(self, w, score_temp, wk_temp, vk_temp, zk_temp, grad, p):
        """Handle convolutional layer neuron-wise pruning"""
        shape0, _, _, _ = w.shape

        # Update auxiliary variables (vectorized for all neurons)
        wk_new = wk_temp - vk_temp/p - grad/p
        b = wk_new + vk_temp/p
        
        # Expand score to match tensor dimensions if needed
        if len(score_temp.shape) == 1:
            score_temp = score_temp.view(shape0, 1, 1, 1).expand_as(w)
            
        u = (self.defaults['C']/self.defaults['N'])/p * torch.abs(score_temp)
        zk_new = soft_thresholding(b, u)

        # Update main variables
        vk_temp.add_((wk_new - zk_new) * p)
        wk_temp.copy_(wk_new)
        zk_temp.copy_(zk_new)
        w.copy_(zk_new)

    def _handle_linear_layer(self, w, score_temp, wk_temp, vk_temp, zk_temp, grad, p):
        """Handle linear layer neuron-wise pruning"""
        shape0, shape1 = w.shape

        # Update auxiliary variables (vectorized for all neurons)
        wk_new = wk_temp - vk_temp/p - grad/p
        b = wk_new + vk_temp/p
        
        # Expand score to match tensor dimensions if needed
        if len(score_temp.shape) == 1:
            score_temp = score_temp.view(shape0, 1).expand_as(w)
            
        u = (self.C/self.N)/self.lr * torch.abs(score_temp)
        zk_new = soft_thresholding(b, u)

        # Update main variables
        vk_temp.add_((wk_new - zk_new) * p)
        wk_temp.copy_(wk_new)
        zk_temp.copy_(zk_new)
        w.copy_(zk_new)

    def _handle_bias_layer(self, w, score_temp, wk_temp, vk_temp, zk_temp, grad, p):
        """Handle bias or batchnorm layer"""

        # Update auxiliary variables
        wk_new = wk_temp - vk_temp/p - grad/p
        b = wk_new + vk_temp/p
        u = (self.C/self.N)/p * torch.abs(score_temp)
        zk_new = soft_thresholding(b, u)

        # Update main variables
        vk_temp.add_((wk_new - zk_new) * p)
        wk_temp.copy_(wk_new)
        zk_temp.copy_(zk_new)
        w.copy_(zk_new)

    def update_base_learning_rate(self, new_lr):
        self.defaults['lr'] = new_lr