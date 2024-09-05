import torch
from torch.optim import Optimizer
import math

def soft_thresholding(x, threshold):
    return torch.sign(x) * torch.max(torch.abs(x) - threshold, torch.zeros_like(x))

class ADMM_Adam_neuron(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                weight_decay=0, amsgrad=False, N=600, C=0.5):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(ADMM_Adam_neuron, self).__init__(params, defaults)
        
        self.N = N
        self.C = C
        
        # Initialize ADMM variables
        self.vk = []
        self.wk = []
        self.yk = []
        self.zk = []
        for group in self.param_groups:
            for p in group['params']:
                self.vk.append(torch.zeros_like(p))
                self.wk.append(torch.zeros_like(p))
                self.yk.append(p.clone().detach())
                self.zk.append(p.clone().detach())

    def __setstate__(self, state):
        super(ADMM_Adam_neuron, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group['amsgrad']:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                # ADMM neuron-wise pruning
                if len(p.shape) == 4:  # Convolutional layer
                    p.data, self.vk[i], self.yk[i], self.zk[i], self.wk[i] = self.CNN_neuronwise_pruning(
                        p.data, self.vk[i], self.yk[i], self.zk[i], self.wk[i],
                        exp_avg, denom, step_size, self.N, self.C
                    )
                elif len(p.shape) == 2:  # Fully connected layer
                    p.data, self.vk[i], self.yk[i], self.zk[i], self.wk[i] = self.fullycont(
                        p.data, self.vk[i], self.yk[i], self.zk[i], self.wk[i],
                        exp_avg, denom, step_size, self.N, self.C
                    )
                elif len(p.shape) == 1:  # Bias or BatchNorm
                    p.data, self.vk[i], self.yk[i], self.zk[i], self.wk[i] = self.batchnorm_and_bias_pruning(
                        p.data, self.vk[i], self.yk[i], self.zk[i], self.wk[i],
                        exp_avg, denom, step_size, self.N, self.C
                    )

        return loss

    def CNN_neuronwise_pruning(self, w, vk, yk, zk, wk, exp_avg, denom, lr, N, C):
        shape0, shape1, shape2, shape3 = w.shape
        p = 1 / lr
        grad = exp_avg / denom

        qk = 0.5 * (yk + zk - vk / p - wk / p - grad / p)

        ck = torch.norm(zk.view(shape0, -1), p=1, dim=1).view(shape0, 1, 1, 1).expand_as(w)
        dk = qk + vk / p
        yita = torch.norm(dk.view(shape0, -1), p=2, dim=1).view(shape0, 1, 1, 1).expand_as(w)
        D_k = torch.div(C / N * ck, p * (yita ** 3))
        C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** 0.5) / 2) ** (1 / 3)
        tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

        if torch.all(dk == 0):
            fangsuo = (ck / p) ** (1 / 3)
            random_tensor = torch.randn_like(yk)
            yk = random_tensor * (fangsuo / torch.norm(random_tensor.view(shape0, -1), p=2, dim=1).view(shape0, 1, 1, 1).expand_as(w))
        else:
            yk = torch.mul(tao_k, dk)

        b = qk + wk / p
        yk_norm = torch.norm(yk.view(shape0, -1), p=2, dim=1).view(shape0, 1, 1, 1).expand_as(w)
        u = (C / N) / (p * yk_norm)
        zk = soft_thresholding(b, u)

        vk = vk + p * (qk - yk)
        wk = wk + p * (qk - zk)
        w = zk

        return w, vk, yk, zk, wk

    def fullycont(self, w, vk, yk, zk, wk, exp_avg, denom, lr, N, C):
        shape0, shape1 = w.shape
        p = 1 / lr
        grad = exp_avg / denom

        qk = 0.5 * (yk + zk - vk / p - wk / p - grad / p)

        ck = torch.norm(zk, p=1, dim=1).view(shape0, 1).expand_as(w)
        dk = qk + vk / p
        yita = torch.norm(dk, p=2, dim=1).view(shape0, 1).expand_as(w)
        D_k = torch.div(C / N * ck, p * (yita ** 3))
        C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** 0.5) / 2) ** (1 / 3)
        tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

        if torch.all(dk == 0):
            fangsuo = (ck / p) ** (1 / 3)
            random_tensor = torch.randn_like(yk)
            yk = random_tensor * (fangsuo / torch.norm(random_tensor, p=2, dim=1).view(shape0, 1).expand_as(w))
        else:
            yk = torch.mul(tao_k, dk)

        b = qk + wk / p
        yk_norm = torch.norm(yk, p=2, dim=1).view(shape0, 1).expand_as(w)
        u = (C / N) / (p * yk_norm)
        zk = soft_thresholding(b, u)

        vk = vk + p * (qk - yk)
        wk = wk + p * (qk - zk)
        w = zk

        return w, vk, yk, zk, wk

    def batchnorm_and_bias_pruning(self, w, vk, yk, zk, wk, exp_avg, denom, lr, N, C):
        p = 1 / lr
        grad = exp_avg / denom

        qk = 0.5 * (yk + zk - vk / p - wk / p - grad / p)

        ck = torch.norm(zk, p=1)
        dk = qk + vk / p
        yita = torch.norm(dk, p=2)
        D_k = (C / N * ck) / (p * (yita ** 3))
        C_K = ((27 * D_k + 2 + ((27 * D_k + 2) ** 2 - 4) ** 0.5) / 2) ** (1 / 3)
        tao_k = 1 / 3 + (1 / 3) * (C_K + 1 / C_K)

        if torch.all(dk == 0):
            fangsuo = (ck / p) ** (1 / 3)
            random_tensor = torch.randn_like(yk)
            yk = random_tensor * (fangsuo / torch.norm(random_tensor, p=2))
        else:
            yk = tao_k * dk

        b = qk + wk / p
        u = (C / N) / (p * torch.norm(yk, p=2))
        zk = soft_thresholding(b, u)

        vk = vk + p * (qk - yk)
        wk = wk + p * (qk - zk)
        w = zk

        return w, vk, yk, zk, wk