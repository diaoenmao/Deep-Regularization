from torch.optim import Optimizer
import torch
import math

class P_Percent_Layer(Optimizer):
    def __init__(self, params, lr, p_percent, score, model, beta, beta2, v0, v1, k, adam):
        self.defaults = dict(lr=lr, p_percent=p_percent, beta=beta, beta2=beta2)
        self.score = score  # importance scores per layer
        self.model = model
        self.v0 = v0
        self.v1 = v1
        self.k = k
        self.adam = adam
        super(P_Percent_Layer, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        for group in self.param_groups:
            for w, score_temp, v0_temp, v1_temp in zip(group['params'], self.score, self.v0, self.v1):
                if w.grad is None:
                    continue

                w_temp = w.clone()

                grad = w.grad
                epi = 1e-8
                lr = self.defaults['lr']

                v0_temp = self.defaults['beta'] * v0_temp + (1 - self.defaults['beta']) * grad
                bias_1 = 1 - self.defaults['beta'] ** (self.k + 1)
                v0_corrected = v0_temp / bias_1

                if self.adam:
                    v1_temp = self.defaults['beta2'] * v1_temp + (1 - self.defaults['beta2']) * grad.pow(2)
                    bias_2 = 1 - self.defaults['beta2'] ** (self.k + 1)
                    v1_corrected = v1_temp / bias_2
                    lr = lr * math.sqrt(bias_2) / bias_1
                    lr = lr / (torch.sqrt(v1_corrected) + epi)

                grad = v0_corrected

                p = 1 / lr

                w_temp = torch.mul(w_temp, score_temp)
                abs_scores = torch.abs(w_temp)

                w = w - grad / p

                # Calculate number of weights to prune in this layer
                k = int(w.numel() * (self.p_percent / 100.0))

                if k > 0:  # only prune if k > 0
                    # Find threshold for this layer
                    threshold = torch.kthvalue(abs_scores.view(-1), k).values

                    # Create pruning mask (1 for keep, 0 for prune)
                    mask = (abs_scores > threshold).float()

                    w_new = mask * w

                    w.copy_(w_new)

        self.k += 1
        return loss

    def update_base_learning_rate(self, new_lr):
        self.defaults['lr'] = new_lr
