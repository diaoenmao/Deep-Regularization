from torch.optim import Optimizer
import torch
import math

class Ppercent_layer(Optimizer):
    def __init__(self, params, lr, p, score):
        self.lr = lr
        self.p = p  # percent to prune in each layer
        self.score = score
        super(Ppercent_layer, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for w, score_temp in zip(group['params'], self.score):
                grad = w.grad

                w_updated = w - grad * self.lr

                # ✅ FIX P0: 删除score与w_updated相乘
                abs_scores = torch.abs(score_temp)

                norm_abs_scores = abs_scores
                total = w.numel()
                k = max(1, int(total * (self.p / 100.0)))
                flatten_scores = norm_abs_scores.view(-1)

                if k > 0:
                    threshold = torch.kthvalue(flatten_scores, k).values

                    # ✅ FIX P1: 改> 为 >=
                    mask = torch.where(norm_abs_scores >= threshold, 1.0, 0.0)
                    w_new = w_updated * mask
                    w.copy_(w_new)
        return loss


    __all__ = ["Ppercent_layer"]

