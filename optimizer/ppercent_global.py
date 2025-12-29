from torch.optim import Optimizer
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import math


class Ppercent_global(Optimizer):
    def __init__(self, params, lr,  p, score):
        self.lr = lr
        self.p = p  # CONSTANT
        self.score = score
        super(Ppercent_global, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Get gradients and weights as vectors
        grad = parameters_to_vector([p.grad for p in self.param_groups[0]['params']])
        score_temp = parameters_to_vector(self.score)
        w = parameters_to_vector(self.param_groups[0]['params'])
        w = w - grad * self.lr
        
        # ✅ FIX: Use score directly for ranking (score already contains weight info for first-order)
        # Don't multiply by w again - that would make it score * w * w
        abs_scores = torch.abs(score_temp)

        k = int(len(w) * (self.p / 100.0))  # number of weights to prune
        if k > 0:  # only prune if k > 0
            threshold = torch.kthvalue(abs_scores, k).values

            # Create pruning mask (1 for keep, 0 for prune)
            mask = torch.where(abs_scores > threshold, 1.0, 0.0)

            # Apply mask to weights
            w = w * mask

        # Update model parameters
        vector_to_parameters(w, self.param_groups[0]['params'])

        return loss


    __all__ = ["Ppercent_global"]