from torch.optim import Optimizer
import torch
import math

class P_Percent_Layer(Optimizer):
    def __init__(self, params, lr, p_percent, score, model, beta, beta2, v0, v1, k):
        self.lr = lr
        self.p_percent = p_percent  # percentage of weights to prune per layer (0-100)
        self.score = score  # importance scores per layer
        self.model = model
        self.beta = beta
        self.beta2 = beta2
        self.v0 = v0
        self.v1 = v1
        self.k = k
        super(P_Percent_Layer, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_weights = 0
        remaining_weights = 0

        for group in self.param_groups:
            for w, score_temp in zip(group['params'], self.score):
                if w.grad is None:
                    continue
                
                w_temp = w.data.clone()

                w_temp = w_temp * score_temp
                # Get absolute values of the weights directly
                abs_weights = torch.abs(w_temp)

                # Calculate number of weights to prune in this layer
                k = int(w.numel() * (self.p_percent / 100.0))

                if k > 0:  # only prune if k > 0
                    # Find threshold for this layer
                    threshold = torch.kthvalue(abs_weights.view(-1), k).values

                    # Create pruning mask (1 for keep, 0 for prune)
                    mask = (abs_weights > threshold).float()

                    # Apply mask to original weights
                    w.data.mul_(mask)  # Use mul_ instead of direct assignment
                    
                    # Count remaining weights (non-zero weights after pruning)
                    remaining_weights += torch.count_nonzero(w.data).item()
                else:
                    remaining_weights += w.numel()

        return loss

    def update_base_learning_rate(self, new_lr):
        self.lr = new_lr

    def update_p_percent(self, new_p_percent):
        """Update the pruning percentage"""
        self.p_percent = new_p_percent
