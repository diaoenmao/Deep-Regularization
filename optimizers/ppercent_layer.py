from torch.optim import Optimizer
import torch

class P_Percent_Layer(Optimizer):
    def __init__(self, params, lr, p_percent, score):
        self.lr = lr
        self.p_percent = p_percent  # percentage of weights to prune per layer (0-100)
        self.score = score  # importance scores per layer
        super(P_Percent_Layer, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for w, score_temp in zip(group['params'], self.score):
                if w.grad is None:
                    continue

                # Standard gradient step
                w.data = w.data - self.lr * w.grad

                # Apply importance scores
                w.data = w.data * score_temp

                # Get absolute values
                abs_weights = torch.abs(w.data)

                # Calculate number of weights to prune in this layer
                k = int(w.data.numel() * (self.p_percent / 100.0))

                if k > 0:  # only prune if k > 0
                    # Find threshold for this layer
                    threshold = torch.kthvalue(abs_weights.view(-1), k).values

                    # Create pruning mask (1 for keep, 0 for prune)
                    mask = torch.where(abs_weights > threshold, 1.0, 0.0)

                    # Apply mask to weights
                    w.data = w.data * mask

        return loss

    def update_base_learning_rate(self, new_lr):
        self.lr = new_lr

    def update_p_percent(self, new_p_percent):
        """Update the pruning percentage"""
        self.p_percent = new_p_percent
