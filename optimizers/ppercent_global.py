from torch.optim import Optimizer
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class P_Percent_Global(Optimizer):
    def __init__(self, params, model, lr, p_percent, score):
        self.model = model
        self.lr = lr
        self.p_percent = p_percent  # percentage of weights to prune (0-100)
        self.score = score  # importance scores
        super(P_Percent_Global, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Get gradients and weights as vectors
        grad = parameters_to_vector([param.grad for param in self.model.parameters()])
        w = parameters_to_vector([param.data for param in self.model.parameters()])

        # Standard gradient step
        w = w - self.lr * grad

        # Apply importance scores
        w = w * self.score

        # Calculate threshold for p-percent pruning
        abs_weights = torch.abs(w)
        k = int(len(w) * (self.p_percent / 100.0))  # number of weights to prune
        if k > 0:  # only prune if k > 0
            threshold = torch.kthvalue(abs_weights, k).values

            # Create pruning mask (1 for keep, 0 for prune)
            mask = torch.where(abs_weights > threshold, 1.0, 0.0)

            # Apply mask to weights
            w = w * mask

        # Update model parameters
        vector_to_parameters(w, self.model.parameters())

        return loss

    def update_base_learning_rate(self, new_lr):
        self.lr = new_lr

    def update_p_percent(self, new_p_percent):
        """Update the pruning percentage"""
        self.p_percent = new_p_percent
