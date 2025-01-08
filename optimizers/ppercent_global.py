from torch.optim import Optimizer
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import math

class P_Percent_Global(Optimizer):
    def __init__(self, params, model, lr, p_percent, score, beta, beta2, v0, v1, k):
        self.defaults = dict(lr=lr, p_percent=p_percent, beta=beta, beta2=beta2)
        super(P_Percent_Global, self).__init__(params, self.defaults)
        self.model = model
        self.state['score'] = score
        
        # Initialize Adam-style momentum states
        self.state['v0'] = v0
        self.state['v1'] = v1
        self.state['k'] = k  # Step counter for bias correction

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Get gradients and weights as vectors
        grad = parameters_to_vector([param.grad for param in self.model.parameters()])
        w = parameters_to_vector([param.data for param in self.model.parameters()])

        # Clip gradients for stability
        grad = torch.clamp(grad, min=-1.0, max=1.0)
        
        # Adam-style updates
        epi = 1e-8
        self.state['v0'] = self.defaults['beta'] * self.state['v0'] + (1 - self.defaults['beta']) * grad
        self.state['v1'] = self.defaults['beta2'] * self.state['v1'] + (1 - self.defaults['beta2']) * grad.pow(2)

        # Bias correction
        bias_correction1 = 1 - self.defaults['beta'] ** (self.state['k'] + 1)
        bias_correction2 = 1 - self.defaults['beta2'] ** (self.state['k'] + 1)
        
        v0_corrected = self.state['v0'] / bias_correction1
        v1_corrected = self.state['v1'] / bias_correction2

        # Compute adaptive learning rate
        lr = self.defaults['lr'] * math.sqrt(bias_correction2) / bias_correction1
        lr = lr / (torch.sqrt(v1_corrected) + epi)
        
        # Clip learning rate
        lr = torch.clamp(lr, min=1e-8, max=1.0)

        # Apply importance scores and gradient update
        w = torch.mul(w, self.state['score'])
        w = w - v0_corrected * lr

        # Calculate threshold for p-percent pruning
        abs_weights = torch.abs(w)
        k = int(len(w) * (self.defaults['p_percent'] / 100.0))  # number of weights to prune
        if k > 0:  # only prune if k > 0
            threshold = torch.kthvalue(abs_weights, k).values

            # Create pruning mask (1 for keep, 0 for prune)
            mask = torch.where(abs_weights > threshold, 1.0, 0.0)

            # Apply mask to weights
            w = w * mask

        # Update model parameters
        vector_to_parameters(w, self.model.parameters())
        
        self.state['k'] += 1
        return loss

    def update_base_learning_rate(self, new_lr):
        self.defaults['lr'] = new_lr

    def update_p_percent(self, new_p_percent):
        """Update the pruning percentage"""
        self.defaults['p_percent'] = new_p_percent