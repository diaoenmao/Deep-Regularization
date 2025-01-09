from torch.optim import Optimizer
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import math

class P_Percent_Global(Optimizer):
    def __init__(self, params, model, lr, p_percent, score, beta, beta2, v0, v1, k, adam):
        self.defaults = dict(lr=lr, p_percent=p_percent, beta=beta, beta2=beta2)
        super(P_Percent_Global, self).__init__(params, self.defaults)
        self.model = model
        self.state['score'] = score
        
        # Initialize Adam-style momentum states
        self.state['v0'] = v0
        self.state['v1'] = v1
        self.state['k'] = k  # Step counter for bias correction
        self.adam = adam

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Get gradients and weights as vectors
        grad = parameters_to_vector([param.grad for param in self.model.parameters()])
        w = parameters_to_vector([param.data for param in self.model.parameters()])

        w_temp = w.clone()
        
        # Adam-style updates
        epi = 1e-8
        self.state['v0'] = self.defaults['beta'] * self.state['v0'] + (1 - self.defaults['beta']) * grad
        bias_correction1 = 1 - self.defaults['beta'] ** (self.state['k'] + 1)
        v0_corrected = self.state['v0'] / bias_correction1

        if self.adam:
            self.state['v1'] = self.defaults['beta2'] * self.state['v1'] + (1 - self.defaults['beta2']) * grad.pow(2)

            # Bias correction
            bias_correction2 = 1 - self.defaults['beta2'] ** (self.state['k'] + 1)
        
            v1_corrected = self.state['v1'] / bias_correction2

        # Compute adaptive learning rate
        lr = self.defaults['lr'] * math.sqrt(bias_correction2) / bias_correction1
        lr = lr / (torch.sqrt(v1_corrected) + epi)
        
        p = 1 / lr
        grad = v0_corrected

        # Apply importance scores and gradient update
        w_temp = torch.mul(w_temp, self.state['score'])

        # Calculate threshold for p-percent pruning
        abs_scores = torch.abs(w_temp)

        w = w - grad / p
        
        k = int(len(w) * (self.defaults['p_percent'] / 100.0))  # number of weights to prune
        if k > 0:  # only prune if k > 0
            threshold = torch.kthvalue(abs_scores, k).values

            # Create pruning mask (1 for keep, 0 for prune)
            mask = torch.where(abs_scores > threshold, 1.0, 0.0)

            # Apply mask to weights
            w = w * mask

        # Update model parameters
        vector_to_parameters(w, self.model.parameters())
        
        self.state['k'] += 1
        return loss

    def update_base_learning_rate(self, new_lr):
        self.defaults['lr'] = new_lr
