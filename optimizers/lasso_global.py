from torch.optim import Optimizer
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class LASSO_Global(Optimizer):

    def __init__(self, params, lr, N, C, score, model):
        self.lr = lr
        self.N = N #NUMBER OF SAMPLE
        self.C = C #CONSTANT
        self.model = model
        
        # Convert score list to flattened vector and move to correct device
        device = next(model.parameters()).device
        score_vector = []
        for s in score:
            score_vector.append(s.view(-1).to(device))
        self.score = torch.cat(score_vector)
        
        super(LASSO_Global, self).__init__(params, {})

    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad = parameters_to_vector([param.grad for param in self.model.parameters()])
        w = parameters_to_vector([param.data for param in self.model.parameters()])

        # Apply gradient step and score
        w = w * self.score - self.lr * grad

        w_copy = w.clone()

        mask1 = torch.where(w > 0, 1, 0.0)
        mask2 = torch.where(w < 0, 1, 0.0)
        w = mask1 * (w - (self.C / self.N) * self.lr) + mask2 * (w + (self.C / self.N) * self.lr)
        w = (torch.where(abs(w - w_copy) < abs(w_copy), 1, 0)) * w
        
        vector_to_parameters(w, self.model.parameters())
        
        return loss
    
    def update_base_learning_rate(self, new_lr):
        self.lr = new_lr
