from torch.optim import Optimizer
import torch

class LASSO_Layer(Optimizer):
    def __init__(self, params, lr, N, C, score, model):
        self.lr = lr
        self.N = N  # NUMBER OF SAMPLES
        self.C = C  # REGULARIZATION CONSTANT
        self.score = score 
        self.model = model
        super(LASSO_Layer, self).__init__(params, {})

    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for w, score_temp in zip(group['params'], self.score):
                if w.grad is None:
                    continue

                w.data = w.data * score_temp - self.lr * w.grad

                w_copy = w.data
                mask1 = torch.where(w.data > 0, 1, 0.0)
                mask2 = torch.where(w.data < 0, 1, 0.0)
                w.data = mask1 * (w.data - (self.C / self.N) * self.lr) + mask2 * (w.data + (self.C / self.N) * self.lr)
                w.data = (torch.where(abs(w.data - w_copy) < abs(w_copy), 1, 0)) * w.data

        return loss


    def update_base_learning_rate(self, new_lr):
        self.lr = new_lr