"""Neuron-wise percentile pruning (p = percentage to prune)."""
from torch.optim import Optimizer
import torch


class Ppercent_neuron(Optimizer):
    """
    Neuron-wise P-percent pruning with consistent semantics:
    - p means "prune p%" (keep = 1 - p/100).
    - Uses score tensors only for ranking; never multiplies scores into weights.
    - Uses topk to select kept weights for accurate ratios.
    """

    def __init__(self, params, lr, p, score):
        self.lr = lr
        self.p = p
        self.score = score
        super(Ppercent_neuron, self).__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            for w, score_temp in zip(group["params"], self.score):
                if w.grad is None:
                    continue

                if w.dim() == 4:
                    w.copy_(self.handle_conv_layer(w, score_temp, self.p))
                elif w.dim() == 2:
                    w.copy_(self.handle_linear_layer(w, score_temp, self.p))
                elif w.dim() == 1:
                    w.copy_(self.handle_bias_layer(w, score_temp, self.p))

        return None

    def handle_conv_layer(self, w, score_temp, p):
        shape0, a1, b1, c1 = w.shape
        grad = w.grad
        w_updated = w - grad * self.lr
        abs_scores = torch.abs(score_temp)

        abs_temp = abs_scores.view(shape0, -1)
        each_length = a1 * b1 * c1
        num_prune = int(each_length * (p / 100.0))
        num_keep = max(1, each_length - num_prune)

        _, indices = torch.topk(abs_temp, k=min(num_keep, each_length), dim=1, largest=True, sorted=False)
        mask = torch.zeros_like(abs_temp, dtype=w.dtype)
        for i in range(shape0):
            mask[i, indices[i]] = 1.0

        mask = mask.view(w.shape)
        return w_updated * mask

    def handle_linear_layer(self, w, score_temp, p):
        shape0, a1 = w.shape
        grad = w.grad
        w_updated = w - grad * self.lr
        abs_scores = torch.abs(score_temp)

        each_length = a1
        num_prune = int(each_length * (p / 100.0))
        num_keep = max(1, each_length - num_prune)

        _, indices = torch.topk(abs_scores, k=min(num_keep, each_length), dim=1, largest=True, sorted=False)
        mask = torch.zeros_like(abs_scores, dtype=w.dtype)
        for i in range(shape0):
            mask[i, indices[i]] = 1.0

        return w_updated * mask

    def handle_bias_layer(self, w, score_temp, p):
        grad = w.grad
        w_updated = w - grad * self.lr
        abs_scores = torch.abs(score_temp)

        total = w.numel()
        num_prune = int(total * (p / 100.0))
        num_keep = max(1, total - num_prune)

        flatten_scores = abs_scores.view(-1)
        _, indices = torch.topk(flatten_scores, k=min(num_keep, total), largest=True, sorted=False)

        mask = torch.zeros_like(flatten_scores, dtype=w.dtype)
        mask[indices] = 1.0
        mask = mask.view(w.shape)
        return w_updated * mask

__all__ = ["Ppercent_neuron"]