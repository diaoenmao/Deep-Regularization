import torch
from .utils import GradientCollector

# weight * gradient

class LoraScore:
    def __init__(self, model):
        self.model = model

    def compute_lora_scores(self):
        lora_scores = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
            else:
                grad = torch.ones_like(param)
            lora_scores[name] = grad
        return lora_scores
