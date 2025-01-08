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
                lora_scores[name] = param.grad.detach()
            else:
                lora_scores[name] = torch.ones_like(param)
        return lora_scores
