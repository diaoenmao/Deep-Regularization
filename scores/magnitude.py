import torch

class MagnitudeScore:
    def __init__(self, model):
        self.model = model

    def compute_magnitude_scores(self):
        scores = {}
        for name, param in self.model.named_parameters():
            scores[name] = torch.ones_like(param)
        return scores
