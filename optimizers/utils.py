import torch

def soft_thresholding(b, u):
    z = torch.sign(b) * torch.max(torch.zeros_like(b), torch.abs(b)-u)

    return z