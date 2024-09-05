import torch
import numpy as np

def create_mask(model):
    mask = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask[name] = torch.ones_like(param)
    return mask

def prune_by_percentile(model, mask, prune_percent):
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]
            percentile_value = np.percentile(abs(alive), prune_percent)
            
            current_mask = mask[name].cpu().numpy()
            
            new_mask = np.where(abs(tensor) < percentile_value, 0, current_mask)
            pruned = np.sum(new_mask == 0) - np.sum(current_mask == 0)
            print(f"Layer {name}: pruned {pruned} weights")
            
            mask[name] = torch.from_numpy(new_mask).to(param.device)
    return mask

def apply_mask(model, mask):
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data.mul_(mask[name])
    return model