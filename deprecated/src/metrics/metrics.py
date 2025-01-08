import torch
import numpy as np
import matplotlib.pyplot as plt
import math 

def avgParam(model:torch.nn.Module): 
    out = []
    for param in model.parameters(): 
        out.append(torch.mean(param).item())
    return np.mean(out)

def L0_sparsity(model:torch.nn.Module): 
    total = 0 
    zeros = 0 
    for W in model.parameters(): 
        size = math.prod(W.size())
        total += size
        zeros += size - torch.count_nonzero(W).item()
        
    return zeros/total 

def parameterDistribution(model:torch.nn.Module): 
    vals = []
    for W in model.parameters(): 
        vals += list(W.cpu().data.detach().numpy().reshape(-1))
    plt.hist(vals, bins=np.linspace(-0.05, 0.05, 100)) 
    plt.show() 