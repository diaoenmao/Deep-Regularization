import torch
import numpy as np 
import math
from torch.nn import Module
from torch import Tensor
import matplotlib.pyplot as plt

def p_norm(model:Module, device:str, p:float, normalize:bool = True) -> float: 
  p_norm = Tensor([0.]).to(device)
  N = 0
  for W in model.parameters(): 
    p_norm += W.norm(p) ** p

    if normalize: 
      N += math.prod(W.size())

  p_norm = p_norm ** (1 / p)

  if normalize and N > 0:
    p_norm /= N

  return p_norm.cpu().detach().item()
 
def PQI(model:Module, device:str, p:float, q:float) -> float:
    
  d = sum(math.prod(param.size()) for param in model.parameters())
  pq = d ** ((1/q) - (1/p)) * (p_norm(model, device, p, normalize=False)/p_norm(model, device, q, normalize=False))
  
  return 1 - pq

def avgParam(model:Module) -> float: 
  out = []
  for param in model.parameters(): 
    out.append(torch.mean(param).item())
  return np.mean(out).item()

def L0_sparsity(model:Module) -> float: 
  total = 0 
  zeros = 0 
  for W in model.parameters(): 
    size = math.prod(W.size())
    total += size
    zeros += size - torch.count_nonzero(W).item()
      
  return zeros/total 

def parameterDistribution(model:Module): 
  vals = []
  for W in model.parameters(): 
    vals += list(W.cpu().data.detach().numpy().reshape(-1))
  plt.hist(vals, bins=list(np.linspace(-0.05, 0.05, 100)))
  plt.show() 

