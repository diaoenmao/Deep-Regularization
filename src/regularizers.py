import torch 
import math
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F 
from torch.optim import Optimizer, SGD, LBFGS

from abc import ABC, abstractmethod


class Regularizer(ABC): 
  """
  All subclasses of Regularizer reprsents certain method of regularization. 
  They can all be initialized in their own way, but they must have
    - the model attribute
    - the device attribute 
    - some set of hyperparameters (if needed) 
    - a step() method 
  Essentially, during training, the step method will be called, which will 
  add an extra step to the backpropagation according to this regularizer 
  on self.model existing on self.device. The step function should not 
  accept any hyperparameters as arguments. All hyperparameters must be 
  stored as attributes. 
  """

  @abstractmethod 
  def step(self): 
    pass 


def make_regularizer(reg:str, model:Module, device:str, **kwargs) -> Regularizer :
  if reg == "none": 
    return NoReg(model, device)
  elif reg == "l1softthreshold": 
    return L1_SoftThreshold(model, device, kwargs["lmbda"], kwargs["tau"]) 
  elif reg == "l1proximal": 
    return L1_Proximal(model, device, kwargs["lmbda"], kwargs["tau"], 
                       kwargs["reg_optimizer"], kwargs["reg_initialization"], 
                       kwargs["clipping_scale"], kwargs["line_crossing"])
  elif reg == "l1admm": 
    '''TODO'''
    raise NotImplementedError("Not implemented yet") 
    # return L1_ADMM(model) 
  elif reg == "l1sgd_naive": 
    return L1_SGD_Naive(model, kwargs["optimizer"], device, kwargs["lmbda"]) 
  elif reg == "l2": 
    return L2_Regularizer(model, kwargs["optimizer"], device, kwargs["lmbda"]) 
  elif reg == "pqiproximal": 
    return PQI_Proximal(model, device, kwargs["p"], kwargs["q"], kwargs["lmbda"], 
                        kwargs["tau"], kwargs["reg_optimizer"], 
                        kwargs["reg_initialization"], kwargs["clipping_scale"], 
                        kwargs["line_crossing"])
  elif reg == "pqiadmm": 
    "TODO"
    raise NotImplementedError("Not implemented yet")
  else: 
    raise Exception("Not a valid type.")

class NoReg(Regularizer): 

  def __init__(self, model:Module, device:str): 
    self.model = model 
    self.device = device 
    self.penalty = torch.Tensor([1.0]).to(device)

  def step(self) -> None: 
    pass

class L1_SoftThreshold(Regularizer): 
  """
  Implementation of the L1 threshold regularizer
  """    
  def __init__(self, model:Module, device:str, lmbda:Tensor, tau:Tensor): 
    """
    Arguments 
    lmbda - strength of L1 regularization
    tau - step size of our gradient step 
    """ 
    self.model = model 
    self.device = device
    self.lmbda = lmbda.to(device)
    self.tau = tau.to(device)
    self.penalty = self.lmbda * p_norm(self.model, self.device, 1)
  
  def soft_threshold(self, param:Tensor, lmbda:Tensor) -> Tensor: 
    """
    Helper function that takes in parameter tensor and does the soft_threshold 
    function on each element. Outputs the tensor.  
    """
    x = torch.sign(param) * torch.max(torch.abs(param) - lmbda, torch.zeros_like(param))
    return x
  
  def step(self) -> None: 
    with torch.no_grad(): 
      # updating parameters with gradient data, so you should remove gradient information. 
      for name, param in self.model.named_parameters(): 
        if "bias" not in name:
          # only focus on non-bias terms
          # update the parameters of the model in place with copy_() method 
          param.copy_(self.soft_threshold(param, self.tau * torch.tensor(self.lmbda)))

    self.penalty = self.lmbda * p_norm(self.model, self.device, 1)

class L1_Proximal(Regularizer): 
    
  def __init__(self, model:Module, device:str, lmbda:Tensor, tau:Tensor, 
               optimizer:str, initialization:str = "inplace", 
               clipping_scale:float = 1.0, line_crossing:bool = False): 
    """
    Arguments 
    optimizer - which optimizer to use for optimizing the argmin of the proximal step (LBFGS or batch GD)
    initialization - where we initialize the model parameters before optimization ("inplace", "rand", "zeros")
    clipping - the threshold to clip all param values to 0 
    line_crossing - whether we should clip all values that change signs during each proximal optimizer step
    """
    
    self.lmbda = lmbda.to(device)
    self.tau = tau.to(device)
    self.model = model 
    self.device = device
    self.penalty = self.lmbda * p_norm(self.model, self.device, 1)
    
    if optimizer in ["SGD", "GD"]: 
      self.optimizer = SGD(model.parameters(), lr=(lmbda*tau*0.1).item(), momentum=0.0)
    elif optimizer in ["LBFGS"]: 
      self.optimizer = LBFGS(model.parameters(), lr=(lmbda*tau*0.1).item(), tolerance_change=1e-6)
    else: 
      raise Exception("Not a valid optimizer. ")
    
    if initialization in ["inplace", "zeros", "rand"]: 
      self.initialization = initialization 
    else: 
      raise Exception("Not a valid initialization. ")
    
    self.clipping_scale = clipping_scale
    self.line_crossing = line_crossing 

  def initialize(self): 
    if self.initialization == "inplace": 
      pass 
    elif self.initialization == "rand": 
      with torch.no_grad(): 
        for _, param in self.model.named_parameters(): 
          param.copy_(0.01 * torch.randn_like(param))
    elif self.initialization == "zeros": 
      with torch.no_grad(): 
        for _, param in self.model.named_parameters(): 
          param.copy_(0. * param)

  def step(self): 
      
    u = [W.data.detach().clone() for W in self.model.parameters()]
    
    def objective_function(model): 
      one_norm = torch.norm(torch.stack([torch.norm(param, 1) for param in model.parameters()]), 1)
      # print("one_norm", one_norm.item())
      l2_diff = torch.norm(torch.stack([torch.norm([W for W in model.parameters()][i] - u[i]) for i in range(len(u))]))
      # print("l2_diff", l2_diff.item() ** 2)
      output = 1/(2*self.tau) * l2_diff ** 2 + self.lmbda * one_norm 
      return output.to(self.device)
    
    self.initialize()
    
    def closure():
      self.optimizer.zero_grad()
      loss = objective_function(self.model)
      loss.backward()
      return loss
    
    for i in range(500): 
      positives1 = []
      negatives1 = []

        
      if self.line_crossing is True: 
        for W in self.model.parameters(): 
          mask1 = torch.where(W.data >= 0, 1.0, 0.0)
          mask2 = torch.where(W.data <= 0, 1.0, 0.0)
          positives1.append(mask1) 
          negatives1.append(mask2)
          
      self.optimizer.step(closure)
      
      if self.line_crossing is True: 
        positives2 = [] 
        negatives2 = [] 
        for W in self.model.parameters(): 
          mask1 = torch.where(W.data >= 0, 1.0, 0.0)
          mask2 = torch.where(W.data <= 0, 1.0, 0.0)
          positives2.append(mask1) 
          negatives2.append(mask2)

        clip_mask = [] 
        for i in range(len(positives1)): 
          clip = positives1[i] * positives2[i] + negatives1[i] * negatives2[i]
          clip_mask.append(clip)

        with torch.no_grad(): 
          i = 0
          for _, param in self.model.named_parameters(): 
            param.copy_(clip_mask[i] * param)
            i += 1 

    with torch.no_grad(): 
      for _, param in self.model.named_parameters(): 
        clipped_param = param.clone()
        clipped_param[torch.logical_and(clipped_param >-self.clipping_scale * self.lmbda*self.tau, clipped_param < self.clipping_scale * self.lmbda*self.tau)] = 0.0
        param.copy_(clipped_param)

    self.penalty = self.lmbda * p_norm(self.model, self.device, 1)

class L1_ADMM(Regularizer): 
    
  def __init__(self, model:Module, device:str, lmbda:float, rho:float, 
               initialization:str = "inplace"): 
    self.model = model 
    self.device = device 
    self.lmbda = lmbda 
    self.rho = rho

    if initialization in ["inplace", "zeros", "rand"]: 
      self.initialization = initialization 
    else: 
      raise Exception("Not a valid initialization. ")

  def initialize(self): 
      
    if self.initialization == "inplace": 
      pass 
    elif self.initialization == "rand": 
      with torch.no_grad(): 
        for _ , param in self.model.named_parameters(): 
          param.copy_(0.01 * torch.randn_like(param))
    elif self.initialization == "zeros": 
      with torch.no_grad(): 
        for _, param in self.model.named_parameters(): 
          param.copy_(0. * param)

  def step(self):
    # ADMM consists of primal variable update (x), auxiliary variable update (z), and dual variable update (u)
    z_old = {name: val.clone().detach() for name, val in self.model.named_parameters()}  # Save old z for dual variable update
    u = {name: torch.zeros_like(val) for name, val in self.model.named_parameters()}  # Initialize dual variables


    # Primal variable update (typically a step of gradient descent or other optimization on the original objective)
    # Assuming an optimizer is defined outside of this function, and a single step is performed before this function is called
    # optimizer.step()

    # Auxiliary variable update (L1 regularization applied here)
    with torch.no_grad():
      for name, param in self.model.named_parameters():
        # Soft thresholding for L1 regularization
        z = param + u[name]  # Combine parameter and dual variable
        z_abs = torch.abs(z)
        z_sign = torch.sign(z)
        z[name] = z_sign * F.relu(z_abs - self.lmbda / self.rho)

    # Dual variable update
    with torch.no_grad():
      for name, param in self.model.named_parameters():
        u[name] += param - z_old[name]

class L1_SGD_Naive(Regularizer): 
    
  '''Does an extra gradient step on the regularization term and clips all values that cross 0'''
  
  def __init__(self, model:Module, optimizer:Optimizer, device:str, lmbda:float): 
    self.model = model 
    self.optimizer = optimizer
    self.device = device 
    self.lmbda = float(lmbda)
    self.penalty = self.lmbda * p_norm(self.model, self.device, 1)

  def step(self): 
    positives1 = []
    negatives1 = []
    for W in self.model.parameters(): 
      mask1 = torch.where(W.data > 0, 1.0, 0.0)
      mask2 = torch.where(W.data < 0, 1.0, 0.0)
      positives1.append(mask1) 
      negatives1.append(mask2)

    reg_loss = self.lmbda * p_norm(self.model, self.device, 1, normalize=True)
    reg_loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    
    positives2 = [] 
    negatives2 = [] 
    for W in self.model.parameters(): 
      mask1 = torch.where(W.data > 0, 1.0, 0.0)
      mask2 = torch.where(W.data < 0, 1.0, 0.0)
      positives2.append(mask1) 
      negatives2.append(mask2)
  
    clip_mask = [] 
    for i in range(len(positives1)): 
      clip = positives1[i] * positives2[i] + negatives1[i] * negatives2[i]
      clip_mask.append(clip)
    
    with torch.no_grad(): 
      i = 0
      for _, param in self.model.named_parameters(): 
        param.copy_(clip_mask[i] * param)
        i += 1 

    self.penalty = self.lmbda * p_norm(self.model, self.device, 1)

class L2_Regularizer(Regularizer): 
    
  def __init__(self, model:Module, optimizer:Optimizer, device:str, lmbda:Tensor): 
      
    self.model = model 
    self.optimizer = optimizer
    self.device = device 
    self.lmbda = lmbda 
    self.penalty = self.lmbda * p_norm(self.model, self.device, 2)

  def step(self): 
    reg_loss = self.lmbda * p_norm(self.model, self.device, 2, normalize=True)
    reg_loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    
    self.penalty = self.lmbda * p_norm(self.model, self.device, 2)

class PQI_Proximal(Regularizer): 

  def __init__(self, model:Module, device:str, p:float, q:float, lmbda:float, 
               tau:float, optimizer:str, initialization:str = "inplace", 
               clipping_scale:float = 1.0, line_crossing:bool = False): 
    """
    Arguments 
    model - our model 
    optimizer - which optimizer to use for optimizing the argmin of the proximal step (LBFGS, or batch GD)
    initialization - where we initialize the model parameters before optimization ("inplace", "rand", "zeros")
    clipping - the threshold to clip all param values to 0 
    line_crossing - whether we should clip all values that change signs during each proximal optimizer step
    """
    
    self.p = float(p) 
    self.q = float(q)
    if p > q: 
      raise Exception("p should be less than q. ")
    
    self.lmbda = float(lmbda) 
    self.tau = float(tau) 
    self.model = model 
    self.device = device
    self.penalty = self.lmbda * PQI(self.model, self.device, self.p, self.q)
    
    if optimizer in ["SGD", "GD"]: 
      self.optimizer = SGD(model.parameters(), lr=lmbda*tau * 0.1, momentum=0.0)
    elif optimizer in ["LBFGS"]: 
      self.optimizer = LBFGS(model.parameters(), lr=lmbda*tau * 0.1, tolerance_change=1e-6)
    else: 
      raise Exception("Not a valid optimizer. ")
    
    if initialization in ["inplace", "zeros", "rand"]: 
      self.initialization = initialization 
    else: 
      raise Exception("Not a valid initialization. ")
    
    self.clipping_scale = clipping_scale
    self.line_crossing = line_crossing 
  
  def initialize(self): 
      
    if self.initialization == "inplace": 
      pass 
    elif self.initialization == "rand": 
      with torch.no_grad(): 
        for _, param in self.model.named_parameters(): 
          param.copy_(0.01 * torch.randn_like(param))
    elif self.initialization == "zeros": 
      with torch.no_grad(): 
        for _, param in self.model.named_parameters(): 
          param.copy_(0. * param)

  def step(self): 
      
    d = 0
    for param in self.model.parameters(): 
      d += math.prod(param.size())
    
    u = [W.data.detach().clone() for W in self.model.parameters()]
    
    def objective_function(model): 
      p_norm = torch.norm(torch.stack([torch.norm(param, self.p) for param in model.parameters()]), self.p)
      q_norm = torch.norm(torch.stack([torch.norm(param, self.q) for param in model.parameters()]), self.q)
      l2_diff = torch.norm(torch.stack([torch.norm([W for W in model.parameters()][i] - u[i]) for i in range(len(u))]))
      output = self.lmbda * (d ** ((1/self.q) - (1/self.p)) * (p_norm/q_norm)) + 1/(2*self.tau) * l2_diff ** 2
      return  output.to(self.device)
    
    self.initialize()
    
    def closure():
      self.optimizer.zero_grad()
      loss = objective_function(self.model)
      loss.backward()
      return loss
    
    for i in range(500): 
      positives1 = []
      negatives1 = []
        
      if self.line_crossing is True: 
        positives1 = []
        negatives1 = []
        for W in self.model.parameters(): 
          mask1 = torch.where(W.data >= 0, 1.0, 0.0)
          mask2 = torch.where(W.data <= 0, 1.0, 0.0)
          positives1.append(mask1) 
          negatives1.append(mask2)
          
      self.optimizer.step(closure)
      
      if self.line_crossing is True: 
        positives2 = [] 
        negatives2 = [] 
        for W in self.model.parameters(): 
          mask1 = torch.where(W.data >= 0, 1.0, 0.0)
          mask2 = torch.where(W.data <= 0, 1.0, 0.0)
          positives2.append(mask1) 
          negatives2.append(mask2)

        clip_mask = [] 
        for i in range(len(positives1)): 
          clip = positives1[i] * positives2[i] + negatives1[i] * negatives2[i]
          clip_mask.append(clip)
      
        with torch.no_grad(): 
          i = 0
          for _, param in self.model.named_parameters(): 
            param.copy_(clip_mask[i] * param)
            i += 1 
        
    with torch.no_grad(): 
      for _, param in self.model.named_parameters(): 
        clipped_param = param.clone()
        clipped_param[torch.logical_and(-self.clipping_scale * self.lmbda*self.tau < clipped_param, clipped_param < self.clipping_scale * self.lmbda*self.tau)] = 0.0
        param.copy_(clipped_param)

    self.penalty = self.lmbda * PQI(self.model, self.device, self.p, self.q)

class PQI_ADMM(Regularizer): 
    
  def __init__(self, model:Module): 
    pass 
  
def p_norm(model:Module, device:str, p:float, normalize:bool = True) -> Tensor: 
  p_norm = Tensor([0.]).to(device)
  N = 0
  for W in model.parameters(): 
    p_norm += W.norm(p) ** p

    if normalize: 
      N += math.prod(W.size())

  p_norm = p_norm ** (1 / p)

  if normalize and N > 0:
    p_norm /= N

  return p_norm.to(device)
 
def PQI(model:Module, device:str, p:float, q:float): 
    
  d = sum(math.prod(param.size()) for param in model.parameters())
  pq = d ** ((1/q) - (1/p)) * (p_norm(model, device, p, normalize=False)/p_norm(model, device, q, normalize=False))
  
  return 1 - pq

