import torch 
import math
import numpy as np
from abc import ABC, abstractmethod

class Regularizer(ABC): 
    @abstractmethod 
    def step(self): 
        pass 
    
class L1_SoftThreshold(Regularizer): 
    
    def __init__(self, model:torch.nn.Module, device:str, lmbda:float, tau:float): 
        """
        Arguments 
        model - our model 
        device = "cpu" or "cuda" 
        lmbda - regularization parameter 
        tau - step size of our gradient step 
        """
        
        self.model = model 
        self.device = device 
        self.lmbda = lmbda 
        self.tau = tau 
        self.penalty = lmbda * p_norm(model, device, 1)
    
    def soft_threshold(self, param:torch.tensor, lmbda:torch.tensor): 
        x = torch.sign(param) * torch.max(torch.abs(param) - lmbda, torch.zeros_like(param))
        return x
    
    def step(self): 
        with torch.no_grad(): 
            for name, param in self.model.named_parameters(): 
                param.copy_(self.soft_threshold(param, self.tau * torch.tensor(self.lmbda)))
                
        self.penalty = self.lmbda * p_norm(self.model, self.device, 1)

class L1_Proximal(Regularizer): 
    
    def __init__(self, model:torch.nn.Module, device:str, lmbda:float, tau:float, optimizer:str, initialization:str = "inplace", clipping_scale:float = 1.0, line_crossing:bool = False): 
        """
        Arguments 
        model - our model 
        optimizer - which optimizer to use for optimizing the argmin of the proximal step (LBFGS or batch GD)
        initialization - where we initialize the model parameters before optimization ("inplace", "rand", "zeros")
        clipping - the threshold to clip all param values to 0 
        line_crossing - whether we should clip all values that change signs during each proximal optimizer step
        """
        
        self.lmbda = lmbda 
        self.tau = tau 
        self.model = model 
        self.device = device
        self.penalty = lmbda * p_norm(model, device, 1)
        
        if optimizer in ["SGD", "GD"]: 
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lmbda*tau * 0.1, momentum=0.0)
        elif optimizer in ["LBFGS"]: 
            self.optimizer = torch.optim.LBFGS(model.parameters(), lr=lmbda*tau * 0.1, tolerance_change=1e-6)
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
                for name, param in self.model.named_parameters(): 
                    param.copy_(0.01 * torch.randn_like(param))
        elif self.initialization == "zeros": 
            with torch.no_grad(): 
                for name, param in self.model.named_parameters(): 
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
                    for name, param in self.model.named_parameters(): 
                        param.copy_(clip_mask[i] * param)
                        i += 1 
            
        with torch.no_grad(): 
            for name, param in self.model.named_parameters(): 
                clipped_param = param.clone()
                clipped_param[torch.logical_and(clipped_param >-self.clipping_scale * self.lmbda*self.tau, clipped_param < self.clipping_scale * self.lmbda*self.tau)] = 0.0
                param.copy_(clipped_param)

        self.penalty = self.lmbda * p_norm(self.model, self.device, 1)
        
class L1_ADMM(Regularizer): 
    
    def __init__(self, model:torch.nn.Module): 
        self.model = model 
        
class L1_SGD_Naive(Regularizer): 
    
    '''Does an extra gradient step on the regularization term and clips all values that cross 0'''
    
    def __init__(self, model:torch.nn.Module, optimizer:torch.optim.Optimizer, device:str, lmbda:float): 
        self.model = model 
        self.optimizer = optimizer
        self.device = device 
        self.lmbda = lmbda 
        
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
            for name, param in self.model.named_parameters(): 
                param.copy_(clip_mask[i] * param)
                i += 1 

class L2_Regularizer(Regularizer): 
    
    def __init__(self, model:torch.nn.Module, optimizer:torch.optim.Optimizer, device:str, lmbda:float): 
        
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
    def __init__(self, model:torch.nn.Module, device:str, p:float, q:float, lmbda:float, tau:float, optimizer:str, initialization:str = "inplace", clipping_scale:float = 1.0, line_crossing:bool = False): 
        """
        Arguments 
        model - our model 
        optimizer - which optimizer to use for optimizing the argmin of the proximal step (LBFGS, or batch GD)
        initialization - where we initialize the model parameters before optimization ("inplace", "rand", "zeros")
        clipping - the threshold to clip all param values to 0 
        line_crossing - whether we should clip all values that change signs during each proximal optimizer step
        """
        
        self.p = p 
        self.q = q
        if p > q: 
            raise Exception("p should be less than q. ")
        
        self.lmbda = lmbda 
        self.tau = tau 
        self.model = model 
        self.device = device
        self.penalty = lmbda * PQI(self.model, self.device, self.p, self.q)
        
        if optimizer in ["SGD", "GD"]: 
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lmbda*tau * 0.1, momentum=0.0)
        elif optimizer in ["LBFGS"]: 
            self.optimizer = torch.optim.LBFGS(model.parameters(), lr=lmbda*tau * 0.1, tolerance_change=1e-6)
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
                for name, param in self.model.named_parameters(): 
                    param.copy_(0.01 * torch.randn_like(param))
        elif self.initialization == "zeros": 
            with torch.no_grad(): 
                for name, param in self.model.named_parameters(): 
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
                    for name, param in self.model.named_parameters(): 
                        param.copy_(clip_mask[i] * param)
                        i += 1 
            
        with torch.no_grad(): 
            for name, param in self.model.named_parameters(): 
                clipped_param = param.clone()
                clipped_param[torch.logical_and(-self.clipping_scale * self.lmbda*self.tau < clipped_param, clipped_param < self.clipping_scale * self.lmbda*self.tau)] = 0.0
                param.copy_(clipped_param)

        self.penalty = self.lmbda * PQI(self.model, self.device, self.p, self.q)
        

class PQI_ADMM(Regularizer): 
    
    pass 

# def admm_lasso(Y, X, lmbda, rho=1.0, max_iter=1000, tol=1e-4):
#     n, p = X.shape
#     beta = np.zeros(p)
#     u = np.zeros(p)
#     lambda_dual = np.zeros(p)
#     I = np.eye(p)
    
#     for _ in range(max_iter):
#         # Update beta
#         beta = np.linalg.inv(X.T @ X + rho * I) @ (X.T @ Y + rho * (u - lambda_dual))
        
#         # Update u
#         u = soft_thresholding(beta + lambda_dual, lmbda/rho)
        
#         # Update lambda
#         lambda_dual += rho * (beta - u)
        
#         # Convergence check
#         prim_resid = np.linalg.norm(beta - u)
#         dual_resid = rho * np.linalg.norm(u - soft_thresholding(beta + 2*lambda_dual, lmbda/rho))
#         if prim_resid < tol and dual_resid < tol:
#             break
            
#     return beta

# # Test the function
# Y = np.random.randn(100)
# X = np.random.randn(100, 10)
# lmbda = 0.1
# beta = admm_lasso(Y, X, lmbda, max_iter=10000)
# print(beta)

def p_norm(model:torch.nn.Module, device:str, p:float, normalize:bool=True): 
    p_norm = None
    N = 0
    for W in model.parameters(): 
        if p_norm == None: 
            p_norm = W.norm(p) ** p
        else: 
            p_norm += W.norm(p) ** p
            
        if normalize == True: 
            N += math.prod(W.size())
            
    p_norm = p_norm ** (1/p)
            
    if normalize == True: 
        return (1/N) * p_norm.to(device)
    else: 
        return p_norm.to(device)
    

    
def PQI(model:torch.nn.Module, device:str, p:float, q:float): 
    
    d = sum(math.prod(param.size()) for param in model.parameters())
    pq = d ** ((1/q) - (1/p)) * (p_norm(model, device, p, normalize=False)/p_norm(model, device, q, normalize=False))
    
    return 1 - pq
            


    