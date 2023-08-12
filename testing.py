import torch 
import torch.nn as nn 
from pprint import pprint
from src.device import select_device 
from src.testtrain import * 
from src.data import fetch_dataset, dataloader
from src.models.mlp import MLP 
from src.models.linear import Linear
from src.regularizers import * 
from src.metrics import * 
import matplotlib.pyplot as plt 
import numpy as np

def soft_thresholding(x, lmbda):
    return np.sign(x) * np.maximum(0, np.abs(x) - lmbda)

def admm_lasso(Y, X, lmbda, rho=1.0, max_iter=1000, tol=1e-4):
    n, p = X.shape
    beta = np.zeros(p)
    u = np.zeros(p)
    lambda_dual = np.zeros(p)
    I = np.eye(p)
    
    for _ in range(max_iter):
        # Update beta
        beta = np.linalg.inv(X.T @ X + rho * I) @ (X.T @ Y + rho * (u - lambda_dual))
        
        # Update u
        u = soft_thresholding(beta + lambda_dual, lmbda/rho)
        
        # Update lambda
        lambda_dual += rho * (beta - u)
        
        # Convergence check
        prim_resid = np.linalg.norm(beta - u)
        dual_resid = rho * np.linalg.norm(u - soft_thresholding(beta + 2*lambda_dual, lmbda/rho))
        if prim_resid < tol and dual_resid < tol:
            break
            
    return beta

# Test the function
Y = np.random.randn(100)
X = np.random.randn(100, 10)
lmbda = 0.1
beta = admm_lasso(Y, X, lmbda, max_iter=10000)
print(beta)