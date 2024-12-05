import torch 
from src.config import cfg

def make_optimizer(optimizer_name:str, model:torch.nn.Module): 
    
    if cfg["hyperparameters"]["optimizer_name"] == "SGD": 
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr = cfg["hyperparameters"]["lr"], 
            momentum = cfg["hyperparameters"]["momentum"], 
            weight_decay= cfg["hyperparameters"]["weight_decay"]
        )
    else: 
        pass 
        
        
    return optimizer