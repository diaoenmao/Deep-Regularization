from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from src.traintest.regularizers import * 
import time 


def train(dataloader:DataLoader, model:Module, loss_fn, optimizer:Optimizer, device:str, regularizer:Regularizer = None, t:int = 0):
    print(f"Train Epoch {t}\n-------------------------------")
    
    start = time.time() 
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0.0
    
    model.train()
    for batch_num, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        
        if t != 0: 
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if regularizer is not None: 
                regularizer.step()
            
        if batch_num % int(len(dataloader)/4) == 0: 
            loss, current = loss.item(), (batch_num + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            
    print(f"Epoch Time : {round(time.time() - start, 2)}")
            
    return {"loss" : train_loss / num_batches, 
            "accuracy" : 0.0}
         
        